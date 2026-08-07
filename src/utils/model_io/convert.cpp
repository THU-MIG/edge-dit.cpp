#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

#include "core/runtime/model_loader.h"
#include "utils/model_io/convert.h"
#include "utils/model_io/gguf_io.h"
#include "utils/model_io/safetensors_io.h"
#include "utils/name_conversion.h"
#include "utils/util.h"

#include "ggml-cpu.h"

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    // Only quantize tensors that the runtime model would also quantize.
    //
    // The engine's model graph always allocates 1-D tensors -- biases,
    // LayerNorm/RMSNorm scales, and the fused-QKV split biases the SD3/MMDiT
    // name mapping produces (e.g. "...qkv.bias.1", "...qkv.bias.2") -- at float
    // precision (see Linear::init_params in ggml_extend.hpp, which forces the
    // bias to F32), overriding the on-disk storage type when it binds weights.
    // A converter has no such graph: whatever type we pick here is baked into
    // the output file permanently. ModelLoader::tensor_should_be_converted only
    // skips the *exact* ".bias"/".scale" suffixes, so it still quantizes split
    // biases ("...bias.1") and 1-D norm ".weight" vectors. Quantizing those
    // sensitive 1-D vectors to q8_0/q4_k is very lossy and, worse, diverges from
    // the runtime (which keeps them float), so a pre-quantized GGUF would score
    // measurably worse than on-the-fly quantization. Guard on dimensionality --
    // matching llama.cpp / stable-diffusion.cpp, which only quantize 2-D+
    // tensors -- so convert reproduces the runtime's effective precision.
    const bool is_multi_dim = tensor_storage.n_dims >= 2;
    if (is_multi_dim && model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool load_tensors_for_export(ModelLoader& model_loader,
                                    ggml_context* ggml_ctx,
                                    ggml_type type,
                                    const TensorTypeRules& tensor_type_rules,
                                    std::vector<TensorWriteInfo>& tensors) {
    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        ggml_type tensor_type   = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);

        std::lock_guard<std::mutex> lock(tensor_mutex);
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == nullptr) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        if (!tensor->data) {
            GGML_ASSERT(ggml_nelements(tensor) == 0);
            // Avoid crashing writers by setting a dummy pointer for zero-sized tensors.
            LOG_DEBUG("setting dummy pointer for zero-sized tensor %s", name.c_str());
            tensor->data = ggml_get_mem_buffer(ggml_ctx);
        }

        TensorWriteInfo write_info;
        write_info.tensor = tensor;
        write_info.n_dims = tensor_storage.n_dims;
        for (int i = 0; i < tensor_storage.n_dims; ++i) {
            write_info.ne[i] = tensor_storage.ne[i];
        }

        *dst_tensor = tensor;
        tensors.push_back(std::move(write_info));

        return true;
    };

    bool success = model_loader.load_tensors(on_new_tensor_cb);
    LOG_INFO("load tensors done");
    return success;
}

// Loads an activation-calibrated imatrix GGUF and remaps its keys into edge's canonical tensor-name
// space so lookups during load_tensors line up.
//
// The imatrix file stores importance vectors under ORIGINAL diffusers weight names
// (e.g. "transformer_blocks.0.attn.to_q.weight"), because it is produced by a pure
// PyTorch/diffusers calibration pass. Edge, however, quantizes tensors under their
// canonical names (e.g. "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight"),
// which convert_tensor_name() derives only AFTER the diffusers loader prepends the
// component prefix ("transformer." for the SD3 transformer). So we reproduce that
// exact two-step transform here: prepend "transformer." (matching
// init_from_diffusers_directory's sd_prefix), then run convert_tensor_name() for the
// model's version. For SD3 this also folds to_q/to_k/to_v into the shared
// "...qkv.weight"/".weight.1"/".weight.2" storage names -- all three share the same
// in_features, so the vector length still equals ne[0] and the quantizer accepts it.
// Any key that fails to remap keeps its original name too, so already-canonical
// imatrix files also work. Entries whose length mismatches a tensor's ne[0] are
// silently ignored at quantization time (all-ones fallback).
static std::map<std::string, std::vector<float>> load_and_remap_imatrix(const char* imatrix_path,
                                                                        SDVersion version) {
    std::map<std::string, std::vector<float>> canonical;
    if (imatrix_path == nullptr || std::strlen(imatrix_path) == 0) {
        return canonical;
    }

    std::map<std::string, std::vector<float>> raw;
    std::string error;
    if (!read_imatrix_gguf(imatrix_path, raw, &error)) {
        LOG_WARN("failed to read imatrix '%s': %s -- proceeding with plain quantization",
                 imatrix_path, error.c_str());
        return canonical;
    }

    size_t remapped = 0;
    for (const auto& kv : raw) {
        const std::string& orig = kv.first;
        // Reproduce the loader's diffusers -> canonical transform. Prepending
        // "transformer." is what makes convert_tensor_name trigger the DiT prefix
        // maps (they only match "transformer."/"unet."/... prefixes).
        std::string canon = convert_tensor_name("transformer." + orig, version);
        canonical[canon] = kv.second;
        if (canon != orig) {
            ++remapped;
        }
        // Also keep the original name as a fallback (harmless: separate key), so a
        // pre-canonicalized imatrix or a non-DiT tensor still resolves.
        canonical.emplace(orig, kv.second);
    }

    LOG_INFO("loaded imatrix '%s': %zu raw vectors, %zu remapped to canonical names (version=%s)",
             imatrix_path, raw.size(), remapped, ed_version_name(version));
    return canonical;
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* clip_l_path,
             const char* clip_g_path,
             const char* t5xxl_path,
             const char* output_path,
             ed_dtype_t output_type,
             const char* tensor_type_rules,
             bool convert_name,
             const char* imatrix_path) {
    ModelLoader model_loader;

    if (!model_loader.init_from_file(input_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (vae_path != nullptr && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    // Optional external text encoders, merged under the same canonical prefixes
    // the runtime uses (model_loader.cpp component loading). This lets a bare
    // transformer be combined with its encoders/VAE into one standalone GGUF.
    if (clip_l_path != nullptr && strlen(clip_l_path) > 0) {
        if (!model_loader.init_from_file(clip_l_path, "text_encoders.clip_l.transformer.")) {
            LOG_ERROR("init model loader from file failed: '%s'", clip_l_path);
            return false;
        }
    }
    if (clip_g_path != nullptr && strlen(clip_g_path) > 0) {
        if (!model_loader.init_from_file(clip_g_path, "text_encoders.clip_g.transformer.")) {
            LOG_ERROR("init model loader from file failed: '%s'", clip_g_path);
            return false;
        }
    }
    // Skipping t5xxl_path is the offline equivalent of --no-t5.
    if (t5xxl_path != nullptr && strlen(t5xxl_path) > 0) {
        if (!model_loader.init_from_file(t5xxl_path, "text_encoders.t5xxl.transformer.")) {
            LOG_ERROR("init model loader from file failed: '%s'", t5xxl_path);
            return false;
        }
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    // Install the imatrix (if any) AFTER names are canonicalized: the map is
    // keyed by canonical names, matching tensor_storage.name during load_tensors.
    if (imatrix_path != nullptr && std::strlen(imatrix_path) > 0) {
        // Resolve the version the same way convert_tensors_name() does (it uses a
        // local and does not persist it), so the DiT name mapping is applied.
        SDVersion version = model_loader.version();
        if (version == VERSION_COUNT) {
            version = model_loader.get_ld_version();
        }
        auto imatrix_map = load_and_remap_imatrix(imatrix_path, version);
        if (!imatrix_map.empty()) {
            model_loader.set_imatrix_map(std::move(imatrix_map));
        }
    }

    ggml_type type             = (ggml_type)output_type;
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules != nullptr ? tensor_type_rules : "");

    auto backend    = ggml_backend_cpu_init();
    // Padding note: get_params_mem_size() sizes quantizable tensors at `type`,
    // but get_export_tensor_type() above keeps 1-D tensors (norms/biases) at
    // their original float type, which is larger than a quantized estimate. Add
    // a generous per-tensor slack (>> any single 1-D vector's float-vs-quant byte
    // delta) on top of the flat pad so the ggml context never under-allocates.
    // (Under-allocation would be a clean failure -- ggml_new_tensor returns null
    // -- not corruption, but the slack keeps convert working for large models.)
    size_t mem_size = 1 * 1024 * 1024;  // flat padding
    mem_size += model_loader.get_tensor_storage_map().size() * (ggml_tensor_overhead() + 256 * 1024);
    mem_size += model_loader.get_params_mem_size(backend, type);
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});

    if (ggml_ctx == nullptr) {
        LOG_ERROR("ggml_init failed for converter");
        ggml_backend_free(backend);
        return false;
    }

    std::vector<TensorWriteInfo> tensors;
    bool success = load_tensors_for_export(model_loader, ggml_ctx, type, type_rules, tensors);
    ggml_backend_free(backend);

    std::string error;
    if (success) {
        if (output_is_safetensors) {
            success = write_safetensors_file(output_path, tensors, &error);
        } else {
            // Persist the true model version into the GGUF metadata, so the
            // loader no longer has to guess FLUX-Kontext / Qwen-Image-Edit from
            // the file name. version() reads it from a diffusers config.json;
            // when the source is a bare transformer (no config), fall back to
            // get_ld_version() -- names are already canonical here, so it can
            // recover the family from signature tensors (e.g. SD3 "joint_blocks.").
            SDVersion version = model_loader.version();
            if (version == VERSION_COUNT) {
                version = model_loader.get_ld_version();
            }
            const std::string model_ver = version != VERSION_COUNT ? ed_version_name(version) : "";
            success = write_gguf_file(output_path, tensors, model_ver, &error);
        }
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    ggml_free(ggml_ctx);
    return success;
}
