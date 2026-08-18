#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "core/runtime/model_loader.h"
#include "utils/model_io/convert.h"
#include "utils/model_io/gguf_io.h"
#include "utils/name_conversion.h"
#include "utils/util.h"

#include "ggml-cpu.h"
#include "json.hpp"

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

enum class ConvertInputKind {
    MODEL,
    DIFFUSION_MODEL,
    VAE,
    AUDIO_VAE,
    CLIP_L,
    CLIP_G,
    T5XXL,
    LLM,
    LLM_VISION,
};

static const char* convert_input_kind_name(ConvertInputKind kind) {
    switch (kind) {
        case ConvertInputKind::MODEL: return "model";
        case ConvertInputKind::DIFFUSION_MODEL: return "diffusion-model";
        case ConvertInputKind::VAE: return "vae";
        case ConvertInputKind::AUDIO_VAE: return "audio-vae";
        case ConvertInputKind::CLIP_L: return "clip-l";
        case ConvertInputKind::CLIP_G: return "clip-g";
        case ConvertInputKind::T5XXL: return "t5xxl";
        case ConvertInputKind::LLM: return "llm";
        case ConvertInputKind::LLM_VISION: return "llm-vision";
    }
    return "model";
}

static const char* convert_input_prefix(ConvertInputKind kind) {
    switch (kind) {
        case ConvertInputKind::DIFFUSION_MODEL: return "model.diffusion_model.";
        case ConvertInputKind::VAE: return "vae.";
        case ConvertInputKind::AUDIO_VAE: return "audio_vae.";
        case ConvertInputKind::CLIP_L: return "text_encoders.clip_l.transformer.";
        case ConvertInputKind::CLIP_G: return "text_encoders.clip_g.transformer.";
        case ConvertInputKind::T5XXL: return "text_encoders.t5xxl.transformer.";
        case ConvertInputKind::LLM: return "text_encoders.llm.";
        // Load under the parent LLM prefix first. After name conversion we keep
        // only the canonical visual subtree, allowing --llm-vision to
        // extract a vision tower from a combined Qwen-VL checkpoint as well as
        // convert a vision-only file.
        case ConvertInputKind::LLM_VISION: return "text_encoders.llm.";
        case ConvertInputKind::MODEL: return "";
    }
    return "";
}

static bool non_empty_path(const char* path) {
    return path != nullptr && path[0] != '\0';
}

static std::string resolve_component_input(const char* input_path) {
    std::string resolved = input_path != nullptr ? input_path : "";
    if (!std::filesystem::is_directory(resolved)) {
        return resolved;
    }

    const char* candidates[] = {
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.fp16.json",
        "model.safetensors",
        "model.fp16.safetensors",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
    };
    for (const char* candidate : candidates) {
        const std::filesystem::path path = std::filesystem::path(resolved) / candidate;
        if (std::filesystem::is_regular_file(path)) {
            LOG_INFO("resolved component directory to '%s'", path.string().c_str());
            return path.string();
        }
    }

    std::vector<std::string> supported_files;
    for (const auto& entry : std::filesystem::directory_iterator(resolved)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string path = entry.path().string();
        if (ends_with(path, ".safetensors") ||
            ends_with(path, ".safetensors.index.json") ||
            ends_with(path, ".gguf")) {
            supported_files.push_back(path);
        }
    }
    if (supported_files.size() == 1) {
        LOG_INFO("resolved component directory to '%s'", supported_files.front().c_str());
        return supported_files.front();
    }
    return resolved;
}

static bool safetensors_has_packed_u8_weights(const std::string& path) {
    if (ends_with(path, ".safetensors.index.json")) {
        std::ifstream index_file(path);
        if (!index_file.is_open()) {
            return false;
        }
        try {
            nlohmann::json index;
            index_file >> index;
            if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
                return false;
            }
            std::set<std::string> shards;
            for (const auto& item : index["weight_map"].items()) {
                if (item.value().is_string()) {
                    shards.insert(item.value().get<std::string>());
                }
            }
            const std::filesystem::path base = std::filesystem::path(path).parent_path();
            for (const std::string& shard : shards) {
                if (safetensors_has_packed_u8_weights((base / shard).string())) {
                    return true;
                }
            }
        } catch (const std::exception&) {
            return false;
        }
        return false;
    }
    if (!ends_with(path, ".safetensors")) {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file || header_size == 0 || header_size > 256ULL * 1024ULL * 1024ULL) {
        return false;
    }
    std::string header(header_size, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) {
        return false;
    }
    try {
        const nlohmann::json metadata = nlohmann::json::parse(header);
        for (const auto& item : metadata.items()) {
            if (!item.value().is_object() || !item.value().contains("dtype")) {
                continue;
            }
            if (item.value()["dtype"] == "U8" &&
                (ends_with(item.key(), ".weight") || contains(item.key(), ".comfy_quant"))) {
                return true;
            }
        }
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

bool convert(const char* model_path,
             const char* diffusion_model_path,
             const char* vae_path,
             const char* audio_vae_path,
             const char* clip_l_path,
             const char* clip_g_path,
             const char* t5xxl_path,
             const char* llm_path,
             const char* llm_vision_path,
             const char* output_path,
             ed_dtype_t output_type,
             const char* tensor_type_rules,
             bool convert_name,
             const char* imatrix_path) {
    ModelLoader model_loader;
    struct ComponentInput {
        ConvertInputKind kind;
        const char* path;
    };
    std::vector<ComponentInput> components;
    const ComponentInput requested_components[] = {
        {ConvertInputKind::DIFFUSION_MODEL, diffusion_model_path},
        {ConvertInputKind::VAE, vae_path},
        {ConvertInputKind::AUDIO_VAE, audio_vae_path},
        {ConvertInputKind::CLIP_L, clip_l_path},
        {ConvertInputKind::CLIP_G, clip_g_path},
        {ConvertInputKind::T5XXL, t5xxl_path},
        {ConvertInputKind::LLM, llm_path},
        {ConvertInputKind::LLM_VISION, llm_vision_path},
    };
    for (const ComponentInput& input : requested_components) {
        if (non_empty_path(input.path)) {
            components.push_back(input);
        }
    }

    if (!non_empty_path(output_path)) {
        LOG_ERROR("output path is required");
        return false;
    }

    const bool complete_directory_mode = non_empty_path(model_path);
    if (complete_directory_mode && !components.empty()) {
        LOG_ERROR("--model is mutually exclusive with named component inputs");
        return false;
    }
    if (!complete_directory_mode && components.empty()) {
        LOG_ERROR("one input is required: a complete --model directory or a named component");
        return false;
    }
    if (complete_directory_mode && !std::filesystem::is_directory(model_path)) {
        LOG_ERROR("--model must point to a complete model directory, not a component file: '%s'", model_path);
        return false;
    }
    if (ends_with(output_path, ".safetensors")) {
        LOG_ERROR("ed-convert output must be GGUF");
        return false;
    }
    if (!complete_directory_mode && !convert_name) {
        LOG_ERROR("--raw-names is not valid for component conversion; canonical names are required");
        return false;
    }

    if (complete_directory_mode) {
        if (!model_loader.init_from_diffusers_directory(model_path, "")) {
            LOG_ERROR("init model loader from complete directory failed: '%s'", model_path);
            return false;
        }
    } else {
        // Load all ordinary components first. Vision is handled separately below
        // because a combined Qwen-VL checkpoint must be reduced to its visual subtree.
        for (const ComponentInput& input : components) {
            if (input.kind == ConvertInputKind::LLM_VISION) {
                continue;
            }
            const std::string resolved = resolve_component_input(input.path);
            if (std::filesystem::is_directory(resolved)) {
                LOG_ERROR("%s component directory contains no unique supported weights file: '%s'",
                          convert_input_kind_name(input.kind), input.path);
                return false;
            }
            if (safetensors_has_packed_u8_weights(resolved)) {
                LOG_ERROR("%s input uses packed U8/AWQ/Comfy weights; ed-convert cannot dequantize "
                          "this format. Use BF16/F16/F32/FP8 safetensors or a supported GGUF source",
                          convert_input_kind_name(input.kind));
                return false;
            }
            if (!model_loader.init_from_file(resolved, convert_input_prefix(input.kind))) {
                LOG_ERROR("init %s component failed: '%s'",
                          convert_input_kind_name(input.kind), resolved.c_str());
                return false;
            }
        }
    }

    if (convert_name && !model_loader.get_tensor_storage_map().empty()) {
        model_loader.convert_tensors_name();
    }

    if (!complete_directory_mode && non_empty_path(llm_vision_path)) {
        const std::string resolved = resolve_component_input(llm_vision_path);
        if (std::filesystem::is_directory(resolved)) {
            LOG_ERROR("llm-vision component directory contains no unique supported weights file: '%s'",
                      llm_vision_path);
            return false;
        }
        if (safetensors_has_packed_u8_weights(resolved)) {
            LOG_ERROR("llm-vision input uses packed U8/AWQ/Comfy weights; ed-convert cannot dequantize "
                      "this format. Use BF16/F16/F32/FP8 safetensors or a supported GGUF source");
            return false;
        }

        auto& storage_map = model_loader.get_tensor_storage_map();
        std::set<std::string> existing_names;
        for (const auto& item : storage_map) {
            existing_names.insert(item.first);
        }

        std::string vision_prefix = convert_input_prefix(ConvertInputKind::LLM_VISION);
        if (is_gguf_file(resolved)) {
            std::vector<TensorStorage> source_tensors;
            std::string source_error;
            std::string source_version;
            std::string source_component;
            if (read_gguf_file(resolved, source_tensors, &source_error,
                               &source_version, &source_component) &&
                source_component == "llm-vision") {
                vision_prefix = "text_encoders.llm.visual.";
            }
        }
        if (!model_loader.init_from_file(resolved, vision_prefix)) {
            LOG_ERROR("init llm-vision component failed: '%s'", resolved.c_str());
            return false;
        }

        // A bare Qwen-VL visual checkpoint may contain only `blocks.*` and thus
        // lacks enough pipeline context for generic family inference. The
        // --llm-vision flag itself identifies the Qwen-VL naming transform; use
        // that built-in mapping automatically instead of asking for a model hint.
        if (model_loader.version() == VERSION_COUNT &&
            model_loader.get_ld_version() == VERSION_COUNT) {
            model_loader.set_version_hint(VERSION_MINIMAX_H3);
        }
        model_loader.convert_tensors_name();

        constexpr const char* kLlmPrefix = "text_encoders.llm.";
        constexpr const char* kVisionPrefix = "text_encoders.llm.visual.";
        bool has_canonical_vision = false;
        for (const auto& item : storage_map) {
            if (existing_names.find(item.first) == existing_names.end() &&
                starts_with(item.first, kVisionPrefix)) {
                has_canonical_vision = true;
                break;
            }
        }

        String2TensorStorage vision_map;
        for (const auto& item : storage_map) {
            TensorStorage storage = item.second;
            if (existing_names.find(item.first) != existing_names.end()) {
                vision_map[item.first] = std::move(storage);
                continue;
            }
            if (has_canonical_vision) {
                if (!starts_with(storage.name, kVisionPrefix)) {
                    continue;
                }
            } else {
                // A true vision-only checkpoint often stores bare `blocks.*`,
                // `patch_embed.*`, etc. The input prefix made these
                // `text_encoders.llm.*`; rebase the complete file below visual.
                if (!starts_with(storage.name, kLlmPrefix)) {
                    continue;
                }
                storage.name = std::string(kVisionPrefix) + storage.name.substr(std::strlen(kLlmPrefix));
            }
            vision_map[storage.name] = std::move(storage);
        }
        if (vision_map.size() == existing_names.size()) {
            LOG_ERROR("llm-vision conversion found no visual encoder tensors");
            return false;
        }
        LOG_INFO("llm-vision conversion selected %zu new visual tensors from %zu source tensors",
                 vision_map.size() - existing_names.size(),
                 storage_map.size() - existing_names.size());
        storage_map.swap(vision_map);
    }

    if (convert_name) {
        // Idempotent for already canonical names; catches any component added by
        // the special vision path above.
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
        // Persist the true model version into the GGUF metadata, so the loader
        // no longer has to guess FLUX-Kontext / Qwen-Image-Edit from the file
        // name. A single named input also records its semantic component kind.
        SDVersion version = model_loader.version();
        if (version == VERSION_COUNT) {
            version = model_loader.get_ld_version();
        }
        const std::string model_ver = version != VERSION_COUNT ? ed_version_name(version) : "";
        std::string component_kind = "model";
        if (!complete_directory_mode && components.size() == 1) {
            component_kind = convert_input_kind_name(components.front().kind);
        }
        success = write_gguf_file(output_path, tensors, model_ver, component_kind, &error);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    ggml_free(ggml_ctx);
    return success;
}
