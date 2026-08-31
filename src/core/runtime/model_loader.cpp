#include "core/runtime/model_loader.h"
#include <sstream>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>

#include "ggml.h"
#include "json.hpp"
#include "utils/model_io/gguf_io.h"
#include "utils/model_io/safetensors_io.h"
#include "utils/name_conversion.h"
#include "utils/util.h"

namespace fs = std::filesystem;

static const char* unused_tensors[] = {
    "betas",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "cond_stage_model.1.model.text_model.embeddings.position_ids",
    "cond_stage_model.transformer.vision_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
    "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
    "conditioner.embedders.0.model.logit_scale",
    "conditioner.embedders.1.model.logit_scale",
    "model.diffusion_model.time_embedding.cond_proj.weight",
    "unet.time_embedding.cond_proj.weight",
    "model_ema.decay",
    "model_ema.num_updates",
    "model_ema.diffusion_model",
    "embedding_manager",
    "denoiser.sigmas",
    "text_encoders.t5xxl.transformer.encoder.embed_tokens.weight",
    "ztsnr",
    "edm_vpred.sigma_min",
    "text_encoders.llm.output.weight",
    "text_encoders.llm.lm_head.",
    "first_stage_model.bn.",
};

static bool is_unused_tensor(const std::string& name) {
    for (const char* prefix : unused_tensors) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

static bool split_tensor_chunk_base(const std::string& name, std::string* base, int* chunk_index) {
    if (ends_with(name, ".weight") || ends_with(name, ".bias")) {
        if (base != nullptr) {
            *base = name;
        }
        if (chunk_index != nullptr) {
            *chunk_index = 0;
        }
        return true;
    }

    const size_t dot = name.rfind('.');
    if (dot == std::string::npos || dot + 1 >= name.size()) {
        return false;
    }

    int value = 0;
    for (size_t i = dot + 1; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
        value = value * 10 + (name[i] - '0');
    }

    const std::string candidate_base = name.substr(0, dot);
    if (!ends_with(candidate_base, ".weight") && !ends_with(candidate_base, ".bias")) {
        return false;
    }

    if (base != nullptr) {
        *base = candidate_base;
    }
    if (chunk_index != nullptr) {
        *chunk_index = value;
    }
    return true;
}

static bool tensor_shape_matches_ggml(const ggml_tensor* tensor, const TensorStorage& storage) {
    if (tensor == nullptr) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (tensor->ne[i] != storage.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool find_split_concat_dim(const ggml_tensor* fused,
                                  const TensorStorage& chunk,
                                  int* concat_dim) {
    if (fused == nullptr) {
        return false;
    }

    for (int dim = 0; dim < 4; ++dim) {
        if (fused->ne[dim] <= chunk.ne[dim]) {
            continue;
        }

        bool other_dims_match = true;
        for (int i = 0; i < 4; ++i) {
            if (i == dim) {
                continue;
            }
            if (fused->ne[i] != chunk.ne[i]) {
                other_dims_match = false;
                break;
            }
        }
        if (!other_dims_match) {
            continue;
        }

        if (concat_dim != nullptr) {
            *concat_dim = dim;
        }
        return true;
    }

    return false;
}

static int64_t split_chunk_offset_elems(const String2TensorStorage& storage_map,
                                        const std::string& base,
                                        int chunk_index,
                                        int concat_dim) {
    int64_t offset = 0;
    for (int i = 0; i < chunk_index; ++i) {
        const std::string chunk_name = i == 0 ? base : base + "." + std::to_string(i);
        auto it = storage_map.find(chunk_name);
        if (it == storage_map.end()) {
            return -1;
        }
        offset += it->second.ne[concat_dim];
    }
    return offset;
}

const char* ed_version_name(SDVersion version) {
    switch (version) {
        case VERSION_SD1: return "sd1";
        case VERSION_SD1_INPAINT: return "sd1-inpaint";
        case VERSION_SD1_PIX2PIX: return "sd1-pix2pix";
        case VERSION_SD1_TINY_UNET: return "sd1-tiny-unet";
        case VERSION_SD2: return "sd2";
        case VERSION_SD2_INPAINT: return "sd2-inpaint";
        case VERSION_SD2_TINY_UNET: return "sd2-tiny-unet";
        case VERSION_SDXS_512_DS: return "sdxs-512-ds";
        case VERSION_SDXS_09: return "sdxs-09";
        case VERSION_SDXL: return "sdxl";
        case VERSION_SDXL_INPAINT: return "sdxl-inpaint";
        case VERSION_SDXL_PIX2PIX: return "sdxl-pix2pix";
        case VERSION_SDXL_VEGA: return "sdxl-vega";
        case VERSION_SDXL_SSD1B: return "sdxl-ssd1b";
        case VERSION_SVD: return "svd";
        case VERSION_SD3: return "sd3";
        case VERSION_FLUX: return "flux";
        case VERSION_FLUX_KONTEXT: return "flux-kontext";
        case VERSION_FLUX_FILL: return "flux-fill";
        case VERSION_FLUX_CONTROLS: return "flux-controls";
        case VERSION_FLEX_2: return "flex-2";
        case VERSION_CHROMA_RADIANCE: return "chroma-radiance";
        case VERSION_WAN2: return "wan2";
        case VERSION_WAN2_2_I2V: return "wan2.2-i2v";
        case VERSION_WAN2_2_TI2V: return "wan2.2-ti2v";
        case VERSION_QWEN_IMAGE: return "qwen-image";
        case VERSION_QWEN_IMAGE_EDIT: return "qwen-image-edit";
        case VERSION_MINIMAX_H3: return "minimax-h3";
        case VERSION_ANIMA: return "anima";
        case VERSION_FLUX2: return "flux2";
        case VERSION_FLUX2_KLEIN: return "flux2-klein";
        case VERSION_Z_IMAGE: return "z-image";
        case VERSION_OVIS_IMAGE: return "ovis-image";
        case VERSION_ERNIE_IMAGE: return "ernie-image";
        case VERSION_LTXAV: return "ltxav";
        case VERSION_COUNT:
        default: return "unknown";
    }
}

int detect_distilled_default_steps(const std::vector<std::string>& file_paths,
                                   const char* diffusion_model_path) {
    // Lowercase-scan model/DiT paths for few-step distill keywords. Conservative:
    // keep the keyword list tight so a base checkpoint is not misread as distilled.
    auto to_lower = [](const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return out;
    };

    std::string haystack;
    for (const std::string& p : file_paths) {
        haystack += to_lower(p);
        haystack += '\n';
    }
    if (diffusion_model_path != nullptr && diffusion_model_path[0] != '\0') {
        haystack += to_lower(diffusion_model_path);
    }
    if (haystack.empty()) {
        return 0;
    }

    auto has = [&](const char* kw) { return haystack.find(kw) != std::string::npos; };

    // Reliability ladder, most authoritative first. Step count and "is this
    // distilled" are independent facts: a keyword like `lightning` only says
    // distilled, not how many steps (Lightning/Hyper ship 4- and 8-step variants
    // alike), so the explicit `Nstep` marker in the path wins over any keyword.

    // 1. Explicit step marker in the path, e.g. `4steps`, `8-step`, `2_steps`.
    //    Bounded to a sane few-step range so long shard indices (…00009.safetensors)
    //    can never be misread as a step count.
    std::smatch m;
    static const std::regex step_re(R"(([0-9]{1,2})[ _-]?steps?)");
    if (std::regex_search(haystack, m, step_re)) {
        const int n = std::stoi(m[1].str());
        if (n >= 1 && n <= 64) {
            LOG_INFO("distilled step detection: explicit '%dstep' marker in path -> %d steps", n, n);
            return n;
        }
    }

    // 2. FLUX.1-schnell path fallback (the authoritative schnell signal is the
    //    architecture check in FluxPipeline::resolve_steps; this only helps
    //    same-family path scans). schnell is a fixed 4-step model.
    if (has("schnell")) {
        LOG_INFO("distilled step detection: 'schnell' in path -> 4 steps");
        return 4;
    }

    // 3. Distill keyword but no explicit step count: we know it is distilled but
    //    not how many steps. Default to 8 (the common Lightning/Turbo case) and
    //    say so, since a 4-step checkpoint here would run at the wrong step count.
    if (has("lightning") || has("lightx2v") || has("hyper") ||
        has("turbo") || has("distill")) {
        LOG_WARN("distilled step detection: distill keyword matched but no explicit "
                 "step count in path; defaulting to 8 steps. Pass --steps N to "
                 "override, or add an 'Nsteps' marker to the path/filename.");
        return 8;
    }
    return 0;
}

// Reverse of ed_version_name(): map a version string (as stored in GGUF
// metadata by ed-convert) back to an SDVersion. Returns VERSION_COUNT if the
// string matches no known version. Derived from ed_version_name so the two can
// never drift apart.
SDVersion ed_version_from_name(const std::string& name) {
    if (name == "ltxv" || name == "ltx2" || name == "ltx-2") {
        return VERSION_LTXAV;
    }
    for (int v = 0; v < VERSION_COUNT; ++v) {
        if (name == ed_version_name(static_cast<SDVersion>(v))) {
            return static_cast<SDVersion>(v);
        }
    }
    return VERSION_COUNT;
}

static bool has_suffix(const std::string& path, const std::string& suffix) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ends_with(lower, suffix);
}

static bool is_safetensors_index_file(const std::string& file_path) {
    return file_exists(file_path) && has_suffix(file_path, ".safetensors.index.json");
}

static std::string parent_path(const std::string& file_path) {
    return fs::path(file_path).parent_path().string();
}

static std::string resolve_model_path(const std::string& file_path) {
    if (file_exists(file_path) || is_directory(file_path)) {
        return file_path;
    }

    if (has_suffix(file_path, ".safetensors")) {
        const std::string index_path = file_path + ".index.json";
        if (file_exists(index_path)) {
            return index_path;
        }
    }

    return file_path;
}

static bool read_json_file(const std::string& path, nlohmann::json* json, std::string* error);

static bool is_sd3_diffusers_transformer_file(const std::string& file_path) {
    std::string normalized = file_path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return contains(normalized, "/transformer/") &&
           (ends_with(normalized, "diffusion_pytorch_model.safetensors") ||
            ends_with(normalized, "diffusion_pytorch_model.fp16.safetensors") ||
            ends_with(normalized, "diffusion_pytorch_model.safetensors.index.json"));
}

static SDVersion infer_transformer_file_version(const std::string& file_path) {
    std::string normalized = file_path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    std::string lower_normalized = normalized;
    std::transform(lower_normalized.begin(), lower_normalized.end(), lower_normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (contains(lower_normalized, "kontext")) {
        return VERSION_FLUX_KONTEXT;
    }
    if (!contains(normalized, "/transformer/") &&
        !contains(normalized, "/transformer_ref/")) {
        return VERSION_COUNT;
    }

    const std::string config_path = path_join(parent_path(file_path), "config.json");
    std::string error;
    nlohmann::json config;
    if (!file_exists(config_path) || !read_json_file(config_path, &config, &error)) {
        return is_sd3_diffusers_transformer_file(file_path) ? VERSION_SD3 : VERSION_COUNT;
    }

    if (config.contains("_class_name") && config["_class_name"].is_string()) {
        const std::string klass = config["_class_name"].get<std::string>();
        if (contains(klass, "QwenImageEdit") || contains(klass, "Qwen Image Edit")) {
            return VERSION_QWEN_IMAGE_EDIT;
        }
        if (contains(klass, "QwenImage") || contains(klass, "Qwen")) {
            return VERSION_QWEN_IMAGE;
        }
        if (contains(klass, "Kontext")) {
            return VERSION_FLUX_KONTEXT;
        }
        if (contains(klass, "Flux2")) {
            return contains(klass, "Klein") ? VERSION_FLUX2_KLEIN : VERSION_FLUX2;
        }
        if (contains(klass, "Flux")) {
            return VERSION_FLUX;
        }
        if (contains(klass, "SD3") || contains(klass, "StableDiffusion3")) {
            return VERSION_SD3;
        }
        if (contains(klass, "Wan")) {
            return VERSION_WAN2;
        }
        if (contains(klass, "MiniMaxH3") || contains(klass, "MiniMax-H3")) {
            return VERSION_MINIMAX_H3;
        }
        if (contains(klass, "LTX2") || contains(klass, "LTX")) {
            return VERSION_LTXAV;
        }
    }

    return VERSION_COUNT;
}

static bool is_flux1_family_version(SDVersion version) {
    return version == VERSION_FLUX || version == VERSION_FLUX_KONTEXT;
}

static bool is_flux2_family_version(SDVersion version) {
    return ed_version_is_flux2(version);
}

static std::string resolve_flux_transformer_component_path(const std::string& file_path) {
    std::string normalized = file_path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    if (!contains(normalized, "/transformer/")) {
        return file_path;
    }

    const std::string model_dir = parent_path(parent_path(file_path));
    const std::vector<std::string> top_level_flux_weights = {
        path_join(model_dir, "flux1-kontext-dev.safetensors"),
        path_join(model_dir, "flux1-dev.safetensors"),
        path_join(model_dir, "flux1-schnell.safetensors"),
        path_join(model_dir, "flux.safetensors"),
    };
    for (const std::string& candidate : top_level_flux_weights) {
        if (file_exists(candidate)) {
            LOG_INFO("using top-level Flux transformer weights '%s' instead of component weights '%s'",
                     candidate.c_str(),
                     file_path.c_str());
            return candidate;
        }
    }

    return file_path;
}

static bool read_json_file(const std::string& path, nlohmann::json* json, std::string* error) {
    std::ifstream file(path);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open '" + path + "'";
        }
        return false;
    }
    try {
        *json = nlohmann::json::parse(file);
    } catch (const std::exception& e) {
        if (error != nullptr) {
            *error = "failed to parse '" + path + "': " + e.what();
        }
        return false;
    }
    return true;
}

static SDVersion infer_diffusers_version(const std::string& dir_path) {
    std::string lower_dir = dir_path;
    std::replace(lower_dir.begin(), lower_dir.end(), '\\', '/');
    std::transform(lower_dir.begin(), lower_dir.end(), lower_dir.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (contains(lower_dir, "kontext")) {
        return VERSION_FLUX_KONTEXT;
    }

    std::string error;
    nlohmann::json index;
    const std::string model_index_path = path_join(dir_path, "model_index.json");
    if (file_exists(model_index_path) && read_json_file(model_index_path, &index, &error)) {
        if (index.contains("_class_name") && index["_class_name"].is_string()) {
            const std::string klass = index["_class_name"].get<std::string>();
            if (contains(klass, "Wan")) {
                return VERSION_WAN2;
            }
            if (contains(klass, "MiniMaxH3") || contains(klass, "MiniMax-H3")) {
                return VERSION_MINIMAX_H3;
            }
            if (contains(klass, "LTX2") || contains(klass, "LTX")) {
                return VERSION_LTXAV;
            }
            if (contains(klass, "QwenImageEdit") || contains(klass, "Qwen Image Edit")) {
                return VERSION_QWEN_IMAGE_EDIT;
            }
            if (contains(klass, "QwenImage") || contains(klass, "Qwen Image")) {
                return VERSION_QWEN_IMAGE;
            }
            if (contains(klass, "Kontext")) {
                return VERSION_FLUX_KONTEXT;
            }
            if (contains(klass, "Flux2")) {
                return contains(klass, "Klein") ? VERSION_FLUX2_KLEIN : VERSION_FLUX2;
            }
            if (contains(klass, "Flux")) {
                return VERSION_FLUX;
            }
            if (contains(klass, "StableDiffusion3")) {
                return VERSION_SD3;
            }
            if (contains(klass, "XL")) {
                return VERSION_SDXL;
            }
        }
    }

    const std::string transformer_config_path = path_join(path_join(dir_path, "transformer"), "config.json");
    if (file_exists(transformer_config_path) && read_json_file(transformer_config_path, &index, &error)) {
        if (index.contains("_class_name") && index["_class_name"].is_string()) {
            const std::string klass = index["_class_name"].get<std::string>();
            if (contains(klass, "Wan")) {
                return VERSION_WAN2;
            }
            if (contains(klass, "QwenImageEdit") || contains(klass, "Qwen Image Edit")) {
                return VERSION_QWEN_IMAGE_EDIT;
            }
            if (contains(klass, "QwenImage") || contains(klass, "Qwen")) {
                return VERSION_QWEN_IMAGE;
            }
            if (contains(klass, "Kontext")) {
                return VERSION_FLUX_KONTEXT;
            }
            if (contains(klass, "Flux2")) {
                return contains(klass, "Klein") ? VERSION_FLUX2_KLEIN : VERSION_FLUX2;
            }
            if (contains(klass, "Flux")) {
                return VERSION_FLUX;
            }
            if (contains(klass, "SD3")) {
                return VERSION_SD3;
            }
            if (contains(klass, "LTX2") || contains(klass, "LTX")) {
                return VERSION_LTXAV;
            }
        }
        return VERSION_FLUX;
    }

    if (is_directory(path_join(dir_path, "unet")) && is_directory(path_join(dir_path, "text_encoder_2"))) {
        return VERSION_SDXL;
    }
    if (is_directory(path_join(dir_path, "unet"))) {
        return VERSION_SD1;
    }

    return VERSION_COUNT;
}

static std::vector<std::string> component_weight_candidates(const std::string& component_dir) {
    std::vector<std::string> candidates;
    const std::vector<std::string> names = {
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
        "model.safetensors",
        "pytorch_model.safetensors",
        "adapter_model.safetensors",
    };
    for (const std::string& name : names) {
        candidates.push_back(path_join(component_dir, name));
    }

    const std::vector<std::string> index_names = {
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
    };
    for (const std::string& name : index_names) {
        candidates.push_back(path_join(component_dir, name));
    }

    std::error_code ec;
    if (fs::is_directory(component_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(component_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string path = entry.path().string();
            if (has_suffix(path, ".safetensors") || has_suffix(path, ".safetensors.index.json")) {
                candidates.push_back(path);
            }
        }
    }

    return candidates;
}

static std::string find_top_level_safetensors_file(const std::string& dir_path) {
    std::error_code ec;
    if (!fs::is_directory(dir_path, ec)) {
        return {};
    }

    for (const auto& entry : fs::directory_iterator(dir_path, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string path = entry.path().string();
        if (has_suffix(path, ".safetensors")) {
            return path;
        }
    }

    return {};
}

static uint16_t f8_e4m3_to_f16(uint8_t f8) {
    const uint32_t exponent_bias = 7;
    if (f8 == 0xff) {
        return ggml_fp32_to_fp16(-NAN);
    }
    if (f8 == 0x7f) {
        return ggml_fp32_to_fp16(NAN);
    }

    uint32_t sign     = f8 & 0x80;
    uint32_t exponent = (f8 & 0x78) >> 3;
    uint32_t mantissa = f8 & 0x07;
    uint32_t result   = sign << 24;
    if (exponent == 0) {
        if (mantissa > 0) {
            exponent = 0x7f - exponent_bias;
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            result |= (mantissa & 0x03) << 21;
            result |= exponent << 23;
        }
    } else {
        result |= mantissa << 20;
        exponent += 0x7f - exponent_bias;
        result |= exponent << 23;
    }
    return ggml_fp32_to_fp16(*reinterpret_cast<const float*>(&result));
}

static uint16_t f8_e5m2_to_f16(uint8_t fp8) {
    return static_cast<uint16_t>(fp8) << 8;
}

static void f8_e4m3_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e4m3_to_f16(src[i]);
    }
}

static void f8_e5m2_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e5m2_to_f16(src[i]);
    }
}

static void f64_to_f32_vec(double* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}

static void i64_to_i32_vec(int64_t* src, int32_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
}

// Builds the per-input-channel importance vector handed to ggml_quantize_chunk.
// When `imatrix_override` is non-null AND its length matches n_per_row (the
// in_features dimension the quantizer weights over), it is used verbatim (the imatrix
// path). Otherwise a length-n_per_row all-ones vector is returned, reproducing the
// historical on-the-fly behavior exactly. This keeps runtime quantization (which
// never sets an override) byte-for-byte unchanged.
static std::vector<float> build_quant_imatrix(int n_per_row, const std::vector<float>* imatrix_override) {
    if (imatrix_override != nullptr && static_cast<int>(imatrix_override->size()) == n_per_row) {
        return *imatrix_override;
    }
    return std::vector<float>(static_cast<size_t>(n_per_row), 1.0f);
}

static bool convert_tensor_data(void* src, ggml_type src_type, void* dst, ggml_type dst_type, int nrows, int n_per_row,
                                const std::vector<float>* imatrix_override = nullptr) {
    const int n = nrows * n_per_row;
    if (src_type == dst_type) {
        const size_t nbytes = static_cast<size_t>(n) * ggml_type_size(src_type) / ggml_blck_size(src_type);
        std::memcpy(dst, src, nbytes);
        return true;
    }
    if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(static_cast<float*>(src), static_cast<ggml_fp16_t*>(dst), n);
            return true;
        }
        std::vector<float> imatrix = build_quant_imatrix(n_per_row, imatrix_override);
        return ggml_quantize_chunk(dst_type, static_cast<float*>(src), dst, 0, nrows, n_per_row, imatrix.data()) >= 0;
    }
    if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(static_cast<ggml_fp16_t*>(src), static_cast<float*>(dst), n);
            return true;
        }
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == nullptr) {
            return false;
        }
        qtype->to_float(src, static_cast<float*>(dst), n);
        return true;
    }

    auto qtype = ggml_get_type_traits(src_type);
    if (qtype->to_float == nullptr) {
        return false;
    }
    std::vector<float> tmp(static_cast<size_t>(n));
    qtype->to_float(src, tmp.data(), n);
    if (dst_type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(tmp.data(), static_cast<ggml_fp16_t*>(dst), n);
        return true;
    }
    std::vector<float> imatrix = build_quant_imatrix(n_per_row, imatrix_override);
    return ggml_quantize_chunk(dst_type, tmp.data(), dst, 0, nrows, n_per_row, imatrix.data()) >= 0;
}


void ModelLoader::clear() {
    version_ = VERSION_COUNT;

    file_paths_.clear();
    tensor_storage_map_.clear();
    last_error_.clear();

    tensors_.clear();
    ignore_tensors_.clear();

    external_vae_is_invalid_ = false;
    use_tae_ = false;
    tae_preview_only_ = false;
    use_pmid_ = false;
    skip_t5_ = false;
    qwen_image_zero_cond_t_ = false;
}

void ModelLoader::reset() {
    clear();
}

bool ModelLoader::non_empty(const char* path) {
    return path != nullptr && path[0] != '\0';
}

bool ModelLoader::load_optional_file(const char* path,
                                     const std::string& prefix,
                                     const char* label,
                                     bool required,
                                     std::string* error) {
    if (!non_empty(path)) {
        if (required) {
            const std::string msg = std::string("missing required ") + label + " path";
            if (error != nullptr) {
                *error = msg;
            }
            set_error(msg);
            return false;
        }
        return false;
    }

    LOG_INFO("loading %s from '%s'", label, path);

    if (!init_from_file(path, prefix)) {
        const std::string msg = last_error_.empty()
                                    ? std::string("loading ") + label + " from '" + path + "' failed"
                                    : last_error_;

        if (required) {
            if (error != nullptr) {
                *error = msg;
            }
            set_error(msg);
            return false;
        }

        LOG_WARN("%s", msg.c_str());
        return false;
    }

    return true;
}

bool ModelLoader::load_model_files(const ed_context_params_t& params,
                                   std::string* error) {
    bool loaded_any = false;
    skip_t5_ = params.skip_t5;
    auto load_user_component = [&](const char* path,
                                   const std::string& prefix,
                                   const char* label) -> bool {
        if (!non_empty(path)) {
            return true;
        }
        if (!load_optional_file(path, prefix, label, true, error)) {
            return false;
        }
        loaded_any = true;
        return true;
    };

    loaded_any = load_optional_file(params.model_path,
                                    "",
                                    "model",
                                    false,
                                    error) || loaded_any;

    const SDVersion hint_version = get_ld_version();
    const bool is_unet_hint = hint_version != VERSION_COUNT &&
                              ed_version_is_unet(hint_version);

    std::string diffusion_model_path;
    if (non_empty(params.diffusion_model_path)) {
        diffusion_model_path = params.diffusion_model_path;
        const SDVersion transformer_file_version = infer_transformer_file_version(diffusion_model_path);
        if (transformer_file_version != VERSION_COUNT) {
            version_ = transformer_file_version;
        }
        if (is_flux1_family_version(version_)) {
            diffusion_model_path = resolve_flux_transformer_component_path(diffusion_model_path);
        }
    }

    if (!load_user_component(diffusion_model_path.empty() ? nullptr : diffusion_model_path.c_str(),
                             "model.diffusion_model.",
                             "diffusion model")) {
        return false;
    }

    if (!load_user_component(params.high_noise_diffusion_model_path,
                             "model.high_noise_diffusion_model.",
                             "high noise diffusion model")) {
        return false;
    }

    if (!load_user_component(params.clip_l_path,
                             is_unet_hint
                                 ? "cond_stage_model.transformer."
                                 : "text_encoders.clip_l.transformer.",
                             "clip_l")) {
        return false;
    }

    if (!load_user_component(params.clip_g_path,
                             is_unet_hint
                                 ? "cond_stage_model.1.transformer."
                                 : "text_encoders.clip_g.transformer.",
                             "clip_g")) {
        return false;
    }

    if (!load_user_component(params.clip_vision_path,
                             "cond_stage_model.transformer.",
                             "clip_vision")) {
        return false;
    }

    if (!params.skip_t5) {
        if (!load_user_component(params.t5xxl_path,
                                 "text_encoders.t5xxl.transformer.",
                                 "t5xxl")) {
            return false;
        }
    }

    if (!load_user_component(params.llm_path,
                             "text_encoders.llm.",
                             "llm")) {
        return false;
    }

    if (!load_user_component(params.llm_vision_path,
                             "text_encoders.llm.visual.",
                             "llm vision")) {
        return false;
    }

    if (!load_user_component(params.embeddings_connectors_path,
                             "",
                             "embeddings connectors")) {
        return false;
    }

    if (non_empty(params.vae_path)) {
        const bool ok = load_optional_file(params.vae_path,
                                           "vae.",
                                           "vae",
                                           true,
                                           error);
        external_vae_is_invalid_ = !ok;
        if (!ok) {
            return false;
        }
        loaded_any = true;
    }

    if (non_empty(params.audio_vae_path)) {
        const SDVersion component_version = version_ == VERSION_COUNT ? get_ld_version() : version_;
        if (!load_optional_file(params.audio_vae_path,
                                ed_version_is_ltxav(component_version) ? "" : "audio_vae.",
                                "audio vae",
                                true,
                                error)) {
            return false;
        }
        loaded_any = true;
    }

    if (non_empty(params.taesd_path)) {
        const bool ok = load_optional_file(params.taesd_path,
                                           "tae.",
                                           "tae",
                                           true,
                                           error);
        use_tae_ = true;
        if (!ok) {
            return false;
        }
        loaded_any = true;
    }

    if (!loaded_any || tensor_storage_map_.empty()) {
        const std::string msg = "no model tensors were loaded";
        if (error != nullptr) {
            *error = msg;
        }
        set_error(msg);
        return false;
    }

    return true;
}

bool ModelLoader::finalize_names_and_version(std::string* error) {
    const SDVersion hinted_version = version_;
    convert_tensors_name();

    const SDVersion inferred_version = get_ld_version();
    if (hinted_version == VERSION_QWEN_IMAGE_EDIT && inferred_version == VERSION_QWEN_IMAGE) {
        version_ = hinted_version;
    } else if (hinted_version == VERSION_FLUX_KONTEXT && inferred_version == VERSION_FLUX) {
        version_ = hinted_version;
    } else {
        version_ = inferred_version != VERSION_COUNT ? inferred_version : hinted_version;
    }
    if (version_ == VERSION_COUNT) {
        const std::string msg = "failed to infer model version from loaded tensors";
        if (error != nullptr) {
            *error = msg;
        }
        set_error(msg);
        return false;
    }

    LOG_INFO("model loader initialized: version=%s, files=%zu, tensors=%zu",
             ed_version_name(version_),
             file_paths_.size(),
             tensor_storage_map_.size());

    return true;
}

bool ModelLoader::apply_dtype_policy(const ed_context_params_t& params,
                                     std::string* error) {
    (void)error;

    ggml_type wtype = ed_dtype_to_ggml(params.weight_type);
    // Qwen-Image / Qwen-Image-Edit (incl. their distilled/lightning variants) produce
    // corrupt (all-white) output under FP16: the DiT activations exceed FP16's dynamic
    // range and silently saturate to Inf/NaN, which the VAE clamps to white. BF16 has the
    // same width but a wider exponent, so it runs correctly. Until FP16 is properly
    // supported, transparently switch a requested F16 to BF16 for these models and warn.
    // Reached by BOTH the CLI and the benchmark (they share this loader), so this one guard
    // covers every entry point. version_ is already resolved here (finalize_names_and_version
    // runs before apply_dtype_policy).
    if (wtype == GGML_TYPE_F16 &&
        (ed_version_is_qwen_image(version_) || ed_version_is_qwen_image_edit(version_))) {
        LOG_WARN("edge-dit does not yet support FP16 for Qwen models; automatically switched to BF16");
        wtype = GGML_TYPE_BF16;
    }
    const std::string tensor_type_rules =
        params.tensor_type_rules != nullptr ? params.tensor_type_rules : "";
    if (wtype != GGML_TYPE_COUNT || !tensor_type_rules.empty()) {
        set_wtype_override(wtype, tensor_type_rules);
    }

    return true;
}

bool ModelLoader::bind_weights(int n_threads,
                               bool use_mmap,
                               std::string* error) {
    if (tensors_.empty()) {
        const std::string msg = "no target tensors were prepared for model weights";
        if (error != nullptr) {
            *error = msg;
        }
        set_error(msg);
        return false;
    }

    if (!load_tensors(tensors_, ignore_tensors_, n_threads, use_mmap)) {
        const std::string msg = last_error_.empty()
                                    ? "failed to load tensors"
                                    : last_error_;
        if (error != nullptr) {
            *error = msg;
        }
        set_error(msg);
        return false;
    }

    return true;
}

bool ModelLoader::bind_weights(const TensorMap& tensors,
                               const IgnoreTensorSet& ignore_tensors,
                               int n_threads,
                               bool use_mmap,
                               std::string* error) {
    tensors_ = tensors;
    ignore_tensors_ = ignore_tensors;
    return bind_weights(n_threads, use_mmap, error);
}

ggml_type ModelLoader::ed_dtype_to_ggml(ed_dtype_t dtype) {
    switch (dtype) {
        case ED_DTYPE_F32:  return GGML_TYPE_F32;
        case ED_DTYPE_F16:  return GGML_TYPE_F16;
        case ED_DTYPE_BF16: return GGML_TYPE_BF16;
        case ED_DTYPE_Q4_0: return GGML_TYPE_Q4_0;
        case ED_DTYPE_Q4_1: return GGML_TYPE_Q4_1;
        case ED_DTYPE_Q5_0: return GGML_TYPE_Q5_0;
        case ED_DTYPE_Q5_1: return GGML_TYPE_Q5_1;
        case ED_DTYPE_Q8_0: return GGML_TYPE_Q8_0;
        case ED_DTYPE_Q2_K: return GGML_TYPE_Q2_K;
        case ED_DTYPE_Q3_K: return GGML_TYPE_Q3_K;
        case ED_DTYPE_Q4_K: return GGML_TYPE_Q4_K;
        case ED_DTYPE_Q5_K: return GGML_TYPE_Q5_K;
        case ED_DTYPE_Q6_K: return GGML_TYPE_Q6_K;
        case ED_DTYPE_AUTO:
        default:
            return GGML_TYPE_COUNT;
    }
}

std::string ModelLoader::wtype_stat_to_str(const std::map<ggml_type, uint32_t>& stat) {
    std::ostringstream ss;
    bool first = true;

    for (const auto& [type, count] : stat) {
        if (!first) {
            ss << "|";
        }
        first = false;
        ss << ggml_type_name(type) << ":" << count;
    }

    return ss.str();
}

void ModelLoader::log_weight_stats() const {
    LOG_INFO("Weight type stat: %s",
             wtype_stat_to_str(get_wtype_stat()).c_str());

    LOG_INFO("Conditioner weight type stat: %s",
             wtype_stat_to_str(get_conditioner_wtype_stat()).c_str());

    LOG_INFO("Diffusion model weight type stat: %s",
             wtype_stat_to_str(get_diffusion_model_wtype_stat()).c_str());

    LOG_INFO("VAE weight type stat: %s",
             wtype_stat_to_str(get_vae_wtype_stat()).c_str());
}

void ModelLoader::set_error(const std::string& error) {
    last_error_ = error;
    if (!error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }
}

void ModelLoader::add_tensor_storage(const TensorStorage& tensor_storage) {
    tensor_storage_map_[tensor_storage.name] = tensor_storage;
}

std::string ModelLoader::resolve_bare_transformer_prefix(const std::string& resolved_path,
                                                         const std::string& prefix) {
    std::string lower_name = fs::path(resolved_path).filename().string();
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    // A caller-supplied prefix wins for tensor placement (for example,
    // --diffusion-model supplies "model.diffusion_model."). It must not suppress
    // architecture discovery, though: standalone transformer conversion still
    // needs the sibling config.json/name hint to canonicalize family-specific
    // tensor names and persist model_version without a user-provided hint.
    if (!prefix.empty()) {
        if (version_ == VERSION_COUNT) {
            if (contains(lower_name, "flux")) {
                version_ = contains(lower_name, "kontext") ? VERSION_FLUX_KONTEXT : VERSION_FLUX;
            } else if (contains(lower_name, "minimax_h3") || contains(lower_name, "minimax-h3")) {
                version_ = VERSION_MINIMAX_H3;
            } else {
                version_ = infer_transformer_file_version(resolved_path);
            }
        }
        return prefix;
    }

    if (contains(lower_name, "flux")) {
        version_ = contains(lower_name, "kontext") ? VERSION_FLUX_KONTEXT : VERSION_FLUX;
        return "transformer.";
    }
    if (contains(lower_name, "minimax_h3") || contains(lower_name, "minimax-h3")) {
        version_ = VERSION_MINIMAX_H3;
        return prefix;
    }

    // A bare diffusers transformer file/shard-index (…/transformer/…) carries no
    // top-level config, so get_ld_version() -- which keys on canonical names --
    // cannot recover the family before convert_tensors_name() runs, and that name
    // mapping itself needs the version (chicken-and-egg). Seed version_ from the
    // sibling transformer/config.json (_class_name), then return the "transformer."
    // component prefix so convert_tensor_name rewrites the DiT to
    // "model.diffusion_model.*" (matching how init_from_diffusers_directory loads
    // the transformer/ subdir). This lets offline convert of a standalone
    // transformer record the right family in the GGUF metadata and canonicalize
    // names, so the result loads standalone or via --diffusion-model.
    if (version_ == VERSION_COUNT) {
        const SDVersion transformer_file_version = infer_transformer_file_version(resolved_path);
        if (transformer_file_version != VERSION_COUNT) {
            version_ = transformer_file_version;
            return "transformer.";
        }
    }
    return prefix;
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    last_error_.clear();
    const std::string resolved_path = resolve_model_path(file_path);
    if (resolved_path != file_path) {
        LOG_INFO("resolved model path '%s' to '%s'", file_path.c_str(), resolved_path.c_str());
    }

    if (is_directory(resolved_path)) {
        LOG_INFO("load %s using diffusers directory format", resolved_path.c_str());
        return init_from_diffusers_directory(resolved_path, prefix);
    }
    if (is_gguf_file(resolved_path)) {
        LOG_INFO("load %s using gguf format", resolved_path.c_str());
        // FLUX-Kontext and Qwen-Image-Edit are architecturally identical to their
        // base variants (plain FLUX / Qwen-Image) -- same tensor names and shapes --
        // so get_ld_version() cannot tell them apart. Diffusers loading distinguishes
        // them via config.json, but a converted GGUF has no config. Seed a version
        // hint from the file name (the only surviving signal); finalize_names_and_version
        // keeps this hint when tensor inference falls back to the base variant.
        if (prefix.empty()) {
            std::string lower_name = fs::path(resolved_path).filename().string();
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (contains(lower_name, "kontext")) {
                version_ = VERSION_FLUX_KONTEXT;
            } else if (contains(lower_name, "qwen") && contains(lower_name, "edit")) {
                version_ = VERSION_QWEN_IMAGE_EDIT;
            }
        }
        return init_from_gguf_file(resolved_path, prefix);
    }
    if (is_safetensors_file(resolved_path)) {
        LOG_INFO("load %s using safetensors format", resolved_path.c_str());
        const std::string effective_prefix = resolve_bare_transformer_prefix(resolved_path, prefix);
        return init_from_safetensors_file(resolved_path, effective_prefix);
    }
    if (is_safetensors_index_file(resolved_path)) {
        LOG_INFO("load %s using safetensors shard index format", resolved_path.c_str());
        const std::string effective_prefix = resolve_bare_transformer_prefix(resolved_path, prefix);
        return init_from_safetensors_index_file(resolved_path, effective_prefix);
    }

    set_error(file_exists(resolved_path) ? "unsupported model format: " + resolved_path : "model path not found: " + resolved_path);
    return false;
}

void ModelLoader::convert_tensors_name() {
    if (version_ == VERSION_COUNT) {
        version_ = get_ld_version();
    }
    SDVersion version = version_;
    if (version == VERSION_COUNT) {
        LOG_WARN("model version is unknown; tensor names are left mostly unchanged");
    }

    String2TensorStorage new_map;
    size_t swiglu_half_swaps = 0;
    for (auto& item : tensor_storage_map_) {
        TensorStorage tensor_storage = item.second;
        const std::string original_name = tensor_storage.name;
        tensor_storage.name = convert_tensor_name(tensor_storage.name, version);
        // diffusers FLUX final adaLN modulation ("norm_out.linear") is laid out as
        // [shift, scale] on dim0; the engine and BFL checkpoints use [scale, shift].
        // Flag it so the two halves are swapped when the weight data is loaded.
        if ((ed_version_is_flux(version) || ed_version_is_flux2(version)) &&
            contains(original_name, "norm_out.linear.")) {
            tensor_storage.swap_scale_shift = true;
        }
        if (ed_version_is_minimax_h3(version) &&
            contains(original_name, ".ff.net.0.proj.")) {
            tensor_storage.swap_swiglu_halves = true;
            ++swiglu_half_swaps;
        }
        new_map[tensor_storage.name] = std::move(tensor_storage);
    }
    tensor_storage_map_.swap(new_map);
    if (swiglu_half_swaps > 0) {
        LOG_DEBUG("marked %zu MiniMax-H3 diffusers SwiGLU tensors for half swap", swiglu_half_swaps);
    }
}

bool ModelLoader::init_from_file_and_convert_name(const std::string& file_path, const std::string& prefix, SDVersion version) {
    if (!init_from_file(file_path, prefix)) {
        return false;
    }
    if (version != VERSION_COUNT) {
        version_ = version;
    }
    if (version_ == VERSION_COUNT) {
        version_ = get_ld_version();
    }
    convert_tensors_name();
    LOG_INFO("model loader initialized: version=%s, files=%zu, tensors=%zu",
             ed_version_name(version_), file_paths_.size(), tensor_storage_map_.size());
    return true;
}

static const char* canonical_component_prefix(const std::string& component_kind) {
    if (component_kind == "diffusion-model") return "model.diffusion_model.";
    // convert_tensor_name canonicalizes the load-site `vae.` prefix into
    // `first_stage_model.` before ed-convert writes the component GGUF.
    if (component_kind == "vae") return "first_stage_model.";
    if (component_kind == "audio-vae") return "audio_vae.";
    if (component_kind == "clip-l") return "text_encoders.clip_l.transformer.";
    if (component_kind == "clip-g") return "text_encoders.clip_g.transformer.";
    if (component_kind == "t5xxl") return "text_encoders.t5xxl.transformer.";
    if (component_kind == "llm") return "text_encoders.llm.";
    if (component_kind == "llm-vision") return "text_encoders.llm.visual.";
    return "";
}

static std::string expected_component_kind(const std::string& prefix) {
    if (prefix == "model.diffusion_model.") return "diffusion-model";
    if (prefix == "vae.") return "vae";
    if (prefix == "audio_vae.") return "audio-vae";
    if (prefix == "text_encoders.clip_l.transformer." ||
        prefix == "cond_stage_model.transformer.") {
        return "clip-l";
    }
    if (prefix == "text_encoders.clip_g.transformer." ||
        prefix == "cond_stage_model.1.transformer.") {
        return "clip-g";
    }
    if (prefix == "text_encoders.t5xxl.transformer.") return "t5xxl";
    if (prefix == "text_encoders.llm.") return "llm";
    if (prefix == "text_encoders.llm.visual.") return "llm-vision";
    return "";
}

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    std::vector<TensorStorage> tensor_storages;
    std::string error;
    std::string metadata_version;
    std::string metadata_component;
    if (!read_gguf_file(file_path, tensor_storages, &error, &metadata_version, &metadata_component)) {
        set_error(error);
        return false;
    }

    if (!prefix.empty() && !metadata_component.empty()) {
        const std::string expected = expected_component_kind(prefix);
        if (!expected.empty() && expected != metadata_component) {
            set_error("GGUF component kind mismatch: '" + file_path + "' contains " +
                      metadata_component + " but was loaded as " + expected);
            return false;
        }
    }

    // Prefer the explicit version stored in the GGUF metadata (written by
    // ed-convert) over the file-name keyword hint seeded in init_from_file.
    // This is what lets a renamed FLUX-Kontext / Qwen-Image-Edit GGUF still be
    // recognized. Older GGUFs lack this key, so the file-name fallback stays.
    if (prefix.empty() && !metadata_version.empty()) {
        const SDVersion meta_version = ed_version_from_name(metadata_version);
        if (meta_version != VERSION_COUNT) {
            version_ = meta_version;
            LOG_INFO("gguf metadata model version: %s", metadata_version.c_str());
        } else {
            LOG_WARN("gguf metadata has unknown model version '%s'; falling back to file-name hint",
                     metadata_version.c_str());
        }
    }

    file_paths_.push_back(file_path);
    const size_t file_index = file_paths_.size() - 1;
    for (TensorStorage tensor_storage : tensor_storages) {
        if (!prefix.empty() && !metadata_component.empty()) {
            // Standalone component GGUFs persist a stable canonical prefix plus
            // semantic component metadata. Rebase that prefix to the load-site
            // prefix so the same CLIP-L file works for both UNet pipelines
            // (cond_stage_model.*) and DiT pipelines (text_encoders.*).
            const std::string stored_prefix = canonical_component_prefix(metadata_component);
            if (!stored_prefix.empty() && starts_with(tensor_storage.name, stored_prefix)) {
                tensor_storage.name = prefix + tensor_storage.name.substr(stored_prefix.size());
            } else if (!starts_with(tensor_storage.name, prefix)) {
                tensor_storage.name = prefix + tensor_storage.name;
            }
        } else if (!prefix.empty() && !starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;
        add_tensor_storage(tensor_storage);
    }
    return true;
}

bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_safetensors_file(file_path, tensor_storages, &error)) {
        set_error(error);
        return false;
    }

    file_paths_.push_back(file_path);
    const size_t file_index = file_paths_.size() - 1;
    for (TensorStorage tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        if (!prefix.empty() && !starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;
        add_tensor_storage(tensor_storage);
    }
    return true;
}

bool ModelLoader::init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix) {
    nlohmann::json index;
    std::string error;
    if (!read_json_file(file_path, &index, &error)) {
        set_error(error);
        return false;
    }
    if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
        set_error("invalid safetensors index '" + file_path + "': missing weight_map");
        return false;
    }

    std::set<std::string> shard_names;
    for (const auto& item : index["weight_map"].items()) {
        if (!item.value().is_string()) {
            set_error("invalid safetensors index '" + file_path + "': non-string shard name");
            return false;
        }
        shard_names.insert(item.value().get<std::string>());
    }
    if (shard_names.empty()) {
        set_error("invalid safetensors index '" + file_path + "': empty weight_map");
        return false;
    }

    const std::string base_dir = parent_path(file_path);
    for (const std::string& shard_name : shard_names) {
        const std::string shard_path = path_join(base_dir, shard_name);
        if (!init_from_safetensors_file(shard_path, prefix)) {
            return false;
        }
    }
    return true;
}

bool ModelLoader::init_from_diffusers_directory(const std::string& dir_path, const std::string& prefix) {
    (void)prefix;
    version_ = infer_diffusers_version(dir_path);
    qwen_image_zero_cond_t_ = false;
    if (ed_version_is_qwen_image(version_) || ed_version_is_qwen_image_edit(version_)) {
        const std::string transformer_config_path = path_join(path_join(dir_path, "transformer"), "config.json");
        nlohmann::json transformer_config;
        std::string error;
        if (file_exists(transformer_config_path) &&
            read_json_file(transformer_config_path, &transformer_config, &error) &&
            transformer_config.contains("zero_cond_t") &&
            transformer_config["zero_cond_t"].is_boolean()) {
            qwen_image_zero_cond_t_ = transformer_config["zero_cond_t"].get<bool>();
            if (qwen_image_zero_cond_t_) {
                LOG_INFO("qwen-image transformer config enables zero_cond_t");
            }
        }
    }

    struct Component {
        const char* dir;
        const char* sd_prefix;
        const char* flux_prefix;
        bool required_for_flux;
    };

    const Component components[] = {
        {"transformer", "transformer.", "transformer.", true},
        {"unet", "unet.", "unet.", false},
        {"vae", "vae.", "vae.", false},
        {"text_encoder", "te.", "te1.", false},
        {"text_encoder_2", "te2.", "te3.", false},
        {"text_encoder_3", "te3.", "te3.", false},
    };

    size_t before_all = tensor_storage_map_.size();
    for (const Component& component : components) {
        if (skip_t5_ && std::strcmp(component.dir, "text_encoder_3") == 0) {
            continue;
        }
        const std::string component_dir = path_join(dir_path, component.dir);
        if (!is_directory(component_dir)) {
            if (component.required_for_flux && (is_flux1_family_version(version_) || is_flux2_family_version(version_))) {
                LOG_WARN("diffusers component '%s' not found", component.dir);
            }
            continue;
        }

        std::string component_prefix = is_flux1_family_version(version_) ? component.flux_prefix : component.sd_prefix;
        if (ed_version_is_wan(version_)) {
            if (std::strcmp(component.dir, "text_encoder") == 0) {
                component_prefix = "text_encoders.t5xxl.transformer.";
            } else if (std::strcmp(component.dir, "text_encoder_2") == 0 ||
                       std::strcmp(component.dir, "text_encoder_3") == 0 ||
                       std::strcmp(component.dir, "unet") == 0) {
                continue;
            }
        }
        if (is_flux2_family_version(version_)) {
            if (std::strcmp(component.dir, "text_encoder") == 0) {
                component_prefix = "text_encoders.llm.";
            } else if (std::strcmp(component.dir, "text_encoder_2") == 0 ||
                       std::strcmp(component.dir, "text_encoder_3") == 0 ||
                       std::strcmp(component.dir, "unet") == 0) {
                continue;
            }
        }
        if (ed_version_is_qwen_image(version_) || ed_version_is_qwen_image_edit(version_)) {
            if (std::strcmp(component.dir, "text_encoder") == 0) {
                component_prefix = "text_encoders.llm.";
            } else if (std::strcmp(component.dir, "text_encoder_2") == 0 ||
                       std::strcmp(component.dir, "text_encoder_3") == 0 ||
                       std::strcmp(component.dir, "unet") == 0) {
                continue;
            }
        }
        bool loaded = false;
        std::set<std::string> tried;

        if (is_flux2_family_version(version_) && std::strcmp(component.dir, "transformer") == 0) {
            const std::string top_level_flux = find_top_level_safetensors_file(dir_path);
            if (!top_level_flux.empty()) {
                const size_t before = tensor_storage_map_.size();
                loaded = init_from_safetensors_file(top_level_flux, component_prefix);
                if (loaded) {
                    LOG_INFO("loaded diffusers component '%s' from top-level Flux2 weights '%s' (%zu tensors)",
                             component.dir,
                             top_level_flux.c_str(),
                             tensor_storage_map_.size() - before);
                }
            }
        }
        if (is_flux1_family_version(version_) && std::strcmp(component.dir, "transformer") == 0) {
            const std::vector<std::string> top_level_flux_weights = {
                path_join(dir_path, "flux1-kontext-dev.safetensors"),
                path_join(dir_path, "flux1-dev.safetensors"),
                path_join(dir_path, "flux1-schnell.safetensors"),
                path_join(dir_path, "flux.safetensors"),
            };
            for (const std::string& top_level_flux : top_level_flux_weights) {
                if (!file_exists(top_level_flux)) {
                    continue;
                }
                const size_t before = tensor_storage_map_.size();
                loaded = init_from_safetensors_file(top_level_flux, component_prefix);
                if (loaded) {
                    LOG_INFO("loaded diffusers component '%s' from top-level Flux weights '%s' (%zu tensors)",
                             component.dir,
                             top_level_flux.c_str(),
                             tensor_storage_map_.size() - before);
                    break;
                }
            }
        }
        if (is_flux1_family_version(version_) && std::strcmp(component.dir, "vae") == 0) {
            const std::string top_level_ae = path_join(dir_path, "ae.safetensors");
            if (file_exists(top_level_ae)) {
                const size_t before = tensor_storage_map_.size();
                loaded = init_from_safetensors_file(top_level_ae, component_prefix);
                if (loaded) {
                    LOG_INFO("loaded diffusers component '%s' from '%s' (%zu tensors)",
                             component.dir, top_level_ae.c_str(), tensor_storage_map_.size() - before);
                }
            }
        }
        for (const std::string& candidate : component_weight_candidates(component_dir)) {
            if (loaded) {
                break;
            }
            if (!tried.insert(candidate).second || !file_exists(candidate)) {
                continue;
            }
            const size_t before = tensor_storage_map_.size();
            if (is_safetensors_index_file(candidate)) {
                loaded = init_from_safetensors_index_file(candidate, component_prefix);
            } else if (is_safetensors_file(candidate)) {
                loaded = init_from_safetensors_file(candidate, component_prefix);
            }
            if (loaded) {
                LOG_INFO("loaded diffusers component '%s' from '%s' (%zu tensors)",
                         component.dir, candidate.c_str(), tensor_storage_map_.size() - before);
                break;
            }
        }

        if (!loaded) {
            LOG_WARN("diffusers component '%s' exists but no supported safetensors weights were loaded", component.dir);
        }
    }

    if (tensor_storage_map_.size() == before_all) {
        const std::string kontext = path_join(dir_path, "flux1-kontext-dev.safetensors");
        if (file_exists(kontext)) {
            LOG_INFO("loading top-level safetensors '%s'", kontext.c_str());
            return init_from_safetensors_file(kontext, "transformer.");
        }
        const std::string single = path_join(dir_path, "flux1-dev.safetensors");
        if (file_exists(single)) {
            LOG_INFO("loading top-level safetensors '%s'", single.c_str());
            return init_from_safetensors_file(single, "transformer.");
        }
        set_error("diffusers directory contains no supported safetensors weights: " + dir_path);
        return false;
    }

    return true;
}

SDVersion ModelLoader::get_ld_version() {
    bool has_flux_double = false;
    bool has_flux_single = false;
    bool has_transformer_blocks = false;
    bool has_unet = false;
    bool has_second_text_encoder = false;
    bool has_qwen3_vl_language = false;
    bool has_qwen3_vl_vision = false;
    bool has_flux2 = false;
    bool has_single_block_47 = false;

    TensorStorage input_block_weight;
    TensorStorage token_embedding_weight;

    for (const auto& item : tensor_storage_map_) {
        const std::string& name = item.second.name;
        if (contains(name, "model.diffusion_model.joint_blocks.")) {
            return VERSION_SD3;
        }
        if (contains(name, "model.diffusion_model.transformer_blocks.0.img_mod.1.weight")) {
            return VERSION_QWEN_IMAGE;
        }
        if (contains(name, "model.diffusion_model.video_patch_proj.weight") &&
            tensor_storage_map_.find("model.diffusion_model.audio_patch_proj.weight") != tensor_storage_map_.end()) {
            return VERSION_MINIMAX_H3;
        }
        if (contains(name, "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias")) {
            return VERSION_LTXAV;
        }
        if (contains(name, "model.language_model.layers.") ||
            contains(name, "text_encoders.llm.model.language_model.layers.")) {
            has_qwen3_vl_language = true;
        }
        if (contains(name, "model.visual.blocks.") ||
            contains(name, "text_encoders.llm.model.visual.blocks.")) {
            has_qwen3_vl_vision = true;
        }
        // Wan video DiT: blocks carry a cross_attn sub-module (text conditioning)
        // that no other supported architecture uses, and a 3-D patch_embedding.
        // Diffusers loading recognizes Wan via config.json's "Wan" class, but a
        // converted GGUF has no config, so infer it from these signature tensors.
        // All Wan sub-variants map to VERSION_WAN2 here, matching infer_diffusers_version.
        if (contains(name, "model.diffusion_model.blocks.") && contains(name, ".cross_attn.")) {
            return VERSION_WAN2;
        }
        if (contains(name, "model.diffusion_model.double_blocks.") || contains(name, "transformer.double_blocks.")) {
            has_flux_double = true;
        }
        if (contains(name, "model.diffusion_model.double_stream_modulation_img.lin.weight") ||
            contains(name, "model.diffusion_model.double_stream_modulation_img.linear.weight")) {
            has_flux2 = true;
        }
        if (contains(name, "single_transformer_blocks.") || contains(name, "single_blocks.")) {
            has_flux_single = true;
        }
        if (contains(name, "single_blocks.47.linear1.weight")) {
            has_single_block_47 = true;
        }
        if (contains(name, "transformer.transformer_blocks.") || contains(name, "transformer_blocks.")) {
            has_transformer_blocks = true;
        }
        if (contains(name, "model.diffusion_model.input_blocks.") || contains(name, "unet.down_blocks.") || contains(name, "unet.conv_in.weight")) {
            has_unet = true;
        }
        if (starts_with(name, "te2.") || starts_with(name, "text_encoder_2.") || starts_with(name, "cond_stage_model.1")) {
            has_second_text_encoder = true;
        }
        if (name == "model.diffusion_model.img_in.weight" || name == "transformer.x_embedder.weight" || name == "transformer.img_in.weight") {
            input_block_weight = item.second;
        }
        if (ends_with(name, "text_model.embeddings.token_embedding.weight")) {
            token_embedding_weight = item.second;
        }
    }

    if (has_flux2) {
        if (has_single_block_47) {
            return VERSION_FLUX2;
        }
        return VERSION_FLUX2_KLEIN;
    }
    if (has_flux_double || (has_transformer_blocks && has_flux_single)) {
        if (input_block_weight.ne[0] == 384) {
            return VERSION_FLUX_FILL;
        }
        if (input_block_weight.ne[0] == 128) {
            return VERSION_FLUX_CONTROLS;
        }
        return VERSION_FLUX;
    }
    // Either half is enough to identify a Qwen3-VL component during standalone
    // conversion. Requiring both forced callers to supply an unrelated pipeline
    // hint for a language-only or vision-only checkpoint.
    if (has_qwen3_vl_language || has_qwen3_vl_vision) {
        return VERSION_MINIMAX_H3;
    }
    if (has_unet && has_second_text_encoder) {
        return VERSION_SDXL;
    }
    if (token_embedding_weight.ne[0] == 1024) {
        return VERSION_SD2;
    }
    if (token_embedding_weight.ne[0] == 768 || has_unet) {
        return VERSION_SD1;
    }
    return VERSION_COUNT;
}

static std::map<ggml_type, uint32_t> collect_wtype_stat(const String2TensorStorage& map, const std::function<bool(const std::string&)>& predicate) {
    std::map<ggml_type, uint32_t> stat;
    for (const auto& item : map) {
        const TensorStorage& tensor_storage = item.second;
        if (is_unused_tensor(tensor_storage.name) || !predicate(tensor_storage.name)) {
            continue;
        }
        const ggml_type effective_type = tensor_storage.expected_type != GGML_TYPE_COUNT
                                             ? tensor_storage.expected_type
                                             : tensor_storage.type;
        stat[effective_type]++;
    }
    return stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_wtype_stat() const {
    return collect_wtype_stat(tensor_storage_map_, [](const std::string&) { return true; });
}

std::map<ggml_type, uint32_t> ModelLoader::get_conditioner_wtype_stat() const {
    return collect_wtype_stat(tensor_storage_map_, [](const std::string& name) {
        return contains(name, "text_encoders") || contains(name, "cond_stage_model") ||
               contains(name, "te.") || contains(name, "te1.") || contains(name, "te2.") || contains(name, "te3.");
    });
}

std::map<ggml_type, uint32_t> ModelLoader::get_diffusion_model_wtype_stat() const {
    return collect_wtype_stat(tensor_storage_map_, [](const std::string& name) {
        return contains(name, "model.diffusion_model.") || contains(name, "unet.") || contains(name, "transformer.");
    });
}

std::map<ggml_type, uint32_t> ModelLoader::get_vae_wtype_stat() const {
    return collect_wtype_stat(tensor_storage_map_, [](const std::string& name) {
        return contains(name, "vae.") || contains(name, "first_stage_model");
    });
}

std::vector<std::string> ModelLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_storage_map_.size());

    for (const auto& item : tensor_storage_map_) {
        names.push_back(item.first);
    }

    return names;
}

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules) {
    TensorTypeRules result;
    for (const std::string& item : split_string(tensor_type_rules, ',')) {
        if (item.empty()) {
            continue;
        }
        const size_t pos = item.find('=');
        if (pos == std::string::npos) {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
            continue;
        }
        const std::string tensor_pattern = item.substr(0, pos);
        const std::string type_name = item.substr(pos + 1);
        ggml_type tensor_type = GGML_TYPE_COUNT;
        if (type_name == "f32") {
            tensor_type = GGML_TYPE_F32;
        } else {
            for (int i = 0; i < static_cast<int>(GGML_TYPE_COUNT); i++) {
                const ggml_type candidate = static_cast<ggml_type>(i);
                if (type_name == ggml_type_name(candidate)) {
                    tensor_type = candidate;
                    break;
                }
            }
        }
        if (tensor_type == GGML_TYPE_COUNT) {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
            continue;
        }
        result.emplace_back(tensor_pattern, tensor_type);
    }
    return result;
}

void ModelLoader::set_wtype_override(ggml_type wtype, std::string tensor_type_rules) {
    const TensorTypeRules map_rules = parse_tensor_type_rules(tensor_type_rules);

    // Precompile regex patterns once to avoid repeated compilation in the inner loop
    std::vector<std::pair<std::regex, ggml_type>> compiled_rules;
    compiled_rules.reserve(map_rules.size());
    for (const auto& rule : map_rules) {
        try {
            compiled_rules.emplace_back(std::regex(rule.first), rule.second);
        } catch (const std::regex_error& e) {
            LOG_WARN("invalid regex in tensor-type-rules: \"%s\" (%s)", rule.first.c_str(), e.what());
            continue;
        }
    }

    size_t converted = 0;
    for (auto& item : tensor_storage_map_) {
        ggml_type dst_type = wtype;
        for (const auto& compiled_rule : compiled_rules) {
            if (std::regex_search(item.first, compiled_rule.first)) {
                dst_type = compiled_rule.second;
                break;
            }
        }
        // Qwen-Image modulation projections (img_mod.1 / txt_mod.1): floor at q8_0.
        // On edit checkpoints (zero_cond_t) these feed a per-token where-select
        // mod_0 + index*(mod_1-mod_0); the subtraction amplifies k-quant error and
        // breaks edit instruction-following at q4_k (verified: q4 leaves the teapot
        // body glass, q8_0 restores full brushed metal, esp. few-step lightning).
        // Compare bits-per-weight (type_size is per-block bytes and misleads: q4_K
        // has a 256-elem block so its type_size 144 > q8_0's 34). Only lift a more
        // aggressive quant; leave f16/bf16/q8_0 and explicit --tensor-type-rules alone.
        if ((contains(item.first, "img_mod.1.") || contains(item.first, "txt_mod.1.")) &&
            dst_type != GGML_TYPE_COUNT && ggml_is_quantized(dst_type)) {
            const double dst_bpw = 8.0 * ggml_type_size(dst_type) / ggml_blck_size(dst_type);
            const double q8_bpw  = 8.0 * ggml_type_size(GGML_TYPE_Q8_0) / ggml_blck_size(GGML_TYPE_Q8_0);
            if (dst_bpw < q8_bpw) {
                dst_type = GGML_TYPE_Q8_0;
            }
        }
        if (dst_type != GGML_TYPE_COUNT && tensor_should_be_converted(item.second, dst_type)) {
            item.second.expected_type = dst_type;
            ++converted;
        }
    }
    LOG_INFO("set_wtype_override: wtype=%s rules='%s' marked %zu/%zu tensors for conversion",
             wtype == GGML_TYPE_COUNT ? "auto" : ggml_type_name(wtype),
             tensor_type_rules.c_str(),
             converted,
             tensor_storage_map_.size());
}

size_t ModelLoader::override_component_wtype(const std::string& prefix,
                                             ggml_type dst_type,
                                             bool allow_precision_increase) {
    if (dst_type == GGML_TYPE_COUNT) {
        return 0;
    }
    size_t changed = 0;
    for (auto& item : tensor_storage_map_) {
        if (item.first.rfind(prefix, 0) != 0) {
            continue;  // not in this component
        }
        const ggml_type current_type = item.second.expected_type != GGML_TYPE_COUNT
                                           ? item.second.expected_type
                                           : item.second.type;
        if (!allow_precision_increase && ggml_is_quantized(current_type)) {
            const double current_bpw = 8.0 * ggml_type_size(current_type) / ggml_blck_size(current_type);
            const double target_bpw  = 8.0 * ggml_type_size(dst_type) / ggml_blck_size(dst_type);
            if (target_bpw >= current_bpw) {
                continue;
            }
        }
        if (!tensor_should_be_converted(item.second, dst_type)) {
            continue;  // biases/norms/embeds etc. stay as-is (mirrors set_wtype_override)
        }
        if (item.second.expected_type != dst_type) {
            item.second.expected_type = dst_type;
            ++changed;
        }
    }
    return changed;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads_p, bool enable_mmap) {
    if (!on_new_tensor_cb) {
        set_error("load_tensors called without callback");
        return false;
    }

    const int num_threads_to_use = std::max(1, n_threads_p > 0 ? n_threads_p : ed_get_num_physical_cores());
    const int64_t start_time = ggml_time_ms();
    std::atomic<uint64_t> bytes_processed(0);
    size_t total_tensors_processed = 0;

    std::vector<TensorStorage> tensors;
    for (const auto& item : tensor_storage_map_) {
        if (!is_unused_tensor(item.second.name)) {
            tensors.push_back(item.second);
        }
    }

    for (size_t file_index = 0; file_index < file_paths_.size(); ++file_index) {
        std::vector<const TensorStorage*> file_tensors;
        for (const TensorStorage& tensor_storage : tensors) {
            if (tensor_storage.file_index == file_index) {
                file_tensors.push_back(&tensor_storage);
            }
        }
        if (file_tensors.empty()) {
            continue;
        }

        std::unique_ptr<MmapWrapper> mmapped;
        if (enable_mmap) {
            mmapped = MmapWrapper::create(file_paths_[file_index]);
        }

        std::atomic<size_t> tensor_idx(0);
        std::atomic<bool> failed(false);
        std::vector<std::thread> workers;
        const int n_threads = std::max(1, std::min(num_threads_to_use, static_cast<int>(file_tensors.size())));

        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([&, file_index]() {
                std::ifstream file;
                if (!mmapped) {
                    file.open(file_paths_[file_index], std::ios::binary);
                    if (!file.is_open()) {
                        failed = true;
                        return;
                    }
                }

                std::vector<uint8_t> read_buffer;
                std::vector<uint8_t> convert_buffer;

                while (!failed) {
                    const size_t idx = tensor_idx.fetch_add(1);
                    if (idx >= file_tensors.size()) {
                        break;
                    }

                    const TensorStorage& tensor_storage = *file_tensors[idx];
                    ggml_tensor* dst_tensor = nullptr;
                    if (!on_new_tensor_cb(tensor_storage, &dst_tensor)) {
                        failed = true;
                        break;
                    }
                    if (dst_tensor == nullptr) {
                        continue;
                    }

                    const size_t nbytes_to_read = static_cast<size_t>(tensor_storage.nbytes_to_read());
                    char* read_buf = nullptr;
                    char* target_buf = nullptr;
                    char* convert_buf = nullptr;

                    if (dst_tensor->buffer == nullptr || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        if (tensor_storage.type == dst_tensor->type && !tensor_storage.is_f64 && !tensor_storage.is_i64) {
                            read_buf = static_cast<char*>(dst_tensor->data);
                            target_buf = static_cast<char*>(dst_tensor->data);
                        } else {
                            read_buffer.resize(std::max(static_cast<size_t>(tensor_storage.nbytes()), nbytes_to_read));
                            read_buf = reinterpret_cast<char*>(read_buffer.data());
                            target_buf = read_buf;
                            convert_buf = static_cast<char*>(dst_tensor->data);
                        }
                    } else {
                        read_buffer.resize(std::max(static_cast<size_t>(tensor_storage.nbytes()), nbytes_to_read));
                        read_buf = reinterpret_cast<char*>(read_buffer.data());
                        target_buf = read_buf;
                        if (tensor_storage.type != dst_tensor->type) {
                            convert_buffer.resize(ggml_nbytes(dst_tensor));
                            convert_buf = reinterpret_cast<char*>(convert_buffer.data());
                        }
                    }

                    bool read_ok = false;
                    if (mmapped) {
                        read_ok = mmapped->copy_data(read_buf, nbytes_to_read, tensor_storage.offset);
                    } else {
                        file.seekg(static_cast<std::streamoff>(tensor_storage.offset));
                        file.read(read_buf, static_cast<std::streamsize>(nbytes_to_read));
                        read_ok = static_cast<bool>(file);
                    }
                    if (!read_ok) {
                        failed = true;
                        break;
                    }

                    if (tensor_storage.is_f8_e4m3) {
                        f8_e4m3_to_f16_vec(reinterpret_cast<uint8_t*>(read_buf), reinterpret_cast<uint16_t*>(target_buf), tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        f8_e5m2_to_f16_vec(reinterpret_cast<uint8_t*>(read_buf), reinterpret_cast<uint16_t*>(target_buf), tensor_storage.nelements());
                    } else if (tensor_storage.is_f64) {
                        f64_to_f32_vec(reinterpret_cast<double*>(read_buf), reinterpret_cast<float*>(target_buf), tensor_storage.nelements());
                    } else if (tensor_storage.is_i64) {
                        i64_to_i32_vec(reinterpret_cast<int64_t*>(read_buf), reinterpret_cast<int32_t*>(target_buf), tensor_storage.nelements());
                    }

                    if (tensor_storage.swap_swiglu_halves) {
                        const size_t total = static_cast<size_t>(tensor_storage.nbytes());
                        if (total % 2 == 0 && target_buf != nullptr) {
                            const size_t half = total / 2;
                            std::vector<uint8_t> tmp(target_buf, target_buf + half);
                            std::memmove(target_buf, target_buf + half, half);
                            std::memcpy(target_buf + half, tmp.data(), half);
                        } else {
                            LOG_WARN("tensor half swap skipped for '%s': odd byte size %zu",
                                     tensor_storage.name.c_str(), total);
                        }
                    }

                    if (tensor_storage.type != dst_tensor->type) {
                        if (convert_buf == nullptr) {
                            failed = true;
                            break;
                        }
                        // imatrix lookup (offline convert only; map is empty at
                        // runtime). tensor_storage.name is already canonical here, and
                        // imatrix_map_ is keyed by canonical name, so a direct lookup
                        // aligns. build_quant_imatrix() falls back to all-ones when the
                        // entry is missing or its length != ne[0], so this never breaks
                        // a tensor. Read-only concurrent access is safe (set once before).
                        const std::vector<float>* imatrix_override = nullptr;
                        if (!imatrix_map_.empty()) {
                            auto it = imatrix_map_.find(tensor_storage.name);
                            if (it != imatrix_map_.end()) {
                                imatrix_override = &it->second;
                            }
                        }
                        if (!convert_tensor_data(target_buf,
                                                 tensor_storage.type,
                                                 convert_buf,
                                                 dst_tensor->type,
                                                 static_cast<int>(tensor_storage.nelements() / tensor_storage.ne[0]),
                                                 static_cast<int>(tensor_storage.ne[0]),
                                                 imatrix_override)) {
                            failed = true;
                            break;
                        }
                    } else {
                        convert_buf = target_buf;
                    }

                    // diffusers FLUX final adaLN modulation is stored as [shift, scale]
                    // along the outermost (2*hidden) axis; the engine expects [scale,
                    // shift]. Swap the two contiguous byte halves. The split lands on a
                    // row boundary for both the weight ([in, 2*hidden]) and bias
                    // ([2*hidden]), so it is quantization-block-aligned for any dtype.
                    if (tensor_storage.swap_scale_shift) {
                        const size_t total = ggml_nbytes(dst_tensor);
                        if (total % 2 == 0 && convert_buf != nullptr) {
                            const size_t half = total / 2;
                            std::vector<uint8_t> tmp(convert_buf, convert_buf + half);
                            std::memmove(convert_buf, convert_buf + half, half);
                            std::memcpy(convert_buf + half, tmp.data(), half);
                        } else {
                            LOG_WARN("swap_scale_shift skipped for '%s': odd byte size %zu",
                                     tensor_storage.name.c_str(), total);
                        }
                    }

                    if (dst_tensor->buffer != nullptr && !ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        ggml_backend_tensor_set(dst_tensor, convert_buf, 0, ggml_nbytes(dst_tensor));
                    }

                    bytes_processed.fetch_add(nbytes_to_read);
                }
            });
        }

        while (tensor_idx.load() < file_tensors.size() && !failed) {
            const size_t curr = total_tensors_processed + std::min(tensor_idx.load(), file_tensors.size());
            pretty_bytes_progress(static_cast<int>(curr),
                                  static_cast<int>(tensors.size()),
                                  bytes_processed.load(),
                                  (ggml_time_ms() - start_time) / 1000.0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        for (std::thread& worker : workers) {
            worker.join();
        }
        if (failed) {
            set_error("failed to load tensor data from '" + file_paths_[file_index] + "'");
            return false;
        }
        total_tensors_processed += file_tensors.size();
    }

    LOG_INFO("loading tensor data completed in %.2fs", (ggml_time_ms() - start_time) / 1000.0f);
    return true;
}

bool ModelLoader::load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               std::set<std::string> ignore_tensors,
                               int n_threads,
                               bool enable_mmap) {
    std::set<std::string> tensor_names_in_file;
    std::mutex load_mutex;
    std::vector<std::unique_ptr<ggml_tensor>> tensor_views;

    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        std::lock_guard<std::mutex> lock(load_mutex);

        auto it = tensors.find(name);
        if (it != tensors.end()) {
            ggml_tensor* real = it->second;
            if (getenv("ED_DUMP_TENSOR_DTYPE")) {
                fprintf(stderr, "EDDUMP\t%s\tstorage=%s\tgraph=%s\tndims=%d\tne0=%lld\n",
                        name.c_str(), ggml_type_name(tensor_storage.type),
                        ggml_type_name(real->type), tensor_storage.n_dims,
                        (long long)tensor_storage.ne[0]);
            }
            if (!tensor_shape_matches_ggml(real, tensor_storage)) {
                int concat_dim = -1;
                if (!find_split_concat_dim(real, tensor_storage, &concat_dim)) {
                    LOG_ERROR("tensor '%s' has wrong shape in model file", name.c_str());
                    return false;
                }
            } else {
                tensor_names_in_file.insert(name);
                *dst_tensor = real;
                return true;
            }
        }

        std::string base_name;
        int chunk_index = 0;
        if (!split_tensor_chunk_base(name, &base_name, &chunk_index)) {
            base_name = name;
        }

        auto fused_it = tensors.find(base_name);
        if (fused_it == tensors.end()) {
            for (const std::string& ignore_tensor : ignore_tensors) {
                if (starts_with(name, ignore_tensor)) {
                    return true;
                }
            }
            LOG_INFO("unknown tensor '%s' in model file", tensor_storage.to_string().c_str());
            return true;
        }

        ggml_tensor* fused = fused_it->second;
        int concat_dim = -1;
        if (!find_split_concat_dim(fused, tensor_storage, &concat_dim)) {
            LOG_ERROR("tensor '%s' cannot be loaded into '%s': incompatible split shape",
                      name.c_str(),
                      base_name.c_str());
            return false;
        }

        const int64_t offset_elems = split_chunk_offset_elems(tensor_storage_map_,
                                                              base_name,
                                                              chunk_index,
                                                              concat_dim);
        if (offset_elems < 0 || offset_elems + tensor_storage.ne[concat_dim] > fused->ne[concat_dim]) {
            LOG_ERROR("tensor '%s' cannot be loaded into '%s': invalid split offset",
                      name.c_str(),
                      base_name.c_str());
            return false;
        }

        if (fused->data == nullptr) {
            LOG_ERROR("tensor '%s' cannot be loaded into '%s': destination has no data buffer",
                      name.c_str(),
                      base_name.c_str());
            return false;
        }

        tensor_views.emplace_back(new ggml_tensor(*fused));
        ggml_tensor* view = tensor_views.back().get();
        for (int i = 0; i < 4; ++i) {
            view->ne[i] = tensor_storage.ne[i];
        }

        size_t byte_offset = 0;
        if (concat_dim == 0) {
            byte_offset = static_cast<size_t>(offset_elems) * ggml_type_size(fused->type) / ggml_blck_size(fused->type);
        } else {
            byte_offset = static_cast<size_t>(offset_elems) * static_cast<size_t>(fused->nb[concat_dim]);
        }
        view->data = static_cast<char*>(fused->data) + byte_offset;

        tensor_names_in_file.insert(base_name);
        *dst_tensor = view;
        return true;
    };

    if (!load_tensors(on_new_tensor_cb, n_threads, enable_mmap)) {
        return false;
    }

    for (const auto& item : tensors) {
        if (starts_with(item.first, "__ed_")) {
            continue;
        }
        bool ignored = false;
        for (const std::string& ignore_tensor : ignore_tensors) {
            if (starts_with(item.first, ignore_tensor)) {
                ignored = true;
                break;
            }
        }
        if (ignored) {
            continue;
        }
        if (tensor_names_in_file.find(item.first) == tensor_names_in_file.end()) {
            LOG_ERROR("tensor '%s' not in model file", item.first.c_str());
            return false;
        }
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) const {
    const std::string& name = tensor_storage.name;
    if (type == GGML_TYPE_COUNT) {
        return false;
    }
    if (ggml_is_quantized(type) && tensor_storage.ne[0] % ggml_blck_size(type) != 0) {
        return false;
    }
    if (ends_with(name, ".bias") || ends_with(name, ".scale") || contains(name, "embedding")) {
        return false;
    }
    // LTX modulation tables are consumed directly by repeat/broadcast and must
    // remain floating point; quantizing them makes the CUDA broadcast kernel reject
    // the graph even though the surrounding Linear weights support Q8/Q4.
    if (contains(name, "scale_shift_table")) {
        return false;
    }
    if (contains(name, "img_in.") || contains(name, "txt_in.") || contains(name, "time_in.") ||
        contains(name, "vector_in.") || contains(name, "guidance_in.") || contains(name, "final_layer.")) {
        return false;
    }
    // SD3/MMDiT conditioning-injection + positional layers run at F32 at runtime
    // (Linear force_f32 / PatchEmbed pos_embed). The engine graph overrides the
    // stored type, but a pre-quantized GGUF has no graph, so these must not be
    // quantized on disk or the GGUF diverges from on-the-fly (see H Bug#1).
    if (contains(name, "context_embedder") || contains(name, "t_embedder.") ||
        contains(name, "y_embedder.") || contains(name, "x_embedder.") ||
        contains(name, "pos_embed")) {
        return false;
    }
    // Text-encoder projections and T5 shared embedding stay float at runtime
    // (CLIPTextModel::init_params uses the source type; Embedding::init_params
    // falls back to F32 for get_rows). convert has no graph, so exclude by name.
    if (ends_with(name, ".text_projection") || ends_with(name, ".shared.weight")) {
        return false;
    }
    return true;
}

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend, ggml_type type) const {
    size_t alignment = 128;
    if (backend != nullptr) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;
    for (const auto& item : tensor_storage_map_) {
        TensorStorage tensor_storage = item.second;
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_storage.type = type;
        }
        mem_size += tensor_storage.nbytes() + static_cast<int64_t>(alignment);
    }
    return mem_size;
}
