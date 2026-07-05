#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

namespace {

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string normalize_token(std::string value) {
    value = lower_ascii(std::move(value));
    for (char& c : value) {
        if (c == '_' || c == '.' || c == ' ') {
            c = '-';
        }
    }
    return value;
}

bool json_get_bool(const json& obj, const char* key, bool fallback) {
    if (!obj.contains(key)) {
        return fallback;
    }
    const json& value = obj.at(key);
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int>() != 0;
    }
    if (value.is_string()) {
        const std::string text = normalize_token(value.get<std::string>());
        return text == "1" || text == "true" || text == "yes" || text == "on";
    }
    return fallback;
}

template <typename T>
T json_get_number(const json& obj, const char* key, T fallback) {
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return fallback;
    }
    try {
        return obj.at(key).get<T>();
    } catch (...) {
        return fallback;
    }
}

std::string json_get_string(const json& obj, const char* key, const std::string& fallback = "") {
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return fallback;
    }
    if (obj.at(key).is_string()) {
        return obj.at(key).get<std::string>();
    }
    return fallback;
}

const json* cache_object(const json& body) {
    if (body.contains("cache") && body.at("cache").is_object()) {
        return &body.at("cache");
    }
    return nullptr;
}

template <typename T>
T get_cache_number(const json& body, const json* cache, const char* short_key, const char* full_key, T fallback) {
    if (cache != nullptr && cache->contains(short_key)) {
        return json_get_number(*cache, short_key, fallback);
    }
    return json_get_number(body, full_key, fallback);
}

bool get_cache_bool(const json& body, const json* cache, const char* short_key, const char* full_key, bool fallback) {
    if (cache != nullptr && cache->contains(short_key)) {
        return json_get_bool(*cache, short_key, fallback);
    }
    return json_get_bool(body, full_key, fallback);
}

std::string get_cache_string(const json& body,
                             const json* cache,
                             const char* short_key,
                             const char* full_key,
                             const std::string& fallback = "") {
    if (cache != nullptr && cache->contains(short_key)) {
        return json_get_string(*cache, short_key, fallback);
    }
    return json_get_string(body, full_key, fallback);
}

void append_png_bytes(void* context, void* data, int size) {
    if (context == nullptr || data == nullptr || size <= 0) {
        return;
    }
    auto* out = static_cast<std::vector<uint8_t>*>(context);
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

bool validate_image_params(const ed_image_generation_params_t& params, std::string* error) {
    if (params.prompt == nullptr || std::strlen(params.prompt) == 0) {
        if (error != nullptr) {
            *error = "prompt is required";
        }
        return false;
    }
    if (params.width <= 0 || params.height <= 0) {
        if (error != nullptr) {
            *error = "width and height must be positive";
        }
        return false;
    }
    if (params.sample.steps <= 0) {
        if (error != nullptr) {
            *error = "steps must be positive";
        }
        return false;
    }
    if (params.batch_count <= 0) {
        if (error != nullptr) {
            *error = "batch_count must be positive";
        }
        return false;
    }
    if (params.sample.cache_start_percent < 0.0f ||
        params.sample.cache_start_percent > 1.0f ||
        params.sample.cache_end_percent < 0.0f ||
        params.sample.cache_end_percent > 1.0f ||
        params.sample.cache_start_percent >= params.sample.cache_end_percent) {
        if (error != nullptr) {
            *error = "cache window must satisfy 0 <= cache_start_percent < cache_end_percent <= 1";
        }
        return false;
    }
    if (params.sample.cache_reuse_threshold < 0.0f && !std::isinf(params.sample.cache_reuse_threshold)) {
        if (error != nullptr) {
            *error = "cache_reuse_threshold must be non-negative";
        }
        return false;
    }
    if (params.sample.cache_error_decay_rate < 0.0f || params.sample.cache_error_decay_rate > 1.0f) {
        if (error != nullptr) {
            *error = "cache_error_decay_rate must be in [0, 1]";
        }
        return false;
    }
    if (params.sample.cache_Fn_compute_blocks < 0 || params.sample.cache_Bn_compute_blocks < 0) {
        if (error != nullptr) {
            *error = "cache_Fn_compute_blocks and cache_Bn_compute_blocks must be non-negative";
        }
        return false;
    }
    if (params.sample.cache_residual_diff_threshold < 0.0f) {
        if (error != nullptr) {
            *error = "cache_residual_diff_threshold must be non-negative";
        }
        return false;
    }
    if (params.sample.cache_max_accumulated_residual_diff < -1.0f) {
        if (error != nullptr) {
            *error = "cache_max_accumulated_residual_diff must be >= -1";
        }
        return false;
    }
    if (params.sample.cache_max_warmup_steps < 0) {
        if (error != nullptr) {
            *error = "cache_max_warmup_steps must be non-negative";
        }
        return false;
    }
    if (params.sample.cache_taylorseer_n_derivatives < 1) {
        if (error != nullptr) {
            *error = "cache_taylorseer_n_derivatives must be >= 1";
        }
        return false;
    }
    if (params.sample.cache_taylorseer_skip_interval < 0) {
        if (error != nullptr) {
            *error = "cache_taylorseer_skip_interval must be non-negative";
        }
        return false;
    }
    return true;
}

}  // namespace

std::string ed_status_to_string(ed_status_t status) {
    switch (status) {
        case ED_STATUS_OK: return "ok";
        case ED_STATUS_ERROR: return "error";
        case ED_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case ED_STATUS_MODEL_LOAD_FAILED: return "model_load_failed";
        case ED_STATUS_GENERATION_FAILED: return "generation_failed";
        case ED_STATUS_OUT_OF_MEMORY: return "out_of_memory";
        case ED_STATUS_UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}

std::string ed_cache_mode_to_string(ed_cache_mode_t mode) {
    switch (mode) {
        case ED_CACHE_DISABLED: return "disabled";
        case ED_CACHE_EASYCACHE: return "easycache";
        case ED_CACHE_UCACHE: return "ucache";
        case ED_CACHE_DBCACHE: return "dbcache";
        case ED_CACHE_TAYLORSEER: return "taylorseer";
        case ED_CACHE_CACHE_DIT: return "cache-dit";
    }
    return "disabled";
}

bool ed_cache_mode_from_string(const std::string& text, ed_cache_mode_t* mode) {
    const std::string value = normalize_token(text);
    if (value == "off" || value == "none" || value == "disabled" || value == "disable" || value == "0") {
        if (mode != nullptr) {
            *mode = ED_CACHE_DISABLED;
        }
        return true;
    }
    if (value == "easycache" || value == "easy") {
        if (mode != nullptr) {
            *mode = ED_CACHE_EASYCACHE;
        }
        return true;
    }
    if (value == "ucache" || value == "u") {
        if (mode != nullptr) {
            *mode = ED_CACHE_UCACHE;
        }
        return true;
    }
    if (value == "dbcache" || value == "db") {
        if (mode != nullptr) {
            *mode = ED_CACHE_DBCACHE;
        }
        return true;
    }
    if (value == "taylorseer" || value == "taylor-seer" || value == "taylor") {
        if (mode != nullptr) {
            *mode = ED_CACHE_TAYLORSEER;
        }
        return true;
    }
    if (value == "cache-dit" || value == "cachedit") {
        if (mode != nullptr) {
            *mode = ED_CACHE_CACHE_DIT;
        }
        return true;
    }
    return false;
}

bool ed_sampler_from_string(const std::string& text, ed_sampler_t* sampler) {
    const std::string value = normalize_token(text);
    struct Entry {
        const char* name;
        ed_sampler_t sampler;
    };
    static const Entry entries[] = {
        {"auto", ED_SAMPLER_AUTO},
        {"euler", ED_SAMPLER_EULER},
        {"euler-a", ED_SAMPLER_EULER_A},
        {"heun", ED_SAMPLER_HEUN},
        {"dpm2", ED_SAMPLER_DPM2},
        {"dpm-plus-plus-2s-a", ED_SAMPLER_DPM_PLUS_PLUS_2S_A},
        {"dpm++-2s-a", ED_SAMPLER_DPM_PLUS_PLUS_2S_A},
        {"dpm-plus-plus-2m", ED_SAMPLER_DPM_PLUS_PLUS_2M},
        {"dpm++-2m", ED_SAMPLER_DPM_PLUS_PLUS_2M},
        {"dpm-plus-plus-2m-v2", ED_SAMPLER_DPM_PLUS_PLUS_2M_V2},
        {"dpm++-2m-v2", ED_SAMPLER_DPM_PLUS_PLUS_2M_V2},
        {"ipndm", ED_SAMPLER_IPNDM},
        {"ipndm-v", ED_SAMPLER_IPNDM_V},
        {"lcm", ED_SAMPLER_LCM},
        {"ddim-trailing", ED_SAMPLER_DDIM_TRAILING},
        {"ddim", ED_SAMPLER_DDIM_TRAILING},
        {"tcd", ED_SAMPLER_TCD},
        {"res-multistep", ED_SAMPLER_RES_MULTISTEP},
        {"res-2s", ED_SAMPLER_RES_2S},
        {"er-sde", ED_SAMPLER_ER_SDE},
    };
    for (const Entry& entry : entries) {
        if (value == entry.name) {
            if (sampler != nullptr) {
                *sampler = entry.sampler;
            }
            return true;
        }
    }
    return false;
}

bool ed_scheduler_from_string(const std::string& text, ed_scheduler_t* scheduler) {
    const std::string value = normalize_token(text);
    struct Entry {
        const char* name;
        ed_scheduler_t scheduler;
    };
    static const Entry entries[] = {
        {"auto", ED_SCHEDULER_AUTO},
        {"discrete", ED_SCHEDULER_DISCRETE},
        {"karras", ED_SCHEDULER_KARRAS},
        {"exponential", ED_SCHEDULER_EXPONENTIAL},
        {"ays", ED_SCHEDULER_AYS},
        {"gits", ED_SCHEDULER_GITS},
        {"sgm-uniform", ED_SCHEDULER_SGM_UNIFORM},
        {"simple", ED_SCHEDULER_SIMPLE},
        {"smoothstep", ED_SCHEDULER_SMOOTHSTEP},
        {"kl-optimal", ED_SCHEDULER_KL_OPTIMAL},
        {"lcm", ED_SCHEDULER_LCM},
        {"bong-tangent", ED_SCHEDULER_BONG_TANGENT},
    };
    for (const Entry& entry : entries) {
        if (value == entry.name) {
            if (scheduler != nullptr) {
                *scheduler = entry.scheduler;
            }
            return true;
        }
    }
    return false;
}

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((bytes.size() + 2) / 3) * 4);

    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t b0 = bytes[i];
        const uint32_t b1 = i + 1 < bytes.size() ? bytes[i + 1] : 0;
        const uint32_t b2 = i + 2 < bytes.size() ? bytes[i + 2] : 0;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(table[(triple >> 18) & 0x3f]);
        out.push_back(table[(triple >> 12) & 0x3f]);
        out.push_back(i + 1 < bytes.size() ? table[(triple >> 6) & 0x3f] : '=');
        out.push_back(i + 2 < bytes.size() ? table[triple & 0x3f] : '=');
    }

    return out;
}

bool image_to_png_bytes(const ed_image_t& image, std::vector<uint8_t>* bytes) {
    if (bytes == nullptr || image.data == nullptr || image.width == 0 || image.height == 0) {
        return false;
    }
    if (image.channels == 0 || image.channels > 4) {
        return false;
    }

    bytes->clear();
    return stbi_write_png_to_func(append_png_bytes,
                                  bytes,
                                  static_cast<int>(image.width),
                                  static_cast<int>(image.height),
                                  static_cast<int>(image.channels),
                                  image.data,
                                  0) != 0;
}

bool build_image_request(const json& body,
                         const EdgeDitServerRuntime& runtime,
                         EdgeDitImageRequest* request,
                         std::string* error) {
    if (request == nullptr) {
        if (error != nullptr) {
            *error = "internal error: null request";
        }
        return false;
    }

    *request = {};
    ed_image_generation_params_init(&request->params);

    request->prompt = json_get_string(body, "prompt");
    const bool has_negative_prompt = body.contains("negative_prompt") && !body.at("negative_prompt").is_null();
    request->negative_prompt = json_get_string(body, "negative_prompt");

    request->params.prompt = request->prompt.c_str();
    request->params.negative_prompt = has_negative_prompt ? request->negative_prompt.c_str() : nullptr;
    request->params.width = json_get_number(body, "width", runtime.defaults->width);
    request->params.height = json_get_number(body, "height", runtime.defaults->height);
    request->params.seed = json_get_number<int64_t>(body, "seed", runtime.defaults->seed);
    request->params.batch_count = json_get_number(body, "batch_count", json_get_number(body, "batch_size", 1));
    request->params.strength = json_get_number(body, "strength", request->params.strength);
    request->params.control_strength = json_get_number(body, "control_strength", request->params.control_strength);

    request->params.sample.sampler = runtime.defaults->sampler;
    request->params.sample.scheduler = runtime.defaults->scheduler;
    request->params.sample.steps = json_get_number(body, "steps", runtime.defaults->steps);
    request->params.sample.cfg_scale = json_get_number(body, "cfg_scale", runtime.defaults->cfg_scale);
    request->params.sample.image_cfg_scale =
        json_get_number(body, "image_cfg_scale", runtime.defaults->image_cfg_scale);
    request->params.sample.distilled_guidance =
        json_get_number(body, "distilled_guidance", runtime.defaults->distilled_guidance);
    request->params.sample.flow_shift = json_get_number(body, "flow_shift", runtime.defaults->flow_shift);
    request->params.sample.cache_mode = runtime.defaults->cache_mode;

    const std::string sampler_name = json_get_string(body, "sampler");
    if (!sampler_name.empty() && !ed_sampler_from_string(sampler_name, &request->params.sample.sampler)) {
        if (error != nullptr) {
            *error = "unsupported sampler: " + sampler_name;
        }
        return false;
    }

    const std::string scheduler_name = json_get_string(body, "scheduler");
    if (!scheduler_name.empty() && !ed_scheduler_from_string(scheduler_name, &request->params.sample.scheduler)) {
        if (error != nullptr) {
            *error = "unsupported scheduler: " + scheduler_name;
        }
        return false;
    }

    const json* cache = cache_object(body);
    std::string cache_mode = get_cache_string(body, cache, "mode", "cache_mode");
    if (!cache_mode.empty() && !ed_cache_mode_from_string(cache_mode, &request->params.sample.cache_mode)) {
        if (error != nullptr) {
            *error = "unsupported cache_mode: " + cache_mode;
        }
        return false;
    }

    request->params.sample.cache_reuse_threshold =
        get_cache_number(body, cache, "reuse_threshold", "cache_reuse_threshold",
                         get_cache_number(body, cache, "threshold", "cache_threshold",
                                          request->params.sample.cache_reuse_threshold));
    request->params.sample.cache_start_percent =
        get_cache_number(body, cache, "start_percent", "cache_start_percent",
                         get_cache_number(body, cache, "start", "cache_start",
                                          request->params.sample.cache_start_percent));
    request->params.sample.cache_end_percent =
        get_cache_number(body, cache, "end_percent", "cache_end_percent",
                         get_cache_number(body, cache, "end", "cache_end",
                                          request->params.sample.cache_end_percent));
    request->params.sample.cache_error_decay_rate =
        get_cache_number(body, cache, "error_decay_rate", "cache_error_decay_rate",
                         get_cache_number(body, cache, "error_decay", "cache_error_decay",
                                          request->params.sample.cache_error_decay_rate));
    request->params.sample.cache_use_relative_threshold =
        get_cache_bool(body, cache, "use_relative_threshold", "cache_use_relative_threshold",
                       request->params.sample.cache_use_relative_threshold);
    request->params.sample.cache_reset_error_on_compute =
        get_cache_bool(body, cache, "reset_error_on_compute", "cache_reset_error_on_compute",
                       request->params.sample.cache_reset_error_on_compute);
    request->params.sample.cache_Fn_compute_blocks =
        get_cache_number(body, cache, "Fn_compute_blocks", "cache_Fn_compute_blocks",
                         get_cache_number(body, cache, "fn_blocks", "cache_fn_blocks",
                                          request->params.sample.cache_Fn_compute_blocks));
    request->params.sample.cache_Bn_compute_blocks =
        get_cache_number(body, cache, "Bn_compute_blocks", "cache_Bn_compute_blocks",
                         get_cache_number(body, cache, "bn_blocks", "cache_bn_blocks",
                                          request->params.sample.cache_Bn_compute_blocks));
    request->params.sample.cache_residual_diff_threshold =
        get_cache_number(body, cache, "residual_diff_threshold", "cache_residual_diff_threshold",
                         get_cache_number(body, cache, "residual_threshold", "cache_residual_threshold",
                                          request->params.sample.cache_residual_diff_threshold));
    request->params.sample.cache_max_accumulated_residual_diff =
        get_cache_number(body, cache, "max_accumulated_residual_diff",
                         "cache_max_accumulated_residual_diff",
                         request->params.sample.cache_max_accumulated_residual_diff);
    request->params.sample.cache_max_warmup_steps =
        get_cache_number(body, cache, "max_warmup_steps", "cache_max_warmup_steps",
                         get_cache_number(body, cache, "warmup_steps", "cache_warmup_steps",
                                          request->params.sample.cache_max_warmup_steps));
    request->params.sample.cache_max_cached_steps =
        get_cache_number(body, cache, "max_cached_steps", "cache_max_cached_steps",
                         request->params.sample.cache_max_cached_steps);
    request->params.sample.cache_max_continuous_cached_steps =
        get_cache_number(body, cache, "max_continuous_cached_steps", "cache_max_continuous_cached_steps",
                         request->params.sample.cache_max_continuous_cached_steps);
    request->params.sample.cache_taylorseer_n_derivatives =
        get_cache_number(body, cache, "taylorseer_n_derivatives", "cache_taylorseer_n_derivatives",
                         get_cache_number(body, cache, "taylor_order", "cache_taylor_order",
                                          request->params.sample.cache_taylorseer_n_derivatives));
    request->params.sample.cache_taylorseer_skip_interval =
        get_cache_number(body, cache, "taylorseer_skip_interval", "cache_taylorseer_skip_interval",
                         get_cache_number(body, cache, "taylor_skip", "cache_taylor_skip",
                                          request->params.sample.cache_taylorseer_skip_interval));
    request->cache_scm_mask = get_cache_string(body, cache, "scm_mask", "cache_scm_mask");
    request->params.sample.cache_scm_mask = request->cache_scm_mask.empty() ? nullptr : request->cache_scm_mask.c_str();
    request->params.sample.cache_scm_policy_dynamic =
        get_cache_bool(body, cache, "scm_policy_dynamic", "cache_scm_policy_dynamic",
                       request->params.sample.cache_scm_policy_dynamic);
    if (cache != nullptr && cache->contains("static_scm")) {
        request->params.sample.cache_scm_policy_dynamic = !json_get_bool(*cache, "static_scm", false);
    }
    if (body.contains("cache_static_scm")) {
        request->params.sample.cache_scm_policy_dynamic = !json_get_bool(body, "cache_static_scm", false);
    }

    return validate_image_params(request->params, error);
}

json build_capabilities_response(const EdgeDitServerRuntime& runtime) {
    json result;
    result["service"] = "edge-dit";
    result["model"] = runtime.display_model_path;
    result["endpoints"] = {
        "/ed/v1/health",
        "/ed/v1/models",
        "/ed/v1/capabilities",
        "/ed/v1/images/generations",
    };
    result["aliases"] = {
        "/edgedit/v1",
        "/edge-dit/v1",
    };
    result["cache_modes"] = {
        "disabled",
        "easycache",
        "ucache",
        "dbcache",
        "taylorseer",
        "cache-dit",
    };
    result["samplers"] = {
        "auto",
        "euler",
        "euler-a",
        "heun",
        "dpm2",
        "dpm++-2s-a",
        "dpm++-2m",
        "dpm++-2m-v2",
        "ipndm",
        "ipndm-v",
        "lcm",
        "ddim-trailing",
        "tcd",
        "res-multistep",
        "res-2s",
        "er-sde",
    };
    result["schedulers"] = {
        "auto",
        "discrete",
        "karras",
        "exponential",
        "ays",
        "gits",
        "sgm-uniform",
        "simple",
        "smoothstep",
        "kl-optimal",
        "lcm",
        "bong-tangent",
    };
    result["defaults"] = {
        {"width", runtime.defaults->width},
        {"height", runtime.defaults->height},
        {"steps", runtime.defaults->steps},
        {"seed", runtime.defaults->seed},
        {"cfg_scale", runtime.defaults->cfg_scale},
        {"image_cfg_scale", runtime.defaults->image_cfg_scale},
        {"distilled_guidance", runtime.defaults->distilled_guidance},
        {"flow_shift", runtime.defaults->flow_shift},
        {"cache_mode", ed_cache_mode_to_string(runtime.defaults->cache_mode)},
    };
    return result;
}
