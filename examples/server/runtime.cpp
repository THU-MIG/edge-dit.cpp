#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "ggml/examples/stb_image.h"
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

bool base64_decode(const std::string& text, std::vector<uint8_t>* out) {
    static const std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int value=0,bits=-8; out->clear(); std::string payload=text; const size_t comma=payload.find(','); if(payload.rfind("data:",0)==0&&comma!=std::string::npos) payload=payload.substr(comma+1);
    for(unsigned char c:payload){ if(std::isspace(c))continue; if(c=='=')break; const size_t index=alphabet.find(c); if(index==std::string::npos)return false; value=(value<<6)|static_cast<int>(index); bits+=6; if(bits>=0){out->push_back(static_cast<uint8_t>((value>>bits)&0xff));bits-=8;} } return !out->empty();
}

bool decode_image(const std::string& encoded, std::vector<uint8_t>* pixels, ed_image_t* image) {
    std::vector<uint8_t> bytes; if(!base64_decode(encoded,&bytes))return false; int width=0,height=0,channels=0; unsigned char* raw=stbi_load_from_memory(bytes.data(),static_cast<int>(bytes.size()),&width,&height,&channels,3); if(!raw||width<=0||height<=0){if(raw)stbi_image_free(raw);return false;}
    pixels->assign(raw,raw+static_cast<size_t>(width)*height*3); stbi_image_free(raw); *image={static_cast<uint32_t>(width),static_cast<uint32_t>(height),3,pixels->data()}; return true;
}

bool decode_audio(const json& value, std::vector<float>* samples, ed_audio_t* audio) {
    if (!value.is_object() || !value.contains("b64_f32le") || !value.at("b64_f32le").is_string()) return false;
    const int sample_rate=json_get_number(value,"sample_rate",0), channels=json_get_number(value,"channels",0); std::vector<uint8_t> bytes;
    if(sample_rate<=0||channels<=0||!base64_decode(value.at("b64_f32le").get<std::string>(),&bytes)||bytes.size()%sizeof(float)!=0)return false;
    samples->resize(bytes.size()/sizeof(float)); std::memcpy(samples->data(),bytes.data(),bytes.size()); if(samples->empty()||samples->size()%channels!=0)return false;
    *audio={static_cast<uint32_t>(sample_rate),static_cast<uint32_t>(channels),samples->size()/channels,samples->data()}; return true;
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
    if (std::isfinite(params.sample.cache_residual_diff_threshold) &&
        params.sample.cache_residual_diff_threshold < 0.0f) {
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

bool validate_video_params(const ed_video_generation_params_t& params, std::string* error) {
    if (params.prompt == nullptr || std::strlen(params.prompt) == 0) { if (error) *error = "prompt is required"; return false; }
    if (params.width <= 0 || params.height <= 0 || params.frames <= 0 || params.fps <= 0) { if (error) *error = "width, height, frames, and fps must be positive"; return false; }
    if (params.sample.steps <= 0) { if (error) *error = "steps must be positive"; return false; }
    if (params.hires_enabled && params.hires_steps <= 0 && params.hires_sigmas_count == 0) { if (error) *error = "hires_steps must be positive"; return false; }
    if (params.hires_enabled && (!std::isfinite(params.hires_denoising_strength) || params.hires_denoising_strength <= 0.0f || params.hires_denoising_strength > 1.0f)) { if (error) *error = "hires_denoising_strength must be in (0, 1]"; return false; }
    if (params.hires_sigmas_count > 0) {
        if (params.hires_sigmas == nullptr || params.hires_sigmas_count < 2) {
            if (error) *error = "hires_sigmas must contain at least two values";
            return false;
        }
        for (int i = 0; i < params.hires_sigmas_count; ++i) {
            const float sigma = params.hires_sigmas[i];
            if (!std::isfinite(sigma) || sigma < 0.0f ||
                (i > 0 && sigma > params.hires_sigmas[i - 1])) {
                if (error) *error = "hires_sigmas must be finite, non-negative, and non-increasing";
                return false;
            }
        }
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
        case ED_STATUS_CANCELLED: return "cancelled";
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
        case ED_CACHE_MAGCACHE: return "magcache";
        case ED_CACHE_DICACHE: return "dicache";
        case ED_CACHE_SENCACHE: return "sencache";
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
    if (value == "magcache" || value == "mag") {
        if (mode != nullptr) {
            *mode = ED_CACHE_MAGCACHE;
        }
        return true;
    }
    if (value == "dicache" || value == "di") {
        if (mode != nullptr) {
            *mode = ED_CACHE_DICACHE;
        }
        return true;
    }
    if (value == "sencache" || value == "sen") {
        if (mode != nullptr) *mode = ED_CACHE_SENCACHE;
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
        {"ltx2", ED_SCHEDULER_LTX2},
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

bool build_video_request(const json& body,
                         const EdgeDitServerRuntime& runtime,
                         EdgeDitVideoRequest* request,
                         std::string* error) {
    if (request == nullptr) { if (error) *error = "internal error: null request"; return false; }
    *request = {}; ed_video_generation_params_init(&request->params);
    request->prompt = json_get_string(body, "prompt");
    request->negative_prompt = json_get_string(body, "negative_prompt");
    request->params.prompt = request->prompt.c_str();
    request->params.negative_prompt = request->negative_prompt.empty() ? nullptr : request->negative_prompt.c_str();
    request->params.width = json_get_number(body, "width", runtime.defaults->width);
    request->params.height = json_get_number(body, "height", runtime.defaults->height);
    request->params.frames = json_get_number(body, "frames", runtime.defaults->frames);
    request->params.fps = json_get_number(body, "fps", runtime.defaults->fps);
    request->params.seed = json_get_number<int64_t>(body, "seed", runtime.defaults->seed);
    request->params.hires_enabled = json_get_bool(body, "hires", false);
    request->params.hires_steps = json_get_number(body, "hires_steps", request->params.hires_steps);
    request->params.hires_denoising_strength = json_get_number(body, "hires_denoising_strength", request->params.hires_denoising_strength);
    if (body.contains("hires_sigmas")) {
        if (!body.at("hires_sigmas").is_array() || body.at("hires_sigmas").size() < 2) { if(error)*error="hires_sigmas must be an array with at least two values"; return false; }
        for (const auto& value : body.at("hires_sigmas")) {
            if (!value.is_number()) { if(error)*error="hires_sigmas must contain only numbers"; return false; }
            request->hires_sigmas.push_back(value.get<float>());
        }
        request->params.hires_sigmas = request->hires_sigmas.data();
        request->params.hires_sigmas_count = static_cast<int>(request->hires_sigmas.size());
    }
    size_t image_count=0; if(body.contains("init_image_b64"))++image_count; if(body.contains("end_image_b64"))++image_count; if(body.contains("ref_images_b64")&&body.at("ref_images_b64").is_array())image_count+=body.at("ref_images_b64").size();
    if(body.contains("ref_videos")){if(!body.at("ref_videos").is_array()){if(error)*error="ref_videos must be an array";return false;}for(const auto& video:body.at("ref_videos")){if(!video.is_object()||!video.contains("frames_b64")||!video.at("frames_b64").is_array()){if(error)*error="each ref_videos entry needs frames_b64";return false;}image_count+=video.at("frames_b64").size();}}
    request->image_storage.resize(image_count); size_t image_index=0;
    if(body.contains("init_image_b64")){ if(!body.at("init_image_b64").is_string()||!decode_image(body.at("init_image_b64").get<std::string>(),&request->image_storage[image_index++],&request->init_image)){if(error)*error="init_image_b64 is not a valid base64 image";return false;} request->params.init_image=&request->init_image; }
    if(body.contains("end_image_b64")){ if(!body.at("end_image_b64").is_string()||!decode_image(body.at("end_image_b64").get<std::string>(),&request->image_storage[image_index++],&request->end_image)){if(error)*error="end_image_b64 is not a valid base64 image";return false;} request->params.end_image=&request->end_image; }
    if(body.contains("ref_images_b64")){ if(!body.at("ref_images_b64").is_array()){if(error)*error="ref_images_b64 must be an array";return false;} request->ref_images.resize(body.at("ref_images_b64").size()); for(size_t i=0;i<request->ref_images.size();++i){const auto& value=body.at("ref_images_b64").at(i);if(!value.is_string()||!decode_image(value.get<std::string>(),&request->image_storage[image_index++],&request->ref_images[i])){if(error)*error="ref_images_b64 contains an invalid image";return false;}} request->params.ref_images=request->ref_images.data();request->params.ref_image_count=static_cast<int>(request->ref_images.size()); }
    const std::string ref_size=json_get_string(body,"ref_image_size","max"); if(ref_size=="match")request->params.ref_image_size=ED_REF_IMAGE_SIZE_MATCH;else if(ref_size!="max"){if(error)*error="ref_image_size must be max or match";return false;}
    size_t audio_count=body.contains("ref_audios")&&body.at("ref_audios").is_array()?body.at("ref_audios").size():0; if(body.contains("ref_videos"))for(const auto& video:body.at("ref_videos"))if(video.contains("audio"))++audio_count; request->audio_storage.resize(audio_count); size_t audio_index=0;
    if(body.contains("ref_videos")){request->ref_video_frames.resize(body.at("ref_videos").size());request->ref_videos.resize(body.at("ref_videos").size());for(size_t i=0;i<request->ref_videos.size();++i){const auto& value=body.at("ref_videos").at(i);const auto& encoded=value.at("frames_b64");auto& frames=request->ref_video_frames[i];frames.resize(encoded.size());for(size_t j=0;j<frames.size();++j)if(!encoded.at(j).is_string()||!decode_image(encoded.at(j).get<std::string>(),&request->image_storage[image_index++],&frames[j])){if(error)*error="ref_videos contains an invalid frame";return false;}auto& video=request->ref_videos[i];video.frames=frames.data();video.frame_count=static_cast<int>(frames.size());video.fps=json_get_number(value,"fps",runtime.defaults->fps);if(video.fps<=0){if(error)*error="reference video fps must be positive";return false;}if(value.contains("audio")&&!decode_audio(value.at("audio"),&request->audio_storage[audio_index++],&video.audio)){if(error)*error="reference video audio must be b64_f32le with sample_rate/channels";return false;}}request->params.ref_videos=request->ref_videos.data();request->params.ref_video_count=static_cast<int>(request->ref_videos.size());}
    if(body.contains("ref_audios")){if(!body.at("ref_audios").is_array()){if(error)*error="ref_audios must be an array";return false;}request->ref_audios.resize(body.at("ref_audios").size());for(size_t i=0;i<request->ref_audios.size();++i)if(!decode_audio(body.at("ref_audios").at(i),&request->audio_storage[audio_index++],&request->ref_audios[i])){if(error)*error="ref_audios entries must be b64_f32le with sample_rate/channels";return false;}request->params.ref_audios=request->ref_audios.data();request->params.ref_audio_count=static_cast<int>(request->ref_audios.size());if(request->ref_images.empty()&&request->ref_videos.empty()&&!request->ref_audios.empty()){if(error)*error="ref_audios require at least one reference image or video";return false;}}
    request->params.strength = json_get_number(body, "strength", request->params.strength);
    request->params.vace_strength = json_get_number(body, "vace_strength", request->params.vace_strength);
    request->params.moe_boundary = json_get_number(body, "moe_boundary", request->params.moe_boundary);
    request->params.sample.sampler = runtime.defaults->sampler;
    request->params.sample.scheduler = runtime.defaults->scheduler;
    request->params.sample.steps = json_get_number(body, "steps", runtime.defaults->steps);
    request->params.sample.cfg_scale = json_get_number(body, "cfg_scale", runtime.defaults->cfg_scale);
    request->params.sample.distilled_guidance = json_get_number(body, "distilled_guidance", runtime.defaults->distilled_guidance);
    request->params.sample.flow_shift = json_get_number(body, "flow_shift", runtime.defaults->flow_shift);
    request->params.sample.cache_mode = runtime.defaults->cache_mode;
    const std::string sampler = json_get_string(body, "sampler");
    if (!sampler.empty() && !ed_sampler_from_string(sampler, &request->params.sample.sampler)) { if (error) *error="unsupported sampler: "+sampler; return false; }
    const std::string scheduler = json_get_string(body, "scheduler");
    if (!scheduler.empty() && !ed_scheduler_from_string(scheduler, &request->params.sample.scheduler)) { if (error) *error="unsupported scheduler: "+scheduler; return false; }
    const json* cache = cache_object(body); const std::string cache_mode=get_cache_string(body,cache,"mode","cache_mode");
    if (!cache_mode.empty() && !ed_cache_mode_from_string(cache_mode,&request->params.sample.cache_mode)) { if(error)*error="unsupported cache_mode: "+cache_mode; return false; }
    request->params.sample.cache_reuse_threshold=get_cache_number(body,cache,"reuse_threshold","cache_reuse_threshold",request->params.sample.cache_reuse_threshold);
    request->params.sample.cache_start_percent=get_cache_number(body,cache,"start_percent","cache_start_percent",request->params.sample.cache_start_percent);
    request->params.sample.cache_end_percent=get_cache_number(body,cache,"end_percent","cache_end_percent",request->params.sample.cache_end_percent);
    request->cache_scm_mask=get_cache_string(body,cache,"scm_mask","cache_scm_mask"); request->params.sample.cache_scm_mask=request->cache_scm_mask.empty()?nullptr:request->cache_scm_mask.c_str();
    return validate_video_params(request->params,error);
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
        "/ed/v1/videos/generations",
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
        "magcache",
        "dicache",
        "sencache",
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
        "ltx2",
    };
    result["defaults"] = {
        {"width", runtime.defaults->width},
        {"height", runtime.defaults->height},
        {"frames", runtime.defaults->frames},
        {"fps", runtime.defaults->fps},
        {"steps", runtime.defaults->steps},
        {"seed", runtime.defaults->seed},
        {"cfg_scale", runtime.defaults->cfg_scale},
        {"image_cfg_scale", runtime.defaults->image_cfg_scale},
        {"distilled_guidance", runtime.defaults->distilled_guidance},
        {"flow_shift", runtime.defaults->flow_shift},
        {"cache_mode", ed_cache_mode_to_string(runtime.defaults->cache_mode)},
    };
    result["pipeline_name"] = runtime.ctx ? ed_context_pipeline_name(runtime.ctx) : nullptr;
    result["supports"] = {{"image", runtime.ctx && ed_context_supports_image(runtime.ctx)}, {"video", runtime.ctx && ed_context_supports_video(runtime.ctx)}, {"audio_output", true}, {"ltx_hires", runtime.context && runtime.context->latent_upscaler_path != nullptr}};
    return result;
}
