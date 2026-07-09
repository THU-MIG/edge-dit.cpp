// Shared CLI surface for the edge-dit example binaries (ed-cli, ed-sample).
//
// This is the single source of truth for argument parsing, cache-flag mapping,
// distributed re-exec, and image I/O. Both entry points include it so their
// flag vocabulary can never drift apart.
//
// Include-order requirement: the including translation unit MUST define the STB
// implementation macros and include the stb image headers BEFORE this header,
// because save_png()/load_image() reference stbi_* symbols, e.g.
//
//   #define STB_IMAGE_IMPLEMENTATION
//   #include "ggml/examples/stb_image.h"
//   #define STB_IMAGE_WRITE_IMPLEMENTATION
//   #define STB_IMAGE_WRITE_STATIC
//   #include "stb_image_write.h"
//   #include "cli_common.hpp"

#pragma once

#include "edge-dit.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Image I/O (backed by stb, which the includer must have pulled in first).
// ---------------------------------------------------------------------------

inline bool save_png(const char* path, const ed_image_t& image) {
    if (path == nullptr || image.data == nullptr) {
        return false;
    }

    if (image.channels == 0 || image.channels > 4) {
        std::fprintf(stderr, "unsupported channel count: %u\n", image.channels);
        return false;
    }

    return stbi_write_png(
        path,
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        static_cast<int>(image.channels),
        image.data,
        0,
        nullptr
    ) != 0;
}

inline bool load_image(const char* path, ed_image_t* image) {
    if (path == nullptr || image == nullptr) {
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path, &width, &height, &channels, 3);
    if (data == nullptr || width <= 0 || height <= 0) {
        std::fprintf(stderr, "failed to load input image '%s': %s\n",
                     path,
                     stbi_failure_reason() != nullptr ? stbi_failure_reason() : "unknown error");
        if (data != nullptr) {
            stbi_image_free(data);
        }
        return false;
    }

    const size_t nbytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    uint8_t* owned = static_cast<uint8_t*>(std::malloc(nbytes));
    if (owned == nullptr) {
        stbi_image_free(data);
        std::fprintf(stderr, "failed to allocate input image buffer\n");
        return false;
    }
    std::memcpy(owned, data, nbytes);
    stbi_image_free(data);

    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = 3;
    image->data = owned;
    return true;
}

// ---------------------------------------------------------------------------
// Small string / env helpers.
// ---------------------------------------------------------------------------

inline std::string shell_quote(const char* value) {
    std::string quoted = "'";
    const char* text = value != nullptr ? value : "";
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p == '\'') {
            quoted += "'\\''";
        } else {
            quoted += *p;
        }
    }
    quoted += "'";
    return quoted;
}

inline std::string lowercase(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return value;
}

inline int env_int_value(const char* name, int fallback = 0) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return static_cast<int>(parsed);
}

inline bool is_distributed_process() {
    int world_size = env_int_value("WORLD_SIZE", 1);
    world_size = env_int_value("OMPI_COMM_WORLD_SIZE", world_size);
    world_size = env_int_value("MV2_COMM_WORLD_SIZE", world_size);
    world_size = env_int_value("SLURM_NTASKS", world_size);
    world_size = env_int_value("PMI_SIZE", world_size);
    return world_size > 1;
}

inline int count_csv_values(const char* csv) {
    if (csv == nullptr || csv[0] == '\0') {
        return 0;
    }

    int count = 0;
    bool in_value = false;
    for (const char* p = csv; *p != '\0'; ++p) {
        if (*p == ',') {
            if (in_value) {
                ++count;
                in_value = false;
            }
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(*p))) {
            in_value = true;
        }
    }
    return in_value ? count + 1 : count;
}

inline std::vector<std::string> split_csv_values(const char* csv) {
    std::vector<std::string> values;
    if (csv == nullptr || csv[0] == '\0') {
        return values;
    }

    std::string current;
    auto flush = [&]() {
        size_t begin = 0;
        while (begin < current.size() &&
               std::isspace(static_cast<unsigned char>(current[begin]))) {
            ++begin;
        }
        size_t end = current.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(current[end - 1]))) {
            --end;
        }
        if (end > begin) {
            values.emplace_back(current.substr(begin, end - begin));
        }
        current.clear();
    };

    for (const char* p = csv; *p != '\0'; ++p) {
        if (*p == ',') {
            flush();
        } else {
            current.push_back(*p);
        }
    }
    flush();
    return values;
}

inline bool is_cpu_backend_name(const char* backend) {
    return lowercase(backend != nullptr ? backend : "") == "cpu";
}

inline std::string normalized_video_format(const char* format) {
    std::string value = format != nullptr && format[0] != '\0' ? format : "auto";
    value = lowercase(value);
    if (!value.empty() && value[0] == '.') {
        value.erase(value.begin());
    }
    return value;
}

// ---------------------------------------------------------------------------
// Numeric parsers (locale-independent, tolerant of leading whitespace).
// ---------------------------------------------------------------------------

inline int parse_int_value(const char* text, int fallback = 0) {
    if (text == nullptr) {
        return fallback;
    }

    while (std::isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }

    bool negative = false;
    if (*text == '+' || *text == '-') {
        negative = *text == '-';
        ++text;
    }

    int value = 0;
    bool any = false;
    while (*text >= '0' && *text <= '9') {
        any = true;
        value = value * 10 + (*text - '0');
        ++text;
    }
    if (!any) {
        return fallback;
    }
    return negative ? -value : value;
}

inline int64_t parse_i64_value(const char* text, int64_t fallback = 0) {
    if (text == nullptr) {
        return fallback;
    }

    while (std::isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }

    bool negative = false;
    if (*text == '+' || *text == '-') {
        negative = *text == '-';
        ++text;
    }

    int64_t value = 0;
    bool any = false;
    while (*text >= '0' && *text <= '9') {
        any = true;
        value = value * 10 + (*text - '0');
        ++text;
    }
    if (!any) {
        return fallback;
    }
    return negative ? -value : value;
}

inline float parse_float_value(const char* text, float fallback = 0.0f) {
    if (text == nullptr) {
        return fallback;
    }

    bool negative = false;
    while (std::isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }
    if (*text == '+' || *text == '-') {
        negative = *text == '-';
        ++text;
    }

    double value = 0.0;
    bool any = false;
    while (*text >= '0' && *text <= '9') {
        any = true;
        value = value * 10.0 + static_cast<double>(*text - '0');
        ++text;
    }

    if (*text == '.') {
        ++text;
        double scale = 0.1;
        while (*text >= '0' && *text <= '9') {
            any = true;
            value += static_cast<double>(*text - '0') * scale;
            scale *= 0.1;
            ++text;
        }
    }

    if (!any) {
        return fallback;
    }

    if (*text == 'e' || *text == 'E') {
        ++text;
        const int exp = parse_int_value(text, 0);
        value *= std::pow(10.0, static_cast<double>(exp));
    }

    if (negative) {
        value = -value;
    }
    return static_cast<float>(value);
}

// ---------------------------------------------------------------------------
// Cache-mode enum <-> string.
// ---------------------------------------------------------------------------

inline ed_cache_mode_t parse_cache_mode(const char* text, bool* ok) {
    if (ok != nullptr) {
        *ok = true;
    }
    std::string mode = lowercase(text != nullptr ? text : "off");
    for (char& c : mode) {
        if (c == '_' || c == '.') {
            c = '-';
        }
    }

    // "original" is the benchmark's name for the no-cache baseline.
    if (mode == "off" || mode == "none" || mode == "disabled" || mode == "disable" ||
        mode == "original" || mode == "0") {
        return ED_CACHE_DISABLED;
    }
    if (mode == "easycache" || mode == "easy") {
        return ED_CACHE_EASYCACHE;
    }
    if (mode == "ucache" || mode == "u") {
        return ED_CACHE_UCACHE;
    }
    if (mode == "dbcache" || mode == "db") {
        return ED_CACHE_DBCACHE;
    }
    if (mode == "taylorseer" || mode == "taylor" || mode == "taylor-seer") {
        return ED_CACHE_TAYLORSEER;
    }
    if (mode == "cache-dit" || mode == "cachedit") {
        return ED_CACHE_CACHE_DIT;
    }
    if (mode == "magcache" || mode == "mag") {
        return ED_CACHE_MAGCACHE;
    }
    if (mode == "dicache" || mode == "di") {
        return ED_CACHE_DICACHE;
    }
    if (mode == "sencache" || mode == "sen") {
        return ED_CACHE_SENCACHE;
    }

    if (ok != nullptr) {
        *ok = false;
    }
    return ED_CACHE_DISABLED;
}

inline const char* cache_mode_name(ed_cache_mode_t mode) {
    switch (mode) {
        case ED_CACHE_DISABLED:   return "original";
        case ED_CACHE_EASYCACHE:  return "easycache";
        case ED_CACHE_UCACHE:     return "ucache";
        case ED_CACHE_DBCACHE:    return "dbcache";
        case ED_CACHE_TAYLORSEER: return "taylorseer";
        case ED_CACHE_CACHE_DIT:  return "cache-dit";
        case ED_CACHE_MAGCACHE:   return "magcache";
        case ED_CACHE_DICACHE:    return "dicache";
        case ED_CACHE_SENCACHE:   return "sencache";
        default:                  return "unknown";
    }
}

inline ed_dtype_t parse_weight_type(const char* text, bool* ok) {
    if (ok != nullptr) {
        *ok = true;
    }
    std::string type = lowercase(text != nullptr ? text : "");
    for (char& c : type) {
        if (c == '-' || c == '.') {
            c = '_';
        }
    }

    if (type == "auto" || type.empty()) {
        return ED_DTYPE_AUTO;
    }
    if (type == "f32" || type == "fp32") {
        return ED_DTYPE_F32;
    }
    if (type == "f16" || type == "fp16") {
        return ED_DTYPE_F16;
    }
    if (type == "bf16") {
        return ED_DTYPE_BF16;
    }
    if (type == "q4_0") {
        return ED_DTYPE_Q4_0;
    }
    if (type == "q4_1") {
        return ED_DTYPE_Q4_1;
    }
    if (type == "q5_0") {
        return ED_DTYPE_Q5_0;
    }
    if (type == "q5_1") {
        return ED_DTYPE_Q5_1;
    }
    if (type == "q8_0") {
        return ED_DTYPE_Q8_0;
    }
    if (type == "q2_k") {
        return ED_DTYPE_Q2_K;
    }
    if (type == "q3_k") {
        return ED_DTYPE_Q3_K;
    }
    if (type == "q4_k") {
        return ED_DTYPE_Q4_K;
    }
    if (type == "q5_k") {
        return ED_DTYPE_Q5_K;
    }
    if (type == "q6_k") {
        return ED_DTYPE_Q6_K;
    }

    if (ok != nullptr) {
        *ok = false;
    }
    return ED_DTYPE_AUTO;
}

// ---------------------------------------------------------------------------
// Parsed CLI arguments, shared by ed-cli (single prompt) and ed-sample (batch
// prompt file). The batch/timing fields at the bottom are unused by ed-cli.
// ---------------------------------------------------------------------------

struct FluxCliArgs {
    const char* model_path = nullptr;
    const char* diffusion_model_path = nullptr;
    const char* vae_path = nullptr;
    const char* clip_l_path = nullptr;
    const char* clip_g_path = nullptr;
    const char* t5xxl_path = nullptr;
    const char* prompt = nullptr;
    const char* negative_prompt = nullptr;
    const char* image_path = nullptr;
    const char* output_path = "output.png";
    const char* video_format = nullptr;
    const char* backend = nullptr;
    const char* devices = nullptr;
    const char* weight_type = nullptr;
    const char* tensor_type_rules = nullptr;
    int cfg_parallel_size = 1;
    int tp_parallel_size = 1;
    int sp_parallel_size = 1;
    int profile_graph_cuts_top = 8;

    bool video = false;
    bool profile_graph_cuts = false;
    bool profile_graph_cuts_all_ranks = false;
    bool flash_attention = true;
    bool qwen_image_zero_cond_t = false;
    int width = 1024;
    int height = 1024;
    int frames = 1;
    int fps = 16;
    int steps = 20;
    int threads = 0;

    int64_t seed = -1;
    float guidance = 3.5f;
    float cfg_scale = 1.0f;
    float flow_shift = 0.0f;

    ed_cache_mode_t cache_mode = ED_CACHE_DISABLED;
    float cache_reuse_threshold = std::numeric_limits<float>::infinity();
    float cache_start_percent = 0.15f;
    float cache_end_percent = 0.95f;
    float cache_error_decay_rate = 1.0f;
    bool cache_use_relative_threshold = true;
    bool cache_reset_error_on_compute = true;
    int cache_Fn_compute_blocks = 8;
    int cache_Bn_compute_blocks = 0;
    float cache_residual_diff_threshold = 0.08f;
    float cache_max_accumulated_residual_diff = -1.0f;
    int cache_max_warmup_steps = 8;
    int cache_max_cached_steps = -1;
    int cache_max_continuous_cached_steps = -1;
    int cache_taylorseer_n_derivatives = 1;
    int cache_taylorseer_skip_interval = 1;
    const char* cache_scm_mask = nullptr;
    bool cache_scm_policy_dynamic = true;
    const char* cache_calibrate_path = nullptr;
    const char* cache_profile_path = nullptr;
    bool no_t5 = false;
    int vae_tiling = -1;  // -1=default(off), 1=on
    float vae_tile_size = 0.0f;
    bool offload_to_cpu = false;
    bool keep_text_encoder_on_cpu = false;
    bool keep_vae_on_cpu = false;
    float max_vram = 0.0f;

    // --- batch / timing extension (used by ed-sample, ignored by ed-cli) ---
    const char* prompt_file = nullptr;
    const char* output_dir = nullptr;
    int start_index = 0;
    int end_index = -1;  // -1 means "until the end"
    int warmup = 1;
    int repeat = 1;
};

// Parses argv into `args`. Returns false on any error or --help (the caller is
// responsible for printing its own usage text). Accepts both ed-cli's canonical
// flags and ed-sample's snake_case aliases (--num_steps, --guidance_scale,
// --cfg_scale, --cache_method, --prompt_file, --output_dir, ...).
inline bool parse_args(int argc, char** argv, FluxCliArgs* args) {
    if (args == nullptr) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];

        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--video") == 0) {
            args->video = true;
        } else if (std::strcmp(key, "--video-format") == 0) {
            args->video_format = require_value(key);
        } else if (std::strcmp(key, "--model") == 0 || std::strcmp(key, "--model_path") == 0) {
            args->model_path = require_value(key);
        } else if (std::strcmp(key, "--diffusion-model") == 0) {
            args->diffusion_model_path = require_value(key);
        } else if (std::strcmp(key, "--vae") == 0) {
            args->vae_path = require_value(key);
        } else if (std::strcmp(key, "--clip_l") == 0) {
            args->clip_l_path = require_value(key);
        } else if (std::strcmp(key, "--clip_g") == 0) {
            args->clip_g_path = require_value(key);
        } else if (std::strcmp(key, "--t5xxl") == 0) {
            args->t5xxl_path = require_value(key);
        } else if (std::strcmp(key, "--prompt") == 0 || std::strcmp(key, "-p") == 0) {
            args->prompt = require_value(key);
        } else if (std::strcmp(key, "--prompt_file") == 0 || std::strcmp(key, "--prompt-file") == 0) {
            args->prompt_file = require_value(key);
        } else if (std::strcmp(key, "--output_dir") == 0 || std::strcmp(key, "--output-dir") == 0) {
            args->output_dir = require_value(key);
        } else if (std::strcmp(key, "--negative-prompt") == 0 ||
                   std::strcmp(key, "--negative_prompt") == 0) {
            args->negative_prompt = require_value(key);
            if (!args->negative_prompt) return false;
        } else if (std::strcmp(key, "--image") == 0 || std::strcmp(key, "-i") == 0) {
            args->image_path = require_value(key);
            if (!args->image_path) return false;
        } else if (std::strcmp(key, "--output") == 0 || std::strcmp(key, "-o") == 0) {
            args->output_path = require_value(key);
        } else if (std::strcmp(key, "--width") == 0 || std::strcmp(key, "-W") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->width = parse_int_value(v, args->width);
        } else if (std::strcmp(key, "--height") == 0 || std::strcmp(key, "-H") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->height = parse_int_value(v, args->height);
        } else if (std::strcmp(key, "--frames") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->frames = parse_int_value(v, args->frames);
        } else if (std::strcmp(key, "--fps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->fps = parse_int_value(v, args->fps);
        } else if (std::strcmp(key, "--steps") == 0 || std::strcmp(key, "--num_steps") == 0 ||
                   std::strcmp(key, "--num-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->steps = parse_int_value(v, args->steps);
        } else if (std::strcmp(key, "--threads") == 0 || std::strcmp(key, "-t") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->threads = parse_int_value(v, args->threads);
        } else if (std::strcmp(key, "--seed") == 0 || std::strcmp(key, "-s") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->seed = parse_i64_value(v, args->seed);
        } else if (std::strcmp(key, "--guidance") == 0 || std::strcmp(key, "--guidance_scale") == 0 ||
                   std::strcmp(key, "--guidance-scale") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->guidance = parse_float_value(v, args->guidance);
        } else if (std::strcmp(key, "--cfg-scale") == 0 || std::strcmp(key, "--cfg_scale") == 0 ||
                   std::strcmp(key, "--true_cfg_scale") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cfg_scale = parse_float_value(v, args->cfg_scale);
        } else if (std::strcmp(key, "--flow-shift") == 0 || std::strcmp(key, "--flow_shift") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->flow_shift = parse_float_value(v, args->flow_shift);
        } else if (std::strcmp(key, "--start_index") == 0 || std::strcmp(key, "--start-index") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->start_index = parse_int_value(v, args->start_index);
        } else if (std::strcmp(key, "--end_index") == 0 || std::strcmp(key, "--end-index") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->end_index = parse_int_value(v, args->end_index);
        } else if (std::strcmp(key, "--warmup") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->warmup = parse_int_value(v, args->warmup);
        } else if (std::strcmp(key, "--repeat") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->repeat = parse_int_value(v, args->repeat);
        } else if (std::strcmp(key, "--qwen-image-zero-cond-t") == 0) {
            args->qwen_image_zero_cond_t = true;
        } else if (std::strcmp(key, "--cache") == 0 || std::strcmp(key, "--cache-mode") == 0 ||
                   std::strcmp(key, "--cache_method") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            bool ok = false;
            args->cache_mode = parse_cache_mode(v, &ok);
            if (!ok) {
                std::fprintf(stderr, "unsupported cache mode: %s\n", v);
                return false;
            }
        } else if (std::strcmp(key, "--cache-threshold") == 0 ||
                   std::strcmp(key, "--cache-reuse-threshold") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_reuse_threshold = parse_float_value(v, args->cache_reuse_threshold);
        } else if (std::strcmp(key, "--cache-start") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_start_percent = parse_float_value(v, args->cache_start_percent);
        } else if (std::strcmp(key, "--cache-end") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_end_percent = parse_float_value(v, args->cache_end_percent);
        } else if (std::strcmp(key, "--cache-error-decay") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_error_decay_rate = parse_float_value(v, args->cache_error_decay_rate);
        } else if (std::strcmp(key, "--cache-relative-threshold") == 0) {
            args->cache_use_relative_threshold = true;
        } else if (std::strcmp(key, "--cache-absolute-threshold") == 0) {
            args->cache_use_relative_threshold = false;
        } else if (std::strcmp(key, "--cache-no-reset-error") == 0) {
            args->cache_reset_error_on_compute = false;
        } else if (std::strcmp(key, "--cache-reset-error") == 0) {
            args->cache_reset_error_on_compute = true;
        } else if (std::strcmp(key, "--cache-fn-blocks") == 0 ||
                   std::strcmp(key, "--cache-Fn-compute-blocks") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_Fn_compute_blocks = parse_int_value(v, args->cache_Fn_compute_blocks);
        } else if (std::strcmp(key, "--cache-bn-blocks") == 0 ||
                   std::strcmp(key, "--cache-Bn-compute-blocks") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_Bn_compute_blocks = parse_int_value(v, args->cache_Bn_compute_blocks);
        } else if (std::strcmp(key, "--cache-residual-threshold") == 0 ||
                   std::strcmp(key, "--cache-residual-diff-threshold") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_residual_diff_threshold = parse_float_value(v, args->cache_residual_diff_threshold);
        } else if (std::strcmp(key, "--cache-max-accumulated-residual-diff") == 0 ||
                   std::strcmp(key, "--cache-max-accumulated-residual-diff-threshold") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_accumulated_residual_diff =
                parse_float_value(v, args->cache_max_accumulated_residual_diff);
        } else if (std::strcmp(key, "--cache-warmup-steps") == 0 ||
                   std::strcmp(key, "--cache-max-warmup-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_warmup_steps = parse_int_value(v, args->cache_max_warmup_steps);
        } else if (std::strcmp(key, "--cache-max-cached-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_cached_steps = parse_int_value(v, args->cache_max_cached_steps);
        } else if (std::strcmp(key, "--cache-max-continuous-cached-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_continuous_cached_steps = parse_int_value(v, args->cache_max_continuous_cached_steps);
        } else if (std::strcmp(key, "--cache-taylor-order") == 0 ||
                   std::strcmp(key, "--cache-taylorseer-n-derivatives") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_taylorseer_n_derivatives = parse_int_value(v, args->cache_taylorseer_n_derivatives);
        } else if (std::strcmp(key, "--cache-taylor-skip") == 0 ||
                   std::strcmp(key, "--cache-taylorseer-skip-interval") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_taylorseer_skip_interval = parse_int_value(v, args->cache_taylorseer_skip_interval);
        } else if (std::strcmp(key, "--cache-scm-mask") == 0) {
            args->cache_scm_mask = require_value(key);
        } else if (std::strcmp(key, "--cache-calibrate") == 0) {
            args->cache_calibrate_path = require_value(key);
        } else if (std::strcmp(key, "--cache-profile") == 0) {
            args->cache_profile_path = require_value(key);
        } else if (std::strcmp(key, "--cache-static-scm") == 0) {
            args->cache_scm_policy_dynamic = false;
        } else if (std::strcmp(key, "--cache-dynamic-scm") == 0) {
            args->cache_scm_policy_dynamic = true;
        } else if (std::strcmp(key, "--backend") == 0) {
            args->backend = require_value(key);
        } else if (std::strcmp(key, "--gpu") == 0) {
            args->backend = "gpu";
        } else if (std::strcmp(key, "--devices") == 0 ||
                   std::strcmp(key, "--gpus") == 0) {
            args->devices = require_value(key);
        } else if (std::strcmp(key, "--type") == 0 ||
                   std::strcmp(key, "--weight-type") == 0) {
            args->weight_type = require_value(key);
        } else if (std::strcmp(key, "--tensor-type-rules") == 0) {
            args->tensor_type_rules = require_value(key);
        } else if (std::strcmp(key, "--no-t5") == 0) {
            args->no_t5 = true;
        } else if (std::strcmp(key, "--vae-tiling") == 0) {
            args->vae_tiling = 1;
        } else if (std::strcmp(key, "--vae-tile-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->vae_tile_size = parse_float_value(v, 0.0f);
        } else if (std::strcmp(key, "--offload-to-cpu") == 0) {
            args->offload_to_cpu = true;
        } else if (std::strcmp(key, "--keep-text-encoder-on-cpu") == 0) {
            args->keep_text_encoder_on_cpu = true;
        } else if (std::strcmp(key, "--keep-vae-on-cpu") == 0) {
            args->keep_vae_on_cpu = true;
        } else if (std::strcmp(key, "--max-vram") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->max_vram = parse_float_value(v, 0.0f);
        } else if (std::strcmp(key, "--flash-attention") == 0 ||
                   std::strcmp(key, "--flash-attn") == 0) {
            args->flash_attention = true;
        } else if (std::strcmp(key, "--no-flash-attention") == 0 ||
                   std::strcmp(key, "--no-flash-attn") == 0) {
            args->flash_attention = false;
        } else if (std::strcmp(key, "--cfg-parallel-size") == 0 ||
                   std::strcmp(key, "--cfg-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cfg_parallel_size = parse_int_value(v, args->cfg_parallel_size);
        } else if (std::strcmp(key, "--tp-size") == 0 ||
                   std::strcmp(key, "--tensor-parallel-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->tp_parallel_size = parse_int_value(v, args->tp_parallel_size);
        } else if (std::strcmp(key, "--sp-size") == 0 ||
                   std::strcmp(key, "--sequence-parallel-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->sp_parallel_size = parse_int_value(v, args->sp_parallel_size);
        } else if (std::strcmp(key, "--profile-graph-cuts") == 0) {
            args->profile_graph_cuts = true;
        } else if (std::strcmp(key, "--profile-graph-cuts-top") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->profile_graph_cuts_top = parse_int_value(v, args->profile_graph_cuts_top);
        } else if (std::strcmp(key, "--profile-graph-cuts-all-ranks") == 0) {
            args->profile_graph_cuts = true;
            args->profile_graph_cuts_all_ranks = true;
        } else if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            return false;
        }
    }

    const bool has_full_model = args->model_path != nullptr && std::strlen(args->model_path) > 0;
    const bool has_components =
        args->diffusion_model_path != nullptr && std::strlen(args->diffusion_model_path) > 0 &&
        args->vae_path != nullptr && std::strlen(args->vae_path) > 0 &&
        args->clip_l_path != nullptr && std::strlen(args->clip_l_path) > 0 &&
        (args->no_t5 || (args->t5xxl_path != nullptr && std::strlen(args->t5xxl_path) > 0));

    if (!has_full_model && !has_components) {
        std::fprintf(stderr, "--model or the full --diffusion-model/--vae/--clip_l/(--t5xxl or --no-t5) set is required\n");
        return false;
    }

    // ed-cli uses --prompt; ed-sample uses --prompt_file. Exactly one path must
    // supply the prompt(s).
    const bool has_prompt = args->prompt != nullptr && std::strlen(args->prompt) > 0;
    const bool has_prompt_file = args->prompt_file != nullptr && std::strlen(args->prompt_file) > 0;
    if (!has_prompt && !has_prompt_file) {
        std::fprintf(stderr, "--prompt or --prompt_file is required\n");
        return false;
    }

    if (args->width <= 0 || args->height <= 0) {
        std::fprintf(stderr, "width and height must be positive\n");
        return false;
    }

    if (args->frames <= 0) {
        std::fprintf(stderr, "frames must be positive\n");
        return false;
    }

    if (args->fps <= 0) {
        std::fprintf(stderr, "fps must be positive\n");
        return false;
    }

    const std::string video_format = normalized_video_format(args->video_format);
    if (video_format != "auto" &&
        video_format != "avi" &&
        video_format != "mp4" &&
        video_format != "mov" &&
        video_format != "mkv" &&
        video_format != "webm") {
        std::fprintf(stderr, "unsupported video format: %s\n", video_format.c_str());
        return false;
    }

    if (args->steps <= 0) {
        std::fprintf(stderr, "steps must be positive\n");
        return false;
    }

    if (args->cache_start_percent < 0.0f || args->cache_start_percent > 1.0f ||
        args->cache_end_percent < 0.0f || args->cache_end_percent > 1.0f ||
        args->cache_start_percent >= args->cache_end_percent) {
        std::fprintf(stderr, "cache window must satisfy 0 <= start < end <= 1\n");
        return false;
    }

    if (args->cache_reuse_threshold < 0.0f && !std::isinf(args->cache_reuse_threshold)) {
        std::fprintf(stderr, "cache threshold must be non-negative\n");
        return false;
    }

    if (args->cache_error_decay_rate < 0.0f || args->cache_error_decay_rate > 1.0f) {
        std::fprintf(stderr, "cache error decay must be in [0, 1]\n");
        return false;
    }

    if (args->cache_Fn_compute_blocks < 0 || args->cache_Bn_compute_blocks < 0) {
        std::fprintf(stderr, "cache block counts must be non-negative\n");
        return false;
    }

    if (args->cache_residual_diff_threshold < 0.0f) {
        std::fprintf(stderr, "cache residual threshold must be non-negative\n");
        return false;
    }

    if (args->cache_max_accumulated_residual_diff < -1.0f) {
        std::fprintf(stderr, "cache max accumulated residual diff must be >= -1\n");
        return false;
    }

    if (args->cache_max_warmup_steps < 0) {
        std::fprintf(stderr, "cache warmup steps must be non-negative\n");
        return false;
    }

    if (args->cache_taylorseer_n_derivatives < 1) {
        std::fprintf(stderr, "cache Taylor order must be >= 1\n");
        return false;
    }

    if (args->cache_taylorseer_skip_interval < 0) {
        std::fprintf(stderr, "cache Taylor skip interval must be non-negative\n");
        return false;
    }

    const bool want_calibrate =
        args->cache_calibrate_path != nullptr && args->cache_calibrate_path[0] != '\0';
    if (want_calibrate) {
        if (args->cache_mode == ED_CACHE_DISABLED) {
            std::fprintf(stderr,
                         "--cache-calibrate requires a cache method: pass a calibration-capable "
                         "--cache <mode> (e.g. --cache magcache)\n");
            return false;
        }
        if (!ed_cache_mode_supports_calibration(args->cache_mode)) {
            std::fprintf(stderr,
                         "--cache-calibrate is not supported by the selected cache method; "
                         "it only applies to table-driven methods (e.g. magcache)\n");
            return false;
        }
    }

    // SenCache's sensitivity bound is meaningless without a calibrated (J_z, J_t)
    // profile; refuse to run rather than silently degrade to no caching.
    if (args->cache_mode == ED_CACHE_SENCACHE && !want_calibrate) {
        const bool have_profile =
            args->cache_profile_path != nullptr && args->cache_profile_path[0] != '\0';
        if (!have_profile) {
            std::fprintf(stderr,
                         "--cache sencache requires a calibrated profile: pass --cache-profile "
                         "<path>, or produce one first with --cache-calibrate <path>\n");
            return false;
        }
    }

    if (args->cfg_parallel_size != 1 && args->cfg_parallel_size != 2) {
        std::fprintf(stderr, "cfg parallel size currently supports 1 or 2\n");
        return false;
    }

    if (args->tp_parallel_size != 1) {
        std::fprintf(stderr, "tensor parallel is not implemented yet; use --tp-size 1\n");
        return false;
    }

    if (args->sp_parallel_size <= 0) {
        std::fprintf(stderr, "--sp-size must be positive\n");
        return false;
    }

    if (args->profile_graph_cuts_top < 0) {
        std::fprintf(stderr, "--profile-graph-cuts-top must be non-negative\n");
        return false;
    }

    if (args->weight_type != nullptr) {
        bool ok = false;
        parse_weight_type(args->weight_type, &ok);
        if (!ok) {
            std::fprintf(stderr, "unsupported weight type: %s\n", args->weight_type);
            return false;
        }
    }

    return true;
}

inline void apply_cache_args(const FluxCliArgs& args, ed_sample_params_t* sample) {
    if (sample == nullptr) {
        return;
    }

    sample->cache_mode = args.cache_mode;
    sample->cache_reuse_threshold = args.cache_reuse_threshold;
    sample->cache_start_percent = args.cache_start_percent;
    sample->cache_end_percent = args.cache_end_percent;
    sample->cache_error_decay_rate = args.cache_error_decay_rate;
    sample->cache_use_relative_threshold = args.cache_use_relative_threshold;
    sample->cache_reset_error_on_compute = args.cache_reset_error_on_compute;
    sample->cache_Fn_compute_blocks = args.cache_Fn_compute_blocks;
    sample->cache_Bn_compute_blocks = args.cache_Bn_compute_blocks;
    sample->cache_residual_diff_threshold = args.cache_residual_diff_threshold;
    sample->cache_max_accumulated_residual_diff = args.cache_max_accumulated_residual_diff;
    sample->cache_max_warmup_steps = args.cache_max_warmup_steps;
    sample->cache_max_cached_steps = args.cache_max_cached_steps;
    sample->cache_max_continuous_cached_steps = args.cache_max_continuous_cached_steps;
    sample->cache_taylorseer_n_derivatives = args.cache_taylorseer_n_derivatives;
    sample->cache_taylorseer_skip_interval = args.cache_taylorseer_skip_interval;
    sample->cache_scm_mask = args.cache_scm_mask;
    sample->cache_scm_policy_dynamic = args.cache_scm_policy_dynamic;
    sample->cache_calibrate_path = args.cache_calibrate_path;
    sample->cache_profile_path = args.cache_profile_path;
}

inline int requested_parallel_size(const FluxCliArgs& args) {
    return std::max({1, args.cfg_parallel_size, args.tp_parallel_size, args.sp_parallel_size});
}

inline int launch_distributed_cli(int argc, char** argv, const FluxCliArgs& args, int device_count) {
    const int parallel_size = requested_parallel_size(args);
    if (parallel_size <= 1 || is_distributed_process()) {
        return -1;
    }

    if (args.devices == nullptr || std::strlen(args.devices) == 0) {
        std::fprintf(stderr, "parallel generation requires --devices with at least %d GPU ids\n", parallel_size);
        return 1;
    }
    if (device_count != parallel_size) {
        std::fprintf(stderr,
                     "parallel size (%d) must match --devices count (%d)\n",
                     parallel_size,
                     device_count);
        return 1;
    }

    const std::vector<std::string> devices = split_csv_values(args.devices);
    if (static_cast<int>(devices.size()) != parallel_size) {
        std::fprintf(stderr,
                     "parallel size (%d) must match parsed --devices count (%zu)\n",
                     parallel_size,
                     devices.size());
        return 1;
    }

    std::string argv_cmd;
    for (int i = 0; i < argc; ++i) {
        argv_cmd += " ";
        argv_cmd += shell_quote(argv[i]);
    }

    std::string cmd = "mpirun";
    for (int rank = 0; rank < parallel_size; ++rank) {
        if (rank > 0) {
            cmd += " :";
        }
        cmd += " -np 1 env ED_CLI_LAUNCHED=1 ED_CLI_SINGLE_VISIBLE_DEVICE=1 CUDA_VISIBLE_DEVICES=";
        cmd += shell_quote(devices[static_cast<size_t>(rank)].c_str());
        cmd += argv_cmd;
    }

    std::fprintf(stderr,
                 "launching %d parallel workers on devices %s\n",
                 parallel_size,
                 args.devices);
    const int status = std::system(cmd.c_str());
    if (status == -1) {
        std::fprintf(stderr, "failed to launch distributed workers with mpirun\n");
        return 127;
    }
    return status == 0 ? 0 : 1;
}
