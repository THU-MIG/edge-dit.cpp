#include "core/optimization/cache/cache_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace edgedit {
namespace cache {

namespace {

float clamp_percent(float value, float fallback) {
    if (!std::isfinite(value)) {
        return fallback;
    }
    return std::max(0.0f, std::min(1.0f, value));
}

int parse_int_token(const std::string& token) {
    size_t pos = 0;
    while (pos < token.size() && std::isspace(static_cast<unsigned char>(token[pos]))) {
        ++pos;
    }

    bool negative = false;
    if (pos < token.size() && (token[pos] == '+' || token[pos] == '-')) {
        negative = token[pos] == '-';
        ++pos;
    }

    int value = 0;
    while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos]))) {
        value = value * 10 + (token[pos] - '0');
        ++pos;
    }

    return negative ? -value : value;
}

}  // namespace

std::vector<int> parse_steps_computation_mask(const char* mask) {
    std::vector<int> result;
    if (mask == nullptr || mask[0] == '\0') {
        return result;
    }

    const std::string mask_str(mask);
    size_t start = 0;
    while (start < mask_str.size()) {
        size_t pos = mask_str.find(',', start);
        const std::string token = mask_str.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!token.empty()) {
            result.push_back(parse_int_token(token) != 0 ? 1 : 0);
        }
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return result;
}

static float cache_reuse_threshold(const ed_sample_params_t& params, CacheMode mode) {
    float threshold = params.cache_reuse_threshold;
    if (std::isinf(threshold)) {
        if (mode == CacheMode::EasyCache) {
            threshold = 0.2f;
        } else if (mode == CacheMode::UCache) {
            threshold = 1.0f;
        }
    }
    return std::max(0.0f, threshold);
}

CacheConfig cache_config_from_sample_params(const ed_sample_params_t& params) {
    CacheConfig cfg;
    cfg.mode = cache_mode_from_ld(params.cache_mode);
    const float start_percent = clamp_percent(params.cache_start_percent, 0.15f);
    const float end_percent = clamp_percent(params.cache_end_percent, 0.95f);

    cfg.easycache.enabled = cfg.mode == CacheMode::EasyCache;
    cfg.easycache.reuse_threshold = cache_reuse_threshold(params, cfg.mode);
    cfg.easycache.start_percent = start_percent;
    cfg.easycache.end_percent = end_percent;

    cfg.ucache.enabled = cfg.mode == CacheMode::UCache;
    cfg.ucache.reuse_threshold = cache_reuse_threshold(params, cfg.mode);
    cfg.ucache.start_percent = start_percent;
    cfg.ucache.end_percent = end_percent;
    cfg.ucache.error_decay_rate = std::max(0.0f, std::min(1.0f, params.cache_error_decay_rate));
    cfg.ucache.use_relative_threshold = params.cache_use_relative_threshold;
    cfg.ucache.reset_error_on_compute = params.cache_reset_error_on_compute;

    cfg.dbcache.enabled = cfg.mode == CacheMode::DBCache || cfg.mode == CacheMode::CacheDiT;
    cfg.dbcache.fn_compute_blocks = params.cache_Fn_compute_blocks;
    cfg.dbcache.bn_compute_blocks = params.cache_Bn_compute_blocks;
    cfg.dbcache.start_percent = start_percent;
    cfg.dbcache.end_percent = end_percent;
    cfg.dbcache.residual_diff_threshold = params.cache_residual_diff_threshold;
    cfg.dbcache.max_accumulated_residual_diff = params.cache_max_accumulated_residual_diff;
    cfg.dbcache.max_warmup_steps = params.cache_max_warmup_steps;
    cfg.dbcache.max_cached_steps = params.cache_max_cached_steps;
    cfg.dbcache.max_continuous_cached_steps = params.cache_max_continuous_cached_steps;
    cfg.dbcache.steps_computation_mask = parse_steps_computation_mask(params.cache_scm_mask);
    cfg.dbcache.scm_policy_dynamic = params.cache_scm_policy_dynamic;

    cfg.taylorseer.enabled = cfg.mode == CacheMode::TaylorSeer || cfg.mode == CacheMode::CacheDiT;
    cfg.taylorseer.n_derivatives = params.cache_taylorseer_n_derivatives;
    cfg.taylorseer.max_warmup_steps = params.cache_max_warmup_steps;
    cfg.taylorseer.skip_interval_steps = params.cache_taylorseer_skip_interval;
    cfg.taylorseer.start_percent = start_percent;
    cfg.taylorseer.end_percent = end_percent;

    cfg.magcache.enabled = cfg.mode == CacheMode::MagCache;
    if (std::isfinite(params.cache_residual_diff_threshold) && params.cache_residual_diff_threshold > 0.0f) {
        cfg.magcache.mag_thresh = params.cache_residual_diff_threshold;
    }
    if (params.cache_max_continuous_cached_steps > 0) {
        cfg.magcache.max_skip_steps = params.cache_max_continuous_cached_steps;
    }
    cfg.magcache.start_percent = start_percent;
    cfg.magcache.end_percent = end_percent;

    cfg.dicache.enabled = cfg.mode == CacheMode::DiCache;
    if (params.cache_Fn_compute_blocks > 0) {
        cfg.dicache.probe_depth = params.cache_Fn_compute_blocks;
    }
    if (std::isfinite(params.cache_residual_diff_threshold) && params.cache_residual_diff_threshold > 0.0f) {
        cfg.dicache.rel_l1_thresh = params.cache_residual_diff_threshold;
    }
    cfg.dicache.start_percent = start_percent;
    cfg.dicache.end_percent = end_percent;

    return cfg;
}

}  // namespace cache
}  // namespace edgedit
