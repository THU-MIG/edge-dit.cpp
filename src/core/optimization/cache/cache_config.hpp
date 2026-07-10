#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/cache_types.hpp"

namespace edgedit {
namespace cache {

struct DBCacheConfig {
    bool enabled = false;
    int fn_compute_blocks = 8;
    int bn_compute_blocks = 0;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
    float residual_diff_threshold = 0.08f;
    int max_warmup_steps = 8;
    int max_cached_steps = -1;
    int max_continuous_cached_steps = -1;
    float max_accumulated_residual_diff = -1.0f;
    std::vector<int> steps_computation_mask;
    bool scm_policy_dynamic = true;
};

struct TaylorSeerConfig {
    bool enabled = false;
    int n_derivatives = 1;
    int max_warmup_steps = 2;
    int skip_interval_steps = 1;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
};

struct EasyCacheConfig {
    bool enabled = false;
    float reuse_threshold = 0.2f;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
};

struct UCacheConfig {
    bool enabled = false;
    float reuse_threshold = 1.0f;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
    float error_decay_rate = 1.0f;
    bool use_relative_threshold = true;
    bool adaptive_threshold = true;
    float early_step_multiplier = 0.5f;
    float late_step_multiplier = 1.5f;
    float relative_norm_gain = 1.6f;
    bool reset_error_on_compute = true;
};

// MagCache: magnitude-ratio error accumulation over a precalibrated per-step
// ratio table. Reuses the block-stack residual like TaylorSeer (Feature level).
struct MagCacheConfig {
    bool enabled = false;
    float mag_thresh = 0.24f;       // accumulated-error skip threshold
    int max_skip_steps = 5;         // K: max consecutive skips
    float retention_ratio = 0.1f;   // fraction of leading steps always computed
    float start_percent = 0.15f;
    float end_percent = 0.95f;
    // When set, write a calibrated per-step magnitude-ratio profile to this path
    // (forces full compute every step). When set, load the ratio table from this
    // path instead of the MagCache policy's built-in table.
    std::string calibrate_path;
    std::string profile_path;
};

// DiCache: shallow-probe trajectory alignment (Probe level).
// error_choice picks the probe error metric and its paired defaults, matching
// the reference (bidcache src/flux/cache_methods/dicache):
//   delta_y      -> error = delta_y,            thresh 0.4,  ret_ratio 0.2
//   delta_minus  -> error = |delta_y - delta_x|, thresh 0.08, ret_ratio 0.0
enum class DiCacheErrorChoice { DeltaY, DeltaMinus };

struct DiCacheConfig {
    bool enabled = false;
    int probe_depth = 1;            // # front blocks run as the probe (ref default 1)
    float rel_l1_thresh = 0.4f;     // accumulated skip threshold (delta_y default)
    float retention_ratio = 0.2f;   // leading steps always computed (delta_y default)
    DiCacheErrorChoice error_choice = DiCacheErrorChoice::DeltaY;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
};


// SenCache: sensitivity-aware caching (Feature level). Skips when a first-order
// caching-error bound J_z*||dz|| + J_t*|dt| (normalized by sqrt(numel) so the
// threshold is resolution-independent) stays under a dual threshold. Reuses the
// block-stack residual like MagCache; consumes a precalibrated per-timestep
// (J_z, J_t) profile, so it requires --cache-profile or --cache-calibrate.
struct SenCacheConfig {
    bool enabled = false;
    float thresh_main = 0.07f;      // error threshold after the switch step
    float thresh_start = 0.005f;    // tighter threshold for early steps
    int max_skip_steps = 10;        // K: max consecutive skips
    float retention_ratio = 0.1f;   // fraction of leading steps always computed
    float switch_ratio = 0.2f;      // step fraction where thresh_start -> thresh_main
    float start_percent = 0.15f;
    float end_percent = 0.95f;
    // When set, write a calibrated per-step (J_z, J_t) profile to this path
    // (forces full compute + finite-diff Jacobian passes every step). When set,
    // load the profile from this path.
    std::string calibrate_path;
    std::string profile_path;
};

struct CacheConfig {
    CacheMode mode = CacheMode::Disabled;
    EasyCacheConfig easycache;
    UCacheConfig ucache;
    DBCacheConfig dbcache;
    TaylorSeerConfig taylorseer;
    MagCacheConfig magcache;
    DiCacheConfig dicache;
    SenCacheConfig sencache;
    int double_fn_blocks = -1;
    int double_bn_blocks = -1;
    int single_fn_blocks = -1;
    int single_bn_blocks = -1;
};

std::vector<int> parse_steps_computation_mask(const char* mask);
CacheConfig cache_config_from_sample_params(const ed_sample_params_t& params);

}  // namespace cache
}  // namespace edgedit
