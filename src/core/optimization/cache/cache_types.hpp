#pragma once

#include <string>
#include <vector>

#include "edge-dit.h"
#include "runtime/model_loader.h"

namespace edgedit {
namespace cache {

enum class CacheMode {
    Disabled,
    EasyCache,
    UCache,
    DBCache,
    TaylorSeer,
    CacheDiT,
    MagCache,
    DiCache,
};

enum class CacheExecType {
    Full,
    Reuse,
    Probe,
    ResumeFull,
    Taylor,
    Disabled,
};

enum class CacheStorageKind {
    None,
    HiddenState,
    Residual,
    BlockResidual,
    Token,
    Attention,
    Custom,
};

enum class CacheSelectorKind {
    All,
    Random,
    Attention,
    Norm,
    Score,
    Custom,
};

enum class CacheBranch {
    Main,
    Cond,
    Uncond,
};

// Decision granularity of a cache policy. Determines which lifecycle hooks the
// CacheController drives and whether it needs the model's build-time seam.
//   Output  - operates on whole-model input latent / output noise (black box).
//   Feature - operates on the block-stack residual captured via the model seam.
//   Probe   - runs a shallow prefix of blocks, then decides skip vs. full.
enum class CacheGranularity {
    Output,
    Feature,
    Probe,
};

enum class CacheRegionPattern {
    HiddenOnly,
    HiddenContext,
    ImageText,
    PackedImageText,
    Custom,
};

struct CacheRegionSpec {
    std::string id;
    std::string graph_prefix;
    int block_count = 0;
    CacheRegionPattern pattern = CacheRegionPattern::HiddenOnly;
    std::vector<std::string> input_keys;
    std::vector<std::string> output_keys;
    int default_fn_blocks = 0;
    int default_bn_blocks = 0;
};

struct CacheModelSpec {
    std::string model_name;
    SDVersion version = VERSION_COUNT;
    std::vector<CacheRegionSpec> regions;
    bool separate_cfg = false;
};

struct CacheStepInfo {
    int step_index = -1;
    int num_steps = 0;
    float sigma = 0.0f;
    float sigma_next = 0.0f;
};

// Per-branch context handed to a policy hook. condition_key identifies the
// SDCondition instance so the controller / policy can keep cond and uncond
// cache histories isolated across a CFG-parallel or dual-pass step.
struct CacheForwardContext {
    CacheStepInfo step;
    CacheBranch branch = CacheBranch::Main;
    const void* condition_key = nullptr;
};

// What the CacheController must do this step for a Feature/Probe policy.
struct CacheStepDecision {
    enum class Kind {
        Full,            // run the whole block stack; capture the feature
        SkipStackReuse,  // skip the block stack; inject a reconstructed feature
        Probe,           // run only probe_depth blocks, then decide
    };
    Kind kind = Kind::Full;
    int probe_depth = 0;  // meaningful only when kind == Probe
};

struct CacheRegionFrame {
    CacheStepInfo step;
    CacheBranch branch = CacheBranch::Main;
    std::string region_id;
    int block_index = -1;
};

struct CacheRegionPlan {
    CacheExecType exec_type = CacheExecType::Full;
    CacheStorageKind storage_kind = CacheStorageKind::Residual;
    CacheSelectorKind selector_kind = CacheSelectorKind::All;
    int fn_blocks = 0;
    int bn_blocks = 0;
    int probe_blocks = 0;
    bool needs_input_snapshot = false;
    bool needs_output_snapshot = true;
    bool can_reuse = false;
};

inline CacheMode cache_mode_from_ld(ed_cache_mode_t mode) {
    switch (mode) {
        case ED_CACHE_EASYCACHE: return CacheMode::EasyCache;
        case ED_CACHE_UCACHE: return CacheMode::UCache;
        case ED_CACHE_DBCACHE: return CacheMode::DBCache;
        case ED_CACHE_TAYLORSEER: return CacheMode::TaylorSeer;
        case ED_CACHE_CACHE_DIT: return CacheMode::CacheDiT;
        case ED_CACHE_MAGCACHE: return CacheMode::MagCache;
        case ED_CACHE_DICACHE: return CacheMode::DiCache;
        case ED_CACHE_DISABLED:
        default: return CacheMode::Disabled;
    }
}

const char* cache_mode_name(CacheMode mode);
CacheModelSpec cache_model_spec_for_version(SDVersion version);

}  // namespace cache
}  // namespace edgedit
