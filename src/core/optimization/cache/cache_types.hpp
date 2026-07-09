#pragma once

#include <string>
#include <vector>

#include "edge-dit.h"
#include "runtime/model_loader.h"
#include "utils/tensor.hpp"

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
    SenCache,
};

enum class CacheBranch {
    Main,
    Cond,
    Uncond,
};

// Decision granularity of a cache policy. Determines which lifecycle hooks the
// engine drives and whether the method needs the model's block-stack seam.
//   Output  - operates on whole-model input latent / output noise (black box).
//   Feature - operates on the block-stack residual captured via the model seam.
//   Probe   - runs a shallow prefix of blocks, then decides skip vs. full.
enum class CacheGranularity {
    Output,
    Feature,
    Probe,
};

// Per-model block-count / naming metadata, consumed when a policy needs the
// model name (MagCache/SenCache profiles) or block count. The declarative
// contract (DiTModelCacheContract) is derived from the same version data.
struct CacheModelSpec {
    std::string model_name;
    SDVersion version = VERSION_COUNT;
    int block_count = 0;
    bool separate_cfg = false;
};

struct CacheStepInfo {
    int step_index = -1;
    int num_steps = 0;
    float sigma = 0.0f;
    float sigma_next = 0.0f;
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
        case ED_CACHE_SENCACHE: return CacheMode::SenCache;
        case ED_CACHE_DISABLED:
        default: return CacheMode::Disabled;
    }
}

const char* cache_mode_name(CacheMode mode);
bool cache_mode_supports_calibration(CacheMode mode);
CacheModelSpec cache_model_spec_for_version(SDVersion version);

}  // namespace cache
}  // namespace edgedit
