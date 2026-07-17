#pragma once

#include "core/optimization/cache/ir/tensor_spec.hpp"  // CacheDataType
#include "runtime/model_loader.h"                       // SDVersion

namespace edgedit {
namespace cache {

// Coarse model family, used by policies to express structural requirements
// (e.g. "dual-stream topology required") without naming a concrete model class.
enum class ModelFamily {
    Unknown,
    Flux,       // double-stream (img/txt) then single-stream
    MMDiT,      // joint image/text blocks (SD3)
    QwenImage,  // dual-stream transformer blocks
    WanVideo,   // video DiT with (optional) spatial/temporal attention
};

// Static description of a model. Describes properties only — no mutable
// inference state. Derived from the per-version data in
// cache_model_spec_for_version(); the primary axis policies negotiate against.
struct ModelSchema {
    SDVersion version = VERSION_COUNT;
    ModelFamily family = ModelFamily::Unknown;

    // Number of cacheable transformer blocks in the block stack. For flux this
    // is double_depth + single_depth; the whole-stack residual spans both.
    int num_blocks = 0;

    bool has_dual_stream = false;        // separate img/txt streams (flux, qwen)
    bool has_spatial_attention = false;  // video DiT
    bool has_temporal_attention = false;

    // True when the model's pipeline wires host feature-capture hooks
    // (substep_capture_host/inject_host). Host-capable models keep the
    // Feature-history methods (TaylorSeer/SenCache) even when a device store is
    // wired for MagCache; device-only runners (Flux/Qwen) do not.
    bool host_feature_capture = false;

    CacheDataType compute_dtype = CacheDataType::F32;
};

}  // namespace cache
}  // namespace edgedit
