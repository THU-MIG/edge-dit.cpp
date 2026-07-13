#pragma once

#include <string>

#include "core/optimization/cache/ir/tensor_spec.hpp"

namespace edgedit {
namespace cache {

using CacheSlotId = int;

// How long a slot's contents must survive.
enum class CacheLifetime {
    ONE_STEP,        // scratch within a single step
    MULTI_STEP,      // survives across steps (the common case: cached residual)
    FULL_INFERENCE,  // survives the whole run (e.g. calibration tables)
};

enum class CacheAccessMode {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE,
};

// A declared unit of cache state. The CacheStateManager allocates one backing
// store per (condition_key, slot) so CFG cond/uncond branches stay isolated.
// history_depth > 1 requests a ring buffer (K-order prediction / double buffer).
struct CacheSlotDesc {
    CacheSlotId id = -1;
    std::string name;

    TensorSpec spec;  // usually dynamic; sized on first write
    CacheLifetime lifetime = CacheLifetime::MULTI_STEP;
    CacheAccessMode access = CacheAccessMode::READ_WRITE;

    int history_depth = 1;
    bool persistent = true;

    // Opt-in: when the state manager has a device store wired (GPU runner + seam
    // available), back this slot's ring entries with persistent device tensors so
    // the residual can be reused on-device without a host round-trip. Ignored when
    // no device store is present (CPU backend, SP/CFG, mmdit/wan) — the slot then
    // falls back to the host ring. Set by the policy's program builder for the
    // block-stack residual/feature slots that have an on-GPU reuse path.
    bool device_backed = false;
};

}  // namespace cache
}  // namespace edgedit
