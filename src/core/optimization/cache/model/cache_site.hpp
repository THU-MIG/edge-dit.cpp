#pragma once

#include <cstdint>

#include "core/optimization/cache/ir/tensor_spec.hpp"

namespace edgedit {
namespace cache {

// A semantic position inside the model that the cache system may read, write,
// or replace. Roles are stable across models; a method declares which role it
// needs and the contract reports whether the model exposes it (and with what
// capability). This is the anchor of capability negotiation.
enum class CacheSiteRole {
    DENOISER_OUTPUT,     // whole-model output noise (Output-granularity methods)
    BLOCK_STACK_OUTPUT,  // block-stack residual (Feature methods reuse this)
    BLOCK_STACK_PROBE,   // hidden state after a prefix of blocks (Probe methods)
};

using CacheSiteId = int;

// What operations a given site supports on a given model/run. A site is only
// `replaceable` when the model's build-time seam can genuinely inject over it
// this run (e.g. false under SP-parallel, where block tensors are sharded).
struct CacheSiteCapability {
    bool readable = true;
    bool replaceable = false;
    bool storable = true;
    bool supports_history = true;
    bool supports_partial_compute = false;  // sub-block (attention/FFN) cuts
};

struct CacheSiteDesc {
    CacheSiteId id = -1;
    CacheSiteRole role = CacheSiteRole::DENOISER_OUTPUT;

    int segment_id = -1;  // segment this site lives on (ModelTopology)

    // Shape is usually dynamic (resolved at first write); dtype is known.
    TensorSpec tensor_spec;
    CacheSiteCapability capability;
};

}  // namespace cache
}  // namespace edgedit
