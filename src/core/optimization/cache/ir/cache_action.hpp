#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_slot.hpp"
#include "core/optimization/cache/model/cache_site.hpp"

namespace edgedit {
namespace cache {

// Model-agnostic cache math, resolved to a registered CacheOperator at lowering
// time. Option A lowers these on the host; the deferred backend path lowers the
// same kinds to ggml nodes without changing the IR.
enum class CacheActionKind {
    COMPUTE,         // run the segment for real (no cache op)
    LOAD,            // read a slot into a value
    STORE,           // write a value into a slot
    DIFFERENCE,      // out = b - a  (residual capture)
    PREDICT,         // extrapolate from history (linear / polynomial)
    BLEND,           // weighted combine of inputs
    ROTATE_HISTORY,  // advance a slot's ring buffer after a committed step
};

// Identifies a value flowing between actions within a segment plan. Values are
// either a model site (captured/injected via the seam) or a slot (cache state).
struct ValueRef {
    enum class Kind { Site, Slot, Ambient, Temp } kind = Kind::Site;
    CacheSiteId site = -1;
    CacheSlotId slot = -1;
    int id = -1;  // Ambient/Temp local id

    static ValueRef of_site(CacheSiteId s) { return {Kind::Site, s, -1, -1}; }
    static ValueRef of_slot(CacheSlotId s) { return {Kind::Slot, -1, s, -1}; }
    static ValueRef of_temp(int t) { return {Kind::Temp, -1, -1, t}; }
    static ValueRef of_ambient(int a) { return {Kind::Ambient, -1, -1, a}; }
};

// Ambient values the lowering binds before interpreting an action list. These
// are the tensors that aren't cache slots: the block-stack input latent and the
// model output produced by a full compute this step.
enum : int {
    kAmbientInput = 0,           // block-stack input latent (hooks.input)
    kAmbientModelOutput = 1,     // result of a full compute this step
    kAmbientCapturedFeature = 2, // Feature seam: residual captured on a full step
    kAmbientInjectFeature = 3,   // Feature seam: residual to inject on a reuse step
};

// Named operator to invoke (e.g. "cache.linear_predict"). Parameters are a small
// untyped bag so operators stay model-agnostic and the IR stays serializable.
struct CacheOperatorParams {
    std::vector<float> floats;
    std::vector<int> ints;
};

struct CacheAction {
    CacheActionKind kind = CacheActionKind::COMPUTE;

    std::optional<CacheSiteId> site;
    std::optional<CacheSlotId> slot;

    std::string op;  // registered operator id; empty for COMPUTE
    CacheOperatorParams params;

    std::vector<ValueRef> inputs;
    std::vector<ValueRef> outputs;
};

}  // namespace cache
}  // namespace edgedit
