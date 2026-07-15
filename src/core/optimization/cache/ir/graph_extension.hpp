#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_action.hpp"  // CacheOperatorParams
#include "core/optimization/cache/model/anchor.hpp"      // AnchorRef

// Forward declarations keep this IR header free of a real ggml.h include, so the
// declarative layer stays serializable and cheap to compile. Only the runner and
// the operator lowering (which dereference these) include the full definitions.
struct ggml_tensor;

namespace edgedit {
namespace cache {

class ICacheOperator;

// A cache-layer instruction to EXTEND the model's compute graph: "run this
// operator over these tapped tensors and pin the result under output_name". The
// runner executes it blindly via op->lower() — it never learns what the math
// means (residual? blend?). This is what replaces the runner's hardcoded
// ggml_sub/ggml_add cache math: the cache layer now owns the graph nodes it wants
// woven, the model only reports structural landmarks (taps).
//
// Built by the lowering (which resolves `op` from the operator registry) and
// threaded to the runner through a substep hook. `op` points into the
// CacheEngine-owned registry, which outlives the pass.
struct GraphExtension {
    // Where the woven output goes.
    //   CaptureToSlot : pinned under output_name, then d2d-copied into a device
    //                   cache slot by the pass handoff (MagCache residual capture).
    //   ReplaceStream : the woven node replaces the block-stack stream at reuse
    //                   (wired in the reuse step; not produced yet).
    enum class Sink { CaptureToSlot, ReplaceStream };

    const ICacheOperator* op = nullptr;      // resolved operator (op->lower emits nodes)
    std::string op_id;                       // operator id, for diagnostics
    std::vector<AnchorRef> input_anchors;    // structural taps feeding the operator, in order
    std::vector<ggml_tensor*> extra_inputs;  // non-tap device operands (slot / runtime scalar)
    std::vector<ggml_tensor*> runtime_scalars;  // per-input broadcastable [1] device scalars
                                                // (empty = use params.floats constants); DiCache
                                                // gamma enters the blend here without a host copy
    CacheOperatorParams params;              // operator params (e.g. blend weights)
    std::string output_name;                 // name to pin the woven node under (readback / d2d)
    Sink sink = Sink::CaptureToSlot;

    // CaptureToSlot: allocate the device slot for the woven tensor's shape (known
    // only post-capture). Mirrors the old alloc_slot hook contract.
    std::function<void*(const std::vector<int64_t>&)> alloc_slot;
};

}  // namespace cache
}  // namespace edgedit
