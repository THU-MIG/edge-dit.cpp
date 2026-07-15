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
    //   Indicator     : the woven node is a scalar decision metric (DiCache
    //                   delta_y/gamma); pinned under output_name (a "cache_ind:<x>"
    //                   name) and read back to host by run_substep_pass. No stream
    //                   substitution and no slot d2d.
    enum class Sink { CaptureToSlot, ReplaceStream, Indicator };

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

// Callback bridge for DiCache's multi-slot ring, letting the runner drive the
// CacheStateManager device slots WITHOUT depending on the state manager type
// (same decoupling trick as GraphExtension::alloc_slot). The lowering binds these
// to state.rotate_history / alloc_device_entry / read_history for a fixed
// condition_key; the runner calls them by slot id.
//
// ⚠️ DEPTH CONVENTION (face C, mock-verified in /tmp/rotate_equiv.cpp):
// writeback is ROTATE-FIRST then write, so the NEWEST entry is at depth 0
// (read(slot)=read_history(slot,0)) and the previous at depth 1. This DIVERGES
// from the host make_dicache_program, which reads of_slot_history(slot,1)+(,2).
// The device path never executes those host LOAD actions, so the divergence is
// benign — but if DiCache reuse/gamma ever reads the WRONG entry (off-by-one in
// the residual blend or a stale gamma), THIS is the first place to check: confirm
// rotate() is called BEFORE alloc()/write() in writeback, and that reads use
// depths 0/1 (newest/prev), not 1/2.
struct DiCacheSlotBridge {
    // rotate the slot's ring so the next alloc targets a fresh newest entry.
    std::function<void(int /*slot*/)> rotate;
    // allocate/fetch the current newest ring entry at `shape`; returns device tensor.
    std::function<void*(int /*slot*/, const std::vector<int64_t>& /*shape*/)> alloc;
    // read a history entry (depth 0 = newest after writeback); nullptr if unfilled.
    std::function<void*(int /*slot*/, int /*depth*/)> read;
    // how many entries are filled in the slot's ring (availability gate).
    std::function<int(int /*slot*/)> filled;
    bool valid() const { return rotate && alloc && read && filled; }
};

}  // namespace cache
}  // namespace edgedit
