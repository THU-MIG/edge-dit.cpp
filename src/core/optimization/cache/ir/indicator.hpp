#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_slot.hpp"  // CacheSlotId
#include "core/optimization/cache/model/anchor.hpp"

namespace edgedit {
namespace cache {

// A cheap decision signal computed IN the graph, off tapped anchors and/or cache
// slots, whose only host-visible product is a scalar. Replaces the hardcoded
// delta_y/delta_x/gamma nodes in CacheGraphScope: MagCache's norm-ratio and
// DiCache's probe-delta become two instances of one type, not two hand-written
// paths.
//
// Semantics (used by the on-device lowering, Phase 3):
//   RelL1(cur, ref) = sum|cur - ref| / sum|ref|   (matches build_probe_metrics rel())
//   L2Norm(a)       = sqrt(sum(a^2))
//   L2Delta(a, b)   = sqrt(sum((a-b)^2))
//   Dot(a, b)       = sum(a * b)
// anchors carries the operand anchors in order; slots carries cross-step operands
// (e.g. previous-step residual). Only the reduced scalar is read back to host —
// large anchor tensors never leave the device (the on-device red line).
struct Indicator {
    std::string name;  // "residual_norm_ratio" / "probe_delta"
    enum Kind { L2Norm, RelL1, L2Delta, Dot };
    Kind kind = RelL1;

    std::vector<AnchorRef> anchors;   // in-graph operands (RelL1: {cur, ref})
    std::vector<CacheSlotId> slots;   // cross-step operands (prev residual, ...)

    // Sharded anchors (sequence-parallel) need an all-reduce before the ratio is
    // meaningful; the middle layer inserts a mark_comm_op when set. Phase 5.
    bool reduce_across_sp = false;
};

}  // namespace cache
}  // namespace edgedit
