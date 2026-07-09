#pragma once

#include <vector>

namespace edgedit {
namespace cache {

// A stable, independently-skippable computation unit of the model. The current
// (Option A) topology is deliberately coarse: input projection, one block-stack
// segment, output projection. Finer segments (per-block, attention/FFN) are
// additive and unlock the deferred per-segment graph builders.
enum class SegmentKind {
    INPUT_PROJECTION,
    BLOCK_STACK,   // the whole transformer block stack (double+single for flux)
    OUTPUT_PROJECTION,
};

// What a segment permits the cache system to do at its boundaries. Kept minimal
// for Option A; grows as models expose finer seams.
struct SegmentCapabilities {
    bool can_capture_output = false;  // expose block-stack residual (Feature methods)
    bool can_inject_output = false;   // replace the residual on a reuse step
    bool can_probe = false;           // stop after a prefix of blocks (Probe methods)
};

struct SegmentDesc {
    int id = -1;
    SegmentKind kind = SegmentKind::BLOCK_STACK;

    // Block interval [block_start, block_end) this segment covers within the
    // model's block stack. Only meaningful for BLOCK_STACK.
    int block_start = 0;
    int block_end = 0;

    SegmentCapabilities capabilities;
};

// Ordered segments of a model. For Option A there is exactly one BLOCK_STACK
// segment; the vector shape leaves room for finer decomposition later.
struct ModelTopology {
    std::vector<SegmentDesc> segments;

    const SegmentDesc* block_stack() const {
        for (const auto& s : segments) {
            if (s.kind == SegmentKind::BLOCK_STACK) {
                return &s;
            }
        }
        return nullptr;
    }
};

}  // namespace cache
}  // namespace edgedit
