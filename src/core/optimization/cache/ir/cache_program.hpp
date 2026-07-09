#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_action.hpp"
#include "core/optimization/cache/ir/cache_slot.hpp"

namespace edgedit {
namespace cache {

// How a segment is executed for a given graph variant.
enum class SegmentExecutionMode {
    FULL_COMPUTE,         // build & run the segment
    LOAD_CACHED,          // skip the segment; inject a cached/blended value
    PREDICT_FROM_HISTORY,  // skip the segment; inject an extrapolated value
    PROBE,                // run only a prefix of the segment's blocks
};

// Multi-phase execution support (Phase 4 trajectory correction). Option A only
// ever uses FORWARD, but the field is carried so variant keys are future-proof.
enum class CachePhase {
    FORWARD,
    PROBE,
    FULL_ANCHOR,
    CORRECTION,
    REINTEGRATION,
};

// What to do for one segment in one variant: cache actions before it, how to
// execute it, cache actions after it.
struct SegmentPlan {
    int segment_id = -1;

    std::vector<CacheAction> before;
    SegmentExecutionMode execution = SegmentExecutionMode::FULL_COMPUTE;
    std::vector<CacheAction> after;

    // For PROBE / sub-region execution: the block interval that actually runs.
    int region_start = 0;
    int region_end = -1;   // < 0 => to end of stack
    int probe_depth = 0;   // meaningful only when execution == PROBE
};

using GraphVariantId = int;

enum class GraphVariantKind {
    FULL,     // compute everything, capture state
    REUSE,    // skip the cached region, inject loaded/blended state
    PREDICT,  // skip the cached region, inject extrapolated state
    PROBE,    // shallow prefix, then a runtime decision picks FULL or REUSE
};

// One selectable static-graph shape. The runtime picks a variant per step; the
// lowering turns it into the concrete runner pass(es).
struct GraphVariantPlan {
    GraphVariantId id = -1;
    GraphVariantKind kind = GraphVariantKind::FULL;
    CachePhase phase = CachePhase::FORWARD;

    std::vector<SegmentPlan> segments;
};

// The declarative output of ICachePolicy::compile(): the slots the method needs
// and the graph variants it can select among at runtime. Serializable for
// --dump-cache-program.
struct CacheProgram {
    std::string method_name;

    std::vector<CacheSlotDesc> slots;
    std::vector<GraphVariantPlan> variants;

    const GraphVariantPlan* find_variant(GraphVariantId id) const {
        for (const auto& v : variants) {
            if (v.id == id) {
                return &v;
            }
        }
        return nullptr;
    }
    const GraphVariantPlan* find_kind(GraphVariantKind kind) const {
        for (const auto& v : variants) {
            if (v.kind == kind) {
                return &v;
            }
        }
        return nullptr;
    }
};

std::string dump_cache_program(const CacheProgram& program);

}  // namespace cache
}  // namespace edgedit
