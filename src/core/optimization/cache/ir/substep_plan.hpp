#pragma once

#include <optional>
#include <unordered_map>
#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_slot.hpp"  // CacheSlotId
#include "core/optimization/cache/ir/indicator.hpp"
#include "core/optimization/cache/model/anchor.hpp"

namespace edgedit {
namespace cache {

// Where a substep's activation stream begins.
struct InputSource {
    enum Kind { FreshLatent, FromSlot };
    Kind kind = FreshLatent;
    CacheSlotId slot = -1;  // meaningful for FromSlot (continuation from a stashed activation)
};

// Half-open block interval [begin, end) the substep actually computes. end < 0
// means "to the end of the stack". empty() => a pure-reuse substep (zero blocks).
struct BlockRange {
    int begin = 0;
    int end = -1;
    bool empty() const { return end >= 0 && begin == end; }
};

// How the substep turns computed/cached tensors into its output. Kept as a light
// tag for the Phase-2 adapter (dispatches to existing hooks). The declarative
// CacheAction lowering is a Phase-3 concern.
enum class SubstepOpKind {
    Identity,        // output = the computed block output as-is
    ApplyResidual,   // output = input + cached residual (slot)  [MagCache skip / DiCache reuse]
    Extrapolate,     // output = history extrapolation (blend coeffs)  [TaylorSeer / DiCache gamma]
    Stash,           // store the computed activation into a slot for a later substep [DiCache probe]
    OutputReuse,     // whole-denoiser-output reuse: LOAD diff slot + BLEND with input  [EasyCache/UCache/Condition]
    OutputCompute,   // full forward, then STORE (output - input) diff to slot          [Output methods]
    FeatureReuse,    // host feature-ring reuse: PREDICT/REUSE before-actions -> inject  [TaylorSeer/SenCache]
    FeatureCompute,  // full forward via hooks.capture, then STORE+ROTATE ring           [TaylorSeer/SenCache]
    CaptureProbeSeed,// DiCache seed: full forward + tap-driven writeback of the cross-step
                     // residual/probe rings (device) or host feature readback           [DiCache]
};

// Which declarative variant the Output-method substep drives (whole-output diff),
// interpreted via the CacheProgram's operators rather than the block-stack seam.

struct SubstepOp {
    SubstepOpKind kind = SubstepOpKind::Identity;
    std::vector<CacheSlotId> slots;   // operand slots (residual / stash target)
    std::vector<float> coeffs;        // blend/extrapolation weights (Extrapolate)
};

// One clean decide->build->run unit. A denoise step is 1+ substeps: most methods
// yield exactly one; DiCache yields two (probe, then continuation). The middle
// layer runs a uniform loop over these — DiCache is not a special case, it just
// yields twice.
struct SubstepPlan {
    InputSource            input;
    BlockRange             blocks;
    SubstepOp              op;
    std::vector<CacheSlotId> writes;       // slots to store into after compute (residual capture)
    std::vector<AnchorRef>   taps;         // anchors to register this substep
    std::vector<Indicator>   indicators;   // in-graph scalars, read back to host after run
    bool                     produces_output = false;  // does this substep produce the step's latent?
};

// Host-visible result of running one substep: the reduced indicator scalars.
struct SubstepResult {
    std::unordered_map<std::string, float> indicators;
    float get(const std::string& n, float dflt = 0.0f) const {
        auto it = indicators.find(n);
        return it != indicators.end() ? it->second : dflt;
    }
};

}  // namespace cache
}  // namespace edgedit
