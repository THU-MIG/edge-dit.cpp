#pragma once

#include "core/optimization/cache/ir/cache_program.hpp"
#include "core/optimization/cache/ir/runtime_decision.hpp"
#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/cache_graph_scope.hpp"  // sd::DiffusionCacheResult
#include "utils/tensor.hpp"

#include <functional>

namespace edgedit {
namespace cache {

// Model-agnostic compute callbacks the pipeline wires to its exact runner calls.
// Unchanged in spirit from the pre-redesign CacheRunnerHooks: the lowering
// invokes whichever subset the chosen variant needs.
//   full   - normal full transformer forward (also every fallback path).
//   input  - the block-stack input latent (Output-level diff + SenCache).
//   capture/inject/probe - the Feature/Probe seam passes.
// feature_supported is false when the model can't cut its block stack this run.
struct CacheRunnerHooks {
    const sd::Tensor<float>* input = nullptr;
    std::function<sd::Tensor<float>()> full;
    bool feature_supported = false;
    std::function<sd::DiffusionCacheResult(int, int)> capture;             // (region_start, region_end)
    std::function<sd::Tensor<float>(const sd::Tensor<float>&, int, int)> inject;  // (feature, start, end)
    std::function<sd::DiffusionCacheResult(int)> probe;                    // (probe_depth)
    // Calibration evaluator: model velocity prediction at a raw latent + sigma,
    // with the pipeline's own input construction + CFG combine. Wired only on a
    // --cache-calibrate run for a method whose CalibrationSpec needs it.
    std::function<sd::Tensor<float>(const sd::Tensor<float>&, float)> forward_at;
};

// Executes one chosen GraphVariantPlan against the runner hooks, using the
// policy for host-side reconstruction and observation. This is the seam between
// the declarative program and the current CacheGraphScope-based runner passes
// (Option A lowering — see the plan's "Key architectural decision").
class CacheGraphLowering {
public:
    // Run `variant` for one CFG branch. Returns the model output for the branch
    // (possibly a cache reconstruction). Always falls back to hooks.full() when
    // the seam is unavailable or a cache step can't be served.
    static sd::Tensor<float> execute(ICachePolicy& policy,
                                     const CacheProgram& program,
                                     const RuntimeDecision& decision,
                                     const StepContext& step,
                                     const void* condition_key,
                                     CacheBranch branch,
                                     const CacheRunnerHooks& hooks);
};

}  // namespace cache
}  // namespace edgedit
