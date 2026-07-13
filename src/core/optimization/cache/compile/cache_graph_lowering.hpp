#pragma once

#include "core/optimization/cache/ir/cache_program.hpp"
#include "core/optimization/cache/ir/runtime_decision.hpp"
#include "core/optimization/cache/operator/cache_operator_registry.hpp"
#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/state/cache_state_manager.hpp"
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
    // GPU DiCache reuse: reconstruct + inject on-device from the persistent
    // residual ring using a host-clamped gamma. (gamma, region_start, region_end).
    // Empty result => insufficient history; lowering falls back to a full step.
    std::function<sd::Tensor<float>(float, int, int)> inject_gpu;
    // Feature-granularity on-GPU reuse (MagCache/TaylorSeer): inject
    // x_before + last_captured_residual straight from device memory, avoiding the
    // host reconstruct copy + H2D upload the plain `inject` hook pays. When set,
    // the `capture` hook also snapshots the residual to device. (start, end).
    // Empty result => no residual captured yet -> lowering falls back to full.
    std::function<sd::Tensor<float>(int, int)> inject_feature_gpu;
    // ---- Declarative device-slot seam (B2). When the CacheStateManager backs a
    // Feature slot with a device tensor, the lowering drives store/reuse through
    // these instead of the legacy inject_feature_gpu path. The void* is the slot's
    // opaque ggml_tensor* (from CacheSlotHandle::buffer).
    //   capture_to_slot: full compute; AFTER the block-stack residual shape is
    //     known, calls alloc_slot(residual_shape) to obtain the device slot tensor
    //     (the StateManager allocates it at that shape), copies the residual into
    //     it on-device, and returns the model output (empty => fail). Passing the
    //     allocator (not a pre-sized slot) is required because the residual shape
    //     is the packed block-stack seq shape, unknown to the lowering pre-capture.
    //   inject_from_slot: reuse step injecting x_before + slot residual on-device;
    //     returns output (empty => slot not ready -> caller falls back to capture).
    std::function<sd::Tensor<float>(
        const std::function<void*(const std::vector<int64_t>&)>&, int, int)> capture_to_slot;
    std::function<sd::Tensor<float>(void*, int, int)> inject_from_slot;  // (slot, start, end)
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
                                     const CacheRunnerHooks& hooks,
                                     CacheStateManager& state,
                                     const CacheOperatorRegistry& operators);
};

}  // namespace cache
}  // namespace edgedit
