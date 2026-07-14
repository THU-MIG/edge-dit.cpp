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
    // ---- Declarative device-slot seam (B2). When the CacheStateManager backs a
    // Feature slot with a device tensor, the lowering drives store/reuse through
    // these (the on-GPU feature-reuse path for MagCache/TaylorSeer). The void* is
    // the slot's opaque ggml_tensor* (from CacheSlotHandle::buffer).
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

    // Substep-path (ED_CACHE_SUBSTEP) tap-driven capture: same contract as
    // capture_to_slot (alloc_slot returns the device slot for the residual shape)
    // but the runner drives it through the TapRegistry, not CacheGraphScope. Null
    // on the legacy path.
    std::function<sd::Tensor<float>(
        const std::function<void*(const std::vector<int64_t>&)>&)> substep_capture;

    // Substep-path (ED_CACHE_SUBSTEP) tap-driven DiCache probe: runs the shallow
    // prefix and returns delta_y/delta_x/gamma computed on-device from taps +
    // persistent operands, via the TapRegistry (no CacheGraphScope). (probe_depth).
    // Null on the legacy path.
    std::function<sd::DiffusionCacheResult(int)> substep_probe;

    // Host-path substep variants (models with no device slot — Wan). Tap-driven like
    // the device ones, but the residual / probe tensors are read back to host.
    // substep_capture_host: full forward, host feature (ModelOut-ModelIn).
    // substep_probe_host: shallow prefix, host before/probe. Null on device/legacy.
    std::function<sd::DiffusionCacheResult()> substep_capture_host;
    std::function<sd::DiffusionCacheResult(int)> substep_probe_host;
    // Host reuse (Wan): tap-driven inject of a host-reconstructed residual. (feature).
    std::function<sd::Tensor<float>(const sd::Tensor<float>&)> substep_inject_host;
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

    // Substep path (ED_CACHE_SUBSTEP): drive the policy's next_substep() loop for
    // one CFG branch, translating each SubstepPlan into the existing runner hooks
    // (capture_to_slot / inject_from_slot / full). Reuses all model plumbing; the
    // model forward is unchanged. Falls back to full() when a substep can't be
    // served. Only called for policies whose supports_substep() is true.
    static sd::Tensor<float> execute_substeps(ICachePolicy& policy,
                                              const CacheProgram& program,
                                              const StepContext& step,
                                              const void* condition_key,
                                              CacheBranch branch,
                                              const CacheRunnerHooks& hooks,
                                              CacheStateManager& state,
                                              const CacheOperatorRegistry& operators);
};

}  // namespace cache
}  // namespace edgedit
