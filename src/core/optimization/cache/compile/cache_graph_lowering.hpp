#pragma once

#include "core/optimization/cache/ir/cache_program.hpp"
#include "core/optimization/cache/ir/runtime_decision.hpp"
#include "core/optimization/cache/ir/graph_extension.hpp"
#include "core/optimization/cache/operator/cache_operator_registry.hpp"
#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/state/cache_state_manager.hpp"
#include "core/optimization/cache/cache_graph_scope.hpp"  // sd::DiffusionCacheResult
#include "utils/tensor.hpp"

#include <functional>

namespace edgedit {
namespace cache {

// Model-agnostic compute callbacks the pipeline wires to its exact runner calls.
// The runner entry points the lowering invokes to run one substep. All cache
// execution is tap-driven (TapRegistry, no CacheGraphScope); the lowering calls
// whichever subset the chosen SubstepPlan needs.
//   full  - normal full transformer forward (also every fallback path).
//   input - the block-stack input latent (Output-level diff + SenCache).
//   substep_* - the tap-driven capture/probe/inject passes (device or host).
struct CacheRunnerHooks {
    const sd::Tensor<float>* input = nullptr;
    std::function<sd::Tensor<float>()> full;

    // Substep-path tap-driven capture (MagCache device slot): the lowering builds
    // the GraphExtension list (a DIFFERENCE that weaves the residual and a slot to
    // d2d it into); the runner requests the taps those extensions reference, runs
    // the forward, weaves the operator nodes, and hands the pinned residual off to
    // the slot. The runner never learns the math is a residual.
    std::function<sd::Tensor<float>(std::vector<GraphExtension>)> substep_capture;

    // Substep-path tap-driven DiCache device seed capture: full forward whose
    // post-readback refreshes the cross-step residual/probe rings device-to-device
    // from the tap-woven nodes. The ring lives in CacheStateManager device slots
    // (face C); the bridge lets the runner drive rotate/alloc without depending on
    // the state manager type. Returns the step output. Null on the host path.
    std::function<sd::Tensor<float>(DiCacheSlotBridge)> substep_capture_probe;

    // Substep-path tap-driven DiCache probe: runs the shallow prefix and returns
    // delta_y/delta_x/gamma. The metrics are woven by cache.rel_l1 / cache.gamma_indicator
    // operators (resolved from the passed registry); the probe-history device operands
    // come from the CacheStateManager slots via the bridge.
    // (probe_depth, operators, bridge). Null on the host path.
    std::function<sd::DiffusionCacheResult(int, const CacheOperatorRegistry&, DiCacheSlotBridge)> substep_probe;

    // Host-path substep variants (models with no device slot — Wan). Tap-driven like
    // the device ones, but the residual / probe tensors are read back to host.
    // substep_capture_host: full forward, host feature (ModelOut-ModelIn).
    // substep_probe_host: shallow prefix, host before/probe. Null on device/legacy.
    std::function<sd::DiffusionCacheResult()> substep_capture_host;
    std::function<sd::DiffusionCacheResult(int)> substep_probe_host;
    // Host reuse (Wan): tap-driven inject of a host-reconstructed residual. (feature).
    std::function<sd::Tensor<float>(const sd::Tensor<float>&)> substep_inject_host;
    // Device reuse tap-driven inject (Qwen/Flux). MagCache: the lowering hands over
    // a ReplaceStream GraphExtension (WeightedBlend of x_before + device slot); the
    // runner requests the region taps, weaves op->lower(), and the forward
    // substitutes it over the reuse region. The runner never learns the math is a
    // residual add. substep_inject_gpu (DiCache gamma-blend) still uses the legacy
    // hardcoded path pending its own migration.
    std::function<sd::Tensor<float>(std::vector<GraphExtension>)> substep_inject_slot;
    // DiCache device reuse: the lowering hands over a ReplaceStream GraphExtension
    // (cache.gamma_blend, gamma baked into params.floats as a host constant); the
    // model reads the residual-ring operands from the CacheStateManager slots via
    // the bridge and build_stream_override weaves the blend. The runner never learns
    // the math is a gamma extrapolation. (extensions, bridge).
    std::function<sd::Tensor<float>(std::vector<GraphExtension>, DiCacheSlotBridge)> substep_inject_gpu;
};

// Drives the policy's substep loop against the runner hooks, using the policy for
// host-side reconstruction and observation. This is the seam between the
// declarative program and the tap-driven runner passes.
class CacheGraphLowering {
public:
    // The substep loop: drive the policy's next_substep() for one CFG branch,
    // translating each SubstepPlan into the runner hooks (tap-driven capture/probe/
    // inject, or the host/device reuse paths). The only execution path — every cache
    // method implements next_substep(). Falls back to hooks.full() when a substep
    // can't be served.
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
