#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "core/optimization/cache/ir/cache_program.hpp"
#include "core/optimization/cache/ir/runtime_decision.hpp"
#include "core/optimization/cache/ir/graph_extension.hpp"
#include "core/optimization/cache/model/dit_model_cache_contract.hpp"
#include "core/optimization/cache/operator/cache_operator_registry.hpp"
#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/state/cache_state_manager.hpp"
#include "core/optimization/cache/cache_graph_scope.hpp"  // sd::DiffusionCacheResult
#include "core/optimization/cache/cache_config.hpp"
#include "core/optimization/cache/cache_types.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Model-agnostic compute callbacks the pipeline wires to its exact runner calls.
// The runner entry points the engine invokes to run one substep. All cache
// execution is tap-driven (TapRegistry, no CacheGraphScope); the engine calls
// whichever subset the chosen SubstepPlan needs.
//   full  - normal full transformer forward (also every fallback path).
//   input - the block-stack input latent (Output-level diff + SenCache).
//   substep_* - the tap-driven capture/probe/inject passes (device or host).
struct CacheRunnerHooks {
    const sd::Tensor<float>* input = nullptr;
    std::function<sd::Tensor<float>()> full;

    // Substep-path tap-driven capture (MagCache device slot): the engine builds
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
    // Device reuse tap-driven inject (Qwen/Flux). MagCache: the engine hands over
    // a ReplaceStream GraphExtension (WeightedBlend of x_before + device slot); the
    // runner requests the region taps, weaves op->lower(), and the forward
    // substitutes it over the reuse region. The runner never learns the math is a
    // residual add. substep_inject_gpu (DiCache gamma-blend) still uses the legacy
    // hardcoded path pending its own migration.
    std::function<sd::Tensor<float>(std::vector<GraphExtension>)> substep_inject_slot;
    // DiCache device reuse: the engine hands over a ReplaceStream GraphExtension
    // (cache.gamma_blend, gamma baked into params.floats as a host constant); the
    // model reads the residual-ring operands from the CacheStateManager slots via
    // the bridge and build_stream_override weaves the blend. The runner never learns
    // the math is a gamma extrapolation. (extensions, bridge).
    std::function<sd::Tensor<float>(std::vector<GraphExtension>, DiCacheSlotBridge)> substep_inject_gpu;
};

// The declarative-era replacement for CacheController. Owns the policy, the
// model contract, the compiled program, and the cache state. Keeps the same
// init()/enabled()/begin_step()/end_step()/run_branch()/log_summary() surface so
// pipelines change only their type name and header.
//
// init() runs capability negotiation before compiling the program: an
// unsupported method fails explicitly (or, with allow_fallback, drops to
// no-cache and logs it) instead of the old silent full-compute fallback.
class CacheEngine {
public:
    CacheEngine() = default;

    // seam_available: whether the model's block-stack seam is usable this run
    // (false under SP-parallel or when the runner can't cut its stack). Drives
    // capability negotiation for Feature/Probe methods.
    // device_store: optional device backing for on-GPU residual slots (the runner
    // supplies its own; null => host-only slots, the CPU/SP/mmdit/wan path).
    // cfg_parallel: true when this run splits cond/uncond across ranks. Output
    // methods keep no shared state across ranks, so their per-rank skip decisions
    // could diverge (one rank reuses while the other computes) and silently corrupt
    // the CFG combine — init() rejects Output caching under CFG-parallel.
    bool init(const ed_sample_params_t& sample_params,
              SDVersion version,
              const std::vector<float>& sigmas,
              bool seam_available,
              ICacheDeviceStore* device_store = nullptr,
              bool cfg_parallel = false);

    bool enabled() const { return policy_ != nullptr && policy_->enabled(); }
    CacheMode mode() const { return config_.mode; }
    // Resolved DiCache probe depth (number of front blocks the probe pass runs).
    // The pipeline uses this to size the persistent GPU probe-state buffer;
    // reading it here keeps the pipeline from re-deriving it from raw params.
    int dicache_probe_depth() const { return std::max(1, config_.dicache.probe_depth); }
    // DiCache error metric: true when the delta-minus form (|delta_y - delta_x|) is
    // used, so the substep probe knows to also compute delta_x on-device.
    bool dicache_delta_minus() const {
        return config_.dicache.error_choice == DiCacheErrorChoice::DeltaMinus;
    }
    CacheGranularity granularity() const {
        return policy_ != nullptr ? policy_->requirements().granularity : CacheGranularity::Output;
    }
    bool needs_feature_seam() const {
        return enabled() && granularity() != CacheGranularity::Output;
    }
    // True when this run is calibrating a table for the active method and the
    // pipeline must drive the calibration protocol (e.g. SenCache's finite-diff
    // forwards). The method itself owns what calibration measures.
    bool needs_calibration() const {
        return policy_ != nullptr && policy_->calibration_spec().active;
    }
    // Whether the active calibration needs the pipeline to wire forward_at.
    bool calibration_needs_forward_evaluator() const {
        return policy_ != nullptr && policy_->calibration_spec().needs_forward_evaluator;
    }

    void begin_step(const CacheStepInfo& step);
    void end_step(const CacheStepInfo& step);
    void log_summary(size_t total_steps) const;

    // Drive one step/branch of the active method's calibration protocol. The
    // pipeline supplies the model prediction already computed this step and a
    // model-specific evaluator; the policy owns the measurement. No-op when not
    // calibrating.
    void calibrate(CacheBranch branch,
                   const void* condition_key,
                   const sd::Tensor<float>& latent,
                   const sd::Tensor<float>& prediction,
                   std::function<sd::Tensor<float>(const sd::Tensor<float>&, float)> forward_at);

    // Run one CFG branch this step. Falls back to hooks.full() when caching is
    // unavailable or declines to reuse.
    sd::Tensor<float> run_branch(CacheBranch branch,
                                 const void* condition_key,
                                 const CacheRunnerHooks& hooks);

private:
    StepContext make_step_context() const;

    CacheConfig config_;
    SDVersion version_ = VERSION_COUNT;
    int num_steps_ = 0;
    CacheStepInfo current_step_;

    std::unique_ptr<ICachePolicy> policy_;
    std::unique_ptr<DiTModelCacheContract> contract_;
    CacheProgram program_;
    CacheStateManager state_;
    CacheOperatorRegistry operators_;
};

// Transitional alias so existing includes keep compiling during migration.
using CacheController = CacheEngine;
using CacheRuntime = CacheEngine;

}  // namespace cache
}  // namespace edgedit
