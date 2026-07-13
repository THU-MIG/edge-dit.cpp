#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "core/optimization/cache/compile/cache_graph_lowering.hpp"  // CacheRunnerHooks
#include "core/optimization/cache/ir/cache_program.hpp"
#include "core/optimization/cache/model/dit_model_cache_contract.hpp"
#include "core/optimization/cache/operator/cache_operator_registry.hpp"
#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/state/cache_state_manager.hpp"
#include "core/optimization/cache/cache_config.hpp"
#include "core/optimization/cache/cache_types.hpp"

namespace edgedit {
namespace cache {

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
    bool init(const ed_sample_params_t& sample_params,
              SDVersion version,
              const std::vector<float>& sigmas,
              bool seam_available,
              ICacheDeviceStore* device_store = nullptr);

    bool enabled() const { return policy_ != nullptr && policy_->enabled(); }
    CacheMode mode() const { return config_.mode; }
    // Resolved DiCache probe depth (number of front blocks the probe pass runs).
    // The pipeline uses this to size the persistent GPU probe-state buffer;
    // reading it here keeps the pipeline from re-deriving it from raw params.
    int dicache_probe_depth() const { return std::max(1, config_.dicache.probe_depth); }
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
