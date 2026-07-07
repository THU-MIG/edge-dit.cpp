#include "core/optimization/cache/cache_runtime.hpp"

#include "utils/util.h"

namespace edgedit {
namespace cache {

bool CacheController::init(const ed_sample_params_t& sample_params,
                          SDVersion version,
                          const std::vector<float>& sigmas) {
    config_ = cache_config_from_sample_params(sample_params);
    model_spec_ = cache_model_spec_for_version(version);
    policy_ = create_cache_policy(config_.mode);
    if (policy_ == nullptr) {
        return false;
    }

    policy_->init(config_, model_spec_, sigmas);
    if (!policy_->enabled()) {
        policy_.reset();
        return false;
    }

    CacheRunInfo run;
    run.num_steps = sigmas.size() >= 2 ? static_cast<int>(sigmas.size() - 1) : 0;
    run.num_blocks = model_spec_.regions.empty() ? 0 : model_spec_.regions.front().block_count;
    run.separate_cfg = model_spec_.separate_cfg;
    run.version = version;
    policy_->begin_run(run);

    LOG_INFO("cache enabled: mode=%s model=%s granularity=%s",
             cache_mode_name(config_.mode),
             model_spec_.model_name.c_str(),
             granularity() == CacheGranularity::Output ? "output"
             : granularity() == CacheGranularity::Feature ? "feature"
                                                          : "probe");
    return true;
}

void CacheController::begin_step(const CacheStepInfo& step) {
    current_step_ = step;
    if (policy_ != nullptr) {
        policy_->begin_step(step);
    }
}

void CacheController::end_step(const CacheStepInfo& step) {
    if (policy_ != nullptr) {
        policy_->end_step(step);
    }
}

void CacheController::log_summary(size_t total_steps) const {
    if (policy_ != nullptr) {
        policy_->log_summary(total_steps);
    }
}

sd::Tensor<float> CacheController::run_branch(CacheBranch branch,
                                             const void* condition_key,
                                             const CacheRunnerHooks& hooks) {
    if (policy_ == nullptr || !policy_->enabled()) {
        return hooks.full();
    }

    CacheForwardContext frame;
    frame.step = current_step_;
    frame.branch = branch;
    frame.condition_key = condition_key;

    switch (policy_->granularity()) {
        case CacheGranularity::Feature:
            return run_feature_branch(frame, hooks);
        case CacheGranularity::Probe:
            return run_probe_branch(frame, hooks);
        case CacheGranularity::Output:
        default:
            return run_output_branch(frame, hooks);
    }
}

// Output granularity: the classic black-box path. before_forward may serve the
// step from cache; otherwise full-compute and record the whole-model residual.
sd::Tensor<float> CacheController::run_output_branch(const CacheForwardContext& frame,
                                                     const CacheRunnerHooks& hooks) {
    sd::Tensor<float> output;
    if (hooks.input != nullptr &&
        policy_->before_forward(frame, *hooks.input, &output)) {
        return output;
    }
    output = hooks.full();
    if (!output.empty() && hooks.input != nullptr) {
        policy_->after_forward(frame, *hooks.input, output);
    }
    return output;
}

// Feature granularity (TaylorSeer, MagCache): reuse the cached region's
// residual. plan_step decides skip vs. full and picks the region [start, end)
// (defaulting to the whole stack); skip injects a reconstructed residual over
// that region, full captures the fresh residual for the policy to observe.
sd::Tensor<float> CacheController::run_feature_branch(const CacheForwardContext& frame,
                                                      const CacheRunnerHooks& hooks) {
    if (!hooks.feature_supported || !hooks.capture || !hooks.inject) {
        return hooks.full();  // model can't cut into its stack this step
    }

    const CacheStepDecision decision = policy_->plan_step(frame);
    if (decision.kind == CacheStepDecision::Kind::SkipStackReuse) {
        sd::Tensor<float> feature = policy_->reconstruct_feature(frame);
        if (!feature.empty()) {
            sd::Tensor<float> output =
                hooks.inject(feature, decision.region_start, decision.region_end);
            if (!output.empty()) {
                return output;
            }
        }
        // Reconstruction unavailable -> fall through to a capturing full step.
    }

    sd::DiffusionCacheResult res = hooks.capture(decision.region_start, decision.region_end);
    if (res.output.empty()) {
        return hooks.full();  // capture failed; last-resort plain compute
    }
    if (!res.feature.empty()) {
        policy_->observe_feature(frame, res.feature);
    }
    return std::move(res.output);
}

// Probe granularity (DiCache): run a shallow prefix, decide, then skip or full.
sd::Tensor<float> CacheController::run_probe_branch(const CacheForwardContext& frame,
                                                    const CacheRunnerHooks& hooks) {
    if (!hooks.feature_supported || !hooks.capture || !hooks.inject || !hooks.probe) {
        return hooks.full();
    }

    const CacheStepDecision decision = policy_->plan_step(frame);
    if (decision.kind == CacheStepDecision::Kind::Probe && decision.probe_depth > 0) {
        sd::DiffusionCacheResult probe_res = hooks.probe(decision.probe_depth);
        const CacheStepDecision after =
            policy_->decide_after_probe(frame, probe_res.before, probe_res.probe);
        if (after.kind == CacheStepDecision::Kind::SkipStackReuse) {
            sd::Tensor<float> feature = policy_->reconstruct_feature(frame);
            if (!feature.empty()) {
                sd::Tensor<float> output =
                    hooks.inject(feature, after.region_start, after.region_end);
                if (!output.empty()) {
                    return output;
                }
            }
        }
    }

    // Full compute with capture (also seeds probe/deep-residual history).
    sd::DiffusionCacheResult res = hooks.capture(decision.region_start, decision.region_end);
    if (res.output.empty()) {
        return hooks.full();
    }
    if (!res.feature.empty()) {
        policy_->observe_feature(frame, res.feature);
    }
    return std::move(res.output);
}

}  // namespace cache
}  // namespace edgedit
