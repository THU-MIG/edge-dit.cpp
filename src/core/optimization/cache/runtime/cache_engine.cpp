#include "core/optimization/cache/runtime/cache_engine.hpp"

#include <cstdlib>

#include "core/optimization/cache/runtime/capability_negotiation.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {

namespace {

const char* granularity_name(CacheGranularity g) {
    switch (g) {
        case CacheGranularity::Output: return "output";
        case CacheGranularity::Feature: return "feature";
        case CacheGranularity::Probe: return "probe";
    }
    return "?";
}

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

}  // namespace

bool CacheEngine::init(const ed_sample_params_t& sample_params,
                       SDVersion version,
                       const std::vector<float>& sigmas,
                       bool seam_available,
                       ICacheDeviceStore* device_store,
                       bool cfg_parallel) {
    config_ = cache_config_from_sample_params(sample_params);
    version_ = version;
    num_steps_ = sigmas.size() >= 2 ? static_cast<int>(sigmas.size() - 1) : 0;

    policy_ = create_cache_policy(config_.mode);
    if (policy_ == nullptr) {
        return false;
    }

    contract_ = std::make_unique<DiTModelCacheContract>(version, seam_available);

    // Capability negotiation BEFORE compile: an unsupported method is reported
    // explicitly instead of silently degrading to full compute.
    const CacheRequirements reqs = policy_->requirements();
    const ValidationResult validation = validate_requirements(reqs, *contract_);
    if (!validation.ok) {
        if (config_.mode != CacheMode::Disabled) {
            LOG_WARN("cache disabled: %s", validation.message.c_str());
        }
        policy_.reset();
        contract_.reset();
        return false;
    }

    // Reject Output-granularity caching under CFG-parallel. Feature/Probe methods
    // are already refused above (seam_available=false), but Output methods need no
    // seam and would otherwise run per-rank. rank0 (uncond) and rank1 (cond) hold
    // independent policy state and decide skips from their own branch's metrics, so
    // they can diverge (one reuses while the other computes); the CFG combine then
    // mixes a stale-cached branch with a fresh one and silently drifts. Disable
    // explicitly rather than corrupt output — matches the warn-and-disable contract.
    if (cfg_parallel && reqs.granularity == CacheGranularity::Output &&
        config_.mode != CacheMode::Disabled) {
        LOG_WARN("cache disabled: Output-granularity caching (mode=%s) is not supported "
                 "under CFG-parallel (per-rank skip decisions could diverge across the "
                 "cond/uncond ranks and corrupt the CFG combine).",
                 cache_mode_name(config_.mode));
        policy_.reset();
        contract_.reset();
        return false;
    }

    InferenceConfig inf;
    inf.config = &config_;
    inf.sigmas = &sigmas;
    inf.num_steps = num_steps_;
    inf.separate_cfg = contract_->schema().family == ModelFamily::MMDiT ||
                       contract_->schema().family == ModelFamily::WanVideo;

    program_ = policy_->compile(contract_->schema(), contract_->topology(), inf);
    // Wire the device store BEFORE initialize() so device_backed slots allocate
    // on-device. Null store (CPU/SP/mmdit/wan) leaves every slot host-backed.
    state_.set_device_store(device_store);
    state_.initialize(program_.slots);
    register_builtin_cache_operators(&operators_);

    if (!policy_->enabled()) {
        // e.g. SenCache with no usable profile: compile() logged the reason.
        policy_.reset();
        contract_.reset();
        return false;
    }

    LOG_INFO("cache enabled: mode=%s model=%s granularity=%s matched_sites=%zu variants=%zu slots=%zu",
             cache_mode_name(config_.mode),
             cache_model_spec_for_version(version).model_name.c_str(),
             granularity_name(granularity()),
             reqs.required_sites.size(),
             program_.variants.size(),
             program_.slots.size());

    if (env_flag("ED_DUMP_CACHE_PROGRAM")) {
        const std::string dump = dump_cache_program(program_);
        LOG_INFO("cache program:\n%s", dump.c_str());
    }
    return true;
}

StepContext CacheEngine::make_step_context() const {
    StepContext s;
    s.step_index = current_step_.step_index;
    s.num_steps = current_step_.num_steps;
    s.sigma = current_step_.sigma;
    s.sigma_next = current_step_.sigma_next;
    s.is_first_step = current_step_.step_index == 0;
    s.is_last_step = current_step_.num_steps > 0 &&
                     current_step_.step_index == current_step_.num_steps - 1;
    return s;
}

void CacheEngine::begin_step(const CacheStepInfo& step) {
    current_step_ = step;
    state_.begin_step(step.step_index);
    if (policy_ != nullptr) {
        policy_->begin_step(make_step_context());
    }
}

void CacheEngine::end_step(const CacheStepInfo& step) {
    if (policy_ != nullptr) {
        policy_->end_step(make_step_context());
    }
    state_.commit_step(step.step_index);
}

void CacheEngine::log_summary(size_t total_steps) const {
    if (policy_ != nullptr) {
        policy_->log_summary(total_steps);
    }
}

void CacheEngine::calibrate(CacheBranch branch,
                            const void* condition_key,
                            const sd::Tensor<float>& latent,
                            const sd::Tensor<float>& prediction,
                            std::function<sd::Tensor<float>(const sd::Tensor<float>&, float)> forward_at) {
    if (policy_ == nullptr || !policy_->calibration_spec().active) {
        return;
    }
    CalibrationContext ctx;
    ctx.step = make_step_context();
    ctx.condition_key = condition_key;
    ctx.branch = branch;
    ctx.latent = &latent;
    ctx.prediction = &prediction;
    ctx.forward_at = std::move(forward_at);
    policy_->calibrate_step(ctx);
}

sd::Tensor<float> CacheEngine::run_branch(CacheBranch branch,
                                          const void* condition_key,
                                          const CacheRunnerHooks& hooks) {
    if (policy_ == nullptr || !policy_->enabled()) {
        return hooks.full();
    }
    const StepContext step = make_step_context();

    // Substep path (ED_CACHE_SUBSTEP): drive the policy's next_substep() loop,
    // translating each SubstepPlan into the existing runner hooks. Gated + only for
    // policies that opted in (supports_substep). Unmigrated methods keep the legacy
    // decide()/execute() path below unchanged.
    static const bool substep_path = env_flag("ED_CACHE_SUBSTEP");
    if (substep_path && policy_->supports_substep()) {
        return CacheGraphLowering::execute_substeps(*policy_, program_, step,
                                                    condition_key, branch, hooks,
                                                    state_, operators_);
    }

    CacheRuntimeMetrics metrics;
    metrics.branch = branch;
    metrics.condition_key = condition_key;
    metrics.input = hooks.input;

    const RuntimeDecision decision = policy_->decide(step, metrics);
    return CacheGraphLowering::execute(*policy_, program_, decision, step,
                                       condition_key, branch, hooks, state_, operators_);
}

}  // namespace cache
}  // namespace edgedit
