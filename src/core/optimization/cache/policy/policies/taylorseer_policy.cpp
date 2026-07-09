#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/policy/policy_common.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {
namespace {

using detail::kVariantFull;
using detail::kVariantPredict;
using detail::TaylorSeerState;

// TaylorSeer as a FEATURE-level policy: extrapolates the block-stack residual
// captured by the model seam. Skip decision is a fixed warmup+interval schedule,
// so it is deterministic across CFG-parallel ranks. Math ported verbatim from
// the old TaylorSeerFeaturePolicy.
class TaylorSeerFeaturePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "TaylorSeer"; }
    CacheMode mode() const override { return CacheMode::TaylorSeer; }
    bool enabled() const override { return initialized_ && config_.enabled; }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Feature;
        r.required_sites.push_back({CacheSiteRole::BLOCK_STACK_OUTPUT, true, false});
        r.history_length = std::max(1, config_.n_derivatives + 1);
        return r;
    }

    CacheProgram compile(const ModelSchema&, const ModelTopology& topo, const InferenceConfig& inf) override {
        config_ = inf.config->taylorseer;
        initialized_ = config_.enabled;
        window_.configure(inf.sigmas ? *inf.sigmas : std::vector<float>{}, config_.start_percent, config_.end_percent);
        reset();
        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        return detail::make_reuse_program("TaylorSeer", seg, SegmentExecutionMode::PREDICT_FROM_HISTORY,
                                          detail::make_slot(0, "block_stack_residual",
                                                            std::max(1, config_.n_derivatives + 1)));
    }

    void begin_step(const StepContext& step) override {
        current_step_index_ = step.step_index;
        step_active_ = window_.contains(step.sigma);
    }

    RuntimeDecision decide(const StepContext&, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        if (!enabled() || !step_active_ || !should_extrapolate_this_step()) {
            return d;
        }
        auto it = states_.find(m.condition_key);
        if (it == states_.end() || !it->second.ready()) {
            return d;  // not enough history yet -> full compute
        }
        d.variant = kVariantPredict;
        return d;
    }

    void observe(const CacheObservation& obs) override {
        if (obs.kind != CacheObservation::Kind::Feature || obs.feature == nullptr || obs.feature->empty()) {
            return;
        }
        if (!enabled()) {
            return;
        }
        TaylorSeerState& state = state_for(obs.condition_key);
        state.update_derivatives(obs.feature->data(), static_cast<size_t>(obs.feature->numel()), current_step_index_);
        last_feature_shape_[obs.condition_key] = obs.feature->shape();
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext& ctx) override {
        auto it = states_.find(ctx.condition_key);
        auto shape_it = last_feature_shape_.find(ctx.condition_key);
        if (it == states_.end() || shape_it == last_feature_shape_.end()) {
            return {};
        }
        sd::Tensor<float> out;
        if (!it->second.approximate(&out, shape_it->second, current_step_index_)) {
            return {};
        }
        total_steps_skipped_++;
        return out;
    }

    void end_step(const StepContext&) override {}

    void log_summary(size_t total_steps) const override {
        if (!enabled() || total_steps == 0) {
            return;
        }
        if (total_steps_skipped_ > 0 && total_steps_skipped_ < static_cast<int>(total_steps)) {
            const double speedup = static_cast<double>(total_steps) /
                                   static_cast<double>(total_steps - total_steps_skipped_);
            LOG_INFO("TaylorSeer reused %d/%zu steps (%.2fx)", total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("TaylorSeer reused %d/%zu steps", total_steps_skipped_, total_steps);
        }
    }

    void reset() override {
        step_active_ = false;
        current_step_index_ = -1;
        total_steps_skipped_ = 0;
        states_.clear();
        last_feature_shape_.clear();
    }

private:
    TaylorSeerState& state_for(const void* cond) {
        TaylorSeerState& state = states_[cond];
        if (state.dy_current.empty()) {
            state.init(config_.n_derivatives);
        }
        return state;
    }

    bool should_extrapolate_this_step() const {
        if (current_step_index_ < config_.max_warmup_steps) {
            return false;
        }
        int interval = config_.skip_interval_steps;
        if (interval <= 0) {
            interval = 1;
        }
        return (current_step_index_ % (interval + 1)) != 0;
    }

    TaylorSeerConfig config_;
    detail::SigmaWindow window_;
    bool initialized_ = false;
    bool step_active_ = false;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::unordered_map<const void*, TaylorSeerState> states_;
    std::unordered_map<const void*, std::vector<int64_t>> last_feature_shape_;
};

}  // namespace

std::unique_ptr<ICachePolicy> create_taylorseer_policy() { return std::make_unique<TaylorSeerFeaturePolicy>(); }

}  // namespace cache
}  // namespace edgedit
