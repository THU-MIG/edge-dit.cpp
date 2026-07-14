#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/policy/policy_common.hpp"
#include "core/optimization/cache/tensor_cache_utils.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {
namespace {

using detail::kVariantFull;
using detail::kVariantReuse;

// UCache (Output granularity): decayed accumulated-error gate with an adaptive
// threshold and relative normalization. Math ported verbatim from the old
// UCachePolicy.
class UCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "UCache"; }
    CacheMode mode() const override { return CacheMode::UCache; }
    bool enabled() const override { return initialized_ && config_.enabled; }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Output;
        r.required_sites.push_back({CacheSiteRole::DENOISER_OUTPUT, true, false});
        r.history_length = 1;
        return r;
    }

    CacheProgram compile(const ModelSchema&, const ModelTopology& topo, const InferenceConfig& inf) override {
        config_ = inf.config->ucache;
        initialized_ = config_.enabled;
        expected_total_steps_ = inf.num_steps;
        window_.configure(inf.sigmas ? *inf.sigmas : std::vector<float>{}, config_.start_percent, config_.end_percent);
        reset();
        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        const char* substep_env = std::getenv("ED_CACHE_SUBSTEP");
        substep_env_on_ = substep_env != nullptr && substep_env[0] != '\0' && substep_env[0] != '0';
        // Declarative program: residual slot + difference/blend operators driven
        // by the lowering; policy keeps only its scalar decision state.
        return detail::make_output_diff_program("UCache", seg);
    }

    void begin_step(const StepContext& step) override {
        if (!enabled() || step.step_index == current_step_index_) {
            return;
        }
        current_step_index_ = step.step_index;
        skip_current_step_ = false;
        has_last_input_change_ = false;
        step_active_ = window_.contains(step.sigma);
        if (step_active_) {
            total_active_steps_++;
        }
    }

    RuntimeDecision decide(const StepContext& step, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        if (!enabled() || step.step_index < 0 || !step_active_ || m.input == nullptr) {
            return d;
        }
        const void* cond = m.condition_key;
        if (initial_step_) {
            anchor_condition_ = cond;
            initial_step_ = false;
        }
        if (cond != anchor_condition_ || !has_prev_input_ || !has_prev_output_ || !has_cache(cond)) {
            return d;
        }
        const size_t ne = static_cast<size_t>(m.input->numel());
        if (prev_input_.size() != ne) {
            return d;
        }
        const float* input_data = m.input->data();
        last_input_change_ = 0.0f;
        for (size_t i = 0; i < ne; ++i) {
            last_input_change_ += std::fabs(input_data[i] - prev_input_[i]);
        }
        if (ne > 0) {
            last_input_change_ /= static_cast<float>(ne);
        }
        has_last_input_change_ = true;

        if (has_output_prev_norm_ && has_relative_transformation_rate_ &&
            last_input_change_ > 0.0f && output_prev_norm_ > 0.0f) {
            float approx_output_change = relative_transformation_rate_ * last_input_change_;
            float approx_output_change_rate = approx_output_change;
            if (config_.use_relative_threshold) {
                const float base_scale = std::max(output_prev_norm_, 1e-6f);
                const float dyn_scale = has_output_change_ema_
                                            ? std::max(output_change_ema_ * std::max(1.0f, config_.relative_norm_gain), 1e-6f)
                                            : base_scale;
                const float scale = std::sqrt(base_scale * dyn_scale);
                approx_output_change_rate = approx_output_change / scale;
            }
            approx_output_change_rate *= (1.0f + 0.50f * consecutive_skipped_steps_);
            accumulated_error_ = accumulated_error_ * config_.error_decay_rate + approx_output_change_rate;

            float effective_threshold = get_adaptive_threshold();
            if (!config_.use_relative_threshold && output_prev_norm_ > 0.0f) {
                effective_threshold *= output_prev_norm_;
            }
            if (accumulated_error_ < effective_threshold) {
                skip_current_step_ = true;
                total_steps_skipped_++;
                consecutive_skipped_steps_++;
                d.variant = kVariantReuse;
                return d;
            }
            if (config_.reset_error_on_compute) {
                accumulated_error_ = 0.0f;
            }
        }
        return d;
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext&) override {
        // Reuse served declaratively by the lowering (LOAD slot + BLEND input).
        return {};
    }

    // ---- Substep interface (ED_CACHE_SUBSTEP). Output method: one substep, reuse
    // or compute; decision reuses decide(). ----
    bool supports_substep() const override { return substep_env_on_; }
    void set_substep_input(const sd::Tensor<float>* input) override { substep_input_ = input; }
    void begin_substeps(const StepContext& step, const void* condition_key) override {
        begin_step(step);
        substep_done_ = false;
        substep_key_ = condition_key;
        substep_step_ = step;
    }
    std::optional<SubstepPlan> next_substep() override {
        if (substep_done_) {
            return std::nullopt;
        }
        substep_done_ = true;
        CacheRuntimeMetrics m;
        m.condition_key = substep_key_;
        m.input = substep_input_;
        const RuntimeDecision d = decide(substep_step_, m);
        SubstepPlan p;
        p.produces_output = true;
        p.op.kind = (d.variant == kVariantReuse) ? SubstepOpKind::OutputReuse
                                                 : SubstepOpKind::OutputCompute;
        return p;
    }
    void observe_substep(const SubstepResult&) override {}

    void observe(const CacheObservation& obs) override {
        if (obs.kind != CacheObservation::Kind::Feature || obs.input == nullptr || obs.feature == nullptr) {
            return;
        }
        if (!step_active_ || obs.input->numel() != obs.feature->numel()) {
            return;
        }
        const sd::Tensor<float>& input = *obs.input;
        const sd::Tensor<float>& output = *obs.feature;
        // Residual diff stored into the slot by the FULL variant's after-actions;
        // record only that a cached diff now exists for this branch.
        stored_[obs.condition_key] = true;
        if (obs.condition_key != anchor_condition_) {
            return;
        }
        steps_computed_since_active_++;
        consecutive_skipped_steps_ = 0;

        const size_t ne = static_cast<size_t>(input.numel());
        prev_input_.assign(input.data(), input.data() + ne);
        has_prev_input_ = true;

        const float* out_data = output.data();
        float output_change = 0.0f;
        if (has_prev_output_ && prev_output_.size() == ne) {
            for (size_t i = 0; i < ne; ++i) {
                output_change += std::fabs(out_data[i] - prev_output_[i]);
            }
            if (ne > 0) {
                output_change /= static_cast<float>(ne);
            }
        }
        if (std::isfinite(output_change) && output_change > 0.0f) {
            output_change_ema_ = has_output_change_ema_ ? 0.8f * output_change_ema_ + 0.2f * output_change : output_change;
            has_output_change_ema_ = true;
        }
        prev_output_.assign(out_data, out_data + ne);
        has_prev_output_ = true;
        output_prev_norm_ = detail::mean_abs(out_data, ne);
        has_output_prev_norm_ = output_prev_norm_ > 0.0f;

        if (has_last_input_change_ && last_input_change_ > 0.0f && output_change > 0.0f) {
            const float rate = output_change / last_input_change_;
            if (std::isfinite(rate)) {
                relative_transformation_rate_ = rate;
                has_relative_transformation_rate_ = true;
            }
        }
        has_last_input_change_ = false;
    }

    void end_step(const StepContext&) override {}

    void log_summary(size_t total_steps) const override {
        if (!enabled() || total_steps == 0) {
            return;
        }
        if (total_steps_skipped_ > 0 && total_steps_skipped_ < static_cast<int>(total_steps)) {
            const double speedup = static_cast<double>(total_steps) /
                                   static_cast<double>(total_steps - total_steps_skipped_);
            LOG_INFO("UCache skipped %d/%zu steps (%.2fx estimated speedup)",
                     total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("UCache skipped %d/%zu steps", total_steps_skipped_, total_steps);
        }
    }

    void reset() override {
        initial_step_ = true;
        skip_current_step_ = false;
        step_active_ = false;
        anchor_condition_ = nullptr;
        stored_.clear();
        prev_input_.clear();
        prev_output_.clear();
        output_prev_norm_ = 0.0f;
        has_prev_input_ = false;
        has_prev_output_ = false;
        has_output_prev_norm_ = false;
        has_relative_transformation_rate_ = false;
        relative_transformation_rate_ = 0.0f;
        last_input_change_ = 0.0f;
        has_last_input_change_ = false;
        output_change_ema_ = 0.0f;
        has_output_change_ema_ = false;
        total_steps_skipped_ = 0;
        current_step_index_ = -1;
        steps_computed_since_active_ = 0;
        consecutive_skipped_steps_ = 0;
        accumulated_error_ = 0.0f;
        total_active_steps_ = 0;
    }

private:
    float get_adaptive_threshold() const {
        if (!config_.adaptive_threshold) {
            return config_.reuse_threshold;
        }
        int effective_total = expected_total_steps_;
        if (effective_total <= 0) {
            effective_total = std::max(20, steps_computed_since_active_ * 2);
        }
        float progress = effective_total > 0
                             ? static_cast<float>(steps_computed_since_active_) / static_cast<float>(effective_total)
                             : 0.0f;
        progress = std::max(0.0f, std::min(1.0f, progress));
        float multiplier = 1.0f;
        if (progress < 0.2f) {
            multiplier = config_.early_step_multiplier;
        } else if (progress > 0.8f) {
            multiplier = config_.late_step_multiplier;
        }
        return config_.reuse_threshold * multiplier;
    }

    bool has_cache(const void* cond) const {
        auto it = stored_.find(cond);
        return it != stored_.end() && it->second;
    }

    UCacheConfig config_;
    detail::SigmaWindow window_;
    bool initialized_ = false;
    bool initial_step_ = true;
    bool skip_current_step_ = false;
    bool step_active_ = false;
    const void* anchor_condition_ = nullptr;
    std::unordered_map<const void*, bool> stored_;
    std::vector<float> prev_input_;
    std::vector<float> prev_output_;
    float output_prev_norm_ = 0.0f;
    bool has_prev_input_ = false;
    bool has_prev_output_ = false;
    bool has_output_prev_norm_ = false;
    bool has_relative_transformation_rate_ = false;
    float relative_transformation_rate_ = 0.0f;
    float last_input_change_ = 0.0f;
    bool has_last_input_change_ = false;
    float output_change_ema_ = 0.0f;
    bool has_output_change_ema_ = false;
    int total_steps_skipped_ = 0;
    int current_step_index_ = -1;
    int steps_computed_since_active_ = 0;
    int expected_total_steps_ = 0;
    int consecutive_skipped_steps_ = 0;
    float accumulated_error_ = 0.0f;
    int total_active_steps_ = 0;
    bool substep_env_on_ = false;
    bool substep_done_ = false;
    const void* substep_key_ = nullptr;
    const sd::Tensor<float>* substep_input_ = nullptr;
    StepContext substep_step_;
};

}  // namespace

std::unique_ptr<ICachePolicy> create_ucache_policy() { return std::make_unique<UCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
