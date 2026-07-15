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

// Standard program for an Output-granularity method: a black-box FULL variant
// and a REUSE variant that serves the whole denoiser output from cache.
CacheProgram make_output_program(const char* method, int block_segment_id) {
    return detail::make_reuse_program(method, block_segment_id,
                                      SegmentExecutionMode::LOAD_CACHED,
                                      detail::make_slot(0, "denoiser_output_diff"));
}

constexpr GraphVariantId kFull = kVariantFull;
constexpr GraphVariantId kReuse = kVariantReuse;

// ---------------------------------------------------------------------------
// No-cache.
// ---------------------------------------------------------------------------
class NullCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "None"; }
    CacheMode mode() const override { return CacheMode::Disabled; }
    bool enabled() const override { return false; }
    CacheRequirements requirements() const override { return {}; }
    CacheProgram compile(const ModelSchema&, const ModelTopology&, const InferenceConfig&) override {
        return make_output_program("None", 1);
    }
    void begin_step(const StepContext&) override {}
    void end_step(const StepContext&) override {}
    RuntimeDecision decide(const StepContext&, const CacheRuntimeMetrics&) override {
        RuntimeDecision d;
        d.variant = kFull;
        return d;
    }
    void reset() override {}
};

// ---------------------------------------------------------------------------
// EasyCache (Output granularity): cumulative approx-output-change-rate gate.
// Math ported verbatim from the old EasyCachePolicy.
// ---------------------------------------------------------------------------
class EasyCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "EasyCache"; }
    CacheMode mode() const override { return CacheMode::EasyCache; }
    bool enabled() const override { return initialized_ && config_.enabled; }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Output;
        r.required_sites.push_back({CacheSiteRole::DENOISER_OUTPUT, true, false});
        r.history_length = 1;
        return r;
    }

    CacheProgram compile(const ModelSchema&, const ModelTopology& topo, const InferenceConfig& inf) override {
        config_ = inf.config->easycache;
        initialized_ = config_.enabled;
        window_.configure(inf.sigmas ? *inf.sigmas : std::vector<float>{}, config_.start_percent, config_.end_percent);
        reset();
        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        // Declarative program: the residual slot + difference/blend operators are
        // driven by the lowering. This policy now only owns its scalar decision
        // state (prev input/output metrics); the cached diff tensor lives in the
        // CacheStateManager slot, not in this object.
        return detail::make_output_diff_program("EasyCache", seg);
    }

    void begin_step(const StepContext& step) override {
        if (!enabled() || step.step_index == current_step_index_) {
            return;
        }
        current_step_index_ = step.step_index;
        skip_current_step_ = false;
        has_last_input_change_ = false;
        step_active_ = window_.contains(step.sigma);
    }

    RuntimeDecision decide(const StepContext& step, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kFull;
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
            const float approx_output_change_rate =
                (relative_transformation_rate_ * last_input_change_) / output_prev_norm_;
            cumulative_change_rate_ += approx_output_change_rate;
            if (cumulative_change_rate_ < config_.reuse_threshold) {
                skip_current_step_ = true;
                total_steps_skipped_++;
                d.variant = kReuse;
                return d;
            }
            cumulative_change_rate_ = 0.0f;
        }
        return d;
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext&) override {
        // Reuse is served declaratively by the lowering (LOAD slot + BLEND with
        // input). The policy no longer reconstructs on the host.
        return {};
    }

    // ---- Substep interface. Output method: one substep per step, reuse
    // (whole-output diff) or compute; decision reuses decide(). ----
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
        p.op.kind = (d.variant == kReuse) ? SubstepOpKind::OutputReuse
                                          : SubstepOpKind::OutputCompute;
        return p;
    }

    void observe_substep(const SubstepResult&) override {}
    void set_substep_input(const sd::Tensor<float>* input) override { substep_input_ = input; }

    void observe(const CacheObservation& obs) override {
        // Output methods observe via the (input, output) pair on a full step.
        if (obs.kind != CacheObservation::Kind::Feature || obs.input == nullptr || obs.feature == nullptr) {
            return;
        }
        if (!step_active_ || obs.input->numel() != obs.feature->numel()) {
            return;
        }
        const sd::Tensor<float>& input = *obs.input;
        const sd::Tensor<float>& output = *obs.feature;
        // The residual diff itself is stored into the cache slot by the FULL
        // variant's declarative after-actions; here we only record that a cached
        // diff now exists for this branch (replaces the old cache_diffs_ map) and
        // update the scalar decision metrics.
        stored_[obs.condition_key] = true;
        if (obs.condition_key != anchor_condition_) {
            return;
        }
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
        cumulative_change_rate_ = 0.0f;
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
            LOG_INFO("EasyCache skipped %d/%zu steps (%.2fx estimated speedup)",
                     total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("EasyCache skipped %d/%zu steps", total_steps_skipped_, total_steps);
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
        cumulative_change_rate_ = 0.0f;
        last_input_change_ = 0.0f;
        has_last_input_change_ = false;
        total_steps_skipped_ = 0;
        current_step_index_ = -1;
    }

private:
    bool has_cache(const void* cond) const {
        auto it = stored_.find(cond);
        return it != stored_.end() && it->second;
    }

    EasyCacheConfig config_;
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
    float cumulative_change_rate_ = 0.0f;
    float last_input_change_ = 0.0f;
    bool has_last_input_change_ = false;
    int total_steps_skipped_ = 0;
    int current_step_index_ = -1;
    bool substep_done_ = false;
    const void* substep_key_ = nullptr;
    const sd::Tensor<float>* substep_input_ = nullptr;
    StepContext substep_step_;
};

}  // namespace

std::unique_ptr<ICachePolicy> create_null_policy() { return std::make_unique<NullCachePolicy>(); }
std::unique_ptr<ICachePolicy> create_easycache_policy() { return std::make_unique<EasyCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
