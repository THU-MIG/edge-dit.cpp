#include <algorithm>
#include <cmath>
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
using detail::TaylorSeerState;

const char* method_label_for_mode(CacheMode mode) {
    switch (mode) {
        case CacheMode::DBCache: return "DBCache";
        case CacheMode::CacheDiT: return "CacheDiT";
        default: return cache_mode_name(mode);
    }
}

// DBCache + CacheDiT: residual-diff gated whole-model reuse, optionally with a
// Taylor extrapolation of the whole output (CacheDiT). Output granularity.
// Control flow ported verbatim from the old ConditionCachePolicy: begin_step
// computes the per-step gates, decide() applies the residual-diff test, and
// reconstruct() serves either the Taylor extrapolation or the cached diff.
class ConditionCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return method_label_for_mode(mode_); }
    CacheMode mode() const override { return mode_; }
    bool enabled() const override { return initialized_ && (db_config_.enabled || taylor_config_.enabled); }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Output;
        r.required_sites.push_back({CacheSiteRole::DENOISER_OUTPUT, true, false});
        r.history_length = 1;
        return r;
    }

    CacheProgram compile(const ModelSchema&, const ModelTopology& topo, const InferenceConfig& inf) override {
        mode_ = inf.config->mode;
        db_config_ = inf.config->dbcache;
        taylor_config_ = inf.config->taylorseer;
        initialized_ = db_config_.enabled || taylor_config_.enabled;
        window_.configure(inf.sigmas ? *inf.sigmas : std::vector<float>{},
                          db_config_.enabled ? db_config_.start_percent : taylor_config_.start_percent,
                          db_config_.enabled ? db_config_.end_percent : taylor_config_.end_percent);
        reset();
        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        // DBCache mode (no Taylor) is pure diff-reuse: drive it declaratively so
        // the residual slot + difference/blend operators are load-bearing. CacheDiT
        // mode additionally extrapolates the whole output (Taylor), which the
        // declarative Output path can't serve yet, so it stays on the legacy
        // reconstruct()/observe() callback path.
        declarative_ = !taylor_config_.enabled;
        if (declarative_) {
            return detail::make_output_diff_program(method_label_for_mode(mode_), seg);
        }
        return detail::make_reuse_program(method_label_for_mode(mode_), seg,
                                          SegmentExecutionMode::LOAD_CACHED,
                                          detail::make_slot(0, "denoiser_output_diff"));
    }

    void begin_step(const StepContext& step) override {
        if (!enabled() || step.step_index == current_step_index_) {
            return;
        }
        current_step_index_ = step.step_index;
        skip_current_step_ = false;
        step_active_ = false;
        warmup_step_ = false;
        can_cache_this_step_ = false;
        static_cache_step_ = false;

        if (!window_.contains(step.sigma)) {
            return;
        }
        step_active_ = true;

        if (!db_config_.enabled) {
            can_cache_this_step_ = true;
            return;
        }
        const bool in_warmup = warmup_remaining_ > 0;
        if (in_warmup) {
            warmup_remaining_--;
        }
        bool scm_allows_cache = true;
        if (!db_config_.steps_computation_mask.empty() &&
            step.step_index < static_cast<int>(db_config_.steps_computation_mask.size())) {
            scm_allows_cache = db_config_.steps_computation_mask[step.step_index] == 0;
            static_cache_step_ = scm_allows_cache && !db_config_.scm_policy_dynamic;
            if (static_cache_step_) {
                can_cache_this_step_ = true;
                return;
            }
        }
        if (in_warmup) {
            warmup_step_ = true;
            return;
        }
        if (db_config_.max_cached_steps >= 0 &&
            static_cast<int>(cached_steps_.size()) >= db_config_.max_cached_steps) {
            return;
        }
        if (db_config_.max_continuous_cached_steps >= 0 &&
            continuous_cached_steps_ >= db_config_.max_continuous_cached_steps) {
            return;
        }
        if (db_config_.max_accumulated_residual_diff >= 0.0f &&
            accumulated_residual_diff_ >= db_config_.max_accumulated_residual_diff) {
            return;
        }
        can_cache_this_step_ = scm_allows_cache;
    }

    RuntimeDecision decide(const StepContext& step, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        pending_taylor_ = false;
        if (!enabled() || step.step_index < 0 || !step_active_ ||
            (warmup_step_ && !static_cache_step_) || m.input == nullptr) {
            return d;
        }
        const void* cond = m.condition_key;
        if (initial_step_) {
            anchor_condition_ = cond;
            initial_step_ = false;
        }

        // CacheDiT / taylor-only path: extrapolate the whole output on schedule.
        if ((mode_ == CacheMode::TaylorSeer || !db_config_.enabled) && taylor_config_.enabled &&
            can_cache_this_step_ && should_use_taylor_this_step() && cond == anchor_condition_ &&
            taylor_ready(cond)) {
            skip_current_step_ = true;
            total_steps_skipped_++;
            record_cached_step(0.0f);
            pending_taylor_ = true;
            d.variant = kVariantReuse;
            return d;
        }

        if (!db_config_.enabled || !can_cache_this_step_ || cond != anchor_condition_) {
            return d;
        }
        auto it = cache_entries_.find(cond);
        if (it == cache_entries_.end() || !it->second.has_prev) {
            return d;
        }
        const size_t ne = static_cast<size_t>(m.input->numel());
        if (it->second.prev_input.size() != ne) {
            return d;
        }
        const float diff = residual_diff(it->second.prev_input.data(), m.input->data(), ne);
        if (static_cache_step_ || diff < db_config_.residual_diff_threshold) {
            skip_current_step_ = true;
            total_steps_skipped_++;
            record_cached_step(diff);
            pending_taylor_ = taylor_config_.enabled && taylor_ready(cond);
            d.variant = kVariantReuse;
            return d;
        }
        continuous_cached_steps_ = 0;
        return d;
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext& ctx) override {
        if (ctx.input == nullptr) {
            return {};
        }
        // Prefer Taylor extrapolation when it was selected this step (CacheDiT).
        if (pending_taylor_) {
            auto it = taylor_states_.find(ctx.condition_key);
            if (it != taylor_states_.end()) {
                sd::Tensor<float> out;
                if (it->second.approximate(&out, ctx.input->shape(), current_step_index_)) {
                    return out;
                }
            }
        }
        // Declarative (DBCache) mode: the diff reuse is served by the lowering's
        // LOAD slot + BLEND actions, not here.
        if (declarative_) {
            return {};
        }
        sd::Tensor<float> out;
        auto it = cache_entries_.find(ctx.condition_key);
        if (it == cache_entries_.end() || !apply_tensor_diff(it->second.diff, *ctx.input, &out)) {
            return {};
        }
        return out;
    }

    void observe(const CacheObservation& obs) override {
        if (obs.kind != CacheObservation::Kind::Feature || obs.input == nullptr || obs.feature == nullptr) {
            return;
        }
        if (!step_active_ || obs.input->numel() != obs.feature->numel()) {
            return;
        }
        update_cache(obs.condition_key, *obs.input, *obs.feature);
        if (taylor_config_.enabled) {
            TaylorSeerState& state = taylor_state_for(obs.condition_key);
            state.update_derivatives(obs.feature->data(),
                                     static_cast<size_t>(obs.feature->numel()), current_step_index_);
        }
    }

    void end_step(const StepContext&) override {
        if (skip_current_step_) {
            return;
        }
        continuous_cached_steps_ = 0;
    }

    void log_summary(size_t total_steps) const override {
        if (!enabled() || total_steps == 0) {
            return;
        }
        if (total_steps_skipped_ > 0 && total_steps_skipped_ < static_cast<int>(total_steps)) {
            const double speedup = static_cast<double>(total_steps) /
                                   static_cast<double>(total_steps - total_steps_skipped_);
            LOG_INFO("%s skipped %d/%zu steps (%.2fx estimated speedup), accum_diff=%.4f",
                     method_label_for_mode(mode_), total_steps_skipped_, total_steps, speedup, accumulated_residual_diff_);
        } else {
            LOG_INFO("%s skipped %d/%zu steps, accum_diff=%.4f",
                     method_label_for_mode(mode_), total_steps_skipped_, total_steps, accumulated_residual_diff_);
        }
    }

    void reset() override {
        current_step_index_ = -1;
        step_active_ = false;
        warmup_step_ = false;
        can_cache_this_step_ = false;
        static_cache_step_ = false;
        skip_current_step_ = false;
        pending_taylor_ = false;
        initial_step_ = true;
        warmup_remaining_ = db_config_.max_warmup_steps;
        cached_steps_.clear();
        continuous_cached_steps_ = 0;
        accumulated_residual_diff_ = 0.0f;
        total_steps_skipped_ = 0;
        anchor_condition_ = nullptr;
        cache_entries_.clear();
        taylor_states_.clear();
    }

private:
    struct CacheEntry {
        std::vector<float> diff;
        std::vector<float> prev_input;
        std::vector<float> prev_output;
        bool has_prev = false;
    };

    void update_cache(const void* cond, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        CacheEntry& entry = cache_entries_[cond];
        // In declarative mode the residual diff lives in the CacheStateManager slot
        // (written by the FULL variant's after-actions), so the policy only tracks
        // prev_input/output for the residual-diff decision gate. In legacy mode it
        // also keeps the diff vector for reconstruct().
        if (declarative_) {
            const size_t in = static_cast<size_t>(input.numel());
            const size_t on = static_cast<size_t>(output.numel());
            if (in == 0 || in != on) {
                entry.prev_input.clear();
                entry.prev_output.clear();
                entry.has_prev = false;
                return;
            }
            entry.prev_input.assign(input.data(), input.data() + in);
            entry.prev_output.assign(output.data(), output.data() + on);
            entry.has_prev = true;
            return;
        }
        if (!store_tensor_diff(&entry.diff, input, output)) {
            entry.prev_input.clear();
            entry.prev_output.clear();
            entry.has_prev = false;
            return;
        }
        const size_t size = static_cast<size_t>(output.numel());
        entry.prev_input.assign(input.data(), input.data() + size);
        entry.prev_output.assign(output.data(), output.data() + size);
        entry.has_prev = true;
    }

    TaylorSeerState& taylor_state_for(const void* cond) {
        TaylorSeerState& state = taylor_states_[cond];
        if (state.dy_current.empty()) {
            state.init(taylor_config_.n_derivatives);
        }
        return state;
    }

    bool taylor_ready(const void* cond) const {
        auto it = taylor_states_.find(cond);
        return it != taylor_states_.end() && it->second.ready();
    }

    void record_cached_step(float residual) {
        cached_steps_.push_back(current_step_index_);
        continuous_cached_steps_++;
        accumulated_residual_diff_ += residual;
    }

    bool should_use_taylor_this_step() const {
        if (!taylor_config_.enabled || current_step_index_ < taylor_config_.max_warmup_steps) {
            return false;
        }
        int interval = taylor_config_.skip_interval_steps;
        if (interval <= 0) {
            interval = 1;
        }
        return (current_step_index_ % (interval + 1)) != 0;
    }

    CacheMode mode_ = CacheMode::CacheDiT;
    bool declarative_ = false;
    DBCacheConfig db_config_;
    TaylorSeerConfig taylor_config_;
    detail::SigmaWindow window_;
    bool initialized_ = false;
    int current_step_index_ = -1;
    bool step_active_ = false;
    bool warmup_step_ = false;
    bool can_cache_this_step_ = false;
    bool static_cache_step_ = false;
    bool skip_current_step_ = false;
    bool pending_taylor_ = false;
    bool initial_step_ = true;
    int warmup_remaining_ = 0;
    std::vector<int> cached_steps_;
    int continuous_cached_steps_ = 0;
    float accumulated_residual_diff_ = 0.0f;
    int total_steps_skipped_ = 0;
    const void* anchor_condition_ = nullptr;
    std::unordered_map<const void*, CacheEntry> cache_entries_;
    std::unordered_map<const void*, TaylorSeerState> taylor_states_;
};

}  // namespace

std::unique_ptr<ICachePolicy> create_condition_policy() { return std::make_unique<ConditionCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
