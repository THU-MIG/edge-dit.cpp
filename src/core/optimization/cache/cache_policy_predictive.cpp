#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/cache_policy.hpp"
#include "core/optimization/cache/magcache_tables.hpp"
#include "core/optimization/cache/tensor_cache_utils.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {
namespace {

float mean_abs(const float* data, size_t ne) {
    if (data == nullptr || ne == 0) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < ne; ++i) {
        sum += std::fabs(data[i]);
    }
    return sum / static_cast<float>(ne);
}

// mean(|a - b|) / mean(|b|) — the relative-L1 metric used by both references.
float rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) {
        return 1.0f;
    }
    float num = 0.0f;
    float den = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs(a[i] - b[i]);
        den += std::fabs(b[i]);
    }
    return num / (den + 1e-6f);
}

// ---------------------------------------------------------------------------
// MagCache (Feature granularity): magnitude-ratio error accumulation.
// Reuses the block-stack residual capture/inject 1:1 with TaylorSeer — no new
// graph plumbing. Verified against github.com/Zehong-Ma/MagCache.
// ---------------------------------------------------------------------------
class MagCachePolicy final : public CachePolicy {
public:
    const char* name() const override { return "MagCache"; }
    bool enabled() const override { return initialized_ && config_.enabled; }
    CacheGranularity granularity() const override { return CacheGranularity::Feature; }

    void init(const CacheConfig& config, const CacheModelSpec& model_spec, const std::vector<float>& sigmas) override {
        config_ = config.magcache;
        version_ = model_spec.version;
        initialized_ = config_.enabled;
        num_steps_ = sigmas.size() >= 2 ? static_cast<int>(sigmas.size() - 1) : 0;
        mag_ratios_ = magcache_table_for(version_, num_steps_);
        retention_steps_ = static_cast<int>(config_.retention_ratio * num_steps_ + 0.5f);
        states_.clear();
        total_steps_skipped_ = 0;
    }

    void begin_step(const CacheStepInfo& step) override {
        current_step_index_ = step.step_index;
    }

    CacheStepDecision plan_step(const CacheForwardContext& frame) override {
        CacheStepDecision decision;  // Full by default
        if (!enabled() || current_step_index_ < retention_steps_) {
            return decision;
        }
        Branch& b = branch_for(frame.condition_key);
        if (!b.has_residual) {
            return decision;
        }

        const int cnt = current_step_index_;
        const float cur_scale = (cnt >= 0 && cnt < static_cast<int>(mag_ratios_.size()))
                                    ? mag_ratios_[static_cast<size_t>(cnt)]
                                    : 1.0f;
        // Speculatively accumulate; commit only if we actually skip so a
        // compute step resets cleanly (mirrors the reference control flow).
        const float accum_ratio = b.accumulated_ratio * cur_scale;
        const float accum_err = b.accumulated_err + std::fabs(1.0f - accum_ratio);
        const int accum_steps = b.accumulated_steps + 1;

        if (accum_err <= config_.mag_thresh && accum_steps <= config_.max_skip_steps) {
            b.accumulated_ratio = accum_ratio;
            b.accumulated_err = accum_err;
            b.accumulated_steps = accum_steps;
            decision.kind = CacheStepDecision::Kind::SkipStackReuse;
            return decision;
        }

        // Reset accumulators and recompute fully.
        b.accumulated_ratio = 1.0f;
        b.accumulated_err = 0.0f;
        b.accumulated_steps = 0;
        return decision;
    }

    void observe_feature(const CacheForwardContext& frame, const sd::Tensor<float>& feature) override {
        if (!enabled() || feature.empty()) {
            return;
        }
        Branch& b = branch_for(frame.condition_key);
        b.residual.assign(feature.data(), feature.data() + feature.numel());
        b.shape = feature.shape();
        b.has_residual = true;
    }

    sd::Tensor<float> reconstruct_feature(const CacheForwardContext& frame) override {
        Branch& b = branch_for(frame.condition_key);
        if (!b.has_residual) {
            return {};
        }
        sd::Tensor<float> out(b.shape);
        if (out.numel() != static_cast<int64_t>(b.residual.size())) {
            return {};
        }
        std::copy(b.residual.begin(), b.residual.end(), out.data());
        total_steps_skipped_++;
        return out;
    }

    void end_step(const CacheStepInfo&) override {}

    void log_summary(size_t total_steps) const override {
        if (!enabled() || total_steps == 0) {
            return;
        }
        if (total_steps_skipped_ > 0 && total_steps_skipped_ < static_cast<int>(total_steps)) {
            const double speedup = static_cast<double>(total_steps) /
                                   static_cast<double>(total_steps - total_steps_skipped_);
            LOG_INFO("MagCache reused %d/%zu steps (%.2fx)", total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("MagCache reused %d/%zu steps", total_steps_skipped_, total_steps);
        }
    }

private:
    struct Branch {
        float accumulated_ratio = 1.0f;
        float accumulated_err = 0.0f;
        int accumulated_steps = 0;
        std::vector<float> residual;
        std::vector<int64_t> shape;
        bool has_residual = false;
    };

    MagCacheConfig config_;
    SDVersion version_ = VERSION_COUNT;
    bool initialized_ = false;
    int num_steps_ = 0;
    int retention_steps_ = 0;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::vector<float> mag_ratios_;
    std::unordered_map<const void*, Branch> states_;

    Branch& branch_for(const void* cond) { return states_[cond]; }
};

// ---------------------------------------------------------------------------
// DiCache (Probe granularity): shallow-probe trajectory alignment.
// Runs the first probe_depth blocks, measures how far the probe state moved
// vs. last step, and decides skip vs. full. On skip, reconstructs the deep
// residual by linear extrapolation scaled by the probe-trajectory ratio.
// Verified against github.com/Bujiazi/DiCache (FLUX/run_flux_dicache.py).
// ---------------------------------------------------------------------------
class DiCachePolicy final : public CachePolicy {
public:
    const char* name() const override { return "DiCache"; }
    bool enabled() const override { return initialized_ && config_.enabled; }
    CacheGranularity granularity() const override { return CacheGranularity::Probe; }

    void init(const CacheConfig& config, const CacheModelSpec&, const std::vector<float>& sigmas) override {
        config_ = config.dicache;
        initialized_ = config_.enabled;
        num_steps_ = sigmas.size() >= 2 ? static_cast<int>(sigmas.size() - 1) : 0;
        retention_steps_ = static_cast<int>(config_.retention_ratio * num_steps_);
        states_.clear();
        total_steps_skipped_ = 0;
    }

    void begin_step(const CacheStepInfo& step) override {
        current_step_index_ = step.step_index;
    }

    CacheStepDecision plan_step(const CacheForwardContext& frame) override {
        CacheStepDecision decision;  // Full by default
        if (!enabled()) {
            return decision;
        }
        Branch& b = branch_for(frame.condition_key);
        // Warmup, last step, or no probe history yet -> full compute.
        if (current_step_index_ <= retention_steps_ ||
            current_step_index_ >= num_steps_ - 1 ||
            !b.has_probe_history) {
            return decision;
        }
        decision.kind = CacheStepDecision::Kind::Probe;
        decision.probe_depth = std::max(1, config_.probe_depth);
        return decision;
    }

    CacheStepDecision decide_after_probe(const CacheForwardContext& frame,
                                         const sd::Tensor<float>& before,
                                         const sd::Tensor<float>& probe) override {
        CacheStepDecision decision;  // Full by default
        Branch& b = branch_for(frame.condition_key);
        if (before.empty() || probe.empty()) {
            return decision;
        }

        // Always stash the current probe state so history is seeded even on the
        // first probe step (when there is nothing to compare against yet).
        b.cur_input.assign(before.data(), before.data() + before.numel());
        b.cur_probe.assign(probe.data(), probe.data() + probe.numel());
        b.have_cur = true;

        if (!b.has_probe_history) {
            return decision;  // no baseline yet -> compute, seed history below
        }

        const float delta_x = rel_l1(b.cur_input, b.prev_input);
        const float delta_y = rel_l1(b.cur_probe, b.prev_probe);
        const float error = std::fabs(delta_y - delta_x);
        b.accumulated_rel_l1 += error;

        if (b.accumulated_rel_l1 < config_.rel_l1_thresh) {
            decision.kind = CacheStepDecision::Kind::SkipStackReuse;
        } else {
            b.accumulated_rel_l1 = 0.0f;
        }
        return decision;
    }

    void observe_feature(const CacheForwardContext& frame, const sd::Tensor<float>& feature) override {
        if (!enabled() || feature.empty()) {
            return;
        }
        Branch& b = branch_for(frame.condition_key);
        // Deep residual window (keep last two).
        b.resid_prev2 = std::move(b.resid_prev1);
        b.resid_prev1.assign(feature.data(), feature.data() + feature.numel());
        b.have_resid2 = !b.resid_prev2.empty();
        b.shape = feature.shape();
        b.has_residual = true;
        commit_probe_history(b);
    }

    sd::Tensor<float> reconstruct_feature(const CacheForwardContext& frame) override {
        Branch& b = branch_for(frame.condition_key);
        if (!b.has_residual) {
            return {};
        }
        sd::Tensor<float> out(b.shape);
        if (out.numel() != static_cast<int64_t>(b.resid_prev1.size())) {
            return {};
        }
        float* data = out.data();

        if (b.have_resid2 && b.resid_prev2.size() == b.resid_prev1.size()) {
            // gamma-scaled linear extrapolation of the deep residual, where
            // gamma tracks the probe-trajectory ratio (clamped to [1, 1.5]).
            float gamma = 1.0f;
            if (b.have_probe_prev2 && b.probe_prev2.size() == b.cur_probe.size()) {
                const float num = rel_l1_abs(b.cur_probe, b.probe_prev2);
                const float den = rel_l1_abs(b.probe_prev1, b.probe_prev2);
                gamma = den > 1e-6f ? num / den : 1.0f;
                gamma = std::max(1.0f, std::min(1.5f, gamma));
            }
            for (size_t i = 0; i < b.resid_prev1.size(); ++i) {
                data[i] = b.resid_prev2[i] + gamma * (b.resid_prev1[i] - b.resid_prev2[i]);
            }
        } else {
            std::copy(b.resid_prev1.begin(), b.resid_prev1.end(), data);
        }

        // On a skip step the probe still ran; fold its state into history.
        commit_probe_history(b);
        total_steps_skipped_++;
        return out;
    }

    void end_step(const CacheStepInfo&) override {}

    void log_summary(size_t total_steps) const override {
        if (!enabled() || total_steps == 0) {
            return;
        }
        if (total_steps_skipped_ > 0 && total_steps_skipped_ < static_cast<int>(total_steps)) {
            const double speedup = static_cast<double>(total_steps) /
                                   static_cast<double>(total_steps - total_steps_skipped_);
            LOG_INFO("DiCache reused %d/%zu steps (%.2fx)", total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("DiCache reused %d/%zu steps", total_steps_skipped_, total_steps);
        }
    }

private:
    struct Branch {
        // probe trajectory history
        std::vector<float> prev_input;
        std::vector<float> prev_probe;
        std::vector<float> probe_prev1;  // last probe residual (probe - input)
        std::vector<float> probe_prev2;
        bool has_probe_history = false;
        bool have_probe_prev2 = false;
        // deep residual window
        std::vector<float> resid_prev1;
        std::vector<float> resid_prev2;
        bool have_resid2 = false;
        std::vector<int64_t> shape;
        bool has_residual = false;
        float accumulated_rel_l1 = 0.0f;
        // scratch for the current step (set in decide_after_probe)
        std::vector<float> cur_input;
        std::vector<float> cur_probe;
        bool have_cur = false;
    };

    static float rel_l1_abs(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || a.size() != b.size()) {
            return 0.0f;
        }
        float num = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            num += std::fabs(a[i] - b[i]);
        }
        return num / static_cast<float>(a.size());
    }

    // Roll the current step's probe state into the branch history.
    void commit_probe_history(Branch& b) {
        if (!b.have_cur) {
            return;
        }
        // probe residual = probe_state - input, tracked as a window of two.
        std::vector<float> probe_resid(b.cur_probe.size());
        if (b.cur_input.size() == b.cur_probe.size()) {
            for (size_t i = 0; i < probe_resid.size(); ++i) {
                probe_resid[i] = b.cur_probe[i] - b.cur_input[i];
            }
        } else {
            probe_resid = b.cur_probe;
        }
        b.probe_prev2 = std::move(b.probe_prev1);
        b.probe_prev1 = std::move(probe_resid);
        b.have_probe_prev2 = !b.probe_prev2.empty();

        b.prev_input = std::move(b.cur_input);
        b.prev_probe = std::move(b.cur_probe);
        b.has_probe_history = true;
        b.have_cur = false;
    }

    DiCacheConfig config_;
    bool initialized_ = false;
    int num_steps_ = 0;
    int retention_steps_ = 0;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::unordered_map<const void*, Branch> states_;

    Branch& branch_for(const void* cond) { return states_[cond]; }
};

}  // namespace

std::unique_ptr<CachePolicy> create_magcache_policy() {
    return std::make_unique<MagCachePolicy>();
}

std::unique_ptr<CachePolicy> create_dicache_policy() {
    return std::make_unique<DiCachePolicy>();
}

}  // namespace cache
}  // namespace edgedit
