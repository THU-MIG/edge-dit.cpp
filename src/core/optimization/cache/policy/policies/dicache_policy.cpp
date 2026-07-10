#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/policy/policy_common.hpp"
#include "utils/util.h"
#include "ggml.h"

namespace edgedit {
namespace cache {
namespace {

using detail::kVariantFull;
using detail::kVariantProbe;
using detail::kVariantReuse;
using detail::rel_l1;
using detail::rel_l1_abs;
using detail::rel_l1_ptr;

// ===========================================================================
// DiCache (Probe granularity): shallow-probe trajectory alignment. Math ported
// verbatim from the old DiCachePolicy. Verified against Bujiazi/DiCache.
// ===========================================================================
class DiCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "DiCache"; }
    CacheMode mode() const override { return CacheMode::DiCache; }
    bool enabled() const override { return initialized_ && config_.enabled; }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Probe;
        r.required_sites.push_back({CacheSiteRole::BLOCK_STACK_PROBE, false, false});
        r.required_sites.push_back({CacheSiteRole::BLOCK_STACK_OUTPUT, true, false});
        r.history_length = 2;
        r.requires_probe = true;
        return r;
    }

    CacheProgram compile(const ModelSchema&, const ModelTopology& topo, const InferenceConfig& inf) override {
        config_ = inf.config->dicache;
        initialized_ = config_.enabled;
        num_steps_ = inf.num_steps;
        retention_steps_ = static_cast<int>(config_.retention_ratio * num_steps_);
        states_.clear();
        total_steps_skipped_ = 0;

        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        CacheProgram program = detail::make_reuse_program("DiCache", seg,
                                                          SegmentExecutionMode::LOAD_CACHED,
                                                          detail::make_slot(0, "block_stack_residual", 2));
        // Add the PROBE variant: run probe_depth blocks then decide.
        GraphVariantPlan probe;
        probe.id = kVariantProbe;
        probe.kind = GraphVariantKind::PROBE;
        SegmentPlan probe_seg;
        probe_seg.segment_id = seg;
        probe_seg.execution = SegmentExecutionMode::PROBE;
        probe_seg.probe_depth = std::max(1, config_.probe_depth);
        probe.segments.push_back(probe_seg);
        program.variants.push_back(probe);
        return program;
    }

    void begin_step(const StepContext& step) override { current_step_index_ = step.step_index; }

    RuntimeDecision decide(const StepContext&, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        if (!enabled()) {
            return d;
        }
        (void)m;
        // Probe every eligible step: the probe pass measures the trajectory and
        // seeds history on its first run, so the skip decision can bootstrap.
        // (Gating this on has_probe_history would deadlock — history is only
        // built from probe passes.)
        if (current_step_index_ <= retention_steps_ ||
            current_step_index_ >= num_steps_ - 1) {
            return d;
        }
        d.variant = kVariantProbe;
        return d;
    }

    RuntimeDecision decide_after_probe(const StepContext&, const CacheObservation& obs) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        Branch& b = branch_for(obs.condition_key);
        // GPU path: the seam already computed delta_y/delta_x on-device and handed
        // back scalars (no ~50MB host tensors). Use them directly.
        if (!std::isnan(obs.delta_y)) {
            if (!b.gpu_has_history) {
                b.gpu_has_history = true;
                return d;  // first probe-eligible step seeds history via capture
            }
            const float error = config_.error_choice == DiCacheErrorChoice::DeltaMinus &&
                                        !std::isnan(obs.delta_x)
                                    ? std::fabs(obs.delta_y - obs.delta_x)
                                    : obs.delta_y;
            b.accumulated_rel_l1 += error;
            if (b.accumulated_rel_l1 < config_.rel_l1_thresh) {
                d.variant = kVariantReuse;
                total_steps_skipped_++;  // GPU path skips via inject_gpu, not reconstruct()
            } else {
                b.accumulated_rel_l1 = 0.0f;
            }
            return d;
        }
        if (obs.before == nullptr || obs.probe == nullptr || obs.before->empty() || obs.probe->empty()) {
            return d;
        }
        // Host path: compute the metric straight off the observation data (no
        // pre-copy). The cur_input/cur_probe copies below are only needed as the
        // NEXT step's history; delta_x is only used by the non-default delta_minus.
        float delta_y = 1.0f;
        float delta_x = 1.0f;
        if (b.has_probe_history &&
            b.prev_probe.size() == static_cast<size_t>(obs.probe->numel())) {
            delta_y = rel_l1_ptr(obs.probe->data(), b.prev_probe.data(), b.prev_probe.size());
            if (config_.error_choice == DiCacheErrorChoice::DeltaMinus &&
                b.prev_input.size() == static_cast<size_t>(obs.before->numel())) {
                delta_x = rel_l1_ptr(obs.before->data(), b.prev_input.data(), b.prev_input.size());
            }
        }
        b.cur_input.assign(obs.before->data(), obs.before->data() + obs.before->numel());
        b.cur_probe.assign(obs.probe->data(), obs.probe->data() + obs.probe->numel());
        b.have_cur = true;
        if (!b.has_probe_history) {
            return d;  // no baseline yet -> compute, seed history in observe()
        }
        // Match the reference metric (bidcache dicache forwards.py): delta_y uses
        // the probe trajectory movement directly; delta_minus uses |delta_y-delta_x|.
        // The old code hardcoded delta_minus but kept delta_y's 0.4 threshold, which
        // made the error much smaller than the threshold intended -> massive
        // over-skipping and quality collapse.
        const float error = config_.error_choice == DiCacheErrorChoice::DeltaMinus
                                ? std::fabs(delta_y - delta_x)
                                : delta_y;
        b.accumulated_rel_l1 += error;
        if (b.accumulated_rel_l1 < config_.rel_l1_thresh) {
            d.variant = kVariantReuse;
        } else {
            b.accumulated_rel_l1 = 0.0f;
        }
        return d;
    }

    void observe(const CacheObservation& obs) override {
        if (obs.kind != CacheObservation::Kind::Feature || obs.feature == nullptr || obs.feature->empty()) {
            return;
        }
        if (!enabled()) {
            return;
        }
        Branch& b = branch_for(obs.condition_key);
        b.resid_prev2 = std::move(b.resid_prev1);
        b.resid_prev1.assign(obs.feature->data(), obs.feature->data() + obs.feature->numel());
        b.have_resid2 = !b.resid_prev2.empty();
        b.shape = obs.feature->shape();
        b.has_residual = true;
        commit_probe_history(b);
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext& ctx) override {
        Branch& b = branch_for(ctx.condition_key);
        if (!b.has_residual) {
            return {};
        }
        sd::Tensor<float> out(b.shape);
        if (out.numel() != static_cast<int64_t>(b.resid_prev1.size())) {
            return {};
        }
        float* data = out.data();
        if (b.have_resid2 && b.resid_prev2.size() == b.resid_prev1.size()) {
            float gamma = 1.0f;
            // Dynamic Cache Trajectory Alignment (ref forwards.py _dicache_apply_cached_residual):
            //   current_residual_indicator = probe_state - input   (this step's probe residual)
            //   gamma = clamp( mean|cur_probe_resid - probe_resid[-2]|
            //                  / mean|probe_resid[-1] - probe_resid[-2]|, 1.0, 1.5)
            // The old code used the raw probe STATE (cur_probe) in the numerator
            // instead of the probe RESIDUAL (cur_probe - cur_input), mismatching the
            // reference and distorting gamma on every skipped step.
            if (b.have_probe_prev2 && b.have_cur &&
                b.cur_probe.size() == b.cur_input.size() &&
                b.probe_prev2.size() == b.cur_probe.size() &&
                b.probe_prev1.size() == b.cur_probe.size()) {
                // Fused single pass: form (cur_probe - cur_input) on the fly and
                // accumulate both |cur_probe_resid - probe_prev2| (num) and
                // |probe_prev1 - probe_prev2| (den) without materializing a temp
                // ~50MB residual vector (was: 1 temp build + 2 separate passes).
                float num = 0.0f, den = 0.0f;
                const float* cp = b.cur_probe.data();
                const float* ci = b.cur_input.data();
                const float* pp1 = b.probe_prev1.data();
                const float* pp2 = b.probe_prev2.data();
                const size_t n = b.cur_probe.size();
                for (size_t i = 0; i < n; ++i) {
                    num += std::fabs((cp[i] - ci[i]) - pp2[i]);
                    den += std::fabs(pp1[i] - pp2[i]);
                }
                gamma = den > 1e-6f ? num / den : 1.0f;
                gamma = std::max(1.0f, std::min(1.5f, gamma));
            }
            for (size_t i = 0; i < b.resid_prev1.size(); ++i) {
                data[i] = b.resid_prev2[i] + gamma * (b.resid_prev1[i] - b.resid_prev2[i]);
            }
        } else {
            std::copy(b.resid_prev1.begin(), b.resid_prev1.end(), data);
        }
        commit_probe_history(b);
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
            LOG_INFO("DiCache reused %d/%zu steps (%.2fx)", total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("DiCache reused %d/%zu steps", total_steps_skipped_, total_steps);
        }
    }

    void reset() override {
        states_.clear();
        total_steps_skipped_ = 0;
        current_step_index_ = -1;
    }

private:
    struct Branch {
        std::vector<float> prev_input;
        std::vector<float> prev_probe;
        std::vector<float> probe_prev1;
        std::vector<float> probe_prev2;
        bool has_probe_history = false;
        bool have_probe_prev2 = false;
        std::vector<float> resid_prev1;
        std::vector<float> resid_prev2;
        bool have_resid2 = false;
        std::vector<int64_t> shape;
        bool has_residual = false;
        float accumulated_rel_l1 = 0.0f;
        std::vector<float> cur_input;
        std::vector<float> cur_probe;
        bool have_cur = false;
        bool gpu_has_history = false;  // GPU path: seeded after the first capture
    };

    void commit_probe_history(Branch& b) {
        if (!b.have_cur) {
            return;
        }
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

std::unique_ptr<ICachePolicy> create_dicache_policy() { return std::make_unique<DiCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
