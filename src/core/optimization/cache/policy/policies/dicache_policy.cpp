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
        // Declarative Probe program: residual ring + PROBE/REUSE/FULL variants. On
        // the host path (ED_DICACHE_GPU=0, so the pipeline leaves inject_gpu unset)
        // the lowering runs the probe, calls decide_after_probe (which supplies the
        // gamma blend weights), and injects the declarative blend of the ring. The
        // default on-GPU path leaves inject_gpu set, so it takes the legacy seam
        // control flow (which this same program also drives).
        // Substep path opt-in: gated by ED_CACHE_SUBSTEP AND a viable on-device
        // metric path (a device store wired by the runner). Wan has no device store,
        // so its host DiCache stays on the legacy path. The engine calls
        // set_substep_gpu_path() after init with the device-store signal; here we
        // only record the env opt-in, ANDed in supports_substep().
        const char* substep_env = std::getenv("ED_CACHE_SUBSTEP");
        substep_env_on_ = substep_env != nullptr && substep_env[0] != '\0' && substep_env[0] != '0';
        substep_gpu_path_ = false;  // set true by set_substep_gpu_path() when a store exists
        return detail::make_dicache_program("DiCache", seg, std::max(1, config_.probe_depth));
    }

    void begin_step(const StepContext& step) override { current_step_index_ = step.step_index; }

    // ---- Substep interface (ED_CACHE_SUBSTEP). DiCache is the two-yield case that
    // proves the uniform loop: substep 1 = probe (measures delta_y/gamma on-device,
    // produces no output), then observe_substep folds the scalars via the same
    // decide_after_probe math; substep 2 = reuse (Extrapolate with gamma) or a
    // capturing full compute. Both paths are migrated: the on-device metric path
    // (Qwen/Flux, device store) and the tap-driven host path (Wan, no store, uses
    // observe_substep_probe_host + host reconstruct). Gate only on the env opt-in.
    bool supports_substep() const override { return substep_env_on_; }

    void set_substep_device_available(bool available) override { substep_gpu_path_ = available; }

    void begin_substeps(const StepContext& step, const void* condition_key) override {
        current_step_index_ = step.step_index;
        substep_key_ = condition_key;
        substep_phase_ = probe_eligible_() ? SubstepPhase::Probe : SubstepPhase::SingleFull;
        substep_reuse_ = false;
        substep_gamma_ = 1.0f;
        substep_host_reuse_ = false;
        substep_host_coeffs_.clear();
    }

    std::optional<SubstepPlan> next_substep() override {
        switch (substep_phase_) {
            case SubstepPhase::SingleFull: {
                substep_phase_ = SubstepPhase::Done;
                SubstepPlan p;
                p.input = InputSource{InputSource::FreshLatent, -1};
                p.blocks = BlockRange{0, -1};
                p.produces_output = true;
                p.writes = {0};  // capture residual into the ring
                return p;
            }
            case SubstepPhase::Probe: {
                substep_phase_ = SubstepPhase::Continue;
                SubstepPlan p;
                p.input = InputSource{InputSource::FreshLatent, -1};
                p.blocks = BlockRange{0, std::max(1, config_.probe_depth)};
                p.op = SubstepOp{SubstepOpKind::Stash, {}, {}};
                p.produces_output = false;
                // Indicators are computed on-device by the seam (delta_y/gamma) and
                // surfaced through the adapter as substep-result scalars.
                p.indicators.push_back(Indicator{"delta_y", Indicator::RelL1, {}, {}, false});
                p.indicators.push_back(Indicator{"gamma", Indicator::RelL1, {}, {}, false});
                return p;
            }
            case SubstepPhase::Continue: {
                substep_phase_ = SubstepPhase::Done;
                SubstepPlan p;
                p.input = InputSource{InputSource::FreshLatent, -1};
                p.produces_output = true;
                if (substep_reuse_) {
                    p.blocks = BlockRange{0, 0};  // zero compute; reuse the cached residual
                    if (substep_host_reuse_) {
                        // Host reuse (Wan): the lowering calls reconstruct() for the
                        // gamma-blended residual and injects x_before + residual.
                        p.op = SubstepOp{SubstepOpKind::ApplyResidual, {0}, {}};
                    } else {
                        // Device reuse (Qwen/Flux): on-device gamma-blend via inject_gpu.
                        p.op = SubstepOp{SubstepOpKind::Extrapolate, {0}, {substep_gamma_}};
                    }
                } else {
                    p.blocks = BlockRange{0, -1};
                    p.writes = {0};
                }
                return p;
            }
            default:
                return std::nullopt;
        }
    }

    void observe_substep(const SubstepResult& r) override {
        // Only meaningful after the probe substep. Reproduces decide_after_probe's
        // GPU-scalar branch: accumulate the trajectory error, skip when under
        // threshold, and clamp gamma for the reuse blend.
        if (substep_phase_ != SubstepPhase::Continue) {
            return;
        }
        const float dy = r.get("delta_y", std::numeric_limits<float>::quiet_NaN());
        if (std::isnan(dy)) {
            substep_reuse_ = false;
            return;
        }
        Branch& b = branch_for(substep_key_);
        if (!b.gpu_has_history) {
            b.gpu_has_history = true;  // first probe-eligible step seeds via capture
            substep_reuse_ = false;
            return;
        }
        const float dx = r.get("delta_x", std::numeric_limits<float>::quiet_NaN());
        const float error = config_.error_choice == DiCacheErrorChoice::DeltaMinus && !std::isnan(dx)
                                ? std::fabs(dy - dx)
                                : dy;
        b.accumulated_rel_l1 += error;
        if (b.accumulated_rel_l1 < config_.rel_l1_thresh) {
            substep_reuse_ = true;
            const float g = r.get("gamma", 1.0f);
            substep_gamma_ = std::max(1.0f, std::min(1.5f, g));
            total_steps_skipped_++;
        } else {
            b.accumulated_rel_l1 = 0.0f;
            substep_reuse_ = false;
        }
    }

    // Host substep probe: the tap-driven probe handed back the before/probe host
    // tensors. Delegate to decide_after_probe (the verified host metric + gamma
    // logic), then translate its decision into the substep reuse flag + host reuse
    // coeffs the continuation uses. No divergence from the legacy host path — same
    // code computes the decision.
    void observe_substep_probe_host(const sd::Tensor<float>* before,
                                    const sd::Tensor<float>* probe,
                                    const StepContext& step,
                                    const void* condition_key) override {
        if (substep_phase_ != SubstepPhase::Continue) {
            return;
        }
        CacheObservation obs;
        obs.kind = CacheObservation::Kind::Probe;
        obs.step = step;
        obs.condition_key = condition_key;
        obs.before = before;
        obs.probe = probe;
        // delta_y/delta_x/gamma stay NaN so decide_after_probe takes the host branch.
        const RuntimeDecision d = decide_after_probe(step, obs);
        const bool reuse = (d.variant == kVariantReuse);
        substep_reuse_ = reuse;
        substep_host_reuse_ = reuse;
        substep_host_coeffs_ = d.reuse_coeffs;  // [gamma, 1-gamma] over ring depths [1,2]
    }

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
            // Supply the gamma-aligned blend weights so the declarative REUSE action
            // forms resid_prev2 + gamma*(resid_prev1 - resid_prev2) as a weighted
            // blend [gamma, 1-gamma] over ring depths [1, 2].
            d.reuse_coeffs = compute_blend_coeffs(b);
            d.variant = kVariantReuse;
            total_steps_skipped_++;
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
        // Device path: reuse is served declaratively (inject_gpu) and this returns
        // empty. Host substep path (Wan): blend the residual ring on host with the
        // coeffs observe_substep_probe_host stored — w1*resid_prev1 + w2*resid_prev2.
        if (!substep_host_reuse_ || substep_host_coeffs_.empty()) {
            return {};
        }
        Branch& b = branch_for(ctx.condition_key);
        if (!b.has_residual || b.resid_prev1.empty() || b.shape.empty()) {
            return {};
        }
        const float w1 = substep_host_coeffs_[0];
        const float w2 = substep_host_coeffs_.size() > 1 ? substep_host_coeffs_[1] : 0.0f;
        const bool have2 = b.have_resid2 && b.resid_prev2.size() == b.resid_prev1.size();
        sd::Tensor<float> feature(b.shape);
        float* out = feature.data();
        const float* r1 = b.resid_prev1.data();
        const float* r2 = have2 ? b.resid_prev2.data() : nullptr;
        const size_t n = b.resid_prev1.size();
        for (size_t i = 0; i < n; ++i) {
            out[i] = have2 ? (w1 * r1[i] + w2 * r2[i]) : r1[i];
        }
        return feature;
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
    // Same probe-eligibility gate as decide(): skip the retention prefix and the
    // final step. The substep path uses this to choose Probe vs a plain full step.
    bool probe_eligible_() const {
        if (!enabled()) {
            return false;
        }
        return !(current_step_index_ <= retention_steps_ ||
                 current_step_index_ >= num_steps_ - 1);
    }

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

    // Blend weights [w1, w2] over residual ring depths [1, 2] (resid_prev1,
    // resid_prev2). resid_prev2 + gamma*(resid_prev1 - resid_prev2) rearranges to
    // gamma*resid_prev1 + (1-gamma)*resid_prev2, so w1=gamma, w2=1-gamma. With only
    // one residual available, serve resid_prev1 directly via [1, 0]. gamma is the
    // trajectory-alignment ratio (ref forwards.py _dicache_apply_cached_residual):
    //   gamma = clamp( mean|cur_probe_resid - probe_resid[-2]|
    //                  / mean|probe_resid[-1] - probe_resid[-2]|, 1.0, 1.5)
    // Commits the probe history (the skip-step analogue of observe()'s full-step
    // commit), matching the legacy reconstruct()'s commit-then-return ordering.
    std::vector<float> compute_blend_coeffs(Branch& b) {
        float gamma = 1.0f;
        const bool have_two = b.have_resid2;
        if (have_two && b.have_probe_prev2 && b.have_cur &&
            b.cur_probe.size() == b.cur_input.size() &&
            b.probe_prev2.size() == b.cur_probe.size() &&
            b.probe_prev1.size() == b.cur_probe.size()) {
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
        commit_probe_history(b);
        if (!have_two) {
            return {1.0f, 0.0f};  // only resid_prev1 exists -> serve it directly
        }
        return {gamma, 1.0f - gamma};
    }

    DiCacheConfig config_;
    bool initialized_ = false;
    int num_steps_ = 0;
    int retention_steps_ = 0;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::unordered_map<const void*, Branch> states_;

    // Substep state (ED_CACHE_SUBSTEP GPU path).
    enum class SubstepPhase { Probe, Continue, SingleFull, Done };
    bool substep_env_on_ = false;
    bool substep_gpu_path_ = false;
    SubstepPhase substep_phase_ = SubstepPhase::Done;
    const void* substep_key_ = nullptr;
    bool substep_reuse_ = false;
    float substep_gamma_ = 1.0f;
    // Host substep reuse (Wan): the continuation injects a host-reconstructed
    // gamma-blend residual instead of the on-device inject_gpu path.
    bool substep_host_reuse_ = false;
    std::vector<float> substep_host_coeffs_;

    Branch& branch_for(const void* cond) { return states_[cond]; }
};

}  // namespace

std::unique_ptr<ICachePolicy> create_dicache_policy() { return std::make_unique<DiCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
