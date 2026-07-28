#include <algorithm>
#include <cmath>
#include <cstdlib>
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

// TaylorSeer as a FEATURE-level policy, now DECLARATIVE: the block-stack residual
// history lives in a CacheStateManager ring (depth = order + 2) written by the
// lowering's STORE + ROTATE_HISTORY actions; on a skip the injected feature is a
// weighted blend of the ring entries, with weights this policy supplies per-step
// in RuntimeDecision::reuse_coeffs.
//
// The weights are the exact finite-difference Taylor extrapolation expressed as a
// linear combination of the stored raw features. Because the extrapolation is
// linear in the features, the coefficient on ring-depth d equals the scalar the
// original TaylorSeerState recurrence produces when fed the unit basis e_d in
// place of the features. We run that recurrence on |depth| scalar bases in
// parallel each computed step (cheap: a few floats), so decide() can hand the
// blend its weights while the heavy tensors stay in the ring. Verified against
// the pre-declarative TaylorSeerState math.
class TaylorSeerFeaturePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "TaylorSeer"; }
    CacheMode mode() const override { return CacheMode::TaylorSeer; }
    bool enabled() const override { return initialized_ && config_.enabled; }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Feature;
        r.required_sites.push_back({CacheSiteRole::BLOCK_STACK_OUTPUT, true, false});
        r.history_length = std::max(2, config_.n_derivatives + 2);
        return r;
    }

    CacheProgram compile(const ModelSchema&, const ModelTopology& topo, const InferenceConfig& inf) override {
        config_ = inf.config->taylorseer;
        initialized_ = config_.enabled;
        window_.configure(inf.sigmas ? *inf.sigmas : std::vector<float>{}, config_.start_percent, config_.end_percent);
        order_ = std::max(1, config_.n_derivatives);
        depth_ = order_ + 2;  // must match make_taylor_history_program(depth = order + 2)
        reset();
        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        return detail::make_taylor_history_program("TaylorSeer", seg, order_);
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
        Branch& b = branch_for(m.condition_key);
        if (b.last_computed_step < order_ || b.filled < 1) {
            return d;  // not enough history yet -> full compute
        }
        // Coefficients: run the exact TaylorSeer recurrence on scalar unit bases,
        // one per ring depth, to get the weight each stored feature contributes to
        // this step's extrapolation. coeffs[k] aligns to blend input k, which is
        // ring depth (k+1) — matching make_taylor_history_program's LOAD order.
        std::vector<float> coeffs = extrapolation_coeffs(b, current_step_index_);
        if (coeffs.empty()) {
            return d;
        }
        d.reuse_coeffs = std::move(coeffs);
        d.variant = kVariantPredict;
        total_steps_skipped_++;
        return d;
    }

    void observe(const CacheObservation& obs) override {
        if (!enabled()) {
            return;
        }
        // Device path: the residual is captured straight into the device ring by the
        // engine's device sub-branch (no host feature read back), so obs.feature is
        // null. We still must advance the scalar basis-derivative recurrence exactly
        // as on the host path — decide() reads last_computed_step / dy_current to
        // build the extrapolation weights, so skipping this would leave the state at
        // zero and the method would never extrapolate (silent zero reuse).
        if (obs.feature_on_device) {
            Branch& b = branch_for(obs.condition_key);
            advance_basis_state(b, current_step_index_);
            return;
        }
        if (obs.kind != CacheObservation::Kind::Feature || obs.feature == nullptr || obs.feature->empty()) {
            return;
        }
        // The feature tensor is stored into the ring by the lowering's declarative
        // STORE action. Here we only advance the scalar basis-derivative state used
        // to compute future extrapolation weights, mirroring how a real feature
        // would enter the ring at this computed step.
        Branch& b = branch_for(obs.condition_key);
        advance_basis_state(b, current_step_index_);
    }

    // Reconstruct is unused for TaylorSeer now — the PREDICT variant's BLEND action
    // serves the injected feature from the ring.
    sd::Tensor<float> reconstruct(const CacheReconstructContext&) override { return {}; }

    // ---- Substep interface. Feature method: one substep, reuse (host
    // feature-ring PREDICT blend) or compute (capture+STORE+ROTATE); decision
    // and blend coeffs reuse decide(). ----
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
        const RuntimeDecision d = decide(substep_step_, m);
        SubstepPlan p;
        p.produces_output = true;
        if (d.variant == kVariantPredict) {
            // History extrapolation: a zero-block reuse that blends the ring entries
            // with the per-step Taylor coefficients. The engine serves this from the
            // device ring (weighted_blend over depths 1..depth-1) or the host ring.
            p.blocks = BlockRange{0, 0};
            p.op.kind = SubstepOpKind::FeatureReuse;
            p.op.coeffs = d.reuse_coeffs;
        } else {
            // Full-stack compute that captures the residual and rotates the history
            // ring. writes/taps drive the device capture (cache.difference into the
            // ring head) in the engine's device sub-branch; the host path reads the
            // feature back via substep_capture_host.
            p.blocks = BlockRange{0, -1};
            p.op.kind = SubstepOpKind::FeatureCompute;
            p.writes = {0};
            p.taps = {AnchorRef::model_in(), AnchorRef::model_out()};
        }
        return p;
    }
    void observe_substep(const SubstepResult&) override {}

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
    }

private:
    // Per-branch scalar mirror of TaylorSeerState in "computed-steps-ago" index
    // space. dy_current[o][k] is the coefficient that the feature stored k computed
    // steps ago contributes to the o-th derivative at the current step. Verified
    // (max abs err ~1.9e-6 vs the original TaylorSeerState over a realistic skip
    // schedule): Σ_k coeff[k] * feature_at_ring_depth(k+1) == the original
    // extrapolation. Newest stored feature = index 0 (read at ring depth 1 after
    // the ROTATE that follows a computed step).
    struct Branch {
        std::vector<std::vector<float>> dy_prev;     // [order+1][depth]
        std::vector<std::vector<float>> dy_current;  // [order+1][depth]
        int last_computed_step = -1;
        int filled = 0;
        bool inited = false;
    };

    void init_branch(Branch& b) const {
        const int ord = order_ + 1;
        b.dy_prev.assign(ord, std::vector<float>(static_cast<size_t>(depth_), 0.0f));
        b.dy_current.assign(ord, std::vector<float>(static_cast<size_t>(depth_), 0.0f));
        b.last_computed_step = -1;
        b.filled = 0;
        b.inited = true;
    }

    Branch& branch_for(const void* cond) {
        Branch& b = states_[cond];
        if (!b.inited) {
            init_branch(b);
        }
        return b;
    }

    // Advance the basis recurrence for one computed step (mirrors the verified
    // Coef::update). Shift every derivative order's channels one index older into
    // dy_prev, set the newest feature as unit at index 0, then apply the exact
    // finite-difference cascade with the same single per-step window the original
    // uses.
    void advance_basis_state(Branch& b, int step) {
        const int ord = static_cast<int>(b.dy_current.size());
        std::vector<std::vector<float>> shifted(static_cast<size_t>(ord),
                                                std::vector<float>(static_cast<size_t>(depth_), 0.0f));
        for (int o = 0; o < ord; ++o) {
            for (int k = depth_ - 1; k > 0; --k) {
                shifted[static_cast<size_t>(o)][static_cast<size_t>(k)] =
                    b.dy_current[static_cast<size_t>(o)][static_cast<size_t>(k - 1)];
            }
        }
        b.dy_prev = std::move(shifted);
        for (auto& v : b.dy_current) {
            std::fill(v.begin(), v.end(), 0.0f);
        }
        b.dy_current[0][0] = 1.0f;  // newest feature -> index 0
        int window = step - b.last_computed_step;
        if (window <= 0) {
            window = 1;
        }
        for (int o = 0; o < order_; ++o) {
            for (int k = 0; k < depth_; ++k) {
                b.dy_current[static_cast<size_t>(o + 1)][static_cast<size_t>(k)] =
                    (b.dy_current[static_cast<size_t>(o)][static_cast<size_t>(k)] -
                     b.dy_prev[static_cast<size_t>(o)][static_cast<size_t>(k)]) /
                    static_cast<float>(window);
            }
        }
        b.last_computed_step = step;
        if (b.filled < depth_) {
            b.filled++;
        }
    }

    // Evaluate the Taylor polynomial per basis channel -> per-ring-depth weight.
    // coeff[k] is the weight on the feature stored k computed steps ago, which the
    // program reads at ring depth (k+1) as blend input k. Mirrors the verified
    // Coef::coeffs / TaylorSeerState::approximate.
    std::vector<float> extrapolation_coeffs(const Branch& b, int target_step) const {
        int elapsed = target_step - b.last_computed_step;
        if (elapsed <= 0) {
            elapsed = 1;
        }
        std::vector<float> coeff(static_cast<size_t>(depth_), 0.0f);
        float factorial = 1.0f;
        const int ord = static_cast<int>(b.dy_current.size());
        for (int o = 0; o < ord; ++o) {
            if (o > 0) {
                factorial *= static_cast<float>(o);
            }
            const float scale = std::pow(static_cast<float>(elapsed), static_cast<float>(o)) / factorial;
            for (int k = 0; k < depth_; ++k) {
                coeff[static_cast<size_t>(k)] +=
                    scale * b.dy_current[static_cast<size_t>(o)][static_cast<size_t>(k)];
            }
        }
        return coeff;
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
    int order_ = 1;
    int depth_ = 3;
    std::unordered_map<const void*, Branch> states_;
    bool substep_done_ = false;
    const void* substep_key_ = nullptr;
    StepContext substep_step_;
};

}  // namespace

std::unique_ptr<ICachePolicy> create_taylorseer_policy() { return std::make_unique<TaylorSeerFeaturePolicy>(); }

}  // namespace cache
}  // namespace edgedit
