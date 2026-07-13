#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/policy/cache_policy.hpp"
#include "core/optimization/cache/policy/policy_common.hpp"
#include "json.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {
namespace {

using detail::kVariantFull;
using detail::kVariantReuse;

// ===========================================================================
// SenCache sensitivity profile (struct + I/O). Folded in from the former
// sencache_profile.{hpp,cpp}: SenCache-only, so it lives here. Per-timestep
// finite-difference Jacobian norms J_z / J_t with the sigma they were measured
// at. Unlike MagCache there is no neutral default — SenCache refuses to run
// without a complete profile.
// ===========================================================================
struct SenCacheProfile {
    std::string model;
    std::vector<float> sigmas;
    std::vector<float> j_z;
    std::vector<float> j_t;

    bool empty() const { return sigmas.empty() || j_z.empty() || j_t.empty(); }
    size_t size() const { return sigmas.size(); }
};

bool save_sencache_profile(const std::string& path, const std::string& model,
                           const std::vector<float>& sigmas,
                           const std::vector<float>& j_z, const std::vector<float>& j_t) {
    if (sigmas.empty() || sigmas.size() != j_z.size() || sigmas.size() != j_t.size()) {
        LOG_ERROR("SenCache profile refusing to write: array lengths disagree "
                  "(sigmas=%zu j_z=%zu j_t=%zu)",
                  sigmas.size(), j_z.size(), j_t.size());
        return false;
    }
    nlohmann::json out;
    out["type"] = "sencache_profile";
    out["model"] = model;
    out["num_steps"] = static_cast<int>(sigmas.size());
    out["sigmas"] = sigmas;
    out["j_z"] = j_z;
    out["j_t"] = j_t;
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("failed to open SenCache profile for writing: %s", path.c_str());
        return false;
    }
    file << out.dump(2) << "\n";
    if (!file.good()) {
        LOG_ERROR("failed to write SenCache profile: %s", path.c_str());
        return false;
    }
    return true;
}

// Extract a float array field; returns false if absent, non-array, or malformed.
inline bool sencache_read_float_array(const nlohmann::json& json, const char* key,
                                      const std::string& path, std::vector<float>* out) {
    if (!json.contains(key) || !json[key].is_array()) {
        LOG_WARN("SenCache profile '%s' has no %s array", path.c_str(), key);
        return false;
    }
    try {
        *out = json[key].get<std::vector<float>>();
    } catch (const std::exception& e) {
        LOG_WARN("SenCache profile '%s' has malformed %s: %s", path.c_str(), key, e.what());
        return false;
    }
    return true;
}

SenCacheProfile load_sencache_profile(const std::string& path, bool* ok) {
    if (ok != nullptr) {
        *ok = false;
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_WARN("failed to open SenCache profile: %s", path.c_str());
        return {};
    }
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(file);
    } catch (const std::exception& e) {
        LOG_WARN("failed to parse SenCache profile '%s': %s", path.c_str(), e.what());
        return {};
    }
    SenCacheProfile profile;
    if (json.contains("model") && json["model"].is_string()) {
        profile.model = json["model"].get<std::string>();
    }
    if (!sencache_read_float_array(json, "sigmas", path, &profile.sigmas) ||
        !sencache_read_float_array(json, "j_z", path, &profile.j_z) ||
        !sencache_read_float_array(json, "j_t", path, &profile.j_t)) {
        return {};
    }
    if (profile.sigmas.empty()) {
        LOG_WARN("SenCache profile '%s' has an empty table", path.c_str());
        return {};
    }
    if (profile.sigmas.size() != profile.j_z.size() ||
        profile.sigmas.size() != profile.j_t.size()) {
        LOG_WARN("SenCache profile '%s' array lengths disagree "
                 "(sigmas=%zu j_z=%zu j_t=%zu)",
                 path.c_str(), profile.sigmas.size(), profile.j_z.size(), profile.j_t.size());
        return {};
    }
    if (json.contains("num_steps") && json["num_steps"].is_number_integer()) {
        const int declared = json["num_steps"].get<int>();
        if (declared != static_cast<int>(profile.sigmas.size())) {
            LOG_WARN("SenCache profile '%s' num_steps=%d disagrees with table length %zu",
                     path.c_str(), declared, profile.sigmas.size());
            return {};
        }
    }
    if (ok != nullptr) {
        *ok = true;
    }
    return profile;
}

// ===========================================================================
// SenCache (Feature granularity): sensitivity-aware caching. Math ported
// verbatim from the old SenCachePolicy. Verified against vita-epfl/SenCache.
// ===========================================================================
class SenCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "SenCache"; }
    CacheMode mode() const override { return CacheMode::SenCache; }
    bool enabled() const override { return initialized_ && config_.enabled; }
    bool supports_calibration() const override { return true; }

    CacheRequirements requirements() const override {
        CacheRequirements r;
        r.granularity = CacheGranularity::Feature;
        r.required_sites.push_back({CacheSiteRole::BLOCK_STACK_OUTPUT, true, false});
        r.history_length = 1;
        return r;
    }

    CacheProgram compile(const ModelSchema& schema, const ModelTopology& topo, const InferenceConfig& inf) override {
        config_ = inf.config->sencache;
        model_name_ = cache_model_spec_for_version(schema.version).model_name;
        num_steps_ = inf.num_steps;
        sigmas_ = inf.sigmas ? *inf.sigmas : std::vector<float>{};
        states_.clear();
        total_steps_skipped_ = 0;
        observed_any_ = false;
        calibrating_ = config_.enabled && !config_.calibrate_path.empty();
        retention_steps_ = static_cast<int>(config_.retention_ratio * num_steps_ + 0.5f);
        switch_step_ = static_cast<int>(config_.switch_ratio * num_steps_ + 0.5f);

        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        // Declarative Feature program for the reuse (non-calibrating) path: the
        // block-stack residual lives in the CacheStateManager slot, injected by the
        // lowering. Calibration never reuses, so it keeps the plain (no-action)
        // program and drives forward-evaluator passes instead.
        CacheProgram program = calibrating_
            ? detail::make_reuse_program("SenCache", seg, SegmentExecutionMode::LOAD_CACHED,
                                         detail::make_slot(0, "block_stack_residual"))
            : detail::make_feature_reuse_program("SenCache", seg);

        if (calibrating_) {
            initialized_ = true;
            LOG_INFO("SenCache calibration: profiling %d steps -> %s",
                     num_steps_, config_.calibrate_path.c_str());
            return program;
        }
        if (config_.profile_path.empty()) {
            LOG_ERROR("SenCache requires a calibrated profile: pass --cache-profile "
                      "<path> (or produce one with --cache-calibrate). Caching disabled.");
            initialized_ = false;
            return program;
        }
        bool ok = false;
        profile_ = load_sencache_profile(config_.profile_path, &ok);
        if (!ok || profile_.empty()) {
            LOG_ERROR("SenCache profile %s unusable; caching disabled (no neutral "
                      "fallback for a sensitivity table).", config_.profile_path.c_str());
            initialized_ = false;
            return program;
        }
        initialized_ = config_.enabled;
        LOG_INFO("SenCache using profile %s (%zu calibrated steps, model=%s)",
                 config_.profile_path.c_str(), profile_.size(), profile_.model.c_str());
        return program;
    }

    void begin_step(const StepContext& step) override { current_step_index_ = step.step_index; }

    RuntimeDecision decide(const StepContext& step, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        if (!enabled() || calibrating_ || current_step_index_ < retention_steps_) {
            return d;
        }
        if (m.input == nullptr || m.input->empty()) {
            return d;
        }
        Branch& b = branch_for(m.condition_key);
        if (!b.has_residual || !b.has_reference) {
            return d;
        }
        const int64_t numel = m.input->numel();
        if (numel <= 0 || static_cast<int64_t>(b.cached_z.size()) != numel) {
            return d;
        }
        const float* z = m.input->data();
        double dz_sq = 0.0;
        for (int64_t i = 0; i < numel; ++i) {
            const double dv = static_cast<double>(z[i]) - static_cast<double>(b.cached_z[i]);
            dz_sq += dv * dv;
        }
        const double norm_dz = std::sqrt(dz_sq);
        const double norm_dt = std::fabs(static_cast<double>(step.sigma) - static_cast<double>(b.cached_sigma));
        const double error =
            (b.cached_j_z * norm_dz + b.cached_j_t * norm_dt) / std::sqrt(static_cast<double>(numel));
        const float threshold = current_step_index_ < switch_step_ ? config_.thresh_start : config_.thresh_main;
        if (error < threshold && b.accumulated_skips < config_.max_skip_steps) {
            b.accumulated_skips++;
            // Count the skip here (reconstruct no longer runs on the declarative
            // path). A rare inject-fail falls back to full compute -> negligible
            // miscount, same convention as MagCache.
            total_steps_skipped_++;
            d.variant = kVariantReuse;
            return d;
        }
        b.accumulated_skips = 0;
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
        if (calibrating_) {
            b.branch = obs.branch;
            // Calibration keeps the residual on the host (plain program, no slot).
            b.residual.assign(obs.feature->data(), obs.feature->data() + obs.feature->numel());
            b.shape = obs.feature->shape();
        }
        // Non-calibrating: the residual tensor is stored into the slot by the FULL
        // variant's declarative STORE action; the policy only tracks that a residual
        // exists plus the sensitivity reference (cached_z / sigma / Jacobians).
        b.has_residual = true;
        if (obs.input != nullptr && !obs.input->empty()) {
            b.cached_z.assign(obs.input->data(), obs.input->data() + obs.input->numel());
            b.cached_sigma = obs.step.sigma;
            lookup_jacobians(obs.step.sigma, &b.cached_j_z, &b.cached_j_t);
            b.has_reference = true;
        }
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext& ctx) override {
        // Reuse is served declaratively (LOAD slot -> hooks.inject) in the normal
        // run. Only the calibration path (plain program) would ever land here, and
        // it never reuses, so return empty.
        (void)ctx;
        return {};
    }

    CalibrationSpec calibration_spec() const override {
        CalibrationSpec spec;
        spec.active = calibrating_;
        spec.needs_forward_evaluator = calibrating_;  // finite-diff Jacobians
        return spec;
    }

    // Measure finite-difference Jacobian norms on the CFG-combined velocity and
    // record them for this step/branch. Folded in from the former standalone
    // sencache_calibrate(): both are one-sided differences in sigma-space,
    //   j_t = ||pred(x, sigma_next) - pred(x, sigma)|| / |sigma_next - sigma|
    //   j_z = ||pred(x_next,  sigma) - pred(x, sigma)|| / ||x_next - x||
    // with x_next = x + pred * (sigma_next - sigma) one Euler step ahead.
    void calibrate_step(const CalibrationContext& ctx) override {
        if (!calibrating_ || !ctx.forward_at || ctx.latent == nullptr || ctx.prediction == nullptr) {
            return;
        }
        const sd::Tensor<float>& x = *ctx.latent;
        const sd::Tensor<float>& pred = *ctx.prediction;
        if (x.empty() || pred.empty()) {
            return;
        }
        const float dt = ctx.step.sigma_next - ctx.step.sigma;
        if (std::fabs(dt) < 1e-8f) {
            return;
        }
        // J_t: perturb the timestep, hold the latent fixed.
        const sd::Tensor<float> pred_t = ctx.forward_at(x, ctx.step.sigma_next);
        const double dpred_t = detail::l2_diff(pred_t, pred);
        if (dpred_t < 0.0) {
            return;
        }
        // J_z: perturb the latent one Euler step ahead, hold the timestep.
        sd::Tensor<float> x_next = x + pred * dt;
        const double dz = detail::l2_diff(x_next, x);
        if (dz < 1e-8) {
            return;
        }
        const sd::Tensor<float> pred_z = ctx.forward_at(x_next, ctx.step.sigma);
        const double dpred_z = detail::l2_diff(pred_z, pred);
        if (dpred_z < 0.0) {
            return;
        }
        const float j_t = static_cast<float>(dpred_t / std::fabs(static_cast<double>(dt)));
        const float j_z = static_cast<float>(dpred_z / dz);
        if (!std::isfinite(j_t) || !std::isfinite(j_z)) {
            return;
        }
        record_jacobians(ctx.condition_key, ctx.branch, j_z, j_t);
    }

    void end_step(const StepContext& step) override {
        if (calibrating_ && step.step_index == num_steps_ - 1) {
            finalize_calibration();
        }
    }

    void log_summary(size_t total_steps) const override {
        if (!enabled() || total_steps == 0 || calibrating_) {
            return;
        }
        if (total_steps_skipped_ > 0 && total_steps_skipped_ < static_cast<int>(total_steps)) {
            const double speedup = static_cast<double>(total_steps) /
                                   static_cast<double>(total_steps - total_steps_skipped_);
            LOG_INFO("SenCache reused %d/%zu steps (%.2fx)", total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("SenCache reused %d/%zu steps", total_steps_skipped_, total_steps);
        }
    }

    void reset() override {
        states_.clear();
        total_steps_skipped_ = 0;
        current_step_index_ = -1;
        observed_any_ = false;
    }

private:
    struct Branch {
        std::vector<float> cached_z;
        float cached_sigma = 0.0f;
        float cached_j_z = 0.0f;
        float cached_j_t = 0.0f;
        bool has_reference = false;
        int accumulated_skips = 0;
        std::vector<float> residual;
        std::vector<int64_t> shape;
        bool has_residual = false;
        std::vector<float> calib_j_z;
        std::vector<float> calib_j_t;
        std::vector<float> calib_sigma;
        CacheBranch branch = CacheBranch::Main;
    };

    void record_jacobians(const void* condition_key, CacheBranch branch, float j_z, float j_t) {
        if (!calibrating_) {
            return;
        }
        Branch& b = branch_for(condition_key);
        b.branch = branch;
        if (b.calib_j_z.empty()) {
            b.calib_j_z.assign(static_cast<size_t>(std::max(0, num_steps_)),
                               std::numeric_limits<float>::quiet_NaN());
            b.calib_j_t = b.calib_j_z;
            b.calib_sigma = b.calib_j_z;
        }
        const int step = current_step_index_;
        if (step < 0 || step >= static_cast<int>(b.calib_j_z.size())) {
            return;
        }
        observed_any_ = true;
        b.calib_j_z[static_cast<size_t>(step)] = j_z;
        b.calib_j_t[static_cast<size_t>(step)] = j_t;
        b.calib_sigma[static_cast<size_t>(step)] =
            (step >= 0 && step < static_cast<int>(sigmas_.size())) ? sigmas_[static_cast<size_t>(step)] : 0.0f;
    }

    void lookup_jacobians(float sigma, float* out_j_z, float* out_j_t) const {
        *out_j_z = 0.0f;
        *out_j_t = 0.0f;
        if (profile_.empty()) {
            return;
        }
        size_t best = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < profile_.sigmas.size(); ++i) {
            const float dist = std::fabs(profile_.sigmas[i] - sigma);
            if (dist < best_dist) {
                best_dist = dist;
                best = i;
            }
        }
        *out_j_z = profile_.j_z[best];
        *out_j_t = profile_.j_t[best];
    }

    void finalize_calibration() {
        if (!observed_any_) {
            LOG_ERROR("SenCache calibration produced no data: the feature seam was "
                      "unavailable (disable CFG-parallel/VACE and use a single GPU). "
                      "No profile written to %s.", config_.calibrate_path.c_str());
            return;
        }
        const Branch* chosen = nullptr;
        for (CacheBranch want : {CacheBranch::Cond, CacheBranch::Main, CacheBranch::Uncond}) {
            for (const auto& kv : states_) {
                const Branch& cand = kv.second;
                if (cand.branch == want && !cand.calib_j_z.empty() && (chosen == nullptr)) {
                    chosen = &cand;
                }
            }
            if (chosen != nullptr) {
                break;
            }
        }
        if (chosen == nullptr || chosen->calib_j_z.empty()) {
            LOG_ERROR("SenCache calibration found no branch series; no profile written to %s.",
                      config_.calibrate_path.c_str());
            return;
        }
        for (int i = 0; i < num_steps_; ++i) {
            if (!std::isfinite(chosen->calib_j_z[static_cast<size_t>(i)]) ||
                !std::isfinite(chosen->calib_j_t[static_cast<size_t>(i)])) {
                LOG_ERROR("SenCache calibration missing step %d/%d; no profile written to %s.",
                          i, num_steps_, config_.calibrate_path.c_str());
                return;
            }
        }
        if (save_sencache_profile(config_.calibrate_path, model_name_,
                                  chosen->calib_sigma, chosen->calib_j_z, chosen->calib_j_t)) {
            LOG_INFO("SenCache calibration wrote %d-step profile to %s",
                     num_steps_, config_.calibrate_path.c_str());
        }
    }

    SenCacheConfig config_;
    std::string model_name_;
    bool initialized_ = false;
    bool calibrating_ = false;
    bool observed_any_ = false;
    int num_steps_ = 0;
    int retention_steps_ = 0;
    int switch_step_ = 0;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::vector<float> sigmas_;
    SenCacheProfile profile_;
    std::unordered_map<const void*, Branch> states_;

    Branch& branch_for(const void* cond) { return states_[cond]; }
};

}  // namespace

std::unique_ptr<ICachePolicy> create_sencache_policy() { return std::make_unique<SenCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
