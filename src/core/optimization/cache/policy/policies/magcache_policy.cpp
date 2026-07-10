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
#include "runtime/model_loader.h"
#include "utils/util.h"

namespace edgedit {
namespace cache {
namespace {

using detail::kVariantFull;
using detail::kVariantReuse;

// ===========================================================================
// MagCache precalibrated magnitude-ratio tables + profile I/O.
// Folded in from the former magcache_tables.hpp / magcache_profile.cpp: these
// are MagCache-only, so they live in this translation unit rather than as
// standalone files. Each table entry is ||residual_t|| / ||residual_{t-1}||
// measured offline on the reference (github.com/Zehong-Ma/MagCache). Tables are
// stored at native calibration length and resampled to the actual step count.
// ===========================================================================

// FLUX.1-dev, 28-step calibration.
inline const std::vector<float>& magcache_flux_table() {
    static const std::vector<float> table = {
        1.0f, 1.21094f, 1.11719f, 1.07812f, 1.0625f, 1.03906f, 1.03125f,
        1.03906f, 1.02344f, 1.03125f, 1.02344f, 0.98047f, 1.01562f, 1.00781f,
        1.0f, 1.00781f, 1.0f, 1.00781f, 1.0f, 1.0f, 0.99609f,
        0.99609f, 0.98047f, 0.98828f, 0.96484f, 0.95703f, 0.93359f, 0.89062f};
    return table;
}

// Qwen-Image, 50-step calibration (single-branch; cond/uncond share).
inline const std::vector<float>& magcache_qwen_table() {
    static const std::vector<float> table = {
        1.0f, 1.06934f, 1.04943f, 1.03855f, 1.02832f, 1.02637f, 1.02051f,
        1.01953f, 1.01758f, 1.01465f, 1.01367f, 1.01172f, 1.01074f, 1.00977f,
        1.00879f, 1.00781f, 1.00684f, 1.00684f, 1.00586f, 1.00488f, 1.00488f,
        1.00391f, 1.00391f, 1.00293f, 1.00293f, 1.00195f, 1.00195f, 1.00195f,
        1.00098f, 1.00098f, 1.00098f, 1.0f, 1.0f, 1.0f, 0.99951f,
        0.99902f, 0.99902f, 0.99854f, 0.99805f, 0.99756f, 0.99658f, 0.99609f,
        0.99512f, 0.99414f, 0.99268f, 0.99072f, 0.98828f, 0.98486f, 0.97949f,
        0.96924f};
    return table;
}

// Resample a native calibration table to `num_steps` via nearest-neighbour.
inline std::vector<float> magcache_resample(const std::vector<float>& table, int num_steps) {
    std::vector<float> out;
    if (num_steps <= 0) {
        return out;
    }
    out.resize(static_cast<size_t>(num_steps), 1.0f);
    if (table.empty()) {
        return out;
    }
    if (static_cast<int>(table.size()) == num_steps) {
        return table;
    }
    const int src = static_cast<int>(table.size());
    for (int i = 0; i < num_steps; ++i) {
        int idx = num_steps > 1
                      ? static_cast<int>(std::lround(static_cast<double>(i) * (src - 1) / (num_steps - 1)))
                      : 0;
        idx = std::max(0, std::min(src - 1, idx));
        out[static_cast<size_t>(i)] = table[static_cast<size_t>(idx)];
    }
    return out;
}

// Pick the calibrated table for a model version, resampled to num_steps.
// Falls back to neutral 1.0 for models without a table.
inline std::vector<float> magcache_table_for(SDVersion version, int num_steps) {
    if (ed_version_is_flux(version) || ed_version_is_flux2(version)) {
        return magcache_resample(magcache_flux_table(), num_steps);
    }
    if (ed_version_is_qwen_image(version)) {
        return magcache_resample(magcache_qwen_table(), num_steps);
    }
    return std::vector<float>(static_cast<size_t>(std::max(0, num_steps)), 1.0f);
}

// Write a user-calibrated per-step magnitude-ratio table to `path`.
bool save_magcache_profile(const std::string& path, const std::string& model,
                           int num_steps, const std::vector<float>& mag_ratios) {
    nlohmann::json out;
    out["type"] = "magcache_profile";
    out["model"] = model;
    out["num_steps"] = num_steps;
    out["mag_ratios"] = mag_ratios;
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("failed to open MagCache profile for writing: %s", path.c_str());
        return false;
    }
    file << out.dump(2) << "\n";
    if (!file.good()) {
        LOG_ERROR("failed to write MagCache profile: %s", path.c_str());
        return false;
    }
    return true;
}

// Load the native-length mag_ratios from `path`. Sets *ok=false and returns an
// empty vector on any failure; an empty table must never be treated as valid.
std::vector<float> load_magcache_profile(const std::string& path, bool* ok) {
    if (ok != nullptr) {
        *ok = false;
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_WARN("failed to open MagCache profile: %s", path.c_str());
        return {};
    }
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(file);
    } catch (const std::exception& e) {
        LOG_WARN("failed to parse MagCache profile '%s': %s", path.c_str(), e.what());
        return {};
    }
    if (!json.contains("mag_ratios") || !json["mag_ratios"].is_array()) {
        LOG_WARN("MagCache profile '%s' has no mag_ratios array", path.c_str());
        return {};
    }
    std::vector<float> ratios;
    try {
        ratios = json["mag_ratios"].get<std::vector<float>>();
    } catch (const std::exception& e) {
        LOG_WARN("MagCache profile '%s' has malformed mag_ratios: %s", path.c_str(), e.what());
        return {};
    }
    if (ratios.empty()) {
        LOG_WARN("MagCache profile '%s' has an empty mag_ratios table", path.c_str());
        return {};
    }
    if (json.contains("num_steps") && json["num_steps"].is_number_integer()) {
        const int declared = json["num_steps"].get<int>();
        if (declared != static_cast<int>(ratios.size())) {
            LOG_WARN("MagCache profile '%s' num_steps=%d disagrees with mag_ratios length %zu",
                     path.c_str(), declared, ratios.size());
            return {};
        }
    }
    if (ok != nullptr) {
        *ok = true;
    }
    return ratios;
}

// ===========================================================================
// MagCache (Feature granularity): magnitude-ratio error accumulation over a
// precalibrated per-step ratio table. Verified against Zehong-Ma/MagCache.
// Math ported verbatim from the old MagCachePolicy.
// ===========================================================================
class MagCachePolicy final : public ICachePolicy {
public:
    std::string_view name() const override { return "MagCache"; }
    CacheMode mode() const override { return CacheMode::MagCache; }
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
        config_ = inf.config->magcache;
        version_ = schema.version;
        model_name_ = cache_model_spec_for_version(schema.version).model_name;
        initialized_ = config_.enabled;
        num_steps_ = inf.num_steps;

        calibrating_ = !config_.calibrate_path.empty();
        observed_any_ = false;
        if (calibrating_) {
            mag_ratios_.assign(static_cast<size_t>(std::max(0, num_steps_)), 1.0f);
            LOG_INFO("MagCache calibration: profiling %d steps -> %s",
                     num_steps_, config_.calibrate_path.c_str());
        } else if (!config_.profile_path.empty()) {
            bool ok = false;
            std::vector<float> loaded = load_magcache_profile(config_.profile_path, &ok);
            if (ok && !loaded.empty()) {
                mag_ratios_ = magcache_resample(loaded, num_steps_);
                LOG_INFO("MagCache using calibrated profile %s (%zu -> %d steps)",
                         config_.profile_path.c_str(), loaded.size(), num_steps_);
            } else {
                mag_ratios_ = magcache_table_for(version_, num_steps_);
                LOG_WARN("MagCache profile %s unusable; falling back to built-in table",
                         config_.profile_path.c_str());
            }
        } else {
            mag_ratios_ = magcache_table_for(version_, num_steps_);
        }
        retention_steps_ = static_cast<int>(config_.retention_ratio * num_steps_ + 0.5f);
        states_.clear();
        total_steps_skipped_ = 0;

        const int seg = topo.block_stack() ? topo.block_stack()->id : 1;
        return detail::make_reuse_program("MagCache", seg, SegmentExecutionMode::LOAD_CACHED,
                                          detail::make_slot(0, "block_stack_residual"));
    }

    void begin_step(const StepContext& step) override { current_step_index_ = step.step_index; }

    RuntimeDecision decide(const StepContext&, const CacheRuntimeMetrics& m) override {
        RuntimeDecision d;
        d.variant = kVariantFull;
        if (!enabled() || calibrating_ || current_step_index_ < retention_steps_) {
            return d;
        }
        Branch& b = branch_for(m.condition_key);
        if (!b.has_residual) {
            return d;
        }
        const int cnt = current_step_index_;
        const float cur_scale = (cnt >= 0 && cnt < static_cast<int>(mag_ratios_.size()))
                                    ? mag_ratios_[static_cast<size_t>(cnt)]
                                    : 1.0f;
        const float accum_ratio = b.accumulated_ratio * cur_scale;
        const float accum_err = b.accumulated_err + std::fabs(1.0f - accum_ratio);
        const int accum_steps = b.accumulated_steps + 1;

        if (accum_err <= config_.mag_thresh && accum_steps <= config_.max_skip_steps) {
            b.accumulated_ratio = accum_ratio;
            b.accumulated_err = accum_err;
            b.accumulated_steps = accum_steps;
            d.variant = kVariantReuse;
            // Count the skip here (not in reconstruct) so the GPU inject path —
            // which bypasses reconstruct() — is also counted. The host path's
            // reconstruct() no longer increments; a rare reconstruct-fail falls
            // back to a full compute but is a negligible miscount.
            total_steps_skipped_++;
            return d;
        }
        b.accumulated_ratio = 1.0f;
        b.accumulated_err = 0.0f;
        b.accumulated_steps = 0;
        return d;
    }

    void observe(const CacheObservation& obs) override {
        if (!enabled()) {
            return;
        }
        // GPU feature reuse: the residual is resident on-device (not read back to
        // host). Mark it available so decide() will skip; the GPU inject path
        // supplies the data, so no host residual/shape is needed here.
        if (obs.feature_on_device) {
            Branch& b = branch_for(obs.condition_key);
            b.branch = obs.branch;
            b.has_residual = true;
            return;
        }
        if (obs.kind != CacheObservation::Kind::Feature || obs.feature == nullptr || obs.feature->empty()) {
            return;
        }
        Branch& b = branch_for(obs.condition_key);
        if (calibrating_) {
            b.branch = obs.branch;
            record_calibration_ratio(b, *obs.feature);
        }
        b.residual.assign(obs.feature->data(), obs.feature->data() + obs.feature->numel());
        b.shape = obs.feature->shape();
        b.has_residual = true;
    }

    sd::Tensor<float> reconstruct(const CacheReconstructContext& ctx) override {
        Branch& b = branch_for(ctx.condition_key);
        if (!b.has_residual) {
            return {};
        }
        sd::Tensor<float> out(b.shape);
        if (out.numel() != static_cast<int64_t>(b.residual.size())) {
            return {};
        }
        std::copy(b.residual.begin(), b.residual.end(), out.data());
        return out;
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
            LOG_INFO("MagCache reused %d/%zu steps (%.2fx)", total_steps_skipped_, total_steps, speedup);
        } else {
            LOG_INFO("MagCache reused %d/%zu steps", total_steps_skipped_, total_steps);
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
        float accumulated_ratio = 1.0f;
        float accumulated_err = 0.0f;
        int accumulated_steps = 0;
        std::vector<float> residual;
        std::vector<int64_t> shape;
        bool has_residual = false;
        std::vector<float> calib_ratios;
        int calib_count = 0;
        CacheBranch branch = CacheBranch::Main;
    };

    void record_calibration_ratio(Branch& b, const sd::Tensor<float>& feature) {
        if (b.calib_ratios.empty()) {
            b.calib_ratios.assign(static_cast<size_t>(std::max(0, num_steps_)),
                                  std::numeric_limits<float>::quiet_NaN());
        }
        const int step = current_step_index_;
        if (step < 0 || step >= static_cast<int>(b.calib_ratios.size())) {
            return;
        }
        observed_any_ = true;
        const auto& shape = feature.shape();
        const int64_t hidden = shape.empty() ? feature.numel() : shape[0];
        if (hidden <= 0) {
            return;
        }
        const int64_t total = feature.numel();
        const int64_t tokens = total / hidden;
        float ratio = 1.0f;
        if (step >= 1 && b.has_residual &&
            static_cast<int64_t>(b.residual.size()) == total && tokens > 0) {
            const float* cur = feature.data();
            const float* prev = b.residual.data();
            double sum_ratio = 0.0;
            int64_t counted = 0;
            for (int64_t t = 0; t < tokens; ++t) {
                const float* c = cur + t * hidden;
                const float* p = prev + t * hidden;
                double cur_norm_sq = 0.0;
                double prev_norm_sq = 0.0;
                for (int64_t h = 0; h < hidden; ++h) {
                    cur_norm_sq += static_cast<double>(c[h]) * c[h];
                    prev_norm_sq += static_cast<double>(p[h]) * p[h];
                }
                const double prev_norm = std::sqrt(prev_norm_sq);
                if (prev_norm > 1e-12) {
                    sum_ratio += std::sqrt(cur_norm_sq) / prev_norm;
                    ++counted;
                }
            }
            ratio = counted > 0 ? static_cast<float>(sum_ratio / static_cast<double>(counted)) : 1.0f;
        }
        b.calib_ratios[static_cast<size_t>(step)] = ratio;
        ++b.calib_count;
    }

    void finalize_calibration() {
        if (!observed_any_) {
            LOG_ERROR("MagCache calibration produced no data: the feature seam was "
                      "unavailable (disable CFG-parallel and use a single GPU). "
                      "No profile written to %s.", config_.calibrate_path.c_str());
            return;
        }
        const Branch* chosen = nullptr;
        for (CacheBranch want : {CacheBranch::Cond, CacheBranch::Main, CacheBranch::Uncond}) {
            for (const auto& kv : states_) {
                const Branch& cand = kv.second;
                if (cand.branch == want && cand.calib_count > 0 &&
                    (chosen == nullptr || cand.calib_count > chosen->calib_count)) {
                    chosen = &cand;
                }
            }
            if (chosen != nullptr) {
                break;
            }
        }
        if (chosen == nullptr || chosen->calib_ratios.empty()) {
            LOG_ERROR("MagCache calibration found no complete branch series; "
                      "no profile written to %s.", config_.calibrate_path.c_str());
            return;
        }
        std::vector<float> ratios = chosen->calib_ratios;
        for (size_t i = 0; i < ratios.size(); ++i) {
            if (!std::isfinite(ratios[i])) {
                LOG_ERROR("MagCache calibration missing step %zu/%d; no profile written to %s.",
                          i, num_steps_, config_.calibrate_path.c_str());
                return;
            }
        }
        ratios[0] = 1.0f;
        if (save_magcache_profile(config_.calibrate_path, model_name_, num_steps_, ratios)) {
            LOG_INFO("MagCache calibration wrote %d-step profile to %s",
                     num_steps_, config_.calibrate_path.c_str());
        }
    }

    MagCacheConfig config_;
    SDVersion version_ = VERSION_COUNT;
    std::string model_name_;
    bool initialized_ = false;
    bool calibrating_ = false;
    bool observed_any_ = false;
    int num_steps_ = 0;
    int retention_steps_ = 0;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::vector<float> mag_ratios_;
    std::unordered_map<const void*, Branch> states_;

    Branch& branch_for(const void* cond) { return states_[cond]; }
};

}  // namespace

std::unique_ptr<ICachePolicy> create_magcache_policy() { return std::make_unique<MagCachePolicy>(); }

}  // namespace cache
}  // namespace edgedit
