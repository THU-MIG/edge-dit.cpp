
#include "core/optimization/cache/cache_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "core/optimization/cache/tensor_cache_utils.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {

// Declared in cache_policy_predictive.cpp (MagCache / DiCache).
std::unique_ptr<CachePolicy> create_magcache_policy();
std::unique_ptr<CachePolicy> create_dicache_policy();

namespace {

// Finite-difference Taylor extrapolation state. Ported verbatim from the prior
// cache_methods.cpp; the math is correct and shared by TaylorSeer (feature) and
// the CacheDiT combined policy (output).
struct TaylorSeerState {
    int n_derivatives = 1;
    int current_step = -1;
    int last_computed_step = -1;
    std::vector<std::vector<float>> dy_prev;
    std::vector<std::vector<float>> dy_current;

    void init(int n_deriv) {
        n_derivatives = std::max(1, n_deriv);
        const int order = n_derivatives + 1;
        dy_prev.assign(order, {});
        dy_current.assign(order, {});
        current_step = -1;
        last_computed_step = -1;
    }

    void reset() {
        for (auto& v : dy_prev) {
            v.clear();
        }
        for (auto& v : dy_current) {
            v.clear();
        }
        current_step = -1;
        last_computed_step = -1;
    }

    bool can_approximate(size_t size) const {
        return last_computed_step >= n_derivatives && !dy_current.empty() && dy_current[0].size() == size;
    }

    // True once enough full steps have been observed to extrapolate at all
    // (independent of tensor size — used by plan_step before a size is known).
    bool ready() const {
        return last_computed_step >= n_derivatives && !dy_current.empty() && !dy_current[0].empty();
    }

    void update_derivatives(const float* y, size_t size, int step) {
        if (y == nullptr || size == 0) {
            return;
        }

        dy_prev = dy_current;
        dy_current[0].assign(y, y + size);

        int window = step - last_computed_step;
        if (window <= 0) {
            window = 1;
        }

        for (int d = 0; d < n_derivatives; ++d) {
            if (!dy_prev[d].empty() && dy_prev[d].size() == size) {
                dy_current[d + 1].resize(size);
                for (size_t i = 0; i < size; ++i) {
                    dy_current[d + 1][i] = (dy_current[d][i] - dy_prev[d][i]) / static_cast<float>(window);
                }
            } else {
                dy_current[d + 1].clear();
            }
        }

        current_step = step;
        last_computed_step = step;
    }

    bool approximate(sd::Tensor<float>* output, const std::vector<int64_t>& shape, int target_step) const {
        if (output == nullptr) {
            return false;
        }
        const size_t size = dy_current.empty() ? 0 : dy_current[0].size();
        if (!can_approximate(size)) {
            return false;
        }

        *output = sd::Tensor<float>(shape);
        float* data = output->data();
        if (data == nullptr) {
            return false;
        }

        int elapsed = target_step - last_computed_step;
        if (elapsed <= 0) {
            elapsed = 1;
        }

        std::fill(data, data + size, 0.0f);
        float factorial = 1.0f;
        const int order = static_cast<int>(dy_current.size());
        for (int o = 0; o < order; ++o) {
            if (dy_current[o].empty() || dy_current[o].size() != size) {
                continue;
            }
            if (o > 0) {
                factorial *= static_cast<float>(o);
            }
            const float coeff = std::pow(static_cast<float>(elapsed), static_cast<float>(o)) / factorial;
            for (size_t i = 0; i < size; ++i) {
                data[i] += coeff * dy_current[o][i];
            }
        }
        return true;
    }
};

const char* method_label_for_mode(CacheMode mode) {
    switch (mode) {
        case CacheMode::DBCache:
            return "DBCache";
        case CacheMode::CacheDiT:
            return "CacheDiT";
        default:
            return cache_mode_name(mode);
    }
}

class NullCachePolicy final : public CachePolicy {
public:
    const char* name() const override { return "None"; }
    bool enabled() const override { return false; }
    CacheGranularity granularity() const override { return CacheGranularity::Output; }
    void init(const CacheConfig&, const CacheModelSpec&, const std::vector<float>&) override {}
    void begin_step(const CacheStepInfo&) override {}
    void end_step(const CacheStepInfo&) override {}
    void log_summary(size_t) const override {}
};

class EasyCachePolicy final : public CachePolicy {
public:
    const char* name() const override { return "EasyCache"; }
    bool enabled() const override { return initialized_ && config_.enabled; }
    CacheGranularity granularity() const override { return CacheGranularity::Output; }

    void init(const CacheConfig& config, const CacheModelSpec&, const std::vector<float>& sigmas) override {
        config_ = config.easycache;
        initialized_ = config_.enabled;
        reset_runtime();
        set_sigmas(sigmas);
    }

    void begin_step(const CacheStepInfo& step) override {
        if (!enabled() || step.step_index == current_step_index_) {
            return;
        }
        current_step_index_ = step.step_index;
        skip_current_step_ = false;
        has_last_input_change_ = false;
        step_active_ = in_active_sigma_window(step.sigma);
    }

    bool before_forward(const CacheForwardContext& frame,
                        const sd::Tensor<float>& input,
                        sd::Tensor<float>* output) override {
        if (!enabled() || frame.step.step_index < 0 || output == nullptr) {
            return false;
        }
        if (frame.step.step_index != current_step_index_) {
            begin_step(frame.step);
        }
        if (!step_active_) {
            return false;
        }

        const void* cond = frame.condition_key;
        if (initial_step_) {
            anchor_condition_ = cond;
            initial_step_ = false;
        }

        if (skip_current_step_) {
            return has_cache(cond) && apply_cache(cond, input, output);
        }
        if (cond != anchor_condition_ || !has_prev_input_ || !has_prev_output_ || !has_cache(cond)) {
            return false;
        }

        const size_t ne = static_cast<size_t>(input.numel());
        if (prev_input_.size() != ne) {
            return false;
        }

        const float* input_data = input.data();
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
                return apply_cache(cond, input, output);
            }
            cumulative_change_rate_ = 0.0f;
        }

        return false;
    }

    void after_forward(const CacheForwardContext& frame,
                       const sd::Tensor<float>& input,
                       const sd::Tensor<float>& output) override {
        if (!step_is_active()) {
            return;
        }
        if (input.numel() != output.numel()) {
            return;
        }
        update_cache(frame.condition_key, input, output);
        if (frame.condition_key != anchor_condition_) {
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

        output_prev_norm_ = mean_abs(out_data, ne);
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

    void end_step(const CacheStepInfo&) override {}

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

private:
    struct CacheEntry {
        std::vector<float> diff;
    };

    EasyCacheConfig config_;
    bool initialized_ = false;
    bool initial_step_ = true;
    bool skip_current_step_ = false;
    bool step_active_ = false;
    const void* anchor_condition_ = nullptr;
    std::unordered_map<const void*, CacheEntry> cache_diffs_;
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
    float start_sigma_ = std::numeric_limits<float>::max();
    float end_sigma_ = 0.0f;

    void reset_runtime() {
        initial_step_ = true;
        skip_current_step_ = false;
        step_active_ = false;
        anchor_condition_ = nullptr;
        cache_diffs_.clear();
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

    void set_sigmas(const std::vector<float>& sigmas) {
        if (sigmas.size() < 2) {
            return;
        }
        size_t n_steps = sigmas.size() - 1;
        size_t start_step = static_cast<size_t>(config_.start_percent * n_steps);
        size_t end_step = static_cast<size_t>(config_.end_percent * n_steps);
        if (start_step >= n_steps) {
            start_step = n_steps - 1;
        }
        if (end_step >= n_steps) {
            end_step = n_steps - 1;
        }
        start_sigma_ = sigmas[start_step];
        end_sigma_ = sigmas[end_step];
        if (start_sigma_ < end_sigma_) {
            std::swap(start_sigma_, end_sigma_);
        }
    }

    bool in_active_sigma_window(float sigma) const {
        return sigma <= start_sigma_ && sigma > end_sigma_;
    }

    bool step_is_active() const {
        return enabled() && step_active_;
    }

    bool has_cache(const void* cond) const {
        auto it = cache_diffs_.find(cond);
        return it != cache_diffs_.end() && !it->second.diff.empty();
    }

    void update_cache(const void* cond, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        store_tensor_diff(&cache_diffs_[cond].diff, input, output);
    }

    bool apply_cache(const void* cond, const sd::Tensor<float>& input, sd::Tensor<float>* output) {
        auto it = cache_diffs_.find(cond);
        return it != cache_diffs_.end() && apply_tensor_diff(it->second.diff, input, output);
    }

    static float mean_abs(const float* data, size_t ne) {
        if (data == nullptr || ne == 0) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < ne; ++i) {
            sum += std::fabs(data[i]);
        }
        return sum / static_cast<float>(ne);
    }
};

class UCachePolicy final : public CachePolicy {
public:
    const char* name() const override { return "UCache"; }
    bool enabled() const override { return initialized_ && config_.enabled; }
    CacheGranularity granularity() const override { return CacheGranularity::Output; }

    void init(const CacheConfig& config, const CacheModelSpec&, const std::vector<float>& sigmas) override {
        config_ = config.ucache;
        initialized_ = config_.enabled;
        reset_runtime();
        set_sigmas(sigmas);
    }

    void begin_step(const CacheStepInfo& step) override {
        if (!enabled() || step.step_index == current_step_index_) {
            return;
        }
        current_step_index_ = step.step_index;
        skip_current_step_ = false;
        has_last_input_change_ = false;
        step_active_ = in_active_sigma_window(step.sigma);
        if (step_active_) {
            total_active_steps_++;
        }
    }

    bool before_forward(const CacheForwardContext& frame,
                        const sd::Tensor<float>& input,
                        sd::Tensor<float>* output) override {
        if (!enabled() || frame.step.step_index < 0 || output == nullptr) {
            return false;
        }
        if (frame.step.step_index != current_step_index_) {
            begin_step(frame.step);
        }
        if (!step_active_) {
            return false;
        }

        const void* cond = frame.condition_key;
        if (initial_step_) {
            anchor_condition_ = cond;
            initial_step_ = false;
        }

        if (skip_current_step_) {
            return has_cache(cond) && apply_cache(cond, input, output);
        }
        if (cond != anchor_condition_ || !has_prev_input_ || !has_prev_output_ || !has_cache(cond)) {
            return false;
        }

        const size_t ne = static_cast<size_t>(input.numel());
        if (prev_input_.size() != ne) {
            return false;
        }

        const float* input_data = input.data();
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
                return apply_cache(cond, input, output);
            }
            if (config_.reset_error_on_compute) {
                accumulated_error_ = 0.0f;
            }
        }

        return false;
    }

    void after_forward(const CacheForwardContext& frame,
                       const sd::Tensor<float>& input,
                       const sd::Tensor<float>& output) override {
        if (!step_is_active()) {
            return;
        }
        if (input.numel() != output.numel()) {
            return;
        }

        update_cache(frame.condition_key, input, output);
        if (frame.condition_key != anchor_condition_) {
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

        output_prev_norm_ = mean_abs(out_data, ne);
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

    void end_step(const CacheStepInfo&) override {}

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

private:
    struct CacheEntry {
        std::vector<float> diff;
    };

    UCacheConfig config_;
    bool initialized_ = false;
    bool initial_step_ = true;
    bool skip_current_step_ = false;
    bool step_active_ = false;
    const void* anchor_condition_ = nullptr;
    std::unordered_map<const void*, CacheEntry> cache_diffs_;
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
    float start_sigma_ = std::numeric_limits<float>::max();
    float end_sigma_ = 0.0f;

    void reset_runtime() {
        initial_step_ = true;
        skip_current_step_ = false;
        step_active_ = false;
        anchor_condition_ = nullptr;
        cache_diffs_.clear();
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
        expected_total_steps_ = 0;
        consecutive_skipped_steps_ = 0;
        accumulated_error_ = 0.0f;
        total_active_steps_ = 0;
    }

    void set_sigmas(const std::vector<float>& sigmas) {
        if (sigmas.size() < 2) {
            return;
        }
        size_t n_steps = sigmas.size() - 1;
        expected_total_steps_ = static_cast<int>(n_steps);
        size_t start_step = static_cast<size_t>(config_.start_percent * n_steps);
        size_t end_step = static_cast<size_t>(config_.end_percent * n_steps);
        if (start_step >= n_steps) {
            start_step = n_steps - 1;
        }
        if (end_step >= n_steps) {
            end_step = n_steps - 1;
        }
        start_sigma_ = sigmas[start_step];
        end_sigma_ = sigmas[end_step];
        if (start_sigma_ < end_sigma_) {
            std::swap(start_sigma_, end_sigma_);
        }
    }

    bool in_active_sigma_window(float sigma) const {
        return sigma <= start_sigma_ && sigma > end_sigma_;
    }

    bool step_is_active() const {
        return enabled() && step_active_;
    }

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
        auto it = cache_diffs_.find(cond);
        return it != cache_diffs_.end() && !it->second.diff.empty();
    }

    void update_cache(const void* cond, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        store_tensor_diff(&cache_diffs_[cond].diff, input, output);
    }

    bool apply_cache(const void* cond, const sd::Tensor<float>& input, sd::Tensor<float>* output) {
        auto it = cache_diffs_.find(cond);
        return it != cache_diffs_.end() && apply_tensor_diff(it->second.diff, input, output);
    }

    static float mean_abs(const float* data, size_t ne) {
        if (data == nullptr || ne == 0) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < ne; ++i) {
            sum += std::fabs(data[i]);
        }
        return sum / static_cast<float>(ne);
    }
};

// DBCache + CacheDiT: residual-diff gated whole-model reuse, optionally with a
// Taylor extrapolation of the whole output (CacheDiT). Output granularity.
// Ported from the prior CacheDiTConditionMethod (minus the unused plan_region).
class ConditionCachePolicy final : public CachePolicy {
public:
    const char* name() const override { return method_label_for_mode(mode_); }
    bool enabled() const override { return initialized_ && (db_config_.enabled || taylor_config_.enabled); }
    CacheGranularity granularity() const override { return CacheGranularity::Output; }

    void init(const CacheConfig& config, const CacheModelSpec& model_spec, const std::vector<float>& sigmas) override {
        mode_ = config.mode;
        db_config_ = config.dbcache;
        taylor_config_ = config.taylorseer;
        model_spec_ = model_spec;
        initialized_ = db_config_.enabled || taylor_config_.enabled;
        reset_runtime();
        set_sigmas(sigmas);
    }

    void begin_step(const CacheStepInfo& step) override {
        if (!enabled() || step.step_index == current_step_index_) {
            return;
        }

        current_step_index_ = step.step_index;
        skip_current_step_ = false;
        step_active_ = false;
        warmup_step_ = false;
        can_cache_this_step_ = false;
        static_cache_step_ = false;

        if (step.sigma > start_sigma_ || !(step.sigma > end_sigma_)) {
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

    bool before_forward(const CacheForwardContext& frame,
                        const sd::Tensor<float>& input,
                        sd::Tensor<float>* output) override {
        if (!enabled() || frame.step.step_index < 0 || output == nullptr) {
            return false;
        }
        if (frame.step.step_index != current_step_index_) {
            begin_step(frame.step);
        }
        if (!step_active_ || (warmup_step_ && !static_cache_step_)) {
            return false;
        }

        const void* cond = frame.condition_key;
        if (initial_step_) {
            anchor_condition_ = cond;
            initial_step_ = false;
        }

        if (skip_current_step_) {
            if (taylor_config_.enabled && approximate_with_taylor(cond, output, input.shape())) {
                return true;
            }
            return has_cache(cond) && apply_cache(cond, input, output);
        }

        if ((mode_ == CacheMode::TaylorSeer || !db_config_.enabled) && taylor_config_.enabled &&
            can_cache_this_step_ && should_use_taylor_this_step() &&
            cond == anchor_condition_ &&
            approximate_with_taylor(cond, output, input.shape())) {
            skip_current_step_ = true;
            total_steps_skipped_++;
            record_cached_step(0.0f);
            return true;
        }

        if (!db_config_.enabled || !can_cache_this_step_ || cond != anchor_condition_) {
            return false;
        }

        auto it = cache_diffs_.find(cond);
        if (it == cache_diffs_.end() || !it->second.has_prev) {
            return false;
        }

        const size_t ne = static_cast<size_t>(input.numel());
        if (it->second.prev_input.size() != ne) {
            return false;
        }

        const float diff = residual_diff(it->second.prev_input.data(), input.data(), ne);
        if (static_cache_step_ || diff < db_config_.residual_diff_threshold) {
            skip_current_step_ = true;
            total_steps_skipped_++;
            record_cached_step(diff);
            if (taylor_config_.enabled && approximate_with_taylor(cond, output, input.shape())) {
                return true;
            }
            return apply_cache(cond, input, output);
        }

        continuous_cached_steps_ = 0;
        return false;
    }

    void after_forward(const CacheForwardContext& frame,
                       const sd::Tensor<float>& input,
                       const sd::Tensor<float>& output) override {
        if (!step_is_active()) {
            return;
        }
        if (input.numel() != output.numel()) {
            return;
        }

        update_cache(frame.condition_key, input, output);

        if (taylor_config_.enabled) {
            TaylorSeerState& state = taylor_state_for(frame.condition_key);
            state.update_derivatives(output.data(), static_cast<size_t>(output.numel()), current_step_index_);
        }
    }

    void end_step(const CacheStepInfo&) override {
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
                     name(), total_steps_skipped_, total_steps, speedup, accumulated_residual_diff_);
        } else {
            LOG_INFO("%s skipped %d/%zu steps, accum_diff=%.4f",
                     name(), total_steps_skipped_, total_steps, accumulated_residual_diff_);
        }
    }

private:
    struct CacheEntry {
        std::vector<float> diff;
        std::vector<float> prev_input;
        std::vector<float> prev_output;
        bool has_prev = false;
    };

    CacheMode mode_ = CacheMode::CacheDiT;
    DBCacheConfig db_config_;
    TaylorSeerConfig taylor_config_;
    CacheModelSpec model_spec_;
    bool initialized_ = false;
    int current_step_index_ = -1;
    bool step_active_ = false;
    bool warmup_step_ = false;
    bool can_cache_this_step_ = false;
    bool static_cache_step_ = false;
    bool skip_current_step_ = false;
    bool initial_step_ = true;
    int warmup_remaining_ = 0;
    std::vector<int> cached_steps_;
    int continuous_cached_steps_ = 0;
    float accumulated_residual_diff_ = 0.0f;
    int total_steps_skipped_ = 0;
    const void* anchor_condition_ = nullptr;
    std::unordered_map<const void*, CacheEntry> cache_diffs_;
    std::unordered_map<const void*, TaylorSeerState> taylor_states_;
    float start_sigma_ = std::numeric_limits<float>::max();
    float end_sigma_ = 0.0f;

    void reset_runtime() {
        current_step_index_ = -1;
        step_active_ = false;
        warmup_step_ = false;
        can_cache_this_step_ = false;
        static_cache_step_ = false;
        skip_current_step_ = false;
        initial_step_ = true;
        warmup_remaining_ = db_config_.max_warmup_steps;
        cached_steps_.clear();
        continuous_cached_steps_ = 0;
        accumulated_residual_diff_ = 0.0f;
        total_steps_skipped_ = 0;
        anchor_condition_ = nullptr;
        cache_diffs_.clear();
        taylor_states_.clear();
    }

    void set_sigmas(const std::vector<float>& sigmas) {
        if (sigmas.size() < 2) {
            return;
        }
        const size_t n_steps = sigmas.size() - 1;
        const float start_percent = db_config_.enabled ? db_config_.start_percent : taylor_config_.start_percent;
        const float end_percent = db_config_.enabled ? db_config_.end_percent : taylor_config_.end_percent;
        size_t start_step = static_cast<size_t>(start_percent * n_steps);
        size_t end_step = static_cast<size_t>(end_percent * n_steps);
        if (start_step >= n_steps) {
            start_step = n_steps - 1;
        }
        if (end_step >= n_steps) {
            end_step = n_steps - 1;
        }
        start_sigma_ = sigmas[start_step];
        end_sigma_ = sigmas[end_step];
        if (start_sigma_ < end_sigma_) {
            std::swap(start_sigma_, end_sigma_);
        }
    }

    bool step_is_active() const {
        return enabled() && step_active_;
    }

    bool has_cache(const void* cond) const {
        auto it = cache_diffs_.find(cond);
        return it != cache_diffs_.end() && !it->second.diff.empty();
    }

    void update_cache(const void* cond, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        CacheEntry& entry = cache_diffs_[cond];
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

    bool apply_cache(const void* cond, const sd::Tensor<float>& input, sd::Tensor<float>* output) {
        auto it = cache_diffs_.find(cond);
        return it != cache_diffs_.end() && apply_tensor_diff(it->second.diff, input, output);
    }

    TaylorSeerState& taylor_state_for(const void* cond) {
        TaylorSeerState& state = taylor_states_[cond];
        if (state.dy_current.empty()) {
            state.init(taylor_config_.n_derivatives);
        }
        return state;
    }

    bool approximate_with_taylor(const void* cond,
                                 sd::Tensor<float>* output,
                                 const std::vector<int64_t>& shape) {
        auto it = taylor_states_.find(cond);
        if (it == taylor_states_.end()) {
            return false;
        }
        return it->second.approximate(output, shape, current_step_index_);
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
};

// TaylorSeer as a FEATURE-level policy: extrapolates the block-stack residual
// captured by the model seam (faithful to the reference), instead of the
// black-box whole-model output. Skip decision is a fixed warmup+interval
// schedule, so it is deterministic across CFG-parallel ranks.
class TaylorSeerFeaturePolicy final : public CachePolicy {
public:
    const char* name() const override { return "TaylorSeer"; }
    bool enabled() const override { return initialized_ && config_.enabled; }
    CacheGranularity granularity() const override { return CacheGranularity::Feature; }

    void init(const CacheConfig& config, const CacheModelSpec&, const std::vector<float>& sigmas) override {
        config_ = config.taylorseer;
        initialized_ = config_.enabled;
        reset_runtime();
        set_sigmas(sigmas);
    }

    void begin_step(const CacheStepInfo& step) override {
        current_step_index_ = step.step_index;
        step_active_ = in_active_sigma_window(step.sigma);
    }

    CacheStepDecision plan_step(const CacheForwardContext& frame) override {
        CacheStepDecision decision;  // defaults to Full
        if (!enabled() || !step_active_) {
            return decision;
        }
        if (!should_extrapolate_this_step()) {
            return decision;
        }
        auto it = states_.find(frame.condition_key);
        if (it == states_.end() || !it->second.ready()) {
            return decision;  // not enough history yet -> full compute
        }
        decision.kind = CacheStepDecision::Kind::SkipStackReuse;
        return decision;
    }

    void observe_feature(const CacheForwardContext& frame,
                         const sd::Tensor<float>& feature) override {
        if (!enabled() || feature.empty()) {
            return;
        }
        TaylorSeerState& state = state_for(frame.condition_key);
        state.update_derivatives(feature.data(), static_cast<size_t>(feature.numel()), current_step_index_);
        last_feature_shape_[frame.condition_key] = feature.shape();
    }

    sd::Tensor<float> reconstruct_feature(const CacheForwardContext& frame) override {
        auto it = states_.find(frame.condition_key);
        auto shape_it = last_feature_shape_.find(frame.condition_key);
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

    void end_step(const CacheStepInfo&) override {}

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

private:
    TaylorSeerConfig config_;
    bool initialized_ = false;
    bool step_active_ = false;
    int current_step_index_ = -1;
    int total_steps_skipped_ = 0;
    std::unordered_map<const void*, TaylorSeerState> states_;
    std::unordered_map<const void*, std::vector<int64_t>> last_feature_shape_;
    float start_sigma_ = std::numeric_limits<float>::max();
    float end_sigma_ = 0.0f;

    void reset_runtime() {
        step_active_ = false;
        current_step_index_ = -1;
        total_steps_skipped_ = 0;
        states_.clear();
        last_feature_shape_.clear();
    }

    TaylorSeerState& state_for(const void* cond) {
        TaylorSeerState& state = states_[cond];
        if (state.dy_current.empty()) {
            state.init(config_.n_derivatives);
        }
        return state;
    }

    void set_sigmas(const std::vector<float>& sigmas) {
        if (sigmas.size() < 2) {
            return;
        }
        const size_t n_steps = sigmas.size() - 1;
        size_t start_step = static_cast<size_t>(config_.start_percent * n_steps);
        size_t end_step = static_cast<size_t>(config_.end_percent * n_steps);
        if (start_step >= n_steps) {
            start_step = n_steps - 1;
        }
        if (end_step >= n_steps) {
            end_step = n_steps - 1;
        }
        start_sigma_ = sigmas[start_step];
        end_sigma_ = sigmas[end_step];
        if (start_sigma_ < end_sigma_) {
            std::swap(start_sigma_, end_sigma_);
        }
    }

    bool in_active_sigma_window(float sigma) const {
        return sigma <= start_sigma_ && sigma > end_sigma_;
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
};

}  // namespace

std::unique_ptr<CachePolicy> create_cache_policy(CacheMode mode) {
    switch (mode) {
        case CacheMode::EasyCache:
            return std::make_unique<EasyCachePolicy>();
        case CacheMode::UCache:
            return std::make_unique<UCachePolicy>();
        case CacheMode::DBCache:
        case CacheMode::CacheDiT:
            return std::make_unique<ConditionCachePolicy>();
        case CacheMode::TaylorSeer:
            return std::make_unique<TaylorSeerFeaturePolicy>();
        case CacheMode::MagCache:
            return create_magcache_policy();
        case CacheMode::DiCache:
            return create_dicache_policy();
        case CacheMode::Disabled:
        default:
            return std::make_unique<NullCachePolicy>();
    }
}

}  // namespace cache
}  // namespace edgedit
