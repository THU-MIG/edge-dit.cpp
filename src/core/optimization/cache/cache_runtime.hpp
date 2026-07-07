#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "core/optimization/cache/cache_graph_scope.hpp"  // sd::DiffusionCacheResult
#include "core/optimization/cache/cache_policy.hpp"
#include "core/optimization/cache/cache_types.hpp"

namespace edgedit {
namespace cache {

// Model-agnostic compute callbacks the pipeline wires to its exact runner
// calls. The controller invokes whichever subset a policy's granularity needs.
//   full   - normal full transformer forward (Output path + all fallbacks).
//   input  - the block-stack input latent (needed by Output-level diff cache).
//   capture/inject/probe - the Feature/Probe seam passes (Layer C helpers).
// capture/inject take the cached region [start, end) (end < 0 => whole stack)
// so a policy can restrict the seam to a sub-interval of the block stack.
// feature_supported is false when the model can't cut into its block stack (or
// the step is gated onto the plain-only path), forcing a full-compute fallback.
struct CacheRunnerHooks {
    const sd::Tensor<float>* input = nullptr;
    std::function<sd::Tensor<float>()> full;
    bool feature_supported = false;
    std::function<sd::DiffusionCacheResult(int, int)> capture;
    std::function<sd::Tensor<float>(const sd::Tensor<float>&, int, int)> inject;
    std::function<sd::DiffusionCacheResult(int)> probe;
};

// Facade over a CachePolicy. Owns one policy instance; cond/uncond isolation is
// handled inside the policy via the condition_key. Replaces the old
// CacheRuntime: same init()/enabled()/begin_step()/end_step()/log_summary()
// surface, plus run_branch() which orchestrates the (possibly multi-pass)
// compute for one CFG branch this step.
class CacheController {
public:
    CacheController() = default;

    bool init(const ed_sample_params_t& sample_params,
              SDVersion version,
              const std::vector<float>& sigmas);

    bool enabled() const { return policy_ != nullptr && policy_->enabled(); }
    CacheMode mode() const { return config_.mode; }
    CacheGranularity granularity() const {
        return policy_ != nullptr ? policy_->granularity() : CacheGranularity::Output;
    }
    // True when the active policy needs the model's block-stack seam.
    bool needs_feature_seam() const {
        return enabled() && granularity() != CacheGranularity::Output;
    }

    void begin_step(const CacheStepInfo& step);
    void end_step(const CacheStepInfo& step);
    void log_summary(size_t total_steps) const;

    // Run one CFG branch this step, applying the cache policy. Returns the model
    // output for the branch (may be a cache reconstruction). Falls back to
    // hooks.full() whenever caching is unavailable or declines to reuse.
    sd::Tensor<float> run_branch(CacheBranch branch,
                                 const void* condition_key,
                                 const CacheRunnerHooks& hooks);

private:
    CacheConfig config_;
    CacheModelSpec model_spec_;
    CacheStepInfo current_step_;
    std::unique_ptr<CachePolicy> policy_;

    sd::Tensor<float> run_output_branch(const CacheForwardContext& frame,
                                        const CacheRunnerHooks& hooks);
    sd::Tensor<float> run_feature_branch(const CacheForwardContext& frame,
                                         const CacheRunnerHooks& hooks);
    sd::Tensor<float> run_probe_branch(const CacheForwardContext& frame,
                                       const CacheRunnerHooks& hooks);
};

// Transitional alias so existing includes keep compiling during migration.
using CacheRuntime = CacheController;

}  // namespace cache
}  // namespace edgedit
