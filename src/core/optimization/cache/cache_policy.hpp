#pragma once

#include <cstddef>
#include <memory>

#include "core/optimization/cache/cache_config.hpp"
#include "core/optimization/cache/cache_types.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Metadata for one generation run, handed to a policy before the first step so
// it can size its schedule (warmup / skip intervals depend on num_steps).
struct CacheRunInfo {
    int num_steps = 0;
    int num_blocks = 0;
    bool separate_cfg = false;
    SDVersion version = VERSION_COUNT;
};

// A self-contained cache strategy. This is the ONLY abstraction a new cache
// method implements; it is pure host code with no knowledge of ggml or of any
// model graph. The CacheController drives these hooks and, for Feature/Probe
// policies, uses the model's build-time seam to capture/inject the block-stack
// residual on its behalf.
//
// Design note: edge-dit builds a fresh compute graph each step and executes it
// once, so "skip a block" is expressed as a build-time decision returned from
// plan_step()/decide_after_probe() rather than an eager per-block callback.
// The xLLM before/after_step + before/after_block lifecycle maps onto:
//   Output  policies: before_forward / after_forward (whole-model black box)
//   Feature policies: plan_step -> {Full: observe_feature | SkipStackReuse:
//                     reconstruct_feature}
//   Probe   policies: plan_step(Probe) -> decide_after_probe -> {skip|full}
struct CachePolicy {
    virtual ~CachePolicy() = default;

    virtual const char* name() const = 0;
    virtual bool enabled() const = 0;
    virtual CacheGranularity granularity() const = 0;

    virtual void init(const CacheConfig& config,
                      const CacheModelSpec& model_spec,
                      const std::vector<float>& sigmas) = 0;

    virtual void begin_run(const CacheRunInfo& run) { (void)run; }
    virtual void begin_step(const CacheStepInfo& step) = 0;
    virtual void end_step(const CacheStepInfo& step) = 0;

    // ---- Output granularity (whole transformer in -> out) ----
    // Returns true and fills *output if the step can be served from cache.
    virtual bool before_forward(const CacheForwardContext& frame,
                                const sd::Tensor<float>& input,
                                sd::Tensor<float>* output) {
        (void)frame;
        (void)input;
        (void)output;
        return false;
    }
    virtual void after_forward(const CacheForwardContext& frame,
                               const sd::Tensor<float>& input,
                               const sd::Tensor<float>& output) {
        (void)frame;
        (void)input;
        (void)output;
    }

    // ---- Feature / Probe granularity (block-stack residual) ----
    // What to do this step. Default keeps the model on the full path.
    virtual CacheStepDecision plan_step(const CacheForwardContext& frame) {
        (void)frame;
        return {};
    }
    // Record the freshly captured block-stack residual after a full step.
    virtual void observe_feature(const CacheForwardContext& frame,
                                 const sd::Tensor<float>& feature) {
        (void)frame;
        (void)feature;
    }
    // Produce the residual to inject on a skip step (empty => cannot reuse).
    virtual sd::Tensor<float> reconstruct_feature(const CacheForwardContext& frame) {
        (void)frame;
        return {};
    }
    // Probe policies decide skip vs. compute once probe values are known.
    // `before` is the block-stack input, `probe` the hidden state after
    // probe_depth blocks.
    virtual CacheStepDecision decide_after_probe(const CacheForwardContext& frame,
                                                 const sd::Tensor<float>& before,
                                                 const sd::Tensor<float>& probe) {
        (void)before;
        (void)probe;
        return plan_step(frame);
    }

    virtual void log_summary(size_t total_steps) const = 0;
};

std::unique_ptr<CachePolicy> create_cache_policy(CacheMode mode);

}  // namespace cache
}  // namespace edgedit
