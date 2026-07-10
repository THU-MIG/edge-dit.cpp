#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string_view>

#include "core/optimization/cache/ir/cache_requirements.hpp"
#include "core/optimization/cache/ir/cache_program.hpp"
#include "core/optimization/cache/ir/runtime_decision.hpp"
#include "core/optimization/cache/model/model_schema.hpp"
#include "core/optimization/cache/model/model_topology.hpp"
#include "core/optimization/cache/cache_config.hpp"
#include "core/optimization/cache/cache_types.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Everything a policy learns at compile() time that isn't in the schema.
struct InferenceConfig {
    const CacheConfig* config = nullptr;   // parsed CLI/config knobs
    const std::vector<float>* sigmas = nullptr;
    int num_steps = 0;
    bool separate_cfg = false;
};

// What the engine hands a policy when a captured value becomes available, so
// the policy can update its host state (residual history, derivative buffers,
// probe trajectory). Unifies the old observe_feature / probe-readback callbacks.
struct CacheObservation {
    enum class Kind { Feature, Probe } kind = Kind::Feature;

    StepContext step;
    const void* condition_key = nullptr;
    CacheBranch branch = CacheBranch::Main;

    const sd::Tensor<float>* feature = nullptr;  // Feature: captured residual
    const sd::Tensor<float>* before = nullptr;   // Probe: region input
    const sd::Tensor<float>* probe = nullptr;    // Probe: shallow state
    const sd::Tensor<float>* input = nullptr;    // block-stack input latent
    // Feature on-GPU reuse: the residual was captured to device memory (not read
    // back to host), so `feature` is null but a reuse IS possible. The policy
    // should mark a residual available and rely on the GPU inject path.
    bool feature_on_device = false;
    // GPU DiCache: decision scalars computed on-device (NaN when the host path is
    // used and full before/probe tensors are provided instead).
    float delta_y = std::numeric_limits<float>::quiet_NaN();
    float delta_x = std::numeric_limits<float>::quiet_NaN();
    float gamma = std::numeric_limits<float>::quiet_NaN();
};

// Context for producing the residual to inject on a reuse/predict step.
struct CacheReconstructContext {
    StepContext step;
    const void* condition_key = nullptr;
    const sd::Tensor<float>* input = nullptr;
};

// What a policy needs from the pipeline during a --cache-calibrate run, declared
// so the pipeline drives exactly the right extra work without knowing the method.
// A method that profiles purely from the captured feature residual (MagCache)
// needs nothing here; one that measures finite-difference sensitivities
// (SenCache) sets needs_forward_evaluator so the pipeline wires forward_at.
struct CalibrationSpec {
    bool active = false;             // this run is calibrating this method
    bool needs_forward_evaluator = false;  // policy will call ctx.forward_at
};

// Handed to calibrate_step(). Carries the current step's latent and the model
// prediction already computed this step, plus a model-specific evaluator the
// policy may call to probe the model at a perturbed (latent, sigma). The
// evaluator is the calibration analogue of CacheRunnerHooks::full — the pipeline
// owns input construction / CFG combine; the policy owns the calibration math.
struct CalibrationContext {
    StepContext step;
    const void* condition_key = nullptr;
    CacheBranch branch = CacheBranch::Main;
    const sd::Tensor<float>* latent = nullptr;      // raw latent x this step
    const sd::Tensor<float>* prediction = nullptr;  // model output at (x, sigma)
    // Evaluate the model's velocity prediction at a raw latent and sigma, with
    // the pipeline's own input scaling + CFG combine. Empty result => failure.
    std::function<sd::Tensor<float>(const sd::Tensor<float>&, float)> forward_at;
};

// The single abstraction a new cache method implements. Pure host code: no ggml,
// no model classes. requirements() drives capability negotiation; compile()
// emits the declarative program; decide() picks a variant per step; observe()
// folds captured state back in; reconstruct() produces the residual to inject on
// a reuse step (host-side for Option A — the deferred backend path lowers this
// into the program's PREDICT/BLEND operators instead).
class ICachePolicy {
public:
    virtual ~ICachePolicy() = default;

    virtual std::string_view name() const = 0;
    virtual CacheMode mode() const = 0;
    virtual CacheRequirements requirements() const = 0;

    // True when the method consumes a precalibrated table and can profile one
    // from a forced-full run (--cache-calibrate).
    virtual bool supports_calibration() const { return false; }

    // What this run's calibration needs from the pipeline (empty by default).
    // Meaningful only while calibrating; the engine reports active=false
    // otherwise so pipelines skip the extra work.
    virtual CalibrationSpec calibration_spec() const { return {}; }

    // Run this method's calibration protocol for one step/branch, using the
    // pipeline-supplied evaluator. Owns whatever the method measures (e.g.
    // SenCache's finite-difference Jacobian norms). No-op by default.
    virtual void calibrate_step(const CalibrationContext& ctx) { (void)ctx; }

    // Whether the policy actually activated (config enabled + prerequisites met,
    // e.g. SenCache found a usable profile). A disabled policy is treated as
    // no-cache by the engine.
    virtual bool enabled() const = 0;

    // Emit the declarative program (slots + graph variants) for this run.
    virtual CacheProgram compile(const ModelSchema& schema,
                                 const ModelTopology& topology,
                                 const InferenceConfig& inference) = 0;

    // Per-step lifecycle (runs once per step, before/after all CFG branches).
    virtual void begin_step(const StepContext& step) = 0;
    virtual void end_step(const StepContext& step) = 0;

    // Per-branch: which variant to run. metrics carries probe/prior results.
    virtual RuntimeDecision decide(const StepContext& step,
                                   const CacheRuntimeMetrics& metrics) = 0;

    // A probe pass ran; decide the post-probe variant (default: honor decide()).
    virtual RuntimeDecision decide_after_probe(const StepContext& step,
                                               const CacheObservation& probe) {
        (void)probe;
        return decide(step, {});
    }

    // Fold a captured value into host state.
    virtual void observe(const CacheObservation& obs) { (void)obs; }

    // Produce the residual to inject on a reuse step (empty => cannot reuse).
    virtual sd::Tensor<float> reconstruct(const CacheReconstructContext& ctx) {
        (void)ctx;
        return {};
    }

    virtual void log_summary(size_t total_steps) const { (void)total_steps; }
    virtual void reset() = 0;
};

std::unique_ptr<ICachePolicy> create_cache_policy(CacheMode mode);

}  // namespace cache
}  // namespace edgedit
