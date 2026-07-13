#include "core/optimization/cache/compile/cache_graph_lowering.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "core/optimization/cache/ir/cache_action.hpp"

namespace edgedit {
namespace cache {

namespace {

// Whether a program carries any declarative action list. When true, the lowering
// interprets the chosen variant's actions over the CacheStateManager + operator
// registry instead of calling the policy's host reconstruct()/observe()
// callbacks. This is the per-method switch that lets Output methods migrate onto
// the declarative machinery one at a time under a single lowering.
bool variant_has_actions(const CacheProgram& program) {
    for (const auto& v : program.variants) {
        for (const auto& seg : v.segments) {
            if (!seg.before.empty() || !seg.after.empty()) {
                return true;
            }
        }
    }
    return false;
}

// Host-side interpreter for one segment's action list. Resolves ValueRefs to
// tensors (ambient inputs, cache slots via the state manager, or local temps),
// runs each action's operator, and reports success. A failed LOAD (no residual
// yet) or operator precondition returns false so the caller falls back to a full
// compute — the declarative analogue of reconstruct() returning empty.
class ActionInterpreter {
public:
    ActionInterpreter(CacheStateManager& state, const CacheOperatorRegistry& operators,
                      const void* condition_key)
        : state_(state), operators_(operators), condition_key_(condition_key) {}

    void bind_ambient(int id, const sd::Tensor<float>* t) { ambient_[id] = t; }
    const sd::Tensor<float>* ambient(int id) const {
        auto it = ambient_.find(id);
        return it == ambient_.end() ? nullptr : it->second;
    }

    // Per-step operator coefficients supplied by the runtime decision (used by
    // actions with coeffs_from_decision = true, e.g. TaylorSeer's extrapolation
    // weights computed from the step indices this step).
    void set_step_coeffs(const std::vector<float>* coeffs) { step_coeffs_ = coeffs; }

    bool run(const std::vector<CacheAction>& actions) {
        for (const auto& a : actions) {
            if (!run_one(a)) {
                return false;
            }
        }
        return true;
    }

private:
    const sd::Tensor<float>* resolve(const ValueRef& ref) const {
        switch (ref.kind) {
            case ValueRef::Kind::Ambient: return ambient(ref.id);
            case ValueRef::Kind::Temp: {
                auto it = temps_.find(ref.id);
                return it == temps_.end() ? nullptr : &it->second;
            }
            case ValueRef::Kind::Slot: {
                CacheSlotHandle h = state_.read_history(condition_key_, ref.slot, ref.depth);
                return h.valid ? h.host : nullptr;
            }
            case ValueRef::Kind::Site:
            default:
                return nullptr;
        }
    }

    // Store a produced tensor into the action's first output (temp or slot).
    bool commit_output(const ValueRef& out, sd::Tensor<float>&& value) {
        if (out.kind == ValueRef::Kind::Temp) {
            temps_[out.id] = std::move(value);
            return true;
        }
        if (out.kind == ValueRef::Kind::Ambient) {
            owned_ambient_[out.id] = std::move(value);
            ambient_[out.id] = &owned_ambient_[out.id];
            return true;
        }
        if (out.kind == ValueRef::Kind::Slot) {
            CacheSlotHandle h = state_.write(condition_key_, out.slot);
            if (!h.valid || h.host == nullptr) {
                return false;
            }
            *h.host = std::move(value);
            return true;
        }
        return false;
    }

    bool run_one(const CacheAction& a) {
        switch (a.kind) {
            case CacheActionKind::LOAD: {
                // slot -> output temp (identity copy through the state handle).
                if (a.inputs.empty() || a.outputs.empty()) return false;
                const sd::Tensor<float>* src = resolve(a.inputs[0]);
                if (src == nullptr || src->empty()) return false;
                return commit_output(a.outputs[0], sd::Tensor<float>(*src));
            }
            case CacheActionKind::STORE: {
                if (a.inputs.empty() || a.outputs.empty()) return false;
                const sd::Tensor<float>* src = resolve(a.inputs[0]);
                if (src == nullptr || src->empty()) return false;
                return commit_output(a.outputs[0], sd::Tensor<float>(*src));
            }
            case CacheActionKind::DIFFERENCE:
            case CacheActionKind::PREDICT:
            case CacheActionKind::BLEND: {
                const ICacheOperator* op = operators_.find(a.op);
                if (op == nullptr || a.outputs.empty()) return false;
                std::vector<const sd::Tensor<float>*> ins;
                ins.reserve(a.inputs.size());
                for (const auto& in : a.inputs) {
                    const sd::Tensor<float>* t = resolve(in);
                    if (t == nullptr) return false;
                    ins.push_back(t);
                }
                std::vector<sd::Tensor<float>> outs;
                // Substitute per-step coefficients from the runtime decision when
                // the action requests it (e.g. TaylorSeer weights). Otherwise use
                // the program's compile-time params verbatim.
                if (a.coeffs_from_decision && step_coeffs_ != nullptr) {
                    CacheOperatorParams p = a.params;
                    p.floats = *step_coeffs_;
                    if (!op->apply_host(ins, p, &outs) || outs.empty()) {
                        return false;
                    }
                } else if (!op->apply_host(ins, a.params, &outs) || outs.empty()) {
                    return false;
                }
                return commit_output(a.outputs[0], std::move(outs[0]));
            }
            case CacheActionKind::ROTATE_HISTORY: {
                // Advance a history ring so the entry just written stays readable
                // at depth 1 next step and the next STORE targets a fresh slot.
                // No-op for depth-1 slots (manager guards ring.size() <= 1).
                if (a.slot.has_value()) {
                    state_.rotate_history(condition_key_, *a.slot);
                }
                return true;
            }
            case CacheActionKind::COMPUTE:
            default:
                return true;  // handled outside the operator interpreter
        }
    }

    CacheStateManager& state_;
    const CacheOperatorRegistry& operators_;
    const void* condition_key_;
    const std::vector<float>* step_coeffs_ = nullptr;
    std::unordered_map<int, const sd::Tensor<float>*> ambient_;
    std::unordered_map<int, sd::Tensor<float>> owned_ambient_;
    std::unordered_map<int, sd::Tensor<float>> temps_;
};

// Declarative execution for an Output-granularity variant that carries actions.
// REUSE: run the REUSE variant's before-actions; they must produce
// kAmbientModelOutput (e.g. LOAD cached diff -> BLEND with input). On a full
// compute (decided-FULL, or a REUSE that could not be served), hand the policy
// the (input, output) pair for its scalar metrics, then run the FULL variant's
// after-actions to store the residual into the slot.
sd::Tensor<float> execute_declarative_output(ICachePolicy& policy,
                                             const CacheProgram& program,
                                             const GraphVariantPlan& decided,
                                             const StepContext& step,
                                             const void* condition_key,
                                             CacheBranch branch,
                                             const CacheRunnerHooks& hooks,
                                             CacheStateManager& state,
                                             const CacheOperatorRegistry& operators) {
    ActionInterpreter interp(state, operators, condition_key);
    interp.bind_ambient(kAmbientInput, hooks.input);

    if (decided.kind == GraphVariantKind::REUSE || decided.kind == GraphVariantKind::PREDICT) {
        bool ok = true;
        for (const auto& seg : decided.segments) {
            if (!interp.run(seg.before)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            const sd::Tensor<float>* out = interp.ambient(kAmbientModelOutput);
            if (out != nullptr && !out->empty()) {
                return *out;
            }
        }
        // Reuse could not be served (no residual yet) -> fall through to full.
    }

    sd::Tensor<float> out = hooks.full();
    if (!out.empty() && hooks.input != nullptr) {
        CacheObservation obs;
        obs.kind = CacheObservation::Kind::Feature;
        obs.step = step;
        obs.condition_key = condition_key;
        obs.branch = branch;
        obs.input = hooks.input;
        obs.feature = &out;
        policy.observe(obs);

        // Store via the FULL variant's after-actions (not the decided variant's,
        // which is empty on a fallen-through REUSE).
        interp.bind_ambient(kAmbientModelOutput, &out);
        const GraphVariantPlan* full = program.find_kind(GraphVariantKind::FULL);
        if (full != nullptr) {
            for (const auto& seg : full->segments) {
                interp.run(seg.after);
            }
        }
    }
    return out;
}

// Forward declaration (defined below): the (start,end) block region a variant's
// block-stack segment covers, for capture/inject.
void seam_region(const GraphVariantPlan& v, int* start, int* end, int* probe_depth);

// Whether the Feature seam can be driven declaratively THIS step: a Feature
// method that emitted actions, running on the host readback path (no on-GPU
// inject). The GPU inject paths (inject_feature_gpu / inject_gpu) and Probe
// methods keep the legacy seam control flow until their own slices land.
bool feature_declarative_host_path(const CacheProgram& program,
                                   const GraphVariantPlan& variant,
                                   const CacheRunnerHooks& hooks) {
    if (!variant_has_actions(program)) {
        return false;
    }
    if (variant.kind == GraphVariantKind::PROBE) {
        return false;
    }
    if (hooks.inject_gpu || hooks.inject_feature_gpu) {
        return false;  // on-GPU reuse -> handled by a later slice
    }
    return true;
}

// Declarative Feature-granularity execution on the host readback path. The seam
// compute calls (capture / inject) remain the model-compute boundary; only the
// residual STORAGE moves into the CacheStateManager slot via LOAD/STORE actions.
//   REUSE.before : LOAD slot -> kAmbientInjectFeature ; lowering calls inject().
//   FULL.after   : STORE kAmbientCapturedFeature -> slot after capture().
sd::Tensor<float> execute_declarative_feature(ICachePolicy& policy,
                                              const CacheProgram& program,
                                              const GraphVariantPlan& decided,
                                              const RuntimeDecision& decision,
                                              const StepContext& step,
                                              const void* condition_key,
                                              CacheBranch branch,
                                              const CacheRunnerHooks& hooks,
                                              CacheStateManager& state,
                                              const CacheOperatorRegistry& operators) {
    int region_start = 0, region_end = -1, probe_depth = 0;
    seam_region(decided, &region_start, &region_end, &probe_depth);
    ActionInterpreter interp(state, operators, condition_key);
    interp.bind_ambient(kAmbientInput, hooks.input);
    interp.set_step_coeffs(&decision.reuse_coeffs);

    if (decided.kind == GraphVariantKind::REUSE || decided.kind == GraphVariantKind::PREDICT) {
        bool ok = true;
        for (const auto& seg : decided.segments) {
            if (!interp.run(seg.before)) { ok = false; break; }
        }
        const sd::Tensor<float>* feat = ok ? interp.ambient(kAmbientInjectFeature) : nullptr;
        if (feat != nullptr && !feat->empty() && hooks.inject) {
            sd::Tensor<float> out = hooks.inject(*feat, region_start, region_end);
            if (!out.empty()) {
                return out;
            }
        }
        // History not ready / inject failed -> capturing full step below.
    }

    sd::DiffusionCacheResult res = hooks.capture(region_start, region_end);
    if (res.output.empty()) {
        return hooks.full();
    }
    if (!res.feature.empty()) {
        // Let the policy update its scalar decision state (has_residual flag,
        // calibration), then STORE the residual into the slot declaratively.
        CacheObservation obs;
        obs.kind = CacheObservation::Kind::Feature;
        obs.step = step;
        obs.condition_key = condition_key;
        obs.branch = branch;
        obs.input = hooks.input;
        obs.feature = &res.feature;
        policy.observe(obs);

        const GraphVariantPlan* full = program.find_kind(GraphVariantKind::FULL);
        if (full != nullptr) {
            interp.bind_ambient(kAmbientCapturedFeature, &res.feature);
            for (const auto& seg : full->segments) {
                interp.run(seg.after);
            }
        }
    }
    return std::move(res.output);
}

// Whether this variant plan touches the block-stack seam (capture/inject/probe)
// or is a black-box Output-level variant that only uses full() + host diff.
bool variant_uses_seam(const GraphVariantPlan& v) {
    for (const auto& seg : v.segments) {
        if (seg.execution == SegmentExecutionMode::PROBE ||
            seg.execution == SegmentExecutionMode::LOAD_CACHED ||
            seg.execution == SegmentExecutionMode::PREDICT_FROM_HISTORY) {
            return true;
        }
    }
    return false;
}

// The (start,end) region the block-stack segment covers, for capture/inject.
void seam_region(const GraphVariantPlan& v, int* start, int* end, int* probe_depth) {
    *start = 0;
    *end = -1;
    *probe_depth = 0;
    for (const auto& seg : v.segments) {
        if (seg.execution == SegmentExecutionMode::PROBE) {
            *probe_depth = seg.probe_depth;
        }
        *start = seg.region_start;
        *end = seg.region_end;
    }
}

}  // namespace

sd::Tensor<float> CacheGraphLowering::execute(ICachePolicy& policy,
                                              const CacheProgram& program,
                                              const RuntimeDecision& decision,
                                              const StepContext& step,
                                              const void* condition_key,
                                              CacheBranch branch,
                                              const CacheRunnerHooks& hooks,
                                              CacheStateManager& state,
                                              const CacheOperatorRegistry& operators) {
    const GraphVariantPlan* variant = program.find_variant(decision.variant);
    if (variant == nullptr) {
        return hooks.full();
    }

    const bool is_output_method = policy.requirements().granularity == CacheGranularity::Output;

    // ---- Declarative Output path: a method that emits CacheActions is driven
    // by the state manager + operator registry instead of reconstruct()/observe
    // host callbacks. Guarded per-method by variant_has_actions so methods
    // migrate one at a time; methods with no actions keep the legacy path below
    // byte-for-byte. ----
    if (is_output_method && variant_has_actions(program)) {
        return execute_declarative_output(policy, program, *variant, step,
                                          condition_key, branch, hooks, state, operators);
    }

    // ---- Output granularity: black-box full / cached-diff reuse. ----
    if (is_output_method) {
        if (variant->kind == GraphVariantKind::REUSE) {
            CacheReconstructContext rc;
            rc.step = step;
            rc.condition_key = condition_key;
            rc.input = hooks.input;
            sd::Tensor<float> out = policy.reconstruct(rc);
            if (!out.empty()) {
                return out;
            }
            // Reconstruction unavailable -> full compute + observe below.
        }
        sd::Tensor<float> out = hooks.full();
        if (!out.empty() && hooks.input != nullptr) {
            CacheObservation obs;
            obs.kind = CacheObservation::Kind::Feature;  // Output reuses the Feature slot
            obs.step = step;
            obs.condition_key = condition_key;
            obs.branch = branch;
            obs.input = hooks.input;
            obs.feature = &out;  // the whole-model output
            policy.observe(obs);
        }
        return out;
    }

    // ---- Feature / Probe granularity: use the block-stack seam. ----
    if (!hooks.feature_supported || !hooks.capture || !hooks.inject) {
        return hooks.full();  // model can't cut its stack this step
    }

    // Declarative Feature host path: residual storage lives in the state-manager
    // slot, driven by LOAD/STORE actions. Gated to the host readback path; GPU
    // inject and Probe methods keep the legacy seam control flow below.
    if (feature_declarative_host_path(program, *variant, hooks)) {
        return execute_declarative_feature(policy, program, *variant, decision, step,
                                           condition_key, branch, hooks, state, operators);
    }

    int region_start = 0;
    int region_end = -1;
    int probe_depth = 0;
    seam_region(*variant, &region_start, &region_end, &probe_depth);

    // Probe variant: run the shallow prefix, let the policy decide, then either
    // inject a reconstructed residual or fall through to a capturing full step.
    RuntimeDecision effective = decision;
    sd::DiffusionCacheResult probe_res;
    if (variant->kind == GraphVariantKind::PROBE && hooks.probe && probe_depth > 0) {
        probe_res = hooks.probe(probe_depth);
        CacheObservation probe_obs;
        probe_obs.kind = CacheObservation::Kind::Probe;
        probe_obs.step = step;
        probe_obs.condition_key = condition_key;
        probe_obs.branch = branch;
        probe_obs.before = &probe_res.before;
        probe_obs.probe = &probe_res.probe;
        probe_obs.delta_y = probe_res.delta_y;  // GPU path: scalars (NaN on host path)
        probe_obs.delta_x = probe_res.delta_x;
        probe_obs.gamma = probe_res.gamma;
        effective = policy.decide_after_probe(step, probe_obs);
    }

    const GraphVariantPlan* eff_variant = program.find_variant(effective.variant);
    const GraphVariantKind eff_kind = eff_variant ? eff_variant->kind : GraphVariantKind::FULL;

    if (eff_kind == GraphVariantKind::REUSE || eff_kind == GraphVariantKind::PREDICT) {
        int s = region_start;
        int e = region_end;
        int pd = 0;
        if (eff_variant) {
            seam_region(*eff_variant, &s, &e, &pd);
        }
        // GPU DiCache reuse: reconstruct on-device from the persistent residual
        // ring using the probe pass's gamma (clamped here). No host residual.
        if (hooks.inject_gpu && !std::isnan(probe_res.gamma)) {
            const float gamma = std::max(1.0f, std::min(1.5f, probe_res.gamma));
            sd::Tensor<float> out = hooks.inject_gpu(gamma, s, e);
            if (!out.empty()) {
                return out;
            }
            // GPU inject not ready (insufficient history) -> capturing full step.
        } else if (hooks.inject_feature_gpu) {
            // Feature-granularity on-GPU reuse (MagCache/TaylorSeer): inject the
            // last captured residual straight from device memory — no host
            // reconstruct copy, no H2D upload.
            sd::Tensor<float> out = hooks.inject_feature_gpu(s, e);
            if (!out.empty()) {
                return out;
            }
            // Device residual not ready yet -> capturing full step below.
        } else {
            CacheReconstructContext rc;
            rc.step = step;
            rc.condition_key = condition_key;
            rc.input = hooks.input;
            sd::Tensor<float> feature = policy.reconstruct(rc);
            if (!feature.empty()) {
                sd::Tensor<float> out = hooks.inject(feature, s, e);
                if (!out.empty()) {
                    return out;
                }
            }
            // Reconstruction/inject failed -> capturing full step.
        }
    }

    // Full compute with capture (also seeds probe/residual history).
    sd::DiffusionCacheResult res = hooks.capture(region_start, region_end);
    if (res.output.empty()) {
        return hooks.full();  // capture failed; last-resort plain compute
    }
    if (!res.feature.empty()) {
        CacheObservation obs;
        obs.kind = CacheObservation::Kind::Feature;
        obs.step = step;
        obs.condition_key = condition_key;
        obs.branch = branch;
        obs.input = hooks.input;
        obs.feature = &res.feature;
        policy.observe(obs);
    } else if (hooks.inject_feature_gpu) {
        // GPU feature reuse: the residual was captured to device memory (no host
        // readback). Signal the policy that a residual is available so its skip
        // decision can fire; the actual data lives on-device for inject_feature_gpu.
        CacheObservation obs;
        obs.kind = CacheObservation::Kind::Feature;
        obs.step = step;
        obs.condition_key = condition_key;
        obs.branch = branch;
        obs.input = hooks.input;
        obs.feature_on_device = true;
        policy.observe(obs);
    }
    return std::move(res.output);
}

}  // namespace cache
}  // namespace edgedit
