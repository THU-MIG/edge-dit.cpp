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
// method that emitted actions, running on either the host readback path OR the
// device-slot path (capture_to_slot/inject_from_slot). The legacy GPU inject path
// (inject_gpu, DiCache probe reuse) and Probe methods keep the legacy seam control
// flow until their own slices land.
bool feature_declarative_host_path(const CacheProgram& program,
                                   const GraphVariantPlan& variant,
                                   const CacheRunnerHooks& hooks) {
    if (!variant_has_actions(program)) {
        return false;
    }
    if (variant.kind == GraphVariantKind::PROBE) {
        return false;
    }
    // Device-slot hooks drive the declarative path (they replace the legacy
    // per-runner GPU feature reuse) — take it regardless of the legacy GPU flags.
    if (hooks.capture_to_slot && hooks.inject_from_slot) {
        return true;
    }
    if (hooks.inject_gpu) {
        return false;  // on-GPU DiCache probe reuse -> legacy seam path
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

    // ---- Device-slot path (B2): the Feature slot is backed by a persistent device
    // tensor, so store/reuse happen on-device via capture_to_slot/inject_from_slot
    // (no host residual round-trip). Single-residual reuse (MagCache): slot depth 1,
    // reuse injects x_before + slot. Gated on a wired device store + both hooks. ----
    const int slot0 = program.slots.empty() ? 0 : program.slots.front().id;
    const bool device_slot = state.has_device_store() && !program.slots.empty() &&
                             program.slots.front().device_backed &&
                             hooks.capture_to_slot && hooks.inject_from_slot;
    if (device_slot) {
        if (decided.kind == GraphVariantKind::REUSE || decided.kind == GraphVariantKind::PREDICT) {
            CacheSlotHandle h = state.read(condition_key, slot0);
            if (h.valid && h.buffer != nullptr) {
                sd::Tensor<float> out = hooks.inject_from_slot(h.buffer, region_start, region_end);
                if (!out.empty()) {
                    return out;
                }
            }
            // Slot not ready / inject failed -> capturing full step below.
        }
        // Full compute: the runner captures the residual, then calls alloc_slot with
        // the true residual shape (packed block-stack seq shape, unknown here) to get
        // the device slot tensor from the StateManager, and copies into it on-device.
        auto alloc_slot = [&](const std::vector<int64_t>& residual_shape) -> void* {
            return state.alloc_device_entry(condition_key, slot0, residual_shape);
        };
        sd::Tensor<float> out = hooks.capture_to_slot(alloc_slot, region_start, region_end);
        if (!out.empty()) {
            // Mark the residual available so decide() will consider reuse next step;
            // the data lives on-device (no host feature to observe).
            CacheObservation obs;
            obs.kind = CacheObservation::Kind::Feature;
            obs.step = step;
            obs.condition_key = condition_key;
            obs.branch = branch;
            obs.input = hooks.input;
            obs.feature_on_device = true;
            policy.observe(obs);
            return out;
        }
        // Device capture failed -> fall through to the host path as a safety net.
    }

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

    // Device-slot runs have no host capture/inject wired; if we reach here (device
    // capture failed above, or a REUSE couldn't be served) fall back to full.
    if (!hooks.capture) {
        return hooks.full();
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

// Whether the Probe seam can be driven declaratively THIS step: a Probe method
// that emitted actions, running on the host readback path (no on-GPU inject). The
// pipeline only wires inject_gpu when the model's GPU probe path is active, so a
// null inject_gpu means the host readback path is in use.
bool probe_declarative_host_path(const CacheProgram& program,
                                 const CacheRunnerHooks& hooks) {
    if (!variant_has_actions(program) || !hooks.probe) {
        return false;
    }
    if (hooks.inject_gpu) {
        return false;  // on-GPU reuse -> legacy seam path (needs GPU operator lowering)
    }
    return true;
}

// Declarative Probe-granularity execution (DiCache) on the host readback path.
// Runs the shallow probe, hands the observation to decide_after_probe (which
// computes the gamma-aligned blend weights), then either injects the declarative
// blend of the residual ring or falls through to a capturing full step. The
// probe/capture/inject seam calls remain the model-compute boundary.
sd::Tensor<float> execute_declarative_probe(ICachePolicy& policy,
                                            const CacheProgram& program,
                                            const GraphVariantPlan& probe_variant,
                                            const RuntimeDecision& decision,
                                            const StepContext& step,
                                            const void* condition_key,
                                            CacheBranch branch,
                                            const CacheRunnerHooks& hooks,
                                            CacheStateManager& state,
                                            const CacheOperatorRegistry& operators) {
    int region_start = 0, region_end = -1, probe_depth = 0;
    seam_region(probe_variant, &region_start, &region_end, &probe_depth);

    RuntimeDecision effective = decision;
    if (probe_variant.kind == GraphVariantKind::PROBE && probe_depth > 0) {
        sd::DiffusionCacheResult probe_res = hooks.probe(probe_depth);
        CacheObservation probe_obs;
        probe_obs.kind = CacheObservation::Kind::Probe;
        probe_obs.step = step;
        probe_obs.condition_key = condition_key;
        probe_obs.branch = branch;
        probe_obs.before = &probe_res.before;
        probe_obs.probe = &probe_res.probe;
        probe_obs.delta_y = probe_res.delta_y;  // NaN on host path
        probe_obs.delta_x = probe_res.delta_x;
        probe_obs.gamma = probe_res.gamma;
        effective = policy.decide_after_probe(step, probe_obs);
    }

    const GraphVariantPlan* eff = program.find_variant(effective.variant);
    if (eff != nullptr &&
        (eff->kind == GraphVariantKind::REUSE || eff->kind == GraphVariantKind::PREDICT)) {
        int s = region_start, e = region_end, pd = 0;
        seam_region(*eff, &s, &e, &pd);
        ActionInterpreter interp(state, operators, condition_key);
        interp.bind_ambient(kAmbientInput, hooks.input);
        interp.set_step_coeffs(&effective.reuse_coeffs);
        bool ok = true;
        for (const auto& seg : eff->segments) {
            if (!interp.run(seg.before)) { ok = false; break; }
        }
        const sd::Tensor<float>* feat = ok ? interp.ambient(kAmbientInjectFeature) : nullptr;
        if (feat != nullptr && !feat->empty() && hooks.inject) {
            sd::Tensor<float> out = hooks.inject(*feat, s, e);
            if (!out.empty()) {
                return out;
            }
        }
        // Ring not ready / inject failed -> capturing full step below.
    }

    sd::DiffusionCacheResult res = hooks.capture(region_start, region_end);
    if (res.output.empty()) {
        return hooks.full();
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

        const GraphVariantPlan* full = program.find_kind(GraphVariantKind::FULL);
        if (full != nullptr) {
            ActionInterpreter interp(state, operators, condition_key);
            interp.bind_ambient(kAmbientCapturedFeature, &res.feature);
            for (const auto& seg : full->segments) {
                interp.run(seg.after);
            }
        }
    }
    return std::move(res.output);
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

sd::Tensor<float> CacheGraphLowering::execute_substeps(ICachePolicy& policy,
                                                       const CacheProgram& program,
                                                       const StepContext& step,
                                                       const void* condition_key,
                                                       CacheBranch branch,
                                                       const CacheRunnerHooks& hooks,
                                                       CacheStateManager& state,
                                                       const CacheOperatorRegistry& operators) {
    (void)operators;
    policy.begin_substeps(step, condition_key);

    const int slot0 = program.slots.empty() ? 0 : program.slots.front().id;
    const bool device_slot = state.has_device_store() && !program.slots.empty() &&
                             program.slots.front().device_backed &&
                             hooks.capture_to_slot && hooks.inject_from_slot;

    sd::Tensor<float> out;
    while (auto plan_opt = policy.next_substep()) {
        const SubstepPlan& plan = *plan_opt;
        SubstepResult result;
        sd::Tensor<float> y;

        const int region_end = plan.blocks.end;  // -1 => to end of stack

        // ---- Probe substep (DiCache): run the shallow prefix, surface the seam's
        // on-device decision scalars (delta_y/delta_x/gamma) as indicator results.
        // Produces no output; observe_substep folds the scalars into the skip
        // decision that the following continuation substep reads. ----
        if (plan.op.kind == SubstepOpKind::Stash) {
            const int depth = plan.blocks.end > plan.blocks.begin
                                  ? plan.blocks.end
                                  : std::max(1, plan.blocks.begin);
            if (hooks.substep_probe) {
                // Tap-driven probe (ED_CACHE_SUBSTEP): delta_y/gamma computed
                // on-device from taps + persistent operands, no CacheGraphScope.
                sd::DiffusionCacheResult pr = hooks.substep_probe(depth);
                result.indicators["delta_y"] = pr.delta_y;
                result.indicators["delta_x"] = pr.delta_x;
                result.indicators["gamma"] = pr.gamma;
                policy.observe_substep(result);
            } else if (hooks.substep_probe_host) {
                // Tap-driven host probe (Wan, no device metric): the before/probe
                // tensors come back to host; the policy computes delta_y/gamma on
                // host (reusing decide_after_probe). No CacheGraphScope.
                sd::DiffusionCacheResult pr = hooks.substep_probe_host(depth);
                policy.observe_substep_probe_host(&pr.before, &pr.probe, step, condition_key);
            } else if (hooks.probe) {
                sd::DiffusionCacheResult pr = hooks.probe(depth);
                result.indicators["delta_y"] = pr.delta_y;
                result.indicators["delta_x"] = pr.delta_x;
                result.indicators["gamma"] = pr.gamma;
                policy.observe_substep(result);
            }
            continue;  // probe never produces the step output
        }

        // ---- Extrapolate reuse (DiCache): reconstruct on-device from the residual
        // ring using the clamped gamma the probe produced. Zero compute. ----
        if (plan.op.kind == SubstepOpKind::Extrapolate) {
            if (!plan.op.coeffs.empty()) {
                const float gamma = plan.op.coeffs.front();
                // Prefer the tap-driven device inject (no CacheGraphScope); fall back
                // to the legacy inject_gpu only if the tap path isn't wired.
                if (hooks.substep_inject_gpu) {
                    y = hooks.substep_inject_gpu(gamma, 0, -1);
                } else if (hooks.inject_gpu) {
                    y = hooks.inject_gpu(gamma, 0, -1);
                }
            }
            if (y.empty()) {
                // Ring not ready -> capturing full compute (seeds the ring).
                y = hooks.full();
            }
            policy.observe_substep(result);
            if (plan.produces_output) {
                out = std::move(y);
            }
            continue;
        }

        const bool is_reuse = plan.op.kind == SubstepOpKind::ApplyResidual;
        // Seam region semantics: SubstepPlan.blocks is the interval to COMPUTE. The
        // inject/capture region is the block interval the residual spans. For the
        // whole-stack MagCache slice that is [0, end-of-stack) = [0, -1); a reuse
        // plan computes zero blocks but still injects over the whole stack.
        if (is_reuse && device_slot) {
            // Serve the whole-stack residual from the persistent device slot:
            // out = x_before + slot. The inject region is the whole stack [0, -1),
            // NOT the (empty) compute range. Slot-not-ready falls through to a
            // capture below.
            CacheSlotHandle h = state.read(condition_key, slot0);
            if (h.valid && h.buffer != nullptr) {
                // Prefer the tap-driven device inject (no CacheGraphScope); fall back
                // to the legacy inject_from_slot only if the tap path isn't wired.
                y = hooks.substep_inject_slot ? hooks.substep_inject_slot(h.buffer, 0, -1)
                                              : hooks.inject_from_slot(h.buffer, 0, -1);
            }
        } else if (is_reuse && (hooks.substep_inject_host || hooks.inject)) {
            // Host reuse (no device slot — Wan): the policy reconstructs the residual
            // on host; inject it as x_before + feature over the whole stack. Empty
            // reconstruct (history not ready) falls through to a capture below.
            CacheReconstructContext rc;
            rc.step = step;
            rc.condition_key = condition_key;
            rc.input = hooks.input;
            sd::Tensor<float> feature = policy.reconstruct(rc);
            if (!feature.empty()) {
                // Prefer the tap-driven inject (no CacheGraphScope); fall back to the
                // legacy compute_inject only if the tap path isn't wired.
                y = hooks.substep_inject_host ? hooks.substep_inject_host(feature)
                                              : hooks.inject(feature, 0, -1);
            }
        }

        if (y.empty()) {
            // Compute substep: full block-stack forward, capturing the residual
            // into the device slot when this plan writes it.
            const bool wants_capture = !plan.writes.empty();
            if (device_slot && wants_capture && hooks.substep_capture) {
                // Tap-driven capture (ED_CACHE_SUBSTEP): the runner requests
                // ModelIn/ModelOut taps and weaves the residual — no CacheGraphScope.
                auto alloc_slot = [&](const std::vector<int64_t>& residual_shape) -> void* {
                    return state.alloc_device_entry(condition_key, slot0, residual_shape);
                };
                y = hooks.substep_capture(alloc_slot);
                if (!y.empty()) {
                    CacheObservation obs;
                    obs.kind = CacheObservation::Kind::Feature;
                    obs.step = step;
                    obs.condition_key = condition_key;
                    obs.branch = branch;
                    obs.input = hooks.input;
                    obs.feature_on_device = true;
                    policy.observe(obs);
                }
            } else if (device_slot && wants_capture && hooks.capture_to_slot) {
                auto alloc_slot = [&](const std::vector<int64_t>& residual_shape) -> void* {
                    return state.alloc_device_entry(condition_key, slot0, residual_shape);
                };
                y = hooks.capture_to_slot(alloc_slot, 0, region_end);
                if (!y.empty()) {
                    CacheObservation obs;
                    obs.kind = CacheObservation::Kind::Feature;
                    obs.step = step;
                    obs.condition_key = condition_key;
                    obs.branch = branch;
                    obs.input = hooks.input;
                    obs.feature_on_device = true;
                    policy.observe(obs);
                }
            } else if (wants_capture && hooks.substep_capture_host) {
                // Tap-driven host capture (ED_CACHE_SUBSTEP, no device slot — Wan):
                // the runner weaves (ModelOut - ModelIn) and reads it back to host.
                // No CacheGraphScope. The policy stores the host residual for reuse.
                sd::DiffusionCacheResult res = hooks.substep_capture_host();
                y = std::move(res.output);
                if (!y.empty() && !res.feature.empty()) {
                    CacheObservation obs;
                    obs.kind = CacheObservation::Kind::Feature;
                    obs.step = step;
                    obs.condition_key = condition_key;
                    obs.branch = branch;
                    obs.input = hooks.input;
                    obs.feature = &res.feature;
                    policy.observe(obs);
                }
            } else if (wants_capture && hooks.capture) {
                // Host/GPU-ring capture (DiCache): compute_capture runs the full
                // forward AND seeds the cross-step residual/probe ring via its
                // handoff. Must be used instead of full() so the ring updates.
                sd::DiffusionCacheResult res = hooks.capture(0, region_end);
                y = std::move(res.output);
                if (!y.empty() && !res.feature.empty()) {
                    CacheObservation obs;
                    obs.kind = CacheObservation::Kind::Feature;
                    obs.step = step;
                    obs.condition_key = condition_key;
                    obs.branch = branch;
                    obs.input = hooks.input;
                    obs.feature = &res.feature;
                    policy.observe(obs);
                }
            }
            if (y.empty()) {
                // No capture wired / capture failed / plain compute substep.
                y = hooks.full();
            }
        }

        policy.observe_substep(result);

        if (plan.produces_output) {
            out = std::move(y);
        }
    }

    if (out.empty()) {
        // Defensive: a policy that yielded no output-producing substep.
        out = hooks.full();
    }
    return out;
}

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
    // The device-slot seam (capture_to_slot/inject_from_slot) is an alternative to
    // the host capture/inject pair, so it satisfies the "can cut the stack" guard.
    const bool device_slot_seam = hooks.capture_to_slot && hooks.inject_from_slot;
    if (!hooks.feature_supported ||
        (!device_slot_seam && (!hooks.capture || !hooks.inject))) {
        return hooks.full();  // model can't cut its stack this step
    }

    // Declarative Feature host path: residual storage lives in the state-manager
    // slot, driven by LOAD/STORE actions. Gated to the host readback path; GPU
    // inject and Probe methods keep the legacy seam control flow below.
    if (feature_declarative_host_path(program, *variant, hooks)) {
        return execute_declarative_feature(policy, program, *variant, decision, step,
                                           condition_key, branch, hooks, state, operators);
    }

    // Declarative Probe host path (DiCache): shallow probe -> decide_after_probe
    // (computes gamma weights) -> blend the residual ring from the slot, or a
    // capturing full step. Gated to the host readback path; the GPU probe path
    // keeps the legacy seam control flow below.
    if (variant->kind == GraphVariantKind::PROBE &&
        probe_declarative_host_path(program, hooks)) {
        return execute_declarative_probe(policy, program, *variant, decision, step,
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
    }
    return std::move(res.output);
}

}  // namespace cache
}  // namespace edgedit
