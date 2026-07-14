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

// block-stack segment covers, for capture/inject.
void seam_region(const GraphVariantPlan& v, int* start, int* end, int* probe_depth);

// Whether the Feature seam can be driven declaratively THIS step: a Feature
// method that emitted actions, running on either the host readback path OR the
// device-slot path (capture_to_slot/inject_from_slot). The legacy GPU inject path
// (inject_gpu, DiCache probe reuse) and Probe methods keep the legacy seam control
// flow until their own slices land.
//   FULL.after   : STORE kAmbientCapturedFeature -> slot after capture().
// null inject_gpu means the host readback path is in use.
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
    policy.set_substep_input(hooks.input);
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

        // ---- Output-granularity substep (EasyCache/UCache/Condition): the reuse /
        // capture works on the whole denoiser output via the declarative operators
        // (LOAD/STORE/DIFFERENCE/BLEND over the state-manager slot), not the block
        // seam. Mirrors execute_declarative_output but driven by next_substep. ----
        if (plan.op.kind == SubstepOpKind::OutputReuse ||
            plan.op.kind == SubstepOpKind::OutputCompute) {
            ActionInterpreter interp(state, operators, condition_key);
            interp.bind_ambient(kAmbientInput, hooks.input);
            if (plan.op.kind == SubstepOpKind::OutputReuse) {
                const GraphVariantPlan* reuse = program.find_kind(GraphVariantKind::REUSE);
                bool ok = reuse != nullptr;
                for (const auto& seg : (reuse ? reuse->segments : std::vector<SegmentPlan>{})) {
                    if (!interp.run(seg.before)) { ok = false; break; }
                }
                if (ok) {
                    const sd::Tensor<float>* o = interp.ambient(kAmbientModelOutput);
                    if (o != nullptr && !o->empty()) {
                        y = *o;
                    }
                }
                // Reuse could not be served (no diff yet) -> fall through to compute.
            }
            if (y.empty()) {
                y = hooks.full();
                if (!y.empty() && hooks.input != nullptr) {
                    CacheObservation obs;
                    obs.kind = CacheObservation::Kind::Feature;
                    obs.step = step;
                    obs.condition_key = condition_key;
                    obs.branch = branch;
                    obs.input = hooks.input;
                    obs.feature = &y;
                    policy.observe(obs);
                    // Store the (output - input) diff via the FULL variant's after-actions.
                    interp.bind_ambient(kAmbientModelOutput, &y);
                    const GraphVariantPlan* full = program.find_kind(GraphVariantKind::FULL);
                    if (full != nullptr) {
                        for (const auto& seg : full->segments) {
                            interp.run(seg.after);
                        }
                    }
                }
            }
            policy.observe_substep(result);
            if (plan.produces_output) {
                out = std::move(y);
            }
            continue;
        }

        // ---- Feature-ring host substep (TaylorSeer/SenCache): reuse runs the
        // PREDICT/REUSE before-actions (LOAD ring + BLEND with per-step coeffs) into
        // the inject feature, then hooks.inject; compute runs hooks.capture, hands the
        // policy the residual, and STORE+ROTATEs the ring via the FULL after-actions.
        // Mirrors execute_declarative_feature's host path, driven by next_substep. ----
        if (plan.op.kind == SubstepOpKind::FeatureReuse ||
            plan.op.kind == SubstepOpKind::FeatureCompute) {
            int fr_start = 0, fr_end = -1, fr_probe = 0;
            const GraphVariantPlan* fr_variant = program.find_kind(GraphVariantKind::FULL);
            if (fr_variant != nullptr) {
                seam_region(*fr_variant, &fr_start, &fr_end, &fr_probe);
            }
            ActionInterpreter interp(state, operators, condition_key);
            interp.bind_ambient(kAmbientInput, hooks.input);
            interp.set_step_coeffs(&plan.op.coeffs);
            if (plan.op.kind == SubstepOpKind::FeatureReuse) {
                const GraphVariantPlan* pred = program.find_kind(GraphVariantKind::PREDICT);
                if (pred == nullptr) {
                    pred = program.find_kind(GraphVariantKind::REUSE);
                }
                bool ok = pred != nullptr;
                for (const auto& seg : (pred ? pred->segments : std::vector<SegmentPlan>{})) {
                    if (!interp.run(seg.before)) { ok = false; break; }
                }
                const sd::Tensor<float>* feat = ok ? interp.ambient(kAmbientInjectFeature) : nullptr;
                if (feat != nullptr && !feat->empty() && hooks.inject) {
                    y = hooks.inject(*feat, fr_start, fr_end);
                }
                // Ring not ready / inject failed -> capture below.
            }
            if (y.empty() && hooks.capture) {
                sd::DiffusionCacheResult res = hooks.capture(fr_start, fr_end);
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
                    interp.bind_ambient(kAmbientCapturedFeature, &res.feature);
                    const GraphVariantPlan* full = program.find_kind(GraphVariantKind::FULL);
                    if (full != nullptr) {
                        for (const auto& seg : full->segments) {
                            interp.run(seg.after);
                        }
                    }
                }
            }
            if (y.empty()) {
                y = hooks.full();
            }
            policy.observe_substep(result);
            if (plan.produces_output) {
                out = std::move(y);
            }
            continue;
        }

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

}  // namespace cache
}  // namespace edgedit
