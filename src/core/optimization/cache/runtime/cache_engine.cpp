#include "core/optimization/cache/runtime/cache_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

#include "core/optimization/cache/ir/cache_action.hpp"
#include "core/optimization/cache/runtime/capability_negotiation.hpp"
#include "utils/util.h"

namespace edgedit {
namespace cache {

namespace {

const char* granularity_name(CacheGranularity g) {
    switch (g) {
        case CacheGranularity::Output: return "output";
        case CacheGranularity::Feature: return "feature";
        case CacheGranularity::Probe: return "probe";
    }
    return "?";
}

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
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

// Drives the policy's substep loop against the runner hooks, using the policy for
// host-side reconstruction and observation. The seam between the declarative
// program and the tap-driven runner passes (formerly CacheGraphLowering).
sd::Tensor<float> run_substep_loop(ICachePolicy& policy,
                                   const CacheProgram& program,
                                   const StepContext& step,
                                   const void* condition_key,
                                   CacheBranch branch,
                                   const CacheRunnerHooks& hooks,
                                   CacheStateManager& state,
                                   const CacheOperatorRegistry& operators) {
    policy.set_substep_input(hooks.input);
    policy.begin_substeps(step, condition_key);

    // Bridge the DiCache multi-slot ring to the runner without exposing the state
    // manager type. rotate/alloc/read/filled forward to CacheStateManager for this
    // branch's condition_key. See DiCacheSlotBridge's depth-convention warning.
    DiCacheSlotBridge dicache_bridge;
    dicache_bridge.rotate = [&state, condition_key](int slot) {
        state.rotate_history(condition_key, slot);
    };
    dicache_bridge.alloc = [&state, condition_key](int slot, const std::vector<int64_t>& shape) -> void* {
        return state.alloc_device_entry(condition_key, slot, shape);
    };
    dicache_bridge.read = [&state, condition_key](int slot, int depth) -> void* {
        CacheSlotHandle h = state.read_history(condition_key, slot, depth);
        return h.valid ? h.buffer : nullptr;
    };
    dicache_bridge.filled = [&state, condition_key](int slot) -> int {
        // Derive fill count from read_history validity (no dedicated accessor):
        // probe onto increasing depths until one is invalid.
        int n = 0;
        while (state.read_history(condition_key, slot, n).valid) {
            ++n;
        }
        return n;
    };

    const int slot0 = program.slots.empty() ? 0 : program.slots.front().id;
    const bool device_slot = state.has_device_store() && !program.slots.empty() &&
                             program.slots.front().device_backed &&
                             hooks.substep_capture && hooks.substep_inject_slot;
    // History depth of slot 0 (1 for single-residual MagCache/SenCache, order+2 for
    // TaylorSeer's feature ring). Used by the device history-reuse path to know how
    // many ring entries to blend.
    const int slot0_depth = program.slots.empty()
                                ? 1
                                : std::max(1, program.slots.front().history_depth);

    // Build the device residual-capture extension used by every compute substep that
    // writes slot 0: a cache.difference (model_out - model_in) woven into a freshly
    // allocated device ring entry. Shared by the MagCache/SenCache ApplyResidual tail
    // and the TaylorSeer feature-ring compute sub-branch so the capture math lives in
    // one place.
    auto make_device_capture_ext = [&]() -> std::vector<GraphExtension> {
        std::vector<GraphExtension> exts;
        GraphExtension cap;
        cap.op = operators.find("cache.difference");
        cap.op_id = "cache.difference";
        // DifferenceOperator computes inputs[1] - inputs[0] => model_out - model_in,
        // byte-identical to the old ggml_sub(mout, min).
        cap.input_anchors = {AnchorRef::model_in(), AnchorRef::model_out()};
        cap.output_name = "ed_cache_feature";
        cap.sink = GraphExtension::Sink::CaptureToSlot;
        cap.alloc_slot = [&](const std::vector<int64_t>& residual_shape) -> void* {
            return state.alloc_device_entry(condition_key, slot0, residual_shape);
        };
        if (cap.op != nullptr) {
            exts.push_back(std::move(cap));
        }
        return exts;
    };

    sd::Tensor<float> out;
    while (auto plan_opt = policy.next_substep()) {
        const SubstepPlan& plan = *plan_opt;
        SubstepResult result;
        sd::Tensor<float> y;

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
            // ---- Device feature-ring sub-path (TaylorSeer on Flux/Qwen/SD3/Wan with
            // a store): the residual history lives in the device ring (slot 0). Reuse
            // blends the ring entries on-device with the policy's per-step coeffs;
            // compute captures (model_out - model_in) into a fresh ring head, then
            // rotates so this residual is readable at depth 1 next step. Mirrors the
            // host FULL STORE-then-ROTATE convention (newest = depth 1). ----
            if (device_slot) {
                if (plan.op.kind == SubstepOpKind::FeatureReuse) {
                    // Gather the ring residuals at depths 1..slot0_depth-1 (depth 0 is
                    // the fresh, not-yet-written head after the last rotate). coeffs[k]
                    // weights the entry at depth (k+1), matching the policy's
                    // extrapolation_coeffs order and the host PREDICT LOAD order.
                    // All-or-nothing like the host path: if the ring is not deep enough
                    // yet (any depth invalid), abandon the blend and full-compute below.
                    // The policy's order gate guarantees the ring is full before it ever
                    // returns a reuse decision, so this is a safety net, not a hot path.
                    std::vector<ggml_tensor*> ring_bufs;
                    ring_bufs.reserve(static_cast<size_t>(std::max(0, slot0_depth - 1)));
                    for (int k = 1; k < slot0_depth; ++k) {
                        CacheSlotHandle h = state.read_history(condition_key, slot0, k);
                        if (!h.valid || h.buffer == nullptr) {
                            ring_bufs.clear();
                            break;
                        }
                        ring_bufs.push_back(static_cast<ggml_tensor*>(h.buffer));
                    }
                    if (!ring_bufs.empty() &&
                        plan.op.coeffs.size() >= ring_bufs.size()) {
                        // out = 1.0*x_before + Σ_k coeff[k]*ring_bufs[k]. build_stream_override
                        // prepends x_before as inputs[0]; the ring residuals follow.
                        GraphExtension inj;
                        inj.op = operators.find("cache.weighted_blend");
                        inj.op_id = "cache.weighted_blend";
                        inj.extra_inputs = ring_bufs;
                        inj.params.floats.reserve(ring_bufs.size() + 1);
                        inj.params.floats.push_back(1.0f);  // x_before weight
                        for (size_t k = 0; k < ring_bufs.size(); ++k) {
                            inj.params.floats.push_back(plan.op.coeffs[k]);
                        }
                        inj.sink = GraphExtension::Sink::ReplaceStream;
                        std::vector<GraphExtension> exts;
                        if (inj.op != nullptr) {
                            exts.push_back(std::move(inj));
                        }
                        if (!exts.empty()) {
                            y = hooks.substep_inject_slot(std::move(exts));
                        }
                    }
                    // Ring not ready / inject failed -> fall through to capture below.
                }
                if (y.empty() && !plan.writes.empty()) {
                    // Compute substep: capture the residual into a fresh ring head,
                    // then rotate so it is readable at depth 1 next step.
                    y = hooks.substep_capture(make_device_capture_ext());
                    if (!y.empty()) {
                        CacheObservation obs;
                        obs.kind = CacheObservation::Kind::Feature;
                        obs.step = step;
                        obs.condition_key = condition_key;
                        obs.branch = branch;
                        obs.input = hooks.input;
                        obs.feature_on_device = true;
                        policy.observe(obs);
                        state.rotate_history(condition_key, slot0);
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
                if (feat != nullptr && !feat->empty() && hooks.substep_inject_host) {
                    // Tap-driven host inject (Wan/SD3, no CacheGraphScope).
                    y = hooks.substep_inject_host(*feat);
                }
                // Ring not ready / inject failed -> capture below.
            }
            if (y.empty() && hooks.substep_capture_host) {
                // Tap-driven host capture (Wan/SD3, no CacheGraphScope). Feature methods
                // are always whole-stack, so the no-arg host variant is exact.
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
                // Tap-driven probe: delta_y/delta_x/gamma woven by cache operators
                // (rel_l1 / gamma_indicator), resolved from the registry; the model
                // supplies the probe-history device operands.
                sd::DiffusionCacheResult pr = hooks.substep_probe(depth, operators, dicache_bridge);
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
            }
            continue;  // probe never produces the step output
        }

        // ---- Extrapolate reuse (DiCache): reconstruct on-device from the residual
        // ring using the clamped gamma the probe produced. Zero compute. ----
        if (plan.op.kind == SubstepOpKind::Extrapolate) {
            if (!plan.op.coeffs.empty() && hooks.substep_inject_gpu) {
                // Cache-driven gamma-blend: gamma is the host constant the probe
                // produced (already clamped by the policy), baked into params.floats.
                // The model fills resid1/resid2 from its ring; build_stream_override
                // weaves x_before + resid2 + gamma*(resid1-resid2) via op->lower().
                GraphExtension inj;
                inj.op = operators.find("cache.gamma_blend");
                inj.op_id = "cache.gamma_blend";
                inj.params.floats = {plan.op.coeffs.front()};
                inj.sink = GraphExtension::Sink::ReplaceStream;
                std::vector<GraphExtension> exts;
                if (inj.op != nullptr) {
                    exts.push_back(std::move(inj));
                }
                if (!exts.empty()) {
                    y = hooks.substep_inject_gpu(std::move(exts), dicache_bridge);
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

        // ---- DiCache seed compute: full forward that refreshes the cross-step
        // residual/probe rings. Device path (Qwen/Flux) writes them on-device via the
        // tap-driven substep_capture_probe; host path (Wan/SD3) reads the feature back
        // for the policy's host ring. Replaces the legacy hooks.capture path (which ran
        // run_cache_pass + CacheGraphScope). Produces the step output. ----
        if (plan.op.kind == SubstepOpKind::CaptureProbeSeed) {
            if (hooks.substep_capture_probe) {
                y = hooks.substep_capture_probe(dicache_bridge);
                if (!y.empty()) {
                    CacheObservation obs;
                    obs.kind = CacheObservation::Kind::Feature;
                    obs.step = step;
                    obs.condition_key = condition_key;
                    obs.branch = branch;
                    obs.input = hooks.input;
                    obs.feature_on_device = true;  // DiCache observe() is a no-op on the device path
                    policy.observe(obs);
                }
            } else if (hooks.substep_capture_host) {
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

        const bool is_reuse = plan.op.kind == SubstepOpKind::ApplyResidual;
        // Seam region semantics: SubstepPlan.blocks is the interval to COMPUTE. The
        // inject/capture region is the block interval the residual spans. For the
        // whole-stack MagCache slice that is [0, end-of-stack) = [0, -1); a reuse
        // plan computes zero blocks but still injects over the whole stack.
        if (is_reuse && device_slot) {
            // Serve the whole-stack residual from the persistent device slot as a
            // cache-driven weave: out = WeightedBlend([x_before, slot], {1,1}) =
            // x_before + slot. The engine hands the runner a ReplaceStream
            // extension; the model forward substitutes its output over the reuse
            // region [0,-1) (whole stack) and never learns the math is a residual
            // add. Slot-not-ready falls through to a capture below.
            CacheSlotHandle h = state.read(condition_key, slot0);
            if (h.valid && h.buffer != nullptr) {
                GraphExtension inj;
                inj.op = operators.find("cache.weighted_blend");
                inj.op_id = "cache.weighted_blend";
                // WeightedBlend: out = sum_k w[k]*inputs[k]. build_stream_override
                // prepends x_before as inputs[0]; slot is inputs[1]. weights {1,1}
                // => x_before + slot, byte-identical to the old build_tap_inject
                // DeviceResidual branch (ggml_add(x_before, slot)).
                inj.extra_inputs = {static_cast<ggml_tensor*>(h.buffer)};
                inj.params.floats = {1.0f, 1.0f};
                inj.sink = GraphExtension::Sink::ReplaceStream;
                std::vector<GraphExtension> exts;
                if (inj.op != nullptr) {
                    exts.push_back(std::move(inj));
                }
                if (!exts.empty()) {
                    y = hooks.substep_inject_slot(std::move(exts));
                }
            }
        } else if (is_reuse && hooks.substep_inject_host) {
            // Host reuse (no device slot — Wan/SD3): the policy reconstructs the
            // residual on host; inject it as x_before + feature over the whole stack.
            // Empty reconstruct (history not ready) falls through to a capture below.
            CacheReconstructContext rc;
            rc.step = step;
            rc.condition_key = condition_key;
            rc.input = hooks.input;
            sd::Tensor<float> feature = policy.reconstruct(rc);
            if (!feature.empty()) {
                // Tap-driven host inject (no CacheGraphScope).
                y = hooks.substep_inject_host(feature);
            }
        }

        if (y.empty()) {
            // Compute substep: full block-stack forward, capturing the residual
            // into the device slot when this plan writes it.
            const bool wants_capture = !plan.writes.empty();
            if (device_slot && wants_capture && hooks.substep_capture) {
                // Tap-driven capture, cache-driven weave: the cache layer hands the
                // runner a DIFFERENCE extension (model_out - model_in) and a slot to
                // d2d the woven residual into. The runner requests the taps the
                // extension references, weaves op->lower(), and pins the result — it
                // never learns the math is a residual. Replaces the runner's old
                // hardcoded ggml_sub + set_capture_residual seam.
                y = hooks.substep_capture(make_device_capture_ext());
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
                // Tap-driven host capture (no device slot — Wan/SD3):
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

}  // namespace

bool CacheEngine::init(const ed_sample_params_t& sample_params,
                       SDVersion version,
                       const std::vector<float>& sigmas,
                       bool seam_available,
                       ICacheDeviceStore* device_store,
                       bool cfg_parallel) {
    config_ = cache_config_from_sample_params(sample_params);
    version_ = version;
    num_steps_ = sigmas.size() >= 2 ? static_cast<int>(sigmas.size() - 1) : 0;

    policy_ = create_cache_policy(config_.mode);
    if (policy_ == nullptr) {
        return false;
    }

    contract_ = std::make_unique<DiTModelCacheContract>(version, seam_available);

    // Capability negotiation BEFORE compile: an unsupported method is reported
    // explicitly instead of silently degrading to full compute.
    const CacheRequirements reqs = policy_->requirements();
    const ValidationResult validation = validate_requirements(reqs, *contract_);
    if (!validation.ok) {
        if (config_.mode != CacheMode::Disabled) {
            LOG_WARN("cache disabled: %s", validation.message.c_str());
        }
        policy_.reset();
        contract_.reset();
        return false;
    }

    // Reject Output-granularity caching under CFG-parallel. Feature/Probe methods
    // are already refused above (seam_available=false), but Output methods need no
    // seam and would otherwise run per-rank. rank0 (uncond) and rank1 (cond) hold
    // independent policy state and decide skips from their own branch's metrics, so
    // they can diverge (one reuses while the other computes); the CFG combine then
    // mixes a stale-cached branch with a fresh one and silently drifts. Disable
    // explicitly rather than corrupt output — matches the warn-and-disable contract.
    if (cfg_parallel && reqs.granularity == CacheGranularity::Output &&
        config_.mode != CacheMode::Disabled) {
        LOG_WARN("cache disabled: Output-granularity caching (mode=%s) is not supported "
                 "under CFG-parallel (per-rank skip decisions could diverge across the "
                 "cond/uncond ranks and corrupt the CFG combine).",
                 cache_mode_name(config_.mode));
        policy_.reset();
        contract_.reset();
        return false;
    }

    InferenceConfig inf;
    inf.config = &config_;
    inf.sigmas = &sigmas;
    inf.num_steps = num_steps_;
    inf.separate_cfg = contract_->schema().family == ModelFamily::MMDiT ||
                       contract_->schema().family == ModelFamily::WanVideo;

    program_ = policy_->compile(contract_->schema(), contract_->topology(), inf);
    // Wire the device store BEFORE initialize() so device_backed slots allocate
    // on-device. Null store (CPU/SP/mmdit/wan) leaves every slot host-backed.
    state_.set_device_store(device_store);
    state_.initialize(program_.slots);
    register_builtin_cache_operators(&operators_);

    if (!policy_->enabled()) {
        // e.g. SenCache with no usable profile: compile() logged the reason.
        policy_.reset();
        contract_.reset();
        return false;
    }

    LOG_INFO("cache enabled: mode=%s model=%s granularity=%s matched_sites=%zu variants=%zu slots=%zu",
             cache_mode_name(config_.mode),
             cache_model_spec_for_version(version).model_name.c_str(),
             granularity_name(granularity()),
             reqs.required_sites.size(),
             program_.variants.size(),
             program_.slots.size());

    if (env_flag("ED_DUMP_CACHE_PROGRAM")) {
        const std::string dump = dump_cache_program(program_);
        LOG_INFO("cache program:\n%s", dump.c_str());
    }
    return true;
}

StepContext CacheEngine::make_step_context() const {
    StepContext s;
    s.step_index = current_step_.step_index;
    s.num_steps = current_step_.num_steps;
    s.sigma = current_step_.sigma;
    s.sigma_next = current_step_.sigma_next;
    s.is_first_step = current_step_.step_index == 0;
    s.is_last_step = current_step_.num_steps > 0 &&
                     current_step_.step_index == current_step_.num_steps - 1;
    return s;
}

void CacheEngine::begin_step(const CacheStepInfo& step) {
    current_step_ = step;
    state_.begin_step(step.step_index);
    if (policy_ != nullptr) {
        policy_->begin_step(make_step_context());
    }
}

void CacheEngine::end_step(const CacheStepInfo& step) {
    if (policy_ != nullptr) {
        policy_->end_step(make_step_context());
    }
    state_.commit_step(step.step_index);
}

void CacheEngine::log_summary(size_t total_steps) const {
    if (policy_ != nullptr) {
        policy_->log_summary(total_steps);
    }
}

void CacheEngine::calibrate(CacheBranch branch,
                            const void* condition_key,
                            const sd::Tensor<float>& latent,
                            const sd::Tensor<float>& prediction,
                            std::function<sd::Tensor<float>(const sd::Tensor<float>&, float)> forward_at) {
    if (policy_ == nullptr || !policy_->calibration_spec().active) {
        return;
    }
    CalibrationContext ctx;
    ctx.step = make_step_context();
    ctx.condition_key = condition_key;
    ctx.branch = branch;
    ctx.latent = &latent;
    ctx.prediction = &prediction;
    ctx.forward_at = std::move(forward_at);
    policy_->calibrate_step(ctx);
}

sd::Tensor<float> CacheEngine::run_branch(CacheBranch branch,
                                          const void* condition_key,
                                          const CacheRunnerHooks& hooks) {
    if (policy_ == nullptr || !policy_->enabled()) {
        return hooks.full();
    }
    const StepContext step = make_step_context();

    // The substep loop is the only path: every cache method implements next_substep()
    // (run_substep_loop translates each SubstepPlan into the runner hooks).
    return run_substep_loop(*policy_, program_, step,
                            condition_key, branch, hooks,
                            state_, operators_);
}

}  // namespace cache
}  // namespace edgedit
