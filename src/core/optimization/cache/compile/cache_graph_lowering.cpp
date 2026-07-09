#include "core/optimization/cache/compile/cache_graph_lowering.hpp"

namespace edgedit {
namespace cache {

namespace {

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
                                              const CacheRunnerHooks& hooks) {
    const GraphVariantPlan* variant = program.find_variant(decision.variant);
    if (variant == nullptr) {
        return hooks.full();
    }

    const bool is_output_method = policy.requirements().granularity == CacheGranularity::Output;

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

    int region_start = 0;
    int region_end = -1;
    int probe_depth = 0;
    seam_region(*variant, &region_start, &region_end, &probe_depth);

    // Probe variant: run the shallow prefix, let the policy decide, then either
    // inject a reconstructed residual or fall through to a capturing full step.
    RuntimeDecision effective = decision;
    if (variant->kind == GraphVariantKind::PROBE && hooks.probe && probe_depth > 0) {
        sd::DiffusionCacheResult probe_res = hooks.probe(probe_depth);
        CacheObservation probe_obs;
        probe_obs.kind = CacheObservation::Kind::Probe;
        probe_obs.step = step;
        probe_obs.condition_key = condition_key;
        probe_obs.branch = branch;
        probe_obs.before = &probe_res.before;
        probe_obs.probe = &probe_res.probe;
        effective = policy.decide_after_probe(step, probe_obs);
    }

    const GraphVariantPlan* eff_variant = program.find_variant(effective.variant);
    const GraphVariantKind eff_kind = eff_variant ? eff_variant->kind : GraphVariantKind::FULL;

    if (eff_kind == GraphVariantKind::REUSE || eff_kind == GraphVariantKind::PREDICT) {
        CacheReconstructContext rc;
        rc.step = step;
        rc.condition_key = condition_key;
        rc.input = hooks.input;
        sd::Tensor<float> feature = policy.reconstruct(rc);
        if (!feature.empty()) {
            int s = region_start;
            int e = region_end;
            int pd = 0;
            if (eff_variant) {
                seam_region(*eff_variant, &s, &e, &pd);
            }
            sd::Tensor<float> out = hooks.inject(feature, s, e);
            if (!out.empty()) {
                return out;
            }
        }
        // Reconstruction/inject failed -> capturing full step.
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
