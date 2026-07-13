#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/optimization/cache/ir/cache_program.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {
namespace detail {

// Standard variant ids shared by every method (FULL always 0).
constexpr GraphVariantId kVariantFull = 0;
constexpr GraphVariantId kVariantReuse = 1;
constexpr GraphVariantId kVariantPredict = 2;
constexpr GraphVariantId kVariantProbe = 3;

// Build a FULL + REUSE program over one block-stack segment. `reuse_mode`
// selects LOAD_CACHED (diff/residual reuse) or PREDICT_FROM_HISTORY
// (extrapolation). Covers Output methods (whole-output diff) and Feature methods
// (block-stack residual) alike — the difference is only which site the lowering
// captures/injects, carried by the policy's requirements.
inline CacheProgram make_reuse_program(const char* method,
                                       int block_segment_id,
                                       SegmentExecutionMode reuse_mode,
                                       const CacheSlotDesc& slot) {
    CacheProgram program;
    program.method_name = method;
    program.slots.push_back(slot);

    GraphVariantPlan full;
    full.id = kVariantFull;
    full.kind = GraphVariantKind::FULL;
    SegmentPlan full_seg;
    full_seg.segment_id = block_segment_id;
    full_seg.execution = SegmentExecutionMode::FULL_COMPUTE;
    full.segments.push_back(full_seg);
    program.variants.push_back(full);

    GraphVariantPlan reuse;
    reuse.id = (reuse_mode == SegmentExecutionMode::PREDICT_FROM_HISTORY) ? kVariantPredict : kVariantReuse;
    reuse.kind = (reuse_mode == SegmentExecutionMode::PREDICT_FROM_HISTORY) ? GraphVariantKind::PREDICT
                                                                           : GraphVariantKind::REUSE;
    SegmentPlan reuse_seg;
    reuse_seg.segment_id = block_segment_id;
    reuse_seg.execution = reuse_mode;
    reuse.segments.push_back(reuse_seg);
    program.variants.push_back(reuse);

    return program;
}

// A simple single-slot descriptor helper.
inline CacheSlotDesc make_slot(int id, const char* name, int history_depth = 1,
                               bool device_backed = false) {
    CacheSlotDesc s;
    s.id = id;
    s.name = name;
    s.lifetime = CacheLifetime::MULTI_STEP;
    s.access = CacheAccessMode::READ_WRITE;
    s.history_depth = history_depth;
    s.device_backed = device_backed;
    return s;
}

// Build the declarative program for an Output-granularity diff-reuse method
// (EasyCache / UCache / DBCache-static). The residual (output - input) lives in
// one cache slot; the lowering's action interpreter drives capture and reuse, so
// the policy needs no reconstruct()/observe-store — only its scalar decision
// state. Math is bit-identical to the old inline store_tensor_diff /
// apply_tensor_diff: cache.difference computes inputs[1]-inputs[0] = output-input;
// cache.weighted_blend with weights {1,1} computes input + diff.
//   FULL.after  : DIFFERENCE(input, output) -> slot
//   REUSE.before: BLEND(input, LOAD slot)   -> model output
inline CacheProgram make_output_diff_program(const char* method, int block_segment_id) {
    CacheProgram program;
    program.method_name = method;
    program.slots.push_back(make_slot(0, "denoiser_output_diff"));

    GraphVariantPlan full;
    full.id = kVariantFull;
    full.kind = GraphVariantKind::FULL;
    SegmentPlan full_seg;
    full_seg.segment_id = block_segment_id;
    full_seg.execution = SegmentExecutionMode::FULL_COMPUTE;
    {
        CacheAction diff;
        diff.kind = CacheActionKind::DIFFERENCE;
        diff.op = "cache.difference";
        diff.inputs = {ValueRef::of_ambient(kAmbientInput), ValueRef::of_ambient(kAmbientModelOutput)};
        diff.outputs = {ValueRef::of_slot(0)};
        full_seg.after.push_back(diff);
    }
    full.segments.push_back(full_seg);
    program.variants.push_back(full);

    GraphVariantPlan reuse;
    reuse.id = kVariantReuse;
    reuse.kind = GraphVariantKind::REUSE;
    SegmentPlan reuse_seg;
    reuse_seg.segment_id = block_segment_id;
    reuse_seg.execution = SegmentExecutionMode::LOAD_CACHED;
    {
        CacheAction load;
        load.kind = CacheActionKind::LOAD;
        load.inputs = {ValueRef::of_slot(0)};
        load.outputs = {ValueRef::of_temp(0)};
        reuse_seg.before.push_back(load);

        CacheAction blend;
        blend.kind = CacheActionKind::BLEND;
        blend.op = "cache.weighted_blend";
        blend.params.floats = {1.0f, 1.0f};
        blend.inputs = {ValueRef::of_ambient(kAmbientInput), ValueRef::of_temp(0)};
        blend.outputs = {ValueRef::of_ambient(kAmbientModelOutput)};
        reuse_seg.before.push_back(blend);
    }
    reuse.segments.push_back(reuse_seg);
    program.variants.push_back(reuse);

    return program;
}

// Build the declarative program for a Feature-granularity single-residual reuse
// method (MagCache). The captured block-stack residual lives in one slot; the
// seam's capture/inject calls remain the model-compute boundary, but the residual
// STORAGE is declarative:
//   FULL.after   : STORE captured-feature -> slot
//   REUSE.before : LOAD slot -> inject-feature (lowering calls hooks.inject)
// device_backed: when true (MagCache on a GPU runner with a device store wired),
// slot0 is a device tensor so the residual is stored/reused on-device without a
// host round-trip. SenCache leaves it false (host declarative path).
inline CacheProgram make_feature_reuse_program(const char* method, int block_segment_id,
                                               bool device_backed = false) {
    CacheProgram program;
    program.method_name = method;
    program.slots.push_back(make_slot(0, "block_stack_residual", 1, device_backed));

    GraphVariantPlan full;
    full.id = kVariantFull;
    full.kind = GraphVariantKind::FULL;
    SegmentPlan full_seg;
    full_seg.segment_id = block_segment_id;
    full_seg.execution = SegmentExecutionMode::FULL_COMPUTE;
    {
        CacheAction store;
        store.kind = CacheActionKind::STORE;
        store.inputs = {ValueRef::of_ambient(kAmbientCapturedFeature)};
        store.outputs = {ValueRef::of_slot(0)};
        full_seg.after.push_back(store);
    }
    full.segments.push_back(full_seg);
    program.variants.push_back(full);

    GraphVariantPlan reuse;
    reuse.id = kVariantReuse;
    reuse.kind = GraphVariantKind::REUSE;
    SegmentPlan reuse_seg;
    reuse_seg.segment_id = block_segment_id;
    reuse_seg.execution = SegmentExecutionMode::LOAD_CACHED;
    {
        CacheAction load;
        load.kind = CacheActionKind::LOAD;
        load.inputs = {ValueRef::of_slot(0)};
        load.outputs = {ValueRef::of_ambient(kAmbientInjectFeature)};
        reuse_seg.before.push_back(load);
    }
    reuse.segments.push_back(reuse_seg);
    program.variants.push_back(reuse);

    return program;
}

// Build the declarative program for a Feature-granularity HISTORY-EXTRAPOLATION
// method (TaylorSeer). The captured block-stack residual is kept in a history
// ring of raw features (depth = order + 1); on a skip the injected feature is a
// weighted blend of the last `order+1` ring entries, with the weights supplied
// per-step by the policy's decide() (RuntimeDecision::reuse_coeffs). This is the
// finite-difference Taylor extrapolation expressed as a linear combination of raw
// features — the ring buffer's real driver.
//   FULL.after    : STORE captured-feature -> slot0 ; ROTATE_HISTORY slot0
//   PREDICT.before: LOAD slot0@1..depth -> temps ; BLEND(coeffs_from_decision)
//                   -> inject-feature (lowering calls hooks.inject)
inline CacheProgram make_taylor_history_program(const char* method, int block_segment_id,
                                                int order) {
    const int depth = std::max(2, order + 2);
    CacheProgram program;
    program.method_name = method;
    program.slots.push_back(make_slot(0, "block_stack_feature_history", depth));

    GraphVariantPlan full;
    full.id = kVariantFull;
    full.kind = GraphVariantKind::FULL;
    SegmentPlan full_seg;
    full_seg.segment_id = block_segment_id;
    full_seg.execution = SegmentExecutionMode::FULL_COMPUTE;
    {
        CacheAction store;
        store.kind = CacheActionKind::STORE;
        store.inputs = {ValueRef::of_ambient(kAmbientCapturedFeature)};
        store.outputs = {ValueRef::of_slot(0)};
        full_seg.after.push_back(store);

        CacheAction rotate;
        rotate.kind = CacheActionKind::ROTATE_HISTORY;
        rotate.slot = 0;
        full_seg.after.push_back(rotate);
    }
    full.segments.push_back(full_seg);
    program.variants.push_back(full);

    GraphVariantPlan predict;
    predict.id = kVariantPredict;
    predict.kind = GraphVariantKind::PREDICT;
    SegmentPlan pred_seg;
    pred_seg.segment_id = block_segment_id;
    pred_seg.execution = SegmentExecutionMode::PREDICT_FROM_HISTORY;
    {
        // LOAD each history depth into its own temp. After a committed FULL step
        // we ROTATE, so at PREDICT time depth 1 is the newest stored feature
        // (depth 0 is the fresh, not-yet-written ring head). The blend combines
        // depths 1..depth-1 with per-step weights the policy supplies.
        CacheAction blend;
        blend.kind = CacheActionKind::BLEND;
        blend.op = "cache.weighted_blend";
        blend.coeffs_from_decision = true;
        for (int k = 1; k < depth; ++k) {
            CacheAction load;
            load.kind = CacheActionKind::LOAD;
            load.inputs = {ValueRef::of_slot_history(0, k)};
            load.outputs = {ValueRef::of_temp(k - 1)};
            pred_seg.before.push_back(load);
            blend.inputs.push_back(ValueRef::of_temp(k - 1));
        }
        blend.outputs = {ValueRef::of_ambient(kAmbientInjectFeature)};
        pred_seg.before.push_back(blend);
    }
    predict.segments.push_back(pred_seg);
    program.variants.push_back(predict);

    return program;
}

// Build the declarative program for a Probe-granularity trajectory-aligned reuse
// method (DiCache). A shallow PROBE variant runs first; the policy's
// decide_after_probe() then picks FULL or REUSE and, for REUSE, supplies the
// gamma-aligned blend weights via RuntimeDecision::reuse_coeffs. The block-stack
// residual is kept in a ring (depth 3: readable history 2 + 1 consumed by
// STORE-then-ROTATE):
//   FULL.after    : STORE captured-feature -> slot0 ; ROTATE_HISTORY slot0
//   PROBE         : shallow prefix of `probe_depth` blocks (no actions)
//   REUSE.before  : LOAD slot0@1,@2 -> temps ; BLEND(coeffs_from_decision)
//                   -> inject-feature.  weights = [gamma, 1-gamma] give
//                   resid_prev1*gamma + resid_prev2*(1-gamma)
//                   = resid_prev2 + gamma*(resid_prev1 - resid_prev2).
inline CacheProgram make_dicache_program(const char* method, int block_segment_id,
                                         int probe_depth) {
    CacheProgram program;
    program.method_name = method;
    program.slots.push_back(make_slot(0, "block_stack_residual", 3));

    GraphVariantPlan full;
    full.id = kVariantFull;
    full.kind = GraphVariantKind::FULL;
    SegmentPlan full_seg;
    full_seg.segment_id = block_segment_id;
    full_seg.execution = SegmentExecutionMode::FULL_COMPUTE;
    {
        CacheAction store;
        store.kind = CacheActionKind::STORE;
        store.inputs = {ValueRef::of_ambient(kAmbientCapturedFeature)};
        store.outputs = {ValueRef::of_slot(0)};
        full_seg.after.push_back(store);
        CacheAction rotate;
        rotate.kind = CacheActionKind::ROTATE_HISTORY;
        rotate.slot = 0;
        full_seg.after.push_back(rotate);
    }
    full.segments.push_back(full_seg);
    program.variants.push_back(full);

    GraphVariantPlan reuse;
    reuse.id = kVariantReuse;
    reuse.kind = GraphVariantKind::REUSE;
    SegmentPlan reuse_seg;
    reuse_seg.segment_id = block_segment_id;
    reuse_seg.execution = SegmentExecutionMode::LOAD_CACHED;
    {
        CacheAction load1;
        load1.kind = CacheActionKind::LOAD;
        load1.inputs = {ValueRef::of_slot_history(0, 1)};
        load1.outputs = {ValueRef::of_temp(0)};
        reuse_seg.before.push_back(load1);
        CacheAction load2;
        load2.kind = CacheActionKind::LOAD;
        load2.inputs = {ValueRef::of_slot_history(0, 2)};
        load2.outputs = {ValueRef::of_temp(1)};
        reuse_seg.before.push_back(load2);
        CacheAction blend;
        blend.kind = CacheActionKind::BLEND;
        blend.op = "cache.weighted_blend";
        blend.coeffs_from_decision = true;
        blend.inputs = {ValueRef::of_temp(0), ValueRef::of_temp(1)};
        blend.outputs = {ValueRef::of_ambient(kAmbientInjectFeature)};
        reuse_seg.before.push_back(blend);
    }
    reuse.segments.push_back(reuse_seg);
    program.variants.push_back(reuse);

    GraphVariantPlan probe;
    probe.id = kVariantProbe;
    probe.kind = GraphVariantKind::PROBE;
    SegmentPlan probe_seg;
    probe_seg.segment_id = block_segment_id;
    probe_seg.execution = SegmentExecutionMode::PROBE;
    probe_seg.probe_depth = std::max(1, probe_depth);
    probe.segments.push_back(probe_seg);
    program.variants.push_back(probe);

    return program;
}

// Map [start_percent, end_percent] of the sampling schedule to a sigma window.
// Ported verbatim from the old policies' set_sigmas(); shared so every method
// computes the active window identically.
struct SigmaWindow {
    float start_sigma = std::numeric_limits<float>::max();
    float end_sigma = 0.0f;

    void configure(const std::vector<float>& sigmas, float start_percent, float end_percent) {
        if (sigmas.size() < 2) {
            return;
        }
        const size_t n_steps = sigmas.size() - 1;
        size_t start_step = static_cast<size_t>(start_percent * n_steps);
        size_t end_step = static_cast<size_t>(end_percent * n_steps);
        if (start_step >= n_steps) {
            start_step = n_steps - 1;
        }
        if (end_step >= n_steps) {
            end_step = n_steps - 1;
        }
        start_sigma = sigmas[start_step];
        end_sigma = sigmas[end_step];
        if (start_sigma < end_sigma) {
            std::swap(start_sigma, end_sigma);
        }
    }

    bool contains(float sigma) const {
        return sigma <= start_sigma && sigma > end_sigma;
    }
};

inline float mean_abs(const float* data, size_t ne) {
    if (data == nullptr || ne == 0) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < ne; ++i) {
        sum += std::fabs(data[i]);
    }
    return sum / static_cast<float>(ne);
}

// mean(|a - b|) / mean(|b|) — relative-L1 metric used by DiCache.
inline float rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) {
        return 1.0f;
    }
    float num = 0.0f;
    float den = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs(a[i] - b[i]);
        den += std::fabs(b[i]);
    }
    return num / (den + 1e-6f);
}

// Raw-pointer variant so callers can compute the metric straight off tensor data
// without first copying it into a std::vector (the copy of a ~50MB probe/before
// tensor is itself a bandwidth-bound pass; DiCache runs this every step).
inline float rel_l1_ptr(const float* a, const float* b, size_t n) {
    if (a == nullptr || b == nullptr || n == 0) {
        return 1.0f;
    }
    float num = 0.0f;
    float den = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        num += std::fabs(a[i] - b[i]);
        den += std::fabs(b[i]);
    }
    return num / (den + 1e-6f);
}

inline float rel_l1_abs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) {
        return 0.0f;
    }
    float num = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs(a[i] - b[i]);
    }
    return num / static_cast<float>(a.size());
}

// L2 norm of (a - b); returns -1 on empty/shape mismatch. Used by SenCache's
// finite-difference Jacobian calibration.
inline double l2_diff(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
    if (a.empty() || b.empty() || a.numel() != b.numel()) {
        return -1.0;
    }
    const float* pa = a.data();
    const float* pb = b.data();
    const int64_t n = a.numel();
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
        sum += d * d;
    }
    return std::sqrt(sum);
}

// Finite-difference Taylor extrapolation state. Ported verbatim from the old
// cache_policy.cpp; shared by TaylorSeer (feature) and CacheDiT (output).
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
        return last_computed_step >= n_derivatives && !dy_current.empty() &&
               dy_current[0].size() == size;
    }

    bool ready() const {
        return last_computed_step >= n_derivatives && !dy_current.empty() &&
               !dy_current[0].empty();
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

}  // namespace detail
}  // namespace cache
}  // namespace edgedit
