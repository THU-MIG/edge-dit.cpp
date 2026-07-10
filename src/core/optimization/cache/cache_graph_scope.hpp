#pragma once

#include <algorithm>
#include <limits>

#include "ggml.h"
#include "utils/tensor.hpp"

namespace sd {

// Result of a cache-aware compute pass, produced by the runner device helpers
// and consumed by the CacheController. `output` is the usual denoiser output
// (empty on a probe pass). `feature`/`before`/`probe` carry the block-stack
// tensors a Feature/Probe policy needs, already read back to host.
struct DiffusionCacheResult {
    Tensor<float> output;
    Tensor<float> feature;  // Capture: cached-region residual
    Tensor<float> before;   // Probe: region input
    Tensor<float> probe;    // Probe: hidden state after probe_depth blocks
    // GPU DiCache scalar readbacks (NaN when not produced this pass).
    float delta_y = std::numeric_limits<float>::quiet_NaN();
    float delta_x = std::numeric_limits<float>::quiet_NaN();
    float gamma = std::numeric_limits<float>::quiet_NaN();
};

// Build-time cache seam shared by all DiT model graphs. A model's forward()
// consults a nullable CacheGraphScope* (carried on GGMLRunnerContext) inside its
// block loop; when null the graph is byte-identical to the uncached path. The
// scope is method-AGNOSTIC — it only knows the graph shapes a cache step can
// take, never which cache algorithm requested them. The runner owns the
// ggml_cgraph and does the cache()/ggml_set_output/expand wiring using the
// tensor pointers the model records here.
//
// The cached region is the contiguous block interval [region_start, region_end)
// of the model's block stack. region_end < 0 means "to the end of the stack".
// The default (0, -1) is the whole stack, which the current whole-model methods
// (TaylorSeer / MagCache / DiCache) use. A policy MAY instead narrow it to a
// sub-interval: blocks before region_start and from region_end on always run;
// only the region is captured on full steps and skipped (residual injected) on
// reuse steps. This costs nothing beyond the one compute pass the step already
// runs — it is a pure build-time structural decision, not an extra pass.
//
// Modes:
//   Capture - run every block; expose feature = region_after - region_before.
//   Inject  - run the blocks OUTSIDE the region; at region_start, replace the
//             region with region_before + inject_feature and resume at region_end.
//   Probe   - run probe_depth blocks from region_start; expose that hidden state,
//             then the model returns it directly (final layer is not run).
//
// Two API levels:
//   * In-loop sugar (begin_region / step_inject_region / end_region /
//     stop_after_block) — used by uniform single-loop models (mmdit, qwen, wan)
//     and by flux for sub-regions inside its double-block loop.
//   * Low-level primitives (snapshot_before / build_feature / add_injected) —
//     used by flux's whole-stack path, whose residual spans the double AND single
//     loops and is captured after the streams are recombined.
struct CacheGraphScope {
    enum class Mode { Capture, Inject, Probe };

    Mode mode = Mode::Capture;
    int probe_depth = 0;

    // Cached region = blocks [region_start, region_end). Defaults to whole stack.
    int region_start = 0;
    int region_end = -1;  // < 0 => to the end of the stack

    // Inject input: reconstructed residual, created by the runner with the same
    // shape as the region's input state.
    ggml_tensor* inject_feature = nullptr;

    // ---- GPU-side DiCache (Probe granularity) ----
    // When these persistent cross-step tensors are provided, the seam computes the
    // DiCache decision metric and residual reconstruction ON the GPU instead of
    // reading ~50MB tensors back to host every step. All are runner-owned and
    // persist across the sampling loop; nullptr => host-side path (or non-DiCache).
    ggml_tensor* prev_probe_node = nullptr;   // probe state from the last computed step
    ggml_tensor* prev_input_node = nullptr;   // block-stack input from last computed step
    ggml_tensor* probe_prev1_node = nullptr;  // newest probe residual (probe - before)
    ggml_tensor* probe_prev2_node = nullptr;  // 2nd-newest probe residual
    ggml_tensor* resid_prev1_node = nullptr;  // newest full block-stack residual
    ggml_tensor* resid_prev2_node = nullptr;  // 2nd-newest full residual
    ggml_tensor* gamma_scalar = nullptr;      // Inject: trajectory-alignment gamma [1]
    bool gpu_metric = false;                  // Probe: emit delta_y/delta_x/gamma scalar nodes

    // Recorded by the runner after a Probe pass (scalar readbacks); consumed by the
    // policy. Named nodes: kCacheDeltaYName / kCacheDeltaXName / kCacheGammaName.
    ggml_tensor* delta_y_node = nullptr;
    ggml_tensor* delta_x_node = nullptr;
    ggml_tensor* gamma_node = nullptr;
    ggml_tensor* probe_resid_node = nullptr;  // Capture: (probe_state - block_input)

    // Recorded by the model, consumed by the runner after forward() returns.
    ggml_tensor* before_node = nullptr;   // region input (Capture anchor / Probe readback)
    ggml_tensor* probe_node = nullptr;    // hidden state after probe_depth blocks (Probe)
    ggml_tensor* feature_node = nullptr;  // region_after - region_before (Capture)

    bool capture_mode() const { return mode == Mode::Capture; }
    bool inject_mode() const { return mode == Mode::Inject; }
    bool probe_mode() const { return mode == Mode::Probe; }

    // A finite region_end marks a sub-interval; region_end < 0 is the whole stack.
    bool is_subregion() const { return region_end >= 0; }
    int resolved_region_end(int n) const { return region_end < 0 ? n : std::min(region_end, n); }
    bool at_region_start(int i) const { return i == region_start; }
    // Block index to resume at after an injected region is skipped.
    int inject_resume_index(int n) const { return resolved_region_end(n); }

    // ---- Low-level primitives (whole-stack path) ----
    void snapshot_before(ggml_tensor* x) { before_node = x; }
    ggml_tensor* add_injected(ggml_context* ctx, ggml_tensor* x_before) const {
        // GPU-side Dynamic Cache Trajectory Alignment: when the persistent residual
        // nodes and the gamma scalar are present, reconstruct
        //   x_before + resid_prev2 + gamma*(resid_prev1 - resid_prev2)
        // entirely on-device (matches ref _dicache_apply_cached_residual), instead
        // of injecting a host-reconstructed residual. Falls back to the simple
        // x_before + inject_feature path for host-side / feature-cache methods.
        if (gpu_reconstruct_available()) {
            ggml_tensor* diff = ggml_sub(ctx, resid_prev1_node, resid_prev2_node);
            // gamma is a runtime [1] scalar tensor -> broadcast-multiply.
            ggml_tensor* scaled = ggml_mul(ctx, diff, gamma_scalar);
            ggml_tensor* aligned = ggml_add(ctx, resid_prev2_node, scaled);
            return ggml_add(ctx, x_before, aligned);
        }
        // GPU-side single-residual reuse (MagCache/TaylorSeer feature reuse):
        // inject the last captured residual straight from device memory, avoiding
        // the ~50MB host reconstruct copy + H2D upload the inject_feature path pays.
        if (gpu_reuse_available()) {
            return ggml_add(ctx, x_before, resid_prev1_node);
        }
        return ggml_add(ctx, x_before, inject_feature);
    }
    bool gpu_reconstruct_available() const {
        return resid_prev1_node != nullptr && resid_prev2_node != nullptr &&
               gamma_scalar != nullptr;
    }
    // Single-residual on-GPU reuse: resid_prev1 present, no ring/gamma (MagCache).
    bool gpu_reuse_available() const {
        return resid_prev1_node != nullptr && resid_prev2_node == nullptr &&
               gamma_scalar == nullptr;
    }
    void build_feature(ggml_context* ctx, ggml_tensor* x_after) {
        if (before_node != nullptr) {
            feature_node = ggml_sub(ctx, x_after, before_node);
        }
    }
    // Build the GPU-side DiCache decision scalars from the probe state. Called by
    // the runner after the model records before_node/probe_node on a Probe pass,
    // when gpu_metric is set and the persistent prev_* nodes are available.
    //   delta_y = sum|probe - prev_probe| / sum|prev_probe|
    //   delta_x = sum|before - prev_input| / sum|prev_input|   (delta_minus only)
    //   gamma   = clamp( sum|(probe-before) - probe_prev2| / sum|probe_prev1 - probe_prev2|, 1, 1.5)
    // (mean's 1/N cancels in each ratio, so sum is used directly.)
    void build_probe_metrics(ggml_context* ctx) {
        if (!gpu_metric || probe_node == nullptr || before_node == nullptr) {
            return;
        }
        auto rel = [&](ggml_tensor* cur, ggml_tensor* ref) -> ggml_tensor* {
            ggml_tensor* num = ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, cur, ref)));
            ggml_tensor* den = ggml_sum(ctx, ggml_abs(ctx, ref));
            return ggml_div(ctx, num, den);
        };
        if (prev_probe_node != nullptr) {
            delta_y_node = rel(probe_node, prev_probe_node);
        }
        if (prev_input_node != nullptr) {
            delta_x_node = rel(before_node, prev_input_node);
        }
        if (probe_prev1_node != nullptr && probe_prev2_node != nullptr) {
            ggml_tensor* cur_resid = ggml_sub(ctx, probe_node, before_node);
            ggml_tensor* num = ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, cur_resid, probe_prev2_node)));
            ggml_tensor* den = ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, probe_prev1_node, probe_prev2_node)));
            gamma_node = ggml_div(ctx, num, den);
        }
    }


    // ---- In-loop sugar (per-block region boundaries) ----
    // Call before block i runs: snapshot the region-input anchor iff i is the
    // region's first block and we are capturing/probing (not injecting).
    void begin_region(int i, ggml_tensor* x) {
        if (!inject_mode() && at_region_start(i)) {
            snapshot_before(x);
        }
    }
    // Call at the top of iteration i under Inject: if i is the region start,
    // returns region_before + inject_feature (caller assigns it and jumps to
    // inject_resume_index). Returns nullptr when block i should just run.
    ggml_tensor* step_inject_region(ggml_context* ctx, int i, ggml_tensor* x) const {
        if (inject_mode() && at_region_start(i)) {
            return add_injected(ctx, x);
        }
        return nullptr;
    }
    // GPU DiCache: during a Capture (full compute) step, also snapshot the
    // hidden state after probe_depth blocks so the runner can refresh the
    // persistent prev_probe / probe-residual used by the NEXT step's decision.
    // No effect unless capturing with gpu_metric and probe_depth>0.
    void record_capture_probe_state(ggml_context* ctx, int i, ggml_tensor* x) {
        if (capture_mode() && gpu_metric && probe_depth > 0 &&
            (i - region_start) + 1 == probe_depth) {
            probe_node = x;
            // Probe residual (probe_state - block_input) for the ring; before_node
            // is the whole-stack input, snapshotted at region start.
            if (before_node != nullptr) {
                probe_resid_node = ggml_sub(ctx, x, before_node);
            }
        }
    }
    // Call after block i runs: on Capture, build the region residual once i is
    // the region's last block.
    void end_region(ggml_context* ctx, int i, int n, ggml_tensor* x) {
        if (mode == Mode::Capture && before_node != nullptr && i + 1 == resolved_region_end(n)) {
            build_feature(ctx, x);
        }
    }

    // True when a Probe should stop right after this 0-based block index
    // (counted from region_start).
    bool stop_after_block(int i) const {
        return mode == Mode::Probe && (i - region_start) + 1 >= probe_depth;
    }
    void on_probe(ggml_tensor* x_probe) { probe_node = x_probe; }
};

}  // namespace sd
