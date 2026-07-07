#pragma once

#include <algorithm>

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
        return ggml_add(ctx, x_before, inject_feature);
    }
    void build_feature(ggml_context* ctx, ggml_tensor* x_after) {
        if (before_node != nullptr) {
            feature_node = ggml_sub(ctx, x_after, before_node);
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
