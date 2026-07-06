#pragma once

#include "ggml.h"
#include "utils/tensor.hpp"

namespace sd {

// Result of a cache-aware compute pass, produced by the runner device helpers
// and consumed by the CacheController. `output` is the usual denoiser output
// (empty on a probe pass). `feature`/`before`/`probe` carry the block-stack
// tensors a Feature/Probe policy needs, already read back to host.
struct DiffusionCacheResult {
    Tensor<float> output;
    Tensor<float> feature;  // Capture: block-stack residual
    Tensor<float> before;   // Probe: block-stack input
    Tensor<float> probe;    // Probe: hidden state after probe_depth blocks
};

// Build-time cache seam shared by all DiT model graphs. A model's forward()
// consults a nullable CacheGraphScope* (carried on GGMLRunnerContext) at its
// block-stack boundary; when null the graph is byte-identical to the uncached
// path. The scope is method-AGNOSTIC — it only knows the three graph shapes a
// cache step can take, never which cache algorithm requested them. The runner
// owns the ggml_cgraph and does the cache()/ggml_set_output/expand wiring using
// the tensor pointers the model records here.
//
// Modes:
//   Capture - run the whole block stack; expose feature = main_after - main_before.
//   Inject  - skip the block stack; main = main_before + inject_feature.
//   Probe   - run only probe_depth blocks; expose the probe hidden state, then
//             the model returns it directly (its final layer is not run).
struct CacheGraphScope {
    enum class Mode { Capture, Inject, Probe };

    Mode mode = Mode::Capture;
    int probe_depth = 0;

    // Inject input: reconstructed residual, created by the runner with the same
    // shape as the main-stream block-stack input.
    ggml_tensor* inject_feature = nullptr;

    // Recorded by the model, consumed by the runner after forward() returns:
    ggml_tensor* before_node = nullptr;   // block-stack input (Capture anchor / Probe readback)
    ggml_tensor* probe_node = nullptr;    // hidden state after probe_depth blocks (Probe)
    ggml_tensor* feature_node = nullptr;  // main_after - main_before (Capture)

    bool capture_mode() const { return mode == Mode::Capture; }
    bool inject_mode() const { return mode == Mode::Inject; }
    bool probe_mode() const { return mode == Mode::Probe; }

    // Called just before the block loop with the main-stream input tensor.
    void on_stack_begin(ggml_tensor* x_before) { before_node = x_before; }

    // Inject path: the whole stack collapses to a residual add.
    ggml_tensor* injected_stack_output(ggml_context* ctx, ggml_tensor* x_before) const {
        return ggml_add(ctx, x_before, inject_feature);
    }

    // True when the probe should stop right after this 0-based block index.
    bool stop_after_block(int block_index) const {
        return mode == Mode::Probe && block_index + 1 >= probe_depth;
    }

    // Probe path: record the shallow hidden state; the model then returns it.
    void on_probe(ggml_tensor* x_probe) { probe_node = x_probe; }

    // Capture path: build the block-stack residual node for the runner to read.
    void on_stack_end(ggml_context* ctx, ggml_tensor* x_after) {
        if (mode == Mode::Capture && before_node != nullptr) {
            feature_node = ggml_sub(ctx, x_after, before_node);
        }
    }
};

}  // namespace sd
