#pragma once

#include <limits>

#include "utils/tensor.hpp"

namespace sd {

// Result of a cache-aware compute pass, produced by the runner's substep helpers
// and consumed by the cache lowering. `output` is the usual denoiser output
// (empty on a probe pass). `feature`/`before`/`probe` carry the block-stack
// tensors a Feature/Probe policy needs, already read back to host; the scalar
// fields carry the on-device DiCache decision metrics (NaN when not produced).
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

}  // namespace sd
