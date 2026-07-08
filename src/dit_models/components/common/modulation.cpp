#include "dit_models/components/common/modulation.hpp"

#ifdef ED_ENABLE_CUDA_MODULATION
#include "backend/ggml/ed_ggml_modulation_ext.hpp"
#endif

namespace dit {

ggml_tensor* modulate(ggml_context* ctx,
                      ggml_tensor* x,
                      ggml_tensor* shift,
                      ggml_tensor* scale,
                      bool skip_reshape) {
    if (!skip_reshape) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
        shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);
    }
#ifdef ED_ENABLE_CUDA_MODULATION
    // Fuse x + x*scale + shift into one CUDA kernel; falls back to split ops
    // (and on CPU) when shapes are unsupported.
    if (auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx, x, shift, scale)) {
        return fused;
    }
#endif
    x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    x = ggml_add(ctx, x, shift);
    return x;
}

ggml_tensor* modulate(ggml_context* ctx,
                      ggml_tensor* x,
                      ggml_tensor* scale,
                      bool skip_reshape) {
    if (!skip_reshape) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
    }
    x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    return x;
}

ggml_tensor* residual_gate(ggml_context* ctx,
                           ggml_tensor* residual,
                           ggml_tensor* x,
                           ggml_tensor* gate,
                           bool skip_reshape) {
    if (!skip_reshape) {
        gate = ggml_reshape_3d(ctx, gate, gate->ne[0], 1, gate->ne[1]);
    }
#ifdef ED_ENABLE_CUDA_MODULATION
    if (auto fused = edgedit::ggml_ext::fused_residual_gate_custom(ctx, residual, x, gate)) {
        return fused;
    }
#endif
    return ggml_add(ctx, residual, ggml_mul(ctx, x, gate));
}

}  // namespace dit
