#pragma once

#include "ggml.h"

// SageAttention2-style INT8-QK + F16-PV fused attention custom op.
//
// The compute entry point is defined inside ggml-cuda.cu (so it can take the
// ggml_backend_cuda_context and use ctx.pool() for CUDA-graph-safe scratch),
// and forwards to the launcher below which lives in ed_cuda_sage_attn.cu.
//
// supported() is safe to call from any TU (only inspects the tensor/op).

bool ed_cuda_sage_attn_custom_supported(const ggml_tensor * dst);

// Scratch is provided by the caller (allocated from the ggml CUDA pool). The
// launcher computes the exact byte sizes it needs via
// ed_cuda_sage_attn_scratch_sizes(); the caller allocates and passes the
// buffers back in. Returns false if unsupported (caller must fall back).
struct ed_cuda_sage_attn_scratch {
    void * q_int8  = nullptr; // int8 [d_head * L_q * HN]
    void * k_int8  = nullptr; // int8 [d_head * L_k * HN]
    void * q_scale = nullptr; // float
    void * k_scale = nullptr; // float
    void * km      = nullptr; // float [d_head * HN]
    void * v_f16   = nullptr; // half  [d_head * L_k * HN]
    void * o_half  = nullptr; // half  [d_head * L_q * HN]
    // Smooth-Q ("Thorough Outlier Smoothing"). Only allocated/used when smooth-Q
    // is enabled (head_dim==128 by default). qm is Q's per-channel mean; qk_bias
    // is the rank-1 per-key compensation qm·Kᵀ added back before softmax.
    void * qm      = nullptr; // float [d_head * HN]
    void * qk_bias = nullptr; // float [HN * L_k]
    // Stage-2 Part A (FP8-PV). Only allocated/used when ED_SAGE_PV_FP8 is on.
    void * v_fp8   = nullptr; // int8 (e4m3) [HN * d_head * L_k_pad] transposed+permuted
    void * v_scale = nullptr; // float [HN * d_head] per-channel V scale
    // Near-uniform head fallback (head_dim==128). uniform_flag[hn] is set by the
    // detection kernel to 1 when head hn's attention is near-uniform (collapses
    // under the INT8-PV kernel); the overwrite kernel then recomputes those heads
    // in full precision. head_entropy[hn] holds the measured normalized entropy
    // (for ED_SAGE_ATTN_DEBUG reporting). Both are one entry per head-batch.
    void * uniform_flag  = nullptr; // int32 [HN]
    void * head_entropy  = nullptr; // float [HN]
    void * detect_acc    = nullptr; // float [4*HN] atomics: {sse, ref2, ent_sum, n}
};

struct ed_cuda_sage_attn_scratch_sizes {
    size_t q_int8_bytes  = 0;
    size_t k_int8_bytes  = 0;
    size_t q_scale_bytes = 0;
    size_t k_scale_bytes = 0;
    size_t km_bytes      = 0;
    size_t v_f16_bytes   = 0;
    size_t o_half_bytes  = 0;
    size_t qm_bytes      = 0;
    size_t qk_bias_bytes = 0;
    size_t v_fp8_bytes   = 0;
    size_t v_scale_bytes = 0;
    size_t uniform_flag_bytes = 0;
    size_t head_entropy_bytes = 0;
    size_t detect_acc_bytes   = 0;
};

// Fills `out` with the scratch byte sizes required for `dst`. Returns false if
// the op is unsupported.
bool ed_cuda_sage_attn_scratch_sizes_for(const ggml_tensor * dst, ed_cuda_sage_attn_scratch_sizes * out);

// Runs the full pipeline (K-mean, quant+smooth-K, V->f16, sage kernel, O->f32)
// on the given stream using caller-provided scratch. Returns false if
// unsupported.
bool ed_cuda_sage_attn_custom_compute(ggml_tensor * dst,
                                      const ed_cuda_sage_attn_scratch * scratch,
                                      void * stream);
