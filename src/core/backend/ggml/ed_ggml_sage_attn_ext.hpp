#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

// Parameter packing helpers for the SageAttention2-style INT8-QK + F16-PV
// fused attention custom op. Mirrors the pattern used by
// ed_ggml_attention_ext.hpp (magic + small field packed into the custom-op
// userdata pointer). Kept header-only so both the graph-build side
// (ggml_extend.hpp) and the CUDA compute side (ed_cuda_sage_attn.cu) share one
// definition.

namespace edgedit::ggml_ext {

constexpr uint32_t kSageAttnCustomMagic = 0x53414751u; // "SAGQ"

struct SageAttnCustomParams {
    uint32_t magic  = kSageAttnCustomMagic;
    int32_t  n_head = 0;
};

// Global on/off switch. Default OFF: only enabled when ED_SAGE_ATTN is set to a
// truthy value. This keeps the original F16 flash-attn path as the default and
// makes A/B comparison a single env-var flip.
inline bool sage_attn_enabled() {
    const char* env = std::getenv("ED_SAGE_ATTN");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0 &&
           std::strcmp(env, "off") != 0 &&
           std::strcmp(env, "OFF") != 0;
}

// Stage-2 Part A: FP8-E4M3 PV upgrade. When enabled, the P*V step uses the
// FP8 tensor core (E4M3) with FP32 accumulation instead of F16-PV. Default OFF
// so the stable F16-PV path stays the default and A/B is a single env flip.
// Requires the sage TU to be compiled with an nvcc frontend >= 12.4 and
// ptxas >= 8.4 (SM89 FP8 MMA). Both are satisfied by the conda `edge` CUDA 13
// toolchain used to build the sage object file.
inline bool sage_pv_fp8_enabled() {
    const char* env = std::getenv("ED_SAGE_PV_FP8");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0 &&
           std::strcmp(env, "off") != 0 &&
           std::strcmp(env, "OFF") != 0;
}

inline int sage_attn_env_int(const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    return std::atoi(env);
}

// Smooth-Q ("Thorough Outlier Smoothing", SageAttention2): subtract Q's
// per-channel mean before per-token INT8 quantization and add the rank-1
// per-key term qm·Kᵀ back into each score row. This fixes the FLUX quality
// collapse caused by DC-dominated outlier channels saturating the per-token
// amax (head_dim==128). SD3 (head_dim==64) has no such outliers, so smooth-Q
// is left OFF there by default to keep its proven no-loss path byte-identical.
//   ED_SAGE_SMOOTH_Q unset -> auto: ON for head_dim==128, OFF for head_dim==64
//   ED_SAGE_SMOOTH_Q=1      -> force ON
//   ED_SAGE_SMOOTH_Q=0      -> force OFF
inline bool sage_smooth_q_enabled(int64_t head_dim) {
    const char* env = std::getenv("ED_SAGE_SMOOTH_Q");
    if (env == nullptr || env[0] == '\0') {
        return head_dim == 128; // auto default
    }
    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0 &&
           std::strcmp(env, "off") != 0 &&
           std::strcmp(env, "OFF") != 0;
}

// Near-uniform head fallback (head_dim==128 only). The vendored SageAttention2
// INT8-QK + F16-PV CUDA kernel has a confirmed, upstream-unfixable precision bug
// at head_dim==128: on heads whose attention distribution is near-uniform, the
// PV numerator is amplified (~1.75x) and the head output collapses (cos
// 0.59-0.74 vs full precision). head_dim==64 (SD3) is unaffected. Rather than
// patch the kernel, we detect these heads at runtime by their (normalized)
// attention entropy and recompute ONLY those heads in full precision (F16-eq),
// leaving the peaked heads on the fast sage path.
//   ED_SAGE_FALLBACK unset -> auto: ON for head_dim==128, OFF for head_dim==64
//   ED_SAGE_FALLBACK=1      -> force ON
//   ED_SAGE_FALLBACK=0      -> force OFF (reproduces the raw sage output for A/B)
inline bool sage_fallback_enabled(int64_t head_dim) {
    const char* env = std::getenv("ED_SAGE_FALLBACK");
    if (env == nullptr || env[0] == '\0') {
        (void)head_dim;
        return false; // default OFF: the d128 fallback cannot fix FLUX (collapse is deeper than the PV output) and is pure overhead on non-collapsing models like Wan. Opt in with ED_SAGE_FALLBACK=1.
    }
    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0 &&
           std::strcmp(env, "off") != 0 &&
           std::strcmp(env, "OFF") != 0;
}

// Relative-L2 error threshold for flagging a head as "collapsed" under the
// INT8-PV kernel. The detection kernel measures each head's actual sampled
// rel_L2 (sage output vs a full-precision reference) and flags heads above this
// value for full-precision recompute. Default 0.25: the collapsing FLUX heads
// sit at rel_L2 0.7-1.3 (cos 0.6-0.77) while healthy heads sit below ~0.15, so
// 0.25 cleanly separates them with wide margin. Measuring the real error avoids
// mis-flagging near-uniform-but-accurate heads (see the kernel comment).
inline float sage_fallback_err_thr() {
    const char* env = std::getenv("ED_SAGE_FALLBACK_ERR");
    if (env == nullptr || env[0] == '\0') {
        return 0.25f;
    }
    return static_cast<float>(std::atof(env));
}

// Number of query rows sampled per head when estimating attention entropy in the
// detection kernel. Uniformity is a per-head property, so a modest, evenly
// spaced sample classifies the head at ~sample_rows/L_q of a full attention
// pass. Default 32.
inline int sage_fallback_sample_rows() {
    return sage_attn_env_int("ED_SAGE_UNIFORM_SAMPLES", 32);
}

// Layer-skip policy: keep the first ED_SAGE_SKIP_FIRST and last
// ED_SAGE_SKIP_LAST transformer blocks in F16 (they are most sensitive to INT8
// quantization), running sage only on the middle blocks. Returns true if the
// given layer should SKIP sage (fall back to F16 flash). When layer_idx or
// total_layers is unknown (-1), never skips -> sage applies to all layers.
//
// Defaults (when the env vars are unset) are skip_first=2, skip_last=2: this is
// the accuracy sweet spot found by sweeping SD3 (PSNR ~36 dB vs the F16 baseline
// with ~5% DiT speedup). So `ED_SAGE_ATTN=1` alone gives the best config out of
// the box. To reproduce the stage-1 "quantize every layer" behavior for A/B
// comparison, set `ED_SAGE_SKIP_FIRST=0 ED_SAGE_SKIP_LAST=0`.
inline bool sage_attn_skip_layer(int layer_idx, int total_layers) {
    if (layer_idx < 0 || total_layers <= 0) {
        return false;
    }
    const int skip_first = sage_attn_env_int("ED_SAGE_SKIP_FIRST", 2);
    const int skip_last  = sage_attn_env_int("ED_SAGE_SKIP_LAST", 2);
    if (layer_idx < skip_first) {
        return true;
    }
    if (layer_idx >= total_layers - skip_last) {
        return true;
    }
    return false;
}

inline void* sage_attn_params_to_userdata(int64_t n_head) {
    uintptr_t packed = static_cast<uintptr_t>(kSageAttnCustomMagic);
    packed |= (static_cast<uintptr_t>(n_head) << 32);
    return reinterpret_cast<void*>(packed);
}

inline SageAttnCustomParams sage_attn_params_from_userdata(void* userdata) {
    SageAttnCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic  = static_cast<uint32_t>(packed & 0xffffffffu);
    params.n_head = static_cast<int32_t>((packed >> 32) & 0x7fffffffu);
    return params;
}

inline bool sage_attn_params_valid(const SageAttnCustomParams& params) {
    return params.magic == kSageAttnCustomMagic && params.n_head > 0;
}

// Shape contract at the injection point (build_kqv in ggml_extend.hpp):
//   q_in : [d_head, L_q, n_head * N]   (contiguous, F32 or F16, head-major)
//   k_in : [d_head, L_k, n_head * N]   (contiguous, F32 or F16, head-major)
//   v_in : [d_head, n_head, L_k, N]    (4D, token-major: ne[1]=n_head, ne[2]=L_k)
//          OR [d_head, L_k, n_head]    (3D packed, ne[1]=L_k, ne[2]=n_head, N==1)
// We only support the SD3 slice: head_dim == 64, no GQA (n_kv_head ==
// n_head), no mask. Everything else must fall back to the original path.
inline bool sage_attn_v_is_4d(const ggml_tensor* v, int64_t d_head, int64_t n_head, int64_t L_k, int64_t N) {
    return v->ne[0] == d_head && v->ne[1] == n_head && v->ne[2] == L_k && v->ne[3] == N;
}

inline bool sage_attn_v_is_3d_packed(const ggml_tensor* v, int64_t d_head, int64_t n_head, int64_t L_k, int64_t N) {
    // 3D packed layout used by the MMDiT fused pair-pack path: [d_head, L_k, n_head], N==1
    return N == 1 && v->ne[0] == d_head && v->ne[1] == L_k && v->ne[2] == n_head && v->ne[3] == 1;
}

inline bool sage_attn_shape_supported(const ggml_tensor* q,
                                      const ggml_tensor* k,
                                      const ggml_tensor* v,
                                      int64_t n_head) {
    if (q == nullptr || k == nullptr || v == nullptr || n_head <= 0) {
        return false;
    }
    // Only F16/F32 inputs are handled by the quantizer.
    auto type_ok = [](const ggml_tensor* t) {
        return t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32;
    };
    if (!type_ok(q) || !type_ok(k) || !type_ok(v)) {
        return false;
    }
    const int64_t d_head = q->ne[0];
    if (d_head != 64 && d_head != 128) {
        return false;
    }
    const int64_t L_q = q->ne[1];
    const int64_t L_k = k->ne[1];
    if (L_q <= 0 || L_k <= 0) {
        return false;
    }
    // q/k are head-major 3D: ne[2] == n_head * N
    if (q->ne[2] % n_head != 0 || k->ne[2] % n_head != 0) {
        return false;
    }
    const int64_t N = q->ne[2] / n_head;
    if (N <= 0 || k->ne[2] != n_head * N) {
        return false;
    }
    if (k->ne[0] != d_head) {
        return false;
    }
    // v is either 4D token-major [d_head, n_head, L_k, N] or 3D packed
    // [d_head, L_k, n_head] (N==1). No GQA support.
    if (!sage_attn_v_is_4d(v, d_head, n_head, L_k, N) &&
        !sage_attn_v_is_3d_packed(v, d_head, n_head, L_k, N)) {
        return false;
    }
    return true;
}

// A no-op CPU custom-op body. The sage attention op is CUDA-only: on CUDA the
// GGML_OP_CUSTOM dispatch routes to the fused kernel, and ggml_backend_supports_op
// returns false on every other backend so this CPU stub is never invoked. It
// exists only because ggml_custom_4d requires a function pointer.
inline void sage_attn_cpu_custom_op_stub(ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)dst; (void)ith; (void)nth; (void)userdata;
}

// Build the GGML_OP_CUSTOM node for the fused INT8-QK + F16-PV attention.
//   q, k : [d_head, L, n_head*N]  (head-major, F16 or F32)
//   v    : [d_head, n_head, L_k, N]  (4D token-major)
// Output: F32 [d_head, n_head*N, L_q, 1] -- byte-identical to the tensor
// produced by ggml_flash_attn_ext, so the downstream view/cont/reshape in
// ggml_ext_attention_ext is unchanged.
inline ggml_tensor* sage_attn_custom(ggml_context* ctx,
                                     ggml_tensor* q,
                                     ggml_tensor* k,
                                     ggml_tensor* v,
                                     int64_t n_head) {
    if (!sage_attn_shape_supported(q, k, v, n_head)) {
        if (std::getenv("ED_SAGE_ATTN_DEBUG")) {
            fprintf(stderr, "[sage] shape NOT supported: q=[%lld,%lld,%lld,%lld] k=[%lld,%lld,%lld,%lld] v=[%lld,%lld,%lld,%lld] n_head=%lld types q/k/v=%d/%d/%d\n",
                    (long long)(q?q->ne[0]:-1),(long long)(q?q->ne[1]:-1),(long long)(q?q->ne[2]:-1),(long long)(q?q->ne[3]:-1),
                    (long long)(k?k->ne[0]:-1),(long long)(k?k->ne[1]:-1),(long long)(k?k->ne[2]:-1),(long long)(k?k->ne[3]:-1),
                    (long long)(v?v->ne[0]:-1),(long long)(v?v->ne[1]:-1),(long long)(v?v->ne[2]:-1),(long long)(v?v->ne[3]:-1),
                    (long long)n_head, q?(int)q->type:-1, k?(int)k->type:-1, v?(int)v->type:-1);
        }
        return nullptr;
    }
    if (std::getenv("ED_SAGE_ATTN_DEBUG")) {
        fprintf(stderr, "[sage] shape SUPPORTED, building custom node: d_head=%lld L_q=%lld L_k=%lld HN=%lld n_head=%lld\n",
                (long long)q->ne[0], (long long)q->ne[1], (long long)k->ne[1], (long long)q->ne[2], (long long)n_head);
    }
    const int64_t d_head = q->ne[0];
    const int64_t L_q    = q->ne[1];
    const int64_t HN     = q->ne[2];
    ggml_tensor* args[] = { q, k, v };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      d_head,
                                      HN,
                                      L_q,
                                      1,
                                      args,
                                      3,
                                      sage_attn_cpu_custom_op_stub,
                                      1,
                                      sage_attn_params_to_userdata(n_head));
    ggml_set_name(out, "ed_sage_attn_int8qk_f16pv");
    return out;
}

} // namespace edgedit::ggml_ext
