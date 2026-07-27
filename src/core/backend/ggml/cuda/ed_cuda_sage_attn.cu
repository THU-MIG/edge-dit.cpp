// SageAttention2-style INT8-QK + F16-PV fused attention, wired into edge-dit as
// a ggml GGML_OP_CUSTOM op. Vendored kernel math comes from thu-ml/SageAttention
// (Apache-2.0); the quantization / smooth-K kernels and the raw-pointer host
// launcher below are written for the ggml tensor ABI (no torch dependency).
//
// Stage 1 scope: head_dim == 64, no mask, no GQA, CUDA/SM89, F16-PV (float
// accumulation). INT8 QK^T runs on the IMMA tensor core; PV runs on the F16
// tensor core with FP32 accumulation.

#include "ed_cuda_sage_attn.h"

#include "backend/ggml/ed_ggml_sage_attn_ext.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "sage/sage_qk_int_sv_f16_kernel.cuh"
#include "sage/sage_qk_int_sv_f8_kernel.cuh"
#include <cuda_fp8.h>

namespace {

using edgedit::ggml_ext::SageAttnCustomParams;

// Kernel tiling constants must match the sage kernel launch (CTA_Q=128,
// CTA_K=64, WARP_Q=32, WARP_K=64).
constexpr int kCtaQ  = 128;
constexpr int kCtaK  = 64;
constexpr int kWarpQ = 32;
constexpr int kWarpK = 64;
constexpr int kHeadDim = 64;

static inline int div_ceil_i(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Quantization + smooth-K kernels
// ---------------------------------------------------------------------------
// Layout at the injection point (all head-major, d fastest):
//   Q/K global: element (d, token t, head-batch hn) at
//               base + t*stride_seq + hn*stride_h + d      (stride in elements)
//   For contiguous [d_head, L, H*N] this is stride_seq=d_head, stride_h=d_head*L.
// The quantized int8 output we produce is contiguous [d_head, L, H*N] as well,
// which the sage kernel reads with stride_seq_q=head_dim, stride_h_q=head_dim*L.

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ int8_t f2i8_rn(float x) {
    // round to nearest even, saturate to [-127,127]
    x = fmaxf(fminf(x, 127.0f), -127.0f);
    return static_cast<int8_t>(__float2int_rn(x));
}

// One block handles one WARP_Q (=32) chunk of Q tokens for a given (head-batch).
// blockIdx.x = warp-chunk index within padded L (0 .. ceil(L/128)*4 - 1),
// blockIdx.y = head-batch index (hn), blockDim.x = 32 lanes * something.
// We compute a single amax over the whole 32-token x head_dim tile -> one int8
// scale per warp chunk, matching QuantGranularity::kPerWarp with WARP_Q=32.
template <typename T, int HEAD_DIM, int WARP_Q>
__global__ void sage_quant_q_perwarp_kernel(const T* __restrict__ q,
                                            const float* __restrict__ qm, // [H*N, HEAD_DIM] per-channel mean (may be null)
                                            int8_t* __restrict__ q_int8,
                                            float* __restrict__ q_scale,
                                            int L,
                                            int num_warp_chunks,      // ceil(L/CTA_Q)*(CTA_Q/WARP_Q) per head
                                            int64_t stride_seq,       // elements
                                            int64_t stride_h) {
    const int chunk = blockIdx.x;            // warp chunk over tokens
    const int hn    = blockIdx.y;            // head-batch
    const int tid   = threadIdx.x;           // 0..blockDim.x-1
    const int nthreads = blockDim.x;

    const int tok0 = chunk * WARP_Q;
    const int64_t base = static_cast<int64_t>(hn) * stride_h;
    const float* qm_row = (qm != nullptr) ? (qm + static_cast<int64_t>(hn) * HEAD_DIM) : nullptr;

    // pass 1: local amax over this 32 x HEAD_DIM tile
    float amax = 1e-7f;
    for (int idx = tid; idx < WARP_Q * HEAD_DIM; idx += nthreads) {
        const int t = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int tok = tok0 + t;
        float val = 0.0f;
        if (tok < L) {
            val = to_float<T>(q[base + static_cast<int64_t>(tok) * stride_seq + d]);
            if (qm_row != nullptr) val -= qm_row[d];
        }
        amax = fmaxf(amax, fabsf(val));
    }
    // block reduce max
    __shared__ float sh[32];
    // warp reduce
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m, 32));
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    if (lane == 0) sh[wid] = amax;
    __syncthreads();
    float block_amax = (tid < (nthreads + 31) / 32) ? sh[lane] : 1e-7f;
    if (wid == 0) {
        for (int m = 16; m > 0; m >>= 1) block_amax = fmaxf(block_amax, __shfl_xor_sync(0xffffffff, block_amax, m, 32));
    }
    __shared__ float s_scale;
    if (tid == 0) {
        const float scale = block_amax / 127.0f;
        s_scale = scale;
        // q_scale layout per head-batch: [num_warp_chunks]; global index hn*num_warp_chunks+chunk
        q_scale[static_cast<int64_t>(hn) * num_warp_chunks + chunk] = scale;
    }
    __syncthreads();
    const float inv_scale = 1.0f / s_scale;

    // pass 2: quantize (read with input strides, write contiguous [hn][tok][d])
    for (int idx = tid; idx < WARP_Q * HEAD_DIM; idx += nthreads) {
        const int t = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int tok = tok0 + t;
        if (tok < L) {
            const int64_t in_off = base + static_cast<int64_t>(tok) * stride_seq + d;
            const int64_t out_off = (static_cast<int64_t>(hn) * L + tok) * HEAD_DIM + d;
            float val = to_float<T>(q[in_off]);
            if (qm_row != nullptr) val -= qm_row[d];
            q_int8[out_off] = f2i8_rn(val * inv_scale);
        }
    }
}

// K quantization: per-block (BLKK=64 tokens) int8 scale, with optional smooth-K
// (subtract per-channel mean km[hn][d] computed over all K tokens).
// One block handles one 64-token block for a (head-batch).
template <typename T, int HEAD_DIM, int BLK_K>
__global__ void sage_quant_k_perblock_smoothk_kernel(const T* __restrict__ k,
                                                      const float* __restrict__ km, // [H*N, HEAD_DIM] per-channel mean (may be null)
                                                      int8_t* __restrict__ k_int8,
                                                      float* __restrict__ k_scale,
                                                      int L,
                                                      int num_blocks,           // ceil(L/BLK_K) per head
                                                      int64_t stride_seq,
                                                      int64_t stride_h) {
    const int blk = blockIdx.x;
    const int hn  = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int tok0 = blk * BLK_K;
    const int64_t base = static_cast<int64_t>(hn) * stride_h;
    const float* km_row = (km != nullptr) ? (km + static_cast<int64_t>(hn) * HEAD_DIM) : nullptr;

    float amax = 1e-7f;
    for (int idx = tid; idx < BLK_K * HEAD_DIM; idx += nthreads) {
        const int t = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int tok = tok0 + t;
        float val = 0.0f;
        if (tok < L) {
            val = to_float<T>(k[base + static_cast<int64_t>(tok) * stride_seq + d]);
            if (km_row != nullptr) val -= km_row[d];
        }
        amax = fmaxf(amax, fabsf(val));
    }
    __shared__ float sh[32];
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m, 32));
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    if (lane == 0) sh[wid] = amax;
    __syncthreads();
    float block_amax = (tid < (nthreads + 31) / 32) ? sh[lane] : 1e-7f;
    if (wid == 0) {
        for (int m = 16; m > 0; m >>= 1) block_amax = fmaxf(block_amax, __shfl_xor_sync(0xffffffff, block_amax, m, 32));
    }
    __shared__ float s_scale;
    if (tid == 0) {
        const float scale = block_amax / 127.0f;
        s_scale = scale;
        k_scale[static_cast<int64_t>(hn) * num_blocks + blk] = scale;
    }
    __syncthreads();
    const float inv_scale = 1.0f / s_scale;

    for (int idx = tid; idx < BLK_K * HEAD_DIM; idx += nthreads) {
        const int t = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int tok = tok0 + t;
        if (tok < L) {
            const int64_t in_off = base + static_cast<int64_t>(tok) * stride_seq + d;
            const int64_t out_off = (static_cast<int64_t>(hn) * L + tok) * HEAD_DIM + d;
            float val = to_float<T>(k[in_off]);
            if (km_row != nullptr) val -= km_row[d];
            k_int8[out_off] = f2i8_rn(val * inv_scale);
        }
    }
}

// Per-channel mean of K over the token dimension, per (head-batch).
// grid.x = head-batch, blockDim.x = HEAD_DIM threads (one per channel).
template <typename T, int HEAD_DIM>
__global__ void sage_k_mean_kernel(const T* __restrict__ k,
                                   float* __restrict__ km, // [H*N, HEAD_DIM]
                                   int L,
                                   int64_t stride_seq,
                                   int64_t stride_h) {
    const int hn = blockIdx.x;
    const int d  = threadIdx.x;
    if (d >= HEAD_DIM) return;
    const int64_t base = static_cast<int64_t>(hn) * stride_h + d;
    float sum = 0.0f;
    for (int t = 0; t < L; ++t) {
        sum += to_float<T>(k[base + static_cast<int64_t>(t) * stride_seq]);
    }
    km[static_cast<int64_t>(hn) * HEAD_DIM + d] = sum / static_cast<float>(L);
}

// Per-channel mean of Q over the token dimension, per (head-batch). Same layout
// as sage_k_mean_kernel. Used by smooth-Q ("Thorough Outlier Smoothing"): the
// mean is subtracted from Q before per-token INT8 quantization so DC-dominated
// outlier channels no longer saturate the per-token amax.
template <typename T, int HEAD_DIM>
__global__ void sage_q_mean_kernel(const T* __restrict__ q,
                                   float* __restrict__ qm, // [H*N, HEAD_DIM]
                                   int L,
                                   int64_t stride_seq,
                                   int64_t stride_h) {
    const int hn = blockIdx.x;
    const int d  = threadIdx.x;
    if (d >= HEAD_DIM) return;
    const int64_t base = static_cast<int64_t>(hn) * stride_h + d;
    float sum = 0.0f;
    for (int t = 0; t < L; ++t) {
        sum += to_float<T>(q[base + static_cast<int64_t>(t) * stride_seq]);
    }
    qm[static_cast<int64_t>(hn) * HEAD_DIM + d] = sum / static_cast<float>(L);
}

// Rank-1 smooth-Q compensation bias c[hn][j] = sum_d qm[hn][d] * K_orig[hn][j][d].
// This is the per-key term the smooth-Q expansion removes and must be added back
// to each score row before softmax (see add_smooth_q_bias in attn_utils.cuh).
// K is read from the ORIGINAL (un-smoothed) global K with its input strides.
// One block per (key j, head-batch hn); blockDim.x = HEAD_DIM lanes, reduced to
// one bias value. Output layout: [H*N, L_k] contiguous (row stride L_k).
template <typename T, int HEAD_DIM>
__global__ void sage_smooth_q_bias_kernel(const T* __restrict__ k,
                                          const float* __restrict__ qm, // [H*N, HEAD_DIM]
                                          float* __restrict__ bias,     // [H*N, L_k]
                                          int L_k,
                                          int64_t stride_seq,
                                          int64_t stride_h) {
    const int tok = blockIdx.x;  // key index
    const int hn  = blockIdx.y;  // head-batch
    const int d   = threadIdx.x; // 0..HEAD_DIM-1 (HEAD_DIM is 64 or 128 -> 2 or 4 warps)

    const int64_t base = static_cast<int64_t>(hn) * stride_h + static_cast<int64_t>(tok) * stride_seq;
    const float* qm_row = qm + static_cast<int64_t>(hn) * HEAD_DIM;

    float partial = to_float<T>(k[base + d]) * qm_row[d];
    // reduce across the 32 lanes of this warp
    for (int m = 16; m > 0; m >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, m, 32);
    // HEAD_DIM > 32: combine the per-warp partials via shared memory.
    constexpr int num_warps_row = HEAD_DIM / 32; // 2 (d=64) or 4 (d=128)
    __shared__ float sh[num_warps_row];
    const int warp_in_row = d >> 5;
    if ((d & 31) == 0) sh[warp_in_row] = partial;
    __syncthreads();
    if (d == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < num_warps_row; ++w) total += sh[w];
        bias[static_cast<int64_t>(hn) * L_k + tok] = total;
    }
}

// Copy V (4D token-major [d_head, n_head, L_k, N]) into a contiguous
// [d_head, L_k, H*N] half buffer that the sage kernel reads with
// stride_seq_v = head_dim, stride_h_v = head_dim * L_k.
// hn index convention: hn = h + n_head * b  (matches q/k head-batch order).
template <typename T, int HEAD_DIM>
__global__ void sage_v_to_f16_kernel(const T* __restrict__ v,
                                     half* __restrict__ v_f16,
                                     int L, int n_head, int N,
                                     int64_t v_s0, int64_t v_s1, int64_t v_s2, int64_t v_s3) {
    // total = HEAD_DIM * L * n_head * N
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(HEAD_DIM) * L * n_head * N;
    if (idx >= total) return;
    const int d = idx % HEAD_DIM; idx /= HEAD_DIM;
    const int t = idx % L;        idx /= L;
    const int h = idx % n_head;   idx /= n_head;
    const int b = idx;            // batch
    // source: v[d*v_s0 + h*v_s1 + t*v_s2 + b*v_s3]  (v ne = [d_head,n_head,L,N])
    const float val = to_float<T>(v[static_cast<int64_t>(d) * v_s0 +
                                    static_cast<int64_t>(h) * v_s1 +
                                    static_cast<int64_t>(t) * v_s2 +
                                    static_cast<int64_t>(b) * v_s3]);
    const int hn = h + n_head * b;
    const int64_t off = static_cast<int64_t>(hn) * HEAD_DIM * L +
                        static_cast<int64_t>(t) * HEAD_DIM + d;
    v_f16[off] = __float2half_rn(val);
}

// Convert the sage kernel's half output [d_head, L_q, H*N] (contiguous,
// stride_seq_o=head_dim, stride_h_o=head_dim*L_q) into the F32 destination that
// matches ggml_flash_attn_ext: dst is [d_head, H*N, L_q, 1] with
// dst[d + hn*head_dim + t*head_dim*(H*N)].
template <int HEAD_DIM>
__global__ void sage_o_half_to_f32_kernel(const half* __restrict__ o_half,
                                          float* __restrict__ dst,
                                          int L_q, int HN) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(HEAD_DIM) * L_q * HN;
    if (idx >= total) return;
    const int d  = idx % HEAD_DIM; idx /= HEAD_DIM;
    const int t  = idx % L_q;      idx /= L_q;
    const int hn = idx;
    // o_half layout: [hn][t][d]
    const int64_t src_off = static_cast<int64_t>(hn) * HEAD_DIM * L_q +
                            static_cast<int64_t>(t) * HEAD_DIM + d;
    // dst layout (flash_attn_ext): ne=[d_head, HN, L_q, 1]
    //   dst[d + hn*head_dim + t*head_dim*HN]
    const int64_t dst_off = static_cast<int64_t>(d) +
                            static_cast<int64_t>(hn) * HEAD_DIM +
                            static_cast<int64_t>(t) * HEAD_DIM * HN;
    dst[dst_off] = __half2float(o_half[src_off]);
}

// Diagnostic: full-precision reference attention compared against the sage dst,
// measured PER head-batch. Accumulates, per hn, sum of squared error, sum of
// squared reference, sum of squared sage output, and dot(ref,sage) so the host
// can print per-head relative L2 error and cosine. Reads ORIGINAL q/k/v
// (pre-quant) with their semantic strides. Gated by ED_SAGE_SELFCHECK.
template <typename T, int HEAD_DIM>
__global__ void sage_selfcheck_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                      const T* __restrict__ v, const float* __restrict__ dst,
                                      int L_q, int L_k, int HN,
                                      int64_t q_s_seq, int64_t q_s_h,
                                      int64_t k_s_seq, int64_t k_s_h,
                                      int64_t v_sd, int64_t v_sh, int64_t v_st, int64_t v_sb,
                                      int n_head, float sm_scale, double* __restrict__ acc) {
    const int hn = blockIdx.y;     // head-batch
    const int tq = blockIdx.x;     // query row
    const int d  = threadIdx.x;    // 0..HEAD_DIM-1
    if (tq >= L_q) return;

    __shared__ float qrow[HEAD_DIM];
    qrow[d] = to_float<T>(q[static_cast<int64_t>(hn) * q_s_h + static_cast<int64_t>(tq) * q_s_seq + d]);
    __syncthreads();

    // online softmax over keys, accumulating O[d] for this lane's channel
    const int h = hn % n_head;
    const int b = hn / n_head;
    float m = -1e30f, denom = 0.0f, o_acc = 0.0f;
    for (int j = 0; j < L_k; ++j) {
        // score = sum_d qrow[d]*K[j,d]
        float partial = qrow[d] * to_float<T>(k[static_cast<int64_t>(hn) * k_s_h + static_cast<int64_t>(j) * k_s_seq + d]);
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off, 32);
        __shared__ float sh[HEAD_DIM / 32];
        if ((d & 31) == 0) sh[d >> 5] = partial;
        __syncthreads();
        float score = 0.0f;
#pragma unroll
        for (int w = 0; w < HEAD_DIM / 32; ++w) score += sh[w];
        score *= sm_scale;
        // v value for this channel: v[d, h, j, b]
        const float vval = to_float<T>(v[static_cast<int64_t>(d) * v_sd + static_cast<int64_t>(h) * v_sh +
                                         static_cast<int64_t>(j) * v_st + static_cast<int64_t>(b) * v_sb]);
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float p = __expf(score - m_new);
        denom = denom * corr + p;
        o_acc = o_acc * corr + p * vval;
        m = m_new;
        __syncthreads();
    }
    const float o_ref = o_acc / denom;
    const int64_t dst_off = static_cast<int64_t>(d) + static_cast<int64_t>(hn) * HEAD_DIM +
                            static_cast<int64_t>(tq) * HEAD_DIM * HN;
    const float o_sage = dst[dst_off];
    const float diff = o_ref - o_sage;
    atomicAdd(&acc[2 * hn + 0], (double)(diff * diff));
    atomicAdd(&acc[2 * hn + 1], (double)(o_ref * o_ref));
    atomicAdd(&acc[2 * HN + hn], (double)(o_sage * o_sage));   // sage magnitude^2
    atomicAdd(&acc[3 * HN + hn], (double)(o_ref * o_sage));    // dot(ref,sage) for cos
}


// ---------------------------------------------------------------------------
// Near-uniform head fallback (head_dim==128 precision bug mitigation)
// ---------------------------------------------------------------------------
// The vendored SageAttention2 INT8-QK + F16-PV kernel amplifies the PV numerator
// (~1.75x) on heads whose attention is near-uniform at head_dim==128, collapsing
// their output (cos 0.59-0.74). We detect those heads by their normalized
// attention entropy and recompute ONLY them in full precision, leaving peaked
// heads on the fast sage path. head_dim==64 (SD3) has no such bug and is never
// touched (the host gates this on HEAD_DIM==128).

// Detection (warp-per-row). Measures, per head-batch, on a set of sampled query
// rows, BOTH
//   (a) the head's mean normalized attention entropy H/ln(L_k)  -- a "how
//       near-uniform is this head" diagnostic (task-2 uniform-ratio metric), and
//   (b) the ACTUAL relative-L2 error of the already-computed sage output vs a
//       full-precision reference on those rows.
// The head is flagged for full-precision recompute when its sampled rel_L2
// exceeds `err_thr`. Measuring the real error (b) -- rather than trusting the
// entropy proxy (a) -- is essential: at head_dim==128 there exist near-uniform
// heads (entropy ~0.999) whose sage output is nonetheless accurate, while the
// collapsing heads are a specific subset.
//
// Performance: ONE WARP handles one (head, sampled-row) pair. Each of the 32
// lanes owns HEAD_DIM/32 (=4 at d128) channels; the QK dot uses a single
// warp-shuffle reduction per key with NO __syncthreads and NO shared memory, so
// the whole grid is HN*n_sampled warps -- enough to fill the GPU (the earlier
// one-block-per-head version left most SMs idle and was ~8x slower than sage).
// Results are accumulated per head via atomics into acc[hn] = {sse, ref2,
// ent_sum, n}; a tiny finalize kernel turns them into flags.
template <typename T, int HEAD_DIM>
__global__ void sage_detect_uniform_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                           const T* __restrict__ v, const float* __restrict__ dst,
                                           int L_q, int L_k, int HN,
                                           int64_t q_s_seq, int64_t q_s_h,
                                           int64_t k_s_seq, int64_t k_s_h,
                                           int64_t v_sd, int64_t v_sh, int64_t v_st, int64_t v_sb,
                                           int n_head, float sm_scale, int n_sampled, int row_stride,
                                           float* __restrict__ acc) {
    constexpr int CPL = HEAD_DIM / 32;   // channels per lane (4 at d128)
    const int warp_gid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = HN * n_sampled;
    if (warp_gid >= total_warps) return;
    const int hn = warp_gid / n_sampled;
    const int si = warp_gid % n_sampled;
    const int tq = si * row_stride;
    if (tq >= L_q) return;
    const int h = hn % n_head;
    const int b = hn / n_head;
    const float ln_Lk = (L_k > 1) ? logf((float)L_k) : 1.0f;

    // load this lane's channels of the query row
    float qreg[CPL];
    const int64_t q_base = static_cast<int64_t>(hn) * q_s_h + static_cast<int64_t>(tq) * q_s_seq;
#pragma unroll
    for (int c = 0; c < CPL; ++c) qreg[c] = to_float<T>(q[q_base + lane + c * 32]);

    // online softmax over keys: running max m, normalizer Z, entropy term Tt,
    // and this lane's partial output channels o_acc[c].
    float m = -1e30f, Z = 0.0f, Tt = 0.0f, o_acc[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) o_acc[c] = 0.0f;
    const int64_t k_base = static_cast<int64_t>(hn) * k_s_h;
    for (int j = 0; j < L_k; ++j) {
        float partial = 0.0f;
        const int64_t koff = k_base + static_cast<int64_t>(j) * k_s_seq;
#pragma unroll
        for (int c = 0; c < CPL; ++c) partial += qreg[c] * to_float<T>(k[koff + lane + c * 32]);
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off, 32);
        const float score = partial * sm_scale;   // identical on all lanes
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float e = __expf(score - m_new);
        Z  = Z * corr + e;
        Tt = Tt * corr + e * score;
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            const float vval = to_float<T>(v[static_cast<int64_t>(lane + c * 32) * v_sd +
                                             static_cast<int64_t>(h) * v_sh +
                                             static_cast<int64_t>(j) * v_st +
                                             static_cast<int64_t>(b) * v_sb]);
            o_acc[c] = o_acc[c] * corr + e * vval;
        }
        m = m_new;
    }
    // per-lane channel error vs the already-written sage output
    const int64_t dst_base = static_cast<int64_t>(hn) * HEAD_DIM + static_cast<int64_t>(tq) * HEAD_DIM * HN;
    float sd = 0.0f, sr = 0.0f;
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
        const float o_ref = o_acc[c] / Z;
        const float o_sage = dst[dst_base + lane + c * 32];
        const float diff = o_ref - o_sage;
        sd += diff * diff;
        sr += o_ref * o_ref;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sd += __shfl_xor_sync(0xffffffff, sd, off, 32);
        sr += __shfl_xor_sync(0xffffffff, sr, off, 32);
    }
    if (lane == 0) {
        const float H = m + logf(Z) - Tt / Z;
        const float norm_ent = (L_k > 1) ? (H / ln_Lk) : 0.0f;
        atomicAdd(&acc[4 * hn + 0], sd);
        atomicAdd(&acc[4 * hn + 1], sr);
        atomicAdd(&acc[4 * hn + 2], norm_ent);
        atomicAdd(&acc[4 * hn + 3], 1.0f);
    }
}

// Finalize: turn per-head accumulators into a flag + reported entropy.
__global__ void sage_detect_finalize_kernel(const float* __restrict__ acc, int HN, float err_thr,
                                            int* __restrict__ uniform_flag, float* __restrict__ head_entropy) {
    const int hn = blockIdx.x * blockDim.x + threadIdx.x;
    if (hn >= HN) return;
    const float sse  = acc[4 * hn + 0];
    const float ref2 = acc[4 * hn + 1];
    const float esum = acc[4 * hn + 2];
    const float n    = acc[4 * hn + 3];
    const float rel  = (ref2 > 0.0f) ? sqrtf(sse / ref2) : 0.0f;
    head_entropy[hn] = (n > 0.0f) ? (esum / n) : 0.0f;
    uniform_flag[hn] = (rel > err_thr) ? 1 : 0;
}

// Overwrite (warp-per-row): for each flagged head, recompute its output in full
// precision (same online-softmax math) and write it into dst, replacing the
// collapsed sage output. Peaked heads early-exit so they keep the fast sage
// result untouched. ONE WARP per (query row, head); 32 lanes each own HEAD_DIM/32
// output channels. No shared memory, no per-key __syncthreads.
template <typename T, int HEAD_DIM>
__global__ void sage_fallback_overwrite_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                               const T* __restrict__ v, float* __restrict__ dst,
                                               int L_q, int L_k, int HN,
                                               int64_t q_s_seq, int64_t q_s_h,
                                               int64_t k_s_seq, int64_t k_s_h,
                                               int64_t v_sd, int64_t v_sh, int64_t v_st, int64_t v_sb,
                                               int n_head, float sm_scale,
                                               const int* __restrict__ uniform_flag) {
    constexpr int CPL = HEAD_DIM / 32;
    const int hn = blockIdx.y;
    if (uniform_flag[hn] == 0) return;   // peaked head: keep the sage output
    const int warps_per_block = blockDim.x >> 5;
    const int tq = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (tq >= L_q) return;
    const int lane = threadIdx.x & 31;
    const int h = hn % n_head;
    const int b = hn / n_head;

    float qreg[CPL];
    const int64_t q_base = static_cast<int64_t>(hn) * q_s_h + static_cast<int64_t>(tq) * q_s_seq;
#pragma unroll
    for (int c = 0; c < CPL; ++c) qreg[c] = to_float<T>(q[q_base + lane + c * 32]);

    float m = -1e30f, denom = 0.0f, o_acc[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) o_acc[c] = 0.0f;
    const int64_t k_base = static_cast<int64_t>(hn) * k_s_h;
    for (int j = 0; j < L_k; ++j) {
        float partial = 0.0f;
        const int64_t koff = k_base + static_cast<int64_t>(j) * k_s_seq;
#pragma unroll
        for (int c = 0; c < CPL; ++c) partial += qreg[c] * to_float<T>(k[koff + lane + c * 32]);
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off, 32);
        const float score = partial * sm_scale;
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float p = __expf(score - m_new);
        denom = denom * corr + p;
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            const float vval = to_float<T>(v[static_cast<int64_t>(lane + c * 32) * v_sd +
                                             static_cast<int64_t>(h) * v_sh +
                                             static_cast<int64_t>(j) * v_st +
                                             static_cast<int64_t>(b) * v_sb]);
            o_acc[c] = o_acc[c] * corr + p * vval;
        }
        m = m_new;
    }
    const int64_t dst_base = static_cast<int64_t>(hn) * HEAD_DIM + static_cast<int64_t>(tq) * HEAD_DIM * HN;
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
        dst[dst_base + lane + c * 32] = o_acc[c] / denom;
    }
}


// ---------------------------------------------------------------------------
// Stage-2 Part A: FP8-E4M3 PV support
// ---------------------------------------------------------------------------
// The FP8 sage kernel reads V transposed: layout [hn][head_dim][L_k_pad] with
// the token axis innermost (contiguous) and padded up to a multiple of CTA_K.
// The seq axis is permuted inside every block of 16 tokens by the bit-permute
//   out = perm(src):  src bits (b3,b2,b1,b0) -> (b2,b1,b3,b0)
//   perm = 0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15
// to match the fp8-mma ldmatrix layout (see SageAttention TransposePadPermute).
// V is quantized per-channel (one e4m3 scale per (hn,d)) as amax(|v|)/scale_max
// with scale_max=448 (E4M3 max). The kernel multiplies the per-channel v_scale
// back at the end via fuse_v_scale. We do NOT subtract the V mean (fuse_v_mean
// off) to keep the pipeline simple and avoid an extra reduction pass.

// Inverse of the block-16 seq permute: given an output token residue r in
// [0,16), returns the source residue s such that perm(s) == r.
// perm maps bits (b3,b2,b1,b0)->(b2,b1,b3,b0); the inverse maps
// (r3,r2,r1,r0)->(r1,r3,r2,r0).
__device__ __forceinline__ int seq_permute_16_inv(int r) {
    return 8 * ((r / 2) % 2) + 4 * ((r / 8) % 2) + 2 * ((r / 4) % 2) + (r % 2);
}

// Pass 1: per-channel amax over tokens -> v_scale[hn][d] = amax/scale_max.
// grid.x = head-batch (hn), block.x = HEAD_DIM threads (one per channel d).
template <typename T, int HEAD_DIM>
__global__ void sage_v_channel_scale_kernel(const T* __restrict__ v,
                                            float* __restrict__ v_scale,
                                            int L, int n_head, int N,
                                            int64_t v_sd, int64_t v_sh, int64_t v_st, int64_t v_sb,
                                            float scale_max) {
    const int hn = blockIdx.x;      // hn = h + n_head * b
    const int d  = threadIdx.x;
    if (d >= HEAD_DIM) return;
    const int h = hn % n_head;
    const int b = hn / n_head;
    const int64_t base = static_cast<int64_t>(d) * v_sd +
                         static_cast<int64_t>(h) * v_sh +
                         static_cast<int64_t>(b) * v_sb;
    float amax = 1e-8f;
    for (int t = 0; t < L; ++t) {
        const float val = to_float<T>(v[base + static_cast<int64_t>(t) * v_st]);
        amax = fmaxf(amax, fabsf(val));
    }
    v_scale[static_cast<int64_t>(hn) * HEAD_DIM + d] = amax / scale_max;
}

// Pass 2: quantize + transpose + seq-permute into e4m3.
// Output layout: v_fp8[hn][d][t_pad], token innermost, t_pad in [0, L_pad).
// One thread per (hn, d, t_pad).
template <typename T, int HEAD_DIM>
__global__ void sage_v_to_fp8_transposed_kernel(const T* __restrict__ v,
                                                 int8_t* __restrict__ v_fp8, // e4m3 bytes
                                                 const float* __restrict__ v_scale,
                                                 int L, int L_pad, int n_head, int N,
                                                 int64_t v_sd, int64_t v_sh, int64_t v_st, int64_t v_sb) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(HEAD_DIM) * L_pad * n_head * N;
    if (idx >= total) return;
    // decode idx as [hn][d][t_pad]  (t_pad fastest)
    const int t_pad = idx % L_pad; idx /= L_pad;
    const int d     = idx % HEAD_DIM; idx /= HEAD_DIM;
    const int hn    = idx; // 0..HN-1
    const int h = hn % n_head;
    const int b = hn / n_head;

    // output token t_pad receives source token whose perm() maps to t_pad.
    const int blk_base = (t_pad / 16) * 16;
    const int t_src    = blk_base + seq_permute_16_inv(t_pad % 16);

    float q = 0.0f;
    if (t_src < L) {
        const int64_t in_off = static_cast<int64_t>(d) * v_sd +
                               static_cast<int64_t>(h) * v_sh +
                               static_cast<int64_t>(t_src) * v_st +
                               static_cast<int64_t>(b) * v_sb;
        const float val = to_float<T>(v[in_off]);
        const float sc  = v_scale[static_cast<int64_t>(hn) * HEAD_DIM + d];
        q = (sc > 0.0f) ? (val / sc) : 0.0f;
    }
    const int64_t out_off = (static_cast<int64_t>(hn) * HEAD_DIM + d) * static_cast<int64_t>(L_pad) + t_pad;
    __nv_fp8_e4m3 e = __nv_fp8_e4m3(q);
    v_fp8[out_off] = *reinterpret_cast<const int8_t*>(&e);
}


// ---------------------------------------------------------------------------
// Op param decode helpers (mirror ed_cuda_attention_v_prep.cu)
// ---------------------------------------------------------------------------
struct ggml_custom_op_params_local {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
};

static SageAttnCustomParams decode_params(const ggml_tensor* dst) {
    ggml_custom_op_params_local op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    std::memcpy(&op_params, dst->op_params, sizeof(op_params));
    return edgedit::ggml_ext::sage_attn_params_from_userdata(op_params.userdata);
}

static inline int64_t elem_stride(const ggml_tensor* t, int dim) {
    return t->nb[dim] / ggml_type_size(t->type);
}

// Compute the flattened dims used for scratch sizing.
struct sage_dims {
    int L_q, L_k, HN, N, n_head;
    int num_warp_chunks; // per head-batch, for q_scale
    int num_k_blocks;    // per head-batch, for k_scale
};

static bool compute_dims(const ggml_tensor* dst, int n_head, sage_dims* out) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    if (q == nullptr || k == nullptr) return false;
    out->L_q = (int)q->ne[1];
    out->L_k = (int)k->ne[1];
    out->HN  = (int)q->ne[2];
    out->n_head = n_head;
    out->N   = out->HN / n_head;
    out->num_warp_chunks = div_ceil_i(out->L_q, kCtaQ) * (kCtaQ / kWarpQ);
    out->num_k_blocks    = div_ceil_i(out->L_k, kCtaK);
    return true;
}


// Launch the vendored sage kernel over a contiguous int8 Q/K + f16 V producing
// f16 O. All buffers are contiguous head-major (see layout comments above).
// When smooth_q is true, qk_bias ([HN, L_k], row-major) supplies the rank-1
// per-key compensation term added back into each score row before softmax.
template <int HEAD_DIM, bool SMOOTH_Q>
static void launch_sage_kernel(const int8_t* q_int8, const int8_t* k_int8, const half* v_f16,
                               half* o_half, const float* q_scale, const float* k_scale,
                               int L_q, int L_k, int HN,
                               float sm_scale, cudaStream_t stream,
                               const float* qk_bias = nullptr) {
    // strides (in elements) for contiguous head-major buffers
    const uint32_t stride_h_q = static_cast<uint32_t>(HEAD_DIM) * L_q;
    const uint32_t stride_seq_q = HEAD_DIM;
    const uint32_t stride_h_k = static_cast<uint32_t>(HEAD_DIM) * L_k;
    const uint32_t stride_seq_k = HEAD_DIM;
    const uint32_t stride_h_v = static_cast<uint32_t>(HEAD_DIM) * L_k;
    const uint32_t stride_seq_v = HEAD_DIM;
    const uint32_t stride_h_o = static_cast<uint32_t>(HEAD_DIM) * L_q;
    const uint32_t stride_seq_o = HEAD_DIM;

    // batch stride collapses all head-batches into gridDim.y; we launch with
    // batch_size = HN and num_qo_heads = 1 so the kernel's
    // (batch_id, head_id) both index into our flattened hn axis via head stride.
    // Simpler: treat gridDim.z = HN (batch), gridDim.y = 1 head. Then
    // stride_bz_* must advance one head-batch, stride_h_* = 0.
    const uint32_t stride_bz_q = stride_h_q;
    const uint32_t stride_bz_k = stride_h_k;
    const uint32_t stride_bz_v = stride_h_v;
    const uint32_t stride_bz_o = stride_h_o;
    // qk_bias is [HN, L_k]: one row of L_k values per head-batch.
    const uint32_t stride_bz_bias = static_cast<uint32_t>(L_k);

    constexpr int num_warps_q = kCtaQ / kWarpQ; // 4
    constexpr int num_warps_k = kCtaK / kWarpK; // 1
    constexpr int num_warps = num_warps_q * num_warps_k; // 4

    auto kernel_func = qk_int_sv_f16_attn_kernel<
        kCtaQ, kCtaK, kWarpQ, kWarpK, HEAD_DIM,
        DataType::kInt8,
        QuantGranularity::kPerWarp, QuantGranularity::kPerBlock,
        /*DTypeSVAccum=*/float, /*use_inst_buffer=*/false, /*DTypeOut=*/half,
        ComputeUnit::kTensorCore, MaskMode::kNone, /*return_lse=*/false,
        /*fuse_v_mean=*/false, /*smooth_q=*/SMOOTH_Q>;

    size_t smem_max = std::max(
        (size_t)kCtaQ * HEAD_DIM * sizeof(int8_t) + (size_t)kCtaK * HEAD_DIM * sizeof(int8_t) + (size_t)kCtaK * HEAD_DIM * sizeof(half),
        (size_t)kCtaQ * HEAD_DIM * sizeof(half));

    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max);

    dim3 grid(div_ceil_i(L_q, kCtaQ), 1, HN);
    dim3 block(32, num_warps);

    kernel_func<<<grid, block, smem_max, stream>>>(
        const_cast<int8_t*>(q_int8), const_cast<int8_t*>(k_int8), const_cast<half*>(v_f16),
        o_half, /*Lse=*/nullptr,
        const_cast<float*>(q_scale), const_cast<float*>(k_scale), /*V_mean=*/nullptr,
        (uint32_t)L_q, (uint32_t)L_k, /*num_kv_groups=*/1,
        stride_bz_q, stride_seq_q, /*stride_h_q=*/0u,
        stride_bz_k, stride_seq_k, /*stride_h_k=*/0u,
        stride_bz_v, stride_seq_v, /*stride_h_v=*/0u,
        stride_bz_o, stride_seq_o, /*stride_h_o=*/0u,
        sm_scale,
        qk_bias, stride_bz_bias);
}


// Launch the vendored FP8 sage kernel. Q/K are the same int8 contiguous
// head-major buffers as the F16 path; V is the transposed+permuted e4m3 buffer
// [hn][head_dim][L_k_pad]; v_scale is [hn][head_dim]. Output is f16 [d,L_q,HN].
template <int HEAD_DIM>
static void launch_sage_fp8_kernel(const int8_t* q_int8, const int8_t* k_int8, const int8_t* v_fp8,
                                   half* o_half, const float* q_scale, const float* k_scale,
                                   const float* v_scale,
                                   int L_q, int L_k, int L_k_pad, int HN,
                                   float sm_scale, cudaStream_t stream) {
    // Q/K: contiguous head-major [d_head, L, HN] -> per-hn slice via batch stride.
    const uint32_t stride_seq_q = HEAD_DIM;
    const uint32_t stride_seq_k = HEAD_DIM;
    const uint32_t stride_bz_q  = static_cast<uint32_t>(HEAD_DIM) * L_q;
    const uint32_t stride_bz_k  = static_cast<uint32_t>(HEAD_DIM) * L_k;
    // V transposed [hn][head_dim][L_k_pad]: d-axis stride = L_k_pad, token stride = 1.
    const uint32_t stride_d_v   = static_cast<uint32_t>(L_k_pad);
    const uint32_t stride_bz_v  = static_cast<uint32_t>(HEAD_DIM) * L_k_pad;
    // O: contiguous head-major [d_head, L_q, HN].
    const uint32_t stride_seq_o = HEAD_DIM;
    const uint32_t stride_bz_o  = static_cast<uint32_t>(HEAD_DIM) * L_q;

    constexpr int num_warps_q = kCtaQ / kWarpQ; // 4
    constexpr int num_warps_k = kCtaK / kWarpK; // 1
    (void)num_warps_k;

    auto kernel_func = qk_int_sv_f8_attn_kernel<
        kCtaQ, kCtaK, kWarpQ, kWarpK, HEAD_DIM,
        DataType::kInt8,
        QuantGranularity::kPerWarp, QuantGranularity::kPerBlock,
        /*DTypeSVAccum=*/float, /*use_inst_buffer=*/false, /*DTypeOut=*/half,
        ComputeUnit::kCudaCore, MaskMode::kNone, /*return_lse=*/false,
        /*fuse_v_scale=*/true, /*fuse_v_mean=*/false, /*use_pv_fp16_accu=*/false>;

    // smem: max(Q_int8 + K_int8 + V_fp8, O_half)
    size_t smem_max = std::max(
        (size_t)kCtaQ * HEAD_DIM * sizeof(int8_t) + (size_t)kCtaK * HEAD_DIM * sizeof(int8_t) + (size_t)kCtaK * HEAD_DIM * sizeof(int8_t),
        (size_t)kCtaQ * HEAD_DIM * sizeof(half));

    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max);

    dim3 grid(div_ceil_i(L_q, kCtaQ), 1, HN);
    dim3 block(32, num_warps_q * num_warps_k);

    kernel_func<<<grid, block, smem_max, stream>>>(
        const_cast<int8_t*>(q_int8), const_cast<int8_t*>(k_int8), const_cast<int8_t*>(v_fp8),
        o_half, /*Lse=*/nullptr,
        const_cast<float*>(q_scale), const_cast<float*>(k_scale),
        const_cast<float*>(v_scale), /*V_mean=*/nullptr,
        (uint32_t)L_q, (uint32_t)L_k, /*num_kv_groups=*/1,
        stride_bz_q, stride_seq_q, /*stride_h_q=*/0u,
        stride_bz_k, stride_seq_k, /*stride_h_k=*/0u,
        stride_bz_v, /*stride_h_v=*/0u, stride_d_v,
        stride_bz_o, stride_seq_o, /*stride_h_o=*/0u,
        sm_scale);
}


template <typename T, int HEAD_DIM>
static bool run_sage(ggml_tensor* dst, const ed_cuda_sage_attn_scratch* scratch,
                     cudaStream_t stream, int n_head) {
    const ggml_tensor* q = dst->src[0]; // [d_head, L_q, HN]
    const ggml_tensor* k = dst->src[1]; // [d_head, L_k, HN]
    const ggml_tensor* v = dst->src[2]; // [d_head, n_head, L_k, N]

    const int L_q = (int)q->ne[1];
    const int L_k = (int)k->ne[1];
    const int HN  = (int)q->ne[2];
    const int N   = HN / n_head;


    const int64_t q_s_seq = elem_stride(q, 1);
    const int64_t q_s_h   = elem_stride(q, 2);
    const int64_t k_s_seq = elem_stride(k, 1);
    const int64_t k_s_h   = elem_stride(k, 2);

    const int num_warp_chunks = div_ceil_i(L_q, kCtaQ) * (kCtaQ / kWarpQ); // per head-batch
    const int num_k_blocks    = div_ceil_i(L_k, kCtaK);                    // per head-batch

    int8_t* q_int8 = static_cast<int8_t*>(scratch->q_int8);
    int8_t* k_int8 = static_cast<int8_t*>(scratch->k_int8);
    float*  q_scale = static_cast<float*>(scratch->q_scale);
    float*  k_scale = static_cast<float*>(scratch->k_scale);
    float*  km      = static_cast<float*>(scratch->km);
    float*  qm      = static_cast<float*>(scratch->qm);
    float*  qk_bias = static_cast<float*>(scratch->qk_bias);
    half*   v_f16   = static_cast<half*>(scratch->v_f16);
    half*   o_half  = static_cast<half*>(scratch->o_half);
    int8_t* v_fp8   = static_cast<int8_t*>(scratch->v_fp8);
    float*  v_scale = static_cast<float*>(scratch->v_scale);
    const bool use_pv_fp8 = edgedit::ggml_ext::sage_pv_fp8_enabled();
    if (!q_int8 || !k_int8 || !q_scale || !k_scale || !km || !o_half) {
        return false;
    }
    if (use_pv_fp8) {
        if (!v_fp8 || !v_scale) return false;
    } else {
        if (!v_f16) return false;
    }
    // Smooth-Q needs the F16-PV score-bias path (not implemented for FP8-PV) and
    // its own scratch. It also requires BOTH the Q-mean subtraction AND the
    // rank-1 add-back together (dropping the add-back is worse than neither).
    const bool use_smooth_q = edgedit::ggml_ext::sage_smooth_q_enabled(HEAD_DIM) &&
                              !use_pv_fp8 && qm != nullptr && qk_bias != nullptr;
    static bool logged_sq = false;
    if (std::getenv("ED_SAGE_ATTN_DEBUG") && !logged_sq) {
        fprintf(stderr, "[sage] run_sage HEAD_DIM=%d use_smooth_q=%d use_pv_fp8=%d L_q=%d L_k=%d HN=%d\n",
                (int)HEAD_DIM, (int)use_smooth_q, (int)use_pv_fp8, L_q, L_k, HN);
        logged_sq = true;
    }

    const T* q_ptr = static_cast<const T*>(q->data);
    const T* k_ptr = static_cast<const T*>(k->data);
    const T* v_ptr = static_cast<const T*>(v->data);

    // 1) per-channel K mean (smooth-K). Can be disabled for A/B via env.
    const bool use_smooth_k = (std::getenv("ED_SAGE_NO_SMOOTHK") == nullptr);
    if (use_smooth_k) {
        sage_k_mean_kernel<T, HEAD_DIM><<<HN, HEAD_DIM, 0, stream>>>(
            k_ptr, km, L_k, k_s_seq, k_s_h);
    }
    float* km_arg = use_smooth_k ? km : nullptr;

    // 1b) smooth-Q: per-channel Q mean + rank-1 per-key bias qm·K_origᵀ.
    if (use_smooth_q) {
        sage_q_mean_kernel<T, HEAD_DIM><<<HN, HEAD_DIM, 0, stream>>>(
            q_ptr, qm, L_q, q_s_seq, q_s_h);
        // bias[hn][j] = sum_d qm[hn][d] * K_orig[hn][j][d], one block per (key, hn)
        dim3 grid(L_k, HN, 1);
        sage_smooth_q_bias_kernel<T, HEAD_DIM><<<grid, HEAD_DIM, 0, stream>>>(
            k_ptr, qm, qk_bias, L_k, k_s_seq, k_s_h);
    }
    float* qm_arg = use_smooth_q ? qm : nullptr;

    // 2) quantize Q (per-warp), subtracting qm when smooth-Q, into [d_head, L_q, HN]
    {
        dim3 grid(num_warp_chunks, HN, 1);
        dim3 block(256);
        sage_quant_q_perwarp_kernel<T, HEAD_DIM, kWarpQ><<<grid, block, 0, stream>>>(
            q_ptr, qm_arg, q_int8, q_scale, L_q, num_warp_chunks, q_s_seq, q_s_h);
    }
    // 3) quantize K (per-block) with smooth-K into contiguous [d_head, L_k, HN]
    {
        dim3 grid(num_k_blocks, HN, 1);
        dim3 block(256);
        sage_quant_k_perblock_smoothk_kernel<T, HEAD_DIM, kCtaK><<<grid, block, 0, stream>>>(
            k_ptr, km_arg, k_int8, k_scale, L_k, num_k_blocks, k_s_seq, k_s_h);
    }
    // 4) V preparation. Resolve V strides per semantic role first:
    //      v_sd = stride of d_head axis, v_sh = stride of head axis,
    //      v_st = stride of token axis, v_sb = stride of batch axis.
    int64_t v_sd, v_sh, v_st, v_sb;
    if (v->ne[1] == n_head && v->ne[2] == L_k) {
        // 4D token-major [d_head, n_head, L_k, N]
        v_sd = elem_stride(v, 0);
        v_sh = elem_stride(v, 1);
        v_st = elem_stride(v, 2);
        v_sb = elem_stride(v, 3);
    } else {
        // 3D packed [d_head, L_k, n_head] (N == 1)
        v_sd = elem_stride(v, 0);
        v_st = elem_stride(v, 1);
        v_sh = elem_stride(v, 2);
        v_sb = elem_stride(v, 3);
    }

    const float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);
    const int L_k_pad = div_ceil_i(L_k, kCtaK) * kCtaK;


    if (use_pv_fp8) {
        // 4a) per-channel V scale (amax/448), then transpose+permute+quant to e4m3
        constexpr float kE4M3Max = 448.0f;
        sage_v_channel_scale_kernel<T, HEAD_DIM><<<HN, HEAD_DIM, 0, stream>>>(
            v_ptr, v_scale, L_k, n_head, N, v_sd, v_sh, v_st, v_sb, kE4M3Max);
        {
            const int64_t total = (int64_t)HEAD_DIM * L_k_pad * n_head * N;
            const int threads = 256;
            const int blocks = (int)((total + threads - 1) / threads);
            sage_v_to_fp8_transposed_kernel<T, HEAD_DIM><<<blocks, threads, 0, stream>>>(
                v_ptr, v_fp8, v_scale, L_k, L_k_pad, n_head, N, v_sd, v_sh, v_st, v_sb);
        }
        // 5a) FP8-PV sage kernel (smooth-Q not supported on this path)
        launch_sage_fp8_kernel<HEAD_DIM>(q_int8, k_int8, v_fp8, o_half, q_scale, k_scale, v_scale,
                               L_q, L_k, L_k_pad, HN, sm_scale, stream);
    } else {
        // 4b) V -> contiguous f16 [d_head, L_k, HN] (hn = h + n_head*b)
        {
            const int64_t total = (int64_t)HEAD_DIM * L_k * n_head * N;
            const int threads = 256;
            const int blocks = (int)((total + threads - 1) / threads);
            sage_v_to_f16_kernel<T, HEAD_DIM><<<blocks, threads, 0, stream>>>(
                v_ptr, v_f16, L_k, n_head, N, v_sd, v_sh, v_st, v_sb);
        }
        // 5b) F16-PV sage kernel: o_half [d_head, L_q, HN]
        if (use_smooth_q) {
            launch_sage_kernel<HEAD_DIM, true>(q_int8, k_int8, v_f16, o_half, q_scale, k_scale,
                               L_q, L_k, HN, sm_scale, stream, qk_bias);
        } else {
            launch_sage_kernel<HEAD_DIM, false>(q_int8, k_int8, v_f16, o_half, q_scale, k_scale,
                               L_q, L_k, HN, sm_scale, stream, nullptr);
        }
    }

    // 6) o_half -> dst F32 [d_head, HN, L_q, 1]
    {
        const int64_t total = (int64_t)HEAD_DIM * L_q * HN;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        sage_o_half_to_f32_kernel<HEAD_DIM><<<blocks, threads, 0, stream>>>(
            o_half, static_cast<float*>(dst->data), L_q, HN);
    }

    // 7) Near-uniform head fallback (head_dim==128 only). Detect heads whose
    // attention is near-uniform (the ones the INT8-PV kernel collapses) and
    // recompute ONLY those in full precision, overwriting their dst slice. Pure
    // device-side on `stream` (no host sync) so it is CUDA-graph safe. Gated on
    // HEAD_DIM==128 at compile time -> SD3 (d64) never enters this path.
    int* uniform_flag = static_cast<int*>(scratch->uniform_flag);
    if (HEAD_DIM == 128 &&
        edgedit::ggml_ext::sage_fallback_enabled(HEAD_DIM) &&
        uniform_flag != nullptr) {
        float* head_entropy = static_cast<float*>(scratch->head_entropy);
        float* detect_acc   = static_cast<float*>(scratch->detect_acc); // [4*HN]
        const int   n_samples = std::max(1, std::min(L_q, edgedit::ggml_ext::sage_fallback_sample_rows()));
        const float err_thr   = edgedit::ggml_ext::sage_fallback_err_thr();
        const int   row_stride = (L_q > n_samples) ? (L_q / n_samples) : 1;
        // detection: HN*n_samples warps, each handles one (head, sampled row).
        // Measures each head's sampled rel_L2 (sage vs full precision) via atomics
        // into detect_acc, then finalize turns it into a flag.
        if (detect_acc != nullptr) {
            cudaMemsetAsync(detect_acc, 0, (size_t)4 * HN * sizeof(float), stream);
            const int total_warps = HN * n_samples;
            const int warps_per_block = 8;             // 256 threads/block
            const int threads = warps_per_block * 32;
            const int blocks = div_ceil_i(total_warps, warps_per_block);
            sage_detect_uniform_kernel<T, HEAD_DIM><<<blocks, threads, 0, stream>>>(
                q_ptr, k_ptr, v_ptr, static_cast<const float*>(dst->data),
                L_q, L_k, HN, q_s_seq, q_s_h, k_s_seq, k_s_h,
                v_sd, v_sh, v_st, v_sb, n_head, sm_scale, n_samples, row_stride, detect_acc);
            sage_detect_finalize_kernel<<<div_ceil_i(HN, 128), 128, 0, stream>>>(
                detect_acc, HN, err_thr, uniform_flag, head_entropy);
        }
        // overwrite: full-precision recompute for flagged heads only. One warp per
        // (query row, head); early-exits whole blocks for non-flagged heads.
        {
            const int warps_per_block = 8;
            const int threads = warps_per_block * 32;
            dim3 grid_fb(div_ceil_i(L_q, warps_per_block), HN, 1);
            sage_fallback_overwrite_kernel<T, HEAD_DIM><<<grid_fb, threads, 0, stream>>>(
                q_ptr, k_ptr, v_ptr, static_cast<float*>(dst->data),
                L_q, L_k, HN, q_s_seq, q_s_h, k_s_seq, k_s_h,
                v_sd, v_sh, v_st, v_sb, n_head, sm_scale, uniform_flag);
        }
        // First-dispatch report (host sync -> debug only, never the hot path).
        if (std::getenv("ED_SAGE_ATTN_DEBUG")) {
            static bool logged_fb = false;
            if (!logged_fb) {
                logged_fb = true;
                std::vector<int>   h_flag(HN, 0);
                std::vector<float> h_ent(HN, 0.0f);
                cudaMemcpyAsync(h_flag.data(), uniform_flag, HN * sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_ent.data(), head_entropy, HN * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                int n_fb = 0;
                for (int hn = 0; hn < HN; ++hn) n_fb += h_flag[hn];
                fprintf(stderr, "[sage] FALLBACK HEAD_DIM=%d err_thr=%.3f n_samples=%d L_q=%d L_k=%d HN=%d -> %d heads recomputed F16, %d stay sage\n",
                        (int)HEAD_DIM, err_thr, n_samples, L_q, L_k, HN, n_fb, HN - n_fb);
                for (int hn = 0; hn < HN; ++hn) {
                    fprintf(stderr, "[sage] FALLBACK hn=%d norm_entropy=%.4f %s\n",
                            hn, h_ent[hn], h_flag[hn] ? "-> F16 (collapsed)" : "sage");
                }
            }
        }
        // Aggregate stats across ALL dispatches (layers x steps): accumulates how
        // many head-instances collapse (flagged) and how many are near-uniform by
        // entropy, then prints the run-wide ratios at exit. Syncs per dispatch, so
        // it is a measurement mode only (ED_SAGE_FALLBACK_STATS), never the hot
        // path. This is what answers "what fraction of heads collapse" honestly.
        if (std::getenv("ED_SAGE_FALLBACK_STATS")) {
            static long long tot_head_inst = 0, tot_flagged = 0, tot_near_uniform = 0, n_dispatch = 0;
            static bool atexit_set = false;
            static float uniform_ent_thr = 0.90f; // "near-uniform" definition for the ratio
            std::vector<int>   h_flag(HN, 0);
            std::vector<float> h_ent(HN, 0.0f);
            cudaMemcpyAsync(h_flag.data(), uniform_flag, HN * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_ent.data(), head_entropy, HN * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int hn = 0; hn < HN; ++hn) {
                tot_head_inst++;
                if (h_flag[hn]) tot_flagged++;
                if (h_ent[hn] > uniform_ent_thr) tot_near_uniform++;
            }
            n_dispatch++;
            if (!atexit_set) {
                atexit_set = true;
                std::atexit([](){
                    fprintf(stderr, "[sage] FALLBACK_STATS dispatches=%lld head-instances=%lld  collapsed(flagged rel_L2>thr)=%lld (%.2f%%)  near-uniform(entropy>%.2f)=%lld (%.2f%%)\n",
                            n_dispatch, tot_head_inst, tot_flagged,
                            tot_head_inst ? 100.0 * (double)tot_flagged / (double)tot_head_inst : 0.0,
                            uniform_ent_thr, tot_near_uniform,
                            tot_head_inst ? 100.0 * (double)tot_near_uniform / (double)tot_head_inst : 0.0);
                });
            }
        }
    }

    // Optional diagnostic: dump one head-batch's original Q/K/V and sage output
    // Optional diagnostic: measure the sage kernel's relative L2 error against a
    // full-precision reference, PER head-batch, for the FIRST dispatch only.
    if (std::getenv("ED_SAGE_SELFCHECK")) {
        static bool done = false;
        if (!done) {
            done = true;
            const int nacc = 2 * HN;
            double* acc = nullptr;
            cudaMalloc(&acc, (nacc + 2 * HN) * sizeof(double));
            cudaMemsetAsync(acc, 0, (nacc + 2 * HN) * sizeof(double), stream);
            dim3 grid(L_q, HN, 1);
            sage_selfcheck_kernel<T, HEAD_DIM><<<grid, HEAD_DIM, 0, stream>>>(
                q_ptr, k_ptr, v_ptr, static_cast<const float*>(dst->data),
                L_q, L_k, HN, q_s_seq, q_s_h, k_s_seq, k_s_h,
                v_sd, v_sh, v_st, v_sb, n_head, sm_scale, acc);
            std::vector<double> h_acc(nacc + 2 * HN, 0.0);
            cudaMemcpyAsync(h_acc.data(), acc, (nacc + 2 * HN) * sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            double sse_all = 0.0, ref_all = 0.0;
            double worst = -1.0; int worst_hn = -1;
            for (int hn = 0; hn < HN; ++hn) {
                const double sse = h_acc[2 * hn + 0];
                const double ref = h_acc[2 * hn + 1];
                const double sag = h_acc[2 * HN + hn];
                const double dot = h_acc[3 * HN + hn];
                sse_all += sse; ref_all += ref;
                const double rel = (ref > 0) ? sqrt(sse / ref) : -1.0;
                const double cosv = (ref > 0 && sag > 0) ? dot / (sqrt(ref) * sqrt(sag)) : 0.0;
                if (rel > worst) { worst = rel; worst_hn = hn; }
                fprintf(stderr, "[sage] SELFCHECK HEAD_DIM=%d smooth_q=%d hn=%d rel_L2=%.4f cos=%.4f |ref|=%.3e |sage|=%.3e\n",
                        (int)HEAD_DIM, (int)use_smooth_q, hn, rel, cosv, sqrt(ref), sqrt(sag));
            }
            const double rel_all = (ref_all > 0) ? sqrt(sse_all / ref_all) : -1.0;
            fprintf(stderr, "[sage] SELFCHECK HEAD_DIM=%d smooth_q=%d OVERALL rel_L2=%.4f  WORST hn=%d rel_L2=%.4f\n",
                    (int)HEAD_DIM, (int)use_smooth_q, rel_all, worst_hn, worst);
            cudaFree(acc);
        }
    }
    return true;
}

} // namespace

bool ed_cuda_sage_attn_custom_supported(const ggml_tensor* dst) {
    if (!edgedit::ggml_ext::sage_attn_enabled()) {
        return false;
    }
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM) {
        return false;
    }
    if (dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }
    const SageAttnCustomParams params = decode_params(dst);
    if (!edgedit::ggml_ext::sage_attn_params_valid(params)) {
        return false;
    }
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    if (!edgedit::ggml_ext::sage_attn_shape_supported(q, k, v, params.n_head)) {
        return false;
    }
    // dst must be F32 [d_head, HN, L_q, 1] and contiguous
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t HN = q->ne[2];
    const int64_t d_head = q->ne[0];
    if ((d_head != 64 && d_head != 128) ||
        dst->ne[0] != d_head || dst->ne[1] != HN || dst->ne[2] != q->ne[1] || dst->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    // last dim of q/k/v must be contiguous (d fastest)
    if (elem_stride(q, 0) != 1 || elem_stride(k, 0) != 1 || elem_stride(v, 0) != 1) {
        return false;
    }
    return true;
}

bool ed_cuda_sage_attn_scratch_sizes_for(const ggml_tensor* dst, ed_cuda_sage_attn_scratch_sizes* out) {
    if (out == nullptr || !ed_cuda_sage_attn_custom_supported(dst)) {
        return false;
    }
    const SageAttnCustomParams params = decode_params(dst);
    sage_dims d{};
    if (!compute_dims(dst, params.n_head, &d)) {
        return false;
    }
    const size_t HD = (size_t)dst->src[0]->ne[0]; // actual head_dim (64 or 128)
    out->q_int8_bytes  = HD * d.L_q * d.HN * sizeof(int8_t);
    out->k_int8_bytes  = HD * d.L_k * d.HN * sizeof(int8_t);
    out->q_scale_bytes = (size_t)d.num_warp_chunks * d.HN * sizeof(float);
    out->k_scale_bytes = (size_t)d.num_k_blocks * d.HN * sizeof(float);
    out->km_bytes      = HD * d.HN * sizeof(float);
    out->v_f16_bytes   = HD * d.L_k * d.HN * sizeof(half);
    out->o_half_bytes  = HD * d.L_q * d.HN * sizeof(half);
    // Smooth-Q scratch: qm is one float per (hn, channel); qk_bias is one float
    // per (hn, key). Always sized (cheap) so the dispatch can allocate
    // unconditionally; only used when smooth-Q is enabled (head_dim==128).
    out->qm_bytes      = HD * d.HN * sizeof(float);
    out->qk_bias_bytes = (size_t)d.HN * d.L_k * sizeof(float);
    // FP8-PV scratch: V is transposed+padded to L_k_pad and stored as e4m3
    // bytes; v_scale is one float per (hn, channel). Always sized (cheap) so
    // the dispatch can allocate unconditionally; only used when ED_SAGE_PV_FP8.
    const int L_k_pad  = div_ceil_i(d.L_k, kCtaK) * kCtaK;
    out->v_fp8_bytes   = HD * L_k_pad * d.HN * sizeof(int8_t);
    out->v_scale_bytes = HD * d.HN * sizeof(float);
    // Near-uniform head fallback scratch: one flag + one entropy value per
    // head-batch. Tiny; always sized so the dispatch can allocate it
    // unconditionally (only used when the fallback runs, head_dim==128).
    out->uniform_flag_bytes = (size_t)d.HN * sizeof(int);
    out->head_entropy_bytes = (size_t)d.HN * sizeof(float);
    out->detect_acc_bytes   = (size_t)4 * d.HN * sizeof(float);
    return true;
}

bool ed_cuda_sage_attn_custom_compute(ggml_tensor* dst,
                                      const ed_cuda_sage_attn_scratch* scratch,
                                      void* stream) {
    if (!ed_cuda_sage_attn_custom_supported(dst) || scratch == nullptr) {
        return false;
    }
    const SageAttnCustomParams params = decode_params(dst);
    cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
    const ggml_tensor* q = dst->src[0];
    const int64_t d_head = q->ne[0];
    bool ok = false;
    if (q->type == GGML_TYPE_F32) {
        if (d_head == 64)       ok = run_sage<float, 64>(dst, scratch, cu_stream, params.n_head);
        else if (d_head == 128) ok = run_sage<float, 128>(dst, scratch, cu_stream, params.n_head);
    } else if (q->type == GGML_TYPE_F16) {
        if (d_head == 64)       ok = run_sage<half, 64>(dst, scratch, cu_stream, params.n_head);
        else if (d_head == 128) ok = run_sage<half, 128>(dst, scratch, cu_stream, params.n_head);
    }
    return ok;
}
