#include "ed_cuda_sp_flux.h"

#include "backend/ggml/ed_ggml_sp_flux_ext.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using edgedit::ggml_ext::FluxSPQKVRecvPrepCustomParams;
using edgedit::ggml_ext::FluxSPQKVPairRecvPrepCustomParams;
using edgedit::ggml_ext::FluxSPQKVMixedRecvPrepCustomParams;
using edgedit::ggml_ext::FluxSPQKVPairMixedRecvPrepCustomParams;
using edgedit::ggml_ext::FluxSPQKVRecvPrepBundleCustomParams;
using edgedit::ggml_ext::FluxSPQKVCombinedPairRecvPrepCustomParams;
using edgedit::ggml_ext::FluxSPQKVCombinedPairRecvPrepBundleCustomParams;
using edgedit::ggml_ext::FluxSPQKVRecvPrepPlane;
using edgedit::ggml_ext::FluxSPAllToAllCustomParams;
using edgedit::ggml_ext::FluxSPAllGatherCustomParams;

static bool flux_sp_profile_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("ED_PROFILE_FLUX_SP_CUSTOM");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("ED_PROFILE_FLUX_SP_CUSTOM_OPS");
        }
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

static int flux_sp_profile_rank_fallback() {
    const char* names[] = {
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
        "RANK",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return std::atoi(value);
        }
    }
    return -1;
}

static const char* flux_sp_profile_category(const ggml_tensor* dst) {
    const char* name = (dst != nullptr && dst->name[0] != '\0') ? dst->name : "";
    if (std::strstr(name, "_qkv_seq_to_head") != nullptr) {
        return "qkv_seq_to_head";
    }
    if (std::strstr(name, "_txt_img_attn_head_to_seq") != nullptr) {
        return "double_head_to_seq";
    }
    if (std::strstr(name, "_attn_head_to_seq") != nullptr) {
        return "single_head_to_seq";
    }
    return "other";
}

static const char* flux_sp_profile_plane_name(FluxSPQKVRecvPrepPlane plane) {
    switch (plane) {
        case FluxSPQKVRecvPrepPlane::Q:
            return "q";
        case FluxSPQKVRecvPrepPlane::K:
            return "k";
        case FluxSPQKVRecvPrepPlane::V:
            return "v";
    }
    return "unknown";
}

static edgedit::parallel::DataType flux_sp_cuda_all_to_all_data_type(const ggml_tensor* tensor) {
    if (tensor != nullptr && tensor->type == GGML_TYPE_F16) {
        return edgedit::parallel::DataType::kFloat16;
    }
    return edgedit::parallel::DataType::kFloat32;
}

static int64_t flux_sp_cuda_tensor_element_size(const ggml_tensor* tensor) {
    if (tensor != nullptr && tensor->type == GGML_TYPE_F16) {
        return static_cast<int64_t>(sizeof(uint16_t));
    }
    return static_cast<int64_t>(sizeof(float));
}

static bool flux_sp_profile_begin(cudaStream_t stream, cudaEvent_t* start, cudaEvent_t* stop) {
    if (!flux_sp_profile_enabled()) {
        return false;
    }
    *start = nullptr;
    *stop = nullptr;
    if (cudaEventCreate(start) != cudaSuccess) {
        return false;
    }
    if (cudaEventCreate(stop) != cudaSuccess) {
        cudaEventDestroy(*start);
        *start = nullptr;
        return false;
    }
    if (cudaEventRecord(*start, stream) != cudaSuccess) {
        cudaEventDestroy(*start);
        cudaEventDestroy(*stop);
        *start = nullptr;
        *stop = nullptr;
        return false;
    }
    return true;
}

static void flux_sp_profile_end(cudaStream_t stream,
                                cudaEvent_t start,
                                cudaEvent_t stop,
                                const char* kind,
                                const char* detail,
                                const ggml_tensor* dst,
                                int rank,
                                int64_t elems,
                                int64_t bytes) {
    if (start == nullptr || stop == nullptr) {
        return;
    }
    float elapsed_ms = 0.0f;
    if (cudaEventRecord(stop, stream) == cudaSuccess &&
        cudaEventSynchronize(stop) == cudaSuccess &&
        cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess) {
        const char* name = (dst != nullptr && dst->name[0] != '\0') ? dst->name : "-";
        std::fprintf(stderr,
                     "ED_FLUX_SP_CUSTOM_PROFILE rank=%d kind=%s detail=%s category=%s name=%s elems=%lld bytes_mib=%.3f elapsed_ms=%.3f\n",
                     rank,
                     kind != nullptr ? kind : "-",
                     detail != nullptr ? detail : "-",
                     flux_sp_profile_category(dst),
                     name,
                     static_cast<long long>(elems),
                     static_cast<double>(bytes) / (1024.0 * 1024.0),
                     static_cast<double>(elapsed_ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

static inline int64_t elem_stride(const ggml_tensor* t, int dim) {
    return t->nb[dim] / ggml_type_size(t->type);
}

static bool is_contiguous_3d_output(const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return false;
    }
    const size_t ts = ggml_type_size(t->type) / ggml_blck_size(t->type);
    return t->nb[0] == ts &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * ts &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * ts;
}

static bool is_contiguous_4d_output(const ggml_tensor* t) {
    if (t == nullptr || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        return false;
    }
    const size_t ts = ggml_type_size(t->type) / ggml_blck_size(t->type);
    return t->nb[0] == ts &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * ts &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * ts &&
           t->nb[3] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * static_cast<size_t>(t->ne[2]) * ts;
}

static bool is_contiguous_1d_output(const ggml_tensor* t) {
    if (t == nullptr || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        return false;
    }
    const size_t ts = ggml_type_size(t->type) / ggml_blck_size(t->type);
    return t->ne[1] == 1 &&
           t->ne[2] == 1 &&
           t->ne[3] == 1 &&
           t->nb[0] == ts;
}

static bool pe_is_matrix_layout(const ggml_tensor* pe, int64_t seq, int64_t half) {
    return pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= seq;
}

static bool pe_is_prepared_layout(const ggml_tensor* pe, int64_t seq, int64_t half) {
    return pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= seq && pe->ne[3] == 2;
}

__device__ __forceinline__ float load_pe(const float* pe,
                                         bool prepared,
                                         int64_t s,
                                         int64_t p,
                                         int row,
                                         int col,
                                         int64_t pe_s0,
                                         int64_t pe_s1,
                                         int64_t pe_s2,
                                         int64_t pe_s3) {
    if (prepared) {
        return pe[row * pe_s0 + p * pe_s1 + s * pe_s2 + col * pe_s3];
    }
    return pe[row * pe_s0 + col * pe_s1 + p * pe_s2 + s * pe_s3];
}

__device__ __forceinline__ float load_recv(const float* recv_flat,
                                           int plane,
                                           int d,
                                           int seq,
                                           int head,
                                           int head_dim,
                                           int shard_heads,
                                           int shard_sequence,
                                           int world_size) {
    const int src_peer = seq / shard_sequence;
    const int local_seq = seq - src_peer * shard_sequence;
    const int local_head = head;
    const int total_head_dim = head_dim * 3;
    const int64_t idx =
        static_cast<int64_t>(plane) * head_dim +
        d +
        static_cast<int64_t>(local_head) * total_head_dim +
        static_cast<int64_t>(local_seq) * total_head_dim * shard_heads +
        static_cast<int64_t>(src_peer) * total_head_dim * shard_heads * shard_sequence;
    (void)world_size;
    return recv_flat[idx];
}

__device__ __forceinline__ float load_recv_f16(const __half* recv_flat,
                                               int plane,
                                               int d,
                                               int seq,
                                               int head,
                                               int head_dim,
                                               int shard_heads,
                                               int shard_sequence,
                                               int world_size) {
    const int src_peer = seq / shard_sequence;
    const int local_seq = seq - src_peer * shard_sequence;
    const int local_head = head;
    const int total_head_dim = head_dim * 3;
    const int64_t idx =
        static_cast<int64_t>(plane) * head_dim +
        d +
        static_cast<int64_t>(local_head) * total_head_dim +
        static_cast<int64_t>(local_seq) * total_head_dim * shard_heads +
        static_cast<int64_t>(src_peer) * total_head_dim * shard_heads * shard_sequence;
    (void)world_size;
    return __half2float(recv_flat[idx]);
}

__device__ __forceinline__ int64_t recv_pair_index(int plane,
                                                   int pair,
                                                   int seq,
                                                   int head,
                                                   int head_dim,
                                                   int shard_heads,
                                                   int shard_sequence) {
    const int src_peer = seq / shard_sequence;
    const int local_seq = seq - src_peer * shard_sequence;
    const int total_head_dim = head_dim * 3;
    return static_cast<int64_t>(plane) * head_dim +
           2 * pair +
           static_cast<int64_t>(head) * total_head_dim +
           static_cast<int64_t>(local_seq) * total_head_dim * shard_heads +
           static_cast<int64_t>(src_peer) * total_head_dim * shard_heads * shard_sequence;
}

__device__ __forceinline__ half2 load_recv_f16_half2(const __half* recv_flat,
                                                     int plane,
                                                     int pair,
                                                     int seq,
                                                     int head,
                                                     int head_dim,
                                                     int shard_heads,
                                                     int shard_sequence) {
    const int64_t idx = recv_pair_index(plane, pair, seq, head, head_dim, shard_heads, shard_sequence);
    return *reinterpret_cast<const half2*>(recv_flat + idx);
}

__device__ __forceinline__ float2 load_recv_f16_pair(const __half* recv_flat,
                                                     int plane,
                                                     int pair,
                                                     int seq,
                                                     int head,
                                                     int head_dim,
                                                     int shard_heads,
                                                     int shard_sequence) {
    return __half22float2(load_recv_f16_half2(recv_flat, plane, pair, seq, head, head_dim, shard_heads, shard_sequence));
}

__device__ __forceinline__ int64_t combined_pair_recv_index(int plane,
                                                            int d,
                                                            int seq,
                                                            int head,
                                                            int head_dim,
                                                            int shard_heads,
                                                            int world_size,
                                                            int first_shard_sequence,
                                                            int second_shard_sequence) {
    const int first_sequence = first_shard_sequence * world_size;
    const bool use_first = seq < first_sequence;
    const int stream_seq = use_first ? seq : seq - first_sequence;
    const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;
    const int src_peer = stream_seq / shard_sequence;
    const int local_seq = stream_seq - src_peer * shard_sequence;
    const int total_head_dim = head_dim * 3;
    const int64_t first_chunk = static_cast<int64_t>(total_head_dim) * shard_heads * first_shard_sequence;
    const int64_t second_chunk = static_cast<int64_t>(total_head_dim) * shard_heads * second_shard_sequence;
    const int64_t stream_offset = use_first ? 0 : first_chunk;
    return static_cast<int64_t>(src_peer) * (first_chunk + second_chunk) +
           stream_offset +
           static_cast<int64_t>(plane) * head_dim +
           d +
           static_cast<int64_t>(head) * total_head_dim +
           static_cast<int64_t>(local_seq) * total_head_dim * shard_heads;
}

__device__ __forceinline__ float load_combined_pair_recv(const float* recv_flat,
                                                         int plane,
                                                         int d,
                                                         int seq,
                                                         int head,
                                                         int head_dim,
                                                         int shard_heads,
                                                         int world_size,
                                                         int first_shard_sequence,
                                                         int second_shard_sequence) {
    return recv_flat[combined_pair_recv_index(plane,
                                             d,
                                             seq,
                                             head,
                                             head_dim,
                                             shard_heads,
                                             world_size,
                                             first_shard_sequence,
                                             second_shard_sequence)];
}

__device__ __forceinline__ float load_combined_pair_recv_f16(const __half* recv_flat,
                                                             int plane,
                                                             int d,
                                                             int seq,
                                                             int head,
                                                             int head_dim,
                                                             int shard_heads,
                                                             int world_size,
                                                             int first_shard_sequence,
                                                             int second_shard_sequence) {
    return __half2float(recv_flat[combined_pair_recv_index(plane,
                                                          d,
                                                          seq,
                                                          head,
                                                          head_dim,
                                                          shard_heads,
                                                          world_size,
                                                          first_shard_sequence,
                                                          second_shard_sequence)]);
}

__device__ __forceinline__ float2 load_combined_pair_recv_f16_pair(const __half* recv_flat,
                                                                   int plane,
                                                                   int pair,
                                                                   int seq,
                                                                   int head,
                                                                   int head_dim,
                                                                   int shard_heads,
                                                                   int world_size,
                                                                   int first_shard_sequence,
                                                                   int second_shard_sequence) {
    const int64_t idx = combined_pair_recv_index(plane,
                                                2 * pair,
                                                seq,
                                                head,
                                                head_dim,
                                                shard_heads,
                                                world_size,
                                                first_shard_sequence,
                                                second_shard_sequence);
    return __half22float2(*reinterpret_cast<const half2*>(recv_flat + idx));
}

__device__ __forceinline__ half2 load_combined_pair_recv_f16_half2(const __half* recv_flat,
                                                                   int plane,
                                                                   int pair,
                                                                   int seq,
                                                                   int head,
                                                                   int head_dim,
                                                                   int shard_heads,
                                                                   int world_size,
                                                                   int first_shard_sequence,
                                                                   int second_shard_sequence) {
    const int64_t idx = combined_pair_recv_index(plane,
                                                2 * pair,
                                                seq,
                                                head,
                                                head_dim,
                                                shard_heads,
                                                world_size,
                                                first_shard_sequence,
                                                second_shard_sequence);
    return *reinterpret_cast<const half2*>(recv_flat + idx);
}

__device__ __forceinline__ float load_mixed_recv(const uint32_t* recv_flat,
                                                 int plane,
                                                 int d,
                                                 int seq,
                                                 int head,
                                                 int head_dim,
                                                 int shard_heads,
                                                 int shard_sequence) {
    const int src_peer = seq / shard_sequence;
    const int local_seq = seq - src_peer * shard_sequence;
    const int packed_dim = head_dim * 2;
    const int64_t idx =
        (plane == 0 ? d : head_dim + d) +
        static_cast<int64_t>(head) * packed_dim +
        static_cast<int64_t>(local_seq) * packed_dim * shard_heads +
        static_cast<int64_t>(src_peer) * packed_dim * shard_heads * shard_sequence;
    const uint32_t packed = recv_flat[idx];
    if (plane == 0) {
        return __uint_as_float(packed);
    }
    const unsigned short half_bits = static_cast<unsigned short>(plane == 2 ? (packed >> 16) : (packed & 0xffffu));
    return __half2float(__ushort_as_half(half_bits));
}

__device__ __forceinline__ void store_q_rope_pair(float* dst, int64_t dst_idx, float y0, float y1) {
    dst[dst_idx + 0] = y0;
    dst[dst_idx + 1] = y1;
}

__device__ __forceinline__ void store_q_rope_pair(__half* dst, int64_t dst_idx, float y0, float y1) {
    reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
}

template <typename DstT>
__global__ void flux_sp_qkv_recv_q_rope_kernel(const float* __restrict__ recv_flat,
                                               const float* __restrict__ pe,
                                               DstT* __restrict__ dst,
                                               int plane,
                                               int head_dim,
                                               int sequence,
                                               int shard_heads,
                                               int shard_sequence,
                                               int world_size,
                                               bool pe_prepared,
                                               int64_t pe_s0,
                                               int64_t pe_s1,
                                               int64_t pe_s2,
                                               int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float x0 = load_recv(recv_flat, plane, 2 * p + 0, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const float x1 = load_recv(recv_flat, plane, 2 * p + 1, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

__global__ void flux_sp_qkv_recv_k_rope_kernel(const float* __restrict__ recv_flat,
                                               const float* __restrict__ pe,
                                               __half* __restrict__ dst,
                                               int plane,
                                               int head_dim,
                                               int sequence,
                                               int shard_heads,
                                               int shard_sequence,
                                               int world_size,
                                               bool pe_prepared,
                                               int64_t pe_s0,
                                               int64_t pe_s1,
                                               int64_t pe_s2,
                                               int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float x0 = load_recv(recv_flat, plane, 2 * p + 0, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const float x1 = load_recv(recv_flat, plane, 2 * p + 1, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_recv_v_prep_kernel(const float* __restrict__ recv_flat,
                                               __half* __restrict__ dst,
                                               int plane,
                                               int head_dim,
                                               int sequence,
                                               int shard_heads,
                                               int shard_sequence,
                                               int world_size) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;

        const float v0 = load_recv(recv_flat, plane, d + 0, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const float v1 = load_recv(recv_flat, plane, d + 1, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_recv_q_rope_f16_kernel(const __half* __restrict__ recv_flat,
                                                   const float* __restrict__ pe,
                                                   DstT* __restrict__ dst,
                                                   int plane,
                                                   int head_dim,
                                                   int sequence,
                                                   int shard_heads,
                                                   int shard_sequence,
                                                   int world_size,
                                                   bool pe_prepared,
                                                   int64_t pe_s0,
                                                   int64_t pe_s1,
                                                   int64_t pe_s2,
                                                   int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float2 x = load_recv_f16_pair(recv_flat, plane, p, s, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x.x * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x.x * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

__global__ void flux_sp_qkv_recv_k_rope_f16_kernel(const __half* __restrict__ recv_flat,
                                                   const float* __restrict__ pe,
                                                   __half* __restrict__ dst,
                                                   int plane,
                                                   int head_dim,
                                                   int sequence,
                                                   int shard_heads,
                                                   int shard_sequence,
                                                   int world_size,
                                                   bool pe_prepared,
                                                   int64_t pe_s0,
                                                   int64_t pe_s1,
                                                   int64_t pe_s2,
                                                   int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float2 x = load_recv_f16_pair(recv_flat, plane, p, s, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x.x * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x.x * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_recv_v_prep_f16_kernel(const __half* __restrict__ recv_flat,
                                                   __half* __restrict__ dst,
                                                   int plane,
                                                   int head_dim,
                                                   int sequence,
                                                   int shard_heads,
                                                   int shard_sequence,
                                                   int world_size) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;

        const float v0 = load_recv_f16(recv_flat, plane, d + 0, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const float v1 = load_recv_f16(recv_flat, plane, d + 1, s, h, head_dim, shard_heads, shard_sequence, world_size);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

__global__ void flux_sp_qkv_recv_prep_bundle_f16_kernel(const __half* __restrict__ recv_flat,
                                                        const float* __restrict__ pe,
                                                        __half* __restrict__ dst,
                                                        int head_dim,
                                                        int sequence,
                                                        int shard_heads,
                                                        int shard_sequence,
                                                        bool pe_prepared,
                                                        int64_t pe_s0,
                                                        int64_t pe_s1,
                                                        int64_t pe_s2,
                                                        int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t plane_stride = static_cast<int64_t>(head_dim) * sequence * shard_heads;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int pair = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = pair * 2;
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;

        const float pe00 = load_pe(pe, pe_prepared, s, pair, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float pe10 = load_pe(pe, pe_prepared, s, pair, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float pe01 = load_pe(pe, pe_prepared, s, pair, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const float pe11 = load_pe(pe, pe_prepared, s, pair, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);

        const float2 q = load_recv_f16_pair(recv_flat, 0, pair, s, h, head_dim, shard_heads, shard_sequence);
        const float q0 = q.x * pe00 + q.y * pe10;
        const float q1 = q.x * pe01 + q.y * pe11;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(q0, q1);

        const float2 k = load_recv_f16_pair(recv_flat, 1, pair, s, h, head_dim, shard_heads, shard_sequence);
        const float k0 = k.x * pe00 + k.y * pe10;
        const float k1 = k.x * pe01 + k.y * pe11;
        reinterpret_cast<half2*>(dst + plane_stride + dst_idx)[0] = __floats2half2_rn(k0, k1);

        reinterpret_cast<half2*>(dst + 2 * plane_stride + dst_idx)[0] =
            load_recv_f16_half2(recv_flat, 2, pair, s, h, head_dim, shard_heads, shard_sequence);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_mixed_recv_q_rope_kernel(const uint32_t* __restrict__ recv_flat,
                                                     const float* __restrict__ pe,
                                                     DstT* __restrict__ dst,
                                                     int plane,
                                                     int head_dim,
                                                     int sequence,
                                                     int shard_heads,
                                                     int shard_sequence,
                                                     int world_size,
                                                     bool pe_prepared,
                                                     int64_t pe_s0,
                                                     int64_t pe_s1,
                                                     int64_t pe_s2,
                                                     int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float x0 = load_mixed_recv(recv_flat, plane, 2 * p + 0, s, h, head_dim, shard_heads, shard_sequence);
        const float x1 = load_mixed_recv(recv_flat, plane, 2 * p + 1, s, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

__global__ void flux_sp_qkv_mixed_recv_k_rope_kernel(const uint32_t* __restrict__ recv_flat,
                                                     const float* __restrict__ pe,
                                                     __half* __restrict__ dst,
                                                     int plane,
                                                     int head_dim,
                                                     int sequence,
                                                     int shard_heads,
                                                     int shard_sequence,
                                                     int world_size,
                                                     bool pe_prepared,
                                                     int64_t pe_s0,
                                                     int64_t pe_s1,
                                                     int64_t pe_s2,
                                                     int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float x0 = load_mixed_recv(recv_flat, plane, 2 * p + 0, s, h, head_dim, shard_heads, shard_sequence);
        const float x1 = load_mixed_recv(recv_flat, plane, 2 * p + 1, s, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_mixed_recv_v_prep_kernel(const uint32_t* __restrict__ recv_flat,
                                                     __half* __restrict__ dst,
                                                     int plane,
                                                     int head_dim,
                                                     int sequence,
                                                     int shard_heads,
                                                     int shard_sequence) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;

        const float v0 = load_mixed_recv(recv_flat, plane, d + 0, s, h, head_dim, shard_heads, shard_sequence);
        const float v1 = load_mixed_recv(recv_flat, plane, d + 1, s, h, head_dim, shard_heads, shard_sequence);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_pair_recv_q_rope_kernel(const float* __restrict__ first_recv_flat,
                                                    const float* __restrict__ second_recv_flat,
                                                    const float* __restrict__ pe,
                                                    DstT* __restrict__ dst,
                                                    int plane,
                                                    int head_dim,
                                                    int first_sequence,
                                                    int sequence,
                                                    int shard_heads,
                                                    int first_shard_sequence,
                                                    int second_shard_sequence,
                                                    int world_size,
                                                    bool pe_prepared,
                                                    int64_t pe_s0,
                                                    int64_t pe_s1,
                                                    int64_t pe_s2,
                                                    int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const bool use_first = s < first_sequence;
        const float* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float x0 = load_recv(recv_flat, plane, 2 * p + 0, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const float x1 = load_recv(recv_flat, plane, 2 * p + 1, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_pair_recv_q_rope_f16_kernel(const __half* __restrict__ first_recv_flat,
                                                        const __half* __restrict__ second_recv_flat,
                                                        const float* __restrict__ pe,
                                                        DstT* __restrict__ dst,
                                                        int plane,
                                                        int head_dim,
                                                        int first_sequence,
                                                        int sequence,
                                                        int shard_heads,
                                                        int first_shard_sequence,
                                                        int second_shard_sequence,
                                                        int world_size,
                                                        bool pe_prepared,
                                                        int64_t pe_s0,
                                                        int64_t pe_s1,
                                                        int64_t pe_s2,
                                                        int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const bool use_first = s < first_sequence;
        const __half* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float2 x = load_recv_f16_pair(recv_flat, plane, p, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x.x * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x.x * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

__global__ void flux_sp_qkv_pair_recv_k_rope_kernel(const float* __restrict__ first_recv_flat,
                                                    const float* __restrict__ second_recv_flat,
                                                    const float* __restrict__ pe,
                                                    __half* __restrict__ dst,
                                                    int plane,
                                                    int head_dim,
                                                    int first_sequence,
                                                    int sequence,
                                                    int shard_heads,
                                                    int first_shard_sequence,
                                                    int second_shard_sequence,
                                                    int world_size,
                                                    bool pe_prepared,
                                                    int64_t pe_s0,
                                                    int64_t pe_s1,
                                                    int64_t pe_s2,
                                                    int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const bool use_first = s < first_sequence;
        const float* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float x0 = load_recv(recv_flat, plane, 2 * p + 0, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const float x1 = load_recv(recv_flat, plane, 2 * p + 1, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_pair_recv_k_rope_f16_kernel(const __half* __restrict__ first_recv_flat,
                                                        const __half* __restrict__ second_recv_flat,
                                                        const float* __restrict__ pe,
                                                        __half* __restrict__ dst,
                                                        int plane,
                                                        int head_dim,
                                                        int first_sequence,
                                                        int sequence,
                                                        int shard_heads,
                                                        int first_shard_sequence,
                                                        int second_shard_sequence,
                                                        int world_size,
                                                        bool pe_prepared,
                                                        int64_t pe_s0,
                                                        int64_t pe_s1,
                                                        int64_t pe_s2,
                                                        int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const bool use_first = s < first_sequence;
        const __half* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float2 x = load_recv_f16_pair(recv_flat, plane, p, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x.x * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x.x * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_pair_recv_v_prep_kernel(const float* __restrict__ first_recv_flat,
                                                    const float* __restrict__ second_recv_flat,
                                                    __half* __restrict__ dst,
                                                    int plane,
                                                    int head_dim,
                                                    int first_sequence,
                                                    int sequence,
                                                    int shard_heads,
                                                    int first_shard_sequence,
                                                    int second_shard_sequence,
                                                    int world_size) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;
        const bool use_first = s < first_sequence;
        const float* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float v0 = load_recv(recv_flat, plane, d + 0, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const float v1 = load_recv(recv_flat, plane, d + 1, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

__global__ void flux_sp_qkv_pair_recv_v_prep_f16_kernel(const __half* __restrict__ first_recv_flat,
                                                        const __half* __restrict__ second_recv_flat,
                                                        __half* __restrict__ dst,
                                                        int plane,
                                                        int head_dim,
                                                        int first_sequence,
                                                        int sequence,
                                                        int shard_heads,
                                                        int first_shard_sequence,
                                                        int second_shard_sequence,
                                                        int world_size) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;
        const bool use_first = s < first_sequence;
        const __half* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float v0 = load_recv_f16(recv_flat, plane, d + 0, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const float v1 = load_recv_f16(recv_flat, plane, d + 1, stream_seq, h, head_dim, shard_heads, shard_sequence, world_size);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_combined_pair_recv_q_rope_kernel(const float* __restrict__ recv_flat,
                                                             const float* __restrict__ pe,
                                                             DstT* __restrict__ dst,
                                                             int plane,
                                                             int head_dim,
                                                             int sequence,
                                                             int shard_heads,
                                                             int first_shard_sequence,
                                                             int second_shard_sequence,
                                                             int world_size,
                                                             bool pe_prepared,
                                                             int64_t pe_s0,
                                                             int64_t pe_s1,
                                                             int64_t pe_s2,
                                                             int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float x0 = load_combined_pair_recv(recv_flat,
                                                  plane,
                                                  2 * p + 0,
                                                  s,
                                                  h,
                                                  head_dim,
                                                  shard_heads,
                                                  world_size,
                                                  first_shard_sequence,
                                                  second_shard_sequence);
        const float x1 = load_combined_pair_recv(recv_flat,
                                                  plane,
                                                  2 * p + 1,
                                                  s,
                                                  h,
                                                  head_dim,
                                                  shard_heads,
                                                  world_size,
                                                  first_shard_sequence,
                                                  second_shard_sequence);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_combined_pair_recv_q_rope_f16_kernel(const __half* __restrict__ recv_flat,
                                                                 const float* __restrict__ pe,
                                                                 DstT* __restrict__ dst,
                                                                 int plane,
                                                                 int head_dim,
                                                                 int sequence,
                                                                 int shard_heads,
                                                                 int first_shard_sequence,
                                                                 int second_shard_sequence,
                                                                 int world_size,
                                                                 bool pe_prepared,
                                                                 int64_t pe_s0,
                                                                 int64_t pe_s1,
                                                                 int64_t pe_s2,
                                                                 int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float2 x = load_combined_pair_recv_f16_pair(recv_flat,
                                                          plane,
                                                          p,
                                                          s,
                                                          h,
                                                          head_dim,
                                                          shard_heads,
                                                          world_size,
                                                          first_shard_sequence,
                                                          second_shard_sequence);
        const float y0 = x.x * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x.x * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

__global__ void flux_sp_qkv_combined_pair_recv_k_rope_kernel(const float* __restrict__ recv_flat,
                                                             const float* __restrict__ pe,
                                                             __half* __restrict__ dst,
                                                             int plane,
                                                             int head_dim,
                                                             int sequence,
                                                             int shard_heads,
                                                             int first_shard_sequence,
                                                             int second_shard_sequence,
                                                             int world_size,
                                                             bool pe_prepared,
                                                             int64_t pe_s0,
                                                             int64_t pe_s1,
                                                             int64_t pe_s2,
                                                             int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float x0 = load_combined_pair_recv(recv_flat, plane, 2 * p + 0, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const float x1 = load_combined_pair_recv(recv_flat, plane, 2 * p + 1, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_combined_pair_recv_k_rope_f16_kernel(const __half* __restrict__ recv_flat,
                                                                 const float* __restrict__ pe,
                                                                 __half* __restrict__ dst,
                                                                 int plane,
                                                                 int head_dim,
                                                                 int sequence,
                                                                 int shard_heads,
                                                                 int first_shard_sequence,
                                                                 int second_shard_sequence,
                                                                 int world_size,
                                                                 bool pe_prepared,
                                                                 int64_t pe_s0,
                                                                 int64_t pe_s1,
                                                                 int64_t pe_s2,
                                                                 int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;

        const float2 x = load_combined_pair_recv_f16_pair(recv_flat, plane, p, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const float y0 = x.x * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x.x * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x.y * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_combined_pair_recv_v_prep_kernel(const float* __restrict__ recv_flat,
                                                             __half* __restrict__ dst,
                                                             int plane,
                                                             int head_dim,
                                                             int sequence,
                                                             int shard_heads,
                                                             int first_shard_sequence,
                                                             int second_shard_sequence,
                                                             int world_size) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;

        const float v0 = load_combined_pair_recv(recv_flat, plane, d + 0, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const float v1 = load_combined_pair_recv(recv_flat, plane, d + 1, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

__global__ void flux_sp_qkv_combined_pair_recv_v_prep_f16_kernel(const __half* __restrict__ recv_flat,
                                                                 __half* __restrict__ dst,
                                                                 int plane,
                                                                 int head_dim,
                                                                 int sequence,
                                                                 int shard_heads,
                                                                 int first_shard_sequence,
                                                                 int second_shard_sequence,
                                                                 int world_size) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;

        const float v0 = load_combined_pair_recv_f16(recv_flat, plane, d + 0, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const float v1 = load_combined_pair_recv_f16(recv_flat, plane, d + 1, s, h, head_dim, shard_heads, world_size, first_shard_sequence, second_shard_sequence);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

__global__ void flux_sp_qkv_combined_pair_recv_prep_bundle_f16_kernel(const __half* __restrict__ recv_flat,
                                                                      const float* __restrict__ pe,
                                                                      __half* __restrict__ dst,
                                                                      int head_dim,
                                                                      int sequence,
                                                                      int shard_heads,
                                                                      int first_shard_sequence,
                                                                      int second_shard_sequence,
                                                                      int world_size,
                                                                      bool pe_prepared,
                                                                      int64_t pe_s0,
                                                                      int64_t pe_s1,
                                                                      int64_t pe_s2,
                                                                      int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t plane_stride = static_cast<int64_t>(head_dim) * sequence * shard_heads;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int pair = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = pair * 2;
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;

        const float pe00 = load_pe(pe, pe_prepared, s, pair, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float pe10 = load_pe(pe, pe_prepared, s, pair, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float pe01 = load_pe(pe, pe_prepared, s, pair, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const float pe11 = load_pe(pe, pe_prepared, s, pair, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);

        const float2 q = load_combined_pair_recv_f16_pair(recv_flat,
                                                          static_cast<int>(FluxSPQKVRecvPrepPlane::Q),
                                                          pair,
                                                          s,
                                                          h,
                                                          head_dim,
                                                          shard_heads,
                                                          world_size,
                                                          first_shard_sequence,
                                                          second_shard_sequence);
        const float q0 = q.x * pe00 + q.y * pe10;
        const float q1 = q.x * pe01 + q.y * pe11;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(q0, q1);

        const float2 k = load_combined_pair_recv_f16_pair(recv_flat,
                                                          static_cast<int>(FluxSPQKVRecvPrepPlane::K),
                                                          pair,
                                                          s,
                                                          h,
                                                          head_dim,
                                                          shard_heads,
                                                          world_size,
                                                          first_shard_sequence,
                                                          second_shard_sequence);
        const float k0 = k.x * pe00 + k.y * pe10;
        const float k1 = k.x * pe01 + k.y * pe11;
        reinterpret_cast<half2*>(dst + plane_stride + dst_idx)[0] = __floats2half2_rn(k0, k1);

        reinterpret_cast<half2*>(dst + 2 * plane_stride + dst_idx)[0] =
            load_combined_pair_recv_f16_half2(recv_flat,
                                              static_cast<int>(FluxSPQKVRecvPrepPlane::V),
                                              pair,
                                              s,
                                              h,
                                              head_dim,
                                              shard_heads,
                                              world_size,
                                              first_shard_sequence,
                                              second_shard_sequence);
    }
}

template <typename DstT>
__global__ void flux_sp_qkv_pair_mixed_recv_q_rope_kernel(const uint32_t* __restrict__ first_recv_flat,
                                                          const uint32_t* __restrict__ second_recv_flat,
                                                          const float* __restrict__ pe,
                                                          DstT* __restrict__ dst,
                                                          int plane,
                                                          int head_dim,
                                                          int first_sequence,
                                                          int sequence,
                                                          int shard_heads,
                                                          int first_shard_sequence,
                                                          int second_shard_sequence,
                                                          bool pe_prepared,
                                                          int64_t pe_s0,
                                                          int64_t pe_s1,
                                                          int64_t pe_s2,
                                                          int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const bool use_first = s < first_sequence;
        const uint32_t* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float x0 = load_mixed_recv(recv_flat, plane, 2 * p + 0, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float x1 = load_mixed_recv(recv_flat, plane, 2 * p + 1, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        store_q_rope_pair(dst, dst_idx, y0, y1);
    }
}

__global__ void flux_sp_qkv_pair_mixed_recv_k_rope_kernel(const uint32_t* __restrict__ first_recv_flat,
                                                          const uint32_t* __restrict__ second_recv_flat,
                                                          const float* __restrict__ pe,
                                                          __half* __restrict__ dst,
                                                          int plane,
                                                          int head_dim,
                                                          int first_sequence,
                                                          int sequence,
                                                          int shard_heads,
                                                          int first_shard_sequence,
                                                          int second_shard_sequence,
                                                          bool pe_prepared,
                                                          int64_t pe_s0,
                                                          int64_t pe_s1,
                                                          int64_t pe_s2,
                                                          int64_t pe_s3) {
    const int half = head_dim / 2;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int p = rem % half;
        rem /= half;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const bool use_first = s < first_sequence;
        const uint32_t* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float x0 = load_mixed_recv(recv_flat, plane, 2 * p + 0, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float x1 = load_mixed_recv(recv_flat, plane, 2 * p + 1, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float y0 = x0 * load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3);
        const float y1 = x0 * load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3) +
                         x1 * load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + 2 * p;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(y0, y1);
    }
}

__global__ void flux_sp_qkv_pair_mixed_recv_v_prep_kernel(const uint32_t* __restrict__ first_recv_flat,
                                                          const uint32_t* __restrict__ second_recv_flat,
                                                          __half* __restrict__ dst,
                                                          int plane,
                                                          int head_dim,
                                                          int first_sequence,
                                                          int sequence,
                                                          int shard_heads,
                                                          int first_shard_sequence,
                                                          int second_shard_sequence) {
    const int d_head2 = head_dim / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * sequence * shard_heads;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = idx;
        const int d2 = rem % d_head2;
        rem /= d_head2;
        const int s = rem % sequence;
        rem /= sequence;
        const int h = rem;
        const int d = d2 * 2;
        const bool use_first = s < first_sequence;
        const uint32_t* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int stream_seq = use_first ? s : s - first_sequence;
        const int shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        const float v0 = load_mixed_recv(recv_flat, plane, d + 0, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const float v1 = load_mixed_recv(recv_flat, plane, d + 1, stream_seq, h, head_dim, shard_heads, shard_sequence);
        const int64_t dst_idx = (static_cast<int64_t>(h) * sequence + s) * head_dim + d;
        reinterpret_cast<half2*>(dst + dst_idx)[0] = __floats2half2_rn(v0, v1);
    }
}

} // namespace

bool ed_cuda_flux_sp_all_to_all_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM || dst->src[0] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_all_to_all_cpu_custom_op ||
        op_params.userdata == nullptr ||
        (reinterpret_cast<uintptr_t>(op_params.userdata) % alignof(FluxSPAllToAllCustomParams)) != 0) {
        return false;
    }

    const auto* params = static_cast<const FluxSPAllToAllCustomParams*>(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_all_to_all_shape_supported(dst->src[0], dst, params)) {
        return false;
    }
    return is_contiguous_1d_output(dst->src[0]) && is_contiguous_1d_output(dst);
}

bool ed_cuda_flux_sp_all_to_all_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_all_to_all_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    auto* params = static_cast<FluxSPAllToAllCustomParams*>(op_params.userdata);
    const edgedit::parallel::DataType data_type = flux_sp_cuda_all_to_all_data_type(dst);
    edgedit::parallel::Buffer input{
        dst->src[0]->data,
        static_cast<size_t>(dst->src[0]->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    edgedit::parallel::Buffer output{
        dst->data,
        static_cast<size_t>(dst->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);
    (void)params->process_group->all_to_all_async_on_stream(input,
                                                            output,
                                                            static_cast<size_t>(params->count_per_peer),
                                                            stream);
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "all_to_all",
                            "nccl",
                            dst,
                            params->process_group->local_rank(),
                            dst->ne[0],
                            dst->ne[0] * flux_sp_cuda_tensor_element_size(dst));
    }
    return true;
}

bool ed_cuda_flux_sp_all_gather_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM || dst->src[0] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_all_gather_cpu_custom_op ||
        op_params.userdata == nullptr ||
        (reinterpret_cast<uintptr_t>(op_params.userdata) % alignof(FluxSPAllGatherCustomParams)) != 0) {
        return false;
    }

    const auto* params = static_cast<const FluxSPAllGatherCustomParams*>(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_all_gather_shape_supported(dst->src[0], dst, params)) {
        return false;
    }
    return is_contiguous_1d_output(dst->src[0]) && is_contiguous_1d_output(dst);
}

bool ed_cuda_flux_sp_all_gather_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_all_gather_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    auto* params = static_cast<FluxSPAllGatherCustomParams*>(op_params.userdata);
    const edgedit::parallel::DataType data_type = flux_sp_cuda_all_to_all_data_type(dst);
    edgedit::parallel::Buffer input{
        dst->src[0]->data,
        static_cast<size_t>(dst->src[0]->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    edgedit::parallel::Buffer output{
        dst->data,
        static_cast<size_t>(dst->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);
    (void)params->process_group->all_gather_async_on_stream(input,
                                                            output,
                                                            stream);
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "all_gather",
                            "nccl",
                            dst,
                            params->process_group->local_rank(),
                            dst->ne[0],
                            dst->ne[0] * flux_sp_cuda_tensor_element_size(dst));
    }
    return true;
}

static bool flux_sp_qkv_recv_prep_dst_supported(const ggml_tensor* dst,
                                                FluxSPQKVRecvPrepPlane plane,
                                                int64_t head_dim,
                                                int64_t sequence,
                                                int64_t shard_heads) {
    return dst != nullptr &&
           is_contiguous_3d_output(dst) &&
           dst->ne[0] == head_dim &&
           dst->ne[1] == sequence &&
           dst->ne[2] == shard_heads &&
           dst->ne[3] == 1 &&
           edgedit::ggml_ext::flux_sp_qkv_recv_prep_output_type_supported(plane, dst->type);
}

bool ed_cuda_flux_sp_qkv_recv_prep_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_recv_prep_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_recv_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_recv_prep_params_valid(params)) {
        return false;
    }
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    if (!edgedit::ggml_ext::flux_sp_qkv_recv_prep_shape_supported(recv_flat,
                                                                   pe,
                                                                   plane,
                                                                   params.world_size,
                                                                   params.heads,
                                                                   params.head_dim)) {
        return false;
    }
    if (!is_contiguous_3d_output(dst)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (static_cast<int64_t>(params.head_dim) * 3 * shard_heads * params.world_size);
    const int64_t sequence = shard_sequence * params.world_size;
    return flux_sp_qkv_recv_prep_dst_supported(dst, plane, params.head_dim, sequence, shard_heads);
}

bool ed_cuda_flux_sp_qkv_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_recv_prep_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_recv_prep_params_from_userdata(op_params.userdata);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    const bool recv_f16 = recv_flat->type == GGML_TYPE_F16;
    const bool dst_f16 = dst->type == GGML_TYPE_F16;

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int shard_sequence = static_cast<int>(recv_flat->ne[0] / (static_cast<int64_t>(head_dim) * 3 * shard_heads * world_size));
    const int sequence = shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = (plane == FluxSPQKVRecvPrepPlane::V) ?
                              static_cast<int64_t>(head_dim / 2) * sequence * shard_heads :
                              static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    if (plane == FluxSPQKVRecvPrepPlane::Q && recv_f16) {
        if (dst_f16) {
            flux_sp_qkv_recv_q_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_recv_q_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::Q) {
        if (dst_f16) {
            flux_sp_qkv_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::K && recv_f16) {
        flux_sp_qkv_recv_k_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else if (plane == FluxSPQKVRecvPrepPlane::K) {
        flux_sp_qkv_recv_k_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else if (recv_f16) {
        flux_sp_qkv_recv_v_prep_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            shard_sequence,
            world_size);
    } else {
        flux_sp_qkv_recv_v_prep_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            shard_sequence,
            world_size);
    }
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_recv_prep",
                            flux_sp_profile_plane_name(plane),
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}

bool ed_cuda_flux_sp_qkv_recv_prep_bundle_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_recv_prep_bundle_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVRecvPrepBundleCustomParams params =
        edgedit::ggml_ext::flux_sp_qkv_recv_prep_bundle_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_recv_prep_bundle_params_valid(params)) {
        return false;
    }

    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    if (!edgedit::ggml_ext::flux_sp_qkv_recv_prep_bundle_shape_supported(recv_flat,
                                                                          pe,
                                                                          params.world_size,
                                                                          params.heads,
                                                                          params.head_dim)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F16 || !is_contiguous_4d_output(dst)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (static_cast<int64_t>(params.head_dim) * 3 * shard_heads * params.world_size);
    const int64_t sequence = shard_sequence * params.world_size;
    return dst->ne[0] == params.head_dim &&
           dst->ne[1] == sequence &&
           dst->ne[2] == shard_heads &&
           dst->ne[3] == 3;
}

bool ed_cuda_flux_sp_qkv_recv_prep_bundle_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_recv_prep_bundle_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVRecvPrepBundleCustomParams params =
        edgedit::ggml_ext::flux_sp_qkv_recv_prep_bundle_params_from_userdata(op_params.userdata);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int shard_sequence = static_cast<int>(recv_flat->ne[0] / (static_cast<int64_t>(head_dim) * 3 * shard_heads * world_size));
    const int sequence = shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    flux_sp_qkv_recv_prep_bundle_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const __half*>(recv_flat->data),
        static_cast<const float*>(pe->data),
        static_cast<__half*>(dst->data),
        head_dim,
        sequence,
        shard_heads,
        shard_sequence,
        pe_prepared,
        elem_stride(pe, 0),
        elem_stride(pe, 1),
        elem_stride(pe, 2),
        elem_stride(pe, 3));

    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_recv_prep_bundle",
                            "qkv",
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}

bool ed_cuda_flux_sp_qkv_mixed_recv_prep_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVMixedRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_params_valid(params)) {
        return false;
    }
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    if (!edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_shape_supported(recv_flat,
                                                                         pe,
                                                                         plane,
                                                                         params.world_size,
                                                                         params.heads,
                                                                         params.head_dim)) {
        return false;
    }
    if (!is_contiguous_3d_output(dst)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (static_cast<int64_t>(params.head_dim) * 2 * shard_heads * params.world_size);
    const int64_t sequence = shard_sequence * params.world_size;
    return flux_sp_qkv_recv_prep_dst_supported(dst, plane, params.head_dim, sequence, shard_heads);
}

bool ed_cuda_flux_sp_qkv_mixed_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_mixed_recv_prep_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVMixedRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_params_from_userdata(op_params.userdata);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    const bool dst_f16 = dst->type == GGML_TYPE_F16;

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int shard_sequence = static_cast<int>(recv_flat->ne[0] / (static_cast<int64_t>(head_dim) * 2 * shard_heads * world_size));
    const int sequence = shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    if (plane == FluxSPQKVRecvPrepPlane::Q) {
        if (dst_f16) {
            flux_sp_qkv_mixed_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const uint32_t*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_mixed_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const uint32_t*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::K) {
        flux_sp_qkv_mixed_recv_k_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else {
        flux_sp_qkv_mixed_recv_v_prep_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            shard_sequence);
    }
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_mixed_recv_prep",
                            flux_sp_profile_plane_name(plane),
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}

bool ed_cuda_flux_sp_qkv_pair_mixed_recv_prep_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVPairMixedRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_params_valid(params)) {
        return false;
    }
    const ggml_tensor* first_recv_flat = dst->src[0];
    const ggml_tensor* second_recv_flat = dst->src[1];
    const ggml_tensor* pe = dst->src[2];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    if (!edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_shape_supported(first_recv_flat,
                                                                              second_recv_flat,
                                                                              pe,
                                                                              plane,
                                                                              params.world_size,
                                                                              params.heads,
                                                                              params.head_dim)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t denom = static_cast<int64_t>(params.head_dim) * 2 * shard_heads * params.world_size;
    const int64_t sequence = (first_recv_flat->ne[0] / denom + second_recv_flat->ne[0] / denom) * params.world_size;
    return flux_sp_qkv_recv_prep_dst_supported(dst, plane, params.head_dim, sequence, shard_heads);
}

bool ed_cuda_flux_sp_qkv_pair_mixed_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_pair_mixed_recv_prep_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVPairMixedRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_params_from_userdata(op_params.userdata);
    const ggml_tensor* first_recv_flat = dst->src[0];
    const ggml_tensor* second_recv_flat = dst->src[1];
    const ggml_tensor* pe = dst->src[2];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    const bool dst_f16 = dst->type == GGML_TYPE_F16;

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int denom = head_dim * 2 * shard_heads * world_size;
    const int first_shard_sequence = static_cast<int>(first_recv_flat->ne[0] / denom);
    const int second_shard_sequence = static_cast<int>(second_recv_flat->ne[0] / denom);
    const int first_sequence = first_shard_sequence * world_size;
    const int sequence = first_sequence + second_shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    if (plane == FluxSPQKVRecvPrepPlane::Q) {
        if (dst_f16) {
            flux_sp_qkv_pair_mixed_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const uint32_t*>(first_recv_flat->data),
                static_cast<const uint32_t*>(second_recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                first_sequence,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_pair_mixed_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const uint32_t*>(first_recv_flat->data),
                static_cast<const uint32_t*>(second_recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                first_sequence,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::K) {
        flux_sp_qkv_pair_mixed_recv_k_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(first_recv_flat->data),
            static_cast<const uint32_t*>(second_recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            first_sequence,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else {
        flux_sp_qkv_pair_mixed_recv_v_prep_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(first_recv_flat->data),
            static_cast<const uint32_t*>(second_recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            first_sequence,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence);
    }
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_pair_mixed_recv_prep",
                            flux_sp_profile_plane_name(plane),
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}

bool ed_cuda_flux_sp_qkv_pair_recv_prep_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVPairRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_params_valid(params)) {
        return false;
    }
    const ggml_tensor* first_recv_flat = dst->src[0];
    const ggml_tensor* second_recv_flat = dst->src[1];
    const ggml_tensor* pe = dst->src[2];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    if (!edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_shape_supported(first_recv_flat,
                                                                        second_recv_flat,
                                                                        pe,
                                                                        plane,
                                                                        params.world_size,
                                                                        params.heads,
                                                                        params.head_dim)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t denom = static_cast<int64_t>(params.head_dim) * 3 * shard_heads * params.world_size;
    const int64_t sequence = (first_recv_flat->ne[0] / denom + second_recv_flat->ne[0] / denom) * params.world_size;
    return flux_sp_qkv_recv_prep_dst_supported(dst, plane, params.head_dim, sequence, shard_heads);
}

bool ed_cuda_flux_sp_qkv_pair_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_pair_recv_prep_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVPairRecvPrepCustomParams params = edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_params_from_userdata(op_params.userdata);
    const ggml_tensor* first_recv_flat = dst->src[0];
    const ggml_tensor* second_recv_flat = dst->src[1];
    const ggml_tensor* pe = dst->src[2];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    const bool recv_f16 = first_recv_flat->type == GGML_TYPE_F16;
    const bool dst_f16 = dst->type == GGML_TYPE_F16;

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int denom = head_dim * 3 * shard_heads * world_size;
    const int first_shard_sequence = static_cast<int>(first_recv_flat->ne[0] / denom);
    const int second_shard_sequence = static_cast<int>(second_recv_flat->ne[0] / denom);
    const int first_sequence = first_shard_sequence * world_size;
    const int sequence = first_sequence + second_shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    if (plane == FluxSPQKVRecvPrepPlane::Q && recv_f16) {
        if (dst_f16) {
            flux_sp_qkv_pair_recv_q_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(first_recv_flat->data),
                static_cast<const __half*>(second_recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                first_sequence,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_pair_recv_q_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(first_recv_flat->data),
                static_cast<const __half*>(second_recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                first_sequence,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::Q) {
        if (dst_f16) {
            flux_sp_qkv_pair_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(first_recv_flat->data),
                static_cast<const float*>(second_recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                first_sequence,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_pair_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(first_recv_flat->data),
                static_cast<const float*>(second_recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                first_sequence,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::K && recv_f16) {
        flux_sp_qkv_pair_recv_k_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(first_recv_flat->data),
            static_cast<const __half*>(second_recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            first_sequence,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else if (plane == FluxSPQKVRecvPrepPlane::K) {
        flux_sp_qkv_pair_recv_k_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(first_recv_flat->data),
            static_cast<const float*>(second_recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            first_sequence,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else if (recv_f16) {
        flux_sp_qkv_pair_recv_v_prep_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(first_recv_flat->data),
            static_cast<const __half*>(second_recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            first_sequence,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size);
    } else {
        flux_sp_qkv_pair_recv_v_prep_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(first_recv_flat->data),
            static_cast<const float*>(second_recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            first_sequence,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size);
    }
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_pair_recv_prep",
                            flux_sp_profile_plane_name(plane),
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVCombinedPairRecvPrepCustomParams params =
        edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_params_valid(params)) {
        return false;
    }
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    if (!edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_shape_supported(recv_flat,
                                                                                 pe,
                                                                                 plane,
                                                                                 params.world_size,
                                                                                 params.heads,
                                                                                 params.head_dim,
                                                                                 params.first_shard_sequence)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t denom = static_cast<int64_t>(params.head_dim) * 3 * shard_heads * params.world_size;
    const int64_t sequence = (recv_flat->ne[0] / denom) * params.world_size;
    return flux_sp_qkv_recv_prep_dst_supported(dst, plane, params.head_dim, sequence, shard_heads);
}

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_combined_pair_recv_prep_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVCombinedPairRecvPrepCustomParams params =
        edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_params_from_userdata(op_params.userdata);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    const bool recv_f16 = recv_flat->type == GGML_TYPE_F16;
    const bool dst_f16 = dst->type == GGML_TYPE_F16;

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int denom = head_dim * 3 * shard_heads * world_size;
    const int combined_shard_sequence = static_cast<int>(recv_flat->ne[0] / denom);
    const int first_shard_sequence = params.first_shard_sequence;
    const int second_shard_sequence = combined_shard_sequence - first_shard_sequence;
    const int sequence = combined_shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (second_shard_sequence <= 0 || (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half))) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    if (plane == FluxSPQKVRecvPrepPlane::Q && recv_f16) {
        if (dst_f16) {
            flux_sp_qkv_combined_pair_recv_q_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_combined_pair_recv_q_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::Q) {
        if (dst_f16) {
            flux_sp_qkv_combined_pair_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<__half*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        } else {
            flux_sp_qkv_combined_pair_recv_q_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(recv_flat->data),
                static_cast<const float*>(pe->data),
                static_cast<float*>(dst->data),
                params.plane,
                head_dim,
                sequence,
                shard_heads,
                first_shard_sequence,
                second_shard_sequence,
                world_size,
                pe_prepared,
                elem_stride(pe, 0),
                elem_stride(pe, 1),
                elem_stride(pe, 2),
                elem_stride(pe, 3));
        }
    } else if (plane == FluxSPQKVRecvPrepPlane::K && recv_f16) {
        flux_sp_qkv_combined_pair_recv_k_rope_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else if (plane == FluxSPQKVRecvPrepPlane::K) {
        flux_sp_qkv_combined_pair_recv_k_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(recv_flat->data),
            static_cast<const float*>(pe->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size,
            pe_prepared,
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    } else if (recv_f16) {
        flux_sp_qkv_combined_pair_recv_v_prep_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size);
    } else {
        flux_sp_qkv_combined_pair_recv_v_prep_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(recv_flat->data),
            static_cast<__half*>(dst->data),
            params.plane,
            head_dim,
            sequence,
            shard_heads,
            first_shard_sequence,
            second_shard_sequence,
            world_size);
    }
    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_combined_pair_recv_prep",
                            flux_sp_profile_plane_name(plane),
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_bundle_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    if (op_params.fun != edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_bundle_cpu_custom_op) {
        return false;
    }
    const FluxSPQKVCombinedPairRecvPrepBundleCustomParams params =
        edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_bundle_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_bundle_params_valid(params)) {
        return false;
    }

    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    if (!edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_bundle_shape_supported(recv_flat,
                                                                                        pe,
                                                                                        params.world_size,
                                                                                        params.heads,
                                                                                        params.head_dim,
                                                                                        params.first_shard_sequence)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F16 || !is_contiguous_4d_output(dst)) {
        return false;
    }
    const int64_t shard_heads = params.heads / params.world_size;
    const int64_t denom = static_cast<int64_t>(params.head_dim) * 3 * shard_heads * params.world_size;
    const int64_t sequence = (recv_flat->ne[0] / denom) * params.world_size;
    return dst->ne[0] == params.head_dim &&
           dst->ne[1] == sequence &&
           dst->ne[2] == shard_heads &&
           dst->ne[3] == 3;
}

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_bundle_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream) {
    if (!ed_cuda_flux_sp_qkv_combined_pair_recv_prep_bundle_custom_supported(dst)) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const FluxSPQKVCombinedPairRecvPrepBundleCustomParams params =
        edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_bundle_params_from_userdata(op_params.userdata);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];

    const int head_dim = params.head_dim;
    const int world_size = params.world_size;
    const int shard_heads = params.heads / params.world_size;
    const int denom = head_dim * 3 * shard_heads * world_size;
    const int combined_shard_sequence = static_cast<int>(recv_flat->ne[0] / denom);
    const int first_shard_sequence = params.first_shard_sequence;
    const int second_shard_sequence = combined_shard_sequence - first_shard_sequence;
    const int sequence = combined_shard_sequence * world_size;
    const int half = head_dim / 2;
    const bool pe_prepared = pe_is_prepared_layout(pe, sequence, half);
    if (second_shard_sequence <= 0 || (!pe_prepared && !pe_is_matrix_layout(pe, sequence, half))) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(half) * sequence * shard_heads;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool profile = flux_sp_profile_begin(cuda_stream, &profile_start, &profile_stop);

    flux_sp_qkv_combined_pair_recv_prep_bundle_f16_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const __half*>(recv_flat->data),
        static_cast<const float*>(pe->data),
        static_cast<__half*>(dst->data),
        head_dim,
        sequence,
        shard_heads,
        first_shard_sequence,
        second_shard_sequence,
        world_size,
        pe_prepared,
        elem_stride(pe, 0),
        elem_stride(pe, 1),
        elem_stride(pe, 2),
        elem_stride(pe, 3));

    if (profile) {
        flux_sp_profile_end(cuda_stream,
                            profile_start,
                            profile_stop,
                            "qkv_combined_pair_recv_prep_bundle",
                            "qkv",
                            dst,
                            flux_sp_profile_rank_fallback(),
                            ggml_nelements(dst),
                            ggml_nelements(dst) * static_cast<int64_t>(ggml_type_size(dst->type) / ggml_blck_size(dst->type)));
    }
    return true;
}
