#include "ed_cuda_attention_v_prep.h"

#include "backend/ggml/ed_ggml_attention_ext.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace {

using edgedit::ggml_ext::AttentionVPrepCustomParams;
using edgedit::ggml_ext::AttentionPairPackCustomParams;
using edgedit::ggml_ext::AttentionQKVPairPackCustomParams;

static inline int64_t elem_stride(const ggml_tensor* t, int dim) {
    return t->nb[dim] / ggml_type_size(t->type);
}

static bool is_contiguous_3d_f16_output(const ggml_tensor* t) {
    return t->type == GGML_TYPE_F16 &&
           t->nb[0] == sizeof(__half) &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(__half) &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * sizeof(__half);
}

static bool is_aligned_to(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

static bool can_vec2_load_f32(const ggml_tensor* t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           elem_stride(t, 0) == 1 &&
           (elem_stride(t, 1) % 2) == 0 &&
           is_aligned_to(t->data, alignof(float2));
}

static bool can_vec2_store_f16(const ggml_tensor* t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F16 &&
           (t->ne[0] % 2) == 0 &&
           is_aligned_to(t->data, alignof(half2));
}

__global__ void attention_v_prep_f16_kernel(const float* v,
                                            __half* dst,
                                            int v_is_seq_major,
                                            int d_head,
                                            int seq,
                                            int n_head,
                                            int batch,
                                            int64_t v_s0,
                                            int64_t v_s1,
                                            int64_t v_s2,
                                            int64_t v_s3) {
    const int64_t total = static_cast<int64_t>(d_head) * seq * n_head * batch;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int d = idx % d_head;
    idx /= d_head;
    const int s = idx % seq;
    idx /= seq;
    const int h_total = idx;
    const int h = h_total % n_head;
    const int b = h_total / n_head;

    const float value = v_is_seq_major ?
                            v[d * v_s0 + s * v_s1 + h * v_s2 + b * v_s3] :
                            v[d * v_s0 + h * v_s1 + s * v_s2 + b * v_s3];
    const int64_t dst_idx = (static_cast<int64_t>(h_total) * seq + s) * d_head + d;
    dst[dst_idx] = __float2half_rn(value);
}

__global__ void attention_v_prep_f16_vec2_kernel(const float* v,
                                                 __half* dst,
                                                 int v_is_seq_major,
                                                 int d_head,
                                                 int seq,
                                                 int n_head,
                                                 int batch,
                                                 int64_t v_s1,
                                                 int64_t v_s2,
                                                 int64_t v_s3) {
    const int d_head2 = d_head / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * seq * n_head * batch;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int d2 = idx % d_head2;
    idx /= d_head2;
    const int s = idx % seq;
    idx /= seq;
    const int h_total = idx;
    const int h = h_total % n_head;
    const int b = h_total / n_head;
    const int d = d2 * 2;

    const int64_t src_idx = v_is_seq_major ?
                                static_cast<int64_t>(d) + static_cast<int64_t>(s) * v_s1 + static_cast<int64_t>(h) * v_s2 + static_cast<int64_t>(b) * v_s3 :
                                static_cast<int64_t>(d) + static_cast<int64_t>(h) * v_s1 + static_cast<int64_t>(s) * v_s2 + static_cast<int64_t>(b) * v_s3;
    const float2 value = reinterpret_cast<const float2*>(v + src_idx)[0];
    const int64_t dst_idx = (static_cast<int64_t>(h_total) * seq + s) * d_head + d;
    reinterpret_cast<half2*>(dst + dst_idx)[0] = __float22half2_rn(value);
}

__global__ void attention_pair_pack_f16_kernel(const float* first,
                                               const float* second,
                                               __half* dst,
                                               int d_head,
                                               int first_seq,
                                               int second_seq,
                                               int n_head,
                                               int batch,
                                               int64_t first_s0,
                                               int64_t first_s1,
                                               int64_t first_s2,
                                               int64_t second_s0,
                                               int64_t second_s1,
                                               int64_t second_s2) {
    const int total_seq = first_seq + second_seq;
    const int64_t total = static_cast<int64_t>(d_head) * total_seq * n_head * batch;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int d = idx % d_head;
    idx /= d_head;
    const int s = idx % total_seq;
    idx /= total_seq;
    const int h = idx % n_head;
    const int b = idx / n_head;
    const int c = h * d_head + d;

    const bool use_first = s < first_seq;
    const int src_s = use_first ? s : s - first_seq;
    const float* src = use_first ? first : second;
    const int64_t s0 = use_first ? first_s0 : second_s0;
    const int64_t s1 = use_first ? first_s1 : second_s1;
    const int64_t s2 = use_first ? first_s2 : second_s2;
    dst[((static_cast<int64_t>(b) * n_head + h) * total_seq + s) * d_head + d] =
        __float2half_rn(src[static_cast<int64_t>(c) * s0 + static_cast<int64_t>(src_s) * s1 + static_cast<int64_t>(b) * s2]);
}

__global__ void attention_pair_pack_f16_vec2_kernel(const float* first,
                                                    const float* second,
                                                    __half* dst,
                                                    int d_head,
                                                    int first_seq,
                                                    int second_seq,
                                                    int n_head,
                                                    int batch,
                                                    int64_t first_s1,
                                                    int64_t first_s2,
                                                    int64_t second_s1,
                                                    int64_t second_s2) {
    const int total_seq = first_seq + second_seq;
    const int d_head2 = d_head / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * total_seq * n_head * batch;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int d2 = idx % d_head2;
    idx /= d_head2;
    const int s = idx % total_seq;
    idx /= total_seq;
    const int h = idx % n_head;
    const int b = idx / n_head;
    const int c = h * d_head + d2 * 2;

    const bool use_first = s < first_seq;
    const int src_s = use_first ? s : s - first_seq;
    const float* src = use_first ? first : second;
    const int64_t s1 = use_first ? first_s1 : second_s1;
    const int64_t s2 = use_first ? first_s2 : second_s2;
    const float2 value = reinterpret_cast<const float2*>(
        src + static_cast<int64_t>(c) + static_cast<int64_t>(src_s) * s1 + static_cast<int64_t>(b) * s2)[0];

    const int64_t dst_idx = ((static_cast<int64_t>(b) * n_head + h) * total_seq + s) * d_head + d2 * 2;
    reinterpret_cast<half2*>(dst + dst_idx)[0] = __float22half2_rn(value);
}

__device__ __forceinline__ float attention_qkv_pair_pack_load(const float* src,
                                                              int c,
                                                              int s,
                                                              int64_t s0,
                                                              int64_t s1) {
    return src[static_cast<int64_t>(c) * s0 + static_cast<int64_t>(s) * s1];
}

__global__ void attention_qkv_pair_pack_f16_kernel(const float* q_first,
                                                   const float* k_first,
                                                   const float* v_first,
                                                   const float* q_second,
                                                   const float* k_second,
                                                   const float* v_second,
                                                   __half* dst,
                                                   int d_head,
                                                   int first_seq,
                                                   int second_seq,
                                                   int n_head,
                                                   int64_t q_first_s0,
                                                   int64_t q_first_s1,
                                                   int64_t k_first_s0,
                                                   int64_t k_first_s1,
                                                   int64_t v_first_s0,
                                                   int64_t v_first_s1,
                                                   int64_t q_second_s0,
                                                   int64_t q_second_s1,
                                                   int64_t k_second_s0,
                                                   int64_t k_second_s1,
                                                   int64_t v_second_s0,
                                                   int64_t v_second_s1) {
    const int total_seq = first_seq + second_seq;
    const int64_t total = static_cast<int64_t>(d_head) * total_seq * n_head * 3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int d = idx % d_head;
    idx /= d_head;
    const int s = idx % total_seq;
    idx /= total_seq;
    const int h = idx % n_head;
    const int plane = idx / n_head;
    const int c = h * d_head + d;
    const bool use_first = s < first_seq;
    const int src_s = use_first ? s : s - first_seq;

    float value = 0.0f;
    if (plane == 0) {
        value = use_first ?
                    attention_qkv_pair_pack_load(q_first, c, src_s, q_first_s0, q_first_s1) :
                    attention_qkv_pair_pack_load(q_second, c, src_s, q_second_s0, q_second_s1);
    } else if (plane == 1) {
        value = use_first ?
                    attention_qkv_pair_pack_load(k_first, c, src_s, k_first_s0, k_first_s1) :
                    attention_qkv_pair_pack_load(k_second, c, src_s, k_second_s0, k_second_s1);
    } else {
        value = use_first ?
                    attention_qkv_pair_pack_load(v_first, c, src_s, v_first_s0, v_first_s1) :
                    attention_qkv_pair_pack_load(v_second, c, src_s, v_second_s0, v_second_s1);
    }

    dst[((static_cast<int64_t>(plane) * n_head + h) * total_seq + s) * d_head + d] = __float2half_rn(value);
}

__global__ void attention_qkv_pair_pack_f16_vec2_kernel(const float* q_first,
                                                        const float* k_first,
                                                        const float* v_first,
                                                        const float* q_second,
                                                        const float* k_second,
                                                        const float* v_second,
                                                        __half* dst,
                                                        int d_head,
                                                        int first_seq,
                                                        int second_seq,
                                                        int n_head,
                                                        int64_t q_first_s1,
                                                        int64_t k_first_s1,
                                                        int64_t v_first_s1,
                                                        int64_t q_second_s1,
                                                        int64_t k_second_s1,
                                                        int64_t v_second_s1) {
    const int total_seq = first_seq + second_seq;
    const int d_head2 = d_head / 2;
    const int64_t total = static_cast<int64_t>(d_head2) * total_seq * n_head;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int d2 = idx % d_head2;
    idx /= d_head2;
    const int s = idx % total_seq;
    idx /= total_seq;
    const int h = idx;
    const int c = h * d_head + d2 * 2;
    const bool use_first = s < first_seq;
    const int src_s = use_first ? s : s - first_seq;

    const float* q_src = use_first ? q_first : q_second;
    const float* k_src = use_first ? k_first : k_second;
    const float* v_src = use_first ? v_first : v_second;
    const int64_t q_s1 = use_first ? q_first_s1 : q_second_s1;
    const int64_t k_s1 = use_first ? k_first_s1 : k_second_s1;
    const int64_t v_s1 = use_first ? v_first_s1 : v_second_s1;

    const float2 qv = reinterpret_cast<const float2*>(q_src + static_cast<int64_t>(c) + static_cast<int64_t>(src_s) * q_s1)[0];
    const float2 kv = reinterpret_cast<const float2*>(k_src + static_cast<int64_t>(c) + static_cast<int64_t>(src_s) * k_s1)[0];
    const float2 vv = reinterpret_cast<const float2*>(v_src + static_cast<int64_t>(c) + static_cast<int64_t>(src_s) * v_s1)[0];

    const int64_t dst_idx = (static_cast<int64_t>(h) * total_seq + s) * d_head + d2 * 2;
    const int64_t plane_stride = static_cast<int64_t>(n_head) * total_seq * d_head;
    reinterpret_cast<half2*>(dst + dst_idx)[0] = __float22half2_rn(qv);
    reinterpret_cast<half2*>(dst + plane_stride + dst_idx)[0] = __float22half2_rn(kv);
    reinterpret_cast<half2*>(dst + plane_stride * 2 + dst_idx)[0] = __float22half2_rn(vv);
}

} // namespace

bool ed_cuda_attention_v_prep_custom_supported(const ggml_tensor * dst) {
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

    const AttentionVPrepCustomParams params = edgedit::ggml_ext::attention_v_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::attention_v_prep_params_valid(params)) {
        return false;
    }

    const ggml_tensor* v = dst->src[0];
    const bool v_is_seq_major = params.v_is_seq_major != 0;
    if (!edgedit::ggml_ext::attention_v_prep_shape_supported(v, v_is_seq_major)) {
        return false;
    }
    if (!is_contiguous_3d_f16_output(dst)) {
        return false;
    }

    const int d_head = static_cast<int>(v->ne[0]);
    const int seq = static_cast<int>(v_is_seq_major ? v->ne[1] : v->ne[2]);
    const int n_head = static_cast<int>(v_is_seq_major ? v->ne[2] : v->ne[1]);
    const int batch = static_cast<int>(v->ne[3]);
    return dst->ne[0] == d_head &&
           dst->ne[1] == seq &&
           dst->ne[2] == static_cast<int64_t>(n_head) * batch;
}

bool ed_cuda_attention_v_prep_custom_compute(ggml_tensor * dst, ed_cuda_attention_v_prep_stream_t stream) {
    if (!ed_cuda_attention_v_prep_custom_supported(dst)) {
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

    const AttentionVPrepCustomParams params = edgedit::ggml_ext::attention_v_prep_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::attention_v_prep_params_valid(params)) {
        return false;
    }

    const ggml_tensor* v = dst->src[0];
    const bool v_is_seq_major = params.v_is_seq_major != 0;
    const int d_head = static_cast<int>(v->ne[0]);
    const int seq = static_cast<int>(v_is_seq_major ? v->ne[1] : v->ne[2]);
    const int n_head = static_cast<int>(v_is_seq_major ? v->ne[2] : v->ne[1]);
    const int batch = static_cast<int>(v->ne[3]);

    const int64_t total = static_cast<int64_t>(d_head) * seq * n_head * batch;
    constexpr int threads = 256;
    if (can_vec2_load_f32(v) && can_vec2_store_f16(dst)) {
        const int64_t total_vec = total / 2;
        const int blocks = static_cast<int>((total_vec + threads - 1) / threads);
        attention_v_prep_f16_vec2_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(v->data),
            static_cast<__half*>(dst->data),
            v_is_seq_major ? 1 : 0,
            d_head,
            seq,
            n_head,
            batch,
            elem_stride(v, 1),
            elem_stride(v, 2),
            elem_stride(v, 3));
    } else {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        attention_v_prep_f16_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(v->data),
            static_cast<__half*>(dst->data),
            v_is_seq_major ? 1 : 0,
            d_head,
            seq,
            n_head,
            batch,
            elem_stride(v, 0),
            elem_stride(v, 1),
            elem_stride(v, 2),
            elem_stride(v, 3));
    }
    return true;
}

bool ed_cuda_attention_pair_pack_custom_supported(const ggml_tensor * dst) {
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

    const AttentionPairPackCustomParams params = edgedit::ggml_ext::attention_pair_pack_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::attention_pair_pack_params_valid(params)) {
        return false;
    }

    const ggml_tensor* first = dst->src[0];
    const ggml_tensor* second = dst->src[1];
    if (!edgedit::ggml_ext::attention_pair_pack_shape_supported(first, second, params.n_head)) {
        return false;
    }
    if (!is_contiguous_3d_f16_output(dst)) {
        return false;
    }

    const int64_t d_head = first->ne[0] / params.n_head;
    return dst->ne[0] == d_head &&
           dst->ne[1] == first->ne[1] + second->ne[1] &&
           dst->ne[2] == params.n_head &&
           dst->ne[3] == first->ne[2];
}

bool ed_cuda_attention_pair_pack_custom_compute(ggml_tensor * dst, ed_cuda_attention_v_prep_stream_t stream) {
    if (!ed_cuda_attention_pair_pack_custom_supported(dst)) {
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

    const AttentionPairPackCustomParams params = edgedit::ggml_ext::attention_pair_pack_params_from_userdata(op_params.userdata);
    const ggml_tensor* first = dst->src[0];
    const ggml_tensor* second = dst->src[1];
    const int d_head = static_cast<int>(first->ne[0] / params.n_head);
    const int first_seq = static_cast<int>(first->ne[1]);
    const int second_seq = static_cast<int>(second->ne[1]);
    const int n_head = static_cast<int>(params.n_head);
    const int batch = static_cast<int>(first->ne[2]);

    const int64_t total = static_cast<int64_t>(d_head) * (first_seq + second_seq) * n_head * batch;
    constexpr int threads = 256;
    if (can_vec2_load_f32(first) && can_vec2_load_f32(second) && can_vec2_store_f16(dst)) {
        const int64_t total_vec = total / 2;
        const int blocks = static_cast<int>((total_vec + threads - 1) / threads);
        attention_pair_pack_f16_vec2_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(first->data),
            static_cast<const float*>(second->data),
            static_cast<__half*>(dst->data),
            d_head,
            first_seq,
            second_seq,
            n_head,
            batch,
            elem_stride(first, 1),
            elem_stride(first, 2),
            elem_stride(second, 1),
            elem_stride(second, 2));
    } else {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        attention_pair_pack_f16_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(first->data),
            static_cast<const float*>(second->data),
            static_cast<__half*>(dst->data),
            d_head,
            first_seq,
            second_seq,
            n_head,
            batch,
            elem_stride(first, 0),
            elem_stride(first, 1),
            elem_stride(first, 2),
            elem_stride(second, 0),
            elem_stride(second, 1),
            elem_stride(second, 2));
    }
    return true;
}

bool ed_cuda_attention_qkv_pair_pack_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (dst->src[i] == nullptr) {
            return false;
        }
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    memcpy(&op_params, dst->op_params, sizeof(op_params));

    const AttentionQKVPairPackCustomParams params = edgedit::ggml_ext::attention_qkv_pair_pack_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::attention_qkv_pair_pack_params_valid(params)) {
        return false;
    }

    const ggml_tensor* q_first = dst->src[0];
    const ggml_tensor* k_first = dst->src[1];
    const ggml_tensor* v_first = dst->src[2];
    const ggml_tensor* q_second = dst->src[3];
    const ggml_tensor* k_second = dst->src[4];
    const ggml_tensor* v_second = dst->src[5];
    if (!edgedit::ggml_ext::attention_qkv_pair_pack_shape_supported(q_first, k_first, v_first, q_second, k_second, v_second, params.n_head)) {
        return false;
    }
    if (!is_contiguous_3d_f16_output(dst)) {
        return false;
    }

    const int64_t d_head = q_first->ne[0] / params.n_head;
    return dst->ne[0] == d_head &&
           dst->ne[1] == q_first->ne[1] + q_second->ne[1] &&
           dst->ne[2] == params.n_head &&
           dst->ne[3] == 3;
}

bool ed_cuda_attention_qkv_pair_pack_custom_compute(ggml_tensor * dst, ed_cuda_attention_v_prep_stream_t stream) {
    if (!ed_cuda_attention_qkv_pair_pack_custom_supported(dst)) {
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

    const AttentionQKVPairPackCustomParams params = edgedit::ggml_ext::attention_qkv_pair_pack_params_from_userdata(op_params.userdata);
    const ggml_tensor* q_first = dst->src[0];
    const ggml_tensor* k_first = dst->src[1];
    const ggml_tensor* v_first = dst->src[2];
    const ggml_tensor* q_second = dst->src[3];
    const ggml_tensor* k_second = dst->src[4];
    const ggml_tensor* v_second = dst->src[5];
    const int d_head = static_cast<int>(q_first->ne[0] / params.n_head);
    const int first_seq = static_cast<int>(q_first->ne[1]);
    const int second_seq = static_cast<int>(q_second->ne[1]);
    const int n_head = static_cast<int>(params.n_head);

    const int64_t total = static_cast<int64_t>(d_head) * (first_seq + second_seq) * n_head * 3;
    constexpr int threads = 256;
    if (can_vec2_load_f32(q_first) &&
        can_vec2_load_f32(k_first) &&
        can_vec2_load_f32(v_first) &&
        can_vec2_load_f32(q_second) &&
        can_vec2_load_f32(k_second) &&
        can_vec2_load_f32(v_second) &&
        can_vec2_store_f16(dst)) {
        const int64_t total_vec = total / 6;
        const int blocks = static_cast<int>((total_vec + threads - 1) / threads);
        attention_qkv_pair_pack_f16_vec2_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(q_first->data),
            static_cast<const float*>(k_first->data),
            static_cast<const float*>(v_first->data),
            static_cast<const float*>(q_second->data),
            static_cast<const float*>(k_second->data),
            static_cast<const float*>(v_second->data),
            static_cast<__half*>(dst->data),
            d_head,
            first_seq,
            second_seq,
            n_head,
            elem_stride(q_first, 1),
            elem_stride(k_first, 1),
            elem_stride(v_first, 1),
            elem_stride(q_second, 1),
            elem_stride(k_second, 1),
            elem_stride(v_second, 1));
    } else {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        attention_qkv_pair_pack_f16_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(q_first->data),
            static_cast<const float*>(k_first->data),
            static_cast<const float*>(v_first->data),
            static_cast<const float*>(q_second->data),
            static_cast<const float*>(k_second->data),
            static_cast<const float*>(v_second->data),
            static_cast<__half*>(dst->data),
            d_head,
            first_seq,
            second_seq,
            n_head,
            elem_stride(q_first, 0),
            elem_stride(q_first, 1),
            elem_stride(k_first, 0),
            elem_stride(k_first, 1),
            elem_stride(v_first, 0),
            elem_stride(v_first, 1),
            elem_stride(q_second, 0),
            elem_stride(q_second, 1),
            elem_stride(k_second, 0),
            elem_stride(k_second, 1),
            elem_stride(v_second, 0),
            elem_stride(v_second, 1));
    }
    return true;
}
