#include "ed_cuda_attention_v_prep.h"

#include "backend/ggml/ed_ggml_attention_ext.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace {

using edgedit::ggml_ext::AttentionVPrepCustomParams;

static inline int64_t elem_stride(const ggml_tensor* t, int dim) {
    return t->nb[dim] / ggml_type_size(t->type);
}

static bool is_contiguous_3d_f16_output(const ggml_tensor* t) {
    return t->type == GGML_TYPE_F16 &&
           t->nb[0] == sizeof(__half) &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(__half) &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * sizeof(__half);
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

} // namespace

bool ed_cuda_attention_v_prep_custom_compute(ggml_tensor * dst, ed_cuda_attention_v_prep_stream_t stream) {
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
    if (dst->ne[0] != d_head || dst->ne[1] != seq || dst->ne[2] != static_cast<int64_t>(n_head) * batch) {
        return false;
    }

    const int64_t total = static_cast<int64_t>(d_head) * seq * n_head * batch;
    constexpr int threads = 256;
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
    return true;
}
