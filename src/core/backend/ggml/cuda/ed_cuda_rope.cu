#include "ed_cuda_rope.h"

#include "backend/ggml/ed_ggml_rope_ext.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

using edgedit::ggml_ext::RopeCustomParams;
using edgedit::ggml_ext::RopeInputLayout;

static inline int64_t elem_stride(const ggml_tensor* t, int dim) {
    return t->nb[dim] / ggml_type_size(t->type);
}

static bool is_contiguous_3d_output(const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return false;
    }
    const size_t ts = ggml_type_size(t->type) / ggml_blck_size(t->type);
    return t->nb[0] == ts &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * ts &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * ts;
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

template <typename dst_t>
__device__ __forceinline__ dst_t rope_cast(float v) {
    return static_cast<dst_t>(v);
}

template <>
__device__ __forceinline__ __half rope_cast<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 rope_cast<nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <typename src_t>
__device__ __forceinline__ float rope_load_x(const src_t* x, int64_t offset) {
    return static_cast<float>(x[offset]);
}

template <>
__device__ __forceinline__ float rope_load_x<nv_bfloat16>(const nv_bfloat16* x, int64_t offset) {
    return __bfloat162float(x[offset]);
}

template <typename src_t, typename dst_t>
__global__ void rope_kernel(const src_t* x,
                            const float* pe,
                            dst_t* dst,
                            int layout,
                            int interleaved,
                            int d_head,
                            int seq,
                            int n_head,
                            int batch,
                            bool pe_prepared,
                            int64_t x_s0,
                            int64_t x_s1,
                            int64_t x_s2,
                            int64_t x_s3,
                            int64_t pe_s0,
                            int64_t pe_s1,
                            int64_t pe_s2,
                            int64_t pe_s3) {
    const int half = d_head / 2;
    const int64_t heads_total = static_cast<int64_t>(n_head) * batch;
    const int64_t total = static_cast<int64_t>(half) * seq * heads_total;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int p = idx % half;
    idx /= half;
    const int s = idx % seq;
    const int h_total = idx / seq;

    float x0 = 0.0f;
    float x1 = 0.0f;
    if (layout == static_cast<int>(RopeInputLayout::NSeqHeadDim)) {
        const int h = h_total % n_head;
        const int b = h_total / n_head;
        if (interleaved) {
            x0 = rope_load_x(x, (2 * p + 0) * x_s0 + h * x_s1 + s * x_s2 + b * x_s3);
            x1 = rope_load_x(x, (2 * p + 1) * x_s0 + h * x_s1 + s * x_s2 + b * x_s3);
        } else {
            x0 = rope_load_x(x, p * x_s0 + h * x_s1 + s * x_s2 + b * x_s3);
            x1 = rope_load_x(x, (p + half) * x_s0 + h * x_s1 + s * x_s2 + b * x_s3);
        }
    } else if (layout == static_cast<int>(RopeInputLayout::SeqHeadDim)) {
        const int h = h_total % n_head;
        const int b = h_total / n_head;
        if (interleaved) {
            x0 = rope_load_x(x, (2 * p + 0) * x_s0 + s * x_s1 + h * x_s2 + b * x_s3);
            x1 = rope_load_x(x, (2 * p + 1) * x_s0 + s * x_s1 + h * x_s2 + b * x_s3);
        } else {
            x0 = rope_load_x(x, p * x_s0 + s * x_s1 + h * x_s2 + b * x_s3);
            x1 = rope_load_x(x, (p + half) * x_s0 + s * x_s1 + h * x_s2 + b * x_s3);
        }
    } else {
        x0 = rope_load_x(x, p * x_s0 + s * x_s1 + h_total * x_s2 + 0 * x_s3);
        x1 = rope_load_x(x, p * x_s0 + s * x_s1 + h_total * x_s2 + 1 * x_s3);
    }

    const float y0 = __fadd_rn(
        __fmul_rn(x0, load_pe(pe, pe_prepared, s, p, 0, 0, pe_s0, pe_s1, pe_s2, pe_s3)),
        __fmul_rn(x1, load_pe(pe, pe_prepared, s, p, 1, 0, pe_s0, pe_s1, pe_s2, pe_s3)));
    const float y1 = __fadd_rn(
        __fmul_rn(x0, load_pe(pe, pe_prepared, s, p, 0, 1, pe_s0, pe_s1, pe_s2, pe_s3)),
        __fmul_rn(x1, load_pe(pe, pe_prepared, s, p, 1, 1, pe_s0, pe_s1, pe_s2, pe_s3)));

    const int64_t dst_base = static_cast<int64_t>(h_total) * d_head * seq + static_cast<int64_t>(s) * d_head;
    if (interleaved || layout == static_cast<int>(RopeInputLayout::Work)) {
        dst[dst_base + 2 * p + 0] = rope_cast<dst_t>(y0);
        dst[dst_base + 2 * p + 1] = rope_cast<dst_t>(y1);
    } else {
        dst[dst_base + p] = rope_cast<dst_t>(y0);
        dst[dst_base + p + half] = rope_cast<dst_t>(y1);
    }
}

} // namespace

bool ed_cuda_rope_custom_supported(const ggml_tensor * dst) {
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

    const RopeCustomParams params = edgedit::ggml_ext::rope_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::rope_params_valid(params)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto layout = static_cast<RopeInputLayout>(params.input_layout);
    if (!edgedit::ggml_ext::rope_custom_shape_supported(x, pe, layout, params.interleaved != 0, params.d_head)) {
        return false;
    }
    if (!is_contiguous_3d_output(dst)) {
        return false;
    }

    const int d_head = params.d_head;
    const int half = d_head / 2;
    int seq = 0;
    if (layout == RopeInputLayout::NSeqHeadDim) {
        seq = static_cast<int>(x->ne[2]);
    } else if (layout == RopeInputLayout::SeqHeadDim) {
        seq = static_cast<int>(x->ne[1]);
    } else {
        seq = static_cast<int>(x->ne[1]);
    }

    const bool pe_prepared = pe_is_prepared_layout(pe, seq, half);
    return pe_prepared || pe_is_matrix_layout(pe, seq, half);
}

bool ed_cuda_rope_custom_compute(ggml_tensor * dst, ed_cuda_rope_stream_t stream) {
    if (!ed_cuda_rope_custom_supported(dst)) {
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

    const RopeCustomParams params = edgedit::ggml_ext::rope_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::rope_params_valid(params)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto layout = static_cast<RopeInputLayout>(params.input_layout);
    const int d_head = params.d_head;
    const int half = d_head / 2;
    int seq = 0;
    int n_head = 0;
    int batch = 1;
    if (layout == RopeInputLayout::NSeqHeadDim) {
        seq = static_cast<int>(x->ne[2]);
        n_head = static_cast<int>(x->ne[1]);
        batch = static_cast<int>(x->ne[3]);
    } else if (layout == RopeInputLayout::SeqHeadDim) {
        seq = static_cast<int>(x->ne[1]);
        n_head = static_cast<int>(x->ne[2]);
        batch = static_cast<int>(x->ne[3]);
    } else {
        seq = static_cast<int>(x->ne[1]);
        n_head = static_cast<int>(x->ne[2]);
        batch = 1;
    }

    const bool pe_prepared = pe_is_prepared_layout(pe, seq, half);
    if (!pe_prepared && !pe_is_matrix_layout(pe, seq, half)) {
        return false;
    }

    const int64_t total = static_cast<int64_t>(half) * seq * n_head * batch;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto launch = [&](auto* x_ptr, auto* dst_ptr) {
        rope_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            x_ptr,
            static_cast<const float*>(pe->data),
            dst_ptr,
            params.input_layout,
            params.interleaved,
            d_head,
            seq,
            n_head,
            batch,
            pe_prepared,
            elem_stride(x, 0),
            elem_stride(x, 1),
            elem_stride(x, 2),
            elem_stride(x, 3),
            elem_stride(pe, 0),
            elem_stride(pe, 1),
            elem_stride(pe, 2),
            elem_stride(pe, 3));
    };

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch(static_cast<const float*>(x->data), static_cast<float*>(dst->data));
    } else if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        launch(static_cast<const float*>(x->data), static_cast<__half*>(dst->data));
    } else if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        launch(static_cast<const float*>(x->data), static_cast<nv_bfloat16*>(dst->data));
    } else if (x->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        launch(static_cast<const nv_bfloat16*>(x->data), static_cast<nv_bfloat16*>(dst->data));
    } else if (x->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        launch(static_cast<const nv_bfloat16*>(x->data), static_cast<float*>(dst->data));
    } else if (x->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F16) {
        // bf16 input, f16 output: arises under bf16 flux with offload (the DiT computes in
        // bf16 but RoPE emits an f16 tensor). Was missing -> GGML_ABORT("unsupported CUDA
        // custom op"). The kernel is templated on both types, so this instantiates directly.
        launch(static_cast<const nv_bfloat16*>(x->data), static_cast<__half*>(dst->data));
    } else {
        return false;
    }
    return true;
}
