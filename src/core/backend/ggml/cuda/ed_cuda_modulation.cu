#include "ed_cuda_modulation.h"

#include "backend/ggml/ed_ggml_modulation_ext.hpp"

#include <cuda_runtime.h>
#include <cstring>

namespace {

static inline int64_t elem_stride(const ggml_tensor* t, int dim) {
    return t->nb[dim] / ggml_type_size(t->type);
}

static bool is_contiguous_f32_output(const ggml_tensor* t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(float) &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * sizeof(float) &&
           t->nb[3] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * static_cast<size_t>(t->ne[2]) * sizeof(float);
}

static bool is_aligned_to(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

static bool can_vec4_modulate(const ggml_tensor* x,
                              const ggml_tensor* shift,
                              const ggml_tensor* scale,
                              const ggml_tensor* dst) {
    return x != nullptr &&
           shift != nullptr &&
           scale != nullptr &&
           dst != nullptr &&
           (dst->ne[0] % 4) == 0 &&
           x->ne[0] == dst->ne[0] &&
           shift->ne[0] == dst->ne[0] &&
           scale->ne[0] == dst->ne[0] &&
           is_contiguous_f32_output(dst) &&
           elem_stride(x, 0) == 1 &&
           elem_stride(shift, 0) == 1 &&
           elem_stride(scale, 0) == 1 &&
           is_aligned_to(x->data, alignof(float4)) &&
           is_aligned_to(shift->data, alignof(float4)) &&
           is_aligned_to(scale->data, alignof(float4)) &&
           is_aligned_to(dst->data, alignof(float4));
}

static bool can_vec4_residual_gate(const ggml_tensor* residual,
                                   const ggml_tensor* x,
                                   const ggml_tensor* gate,
                                   const ggml_tensor* dst) {
    return residual != nullptr &&
           x != nullptr &&
           gate != nullptr &&
           dst != nullptr &&
           (dst->ne[0] % 4) == 0 &&
           residual->ne[0] == dst->ne[0] &&
           x->ne[0] == dst->ne[0] &&
           gate->ne[0] == dst->ne[0] &&
           is_contiguous_f32_output(dst) &&
           elem_stride(residual, 0) == 1 &&
           elem_stride(x, 0) == 1 &&
           elem_stride(gate, 0) == 1 &&
           is_aligned_to(residual->data, alignof(float4)) &&
           is_aligned_to(x->data, alignof(float4)) &&
           is_aligned_to(gate->data, alignof(float4)) &&
           is_aligned_to(dst->data, alignof(float4));
}

__global__ void fused_modulate_f32_kernel(const float* x,
                                          const float* shift,
                                          const float* scale,
                                          float* dst,
                                          int64_t ne0,
                                          int64_t ne1,
                                          int64_t ne2,
                                          int64_t ne3,
                                          int64_t x_s0,
                                          int64_t x_s1,
                                          int64_t x_s2,
                                          int64_t x_s3,
                                          int64_t shift_s0,
                                          int64_t shift_s1,
                                          int64_t shift_s2,
                                          int64_t shift_s3,
                                          int64_t scale_s0,
                                          int64_t scale_s1,
                                          int64_t scale_s2,
                                          int64_t scale_s3,
                                          int shift_b0,
                                          int shift_b1,
                                          int shift_b2,
                                          int shift_b3,
                                          int scale_b0,
                                          int scale_b1,
                                          int scale_b2,
                                          int scale_b3) {
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    int64_t rest = idx / ne0;
    const int64_t i1 = rest % ne1;
    rest /= ne1;
    const int64_t i2 = rest % ne2;
    const int64_t i3 = rest / ne2;

    const int64_t x_idx = i0 * x_s0 + i1 * x_s1 + i2 * x_s2 + i3 * x_s3;
    const int64_t shift_idx = (shift_b0 ? 0 : i0) * shift_s0 +
                              (shift_b1 ? 0 : i1) * shift_s1 +
                              (shift_b2 ? 0 : i2) * shift_s2 +
                              (shift_b3 ? 0 : i3) * shift_s3;
    const int64_t scale_idx = (scale_b0 ? 0 : i0) * scale_s0 +
                              (scale_b1 ? 0 : i1) * scale_s1 +
                              (scale_b2 ? 0 : i2) * scale_s2 +
                              (scale_b3 ? 0 : i3) * scale_s3;

    const float xv = x[x_idx];
    const float prod = __fmul_rn(xv, scale[scale_idx]);
    const float y = __fadd_rn(xv, prod);
    dst[idx] = __fadd_rn(y, shift[shift_idx]);
}

__global__ void fused_modulate_f32_vec4_kernel(const float* x,
                                               const float* shift,
                                               const float* scale,
                                               float* dst,
                                               int64_t ne0,
                                               int64_t ne1,
                                               int64_t ne2,
                                               int64_t ne3,
                                               int64_t x_s1,
                                               int64_t x_s2,
                                               int64_t x_s3,
                                               int64_t shift_s1,
                                               int64_t shift_s2,
                                               int64_t shift_s3,
                                               int64_t scale_s1,
                                               int64_t scale_s2,
                                               int64_t scale_s3,
                                               int shift_b1,
                                               int shift_b2,
                                               int shift_b3,
                                               int scale_b1,
                                               int scale_b2,
                                               int scale_b3) {
    const int64_t ne0_vec = ne0 / 4;
    const int64_t total = ne0_vec * ne1 * ne2 * ne3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int64_t i0_vec = idx % ne0_vec;
    int64_t rest = idx / ne0_vec;
    const int64_t i1 = rest % ne1;
    rest /= ne1;
    const int64_t i2 = rest % ne2;
    const int64_t i3 = rest / ne2;
    const int64_t i0 = i0_vec * 4;

    const int64_t x_idx = i0 + i1 * x_s1 + i2 * x_s2 + i3 * x_s3;
    const int64_t shift_idx = i0 +
                              (shift_b1 ? 0 : i1) * shift_s1 +
                              (shift_b2 ? 0 : i2) * shift_s2 +
                              (shift_b3 ? 0 : i3) * shift_s3;
    const int64_t scale_idx = i0 +
                              (scale_b1 ? 0 : i1) * scale_s1 +
                              (scale_b2 ? 0 : i2) * scale_s2 +
                              (scale_b3 ? 0 : i3) * scale_s3;

    const float4 xv = reinterpret_cast<const float4*>(x + x_idx)[0];
    const float4 sv = reinterpret_cast<const float4*>(scale + scale_idx)[0];
    const float4 bv = reinterpret_cast<const float4*>(shift + shift_idx)[0];

    float4 out;
    float prod = __fmul_rn(xv.x, sv.x);
    float y = __fadd_rn(xv.x, prod);
    out.x = __fadd_rn(y, bv.x);
    prod = __fmul_rn(xv.y, sv.y);
    y = __fadd_rn(xv.y, prod);
    out.y = __fadd_rn(y, bv.y);
    prod = __fmul_rn(xv.z, sv.z);
    y = __fadd_rn(xv.z, prod);
    out.z = __fadd_rn(y, bv.z);
    prod = __fmul_rn(xv.w, sv.w);
    y = __fadd_rn(xv.w, prod);
    out.w = __fadd_rn(y, bv.w);
    reinterpret_cast<float4*>(dst)[idx] = out;
}

__global__ void fused_residual_gate_f32_kernel(const float* residual,
                                               const float* x,
                                               const float* gate,
                                               float* dst,
                                               int64_t ne0,
                                               int64_t ne1,
                                               int64_t ne2,
                                               int64_t ne3,
                                               int64_t residual_s0,
                                               int64_t residual_s1,
                                               int64_t residual_s2,
                                               int64_t residual_s3,
                                               int64_t x_s0,
                                               int64_t x_s1,
                                               int64_t x_s2,
                                               int64_t x_s3,
                                               int64_t gate_s0,
                                               int64_t gate_s1,
                                               int64_t gate_s2,
                                               int64_t gate_s3,
                                               int gate_b0,
                                               int gate_b1,
                                               int gate_b2,
                                               int gate_b3) {
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    int64_t rest = idx / ne0;
    const int64_t i1 = rest % ne1;
    rest /= ne1;
    const int64_t i2 = rest % ne2;
    const int64_t i3 = rest / ne2;

    const int64_t residual_idx = i0 * residual_s0 + i1 * residual_s1 + i2 * residual_s2 + i3 * residual_s3;
    const int64_t x_idx = i0 * x_s0 + i1 * x_s1 + i2 * x_s2 + i3 * x_s3;
    const int64_t gate_idx = (gate_b0 ? 0 : i0) * gate_s0 +
                             (gate_b1 ? 0 : i1) * gate_s1 +
                             (gate_b2 ? 0 : i2) * gate_s2 +
                             (gate_b3 ? 0 : i3) * gate_s3;

    const float prod = __fmul_rn(x[x_idx], gate[gate_idx]);
    dst[idx] = __fadd_rn(residual[residual_idx], prod);
}

__global__ void fused_residual_gate_f32_vec4_kernel(const float* residual,
                                                    const float* x,
                                                    const float* gate,
                                                    float* dst,
                                                    int64_t ne0,
                                                    int64_t ne1,
                                                    int64_t ne2,
                                                    int64_t ne3,
                                                    int64_t residual_s1,
                                                    int64_t residual_s2,
                                                    int64_t residual_s3,
                                                    int64_t x_s1,
                                                    int64_t x_s2,
                                                    int64_t x_s3,
                                                    int64_t gate_s1,
                                                    int64_t gate_s2,
                                                    int64_t gate_s3,
                                                    int gate_b1,
                                                    int gate_b2,
                                                    int gate_b3) {
    const int64_t ne0_vec = ne0 / 4;
    const int64_t total = ne0_vec * ne1 * ne2 * ne3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int64_t i0_vec = idx % ne0_vec;
    int64_t rest = idx / ne0_vec;
    const int64_t i1 = rest % ne1;
    rest /= ne1;
    const int64_t i2 = rest % ne2;
    const int64_t i3 = rest / ne2;
    const int64_t i0 = i0_vec * 4;

    const int64_t residual_idx = i0 + i1 * residual_s1 + i2 * residual_s2 + i3 * residual_s3;
    const int64_t x_idx = i0 + i1 * x_s1 + i2 * x_s2 + i3 * x_s3;
    const int64_t gate_idx = i0 +
                             (gate_b1 ? 0 : i1) * gate_s1 +
                             (gate_b2 ? 0 : i2) * gate_s2 +
                             (gate_b3 ? 0 : i3) * gate_s3;

    const float4 rv = reinterpret_cast<const float4*>(residual + residual_idx)[0];
    const float4 xv = reinterpret_cast<const float4*>(x + x_idx)[0];
    const float4 gv = reinterpret_cast<const float4*>(gate + gate_idx)[0];

    float4 out;
    out.x = __fadd_rn(rv.x, __fmul_rn(xv.x, gv.x));
    out.y = __fadd_rn(rv.y, __fmul_rn(xv.y, gv.y));
    out.z = __fadd_rn(rv.z, __fmul_rn(xv.z, gv.z));
    out.w = __fadd_rn(rv.w, __fmul_rn(xv.w, gv.w));
    reinterpret_cast<float4*>(dst)[idx] = out;
}

} // namespace

static bool ed_cuda_fused_modulate_op_supported(const ggml_tensor * dst) {
    if (dst == nullptr ||
        dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr ||
        dst->src[1] == nullptr ||
        dst->src[2] == nullptr) {
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

    const auto params = edgedit::ggml_ext::fused_modulate_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::fused_modulate_params_valid(params)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* shift = dst->src[1];
    const ggml_tensor* scale = dst->src[2];
    if (!edgedit::ggml_ext::fused_modulate_shape_supported(x, shift, scale)) {
        return false;
    }
    if (!is_contiguous_f32_output(dst)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] != x->ne[i]) {
            return false;
        }
    }
    return true;
}

static bool ed_cuda_fused_residual_gate_op_supported(const ggml_tensor * dst) {
    if (dst == nullptr ||
        dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr ||
        dst->src[1] == nullptr ||
        dst->src[2] == nullptr) {
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

    const auto params = edgedit::ggml_ext::fused_residual_gate_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::fused_residual_gate_params_valid(params)) {
        return false;
    }

    const ggml_tensor* residual = dst->src[0];
    const ggml_tensor* x = dst->src[1];
    const ggml_tensor* gate = dst->src[2];
    if (!edgedit::ggml_ext::fused_residual_gate_shape_supported(residual, x, gate)) {
        return false;
    }
    if (!is_contiguous_f32_output(dst)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] != x->ne[i]) {
            return false;
        }
    }
    return true;
}

bool ed_cuda_fused_modulate_custom_supported(const ggml_tensor * dst) {
    return ed_cuda_fused_modulate_op_supported(dst) ||
           ed_cuda_fused_residual_gate_op_supported(dst);
}

static bool ed_cuda_fused_modulate_op_compute(ggml_tensor * dst, ed_cuda_modulation_stream_t stream) {
    if (!ed_cuda_fused_modulate_op_supported(dst)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* shift = dst->src[1];
    const ggml_tensor* scale = dst->src[2];

    const int64_t total = ggml_nelements(dst);
    constexpr int threads = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (can_vec4_modulate(x, shift, scale, dst)) {
        const int64_t total_vec = total / 4;
        const int blocks = static_cast<int>((total_vec + threads - 1) / threads);
        fused_modulate_f32_vec4_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(x->data),
            static_cast<const float*>(shift->data),
            static_cast<const float*>(scale->data),
            static_cast<float*>(dst->data),
            dst->ne[0],
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            elem_stride(x, 1),
            elem_stride(x, 2),
            elem_stride(x, 3),
            elem_stride(shift, 1),
            elem_stride(shift, 2),
            elem_stride(shift, 3),
            elem_stride(scale, 1),
            elem_stride(scale, 2),
            elem_stride(scale, 3),
            shift->ne[1] == 1 ? 1 : 0,
            shift->ne[2] == 1 ? 1 : 0,
            shift->ne[3] == 1 ? 1 : 0,
            scale->ne[1] == 1 ? 1 : 0,
            scale->ne[2] == 1 ? 1 : 0,
            scale->ne[3] == 1 ? 1 : 0);
    } else {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        fused_modulate_f32_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(x->data),
            static_cast<const float*>(shift->data),
            static_cast<const float*>(scale->data),
            static_cast<float*>(dst->data),
            dst->ne[0],
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            elem_stride(x, 0),
            elem_stride(x, 1),
            elem_stride(x, 2),
            elem_stride(x, 3),
            elem_stride(shift, 0),
            elem_stride(shift, 1),
            elem_stride(shift, 2),
            elem_stride(shift, 3),
            elem_stride(scale, 0),
            elem_stride(scale, 1),
            elem_stride(scale, 2),
            elem_stride(scale, 3),
            shift->ne[0] == 1 ? 1 : 0,
            shift->ne[1] == 1 ? 1 : 0,
            shift->ne[2] == 1 ? 1 : 0,
            shift->ne[3] == 1 ? 1 : 0,
            scale->ne[0] == 1 ? 1 : 0,
            scale->ne[1] == 1 ? 1 : 0,
            scale->ne[2] == 1 ? 1 : 0,
            scale->ne[3] == 1 ? 1 : 0);
    }
    return true;
}

static bool ed_cuda_fused_residual_gate_op_compute(ggml_tensor * dst, ed_cuda_modulation_stream_t stream) {
    if (!ed_cuda_fused_residual_gate_op_supported(dst)) {
        return false;
    }

    const ggml_tensor* residual = dst->src[0];
    const ggml_tensor* x = dst->src[1];
    const ggml_tensor* gate = dst->src[2];

    const int64_t total = ggml_nelements(dst);
    constexpr int threads = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (can_vec4_residual_gate(residual, x, gate, dst)) {
        const int64_t total_vec = total / 4;
        const int blocks = static_cast<int>((total_vec + threads - 1) / threads);
        fused_residual_gate_f32_vec4_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(residual->data),
            static_cast<const float*>(x->data),
            static_cast<const float*>(gate->data),
            static_cast<float*>(dst->data),
            dst->ne[0],
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            elem_stride(residual, 1),
            elem_stride(residual, 2),
            elem_stride(residual, 3),
            elem_stride(x, 1),
            elem_stride(x, 2),
            elem_stride(x, 3),
            elem_stride(gate, 1),
            elem_stride(gate, 2),
            elem_stride(gate, 3),
            gate->ne[1] == 1 ? 1 : 0,
            gate->ne[2] == 1 ? 1 : 0,
            gate->ne[3] == 1 ? 1 : 0);
    } else {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        fused_residual_gate_f32_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const float*>(residual->data),
            static_cast<const float*>(x->data),
            static_cast<const float*>(gate->data),
            static_cast<float*>(dst->data),
            dst->ne[0],
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            elem_stride(residual, 0),
            elem_stride(residual, 1),
            elem_stride(residual, 2),
            elem_stride(residual, 3),
            elem_stride(x, 0),
            elem_stride(x, 1),
            elem_stride(x, 2),
            elem_stride(x, 3),
            elem_stride(gate, 0),
            elem_stride(gate, 1),
            elem_stride(gate, 2),
            elem_stride(gate, 3),
            gate->ne[0] == 1 ? 1 : 0,
            gate->ne[1] == 1 ? 1 : 0,
            gate->ne[2] == 1 ? 1 : 0,
            gate->ne[3] == 1 ? 1 : 0);
    }
    return true;
}

bool ed_cuda_fused_modulate_custom_compute(ggml_tensor * dst, ed_cuda_modulation_stream_t stream) {
    return ed_cuda_fused_modulate_op_compute(dst, stream) ||
           ed_cuda_fused_residual_gate_op_compute(dst, stream);
}
