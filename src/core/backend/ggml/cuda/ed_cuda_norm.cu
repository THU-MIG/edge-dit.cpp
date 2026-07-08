#include "ed_cuda_norm.h"

#include "backend/ggml/ed_ggml_norm_ext.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace {

constexpr int WARP_SIZE_ED = 32;
constexpr int MAX_SMALL_D = 256;
constexpr int MIN_TOTAL_ROWS = 1024;

static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE_ED / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset, WARP_SIZE_ED);
    }
    return v;
}

static __global__ void rms_norm_mul_small_d_warp_f32(const float * x,
                                                     const float * mul,
                                                     float *       dst,
                                                     const int     ncols,
                                                     const int     nrows,
                                                     const int     nchannels,
                                                     const int     nsamples,
                                                     const int64_t stride_row,
                                                     const int64_t stride_channel,
                                                     const int64_t stride_sample,
                                                     const float   eps) {
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE_ED;
    const int total_rows  = nrows * nchannels * nsamples;
    if (global_warp >= total_rows) {
        return;
    }

    const int lane    = threadIdx.x % WARP_SIZE_ED;
    const int row     = global_warp % nrows;
    const int channel = (global_warp / nrows) % nchannels;
    const int sample  = global_warp / (nrows * nchannels);

    const float * x_row = x + sample * stride_sample + channel * stride_channel + row * stride_row;
    float *       y_row = dst + static_cast<int64_t>(global_warp) * ncols;

    float sum = 0.0f;
    for (int col = lane; col < ncols; col += WARP_SIZE_ED) {
        const float v = x_row[col];
        sum += v * v;
    }

    sum = warp_reduce_sum(sum);
    const float scale = rsqrtf(sum / static_cast<float>(ncols) + eps);

    for (int col = lane; col < ncols; col += WARP_SIZE_ED) {
        y_row[col] = x_row[col] * scale * mul[col];
    }
}

static __global__ void rms_norm_mul_small_d_warp_f16_out(const float * __restrict__ x,
                                                         const float * __restrict__ weight,
                                                         half * __restrict__ dst,
                                                         int ncols,
                                                         int nrows,
                                                         int nchannels,
                                                         int nsamples,
                                                         int64_t stride_row,
                                                         int64_t stride_channel,
                                                         int64_t stride_sample,
                                                         float eps) {
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE_ED;
    const int total_rows  = nrows * nchannels * nsamples;
    if (global_warp >= total_rows) {
        return;
    }

    const int lane    = threadIdx.x % WARP_SIZE_ED;
    const int row     = global_warp % nrows;
    const int channel = (global_warp / nrows) % nchannels;
    const int sample  = global_warp / (nrows * nchannels);

    const float * x_row = x + sample * stride_sample + channel * stride_channel + row * stride_row;
    half * y_row = dst + static_cast<int64_t>(global_warp) * ncols;

    float sum = 0.0f;
    for (int col = lane; col < ncols; col += WARP_SIZE_ED) {
        const float v = x_row[col];
        sum += v * v;
    }

    sum = warp_reduce_sum(sum);
    const float scale = rsqrtf(sum / static_cast<float>(ncols) + eps);

    for (int col = lane; col < ncols; col += WARP_SIZE_ED) {
        y_row[col] = __float2half_rn(x_row[col] * scale * weight[col]);
    }
}

static bool can_use_small_d_rms_norm_mul(int      ncols,
                                         int      nrows,
                                         int      nchannels,
                                         int      nsamples,
                                         int64_t  stride_row,
                                         int64_t  mul_stride_row,
                                         int64_t  mul_stride_channel,
                                         int64_t  mul_stride_sample,
                                         uint32_t mul_ncols,
                                         uint32_t mul_nrows,
                                         uint32_t mul_nchannels,
                                         uint32_t mul_nsamples) {
    const int64_t total_rows = static_cast<int64_t>(nrows) * nchannels * nsamples;
    return ncols > 0 &&
           ncols <= MAX_SMALL_D &&
           total_rows >= MIN_TOTAL_ROWS &&
           stride_row == ncols &&
           mul_ncols == static_cast<uint32_t>(ncols) &&
           mul_nrows == 1 &&
           mul_nchannels == 1 &&
           mul_nsamples == 1 &&
           mul_stride_row == static_cast<int64_t>(ncols) &&
           mul_stride_channel == static_cast<int64_t>(ncols) &&
           mul_stride_sample == static_cast<int64_t>(ncols);
}

static bool is_contiguous_f32_output(const ggml_tensor* t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(float) &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * sizeof(float) &&
           t->nb[3] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * static_cast<size_t>(t->ne[2]) * sizeof(float);
}

static bool is_contiguous_f16_output(const ggml_tensor* t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F16 &&
           t->nb[0] == sizeof(half) &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(half) &&
           t->nb[2] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * sizeof(half) &&
           t->nb[3] == static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]) * static_cast<size_t>(t->ne[2]) * sizeof(half);
}

static bool is_contiguous_f32_1d(const ggml_tensor* t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float);
}

template <int ROWS, int CTHREADS>
static __global__ void channel_rms_norm_f32_tile_kernel(const float* __restrict__ x,
                                                        const float* __restrict__ weight,
                                                        float* __restrict__ dst,
                                                        int64_t outer,
                                                        int channels,
                                                        float eps) {
    const int row_lane = threadIdx.x;
    const int c_lane = threadIdx.y;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS + row_lane;

    float sum = 0.0f;
    if (row < outer) {
        for (int c = c_lane; c < channels; c += CTHREADS) {
            const float v = x[static_cast<int64_t>(c) * outer + row];
            sum += v * v;
        }
    }

    __shared__ float shared[CTHREADS][ROWS];
    shared[c_lane][row_lane] = sum;
    __syncthreads();

#pragma unroll
    for (int offset = CTHREADS / 2; offset > 0; offset >>= 1) {
        if (c_lane < offset) {
            shared[c_lane][row_lane] += shared[c_lane + offset][row_lane];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(shared[0][row_lane] / static_cast<float>(channels) + eps);
    if (row < outer) {
        for (int c = c_lane; c < channels; c += CTHREADS) {
            const int64_t idx = static_cast<int64_t>(c) * outer + row;
            dst[idx] = x[idx] * scale * weight[c];
        }
    }
}

} // namespace

bool ed_cuda_rms_norm_mul_f32(const float *     x,
                              const float *     mul,
                              float *           dst,
                              int               ncols,
                              int               nrows,
                              int               nchannels,
                              int               nsamples,
                              int64_t           stride_row,
                              int64_t           stride_channel,
                              int64_t           stride_sample,
                              int64_t           mul_stride_row,
                              int64_t           mul_stride_channel,
                              int64_t           mul_stride_sample,
                              uint32_t          mul_ncols,
                              uint32_t          mul_nrows,
                              uint32_t          mul_nchannels,
                              uint32_t          mul_nsamples,
                              float             eps,
                              ed_cuda_norm_stream_t stream) {
    if (x == nullptr || mul == nullptr || dst == nullptr) {
        return false;
    }
    if (!can_use_small_d_rms_norm_mul(ncols,
                                      nrows,
                                      nchannels,
                                      nsamples,
                                      stride_row,
                                      mul_stride_row,
                                      mul_stride_channel,
                                      mul_stride_sample,
                                      mul_ncols,
                                      mul_nrows,
                                      mul_nchannels,
                                      mul_nsamples)) {
        return false;
    }

    constexpr int threads = 256;
    constexpr int warps_per_block = threads / WARP_SIZE_ED;
    const int total_rows = nrows * nchannels * nsamples;
    const int blocks = (total_rows + warps_per_block - 1) / warps_per_block;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    rms_norm_mul_small_d_warp_f32<<<blocks, threads, 0, cuda_stream>>>(
        x, mul, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    return true;
}

bool ed_cuda_channel_rms_norm_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr ||
        dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr ||
        dst->src[1] == nullptr) {
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

    const auto params = edgedit::ggml_ext::channel_rms_norm_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::channel_rms_norm_params_valid(params)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    if (!edgedit::ggml_ext::channel_rms_norm_shape_supported(x, weight)) {
        return false;
    }
    if (!is_contiguous_f32_output(dst) || !is_contiguous_f32_1d(weight)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] != x->ne[i]) {
            return false;
        }
    }
    return true;
}

bool ed_cuda_channel_rms_norm_custom_compute(ggml_tensor * dst, ed_cuda_norm_stream_t stream) {
    if (!ed_cuda_channel_rms_norm_custom_supported(dst)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    const int channels = static_cast<int>(x->ne[3]);
    if (channels <= 0 || channels > 1024) {
        return false;
    }

    const int64_t outer = x->ne[0] * x->ne[1] * x->ne[2];
    if (outer <= 0) {
        return false;
    }

    constexpr int rows = 128;
    constexpr int channel_threads = 4;
    const dim3 threads(rows, channel_threads, 1);
    const int blocks = static_cast<int>((outer + rows - 1) / rows);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    channel_rms_norm_f32_tile_kernel<rows, channel_threads><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const float*>(x->data),
        static_cast<const float*>(weight->data),
        static_cast<float*>(dst->data),
        outer,
        channels,
        1e-12f);
    return true;
}

bool ed_cuda_rms_norm_mul_f16_custom_supported(const ggml_tensor * dst) {
    if (dst == nullptr ||
        dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr ||
        dst->src[1] == nullptr) {
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

    const auto params = edgedit::ggml_ext::rms_norm_mul_f16_params_from_userdata(op_params.userdata);
    if (!edgedit::ggml_ext::rms_norm_mul_f16_params_valid(params)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    if (!edgedit::ggml_ext::rms_norm_mul_f16_shape_supported(x, weight)) {
        return false;
    }
    if (!is_contiguous_f16_output(dst) || !is_contiguous_f32_1d(weight)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] != x->ne[i]) {
            return false;
        }
    }
    return true;
}

bool ed_cuda_rms_norm_mul_f16_custom_compute(ggml_tensor * dst, ed_cuda_norm_stream_t stream) {
    if (!ed_cuda_rms_norm_mul_f16_custom_supported(dst)) {
        return false;
    }

    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    const int ncols = static_cast<int>(x->ne[0]);
    const int nrows = static_cast<int>(x->ne[1]);
    const int nchannels = static_cast<int>(x->ne[2]);
    const int nsamples = static_cast<int>(x->ne[3]);
    if (ncols <= 0 || ncols > MAX_SMALL_D ||
        nrows <= 0 || nchannels <= 0 || nsamples <= 0) {
        return false;
    }

    const size_t ts0 = ggml_type_size(x->type);
    if (x->nb[0] != ts0) {
        return false;
    }

    const auto params = edgedit::ggml_ext::rms_norm_mul_f16_params_from_userdata([&]() -> void* {
        struct ggml_custom_op_params {
            ggml_custom_op_t fun;
            int n_tasks;
            void* userdata;
        };
        ggml_custom_op_params op_params{};
        memcpy(&op_params, dst->op_params, sizeof(op_params));
        return op_params.userdata;
    }());
    const float eps = edgedit::ggml_ext::rms_norm_mul_f16_params_eps(params);
    if (eps < 0.0f) {
        return false;
    }

    constexpr int threads = 256;
    constexpr int warps_per_block = threads / WARP_SIZE_ED;
    const int total_rows = nrows * nchannels * nsamples;
    const int blocks = (total_rows + warps_per_block - 1) / warps_per_block;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    rms_norm_mul_small_d_warp_f16_out<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const float*>(x->data),
        static_cast<const float*>(weight->data),
        static_cast<half*>(dst->data),
        ncols,
        nrows,
        nchannels,
        nsamples,
        x->nb[1] / ts0,
        x->nb[2] / ts0,
        x->nb[3] / ts0,
        eps);
    return true;
}
