#include "ed_cuda_norm.h"

#include <cuda_runtime.h>

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
