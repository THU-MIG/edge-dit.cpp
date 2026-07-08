#pragma once

#include <cstdint>

#include "ggml.h"

using ed_cuda_norm_stream_t = void *;

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
                              ed_cuda_norm_stream_t stream);

bool ed_cuda_channel_rms_norm_custom_supported(const ggml_tensor * dst);

bool ed_cuda_channel_rms_norm_custom_compute(ggml_tensor * dst, ed_cuda_norm_stream_t stream);

bool ed_cuda_rms_norm_mul_f16_custom_supported(const ggml_tensor * dst);

bool ed_cuda_rms_norm_mul_f16_custom_compute(ggml_tensor * dst, ed_cuda_norm_stream_t stream);
