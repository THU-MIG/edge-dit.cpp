#pragma once

#include "ggml.h"

#include <cstdint>

using ed_cudnn_sdpa_stream_t = void *;

enum ed_cudnn_sdpa_result_t {
    ED_CUDNN_SDPA_SUCCESS = 0,
    ED_CUDNN_SDPA_DISABLED,
    ED_CUDNN_SDPA_UNSUPPORTED,
    ED_CUDNN_SDPA_BUILD_PENDING,
    ED_CUDNN_SDPA_BUILD_FAILED,
    ED_CUDNN_SDPA_EXECUTE_FAILED,
};

ed_cudnn_sdpa_result_t ed_cudnn_sdpa_compute(ggml_tensor * dst, ed_cudnn_sdpa_stream_t stream);

bool ed_cudnn_sdpa_supported(const ggml_tensor * dst, int device);

bool ed_cudnn_sdpa_allows_fallback();

const char * ed_cudnn_sdpa_result_name(ed_cudnn_sdpa_result_t result);
