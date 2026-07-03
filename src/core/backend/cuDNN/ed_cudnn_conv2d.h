#pragma once

#include "ggml.h"

using ed_cudnn_conv2d_stream_t = void *;

enum ed_cudnn_conv2d_result_t {
    ED_CUDNN_CONV2D_SUCCESS = 0,
    ED_CUDNN_CONV2D_UNSUPPORTED,
    ED_CUDNN_CONV2D_BUILD_FAILED,
    ED_CUDNN_CONV2D_EXECUTE_FAILED,
};

ed_cudnn_conv2d_result_t ed_cudnn_conv2d_compute(ggml_tensor * dst, ed_cudnn_conv2d_stream_t stream);

const char * ed_cudnn_conv2d_result_name(ed_cudnn_conv2d_result_t result);
