#pragma once

#include "ggml.h"

using ed_cudnn_conv3d_stream_t = void *;

enum ed_cudnn_conv3d_result_t {
    ED_CUDNN_CONV3D_SUCCESS = 0,
    ED_CUDNN_CONV3D_UNSUPPORTED,
    ED_CUDNN_CONV3D_BUILD_FAILED,
    ED_CUDNN_CONV3D_EXECUTE_FAILED,
};

ed_cudnn_conv3d_result_t ed_cudnn_conv3d_compute(ggml_tensor * dst, ed_cudnn_conv3d_stream_t stream);

bool ed_cudnn_conv3d_supported(const ggml_tensor * dst, int device);

const char * ed_cudnn_conv3d_result_name(ed_cudnn_conv3d_result_t result);
