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

ed_cudnn_sdpa_result_t ed_cudnn_sdpa_prewarm_self_attn(int device,
                                                       ggml_type dst_type,
                                                       int64_t d,
                                                       int64_t h,
                                                       int64_t seq,
                                                       float attn_scale,
                                                       bool sync_build = false);

ed_cudnn_sdpa_result_t ed_cudnn_sdpa_prewarm_cross_attn(int device,
                                                        ggml_type dst_type,
                                                        int64_t d,
                                                        int64_t h,
                                                        int64_t seq_q,
                                                        int64_t seq_kv,
                                                        float attn_scale,
                                                        bool sync_build = false);

bool ed_cudnn_sdpa_supported(const ggml_tensor * dst, int device);

bool ed_cudnn_sdpa_allows_fallback();

const char * ed_cudnn_sdpa_result_name(ed_cudnn_sdpa_result_t result);
