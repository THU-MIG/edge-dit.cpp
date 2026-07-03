#pragma once

#include "ggml.h"

using ed_cuda_attention_v_prep_stream_t = void *;

bool ed_cuda_attention_v_prep_custom_compute(ggml_tensor * dst, ed_cuda_attention_v_prep_stream_t stream);
