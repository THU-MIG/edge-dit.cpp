#pragma once

#include "ggml.h"

using ed_cuda_rope_stream_t = void *;

bool ed_cuda_rope_custom_compute(ggml_tensor * dst, ed_cuda_rope_stream_t stream);
