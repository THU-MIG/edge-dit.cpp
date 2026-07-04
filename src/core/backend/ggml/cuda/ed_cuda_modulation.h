#pragma once

using ed_cuda_modulation_stream_t = void *;

struct ggml_tensor;

bool ed_cuda_fused_modulate_custom_supported(const ggml_tensor * dst);

bool ed_cuda_fused_modulate_custom_compute(ggml_tensor * dst, ed_cuda_modulation_stream_t stream);
