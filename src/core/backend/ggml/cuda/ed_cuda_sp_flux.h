#pragma once

#include "ggml.h"

using ed_cuda_sp_flux_stream_t = void *;

bool ed_cuda_flux_sp_qkv_recv_prep_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_qkv_recv_prep_bundle_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_recv_prep_bundle_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_qkv_pair_recv_prep_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_pair_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_bundle_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_combined_pair_recv_prep_bundle_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_qkv_mixed_recv_prep_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_mixed_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_qkv_pair_mixed_recv_prep_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_qkv_pair_mixed_recv_prep_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_all_to_all_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_all_to_all_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);

bool ed_cuda_flux_sp_all_gather_custom_supported(const ggml_tensor * dst);

bool ed_cuda_flux_sp_all_gather_custom_compute(ggml_tensor * dst, ed_cuda_sp_flux_stream_t stream);
