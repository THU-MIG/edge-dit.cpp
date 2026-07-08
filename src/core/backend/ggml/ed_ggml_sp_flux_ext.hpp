#pragma once

#include "ggml.h"
#include "parallel/process_group.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace edgedit::ggml_ext {

constexpr uint32_t kFluxSPQKVRecvPrepCustomMagic = 0x45465152u; // "EFQR"
constexpr uint32_t kFluxSPQKVPairRecvPrepCustomMagic = 0x45465150u; // "EFQP"
constexpr uint32_t kFluxSPQKVMixedRecvPrepCustomMagic = 0x45464d52u; // "EFMR"
constexpr uint32_t kFluxSPQKVPairMixedRecvPrepCustomMagic = 0x45464d50u; // "EFMP"
constexpr uint32_t kFluxSPQKVRecvPrepBundleCustomMagic = 0x45465142u; // "EFQB"
constexpr uint32_t kFluxSPQKVCombinedPairRecvPrepCustomMagic = 0x45465143u; // "EFQC"
constexpr uint32_t kFluxSPQKVCombinedPairRecvPrepBundleCustomMagic = 0x45464342u; // "EFCB"
constexpr uint32_t kFluxSPConcatLinearCustomMagic = 0x45464c32u; // "EFL2"
constexpr uint32_t kFluxSPConcatLinearResidualGateCustomMagic = 0x45464c47u; // "EFLG"
constexpr uint32_t kFluxSPGeluBF16CustomMagic = 0x45464742u; // "EFGB"
constexpr uint32_t kFluxSPAllToAllCustomMagic = 0x45464132u; // "EFA2"
constexpr uint32_t kFluxSPAllGatherCustomMagic = 0x45464147u; // "EFAG"

enum class FluxSPQKVRecvPrepPlane : int32_t {
    Q = 0,
    K = 1,
    V = 2,
};

struct FluxSPQKVRecvPrepCustomParams {
    uint32_t magic = kFluxSPQKVRecvPrepCustomMagic;
    int32_t plane = 0;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
};

struct FluxSPQKVPairRecvPrepCustomParams {
    uint32_t magic = kFluxSPQKVPairRecvPrepCustomMagic;
    int32_t plane = 0;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
};

struct FluxSPQKVMixedRecvPrepCustomParams {
    uint32_t magic = kFluxSPQKVMixedRecvPrepCustomMagic;
    int32_t plane = 0;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
};

struct FluxSPQKVPairMixedRecvPrepCustomParams {
    uint32_t magic = kFluxSPQKVPairMixedRecvPrepCustomMagic;
    int32_t plane = 0;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
};

struct FluxSPQKVRecvPrepBundleCustomParams {
    uint32_t magic = kFluxSPQKVRecvPrepBundleCustomMagic;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
};

struct FluxSPQKVCombinedPairRecvPrepCustomParams {
    uint32_t magic = kFluxSPQKVCombinedPairRecvPrepCustomMagic;
    int32_t plane = 0;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
    int32_t first_shard_sequence = 0;
};

struct FluxSPQKVCombinedPairRecvPrepBundleCustomParams {
    uint32_t magic = kFluxSPQKVCombinedPairRecvPrepBundleCustomMagic;
    int32_t world_size = 0;
    int32_t heads = 0;
    int32_t head_dim = 0;
    int32_t first_shard_sequence = 0;
};

struct FluxSPConcatLinearCustomParams {
    uint32_t magic = kFluxSPConcatLinearCustomMagic;
};

struct FluxSPConcatLinearResidualGateCustomParams {
    uint32_t magic = kFluxSPConcatLinearResidualGateCustomMagic;
};

struct FluxSPGeluBF16CustomParams {
    uint32_t magic = kFluxSPGeluBF16CustomMagic;
};

struct FluxSPAllToAllCustomParams {
    uint32_t magic = kFluxSPAllToAllCustomMagic;
    int64_t count_per_peer = 0;
    int64_t world_size = 0;
    edgedit::parallel::ProcessGroup* process_group = nullptr;
};

struct FluxSPAllGatherCustomParams {
    uint32_t magic = kFluxSPAllGatherCustomMagic;
    int64_t count_per_rank = 0;
    int64_t world_size = 0;
    edgedit::parallel::ProcessGroup* process_group = nullptr;
};

inline std::mutex& flux_sp_all_to_all_params_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::vector<std::unique_ptr<FluxSPAllToAllCustomParams>>& flux_sp_all_to_all_params_store() {
    static std::vector<std::unique_ptr<FluxSPAllToAllCustomParams>> store;
    return store;
}

inline std::vector<std::unique_ptr<FluxSPAllGatherCustomParams>>& flux_sp_all_gather_params_store() {
    static std::vector<std::unique_ptr<FluxSPAllGatherCustomParams>> store;
    return store;
}

inline FluxSPAllToAllCustomParams* flux_sp_all_to_all_make_params(int64_t count_per_peer,
                                                                  int64_t world_size,
                                                                  edgedit::parallel::ProcessGroup* process_group) {
    auto params = std::make_unique<FluxSPAllToAllCustomParams>();
    params->magic = kFluxSPAllToAllCustomMagic;
    params->count_per_peer = count_per_peer;
    params->world_size = world_size;
    params->process_group = process_group;
    FluxSPAllToAllCustomParams* raw = params.get();
    std::lock_guard<std::mutex> lock(flux_sp_all_to_all_params_mutex());
    flux_sp_all_to_all_params_store().push_back(std::move(params));
    return raw;
}

inline FluxSPAllGatherCustomParams* flux_sp_all_gather_make_params(int64_t count_per_rank,
                                                                   int64_t world_size,
                                                                   edgedit::parallel::ProcessGroup* process_group) {
    auto params = std::make_unique<FluxSPAllGatherCustomParams>();
    params->magic = kFluxSPAllGatherCustomMagic;
    params->count_per_rank = count_per_rank;
    params->world_size = world_size;
    params->process_group = process_group;
    FluxSPAllGatherCustomParams* raw = params.get();
    std::lock_guard<std::mutex> lock(flux_sp_all_to_all_params_mutex());
    flux_sp_all_gather_params_store().push_back(std::move(params));
    return raw;
}

inline bool flux_sp_qkv_recv_prep_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_FLUX_SP_QKV_RECV_PREP");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline bool flux_sp_env_flag_enabled_or_default(const char* name, bool default_enabled) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_enabled;
    }
    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0 &&
           std::strcmp(env, "off") != 0 &&
           std::strcmp(env, "OFF") != 0;
}

inline bool flux_sp_all_to_all_custom_enabled() {
    return flux_sp_env_flag_enabled_or_default("ED_FLUX_SP_COMM_CUSTOM_OP", true);
}

inline bool flux_sp_all_to_all_dtype_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

inline edgedit::parallel::DataType flux_sp_all_to_all_data_type(ggml_type type) {
    GGML_ASSERT(flux_sp_all_to_all_dtype_supported(type));
    return type == GGML_TYPE_F16 ? edgedit::parallel::DataType::kFloat16 :
                                   edgedit::parallel::DataType::kFloat32;
}

inline bool flux_sp_concat_linear_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_FLUX_SP_CONCAT_LINEAR");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline bool flux_sp_gelu_bf16_enabled() {
    const char* env = std::getenv("ED_FLUX_SP_MLP_GELU_BF16");
    return env != nullptr && std::atoi(env) != 0;
}

inline FluxSPQKVRecvPrepCustomParams flux_sp_qkv_recv_prep_params_from_userdata(void* userdata) {
    FluxSPQKVRecvPrepCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.plane = static_cast<int32_t>((packed >> 32) & 0xffu);
    params.world_size = static_cast<int32_t>((packed >> 40) & 0xffu);
    params.heads = static_cast<int32_t>((packed >> 48) & 0xffu);
    params.head_dim = static_cast<int32_t>((packed >> 56) & 0xffu);
    return params;
}

inline void* flux_sp_qkv_recv_prep_params_to_userdata(FluxSPQKVRecvPrepPlane plane,
                                                       int64_t world_size,
                                                       int64_t heads,
                                                       int64_t head_dim) {
    uintptr_t packed = static_cast<uintptr_t>(kFluxSPQKVRecvPrepCustomMagic);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(plane)) << 32);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, world_size))) << 40);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, heads))) << 48);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, head_dim))) << 56);
    return reinterpret_cast<void*>(packed);
}

inline FluxSPQKVPairRecvPrepCustomParams flux_sp_qkv_pair_recv_prep_params_from_userdata(void* userdata) {
    FluxSPQKVPairRecvPrepCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.plane = static_cast<int32_t>((packed >> 32) & 0xffu);
    params.world_size = static_cast<int32_t>((packed >> 40) & 0xffu);
    params.heads = static_cast<int32_t>((packed >> 48) & 0xffu);
    params.head_dim = static_cast<int32_t>((packed >> 56) & 0xffu);
    return params;
}

inline FluxSPQKVMixedRecvPrepCustomParams flux_sp_qkv_mixed_recv_prep_params_from_userdata(void* userdata) {
    FluxSPQKVMixedRecvPrepCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.plane = static_cast<int32_t>((packed >> 32) & 0xffu);
    params.world_size = static_cast<int32_t>((packed >> 40) & 0xffu);
    params.heads = static_cast<int32_t>((packed >> 48) & 0xffu);
    params.head_dim = static_cast<int32_t>((packed >> 56) & 0xffu);
    return params;
}

inline FluxSPQKVPairMixedRecvPrepCustomParams flux_sp_qkv_pair_mixed_recv_prep_params_from_userdata(void* userdata) {
    FluxSPQKVPairMixedRecvPrepCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.plane = static_cast<int32_t>((packed >> 32) & 0xffu);
    params.world_size = static_cast<int32_t>((packed >> 40) & 0xffu);
    params.heads = static_cast<int32_t>((packed >> 48) & 0xffu);
    params.head_dim = static_cast<int32_t>((packed >> 56) & 0xffu);
    return params;
}

inline FluxSPQKVRecvPrepBundleCustomParams flux_sp_qkv_recv_prep_bundle_params_from_userdata(void* userdata) {
    FluxSPQKVRecvPrepBundleCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.world_size = static_cast<int32_t>((packed >> 32) & 0xffu);
    params.heads = static_cast<int32_t>((packed >> 40) & 0xffu);
    params.head_dim = static_cast<int32_t>((packed >> 48) & 0xffu);
    return params;
}

inline FluxSPQKVCombinedPairRecvPrepCustomParams flux_sp_qkv_combined_pair_recv_prep_params_from_userdata(void* userdata) {
    auto* ptr = static_cast<FluxSPQKVCombinedPairRecvPrepCustomParams*>(userdata);
    if (ptr != nullptr) {
        return *ptr;
    }
    return {};
}

inline FluxSPQKVCombinedPairRecvPrepBundleCustomParams flux_sp_qkv_combined_pair_recv_prep_bundle_params_from_userdata(void* userdata) {
    auto* ptr = static_cast<FluxSPQKVCombinedPairRecvPrepBundleCustomParams*>(userdata);
    if (ptr != nullptr) {
        return *ptr;
    }
    return {};
}

inline void* flux_sp_qkv_pair_recv_prep_params_to_userdata(FluxSPQKVRecvPrepPlane plane,
                                                            int64_t world_size,
                                                            int64_t heads,
                                                            int64_t head_dim) {
    uintptr_t packed = static_cast<uintptr_t>(kFluxSPQKVPairRecvPrepCustomMagic);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(plane)) << 32);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, world_size))) << 40);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, heads))) << 48);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, head_dim))) << 56);
    return reinterpret_cast<void*>(packed);
}

inline void* flux_sp_qkv_mixed_recv_prep_params_to_userdata(FluxSPQKVRecvPrepPlane plane,
                                                            int64_t world_size,
                                                            int64_t heads,
                                                            int64_t head_dim) {
    uintptr_t packed = static_cast<uintptr_t>(kFluxSPQKVMixedRecvPrepCustomMagic);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(plane)) << 32);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, world_size))) << 40);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, heads))) << 48);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, head_dim))) << 56);
    return reinterpret_cast<void*>(packed);
}

inline void* flux_sp_qkv_pair_mixed_recv_prep_params_to_userdata(FluxSPQKVRecvPrepPlane plane,
                                                                 int64_t world_size,
                                                                 int64_t heads,
                                                                 int64_t head_dim) {
    uintptr_t packed = static_cast<uintptr_t>(kFluxSPQKVPairMixedRecvPrepCustomMagic);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(plane)) << 32);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, world_size))) << 40);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, heads))) << 48);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, head_dim))) << 56);
    return reinterpret_cast<void*>(packed);
}

inline void* flux_sp_qkv_recv_prep_bundle_params_to_userdata(int64_t world_size,
                                                             int64_t heads,
                                                             int64_t head_dim) {
    uintptr_t packed = static_cast<uintptr_t>(kFluxSPQKVRecvPrepBundleCustomMagic);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, world_size))) << 32);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, heads))) << 40);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(std::max<int64_t>(0, head_dim))) << 48);
    return reinterpret_cast<void*>(packed);
}

inline std::mutex& flux_sp_qkv_combined_pair_recv_prep_params_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::vector<std::unique_ptr<FluxSPQKVCombinedPairRecvPrepCustomParams>>& flux_sp_qkv_combined_pair_recv_prep_params_store() {
    static std::vector<std::unique_ptr<FluxSPQKVCombinedPairRecvPrepCustomParams>> store;
    return store;
}

inline std::vector<std::unique_ptr<FluxSPQKVCombinedPairRecvPrepBundleCustomParams>>& flux_sp_qkv_combined_pair_recv_prep_bundle_params_store() {
    static std::vector<std::unique_ptr<FluxSPQKVCombinedPairRecvPrepBundleCustomParams>> store;
    return store;
}

inline FluxSPQKVCombinedPairRecvPrepCustomParams* flux_sp_qkv_combined_pair_recv_prep_params_to_userdata(
    FluxSPQKVRecvPrepPlane plane,
    int64_t world_size,
    int64_t heads,
    int64_t head_dim,
    int64_t first_shard_sequence) {
    auto params = std::make_unique<FluxSPQKVCombinedPairRecvPrepCustomParams>();
    params->magic = kFluxSPQKVCombinedPairRecvPrepCustomMagic;
    params->plane = static_cast<int32_t>(plane);
    params->world_size = static_cast<int32_t>(world_size);
    params->heads = static_cast<int32_t>(heads);
    params->head_dim = static_cast<int32_t>(head_dim);
    params->first_shard_sequence = static_cast<int32_t>(first_shard_sequence);
    FluxSPQKVCombinedPairRecvPrepCustomParams* raw = params.get();
    std::lock_guard<std::mutex> lock(flux_sp_qkv_combined_pair_recv_prep_params_mutex());
    flux_sp_qkv_combined_pair_recv_prep_params_store().push_back(std::move(params));
    return raw;
}

inline FluxSPQKVCombinedPairRecvPrepBundleCustomParams* flux_sp_qkv_combined_pair_recv_prep_bundle_params_to_userdata(
    int64_t world_size,
    int64_t heads,
    int64_t head_dim,
    int64_t first_shard_sequence) {
    auto params = std::make_unique<FluxSPQKVCombinedPairRecvPrepBundleCustomParams>();
    params->magic = kFluxSPQKVCombinedPairRecvPrepBundleCustomMagic;
    params->world_size = static_cast<int32_t>(world_size);
    params->heads = static_cast<int32_t>(heads);
    params->head_dim = static_cast<int32_t>(head_dim);
    params->first_shard_sequence = static_cast<int32_t>(first_shard_sequence);
    FluxSPQKVCombinedPairRecvPrepBundleCustomParams* raw = params.get();
    std::lock_guard<std::mutex> lock(flux_sp_qkv_combined_pair_recv_prep_params_mutex());
    flux_sp_qkv_combined_pair_recv_prep_bundle_params_store().push_back(std::move(params));
    return raw;
}

inline FluxSPConcatLinearCustomParams flux_sp_concat_linear_params_from_userdata(void* userdata) {
    FluxSPConcatLinearCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* flux_sp_concat_linear_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kFluxSPConcatLinearCustomMagic));
}

inline FluxSPConcatLinearResidualGateCustomParams flux_sp_concat_linear_residual_gate_params_from_userdata(void* userdata) {
    FluxSPConcatLinearResidualGateCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* flux_sp_concat_linear_residual_gate_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kFluxSPConcatLinearResidualGateCustomMagic));
}

inline FluxSPGeluBF16CustomParams flux_sp_gelu_bf16_params_from_userdata(void* userdata) {
    FluxSPGeluBF16CustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* flux_sp_gelu_bf16_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kFluxSPGeluBF16CustomMagic));
}

inline bool flux_sp_qkv_recv_prep_params_valid(const FluxSPQKVRecvPrepCustomParams& params) {
    return params.magic == kFluxSPQKVRecvPrepCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           (params.head_dim % 2) == 0 &&
           (params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::Q) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::K) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::V));
}

inline bool flux_sp_qkv_pair_recv_prep_params_valid(const FluxSPQKVPairRecvPrepCustomParams& params) {
    return params.magic == kFluxSPQKVPairRecvPrepCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           (params.head_dim % 2) == 0 &&
           (params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::Q) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::K) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::V));
}

inline bool flux_sp_qkv_mixed_recv_prep_params_valid(const FluxSPQKVMixedRecvPrepCustomParams& params) {
    return params.magic == kFluxSPQKVMixedRecvPrepCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           (params.head_dim % 2) == 0 &&
           (params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::Q) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::K) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::V));
}

inline bool flux_sp_qkv_pair_mixed_recv_prep_params_valid(const FluxSPQKVPairMixedRecvPrepCustomParams& params) {
    return params.magic == kFluxSPQKVPairMixedRecvPrepCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           (params.head_dim % 2) == 0 &&
           (params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::Q) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::K) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::V));
}

inline bool flux_sp_qkv_recv_prep_bundle_params_valid(const FluxSPQKVRecvPrepBundleCustomParams& params) {
    return params.magic == kFluxSPQKVRecvPrepBundleCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           (params.head_dim % 2) == 0;
}

inline bool flux_sp_qkv_combined_pair_recv_prep_params_valid(const FluxSPQKVCombinedPairRecvPrepCustomParams& params) {
    return params.magic == kFluxSPQKVCombinedPairRecvPrepCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           params.first_shard_sequence > 0 &&
           (params.head_dim % 2) == 0 &&
           (params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::Q) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::K) ||
            params.plane == static_cast<int32_t>(FluxSPQKVRecvPrepPlane::V));
}

inline bool flux_sp_qkv_combined_pair_recv_prep_bundle_params_valid(const FluxSPQKVCombinedPairRecvPrepBundleCustomParams& params) {
    return params.magic == kFluxSPQKVCombinedPairRecvPrepBundleCustomMagic &&
           params.world_size > 0 &&
           params.heads > 0 &&
           params.head_dim > 0 &&
           params.first_shard_sequence > 0 &&
           (params.head_dim % 2) == 0;
}

inline bool flux_sp_concat_linear_params_valid(const FluxSPConcatLinearCustomParams& params) {
    return params.magic == kFluxSPConcatLinearCustomMagic;
}

inline bool flux_sp_concat_linear_residual_gate_params_valid(const FluxSPConcatLinearResidualGateCustomParams& params) {
    return params.magic == kFluxSPConcatLinearResidualGateCustomMagic;
}

inline bool flux_sp_gelu_bf16_params_valid(const FluxSPGeluBF16CustomParams& params) {
    return params.magic == kFluxSPGeluBF16CustomMagic;
}

inline bool flux_sp_all_to_all_params_valid(const FluxSPAllToAllCustomParams* params) {
    return params != nullptr &&
           params->magic == kFluxSPAllToAllCustomMagic &&
           params->count_per_peer > 0 &&
           params->world_size > 0 &&
           params->process_group != nullptr &&
           params->process_group->size() == params->world_size;
}

inline bool flux_sp_all_gather_params_valid(const FluxSPAllGatherCustomParams* params) {
    return params != nullptr &&
           params->magic == kFluxSPAllGatherCustomMagic &&
           params->count_per_rank > 0 &&
           params->world_size > 0 &&
           params->process_group != nullptr &&
           params->process_group->size() == params->world_size;
}

inline bool flux_sp_all_to_all_shape_supported(const ggml_tensor* send_flat,
                                               const ggml_tensor* recv_flat,
                                               const FluxSPAllToAllCustomParams* params) {
    if (!flux_sp_all_to_all_custom_enabled() ||
        send_flat == nullptr ||
        recv_flat == nullptr ||
        !flux_sp_all_to_all_params_valid(params) ||
        send_flat->type != recv_flat->type ||
        !flux_sp_all_to_all_dtype_supported(send_flat->type) ||
        send_flat->ne[1] != 1 ||
        send_flat->ne[2] != 1 ||
        send_flat->ne[3] != 1 ||
        recv_flat->ne[1] != 1 ||
        recv_flat->ne[2] != 1 ||
        recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t expected = params->count_per_peer * params->world_size;
    return send_flat->ne[0] == expected && recv_flat->ne[0] == expected;
}

inline bool flux_sp_all_gather_shape_supported(const ggml_tensor* send_flat,
                                               const ggml_tensor* recv_flat,
                                               const FluxSPAllGatherCustomParams* params) {
    if (!flux_sp_all_to_all_custom_enabled() ||
        send_flat == nullptr ||
        recv_flat == nullptr ||
        !flux_sp_all_gather_params_valid(params) ||
        send_flat->type != recv_flat->type ||
        !flux_sp_all_to_all_dtype_supported(send_flat->type) ||
        send_flat->ne[1] != 1 ||
        send_flat->ne[2] != 1 ||
        send_flat->ne[3] != 1 ||
        recv_flat->ne[1] != 1 ||
        recv_flat->ne[2] != 1 ||
        recv_flat->ne[3] != 1) {
        return false;
    }
    return send_flat->ne[0] == params->count_per_rank &&
           recv_flat->ne[0] == params->count_per_rank * params->world_size;
}

inline bool flux_sp_concat_linear_shape_supported(const ggml_tensor* x0,
                                                  const ggml_tensor* x1,
                                                  const ggml_tensor* weight,
                                                  const ggml_tensor* bias) {
    if (!flux_sp_concat_linear_enabled() ||
        x0 == nullptr ||
        x1 == nullptr ||
        weight == nullptr ||
        (x0->type != GGML_TYPE_F32 && x0->type != GGML_TYPE_F16 && x0->type != GGML_TYPE_BF16) ||
        (x1->type != GGML_TYPE_F32 && x1->type != GGML_TYPE_F16 && x1->type != GGML_TYPE_BF16) ||
        (weight->type != GGML_TYPE_BF16 && weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_F32) ||
        (bias != nullptr && bias->type != GGML_TYPE_F32)) {
        return false;
    }
    if (x0->ne[0] <= 0 || x1->ne[0] <= 0 || weight->ne[0] != x0->ne[0] + x1->ne[0] || weight->ne[1] <= 0) {
        return false;
    }
    if (x0->ne[1] != x1->ne[1] || x0->ne[2] != x1->ne[2] || x0->ne[3] != x1->ne[3]) {
        return false;
    }
    if (bias != nullptr && !(bias->ne[0] == weight->ne[1] && bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1)) {
        return false;
    }
    return true;
}

inline bool flux_sp_concat_linear_residual_gate_shape_supported(const ggml_tensor* residual,
                                                                const ggml_tensor* x0,
                                                                const ggml_tensor* x1,
                                                                const ggml_tensor* weight,
                                                                const ggml_tensor* bias,
                                                                const ggml_tensor* gate) {
    if (!flux_sp_concat_linear_shape_supported(x0, x1, weight, bias) ||
        residual == nullptr ||
        gate == nullptr ||
        residual->type != GGML_TYPE_F32 ||
        gate->type != GGML_TYPE_F32) {
        return false;
    }
    if (residual->ne[0] != weight->ne[1] ||
        residual->ne[1] != x0->ne[1] ||
        residual->ne[2] != x0->ne[2] ||
        residual->ne[3] != x0->ne[3]) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (residual->ne[i] <= 0 || gate->ne[i] <= 0) {
            return false;
        }
        if (!(gate->ne[i] == 1 || gate->ne[i] == residual->ne[i])) {
            return false;
        }
    }
    return true;
}

inline bool flux_sp_gelu_bf16_shape_supported(const ggml_tensor* x) {
    return flux_sp_gelu_bf16_enabled() &&
           x != nullptr &&
           (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(x);
}

inline float flux_sp_concat_linear_tensor_f32_at(const ggml_tensor* t,
                                                 int64_t i0,
                                                 int64_t i1,
                                                 int64_t i2,
                                                 int64_t i3) {
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    if (t->type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(ptr);
    }
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
    }
    GGML_ASSERT(t->type == GGML_TYPE_BF16);
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(ptr));
}

inline void flux_sp_concat_linear_tensor_set(ggml_tensor* t,
                                             int64_t i0,
                                             int64_t i1,
                                             int64_t i2,
                                             int64_t i3,
                                             float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    *reinterpret_cast<float*>(ptr) = v;
}

inline void flux_sp_tensor_set_bf16(ggml_tensor* t,
                                    int64_t i0,
                                    int64_t i1,
                                    int64_t i2,
                                    int64_t i3,
                                    float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_BF16);
    *reinterpret_cast<ggml_bf16_t*>(ptr) = ggml_fp32_to_bf16(v);
}

inline float flux_sp_tensor_f32_at_broadcast(const ggml_tensor* t,
                                             int64_t i0,
                                             int64_t i1,
                                             int64_t i2,
                                             int64_t i3) {
    const int64_t j0 = t->ne[0] == 1 ? 0 : i0;
    const int64_t j1 = t->ne[1] == 1 ? 0 : i1;
    const int64_t j2 = t->ne[2] == 1 ? 0 : i2;
    const int64_t j3 = t->ne[3] == 1 ? 0 : i3;
    return flux_sp_concat_linear_tensor_f32_at(t, j0, j1, j2, j3);
}

inline float flux_sp_gelu_tanh_f32(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kGeluCoeff = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * x * (1.0f + kGeluCoeff * x * x)));
}

inline void flux_sp_gelu_bf16_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPGeluBF16CustomParams params = flux_sp_gelu_bf16_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_gelu_bf16_params_valid(params));
    GGML_ASSERT(dst != nullptr && dst->src[0] != nullptr);

    const ggml_tensor* x = dst->src[0];
    GGML_ASSERT(flux_sp_gelu_bf16_shape_supported(x));
    GGML_ASSERT(dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_are_same_shape(dst, x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t total = ggml_nelements(dst);
    const int64_t chunk = (total + nth - 1) / nth;
    const int64_t begin = std::min<int64_t>(total, ith * chunk);
    const int64_t end = std::min<int64_t>(total, begin + chunk);
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];

    for (int64_t linear = begin; linear < end; ++linear) {
        const int64_t i0 = linear % ne0;
        int64_t rem = linear / ne0;
        const int64_t i1 = rem % ne1;
        rem /= ne1;
        const int64_t i2 = rem % ne2;
        const int64_t i3 = rem / ne2;
        const float v = flux_sp_concat_linear_tensor_f32_at(x, i0, i1, i2, i3);
        flux_sp_tensor_set_bf16(dst, i0, i1, i2, i3, flux_sp_gelu_tanh_f32(v));
    }
}

inline void flux_sp_concat_linear_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPConcatLinearCustomParams params = flux_sp_concat_linear_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_concat_linear_params_valid(params));
    GGML_ASSERT(dst != nullptr && dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);

    const ggml_tensor* x0 = dst->src[0];
    const ggml_tensor* x1 = dst->src[1];
    const ggml_tensor* weight = dst->src[2];
    const ggml_tensor* bias = dst->src[3];
    GGML_ASSERT(flux_sp_concat_linear_shape_supported(x0, x1, weight, bias));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t out_features = weight->ne[1];
    const int64_t x0_features = x0->ne[0];
    const int64_t x1_features = x1->ne[0];
    const int64_t cols = x0->ne[1] * x0->ne[2] * x0->ne[3];
    const int64_t total = out_features * cols;
    const int64_t chunk = (total + nth - 1) / nth;
    const int64_t begin = std::min<int64_t>(total, ith * chunk);
    const int64_t end = std::min<int64_t>(total, begin + chunk);

    for (int64_t linear = begin; linear < end; ++linear) {
        const int64_t o = linear % out_features;
        int64_t rem = linear / out_features;
        const int64_t c1 = rem % x0->ne[1];
        rem /= x0->ne[1];
        const int64_t c2 = rem % x0->ne[2];
        const int64_t c3 = rem / x0->ne[2];

        float acc = bias != nullptr ? flux_sp_concat_linear_tensor_f32_at(bias, o, 0, 0, 0) : 0.0f;
        for (int64_t k = 0; k < x0_features; ++k) {
            acc += flux_sp_concat_linear_tensor_f32_at(weight, k, o, 0, 0) *
                   flux_sp_concat_linear_tensor_f32_at(x0, k, c1, c2, c3);
        }
        for (int64_t k = 0; k < x1_features; ++k) {
            acc += flux_sp_concat_linear_tensor_f32_at(weight, x0_features + k, o, 0, 0) *
                   flux_sp_concat_linear_tensor_f32_at(x1, k, c1, c2, c3);
        }
        flux_sp_concat_linear_tensor_set(dst, o, c1, c2, c3, acc);
    }
}

inline void flux_sp_concat_linear_residual_gate_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPConcatLinearResidualGateCustomParams params =
        flux_sp_concat_linear_residual_gate_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_concat_linear_residual_gate_params_valid(params));
    GGML_ASSERT(dst != nullptr &&
                dst->src[0] != nullptr &&
                dst->src[1] != nullptr &&
                dst->src[2] != nullptr &&
                dst->src[3] != nullptr &&
                dst->src[5] != nullptr);

    const ggml_tensor* residual = dst->src[0];
    const ggml_tensor* x0 = dst->src[1];
    const ggml_tensor* x1 = dst->src[2];
    const ggml_tensor* weight = dst->src[3];
    const ggml_tensor* bias = dst->src[4];
    const ggml_tensor* gate = dst->src[5];
    GGML_ASSERT(flux_sp_concat_linear_residual_gate_shape_supported(residual, x0, x1, weight, bias, gate));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t out_features = weight->ne[1];
    const int64_t x0_features = x0->ne[0];
    const int64_t x1_features = x1->ne[0];
    const int64_t cols = x0->ne[1] * x0->ne[2] * x0->ne[3];
    const int64_t total = out_features * cols;
    const int64_t chunk = (total + nth - 1) / nth;
    const int64_t begin = std::min<int64_t>(total, ith * chunk);
    const int64_t end = std::min<int64_t>(total, begin + chunk);

    for (int64_t linear = begin; linear < end; ++linear) {
        const int64_t o = linear % out_features;
        int64_t rem = linear / out_features;
        const int64_t c1 = rem % x0->ne[1];
        rem /= x0->ne[1];
        const int64_t c2 = rem % x0->ne[2];
        const int64_t c3 = rem / x0->ne[2];

        float acc = bias != nullptr ? flux_sp_concat_linear_tensor_f32_at(bias, o, 0, 0, 0) : 0.0f;
        for (int64_t k = 0; k < x0_features; ++k) {
            acc += flux_sp_concat_linear_tensor_f32_at(weight, k, o, 0, 0) *
                   flux_sp_concat_linear_tensor_f32_at(x0, k, c1, c2, c3);
        }
        for (int64_t k = 0; k < x1_features; ++k) {
            acc += flux_sp_concat_linear_tensor_f32_at(weight, x0_features + k, o, 0, 0) *
                   flux_sp_concat_linear_tensor_f32_at(x1, k, c1, c2, c3);
        }
        const float rv = flux_sp_tensor_f32_at_broadcast(residual, o, c1, c2, c3);
        const float gv = flux_sp_tensor_f32_at_broadcast(gate, o, c1, c2, c3);
        flux_sp_concat_linear_tensor_set(dst, o, c1, c2, c3, rv + acc * gv);
    }
}

inline ggml_tensor* flux_sp_concat_linear_custom(ggml_context* ctx,
                                                 ggml_tensor* x0,
                                                 ggml_tensor* x1,
                                                 ggml_tensor* weight,
                                                 ggml_tensor* bias) {
    if (!flux_sp_concat_linear_shape_supported(x0, x1, weight, bias)) {
        return nullptr;
    }
    ggml_tensor* args[] = { x0, x1, weight, bias };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      weight->ne[1],
                                      x0->ne[1],
                                      x0->ne[2],
                                      x0->ne[3],
                                      args,
                                      bias != nullptr ? 4 : 3,
                                      flux_sp_concat_linear_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_concat_linear_params_to_userdata());
    ggml_set_name(out, "ed_flux_sp_concat_linear");
    return out;
}

inline ggml_tensor* flux_sp_concat_linear_residual_gate_custom(ggml_context* ctx,
                                                               ggml_tensor* residual,
                                                               ggml_tensor* x0,
                                                               ggml_tensor* x1,
                                                               ggml_tensor* weight,
                                                               ggml_tensor* bias,
                                                               ggml_tensor* gate) {
    if (!flux_sp_concat_linear_residual_gate_shape_supported(residual, x0, x1, weight, bias, gate)) {
        return nullptr;
    }
    ggml_tensor* args[] = { residual, x0, x1, weight, bias, gate };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      weight->ne[1],
                                      x0->ne[1],
                                      x0->ne[2],
                                      x0->ne[3],
                                      args,
                                      6,
                                      flux_sp_concat_linear_residual_gate_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_concat_linear_residual_gate_params_to_userdata());
    ggml_set_name(out, "ed_flux_sp_concat_linear_residual_gate");
    return out;
}

inline ggml_tensor* flux_sp_gelu_bf16_custom(ggml_context* ctx,
                                             ggml_tensor* x) {
    if (!flux_sp_gelu_bf16_shape_supported(x)) {
        return nullptr;
    }
    ggml_tensor* args[] = { x };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_BF16,
                                      x->ne[0],
                                      x->ne[1],
                                      x->ne[2],
                                      x->ne[3],
                                      args,
                                      1,
                                      flux_sp_gelu_bf16_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_gelu_bf16_params_to_userdata());
    ggml_set_name(out, "ed_flux_sp_gelu_bf16");
    return out;
}

inline void flux_sp_all_to_all_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)ith;
    (void)nth;
    const auto* params = static_cast<const FluxSPAllToAllCustomParams*>(userdata);
    GGML_ASSERT(flux_sp_all_to_all_shape_supported(dst->src[0], dst, params));
    const edgedit::parallel::DataType data_type = flux_sp_all_to_all_data_type(dst->type);
    edgedit::parallel::Buffer input{
        dst->src[0]->data,
        static_cast<size_t>(dst->src[0]->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    edgedit::parallel::Buffer output{
        dst->data,
        static_cast<size_t>(dst->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    params->process_group->all_to_all(input,
                                      output,
                                      static_cast<size_t>(params->count_per_peer));
}

inline void flux_sp_all_gather_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)ith;
    (void)nth;
    const auto* params = static_cast<const FluxSPAllGatherCustomParams*>(userdata);
    GGML_ASSERT(flux_sp_all_gather_shape_supported(dst->src[0], dst, params));
    const edgedit::parallel::DataType data_type = flux_sp_all_to_all_data_type(dst->type);
    edgedit::parallel::Buffer input{
        dst->src[0]->data,
        static_cast<size_t>(dst->src[0]->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    edgedit::parallel::Buffer output{
        dst->data,
        static_cast<size_t>(dst->ne[0]),
        data_type,
        params->process_group->local_rank(),
    };
    params->process_group->all_gather(input, output);
}

inline ggml_tensor* flux_sp_all_to_all_custom(ggml_context* ctx,
                                              ggml_tensor* send_flat,
                                              int64_t recv_elems,
                                              int64_t count_per_peer,
                                              int64_t world_size,
                                              edgedit::parallel::ProcessGroup* process_group,
                                              const std::string& name) {
    if (!flux_sp_all_to_all_custom_enabled() ||
        ctx == nullptr ||
        send_flat == nullptr ||
        process_group == nullptr ||
        !flux_sp_all_to_all_dtype_supported(send_flat->type) ||
        recv_elems <= 0 ||
        count_per_peer <= 0 ||
        world_size <= 0 ||
        recv_elems != count_per_peer * world_size ||
        send_flat->ne[0] != recv_elems ||
        send_flat->ne[1] != 1 ||
        send_flat->ne[2] != 1 ||
        send_flat->ne[3] != 1 ||
        process_group->size() != world_size) {
        return nullptr;
    }
    auto* params = flux_sp_all_to_all_make_params(count_per_peer,
                                                  world_size,
                                                  process_group);
    ggml_tensor* args[] = { send_flat };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      send_flat->type,
                                      recv_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      flux_sp_all_to_all_cpu_custom_op,
                                      1,
                                      params);
    if (!name.empty()) {
        ggml_set_name(out, (name + "_all_to_all_recv").c_str());
    } else {
        ggml_set_name(out, "ed_flux_sp_all_to_all_recv");
    }
    return out;
}

inline ggml_tensor* flux_sp_all_gather_custom(ggml_context* ctx,
                                              ggml_tensor* send_flat,
                                              int64_t recv_elems,
                                              int64_t count_per_rank,
                                              int64_t world_size,
                                              edgedit::parallel::ProcessGroup* process_group,
                                              const std::string& name) {
    if (!flux_sp_all_to_all_custom_enabled() ||
        ctx == nullptr ||
        send_flat == nullptr ||
        process_group == nullptr ||
        !flux_sp_all_to_all_dtype_supported(send_flat->type) ||
        recv_elems <= 0 ||
        count_per_rank <= 0 ||
        world_size <= 0 ||
        recv_elems != count_per_rank * world_size ||
        send_flat->ne[0] != count_per_rank ||
        send_flat->ne[1] != 1 ||
        send_flat->ne[2] != 1 ||
        send_flat->ne[3] != 1 ||
        process_group->size() != world_size) {
        return nullptr;
    }
    auto* params = flux_sp_all_gather_make_params(count_per_rank,
                                                  world_size,
                                                  process_group);
    ggml_tensor* args[] = { send_flat };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      send_flat->type,
                                      recv_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      flux_sp_all_gather_cpu_custom_op,
                                      1,
                                      params);
    if (!name.empty()) {
        ggml_set_name(out, (name + "_all_gather_recv").c_str());
    } else {
        ggml_set_name(out, "ed_flux_sp_all_gather_recv");
    }
    return out;
}

inline bool flux_sp_qkv_recv_prep_shape_supported(const ggml_tensor* recv_flat,
                                                   const ggml_tensor* pe,
                                                   FluxSPQKVRecvPrepPlane plane,
                                                   int64_t world_size,
                                                   int64_t heads,
                                                   int64_t head_dim) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        recv_flat == nullptr ||
        pe == nullptr ||
        (recv_flat->type != GGML_TYPE_F32 && recv_flat->type != GGML_TYPE_F16) ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (recv_flat->ne[1] != 1 || recv_flat->ne[2] != 1 || recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t denom = total_head_dim * shard_heads * world_size;
    if (denom <= 0 || recv_flat->ne[0] % denom != 0) {
        return false;
    }
    const int64_t shard_sequence = recv_flat->ne[0] / denom;
    const int64_t sequence = shard_sequence * world_size;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    if (!pe_matrix && !pe_prepared) {
        return false;
    }
    return plane == FluxSPQKVRecvPrepPlane::Q ||
           plane == FluxSPQKVRecvPrepPlane::K ||
           plane == FluxSPQKVRecvPrepPlane::V;
}

inline bool flux_sp_qkv_pair_recv_prep_shape_supported(const ggml_tensor* first_recv_flat,
                                                        const ggml_tensor* second_recv_flat,
                                                        const ggml_tensor* pe,
                                                        FluxSPQKVRecvPrepPlane plane,
                                                        int64_t world_size,
                                                        int64_t heads,
                                                        int64_t head_dim) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        first_recv_flat == nullptr ||
        second_recv_flat == nullptr ||
        pe == nullptr ||
        (first_recv_flat->type != GGML_TYPE_F32 && first_recv_flat->type != GGML_TYPE_F16) ||
        (second_recv_flat->type != GGML_TYPE_F32 && second_recv_flat->type != GGML_TYPE_F16) ||
        first_recv_flat->type != second_recv_flat->type ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (first_recv_flat->ne[1] != 1 || first_recv_flat->ne[2] != 1 || first_recv_flat->ne[3] != 1 ||
        second_recv_flat->ne[1] != 1 || second_recv_flat->ne[2] != 1 || second_recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t denom = total_head_dim * shard_heads * world_size;
    if (denom <= 0 || first_recv_flat->ne[0] % denom != 0 || second_recv_flat->ne[0] % denom != 0) {
        return false;
    }
    const int64_t first_sequence = (first_recv_flat->ne[0] / denom) * world_size;
    const int64_t second_sequence = (second_recv_flat->ne[0] / denom) * world_size;
    const int64_t sequence = first_sequence + second_sequence;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    if (!pe_matrix && !pe_prepared) {
        return false;
    }
    return plane == FluxSPQKVRecvPrepPlane::Q ||
           plane == FluxSPQKVRecvPrepPlane::K ||
           plane == FluxSPQKVRecvPrepPlane::V;
}

inline bool flux_sp_qkv_mixed_recv_prep_shape_supported(const ggml_tensor* recv_flat,
                                                        const ggml_tensor* pe,
                                                        FluxSPQKVRecvPrepPlane plane,
                                                        int64_t world_size,
                                                        int64_t heads,
                                                        int64_t head_dim) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        recv_flat == nullptr ||
        pe == nullptr ||
        recv_flat->type != GGML_TYPE_F32 ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (recv_flat->ne[1] != 1 || recv_flat->ne[2] != 1 || recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t packed_dim = head_dim * 2;
    const int64_t denom = packed_dim * shard_heads * world_size;
    if (denom <= 0 || recv_flat->ne[0] % denom != 0) {
        return false;
    }
    const int64_t shard_sequence = recv_flat->ne[0] / denom;
    const int64_t sequence = shard_sequence * world_size;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    if (!pe_matrix && !pe_prepared) {
        return false;
    }
    return plane == FluxSPQKVRecvPrepPlane::Q ||
           plane == FluxSPQKVRecvPrepPlane::K ||
           plane == FluxSPQKVRecvPrepPlane::V;
}

inline bool flux_sp_qkv_pair_mixed_recv_prep_shape_supported(const ggml_tensor* first_recv_flat,
                                                             const ggml_tensor* second_recv_flat,
                                                             const ggml_tensor* pe,
                                                             FluxSPQKVRecvPrepPlane plane,
                                                             int64_t world_size,
                                                             int64_t heads,
                                                             int64_t head_dim) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        first_recv_flat == nullptr ||
        second_recv_flat == nullptr ||
        pe == nullptr ||
        first_recv_flat->type != GGML_TYPE_F32 ||
        second_recv_flat->type != GGML_TYPE_F32 ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (first_recv_flat->ne[1] != 1 || first_recv_flat->ne[2] != 1 || first_recv_flat->ne[3] != 1 ||
        second_recv_flat->ne[1] != 1 || second_recv_flat->ne[2] != 1 || second_recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t packed_dim = head_dim * 2;
    const int64_t denom = packed_dim * shard_heads * world_size;
    if (denom <= 0 || first_recv_flat->ne[0] % denom != 0 || second_recv_flat->ne[0] % denom != 0) {
        return false;
    }
    const int64_t first_sequence = (first_recv_flat->ne[0] / denom) * world_size;
    const int64_t second_sequence = (second_recv_flat->ne[0] / denom) * world_size;
    const int64_t sequence = first_sequence + second_sequence;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    if (!pe_matrix && !pe_prepared) {
        return false;
    }
    return plane == FluxSPQKVRecvPrepPlane::Q ||
           plane == FluxSPQKVRecvPrepPlane::K ||
           plane == FluxSPQKVRecvPrepPlane::V;
}

inline bool flux_sp_qkv_recv_prep_bundle_shape_supported(const ggml_tensor* recv_flat,
                                                         const ggml_tensor* pe,
                                                         int64_t world_size,
                                                         int64_t heads,
                                                         int64_t head_dim) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        recv_flat == nullptr ||
        pe == nullptr ||
        recv_flat->type != GGML_TYPE_F16 ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (recv_flat->ne[1] != 1 || recv_flat->ne[2] != 1 || recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t denom = total_head_dim * shard_heads * world_size;
    if (denom <= 0 || recv_flat->ne[0] % denom != 0) {
        return false;
    }
    const int64_t shard_sequence = recv_flat->ne[0] / denom;
    const int64_t sequence = shard_sequence * world_size;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    return pe_matrix || pe_prepared;
}

inline bool flux_sp_qkv_combined_pair_recv_prep_shape_supported(const ggml_tensor* recv_flat,
                                                                const ggml_tensor* pe,
                                                                FluxSPQKVRecvPrepPlane plane,
                                                                int64_t world_size,
                                                                int64_t heads,
                                                                int64_t head_dim,
                                                                int64_t first_shard_sequence) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        recv_flat == nullptr ||
        pe == nullptr ||
        (recv_flat->type != GGML_TYPE_F32 && recv_flat->type != GGML_TYPE_F16) ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        first_shard_sequence <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (recv_flat->ne[1] != 1 || recv_flat->ne[2] != 1 || recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t peer_denom = total_head_dim * shard_heads;
    if (peer_denom <= 0 || recv_flat->ne[0] % (peer_denom * world_size) != 0) {
        return false;
    }
    const int64_t combined_shard_sequence = recv_flat->ne[0] / (peer_denom * world_size);
    if (first_shard_sequence >= combined_shard_sequence) {
        return false;
    }
    const int64_t sequence = combined_shard_sequence * world_size;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    if (!pe_matrix && !pe_prepared) {
        return false;
    }
    return plane == FluxSPQKVRecvPrepPlane::Q ||
           plane == FluxSPQKVRecvPrepPlane::K ||
           plane == FluxSPQKVRecvPrepPlane::V;
}

inline bool flux_sp_qkv_combined_pair_recv_prep_bundle_shape_supported(const ggml_tensor* recv_flat,
                                                                       const ggml_tensor* pe,
                                                                       int64_t world_size,
                                                                       int64_t heads,
                                                                       int64_t head_dim,
                                                                       int64_t first_shard_sequence) {
    if (!flux_sp_qkv_recv_prep_enabled() ||
        recv_flat == nullptr ||
        pe == nullptr ||
        recv_flat->type != GGML_TYPE_F16 ||
        pe->type != GGML_TYPE_F32 ||
        world_size <= 0 ||
        heads <= 0 ||
        head_dim <= 0 ||
        first_shard_sequence <= 0 ||
        (head_dim % 2) != 0 ||
        heads % world_size != 0) {
        return false;
    }
    if (recv_flat->ne[1] != 1 || recv_flat->ne[2] != 1 || recv_flat->ne[3] != 1) {
        return false;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t peer_denom = total_head_dim * shard_heads;
    if (peer_denom <= 0 || recv_flat->ne[0] % (peer_denom * world_size) != 0) {
        return false;
    }
    const int64_t combined_shard_sequence = recv_flat->ne[0] / (peer_denom * world_size);
    if (first_shard_sequence >= combined_shard_sequence) {
        return false;
    }
    const int64_t sequence = combined_shard_sequence * world_size;
    const int64_t half = head_dim / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half && pe->ne[3] >= sequence;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[2] >= sequence && pe->ne[3] == 2;
    return pe_matrix || pe_prepared;
}

inline bool flux_sp_qkv_recv_prep_output_type_supported(FluxSPQKVRecvPrepPlane plane, ggml_type type) {
    return (plane == FluxSPQKVRecvPrepPlane::Q && (type == GGML_TYPE_F32 || type == GGML_TYPE_F16)) ||
           (plane != FluxSPQKVRecvPrepPlane::Q && type == GGML_TYPE_F16);
}

inline ggml_type flux_sp_qkv_recv_prep_output_type(FluxSPQKVRecvPrepPlane plane, ggml_type q_output_type) {
    if (plane != FluxSPQKVRecvPrepPlane::Q) {
        return GGML_TYPE_F16;
    }
    return q_output_type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

inline float flux_sp_qkv_recv_prep_tensor_f32_at(const ggml_tensor* t,
                                                  int64_t i0,
                                                  int64_t i1,
                                                  int64_t i2,
                                                  int64_t i3) {
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    return *reinterpret_cast<const float*>(ptr);
}

inline void flux_sp_qkv_recv_prep_tensor_set(ggml_tensor* t,
                                              int64_t i0,
                                              int64_t i1,
                                              int64_t i2,
                                              int64_t i3,
                                              float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    if (t->type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(ptr) = v;
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F16);
        *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(v);
    }
}

inline float flux_sp_qkv_recv_prep_pe_at(const ggml_tensor* pe,
                                          int64_t seq,
                                          int64_t pair,
                                          int64_t row,
                                          int64_t col) {
    if (pe->ne[0] == 2 && pe->ne[1] == 2) {
        return flux_sp_qkv_recv_prep_tensor_f32_at(pe, row, col, pair, seq);
    }
    return flux_sp_qkv_recv_prep_tensor_f32_at(pe, row, pair, seq, col);
}

inline float flux_sp_qkv_recv_prep_load_recv(const ggml_tensor* recv_flat,
                                              int64_t plane,
                                              int64_t pair,
                                              int64_t part,
                                              int64_t seq,
                                              int64_t head,
                                              int64_t head_dim,
                                              int64_t shard_heads,
                                              int64_t shard_sequence) {
    const int64_t src_peer = seq / shard_sequence;
    const int64_t local_seq = seq - src_peer * shard_sequence;
    const int64_t local_head = head;
    const int64_t src_idx =
        plane * head_dim +
        pair * 2 + part +
        local_head * head_dim * 3 +
        local_seq * head_dim * 3 * shard_heads +
        src_peer * head_dim * 3 * shard_heads * shard_sequence;
    if (recv_flat->type == GGML_TYPE_F16) {
        const char* data = static_cast<const char*>(recv_flat->data) + src_idx * recv_flat->nb[0];
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data));
    }
    return flux_sp_qkv_recv_prep_tensor_f32_at(recv_flat, src_idx, 0, 0, 0);
}

inline float flux_sp_qkv_mixed_recv_prep_load_recv(const ggml_tensor* recv_flat,
                                                    int64_t plane,
                                                    int64_t d,
                                                    int64_t seq,
                                                    int64_t head,
                                                    int64_t head_dim,
                                                    int64_t shard_heads,
                                                    int64_t shard_sequence) {
    const int64_t src_peer = seq / shard_sequence;
    const int64_t local_seq = seq - src_peer * shard_sequence;
    const int64_t packed_dim = head_dim * 2;
    const int64_t src_idx =
        (plane == static_cast<int64_t>(FluxSPQKVRecvPrepPlane::Q) ? d : head_dim + d) +
        head * packed_dim +
        local_seq * packed_dim * shard_heads +
        src_peer * packed_dim * shard_heads * shard_sequence;
    const char* data = static_cast<const char*>(recv_flat->data) + src_idx * recv_flat->nb[0];
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(data);
    if (plane == static_cast<int64_t>(FluxSPQKVRecvPrepPlane::Q)) {
        float value;
        std::memcpy(&value, &packed, sizeof(value));
        return value;
    }
    const uint16_t half_bits = static_cast<uint16_t>(
        plane == static_cast<int64_t>(FluxSPQKVRecvPrepPlane::V) ? (packed >> 16) : (packed & 0xffffu));
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(half_bits));
}

inline float flux_sp_qkv_combined_pair_recv_prep_load_recv(const ggml_tensor* recv_flat,
                                                            int64_t plane,
                                                            int64_t d,
                                                            int64_t seq,
                                                            int64_t head,
                                                            int64_t head_dim,
                                                            int64_t shard_heads,
                                                            int64_t world_size,
                                                            int64_t first_shard_sequence,
                                                            int64_t second_shard_sequence) {
    const int64_t first_sequence = first_shard_sequence * world_size;
    const bool use_first = seq < first_sequence;
    const int64_t stream_seq = use_first ? seq : seq - first_sequence;
    const int64_t shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;
    const int64_t src_peer = stream_seq / shard_sequence;
    const int64_t local_seq = stream_seq - src_peer * shard_sequence;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t first_chunk = total_head_dim * shard_heads * first_shard_sequence;
    const int64_t second_chunk = total_head_dim * shard_heads * second_shard_sequence;
    const int64_t stream_offset = use_first ? 0 : first_chunk;
    const int64_t src_idx =
        src_peer * (first_chunk + second_chunk) +
        stream_offset +
        plane * head_dim +
        d +
        head * total_head_dim +
        local_seq * total_head_dim * shard_heads;
    if (recv_flat->type == GGML_TYPE_F16) {
        const char* data = static_cast<const char*>(recv_flat->data) + src_idx * recv_flat->nb[0];
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data));
    }
    return flux_sp_qkv_recv_prep_tensor_f32_at(recv_flat, src_idx, 0, 0, 0);
}

inline void flux_sp_qkv_recv_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVRecvPrepCustomParams params = flux_sp_qkv_recv_prep_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_recv_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    GGML_ASSERT(flux_sp_qkv_recv_prep_shape_supported(recv_flat, pe, plane, params.world_size, params.heads, params.head_dim));
    GGML_ASSERT(flux_sp_qkv_recv_prep_output_type_supported(plane, dst->type));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (head_dim * 3 * shard_heads * world_size);
    const int64_t sequence = shard_sequence * world_size;
    const int64_t plane_idx = params.plane;

    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 1);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;
        const int64_t head = out_head;

        if (plane == FluxSPQKVRecvPrepPlane::V) {
            const int64_t src_peer = seq / shard_sequence;
            const int64_t local_seq = seq - src_peer * shard_sequence;
            const int64_t local_head = head;
            const int64_t src_idx =
                plane_idx * head_dim +
                d +
                local_head * head_dim * 3 +
                local_seq * head_dim * 3 * shard_heads +
                src_peer * head_dim * 3 * shard_heads * shard_sequence;
            const float value = flux_sp_qkv_recv_prep_tensor_f32_at(recv_flat, src_idx, 0, 0, 0);
            flux_sp_qkv_recv_prep_tensor_set(dst, d, seq, out_head, 0, value);
            continue;
        }

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        const float x0 = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                          plane_idx,
                                                          pair,
                                                          0,
                                                          seq,
                                                          head,
                                                          head_dim,
                                                          shard_heads,
                                                          shard_sequence);
        const float x1 = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                          plane_idx,
                                                          pair,
                                                          1,
                                                          seq,
                                                          head,
                                                          head_dim,
                                                          shard_heads,
                                                          shard_sequence);
        const float y0 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 0, 0) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 1, 0);
        const float y1 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 0, 1) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 1, 1);
        flux_sp_qkv_recv_prep_tensor_set(dst, d, seq, out_head, 0, part == 0 ? y0 : y1);
    }
}

inline void flux_sp_qkv_mixed_recv_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVMixedRecvPrepCustomParams params = flux_sp_qkv_mixed_recv_prep_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_mixed_recv_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    GGML_ASSERT(flux_sp_qkv_mixed_recv_prep_shape_supported(recv_flat, pe, plane, params.world_size, params.heads, params.head_dim));
    GGML_ASSERT(flux_sp_qkv_recv_prep_output_type_supported(plane, dst->type));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (head_dim * 2 * shard_heads * world_size);
    const int64_t sequence = shard_sequence * world_size;
    const int64_t plane_idx = params.plane;

    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 1);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;

        if (plane == FluxSPQKVRecvPrepPlane::V) {
            const float value = flux_sp_qkv_mixed_recv_prep_load_recv(recv_flat,
                                                                       plane_idx,
                                                                       d,
                                                                       seq,
                                                                       out_head,
                                                                       head_dim,
                                                                       shard_heads,
                                                                       shard_sequence);
            flux_sp_qkv_recv_prep_tensor_set(dst, d, seq, out_head, 0, value);
            continue;
        }

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        const float x0 = flux_sp_qkv_mixed_recv_prep_load_recv(recv_flat,
                                                                plane_idx,
                                                                pair * 2 + 0,
                                                                seq,
                                                                out_head,
                                                                head_dim,
                                                                shard_heads,
                                                                shard_sequence);
        const float x1 = flux_sp_qkv_mixed_recv_prep_load_recv(recv_flat,
                                                                plane_idx,
                                                                pair * 2 + 1,
                                                                seq,
                                                                out_head,
                                                                head_dim,
                                                                shard_heads,
                                                                shard_sequence);
        const float y0 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 0, 0) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 1, 0);
        const float y1 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 0, 1) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 1, 1);
        flux_sp_qkv_recv_prep_tensor_set(dst, d, seq, out_head, 0, part == 0 ? y0 : y1);
    }
}

inline void flux_sp_qkv_recv_prep_bundle_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVRecvPrepBundleCustomParams params = flux_sp_qkv_recv_prep_bundle_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_recv_prep_bundle_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    GGML_ASSERT(flux_sp_qkv_recv_prep_bundle_shape_supported(recv_flat, pe, params.world_size, params.heads, params.head_dim));
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (head_dim * 3 * shard_heads * world_size);
    const int64_t sequence = shard_sequence * world_size;
    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 3);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        for (int64_t plane_idx = 0; plane_idx < 3; ++plane_idx) {
            if (plane_idx == static_cast<int64_t>(FluxSPQKVRecvPrepPlane::V)) {
                const float value = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                                     plane_idx,
                                                                     pair,
                                                                     part,
                                                                     seq,
                                                                     out_head,
                                                                     head_dim,
                                                                     shard_heads,
                                                                     shard_sequence);
                flux_sp_qkv_recv_prep_tensor_set(dst, d, seq, out_head, plane_idx, value);
                continue;
            }

            const float x0 = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                              plane_idx,
                                                              pair,
                                                              0,
                                                              seq,
                                                              out_head,
                                                              head_dim,
                                                              shard_heads,
                                                              shard_sequence);
            const float x1 = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                              plane_idx,
                                                              pair,
                                                              1,
                                                              seq,
                                                              out_head,
                                                              head_dim,
                                                              shard_heads,
                                                              shard_sequence);
            const float y0 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 0, 0) +
                             x1 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 1, 0);
            const float y1 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 0, 1) +
                             x1 * flux_sp_qkv_recv_prep_pe_at(pe, seq, pair, 1, 1);
            flux_sp_qkv_recv_prep_tensor_set(dst, d, seq, out_head, plane_idx, part == 0 ? y0 : y1);
        }
    }
}

inline void flux_sp_qkv_pair_recv_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVPairRecvPrepCustomParams params = flux_sp_qkv_pair_recv_prep_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_pair_recv_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);
    const ggml_tensor* first_recv_flat = dst->src[0];
    const ggml_tensor* second_recv_flat = dst->src[1];
    const ggml_tensor* pe = dst->src[2];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    GGML_ASSERT(flux_sp_qkv_pair_recv_prep_shape_supported(first_recv_flat,
                                                            second_recv_flat,
                                                            pe,
                                                            plane,
                                                            params.world_size,
                                                            params.heads,
                                                            params.head_dim));
    GGML_ASSERT(flux_sp_qkv_recv_prep_output_type_supported(plane, dst->type));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t denom = head_dim * 3 * shard_heads * world_size;
    const int64_t first_shard_sequence = first_recv_flat->ne[0] / denom;
    const int64_t second_shard_sequence = second_recv_flat->ne[0] / denom;
    const int64_t first_sequence = first_shard_sequence * world_size;
    const int64_t second_sequence = second_shard_sequence * world_size;
    const int64_t sequence = first_sequence + second_sequence;
    const int64_t plane_idx = params.plane;

    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 1);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t total_seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;

        const bool use_first = total_seq < first_sequence;
        const ggml_tensor* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int64_t stream_seq = use_first ? total_seq : total_seq - first_sequence;
        const int64_t shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        if (plane == FluxSPQKVRecvPrepPlane::V) {
            const float value = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                                 plane_idx,
                                                                 d / 2,
                                                                 d & 1,
                                                                 stream_seq,
                                                                 out_head,
                                                                 head_dim,
                                                                 shard_heads,
                                                                 shard_sequence);
            flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, 0, value);
            continue;
        }

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        const float x0 = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                          plane_idx,
                                                          pair,
                                                          0,
                                                          stream_seq,
                                                          out_head,
                                                          head_dim,
                                                          shard_heads,
                                                          shard_sequence);
        const float x1 = flux_sp_qkv_recv_prep_load_recv(recv_flat,
                                                          plane_idx,
                                                          pair,
                                                          1,
                                                          stream_seq,
                                                          out_head,
                                                          head_dim,
                                                          shard_heads,
                                                          shard_sequence);
        const float y0 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 0) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 0);
        const float y1 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 1) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 1);
        flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, 0, part == 0 ? y0 : y1);
    }
}

inline void flux_sp_qkv_pair_mixed_recv_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVPairMixedRecvPrepCustomParams params = flux_sp_qkv_pair_mixed_recv_prep_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_pair_mixed_recv_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);
    const ggml_tensor* first_recv_flat = dst->src[0];
    const ggml_tensor* second_recv_flat = dst->src[1];
    const ggml_tensor* pe = dst->src[2];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    GGML_ASSERT(flux_sp_qkv_pair_mixed_recv_prep_shape_supported(first_recv_flat,
                                                                  second_recv_flat,
                                                                  pe,
                                                                  plane,
                                                                  params.world_size,
                                                                  params.heads,
                                                                  params.head_dim));
    GGML_ASSERT(flux_sp_qkv_recv_prep_output_type_supported(plane, dst->type));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t denom = head_dim * 2 * shard_heads * world_size;
    const int64_t first_shard_sequence = first_recv_flat->ne[0] / denom;
    const int64_t second_shard_sequence = second_recv_flat->ne[0] / denom;
    const int64_t first_sequence = first_shard_sequence * world_size;
    const int64_t second_sequence = second_shard_sequence * world_size;
    const int64_t sequence = first_sequence + second_sequence;
    const int64_t plane_idx = params.plane;

    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 1);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t total_seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;

        const bool use_first = total_seq < first_sequence;
        const ggml_tensor* recv_flat = use_first ? first_recv_flat : second_recv_flat;
        const int64_t stream_seq = use_first ? total_seq : total_seq - first_sequence;
        const int64_t shard_sequence = use_first ? first_shard_sequence : second_shard_sequence;

        if (plane == FluxSPQKVRecvPrepPlane::V) {
            const float value = flux_sp_qkv_mixed_recv_prep_load_recv(recv_flat,
                                                                       plane_idx,
                                                                       d,
                                                                       stream_seq,
                                                                       out_head,
                                                                       head_dim,
                                                                       shard_heads,
                                                                       shard_sequence);
            flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, 0, value);
            continue;
        }

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        const float x0 = flux_sp_qkv_mixed_recv_prep_load_recv(recv_flat,
                                                                plane_idx,
                                                                pair * 2 + 0,
                                                                stream_seq,
                                                                out_head,
                                                                head_dim,
                                                                shard_heads,
                                                                shard_sequence);
        const float x1 = flux_sp_qkv_mixed_recv_prep_load_recv(recv_flat,
                                                                plane_idx,
                                                                pair * 2 + 1,
                                                                stream_seq,
                                                                out_head,
                                                                head_dim,
                                                                shard_heads,
                                                                shard_sequence);
        const float y0 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 0) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 0);
        const float y1 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 1) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 1);
        flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, 0, part == 0 ? y0 : y1);
    }
}

inline void flux_sp_qkv_combined_pair_recv_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVCombinedPairRecvPrepCustomParams params = flux_sp_qkv_combined_pair_recv_prep_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_combined_pair_recv_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    const auto plane = static_cast<FluxSPQKVRecvPrepPlane>(params.plane);
    GGML_ASSERT(flux_sp_qkv_combined_pair_recv_prep_shape_supported(recv_flat,
                                                                     pe,
                                                                     plane,
                                                                     params.world_size,
                                                                     params.heads,
                                                                     params.head_dim,
                                                                     params.first_shard_sequence));
    GGML_ASSERT(flux_sp_qkv_recv_prep_output_type_supported(plane, dst->type));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t combined_shard_sequence = recv_flat->ne[0] / (total_head_dim * shard_heads * world_size);
    const int64_t first_shard_sequence = params.first_shard_sequence;
    const int64_t second_shard_sequence = combined_shard_sequence - first_shard_sequence;
    const int64_t sequence = combined_shard_sequence * world_size;
    const int64_t plane_idx = params.plane;

    GGML_ASSERT(second_shard_sequence > 0);
    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 1);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t total_seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;

        if (plane == FluxSPQKVRecvPrepPlane::V) {
            const float value = flux_sp_qkv_combined_pair_recv_prep_load_recv(recv_flat,
                                                                               plane_idx,
                                                                               d,
                                                                               total_seq,
                                                                               out_head,
                                                                               head_dim,
                                                                               shard_heads,
                                                                               world_size,
                                                                               first_shard_sequence,
                                                                               second_shard_sequence);
            flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, 0, value);
            continue;
        }

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        const float x0 = flux_sp_qkv_combined_pair_recv_prep_load_recv(recv_flat,
                                                                        plane_idx,
                                                                        pair * 2 + 0,
                                                                        total_seq,
                                                                        out_head,
                                                                        head_dim,
                                                                        shard_heads,
                                                                        world_size,
                                                                        first_shard_sequence,
                                                                        second_shard_sequence);
        const float x1 = flux_sp_qkv_combined_pair_recv_prep_load_recv(recv_flat,
                                                                        plane_idx,
                                                                        pair * 2 + 1,
                                                                        total_seq,
                                                                        out_head,
                                                                        head_dim,
                                                                        shard_heads,
                                                                        world_size,
                                                                        first_shard_sequence,
                                                                        second_shard_sequence);
        const float y0 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 0) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 0);
        const float y1 = x0 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 1) +
                         x1 * flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 1);
        flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, 0, part == 0 ? y0 : y1);
    }
}

inline void flux_sp_qkv_combined_pair_recv_prep_bundle_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FluxSPQKVCombinedPairRecvPrepBundleCustomParams params =
        flux_sp_qkv_combined_pair_recv_prep_bundle_params_from_userdata(userdata);
    GGML_ASSERT(flux_sp_qkv_combined_pair_recv_prep_bundle_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* recv_flat = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    GGML_ASSERT(flux_sp_qkv_combined_pair_recv_prep_bundle_shape_supported(recv_flat,
                                                                            pe,
                                                                            params.world_size,
                                                                            params.heads,
                                                                            params.head_dim,
                                                                            params.first_shard_sequence));
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const int64_t head_dim = params.head_dim;
    const int64_t world_size = params.world_size;
    const int64_t heads = params.heads;
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t combined_shard_sequence = recv_flat->ne[0] / (total_head_dim * shard_heads * world_size);
    const int64_t first_shard_sequence = params.first_shard_sequence;
    const int64_t second_shard_sequence = combined_shard_sequence - first_shard_sequence;
    const int64_t sequence = combined_shard_sequence * world_size;

    GGML_ASSERT(second_shard_sequence > 0);
    GGML_ASSERT(dst->ne[0] == head_dim && dst->ne[1] == sequence && dst->ne[2] == shard_heads && dst->ne[3] == 3);

    const int64_t total = head_dim * sequence * shard_heads;
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t total_seq = rem % sequence;
        rem /= sequence;
        const int64_t out_head = rem;

        const int64_t pair = d / 2;
        const int64_t part = d - pair * 2;
        const float pe00 = flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 0);
        const float pe10 = flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 0);
        const float pe01 = flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 0, 1);
        const float pe11 = flux_sp_qkv_recv_prep_pe_at(pe, total_seq, pair, 1, 1);

        for (int64_t plane_idx = 0; plane_idx < 3; ++plane_idx) {
            if (plane_idx == static_cast<int64_t>(FluxSPQKVRecvPrepPlane::V)) {
                const float value = flux_sp_qkv_combined_pair_recv_prep_load_recv(recv_flat,
                                                                                   plane_idx,
                                                                                   d,
                                                                                   total_seq,
                                                                                   out_head,
                                                                                   head_dim,
                                                                                   shard_heads,
                                                                                   world_size,
                                                                                   first_shard_sequence,
                                                                                   second_shard_sequence);
                flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, plane_idx, value);
                continue;
            }

            const float x0 = flux_sp_qkv_combined_pair_recv_prep_load_recv(recv_flat,
                                                                            plane_idx,
                                                                            pair * 2 + 0,
                                                                            total_seq,
                                                                            out_head,
                                                                            head_dim,
                                                                            shard_heads,
                                                                            world_size,
                                                                            first_shard_sequence,
                                                                            second_shard_sequence);
            const float x1 = flux_sp_qkv_combined_pair_recv_prep_load_recv(recv_flat,
                                                                            plane_idx,
                                                                            pair * 2 + 1,
                                                                            total_seq,
                                                                            out_head,
                                                                            head_dim,
                                                                            shard_heads,
                                                                            world_size,
                                                                            first_shard_sequence,
                                                                            second_shard_sequence);
            const float y0 = x0 * pe00 + x1 * pe10;
            const float y1 = x0 * pe01 + x1 * pe11;
            flux_sp_qkv_recv_prep_tensor_set(dst, d, total_seq, out_head, plane_idx, part == 0 ? y0 : y1);
        }
    }
}

inline ggml_tensor* flux_sp_qkv_mixed_recv_prep_custom(ggml_context* ctx,
                                                       ggml_tensor* recv_flat,
                                                       ggml_tensor* pe,
                                                       FluxSPQKVRecvPrepPlane plane,
                                                       int64_t world_size,
                                                       int64_t heads,
                                                       int64_t head_dim,
                                                       ggml_type q_output_type = GGML_TYPE_F32) {
    if (!flux_sp_qkv_mixed_recv_prep_shape_supported(recv_flat, pe, plane, world_size, heads, head_dim)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (head_dim * 2 * shard_heads * world_size);
    const int64_t sequence = shard_sequence * world_size;
    const ggml_type out_type = flux_sp_qkv_recv_prep_output_type(plane, q_output_type);
    ggml_tensor* args[] = { recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      out_type,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      1,
                                      args,
                                      2,
                                      flux_sp_qkv_mixed_recv_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_mixed_recv_prep_params_to_userdata(plane, world_size, heads, head_dim));
    ggml_set_name(out,
                  plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_mixed_q_recv_rope" :
                  plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_mixed_k_recv_rope" :
                                                       "ed_flux_sp_mixed_v_recv_prep");
    return out;
}

inline ggml_tensor* flux_sp_qkv_recv_prep_custom(ggml_context* ctx,
                                                  ggml_tensor* recv_flat,
                                                  ggml_tensor* pe,
                                                  FluxSPQKVRecvPrepPlane plane,
                                                  int64_t world_size,
                                                  int64_t heads,
                                                  int64_t head_dim,
                                                  ggml_type q_output_type = GGML_TYPE_F32) {
    if (!flux_sp_qkv_recv_prep_shape_supported(recv_flat, pe, plane, world_size, heads, head_dim)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (head_dim * 3 * shard_heads * world_size);
    const int64_t sequence = shard_sequence * world_size;
    const ggml_type out_type = flux_sp_qkv_recv_prep_output_type(plane, q_output_type);
    ggml_tensor* args[] = { recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      out_type,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      1,
                                      args,
                                      2,
                                      flux_sp_qkv_recv_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_recv_prep_params_to_userdata(plane, world_size, heads, head_dim));
    ggml_set_name(out,
                  recv_flat->type == GGML_TYPE_F16 ?
                      (plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_f16_q_recv_rope" :
                       plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_f16_k_recv_rope" :
                                                            "ed_flux_sp_f16_v_recv_prep") :
                      (plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_q_recv_rope" :
                       plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_k_recv_rope" :
                                                            "ed_flux_sp_v_recv_prep"));
    return out;
}

inline ggml_tensor* flux_sp_qkv_recv_prep_bundle_custom(ggml_context* ctx,
                                                        ggml_tensor* recv_flat,
                                                        ggml_tensor* pe,
                                                        int64_t world_size,
                                                        int64_t heads,
                                                        int64_t head_dim) {
    if (!flux_sp_qkv_recv_prep_bundle_shape_supported(recv_flat, pe, world_size, heads, head_dim)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t shard_sequence = recv_flat->ne[0] / (head_dim * 3 * shard_heads * world_size);
    const int64_t sequence = shard_sequence * world_size;
    ggml_tensor* args[] = { recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      3,
                                      args,
                                      2,
                                      flux_sp_qkv_recv_prep_bundle_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_recv_prep_bundle_params_to_userdata(world_size, heads, head_dim));
    ggml_set_name(out, "ed_flux_sp_f16_qkv_recv_prep_bundle");
    return out;
}

inline ggml_tensor* flux_sp_qkv_pair_recv_prep_custom(ggml_context* ctx,
                                                      ggml_tensor* first_recv_flat,
                                                      ggml_tensor* second_recv_flat,
                                                       ggml_tensor* pe,
                                                       FluxSPQKVRecvPrepPlane plane,
                                                       int64_t world_size,
                                                       int64_t heads,
                                                       int64_t head_dim,
                                                       ggml_type q_output_type = GGML_TYPE_F32) {
    if (!flux_sp_qkv_pair_recv_prep_shape_supported(first_recv_flat,
                                                     second_recv_flat,
                                                     pe,
                                                     plane,
                                                     world_size,
                                                     heads,
                                                     head_dim)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t denom = head_dim * 3 * shard_heads * world_size;
    const int64_t sequence = (first_recv_flat->ne[0] / denom + second_recv_flat->ne[0] / denom) * world_size;
    const ggml_type out_type = flux_sp_qkv_recv_prep_output_type(plane, q_output_type);
    ggml_tensor* args[] = { first_recv_flat, second_recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      out_type,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      1,
                                      args,
                                      3,
                                      flux_sp_qkv_pair_recv_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_pair_recv_prep_params_to_userdata(plane, world_size, heads, head_dim));
    ggml_set_name(out,
                  first_recv_flat->type == GGML_TYPE_F16 ?
                      (plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_pair_f16_q_recv_rope" :
                       plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_pair_f16_k_recv_rope" :
                                                            "ed_flux_sp_pair_f16_v_recv_prep") :
                      (plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_pair_q_recv_rope" :
                       plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_pair_k_recv_rope" :
                                                            "ed_flux_sp_pair_v_recv_prep"));
    return out;
}

inline ggml_tensor* flux_sp_qkv_pair_mixed_recv_prep_custom(ggml_context* ctx,
                                                            ggml_tensor* first_recv_flat,
                                                            ggml_tensor* second_recv_flat,
                                                            ggml_tensor* pe,
                                                            FluxSPQKVRecvPrepPlane plane,
                                                            int64_t world_size,
                                                            int64_t heads,
                                                            int64_t head_dim,
                                                            ggml_type q_output_type = GGML_TYPE_F32) {
    if (!flux_sp_qkv_pair_mixed_recv_prep_shape_supported(first_recv_flat,
                                                          second_recv_flat,
                                                          pe,
                                                          plane,
                                                          world_size,
                                                          heads,
                                                          head_dim)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t denom = head_dim * 2 * shard_heads * world_size;
    const int64_t sequence = (first_recv_flat->ne[0] / denom + second_recv_flat->ne[0] / denom) * world_size;
    const ggml_type out_type = flux_sp_qkv_recv_prep_output_type(plane, q_output_type);
    ggml_tensor* args[] = { first_recv_flat, second_recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      out_type,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      1,
                                      args,
                                      3,
                                      flux_sp_qkv_pair_mixed_recv_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_pair_mixed_recv_prep_params_to_userdata(plane, world_size, heads, head_dim));
    ggml_set_name(out,
                  plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_pair_mixed_q_recv_rope" :
                  plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_pair_mixed_k_recv_rope" :
                                                       "ed_flux_sp_pair_mixed_v_recv_prep");
    return out;
}

inline ggml_tensor* flux_sp_qkv_combined_pair_recv_prep_custom(ggml_context* ctx,
                                                               ggml_tensor* recv_flat,
                                                               ggml_tensor* pe,
                                                               FluxSPQKVRecvPrepPlane plane,
                                                               int64_t world_size,
                                                               int64_t heads,
                                                               int64_t head_dim,
                                                               int64_t first_shard_sequence,
                                                               ggml_type q_output_type = GGML_TYPE_F32) {
    if (!flux_sp_qkv_combined_pair_recv_prep_shape_supported(recv_flat,
                                                              pe,
                                                              plane,
                                                              world_size,
                                                              heads,
                                                              head_dim,
                                                              first_shard_sequence)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t denom = head_dim * 3 * shard_heads * world_size;
    const int64_t sequence = (recv_flat->ne[0] / denom) * world_size;
    const ggml_type out_type = flux_sp_qkv_recv_prep_output_type(plane, q_output_type);
    ggml_tensor* args[] = { recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      out_type,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      1,
                                      args,
                                      2,
                                      flux_sp_qkv_combined_pair_recv_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_combined_pair_recv_prep_params_to_userdata(plane,
                                                                                            world_size,
                                                                                            heads,
                                                                                            head_dim,
                                                                                            first_shard_sequence));
    ggml_set_name(out,
                  recv_flat->type == GGML_TYPE_F16 ?
                      (plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_combined_pair_f16_q_recv_rope" :
                       plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_combined_pair_f16_k_recv_rope" :
                                                            "ed_flux_sp_combined_pair_f16_v_recv_prep") :
                      (plane == FluxSPQKVRecvPrepPlane::Q ? "ed_flux_sp_combined_pair_q_recv_rope" :
                       plane == FluxSPQKVRecvPrepPlane::K ? "ed_flux_sp_combined_pair_k_recv_rope" :
                                                            "ed_flux_sp_combined_pair_v_recv_prep"));
    return out;
}

inline ggml_tensor* flux_sp_qkv_combined_pair_recv_prep_bundle_custom(ggml_context* ctx,
                                                                      ggml_tensor* recv_flat,
                                                                      ggml_tensor* pe,
                                                                      int64_t world_size,
                                                                      int64_t heads,
                                                                      int64_t head_dim,
                                                                      int64_t first_shard_sequence) {
    if (!flux_sp_qkv_combined_pair_recv_prep_bundle_shape_supported(recv_flat,
                                                                     pe,
                                                                     world_size,
                                                                     heads,
                                                                     head_dim,
                                                                     first_shard_sequence)) {
        return nullptr;
    }
    const int64_t shard_heads = heads / world_size;
    const int64_t denom = head_dim * 3 * shard_heads * world_size;
    const int64_t sequence = (recv_flat->ne[0] / denom) * world_size;
    ggml_tensor* args[] = { recv_flat, pe };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      head_dim,
                                      sequence,
                                      shard_heads,
                                      3,
                                      args,
                                      2,
                                      flux_sp_qkv_combined_pair_recv_prep_bundle_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      flux_sp_qkv_combined_pair_recv_prep_bundle_params_to_userdata(world_size,
                                                                                                    heads,
                                                                                                    head_dim,
                                                                                                    first_shard_sequence));
    ggml_set_name(out, "ed_flux_sp_combined_pair_f16_qkv_recv_prep_bundle");
    return out;
}

} // namespace edgedit::ggml_ext
