#ifndef __ED_PARALLEL_SP_RECV_PLACEHOLDER_HPP__
#define __ED_PARALLEL_SP_RECV_PLACEHOLDER_HPP__

#include <cstdint>
#include <cstring>

#include "ggml.h"

namespace edgedit::parallel {

struct SPRecvPlaceholderParams {
    uint32_t magic;
    int32_t mode;
    int32_t world_size;
};

constexpr uint32_t SP_RECV_PLACEHOLDER_MAGIC = 0x45535052u; // "ESPR"
constexpr int32_t SP_RECV_PLACEHOLDER_NOOP_MODE = 42;

inline SPRecvPlaceholderParams sp_recv_placeholder_params_from_userdata(void* userdata) {
    SPRecvPlaceholderParams params{};
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.world_size = static_cast<int32_t>((packed >> 32) & 0xffffu);
    params.mode = static_cast<int32_t>((packed >> 48) & 0xffu);
    return params;
}

inline void* sp_recv_placeholder_params_to_userdata(int64_t world_size) {
    uintptr_t packed = static_cast<uintptr_t>(SP_RECV_PLACEHOLDER_MAGIC);
    packed |= (static_cast<uintptr_t>(static_cast<uint16_t>(world_size)) << 32);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(SP_RECV_PLACEHOLDER_NOOP_MODE)) << 48);
    return reinterpret_cast<void*>(packed);
}

inline bool sp_recv_placeholder_params_valid(const SPRecvPlaceholderParams& params) {
    return params.magic == SP_RECV_PLACEHOLDER_MAGIC &&
           params.mode == SP_RECV_PLACEHOLDER_NOOP_MODE &&
           params.world_size > 1;
}

inline bool sp_recv_placeholder_custom_supported(const ggml_tensor* dst) {
    if (dst == nullptr ||
        dst->op != GGML_OP_CUSTOM ||
        dst->src[0] == nullptr ||
        dst->src[1] == nullptr) {
        return false;
    }

    struct ggml_custom_op_params {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    ggml_custom_op_params op_params{};
    static_assert(sizeof(op_params) <= GGML_MAX_OP_PARAMS, "custom op params do not fit");
    std::memcpy(&op_params, dst->op_params, sizeof(op_params));

    return sp_recv_placeholder_params_valid(sp_recv_placeholder_params_from_userdata(op_params.userdata));
}

inline bool sp_recv_placeholder_custom_compute(ggml_tensor* dst) {
    return sp_recv_placeholder_custom_supported(dst);
}

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_SP_RECV_PLACEHOLDER_HPP__
