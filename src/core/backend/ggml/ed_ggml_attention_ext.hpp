#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdlib>

namespace edgedit::ggml_ext {

constexpr uint32_t kAttentionVPrepCustomMagic = 0x45565050u; // "EVPP"

struct AttentionVPrepCustomParams {
    uint32_t magic = kAttentionVPrepCustomMagic;
    int32_t v_is_seq_major = 0;
};

inline bool attention_v_prep_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_ATTENTION_V_PREP");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline AttentionVPrepCustomParams attention_v_prep_params_from_userdata(void* userdata) {
    AttentionVPrepCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.v_is_seq_major = static_cast<int32_t>((packed >> 32) & 0xffu);
    return params;
}

inline void* attention_v_prep_params_to_userdata(bool v_is_seq_major) {
    uintptr_t packed = static_cast<uintptr_t>(kAttentionVPrepCustomMagic);
    packed |= (static_cast<uintptr_t>(v_is_seq_major ? 1u : 0u) << 32);
    return reinterpret_cast<void*>(packed);
}

inline bool attention_v_prep_params_valid(const AttentionVPrepCustomParams& params) {
    return params.magic == kAttentionVPrepCustomMagic &&
           (params.v_is_seq_major == 0 || params.v_is_seq_major == 1);
}

inline float attention_tensor_f32_at(const ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    return *reinterpret_cast<const float*>(ptr);
}

inline void attention_tensor_f16_set(ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F16);
    *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(v);
}

inline bool attention_v_prep_shape_supported(const ggml_tensor* v, bool v_is_seq_major) {
    if (!attention_v_prep_enabled() || v == nullptr || v->type != GGML_TYPE_F32) {
        return false;
    }
    if (v->ne[0] <= 0 || v->ne[1] <= 0 || v->ne[2] <= 0 || v->ne[3] <= 0) {
        return false;
    }
    const int64_t d_head = v->ne[0];
    const int64_t seq = v_is_seq_major ? v->ne[1] : v->ne[2];
    const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
    return d_head > 0 && seq > 0 && n_head > 0;
}

inline void attention_v_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(nth);

    const AttentionVPrepCustomParams params = attention_v_prep_params_from_userdata(userdata);
    GGML_ASSERT(attention_v_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr);
    const ggml_tensor* v = dst->src[0];
    const bool v_is_seq_major = params.v_is_seq_major != 0;
    GGML_ASSERT(attention_v_prep_shape_supported(v, v_is_seq_major));
    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    const int64_t d_head = v->ne[0];
    const int64_t seq = v_is_seq_major ? v->ne[1] : v->ne[2];
    const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
    const int64_t batch = v->ne[3];
    GGML_ASSERT(dst->ne[0] == d_head && dst->ne[1] == seq && dst->ne[2] == n_head * batch);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t d = 0; d < d_head; ++d) {
                    const float value = v_is_seq_major ?
                                            attention_tensor_f32_at(v, d, s, h, b) :
                                            attention_tensor_f32_at(v, d, h, s, b);
                    attention_tensor_f16_set(dst, d, s, h + b * n_head, 0, value);
                }
            }
        }
    }
}

inline ggml_tensor* attention_v_prep_custom_f16(ggml_context* ctx, ggml_tensor* v, bool v_is_seq_major) {
    if (!attention_v_prep_shape_supported(v, v_is_seq_major)) {
        return nullptr;
    }
    ggml_tensor* args[] = { v };
    const int64_t d_head = v->ne[0];
    const int64_t seq = v_is_seq_major ? v->ne[1] : v->ne[2];
    const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
    const int64_t batch = v->ne[3];
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      d_head,
                                      seq,
                                      n_head * batch,
                                      1,
                                      args,
                                      1,
                                      attention_v_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      attention_v_prep_params_to_userdata(v_is_seq_major));
    ggml_set_name(out, "ed_fused_attention_v_f16");
    return out;
}

} // namespace edgedit::ggml_ext
