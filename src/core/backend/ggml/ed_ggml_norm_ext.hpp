#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace edgedit::ggml_ext {

constexpr uint32_t kChannelRmsNormCustomMagic = 0x4543524eu; // "ECRN"

struct ChannelRmsNormCustomParams {
    uint32_t magic = kChannelRmsNormCustomMagic;
};

inline ChannelRmsNormCustomParams channel_rms_norm_params_from_userdata(void* userdata) {
    ChannelRmsNormCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* channel_rms_norm_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kChannelRmsNormCustomMagic));
}

inline bool channel_rms_norm_params_valid(const ChannelRmsNormCustomParams& params) {
    return params.magic == kChannelRmsNormCustomMagic;
}

inline bool channel_rms_norm_shape_supported(const ggml_tensor* x, const ggml_tensor* weight) {
    if (x == nullptr || weight == nullptr) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->ne[0] <= 0 || x->ne[1] <= 0 || x->ne[2] <= 0 || x->ne[3] <= 0) {
        return false;
    }
    if (weight->ne[0] != x->ne[3] || ggml_nelements(weight) != x->ne[3]) {
        return false;
    }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(weight)) {
        return false;
    }
    return true;
}

inline float channel_rms_norm_tensor_f32_at(const ggml_tensor* t,
                                            int64_t i0,
                                            int64_t i1,
                                            int64_t i2,
                                            int64_t i3) {
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    return *reinterpret_cast<const float*>(ptr);
}

inline void channel_rms_norm_tensor_f32_set(ggml_tensor* t,
                                            int64_t i0,
                                            int64_t i1,
                                            int64_t i2,
                                            int64_t i3,
                                            float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    *reinterpret_cast<float*>(ptr) = v;
}

inline void channel_rms_norm_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(nth);

    const ChannelRmsNormCustomParams params = channel_rms_norm_params_from_userdata(userdata);
    GGML_ASSERT(channel_rms_norm_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* weight = dst->src[1];
    GGML_ASSERT(channel_rms_norm_shape_supported(x, weight));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        GGML_ASSERT(dst->ne[i] == x->ne[i]);
    }

    constexpr float eps = 1e-12f;
    const int64_t channels = x->ne[3];
    for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
        for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < x->ne[0]; ++i0) {
                float sum = 0.0f;
                for (int64_t c = 0; c < channels; ++c) {
                    const float v = channel_rms_norm_tensor_f32_at(x, i0, i1, i2, c);
                    sum += v * v;
                }
                const float scale = 1.0f / std::sqrt(sum / static_cast<float>(channels) + eps);
                for (int64_t c = 0; c < channels; ++c) {
                    const float v = channel_rms_norm_tensor_f32_at(x, i0, i1, i2, c);
                    const float w = channel_rms_norm_tensor_f32_at(weight, c, 0, 0, 0);
                    channel_rms_norm_tensor_f32_set(dst, i0, i1, i2, c, v * scale * w);
                }
            }
        }
    }
}

inline ggml_tensor* channel_rms_norm_custom(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight) {
    if (!channel_rms_norm_shape_supported(x, weight)) {
        return nullptr;
    }
    ggml_tensor* args[] = { x, weight };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      x->ne[0],
                                      x->ne[1],
                                      x->ne[2],
                                      x->ne[3],
                                      args,
                                      2,
                                      channel_rms_norm_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      channel_rms_norm_params_to_userdata());
    ggml_set_name(out, "ed_channel_rms_norm");
    return out;
}

} // namespace edgedit::ggml_ext
