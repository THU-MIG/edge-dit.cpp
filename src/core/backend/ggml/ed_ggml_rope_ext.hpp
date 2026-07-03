#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace edgedit::ggml_ext {

constexpr uint32_t kRopeCustomMagic = 0x45525250u; // "ERRP"

enum class RopeInputLayout : int32_t {
    NSeqHeadDim = 0,
    SeqHeadDim  = 1,
    Work        = 2,
};

struct RopeCustomParams {
    uint32_t magic = kRopeCustomMagic;
    int32_t input_layout = static_cast<int32_t>(RopeInputLayout::NSeqHeadDim);
    int32_t interleaved = 1;
    int32_t d_head = 0;
};

inline bool rope_fast_path_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_ROPE");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline bool rope_f16_output_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_ROPE_F16_OUTPUT");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline RopeCustomParams rope_params_from_userdata(void* userdata) {
    RopeCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.input_layout = static_cast<int32_t>((packed >> 32) & 0xffu);
    params.interleaved = static_cast<int32_t>((packed >> 40) & 0xffu);
    params.d_head = static_cast<int32_t>((packed >> 48) & 0xffffu);
    return params;
}

inline void* rope_params_to_userdata(RopeInputLayout layout, bool interleaved, int64_t d_head) {
    uintptr_t packed = static_cast<uintptr_t>(kRopeCustomMagic);
    packed |= (static_cast<uintptr_t>(static_cast<uint8_t>(layout)) << 32);
    packed |= (static_cast<uintptr_t>(interleaved ? 1u : 0u) << 40);
    packed |= (static_cast<uintptr_t>(static_cast<uint16_t>(std::max<int64_t>(0, d_head))) << 48);
    return reinterpret_cast<void*>(packed);
}

inline bool rope_params_valid(const RopeCustomParams& params) {
    return params.magic == kRopeCustomMagic &&
           params.d_head > 0 &&
           (params.d_head % 2) == 0 &&
           (params.input_layout == static_cast<int32_t>(RopeInputLayout::NSeqHeadDim) ||
            params.input_layout == static_cast<int32_t>(RopeInputLayout::SeqHeadDim) ||
            params.input_layout == static_cast<int32_t>(RopeInputLayout::Work));
}

inline float tensor_f32_at(const ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    return *reinterpret_cast<const float*>(ptr);
}

inline void tensor_f32_set(ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    *reinterpret_cast<float*>(ptr) = v;
}

inline void tensor_rope_set(ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    if (t->type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(ptr) = v;
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F16);
        *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(v);
    }
}

inline float rope_pe_at(const ggml_tensor* pe, int64_t seq, int64_t pair, int64_t row, int64_t col) {
    if (pe->ne[0] == 2 && pe->ne[1] == 2) {
        return tensor_f32_at(pe, row, col, pair, seq);
    }
    return tensor_f32_at(pe, row, pair, seq, col);
}

inline void rope_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(nth);

    const RopeCustomParams params = rope_params_from_userdata(userdata);
    GGML_ASSERT(rope_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* pe = dst->src[1];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32 &&
                (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16));

    const int64_t d_head = params.d_head;
    const int64_t half = d_head / 2;
    const bool interleaved = params.interleaved != 0;

    const auto layout = static_cast<RopeInputLayout>(params.input_layout);
    int64_t seq = 0;
    int64_t heads_total = 0;
    if (layout == RopeInputLayout::NSeqHeadDim) {
        seq = x->ne[2];
        heads_total = x->ne[1] * x->ne[3];
    } else if (layout == RopeInputLayout::SeqHeadDim) {
        seq = x->ne[1];
        heads_total = x->ne[2] * x->ne[3];
    } else {
        seq = x->ne[1];
        heads_total = x->ne[2];
    }

    for (int64_t h = 0; h < heads_total; ++h) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t p = 0; p < half; ++p) {
                float x0 = 0.0f;
                float x1 = 0.0f;
                if (layout == RopeInputLayout::NSeqHeadDim) {
                    const int64_t n_head = x->ne[1];
                    const int64_t head = h % n_head;
                    const int64_t n = h / n_head;
                    if (interleaved) {
                        x0 = tensor_f32_at(x, 2 * p + 0, head, s, n);
                        x1 = tensor_f32_at(x, 2 * p + 1, head, s, n);
                    } else {
                        x0 = tensor_f32_at(x, p, head, s, n);
                        x1 = tensor_f32_at(x, p + half, head, s, n);
                    }
                } else if (layout == RopeInputLayout::SeqHeadDim) {
                    const int64_t n_head = x->ne[2];
                    const int64_t head = h % n_head;
                    const int64_t n = h / n_head;
                    if (interleaved) {
                        x0 = tensor_f32_at(x, 2 * p + 0, s, head, n);
                        x1 = tensor_f32_at(x, 2 * p + 1, s, head, n);
                    } else {
                        x0 = tensor_f32_at(x, p, s, head, n);
                        x1 = tensor_f32_at(x, p + half, s, head, n);
                    }
                } else {
                    x0 = tensor_f32_at(x, p, s, h, 0);
                    x1 = tensor_f32_at(x, p, s, h, 1);
                }

                const float y0 = x0 * rope_pe_at(pe, s, p, 0, 0) + x1 * rope_pe_at(pe, s, p, 1, 0);
                const float y1 = x0 * rope_pe_at(pe, s, p, 0, 1) + x1 * rope_pe_at(pe, s, p, 1, 1);
                if (interleaved || layout == RopeInputLayout::Work) {
                    tensor_rope_set(dst, 2 * p + 0, s, h, 0, y0);
                    tensor_rope_set(dst, 2 * p + 1, s, h, 0, y1);
                } else {
                    tensor_rope_set(dst, p, s, h, 0, y0);
                    tensor_rope_set(dst, p + half, s, h, 0, y1);
                }
            }
        }
    }
}

inline bool rope_custom_shape_supported(const ggml_tensor* x,
                                        const ggml_tensor* pe,
                                        RopeInputLayout layout,
                                        bool interleaved,
                                        int64_t d_head) {
    if (!rope_fast_path_enabled() || x == nullptr || pe == nullptr) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || pe->type != GGML_TYPE_F32 || d_head <= 0 || (d_head % 2) != 0) {
        return false;
    }
    if (!interleaved && layout == RopeInputLayout::Work) {
        return false;
    }
    const int64_t half = d_head / 2;
    const bool pe_matrix = pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] >= half;
    const bool pe_prepared = pe->ne[0] == 2 && pe->ne[1] >= half && pe->ne[3] == 2;
    if (!pe_matrix && !pe_prepared) {
        return false;
    }
    if (layout == RopeInputLayout::NSeqHeadDim) {
        return x->ne[0] == d_head && x->ne[1] > 0 && x->ne[2] > 0 && x->ne[3] > 0 &&
               ((pe_matrix && pe->ne[3] >= x->ne[2]) || (pe_prepared && pe->ne[2] >= x->ne[2]));
    }
    if (layout == RopeInputLayout::SeqHeadDim) {
        return x->ne[0] == d_head && x->ne[1] > 0 && x->ne[2] > 0 && x->ne[3] > 0 &&
               ((pe_matrix && pe->ne[3] >= x->ne[1]) || (pe_prepared && pe->ne[2] >= x->ne[1]));
    }
    return x->ne[0] == half && x->ne[1] > 0 && x->ne[2] > 0 && x->ne[3] == 2 &&
           ((pe_matrix && pe->ne[3] >= x->ne[1]) || (pe_prepared && pe->ne[2] >= x->ne[1]));
}

inline ggml_tensor* rope_custom_3d(ggml_context* ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* pe,
                                   RopeInputLayout layout,
                                   bool interleaved,
                                   int64_t d_head,
                                   ggml_type out_type = GGML_TYPE_F32) {
    GGML_ASSERT(out_type == GGML_TYPE_F32 || out_type == GGML_TYPE_F16);
    ggml_tensor* args[] = { x, pe };
    int64_t seq = 0;
    int64_t heads_total = 0;
    if (layout == RopeInputLayout::NSeqHeadDim) {
        seq = x->ne[2];
        heads_total = x->ne[1] * x->ne[3];
    } else if (layout == RopeInputLayout::SeqHeadDim) {
        seq = x->ne[1];
        heads_total = x->ne[2] * x->ne[3];
    } else {
        seq = x->ne[1];
        heads_total = x->ne[2];
    }
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      out_type,
                                      d_head,
                                      seq,
                                      heads_total,
                                      1,
                                      args,
                                      2,
                                      rope_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      rope_params_to_userdata(layout, interleaved, d_head));
    ggml_set_name(out, "ed_fused_rope");
    return out;
}

inline ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe, bool interleaved) {
    const int64_t d_head = x->ne[0];
    if (rope_custom_shape_supported(x, pe, RopeInputLayout::NSeqHeadDim, interleaved, d_head)) {
        return rope_custom_3d(ctx, x, pe, RopeInputLayout::NSeqHeadDim, interleaved, d_head);
    }
    return nullptr;
}

inline ggml_tensor* apply_rope_seq_major(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe, bool interleaved) {
    const int64_t d_head = x->ne[0];
    if (rope_custom_shape_supported(x, pe, RopeInputLayout::SeqHeadDim, interleaved, d_head)) {
        return rope_custom_3d(ctx, x, pe, RopeInputLayout::SeqHeadDim, interleaved, d_head);
    }
    return nullptr;
}

inline ggml_tensor* apply_rope_work_layout(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe, int64_t d_head) {
    if (rope_custom_shape_supported(x, pe, RopeInputLayout::Work, true, d_head)) {
        return rope_custom_3d(ctx, x, pe, RopeInputLayout::Work, true, d_head);
    }
    return nullptr;
}

inline ggml_tensor* apply_rope_f16(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe, bool interleaved) {
    const int64_t d_head = x->ne[0];
    if (rope_f16_output_enabled() && rope_custom_shape_supported(x, pe, RopeInputLayout::NSeqHeadDim, interleaved, d_head)) {
        return rope_custom_3d(ctx, x, pe, RopeInputLayout::NSeqHeadDim, interleaved, d_head, GGML_TYPE_F16);
    }
    return nullptr;
}

inline ggml_tensor* apply_rope_seq_major_f16(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe, bool interleaved) {
    const int64_t d_head = x->ne[0];
    if (rope_f16_output_enabled() && rope_custom_shape_supported(x, pe, RopeInputLayout::SeqHeadDim, interleaved, d_head)) {
        return rope_custom_3d(ctx, x, pe, RopeInputLayout::SeqHeadDim, interleaved, d_head, GGML_TYPE_F16);
    }
    return nullptr;
}

inline ggml_tensor* apply_rope_work_layout_f16(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe, int64_t d_head) {
    if (rope_f16_output_enabled() && rope_custom_shape_supported(x, pe, RopeInputLayout::Work, true, d_head)) {
        return rope_custom_3d(ctx, x, pe, RopeInputLayout::Work, true, d_head, GGML_TYPE_F16);
    }
    return nullptr;
}

} // namespace edgedit::ggml_ext
