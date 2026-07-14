#ifndef __QWEN_IMAGE_HPP__
#define __QWEN_IMAGE_HPP__

#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "dit_models/components/common/common_block.hpp"
#include "dit_models/components/common/common_dit.hpp"
#include "dit_models/components/common/modulation.hpp"
#include "dit_models/components/common/rope.hpp"
#include "backend/ggml/ed_ggml_rope_ext.hpp"
#include "parallel/sp_parallel.hpp"
#ifdef ED_ENABLE_CUDA_MODULATION
#include "backend/ggml/ed_ggml_modulation_ext.hpp"
#include "optimization/cache/compile/indicator_lowering.hpp"
#endif

namespace Qwen {
// Qwen SP custom ops: the CUDA backend intercepts these ggml custom nodes and
// runs the fused pack kernels in ggml-cuda. The CPU callbacks keep the graph
// valid on non-CUDA backends and define the exact layout contract.
struct QwenFusedQKVPackParams {
    uint64_t magic;
    int64_t txt_real_seq;
    int64_t img_real_seq;
    int64_t mode;
    int64_t txt_padded_seq;
    int64_t img_padded_seq;
    int64_t world_size;
    int64_t stream_index;
    float scale;
};

constexpr uint64_t QWEN_FUSED_QKV_PACK_MAGIC = 0x5157454e46514b56ULL; // "QWENFQKV"

static const QwenFusedQKVPackParams QWEN_FUSED_QKV_PACK_USERDATA = {
    QWEN_FUSED_QKV_PACK_MAGIC,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.0f,
};

static inline bool qwen_env_flag_enabled_or_default(const char* name, bool default_enabled) {
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

static inline bool qwen_sp_fuse_qkv_a2a_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SP_FUSE_QKV_A2A", true);
}

static inline bool qwen_sp_f16_qkv_send_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SP_F16_QKV_SEND", true);
}

static inline bool qwen_sp_f16_head_to_seq_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SP_F16_HEAD_TO_SEQ", true);
}

static inline bool qwen_sp_f16_q_attention_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SP_F16_Q_ATTENTION", false);
}

static inline bool qwen_sp_unpadded_flash_attn_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SP_UNPADDED_FLASH_ATTN", true);
}

static inline bool qwen_sp_cont_flash_attn_output_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SP_CONT_FLASH_ATTN_OUTPUT", false);
}

static inline bool qwen_single_fused_attention_enabled() {
    return qwen_env_flag_enabled_or_default("ED_QWEN_SINGLE_FUSED_ATTENTION", true);
}

static std::mutex& qwen_fused_qkv_pack_params_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::vector<std::unique_ptr<QwenFusedQKVPackParams>>& qwen_fused_qkv_pack_params_store() {
    static std::vector<std::unique_ptr<QwenFusedQKVPackParams>> store;
    return store;
}

static QwenFusedQKVPackParams* qwen_fused_qkv_pack_make_params(int64_t txt_real_seq,
                                                               int64_t img_real_seq,
                                                               int64_t mode) {
    auto params = std::make_unique<QwenFusedQKVPackParams>();
    params->magic = QWEN_FUSED_QKV_PACK_MAGIC;
    params->txt_real_seq = txt_real_seq;
    params->img_real_seq = img_real_seq;
    params->mode = mode;
    params->txt_padded_seq = 0;
    params->img_padded_seq = 0;
    params->world_size = 0;
    params->stream_index = 0;
    params->scale = 0.0f;
    QwenFusedQKVPackParams* raw = params.get();
    std::lock_guard<std::mutex> lock(qwen_fused_qkv_pack_params_mutex());
    qwen_fused_qkv_pack_params_store().push_back(std::move(params));
    return raw;
}

static QwenFusedQKVPackParams* qwen_fused_qkv_pack_make_params(int64_t txt_real_seq,
                                                               int64_t img_real_seq,
                                                               int64_t mode,
                                                               int64_t txt_padded_seq,
                                                               int64_t img_padded_seq,
                                                               int64_t world_size,
                                                               int64_t stream_index = 0) {
    auto params = std::make_unique<QwenFusedQKVPackParams>();
    params->magic = QWEN_FUSED_QKV_PACK_MAGIC;
    params->txt_real_seq = txt_real_seq;
    params->img_real_seq = img_real_seq;
    params->mode = mode;
    params->txt_padded_seq = txt_padded_seq;
    params->img_padded_seq = img_padded_seq;
    params->world_size = world_size;
    params->stream_index = stream_index;
    params->scale = 0.0f;
    QwenFusedQKVPackParams* raw = params.get();
    std::lock_guard<std::mutex> lock(qwen_fused_qkv_pack_params_mutex());
    qwen_fused_qkv_pack_params_store().push_back(std::move(params));
    return raw;
}

static QwenFusedQKVPackParams* qwen_fused_qkv_pack_make_scale_params(float scale) {
    auto params = std::make_unique<QwenFusedQKVPackParams>();
    params->magic = QWEN_FUSED_QKV_PACK_MAGIC;
    params->txt_real_seq = 0;
    params->img_real_seq = 0;
    params->mode = 52;
    params->txt_padded_seq = 0;
    params->img_padded_seq = 0;
    params->world_size = 0;
    params->stream_index = 0;
    params->scale = scale;
    QwenFusedQKVPackParams* raw = params.get();
    std::lock_guard<std::mutex> lock(qwen_fused_qkv_pack_params_mutex());
    qwen_fused_qkv_pack_params_store().push_back(std::move(params));
    return raw;
}

static inline float qwen_fused_qkv_pack_get_f32(const ggml_tensor* src,
                                                int64_t i0,
                                                int64_t i1,
                                                int64_t i2,
                                                int64_t i3) {
    const char* data = static_cast<const char*>(src->data);
    return *reinterpret_cast<const float*>(data +
                                           i0 * src->nb[0] +
                                           i1 * src->nb[1] +
                                           i2 * src->nb[2] +
                                           i3 * src->nb[3]);
}

static inline void qwen_fused_qkv_pack_set_f32(ggml_tensor* dst,
                                               int64_t i0,
                                               int64_t i1,
                                               int64_t i2,
                                               int64_t i3,
                                               float value) {
    char* data = static_cast<char*>(dst->data);
    *reinterpret_cast<float*>(data +
                              i0 * dst->nb[0] +
                              i1 * dst->nb[1] +
                              i2 * dst->nb[2] +
                              i3 * dst->nb[3]) = value;
}

static inline float qwen_fused_pack_get_f32_or_f16(const ggml_tensor* src,
                                                   int64_t i0,
                                                   int64_t i1,
                                                   int64_t i2,
                                                   int64_t i3) {
    const char* data = static_cast<const char*>(src->data);
    const char* ptr = data +
                      i0 * src->nb[0] +
                      i1 * src->nb[1] +
                      i2 * src->nb[2] +
                      i3 * src->nb[3];
    if (src->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
    }
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    return *reinterpret_cast<const float*>(ptr);
}

static inline void qwen_fused_pack_set_f32_or_f16(ggml_tensor* dst,
                                                  int64_t i0,
                                                  int64_t i1,
                                                  int64_t i2,
                                                  int64_t i3,
                                                  float value) {
    char* data = static_cast<char*>(dst->data);
    char* ptr = data +
                i0 * dst->nb[0] +
                i1 * dst->nb[1] +
                i2 * dst->nb[2] +
                i3 * dst->nb[3];
    if (dst->type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(value);
        return;
    }
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    *reinterpret_cast<float*>(ptr) = value;
}

static inline void qwen_fused_scale_to_f16_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 52);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));

    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
        int64_t rem = i;
        const int64_t i0 = rem % dst->ne[0];
        rem /= dst->ne[0];
        const int64_t i1 = rem % dst->ne[1];
        rem /= dst->ne[1];
        const int64_t i2 = rem % dst->ne[2];
        rem /= dst->ne[2];
        const int64_t i3 = rem;
        const float value = qwen_fused_pack_get_f32_or_f16(src, i0, i1, i2, i3) * params->scale;
        qwen_fused_pack_set_f32_or_f16(dst, i0, i1, i2, i3, value);
    }
}

static inline void qwen_fused_qkv_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* txt_q = dst->src[0];
    const ggml_tensor* img_q = dst->src[1];
    const ggml_tensor* txt_k = dst->src[2];
    const ggml_tensor* img_k = dst->src[3];
    const ggml_tensor* txt_v = dst->src[4];
    const ggml_tensor* img_v = dst->src[5];
    const ggml_tensor* srcs[3][2] = {{txt_q, img_q}, {txt_k, img_k}, {txt_v, img_v}};

    const int64_t txt_seq     = dst->src[0]->ne[1];
    const int64_t img_seq     = dst->src[1]->ne[1];
    const int64_t seq_total   = txt_seq + img_seq;
    const int64_t qk_half_dim = dst->ne[0];
    const int64_t head_dim    = qk_half_dim * 2;
    const int64_t n_heads     = dst->ne[2];
    const int64_t plane_elems = qk_half_dim * seq_total * n_heads;

    for (int64_t linear = ith; linear < 6 * plane_elems; linear += nth) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % seq_total;
        const int64_t head  = rem / seq_total;
        const bool is_txt   = tok < txt_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part      = plane % 2;
        const ggml_tensor* src  = srcs[qkv_plane][is_txt ? 0 : 1];

        float value = 0.0f;
        if (qkv_plane < 2) {
            value = qwen_fused_qkv_pack_get_f32(src, half, src_t, head, part);
        } else {
            const int64_t d = half + part * qk_half_dim;
            GGML_ASSERT(d < head_dim);
            value = qwen_fused_qkv_pack_get_f32(src, d, src_t, head, 0);
        }
        qwen_fused_qkv_pack_set_f32(dst, half, tok, head, plane, value);
    }
}

static inline void qwen_fused_qk_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* txt_q = dst->src[0];
    const ggml_tensor* img_q = dst->src[1];
    const ggml_tensor* txt_k = dst->src[2];
    const ggml_tensor* img_k = dst->src[3];
    const ggml_tensor* srcs[2][2] = {{txt_q, img_q}, {txt_k, img_k}};

    const int64_t txt_seq     = dst->src[0]->ne[1];
    const int64_t img_seq     = dst->src[1]->ne[1];
    const int64_t seq_total   = txt_seq + img_seq;
    const int64_t qk_half_dim = dst->ne[0];
    const int64_t n_heads     = dst->ne[2];
    const int64_t plane_elems = qk_half_dim * seq_total * n_heads;
    GGML_ASSERT(dst->ne[3] == 4);

    for (int64_t linear = ith; linear < 4 * plane_elems; linear += nth) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % seq_total;
        const int64_t head  = rem / seq_total;
        const bool is_txt   = tok < txt_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_seq;
        const int64_t qk_plane = plane / 2;
        const int64_t part     = plane % 2;
        const ggml_tensor* src = srcs[qk_plane][is_txt ? 0 : 1];

        const float value = qwen_fused_qkv_pack_get_f32(src, half, src_t, head, part);
        qwen_fused_qkv_pack_set_f32(dst, half, tok, head, plane, value);
    }
}

static inline void qwen_fused_v_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* txt_v = dst->src[0];
    const ggml_tensor* img_v = dst->src[1];
    const int64_t txt_seq    = txt_v->ne[1];
    const int64_t img_seq    = img_v->ne[1];
    const int64_t seq_total  = txt_seq + img_seq;
    const int64_t head_dim   = dst->ne[0];
    const int64_t n_heads    = dst->ne[2];
    const int64_t total      = head_dim * seq_total * n_heads;
    GGML_ASSERT(dst->ne[3] == 1);

    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t tok  = rem % seq_total;
        const int64_t head = rem / seq_total;
        const bool is_txt  = tok < txt_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_seq;
        const ggml_tensor* src = is_txt ? txt_v : img_v;
        const float value = qwen_fused_qkv_pack_get_f32(src, d, src_t, head, 0);
        qwen_fused_qkv_pack_set_f32(dst, d, tok, head, 0, value);
    }
}

static inline void qwen_fused_qkv_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 6);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
    const int64_t world_size     = params->txt_real_seq;
    const int64_t head_dim       = q->ne[0];
    const int64_t heads          = q->ne[1];
    const int64_t shard_sequence = q->ne[2];
    const int64_t shard_heads    = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(k->ne[0] == head_dim && v->ne[0] == head_dim);
    GGML_ASSERT(k->ne[1] == heads && v->ne[1] == heads);
    GGML_ASSERT(k->ne[2] == shard_sequence && v->ne[2] == shard_sequence);
    GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == total_head_dim * shard_heads * shard_sequence * world_size);

    const int64_t total = dst->ne[0];
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem         = linear;
        const int64_t d_all = rem % total_head_dim;
        rem /= total_head_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer = rem;
        const int64_t head = head_local + peer * shard_heads;
        const int64_t qkv_plane = d_all / head_dim;
        const int64_t d = d_all - qkv_plane * head_dim;
        const ggml_tensor* src = qkv_plane == 0 ? q : (qkv_plane == 1 ? k : v);
        const float value = qwen_fused_qkv_pack_get_f32(src, d, head, seq, 0);
        qwen_fused_qkv_pack_set_f32(dst, linear, 0, 0, 0, value);
    }
}

static inline void qwen_fused_joint_qkv_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 49);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* txt_q = dst->src[0];
    const ggml_tensor* txt_k = dst->src[1];
    const ggml_tensor* txt_v = dst->src[2];
    const ggml_tensor* img_q = dst->src[3];
    const ggml_tensor* img_k = dst->src[4];
    const ggml_tensor* img_v = dst->src[5];
    GGML_ASSERT(txt_q != nullptr && txt_k != nullptr && txt_v != nullptr);
    GGML_ASSERT(img_q != nullptr && img_k != nullptr && img_v != nullptr);

    const int64_t world_size = params->world_size;
    const int64_t head_dim   = txt_q->ne[0];
    const int64_t heads      = txt_q->ne[1];
    const int64_t txt_seq    = txt_q->ne[2];
    const int64_t img_seq    = img_q->ne[2];
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t txt_chunk = total_head_dim * shard_heads * txt_seq;
    const int64_t img_chunk = total_head_dim * shard_heads * img_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    GGML_ASSERT(world_size > 0 && heads % world_size == 0);
    GGML_ASSERT(txt_k->ne[0] == head_dim && txt_v->ne[0] == head_dim);
    GGML_ASSERT(img_k->ne[0] == head_dim && img_v->ne[0] == head_dim);
    GGML_ASSERT(txt_k->ne[1] == heads && txt_v->ne[1] == heads);
    GGML_ASSERT(img_q->ne[1] == heads && img_k->ne[1] == heads && img_v->ne[1] == heads);
    GGML_ASSERT(txt_k->ne[2] == txt_seq && txt_v->ne[2] == txt_seq);
    GGML_ASSERT(img_k->ne[2] == img_seq && img_v->ne[2] == img_seq);
    GGML_ASSERT(txt_q->ne[3] == 1 && txt_k->ne[3] == 1 && txt_v->ne[3] == 1);
    GGML_ASSERT(img_q->ne[3] == 1 && img_k->ne[3] == 1 && img_v->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
        const int64_t peer = linear / count_per_peer;
        int64_t rem        = linear - peer * count_per_peer;
        const bool is_img  = rem >= txt_chunk;
        if (is_img) {
            rem -= txt_chunk;
        }

        const int64_t d_all = rem % total_head_dim;
        rem /= total_head_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem;
        const int64_t head = head_local + peer * shard_heads;
        const int64_t qkv_plane = d_all / head_dim;
        const int64_t d = d_all - qkv_plane * head_dim;

        const ggml_tensor* src = nullptr;
        if (qkv_plane == 0) {
            src = is_img ? img_q : txt_q;
        } else if (qkv_plane == 1) {
            src = is_img ? img_k : txt_k;
        } else {
            src = is_img ? img_v : txt_v;
        }
        const float value = qwen_fused_qkv_pack_get_f32(src, d, head, seq, 0);
        qwen_fused_qkv_pack_set_f32(dst, linear, 0, 0, 0, value);
    }
}

static inline void qwen_fused_attn_head_to_seq_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 7 || params->mode == 13);
    GGML_ASSERT((params->mode == 7 && dst->type == GGML_TYPE_F32) ||
                (params->mode == 13 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* attn = dst->src[0];
    GGML_ASSERT(attn != nullptr);
    GGML_ASSERT(attn->type == GGML_TYPE_F32 || attn->type == GGML_TYPE_F16);
    const int64_t world_size = params->world_size;
    const int64_t txt_real_seq = params->txt_real_seq;
    const int64_t img_real_seq = params->img_real_seq;
    const int64_t txt_padded_seq = params->txt_padded_seq > 0 ? params->txt_padded_seq : txt_real_seq;
    const int64_t img_padded_seq = params->img_padded_seq > 0 ? params->img_padded_seq : img_real_seq;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq);
    GGML_ASSERT(img_padded_seq >= img_real_seq);
    GGML_ASSERT(txt_padded_seq % world_size == 0);
    GGML_ASSERT(img_padded_seq % world_size == 0);

    const int64_t head_dim = attn->ne[0];
    const int64_t shard_heads = attn->ne[1];
    const int64_t total_real_seq = txt_real_seq + img_real_seq;
    GGML_ASSERT(attn->ne[2] == total_real_seq);
    GGML_ASSERT(attn->ne[3] == 1);

    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    GGML_ASSERT(dst->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
        int64_t rem = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const bool is_img = rem >= txt_chunk;
        if (is_img) {
            rem -= txt_chunk;
        }
        const int64_t shard_seq = is_img ? img_shard_seq : txt_shard_seq;
        const int64_t stream_real_seq = is_img ? img_real_seq : txt_real_seq;
        int64_t d = rem % head_dim;
        rem /= head_dim;
        int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_tok = rem;
        GGML_ASSERT(local_tok < shard_seq);
        const int64_t stream_tok = peer * shard_seq + local_tok;
        float value = 0.0f;
        if (stream_tok < stream_real_seq) {
            const int64_t total_tok = is_img ? txt_real_seq + stream_tok : stream_tok;
            value = qwen_fused_pack_get_f32_or_f16(attn, d, head, total_tok, 0);
        }
        qwen_fused_pack_set_f32_or_f16(dst, linear, 0, 0, 0, value);
    }
}

static inline void qwen_fused_attn_head_to_seq_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 8 || params->mode == 14 || params->mode == 45 || params->mode == 48);
    GGML_ASSERT(((params->mode == 8 || params->mode == 14) && dst->type == GGML_TYPE_F32) ||
                (params->mode == 45 && dst->type == GGML_TYPE_F16) ||
                (params->mode == 48 && dst->type == GGML_TYPE_BF16));
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT((params->mode == 8 && recv_flat->type == GGML_TYPE_F32) ||
                ((params->mode == 14 || params->mode == 45 || params->mode == 48) && recv_flat->type == GGML_TYPE_F16));
    const int64_t world_size = params->world_size;
    const int64_t stream_index = params->stream_index;
    const int64_t txt_padded_seq = params->txt_padded_seq;
    const int64_t img_padded_seq = params->img_padded_seq;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(txt_padded_seq > 0 && img_padded_seq > 0);
    GGML_ASSERT(txt_padded_seq % world_size == 0);
    GGML_ASSERT(img_padded_seq % world_size == 0);

    const int64_t head_dim = dst->ne[0];
    const int64_t heads = dst->ne[1];
    const int64_t shard_heads = heads / world_size;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t out_shard_seq = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_offset = stream_index == 0 ? 0 : txt_chunk;
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(dst->ne[2] == out_shard_seq);
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_tok = rem;
        const int64_t src_peer = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx =
            src_peer * count_per_peer +
            stream_offset +
            d +
            local_head * head_dim +
            local_tok * head_dim * shard_heads;
        const char* data = static_cast<const char*>(recv_flat->data);
        char* dst_data = static_cast<char*>(dst->data);
        char* dst_ptr = dst_data +
                        d * dst->nb[0] +
                        head * dst->nb[1] +
                        local_tok * dst->nb[2];
        if (recv_flat->type == GGML_TYPE_F32) {
            GGML_ASSERT(dst->type == GGML_TYPE_F32);
            *reinterpret_cast<float*>(dst_ptr) =
                *reinterpret_cast<const float*>(data + src_idx * recv_flat->nb[0]);
            continue;
        }

        const ggml_fp16_t value_f16 =
            *reinterpret_cast<const ggml_fp16_t*>(data + src_idx * recv_flat->nb[0]);
        if (dst->type == GGML_TYPE_F16) {
            *reinterpret_cast<ggml_fp16_t*>(dst_ptr) = value_f16;
        } else if (dst->type == GGML_TYPE_BF16) {
            *reinterpret_cast<ggml_bf16_t*>(dst_ptr) =
                ggml_fp32_to_bf16(ggml_fp16_to_fp32(value_f16));
        } else {
            *reinterpret_cast<float*>(dst_ptr) = ggml_fp16_to_fp32(value_f16);
        }
    }
}

static inline float qwen_fused_qkv_recv_get_f32(const ggml_tensor* recv_flat,
                                                int64_t d,
                                                int64_t seq,
                                                int64_t head,
                                                int64_t head_dim,
                                                int64_t n_heads) {
    const int64_t idx = d + head * (3 * head_dim) + seq * (3 * head_dim) * n_heads;
    const char* data = static_cast<const char*>(recv_flat->data);
    if (recv_flat->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data + idx * recv_flat->nb[0]));
    }
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
    return *reinterpret_cast<const float*>(data + idx * recv_flat->nb[0]);
}

static inline void qwen_fused_qkv_pack_from_recv_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->txt_real_seq > 0);
    GGML_ASSERT(params->img_real_seq > 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* txt_recv = dst->src[0];
    const ggml_tensor* img_recv = dst->src[1];
    GGML_ASSERT(txt_recv != nullptr && img_recv != nullptr);

    const int64_t qk_half_dim = dst->ne[0];
    const int64_t head_dim    = qk_half_dim * 2;
    const int64_t total_seq   = dst->ne[1];
    const int64_t n_heads     = dst->ne[2];
    const int64_t plane_elems = qk_half_dim * total_seq * n_heads;
    const int64_t txt_padded_seq = txt_recv->ne[0] / (3 * head_dim * n_heads);
    const int64_t img_padded_seq = img_recv->ne[0] / (3 * head_dim * n_heads);
    GGML_ASSERT(txt_recv->ne[0] == 3 * head_dim * n_heads * txt_padded_seq);
    GGML_ASSERT(img_recv->ne[0] == 3 * head_dim * n_heads * img_padded_seq);
    GGML_ASSERT(params->txt_real_seq <= txt_padded_seq);
    GGML_ASSERT(params->img_real_seq <= img_padded_seq);
    GGML_ASSERT(total_seq == params->txt_real_seq + params->img_real_seq);

    for (int64_t linear = ith; linear < 6 * plane_elems; linear += nth) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % total_seq;
        const int64_t head  = rem / total_seq;
        const bool is_txt   = tok < params->txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - params->txt_real_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part      = plane % 2;
        const ggml_tensor* src  = is_txt ? txt_recv : img_recv;

        int64_t src_d = qkv_plane * head_dim;
        if (qkv_plane < 2) {
            src_d += part + 2 * half;
        } else {
            src_d += half + part * qk_half_dim;
        }
        const float value = qwen_fused_qkv_recv_get_f32(src, src_d, src_t, head, head_dim, n_heads);
        qwen_fused_qkv_pack_set_f32(dst, half, tok, head, plane, value);
    }
}

static inline void qwen_fused_qk_pack_from_recv_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->txt_real_seq > 0 && params->img_real_seq > 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const ggml_tensor* txt_recv = dst->src[0];
    const ggml_tensor* img_recv = dst->src[1];
    const int64_t qk_half_dim = dst->ne[0];
    const int64_t head_dim    = qk_half_dim * 2;
    const int64_t total_seq   = dst->ne[1];
    const int64_t n_heads     = dst->ne[2];
    const int64_t plane_elems = qk_half_dim * total_seq * n_heads;
    GGML_ASSERT(dst->ne[3] == 4);
    GGML_ASSERT(total_seq == params->txt_real_seq + params->img_real_seq);

    for (int64_t linear = ith; linear < 4 * plane_elems; linear += nth) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % total_seq;
        const int64_t head  = rem / total_seq;
        const bool is_txt   = tok < params->txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - params->txt_real_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part      = plane % 2;
        const ggml_tensor* src  = is_txt ? txt_recv : img_recv;
        const int64_t src_d     = qkv_plane * head_dim + part + 2 * half;
        const float value = qwen_fused_qkv_recv_get_f32(src, src_d, src_t, head, head_dim, n_heads);
        qwen_fused_qkv_pack_set_f32(dst, half, tok, head, plane, value);
    }
}

static inline void qwen_fused_v_pack_from_recv_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->txt_real_seq > 0 && params->img_real_seq > 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const ggml_tensor* txt_recv = dst->src[0];
    const ggml_tensor* img_recv = dst->src[1];
    const int64_t head_dim  = dst->ne[0];
    const int64_t total_seq = dst->ne[1];
    const int64_t n_heads   = dst->ne[2];
    const int64_t total     = head_dim * total_seq * n_heads;
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(total_seq == params->txt_real_seq + params->img_real_seq);

    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t tok  = rem % total_seq;
        const int64_t head = rem / total_seq;
        const bool is_txt  = tok < params->txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - params->txt_real_seq;
        const ggml_tensor* src = is_txt ? txt_recv : img_recv;
        const int64_t src_d = 2 * head_dim + d;
        const float value = qwen_fused_qkv_recv_get_f32(src, src_d, src_t, head, head_dim, n_heads);
        qwen_fused_qkv_pack_set_f32(dst, d, tok, head, 0, value);
    }
}

static inline float qwen_fused_joint_qkv_recv_get_f32(const ggml_tensor* recv_flat,
                                                      int64_t d_all,
                                                      int64_t seq,
                                                      int64_t head,
                                                      int64_t head_dim,
                                                      int64_t n_heads,
                                                      int64_t stream_index,
                                                      int64_t txt_padded_seq,
                                                      int64_t img_padded_seq,
                                                      int64_t world_size) {
    const int64_t shard_heads = n_heads;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t txt_chunk = total_head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = total_head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_shard_seq = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t stream_offset = stream_index == 0 ? 0 : txt_chunk;
    const int64_t peer = seq / stream_shard_seq;
    const int64_t local_seq = seq - peer * stream_shard_seq;
    const int64_t idx = peer * count_per_peer +
                        stream_offset +
                        d_all +
                        head * total_head_dim +
                        local_seq * total_head_dim * shard_heads;
    const char* data = static_cast<const char*>(recv_flat->data);
    return *reinterpret_cast<const float*>(data + idx * recv_flat->nb[0]);
}

static inline void qwen_fused_qk_pack_from_joint_recv_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 50);
    GGML_ASSERT(params->txt_real_seq > 0 && params->img_real_seq > 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr &&
                (recv_flat->type == GGML_TYPE_F32 || recv_flat->type == GGML_TYPE_F16));

    const int64_t qk_half_dim = dst->ne[0];
    const int64_t head_dim = qk_half_dim * 2;
    const int64_t total_seq = dst->ne[1];
    const int64_t n_heads = dst->ne[2];
    const int64_t plane_elems = qk_half_dim * total_seq * n_heads;
    GGML_ASSERT(dst->ne[3] == 4);
    GGML_ASSERT(total_seq == params->txt_real_seq + params->img_real_seq);
    GGML_ASSERT(params->txt_padded_seq >= params->txt_real_seq);
    GGML_ASSERT(params->img_padded_seq >= params->img_real_seq);
    GGML_ASSERT(params->world_size > 0);
    GGML_ASSERT(params->txt_padded_seq % params->world_size == 0);
    GGML_ASSERT(params->img_padded_seq % params->world_size == 0);

    const int64_t total_head_dim = head_dim * 3;
    const int64_t txt_chunk = total_head_dim * n_heads * (params->txt_padded_seq / params->world_size);
    const int64_t img_chunk = total_head_dim * n_heads * (params->img_padded_seq / params->world_size);
    GGML_ASSERT(recv_flat->ne[0] == (txt_chunk + img_chunk) * params->world_size);

    for (int64_t linear = ith; linear < 4 * plane_elems; linear += nth) {
        const int64_t plane = linear / plane_elems;
        int64_t rem = linear - plane * plane_elems;
        const int64_t half = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok = rem % total_seq;
        const int64_t head = rem / total_seq;
        const bool is_txt = tok < params->txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - params->txt_real_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part = plane % 2;
        const int64_t src_d = qkv_plane * head_dim + part + 2 * half;
        const float value = qwen_fused_joint_qkv_recv_get_f32(recv_flat,
                                                              src_d,
                                                              src_t,
                                                              head,
                                                              head_dim,
                                                              n_heads,
                                                              is_txt ? 0 : 1,
                                                              params->txt_padded_seq,
                                                              params->img_padded_seq,
                                                              params->world_size);
        qwen_fused_qkv_pack_set_f32(dst, half, tok, head, plane, value);
    }
}

static inline void qwen_fused_v_pack_from_joint_recv_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const QwenFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == QWEN_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 51);
    GGML_ASSERT(params->txt_real_seq > 0 && params->img_real_seq > 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr &&
                (recv_flat->type == GGML_TYPE_F32 || recv_flat->type == GGML_TYPE_F16));

    const int64_t head_dim = dst->ne[0];
    const int64_t total_seq = dst->ne[1];
    const int64_t n_heads = dst->ne[2];
    const int64_t total = head_dim * total_seq * n_heads;
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(total_seq == params->txt_real_seq + params->img_real_seq);
    GGML_ASSERT(params->txt_padded_seq >= params->txt_real_seq);
    GGML_ASSERT(params->img_padded_seq >= params->img_real_seq);
    GGML_ASSERT(params->world_size > 0);
    GGML_ASSERT(params->txt_padded_seq % params->world_size == 0);
    GGML_ASSERT(params->img_padded_seq % params->world_size == 0);

    const int64_t total_head_dim = head_dim * 3;
    const int64_t txt_chunk = total_head_dim * n_heads * (params->txt_padded_seq / params->world_size);
    const int64_t img_chunk = total_head_dim * n_heads * (params->img_padded_seq / params->world_size);
    GGML_ASSERT(recv_flat->ne[0] == (txt_chunk + img_chunk) * params->world_size);

    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t tok = rem % total_seq;
        const int64_t head = rem / total_seq;
        const bool is_txt = tok < params->txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - params->txt_real_seq;
        const int64_t src_d = 2 * head_dim + d;
        const float value = qwen_fused_joint_qkv_recv_get_f32(recv_flat,
                                                              src_d,
                                                              src_t,
                                                              head,
                                                              head_dim,
                                                              n_heads,
                                                              is_txt ? 0 : 1,
                                                              params->txt_padded_seq,
                                                              params->img_padded_seq,
                                                              params->world_size);
        qwen_fused_qkv_pack_set_f32(dst, d, tok, head, 0, value);
    }
}

static inline ggml_tensor* qwen_fused_qkv_pack(ggml_context* ctx,
                                 ggml_tensor* txt_q,
                                 ggml_tensor* img_q,
                                 ggml_tensor* txt_k,
                                 ggml_tensor* img_k,
                                 ggml_tensor* txt_v,
                                 ggml_tensor* img_v,
                                 int64_t txt_seq,
                                 int64_t img_seq,
                                 int64_t head_dim,
                                 int64_t n_heads) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_q != nullptr && img_q != nullptr && txt_k != nullptr && img_k != nullptr && txt_v != nullptr && img_v != nullptr);
    GGML_ASSERT(txt_seq > 0 && img_seq > 0);
    GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(txt_q->type == GGML_TYPE_F32 && img_q->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_k->type == GGML_TYPE_F32 && img_k->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_v->type == GGML_TYPE_F32 && img_v->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_q->ne[0] * 2 == head_dim && img_q->ne[0] * 2 == head_dim);
    GGML_ASSERT(txt_k->ne[0] * 2 == head_dim && img_k->ne[0] * 2 == head_dim);
    GGML_ASSERT(txt_v->ne[0] == head_dim && img_v->ne[0] == head_dim);
    GGML_ASSERT(txt_q->ne[1] == txt_seq && txt_k->ne[1] == txt_seq && txt_v->ne[1] == txt_seq);
    GGML_ASSERT(img_q->ne[1] == img_seq && img_k->ne[1] == img_seq && img_v->ne[1] == img_seq);
    GGML_ASSERT(txt_q->ne[2] == n_heads && txt_k->ne[2] == n_heads && txt_v->ne[2] == n_heads);
    GGML_ASSERT(img_q->ne[2] == n_heads && img_k->ne[2] == n_heads && img_v->ne[2] == n_heads);
    GGML_ASSERT(txt_q->ne[3] == 2 && img_q->ne[3] == 2 && txt_k->ne[3] == 2 && img_k->ne[3] == 2);
    GGML_ASSERT(txt_v->ne[3] == 1 && img_v->ne[3] == 1);

    ggml_tensor* args[] = {txt_q, img_q, txt_k, img_k, txt_v, img_v};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim / 2,
                                      txt_seq + img_seq,
                                      n_heads,
                                      6,
                                      args,
                                      6,
                                      qwen_fused_qkv_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      const_cast<QwenFusedQKVPackParams*>(&QWEN_FUSED_QKV_PACK_USERDATA));
    ggml_set_name(out, "qwen.fused_qkv_pack.out");
    return out;
}

static inline ggml_tensor* qwen_fused_qk_pack(ggml_context* ctx,
                                ggml_tensor* txt_q,
                                ggml_tensor* img_q,
                                ggml_tensor* txt_k,
                                ggml_tensor* img_k,
                                int64_t txt_seq,
                                int64_t img_seq,
                                int64_t head_dim,
                                int64_t n_heads) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_q != nullptr && img_q != nullptr && txt_k != nullptr && img_k != nullptr);
    GGML_ASSERT(txt_seq > 0 && img_seq > 0);
    GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(txt_q->type == GGML_TYPE_F32 && img_q->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_k->type == GGML_TYPE_F32 && img_k->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_q->ne[0] * 2 == head_dim && img_q->ne[0] * 2 == head_dim);
    GGML_ASSERT(txt_k->ne[0] * 2 == head_dim && img_k->ne[0] * 2 == head_dim);
    GGML_ASSERT(txt_q->ne[1] == txt_seq && txt_k->ne[1] == txt_seq);
    GGML_ASSERT(img_q->ne[1] == img_seq && img_k->ne[1] == img_seq);
    GGML_ASSERT(txt_q->ne[2] == n_heads && txt_k->ne[2] == n_heads);
    GGML_ASSERT(img_q->ne[2] == n_heads && img_k->ne[2] == n_heads);
    GGML_ASSERT(txt_q->ne[3] == 2 && img_q->ne[3] == 2 && txt_k->ne[3] == 2 && img_k->ne[3] == 2);

    ggml_tensor* args[] = {txt_q, img_q, txt_k, img_k};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim / 2,
                                      txt_seq + img_seq,
                                      n_heads,
                                      4,
                                      args,
                                      4,
                                      qwen_fused_qk_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_seq, img_seq, 1));
    ggml_set_name(out, "qwen.fused_qk_pack.out");
    return out;
}

static inline ggml_tensor* qwen_fused_v_pack(ggml_context* ctx,
                               ggml_tensor* txt_v,
                               ggml_tensor* img_v,
                               int64_t txt_seq,
                               int64_t img_seq,
                               int64_t head_dim,
                               int64_t n_heads) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_v != nullptr && img_v != nullptr);
    GGML_ASSERT(txt_seq > 0 && img_seq > 0);
    GGML_ASSERT(head_dim > 0);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(txt_v->type == GGML_TYPE_F32 && img_v->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_v->ne[0] == head_dim && img_v->ne[0] == head_dim);
    GGML_ASSERT(txt_v->ne[1] == txt_seq && img_v->ne[1] == img_seq);
    GGML_ASSERT(txt_v->ne[2] == n_heads && img_v->ne[2] == n_heads);
    GGML_ASSERT(txt_v->ne[3] == 1 && img_v->ne[3] == 1);

    ggml_tensor* args[] = {txt_v, img_v};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      txt_seq + img_seq,
                                      n_heads,
                                      1,
                                      args,
                                      2,
                                      qwen_fused_v_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_seq, img_seq, 2));
    ggml_set_name(out, "qwen.fused_v_pack.out");
    return out;
}

static inline ggml_tensor* qwen_fused_qkv_send_pack(ggml_context* ctx,
                                      ggml_tensor* q,
                                      ggml_tensor* k,
                                      ggml_tensor* v,
                                      int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
    GGML_ASSERT(q->ne[0] > 0 && q->ne[1] > 0 && q->ne[2] > 0);
    GGML_ASSERT(q->ne[1] % world_size == 0);
    GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
    GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
    GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
    GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);

    const int64_t total_head_dim = q->ne[0] * 3;
    const int64_t shard_heads    = q->ne[1] / world_size;
    const int64_t flat_elems     = total_head_dim * shard_heads * q->ne[2] * world_size;
    ggml_tensor* args[] = {q, k, v};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      flat_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      3,
                                      qwen_fused_qkv_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(world_size, 0, 6));
    ggml_set_name(out, "qwen.fused_qkv_send_pack.out");
    return out;
}

static inline ggml_tensor* qwen_fused_joint_qkv_send_pack(ggml_context* ctx,
                                                          ggml_tensor* txt_q,
                                                          ggml_tensor* txt_k,
                                                          ggml_tensor* txt_v,
                                                          ggml_tensor* img_q,
                                                          ggml_tensor* img_k,
                                                          ggml_tensor* img_v,
                                                          int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_q != nullptr && txt_k != nullptr && txt_v != nullptr);
    GGML_ASSERT(img_q != nullptr && img_k != nullptr && img_v != nullptr);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(txt_q->type == GGML_TYPE_F32 && txt_k->type == GGML_TYPE_F32 && txt_v->type == GGML_TYPE_F32);
    GGML_ASSERT(img_q->type == GGML_TYPE_F32 && img_k->type == GGML_TYPE_F32 && img_v->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_q->ne[0] > 0 && txt_q->ne[1] > 0 && txt_q->ne[2] > 0);
    GGML_ASSERT(img_q->ne[0] == txt_q->ne[0] && img_q->ne[1] == txt_q->ne[1] && img_q->ne[2] > 0);
    GGML_ASSERT(txt_q->ne[1] % world_size == 0);
    GGML_ASSERT(txt_k->ne[0] == txt_q->ne[0] && txt_v->ne[0] == txt_q->ne[0]);
    GGML_ASSERT(img_k->ne[0] == txt_q->ne[0] && img_v->ne[0] == txt_q->ne[0]);
    GGML_ASSERT(txt_k->ne[1] == txt_q->ne[1] && txt_v->ne[1] == txt_q->ne[1]);
    GGML_ASSERT(img_k->ne[1] == txt_q->ne[1] && img_v->ne[1] == txt_q->ne[1]);
    GGML_ASSERT(txt_k->ne[2] == txt_q->ne[2] && txt_v->ne[2] == txt_q->ne[2]);
    GGML_ASSERT(img_k->ne[2] == img_q->ne[2] && img_v->ne[2] == img_q->ne[2]);
    GGML_ASSERT(txt_q->ne[3] == 1 && txt_k->ne[3] == 1 && txt_v->ne[3] == 1);
    GGML_ASSERT(img_q->ne[3] == 1 && img_k->ne[3] == 1 && img_v->ne[3] == 1);

    const int64_t total_head_dim = txt_q->ne[0] * 3;
    const int64_t shard_heads = txt_q->ne[1] / world_size;
    const int64_t count_per_peer = total_head_dim * shard_heads * (txt_q->ne[2] + img_q->ne[2]);
    ggml_tensor* args[] = {txt_q, txt_k, txt_v, img_q, img_k, img_v};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      count_per_peer * world_size,
                                      1,
                                      1,
                                      1,
                                      args,
                                      6,
                                      qwen_fused_joint_qkv_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_q->ne[2],
                                                                      img_q->ne[2],
                                                                      49,
                                                                      0,
                                                                      0,
                                                                      world_size));
    ggml_set_name(out, "qwen.fused_joint_qkv_send_pack.out");
    return out;
}

static inline ggml_tensor* qwen_fused_qkv_pack_from_recv(ggml_context* ctx,
                                           ggml_tensor* txt_recv_flat,
                                           ggml_tensor* img_recv_flat,
                                           int64_t txt_real_seq,
                                           int64_t img_real_seq,
                                           int64_t head_dim,
                                           int64_t n_heads) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_recv_flat != nullptr && img_recv_flat != nullptr);
    GGML_ASSERT(txt_recv_flat->type == GGML_TYPE_F32 && img_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(txt_recv_flat->ne[0] % (3 * head_dim * n_heads) == 0);
    GGML_ASSERT(img_recv_flat->ne[0] % (3 * head_dim * n_heads) == 0);
    GGML_ASSERT(txt_real_seq <= txt_recv_flat->ne[0] / (3 * head_dim * n_heads));
    GGML_ASSERT(img_real_seq <= img_recv_flat->ne[0] / (3 * head_dim * n_heads));

    ggml_tensor* args[] = {txt_recv_flat, img_recv_flat};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim / 2,
                                      txt_real_seq + img_real_seq,
                                      n_heads,
                                      6,
                                      args,
                                      2,
                                      qwen_fused_qkv_pack_from_recv_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_real_seq, img_real_seq, 3));
    ggml_set_name(out, "qwen.fused_qkv_pack.from_recv.out");
    return out;
}

static inline ggml_tensor* qwen_fused_qk_pack_from_recv(ggml_context* ctx,
                                          ggml_tensor* txt_recv_flat,
                                          ggml_tensor* img_recv_flat,
                                          int64_t txt_real_seq,
                                          int64_t img_real_seq,
                                          int64_t head_dim,
                                          int64_t n_heads) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_recv_flat != nullptr && img_recv_flat != nullptr);
    GGML_ASSERT(txt_recv_flat->type == GGML_TYPE_F32 && img_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(txt_recv_flat->ne[0] % (3 * head_dim * n_heads) == 0);
    GGML_ASSERT(img_recv_flat->ne[0] % (3 * head_dim * n_heads) == 0);

    ggml_tensor* args[] = {txt_recv_flat, img_recv_flat};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim / 2,
                                      txt_real_seq + img_real_seq,
                                      n_heads,
                                      4,
                                      args,
                                      2,
                                      qwen_fused_qk_pack_from_recv_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_real_seq, img_real_seq, 4));
    ggml_set_name(out, "qwen.fused_qk_pack.from_recv.out");
    return out;
}

static inline ggml_tensor* qwen_fused_v_pack_from_recv(ggml_context* ctx,
                                         ggml_tensor* txt_recv_flat,
                                         ggml_tensor* img_recv_flat,
                                         int64_t txt_real_seq,
                                         int64_t img_real_seq,
                                         int64_t head_dim,
                                         int64_t n_heads) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(txt_recv_flat != nullptr && img_recv_flat != nullptr);
    GGML_ASSERT(txt_recv_flat->type == GGML_TYPE_F32 && img_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(head_dim > 0);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(txt_recv_flat->ne[0] % (3 * head_dim * n_heads) == 0);
    GGML_ASSERT(img_recv_flat->ne[0] % (3 * head_dim * n_heads) == 0);

    ggml_tensor* args[] = {txt_recv_flat, img_recv_flat};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      txt_real_seq + img_real_seq,
                                      n_heads,
                                      1,
                                      args,
                                      2,
                                      qwen_fused_v_pack_from_recv_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_real_seq, img_real_seq, 5));
    ggml_set_name(out, "qwen.fused_v_pack.from_recv.out");
    return out;
}

static inline ggml_tensor* qwen_fused_qk_pack_from_joint_recv(ggml_context* ctx,
                                                              ggml_tensor* recv_flat,
                                                              int64_t txt_padded_seq,
                                                              int64_t img_padded_seq,
                                                              int64_t txt_real_seq,
                                                              int64_t img_real_seq,
                                                              int64_t head_dim,
                                                              int64_t n_heads,
                                                              int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 || recv_flat->type == GGML_TYPE_F16);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq && img_padded_seq >= img_real_seq);
    GGML_ASSERT(world_size > 0 && txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);
    GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
    GGML_ASSERT(n_heads > 0);
    const int64_t expected = 3 * head_dim * n_heads * (txt_padded_seq + img_padded_seq);
    GGML_ASSERT(recv_flat->ne[0] == expected);

    ggml_tensor* args[] = {recv_flat};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim / 2,
                                      txt_real_seq + img_real_seq,
                                      n_heads,
                                      4,
                                      args,
                                      1,
                                      qwen_fused_qk_pack_from_joint_recv_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_real_seq,
                                                                      img_real_seq,
                                                                      50,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size));
    ggml_set_name(out, "qwen.fused_qk_pack.from_joint_recv.out");
    return out;
}

static inline ggml_tensor* qwen_fused_v_pack_from_joint_recv(ggml_context* ctx,
                                                             ggml_tensor* recv_flat,
                                                             int64_t txt_padded_seq,
                                                             int64_t img_padded_seq,
                                                             int64_t txt_real_seq,
                                                             int64_t img_real_seq,
                                                             int64_t head_dim,
                                                             int64_t n_heads,
                                                             int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 || recv_flat->type == GGML_TYPE_F16);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq && img_padded_seq >= img_real_seq);
    GGML_ASSERT(world_size > 0 && txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);
    GGML_ASSERT(head_dim > 0);
    GGML_ASSERT(n_heads > 0);
    const int64_t expected = 3 * head_dim * n_heads * (txt_padded_seq + img_padded_seq);
    GGML_ASSERT(recv_flat->ne[0] == expected);

    ggml_tensor* args[] = {recv_flat};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      txt_real_seq + img_real_seq,
                                      n_heads,
                                      1,
                                      args,
                                      1,
                                      qwen_fused_v_pack_from_joint_recv_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_real_seq,
                                                                      img_real_seq,
                                                                      51,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size));
    ggml_set_name(out, "qwen.fused_v_pack.from_joint_recv.out");
    return out;
}

static inline ggml_tensor* qwen_fused_scale_to_f16(ggml_context* ctx,
                                                   ggml_tensor* src,
                                                   float scale,
                                                   const char* name) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    if (!ggml_is_contiguous(src)) {
        src = ggml_cont(ctx, src);
    }

    ggml_tensor* args[] = {src};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      src->ne[0],
                                      src->ne[1],
                                      src->ne[2],
                                      src->ne[3],
                                      args,
                                      1,
                                      qwen_fused_scale_to_f16_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_scale_params(scale));
    ggml_set_name(out, name);
    return out;
}

static inline ggml_tensor* qwen_fused_attn_head_to_seq_send_pack(ggml_context* ctx,
                                                   ggml_tensor* attn_4d,
                                                   int64_t txt_real_seq,
                                                   int64_t img_real_seq,
                                                   int64_t txt_padded_seq,
                                                   int64_t img_padded_seq,
                                                   int world_size,
                                                   bool output_f16 = false) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(attn_4d != nullptr);
    GGML_ASSERT(attn_4d->type == GGML_TYPE_F32 || attn_4d->type == GGML_TYPE_F16);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq);
    GGML_ASSERT(img_padded_seq >= img_real_seq);
    GGML_ASSERT(txt_padded_seq % world_size == 0);
    GGML_ASSERT(img_padded_seq % world_size == 0);
    GGML_ASSERT(attn_4d->ne[2] == txt_real_seq + img_real_seq);
    GGML_ASSERT(attn_4d->ne[3] == 1);

    const int64_t count_per_peer =
        attn_4d->ne[0] * attn_4d->ne[1] * (txt_padded_seq + img_padded_seq) / world_size;
    ggml_tensor* args[] = {attn_4d};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      output_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                      count_per_peer * world_size,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      qwen_fused_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(txt_real_seq,
                                                                      img_real_seq,
                                                                      output_f16 ? 13 : 7,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size));
    ggml_set_name(out, "qwen.fused_attn_head_to_seq_send_pack.out");
    return out;
}

static inline ggml_tensor* qwen_fused_attn_head_to_seq_recv_unpack(ggml_context* ctx,
                                                     ggml_tensor* recv_flat,
                                                     int64_t stream_index,
                                                     int64_t txt_padded_seq,
                                                     int64_t img_padded_seq,
                                                     int64_t head_dim,
                                                     int64_t heads,
                                                     int world_size,
                                                     ggml_type output_type = GGML_TYPE_F32) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 || recv_flat->type == GGML_TYPE_F16);
    GGML_ASSERT(output_type == GGML_TYPE_F32 || output_type == GGML_TYPE_F16 || output_type == GGML_TYPE_BF16);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(head_dim > 0 && heads > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(txt_padded_seq > 0 && img_padded_seq > 0);
    GGML_ASSERT(txt_padded_seq % world_size == 0);
    GGML_ASSERT(img_padded_seq % world_size == 0);
    const int64_t out_seq = (stream_index == 0 ? txt_padded_seq : img_padded_seq) / world_size;
    const int64_t count_per_peer = head_dim * (heads / world_size) *
                                   ((txt_padded_seq + img_padded_seq) / world_size);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F16 || output_type == GGML_TYPE_F32);

    const int64_t mode = recv_flat->type == GGML_TYPE_F32 ? 8 :
                         output_type == GGML_TYPE_F16 ? 45 :
                         output_type == GGML_TYPE_BF16 ? 48 :
                                                          14;

    ggml_tensor* args[] = {recv_flat};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      output_type,
                                      head_dim,
                                      heads,
                                      out_seq,
                                      1,
                                      args,
                                      1,
                                      qwen_fused_attn_head_to_seq_recv_unpack_cpu,
                                      GGML_N_TASKS_MAX,
                                      qwen_fused_qkv_pack_make_params(0,
                                                                      0,
                                                                      mode,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size,
                                                                      stream_index));
    ggml_set_name(out, stream_index == 0 ?
                           "qwen.fused_attn_head_to_seq_recv_unpack.txt" :
                           "qwen.fused_attn_head_to_seq_recv_unpack.img");
    return out;
}


    constexpr int QWEN_IMAGE_GRAPH_SIZE = 20480;

    static inline bool qwen_sp_enabled(GGMLRunnerContext* ctx) {
        return ctx != nullptr &&
               ctx->process_group != nullptr &&
               ctx->process_group->enabled() &&
               ctx->process_group->size() > 1;
    }

    static inline int qwen_sp_rank(GGMLRunnerContext* ctx) {
        return ctx->process_group->rank();
    }

    static inline int qwen_sp_world_size(GGMLRunnerContext* ctx) {
        return ctx->process_group->size();
    }

    static inline ggml_tensor* qwen_sp_prepare_rope_pe_seq_major(ggml_context* ctx,
                                                                 ggml_tensor* pe) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(pe != nullptr);
        return ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));
    }

    static inline ggml_tensor* qwen_sp_apply_rope_seq_major(ggml_context* ctx,
                                                            ggml_tensor* x,
                                                            ggml_tensor* pe,
                                                            ggml_tensor* prepared_pe = nullptr,
                                                            ggml_backend_t backend = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);

        const int64_t d_head = x->ne[0];
        const int64_t L      = x->ne[1];
        const int64_t n_head = x->ne[2];
        const int64_t N      = x->ne[3];
        ggml_tensor* fused = edgedit::ggml_ext::apply_rope_seq_major(ctx,
                                                                     x,
                                                                     prepared_pe != nullptr ? prepared_pe : pe,
                                                                     true);
        if (fused != nullptr && (backend == nullptr || ggml_backend_supports_op(backend, fused))) {
            return fused;
        }

        x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));

        int64_t offset = x->nb[2] * x->ne[2];
        auto x_0       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);
        auto x_1       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);
        x_0            = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);
        x_1            = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);
        auto temp_x    = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
        x_0            = ggml_repeat(ctx, x_0, temp_x);
        x_1            = ggml_repeat(ctx, x_1, temp_x);

        pe        = prepared_pe != nullptr ? prepared_pe :
                    qwen_sp_prepare_rope_pe_seq_major(ctx, pe);
        offset    = pe->nb[2] * pe->ne[2];
        auto pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);
        auto pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);
    }

    static inline ggml_tensor* qwen_sp_apply_rope_seq_major_work_layout(ggml_context* ctx,
                                                                        ggml_tensor* x,
                                                                        ggml_tensor* pe,
                                                                        int64_t d_head,
                                                                        ggml_tensor* prepared_pe = nullptr,
                                                                        ggml_backend_t backend = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);
        GGML_ASSERT(x->ne[0] * 2 == d_head);
        GGML_ASSERT(x->ne[3] == 2);
        ggml_tensor* fused = edgedit::ggml_ext::apply_rope_work_layout(ctx,
                                                                      x,
                                                                      prepared_pe != nullptr ? prepared_pe : pe,
                                                                      d_head);
        if (fused != nullptr && (backend == nullptr || ggml_backend_supports_op(backend, fused))) {
            return fused;
        }

        const int64_t L      = x->ne[1];
        const int64_t n_head = x->ne[2];
        const int64_t offset = x->nb[2] * x->ne[2];
        auto x_0             = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);
        auto x_1             = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);
        x_0                  = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);
        x_1                  = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);
        auto temp_x          = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
        x_0                  = ggml_repeat(ctx, x_0, temp_x);
        x_1                  = ggml_repeat(ctx, x_1, temp_x);

        pe        = prepared_pe != nullptr ? prepared_pe :
                    qwen_sp_prepare_rope_pe_seq_major(ctx, pe);
        auto pe_offset = pe->nb[2] * pe->ne[2];
        auto pe_0      = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], pe_offset * 0);
        auto pe_1      = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], pe_offset * 1);

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head);
    }

    static inline ggml_tensor* qwen_sp_flash_attention_seq_major(GGMLRunnerContext* ctx,
                                                                 ggml_tensor* q,
                                                                 ggml_tensor* k,
                                                                 ggml_tensor* v,
                                                                 int64_t n_head,
                                                                 float kv_scale) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(ctx->backend != nullptr);

        const int64_t d_head    = q->ne[0];
        const int64_t L_q       = q->ne[1];
        const int64_t L_k       = k->ne[1];
        const int64_t N         = v->ne[3];
        const int64_t n_kv_head = v->ne[2];

        GGML_ASSERT(k->ne[0] == d_head);
        GGML_ASSERT(v->ne[0] == d_head);
        GGML_ASSERT(v->ne[1] == L_k);
        GGML_ASSERT(q->ne[2] == n_head * N);
        GGML_ASSERT(k->ne[2] == n_kv_head * N);

        const bool use_unpadded_flash_attn = qwen_sp_unpadded_flash_attn_enabled();
        int kv_pad = 0;
        if (!use_unpadded_flash_attn && L_k % 256 != 0) {
            kv_pad = GGML_PAD(L_k, 256) - static_cast<int>(L_k);
        }

        float scale = 1.0f / sqrtf(static_cast<float>(d_head));
        auto scale_to_f16 = [&](ggml_tensor* src, float factor, const char* name) -> ggml_tensor* {
            if (factor == 1.0f) {
                return ggml_cast(ctx->ggml_ctx, src, GGML_TYPE_F16);
            }
            ggml_tensor* fused = qwen_fused_scale_to_f16(ctx->ggml_ctx, src, factor, name);
            if (ggml_backend_supports_op(ctx->backend, fused)) {
                return fused;
            }
            src = ggml_ext_scale(ctx->ggml_ctx, src, factor);
            return ggml_cast(ctx->ggml_ctx, src, GGML_TYPE_F16);
        };

        ggml_tensor* k_in = k;
        if (kv_pad != 0) {
            k_in = ggml_pad(ctx->ggml_ctx, k_in, 0, kv_pad, 0, 0);
        }
        k_in = scale_to_f16(k_in, kv_scale, "qwen.sp.flash_k.scale_to_f16");

        ggml_tensor* v_in = ggml_reshape_3d(ctx->ggml_ctx, v, d_head, L_k, n_kv_head * N);
        if (kv_pad != 0) {
            v_in = ggml_pad(ctx->ggml_ctx, v_in, 0, kv_pad, 0, 0);
        }
        v_in = scale_to_f16(v_in, kv_scale, "qwen.sp.flash_v.scale_to_f16");

        ggml_tensor* mask_in = nullptr;
        if (kv_pad > 0) {
            mask_in         = ggml_ext_zeros(ctx->ggml_ctx, L_k, L_q, 1, 1);
            auto pad_tensor = ggml_ext_full(ctx->ggml_ctx, -INFINITY, kv_pad, L_q, 1, 1);
            mask_in         = ggml_concat(ctx->ggml_ctx, mask_in, pad_tensor, 0);
            mask_in         = ggml_cast(ctx->ggml_ctx, mask_in, GGML_TYPE_F16);
        }

        ggml_tensor* q_in = q;
        if (qwen_sp_f16_q_attention_enabled() && q_in->type != GGML_TYPE_F16) {
            q_in = ggml_cast(ctx->ggml_ctx, q_in, GGML_TYPE_F16);
        }

        ggml_tensor* out = ggml_flash_attn_ext_with_type(ctx->ggml_ctx,
                                                         q_in,
                                                         k_in,
                                                         v_in,
                                                         mask_in,
                                                         scale / kv_scale,
                                                         0,
                                                         0,
                                                         GGML_TYPE_F32);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (!ggml_backend_supports_op(ctx->backend, out)) {
            return nullptr;
        }
        if (kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx->ggml_ctx, out, 1.0f / kv_scale);
        }

        if (qwen_sp_cont_flash_attn_output_enabled()) {
            out = ggml_view_3d(ctx->ggml_ctx, out, d_head, n_head, L_q, out->nb[1], out->nb[2], 0);
            out = ggml_cont(ctx->ggml_ctx, out);
        }
        return ggml_reshape_3d(ctx->ggml_ctx, out, d_head * n_head, L_q, N);
    }

    static inline ggml_tensor* qwen_sp_attention(GGMLRunnerContext* ctx,
                                                 ggml_tensor* q,
                                                 ggml_tensor* k,
                                                 ggml_tensor* v,
                                                 ggml_tensor* pe,
                                                 float kv_scale) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        q = Rope::apply_rope(ctx->ggml_ctx, q, pe, true, ctx->backend);
        k = Rope::apply_rope(ctx->ggml_ctx, k, pe, true, ctx->backend);

        const int64_t n_head = v->ne[1];
        GGML_ASSERT(q->ne[0] == k->ne[0]);
        GGML_ASSERT(q->ne[1] == k->ne[1]);
        GGML_ASSERT(q->ne[2] == k->ne[2]);
        GGML_ASSERT(v->ne[0] == q->ne[0]);
        GGML_ASSERT(v->ne[1] == n_head);
        GGML_ASSERT(v->ne[2] == q->ne[1]);
        GGML_ASSERT(v->ne[3] == 1);

        ggml_tensor* attn = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                   ctx->backend,
                                                   q,
                                                   k,
                                                   v,
                                                   n_head,
                                                   nullptr,
                                                   true,
                                                   ctx->flash_attn_enabled,
                                                   kv_scale);
        return attn;
    }

    static inline ggml_tensor* qwen_sp_attention_seq_major(GGMLRunnerContext* ctx,
                                                           ggml_tensor* q,
                                                           ggml_tensor* k,
                                                           ggml_tensor* v,
                                                           ggml_tensor* pe,
                                                           float kv_scale,
                                                           ggml_tensor* prepared_pe_seq_major = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        const int64_t n_head = v->ne[2];
        ggml_tensor* prepared_pe = prepared_pe_seq_major != nullptr ?
                                       prepared_pe_seq_major :
                                       qwen_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx, pe);
        q = qwen_sp_apply_rope_seq_major(ctx->ggml_ctx, q, pe, prepared_pe, ctx->backend);
        k = qwen_sp_apply_rope_seq_major(ctx->ggml_ctx, k, pe, prepared_pe, ctx->backend);

        ggml_tensor* attn = nullptr;
        if (ctx->flash_attn_enabled) {
            attn = qwen_sp_flash_attention_seq_major(ctx, q, k, v, n_head, kv_scale);
        }
        if (attn != nullptr) {
            return attn;
        }

        ggml_tensor* v_head_major = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));
        return ggml_ext_attention_ext(ctx->ggml_ctx,
                                      ctx->backend,
                                      q,
                                      k,
                                      v_head_major,
                                      n_head,
                                      nullptr,
                                      true,
                                      false,
                                      kv_scale);
    }

    static inline ggml_tensor* qwen_sp_attention_from_rope_work_layout(GGMLRunnerContext* ctx,
                                                                       ggml_tensor* q,
                                                                       ggml_tensor* k,
                                                                       ggml_tensor* v,
                                                                       ggml_tensor* pe,
                                                                       int64_t d_head,
                                                                       float kv_scale,
                                                                       ggml_tensor* prepared_pe_seq_major = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        const int64_t n_head = v->ne[2];
        ggml_tensor* prepared_pe = prepared_pe_seq_major != nullptr ?
                                       prepared_pe_seq_major :
                                       qwen_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx, pe);
        q = qwen_sp_apply_rope_seq_major_work_layout(ctx->ggml_ctx, q, pe, d_head, prepared_pe, ctx->backend);
        k = qwen_sp_apply_rope_seq_major_work_layout(ctx->ggml_ctx, k, pe, d_head, prepared_pe, ctx->backend);

        ggml_tensor* attn = nullptr;
        if (ctx->flash_attn_enabled) {
            attn = qwen_sp_flash_attention_seq_major(ctx, q, k, v, n_head, kv_scale);
        }
        if (attn != nullptr) {
            return attn;
        }

        ggml_tensor* v_head_major = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));
        return ggml_ext_attention_ext(ctx->ggml_ctx,
                                      ctx->backend,
                                      q,
                                      k,
                                      v_head_major,
                                      n_head,
                                      nullptr,
                                      true,
                                      false,
                                      kv_scale);
    }

    static inline ggml_tensor* qwen_single_seq_major_view(ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          int64_t d_head) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(x->ne[0] == d_head);
        GGML_ASSERT(x->ne[3] == 1);
        return ggml_permute(ctx, x, 0, 2, 1, 3); // [d_head, seq, n_head, 1]
    }

    static inline ggml_tensor* qwen_single_attention_seq_major(GGMLRunnerContext* ctx,
                                                               ggml_tensor* q,
                                                               ggml_tensor* k,
                                                               ggml_tensor* v,
                                                               ggml_tensor* pe,
                                                               float kv_scale) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        const int64_t n_head = v->ne[2];
        q = qwen_sp_apply_rope_seq_major(ctx->ggml_ctx, q, pe, nullptr, ctx->backend);
        k = qwen_sp_apply_rope_seq_major(ctx->ggml_ctx, k, pe, nullptr, ctx->backend);

        if (ctx->flash_attn_enabled) {
            ggml_tensor* attn = qwen_sp_flash_attention_seq_major(ctx, q, k, v, n_head, kv_scale);
            if (attn != nullptr) {
                return attn;
            }
        }

        ggml_tensor* v_head_major = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));
        return ggml_ext_attention_ext(ctx->ggml_ctx,
                                      ctx->backend,
                                      q,
                                      k,
                                      v_head_major,
                                      n_head,
                                      nullptr,
                                      true,
                                      false,
                                      kv_scale);
    }

    static inline ggml_tensor* qwen_sp_view_head_sequence(ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          int64_t start,
                                                          int64_t length) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(start >= 0);
        GGML_ASSERT(length > 0);
        GGML_ASSERT(start + length <= x->ne[2]);

        return ggml_view_4d(ctx,
                            x,
                            x->ne[0],
                            x->ne[1],
                            length,
                            x->ne[3],
                            x->nb[1],
                            x->nb[2],
                            x->nb[3],
                            static_cast<size_t>(start) * x->nb[2]);
    }

    static inline ggml_tensor* qwen_sp_concat_real_attention_sequence(ggml_context* ctx,
                                                                     ggml_tensor* txt,
                                                                     ggml_tensor* img,
                                                                     int64_t txt_pad,
                                                                     int64_t img_pad) {
        GGML_ASSERT(txt != nullptr);
        GGML_ASSERT(img != nullptr);
        GGML_ASSERT(txt_pad >= 0 && txt_pad <= txt->ne[2]);
        GGML_ASSERT(img_pad >= 0 && img_pad <= img->ne[2]);

        const int64_t txt_real_seq = txt->ne[2] - txt_pad;
        const int64_t img_real_seq = img->ne[2] - img_pad;
        GGML_ASSERT(txt_real_seq > 0);
        GGML_ASSERT(img_real_seq > 0);

        ggml_tensor* txt_real = qwen_sp_view_head_sequence(ctx,
                                                           txt,
                                                           0,
                                                           txt_real_seq);
        ggml_tensor* img_real = qwen_sp_view_head_sequence(ctx,
                                                           img,
                                                           0,
                                                           img_real_seq);
        return ggml_concat(ctx, txt_real, img_real, 2);
    }

    static inline ggml_tensor* qwen_sp_view_seq_major_sequence(ggml_context* ctx,
                                                              ggml_tensor* x,
                                                              int64_t start,
                                                              int64_t length) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(start >= 0);
        GGML_ASSERT(length > 0);
        GGML_ASSERT(start + length <= x->ne[1]);

        return ggml_view_4d(ctx,
                            x,
                            x->ne[0],
                            length,
                            x->ne[2],
                            x->ne[3],
                            x->nb[1],
                            x->nb[2],
                            x->nb[3],
                            static_cast<size_t>(start) * x->nb[1]);
    }

    static inline ggml_tensor* qwen_sp_concat_real_attention_sequence_seq_major(ggml_context* ctx,
                                                                               ggml_tensor* txt,
                                                                               ggml_tensor* img,
                                                                               int64_t txt_pad,
                                                                               int64_t img_pad) {
        GGML_ASSERT(txt != nullptr);
        GGML_ASSERT(img != nullptr);
        GGML_ASSERT(txt_pad >= 0 && txt_pad <= txt->ne[1]);
        GGML_ASSERT(img_pad >= 0 && img_pad <= img->ne[1]);

        const int64_t txt_real_seq = txt->ne[1] - txt_pad;
        const int64_t img_real_seq = img->ne[1] - img_pad;
        GGML_ASSERT(txt_real_seq > 0);
        GGML_ASSERT(img_real_seq > 0);

        ggml_tensor* txt_real = qwen_sp_view_seq_major_sequence(ctx, txt, 0, txt_real_seq);
        ggml_tensor* img_real = qwen_sp_view_seq_major_sequence(ctx, img, 0, img_real_seq);
        return ggml_concat(ctx, txt_real, img_real, 1);
    }

    struct QwenSPFlashQKVViews {
        ggml_tensor* q = nullptr;
        ggml_tensor* k = nullptr;
        ggml_tensor* v = nullptr;
    };

    static inline QwenSPFlashQKVViews qwen_sp_fused_pack_flash_qkv_from_outputs(ggml_context* ctx,
                                                                                ggml_tensor* txt_q,
                                                                                ggml_tensor* img_q,
                                                                                ggml_tensor* txt_k,
                                                                                ggml_tensor* img_k,
                                                                                ggml_tensor* txt_v,
                                                                                ggml_tensor* img_v,
                                                                                int64_t head_dim,
                                                                                int64_t n_heads,
                                                                                const std::string& name_prefix) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(txt_q != nullptr && img_q != nullptr && txt_k != nullptr && img_k != nullptr);
        GGML_ASSERT(txt_v != nullptr && img_v != nullptr);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(n_heads > 0);
        GGML_ASSERT(txt_q->ne[1] > 0 && img_q->ne[1] > 0);
        GGML_ASSERT(txt_q->ne[1] == txt_k->ne[1] && txt_q->ne[1] == txt_v->ne[1]);
        GGML_ASSERT(img_q->ne[1] == img_k->ne[1] && img_q->ne[1] == img_v->ne[1]);

        const int64_t txt_seq   = txt_q->ne[1];
        const int64_t img_seq   = img_q->ne[1];
        const int64_t total_seq = txt_seq + img_seq;

        ggml_tensor* qk = qwen_fused_qk_pack(ctx,
                                             txt_q,
                                             img_q,
                                             txt_k,
                                             img_k,
                                             txt_seq,
                                             img_seq,
                                             head_dim,
                                             n_heads);
        ggml_tensor* v = qwen_fused_v_pack(ctx,
                                           txt_v,
                                           img_v,
                                           txt_seq,
                                           img_seq,
                                           head_dim,
                                           n_heads);
        ggml_set_name(qk, (name_prefix + "_qwen.fused_qk_pack.out").c_str());
        ggml_set_name(v, (name_prefix + "_qwen.fused_v_pack.out").c_str());

        GGML_ASSERT(qk->ne[0] == head_dim / 2);
        GGML_ASSERT(qk->ne[1] == total_seq);
        GGML_ASSERT(qk->ne[2] == n_heads);
        GGML_ASSERT(qk->ne[3] == 4);
        GGML_ASSERT(v->ne[0] == head_dim);
        GGML_ASSERT(v->ne[1] == total_seq);
        GGML_ASSERT(v->ne[2] == n_heads);
        GGML_ASSERT(v->ne[3] == 1);

        const size_t rope_plane_bytes = static_cast<size_t>(qk->nb[3]);
        QwenSPFlashQKVViews views;
        views.q = ggml_view_4d(ctx,
                               qk,
                               head_dim / 2,
                               total_seq,
                               n_heads,
                               2,
                               qk->nb[1],
                               qk->nb[2],
                               qk->nb[3],
                               0);
        views.k = ggml_view_4d(ctx,
                               qk,
                               head_dim / 2,
                               total_seq,
                               n_heads,
                               2,
                               qk->nb[1],
                               qk->nb[2],
                               qk->nb[3],
                               rope_plane_bytes * 2);
        views.v = v;
        ggml_set_name(views.q, (name_prefix + "_qwen.flash_q.from_fused_qk_pack").c_str());
        ggml_set_name(views.k, (name_prefix + "_qwen.flash_k.from_fused_qk_pack").c_str());
        ggml_set_name(views.v, (name_prefix + "_qwen.flash_v.from_fused_v_pack").c_str());
        return views;
    }

    static inline QwenSPFlashQKVViews qwen_sp_fused_pack_flash_qkv(ggml_context* ctx,
                                                                   ggml_tensor* txt_recv_flat,
                                                                   ggml_tensor* img_recv_flat,
                                                                   int64_t txt_padded_seq,
                                                                   int64_t img_padded_seq,
                                                                   int64_t txt_real_seq,
                                                                   int64_t img_real_seq,
                                                                   int64_t head_dim,
                                                                   int64_t n_heads,
                                                                   const std::string& name_prefix) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(txt_recv_flat != nullptr);
        GGML_ASSERT(img_recv_flat != nullptr);
        GGML_ASSERT(txt_padded_seq >= txt_real_seq);
        GGML_ASSERT(img_padded_seq >= img_real_seq);
        GGML_ASSERT(txt_real_seq > 0);
        GGML_ASSERT(img_real_seq > 0);
        const int64_t total_seq    = txt_real_seq + img_real_seq;
        GGML_ASSERT(total_seq == txt_real_seq + img_real_seq);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(n_heads > 0);
        GGML_ASSERT(txt_recv_flat->ne[0] == 3 * head_dim * n_heads * txt_padded_seq);
        GGML_ASSERT(img_recv_flat->ne[0] == 3 * head_dim * n_heads * img_padded_seq);

        ggml_tensor* qk = qwen_fused_qk_pack_from_recv(ctx,
                                                       txt_recv_flat,
                                                       img_recv_flat,
                                                       txt_real_seq,
                                                       img_real_seq,
                                                       head_dim,
                                                       n_heads);
        ggml_tensor* v = qwen_fused_v_pack_from_recv(ctx,
                                                     txt_recv_flat,
                                                     img_recv_flat,
                                                     txt_real_seq,
                                                     img_real_seq,
                                                     head_dim,
                                                     n_heads);
        ggml_set_name(qk, (name_prefix + "_qwen.fused_qk_pack.from_recv.out").c_str());
        ggml_set_name(v, (name_prefix + "_qwen.fused_v_pack.from_recv.out").c_str());

        GGML_ASSERT(qk->ne[0] == head_dim / 2);
        GGML_ASSERT(qk->ne[1] == total_seq);
        GGML_ASSERT(qk->ne[2] == n_heads);
        GGML_ASSERT(qk->ne[3] == 4);
        GGML_ASSERT(v->ne[0] == head_dim);
        GGML_ASSERT(v->ne[1] == total_seq);
        GGML_ASSERT(v->ne[2] == n_heads);
        GGML_ASSERT(v->ne[3] == 1);
        const size_t rope_plane_bytes = static_cast<size_t>(qk->nb[3]);

        QwenSPFlashQKVViews views;
        views.q = ggml_view_4d(ctx,
                               qk,
                               head_dim / 2,
                               total_seq,
                               n_heads,
                               2,
                               qk->nb[1],
                               qk->nb[2],
                               qk->nb[3],
                               0);
        views.k = ggml_view_4d(ctx,
                               qk,
                               head_dim / 2,
                               total_seq,
                               n_heads,
                               2,
                               qk->nb[1],
                               qk->nb[2],
                               qk->nb[3],
                               rope_plane_bytes * 2);
        views.v = v;
        ggml_set_name(views.q, (name_prefix + "_qwen.flash_q.from_fused_pack").c_str());
        ggml_set_name(views.k, (name_prefix + "_qwen.flash_k.from_fused_pack").c_str());
        ggml_set_name(views.v, (name_prefix + "_qwen.flash_v.from_fused_pack").c_str());
        return views;
    }

    static inline QwenSPFlashQKVViews qwen_sp_fused_pack_flash_qkv_from_joint_recv(ggml_context* ctx,
                                                                                   ggml_tensor* joint_recv_flat,
                                                                                   int64_t txt_padded_seq,
                                                                                   int64_t img_padded_seq,
                                                                                   int64_t txt_real_seq,
                                                                                   int64_t img_real_seq,
                                                                                   int64_t head_dim,
                                                                                   int64_t n_heads,
                                                                                   int world_size,
                                                                                   const std::string& name_prefix) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(joint_recv_flat != nullptr);
        GGML_ASSERT(txt_padded_seq >= txt_real_seq);
        GGML_ASSERT(img_padded_seq >= img_real_seq);
        GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(n_heads > 0);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(txt_padded_seq % world_size == 0);
        GGML_ASSERT(img_padded_seq % world_size == 0);
        GGML_ASSERT(joint_recv_flat->ne[0] == 3 * head_dim * n_heads * (txt_padded_seq + img_padded_seq));

        const int64_t total_seq = txt_real_seq + img_real_seq;
        ggml_tensor* qk = qwen_fused_qk_pack_from_joint_recv(ctx,
                                                             joint_recv_flat,
                                                             txt_padded_seq,
                                                             img_padded_seq,
                                                             txt_real_seq,
                                                             img_real_seq,
                                                             head_dim,
                                                             n_heads,
                                                             world_size);
        ggml_tensor* v = qwen_fused_v_pack_from_joint_recv(ctx,
                                                           joint_recv_flat,
                                                           txt_padded_seq,
                                                           img_padded_seq,
                                                           txt_real_seq,
                                                           img_real_seq,
                                                           head_dim,
                                                           n_heads,
                                                           world_size);
        ggml_set_name(qk, (name_prefix + "_qwen.fused_qk_pack.from_joint_recv.out").c_str());
        ggml_set_name(v, (name_prefix + "_qwen.fused_v_pack.from_joint_recv.out").c_str());

        GGML_ASSERT(qk->ne[0] == head_dim / 2);
        GGML_ASSERT(qk->ne[1] == total_seq);
        GGML_ASSERT(qk->ne[2] == n_heads);
        GGML_ASSERT(qk->ne[3] == 4);
        GGML_ASSERT(v->ne[0] == head_dim);
        GGML_ASSERT(v->ne[1] == total_seq);
        GGML_ASSERT(v->ne[2] == n_heads);
        GGML_ASSERT(v->ne[3] == 1);
        const size_t rope_plane_bytes = static_cast<size_t>(qk->nb[3]);

        QwenSPFlashQKVViews views;
        views.q = ggml_view_4d(ctx,
                               qk,
                               head_dim / 2,
                               total_seq,
                               n_heads,
                               2,
                               qk->nb[1],
                               qk->nb[2],
                               qk->nb[3],
                               0);
        views.k = ggml_view_4d(ctx,
                               qk,
                               head_dim / 2,
                               total_seq,
                               n_heads,
                               2,
                               qk->nb[1],
                               qk->nb[2],
                               qk->nb[3],
                               rope_plane_bytes * 2);
        views.v = v;
        ggml_set_name(views.q, (name_prefix + "_qwen.flash_q.from_joint_fused_pack").c_str());
        ggml_set_name(views.k, (name_prefix + "_qwen.flash_k.from_joint_fused_pack").c_str());
        ggml_set_name(views.v, (name_prefix + "_qwen.flash_v.from_joint_fused_pack").c_str());
        return views;
    }

    static inline ggml_tensor* qwen_sp_pad_head_sequence(ggml_context* ctx,
                                                        ggml_tensor* x,
                                                        int64_t pad) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pad >= 0);
        if (pad <= 0) {
            return x;
        }

        ggml_tensor* pad_tensor = ggml_ext_zeros(ctx, x->ne[0], x->ne[1], pad, x->ne[3]);
        return ggml_concat(ctx, x, pad_tensor, 2);
    }

    static inline ggml_tensor* qwen_sp_extract_stream_attention(ggml_context* ctx,
                                                               ggml_tensor* combined,
                                                               int64_t real_start,
                                                               int64_t real_seq,
                                                               int64_t pad) {
        ggml_tensor* out = qwen_sp_view_head_sequence(ctx,
                                                      combined,
                                                      real_start,
                                                      real_seq);
        out              = qwen_sp_pad_head_sequence(ctx, out, pad);
        return ggml_cont(ctx, out);
    }

    struct TimestepEmbedding : public GGMLBlock {
    public:
        TimestepEmbedding(int64_t in_channels,
                          int64_t time_embed_dim,
                          int64_t out_dim       = 0,
                          int64_t cond_proj_dim = 0,
                          bool sample_proj_bias = true) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim, sample_proj_bias));
            if (cond_proj_dim > 0) {
                blocks["cond_proj"] = std::shared_ptr<GGMLBlock>(new Linear(cond_proj_dim, in_channels, false));
            }
            if (out_dim <= 0) {
                out_dim = time_embed_dim;
            }
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, out_dim, sample_proj_bias));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* sample,
                             ggml_tensor* condition = nullptr) {
            if (condition != nullptr) {
                auto cond_proj = std::dynamic_pointer_cast<Linear>(blocks["cond_proj"]);
                sample         = ggml_add(ctx->ggml_ctx, sample, cond_proj->forward(ctx, condition));
            }
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            sample = linear_1->forward(ctx, sample);
            sample = ggml_silu_inplace(ctx->ggml_ctx, sample);
            sample = linear_2->forward(ctx, sample);
            return sample;
        }
    };

    struct QwenTimestepProjEmbeddings : public GGMLBlock {
    public:
        QwenTimestepProjEmbeddings(int64_t embedding_dim) {
            blocks["timestep_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedding(256, embedding_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* timesteps) {
            // timesteps: [N,]
            // return: [N, embedding_dim]
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["timestep_embedder"]);

            auto timesteps_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1.f);
            auto timesteps_emb  = timestep_embedder->forward(ctx, timesteps_proj);
            return timesteps_emb;
        }
    };

    struct QwenImageAttention : public GGMLBlock {
    protected:
        int64_t dim_head;

    public:
        QwenImageAttention(int64_t query_dim,
                           int64_t dim_head,
                           int64_t num_heads,
                           int64_t out_dim         = 0,
                           int64_t out_context_dim = 0,
                           bool bias               = true,
                           bool out_bias           = true,
                           float eps               = 1e-6)
            : dim_head(dim_head) {
            int64_t inner_dim = out_dim > 0 ? out_dim : dim_head * num_heads;
            out_dim           = out_dim > 0 ? out_dim : query_dim;
            out_context_dim   = out_context_dim > 0 ? out_context_dim : query_dim;

            blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));

            blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));
            blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));

            blocks["add_q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["add_k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["add_v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));

            blocks["norm_added_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));
            blocks["norm_added_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));

            float scale         = 1.f / 32.f;
            bool force_prec_f32 = false;

            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example when using CUDA but the weights are k-quants (not all prompts).
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim,
                                                                        out_dim,
                                                                        out_bias,
                                                                        false,
                                                                        force_prec_f32,
                                                                        scale,
                                                                        true));
            // to_out.1 is nn.Dropout

            blocks["to_add_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim,
                                                                          out_context_dim,
                                                                          out_bias,
                                                                          false,
                                                                          false,
                                                                          scale,
                                                                          true));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* mask = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])

            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                to_out_0->set_force_prec_f32(true);
            }

            auto norm_added_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_q"]);
            auto norm_added_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_k"]);

            auto add_q_proj = std::dynamic_pointer_cast<Linear>(blocks["add_q_proj"]);
            auto add_k_proj = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
            auto add_v_proj = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
            auto to_add_out = std::dynamic_pointer_cast<Linear>(blocks["to_add_out"]);

            int64_t N           = img->ne[2];
            int64_t n_img_token = img->ne[1];
            int64_t n_txt_token = txt->ne[1];

            auto img_q        = to_q->forward(ctx, img);
            int64_t num_heads = img_q->ne[0] / dim_head;
            img_q             = ggml_reshape_4d(ctx->ggml_ctx, img_q, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]
            auto img_k        = to_k->forward(ctx, img);
            img_k             = ggml_reshape_4d(ctx->ggml_ctx, img_k, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]
            auto img_v        = to_v->forward(ctx, img);
            img_v             = ggml_reshape_4d(ctx->ggml_ctx, img_v, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]

            img_q = norm_q->forward(ctx, img_q);
            img_k = norm_k->forward(ctx, img_k);

            auto txt_q = add_q_proj->forward(ctx, txt);
            txt_q      = ggml_reshape_4d(ctx->ggml_ctx, txt_q, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]
            auto txt_k = add_k_proj->forward(ctx, txt);
            txt_k      = ggml_reshape_4d(ctx->ggml_ctx, txt_k, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]
            auto txt_v = add_v_proj->forward(ctx, txt);
            txt_v      = ggml_reshape_4d(ctx->ggml_ctx, txt_v, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]

            txt_q = norm_added_q->forward(ctx, txt_q);
            txt_k = norm_added_k->forward(ctx, txt_k);

            const bool qkv_quantized =
                to_q->weight_is_quantized() ||
                to_k->weight_is_quantized() ||
                to_v->weight_is_quantized() ||
                add_q_proj->weight_is_quantized() ||
                add_k_proj->weight_is_quantized() ||
                add_v_proj->weight_is_quantized();
            ggml_tensor* attn = nullptr;
            if (qwen_single_fused_attention_enabled() &&
                !qkv_quantized &&
                mask == nullptr &&
                N == 1 &&
                ctx->flash_attn_enabled &&
                sd_backend_is(ctx->backend, "CUDA")) {
                auto q = ggml_concat(ctx->ggml_ctx,
                                     qwen_single_seq_major_view(ctx->ggml_ctx, txt_q, dim_head),
                                     qwen_single_seq_major_view(ctx->ggml_ctx, img_q, dim_head),
                                     1);
                auto k = ggml_concat(ctx->ggml_ctx,
                                     qwen_single_seq_major_view(ctx->ggml_ctx, txt_k, dim_head),
                                     qwen_single_seq_major_view(ctx->ggml_ctx, img_k, dim_head),
                                     1);
                auto v = ggml_concat(ctx->ggml_ctx,
                                     qwen_single_seq_major_view(ctx->ggml_ctx, txt_v, dim_head),
                                     qwen_single_seq_major_view(ctx->ggml_ctx, img_v, dim_head),
                                     1);
                attn = qwen_single_attention_seq_major(ctx, q, k, v, pe, 1.0f);
            } else {
                auto q = ggml_concat(ctx->ggml_ctx, txt_q, img_q, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
                auto k = ggml_concat(ctx->ggml_ctx, txt_k, img_k, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
                auto v = ggml_concat(ctx->ggml_ctx, txt_v, img_v, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
                attn   = Rope::attention(ctx, q, k, v, pe, mask, (1.0f / 128.f));  // [N, n_txt_token + n_img_token, n_head*d_head]
            }
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             txt->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);  // [N, n_txt_token, n_head*d_head]
            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             img->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             txt->ne[1] * attn->nb[1]);  // [N, n_img_token, n_head*d_head]
            img_attn_out      = ggml_cont(ctx->ggml_ctx, img_attn_out);
            txt_attn_out      = ggml_cont(ctx->ggml_ctx, txt_attn_out);

            img_attn_out = to_out_0->forward(ctx, img_attn_out);
            txt_attn_out = to_add_out->forward(ctx, txt_attn_out);

            return {img_attn_out, txt_attn_out};
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward_sp(GGMLRunnerContext* ctx,
                                                         ggml_tensor* img,
                                                         ggml_tensor* txt,
                                                         ggml_tensor* pe,
                                                         ggml_tensor* prepared_pe_seq_major,
                                                         int64_t txt_pad,
                                                         int64_t img_pad,
                                                         const std::string& name_prefix) {
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                to_out_0->set_force_prec_f32(true);
            }

            auto norm_added_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_q"]);
            auto norm_added_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_k"]);

            auto add_q_proj = std::dynamic_pointer_cast<Linear>(blocks["add_q_proj"]);
            auto add_k_proj = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
            auto add_v_proj = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
            auto to_add_out = std::dynamic_pointer_cast<Linear>(blocks["to_add_out"]);

            const int world_size       = qwen_sp_world_size(ctx);
            const int64_t N            = img->ne[2];
            const int64_t img_local_seq = img->ne[1];
            const int64_t txt_local_seq = txt->ne[1];

            GGML_ASSERT(N == 1);
            GGML_ASSERT(txt->ne[2] == N);

            auto img_q        = to_q->forward(ctx, img);
            const int64_t num_heads = img_q->ne[0] / dim_head;
            img_q             = ggml_reshape_4d(ctx->ggml_ctx, img_q, dim_head, num_heads, img_local_seq, N);
            auto img_k        = to_k->forward(ctx, img);
            img_k             = ggml_reshape_4d(ctx->ggml_ctx, img_k, dim_head, num_heads, img_local_seq, N);
            auto img_v        = to_v->forward(ctx, img);
            img_v             = ggml_reshape_4d(ctx->ggml_ctx, img_v, dim_head, num_heads, img_local_seq, N);

            img_q = norm_q->forward(ctx, img_q);
            img_k = norm_k->forward(ctx, img_k);

            auto txt_q = add_q_proj->forward(ctx, txt);
            txt_q      = ggml_reshape_4d(ctx->ggml_ctx, txt_q, dim_head, num_heads, txt_local_seq, N);
            auto txt_k = add_k_proj->forward(ctx, txt);
            txt_k      = ggml_reshape_4d(ctx->ggml_ctx, txt_k, dim_head, num_heads, txt_local_seq, N);
            auto txt_v = add_v_proj->forward(ctx, txt);
            txt_v      = ggml_reshape_4d(ctx->ggml_ctx, txt_v, dim_head, num_heads, txt_local_seq, N);

            txt_q = norm_added_q->forward(ctx, txt_q);
            txt_k = norm_added_k->forward(ctx, txt_k);

            const int64_t txt_full_seq_for_attn = txt_local_seq * world_size;
            const int64_t img_full_seq_for_attn = img_local_seq * world_size;
            const int64_t txt_real_seq_for_attn = txt_full_seq_for_attn - txt_pad;
            const int64_t img_real_seq_for_attn = img_full_seq_for_attn - img_pad;
            QwenSPFlashQKVViews fused_qkv;
            if (qwen_sp_fuse_qkv_a2a_enabled()) {
                auto joint_qkv_head =
                    qwen_sp_f16_qkv_send_enabled() ?
                        edgedit::parallel::sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only(
                            ctx->ggml_ctx,
                            txt_q,
                            txt_k,
                            txt_v,
                            img_q,
                            img_k,
                            img_v,
                            ctx->process_group,
                            world_size,
                            name_prefix + "_txt_img_qkv_seq_to_head") :
                        [&]() {
                            ggml_tensor* joint_qkv_send_flat = qwen_fused_joint_qkv_send_pack(ctx->ggml_ctx,
                                                                                              txt_q,
                                                                                              txt_k,
                                                                                              txt_v,
                                                                                              img_q,
                                                                                              img_k,
                                                                                              img_v,
                                                                                              world_size);
                            ggml_set_name(joint_qkv_send_flat, (name_prefix + "_txt_img_qkv_fused_send_pack").c_str());
                            return edgedit::parallel::sp_all_to_all_4d_seq_to_head_packed_recv_only(
                                ctx->ggml_ctx,
                                joint_qkv_send_flat,
                                dim_head * 3,
                                num_heads,
                                txt_local_seq + img_local_seq,
                                N,
                                ctx->process_group,
                                world_size,
                                name_prefix + "_txt_img_qkv_seq_to_head");
                        }();
                GGML_ASSERT(joint_qkv_head.recv_flat != nullptr);
                fused_qkv = qwen_sp_fused_pack_flash_qkv_from_joint_recv(ctx->ggml_ctx,
                                                                         joint_qkv_head.recv_flat,
                                                                         txt_full_seq_for_attn,
                                                                         img_full_seq_for_attn,
                                                                         txt_real_seq_for_attn,
                                                                         img_real_seq_for_attn,
                                                                         dim_head,
                                                                         num_heads / world_size,
                                                                         world_size,
                                                                         name_prefix);
            } else {
                ggml_tensor* txt_qkv_send_flat = qwen_fused_qkv_send_pack(ctx->ggml_ctx,
                                                                          txt_q,
                                                                          txt_k,
                                                                          txt_v,
                                                                          world_size);
                ggml_set_name(txt_qkv_send_flat, (name_prefix + "_txt_qkv_fused_send_pack").c_str());
                auto txt_qkv_head = edgedit::parallel::sp_all_to_all_4d_seq_to_head_packed_recv_only(
                    ctx->ggml_ctx,
                    txt_qkv_send_flat,
                    dim_head * 3,
                    num_heads,
                    txt_local_seq,
                    N,
                    world_size,
                    name_prefix + "_txt_qkv_seq_to_head");
                ggml_tensor* img_qkv_send_flat = qwen_fused_qkv_send_pack(ctx->ggml_ctx,
                                                                          img_q,
                                                                          img_k,
                                                                          img_v,
                                                                          world_size);
                ggml_set_name(img_qkv_send_flat, (name_prefix + "_img_qkv_fused_send_pack").c_str());
                auto img_qkv_head = edgedit::parallel::sp_all_to_all_4d_seq_to_head_packed_recv_only(
                    ctx->ggml_ctx,
                    img_qkv_send_flat,
                    dim_head * 3,
                    num_heads,
                    img_local_seq,
                    N,
                    world_size,
                    name_prefix + "_img_qkv_seq_to_head");
                GGML_ASSERT(txt_qkv_head.recv_flat != nullptr);
                GGML_ASSERT(img_qkv_head.recv_flat != nullptr);

                fused_qkv = qwen_sp_fused_pack_flash_qkv(ctx->ggml_ctx,
                                                         txt_qkv_head.recv_flat,
                                                         img_qkv_head.recv_flat,
                                                         txt_full_seq_for_attn,
                                                         img_full_seq_for_attn,
                                                         txt_real_seq_for_attn,
                                                         img_real_seq_for_attn,
                                                         dim_head,
                                                         num_heads / world_size,
                                                         name_prefix);
            }
            ggml_tensor* q = fused_qkv.q;
            ggml_tensor* k = fused_qkv.k;
            ggml_tensor* v = fused_qkv.v;

            const int64_t shard_heads = num_heads / world_size;
            const int64_t txt_full_seq = txt_local_seq * world_size;
            const int64_t img_full_seq = img_local_seq * world_size;
            const int64_t txt_real_seq = txt_full_seq - txt_pad;
            const int64_t img_real_seq = img_full_seq - img_pad;
            const int64_t total_seq    = txt_real_seq + img_real_seq;

            ggml_tensor* attn = qwen_sp_attention_from_rope_work_layout(ctx,
                                                                        q,
                                                                        k,
                                                                        v,
                                                                        pe,
                                                                        dim_head,
                                                                        1.0f / 128.f,
                                                                        prepared_pe_seq_major);
            sd::ggml_graph_cut::mark_graph_cut(attn, name_prefix + ".sp_attention", "attn");

            ggml_tensor* attn_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                   attn,
                                                   dim_head,
                                                   shard_heads,
                                                   total_seq,
                                                   N);

            const bool use_f16_head_to_seq = qwen_sp_f16_head_to_seq_enabled();
            ggml_tensor* attn_head_to_seq_send = qwen_fused_attn_head_to_seq_send_pack(ctx->ggml_ctx,
                                                                                       attn_4d,
                                                                                       txt_real_seq,
                                                                                       img_real_seq,
                                                                                       txt_full_seq,
                                                                                       img_full_seq,
                                                                                       world_size,
                                                                                       use_f16_head_to_seq);
            ggml_set_name(attn_head_to_seq_send, (name_prefix + "_qwen.fused_attn_head_to_seq_send_pack").c_str());
            auto attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_recv_only(
                ctx->ggml_ctx,
                attn_head_to_seq_send,
                dim_head,
                shard_heads,
                {txt_full_seq, img_full_seq},
                ctx->process_group,
                world_size,
                name_prefix + "_txt_img_attn_head_to_seq");
            GGML_ASSERT(attn_local.recv_flat != nullptr);

            ggml_tensor* txt_attn_local = qwen_fused_attn_head_to_seq_recv_unpack(ctx->ggml_ctx,
                                                                                 attn_local.recv_flat,
                                                                                 0,
                                                                                 txt_full_seq,
                                                                                 img_full_seq,
                                                                                 dim_head,
                                                                                 num_heads,
                                                                                 world_size,
                                                                                 GGML_TYPE_F32);
            ggml_tensor* img_attn_local = qwen_fused_attn_head_to_seq_recv_unpack(ctx->ggml_ctx,
                                                                                 attn_local.recv_flat,
                                                                                 1,
                                                                                 txt_full_seq,
                                                                                 img_full_seq,
                                                                                 dim_head,
                                                                                 num_heads,
                                                                                 world_size,
                                                                                 GGML_TYPE_F32);
            ggml_set_name(txt_attn_local, (name_prefix + "_qwen.fused_attn_recv_unpack_txt").c_str());
            ggml_set_name(img_attn_local, (name_prefix + "_qwen.fused_attn_recv_unpack_img").c_str());

            ggml_tensor* txt_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                                        txt_attn_local,
                                                        dim_head * num_heads,
                                                        txt_attn_local->ne[2],
                                                        N);
            ggml_tensor* img_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                                        img_attn_local,
                                                        dim_head * num_heads,
                                                        img_attn_local->ne[2],
                                                        N);

            img_attn_out = to_out_0->forward(ctx, img_attn_out);
            txt_attn_out = to_add_out->forward(ctx, txt_attn_out);

            return {img_attn_out, txt_attn_out};
        }
    };

    class QwenImageTransformerBlock : public GGMLBlock {
    protected:
        bool zero_cond_t;

    public:
        QwenImageTransformerBlock(int64_t dim,
                                  int64_t num_attention_heads,
                                  int64_t attention_head_dim,
                                  float eps        = 1e-6,
                                  bool zero_cond_t = false)
            : zero_cond_t(zero_cond_t) {
            // img_mod.0 is nn.SiLU()
            blocks["img_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 6 * dim, true));

            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_mlp"]   = std::shared_ptr<GGMLBlock>(
                new FeedForward(dim, dim, 4, FeedForward::Activation::GELU, true, true));

            // txt_mod.0 is nn.SiLU()
            blocks["txt_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 6 * dim, true));

            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_mlp"]   = std::shared_ptr<GGMLBlock>(
                new FeedForward(dim, dim, 4, FeedForward::Activation::GELU, true, true));

            blocks["attn"] = std::shared_ptr<GGMLBlock>(new QwenImageAttention(dim,
                                                                               attention_head_dim,
                                                                               num_attention_heads,
                                                                               0,     // out_dim
                                                                               0,     // out_context-dim
                                                                               true,  // bias
                                                                               true,  // out_bias
                                                                               eps));
        }

        std::vector<ggml_tensor*> get_mod_params_vec(ggml_context* ctx, ggml_tensor* mod_params, ggml_tensor* index = nullptr) {
            // index: [N, n_img_token]
            // mod_params: [N, hidden_size * 12]
            if (index == nullptr) {
                return ggml_ext_chunk(ctx, mod_params, 6, 0);
            }
            mod_params          = ggml_reshape_1d(ctx, mod_params, ggml_nelements(mod_params));
            auto mod_params_vec = ggml_ext_chunk(ctx, mod_params, 12, 0);
            index               = ggml_reshape_3d(ctx, index, 1, index->ne[0], index->ne[1]);                                      // [N, n_img_token, 1]
            index               = ggml_repeat_4d(ctx, index, mod_params_vec[0]->ne[0], index->ne[1], index->ne[2], index->ne[3]);  // [N, n_img_token, hidden_size]
            std::vector<ggml_tensor*> mod_results;
            for (int i = 0; i < 6; i++) {
                auto mod_0 = mod_params_vec[i];
                auto mod_1 = mod_params_vec[i + 6];

                // mod_result = torch.where(index == 0, mod_0, mod_1)
                // mod_result = (1 - index)*mod_0 + index*mod_1
                mod_0           = ggml_sub(ctx, ggml_repeat(ctx, mod_0, index), ggml_mul(ctx, index, mod_0));  // [N, n_img_token, hidden_size]
                mod_1           = ggml_mul(ctx, index, mod_1);                                                 // [N, n_img_token, hidden_size]
                auto mod_result = ggml_add(ctx, mod_0, mod_1);
                mod_results.push_back(mod_result);
            }
            return mod_results;
        }

        // residual + x * gate, fused into one CUDA kernel when shapes allow.
        // gate is [dim, N] (2D) on the normal path or [dim, token, N] (3D) on the
        // modulate_index path; reshape the 2D case to [dim, 1, N] so it broadcasts
        // over the token axis exactly like the ggml_mul fallback below.
        static ggml_tensor* residual_gate(ggml_context* ctx, ggml_tensor* residual, ggml_tensor* x, ggml_tensor* gate) {
#ifdef ED_ENABLE_CUDA_MODULATION
            ggml_tensor* gate_b = gate;
            if (ggml_n_dims(gate) == 2) {
                gate_b = ggml_reshape_3d(ctx, gate, gate->ne[0], 1, gate->ne[1]);
            }
            if (auto fused = edgedit::ggml_ext::fused_residual_gate_custom(ctx, residual, x, gate_b)) {
                return fused;
            }
#endif
            return ggml_add(ctx, residual, ggml_mul(ctx, x, gate));
        }

        virtual std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                              ggml_tensor* img,
                                                              ggml_tensor* txt,
                                                              ggml_tensor* t_emb,
                                                              ggml_tensor* pe,
                                                              ggml_tensor* modulate_index = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])

            auto img_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["img_mlp"]);

            auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["txt_mod.1"]);
            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["txt_mlp"]);

            auto attn = std::dynamic_pointer_cast<QwenImageAttention>(blocks["attn"]);

            auto img_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            img_mod_params         = img_mod_1->forward(ctx, img_mod_params);
            auto img_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, img_mod_params, modulate_index);

            if (zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            auto txt_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            txt_mod_params         = txt_mod_1->forward(ctx, txt_mod_params);
            auto txt_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, txt_mod_params);

            auto img_normed    = img_norm1->forward(ctx, img);
            auto img_modulated = dit::modulate(ctx->ggml_ctx, img_normed, img_mod_param_vec[0], img_mod_param_vec[1], modulate_index != nullptr);
            auto img_gate1     = img_mod_param_vec[2];

            auto txt_normed    = txt_norm1->forward(ctx, txt);
            auto txt_modulated = dit::modulate(ctx->ggml_ctx, txt_normed, txt_mod_param_vec[0], txt_mod_param_vec[1]);
            auto txt_gate1     = txt_mod_param_vec[2];

            auto [img_attn_output, txt_attn_output] = attn->forward(ctx, img_modulated, txt_modulated, pe);

            img = residual_gate(ctx->ggml_ctx, img, img_attn_output, img_gate1);
            txt = residual_gate(ctx->ggml_ctx, txt, txt_attn_output, txt_gate1);

            auto img_normed2    = img_norm2->forward(ctx, img);
            auto img_modulated2 = dit::modulate(ctx->ggml_ctx, img_normed2, img_mod_param_vec[3], img_mod_param_vec[4], modulate_index != nullptr);
            auto img_gate2      = img_mod_param_vec[5];

            auto txt_normed2    = txt_norm2->forward(ctx, txt);
            auto txt_modulated2 = dit::modulate(ctx->ggml_ctx, txt_normed2, txt_mod_param_vec[3], txt_mod_param_vec[4]);
            auto txt_gate2      = txt_mod_param_vec[5];

            auto img_mlp_out = img_mlp->forward(ctx, img_modulated2);
            auto txt_mlp_out = txt_mlp->forward(ctx, txt_modulated2);

            img = residual_gate(ctx->ggml_ctx, img, img_mlp_out, img_gate2);
            txt = residual_gate(ctx->ggml_ctx, txt, txt_mlp_out, txt_gate2);

            return {img, txt};
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward_sp(GGMLRunnerContext* ctx,
                                                         ggml_tensor* img,
                                                         ggml_tensor* txt,
                                                         ggml_tensor* t_emb,
                                                         ggml_tensor* pe,
                                                         ggml_tensor* prepared_pe_seq_major,
                                                         ggml_tensor* modulate_index,
                                                         int64_t txt_pad,
                                                         int64_t img_pad,
                                                         const std::string& name_prefix) {
            auto img_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["img_mlp"]);

            auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["txt_mod.1"]);
            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["txt_mlp"]);

            auto attn = std::dynamic_pointer_cast<QwenImageAttention>(blocks["attn"]);

            auto img_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            img_mod_params         = img_mod_1->forward(ctx, img_mod_params);
            auto img_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, img_mod_params, modulate_index);

            if (zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            auto txt_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            txt_mod_params         = txt_mod_1->forward(ctx, txt_mod_params);
            auto txt_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, txt_mod_params);

            auto img_normed    = img_norm1->forward(ctx, img);
            auto img_modulated = dit::modulate(ctx->ggml_ctx, img_normed, img_mod_param_vec[0], img_mod_param_vec[1], modulate_index != nullptr);
            auto img_gate1     = img_mod_param_vec[2];

            auto txt_normed    = txt_norm1->forward(ctx, txt);
            auto txt_modulated = dit::modulate(ctx->ggml_ctx, txt_normed, txt_mod_param_vec[0], txt_mod_param_vec[1]);
            auto txt_gate1     = txt_mod_param_vec[2];

            auto [img_attn_output, txt_attn_output] = attn->forward_sp(ctx,
                                                                       img_modulated,
                                                                       txt_modulated,
                                                                       pe,
                                                                       prepared_pe_seq_major,
                                                                       txt_pad,
                                                                       img_pad,
                                                                       name_prefix);

            img = residual_gate(ctx->ggml_ctx, img, img_attn_output, img_gate1);
            txt = residual_gate(ctx->ggml_ctx, txt, txt_attn_output, txt_gate1);

            auto img_normed2    = img_norm2->forward(ctx, img);
            auto img_modulated2 = dit::modulate(ctx->ggml_ctx, img_normed2, img_mod_param_vec[3], img_mod_param_vec[4], modulate_index != nullptr);
            auto img_gate2      = img_mod_param_vec[5];

            auto txt_normed2    = txt_norm2->forward(ctx, txt);
            auto txt_modulated2 = dit::modulate(ctx->ggml_ctx, txt_normed2, txt_mod_param_vec[3], txt_mod_param_vec[4]);
            auto txt_gate2      = txt_mod_param_vec[5];

            auto img_mlp_out = img_mlp->forward(ctx, img_modulated2);
            auto txt_mlp_out = txt_mlp->forward(ctx, txt_modulated2);

            img = residual_gate(ctx->ggml_ctx, img, img_mlp_out, img_gate2);
            txt = residual_gate(ctx->ggml_ctx, txt, txt_mlp_out, txt_gate2);

            return {img, txt};
        }
    };

    struct AdaLayerNormContinuous : public GGMLBlock {
    public:
        AdaLayerNormContinuous(int64_t embedding_dim,
                               int64_t conditioning_embedding_dim,
                               bool elementwise_affine = true,
                               float eps               = 1e-5f,
                               bool bias               = true) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(conditioning_embedding_dim, eps, elementwise_affine, bias));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(conditioning_embedding_dim, embedding_dim * 2, bias));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]

            auto norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto emb   = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
            auto mods  = ggml_ext_chunk(ctx->ggml_ctx, emb, 2, 0);
            auto scale = mods[0];
            auto shift = mods[1];

            x = norm->forward(ctx, x);
            x = dit::modulate(ctx->ggml_ctx, x, shift, scale);

            return x;
        }
    };

    struct QwenImageParams {
        int patch_size              = 2;
        int64_t in_channels         = 64;
        int64_t out_channels        = 16;
        int num_layers              = 60;
        int64_t attention_head_dim  = 128;
        int64_t num_attention_heads = 24;
        int64_t joint_attention_dim = 3584;
        int theta                   = 10000;
        std::vector<int> axes_dim   = {16, 56, 56};
        int axes_dim_sum            = 128;
        bool zero_cond_t            = false;
    };

    class QwenImageModel : public GGMLBlock {
    protected:
        QwenImageParams params;

    public:
        QwenImageModel() {}
        QwenImageModel(QwenImageParams params)
            : params(params) {
            int64_t inner_dim         = params.num_attention_heads * params.attention_head_dim;
            blocks["time_text_embed"] = std::shared_ptr<GGMLBlock>(new QwenTimestepProjEmbeddings(inner_dim));
            blocks["txt_norm"]        = std::shared_ptr<GGMLBlock>(new RMSNorm(params.joint_attention_dim, 1e-6f));
            blocks["img_in"]          = std::shared_ptr<GGMLBlock>(new Linear(params.in_channels, inner_dim));
            blocks["txt_in"]          = std::shared_ptr<GGMLBlock>(new Linear(params.joint_attention_dim, inner_dim));

            // blocks
            for (int i = 0; i < params.num_layers; i++) {
                auto block                                        = std::shared_ptr<GGMLBlock>(new QwenImageTransformerBlock(inner_dim,
                                                                                                                             params.num_attention_heads,
                                                                                                                             params.attention_head_dim,
                                                                                                                             1e-6f,
                                                                                                                             params.zero_cond_t));
                blocks["transformer_blocks." + std::to_string(i)] = block;
            }

            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormContinuous(inner_dim, inner_dim, false, 1e-6f));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, params.patch_size * params.patch_size * params.out_channels));
        }

        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe,
                                  ggml_tensor* modulate_index = nullptr) {
            auto time_text_embed = std::dynamic_pointer_cast<QwenTimestepProjEmbeddings>(blocks["time_text_embed"]);
            auto txt_norm        = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);
            auto img_in          = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in          = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto norm_out        = std::dynamic_pointer_cast<AdaLayerNormContinuous>(blocks["norm_out"]);
            auto proj_out        = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            auto t_emb = time_text_embed->forward(ctx, timestep);
            if (params.zero_cond_t) {
                auto t_emb_0 = time_text_embed->forward(ctx, ggml_ext_zeros_like(ctx->ggml_ctx, timestep));
                t_emb        = ggml_concat(ctx->ggml_ctx, t_emb, t_emb_0, 1);
            }
            auto img = img_in->forward(ctx, x);
            auto txt = txt_norm->forward(ctx, context);
            txt      = txt_in->forward(ctx, txt);

            bool use_sp_mainline = qwen_sp_enabled(ctx);
            edgedit::parallel::SPSequenceSplit img_sp_split;
            edgedit::parallel::SPSequenceSplit txt_sp_split;
            edgedit::parallel::SPSequenceSplit modulate_index_sp_split;
            ggml_tensor* sp_pe = pe;
            ggml_tensor* sp_prepared_pe_seq_major = nullptr;
            if (use_sp_mainline) {
                const int rank       = qwen_sp_rank(ctx);
                const int world_size = qwen_sp_world_size(ctx);
                const int64_t img_pad = edgedit::parallel::sp_sequence_padding(img->ne[1],
                                                                                world_size);
                const int64_t txt_pad = edgedit::parallel::sp_sequence_padding(txt->ne[1],
                                                                                world_size);
                const int64_t heads_pad = params.num_attention_heads % world_size;
                if (heads_pad != 0 || img->ne[2] != 1 || txt->ne[2] != 1) {
                    use_sp_mainline = false;
                } else {
                    img_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                        img,
                                                                        rank,
                                                                        world_size,
                                                                        1,
                                                                        "qwen_image_sp_img_split");
                    txt_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                        txt,
                                                                        rank,
                                                                        world_size,
                                                                        1,
                                                                        "qwen_image_sp_txt_split");
                    img = img_sp_split.local;
                    txt = txt_sp_split.local;

                    if (modulate_index != nullptr) {
                        ggml_tensor* modulate_index_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                                          modulate_index,
                                                                          1,
                                                                          modulate_index->ne[0],
                                                                          1,
                                                                          1);
                        modulate_index_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                                       modulate_index_4d,
                                                                                       rank,
                                                                                       world_size,
                                                                                       1,
                                                                                       "qwen_image_sp_modulate_index_split");
                        modulate_index = ggml_reshape_2d(ctx->ggml_ctx,
                                                         modulate_index_sp_split.local,
                                                         modulate_index_sp_split.local->ne[1],
                                                         modulate_index_sp_split.local->ne[2]);
                    }
                    sp_prepared_pe_seq_major = qwen_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx, sp_pe);
                }
            }
            sd::ggml_graph_cut::mark_graph_cut(img, "qwen_image.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(txt, "qwen_image.prelude", "txt");
            // sd::ggml_graph_cut::mark_graph_cut(t_emb, "qwen_image.prelude", "t_emb");

            // Cache seam: the transformer stack transforms the image stream
            // `img`. Disabled under SP (block-loop tensors are sequence-sharded).
            // The cached region is blocks [region_start, region_end); the default
            // whole-stack region matches the pre-region behaviour. Capture builds
            // the region residual in-loop at the region's last block, so the seam
            // is only active on the non-SP path (SP is gated off above) where no
            // post-loop gather rewrites `img`.
            sd::CacheGraphScope* cache_scope = use_sp_mainline ? nullptr : ctx->cache_scope;
            const bool cache_inject = cache_scope != nullptr && cache_scope->inject_mode();

            // Substep tap: block-stack input anchor (ModelIn). Conditional — a no-op
            // unless the middle layer requested it this substep. Coexists with the
            // legacy cache_scope path above (double-write during migration).
            tap(ctx, edgedit::cache::AnchorRef::model_in(), img);

            for (int i = 0; i < params.num_layers; i++) {
                if (cache_inject) {
                    if (ggml_tensor* injected = cache_scope->step_inject_region(ctx->ggml_ctx, i, img)) {
                        img = injected;
                        i = cache_scope->inject_resume_index(params.num_layers) - 1;
                        continue;
                    }
                }
                if (cache_scope != nullptr) {
                    cache_scope->begin_region(i, img);
                }

                auto block = std::dynamic_pointer_cast<QwenImageTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);

                std::pair<ggml_tensor*, ggml_tensor*> result;
                if (use_sp_mainline) {
                    result = block->forward_sp(ctx,
                                               img,
                                               txt,
                                               t_emb,
                                               sp_pe,
                                               sp_prepared_pe_seq_major,
                                               modulate_index,
                                               txt_sp_split.pad,
                                               img_sp_split.pad,
                                               "qwen_image_block" + std::to_string(i));
                } else
                {
                    result = block->forward(ctx, img, txt, t_emb, pe, modulate_index);
                }
                img         = result.first;
                txt         = result.second;
                sd::ggml_graph_cut::mark_graph_cut(img, "qwen_image.transformer_blocks." + std::to_string(i), "img");
                if (i + 1 < params.num_layers) {
                    sd::ggml_graph_cut::mark_graph_cut(txt, "qwen_image.transformer_blocks." + std::to_string(i), "txt");
                }
                // Substep tap: block output k (BlockOut[i]). Conditional no-op unless
                // requested. Also drives substep stop (probe): when the registry marks
                // a stop-after-block, return the block-stack state here.
                tap(ctx, edgedit::cache::AnchorRef::block_out(i), img);
                if (ctx->tap_registry != nullptr && ctx->tap_registry->stop_after(i)) {
                    return img;
                }
                if (cache_scope != nullptr) {
                    cache_scope->end_region(ctx->ggml_ctx, i, params.num_layers, img);
                    // GPU DiCache: on a full (capture) step, snapshot the
                    // probe-depth hidden state so the runner refreshes prev_probe
                    // for the next step's decision.
                    cache_scope->record_capture_probe_state(ctx->ggml_ctx, i, img);
                    if (cache_scope->stop_after_block(i)) {
                        cache_scope->on_probe(img);
                        return img;
                    }
                }
            }

            // Substep tap: block-stack output anchor (ModelOut). This is the
            // block-stack residual's "after" point (matches build_feature's capture
            // site), i.e. before norm_out/proj_out. Conditional no-op unless
            // requested.
            tap(ctx, edgedit::cache::AnchorRef::model_out(), img);

            if (use_sp_mainline) {
                auto img_gather = edgedit::parallel::sp_mark_gather_sequence(ctx->ggml_ctx,
                                                                                 img,
                                                                                 qwen_sp_world_size(ctx),
                                                                                 1,
                                                                                 img_sp_split.pad,
                                                                                 "qwen_image_sp_final_img_gather",
                                                                                 ctx->process_group);
                img = img_gather.gathered;
            }

            if (params.zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            img = norm_out->forward(ctx, img, t_emb);
            img = proj_out->forward(ctx, img);

            return img;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {},
                             ggml_tensor* modulate_index           = nullptr) {
            // Forward pass of DiT.
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            auto img           = DiT::pad_and_patchify(ctx, x, params.patch_size, params.patch_size);
            int64_t img_tokens = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, params.patch_size, params.patch_size);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_orig(ctx, img, timestep, context, pe, modulate_index);  // [N, h_len*w_len, ph*pw*C]

            if (out->ne[1] > img_tokens) {
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [num_tokens, N, C * patch_size * patch_size]
                out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], out->ne[1], img_tokens, out->nb[1], out->nb[2], 0);
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [N, h*w, C * patch_size * patch_size]
            }

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, params.patch_size, params.patch_size);  // [N, C, H, W]

            return out;
        }
    };

    struct QwenImageRunner : public GGMLRunner {
    public:
        QwenImageParams qwen_image_params;
        QwenImageModel qwen_image;
        std::vector<float> pe_vec;
        std::vector<float> modulate_index_vec;
        SDVersion version;
        sd::Tensor<float> inject_feature_host_;  // kept alive across cache inject build
        sd::Tensor<float> gamma_scalar_host_;    // GPU DiCache: [1] gamma for reconstruct

        // ---- Persistent cross-step GPU state for DiCache (Probe granularity) ----
        // Mirrors FluxRunner::DiCacheGpuState: holds the last computed step's probe
        // state, block input, probe residuals (2-deep) and full residuals (2-deep)
        // as GPU tensors, per CFG branch, so the decision metric and residual
        // reconstruction run on-device instead of reading ~50MB back to host each
        // step. Allocated lazily at the live img shape; freed on reset / shape
        // change. See dicache_perf notes.
        struct DiCacheGpuState {
            ggml_context* ctx = nullptr;
            ggml_backend_buffer_t buffer = nullptr;
            ggml_tensor* prev_probe = nullptr;
            ggml_tensor* prev_input = nullptr;
            ggml_tensor* probe_prev1 = nullptr;
            ggml_tensor* probe_prev2 = nullptr;
            ggml_tensor* resid_prev1 = nullptr;
            ggml_tensor* resid_prev2 = nullptr;
            std::vector<int64_t> shape;   // [ne0, ne1, ne2] the tensors were built for
            bool has_probe_hist = false;  // prev_probe/prev_input valid
            int probe_resid_count = 0;    // 0/1/2 valid probe residuals
            int resid_count = 0;          // 0/1/2 valid full residuals
            void free() {
                if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
                prev_probe = prev_input = probe_prev1 = probe_prev2 = resid_prev1 = resid_prev2 = nullptr;
                shape.clear(); has_probe_hist = false; probe_resid_count = 0; resid_count = 0;
            }
        };
        std::unordered_map<const void*, DiCacheGpuState> dicache_gpu_states_;
        int dicache_probe_depth_ = 1;  // set by the pipeline before capture/probe

        // GPU DiCache is the default: on-device metric + reconstruction avoids the
        // per-step host scalar work and ~50MB GPU->host readback. Set ED_DICACHE_GPU=0
        // to fall back to the host path.
        static bool dicache_gpu_enabled() {
            const char* v = std::getenv("ED_DICACHE_GPU");
            if (v == nullptr || v[0] == '\0') {
                return true;  // default on
            }
            return v[0] != '0';
        }

        // Ensure the per-branch persistent DiCache GPU state exists and matches the
        // current block-stack shape [ne0, ne1, ne2]. (Re)allocates on first use or a
        // shape change; tensors are marked invalid until the first capture fills them.
        DiCacheGpuState& ensure_dicache_gpu_state(const void* branch_key,
                                                  int64_t ne0, int64_t ne1, int64_t ne2) {
            DiCacheGpuState& s = dicache_gpu_states_[branch_key];
            const std::vector<int64_t> want = {ne0, ne1, ne2};
            if (s.buffer != nullptr && s.shape == want) {
                return s;
            }
            s.free();
            s.ctx = new_cache_context(16);
            auto mk = [&]() { return ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, ne0, ne1, ne2); };
            s.prev_probe = mk();
            s.prev_input = mk();
            s.probe_prev1 = mk();
            s.probe_prev2 = mk();
            s.resid_prev1 = mk();
            s.resid_prev2 = mk();
            s.buffer = ggml_backend_alloc_ctx_tensors(s.ctx, runtime_backend);
            GGML_ASSERT(s.buffer != nullptr);
            s.shape = want;
            s.has_probe_hist = false;
            s.probe_resid_count = 0;
            s.resid_count = 0;
            return s;
        }

        void reset_dicache_gpu_states() {
            for (auto& kv : dicache_gpu_states_) {
                kv.second.free();
            }
            dicache_gpu_states_.clear();
        }

        // Ensure per-branch persistent state sized to the just-computed named cache
        // tensors (feature/probe/before all share the block-stack shape).
        DiCacheGpuState& ensure_dicache_gpu_state_from_named(const void* branch_key) {
            ggml_tensor* feat = get_cache_tensor_by_name(kCacheFeatureName);
            if (feat == nullptr) {
                return dicache_gpu_states_[branch_key];  // empty; caller checks buffer
            }
            return ensure_dicache_gpu_state(branch_key, feat->ne[0], feat->ne[1], feat->ne[2]);
        }

        QwenImageRunner(ggml_backend_t backend,
                        bool offload_params_to_cpu,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "",
                        SDVersion version                              = VERSION_QWEN_IMAGE,
                        bool zero_cond_t                               = false)
            : GGMLRunner(backend, offload_params_to_cpu) {
            qwen_image_params.num_layers  = 0;
            qwen_image_params.zero_cond_t = zero_cond_t;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                if (tensor_name.find("__index_timestep_zero__") != std::string::npos) {
                    qwen_image_params.zero_cond_t = true;
                }
                size_t pos = tensor_name.find("transformer_blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > qwen_image_params.num_layers) {
                            qwen_image_params.num_layers = block_index + 1;
                        }
                    }
                    continue;
                }
            }
            LOG_INFO("qwen_image_params.num_layers: %ld", qwen_image_params.num_layers);
            if (qwen_image_params.zero_cond_t) {
                LOG_INFO("use zero_cond_t");
            }
            qwen_image = QwenImageModel(qwen_image_params);
            qwen_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "qwen_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            qwen_image.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false) {
            ggml_cgraph* gf        = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);
            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            pe_vec      = Rope::gen_qwen_image_pe(static_cast<int>(x->ne[1]),
                                                  static_cast<int>(x->ne[0]),
                                                  qwen_image_params.patch_size,
                                                  static_cast<int>(x->ne[3]),
                                                  static_cast<int>(context->ne[1]),
                                                  ref_latents,
                                                  increase_ref_index,
                                                  qwen_image_params.theta,
                                                  circular_y_enabled,
                                                  circular_x_enabled,
                                                  qwen_image_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / qwen_image_params.axes_dim_sum / 2);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, qwen_image_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            ggml_tensor* modulate_index = nullptr;
            if (qwen_image_params.zero_cond_t) {
                modulate_index_vec.clear();

                int64_t h_len          = ((x->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t w_len          = ((x->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t num_img_tokens = h_len * w_len;

                modulate_index_vec.insert(modulate_index_vec.end(), num_img_tokens, 0.f);
                int64_t num_ref_img_tokens = 0;
                for (ggml_tensor* ref : ref_latents) {
                    int64_t h_len = ((ref->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                    int64_t w_len = ((ref->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);

                    num_ref_img_tokens += h_len * w_len;
                }

                if (num_ref_img_tokens > 0) {
                    modulate_index_vec.insert(modulate_index_vec.end(), num_ref_img_tokens, 1.f);
                }

                modulate_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, modulate_index_vec.size());
                set_backend_tensor_data(modulate_index, modulate_index_vec.data());
            }

            auto runner_ctx = get_context();

            ggml_tensor* out = qwen_image.forward(&runner_ctx,
                                                  x,
                                                  timesteps,
                                                  context,
                                                  pe,
                                                  ref_latents,
                                                  modulate_index);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };

            // Experimental (ED_CACHE_COMPILED_GRAPHS): build the forward graph once
            // and reuse it across sampling steps, refreshing only the input bytes.
            // Gated to the plain non-segmented path. build_graph()'s make_input order
            // is {x, timesteps, context, ref_latents...}; pe / modulate_index are
            // set_backend_tensor_data leaves (shape-constant, timestep-independent),
            // so they stay valid on a reused graph and are not part of ordered_inputs.
            const bool reuse_graphs =
                qwen_env_flag_enabled_or_default("ED_CACHE_COMPILED_GRAPHS", false) &&
                !can_attempt_graph_cut_segmented_compute();
            if (reuse_graphs) {
                std::vector<const sd::Tensor<float>*> ordered_inputs;
                ordered_inputs.push_back(&x);
                ordered_inputs.push_back(&timesteps);
                ordered_inputs.push_back(&context);
                for (const auto& ref : ref_latents) {
                    ordered_inputs.push_back(&ref);
                }
                return restore_trailing_singleton_dims(
                    GGMLRunner::compute_reuse<float>(get_graph, ordered_inputs, n_threads), x.dim());
            }

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        // ---- Cache seam passes (Feature/Probe policies). ----
        sd::DiffusionCacheResult compute_capture(int n_threads,
                                             const sd::Tensor<float>& x,
                                             const sd::Tensor<float>& timesteps,
                                             const sd::Tensor<float>& context,
                                             const std::vector<sd::Tensor<float>>& ref_latents,
                                             bool increase_ref_index,
                                             int region_start = 0,
                                             int region_end = -1,
                                             const void* branch_key = nullptr) {
            sd::CacheGraphScope scope;
            scope.mode = sd::CacheGraphScope::Mode::Capture;
            scope.region_start = region_start;
            scope.region_end = region_end;
            const bool gpu = dicache_gpu_enabled() && branch_key != nullptr;
            if (gpu) {
                scope.gpu_metric = true;         // record probe-depth state during capture
                scope.probe_depth = std::max(1, dicache_probe_depth_);
            }
            std::function<void()> handoff = nullptr;
            if (gpu) {
                handoff = [&]() {
                    // Refresh the persistent cross-step state device-to-device from
                    // the just-computed named tensors, with a 2-deep ping-pong ring.
                    DiCacheGpuState& s = ensure_dicache_gpu_state_from_named(branch_key);
                    if (s.buffer == nullptr) return;
                    // full residual ring: prev2 <- prev1, prev1 <- feature
                    std::swap(s.resid_prev1, s.resid_prev2);
                    copy_named_cache_tensor_to(kCacheFeatureName, s.resid_prev1);
                    s.resid_count = std::min(2, s.resid_count + 1);
                    // probe residual ring: prev2 <- prev1, prev1 <- (probe - before)
                    std::swap(s.probe_prev1, s.probe_prev2);
                    copy_named_cache_tensor_to(kCacheProbeResidName, s.probe_prev1);
                    s.probe_resid_count = std::min(2, s.probe_resid_count + 1);
                    // raw probe & input for the delta_y / delta_x decision metrics
                    copy_named_cache_tensor_to(kCacheProbeName, s.prev_probe);
                    copy_named_cache_tensor_to(kCacheBeforeName, s.prev_input);
                    s.has_probe_hist = true;
                };
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_cache_pass(get_graph, n_threads, &scope, x.dim(), handoff);
            sd::DiffusionCacheResult out;
            out.output = std::move(pass.output);
            out.feature = std::move(pass.feature);
            return out;
        }

        sd::Tensor<float> compute_inject(int n_threads,
                                         const sd::Tensor<float>& x,
                                         const sd::Tensor<float>& timesteps,
                                         const sd::Tensor<float>& context,
                                         const std::vector<sd::Tensor<float>>& ref_latents,
                                         bool increase_ref_index,
                                         const sd::Tensor<float>& feature,
                                         int region_start = 0,
                                         int region_end = -1) {
            sd::CacheGraphScope scope;
            scope.mode = sd::CacheGraphScope::Mode::Inject;
            scope.region_start = region_start;
            scope.region_end = region_end;
            inject_feature_host_ = feature;
            auto get_graph = [&]() -> ggml_cgraph* {
                scope.inject_feature = make_input(inject_feature_host_);
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_cache_pass(get_graph, n_threads, &scope, x.dim());
            return std::move(pass.output);
        }

        // GPU DiCache reuse step: reconstruct the residual on-device from the
        // persistent full-residual ring and a host-clamped gamma, injecting
        //   x_before + resid_prev2 + gamma*(resid_prev1 - resid_prev2)
        // No ~50MB host residual is built or uploaded. Returns empty if state is
        // not ready (caller falls back to a capturing full step).
        sd::Tensor<float> compute_inject_gpu(int n_threads,
                                             const sd::Tensor<float>& x,
                                             const sd::Tensor<float>& timesteps,
                                             const sd::Tensor<float>& context,
                                             const std::vector<sd::Tensor<float>>& ref_latents,
                                             bool increase_ref_index,
                                             float gamma,
                                             const void* branch_key,
                                             int region_start = 0,
                                             int region_end = -1) {
            auto it = dicache_gpu_states_.find(branch_key);
            if (it == dicache_gpu_states_.end() || it->second.buffer == nullptr ||
                it->second.resid_count < 2) {
                return {};  // not enough history yet
            }
            DiCacheGpuState& s = it->second;
            sd::CacheGraphScope scope;
            scope.mode = sd::CacheGraphScope::Mode::Inject;
            scope.region_start = region_start;
            scope.region_end = region_end;
            scope.resid_prev1_node = s.resid_prev1;
            scope.resid_prev2_node = s.resid_prev2;
            gamma_scalar_host_ = sd::Tensor<float>({1}, std::vector<float>{gamma});
            auto get_graph = [&]() -> ggml_cgraph* {
                scope.gamma_scalar = make_input(gamma_scalar_host_);
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_cache_pass(get_graph, n_threads, &scope, x.dim());
            return std::move(pass.output);
        }

        // ---- Feature-granularity on-GPU reuse (MagCache / TaylorSeer-single) ----
        // When enabled, the last captured block-stack residual stays resident
        // on-device and is injected as x_before + residual on skips, avoiding the
        // ~50MB host reconstruct copy + H2D upload the plain inject path pays per
        // skip. The residual is stored in a CacheStateManager device slot; see
        // compute_capture_to_slot / compute_inject_from_slot below.
        static bool feature_gpu_enabled() {
            const char* v = std::getenv("ED_FEATURE_CACHE_GPU");
            if (v == nullptr || v[0] == '\0') {
                return true;  // default on
            }
            return v[0] != '0';
        }

        // ---- Declarative device-slot seam (B2): same on-device single-residual
        // reuse as compute_*_feature_gpu, but the persistent residual tensor is a
        // CacheStateManager device slot (passed in) instead of DiCacheGpuState.
        // capture copies the block-stack residual into `slot`; inject reads it. ----
        sd::Tensor<float> compute_capture_to_slot(int n_threads,
                                             const sd::Tensor<float>& x,
                                             const sd::Tensor<float>& timesteps,
                                             const sd::Tensor<float>& context,
                                             const std::vector<sd::Tensor<float>>& ref_latents,
                                             bool increase_ref_index,
                                             const std::function<void*(const std::vector<int64_t>&)>& alloc_slot,
                                             int region_start,
                                             int region_end) {
            sd::CacheGraphScope scope;
            scope.mode = sd::CacheGraphScope::Mode::Capture;
            scope.region_start = region_start;
            scope.region_end = region_end;
            std::function<void()> handoff = [&]() {
                // The residual named tensor is resident; allocate the device slot at
                // its true (packed block-stack seq) shape, then d2d-copy into it.
                ggml_tensor* feat = get_cache_tensor_by_name(kCacheFeatureName);
                if (feat == nullptr) {
                    return;
                }
                // Copy feat's exact dims so the slot is byte- and broadcast-identical
                // (add_injected does ggml_add(x_before, slot)). ggml_n_dims collapses
                // trailing singletons but never a size-1 inner dim.
                std::vector<int64_t> shape;
                const int nd = std::max(1, ggml_n_dims(feat));
                for (int i = 0; i < nd; ++i) {
                    shape.push_back(feat->ne[i]);
                }
                ggml_tensor* slot = static_cast<ggml_tensor*>(alloc_slot(shape));
                if (slot != nullptr && ggml_nbytes(slot) == ggml_nbytes(feat)) {
                    copy_named_cache_tensor_to(kCacheFeatureName, slot);
                }
            };
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_cache_pass(get_graph, n_threads, &scope, x.dim(), handoff);
            return std::move(pass.output);
        }

        sd::Tensor<float> compute_inject_from_slot(int n_threads,
                                             const sd::Tensor<float>& x,
                                             const sd::Tensor<float>& timesteps,
                                             const sd::Tensor<float>& context,
                                             const std::vector<sd::Tensor<float>>& ref_latents,
                                             bool increase_ref_index,
                                             ggml_tensor* slot,
                                             int region_start,
                                             int region_end) {
            if (slot == nullptr) {
                return {};
            }
            sd::CacheGraphScope scope;
            scope.mode = sd::CacheGraphScope::Mode::Inject;
            scope.region_start = region_start;
            scope.region_end = region_end;
            scope.resid_prev1_node = slot;  // single residual: gpu_reuse_available()
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_cache_pass(get_graph, n_threads, &scope, x.dim());
            return std::move(pass.output);
        }

        // ---- Substep-path (ED_CACHE_SUBSTEP) tap-driven capture. Replaces the
        // capture_to_slot hook: sets a TapRegistry requesting ModelIn/ModelOut,
        // asks build_graph to weave the (ModelOut-ModelIn) residual, runs the pass,
        // then d2d-copies the residual into the device slot. No CacheGraphScope,
        // no kCache*Name hook plumbing beyond the shared feature node name. ----
        sd::Tensor<float> compute_substep_capture(int n_threads,
                                                  const sd::Tensor<float>& x,
                                                  const sd::Tensor<float>& timesteps,
                                                  const sd::Tensor<float>& context,
                                                  const std::vector<sd::Tensor<float>>& ref_latents,
                                                  bool increase_ref_index,
                                                  const std::function<void*(const std::vector<int64_t>&)>& alloc_slot) {
            edgedit::cache::TapRegistry reg;
            reg.set_requested({edgedit::cache::AnchorRef::model_in(),
                               edgedit::cache::AnchorRef::model_out()});
            reg.set_capture_residual(true);
            std::function<void()> handoff = [&]() {
                ggml_tensor* feat = get_cache_tensor_by_name("ed_cache_feature");
                if (feat == nullptr) {
                    return;
                }
                std::vector<int64_t> shape;
                const int nd = std::max(1, ggml_n_dims(feat));
                for (int i = 0; i < nd; ++i) {
                    shape.push_back(feat->ne[i]);
                }
                ggml_tensor* slot = static_cast<ggml_tensor*>(alloc_slot(shape));
                if (slot != nullptr && ggml_nbytes(slot) == ggml_nbytes(feat)) {
                    copy_named_cache_tensor_to("ed_cache_feature", slot);
                }
            };
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {}, handoff);
            return std::move(pass.output);
        }

        sd::DiffusionCacheResult compute_probe(int n_threads,
                                           const sd::Tensor<float>& x,
                                           const sd::Tensor<float>& timesteps,
                                           const sd::Tensor<float>& context,
                                           const std::vector<sd::Tensor<float>>& ref_latents,
                                           bool increase_ref_index,
                                           int probe_depth,
                                           const void* branch_key = nullptr) {
            sd::CacheGraphScope scope;
            scope.mode = sd::CacheGraphScope::Mode::Probe;
            scope.probe_depth = probe_depth;
            // GPU DiCache path: attach the persistent prev_* tensors so
            // build_probe_metrics emits delta_y/delta_x/gamma reductions in-graph;
            // only scalars are read back.
            if (dicache_gpu_enabled() && branch_key != nullptr) {
                auto it = dicache_gpu_states_.find(branch_key);
                if (it != dicache_gpu_states_.end() && it->second.has_probe_hist) {
                    DiCacheGpuState& gpu = it->second;
                    scope.gpu_metric = true;
                    scope.prev_probe_node = gpu.prev_probe;
                    scope.prev_input_node = gpu.prev_input;
                    if (gpu.probe_resid_count >= 2) {
                        scope.probe_prev1_node = gpu.probe_prev1;
                        scope.probe_prev2_node = gpu.probe_prev2;
                    }
                }
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };
            auto pass = run_cache_pass(get_graph, n_threads, &scope, x.dim());
            sd::DiffusionCacheResult out;
            if (scope.gpu_metric) {
                // GPU path: no ~50MB host tensors; only scalars.
                out.delta_y = pass.delta_y;
                out.delta_x = pass.delta_x;
                out.gamma = pass.gamma;
                return out;
            }
            out.probe = pass.probe.empty() ? std::move(pass.output) : std::move(pass.probe);
            out.before = std::move(pass.before);
            return out;
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = sd::load_tensor_from_file_as_tensor<float>("./qwen_image_x.bin");
                print_sd_tensor(x);

                std::vector<float> timesteps_vec(1, 1000.f);
                auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);

                // auto context = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3584, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = sd::load_tensor_from_file_as_tensor<float>("./qwen_image_context.bin");
                print_sd_tensor(context);

                sd::Tensor<float> out;

                auto out_opt = compute(8,
                                       x,
                                       timesteps,
                                       context,
                                       {},
                                       false);

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend    = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            for (auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "weight")) {
                    tensor_storage.expected_type = model_data_type;
                }
            }

            std::shared_ptr<QwenImageRunner> qwen_image = std::make_shared<QwenImageRunner>(backend,
                                                                                            false,
                                                                                            tensor_storage_map,
                                                                                            "model.diffusion_model",
                                                                                            VERSION_QWEN_IMAGE);

            qwen_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            qwen_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("qwen_image model loaded");
            qwen_image->test();
        }
    };

}  // namespace name

#endif  // __QWEN_IMAGE_HPP__
