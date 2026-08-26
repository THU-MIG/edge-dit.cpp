#ifndef __WAN_HPP__
#define __WAN_HPP__

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "dit_models/components/common/common_block.hpp"
#include "dit_models/components/autoencoders/vae.hpp"
#include "dit_models/components/common/rope.hpp"
#include "backend/ggml/ed_ggml_attention_ext.hpp"
#include "backend/ggml/ed_ggml_modulation_ext.hpp"
#include "backend/ggml/ed_ggml_norm_ext.hpp"
#include "parallel/sp_parallel.hpp"

namespace WAN {

    constexpr int CACHE_T        = 2;
    constexpr int WAN_GRAPH_SIZE = 32768;

    static inline bool wan_env_flag_enabled_or_default(const char* name, bool default_enabled) {
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

    static inline bool wan_sp_f16_q_enabled() {
        return wan_env_flag_enabled_or_default("ED_WAN_SP_F16_Q", true);
    }

    static inline bool wan_sp_f16_head_to_seq_enabled() {
        return wan_env_flag_enabled_or_default("ED_WAN_SP_F16_HEAD_TO_SEQ", true);
    }

    static inline bool wan_sp_f16_cross_attn_enabled() {
        return wan_env_flag_enabled_or_default("ED_WAN_SP_F16_CROSS_ATTN", true);
    }

    static inline bool wan_sp_local_head_before_gather_enabled() {
        return wan_env_flag_enabled_or_default("ED_WAN_SP_LOCAL_HEAD_BEFORE_GATHER", true);
    }

    static inline bool wan_sp_attention_pg_comm_enabled() {
        return wan_env_flag_enabled_or_default("ED_WAN_SP_ATTENTION_PG_COMM", true);
    }

    // Reuse the existing ggml-cuda fused-qkv custom-op ABI. Mode 6 packs
    // q/k/v directly into the seq-to-head all-to-all send-flat layout.
    struct WanFusedQKVPackParams {
        uint64_t magic;
        int64_t world_size;
        int64_t unused_seq;
        int64_t mode;
        int64_t unused0;
        int64_t unused1;
        int64_t unused2;
        int64_t unused3;
    };

    constexpr uint64_t WAN_FUSED_QKV_PACK_MAGIC = 0x5157454e46514b56ULL;

    static std::mutex& wan_fused_qkv_pack_params_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::vector<std::unique_ptr<WanFusedQKVPackParams>>& wan_fused_qkv_pack_params_store() {
        static std::vector<std::unique_ptr<WanFusedQKVPackParams>> store;
        return store;
    }

    static WanFusedQKVPackParams* wan_fused_qkv_pack_make_params(int64_t world_size,
                                                                 int64_t mode = 6,
                                                                 int64_t stream_index = 0,
                                                                 int64_t aux0 = 0,
                                                                 int64_t aux1 = 0,
                                                                 int64_t aux2 = 0,
                                                                 int64_t aux3 = 0) {
        auto params        = std::make_unique<WanFusedQKVPackParams>();
        params->magic      = WAN_FUSED_QKV_PACK_MAGIC;
        params->world_size = world_size;
        params->unused_seq = aux0;
        params->mode       = mode;
        params->unused0    = aux1;
        params->unused1    = aux2;
        params->unused2    = aux3;
        params->unused3    = stream_index;
        WanFusedQKVPackParams* raw = params.get();
        std::lock_guard<std::mutex> lock(wan_fused_qkv_pack_params_mutex());
        wan_fused_qkv_pack_params_store().push_back(std::move(params));
        return raw;
    }

    static inline float wan_fused_qkv_get_f32(const ggml_tensor* src,
                                              int64_t i0,
                                              int64_t i1,
                                              int64_t i2,
                                              int64_t i3) {
        const char* data = static_cast<const char*>(src->data) +
                           i0 * src->nb[0] +
                           i1 * src->nb[1] +
                           i2 * src->nb[2] +
                           i3 * src->nb[3];
        return *reinterpret_cast<const float*>(data);
    }

    static inline float wan_fused_qkv_get_f32_or_f16(const ggml_tensor* src,
                                                     int64_t i0,
                                                     int64_t i1,
                                                     int64_t i2,
                                                     int64_t i3) {
        const char* data = static_cast<const char*>(src->data) +
                           i0 * src->nb[0] +
                           i1 * src->nb[1] +
                           i2 * src->nb[2] +
                           i3 * src->nb[3];
        if (src->type == GGML_TYPE_F16) {
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data));
        }
        GGML_ASSERT(src->type == GGML_TYPE_F32);
        return *reinterpret_cast<const float*>(data);
    }

    static inline void wan_fused_qkv_set_f32(ggml_tensor* dst, int64_t linear, float value) {
        char* data = static_cast<char*>(dst->data) + linear * dst->nb[0];
        *reinterpret_cast<float*>(data) = value;
    }

    static inline void wan_fused_qkv_set_f16(ggml_tensor* dst, int64_t linear, float value) {
        char* data = static_cast<char*>(dst->data) + linear * dst->nb[0];
        *reinterpret_cast<ggml_fp16_t*>(data) = ggml_fp32_to_fp16(value);
    }

    static inline void wan_fused_qkv_set_f16_bits(ggml_tensor* dst, int64_t linear, uint32_t bits) {
        char* data = static_cast<char*>(dst->data) + linear * dst->nb[0];
        *reinterpret_cast<ggml_fp16_t*>(data) = static_cast<ggml_fp16_t>(bits & 0xffffu);
    }

    static inline void wan_fused_qkv_set_u32(ggml_tensor* dst, int64_t linear, uint32_t value) {
        char* data = static_cast<char*>(dst->data) + linear * dst->nb[0];
        *reinterpret_cast<uint32_t*>(data) = value;
    }

    static inline uint32_t wan_fused_qkv_get_u32(const ggml_tensor* src, int64_t linear) {
        const char* data = static_cast<const char*>(src->data) + linear * src->nb[0];
        return *reinterpret_cast<const uint32_t*>(data);
    }

    static inline uint32_t wan_fused_qkv_f32_bits(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    static inline float wan_fused_qkv_apply_rope_value(const ggml_tensor* x,
                                                       const ggml_tensor* pe,
                                                       int64_t d,
                                                       int64_t head,
                                                       int64_t local_seq,
                                                       int64_t global_seq) {
        const int64_t part = d & 1;
        const int64_t half = d >> 1;
        const float x0     = wan_fused_qkv_get_f32(x, 2 * half, head, local_seq, 0);
        const float x1     = wan_fused_qkv_get_f32(x, 2 * half + 1, head, local_seq, 0);
        const float pe0    = wan_fused_qkv_get_f32(pe, part, half, global_seq, 0);
        const float pe1    = wan_fused_qkv_get_f32(pe, part, half, global_seq, 1);
        return x0 * pe0 + x1 * pe1;
    }

    static inline void wan_fused_qkv_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 6);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        const int64_t world_size     = params->world_size;
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
            int64_t rem          = linear;
            const int64_t d_all  = rem % total_head_dim;
            rem /= total_head_dim;
            const int64_t h_local = rem % shard_heads;
            rem /= shard_heads;
            const int64_t seq = rem % shard_sequence;
            rem /= shard_sequence;
            const int64_t peer = rem;
            const int64_t head = h_local + peer * shard_heads;
            const int64_t plane = d_all / head_dim;
            const int64_t d     = d_all - plane * head_dim;
            const ggml_tensor* src = plane == 0 ? q : (plane == 1 ? k : v);
            wan_fused_qkv_set_f32(dst, linear, wan_fused_qkv_get_f32(src, d, head, seq, 0));
        }
    }

    static inline void wan_fused_qkv_vhalf_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 33);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        const int64_t world_size     = params->world_size;
        const int64_t head_dim       = q->ne[0];
        const int64_t heads          = q->ne[1];
        const int64_t shard_sequence = q->ne[2];
        const int64_t shard_heads    = heads / world_size;
        const int64_t packed_dim     = head_dim * 2 + head_dim / 2;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(heads % world_size == 0);
        GGML_ASSERT(k->ne[0] == head_dim && v->ne[0] == head_dim);
        GGML_ASSERT(k->ne[1] == heads && v->ne[1] == heads);
        GGML_ASSERT(k->ne[2] == shard_sequence && v->ne[2] == shard_sequence);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        GGML_ASSERT(dst->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

        for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
            int64_t rem         = linear;
            const int64_t d_all = rem % packed_dim;
            rem /= packed_dim;
            const int64_t h_local = rem % shard_heads;
            rem /= shard_heads;
            const int64_t seq = rem % shard_sequence;
            rem /= shard_sequence;
            const int64_t peer = rem;
            const int64_t head = h_local + peer * shard_heads;

            if (d_all < head_dim) {
                const float value = wan_fused_qkv_get_f32(q, d_all, head, seq, 0);
                wan_fused_qkv_set_u32(dst, linear, wan_fused_qkv_f32_bits(value));
            } else if (d_all < head_dim * 2) {
                const int64_t d = d_all - head_dim;
                const float value = wan_fused_qkv_get_f32(k, d, head, seq, 0);
                wan_fused_qkv_set_u32(dst, linear, wan_fused_qkv_f32_bits(value));
            } else {
                const int64_t half = d_all - head_dim * 2;
                const int64_t d0   = half * 2;
                const int64_t d1   = d0 + 1;
                const uint32_t v0  = static_cast<uint32_t>(ggml_fp32_to_fp16(wan_fused_qkv_get_f32(v, d0, head, seq, 0)));
                const uint32_t v1  = static_cast<uint32_t>(ggml_fp32_to_fp16(wan_fused_qkv_get_f32(v, d1, head, seq, 0)));
                wan_fused_qkv_set_u32(dst, linear, v0 | (v1 << 16));
            }
        }
    }

    static inline void wan_fused_qkv_roped_half_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 37);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* q  = dst->src[0];
        const ggml_tensor* k  = dst->src[1];
        const ggml_tensor* v  = dst->src[2];
        const ggml_tensor* pe = dst->src[3];
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr && pe != nullptr);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(pe->type == GGML_TYPE_F32);
        const int64_t world_size     = params->world_size;
        const int64_t rank           = params->unused_seq;
        const int64_t head_dim       = q->ne[0];
        const int64_t half_dim       = head_dim / 2;
        const int64_t heads          = q->ne[1];
        const int64_t shard_sequence = q->ne[2];
        const int64_t shard_heads    = heads / world_size;
        const int64_t packed_dim     = head_dim * 2;
        GGML_ASSERT(world_size > 0 && rank >= 0 && rank < world_size);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(heads % world_size == 0);
        GGML_ASSERT(k->ne[0] == head_dim && v->ne[0] == head_dim);
        GGML_ASSERT(k->ne[1] == heads && v->ne[1] == heads);
        GGML_ASSERT(k->ne[2] == shard_sequence && v->ne[2] == shard_sequence);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        GGML_ASSERT(pe->ne[0] == 2 && pe->ne[1] == half_dim);
        GGML_ASSERT(pe->ne[2] >= shard_sequence * world_size && pe->ne[3] == 2);
        GGML_ASSERT(dst->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

        for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
            int64_t rem         = linear;
            const int64_t d_all = rem % packed_dim;
            rem /= packed_dim;
            const int64_t h_local = rem % shard_heads;
            rem /= shard_heads;
            const int64_t seq = rem % shard_sequence;
            rem /= shard_sequence;
            const int64_t peer       = rem;
            const int64_t head       = h_local + peer * shard_heads;
            const int64_t global_seq = rank * shard_sequence + seq;

            if (d_all < head_dim) {
                const float value = wan_fused_qkv_apply_rope_value(q, pe, d_all, head, seq, global_seq);
                wan_fused_qkv_set_u32(dst, linear, wan_fused_qkv_f32_bits(value));
            } else if (d_all < head_dim + half_dim) {
                const int64_t half = d_all - head_dim;
                const int64_t d0   = half * 2;
                const int64_t d1   = d0 + 1;
                const uint32_t k0  = static_cast<uint32_t>(ggml_fp32_to_fp16(
                    wan_fused_qkv_apply_rope_value(k, pe, d0, head, seq, global_seq)));
                const uint32_t k1  = static_cast<uint32_t>(ggml_fp32_to_fp16(
                    wan_fused_qkv_apply_rope_value(k, pe, d1, head, seq, global_seq)));
                wan_fused_qkv_set_u32(dst, linear, k0 | (k1 << 16));
            } else {
                const int64_t half = d_all - head_dim - half_dim;
                const int64_t d0   = half * 2;
                const int64_t d1   = d0 + 1;
                const uint32_t v0  = static_cast<uint32_t>(ggml_fp32_to_fp16(wan_fused_qkv_get_f32(v, d0, head, seq, 0)));
                const uint32_t v1  = static_cast<uint32_t>(ggml_fp32_to_fp16(wan_fused_qkv_get_f32(v, d1, head, seq, 0)));
                wan_fused_qkv_set_u32(dst, linear, v0 | (v1 << 16));
            }
        }
    }

    static inline void wan_fused_qkv_roped_all_half_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 56);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* q  = dst->src[0];
        const ggml_tensor* k  = dst->src[1];
        const ggml_tensor* v  = dst->src[2];
        const ggml_tensor* pe = dst->src[3];
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr && pe != nullptr);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(pe->type == GGML_TYPE_F32);
        const int64_t world_size     = params->world_size;
        const int64_t rank           = params->unused_seq;
        const int64_t head_dim       = q->ne[0];
        const int64_t half_dim       = head_dim / 2;
        const int64_t heads          = q->ne[1];
        const int64_t shard_sequence = q->ne[2];
        const int64_t shard_heads    = heads / world_size;
        const int64_t packed_dim     = half_dim * 3;
        GGML_ASSERT(world_size > 0 && rank >= 0 && rank < world_size);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(heads % world_size == 0);
        GGML_ASSERT(k->ne[0] == head_dim && v->ne[0] == head_dim);
        GGML_ASSERT(k->ne[1] == heads && v->ne[1] == heads);
        GGML_ASSERT(k->ne[2] == shard_sequence && v->ne[2] == shard_sequence);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        GGML_ASSERT(pe->ne[0] == 2 && pe->ne[1] == half_dim);
        GGML_ASSERT(pe->ne[2] >= shard_sequence * world_size && pe->ne[3] == 2);
        GGML_ASSERT(dst->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

        for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
            int64_t rem         = linear;
            const int64_t d_all = rem % packed_dim;
            rem /= packed_dim;
            const int64_t h_local = rem % shard_heads;
            rem /= shard_heads;
            const int64_t seq = rem % shard_sequence;
            rem /= shard_sequence;
            const int64_t peer       = rem;
            const int64_t head       = h_local + peer * shard_heads;
            const int64_t global_seq = rank * shard_sequence + seq;

            const int64_t plane = d_all / half_dim;
            const int64_t half  = d_all - plane * half_dim;
            const int64_t d0    = half * 2;
            const int64_t d1    = d0 + 1;
            uint32_t v0;
            uint32_t v1;
            if (plane == 0) {
                v0 = static_cast<uint32_t>(ggml_fp32_to_fp16(
                    wan_fused_qkv_apply_rope_value(q, pe, d0, head, seq, global_seq)));
                v1 = static_cast<uint32_t>(ggml_fp32_to_fp16(
                    wan_fused_qkv_apply_rope_value(q, pe, d1, head, seq, global_seq)));
            } else if (plane == 1) {
                v0 = static_cast<uint32_t>(ggml_fp32_to_fp16(
                    wan_fused_qkv_apply_rope_value(k, pe, d0, head, seq, global_seq)));
                v1 = static_cast<uint32_t>(ggml_fp32_to_fp16(
                    wan_fused_qkv_apply_rope_value(k, pe, d1, head, seq, global_seq)));
            } else {
                v0 = static_cast<uint32_t>(ggml_fp32_to_fp16(wan_fused_qkv_get_f32(v, d0, head, seq, 0)));
                v1 = static_cast<uint32_t>(ggml_fp32_to_fp16(wan_fused_qkv_get_f32(v, d1, head, seq, 0)));
            }
            wan_fused_qkv_set_u32(dst, linear, v0 | (v1 << 16));
        }
    }

    static inline void wan_fused_qkv_roped_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 38 || params->mode == 39 || params->mode == 40);
        GGML_ASSERT(params->mode == 38 ? dst->type == GGML_TYPE_F32 : dst->type == GGML_TYPE_F16);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        GGML_ASSERT(dst->ne[3] == 1);
        const int64_t head_dim       = dst->ne[0];
        const int64_t half_dim       = head_dim / 2;
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t shard_heads    = dst->ne[2];
        const int64_t packed_dim     = head_dim * 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_sequence > 0 && shard_heads > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
            int64_t rem         = linear;
            const int64_t d     = rem % head_dim;
            rem /= head_dim;
            const int64_t seq   = rem % sequence;
            rem /= sequence;
            const int64_t head  = rem % shard_heads;
            const int64_t peer  = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t base = head * packed_dim +
                                 local_seq * packed_dim * shard_heads +
                                 peer * packed_dim * shard_heads * shard_sequence;

            if (params->mode == 38) {
                wan_fused_qkv_set_u32(dst, linear, wan_fused_qkv_get_u32(recv_flat, base + d));
            } else {
                const int64_t half_plane = params->mode == 39 ? 0 : 1;
                const int64_t src_idx    = base + head_dim + half_plane * half_dim + d / 2;
                const uint32_t packed    = wan_fused_qkv_get_u32(recv_flat, src_idx);
                const uint32_t bits      = (d & 1) == 0 ? (packed & 0xffffu) : (packed >> 16);
                wan_fused_qkv_set_f16_bits(dst, linear, bits);
            }
        }
    }

    static inline void wan_fused_roped_kv_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 41);
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        GGML_ASSERT(dst->ne[3] == 2);
        const int64_t head_dim       = dst->ne[0];
        const int64_t half_dim       = head_dim / 2;
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t shard_heads    = dst->ne[2];
        const int64_t packed_dim     = head_dim * 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_sequence > 0 && shard_heads > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
            int64_t rem         = linear;
            const int64_t d     = rem % head_dim;
            rem /= head_dim;
            const int64_t seq   = rem % sequence;
            rem /= sequence;
            const int64_t head  = rem % shard_heads;
            rem /= shard_heads;
            const int64_t plane = rem;
            const int64_t peer  = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t base = head * packed_dim +
                                 local_seq * packed_dim * shard_heads +
                                 peer * packed_dim * shard_heads * shard_sequence;
            const int64_t src_idx = base + head_dim + plane * half_dim + d / 2;
            const uint32_t packed = wan_fused_qkv_get_u32(recv_flat, src_idx);
            const uint32_t bits   = (d & 1) == 0 ? (packed & 0xffffu) : (packed >> 16);
            wan_fused_qkv_set_f16_bits(dst, linear, bits);
        }
    }

    static inline void wan_fused_roped_all_half_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 57 || params->mode == 58);
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        GGML_ASSERT(params->mode == 57 ? dst->ne[3] == 1 : dst->ne[3] == 2);
        const int64_t head_dim       = dst->ne[0];
        const int64_t half_dim       = head_dim / 2;
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t shard_heads    = dst->ne[2];
        const int64_t packed_dim     = half_dim * 3;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_sequence > 0 && shard_heads > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        const int64_t first_plane = params->mode == 57 ? 0 : 1;
        for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
            int64_t rem         = linear;
            const int64_t d     = rem % head_dim;
            rem /= head_dim;
            const int64_t seq   = rem % sequence;
            rem /= sequence;
            const int64_t head  = rem % shard_heads;
            rem /= shard_heads;
            const int64_t plane = first_plane + rem;
            const int64_t peer  = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t base = head * packed_dim +
                                 local_seq * packed_dim * shard_heads +
                                 peer * packed_dim * shard_heads * shard_sequence;
            const int64_t src_idx = base + plane * half_dim + d / 2;
            const uint32_t packed = wan_fused_qkv_get_u32(recv_flat, src_idx);
            const uint32_t bits   = (d & 1) == 0 ? (packed & 0xffffu) : (packed >> 16);
            wan_fused_qkv_set_f16_bits(dst, linear, bits);
        }
    }

    static inline void wan_fused_qkv_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 22 || params->mode == 23 || params->mode == 26);
        GGML_ASSERT(params->mode == 26 ? dst->type == GGML_TYPE_F16 : dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = params->world_size;
        const int64_t plane      = params->unused3;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t shard_heads    = dst->ne[2];
        const int64_t head_dim       = params->mode == 22 ? dst->ne[0] * 2 : dst->ne[0];
        const int64_t total_head_dim = head_dim * 3;
        GGML_ASSERT(shard_sequence > 0 && shard_heads > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == total_head_dim * shard_heads * shard_sequence * world_size);

        if (params->mode == 22) {
            GGML_ASSERT(plane == 0 || plane == 1);
            GGML_ASSERT(dst->ne[3] == 2);
            const int64_t half_dim = dst->ne[0];
            const int64_t total    = ggml_nelements(dst);
            for (int64_t linear = ith; linear < total; linear += nth) {
                int64_t rem          = linear;
                const int64_t half   = rem % half_dim;
                rem /= half_dim;
                const int64_t seq    = rem % sequence;
                rem /= sequence;
                const int64_t head   = rem % shard_heads;
                rem /= shard_heads;
                const int64_t part   = rem;
                const int64_t peer   = seq / shard_sequence;
                const int64_t local_seq = seq - peer * shard_sequence;
                const int64_t src_d  = plane * head_dim + part + 2 * half;
                const int64_t src_idx = src_d +
                                        head * total_head_dim +
                                        local_seq * total_head_dim * shard_heads +
                                        peer * total_head_dim * shard_heads * shard_sequence;
                wan_fused_qkv_set_f32(dst, linear, wan_fused_qkv_get_f32(recv_flat, src_idx, 0, 0, 0));
            }
            return;
        }

        GGML_ASSERT(params->mode == 23 || params->mode == 26);
        GGML_ASSERT(plane == 2);
        GGML_ASSERT(dst->ne[3] == 1);
        const bool output_f16 = params->mode == 26;
        const int64_t total = ggml_nelements(dst);
        for (int64_t linear = ith; linear < total; linear += nth) {
            int64_t rem         = linear;
            const int64_t d     = rem % head_dim;
            rem /= head_dim;
            const int64_t seq   = rem % sequence;
            rem /= sequence;
            const int64_t head  = rem % shard_heads;
            const int64_t peer  = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t src_d = 2 * head_dim + d;
            const int64_t src_idx = src_d +
                                    head * total_head_dim +
                                    local_seq * total_head_dim * shard_heads +
                                    peer * total_head_dim * shard_heads * shard_sequence;
            const float value = wan_fused_qkv_get_f32(recv_flat, src_idx, 0, 0, 0);
            if (output_f16) {
                wan_fused_qkv_set_f16(dst, linear, value);
            } else {
                wan_fused_qkv_set_f32(dst, linear, value);
            }
        }
    }

    static inline void wan_fused_attn_head_to_seq_send_pack_cpu(ggml_tensor* dst,
                                                                int ith,
                                                                int nth,
                                                                void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 24 || params->mode == 59);
        const bool pack_f16 = params->mode == 59;
        GGML_ASSERT(dst->type == (pack_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32));
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* attn = dst->src[0];
        GGML_ASSERT(attn != nullptr);
        GGML_ASSERT(attn->type == GGML_TYPE_F32 || (pack_f16 && attn->type == GGML_TYPE_F16));
        const int64_t world_size = params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(attn->ne[3] == 1);
        GGML_ASSERT(attn->ne[2] % world_size == 0);
        const int64_t head_dim       = attn->ne[0];
        const int64_t shard_heads    = attn->ne[1];
        const int64_t sequence       = attn->ne[2];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t count_per_peer = head_dim * shard_heads * shard_sequence;
        GGML_ASSERT(dst->ne[0] == count_per_peer * world_size);

        for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
            int64_t rem        = linear;
            const int64_t peer = rem / count_per_peer;
            rem -= peer * count_per_peer;
            const int64_t d = rem % head_dim;
            rem /= head_dim;
            const int64_t head = rem % shard_heads;
            rem /= shard_heads;
            const int64_t local_seq = rem;
            const int64_t seq       = peer * shard_sequence + local_seq;
            const float value = wan_fused_qkv_get_f32_or_f16(attn, d, head, seq, 0);
            if (pack_f16) {
                wan_fused_qkv_set_f16(dst, linear, value);
            } else {
                wan_fused_qkv_set_f32(dst, linear, value);
            }
        }
    }

    static inline void wan_fused_attn_head_to_seq_recv_unpack_cpu(ggml_tensor* dst,
                                                                  int ith,
                                                                  int nth,
                                                                  void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 25 || params->mode == 60);
        const bool unpack_f16 = params->mode == 60;
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == (unpack_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32));
        const int64_t world_size = params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        GGML_ASSERT(dst->ne[3] == 1);
        const int64_t head_dim       = dst->ne[0];
        const int64_t heads          = dst->ne[1];
        const int64_t shard_sequence = dst->ne[2];
        const int64_t shard_heads    = heads / world_size;
        const int64_t count_per_peer = head_dim * shard_heads * shard_sequence;
        GGML_ASSERT(heads % world_size == 0);
        GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

        for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
            int64_t rem    = linear;
            const int64_t d = rem % head_dim;
            rem /= head_dim;
            const int64_t head = rem % heads;
            rem /= heads;
            const int64_t local_seq  = rem;
            const int64_t src_peer   = head / shard_heads;
            const int64_t local_head = head - src_peer * shard_heads;
            const int64_t src_idx    = src_peer * count_per_peer +
                                    d +
                                    local_head * head_dim +
                                    local_seq * head_dim * shard_heads;
            wan_fused_qkv_set_f32(dst, linear, wan_fused_qkv_get_f32_or_f16(recv_flat, src_idx, 0, 0, 0));
        }
    }

    static inline void wan_sp_rope_custom_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 29 || params->mode == 30);
        GGML_ASSERT((params->mode == 29 && dst->type == GGML_TYPE_F16) ||
                    (params->mode == 30 && dst->type == GGML_TYPE_F32));
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* x  = dst->src[0];
        const ggml_tensor* pe = dst->src[1];
        GGML_ASSERT(x != nullptr && pe != nullptr);
        GGML_ASSERT(x->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);
        GGML_ASSERT(x->ne[3] == 2);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == x->ne[0]);
        GGML_ASSERT(pe->ne[2] >= x->ne[1]);
        GGML_ASSERT(pe->ne[3] == 2);
        GGML_ASSERT(dst->ne[0] == x->ne[0] * 2);
        GGML_ASSERT(dst->ne[1] == x->ne[1]);
        GGML_ASSERT(dst->ne[2] == x->ne[2]);
        GGML_ASSERT(dst->ne[3] == 1);

        const int64_t half_dim = x->ne[0];
        const int64_t sequence = x->ne[1];
        const int64_t heads    = x->ne[2];
        const int64_t head_dim = half_dim * 2;
        const int64_t total    = ggml_nelements(dst);

        for (int64_t linear = ith; linear < total; linear += nth) {
            int64_t rem        = linear;
            const int64_t d    = rem % head_dim;
            rem /= head_dim;
            const int64_t seq  = rem % sequence;
            rem /= sequence;
            const int64_t head = rem % heads;

            const int64_t part = d & 1;
            const int64_t half = d >> 1;
            const float x0     = wan_fused_qkv_get_f32(x, half, seq, head, 0);
            const float x1     = wan_fused_qkv_get_f32(x, half, seq, head, 1);
            const float pe0    = wan_fused_qkv_get_f32(pe, part, half, seq, 0);
            const float pe1    = wan_fused_qkv_get_f32(pe, part, half, seq, 1);
            const float value  = x0 * pe0 + x1 * pe1;
            if (dst->type == GGML_TYPE_F16) {
                wan_fused_qkv_set_f16(dst, linear, value);
            } else {
                wan_fused_qkv_set_f32(dst, linear, value);
            }
        }
    }

    static inline void wan_fused_qk_recv_rope_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 31 || params->mode == 32);
        GGML_ASSERT((params->mode == 31 && dst->type == GGML_TYPE_F32) ||
                    (params->mode == 32 && dst->type == GGML_TYPE_F16));
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        const ggml_tensor* pe        = dst->src[1];
        GGML_ASSERT(recv_flat != nullptr && pe != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);

        const int64_t world_size = params->world_size;
        const int64_t plane      = params->unused3;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);

        const int64_t head_dim       = dst->ne[0];
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_heads    = dst->ne[2];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t half_dim       = head_dim / 2;
        const int64_t total_head_dim = head_dim * 3;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == total_head_dim * shard_heads * shard_sequence * world_size);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == half_dim);
        GGML_ASSERT(pe->ne[2] >= sequence);
        GGML_ASSERT(pe->ne[3] == 2);

        const bool output_f16 = params->mode == 32;
        const int64_t total   = ggml_nelements(dst);
        for (int64_t linear = ith; linear < total; linear += nth) {
            int64_t rem        = linear;
            const int64_t d    = rem % head_dim;
            rem /= head_dim;
            const int64_t seq  = rem % sequence;
            rem /= sequence;
            const int64_t head = rem % shard_heads;

            const int64_t part      = d & 1;
            const int64_t half      = d >> 1;
            const int64_t peer      = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t src_d0    = plane * head_dim + 2 * half;
            const int64_t src_d1    = src_d0 + 1;
            const int64_t base_idx  = head * total_head_dim +
                                     local_seq * total_head_dim * shard_heads +
                                     peer * total_head_dim * shard_heads * shard_sequence;
            const float x0    = wan_fused_qkv_get_f32(recv_flat, src_d0 + base_idx, 0, 0, 0);
            const float x1    = wan_fused_qkv_get_f32(recv_flat, src_d1 + base_idx, 0, 0, 0);
            const float pe0   = wan_fused_qkv_get_f32(pe, part, half, seq, 0);
            const float pe1   = wan_fused_qkv_get_f32(pe, part, half, seq, 1);
            const float value = x0 * pe0 + x1 * pe1;

            if (output_f16) {
                wan_fused_qkv_set_f16(dst, linear, value);
            } else {
                wan_fused_qkv_set_f32(dst, linear, value);
            }
        }
    }

    static inline void wan_fused_qk_recv_rope_vhalf_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 34 || params->mode == 35);
        GGML_ASSERT((params->mode == 34 && dst->type == GGML_TYPE_F32) ||
                    (params->mode == 35 && dst->type == GGML_TYPE_F16));
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        const ggml_tensor* pe        = dst->src[1];
        GGML_ASSERT(recv_flat != nullptr && pe != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);

        const int64_t world_size = params->world_size;
        const int64_t plane      = params->unused3;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);

        const int64_t head_dim       = dst->ne[0];
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_heads    = dst->ne[2];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t half_dim       = head_dim / 2;
        const int64_t packed_dim     = head_dim * 2 + half_dim;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == half_dim);
        GGML_ASSERT(pe->ne[2] >= sequence);
        GGML_ASSERT(pe->ne[3] == 2);

        const bool output_f16 = params->mode == 35;
        const int64_t total   = ggml_nelements(dst);
        for (int64_t linear = ith; linear < total; linear += nth) {
            int64_t rem        = linear;
            const int64_t d    = rem % head_dim;
            rem /= head_dim;
            const int64_t seq  = rem % sequence;
            rem /= sequence;
            const int64_t head = rem % shard_heads;

            const int64_t part      = d & 1;
            const int64_t half      = d >> 1;
            const int64_t peer      = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t src_d0    = plane * head_dim + 2 * half;
            const int64_t src_d1    = src_d0 + 1;
            const int64_t base_idx  = head * packed_dim +
                                     local_seq * packed_dim * shard_heads +
                                     peer * packed_dim * shard_heads * shard_sequence;
            const uint32_t x0_bits = wan_fused_qkv_get_u32(recv_flat, src_d0 + base_idx);
            const uint32_t x1_bits = wan_fused_qkv_get_u32(recv_flat, src_d1 + base_idx);
            float x0;
            float x1;
            std::memcpy(&x0, &x0_bits, sizeof(x0));
            std::memcpy(&x1, &x1_bits, sizeof(x1));
            const float pe0   = wan_fused_qkv_get_f32(pe, part, half, seq, 0);
            const float pe1   = wan_fused_qkv_get_f32(pe, part, half, seq, 1);
            const float value = x0 * pe0 + x1 * pe1;

            if (output_f16) {
                wan_fused_qkv_set_f16(dst, linear, value);
            } else {
                wan_fused_qkv_set_f32(dst, linear, value);
            }
        }
    }

    static inline void wan_fused_vhalf_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
        auto* params = static_cast<const WanFusedQKVPackParams*>(userdata);
        GGML_ASSERT(params != nullptr);
        GGML_ASSERT(params->magic == WAN_FUSED_QKV_PACK_MAGIC);
        GGML_ASSERT(params->mode == 36);
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        GGML_ASSERT(ith >= 0 && nth > 0);

        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);

        const int64_t head_dim       = dst->ne[0];
        const int64_t sequence       = dst->ne[1];
        const int64_t shard_heads    = dst->ne[2];
        const int64_t shard_sequence = sequence / world_size;
        const int64_t packed_dim     = head_dim * 2 + head_dim / 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        for (int64_t linear = ith; linear < total; linear += nth) {
            int64_t rem         = linear;
            const int64_t d     = rem % head_dim;
            rem /= head_dim;
            const int64_t seq   = rem % sequence;
            rem /= sequence;
            const int64_t head  = rem % shard_heads;
            const int64_t peer  = seq / shard_sequence;
            const int64_t local_seq = seq - peer * shard_sequence;
            const int64_t src_d = head_dim * 2 + d / 2;
            const int64_t src_idx = src_d +
                                    head * packed_dim +
                                    local_seq * packed_dim * shard_heads +
                                    peer * packed_dim * shard_heads * shard_sequence;
            const uint32_t packed = wan_fused_qkv_get_u32(recv_flat, src_idx);
            const uint32_t bits   = (d & 1) == 0 ? (packed & 0xffffu) : (packed >> 16);
            wan_fused_qkv_set_f16_bits(dst, linear, bits);
        }
    }

    static inline ggml_tensor* wan_fused_qkv_send_pack(ggml_context* ctx,
                                                       ggml_tensor* q,
                                                       ggml_tensor* k,
                                                       ggml_tensor* v,
                                                       int world_size) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);

        const int64_t total_head_dim = q->ne[0] * 3;
        const int64_t shard_heads    = q->ne[1] / world_size;
        const int64_t flat_elems     = total_head_dim * shard_heads * q->ne[2] * world_size;
        ggml_tensor* args[]          = {q, k, v};
        ggml_tensor* out             = ggml_custom_4d(ctx,
                                          GGML_TYPE_F32,
                                          flat_elems,
                                          1,
                                          1,
                                          1,
                                          args,
                                          3,
                                          wan_fused_qkv_send_pack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size));
        ggml_set_name(out, "wan.fused_qkv_send_pack.out");
        return out;
    }

    static inline ggml_tensor* wan_fused_qkv_vhalf_send_pack(ggml_context* ctx,
                                                             ggml_tensor* q,
                                                             ggml_tensor* k,
                                                             ggml_tensor* v,
                                                             int world_size) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(q->ne[0] > 0 && q->ne[0] % 2 == 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);

        const int64_t packed_dim  = q->ne[0] * 2 + q->ne[0] / 2;
        const int64_t shard_heads = q->ne[1] / world_size;
        const int64_t flat_elems  = packed_dim * shard_heads * q->ne[2] * world_size;
        ggml_tensor* args[]       = {q, k, v};
        ggml_tensor* out          = ggml_custom_4d(ctx,
                                          GGML_TYPE_F32,
                                          flat_elems,
                                          1,
                                          1,
                                          1,
                                          args,
                                          3,
                                          wan_fused_qkv_vhalf_send_pack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 33, 0));
        ggml_set_name(out, "wan.fused_qkv_vhalf_send_pack.out");
        return out;
    }

    static inline ggml_tensor* wan_fused_qkv_roped_half_send_pack(ggml_context* ctx,
                                                                  ggml_tensor* q,
                                                                  ggml_tensor* k,
                                                                  ggml_tensor* v,
                                                                  ggml_tensor* prepared_pe,
                                                                  int world_size,
                                                                  int rank) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr && prepared_pe != nullptr);
        GGML_ASSERT(world_size > 0 && rank >= 0 && rank < world_size);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(prepared_pe->type == GGML_TYPE_F32);
        GGML_ASSERT(q->ne[0] > 0 && q->ne[0] % 2 == 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        GGML_ASSERT(prepared_pe->ne[0] == 2);
        GGML_ASSERT(prepared_pe->ne[1] == q->ne[0] / 2);
        GGML_ASSERT(prepared_pe->ne[2] >= q->ne[2] * world_size);
        GGML_ASSERT(prepared_pe->ne[3] == 2);

        const int64_t packed_dim  = q->ne[0] * 2;
        const int64_t shard_heads = q->ne[1] / world_size;
        const int64_t flat_elems  = packed_dim * shard_heads * q->ne[2] * world_size;
        ggml_tensor* args[]       = {q, k, v, prepared_pe};
        ggml_tensor* out          = ggml_custom_4d(ctx,
                                          GGML_TYPE_F32,
                                          flat_elems,
                                          1,
                                          1,
                                          1,
                                          args,
                                          4,
                                          wan_fused_qkv_roped_half_send_pack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 37, 0, rank));
        ggml_set_name(out, "wan.fused_qkv_roped_half_send_pack.out");
        return out;
    }

    static inline ggml_tensor* wan_fused_qkv_roped_all_half_send_pack(ggml_context* ctx,
                                                                      ggml_tensor* q,
                                                                      ggml_tensor* k,
                                                                      ggml_tensor* v,
                                                                      ggml_tensor* prepared_pe,
                                                                      int world_size,
                                                                      int rank) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr && prepared_pe != nullptr);
        GGML_ASSERT(world_size > 0 && rank >= 0 && rank < world_size);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(prepared_pe->type == GGML_TYPE_F32);
        GGML_ASSERT(q->ne[0] > 0 && q->ne[0] % 2 == 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        GGML_ASSERT(prepared_pe->ne[0] == 2);
        GGML_ASSERT(prepared_pe->ne[1] == q->ne[0] / 2);
        GGML_ASSERT(prepared_pe->ne[2] >= q->ne[2] * world_size);
        GGML_ASSERT(prepared_pe->ne[3] == 2);

        const int64_t packed_dim  = q->ne[0] * 3 / 2;
        const int64_t shard_heads = q->ne[1] / world_size;
        const int64_t flat_elems  = packed_dim * shard_heads * q->ne[2] * world_size;
        ggml_tensor* args[]       = {q, k, v, prepared_pe};
        ggml_tensor* out          = ggml_custom_4d(ctx,
                                          GGML_TYPE_F32,
                                          flat_elems,
                                          1,
                                          1,
                                          1,
                                          args,
                                          4,
                                          wan_fused_qkv_roped_all_half_send_pack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 56, 0, rank));
        ggml_set_name(out, "wan.fused_qkv_roped_all_half_send_pack.out");
        return out;
    }

    static inline ggml_tensor* wan_fused_attn_head_to_seq_send_pack(ggml_context* ctx,
                                                                    ggml_tensor* attn_head,
                                                                    int world_size,
                                                                    const std::string& name,
                                                                    bool output_f16 = false) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(attn_head != nullptr);
        GGML_ASSERT(attn_head->type == GGML_TYPE_F32 || (output_f16 && attn_head->type == GGML_TYPE_F16));
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(attn_head->ne[3] == 1);
        GGML_ASSERT(attn_head->ne[2] % world_size == 0);

        const int64_t count_per_peer = attn_head->ne[0] *
                                       attn_head->ne[1] *
                                       (attn_head->ne[2] / world_size);
        ggml_tensor* args[] = {attn_head};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          output_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                          count_per_peer * world_size,
                                          1,
                                          1,
                                          1,
                                          args,
                                          1,
                                          wan_fused_attn_head_to_seq_send_pack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, output_f16 ? 59 : 24, 0));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_attn_head_to_seq_recv_unpack(ggml_context* ctx,
                                                                      ggml_tensor* recv_flat,
                                                                      int64_t head_dim,
                                                                      int64_t heads,
                                                                      int64_t shard_sequence,
                                                                      int world_size,
                                                                      const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 || recv_flat->type == GGML_TYPE_F16);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && heads > 0 && shard_sequence > 0);
        GGML_ASSERT(heads % world_size == 0);
        const int64_t count_per_peer = head_dim * (heads / world_size) * shard_sequence;
        GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

        ggml_tensor* args[] = {recv_flat};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F32,
                                          head_dim,
                                          heads,
                                          shard_sequence,
                                          1,
                                          args,
                                          1,
                                          wan_fused_attn_head_to_seq_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size,
                                                                         recv_flat->type == GGML_TYPE_F16 ? 60 : 25,
                                                                         0));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_qkv_recv_unpack(ggml_context* ctx,
                                                         ggml_tensor* recv_flat,
                                                         int64_t head_dim,
                                                         int64_t shard_heads,
                                                         int64_t shard_sequence,
                                                         int world_size,
                                                         int plane,
                                                         const std::string& name,
                                                         bool output_f16 = false) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(plane >= 0 && plane < 3);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t total_head_dim = head_dim * 3;
        GGML_ASSERT(ggml_nelements(recv_flat) == total_head_dim * shard_heads * shard_sequence * world_size);

        const bool is_qk       = plane < 2;
        output_f16             = output_f16 && !is_qk;
        const int64_t sequence = shard_sequence * world_size;
        ggml_tensor* args[]    = {recv_flat};
        ggml_tensor* out       = ggml_custom_4d(ctx,
                                          output_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                          is_qk ? head_dim / 2 : head_dim,
                                          sequence,
                                          shard_heads,
                                          is_qk ? 2 : 1,
                                          args,
                                          1,
                                          wan_fused_qkv_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, is_qk ? 22 : (output_f16 ? 26 : 23), plane));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_qk_recv_rope(ggml_context* ctx,
                                                      ggml_tensor* recv_flat,
                                                      ggml_tensor* prepared_pe,
                                                      int64_t head_dim,
                                                      int64_t shard_heads,
                                                      int64_t shard_sequence,
                                                      int world_size,
                                                      int plane,
                                                      const std::string& name,
                                                      bool output_f16) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr && prepared_pe != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(prepared_pe->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        const int64_t sequence       = shard_sequence * world_size;
        const int64_t half_dim       = head_dim / 2;
        const int64_t total_head_dim = head_dim * 3;
        GGML_ASSERT(ggml_nelements(recv_flat) == total_head_dim * shard_heads * shard_sequence * world_size);
        GGML_ASSERT(prepared_pe->ne[0] == 2);
        GGML_ASSERT(prepared_pe->ne[1] == half_dim);
        GGML_ASSERT(prepared_pe->ne[2] >= sequence);
        GGML_ASSERT(prepared_pe->ne[3] == 2);

        ggml_tensor* args[] = {recv_flat, prepared_pe};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          output_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          1,
                                          args,
                                          2,
                                          wan_fused_qk_recv_rope_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, output_f16 ? 32 : 31, plane));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_qk_recv_rope_vhalf(ggml_context* ctx,
                                                            ggml_tensor* recv_flat,
                                                            ggml_tensor* prepared_pe,
                                                            int64_t head_dim,
                                                            int64_t shard_heads,
                                                            int64_t shard_sequence,
                                                            int world_size,
                                                            int plane,
                                                            const std::string& name,
                                                            bool output_f16) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr && prepared_pe != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(prepared_pe->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t half_dim   = head_dim / 2;
        const int64_t packed_dim = head_dim * 2 + half_dim;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);
        GGML_ASSERT(prepared_pe->ne[0] == 2);
        GGML_ASSERT(prepared_pe->ne[1] == half_dim);
        GGML_ASSERT(prepared_pe->ne[2] >= sequence);
        GGML_ASSERT(prepared_pe->ne[3] == 2);

        ggml_tensor* args[] = {recv_flat, prepared_pe};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          output_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          1,
                                          args,
                                          2,
                                          wan_fused_qk_recv_rope_vhalf_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, output_f16 ? 35 : 34, plane));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_vhalf_recv_unpack(ggml_context* ctx,
                                                           ggml_tensor* recv_flat,
                                                           int64_t head_dim,
                                                           int64_t shard_heads,
                                                           int64_t shard_sequence,
                                                           int world_size,
                                                           const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t packed_dim = head_dim * 2 + head_dim / 2;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        ggml_tensor* args[] = {recv_flat};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F16,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          1,
                                          args,
                                          1,
                                          wan_fused_vhalf_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 36, 2));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_roped_qkv_recv_unpack(ggml_context* ctx,
                                                               ggml_tensor* recv_flat,
                                                               int64_t head_dim,
                                                               int64_t shard_heads,
                                                               int64_t shard_sequence,
                                                               int world_size,
                                                               int plane,
                                                               const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        GGML_ASSERT(plane >= 0 && plane < 3);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t packed_dim = head_dim * 2;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        ggml_tensor* args[] = {recv_flat};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          plane == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          1,
                                          args,
                                          1,
                                          wan_fused_qkv_roped_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, plane == 0 ? 38 : (plane == 1 ? 39 : 40), plane));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_roped_kv_recv_unpack(ggml_context* ctx,
                                                              ggml_tensor* recv_flat,
                                                              int64_t head_dim,
                                                              int64_t shard_heads,
                                                              int64_t shard_sequence,
                                                              int world_size,
                                                              const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t packed_dim = head_dim * 2;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        ggml_tensor* args[] = {recv_flat};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F16,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          2,
                                          args,
                                          1,
                                          wan_fused_roped_kv_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 41));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_roped_all_half_q_recv_unpack(ggml_context* ctx,
                                                                      ggml_tensor* recv_flat,
                                                                      int64_t head_dim,
                                                                      int64_t shard_heads,
                                                                      int64_t shard_sequence,
                                                                      int world_size,
                                                                      const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t packed_dim = head_dim * 3 / 2;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        ggml_tensor* args[] = {recv_flat};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F16,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          1,
                                          args,
                                          1,
                                          wan_fused_roped_all_half_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 57));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_fused_roped_all_half_kv_recv_unpack(ggml_context* ctx,
                                                                       ggml_tensor* recv_flat,
                                                                       int64_t head_dim,
                                                                       int64_t shard_heads,
                                                                       int64_t shard_sequence,
                                                                       int world_size,
                                                                       const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t packed_dim = head_dim * 3 / 2;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);

        ggml_tensor* args[] = {recv_flat};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F16,
                                          head_dim,
                                          sequence,
                                          shard_heads,
                                          2,
                                          args,
                                          1,
                                          wan_fused_roped_all_half_recv_unpack_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(world_size, 58));
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_roped_kv_recv_view(ggml_context* ctx,
                                                      ggml_tensor* kv,
                                                      int plane,
                                                      const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(kv != nullptr);
        GGML_ASSERT(kv->type == GGML_TYPE_F16);
        GGML_ASSERT(kv->ne[3] == 2);
        GGML_ASSERT(plane == 0 || plane == 1);
        ggml_tensor* out = ggml_view_4d(ctx,
                                        kv,
                                        kv->ne[0],
                                        kv->ne[1],
                                        kv->ne[2],
                                        1,
                                        kv->nb[1],
                                        kv->nb[2],
                                        kv->nb[3],
                                        static_cast<size_t>(plane) * kv->nb[3]);
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_roped_q_recv_view(ggml_context* ctx,
                                                     ggml_tensor* recv_flat,
                                                     int64_t head_dim,
                                                     int64_t shard_heads,
                                                     int64_t shard_sequence,
                                                     int world_size,
                                                     const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(shard_heads > 0 && shard_sequence > 0);
        const int64_t sequence   = shard_sequence * world_size;
        const int64_t packed_dim = head_dim * 2;
        GGML_ASSERT(ggml_nelements(recv_flat) == packed_dim * shard_heads * shard_sequence * world_size);
        ggml_tensor* out = ggml_view_4d(ctx,
                                        recv_flat,
                                        head_dim,
                                        sequence,
                                        shard_heads,
                                        1,
                                        recv_flat->nb[0] * packed_dim * shard_heads,
                                        recv_flat->nb[0] * packed_dim,
                                        recv_flat->nb[0] * packed_dim * shard_heads * sequence,
                                        0);
        if (!name.empty()) {
            ggml_set_name(out, name.c_str());
        }
        return out;
    }

    static inline ggml_tensor* wan_sp_cont_4d_if_needed(ggml_context* ctx,
                                                        ggml_tensor* x,
                                                        int64_t ne0,
                                                        int64_t ne1,
                                                        int64_t ne2,
                                                        int64_t ne3) {
        if (ggml_is_contiguous(x)) {
            return ggml_reshape_4d(ctx, x, ne0, ne1, ne2, ne3);
        }
        return ggml_cont_4d(ctx, x, ne0, ne1, ne2, ne3);
    }

    static inline std::vector<ggml_tensor*> wan_sp_qkv_from_packed_seq_to_head_recv(ggml_context* ctx,
                                                                                    ggml_tensor* recv_flat,
                                                                                    int64_t head_dim,
                                                                                    int64_t heads,
                                                                                    int64_t shard_sequence,
                                                                                    int world_size,
                                                                                    const std::string& name,
                                                                                    bool v_output_f16 = false) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(recv_flat != nullptr);
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(heads > 0 && heads % world_size == 0);
        GGML_ASSERT(shard_sequence > 0);
        const int64_t shard_heads    = heads / world_size;
        const int64_t sequence       = shard_sequence * world_size;
        const int64_t total_head_dim = head_dim * 3;
        GGML_ASSERT(ggml_nelements(recv_flat) == total_head_dim * shard_heads * shard_sequence * world_size);

        std::vector<ggml_tensor*> outputs;
        outputs.reserve(3);
        for (int plane = 0; plane < 3; ++plane) {
            ggml_tensor* output = wan_fused_qkv_recv_unpack(ctx,
                                                            recv_flat,
                                                            head_dim,
                                                            shard_heads,
                                                            shard_sequence,
                                                            world_size,
                                                            plane,
                                                            name.empty() ? "" : (name + "_output_" + std::to_string(plane)),
                                                            v_output_f16 && plane == 2);
            outputs.push_back(output);
        }
        return outputs;
    }

    static inline bool wan_sp_enabled(GGMLRunnerContext* ctx) {
        return ctx != nullptr &&
               ctx->process_group != nullptr &&
               ctx->process_group->enabled() &&
               ctx->process_group->size() > 1;
    }

    static inline int wan_sp_rank(GGMLRunnerContext* ctx) {
        return ctx->process_group->rank();
    }

    static inline int wan_sp_world_size(GGMLRunnerContext* ctx) {
        return ctx->process_group->size();
    }

    static inline ggml_tensor* wan_sp_view_head_sequence(ggml_context* ctx,
                                                         ggml_tensor* x,
                                                         int64_t start,
                                                         int64_t length,
                                                         const std::string& name) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(start >= 0);
        GGML_ASSERT(length > 0);
        GGML_ASSERT(start + length <= x->ne[2]);

        ggml_tensor* view = ggml_view_4d(ctx,
                                         x,
                                         x->ne[0],
                                         x->ne[1],
                                         length,
                                         x->ne[3],
                                         x->nb[1],
                                         x->nb[2],
                                         x->nb[3],
                                         static_cast<size_t>(start) * x->nb[2]);
        ggml_set_name(view, (name + "_view").c_str());
        return view;
    }

    static inline ggml_tensor* wan_sp_real_head_sequence(ggml_context* ctx,
                                                         ggml_tensor* x,
                                                         int64_t pad,
                                                         const std::string& name) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pad >= 0 && pad <= x->ne[2]);
        if (pad <= 0) {
            ggml_set_name(x, name.c_str());
            return x;
        }
        const int64_t real_seq = x->ne[2] - pad;
        GGML_ASSERT(real_seq > 0);

        ggml_tensor* out = wan_sp_view_head_sequence(ctx, x, 0, real_seq, name + "_real");
        out              = ggml_cont(ctx, out);
        ggml_set_name(out, name.c_str());
        return out;
    }

    static inline ggml_tensor* wan_sp_view_sequence_dim1(ggml_context* ctx,
                                                         ggml_tensor* x,
                                                         int64_t start,
                                                         int64_t length,
                                                         const std::string& name) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(start >= 0);
        GGML_ASSERT(length > 0);
        GGML_ASSERT(start + length <= x->ne[1]);

        ggml_tensor* view = ggml_view_4d(ctx,
                                         x,
                                         x->ne[0],
                                         length,
                                         x->ne[2],
                                         x->ne[3],
                                         x->nb[1],
                                         x->nb[2],
                                         x->nb[3],
                                         static_cast<size_t>(start) * x->nb[1]);
        ggml_set_name(view, (name + "_view").c_str());
        return view;
    }

    static inline ggml_tensor* wan_sp_real_sequence_dim1(ggml_context* ctx,
                                                         ggml_tensor* x,
                                                         int64_t pad,
                                                         const std::string& name) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pad >= 0 && pad <= x->ne[1]);
        if (pad <= 0) {
            ggml_set_name(x, name.c_str());
            return x;
        }
        const int64_t real_seq = x->ne[1] - pad;
        GGML_ASSERT(real_seq > 0);

        ggml_tensor* out = wan_sp_view_sequence_dim1(ctx, x, 0, real_seq, name + "_real");
        out              = ggml_cont(ctx, out);
        ggml_set_name(out, name.c_str());
        return out;
    }

    static inline ggml_tensor* wan_sp_pad_head_sequence(ggml_context* ctx,
                                                        ggml_tensor* x,
                                                        int64_t pad,
                                                        const std::string& name) {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pad >= 0);
        if (pad <= 0) {
            ggml_set_name(x, name.c_str());
            return x;
        }

        ggml_tensor* pad_tensor = ggml_ext_zeros(ctx, x->ne[0], x->ne[1], pad, x->ne[3]);
        ggml_set_name(pad_tensor, (name + "_pad").c_str());
        x = ggml_concat(ctx, x, pad_tensor, 2);
        ggml_set_name(x, name.c_str());
        return x;
    }

    static inline ggml_tensor* wan_sp_prepare_rope_pe_seq_major(ggml_context* ctx,
                                                                ggml_tensor* pe,
                                                                const std::string& name = "") {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(pe != nullptr);
        ggml_tensor* prepared = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));
        if (!name.empty()) {
            ggml_set_name(prepared, name.c_str());
        }
        return prepared;
    }

    static inline ggml_tensor* wan_sp_apply_rope_seq_major_work_layout(ggml_context* ctx,
                                                                       ggml_tensor* x,
                                                                       ggml_tensor* pe,
                                                                       int64_t d_head,
                                                                       ggml_tensor* prepared_pe = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);
        GGML_ASSERT(x->ne[0] * 2 == d_head);
        GGML_ASSERT(x->ne[3] == 2);

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

        pe             = prepared_pe != nullptr ? prepared_pe : wan_sp_prepare_rope_pe_seq_major(ctx, pe);
        auto pe_offset = pe->nb[2] * pe->ne[2];
        auto pe_0      = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], pe_offset * 0);
        auto pe_1      = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], pe_offset * 1);

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head);
    }

    static inline ggml_tensor* wan_sp_apply_rope_seq_major_work_layout_f16(ggml_context* ctx,
                                                                           ggml_tensor* x,
                                                                           ggml_tensor* pe,
                                                                           int64_t d_head,
                                                                           ggml_tensor* prepared_pe = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);
        GGML_ASSERT(x->type == GGML_TYPE_F32);
        GGML_ASSERT(x->ne[0] * 2 == d_head);
        GGML_ASSERT(x->ne[3] == 2);

        prepared_pe = prepared_pe != nullptr ? prepared_pe : wan_sp_prepare_rope_pe_seq_major(ctx, pe);
        GGML_ASSERT(prepared_pe->type == GGML_TYPE_F32);
        GGML_ASSERT(prepared_pe->ne[0] == 2);
        GGML_ASSERT(prepared_pe->ne[1] == x->ne[0]);
        GGML_ASSERT(prepared_pe->ne[2] >= x->ne[1]);
        GGML_ASSERT(prepared_pe->ne[3] == 2);

        ggml_tensor* args[] = {x, prepared_pe};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F16,
                                          d_head,
                                          x->ne[1],
                                          x->ne[2],
                                          1,
                                          args,
                                          2,
                                          wan_sp_rope_custom_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(1, 29, 0));
        return out;
    }

    static inline ggml_tensor* wan_sp_apply_rope_seq_major_work_layout_f32_fused(ggml_context* ctx,
                                                                                 ggml_tensor* x,
                                                                                 ggml_tensor* pe,
                                                                                 int64_t d_head,
                                                                                 ggml_tensor* prepared_pe = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);
        GGML_ASSERT(x->type == GGML_TYPE_F32);
        GGML_ASSERT(x->ne[0] * 2 == d_head);
        GGML_ASSERT(x->ne[3] == 2);

        prepared_pe = prepared_pe != nullptr ? prepared_pe : wan_sp_prepare_rope_pe_seq_major(ctx, pe);
        GGML_ASSERT(prepared_pe->type == GGML_TYPE_F32);
        GGML_ASSERT(prepared_pe->ne[0] == 2);
        GGML_ASSERT(prepared_pe->ne[1] == x->ne[0]);
        GGML_ASSERT(prepared_pe->ne[2] >= x->ne[1]);
        GGML_ASSERT(prepared_pe->ne[3] == 2);

        ggml_tensor* args[] = {x, prepared_pe};
        ggml_tensor* out    = ggml_custom_4d(ctx,
                                          GGML_TYPE_F32,
                                          d_head,
                                          x->ne[1],
                                          x->ne[2],
                                          1,
                                          args,
                                          2,
                                          wan_sp_rope_custom_cpu,
                                          GGML_N_TASKS_MAX,
                                          wan_fused_qkv_pack_make_params(1, 30, 0));
        return out;
    }

    static inline ggml_tensor* wan_sp_attention(GGMLRunnerContext* ctx,
                                                ggml_tensor* q,
                                                ggml_tensor* k,
                                                ggml_tensor* v,
                                                ggml_tensor* pe,
                                                const std::string& name_prefix,
                                                bool v_is_seq_major = false) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        q = Rope::apply_rope(ctx->ggml_ctx, q, pe, true, ctx->backend);
        k = Rope::apply_rope(ctx->ggml_ctx, k, pe, true, ctx->backend);

        q = ggml_cont(ctx->ggml_ctx, q);
        ggml_set_name(q, (name_prefix + "_q_rope").c_str());
        k = ggml_cont(ctx->ggml_ctx, k);
        ggml_set_name(k, (name_prefix + "_k_rope").c_str());
        if (!ggml_is_contiguous(v)) {
            v = ggml_cont(ctx->ggml_ctx, v);
        }
        ggml_set_name(v, (name_prefix + "_v_attn").c_str());

        const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
        GGML_ASSERT(q->ne[0] == k->ne[0]);
        GGML_ASSERT(q->ne[1] == k->ne[1]);
        GGML_ASSERT(q->ne[2] == k->ne[2]);
        GGML_ASSERT(v->ne[0] == q->ne[0]);
        if (v_is_seq_major) {
            GGML_ASSERT(v->ne[1] == q->ne[1]);
            GGML_ASSERT(v->ne[2] == n_head);
        } else {
            GGML_ASSERT(v->ne[1] == n_head);
            GGML_ASSERT(v->ne[2] == q->ne[1]);
        }
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
                                                   1.0f,
                                                   false,
                                                   v_is_seq_major);
        ggml_set_name(attn, (name_prefix + "_attn").c_str());
        return attn;
    }

    static inline ggml_tensor* wan_sp_attention_from_rope_work_layout(GGMLRunnerContext* ctx,
                                                                      ggml_tensor* q,
                                                                      ggml_tensor* k,
                                                                      ggml_tensor* v,
                                                                      ggml_tensor* pe,
                                                                      int64_t d_head,
                                                                      const std::string& name_prefix,
                                                                      bool v_is_seq_major = false,
                                                                      ggml_tensor* prepared_pe = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        if (prepared_pe == nullptr) {
            prepared_pe = wan_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                           pe,
                                                           name_prefix + "_pe_seq_major");
        }
        q = wan_sp_apply_rope_seq_major_work_layout_f32_fused(ctx->ggml_ctx, q, pe, d_head, prepared_pe);
        ggml_set_name(q, (name_prefix + "_q_rope").c_str());
        k = ctx->flash_attn_enabled ?
                wan_sp_apply_rope_seq_major_work_layout_f16(ctx->ggml_ctx, k, pe, d_head, prepared_pe) :
                wan_sp_apply_rope_seq_major_work_layout(ctx->ggml_ctx, k, pe, d_head, prepared_pe);
        ggml_set_name(k, (name_prefix + "_k_rope").c_str());
        if (!ggml_is_contiguous(v)) {
            v = ggml_cont(ctx->ggml_ctx, v);
        }
        ggml_set_name(v, (name_prefix + "_v_attn").c_str());

        const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
        GGML_ASSERT(q->ne[0] == k->ne[0]);
        GGML_ASSERT(q->ne[1] == k->ne[1]);
        GGML_ASSERT(q->ne[2] == k->ne[2]);
        GGML_ASSERT(v->ne[0] == q->ne[0]);
        if (v_is_seq_major) {
            GGML_ASSERT(v->ne[1] == q->ne[1]);
            GGML_ASSERT(v->ne[2] == n_head);
        } else {
            GGML_ASSERT(v->ne[1] == n_head);
            GGML_ASSERT(v->ne[2] == q->ne[1]);
        }
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
                                                   1.0f,
                                                   true,
                                                   v_is_seq_major);
        ggml_set_name(attn, (name_prefix + "_attn").c_str());
        return attn;
    }

    static inline ggml_tensor* wan_sp_attention_prepared_qk(GGMLRunnerContext* ctx,
                                                            ggml_tensor* q,
                                                            ggml_tensor* k,
                                                            ggml_tensor* v,
                                                            const std::string& name_prefix,
                                                            bool v_is_seq_major = false) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);

        ggml_set_name(q, (name_prefix + "_q_rope").c_str());
        ggml_set_name(k, (name_prefix + "_k_rope").c_str());
        if (!ggml_is_contiguous(v)) {
            v = ggml_cont(ctx->ggml_ctx, v);
        }
        ggml_set_name(v, (name_prefix + "_v_attn").c_str());

        const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
        GGML_ASSERT(q->ne[0] == k->ne[0]);
        GGML_ASSERT(q->ne[1] == k->ne[1]);
        GGML_ASSERT(q->ne[2] == k->ne[2]);
        GGML_ASSERT(v->ne[0] == q->ne[0]);
        if (v_is_seq_major) {
            GGML_ASSERT(v->ne[1] == q->ne[1]);
            GGML_ASSERT(v->ne[2] == n_head);
        } else {
            GGML_ASSERT(v->ne[1] == n_head);
            GGML_ASSERT(v->ne[2] == q->ne[1]);
        }
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
                                                   1.0f,
                                                   false,
                                                   v_is_seq_major);
        ggml_set_name(attn, (name_prefix + "_attn").c_str());
        return attn;
    }

    static inline ggml_tensor* wan_sp_flash_attention_seq_major(GGMLRunnerContext* ctx,
                                                                ggml_tensor* q,
                                                                ggml_tensor* k,
                                                                ggml_tensor* v,
                                                                int64_t n_head,
                                                                const std::string& name_prefix) {
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

        ggml_tensor* k_in = k;
        if (k_in->type != GGML_TYPE_F16 || !ggml_is_contiguous(k_in)) {
            k_in = ggml_cast(ctx->ggml_ctx, k_in, GGML_TYPE_F16);
        }

        ggml_tensor* v_in = v;
        if (v_in->type != GGML_TYPE_F16 || !ggml_is_contiguous(v_in)) {
            v_in = ggml_cast(ctx->ggml_ctx, v_in, GGML_TYPE_F16);
        }

        ggml_tensor* out = ggml_flash_attn_ext_with_type(ctx->ggml_ctx,
                                                         q,
                                                         k_in,
                                                         v_in,
                                                         nullptr,
                                                         1.0f / std::sqrt(static_cast<float>(d_head)),
                                                         0,
                                                         0,
                                                         GGML_TYPE_F32);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_set_name(out, (name_prefix + "_attn").c_str());
        if (!ggml_backend_supports_op(ctx->backend, out)) {
            return nullptr;
        }

        out = ggml_view_3d(ctx->ggml_ctx, out, d_head, n_head, L_q, out->nb[1], out->nb[2], 0);
        out = ggml_ext_cont(ctx->ggml_ctx, out);
        out = ggml_reshape_3d(ctx->ggml_ctx, out, d_head * n_head, L_q, N);
        ggml_set_name(out, (name_prefix + "_attn").c_str());
        return out;
    }

    static inline ggml_tensor* wan_attention_prepared_seq_major_no_rope(GGMLRunnerContext* ctx,
                                                                        ggml_tensor* q,
                                                                        ggml_tensor* k,
                                                                        ggml_tensor* v,
                                                                        int64_t n_head,
                                                                        const std::string& name_prefix) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(n_head > 0);
        GGML_ASSERT(q->ne[0] == k->ne[0]);
        GGML_ASSERT(q->ne[0] == v->ne[0]);
        GGML_ASSERT(q->ne[2] == n_head);
        GGML_ASSERT(k->ne[2] == n_head);
        GGML_ASSERT(v->ne[2] == n_head);
        GGML_ASSERT(q->ne[3] == v->ne[3]);
        GGML_ASSERT(k->ne[3] == v->ne[3]);

        ggml_set_name(q, (name_prefix + "_q_attn").c_str());
        ggml_set_name(k, (name_prefix + "_k_attn").c_str());
        ggml_set_name(v, (name_prefix + "_v_attn").c_str());

        ggml_tensor* attn = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                   ctx->backend,
                                                   q,
                                                   k,
                                                   v,
                                                   n_head,
                                                   nullptr,
                                                   true,
                                                   ctx->flash_attn_enabled,
                                                   1.0f,
                                                   true,
                                                   true);
        ggml_set_name(attn, (name_prefix + "_attn").c_str());
        return attn;
    }

    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> padding;
        std::tuple<int, int, int> dilation;
        bool bias;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["weight"] = ggml_new_tensor_4d(ctx,
                                                  GGML_TYPE_F16,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride   = {1, 1, 1},
                     std::tuple<int, int, int> padding  = {0, 0, 0},
                     std::tuple<int, int, int> dilation = {1, 1, 1},
                     bool bias                          = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(std::move(kernel_size)),
              stride(std::move(stride)),
              padding(std::move(padding)),
              dilation(std::move(dilation)),
              bias(bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* cache_x = nullptr) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, ID, IH, IW]
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = nullptr;
            if (bias) {
                b = params["bias"];
            }

            int lp0 = std::get<2>(padding);
            int rp0 = std::get<2>(padding);
            int lp1 = std::get<1>(padding);
            int rp1 = std::get<1>(padding);
            int lp2 = 2 * std::get<0>(padding);
            int rp2 = 0;
            int conv_p0 = 0;
            int conv_p1 = 0;

            if (cache_x != nullptr && lp2 > 0) {
                x = ggml_concat(ctx->ggml_ctx, cache_x, x, 2);
                lp2 -= (int)cache_x->ne[2];
            }

            if (!ctx->circular_x_enabled && !ctx->circular_y_enabled) {
                conv_p0 = lp0;
                conv_p1 = lp1;
                lp0 = rp0 = 0;
                lp1 = rp1 = 0;
            }

            x = ggml_ext_pad_ext(ctx->ggml_ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            const bool cudnn_kernel_supported =
                (std::get<0>(kernel_size) == 3 && std::get<1>(kernel_size) == 3 && std::get<2>(kernel_size) == 3) ||
                (std::get<0>(kernel_size) == 3 && std::get<1>(kernel_size) == 1 && std::get<2>(kernel_size) == 1) ||
                (std::get<0>(kernel_size) == 1 && std::get<1>(kernel_size) == 1 && std::get<2>(kernel_size) == 1);
            const bool use_direct = ctx->conv3d_auto_direct_enabled &&
                                    x != nullptr &&
                                    w != nullptr &&
                                    x->type == GGML_TYPE_F32 &&
                                    w->type == GGML_TYPE_F16 &&
                                    cudnn_kernel_supported &&
                                    in_channels >= 16 &&
                                    ggml_is_contiguous(x) &&
                                    ggml_is_contiguous(w);
            return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                    std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                    conv_p0, conv_p1, 0,
                                    std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                    use_direct);
        }
    };

    class RMS_norm : public UnaryBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            auto iter       = tensor_storage_map.find(prefix + "gamma");
            if (iter != tensor_storage_map.end()) {
                params["gamma"] = ggml_new_tensor(ctx, wtype, iter->second.n_dims, &iter->second.ne[0]);
            } else {
                params["gamma"] = ggml_new_tensor_1d(ctx, wtype, dim);
            }
        }

    public:
        RMS_norm(int64_t dim)
            : dim(dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [N*IC, ID, IH, IW], IC == dim
            // assert N == 1

            ggml_tensor* w = params["gamma"];
            w              = ggml_reshape_1d(ctx->ggml_ctx, w, ggml_nelements(w));
            if (auto h = edgedit::ggml_ext::channel_rms_norm_custom(ctx->ggml_ctx, x, w)) {
                // GGML_OP_CUSTOM is only executed by backends that intercept it (CPU,
                // CUDA). Vulkan reports supports_op=false and, with no scheduler/CPU
                // fallback in single-backend compute, would silently skip it -> garbage.
                // Fall through to the standard-ggml path when the backend can't run it.
                if (ctx->backend == nullptr || ggml_backend_supports_op(ctx->backend, h)) {
                    return h;
                }
            }
            auto h         = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));  // [ID, IH, IW, N*IC]
            h              = ggml_rms_norm(ctx->ggml_ctx, h, 1e-12f);
            h              = ggml_mul(ctx->ggml_ctx, h, w);
            h              = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));

            return h;
        }
    };

    class Resample : public GGMLBlock {
    protected:
        int64_t dim;
        std::string mode;

    public:
        Resample(int64_t dim, const std::string& mode, bool wan2_2 = false)
            : dim(dim), mode(mode) {
            if (mode == "upsample2d") {
                if (wan2_2) {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {1, 1}, {1, 1}));
                } else {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                }
            } else if (mode == "upsample3d") {
                if (wan2_2) {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {1, 1}, {1, 1}));
                } else {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                }
                blocks["time_conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim * 2, {3, 1, 1}, {1, 1, 1}, {1, 0, 0}));
            } else if (mode == "downsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
            } else if (mode == "downsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
                blocks["time_conv"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim, {3, 1, 1}, {2, 1, 1}, {0, 0, 0}));
            } else if (mode == "none") {
                // nn.Identity()
            } else {
                GGML_ASSERT(false && "invalid mode");
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            int64_t c = x->ne[3] / b;
            int64_t t = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            if (mode == "upsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    feat_idx += 1;
                    if (chunk_idx == 0) {
                        // feat_cache[idx] == nullptr, pass
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {  // chunk_idx >= 2
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }
                        if (chunk_idx == 1 && cache_x->ne[2] < 2) {  // Rep
                            cache_x = ggml_pad_ext(ctx->ggml_ctx, cache_x, 0, 0, 0, 0, (int)cache_x->ne[2], 0, 0, 0);
                            // aka cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device),cache_x],dim=2)
                        }
                        if (chunk_idx == 1) {
                            x = time_conv->forward(ctx, x);
                        } else {
                            x = time_conv->forward(ctx, x, feat_cache[idx]);
                        }
                        feat_cache[idx] = cache_x;
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w * h, t, c, 2);                                   // (2, c, t, h*w)
                        x               = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 3, 1, 2));  // (c, t, 2, h*w)
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, 2 * t, c);                                   // (c, t*2, h, w)
                    }
                }
            }

            t = x->ne[2];
            if (mode != "none") {
                auto resample_1 = std::dynamic_pointer_cast<Conv2d>(blocks["resample.1"]);

                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)
                if (mode == "upsample2d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "upsample3d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "downsample2d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                } else if (mode == "downsample3d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                }
                x = resample_1->forward(ctx, x);
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)
            }

            if (mode == "downsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    if (feat_cache[idx] == nullptr) {
                        feat_cache[idx] = x;
                        feat_idx += 1;
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x    = ggml_ext_slice(ctx->ggml_ctx, x, 2, -1, x->ne[2]);
                        x               = ggml_concat(ctx->ggml_ctx,
                                                      ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                      x,
                                                      2);
                        x               = time_conv->forward(ctx, x);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                }
            }

            return x;
        }
    };

    class AvgDown3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int factor_t;
        int factor_s;
        int factor;
        int64_t group_size;

    public:
        AvgDown3D(int64_t in_channels, int64_t out_channels, int factor_t, int factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(in_channels * factor % out_channels == 0);
            group_size = in_channels * factor / out_channels;
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t B = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            int pad_t = (factor_t - T % factor_t) % factor_t;

            x = ggml_pad_ext(ctx->ggml_ctx, x, 0, 0, 0, 0, pad_t, 0, 0, 0);
            T = x->ne[2];

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, factor_t, T / factor_t, C);                                                  // [C, T/factor_t, factor_t, H*W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C, factor_t, T/factor_t, H*W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, factor_s, (H / factor_s) * (T / factor_t), factor_t * C);                        // [C*factor_t, T/factor_t*H/factor_s, factor_s, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C*factor_t, factor_s, T/factor_t*H/factor_s, W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s, W / factor_s, (H / factor_s) * (T / factor_t), factor_s * factor_t * C);  // [C*factor_t*factor_s, T/factor_t*H/factor_s, W/factor_s, factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));                                       // [C*factor_t*factor_s, factor_s, T/factor_t*H/factor_s, W/factor_s]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, (W / factor_s) * (H / factor_s) * (T / factor_t), group_size, out_channels);        // [out_channels, group_size, T/factor_t*H/factor_s*W/factor_s]

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [out_channels, T/factor_t*H/factor_s*W/factor_s, group_size]
            x = ggml_mean(ctx->ggml_ctx, x);                                                     // [out_channels, T/factor_t*H/factor_s*W/factor_s, 1]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W / factor_s, H / factor_s, T / factor_t, out_channels);
            return x;
        }
    };

    class DupUp3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t factor_t;
        int64_t factor_s;
        int64_t factor;
        int64_t repeats;

    public:
        DupUp3D(int64_t in_channels, int64_t out_channels, int64_t factor_t, int64_t factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(out_channels * factor % in_channels == 0);
            repeats = out_channels * factor / in_channels;
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool first_chunk = false,
                             int64_t B        = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            auto x_ = x;
            for (int64_t i = 1; i < repeats; i++) {
                x = ggml_concat(ctx->ggml_ctx, x, x_, 2);
            }

            C = out_channels;

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, factor_s, factor_s * factor_t * C);  // [C*factor_t*factor_s, factor_s, T*H, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [C*factor_t*factor_s, T*H, W, factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, H * T, factor_s, factor_t * C);  // [C*factor_t, factor_s, T*H, W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C*factor_t, T*H, factor_s, W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W * factor_s * H, T, factor_t, C);  // [C, factor_t, T, H*factor_s*W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C, T, factor_t, H*factor_s*W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, factor_s * H, factor_t * T, C);  // [C, T*factor_t, H*factor_s, W*factor_s]

            if (first_chunk) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 2, factor_t - 1, x->ne[2]);
            }

            return x;
        }
    };

    class ResidualBlock : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t out_dim;

    public:
        ResidualBlock(int64_t in_dim, int64_t out_dim)
            : in_dim(in_dim), out_dim(out_dim) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(in_dim));
            // residual.1 is nn.SiLU()
            blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            if (in_dim != out_dim) {
                blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* h = x;
            if (in_dim != out_dim) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);

                h = shortcut->forward(ctx, x);
            }

            for (int i = 0; i < 7; i++) {
                if (i == 0 || i == 3) {  // RMS_norm
                    auto layer = std::dynamic_pointer_cast<RMS_norm>(blocks["residual." + std::to_string(i)]);
                    x          = layer->forward(ctx, x);
                } else if (i == 2 || i == 6) {  // CausalConv3d
                    auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                    if (feat_cache.size() > 0) {
                        int idx      = feat_idx;
                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }

                        x               = layer->forward(ctx, x, feat_cache[idx]);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                } else if (i == 1 || i == 4) {
                    x = ggml_silu(ctx->ggml_ctx, x);
                } else {  // i == 5
                    // nn.Dropout(), ignore
                }
            }

            x = ggml_add(ctx->ggml_ctx, x, h);
            return x;
        }
    };

    class Down_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool down_flag;

    public:
        Down_ResidualBlock(int64_t in_dim,
                           int64_t out_dim,
                           int mult,
                           bool temperal_downsample = false,
                           bool down_flag           = false)
            : mult(mult), down_flag(down_flag) {
            blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new AvgDown3D(in_dim, out_dim, temperal_downsample ? 2 : 1, down_flag ? 2 : 1));

            int i = 0;
            for (; i < mult; i++) {
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                     = out_dim;
            }
            if (down_flag) {
                std::string mode                           = temperal_downsample ? "downsample3d" : "downsample2d";
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode, true));
                i++;
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* x_copy = x;

            auto avg_shortcut = std::dynamic_pointer_cast<AvgDown3D>(blocks["avg_shortcut"]);

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (down_flag) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
            }

            auto shortcut = avg_shortcut->forward(ctx, x_copy, b);

            x = ggml_add(ctx->ggml_ctx, x, shortcut);

            return x;
        }
    };

    class Up_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool up_flag;

    public:
        Up_ResidualBlock(int64_t in_dim,
                         int64_t out_dim,
                         int mult,
                         bool temperal_upsample = false,
                         bool up_flag           = false)
            : mult(mult), up_flag(up_flag) {
            if (up_flag) {
                blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new DupUp3D(in_dim, out_dim, temperal_upsample ? 2 : 1, up_flag ? 2 : 1));
            }

            int i = 0;
            for (; i < mult; i++) {
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                   = out_dim;
            }
            if (up_flag) {
                std::string mode                         = temperal_upsample ? "upsample3d" : "upsample2d";
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode, true));
                i++;
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* x_copy = x;

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (up_flag) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);

                auto avg_shortcut = std::dynamic_pointer_cast<DupUp3D>(blocks["avg_shortcut"]);
                auto shortcut     = avg_shortcut->forward(ctx, x_copy, chunk_idx == 0, b);

                x = ggml_add(ctx->ggml_ctx, x, shortcut);
            }

            return x;
        }
    };

    class AttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

    public:
        AttentionBlock(int64_t dim)
            : dim(dim) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new RMS_norm(dim));
            blocks["to_qkv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim * 3, {1, 1}));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto norm   = std::dynamic_pointer_cast<RMS_norm>(blocks["norm"]);
            auto to_qkv = std::dynamic_pointer_cast<Conv2d>(blocks["to_qkv"]);
            auto proj   = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

            auto identity = x;

            x = norm->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)

            const int64_t n = x->ne[3];
            const int64_t c = x->ne[2];
            const int64_t h = x->ne[1];
            const int64_t w = x->ne[0];

            auto qkv     = to_qkv->forward(ctx, x);
            auto qkv_vec = split_image_qkv(ctx->ggml_ctx, qkv);

            auto q = qkv_vec[0];
            q      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 2, 0, 1, 3));  // [t, h, w, c]
            q      = ggml_reshape_3d(ctx->ggml_ctx, q, c, h * w, n);                                      // [t, h * w, c]

            auto k = qkv_vec[1];
            k      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 2, 0, 1, 3));  // [t, h, w, c]
            k      = ggml_reshape_3d(ctx->ggml_ctx, k, c, h * w, n);                                      // [t, h * w, c]

            auto v = qkv_vec[2];
            v      = ggml_reshape_3d(ctx->ggml_ctx, v, h * w, c, n);  // [t, c, h * w]

            v = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 1, 0, 2, 3));                            // [t, h * w, c]
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);  // [t, h * w, c]

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [t, c, h * w]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, c, n);                             // [t, c, h, w]

            x = proj->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)

            x = ggml_add(ctx->ggml_ctx, x, identity);
            return x;
        }
    };

    class Encoder3d : public GGMLBlock {
    protected:
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;

    public:
        Encoder3d(int64_t dim                           = 128,
                  int64_t z_dim                         = 4,
                  std::vector<int> dim_mult             = {1, 2, 4, 4},
                  int num_res_blocks                    = 2,
                  std::vector<bool> temperal_downsample = {false, true, true},
                  bool wan2_2                           = false)
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_downsample(temperal_downsample),
              wan2_2(wan2_2) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }

            if (wan2_2) {
                blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(12, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            } else {
                blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(3, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }

            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                if (wan2_2) {
                    bool t_down_flag = i < temperal_downsample.size() ? temperal_downsample[i] : false;
                    auto block       = std::shared_ptr<GGMLBlock>(new Down_ResidualBlock(in_dim,
                                                                                         out_dim,
                                                                                         num_res_blocks,
                                                                                         t_down_flag,
                                                                                         i != dim_mult.size() - 1));

                    blocks["downsamples." + std::to_string(index++)] = block;
                } else {
                    for (int j = 0; j < num_res_blocks; j++) {
                        auto block                                       = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                        blocks["downsamples." + std::to_string(index++)] = block;
                        in_dim                                           = out_dim;
                    }

                    if (i != dim_mult.size() - 1) {
                        std::string mode                                 = temperal_downsample[i] ? "downsample3d" : "downsample2d";
                        auto block                                       = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                        blocks["downsamples." + std::to_string(index++)] = block;
                    }
                }
            }

            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(out_dim));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));

            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, z_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.prelude", "x");

            // downsamples
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                if (wan2_2) {
                    auto layer = std::dynamic_pointer_cast<Down_ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                } else {
                    for (int j = 0; j < num_res_blocks; j++) {
                        auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                    }

                    if (i != dim_mult.size() - 1) {
                        auto layer = std::dynamic_pointer_cast<Resample>(blocks["downsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                    }
                }
                // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.down." + std::to_string(i), "x");
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.mid", "x");

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class Decoder3d : public GGMLBlock {
    protected:
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;

    public:
        Decoder3d(int64_t dim                         = 128,
                  int64_t z_dim                       = 4,
                  std::vector<int> dim_mult           = {1, 2, 4, 4},
                  int num_res_blocks                  = 2,
                  std::vector<bool> temperal_upsample = {true, true, false},
                  bool wan2_2                         = false)
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_upsample(temperal_upsample),
              wan2_2(wan2_2) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }

            // init block
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            // middle blocks
            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(dims[0]));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));

            // upsample blocks
            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                if (wan2_2) {
                    bool t_up_flag = i < temperal_upsample.size() ? temperal_upsample[i] : false;
                    auto block     = std::shared_ptr<GGMLBlock>(new Up_ResidualBlock(in_dim,
                                                                                     out_dim,
                                                                                     num_res_blocks + 1,
                                                                                     t_up_flag,
                                                                                     i != dim_mult.size() - 1));

                    blocks["upsamples." + std::to_string(index++)] = block;
                } else {
                    if (i == 1 || i == 2 || i == 3) {
                        in_dim = in_dim / 2;
                    }
                    for (int j = 0; j < num_res_blocks + 1; j++) {
                        auto block                                     = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                        blocks["upsamples." + std::to_string(index++)] = block;
                        in_dim                                         = out_dim;
                    }

                    if (i != dim_mult.size() - 1) {
                        std::string mode                               = temperal_upsample[i] ? "upsample3d" : "upsample2d";
                        auto block                                     = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                        blocks["upsamples." + std::to_string(index++)] = block;
                    }
                }
            }

            // output blocks
            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            if (wan2_2) {
                blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 12, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            } else {
                blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 3, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.prelude", "x");

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.mid", "x");

            // upsamples
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                if (wan2_2) {
                    auto layer = std::dynamic_pointer_cast<Up_ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                } else {
                    for (int j = 0; j < num_res_blocks + 1; j++) {
                        auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                    }

                    if (i != dim_mult.size() - 1) {
                        auto layer = std::dynamic_pointer_cast<Resample>(blocks["upsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                    }
                }
                // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.up." + std::to_string(i), "x");
            }

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class WanVAE : public GGMLBlock {
    public:
        bool wan2_2                           = false;
        bool decode_only                      = true;
        int64_t dim                           = 96;
        int64_t dec_dim                       = 96;
        int64_t z_dim                         = 16;
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};

        int _conv_num = 33;
        int _conv_idx = 0;
        std::vector<ggml_tensor*> _feat_map;
        int _enc_conv_num = 28;
        int _enc_conv_idx = 0;
        std::vector<ggml_tensor*> _enc_feat_map;

        void clear_cache() {
            _conv_idx     = 0;
            _feat_map     = std::vector<ggml_tensor*>(_conv_num, nullptr);
            _enc_conv_idx = 0;
            _enc_feat_map = std::vector<ggml_tensor*>(_enc_conv_num, nullptr);
        }

    public:
        WanVAE(bool decode_only = true, bool wan2_2 = false)
            : decode_only(decode_only), wan2_2(wan2_2) {
            // attn_scales is always []
            if (wan2_2) {
                dim     = 160;
                dec_dim = 256;
                z_dim   = 48;

                _conv_num     = 34;
                _enc_conv_num = 26;
            }
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample, wan2_2));
                blocks["conv1"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, temperal_upsample, wan2_2));
            blocks["conv2"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
        }

        static ggml_tensor* patchify(ggml_context* ctx,
                                     ggml_tensor* x,
                                     int64_t patch_size,
                                     int64_t b = 1) {
            // x: [b*c, f, h*q, w*r]
            // return: [b*c*r*q, f, h, w]
            if (patch_size == 1) {
                return x;
            }
            int64_t r = patch_size;
            int64_t q = patch_size;
            int64_t c = x->ne[3] / b;
            int64_t f = x->ne[2];
            int64_t h = x->ne[1] / q;
            int64_t w = x->ne[0] / r;

            x = ggml_reshape_4d(ctx, x, r * w, q, h, f * c * b);                 // [b*c*f, h, q, w*r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c*f, q, h, w*r]
            x = ggml_reshape_4d(ctx, x, r, w, h * q, f * c * b);                 // [b*c*f, q*h, w, r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [b*c*f, r, q*h, w]
            x = ggml_reshape_4d(ctx, x, w * h, q * r, f, c * b);                 // [b*c, f, r*q, h*w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c, r*q, f, h*w]
            x = ggml_reshape_4d(ctx, x, w, h, f, q * r * c * b);                 // [b*c*r*q, f, h, w]

            return x;
        }

        static ggml_tensor* unpatchify(ggml_context* ctx,
                                       ggml_tensor* x,
                                       int64_t patch_size,
                                       int64_t b = 1) {
            // x: [b*c*r*q, f, h, w]
            // return: [b*c, f, h*q, w*r]
            if (patch_size == 1) {
                return x;
            }
            int64_t r = patch_size;
            int64_t q = patch_size;
            int64_t c = x->ne[3] / b / q / r;
            int64_t f = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            x = ggml_reshape_4d(ctx, x, w * h, f, q * r, c * b);                 // [b*c, r*q, f, h*w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c, f, r*q, h*w]
            x = ggml_reshape_4d(ctx, x, w, h * q, r, f * c * b);                 // [b*c*f, r, q*h, w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [b*c*f, q*h, w, r]
            x = ggml_reshape_4d(ctx, x, r * w, h, q, f * c * b);                 // [b*c*f, q, h, w*r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c*f, h, q, w*r]
            x = ggml_reshape_4d(ctx, x, r * w, q * h, f, c * b);                 // [b*c, f, h*q, w*r]
            return x;
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx,
                            ggml_tensor* x,
                            int64_t b = 1) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            if (wan2_2) {
                x = patchify(ctx->ggml_ctx, x, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encode.prelude", "x");

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            int64_t t     = x->ne[2];
            int64_t iter_ = 1 + (t - 1) / 4;
            ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _enc_conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);  // [b*c, 1, h, w]
                    out     = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1 + 4 * (i - 1), 1 + 4 * i);  // [b*c, 4, h, w]
                    auto out_ = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            out     = conv1->forward(ctx, out);
            auto mu = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3)[0];
            // sd::ggml_graph_cut::mark_graph_cut(mu, "wan_vae.encode.final", "mu");
            clear_cache();
            return mu;
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* z,
                            int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            clear_cache();

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            int64_t iter_ = z->ne[2];
            auto x        = conv2->forward(ctx, z);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decode.prelude", "x");
            ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    out     = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    auto out_ = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(out, "wan_vae.decode.final", "out");
            clear_cache();
            return out;
        }

        ggml_tensor* decode_partial(GGMLRunnerContext* ctx,
                                    ggml_tensor* z,
                                    int i,
                                    int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto x = conv2->forward(ctx, z);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decode_partial.prelude", "x");
            auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
            _conv_idx = 0;
            auto out  = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(out, "wan_vae.decode_partial.final", "out");
            return out;
        }
    };

    struct WanVAERunner : public VAE {
        float scale_factor = 1.0f;
        bool decode_only   = true;
        WanVAE ae;

        WanVAERunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false,
                     SDVersion version                              = VERSION_WAN2)
            : decode_only(decode_only), ae(decode_only, version == VERSION_WAN2_2_TI2V), VAE(version, backend, offload_params_to_cpu) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "wan_vae";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
            ED_UNUSED(rng);
            return vae_output;
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> get_latents_mean_std(const sd::Tensor<float>& latents) {
            int channel_dim = latents.dim() == 5 ? 3 : 2;
            std::vector<int64_t> stats_shape(static_cast<size_t>(latents.dim()), 1);
            if (latents.shape()[channel_dim] == 16) {  // Wan2.1 VAE
                stats_shape[static_cast<size_t>(channel_dim)] = 16;

                auto mean_tensor = sd::Tensor<float>::from_vector({-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                                                   0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f});
                mean_tensor.reshape_(stats_shape);
                auto std_tensor = sd::Tensor<float>::from_vector({2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                                                  3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f});
                std_tensor.reshape_(stats_shape);
                return {std::move(mean_tensor), std::move(std_tensor)};
            }
            if (latents.shape()[channel_dim] == 48) {  // Wan2.2 VAE
                stats_shape[static_cast<size_t>(channel_dim)] = 48;

                auto mean_tensor = sd::Tensor<float>::from_vector({-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f,
                                                                   -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f,
                                                                   -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f,
                                                                   -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f,
                                                                   -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f,
                                                                   0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f});
                mean_tensor.reshape_(stats_shape);
                auto std_tensor = sd::Tensor<float>::from_vector({0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
                                                                  0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
                                                                  0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
                                                                  0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
                                                                  0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
                                                                  0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f});
                std_tensor.reshape_(stats_shape);
                return {std::move(mean_tensor), std::move(std_tensor)};
            }
            GGML_ABORT("unexpected latent channel dimension %lld for version %d",
                       (long long)latents.shape()[channel_dim],
                       version);
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents);
            return (latents * std_tensor) / scale_factor + mean_tensor;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents);
            return ((latents - mean_tensor) * scale_factor) / std_tensor;
        }

        int get_encoder_output_channels(int input_channels) {
            return static_cast<int>(ae.z_dim);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
            ggml_cgraph* gf = new_graph_custom(10240 * z_tensor.shape()[2]);
            ggml_tensor* z  = make_input(z_tensor);

            auto runner_ctx = get_context();
            runner_ctx.conv2d_auto_direct_enabled = decode_graph && should_use_cuda_auto_conv2d();
            runner_ctx.conv3d_auto_direct_enabled = decode_graph && should_use_cuda_auto_conv3d();
#if !defined(ED_ENABLE_CUDNN_CONV2D) && !defined(ED_ENABLE_CUDNN_CONV3D)
            // CPU backend (no cuDNN): Wan/Qwen VAE is dominated by 3x3x3 CausalConv3d over large
            // feature maps; im2col+GEMM materializes a huge buffer and is bandwidth bound. Enable
            // ggml's direct conv (conv2d for resample/attn 1x1, conv3d for the main trunk). Applies
            // to both encode (img2img/edit input encoding, ~13s otherwise) and decode. The conv3d
            // use_direct gate (ggml_extend.hpp) guards on kernel 3x3x3 / in_channels>=64 /
            // spatial>=128 / w f16, so early small convs fall back automatically.
            //
            // Keep this optimization CPU-only. CUDA and Vulkan do not implement the native
            // GGML_OP_CONV_3D used by the direct path; both must use the portable non-direct
            // lowering (im2col_3d + mul_mat) when cuDNN conv3d is not compiled in.
            if (sd_backend_is(runner_ctx.backend, "CPU")) {
                runner_ctx.conv2d_direct_enabled      = true;
                runner_ctx.conv3d_auto_direct_enabled = true;
            }
#endif

            ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        ggml_cgraph* build_graph_partial(const sd::Tensor<float>& z_tensor, bool decode_graph, int i) {
            ggml_cgraph* gf = new_graph_custom(20480);

            ae.clear_cache();

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                auto feat_cache        = get_cache_tensor_by_name("feat_idx:" + std::to_string(feat_idx));
                ae._feat_map[feat_idx] = feat_cache;
            }

            ggml_tensor* z = make_input(z_tensor);

            auto runner_ctx = get_context();
            runner_ctx.conv2d_auto_direct_enabled = decode_graph && should_use_cuda_auto_conv2d();
            runner_ctx.conv3d_auto_direct_enabled = decode_graph && should_use_cuda_auto_conv3d();

            ggml_tensor* out = decode_graph ? ae.decode_partial(&runner_ctx, z, i) : ae.encode(&runner_ctx, z);

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                ggml_tensor* feat_cache = ae._feat_map[feat_idx];
                if (feat_cache != nullptr) {
                    cache("feat_idx:" + std::to_string(feat_idx), feat_cache);
                    ggml_build_forward_expand(gf, feat_cache);
                }
            }

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            if (true) {
                sd::Tensor<float> input;
                if (z.dim() == 4) {
                    input = z.unsqueeze(2);
                }
                auto get_graph = [&]() -> ggml_cgraph* {
                    if (input.empty()) {
                        return build_graph(z, decode_graph);
                    } else {
                        return build_graph(input, decode_graph);
                    }
                };
                auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, true),
                                                              input.empty() ? z.dim() : input.dim());
                if (!result.empty() && z.dim() == 4) {
                    result.squeeze_(2);
                }
                return result;
            } else {  // chunk 1 result is weird
                ae.clear_cache();
                int64_t t      = z.shape()[2];
                int i          = 0;
                auto get_graph = [&]() -> ggml_cgraph* {
                    return build_graph_partial(z, decode_graph, i);
                };
                auto out_opt = GGMLRunner::compute<float>(get_graph, n_threads, true);
                if (!out_opt.has_value()) {
                    return {};
                }
                sd::Tensor<float> out = std::move(*out_opt);
                ae.clear_cache();
                if (t == 1) {
                    return out;
                }

                sd::Tensor<float> output = std::move(out);

                for (i = 1; i < t; i++) {
                    auto chunk_opt = GGMLRunner::compute<float>(get_graph, n_threads, true);
                    if (!chunk_opt.has_value()) {
                        return {};
                    }
                    out = std::move(*chunk_opt);
                    ae.clear_cache();
                    output = sd::ops::concat(output, out, 2);
                }
                free_cache_ctx_and_buffer();
                return output;
            }
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            if (true) {
                // cpu f32, pass
                // cpu f16, pass
                // cuda f16, pass
                // cuda f32, pass
                auto z = sd::load_tensor_from_file_as_tensor<float>("wan_vae_z.bin");
                print_sd_tensor(z);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = _compute(8, z, true);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("decode test done in %ldms", t1 - t0);
            }
        };

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend            = ggml_backend_cpu_init();
            ggml_type model_data_type         = GGML_TYPE_F16;
            std::shared_ptr<WanVAERunner> vae = std::make_shared<WanVAERunner>(backend, false, String2TensorStorage{}, "", false, VERSION_WAN2_2_TI2V);
            {
                LOG_INFO("loading from '%s'", file_path.c_str());

                vae->alloc_params_buffer();
                std::map<std::string, ggml_tensor*> tensors;
                vae->get_param_tensors(tensors, "first_stage_model");

                ModelLoader model_loader;
                if (!model_loader.init_from_file_and_convert_name(file_path, "vae.")) {
                    LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                    return;
                }

                bool success = model_loader.load_tensors(tensors);

                if (!success) {
                    LOG_ERROR("load tensors from model loader failed");
                    return;
                }

                LOG_INFO("vae model loaded");
            }
            vae->test();
        }
    };

    class WanSelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t head_dim;

    public:
        WanSelfAttention(int64_t dim,
                         int64_t num_heads,
                         bool qk_norm = true,
                         float eps    = 1e-6)
            : num_heads(num_heads) {
            head_dim    = dim / num_heads;
            blocks["q"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["k"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["o"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));

            if (qk_norm) {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new Identity());
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        virtual ggml_tensor* forward(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* pe,
                                     ggml_tensor* mask = nullptr) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, x);
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, x);  // [N, n_token, n_head*d_head]

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.0f, true, true);  // [N, n_token, dim]

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* pe,
                                int64_t x_pad,
                                const std::string& name_prefix,
                                ggml_tensor* prepared_pe = nullptr) {
            int64_t N             = x->ne[2];
            int64_t local_n_token = x->ne[1];
            const int world_size  = wan_sp_world_size(ctx);

            GGML_ASSERT(N == 1);
            GGML_ASSERT(num_heads % world_size == 0);

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, x);
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, x);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, local_n_token, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, local_n_token, N);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, local_n_token, N);

            ggml_set_name(q, (name_prefix + "_q_local").c_str());
            ggml_set_name(k, (name_prefix + "_k_local").c_str());
            ggml_set_name(v, (name_prefix + "_v_local").c_str());

            const bool use_roped_half_qkv = x_pad == 0 && ctx->flash_attn_enabled;
            const bool use_roped_all_half_qkv =
                use_roped_half_qkv && wan_sp_f16_q_enabled() && head_dim == 128;
            if (use_roped_half_qkv && prepared_pe == nullptr) {
                prepared_pe = wan_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                               pe,
                                                               name_prefix + "_pe_seq_major");
            }
            ggml_tensor* qkv_send_flat = use_roped_all_half_qkv ?
                                             wan_fused_qkv_roped_all_half_send_pack(ctx->ggml_ctx,
                                                                                   q,
                                                                                   k,
                                                                                   v,
                                                                                   prepared_pe,
                                                                                   world_size,
                                                                                   wan_sp_rank(ctx)) :
                                             (use_roped_half_qkv ?
                                                  wan_fused_qkv_roped_half_send_pack(ctx->ggml_ctx,
                                                                                     q,
                                                                                     k,
                                                                                     v,
                                                                                     prepared_pe,
                                                                                     world_size,
                                                                                     wan_sp_rank(ctx)) :
                                                  wan_fused_qkv_send_pack(ctx->ggml_ctx, q, k, v, world_size));
            ggml_set_name(qkv_send_flat, (name_prefix + "_fused_qkv_send_pack").c_str());
            auto qkv_head = edgedit::parallel::sp_all_to_all_4d_seq_to_head_packed_recv_only(
                ctx->ggml_ctx,
                qkv_send_flat,
                use_roped_all_half_qkv ? (head_dim * 3 / 2) : (use_roped_half_qkv ? (head_dim * 2) : (head_dim * 3)),
                num_heads,
                local_n_token,
                N,
                wan_sp_attention_pg_comm_enabled() ? ctx->process_group : nullptr,
                world_size,
                name_prefix + "_qkv_seq_to_head");
            GGML_ASSERT(qkv_head.recv_flat != nullptr);
            const int64_t shard_heads = num_heads / world_size;
            ggml_tensor* attn      = nullptr;
            ggml_tensor* attn_head = nullptr;
            if (x_pad == 0 && ctx->flash_attn_enabled) {
                ggml_tensor* q_attn = use_roped_all_half_qkv ?
                                           wan_fused_roped_all_half_q_recv_unpack(ctx->ggml_ctx,
                                                                                  qkv_head.recv_flat,
                                                                                  head_dim,
                                                                                  shard_heads,
                                                                                  local_n_token,
                                                                                  world_size,
                                                                                  name_prefix + "_q_attn_in") :
                                           wan_roped_q_recv_view(ctx->ggml_ctx,
                                                                 qkv_head.recv_flat,
                                                                 head_dim,
                                                                 shard_heads,
                                                                 local_n_token,
                                                                 world_size,
                                                                 name_prefix + "_q_attn_in");
                ggml_tensor* kv_attn = use_roped_all_half_qkv ?
                                            wan_fused_roped_all_half_kv_recv_unpack(ctx->ggml_ctx,
                                                                                    qkv_head.recv_flat,
                                                                                    head_dim,
                                                                                    shard_heads,
                                                                                    local_n_token,
                                                                                    world_size,
                                                                                    name_prefix + "_kv_attn_in") :
                                            wan_fused_roped_kv_recv_unpack(ctx->ggml_ctx,
                                                                           qkv_head.recv_flat,
                                                                           head_dim,
                                                                           shard_heads,
                                                                           local_n_token,
                                                                           world_size,
                                                                           name_prefix + "_kv_attn_in");
                ggml_tensor* k_attn = wan_roped_kv_recv_view(ctx->ggml_ctx,
                                                             kv_attn,
                                                             0,
                                                             name_prefix + "_k_attn_in");
                ggml_tensor* v_attn = wan_roped_kv_recv_view(ctx->ggml_ctx,
                                                             kv_attn,
                                                             1,
                                                             name_prefix + "_v_attn_in");
                if (use_roped_all_half_qkv) {
                    attn = wan_sp_flash_attention_seq_major(ctx,
                                                            q_attn,
                                                            k_attn,
                                                            v_attn,
                                                            shard_heads,
                                                            name_prefix);
                }
                if (attn == nullptr) {
                    attn = wan_sp_attention_prepared_qk(ctx,
                                                        q_attn,
                                                        k_attn,
                                                        v_attn,
                                                        name_prefix,
                                                        true);
                }
            } else {
                auto qkv_outputs = wan_sp_qkv_from_packed_seq_to_head_recv(ctx->ggml_ctx,
                                                                           qkv_head.recv_flat,
                                                                           head_dim,
                                                                           num_heads,
                                                                           local_n_token,
                                                                           world_size,
                                                                           name_prefix + "_qkv_seq_to_head",
                                                                           true);
                GGML_ASSERT(qkv_outputs.size() == 3);

                ggml_tensor* q_attn = wan_sp_real_sequence_dim1(ctx->ggml_ctx,
                                                                qkv_outputs[0],
                                                                x_pad,
                                                                name_prefix + "_q_attn_in");
                ggml_tensor* k_attn = wan_sp_real_sequence_dim1(ctx->ggml_ctx,
                                                                qkv_outputs[1],
                                                                x_pad,
                                                                name_prefix + "_k_attn_in");
                ggml_tensor* v_attn = wan_sp_real_sequence_dim1(ctx->ggml_ctx,
                                                                qkv_outputs[2],
                                                                x_pad,
                                                                name_prefix + "_v_attn_in");

                attn = wan_sp_attention_from_rope_work_layout(ctx,
                                                              q_attn,
                                                              k_attn,
                                                              v_attn,
                                                              pe,
                                                              head_dim,
                                                              name_prefix,
                                                              true,
                                                              prepared_pe);
            }
            if (attn_head == nullptr) {
                const int64_t real_seq = attn->ne[1];
                attn_head = ggml_reshape_4d(ctx->ggml_ctx,
                                            attn,
                                            head_dim,
                                            shard_heads,
                                            real_seq,
                                            N);
                ggml_set_name(attn_head, (name_prefix + "_attn_4d").c_str());
            }

            attn_head = wan_sp_pad_head_sequence(ctx->ggml_ctx,
                                                 attn_head,
                                                 x_pad,
                                                 name_prefix + "_attn_head_padded");
            if (!ggml_is_contiguous(attn_head)) {
                attn_head = ggml_cont(ctx->ggml_ctx, attn_head);
            }
            ggml_set_name(attn_head, (name_prefix + "_attn_head").c_str());

            const bool use_f16_head_to_seq = wan_sp_f16_head_to_seq_enabled();
            ggml_tensor* attn_head_to_seq_send = use_f16_head_to_seq ?
                                                     wan_fused_attn_head_to_seq_send_pack(
                                                         ctx->ggml_ctx,
                                                         attn_head,
                                                         world_size,
                                                         name_prefix + "_attn_head_to_seq_send_pack_f16",
                                                         true) :
                                                     ggml_reshape_1d(ctx->ggml_ctx,
                                                                     attn_head,
                                                                     ggml_nelements(attn_head));
            ggml_set_name(attn_head_to_seq_send,
                          (name_prefix + "_attn_head_to_seq_send_flat").c_str());
            auto attn_head_to_seq = use_f16_head_to_seq ?
                                        edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_recv_only_f16(
                                            ctx->ggml_ctx,
                                            attn_head_to_seq_send,
                                            head_dim,
                                            shard_heads,
                                            std::vector<int64_t>{attn_head->ne[2]},
                                            wan_sp_attention_pg_comm_enabled() ? ctx->process_group : nullptr,
                                            world_size,
                                            name_prefix + "_attn_head_to_seq") :
                                        edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_recv_only(
                                            ctx->ggml_ctx,
                                            attn_head_to_seq_send,
                                            head_dim,
                                            shard_heads,
                                            std::vector<int64_t>{attn_head->ne[2]},
                                            wan_sp_attention_pg_comm_enabled() ? ctx->process_group : nullptr,
                                            world_size,
                                            name_prefix + "_attn_head_to_seq");
            GGML_ASSERT(attn_head_to_seq.recv_flat != nullptr);
            ggml_tensor* attn_output = wan_fused_attn_head_to_seq_recv_unpack(
                ctx->ggml_ctx,
                attn_head_to_seq.recv_flat,
                head_dim,
                num_heads,
                attn_head->ne[2] / world_size,
                world_size,
                name_prefix + "_attn_head_to_seq_output");
            ggml_tensor* out = ggml_reshape_3d(ctx->ggml_ctx,
                                               attn_output,
                                               head_dim * num_heads,
                                               attn_output->ne[2],
                                               N);
            ggml_set_name(out, (name_prefix + "_attn_out").c_str());

            out = o_proj->forward(ctx, out);
            return out;
        }
    };

    class WanCrossAttention : public WanSelfAttention {
    public:
        WanCrossAttention(int64_t dim,
                          int64_t num_heads,
                          bool qk_norm = true,
                          float eps    = 1e-6)
            : WanSelfAttention(dim, num_heads, qk_norm, eps) {}
        virtual ggml_tensor* forward(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* context,
                                     int64_t context_img_len) = 0;

        virtual ggml_tensor* forward_named(GGMLRunnerContext* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* context,
                                           int64_t context_img_len,
                                           const std::string& name_prefix) {
            (void)name_prefix;
            return forward(ctx, x, context, context_img_len);
        }

        virtual ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* context,
                                        int64_t context_img_len,
                                        const std::string& name_prefix) {
            (void)ctx;
            (void)x;
            (void)context;
            (void)context_img_len;
            (void)name_prefix;
            return forward(ctx, x, context, context_img_len);
        }
    };

    class WanT2VCrossAttention : public WanCrossAttention {
    public:
        WanT2VCrossAttention(int64_t dim,
                             int64_t num_heads,
                             bool qk_norm = true,
                             float eps    = 1e-6)
            : WanCrossAttention(dim, num_heads, qk_norm, eps) {}
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             int64_t context_img_len) override {
            return forward_named(ctx, x, context, context_img_len, "");
        }

        ggml_tensor* forward_named(GGMLRunnerContext* ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* context,
                                   int64_t context_img_len,
                                   const std::string& name_prefix) override {
            // x: [N, n_token, dim]
            // context: [N, n_context, dim]
            // context_img_len: unused
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context);  // [N, n_context, dim]
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context);  // [N, n_context, dim]

            ggml_tensor* attn = nullptr;
            const bool use_named_cudnn_cross_attn =
                !name_prefix.empty() &&
                wan_sp_f16_cross_attn_enabled() &&
                ctx->flash_attn_enabled &&
                head_dim == 128;
            if (use_named_cudnn_cross_attn) {
                ggml_tensor* q4 = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
                ggml_tensor* k4 = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, context->ne[1], N);
                ggml_tensor* v4 = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, context->ne[1], N);
                ggml_tensor* q_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx->ggml_ctx, q4, false);
                ggml_tensor* k_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx->ggml_ctx, k4, false);
                ggml_tensor* v_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx->ggml_ctx, v4, false);
                if (q_f16 != nullptr && k_f16 != nullptr && v_f16 != nullptr &&
                    ggml_backend_supports_op(ctx->backend, q_f16) &&
                    ggml_backend_supports_op(ctx->backend, k_f16) &&
                    ggml_backend_supports_op(ctx->backend, v_f16)) {
                    ggml_set_name(q_f16, (name_prefix + "_q_attn_f16").c_str());
                    ggml_set_name(k_f16, (name_prefix + "_k_attn_f16").c_str());
                    ggml_set_name(v_f16, (name_prefix + "_v_attn_f16").c_str());
                    attn = wan_sp_flash_attention_seq_major(ctx,
                                                            q_f16,
                                                            k_f16,
                                                            v_f16,
                                                            num_heads,
                                                            name_prefix);
                    if (attn != nullptr) {
                        ggml_set_name(attn, (name_prefix + "_attn").c_str());
                    }
                }
            }

            if (attn == nullptr) {
                attn = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
            }

            x = o_proj->forward(ctx, attn);  // [N, n_token, dim]
            return x;
        }

        ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* context,
                                int64_t context_img_len,
                                const std::string& name_prefix) override {
            (void)context_img_len;
            const int64_t N           = x->ne[2];
            const int64_t n_token     = x->ne[1];
            const int64_t context_len = context->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context);
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context);

            ggml_tensor* attn = nullptr;
            const bool use_f16_cross_attn =
                wan_sp_f16_cross_attn_enabled() && ctx->flash_attn_enabled && head_dim == 128;
            if (use_f16_cross_attn) {
                ggml_tensor* q4 = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
                ggml_tensor* k4 = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, context_len, N);
                ggml_tensor* v4 = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, context_len, N);
                ggml_tensor* q_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx->ggml_ctx, q4, false);
                ggml_tensor* k_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx->ggml_ctx, k4, false);
                ggml_tensor* v_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx->ggml_ctx, v4, false);
                if (q_f16 != nullptr && k_f16 != nullptr && v_f16 != nullptr &&
                    ggml_backend_supports_op(ctx->backend, q_f16) &&
                    ggml_backend_supports_op(ctx->backend, k_f16) &&
                    ggml_backend_supports_op(ctx->backend, v_f16)) {
                    ggml_set_name(q_f16, (name_prefix + "_q_attn_f16").c_str());
                    ggml_set_name(k_f16, (name_prefix + "_k_attn_f16").c_str());
                    ggml_set_name(v_f16, (name_prefix + "_v_attn_f16").c_str());
                    attn = wan_sp_flash_attention_seq_major(ctx,
                                                            q_f16,
                                                            k_f16,
                                                            v_f16,
                                                            num_heads,
                                                            name_prefix);
                    if (attn != nullptr) {
                        ggml_set_name(attn, (name_prefix + "_attn").c_str());
                    }
                }
            }

            if (attn == nullptr) {
                attn = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              num_heads,
                                              nullptr,
                                              false,
                                              ctx->flash_attn_enabled);
            }

            x = o_proj->forward(ctx, attn);
            return x;
        }

    };

    class WanI2VCrossAttention : public WanCrossAttention {
    public:
        WanI2VCrossAttention(int64_t dim,
                             int64_t num_heads,
                             bool qk_norm = true,
                             float eps    = 1e-6)
            : WanCrossAttention(dim, num_heads, qk_norm, eps) {
            blocks["k_img"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v_img"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));

            if (qk_norm) {
                blocks["norm_k_img"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_k_img"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             int64_t context_img_len) override {
            // x: [N, n_token, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

            auto k_img_proj = std::dynamic_pointer_cast<Linear>(blocks["k_img"]);
            auto v_img_proj = std::dynamic_pointer_cast<Linear>(blocks["v_img"]);

            auto norm_q     = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k     = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);
            auto norm_k_img = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k_img"]);

            int64_t N               = x->ne[2];
            int64_t n_token         = x->ne[1];
            int64_t dim             = x->ne[0];
            int64_t context_txt_len = context->ne[1] - context_img_len;

            auto context_img = ggml_view_3d(ctx->ggml_ctx, context, dim, context_img_len, N, context->nb[1], context->nb[2], 0);                                 // [N, context_img_len, dim]
            auto context_txt = ggml_view_3d(ctx->ggml_ctx, context, dim, context_txt_len, N, context->nb[1], context->nb[2], context_img_len * context->nb[1]);  // [N, context_txt_len, dim]

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context_txt);  // [N, context_txt_len, dim]
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context_txt);  // [N, context_txt_len, dim]

            auto k_img = k_img_proj->forward(ctx, context_img);  // [N, context_img_len, dim]
            k_img      = norm_k_img->forward(ctx, k_img);
            auto v_img = v_img_proj->forward(ctx, context_img);  // [N, context_img_len, dim]

            auto img_x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k_img, v_img, num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
            x          = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, ctx->flash_attn_enabled);          // [N, n_token, dim]

            x = ggml_add(ctx->ggml_ctx, x, img_x);

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* context,
                                int64_t context_img_len,
                                const std::string& name_prefix) override {
            (void)name_prefix;
            return forward(ctx, x, context, context_img_len);
        }
    };

    static ggml_tensor* modulate_add(ggml_context* ctx, ggml_tensor* x, ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_add(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_add(ctx, x, e);
        }
        return x;
    }

    static ggml_tensor* modulate_mul(ggml_context* ctx, ggml_tensor* x, ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_mul(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_mul(ctx, x, e);
        }
        return x;
    }

    static ggml_tensor* modulate_shift_scale(ggml_context* ctx, ggml_tensor* x, ggml_tensor* shift, ggml_tensor* scale) {
        // Equivalent to: modulate_add(x + modulate_mul(x, scale), shift).
#ifdef ED_ENABLE_CUDA_MODULATION
        if (ggml_n_dims(shift) == 3 && ggml_n_dims(scale) == 3 && shift->ne[2] == scale->ne[2]) {
            const int64_t T = shift->ne[2];
            if (T > 0 && x->ne[1] % T == 0) {
                ggml_tensor* x4 = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);
                if (auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx, x4, shift, scale)) {
                    return ggml_reshape_3d(ctx, fused, fused->ne[0], fused->ne[1] * fused->ne[2], fused->ne[3]);
                }
            }
        } else if (ggml_n_dims(shift) != 3 && ggml_n_dims(scale) != 3) {
            if (auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx, x, shift, scale)) {
                return fused;
            }
        }
#endif

        ggml_tensor* y = ggml_add(ctx, x, modulate_mul(ctx, x, scale));
        return modulate_add(ctx, y, shift);
    }

    static ggml_tensor* residual_gate(ggml_context* ctx, ggml_tensor* residual, ggml_tensor* x, ggml_tensor* gate) {
#ifdef ED_ENABLE_CUDA_MODULATION
        if (auto fused = edgedit::ggml_ext::fused_residual_gate_custom(ctx, residual, x, gate)) {
            return fused;
        }
#endif
        return ggml_add(ctx, residual, modulate_mul(ctx, x, gate));
    }

    class WanAttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        WanAttentionBlock(bool t2v_cross_attn,
                          int64_t dim,
                          int64_t ffn_dim,
                          int64_t num_heads,
                          bool qk_norm         = true,
                          bool cross_attn_norm = false,
                          float eps            = 1e-6)
            : dim(dim) {
            blocks["norm1"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new WanSelfAttention(dim, num_heads, qk_norm, eps));
            if (cross_attn_norm) {
                blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, true));
            } else {
                blocks["norm3"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
            if (t2v_cross_attn) {
                blocks["cross_attn"] = std::shared_ptr<GGMLBlock>(new WanT2VCrossAttention(dim, num_heads, qk_norm, eps));
            } else {
                blocks["cross_attn"] = std::shared_ptr<GGMLBlock>(new WanI2VCrossAttention(dim, num_heads, qk_norm, eps));
            }

            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));

            blocks["ffn.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, ffn_dim));
            // ffn.1 is nn.GELU(approximate='tanh')
            blocks["ffn.2"] = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, dim));
        }

        virtual ggml_tensor* forward(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* e,
                                     ggml_tensor* pe,
                                     ggml_tensor* context,
                                     int64_t context_img_len = 257,
                                     const std::string& name_prefix = "") {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]

            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 6, dim] or [N, T, 6, dim]
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);  // ([N, 1, dim], ...) or [N, T, 1, dim]

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            // self-attention
            auto y = norm1->forward(ctx, x);
            y      = modulate_shift_scale(ctx->ggml_ctx, y, es[0], es[1]);
            y      = self_attn->forward(ctx, y, pe);

            x = residual_gate(ctx->ggml_ctx, x, y, es[2]);

            // cross-attention
            x = ggml_add(ctx->ggml_ctx,
                         x,
                         cross_attn->forward_named(ctx,
                                                   norm3->forward(ctx, x),
                                                   context,
                                                   context_img_len,
                                                   name_prefix + "_cross"));

            // ffn
            y = norm2->forward(ctx, x);
            y = modulate_shift_scale(ctx->ggml_ctx, y, es[3], es[4]);

            y = ffn_0->forward(ctx, y);
            y = ggml_ext_gelu(ctx->ggml_ctx, y, true, ctx->backend);
            y = ffn_2->forward(ctx, y);

            x = residual_gate(ctx->ggml_ctx, x, y, es[5]);

            return x;
        }

        virtual ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* e,
                                        ggml_tensor* pe,
                                        ggml_tensor* context,
                                        int64_t x_pad,
                                        const std::string& name_prefix,
                                        int64_t context_img_len = 257,
                                        ggml_tensor* prepared_pe = nullptr) {
            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            auto y = norm1->forward(ctx, x);
            y      = modulate_shift_scale(ctx->ggml_ctx, y, es[0], es[1]);
            y      = self_attn->forward_sp(ctx, y, pe, x_pad, name_prefix + "_self", prepared_pe);

            x = residual_gate(ctx->ggml_ctx, x, y, es[2]);

            x = ggml_add(ctx->ggml_ctx,
                         x,
                         cross_attn->forward_sp(ctx,
                                                norm3->forward(ctx, x),
                                                context,
                                                context_img_len,
                                                name_prefix + "_cross"));

            y = norm2->forward(ctx, x);
            y = modulate_shift_scale(ctx->ggml_ctx, y, es[3], es[4]);

            y = ffn_0->forward(ctx, y);
            y = ggml_ext_gelu(ctx->ggml_ctx, y, true, ctx->backend);
            y = ffn_2->forward(ctx, y);

            x = residual_gate(ctx->ggml_ctx, x, y, es[5]);

            return x;
        }
    };

    class VaceWanAttentionBlock : public WanAttentionBlock {
    protected:
        int block_id;
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        VaceWanAttentionBlock(bool t2v_cross_attn,
                              int64_t dim,
                              int64_t ffn_dim,
                              int64_t num_heads,
                              bool qk_norm         = true,
                              bool cross_attn_norm = false,
                              float eps            = 1e-6,
                              int block_id         = 0)
            : WanAttentionBlock(t2v_cross_attn, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps), block_id(block_id) {
            if (block_id == 0) {
                blocks["before_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            }
            blocks["after_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* c,
                                                      ggml_tensor* x,
                                                      ggml_tensor* e,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* context,
                                                      int64_t context_img_len = 257) {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]
            if (block_id == 0) {
                auto before_proj = std::dynamic_pointer_cast<Linear>(blocks["before_proj"]);

                c = before_proj->forward(ctx, c);
                c = ggml_add(ctx->ggml_ctx, c, x);
            }

            auto after_proj = std::dynamic_pointer_cast<Linear>(blocks["after_proj"]);

            c           = WanAttentionBlock::forward(ctx, c, e, pe, context, context_img_len);
            auto c_skip = after_proj->forward(ctx, c);

            return {c_skip, c};
        }
    };

    class Head : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 2, 1);
        }

    public:
        Head(int64_t dim,
             int64_t out_dim,
             std::tuple<int, int, int> patch_size,
             float eps = 1e-6)
            : dim(dim) {
            out_dim = out_dim * std::get<0>(patch_size) * std::get<1>(patch_size) * std::get<2>(patch_size);

            blocks["norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* e) {
            // x: [N, n_token, dim]
            // e: [N, dim] or [N, T, dim]
            // return [N, n_token, out_dim]

            auto modulation = params["modulation"];
            e               = ggml_reshape_4d(ctx->ggml_ctx, e, e->ne[0], 1, e->ne[1], e->ne[2]);  // [N, 1, dim] or [N, T, 1, dim]
            e               = ggml_repeat_4d(ctx->ggml_ctx, e, e->ne[0], 2, e->ne[2], e->ne[3]);   // [N, 2, dim] or [N, T, 2, dim]

            e       = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 2, dim] or [N, T, 2, dim]
            auto es = ggml_ext_chunk(ctx->ggml_ctx, e, 2, 1);  // ([N, 1, dim], ...) or ([N, T, 1, dim], ...)

            auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto head = std::dynamic_pointer_cast<Linear>(blocks["head"]);

            x = norm->forward(ctx, x);
            x = modulate_shift_scale(ctx->ggml_ctx, x, es[0], es[1]);
            x = head->forward(ctx, x);
            return x;
        }
    };

    class MLPProj : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t flf_pos_embed_token_number;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            if (flf_pos_embed_token_number > 0) {
                params["emb_pos"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, flf_pos_embed_token_number, 1);
            }
        }

    public:
        MLPProj(int64_t in_dim,
                int64_t out_dim,
                int64_t flf_pos_embed_token_number = 0)
            : in_dim(in_dim), flf_pos_embed_token_number(flf_pos_embed_token_number) {
            blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
            blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, in_dim));
            // proj.2 is nn.GELU()
            blocks["proj.3"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, out_dim));
            blocks["proj.4"] = std::shared_ptr<GGMLBlock>(new LayerNorm(out_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* image_embeds) {
            if (flf_pos_embed_token_number > 0) {
                auto emb_pos = params["emb_pos"];

                auto a = ggml_ext_slice(ctx->ggml_ctx, image_embeds, 1, 0, emb_pos->ne[1]);
                auto b = ggml_ext_slice(ctx->ggml_ctx, emb_pos, 1, 0, image_embeds->ne[1]);

                image_embeds = ggml_add(ctx->ggml_ctx, a, b);
            }

            auto proj_0 = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.0"]);
            auto proj_1 = std::dynamic_pointer_cast<Linear>(blocks["proj.1"]);
            auto proj_3 = std::dynamic_pointer_cast<Linear>(blocks["proj.3"]);
            auto proj_4 = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.4"]);

            auto x = proj_0->forward(ctx, image_embeds);
            x      = proj_1->forward(ctx, x);
            x      = ggml_ext_gelu(ctx->ggml_ctx, x, true, ctx->backend);
            x      = proj_3->forward(ctx, x);
            x      = proj_4->forward(ctx, x);

            return x;  // clip_extra_context_tokens
        }
    };

    struct WanParams {
        std::string model_type                 = "t2v";
        std::tuple<int, int, int> patch_size   = {1, 2, 2};
        int64_t text_len                       = 512;
        int64_t in_dim                         = 16;
        int64_t dim                            = 2048;
        int64_t ffn_dim                        = 8192;
        int freq_dim                           = 256;
        int64_t text_dim                       = 4096;
        int64_t out_dim                        = 16;
        int64_t num_heads                      = 16;
        int num_layers                         = 32;
        int vace_layers                        = 0;
        int64_t vace_in_dim                    = 96;
        std::map<int, int> vace_layers_mapping = {};
        bool qk_norm                           = true;
        bool cross_attn_norm                   = true;
        float eps                              = 1e-6f;
        int64_t flf_pos_embed_token_number     = 0;
        int theta                              = 10000;
        // wan2.1 1.3B: 1536/12, wan2.1/2.2 14B: 5120/40, wan2.2 5B: 3074/24
        std::vector<int> axes_dim = {44, 42, 42};
        int64_t axes_dim_sum      = 128;
    };

    class Wan : public GGMLBlock {
    protected:
        WanParams params;

    public:
        Wan() {}
        Wan(WanParams params)
            : params(params) {
            // patch_embedding
            blocks["patch_embedding"] = std::shared_ptr<GGMLBlock>(new Conv3d(params.in_dim, params.dim, params.patch_size, params.patch_size));

            // text_embedding
            blocks["text_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.text_dim, params.dim));
            // text_embedding.1 is nn.GELU()
            blocks["text_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim));

            // time_embedding
            blocks["time_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.freq_dim, params.dim));
            // time_embedding.1 is nn.SiLU()
            blocks["time_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim));

            // time_projection.0 is nn.SiLU()
            blocks["time_projection.1"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim * 6));

            // blocks
            for (int i = 0; i < params.num_layers; i++) {
                auto block                            = std::shared_ptr<GGMLBlock>(new WanAttentionBlock(params.model_type == "t2v",
                                                                                                         params.dim,
                                                                                                         params.ffn_dim,
                                                                                                         params.num_heads,
                                                                                                         params.qk_norm,
                                                                                                         params.cross_attn_norm,
                                                                                                         params.eps));
                blocks["blocks." + std::to_string(i)] = block;
            }

            // head
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Head(params.dim, params.out_dim, params.patch_size, params.eps));

            // img_emb
            if (params.model_type == "i2v") {
                blocks["img_emb"] = std::shared_ptr<GGMLBlock>(new MLPProj(1280, params.dim, params.flf_pos_embed_token_number));
            }

            // vace
            if (params.vace_layers > 0) {
                for (int i = 0; i < params.vace_layers; i++) {
                    auto block                                 = std::shared_ptr<GGMLBlock>(new VaceWanAttentionBlock(params.model_type == "t2v",
                                                                                                                      params.dim,
                                                                                                                      params.ffn_dim,
                                                                                                                      params.num_heads,
                                                                                                                      params.qk_norm,
                                                                                                                      params.cross_attn_norm,
                                                                                                                      params.eps,
                                                                                                                      i));
                    blocks["vace_blocks." + std::to_string(i)] = block;
                }

                int step = params.num_layers / params.vace_layers;
                int n    = 0;
                for (int i = 0; i < params.num_layers; i += step) {
                    this->params.vace_layers_mapping[i] = n;
                    n++;
                }

                blocks["vace_patch_embedding"] = std::shared_ptr<GGMLBlock>(new Conv3d(params.vace_in_dim, params.dim, params.patch_size, params.patch_size));
            }
        }

        ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                       ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];

            int pad_t = (std::get<0>(params.patch_size) - T % std::get<0>(params.patch_size)) % std::get<0>(params.patch_size);
            int pad_h = (std::get<1>(params.patch_size) - H % std::get<1>(params.patch_size)) % std::get<1>(params.patch_size);
            int pad_w = (std::get<2>(params.patch_size) - W % std::get<2>(params.patch_size)) % std::get<2>(params.patch_size);
            x = ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, pad_t, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        ggml_tensor* unpatchify(ggml_context* ctx,
                                ggml_tensor* x,
                                int64_t t_len,
                                int64_t h_len,
                                int64_t w_len) {
            // x: [N, t_len*h_len*w_len, pt*ph*pw*C]
            // return: [N*C, t_len*pt, h_len*ph, w_len*pw]
            int64_t N  = x->ne[3];
            int64_t pt = std::get<0>(params.patch_size);
            int64_t ph = std::get<1>(params.patch_size);
            int64_t pw = std::get<2>(params.patch_size);
            int64_t C  = x->ne[0] / pt / ph / pw;

            GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, pt*ph*pw, C]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
            x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);  // [N*C*t_len*h_len, w_len, pt*ph, pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt*ph, w_len, pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt, ph, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, ph, pt, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);  // [N*C*t_len, h_len*ph, pt, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, pt, h_len*ph, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);  // [N*C, t_len*pt, h_len*ph, w_len*pw]
            return x;
        }

        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe,
                                  ggml_tensor* clip_fea     = nullptr,
                                  ggml_tensor* vace_context = nullptr,
                                  float vace_strength       = 1.f,
                                  int64_t N                 = 1) {
            // x: [N*C, T, H, W], C => in_dim
            // vace_context: [N*vace_in_dim, T, H, W]
            // timestep: [N,] or [T]
            // context: [N, L, text_dim]
            // return: [N, t_len*h_len*w_len, out_dim*pt*ph*pw]

            GGML_ASSERT(N == 1);

            auto patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);

            auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);

            auto time_embedding_0  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);

            auto head = std::dynamic_pointer_cast<Head>(blocks["head"]);

            // patch_embedding
            x = patch_embedding->forward(ctx, x);                                                    // [N*dim, t_len, h_len, w_len]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]

            // time_embedding
            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, params.freq_dim);
            e      = time_embedding_0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = time_embedding_2->forward(ctx, e);  // [N, dim] or [N, T, dim]

            // time_projection
            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = time_projection_1->forward(ctx, e0);
            e0      = ggml_reshape_4d(ctx->ggml_ctx, e0, e0->ne[0] / 6, 6, e0->ne[1], e0->ne[2]);  //  [N, 6, dim] or [N, T, 6, dim]

            context = text_embedding_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context, false, ctx->backend);
            context = text_embedding_2->forward(ctx, context);  // [N, context_txt_len, dim]

            int64_t context_img_len = 0;
            if (clip_fea != nullptr) {
                if (params.model_type == "i2v") {
                    auto img_emb     = std::dynamic_pointer_cast<MLPProj>(blocks["img_emb"]);
                    auto context_img = img_emb->forward(ctx, clip_fea);                      // [N, context_img_len, dim]
                    context          = ggml_concat(ctx->ggml_ctx, context_img, context, 1);  // [N, context_img_len + context_txt_len, dim]
                }
                context_img_len = clip_fea->ne[1];  // 257
            }

            // vace_patch_embedding
            ggml_tensor* c = nullptr;
            if (params.vace_layers > 0) {
                auto vace_patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["vace_patch_embedding"]);

                c = vace_patch_embedding->forward(ctx, vace_context);                                    // [N*dim, t_len, h_len, w_len]
                c = ggml_reshape_3d(ctx->ggml_ctx, c, c->ne[0] * c->ne[1] * c->ne[2], c->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
                c = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, c, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]
            }

            bool use_sp_mainline = wan_sp_enabled(ctx);
            edgedit::parallel::SPSequenceSplit x_sp_split;
            if (use_sp_mainline) {
                const int rank       = wan_sp_rank(ctx);
                const int world_size = wan_sp_world_size(ctx);
                const int64_t x_pad  = edgedit::parallel::sp_sequence_padding(x->ne[1],
                                                                              world_size);
                if (params.model_type != "t2v" ||
                    params.vace_layers > 0 ||
                    clip_fea != nullptr ||
                    vace_context != nullptr ||
                    timestep->ne[0] != 1 ||
                    x->ne[2] != 1 ||
                    context == nullptr ||
                    context->ne[2] != 1 ||
                    params.num_heads % world_size != 0) {
                    LOG_WARN("wan SP mainline disabled: rank=%d world_size=%d model_type=%s x_seq=%" PRId64 " x_pad=%" PRId64 " heads=%" PRId64 " timestep_len=%" PRId64 " batch=[%" PRId64 ",%" PRId64 "] vace_layers=%d clip=%s vace=%s",
                             rank,
                             world_size,
                             params.model_type.c_str(),
                             x->ne[1],
                             x_pad,
                             params.num_heads,
                             timestep->ne[0],
                             x->ne[2],
                             context == nullptr ? 0 : context->ne[2],
                             params.vace_layers,
                             clip_fea == nullptr ? "null" : "set",
                             vace_context == nullptr ? "null" : "set");
                    use_sp_mainline = false;
                } else {
                    x_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                      x,
                                                                      rank,
                                                                      world_size,
                                                                      1,
                                                                      "wan_sp_x_split");
                    x = x_sp_split.local;
                    LOG_DEBUG("wan SP mainline enabled: rank=%d world_size=%d x_seq=%" PRId64 "->%" PRId64,
                              rank,
                              world_size,
                              x_sp_split.original_seq_len,
                              x_sp_split.local_seq_len);
                }
            }

            sd::ggml_graph_cut::mark_graph_cut(x, "wan.prelude", "x");
            sd::ggml_graph_cut::mark_graph_cut(e, "wan.prelude", "e");
            sd::ggml_graph_cut::mark_graph_cut(e0, "wan.prelude", "e0");
            sd::ggml_graph_cut::mark_graph_cut(context, "wan.prelude", "context");
            if (c != nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(c, "wan.prelude", "c");
            }

            auto x_orig = x;
            // Cache seam: the block stack transforms the single stream `x`.
            // The cached region is blocks [region_start, region_end); the default
            // whole-stack region matches the pre-region behaviour.
            // Substep tap: block-stack input anchor (ModelIn). Conditional no-op
            // unless requested.
            tap(ctx, edgedit::cache::AnchorRef::model_in(), x_orig);
            // Whole-stack device inject (MagCache GPU): skip the block loop and
            // reconstruct x_before + residual from the device slot after the loop.
            const bool whole_inject = ctx->tap_registry != nullptr &&
                                      ctx->tap_registry->has_stream_override();
            ggml_tensor* sp_prepared_pe = nullptr;
            if (use_sp_mainline && !whole_inject) {
                sp_prepared_pe = wan_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                  pe,
                                                                  "wan_sp_pe_seq_major");
                sd::ggml_graph_cut::mark_graph_cut(sp_prepared_pe, "wan.prelude", "pe_seq_major");
            }

            for (int i = 0; i < params.num_layers && !whole_inject; i++) {
                // Tap-driven inject (substep reuse): at the region start, replace the
                // stream with x_before + inject_input and jump past the region.
                // x_orig is the block-stack input (ModelIn).
                if (ctx->tap_registry != nullptr && ctx->tap_registry->inject_at(i)) {
                    x = build_tap_inject(ctx, x_orig);
                    i = ctx->tap_registry->inject_resume() - 1;
                    continue;
                }

                auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);

                // Expose the current block index to the sage attention fast path so
                // it can keep the first/last few layers in F16 (layer-skip policy).
                // Only Wan self-attention (square L_q==L_k) consumes this; cross-attn
                // (L_q!=L_k) is excluded inside ggml_ext_attention_ext.
                ctx->sage_layer_idx    = i;
                ctx->sage_total_layers = params.num_layers;

                if (use_sp_mainline) {
                    x = block->forward_sp(ctx,
                                          x,
                                          e0,
                                          pe,
                                          context,
                                          x_sp_split.pad,
                                          "wan_block" + std::to_string(i),
                                          context_img_len,
                                          sp_prepared_pe);
                } else
                {
                    x = block->forward(ctx,
                                       x,
                                       e0,
                                       pe,
                                       context,
                                       context_img_len,
                                       "wan_block" + std::to_string(i));
                }

                auto iter = params.vace_layers_mapping.find(i);
                if (iter != params.vace_layers_mapping.end()) {
                    int n = iter->second;

                    auto vace_block = std::dynamic_pointer_cast<VaceWanAttentionBlock>(blocks["vace_blocks." + std::to_string(n)]);

                    auto result = vace_block->forward(ctx, c, x_orig, e0, pe, context, context_img_len);
                    auto c_skip = result.first;
                    c           = result.second;
                    c_skip      = ggml_ext_scale(ctx->ggml_ctx, c_skip, vace_strength);
                    x           = ggml_add(ctx->ggml_ctx, x, c_skip);
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "wan.blocks." + std::to_string(i), "x");
                if (c != nullptr) {
                    sd::ggml_graph_cut::mark_graph_cut(c, "wan.blocks." + std::to_string(i), "c");
                }
                // Substep tap: block output k (BlockOut[i]) — the DiCache probe point.
                // Conditional no-op unless requested; also drives the substep probe stop.
                tap(ctx, edgedit::cache::AnchorRef::block_out(i), x);
                if (ctx->tap_registry != nullptr && ctx->tap_registry->stop_after(i)) {
                    return x;
                }
            }

            // Cache seam (whole-stack device path): the loop ran zero blocks, so
            // reconstruct x_before + residual from the device slot via the registry.
            if (whole_inject) {
                x = build_stream_override(ctx, x_orig);
            }
            // Substep tap: block-stack output anchor (ModelOut) — the residual's
            // "after" point (after the block loop, before head). Conditional no-op
            // unless requested.
            tap(ctx, edgedit::cache::AnchorRef::model_out(), x);

            if (use_sp_mainline && wan_sp_local_head_before_gather_enabled()) {
                x = head->forward(ctx, x, e);  // local [N, shard_tokens, pt*ph*pw*out_dim]

                auto x_gather = edgedit::parallel::sp_mark_gather_sequence(ctx->ggml_ctx,
                                                                           x,
                                                                           wan_sp_world_size(ctx),
                                                                           1,
                                                                           x_sp_split.pad,
                                                                           "wan_sp_final_head_gather",
                                                                           ctx->process_group);
                x             = x_gather.gathered;
            } else {
                if (use_sp_mainline) {
                    auto x_gather = edgedit::parallel::sp_mark_gather_sequence(ctx->ggml_ctx,
                                                                               x,
                                                                               wan_sp_world_size(ctx),
                                                                               1,
                                                                               x_sp_split.pad,
                                                                               "wan_sp_final_x_gather",
                                                                               ctx->process_group);
                    x             = x_gather.gathered;
                }

                x = head->forward(ctx, x, e);  // [N, t_len*h_len*w_len, pt*ph*pw*out_dim]
            }

            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* clip_fea        = nullptr,
                             ggml_tensor* time_dim_concat = nullptr,
                             ggml_tensor* vace_context    = nullptr,
                             float vace_strength          = 1.f,
                             int64_t N                    = 1) {
            // Forward pass of DiT.
            // x: [N*C, T, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // time_dim_concat: [N*C, T2, H, W]
            // return: [N*C, T, H, W]

            GGML_ASSERT(N == 1);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            int64_t C = x->ne[3];

            x = pad_to_patch_size(ctx, x);

            int64_t t_len = ((T + (std::get<0>(params.patch_size) / 2)) / std::get<0>(params.patch_size));
            int64_t h_len = ((H + (std::get<1>(params.patch_size) / 2)) / std::get<1>(params.patch_size));
            int64_t w_len = ((W + (std::get<2>(params.patch_size) / 2)) / std::get<2>(params.patch_size));

            if (time_dim_concat != nullptr) {
                time_dim_concat = pad_to_patch_size(ctx, time_dim_concat);
                x               = ggml_concat(ctx->ggml_ctx, x, time_dim_concat, 2);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]
                t_len           = ((x->ne[2] + (std::get<0>(params.patch_size) / 2)) / std::get<0>(params.patch_size));
            }

            auto out = forward_orig(ctx, x, timestep, context, pe, clip_fea, vace_context, vace_strength, N);  // [N, t_len*h_len*w_len, pt*ph*pw*C]

            out = unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]

            // slice

            out = ggml_ext_slice(ctx->ggml_ctx, out, 2, 0, T);  // [N*C, T, H + pad_h, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H);  // [N*C, T, H, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W);  // [N*C, T, H, W]

            return out;
        }
    };

    struct WanRunner : public GGMLRunner {
    public:
        std::string desc = "wan";
        WanParams wan_params;
        Wan wan;
        // Cross-step RoPE table memoization; see Rope::MemoizedPe. Replaces the
        // former bespoke pe_cache_* field set with the shared mechanism.
        Rope::MemoizedPe pe_memo_;
        SDVersion version;
        sd::Tensor<float> inject_feature_host_;  // kept alive across cache inject build

        const std::vector<float>& get_wan_pe_vec(int t, int h, int w) {
            const int pt = std::get<0>(wan_params.patch_size);
            const int ph = std::get<1>(wan_params.patch_size);
            const int pw = std::get<2>(wan_params.patch_size);
            std::vector<int64_t> pe_key;
            pe_key.reserve(8 + wan_params.axes_dim.size());
            pe_key.push_back(t);
            pe_key.push_back(h);
            pe_key.push_back(w);
            pe_key.push_back(pt);
            pe_key.push_back(ph);
            pe_key.push_back(pw);
            pe_key.push_back(static_cast<int64_t>(wan_params.theta));
            pe_key.push_back(-1);  // separator
            for (int d : wan_params.axes_dim) pe_key.push_back(d);
            return pe_memo_.get(std::move(pe_key), [&] {
                return Rope::gen_wan_pe(t, h, w, pt, ph, pw, 1, wan_params.theta, wan_params.axes_dim);
            });
        }

        WanRunner(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map = {},
                  const std::string prefix                       = "",
                  SDVersion version                              = VERSION_WAN2)
            : GGMLRunner(backend, offload_params_to_cpu) {
            wan_params.num_layers = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("vace_blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > wan_params.vace_layers) {
                            wan_params.vace_layers = block_index + 1;
                        }
                    }
                    continue;
                }
                pos = tensor_name.find("blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > wan_params.num_layers) {
                            wan_params.num_layers = block_index + 1;
                        }
                    }
                    continue;
                }
                if (tensor_name.find("img_emb") != std::string::npos) {
                    wan_params.model_type = "i2v";
                }
                if (tensor_name.find("img_emb.emb_pos") != std::string::npos) {
                    wan_params.flf_pos_embed_token_number = 514;
                }
            }

            if (wan_params.num_layers == 30) {
                if (version == VERSION_WAN2_2_TI2V) {
                    desc                 = "Wan2.2-TI2V-5B";
                    wan_params.dim       = 3072;
                    wan_params.eps       = 1e-06f;
                    wan_params.ffn_dim   = 14336;
                    wan_params.freq_dim  = 256;
                    wan_params.in_dim    = 48;
                    wan_params.num_heads = 24;
                    wan_params.out_dim   = 48;
                    wan_params.text_len  = 512;
                } else {
                    if (wan_params.vace_layers > 0) {
                        desc              = "Wan2.1-VACE-1.3B";
                        wan_params.in_dim = 16;
                    } else if (wan_params.model_type == "i2v") {
                        desc              = "Wan2.1-I2V-1.3B";
                        wan_params.in_dim = 36;
                    } else {
                        desc              = "Wan2.1-T2V-1.3B";
                        wan_params.in_dim = 16;
                    }
                    wan_params.dim       = 1536;
                    wan_params.eps       = 1e-06f;
                    wan_params.ffn_dim   = 8960;
                    wan_params.freq_dim  = 256;
                    wan_params.num_heads = 12;
                    wan_params.out_dim   = 16;
                    wan_params.text_len  = 512;
                }
            } else if (wan_params.num_layers == 40) {
                if (wan_params.model_type == "t2v") {
                    if (version == VERSION_WAN2_2_I2V) {
                        desc              = "Wan2.2-I2V-14B";
                        wan_params.in_dim = 36;
                    } else {
                        if (wan_params.vace_layers > 0) {
                            desc = "Wan2.x-VACE-14B";
                        } else {
                            desc = "Wan2.x-T2V-14B";
                        }
                        wan_params.in_dim = 16;
                    }
                } else {
                    wan_params.in_dim = 36;
                    if (wan_params.flf_pos_embed_token_number > 0) {
                        desc = "Wan2.1-FLF2V-14B";
                    } else {
                        desc = "Wan2.1-I2V-14B";
                    }
                }
                wan_params.dim       = 5120;
                wan_params.eps       = 1e-06f;
                wan_params.ffn_dim   = 13824;
                wan_params.freq_dim  = 256;
                wan_params.num_heads = 40;
                wan_params.out_dim   = 16;
                wan_params.text_len  = 512;
            } else {
                GGML_ABORT("invalid num_layers(%d) of wan", wan_params.num_layers);
            }

            LOG_INFO("%s", desc.c_str());

            wan = Wan(wan_params);
            wan.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return desc;
        }

        int64_t num_heads() const {
            return wan_params.num_heads;
        }

        int64_t head_dim() const {
            return wan_params.num_heads > 0 ? wan_params.dim / wan_params.num_heads : 0;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            wan.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor         = {},
                                 const sd::Tensor<float>& clip_fea_tensor        = {},
                                 const sd::Tensor<float>& c_concat_tensor        = {},
                                 const sd::Tensor<float>& time_dim_concat_tensor = {},
                                 const sd::Tensor<float>& vace_context_tensor    = {},
                                 float vace_strength                             = 1.f) {
            ggml_cgraph* gf = new_graph_custom(WAN_GRAPH_SIZE);

            ggml_tensor* x               = make_input(x_tensor);
            ggml_tensor* timesteps       = make_input(timesteps_tensor);
            ggml_tensor* context         = make_optional_input(context_tensor);
            if (context != nullptr) {
                ggml_set_name(context, "wan.context");
            }
            ggml_tensor* clip_fea        = make_optional_input(clip_fea_tensor);
            ggml_tensor* c_concat        = make_optional_input(c_concat_tensor);
            ggml_tensor* time_dim_concat = make_optional_input(time_dim_concat_tensor);
            ggml_tensor* vace_context    = make_optional_input(vace_context_tensor);

            const std::vector<float>& pe_host = get_wan_pe_vec(static_cast<int>(x->ne[2]),
                                                               static_cast<int>(x->ne[1]),
                                                               static_cast<int>(x->ne[0]));
            int pos_len = static_cast<int>(pe_host.size() / wan_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, wan_params.axes_dim_sum / 2, pos_len);
            ggml_set_name(pe, "pe");
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_host.data());

            if (c_concat != nullptr) {
                x = ggml_concat(compute_ctx, x, c_concat, 3);
            }

            auto runner_ctx = get_context();

            ggml_tensor* out = wan.forward(&runner_ctx,
                                           x,
                                           timesteps,
                                           context,
                                           pe,
                                           clip_fea,
                                           time_dim_concat,
                                           vace_context,
                                           vace_strength);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        // Measure DiT compute-buffer footprint at a target video latent size without
        // device alloc or weight load, for the auto-fit/auto-allocate scheduler. Mirrors
        // Flux::FluxRunner::measure_compute_buffer_at, but video adds the time/frame
        // dimension: x is 5D {W,H,T,C,1} and T=latent_frames drives the sequence length
        // (activation grows with t*h*w), so the frame count MUST be passed. latent_t
        // mirrors the pipeline's latent_frames(): ((frames-1)/4)+1. TI2V-5B uses per-frame
        // timesteps. i2v adds clip_fea cross-attn tokens. Returns 0 on failure.
        size_t measure_compute_buffer_at(int latent_w, int latent_h, int frames) {
            if (latent_w <= 0 || latent_h <= 0 || frames <= 0) {
                return 0;
            }
            const int latent_t = ((frames - 1) / 4) + 1;
            sd::Tensor<float> x = sd::zeros<float>(
                {latent_w, latent_h, latent_t, static_cast<int>(wan_params.in_dim), 1});
            sd::Tensor<float> timesteps = (version == VERSION_WAN2_2_TI2V)
                                              ? sd::zeros<float>({latent_t})
                                              : sd::zeros<float>({1});
            sd::Tensor<float> context = sd::zeros<float>(
                {static_cast<int>(wan_params.text_dim), static_cast<int>(wan_params.text_len), 1});
            sd::Tensor<float> clip_fea;
            if (wan_params.model_type == "i2v") {
                clip_fea = sd::zeros<float>({1280, 257, 1});
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, {}, {}, {}, 1.f);
            };
            return measure_compute_buffer(get_graph);
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context         = {},
                                  const sd::Tensor<float>& clip_fea        = {},
                                  const sd::Tensor<float>& c_concat        = {},
                                  const sd::Tensor<float>& time_dim_concat = {},
                                  const sd::Tensor<float>& vace_context    = {},
                                  float vace_strength                      = 1.f) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, time_dim_concat, vace_context, vace_strength);
            };

            // Experimental (ED_CACHE_COMPILED_GRAPHS): build the forward graph once
            // and reuse it across sampling steps, refreshing only the input bytes.
            // Gated to the plain non-segmented path — note this only fires when
            // params are NOT offloaded (params_backend==runtime_backend), no
            // --max-vram budget, and no SP; on a typical offload/SP config Wan runs
            // segmented and this is skipped. build_graph()'s make_input order is
            // {x, timesteps, context?, clip_fea?, c_concat?, time_dim_concat?,
            // vace_context?}; make_optional_input creates NO slot for an empty
            // tensor, so ordered_inputs lists only the present ones. pe is a
            // set_backend_tensor_data leaf (shape-keyed, step-invariant), not a
            // graph input, so it stays valid on a reused graph.
            const bool reuse_graphs =
                wan_env_flag_enabled_or_default("ED_CACHE_COMPILED_GRAPHS", false) &&
                !can_attempt_graph_cut_segmented_compute();
            if (reuse_graphs) {
                std::vector<const sd::Tensor<float>*> ordered_inputs;
                ordered_inputs.push_back(&x);
                ordered_inputs.push_back(&timesteps);
                if (!context.empty())         ordered_inputs.push_back(&context);
                if (!clip_fea.empty())        ordered_inputs.push_back(&clip_fea);
                if (!c_concat.empty())        ordered_inputs.push_back(&c_concat);
                if (!time_dim_concat.empty()) ordered_inputs.push_back(&time_dim_concat);
                if (!vace_context.empty())    ordered_inputs.push_back(&vace_context);
                return restore_trailing_singleton_dims(
                    GGMLRunner::compute_reuse<float>(get_graph, ordered_inputs, n_threads), x.dim());
            }

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        // ---- Substep-path tap-driven capture (host path — CPU/SP fallback for when
        // no device store is wired). The residual is read back to host (feature);
        // the device-slot counterpart (compute_substep_capture_slot below) keeps it
        // on-GPU. The runner weaves (ModelOut - ModelIn) from the taps. ----
        sd::DiffusionCacheResult compute_substep_capture(int n_threads,
                                                         const sd::Tensor<float>& x,
                                                         const sd::Tensor<float>& timesteps,
                                                         const sd::Tensor<float>& context,
                                                         const sd::Tensor<float>& clip_fea,
                                                         const sd::Tensor<float>& c_concat) {
            edgedit::cache::TapRegistry reg;
            reg.set_requested({edgedit::cache::AnchorRef::model_in(),
                               edgedit::cache::AnchorRef::model_out()});
            reg.set_capture_residual(true);
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {},
                                         nullptr, /*read_feature=*/true, /*read_taps=*/false);
            sd::DiffusionCacheResult out;
            out.output = std::move(pass.output);
            out.feature = std::move(pass.feature);
            return out;
        }

        // ---- Substep-path tap-driven probe (host path). Requests
        // ModelIn + BlockOut[m-1] taps, stops after m blocks, reads the before/probe
        // tensors back to host for the host DiCache metric. ----
        sd::DiffusionCacheResult compute_substep_probe(int n_threads,
                                                       const sd::Tensor<float>& x,
                                                       const sd::Tensor<float>& timesteps,
                                                       const sd::Tensor<float>& context,
                                                       const sd::Tensor<float>& clip_fea,
                                                       const sd::Tensor<float>& c_concat,
                                                       int probe_depth) {
            const int m = std::max(1, probe_depth);
            edgedit::cache::TapRegistry reg;
            const auto probe_anchor = edgedit::cache::AnchorRef::block_out(m - 1);
            const auto before_anchor = edgedit::cache::AnchorRef::model_in();
            reg.set_requested({before_anchor, probe_anchor});
            reg.set_stop_after(m - 1);
            // Record the anchor roles so run_substep_pass's read_taps returns them.
            reg.set_probe_metrics(probe_anchor, before_anchor, {});
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {},
                                         nullptr, /*read_feature=*/false, /*read_taps=*/true);
            sd::DiffusionCacheResult out;
            out.probe = pass.probe.empty() ? std::move(pass.output) : std::move(pass.probe);
            out.before = std::move(pass.before);
            return out;
        }

        // ---- Substep-path tap-driven inject (host reuse). Uploads
        // the host residual as a graph input and drives the forward's registry inject:
        // at the region start the stream becomes x_before + inject_input and the
        // region's blocks are skipped. ----
        sd::Tensor<float> compute_substep_inject(int n_threads,
                                                 const sd::Tensor<float>& x,
                                                 const sd::Tensor<float>& timesteps,
                                                 const sd::Tensor<float>& context,
                                                 const sd::Tensor<float>& clip_fea,
                                                 const sd::Tensor<float>& c_concat,
                                                 const sd::Tensor<float>& feature,
                                                 int region_start,
                                                 int region_end) {
            edgedit::cache::TapRegistry reg;
            inject_feature_host_ = feature;
            const int resume = region_end < 0 ? wan_params.num_layers : region_end;
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_tensor* inject_input = make_input(inject_feature_host_);
                reg.set_inject_host(inject_input, region_start, resume);
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {});
            return std::move(pass.output);
        }

        // ---- On-GPU device-slot MagCache path (parity with FluxRunner/MMDiTRunner).
        // The last block-stack residual stays on-device in a CacheStateManager slot
        // and is re-injected via build_stream_override with no host round-trip.
        // NOTE: Wan's video sequence is far longer than image models (tens of
        // thousands of tokens), so the resident device residual costs hundreds of MB
        // (MagCache) and the DiCache rings can reach several GB at 14B on long videos.
        // ----

        // Tap-driven device capture: the cache layer hands us GraphExtensions (a
        // DIFFERENCE weaving the residual + a slot to d2d it into); we request the
        // taps they reference, run the forward, then hand each CaptureToSlot result
        // off to its device slot. Named _slot to avoid clashing with the host
        // compute_substep_capture above (which returns DiffusionCacheResult).
        sd::Tensor<float> compute_substep_capture_slot(int n_threads,
                                                       const sd::Tensor<float>& x,
                                                       const sd::Tensor<float>& timesteps,
                                                       const sd::Tensor<float>& context,
                                                       const sd::Tensor<float>& clip_fea,
                                                       const sd::Tensor<float>& c_concat,
                                                       std::vector<edgedit::cache::GraphExtension> extensions) {
            edgedit::cache::TapRegistry reg;
            std::vector<edgedit::cache::AnchorRef> anchors;
            for (const auto& ext : extensions) {
                for (const auto& a : ext.input_anchors) {
                    anchors.push_back(a);
                }
            }
            reg.set_requested(anchors);
            reg.set_extensions(extensions);
            std::function<void()> handoff = [&]() {
                for (const auto& ext : extensions) {
                    if (ext.sink != edgedit::cache::GraphExtension::Sink::CaptureToSlot ||
                        !ext.alloc_slot) {
                        continue;
                    }
                    ggml_tensor* feat = get_cache_tensor_by_name(ext.output_name);
                    if (feat == nullptr) {
                        continue;
                    }
                    std::vector<int64_t> shape;
                    const int nd = std::max(1, ggml_n_dims(feat));
                    for (int i = 0; i < nd; ++i) {
                        shape.push_back(feat->ne[i]);
                    }
                    ggml_tensor* slot = static_cast<ggml_tensor*>(ext.alloc_slot(shape));
                    if (slot != nullptr && ggml_nbytes(slot) == ggml_nbytes(feat)) {
                        copy_named_cache_tensor_to(ext.output_name, slot);
                    }
                }
            };
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {}, handoff);
            return std::move(pass.output);
        }

        // Tap-driven device inject: the whole block stack is skipped and
        // x_before + residual is reconstructed via build_stream_override from the
        // device slot carried in the extension's extra_inputs.
        sd::Tensor<float> compute_substep_inject_slot(int n_threads,
                                                      const sd::Tensor<float>& x,
                                                      const sd::Tensor<float>& timesteps,
                                                      const sd::Tensor<float>& context,
                                                      const sd::Tensor<float>& clip_fea,
                                                      const sd::Tensor<float>& c_concat,
                                                      std::vector<edgedit::cache::GraphExtension> extensions) {
            if (extensions.empty()) {
                return {};
            }
            edgedit::cache::TapRegistry reg;
            const int resume = wan_params.num_layers;
            reg.set_extensions(std::move(extensions));
            reg.set_override_region(0, resume);
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {});
            return std::move(pass.output);
        }

        // ---- On-GPU DiCache path (parity with FluxRunner/MMDiTRunner/QwenImageRunner).
        // The probe metric (delta_y/delta_x/gamma) is computed on-device via cache-layer
        // indicator operators, and the residual/probe rings live in CacheStateManager
        // device slots (via DiCacheSlotBridge), blended on-GPU with no host round-trip.
        // ⚠️ Wan's DiCache device rings are far larger than the image models' (video
        // sequence -> 6 ring entries can reach several GB at 14B). ----

        // Refresh the cross-step residual/probe rings device-to-device from the
        // tap-woven nodes into CacheStateManager device slots via the bridge.
        // ⚠️ ORDER IS LOAD-BEARING: for the 2-deep rings we ROTATE FIRST, then alloc
        // the (new) newest head, then d2d — so read(slot,0)=newest, read(slot,1)=prev.
        // Do not reorder; a stale/off-by-one residual points here first.
        void writeback_to_slots(const edgedit::cache::DiCacheSlotBridge& bridge, int probe_depth) {
            if (!bridge.valid()) {
                return;
            }
            ggml_tensor* feat = get_cache_tensor_by_name("ed_cache_feature");
            if (feat == nullptr) {
                return;
            }
            std::vector<int64_t> shape;
            const int nd = std::max(1, ggml_n_dims(feat));
            for (int i = 0; i < nd; ++i) {
                shape.push_back(feat->ne[i]);
            }
            const int m = std::max(1, probe_depth);
            const std::string probe_tap = "ed_tap:block_out[" + std::to_string(m - 1) + "]";
            auto cp = [&](const char* name, void* dst) {
                ggml_tensor* d = static_cast<ggml_tensor*>(dst);
                if (d == nullptr || !copy_named_cache_tensor_to(name, d)) {
                    LOG_ERROR("dicache writeback copy failed: %s", name);
                }
            };
            // slot0 full residual ring (depth2): rotate, then write newest head.
            bridge.rotate(0);
            cp("ed_cache_feature", bridge.alloc(0, shape));
            // slot1 probe residual ring (depth2): rotate, then write newest head.
            bridge.rotate(1);
            cp("ed_cache_probe_resid", bridge.alloc(1, shape));
            // slot2 prev_probe / slot3 prev_input (depth1 snapshots): no rotate.
            cp(probe_tap.c_str(), bridge.alloc(2, shape));
            cp("ed_tap:model_in", bridge.alloc(3, shape));
        }

        // DiCache seed capture (device): a full forward whose post-readback refreshes
        // the residual/probe rings device-to-device via the bridge.
        sd::Tensor<float> compute_substep_capture_probe(int n_threads,
                                                        const sd::Tensor<float>& x,
                                                        const sd::Tensor<float>& timesteps,
                                                        const sd::Tensor<float>& context,
                                                        const sd::Tensor<float>& clip_fea,
                                                        const sd::Tensor<float>& c_concat,
                                                        int probe_depth,
                                                        const edgedit::cache::DiCacheSlotBridge& bridge) {
            const int m = std::max(1, probe_depth);
            edgedit::cache::TapRegistry reg;
            reg.set_requested({edgedit::cache::AnchorRef::model_in(),
                               edgedit::cache::AnchorRef::block_out(m - 1),
                               edgedit::cache::AnchorRef::model_out()});
            reg.set_capture_residual(true);   // weave ed_cache_feature = ModelOut - ModelIn
            reg.set_capture_writeback(m);     // weave ed_cache_probe_resid = block_out(m-1) - ModelIn
            std::function<void()> handoff = [&]() { writeback_to_slots(bridge, m); };
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {}, handoff);
            return std::move(pass.output);
        }

        // DiCache probe (device): stops after m blocks and weaves delta_y/delta_x/gamma
        // decision scalars on-device from the taps + persistent ring operands. Distinct
        // from the host compute_substep_probe(..., int probe_depth) overload above.
        sd::DiffusionCacheResult compute_substep_probe(int n_threads,
                                                       const sd::Tensor<float>& x,
                                                       const sd::Tensor<float>& timesteps,
                                                       const sd::Tensor<float>& context,
                                                       const sd::Tensor<float>& clip_fea,
                                                       const sd::Tensor<float>& c_concat,
                                                       int probe_depth,
                                                       const void* branch_key,
                                                       bool delta_minus,
                                                       const edgedit::cache::CacheOperatorRegistry& operators,
                                                       const edgedit::cache::DiCacheSlotBridge& bridge) {
            (void)branch_key;
            const int m = std::max(1, probe_depth);
            edgedit::cache::TapRegistry reg;
            const auto probe_anchor = edgedit::cache::AnchorRef::block_out(m - 1);
            const auto before_anchor = edgedit::cache::AnchorRef::model_in();
            reg.set_requested({before_anchor, probe_anchor});
            reg.set_stop_after(m - 1);

            // Decision-metric extensions the runner weaves. Probe-history device
            // operands come from CacheStateManager slots via the bridge:
            //   slot2 prev_probe (depth1), slot3 prev_input (depth1),
            //   slot1 probe-residual ring (depth0=newest, depth1=prev).
            // ⚠️ depths 0/1, NOT 1/2 (rotate-first writeback; see DiCacheSlotBridge).
            std::vector<edgedit::cache::GraphExtension> exts;
            const bool history_ready = bridge.valid() && bridge.filled(2) >= 1;
            if (history_ready) {
                using edgedit::cache::GraphExtension;
                ggml_tensor* prev_probe = static_cast<ggml_tensor*>(bridge.read(2, 0));
                ggml_tensor* prev_input = static_cast<ggml_tensor*>(bridge.read(3, 0));
                // delta_y = rel_l1(probe, prev_probe)
                if (prev_probe != nullptr) {
                    GraphExtension dy;
                    dy.op = operators.find("cache.rel_l1");
                    dy.op_id = "cache.rel_l1";
                    dy.input_anchors = {probe_anchor};
                    dy.extra_inputs = {prev_probe};
                    dy.output_name = "cache_ind:delta_y";
                    dy.sink = GraphExtension::Sink::Indicator;
                    if (dy.op != nullptr) exts.push_back(std::move(dy));
                }
                if (delta_minus && prev_input != nullptr) {
                    // delta_x = rel_l1(before, prev_input)
                    GraphExtension dx;
                    dx.op = operators.find("cache.rel_l1");
                    dx.op_id = "cache.rel_l1";
                    dx.input_anchors = {before_anchor};
                    dx.extra_inputs = {prev_input};
                    dx.output_name = "cache_ind:delta_x";
                    dx.sink = GraphExtension::Sink::Indicator;
                    if (dx.op != nullptr) exts.push_back(std::move(dx));
                }
                if (bridge.filled(1) >= 2) {
                    // gamma = gamma_indicator(probe, before, probe_prev1, probe_prev2)
                    ggml_tensor* probe_prev1 = static_cast<ggml_tensor*>(bridge.read(1, 0));
                    ggml_tensor* probe_prev2 = static_cast<ggml_tensor*>(bridge.read(1, 1));
                    if (probe_prev1 != nullptr && probe_prev2 != nullptr) {
                        GraphExtension gm;
                        gm.op = operators.find("cache.gamma_indicator");
                        gm.op_id = "cache.gamma_indicator";
                        gm.input_anchors = {probe_anchor, before_anchor};
                        gm.extra_inputs = {probe_prev1, probe_prev2};
                        gm.output_name = "cache_ind:gamma";
                        gm.sink = GraphExtension::Sink::Indicator;
                        if (gm.op != nullptr) exts.push_back(std::move(gm));
                    }
                }
            }
            reg.set_extensions(std::move(exts));

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(),
                                         {"delta_y", "delta_x", "gamma"});
            sd::DiffusionCacheResult out;
            auto g = [&](const char* k) {
                auto it = pass.indicators.find(k);
                return it != pass.indicators.end() ? it->second
                                                   : std::numeric_limits<float>::quiet_NaN();
            };
            out.delta_y = g("delta_y");
            out.delta_x = g("delta_x");
            out.gamma = g("gamma");
            return out;
        }

        // DiCache reuse (device): gamma-blend extrapolation from the residual ring.
        sd::Tensor<float> compute_substep_inject_gpu(int n_threads,
                                                     const sd::Tensor<float>& x,
                                                     const sd::Tensor<float>& timesteps,
                                                     const sd::Tensor<float>& context,
                                                     const sd::Tensor<float>& clip_fea,
                                                     const sd::Tensor<float>& c_concat,
                                                     std::vector<edgedit::cache::GraphExtension> extensions,
                                                     const edgedit::cache::DiCacheSlotBridge& bridge) {
            if (!bridge.valid() || extensions.empty() || bridge.filled(0) < 2) {
                return {};  // not enough history yet -> lowering falls back to full
            }
            ggml_tensor* resid_newest = static_cast<ggml_tensor*>(bridge.read(0, 0));
            ggml_tensor* resid_prev = static_cast<ggml_tensor*>(bridge.read(0, 1));
            if (resid_newest == nullptr || resid_prev == nullptr) {
                return {};
            }
            edgedit::cache::TapRegistry reg;
            const int resume = wan_params.num_layers;
            extensions[0].extra_inputs = {resid_newest, resid_prev};
            reg.set_extensions(std::move(extensions));
            reg.set_override_region(0, resume);
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, {}, {}, 1.f);
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {});
            return std::move(pass.output);
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(200 * 1024 * 1024);  // 200 MB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // cpu f16: pass
                // cuda f16: pass
                // cpu q8_0: pass
                // auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 104, 60, 1, 16);
                // ggml_set_f32(x, 0.01f);
                auto x = sd::load_tensor_from_file_as_tensor<float>("wan_dit_x.bin");
                print_sd_tensor(x);

                std::vector<float> timesteps_vec(3, 1000.f);
                timesteps_vec[0] = 0.f;
                auto timesteps   = sd::Tensor<float>::from_vector(timesteps_vec);

                // auto context = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 512, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = sd::load_tensor_from_file_as_tensor<float>("wan_dit_context.bin");
                print_sd_tensor(context);
                // auto clip_fea = load_tensor_from_file(ctx, "wan_dit_clip_fea.bin");
                // print_ggml_tensor(clip_fea);

                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = compute(8, x, timesteps, context, {}, {}, {}, {}, 1.f);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("wan test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_F16;
            LOG_INFO("loading from '%s'", file_path.c_str());

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

            std::shared_ptr<WanRunner> wan = std::make_shared<WanRunner>(backend,
                                                                         false,
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         VERSION_WAN2_2_TI2V);

            wan->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            wan->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("wan model loaded");

            wan->test();
        }
    };

}  // namespace WAN

#endif  // __WAN_HPP__
