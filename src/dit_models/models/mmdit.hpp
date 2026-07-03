#ifndef __MMDIT_HPP__
#define __MMDIT_HPP__

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "backend/ggml/ggml_extend.hpp"
#include "parallel/sp_parallel.hpp"

#define MMDIT_GRAPH_SIZE 32768

// Matches the existing fused-qkv custom-op userdata ABI in ggml-cuda.
// Mode 6 is model-agnostic: it packs q/k/v for seq-to-head all-to-all.
struct MMDiTFusedQKVPackParams {
    uint64_t magic;
    int64_t txt_real_seq;
    int64_t img_real_seq;
    int64_t mode;
    int64_t txt_padded_seq;
    int64_t img_padded_seq;
    int64_t world_size;
    int64_t stream_index;
};

constexpr uint64_t MMDIT_FUSED_QKV_PACK_MAGIC = 0x5157454e46514b56ULL; // legacy fused-qkv custom-op tag

struct MMDiTSPQKVSendPack {
    ggml_tensor* send_flat = nullptr;
    ggml_tensor* mixed_send_flat = nullptr;
    int64_t head_dim = 0;
    int64_t heads = 0;
    int64_t shard_sequence = 0;
    int64_t batch = 1;
    int64_t pad = 0;
};

struct MMDiTSPQKVMeta {
    int64_t head_dim = 0;
    int64_t heads = 0;
    int64_t shard_sequence = 0;
    int64_t batch = 1;
    int64_t pad = 0;
};

static inline std::vector<ggml_tensor*> mmdit_sp_split_qkv_qk_cont_v_view(ggml_context* ctx,
                                                                          ggml_tensor* qkv,
                                                                          int64_t num_heads) {
    GGML_ASSERT(qkv != nullptr);
    GGML_ASSERT(qkv->ne[0] % 3 == 0);
    const int64_t qkv_dim = qkv->ne[0] / 3;
    GGML_ASSERT(qkv_dim % num_heads == 0);
    const int64_t head_dim = qkv_dim / num_heads;
    const int64_t seq      = qkv->ne[1];
    const int64_t batch    = qkv->ne[2];
    const size_t plane_stride = qkv_dim * qkv->nb[0];
    const size_t head_stride  = head_dim * qkv->nb[0];

    auto q_view = ggml_view_4d(ctx, qkv, head_dim, num_heads, seq, batch, head_stride, qkv->nb[1], qkv->nb[2], plane_stride * 0);
    auto k_view = ggml_view_4d(ctx, qkv, head_dim, num_heads, seq, batch, head_stride, qkv->nb[1], qkv->nb[2], plane_stride * 1);
    auto v_view = ggml_view_4d(ctx, qkv, head_dim, num_heads, seq, batch, head_stride, qkv->nb[1], qkv->nb[2], plane_stride * 2);

    auto q = ggml_cont(ctx, q_view);
    auto k = ggml_cont(ctx, k_view);
    return {q, k, v_view};
}

static inline std::vector<ggml_tensor*> mmdit_split_qkv_qk_head_v_seq_view(ggml_context* ctx,
                                                                           ggml_tensor* qkv,
                                                                           int64_t num_heads) {
    GGML_ASSERT(qkv != nullptr);
    GGML_ASSERT(qkv->ne[0] % 3 == 0);
    const int64_t qkv_dim = qkv->ne[0] / 3;
    GGML_ASSERT(qkv_dim % num_heads == 0);
    const int64_t head_dim = qkv_dim / num_heads;
    const int64_t seq      = qkv->ne[1];
    const int64_t batch    = qkv->ne[2];
    const size_t plane_stride = qkv_dim * qkv->nb[0];
    const size_t head_stride  = head_dim * qkv->nb[0];

    auto q_head = ggml_view_4d(ctx, qkv, head_dim, num_heads, seq, batch, head_stride, qkv->nb[1], qkv->nb[2], plane_stride * 0);
    auto k_head = ggml_view_4d(ctx, qkv, head_dim, num_heads, seq, batch, head_stride, qkv->nb[1], qkv->nb[2], plane_stride * 1);
    auto v_seq  = ggml_view_3d(ctx, qkv, qkv_dim, seq, batch, qkv->nb[1], qkv->nb[2], plane_stride * 2);
    return {q_head, k_head, v_seq};
}

static inline std::vector<ggml_tensor*> mmdit_split_qkv_plane_views(ggml_context* ctx,
                                                                    ggml_tensor* qkv) {
    GGML_ASSERT(qkv != nullptr);
    GGML_ASSERT(qkv->ne[0] % 3 == 0);
    const int64_t qkv_dim = qkv->ne[0] / 3;
    const int64_t seq     = qkv->ne[1];
    const int64_t batch   = qkv->ne[2];
    const size_t plane_stride = qkv_dim * qkv->nb[0];

    auto q = ggml_view_3d(ctx, qkv, qkv_dim, seq, batch, qkv->nb[1], qkv->nb[2], plane_stride * 0);
    auto k = ggml_view_3d(ctx, qkv, qkv_dim, seq, batch, qkv->nb[1], qkv->nb[2], plane_stride * 1);
    auto v = ggml_view_3d(ctx, qkv, qkv_dim, seq, batch, qkv->nb[1], qkv->nb[2], plane_stride * 2);
    return {q, k, v};
}

static std::mutex& mmdit_fused_qkv_pack_params_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::vector<std::unique_ptr<MMDiTFusedQKVPackParams>>& mmdit_fused_qkv_pack_params_store() {
    static std::vector<std::unique_ptr<MMDiTFusedQKVPackParams>> store;
    return store;
}

static MMDiTFusedQKVPackParams* mmdit_fused_qkv_pack_make_params(int64_t world_size, int64_t mode) {
    auto params = std::make_unique<MMDiTFusedQKVPackParams>();
    params->magic          = MMDIT_FUSED_QKV_PACK_MAGIC;
    // The CUDA mode-6 path reads txt_real_seq as world_size.
    params->txt_real_seq   = world_size;
    params->img_real_seq   = 0;
    params->mode           = mode;
    params->txt_padded_seq = 0;
    params->img_padded_seq = 0;
    params->world_size     = 0;
    params->stream_index   = 0;
    MMDiTFusedQKVPackParams* raw = params.get();
    std::lock_guard<std::mutex> lock(mmdit_fused_qkv_pack_params_mutex());
    mmdit_fused_qkv_pack_params_store().push_back(std::move(params));
    return raw;
}

static MMDiTFusedQKVPackParams* mmdit_fused_qkv_pack_make_params(int64_t txt_real_seq,
                                                                  int64_t img_real_seq,
                                                                  int64_t mode,
                                                                  int64_t txt_padded_seq,
                                                                  int64_t img_padded_seq,
                                                                  int64_t world_size,
                                                                  int64_t stream_index = 0) {
    auto params = std::make_unique<MMDiTFusedQKVPackParams>();
    params->magic          = MMDIT_FUSED_QKV_PACK_MAGIC;
    params->txt_real_seq   = txt_real_seq;
    params->img_real_seq   = img_real_seq;
    params->mode           = mode;
    params->txt_padded_seq = txt_padded_seq;
    params->img_padded_seq = img_padded_seq;
    params->world_size     = world_size;
    params->stream_index   = stream_index;
    MMDiTFusedQKVPackParams* raw = params.get();
    std::lock_guard<std::mutex> lock(mmdit_fused_qkv_pack_params_mutex());
    mmdit_fused_qkv_pack_params_store().push_back(std::move(params));
    return raw;
}

static inline float mmdit_fused_qkv_get_f32(const ggml_tensor* src,
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

static inline void mmdit_fused_qkv_set_f32(ggml_tensor* dst,
                                           int64_t i0,
                                           int64_t i1,
                                           int64_t i2,
                                           int64_t i3,
                                           float value) {
    char* data = static_cast<char*>(dst->data) +
                 i0 * dst->nb[0] +
                 i1 * dst->nb[1] +
                 i2 * dst->nb[2] +
                 i3 * dst->nb[3];
    *reinterpret_cast<float*>(data) = value;
}

static inline void mmdit_fused_qkv_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
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
        int64_t rem             = linear;
        const int64_t d_all     = rem % total_head_dim;
        rem /= total_head_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq       = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer      = rem;
        const int64_t head      = head_local + peer * shard_heads;
        const int64_t plane     = d_all / head_dim;
        const int64_t d         = d_all - plane * head_dim;
        const ggml_tensor* src  = plane == 0 ? q : (plane == 1 ? k : v);
        const float value       = mmdit_fused_qkv_get_f32(src, d, head, seq, 0);
        mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0, value);
    }
}

static inline ggml_tensor* mmdit_fused_qkv_send_pack(ggml_context* ctx,
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
    ggml_tensor* args[] = {q, k, v};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      flat_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      3,
                                      mmdit_fused_qkv_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(world_size, 6));
    ggml_set_name(out, "mmdit.fused_qkv_send_pack.out");
    return out;
}

static inline void mmdit_fused_qkv_send_pack_mixed_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 18);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
    const int64_t world_size     = params->txt_real_seq;
    const int64_t head_dim       = q->ne[0];
    const int64_t heads          = q->ne[1];
    const int64_t shard_sequence = q->ne[2];
    const int64_t shard_heads    = heads / world_size;
    const int64_t packed_dim     = head_dim * 2;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(k->ne[0] == head_dim && v->ne[0] == head_dim);
    GGML_ASSERT(k->ne[1] == heads && v->ne[1] == heads);
    GGML_ASSERT(k->ne[2] == shard_sequence && v->ne[2] == shard_sequence);
    GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

    const int64_t total = dst->ne[0];
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem             = linear;
        const int64_t d_all     = rem % packed_dim;
        rem /= packed_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq       = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer      = rem;
        const int64_t head      = head_local + peer * shard_heads;

        char* dst_data = static_cast<char*>(dst->data) + linear * dst->nb[0];
        if (d_all < head_dim) {
            const float value = mmdit_fused_qkv_get_f32(q, d_all, head, seq, 0);
            *reinterpret_cast<uint32_t*>(dst_data) = *reinterpret_cast<const uint32_t*>(&value);
        } else {
            const int64_t d = d_all - head_dim;
            const float k_value = mmdit_fused_qkv_get_f32(k, d, head, seq, 0);
            const float v_value = mmdit_fused_qkv_get_f32(v, d, head, seq, 0);
            const uint32_t k_half = static_cast<uint16_t>(ggml_fp32_to_fp16(k_value));
            const uint32_t v_half = static_cast<uint16_t>(ggml_fp32_to_fp16(v_value));
            *reinterpret_cast<uint32_t*>(dst_data) = k_half | (v_half << 16);
        }
    }
}

static inline ggml_tensor* mmdit_fused_qkv_send_pack_mixed(ggml_context* ctx,
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

    const int64_t packed_dim  = q->ne[0] * 2;
    const int64_t shard_heads = q->ne[1] / world_size;
    const int64_t flat_elems  = packed_dim * shard_heads * q->ne[2] * world_size;
    ggml_tensor* args[] = {q, k, v};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      flat_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      3,
                                      mmdit_fused_qkv_send_pack_mixed_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(world_size, 18));
    ggml_set_name(out, "mmdit.fused_qkv_send_pack_mixed.out");
    return out;
}

static inline ggml_tensor* mmdit_sp_peer_concat_flat(ggml_context* ctx,
                                                     ggml_tensor* first,
                                                     ggml_tensor* second,
                                                     int64_t first_chunk,
                                                     int64_t second_chunk,
                                                     int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(first != nullptr && second != nullptr);
    GGML_ASSERT(first_chunk > 0 && second_chunk > 0 && world_size > 0);
    GGML_ASSERT(first->ne[0] == first_chunk * world_size);
    GGML_ASSERT(second->ne[0] == second_chunk * world_size);

    ggml_tensor* first_peer_chunks  = ggml_reshape_2d(ctx, first, first_chunk, world_size);
    ggml_tensor* second_peer_chunks = ggml_reshape_2d(ctx, second, second_chunk, world_size);
    ggml_tensor* joined            = ggml_concat(ctx, first_peer_chunks, second_peer_chunks, 0);
    return ggml_reshape_1d(ctx, joined, (first_chunk + second_chunk) * world_size);
}

static inline MMDiTSPQKVMeta mmdit_full_qkv_meta(const std::vector<ggml_tensor*>& qkv,
                                                 int64_t num_heads,
                                                 int world_size) {
    GGML_ASSERT(qkv.size() == 3);
    GGML_ASSERT(qkv[0] != nullptr && qkv[1] != nullptr && qkv[2] != nullptr);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(qkv[0]->ne[0] == qkv[1]->ne[0] && qkv[0]->ne[0] == qkv[2]->ne[0]);
    GGML_ASSERT(qkv[0]->ne[1] == qkv[1]->ne[1] && qkv[0]->ne[1] == qkv[2]->ne[1]);
    GGML_ASSERT(qkv[0]->ne[2] == qkv[1]->ne[2] && qkv[0]->ne[2] == qkv[2]->ne[2]);

    const int64_t dim = qkv[0]->ne[0];
    const int64_t seq = qkv[0]->ne[1];
    MMDiTSPQKVMeta meta;
    GGML_ASSERT(dim % num_heads == 0);
    meta.head_dim = dim / num_heads;
    meta.heads = num_heads;
    meta.batch = qkv[0]->ne[2];
    meta.pad = (world_size - (seq % world_size)) % world_size;
    meta.shard_sequence = (seq + meta.pad) / world_size;
    return meta;
}

static inline void mmdit_fused_two_stream_flat_send_pack_cpu(ggml_tensor* dst,
                                                             int ith,
                                                             int nth,
                                                             void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 9);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* first  = dst->src[0];
    const ggml_tensor* second = dst->src[1];
    GGML_ASSERT(first != nullptr && second != nullptr);
    GGML_ASSERT(first->type == GGML_TYPE_F32 && second->type == GGML_TYPE_F32);
    const int64_t first_chunk    = params->txt_real_seq;
    const int64_t second_chunk   = params->img_real_seq;
    const int64_t world_size     = params->world_size;
    const int64_t count_per_peer = first_chunk + second_chunk;
    GGML_ASSERT(first_chunk > 0 && second_chunk > 0 && world_size > 0);
    GGML_ASSERT(first->ne[0] == first_chunk * world_size);
    GGML_ASSERT(second->ne[0] == second_chunk * world_size);
    GGML_ASSERT(dst->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
        const int64_t peer = linear / count_per_peer;
        const int64_t rem  = linear - peer * count_per_peer;
        if (rem < first_chunk) {
            const char* data = static_cast<const char*>(first->data);
            const int64_t src_idx = peer * first_chunk + rem;
            mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0,
                                    *reinterpret_cast<const float*>(data + src_idx * first->nb[0]));
        } else {
            const char* data = static_cast<const char*>(second->data);
            const int64_t src_idx = peer * second_chunk + (rem - first_chunk);
            mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0,
                                    *reinterpret_cast<const float*>(data + src_idx * second->nb[0]));
        }
    }
}

static inline void mmdit_fused_two_stream_flat_recv_unpack_cpu(ggml_tensor* dst,
                                                               int ith,
                                                               int nth,
                                                               void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 10);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
    const int64_t first_chunk    = params->txt_real_seq;
    const int64_t second_chunk   = params->img_real_seq;
    const int64_t world_size     = params->world_size;
    const int64_t stream_index   = params->stream_index;
    const int64_t stream_chunk   = stream_index == 0 ? first_chunk : second_chunk;
    const int64_t stream_offset  = stream_index == 0 ? 0 : first_chunk;
    const int64_t count_per_peer = first_chunk + second_chunk;
    GGML_ASSERT(first_chunk > 0 && second_chunk > 0 && world_size > 0);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);
    GGML_ASSERT(dst->ne[0] == stream_chunk * world_size);

    for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
        const int64_t peer = linear / stream_chunk;
        const int64_t rem  = linear - peer * stream_chunk;
        const int64_t src_idx = peer * count_per_peer + stream_offset + rem;
        const char* data = static_cast<const char*>(recv_flat->data);
        mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0,
                                *reinterpret_cast<const float*>(data + src_idx * recv_flat->nb[0]));
    }
}

static inline ggml_tensor* mmdit_fused_two_stream_flat_send_pack(ggml_context* ctx,
                                                                 ggml_tensor* first,
                                                                 ggml_tensor* second,
                                                                 int64_t first_chunk,
                                                                 int64_t second_chunk,
                                                                 int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(first != nullptr && second != nullptr);
    GGML_ASSERT(first->type == GGML_TYPE_F32 && second->type == GGML_TYPE_F32);
    GGML_ASSERT(first_chunk > 0 && second_chunk > 0 && world_size > 0);
    GGML_ASSERT(first->ne[0] == first_chunk * world_size);
    GGML_ASSERT(second->ne[0] == second_chunk * world_size);
    ggml_tensor* args[] = {first, second};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      (first_chunk + second_chunk) * world_size,
                                      1,
                                      1,
                                      1,
                                      args,
                                      2,
                                      mmdit_fused_two_stream_flat_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(first_chunk,
                                                                      second_chunk,
                                                                      9,
                                                                      0,
                                                                      0,
                                                                      world_size));
    ggml_set_name(out, "mmdit.fused_two_stream_qkv_send_pack.out");
    return out;
}

static inline ggml_tensor* mmdit_fused_two_stream_flat_recv_unpack(ggml_context* ctx,
                                                                   ggml_tensor* recv_flat,
                                                                   int64_t stream_index,
                                                                   int64_t first_chunk,
                                                                   int64_t second_chunk,
                                                                   int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(first_chunk > 0 && second_chunk > 0 && world_size > 0);
    GGML_ASSERT(recv_flat->ne[0] == (first_chunk + second_chunk) * world_size);
    const int64_t out_chunk = stream_index == 0 ? first_chunk : second_chunk;
    ggml_tensor* args[] = {recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      out_chunk * world_size,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_two_stream_flat_recv_unpack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(first_chunk,
                                                                      second_chunk,
                                                                      10,
                                                                      0,
                                                                      0,
                                                                      world_size,
                                                                      stream_index));
    ggml_set_name(out, stream_index == 0 ? "mmdit.fused_two_stream_qkv_recv_unpack.first" :
                                           "mmdit.fused_two_stream_qkv_recv_unpack.second");
    return out;
}

static inline int64_t mmdit_fused_qkv_joint_src_index(int64_t d,
                                                      int64_t local_head,
                                                      int64_t local_seq,
                                                      int64_t peer,
                                                      int64_t plane,
                                                      int64_t head_dim,
                                                      int64_t shard_heads,
                                                      int64_t shard_sequence,
                                                      int64_t stream_offset,
                                                      int64_t count_per_peer) {
    const int64_t total_head_dim = head_dim * 3;
    return peer * count_per_peer +
           stream_offset +
           plane * head_dim +
           d +
           local_head * total_head_dim +
           local_seq * total_head_dim * shard_heads;
}

static inline void mmdit_fused_joint_qkv_to_seq_major_cpu(ggml_tensor* dst,
                                                          int ith,
                                                          int nth,
                                                          void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 11);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr && recv_flat->type == GGML_TYPE_F32);
    const int64_t context_real_seq = params->txt_real_seq;
    const int64_t x_real_seq       = params->img_real_seq;
    const int64_t context_full_seq = params->txt_padded_seq;
    const int64_t x_full_seq       = params->img_padded_seq;
    const int64_t world_size       = params->world_size;
    const int64_t plane            = params->stream_index;
    GGML_ASSERT(context_real_seq > 0 && x_real_seq > 0);
    GGML_ASSERT(context_full_seq >= context_real_seq && x_full_seq >= x_real_seq);
    GGML_ASSERT(world_size > 0 && context_full_seq % world_size == 0 && x_full_seq % world_size == 0);
    GGML_ASSERT(plane == 0 || plane == 1);

    const int64_t head_dim              = dst->ne[0];
    const int64_t total_real_seq        = context_real_seq + x_real_seq;
    const int64_t shard_heads           = dst->ne[2];
    const int64_t context_shard_seq     = context_full_seq / world_size;
    const int64_t x_shard_seq           = x_full_seq / world_size;
    const int64_t total_head_dim        = head_dim * 3;
    const int64_t context_chunk         = total_head_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk               = total_head_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer        = context_chunk + x_chunk;
    GGML_ASSERT(dst->ne[1] == total_real_seq && dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset = is_x ? context_chunk : 0;

        const int64_t peer       = stream_tok / stream_shard_seq;
        const int64_t local_seq  = stream_tok - peer * stream_shard_seq;
        const int64_t local_head = out_head;
        GGML_ASSERT(local_seq < stream_shard_seq);
        GGML_ASSERT(peer < world_size);

        const int64_t src_idx = mmdit_fused_qkv_joint_src_index(d,
                                                                local_head,
                                                                local_seq,
                                                                peer,
                                                                plane,
                                                                head_dim,
                                                                shard_heads,
                                                                stream_shard_seq,
                                                                stream_offset,
                                                                count_per_peer);
        const char* data = static_cast<const char*>(recv_flat->data);
        mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0,
                                *reinterpret_cast<const float*>(data + src_idx * recv_flat->nb[0]));
    }
}

static inline void mmdit_fused_joint_qkv_v_to_seq_major_cpu(ggml_tensor* dst,
                                                            int ith,
                                                            int nth,
                                                            void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 12);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr && recv_flat->type == GGML_TYPE_F32);
    const int64_t context_real_seq = params->txt_real_seq;
    const int64_t x_real_seq       = params->img_real_seq;
    const int64_t context_full_seq = params->txt_padded_seq;
    const int64_t x_full_seq       = params->img_padded_seq;
    const int64_t world_size       = params->world_size;
    GGML_ASSERT(context_real_seq > 0 && x_real_seq > 0);
    GGML_ASSERT(context_full_seq >= context_real_seq && x_full_seq >= x_real_seq);
    GGML_ASSERT(world_size > 0 && context_full_seq % world_size == 0 && x_full_seq % world_size == 0);

    const int64_t head_dim          = dst->ne[0];
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t shard_heads       = dst->ne[2];
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t total_head_dim    = head_dim * 3;
    const int64_t context_chunk     = total_head_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = total_head_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    GGML_ASSERT(dst->ne[1] == total_real_seq && dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset = is_x ? context_chunk : 0;

        const int64_t local_head = out_head;
        const int64_t local_seq  = stream_tok % stream_shard_seq;
        const int64_t peer       = stream_tok / stream_shard_seq;
        GGML_ASSERT(peer < world_size);

        const int64_t src_idx = mmdit_fused_qkv_joint_src_index(d,
                                                                local_head,
                                                                local_seq,
                                                                peer,
                                                                2,
                                                                head_dim,
                                                                shard_heads,
                                                                stream_shard_seq,
                                                                stream_offset,
                                                                count_per_peer);
        const char* data = static_cast<const char*>(recv_flat->data);
        mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0,
                                *reinterpret_cast<const float*>(data + src_idx * recv_flat->nb[0]));
    }
}

static inline ggml_tensor* mmdit_fused_joint_qkv_to_seq_major(ggml_context* ctx,
                                                              ggml_tensor* joint_recv_flat,
                                                              int64_t plane,
                                                              int64_t context_real_seq,
                                                              int64_t x_real_seq,
                                                              int64_t context_full_seq,
                                                              int64_t x_full_seq,
                                                              int64_t head_dim,
                                                              int64_t shard_heads,
                                                              int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(joint_recv_flat != nullptr && joint_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(plane == 0 || plane == 1);
    GGML_ASSERT(head_dim > 0 && shard_heads > 0);
    ggml_tensor* args[] = {joint_recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      context_real_seq + x_real_seq,
                                      shard_heads,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_joint_qkv_to_seq_major_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(context_real_seq,
                                                                      x_real_seq,
                                                                      11,
                                                                      context_full_seq,
                                                                      x_full_seq,
                                                                      world_size,
                                                                      plane));
    ggml_set_name(out, plane == 0 ? "mmdit.fused_joint_q_to_flash_layout" :
                                    "mmdit.fused_joint_k_to_flash_layout");
    return out;
}

static inline ggml_tensor* mmdit_fused_joint_qkv_v_to_seq_major(ggml_context* ctx,
                                                                ggml_tensor* joint_recv_flat,
                                                                int64_t context_real_seq,
                                                                int64_t x_real_seq,
                                                                int64_t context_full_seq,
                                                                int64_t x_full_seq,
                                                                int64_t head_dim,
                                                                int64_t shard_heads,
                                                                int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(joint_recv_flat != nullptr && joint_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(head_dim > 0 && shard_heads > 0);
    ggml_tensor* args[] = {joint_recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      context_real_seq + x_real_seq,
                                      shard_heads,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_joint_qkv_v_to_seq_major_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(context_real_seq,
                                                                      x_real_seq,
                                                                      12,
                                                                      context_full_seq,
                                                                      x_full_seq,
                                                                      world_size));
    ggml_set_name(out, "mmdit.fused_joint_v_to_flash_seq_layout");
    return out;
}

static inline int64_t mmdit_fused_joint_mixed_src_index(int64_t d_all,
                                                        int64_t local_head,
                                                        int64_t local_seq,
                                                        int64_t peer,
                                                        int64_t head_dim,
                                                        int64_t shard_heads,
                                                        int64_t stream_offset,
                                                        int64_t count_per_peer) {
    const int64_t packed_dim = head_dim * 2;
    return peer * count_per_peer +
           stream_offset +
           d_all +
           local_head * packed_dim +
           local_seq * packed_dim * shard_heads;
}

static inline void mmdit_fused_joint_mixed_q_to_seq_major_cpu(ggml_tensor* dst,
                                                              int ith,
                                                              int nth,
                                                              void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 19);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr && recv_flat->type == GGML_TYPE_F32);
    const int64_t context_real_seq = params->txt_real_seq;
    const int64_t x_real_seq       = params->img_real_seq;
    const int64_t context_full_seq = params->txt_padded_seq;
    const int64_t x_full_seq       = params->img_padded_seq;
    const int64_t world_size       = params->world_size;
    GGML_ASSERT(context_real_seq > 0 && x_real_seq > 0);
    GGML_ASSERT(context_full_seq >= context_real_seq && x_full_seq >= x_real_seq);
    GGML_ASSERT(world_size > 0 && context_full_seq % world_size == 0 && x_full_seq % world_size == 0);

    const int64_t head_dim          = dst->ne[0];
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t shard_heads       = dst->ne[2];
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t packed_dim        = head_dim * 2;
    const int64_t context_chunk     = packed_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = packed_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    GGML_ASSERT(dst->ne[1] == total_real_seq && dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok       = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset    = is_x ? context_chunk : 0;
        const int64_t peer             = stream_tok / stream_shard_seq;
        const int64_t local_seq        = stream_tok - peer * stream_shard_seq;
        const int64_t src_idx          = mmdit_fused_joint_mixed_src_index(d,
                                                                  out_head,
                                                                  local_seq,
                                                                  peer,
                                                                  head_dim,
                                                                  shard_heads,
                                                                  stream_offset,
                                                                  count_per_peer);
        const char* data = static_cast<const char*>(recv_flat->data);
        const uint32_t bits = *reinterpret_cast<const uint32_t*>(data + src_idx * recv_flat->nb[0]);
        float value;
        memcpy(&value, &bits, sizeof(value));
        mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0, value);
    }
}

static inline void mmdit_fused_joint_mixed_kv_to_seq_major_cpu(ggml_tensor* dst,
                                                               int ith,
                                                               int nth,
                                                               void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 20 || params->mode == 21);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr && recv_flat->type == GGML_TYPE_F32);
    const int64_t context_real_seq = params->txt_real_seq;
    const int64_t x_real_seq       = params->img_real_seq;
    const int64_t context_full_seq = params->txt_padded_seq;
    const int64_t x_full_seq       = params->img_padded_seq;
    const int64_t world_size       = params->world_size;
    GGML_ASSERT(context_real_seq > 0 && x_real_seq > 0);
    GGML_ASSERT(context_full_seq >= context_real_seq && x_full_seq >= x_real_seq);
    GGML_ASSERT(world_size > 0 && context_full_seq % world_size == 0 && x_full_seq % world_size == 0);

    const int64_t head_dim          = dst->ne[0];
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t shard_heads       = dst->ne[2];
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t packed_dim        = head_dim * 2;
    const int64_t context_chunk     = packed_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = packed_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    const bool unpack_v             = params->mode == 21;
    GGML_ASSERT(dst->ne[1] == total_real_seq && dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok       = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset    = is_x ? context_chunk : 0;
        const int64_t peer             = stream_tok / stream_shard_seq;
        const int64_t local_seq        = stream_tok - peer * stream_shard_seq;
        const int64_t src_idx          = mmdit_fused_joint_mixed_src_index(head_dim + d,
                                                                  out_head,
                                                                  local_seq,
                                                                  peer,
                                                                  head_dim,
                                                                  shard_heads,
                                                                  stream_offset,
                                                                  count_per_peer);
        const char* data = static_cast<const char*>(recv_flat->data);
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(data + src_idx * recv_flat->nb[0]);
        char* dst_data = static_cast<char*>(dst->data) + linear * dst->nb[0];
        *reinterpret_cast<ggml_fp16_t*>(dst_data) = static_cast<ggml_fp16_t>(unpack_v ? (packed >> 16) : (packed & 0xffffu));
    }
}

static inline ggml_tensor* mmdit_fused_joint_mixed_q_to_seq_major(ggml_context* ctx,
                                                                  ggml_tensor* joint_recv_flat,
                                                                  int64_t context_real_seq,
                                                                  int64_t x_real_seq,
                                                                  int64_t context_full_seq,
                                                                  int64_t x_full_seq,
                                                                  int64_t head_dim,
                                                                  int64_t shard_heads,
                                                                  int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(joint_recv_flat != nullptr && joint_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(head_dim > 0 && shard_heads > 0);
    ggml_tensor* args[] = {joint_recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      context_real_seq + x_real_seq,
                                      shard_heads,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_joint_mixed_q_to_seq_major_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(context_real_seq,
                                                                      x_real_seq,
                                                                      19,
                                                                      context_full_seq,
                                                                      x_full_seq,
                                                                      world_size));
    ggml_set_name(out, "mmdit.fused_joint_mixed_q_to_flash_layout");
    return out;
}

static inline ggml_tensor* mmdit_fused_joint_mixed_kv_to_seq_major(ggml_context* ctx,
                                                                   ggml_tensor* joint_recv_flat,
                                                                   bool unpack_v,
                                                                   int64_t context_real_seq,
                                                                   int64_t x_real_seq,
                                                                   int64_t context_full_seq,
                                                                   int64_t x_full_seq,
                                                                   int64_t head_dim,
                                                                   int64_t shard_heads,
                                                                   int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(joint_recv_flat != nullptr && joint_recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(head_dim > 0 && shard_heads > 0);
    ggml_tensor* args[] = {joint_recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      head_dim,
                                      context_real_seq + x_real_seq,
                                      shard_heads,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_joint_mixed_kv_to_seq_major_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(context_real_seq,
                                                                      x_real_seq,
                                                                      unpack_v ? 21 : 20,
                                                                      context_full_seq,
                                                                      x_full_seq,
                                                                      world_size));
    ggml_set_name(out, unpack_v ? "mmdit.fused_joint_mixed_v_to_flash_layout" :
                                  "mmdit.fused_joint_mixed_k_to_flash_layout");
    return out;
}

static inline void mmdit_fused_attn_head_to_seq_send_pack_cpu(ggml_tensor* dst,
                                                              int ith,
                                                              int nth,
                                                              void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 7);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* attn = dst->src[0];
    GGML_ASSERT(attn != nullptr);
    GGML_ASSERT(attn->type == GGML_TYPE_F32);
    const int64_t world_size     = params->world_size;
    const int64_t txt_real_seq   = params->txt_real_seq;
    const int64_t img_real_seq   = params->img_real_seq;
    const int64_t txt_padded_seq = params->txt_padded_seq;
    const int64_t img_padded_seq = params->img_padded_seq;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq && img_padded_seq >= img_real_seq);
    GGML_ASSERT(txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);
    GGML_ASSERT(attn->ne[2] == txt_real_seq + img_real_seq);
    GGML_ASSERT(attn->ne[3] == 1);

    const int64_t head_dim       = attn->ne[0];
    const int64_t shard_heads    = attn->ne[1];
    const int64_t txt_shard_seq  = txt_padded_seq / world_size;
    const int64_t img_shard_seq  = img_padded_seq / world_size;
    const int64_t txt_chunk      = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk      = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    GGML_ASSERT(dst->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < dst->ne[0]; linear += nth) {
        int64_t rem        = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const bool is_img = rem >= txt_chunk;
        if (is_img) {
            rem -= txt_chunk;
        }
        const int64_t shard_seq       = is_img ? img_shard_seq : txt_shard_seq;
        const int64_t stream_real_seq = is_img ? img_real_seq : txt_real_seq;
        const int64_t d               = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_tok  = rem;
        const int64_t stream_tok = peer * shard_seq + local_tok;
        float value              = 0.0f;
        if (stream_tok < stream_real_seq) {
            const int64_t total_tok = is_img ? txt_real_seq + stream_tok : stream_tok;
            value                   = mmdit_fused_qkv_get_f32(attn, d, head, total_tok, 0);
        }
        mmdit_fused_qkv_set_f32(dst, linear, 0, 0, 0, value);
    }
}

static inline void mmdit_fused_attn_head_to_seq_recv_unpack_cpu(ggml_tensor* dst,
                                                                int ith,
                                                                int nth,
                                                                void* userdata) {
    auto* params = static_cast<const MMDiTFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == MMDIT_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 8);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
    const int64_t world_size     = params->world_size;
    const int64_t stream_index   = params->stream_index;
    const int64_t txt_padded_seq = params->txt_padded_seq;
    const int64_t img_padded_seq = params->img_padded_seq;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(txt_padded_seq > 0 && img_padded_seq > 0);
    GGML_ASSERT(txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);

    const int64_t head_dim       = dst->ne[0];
    const int64_t heads          = dst->ne[1];
    const int64_t shard_heads    = heads / world_size;
    const int64_t txt_shard_seq  = txt_padded_seq / world_size;
    const int64_t img_shard_seq  = img_padded_seq / world_size;
    const int64_t out_shard_seq  = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t txt_chunk      = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk      = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_offset  = stream_index == 0 ? 0 : txt_chunk;
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(dst->ne[2] == out_shard_seq);
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < ggml_nelements(dst); linear += nth) {
        int64_t rem    = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_tok  = rem;
        const int64_t src_peer   = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx    = src_peer * count_per_peer +
                                stream_offset +
                                d +
                                local_head * head_dim +
                                local_tok * head_dim * shard_heads;
        const char* data  = static_cast<const char*>(recv_flat->data);
        const float value = *reinterpret_cast<const float*>(data + src_idx * recv_flat->nb[0]);
        mmdit_fused_qkv_set_f32(dst, d, head, local_tok, 0, value);
    }
}

static inline ggml_tensor* mmdit_fused_attn_head_to_seq_send_pack(ggml_context* ctx,
                                                                  ggml_tensor* attn_4d,
                                                                  int64_t txt_real_seq,
                                                                  int64_t img_real_seq,
                                                                  int64_t txt_padded_seq,
                                                                  int64_t img_padded_seq,
                                                                  int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(attn_4d != nullptr);
    GGML_ASSERT(attn_4d->type == GGML_TYPE_F32);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq && img_padded_seq >= img_real_seq);
    GGML_ASSERT(txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);
    GGML_ASSERT(attn_4d->ne[2] == txt_real_seq + img_real_seq);
    GGML_ASSERT(attn_4d->ne[3] == 1);

    const int64_t count_per_peer =
        attn_4d->ne[0] * attn_4d->ne[1] * (txt_padded_seq + img_padded_seq) / world_size;
    ggml_tensor* args[] = {attn_4d};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      count_per_peer * world_size,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(txt_real_seq,
                                                                      img_real_seq,
                                                                      7,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size));
    ggml_set_name(out, "mmdit.fused_attn_head_to_seq_send_pack.out");
    return out;
}

static inline ggml_tensor* mmdit_fused_attn_head_to_seq_recv_unpack(ggml_context* ctx,
                                                                    ggml_tensor* recv_flat,
                                                                    int64_t stream_index,
                                                                    int64_t txt_padded_seq,
                                                                    int64_t img_padded_seq,
                                                                    int64_t head_dim,
                                                                    int64_t heads,
                                                                    int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(head_dim > 0 && heads > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(txt_padded_seq > 0 && img_padded_seq > 0);
    GGML_ASSERT(txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);

    const int64_t out_seq = (stream_index == 0 ? txt_padded_seq : img_padded_seq) / world_size;
    const int64_t count_per_peer = head_dim * (heads / world_size) *
                                   ((txt_padded_seq + img_padded_seq) / world_size);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);
    ggml_tensor* args[] = {recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      heads,
                                      out_seq,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_attn_head_to_seq_recv_unpack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(0,
                                                                      0,
                                                                      8,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size,
                                                                      stream_index));
    ggml_set_name(out, stream_index == 0 ? "mmdit.fused_attn_head_to_seq_recv_unpack.context" :
                                           "mmdit.fused_attn_head_to_seq_recv_unpack.x");
    return out;
}

static inline ggml_tensor* mmdit_fused_attn_head_to_seq_send_pack_f16(ggml_context* ctx,
                                                                      ggml_tensor* attn_4d,
                                                                      int64_t txt_real_seq,
                                                                      int64_t img_real_seq,
                                                                      int64_t txt_padded_seq,
                                                                      int64_t img_padded_seq,
                                                                      int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(attn_4d != nullptr);
    GGML_ASSERT(attn_4d->type == GGML_TYPE_F32);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(txt_real_seq > 0 && img_real_seq > 0);
    GGML_ASSERT(txt_padded_seq >= txt_real_seq && img_padded_seq >= img_real_seq);
    GGML_ASSERT(txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);
    GGML_ASSERT(attn_4d->ne[2] == txt_real_seq + img_real_seq);
    GGML_ASSERT(attn_4d->ne[3] == 1);

    const int64_t count_per_peer =
        attn_4d->ne[0] * attn_4d->ne[1] * (txt_padded_seq + img_padded_seq) / world_size;
    ggml_tensor* args[] = {attn_4d};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      count_per_peer * world_size,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(txt_real_seq,
                                                                      img_real_seq,
                                                                      13,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size));
    ggml_set_name(out, "mmdit.fused_attn_head_to_seq_send_pack_f16.out");
    return out;
}

static inline ggml_tensor* mmdit_fused_attn_head_to_seq_recv_unpack_f16(ggml_context* ctx,
                                                                        ggml_tensor* recv_flat,
                                                                        int64_t stream_index,
                                                                        int64_t txt_padded_seq,
                                                                        int64_t img_padded_seq,
                                                                        int64_t head_dim,
                                                                        int64_t heads,
                                                                        int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(recv_flat->type == GGML_TYPE_F16);
    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(head_dim > 0 && heads > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(txt_padded_seq > 0 && img_padded_seq > 0);
    GGML_ASSERT(txt_padded_seq % world_size == 0 && img_padded_seq % world_size == 0);

    const int64_t out_seq = (stream_index == 0 ? txt_padded_seq : img_padded_seq) / world_size;
    ggml_tensor* args[] = {recv_flat};
    ggml_tensor* out    = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      head_dim,
                                      heads,
                                      out_seq,
                                      1,
                                      args,
                                      1,
                                      mmdit_fused_attn_head_to_seq_recv_unpack_cpu,
                                      GGML_N_TASKS_MAX,
                                      mmdit_fused_qkv_pack_make_params(0,
                                                                      0,
                                                                      14,
                                                                      txt_padded_seq,
                                                                      img_padded_seq,
                                                                      world_size,
                                                                      stream_index));
    ggml_set_name(out, stream_index == 0 ? "mmdit.fused_attn_head_to_seq_recv_unpack_f16.context" :
                                           "mmdit.fused_attn_head_to_seq_recv_unpack_f16.x");
    return out;
}

struct Mlp : public GGMLBlock {
public:
    Mlp(int64_t in_features,
        int64_t hidden_features = -1,
        int64_t out_features    = -1,
        bool bias               = true) {
        // act_layer is always lambda: nn.GELU(approximate="tanh")
        // norm_layer is always None
        // use_conv is always False
        if (hidden_features == -1) {
            hidden_features = in_features;
        }
        if (out_features == -1) {
            out_features = in_features;
        }
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_features, bias));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_features, out_features, bias));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, n_token, in_features]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct PatchEmbed : public GGMLBlock {
    // 2D Image to Patch Embedding
protected:
    bool flatten;
    bool dynamic_img_pad;
    int patch_size;

public:
    PatchEmbed(int64_t img_size     = 224,
               int patch_size       = 16,
               int64_t in_chans     = 3,
               int64_t embed_dim    = 1536,
               bool bias            = true,
               bool flatten         = true,
               bool dynamic_img_pad = true)
        : patch_size(patch_size),
          flatten(flatten),
          dynamic_img_pad(dynamic_img_pad) {
        // img_size is always None
        // patch_size is always 2
        // in_chans is always 16
        // norm_layer is always False
        // strict_img_size is always true, but not used

        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_chans,
                                                               embed_dim,
                                                               {patch_size, patch_size},
                                                               {patch_size, patch_size},
                                                               {0, 0},
                                                               {1, 1},
                                                               bias));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, C, H, W]
        // return: [N, H*W, embed_dim]
        auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

        if (dynamic_img_pad) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int pad_h = (patch_size - H % patch_size) % patch_size;
            int pad_w = (patch_size - W % patch_size) % patch_size;
            x         = ggml_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0);  // TODO: reflect pad mode
        }
        x = proj->forward(ctx, x);

        if (flatten) {
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        }
        return x;
    }
};

struct TimestepEmbedder : public GGMLBlock {
    // Embeds scalar timesteps into vector representations.
protected:
    int frequency_embedding_size;

public:
    TimestepEmbedder(int64_t hidden_size,
                     int frequency_embedding_size = 256,
                     int64_t out_channels         = 0)
        : frequency_embedding_size(frequency_embedding_size) {
        if (out_channels <= 0) {
            out_channels = hidden_size;
        }
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true, true));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, out_channels, true, true));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
        // t: [N, ]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        auto t_freq = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size);  // [N, frequency_embedding_size]

        auto t_emb = mlp_0->forward(ctx, t_freq);
        t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
        t_emb      = mlp_2->forward(ctx, t_emb);
        return t_emb;
    }
};

struct VectorEmbedder : public GGMLBlock {
    // Embeds a flat vector of dimension input_dim
public:
    VectorEmbedder(int64_t input_dim,
                   int64_t hidden_size) {
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(input_dim, hidden_size, true, true));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, true));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, input_dim]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        x = mlp_0->forward(ctx, x);
        x = ggml_silu_inplace(ctx->ggml_ctx, x);
        x = mlp_2->forward(ctx, x);
        return x;
    }
};

static inline bool mmdit_sp_enabled(GGMLRunnerContext* ctx) {
    return ctx != nullptr &&
           ctx->process_group != nullptr &&
           ctx->process_group->enabled() &&
           ctx->process_group->size() > 1;
}

static inline int mmdit_sp_rank(GGMLRunnerContext* ctx) {
    return ctx->process_group->rank();
}

static inline int mmdit_sp_world_size(GGMLRunnerContext* ctx) {
    return ctx->process_group->size();
}

static inline ggml_tensor* mmdit_sp_view_head_sequence(ggml_context* ctx,
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

static inline ggml_tensor* mmdit_sp_pad_head_sequence(ggml_context* ctx,
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

static inline ggml_tensor* mmdit_sp_concat_real_attention_sequence(ggml_context* ctx,
                                                                   ggml_tensor* context,
                                                                   ggml_tensor* x,
                                                                   int64_t context_pad,
                                                                   int64_t x_pad) {
    GGML_ASSERT(context != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(context_pad >= 0 && context_pad <= context->ne[2]);
    GGML_ASSERT(x_pad >= 0 && x_pad <= x->ne[2]);

    const int64_t context_real_seq = context->ne[2] - context_pad;
    const int64_t x_real_seq       = x->ne[2] - x_pad;
    GGML_ASSERT(context_real_seq > 0);
    GGML_ASSERT(x_real_seq > 0);

    ggml_tensor* context_real = mmdit_sp_view_head_sequence(ctx,
                                                            context,
                                                            0,
                                                            context_real_seq);
    ggml_tensor* x_real       = mmdit_sp_view_head_sequence(ctx,
                                                            x,
                                                            0,
                                                            x_real_seq);
    return ggml_concat(ctx, context_real, x_real, 2);
}

static inline ggml_tensor* mmdit_sp_concat_real_attention_sequence_seq_major(ggml_context* ctx,
                                                                             ggml_tensor* context,
                                                                             ggml_tensor* x,
                                                                             int64_t context_pad,
                                                                             int64_t x_pad) {
    GGML_ASSERT(context != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(context_pad >= 0 && context_pad <= context->ne[1]);
    GGML_ASSERT(x_pad >= 0 && x_pad <= x->ne[1]);

    const int64_t context_real_seq = context->ne[1] - context_pad;
    const int64_t x_real_seq       = x->ne[1] - x_pad;
    GGML_ASSERT(context_real_seq > 0);
    GGML_ASSERT(x_real_seq > 0);

    ggml_tensor* context_real = ggml_view_4d(ctx,
                                             context,
                                             context->ne[0],
                                             context_real_seq,
                                             context->ne[2],
                                             context->ne[3],
                                             context->nb[1],
                                             context->nb[2],
                                             context->nb[3],
                                             0);
    ggml_tensor* x_real       = ggml_view_4d(ctx,
                                             x,
                                             x->ne[0],
                                             x_real_seq,
                                             x->ne[2],
                                             x->ne[3],
                                             x->nb[1],
                                             x->nb[2],
                                             x->nb[3],
                                             0);
    return ggml_concat(ctx, context_real, x_real, 1);
}

static inline std::vector<ggml_tensor*> mmdit_sp_restore_qkv_from_recv(ggml_context* ctx,
                                                                       ggml_tensor* recv_flat,
                                                                       int64_t head_dim,
                                                                       int64_t heads,
                                                                       int64_t shard_sequence,
                                                                       int world_size) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT(head_dim > 0);
    GGML_ASSERT(heads > 0 && heads % world_size == 0);
    GGML_ASSERT(shard_sequence > 0);
    GGML_ASSERT(world_size > 0);

    const int64_t shard_heads    = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    ggml_tensor* mid             = ggml_reshape_4d(ctx,
                                       recv_flat,
                                       total_head_dim,
                                       shard_heads,
                                       shard_sequence,
                                       world_size);

    std::vector<ggml_tensor*> outputs;
    outputs.reserve(3);
    for (int i = 0; i < 3; ++i) {
        ggml_tensor* view = ggml_view_4d(ctx,
                                         mid,
                                         head_dim,
                                         shard_heads,
                                         shard_sequence,
                                         world_size,
                                         mid->nb[1],
                                         mid->nb[2],
                                         mid->nb[3],
                                         static_cast<size_t>(i) * static_cast<size_t>(head_dim) * ggml_type_size(mid->type));
        ggml_tensor* out = nullptr;
        if (i < 2) {
            out = ggml_cont_4d(ctx,
                               ggml_permute(ctx, view, 0, 3, 1, 2),
                               head_dim,
                               shard_sequence * world_size,
                               shard_heads,
                               1);
        } else {
            out = ggml_cont_4d(ctx,
                               view,
                               head_dim,
                               shard_heads,
                               shard_sequence * world_size,
                               1);
        }
        ggml_set_name(out, i == 0 ? "mmdit.qkv_recv.q_seq_major" :
                           i == 1 ? "mmdit.qkv_recv.k_seq_major" :
                                    "mmdit.qkv_recv.v_head_major");
        outputs.push_back(out);
    }
    return outputs;
}

static inline ggml_tensor* mmdit_sp_extract_stream_attention(ggml_context* ctx,
                                                            ggml_tensor* combined,
                                                            int64_t real_start,
                                                            int64_t real_seq,
                                                            int64_t pad) {
    ggml_tensor* out = mmdit_sp_view_head_sequence(ctx,
                                                   combined,
                                                   real_start,
                                                   real_seq);
    out              = mmdit_sp_pad_head_sequence(ctx, out, pad);
    return ggml_cont(ctx, out);
}

static inline ggml_tensor* mmdit_sp_qk_to_attention_layout(ggml_context* ctx,
                                                          ggml_tensor* x) {
    GGML_ASSERT(x != nullptr);
    ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, out, out->ne[0], out->ne[1], out->ne[2] * out->ne[3]);
}

static inline ggml_tensor* mmdit_attention(GGMLRunnerContext* ctx,
                                           ggml_tensor* q,
                                           ggml_tensor* k,
                                           ggml_tensor* v,
                                           int64_t n_head,
                                           bool skip_reshape,
                                           bool v_is_seq_major = false) {
    if (!v_is_seq_major) {
        return ggml_ext_attention_ext(ctx->ggml_ctx,
                                      ctx->backend,
                                      q,
                                      k,
                                      v,
                                      n_head,
                                      nullptr,
                                      skip_reshape,
                                      ctx->flash_attn_enabled);
    }
    return ggml_ext_attention_ext(ctx->ggml_ctx,
                                  ctx->backend,
                                  q,
                                  k,
                                  v,
                                  n_head,
                                  nullptr,
                                  skip_reshape,
                                  ctx->flash_attn_enabled,
                                  1.0f,
                                  true,
                                  true);
}

static inline bool mmdit_cudnn_unpadded_attention_enabled(GGMLRunnerContext* ctx,
                                                          int64_t total_seq,
                                                          int64_t head_dim) {
#ifdef ED_ENABLE_CUDNN_SDPA
    return ctx != nullptr &&
           ctx->flash_attn_enabled &&
           total_seq >= 4096 &&
           (head_dim == 64 || head_dim == 128) &&
           sd_backend_is(ctx->backend, "CUDA") &&
           !ggml_ext_env_flag_enabled("ED_DISABLE_CUDNN_SDPA") &&
           !ggml_ext_env_flag_enabled("ED_DISABLE_CUDNN_SDPA_UNPAD");
#else
    ED_UNUSED(ctx);
    ED_UNUSED(total_seq);
    ED_UNUSED(head_dim);
    return false;
#endif
}

static inline ggml_tensor* mmdit_fused_pair_pack_attention(GGMLRunnerContext* ctx,
                                                           const std::vector<ggml_tensor*>& context_qkv,
                                                           const std::vector<ggml_tensor*>& x_qkv,
                                                           int64_t n_head) {
    GGML_ASSERT(context_qkv.size() == 3 && x_qkv.size() == 3);
    GGML_ASSERT(context_qkv[0] != nullptr && context_qkv[1] != nullptr && context_qkv[2] != nullptr);
    GGML_ASSERT(x_qkv[0] != nullptr && x_qkv[1] != nullptr && x_qkv[2] != nullptr);
    const int64_t total_seq = context_qkv[0]->ne[1] + x_qkv[0]->ne[1];
    const int64_t head_dim = context_qkv[0]->ne[0] / n_head;
    if (!mmdit_cudnn_unpadded_attention_enabled(ctx, total_seq, head_dim)) {
        return nullptr;
    }

    ggml_tensor* q = nullptr;
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;

    if (auto qkv_pack = edgedit::ggml_ext::attention_qkv_pair_pack_custom_f16(ctx->ggml_ctx,
                                                                              context_qkv[0],
                                                                              context_qkv[1],
                                                                              context_qkv[2],
                                                                              x_qkv[0],
                                                                              x_qkv[1],
                                                                              x_qkv[2],
                                                                              n_head)) {
        q = ggml_view_3d(ctx->ggml_ctx, qkv_pack, head_dim, total_seq, n_head, qkv_pack->nb[1], qkv_pack->nb[2], 0);
        k = ggml_view_3d(ctx->ggml_ctx, qkv_pack, head_dim, total_seq, n_head, qkv_pack->nb[1], qkv_pack->nb[2], qkv_pack->nb[3]);
        v = ggml_view_3d(ctx->ggml_ctx, qkv_pack, head_dim, total_seq, n_head, qkv_pack->nb[1], qkv_pack->nb[2], qkv_pack->nb[3] * 2);
    } else {
        auto pack = [&](ggml_tensor* first, ggml_tensor* second) -> ggml_tensor* {
            return edgedit::ggml_ext::attention_pair_pack_custom_f16(ctx->ggml_ctx, first, second, n_head);
        };

        q = pack(context_qkv[0], x_qkv[0]);
        k = pack(context_qkv[1], x_qkv[1]);
        v = pack(context_qkv[2], x_qkv[2]);
    }
    if (q == nullptr || k == nullptr || v == nullptr) {
        return nullptr;
    }

    auto out = ggml_flash_attn_ext(ctx->ggml_ctx, q, k, v, nullptr, 1.0f / sqrt((float) head_dim), 0, 0);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_view_3d(ctx->ggml_ctx, out, head_dim, n_head, total_seq, out->nb[1], out->nb[2], 0);
    out = ggml_reshape_3d(ctx->ggml_ctx, out, head_dim * n_head, total_seq, 1);
    return out;
}

static inline ggml_tensor* mmdit_sp_attention_seq_major(GGMLRunnerContext* ctx,
                                                        ggml_tensor* q,
                                                        ggml_tensor* k,
                                                        ggml_tensor* v,
                                                        bool v_is_seq_major = false) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(q != nullptr);
    GGML_ASSERT(k != nullptr);
    GGML_ASSERT(v != nullptr);

    q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);
    k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);
    v = ggml_is_contiguous(v) ? v : ggml_cont(ctx->ggml_ctx, v);

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

    return mmdit_attention(ctx, q, k, v, n_head, true, v_is_seq_major);
}

static inline ggml_tensor* mmdit_sp_attention(GGMLRunnerContext* ctx,
                                              ggml_tensor* q,
                                              ggml_tensor* k,
                                              ggml_tensor* v) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(q != nullptr);
    GGML_ASSERT(k != nullptr);
    GGML_ASSERT(v != nullptr);

    q = mmdit_sp_qk_to_attention_layout(ctx->ggml_ctx, q);
    k = mmdit_sp_qk_to_attention_layout(ctx->ggml_ctx, k);
    v = ggml_is_contiguous(v) ? v : ggml_cont(ctx->ggml_ctx, v);

    const int64_t n_head = v->ne[1];
    GGML_ASSERT(q->ne[0] == k->ne[0]);
    GGML_ASSERT(q->ne[1] == k->ne[1]);
    GGML_ASSERT(q->ne[2] == k->ne[2]);
    GGML_ASSERT(v->ne[0] == q->ne[0]);
    GGML_ASSERT(v->ne[1] == n_head);
    GGML_ASSERT(v->ne[2] == q->ne[1]);
    GGML_ASSERT(v->ne[3] == 1);

    ggml_tensor* attn = mmdit_attention(ctx, q, k, v, n_head, true);
    return attn;
}

class SelfAttention : public GGMLBlock {
public:
    int64_t num_heads;
    bool pre_only;
    std::string qk_norm;

public:
    SelfAttention(int64_t dim,
                  int64_t num_heads   = 8,
                  std::string qk_norm = "",
                  bool qkv_bias       = false,
                  bool pre_only       = false)
        : num_heads(num_heads), pre_only(pre_only), qk_norm(qk_norm) {
        int64_t d_head = dim / num_heads;
        blocks["qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
        if (!pre_only) {
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }
        if (qk_norm == "rms") {
            blocks["ln_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(d_head, 1.0e-6f));
            blocks["ln_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(d_head, 1.0e-6f));
        } else if (qk_norm == "ln") {
            blocks["ln_q"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_head, 1.0e-6f));
            blocks["ln_k"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_head, 1.0e-6f));
        }
    }

    std::vector<ggml_tensor*> pre_attention(GGMLRunnerContext* ctx,
                                            ggml_tensor* x,
                                            bool pair_pack_plane_views = false) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);

        auto qkv = qkv_proj->forward(ctx, x);
        if (pair_pack_plane_views && qk_norm.empty()) {
            return mmdit_split_qkv_plane_views(ctx->ggml_ctx, qkv);
        }

        auto qkv_vec     = mmdit_split_qkv_qk_head_v_seq_view(ctx->ggml_ctx, qkv, num_heads);
        int64_t head_dim = qkv_vec[0]->ne[0];
        auto q           = qkv_vec[0];  // [N, n_token, n_head, d_head]
        auto k           = qkv_vec[1];  // [N, n_token, n_head, d_head]
        auto v           = qkv_vec[2];  // [N, n_token, n_head*d_head]

        if (qk_norm == "rms" || qk_norm == "ln") {
            auto ln_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_q"]);
            auto ln_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_k"]);
            q         = ln_q->forward(ctx, q);
            k         = ln_k->forward(ctx, k);
        } else {
            q = ggml_cont(ctx->ggml_ctx, q);
            k = ggml_cont(ctx->ggml_ctx, k);
        }

        q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0] * q->ne[1], q->ne[2], q->ne[3]);  // [N, n_token, n_head*d_head]
        k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0] * k->ne[1], k->ne[2], k->ne[3]);  // [N, n_token, n_head*d_head]

        return {q, k, v};
    }

    std::vector<ggml_tensor*> pre_attention_sp(GGMLRunnerContext* ctx,
                                               ggml_tensor* x,
                                               const std::string& name_prefix) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);

        auto qkv         = qkv_proj->forward(ctx, x);
        auto qkv_vec     = split_qkv(ctx->ggml_ctx, qkv);
        int64_t head_dim = qkv_vec[0]->ne[0] / num_heads;
        int64_t seq      = qkv_vec[0]->ne[1];
        int64_t N        = qkv_vec[0]->ne[2];
        GGML_ASSERT(N == 1);

        auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, seq, N);
        auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, seq, N);
        auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, seq, N);

        if (qk_norm == "rms" || qk_norm == "ln") {
            auto ln_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_q"]);
            auto ln_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_k"]);
            q         = ln_q->forward(ctx, q);
            k         = ln_k->forward(ctx, k);
        }

        auto qkv_head = edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched_mixed(
            ctx->ggml_ctx,
            {q, k, v},
            {true, true, false},
            mmdit_sp_world_size(ctx),
            name_prefix + "_qkv_seq_to_head");
        GGML_ASSERT(qkv_head.outputs.size() == 3);
        return qkv_head.outputs;
    }

    MMDiTSPQKVSendPack pre_attention_sp_send_pack(GGMLRunnerContext* ctx,
                                                  ggml_tensor* x,
                                                  const std::string& name_prefix) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);

        auto qkv         = qkv_proj->forward(ctx, x);
        auto qkv_vec     = mmdit_sp_split_qkv_qk_cont_v_view(ctx->ggml_ctx, qkv, num_heads);
        int64_t head_dim = qkv_vec[0]->ne[0];
        int64_t seq      = qkv_vec[0]->ne[2];
        int64_t N        = qkv_vec[0]->ne[3];
        GGML_ASSERT(N == 1);

        auto q = qkv_vec[0];
        auto k = qkv_vec[1];
        auto v = qkv_vec[2];

        if (qk_norm == "rms" || qk_norm == "ln") {
            auto ln_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_q"]);
            auto ln_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_k"]);
            q         = ln_q->forward(ctx, q);
            k         = ln_k->forward(ctx, k);
        }

        const int world_size = mmdit_sp_world_size(ctx);
        MMDiTSPQKVSendPack pack;
        pack.send_flat       = mmdit_fused_qkv_send_pack(ctx->ggml_ctx, q, k, v, world_size);
        pack.mixed_send_flat = mmdit_fused_qkv_send_pack_mixed(ctx->ggml_ctx, q, k, v, world_size);
        pack.head_dim        = head_dim;
        pack.heads           = num_heads;
        pack.shard_sequence  = seq;
        pack.batch           = N;
        ggml_set_name(pack.send_flat, (name_prefix + "_fused_qkv_send_pack").c_str());
        ggml_set_name(pack.mixed_send_flat, (name_prefix + "_fused_qkv_send_pack_mixed").c_str());
        return pack;
    }

    ggml_tensor* post_attention(GGMLRunnerContext* ctx, ggml_tensor* x) {
        GGML_ASSERT(!pre_only);

        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

        x = proj->forward(ctx, x);  // [N, n_token, dim]
        return x;
    }

    // x: [N, n_token, dim]
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x) {
        auto qkv = pre_attention(ctx, x);
        x        = mmdit_attention(ctx, qkv[0], qkv[1], qkv[2], num_heads, false);  // [N, n_token, dim]
        x        = post_attention(ctx, x);                                      // [N, n_token, dim]
        return x;
    }
};

__STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* shift,
                                        ggml_tensor* scale) {
    // x: [N, L, C]
    // scale: [N, C]
    // shift: [N, C]
    scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
    shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
    x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    x     = ggml_add(ctx, x, shift);
    return x;
}

struct DismantledBlock : public GGMLBlock {
    // A DiT block with gated adaptive layer norm (adaLN) conditioning.
public:
    int64_t num_heads;
    bool pre_only;
    bool self_attn;

public:
    DismantledBlock(int64_t hidden_size,
                    int64_t num_heads,
                    float mlp_ratio     = 4.0,
                    std::string qk_norm = "",
                    bool qkv_bias       = false,
                    bool pre_only       = false,
                    bool self_attn      = false)
        : num_heads(num_heads), pre_only(pre_only), self_attn(self_attn) {
        // rmsnorm is always Flase
        // scale_mod_only is always Flase
        // swiglu is always Flase
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qk_norm, qkv_bias, pre_only));

        if (self_attn) {
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qk_norm, qkv_bias, false));
        }

        if (!pre_only) {
            blocks["norm2"]        = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            int64_t mlp_hidden_dim = (int64_t)(hidden_size * mlp_ratio);
            blocks["mlp"]          = std::shared_ptr<GGMLBlock>(new Mlp(hidden_size, mlp_hidden_dim));
        }

        int64_t n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        if (self_attn) {
            n_mods = 9;
        }
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, n_mods * hidden_size));
    }

    std::tuple<std::vector<ggml_tensor*>, std::vector<ggml_tensor*>, std::vector<ggml_tensor*>> pre_attention_x(GGMLRunnerContext* ctx,
                                                                                                                ggml_tensor* x,
                                                                                                                ggml_tensor* c,
                                                                                                                bool pair_pack_plane_views = false) {
        GGML_ASSERT(self_attn);
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto attn2              = std::dynamic_pointer_cast<SelfAttention>(blocks["attn2"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 9;
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, n_mods * hidden_size]
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0, false);

        auto shift_msa  = m_vec[0];  // [N, hidden_size]
        auto scale_msa  = m_vec[1];  // [N, hidden_size]
        auto gate_msa   = m_vec[2];  // [N, hidden_size]
        auto shift_mlp  = m_vec[3];  // [N, hidden_size]
        auto scale_mlp  = m_vec[4];  // [N, hidden_size]
        auto gate_mlp   = m_vec[5];  // [N, hidden_size]
        auto shift_msa2 = m_vec[6];  // [N, hidden_size]
        auto scale_msa2 = m_vec[7];  // [N, hidden_size]
        auto gate_msa2  = m_vec[8];  // [N, hidden_size]

        auto x_norm = norm1->forward(ctx, x);

        auto attn_in = modulate(ctx->ggml_ctx, x_norm, shift_msa, scale_msa);
        auto qkv     = attn->pre_attention(ctx, attn_in, pair_pack_plane_views);

        auto attn2_in = modulate(ctx->ggml_ctx, x_norm, shift_msa2, scale_msa2);
        auto qkv2     = attn2->pre_attention(ctx, attn2_in);

        return {qkv, qkv2, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2}};
    }

    std::tuple<std::vector<ggml_tensor*>, std::vector<ggml_tensor*>, std::vector<ggml_tensor*>> pre_attention_x_sp(GGMLRunnerContext* ctx,
                                                                                                                   ggml_tensor* x,
                                                                                                                   ggml_tensor* c,
                                                                                                                   const std::string& name_prefix) {
        GGML_ASSERT(self_attn);
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto attn2              = std::dynamic_pointer_cast<SelfAttention>(blocks["attn2"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 9;
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0, false);

        auto shift_msa  = m_vec[0];
        auto scale_msa  = m_vec[1];
        auto gate_msa   = m_vec[2];
        auto shift_mlp  = m_vec[3];
        auto scale_mlp  = m_vec[4];
        auto gate_mlp   = m_vec[5];
        auto shift_msa2 = m_vec[6];
        auto scale_msa2 = m_vec[7];
        auto gate_msa2  = m_vec[8];

        auto x_norm = norm1->forward(ctx, x);

        auto attn_in = modulate(ctx->ggml_ctx, x_norm, shift_msa, scale_msa);
        auto qkv     = attn->pre_attention_sp(ctx, attn_in, name_prefix + "_joint");

        auto attn2_in = modulate(ctx->ggml_ctx, x_norm, shift_msa2, scale_msa2);
        auto qkv2     = attn2->pre_attention_sp(ctx, attn2_in, name_prefix + "_self");

        return {qkv, qkv2, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2}};
    }

    std::tuple<MMDiTSPQKVSendPack, std::vector<ggml_tensor*>, std::vector<ggml_tensor*>> pre_attention_x_sp_send_pack(GGMLRunnerContext* ctx,
                                                                                                                       ggml_tensor* x,
                                                                                                                       ggml_tensor* c,
                                                                                                                       const std::string& name_prefix) {
        GGML_ASSERT(self_attn);
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto attn2              = std::dynamic_pointer_cast<SelfAttention>(blocks["attn2"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 9;
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0, false);

        auto shift_msa  = m_vec[0];
        auto scale_msa  = m_vec[1];
        auto gate_msa   = m_vec[2];
        auto shift_mlp  = m_vec[3];
        auto scale_mlp  = m_vec[4];
        auto gate_mlp   = m_vec[5];
        auto shift_msa2 = m_vec[6];
        auto scale_msa2 = m_vec[7];
        auto gate_msa2  = m_vec[8];

        auto x_norm = norm1->forward(ctx, x);

        auto attn_in = modulate(ctx->ggml_ctx, x_norm, shift_msa, scale_msa);
        auto qkv     = attn->pre_attention_sp_send_pack(ctx, attn_in, name_prefix + "_joint");

        auto attn2_in = modulate(ctx->ggml_ctx, x_norm, shift_msa2, scale_msa2);
        auto qkv2     = attn2->pre_attention_sp(ctx, attn2_in, name_prefix + "_self");

        return {qkv, qkv2, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2}};
    }

    std::pair<std::vector<ggml_tensor*>, std::vector<ggml_tensor*>> pre_attention(GGMLRunnerContext* ctx,
                                                                                  ggml_tensor* x,
                                                                                  ggml_tensor* c,
                                                                                  bool pair_pack_plane_views = false) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, n_mods * hidden_size]
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0, false);

        if (!pre_only) {
            auto shift_msa = m_vec[0];  // [N, hidden_size]
            auto scale_msa = m_vec[1];  // [N, hidden_size]
            auto gate_msa  = m_vec[2];  // [N, hidden_size]
            auto shift_mlp = m_vec[3];  // [N, hidden_size]
            auto scale_mlp = m_vec[4];  // [N, hidden_size]
            auto gate_mlp  = m_vec[5];  // [N, hidden_size]

            auto attn_in = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);

            auto qkv = attn->pre_attention(ctx, attn_in, pair_pack_plane_views);

            return {qkv, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp}};
        } else {
            auto scale_msa = m_vec[0];  // [N, hidden_size]
            auto shift_msa = m_vec[1];  // [N, hidden_size]
            auto attn_in = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv     = attn->pre_attention(ctx, attn_in, pair_pack_plane_views);

            return {qkv, {nullptr, nullptr, nullptr, nullptr, nullptr}};
        }
    }

    std::pair<std::vector<ggml_tensor*>, std::vector<ggml_tensor*>> pre_attention_sp(GGMLRunnerContext* ctx,
                                                                                     ggml_tensor* x,
                                                                                     ggml_tensor* c,
                                                                                     const std::string& name_prefix) {
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0, false);

        if (!pre_only) {
            auto shift_msa = m_vec[0];
            auto scale_msa = m_vec[1];
            auto gate_msa  = m_vec[2];
            auto shift_mlp = m_vec[3];
            auto scale_mlp = m_vec[4];
            auto gate_mlp  = m_vec[5];

            auto attn_in = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv     = attn->pre_attention_sp(ctx, attn_in, name_prefix + "_attn");

            return {qkv, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp}};
        } else {
            auto scale_msa = m_vec[0];
            auto shift_msa = m_vec[1];
            auto attn_in   = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv       = attn->pre_attention_sp(ctx, attn_in, name_prefix + "_attn");

            return {qkv, {nullptr, nullptr, nullptr, nullptr, nullptr}};
        }
    }

    std::pair<MMDiTSPQKVSendPack, std::vector<ggml_tensor*>> pre_attention_sp_send_pack(GGMLRunnerContext* ctx,
                                                                                        ggml_tensor* x,
                                                                                        ggml_tensor* c,
                                                                                        const std::string& name_prefix) {
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0, false);

        if (!pre_only) {
            auto shift_msa = m_vec[0];
            auto scale_msa = m_vec[1];
            auto gate_msa  = m_vec[2];
            auto shift_mlp = m_vec[3];
            auto scale_mlp = m_vec[4];
            auto gate_mlp  = m_vec[5];

            auto attn_in = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv     = attn->pre_attention_sp_send_pack(ctx, attn_in, name_prefix + "_attn");

            return {qkv, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp}};
        } else {
            auto scale_msa = m_vec[0];
            auto shift_msa = m_vec[1];
            auto attn_in   = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv       = attn->pre_attention_sp_send_pack(ctx, attn_in, name_prefix + "_attn");

            return {qkv, {nullptr, nullptr, nullptr, nullptr, nullptr}};
        }
    }

    ggml_tensor* post_attention_x(GGMLRunnerContext* ctx,
                                  ggml_tensor* attn_out,
                                  ggml_tensor* attn2_out,
                                  ggml_tensor* x,
                                  ggml_tensor* gate_msa,
                                  ggml_tensor* shift_mlp,
                                  ggml_tensor* scale_mlp,
                                  ggml_tensor* gate_mlp,
                                  ggml_tensor* gate_msa2) {
        // attn_out: [N, n_token, hidden_size]
        // x: [N, n_token, hidden_size]
        // gate_msa: [N, hidden_size]
        // shift_mlp: [N, hidden_size]
        // scale_mlp: [N, hidden_size]
        // gate_mlp: [N, hidden_size]
        // return: [N, n_token, hidden_size]
        GGML_ASSERT(!pre_only);

        auto attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto attn2 = std::dynamic_pointer_cast<SelfAttention>(blocks["attn2"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto mlp   = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);

        gate_msa  = ggml_reshape_3d(ctx->ggml_ctx, gate_msa, gate_msa->ne[0], 1, gate_msa->ne[1]);     // [N, 1, hidden_size]
        gate_mlp  = ggml_reshape_3d(ctx->ggml_ctx, gate_mlp, gate_mlp->ne[0], 1, gate_mlp->ne[1]);     // [N, 1, hidden_size]
        gate_msa2 = ggml_reshape_3d(ctx->ggml_ctx, gate_msa2, gate_msa2->ne[0], 1, gate_msa2->ne[1]);  // [N, 1, hidden_size]

        attn_out  = attn->post_attention(ctx, attn_out);
        attn2_out = attn2->post_attention(ctx, attn2_out);

        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));
        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn2_out, gate_msa2));
        auto mlp_out = mlp->forward(ctx, modulate(ctx->ggml_ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp));
        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, gate_mlp));

        return x;
    }

    ggml_tensor* post_attention(GGMLRunnerContext* ctx,
                                ggml_tensor* attn_out,
                                ggml_tensor* x,
                                ggml_tensor* gate_msa,
                                ggml_tensor* shift_mlp,
                                ggml_tensor* scale_mlp,
                                ggml_tensor* gate_mlp) {
        // attn_out: [N, n_token, hidden_size]
        // x: [N, n_token, hidden_size]
        // gate_msa: [N, hidden_size]
        // shift_mlp: [N, hidden_size]
        // scale_mlp: [N, hidden_size]
        // gate_mlp: [N, hidden_size]
        // return: [N, n_token, hidden_size]
        GGML_ASSERT(!pre_only);

        auto attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto mlp   = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);

        gate_msa = ggml_reshape_3d(ctx->ggml_ctx, gate_msa, gate_msa->ne[0], 1, gate_msa->ne[1]);  // [N, 1, hidden_size]
        gate_mlp = ggml_reshape_3d(ctx->ggml_ctx, gate_mlp, gate_mlp->ne[0], 1, gate_mlp->ne[1]);  // [N, 1, hidden_size]

        attn_out = attn->post_attention(ctx, attn_out);

        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));
        auto mlp_out = mlp->forward(ctx, modulate(ctx->ggml_ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp));
        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, gate_mlp));

        return x;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, hidden_size]

        auto attn = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        if (self_attn) {
            auto qkv_intermediates = pre_attention_x(ctx, x, c);
            // auto qkv               = qkv_intermediates.first;
            // auto intermediates     = qkv_intermediates.second;
            // no longer a pair, but a tuple
            auto qkv           = std::get<0>(qkv_intermediates);
            auto qkv2          = std::get<1>(qkv_intermediates);
            auto intermediates = std::get<2>(qkv_intermediates);

            auto attn_out  = mmdit_attention(ctx, qkv[0], qkv[1], qkv[2], num_heads, false);     // [N, n_token, dim]
            auto attn2_out = mmdit_attention(ctx, qkv2[0], qkv2[1], qkv2[2], num_heads, false);  // [N, n_token, dim]
            x              = post_attention_x(ctx,
                                              attn_out,
                                              attn2_out,
                                              intermediates[0],
                                              intermediates[1],
                                              intermediates[2],
                                              intermediates[3],
                                              intermediates[4],
                                              intermediates[5]);
            return x;  // [N, n_token, dim]
        } else {
            auto qkv_intermediates = pre_attention(ctx, x, c);
            auto qkv               = qkv_intermediates.first;
            auto intermediates     = qkv_intermediates.second;

            auto attn_out = mmdit_attention(ctx, qkv[0], qkv[1], qkv[2], num_heads, false);  // [N, n_token, dim]
            x             = post_attention(ctx,
                                           attn_out,
                                           intermediates[0],
                                           intermediates[1],
                                           intermediates[2],
                                           intermediates[3],
                                           intermediates[4]);
            return x;  // [N, n_token, dim]
        }
    }
};

__STATIC_INLINE__ std::pair<ggml_tensor*, ggml_tensor*>
block_mixing(GGMLRunnerContext* ctx,
             ggml_tensor* context,
             ggml_tensor* x,
             ggml_tensor* c,
             std::shared_ptr<DismantledBlock> context_block,
             std::shared_ptr<DismantledBlock> x_block) {
    // context: [N, n_context, hidden_size]
    // x: [N, n_token, hidden_size]
    // c: [N, hidden_size]
    auto context_qkv_intermediates = context_block->pre_attention(ctx, context, c, true);
    auto context_qkv               = context_qkv_intermediates.first;
    auto context_intermediates     = context_qkv_intermediates.second;

    std::vector<ggml_tensor*> x_qkv, x_qkv2, x_intermediates;

    if (x_block->self_attn) {
        auto x_qkv_intermediates = x_block->pre_attention_x(ctx, x, c, true);
        x_qkv                    = std::get<0>(x_qkv_intermediates);
        x_qkv2                   = std::get<1>(x_qkv_intermediates);
        x_intermediates          = std::get<2>(x_qkv_intermediates);
    } else {
        auto x_qkv_intermediates = x_block->pre_attention(ctx, x, c, true);
        x_qkv                    = x_qkv_intermediates.first;
        x_intermediates          = x_qkv_intermediates.second;
    }
    auto attn = mmdit_fused_pair_pack_attention(ctx, context_qkv, x_qkv, x_block->num_heads);
    if (attn == nullptr) {
        std::vector<ggml_tensor*> qkv;
        for (int i = 0; i < 3; i++) {
            qkv.push_back(ggml_concat(ctx->ggml_ctx, context_qkv[i], x_qkv[i], 1));
        }

        attn = mmdit_attention(ctx, qkv[0], qkv[1], qkv[2], x_block->num_heads, false);  // [N, n_context + n_token, hidden_size]
    }

    auto context_attn = ggml_view_3d(ctx->ggml_ctx,
                                     attn,
                                     attn->ne[0],
                                     context->ne[1],
                                     attn->ne[2],
                                     attn->nb[1],
                                     attn->nb[2],
                                     0);  // [N, n_context, hidden_size]
    auto x_attn       = ggml_view_3d(ctx->ggml_ctx,
                                     attn,
                                     attn->ne[0],
                                     x->ne[1],
                                     attn->ne[2],
                                     attn->nb[1],
                                     attn->nb[2],
                                     context->ne[1] * attn->nb[1]);  // [N, n_token, hidden_size]

    if (!context_block->pre_only) {
        context = context_block->post_attention(ctx,
                                                context_attn,
                                                context_intermediates[0],
                                                context_intermediates[1],
                                                context_intermediates[2],
                                                context_intermediates[3],
                                                context_intermediates[4]);
    } else {
        context = nullptr;
    }

    if (x_block->self_attn) {
        auto attn2 = mmdit_attention(ctx, x_qkv2[0], x_qkv2[1], x_qkv2[2], x_block->num_heads, false);  // [N, n_token, hidden_size]

        x = x_block->post_attention_x(ctx,
                                      x_attn,
                                      attn2,
                                      x_intermediates[0],
                                      x_intermediates[1],
                                      x_intermediates[2],
                                      x_intermediates[3],
                                      x_intermediates[4],
                                      x_intermediates[5]);
    } else {
        x = x_block->post_attention(ctx,
                                    x_attn,
                                    x_intermediates[0],
                                    x_intermediates[1],
                                    x_intermediates[2],
                                    x_intermediates[3],
                                    x_intermediates[4]);
    }

    return {context, x};
}

__STATIC_INLINE__ std::pair<ggml_tensor*, ggml_tensor*>
block_mixing_sp(GGMLRunnerContext* ctx,
                ggml_tensor* context,
                ggml_tensor* x,
                ggml_tensor* c,
                std::shared_ptr<DismantledBlock> context_block,
                std::shared_ptr<DismantledBlock> x_block,
                int64_t context_pad,
                int64_t x_pad,
                const std::string& name_prefix) {
    GGML_ASSERT(context != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(context->ne[2] == 1);
    GGML_ASSERT(x->ne[2] == 1);

    const int world_size = mmdit_sp_world_size(ctx);
    const int64_t N      = x->ne[2];

    auto context_qkv_intermediates = context_block->pre_attention_sp_send_pack(ctx,
                                                                                context,
                                                                                c,
                                                                                name_prefix + "_context");
    auto context_pack              = context_qkv_intermediates.first;
    auto context_intermediates     = context_qkv_intermediates.second;

    MMDiTSPQKVSendPack x_pack;
    std::vector<ggml_tensor*> x_qkv2, x_intermediates;
    if (x_block->self_attn) {
        auto x_qkv_intermediates = x_block->pre_attention_x_sp_send_pack(ctx,
                                                                         x,
                                                                         c,
                                                                         name_prefix + "_x");
        x_pack                   = std::get<0>(x_qkv_intermediates);
        x_qkv2                   = std::get<1>(x_qkv_intermediates);
        x_intermediates          = std::get<2>(x_qkv_intermediates);
    } else {
        auto x_qkv_intermediates = x_block->pre_attention_sp_send_pack(ctx,
                                                                       x,
                                                                       c,
                                                                       name_prefix + "_x");
        x_pack                   = x_qkv_intermediates.first;
        x_intermediates          = x_qkv_intermediates.second;
    }

    GGML_ASSERT(context_pack.mixed_send_flat != nullptr && x_pack.mixed_send_flat != nullptr);
    GGML_ASSERT(context_pack.head_dim == x_pack.head_dim);
    GGML_ASSERT(context_pack.heads == x_pack.heads);
    GGML_ASSERT(context_pack.batch == x_pack.batch);
    const int64_t head_dim        = context_pack.head_dim;
    const int64_t qkv_shard_heads = context_pack.heads / world_size;

    const int64_t context_full_seq = context_pack.shard_sequence * world_size;
    const int64_t x_full_seq       = x_pack.shard_sequence * world_size;
    const int64_t context_real_seq = context_full_seq - context_pad;
    const int64_t x_real_seq       = x_full_seq - x_pad;
    GGML_ASSERT(context_real_seq > 0 && x_real_seq > 0);
    GGML_ASSERT(context_real_seq + x_real_seq > 0);
    GGML_ASSERT(context_pack.pad == 0 || context_pack.pad == context_pad);
    GGML_ASSERT(x_pack.pad == 0 || x_pack.pad == x_pad);

    const int64_t packed_qkv_dim              = head_dim * 2;
    const int64_t context_mixed_chunk_per_peer = packed_qkv_dim * qkv_shard_heads * context_pack.shard_sequence;
    const int64_t x_mixed_chunk_per_peer       = packed_qkv_dim * qkv_shard_heads * x_pack.shard_sequence;
    ggml_tensor* joint_qkv_send                = mmdit_sp_peer_concat_flat(ctx->ggml_ctx,
                                                            context_pack.mixed_send_flat,
                                                            x_pack.mixed_send_flat,
                                                            context_mixed_chunk_per_peer,
                                                            x_mixed_chunk_per_peer,
                                                            world_size);
    auto joint_qkv_recv                  = edgedit::parallel::sp_all_to_all_4d_seq_to_head_packed_recv_only(
        ctx->ggml_ctx,
        joint_qkv_send,
        packed_qkv_dim,
        context_pack.heads,
        context_pack.shard_sequence + x_pack.shard_sequence,
        context_pack.batch,
        world_size,
        name_prefix + "_joint_attn_qkv_seq_to_head");

    std::vector<ggml_tensor*> qkv;
    qkv.reserve(3);
    qkv.push_back(mmdit_fused_joint_mixed_q_to_seq_major(ctx->ggml_ctx,
                                                         joint_qkv_recv.recv_flat,
                                                         context_real_seq,
                                                         x_real_seq,
                                                         context_full_seq,
                                                         x_full_seq,
                                                         head_dim,
                                                         qkv_shard_heads,
                                                         world_size));
    qkv.push_back(mmdit_fused_joint_mixed_kv_to_seq_major(ctx->ggml_ctx,
                                                          joint_qkv_recv.recv_flat,
                                                          false,
                                                          context_real_seq,
                                                          x_real_seq,
                                                          context_full_seq,
                                                          x_full_seq,
                                                          head_dim,
                                                          qkv_shard_heads,
                                                          world_size));
    qkv[1] = ggml_cast(ctx->ggml_ctx, qkv[1], GGML_TYPE_F32);
    qkv.push_back(mmdit_fused_joint_mixed_kv_to_seq_major(ctx->ggml_ctx,
                                                          joint_qkv_recv.recv_flat,
                                                          true,
                                                          context_real_seq,
                                                          x_real_seq,
                                                          context_full_seq,
                                                          x_full_seq,
                                                          head_dim,
                                                          qkv_shard_heads,
                                                          world_size));
    qkv[2] = ggml_cast(ctx->ggml_ctx, qkv[2], GGML_TYPE_F32);

    ggml_tensor* attn = mmdit_sp_attention_seq_major(ctx,
                                                     qkv[0],
                                                     qkv[1],
                                                     qkv[2],
                                                     true);
    sd::ggml_graph_cut::mark_graph_cut(attn, name_prefix + ".sp_attention", "attn");

    ggml_tensor* attn_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                           attn,
                                           head_dim,
                                           qkv_shard_heads,
                                           context_real_seq + x_real_seq,
                                           N);
    ggml_tensor* attn_head_to_seq_send = mmdit_fused_attn_head_to_seq_send_pack_f16(ctx->ggml_ctx,
                                                                                    attn_4d,
                                                                                    context_real_seq,
                                                                                    x_real_seq,
                                                                                    context_full_seq,
                                                                                    x_full_seq,
                                                                                    world_size);
    auto attn_recv = edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_recv_only_f16(
        ctx->ggml_ctx,
        attn_head_to_seq_send,
        head_dim,
        qkv_shard_heads,
        {context_full_seq, x_full_seq},
        world_size,
        name_prefix + "_context_x_attn_head_to_seq");
    ggml_tensor* context_attn_local = mmdit_fused_attn_head_to_seq_recv_unpack_f16(ctx->ggml_ctx,
                                                                                  attn_recv.recv_flat,
                                                                                  0,
                                                                                  context_full_seq,
                                                                                  x_full_seq,
                                                                                  head_dim,
                                                                                  context_pack.heads,
                                                                                  world_size);
    ggml_tensor* x_attn_local       = mmdit_fused_attn_head_to_seq_recv_unpack_f16(ctx->ggml_ctx,
                                                                            attn_recv.recv_flat,
                                                                            1,
                                                                            context_full_seq,
                                                                            x_full_seq,
                                                                            head_dim,
                                                                            context_pack.heads,
                                                                            world_size);

    ggml_tensor* context_attn_out = nullptr;
    if (!context_block->pre_only) {
        context_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                           context_attn_local,
                                           context_attn_local->ne[0] * context_attn_local->ne[1],
                                           context_attn_local->ne[2],
                                           N);
    }

    ggml_tensor* x_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                              x_attn_local,
                                              x_attn_local->ne[0] * x_attn_local->ne[1],
                                              x_attn_local->ne[2],
                                              N);

    if (!context_block->pre_only) {
        context = context_block->post_attention(ctx,
                                                context_attn_out,
                                                context_intermediates[0],
                                                context_intermediates[1],
                                                context_intermediates[2],
                                                context_intermediates[3],
                                                context_intermediates[4]);
    } else {
        context = nullptr;
    }

    if (x_block->self_attn) {
        ggml_tensor* q2 = mmdit_sp_view_head_sequence(ctx->ggml_ctx,
                                                      x_qkv2[0],
                                                      0,
                                                      x_real_seq);
        ggml_tensor* k2 = mmdit_sp_view_head_sequence(ctx->ggml_ctx,
                                                      x_qkv2[1],
                                                      0,
                                                      x_real_seq);
        ggml_tensor* v2 = mmdit_sp_view_head_sequence(ctx->ggml_ctx,
                                                      x_qkv2[2],
                                                      0,
                                                      x_real_seq);
        q2              = ggml_is_contiguous(q2) ? q2 : ggml_cont(ctx->ggml_ctx, q2);
        k2              = ggml_is_contiguous(k2) ? k2 : ggml_cont(ctx->ggml_ctx, k2);
        v2              = ggml_is_contiguous(v2) ? v2 : ggml_cont(ctx->ggml_ctx, v2);

        ggml_tensor* attn2 = mmdit_sp_attention(ctx,
                                                q2,
                                                k2,
                                                v2);
        sd::ggml_graph_cut::mark_graph_cut(attn2, name_prefix + ".sp_attention2", "attn");

        ggml_tensor* attn2_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                attn2,
                                                head_dim,
                                                qkv_shard_heads,
                                                x_real_seq,
                                                N);

        ggml_tensor* x_attn2_head = mmdit_sp_extract_stream_attention(ctx->ggml_ctx,
                                                                      attn2_4d,
                                                                      0,
                                                                      x_real_seq,
                                                                      x_pad);
        auto x_attn2_local        = edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx->ggml_ctx,
                                                                                    x_attn2_head,
                                                                                    world_size,
                                                                                    name_prefix + "_x_self_attn_head_to_seq");
        ggml_tensor* attn2_out    = ggml_reshape_3d(ctx->ggml_ctx,
                                                    x_attn2_local.output,
                                                    x_attn2_local.output->ne[0] * x_attn2_local.output->ne[1],
                                                    x_attn2_local.output->ne[2],
                                                    N);

        x = x_block->post_attention_x(ctx,
                                      x_attn_out,
                                      attn2_out,
                                      x_intermediates[0],
                                      x_intermediates[1],
                                      x_intermediates[2],
                                      x_intermediates[3],
                                      x_intermediates[4],
                                      x_intermediates[5]);
    } else {
        x = x_block->post_attention(ctx,
                                    x_attn_out,
                                    x_intermediates[0],
                                    x_intermediates[1],
                                    x_intermediates[2],
                                    x_intermediates[3],
                                    x_intermediates[4]);
    }

    return {context, x};
}

struct JointBlock : public GGMLBlock {
public:
    JointBlock(int64_t hidden_size,
               int64_t num_heads,
               float mlp_ratio     = 4.0,
               std::string qk_norm = "",
               bool qkv_bias       = false,
               bool pre_only       = false,
               bool self_attn_x    = false) {
        blocks["context_block"] = std::shared_ptr<GGMLBlock>(new DismantledBlock(hidden_size, num_heads, mlp_ratio, qk_norm, qkv_bias, pre_only, false));
        blocks["x_block"]       = std::shared_ptr<GGMLBlock>(new DismantledBlock(hidden_size, num_heads, mlp_ratio, qk_norm, qkv_bias, false, self_attn_x));
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                  ggml_tensor* context,
                                                  ggml_tensor* x,
                                                  ggml_tensor* c) {
        auto context_block = std::dynamic_pointer_cast<DismantledBlock>(blocks["context_block"]);
        auto x_block       = std::dynamic_pointer_cast<DismantledBlock>(blocks["x_block"]);

        return block_mixing(ctx, context, x, c, context_block, x_block);
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward_sp(GGMLRunnerContext* ctx,
                                                     ggml_tensor* context,
                                                     ggml_tensor* x,
                                                     ggml_tensor* c,
                                                     int64_t context_pad,
                                                     int64_t x_pad,
                                                     const std::string& name_prefix) {
        auto context_block = std::dynamic_pointer_cast<DismantledBlock>(blocks["context_block"]);
        auto x_block       = std::dynamic_pointer_cast<DismantledBlock>(blocks["x_block"]);

        return block_mixing_sp(ctx, context, x, c, context_block, x_block, context_pad, x_pad, name_prefix);
    }
};

struct FinalLayer : public GGMLBlock {
    // The final layer of DiT.
public:
    FinalLayer(int64_t hidden_size,
               int64_t patch_size,
               int64_t out_channels) {
        // total_out_channels is always None
        blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels, true, true));
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, patch_size * patch_size * out_channels]
        auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
        auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, 2 * hidden_size]
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, 2, 0, false);
        auto scale = m_vec[0];  // [N, hidden_size]
        auto shift = m_vec[1];  // [N, hidden_size]

        x = modulate(ctx->ggml_ctx, norm_final->forward(ctx, x), shift, scale);
        x = linear->forward(ctx, x);

        return x;
    }
};

struct MMDiT : public GGMLBlock {
    // Diffusion model with a Transformer backbone.
protected:
    int64_t input_size               = -1;
    int patch_size                   = 2;
    int64_t in_channels              = 16;
    int64_t d_self                   = -1;  // >=0 for MMdiT-X
    int64_t depth                    = 24;
    float mlp_ratio                  = 4.0f;
    int64_t adm_in_channels          = 2048;
    int64_t out_channels             = 16;
    int64_t pos_embed_max_size       = 192;
    int64_t num_patchs               = 36864;  // 192 * 192
    int64_t context_size             = 4096;
    int64_t context_embedder_out_dim = 1536;
    int64_t hidden_size;
    std::string qk_norm;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        enum ggml_type wtype = GGML_TYPE_F32;
        params["pos_embed"]  = ggml_new_tensor_3d(ctx, wtype, hidden_size, num_patchs, 1);
    }

public:
    MMDiT(const String2TensorStorage& tensor_storage_map = {}) {
        // input_size is always None
        // learn_sigma is always False
        // register_length is alwalys 0
        // rmsnorm is alwalys False
        // scale_mod_only is alwalys False
        // swiglu is alwalys False
        // qkv_bias is always True
        // context_processor_layers is always None
        // pos_embed_scaling_factor is not used
        // pos_embed_offset is not used
        // context_embedder_config is always {'target': 'torch.nn.Linear', 'params': {'in_features': 4096, 'out_features': 1536}}

        for (auto pair : tensor_storage_map) {
            std::string tensor_name = pair.first;
            if (tensor_name.find("model.diffusion_model.") == std::string::npos)
                continue;
            size_t jb = tensor_name.find("joint_blocks.");
            if (jb != std::string::npos) {
                tensor_name     = tensor_name.substr(jb);  // remove prefix
                int block_depth = atoi(tensor_name.substr(13, tensor_name.find(".", 13)).c_str());
                if (block_depth + 1 > depth) {
                    depth = block_depth + 1;
                }
                if (tensor_name.find("attn.ln") != std::string::npos) {
                    if (tensor_name.find(".bias") != std::string::npos) {
                        qk_norm = "ln";
                    } else {
                        qk_norm = "rms";
                    }
                }
                if (tensor_name.find("attn2") != std::string::npos) {
                    if (block_depth > d_self) {
                        d_self = block_depth;
                    }
                }
            }
        }

        if (d_self >= 0) {
            pos_embed_max_size *= 2;
            num_patchs *= 4;
        }

        LOG_INFO("MMDiT layers: %d (including %d MMDiT-x layers)", depth, d_self + 1);

        int64_t default_out_channels = in_channels;
        hidden_size                  = 64 * depth;
        context_embedder_out_dim     = 64 * depth;
        int64_t num_heads            = depth;

        blocks["x_embedder"] = std::shared_ptr<GGMLBlock>(new PatchEmbed(input_size, patch_size, in_channels, hidden_size, true));
        blocks["t_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedder(hidden_size));

        if (adm_in_channels != -1) {
            blocks["y_embedder"] = std::shared_ptr<GGMLBlock>(new VectorEmbedder(adm_in_channels, hidden_size));
        }

        blocks["context_embedder"] = std::shared_ptr<GGMLBlock>(new Linear(4096, context_embedder_out_dim, true, true));

        for (int i = 0; i < depth; i++) {
            blocks["joint_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new JointBlock(hidden_size,
                                                                                                    num_heads,
                                                                                                    mlp_ratio,
                                                                                                    qk_norm,
                                                                                                    true,
                                                                                                    i == depth - 1,
                                                                                                    i <= d_self));
        }

        blocks["final_layer"] = std::shared_ptr<GGMLBlock>(new FinalLayer(hidden_size, patch_size, out_channels));
    }

    ggml_tensor*
    cropped_pos_embed(ggml_context* ctx,
                      int64_t h,
                      int64_t w) {
        auto pos_embed = params["pos_embed"];

        h = (h + 1) / patch_size;
        w = (w + 1) / patch_size;

        GGML_ASSERT(h <= pos_embed_max_size && h > 0);
        GGML_ASSERT(w <= pos_embed_max_size && w > 0);

        int64_t top  = (pos_embed_max_size - h) / 2;
        int64_t left = (pos_embed_max_size - w) / 2;

        auto spatial_pos_embed = ggml_reshape_3d(ctx, pos_embed, hidden_size, pos_embed_max_size, pos_embed_max_size);

        // spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = ggml_view_3d(ctx,
                                         spatial_pos_embed,
                                         hidden_size,
                                         pos_embed_max_size,
                                         h,
                                         spatial_pos_embed->nb[1],
                                         spatial_pos_embed->nb[2],
                                         spatial_pos_embed->nb[2] * top);                      // [h, pos_embed_max_size, hidden_size]
        spatial_pos_embed = ggml_cont(ctx, ggml_permute(ctx, spatial_pos_embed, 0, 2, 1, 3));  // [pos_embed_max_size, h, hidden_size]
        spatial_pos_embed = ggml_view_3d(ctx,
                                         spatial_pos_embed,
                                         hidden_size,
                                         h,
                                         w,
                                         spatial_pos_embed->nb[1],
                                         spatial_pos_embed->nb[2],
                                         spatial_pos_embed->nb[2] * left);                     // [w, h, hidden_size]
        spatial_pos_embed = ggml_cont(ctx, ggml_permute(ctx, spatial_pos_embed, 0, 2, 1, 3));  // [h, w, hidden_size]
        spatial_pos_embed = ggml_reshape_3d(ctx, spatial_pos_embed, hidden_size, h * w, 1);    // [1, h*w, hidden_size]
        return spatial_pos_embed;
    }

    ggml_tensor* forward_core_with_concat(GGMLRunnerContext* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* c_mod,
                                          ggml_tensor* context,
                                          std::vector<int> skip_layers = std::vector<int>(),
                                          bool use_sp_mainline = false,
                                          int64_t context_pad  = 0,
                                          int64_t x_pad        = 0
    ) {
        // x: [N, H*W, hidden_size]
        // context: [N, n_context, d_context]
        // c: [N, hidden_size]
        // return: [N, N*W, patch_size * patch_size * out_channels]
        auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

        for (int i = 0; i < depth; i++) {
            // skip iteration if i is in skip_layers
            if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                continue;
            }

            auto block = std::dynamic_pointer_cast<JointBlock>(blocks["joint_blocks." + std::to_string(i)]);

            std::pair<ggml_tensor*, ggml_tensor*> context_x;
            if (use_sp_mainline) {
                context_x = block->forward_sp(ctx,
                                              context,
                                              x,
                                              c_mod,
                                              context_pad,
                                              x_pad,
                                              "mmdit_block" + std::to_string(i));
            } else
            {
                context_x = block->forward(ctx, context, x, c_mod);
            }
            context        = context_x.first;
            x              = context_x.second;
            sd::ggml_graph_cut::mark_graph_cut(context, "mmdit.joint_blocks." + std::to_string(i), "context");
            sd::ggml_graph_cut::mark_graph_cut(x, "mmdit.joint_blocks." + std::to_string(i), "x");
        }

        x = final_layer->forward(ctx, x, c_mod);  // (N, T, patch_size ** 2 * out_channels)

        if (use_sp_mainline) {
            auto x_gather = edgedit::parallel::sp_mark_gather_sequence(ctx->ggml_ctx,
                                                                       x,
                                                                       mmdit_sp_world_size(ctx),
                                                                       1,
                                                                       x_pad,
                                                                       "mmdit_sp_final_x_gather");
            x             = x_gather.gathered;
        }

        return x;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* t,
                         ggml_tensor* y               = nullptr,
                         ggml_tensor* context         = nullptr,
                         std::vector<int> skip_layers = std::vector<int>()) {
        // Forward pass of DiT.
        // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        // t: (N,) tensor of diffusion timesteps
        // y: (N, adm_in_channels) tensor of class labels
        // context: (N, L, D)
        // return: (N, C, H, W)
        auto x_embedder = std::dynamic_pointer_cast<PatchEmbed>(blocks["x_embedder"]);
        auto t_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);

        int64_t W = x->ne[0];
        int64_t H = x->ne[1];

        auto patch_embed = x_embedder->forward(ctx, x);                      // [N, H*W, hidden_size]
        auto pos_embed   = cropped_pos_embed(ctx->ggml_ctx, H, W);           // [1, H*W, hidden_size]
        x                = ggml_add(ctx->ggml_ctx, patch_embed, pos_embed);  // [N, H*W, hidden_size]

        auto c = t_embedder->forward(ctx, t);  // [N, hidden_size]
        if (y != nullptr && adm_in_channels != -1) {
            auto y_embedder = std::dynamic_pointer_cast<VectorEmbedder>(blocks["y_embedder"]);

            y = y_embedder->forward(ctx, y);  // [N, hidden_size]
            c = ggml_add(ctx->ggml_ctx, c, y);
        }

        if (context != nullptr) {
            auto context_embedder = std::dynamic_pointer_cast<Linear>(blocks["context_embedder"]);

            context = context_embedder->forward(ctx, context);  // [N, L, D] aka [N, L, 1536]
        }

        bool use_sp_mainline = mmdit_sp_enabled(ctx);
        edgedit::parallel::SPSequenceSplit x_sp_split;
        edgedit::parallel::SPSequenceSplit context_sp_split;
        if (use_sp_mainline) {
            const int rank       = mmdit_sp_rank(ctx);
            const int world_size = mmdit_sp_world_size(ctx);
            if (context == nullptr ||
                x->ne[2] != 1 ||
                context->ne[2] != 1 ||
                depth % world_size != 0) {
                use_sp_mainline = false;
            } else {
                x_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                  x,
                                                                  rank,
                                                                  world_size,
                                                                  1,
                                                                  "mmdit_sp_x_split");
                context_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                        context,
                                                                        rank,
                                                                        world_size,
                                                                        1,
                                                                        "mmdit_sp_context_split");
                x       = x_sp_split.local;
                context = context_sp_split.local;
            }
        }

        sd::ggml_graph_cut::mark_graph_cut(x, "mmdit.prelude", "x");
        sd::ggml_graph_cut::mark_graph_cut(c, "mmdit.prelude", "c");
        if (context != nullptr) {
            sd::ggml_graph_cut::mark_graph_cut(context, "mmdit.prelude", "context");
        }

        x = forward_core_with_concat(ctx,
                                     x,
                                     c,
                                     context,
                                     skip_layers,
                                     use_sp_mainline,
                                     context_sp_split.pad,
                                     x_sp_split.pad
        );  // (N, H*W, patch_size ** 2 * out_channels)

        x = DiT::unpatchify_and_crop(ctx->ggml_ctx, x, H, W, patch_size, patch_size, /*patch_last*/ false);  // [N, C, H, W]

        return x;
    }
};
struct MMDiTRunner : public GGMLRunner {
    MMDiT mmdit;

    MMDiTRunner(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                const std::string prefix                       = "")
        : GGMLRunner(backend, offload_params_to_cpu), mmdit(tensor_storage_map) {
        mmdit.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "mmdit";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        mmdit.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                             const sd::Tensor<float>& timesteps_tensor,
                             const sd::Tensor<float>& context_tensor = {},
                             const sd::Tensor<float>& y_tensor       = {},
                             std::vector<int> skip_layers            = std::vector<int>()) {
        ggml_cgraph* gf = new_graph_custom(MMDIT_GRAPH_SIZE);

        ggml_tensor* x         = make_input(x_tensor);
        ggml_tensor* timesteps = make_input(timesteps_tensor);
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* y         = make_optional_input(y_tensor);

        auto runner_ctx  = get_context();
        ggml_tensor* out = mmdit.forward(&runner_ctx,
                                         x,
                                         timesteps,
                                         y,
                                         context,
                                         skip_layers);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    sd::Tensor<float> compute(int n_threads,
                              const sd::Tensor<float>& x,
                              const sd::Tensor<float>& timesteps,
                              const sd::Tensor<float>& context = {},
                              const sd::Tensor<float>& y       = {},
                              std::vector<int> skip_layers     = std::vector<int>()) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, timesteps, context, y, skip_layers);
        };

        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
    }

    void test() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        {
            // cpu f16: pass
            // cpu f32: pass
            // cuda f16: pass
            // cuda f32: pass
            sd::Tensor<float> x({128, 128, 16, 1});
            std::vector<float> timesteps_vec(1, 999.f);
            auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);
            x.fill_(0.01f);
            // print_ggml_tensor(x);

            sd::Tensor<float> context({4096, 154, 1});
            context.fill_(0.01f);
            // print_ggml_tensor(context);

            sd::Tensor<float> y({2048, 1});
            y.fill_(0.01f);
            // print_ggml_tensor(y);

            sd::Tensor<float> out;

            auto out_opt = compute(8,
                                   x,
                                   timesteps,
                                   context,
                                   y);

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
        }
    }

    static void load_from_file_and_test(const std::string& file_path) {
        // ggml_backend_t backend    = ggml_backend_cuda_init(0);
        ggml_backend_t backend             = ggml_backend_cpu_init();
        ggml_type model_data_type          = GGML_TYPE_F16;
        std::shared_ptr<MMDiTRunner> mmdit = std::make_shared<MMDiTRunner>(backend, false);
        {
            LOG_INFO("loading from '%s'", file_path.c_str());

            mmdit->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            mmdit->get_param_tensors(tensors, "model.diffusion_model");

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("mmdit model loaded");
        }
        mmdit->test();
    }
};

#endif
