#include "parallel/sp_parallel.hpp"

#include "backend/ggml/ed_ggml_sp_flux_ext.hpp"
#include "backend/ggml/ggml_graph_cut.h"
#include "parallel/sp_recv_placeholder.hpp"

#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace edgedit::parallel {
namespace {

struct SPFusedQKVPackParams {
    uint64_t magic;
    int64_t txt_real_seq;
    int64_t img_real_seq;
    int64_t mode;
    int64_t txt_padded_seq;
    int64_t img_padded_seq;
    int64_t world_size;
    int64_t stream_index;
};

constexpr uint64_t SP_FUSED_QKV_PACK_MAGIC = 0x5157454e46514b56ULL; // legacy fused-qkv custom-op tag

std::mutex& sp_fused_qkv_pack_params_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<SPFusedQKVPackParams>>& sp_fused_qkv_pack_params_store() {
    static std::vector<std::unique_ptr<SPFusedQKVPackParams>> store;
    return store;
}

SPFusedQKVPackParams* sp_fused_qkv_pack_make_params(int64_t txt_real_seq,
                                                    int64_t img_real_seq,
                                                    int64_t mode,
                                                    int64_t txt_padded_seq = 0,
                                                    int64_t img_padded_seq = 0,
                                                    int64_t world_size = 0,
                                                    int64_t stream_index = 0) {
    auto params = std::make_unique<SPFusedQKVPackParams>();
    params->magic          = SP_FUSED_QKV_PACK_MAGIC;
    params->txt_real_seq   = txt_real_seq;
    params->img_real_seq   = img_real_seq;
    params->mode           = mode;
    params->txt_padded_seq = txt_padded_seq;
    params->img_padded_seq = img_padded_seq;
    params->world_size     = world_size;
    params->stream_index   = stream_index;
    SPFusedQKVPackParams* raw = params.get();
    std::lock_guard<std::mutex> lock(sp_fused_qkv_pack_params_mutex());
    sp_fused_qkv_pack_params_store().push_back(std::move(params));
    return raw;
}

float sp_fused_qkv_pack_get_f32(const ggml_tensor* src,
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

bool sp_fused_qkv_pack_f32_or_f16(const ggml_tensor* t) {
    return t != nullptr && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
}

void sp_fused_qkv_pack_set_f32(ggml_tensor* dst,
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

void sp_qkv_seq_to_head_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const SPFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == SP_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 6);
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
        const float value = sp_fused_qkv_pack_get_f32(src, d, head, seq, 0);
        sp_fused_qkv_pack_set_f32(dst, linear, 0, 0, 0, value);
    }
}

void sp_qkv_seq_to_head_send_pack_mixed_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const SPFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == SP_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 18);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    GGML_ASSERT(q != nullptr && k != nullptr && v != nullptr);
    GGML_ASSERT(sp_fused_qkv_pack_f32_or_f16(q) &&
                sp_fused_qkv_pack_f32_or_f16(k) &&
                sp_fused_qkv_pack_f32_or_f16(v));

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
        int64_t rem         = linear;
        const int64_t d_all = rem % packed_dim;
        rem /= packed_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer = rem;
        const int64_t head = head_local + peer * shard_heads;

        if (d_all < head_dim) {
            const float value = sp_fused_qkv_pack_get_f32(q, d_all, head, seq, 0);
            uint32_t bits;
            static_assert(sizeof(bits) == sizeof(value));
            memcpy(&bits, &value, sizeof(bits));
            *reinterpret_cast<uint32_t*>(static_cast<char*>(dst->data) + linear * dst->nb[0]) = bits;
        } else {
            const int64_t d = d_all - head_dim;
            const float k_value = sp_fused_qkv_pack_get_f32(k, d, head, seq, 0);
            const float v_value = sp_fused_qkv_pack_get_f32(v, d, head, seq, 0);
            const uint32_t k_half = static_cast<uint32_t>(ggml_fp32_to_fp16(k_value));
            const uint32_t v_half = static_cast<uint32_t>(ggml_fp32_to_fp16(v_value));
            *reinterpret_cast<uint32_t*>(static_cast<char*>(dst->data) + linear * dst->nb[0]) =
                k_half | (v_half << 16);
        }
    }
}

void sp_qkv_seq_to_head_send_pack_f16_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const SPFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == SP_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 46);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
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
        const float value = sp_fused_qkv_pack_get_f32(src, d, head, seq, 0);
        *reinterpret_cast<ggml_fp16_t*>(static_cast<char*>(dst->data) + linear * dst->nb[0]) =
            ggml_fp32_to_fp16(value);
    }
}

void sp_double_qkv_seq_to_head_send_pack_f16_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const SPFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == SP_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 47);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ith >= 0 && nth > 0);

    const ggml_tensor* first_q = dst->src[0];
    const ggml_tensor* first_k = dst->src[1];
    const ggml_tensor* first_v = dst->src[2];
    const ggml_tensor* second_q = dst->src[3];
    const ggml_tensor* second_k = dst->src[4];
    const ggml_tensor* second_v = dst->src[5];
    GGML_ASSERT(first_q != nullptr && first_k != nullptr && first_v != nullptr);
    GGML_ASSERT(second_q != nullptr && second_k != nullptr && second_v != nullptr);
    GGML_ASSERT(sp_fused_qkv_pack_f32_or_f16(first_q) &&
                sp_fused_qkv_pack_f32_or_f16(first_k) &&
                sp_fused_qkv_pack_f32_or_f16(first_v) &&
                sp_fused_qkv_pack_f32_or_f16(second_q) &&
                sp_fused_qkv_pack_f32_or_f16(second_k) &&
                sp_fused_qkv_pack_f32_or_f16(second_v));

    const int64_t world_size = params->world_size;
    const int64_t head_dim = first_q->ne[0];
    const int64_t heads = first_q->ne[1];
    const int64_t first_shard_sequence = first_q->ne[2];
    const int64_t second_shard_sequence = second_q->ne[2];
    const int64_t shard_heads = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t first_chunk = total_head_dim * shard_heads * first_shard_sequence;
    const int64_t second_chunk = total_head_dim * shard_heads * second_shard_sequence;
    const int64_t count_per_peer = first_chunk + second_chunk;
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(first_k->ne[0] == head_dim && first_v->ne[0] == head_dim);
    GGML_ASSERT(second_k->ne[0] == head_dim && second_v->ne[0] == head_dim);
    GGML_ASSERT(first_k->ne[1] == heads && first_v->ne[1] == heads &&
                second_q->ne[1] == heads && second_k->ne[1] == heads && second_v->ne[1] == heads);
    GGML_ASSERT(first_k->ne[2] == first_shard_sequence && first_v->ne[2] == first_shard_sequence);
    GGML_ASSERT(second_k->ne[2] == second_shard_sequence && second_v->ne[2] == second_shard_sequence);
    GGML_ASSERT(first_q->ne[3] == 1 && first_k->ne[3] == 1 && first_v->ne[3] == 1);
    GGML_ASSERT(second_q->ne[3] == 1 && second_k->ne[3] == 1 && second_v->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == count_per_peer * world_size);

    const int64_t total = dst->ne[0];
    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const bool is_second = rem >= first_chunk;
        if (is_second) {
            rem -= first_chunk;
        }
        const int64_t shard_sequence = is_second ? second_shard_sequence : first_shard_sequence;
        const int64_t d_all = rem % total_head_dim;
        rem /= total_head_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem;
        const int64_t head = head_local + peer * shard_heads;
        const int64_t qkv_plane = d_all / head_dim;
        const int64_t d = d_all - qkv_plane * head_dim;
        GGML_ASSERT(seq < shard_sequence);

        const ggml_tensor* src = nullptr;
        if (!is_second) {
            src = qkv_plane == 0 ? first_q : (qkv_plane == 1 ? first_k : first_v);
        } else {
            src = qkv_plane == 0 ? second_q : (qkv_plane == 1 ? second_k : second_v);
        }
        const float value = sp_fused_qkv_pack_get_f32(src, d, head, seq, 0);
        *reinterpret_cast<ggml_fp16_t*>(static_cast<char*>(dst->data) + linear * dst->nb[0]) =
            ggml_fp32_to_fp16(value);
    }
}

void sp_attn_head_to_seq_send_pack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const SPFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == SP_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 7 || params->mode == 13);
    GGML_ASSERT((params->mode == 7 && dst->type == GGML_TYPE_F32) ||
                (params->mode == 13 && dst->type == GGML_TYPE_F16));

    const ggml_tensor* attn = dst->src[0];
    GGML_ASSERT(attn != nullptr);
    GGML_ASSERT(attn->type == GGML_TYPE_F32 || attn->type == GGML_TYPE_F16);
    const int64_t first_sequence  = params->txt_real_seq;
    const int64_t second_sequence = params->img_real_seq;
    const int64_t world_size      = params->world_size;
    const int64_t first_padded    = params->txt_padded_seq;
    const int64_t second_padded   = params->img_padded_seq;
    const int64_t head_dim        = attn->ne[0];
    const int64_t shard_heads     = attn->ne[1];
    const int64_t first_shard_seq = first_padded / world_size;
    const int64_t second_shard_seq = second_padded / world_size;
    const int64_t first_chunk     = head_dim * shard_heads * first_shard_seq;
    const int64_t second_chunk    = head_dim * shard_heads * second_shard_seq;
    const int64_t count_per_peer  = first_chunk + second_chunk;
    const int64_t total           = count_per_peer * world_size;

    GGML_ASSERT(first_sequence > 0 && second_sequence >= 0);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(first_padded >= first_sequence && second_padded >= second_sequence);
    GGML_ASSERT(first_padded % world_size == 0 && second_padded % world_size == 0);
    GGML_ASSERT(attn->ne[2] == first_sequence + second_sequence);
    GGML_ASSERT(attn->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == total);

    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const bool is_second = rem >= first_chunk;
        if (is_second) {
            rem -= first_chunk;
        }
        const int64_t shard_seq = is_second ? second_shard_seq : first_shard_seq;
        const int64_t real_seq  = is_second ? second_sequence : first_sequence;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_seq = rem;
        const int64_t stream_seq = peer * shard_seq + local_seq;

        float value = 0.0f;
        if (stream_seq < real_seq) {
            const int64_t total_seq = is_second ? first_sequence + stream_seq : stream_seq;
            value = sp_fused_qkv_pack_get_f32(attn, d, head, total_seq, 0);
        }
        if (dst->type == GGML_TYPE_F32) {
            sp_fused_qkv_pack_set_f32(dst, linear, 0, 0, 0, value);
        } else {
            char* data = static_cast<char*>(dst->data);
            *reinterpret_cast<ggml_fp16_t*>(data + linear * dst->nb[0]) = ggml_fp32_to_fp16(value);
        }
    }
}

void sp_attn_head_to_seq_recv_unpack_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const SPFusedQKVPackParams*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->magic == SP_FUSED_QKV_PACK_MAGIC);
    GGML_ASSERT(params->mode == 8 || params->mode == 14 || params->mode == 45 || params->mode == 48);
    GGML_ASSERT(((params->mode == 8 || params->mode == 14) && dst->type == GGML_TYPE_F32) ||
                (params->mode == 45 && dst->type == GGML_TYPE_F16) ||
                (params->mode == 48 && dst->type == GGML_TYPE_BF16));

    const ggml_tensor* recv_flat = dst->src[0];
    GGML_ASSERT(recv_flat != nullptr);
    GGML_ASSERT((params->mode == 8 && recv_flat->type == GGML_TYPE_F32) ||
                ((params->mode == 14 || params->mode == 45 || params->mode == 48) && recv_flat->type == GGML_TYPE_F16));
    const int64_t stream_index   = params->stream_index;
    const int64_t world_size     = params->world_size;
    const int64_t first_padded   = params->txt_padded_seq;
    const int64_t second_padded  = params->img_padded_seq;
    const int64_t head_dim       = dst->ne[0];
    const int64_t heads          = dst->ne[1];
    const int64_t shard_heads    = heads / world_size;
    const int64_t first_shard_seq = first_padded / world_size;
    const int64_t second_shard_seq = second_padded / world_size;
    const int64_t out_shard_seq = stream_index == 0 ? first_shard_seq : second_shard_seq;
    const int64_t first_chunk    = head_dim * shard_heads * first_shard_seq;
    const int64_t second_chunk   = head_dim * shard_heads * second_shard_seq;
    const int64_t count_per_peer = first_chunk + second_chunk;
    const int64_t stream_offset  = stream_index == 0 ? 0 : first_chunk;
    const int64_t total          = head_dim * heads * out_shard_seq;

    GGML_ASSERT(stream_index == 0 || stream_index == 1);
    GGML_ASSERT(world_size > 0);
    GGML_ASSERT(heads % world_size == 0);
    GGML_ASSERT(first_padded > 0 && second_padded >= 0);
    GGML_ASSERT(first_padded % world_size == 0 && second_padded % world_size == 0);
    GGML_ASSERT(dst->ne[2] == out_shard_seq);
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(recv_flat->ne[0] == count_per_peer * world_size);

    for (int64_t linear = ith; linear < total; linear += nth) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_seq = rem;
        const int64_t src_peer = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx =
            src_peer * count_per_peer +
            stream_offset +
            d +
            local_head * head_dim +
            local_seq * head_dim * shard_heads;
        const ggml_fp16_t value_f16 = recv_flat->type == GGML_TYPE_F16 ?
                                          *reinterpret_cast<const ggml_fp16_t*>(
                                              static_cast<const char*>(recv_flat->data) + src_idx * recv_flat->nb[0]) :
                                          ggml_fp32_to_fp16(sp_fused_qkv_pack_get_f32(recv_flat, src_idx, 0, 0, 0));
        if (dst->type == GGML_TYPE_F16) {
            char* data = static_cast<char*>(dst->data);
            *reinterpret_cast<ggml_fp16_t*>(data +
                                            d * dst->nb[0] +
                                            head * dst->nb[1] +
                                            local_seq * dst->nb[2]) = value_f16;
        } else if (dst->type == GGML_TYPE_BF16) {
            char* data = static_cast<char*>(dst->data);
            *reinterpret_cast<ggml_bf16_t*>(data +
                                            d * dst->nb[0] +
                                            head * dst->nb[1] +
                                            local_seq * dst->nb[2]) =
                ggml_fp32_to_bf16(ggml_fp16_to_fp32(value_f16));
        } else {
            sp_fused_qkv_pack_set_f32(dst, d, head, local_seq, 0, ggml_fp16_to_fp32(value_f16));
        }
    }
}

void sp_recv_placeholder_noop_cpu(ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)dst;
    (void)ith;
    (void)nth;
    const SPRecvPlaceholderParams params = sp_recv_placeholder_params_from_userdata(userdata);
    GGML_ASSERT(sp_recv_placeholder_params_valid(params));
}

void check_context_tensor(ggml_context* ctx,
                          ggml_tensor* tensor,
                          const char* fn) {
    if (ctx == nullptr) {
        throw std::invalid_argument(std::string(fn) + " ctx is null");
    }
    if (tensor == nullptr) {
        throw std::invalid_argument(std::string(fn) + " tensor is null");
    }
}

void check_rank_world(int rank,
                      int world_size,
                      const char* fn) {
    if (world_size <= 0) {
        throw std::invalid_argument(std::string(fn) + " world_size must be positive");
    }
    if (rank < 0 || rank >= world_size) {
        std::ostringstream oss;
        oss << fn << " rank must be in [0, world_size): rank="
            << rank << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
}

void check_world_size(int world_size,
                      const char* fn) {
    if (world_size <= 0) {
        throw std::invalid_argument(std::string(fn) + " world_size must be positive");
    }
}

void check_seq_dim(int seq_dim,
                   const char* fn) {
    if (seq_dim != 1) {
        std::ostringstream oss;
        oss << fn << " currently supports seq_dim == 1 for DiT [hidden, sequence, batch, 1] tensors, got "
            << seq_dim;
        throw std::invalid_argument(oss.str());
    }
}

ggml_tensor* pad_sequence_dim(ggml_context* ctx,
                              ggml_tensor* input,
                              int seq_dim,
                              int64_t pad) {
    if (pad <= 0) {
        return input;
    }
    check_seq_dim(seq_dim, "pad_sequence_dim");
    if (pad > static_cast<int64_t>(INT32_MAX)) {
        throw std::invalid_argument("pad_sequence_dim pad is too large for ggml_pad");
    }
    return ggml_pad(ctx, input, 0, static_cast<int>(pad), 0, 0);
}

ggml_tensor* view_sequence_dim_1(ggml_context* ctx,
                                 ggml_tensor* input,
                                 int64_t start,
                                 int64_t length) {
    if (start < 0 || length <= 0 || start + length > input->ne[1]) {
        std::ostringstream oss;
        oss << "view_sequence_dim_1 invalid range: start=" << start
            << " length=" << length
            << " seq_len=" << input->ne[1];
        throw std::invalid_argument(oss.str());
    }
    return ggml_view_4d(ctx,
                        input,
                        input->ne[0],
                        length,
                        input->ne[2],
                        input->ne[3],
                        input->nb[1],
                        input->nb[2],
                        input->nb[3],
                        static_cast<size_t>(start) * input->nb[1]);
}

ggml_tensor* new_recv_placeholder(ggml_context* ctx,
                                  ggml_tensor* like,
                                  ggml_tensor* dependency,
                                  int64_t ne0,
                                  int64_t ne1,
                                  int64_t ne2,
                                  int64_t ne3,
                                  const std::string& name,
                                  int world_size) {
    ggml_tensor* leaf = ggml_new_tensor_4d(ctx, like->type, ne0, ne1, ne2, ne3);
    if (!name.empty()) {
        ggml_set_name(leaf, (name + "_recv_leaf").c_str());
    }
    ggml_tensor* recv = nullptr;
    if (world_size > 1) {
        ggml_tensor* args[] = {leaf, dependency};
        recv = ggml_custom_4d(ctx,
                              like->type,
                              ne0,
                              ne1,
                              ne2,
                              ne3,
                              args,
                              2,
                              sp_recv_placeholder_noop_cpu,
                              1,
                              sp_recv_placeholder_params_to_userdata(world_size));
    } else {
        recv = ggml_dup(ctx, leaf);
    }
    recv->src[1] = dependency;
    if (!name.empty()) {
        ggml_set_name(recv, (name + "_recv").c_str());
    }
    return recv;
}

std::string graph_cut_group_name(const std::string& name) {
    if (name.empty()) {
        return "sp";
    }
    return "sp:" + name;
}

void mark_all_to_all_flat(ggml_tensor* send_flat,
                          ggml_tensor* recv_flat,
                          size_t count_per_peer,
                          const std::string& name) {
    sd::ggml_graph_cut::mark_comm_op(send_flat,
                                     recv_flat,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL,
                                     name,
                                     ReduceOp::kSum,
                                     count_per_peer);
}

ggml_tensor* make_all_to_all_recv_flat(ggml_context* ctx,
                                       ggml_tensor* send_flat,
                                       int64_t recv_elems,
                                       size_t count_per_peer,
                                       ProcessGroup* process_group,
                                       int world_size,
                                       const std::string& name) {
    if (process_group != nullptr) {
        ggml_tensor* custom = edgedit::ggml_ext::flux_sp_all_to_all_custom(ctx,
                                                                           send_flat,
                                                                           recv_elems,
                                                                           static_cast<int64_t>(count_per_peer),
                                                                           world_size,
                                                                           process_group,
                                                                           name);
        if (custom != nullptr) {
            return custom;
        }
    }

    ggml_tensor* recv_flat = new_recv_placeholder(ctx,
                                                  send_flat,
                                                  send_flat,
                                                  recv_elems,
                                                  1,
                                                  1,
                                                  1,
                                                  name + "_flat",
                                                  world_size);
    mark_all_to_all_flat(send_flat,
                         recv_flat,
                         count_per_peer,
                         name);
    sd::ggml_graph_cut::mark_graph_cut(recv_flat,
                                       graph_cut_group_name(name),
                                       name + "_recv_flat");
    return recv_flat;
}

ggml_tensor* flatten_for_comm_1d(ggml_context* ctx,
                                 ggml_tensor* tensor) {
    if (ggml_is_contiguous(tensor)) {
        return ggml_reshape_1d(ctx, tensor, ggml_nelements(tensor));
    }
    return ggml_cont_1d(ctx, tensor, ggml_nelements(tensor));
}

ggml_tensor* cont_4d_if_needed(ggml_context* ctx,
                               ggml_tensor* tensor,
                               int64_t ne0,
                               int64_t ne1,
                               int64_t ne2,
                               int64_t ne3) {
    if (ggml_is_contiguous(tensor)) {
        return ggml_reshape_4d(ctx, tensor, ne0, ne1, ne2, ne3);
    }
    return ggml_cont_4d(ctx, tensor, ne0, ne1, ne2, ne3);
}

void build_seq_to_head_outputs_from_recv(ggml_context* ctx,
                                         SPAllToAll4DBatchLayout& layout,
                                         const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                         const std::string& name) {
    if (output_layouts.size() != layout.head_dims.size()) {
        throw std::invalid_argument("build_seq_to_head_outputs_from_recv output_layouts size mismatch");
    }

    ggml_tensor* mid = ggml_reshape_4d(ctx,
                                       layout.recv_flat,
                                       layout.total_head_dim,
                                       layout.shard_heads,
                                       layout.shard_sequence,
                                       layout.world_size);

    size_t offset = 0;
    layout.outputs.reserve(layout.head_dims.size());
    for (size_t i = 0; i < layout.head_dims.size(); ++i) {
        const int64_t head_dim = layout.head_dims[i];
        ggml_tensor* output_view = ggml_view_4d(ctx,
                                                mid,
                                                head_dim,
                                                layout.shard_heads,
                                                layout.shard_sequence,
                                                layout.world_size,
                                                mid->nb[1],
                                                mid->nb[2],
                                                mid->nb[3],
                                                offset);
        ggml_tensor* output = nullptr;
        if (output_layouts[i] == SPSeqToHeadOutputLayout::SeqMajor) {
            ggml_tensor* seq_before_heads = ggml_permute(ctx, output_view, 0, 3, 1, 2);
            output = cont_4d_if_needed(ctx,
                                       seq_before_heads,
                                       head_dim,
                                       layout.sequence,
                                       layout.shard_heads,
                                       layout.batch);
        } else if (output_layouts[i] == SPSeqToHeadOutputLayout::SeqMajorRopeInterleaved) {
            if (head_dim % 2 != 0) {
                throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched_layouts rope-interleaved output requires even head_dim");
            }
            if (layout.batch != 1) {
                throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched_layouts rope-interleaved output supports batch == 1 only");
            }
            ggml_tensor* interleaved_view = ggml_view_4d(ctx,
                                                         output_view,
                                                         2,
                                                         head_dim / 2,
                                                         layout.sequence,
                                                         layout.shard_heads,
                                                         output_view->nb[0] * 2,
                                                         output_view->nb[2],
                                                         output_view->nb[1],
                                                         0);
            ggml_tensor* half_seq_heads_interleaved = ggml_permute(ctx,
                                                                   interleaved_view,
                                                                   3,
                                                                   0,
                                                                   1,
                                                                   2);
            output = cont_4d_if_needed(ctx,
                                       half_seq_heads_interleaved,
                                       head_dim / 2,
                                       layout.sequence,
                                       layout.shard_heads,
                                       2);
        } else {
            output = cont_4d_if_needed(ctx,
                                       output_view,
                                       head_dim,
                                       layout.shard_heads,
                                       layout.sequence,
                                       layout.batch);
        }
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        layout.outputs.push_back(output);
        offset += static_cast<size_t>(head_dim) * mid->nb[0];
    }
}

} // namespace

int64_t sp_sequence_padding(int64_t seq_len,
                            int world_size) {
    check_world_size(world_size, "sp_sequence_padding");
    if (seq_len < 0) {
        throw std::invalid_argument("sp_sequence_padding seq_len must be non-negative");
    }
    if (seq_len == 0 || world_size == 1) {
        return 0;
    }
    return (static_cast<int64_t>(world_size) - (seq_len % world_size)) % world_size;
}

ggml_tensor* sp_split_sequence_view(ggml_context* ctx,
                                    ggml_tensor* input,
                                    int rank,
                                    int world_size,
                                    int seq_dim,
                                    int64_t* pad_out) {
    check_context_tensor(ctx, input, "sp_split_sequence_view");
    check_rank_world(rank, world_size, "sp_split_sequence_view");
    check_seq_dim(seq_dim, "sp_split_sequence_view");

    const int64_t pad = sp_sequence_padding(input->ne[seq_dim], world_size);
    if (pad_out != nullptr) {
        *pad_out = pad;
    }

    ggml_tensor* padded = pad_sequence_dim(ctx, input, seq_dim, pad);
    const int64_t local_seq_len = padded->ne[seq_dim] / world_size;
    return view_sequence_dim_1(ctx,
                               padded,
                               static_cast<int64_t>(rank) * local_seq_len,
                               local_seq_len);
}

SPSequenceSplit sp_split_sequence(ggml_context* ctx,
                                  ggml_tensor* input,
                                  int rank,
                                  int world_size,
                                  int seq_dim,
                                  const std::string& name_prefix) {
    check_context_tensor(ctx, input, "sp_split_sequence");
    check_rank_world(rank, world_size, "sp_split_sequence");
    check_seq_dim(seq_dim, "sp_split_sequence");

    SPSequenceSplit split;
    split.rank = rank;
    split.world_size = world_size;
    split.seq_dim = seq_dim;
    split.original_seq_len = input->ne[seq_dim];
    split.pad = sp_sequence_padding(split.original_seq_len, world_size);
    split.padded_seq_len = split.original_seq_len + split.pad;
    split.local_seq_len = split.padded_seq_len / world_size;

    split.input_padded = pad_sequence_dim(ctx, input, seq_dim, split.pad);
    if (!name_prefix.empty() && split.input_padded != input) {
        ggml_set_name(split.input_padded, (name_prefix + "_padded").c_str());
    }

    split.local_view = view_sequence_dim_1(ctx,
                                           split.input_padded,
                                           static_cast<int64_t>(rank) * split.local_seq_len,
                                           split.local_seq_len);
    if (!name_prefix.empty()) {
        ggml_set_name(split.local_view, (name_prefix + "_local_view").c_str());
    }

    split.local = ggml_cont(ctx, split.local_view);
    if (!name_prefix.empty()) {
        ggml_set_name(split.local, (name_prefix + "_local").c_str());
    }

    return split;
}

SPSequenceGather sp_mark_gather_sequence(ggml_context* ctx,
                                         ggml_tensor* local,
                                         int world_size,
                                         int seq_dim,
                                         int64_t pad,
                                         const std::string& name,
                                         ProcessGroup* process_group) {
    check_context_tensor(ctx, local, "sp_mark_gather_sequence");
    check_world_size(world_size, "sp_mark_gather_sequence");
    check_seq_dim(seq_dim, "sp_mark_gather_sequence");
    if (pad < 0 || pad >= local->ne[seq_dim] * static_cast<int64_t>(world_size)) {
        std::ostringstream oss;
        oss << "sp_mark_gather_sequence invalid pad=" << pad
            << " local_seq_len=" << local->ne[seq_dim]
            << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (local->ne[2] != 1 || local->ne[3] != 1) {
        throw std::invalid_argument("sp_mark_gather_sequence phase 1 supports [hidden, sequence, 1, 1] only");
    }

    SPSequenceGather gather;
    gather.world_size = world_size;
    gather.seq_dim = seq_dim;
    gather.local_seq_len = local->ne[seq_dim];
    gather.padded_seq_len = gather.local_seq_len * world_size;
    gather.pad = pad;
    gather.original_seq_len = gather.padded_seq_len - pad;
    gather.count_per_rank = static_cast<size_t>(ggml_nelements(local));

    if (process_group != nullptr) {
        ggml_tensor* send_flat = flatten_for_comm_1d(ctx, local);
        if (!name.empty()) {
            ggml_set_name(send_flat, (name + "_send_flat").c_str());
        }

        ggml_tensor* recv_flat = edgedit::ggml_ext::flux_sp_all_gather_custom(
            ctx,
            send_flat,
            static_cast<int64_t>(gather.count_per_rank) * world_size,
            static_cast<int64_t>(gather.count_per_rank),
            world_size,
            process_group,
            name);
        if (recv_flat != nullptr) {
            gather.recv = recv_flat;
            gather.gathered_padded = ggml_reshape_4d(ctx,
                                                     recv_flat,
                                                     local->ne[0],
                                                     gather.padded_seq_len,
                                                     local->ne[2],
                                                     local->ne[3]);
            if (!name.empty()) {
                ggml_set_name(gather.gathered_padded, (name + "_padded").c_str());
            }

            if (pad > 0) {
                gather.gathered = view_sequence_dim_1(ctx,
                                                      gather.gathered_padded,
                                                      0,
                                                      gather.original_seq_len);
            } else {
                gather.gathered = ggml_dup(ctx, gather.gathered_padded);
            }
            if (!name.empty()) {
                ggml_set_name(gather.gathered, (name + "_output").c_str());
            }
            return gather;
        }
    }

    // With batch == 1, ProcessGroup all_gather rank-major blocks are exactly
    // contiguous sequence shards laid out as [hidden, padded_sequence, 1, 1].
    gather.recv = new_recv_placeholder(ctx,
                                       local,
                                       local,
                                       local->ne[0],
                                       gather.padded_seq_len,
                                       local->ne[2],
                                       local->ne[3],
                                       name,
                                       world_size);

    sd::ggml_graph_cut::mark_comm_op(local,
                                     gather.recv,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER,
                                     name);
    sd::ggml_graph_cut::mark_graph_cut(gather.recv,
                                       graph_cut_group_name(name),
                                       name + "_recv");

    gather.gathered_padded = gather.recv;

    if (pad > 0) {
        gather.gathered = view_sequence_dim_1(ctx,
                                              gather.gathered_padded,
                                              0,
                                              gather.original_seq_len);
    } else {
        gather.gathered = ggml_dup(ctx, gather.gathered_padded);
    }
    if (!name.empty()) {
        ggml_set_name(gather.gathered, (name + "_output").c_str());
    }

    return gather;
}

SPSequenceGatherBatch sp_mark_gather_sequence_batched(ggml_context* ctx,
                                                      const std::vector<ggml_tensor*>& locals,
                                                      int world_size,
                                                      int seq_dim,
                                                      const std::vector<int64_t>& pads,
                                                      const std::string& name) {
    check_world_size(world_size, "sp_mark_gather_sequence_batched");
    check_seq_dim(seq_dim, "sp_mark_gather_sequence_batched");
    if (locals.empty()) {
        throw std::invalid_argument("sp_mark_gather_sequence_batched locals must not be empty");
    }
    if (locals.size() != pads.size()) {
        throw std::invalid_argument("sp_mark_gather_sequence_batched locals and pads size mismatch");
    }

    SPSequenceGatherBatch gather;
    gather.world_size = world_size;
    gather.seq_dim = seq_dim;
    gather.gathered_padded.reserve(locals.size());
    gather.gathered.reserve(locals.size());
    gather.local_seq_lens.reserve(locals.size());
    gather.padded_seq_lens.reserve(locals.size());
    gather.original_seq_lens.reserve(locals.size());
    gather.pads.reserve(locals.size());
    gather.counts_per_input.reserve(locals.size());

    std::vector<ggml_tensor*> send_chunks;
    send_chunks.reserve(locals.size());
    for (size_t i = 0; i < locals.size(); ++i) {
        ggml_tensor* local = locals[i];
        check_context_tensor(ctx, local, "sp_mark_gather_sequence_batched");
        if (local->ne[2] != 1 || local->ne[3] != 1) {
            throw std::invalid_argument("sp_mark_gather_sequence_batched phase 1 supports [hidden, sequence, 1, 1] only");
        }

        const int64_t pad = pads[i];
        if (pad < 0 || pad >= local->ne[seq_dim] * static_cast<int64_t>(world_size)) {
            std::ostringstream oss;
            oss << "sp_mark_gather_sequence_batched invalid pad=" << pad
                << " local_seq_len=" << local->ne[seq_dim]
                << " world_size=" << world_size;
            throw std::invalid_argument(oss.str());
        }

        const int64_t local_seq_len = local->ne[seq_dim];
        const int64_t padded_seq_len = local_seq_len * world_size;
        const int64_t original_seq_len = padded_seq_len - pad;
        const size_t count = static_cast<size_t>(ggml_nelements(local));

        gather.local_seq_lens.push_back(local_seq_len);
        gather.padded_seq_lens.push_back(padded_seq_len);
        gather.original_seq_lens.push_back(original_seq_len);
        gather.pads.push_back(pad);
        gather.counts_per_input.push_back(count);
        gather.count_per_rank += count;

        ggml_tensor* flat = flatten_for_comm_1d(ctx, local);
        if (!name.empty()) {
            ggml_set_name(flat, (name + "_send_" + std::to_string(i)).c_str());
        }
        send_chunks.push_back(flat);
    }

    gather.send_flat = send_chunks.front();
    for (size_t i = 1; i < send_chunks.size(); ++i) {
        gather.send_flat = ggml_concat(ctx, gather.send_flat, send_chunks[i], 0);
    }
    gather.send_flat = flatten_for_comm_1d(ctx, gather.send_flat);
    if (!name.empty()) {
        ggml_set_name(gather.send_flat, (name + "_send_flat").c_str());
    }

    gather.recv_flat = new_recv_placeholder(ctx,
                                            gather.send_flat,
                                            gather.send_flat,
                                            ggml_nelements(gather.send_flat) * world_size,
                                            1,
                                            1,
                                            1,
                                            name + "_flat",
                                            world_size);

    sd::ggml_graph_cut::mark_comm_op(gather.send_flat,
                                     gather.recv_flat,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER,
                                     name);
    sd::ggml_graph_cut::mark_graph_cut(gather.recv_flat,
                                       graph_cut_group_name(name),
                                       name + "_recv_flat");

    size_t input_offset = 0;
    for (size_t i = 0; i < locals.size(); ++i) {
        const size_t count = gather.counts_per_input[i];
        ggml_tensor* flat_view = nullptr;
        for (int src = 0; src < world_size; ++src) {
            const size_t offset = (static_cast<size_t>(src) * gather.count_per_rank +
                                   input_offset) *
                                  gather.recv_flat->nb[0];
            ggml_tensor* src_view = ggml_view_1d(ctx,
                                                 gather.recv_flat,
                                                 count,
                                                 offset);
            if (flat_view == nullptr) {
                flat_view = src_view;
            } else {
                flat_view = ggml_concat(ctx, flat_view, src_view, 0);
            }
        }

        ggml_tensor* padded = ggml_reshape_4d(ctx,
                                              flat_view,
                                              locals[i]->ne[0],
                                              gather.padded_seq_lens[i],
                                              locals[i]->ne[2],
                                              locals[i]->ne[3]);
        if (!name.empty()) {
            ggml_set_name(padded, (name + "_padded_" + std::to_string(i)).c_str());
        }
        gather.gathered_padded.push_back(padded);

        ggml_tensor* output = nullptr;
        if (gather.pads[i] > 0) {
            output = view_sequence_dim_1(ctx,
                                         padded,
                                         0,
                                         gather.original_seq_lens[i]);
        } else {
            output = ggml_dup(ctx, padded);
        }
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        gather.gathered.push_back(output);
        input_offset += count;
    }

    return gather;
}

SPDoubleToSingleReshard sp_double_to_single_reshard_sequence_2way(ggml_context* ctx,
                                                                  ggml_tensor* first_local,
                                                                  ggml_tensor* second_local,
                                                                  int rank,
                                                                  int world_size,
                                                                  ProcessGroup* process_group,
                                                                  const std::string& name) {
    check_context_tensor(ctx, first_local, "sp_double_to_single_reshard_sequence_2way");
    check_context_tensor(ctx, second_local, "sp_double_to_single_reshard_sequence_2way");
    check_rank_world(rank, world_size, "sp_double_to_single_reshard_sequence_2way");

    SPDoubleToSingleReshard reshard;
    reshard.rank = rank;
    reshard.world_size = world_size;

    if (world_size != 2 ||
        process_group == nullptr ||
        first_local->type != second_local->type ||
        (first_local->type != GGML_TYPE_F32 && first_local->type != GGML_TYPE_F16) ||
        first_local->ne[0] != second_local->ne[0] ||
        first_local->ne[2] != 1 ||
        first_local->ne[3] != 1 ||
        second_local->ne[2] != 1 ||
        second_local->ne[3] != 1 ||
        first_local->ne[1] <= 0 ||
        second_local->ne[1] <= 0 ||
        first_local->ne[1] >= second_local->ne[1]) {
        return reshard;
    }

    const int64_t hidden = first_local->ne[0];
    const int64_t first_local_seq = first_local->ne[1];
    const int64_t second_local_seq = second_local->ne[1];
    const int64_t total_first_seq = first_local_seq * 2;
    const int64_t total_second_seq = second_local_seq * 2;
    const int64_t combined_seq = total_first_seq + total_second_seq;
    if ((combined_seq % 2) != 0) {
        return reshard;
    }
    const int64_t local_seq = combined_seq / 2;
    if (local_seq != first_local_seq + second_local_seq) {
        return reshard;
    }

    reshard.first_local_seq = first_local_seq;
    reshard.second_local_seq = second_local_seq;
    reshard.local_seq_len = local_seq;
    reshard.count_per_peer = static_cast<size_t>(hidden * first_local_seq);

    ggml_tensor* first_cont = ggml_is_contiguous(first_local) ?
                                  first_local :
                                  ggml_cont(ctx, first_local);
    ggml_tensor* second_cont = ggml_is_contiguous(second_local) ?
                                   second_local :
                                   ggml_cont(ctx, second_local);
    if (!name.empty()) {
        if (first_cont != first_local) {
            ggml_set_name(first_cont, (name + "_first_local_cont").c_str());
        }
        if (second_cont != second_local) {
            ggml_set_name(second_cont, (name + "_second_local_cont").c_str());
        }
    }

    ggml_tensor* second_tail = view_sequence_dim_1(ctx,
                                                   second_cont,
                                                   second_local_seq - first_local_seq,
                                                   first_local_seq);
    if (!name.empty()) {
        ggml_set_name(second_tail, (name + "_second_tail").c_str());
    }

    reshard.send_flat = ggml_concat(ctx,
                                    flatten_for_comm_1d(ctx, first_cont),
                                    flatten_for_comm_1d(ctx, second_tail),
                                    0);
    reshard.send_flat = flatten_for_comm_1d(ctx, reshard.send_flat);
    if (!name.empty()) {
        ggml_set_name(reshard.send_flat, (name + "_send_flat").c_str());
    }

    reshard.recv_flat = make_all_to_all_recv_flat(ctx,
                                                  reshard.send_flat,
                                                  ggml_nelements(reshard.send_flat),
                                                  reshard.count_per_peer,
                                                  process_group,
                                                  world_size,
                                                  name);

    ggml_tensor* recv_from_rank0 = ggml_view_1d(ctx,
                                                reshard.recv_flat,
                                                static_cast<int64_t>(reshard.count_per_peer),
                                                0);
    ggml_tensor* recv_from_rank1 = ggml_view_1d(ctx,
                                                reshard.recv_flat,
                                                static_cast<int64_t>(reshard.count_per_peer),
                                                static_cast<size_t>(reshard.count_per_peer) * reshard.recv_flat->nb[0]);
    ggml_tensor* peer_boundary = ggml_reshape_4d(ctx,
                                                 rank == 0 ? recv_from_rank1 : recv_from_rank0,
                                                 hidden,
                                                 first_local_seq,
                                                 1,
                                                 1);
    if (!name.empty()) {
        ggml_set_name(peer_boundary, (name + "_peer_boundary").c_str());
    }

    if (rank == 0) {
        ggml_tensor* first_full = ggml_concat(ctx, first_cont, peer_boundary, 1);
        ggml_tensor* second_take = view_sequence_dim_1(ctx,
                                                       second_cont,
                                                       0,
                                                       second_local_seq - first_local_seq);
        reshard.local = ggml_concat(ctx, first_full, second_take, 1);
    } else {
        reshard.local = ggml_concat(ctx, peer_boundary, second_cont, 1);
    }
    if (!name.empty()) {
        ggml_set_name(reshard.local, (name + "_local").c_str());
    }

    return reshard;
}

SPAllToAll4DLayout sp_all_to_all_4d_seq_to_head(ggml_context* ctx,
                                                ggml_tensor* input,
                                                ProcessGroup* process_group,
                                                int world_size,
                                                const std::string& name) {
    check_context_tensor(ctx, input, "sp_all_to_all_4d_seq_to_head");
    check_world_size(world_size, "sp_all_to_all_4d_seq_to_head");
    if (input->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head phase 1 supports batch == 1 only");
    }
    if (input->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_seq_to_head heads must be divisible by world_size: heads="
            << input->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DLayout layout;
    layout.direction = SPAllToAll4DDirection::kSeqToHead;
    layout.world_size = world_size;
    layout.batch = input->ne[3];
    layout.head_dim = input->ne[0];
    layout.heads = input->ne[1];
    layout.shard_heads = input->ne[1] / world_size;
    layout.shard_sequence = input->ne[2];
    layout.sequence = input->ne[2] * world_size;

    // input [hs, hc, shard_seq, 1]
    // -> [hs, shard_heads, P, shard_seq]
    // -> [hs, shard_heads, shard_seq, P] peer-major after contiguous flatten.
    ggml_tensor* reshaped = ggml_reshape_4d(ctx,
                                            input,
                                            layout.head_dim,
                                            layout.shard_heads,
                                            world_size,
                                            layout.shard_sequence);
    ggml_tensor* peer_last = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 1, 3, 2));
    layout.send_flat = flatten_for_comm_1d(ctx, peer_last);
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                            static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 ggml_nelements(layout.send_flat),
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    ggml_tensor* mid = ggml_reshape_4d(ctx,
                                       layout.recv_flat,
                                       layout.head_dim,
                                       layout.shard_heads,
                                       layout.shard_sequence,
                                       world_size);
    layout.output = cont_4d_if_needed(ctx,
                                      mid,
                                      layout.head_dim,
                                      layout.shard_heads,
                                      layout.sequence,
                                      layout.batch);
    if (!name.empty()) {
        ggml_set_name(layout.output, (name + "_output").c_str());
    }
    return layout;
}

SPAllToAll4DLayout sp_all_to_all_4d_seq_to_head(ggml_context* ctx,
                                                ggml_tensor* input,
                                                int world_size,
                                                const std::string& name) {
    return sp_all_to_all_4d_seq_to_head(ctx, input, nullptr, world_size, name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             ProcessGroup* process_group,
                                                             int world_size,
                                                             const std::string& name) {
    return sp_all_to_all_4d_seq_to_head_batched_mixed(ctx,
                                                      inputs,
                                                      std::vector<bool>(inputs.size(), false),
                                                      process_group,
                                                      world_size,
                                                      name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             int world_size,
                                                             const std::string& name) {
    return sp_all_to_all_4d_seq_to_head_batched(ctx, inputs, nullptr, world_size, name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_mixed(ggml_context* ctx,
                                                                   const std::vector<ggml_tensor*>& inputs,
                                                                   const std::vector<bool>& output_seq_major,
                                                                   ProcessGroup* process_group,
                                                                   int world_size,
                                                                   const std::string& name) {
    std::vector<SPSeqToHeadOutputLayout> output_layouts;
    output_layouts.reserve(output_seq_major.size());
    for (bool seq_major : output_seq_major) {
        output_layouts.push_back(seq_major ? SPSeqToHeadOutputLayout::SeqMajor :
                                             SPSeqToHeadOutputLayout::HeadMajor);
    }
    return sp_all_to_all_4d_seq_to_head_batched_layouts(ctx,
                                                        inputs,
                                                        output_layouts,
                                                        process_group,
                                                        world_size,
                                                        name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_mixed(ggml_context* ctx,
                                                                   const std::vector<ggml_tensor*>& inputs,
                                                                   const std::vector<bool>& output_seq_major,
                                                                   int world_size,
                                                                   const std::string& name) {
    return sp_all_to_all_4d_seq_to_head_batched_mixed(ctx,
                                                      inputs,
                                                      output_seq_major,
                                                      nullptr,
                                                      world_size,
                                                      name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_layouts(ggml_context* ctx,
                                                                     const std::vector<ggml_tensor*>& inputs,
                                                                     const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                                                     ProcessGroup* process_group,
                                                                     int world_size,
                                                                     const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_seq_to_head_batched");
    if (inputs.empty()) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched inputs must not be empty");
    }
    if (output_layouts.size() != inputs.size()) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched_layouts output_layouts size mismatch");
    }

    ggml_tensor* first = inputs.front();
    check_context_tensor(ctx, first, "sp_all_to_all_4d_seq_to_head_batched");
    if (first->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched phase 1 supports batch == 1 only");
    }
    if (first->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_seq_to_head_batched heads must be divisible by world_size: heads="
            << first->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kSeqToHead;
    layout.world_size = world_size;
    layout.batch = first->ne[3];
    layout.heads = first->ne[1];
    layout.shard_heads = first->ne[1] / world_size;
    layout.shard_sequence = first->ne[2];
    layout.sequence = first->ne[2] * world_size;
    layout.outputs.reserve(inputs.size());
    layout.head_dims.reserve(inputs.size());

    std::vector<ggml_tensor*> contiguous_inputs;
    contiguous_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        ggml_tensor* input = inputs[i];
        check_context_tensor(ctx, input, "sp_all_to_all_4d_seq_to_head_batched");
        if (input->ne[1] != layout.heads ||
            input->ne[2] != layout.shard_sequence ||
            input->ne[3] != layout.batch) {
            std::ostringstream oss;
            oss << "sp_all_to_all_4d_seq_to_head_batched input " << i
                << " shape mismatch";
            throw std::invalid_argument(oss.str());
        }
        layout.head_dims.push_back(input->ne[0]);
        layout.total_head_dim += input->ne[0];
        contiguous_inputs.push_back(ggml_is_contiguous(input) ? input : ggml_cont(ctx, input));
    }

    ggml_tensor* combined = contiguous_inputs.front();
    for (size_t i = 1; i < contiguous_inputs.size(); ++i) {
        combined = ggml_concat(ctx, combined, contiguous_inputs[i], 0);
    }
    if (!name.empty()) {
        ggml_set_name(combined, (name + "_combined").c_str());
    }

    // combined [sum_head_dim, heads, shard_seq, 1]
    // -> [sum_head_dim, shard_heads, P, shard_seq]
    // -> [sum_head_dim, shard_heads, shard_seq, P] peer-major after flatten.
    ggml_tensor* reshaped = ggml_reshape_4d(ctx,
                                            combined,
                                            layout.total_head_dim,
                                            layout.shard_heads,
                                            world_size,
                                            layout.shard_sequence);
    ggml_tensor* peer_last = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 1, 3, 2));
    layout.send_flat = flatten_for_comm_1d(ctx, peer_last);
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                            static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 ggml_nelements(layout.send_flat),
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    build_seq_to_head_outputs_from_recv(ctx, layout, output_layouts, name);

    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_layouts(ggml_context* ctx,
                                                                     const std::vector<ggml_tensor*>& inputs,
                                                                     const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                                                     int world_size,
                                                                     const std::string& name) {
    return sp_all_to_all_4d_seq_to_head_batched_layouts(ctx,
                                                        inputs,
                                                        output_layouts,
                                                        nullptr,
                                                        world_size,
                                                        name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_packed_recv_only(ggml_context* ctx,
                                                                      ggml_tensor* send_flat,
                                                                      int64_t total_head_dim,
                                                                      int64_t heads,
                                                                      int64_t shard_sequence,
                                                                      int64_t batch,
                                                                      ProcessGroup* process_group,
                                                                      int world_size,
                                                                      const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_seq_to_head_packed_recv_only");
    check_context_tensor(ctx, send_flat, "sp_all_to_all_4d_seq_to_head_packed_recv_only");
    if (batch != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_packed_recv_only supports batch == 1 only");
    }
    if (heads % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_seq_to_head_packed_recv_only heads must be divisible by world_size: heads="
            << heads << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction       = SPAllToAll4DDirection::kSeqToHead;
    layout.world_size      = world_size;
    layout.batch           = batch;
    layout.heads           = heads;
    layout.shard_heads     = heads / world_size;
    layout.shard_sequence  = shard_sequence;
    layout.sequence        = shard_sequence * world_size;
    layout.total_head_dim  = total_head_dim;
    layout.send_flat       = ggml_is_contiguous(send_flat) ? ggml_reshape_1d(ctx, send_flat, ggml_nelements(send_flat))
                                                          : ggml_cont_1d(ctx, send_flat, ggml_nelements(send_flat));
    layout.count_per_peer  = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                             static_cast<size_t>(world_size);

    const int64_t expected = total_head_dim * layout.shard_heads * world_size * shard_sequence * batch;
    if (ggml_nelements(layout.send_flat) != expected) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_seq_to_head_packed_recv_only send_flat size mismatch: got="
            << ggml_nelements(layout.send_flat) << " expected=" << expected;
        throw std::invalid_argument(oss.str());
    }

    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 ggml_nelements(layout.send_flat),
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);
    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_packed_recv_only(ggml_context* ctx,
                                                                      ggml_tensor* send_flat,
                                                                      int64_t total_head_dim,
                                                                      int64_t heads,
                                                                      int64_t shard_sequence,
                                                                      int64_t batch,
                                                                      int world_size,
                                                                      const std::string& name) {
    return sp_all_to_all_4d_seq_to_head_packed_recv_only(ctx,
                                                         send_flat,
                                                         total_head_dim,
                                                         heads,
                                                         shard_sequence,
                                                         batch,
                                                         nullptr,
                                                         world_size,
                                                         name);
}

ggml_tensor* sp_qkv_seq_to_head_send_pack(ggml_context* ctx,
                                          ggml_tensor* q,
                                          ggml_tensor* k,
                                          ggml_tensor* v,
                                          int world_size,
                                          const std::string& name) {
    check_context_tensor(ctx, q, "sp_qkv_seq_to_head_send_pack");
    check_context_tensor(ctx, k, "sp_qkv_seq_to_head_send_pack");
    check_context_tensor(ctx, v, "sp_qkv_seq_to_head_send_pack");
    check_world_size(world_size, "sp_qkv_seq_to_head_send_pack");
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack supports F32 q/k/v only");
    }
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack supports batch == 1 only");
    }
    if (q->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_qkv_seq_to_head_send_pack heads must be divisible by world_size: heads="
            << q->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] ||
        k->ne[1] != q->ne[1] || v->ne[1] != q->ne[1] ||
        k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] ||
        k->ne[3] != q->ne[3] || v->ne[3] != q->ne[3]) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack q/k/v shape mismatch");
    }

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
                                      sp_qkv_seq_to_head_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(world_size, 0, 6));
    if (!name.empty()) {
        ggml_set_name(out, (name + "_fused_send_pack").c_str());
    }
    return out;
}

ggml_tensor* sp_qkv_seq_to_head_send_pack_mixed(ggml_context* ctx,
                                                ggml_tensor* q,
                                                ggml_tensor* k,
                                                ggml_tensor* v,
                                                int world_size,
                                                const std::string& name) {
    check_context_tensor(ctx, q, "sp_qkv_seq_to_head_send_pack_mixed");
    check_context_tensor(ctx, k, "sp_qkv_seq_to_head_send_pack_mixed");
    check_context_tensor(ctx, v, "sp_qkv_seq_to_head_send_pack_mixed");
    check_world_size(world_size, "sp_qkv_seq_to_head_send_pack_mixed");
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack_mixed supports F32 q/k/v only");
    }
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack_mixed supports batch == 1 only");
    }
    if (q->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_qkv_seq_to_head_send_pack_mixed heads must be divisible by world_size: heads="
            << q->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] ||
        k->ne[1] != q->ne[1] || v->ne[1] != q->ne[1] ||
        k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] ||
        k->ne[3] != q->ne[3] || v->ne[3] != q->ne[3]) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack_mixed q/k/v shape mismatch");
    }

    const int64_t packed_dim     = q->ne[0] * 2;
    const int64_t shard_heads    = q->ne[1] / world_size;
    const int64_t flat_elems     = packed_dim * shard_heads * q->ne[2] * world_size;
    ggml_tensor* args[] = {q, k, v};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      flat_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      3,
                                      sp_qkv_seq_to_head_send_pack_mixed_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(world_size, 0, 18));
    if (!name.empty()) {
        ggml_set_name(out, (name + "_mixed_send_pack").c_str());
    }
    return out;
}

ggml_tensor* sp_qkv_seq_to_head_send_pack_f16(ggml_context* ctx,
                                              ggml_tensor* q,
                                              ggml_tensor* k,
                                              ggml_tensor* v,
                                              int world_size,
                                              const std::string& name) {
    check_context_tensor(ctx, q, "sp_qkv_seq_to_head_send_pack_f16");
    check_context_tensor(ctx, k, "sp_qkv_seq_to_head_send_pack_f16");
    check_context_tensor(ctx, v, "sp_qkv_seq_to_head_send_pack_f16");
    check_world_size(world_size, "sp_qkv_seq_to_head_send_pack_f16");
    if (!sp_fused_qkv_pack_f32_or_f16(q) ||
        !sp_fused_qkv_pack_f32_or_f16(k) ||
        !sp_fused_qkv_pack_f32_or_f16(v)) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack_f16 supports F32/F16 q/k/v only");
    }
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack_f16 supports batch == 1 only");
    }
    if (q->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_qkv_seq_to_head_send_pack_f16 heads must be divisible by world_size: heads="
            << q->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] ||
        k->ne[1] != q->ne[1] || v->ne[1] != q->ne[1] ||
        k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] ||
        k->ne[3] != q->ne[3] || v->ne[3] != q->ne[3]) {
        throw std::invalid_argument("sp_qkv_seq_to_head_send_pack_f16 q/k/v shape mismatch");
    }

    const int64_t total_head_dim = q->ne[0] * 3;
    const int64_t shard_heads    = q->ne[1] / world_size;
    const int64_t flat_elems     = total_head_dim * shard_heads * q->ne[2] * world_size;
    ggml_tensor* args[] = {q, k, v};
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      flat_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      3,
                                      sp_qkv_seq_to_head_send_pack_f16_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(world_size, 0, 46));
    if (!name.empty()) {
        ggml_set_name(out, (name + "_f16_send_pack").c_str());
    }
    return out;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_qkv_seq_to_head_packed_layouts(ggml_context* ctx,
                                                                        ggml_tensor* q,
                                                                        ggml_tensor* k,
                                                                        ggml_tensor* v,
                                                                        const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                                                        ProcessGroup* process_group,
                                                                        int world_size,
                                                                        const std::string& name) {
    if (output_layouts.size() != 3) {
        throw std::invalid_argument("sp_all_to_all_4d_qkv_seq_to_head_packed_layouts expects three output layouts");
    }
    ggml_tensor* send_flat = sp_qkv_seq_to_head_send_pack(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          world_size,
                                                          name);
    SPAllToAll4DBatchLayout layout =
        sp_all_to_all_4d_seq_to_head_packed_recv_only(ctx,
                                                      send_flat,
                                                      q->ne[0] * 3,
                                                      q->ne[1],
                                                      q->ne[2],
                                                      q->ne[3],
                                                      process_group,
                                                      world_size,
                                                      name);
    layout.head_dims = {q->ne[0], k->ne[0], v->ne[0]};
    layout.outputs.clear();
    build_seq_to_head_outputs_from_recv(ctx, layout, output_layouts, name);
    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_qkv_seq_to_head_mixed_recv_only(ggml_context* ctx,
                                                                         ggml_tensor* q,
                                                                         ggml_tensor* k,
                                                                         ggml_tensor* v,
                                                                         ProcessGroup* process_group,
                                                                         int world_size,
                                                                         const std::string& name) {
    ggml_tensor* send_flat = sp_qkv_seq_to_head_send_pack_mixed(ctx,
                                                                q,
                                                                k,
                                                                v,
                                                                world_size,
                                                                name);
    SPAllToAll4DBatchLayout layout =
        sp_all_to_all_4d_seq_to_head_packed_recv_only(ctx,
                                                      send_flat,
                                                      q->ne[0] * 2,
                                                      q->ne[1],
                                                      q->ne[2],
                                                      q->ne[3],
                                                      process_group,
                                                      world_size,
                                                      name);
    layout.head_dims = {q->ne[0], k->ne[0], v->ne[0]};
    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_qkv_seq_to_head_f16_recv_only(ggml_context* ctx,
                                                                       ggml_tensor* q,
                                                                       ggml_tensor* k,
                                                                       ggml_tensor* v,
                                                                       ProcessGroup* process_group,
                                                                       int world_size,
                                                                       const std::string& name) {
    ggml_tensor* send_flat = sp_qkv_seq_to_head_send_pack_f16(ctx,
                                                              q,
                                                              k,
                                                              v,
                                                              world_size,
                                                              name);
    SPAllToAll4DBatchLayout layout =
        sp_all_to_all_4d_seq_to_head_packed_recv_only(ctx,
                                                      send_flat,
                                                      q->ne[0] * 3,
                                                      q->ne[1],
                                                      q->ne[2],
                                                      q->ne[3],
                                                      process_group,
                                                      world_size,
                                                      name);
    layout.head_dims = {q->ne[0], k->ne[0], v->ne[0]};
    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only(
    ggml_context* ctx,
    ggml_tensor* first_q,
    ggml_tensor* first_k,
    ggml_tensor* first_v,
    ggml_tensor* second_q,
    ggml_tensor* second_k,
    ggml_tensor* second_v,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name) {
    check_context_tensor(ctx, first_q, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    check_context_tensor(ctx, first_k, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    check_context_tensor(ctx, first_v, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    check_context_tensor(ctx, second_q, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    check_context_tensor(ctx, second_k, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    check_context_tensor(ctx, second_v, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    check_world_size(world_size, "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only");
    if (!sp_fused_qkv_pack_f32_or_f16(first_q) ||
        !sp_fused_qkv_pack_f32_or_f16(first_k) ||
        !sp_fused_qkv_pack_f32_or_f16(first_v) ||
        !sp_fused_qkv_pack_f32_or_f16(second_q) ||
        !sp_fused_qkv_pack_f32_or_f16(second_k) ||
        !sp_fused_qkv_pack_f32_or_f16(second_v)) {
        throw std::invalid_argument("sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only supports F32/F16 q/k/v only");
    }
    if (first_q->ne[3] != 1 || first_k->ne[3] != 1 || first_v->ne[3] != 1 ||
        second_q->ne[3] != 1 || second_k->ne[3] != 1 || second_v->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only supports batch == 1 only");
    }
    if (first_q->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only heads must be divisible by world_size: heads="
            << first_q->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    auto check_same_stream_shape = [](const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v, const char* label) {
        if (k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] ||
            k->ne[1] != q->ne[1] || v->ne[1] != q->ne[1] ||
            k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] ||
            k->ne[3] != q->ne[3] || v->ne[3] != q->ne[3]) {
            std::ostringstream oss;
            oss << "sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only " << label
                << " q/k/v shape mismatch";
            throw std::invalid_argument(oss.str());
        }
    };
    check_same_stream_shape(first_q, first_k, first_v, "first");
    check_same_stream_shape(second_q, second_k, second_v, "second");
    if (second_q->ne[0] != first_q->ne[0] ||
        second_q->ne[1] != first_q->ne[1] ||
        second_q->ne[3] != first_q->ne[3]) {
        throw std::invalid_argument("sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only stream shape mismatch");
    }

    const int64_t total_head_dim = first_q->ne[0] * 3;
    const int64_t shard_heads = first_q->ne[1] / world_size;
    const int64_t first_shard_sequence = first_q->ne[2];
    const int64_t second_shard_sequence = second_q->ne[2];
    const int64_t count_per_peer = total_head_dim * shard_heads * (first_shard_sequence + second_shard_sequence);
    const int64_t flat_elems = count_per_peer * world_size;
    ggml_tensor* args[] = {first_q, first_k, first_v, second_q, second_k, second_v};
    ggml_tensor* send_flat = ggml_custom_4d(ctx,
                                            GGML_TYPE_F16,
                                            flat_elems,
                                            1,
                                            1,
                                            1,
                                            args,
                                            6,
                                            sp_double_qkv_seq_to_head_send_pack_f16_cpu,
                                            GGML_N_TASKS_MAX,
                                            sp_fused_qkv_pack_make_params(first_shard_sequence,
                                                                          second_shard_sequence,
                                                                          47,
                                                                          0,
                                                                          0,
                                                                          world_size));
    if (!name.empty()) {
        ggml_set_name(send_flat, (name + "_f16_send_pack").c_str());
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kSeqToHead;
    layout.world_size = world_size;
    layout.batch = first_q->ne[3];
    layout.heads = first_q->ne[1];
    layout.shard_heads = shard_heads;
    layout.shard_sequence = first_shard_sequence + second_shard_sequence;
    layout.sequence = layout.shard_sequence * world_size;
    layout.total_head_dim = total_head_dim;
    layout.send_flat = send_flat;
    layout.count_per_peer = static_cast<size_t>(count_per_peer);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 flat_elems,
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);
    layout.head_dims = {first_q->ne[0], first_k->ne[0], first_v->ne[0]};
    layout.sequences = {first_shard_sequence * world_size, second_shard_sequence * world_size};
    layout.shard_sequences = {first_shard_sequence, second_shard_sequence};
    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_qkv_seq_to_head_packed_layouts(ggml_context* ctx,
                                                                        ggml_tensor* q,
                                                                        ggml_tensor* k,
                                                                        ggml_tensor* v,
                                                                        const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                                                        int world_size,
                                                                        const std::string& name) {
    return sp_all_to_all_4d_qkv_seq_to_head_packed_layouts(ctx,
                                                           q,
                                                           k,
                                                           v,
                                                           output_layouts,
                                                           nullptr,
                                                           world_size,
                                                           name);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq(ggml_context* ctx,
                                                ggml_tensor* input,
                                                ProcessGroup* process_group,
                                                int world_size,
                                                const std::string& name) {
    check_context_tensor(ctx, input, "sp_all_to_all_4d_head_to_seq");
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq");
    if (input->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq phase 1 supports batch == 1 only");
    }
    if (input->ne[2] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq sequence must be divisible by world_size: sequence="
            << input->ne[2] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DLayout layout;
    layout.direction = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size = world_size;
    layout.batch = input->ne[3];
    layout.head_dim = input->ne[0];
    layout.shard_heads = input->ne[1];
    layout.heads = input->ne[1] * world_size;
    layout.sequence = input->ne[2];
    layout.shard_sequence = input->ne[2] / world_size;

    // input [hs, shard_heads, seq, 1]
    // -> [hs, shard_heads, shard_seq, P], where the last dimension is the
    // destination peer. Contiguous flatten sends one contiguous block per peer.
    ggml_tensor* reshaped = ggml_reshape_4d(ctx,
                                            input,
                                            layout.head_dim,
                                            layout.shard_heads,
                                            layout.shard_sequence,
                                            world_size);
    layout.send_flat = flatten_for_comm_1d(ctx, reshaped);
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                            static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 ggml_nelements(layout.send_flat),
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    ggml_tensor* mid = ggml_reshape_4d(ctx,
                                       layout.recv_flat,
                                       layout.head_dim,
                                       layout.shard_heads,
                                       layout.shard_sequence,
                                       world_size);
    ggml_tensor* heads_before_sequence = ggml_permute(ctx, mid, 0, 1, 3, 2);
    layout.output = cont_4d_if_needed(ctx,
                                      heads_before_sequence,
                                      layout.head_dim,
                                      layout.heads,
                                      layout.shard_sequence,
                                      layout.batch);
    if (!name.empty()) {
        ggml_set_name(layout.output, (name + "_output").c_str());
    }
    return layout;
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq(ggml_context* ctx,
                                                ggml_tensor* input,
                                                int world_size,
                                                const std::string& name) {
    return sp_all_to_all_4d_head_to_seq(ctx, input, nullptr, world_size, name);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed(ggml_context* ctx,
                                                       ggml_tensor* input,
                                                       ProcessGroup* process_group,
                                                       int world_size,
                                                       const std::string& name) {
    check_context_tensor(ctx, input, "sp_all_to_all_4d_head_to_seq_packed");
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq_packed");
    if (input->type != GGML_TYPE_F32) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed supports F32 input only");
    }
    if (input->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed supports batch == 1 only");
    }
    if (input->ne[2] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_packed sequence must be divisible by world_size: sequence="
            << input->ne[2] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DLayout layout;
    layout.direction      = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size     = world_size;
    layout.batch          = input->ne[3];
    layout.head_dim       = input->ne[0];
    layout.shard_heads    = input->ne[1];
    layout.heads          = input->ne[1] * world_size;
    layout.sequence       = input->ne[2];
    layout.shard_sequence = input->ne[2] / world_size;

    const int64_t send_elems =
        layout.head_dim * layout.shard_heads * layout.shard_sequence * world_size;
    ggml_tensor* send_args[] = {input};
    layout.send_flat = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      send_elems,
                                      1,
                                      1,
                                      1,
                                      send_args,
                                      1,
                                      sp_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(layout.sequence,
                                                                    0,
                                                                    7,
                                                                    layout.sequence,
                                                                    0,
                                                                    world_size));
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat_fused").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(send_elems) / static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 send_elems,
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    ggml_tensor* recv_args[] = {layout.recv_flat};
    layout.output = ggml_custom_4d(ctx,
                                   GGML_TYPE_F32,
                                   layout.head_dim,
                                   layout.heads,
                                   layout.shard_sequence,
                                   layout.batch,
                                   recv_args,
                                   1,
                                   sp_attn_head_to_seq_recv_unpack_cpu,
                                   GGML_N_TASKS_MAX,
                                   sp_fused_qkv_pack_make_params(layout.sequence,
                                                                 0,
                                                                 8,
                                                                 layout.sequence,
                                                                 0,
                                                                 world_size,
                                                                 0));
    if (!name.empty()) {
        ggml_set_name(layout.output, (name + "_output_fused").c_str());
    }
    return layout;
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed(ggml_context* ctx,
                                                       ggml_tensor* input,
                                                       int world_size,
                                                       const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed(ctx, input, nullptr, world_size, name);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_f16_impl(ggml_context* ctx,
                                                                ggml_tensor* input,
                                                                ProcessGroup* process_group,
                                                                int world_size,
                                                                const std::string& name,
                                                                ggml_type output_type) {
    const bool keep_output_f16 = output_type == GGML_TYPE_F16;
    const bool keep_output_bf16 = output_type == GGML_TYPE_BF16;
    check_context_tensor(ctx, input, keep_output_bf16 ?
                                     "sp_all_to_all_4d_head_to_seq_packed_bf16_output" :
                                     keep_output_f16 ?
                                         "sp_all_to_all_4d_head_to_seq_packed_f16_output" :
                                         "sp_all_to_all_4d_head_to_seq_packed_f16");
    check_world_size(world_size, keep_output_bf16 ?
                                 "sp_all_to_all_4d_head_to_seq_packed_bf16_output" :
                                 keep_output_f16 ?
                                     "sp_all_to_all_4d_head_to_seq_packed_f16_output" :
                                     "sp_all_to_all_4d_head_to_seq_packed_f16");
    if (input->type != GGML_TYPE_F32 && input->type != GGML_TYPE_F16) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_f16 supports F32/F16 input only");
    }
    if (input->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_f16 supports batch == 1 only");
    }
    if (input->ne[2] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_packed_f16 sequence must be divisible by world_size: sequence="
            << input->ne[2] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DLayout layout;
    layout.direction      = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size     = world_size;
    layout.batch          = input->ne[3];
    layout.head_dim       = input->ne[0];
    layout.shard_heads    = input->ne[1];
    layout.heads          = input->ne[1] * world_size;
    layout.sequence       = input->ne[2];
    layout.shard_sequence = input->ne[2] / world_size;

    const int64_t send_elems =
        layout.head_dim * layout.shard_heads * layout.shard_sequence * world_size;
    ggml_tensor* send_args[] = {input};
    layout.send_flat = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      send_elems,
                                      1,
                                      1,
                                      1,
                                      send_args,
                                      1,
                                      sp_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(layout.sequence,
                                                                    0,
                                                                    13,
                                                                    layout.sequence,
                                                                    0,
                                                                    world_size));
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat_fused_f16").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(send_elems) / static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 send_elems,
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    ggml_tensor* recv_args[] = {layout.recv_flat};
    layout.output = ggml_custom_4d(ctx,
                                   output_type,
                                   layout.head_dim,
                                   layout.heads,
                                   layout.shard_sequence,
                                   layout.batch,
                                   recv_args,
                                   1,
                                   sp_attn_head_to_seq_recv_unpack_cpu,
                                   GGML_N_TASKS_MAX,
                                   sp_fused_qkv_pack_make_params(layout.sequence,
                                                                 0,
                                                                 keep_output_bf16 ? 48 :
                                                                 keep_output_f16 ? 45 : 14,
                                                                 layout.sequence,
                                                                 0,
                                                                 world_size,
                                                                 0));
    if (!name.empty()) {
        ggml_set_name(layout.output, (name + (keep_output_bf16 ? "_output_fused_keep_bf16" :
                                              keep_output_f16 ? "_output_fused_keep_f16" :
                                                                "_output_fused_f16")).c_str());
    }
    return layout;
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_f16(ggml_context* ctx,
                                                           ggml_tensor* input,
                                                           ProcessGroup* process_group,
                                                           int world_size,
                                                           const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_f16_impl(ctx,
                                                       input,
                                                       process_group,
                                                       world_size,
                                                       name,
                                                       GGML_TYPE_F32);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_f16(ggml_context* ctx,
                                                           ggml_tensor* input,
                                                           int world_size,
                                                           const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_f16(ctx, input, nullptr, world_size, name);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_f16_output(
    ggml_context* ctx,
    ggml_tensor* input,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_f16_impl(ctx,
                                                       input,
                                                       process_group,
                                                       world_size,
                                                       name,
                                                       GGML_TYPE_F16);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_f16_output(
    ggml_context* ctx,
    ggml_tensor* input,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_f16_output(ctx, input, nullptr, world_size, name);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_bf16_output(
    ggml_context* ctx,
    ggml_tensor* input,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_f16_impl(ctx,
                                                       input,
                                                       process_group,
                                                       world_size,
                                                       name,
                                                       GGML_TYPE_BF16);
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq_packed_bf16_output(
    ggml_context* ctx,
    ggml_tensor* input,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_bf16_output(ctx, input, nullptr, world_size, name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             ProcessGroup* process_group,
                                                             int world_size,
                                                             const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq_batched");
    if (inputs.empty()) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_batched inputs must not be empty");
    }

    ggml_tensor* first = inputs.front();
    check_context_tensor(ctx, first, "sp_all_to_all_4d_head_to_seq_batched");
    if (first->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_batched phase 1 supports batch == 1 only");
    }
    if (first->ne[2] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_batched sequence must be divisible by world_size: sequence="
            << first->ne[2] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size = world_size;
    layout.batch = first->ne[3];
    layout.head_dim = first->ne[0];
    layout.shard_heads = first->ne[1];
    layout.heads = first->ne[1] * world_size;
    layout.outputs.reserve(inputs.size());
    layout.sequences.reserve(inputs.size());
    layout.shard_sequences.reserve(inputs.size());

    std::vector<int64_t> chunk_nelements;
    chunk_nelements.reserve(inputs.size());
    std::vector<ggml_tensor*> send_chunks;
    send_chunks.reserve(inputs.size() * static_cast<size_t>(world_size));
    for (int dst = 0; dst < world_size; ++dst) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            ggml_tensor* input = inputs[i];
            check_context_tensor(ctx, input, "sp_all_to_all_4d_head_to_seq_batched");
            if (input->ne[0] != layout.head_dim ||
                input->ne[1] != layout.shard_heads ||
                input->ne[3] != layout.batch) {
                std::ostringstream oss;
                oss << "sp_all_to_all_4d_head_to_seq_batched input " << i
                    << " shape mismatch";
                throw std::invalid_argument(oss.str());
            }
            if (input->ne[2] % world_size != 0) {
                std::ostringstream oss;
                oss << "sp_all_to_all_4d_head_to_seq_batched input " << i
                    << " sequence must be divisible by world_size: sequence="
                    << input->ne[2] << " world_size=" << world_size;
                throw std::invalid_argument(oss.str());
            }

            const int64_t shard_sequence = input->ne[2] / world_size;
            if (dst == 0) {
                layout.sequences.push_back(input->ne[2]);
                layout.shard_sequences.push_back(shard_sequence);
                layout.sequence += input->ne[2];
                layout.shard_sequence += shard_sequence;
                chunk_nelements.push_back(input->ne[0] *
                                          input->ne[1] *
                                          shard_sequence *
                                          input->ne[3]);
            }

            ggml_tensor* chunk = ggml_view_4d(ctx,
                                              input,
                                              layout.head_dim,
                                              layout.shard_heads,
                                              shard_sequence,
                                              layout.batch,
                                              input->nb[1],
                                              input->nb[2],
                                              input->nb[3],
                                              static_cast<size_t>(dst) *
                                                  static_cast<size_t>(shard_sequence) *
                                                  input->nb[2]);
            chunk = flatten_for_comm_1d(ctx, chunk);
            if (!name.empty()) {
                ggml_set_name(chunk, (name + "_send_chunk_" + std::to_string(dst) + "_" + std::to_string(i)).c_str());
            }
            send_chunks.push_back(chunk);
        }
    }

    ggml_tensor* send_flat = send_chunks.front();
    for (size_t i = 1; i < send_chunks.size(); ++i) {
        send_flat = ggml_concat(ctx, send_flat, send_chunks[i], 0);
    }
    layout.send_flat = flatten_for_comm_1d(ctx, send_flat);
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                            static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 ggml_nelements(layout.send_flat),
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int64_t shard_sequence = layout.shard_sequences[i];
        const int64_t chunk_ne = chunk_nelements[i];
        size_t input_peer_offset = 0;
        for (size_t j = 0; j < i; ++j) {
            input_peer_offset += static_cast<size_t>(chunk_nelements[j]);
        }

        ggml_tensor* flat_view = nullptr;
        for (int src = 0; src < world_size; ++src) {
            const size_t offset = (static_cast<size_t>(src) * layout.count_per_peer +
                                   input_peer_offset) *
                                  layout.recv_flat->nb[0];
            ggml_tensor* src_view = ggml_view_1d(ctx,
                                                 layout.recv_flat,
                                                 chunk_ne,
                                                 offset);
            if (flat_view == nullptr) {
                flat_view = src_view;
            } else {
                flat_view = ggml_concat(ctx, flat_view, src_view, 0);
            }
        }

        ggml_tensor* src_peer_major = ggml_reshape_4d(ctx,
                                                      flat_view,
                                                      layout.head_dim,
                                                      layout.shard_heads,
                                                      shard_sequence,
                                                      world_size);
        ggml_tensor* heads_before_sequence = ggml_permute(ctx, src_peer_major, 0, 1, 3, 2);
        ggml_tensor* output = ggml_cont_4d(ctx,
                                           heads_before_sequence,
                                           layout.head_dim,
                                           layout.heads,
                                           shard_sequence,
                                           layout.batch);
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        layout.outputs.push_back(output);
    }

    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             int world_size,
                                                             const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_batched(ctx, inputs, nullptr, world_size, name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed(ggml_context* ctx,
                                                                       ggml_tensor* attn,
                                                                       int64_t first_sequence,
                                                                       int64_t second_sequence,
                                                                       ProcessGroup* process_group,
                                                                       int world_size,
                                                                       const std::string& name) {
    check_context_tensor(ctx, attn, "sp_all_to_all_4d_head_to_seq_two_stream_packed");
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq_two_stream_packed");
    if (attn->type != GGML_TYPE_F32) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_two_stream_packed supports F32 attn only");
    }
    if (attn->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_two_stream_packed supports batch == 1 only");
    }
    if (first_sequence <= 0 || second_sequence <= 0 ||
        first_sequence % world_size != 0 ||
        second_sequence % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_two_stream_packed sequences must be positive and divisible by world_size: first="
            << first_sequence << " second=" << second_sequence << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (attn->ne[2] != first_sequence + second_sequence) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_two_stream_packed attn sequence mismatch");
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction      = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size     = world_size;
    layout.batch          = attn->ne[3];
    layout.head_dim       = attn->ne[0];
    layout.shard_heads    = attn->ne[1];
    layout.heads          = attn->ne[1] * world_size;
    layout.sequence       = first_sequence + second_sequence;
    layout.shard_sequence = layout.sequence / world_size;
    layout.sequences      = {first_sequence, second_sequence};
    layout.shard_sequences = {first_sequence / world_size, second_sequence / world_size};

    const int64_t send_elems =
        layout.head_dim * layout.shard_heads *
        (layout.shard_sequences[0] + layout.shard_sequences[1]) *
        world_size;
    ggml_tensor* args[] = {attn};
    layout.send_flat = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      send_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      sp_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(first_sequence,
                                                                    second_sequence,
                                                                    7,
                                                                    first_sequence,
                                                                    second_sequence,
                                                                    world_size));
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat_fused").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(send_elems) / static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 send_elems,
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    layout.outputs.reserve(2);
    for (int64_t stream_index = 0; stream_index < 2; ++stream_index) {
        ggml_tensor* recv_args[] = {layout.recv_flat};
        ggml_tensor* output = ggml_custom_4d(ctx,
                                             GGML_TYPE_F32,
                                             layout.head_dim,
                                             layout.heads,
                                             layout.shard_sequences[stream_index],
                                             layout.batch,
                                             recv_args,
                                             1,
                                             sp_attn_head_to_seq_recv_unpack_cpu,
                                             GGML_N_TASKS_MAX,
                                             sp_fused_qkv_pack_make_params(first_sequence,
                                                                           second_sequence,
                                                                           8,
                                                                           first_sequence,
                                                                           second_sequence,
                                                                           world_size,
                                                                           stream_index));
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(stream_index) + "_fused").c_str());
        }
        layout.outputs.push_back(output);
    }

    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed(ggml_context* ctx,
                                                                       ggml_tensor* attn,
                                                                       int64_t first_sequence,
                                                                       int64_t second_sequence,
                                                                       int world_size,
                                                                       const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed(ctx,
                                                          attn,
                                                          first_sequence,
                                                          second_sequence,
                                                          nullptr,
                                                          world_size,
                                                          name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_impl(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name,
    ggml_type output_type) {
    const bool keep_output_f16 = output_type == GGML_TYPE_F16;
    const bool keep_output_bf16 = output_type == GGML_TYPE_BF16;
    check_context_tensor(ctx, attn, keep_output_bf16 ?
                                    "sp_all_to_all_4d_head_to_seq_two_stream_packed_bf16_output" :
                                    keep_output_f16 ?
                                        "sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_output" :
                                        "sp_all_to_all_4d_head_to_seq_two_stream_packed_f16");
    check_world_size(world_size, keep_output_bf16 ?
                                 "sp_all_to_all_4d_head_to_seq_two_stream_packed_bf16_output" :
                                 keep_output_f16 ?
                                     "sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_output" :
                                     "sp_all_to_all_4d_head_to_seq_two_stream_packed_f16");
    if (attn->type != GGML_TYPE_F32 && attn->type != GGML_TYPE_F16) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_two_stream_packed_f16 supports F32/F16 attn only");
    }
    if (attn->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_two_stream_packed_f16 supports batch == 1 only");
    }
    if (first_sequence <= 0 || second_sequence <= 0 ||
        first_sequence % world_size != 0 ||
        second_sequence % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_two_stream_packed_f16 sequences must be positive and divisible by world_size: first="
            << first_sequence << " second=" << second_sequence << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (attn->ne[2] != first_sequence + second_sequence) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_two_stream_packed_f16 attn sequence mismatch");
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction       = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size      = world_size;
    layout.batch           = attn->ne[3];
    layout.head_dim        = attn->ne[0];
    layout.shard_heads     = attn->ne[1];
    layout.heads           = attn->ne[1] * world_size;
    layout.sequence        = first_sequence + second_sequence;
    layout.shard_sequence  = layout.sequence / world_size;
    layout.sequences       = {first_sequence, second_sequence};
    layout.shard_sequences = {first_sequence / world_size, second_sequence / world_size};

    const int64_t send_elems =
        layout.head_dim * layout.shard_heads *
        (layout.shard_sequences[0] + layout.shard_sequences[1]) *
        world_size;
    ggml_tensor* args[] = {attn};
    layout.send_flat = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      send_elems,
                                      1,
                                      1,
                                      1,
                                      args,
                                      1,
                                      sp_attn_head_to_seq_send_pack_cpu,
                                      GGML_N_TASKS_MAX,
                                      sp_fused_qkv_pack_make_params(first_sequence,
                                                                    second_sequence,
                                                                    13,
                                                                    first_sequence,
                                                                    second_sequence,
                                                                    world_size));
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat_fused_f16").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(send_elems) / static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 send_elems,
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    layout.outputs.reserve(2);
    for (int64_t stream_index = 0; stream_index < 2; ++stream_index) {
        ggml_tensor* recv_args[] = {layout.recv_flat};
        ggml_tensor* output = ggml_custom_4d(ctx,
                                             output_type,
                                             layout.head_dim,
                                             layout.heads,
                                             layout.shard_sequences[stream_index],
                                             layout.batch,
                                             recv_args,
                                             1,
                                             sp_attn_head_to_seq_recv_unpack_cpu,
                                             GGML_N_TASKS_MAX,
                                             sp_fused_qkv_pack_make_params(first_sequence,
                                                                           second_sequence,
                                                                           keep_output_bf16 ? 48 :
                                                                           keep_output_f16 ? 45 : 14,
                                                                           first_sequence,
                                                                           second_sequence,
                                                                           world_size,
                                                                           stream_index));
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(stream_index) +
                                   (keep_output_bf16 ? "_fused_keep_bf16" :
                                    keep_output_f16 ? "_fused_keep_f16" :
                                                      "_fused_f16")).c_str());
        }
        layout.outputs.push_back(output);
    }

    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_f16(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_impl(ctx,
                                                                   attn,
                                                                   first_sequence,
                                                                   second_sequence,
                                                                   process_group,
                                                                   world_size,
                                                                   name,
                                                                   GGML_TYPE_F32);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_f16(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed_f16(ctx,
                                                              attn,
                                                              first_sequence,
                                                              second_sequence,
                                                              nullptr,
                                                              world_size,
                                                              name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_output(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_impl(ctx,
                                                                   attn,
                                                                   first_sequence,
                                                                   second_sequence,
                                                                   process_group,
                                                                   world_size,
                                                                   name,
                                                                   GGML_TYPE_F16);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_output(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_output(ctx,
                                                                     attn,
                                                                     first_sequence,
                                                                     second_sequence,
                                                                     nullptr,
                                                                     world_size,
                                                                     name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_bf16_output(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    ProcessGroup* process_group,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_impl(ctx,
                                                                   attn,
                                                                   first_sequence,
                                                                   second_sequence,
                                                                   process_group,
                                                                   world_size,
                                                                   name,
                                                                   GGML_TYPE_BF16);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_two_stream_packed_bf16_output(
    ggml_context* ctx,
    ggml_tensor* attn,
    int64_t first_sequence,
    int64_t second_sequence,
    int world_size,
    const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_two_stream_packed_bf16_output(ctx,
                                                                      attn,
                                                                      first_sequence,
                                                                      second_sequence,
                                                                      nullptr,
                                                                      world_size,
                                                                      name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_packed_recv_only(ggml_context* ctx,
                                                                      ggml_tensor* send_flat,
                                                                      int64_t head_dim,
                                                                      int64_t shard_heads,
                                                                      const std::vector<int64_t>& sequences,
                                                                      ProcessGroup* process_group,
                                                                      int world_size,
                                                                      const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq_packed_recv_only");
    check_context_tensor(ctx, send_flat, "sp_all_to_all_4d_head_to_seq_packed_recv_only");
    if (sequences.empty()) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_recv_only sequences must not be empty");
    }
    if (head_dim <= 0 || shard_heads <= 0) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_recv_only invalid head shape");
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size = world_size;
    layout.batch = 1;
    layout.head_dim = head_dim;
    layout.shard_heads = shard_heads;
    layout.heads = shard_heads * world_size;
    layout.outputs.reserve(sequences.size());
    layout.sequences.reserve(sequences.size());
    layout.shard_sequences.reserve(sequences.size());

    std::vector<int64_t> chunk_nelements;
    chunk_nelements.reserve(sequences.size());
    int64_t expected_ne = 0;
    for (size_t i = 0; i < sequences.size(); ++i) {
        const int64_t sequence = sequences[i];
        if (sequence <= 0 || sequence % world_size != 0) {
            std::ostringstream oss;
            oss << "sp_all_to_all_4d_head_to_seq_packed_recv_only sequence " << i
                << " must be positive and divisible by world_size: sequence="
                << sequence << " world_size=" << world_size;
            throw std::invalid_argument(oss.str());
        }
        const int64_t shard_sequence = sequence / world_size;
        const int64_t chunk_ne = head_dim * shard_heads * shard_sequence;
        layout.sequences.push_back(sequence);
        layout.shard_sequences.push_back(shard_sequence);
        layout.sequence += sequence;
        layout.shard_sequence += shard_sequence;
        chunk_nelements.push_back(chunk_ne);
        expected_ne += chunk_ne * world_size;
    }

    if (ggml_nelements(send_flat) != expected_ne) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_packed_recv_only send_flat size mismatch: got="
            << ggml_nelements(send_flat) << " expected=" << expected_ne;
        throw std::invalid_argument(oss.str());
    }

    layout.send_flat = send_flat;
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }
    layout.count_per_peer = static_cast<size_t>(expected_ne) / static_cast<size_t>(world_size);
    layout.recv_flat = make_all_to_all_recv_flat(ctx,
                                                 layout.send_flat,
                                                 expected_ne,
                                                 layout.count_per_peer,
                                                 process_group,
                                                 world_size,
                                                 name);

    for (size_t i = 0; i < sequences.size(); ++i) {
        const int64_t shard_sequence = layout.shard_sequences[i];
        const int64_t chunk_ne = chunk_nelements[i];
        size_t input_peer_offset = 0;
        for (size_t j = 0; j < i; ++j) {
            input_peer_offset += static_cast<size_t>(chunk_nelements[j]);
        }

        ggml_tensor* flat_view = nullptr;
        for (int src = 0; src < world_size; ++src) {
            const size_t offset = (static_cast<size_t>(src) * layout.count_per_peer +
                                   input_peer_offset) *
                                  layout.recv_flat->nb[0];
            ggml_tensor* src_view = ggml_view_1d(ctx,
                                                 layout.recv_flat,
                                                 chunk_ne,
                                                 offset);
            if (flat_view == nullptr) {
                flat_view = src_view;
            } else {
                flat_view = ggml_concat(ctx, flat_view, src_view, 0);
            }
        }

        ggml_tensor* src_peer_major = ggml_reshape_4d(ctx,
                                                      flat_view,
                                                      layout.head_dim,
                                                      layout.shard_heads,
                                                      shard_sequence,
                                                      world_size);
        ggml_tensor* heads_before_sequence = ggml_permute(ctx, src_peer_major, 0, 1, 3, 2);
        ggml_tensor* output = ggml_cont_4d(ctx,
                                           heads_before_sequence,
                                           layout.head_dim,
                                           layout.heads,
                                           shard_sequence,
                                           layout.batch);
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        layout.outputs.push_back(output);
    }

    return layout;
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_packed_recv_only(ggml_context* ctx,
                                                                      ggml_tensor* send_flat,
                                                                      int64_t head_dim,
                                                                      int64_t shard_heads,
                                                                      const std::vector<int64_t>& sequences,
                                                                      int world_size,
                                                                      const std::string& name) {
    return sp_all_to_all_4d_head_to_seq_packed_recv_only(ctx,
                                                         send_flat,
                                                         head_dim,
                                                         shard_heads,
                                                         sequences,
                                                         nullptr,
                                                         world_size,
                                                         name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_packed_recv_only_f16(ggml_context* ctx,
                                                                          ggml_tensor* send_flat,
                                                                          int64_t head_dim,
                                                                          int64_t shard_heads,
                                                                          const std::vector<int64_t>& sequences,
                                                                          int world_size,
                                                                          const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq_packed_recv_only_f16");
    check_context_tensor(ctx, send_flat, "sp_all_to_all_4d_head_to_seq_packed_recv_only_f16");
    if (send_flat->type != GGML_TYPE_F16) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_recv_only_f16 send_flat must be F16");
    }
    if (sequences.empty()) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_recv_only_f16 sequences must not be empty");
    }
    if (head_dim <= 0 || shard_heads <= 0) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_packed_recv_only_f16 invalid head shape");
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size = world_size;
    layout.batch = 1;
    layout.head_dim = head_dim;
    layout.shard_heads = shard_heads;
    layout.heads = shard_heads * world_size;
    layout.outputs.reserve(sequences.size());
    layout.sequences.reserve(sequences.size());
    layout.shard_sequences.reserve(sequences.size());

    int64_t expected_ne = 0;
    for (size_t i = 0; i < sequences.size(); ++i) {
        const int64_t sequence = sequences[i];
        if (sequence <= 0 || sequence % world_size != 0) {
            std::ostringstream oss;
            oss << "sp_all_to_all_4d_head_to_seq_packed_recv_only_f16 sequence " << i
                << " must be positive and divisible by world_size: sequence="
                << sequence << " world_size=" << world_size;
            throw std::invalid_argument(oss.str());
        }
        const int64_t shard_sequence = sequence / world_size;
        layout.sequences.push_back(sequence);
        layout.shard_sequences.push_back(shard_sequence);
        layout.sequence += sequence;
        layout.shard_sequence += shard_sequence;
        expected_ne += head_dim * shard_heads * shard_sequence * world_size;
    }

    if (ggml_nelements(send_flat) != expected_ne) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_packed_recv_only_f16 send_flat size mismatch: got="
            << ggml_nelements(send_flat) << " expected=" << expected_ne;
        throw std::invalid_argument(oss.str());
    }

    layout.send_flat = send_flat;
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }
    layout.count_per_peer = static_cast<size_t>(expected_ne) / static_cast<size_t>(world_size);
    layout.recv_flat = new_recv_placeholder(ctx,
                                            layout.send_flat,
                                            layout.send_flat,
                                            expected_ne,
                                            1,
                                            1,
                                            1,
                                            name + "_flat",
                                            world_size);
    mark_all_to_all_flat(layout.send_flat,
                         layout.recv_flat,
                         layout.count_per_peer,
                         name);
    sd::ggml_graph_cut::mark_graph_cut(layout.recv_flat,
                                       graph_cut_group_name(name),
                                       name + "_recv_flat");
    return layout;
}

} // namespace edgedit::parallel
