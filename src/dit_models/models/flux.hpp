#ifndef __FLUX_HPP__
#define __FLUX_HPP__

#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/ggml/ed_ggml_rope_ext.hpp"
#include "backend/ggml/ed_ggml_modulation_ext.hpp"
#include "backend/ggml/ed_ggml_sp_flux_ext.hpp"
#include "dit_models/components/common/common_dit.hpp"
#include "dit_models/components/common/modulation.hpp"
#include "dit_models/components/common/normalization.hpp"
#include "dit_models/components/common/rope.hpp"
#include "edge-dit.h"
#include "core/runtime/model_loader.h"
#include "core/optimization/cache/ir/graph_extension.hpp"
#include "parallel/sp_parallel.hpp"

#define FLUX_GRAPH_SIZE 10240

namespace Flux {

    using RMSNorm = dit::RMSNorm;
    using QKNorm  = dit::QKNorm;

    static inline bool flux_sp_enabled(GGMLRunnerContext* ctx) {
        return ctx != nullptr &&
               ctx->process_group != nullptr &&
               ctx->process_group->enabled() &&
               ctx->process_group->size() > 1;
    }

    static inline int flux_sp_rank(GGMLRunnerContext* ctx) {
        return ctx->process_group->rank();
    }

    static inline int flux_sp_world_size(GGMLRunnerContext* ctx) {
        return ctx->process_group->size();
    }

    static inline bool flux_sp_strict_barrier_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("ED_FLUX_SP_STRICT_BARRIER");
            return env != nullptr && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
        }();
        return enabled;
    }

    static inline bool flux_env_flag_enabled(const char* name) {
        const char* env = std::getenv(name);
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 &&
               std::strcmp(env, "OFF") != 0;
    }

    static inline bool flux_env_flag_enabled_or_default(const char* name, bool default_enabled) {
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

    static inline bool flux_profile_enabled() {
        return flux_env_flag_enabled("ED_PROFILE_GRAPH_CUTS") ||
               flux_env_flag_enabled("ED_PROFILE_FLUX");
    }

    static inline bool flux_profile_should_log_rank(GGMLRunnerContext* ctx) {
        if (flux_env_flag_enabled("ED_PROFILE_GRAPH_CUTS_ALL_RANKS")) {
            return true;
        }
        return ctx == nullptr ||
               ctx->process_group == nullptr ||
               !ctx->process_group->enabled() ||
               ctx->process_group->rank() == 0;
    }

    static inline bool flux_sp_qk_seq_major_enabled() {
        return true;
    }

    static inline bool flux_sp_shared_pe_seq_major_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_SHARED_PE_SEQ_MAJOR");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_SHARED_PE_SEQ_MAJOR");
        }
        return false;
    }

    static inline bool flux_sp_skip_block_cuts_with_custom_comm_enabled(GGMLRunnerContext* ctx) {
        return flux_sp_enabled(ctx) &&
               !flux_sp_strict_barrier_enabled() &&
               edgedit::ggml_ext::flux_sp_all_to_all_custom_enabled() &&
               flux_env_flag_enabled_or_default("ED_FLUX_SP_SKIP_BLOCK_CUTS_WITH_CUSTOM_COMM", true);
    }

    static inline bool flux_sp_double_to_single_reshard_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_DOUBLE_TO_SINGLE_RESHARD");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_DOUBLE_TO_SINGLE_RESHARD");
        }
        return true;
    }

    static inline bool flux_sp_flash_attn_enabled(GGMLRunnerContext* ctx) {
        if (ctx == nullptr) {
            return false;
        }
        const char* env = std::getenv("ED_FLUX_SP_FLASH_ATTN");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FLASH_ATTN");
        }
        return ctx->flash_attn_enabled;
    }

    static inline bool flux_sp_flash_v_seq_major_enabled(GGMLRunnerContext* ctx) {
        return flux_sp_qk_seq_major_enabled() &&
               flux_sp_flash_attn_enabled(ctx);
    }

    static inline bool flux_sp_split_single_linear1_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_SPLIT_SINGLE_LINEAR1");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_SPLIT_SINGLE_LINEAR1");
        }
        return true;
    }

    static inline bool flux_sp_pre_qk_norm_enabled() {
        return true;
    }

    static inline bool flux_sp_fused_qkv_send_pack_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_QKV_SEND_PACK");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_QKV_SEND_PACK");
        }
        return true;
    }

    static inline bool flux_sp_mixed_qkv_send_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_MIXED_QKV_SEND");
    }

    static inline bool flux_sp_f16_qkv_send_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_F16_QKV_SEND", true);
    }

    static inline bool flux_sp_f16_qk_norm_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_F16_QK_NORM", true);
    }

    static inline bool flux_sp_mixed_double_qkv_send_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_MIXED_DOUBLE_QKV_SEND");
    }

    static inline bool flux_sp_split_single_linear2_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_SPLIT_SINGLE_LINEAR2");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_SPLIT_SINGLE_LINEAR2");
        }
        return false;
    }

    static inline bool flux_sp_fused_single_linear2_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_SINGLE_LINEAR2");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_SINGLE_LINEAR2");
        }
        return true;
    }

    static inline bool flux_sp_fused_single_linear2_residual_gate_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_SINGLE_LINEAR2_RESIDUAL_GATE");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_SINGLE_LINEAR2_RESIDUAL_GATE");
        }
        return true;
    }

    static inline bool flux_sp_mlp_gelu_bf16_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_MLP_GELU_BF16");
    }

    static inline bool flux_sp_fused_head_to_seq_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_HEAD_TO_SEQ");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_HEAD_TO_SEQ");
        }
        return true;
    }

    static inline bool flux_sp_f16_head_to_seq_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_F16_HEAD_TO_SEQ");
    }

    static inline bool flux_sp_f16_double_head_to_seq_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_F16_DOUBLE_HEAD_TO_SEQ");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_F16_DOUBLE_HEAD_TO_SEQ");
        }
        return flux_sp_f16_head_to_seq_enabled();
    }

    static inline bool flux_sp_f16_single_head_to_seq_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_F16_SINGLE_HEAD_TO_SEQ");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_F16_SINGLE_HEAD_TO_SEQ");
        }
        return flux_sp_f16_head_to_seq_enabled();
    }

    static inline bool flux_sp_f16_single_head_to_seq_output_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_F16_SINGLE_HEAD_TO_SEQ_OUTPUT");
    }

    static inline bool flux_sp_bf16_single_head_to_seq_output_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_BF16_SINGLE_HEAD_TO_SEQ_OUTPUT", true);
    }

    static inline bool flux_sp_f16_double_head_to_seq_output_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_F16_DOUBLE_HEAD_TO_SEQ_OUTPUT");
    }

    static inline bool flux_sp_bf16_double_head_to_seq_output_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_BF16_DOUBLE_HEAD_TO_SEQ_OUTPUT", true);
    }

    static inline bool flux_sp_f16_q_attention_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_F16_Q_ATTENTION");
    }

    static inline bool flux_sp_f16_attention_output_enabled() {
        return flux_env_flag_enabled("ED_FLUX_SP_F16_ATTN_OUTPUT");
    }

    static inline bool flux_sp_fused_single_head_to_seq_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_SINGLE_HEAD_TO_SEQ");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_SINGLE_HEAD_TO_SEQ");
        }
        return false;
    }

    static inline bool flux_sp_fused_single_qkv_recv_prep_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_SINGLE_QKV_RECV_PREP");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_SINGLE_QKV_RECV_PREP");
        }
        return true;
    }

    static inline bool flux_sp_bundle_single_qkv_recv_prep_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_BUNDLE_SINGLE_QKV_RECV_PREP", true);
    }

    static inline bool flux_sp_bundle_double_qkv_recv_prep_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_BUNDLE_DOUBLE_QKV_RECV_PREP", true);
    }

    static inline bool flux_sp_fused_modulation_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_FUSED_MODULATION", true);
    }

    static inline ggml_tensor* flux_sp_modulate(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* shift,
                                                ggml_tensor* scale) {
#ifdef ED_ENABLE_CUDA_MODULATION
        if (flux_sp_fused_modulation_enabled() &&
            ctx != nullptr &&
            x != nullptr &&
            shift != nullptr &&
            scale != nullptr) {
            ggml_tensor* shift_in = shift;
            ggml_tensor* scale_in = scale;
            if (shift->ne[2] == 1 && shift->ne[3] == 1) {
                shift_in = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);
            }
            if (scale->ne[2] == 1 && scale->ne[3] == 1) {
                scale_in = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
            }
            if (auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx, x, shift_in, scale_in)) {
                return fused;
            }
        }
#endif
        return dit::modulate(ctx, x, shift, scale);
    }

    static inline ggml_tensor* flux_sp_residual_gate(ggml_context* ctx,
                                                     ggml_tensor* residual,
                                                     ggml_tensor* x,
                                                     ggml_tensor* gate) {
#ifdef ED_ENABLE_CUDA_MODULATION
        if (flux_sp_fused_modulation_enabled() &&
            ctx != nullptr &&
            residual != nullptr &&
            x != nullptr &&
            gate != nullptr) {
            ggml_tensor* gate_in = gate;
            if (gate->ne[2] == 1 && gate->ne[3] == 1) {
                gate_in = ggml_reshape_3d(ctx, gate, gate->ne[0], 1, gate->ne[1]);
            }
            if (auto fused = edgedit::ggml_ext::fused_residual_gate_custom(ctx, residual, x, gate_in)) {
                return fused;
            }
        }
#endif
        return ggml_add(ctx, residual, ggml_mul(ctx, x, gate));
    }

    static inline bool flux_sp_fused_double_qkv_recv_prep_enabled() {
        const char* env = std::getenv("ED_FLUX_SP_FUSED_DOUBLE_QKV_RECV_PREP");
        if (env != nullptr && env[0] != '\0') {
            return flux_env_flag_enabled("ED_FLUX_SP_FUSED_DOUBLE_QKV_RECV_PREP");
        }
        return true;
    }

    static inline bool flux_sp_fuse_double_qkv_a2a_enabled() {
        return flux_env_flag_enabled_or_default("ED_FLUX_SP_FUSE_DOUBLE_QKV_A2A", true);
    }

#ifdef ED_DEBUG_SP_COMM
    static inline std::vector<std::string>& debug_sp_output_names() {
        static thread_local std::vector<std::string> names;
        return names;
    }

    static inline void clear_debug_sp_output_names() {
        debug_sp_output_names().clear();
    }

    static inline ggml_tensor*& debug_sp_total_error() {
        static thread_local ggml_tensor* total = nullptr;
        return total;
    }

    static inline std::vector<ggml_tensor*>& debug_sp_error_tensors() {
        static thread_local std::vector<ggml_tensor*> tensors;
        return tensors;
    }

    static inline std::vector<std::string>& debug_sp_error_names() {
        static thread_local std::vector<std::string> names;
        return names;
    }

    static inline void clear_debug_sp_total_error() {
        debug_sp_total_error() = nullptr;
        debug_sp_error_tensors().clear();
        debug_sp_error_names().clear();
    }

    static inline std::string& debug_sp_mainline_compare_stage() {
        static thread_local std::string stage;
        return stage;
    }

    static inline bool debug_sp_mainline_compare_enabled() {
        return !debug_sp_mainline_compare_stage().empty();
    }

    static inline bool& debug_sp_capture_enabled() {
        static thread_local bool enabled = false;
        return enabled;
    }

    class DebugSPCaptureScope {
    public:
        DebugSPCaptureScope()
            : previous_(debug_sp_capture_enabled()) {
            clear_debug_sp_output_names();
            clear_debug_sp_total_error();
            debug_sp_capture_enabled() = true;
        }

        ~DebugSPCaptureScope() {
            debug_sp_capture_enabled() = previous_;
        }

        DebugSPCaptureScope(const DebugSPCaptureScope&)            = delete;
        DebugSPCaptureScope& operator=(const DebugSPCaptureScope&) = delete;

    private:
        bool previous_;
    };

    class DebugSPMainlineCompareScope {
    public:
        explicit DebugSPMainlineCompareScope(const std::string& stage)
            : previous_stage_(debug_sp_mainline_compare_stage()) {
            clear_debug_sp_output_names();
            clear_debug_sp_total_error();
            debug_sp_mainline_compare_stage() = stage;
        }

        ~DebugSPMainlineCompareScope() {
            debug_sp_mainline_compare_stage() = previous_stage_;
        }

        DebugSPMainlineCompareScope(const DebugSPMainlineCompareScope&)            = delete;
        DebugSPMainlineCompareScope& operator=(const DebugSPMainlineCompareScope&) = delete;

    private:
        std::string previous_stage_;
    };

    static inline void register_debug_sp_output(ggml_tensor* tensor) {
        if (tensor != nullptr && tensor->name[0] != '\0') {
            debug_sp_output_names().push_back(tensor->name);
        }
    }

    static inline void expand_debug_sp_outputs(ggml_context* compute_ctx, ggml_cgraph* gf) {
        auto& names = debug_sp_output_names();
        for (const auto& name : names) {
            if (name.empty()) {
                continue;
            }
            ggml_tensor* tensor = ggml_get_tensor(compute_ctx, name.c_str());
            if (tensor == nullptr) {
                LOG_WARN("flux debug SP output not found in graph context: %s", name.c_str());
                continue;
            }
            ggml_build_forward_expand(gf, tensor);
        }
        names.clear();
    }

    static inline ggml_cgraph* build_debug_sp_graph(ggml_context* compute_ctx, size_t graph_size) {
        ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, graph_size, false);
        if (debug_sp_total_error() != nullptr) {
            ggml_tensor* output = ggml_reshape_1d(compute_ctx, debug_sp_total_error(), 1);
            ggml_set_name(output, "flux_debug_sp_total_sse");
            auto& tensors = debug_sp_error_tensors();
            for (size_t i = 0; i < tensors.size(); ++i) {
                if (tensors[i] == nullptr) {
                    continue;
                }
                ggml_tensor* err = ggml_reshape_1d(compute_ctx, tensors[i], 1);
                ggml_set_name(err, ("flux_debug_sp_component_" + std::to_string(i)).c_str());
                output = ggml_concat(compute_ctx, output, err, 0);
                ggml_set_name(output, ("flux_debug_sp_sse_vector_" + std::to_string(i)).c_str());
            }
            ggml_build_forward_expand(gf, output);
            debug_sp_total_error() = nullptr;
            debug_sp_error_tensors().clear();
        } else {
            expand_debug_sp_outputs(compute_ctx, gf);
        }
        return gf;
    }

    static inline void accumulate_debug_sp_error(ggml_context* ctx,
                                                 ggml_tensor* err,
                                                 const std::string& name) {
        if (err == nullptr) {
            return;
        }
        debug_sp_error_tensors().push_back(err);
        debug_sp_error_names().push_back(name);
        if (debug_sp_total_error() == nullptr) {
            debug_sp_total_error() = err;
        } else {
            debug_sp_total_error() = ggml_add(ctx, debug_sp_total_error(), err);
        }
        ggml_set_name(debug_sp_total_error(), (name + "_total").c_str());
    }

    static inline ggml_tensor* flux_debug_sequence_shard_reference(GGMLRunnerContext* ctx,
                                                                   ggml_tensor* full,
                                                                   const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(full != nullptr);
        auto split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                          full,
                                                          flux_sp_rank(ctx),
                                                          flux_sp_world_size(ctx),
                                                          1,
                                                          name + "_seq_ref_split");
        return split.local;
    }

    static inline void log_flux_debug_reshape_4d(ggml_tensor* tensor,
                                                 int64_t ne0,
                                                 int64_t ne1,
                                                 int64_t ne2,
                                                 int64_t ne3,
                                                 const std::string& name) {
        if (tensor == nullptr) {
            LOG_INFO("flux debug reshape_4d %s input=null target=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
                     name.c_str(),
                     ne0,
                     ne1,
                     ne2,
                     ne3);
            return;
        }
        const int64_t actual = ggml_nelements(tensor);
        const int64_t target = ne0 * ne1 * ne2 * ne3;
        LOG_INFO("flux debug reshape_4d %s input=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] actual=%" PRId64 " target=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] expected=%" PRId64,
                 name.c_str(),
                 tensor->ne[0],
                 tensor->ne[1],
                 tensor->ne[2],
                 tensor->ne[3],
                 actual,
                 ne0,
                 ne1,
                 ne2,
                 ne3,
                 target);
    }

    static inline ggml_tensor* flux_debug_head_shard_reference(ggml_context* ctx,
                                                               ggml_tensor* full,
                                                               int rank,
                                                               int world_size,
                                                               const std::string& name) {
        const int64_t shard_heads = full->ne[1] / world_size;
        const size_t offset       = static_cast<size_t>(rank) *
                              static_cast<size_t>(shard_heads) *
                              full->nb[1];

        ggml_tensor* ref_view = ggml_view_4d(ctx,
                                             full,
                                             full->ne[0],
                                             shard_heads,
                                             full->ne[2],
                                             full->ne[3],
                                             full->nb[1],
                                             full->nb[2],
                                             full->nb[3],
                                             offset);
        ggml_set_name(ref_view, (name + "_ref_view").c_str());

        ggml_tensor* ref = ggml_cont(ctx, ref_view);
        ggml_set_name(ref, (name + "_ref").c_str());
        return ref;
    }

    static inline ggml_tensor* flux_debug_sse(ggml_context* ctx,
                                              ggml_tensor* a,
                                              ggml_tensor* b,
                                              const std::string& name) {
        ggml_tensor* diff = ggml_sub(ctx, a, b);
        ggml_set_name(diff, (name + "_diff").c_str());

        ggml_tensor* sq = ggml_mul(ctx, diff, diff);
        ggml_set_name(sq, (name + "_sq").c_str());

        ggml_tensor* err = ggml_sum(ctx, sq);
        ggml_set_name(err, (name + "_sse").c_str());
        return err;
    }

    static inline void mark_flux_debug_compare_tensor(ggml_context* ctx,
                                                      ggml_tensor* a,
                                                      ggml_tensor* b,
                                                      const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(a != nullptr);
        GGML_ASSERT(b != nullptr);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            GGML_ASSERT(a->ne[i] == b->ne[i]);
        }
        ggml_tensor* err = flux_debug_sse(ctx, a, b, name);
        accumulate_debug_sp_error(ctx, err, name);
    }
#endif

    static inline ggml_tensor* flux_sp_materialize_cut(GGMLRunnerContext* ctx,
                                                       ggml_tensor* x,
                                                       const std::string& group,
                                                       const std::string& name) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);

        const bool strict_barrier = flux_sp_strict_barrier_enabled();
        if (strict_barrier || !ggml_is_contiguous(x)) {
            x = ggml_cont(ctx->ggml_ctx, x);
            ggml_set_name(x, (group + "_" + name + "_cont").c_str());
        }

        if (strict_barrier) {
            sd::ggml_graph_cut::mark_graph_cut(x, group, name);
        }
        return x;
    }

    static inline ggml_tensor* flux_sp_prepare_rope_pe_seq_major(ggml_context* ctx,
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

    static inline ggml_tensor* flux_sp_apply_rope_seq_major(ggml_context* ctx,
                                                            ggml_tensor* x,
                                                            ggml_tensor* pe,
                                                            bool rope_interleaved = true,
                                                            ggml_tensor* prepared_pe = nullptr,
                                                            ggml_type out_type = GGML_TYPE_F32) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);
        GGML_ASSERT(out_type == GGML_TYPE_F32 || out_type == GGML_TYPE_F16);

        // x is already [head_dim, full_seq, shard_heads, batch], which is the
        // layout produced by Rope::apply_rope after its first permute+cont.
        int64_t d_head = x->ne[0];
        int64_t L      = x->ne[1];
        int64_t n_head = x->ne[2];
        int64_t N      = x->ne[3];
        ggml_tensor* fused = out_type == GGML_TYPE_F16 ?
                                 edgedit::ggml_ext::apply_rope_seq_major_f16(ctx, x, prepared_pe != nullptr ? prepared_pe : pe, rope_interleaved) :
                                 edgedit::ggml_ext::apply_rope_seq_major(ctx, x, prepared_pe != nullptr ? prepared_pe : pe, rope_interleaved);
        if (fused != nullptr) {
            return fused;
        }
        if (rope_interleaved) {
            x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));
        } else {
            x = ggml_reshape_4d(ctx, x, d_head / 2, 2, L, n_head * N);
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 3, 1));
        }

        int64_t offset = x->nb[2] * x->ne[2];
        auto x_0       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);
        auto x_1       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);
        x_0            = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);
        x_1            = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);
        auto temp_x    = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
        x_0            = ggml_repeat(ctx, x_0, temp_x);
        x_1            = ggml_repeat(ctx, x_1, temp_x);

        pe        = prepared_pe != nullptr ? prepared_pe :
                    flux_sp_prepare_rope_pe_seq_major(ctx, pe);
        offset    = pe->nb[2] * pe->ne[2];
        auto pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);
        auto pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
        if (!rope_interleaved) {
            x_out = ggml_cont(ctx, ggml_permute(ctx, x_out, 1, 0, 2, 3));
        }
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);
    }

    static inline ggml_tensor* flux_sp_apply_rope_seq_major_work_layout(ggml_context* ctx,
                                                                        ggml_tensor* x,
                                                                        ggml_tensor* pe,
                                                                        int64_t d_head,
                                                                        ggml_tensor* prepared_pe = nullptr,
                                                                        ggml_type out_type = GGML_TYPE_F32) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(pe != nullptr);
        GGML_ASSERT(x->ne[0] * 2 == d_head);
        GGML_ASSERT(x->ne[3] == 2);
        GGML_ASSERT(out_type == GGML_TYPE_F32 || out_type == GGML_TYPE_F16);

        ggml_tensor* fused = out_type == GGML_TYPE_F16 ?
                                 edgedit::ggml_ext::apply_rope_work_layout_f16(ctx, x, prepared_pe != nullptr ? prepared_pe : pe, d_head) :
                                 edgedit::ggml_ext::apply_rope_work_layout(ctx, x, prepared_pe != nullptr ? prepared_pe : pe, d_head);
        if (fused != nullptr) {
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
                    flux_sp_prepare_rope_pe_seq_major(ctx, pe);
        auto pe_offset = pe->nb[2] * pe->ne[2];
        auto pe_0      = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], pe_offset * 0);
        auto pe_1      = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], pe_offset * 1);

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head);
    }

    static inline ggml_tensor* flux_sp_flash_attention_seq_major(GGMLRunnerContext* ctx,
                                                                 ggml_tensor* q,
                                                                 ggml_tensor* k,
                                                                 ggml_tensor* v,
                                                                 int64_t n_head,
                                                                 ggml_tensor* mask);

    static inline ggml_tensor* flux_sp_attention_prepared_qk(GGMLRunnerContext* ctx,
                                                             ggml_tensor* q,
                                                             ggml_tensor* k,
                                                             ggml_tensor* v,
                                                             ggml_tensor* mask,
                                                             const std::string& name_prefix) {
        q = flux_sp_materialize_cut(ctx,
                                    q,
                                    name_prefix + ".sp_attn_q",
                                    "q");

        k = flux_sp_materialize_cut(ctx,
                                    k,
                                    name_prefix + ".sp_attn_k",
                                    "k");

        v = flux_sp_materialize_cut(ctx,
                                    v,
                                    name_prefix + ".sp_attn_v",
                                    "v");

        const bool v_seq_major = flux_sp_flash_v_seq_major_enabled(ctx);
        const int64_t n_head   = v_seq_major ? v->ne[2] : v->ne[1];

        GGML_ASSERT(q->ne[0] == k->ne[0]);
        GGML_ASSERT(q->ne[1] == k->ne[1]);
        GGML_ASSERT(q->ne[2] == k->ne[2]);
        GGML_ASSERT(q->ne[3] == k->ne[3]);

        GGML_ASSERT(v->ne[0] == q->ne[0]);
        if (v_seq_major) {
            GGML_ASSERT(v->ne[1] == q->ne[1]);
            GGML_ASSERT(v->ne[2] == q->ne[2]);
        } else {
            GGML_ASSERT(v->ne[1] == q->ne[2]);
            GGML_ASSERT(v->ne[2] == q->ne[1]);
        }
        GGML_ASSERT(v->ne[3] == q->ne[3]);

        if (v_seq_major) {
            ggml_tensor* attn = flux_sp_flash_attention_seq_major(ctx, q, k, v, n_head, mask);
            if (attn != nullptr) {
                ggml_set_name(attn, (name_prefix + "_attn").c_str());
                return attn;
            }
        }

        ggml_tensor* attn = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                   ctx->backend,
                                                   q,
                                                   k,
                                                   v,
                                                   n_head,
                                                   mask,
                                                   true,
                                                   flux_sp_flash_attn_enabled(ctx));
        ggml_set_name(attn, (name_prefix + "_attn").c_str());
        return attn;
    }

    static inline ggml_tensor* flux_sp_flash_attention_seq_major(GGMLRunnerContext* ctx,
                                                                 ggml_tensor* q,
                                                                 ggml_tensor* k,
                                                                 ggml_tensor* v,
                                                                 int64_t n_head,
                                                                 ggml_tensor* mask) {
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

        int kv_pad = 0;
        if (L_k % 256 != 0) {
            kv_pad = GGML_PAD(L_k, 256) - static_cast<int>(L_k);
        }
        if (mask != nullptr && mask->ne[3] != 1) {
            return nullptr;
        }

        float scale = 1.0f / sqrtf(static_cast<float>(d_head));

        ggml_tensor* k_in = k;
        if (kv_pad != 0) {
            k_in = ggml_pad(ctx->ggml_ctx, k_in, 0, kv_pad, 0, 0);
        }
        if (k_in->type != GGML_TYPE_F16) {
            k_in = ggml_cast(ctx->ggml_ctx, k_in, GGML_TYPE_F16);
        }

        ggml_tensor* v_in = ggml_reshape_3d(ctx->ggml_ctx, v, d_head, L_k, n_kv_head * N);
        if (kv_pad != 0) {
            v_in = ggml_pad(ctx->ggml_ctx, v_in, 0, kv_pad, 0, 0);
        }
        if (v_in->type != GGML_TYPE_F16 || !ggml_is_contiguous(v_in)) {
            v_in = ggml_cast(ctx->ggml_ctx, v_in, GGML_TYPE_F16);
        }

        ggml_tensor* mask_in = mask;
        if (mask_in != nullptr) {
            mask_in = ggml_transpose(ctx->ggml_ctx, mask_in);
#ifdef GGML_KQ_MASK_PAD
            int mask_pad = 0;
            if (mask_in->ne[1] % GGML_KQ_MASK_PAD != 0) {
                mask_pad = GGML_PAD(L_q, GGML_KQ_MASK_PAD) - mask_in->ne[1];
            }
            if (mask_pad > 0) {
                mask_in = ggml_pad(ctx->ggml_ctx, mask_in, 0, mask_pad, 0, 0);
            }
#endif
            mask_in = ggml_cast(ctx->ggml_ctx, mask_in, GGML_TYPE_F16);
        } else if (kv_pad > 0) {
            mask_in         = ggml_ext_zeros(ctx->ggml_ctx, L_k, L_q, 1, 1);
            auto pad_tensor = ggml_ext_full(ctx->ggml_ctx, -INFINITY, kv_pad, L_q, 1, 1);
            mask_in         = ggml_concat(ctx->ggml_ctx, mask_in, pad_tensor, 0);
            mask_in         = ggml_cast(ctx->ggml_ctx, mask_in, GGML_TYPE_F16);
        }

        ggml_tensor* q_in = q;
        if (flux_sp_f16_q_attention_enabled() && q_in->type != GGML_TYPE_F16) {
            q_in = ggml_cast(ctx->ggml_ctx, q_in, GGML_TYPE_F16);
        }

        const ggml_type attn_out_type = flux_sp_f16_attention_output_enabled() ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor* out = ggml_flash_attn_ext_with_type(ctx->ggml_ctx, q_in, k_in, v_in, mask_in, scale, 0, 0, attn_out_type);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (!ggml_backend_supports_op(ctx->backend, out)) {
            return nullptr;
        }

        out = ggml_view_3d(ctx->ggml_ctx, out, d_head, n_head, L_q, out->nb[1], out->nb[2], 0);
        out = ggml_ext_cont(ctx->ggml_ctx, out);
        out = ggml_reshape_3d(ctx->ggml_ctx, out, d_head * n_head, L_q, N);
        return out;
    }

    static inline ggml_tensor* flux_sp_attention(GGMLRunnerContext* ctx,
                                                 ggml_tensor* q,
                                                 ggml_tensor* k,
                                                 ggml_tensor* v,
                                                 ggml_tensor* pe,
                                                 ggml_tensor* mask,
                                                 const std::string& name_prefix,
                                                 ggml_tensor* prepared_pe_seq_major = nullptr) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(q != nullptr);
        GGML_ASSERT(k != nullptr);
        GGML_ASSERT(v != nullptr);
        GGML_ASSERT(pe != nullptr);

        const bool qk_seq_major = flux_sp_qk_seq_major_enabled();
        ggml_tensor* prepared_pe = qk_seq_major ?
                                       (prepared_pe_seq_major != nullptr ?
                                            prepared_pe_seq_major :
                                            flux_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                              pe,
                                                                              name_prefix + "_pe_seq_major")) :
                                       nullptr;
        q = qk_seq_major ? flux_sp_apply_rope_seq_major(ctx->ggml_ctx, q, pe, true, prepared_pe) :
                           Rope::apply_rope(ctx->ggml_ctx, q, pe, true, ctx->backend);
        if (qk_seq_major) {
            k = flux_sp_apply_rope_seq_major(ctx->ggml_ctx,
                                             k,
                                             pe,
                                             true,
                                             prepared_pe,
                                             flux_sp_flash_attn_enabled(ctx) ? GGML_TYPE_F16 : GGML_TYPE_F32);
        } else if (flux_sp_flash_attn_enabled(ctx)) {
            ggml_tensor* k_f16 = edgedit::ggml_ext::apply_rope_f16(ctx->ggml_ctx, k, pe, true);
            if (k_f16 != nullptr && ggml_backend_supports_op(ctx->backend, k_f16)) {
                k = k_f16;
            } else {
                k = Rope::apply_rope(ctx->ggml_ctx, k, pe, true, ctx->backend);
            }
        } else {
            k = Rope::apply_rope(ctx->ggml_ctx, k, pe, true, ctx->backend);
        }

        GGML_ASSERT(pe->ne[3] == q->ne[1]);
        return flux_sp_attention_prepared_qk(ctx, q, k, v, mask, name_prefix);
    }

    static inline ggml_tensor* flux_sp_attention_from_rope_work_layout(GGMLRunnerContext* ctx,
                                                                       ggml_tensor* q,
                                                                       ggml_tensor* k,
                                                                       ggml_tensor* v,
                                                                       ggml_tensor* pe,
                                                                       ggml_tensor* mask,
                                                                       int64_t d_head,
                                                                       const std::string& name_prefix,
                                                                       ggml_tensor* prepared_pe_seq_major = nullptr) {
        ggml_tensor* prepared_pe = prepared_pe_seq_major != nullptr ?
                                       prepared_pe_seq_major :
                                       flux_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                         pe,
                                                                         name_prefix + "_pe_seq_major");
        q = flux_sp_apply_rope_seq_major_work_layout(ctx->ggml_ctx, q, pe, d_head, prepared_pe);
        k = flux_sp_apply_rope_seq_major_work_layout(ctx->ggml_ctx,
                                                     k,
                                                     pe,
                                                     d_head,
                                                     prepared_pe,
                                                     flux_sp_flash_attn_enabled(ctx) ? GGML_TYPE_F16 : GGML_TYPE_F32);
        GGML_ASSERT(pe->ne[3] == q->ne[1]);
        return flux_sp_attention_prepared_qk(ctx, q, k, v, mask, name_prefix);
    }

#ifdef ED_DEBUG_SP_COMM
    static inline void mark_flux_debug_seq_to_head_compare(GGMLRunnerContext* ctx,
                                                           ggml_tensor* tensor,
                                                           const std::string& name) {
        if (!debug_sp_capture_enabled()) {
            return;
        }
        if (ctx == nullptr ||
            tensor == nullptr ||
            ctx->process_group == nullptr ||
            !ctx->process_group->enabled()) {
            return;
        }

        const int world_size = ctx->process_group->size();
        if (world_size <= 1) {
            return;
        }

        const int rank = ctx->process_group->rank();

        if (rank < 0 || rank >= world_size) {
            LOG_WARN("flux debug seq_to_head compare skipped: name=%s rank=%d world_size=%d",
                     name.c_str(),
                     rank,
                     world_size);
            return;
        }
        if (tensor->ne[3] != 1) {
            LOG_WARN("flux debug seq_to_head compare skipped: name=%s batch=%" PRId64,
                     name.c_str(),
                     tensor->ne[3]);
            return;
        }
        if (tensor->ne[1] % world_size != 0) {
            LOG_WARN("flux debug seq_to_head compare skipped: name=%s heads=%" PRId64 " world_size=%d",
                     name.c_str(),
                     tensor->ne[1],
                     world_size);
            return;
        }

        const int64_t pad = edgedit::parallel::sp_sequence_padding(tensor->ne[2],
                                                                   world_size);
        ggml_tensor* padded = tensor;
        if (pad > 0) {
            padded = ggml_pad(ctx->ggml_ctx, tensor, 0, 0, static_cast<int>(pad), 0);
            ggml_set_name(padded, (name + "_padded").c_str());
        }

        const int64_t padded_seq = padded->ne[2];
        const int64_t local_seq  = padded_seq / world_size;
        ggml_tensor* local_view = ggml_view_4d(ctx->ggml_ctx,
                                               padded,
                                               padded->ne[0],
                                               padded->ne[1],
                                               local_seq,
                                               padded->ne[3],
                                               padded->nb[1],
                                               padded->nb[2],
                                               padded->nb[3],
                                               static_cast<size_t>(rank) *
                                                   static_cast<size_t>(local_seq) *
                                                   padded->nb[2]);
        ggml_set_name(local_view, (name + "_local_view").c_str());
        ggml_tensor* local = ggml_cont(ctx->ggml_ctx, local_view);
        ggml_set_name(local, (name + "_local").c_str());

        auto layout = edgedit::parallel::sp_all_to_all_4d_seq_to_head(ctx->ggml_ctx,
                                                                      local,
                                                                      ctx->process_group,
                                                                      world_size,
                                                                      name + "_a2a");

        ggml_tensor* ref = flux_debug_head_shard_reference(ctx->ggml_ctx,
                                                           padded,
                                                           rank,
                                                           world_size,
                                                           name);
        ggml_tensor* seq_err = flux_debug_sse(ctx->ggml_ctx,
                                              layout.output,
                                              ref,
                                              name + "_cmp");
        accumulate_debug_sp_error(ctx->ggml_ctx, seq_err, name + "_cmp");

        auto roundtrip = edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx->ggml_ctx,
                                                                         layout.output,
                                                                         ctx->process_group,
                                                                         world_size,
                                                                         name + "_back");
        ggml_tensor* roundtrip_err = flux_debug_sse(ctx->ggml_ctx,
                                                    roundtrip.output,
                                                    local,
                                                    name + "_rt");
        accumulate_debug_sp_error(ctx->ggml_ctx, roundtrip_err, name + "_rt");

        LOG_DEBUG("flux debug seq_to_head compare marker: name=%s rank=%d world_size=%d full=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] local=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] a2a=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] ref=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] back=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] pad=%" PRId64,
                  name.c_str(),
                  rank,
                  world_size,
                  tensor->ne[0],
                  tensor->ne[1],
                  tensor->ne[2],
                  tensor->ne[3],
                  local->ne[0],
                  local->ne[1],
                  local->ne[2],
                  local->ne[3],
                  layout.output->ne[0],
                  layout.output->ne[1],
                  layout.output->ne[2],
                  layout.output->ne[3],
                  ref->ne[0],
                  ref->ne[1],
                  ref->ne[2],
                  ref->ne[3],
                  roundtrip.output->ne[0],
                  roundtrip.output->ne[1],
                  roundtrip.output->ne[2],
                  roundtrip.output->ne[3],
                  pad);
    }
#endif

    struct MLPEmbedder : public UnaryBlock {
    public:
        MLPEmbedder(int64_t in_dim, int64_t hidden_dim, bool bias = true) {
            blocks["in_layer"]  = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, bias));
            blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, bias));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [..., in_dim]
            // return: [..., hidden_dim]
            auto in_layer  = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
            auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);

            x = in_layer->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = out_layer->forward(ctx, x);
            return x;
        }
    };

    struct SelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;

        struct SPQKV {
            ggml_tensor* q = nullptr;
            ggml_tensor* k = nullptr;
            ggml_tensor* v = nullptr;
            ggml_tensor* recv_flat = nullptr;
            bool mixed_recv_flat = false;
            int64_t heads = 0;
            int64_t head_dim = 0;
            int64_t sequence = 0;
            int64_t shard_sequence = 0;
            int world_size = 0;
        };

    public:
        SelfAttention(int64_t dim,
                      int64_t num_heads = 8,
                      bool qkv_bias     = false,
                      bool proj_bias    = true)
            : num_heads(num_heads) {
            int64_t head_dim = dim / num_heads;
            blocks["qkv"]    = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, proj_bias));
        }

        std::vector<ggml_tensor*> pre_attention(GGMLRunnerContext* ctx,
                                                ggml_tensor* x
#ifdef ED_DEBUG_SP_COMM
                                                ,
                                                const char* debug_seq_to_head_prefix = nullptr
#endif
        ) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

            auto qkv         = qkv_proj->forward(ctx, x);
            int64_t head_dim = qkv->ne[0] / 3 / num_heads;
            auto q           = ggml_view_4d(ctx->ggml_ctx, qkv, head_dim, num_heads, qkv->ne[1], qkv->ne[2],
                                            qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], 0);
            auto k           = ggml_view_4d(ctx->ggml_ctx, qkv, head_dim, num_heads, qkv->ne[1], qkv->ne[2],
                                            qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], (qkv->nb[0]) * qkv->ne[0] / 3);
            auto v           = ggml_view_4d(ctx->ggml_ctx, qkv, head_dim, num_heads, qkv->ne[1], qkv->ne[2],
                                            qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], (qkv->nb[0]) * 2 * qkv->ne[0] / 3);
#ifdef ED_DEBUG_SP_COMM
            if (debug_seq_to_head_prefix != nullptr && debug_seq_to_head_prefix[0] != '\0') {
                const std::string prefix(debug_seq_to_head_prefix);
                mark_flux_debug_seq_to_head_compare(ctx, q, prefix + "_q_seq_to_head");
                mark_flux_debug_seq_to_head_compare(ctx, k, prefix + "_k_seq_to_head");
                mark_flux_debug_seq_to_head_compare(ctx, v, prefix + "_v_seq_to_head");
            }
#endif
            q                = norm->query_norm(ctx, q);
            k                = norm->key_norm(ctx, k);
            return {q, k, v};
        }

        SPQKV pre_attention_sp_local_qkv(GGMLRunnerContext* ctx,
                                         ggml_tensor* x_local,
                                         bool qk_norm_f16 = false) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

            ggml_tensor* qkv = qkv_proj->forward(ctx, x_local);
            int64_t head_dim = qkv->ne[0] / 3 / num_heads;

            ggml_tensor* q = ggml_view_4d(ctx->ggml_ctx,
                                          qkv,
                                          head_dim,
                                          num_heads,
                                          qkv->ne[1],
                                          qkv->ne[2],
                                          qkv->nb[0] * head_dim,
                                          qkv->nb[1],
                                          qkv->nb[2],
                                          0);
            ggml_tensor* k = ggml_view_4d(ctx->ggml_ctx,
                                          qkv,
                                          head_dim,
                                          num_heads,
                                          qkv->ne[1],
                                          qkv->ne[2],
                                          qkv->nb[0] * head_dim,
                                          qkv->nb[1],
                                          qkv->nb[2],
                                          qkv->nb[0] * qkv->ne[0] / 3);
            ggml_tensor* v = ggml_view_4d(ctx->ggml_ctx,
                                          qkv,
                                          head_dim,
                                          num_heads,
                                          qkv->ne[1],
                                          qkv->ne[2],
                                          qkv->nb[0] * head_dim,
                                          qkv->nb[1],
                                          qkv->nb[2],
                                          qkv->nb[0] * 2 * qkv->ne[0] / 3);

            const bool pre_qk_norm = flux_sp_pre_qk_norm_enabled();
            if (pre_qk_norm) {
                if (qk_norm_f16) {
                    q = norm->query_norm_f16(ctx, q);
                    k = norm->key_norm_f16(ctx, k);
                } else {
                    q = norm->query_norm(ctx, q);
                    k = norm->key_norm(ctx, k);
                }
            }
            SPQKV out;
            out.q = q;
            out.k = k;
            out.v = v;
            return out;
        }

        SPQKV pre_attention_sp(GGMLRunnerContext* ctx,
                               ggml_tensor* x_local,
                               const std::string& name_prefix) {
            auto norm = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            const int world_size = flux_sp_world_size(ctx);
            SPQKV local_qkv = pre_attention_sp_local_qkv(ctx, x_local);
            ggml_tensor* q = local_qkv.q;
            ggml_tensor* k = local_qkv.k;
            ggml_tensor* v = local_qkv.v;
            const bool pre_qk_norm = flux_sp_pre_qk_norm_enabled();

            std::vector<edgedit::parallel::SPSeqToHeadOutputLayout> output_layouts = {
                flux_sp_qk_seq_major_enabled() ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajor :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
                flux_sp_qk_seq_major_enabled() ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajor :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
                flux_sp_flash_v_seq_major_enabled(ctx) ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajor :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
            };
            auto qkv_head = flux_sp_fused_qkv_send_pack_enabled() ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_packed_layouts(ctx->ggml_ctx,
                                                                                                   q,
                                                                                                   k,
                                                                                                   v,
                                                                                                   output_layouts,
                                                                                                   ctx->process_group,
                                                                                                   world_size,
                                                                                                   name_prefix + "_qkv_seq_to_head") :
                                edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched_layouts(ctx->ggml_ctx,
                                                                                                {q, k, v},
                                                                                                output_layouts,
                                                                                                ctx->process_group,
                                                                                                world_size,
                                                                                                name_prefix + "_qkv_seq_to_head");
            GGML_ASSERT(qkv_head.outputs.size() == 3);

            q = pre_qk_norm ? qkv_head.outputs[0] : norm->query_norm(ctx, qkv_head.outputs[0]);
            k = pre_qk_norm ? qkv_head.outputs[1] : norm->key_norm(ctx, qkv_head.outputs[1]);
            v = qkv_head.outputs[2];

            SPQKV out;
            out.q = q;
            out.k = k;
            out.v = v;
            out.recv_flat = qkv_head.recv_flat;
            out.heads = q->ne[1];
            out.head_dim = q->ne[0];
            out.sequence = qkv_head.sequence;
            out.shard_sequence = qkv_head.shard_sequence;
            out.world_size = world_size;
            return out;
        }

        SPQKV pre_attention_sp_rope_work_qk(GGMLRunnerContext* ctx,
                                            ggml_tensor* x_local,
                                            const std::string& name_prefix,
                                            bool use_mixed_qkv_send = false,
                                            bool use_f16_qkv_send = false) {
            const int world_size = flux_sp_world_size(ctx);
            GGML_ASSERT(flux_sp_pre_qk_norm_enabled());
            use_mixed_qkv_send = use_mixed_qkv_send &&
                                 flux_sp_mixed_qkv_send_enabled() &&
                                 flux_sp_fused_qkv_send_pack_enabled() &&
                                 flux_sp_flash_v_seq_major_enabled(ctx);
            use_f16_qkv_send = !use_mixed_qkv_send &&
                                use_f16_qkv_send &&
                                flux_sp_f16_qkv_send_enabled() &&
                                flux_sp_fused_qkv_send_pack_enabled() &&
                                flux_sp_flash_v_seq_major_enabled(ctx);
            SPQKV local_qkv = pre_attention_sp_local_qkv(ctx,
                                                         x_local,
                                                         flux_sp_f16_qk_norm_enabled() && use_f16_qkv_send);
            ggml_tensor* q = local_qkv.q;
            ggml_tensor* k = local_qkv.k;
            ggml_tensor* v = local_qkv.v;

            std::vector<edgedit::parallel::SPSeqToHeadOutputLayout> output_layouts = {
                edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajorRopeInterleaved,
                edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajorRopeInterleaved,
                flux_sp_flash_v_seq_major_enabled(ctx) ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajor :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
            };
            auto qkv_head = use_mixed_qkv_send ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_mixed_recv_only(ctx->ggml_ctx,
                                                                                                     q,
                                                                                                     k,
                                                                                                     v,
                                                                                                     ctx->process_group,
                                                                                                     world_size,
                                                                                                     name_prefix + "_qkv_seq_to_head") :
                            use_f16_qkv_send ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_f16_recv_only(ctx->ggml_ctx,
                                                                                                  q,
                                                                                                  k,
                                                                                                  v,
                                                                                                  ctx->process_group,
                                                                                                  world_size,
                                                                                                  name_prefix + "_qkv_seq_to_head") :
                            flux_sp_fused_qkv_send_pack_enabled() ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_packed_layouts(ctx->ggml_ctx,
                                                                                                   q,
                                                                                                   k,
                                                                                                   v,
                                                                                                   output_layouts,
                                                                                                   ctx->process_group,
                                                                                                   world_size,
                                                                                                   name_prefix + "_qkv_seq_to_head") :
                                edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched_layouts(ctx->ggml_ctx,
                                                                                                {q, k, v},
                                                                                                output_layouts,
                                                                                                ctx->process_group,
                                                                                                world_size,
                                                                                                name_prefix + "_qkv_seq_to_head");
            if (!use_mixed_qkv_send && !use_f16_qkv_send) {
                GGML_ASSERT(qkv_head.outputs.size() == 3);
            }

            SPQKV out;
            out.q = (use_mixed_qkv_send || use_f16_qkv_send) ? q : qkv_head.outputs[0];
            out.k = (use_mixed_qkv_send || use_f16_qkv_send) ? k : qkv_head.outputs[1];
            out.v = (use_mixed_qkv_send || use_f16_qkv_send) ? v : qkv_head.outputs[2];
            out.recv_flat = qkv_head.recv_flat;
            out.mixed_recv_flat = use_mixed_qkv_send;
            out.heads = q->ne[1];
            out.head_dim = q->ne[0];
            out.sequence = qkv_head.sequence;
            out.shard_sequence = qkv_head.shard_sequence;
            out.world_size = world_size;
            return out;
        }

        ggml_tensor* post_attention(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            x = proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            auto qkv = pre_attention(ctx, x);                                   // q,k,v: [N, n_token, n_head, d_head]
            x        = Rope::attention(ctx, qkv[0], qkv[1], qkv[2], pe, mask, 1.0f, true, true);  // [N, n_token, dim]
            x        = post_attention(ctx, x);                                  // [N, n_token, dim]
            return x;
        }
    };

    struct MLP : public UnaryBlock {
        bool use_mlp_silu_act;

    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool use_mlp_silu_act = false, bool bias = false)
            : use_mlp_silu_act(use_mlp_silu_act) {
            int64_t mlp_mult_factor = use_mlp_silu_act ? 2 : 1;
            blocks["0"]             = std::make_shared<Linear>(hidden_size, intermediate_size * mlp_mult_factor, bias);
            blocks["2"]             = std::make_shared<Linear>(intermediate_size, hidden_size, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

            x = mlp_0->forward(ctx, x);
            if (use_mlp_silu_act) {
                x = ggml_ext_silu_act(ctx->ggml_ctx, x);
            } else {
                x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            }
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct YakMLP : public UnaryBlock {
    public:
        YakMLP(int64_t hidden_size, int64_t intermediate_size, bool bias = true) {
            blocks["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, bias);
            blocks["up_proj"]   = std::make_shared<Linear>(hidden_size, intermediate_size, bias);
            blocks["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto gate = gate_proj->forward(ctx, x);
            gate      = ggml_silu_inplace(ctx->ggml_ctx, gate);
            x         = up_proj->forward(ctx, x);
            x         = ggml_mul(ctx->ggml_ctx, x, gate);
            x         = down_proj->forward(ctx, x);
            return x;
        }
    };

    struct ModulationOut {
        ggml_tensor* shift = nullptr;
        ggml_tensor* scale = nullptr;
        ggml_tensor* gate  = nullptr;

        ModulationOut(ggml_tensor* shift = nullptr, ggml_tensor* scale = nullptr, ggml_tensor* gate = nullptr)
            : shift(shift), scale(scale), gate(gate) {}

        ModulationOut(GGMLRunnerContext* ctx, ggml_tensor* vec, int64_t offset) {
            int64_t stride = vec->nb[1] * vec->ne[1];
            shift          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            scale          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            gate           = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 2));  // [N, dim]
        }
    };

    struct Modulation : public GGMLBlock {
    public:
        bool is_double;
        int multiplier;

    public:
        Modulation(int64_t dim, bool is_double, bool bias = true)
            : is_double(is_double) {
            multiplier    = is_double ? 6 : 3;
            blocks["lin"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * multiplier, bias));
        }

        std::vector<ModulationOut> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            // x: [N, dim]
            // return: [ModulationOut, ModulationOut]
            auto lin = std::dynamic_pointer_cast<Linear>(blocks["lin"]);

            auto out = ggml_silu(ctx->ggml_ctx, vec);
            out      = lin->forward(ctx, out);  // [N, multiplier*dim]

            auto m = ggml_reshape_3d(ctx->ggml_ctx, out, vec->ne[0], multiplier, vec->ne[1]);  // [N, multiplier, dim]
            m      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, m, 0, 2, 1, 3));     // [multiplier, N, dim]

            ModulationOut m_0 = ModulationOut(ctx, m, 0);
            if (is_double) {
                return {m_0, ModulationOut(ctx, m, 3)};
            }

            return {m_0, ModulationOut()};
        }
    };

    struct DoubleStreamBlock : public GGMLBlock {
        bool prune_mod;
        int idx = 0;

    public:
        DoubleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio,
                          int idx               = 0,
                          bool qkv_bias         = false,
                          bool prune_mod        = false,
                          bool share_modulation = false,
                          bool mlp_proj_bias    = true,
                          bool use_yak_mlp      = false,
                          bool use_mlp_silu_act = false)
            : idx(idx), prune_mod(prune_mod) {
            int64_t mlp_hidden_dim = static_cast<int64_t>(hidden_size * mlp_ratio);

            if (!prune_mod && !share_modulation) {
                blocks["img_mod"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, true));
            }
            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["img_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, mlp_proj_bias));

            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            if (use_yak_mlp) {
                blocks["img_mlp"] = std::shared_ptr<GGMLBlock>(new YakMLP(hidden_size, mlp_hidden_dim, mlp_proj_bias));
            } else {
                blocks["img_mlp"] = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, mlp_hidden_dim, use_mlp_silu_act, mlp_proj_bias));
            }

            if (!prune_mod && !share_modulation) {
                blocks["txt_mod"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, true));
            }
            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["txt_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, mlp_proj_bias));

            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            if (use_yak_mlp) {
                blocks["txt_mlp"] = std::shared_ptr<GGMLBlock>(new YakMLP(hidden_size, mlp_hidden_dim, mlp_proj_bias));
            } else {
                blocks["txt_mlp"] = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, mlp_hidden_dim, use_mlp_silu_act, mlp_proj_bias));
            }
        }

        std::vector<ModulationOut> get_distil_img_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            // TODO: not hardcoded?
            const int single_blocks_count = 38;
            const int double_blocks_count = 19;

            int64_t offset = 6 * idx + 3 * single_blocks_count;
            return {ModulationOut(ctx, vec, offset), ModulationOut(ctx, vec, offset + 3)};
        }

        std::vector<ModulationOut> get_distil_txt_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            // TODO: not hardcoded?
            const int single_blocks_count = 38;
            const int double_blocks_count = 19;

            int64_t offset = 6 * idx + 6 * double_blocks_count + 3 * single_blocks_count;
            return {ModulationOut(ctx, vec, offset), ModulationOut(ctx, vec, offset + 3)};
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* vec,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* mask                   = nullptr,
                                                      std::vector<ModulationOut> img_mods = {},
                                                      std::vector<ModulationOut> txt_mods = {}) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_mlp"]);

            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_mlp"]);

            if (img_mods.empty()) {
                if (prune_mod) {
                    img_mods = get_distil_img_mod(ctx, vec);
                } else {
                    auto img_mod = std::dynamic_pointer_cast<Modulation>(blocks["img_mod"]);
                    img_mods     = img_mod->forward(ctx, vec);
                }
            }
            ModulationOut img_mod1 = img_mods[0];
            ModulationOut img_mod2 = img_mods[1];
            if (txt_mods.empty()) {
                if (prune_mod) {
                    txt_mods = get_distil_txt_mod(ctx, vec);
                } else {
                    auto txt_mod = std::dynamic_pointer_cast<Modulation>(blocks["txt_mod"]);
                    txt_mods     = txt_mod->forward(ctx, vec);
                }
            }
            ModulationOut txt_mod1 = txt_mods[0];
            ModulationOut txt_mod2 = txt_mods[1];

            // prepare image for attention
            auto img_modulated = img_norm1->forward(ctx, img);
            img_modulated      = dit::modulate(ctx->ggml_ctx, img_modulated, img_mod1.shift, img_mod1.scale);
#ifdef ED_DEBUG_SP_COMM
            const char* img_debug_prefix = idx == 0 ? "flux_debug_double0_img" : nullptr;
            auto img_qkv                 = img_attn->pre_attention(ctx, img_modulated, img_debug_prefix);  // q,k,v: [N, n_img_token, n_head, d_head]
#else
            auto img_qkv = img_attn->pre_attention(ctx, img_modulated);  // q,k,v: [N, n_img_token, n_head, d_head]
#endif
            auto img_q         = img_qkv[0];
            auto img_k         = img_qkv[1];
            auto img_v         = img_qkv[2];

            // prepare txt for attention
            auto txt_modulated = txt_norm1->forward(ctx, txt);
            txt_modulated      = dit::modulate(ctx->ggml_ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
#ifdef ED_DEBUG_SP_COMM
            const char* txt_debug_prefix = idx == 0 ? "flux_debug_double0_txt" : nullptr;
            auto txt_qkv                 = txt_attn->pre_attention(ctx, txt_modulated, txt_debug_prefix);  // q,k,v: [N, n_txt_token, n_head, d_head]
#else
            auto txt_qkv = txt_attn->pre_attention(ctx, txt_modulated);  // q,k,v: [N, n_txt_token, n_head, d_head]
#endif
            auto txt_q         = txt_qkv[0];
            auto txt_k         = txt_qkv[1];
            auto txt_v         = txt_qkv[2];

            // run actual attention
            auto q = ggml_concat(ctx->ggml_ctx, txt_q, img_q, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto k = ggml_concat(ctx->ggml_ctx, txt_k, img_k, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto v = ggml_concat(ctx->ggml_ctx, txt_v, img_v, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]

            auto attn         = Rope::attention(ctx, q, k, v, pe, mask, 1.0f, true, true);  // [N, n_txt_token + n_img_token, n_head*d_head]
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             txt->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);  // [N, n_txt_token, hidden_size]
            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             img->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             txt->ne[1] * attn->nb[1]);  // [N, n_img_token, hidden_size]

            // calculate the img bloks
            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn->post_attention(ctx, img_attn_out), img_mod1.gate));

            auto img_mlp_out = img_mlp->forward(ctx, dit::modulate(ctx->ggml_ctx, img_norm2->forward(ctx, img), img_mod2.shift, img_mod2.scale));

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_mod2.gate));

            // calculate the txt bloks
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn->post_attention(ctx, txt_attn_out), txt_mod1.gate));

            auto txt_mlp_out = txt_mlp->forward(ctx, dit::modulate(ctx->ggml_ctx, txt_norm2->forward(ctx, txt), txt_mod2.shift, txt_mod2.scale));
            txt              = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_mod2.gate));

            return {img, txt};
        }

#ifndef ED_DEBUG_SP_COMM
        std::pair<ggml_tensor*, ggml_tensor*> forward_sp(GGMLRunnerContext* ctx,
                                                         ggml_tensor* img,
                                                         ggml_tensor* txt,
                                                         ggml_tensor* vec,
                                                         ggml_tensor* pe,
                                                         ggml_tensor* mask                   = nullptr,
                                                         std::vector<ModulationOut> img_mods = {},
                                                         std::vector<ModulationOut> txt_mods = {},
                                                         ggml_tensor* prepared_pe_seq_major  = nullptr) {
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_mlp"]);

            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_mlp"]);

            if (img_mods.empty()) {
                if (prune_mod) {
                    img_mods = get_distil_img_mod(ctx, vec);
                } else {
                    auto img_mod = std::dynamic_pointer_cast<Modulation>(blocks["img_mod"]);
                    img_mods     = img_mod->forward(ctx, vec);
                }
            }
            ModulationOut img_mod1 = img_mods[0];
            ModulationOut img_mod2 = img_mods[1];
            if (txt_mods.empty()) {
                if (prune_mod) {
                    txt_mods = get_distil_txt_mod(ctx, vec);
                } else {
                    auto txt_mod = std::dynamic_pointer_cast<Modulation>(blocks["txt_mod"]);
                    txt_mods     = txt_mod->forward(ctx, vec);
                }
            }
            ModulationOut txt_mod1 = txt_mods[0];
            ModulationOut txt_mod2 = txt_mods[1];

            const std::string prefix = "flux_double" + std::to_string(idx);
            const int world_size     = flux_sp_world_size(ctx);

            ggml_tensor* img_modulated = img_norm1->forward(ctx, img);
            img_modulated              = flux_sp_modulate(ctx->ggml_ctx, img_modulated, img_mod1.shift, img_mod1.scale);

            ggml_tensor* txt_modulated = txt_norm1->forward(ctx, txt);
            txt_modulated              = flux_sp_modulate(ctx->ggml_ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
            const bool qk_seq_major = flux_sp_qk_seq_major_enabled();
            const bool qk_rope_work = qk_seq_major && flux_sp_pre_qk_norm_enabled();
            const bool use_mixed_qkv_send =
                flux_sp_mixed_qkv_send_enabled() &&
                flux_sp_mixed_double_qkv_send_enabled() &&
                flux_sp_fused_qkv_send_pack_enabled() &&
                flux_sp_fused_double_qkv_recv_prep_enabled() &&
                qk_rope_work &&
                flux_sp_flash_v_seq_major_enabled(ctx) &&
                mask == nullptr;
            const bool use_f16_qkv_send =
                !use_mixed_qkv_send &&
                flux_sp_f16_qkv_send_enabled() &&
                flux_sp_fused_qkv_send_pack_enabled() &&
                flux_sp_fused_double_qkv_recv_prep_enabled() &&
                qk_rope_work &&
                flux_sp_flash_v_seq_major_enabled(ctx) &&
                mask == nullptr;
            ggml_tensor* q = nullptr;
            ggml_tensor* k = nullptr;
            ggml_tensor* v = nullptr;
            bool used_fused_double_qkv_recv_prep = false;
            const bool use_combined_double_qkv_a2a =
                flux_sp_fuse_double_qkv_a2a_enabled() &&
                use_f16_qkv_send &&
                flux_sp_f16_qk_norm_enabled() &&
                !use_mixed_qkv_send;
            SelfAttention::SPQKV txt_qkv;
            SelfAttention::SPQKV img_qkv;
            if (use_combined_double_qkv_a2a) {
                txt_qkv = txt_attn->pre_attention_sp_local_qkv(ctx, txt_modulated, true);
                img_qkv = img_attn->pre_attention_sp_local_qkv(ctx, img_modulated, true);
                auto qkv_head =
                    edgedit::parallel::sp_all_to_all_4d_double_qkv_seq_to_head_f16_recv_only(ctx->ggml_ctx,
                                                                                             txt_qkv.q,
                                                                                             txt_qkv.k,
                                                                                             txt_qkv.v,
                                                                                             img_qkv.q,
                                                                                             img_qkv.k,
                                                                                             img_qkv.v,
                                                                                             ctx->process_group,
                                                                                             world_size,
                                                                                             prefix + "_txt_img_qkv_seq_to_head");
                ggml_tensor* prepared_pe = prepared_pe_seq_major != nullptr ?
                                               prepared_pe_seq_major :
                                               flux_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                                 pe,
                                                                                 prefix + "_combined_pair_recv_pe_seq_major");
                const ggml_type q_recv_prep_type = flux_sp_f16_q_attention_enabled() ? GGML_TYPE_F16 : GGML_TYPE_F32;
                ggml_tensor* fused_q = nullptr;
                ggml_tensor* fused_k = nullptr;
                ggml_tensor* fused_v = nullptr;
                if (flux_sp_bundle_double_qkv_recv_prep_enabled() &&
                    q_recv_prep_type == GGML_TYPE_F16) {
                    ggml_tensor* bundle =
                        edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_bundle_custom(ctx->ggml_ctx,
                                                                                              qkv_head.recv_flat,
                                                                                              prepared_pe,
                                                                                              world_size,
                                                                                              txt_qkv.q->ne[1],
                                                                                              txt_qkv.q->ne[0],
                                                                                              txt_qkv.q->ne[2]);
                    if (bundle != nullptr) {
                        const size_t plane_offset = static_cast<size_t>(bundle->nb[3]);
                        fused_q = ggml_view_3d(ctx->ggml_ctx,
                                               bundle,
                                               bundle->ne[0],
                                               bundle->ne[1],
                                               bundle->ne[2],
                                               bundle->nb[1],
                                               bundle->nb[2],
                                               0);
                        fused_k = ggml_view_3d(ctx->ggml_ctx,
                                               bundle,
                                               bundle->ne[0],
                                               bundle->ne[1],
                                               bundle->ne[2],
                                               bundle->nb[1],
                                               bundle->nb[2],
                                               plane_offset);
                        fused_v = ggml_view_3d(ctx->ggml_ctx,
                                               bundle,
                                               bundle->ne[0],
                                               bundle->ne[1],
                                               bundle->ne[2],
                                               bundle->nb[1],
                                               bundle->nb[2],
                                               2 * plane_offset);
                    }
                }
                if (fused_q == nullptr || fused_k == nullptr || fused_v == nullptr) {
                    fused_q = edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_custom(ctx->ggml_ctx,
                                                                                            qkv_head.recv_flat,
                                                                                            prepared_pe,
                                                                                            edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::Q,
                                                                                            world_size,
                                                                                            txt_qkv.q->ne[1],
                                                                                            txt_qkv.q->ne[0],
                                                                                            txt_qkv.q->ne[2],
                                                                                            q_recv_prep_type);
                    fused_k = edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_custom(ctx->ggml_ctx,
                                                                                            qkv_head.recv_flat,
                                                                                            prepared_pe,
                                                                                            edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::K,
                                                                                            world_size,
                                                                                            txt_qkv.q->ne[1],
                                                                                            txt_qkv.q->ne[0],
                                                                                            txt_qkv.q->ne[2]);
                    fused_v = edgedit::ggml_ext::flux_sp_qkv_combined_pair_recv_prep_custom(ctx->ggml_ctx,
                                                                                            qkv_head.recv_flat,
                                                                                            prepared_pe,
                                                                                            edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::V,
                                                                                            world_size,
                                                                                            txt_qkv.q->ne[1],
                                                                                            txt_qkv.q->ne[0],
                                                                                            txt_qkv.q->ne[2]);
                }
                if (fused_q != nullptr && fused_k != nullptr && fused_v != nullptr) {
                    q = fused_q;
                    k = fused_k;
                    v = fused_v;
                    used_fused_double_qkv_recv_prep = true;
                    txt_qkv.recv_flat = qkv_head.recv_flat;
                    img_qkv.recv_flat = qkv_head.recv_flat;
                    txt_qkv.world_size = world_size;
                    img_qkv.world_size = world_size;
                    txt_qkv.heads = txt_qkv.q->ne[1];
                    img_qkv.heads = img_qkv.q->ne[1];
                    txt_qkv.head_dim = txt_qkv.q->ne[0];
                    img_qkv.head_dim = img_qkv.q->ne[0];
                    txt_qkv.sequence = qkv_head.sequences.size() > 0 ? qkv_head.sequences[0] : txt_qkv.q->ne[2] * world_size;
                    img_qkv.sequence = qkv_head.sequences.size() > 1 ? qkv_head.sequences[1] : img_qkv.q->ne[2] * world_size;
                    txt_qkv.shard_sequence = txt_qkv.q->ne[2];
                    img_qkv.shard_sequence = img_qkv.q->ne[2];
                }
            } else {
                img_qkv = qk_rope_work ?
                              img_attn->pre_attention_sp_rope_work_qk(ctx, img_modulated, prefix + "_img", use_mixed_qkv_send, use_f16_qkv_send) :
                              img_attn->pre_attention_sp(ctx, img_modulated, prefix + "_img");
                txt_qkv = qk_rope_work ?
                              txt_attn->pre_attention_sp_rope_work_qk(ctx, txt_modulated, prefix + "_txt", use_mixed_qkv_send, use_f16_qkv_send) :
                              txt_attn->pre_attention_sp(ctx, txt_modulated, prefix + "_txt");
            }
            const bool can_fuse_double_qkv_recv_prep =
                flux_sp_fused_double_qkv_recv_prep_enabled() &&
                !used_fused_double_qkv_recv_prep &&
                qk_rope_work &&
                flux_sp_flash_v_seq_major_enabled(ctx) &&
                mask == nullptr &&
                txt_qkv.recv_flat != nullptr &&
                img_qkv.recv_flat != nullptr &&
                txt_qkv.world_size == world_size &&
                img_qkv.world_size == world_size &&
                txt_qkv.heads == img_qkv.heads &&
                txt_qkv.head_dim == img_qkv.head_dim &&
                txt_qkv.mixed_recv_flat == img_qkv.mixed_recv_flat;
            if (can_fuse_double_qkv_recv_prep) {
                ggml_tensor* prepared_pe = prepared_pe_seq_major != nullptr ?
                                               prepared_pe_seq_major :
                                               flux_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                                 pe,
                                                                                 prefix + "_pair_recv_pe_seq_major");
                const ggml_type q_recv_prep_type = flux_sp_f16_q_attention_enabled() ? GGML_TYPE_F16 : GGML_TYPE_F32;
                ggml_tensor* fused_q = txt_qkv.mixed_recv_flat ?
                                           edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_custom(ctx->ggml_ctx,
                                                                                                      txt_qkv.recv_flat,
                                                                                                      img_qkv.recv_flat,
                                                                                                      prepared_pe,
                                                                                                      edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::Q,
                                                                                                      world_size,
                                                                                                      txt_qkv.heads,
                                                                                                      txt_qkv.head_dim,
                                                                                                      q_recv_prep_type) :
                                           edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_custom(ctx->ggml_ctx,
                                                                                                 txt_qkv.recv_flat,
                                                                                                 img_qkv.recv_flat,
                                                                                                 prepared_pe,
                                                                                                 edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::Q,
                                                                                                 world_size,
                                                                                                 txt_qkv.heads,
                                                                                                 txt_qkv.head_dim,
                                                                                                 q_recv_prep_type);
                ggml_tensor* fused_k = txt_qkv.mixed_recv_flat ?
                                           edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_custom(ctx->ggml_ctx,
                                                                                                      txt_qkv.recv_flat,
                                                                                                      img_qkv.recv_flat,
                                                                                                      prepared_pe,
                                                                                                      edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::K,
                                                                                                      world_size,
                                                                                                      txt_qkv.heads,
                                                                                                      txt_qkv.head_dim) :
                                           edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_custom(ctx->ggml_ctx,
                                                                                                 txt_qkv.recv_flat,
                                                                                                 img_qkv.recv_flat,
                                                                                                 prepared_pe,
                                                                                                 edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::K,
                                                                                                 world_size,
                                                                                                 txt_qkv.heads,
                                                                                                 txt_qkv.head_dim);
                ggml_tensor* fused_v = txt_qkv.mixed_recv_flat ?
                                           edgedit::ggml_ext::flux_sp_qkv_pair_mixed_recv_prep_custom(ctx->ggml_ctx,
                                                                                                      txt_qkv.recv_flat,
                                                                                                      img_qkv.recv_flat,
                                                                                                      prepared_pe,
                                                                                                      edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::V,
                                                                                                      world_size,
                                                                                                      txt_qkv.heads,
                                                                                                      txt_qkv.head_dim) :
                                           edgedit::ggml_ext::flux_sp_qkv_pair_recv_prep_custom(ctx->ggml_ctx,
                                                                                                 txt_qkv.recv_flat,
                                                                                                 img_qkv.recv_flat,
                                                                                                 prepared_pe,
                                                                                                 edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::V,
                                                                                                 world_size,
                                                                                                 txt_qkv.heads,
                                                                                                 txt_qkv.head_dim);
                if (fused_q != nullptr && fused_k != nullptr && fused_v != nullptr) {
                    q = fused_q;
                    k = fused_k;
                    v = fused_v;
                    used_fused_double_qkv_recv_prep = true;
                }
            }
            GGML_ASSERT((!use_mixed_qkv_send && !use_f16_qkv_send) || used_fused_double_qkv_recv_prep);
            if (!used_fused_double_qkv_recv_prep) {
                q = ggml_concat(ctx->ggml_ctx, txt_qkv.q, img_qkv.q, qk_seq_major ? 1 : 2);
                k = ggml_concat(ctx->ggml_ctx, txt_qkv.k, img_qkv.k, qk_seq_major ? 1 : 2);
                v = ggml_concat(ctx->ggml_ctx,
                                txt_qkv.v,
                                img_qkv.v,
                                flux_sp_flash_v_seq_major_enabled(ctx) ? 1 : 2);
            }
            ggml_set_name(q, (prefix + "_q_attn").c_str());
            ggml_set_name(k, (prefix + "_k_attn").c_str());
            ggml_set_name(v, (prefix + "_v_attn").c_str());

            const int64_t head_dim = used_fused_double_qkv_recv_prep ? q->ne[0] :
                                     qk_rope_work ? q->ne[0] * 2 :
                                                    q->ne[0];
            ggml_tensor* attn = used_fused_double_qkv_recv_prep ?
                                    flux_sp_attention_prepared_qk(ctx, q, k, v, mask, prefix) :
                                qk_rope_work ?
                                    flux_sp_attention_from_rope_work_layout(ctx, q, k, v, pe, mask, head_dim, prefix, prepared_pe_seq_major) :
                                    flux_sp_attention(ctx, q, k, v, pe, mask, prefix, prepared_pe_seq_major);
            if (flux_sp_strict_barrier_enabled()) {
                sd::ggml_graph_cut::mark_graph_cut(attn, prefix + ".sp_attention", "attn");
            }

            const int64_t shard_heads  = (qk_seq_major || used_fused_double_qkv_recv_prep) ? q->ne[2] : q->ne[1];
            const int64_t txt_full_seq = txt_qkv.sequence > 0 ? txt_qkv.sequence :
                                         qk_seq_major ? txt_qkv.q->ne[1] :
                                                        txt_qkv.q->ne[2];
            const int64_t img_full_seq = img_qkv.sequence > 0 ? img_qkv.sequence :
                                         qk_seq_major ? img_qkv.q->ne[1] :
                                                        img_qkv.q->ne[2];
            const int64_t total_seq    = txt_full_seq + img_full_seq;

            ggml_tensor* attn_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                   attn,
                                                   head_dim,
                                                   shard_heads,
                                                   total_seq,
                                                   attn->ne[2]);
            ggml_set_name(attn_4d, (prefix + "_attn_4d").c_str());

            edgedit::parallel::SPAllToAll4DBatchLayout attn_local;
            const bool keep_double_attn_bf16 = flux_sp_bf16_double_head_to_seq_output_enabled() &&
                                               attn_4d->type == GGML_TYPE_F16;
            if (keep_double_attn_bf16) {
                attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_two_stream_packed_bf16_output(
                    ctx->ggml_ctx,
                    attn_4d,
                    txt_full_seq,
                    img_full_seq,
                    ctx->process_group,
                    world_size,
                    prefix + "_txt_img_attn_head_to_seq");
            } else if (flux_sp_f16_double_head_to_seq_output_enabled() || attn_4d->type == GGML_TYPE_F16) {
                attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_two_stream_packed_f16_output(
                    ctx->ggml_ctx,
                    attn_4d,
                    txt_full_seq,
                    img_full_seq,
                    ctx->process_group,
                    world_size,
                    prefix + "_txt_img_attn_head_to_seq");
            } else if (flux_sp_f16_double_head_to_seq_enabled()) {
                attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_two_stream_packed_f16(
                    ctx->ggml_ctx,
                    attn_4d,
                    txt_full_seq,
                    img_full_seq,
                    ctx->process_group,
                    world_size,
                    prefix + "_txt_img_attn_head_to_seq");
            } else if (flux_sp_fused_head_to_seq_enabled()) {
                attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_two_stream_packed(ctx->ggml_ctx,
                                                                                               attn_4d,
                                                                                               txt_full_seq,
                                                                                               img_full_seq,
                                                                                               ctx->process_group,
                                                                                               world_size,
                                                                                               prefix + "_txt_img_attn_head_to_seq");
            } else {
                ggml_tensor* txt_attn_head = ggml_view_4d(ctx->ggml_ctx,
                                                          attn_4d,
                                                          head_dim,
                                                          shard_heads,
                                                          txt_full_seq,
                                                          attn_4d->ne[3],
                                                          attn_4d->nb[1],
                                                          attn_4d->nb[2],
                                                          attn_4d->nb[3],
                                                          0);
                if (!ggml_is_contiguous(txt_attn_head)) {
                    txt_attn_head = ggml_cont(ctx->ggml_ctx, txt_attn_head);
                } else {
                    txt_attn_head = ggml_reshape_4d(ctx->ggml_ctx,
                                                    txt_attn_head,
                                                    txt_attn_head->ne[0],
                                                    txt_attn_head->ne[1],
                                                    txt_attn_head->ne[2],
                                                    txt_attn_head->ne[3]);
                }
                ggml_set_name(txt_attn_head, (prefix + "_txt_attn_head").c_str());

                ggml_tensor* img_attn_head = ggml_view_4d(ctx->ggml_ctx,
                                                          attn_4d,
                                                          head_dim,
                                                          shard_heads,
                                                          img_full_seq,
                                                          attn_4d->ne[3],
                                                          attn_4d->nb[1],
                                                          attn_4d->nb[2],
                                                          attn_4d->nb[3],
                                                          static_cast<size_t>(txt_full_seq) * attn_4d->nb[2]);
                if (!ggml_is_contiguous(img_attn_head)) {
                    img_attn_head = ggml_cont(ctx->ggml_ctx, img_attn_head);
                } else {
                    img_attn_head = ggml_reshape_4d(ctx->ggml_ctx,
                                                    img_attn_head,
                                                    img_attn_head->ne[0],
                                                    img_attn_head->ne[1],
                                                    img_attn_head->ne[2],
                                                    img_attn_head->ne[3]);
                }
                ggml_set_name(img_attn_head, (prefix + "_img_attn_head").c_str());

                attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_batched(ctx->ggml_ctx,
                                                                                     {txt_attn_head, img_attn_head},
                                                                                     ctx->process_group,
                                                                                     world_size,
                                                                                     prefix + "_txt_img_attn_head_to_seq");
            }
            GGML_ASSERT(attn_local.outputs.size() == 2);

            ggml_tensor* txt_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                                        attn_local.outputs[0],
                                                        head_dim * shard_heads * world_size,
                                                        attn_local.outputs[0]->ne[2],
                                                        attn_local.outputs[0]->ne[3]);
            ggml_set_name(txt_attn_out, (prefix + "_txt_attn_out").c_str());

            ggml_tensor* img_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                                        attn_local.outputs[1],
                                                        head_dim * shard_heads * world_size,
                                                        attn_local.outputs[1]->ne[2],
                                                        attn_local.outputs[1]->ne[3]);
            ggml_set_name(img_attn_out, (prefix + "_img_attn_out").c_str());

            ggml_tensor* img_post_attn = img_attn->post_attention(ctx, img_attn_out);
            img = flux_sp_residual_gate(ctx->ggml_ctx, img, img_post_attn, img_mod1.gate);

            ggml_tensor* img_mlp_in = flux_sp_modulate(ctx->ggml_ctx,
                                                       img_norm2->forward(ctx, img),
                                                       img_mod2.shift,
                                                       img_mod2.scale);
            auto img_mlp_out = img_mlp->forward(ctx, img_mlp_in);
            img = flux_sp_residual_gate(ctx->ggml_ctx, img, img_mlp_out, img_mod2.gate);

            ggml_tensor* txt_post_attn = txt_attn->post_attention(ctx, txt_attn_out);
            txt = flux_sp_residual_gate(ctx->ggml_ctx, txt, txt_post_attn, txt_mod1.gate);

            ggml_tensor* txt_mlp_in = flux_sp_modulate(ctx->ggml_ctx,
                                                       txt_norm2->forward(ctx, txt),
                                                       txt_mod2.shift,
                                                       txt_mod2.scale);
            auto txt_mlp_out = txt_mlp->forward(ctx, txt_mlp_in);
            txt = flux_sp_residual_gate(ctx->ggml_ctx, txt, txt_mlp_out, txt_mod2.gate);

            return {img, txt};
        }
#endif

#ifdef ED_DEBUG_SP_COMM
        std::pair<ggml_tensor*, ggml_tensor*> forward_sp(GGMLRunnerContext* ctx,
                                                         ggml_tensor* img,
                                                         ggml_tensor* txt,
                                                         ggml_tensor* vec,
                                                         ggml_tensor* pe,
                                                         ggml_tensor* mask                   = nullptr,
                                                         std::vector<ModulationOut> img_mods = {},
                                                         std::vector<ModulationOut> txt_mods = {},
                                                         ggml_tensor* debug_img_ref          = nullptr,
                                                         ggml_tensor* debug_txt_ref          = nullptr) {
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_mlp"]);

            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_mlp"]);

            if (img_mods.empty()) {
                if (prune_mod) {
                    img_mods = get_distil_img_mod(ctx, vec);
                } else {
                    auto img_mod = std::dynamic_pointer_cast<Modulation>(blocks["img_mod"]);
                    img_mods     = img_mod->forward(ctx, vec);
                }
            }
            ModulationOut img_mod1 = img_mods[0];
            ModulationOut img_mod2 = img_mods[1];
            if (txt_mods.empty()) {
                if (prune_mod) {
                    txt_mods = get_distil_txt_mod(ctx, vec);
                } else {
                    auto txt_mod = std::dynamic_pointer_cast<Modulation>(blocks["txt_mod"]);
                    txt_mods     = txt_mod->forward(ctx, vec);
                }
            }
            ModulationOut txt_mod1 = txt_mods[0];
            ModulationOut txt_mod2 = txt_mods[1];

            const std::string prefix = "flux_double" + std::to_string(idx);
            const int world_size     = flux_sp_world_size(ctx);
            const std::string debug_stage = debug_sp_mainline_compare_stage();
            const std::string debug_inner_prefix = "double_inner";
            const bool debug_inner    = debug_img_ref != nullptr &&
                                     debug_txt_ref != nullptr &&
                                     (debug_stage == debug_inner_prefix ||
                                      debug_stage.rfind(debug_inner_prefix + "_", 0) == 0);
            std::string debug_inner_stop;
            if (debug_inner) {
                if (debug_stage.size() > debug_inner_prefix.size() + 1) {
                    debug_inner_stop = debug_stage.substr(debug_inner_prefix.size() + 1);
                } else {
                    const char* stop_env = std::getenv("ED_FLUX_SP_COMPARE_INNER_STOP");
                    if (stop_env != nullptr && stop_env[0] != '\0') {
                        debug_inner_stop = stop_env;
                    }
                }
            }
            auto debug_inner_should_stop = [&](const char* phase) {
                return debug_inner && debug_inner_stop == phase;
            };

            auto debug_compare_sequence = [&](ggml_tensor* local,
                                              ggml_tensor* full,
                                              const std::string& name) {
                if (!debug_inner) {
                    return;
                }
                ggml_tensor* ref = flux_debug_sequence_shard_reference(ctx, full, name);
                mark_flux_debug_compare_tensor(ctx->ggml_ctx, local, ref, name);
            };

            auto debug_compare_head = [&](ggml_tensor* head_local,
                                          ggml_tensor* full,
                                          const std::string& name) {
                if (!debug_inner) {
                    return;
                }
                ggml_tensor* ref = flux_debug_head_shard_reference(ctx->ggml_ctx,
                                                                   full,
                                                                   flux_sp_rank(ctx),
                                                                   world_size,
                                                                   name);
                mark_flux_debug_compare_tensor(ctx->ggml_ctx, head_local, ref, name);
            };

            if (debug_inner) {
                LOG_INFO("flux debug double_inner enter block=%d stop=%s img=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] txt=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] img_ref=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] txt_ref=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
                         idx,
                         debug_inner_stop.c_str(),
                         img->ne[0],
                         img->ne[1],
                         img->ne[2],
                         img->ne[3],
                         txt->ne[0],
                         txt->ne[1],
                         txt->ne[2],
                         txt->ne[3],
                         debug_img_ref->ne[0],
                         debug_img_ref->ne[1],
                         debug_img_ref->ne[2],
                         debug_img_ref->ne[3],
                         debug_txt_ref->ne[0],
                         debug_txt_ref->ne[1],
                         debug_txt_ref->ne[2],
                         debug_txt_ref->ne[3]);
                debug_compare_sequence(img, debug_img_ref, prefix + "_inner_img_input");
                debug_compare_sequence(txt, debug_txt_ref, prefix + "_inner_txt_input");
                if (debug_inner_should_stop("input")) {
                    LOG_INFO("flux debug double_inner stop block=%d phase=input", idx);
                    return {img, txt};
                }
            }

            ggml_tensor* img_modulated = img_norm1->forward(ctx, img);
            img_modulated              = dit::modulate(ctx->ggml_ctx, img_modulated, img_mod1.shift, img_mod1.scale);
            ggml_tensor* debug_img_modulated_ref = nullptr;
            if (debug_inner) {
                debug_img_modulated_ref = img_norm1->forward(ctx, debug_img_ref);
                debug_img_modulated_ref = dit::modulate(ctx->ggml_ctx,
                                                        debug_img_modulated_ref,
                                                        img_mod1.shift,
                                                        img_mod1.scale);
                ggml_set_name(debug_img_modulated_ref, (prefix + "_debug_img_modulated_ref").c_str());
                debug_compare_sequence(img_modulated,
                                       debug_img_modulated_ref,
                                       prefix + "_inner_img_modulated");
                if (debug_inner_should_stop("img_mod")) {
                    return {img, txt};
                }
            }
            auto img_qkv               = img_attn->pre_attention_sp(ctx, img_modulated, prefix + "_img");
            std::vector<ggml_tensor*> debug_img_qkv_ref;
            if (debug_inner) {
                debug_img_qkv_ref = img_attn->pre_attention(ctx, debug_img_modulated_ref);
                debug_compare_head(img_qkv.q, debug_img_qkv_ref[0], prefix + "_inner_img_q");
                debug_compare_head(img_qkv.k, debug_img_qkv_ref[1], prefix + "_inner_img_k");
                debug_compare_head(img_qkv.v, debug_img_qkv_ref[2], prefix + "_inner_img_v");
            }

            ggml_tensor* txt_modulated = txt_norm1->forward(ctx, txt);
            txt_modulated              = dit::modulate(ctx->ggml_ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
            ggml_tensor* debug_txt_modulated_ref = nullptr;
            if (debug_inner) {
                debug_txt_modulated_ref = txt_norm1->forward(ctx, debug_txt_ref);
                debug_txt_modulated_ref = dit::modulate(ctx->ggml_ctx,
                                                        debug_txt_modulated_ref,
                                                        txt_mod1.shift,
                                                        txt_mod1.scale);
                ggml_set_name(debug_txt_modulated_ref, (prefix + "_debug_txt_modulated_ref").c_str());
                debug_compare_sequence(txt_modulated,
                                       debug_txt_modulated_ref,
                                       prefix + "_inner_txt_modulated");
                if (debug_inner_should_stop("txt_mod") ||
                    debug_inner_should_stop("mod")) {
                    return {img, txt};
                }
            }
            auto txt_qkv               = txt_attn->pre_attention_sp(ctx, txt_modulated, prefix + "_txt");
            std::vector<ggml_tensor*> debug_txt_qkv_ref;
            if (debug_inner) {
                debug_txt_qkv_ref = txt_attn->pre_attention(ctx, debug_txt_modulated_ref);
                debug_compare_head(txt_qkv.q, debug_txt_qkv_ref[0], prefix + "_inner_txt_q");
                debug_compare_head(txt_qkv.k, debug_txt_qkv_ref[1], prefix + "_inner_txt_k");
                debug_compare_head(txt_qkv.v, debug_txt_qkv_ref[2], prefix + "_inner_txt_v");
            }

            ggml_tensor* q = ggml_concat(ctx->ggml_ctx, txt_qkv.q, img_qkv.q, 2);
            ggml_tensor* k = ggml_concat(ctx->ggml_ctx, txt_qkv.k, img_qkv.k, 2);
            ggml_tensor* v = ggml_concat(ctx->ggml_ctx, txt_qkv.v, img_qkv.v, 2);
            q              = ggml_cont(ctx->ggml_ctx, q);
            ggml_set_name(q, (prefix + "_q_attn").c_str());
            k = ggml_cont(ctx->ggml_ctx, k);
            ggml_set_name(k, (prefix + "_k_attn").c_str());
            v = ggml_cont(ctx->ggml_ctx, v);
            ggml_set_name(v, (prefix + "_v_attn").c_str());

            ggml_tensor* debug_q_ref    = nullptr;
            ggml_tensor* debug_k_ref    = nullptr;
            ggml_tensor* debug_v_ref    = nullptr;
            ggml_tensor* debug_attn_ref = nullptr;
            if (debug_inner) {
                debug_q_ref = ggml_concat(ctx->ggml_ctx, debug_txt_qkv_ref[0], debug_img_qkv_ref[0], 2);
                ggml_set_name(debug_q_ref, (prefix + "_debug_q_ref").c_str());
                debug_k_ref = ggml_concat(ctx->ggml_ctx, debug_txt_qkv_ref[1], debug_img_qkv_ref[1], 2);
                ggml_set_name(debug_k_ref, (prefix + "_debug_k_ref").c_str());
                debug_v_ref = ggml_concat(ctx->ggml_ctx, debug_txt_qkv_ref[2], debug_img_qkv_ref[2], 2);
                ggml_set_name(debug_v_ref, (prefix + "_debug_v_ref").c_str());

                debug_compare_head(q, debug_q_ref, prefix + "_inner_q_attn");
                debug_compare_head(k, debug_k_ref, prefix + "_inner_k_attn");
                debug_compare_head(v, debug_v_ref, prefix + "_inner_v_attn");
                if (debug_inner_should_stop("qkv")) {
                    return std::pair<ggml_tensor*, ggml_tensor*>{img, txt};
                }

                debug_attn_ref = Rope::attention(ctx, debug_q_ref, debug_k_ref, debug_v_ref, pe, mask);
                debug_attn_ref = ggml_cont(ctx->ggml_ctx, debug_attn_ref);
                ggml_set_name(debug_attn_ref, (prefix + "_debug_attn_ref").c_str());
            }

            ggml_tensor* attn = flux_sp_attention(ctx, q, k, v, pe, mask, prefix);
            if (flux_sp_strict_barrier_enabled()) {
                sd::ggml_graph_cut::mark_graph_cut(attn, prefix + ".sp_attention", "attn");
            }

            const int64_t head_dim     = q->ne[0];
            const int64_t shard_heads  = q->ne[1];
            const int64_t txt_full_seq = txt_qkv.q->ne[2];
            const int64_t img_full_seq = img_qkv.q->ne[2];
            const int64_t total_seq    = txt_full_seq + img_full_seq;

            if (debug_inner) {
                log_flux_debug_reshape_4d(debug_attn_ref,
                                          head_dim,
                                          img_attn->num_heads,
                                          total_seq,
                                          debug_attn_ref->ne[2],
                                          prefix + "_debug_attn_ref_4d");
                ggml_tensor* debug_attn_ref_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                                  debug_attn_ref,
                                                                  head_dim,
                                                                  img_attn->num_heads,
                                                                  total_seq,
                                                                  debug_attn_ref->ne[2]);
                ggml_set_name(debug_attn_ref_4d, (prefix + "_debug_attn_ref_4d").c_str());
                ggml_tensor* debug_attn_ref_head = flux_debug_head_shard_reference(ctx->ggml_ctx,
                                                                                   debug_attn_ref_4d,
                                                                                   flux_sp_rank(ctx),
                                                                                   world_size,
                                                                                   prefix + "_inner_attn");
                ggml_tensor* debug_attn_ref_flat = ggml_reshape_3d(ctx->ggml_ctx,
                                                                    debug_attn_ref_head,
                                                                    head_dim * shard_heads,
                                                                    total_seq,
                                                                    debug_attn_ref_head->ne[3]);
                ggml_set_name(debug_attn_ref_flat, (prefix + "_debug_attn_ref_flat").c_str());
                mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                               attn,
                                               debug_attn_ref_flat,
                                               prefix + "_inner_attn");
                if (debug_inner_should_stop("attn")) {
                    return std::pair<ggml_tensor*, ggml_tensor*>{img, txt};
                }
            }

            if (debug_inner) {
                log_flux_debug_reshape_4d(attn,
                                          head_dim,
                                          shard_heads,
                                          total_seq,
                                          attn->ne[2],
                                          prefix + "_attn_4d");
            }
            ggml_tensor* attn_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                   attn,
                                                   head_dim,
                                                   shard_heads,
                                                   total_seq,
                                                   attn->ne[2]);
            ggml_set_name(attn_4d, (prefix + "_attn_4d").c_str());

            ggml_tensor* debug_txt_attn_out_ref = nullptr;
            ggml_tensor* debug_img_attn_out_ref = nullptr;
            if (debug_inner) {
                debug_txt_attn_out_ref = ggml_view_3d(ctx->ggml_ctx,
                                                       debug_attn_ref,
                                                       debug_attn_ref->ne[0],
                                                       txt_full_seq,
                                                       debug_attn_ref->ne[2],
                                                       debug_attn_ref->nb[1],
                                                       debug_attn_ref->nb[2],
                                                       0);
                debug_txt_attn_out_ref = ggml_cont(ctx->ggml_ctx, debug_txt_attn_out_ref);
                ggml_set_name(debug_txt_attn_out_ref, (prefix + "_debug_txt_attn_out_ref").c_str());

                debug_img_attn_out_ref = ggml_view_3d(ctx->ggml_ctx,
                                                       debug_attn_ref,
                                                       debug_attn_ref->ne[0],
                                                       img_full_seq,
                                                       debug_attn_ref->ne[2],
                                                       debug_attn_ref->nb[1],
                                                       debug_attn_ref->nb[2],
                                                       static_cast<size_t>(txt_full_seq) * debug_attn_ref->nb[1]);
                debug_img_attn_out_ref = ggml_cont(ctx->ggml_ctx, debug_img_attn_out_ref);
                ggml_set_name(debug_img_attn_out_ref, (prefix + "_debug_img_attn_out_ref").c_str());
            }

            ggml_tensor* txt_attn_head = ggml_view_4d(ctx->ggml_ctx,
                                                      attn_4d,
                                                      head_dim,
                                                      shard_heads,
                                                      txt_full_seq,
                                                      attn_4d->ne[3],
                                                      attn_4d->nb[1],
                                                      attn_4d->nb[2],
                                                      attn_4d->nb[3],
                                                      0);
            txt_attn_head = ggml_cont(ctx->ggml_ctx, txt_attn_head);
            ggml_set_name(txt_attn_head, (prefix + "_txt_attn_head").c_str());
            if (debug_inner) {
                log_flux_debug_reshape_4d(debug_txt_attn_out_ref,
                                          head_dim,
                                          txt_attn->num_heads,
                                          txt_full_seq,
                                          debug_txt_attn_out_ref->ne[2],
                                          prefix + "_debug_txt_attn_ref_4d");
                ggml_tensor* debug_txt_attn_ref_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                                     debug_txt_attn_out_ref,
                                                                     head_dim,
                                                                     txt_attn->num_heads,
                                                                     txt_full_seq,
                                                                     debug_txt_attn_out_ref->ne[2]);
                ggml_set_name(debug_txt_attn_ref_4d, (prefix + "_debug_txt_attn_ref_4d").c_str());
                debug_compare_head(txt_attn_head,
                                   debug_txt_attn_ref_4d,
                                   prefix + "_inner_txt_attn_head");
            }

            ggml_tensor* img_attn_head = ggml_view_4d(ctx->ggml_ctx,
                                                      attn_4d,
                                                      head_dim,
                                                      shard_heads,
                                                      img_full_seq,
                                                      attn_4d->ne[3],
                                                      attn_4d->nb[1],
                                                      attn_4d->nb[2],
                                                      attn_4d->nb[3],
                                                      static_cast<size_t>(txt_full_seq) * attn_4d->nb[2]);
            img_attn_head = ggml_cont(ctx->ggml_ctx, img_attn_head);
            ggml_set_name(img_attn_head, (prefix + "_img_attn_head").c_str());
            if (debug_inner) {
                log_flux_debug_reshape_4d(debug_img_attn_out_ref,
                                          head_dim,
                                          img_attn->num_heads,
                                          img_full_seq,
                                          debug_img_attn_out_ref->ne[2],
                                          prefix + "_debug_img_attn_ref_4d");
                ggml_tensor* debug_img_attn_ref_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                                                     debug_img_attn_out_ref,
                                                                     head_dim,
                                                                     img_attn->num_heads,
                                                                     img_full_seq,
                                                                     debug_img_attn_out_ref->ne[2]);
                ggml_set_name(debug_img_attn_ref_4d, (prefix + "_debug_img_attn_ref_4d").c_str());
                debug_compare_head(img_attn_head,
                                   debug_img_attn_ref_4d,
                                   prefix + "_inner_img_attn_head");
                if (debug_inner_should_stop("attn_head")) {
                    return std::pair<ggml_tensor*, ggml_tensor*>{img, txt};
                }
            }

            auto attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq_batched(ctx->ggml_ctx,
                                                                                      {txt_attn_head, img_attn_head},
                                                                                      ctx->process_group,
                                                                                      world_size,
                                                                                      prefix + "_txt_img_attn_head_to_seq");
            GGML_ASSERT(attn_local.outputs.size() == 2);

            ggml_tensor* txt_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                                        attn_local.outputs[0],
                                                        head_dim * shard_heads * world_size,
                                                        attn_local.outputs[0]->ne[2],
                                                        attn_local.outputs[0]->ne[3]);
            ggml_set_name(txt_attn_out, (prefix + "_txt_attn_out").c_str());
            if (debug_inner) {
                debug_compare_sequence(txt_attn_out,
                                       debug_txt_attn_out_ref,
                                       prefix + "_inner_txt_attn_out");
            }

            ggml_tensor* img_attn_out = ggml_reshape_3d(ctx->ggml_ctx,
                                                        attn_local.outputs[1],
                                                        head_dim * shard_heads * world_size,
                                                        attn_local.outputs[1]->ne[2],
                                                        attn_local.outputs[1]->ne[3]);
            ggml_set_name(img_attn_out, (prefix + "_img_attn_out").c_str());
            if (debug_inner) {
                debug_compare_sequence(img_attn_out,
                                       debug_img_attn_out_ref,
                                       prefix + "_inner_img_attn_out");
                if (debug_inner_should_stop("attn_out")) {
                    return std::pair<ggml_tensor*, ggml_tensor*>{img, txt};
                }
            }

            ggml_tensor* img_post_attn = img_attn->post_attention(ctx, img_attn_out);
            ggml_set_name(img_post_attn, (prefix + "_img_post_attn").c_str());
            ggml_tensor* debug_img_post_attn_ref = nullptr;
            if (debug_inner) {
                debug_img_post_attn_ref = img_attn->post_attention(ctx, debug_img_attn_out_ref);
                ggml_set_name(debug_img_post_attn_ref, (prefix + "_debug_img_post_attn_ref").c_str());
                debug_compare_sequence(img_post_attn,
                                       debug_img_post_attn_ref,
                                       prefix + "_inner_img_post_attn");
            }

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_post_attn, img_mod1.gate));
            ggml_set_name(img, (prefix + "_img_after_attn").c_str());
            ggml_tensor* debug_img_after_attn_ref = nullptr;
            if (debug_inner) {
                debug_img_after_attn_ref = ggml_add(ctx->ggml_ctx,
                                                    debug_img_ref,
                                                    ggml_mul(ctx->ggml_ctx,
                                                             debug_img_post_attn_ref,
                                                             img_mod1.gate));
                ggml_set_name(debug_img_after_attn_ref, (prefix + "_debug_img_after_attn_ref").c_str());
                debug_compare_sequence(img,
                                       debug_img_after_attn_ref,
                                       prefix + "_inner_img_after_attn");
            }

            ggml_tensor* img_mlp_in = dit::modulate(ctx->ggml_ctx,
                                                    img_norm2->forward(ctx, img),
                                                    img_mod2.shift,
                                                    img_mod2.scale);
            ggml_set_name(img_mlp_in, (prefix + "_img_mlp_in").c_str());
            ggml_tensor* debug_img_mlp_in_ref = nullptr;
            if (debug_inner) {
                debug_img_mlp_in_ref = dit::modulate(ctx->ggml_ctx,
                                                     img_norm2->forward(ctx, debug_img_after_attn_ref),
                                                     img_mod2.shift,
                                                     img_mod2.scale);
                ggml_set_name(debug_img_mlp_in_ref, (prefix + "_debug_img_mlp_in_ref").c_str());
                debug_compare_sequence(img_mlp_in,
                                       debug_img_mlp_in_ref,
                                       prefix + "_inner_img_mlp_in");
            }

            auto img_mlp_out = img_mlp->forward(ctx, img_mlp_in);
            ggml_set_name(img_mlp_out, (prefix + "_img_mlp_out").c_str());
            ggml_tensor* debug_img_mlp_out_ref = nullptr;
            if (debug_inner) {
                debug_img_mlp_out_ref = img_mlp->forward(ctx, debug_img_mlp_in_ref);
                ggml_set_name(debug_img_mlp_out_ref, (prefix + "_debug_img_mlp_out_ref").c_str());
                debug_compare_sequence(img_mlp_out,
                                       debug_img_mlp_out_ref,
                                       prefix + "_inner_img_mlp_out");
            }

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_mod2.gate));
            ggml_set_name(img, (prefix + "_img_after_mlp").c_str());
            if (debug_inner) {
                ggml_tensor* debug_img_after_mlp_ref = ggml_add(ctx->ggml_ctx,
                                                                debug_img_after_attn_ref,
                                                                ggml_mul(ctx->ggml_ctx,
                                                                         debug_img_mlp_out_ref,
                                                                         img_mod2.gate));
                ggml_set_name(debug_img_after_mlp_ref, (prefix + "_debug_img_after_mlp_ref").c_str());
                debug_compare_sequence(img,
                                       debug_img_after_mlp_ref,
                                       prefix + "_inner_img_after_mlp");
            }

            ggml_tensor* txt_post_attn = txt_attn->post_attention(ctx, txt_attn_out);
            ggml_set_name(txt_post_attn, (prefix + "_txt_post_attn").c_str());
            ggml_tensor* debug_txt_post_attn_ref = nullptr;
            if (debug_inner) {
                debug_txt_post_attn_ref = txt_attn->post_attention(ctx, debug_txt_attn_out_ref);
                ggml_set_name(debug_txt_post_attn_ref, (prefix + "_debug_txt_post_attn_ref").c_str());
                debug_compare_sequence(txt_post_attn,
                                       debug_txt_post_attn_ref,
                                       prefix + "_inner_txt_post_attn");
            }

            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_post_attn, txt_mod1.gate));
            ggml_set_name(txt, (prefix + "_txt_after_attn").c_str());
            ggml_tensor* debug_txt_after_attn_ref = nullptr;
            if (debug_inner) {
                debug_txt_after_attn_ref = ggml_add(ctx->ggml_ctx,
                                                    debug_txt_ref,
                                                    ggml_mul(ctx->ggml_ctx,
                                                             debug_txt_post_attn_ref,
                                                             txt_mod1.gate));
                ggml_set_name(debug_txt_after_attn_ref, (prefix + "_debug_txt_after_attn_ref").c_str());
                debug_compare_sequence(txt,
                                       debug_txt_after_attn_ref,
                                       prefix + "_inner_txt_after_attn");
            }

            ggml_tensor* txt_mlp_in = dit::modulate(ctx->ggml_ctx,
                                                    txt_norm2->forward(ctx, txt),
                                                    txt_mod2.shift,
                                                    txt_mod2.scale);
            ggml_set_name(txt_mlp_in, (prefix + "_txt_mlp_in").c_str());
            ggml_tensor* debug_txt_mlp_in_ref = nullptr;
            if (debug_inner) {
                debug_txt_mlp_in_ref = dit::modulate(ctx->ggml_ctx,
                                                     txt_norm2->forward(ctx, debug_txt_after_attn_ref),
                                                     txt_mod2.shift,
                                                     txt_mod2.scale);
                ggml_set_name(debug_txt_mlp_in_ref, (prefix + "_debug_txt_mlp_in_ref").c_str());
                debug_compare_sequence(txt_mlp_in,
                                       debug_txt_mlp_in_ref,
                                       prefix + "_inner_txt_mlp_in");
            }

            auto txt_mlp_out = txt_mlp->forward(ctx, txt_mlp_in);
            ggml_set_name(txt_mlp_out, (prefix + "_txt_mlp_out").c_str());
            ggml_tensor* debug_txt_mlp_out_ref = nullptr;
            if (debug_inner) {
                debug_txt_mlp_out_ref = txt_mlp->forward(ctx, debug_txt_mlp_in_ref);
                ggml_set_name(debug_txt_mlp_out_ref, (prefix + "_debug_txt_mlp_out_ref").c_str());
                debug_compare_sequence(txt_mlp_out,
                                       debug_txt_mlp_out_ref,
                                       prefix + "_inner_txt_mlp_out");
            }

            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_mod2.gate));
            ggml_set_name(txt, (prefix + "_txt_after_mlp").c_str());
            if (debug_inner) {
                ggml_tensor* debug_txt_after_mlp_ref = ggml_add(ctx->ggml_ctx,
                                                                debug_txt_after_attn_ref,
                                                                ggml_mul(ctx->ggml_ctx,
                                                                         debug_txt_mlp_out_ref,
                                                                         txt_mod2.gate));
                ggml_set_name(debug_txt_after_mlp_ref, (prefix + "_debug_txt_after_mlp_ref").c_str());
                debug_compare_sequence(txt,
                                       debug_txt_after_mlp_ref,
                                       prefix + "_inner_txt_after_mlp");
            }

            return {img, txt};
        }
#endif
    };

    struct SingleStreamBlock : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t hidden_size;
        int64_t mlp_hidden_dim;
        bool prune_mod;
        int idx = 0;
        bool use_yak_mlp;
        bool use_mlp_silu_act;
        int64_t mlp_mult_factor;

    public:
        SingleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio       = 4.0f,
                          int idx               = 0,
                          float qk_scale        = 0.f,
                          bool prune_mod        = false,
                          bool share_modulation = false,
                          bool mlp_proj_bias    = true,
                          bool use_yak_mlp      = false,
                          bool use_mlp_silu_act = false)
            : hidden_size(hidden_size), num_heads(num_heads), idx(idx), prune_mod(prune_mod), use_yak_mlp(use_yak_mlp), use_mlp_silu_act(use_mlp_silu_act) {
            int64_t head_dim = hidden_size / num_heads;
            float scale      = qk_scale;
            if (scale <= 0.f) {
                scale = 1 / sqrt((float)head_dim);
            }
            mlp_hidden_dim  = static_cast<int64_t>(hidden_size * mlp_ratio);
            mlp_mult_factor = 1;
            if (use_yak_mlp || use_mlp_silu_act) {
                mlp_mult_factor = 2;
            }

            blocks["linear1"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * mlp_mult_factor, mlp_proj_bias));
            blocks["linear2"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size + mlp_hidden_dim, hidden_size, mlp_proj_bias));
            blocks["norm"]     = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["pre_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            // mlp_act is nn.GELU(approximate="tanh")
            if (!prune_mod && !share_modulation) {
                blocks["modulation"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, false));
            }
        }

        ModulationOut get_distil_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            int64_t offset = 3 * idx;
            return ModulationOut(ctx, vec, offset);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* vec,
                             ggml_tensor* pe,
                             ggml_tensor* mask               = nullptr,
                             std::vector<ModulationOut> mods = {}) {
            // x: [N, n_token, hidden_size]
            // pe: [n_token, d_head/2, 2, 2]
            // return: [N, n_token, hidden_size]

            auto linear1  = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto linear2  = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);

            ModulationOut mod;
            if (!mods.empty()) {
                mod = mods[0];
            } else {
                if (prune_mod) {
                    mod = get_distil_mod(ctx, vec);
                } else {
                    auto modulation = std::dynamic_pointer_cast<Modulation>(blocks["modulation"]);

                    mod = modulation->forward(ctx, vec)[0];
                }
            }

            auto x_mod   = dit::modulate(ctx->ggml_ctx, pre_norm->forward(ctx, x), mod.shift, mod.scale);
            auto qkv_mlp = linear1->forward(ctx, x_mod);  // [N, n_token, hidden_size * 3 + mlp_hidden_dim*mlp_mult_factor]

            int64_t head_dim = hidden_size / num_heads;

            auto q = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], 0);
            auto k = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], (qkv_mlp->nb[0]) * hidden_size);
            auto v = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], (qkv_mlp->nb[0]) * 2 * hidden_size);

            q         = norm->query_norm(ctx, q);
            k         = norm->key_norm(ctx, k);
            auto attn = Rope::attention(ctx, q, k, v, pe, mask, 1.0f, true, true);  // [N, n_token, hidden_size]

            auto mlp = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, mlp_hidden_dim * mlp_mult_factor, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], hidden_size * 3 * qkv_mlp->nb[0]);
            if (use_yak_mlp) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp, false);
            } else if (use_mlp_silu_act) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp);
            } else {
                mlp = ggml_ext_gelu(ctx->ggml_ctx, mlp, true);
            }
            auto attn_mlp = ggml_concat(ctx->ggml_ctx, attn, mlp, 0);  // [N, n_token, hidden_size + mlp_hidden_dim]
            auto output   = linear2->forward(ctx, attn_mlp);           // [N, n_token, hidden_size]

            output = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, output, mod.gate));
            return output;
        }

#ifndef ED_DEBUG_SP_COMM
        ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* vec,
                                ggml_tensor* pe,
                                ggml_tensor* mask               = nullptr,
                                std::vector<ModulationOut> mods = {},
                                ggml_tensor* prepared_pe_seq_major = nullptr) {
            auto linear1  = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto linear2  = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);

            ModulationOut mod;
            if (!mods.empty()) {
                mod = mods[0];
            } else {
                if (prune_mod) {
                    mod = get_distil_mod(ctx, vec);
                } else {
                    auto modulation = std::dynamic_pointer_cast<Modulation>(blocks["modulation"]);

                    mod = modulation->forward(ctx, vec)[0];
                }
            }

            const int world_size     = flux_sp_world_size(ctx);
            const int64_t head_dim   = hidden_size / num_heads;
            const std::string prefix = "flux_single" + std::to_string(idx);

            auto x_mod   = flux_sp_modulate(ctx->ggml_ctx, pre_norm->forward(ctx, x), mod.shift, mod.scale);
            ggml_tensor* qkv_mlp = nullptr;
            ggml_tensor* mlp = nullptr;
            if (flux_sp_split_single_linear1_enabled()) {
                const int64_t qkv_dim = hidden_size * 3;
                qkv_mlp = linear1->forward_output_slice(ctx, x_mod, 0, qkv_dim);
                mlp = linear1->forward_output_slice(ctx,
                                                    x_mod,
                                                    qkv_dim,
                                                    mlp_hidden_dim * mlp_mult_factor);
            }
            if (qkv_mlp == nullptr || mlp == nullptr) {
                qkv_mlp = linear1->forward(ctx, x_mod);
                mlp = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, mlp_hidden_dim * mlp_mult_factor, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], hidden_size * 3 * qkv_mlp->nb[0]);
            }

            auto q = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], 0);
            auto k = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], qkv_mlp->nb[0] * hidden_size);
            auto v = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], qkv_mlp->nb[0] * 2 * hidden_size);

            const bool pre_qk_norm = flux_sp_pre_qk_norm_enabled();
            std::vector<edgedit::parallel::SPSeqToHeadOutputLayout> output_layouts = {
                flux_sp_qk_seq_major_enabled() ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajorRopeInterleaved :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
                flux_sp_qk_seq_major_enabled() ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajorRopeInterleaved :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
                flux_sp_flash_v_seq_major_enabled(ctx) ?
                    edgedit::parallel::SPSeqToHeadOutputLayout::SeqMajor :
                    edgedit::parallel::SPSeqToHeadOutputLayout::HeadMajor,
            };
            const bool can_fuse_qkv_recv_prep =
                flux_sp_fused_single_qkv_recv_prep_enabled() &&
                pre_qk_norm &&
                flux_sp_qk_seq_major_enabled() &&
                flux_sp_flash_v_seq_major_enabled(ctx) &&
                mask == nullptr;
            const bool use_mixed_qkv_send =
                can_fuse_qkv_recv_prep &&
                flux_sp_mixed_qkv_send_enabled() &&
                flux_sp_fused_qkv_send_pack_enabled();
            const bool use_f16_qkv_send =
                can_fuse_qkv_recv_prep &&
                flux_sp_f16_qkv_send_enabled() &&
                flux_sp_fused_qkv_send_pack_enabled();
            const bool f16_qk_norm =
                pre_qk_norm &&
                flux_sp_f16_qk_norm_enabled() &&
                use_f16_qkv_send &&
                !use_mixed_qkv_send;
            if (pre_qk_norm) {
                if (f16_qk_norm) {
                    q = norm->query_norm_f16(ctx, q);
                    k = norm->key_norm_f16(ctx, k);
                } else {
                    q = norm->query_norm(ctx, q);
                    k = norm->key_norm(ctx, k);
                }
            }
            auto qkv_head = use_f16_qkv_send ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_f16_recv_only(ctx->ggml_ctx,
                                                                                                   q,
                                                                                                   k,
                                                                                                   v,
                                                                                                   ctx->process_group,
                                                                                                   world_size,
                                                                                                   prefix + "_qkv_seq_to_head") :
                            use_mixed_qkv_send ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_mixed_recv_only(ctx->ggml_ctx,
                                                                                                     q,
                                                                                                     k,
                                                                                                     v,
                                                                                                     ctx->process_group,
                                                                                                     world_size,
                                                                                                     prefix + "_qkv_seq_to_head") :
                            flux_sp_fused_qkv_send_pack_enabled() ?
                                edgedit::parallel::sp_all_to_all_4d_qkv_seq_to_head_packed_layouts(ctx->ggml_ctx,
                                                                                                   q,
                                                                                                   k,
                                                                                                   v,
                                                                                                   output_layouts,
                                                                                                   ctx->process_group,
                                                                                                   world_size,
                                                                                                   prefix + "_qkv_seq_to_head") :
                                edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched_layouts(ctx->ggml_ctx,
                                                                                                {q, k, v},
                                                                                                output_layouts,
                                                                                                ctx->process_group,
                                                                                                world_size,
                                                                                                prefix + "_qkv_seq_to_head");
            if (!use_mixed_qkv_send && !use_f16_qkv_send) {
                GGML_ASSERT(qkv_head.outputs.size() == 3);
                q = pre_qk_norm ? qkv_head.outputs[0] : norm->query_norm(ctx, qkv_head.outputs[0]);
                k = pre_qk_norm ? qkv_head.outputs[1] : norm->key_norm(ctx, qkv_head.outputs[1]);
                v = qkv_head.outputs[2];
            }
            ggml_set_name(q, (prefix + "_q_attn").c_str());
            ggml_set_name(k, (prefix + "_k_attn").c_str());
            ggml_set_name(v, (prefix + "_v_attn").c_str());
            ggml_tensor* attn = nullptr;
            if (can_fuse_qkv_recv_prep && qkv_head.recv_flat != nullptr) {
                ggml_tensor* prepared_pe = prepared_pe_seq_major != nullptr ?
                                               prepared_pe_seq_major :
                                               flux_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                                 pe,
                                                                                 prefix + "_fused_recv_pe_seq_major");
                const bool mixed_qkv_recv = use_mixed_qkv_send &&
                                            qkv_head.total_head_dim == head_dim * 2;
                const ggml_type q_recv_prep_type = flux_sp_f16_q_attention_enabled() ? GGML_TYPE_F16 : GGML_TYPE_F32;
                ggml_tensor* fused_q = nullptr;
                ggml_tensor* fused_k = nullptr;
                ggml_tensor* fused_v = nullptr;
                if (flux_sp_bundle_single_qkv_recv_prep_enabled() &&
                    use_f16_qkv_send &&
                    !mixed_qkv_recv &&
                    q_recv_prep_type == GGML_TYPE_F16) {
                    ggml_tensor* bundle = edgedit::ggml_ext::flux_sp_qkv_recv_prep_bundle_custom(ctx->ggml_ctx,
                                                                                                  qkv_head.recv_flat,
                                                                                                  prepared_pe,
                                                                                                  world_size,
                                                                                                  num_heads,
                                                                                                  head_dim);
                    if (bundle != nullptr) {
                        const size_t plane_offset = static_cast<size_t>(bundle->nb[3]);
                        fused_q = ggml_view_3d(ctx->ggml_ctx,
                                               bundle,
                                               bundle->ne[0],
                                               bundle->ne[1],
                                               bundle->ne[2],
                                               bundle->nb[1],
                                               bundle->nb[2],
                                               0);
                        fused_k = ggml_view_3d(ctx->ggml_ctx,
                                               bundle,
                                               bundle->ne[0],
                                               bundle->ne[1],
                                               bundle->ne[2],
                                               bundle->nb[1],
                                               bundle->nb[2],
                                               plane_offset);
                        fused_v = ggml_view_3d(ctx->ggml_ctx,
                                               bundle,
                                               bundle->ne[0],
                                               bundle->ne[1],
                                               bundle->ne[2],
                                               bundle->nb[1],
                                               bundle->nb[2],
                                               2 * plane_offset);
                    }
                }
                if (fused_q == nullptr || fused_k == nullptr || fused_v == nullptr) {
                    fused_q = mixed_qkv_recv ?
                                  edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_custom(ctx->ggml_ctx,
                                                                                         qkv_head.recv_flat,
                                                                                         prepared_pe,
                                                                                         edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::Q,
                                                                                         world_size,
                                                                                         num_heads,
                                                                                         head_dim,
                                                                                         q_recv_prep_type) :
                                  edgedit::ggml_ext::flux_sp_qkv_recv_prep_custom(ctx->ggml_ctx,
                                                                                  qkv_head.recv_flat,
                                                                                  prepared_pe,
                                                                                  edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::Q,
                                                                                  world_size,
                                                                                  num_heads,
                                                                                  head_dim,
                                                                                  q_recv_prep_type);
                    fused_k = mixed_qkv_recv ?
                                  edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_custom(ctx->ggml_ctx,
                                                                                         qkv_head.recv_flat,
                                                                                         prepared_pe,
                                                                                         edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::K,
                                                                                         world_size,
                                                                                         num_heads,
                                                                                         head_dim) :
                                  edgedit::ggml_ext::flux_sp_qkv_recv_prep_custom(ctx->ggml_ctx,
                                                                                  qkv_head.recv_flat,
                                                                                  prepared_pe,
                                                                                  edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::K,
                                                                                  world_size,
                                                                                  num_heads,
                                                                                  head_dim);
                    fused_v = mixed_qkv_recv ?
                                  edgedit::ggml_ext::flux_sp_qkv_mixed_recv_prep_custom(ctx->ggml_ctx,
                                                                                         qkv_head.recv_flat,
                                                                                         prepared_pe,
                                                                                         edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::V,
                                                                                         world_size,
                                                                                         num_heads,
                                                                                         head_dim) :
                                  edgedit::ggml_ext::flux_sp_qkv_recv_prep_custom(ctx->ggml_ctx,
                                                                                  qkv_head.recv_flat,
                                                                                  prepared_pe,
                                                                                  edgedit::ggml_ext::FluxSPQKVRecvPrepPlane::V,
                                                                                  world_size,
                                                                                  num_heads,
                                                                                  head_dim);
                }
                if (fused_q != nullptr && fused_k != nullptr && fused_v != nullptr) {
                    ggml_set_name(fused_q, (prefix + "_q_recv_rope").c_str());
                    ggml_set_name(fused_k, (prefix + "_k_recv_rope").c_str());
                    ggml_set_name(fused_v, (prefix + "_v_recv_prep").c_str());
                    attn = flux_sp_attention_prepared_qk(ctx, fused_q, fused_k, fused_v, mask, prefix);
                }
            }
            GGML_ASSERT((!use_mixed_qkv_send && !use_f16_qkv_send) || attn != nullptr);
            if (attn == nullptr) {
                attn = flux_sp_qk_seq_major_enabled() ?
                           flux_sp_attention_from_rope_work_layout(ctx, q, k, v, pe, mask, head_dim, prefix, prepared_pe_seq_major) :
                           flux_sp_attention(ctx, q, k, v, pe, mask, prefix, prepared_pe_seq_major);
            }
            if (flux_sp_strict_barrier_enabled()) {
                sd::ggml_graph_cut::mark_graph_cut(attn, prefix + ".sp_attention", "attn");
            }

            auto attn_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                           attn,
                                           head_dim,
                                           num_heads / world_size,
                                           attn->ne[1],
                                           attn->ne[2]);
            ggml_set_name(attn_4d, (prefix + "_attn_4d").c_str());

            const bool keep_single_attn_bf16 = flux_sp_bf16_single_head_to_seq_output_enabled() &&
                                               attn_4d->type == GGML_TYPE_F16;
            const bool keep_single_attn_f16 = !keep_single_attn_bf16 &&
                                              (flux_sp_f16_single_head_to_seq_output_enabled() ||
                                               attn_4d->type == GGML_TYPE_F16);
            auto attn_local = keep_single_attn_bf16 ?
                                  edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_bf16_output(ctx->ggml_ctx,
                                                                                                     attn_4d,
                                                                                                     ctx->process_group,
                                                                                                     world_size,
                                                                                                     prefix + "_attn_head_to_seq") :
                              keep_single_attn_f16 ?
                                  edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_f16_output(ctx->ggml_ctx,
                                                                                                    attn_4d,
                                                                                                    ctx->process_group,
                                                                                                    world_size,
                                                                                                    prefix + "_attn_head_to_seq") :
                              flux_sp_f16_single_head_to_seq_enabled() ?
                                  edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed_f16(ctx->ggml_ctx,
                                                                                             attn_4d,
                                                                                             ctx->process_group,
                                                                                             world_size,
                                                                                             prefix + "_attn_head_to_seq") :
                              flux_sp_fused_single_head_to_seq_enabled() ?
                                  edgedit::parallel::sp_all_to_all_4d_head_to_seq_packed(ctx->ggml_ctx,
                                                                                         attn_4d,
                                                                                         ctx->process_group,
                                                                                         world_size,
                                                                                         prefix + "_attn_head_to_seq") :
                                  edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx->ggml_ctx,
                                                                                  attn_4d,
                                                                                  ctx->process_group,
                                                                                  world_size,
                                                                                  prefix + "_attn_head_to_seq");

            auto attn_flat = ggml_reshape_3d(ctx->ggml_ctx,
                                             attn_local.output,
                                             hidden_size,
                                             attn_local.output->ne[2],
                                             attn_local.output->ne[3]);
            ggml_set_name(attn_flat, (prefix + "_attn_flat").c_str());

            if (use_yak_mlp) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp, false);
            } else if (use_mlp_silu_act) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp);
            } else {
                if (flux_sp_mlp_gelu_bf16_enabled()) {
                    if (auto mlp_bf16 = edgedit::ggml_ext::flux_sp_gelu_bf16_custom(ctx->ggml_ctx, mlp)) {
                        mlp = mlp_bf16;
                        ggml_set_name(mlp, (prefix + "_mlp_gelu_bf16").c_str());
                    } else {
                        mlp = ggml_ext_gelu(ctx->ggml_ctx, mlp, true);
                    }
                } else {
                    mlp = ggml_ext_gelu(ctx->ggml_ctx, mlp, true);
                }
            }
            ggml_tensor* output = nullptr;
            bool used_fused_residual_gate = false;
            ggml_tensor* gate_for_concat_linear = mod.gate;
            if (gate_for_concat_linear != nullptr &&
                gate_for_concat_linear->ne[2] == 1 &&
                gate_for_concat_linear->ne[3] == 1) {
                gate_for_concat_linear = ggml_reshape_3d(ctx->ggml_ctx,
                                                         gate_for_concat_linear,
                                                         gate_for_concat_linear->ne[0],
                                                         1,
                                                         gate_for_concat_linear->ne[1]);
            }
            if (flux_sp_fused_modulation_enabled() &&
                flux_sp_fused_single_linear2_enabled() &&
                flux_sp_fused_single_linear2_residual_gate_enabled()) {
                output = linear2->forward_input_concat_residual_gate_fused(ctx, x, attn_flat, mlp, gate_for_concat_linear);
                used_fused_residual_gate = output != nullptr;
            }
            if (output == nullptr && flux_sp_fused_single_linear2_enabled()) {
                output = linear2->forward_input_concat_fused(ctx, attn_flat, mlp);
            }
            if (output == nullptr && attn_flat->type != GGML_TYPE_F32) {
                attn_flat = ggml_cast(ctx->ggml_ctx, attn_flat, GGML_TYPE_F32);
                ggml_set_name(attn_flat, (prefix + "_attn_flat_f32").c_str());
            }
            if (output == nullptr && mlp->type != GGML_TYPE_F32) {
                mlp = ggml_cast(ctx->ggml_ctx, mlp, GGML_TYPE_F32);
                ggml_set_name(mlp, (prefix + "_mlp_f32").c_str());
            }
            if (output == nullptr && flux_sp_split_single_linear2_enabled()) {
                output = linear2->forward_input_concat_split(ctx, attn_flat, mlp);
            }
            if (output == nullptr) {
                auto attn_mlp = ggml_concat(ctx->ggml_ctx, attn_flat, mlp, 0);
                ggml_set_name(attn_mlp, (prefix + "_attn_mlp").c_str());
                output = linear2->forward(ctx, attn_mlp);
            }

            if (!used_fused_residual_gate) {
                output = flux_sp_residual_gate(ctx->ggml_ctx, x, output, mod.gate);
            }
            return output;
        }
#endif

#ifdef ED_DEBUG_SP_COMM
        ggml_tensor* forward_sp(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* vec,
                                ggml_tensor* pe,
                                ggml_tensor* mask               = nullptr,
                                std::vector<ModulationOut> mods = {}) {
            auto linear1  = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto linear2  = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);

            ModulationOut mod;
            if (!mods.empty()) {
                mod = mods[0];
            } else {
                if (prune_mod) {
                    mod = get_distil_mod(ctx, vec);
                } else {
                    auto modulation = std::dynamic_pointer_cast<Modulation>(blocks["modulation"]);

                    mod = modulation->forward(ctx, vec)[0];
                }
            }

            const int world_size   = flux_sp_world_size(ctx);
            const int64_t head_dim = hidden_size / num_heads;
            const std::string prefix = "flux_single" + std::to_string(idx);

            auto x_mod   = dit::modulate(ctx->ggml_ctx, pre_norm->forward(ctx, x), mod.shift, mod.scale);
            auto qkv_mlp = linear1->forward(ctx, x_mod);

            auto q = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], 0);
            auto k = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], qkv_mlp->nb[0] * hidden_size);
            auto v = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                  qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], qkv_mlp->nb[0] * 2 * hidden_size);

            auto qkv_head = edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched(ctx->ggml_ctx,
                                                                                    {q, k, v},
                                                                                    ctx->process_group,
                                                                                    world_size,
                                                                                    prefix + "_qkv_seq_to_head");
            GGML_ASSERT(qkv_head.outputs.size() == 3);

            q         = norm->query_norm(ctx, qkv_head.outputs[0]);
            k         = norm->key_norm(ctx, qkv_head.outputs[1]);
            v         = qkv_head.outputs[2];
            q         = ggml_cont(ctx->ggml_ctx, q);
            ggml_set_name(q, (prefix + "_q_attn").c_str());
            k = ggml_cont(ctx->ggml_ctx, k);
            ggml_set_name(k, (prefix + "_k_attn").c_str());
            v = ggml_cont(ctx->ggml_ctx, v);
            ggml_set_name(v, (prefix + "_v_attn").c_str());
            auto attn = flux_sp_attention(ctx, q, k, v, pe, mask, prefix);
            if (flux_sp_strict_barrier_enabled()) {
                sd::ggml_graph_cut::mark_graph_cut(attn, prefix + ".sp_attention", "attn");
            }

            auto attn_4d = ggml_reshape_4d(ctx->ggml_ctx,
                                           attn,
                                           head_dim,
                                           num_heads / world_size,
                                           attn->ne[1],
                                           attn->ne[2]);
            ggml_set_name(attn_4d, (prefix + "_attn_4d").c_str());

            auto attn_local = edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx->ggml_ctx,
                                                                              attn_4d,
                                                                              ctx->process_group,
                                                                              world_size,
                                                                              prefix + "_attn_head_to_seq");

            auto attn_flat = ggml_reshape_3d(ctx->ggml_ctx,
                                             attn_local.output,
                                             hidden_size,
                                             attn_local.output->ne[2],
                                             attn_local.output->ne[3]);
            ggml_set_name(attn_flat, (prefix + "_attn_flat").c_str());

            auto mlp = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, mlp_hidden_dim * mlp_mult_factor, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], hidden_size * 3 * qkv_mlp->nb[0]);
            if (use_yak_mlp) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp, false);
            } else if (use_mlp_silu_act) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp);
            } else {
                mlp = ggml_ext_gelu(ctx->ggml_ctx, mlp, true);
            }
            auto attn_mlp = ggml_concat(ctx->ggml_ctx, attn_flat, mlp, 0);
            ggml_set_name(attn_mlp, (prefix + "_attn_mlp").c_str());
            auto output = linear2->forward(ctx, attn_mlp);

            output = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, output, mod.gate));
            return output;
        }
#endif
    };

    struct LastLayer : public GGMLBlock {
        bool prune_mod;

    public:
        LastLayer(int64_t hidden_size,
                  int64_t patch_size,
                  int64_t out_channels,
                  bool prune_mod = false,
                  bool bias      = true)
            : prune_mod(prune_mod) {
            blocks["norm_final"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            blocks["linear"]     = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels, bias));
            if (!prune_mod) {
                blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size, bias));
            }
        }

        ModulationOut get_distil_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            int64_t offset = vec->ne[2] - 2;
            int64_t stride = vec->nb[1] * vec->ne[1];
            auto shift     = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            auto scale     = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            // No gate
            return {shift, scale, nullptr};
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            ggml_tensor *shift, *scale;
            if (prune_mod) {
                auto mod = get_distil_mod(ctx, c);
                shift    = mod.shift;
                scale    = mod.scale;
            } else {
                auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

                auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, 2 * hidden_size]
                auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, 2, 0);
                shift      = m_vec[0];  // [N, hidden_size]
                scale      = m_vec[1];  // [N, hidden_size]
            }

            x = dit::modulate(ctx->ggml_ctx, norm_final->forward(ctx, x), shift, scale);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct ChromaApproximator : public GGMLBlock {
        int64_t inner_size = 5120;
        int64_t n_layers   = 5;
        ChromaApproximator(int64_t in_channels = 64, int64_t hidden_size = 3072) {
            blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_size, true));
            for (int i = 0; i < n_layers; i++) {
                blocks["norms." + std::to_string(i)]  = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_size));
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(inner_size, inner_size));
            }
            blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(inner_size, hidden_size, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto in_proj  = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            x = in_proj->forward(ctx, x);
            for (int i = 0; i < n_layers; i++) {
                auto norm  = std::dynamic_pointer_cast<RMSNorm>(blocks["norms." + std::to_string(i)]);
                auto embed = std::dynamic_pointer_cast<MLPEmbedder>(blocks["layers." + std::to_string(i)]);
                x          = ggml_add_inplace(ctx->ggml_ctx, x, embed->forward(ctx, norm->forward(ctx, x)));
            }
            x = out_proj->forward(ctx, x);

            return x;
        }
    };

    struct NerfEmbedder : public GGMLBlock {
        NerfEmbedder(int64_t in_channels,
                     int64_t hidden_size_input,
                     int64_t max_freqs) {
            blocks["embedder.0"] = std::make_shared<Linear>(in_channels + max_freqs * max_freqs, hidden_size_input);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* dct) {
            // x: (B, P^2, C)
            // dct: (1, P^2, max_freqs^2)
            // return: (B, P^2, hidden_size_input)
            auto embedder = std::dynamic_pointer_cast<Linear>(blocks["embedder.0"]);

            dct = ggml_repeat_4d(ctx->ggml_ctx, dct, dct->ne[0], dct->ne[1], x->ne[2], x->ne[3]);
            x   = ggml_concat(ctx->ggml_ctx, x, dct, 0);
            x   = embedder->forward(ctx, x);

            return x;
        }
    };

    struct NerfGLUBlock : public GGMLBlock {
        int64_t mlp_ratio;
        NerfGLUBlock(int64_t hidden_size_s,
                     int64_t hidden_size_x,
                     int64_t mlp_ratio)
            : mlp_ratio(mlp_ratio) {
            int64_t total_params      = 3 * hidden_size_x * hidden_size_x * mlp_ratio;
            blocks["param_generator"] = std::make_shared<Linear>(hidden_size_s, total_params);
            blocks["norm"]            = std::make_shared<RMSNorm>(hidden_size_x);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* s) {
            // x: (batch_size, n_token, hidden_size_x)
            // s: (batch_size, hidden_size_s)
            // return: (batch_size, n_token, hidden_size_x)
            auto param_generator = std::dynamic_pointer_cast<Linear>(blocks["param_generator"]);
            auto norm            = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            int64_t batch_size    = x->ne[2];
            int64_t hidden_size_x = x->ne[0];

            auto mlp_params = param_generator->forward(ctx, s);
            auto fc_params  = ggml_ext_chunk(ctx->ggml_ctx, mlp_params, 3, 0);
            auto fc1_gate   = ggml_reshape_3d(ctx->ggml_ctx, fc_params[0], hidden_size_x * mlp_ratio, hidden_size_x, batch_size);
            auto fc1_value  = ggml_reshape_3d(ctx->ggml_ctx, fc_params[1], hidden_size_x * mlp_ratio, hidden_size_x, batch_size);
            auto fc2        = ggml_reshape_3d(ctx->ggml_ctx, fc_params[2], hidden_size_x, mlp_ratio * hidden_size_x, batch_size);

            fc1_gate  = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc1_gate, 1, 0, 2, 3));  // [batch_size, hidden_size_x*mlp_ratio, hidden_size_x]
            fc1_gate  = ggml_l2_norm(ctx->ggml_ctx, fc1_gate, 1e-12f);
            fc1_value = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc1_value, 1, 0, 2, 3));  // [batch_size, hidden_size_x*mlp_ratio, hidden_size_x]
            fc1_value = ggml_l2_norm(ctx->ggml_ctx, fc1_value, 1e-12f);
            fc2       = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc2, 1, 0, 2, 3));  // [batch_size, hidden_size_x, hidden_size_x*mlp_ratio]
            fc2       = ggml_l2_norm(ctx->ggml_ctx, fc2, 1e-12f);

            auto res_x = x;
            x          = norm->forward(ctx, x);  // [batch_size, n_token, hidden_size_x]

            auto x1 = ggml_mul_mat(ctx->ggml_ctx, fc1_gate, x);  // [batch_size, n_token, hidden_size_x*mlp_ratio]
            x1      = ggml_silu_inplace(ctx->ggml_ctx, x1);

            auto x2 = ggml_mul_mat(ctx->ggml_ctx, fc1_value, x);  // [batch_size, n_token, hidden_size_x*mlp_ratio]

            x = ggml_mul_inplace(ctx->ggml_ctx, x1, x2);  // [batch_size, n_token, hidden_size_x*mlp_ratio]

            x = ggml_mul_mat(ctx->ggml_ctx, fc2, x);  // [batch_size, n_token, hidden_size_x]

            x = ggml_add_inplace(ctx->ggml_ctx, x, res_x);

            return x;
        }
    };

    struct NerfFinalLayer : public GGMLBlock {
        NerfFinalLayer(int64_t hidden_size,
                       int64_t out_channels) {
            blocks["norm"]   = std::make_shared<RMSNorm>(hidden_size);
            blocks["linear"] = std::make_shared<Linear>(hidden_size, out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x) {
            auto norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            x = norm->forward(ctx, x);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct NerfFinalLayerConv : public GGMLBlock {
        NerfFinalLayerConv(int64_t hidden_size,
                           int64_t out_channels) {
            blocks["norm"] = std::make_shared<RMSNorm>(hidden_size);
            blocks["conv"] = std::make_shared<Conv2d>(hidden_size, out_channels, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x) {
            // x: [N, C, H, W]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [N, H, W, C]
            x = norm->forward(ctx, x);
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));  // [N, C, H, W]
            x = conv->forward(ctx, x);

            return x;
        }
    };

    struct ChromaRadianceParams {
        int64_t nerf_hidden_size = 64;
        int nerf_mlp_ratio       = 4;
        int nerf_depth           = 4;
        int nerf_max_freqs       = 8;
        bool use_x0              = false;
        bool fake_patch_size_x2  = false;
    };

    struct FluxParams {
        SDVersion version         = VERSION_FLUX;
        bool is_chroma            = false;
        int patch_size            = 2;
        int64_t in_channels       = 64;
        int64_t out_channels      = 64;
        int64_t vec_in_dim        = 768;
        int64_t context_in_dim    = 4096;
        int64_t hidden_size       = 3072;
        float mlp_ratio           = 4.0f;
        int num_heads             = 24;
        int depth                 = 19;
        int depth_single_blocks   = 38;
        std::vector<int> axes_dim = {16, 56, 56};
        int axes_dim_sum          = 128;
        int theta                 = 10000;
        bool qkv_bias             = true;
        bool guidance_embed       = true;
        int64_t in_dim            = 64;
        bool disable_bias         = false;
        bool share_modulation     = false;
        bool semantic_txt_norm    = false;
        bool use_yak_mlp          = false;
        bool use_mlp_silu_act     = false;
        float ref_index_scale     = 1.f;
        ChromaRadianceParams chroma_radiance_params;
    };

    struct Flux : public GGMLBlock {
    public:
        FluxParams params;
        Flux() {}
        Flux(FluxParams params)
            : params(params) {
            if (params.version == VERSION_CHROMA_RADIANCE) {
                std::pair<int, int> kernel_size = {params.patch_size, params.patch_size};
                if (params.chroma_radiance_params.fake_patch_size_x2) {
                    kernel_size = {params.patch_size / 2, params.patch_size / 2};
                }
                std::pair<int, int> stride = kernel_size;

                blocks["img_in_patch"] = std::make_shared<Conv2d>(params.in_channels,
                                                                  params.hidden_size,
                                                                  kernel_size,
                                                                  stride);
            } else {
                blocks["img_in"] = std::make_shared<Linear>(params.in_channels, params.hidden_size, !params.disable_bias);
            }
            if (params.is_chroma) {
                blocks["distilled_guidance_layer"] = std::make_shared<ChromaApproximator>(params.in_dim, params.hidden_size);
            } else {
                blocks["time_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size, !params.disable_bias);
                if (params.vec_in_dim > 0) {
                    blocks["vector_in"] = std::make_shared<MLPEmbedder>(params.vec_in_dim, params.hidden_size, !params.disable_bias);
                }
                if (params.guidance_embed) {
                    blocks["guidance_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size, !params.disable_bias);
                }
            }
            if (params.semantic_txt_norm) {
                blocks["txt_norm"] = std::make_shared<RMSNorm>(params.context_in_dim);
            }
            blocks["txt_in"] = std::make_shared<Linear>(params.context_in_dim, params.hidden_size, !params.disable_bias);

            for (int i = 0; i < params.depth; i++) {
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   params.qkv_bias,
                                                                                                   params.is_chroma,
                                                                                                   params.share_modulation,
                                                                                                   !params.disable_bias,
                                                                                                   params.use_yak_mlp,
                                                                                                   params.use_mlp_silu_act);
            }

            for (int i = 0; i < params.depth_single_blocks; i++) {
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<SingleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   0.f,
                                                                                                   params.is_chroma,
                                                                                                   params.share_modulation,
                                                                                                   !params.disable_bias,
                                                                                                   params.use_yak_mlp,
                                                                                                   params.use_mlp_silu_act);
            }

            if (params.version == VERSION_CHROMA_RADIANCE) {
                blocks["nerf_image_embedder"] = std::make_shared<NerfEmbedder>(params.in_channels,
                                                                               params.chroma_radiance_params.nerf_hidden_size,
                                                                               params.chroma_radiance_params.nerf_max_freqs);

                for (int i = 0; i < params.chroma_radiance_params.nerf_depth; i++) {
                    blocks["nerf_blocks." + std::to_string(i)] = std::make_shared<NerfGLUBlock>(params.hidden_size,
                                                                                                params.chroma_radiance_params.nerf_hidden_size,
                                                                                                params.chroma_radiance_params.nerf_mlp_ratio);
                }

                blocks["nerf_final_layer_conv"] = std::make_shared<NerfFinalLayerConv>(params.chroma_radiance_params.nerf_hidden_size,
                                                                                       params.in_channels);

            } else {
                blocks["final_layer"] = std::make_shared<LastLayer>(params.hidden_size, 1, params.out_channels, params.is_chroma, !params.disable_bias);
            }

            if (params.share_modulation) {
                blocks["double_stream_modulation_img"] = std::make_shared<Modulation>(params.hidden_size, true, !params.disable_bias);
                blocks["double_stream_modulation_txt"] = std::make_shared<Modulation>(params.hidden_size, true, !params.disable_bias);
                blocks["single_stream_modulation"]     = std::make_shared<Modulation>(params.hidden_size, false, !params.disable_bias);
            }
        }

        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* img,
                                  ggml_tensor* txt,
                                  ggml_tensor* timesteps,
                                  ggml_tensor* y,
                                  ggml_tensor* guidance,
                                  ggml_tensor* pe,
                                  ggml_tensor* mod_index_arange = nullptr,
                                  std::vector<int> skip_layers  = {}) {
            auto img_in      = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in      = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto final_layer = std::dynamic_pointer_cast<LastLayer>(blocks["final_layer"]);

            if (img_in) {
                img = img_in->forward(ctx, img);
            }

            ggml_tensor* vec;
            ggml_tensor* txt_img_mask = nullptr;
            if (params.is_chroma) {
                int64_t mod_index_length = 344;
                auto approx              = std::dynamic_pointer_cast<ChromaApproximator>(blocks["distilled_guidance_layer"]);
                auto distill_timestep    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 16, 10000, 1000.f);
                auto distill_guidance    = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 16, 10000, 1000.f);

                // auto mod_index_arange  = ggml_arange(ctx, 0, (float)mod_index_length, 1);
                // ggml_arange tot working on a lot of backends, precomputing it on CPU instead
                GGML_ASSERT(mod_index_arange != nullptr);
                auto modulation_index = ggml_ext_timestep_embedding(ctx->ggml_ctx, mod_index_arange, 32, 10000, 1000.f);  // [1, 344, 32]

                // Batch broadcast (will it ever be useful)
                modulation_index = ggml_repeat(ctx->ggml_ctx, modulation_index, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, modulation_index->ne[0], modulation_index->ne[1], img->ne[2]));  // [N, 344, 32]

                auto timestep_guidance = ggml_concat(ctx->ggml_ctx, distill_timestep, distill_guidance, 0);  // [N, 1, 32]
                timestep_guidance      = ggml_repeat(ctx->ggml_ctx, timestep_guidance, modulation_index);    // [N, 344, 32]

                vec = ggml_concat(ctx->ggml_ctx, timestep_guidance, modulation_index, 0);  // [N, 344, 64]
                // Permute for consistency with non-distilled modulation implementation
                vec = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, vec, 0, 2, 1, 3));  // [344, N, 64]
                vec = approx->forward(ctx, vec);                                               // [344, N, hidden_size]

                if (y != nullptr) {
                    txt_img_mask = ggml_pad(ctx->ggml_ctx, y, static_cast<int>(img->ne[1]), 0, 0, 0);
                }
            } else {
                auto time_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["time_in"]);
                vec          = time_in->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1000.f));
                if (params.guidance_embed) {
                    GGML_ASSERT(guidance != nullptr);
                    auto guidance_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["guidance_in"]);
                    // bf16 and fp16 result is different
                    auto g_in = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 256, 10000, 1000.f);
                    vec       = ggml_add(ctx->ggml_ctx, vec, guidance_in->forward(ctx, g_in));
                }

                if (params.vec_in_dim > 0) {
                    auto vector_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["vector_in"]);
                    vec            = ggml_add(ctx->ggml_ctx, vec, vector_in->forward(ctx, y));
                }
            }

            std::vector<ModulationOut> ds_img_mods;
            std::vector<ModulationOut> ds_txt_mods;
            std::vector<ModulationOut> ss_mods;
            if (params.share_modulation) {
                auto double_stream_modulation_img = std::dynamic_pointer_cast<Modulation>(blocks["double_stream_modulation_img"]);
                auto double_stream_modulation_txt = std::dynamic_pointer_cast<Modulation>(blocks["double_stream_modulation_txt"]);
                auto single_stream_modulation     = std::dynamic_pointer_cast<Modulation>(blocks["single_stream_modulation"]);

                ds_img_mods = double_stream_modulation_img->forward(ctx, vec);
                ds_txt_mods = double_stream_modulation_txt->forward(ctx, vec);
                ss_mods     = single_stream_modulation->forward(ctx, vec);
            }

            if (params.semantic_txt_norm) {
                auto semantic_txt_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);

                txt = semantic_txt_norm->forward(ctx, txt);
            }

            txt = txt_in->forward(ctx, txt);
            const int64_t flux_full_img_seq = img->ne[1];
            const int64_t flux_full_txt_seq = txt->ne[1];
            // Cache seam: the block stack transforms the image stream `img`. The
            // cached quantity is that stream's residual.
            // Tap-driven whole-stack inject (substep reuse): skip both block loops and
            // reconstruct x_before + residual via the registry.
            const bool tap_whole_inject = ctx->tap_registry != nullptr &&
                                          ctx->tap_registry->has_stream_override();
            const bool whole_inject = tap_whole_inject;
            ggml_tensor* cache_img_before = img;
            // Substep tap: block-stack input anchor (ModelIn). Conditional no-op
            // unless the middle layer requested it this substep.
            tap(ctx, edgedit::cache::AnchorRef::model_in(), cache_img_before);
            bool use_sp_mainline = flux_sp_enabled(ctx);
#ifdef ED_DEBUG_SP_COMM
            use_sp_mainline = use_sp_mainline && !debug_sp_capture_enabled();
            const std::string debug_compare_stage = debug_sp_mainline_compare_stage();
            const bool debug_compare_double_block = debug_compare_stage == "double_block" ||
                                                    debug_compare_stage == "double_block_end";
            const bool debug_compare_double_inner = debug_compare_stage == "double_inner" ||
                                                    debug_compare_stage.rfind("double_inner_", 0) == 0;
            const bool debug_compare_double_end   = debug_compare_stage == "double_end" ||
                                                    debug_compare_stage == "double";
            const bool debug_compare_single_entry = debug_compare_stage == "single_entry";
            const bool debug_compare_single_end   = debug_compare_stage == "single_end";
            const bool debug_compare_final_img    = debug_compare_stage == "final_img" ||
                                                    debug_compare_stage == "final_img_before_layer";
            int debug_compare_double_block_idx = 0;
            if (debug_compare_double_block || debug_compare_double_inner) {
                const char* compare_double_block_env = std::getenv("ED_FLUX_SP_COMPARE_DOUBLE_BLOCK");
                if (compare_double_block_env != nullptr && compare_double_block_env[0] != '\0') {
                    debug_compare_double_block_idx = std::atoi(compare_double_block_env);
                }
                if (debug_compare_double_block_idx < 0 ||
                    debug_compare_double_block_idx >= params.depth) {
                    LOG_WARN("flux SP mainline compare double block index out of range: idx=%d depth=%d; clamping",
                             debug_compare_double_block_idx,
                             params.depth);
                    debug_compare_double_block_idx = std::max(0,
                                                              std::min(debug_compare_double_block_idx,
                                                                       params.depth - 1));
                }
            }
            const bool debug_compare_mainline     = use_sp_mainline &&
                                                    (debug_compare_double_block ||
                                                     debug_compare_double_inner ||
                                                     debug_compare_double_end ||
                                                     debug_compare_single_entry ||
                                                     debug_compare_single_end ||
                                                     debug_compare_final_img);
            if (use_sp_mainline &&
                !debug_compare_stage.empty() &&
                !debug_compare_mainline) {
                LOG_WARN("flux SP mainline compare unknown stage: %s",
                         debug_compare_stage.c_str());
            }

            ggml_tensor* debug_img_after_double_ref        = nullptr;
            ggml_tensor* debug_txt_after_double_ref        = nullptr;
            ggml_tensor* debug_img_double_inner_ref        = nullptr;
            ggml_tensor* debug_txt_double_inner_ref        = nullptr;
            ggml_tensor* debug_txt_img_before_single_ref   = nullptr;
            ggml_tensor* debug_txt_img_after_single_ref    = nullptr;
            ggml_tensor* debug_img_before_final_layer_ref  = nullptr;

            auto debug_run_double_no_sp = [&](ggml_tensor* img_ref,
	                                              ggml_tensor* txt_ref,
	                                              int stop_block) -> std::pair<ggml_tensor*, ggml_tensor*> {
                if (stop_block < 0) {
                    return {img_ref, txt_ref};
                }
                for (int i = 0; i < params.depth; i++) {
                    if (skip_layers.size() > 0 &&
                        std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                        if (i >= stop_block) {
                            break;
                        }
                        continue;
                    }

                    auto block = std::dynamic_pointer_cast<DoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);
                    auto img_txt_ref = block->forward(ctx,
                                                      img_ref,
                                                      txt_ref,
                                                      vec,
                                                      pe,
                                                      txt_img_mask,
                                                      ds_img_mods,
                                                      ds_txt_mods);
                    img_ref = img_txt_ref.first;
                    txt_ref = img_txt_ref.second;
                    if (i >= stop_block) {
                        break;
                    }
                }
                return {img_ref, txt_ref};
            };

            auto debug_run_single_no_sp = [&](ggml_tensor* txt_img_ref) -> ggml_tensor* {
                for (int i = 0; i < params.depth_single_blocks; i++) {
                    if (skip_layers.size() > 0 &&
                        std::find(skip_layers.begin(), skip_layers.end(), i + params.depth) != skip_layers.end()) {
                        continue;
                    }

                    auto block = std::dynamic_pointer_cast<SingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);
                    txt_img_ref = block->forward(ctx,
                                                 txt_img_ref,
                                                 vec,
                                                 pe,
                                                 txt_img_mask,
                                                 ss_mods);
                }
                return txt_img_ref;
            };

            if (debug_compare_mainline) {
                const int debug_double_ref_stop_block = (debug_compare_double_block ||
                                                         debug_compare_double_inner) ?
                                                            debug_compare_double_block_idx :
                                                            params.depth - 1;
                if (debug_compare_double_inner) {
                    auto double_inner_ref = debug_run_double_no_sp(img,
                                                                   txt,
                                                                   debug_compare_double_block_idx - 1);
                    debug_img_double_inner_ref = double_inner_ref.first;
                    debug_txt_double_inner_ref = double_inner_ref.second;
                }

                if (!debug_compare_double_inner) {
                    auto double_ref = debug_run_double_no_sp(img, txt, debug_double_ref_stop_block);
                    debug_img_after_double_ref = double_ref.first;
                    debug_txt_after_double_ref = double_ref.second;
                } else {
                    debug_img_after_double_ref = debug_img_double_inner_ref;
                    debug_txt_after_double_ref = debug_txt_double_inner_ref;
                }

                if (debug_compare_single_entry ||
                    debug_compare_single_end ||
                    debug_compare_final_img) {
                    debug_txt_img_before_single_ref = ggml_concat(ctx->ggml_ctx,
                                                                  debug_txt_after_double_ref,
                                                                  debug_img_after_double_ref,
                                                                  1);
                }
                if (debug_compare_single_end ||
                    debug_compare_final_img) {
                    debug_txt_img_after_single_ref = debug_run_single_no_sp(debug_txt_img_before_single_ref);
                }
                if (debug_compare_final_img) {
                    debug_img_before_final_layer_ref = ggml_view_3d(ctx->ggml_ctx,
                                                                    debug_txt_img_after_single_ref,
                                                                    debug_txt_img_after_single_ref->ne[0],
                                                                    debug_img_after_double_ref->ne[1],
                                                                    debug_txt_img_after_single_ref->ne[2],
                                                                    debug_txt_img_after_single_ref->nb[1],
                                                                    debug_txt_img_after_single_ref->nb[2],
                                                                    debug_txt_after_double_ref->ne[1] *
                                                                        debug_txt_img_after_single_ref->nb[1]);
                    ggml_set_name(debug_img_before_final_layer_ref,
                                  "flux_debug_ref_img_before_final_layer");
                }
            }
#endif

            edgedit::parallel::SPSequenceSplit img_sp_split;
            edgedit::parallel::SPSequenceSplit txt_sp_split;
            edgedit::parallel::SPSequenceSplit txt_img_sp_split;
            if (use_sp_mainline) {
                const int rank       = flux_sp_rank(ctx);
                const int world_size = flux_sp_world_size(ctx);
                const int64_t img_pad = edgedit::parallel::sp_sequence_padding(img->ne[1],
                                                                                world_size);
                const int64_t txt_pad = edgedit::parallel::sp_sequence_padding(txt->ne[1],
                                                                                world_size);
                if (flux_profile_enabled() && flux_profile_should_log_rank(ctx)) {
                    LOG_INFO("flux SP profile double split rank=%d/%d img_seq=%" PRId64 " img_pad=%" PRId64 " txt_seq=%" PRId64 " txt_pad=%" PRId64,
                             rank,
                             world_size,
                             img->ne[1],
                             img_pad,
                             txt->ne[1],
                             txt_pad);
                }
                if (img_pad != 0 || txt_pad != 0) {
                    LOG_WARN("flux SP mainline disabled for padded double-stream sequence: rank=%d world_size=%d img_seq=%" PRId64 " img_pad=%" PRId64 " txt_seq=%" PRId64 " txt_pad=%" PRId64,
                             rank,
                             world_size,
                             img->ne[1],
                             img_pad,
                             txt->ne[1],
                             txt_pad);
                    use_sp_mainline = false;
                } else {
                    img_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                        img,
                                                                        rank,
                                                                        world_size,
                                                                        1,
                                                                        "flux_sp_img_split");
                    txt_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                        txt,
                                                                        rank,
                                                                        world_size,
                                                                        1,
                                                                        "flux_sp_txt_split");

                    img = img_sp_split.local;
                    txt = txt_sp_split.local;
                    if (flux_profile_enabled() && flux_profile_should_log_rank(ctx)) {
                        LOG_INFO("flux SP profile double local rank=%d/%d img_local_seq=%" PRId64 " txt_local_seq=%" PRId64,
                                 rank,
                                 world_size,
                                 img->ne[1],
                                 txt->ne[1]);
                    }
                }
            }
            if (!use_sp_mainline || flux_sp_strict_barrier_enabled()) {
                sd::ggml_graph_cut::mark_graph_cut(img, "flux.prelude", "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "flux.prelude", "txt");
                sd::ggml_graph_cut::mark_graph_cut(vec, "flux.prelude", "vec");
            }

#ifndef ED_DEBUG_SP_COMM
            ggml_tensor* sp_prepared_pe_seq_major = nullptr;
            if (use_sp_mainline &&
                flux_sp_shared_pe_seq_major_enabled() &&
                flux_sp_qk_seq_major_enabled()) {
                sp_prepared_pe_seq_major = flux_sp_prepare_rope_pe_seq_major(ctx->ggml_ctx,
                                                                              pe,
                                                                              "flux_sp_shared_pe_seq_major");
                sd::ggml_graph_cut::mark_graph_cut(sp_prepared_pe_seq_major,
                                                   "flux.shared_pe",
                                                   "seq_major");
            }
#endif

            for (int i = 0; i < params.depth && !whole_inject; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                    continue;
                }

                auto block = std::dynamic_pointer_cast<DoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);

                if (use_sp_mainline) {
#ifdef ED_DEBUG_SP_COMM
                    ggml_tensor* debug_block_img_ref = debug_compare_double_inner &&
                                                       i == debug_compare_double_block_idx ?
                                                           debug_img_double_inner_ref :
                                                           nullptr;
                    ggml_tensor* debug_block_txt_ref = debug_compare_double_inner &&
                                                       i == debug_compare_double_block_idx ?
                                                           debug_txt_double_inner_ref :
                                                           nullptr;
                    auto img_txt = block->forward_sp(ctx,
                                                     img,
                                                     txt,
                                                     vec,
                                                     pe,
                                                     txt_img_mask,
                                                     ds_img_mods,
                                                     ds_txt_mods,
                                                     debug_block_img_ref,
                                                     debug_block_txt_ref);
                    img          = img_txt.first;   // [N, local_img_token, hidden_size]
                    txt          = img_txt.second;  // [N, local_txt_token, hidden_size]
#else
                    auto img_txt = block->forward_sp(ctx,
                                                     img,
                                                     txt,
                                                     vec,
                                                     pe,
                                                     txt_img_mask,
                                                     ds_img_mods,
                                                     ds_txt_mods,
                                                     sp_prepared_pe_seq_major);
                    img          = img_txt.first;   // [N, local_img_token, hidden_size]
                    txt          = img_txt.second;  // [N, local_txt_token, hidden_size]
#endif
                } else {
                    auto img_txt = block->forward(ctx, img, txt, vec, pe, txt_img_mask, ds_img_mods, ds_txt_mods);
                    img          = img_txt.first;   // [N, n_img_token, hidden_size]
                    txt          = img_txt.second;  // [N, n_txt_token, hidden_size]
                }
                // The segmented graph-cut runner only preserves cut outputs
                // across segments. Keep block outputs as cache boundaries so
                // later communication segments do not recompute the residual
                // chain from earlier blocks.
                if (!flux_sp_skip_block_cuts_with_custom_comm_enabled(ctx)) {
                    sd::ggml_graph_cut::mark_graph_cut(img, "flux.double_blocks." + std::to_string(i), "img");
                    sd::ggml_graph_cut::mark_graph_cut(txt, "flux.double_blocks." + std::to_string(i), "txt");
                }
                // Substep tap: double-block output k (BlockOut[i]) — the DiCache probe
                // point (img stream). Conditional no-op unless requested. Also drives
                // the substep probe stop.
                tap(ctx, edgedit::cache::AnchorRef::block_out(i), img);
                if (ctx->tap_registry != nullptr && ctx->tap_registry->stop_after(i)) {
                    return img;
                }
#ifdef ED_DEBUG_SP_COMM
                if (debug_compare_double_inner &&
                    i >= debug_compare_double_block_idx) {
                    // build_debug_sp_graph() expands debug_sp_total_error(); this return only lets build_graph() finish.
                    return img;
                }
                if (debug_compare_double_block &&
                    i >= debug_compare_double_block_idx) {
                    break;
                }
#endif
            }

#ifdef ED_DEBUG_SP_COMM
            auto debug_compare_sequence_shard = [&](ggml_tensor* local,
                                                    ggml_tensor* full,
                                                    const std::string& name) {
                if (local == nullptr || full == nullptr) {
                    return;
                }
                ggml_tensor* ref = flux_debug_sequence_shard_reference(ctx, full, name);
                mark_flux_debug_compare_tensor(ctx->ggml_ctx, local, ref, name);
            };

            auto debug_double_compare_label = [&]() -> std::string {
                if (debug_compare_double_block) {
                    return "flux_mainline_double_block" +
                           std::to_string(debug_compare_double_block_idx);
                }
                return "flux_mainline_double_end";
            };

            if (use_sp_mainline && (debug_compare_double_end || debug_compare_double_block)) {
                const std::string name = debug_double_compare_label();
                debug_compare_sequence_shard(img,
                                             debug_img_after_double_ref,
                                             name + "_img_local");
                debug_compare_sequence_shard(txt,
                                             debug_txt_after_double_ref,
                                             name + "_txt_local");
            }
#endif

            bool txt_img_already_split_for_single = false;
            ggml_tensor* txt_img = nullptr;
            if (use_sp_mainline) {
                const int rank       = flux_sp_rank(ctx);
                const int world_size = flux_sp_world_size(ctx);
#ifndef ED_DEBUG_SP_COMM
                if (flux_sp_double_to_single_reshard_enabled()) {
                    auto reshard = edgedit::parallel::sp_double_to_single_reshard_sequence_2way(ctx->ggml_ctx,
                                                                                               txt,
                                                                                               img,
                                                                                               rank,
                                                                                               world_size,
                                                                                               ctx->process_group,
                                                                                               "flux_sp_double_to_single_reshard");
                    if (reshard.local != nullptr) {
                        txt_img = reshard.local;
                        txt_img_sp_split.local = txt_img;
                        txt_img_sp_split.local_view = txt_img;
                        txt_img_sp_split.input_padded = txt_img;
                        txt_img_sp_split.rank = rank;
                        txt_img_sp_split.world_size = world_size;
                        txt_img_sp_split.seq_dim = 1;
                        txt_img_sp_split.local_seq_len = reshard.local_seq_len;
                        txt_img_sp_split.padded_seq_len = reshard.local_seq_len * world_size;
                        txt_img_sp_split.original_seq_len = txt_img_sp_split.padded_seq_len;
                        txt_img_sp_split.pad = 0;
                        txt_img_already_split_for_single = true;
                        if (flux_profile_enabled() && flux_profile_should_log_rank(ctx)) {
                            LOG_INFO("flux SP profile double-to-single reshard rank=%d/%d txt_local_seq=%" PRId64 " img_local_seq=%" PRId64 " txt_img_local_seq=%" PRId64 " count_per_peer=%.2fMiB",
                                     rank,
                                     world_size,
                                     reshard.first_local_seq,
                                     reshard.second_local_seq,
                                     reshard.local_seq_len,
                                     static_cast<double>(reshard.count_per_peer * ggml_type_size(txt->type) / ggml_blck_size(txt->type)) /
                                         (1024.0 * 1024.0));
                        }
                    }
                }
#endif
                if (!txt_img_already_split_for_single) {
                    auto double_gather = edgedit::parallel::sp_mark_gather_sequence_batched(ctx->ggml_ctx,
                                                                                            {txt, img},
                                                                                            world_size,
                                                                                            1,
                                                                                            {txt_sp_split.pad, img_sp_split.pad},
                                                                                            "flux_sp_double_txt_img_gather");
                    GGML_ASSERT(double_gather.gathered.size() == 2);
                    txt = double_gather.gathered[0];
                    img = double_gather.gathered[1];
                }
#ifdef ED_DEBUG_SP_COMM
                if (debug_compare_double_end || debug_compare_double_block) {
                    const std::string name = debug_double_compare_label();
                    if (txt_sp_split.pad == 0) {
                        mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                                       double_gather.gathered_padded[0],
                                                       debug_txt_after_double_ref,
                                                       name + "_txt_gather_recv");
                    }
                    if (img_sp_split.pad == 0) {
                        mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                                       double_gather.gathered_padded[1],
                                                       debug_img_after_double_ref,
                                                       name + "_img_gather_recv");
                    }
                }
#endif
            }

#ifdef ED_DEBUG_SP_COMM
            if (use_sp_mainline && (debug_compare_double_end || debug_compare_double_block)) {
                const std::string debug_double_compare_name = debug_double_compare_label();
                mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                               img,
                                               debug_img_after_double_ref,
                                               debug_double_compare_name + "_img");
                mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                               txt,
                                               debug_txt_after_double_ref,
                                               debug_double_compare_name + "_txt");
                if (debug_compare_double_block) {
                    // build_debug_sp_graph() expands debug_sp_total_error(); this return only lets build_graph() finish.
                    return img;
                }
            }
#endif

            if (txt_img == nullptr) {
                txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
            }
#ifdef ED_DEBUG_SP_COMM
            if (use_sp_mainline && debug_compare_single_entry) {
                mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                               txt_img,
                                               debug_txt_img_before_single_ref,
                                               "flux_mainline_single_entry_txt_img");
            }
#endif
            if (use_sp_mainline && !txt_img_already_split_for_single) {
                const int rank       = flux_sp_rank(ctx);
                const int world_size = flux_sp_world_size(ctx);
                const int64_t txt_img_pad = edgedit::parallel::sp_sequence_padding(txt_img->ne[1],
                                                                                    world_size);
                if (flux_profile_enabled() && flux_profile_should_log_rank(ctx)) {
                    LOG_INFO("flux SP profile single split rank=%d/%d txt_img_seq=%" PRId64 " txt_img_pad=%" PRId64,
                             rank,
                             world_size,
                             txt_img->ne[1],
                             txt_img_pad);
                }
                if (txt_img_pad != 0) {
                    LOG_WARN("flux SP mainline disabled before single blocks because combined sequence needs padding: rank=%d world_size=%d seq=%" PRId64 " pad=%" PRId64,
                             rank,
                             world_size,
                             txt_img->ne[1],
                             txt_img_pad);
                    use_sp_mainline = false;
                } else {
                    txt_img_sp_split = edgedit::parallel::sp_split_sequence(ctx->ggml_ctx,
                                                                            txt_img,
                                                                            rank,
                                                                            world_size,
                                                                            1,
                                                                            "flux_sp_txt_img_split");
                    txt_img = txt_img_sp_split.local;
                    if (flux_profile_enabled() && flux_profile_should_log_rank(ctx)) {
                        LOG_INFO("flux SP profile single local rank=%d/%d txt_img_local_seq=%" PRId64,
                                 rank,
                                 world_size,
                                 txt_img->ne[1]);
                    }
                }
            }
            for (int i = 0; i < params.depth_single_blocks && !whole_inject; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i + params.depth) != skip_layers.end()) {
                    continue;
                }
                auto block = std::dynamic_pointer_cast<SingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);

                if (use_sp_mainline) {
#ifdef ED_DEBUG_SP_COMM
                    txt_img = block->forward_sp(ctx, txt_img, vec, pe, txt_img_mask, ss_mods);
#else
                    txt_img = block->forward_sp(ctx,
                                                txt_img,
                                                vec,
                                                pe,
                                                txt_img_mask,
                                                ss_mods,
                                                sp_prepared_pe_seq_major);
#endif
                } else {
                    txt_img = block->forward(ctx, txt_img, vec, pe, txt_img_mask, ss_mods);
                }
                // See the double-block cache boundary above. Without this
                // cut, each following qkv communication segment walks the
                // residual chain and replays previous single blocks.
                if (!flux_sp_skip_block_cuts_with_custom_comm_enabled(ctx)) {
                    sd::ggml_graph_cut::mark_graph_cut(txt_img, "flux.single_blocks." + std::to_string(i), "txt_img");
                }
            }

            if (use_sp_mainline) {
                auto txt_img_gather = edgedit::parallel::sp_mark_gather_sequence(ctx->ggml_ctx,
                                                                                 txt_img,
                                                                                 flux_sp_world_size(ctx),
                                                                                 1,
                                                                                 txt_img_sp_split.pad,
                                                                                 "flux_sp_final_txt_img_gather",
                                                                                 ctx->process_group);
                txt_img = txt_img_gather.gathered;
            }

#ifdef ED_DEBUG_SP_COMM
            if (use_sp_mainline && debug_compare_single_end) {
                mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                               txt_img,
                                               debug_txt_img_after_single_ref,
                                               "flux_mainline_single_end_txt_img");
            }
#endif

            img = ggml_view_3d(ctx->ggml_ctx,
                               txt_img,
                               txt_img->ne[0],
                               flux_full_img_seq,
                               txt_img->ne[2],
                               txt_img->nb[1],
                               txt_img->nb[2],
                               flux_full_txt_seq * txt_img->nb[1]);  // [N, n_img_token, hidden_size]

#ifdef ED_DEBUG_SP_COMM
            if (use_sp_mainline && debug_compare_final_img) {
                mark_flux_debug_compare_tensor(ctx->ggml_ctx,
                                               img,
                                               debug_img_before_final_layer_ref,
                                               "flux_mainline_final_img_before_layer");
            }
#endif

            // Cache seam (whole-stack path): tap-driven whole-stack inject
            // (substep reuse): the loops ran zero blocks, so reconstruct
            // x_before + residual via the registry.
            if (tap_whole_inject) {
                img = build_stream_override(ctx, cache_img_before);
            }
            // Substep tap: block-stack output anchor (ModelOut) — the residual's
            // "after" point (post-recombine, before final_layer). Conditional no-op
            // unless requested.
            tap(ctx, edgedit::cache::AnchorRef::model_out(), img);

            if (final_layer) {
                img = final_layer->forward(ctx, img, vec);  // (N, T, patch_size ** 2 * out_channels)
            }

            return img;
        }

        ggml_tensor* _apply_x0_residual(GGMLRunnerContext* ctx,
                                        ggml_tensor* predicted,
                                        ggml_tensor* noisy,
                                        ggml_tensor* timesteps) {
            auto x = ggml_sub(ctx->ggml_ctx, noisy, predicted);
            x      = ggml_div(ctx->ggml_ctx, x, timesteps);
            return x;
        }

        ggml_tensor* forward_chroma_radiance(GGMLRunnerContext* ctx,
                                             ggml_tensor* x,
                                             ggml_tensor* timestep,
                                             ggml_tensor* context,
                                             ggml_tensor* c_concat,
                                             ggml_tensor* y,
                                             ggml_tensor* guidance,
                                             ggml_tensor* pe,
                                             ggml_tensor* mod_index_arange         = nullptr,
                                             ggml_tensor* dct                      = nullptr,
                                             std::vector<ggml_tensor*> ref_latents = {},
                                             std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W      = x->ne[0];
            int64_t H      = x->ne[1];
            int64_t C      = x->ne[2];
            int patch_size = params.patch_size;
            int pad_h      = (patch_size - H % patch_size) % patch_size;
            int pad_w      = (patch_size - W % patch_size) % patch_size;

            auto img      = DiT::pad_to_patch_size(ctx, x, params.patch_size, params.patch_size);
            auto orig_img = img;

            if (params.chroma_radiance_params.fake_patch_size_x2) {
                // It's supposed to be using GGML_SCALE_MODE_NEAREST, but this seems more stable
                // Maybe the implementation of nearest-neighbor interpolation in ggml behaves differently than the one in PyTorch?
                // img = F.interpolate(img, size=(H//2, W//2), mode="nearest")
                img = ggml_interpolate(ctx->ggml_ctx, img, W / 2, H / 2, C, x->ne[3], GGML_SCALE_MODE_BILINEAR);
            }

            auto img_in_patch = std::dynamic_pointer_cast<Conv2d>(blocks["img_in_patch"]);

            img = img_in_patch->forward(ctx, img);                                                       // [N, hidden_size, H/patch_size, W/patch_size]
            img = ggml_reshape_3d(ctx->ggml_ctx, img, img->ne[0] * img->ne[1], img->ne[2], img->ne[3]);  // [N, hidden_size, H/patch_size*W/patch_size]
            img = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img, 1, 0, 2, 3));      // [N, H/patch_size*W/patch_size, hidden_size]

            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, n_img_token, hidden_size]

            // nerf decode
            auto nerf_image_embedder   = std::dynamic_pointer_cast<NerfEmbedder>(blocks["nerf_image_embedder"]);
            auto nerf_final_layer_conv = std::dynamic_pointer_cast<NerfFinalLayerConv>(blocks["nerf_final_layer_conv"]);

            auto nerf_pixels    = DiT::patchify(ctx->ggml_ctx, orig_img, patch_size, patch_size);  // [N, num_patches, C * patch_size * patch_size]
            int64_t num_patches = nerf_pixels->ne[1];
            nerf_pixels         = ggml_reshape_3d(ctx->ggml_ctx,
                                                  nerf_pixels,
                                                  nerf_pixels->ne[0] / C,
                                                  C,
                                                  nerf_pixels->ne[1] * nerf_pixels->ne[2]);                                  // [N*num_patches, C, patch_size*patch_size]
            nerf_pixels         = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, nerf_pixels, 1, 0, 2, 3));  // [N*num_patches, patch_size*patch_size, C]

            auto nerf_hidden = ggml_reshape_2d(ctx->ggml_ctx, out, out->ne[0], out->ne[1] * out->ne[2]);  // [N*num_patches, hidden_size]
            auto img_dct     = nerf_image_embedder->forward(ctx, nerf_pixels, dct);                       // [N*num_patches, patch_size*patch_size, nerf_hidden_size]

            for (int i = 0; i < params.chroma_radiance_params.nerf_depth; i++) {
                auto block = std::dynamic_pointer_cast<NerfGLUBlock>(blocks["nerf_blocks." + std::to_string(i)]);

                img_dct = block->forward(ctx, img_dct, nerf_hidden);
            }

            img_dct = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img_dct, 1, 0, 2, 3));                                 // [N*num_patches, nerf_hidden_size, patch_size*patch_size]
            img_dct = ggml_reshape_3d(ctx->ggml_ctx, img_dct, img_dct->ne[0] * img_dct->ne[1], num_patches, img_dct->ne[2] / num_patches);  // [N, num_patches, nerf_hidden_size*patch_size*patch_size]
            img_dct = DiT::unpatchify(ctx->ggml_ctx, img_dct, (H + pad_h) / patch_size, (W + pad_w) / patch_size, patch_size, patch_size);  // [N, nerf_hidden_size, H, W]

            out = nerf_final_layer_conv->forward(ctx, img_dct);  // [N, C, H, W]

            if (params.chroma_radiance_params.use_x0) {
                out = _apply_x0_residual(ctx, out, orig_img, timestep);
            }

            return out;
        }

        ggml_tensor* forward_flux_chroma(GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* timestep,
                                         ggml_tensor* context,
                                         ggml_tensor* c_concat,
                                         ggml_tensor* y,
                                         ggml_tensor* guidance,
                                         ggml_tensor* pe,
                                         ggml_tensor* mod_index_arange         = nullptr,
                                         ggml_tensor* dct                      = nullptr,
                                         std::vector<ggml_tensor*> ref_latents = {},
                                         std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W      = x->ne[0];
            int64_t H      = x->ne[1];
            int64_t C      = x->ne[2];
            int patch_size = params.patch_size;
            int pad_h      = (patch_size - H % patch_size) % patch_size;
            int pad_w      = (patch_size - W % patch_size) % patch_size;

            auto img           = DiT::pad_and_patchify(ctx, x, patch_size, patch_size);
            int64_t img_tokens = img->ne[1];

            if (params.version == VERSION_FLUX_FILL) {
                GGML_ASSERT(c_concat != nullptr);
                ggml_tensor* masked = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], 0);
                ggml_tensor* mask   = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], 8 * 8, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * C);

                masked = DiT::pad_and_patchify(ctx, masked, patch_size, patch_size);
                mask   = DiT::pad_and_patchify(ctx, mask, patch_size, patch_size);

                img = ggml_concat(ctx->ggml_ctx, img, ggml_concat(ctx->ggml_ctx, masked, mask, 0), 0);
            } else if (params.version == VERSION_FLEX_2) {
                GGML_ASSERT(c_concat != nullptr);
                ggml_tensor* masked  = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], 0);
                ggml_tensor* mask    = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], 1, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * C);
                ggml_tensor* control = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * (C + 1));

                masked  = DiT::pad_and_patchify(ctx, masked, patch_size, patch_size);
                mask    = DiT::pad_and_patchify(ctx, mask, patch_size, patch_size);
                control = DiT::pad_and_patchify(ctx, control, patch_size, patch_size);

                img = ggml_concat(ctx->ggml_ctx, img, ggml_concat(ctx->ggml_ctx, ggml_concat(ctx->ggml_ctx, masked, mask, 0), control, 0), 0);
            } else if (params.version == VERSION_FLUX_CONTROLS) {
                GGML_ASSERT(c_concat != nullptr);

                auto control = DiT::pad_and_patchify(ctx, c_concat, patch_size, patch_size);
                img          = ggml_concat(ctx->ggml_ctx, img, control, 0);
            }

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, patch_size, patch_size);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, num_tokens, C * patch_size * patch_size]

#ifdef ED_DEBUG_SP_COMM
            if (debug_sp_mainline_compare_enabled()) {
                return out;
            }
#endif

            if (out->ne[1] > img_tokens) {
                out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], img_tokens, out->ne[2], out->nb[1], out->nb[2], 0);
                out = ggml_cont(ctx->ggml_ctx, out);
            }

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, patch_size, patch_size);  // [N, C, H, W]
            return out;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* c_concat,
                             ggml_tensor* y,
                             ggml_tensor* guidance,
                             ggml_tensor* pe,
                             ggml_tensor* mod_index_arange         = nullptr,
                             ggml_tensor* dct                      = nullptr,
                             std::vector<ggml_tensor*> ref_latents = {},
                             std::vector<int> skip_layers          = {}) {
            // Forward pass of DiT.
            // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
            // timestep: (N,) tensor of diffusion timesteps
            // context: (N, L, D)
            // c_concat: nullptr, or for (N,C+M, H, W) for Fill
            // y: (N, adm_in_channels) tensor of class labels
            // guidance: (N,)
            // pe: (L, d_head/2, 2, 2)
            // return: (N, C, H, W)

            if (params.version == VERSION_CHROMA_RADIANCE) {
                return forward_chroma_radiance(ctx,
                                               x,
                                               timestep,
                                               context,
                                               c_concat,
                                               y,
                                               guidance,
                                               pe,
                                               mod_index_arange,
                                               dct,
                                               ref_latents,
                                               skip_layers);
            } else {
                return forward_flux_chroma(ctx,
                                           x,
                                           timestep,
                                           context,
                                           c_concat,
                                           y,
                                           guidance,
                                           pe,
                                           mod_index_arange,
                                           dct,
                                           ref_latents,
                                           skip_layers);
            }
        }
    };

    struct FluxRunner : public GGMLRunner {
    public:
        FluxParams flux_params;
        Flux flux;
        std::vector<float> pe_vec;
        std::vector<float> mod_index_arange_vec;
        std::vector<float> dct_vec;
        sd::Tensor<float> guidance_tensor;
        SDVersion version;
        bool use_mask = false;
        sd::Tensor<float> inject_feature_host_;  // kept alive across cache inject build
        sd::Tensor<float> gamma_scalar_host_;    // GPU DiCache: [1] gamma for reconstruct

        // ---- Persistent cross-step GPU state for DiCache (Probe granularity) ----
        // Holds the last computed step's probe state, block input, probe residuals
        // (2-deep) and full residuals (2-deep) as GPU tensors, per CFG branch, so
        // the decision metric and residual reconstruction run on-device instead of
        // reading ~50MB back to host each step. Allocated lazily at the live img
        // shape; freed on reset / shape change. See dicache_perf notes.
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

        FluxRunner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "",
                   SDVersion version                              = VERSION_FLUX,
                   bool use_mask                                  = false)
            : GGMLRunner(backend, offload_params_to_cpu), version(version), use_mask(use_mask) {
            flux_params.version             = version;
            flux_params.guidance_embed      = false;
            flux_params.depth               = 0;
            flux_params.depth_single_blocks = 0;
            if (version == VERSION_FLUX_FILL) {
                flux_params.in_channels = 384;
            } else if (version == VERSION_FLUX_CONTROLS) {
                flux_params.in_channels = 128;
            } else if (version == VERSION_FLEX_2) {
                flux_params.in_channels = 196;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                flux_params.in_channels = 3;
                flux_params.patch_size  = 16;
            } else if (version == VERSION_OVIS_IMAGE) {
                flux_params.semantic_txt_norm = true;
                flux_params.use_yak_mlp       = true;
                flux_params.vec_in_dim        = 0;
            } else if (ed_version_is_flux2(version)) {
                flux_params.in_channels      = 128;
                flux_params.patch_size       = 1;
                flux_params.out_channels     = 128;
                flux_params.mlp_ratio        = 3.f;
                flux_params.theta            = 2000;
                flux_params.axes_dim         = {32, 32, 32, 32};
                flux_params.vec_in_dim       = 0;
                flux_params.qkv_bias         = false;
                flux_params.disable_bias     = true;
                flux_params.share_modulation = true;
                flux_params.ref_index_scale  = 10.f;
                flux_params.use_mlp_silu_act = true;
            }
            int64_t head_dim                   = 0;
            int64_t actual_radiance_patch_size = -1;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (!starts_with(tensor_name, prefix))
                    continue;
                if (tensor_name.find("guidance_in.in_layer.weight") != std::string::npos) {
                    flux_params.guidance_embed = true;
                }
                if (tensor_name.find("__x0__") != std::string::npos) {
                    LOG_DEBUG("using x0 prediction");
                    flux_params.chroma_radiance_params.use_x0 = true;
                }
                if (tensor_name.find("__32x32__") != std::string::npos) {
                    LOG_DEBUG("using patch size 32");
                    flux_params.patch_size = 32;
                }
                if (tensor_name.find("img_in_patch.weight") != std::string::npos) {
                    actual_radiance_patch_size = pair.second.ne[0];
                    LOG_DEBUG("actual radiance patch size: %d", actual_radiance_patch_size);
                }
                if (tensor_name.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                    // Chroma
                    flux_params.is_chroma = true;
                }
                size_t db = tensor_name.find("double_blocks.");
                if (db != std::string::npos) {
                    tensor_name     = tensor_name.substr(db);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth) {
                        flux_params.depth = block_depth + 1;
                    }
                }
                size_t sb = tensor_name.find("single_blocks.");
                if (sb != std::string::npos) {
                    tensor_name     = tensor_name.substr(sb);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth_single_blocks) {
                        flux_params.depth_single_blocks = block_depth + 1;
                    }
                }
                if (ends_with(tensor_name, "txt_in.weight")) {
                    flux_params.context_in_dim = pair.second.ne[0];
                    flux_params.hidden_size    = pair.second.ne[1];
                }
                if (ends_with(tensor_name, "single_blocks.0.norm.key_norm.scale")) {
                    head_dim = pair.second.ne[0];
                }
                if (ends_with(tensor_name, "double_blocks.0.txt_attn.norm.key_norm.scale")) {
                    head_dim = pair.second.ne[0];
                }
            }
            if (actual_radiance_patch_size > 0 && actual_radiance_patch_size != flux_params.patch_size) {
                GGML_ASSERT(flux_params.patch_size == 2 * actual_radiance_patch_size);
                LOG_DEBUG("using fake x2 patch size");
                flux_params.chroma_radiance_params.fake_patch_size_x2 = true;
            }

            flux_params.num_heads = static_cast<int>(flux_params.hidden_size / head_dim);

            LOG_INFO("flux: depth = %d, depth_single_blocks = %d, guidance_embed = %s, context_in_dim = %" PRId64
                     ", hidden_size = %" PRId64 ", num_heads = %d",
                     flux_params.depth,
                     flux_params.depth_single_blocks,
                     flux_params.guidance_embed ? "true" : "false",
                     flux_params.context_in_dim,
                     flux_params.hidden_size,
                     flux_params.num_heads);
            if (flux_params.is_chroma) {
                LOG_INFO("Using pruned modulation (Chroma)");
            }

            flux = Flux(flux_params);
            flux.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "flux";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            flux.get_param_tensors(tensors, prefix);
        }

        std::vector<float> fetch_dct_pos(int patch_size, int max_freqs) {
            const float PI = 3.14159265358979323846f;

            std::vector<float> pos(patch_size);
            for (int i = 0; i < patch_size; ++i) {
                pos[i] = static_cast<float>(i) / static_cast<float>(patch_size - 1);
            }

            std::vector<float> pos_x(patch_size * patch_size);
            std::vector<float> pos_y(patch_size * patch_size);
            for (int i = 0; i < patch_size; ++i) {
                for (int j = 0; j < patch_size; ++j) {
                    pos_x[i * patch_size + j] = pos[j];
                    pos_y[i * patch_size + j] = pos[i];
                }
            }

            std::vector<float> freqs(max_freqs);
            for (int i = 0; i < max_freqs; ++i) {
                freqs[i] = static_cast<float>(i);
            }

            std::vector<float> coeffs(max_freqs * max_freqs);
            for (int fx = 0; fx < max_freqs; ++fx) {
                for (int fy = 0; fy < max_freqs; ++fy) {
                    coeffs[fx * max_freqs + fy] = 1.0f / (1.0f + freqs[fx] * freqs[fy]);
                }
            }

            int num_positions = patch_size * patch_size;
            int num_features  = max_freqs * max_freqs;
            std::vector<float> dct(num_positions * num_features);

            for (int p = 0; p < num_positions; ++p) {
                float px = pos_x[p];
                float py = pos_y[p];

                for (int fx = 0; fx < max_freqs; ++fx) {
                    float cx = std::cos(px * freqs[fx] * PI);
                    for (int fy = 0; fy < max_freqs; ++fy) {
                        float cy                                      = std::cos(py * freqs[fy] * PI);
                        float val                                     = cx * cy * coeffs[fx * max_freqs + fy];
                        dct[p * num_features + (fx * max_freqs + fy)] = val;
                    }
                }
            }

            return dct;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor                  = {},
                                 const sd::Tensor<float>& c_concat_tensor                 = {},
                                 const sd::Tensor<float>& y_tensor                        = {},
                                 const sd::Tensor<float>& guidance_tensor                 = {},
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false,
                                 std::vector<int> skip_layers                             = {}) {
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_optional_input(context_tensor);
            ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
            ggml_tensor* y         = make_optional_input(y_tensor);
            if (flux_params.guidance_embed || flux_params.is_chroma) {
                if (!guidance_tensor.empty()) {
                    this->guidance_tensor = guidance_tensor;
                    if (flux_params.is_chroma) {
                        this->guidance_tensor.fill_(0.f);
                    }
                }
            }
            ggml_tensor* guidance = make_optional_input(this->guidance_tensor);
            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            GGML_ASSERT(x->ne[3] == 1);
            ggml_cgraph* gf = new_graph_custom(FLUX_GRAPH_SIZE);
#ifdef ED_DEBUG_SP_COMM
            clear_debug_sp_output_names();
#endif

            ggml_tensor* mod_index_arange = nullptr;
            ggml_tensor* dct              = nullptr;  // for chroma radiance

            if (flux_params.is_chroma) {
                if (!use_mask) {
                    y = nullptr;
                }

                // ggml_arange is not working on some backends, precompute it
                mod_index_arange_vec = arange(0, 344);
                mod_index_arange     = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, mod_index_arange_vec.size());
                set_backend_tensor_data(mod_index_arange, mod_index_arange_vec.data());
            }
            std::set<int> txt_arange_dims;
            if (ed_version_is_flux2(version)) {
                txt_arange_dims    = {3};
                increase_ref_index = true;
            } else if (version == VERSION_OVIS_IMAGE) {
                txt_arange_dims = {1, 2};
            }

            pe_vec      = Rope::gen_flux_pe(static_cast<int>(x->ne[1]),
                                            static_cast<int>(x->ne[0]),
                                            flux_params.patch_size,
                                            static_cast<int>(x->ne[3]),
                                            static_cast<int>(context->ne[1]),
                                            txt_arange_dims,
                                            ref_latents,
                                            increase_ref_index,
                                            flux_params.ref_index_scale,
                                            flux_params.theta,
                                            circular_y_enabled,
                                            circular_x_enabled,
                                            flux_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / flux_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, flux_params.axes_dim_sum / 2, pos_len);
            ggml_set_name(pe, "pe");
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            if (version == VERSION_CHROMA_RADIANCE) {
                int patch_size     = flux_params.patch_size;
                int nerf_max_freqs = flux_params.chroma_radiance_params.nerf_max_freqs;
                dct_vec            = fetch_dct_pos(patch_size, nerf_max_freqs);
                dct                = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, nerf_max_freqs * nerf_max_freqs, patch_size * patch_size);
                // dct->data = dct_vec.data();
                // print_ggml_tensor(dct);
                // dct->data = nullptr;
                set_backend_tensor_data(dct, dct_vec.data());
            }

            auto runner_ctx = get_context();

            ggml_tensor* out = flux.forward(&runner_ctx,
                                            x,
                                            timesteps,
                                            context,
                                            c_concat,
                                            y,
                                            guidance,
                                            pe,
                                            mod_index_arange,
                                            dct,
                                            ref_latents,
                                            skip_layers);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context                  = {},
                                  const sd::Tensor<float>& c_concat                 = {},
                                  const sd::Tensor<float>& y                        = {},
                                  const sd::Tensor<float>& guidance                 = {},
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false,
                                  std::vector<int> skip_layers                      = std::vector<int>()) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            // y: [N, adm_in_channels] or [1, adm_in_channels]
            // guidance: [N, ]
#ifdef ED_DEBUG_SP_COMM
            const char* run_sp_compare_env = std::getenv("ED_RUN_SP_COMPARE");
            const bool run_sp_layout_compare = run_sp_compare_env != nullptr &&
                                               std::string(run_sp_compare_env) == "1";
            if (run_sp_layout_compare &&
                get_process_group() != nullptr &&
                get_process_group()->enabled() &&
                get_process_group()->size() > 1) {
                auto get_debug_graph = [&]() -> ggml_cgraph* {
                    DebugSPCaptureScope debug_capture_scope;
                    (void)debug_capture_scope;
                    (void)build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers);
                    return build_debug_sp_graph(compute_ctx, FLUX_GRAPH_SIZE);
                };
                auto debug_result = GGMLRunner::compute<float>(get_debug_graph, n_threads, true);
                if (debug_result.has_value() && !debug_result->empty()) {
                    LOG_INFO("flux debug SP compare total_sse = %.9g", debug_result->data()[0]);
                }
            }
            const char* run_sp_mainline_compare_env = std::getenv("ED_FLUX_SP_COMPARE_STAGE");
            if (run_sp_mainline_compare_env != nullptr &&
                run_sp_mainline_compare_env[0] != '\0' &&
                get_process_group() != nullptr &&
                get_process_group()->enabled() &&
                get_process_group()->size() > 1) {
                const std::string compare_stage(run_sp_mainline_compare_env);
                auto get_debug_graph = [&]() -> ggml_cgraph* {
                    DebugSPMainlineCompareScope debug_compare_scope(compare_stage);
                    (void)debug_compare_scope;
                    (void)build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers);
                    const size_t debug_graph_size = compare_stage.rfind("double_inner", 0) == 0 ?
                                                        FLUX_GRAPH_SIZE * 8 :
                                                        FLUX_GRAPH_SIZE * 4;
                    return build_debug_sp_graph(compute_ctx, debug_graph_size);
                };
                auto debug_result = GGMLRunner::compute<float>(get_debug_graph, n_threads, true);
                if (debug_result.has_value() && !debug_result->empty()) {
                    LOG_INFO("flux debug SP mainline compare stage=%s total_sse = %.9g",
                             compare_stage.c_str(),
                             debug_result->data()[0]);
                    const auto& names = debug_sp_error_names();
                    const int64_t n_debug_values = debug_result->numel();
                    for (int64_t i = 1; i < n_debug_values; ++i) {
                        const size_t name_idx = static_cast<size_t>(i - 1);
                        const char* name = name_idx < names.size() ?
                                               names[name_idx].c_str() :
                                               "<unnamed>";
                        LOG_INFO("flux debug SP mainline compare stage=%s component=%s sse = %.9g",
                                 compare_stage.c_str(),
                                 name,
                                 debug_result->data()[i]);
                    }
                    debug_sp_error_names().clear();
                } else {
                    LOG_WARN("flux debug SP mainline compare stage=%s produced no debug result",
                             compare_stage.c_str());
                    debug_sp_error_names().clear();
                }
            }
#endif
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers);
            };

            // Experimental (ED_CACHE_COMPILED_GRAPHS): build the forward graph once
            // and reuse it across sampling steps, refreshing only the input bytes.
            // Gated to the plain non-segmented path and the txt2img case whose graph
            // shape is stable step-to-step: excludes chroma (zeroes guidance
            // in-graph, so the leaf bytes differ from the arg), SLG steps
            // (skip_layers changes structure), and ref-latent variants.
            const bool reuse_graphs =
                flux_env_flag_enabled("ED_CACHE_COMPILED_GRAPHS") &&
                !can_attempt_graph_cut_segmented_compute() &&
                !flux_params.is_chroma &&
                version != VERSION_CHROMA_RADIANCE &&
                skip_layers.empty() &&
                ref_latents.empty();
            if (reuse_graphs) {
                // Order MUST mirror build_graph()'s make_input() call sequence.
                std::vector<const sd::Tensor<float>*> ordered_inputs;
                ordered_inputs.push_back(&x);
                ordered_inputs.push_back(&timesteps);
                if (!context.empty())  ordered_inputs.push_back(&context);
                if (!c_concat.empty()) ordered_inputs.push_back(&c_concat);
                if (!y.empty())        ordered_inputs.push_back(&y);
                if ((flux_params.guidance_embed || flux_params.is_chroma) && !guidance.empty()) {
                    ordered_inputs.push_back(&guidance);
                }
                const bool reuse_profile = flux_profile_enabled();
                const int64_t t_reuse0 = reuse_profile ? ggml_time_ms() : 0;
                auto reuse_result = restore_trailing_singleton_dims(
                    GGMLRunner::compute_reuse<float>(get_graph, ordered_inputs, n_threads), x.dim());
                if (reuse_profile) {
                    LOG_INFO("flux transformer compute (reuse) elapsed = %lld ms",
                             static_cast<long long>(ggml_time_ms() - t_reuse0));
                }
                return reuse_result;
            }

            const bool profile = flux_profile_enabled();
            const int64_t t_flux0 = profile ? ggml_time_ms() : 0;
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
            if (profile) {
                const int64_t t_flux1 = ggml_time_ms();
                auto group = get_process_group();
                const bool sp = group != nullptr && group->enabled() && group->size() > 1;
                const int rank = sp ? group->rank() : 0;
                const int world_size = group != nullptr ? group->size() : 1;
                if (flux_env_flag_enabled("ED_PROFILE_GRAPH_CUTS_ALL_RANKS") || rank == 0) {
                    LOG_INFO("flux transformer compute elapsed = %lld ms sp=%s rank=%d/%d",
                             static_cast<long long>(t_flux1 - t_flux0),
                             sp ? "true" : "false",
                             rank,
                             world_size);
                }
            }
            return result;
        }

        // ---- Cache seam passes (Feature/Probe policies). Reuse build_graph. ----
        // Ensure the per-branch persistent DiCache GPU state exists and matches the
        // current block-stack shape [ne0, ne1, ne2]. (Re)allocates on first use or
        // a shape change. Returns the state; its tensors are zero-initialized and
        // marked invalid (has_probe_hist=false, counts=0) until the first capture
        // fills them.
        DiCacheGpuState& ensure_dicache_gpu_state(const void* branch_key,
                                                  int64_t ne0, int64_t ne1, int64_t ne2) {
            DiCacheGpuState& s = dicache_gpu_states_[branch_key];
            const std::vector<int64_t> want = {ne0, ne1, ne2};
            if (s.buffer != nullptr && s.shape == want) {
                return s;
            }
            s.free();
            s.ctx = new_cache_context(16);
            auto mk = [&]() {
                ggml_tensor* t = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, ne0, ne1, ne2);
                return t;
            };
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

        int dicache_probe_depth_ = 1;  // set by the pipeline before capture/probe

        // ---- Feature-granularity on-GPU reuse (MagCache / TaylorSeer-single) ----
        // MagCache's host reuse pays a ~50MB reconstruct copy + a 2nd host copy +
        // an H2D upload per skip step. When enabled, the last captured block-stack
        // residual stays resident on-device and is injected as x_before + residual
        // straight from device memory. The residual is stored in a CacheStateManager
        // device slot; see compute_capture_to_slot / compute_inject_from_slot below.
        static bool feature_gpu_enabled() {
            const char* v = std::getenv("ED_FEATURE_CACHE_GPU");
            if (v == nullptr || v[0] == '\0') {
                return true;  // default on
            }
            return v[0] != '0';
        }

        // ---- Declarative device-slot seam (B2): on-device single-residual reuse
        // where the persistent residual tensor is a
        // CacheStateManager device slot (passed in) instead of DiCacheGpuState. ----

        // ---- Substep-path tap-driven device inject. The whole
        // stack is skipped (registry inject_active) and x_before + residual is
        // reconstructed via build_tap_inject in the forward. ----
        sd::Tensor<float> compute_substep_inject_slot(int n_threads,
                                             const sd::Tensor<float>& x,
                                             const sd::Tensor<float>& timesteps,
                                             const sd::Tensor<float>& context,
                                             const sd::Tensor<float>& c_concat,
                                             const sd::Tensor<float>& y,
                                             const sd::Tensor<float>& guidance,
                                             const std::vector<sd::Tensor<float>>& ref_latents,
                                             bool increase_ref_index,
                                             std::vector<edgedit::cache::GraphExtension> extensions) {
            if (extensions.empty()) {
                return {};
            }
            edgedit::cache::TapRegistry reg;
            // Whole-stack reuse: the model owns the resume index (its own block
            // count); the cache only asked for a semantic whole-stack override. The
            // ReplaceStream extension carries WHAT stream replaces the region; the
            // region [0, resume) carries WHERE. x_before is the model_in tap the
            // forward passes to build_stream_override directly, so only the slot
            // rides in the extension's extra_inputs.
            const int resume = flux_params.depth + flux_params.depth_single_blocks;
            reg.set_extensions(std::move(extensions));
            reg.set_override_region(0, resume);
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, {});
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {});
            return std::move(pass.output);
        }

        sd::Tensor<float> compute_substep_inject_gpu(int n_threads,
                                             const sd::Tensor<float>& x,
                                             const sd::Tensor<float>& timesteps,
                                             const sd::Tensor<float>& context,
                                             const sd::Tensor<float>& c_concat,
                                             const sd::Tensor<float>& y,
                                             const sd::Tensor<float>& guidance,
                                             const std::vector<sd::Tensor<float>>& ref_latents,
                                             bool increase_ref_index,
                                             float gamma,
                                             const void* branch_key,
                                             int region_start,
                                             int region_end) {
            auto it = dicache_gpu_states_.find(branch_key);
            if (it == dicache_gpu_states_.end() || it->second.buffer == nullptr ||
                it->second.resid_count < 2) {
                return {};  // not enough history yet
            }
            DiCacheGpuState& s = it->second;
            edgedit::cache::TapRegistry reg;
            gamma_scalar_host_ = sd::Tensor<float>({1}, std::vector<float>{gamma});
            const int resume = flux_params.depth + flux_params.depth_single_blocks;
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_tensor* g = make_input(gamma_scalar_host_);
                reg.set_inject_device_blend(s.resid_prev1, s.resid_prev2, g, 0, resume);
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, {});
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {});
            return std::move(pass.output);
        }

        // ---- Substep-path tap-driven capture. The cache layer hands us a set of
        // GraphExtensions (a DIFFERENCE that weaves the residual + a slot to d2d it
        // into); we request the taps they reference, run the forward, the base
        // runner weaves the operator nodes, and we hand each CaptureToSlot result
        // off to its device slot. The runner never learns the math is a residual. ----
        sd::Tensor<float> compute_substep_capture(int n_threads,
                                                  const sd::Tensor<float>& x,
                                                  const sd::Tensor<float>& timesteps,
                                                  const sd::Tensor<float>& context,
                                                  const sd::Tensor<float>& c_concat,
                                                  const sd::Tensor<float>& y,
                                                  const sd::Tensor<float>& guidance,
                                                  const std::vector<sd::Tensor<float>>& ref_latents,
                                                  bool increase_ref_index,
                                                  std::vector<edgedit::cache::GraphExtension> extensions) {
            edgedit::cache::TapRegistry reg;
            // Request exactly the taps the extensions reference (dedup preserved by
            // set_requested's set semantics).
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
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, {});
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {}, handoff);
            return std::move(pass.output);
        }

        // ---- Substep-path tap-driven DiCache probe. Requests
        // ModelIn + BlockOut[m-1] taps, stops after m double blocks, threads the
        // persistent operands, weaves delta_y/delta_x/gamma. ----
        sd::DiffusionCacheResult compute_substep_probe(int n_threads,
                                                       const sd::Tensor<float>& x,
                                                       const sd::Tensor<float>& timesteps,
                                                       const sd::Tensor<float>& context,
                                                       const sd::Tensor<float>& c_concat,
                                                       const sd::Tensor<float>& y,
                                                       const sd::Tensor<float>& guidance,
                                                       const std::vector<sd::Tensor<float>>& ref_latents,
                                                       bool increase_ref_index,
                                                       int probe_depth,
                                                       const void* branch_key,
                                                       bool delta_minus) {
            const int m = std::max(1, probe_depth);
            edgedit::cache::TapRegistry reg;
            const auto probe_anchor = edgedit::cache::AnchorRef::block_out(m - 1);
            const auto before_anchor = edgedit::cache::AnchorRef::model_in();
            reg.set_requested({before_anchor, probe_anchor});
            reg.set_stop_after(m - 1);

            edgedit::cache::TapRegistry::ProbeMetricOperands ops;
            if (branch_key != nullptr) {
                auto it = dicache_gpu_states_.find(branch_key);
                if (it != dicache_gpu_states_.end() && it->second.has_probe_hist) {
                    DiCacheGpuState& gpu = it->second;
                    ops.prev_probe = gpu.prev_probe;
                    ops.prev_input = gpu.prev_input;
                    ops.want_delta_x = delta_minus;
                    if (gpu.probe_resid_count >= 2) {
                        ops.probe_prev1 = gpu.probe_prev1;
                        ops.probe_prev2 = gpu.probe_prev2;
                        ops.want_gamma = true;
                    }
                }
            }
            reg.set_probe_metrics(probe_anchor, before_anchor, ops);

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, {});
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

        // ---- Substep-path tap-driven DiCache seed capture (device). Mirrors Qwen's
        // compute_substep_capture_probe: a full forward whose post-readback refreshes
        // the persistent DiCacheGpuState residual/probe rings device-to-device from the
        // tap-woven residual/probe nodes. No device slot (the ring is runner-owned,
        // not a StateManager slot). ----
        void writeback_from_taps(const void* branch_key) {
            ggml_tensor* feat = get_cache_tensor_by_name("ed_cache_feature");
            if (feat == nullptr) {
                return;
            }
            DiCacheGpuState& s =
                ensure_dicache_gpu_state(branch_key, feat->ne[0], feat->ne[1], feat->ne[2]);
            if (s.buffer == nullptr) {
                return;
            }
            const int m = std::max(1, dicache_probe_depth_);
            const std::string probe_tap = "ed_tap:block_out[" + std::to_string(m - 1) + "]";
            // A silent copy failure (shape/name mismatch) would leave the ring stale
            // and silently degrade the decision, so surface it loudly.
            auto cp = [&](const char* name, ggml_tensor* dst) {
                if (!copy_named_cache_tensor_to(name, dst)) {
                    LOG_ERROR("dicache writeback copy failed: %s", name);
                }
            };
            // full residual ring: prev2 <- prev1, prev1 <- feature
            std::swap(s.resid_prev1, s.resid_prev2);
            cp("ed_cache_feature", s.resid_prev1);
            s.resid_count = std::min(2, s.resid_count + 1);
            // probe residual ring: prev2 <- prev1, prev1 <- (probe - before)
            std::swap(s.probe_prev1, s.probe_prev2);
            cp("ed_cache_probe_resid", s.probe_prev1);
            s.probe_resid_count = std::min(2, s.probe_resid_count + 1);
            // raw probe & input for the delta_y / delta_x decision metrics
            cp(probe_tap.c_str(), s.prev_probe);
            cp("ed_tap:model_in", s.prev_input);
            s.has_probe_hist = true;
        }

        sd::Tensor<float> compute_substep_capture_probe(int n_threads,
                                                        const sd::Tensor<float>& x,
                                                        const sd::Tensor<float>& timesteps,
                                                        const sd::Tensor<float>& context,
                                                        const sd::Tensor<float>& c_concat,
                                                        const sd::Tensor<float>& y,
                                                        const sd::Tensor<float>& guidance,
                                                        const std::vector<sd::Tensor<float>>& ref_latents,
                                                        bool increase_ref_index,
                                                        int probe_depth,
                                                        const void* branch_key) {
            const int m = std::max(1, probe_depth);
            edgedit::cache::TapRegistry reg;
            reg.set_requested({edgedit::cache::AnchorRef::model_in(),
                               edgedit::cache::AnchorRef::block_out(m - 1),
                               edgedit::cache::AnchorRef::model_out()});
            reg.set_capture_residual(true);   // weave ed_cache_feature = ModelOut - ModelIn
            reg.set_capture_writeback(m);     // weave ed_cache_probe_resid = block_out(m-1) - ModelIn
            std::function<void()> handoff = [&]() { writeback_from_taps(branch_key); };
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, {});
            };
            auto pass = run_substep_pass(get_graph, n_threads, &reg, x.dim(), {}, handoff);
            return std::move(pass.output);
        }

        // GPU DiCache is the default: on-device metric + reconstruction avoids the
        // ~4.5s/img host scalar work and the ~50MB/step GPU->host readback, and is
        // slightly MORE accurate (exact GPU reductions). Set ED_DICACHE_GPU=0 to
        // fall back to the host path (e.g. for debugging or a backend without the
        // required ggml reduction ops).
        static bool dicache_gpu_enabled() {
            const char* v = std::getenv("ED_DICACHE_GPU");
            if (v == nullptr || v[0] == '\0') {
                return true;  // default on
            }
            return v[0] != '0';
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // cpu f16:
                // cuda f16: nan
                // cuda q8_0: pass
                sd::Tensor<float> x({16, 16, 128, 1});
                // ggml_set_f32(x, 0.01f);
                // auto x = load_tensor_from_file(ctx, "chroma_x.bin");
                // print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 1.f);
                auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);

                std::vector<float> guidance_vec(1, 0.f);
                auto guidance = sd::Tensor<float>::from_vector(guidance_vec);

                sd::Tensor<float> context({15360, 256, 1});
                // ggml_set_f32(context, 0.01f);
                // auto context = load_tensor_from_file(ctx, "chroma_context.bin");
                // print_ggml_tensor(context);

                // auto y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 1);
                // ggml_set_f32(y, 0.01f);
                auto y = nullptr;
                // print_ggml_tensor(y);

                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = compute(8,
                                       x,
                                       timesteps,
                                       context,
                                       {},
                                       {},
                                       guidance,
                                       {},
                                       false);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("flux test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_COUNT;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            std::shared_ptr<FluxRunner> flux = std::make_shared<FluxRunner>(backend,
                                                                            false,
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            VERSION_FLUX2,
                                                                            false);

            flux->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            flux->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("flux model loaded");
            flux->test();
        }
    };

}  // namespace Flux

#endif  // __FLUX_HPP__
