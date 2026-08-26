#ifndef __LLM_HPP__
#define __LLM_HPP__

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backend/ggml/ggml_extend.hpp"
#include "json.hpp"
#include "dit_models/components/common/rope.hpp"
#include "tokenizers/bpe_tokenizer.h"
#include "tokenizers/gemma_tokenizer.h"
#include "tokenizers/mistral_tokenizer.h"
#include "tokenizers/qwen2_tokenizer.h"

namespace LLM {
    constexpr int LLM_GRAPH_SIZE = 65536;

    static inline bool qwen_align_debug_enabled() {
        const char* env = std::getenv("ED_DEBUG_QWEN_ALIGN");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }

    static inline bool qwen_align_env_flag_enabled_or_default(const char* name, bool default_enabled) {
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

    static inline bool qwen_align_bf16_vision_activations_enabled() {
        return qwen_align_env_flag_enabled_or_default("ED_QWEN_BF16_VISION_ACTIVATIONS", false);
    }

    static inline bool qwen_align_bf16_llm_internals_enabled() {
        return qwen_align_env_flag_enabled_or_default("ED_QWEN_BF16_LLM_INTERNALS", false);
    }

    static inline bool qwen_align_diffusers_vision_dtype_enabled() {
        return qwen_align_env_flag_enabled_or_default("ED_QWEN_DIFFUSERS_VISION_DTYPE", true);
    }

    static inline bool qwen_align_diffusers_text_dtype_enabled() {
        return qwen_align_env_flag_enabled_or_default("ED_QWEN_DIFFUSERS_TEXT_DTYPE", false);
    }

    static inline bool qwen_align_window_sdpa_enabled() {
        return qwen_align_env_flag_enabled_or_default("ED_QWEN_DIFFUSERS_WINDOW_SDPA", true);
    }

    static inline bool qwen_align_patch_embed_cudnn_conv3d_enabled() {
        return qwen_align_env_flag_enabled_or_default("ED_QWEN_PATCH_EMBED_CUDNN_CONV3D", true);
    }

    static inline ggml_tensor* qwen_align_bf16_roundtrip_to_f32(ggml_context* ctx, ggml_tensor* x) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(x != nullptr);
        return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
    }

    static inline ggml_tensor* qwen_align_maybe_bf16_llm_roundtrip(ggml_context* ctx, ggml_tensor* x) {
        return qwen_align_bf16_llm_internals_enabled() ? qwen_align_bf16_roundtrip_to_f32(ctx, x) : x;
    }

    static inline void qwen_align_log_int_samples(const char* name,
                                                  const std::vector<int>& values,
                                                  int64_t start,
                                                  int64_t count) {
        if (!qwen_align_debug_enabled()) {
            return;
        }
        if (values.empty()) {
            LOG_INFO("qwen-align %s: empty", name);
            return;
        }
        start = std::max<int64_t>(0, std::min<int64_t>(start, static_cast<int64_t>(values.size())));
        const int64_t end = std::max<int64_t>(start,
                                              std::min<int64_t>(start + count, static_cast<int64_t>(values.size())));
        std::stringstream ss;
        ss << "qwen-align " << name << ": len=" << values.size() << " range=[" << start << "," << end << ") values=[";
        for (int64_t i = start; i < end; ++i) {
            if (i > start) {
                ss << ' ';
            }
            ss << values[static_cast<size_t>(i)];
        }
        ss << ']';
        LOG_INFO("%s", ss.str().c_str());
    }

    static inline bool qwen_align_csv_contains(const char* csv, const char* target) {
        if (csv == nullptr || csv[0] == '\0' || target == nullptr) {
            return false;
        }
        const std::string haystack(csv);
        size_t pos = 0;
        while (pos <= haystack.size()) {
            size_t end = haystack.find(',', pos);
            if (end == std::string::npos) {
                end = haystack.size();
            }
            size_t begin = pos;
            while (begin < end && std::isspace(static_cast<unsigned char>(haystack[begin]))) {
                ++begin;
            }
            while (end > begin && std::isspace(static_cast<unsigned char>(haystack[end - 1]))) {
                --end;
            }
            if (haystack.compare(begin, end - begin, target) == 0) {
                return true;
            }
            if (end == haystack.size()) {
                break;
            }
            pos = end + 1;
        }
        return false;
    }

    static inline std::string qwen_align_dump_safe_name(const char* name) {
        std::string result = name != nullptr ? name : "tensor";
        for (char& ch : result) {
            const unsigned char uch = static_cast<unsigned char>(ch);
            if (!std::isalnum(uch) && ch != '.' && ch != '_' && ch != '-') {
                ch = '_';
            }
        }
        return result;
    }

    static inline void qwen_align_dump_tensor_if_requested(const char* name,
                                                           const sd::Tensor<float>& tensor) {
        const char* dump_dir = std::getenv("ED_QWEN_ALIGN_DUMP_DIR");
        const char* targets  = std::getenv("ED_QWEN_ALIGN_DUMP_TARGETS");
        if (dump_dir == nullptr || dump_dir[0] == '\0' ||
            !qwen_align_csv_contains(targets, name)) {
            return;
        }
        const std::string base_path = std::string(dump_dir) + "/" +
                                      qwen_align_dump_safe_name(name);
        {
            std::ofstream shape_out(base_path + ".shape");
            const auto& shape = tensor.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) {
                    shape_out << ' ';
                }
                shape_out << shape[i];
            }
            shape_out << '\n';
        }
        std::ofstream data_out(base_path + ".f32.bin", std::ios::binary);
        const auto& values = tensor.values();
        data_out.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    static inline void qwen_align_log_tensor_stats(const char* name, const sd::Tensor<float>& tensor) {
        if (!qwen_align_debug_enabled()) {
            return;
        }
        if (tensor.empty()) {
            LOG_INFO("qwen-align %s: empty", name);
            return;
        }
        qwen_align_dump_tensor_if_requested(name, tensor);

        double sum = 0.0;
        double sum_sq = 0.0;
        float min_value = std::numeric_limits<float>::infinity();
        float max_value = -std::numeric_limits<float>::infinity();
        for (float value : tensor.values()) {
            sum += static_cast<double>(value);
            sum_sq += static_cast<double>(value) * static_cast<double>(value);
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }

        const double count = static_cast<double>(tensor.values().size());
        const double mean = sum / count;
        const double variance = std::max(0.0, sum_sq / count - mean * mean);
        LOG_INFO("qwen-align %s: shape=%s numel=%zu mean=%.9g std=%.9g min=%.9g max=%.9g l2=%.9g",
                 name,
                 sd::tensor_shape_to_string(tensor.shape()).c_str(),
                 tensor.values().size(),
                 mean,
                 std::sqrt(variance),
                 static_cast<double>(min_value),
                 static_cast<double>(max_value),
                 std::sqrt(sum_sq));

        const float* data = tensor.data();
        const size_t n = tensor.values().size();
        auto sample = [&](size_t offset) -> std::array<float, 4> {
            offset = std::min(offset, n > 4 ? n - 4 : 0);
            return {data[offset], data[offset + 1], data[offset + 2], data[offset + 3]};
        };
        const auto a = sample(0);
        const auto b = sample(4);
        const auto c = sample(std::min<size_t>(64, n > 4 ? n - 4 : 0));
        const auto d = sample(n > 4 ? n - 4 : 0);
        LOG_INFO("qwen-align %s flat_samples: f0=[%.9g %.9g %.9g %.9g] f4=[%.9g %.9g %.9g %.9g] f64=[%.9g %.9g %.9g %.9g] flast=[%.9g %.9g %.9g %.9g]",
                 name,
                 static_cast<double>(a[0]), static_cast<double>(a[1]), static_cast<double>(a[2]), static_cast<double>(a[3]),
                 static_cast<double>(b[0]), static_cast<double>(b[1]), static_cast<double>(b[2]), static_cast<double>(b[3]),
                 static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2]), static_cast<double>(c[3]),
                 static_cast<double>(d[0]), static_cast<double>(d[1]), static_cast<double>(d[2]), static_cast<double>(d[3]));
    }

    enum class LLMArch {
        QWEN2_5_VL,
        QWEN3,
        QWEN3_VL,
        MISTRAL_SMALL_3_2,
        MINISTRAL_3_3B,
        GEMMA3_12B,
        ARCH_COUNT,
    };

    static const char* llm_arch_to_str[] = {
        "qwen2.5vl",
        "qwen3",
        "qwen3vl",
        "mistral_small3.2",
        "ministral3.3b",
        "gemma3_12b",
    };

    enum class MLPActivation {
        SILU,
        GELU_TANH,
    };

    enum class LLMVisionArch {
        QWEN2_5_VL,
        QWEN3_VL,
    };

    struct LLMVisionParams {
        LLMVisionArch arch                 = LLMVisionArch::QWEN2_5_VL;
        int num_layers                      = 32;
        int64_t hidden_size                 = 1280;
        int64_t intermediate_size           = 3420;
        int num_heads                       = 16;
        int64_t in_channels                 = 3;
        int64_t out_hidden_size             = 3584;
        int temporal_patch_size             = 2;
        int patch_size                      = 14;
        int spatial_merge_size              = 2;
        int window_size                     = 112;
        int num_position_embeddings         = 0;
        std::vector<int> deepstack_visual_indexes;
        std::set<int> fullatt_block_indexes = {7, 15, 23, 31};
    };

    struct LLMParams {
        LLMArch arch              = LLMArch::QWEN2_5_VL;
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int num_heads             = 28;
        int num_kv_heads          = 4;
        int head_dim              = 128;
        bool qkv_bias             = true;
        bool qk_norm              = false;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;
        bool final_norm           = true;
        bool rms_norm_add         = false;
        bool normalize_input      = false;
        int64_t max_position_embeddings = 128000;
        MLPActivation mlp_activation = MLPActivation::SILU;
        std::vector<float> rope_thetas = {1000000.f};
        std::vector<float> rope_scales = {1.f};
        std::vector<int> sliding_attention;
        LLMVisionParams vision;
    };

    struct LLMImageEmbedInfo {
        int token_index     = 0;
        int64_t token_count = 0;
        int64_t grid_t      = 1;
        int64_t grid_h      = 0;
        int64_t grid_w      = 0;
    };

    struct MLP : public GGMLBlock {
    public:
        MLP(int64_t hidden_size,
            int64_t intermediate_size,
            bool bias = false,
        bool use_model_bias_type = false,
        bool force_prec_f32 = false,
            bool cast_output_to_input_type = false,
            MLPActivation activation_ = MLPActivation::SILU)
            : activation(activation_) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(
                new Linear(hidden_size,
                           intermediate_size,
                           bias,
                           false,
                           force_prec_f32,
                           1.f,
                           false,
                           use_model_bias_type,
                           cast_output_to_input_type));
            blocks["up_proj"] = std::shared_ptr<GGMLBlock>(
                new Linear(hidden_size,
                           intermediate_size,
                           bias,
                           false,
                           force_prec_f32,
                           1.f,
                           false,
                           use_model_bias_type,
                           cast_output_to_input_type));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(
                new Linear(intermediate_size,
                           hidden_size,
                           bias,
                           false,
                           force_prec_f32,
                           1.f,
                           false,
                           use_model_bias_type,
                           cast_output_to_input_type));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             const std::string& debug_target = "",
                             const std::string& debug_prefix = "") {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);
            auto is_debug_target = [&](const char* suffix) {
                return !debug_prefix.empty() && debug_target == debug_prefix + suffix;
            };

            auto h = gate_proj->forward(ctx, x);
            h      = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, h);
            if (is_debug_target(".mlp.gate")) {
                return h;
            }
            if (activation == MLPActivation::GELU_TANH) {
                h = ggml_ext_gelu(ctx->ggml_ctx, h, true);
            } else {
                h = ggml_silu_inplace(ctx->ggml_ctx, h);
            }
            h      = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, h);
            if (is_debug_target(".mlp.act")) {
                return h;
            }
            auto u = up_proj->forward(ctx, x);
            u      = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, u);
            if (is_debug_target(".mlp.up")) {
                return u;
            }
            h      = ggml_mul_inplace(ctx->ggml_ctx, h, u);
            h      = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, h);
            if (is_debug_target(".mlp.mul")) {
                return h;
            }
            h      = down_proj->forward(ctx, h);
            h      = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, h);
            if (is_debug_target(".mlp.out")) {
                return h;
            }
            return h;
        }

    protected:
        MLPActivation activation = MLPActivation::SILU;
    };

    struct GemmaQKRMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         std::string prefix = "") override {
            this->prefix = prefix;
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        }

    public:
        GemmaQKRMSNorm(int64_t hidden_size, float eps)
            : hidden_size(hidden_size), eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* weight = params["weight"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx,
                                                            ctx->backend,
                                                            weight,
                                                            prefix + "weight");
            }
            x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            // Keep Gemma's 256-wide Q/K norm on the standard CUDA path. Edge's
            // small-width fused kernel reduces in a different order than sdcpp.
            x = ggml_dup(ctx->ggml_ctx, x);
            return ggml_mul(ctx->ggml_ctx, x, weight);
        }
    };

    struct VisionPatchEmbed : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        bool bias;
        int patch_size;
        int temporal_patch_size;
        int64_t in_channels;
        int64_t embed_dim;

    public:
        VisionPatchEmbed(bool llama_cpp_style,
                         LLMVisionArch arch,
                         int patch_size          = 14,
                         int temporal_patch_size = 2,
                         int64_t in_channels     = 3,
                         int64_t embed_dim       = 1152)
            : llama_cpp_style(llama_cpp_style),
              bias(arch == LLMVisionArch::QWEN3_VL),
              patch_size(patch_size),
              temporal_patch_size(temporal_patch_size),
              in_channels(in_channels),
              embed_dim(embed_dim) {
            if (llama_cpp_style) {
                blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
                blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
            } else {
                std::tuple<int, int, int> kernel_size = {(int)temporal_patch_size, (int)patch_size, (int)patch_size};
                blocks["proj"]                        = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                                              embed_dim,
                                                                                              kernel_size,
                                                                                              kernel_size,  // stride
                                                                                              {0, 0, 0},    // padding
                                                                                              {1, 1, 1},    // dilation
                                                                                              bias,
                                                                                              true));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N*grid_t*grid_h*grid_w, in_channels, temporal_patch_size*patch_size*patch_size]
            // return: [N*grid_t*grid_h*grid_w, embed_dim]
            if (llama_cpp_style) {
                x = ggml_reshape_4d(ctx->ggml_ctx,
                                    x,
                                    patch_size,
                                    patch_size,
                                    temporal_patch_size,
                                    ggml_nelements(x) / (temporal_patch_size * patch_size * patch_size));

                auto proj_0 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.0"]);
                auto proj_1 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.1"]);

                auto x0 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x0      = ggml_reshape_4d(ctx->ggml_ctx, x0, x0->ne[0], x0->ne[1], in_channels, x0->ne[3] / in_channels);
                x0      = proj_0->forward(ctx, x0);

                auto x1 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, 2);
                x1      = ggml_reshape_4d(ctx->ggml_ctx, x1, x1->ne[0], x1->ne[1], in_channels, x1->ne[3] / in_channels);
                x1      = proj_1->forward(ctx, x1);

                x = ggml_add(ctx->ggml_ctx, x0, x1);
            } else {
                auto proj = std::dynamic_pointer_cast<Conv3d>(blocks["proj"]);

                ggml_tensor* w = proj->weight_for_forward(ctx, true);
                ggml_tensor* b = proj->bias_for_forward(ctx);
                if ((w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16) && x->type != w->type) {
                    x = ggml_cast(ctx->ggml_ctx, x, w->type);
                }

                const int64_t patch_dim   = patch_size * patch_size * temporal_patch_size * in_channels;
                const int64_t patch_count = ggml_nelements(x) / patch_dim;
                if (qwen_align_patch_embed_cudnn_conv3d_enabled() &&
                    sd_backend_is(ctx->backend, "CUDA") &&
                    patch_count > 0 &&
                    ggml_nelements(x) == patch_dim * patch_count &&
                    (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_F32)) {
                    ggml_tensor* x_conv = ggml_reshape_4d(ctx->ggml_ctx,
                                                          x,
                                                          patch_size,
                                                          patch_size,
                                                          temporal_patch_size,
                                                          in_channels * patch_count);
                    ggml_tensor* y_conv = ggml_ext_conv_3d_direct_typed(ctx->ggml_ctx,
                                                                        x_conv,
                                                                        w,
                                                                        b,
                                                                        in_channels,
                                                                        patch_count,
                                                                        embed_dim,
                                                                        patch_size,
                                                                        patch_size,
                                                                        temporal_patch_size,
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        1,
                                                                        w->type);
                    if (ggml_backend_supports_op(ctx->backend, y_conv)) {
                        return ggml_reshape_2d(ctx->ggml_ctx, y_conv, embed_dim, patch_count);
                    }
                }

                w = ggml_reshape_2d(ctx->ggml_ctx,
                                    w,
                                    patch_dim,
                                    embed_dim);
                x = ggml_ext_linear(ctx->ggml_ctx, x, w, b);
                if (qwen_align_diffusers_vision_dtype_enabled() &&
                    (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16) &&
                    x->type != w->type) {
                    x = ggml_cast(ctx->ggml_ctx, x, w->type);
                }
            }

            x = ggml_reshape_2d(ctx->ggml_ctx, x, embed_dim, ggml_nelements(x) / embed_dim);
            return x;
        }
    };

    struct PatchMerger : public GGMLBlock {
    protected:
        LLMVisionArch arch_;
        int64_t hidden_size;

    public:
        PatchMerger(LLMVisionArch arch,
                    int64_t dim,
                    int64_t context_dim,
                    int64_t spatial_merge_size)
            : arch_(arch) {
            const bool diffusers_dtype = qwen_align_diffusers_vision_dtype_enabled();
            hidden_size                = context_dim * spatial_merge_size * spatial_merge_size;
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                blocks["norm"]       = std::make_shared<LayerNorm>(context_dim, 1e-6f);
                blocks["linear_fc1"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
                blocks["linear_fc2"] = std::make_shared<Linear>(hidden_size, dim, true);
                return;
            }
            blocks["ln_q"]             = std::shared_ptr<GGMLBlock>(new RMSNorm(context_dim, 1e-6f, false, true));
            blocks["mlp.0"]            = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                    hidden_size,
                                                                    true,
                                                                    false,
                                                                    diffusers_dtype,
                                                                    1.f,
                                                                    false,
                                                                    false,
                                                                    diffusers_dtype));
            // mlp.1 is nn.GELU()
            blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                    dim,
                                                                    true,
                                                                    false,
                                                                    diffusers_dtype,
                                                                    1.f,
                                                                    false,
                                                                    false,
                                                                    diffusers_dtype));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             const std::string& debug_target = "",
                             const std::string& debug_prefix = "") {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                auto norm       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
                auto linear_fc1 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"]);
                auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);
                if (x->type != GGML_TYPE_F32) {
                    x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
                }
                x               = norm->forward(ctx, x);
                x               = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
                x               = linear_fc1->forward(ctx, x);
                x               = ggml_gelu_erf(ctx->ggml_ctx, x);
                return linear_fc2->forward(ctx, x);
            }
            auto ln_q  = std::dynamic_pointer_cast<RMSNorm>(blocks["ln_q"]);
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto is_debug_target = [&](const char* suffix) {
                return !debug_prefix.empty() && debug_target == debug_prefix + suffix;
            };

            if (is_debug_target(".input")) {
                return x;
            }
            x = ln_q->forward(ctx, x);
            x = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".ln_q")) {
                return x;
            }
            x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
            if (is_debug_target(".reshaped")) {
                return x;
            }
            x = mlp_0->forward(ctx, x);
            x = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".mlp.0")) {
                return x;
            }
            x = ggml_gelu_erf(ctx->ggml_ctx, x);
            x = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".gelu")) {
                return x;
            }
            x = mlp_2->forward(ctx, x);
            x = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".out")) {
                return x;
            }
            return x;
        }
    };

    struct Qwen3VLDeepStackMerger : public GGMLBlock {
    protected:
        int64_t merge_dim;

    public:
        Qwen3VLDeepStackMerger(int64_t dim, int64_t context_dim, int64_t spatial_merge_size)
            : merge_dim(context_dim * spatial_merge_size * spatial_merge_size) {
            blocks["norm"]       = std::make_shared<LayerNorm>(merge_dim, 1e-6f);
            blocks["linear_fc1"] = std::make_shared<Linear>(merge_dim, merge_dim, true);
            blocks["linear_fc2"] = std::make_shared<Linear>(merge_dim, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear_fc1 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"]);
            auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);
            if (x->type != GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            x               = ggml_reshape_2d(ctx->ggml_ctx, x, merge_dim, ggml_nelements(x) / merge_dim);
            x               = norm->forward(ctx, x);
            x               = linear_fc1->forward(ctx, x);
            x               = ggml_gelu_erf(ctx->ggml_ctx, x);
            return linear_fc2->forward(ctx, x);
        }
    };

    struct VisionMLP : public GGMLBlock {
    protected:
        LLMVisionArch arch_;

    public:
        VisionMLP(LLMVisionArch arch, int64_t hidden_size, int64_t intermediate_size)
            : arch_(arch) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                blocks["linear_fc1"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
                blocks["linear_fc2"] = std::make_shared<Linear>(intermediate_size, hidden_size, true);
            } else {
                blocks["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
                blocks["up_proj"]   = std::make_shared<Linear>(hidden_size, intermediate_size, true);
                blocks["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, true);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                x = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"])->forward(ctx, x);
                x = ggml_ext_gelu(ctx->ggml_ctx, x);
                return std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"])->forward(ctx, x);
            }
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);
            auto h         = ggml_silu_inplace(ctx->ggml_ctx, gate_proj->forward(ctx, x));
            h              = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            return down_proj->forward(ctx, h);
        }
    };

    struct VisionAttention : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        LLMVisionArch arch_;
        int head_dim;
        int num_heads;

    public:
        VisionAttention(bool llama_cpp_style,
                        LLMVisionArch arch,
                        int64_t hidden_size,
                        int num_heads)
            : llama_cpp_style(llama_cpp_style), arch_(arch), num_heads(num_heads) {
            head_dim = static_cast<int>(hidden_size / num_heads);
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            const bool diffusers_dtype = arch_ == LLMVisionArch::QWEN2_5_VL && qwen_align_diffusers_vision_dtype_enabled();
            const bool bias = arch_ == LLMVisionArch::QWEN2_5_VL || arch_ == LLMVisionArch::QWEN3_VL;
            if (llama_cpp_style) {
                blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                         hidden_size,
                                                                         bias,
                                                                         false,
                                                                         diffusers_dtype,
                                                                         1.f,
                                                                         false,
                                                                         false,
                                                                         diffusers_dtype));
                blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                         hidden_size,
                                                                         bias,
                                                                         false,
                                                                         diffusers_dtype,
                                                                         1.f,
                                                                         false,
                                                                         false,
                                                                         diffusers_dtype));
                blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                         hidden_size,
                                                                         bias,
                                                                         false,
                                                                         diffusers_dtype,
                                                                         1.f,
                                                                         false,
                                                                         false,
                                                                         diffusers_dtype));
            } else {
                blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                      hidden_size * 3,
                                                                      bias,
                                                                      false,
                                                                      diffusers_dtype,
                                                                      1.f,
                                                                      false,
                                                                      false,
                                                                      diffusers_dtype));
            }
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size,
                                                                   hidden_size,
                                                                   bias,
                                                                   false,
                                                                   diffusers_dtype,
                                                                   1.f,
                                                                   false,
                                                                   false,
                                                                   diffusers_dtype));
        }

        ggml_tensor* window_sdpa_attention(GGMLRunnerContext* ctx,
                                           ggml_tensor* q,
                                           ggml_tensor* k,
                                           ggml_tensor* v,
                                           ggml_tensor* pe,
                                           const std::vector<int>* cu_window_seqlens) {
            if (!qwen_align_window_sdpa_enabled() ||
                cu_window_seqlens == nullptr ||
                cu_window_seqlens->size() < 2 ||
                !sd_backend_is(ctx->backend, "CUDA") ||
                v->ne[3] != 1) {
                return nullptr;
            }

            const int64_t d_head = q->ne[0];
            const int64_t n_head = q->ne[1];
            const int64_t L      = q->ne[2];
            const int64_t N      = q->ne[3];
            if (d_head != head_dim ||
                n_head != num_heads ||
                N != 1 ||
                cu_window_seqlens->front() != 0 ||
                cu_window_seqlens->back() != L) {
                return nullptr;
            }

            q = Rope::apply_rope(ctx->ggml_ctx, q, pe, false, ctx->backend);
            k = Rope::apply_rope(ctx->ggml_ctx, k, pe, false, ctx->backend);

            const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
            ggml_tensor* out_all = nullptr;
            for (size_t i = 0; i + 1 < cu_window_seqlens->size(); ++i) {
                const int64_t start = (*cu_window_seqlens)[i];
                const int64_t end   = (*cu_window_seqlens)[i + 1];
                const int64_t len   = end - start;
                if (len <= 0) {
                    continue;
                }
                if (start < 0 || end > L) {
                    return nullptr;
                }

                ggml_tensor* q_slice = ggml_view_3d(ctx->ggml_ctx,
                                                    q,
                                                    d_head,
                                                    len,
                                                    n_head,
                                                    q->nb[1],
                                                    q->nb[2],
                                                    start * q->nb[1]);
                ggml_tensor* k_slice = ggml_view_3d(ctx->ggml_ctx,
                                                    k,
                                                    d_head,
                                                    len,
                                                    n_head,
                                                    k->nb[1],
                                                    k->nb[2],
                                                    start * k->nb[1]);
                ggml_tensor* v_slice = ggml_view_4d(ctx->ggml_ctx,
                                                    v,
                                                    d_head,
                                                    n_head,
                                                    len,
                                                    N,
                                                    v->nb[1],
                                                    v->nb[2],
                                                    v->nb[3],
                                                    start * v->nb[2]);

                q_slice = ggml_ext_cont(ctx->ggml_ctx, q_slice);
                k_slice = ggml_ext_cont(ctx->ggml_ctx, k_slice);
                if (q_slice->type != GGML_TYPE_F32 &&
                    q_slice->type != GGML_TYPE_F16 &&
                    q_slice->type != GGML_TYPE_BF16) {
                    q_slice = ggml_cast(ctx->ggml_ctx, q_slice, GGML_TYPE_F32);
                }
                v_slice = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v_slice, 0, 2, 1, 3));
                v_slice = ggml_reshape_3d(ctx->ggml_ctx, v_slice, d_head, len, n_head * N);

                ggml_tensor* out = ggml_flash_attn_ext(ctx->ggml_ctx, q_slice, k_slice, v_slice, nullptr, scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
                if (!ggml_backend_supports_op(ctx->backend, out)) {
                    return nullptr;
                }
                out = ggml_view_3d(ctx->ggml_ctx, out, d_head, n_head, len, out->nb[1], out->nb[2], 0);
                out = ggml_ext_cont(ctx->ggml_ctx, out);
                out = ggml_reshape_3d(ctx->ggml_ctx, out, d_head * n_head, len, N);

                out_all = out_all == nullptr ? out : ggml_concat(ctx->ggml_ctx, out_all, out, 1);
            }

            return out_all;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr,
                             const std::vector<int>* cu_window_seqlens = nullptr,
                             const std::string& debug_target = "",
                             const std::string& debug_prefix = "") {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto proj       = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            auto is_debug_target = [&](const char* suffix) {
                return !debug_prefix.empty() && debug_target == debug_prefix + suffix;
            };

            std::vector<ggml_tensor*> qkv_vec;
            if (llama_cpp_style) {
                auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
                auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
                auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);

                auto q = q_proj->forward(ctx, x);
                auto k = k_proj->forward(ctx, x);
                auto v = v_proj->forward(ctx, x);

                qkv_vec = {q, k, v};
            } else {
                auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
                auto qkv      = qkv_proj->forward(ctx, x);
                qkv           = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, qkv);
                if (is_debug_target(".attn.qkv")) {
                    return qkv;
                }
                qkv_vec       = split_qkv(ctx->ggml_ctx, qkv);
            }

            auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
            auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
            auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);  // [N, n_token, n_head, d_head]
            if (is_debug_target(".attn.q")) {
                return q;
            }
            if (is_debug_target(".attn.k")) {
                return k;
            }
            if (is_debug_target(".attn.v")) {
                return v;
            }
            if (is_debug_target(".attn.q_rope")) {
                return Rope::apply_rope(ctx->ggml_ctx, q, pe, false, ctx->backend);
            }
            if (is_debug_target(".attn.k_rope")) {
                return Rope::apply_rope(ctx->ggml_ctx, k, pe, false, ctx->backend);
            }

            const ggml_type attention_output_type = v->type;
            x = window_sdpa_attention(ctx, q, k, v, pe, cu_window_seqlens);
            if (x == nullptr) {
                x = Rope::attention(ctx, q, k, v, pe, mask, 1.f, false);  // [N, n_token, hidden_size]
            }
            if (qwen_align_diffusers_vision_dtype_enabled() &&
                (attention_output_type == GGML_TYPE_F16 || attention_output_type == GGML_TYPE_BF16) &&
                x->type != attention_output_type) {
                x = ggml_cast(ctx->ggml_ctx, x, attention_output_type);
            }
            x = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".attn.preproj")) {
                return x;
            }

            x = proj->forward(ctx, x);  // [N, n_token, hidden_size]
            x = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".attn.out")) {
                return x;
            }
            return x;
        }
    };

    struct VisionBlock : public GGMLBlock {
    protected:
        LLMVisionArch arch_;

        ggml_tensor* forward_norm(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                if (x->type != GGML_TYPE_F32) {
                    x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
                }
                return std::dynamic_pointer_cast<LayerNorm>(blocks[name])->forward(ctx, x);
            }
            return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, x);
        }

    public:
        VisionBlock(bool llama_cpp_style,
                    LLMVisionArch arch,
                    int64_t hidden_size,
                    int64_t intermediate_size,
	                    int num_heads,
	                    float eps = 1e-6f)
            : arch_(arch) {
            const bool diffusers_dtype = qwen_align_diffusers_vision_dtype_enabled();
	            blocks["attn"]  = std::shared_ptr<GGMLBlock>(new VisionAttention(llama_cpp_style, arch_, hidden_size, num_heads));
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                blocks["mlp"] = std::shared_ptr<GGMLBlock>(new VisionMLP(arch_, hidden_size, intermediate_size));
                blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps));
                blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps));
                return;
            }
	            blocks["mlp"]   = std::shared_ptr<GGMLBlock>(new MLP(hidden_size,
                                                                    intermediate_size,
                                                                    true,
                                                                    false,
                                                                    diffusers_dtype,
                                                                    diffusers_dtype));
	            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps, false, true));
	            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps, false, true));
	        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr,
                             const std::vector<int>* cu_window_seqlens = nullptr,
                             const std::string& debug_target = "",
                             const std::string& debug_prefix = "") {
            // x: [N, n_token, hidden_size]
            auto attn  = std::dynamic_pointer_cast<VisionAttention>(blocks["attn"]);
            auto is_debug_target = [&](const char* suffix) {
                return !debug_prefix.empty() && debug_target == debug_prefix + suffix;
            };

            auto residual = x;
            if (is_debug_target(".input")) {
                return x;
            }
            x             = forward_norm(ctx, "norm1", x);
            x             = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".norm1")) {
                return x;
            }
            x             = attn->forward(ctx, x, pe, mask, cu_window_seqlens, debug_target, debug_prefix);
            if (!debug_prefix.empty() && debug_target.rfind(debug_prefix + ".attn.", 0) == 0) {
                return x;
            }
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);
            x             = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".after_attn")) {
                return x;
            }

            residual = x;
            x        = forward_norm(ctx, "norm2", x);
            x        = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".norm2")) {
                return x;
            }
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                x = std::dynamic_pointer_cast<VisionMLP>(blocks["mlp"])->forward(ctx, x);
            } else {
                x = std::dynamic_pointer_cast<MLP>(blocks["mlp"])->forward(ctx, x, debug_target, debug_prefix);
            }
            if (!debug_prefix.empty() && debug_target.rfind(debug_prefix + ".mlp.", 0) == 0) {
                return x;
            }
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);
            x        = qwen_align_maybe_bf16_llm_roundtrip(ctx->ggml_ctx, x);
            if (is_debug_target(".after_mlp")) {
                return x;
            }

            return x;
        }
    };

    struct VisionModel : public GGMLBlock {
    protected:
        LLMVisionArch arch_;
        int num_layers;
        int spatial_merge_size;
        int num_grid_per_side;
        std::set<int> fullatt_block_indexes;
        std::vector<int> deepstack_visual_indexes;

    public:
        VisionModel(bool llama_cpp_style,
                    const LLMVisionParams& vision_params,
                    float eps                           = 1e-6f)
            : arch_(vision_params.arch), num_layers(vision_params.num_layers), spatial_merge_size(vision_params.spatial_merge_size),
              num_grid_per_side(vision_params.num_position_embeddings > 0 ? static_cast<int>(std::sqrt(vision_params.num_position_embeddings)) : 0),
              fullatt_block_indexes(vision_params.fullatt_block_indexes), deepstack_visual_indexes(vision_params.deepstack_visual_indexes) {
            blocks["patch_embed"] = std::shared_ptr<GGMLBlock>(new VisionPatchEmbed(llama_cpp_style, arch_, vision_params.patch_size,
                                                                                    vision_params.temporal_patch_size, vision_params.in_channels, vision_params.hidden_size));
            if (vision_params.num_position_embeddings > 0) {
                blocks["pos_embed"] = std::make_shared<Embedding>(vision_params.num_position_embeddings, vision_params.hidden_size);
            }
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new VisionBlock(llama_cpp_style,
                                                                                                   arch_, vision_params.hidden_size,
                                                                                                   vision_params.intermediate_size,
                                                                                                   vision_params.num_heads,
                                                                                                   eps));
            }
            blocks["merger"] = std::shared_ptr<GGMLBlock>(new PatchMerger(arch_, vision_params.out_hidden_size, vision_params.hidden_size, spatial_merge_size));
            for (size_t i = 0; i < deepstack_visual_indexes.size(); ++i) {
                blocks["deepstack_merger_list." + std::to_string(i)] = std::make_shared<Qwen3VLDeepStackMerger>(vision_params.out_hidden_size, vision_params.hidden_size, spatial_merge_size);
            }
        }

        std::shared_ptr<Embedding> pos_embedder() {
            auto it = blocks.find("pos_embed");
            if (it == blocks.end()) {
                return nullptr;
            }
            return std::dynamic_pointer_cast<Embedding>(it->second);
        }

        int get_num_grid_per_side() const {
            return num_grid_per_side;
        }

        int get_spatial_merge_size() const {
            return spatial_merge_size;
        }

        std::vector<ggml_tensor*> forward_outputs(GGMLRunnerContext* ctx,
                             ggml_tensor* pixel_values,
                             ggml_tensor* pe,
                             ggml_tensor* window_index,
                             ggml_tensor* window_inverse_index,
                             ggml_tensor* window_mask,
                             const std::vector<int>* cu_window_seqlens = nullptr,
                             ggml_tensor* pos_embeds = nullptr,
                             const std::string& debug_target = "") {
            // pixel_values: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            // window_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_inverse_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_mask: [grid_h*grid_w, grid_h*grid_w]
            auto patch_embed = std::dynamic_pointer_cast<VisionPatchEmbed>(blocks["patch_embed"]);
            auto merger      = std::dynamic_pointer_cast<PatchMerger>(blocks["merger"]);

            auto x = patch_embed->forward(ctx, pixel_values);
            sd::ggml_graph_cut::mark_graph_cut(x, "llm.vision.prelude", "x");
            if (pos_embeds != nullptr) {
                x = ggml_add(ctx->ggml_ctx, x, pos_embeds);
            }
            if (debug_target == "patch_embed") {
                return {x};
            }

            if (window_index != nullptr) {
                x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] * spatial_merge_size * spatial_merge_size, x->ne[1] / spatial_merge_size / spatial_merge_size, x->ne[2], x->ne[3]);
                x = ggml_get_rows(ctx->ggml_ctx, x, window_index);
                x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] / spatial_merge_size / spatial_merge_size, x->ne[1] * spatial_merge_size * spatial_merge_size, x->ne[2], x->ne[3]);
            }
            if (debug_target == "windowed") {
                return {x};
            }

            std::vector<ggml_tensor*> deepstack_outputs;
            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<VisionBlock>(blocks["blocks." + std::to_string(i)]);

                auto mask = window_mask;
                const std::vector<int>* layer_cu_window_seqlens = cu_window_seqlens;
                if (fullatt_block_indexes.find(i) != fullatt_block_indexes.end()) {
                    mask                    = nullptr;
                    layer_cu_window_seqlens = nullptr;
                }
                const std::string block_debug_prefix = "block" + std::to_string(i);
                x = block->forward(ctx, x, pe, mask, layer_cu_window_seqlens, debug_target, block_debug_prefix);
                if (debug_target.rfind(block_debug_prefix + ".", 0) == 0) {
                    return {x};
                }
                if (qwen_align_bf16_vision_activations_enabled()) {
                    x = qwen_align_bf16_roundtrip_to_f32(ctx->ggml_ctx, x);
                }
                auto deepstack_it = std::find(deepstack_visual_indexes.begin(), deepstack_visual_indexes.end(), i);
                if (deepstack_it != deepstack_visual_indexes.end()) {
                    size_t deepstack_index = static_cast<size_t>(std::distance(deepstack_visual_indexes.begin(), deepstack_it));
                    auto deepstack_merger = std::dynamic_pointer_cast<Qwen3VLDeepStackMerger>(blocks["deepstack_merger_list." + std::to_string(deepstack_index)]);
                    deepstack_outputs.push_back(deepstack_merger->forward(ctx, x));
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "llm.vision.blocks." + std::to_string(i), "x");
                if (debug_target == "block" + std::to_string(i)) {
                    return {x};
                }
            }

            x = merger->forward(ctx, x, debug_target, "merger");
            if (debug_target.rfind("merger.", 0) == 0) {
                return {x};
            }
            if (qwen_align_bf16_vision_activations_enabled()) {
                x = qwen_align_bf16_roundtrip_to_f32(ctx->ggml_ctx, x);
            }
            sd::ggml_graph_cut::mark_graph_cut(x, "llm.vision.final", "x");
            if (debug_target == "merged") {
                return {x};
            }

            if (window_inverse_index != nullptr) {
                x = ggml_get_rows(ctx->ggml_ctx, x, window_inverse_index);
            }
            std::vector<ggml_tensor*> outputs = {x};
            outputs.insert(outputs.end(), deepstack_outputs.begin(), deepstack_outputs.end());
            return outputs;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* pixel_values,
                             ggml_tensor* pe,
                             ggml_tensor* window_index,
                             ggml_tensor* window_inverse_index,
                             ggml_tensor* window_mask,
                             const std::vector<int>* cu_window_seqlens = nullptr,
                             ggml_tensor* pos_embeds = nullptr,
                             const std::string& debug_target = "") {
            return forward_outputs(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask, cu_window_seqlens, pos_embeds, debug_target)[0];
        }
    };

    struct Attention : public GGMLBlock {
    protected:
        LLMArch arch;
        int head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;
        bool diffusers_dtype;
        int64_t max_position_embeddings;
        std::vector<float> rope_thetas;
        std::vector<float> rope_scales;

    public:
        Attention(const LLMParams& params)
            : arch(params.arch),
              num_heads(params.num_heads),
              num_kv_heads(params.num_kv_heads),
              qk_norm(params.qk_norm),
              diffusers_dtype(params.arch == LLMArch::QWEN2_5_VL &&
                              qwen_align_diffusers_text_dtype_enabled()),
              max_position_embeddings(params.max_position_embeddings),
              rope_thetas(params.rope_thetas),
              rope_scales(params.rope_scales) {
            head_dim = params.head_dim;
            blocks["q_proj"] = std::make_shared<Linear>(params.hidden_size,
                                                        num_heads * head_dim,
                                                        params.qkv_bias,
                                                        false,
                                                        diffusers_dtype,
                                                        1.f,
                                                        false,
                                                        false,
                                                        false);
            blocks["k_proj"] = std::make_shared<Linear>(params.hidden_size,
                                                        num_kv_heads * head_dim,
                                                        params.qkv_bias,
                                                        false,
                                                        diffusers_dtype,
                                                        1.f,
                                                        false,
                                                        false,
                                                        false);
            blocks["v_proj"] = std::make_shared<Linear>(params.hidden_size,
                                                        num_kv_heads * head_dim,
                                                        params.qkv_bias,
                                                        false,
                                                        diffusers_dtype,
                                                        1.f,
                                                        false,
                                                        false,
                                                        false);
            blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim,
                                                        params.hidden_size,
                                                        false,
                                                        false,
                                                        diffusers_dtype,
                                                        1.f,
                                                        false,
                                                        false,
                                                        diffusers_dtype);
            if (params.qk_norm) {
                if (params.arch == LLMArch::GEMMA3_12B) {
                    blocks["q_norm"] = std::make_shared<GemmaQKRMSNorm>(head_dim, params.rms_norm_eps);
                    blocks["k_norm"] = std::make_shared<GemmaQKRMSNorm>(head_dim, params.rms_norm_eps);
                } else {
                    blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
                    blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask = nullptr,
                             int rope_index = 0,
                             const std::string& debug_target = "",
                             const std::string& debug_prefix = "") {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);
            auto is_debug_target = [&](const char* suffix) {
                return !debug_prefix.empty() && debug_target == debug_prefix + suffix;
            };

            auto q = q_proj->forward(ctx, x);  // [N, n_token, num_heads*head_dim]
            auto k = k_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
            auto v = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
            if (is_debug_target(".attn.q_proj")) {
                return q;
            }
            if (is_debug_target(".attn.k_proj")) {
                return k;
            }
            if (is_debug_target(".attn.v_proj")) {
                return v;
            }

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                if (arch == LLMArch::GEMMA3_12B) {
                    auto q_norm = std::dynamic_pointer_cast<GemmaQKRMSNorm>(blocks["q_norm"]);
                    auto k_norm = std::dynamic_pointer_cast<GemmaQKRMSNorm>(blocks["k_norm"]);
                    q = q_norm->forward(ctx, q);
                    k = k_norm->forward(ctx, k);
                } else {
                    auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                    auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
                    q = q_norm->forward(ctx, q);
                    k = k_norm->forward(ctx, k);
                }
            }
            if (is_debug_target(".attn.q")) {
                return q;
            }
            if (is_debug_target(".attn.k")) {
                return k;
            }
            if (is_debug_target(".attn.v")) {
                return v;
            }

            const ggml_type q_rope_output_type = q->type;
            const ggml_type k_rope_output_type = k->type;
            const ggml_type attention_output_type = v->type;
            if (q->type == GGML_TYPE_BF16) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
            }
            if (k->type == GGML_TYPE_BF16) {
                k = ggml_cast(ctx->ggml_ctx, k, GGML_TYPE_F32);
            }

            if (arch == LLMArch::MISTRAL_SMALL_3_2) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::MINISTRAL_3_3B) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 262144, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 262144, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::QWEN3) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::GEMMA3_12B) {
                const size_t index = rope_index < static_cast<int>(rope_thetas.size()) ?
                                         static_cast<size_t>(rope_index) : 0;
                const float rope_theta = rope_thetas.empty() ? 1000000.f : rope_thetas[index];
                const float rope_scale = rope_scales.empty() ? 1.f : rope_scales[std::min(index, rope_scales.size() - 1)];
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, head_dim,
                                  GGML_ROPE_TYPE_NEOX, max_position_embeddings,
                                  rope_theta, 1.f / rope_scale, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, head_dim,
                                  GGML_ROPE_TYPE_NEOX, max_position_embeddings,
                                  rope_theta, 1.f / rope_scale, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::QWEN3_VL) {
                int sections[4] = {24, 20, 20, 0};
                q = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_IMROPE, 262144, 5000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_IMROPE, 262144, 5000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else {
                int sections[4] = {16, 24, 24, 0};
                q               = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k               = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            }
            if (diffusers_dtype) {
                if ((q_rope_output_type == GGML_TYPE_F16 || q_rope_output_type == GGML_TYPE_BF16) &&
                    q->type != q_rope_output_type) {
                    q = ggml_cast(ctx->ggml_ctx, q, q_rope_output_type);
                }
                if ((k_rope_output_type == GGML_TYPE_F16 || k_rope_output_type == GGML_TYPE_BF16) &&
                    k->type != k_rope_output_type) {
                    k = ggml_cast(ctx->ggml_ctx, k, k_rope_output_type);
                }
            }
            if (is_debug_target(".attn.q_rope")) {
                return ggml_cont(ctx->ggml_ctx, q);
            }
            if (is_debug_target(".attn.k_rope")) {
                return ggml_cont(ctx->ggml_ctx, k);
            }

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);      // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);      // [N*num_kv_heads, n_token, head_dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx,
                                       ctx->backend,
                                       q,
                                       k,
                                       v,
                                       num_heads,
                                       attention_mask,
                                       true,
                                       false,
                                       1.0f,
                                       true,
                                       false,
                                       -1,
                                       -1,
                                       false);  // [N, n_token, hidden_size]
            if (diffusers_dtype &&
                (attention_output_type == GGML_TYPE_F16 || attention_output_type == GGML_TYPE_BF16) &&
                x->type != attention_output_type) {
                x = ggml_cast(ctx->ggml_ctx, x, attention_output_type);
            }
            if (is_debug_target(".attn.preproj")) {
                return x;
            }

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            if (is_debug_target(".attn.out")) {
                return x;
            }
            return x;
        }
    };

    struct TransformerBlock : public GGMLBlock {
    protected:
        LLMArch arch;
        int sliding_attention = 0;
        std::string post_attention_norm_name;
        std::string pre_ffw_norm_name;
        std::string post_ffw_norm_name;

    public:
        TransformerBlock(const LLMParams& params, int layer_index)
            : arch(params.arch) {
            const bool cast_rms_output_to_input_type = params.arch == LLMArch::QWEN2_5_VL;
            const bool diffusers_dtype = params.arch == LLMArch::QWEN2_5_VL &&
                                         qwen_align_diffusers_text_dtype_enabled();
            if (params.arch == LLMArch::GEMMA3_12B) {
                post_attention_norm_name = "post_attention_norm";
                pre_ffw_norm_name        = "post_attention_layernorm";
                post_ffw_norm_name       = "post_ffw_norm";
                if (!params.sliding_attention.empty()) {
                    sliding_attention = params.sliding_attention[static_cast<size_t>(layer_index) % params.sliding_attention.size()];
                }
            } else {
                pre_ffw_norm_name = "post_attention_layernorm";
            }
            blocks["self_attn"] = std::make_shared<Attention>(params);
            blocks["mlp"]       = std::make_shared<MLP>(params.hidden_size,
                                                          params.intermediate_size,
                                                          false,
                                                          false,
                                                          diffusers_dtype,
                                                          diffusers_dtype,
                                                          params.mlp_activation);
            blocks["input_layernorm"] = std::make_shared<RMSNorm>(params.hidden_size,
                                                                    params.rms_norm_eps,
                                                                    false,
                                                                    cast_rms_output_to_input_type);
            blocks[pre_ffw_norm_name] = std::make_shared<RMSNorm>(params.hidden_size,
                                                                  params.rms_norm_eps,
                                                                  false,
                                                                  cast_rms_output_to_input_type);
            if (!post_attention_norm_name.empty()) {
                blocks[post_attention_norm_name] = std::make_shared<RMSNorm>(params.hidden_size,
                                                                              params.rms_norm_eps,
                                                                              false,
                                                                              cast_rms_output_to_input_type);
            }
            if (!post_ffw_norm_name.empty()) {
                blocks[post_ffw_norm_name] = std::make_shared<RMSNorm>(params.hidden_size,
                                                                       params.rms_norm_eps,
                                                                       false,
                                                                       cast_rms_output_to_input_type);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask = nullptr,
                             ggml_tensor* sliding_attention_mask = nullptr,
                             const std::string& debug_target = "",
                             const std::string& debug_prefix = "") {
            // x: [N, n_token, hidden_size]
            auto self_attn                = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto pre_ffw_norm = std::dynamic_pointer_cast<RMSNorm>(blocks[pre_ffw_norm_name]);
            std::shared_ptr<RMSNorm> post_attention_norm = nullptr;
            std::shared_ptr<RMSNorm> post_ffw_norm = nullptr;
            if (!post_attention_norm_name.empty()) {
                post_attention_norm = std::dynamic_pointer_cast<RMSNorm>(blocks[post_attention_norm_name]);
            }
            if (!post_ffw_norm_name.empty()) {
                post_ffw_norm = std::dynamic_pointer_cast<RMSNorm>(blocks[post_ffw_norm_name]);
            }
            ggml_tensor* block_attention_mask = attention_mask;
            int rope_index = 0;
            if (arch == LLMArch::GEMMA3_12B && sliding_attention > 0) {
                block_attention_mask = sliding_attention_mask;
                rope_index = 1;
            }
            auto is_debug_target = [&](const char* suffix) {
                return !debug_prefix.empty() && debug_target == debug_prefix + suffix;
            };

            auto residual = x;
            if (is_debug_target(".input")) {
                return x;
            }
            x             = input_layernorm->forward(ctx, x);
            if (is_debug_target(".norm1")) {
                return x;
            }
            x             = self_attn->forward(ctx, x, input_pos, block_attention_mask, rope_index, debug_target, debug_prefix);
            if (!debug_prefix.empty() && debug_target.rfind(debug_prefix + ".attn.", 0) == 0) {
                return x;
            }
            if (post_attention_norm != nullptr) {
                x = post_attention_norm->forward(ctx, x);
            }
            x = ggml_add_inplace(ctx->ggml_ctx, x, residual);
            if (is_debug_target(".after_attn")) {
                return x;
            }

            residual = x;
            x        = pre_ffw_norm->forward(ctx, x);
            if (is_debug_target(".norm2")) {
                return x;
            }
            x        = mlp->forward(ctx, x, debug_target, debug_prefix);
            if (!debug_prefix.empty() && debug_target.rfind(debug_prefix + ".mlp.", 0) == 0) {
                return x;
            }
            if (post_ffw_norm != nullptr) {
                x = post_ffw_norm->forward(ctx, x);
            }
            x = ggml_add_inplace(ctx->ggml_ctx, x, residual);
            if (is_debug_target(".after_mlp")) {
                return x;
            }

            return x;
        }
    };

    struct TextModel : public GGMLBlock {
    protected:
        int64_t num_layers;
        int64_t hidden_size;
        bool diffusers_text_dtype;
        bool final_norm;
        bool normalize_input;

        static ggml_tensor* add_deepstack_image_embeds(GGMLRunnerContext* ctx,
                                                       ggml_tensor* x,
                                                       const std::vector<std::pair<int, ggml_tensor*>>& image_embeds) {
            if (image_embeds.empty()) {
                return x;
            }
            GGML_ASSERT(x->ne[2] == 1);
            auto raw_x = x->type == image_embeds[0].second->type ? x : ggml_cast(ctx->ggml_ctx, x, image_embeds[0].second->type);
            int64_t token_start = 0;
            ggml_tensor* output = nullptr;
            for (const auto& [index, image_embed] : image_embeds) {
                GGML_ASSERT(index >= token_start);
                GGML_ASSERT(index + image_embed->ne[1] <= raw_x->ne[1]);
                if (index > token_start) {
                    auto text_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, token_start, index);
                    output = output == nullptr ? text_embed : ggml_concat(ctx->ggml_ctx, output, text_embed, 1);
                }
                auto visual_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, index, index + image_embed->ne[1]);
                visual_embed = ggml_add(ctx->ggml_ctx, visual_embed, image_embed);
                output = output == nullptr ? visual_embed : ggml_concat(ctx->ggml_ctx, output, visual_embed, 1);
                token_start = index + image_embed->ne[1];
            }
            if (token_start < raw_x->ne[1]) {
                auto text_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, token_start, raw_x->ne[1]);
                output = output == nullptr ? text_embed : ggml_concat(ctx->ggml_ctx, output, text_embed, 1);
            }
            GGML_ASSERT(output != nullptr && output->ne[1] == raw_x->ne[1]);
            return output;
        }

    public:
        TextModel(const LLMParams& params)
            : num_layers(params.num_layers),
              hidden_size(params.hidden_size),
              diffusers_text_dtype(params.arch == LLMArch::QWEN2_5_VL &&
                                   qwen_align_diffusers_text_dtype_enabled()),
              final_norm(params.final_norm),
              normalize_input(params.normalize_input) {
            const bool cast_rms_output_to_input_type = params.arch == LLMArch::QWEN2_5_VL;
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(params.vocab_size, params.hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new TransformerBlock(params, i));
            }
            if (final_norm) {
                blocks["norm"] = std::shared_ptr<GGMLBlock>(
                    new RMSNorm(params.hidden_size, params.rms_norm_eps, false, cast_rms_output_to_input_type));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             ggml_tensor* sliding_attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             const std::vector<std::vector<std::pair<int, ggml_tensor*>>>& deepstack_image_embeds,
                             std::set<int> out_layers,
                             const std::string& debug_target = "") {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]

            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = final_norm
                                    ? std::dynamic_pointer_cast<RMSNorm>(blocks["norm"])
                                    : nullptr;

            auto x = embed_tokens->forward(ctx, input_ids);
            if (normalize_input) {
                x = ggml_ext_scale(ctx->ggml_ctx, x, std::sqrt(static_cast<float>(hidden_size)), true);
            }
            if (diffusers_text_dtype && x->type != GGML_TYPE_BF16) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_BF16);
            }
            sd::ggml_graph_cut::mark_graph_cut(x, "llm.text.prelude", "x");
            if (debug_target == "input_embed") {
                return x;
            }

            std::vector<ggml_tensor*> intermediate_outputs;

            if (image_embeds.size() > 0) {
                GGML_ASSERT(x->ne[2] == 1);  // N == 1

                const ggml_type embed_type  = diffusers_text_dtype ? x->type : image_embeds[0].second->type;
                const ggml_type concat_type = diffusers_text_dtype ? GGML_TYPE_F32 : embed_type;
                auto raw_x                  = x->type == concat_type ? x : ggml_cast(ctx->ggml_ctx, x, concat_type);
                int64_t txt_token_start = 0;
                int64_t txt_token_end   = 0;

                ggml_tensor* input_embed = nullptr;

                for (int i = 0; i < image_embeds.size(); i++) {
                    if (i == 0) {
                        txt_token_start = 0;
                    } else {
                        txt_token_start = image_embeds[i - 1].first + image_embeds[i - 1].second->ne[1];
                    }
                    txt_token_end = image_embeds[i].first;

                    auto txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);
                    if (input_embed == nullptr) {
                        input_embed = txt_embed;
                    } else {
                        input_embed = ggml_concat(ctx->ggml_ctx, input_embed, txt_embed, 1);
                    }

                    auto image_embed = image_embeds[i].second;
                    if (image_embed->type != raw_x->type) {
                        image_embed = ggml_cast(ctx->ggml_ctx, image_embed, raw_x->type);
                    }
                    input_embed      = ggml_concat(ctx->ggml_ctx, input_embed, image_embed, 1);
                }

                txt_token_start = image_embeds[image_embeds.size() - 1].first + image_embeds[image_embeds.size() - 1].second->ne[1];
                txt_token_end   = raw_x->ne[1];

                auto final_txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);

                input_embed = ggml_concat(ctx->ggml_ctx, input_embed, final_txt_embed, 1);
                GGML_ASSERT(raw_x->ne[1] == input_embed->ne[1]);

                if (diffusers_text_dtype && input_embed->type != embed_type) {
                    input_embed = ggml_cast(ctx->ggml_ctx, input_embed, embed_type);
                }
                x = input_embed;
            }

            if (out_layers.find(0) != out_layers.end()) {
                auto input_embed = x;
                if (out_layers.size() > 1) {
                    input_embed = ggml_cont(ctx->ggml_ctx, input_embed);
                    if (input_embed->type != GGML_TYPE_F32) {
                        input_embed = ggml_cast(ctx->ggml_ctx, input_embed, GGML_TYPE_F32);
                    }
                }
                intermediate_outputs.push_back(input_embed);
            }

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);

                const std::string block_debug_prefix = "block" + std::to_string(i);
                x = block->forward(ctx, x, input_pos, attention_mask, sliding_attention_mask, debug_target, block_debug_prefix);
                if (debug_target.rfind(block_debug_prefix + ".", 0) == 0) {
                    return x;
                }
                if (i < static_cast<int>(deepstack_image_embeds.size())) {
                    x = add_deepstack_image_embeds(ctx, x, deepstack_image_embeds[static_cast<size_t>(i)]);
                }
                if (out_layers.size() > 1) {
                    x = ggml_cont(ctx->ggml_ctx, x);
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "llm.text.layers." + std::to_string(i), "x");
                if (debug_target == block_debug_prefix) {
                    return x;
                }
                if (out_layers.find(i + 1) != out_layers.end()) {
                    auto layer_output = x;
                    if (out_layers.size() > 1 && layer_output->type != GGML_TYPE_F32) {
                        layer_output = ggml_cast(ctx->ggml_ctx, layer_output, GGML_TYPE_F32);
                    }
                    intermediate_outputs.push_back(layer_output);
                }
            }

            if (norm != nullptr && out_layers.find(static_cast<int>(num_layers + 1)) != out_layers.end()) {
                intermediate_outputs.push_back(norm->forward(ctx, x));
            } else if (intermediate_outputs.empty() && final_norm) {
                x = norm->forward(ctx, x);
                if (debug_target == "final_norm") {
                    return x;
                }
            }

            if (!intermediate_outputs.empty()) {
                x = intermediate_outputs[0];
                for (int i = 1; i < intermediate_outputs.size(); i++) {
                    x = ggml_concat(ctx->ggml_ctx, x, intermediate_outputs[i], 0);
                }
            }
            return x;
        }
    };

    struct LLM : public GGMLBlock {
        bool enable_vision;
        LLMParams params;

    public:
        LLM() = default;
        LLM(LLMParams params, bool enable_vision = false, bool llama_cpp_style = false)
            : enable_vision(enable_vision), params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new TextModel(params));
            if (enable_vision) {
                blocks["visual"] = std::shared_ptr<GGMLBlock>(new VisionModel(llama_cpp_style, params.vision));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             ggml_tensor* sliding_attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             const std::vector<std::vector<std::pair<int, ggml_tensor*>>>& deepstack_image_embeds,
                             std::set<int> out_layers,
                             const std::string& debug_target = "") {
            // input_ids: [N, n_token]
            auto model = std::dynamic_pointer_cast<TextModel>(blocks["model"]);

            auto x = model->forward(ctx, input_ids, input_pos, attention_mask, sliding_attention_mask,
                                    image_embeds, deepstack_image_embeds, out_layers, debug_target);
            return x;
        }

        std::shared_ptr<VisionModel> vision_model() {
            GGML_ASSERT(enable_vision);
            return std::dynamic_pointer_cast<VisionModel>(blocks["visual"]);
        }

        ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                    ggml_tensor* pixel_values,
                                    ggml_tensor* pe,
                                    ggml_tensor* window_index,
                                    ggml_tensor* window_inverse_index,
                                    ggml_tensor* window_mask,
                                    const std::vector<int>* cu_window_seqlens = nullptr,
                                    ggml_tensor* pos_embeds = nullptr,
                                    const std::string& debug_target = "") {
            GGML_ASSERT(enable_vision);
            return vision_model()->forward(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask, cu_window_seqlens, pos_embeds, debug_target);
        }

        std::vector<ggml_tensor*> vision_forward_outputs(GGMLRunnerContext* ctx,
                                                          ggml_tensor* pixel_values,
                                                          ggml_tensor* pe,
                                                          ggml_tensor* window_index,
                                                          ggml_tensor* window_inverse_index,
                                                          ggml_tensor* window_mask,
                                                          ggml_tensor* pos_embeds = nullptr) {
            GGML_ASSERT(enable_vision);
            return vision_model()->forward_outputs(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask, nullptr, pos_embeds);
        }
    };

    struct LLMRunner : public GGMLRunner {
        LLMParams params;
        bool enable_vision;
        LLM model;

        std::vector<int> input_pos_vec;
        std::vector<float> attention_mask_vec;
        std::vector<float> sliding_attention_mask_vec;
        std::vector<float> window_mask_vec;
        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<int> cu_window_seqlens_vec;
        std::vector<float> pe_vec;
        std::array<std::vector<int32_t>, 4> pos_embed_idx_data;
        std::array<std::vector<float>, 4> pos_embed_weight_data;

        LLMRunner(LLMArch arch,
                  ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool enable_vision_ = false)
            : GGMLRunner(backend, offload_params_to_cpu), enable_vision(enable_vision_) {
            params.arch = arch;
            if (arch == LLMArch::MISTRAL_SMALL_3_2 || arch == LLMArch::MINISTRAL_3_3B) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.rms_norm_eps = 1e-5f;
            } else if (arch == LLMArch::QWEN3 || arch == LLMArch::QWEN3_VL) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.qk_norm      = true;
                params.rms_norm_eps = 1e-6f;
                if (arch == LLMArch::QWEN3_VL) {
                    params.vision.arch = LLMVisionArch::QWEN3_VL;
                }
            } else if (arch == LLMArch::GEMMA3_12B) {
                params.head_dim                = 256;
                params.num_heads               = 16;
                params.num_kv_heads            = 8;
                params.qkv_bias                = false;
                params.qk_norm                 = true;
                params.rms_norm_eps            = 1e-6f;
                params.normalize_input         = true;
                params.max_position_embeddings = 131072;
                params.mlp_activation          = MLPActivation::GELU_TANH;
                params.rope_thetas             = {1000000.f, 10000.f};
                params.rope_scales             = {8.f, 1.f};
                params.sliding_attention       = {1024, 1024, 1024, 1024, 1024, 0};
            }
            bool have_vision_weight = false;
            bool llama_cpp_style    = false;
            int detected_vision_layers = 0;
            params.num_layers       = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("visual.");
                if (pos != std::string::npos) {
                    have_vision_weight = true;
                    if (contains(tensor_name, "attn.q_proj")) {
                        llama_cpp_style = true;
                    }
                    if (contains(tensor_name, "visual.patch_embed.proj.weight")) {
                        params.vision.patch_size = static_cast<int>(pair.second.ne[0]);
                    }
                    if (contains(tensor_name, "visual.patch_embed.proj.bias")) {
                        params.vision.hidden_size = pair.second.ne[0];
                    }
                    if (contains(tensor_name, "visual.pos_embed.weight")) {
                        params.vision.hidden_size             = pair.second.ne[0];
                        params.vision.num_position_embeddings = static_cast<int>(pair.second.ne[1]);
                    }
                    if (contains(tensor_name, "visual.blocks.")) {
                        auto items = split_string(tensor_name.substr(pos), '.');
                        if (items.size() > 2) {
                            detected_vision_layers = std::max(detected_vision_layers, std::atoi(items[2].c_str()) + 1);
                        }
                    }
                    if (contains(tensor_name, "visual.blocks.0.mlp.linear_fc1.weight") ||
                        contains(tensor_name, "visual.blocks.0.mlp.gate_proj.weight")) {
                        params.vision.intermediate_size = pair.second.ne[1];
                    }
                    if (contains(tensor_name, "visual.merger.linear_fc2.weight") ||
                        contains(tensor_name, "visual.merger.mlp.2.weight")) {
                        params.vision.out_hidden_size = pair.second.ne[1];
                    }
                    continue;
                }
                pos = tensor_name.find("layers.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > params.num_layers) {
                            params.num_layers = block_index + 1;
                        }
                    }
                }
                if (contains(tensor_name, "embed_tokens.weight")) {
                    params.hidden_size = pair.second.ne[0];
                    params.vocab_size  = pair.second.ne[1];
                }
                if (contains(tensor_name, "layers.0.mlp.gate_proj.weight")) {
                    params.intermediate_size = pair.second.ne[1];
                }
            }
            if ((arch == LLMArch::QWEN3 || arch == LLMArch::QWEN3_VL) && params.num_layers == 28) {  // Qwen3 2B
                params.num_heads = 16;
            }
            if ((arch == LLMArch::QWEN3 || arch == LLMArch::QWEN3_VL) && params.num_layers == 50 && params.hidden_size == 5120) {
                params.num_heads = 64;
                params.final_norm = false;
            }
            if (detected_vision_layers > 0) {
                params.vision.num_layers = detected_vision_layers;
            }
            if (arch == LLMArch::QWEN3_VL) {
                if (params.vision.num_layers == 24) {
                    params.vision.deepstack_visual_indexes = {5, 11, 17};
                } else if (params.vision.num_layers == 27) {
                    params.vision.deepstack_visual_indexes = {8, 16, 24};
                }
            }
            LOG_DEBUG("llm vision: arch=%d layers=%d hidden=%" PRId64 " heads=%d patch=%d temporal=%d merge=%d",
                      static_cast<int>(params.vision.arch), params.vision.num_layers, params.vision.hidden_size,
                      params.vision.num_heads, params.vision.patch_size, params.vision.temporal_patch_size,
                      params.vision.spatial_merge_size);
            LOG_DEBUG("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
                      params.num_layers,
                      params.vocab_size,
                      params.hidden_size,
                      params.intermediate_size);
            if (enable_vision && !have_vision_weight) {
                LOG_WARN("no vision weights detected, vision disabled");
                enable_vision = false;
            }
            if (enable_vision) {
                LOG_DEBUG("enable llm vision");
                if (llama_cpp_style) {
                    LOG_DEBUG("llama.cpp style vision weight");
                }
            }
            model = LLM(params, enable_vision, llama_cpp_style);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return llm_arch_to_str[static_cast<int>(params.arch)];
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             ggml_tensor* sliding_attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             const std::vector<std::vector<std::pair<int, ggml_tensor*>>>& deepstack_image_embeds,
                             std::set<int> out_layers,
                             const std::string& debug_target = "") {
            auto hidden_states = model.forward(ctx, input_ids, input_pos, attention_mask, sliding_attention_mask,
                                               image_embeds, deepstack_image_embeds, out_layers, debug_target);  // [N, n_token, hidden_size]
            return hidden_states;
        }

        ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                    ggml_tensor* pixel_values,
                                    ggml_tensor* input_pos,
                                    ggml_tensor* window_index,
                                    ggml_tensor* window_inverse_index,
                                    ggml_tensor* window_mask,
                                    const std::string& debug_target = "") {
            auto hidden_states = model.vision_forward(ctx, pixel_values, input_pos, window_index, window_inverse_index, window_mask, &cu_window_seqlens_vec, nullptr, debug_target);
            return hidden_states;
        }

        bool build_qwen25_vl_mrope_positions(int64_t n_tokens,
                                              const std::vector<LLMImageEmbedInfo>& image_embed_infos) {
            if (n_tokens <= 0 || image_embed_infos.empty()) {
                return false;
            }

            std::vector<LLMImageEmbedInfo> infos = image_embed_infos;
            std::sort(infos.begin(),
                      infos.end(),
                      [](const LLMImageEmbedInfo& a, const LLMImageEmbedInfo& b) {
                          return a.token_index < b.token_index;
                      });

            input_pos_vec.resize(static_cast<size_t>(n_tokens) * 4);
            for (int64_t token = 0; token < n_tokens; ++token) {
                input_pos_vec[static_cast<size_t>(token)] = static_cast<int>(token);
                input_pos_vec[static_cast<size_t>(n_tokens + token)] = static_cast<int>(token);
                input_pos_vec[static_cast<size_t>(2 * n_tokens + token)] = static_cast<int>(token);
                input_pos_vec[static_cast<size_t>(3 * n_tokens + token)] = 0;
            }

            int64_t offset = 0;
            for (const auto& info : infos) {
                const int64_t index = static_cast<int64_t>(info.token_index);
                const int64_t size = info.token_count;
                const int64_t merge = std::max<int64_t>(1, params.vision.spatial_merge_size);
                if (info.grid_h <= 0 || info.grid_w <= 0 ||
                    info.grid_h % merge != 0 || info.grid_w % merge != 0) {
                    return false;
                }
                const int64_t grid_h = info.grid_h / merge;
                const int64_t grid_w = info.grid_w / merge;
                const int64_t end = index + size;
                if (index < 0 || end > n_tokens || grid_h <= 0 || grid_w <= 0 ||
                    size != grid_h * grid_w) {
                    return false;
                }

                const int64_t len_max = std::max(grid_h, grid_w);
                const int64_t next_pos = index + len_max + offset;
                for (int64_t token = end; token < n_tokens; ++token) {
                    const int64_t pos = next_pos + token - end;
                    input_pos_vec[static_cast<size_t>(token)] = static_cast<int>(pos);
                    input_pos_vec[static_cast<size_t>(n_tokens + token)] = static_cast<int>(pos);
                    input_pos_vec[static_cast<size_t>(2 * n_tokens + token)] = static_cast<int>(pos);
                }
                for (int64_t token = 0; token < size; ++token) {
                    input_pos_vec[static_cast<size_t>(index + token)] = static_cast<int>(index + offset);
                    input_pos_vec[static_cast<size_t>(n_tokens + index + token)] =
                        static_cast<int>(index + offset + token / grid_w);
                    input_pos_vec[static_cast<size_t>(2 * n_tokens + index + token)] =
                        static_cast<int>(index + offset + token % grid_w);
                }
                offset += len_max - size;
            }
            if (qwen_align_debug_enabled()) {
                LOG_INFO("qwen-align mrope_positions: n_tokens=%" PRId64 " axes=4", n_tokens);
                qwen_align_log_int_samples("mrope.t.head", input_pos_vec, 0, 80);
                qwen_align_log_int_samples("mrope.h.head", input_pos_vec, n_tokens, 80);
                qwen_align_log_int_samples("mrope.w.head", input_pos_vec, 2 * n_tokens, 80);
                for (size_t i = 0; i < infos.size(); ++i) {
                    const auto& info = infos[i];
                    LOG_INFO("qwen-align mrope.image_info[%zu]: token_index=%d token_count=%" PRId64 " grid=[%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                             i,
                             info.token_index,
                             info.token_count,
                             info.grid_t,
                             info.grid_h,
                             info.grid_w);
                    qwen_align_log_int_samples("mrope.t.before_image",
                                               input_pos_vec,
                                               info.token_index - 8,
                                               16);
                    qwen_align_log_int_samples("mrope.h.before_image",
                                               input_pos_vec,
                                               n_tokens + info.token_index - 8,
                                               16);
                    qwen_align_log_int_samples("mrope.w.before_image",
                                               input_pos_vec,
                                               2 * n_tokens + info.token_index - 8,
                                               16);
                    qwen_align_log_int_samples("mrope.t.after_image",
                                               input_pos_vec,
                                               info.token_index + info.token_count - 8,
                                               24);
                    qwen_align_log_int_samples("mrope.h.after_image",
                                               input_pos_vec,
                                               n_tokens + info.token_index + info.token_count - 8,
                                               24);
                    qwen_align_log_int_samples("mrope.w.after_image",
                                               input_pos_vec,
                                               2 * n_tokens + info.token_index + info.token_count - 8,
                                               24);
                }
                qwen_align_log_int_samples("mrope.t.tail", input_pos_vec, n_tokens - 32, 32);
                qwen_align_log_int_samples("mrope.h.tail", input_pos_vec, 2 * n_tokens - 32, 32);
                qwen_align_log_int_samples("mrope.w.tail", input_pos_vec, 3 * n_tokens - 32, 32);
            }
            return true;
        }

        ggml_cgraph* build_graph(const sd::Tensor<int32_t>& input_ids_tensor,
                                 const sd::Tensor<float>& attention_mask_tensor,
                                 const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds_tensor,
                                 const std::vector<std::vector<std::pair<int, sd::Tensor<float>>>>& deepstack_image_embeds_tensor,
                                 std::set<int> out_layers,
                                 const std::vector<LLMImageEmbedInfo>& image_embed_infos = {},
                                 const std::string& debug_target = "") {
            ggml_cgraph* gf        = new_graph_custom(LLM_GRAPH_SIZE);
            ggml_tensor* input_ids = make_input(input_ids_tensor);
            std::vector<std::pair<int, ggml_tensor*>> image_embeds;
            image_embeds.reserve(image_embeds_tensor.size());
            for (const auto& [idx, embed_tensor] : image_embeds_tensor) {
                ggml_tensor* embed = make_input(embed_tensor);
                image_embeds.emplace_back(idx, embed);
            }
            std::vector<std::vector<std::pair<int, ggml_tensor*>>> deepstack_image_embeds(deepstack_image_embeds_tensor.size());
            for (size_t layer = 0; layer < deepstack_image_embeds_tensor.size(); ++layer) {
                for (const auto& [idx, embed_tensor] : deepstack_image_embeds_tensor[layer]) {
                    deepstack_image_embeds[layer].emplace_back(idx, make_input(embed_tensor));
                }
            }

            int64_t n_tokens = input_ids->ne[0];
            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 ||
                params.arch == LLMArch::MINISTRAL_3_3B ||
                params.arch == LLMArch::QWEN3 ||
                params.arch == LLMArch::GEMMA3_12B) {
                input_pos_vec.resize(n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i] = i;
                }
            } else {
                const bool have_mrope_positions = build_qwen25_vl_mrope_positions(n_tokens, image_embed_infos);
                if (!have_mrope_positions) {
                    input_pos_vec.resize(n_tokens * 4);
                    for (int i = 0; i < n_tokens; ++i) {
                        input_pos_vec[i]                = i;
                        input_pos_vec[n_tokens + i]     = i;
                        input_pos_vec[2 * n_tokens + i] = i;
                        input_pos_vec[3 * n_tokens + i] = 0;
                    }
                }
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx,
                                                GGML_TYPE_I32,
                                                input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            ggml_tensor* attention_mask = nullptr;
            if (!attention_mask_tensor.empty()) {
                attention_mask = make_input(attention_mask_tensor);
            } else {
                attention_mask_vec.resize(n_tokens * n_tokens);
                for (int i0 = 0; i0 < n_tokens; i0++) {
                    for (int i1 = 0; i1 < n_tokens; i1++) {
                        float value = 0.f;
                        if (i0 > i1) {
                            value = -INFINITY;
                        }
                        attention_mask_vec[i1 * n_tokens + i0] = value;
                    }
                }
                attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            ggml_tensor* sliding_attention_mask = nullptr;
            if (params.arch == LLMArch::GEMMA3_12B && !params.sliding_attention.empty()) {
                int sliding_window = 0;
                for (int window : params.sliding_attention) {
                    sliding_window = std::max(sliding_window, window);
                }
                sliding_attention_mask_vec = attention_mask_vec;
                if (!attention_mask_tensor.empty()) {
                    GGML_ASSERT(attention_mask_tensor.numel() == n_tokens * n_tokens);
                    sliding_attention_mask_vec = attention_mask_tensor.values();
                }
                for (int i0 = 0; i0 < n_tokens; ++i0) {
                    for (int i1 = 0; i1 < n_tokens; ++i1) {
                        if (sliding_window > 0 && i0 + sliding_window <= i1) {
                            sliding_attention_mask_vec[static_cast<size_t>(i1 * n_tokens + i0)] = -INFINITY;
                        }
                    }
                }
                sliding_attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(sliding_attention_mask, sliding_attention_mask_vec.data());
            }

            auto runner_ctx = get_context();

            ggml_tensor* hidden_states = forward(&runner_ctx,
                                                 input_ids,
                                                 input_pos,
                                                 attention_mask,
                                                 sliding_attention_mask,
                                                 image_embeds,
                                                 deepstack_image_embeds,
                                                 out_layers,
                                                 debug_target);

            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<int32_t>& input_ids,
                                  const sd::Tensor<float>& attention_mask,
                                  const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                  std::set<int> out_layers,
                                  const std::vector<LLMImageEmbedInfo>& image_embed_infos = {},
                                  const std::string& debug_target = "",
                                  const std::vector<std::vector<std::pair<int, sd::Tensor<float>>>>& deepstack_image_embeds = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(input_ids, attention_mask, image_embeds, deepstack_image_embeds, out_layers, image_embed_infos, debug_target);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
        }

        int64_t get_num_image_tokens(int64_t t, int64_t h, int64_t w) {
            int64_t grid_t     = 1;
            int64_t grid_h     = h / params.vision.patch_size;
            int64_t grid_w     = w / params.vision.patch_size;
            int64_t llm_grid_h = grid_h / params.vision.spatial_merge_size;
            int64_t llm_grid_w = grid_w / params.vision.spatial_merge_size;
            return grid_t * grid_h * grid_w;
        }

        ggml_tensor* process_image(ggml_context* ctx, ggml_tensor* image) {
            // image: [C, H, W]
            // return: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw], grid_t == 1
            int64_t C  = image->ne[2];
            int64_t H  = image->ne[1];
            int64_t W  = image->ne[0];
            int64_t mh = params.vision.spatial_merge_size;
            int64_t mw = params.vision.spatial_merge_size;
            int64_t pt = params.vision.temporal_patch_size;
            int64_t ph = params.vision.patch_size;
            int64_t pw = params.vision.patch_size;

            image = ggml_reshape_4d(ctx, image, pw, mw, (W / mw / pw), H * C);                               // [C*H, (W/mw/pw), mw, pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [mw, C*H, (W/mw/pw), pw]
            image = ggml_reshape_4d(ctx, image, pw * (W / mw / pw), H, C, mw);                               // [mw, C, H, (W/mw/pw)*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [H, mw, C, (W/mw/pw)*pw]
            image = ggml_reshape_4d(ctx, image, pw, (W / mw / pw) * C * mw, ph, mh * (H / mh / ph));         // [(H/mh/ph)*mh, ph, mw*C*(W/mw/pw), pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh, mw*C*(W/mw/pw), ph, pw]
            image = ggml_reshape_4d(ctx, image, pw * ph, (W / mw / pw), C, mw * mh * (H / mh / ph));         // [(H/mh/ph)*mh*mw, C, (W/mw/pw), ph*pw]
            image = ggml_concat(ctx, image, image, 0);                                                       // [(H/mh/ph)*mh*mw, C, (W/mw/pw), pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh*mw, (W/mw/pw), C, pt*ph*pw]
            image = ggml_reshape_4d(ctx, image, pw * ph * pt * C, (W / mw / pw), mw * mh, (H / mh / ph));    // [(H/mh/ph), mh*mw, (W/mw/pw), C*pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph), (W/mw/pw), mh*mw, C*pt*ph*pw]
            image = ggml_reshape_2d(ctx, image, pw * ph * pt * C, mw * mh * (W / mw / pw) * (H / mh / ph));  // [(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            return image;
        }

        sd::Tensor<float> process_image_patches_host(const sd::Tensor<float>& image_tensor) {
            if (image_tensor.empty() || image_tensor.dim() < 3) {
                return {};
            }

            const int64_t width    = image_tensor.shape()[0];
            const int64_t height   = image_tensor.shape()[1];
            const int64_t channels = image_tensor.shape()[2];
            const int64_t frames   = image_tensor.dim() >= 4 ? image_tensor.shape()[3] : 1;
            const int64_t mh       = params.vision.spatial_merge_size;
            const int64_t mw       = params.vision.spatial_merge_size;
            const int64_t pt       = params.vision.temporal_patch_size;
            const int64_t ph       = params.vision.patch_size;
            const int64_t pw       = params.vision.patch_size;

            if (width <= 0 || height <= 0 || channels <= 0 || frames <= 0 ||
                width % (pw * mw) != 0 || height % (ph * mh) != 0) {
                return {};
            }

            const int64_t padded_frames = ((frames + pt - 1) / pt) * pt;
            const int64_t grid_t        = padded_frames / pt;
            const int64_t grid_h        = height / ph;
            const int64_t grid_w        = width / pw;
            const int64_t patch_dim     = channels * pt * ph * pw;
            const int64_t patch_count   = grid_t * grid_h * grid_w;

            sd::Tensor<float> patches({patch_dim, patch_count});
            const float* src = image_tensor.data();
            float* dst       = patches.data();

            auto src_at = [&](int64_t x, int64_t y, int64_t c, int64_t f) -> float {
                f = std::min<int64_t>(f, frames - 1);
                return src[x +
                           width * y +
                           width * height * c +
                           width * height * channels * f];
            };

            int64_t token = 0;
            for (int64_t gt = 0; gt < grid_t; ++gt) {
                for (int64_t gh_block = 0; gh_block < grid_h / mh; ++gh_block) {
                    for (int64_t gw_block = 0; gw_block < grid_w / mw; ++gw_block) {
                        for (int64_t merge_h = 0; merge_h < mh; ++merge_h) {
                            for (int64_t merge_w = 0; merge_w < mw; ++merge_w) {
                                int64_t feature = 0;
                                for (int64_t c = 0; c < channels; ++c) {
                                    for (int64_t tp = 0; tp < pt; ++tp) {
                                        const int64_t frame = gt * pt + tp;
                                        for (int64_t py = 0; py < ph; ++py) {
                                            const int64_t y = (gh_block * mh + merge_h) * ph + py;
                                            for (int64_t px = 0; px < pw; ++px) {
                                                const int64_t x = (gw_block * mw + merge_w) * pw + px;
                                                dst[feature + patch_dim * token] = src_at(x, y, c, frame);
                                                ++feature;
                                            }
                                        }
                                    }
                                }
                                ++token;
                            }
                        }
                    }
                }
            }
            GGML_ASSERT(token == patch_count);
            return patches;
        }

        sd::Tensor<float> process_video_block_patches_host(const sd::Tensor<float>& frames) {
            if (frames.empty() || frames.dim() != 5 ||
                frames.shape()[2] != params.vision.temporal_patch_size ||
                frames.shape()[3] != params.vision.in_channels ||
                frames.shape()[4] != 1) {
                return {};
            }

            const int64_t width       = frames.shape()[0];
            const int64_t height      = frames.shape()[1];
            const int64_t temporal    = frames.shape()[2];
            const int64_t channels    = frames.shape()[3];
            const int64_t patch       = params.vision.patch_size;
            const int64_t merge       = params.vision.spatial_merge_size;
            const int64_t grid_w      = width / patch;
            const int64_t grid_h      = height / patch;
            const int64_t feature     = channels * temporal * patch * patch;
            const int64_t token_count = grid_h * grid_w;
            if (width <= 0 || height <= 0 || grid_w <= 0 || grid_h <= 0 ||
                grid_w % merge != 0 || grid_h % merge != 0) {
                return {};
            }

            sd::Tensor<float> output({feature, token_count});
            int64_t token = 0;
            for (int64_t block_h = 0; block_h < grid_h / merge; ++block_h) {
                for (int64_t block_w = 0; block_w < grid_w / merge; ++block_w) {
                    for (int64_t inner_h = 0; inner_h < merge; ++inner_h) {
                        for (int64_t inner_w = 0; inner_w < merge; ++inner_w) {
                            int64_t patch_h = block_h * merge + inner_h;
                            int64_t patch_w = block_w * merge + inner_w;
                            int64_t offset  = 0;
                            for (int64_t c = 0; c < channels; ++c) {
                                for (int64_t t = 0; t < temporal; ++t) {
                                    for (int64_t y = 0; y < patch; ++y) {
                                        for (int64_t x = 0; x < patch; ++x) {
                                            output.index(offset++, token) = frames.index(patch_w * patch + x,
                                                                                        patch_h * patch + y,
                                                                                        t,
                                                                                        c,
                                                                                        0);
                                        }
                                    }
                                }
                            }
                            ++token;
                        }
                    }
                }
            }
            return output;
        }

        ggml_tensor* build_qwen3_vl_patch_pos_embeds(GGMLRunnerContext* runner_ctx,
                                                     std::shared_ptr<VisionModel> vision,
                                                     int grid_h,
                                                     int grid_w) {
            auto pos_embed = vision->pos_embedder();
            GGML_ASSERT(pos_embed != nullptr);
            const int num_grid_per_side = vision->get_num_grid_per_side();
            const int merge_size        = vision->get_spatial_merge_size();
            GGML_ASSERT(num_grid_per_side > 0);
            GGML_ASSERT(grid_h % merge_size == 0);
            GGML_ASSERT(grid_w % merge_size == 0);

            for (int index = 0; index < 4; ++index) {
                pos_embed_idx_data[index].clear();
                pos_embed_weight_data[index].clear();
                pos_embed_idx_data[index].reserve(static_cast<size_t>(grid_h * grid_w));
                pos_embed_weight_data[index].reserve(static_cast<size_t>(grid_h * grid_w));
            }

            const double max_index = static_cast<double>(num_grid_per_side - 1);
            for (int block_h = 0; block_h < grid_h / merge_size; ++block_h) {
                for (int block_w = 0; block_w < grid_w / merge_size; ++block_w) {
                    for (int inner_h = 0; inner_h < merge_size; ++inner_h) {
                        const int h = block_h * merge_size + inner_h;
                        const double h_pos = grid_h == 1 ? 0.0 : max_index * h / static_cast<double>(grid_h - 1);
                        const int h_floor = static_cast<int>(std::floor(h_pos));
                        const int h_ceil = std::min(h_floor + 1, num_grid_per_side - 1);
                        const double dh = h_pos - h_floor;
                        for (int inner_w = 0; inner_w < merge_size; ++inner_w) {
                            const int w = block_w * merge_size + inner_w;
                            const double w_pos = grid_w == 1 ? 0.0 : max_index * w / static_cast<double>(grid_w - 1);
                            const int w_floor = static_cast<int>(std::floor(w_pos));
                            const int w_ceil = std::min(w_floor + 1, num_grid_per_side - 1);
                            const double dw = w_pos - w_floor;
                            const int ids[4] = {
                                h_floor * num_grid_per_side + w_floor,
                                h_floor * num_grid_per_side + w_ceil,
                                h_ceil * num_grid_per_side + w_floor,
                                h_ceil * num_grid_per_side + w_ceil,
                            };
                            const float weights[4] = {
                                static_cast<float>((1.0 - dh) * (1.0 - dw)),
                                static_cast<float>((1.0 - dh) * dw),
                                static_cast<float>(dh * (1.0 - dw)),
                                static_cast<float>(dh * dw),
                            };
                            for (int index = 0; index < 4; ++index) {
                                pos_embed_idx_data[index].push_back(ids[index]);
                                pos_embed_weight_data[index].push_back(weights[index]);
                            }
                        }
                    }
                }
            }

            ggml_tensor* result = nullptr;
            for (int index = 0; index < 4; ++index) {
                auto idx_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, static_cast<int64_t>(pos_embed_idx_data[index].size()));
                set_backend_tensor_data(idx_tensor, pos_embed_idx_data[index].data());
                auto embed = pos_embed->forward(runner_ctx, idx_tensor);
                embed = ggml_reshape_2d(compute_ctx, embed, embed->ne[0], embed->ne[1] * embed->ne[2]);
                auto weight_tensor = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 1, static_cast<int64_t>(pos_embed_weight_data[index].size()));
                set_backend_tensor_data(weight_tensor, pos_embed_weight_data[index].data());
                embed = ggml_mul(compute_ctx, embed, weight_tensor);
                result = result == nullptr ? embed : ggml_add(compute_ctx, result, embed);
            }
            return result;
        }

        ggml_cgraph* build_encode_image_graph(const sd::Tensor<float>& pixel_values_tensor,
                                              int64_t image_width,
                                              int64_t image_height,
                                              const std::string& debug_target = "") {
            ggml_cgraph* gf    = new_graph_custom(LLM_GRAPH_SIZE);
            ggml_tensor* pixel_values = make_input(pixel_values_tensor);

            GGML_ASSERT(image_height % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);
            GGML_ASSERT(image_width % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);

            if (params.vision.arch == LLMVisionArch::QWEN3_VL) {
                const int grid_h = static_cast<int>(image_height) / params.vision.patch_size;
                const int grid_w = static_cast<int>(image_width) / params.vision.patch_size;
                const int head_dim = static_cast<int>(params.vision.hidden_size / params.vision.num_heads);
                auto runner_ctx = get_context();
                auto vision = model.vision_model();
                auto pos_embeds = build_qwen3_vl_patch_pos_embeds(&runner_ctx, vision, grid_h, grid_w);
                window_index_vec.resize(static_cast<size_t>((grid_h / params.vision.spatial_merge_size) *
                                                            (grid_w / params.vision.spatial_merge_size)));
                for (size_t index = 0; index < window_index_vec.size(); ++index) {
                    window_index_vec[index] = static_cast<int>(index);
                }
                pe_vec = Rope::gen_qwen2vl_pe(grid_h,
                                               grid_w,
                                               params.vision.spatial_merge_size,
                                               window_index_vec,
                                               10000,
                                               {head_dim / 2, head_dim / 2});
                const int pos_len = static_cast<int>(pe_vec.size() / head_dim / 2);
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
                set_backend_tensor_data(pe, pe_vec.data());
                auto outputs = model.vision_forward_outputs(&runner_ctx,
                                                            pixel_values,
                                                            pe,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            pos_embeds);
                ggml_build_forward_expand(gf, outputs[0]);
                return gf;
            }

            int grid_t                 = 1;
            int grid_h                 = static_cast<int>(image_height) / params.vision.patch_size;
            int grid_w                 = static_cast<int>(image_width) / params.vision.patch_size;
            int llm_grid_h             = grid_h / params.vision.spatial_merge_size;
            int llm_grid_w             = grid_w / params.vision.spatial_merge_size;
            int vit_merger_window_size = params.vision.window_size / params.vision.patch_size / params.vision.spatial_merge_size;
            LOG_DEBUG("qwen2.5-vl image patches: pixel_values=[%" PRId64 ", %" PRId64 "] image_grid_thw=[%d,%d,%d] llm_tokens=%d",
                      pixel_values->ne[0],
                      pixel_values->ne[1],
                      grid_t,
                      grid_h,
                      grid_w,
                      llm_grid_h * llm_grid_w);

            // window index
            int inverse_index = 0;
            window_index_vec.resize(llm_grid_h * llm_grid_w);
            window_inverse_index_vec.resize(llm_grid_h * llm_grid_w);
            std::vector<int> seqlens;
            for (int ih = 0; ih < llm_grid_h; ih += vit_merger_window_size) {
                for (int iw = 0; iw < llm_grid_w; iw += vit_merger_window_size) {
                    int win_h = std::min(vit_merger_window_size, llm_grid_h - ih);
                    int win_w = std::min(vit_merger_window_size, llm_grid_w - iw);
                    for (int iy = 0; iy < win_h; iy++) {
                        for (int ix = 0; ix < win_w; ix++) {
                            int index                       = (ih + iy) * llm_grid_w + iw + ix;
                            window_index_vec[inverse_index] = index;
                            window_inverse_index_vec[index] = inverse_index;
                            inverse_index++;
                        }
                    }
                    seqlens.push_back(win_h * win_w * params.vision.spatial_merge_size * params.vision.spatial_merge_size);
                }
            }
            // printf("window_index: ");
            // for (int i : window_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("window_inverse_index: ");
            // for (int i : window_inverse_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
	            // printf("seqlens: ");
	            // for (int i : seqlens) {
	            //     printf("%d ", i);
	            // }
	            // printf("\n");
	            cu_window_seqlens_vec.clear();
	            cu_window_seqlens_vec.reserve(seqlens.size() + 1);
	            cu_window_seqlens_vec.push_back(0);
	            for (int seq_len : seqlens) {
	                int next = cu_window_seqlens_vec.back() + seq_len;
	                if (next != cu_window_seqlens_vec.back()) {
	                    cu_window_seqlens_vec.push_back(next);
	                }
	            }
	            if (qwen_align_debug_enabled()) {
	                qwen_align_log_int_samples("vision.cu_window_seqlens.head", cu_window_seqlens_vec, 0, 32);
	                qwen_align_log_int_samples("vision.cu_window_seqlens.tail",
	                                           cu_window_seqlens_vec,
	                                           static_cast<int64_t>(cu_window_seqlens_vec.size()) - 32,
	                                           32);
	            }
	            auto window_index         = ggml_new_tensor_1d(compute_ctx,
	                                                           GGML_TYPE_I32,
	                                                           llm_grid_h * llm_grid_w);
            auto window_inverse_index = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            set_backend_tensor_data(window_index, window_index_vec.data());
            set_backend_tensor_data(window_inverse_index, window_inverse_index_vec.data());

            // window mask
            int seq_window_size = (vit_merger_window_size * params.vision.spatial_merge_size) * (vit_merger_window_size * params.vision.spatial_merge_size);
            window_mask_vec.resize((grid_h * grid_w) * (grid_h * grid_w));
            int window_start_index = 0;
            for (int seq_index = 0; seq_index < seqlens.size(); seq_index++) {
                int window_end_index = window_start_index + seqlens[seq_index];
                // LOG_DEBUG("%d %d", window_start_index, window_end_index);
                GGML_ASSERT(window_end_index <= grid_h * grid_w);
                for (int i = window_start_index; i < window_end_index; i++) {
                    for (int j = 0; j < grid_h * grid_w; j++) {
                        float mask_value = -INFINITY;
                        if (j >= window_start_index && j < window_end_index) {
                            mask_value = 0;
                        }
                        GGML_ASSERT((i * (grid_h * grid_w) + j) < window_mask_vec.size());
                        window_mask_vec[i * (grid_h * grid_w) + j] = mask_value;
                    }
                }
                window_start_index = window_end_index;
                // printf("\n");
            }
            // printf("window_mask: \n");
            // for (int i = 0; i < grid_h*grid_w; i++) {
            //     for (int j = 0; j < grid_h*grid_w; j++) {
            //         printf("%f ", window_mask_vec[i * (grid_h * grid_w) + j]);
            //     }
            //     printf("\n");
            // }
            auto window_mask = ggml_new_tensor_2d(compute_ctx,
                                                  GGML_TYPE_F32,
                                                  grid_h * grid_w,
                                                  grid_h * grid_w);
            set_backend_tensor_data(window_mask, window_mask_vec.data());

            // pe
            int head_dim = static_cast<int>(params.vision.hidden_size / params.vision.num_heads);
            pe_vec       = Rope::gen_qwen2vl_pe(grid_h,
                                                grid_w,
                                                params.vision.spatial_merge_size,
                                                window_inverse_index_vec,
                                                10000,
                                                {head_dim / 2, head_dim / 2});
            int pos_len  = static_cast<int>(pe_vec.size() / head_dim / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            auto runnter_ctx           = get_context();
            ggml_tensor* hidden_states = vision_forward(&runnter_ctx,
                                                        pixel_values,
                                                        pe,
                                                        window_index,
                                                        window_inverse_index,
                                                        window_mask,
                                                        debug_target);
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        sd::Tensor<float> encode_image(const int n_threads,
                                       const sd::Tensor<float>& image) {
            qwen_align_log_tensor_stats("vision.normalized_image", image);
            sd::Tensor<float> pixel_values = process_image_patches_host(image);
            GGML_ASSERT(!pixel_values.empty());
            qwen_align_log_tensor_stats("vision.pixel_values", pixel_values);
            const int64_t image_width  = image.shape()[0];
            const int64_t image_height = image.shape()[1];
            if (qwen_align_debug_enabled()) {
                const std::vector<std::string> debug_targets = {
                    "patch_embed",
                    "windowed",
                    "block0",
                    "block1",
                    "block2",
                    "block4",
                    "block8",
                    "block14",
                    "block15",
                    "block16",
                    "block23",
                    "block24",
                    "block31",
                    "merged",
                };
                std::vector<std::string> requested_debug_targets = debug_targets;
                const char* dump_targets = std::getenv("ED_QWEN_ALIGN_DUMP_TARGETS");
                std::vector<std::string> optional_debug_targets;
                const std::vector<int> optional_debug_layers = {0, 15, 16, 23, 24, 31};
                const std::vector<std::string> optional_debug_suffixes = {
                    ".input",
                    ".norm1",
                    ".attn.qkv",
                    ".attn.q",
                    ".attn.k",
                    ".attn.v",
                    ".attn.q_rope",
                    ".attn.k_rope",
                    ".attn.preproj",
                    ".attn.out",
                    ".after_attn",
                    ".norm2",
                    ".mlp.gate",
                    ".mlp.act",
                    ".mlp.up",
                    ".mlp.mul",
                    ".mlp.out",
                    ".after_mlp",
                };
                for (int layer : optional_debug_layers) {
                    for (const std::string& suffix : optional_debug_suffixes) {
                        optional_debug_targets.push_back("block" + std::to_string(layer) + suffix);
                    }
                }
                const std::vector<std::string> merger_debug_suffixes = {
                    ".input",
                    ".ln_q",
                    ".reshaped",
                    ".mlp.0",
                    ".gelu",
                    ".out",
                };
                for (const std::string& suffix : merger_debug_suffixes) {
                    optional_debug_targets.push_back("merger" + suffix);
                }
                for (const std::string& target : optional_debug_targets) {
                    const std::string full_name = "vision." + target;
                    if (qwen_align_csv_contains(dump_targets, full_name.c_str()) ||
                        qwen_align_csv_contains(dump_targets, target.c_str())) {
                        requested_debug_targets.push_back(target);
                    }
                }
                for (const std::string& target : requested_debug_targets) {
                    auto get_debug_graph = [&]() -> ggml_cgraph* {
                        return build_encode_image_graph(pixel_values, image_width, image_height, target);
                    };
                    auto debug_tensor = take_or_empty(GGMLRunner::compute<float>(get_debug_graph, n_threads, false));
                    qwen_align_log_tensor_stats(("vision." + target).c_str(), debug_tensor);
                }
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_encode_image_graph(pixel_values, image_width, image_height);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }

        std::vector<sd::Tensor<float>> encode_image_outputs(const int n_threads,
                                                             const sd::Tensor<float>& image) {
            if (params.vision.arch != LLMVisionArch::QWEN3_VL) {
                auto output = encode_image(n_threads, image);
                return output.empty() ? std::vector<sd::Tensor<float>>() : std::vector<sd::Tensor<float>>{std::move(output)};
            }
            const int64_t image_width = image.shape()[0];
            const int64_t image_height = image.shape()[1];
            const auto pixel_values = process_image_patches_host(image);
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* graph = new_graph_custom(LLM_GRAPH_SIZE);
                auto pixels = make_input(pixel_values);
                const int grid_h = static_cast<int>(image_height / params.vision.patch_size);
                const int grid_w = static_cast<int>(image_width / params.vision.patch_size);
                const int head_dim = static_cast<int>(params.vision.hidden_size / params.vision.num_heads);
                auto runner_ctx = get_context();
                auto vision = model.vision_model();
                auto pos_embeds = build_qwen3_vl_patch_pos_embeds(&runner_ctx, vision, grid_h, grid_w);
                window_index_vec.resize(static_cast<size_t>((grid_h / params.vision.spatial_merge_size) * (grid_w / params.vision.spatial_merge_size)));
                for (size_t index = 0; index < window_index_vec.size(); ++index) window_index_vec[index] = static_cast<int>(index);
                pe_vec = Rope::gen_qwen2vl_pe(grid_h, grid_w, params.vision.spatial_merge_size, window_index_vec, 10000, {head_dim / 2, head_dim / 2});
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, static_cast<int>(pe_vec.size() / head_dim / 2));
                set_backend_tensor_data(pe, pe_vec.data());
                auto outputs = model.vision_forward_outputs(&runner_ctx, pixels, pe, nullptr, nullptr, nullptr, pos_embeds);
                auto combined = outputs[0];
                for (size_t index = 1; index < outputs.size(); ++index) combined = ggml_concat(compute_ctx, combined, outputs[index], 0);
                ggml_build_forward_expand(graph, combined);
                return graph;
            };
            auto combined = take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
            if (combined.empty()) return {};
            const size_t count = params.vision.deepstack_visual_indexes.size() + 1;
            std::vector<sd::Tensor<float>> outputs;
            outputs.reserve(count);
            for (size_t index = 0; index < count; ++index) {
                outputs.push_back(sd::ops::slice(combined, 0, static_cast<int64_t>(index) * params.hidden_size, static_cast<int64_t>(index + 1) * params.hidden_size));
            }
            return outputs;
        }

        std::vector<sd::Tensor<float>> encode_video_block_outputs(const int n_threads,
                                                                  const sd::Tensor<float>& frames) {
            if (params.vision.arch != LLMVisionArch::QWEN3_VL) {
                return encode_image_outputs(n_threads, frames);
            }
            const int grid_h = static_cast<int>(frames.shape()[1] / params.vision.patch_size);
            const int grid_w = static_cast<int>(frames.shape()[0] / params.vision.patch_size);
            const auto pixel_values = process_video_block_patches_host(frames);
            if (pixel_values.empty()) {
                return {};
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* graph = new_graph_custom(LLM_GRAPH_SIZE);
                auto pixels = make_input(pixel_values);
                auto runner_ctx = get_context();
                auto vision = model.vision_model();
                const int head_dim = static_cast<int>(params.vision.hidden_size / params.vision.num_heads);
                auto pos_embeds = build_qwen3_vl_patch_pos_embeds(&runner_ctx, vision, grid_h, grid_w);
                window_index_vec.resize(static_cast<size_t>((grid_h / params.vision.spatial_merge_size) *
                                                            (grid_w / params.vision.spatial_merge_size)));
                for (size_t index = 0; index < window_index_vec.size(); ++index) {
                    window_index_vec[index] = static_cast<int>(index);
                }
                pe_vec = Rope::gen_qwen2vl_pe(grid_h,
                                               grid_w,
                                               params.vision.spatial_merge_size,
                                               window_index_vec,
                                               10000,
                                               {head_dim / 2, head_dim / 2});
                const int pos_len = static_cast<int>(pe_vec.size() / head_dim / 2);
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
                set_backend_tensor_data(pe, pe_vec.data());
                auto outputs = model.vision_forward_outputs(&runner_ctx,
                                                            pixels,
                                                            pe,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            pos_embeds);
                auto combined = outputs[0];
                for (size_t index = 1; index < outputs.size(); ++index) {
                    combined = ggml_concat(compute_ctx, combined, outputs[index], 0);
                }
                ggml_build_forward_expand(graph, combined);
                return graph;
            };
            auto combined = take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
            if (combined.empty()) return {};
            const size_t count = params.vision.deepstack_visual_indexes.size() + 1;
            std::vector<sd::Tensor<float>> outputs;
            outputs.reserve(count);
            for (size_t index = 0; index < count; ++index) {
                outputs.push_back(sd::ops::slice(combined, 0, static_cast<int64_t>(index) * params.hidden_size, static_cast<int64_t>(index + 1) * params.hidden_size));
            }
            return outputs;
        }
    };

    struct LLMEmbedder {
        std::shared_ptr<BPETokenizer> tokenizer;
        LLMRunner model;

        LLMEmbedder(LLMArch arch,
                    ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "",
                    bool enable_vision                             = false)
            : model(arch, backend, offload_params_to_cpu, tensor_storage_map, prefix, enable_vision) {
            if (arch == LLMArch::MISTRAL_SMALL_3_2 || arch == LLMArch::MINISTRAL_3_3B) {
                tokenizer = std::make_shared<MistralTokenizer>();
            } else {
                tokenizer = std::make_shared<Qwen2Tokenizer>();
            }
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void alloc_params_buffer() {
            model.alloc_params_buffer();
        }

        std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                                  std::pair<int, int> attn_range,
                                                                  size_t max_length = 0,
                                                                  bool padding      = false) {
            std::vector<std::pair<std::string, float>> parsed_attention;
            parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            {
                std::stringstream ss;
                ss << "[";
                for (const auto& item : parsed_attention) {
                    ss << "['" << item.first << "', " << item.second << "], ";
                }
                ss << "]";
                LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
            }

            std::vector<int> tokens;
            std::vector<float> weights;
            for (const auto& item : parsed_attention) {
                const std::string& curr_text = item.first;
                float curr_weight            = item.second;
                std::vector<int> curr_tokens = tokenizer->tokenize(curr_text, nullptr);
                tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
                weights.insert(weights.end(), curr_tokens.size(), curr_weight);
            }

            tokenizer->pad_tokens(tokens, &weights, nullptr, padding ? max_length : 0, padding ? max_length : 100000000, padding);

            // for (int i = 0; i < tokens.size(); i++) {
            //     std::cout << tokens[i] << ":" << weights[i] << ", ";
            // }
            // std::cout << std::endl;

            return {tokens, weights};
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);
            bool test_mistral          = false;
            bool test_qwen3            = true;
            bool test_vit              = false;
            bool test_decoder_with_vit = false;

            if (test_decoder_with_vit) {
                sd::Tensor<float> image_embed;
                {
                    auto image = sd::load_tensor_from_file_as_tensor<float>("qwen2vl_normalized.bin");
                    print_sd_tensor(image, false, "image");
                    sd::Tensor<float> out;

                    int64_t t0   = ggml_time_ms();
                    auto out_opt = model.encode_image(8, image);
                    int64_t t1   = ggml_time_ms();

                    GGML_ASSERT(!out_opt.empty());
                    out = std::move(out_opt);
                    print_sd_tensor(out, false, "image_embed");
                    image_embed = out;
                    LOG_DEBUG("llm encode_image test done in %lldms", t1 - t0);
                }

                std::string placeholder  = "<|image_pad|>";
                std::string img_prompt   = "Picture 1: <|vision_start|>";  // [24669, 220, 16, 25, 220, 151652]
                int64_t num_image_tokens = image_embed.shape()[1];
                img_prompt.reserve(num_image_tokens * placeholder.size());
                for (int i = 0; i < num_image_tokens; i++) {
                    img_prompt += placeholder;
                }
                img_prompt += "<|vision_end|>";

                std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
                image_embeds.emplace_back(64, image_embed);

                std::pair<int, int> prompt_attn_range;
                std::string text = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                text += img_prompt;
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "change 'flux.cpp' to 'edit.cpp'";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";

                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), image_embeds, {});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_vit) {
                // auto image = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 280, 280, 3);
                // ggml_set_f32(image, 0.f);
                auto image = sd::load_tensor_from_file_as_tensor<float>("qwen2vl_normalized.bin");
                print_sd_tensor(image, false, "image");
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.encode_image(8, image);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out, false, "out");

                // auto ref_out = load_tensor_from_file(ctx, "qwen2vl.bin");
                // ggml_ext_tensor_diff(ref_out, out, 0.01f);

                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_mistral) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "[/INST]";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {10, 20, 30});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_qwen3) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {35});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cpu f16: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_COUNT;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "text_encoders.llm.")) {
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

            LLMArch arch = LLMArch::QWEN3;

            std::shared_ptr<LLMEmbedder> llm = std::make_shared<LLMEmbedder>(arch,
                                                                             backend,
                                                                             true,
                                                                             tensor_storage_map,
                                                                             "text_encoders.llm",
                                                                             true);

            llm->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            llm->get_param_tensors(tensors, "text_encoders.llm");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("llm model loaded");
            llm->test();
        }
    };
};  // LLM

#endif  // __LLM_HPP__
