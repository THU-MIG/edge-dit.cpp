#ifndef __ED_MINIMAX_H3_AUDIO_VAE_HPP__
#define __ED_MINIMAX_H3_AUDIO_VAE_HPP__
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "backend/ggml/ggml_extend.hpp"
namespace MiniMaxH3Audio {
namespace Ops {
    static bool h3_audio_env_flag_enabled_or_default(const char* name, bool default_enabled) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_enabled;
        }
        return value[0] != '0';
    }

    static ggml_type audio_conv_weight_type(ggml_type type) {
        return type == GGML_TYPE_BF16 ? GGML_TYPE_F16 : type;
    }

    static ggml_tensor* repeat_with_vulkan_f32_workaround(ggml_backend_t backend,
                                                          ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          int64_t ne0,
                                                          int64_t ne1,
                                                          int64_t ne2,
                                                          int64_t ne3) {
        if (x->type != GGML_TYPE_F32 &&
            (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
            sd_backend_is(backend, "vulkan")) {
            auto x_f32    = ggml_cast(ctx, x, GGML_TYPE_F32);
            auto repeated = ggml_repeat_4d(ctx,
                                           x_f32,
                                           ne0,
                                           ne1,
                                           ne2,
                                           ne3);
            return ggml_cast(ctx, repeated, x->type);
        }
        return ggml_repeat_4d(ctx, x, ne0, ne1, ne2, ne3);
    }

    static ggml_tensor* repeat_1d_value(GGMLRunnerContext* runner_ctx, ggml_tensor* x, int64_t count) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(x->ne[0] == 1);
        return repeat_with_vulkan_f32_workaround(runner_ctx->backend, ctx, x, count, x->ne[1], x->ne[2], x->ne[3]);
    }

    static ggml_tensor* replicate_pad_1d(GGMLRunnerContext* runner_ctx, ggml_tensor* x, int64_t left, int64_t right) {
        auto ctx = runner_ctx->ggml_ctx;
        if (left > 0) {
            auto first = ggml_ext_slice(ctx, x, 0, 0, 1);
            x          = ggml_concat(ctx, repeat_1d_value(runner_ctx, first, left), x, 0);
        }
        if (right > 0) {
            auto last = ggml_ext_slice(ctx, x, 0, x->ne[0] - 1, x->ne[0]);
            x         = ggml_concat(ctx, x, repeat_1d_value(runner_ctx, last, right), 0);
        }
        return x;
    }

    static ggml_tensor* tile_depthwise_filter_1d(GGMLRunnerContext* runner_ctx, ggml_tensor* filter, int64_t channels) {
        auto ctx          = runner_ctx->ggml_ctx;
        ggml_tensor* base = filter;
        if (ggml_n_dims(base) == 3) {
            base = ggml_reshape_4d(ctx, base, base->ne[0], 1, 1, 1);
        } else if (ggml_n_dims(base) == 1) {
            base = ggml_reshape_4d(ctx, base, base->ne[0], 1, 1, 1);
        }
        return repeat_with_vulkan_f32_workaround(runner_ctx->backend, ctx, base, base->ne[0], 1, channels, 1);
    }

    static ggml_tensor* depthwise_conv1d(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* filter,
                                         int stride,
                                         int padding) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1);
        auto tiled = tile_depthwise_filter_1d(runner_ctx, filter, x->ne[1]);
        if (h3_audio_env_flag_enabled_or_default("ED_MINIMAX_H3_AUDIO_DIRECT_DEPTHWISE", true) &&
            sd_backend_is(runner_ctx->backend, "CUDA") && tiled->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32) {
            auto kernel = ggml_reshape_4d(ctx, tiled, tiled->ne[0], 1, 1, x->ne[1]);
            auto input = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], 1);
            auto out = ggml_conv_2d_dw_direct(ctx, kernel, input, stride, 1, padding, 0, 1, 1);
            return ggml_reshape_4d(ctx, out, out->ne[0], out->ne[2], 1, 1);
        }
        auto out   = ggml_conv_1d_dw(ctx, tiled, x, stride, padding, 1);
        return ggml_reshape_4d(ctx, out, out->ne[0], out->ne[1], 1, 1);
    }

    static ggml_tensor* reverse_1d_filter(ggml_context* ctx, ggml_tensor* filter) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(filter != nullptr);
        GGML_ASSERT(filter->ne[1] == 1);
        GGML_ASSERT(filter->ne[2] == 1);
        GGML_ASSERT(filter->ne[3] == 1);

        ggml_tensor* reversed = nullptr;
        for (int64_t k = filter->ne[0] - 1; k >= 0; --k) {
            auto slice = ggml_ext_slice(ctx, filter, 0, k, k + 1);
            reversed   = reversed == nullptr ? slice : ggml_concat(ctx, reversed, slice, 0);
        }
        return reversed;
    }

    static ggml_tensor* depthwise_conv_transpose1d(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* filter,
                                                   int stride) {
        GGML_ASSERT(x->ne[3] == 1);
        GGML_ASSERT(filter->ne[1] == 1);
        GGML_ASSERT(filter->ne[2] == 1 && filter->ne[3] == 1);

        const int64_t time        = x->ne[0];
        const int64_t channels    = x->ne[1];
        const int64_t kernel_size = filter->ne[0];
        const int64_t out_time    = (time - 1) * stride + kernel_size;

        auto x_flat = ggml_reshape_3d(ctx, x, 1, time, channels);
        if (stride > 1) {
            auto zero_unit = ggml_ext_scale(ctx, x_flat, 0.0f);
            auto zero_tail = zero_unit;
            for (int i = 1; i < stride - 1; ++i) {
                zero_tail = ggml_concat(ctx, zero_tail, zero_unit, 0);
            }
            x_flat = ggml_concat(ctx, x_flat, zero_tail, 0);
        }
        x_flat = ggml_reshape_3d(ctx, x_flat, time * stride, 1, channels);

        auto reversed_filter = reverse_1d_filter(ctx, filter);
        ggml_tensor* out = nullptr;
        if (h3_audio_env_flag_enabled_or_default("ED_MINIMAX_H3_AUDIO_DIRECT_DEPTHWISE", true) &&
            reversed_filter->type == GGML_TYPE_F32 && x_flat->type == GGML_TYPE_F32) {
            auto kernel = ggml_reshape_4d(ctx, reversed_filter, kernel_size, 1, 1, 1);
            auto input = ggml_reshape_4d(ctx, x_flat, time * stride, 1, 1, channels);
            out = ggml_conv_2d_dw_direct(ctx,
                                         kernel,
                                         input,
                                         1,
                                         1,
                                         static_cast<int>(kernel_size - 1),
                                         0,
                                         1,
                                         1);
            out = ggml_reshape_3d(ctx, out, out->ne[0], 1, channels);
        } else {
            out = ggml_conv_1d(ctx, reversed_filter, x_flat, 1, static_cast<int>(kernel_size - 1), 1);
        }
        if (out->ne[0] > out_time) {
            out = ggml_ext_slice(ctx, out, 0, 0, out_time);
        }
        GGML_ASSERT(out->ne[0] == out_time);
        GGML_ASSERT(out->ne[1] == 1);
        GGML_ASSERT(out->ne[2] == channels);

        out = ggml_ext_scale(ctx, out, static_cast<float>(stride));
        return ggml_reshape_4d(ctx, out, out_time, channels, 1, 1);
    }
    struct Conv1D : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int stride;
        int padding;
        int dilation;
        bool bias;
        std::string prefix;

        Conv1D(int64_t in_channels,
               int64_t out_channels,
               int kernel_size,
               int stride   = 1,
               int padding  = 0,
               int dilation = 1,
               bool bias    = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding),
              dilation(dilation),
              bias(bias) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            this->prefix     = prefix;
            ggml_type wtype  = audio_conv_weight_type(get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16));
            params["weight"] = ggml_new_tensor_4d(ctx, wtype, kernel_size, in_channels, out_channels, 1);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            x = ggml_conv_1d(ctx->ggml_ctx, params["weight"], x, stride, padding, dilation);
            if (bias) {
                auto b = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, params["bias"]->ne[0], 1, 1);
                x      = ggml_add_inplace(ctx->ggml_ctx, x, b);
            }
            return x;
        }
    };

    struct ConvTranspose1D : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int stride;
        int padding;
        int dilation;
        bool bias;

        ConvTranspose1D(int64_t in_channels,
                        int64_t out_channels,
                        int kernel_size,
                        int stride,
                        int padding,
                        int dilation = 1,
                        bool bias    = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding),
              dilation(dilation),
              bias(bias) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ED_UNUSED(tensor_storage_map);
            ED_UNUSED(prefix);
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kernel_size, out_channels, in_channels, 1);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(dilation == 1);
            x = ggml_conv_transpose_1d(ctx->ggml_ctx, params["weight"], x, stride, 0, dilation);
            ggml_set_name(x, "minimax_h3.audio_vae.conv_transpose_1d");
            if (padding > 0) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 0, padding, x->ne[0] - padding);
            }
            if (bias) {
                auto b = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, params["bias"]->ne[0], 1, 1);
                x      = ggml_add_inplace(ctx->ggml_ctx, x, b);
            }
            return x;
        }
    };

    struct SnakeBeta1D : public UnaryBlock {
        int64_t channels;
        float eps = 1e-9f;

        explicit SnakeBeta1D(int64_t channels)
            : channels(channels) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ED_UNUSED(tensor_storage_map);
            ED_UNUSED(prefix);
            params["alpha"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            params["beta"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto alpha       = ggml_exp(ctx->ggml_ctx, params["alpha"]);
            auto beta        = ggml_exp(ctx->ggml_ctx, params["beta"]);
            alpha            = ggml_reshape_4d(ctx->ggml_ctx, alpha, 1, alpha->ne[0], 1, 1);
            beta             = ggml_reshape_4d(ctx->ggml_ctx, beta, 1, beta->ne[0], 1, 1);
            auto oscillation = ggml_sin(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, alpha));
            oscillation      = ggml_mul(ctx->ggml_ctx, oscillation, oscillation);
            auto eps_tensor  = ggml_ext_scale(ctx->ggml_ctx, ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1), eps);
            oscillation      = ggml_div(ctx->ggml_ctx, oscillation, ggml_add(ctx->ggml_ctx, beta, eps_tensor));
            return ggml_add(ctx->ggml_ctx, x, oscillation);
        }
    };

    struct Activation1D : public GGMLBlock {
        int64_t channels;
        int up_ratio         = 2;
        int down_ratio       = 2;
        int up_kernel_size   = 12;
        int down_kernel_size = 12;

        explicit Activation1D(int64_t channels)
            : channels(channels) {
            blocks["act"] = std::make_shared<SnakeBeta1D>(channels);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ggml_type down_type                 = audio_conv_weight_type(get_type(prefix + "downsample.lowpass.filter", tensor_storage_map, GGML_TYPE_F16));
            if (h3_audio_env_flag_enabled_or_default("ED_MINIMAX_H3_AUDIO_DIRECT_DEPTHWISE", true)) {
                down_type = GGML_TYPE_F32;
            }
            params["downsample.lowpass.filter"] = ggml_new_tensor_3d(ctx, down_type, down_kernel_size, 1, 1);
            params["upsample.filter"]           = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, up_kernel_size, 1, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto act         = std::dynamic_pointer_cast<SnakeBeta1D>(blocks["act"]);
            auto up_filter   = params["upsample.filter"];
            auto down_filter = params["downsample.lowpass.filter"];

            int up_pad       = up_kernel_size / up_ratio - 1;
            int up_pad_left  = up_pad * up_ratio + (up_kernel_size - up_ratio) / 2;
            int up_pad_right = up_pad * up_ratio + (up_kernel_size - up_ratio + 1) / 2;

            x = replicate_pad_1d(ctx, x, up_pad, up_pad);
            x = depthwise_conv_transpose1d(ctx->ggml_ctx, x, up_filter, up_ratio);
            x = ggml_ext_slice(ctx->ggml_ctx, x, 0, up_pad_left, x->ne[0] - up_pad_right);

            x = act->forward(ctx, x);

            int down_pad_left  = down_kernel_size / 2 - (down_kernel_size % 2 == 0 ? 1 : 0);
            int down_pad_right = down_kernel_size / 2;
            x                  = replicate_pad_1d(ctx, x, down_pad_left, down_pad_right);
            x                  = depthwise_conv1d(ctx, x, down_filter, down_ratio, 0);
            return x;
        }
    };
}  // namespace Ops

    struct AudioAMPBlock : public GGMLBlock {
        int channels;

        AudioAMPBlock(int channels,
                      int kernel_size,
                      const std::array<int, 3>& dilations)
            : channels(channels) {
            for (int i = 0; i < 3; ++i) {
                blocks["activations." + std::to_string(i * 2)] =
                    std::make_shared<Ops::Activation1D>(channels);
                blocks["activations." + std::to_string(i * 2 + 1)] =
                    std::make_shared<Ops::Activation1D>(channels);
                blocks["convs1." + std::to_string(i)] =
                    std::make_shared<Ops::Conv1D>(channels,
                                                   channels,
                                                   kernel_size,
                                                   1,
                                                   (kernel_size * dilations[i] - dilations[i]) / 2,
                                                   dilations[i]);
                blocks["convs2." + std::to_string(i)] =
                    std::make_shared<Ops::Conv1D>(channels,
                                                   channels,
                                                   kernel_size,
                                                   1,
                                                   kernel_size / 2);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            for (int i = 0; i < 3; ++i) {
                auto act1 = std::dynamic_pointer_cast<Ops::Activation1D>(
                    blocks["activations." + std::to_string(i * 2)]);
                auto act2 = std::dynamic_pointer_cast<Ops::Activation1D>(
                    blocks["activations." + std::to_string(i * 2 + 1)]);
                auto conv1 = std::dynamic_pointer_cast<Ops::Conv1D>(
                    blocks["convs1." + std::to_string(i)]);
                auto conv2 = std::dynamic_pointer_cast<Ops::Conv1D>(
                    blocks["convs2." + std::to_string(i)]);

                auto h = conv1->forward(ctx, act1->forward(ctx, x));
                h      = conv2->forward(ctx, act2->forward(ctx, h));
                x      = ggml_add(ctx->ggml_ctx, x, h);
            }
            return x;
        }
    };

    struct BigVGAN : public GGMLBlock {
        static constexpr int initial_channels                     = 1024;
        static constexpr int num_kernels                          = 3;
        static constexpr int num_upsamples                        = 7;
        static constexpr std::array<int, num_upsamples> rates     = {5, 5, 2, 2, 2, 2, 2};
        static constexpr std::array<int, num_upsamples> kernels   = {9, 9, 4, 4, 4, 4, 4};
        static constexpr std::array<int, num_kernels> res_kernels = {3, 7, 11};

        BigVGAN() {
            blocks["conv_pre"] = std::make_shared<Ops::Conv1D>(2048,
                                                                initial_channels,
                                                                7,
                                                                1,
                                                                3);
            int channels       = initial_channels;
            for (int i = 0; i < num_upsamples; ++i) {
                int next_channels = initial_channels / (1 << (i + 1));
                blocks["ups." + std::to_string(i) + ".0"] =
                    std::make_shared<Ops::ConvTranspose1D>(channels,
                                                            next_channels,
                                                            kernels[i],
                                                            rates[i],
                                                            (kernels[i] - rates[i]) / 2);
                for (int j = 0; j < num_kernels; ++j) {
                    blocks["resblocks." + std::to_string(i * num_kernels + j)] =
                        std::make_shared<AudioAMPBlock>(next_channels,
                                                        res_kernels[j],
                                                        std::array<int, 3>{1, 3, 5});
                }
                channels = next_channels;
            }
            blocks["activation_post"] = std::make_shared<Ops::Activation1D>(channels);
            blocks["conv_post"]       = std::make_shared<Ops::Conv1D>(channels,
                                                                 1,
                                                                 7,
                                                                 1,
                                                                 3,
                                                                 1,
                                                                 false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv_pre = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["conv_pre"]);
            x             = conv_pre->forward(ctx, x);
            for (int i = 0; i < num_upsamples; ++i) {
                auto up = std::dynamic_pointer_cast<Ops::ConvTranspose1D>(
                    blocks["ups." + std::to_string(i) + ".0"]);
                x = up->forward(ctx, x);

                ggml_tensor* sum = nullptr;
                for (int j = 0; j < num_kernels; ++j) {
                    auto block = std::dynamic_pointer_cast<AudioAMPBlock>(
                        blocks["resblocks." + std::to_string(i * num_kernels + j)]);
                    auto value = block->forward(ctx, x);
                    sum        = sum == nullptr ? value : ggml_add(ctx->ggml_ctx, sum, value);
                }
                x = ggml_ext_scale(ctx->ggml_ctx, sum, 1.f / num_kernels);
            }
            auto activation = std::dynamic_pointer_cast<Ops::Activation1D>(blocks["activation_post"]);
            auto conv_post  = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["conv_post"]);
            return ggml_clamp(ctx->ggml_ctx,
                              conv_post->forward(ctx, activation->forward(ctx, x)),
                              -1.f,
                              1.f);
        }
    };
struct AudioSnake1D : public UnaryBlock {
    int64_t channels;
    explicit AudioSnake1D(int64_t value) : channels(value) {}
    void init_params(ggml_context* ctx, const String2TensorStorage& storage = {}, const std::string prefix = "") override {
        ED_UNUSED(storage); ED_UNUSED(prefix);
        params["alpha"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto alpha = params["alpha"];
        auto oscillation = ggml_sin(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, alpha));
        oscillation = ggml_mul(ctx->ggml_ctx, oscillation, oscillation);
        auto eps = ggml_ext_scale(ctx->ggml_ctx, ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1), 1e-9f);
        return ggml_add(ctx->ggml_ctx, x, ggml_div(ctx->ggml_ctx, oscillation, ggml_add(ctx->ggml_ctx, alpha, eps)));
    }
};

struct AudioEncoderResidualUnit : public GGMLBlock {
    AudioEncoderResidualUnit(int64_t channels, int dilation) {
        blocks["block.0"] = std::make_shared<AudioSnake1D>(channels);
        blocks["block.1"] = std::make_shared<Ops::Conv1D>(channels, channels, 7, 1, 3 * dilation, dilation);
        blocks["block.2"] = std::make_shared<AudioSnake1D>(channels);
        blocks["block.3"] = std::make_shared<Ops::Conv1D>(channels, channels, 1);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto act1 = std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.0"]);
        auto conv1 = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["block.1"]);
        auto act2 = std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.2"]);
        auto conv2 = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["block.3"]);
        auto h = conv2->forward(ctx, act2->forward(ctx, conv1->forward(ctx, act1->forward(ctx, x))));
        if (x->ne[0] != h->ne[0]) {
            const int64_t pad = (x->ne[0] - h->ne[0]) / 2;
            x = ggml_ext_slice(ctx->ggml_ctx, x, 0, pad, x->ne[0] - pad);
        }
        return ggml_add(ctx->ggml_ctx, x, h);
    }
};

struct AudioEncoderBlock : public GGMLBlock {
    AudioEncoderBlock(int64_t out_channels, int stride) {
        const int64_t in_channels = out_channels / 2;
        blocks["block.0"] = std::make_shared<AudioEncoderResidualUnit>(in_channels, 1);
        blocks["block.1"] = std::make_shared<AudioEncoderResidualUnit>(in_channels, 3);
        blocks["block.2"] = std::make_shared<AudioEncoderResidualUnit>(in_channels, 9);
        blocks["block.3"] = std::make_shared<AudioSnake1D>(in_channels);
        blocks["block.4"] = std::make_shared<Ops::Conv1D>(in_channels, out_channels, 2 * stride, stride, static_cast<int>(std::ceil(stride / 2.f)));
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        for (int index = 0; index < 3; ++index) x = std::dynamic_pointer_cast<AudioEncoderResidualUnit>(blocks["block." + std::to_string(index)])->forward(ctx, x);
        return std::dynamic_pointer_cast<Ops::Conv1D>(blocks["block.4"])->forward(ctx, std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.3"])->forward(ctx, x));
    }
};

struct AudioEncoder : public GGMLBlock {
    static constexpr std::array<int, 5> strides = {2, 4, 4, 5, 5};
    AudioEncoder() {
        int64_t channels = 64;
        blocks["block.0"] = std::make_shared<Ops::Conv1D>(1, channels, 7, 1, 3);
        for (size_t index = 0; index < strides.size(); ++index) { channels *= 2; blocks["block." + std::to_string(index + 1)] = std::make_shared<AudioEncoderBlock>(channels, strides[index]); }
        blocks["block.6"] = std::make_shared<AudioSnake1D>(channels);
        blocks["block.7"] = std::make_shared<Ops::Conv1D>(channels, 2048, 3, 1, 1);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        x = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["block.0"])->forward(ctx, x);
        for (size_t index = 0; index < strides.size(); ++index) x = std::dynamic_pointer_cast<AudioEncoderBlock>(blocks["block." + std::to_string(index + 1)])->forward(ctx, x);
        return std::dynamic_pointer_cast<Ops::Conv1D>(blocks["block.7"])->forward(ctx, std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.6"])->forward(ctx, x));
    }
};

struct AudioGeGLUMLP : public GGMLBlock {
    AudioGeGLUMLP(int64_t hidden_size, int64_t intermediate_size) {
        blocks["norm"] = std::make_shared<LayerNorm>(hidden_size);
        blocks["w0"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
        blocks["w1"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
        blocks["w2"] = std::make_shared<Linear>(intermediate_size, hidden_size, true);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        x = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"])->forward(ctx, x);
        auto gate = ggml_ext_gelu(ctx->ggml_ctx, std::dynamic_pointer_cast<Linear>(blocks["w0"])->forward(ctx, x), true);
        return std::dynamic_pointer_cast<Linear>(blocks["w2"])->forward(ctx, ggml_mul(ctx->ggml_ctx, gate, std::dynamic_pointer_cast<Linear>(blocks["w1"])->forward(ctx, x)));
    }
};

struct AudioCausalAttention : public GGMLBlock {
    static constexpr int64_t in_channels = 2048, out_channels = 32, num_head = 8, head_dim = in_channels / num_head;
    AudioCausalAttention() { blocks["qkv"] = std::make_shared<Linear>(in_channels, in_channels * 3, false); blocks["proj"] = std::make_shared<Linear>(out_channels, out_channels, true); }
    void init_params(ggml_context* ctx, const String2TensorStorage& storage = {}, const std::string prefix = "") override {
        GGMLBlock::init_params(ctx, storage, prefix); params["q_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels); params["zero_k_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels); params["v_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto qkv = ggml_ext_chunk(ctx->ggml_ctx, std::dynamic_pointer_cast<Linear>(blocks["qkv"])->forward(ctx, x), 3, 0);
        auto shape_bias = [&](ggml_tensor* value) { return ggml_reshape_4d(ctx->ggml_ctx, value, value->ne[0], 1, 1, 1); };
        auto q = ggml_add(ctx->ggml_ctx, qkv[0], shape_bias(params["q_bias"]));
        auto k = ggml_add(ctx->ggml_ctx, qkv[1], shape_bias(params["zero_k_bias"]));
        auto v = ggml_add(ctx->ggml_ctx, qkv[2], shape_bias(params["v_bias"]));
        const int64_t sequence = x->ne[1];
        auto mask = ggml_diag_mask_inf(ctx->ggml_ctx, ggml_ext_zeros(ctx->ggml_ctx, sequence, sequence, 1, 1), 0);
        auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_head, mask, false, ctx->flash_attn_enabled);
        const int64_t batch = out->ne[2] * out->ne[3];
        out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, ggml_reshape_4d(ctx->ggml_ctx, out, head_dim, num_head, sequence, batch), 1, 0, 2, 3));
        out = ggml_mean(ctx->ggml_ctx, out);
        out = ggml_reshape_3d(ctx->ggml_ctx, out, head_dim, sequence, batch);
        out = ggml_mean(ctx->ggml_ctx, ggml_reshape_4d(ctx->ggml_ctx, out, head_dim / out_channels, out_channels, sequence, batch));
        out = ggml_reshape_3d(ctx->ggml_ctx, out, out_channels, sequence, batch);
        return std::dynamic_pointer_cast<Linear>(blocks["proj"])->forward(ctx, out);
    }
};

struct AudioAttentionProjection : public GGMLBlock {
    AudioAttentionProjection() {
        blocks["norm1"] = std::make_shared<LayerNorm>(2048); blocks["attn"] = std::make_shared<AudioCausalAttention>(); blocks["proj"] = std::make_shared<Linear>(2048, 32, true);
        blocks["norm3"] = std::make_shared<LayerNorm>(2048); blocks["norm2"] = std::make_shared<LayerNorm>(32); blocks["mlp"] = std::make_shared<AudioGeGLUMLP>(32, 64);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto h = ggml_add(ctx->ggml_ctx,
                          std::dynamic_pointer_cast<Linear>(blocks["proj"])->forward(ctx, std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"])->forward(ctx, x)),
                          std::dynamic_pointer_cast<AudioCausalAttention>(blocks["attn"])->forward(ctx, std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"])->forward(ctx, x)));
        return ggml_add(ctx->ggml_ctx, h, std::dynamic_pointer_cast<AudioGeGLUMLP>(blocks["mlp"])->forward(ctx, std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"])->forward(ctx, h)));
    }
};

struct AudioDecoder : public GGMLBlock {
    static constexpr int kLatentChannels = 32;
    AudioDecoder() {
        blocks["dec_in_proj"] = std::make_shared<Ops::Conv1D>(kLatentChannels, 2048, 1);
        blocks["decoder"] = std::make_shared<BigVGAN>();
    }
    void init_params(ggml_context* ctx, const String2TensorStorage& storage = {}, const std::string prefix = "") override {
        ED_UNUSED(storage); ED_UNUSED(prefix);
        params["latents_mean"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatentChannels);
        params["latents_std"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatentChannels);
    }
    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* latent) {
        GGML_ASSERT(latent->ne[1] == 2 && latent->ne[2] == kLatentChannels);
        latent = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, latent, 0, 2, 1, 3));
        auto mean = ggml_reshape_4d(ctx->ggml_ctx, params["latents_mean"], 1, kLatentChannels, 1, 1);
        auto std = ggml_reshape_4d(ctx->ggml_ctx, params["latents_std"], 1, kLatentChannels, 1, 1);
        latent = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, latent, std), mean);
        auto dec_in = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["dec_in_proj"]);
        auto decoder = std::dynamic_pointer_cast<BigVGAN>(blocks["decoder"]);
        const int64_t streams = latent->ne[2] * latent->ne[3];
        latent = ggml_reshape_3d(ctx->ggml_ctx, latent, latent->ne[0], latent->ne[1], streams);
        ggml_tensor* waveform = nullptr;
        for (int64_t stream = 0; stream < streams; ++stream) {
            auto value = decoder->forward(ctx, dec_in->forward(ctx, ggml_ext_slice(ctx->ggml_ctx, latent, 2, stream, stream + 1)));
            waveform = waveform == nullptr ? value : ggml_concat(ctx->ggml_ctx, waveform, value, 2);
        }
        return ggml_reshape_4d(ctx->ggml_ctx, waveform, waveform->ne[0], streams, 1, 1);
    }
};
struct AudioVAE : public GGMLBlock {
    static constexpr int kLatentChannels = 32;
    AudioVAE() {
        blocks["encoder"] = std::make_shared<AudioEncoder>();
        blocks["pre_block"] = std::make_shared<AudioAttentionProjection>();
        blocks["mean_proj"] = std::make_shared<Ops::Conv1D>(kLatentChannels, kLatentChannels, 1);
        blocks["logs_proj"] = std::make_shared<Ops::Conv1D>(kLatentChannels, kLatentChannels, 1);
        blocks["dec_in_proj"] = std::make_shared<Ops::Conv1D>(kLatentChannels, 2048, 1);
        blocks["decoder"] = std::make_shared<BigVGAN>();
    }
    void init_params(ggml_context* ctx, const String2TensorStorage& storage = {}, const std::string prefix = "") override {
        GGMLBlock::init_params(ctx, storage, prefix);
        params["latents_mean"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatentChannels);
        params["latents_std"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatentChannels);
    }
    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* waveform) {
        GGML_ASSERT(waveform->ne[1] == 2);
        auto mean = ggml_reshape_4d(ctx->ggml_ctx, params["latents_mean"], 1, kLatentChannels, 1, 1);
        auto std = ggml_reshape_4d(ctx->ggml_ctx, params["latents_std"], 1, kLatentChannels, 1, 1);
        ggml_tensor* encoded = nullptr;
        for (int64_t channel = 0; channel < waveform->ne[1]; ++channel) {
            auto x = ggml_ext_slice(ctx->ggml_ctx, waveform, 1, channel, channel + 1);
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], 1, 1);
            x = std::dynamic_pointer_cast<AudioEncoder>(blocks["encoder"])->forward(ctx, x);
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x = std::dynamic_pointer_cast<AudioAttentionProjection>(blocks["pre_block"])->forward(ctx, x);
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["mean_proj"])->forward(ctx, x);
            x = ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, x, mean), std);
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));
            encoded = encoded == nullptr ? x : ggml_concat(ctx->ggml_ctx, encoded, x, 1);
        }
        return encoded;
    }
    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* latent) {
        GGML_ASSERT(latent->ne[1] == 2 && latent->ne[2] == kLatentChannels);
        latent = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, latent, 0, 2, 1, 3));
        auto mean = ggml_reshape_4d(ctx->ggml_ctx, params["latents_mean"], 1, kLatentChannels, 1, 1);
        auto std = ggml_reshape_4d(ctx->ggml_ctx, params["latents_std"], 1, kLatentChannels, 1, 1);
        latent = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, latent, std), mean);
        auto decoder_input = std::dynamic_pointer_cast<Ops::Conv1D>(blocks["dec_in_proj"]);
        auto decoder = std::dynamic_pointer_cast<BigVGAN>(blocks["decoder"]);
        const int64_t streams = latent->ne[2] * latent->ne[3];
        latent = ggml_reshape_3d(ctx->ggml_ctx, latent, latent->ne[0], latent->ne[1], streams);
        ggml_tensor* waveform = nullptr;
        for (int64_t stream = 0; stream < streams; ++stream) {
            auto value = decoder->forward(ctx, decoder_input->forward(ctx, ggml_ext_slice(ctx->ggml_ctx, latent, 2, stream, stream + 1)));
            waveform = waveform == nullptr ? value : ggml_concat(ctx->ggml_ctx, waveform, value, 2);
        }
        return ggml_reshape_4d(ctx->ggml_ctx, waveform, waveform->ne[0], streams, 1, 1);
    }
};

struct AudioVAERunner : public GGMLRunner {
    AudioVAE model;
    AudioVAERunner(ggml_backend_t backend, bool offload, const String2TensorStorage& storage, const std::string& prefix = "audio_vae")
        : GGMLRunner(backend, offload) { model.init(params_ctx, storage, prefix); }
    std::string get_desc() override { return "minimax_h3_audio_vae"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) { model.get_param_tensors(tensors, prefix); }
    sd::Tensor<float> decode(int n_threads, const sd::Tensor<float>& latent) {
        auto get_graph = [&]() { auto input = make_input(latent); auto runner_ctx = get_context(); auto graph = new_graph_custom(655360); ggml_build_forward_expand(graph, model.decode(&runner_ctx, input)); return graph; };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), 4);
    }
    sd::Tensor<float> encode(int n_threads, const sd::Tensor<float>& waveform) {
        auto get_graph = [&]() { auto input = make_input(waveform); auto runner_ctx = get_context(); auto graph = new_graph_custom(655360); ggml_build_forward_expand(graph, model.encode(&runner_ctx, input)); return graph; };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), 4);
    }
    int input_sample_rate() const { return 32000; }
};
}  // namespace MiniMaxH3Audio
#endif
