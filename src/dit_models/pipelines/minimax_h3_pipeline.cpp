#include "dit_models/pipelines/minimax_h3_pipeline.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <iomanip>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/components/autoencoders/minimax_h3_vae.hpp"
#include "dit_models/components/autoencoders/minimax_h3_audio_vae.hpp"
#include "dit_models/components/text_encoders/llm.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include "utils/rng_philox.hpp"
#include "utils/util.h"

namespace edgedit {
namespace {

constexpr int H3_REF_IMAGE_SHORT_EDGE = 2048;
constexpr int64_t H3_MIN_GENERATION_PIXELS = 256 * 256;

std::vector<float> h3_resample_audio_sinc(const ed_audio_t& source,
                                          uint64_t source_samples,
                                          uint32_t source_channel,
                                          uint32_t target_sample_rate) {
    if (source.sample_rate == target_sample_rate) {
        std::vector<float> output(source_samples);
        for (uint64_t sample = 0; sample < source_samples; ++sample) {
            output[sample] = std::clamp(source.data[sample * source.channels + source_channel], -1.f, 1.f);
        }
        return output;
    }

    const uint32_t divisor = std::gcd(source.sample_rate, target_sample_rate);
    const int64_t original_rate = source.sample_rate / divisor;
    const int64_t target_rate = target_sample_rate / divisor;
    constexpr double filter_width = 6.0;
    constexpr double rolloff = 0.99;
    const double base_rate = std::min(original_rate, target_rate) * rolloff;
    const int64_t width = static_cast<int64_t>(std::ceil(filter_width * original_rate / base_rate));
    const int64_t kernel_size = width * 2 + original_rate;
    const uint64_t target_samples = (source_samples * static_cast<uint64_t>(target_rate) + original_rate - 1) /
                                    static_cast<uint64_t>(original_rate);
    std::vector<float> kernels(static_cast<size_t>(target_rate * kernel_size));
    for (int64_t phase = 0; phase < target_rate; ++phase) {
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            double time = (static_cast<double>(tap - width) / original_rate -
                           static_cast<double>(phase) / target_rate) * base_rate;
            time = std::clamp(time, -filter_width, filter_width);
            const double window = std::pow(std::cos(time * M_PI / filter_width / 2.0), 2.0);
            const double angle = time * M_PI;
            const double sinc = std::abs(angle) < 1e-12 ? 1.0 : std::sin(angle) / angle;
            kernels[static_cast<size_t>(phase * kernel_size + tap)] =
                static_cast<float>(sinc * window * base_rate / original_rate);
        }
    }

    std::vector<float> output(target_samples, 0.f);
    for (uint64_t sample = 0; sample < target_samples; ++sample) {
        const int64_t frame = static_cast<int64_t>(sample / target_rate);
        const int64_t phase = static_cast<int64_t>(sample % target_rate);
        float value = 0.f;
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            const int64_t input_index = frame * original_rate + tap - width;
            if (input_index < 0 || input_index >= static_cast<int64_t>(source_samples)) {
                continue;
            }
            value += source.data[static_cast<uint64_t>(input_index) * source.channels + source_channel] *
                     kernels[static_cast<size_t>(phase * kernel_size + tap)];
        }
        output[sample] = std::clamp(value, -1.f, 1.f);
    }
    return output;
}

struct ScopedEnvVar {
    std::string name;
    std::string old_value;
    bool had_value = false;

    ScopedEnvVar(const char* key, const std::string& value)
        : name(key) {
        const char* current = std::getenv(key);
        if (current != nullptr) {
            had_value = true;
            old_value = current;
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (had_value) {
            setenv(name.c_str(), old_value.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> callback)
        : callback_(std::move(callback)) {}

    ~ScopeExit() {
        run_now();
    }

    void run_now() {
        if (callback_) {
            callback_();
            callback_ = nullptr;
        }
    }

private:
    std::function<void()> callback_;
};

bool set_minimax_error(std::string* error, const char* message) {
    if (error != nullptr) {
        *error = message;
    }
    LOG_ERROR("%s", message);
    return false;
}

uint8_t h3_to_u8(float value) {
    return static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

float h3_round_to_fp16(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

int64_t h3_resolve_seed(int64_t seed) {
    return seed >= 0 ? seed : static_cast<int64_t>(std::time(nullptr));
}

bool h3_profile_enabled();

bool h3_condition_debug_exit_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_CONDITION_DEBUG_EXIT");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::string h3_text_debug_target() {
    const char* value = std::getenv("ED_MINIMAX_H3_TEXT_DEBUG_TARGET");
    return value == nullptr ? std::string() : std::string(value);
}

std::string h3_dit_debug_target() {
    const char* value = std::getenv("ED_MINIMAX_H3_DEBUG_TARGET");
    return value == nullptr ? std::string() : std::string(value);
}

void h3_dump_vae_latent_if_requested(const sd::Tensor<float>& latent) {
    const char* base_path = std::getenv("ED_MINIMAX_H3_DUMP_VAE_LATENT");
    if (base_path == nullptr || base_path[0] == '\0') {
        return;
    }
    std::ofstream data_out(std::string(base_path) + ".f32.bin", std::ios::binary);
    if (data_out) {
        data_out.write(reinterpret_cast<const char*>(latent.values().data()),
                       static_cast<std::streamsize>(latent.values().size() * sizeof(float)));
    }
    std::ofstream shape_out(std::string(base_path) + ".shape");
    if (shape_out) {
        for (size_t index = 0; index < latent.shape().size(); ++index) {
            if (index > 0) {
                shape_out << ' ';
            }
            shape_out << latent.shape()[index];
        }
        shape_out << '\n';
    }
}

ed_tiling_params_t h3_vae_tiling(const ModelRuntime& runtime) {
    ed_tiling_params_t tiling = runtime.vae_tiling();
    if (tiling.force_disable) {
        tiling.enabled = false;
    }
    if (h3_profile_enabled()) {
        LOG_INFO("minimax-h3 profile VAE tiling: enabled=%d force-disable=%d rel=%.3f,%.3f",
                 tiling.enabled,
                 tiling.force_disable,
                 tiling.rel_size_x,
                 tiling.rel_size_y);
    }
    return tiling;
}

void h3_resize_for_vision(int source_width, int source_height, int* width, int* height) {
    constexpr int factor = 32;
    constexpr int min_pixels = 3136;
    constexpr int max_pixels = 12845056;
    int resized_width = std::max(factor, static_cast<int>(std::round(static_cast<double>(source_width) / factor)) * factor);
    int resized_height = std::max(factor, static_cast<int>(std::round(static_cast<double>(source_height) / factor)) * factor);
    const double area = static_cast<double>(resized_width) * resized_height;
    if (area > max_pixels) {
        const double scale = std::sqrt(static_cast<double>(source_width) * source_height / max_pixels);
        resized_width = std::max(factor, static_cast<int>(std::floor(source_width / scale / factor)) * factor);
        resized_height = std::max(factor, static_cast<int>(std::floor(source_height / scale / factor)) * factor);
    } else if (area < min_pixels) {
        const double scale = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(source_width) * source_height));
        resized_width = static_cast<int>(std::ceil(source_width * scale / factor)) * factor;
        resized_height = static_cast<int>(std::ceil(source_height * scale / factor)) * factor;
    }
    *width = resized_width;
    *height = resized_height;
}

void h3_reference_video_dimensions(const ed_image_t& image, int* width, int* height) {
    const double ratio = static_cast<double>(image.width) / image.height;
    double nominal_width = ratio >= 1.0 ? 768.0 * ratio : 768.0;
    double nominal_height = ratio >= 1.0 ? 768.0 : 768.0 / ratio;
    if (nominal_width * nominal_height > 768.0 * 1344.0) {
        const double scale = std::sqrt((768.0 * 1344.0) / (nominal_width * nominal_height));
        nominal_width *= scale;
        nominal_height *= scale;
    }
    *width = std::max(32, static_cast<int>(std::round(nominal_width / 32.0)) * 32);
    *height = std::max(32, static_cast<int>(std::round(nominal_height / 32.0)) * 32);
}

void h3_reference_image_dimensions(const ed_image_t& image,
                                   ed_ref_image_size_t size_mode,
                                   int canvas_width,
                                   int canvas_height,
                                   int* width,
                                   int* height) {
    if (size_mode == ED_REF_IMAGE_SIZE_MATCH) {
        const double source_area = static_cast<double>(image.width) * image.height;
        const double canvas_area = static_cast<double>(canvas_width) * canvas_height;
        const double scale = std::min(1.0, std::sqrt(canvas_area / source_area));
        *width = std::max(32, static_cast<int>(std::round(image.width * scale / 32.0)) * 32);
        *height = std::max(32, static_cast<int>(std::round(image.height * scale / 32.0)) * 32);
        return;
    }
    const double scale = static_cast<double>(H3_REF_IMAGE_SHORT_EDGE) /
                         static_cast<double>(std::min(image.width, image.height));
    *width = std::max(32, static_cast<int>(std::round(image.width * scale / 32.0)) * 32);
    *height = std::max(32, static_cast<int>(std::round(image.height * scale / 32.0)) * 32);
}

bool h3_reference_aspect_ratio_supported(const ed_image_t& image) {
    return image.width > 0 && image.height > 0 &&
           static_cast<int64_t>(image.width) <= 4 * static_cast<int64_t>(image.height) &&
           static_cast<int64_t>(image.height) <= 4 * static_cast<int64_t>(image.width);
}

float h3_discrete_flow_sigma(int step, int steps, float shift) {
    if (steps <= 1) {
        return step <= 0 ? 1.0f : 0.0f;
    }
    if (step >= steps) {
        return 0.0f;
    }
    const float t_max = 999.0f;
    const float t = t_max - (t_max / static_cast<float>(steps - 1)) * static_cast<float>(step);
    const float sigma = (t + 1.0f) / 1000.0f;
    return shift == 1.0f ? sigma
                         : shift * sigma / (1.0f + (shift - 1.0f) * sigma);
}

float h3_simple_flow_sigma(int step, int steps, float shift) {
    if (steps <= 0 || step >= steps) {
        return 0.0f;
    }
    const float sigma = static_cast<float>(steps - step) / static_cast<float>(steps);
    return shift == 1.0f ? sigma
                         : shift * sigma / (1.0f + (shift - 1.0f) * sigma);
}

bool h3_trace_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_TRACE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool h3_profile_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_PROFILE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool h3_fast_video_postprocess_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

bool h3_verify_fast_video_postprocess_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_VERIFY_FAST_VIDEO_POSTPROCESS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

int h3_fast_video_postprocess_threads() {
    const char* value = std::getenv("ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return 16;
    }
    char* end = nullptr;
    const long requested = std::strtol(value, &end, 10);
    return end != value && requested > 1 && requested <= std::numeric_limits<int>::max()
               ? static_cast<int>(requested)
               : 0;
}

void h3_trace_tensor(const char* name, const sd::Tensor<float>& tensor) {
    if (!h3_trace_enabled()) {
        return;
    }
    if (tensor.empty()) {
        LOG_INFO("minimax-h3 trace %s: empty", name);
        return;
    }

    uint64_t hash = 1469598103934665603ULL;
    double sum = 0.0;
    double squared_sum = 0.0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (float value : tensor.values()) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
        sum += value;
        squared_sum += static_cast<double>(value) * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const double count = static_cast<double>(tensor.numel());
    LOG_INFO("minimax-h3 trace %s: shape=%s n=%lld hash=%016llx mean=%.8g rms=%.8g min=%.8g max=%.8g",
             name,
             sd::tensor_shape_to_string(tensor.shape()).c_str(),
             static_cast<long long>(tensor.numel()),
             static_cast<unsigned long long>(hash),
             sum / count,
             std::sqrt(squared_sum / count),
             minimum,
             maximum);
}

void h3_dump_debug_tensor_if_requested(const std::string& name, const sd::Tensor<float>& tensor) {
    const char* base_path = std::getenv("ED_MINIMAX_H3_DEBUG_DUMP");
    if (base_path == nullptr || base_path[0] == '\0' || tensor.empty()) {
        return;
    }
    const std::string prefix = std::string(base_path) + "." + name;
    std::ofstream data_out(prefix + ".f32.bin", std::ios::binary);
    if (data_out) {
        data_out.write(reinterpret_cast<const char*>(tensor.values().data()),
                       static_cast<std::streamsize>(tensor.values().size() * sizeof(float)));
    }
    std::ofstream shape_out(prefix + ".shape");
    if (shape_out) {
        for (size_t index = 0; index < tensor.shape().size(); ++index) {
            if (index > 0) {
                shape_out << ' ';
            }
            shape_out << tensor.shape()[index];
        }
        shape_out << '\n';
    }
}

sd::Tensor<float> h3_sample_vae_moments(const sd::Tensor<float>& moments,
                                        std::shared_ptr<RNG> rng) {
    if (moments.empty() || moments.shape().size() < 4 || moments.shape()[3] != 48) {
        return {};
    }
    sd::Tensor<float> mean = sd::ops::slice(moments, 3, 0, 24);
    sd::Tensor<float> logvar = sd::ops::slice(moments, 3, 24, 48);
    sd::Tensor<float> stddev = sd::ops::exp(0.5f * sd::ops::clamp(logvar, -30.0f, 20.0f));
    sd::Tensor<float> noise = sd::randn_like<float>(mean, rng);
    sd::Tensor<float> latents = mean + stddev * noise;
    for (int64_t index = 0; index < latents.numel(); ++index) {
        latents[index] = h3_round_to_fp16(latents[index]);
    }
    return latents;
}

sd::Tensor<float> h3_encode_vae_condition(MiniMaxH3VAE::MiniMaxH3VideoVAERunner* vae,
                                          int n_threads,
                                          const sd::Tensor<float>& video,
                                          ed_tiling_params_t tiling) {
    auto encoded = vae->encode(n_threads, video, tiling);
    const char* verify_value = std::getenv("ED_MINIMAX_H3_VERIFY_VAE_ENCODER_OPTIMIZATIONS");
    if (encoded.empty() || verify_value == nullptr || verify_value[0] == '\0' || std::strcmp(verify_value, "0") == 0) {
        return encoded;
    }
    ScopedEnvVar disable_cudnn("ED_MINIMAX_H3_VAE_ENCODER_CUDNN_CONV3D", "0");
    ScopedEnvVar disable_cudnn_all("ED_MINIMAX_H3_VAE_ENCODER_CUDNN_CONV3D_ALL", "0");
    ScopedEnvVar disable_reflect_pad("ED_MINIMAX_H3_VAE_FUSED_REFLECT_PAD", "0");
    ScopedEnvVar disable_group_norm("ED_MINIMAX_H3_VAE_FUSED_TEMPORAL_GROUP_NORM", "0");
    auto baseline = vae->encode(n_threads, video, tiling);
    if (baseline.empty() || baseline.shape() != encoded.shape()) {
        LOG_ERROR("MiniMax-H3 VAE encoder optimization verification failed: output shape mismatch");
        return {};
    }
    double squared_error = 0.0;
    double squared_baseline = 0.0;
    double dot = 0.0;
    double squared_encoded = 0.0;
    float max_abs_error = 0.0f;
    for (int64_t index = 0; index < encoded.numel(); ++index) {
        const double reference = baseline[index];
        const double candidate = encoded[index];
        const double error_value = candidate - reference;
        squared_error += error_value * error_value;
        squared_baseline += reference * reference;
        squared_encoded += candidate * candidate;
        dot += reference * candidate;
        max_abs_error = std::max(max_abs_error, static_cast<float>(std::abs(error_value)));
    }
    const double mse = squared_error / std::max<int64_t>(1, encoded.numel());
    const double rmse = std::sqrt(mse);
    const double signal_rms = std::sqrt(squared_baseline / std::max<int64_t>(1, encoded.numel()));
    const double psnr = mse == 0.0 ? std::numeric_limits<double>::infinity()
                                   : 20.0 * std::log10(std::max(1e-30, signal_rms) / rmse);
    const double cosine = dot / std::max(1e-30, std::sqrt(squared_baseline * squared_encoded));
    LOG_INFO("MiniMax-H3 VAE encoder optimization verification: n=%lld rmse=%.9g signal-rms=%.9g psnr=%.6g dB max-abs=%.9g cosine=%.12g",
             static_cast<long long>(encoded.numel()), rmse, signal_rms, psnr, max_abs_error, cosine);
    return encoded;
}

void h3_apply_condition_noise(sd::Tensor<float>* latent,
                              const std::shared_ptr<RNG>& rng) {
    auto noise = sd::randn_like<float>(*latent, rng);
    *latent = *latent * MiniMaxH3::VISUAL_COND_TIMESTEP +
              noise * (1.0f - MiniMaxH3::VISUAL_COND_TIMESTEP);
}

sd::Tensor<float> h3_pack_audio_and_video_latents(const sd::Tensor<float>& video,
                                                   const sd::Tensor<float>& audio) {
    if (audio.empty()) {
        return video;
    }
    GGML_ASSERT(video.dim() == 5 && video.shape()[4] == 1);
    GGML_ASSERT(audio.dim() == 4 && audio.shape()[3] == 1);

    const int64_t spatial_size = video.shape()[0] * video.shape()[1] * video.shape()[2];
    const int64_t extra_channels = (audio.numel() + spatial_size - 1) / spatial_size;
    std::vector<int64_t> packed_shape = video.shape();
    packed_shape[3] += extra_channels;
    sd::Tensor<float> packed = sd::zeros<float>(packed_shape);
    std::copy_n(video.data(), video.numel(), packed.data());
    std::copy_n(audio.data(), audio.numel(), packed.data() + video.numel());
    return packed;
}

sd::Tensor<float> h3_image_to_tensor(const ed_image_t& image, int width, int height) {
    if (image.data == nullptr || image.width <= 0 || image.height <= 0 || image.channels <= 0) {
        return {};
    }
    const int source_width = static_cast<int>(image.width);
    const int source_height = static_cast<int>(image.height);
    const int source_channels = static_cast<int>(image.channels);
    sd::Tensor<float> source({source_width, source_height, 3, 1});
    for (int y = 0; y < source_height; ++y) {
        for (int x = 0; x < source_width; ++x) {
            const uint8_t* pixel = image.data +
                                   (static_cast<size_t>(y) * source_width + static_cast<size_t>(x)) * source_channels;
            for (int channel = 0; channel < 3; ++channel) {
                source.index(x, y, channel, 0) = static_cast<float>(pixel[std::min(channel, source_channels - 1)]) / 255.0f;
            }
        }
    }
    return sd::ops::interpolate(source,
                                std::vector<int64_t>{width, height, 3, 1},
                                sd::ops::InterpolateMode::Lanczos,
                                false,
                                true);
}

sd::Tensor<float> h3_image_to_tensor_cover_crop(const ed_image_t& image, int width, int height) {
    if (image.data == nullptr || image.width <= 0 || image.height <= 0 || image.channels <= 0) {
        return {};
    }
    const double scale = std::max(static_cast<double>(width) / static_cast<double>(image.width),
                                  static_cast<double>(height) / static_cast<double>(image.height));
    const int resized_width = std::max(width, static_cast<int>(std::round(static_cast<double>(image.width) * scale)));
    const int resized_height = std::max(height, static_cast<int>(std::round(static_cast<double>(image.height) * scale)));
    sd::Tensor<float> resized = h3_image_to_tensor(image, resized_width, resized_height);
    if (resized.empty()) {
        return {};
    }
    const int left = std::max(0, (resized_width - width) / 2);
    const int top = std::max(0, (resized_height - height) / 2);
    sd::Tensor<float> cropped({width, height, 3, 1});
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                cropped.index(x, y, channel, 0) = resized.index(x + left, y + top, channel, 0);
            }
        }
    }
    return cropped;
}

struct MiniMaxH3DetectedConfig {
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t token_refiner_num_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t attention_head_dim = 0;
    int64_t ffn_hidden_size = 0;
    int64_t video_latent_channels = 0;
    int64_t audio_latent_channels = 0;
    int64_t text_dim = 0;
    int64_t adaln_curve_grid = 0;
};

int64_t count_minimax_blocks(const String2TensorStorage& tensors, const std::string& prefix) {
    std::set<int> indices;
    for (const auto& item : tensors) {
        const std::string& name = item.first;
        if (!starts_with(name, prefix)) {
            continue;
        }
        const size_t begin = prefix.size();
        const size_t end = name.find('.', begin);
        if (end != std::string::npos) {
            indices.insert(std::atoi(name.substr(begin, end - begin).c_str()));
        }
    }
    return static_cast<int64_t>(indices.size());
}

MiniMaxH3DetectedConfig detect_minimax_config(const String2TensorStorage& tensors) {
    MiniMaxH3DetectedConfig config;
    const std::string prefix = "model.diffusion_model";
    auto find = [&](const std::string& suffix) -> const TensorStorage* {
        auto it = tensors.find(prefix + "." + suffix);
        return it == tensors.end() ? nullptr : &it->second;
    };
    if (const auto* weight = find("video_patch_proj.weight")) {
        config.video_latent_channels = weight->ne[0] / 4;
        config.hidden_size = weight->ne[1];
    }
    if (const auto* weight = find("audio_patch_proj.weight")) {
        config.audio_latent_channels = weight->ne[0];
    }
    config.num_layers = count_minimax_blocks(tensors, prefix + ".blocks.");
    config.token_refiner_num_layers = count_minimax_blocks(tensors, prefix + ".token_refiner.blocks.");
    if (const auto* weight = find("blocks.0.attn.q_norm.weight")) {
        config.attention_head_dim = weight->ne[0];
    }
    if (const auto* weight = find("blocks.0.attn.qkv_proj.weight")) {
        int64_t qkv_out_dim = weight->ne[1];
        if (const auto* k_weight = find("blocks.0.attn.qkv_proj.weight.1")) {
            qkv_out_dim += k_weight->ne[1];
        }
        if (const auto* v_weight = find("blocks.0.attn.qkv_proj.weight.2")) {
            qkv_out_dim += v_weight->ne[1];
        }
        config.num_attention_heads = qkv_out_dim / (3 * config.attention_head_dim);
    }
    if (const auto* weight = find("blocks.0.mlp.fc1.weight")) {
        config.ffn_hidden_size = weight->ne[1] / 2;
    }
    if (const auto* weight = find("condition_proj.weight")) {
        config.text_dim = weight->ne[0];
    }
    if (const auto* table = find("adaln_t_table")) {
        config.adaln_curve_grid = table->ne[1];
    }
    return config;
}

}  // namespace

struct MiniMaxH3Profile {
    int64_t total_ms = 0;
    int64_t cond_context_ms = 0;
    int64_t uncond_context_ms = 0;
    int64_t vision_image_prepare_ms = 0;
    int64_t vision_image_encode_ms = 0;
    int64_t vision_video_prepare_ms = 0;
    int64_t vision_video_encode_ms = 0;
    int64_t text_tokenize_ms = 0;
    int64_t text_encode_ms = 0;
    int64_t keyframe_vae_encode_ms = 0;
    int64_t reference_video_vae_encode_ms = 0;
    int64_t reference_audio_prepare_ms = 0;
    int64_t reference_audio_vae_encode_ms = 0;
    int64_t noise_init_ms = 0;
    int64_t diffusion_cond_ms = 0;
    int64_t diffusion_uncond_ms = 0;
    int64_t cfg_combine_ms = 0;
    int64_t video_vae_decode_ms = 0;
    int64_t video_postprocess_ms = 0;
    int64_t audio_vae_decode_ms = 0;
    int64_t audio_postprocess_ms = 0;
    int diffusion_steps = 0;
    int diffusion_calls = 0;

    void log() const {
        const int64_t diffusion_ms = diffusion_cond_ms + diffusion_uncond_ms + cfg_combine_ms;
        const int64_t conditioning_ms = cond_context_ms + uncond_context_ms + keyframe_vae_encode_ms +
                                        reference_video_vae_encode_ms + reference_audio_prepare_ms +
                                        reference_audio_vae_encode_ms;
        const int64_t decode_ms = video_vae_decode_ms + video_postprocess_ms + audio_vae_decode_ms + audio_postprocess_ms;
        LOG_INFO("minimax-h3 profile: total=%lld ms | conditioning=%lld ms | diffusion=%lld ms (%d steps, %d calls) | decode=%lld ms | noise=%lld ms",
                 static_cast<long long>(total_ms), static_cast<long long>(conditioning_ms),
                 static_cast<long long>(diffusion_ms), diffusion_steps, diffusion_calls,
                 static_cast<long long>(decode_ms), static_cast<long long>(noise_init_ms));
        LOG_INFO("minimax-h3 profile conditioning: context cond=%lld ms uncond=%lld ms | vision image prepare=%lld ms encode=%lld ms | vision video prepare=%lld ms encode=%lld ms | text tokenize=%lld ms encode=%lld ms | keyframe vae encode=%lld ms | ref visual vae encode=%lld ms | ref audio prepare=%lld ms vae encode=%lld ms",
                 static_cast<long long>(cond_context_ms), static_cast<long long>(uncond_context_ms),
                 static_cast<long long>(vision_image_prepare_ms), static_cast<long long>(vision_image_encode_ms),
                 static_cast<long long>(vision_video_prepare_ms), static_cast<long long>(vision_video_encode_ms),
                 static_cast<long long>(text_tokenize_ms), static_cast<long long>(text_encode_ms),
                 static_cast<long long>(keyframe_vae_encode_ms), static_cast<long long>(reference_video_vae_encode_ms),
                 static_cast<long long>(reference_audio_prepare_ms), static_cast<long long>(reference_audio_vae_encode_ms));
        LOG_INFO("minimax-h3 profile diffusion: conditional=%lld ms unconditional=%lld ms cfg-combine=%lld ms | decode video-vae=%lld ms video-copy=%lld ms audio-vae=%lld ms audio-copy=%lld ms",
                 static_cast<long long>(diffusion_cond_ms), static_cast<long long>(diffusion_uncond_ms),
                 static_cast<long long>(cfg_combine_ms), static_cast<long long>(video_vae_decode_ms),
                 static_cast<long long>(video_postprocess_ms), static_cast<long long>(audio_vae_decode_ms),
                 static_cast<long long>(audio_postprocess_ms));
    }
};

MiniMaxH3Pipeline::MiniMaxH3Pipeline(SDVersion version)
    : version_(version) {
}

MiniMaxH3Pipeline::~MiniMaxH3Pipeline() {
    if (sentinel_ctx_ != nullptr) {
        ggml_free(sentinel_ctx_);
        sentinel_ctx_ = nullptr;
        sentinel_tensor_ = nullptr;
    }
}

bool MiniMaxH3Pipeline::prepare(const ed_context_params_t& params,
                                ModelRuntime& runtime,
                                const ModelLoader& loader,
                                PipelineTensorRegistry& registry,
                                std::string* error) {
    (void)params;
    runtime_ = &runtime;
    registry.clear();
    const MiniMaxH3DetectedConfig config = detect_minimax_config(loader.get_tensor_storage_map());
    if (config.num_layers <= 0 || config.hidden_size <= 0 ||
        config.video_latent_channels <= 0 || config.audio_latent_channels <= 0) {
        return set_minimax_error(error, "MiniMax-H3 diffusion transformer signature is incomplete");
    }

    const bool has_audio_vae = std::any_of(loader.get_tensor_storage_map().begin(),
                                           loader.get_tensor_storage_map().end(),
                                           [](const auto& item) { return starts_with(item.first, "audio_vae."); });

    runtime.reset_auto_allocate_state();
    runtime.set_measured_dit_headroom(0);
    if (runtime.auto_fit() && runtime.fit_width() > 0 && runtime.fit_height() > 0 && runtime.fit_frames() > 0) {
        auto measure_runner = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(runtime.backend(),
                                                                          loader.get_tensor_storage_map(),
                                                                          "model.diffusion_model",
                                                                          false);
        measure_runner->set_flash_attention_enabled(runtime.flash_attention());
        const int latent_width = runtime.fit_width() / 16;
        const int latent_height = runtime.fit_height() / 16;
        const size_t measured = measure_runner->measure_compute_buffer_at(latent_width,
                                                                          latent_height,
                                                                          runtime.fit_frames());
        if (measured > 0) {
            runtime.set_measured_dit_headroom(measured);
        }
        LOG_INFO("auto-allocate: measured MiniMax-H3 DiT compute buffer = %.2f GB at latent %dx%dx%d",
                 measured / (1024.0 * 1024.0 * 1024.0),
                 latent_width,
                 latent_height,
                 runtime.fit_frames());
    }

    constexpr size_t minimax_staging_slack = static_cast<size_t>(3) * 1024 * 1024 * 1024;
    const size_t effective_budget = runtime.effective_budget_bytes();
    const size_t placement_budget = effective_budget > minimax_staging_slack
                                        ? effective_budget - minimax_staging_slack
                                        : effective_budget;
    size_t remaining_free = placement_budget;
    const bool diffusion_policy_offload = runtime.dit_offload_params_to_cpu() ||
                                          runtime.plan_component_offload(loader, "model.diffusion_model", remaining_free);
    const bool text_policy_offload = runtime.clip_offload_params_to_cpu() ||
                                     runtime.plan_component_offload(loader, "text_encoders.llm", remaining_free);
    const bool vae_policy_offload = runtime.vae_offload_params_to_cpu() ||
                                    runtime.plan_component_offload(loader, "first_stage_model", remaining_free);
    const bool audio_vae_policy_offload = runtime.vae_offload_params_to_cpu() ||
                                          (has_audio_vae && runtime.plan_component_offload(loader, "audio_vae", remaining_free));
    const bool lifecycle_requested = runtime.minimax_h3_stage_lifecycle() ||
                                     (runtime.auto_fit() && runtime.max_vram() > 0.0f);
    stage_diffusion_lifecycle_ = lifecycle_requested && !diffusion_policy_offload;
    stage_text_lifecycle_ = lifecycle_requested && !text_policy_offload;
    stage_video_vae_lifecycle_ = lifecycle_requested && !vae_policy_offload;
    stage_audio_vae_lifecycle_ = lifecycle_requested && has_audio_vae && !audio_vae_policy_offload;
    const bool diffusion_offload = diffusion_policy_offload || stage_diffusion_lifecycle_;
    const bool text_offload = text_policy_offload || stage_text_lifecycle_;
    const bool vae_offload = vae_policy_offload || stage_video_vae_lifecycle_;
    const bool audio_vae_offload = audio_vae_policy_offload || stage_audio_vae_lifecycle_;
    runtime.finalize_auto_segment_budget(effective_budget, minimax_staging_slack);

    diffusion_ = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(runtime.backend(),
                                                              loader.get_tensor_storage_map(),
                                                              "model.diffusion_model",
                                                              diffusion_offload);
    diffusion_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
    diffusion_->set_flash_attention_enabled(runtime.flash_attention());
    if (auto process_group = runtime.graph_process_group_ref()) {
        diffusion_->set_process_group(process_group);
    }

    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors(), "model.diffusion_model");

    const std::string rope_inv_freq_name = "model.diffusion_model.rope.inv_freq";
    if (loader.get_tensor_storage_map().find(rope_inv_freq_name) == loader.get_tensor_storage_map().end()) {
        auto rope_it = registry.tensors().find(rope_inv_freq_name);
        if (rope_it == registry.tensors().end() || rope_it->second == nullptr) {
            return set_minimax_error(error, "MiniMax-H3 RoPE parameter was not allocated");
        }
        ggml_tensor* rope_inv_freq = rope_it->second;
        std::vector<float> values(static_cast<size_t>(rope_inv_freq->ne[0]));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(std::pow(10000.0, -static_cast<double>(i) / values.size()));
        }
        ggml_backend_tensor_set(rope_inv_freq, values.data(), 0, values.size() * sizeof(float));
        LOG_INFO("MiniMax-H3 synthesized missing rope.inv_freq (%zu values)", values.size());
    }

    String2TensorStorage conditioner_tensors = loader.get_tensor_storage_map();
    const std::regex language_layer_pattern(R"(^text_encoders\.llm\.model\.layers\.(\d+)\.)");
    for (auto it = conditioner_tensors.begin(); it != conditioner_tensors.end();) {
        std::smatch match;
        const bool unused_language_layer = std::regex_search(it->first, match, language_layer_pattern) &&
                                           std::stoi(match[1].str()) >= 50;
        const bool unused_language_head = it->first == "text_encoders.llm.model.norm.weight" ||
                                          starts_with(it->first, "text_encoders.llm.lm_head.");
        if (unused_language_layer || unused_language_head) {
            it = conditioner_tensors.erase(it);
        } else {
            ++it;
        }
    }
    conditioner_ = std::make_unique<LLM::LLMEmbedder>(LLM::LLMArch::QWEN3_VL,
                                                       runtime.clip_backend(),
                                                       text_offload,
                                                       conditioner_tensors,
                                                       "text_encoders.llm",
                                                       true);
    conditioner_->model.set_flash_attention_enabled(runtime.flash_attention());
    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors(), "text_encoders.llm");
    conditioner_->model.set_max_graph_vram_bytes(
        runtime.text_encoder_segment_budget(conditioner_->model.get_params_buffer_size(), text_offload));

    vae_ = std::make_unique<MiniMaxH3VAE::MiniMaxH3VideoVAERunner>(runtime.vae_backend(),
                                                                    vae_offload,
                                                                    loader.get_tensor_storage_map(),
                                                                    "first_stage_model");
    vae_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
    vae_->set_flash_attention_enabled(runtime.flash_attention());
    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), "first_stage_model");

    if (has_audio_vae) {
        audio_vae_ = std::make_unique<MiniMaxH3Audio::AudioVAERunner>(runtime.vae_backend(),
                                                                        audio_vae_offload,
                                                                        loader.get_tensor_storage_map(),
                                                                        "audio_vae");
        audio_vae_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
        audio_vae_->set_flash_attention_enabled(runtime.flash_attention());
        audio_vae_->alloc_params_buffer();
        audio_vae_->get_param_tensors(registry.tensors(), "audio_vae");
    } else {
        LOG_INFO("MiniMax-H3 audio VAE not provided; generated video will have no decoded audio track");
    }

    registry.ignore_prefix("first_stage_model.encoder.");
    registry.ignore_prefix("text_encoders.llm.visual.");
    registry.ignore_prefix("text_encoders.llm.");
    registry.ignore_prefix("model.diffusion_model.rope.inv_freq");

    if (sentinel_ctx_ == nullptr) {
        ggml_init_params init_params{};
        init_params.mem_size = ggml_tensor_overhead() + 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc = false;
        sentinel_ctx_ = ggml_init(init_params);
        if (sentinel_ctx_ == nullptr) {
            return set_minimax_error(error, "failed to allocate MiniMax-H3 sentinel tensor context");
        }
        sentinel_tensor_ = ggml_new_tensor_1d(sentinel_ctx_, GGML_TYPE_F32, 1);
    }
    registry.add("__ed_minimax_h3_sentinel.weight", sentinel_tensor_);

    ready_ = true;
    LOG_INFO("MiniMax-H3 detected: layers=%lld token_refiner_layers=%lld hidden=%lld heads=%lld head_dim=%lld ffn=%lld video_latent=%lld audio_latent=%lld text_dim=%lld adaln_curve_grid=%lld",
             (long long)config.num_layers,
             (long long)config.token_refiner_num_layers,
             (long long)config.hidden_size,
             (long long)config.num_attention_heads,
             (long long)config.attention_head_dim,
             (long long)config.ffn_hidden_size,
             (long long)config.video_latent_channels,
             (long long)config.audio_latent_channels,
             (long long)config.text_dim,
             (long long)config.adaln_curve_grid);
    return true;
}

void MiniMaxH3Pipeline::mark_ready() {
    ready_ = true;
}

ed_status_t MiniMaxH3Pipeline::generate_image(const ed_image_generation_params_t*,
                                              ed_image_batch_t*,
                                              std::string* error) {
    set_minimax_error(error, "MiniMax-H3 supports video generation only");
    return ED_STATUS_UNSUPPORTED;
}

bool MiniMaxH3Pipeline::build_text_context(const char* prompt,
                                           const ed_image_t* init_image,
                                           const ed_image_t* end_image,
                                           int canvas_width,
                                           int canvas_height,
                                           const ed_image_t* ref_images,
                                           int ref_image_count,
                                           ed_ref_image_size_t ref_image_size,
                                           const ed_ref_video_t* ref_videos,
                                           int ref_video_count,
                                           int ref_audio_count,
                                           int max_video_frames,
                                           sd::Tensor<float>* context,
                                           sd::Tensor<int32_t>* token_tags,
                                           MiniMaxH3Profile* profile,
                                           std::string* error) {
    if (context == nullptr || token_tags == nullptr || conditioner_ == nullptr || runtime_ == nullptr) {
        return set_minimax_error(error, "MiniMax-H3 text conditioner is not initialized");
    }
    std::string pending_text;
    std::vector<int> tokens;
    int64_t tokenization_ms = 0;
    const char* verify_token_build_value = std::getenv("ED_MINIMAX_H3_VERIFY_DIRECT_TOKEN_BUILD");
    const bool verify_token_build = verify_token_build_value != nullptr &&
                                   verify_token_build_value[0] != '\0' &&
                                   std::strcmp(verify_token_build_value, "0") != 0;
    std::string legacy_presentation;
    std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
    std::vector<LLM::LLMImageEmbedInfo> image_embed_infos;
    std::vector<std::vector<std::pair<int, sd::Tensor<float>>>> deepstack_image_embeds(3);
    const int vision_patch_size = conditioner_->model.params.vision.patch_size;
    const std::string vision_start = "<|vision_start|>";
    const std::string vision_end = "<|vision_end|>";
    const std::string image_pad = "<|image_pad|>";
    const auto vision_start_tokens = conditioner_->tokenizer->tokenize(vision_start, nullptr, true, 0, 0, false);
    const auto vision_end_tokens = conditioner_->tokenizer->tokenize(vision_end, nullptr, true, 0, 0, false);
    const auto image_pad_tokens = conditioner_->tokenizer->tokenize(image_pad, nullptr, true, 0, 0, false);
    if (vision_start_tokens.size() != 1 || vision_end_tokens.size() != 1 || image_pad_tokens.size() != 1) {
        return set_minimax_error(error, "MiniMax-H3 tokenizer special tokens are invalid");
    }
    auto flush_text = [&]() {
        if (pending_text.empty()) {
            return;
        }
        const int64_t begin = profile != nullptr ? ggml_time_ms() : 0;
        auto text_tokens = conditioner_->tokenizer->tokenize(pending_text, nullptr, false, 0, 0, false);
        if (profile != nullptr) {
            tokenization_ms += ggml_time_ms() - begin;
        }
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
        pending_text.clear();
    };
    auto append_text = [&](const std::string& text) {
        flush_text();
        pending_text = text;
        if (verify_token_build) {
            legacy_presentation += text;
        }
    };
    auto append_vision = [&](int64_t count) {
        flush_text();
        tokens.push_back(vision_start_tokens[0]);
        const int embed_index = static_cast<int>(tokens.size());
        tokens.insert(tokens.end(), static_cast<size_t>(count), image_pad_tokens[0]);
        tokens.push_back(vision_end_tokens[0]);
        if (verify_token_build) {
            legacy_presentation += vision_start;
            for (int64_t index = 0; index < count; ++index) {
                legacy_presentation += image_pad;
            }
            legacy_presentation += vision_end;
        }
        return embed_index;
    };
    auto add_vision_image = [&](const ed_image_t& image,
                                int image_index,
                                int width,
                                int height,
                                bool keyframe) -> bool {
        const int64_t prepare_begin = profile != nullptr ? ggml_time_ms() : 0;
        sd::Tensor<float> image_tensor = keyframe && image_index > 0
                                             ? h3_image_to_tensor_cover_crop(image, width, height)
                                             : h3_image_to_tensor(image, width, height);
        if (image_tensor.empty()) {
            return set_minimax_error(error, keyframe ? "MiniMax-H3 keyframe vision image is invalid"
                                                     : "MiniMax-H3 Ref2VA image reference is invalid");
        }
        image_tensor = image_tensor * 2.0f - 1.0f;
        if (profile != nullptr) profile->vision_image_prepare_ms += ggml_time_ms() - prepare_begin;
        const int64_t encode_begin = profile != nullptr ? ggml_time_ms() : 0;
        std::vector<sd::Tensor<float>> image_outputs = conditioner_->model.encode_image_outputs(runtime_->n_threads(), image_tensor);
        if (profile != nullptr) profile->vision_image_encode_ms += ggml_time_ms() - encode_begin;
        if (image_outputs.size() != 4 || image_outputs[0].empty()) {
            return set_minimax_error(error, keyframe ? "MiniMax-H3 keyframe vision encoder failed"
                                                     : "MiniMax-H3 Ref2VA vision encoder failed");
        }
        sd::Tensor<float> image_embed = std::move(image_outputs[0]);
        const int64_t image_tokens = image_embed.shape()[1];
        append_text("<Picture " + std::to_string(image_index + 1) + ">: ");
        const int embed_index = append_vision(image_tokens);
        image_embeds.emplace_back(embed_index, std::move(image_embed));
        for (size_t layer = 0; layer < deepstack_image_embeds.size(); ++layer) {
            deepstack_image_embeds[layer].emplace_back(embed_index, std::move(image_outputs[layer + 1]));
        }
        image_embed_infos.push_back({embed_index, image_tokens, 1, height / vision_patch_size, width / vision_patch_size});
        return true;
    };
    int keyframe_image_index = 0;
    if (init_image != nullptr && init_image->data != nullptr) {
        if (!add_vision_image(*init_image, keyframe_image_index++, canvas_width, canvas_height, true)) {
            return false;
        }
    }
    if (end_image != nullptr && end_image->data != nullptr) {
        if (!add_vision_image(*end_image, keyframe_image_index++, canvas_width, canvas_height, true)) {
            return false;
        }
    }
    for (int image_index = 0; image_index < ref_image_count; ++image_index) {
        const ed_image_t& image = ref_images[image_index];
        int width = 0;
        int height = 0;
        h3_reference_image_dimensions(image, ref_image_size, canvas_width, canvas_height, &width, &height);
        LOG_DEBUG("MiniMax-H3 Ref2VA vision image=%dx%d tensor=%s",
                  width,
                  height,
                  "");
        if (!add_vision_image(image, image_index, width, height, false)) {
            return false;
        }
    }
    int video_number = 0;
    int audio_number = 0;
    for (int video_index = 0; video_index < ref_video_count; ++video_index) {
        const ed_ref_video_t& reference = ref_videos[video_index];
        if (reference.frames == nullptr || reference.frame_count <= 0) {
            return set_minimax_error(error, "MiniMax-H3 Ref2VA video reference is invalid");
        }
        const int source_fps = reference.fps > 0 ? reference.fps : 24;
        int normalized_frames = static_cast<int>(std::lround(static_cast<double>(reference.frame_count) * 24.0 / source_fps));
        normalized_frames = std::min(normalized_frames, max_video_frames);
        if (normalized_frames < 5) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video reference needs at least 5 frames at 24 fps");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        if (reference.audio.data != nullptr && reference.audio.sample_count > 0) {
            append_text("<Audio " + std::to_string(++audio_number) + ">: ");
        }
        append_text("<Video " + std::to_string(++video_number) + ">: ");
        while (normalized_frames % 17 != 5) --normalized_frames;
        std::vector<int> sampled_frames;
        sampled_frames.reserve(static_cast<size_t>((normalized_frames + 11) / 12));
        for (int frame = 0; frame < normalized_frames; frame += 12) {
            sampled_frames.push_back(frame);
        }
        for (size_t sampled_index = 0; sampled_index < sampled_frames.size(); sampled_index += 2) {
            const int64_t prepare_begin = profile != nullptr ? ggml_time_ms() : 0;
            const size_t next_sampled_index = std::min(sampled_index + 1, sampled_frames.size() - 1);
            const int first_frame = sampled_frames[sampled_index];
            const int second_frame = sampled_frames[next_sampled_index];
            const int first_index = std::min(reference.frame_count - 1, static_cast<int>(std::floor(first_frame * source_fps / 24.0)));
            const int second_index = std::min(reference.frame_count - 1, static_cast<int>(std::floor(second_frame * source_fps / 24.0)));
            const ed_image_t& first_image = reference.frames[first_index];
            int width = 0;
            int height = 0;
            h3_reference_video_dimensions(first_image, &width, &height);
            if (h3_trace_enabled() && sampled_index == 0) {
                LOG_INFO("MiniMax-H3 Ref2VA video %d vision geometry: source=%dx%d normalized=%dx%d frames=%d",
                         video_index + 1,
                         first_image.width,
                         first_image.height,
                         width,
                         height,
                         normalized_frames);
            }
            auto first = h3_image_to_tensor(first_image, width, height);
            auto second = h3_image_to_tensor(reference.frames[second_index], width, height);
            if (first.empty() || second.empty()) return set_minimax_error(error, "MiniMax-H3 Ref2VA video frame is invalid");
            sd::Tensor<float> pair({width, height, 2, 3, 1});
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        pair.index(x, y, 0, channel, 0) = first.index(x, y, channel, 0) * 2.f - 1.f;
                        pair.index(x, y, 1, channel, 0) = second.index(x, y, channel, 0) * 2.f - 1.f;
                    }
                }
            }
            if (profile != nullptr) profile->vision_video_prepare_ms += ggml_time_ms() - prepare_begin;
            const int64_t encode_begin = profile != nullptr ? ggml_time_ms() : 0;
            auto outputs = conditioner_->model.encode_video_block_outputs(runtime_->n_threads(), pair);
            if (profile != nullptr) profile->vision_video_encode_ms += ggml_time_ms() - encode_begin;
            if (outputs.size() != 4 || outputs[0].empty()) return set_minimax_error(error, "MiniMax-H3 Ref2VA video vision encoder failed");
            const float first_time = static_cast<float>(first_frame) / 24.0f;
            const float second_time = static_cast<float>(second_frame) / 24.0f;
            std::ostringstream timestamp_stream;
            timestamp_stream << '<' << std::fixed << std::setprecision(1)
                             << (first_time + second_time) * 0.5f << " seconds>";
            const std::string timestamp = timestamp_stream.str();
            append_text(timestamp);
            const int64_t vision_tokens = outputs[0].shape()[1];
            const int embed_index = append_vision(vision_tokens);
            image_embeds.emplace_back(embed_index, std::move(outputs[0]));
            for (size_t layer = 0; layer < deepstack_image_embeds.size(); ++layer) deepstack_image_embeds[layer].emplace_back(embed_index, std::move(outputs[layer + 1]));
            image_embed_infos.push_back({embed_index, vision_tokens, 1, height / vision_patch_size, width / vision_patch_size});
        }
    }
    for (int audio_index = 0; audio_index < ref_audio_count; ++audio_index) {
        append_text("<Audio " + std::to_string(++audio_number) + ">: ");
    }
    append_text(prompt == nullptr ? "" : prompt);
    flush_text();
    if (profile != nullptr) profile->text_tokenize_ms += tokenization_ms;
    if (verify_token_build) {
        LOG_INFO("MiniMax-H3 segmented presentation token build: tokens=%zu bytes=%zu",
                 tokens.size(),
                 legacy_presentation.size());
    }
    if (tokens.empty()) {
        return set_minimax_error(error, "MiniMax-H3 prompt tokenization produced no tokens");
    }
    std::vector<int32_t> tags(tokens.size(), 1);
    for (const auto& [index, image_embed] : image_embeds) {
        const int64_t begin = std::max<int64_t>(0, index - 1);
        const int64_t end = std::min<int64_t>(static_cast<int64_t>(tags.size()),
                                              index + image_embed.shape()[1] + 1);
        std::fill(tags.begin() + begin, tags.begin() + end, 0);
    }
    if (h3_trace_enabled()) {
        const auto zero_tags = std::count(tags.begin(), tags.end(), 0);
        std::ostringstream summary;
        summary << "MiniMax-H3 presentation: tokens=" << tokens.size()
                << " video-tags=" << zero_tags
                << " text-tags=" << tags.size() - static_cast<size_t>(zero_tags)
                << " head=";
        for (size_t index = 0; index < std::min<size_t>(tokens.size(), 16); ++index) {
            if (index > 0) summary << ',';
            summary << tokens[index];
        }
        summary << " tail=";
        const size_t tail_begin = tokens.size() > 16 ? tokens.size() - 16 : 0;
        for (size_t index = tail_begin; index < tokens.size(); ++index) {
            if (index > tail_begin) summary << ',';
            summary << tokens[index];
        }
        LOG_INFO("%s", summary.str().c_str());
    }
    sd::Tensor<int32_t> ids({static_cast<int64_t>(tokens.size())}, tokens);
    const int64_t text_encode_begin = profile != nullptr ? ggml_time_ms() : 0;
    *context = conditioner_->model.compute(runtime_->n_threads(), ids, {}, image_embeds, {50}, image_embed_infos, h3_text_debug_target(), deepstack_image_embeds);
    if (profile != nullptr) profile->text_encode_ms += ggml_time_ms() - text_encode_begin;
    if (context->empty()) {
        return set_minimax_error(error, "MiniMax-H3 text encoder compute failed");
    }
    if (static_cast<int64_t>(tags.size()) != context->shape()[1]) {
        return set_minimax_error(error, "MiniMax-H3 presentation tags do not match conditioner output");
    }
    const int64_t tag_count = static_cast<int64_t>(tags.size());
    *token_tags = sd::Tensor<int32_t>({tag_count}, std::move(tags));
    h3_trace_tensor("text_context", *context);
    return true;
}

ed_status_t MiniMaxH3Pipeline::decode_video_latent(const sd::Tensor<float>& latent,
                                                    int requested_frames,
                                                    ed_video_t* out,
                                                    MiniMaxH3Profile* profile,
                                                    std::string* error) {
    if (stage_video_vae_lifecycle_ && !vae_->stage_params_for_phase()) {
        set_minimax_error(error, "MiniMax-H3 failed to stage video VAE for decode");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    ScopeExit release_vae([this]() {
        if (vae_ != nullptr) {
            vae_->release_params_after_phase();
        }
    });
    sd::Tensor<float> vae_latent = vae_->diffusion_to_vae_latents(latent);
    h3_dump_vae_latent_if_requested(vae_latent);
    ed_tiling_params_t tiling = h3_vae_tiling(*runtime_);
    const int64_t decode_begin = profile != nullptr ? ggml_time_ms() : 0;
    sd::Tensor<float> video = vae_->decode(runtime_->n_threads(), vae_latent, tiling, true);
    if (profile != nullptr) profile->video_vae_decode_ms += ggml_time_ms() - decode_begin;
    if (video.empty() || video.dim() != 5) {
        set_minimax_error(error, "MiniMax-H3 video VAE decode failed");
        return ED_STATUS_GENERATION_FAILED;
    }
    const int decoded_frames = static_cast<int>(video.shape()[2]);
    const int frames_count = std::max(decoded_frames, requested_frames);
    ed_image_t* frames = static_cast<ed_image_t*>(std::calloc(static_cast<size_t>(frames_count), sizeof(ed_image_t)));
    if (frames == nullptr) {
        set_minimax_error(error, "failed to allocate MiniMax-H3 output frames");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    const size_t width = static_cast<size_t>(video.shape()[0]);
    const size_t height = static_cast<size_t>(video.shape()[1]);
    const size_t channels = static_cast<size_t>(video.shape()[3]);
    const size_t pixels = width * height;
    const int64_t postprocess_begin = profile != nullptr ? ggml_time_ms() : 0;
    const bool fast_postprocess = h3_fast_video_postprocess_enabled();
    const bool verify_fast_postprocess = fast_postprocess && h3_verify_fast_video_postprocess_enabled();
    const float* video_data     = video.data();
    for (int frame = 0; frame < frames_count; ++frame) {
        frames[frame].width = static_cast<int>(width);
        frames[frame].height = static_cast<int>(height);
        frames[frame].channels = static_cast<int>(channels);
        frames[frame].data = static_cast<uint8_t*>(std::malloc(pixels * channels));
        if (frames[frame].data == nullptr) {
            for (int index = 0; index < frame; ++index) std::free(frames[index].data);
            std::free(frames);
            set_minimax_error(error, "failed to allocate MiniMax-H3 frame pixels");
            return ED_STATUS_OUT_OF_MEMORY;
        }
    }
    auto convert_frame = [&](int frame) {
        const int source_frame = std::min(frame, decoded_frames - 1);
        if (fast_postprocess) {
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                const size_t x = pixel % width;
                const size_t y = pixel / width;
                for (size_t channel = 0; channel < channels; ++channel) {
                    const size_t offset = x + width * (y + height * (source_frame + decoded_frames * channel));
                    if (verify_fast_postprocess) {
                        GGML_ASSERT(video_data[offset] == video.index(x, y, source_frame, channel, 0));
                    }
                    frames[frame].data[pixel * channels + channel] = h3_to_u8(video_data[offset]);
                }
            }
        } else {
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                for (size_t channel = 0; channel < channels; ++channel) {
                    frames[frame].data[pixel * channels + channel] =
                        h3_to_u8(video.index(pixel % width, pixel / width, source_frame, channel, 0));
                }
            }
        }
    };
    const int requested_threads = fast_postprocess ? h3_fast_video_postprocess_threads() : 0;
    const int conversion_threads = std::min(frames_count, requested_threads);
    if (conversion_threads > 1) {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(conversion_threads));
        for (int worker = 0; worker < conversion_threads; ++worker) {
            workers.emplace_back([&, worker]() {
                for (int frame = worker; frame < frames_count; frame += conversion_threads) {
                    convert_frame(frame);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    } else {
        for (int frame = 0; frame < frames_count; ++frame) {
            convert_frame(frame);
        }
    }
    out->frames = frames;
    out->frame_count = frames_count;
    if (profile != nullptr) profile->video_postprocess_ms += ggml_time_ms() - postprocess_begin;
    return ED_STATUS_OK;
}

bool MiniMaxH3Pipeline::decode_audio_latent(const sd::Tensor<float>& latent,
                                            ed_video_t* out,
                                            MiniMaxH3Profile* profile,
                                            std::string* error) {
    if (audio_vae_ == nullptr || latent.empty()) {
        return set_minimax_error(error, "MiniMax-H3 audio VAE is not initialized");
    }
    if (stage_audio_vae_lifecycle_ && !audio_vae_->stage_params_for_phase()) {
        return set_minimax_error(error, "MiniMax-H3 failed to stage audio VAE for decode");
    }
    ScopeExit release_audio_vae([this]() {
        if (audio_vae_ != nullptr) {
            audio_vae_->release_params_after_phase();
        }
    });
    const int64_t decode_begin = profile != nullptr ? ggml_time_ms() : 0;
    sd::Tensor<float> waveform = audio_vae_->decode(runtime_->n_threads(), latent);
    if (profile != nullptr) profile->audio_vae_decode_ms += ggml_time_ms() - decode_begin;
    if (waveform.empty() || waveform.dim() != 4 || waveform.shape()[1] != 2) {
        return set_minimax_error(error, "MiniMax-H3 audio VAE decode failed");
    }
    const int64_t sample_count = waveform.shape()[0];
    const int channels = static_cast<int>(waveform.shape()[1]);
    if (sample_count <= 0 || sample_count > std::numeric_limits<int>::max()) {
        return set_minimax_error(error, "MiniMax-H3 audio sample count is invalid");
    }
    float* samples = static_cast<float*>(std::malloc(static_cast<size_t>(sample_count) * channels * sizeof(float)));
    if (samples == nullptr) {
        return set_minimax_error(error, "failed to allocate MiniMax-H3 audio output");
    }
    const int64_t postprocess_begin = profile != nullptr ? ggml_time_ms() : 0;
    for (int64_t sample = 0; sample < sample_count; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
            samples[sample * channels + channel] = std::clamp(waveform.index(sample, channel, 0, 0), -1.0f, 1.0f);
        }
    }
    out->audio = samples;
    out->audio_sample_count = static_cast<int>(sample_count);
    out->audio_channels = channels;
    out->audio_sample_rate = 32000;
    if (profile != nullptr) profile->audio_postprocess_ms += ggml_time_ms() - postprocess_begin;
    return true;
}

ed_status_t MiniMaxH3Pipeline::generate_video(const ed_video_generation_params_t* params,
                                              ed_video_t* out,
                                              std::string* error) {
    MiniMaxH3Profile profile;
    MiniMaxH3Profile* profile_ptr = h3_profile_enabled() ? &profile : nullptr;
    const int64_t generation_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    if (out == nullptr || params == nullptr) {
        set_minimax_error(error, "MiniMax-H3 video parameters or output are null");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    out->frames = nullptr;
    out->frame_count = 0;
    if (!ready_ || runtime_ == nullptr || !diffusion_ || !conditioner_ || !vae_) {
        set_minimax_error(error, "MiniMax-H3 pipeline is not ready");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (params->width <= 0 || params->height <= 0 || params->frames <= 0 ||
        params->width % 32 != 0 || params->height % 32 != 0) {
        set_minimax_error(error, "MiniMax-H3 width and height must be positive multiples of 32");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (static_cast<int64_t>(params->width) * params->height < H3_MIN_GENERATION_PIXELS) {
        set_minimax_error(error,
                          "MiniMax-H3 output canvas must contain at least 65536 pixels (for example 256x256); "
                          "the official recommended output uses a 768-pixel short edge");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const int frames = std::max(5, params->frames);
    if (frames % 17 != 5) {
        set_minimax_error(error, "MiniMax-H3 frame count must satisfy 17k + 5");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const bool has_keyframes = (params->init_image != nullptr && params->init_image->data != nullptr) ||
                               (params->end_image != nullptr && params->end_image->data != nullptr);
    const bool has_references = params->ref_image_count > 0 || params->ref_video_count > 0 || params->ref_audio_count > 0;
    if (has_keyframes && has_references) {
        set_minimax_error(error, "MiniMax-H3 Ref2VA references cannot be combined with init or end keyframes");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (params->ref_image_count < 0 || params->ref_video_count < 0 || params->ref_audio_count < 0 ||
        (params->ref_image_count > 0 && params->ref_images == nullptr) ||
        (params->ref_video_count > 0 && params->ref_videos == nullptr) ||
        (params->ref_audio_count > 0 && params->ref_audios == nullptr)) {
        set_minimax_error(error, "MiniMax-H3 Ref2VA reference arrays are invalid");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (params->ref_audio_count > 0 && params->ref_image_count == 0 && params->ref_video_count == 0) {
        set_minimax_error(error, "MiniMax-H3 Ref2VA audio references must be paired with at least one image or video reference");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    for (int image_index = 0; image_index < params->ref_image_count; ++image_index) {
        if (!h3_reference_aspect_ratio_supported(params->ref_images[image_index])) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA reference images must have a positive size and an aspect ratio from 1:4 to 4:1");
            return ED_STATUS_INVALID_ARGUMENT;
        }
    }
    for (int video_index = 0; video_index < params->ref_video_count; ++video_index) {
        const ed_ref_video_t& reference = params->ref_videos[video_index];
        if (reference.frames == nullptr || reference.frame_count <= 0 ||
            !h3_reference_aspect_ratio_supported(reference.frames[0])) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA reference videos must have frames with a positive size and an aspect ratio from 1:4 to 4:1");
            return ED_STATUS_INVALID_ARGUMENT;
        }
    }
    const int64_t resolved_seed = h3_resolve_seed(params->seed);
    bool need_audio_vae_for_conditioning = params->ref_audio_count > 0;
    for (int video_index = 0; video_index < params->ref_video_count; ++video_index) {
        const ed_audio_t& audio = params->ref_videos[video_index].audio;
        need_audio_vae_for_conditioning = need_audio_vae_for_conditioning ||
                                          (audio.data != nullptr && audio.sample_count > 0);
    }
    const bool need_video_vae_for_conditioning = has_keyframes ||
                                                  params->ref_image_count > 0 ||
                                                  params->ref_video_count > 0;
    if (stage_text_lifecycle_ && !conditioner_->model.stage_params_for_phase()) {
        set_minimax_error(error, "MiniMax-H3 failed to stage text encoder for conditioning");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    ScopeExit release_text_encoder([this]() {
        if (conditioner_ != nullptr) {
            conditioner_->model.release_params_after_phase();
        }
    });
    sd::Tensor<float> context;
    sd::Tensor<int32_t> token_tags;
    const int64_t cond_context_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    if (!build_text_context(params->prompt,
                            params->init_image,
                            params->end_image,
                            params->width,
                            params->height,
                            params->ref_images,
                            params->ref_image_count,
                            params->ref_image_size,
                            params->ref_videos,
                            params->ref_video_count,
                            params->ref_audio_count,
                            frames,
                            &context,
                            &token_tags,
                            profile_ptr,
                            error)) return ED_STATUS_GENERATION_FAILED;
    if (profile_ptr != nullptr) profile.cond_context_ms = ggml_time_ms() - cond_context_begin;
    if (!h3_text_debug_target().empty()) {
        set_minimax_error(error, "MiniMax-H3 text debug target completed");
        return ED_STATUS_GENERATION_FAILED;
    }
    const float cfg_scale = params->sample.cfg_scale;
    const bool use_cfg = cfg_scale != 1.0f;
    sd::Tensor<float> uncond_context;
    sd::Tensor<int32_t> uncond_token_tags;
    const int64_t uncond_context_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    if (use_cfg && !build_text_context(params->negative_prompt,
                                       params->init_image,
                                       params->end_image,
                                       params->width,
                                       params->height,
                                       params->ref_images,
                                       params->ref_image_count,
                                       params->ref_image_size,
                                       params->ref_videos,
                                       params->ref_video_count,
                                       params->ref_audio_count,
                                       frames,
                                       &uncond_context,
                                       &uncond_token_tags,
                                       profile_ptr,
                                       error)) {
        return ED_STATUS_GENERATION_FAILED;
    }
    if (profile_ptr != nullptr && use_cfg) profile.uncond_context_ms = ggml_time_ms() - uncond_context_begin;
    release_text_encoder.run_now();
    const int latent_frames = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
    const int audio_length = std::max(1, static_cast<int>(std::lround(static_cast<double>(frames) * 40.0 / 24.0)));
    const int latent_width = params->width / 16;
    const int latent_height = params->height / 16;
    std::vector<sd::Tensor<float>> keyframe_latents;
    std::vector<int32_t> keyframe_indices;
    auto request_rng = std::make_shared<PhiloxRNG>(static_cast<uint64_t>(resolved_seed));
    if (stage_video_vae_lifecycle_ && need_video_vae_for_conditioning && !vae_->stage_params_for_phase()) {
        set_minimax_error(error, "MiniMax-H3 failed to stage video VAE for conditioning");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    ScopeExit release_video_vae([this, need_video_vae_for_conditioning]() {
        if (need_video_vae_for_conditioning && vae_ != nullptr) {
            vae_->release_params_after_phase();
        }
    });
    auto add_keyframe = [&](const ed_image_t* image, int32_t frame_index, const char* name) -> bool {
        if (image == nullptr || image->data == nullptr) {
            return true;
        }
        sd::Tensor<float> image_tensor = frame_index == 0
                                             ? h3_image_to_tensor(*image, params->width, params->height)
                                             : h3_image_to_tensor_cover_crop(*image, params->width, params->height);
        if (image_tensor.empty()) {
            return set_minimax_error(error, "MiniMax-H3 keyframe image is invalid");
        }
        sd::Tensor<float> video_image = image_tensor.reshape({params->width, params->height, 1, 3, 1});
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        sd::Tensor<float> vae_latent = h3_encode_vae_condition(vae_.get(),
                                                               runtime_->n_threads(),
                                                               video_image,
                                                               h3_vae_tiling(*runtime_));
        if (profile_ptr != nullptr) profile.keyframe_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (vae_latent.empty()) {
            return set_minimax_error(error, "MiniMax-H3 keyframe VAE encode failed");
        }
        sd::Tensor<float> latent = vae_->vae_to_diffusion_latents(vae_latent);
        h3_apply_condition_noise(&latent, request_rng);
        h3_trace_tensor((std::string(name) + "_keyframe_latent").c_str(), latent);
        keyframe_latents.push_back(std::move(latent));
        keyframe_indices.push_back(frame_index);
        return true;
    };
    if (!add_keyframe(params->init_image, 0, "init") ||
        !add_keyframe(params->end_image, frames - 1, "end")) {
        return ED_STATUS_GENERATION_FAILED;
    }
    std::vector<sd::Tensor<float>> reference_latents;
    std::vector<sd::Tensor<float>> reference_audio_latents;
    std::vector<MiniMaxH3ReferenceBlock> reference_blocks;
    std::vector<size_t> video_reference_block_indices;
    auto encode_reference_audio = [&](const ed_audio_t& source, int32_t* index) -> bool {
        if (audio_vae_ == nullptr || source.data == nullptr || source.sample_count == 0 || source.channels == 0 || source.sample_rate == 0) return false;
        const int64_t prepare_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        const uint64_t max_source_samples = static_cast<uint64_t>(
            static_cast<long double>(frames) * source.sample_rate / 24.0L);
        const uint64_t source_samples = std::max<uint64_t>(1, std::min<uint64_t>(source.sample_count, max_source_samples));
        const uint64_t samples = std::max<uint64_t>(1, (source_samples * 32000ULL + source.sample_rate - 1) / source.sample_rate);
        sd::Tensor<float> waveform({static_cast<int64_t>(((samples + 799) / 800) * 800), 2, 1, 1});
        for (uint32_t channel = 0; channel < 2; ++channel) {
            const uint32_t source_channel = source.channels == 1 ? 0 : std::min<uint32_t>(channel, source.channels - 1);
            const auto resampled = h3_resample_audio_sinc(source, source_samples, source_channel, 32000);
            for (uint64_t sample = 0; sample < std::min<uint64_t>(samples, resampled.size()); ++sample) {
                waveform.index(sample, channel, 0, 0) = resampled[sample];
            }
        }
        if (profile_ptr != nullptr) profile.reference_audio_prepare_ms += ggml_time_ms() - prepare_begin;
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        auto encoded = audio_vae_->encode(runtime_->n_threads(), waveform);
        if (profile_ptr != nullptr) profile.reference_audio_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (encoded.empty()) return false;
        h3_trace_tensor("reference_audio_latent", encoded);
        *index = static_cast<int32_t>(reference_audio_latents.size());
        reference_audio_latents.push_back(std::move(encoded));
        return true;
    };
    for (int reference_index = 0; reference_index < params->ref_image_count; ++reference_index) {
        const ed_image_t& source_image = params->ref_images[reference_index];
        if (source_image.width <= 0 || source_image.height <= 0) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA image reference is invalid");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        int reference_width = 0;
        int reference_height = 0;
        h3_reference_image_dimensions(source_image,
                                      params->ref_image_size,
                                      params->width,
                                      params->height,
                                      &reference_width,
                                      &reference_height);
        sd::Tensor<float> image_tensor = h3_image_to_tensor(source_image, reference_width, reference_height);
        if (image_tensor.empty()) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA image reference is invalid");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        sd::Tensor<float> video_image = image_tensor.reshape({reference_width, reference_height, 1, 3, 1});
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        sd::Tensor<float> vae_latent = h3_encode_vae_condition(vae_.get(),
                                                               runtime_->n_threads(),
                                                               video_image,
                                                               h3_vae_tiling(*runtime_));
        if (profile_ptr != nullptr) profile.reference_video_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (vae_latent.empty()) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA image VAE encode failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        sd::Tensor<float> latent = vae_->vae_to_diffusion_latents(vae_latent);
        h3_apply_condition_noise(&latent, request_rng);
        h3_trace_tensor(("ref_image_" + std::to_string(reference_index) + "_latent").c_str(), latent);
        const int32_t encoded_image_index = static_cast<int32_t>(reference_latents.size());
        reference_latents.push_back(std::move(latent));
        reference_blocks.push_back({MiniMaxH3ReferenceKind::IMAGE, encoded_image_index, -1});
    }
    for (int video_index = 0; video_index < params->ref_video_count; ++video_index) {
        const ed_ref_video_t& reference = params->ref_videos[video_index];
        if (reference.frames == nullptr || reference.frame_count <= 0) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video reference is invalid");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        const int source_fps = reference.fps > 0 ? reference.fps : 24;
        int normalized_frames = static_cast<int>(std::lround(static_cast<double>(reference.frame_count) * 24.0 / source_fps));
        normalized_frames = std::min(normalized_frames, frames);
        if (normalized_frames < 5) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video reference needs at least 5 frames at 24 fps");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        while (normalized_frames % 17 != 5) --normalized_frames;
        int reference_width = 0;
        int reference_height = 0;
        h3_reference_video_dimensions(reference.frames[0], &reference_width, &reference_height);
        if (h3_trace_enabled()) {
            LOG_INFO("MiniMax-H3 Ref2VA video %d VAE geometry: source=%dx%d normalized=%dx%d frames=%d",
                     video_index + 1,
                     reference.frames[0].width,
                     reference.frames[0].height,
                     reference_width,
                     reference_height,
                     normalized_frames);
        }
        sd::Tensor<float> video_reference({reference_width, reference_height, normalized_frames, 3, 1});
        for (int frame = 0; frame < normalized_frames; ++frame) {
            const int source_index = std::min(reference.frame_count - 1, static_cast<int>(std::floor(frame * source_fps / 24.0)));
            auto source = h3_image_to_tensor(reference.frames[source_index], reference_width, reference_height);
            if (source.empty()) { set_minimax_error(error, "MiniMax-H3 Ref2VA video frame is invalid"); return ED_STATUS_INVALID_ARGUMENT; }
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < reference_height; ++y) {
                    for (int x = 0; x < reference_width; ++x) {
                        video_reference.index(x, y, frame, channel, 0) = source.index(x, y, channel, 0);
                    }
                }
            }
        }
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        auto vae_latent = h3_encode_vae_condition(vae_.get(),
                                                  runtime_->n_threads(),
                                                  video_reference,
                                                  h3_vae_tiling(*runtime_));
        if (profile_ptr != nullptr) profile.reference_video_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (vae_latent.empty()) { set_minimax_error(error, "MiniMax-H3 Ref2VA video VAE encode failed"); return ED_STATUS_GENERATION_FAILED; }
        auto latent = vae_->vae_to_diffusion_latents(vae_latent);
        h3_apply_condition_noise(&latent, request_rng);
        h3_trace_tensor(("ref_video_" + std::to_string(video_index) + "_latent").c_str(), latent);
        const int32_t encoded_video_index = static_cast<int32_t>(reference_latents.size());
        reference_latents.push_back(std::move(latent));
        video_reference_block_indices.push_back(reference_blocks.size());
        reference_blocks.push_back({MiniMaxH3ReferenceKind::VIDEO, encoded_video_index, -1});
    }
    release_video_vae.run_now();
    if (stage_audio_vae_lifecycle_ && need_audio_vae_for_conditioning &&
        audio_vae_ != nullptr && !audio_vae_->stage_params_for_phase()) {
        set_minimax_error(error, "MiniMax-H3 failed to stage audio VAE for conditioning");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    ScopeExit release_audio_vae([this, need_audio_vae_for_conditioning]() {
        if (need_audio_vae_for_conditioning && audio_vae_ != nullptr) {
            audio_vae_->release_params_after_phase();
        }
    });
    for (int video_index = 0; video_index < params->ref_video_count; ++video_index) {
        const ed_audio_t& reference_audio = params->ref_videos[video_index].audio;
        if (reference_audio.data == nullptr || reference_audio.sample_count == 0) {
            continue;
        }
        int32_t encoded_index = -1;
        if (!encode_reference_audio(reference_audio, &encoded_index)) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video audio encode failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        MiniMaxH3ReferenceBlock& block = reference_blocks[video_reference_block_indices[video_index]];
        block.kind = MiniMaxH3ReferenceKind::VIDEO_AUDIO;
        block.audio_index = encoded_index;
    }
    for (int audio_index = 0; audio_index < params->ref_audio_count; ++audio_index) {
        int32_t encoded_index = -1;
        if (!encode_reference_audio(params->ref_audios[audio_index], &encoded_index)) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA reference audio is invalid or audio VAE encoding failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        reference_blocks.push_back({MiniMaxH3ReferenceKind::AUDIO, -1, encoded_index});
    }
    release_audio_vae.run_now();
    if (h3_condition_debug_exit_enabled()) {
        if (profile_ptr != nullptr) {
            profile.total_ms = ggml_time_ms() - generation_begin;
            profile.log();
            LOG_INFO("minimax-h3 conditioning debug profile: context=%lld ms tokenize=%lld ms text-encode=%lld ms ref-visual-vae-encode=%lld ms ref-audio-vae-encode=%lld ms",
                     static_cast<long long>(profile.cond_context_ms),
                     static_cast<long long>(profile.text_tokenize_ms),
                     static_cast<long long>(profile.text_encode_ms),
                     static_cast<long long>(profile.reference_video_vae_encode_ms),
                     static_cast<long long>(profile.reference_audio_vae_encode_ms));
        }
        set_minimax_error(error, "MiniMax-H3 conditioning debug completed");
        return ED_STATUS_GENERATION_FAILED;
    }
    sd::Tensor<float> video = sd::zeros<float>({latent_width, latent_height, latent_frames, 24, 1});
    sd::Tensor<float> audio = sd::zeros<float>({audio_length, 2, 32, 1});
    const int64_t noise_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    video = sd::randn_like<float>(video, request_rng);
    audio = sd::randn_like<float>(audio, request_rng);
    sd::Tensor<float> packed = h3_pack_audio_and_video_latents(video, audio);
    if (profile_ptr != nullptr) profile.noise_init_ms = ggml_time_ms() - noise_begin;
    h3_trace_tensor("initial_packed_noise", packed);
    if (h3_trace_enabled()) {
        auto reference_kind_name = [](MiniMaxH3ReferenceKind kind) {
            switch (kind) {
                case MiniMaxH3ReferenceKind::IMAGE: return "image";
                case MiniMaxH3ReferenceKind::VIDEO: return "video";
                case MiniMaxH3ReferenceKind::VIDEO_AUDIO: return "video_audio";
                case MiniMaxH3ReferenceKind::AUDIO: return "audio";
            }
            return "unknown";
        };
        std::ostringstream blocks;
        for (size_t index = 0; index < reference_blocks.size(); ++index) {
            const auto& block = reference_blocks[index];
            if (index > 0) {
                blocks << ";";
            }
            blocks << index << ":" << reference_kind_name(block.kind)
                   << "(v=" << block.video_index
                   << ",a=" << block.audio_index << ")";
        }
        LOG_INFO("minimax-h3 trace reference summary: video_latents=%zu audio_latents=%zu blocks=%zu [%s]",
                 reference_latents.size(),
                 reference_audio_latents.size(),
                 reference_blocks.size(),
                 blocks.str().c_str());
    }
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    const float video_sigma_shift = params->sample.flow_shift > 0.0f ? params->sample.flow_shift : 12.0f;
    const ed_sampler_t sampler = params->sample.sampler == ED_SAMPLER_AUTO
                                     ? default_sample_method()
                                     : params->sample.sampler;
    const ed_scheduler_t scheduler = params->sample.scheduler == ED_SCHEDULER_AUTO
                                         ? default_scheduler(sampler)
                                         : params->sample.scheduler;
    if (sampler != ED_SAMPLER_EULER && sampler != ED_SAMPLER_RES_MULTISTEP) {
        set_minimax_error(error, "MiniMax-H3 currently supports Euler and RES multistep samplers");
        return ED_STATUS_UNSUPPORTED;
    }
    if (scheduler != ED_SCHEDULER_DISCRETE && scheduler != ED_SCHEDULER_SIMPLE) {
        set_minimax_error(error, "MiniMax-H3 currently supports discrete and simple schedulers");
        return ED_STATUS_UNSUPPORTED;
    }
    auto sigma_at = [&](int step) {
        return scheduler == ED_SCHEDULER_SIMPLE
                   ? h3_simple_flow_sigma(step, steps, video_sigma_shift)
                   : h3_discrete_flow_sigma(step, steps, video_sigma_shift);
    };
    LOG_INFO("MiniMax-H3 sampling: sampler=%s scheduler=%s steps=%d flow-shift=%.6g",
             sampler == ED_SAMPLER_RES_MULTISTEP ? "res_multistep" : "euler",
             scheduler == ED_SCHEDULER_SIMPLE ? "simple" : "discrete",
             steps,
             video_sigma_shift);
    if (stage_diffusion_lifecycle_ && !diffusion_->stage_params_for_phase()) {
        set_minimax_error(error, "MiniMax-H3 failed to stage diffusion model for denoising");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    ScopeExit release_diffusion([this]() {
        if (diffusion_ != nullptr) {
            diffusion_->release_params_after_phase();
        }
    });
    sd::Tensor<float> old_denoised;
    float old_sigma_down = 0.0f;
    bool have_old_denoised = false;
    for (int step = 0; step < steps; ++step) {
        if (profile_ptr != nullptr) ++profile.diffusion_steps;
        const float sigma = sigma_at(step);
        const float sigma_next = sigma_at(step + 1);
        sd::Tensor<float> model_packed = packed;
        const float audio_sigma = MiniMaxH3::time_shift_sigma(sigma, video_sigma_shift, 3.0f);
        const float audio_sigma_next = MiniMaxH3::time_shift_sigma(sigma_next, video_sigma_shift, 3.0f);
        const float audio_slope = MiniMaxH3::time_shift_slope(sigma, video_sigma_shift, 3.0f);
        float audio_scale = 1.0f;
        if (sampler == ED_SAMPLER_RES_MULTISTEP) {
            audio_scale = video_sigma_shift / 3.0f;
            auto model_av = diffusion_->split_av_latents(packed, audio_length);
            model_av.second *= audio_sigma / sigma;
            model_packed = h3_pack_audio_and_video_latents(model_av.first, model_av.second);
        }
        sd::Tensor<float> timestep({1}, {sigma * 1000.0f});
        DiffusionParams diffusion_params{};
        diffusion_params.x = &model_packed;
        diffusion_params.timesteps = &timestep;
        diffusion_params.context = &context;
        diffusion_params.minimax_text_token_tags = &token_tags;
        diffusion_params.ref_latents = !reference_latents.empty() ? &reference_latents
                                         : (keyframe_latents.empty() ? nullptr : &keyframe_latents);
        diffusion_params.minimax_reference_blocks = reference_blocks.empty() ? nullptr : &reference_blocks;
        diffusion_params.minimax_reference_audio_latents = reference_audio_latents.empty() ? nullptr : &reference_audio_latents;
        sd::Tensor<int32_t> keyframe_index_tensor;
        if (!keyframe_indices.empty()) {
            keyframe_index_tensor = sd::Tensor<int32_t>({static_cast<int64_t>(keyframe_indices.size())}, keyframe_indices);
            diffusion_params.minimax_keyframe_indices = &keyframe_index_tensor;
        }
        diffusion_params.minimax_audio_length = audio_length;
        diffusion_params.minimax_video_sigma_shift = video_sigma_shift;
        diffusion_params.minimax_audio_sigma_shift = 3.0f;
        const int64_t cond_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        ScopedEnvVar current_step_env("ED_MINIMAX_H3_CURRENT_STEP", std::to_string(step));
        sd::Tensor<float> velocity = diffusion_->compute(runtime_->n_threads(), diffusion_params);
        if (profile_ptr != nullptr) {
            profile.diffusion_cond_ms += ggml_time_ms() - cond_begin;
            ++profile.diffusion_calls;
        }
        if (velocity.empty()) {
            set_minimax_error(error, "MiniMax-H3 diffusion compute failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        if (!h3_dit_debug_target().empty()) {
            const std::string debug_name = "dit_" + h3_dit_debug_target();
            h3_trace_tensor(debug_name.c_str(), velocity);
            h3_dump_debug_tensor_if_requested(debug_name, velocity);
            set_minimax_error(error, "MiniMax-H3 DiT debug target completed");
            return ED_STATUS_GENERATION_FAILED;
        }
        if (use_cfg) {
            diffusion_params.context = &uncond_context;
            diffusion_params.minimax_text_token_tags = &uncond_token_tags;
            const int64_t uncond_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
            sd::Tensor<float> uncond_velocity = diffusion_->compute(runtime_->n_threads(), diffusion_params);
            if (profile_ptr != nullptr) {
                profile.diffusion_uncond_ms += ggml_time_ms() - uncond_begin;
                ++profile.diffusion_calls;
            }
            if (uncond_velocity.empty()) {
                set_minimax_error(error, "MiniMax-H3 unconditional diffusion compute failed");
                return ED_STATUS_GENERATION_FAILED;
            }
            const int64_t cfg_combine_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
            velocity = uncond_velocity + (velocity - uncond_velocity) * cfg_scale;
            if (profile_ptr != nullptr) profile.cfg_combine_ms += ggml_time_ms() - cfg_combine_begin;
        }
        if (sampler == ED_SAMPLER_RES_MULTISTEP) {
            auto model_av = diffusion_->split_av_latents(model_packed, audio_length);
            auto velocity_av = diffusion_->split_av_latents(velocity, audio_length);
            velocity_av.second = (1.0f - audio_scale) * model_av.second +
                                 (1.0f + (audio_scale - 1.0f) * audio_sigma) *
                                     (velocity_av.second / audio_slope);
            velocity = h3_pack_audio_and_video_latents(velocity_av.first, velocity_av.second);
        }
        if (h3_trace_enabled()) {
            LOG_INFO("minimax-h3 trace step=%d sigma=%.8g sigma_next=%.8g", step, sigma, sigma_next);
            h3_trace_tensor(("step_" + std::to_string(step) + "_velocity").c_str(), velocity);
        }
        if (sampler == ED_SAMPLER_RES_MULTISTEP) {
            sd::Tensor<float> denoised = packed - velocity * sigma;
            if (sigma_next == 0.0f || !have_old_denoised) {
                packed += velocity * (sigma_next - sigma);
            } else {
                const float t = -std::log(sigma);
                const float t_old = -std::log(old_sigma_down);
                const float t_next = -std::log(sigma_next);
                const float t_prev = -std::log(sigma_at(step - 1));
                const float h = t_next - t;
                const float c2 = (t_prev - t_old) / h;
                const float phi1 = std::expm1(-h) / -h;
                const float phi2 = (phi1 - 1.0f) / -h;
                float b1 = phi1 - phi2 / c2;
                float b2 = phi2 / c2;
                if (!std::isfinite(b1)) b1 = 0.0f;
                if (!std::isfinite(b2)) b2 = 0.0f;
                packed = std::exp(-h) * packed + h * (b1 * denoised + b2 * old_denoised);
            }
            old_denoised = std::move(denoised);
            old_sigma_down = sigma_next;
            have_old_denoised = true;
        } else {
            auto packed_av = diffusion_->split_av_latents(packed, audio_length);
            auto velocity_av = diffusion_->split_av_latents(velocity, audio_length);
            packed_av.first += velocity_av.first * (sigma_next - sigma);
            packed_av.second += (velocity_av.second / audio_slope) * (audio_sigma_next - audio_sigma);
            packed = h3_pack_audio_and_video_latents(packed_av.first, packed_av.second);
        }
        if (h3_trace_enabled()) {
            h3_trace_tensor(("step_" + std::to_string(step) + "_packed").c_str(), packed);
        }
    }
    release_diffusion.run_now();
    auto av = diffusion_->split_av_latents(packed, audio_length);
    if (sampler == ED_SAMPLER_RES_MULTISTEP) {
        av.second /= video_sigma_shift / 3.0f;
    }
    ed_status_t status = decode_video_latent(av.first, frames, out, profile_ptr, error);
    if (status != ED_STATUS_OK) {
        return status;
    }
    if (audio_vae_ != nullptr && !decode_audio_latent(av.second, out, profile_ptr, error)) {
        ed_free_video(out);
        return ED_STATUS_GENERATION_FAILED;
    }
    if (profile_ptr != nullptr) {
        profile.total_ms = ggml_time_ms() - generation_begin;
        profile.log();
    }
    return ED_STATUS_OK;
}

ed_scheduler_t MiniMaxH3Pipeline::default_scheduler(ed_sampler_t) const {
    return ED_SCHEDULER_DISCRETE;
}

}  // namespace edgedit
