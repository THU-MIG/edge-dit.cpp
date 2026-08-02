#include "dit_models/pipelines/qwen_image_edit_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/optimization/cache/runtime/cache_engine.hpp"
#include "dit_models/components/autoencoders/vae.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "dit_models/models/qwen_image.hpp"
#include "dit_models/models/wan.hpp"
#include "dit_models/pipelines/dit_pipeline_utils.hpp"
#include "parallel/cfg_parallel.hpp"
#include "parallel/process_group.hpp"
#include "ggml.h"
#include "utils/util.h"

namespace {

static constexpr int ED_QWEN_EDIT_IMAGE_ALIGN = 32;
static constexpr float ED_QWEN_EDIT_SHIFT_TERMINAL = 0.02f;

static bool qwen_edit_debug_align_enabled() {
    const char* env = std::getenv("ED_DEBUG_QWEN_ALIGN");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool qwen_edit_debug_only_enabled() {
    const char* env = std::getenv("ED_QWEN_ALIGN_DEBUG_ONLY");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool qwen_edit_csv_contains(const char* csv, const std::string& name) {
    if (csv == nullptr || csv[0] == '\0') {
        return false;
    }
    const char* begin = csv;
    while (*begin != '\0') {
        while (*begin == ' ' || *begin == '\t' || *begin == ',') {
            ++begin;
        }
        const char* end = begin;
        while (*end != '\0' && *end != ',') {
            ++end;
        }
        const char* trimmed_end = end;
        while (trimmed_end > begin && (trimmed_end[-1] == ' ' || trimmed_end[-1] == '\t')) {
            --trimmed_end;
        }
        if (name.size() == static_cast<size_t>(trimmed_end - begin) &&
            std::equal(name.begin(), name.end(), begin)) {
            return true;
        }
        begin = end;
    }
    return false;
}

static std::string qwen_edit_dump_safe_name(const std::string& name) {
    std::string safe = name;
    for (char& ch : safe) {
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') ||
                        ch == '.' ||
                        ch == '_' ||
                        ch == '-';
        if (!ok) {
            ch = '_';
        }
    }
    return safe;
}

static void qwen_edit_dump_tensor_if_requested(const char* name, const sd::Tensor<float>& tensor) {
    const char* dump_dir = std::getenv("ED_QWEN_ALIGN_DUMP_DIR");
    const char* targets  = std::getenv("ED_QWEN_ALIGN_DUMP_TARGETS");
    if (dump_dir == nullptr || dump_dir[0] == '\0' || !qwen_edit_csv_contains(targets, name)) {
        return;
    }
    const std::string base_path = std::string(dump_dir) + "/" + qwen_edit_dump_safe_name(name);
    {
        std::ofstream out(base_path + ".f32.bin", std::ios::binary);
        if (out) {
            out.write(reinterpret_cast<const char*>(tensor.values().data()),
                      static_cast<std::streamsize>(tensor.values().size() * sizeof(float)));
        }
    }
    {
        std::ofstream shape_out(base_path + ".shape");
        if (shape_out) {
            for (size_t i = 0; i < tensor.shape().size(); ++i) {
                if (i > 0) {
                    shape_out << " ";
                }
                shape_out << tensor.shape()[i];
            }
            shape_out << "\n";
        }
    }
}

struct QwenEditDimensions {
    int width;
    int height;
};

static QwenEditDimensions qwen_edit_calculate_dimensions(double target_area, double ratio) {
    if (!(target_area > 0.0) || !(ratio > 0.0) || !std::isfinite(ratio)) {
        ratio = 1.0;
    }

    const double width = std::sqrt(target_area * ratio);
    const double height = width / ratio;
    return {
        std::max(ED_QWEN_EDIT_IMAGE_ALIGN,
                 static_cast<int>(std::round(width / ED_QWEN_EDIT_IMAGE_ALIGN) * ED_QWEN_EDIT_IMAGE_ALIGN)),
        std::max(ED_QWEN_EDIT_IMAGE_ALIGN,
                 static_cast<int>(std::round(height / ED_QWEN_EDIT_IMAGE_ALIGN) * ED_QWEN_EDIT_IMAGE_ALIGN)),
    };
}

static void qwen_edit_round_tensor_to_bf16(sd::Tensor<float>& tensor) {
    for (float& value : tensor.values()) {
        value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
    }
}

static double qwen_edit_sinc(double x) {
    if (x == 0.0) {
        return 1.0;
    }
    x *= 3.14159265358979323846;
    return std::sin(x) / x;
}

static double qwen_edit_lanczos_filter(double x) {
    if (-3.0 <= x && x < 3.0) {
        return qwen_edit_sinc(x) * qwen_edit_sinc(x / 3.0);
    }
    return 0.0;
}

struct QwenEditResizeAxisContrib {
    int start = 0;
    std::vector<int32_t> coeffs;
};

static std::vector<QwenEditResizeAxisContrib> qwen_edit_precompute_pillow_lanczos_coeffs(int in_size,
                                                                                         int out_size) {
    static constexpr int precision_bits = 22;
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = std::max(1.0, scale);
    const double support = 3.0 * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    const double ss = 1.0 / filterscale;

    std::vector<QwenEditResizeAxisContrib> result(static_cast<size_t>(out_size));
    for (int out = 0; out < out_size; ++out) {
        const double center = (static_cast<double>(out) + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        xmin = std::max(0, xmin);
        int xmax = static_cast<int>(center + support + 0.5);
        xmax = std::min(in_size, xmax);
        const int count = std::max(0, xmax - xmin);

        std::vector<double> weights(static_cast<size_t>(ksize), 0.0);
        double weight_sum = 0.0;
        for (int i = 0; i < count; ++i) {
            const double weight = qwen_edit_lanczos_filter((static_cast<double>(i + xmin) - center + 0.5) * ss);
            weights[static_cast<size_t>(i)] = weight;
            weight_sum += weight;
        }
        if (weight_sum != 0.0) {
            for (int i = 0; i < count; ++i) {
                weights[static_cast<size_t>(i)] /= weight_sum;
            }
        }

        auto& axis = result[static_cast<size_t>(out)];
        axis.start = xmin;
        axis.coeffs.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const double scaled = weights[static_cast<size_t>(i)] * static_cast<double>(1 << precision_bits);
            axis.coeffs[static_cast<size_t>(i)] = static_cast<int32_t>((scaled < 0.0 ? -0.5 : 0.5) + scaled);
        }
    }
    return result;
}

static uint8_t qwen_edit_pillow_clip8(int64_t value) {
    static constexpr int precision_bits = 22;
    const int64_t shifted = value >> precision_bits;
    if (shifted <= 0) {
        return 0;
    }
    if (shifted >= 255) {
        return 255;
    }
    return static_cast<uint8_t>(shifted);
}

static void qwen_edit_log_tensor_stats(const char* name, const sd::Tensor<float>& tensor) {
    if (!qwen_edit_debug_align_enabled()) {
        return;
    }
    if (tensor.empty()) {
        LOG_INFO("qwen-align %s: empty", name);
        return;
    }
    qwen_edit_dump_tensor_if_requested(name, tensor);

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

    const auto& values = tensor.values();
    if (values.size() >= 4) {
        auto flat_at = [&](size_t index) -> float {
            index = std::min(index, values.size() - 1);
            return values[index];
        };
        const size_t last = values.size() >= 4 ? values.size() - 4 : 0;
        LOG_INFO("qwen-align %s flat_samples: i0=[%.9g %.9g %.9g %.9g] i64=[%.9g %.9g %.9g %.9g] ilast=[%.9g %.9g %.9g %.9g]",
                 name,
                 static_cast<double>(flat_at(0)),
                 static_cast<double>(flat_at(1)),
                 static_cast<double>(flat_at(2)),
                 static_cast<double>(flat_at(3)),
                 static_cast<double>(flat_at(64)),
                 static_cast<double>(flat_at(65)),
                 static_cast<double>(flat_at(66)),
                 static_cast<double>(flat_at(67)),
                 static_cast<double>(flat_at(last + 0)),
                 static_cast<double>(flat_at(last + 1)),
                 static_cast<double>(flat_at(last + 2)),
                 static_cast<double>(flat_at(last + 3)));
    }

    const auto& shape = tensor.shape();
    if (shape.size() >= 2 && shape[0] > 0 && shape[1] > 0) {
        const int64_t dim = shape[0];
        const int64_t seq = shape[1];
        auto at = [&](int64_t token, int64_t channel) -> float {
            token = std::max<int64_t>(0, std::min<int64_t>(seq - 1, token));
            channel = std::max<int64_t>(0, std::min<int64_t>(dim - 1, channel));
            return tensor.values()[static_cast<size_t>(channel + dim * token)];
        };
        LOG_INFO("qwen-align %s samples: t0=[%.9g %.9g %.9g %.9g] t1=[%.9g %.9g %.9g %.9g] t64=[%.9g %.9g %.9g %.9g] tlast=[%.9g %.9g %.9g %.9g]",
                 name,
                 static_cast<double>(at(0, 0)),
                 static_cast<double>(at(0, 1)),
                 static_cast<double>(at(0, 2)),
                 static_cast<double>(at(0, 3)),
                 static_cast<double>(at(1, 0)),
                 static_cast<double>(at(1, 1)),
                 static_cast<double>(at(1, 2)),
                 static_cast<double>(at(1, 3)),
                 static_cast<double>(at(64, 0)),
                 static_cast<double>(at(64, 1)),
                 static_cast<double>(at(64, 2)),
                 static_cast<double>(at(64, 3)),
                 static_cast<double>(at(seq - 1, 0)),
                 static_cast<double>(at(seq - 1, 1)),
                 static_cast<double>(at(seq - 1, 2)),
                 static_cast<double>(at(seq - 1, 3)));
    }
}

static sd::Tensor<float> qwen_edit_pack_latents_2x2(const sd::Tensor<float>& tensor) {
    if (!qwen_edit_debug_align_enabled() || tensor.empty()) {
        return {};
    }
    const auto& shape = tensor.shape();
    if (shape.size() != 4 || shape[3] <= 0 || shape[2] <= 0 ||
        shape[0] <= 0 || shape[1] <= 0 ||
        (shape[0] % 2) != 0 || (shape[1] % 2) != 0) {
        return {};
    }

    const int64_t width = shape[0];
    const int64_t height = shape[1];
    const int64_t channels = shape[2];
    const int64_t batch = shape[3];
    const int64_t packed_channels = channels * 4;
    const int64_t packed_tokens = (width / 2) * (height / 2);
    sd::Tensor<float> packed({packed_channels, packed_tokens, batch});

    const float* src = tensor.data();
    float* dst = packed.data();
    auto src_at = [&](int64_t x, int64_t y, int64_t c, int64_t n) -> float {
        return src[x + width * y + width * height * c + width * height * channels * n];
    };
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t py = 0; py < height / 2; ++py) {
            for (int64_t px = 0; px < width / 2; ++px) {
                const int64_t token = py * (width / 2) + px;
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t dy = 0; dy < 2; ++dy) {
                        for (int64_t dx = 0; dx < 2; ++dx) {
                            const int64_t packed_c = c * 4 + dy * 2 + dx;
                            dst[packed_c + packed_channels * token + packed_channels * packed_tokens * n] =
                                src_at(px * 2 + dx, py * 2 + dy, c, n);
                        }
                    }
                }
            }
        }
    }
    return packed;
}

static void qwen_edit_log_packed_latent_stats(const char* name, const sd::Tensor<float>& tensor) {
    if (!qwen_edit_debug_align_enabled()) {
        return;
    }
    sd::Tensor<float> packed = qwen_edit_pack_latents_2x2(tensor);
    if (!packed.empty()) {
        qwen_edit_log_tensor_stats(name, packed);
    }
}

static void qwen_edit_log_transformer_debug_targets(Qwen::QwenImageRunner* diffusion,
                                                    int n_threads,
                                                    const sd::Tensor<float>& x,
                                                    const sd::Tensor<float>& timesteps,
                                                    const sd::Tensor<float>& context,
                                                    const std::vector<sd::Tensor<float>>& ref_latents) {
    if (!qwen_edit_debug_align_enabled() || diffusion == nullptr) {
        return;
    }
    static const std::vector<const char*> targets = {
        "pe",
        "t_emb",
        "img_in",
        "txt_norm",
        "txt_in",
        "block0.img_mod_params",
        "block0.txt_t_emb",
        "block0.txt_mod_params",
        "block0.img_gate1",
        "block0.txt_gate1",
        "block0.img_normed",
        "block0.img_modulated",
        "block0.txt_normed",
        "block0.txt_modulated",
        "block0.attn.img_q",
        "block0.attn.img_k",
        "block0.attn.img_v",
        "block0.attn.txt_q",
        "block0.attn.txt_k",
        "block0.attn.txt_v",
        "block0.attn.joint_q_rope_seq",
        "block0.attn.joint_k_rope_seq",
        "block0.attn.joint_v_seq",
        "block0.attn.img_preproj",
        "block0.attn.txt_preproj",
        "block0.attn.img_out",
        "block0.attn.txt_out",
        "block0.img_after_attn",
        "block0.txt_after_attn",
        "block0.img_normed2",
        "block0.img_modulated2",
        "block0.img_gate2",
        "block0.img_mlp_out",
        "block0.img_after_mlp",
        "block0_img",
        "block0_txt",
        "norm_out",
        "proj_out",
    };
    const char* target_filter = std::getenv("ED_QWEN_ALIGN_DEBUG_TARGETS");
    for (const char* target : targets) {
        if (target_filter != nullptr &&
            target_filter[0] != '\0' &&
            !qwen_edit_csv_contains(target_filter, target)) {
            continue;
        }
        sd::Tensor<float> tensor = diffusion->compute_debug_target(n_threads,
                                                                   x,
                                                                   timesteps,
                                                                   context,
                                                                   ref_latents,
                                                                   true,
                                                                   target);
        const std::string name = std::string("transformer.cond.") + target;
        qwen_edit_log_tensor_stats(name.c_str(), tensor);
    }
}

static float qwen_edit_time_snr_shift(float shift, float t) {
    return shift * t / (1.0f + (shift - 1.0f) * t);
}

static float qwen_edit_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return qwen_edit_time_snr_shift(shift, t / 1000.0f);
}

static std::vector<float> qwen_edit_discrete_sigmas(int steps, float shift) {
    std::vector<float> result;
    if (steps <= 0) {
        return result;
    }
    if (steps == 1) {
        result.push_back(qwen_edit_t_to_sigma(999.0f, shift));
        result.push_back(0.0f);
        return result;
    }

    result.reserve(static_cast<size_t>(steps) + 1);
    const float step = 999.0f / static_cast<float>(steps - 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        result.push_back(qwen_edit_t_to_sigma(t, shift));
    }

    result.push_back(0.0f);
    return result;
}

static float qwen_edit_time_shift_exponential(float mu, float t) {
    return std::exp(mu) / (std::exp(mu) + std::pow(1.0f / t - 1.0f, 1.0f));
}

static void qwen_edit_stretch_shift_to_terminal(std::vector<float>& sigmas, float terminal) {
    if (sigmas.empty() || !(terminal > 0.0f) || terminal >= 1.0f) {
        return;
    }
    const float one_minus_last = 1.0f - sigmas.back();
    if (one_minus_last == 0.0f) {
        return;
    }
    const float scale = one_minus_last / (1.0f - terminal);
    for (float& sigma : sigmas) {
        sigma = 1.0f - ((1.0f - sigma) / scale);
    }
}

static std::vector<float> qwen_edit_diffusers_sigmas(int steps, int64_t image_seq_len, float* out_mu = nullptr) {
    std::vector<float> result;
    if (steps <= 0) {
        return result;
    }

    const float mu = edgedit::calculate_shift(image_seq_len,
                                              /*base_seq_len=*/256,
                                              /*max_seq_len=*/8192,
                                              /*base_shift=*/0.5f,
                                              /*max_shift=*/0.9f);
    if (out_mu != nullptr) {
        *out_mu = mu;
    }

    result.reserve(static_cast<size_t>(steps) + 1);
    if (steps == 1) {
        result.push_back(1.0f);
    } else {
        const float end = 1.0f / static_cast<float>(steps);
        for (int i = 0; i < steps; ++i) {
            const float r = static_cast<float>(i) / static_cast<float>(steps - 1);
            const float sigma = 1.0f + (end - 1.0f) * r;
            result.push_back(qwen_edit_time_shift_exponential(mu, sigma));
        }
    }

    qwen_edit_stretch_shift_to_terminal(result, ED_QWEN_EDIT_SHIFT_TERMINAL);
    result.push_back(0.0f);
    return result;
}

static ed_status_t qwen_edit_tensor_to_image(const sd::Tensor<float>& tensor, ed_image_t* image) {
    if (image == nullptr || tensor.empty()) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const auto& shape = tensor.shape();
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
        return ED_STATUS_INVALID_ARGUMENT;
    }

    const size_t width = static_cast<size_t>(shape[0]);
    const size_t height = static_cast<size_t>(shape[1]);
    const size_t channels = static_cast<size_t>(shape[2]);
    const size_t nbytes = width * height * channels;
    uint8_t* data = static_cast<uint8_t*>(std::malloc(nbytes));
    if (data == nullptr) {
        return ED_STATUS_OUT_OF_MEMORY;
    }

    auto to_u8 = [](float value) -> uint8_t {
        if (value <= 0.0f) {
            return 0;
        }
        if (value >= 1.0f) {
            return 255;
        }
        return static_cast<uint8_t>(value * 255.0f + 0.5f);
    };

    const size_t pixels = width * height;
    const float* src = tensor.data();
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            data[i * channels + c] = to_u8(src[i + pixels * c]);
        }
    }
    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data = data;
    return ED_STATUS_OK;
}

static sd::Tensor<float> resize_image_to_tensor(const ed_image_t& image, int width, int height) {
    sd::Tensor<float> tensor({width, height, 3, 1});
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channels == 0) {
        return {};
    }

    const uint32_t src_channels = image.channels;
    for (int y = 0; y < height; ++y) {
        const int src_y = std::min<int>(static_cast<int>(image.height) - 1,
                                        static_cast<int>((static_cast<int64_t>(y) * image.height) / height));
        for (int x = 0; x < width; ++x) {
            const int src_x = std::min<int>(static_cast<int>(image.width) - 1,
                                            static_cast<int>((static_cast<int64_t>(x) * image.width) / width));
            const uint8_t* pixel = image.data + (static_cast<size_t>(src_y) * image.width + static_cast<size_t>(src_x)) * src_channels;
            for (int c = 0; c < 3; ++c) {
                const uint8_t value = c < static_cast<int>(src_channels) ? pixel[c] : pixel[0];
                tensor[static_cast<int64_t>(x) +
                       static_cast<int64_t>(width) * static_cast<int64_t>(y) +
                       static_cast<int64_t>(width) * static_cast<int64_t>(height) * static_cast<int64_t>(c)] =
                    static_cast<float>(value) / 255.0f;
            }
        }
    }
    return tensor;
}

static sd::Tensor<float> resize_image_to_tensor_lanczos(const ed_image_t& image, int width, int height) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channels == 0 ||
        width <= 0 || height <= 0) {
        return {};
    }

    const int src_width = static_cast<int>(image.width);
    const int src_height = static_cast<int>(image.height);
    const int src_channels = static_cast<int>(image.channels);
    if (src_width == width && src_height == height) {
        return resize_image_to_tensor(image, width, height);
    }

    const auto x_coeffs = qwen_edit_precompute_pillow_lanczos_coeffs(src_width, width);
    const auto y_coeffs = qwen_edit_precompute_pillow_lanczos_coeffs(src_height, height);
    std::vector<uint8_t> tmp(static_cast<size_t>(src_height) * static_cast<size_t>(width) * 3);

    for (int y = 0; y < src_height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto& coeff = x_coeffs[static_cast<size_t>(x)];
            int64_t accum[3] = {
                1LL << 21,
                1LL << 21,
                1LL << 21,
            };
            for (size_t k = 0; k < coeff.coeffs.size(); ++k) {
                const int src_x = coeff.start + static_cast<int>(k);
                const uint8_t* pixel = image.data +
                                       (static_cast<size_t>(y) * static_cast<size_t>(src_width) +
                                        static_cast<size_t>(src_x)) *
                                           static_cast<size_t>(src_channels);
                for (int c = 0; c < 3; ++c) {
                    const uint8_t value = c < src_channels ? pixel[c] : pixel[0];
                    accum[c] += static_cast<int64_t>(value) * static_cast<int64_t>(coeff.coeffs[k]);
                }
            }

            uint8_t* out = tmp.data() +
                           (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
            out[0] = qwen_edit_pillow_clip8(accum[0]);
            out[1] = qwen_edit_pillow_clip8(accum[1]);
            out[2] = qwen_edit_pillow_clip8(accum[2]);
        }
    }

    sd::Tensor<float> tensor({width, height, 3, 1});
    float* dst = tensor.data();
    for (int y = 0; y < height; ++y) {
        const auto& coeff = y_coeffs[static_cast<size_t>(y)];
        for (int x = 0; x < width; ++x) {
            int64_t accum[3] = {
                1LL << 21,
                1LL << 21,
                1LL << 21,
            };
            for (size_t k = 0; k < coeff.coeffs.size(); ++k) {
                const int src_y = coeff.start + static_cast<int>(k);
                const uint8_t* pixel = tmp.data() +
                                       (static_cast<size_t>(src_y) * static_cast<size_t>(width) +
                                        static_cast<size_t>(x)) *
                                           3;
                for (int c = 0; c < 3; ++c) {
                    accum[c] += static_cast<int64_t>(pixel[c]) * static_cast<int64_t>(coeff.coeffs[k]);
                }
            }

            for (int c = 0; c < 3; ++c) {
                dst[static_cast<int64_t>(x) +
                    static_cast<int64_t>(width) * static_cast<int64_t>(y) +
                    static_cast<int64_t>(width) * static_cast<int64_t>(height) * static_cast<int64_t>(c)] =
                    static_cast<float>(qwen_edit_pillow_clip8(accum[c])) / 255.0f;
            }
        }
    }
    return tensor;
}

static float tensor_l2_norm(const sd::Tensor<float>& tensor) {
    double sum = 0.0;
    for (float value : tensor.values()) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    return static_cast<float>(std::sqrt(std::max(sum, 0.0)));
}

static int64_t tensor_index_4d(int64_t x, int64_t y, int64_t c, int64_t n,
                               int64_t width, int64_t height, int64_t channels) {
    return x + width * y + width * height * c + width * height * channels * n;
}

static sd::Tensor<float> true_cfg_rescale(const sd::Tensor<float>& cond,
                                          const sd::Tensor<float>& uncond,
                                          float cfg_scale,
                                          int patch_size) {
    sd::Tensor<float> guided = uncond + cfg_scale * (cond - uncond);
    const auto& shape = guided.shape();
    if (shape.size() == 4 && shape[3] == 1 && patch_size > 0 &&
        shape[0] % patch_size == 0 && shape[1] % patch_size == 0 &&
        cond.shape() == shape && uncond.shape() == shape) {
        const int64_t width    = shape[0];
        const int64_t height   = shape[1];
        const int64_t channels = shape[2];
        float* guided_data     = guided.data();
        const float* cond_data = cond.data();
        for (int64_t py = 0; py < height; py += patch_size) {
            for (int64_t px = 0; px < width; px += patch_size) {
                double cond_sum   = 0.0;
                double guided_sum = 0.0;
                for (int64_t dy = 0; dy < patch_size; ++dy) {
                    for (int64_t dx = 0; dx < patch_size; ++dx) {
                        for (int64_t c = 0; c < channels; ++c) {
                            const int64_t idx = tensor_index_4d(px + dx, py + dy, c, 0, width, height, channels);
                            cond_sum += static_cast<double>(cond_data[idx]) * cond_data[idx];
                            guided_sum += static_cast<double>(guided_data[idx]) * guided_data[idx];
                        }
                    }
                }
                const float cond_norm   = static_cast<float>(std::sqrt(std::max(cond_sum, 0.0)));
                const float guided_norm = static_cast<float>(std::sqrt(std::max(guided_sum, 0.0)));
                if (cond_norm > 0.0f && guided_norm > 0.0f &&
                    std::isfinite(cond_norm) && std::isfinite(guided_norm)) {
                    const float scale = cond_norm / guided_norm;
                    for (int64_t dy = 0; dy < patch_size; ++dy) {
                        for (int64_t dx = 0; dx < patch_size; ++dx) {
                            for (int64_t c = 0; c < channels; ++c) {
                                guided_data[tensor_index_4d(px + dx, py + dy, c, 0, width, height, channels)] *= scale;
                            }
                        }
                    }
                }
            }
        }
        return guided;
    }

    const float cond_norm = tensor_l2_norm(cond);
    const float guided_norm = tensor_l2_norm(guided);
    if (cond_norm > 0.0f && guided_norm > 0.0f && std::isfinite(cond_norm) && std::isfinite(guided_norm)) {
        guided *= (cond_norm / guided_norm);
    }
    return guided;
}

}  // namespace

namespace edgedit {

QwenImageEditPipeline::QwenImageEditPipeline(SDVersion version)
    : version_(version) {
}

QwenImageEditPipeline::~QwenImageEditPipeline() {
    reset();
}

void QwenImageEditPipeline::reset() {
    conditioner_.reset();
    vae_.reset();
    diffusion_.reset();
    runtime_weights_loaded_ = false;
}

int QwenImageEditPipeline::resolve_steps(int requested_steps) const {
    if (requested_steps > 0) {
        return requested_steps;
    }
    // Auto (--steps not set): few-step distilled checkpoints (Qwen-Image-Edit-
    // Lightning etc.) default to their distilled step count; base defaults to 50.
    return distilled_default_steps_ > 0 ? distilled_default_steps_ : 50;
}

bool QwenImageEditPipeline::prepare(const ed_context_params_t& params,
                                    ModelRuntime& runtime,
                                    const ModelLoader& loader,
                                    PipelineTensorRegistry& registry,
                                    std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    reset();
    distilled_default_steps_ = detect_distilled_default_steps(loader.file_paths(),
                                                              params.diffusion_model_path);

    if (!ed_version_is_qwen_image_edit(version_)) {
        if (error != nullptr) {
            *error = "QwenImageEditPipeline got non-Qwen-Image-Edit model version";
        }
        return false;
    }
    if (!build_components(params, loader, error)) {
        return false;
    }
    const auto diffusion_wtypes = loader.get_diffusion_model_wtype_stat();
    const auto bf16_it = diffusion_wtypes.find(GGML_TYPE_BF16);
    diffusion_bf16_ = bf16_it != diffusion_wtypes.end() &&
                      bf16_it->second > 0 &&
                      diffusion_wtypes.size() == 1;
    if (!register_tensors(registry, error)) {
        return false;
    }
    return true;
}

bool QwenImageEditPipeline::build_components(const ed_context_params_t& params,
                                             const ModelLoader& loader,
                                             std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        if (error != nullptr) {
            *error = "QwenImageEditPipeline requires initialized ModelRuntime backends";
        }
        return false;
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool qwen_zero_cond_t = params.qwen_image_zero_cond_t || loader.qwen_image_zero_cond_t();

    // Auto-allocate: seed tally with hard-cap budget min(--max-vram, free); finalize after.
    runtime_->reset_auto_allocate_state();

    // Measure the DiT compute-buffer at the target resolution for resident-headroom
    // sizing (vs the fixed 4 GiB constant). Temporary QwenImageRunner spec (shapes only).
    // No-op unless auto_allocate + target size; 0 -> caller falls back to fixed headroom.
    runtime_->set_measured_dit_headroom(0);
    if (runtime_->auto_fit() && runtime_->fit_width() > 0 && runtime_->fit_height() > 0) {
        auto measure_runner = std::make_unique<Qwen::QwenImageRunner>(
            runtime_->backend(), false, storage, "model.diffusion_model", version_,
            qwen_zero_cond_t);
        measure_runner->set_flash_attention_enabled(runtime_->flash_attention());
        const int latent_w = runtime_->fit_width() / 8;
        const int latent_h = runtime_->fit_height() / 8;
        const size_t measured = measure_runner->measure_compute_buffer_at(latent_w, latent_h);
        if (measured > 0) {
            runtime_->set_measured_dit_headroom(measured);
        }
        LOG_INFO("auto-allocate: measured DiT compute buffer = %.2f GB at latent %dx%d (fixed fallback = 4.00 GB)",
                 measured / (1024.0 * 1024.0 * 1024.0), latent_w, latent_h);
    }

    const size_t eff_budget = runtime_->effective_budget_bytes();
    size_t remaining_free = eff_budget;
    const bool diffusion_offload = runtime_->dit_offload_params_to_cpu() ||
                                   runtime_->plan_component_offload(loader, "model.diffusion_model", remaining_free);
    const bool te_offload = runtime_->clip_offload_params_to_cpu() ||
                            runtime_->plan_component_offload(loader, "text_encoders", remaining_free);
    const bool vae_offload = runtime_->vae_offload_params_to_cpu() ||
                             runtime_->plan_component_offload(loader, "first_stage_model", remaining_free);
    runtime_->finalize_auto_segment_budget(eff_budget);

    conditioner_ = std::make_shared<LLMEmbedder>(runtime_->clip_backend(),
                                                 te_offload,
                                                 storage,
                                                 version_,
                                                 "",
                                                 true);

    diffusion_.reset(new Qwen::QwenImageRunner(runtime_->backend(),
                                               diffusion_offload,
                                               storage,
                                               "model.diffusion_model",
                                               version_,
                                               qwen_zero_cond_t));
    auto process_group = runtime_->graph_process_group_ref();
    if (process_group != nullptr) {
        diffusion_->set_process_group(process_group);
        LOG_INFO("qwen-image-edit diffusion process group attached: backend=%s rank=%d world_size=%d",
                 edgedit::parallel::backend_name(process_group->backend()),
                 process_group->rank(),
                 process_group->size());
    }

    vae_ = std::make_shared<WAN::WanVAERunner>(runtime_->vae_backend(),
                                               vae_offload,
                                               storage,
                                               "first_stage_model",
                                               false,
                                               version_);

    const size_t max_graph_vram = runtime_->max_graph_vram_bytes();
    conditioner_->set_max_graph_vram_bytes(max_graph_vram);
    diffusion_->set_max_graph_vram_bytes(max_graph_vram);
    vae_->set_max_graph_vram_bytes(max_graph_vram);
    conditioner_->set_flash_attention_enabled(false);
    diffusion_->set_flash_attention_enabled(false);
    vae_->set_flash_attention_enabled(false);
    LOG_INFO("qwen-image-edit flash attention: text=%s diffusion=%s vae=%s",
             "off",
             "auto",
             "off");
    return true;
}

bool QwenImageEditPipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (conditioner_ == nullptr || diffusion_ == nullptr || vae_ == nullptr) {
        if (error != nullptr) {
            *error = "QwenImageEditPipeline components are not initialized";
        }
        return false;
    }

    registry.clear();

    if (!diffusion_->alloc_params_buffer()) {
        if (error != nullptr) {
            *error = "failed to allocate Qwen-Image-Edit transformer parameter buffer";
        }
        return false;
    }
    diffusion_->get_param_tensors(registry.tensors(), "model.diffusion_model");

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());
    // TE params buffer now allocated: real weight size is known. Set a TE-specific segment
    // budget so an offloaded TE segments instead of staging whole. No-op for a resident TE.
    conditioner_->set_max_graph_vram_bytes(
        runtime_->text_encoder_segment_budget(conditioner_->get_params_buffer_size()));

    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), "first_stage_model");

    registry.ignore_prefix("vae.");
    registry.ignore_prefix("cond_stage_model.");
    registry.ignore_prefix("text_encoders.llm.lm_head.");
    registry.ignore_prefix("text_encoders.llm.output.weight");
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");

    runtime_weights_loaded_ = true;
    return true;
}

void QwenImageEditPipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             version_ == VERSION_QWEN_IMAGE_EDIT &&
             runtime_weights_loaded_ &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr;
}

bool QwenImageEditPipeline::can_generate_image() const {
    return ready_ && runtime_weights_loaded_ && conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

bool QwenImageEditPipeline::validate_image_params(const ed_image_generation_params_t* params,
                                                  std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "image generation params are null";
        }
        return false;
    }
    if (params->init_image == nullptr || params->init_image->data == nullptr) {
        if (error != nullptr) {
            *error = "Qwen-Image-Edit requires an input image";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0) {
        if (error != nullptr) {
            *error = "image width and height must be positive";
        }
        return false;
    }
    if (params->width % ED_QWEN_EDIT_IMAGE_ALIGN != 0 || params->height % ED_QWEN_EDIT_IMAGE_ALIGN != 0) {
        if (error != nullptr) {
            *error = sd_format("Qwen-Image-Edit width and height must be divisible by %d", ED_QWEN_EDIT_IMAGE_ALIGN);
        }
        return false;
    }
    if (params->batch_count <= 0) {
        if (error != nullptr) {
            *error = "image batch_count must be positive";
        }
        return false;
    }
    return true;
}

ed_status_t QwenImageEditPipeline::generate_image(const ed_image_generation_params_t* params,
                                                  ed_image_batch_t* out,
                                                  std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "QwenImageEditPipeline is not initialized";
        }
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "image output is null";
        }
        return ED_STATUS_INVALID_ARGUMENT;
    }
    out->images = nullptr;
    out->count = 0;

    if (!validate_image_params(params, error)) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "current Qwen-Image-Edit pipeline needs transformer, LLM/vision encoder, and VAE weights";
        }
        return ED_STATUS_UNSUPPORTED;
    }

    const int count = params->batch_count > 0 ? params->batch_count : 1;
    const int steps = resolve_steps(params->sample.steps);
    if (GenerationControl* control = runtime_->generation_control()) {
        control->start(count * steps);
    }
    ed_image_t* images = static_cast<ed_image_t*>(std::calloc(static_cast<size_t>(count), sizeof(ed_image_t)));
    if (images == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate image batch";
        }
        return ED_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < count; ++i) {
        if (!generate_one_image(params, i, runtime_->n_threads(), &images[i], error)) {
            for (int j = 0; j <= i; ++j) {
                std::free(images[j].data);
            }
            std::free(images);
            return ED_STATUS_GENERATION_FAILED;
        }
    }

    out->images = images;
    out->count = count;
    return ED_STATUS_OK;
}

ed_status_t QwenImageEditPipeline::generate_video(const ed_video_generation_params_t*,
                                                  ed_video_t* out,
                                                  std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (error != nullptr) {
        *error = "video generation is not supported by QwenImageEditPipeline";
    }
    return ED_STATUS_UNSUPPORTED;
}

ed_scheduler_t QwenImageEditPipeline::default_scheduler(ed_sampler_t method) const {
    if (method == ED_SAMPLER_LCM || method == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (method == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

bool QwenImageEditPipeline::generate_one_image(const ed_image_generation_params_t* params,
                                               int batch_index,
                                               int n_threads,
                                               ed_image_t* image,
                                               std::string* error) {
    if (params == nullptr || image == nullptr || params->init_image == nullptr) {
        if (error != nullptr) {
            *error = "invalid Qwen-Image-Edit generation arguments";
        }
        return false;
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    const double init_ratio = params->init_image->height > 0
                                  ? static_cast<double>(params->init_image->width) /
                                        static_cast<double>(params->init_image->height)
                                  : 1.0;
    const QwenEditDimensions edit_image_size = qwen_edit_calculate_dimensions(1024.0 * 1024.0, init_ratio);
    LOG_INFO("qwen-image-edit input image resized from %ux%u to %dx%d for edit conditioning",
             params->init_image->width,
             params->init_image->height,
             edit_image_size.width,
             edit_image_size.height);

    sd::Tensor<float> input_image_tensor = resize_image_to_tensor_lanczos(*params->init_image,
                                                                          edit_image_size.width,
                                                                          edit_image_size.height);
    if (input_image_tensor.empty()) {
        if (error != nullptr) {
            *error = "failed to convert Qwen-Image-Edit input image to tensor";
        }
        return false;
    }
    qwen_edit_log_tensor_stats("preprocess.image_resized_unit", input_image_tensor);
    if (qwen_edit_debug_align_enabled()) {
        sd::Tensor<float> vae_image_tensor = input_image_tensor;
        for (int64_t i = 0; i < vae_image_tensor.numel(); ++i) {
            vae_image_tensor[i] = vae_image_tensor[i] * 2.0f - 1.0f;
        }
        qwen_edit_log_tensor_stats("preprocess.vae_image_tensor", vae_image_tensor);
    }
    std::vector<sd::Tensor<float>> condition_ref_images = {input_image_tensor};

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    cond_params.ref_images = &condition_ref_images;
    emit_phase_marker("encode", "begin");
    const int64_t ed_gen_t0 = ggml_time_ms();
    const int64_t ed_enc_t0 = ed_gen_t0;
    SDCondition condition = conditioner_->get_learned_condition(n_threads, cond_params);
    const int64_t ed_enc_ms = ggml_time_ms() - ed_enc_t0;
    if (condition.empty() || condition.c_crossattn.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image-Edit prompt encoding returned empty condition";
        }
        return false;
    }
    if (diffusion_bf16_) {
        qwen_edit_round_tensor_to_bf16(condition.c_crossattn);
    }
    qwen_edit_log_tensor_stats("cond.c_crossattn", condition.c_crossattn);

    const float cfg_scale = params->sample.cfg_scale > 0.0f ? params->sample.cfg_scale : 4.0f;
    const bool has_negative_prompt = params->negative_prompt != nullptr;
    const bool do_true_cfg = cfg_scale > 1.0f && has_negative_prompt;
    if (cfg_scale > 1.0f && !has_negative_prompt) {
        LOG_WARN("qwen-image-edit true CFG scale %.2f ignored because negative_prompt is not provided", cfg_scale);
    } else if (cfg_scale <= 1.0f && has_negative_prompt) {
        LOG_WARN("qwen-image-edit negative_prompt is provided but true CFG is disabled because cfg_scale <= 1");
    }
    SDCondition uncond;
    if (do_true_cfg) {
        ConditionerParams uncond_params;
        uncond_params.text = params->negative_prompt;
        uncond_params.ref_images = &condition_ref_images;
        uncond = conditioner_->get_learned_condition(n_threads, uncond_params);
        if (uncond.empty() || uncond.c_crossattn.empty()) {
            if (error != nullptr) {
                *error = "Qwen-Image-Edit negative prompt encoding returned empty condition";
            }
            return false;
        }
        if (diffusion_bf16_) {
            qwen_edit_round_tensor_to_bf16(uncond.c_crossattn);
        }
        qwen_edit_log_tensor_stats("uncond.c_crossattn", uncond.c_crossattn);
    }

    const int latent_w = params->width / vae_scale_factor;
    const int latent_h = params->height / vae_scale_factor;
    const int patch_size = std::max<int>(1, diffusion_->qwen_image_params.patch_size);
    if (latent_w % patch_size != 0 || latent_h % patch_size != 0) {
        if (error != nullptr) {
            *error = sd_format("Qwen-Image-Edit latent size %dx%d must be divisible by patch size %d",
                               latent_w,
                               latent_h,
                               patch_size);
        }
        return false;
    }

    sd::Tensor<float> encoded = vae_->encode(n_threads, input_image_tensor, runtime_->vae_tiling(), false, false);
    if (encoded.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image-Edit VAE encode failed";
        }
        return false;
    }
    qwen_edit_log_tensor_stats("vae.encoded", encoded);
    sd::Tensor<float> image_latent = vae_->vae_to_diffusion_latents(encoded);
    if (diffusion_bf16_) {
        qwen_edit_round_tensor_to_bf16(image_latent);
    }
    qwen_edit_log_tensor_stats("vae.image_latent", image_latent);
    qwen_edit_log_packed_latent_stats("image_latents_packed", image_latent);
    std::vector<sd::Tensor<float>> ref_latents = {image_latent};

    const int steps = resolve_steps(params->sample.steps);
    const int image_seq_len = (latent_w / patch_size) * (latent_h / patch_size);
    // The image_seq_len>=4096 gate is a CUDA-era heuristic; keep it for CUDA and
    // any non-CPU backend (do not change their behavior). On CPU it forced the
    // slow native f32 attention (512x512 -> seq 1024 < 4096); the shared
    // QwenImageRunner + ggml_ext_attention_ext CPU flash (oneDNN AMX, KV-nopad)
    // handles arbitrary seq/no-mask, so enable flash there whenever the runtime
    // allows it — matching the t2i Qwen pipeline.
    const bool is_cpu_backend = sd_backend_is(runtime_->backend(), "CPU");
    const bool diffusion_flash = runtime_->flash_attention() &&
                                 (is_cpu_backend || image_seq_len >= 4096);
    diffusion_->set_flash_attention_enabled(diffusion_flash);
    LOG_INFO("qwen-image-edit diffusion flash attention: %s (image_seq_len=%d)",
             diffusion_flash ? "on" : "off",
             image_seq_len);
    const bool has_explicit_flow_shift = params->sample.flow_shift > 0.0f &&
                                         std::isfinite(params->sample.flow_shift);

    const int64_t seed = params->seed >= 0 ? params->seed : 42;
    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "QwenImageEditPipeline has no RNG from ModelRuntime";
        }
        return false;
    }
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    qwen_edit_log_tensor_stats("noise.init_latent", init_latent);
    qwen_edit_log_tensor_stats("noise.randn", noise);
    float dynamic_mu = 0.0f;
    std::vector<float> sigmas = has_explicit_flow_shift
                                    ? qwen_edit_discrete_sigmas(steps, params->sample.flow_shift)
                                    : qwen_edit_diffusers_sigmas(steps, image_seq_len, &dynamic_mu);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create Qwen-Image-Edit sigma schedule";
        }
        return false;
    }

    if (has_explicit_flow_shift) {
        LOG_INFO("qwen-image-edit: %dx%d latent=%dx%d image_seq_len=%d steps=%d shift=%.2f true_cfg=%.2f sigma0=%.6f sigma_end=%.6f seed=%" PRId64,
                 params->width,
                 params->height,
                 latent_w,
                 latent_h,
                 image_seq_len,
                 steps,
                 params->sample.flow_shift,
                 cfg_scale,
                 sigmas.front(),
                 sigmas[static_cast<size_t>(steps - 1)],
                 seed + batch_index);
    } else {
        LOG_INFO("qwen-image-edit: %dx%d latent=%dx%d image_seq_len=%d steps=%d dynamic_mu=%.6f terminal=%.2f true_cfg=%.2f sigma0=%.6f sigma_end=%.6f seed=%" PRId64,
                 params->width,
                 params->height,
                 latent_w,
                 latent_h,
                 image_seq_len,
                 steps,
                 dynamic_mu,
                 ED_QWEN_EDIT_SHIFT_TERMINAL,
                 cfg_scale,
                 sigmas.front(),
                 sigmas[static_cast<size_t>(steps - 1)],
                 seed + batch_index);
    }

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    if (diffusion_bf16_) {
        qwen_edit_round_tensor_to_bf16(x);
    }
    qwen_edit_log_packed_latent_stats("latents_packed", x);
    cache::CacheRuntime cache_runtime;
    const bool cache_use_cfg_parallel = !uncond.empty() &&
                                        parallel::cfg_parallel_available(runtime_->parallel_context());
    const bool cache_seam_available =
        !cache_use_cfg_parallel && diffusion_->feature_cache_available();
    // Wire the device store whenever the block-stack seam is usable: the on-GPU
    // device path (MagCache feature-reuse + DiCache rings, face C) is the only
    // cache path.
    cache::ICacheDeviceStore* cache_store =
        (cache_seam_available && diffusion_ != nullptr)
            ? diffusion_->cache_device_store()
            : nullptr;
    const bool cache_enabled =
        cache_runtime.init(params->sample, version_, sigmas, cache_seam_available, cache_store,
                           cache_use_cfg_parallel);
    // GPU DiCache: set the probe depth for the capture snapshot.
    // Per-generation ring state is owned + freed by CacheStateManager::reset() (face C).
    if (cache_enabled && diffusion_ != nullptr) {
        diffusion_->dicache_probe_depth_ = cache_runtime.dicache_probe_depth();
    }
    const int64_t sample_start_ms = ggml_time_ms();
    emit_phase_marker("encode", "end");
    emit_phase_marker("denoise", "begin");
    GenerationControl* control = runtime_ != nullptr ? runtime_->generation_control() : nullptr;
    for (int step = 0; step < steps; ++step) {
        if (control != nullptr && control->should_cancel()) {
            control->mark_cancelled();
            if (error != nullptr && error->empty()) {
                *error = "generation cancelled";
            }
            diffusion_->free_compute_buffer();
            return false;
        }
        const float sigma = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];

        sd::Tensor<float> timesteps({1}, std::vector<float>{sigma * 1000.0f});
        if (diffusion_bf16_) {
            qwen_edit_round_tensor_to_bf16(timesteps);
        }
        cache::CacheStepInfo cache_step;
        cache_step.step_index = step;
        cache_step.num_steps = steps;
        cache_step.sigma = sigma;
        cache_step.sigma_next = sigma_next;
        if (cache_enabled) {
            cache_runtime.begin_step(cache_step);
        }

        const bool use_cfg_parallel = !uncond.empty() &&
                                      parallel::cfg_parallel_available(runtime_->parallel_context());
        const int cfg_rank = parallel::cfg_parallel_rank(runtime_->parallel_context());

        sd::Tensor<float> model_out;
        const void* condition_key = static_cast<const void*>(&condition);
        const cache::CacheBranch condition_branch = uncond.empty() ? cache::CacheBranch::Main
                                                                   : cache::CacheBranch::Cond;
        // Cache hooks for one condition. Feature/Probe seam gated to the plain
        // path and disabled under CFG-parallel (see flux_pipeline).
        auto make_hooks = [&](const SDCondition& cond_in) {
            cache::CacheRunnerHooks hooks;
            hooks.input = &x;
            hooks.full = [&, cond_in]() {
                return diffusion_->compute(n_threads, x, timesteps, cond_in.c_crossattn,
                                           ref_latents, true);
            };
            const bool seam_ok = !use_cfg_parallel && diffusion_->feature_cache_available();
            if (seam_ok) {
                const void* branch_key = static_cast<const void*>(&cond_in);
                const bool is_probe = cache_runtime.granularity() == cache::CacheGranularity::Probe;
                const bool feature_gpu = !is_probe &&
                    cache_runtime.granularity() == cache::CacheGranularity::Feature;
                if (feature_gpu) {
                    hooks.substep_capture = [&, cond_in](std::vector<cache::GraphExtension> exts) {
                        return diffusion_->compute_substep_capture(
                            n_threads, x, timesteps, cond_in.c_crossattn, ref_latents, true,
                            std::move(exts));
                    };
                    hooks.substep_inject_slot = [&, cond_in](std::vector<cache::GraphExtension> exts) {
                        return diffusion_->compute_substep_inject_slot(
                            n_threads, x, timesteps, cond_in.c_crossattn, ref_latents, true,
                            std::move(exts));
                    };
                }
                if (cache_runtime.granularity() == cache::CacheGranularity::Probe) {
                    const bool delta_minus = cache_runtime.dicache_delta_minus();
                    hooks.substep_probe = [&, cond_in, branch_key, delta_minus](int depth, const cache::CacheOperatorRegistry& operators,
                                                                                const cache::DiCacheSlotBridge& bridge) {
                        return diffusion_->compute_substep_probe(n_threads, x, timesteps,
                                                                 cond_in.c_crossattn, ref_latents,
                                                                 true, depth, branch_key, delta_minus, operators, bridge);
                    };
                    // DiCache is device-only on Qwen (no host fallback wired).
                    hooks.substep_inject_gpu = [&, cond_in](std::vector<cache::GraphExtension> exts,
                                                            const cache::DiCacheSlotBridge& bridge) {
                        return diffusion_->compute_substep_inject_gpu(n_threads, x, timesteps, cond_in.c_crossattn,
                                                                      ref_latents, true, std::move(exts), bridge);
                    };
                    const int probe_depth = cache_runtime.dicache_probe_depth();
                    hooks.substep_capture_probe = [&, cond_in, probe_depth](const cache::DiCacheSlotBridge& bridge) {
                        return diffusion_->compute_substep_capture_probe(
                            n_threads, x, timesteps, cond_in.c_crossattn, ref_latents,
                            true, probe_depth, bridge);
                    };
                }
            }
            return hooks;
        };

        if (step == 0 && !condition.c_crossattn.empty()) {
            qwen_edit_log_transformer_debug_targets(diffusion_.get(),
                                                    n_threads,
                                                    x,
                                                    timesteps,
                                                    condition.c_crossattn,
                                                    ref_latents);
            if (qwen_edit_debug_only_enabled()) {
                break;
            }
        }

        if (use_cfg_parallel) {
            const bool local_is_uncond = cfg_rank == 0;
            const SDCondition& local_condition = local_is_uncond ? uncond : condition;
            const cache::CacheBranch local_branch = local_is_uncond ? cache::CacheBranch::Uncond
                                                                    : cache::CacheBranch::Cond;
            const void* local_key = static_cast<const void*>(&local_condition);
            sd::Tensor<float> local_out = cache_enabled
                ? cache_runtime.run_branch(local_branch, local_key, make_hooks(local_condition))
                : make_hooks(local_condition).full();
            std::vector<sd::Tensor<float>> gathered;
            if (local_out.empty() ||
                !parallel::cfg_all_gather(*runtime_->parallel_context(), local_out, &gathered, error) ||
                gathered.size() != 2) {
                if (error != nullptr && error->empty()) {
                    *error = sd_format("Qwen-Image-Edit CFG parallel gather failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
            model_out = true_cfg_rescale(gathered[1], gathered[0], cfg_scale, patch_size);
        } else {
            model_out = cache_enabled
                ? cache_runtime.run_branch(condition_branch, condition_key, make_hooks(condition))
                : make_hooks(condition).full();
        }

        if (!uncond.empty() && !use_cfg_parallel) {
            const void* uncond_key = static_cast<const void*>(&uncond);
            sd::Tensor<float> uncond_out = cache_enabled
                ? cache_runtime.run_branch(cache::CacheBranch::Uncond, uncond_key, make_hooks(uncond))
                : make_hooks(uncond).full();
            if (step == 0) {
                qwen_edit_log_tensor_stats("step0.uncond_model_out", uncond_out);
                qwen_edit_log_packed_latent_stats("step0.uncond_model_out_packed", uncond_out);
            }
            if (uncond_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("Qwen-Image-Edit unconditional transformer compute failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
            if (step == 0) {
                qwen_edit_log_tensor_stats("step0.cond_model_out", model_out);
                qwen_edit_log_packed_latent_stats("step0.cond_model_out_packed", model_out);
            }
            model_out = true_cfg_rescale(model_out, uncond_out, cfg_scale, patch_size);
        }
        if (model_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("Qwen-Image-Edit transformer compute failed at step %d", step + 1);
            }
            diffusion_->free_compute_buffer();
            return false;
        }
        if (diffusion_bf16_) {
            qwen_edit_round_tensor_to_bf16(model_out);
        }
        if (step == 0) {
            qwen_edit_log_tensor_stats("step0.model_out", model_out);
            qwen_edit_log_packed_latent_stats("step0.model_out_packed", model_out);
        }

        // SenCache calibration: finite-diff sensitivities on the true-CFG-combined
        // velocity. Two extra plain forwards/step, calibration only; off under
        // CFG-parallel. Qwen-Image-Edit feeds timestep = sigma*1000.
        if (cache_enabled && cache_runtime.needs_calibration() && !use_cfg_parallel) {
            auto forward_at = [&](const sd::Tensor<float>& x_raw, float sigma_eval) -> sd::Tensor<float> {
                sd::Tensor<float> ts({1}, std::vector<float>{sigma_eval * 1000.0f});
                sd::Tensor<float> cond_v = diffusion_->compute(n_threads, x_raw, ts, condition.c_crossattn,
                                                               ref_latents, true);
                if (cond_v.empty() || uncond.empty()) {
                    return cond_v;
                }
                sd::Tensor<float> uncond_v = diffusion_->compute(n_threads, x_raw, ts, uncond.c_crossattn,
                                                                 ref_latents, true);
                if (uncond_v.empty()) {
                    return {};
                }
                return true_cfg_rescale(cond_v, uncond_v, cfg_scale, patch_size);
            };
            cache_runtime.calibrate(condition_branch, condition_key, x, model_out, forward_at);
        }

        sd::Tensor<float> denoised = model_out * (-sigma) + x;
        if (sigma == 0.0f) {
            x = denoised;
        } else {
            const sd::Tensor<float> d = (x - denoised) / sigma;
            x += d * (sigma_next - sigma);
        }
        if (diffusion_bf16_) {
            qwen_edit_round_tensor_to_bf16(x);
        }
        LOG_INFO("qwen-image-edit step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
        if (cache_enabled) {
            cache_runtime.end_step(cache_step);
        }
        if (control != nullptr) {
            control->step_done();
        }
    }
    if (cache_enabled) {
        cache_runtime.log_summary(static_cast<size_t>(steps));
    }
    const int64_t sample_end_ms = ggml_time_ms();
    LOG_INFO("qwen-image-edit sampling completed, taking %.2fs", (sample_end_ms - sample_start_ms) / 1000.0f);
    emit_phase_marker("denoise", "end");
    diffusion_->free_compute_buffer();

    if (runtime_->parallel_context() != nullptr && !runtime_->parallel_context()->is_root()) {
        return true;
    }

    emit_phase_marker("decode", "begin");
    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    const int64_t ed_dec_t0 = ggml_time_ms();
    sd::Tensor<float> decoded = vae_->decode(n_threads,
                                             vae_latents,
                                             runtime_->vae_tiling(),
                                             false,
                                             false,
                                             false);
    const int64_t ed_dec_ms = ggml_time_ms() - ed_dec_t0;
    emit_phase_marker("decode", "end");
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image-Edit VAE decode failed";
        }
        return false;
    }

    const ed_status_t status = qwen_edit_tensor_to_image(decoded, image);
    if (status != ED_STATUS_OK) {
        if (error != nullptr) {
            *error = status == ED_STATUS_OUT_OF_MEMORY
                         ? "failed to allocate decoded Qwen-Image-Edit image"
                         : "decoded Qwen-Image-Edit tensor has invalid shape";
        }
        return false;
    }
    const int64_t ed_total_ms = ggml_time_ms() - ed_gen_t0;
    const int64_t ed_sample_ms = sample_end_ms - sample_start_ms;
    LOG_INFO("qwen-image-edit generate breakdown: total=%.2fs | text_encode=%.2fs sampling=%.2fs vae_decode=%.2fs other=%.2fs",
             ed_total_ms/1000.0f, ed_enc_ms/1000.0f, ed_sample_ms/1000.0f, ed_dec_ms/1000.0f,
             (ed_total_ms - ed_enc_ms - ed_sample_ms - ed_dec_ms)/1000.0f);
    return true;
}

}  // namespace edgedit
