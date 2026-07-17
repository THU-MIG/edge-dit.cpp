#include "dit_models/pipelines/qwen_image_edit_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstring>
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

static void qwen_edit_round_tensor_to_bf16(sd::Tensor<float>& tensor) {
    for (float& value : tensor.values()) {
        value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
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
    sd::Tensor<float> tensor = resize_image_to_tensor(image,
                                                      static_cast<int>(image.width),
                                                      static_cast<int>(image.height));
    if (tensor.empty() || static_cast<int>(image.width) == width && static_cast<int>(image.height) == height) {
        return tensor;
    }
    return sd::ops::interpolate(tensor,
                                std::vector<int64_t>{width, height, tensor.shape()[2], tensor.shape()[3]},
                                sd::ops::InterpolateMode::Lanczos,
                                false,
                                true);
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

bool QwenImageEditPipeline::prepare(const ed_context_params_t& params,
                                    ModelRuntime& runtime,
                                    const ModelLoader& loader,
                                    PipelineTensorRegistry& registry,
                                    std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    reset();

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
    const bool offload = runtime_->offload_params_to_cpu();
    conditioner_ = std::make_shared<LLMEmbedder>(runtime_->clip_backend(),
                                                 offload,
                                                 storage,
                                                 version_,
                                                 "",
                                                 true);

    diffusion_.reset(new Qwen::QwenImageRunner(runtime_->backend(),
                                               offload,
                                               storage,
                                               "model.diffusion_model",
                                               version_,
                                               params.qwen_image_zero_cond_t));
    auto process_group = runtime_->graph_process_group_ref();
    if (process_group != nullptr) {
        diffusion_->set_process_group(process_group);
        LOG_INFO("qwen-image-edit diffusion process group attached: backend=%s rank=%d world_size=%d",
                 edgedit::parallel::backend_name(process_group->backend()),
                 process_group->rank(),
                 process_group->size());
    }

    vae_ = std::make_shared<WAN::WanVAERunner>(runtime_->vae_backend(),
                                               offload,
                                               storage,
                                               "first_stage_model",
                                               false,
                                               version_);

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
    const int steps = params->sample.steps > 0 ? params->sample.steps : 50;
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
    sd::Tensor<float> input_image_tensor = resize_image_to_tensor_lanczos(*params->init_image, params->width, params->height);
    if (input_image_tensor.empty()) {
        if (error != nullptr) {
            *error = "failed to convert input image to tensor";
        }
        return false;
    }
    const double init_ratio = params->init_image->height > 0
                                  ? static_cast<double>(params->init_image->width) /
                                        static_cast<double>(params->init_image->height)
                                  : 1.0;
    int condition_w = 384;
    int condition_h = 384;
    if (init_ratio > 0.0 && std::isfinite(init_ratio)) {
        condition_w = static_cast<int>(std::round(std::sqrt(384.0 * 384.0 * init_ratio) / 32.0) * 32.0);
        condition_h = static_cast<int>(std::round((static_cast<double>(condition_w) / init_ratio) / 32.0) * 32.0);
        condition_w = std::max(32, condition_w);
        condition_h = std::max(32, condition_h);
    }
    sd::Tensor<float> condition_image_tensor = resize_image_to_tensor_lanczos(*params->init_image,
                                                                              condition_w,
                                                                              condition_h);
    if (condition_image_tensor.empty()) {
        if (error != nullptr) {
            *error = "failed to convert Qwen-Image-Edit condition image to tensor";
        }
        return false;
    }
    std::vector<sd::Tensor<float>> condition_ref_images = {condition_image_tensor};

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    cond_params.ref_images = &condition_ref_images;
    SDCondition condition = conditioner_->get_learned_condition(n_threads, cond_params);
    if (condition.empty() || condition.c_crossattn.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image-Edit prompt encoding returned empty condition";
        }
        return false;
    }
    if (diffusion_bf16_) {
        qwen_edit_round_tensor_to_bf16(condition.c_crossattn);
    }

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

    ed_tiling_params_t tiling_params{};
    sd::Tensor<float> encoded = vae_->encode(n_threads, input_image_tensor, tiling_params, false, false);
    if (encoded.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image-Edit VAE encode failed";
        }
        return false;
    }
    sd::Tensor<float> image_latent = vae_->vae_to_diffusion_latents(encoded);
    if (diffusion_bf16_) {
        qwen_edit_round_tensor_to_bf16(image_latent);
    }
    std::vector<sd::Tensor<float>> ref_latents = {image_latent};

    const int steps = params->sample.steps > 0 ? params->sample.steps : 50;
    const int image_seq_len = (latent_w / patch_size) * (latent_h / patch_size);
    const bool diffusion_flash = runtime_->flash_attention() && image_seq_len >= 4096;
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
            if (uncond_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("Qwen-Image-Edit unconditional transformer compute failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
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
    diffusion_->free_compute_buffer();

    if (runtime_->parallel_context() != nullptr && !runtime_->parallel_context()->is_root()) {
        return true;
    }

    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    sd::Tensor<float> decoded = vae_->decode(n_threads,
                                             vae_latents,
                                             tiling_params,
                                             false,
                                             false,
                                             false);
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
    return true;
}

}  // namespace edgedit
