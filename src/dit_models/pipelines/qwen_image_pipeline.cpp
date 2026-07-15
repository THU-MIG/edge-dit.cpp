#include "dit_models/pipelines/qwen_image_pipeline.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "parallel/process_group.hpp"
#include "core/optimization/cache/runtime/cache_engine.hpp"
#include "dit_models/components/autoencoders/vae.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "dit_models/models/qwen_image.hpp"
#include "dit_models/models/wan.hpp"
#include "dit_models/pipelines/dit_pipeline_utils.hpp"
#include "ggml.h"
#include "parallel/cfg_parallel.hpp"
#include "utils/util.h"

static constexpr int ED_QWEN_IMAGE_ALIGN = 32;
static constexpr float ED_QWEN_SHIFT_TERMINAL = 0.02f;

static void ed_qwen_round_tensor_to_bf16(sd::Tensor<float>& tensor) {
    for (float& value : tensor.values()) {
        value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
    }
}

static sd::Tensor<float> ed_qwen_apply_true_cfg(const sd::Tensor<float>& cond,
                                                const sd::Tensor<float>& uncond,
                                                float cfg_scale,
                                                int patch_size) {
    if (cond.empty() || uncond.empty() || cond.shape() != uncond.shape()) {
        return {};
    }
    const auto& shape = cond.shape();
    if (shape.size() != 4 || patch_size <= 0 ||
        shape[0] % patch_size != 0 || shape[1] % patch_size != 0) {
        return {};
    }

    const int64_t width = shape[0];
    const int64_t height = shape[1];
    const int64_t channels = shape[2];
    const int64_t batch = shape[3];
    sd::Tensor<float> out(shape);

    auto offset_of = [&](int64_t x, int64_t y, int64_t c, int64_t n) -> int64_t {
        return x + width * (y + height * (c + channels * n));
    };

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t py = 0; py < height; py += patch_size) {
            for (int64_t px = 0; px < width; px += patch_size) {
                double cond_sq = 0.0;
                double guided_sq = 0.0;
                for (int64_t dy = 0; dy < patch_size; ++dy) {
                    for (int64_t dx = 0; dx < patch_size; ++dx) {
                        for (int64_t c = 0; c < channels; ++c) {
                            const int64_t off = offset_of(px + dx, py + dy, c, n);
                            const float cond_value = cond[off];
                            const float guided_value = uncond[off] + cfg_scale * (cond_value - uncond[off]);
                            cond_sq += static_cast<double>(cond_value) * static_cast<double>(cond_value);
                            guided_sq += static_cast<double>(guided_value) * static_cast<double>(guided_value);
                        }
                    }
                }

                const float rescale = guided_sq > 0.0
                                          ? static_cast<float>(std::sqrt(cond_sq) / std::sqrt(guided_sq))
                                          : 0.0f;
                for (int64_t dy = 0; dy < patch_size; ++dy) {
                    for (int64_t dx = 0; dx < patch_size; ++dx) {
                        for (int64_t c = 0; c < channels; ++c) {
                            const int64_t off = offset_of(px + dx, py + dy, c, n);
                            const float guided_value = uncond[off] + cfg_scale * (cond[off] - uncond[off]);
                            out[off] = guided_value * rescale;
                        }
                    }
                }
            }
        }
    }
    return out;
}

static ed_status_t ed_tensor_to_image(const sd::Tensor<float>& tensor, ed_image_t* image) {
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
    auto to_u8 = [](float value) -> uint8_t {
        if (value <= 0.0f) {
            return 0;
        }
        if (value >= 1.0f) {
            return 255;
        }
        return static_cast<uint8_t>(value * 255.0f + 0.5f);
    };
    uint8_t* data = static_cast<uint8_t*>(std::malloc(nbytes));
    if (data == nullptr) {
        return ED_STATUS_OUT_OF_MEMORY;
    }

    const size_t pixels = width * height;
    const float* src = tensor.data();
    if (channels == 3) {
        const float* c0 = src;
        const float* c1 = src + pixels;
        const float* c2 = src + pixels * 2;
        for (size_t i = 0; i < pixels; ++i) {
            data[i * 3 + 0] = to_u8(c0[i]);
            data[i * 3 + 1] = to_u8(c1[i]);
            data[i * 3 + 2] = to_u8(c2[i]);
        }
    } else {
        for (size_t i = 0; i < pixels; ++i) {
            for (size_t c = 0; c < channels; ++c) {
                data[i * channels + c] = to_u8(src[i + pixels * c]);
            }
        }
    }
    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data = data;
    return ED_STATUS_OK;
}

static float ed_qwen_time_snr_shift(float shift, float t) {
    return shift * t / (1.0f + (shift - 1.0f) * t);
}

static float ed_qwen_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return ed_qwen_time_snr_shift(shift, t / 1000.0f);
}

static std::vector<float> ed_qwen_discrete_sigmas(int steps, float shift) {
    std::vector<float> result;
    if (steps <= 0) {
        return result;
    }
    if (steps == 1) {
        result.push_back(ed_qwen_t_to_sigma(999.0f, shift));
        result.push_back(0.0f);
        return result;
    }

    const float step = 999.0f / static_cast<float>(steps - 1);
    result.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        result.push_back(ed_qwen_t_to_sigma(t, shift));
    }
    result.push_back(0.0f);
    return result;
}

static float ed_qwen_time_shift_exponential(float mu, float t) {
    return std::exp(mu) / (std::exp(mu) + std::pow(1.0f / t - 1.0f, 1.0f));
}

static void ed_qwen_stretch_shift_to_terminal(std::vector<float>& sigmas, float terminal) {
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

static std::vector<float> ed_qwen_diffusers_sigmas(int steps, int64_t image_seq_len, float* out_mu = nullptr) {
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
            result.push_back(ed_qwen_time_shift_exponential(mu, sigma));
        }
    }

    ed_qwen_stretch_shift_to_terminal(result, ED_QWEN_SHIFT_TERMINAL);
    result.push_back(0.0f);
    return result;
}

namespace edgedit {

QwenImagePipeline::QwenImagePipeline(SDVersion version)
    : version_(version) {
}

QwenImagePipeline::~QwenImagePipeline() {
    reset();
}

void QwenImagePipeline::reset() {
    conditioner_.reset();
    vae_.reset();
    diffusion_.reset();
    diffusion_bf16_ = false;
    runtime_weights_loaded_ = false;
}

bool QwenImagePipeline::has_prefix(const ModelLoader& loader, const std::string& prefix) const {
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (starts_with(item.second.name, prefix)) {
            return true;
        }
    }
    return false;
}

bool QwenImagePipeline::prepare(const ed_context_params_t& params,
                                ModelRuntime& runtime,
                                const ModelLoader& loader,
                                PipelineTensorRegistry& registry,
                                std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    reset();

    if (!ed_version_is_qwen_image(version_)) {
        if (error != nullptr) {
            *error = "QwenImagePipeline got non-Qwen-Image model version";
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

bool QwenImagePipeline::build_components(const ed_context_params_t& params,
                                         const ModelLoader& loader,
                                         std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        if (error != nullptr) {
            *error = "QwenImagePipeline requires initialized ModelRuntime backends";
        }
        return false;
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool offload = runtime_->offload_params_to_cpu();
    const bool enable_vision = params.llm_vision_path != nullptr && params.llm_vision_path[0] != '\0';

    conditioner_ = std::make_shared<LLMEmbedder>(runtime_->clip_backend(),
                                                 offload,
                                                 storage,
                                                 version_,
                                                 "",
                                                 enable_vision);

    diffusion_.reset(new Qwen::QwenImageRunner(runtime_->backend(),
                                               offload,
                                               storage,
                                               "model.diffusion_model",
                                               version_,
                                               false));
    if (runtime_ != nullptr) {
        auto process_group = runtime_->graph_process_group_ref();
        if (process_group != nullptr) {
            diffusion_->set_process_group(process_group);
            LOG_INFO("qwen-image diffusion process group attached: backend=%s rank=%d world_size=%d",
                    edgedit::parallel::backend_name(process_group->backend()),
                    process_group->rank(),
                    process_group->size());
        }
    }
    vae_ = std::make_shared<WAN::WanVAERunner>(runtime_->vae_backend(),
                                               offload,
                                               storage,
                                               "first_stage_model",
                                               true,
                                               version_);

    const bool text_flash = runtime_->flash_attention();
    const bool diffusion_flash = runtime_->flash_attention();
    conditioner_->set_flash_attention_enabled(text_flash);
    diffusion_->set_flash_attention_enabled(diffusion_flash);
    vae_->set_flash_attention_enabled(text_flash);
    LOG_INFO("qwen-image flash attention: text=%s diffusion=%s vae=%s",
             text_flash ? "on" : "off",
             diffusion_flash ? "on" : "off",
             text_flash ? "on" : "off");

    return true;
}

bool QwenImagePipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (conditioner_ == nullptr || diffusion_ == nullptr || vae_ == nullptr) {
        if (error != nullptr) {
            *error = "QwenImagePipeline components are not initialized";
        }
        return false;
    }

    registry.clear();

    if (!diffusion_->alloc_params_buffer()) {
        if (error != nullptr) {
            *error = "failed to allocate Qwen-Image transformer parameter buffer";
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
    registry.ignore_prefix("first_stage_model.encoder");
    registry.ignore_prefix("first_stage_model.conv1.");
    registry.ignore_prefix("text_encoders.llm.lm_head.");
    registry.ignore_prefix("text_encoders.llm.output.weight");
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");

    registry.ignore_prefix("text_encoders.llm.visual.");

    runtime_weights_loaded_ = true;
    return true;
}

void QwenImagePipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             version_ == VERSION_QWEN_IMAGE &&
             runtime_weights_loaded_ &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr;
}

bool QwenImagePipeline::can_generate_image() const {
    return ready_ && runtime_weights_loaded_ && conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

bool QwenImagePipeline::validate_image_params(const ed_image_generation_params_t* params,
                                              std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "image generation params are null";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0) {
        if (error != nullptr) {
            *error = "image width and height must be positive";
        }
        return false;
    }
    if (params->width % ED_QWEN_IMAGE_ALIGN != 0 || params->height % ED_QWEN_IMAGE_ALIGN != 0) {
        if (error != nullptr) {
            *error = sd_format("Qwen-Image width and height must be divisible by %d", ED_QWEN_IMAGE_ALIGN);
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

ed_status_t QwenImagePipeline::generate_image(const ed_image_generation_params_t* params,
                                              ed_image_batch_t* out,
                                              std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "QwenImagePipeline is not initialized";
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
            *error = "current Qwen-Image pipeline needs transformer, LLM text encoder, and VAE weights";
        }
        return ED_STATUS_UNSUPPORTED;
    }

    const int count = params->batch_count > 0 ? params->batch_count : 1;
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
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

ed_status_t QwenImagePipeline::generate_video(const ed_video_generation_params_t*,
                                              ed_video_t* out,
                                              std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (error != nullptr) {
        *error = "video generation is not supported by QwenImagePipeline";
    }
    return ED_STATUS_UNSUPPORTED;
}

bool QwenImagePipeline::supports_image_generation() const {
    return ready_;
}

bool QwenImagePipeline::supports_video_generation() const {
    return false;
}

ed_sampler_t QwenImagePipeline::default_sample_method() const {
    return ED_SAMPLER_EULER;
}

ed_scheduler_t QwenImagePipeline::default_scheduler(ed_sampler_t method) const {
    if (method == ED_SAMPLER_LCM || method == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (method == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

bool QwenImagePipeline::generate_one_image(const ed_image_generation_params_t* params,
                                           int batch_index,
                                           int n_threads,
                                           ed_image_t* image,
                                           std::string* error) {
    if (params == nullptr || image == nullptr) {
        if (error != nullptr) {
            *error = "invalid Qwen-Image generation arguments";
        }
        return false;
    }

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    SDCondition condition = conditioner_->get_learned_condition(n_threads, cond_params);
    if (condition.empty() || condition.c_crossattn.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image prompt encoding returned empty condition";
        }
        return false;
    }
    if (diffusion_bf16_) {
        ed_qwen_round_tensor_to_bf16(condition.c_crossattn);
    }

    const float cfg_scale = params->sample.cfg_scale > 0.0f ? params->sample.cfg_scale : 1.0f;
    const bool do_true_cfg = cfg_scale > 1.0f && params->negative_prompt != nullptr;
    SDCondition uncond;
    if (do_true_cfg) {
        ConditionerParams uncond_params;
        uncond_params.text = params->negative_prompt;
        uncond = conditioner_->get_learned_condition(n_threads, uncond_params);
        if (uncond.empty() || uncond.c_crossattn.empty()) {
            if (error != nullptr) {
                *error = "Qwen-Image negative prompt encoding returned empty condition";
            }
            return false;
        }
        if (diffusion_bf16_) {
            ed_qwen_round_tensor_to_bf16(uncond.c_crossattn);
        }
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    const int latent_w = params->width / vae_scale_factor;
    const int latent_h = params->height / vae_scale_factor;
    const int patch_size = std::max<int>(1, diffusion_->qwen_image_params.patch_size);
    if (latent_w % patch_size != 0 || latent_h % patch_size != 0) {
        if (error != nullptr) {
            *error = sd_format("Qwen-Image latent size %dx%d must be divisible by patch size %d",
                               latent_w,
                               latent_h,
                               patch_size);
        }
        return false;
    }

    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    const bool has_explicit_flow_shift = params->sample.flow_shift > 0.0f &&
                                         std::isfinite(params->sample.flow_shift);

    const int64_t seed = params->seed >= 0 ? params->seed : 42;
    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "QwenImagePipeline has no RNG from ModelRuntime";
        }
        return false;
    }
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    float dynamic_mu = 0.0f;
    std::vector<float> sigmas = has_explicit_flow_shift
                                    ? ed_qwen_discrete_sigmas(steps, params->sample.flow_shift)
                                    : ed_qwen_diffusers_sigmas(
                                          steps,
                                          static_cast<int64_t>(latent_w / patch_size) *
                                              static_cast<int64_t>(latent_h / patch_size),
                                          &dynamic_mu);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create Qwen-Image sigma schedule";
        }
        return false;
    }

    if (has_explicit_flow_shift) {
        LOG_INFO("qwen-image txt2img: %dx%d latent=%dx%d steps=%d shift=%.2f cfg=%.2f seed=%" PRId64,
                 params->width,
                 params->height,
                 latent_w,
                 latent_h,
                 steps,
                 params->sample.flow_shift,
                 cfg_scale,
                 seed + batch_index);
    } else {
        LOG_INFO("qwen-image txt2img: %dx%d latent=%dx%d steps=%d dynamic_mu=%.6f terminal=%.2f cfg=%.2f seed=%" PRId64,
                 params->width,
                 params->height,
                 latent_w,
                 latent_h,
                 steps,
                 dynamic_mu,
                 ED_QWEN_SHIFT_TERMINAL,
                 cfg_scale,
                 seed + batch_index);
    }

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    if (diffusion_bf16_) {
        ed_qwen_round_tensor_to_bf16(x);
    }
    auto compute_diffusion = [&](const sd::Tensor<float>& input,
                                 const sd::Tensor<float>& timesteps,
                                 const SDCondition& condition) -> sd::Tensor<float> {
        return diffusion_->compute(n_threads,
                                   input,
                                   timesteps,
                                   condition.c_crossattn,
                                   {},
                                   false);
    };
    cache::CacheRuntime cache_runtime;
    const bool cache_use_cfg_parallel = !uncond.empty() &&
                                        parallel::cfg_parallel_available(runtime_->parallel_context());
    const bool cache_seam_available =
        !cache_use_cfg_parallel && diffusion_->feature_cache_available();
    // Wire the device store only when the on-GPU feature-reuse path is active
    // (ED_FEATURE_CACHE_GPU on); with it off, leave the store null so a
    // device_backed slot cleanly falls back to the host declarative path.
    cache::ICacheDeviceStore* cache_store =
        (cache_seam_available && diffusion_ != nullptr &&
         Qwen::QwenImageRunner::feature_gpu_enabled())
            ? diffusion_->cache_device_store()
            : nullptr;
    const bool cache_enabled =
        cache_runtime.init(params->sample, version_, sigmas, cache_seam_available, cache_store,
                           cache_use_cfg_parallel);
    // GPU DiCache (ED_DICACHE_GPU): reset per-generation persistent state and set
    // the probe depth the capture step uses to snapshot its probe residual. Read
    // the resolved depth from the engine so it stays in sync with the policy config.
    if (cache_enabled && diffusion_ != nullptr) {
        diffusion_->reset_dicache_gpu_states();
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
            ed_qwen_round_tensor_to_bf16(timesteps);
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

        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        // Cache hooks for one condition. Feature/Probe seam gated to the plain
        // path and disabled under CFG-parallel (see flux_pipeline).
        auto make_hooks = [&](const SDCondition& cond_in) {
            cache::CacheRunnerHooks hooks;
            hooks.input = &x;
            hooks.full = [&, cond_in]() {
                return diffusion_->compute(n_threads, x, timesteps, cond_in.c_crossattn,
                                           empty_ref_latents, false);
            };
            const bool seam_ok = !use_cfg_parallel && diffusion_->feature_cache_available();
            if (seam_ok) {
                const void* branch_key = static_cast<const void*>(&cond_in);
                // Only DiCache (Probe) uses the on-GPU probe/inject seam that a
                // branch_key drives; passing it into compute_capture for a Feature
                // method (MagCache/TaylorSeer/SenCache) would flip gpu_metric on and
                // suppress the host feature readback those methods rely on.
                const bool is_probe = cache_runtime.granularity() == cache::CacheGranularity::Probe;
                // Feature-granularity on-GPU reuse: keep the captured residual on
                // device and inject it there on skips, avoiding the ~50MB host
                // reconstruct copy + H2D upload the host inject path pays per skip.
                const bool feature_gpu = !is_probe &&
                    cache_runtime.granularity() == cache::CacheGranularity::Feature &&
                    Qwen::QwenImageRunner::feature_gpu_enabled();
                if (feature_gpu) {
                    // Substep-path tap-driven capture: same slot
                    // contract, but driven through the TapRegistry, not CacheGraphScope.
                    hooks.substep_capture = [&, cond_in](std::vector<cache::GraphExtension> exts) {
                        return diffusion_->compute_substep_capture(
                            n_threads, x, timesteps, cond_in.c_crossattn, empty_ref_latents, false,
                            std::move(exts));
                    };
                    // Substep-path tap-driven device inject (MagCache): x_before + slot
                    // via the forward's registry inject, no CacheGraphScope.
                    hooks.substep_inject_slot = [&, cond_in](void* slot, int region_start, int region_end) {
                        return diffusion_->compute_substep_inject_slot(
                            n_threads, x, timesteps, cond_in.c_crossattn, empty_ref_latents, false,
                            static_cast<ggml_tensor*>(slot), region_start, region_end);
                    };
                }
                if (cache_runtime.granularity() == cache::CacheGranularity::Probe) {
                    // Substep-path tap-driven probe: delta_y/gamma
                    // computed on-device from taps + persistent operands, no scope.
                    const bool delta_minus = cache_runtime.dicache_delta_minus();
                    hooks.substep_probe = [&, cond_in, branch_key, delta_minus](int depth) {
                        return diffusion_->compute_substep_probe(n_threads, x, timesteps,
                                                                 cond_in.c_crossattn, empty_ref_latents,
                                                                 false, depth, branch_key, delta_minus);
                    };
                    // Only wire on-GPU inject when the model's GPU DiCache path is
                    // active; with ED_DICACHE_GPU=0 the lowering takes the declarative
                    // host probe path instead (residual-ring blend).
                    if (Qwen::QwenImageRunner::dicache_gpu_enabled()) {
                        // Substep-path tap-driven device inject (DiCache gamma-blend).
                        hooks.substep_inject_gpu = [&, cond_in, branch_key](float gamma, int region_start, int region_end) {
                            return diffusion_->compute_substep_inject_gpu(n_threads, x, timesteps, cond_in.c_crossattn,
                                                                          empty_ref_latents, false, gamma, branch_key,
                                                                          region_start, region_end);
                        };
                        // Substep-path tap-driven seed capture: full forward that
                        // refreshes the DiCacheGpuState rings device-to-device (replaces
                        // the legacy compute_capture / run_cache_pass path).
                        const int probe_depth = cache_runtime.dicache_probe_depth();
                        hooks.substep_capture_probe = [&, cond_in, branch_key, probe_depth]() {
                            return diffusion_->compute_substep_capture_probe(
                                n_threads, x, timesteps, cond_in.c_crossattn, empty_ref_latents,
                                false, probe_depth, branch_key);
                        };
                    }
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
                    *error = sd_format("Qwen-Image CFG parallel gather failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
            model_out = ed_qwen_apply_true_cfg(gathered[1], gathered[0], cfg_scale, patch_size);
            if (model_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("Qwen-Image true CFG rescale failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
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
                    *error = sd_format("Qwen-Image unconditional transformer compute failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
            model_out = ed_qwen_apply_true_cfg(model_out, uncond_out, cfg_scale, patch_size);
            if (model_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("Qwen-Image true CFG rescale failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
        }
        if (model_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("Qwen-Image transformer compute failed at step %d", step + 1);
            }
            diffusion_->free_compute_buffer();
            return false;
        }
        if (diffusion_bf16_) {
            ed_qwen_round_tensor_to_bf16(model_out);
        }

        // SenCache calibration: finite-diff sensitivities on the true-CFG-combined
        // velocity. Two extra plain forwards/step, calibration only; off under
        // CFG-parallel. Qwen feeds timestep = sigma*1000.
        if (cache_enabled && cache_runtime.needs_calibration() && !use_cfg_parallel) {
            auto forward_at = [&](const sd::Tensor<float>& x_raw, float sigma_eval) -> sd::Tensor<float> {
                sd::Tensor<float> ts({1}, std::vector<float>{sigma_eval * 1000.0f});
                sd::Tensor<float> cond_v = diffusion_->compute(n_threads, x_raw, ts, condition.c_crossattn,
                                                               empty_ref_latents, false);
                if (cond_v.empty() || uncond.empty()) {
                    return cond_v;
                }
                sd::Tensor<float> uncond_v = diffusion_->compute(n_threads, x_raw, ts, uncond.c_crossattn,
                                                                 empty_ref_latents, false);
                if (uncond_v.empty()) {
                    return {};
                }
                return ed_qwen_apply_true_cfg(cond_v, uncond_v, cfg_scale, patch_size);
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
            ed_qwen_round_tensor_to_bf16(x);
        }
        LOG_INFO("qwen-image step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
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
    LOG_INFO("qwen-image sampling completed, taking %.2fs", (sample_end_ms - sample_start_ms) / 1000.0f);
    diffusion_->free_compute_buffer();

    if (runtime_->parallel_context() != nullptr && !runtime_->parallel_context()->is_root()) {
        return true;
    }

    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    sd::Tensor<float> decoded = vae_->decode(n_threads,
                                             vae_latents,
                                             runtime_->vae_tiling(),
                                             false,
                                             false,
                                             false);
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image VAE decode failed";
        }
        return false;
    }

    const ed_status_t status = ed_tensor_to_image(decoded, image);
    if (status != ED_STATUS_OK) {
        if (error != nullptr) {
            *error = status == ED_STATUS_OUT_OF_MEMORY
                         ? "failed to allocate decoded Qwen-Image image"
                         : "decoded Qwen-Image tensor has invalid shape";
        }
        return false;
    }
    return true;
}

}  // namespace edgedit
