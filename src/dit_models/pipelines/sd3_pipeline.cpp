#include "dit_models/pipelines/sd3_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include "parallel/process_group.hpp"
#include "core/optimization/cache/runtime/cache_engine.hpp"
#include "dit_models/components/autoencoders/auto_encoder_kl.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "parallel/cfg_parallel.hpp"
#include "utils/util.h"

namespace edgedit {
namespace {

float sd3_time_snr_shift(float shift, float t) {
    return shift * t / (1.0f + (shift - 1.0f) * t);
}

float sd3_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return sd3_time_snr_shift(shift, t / 1000.0f);
}

ed_status_t tensor_to_image(const sd::Tensor<float>& tensor, ed_image_t* image) {
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

    const size_t pixels = width * height;
    const float* src = tensor.data();
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            float value = src[i + pixels * c];
            if (value <= 0.0f) {
                data[i * channels + c] = 0;
            } else if (value >= 1.0f) {
                data[i * channels + c] = 255;
            } else {
                data[i * channels + c] = static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
        }
    }
    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data = data;
    return ED_STATUS_OK;
}

}  // namespace

SD3Pipeline::SD3Pipeline(SDVersion version)
    : version_(version) {
}

bool SD3Pipeline::prepare(const ed_context_params_t&,
                          ModelRuntime& runtime,
                          const ModelLoader& loader,
                          PipelineTensorRegistry& registry,
                          std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    registry.clear();

    if (version_ != VERSION_SD3) {
        if (error != nullptr) {
            *error = "SD3Pipeline got non-SD3 model version";
        }
        return false;
    }
    if (!build_components(loader, error)) {
        return false;
    }
    configure_runtime_flags();
    if (!register_tensors(registry, error)) {
        return false;
    }
    build_ignore_tensors(registry);
    return true;
}

bool SD3Pipeline::build_components(const ModelLoader& loader, std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        if (error != nullptr) {
            *error = "SD3Pipeline requires initialized ModelRuntime backends";
        }
        return false;
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool offload = runtime_->offload_params_to_cpu();
    conditioner_ = std::make_shared<SD3CLIPEmbedder>(runtime_->clip_backend(), offload, storage);
    diffusion_ = std::make_shared<MMDiTModel>(runtime_->backend(), offload, storage);
    vae_ = std::make_shared<AutoEncoderKL>(runtime_->vae_backend(),
                                           offload,
                                           storage,
                                           "first_stage_model",
                                           true,
                                           false,
                                           version_);
    return conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

void SD3Pipeline::configure_runtime_flags() {
    const size_t max_graph_vram = runtime_->max_graph_vram_bytes();

    conditioner_->set_max_graph_vram_bytes(max_graph_vram);
    conditioner_->set_flash_attention_enabled(runtime_->flash_attention());

    diffusion_->set_max_graph_vram_bytes(max_graph_vram);
    diffusion_->set_flash_attention_enabled(runtime_->flash_attention());
    diffusion_->set_circular_axes(runtime_->circular_x(), runtime_->circular_y());

    if (runtime_ != nullptr) {
        auto process_group = runtime_->graph_process_group_ref();
        if (process_group != nullptr) {
            diffusion_->set_process_group(process_group);
            LOG_INFO("sd3 diffusion process group attached: backend=%s rank=%d world_size=%d",
                     edgedit::parallel::backend_name(process_group->backend()),
                     process_group->rank(),
                     process_group->size());
        }
    }

    vae_->set_max_graph_vram_bytes(max_graph_vram);
    vae_->set_flash_attention_enabled(runtime_->flash_attention());
}

bool SD3Pipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (!conditioner_ || !diffusion_ || !vae_) {
        if (error != nullptr) {
            *error = "SD3Pipeline components are not initialized";
        }
        return false;
    }

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());

    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors());

    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), "first_stage_model");

    LOG_INFO("sd3 pipeline registered %zu tensors", registry.tensors().size());
    return true;
}

void SD3Pipeline::build_ignore_tensors(PipelineTensorRegistry& registry) const {
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");
    registry.ignore_prefix("first_stage_model.encoder");
    registry.ignore_prefix("first_stage_model.conv1");
    registry.ignore_prefix("first_stage_model.quant");
}

void SD3Pipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr;
}

ed_status_t SD3Pipeline::generate_image(const ed_image_generation_params_t* params,
                                        ed_image_batch_t* out,
                                        std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "SD3Pipeline is not initialized";
        }
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "image output is null";
        }
        return ED_STATUS_INVALID_ARGUMENT;
    }

    if (!validate_image_params(params, error)) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "current SD3 pipeline needs transformer, text encoders, and VAE weights";
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
        if (!generate_one_image(params, i, &images[i], error)) {
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

ed_status_t SD3Pipeline::generate_video(const ed_video_generation_params_t*,
                                        ed_video_t* out,
                                        std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (error != nullptr) {
        *error = "video generation is not supported by SD3Pipeline";
    }
    return ED_STATUS_UNSUPPORTED;
}

ed_scheduler_t SD3Pipeline::default_scheduler(ed_sampler_t method) const {
    if (method == ED_SAMPLER_LCM || method == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (method == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

bool SD3Pipeline::can_generate_image() const {
    return ready_ && runtime_ != nullptr && conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

bool SD3Pipeline::validate_image_params(const ed_image_generation_params_t* params,
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
    if (params->batch_count <= 0) {
        if (error != nullptr) {
            *error = "image batch_count must be positive";
        }
        return false;
    }
    return true;
}

std::vector<float> SD3Pipeline::build_sigmas(int steps, float shift) const {
    std::vector<float> sigmas;
    if (steps <= 0) {
        return sigmas;
    }
    if (steps == 1) {
        sigmas.push_back(sd3_t_to_sigma(999.0f, shift));
        sigmas.push_back(0.0f);
        return sigmas;
    }

    const float step = 999.0f / static_cast<float>(steps - 1);
    sigmas.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        sigmas.push_back(sd3_t_to_sigma(t, shift));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

bool SD3Pipeline::generate_one_image(const ed_image_generation_params_t* params,
                                     int batch_index,
                                     ed_image_t* image,
                                     std::string* error) {
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "SD3 pipeline is not ready for image generation";
        }
        return false;
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    if (params->width % vae_scale_factor != 0 || params->height % vae_scale_factor != 0) {
        if (error != nullptr) {
            *error = sd_format("SD3 image size must be divisible by VAE scale factor %d", vae_scale_factor);
        }
        return false;
    }

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    cond_params.clip_skip = -1;
    cond_params.width = params->width;
    cond_params.height = params->height;
    cond_params.adm_in_channels = static_cast<int>(diffusion_->get_adm_in_channels());
    SDCondition cond = conditioner_->get_learned_condition(runtime_->n_threads(), cond_params);
    if (cond.empty()) {
        if (error != nullptr) {
            *error = "SD3 prompt encoding returned empty condition";
        }
        return false;
    }
    SDCondition uncond;
    const float cfg_scale = params->sample.cfg_scale > 0.0f ? params->sample.cfg_scale : 1.0f;
    if (cfg_scale != 1.0f) {
        cond_params.text = params->negative_prompt != nullptr ? params->negative_prompt : "";
        uncond = conditioner_->get_learned_condition(runtime_->n_threads(), cond_params);
        if (uncond.empty()) {
            if (error != nullptr) {
                *error = "SD3 negative prompt encoding returned empty condition";
            }
            return false;
        }
    }

    const int latent_w = params->width / vae_scale_factor;
    const int latent_h = params->height / vae_scale_factor;
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    const ed_sampler_t sampler = params->sample.sampler == ED_SAMPLER_AUTO
                                     ? default_sample_method()
                                     : params->sample.sampler;
    if (sampler != ED_SAMPLER_EULER) {
        if (error != nullptr) {
            *error = "SD3Pipeline currently implements the old default Euler flow path only";
        }
        return false;
    }
    float flow_shift = params->sample.flow_shift;
    if (!(flow_shift > 0.0f) || !std::isfinite(flow_shift)) {
        flow_shift = 3.0f;
    }
    int64_t seed = params->seed;
    if (seed < 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seed = std::rand();
    }
    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "SD3Pipeline has no RNG from ModelRuntime";
        }
        return false;
    }
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    std::vector<float> sigmas = build_sigmas(steps, flow_shift);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create SD3 sigma schedule";
        }
        return false;
    }

    LOG_INFO("sd3 txt2img: %dx%d latent=%dx%d steps=%d shift=%.2f cfg=%.2f seed=%" PRId64,
             params->width,
             params->height,
             latent_w,
             latent_h,
             steps,
             flow_shift,
             cfg_scale,
             seed + batch_index);

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    cache::CacheRuntime cache_runtime;
    const auto* cache_parallel_context = runtime_->parallel_context();
    const bool cache_use_sequence_parallel = cache_parallel_context != nullptr &&
                                             cache_parallel_context->sp_parallel_size() > 1;
    const bool cache_use_cfg_parallel = !uncond.empty() &&
                                        !cache_use_sequence_parallel &&
                                        parallel::cfg_parallel_available(cache_parallel_context);
    const bool cache_seam_available =
        !cache_use_cfg_parallel && diffusion_->supports_feature_cache();
    // Wire the device store whenever the block-stack seam is usable: the on-GPU
    // cache path (MagCache feature reuse + DiCache rings) is the default. Host hooks
    // stay wired below so TaylorSeer/SenCache (Feature methods with no device path)
    // still reach the host capture/inject path.
    cache::ICacheDeviceStore* cache_store =
        cache_seam_available
            ? diffusion_->cache_device_store()
            : nullptr;
    LOG_INFO("sd3 cache device path: store=%s",
             cache_store != nullptr ? "on" : "off");
    const bool cache_enabled =
        cache_runtime.init(params->sample, version_, sigmas, cache_seam_available, cache_store,
                           cache_use_cfg_parallel);
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

        cache::CacheStepInfo cache_step;
        cache_step.step_index = step;
        cache_step.num_steps = steps;
        cache_step.sigma = sigma;
        cache_step.sigma_next = sigma_next;
        if (cache_enabled) {
            cache_runtime.begin_step(cache_step);
        }

        const auto* parallel_context = runtime_->parallel_context();
        const bool use_sequence_parallel = parallel_context != nullptr &&
                                           parallel_context->sp_parallel_size() > 1;
        const bool use_cfg_parallel = !uncond.empty() &&
                                      !use_sequence_parallel &&
                                      parallel::cfg_parallel_available(parallel_context);
        const int cfg_rank = parallel::cfg_parallel_rank(parallel_context);

        DiffusionParams diffusion_params;
        diffusion_params.x = &x;
        diffusion_params.timesteps = &timesteps;

        const int n_threads = runtime_->n_threads();
        // Build cache hooks for one condition. Feature/Probe seam is gated to
        // the plain path and disabled under CFG-parallel (see flux_pipeline).
        auto make_hooks = [&](const SDCondition& cond_in) {
            cache::CacheRunnerHooks hooks;
            hooks.input = &x;
            hooks.full = [&, cond_in]() {
                DiffusionParams p = diffusion_params;
                p.context = &cond_in.c_crossattn;
                p.y = &cond_in.c_vector;
                return diffusion_->compute(n_threads, p);
            };
            const bool seam_ok = !use_cfg_parallel && diffusion_->supports_feature_cache();
            if (seam_ok) {
                // Substep-path tap-driven host capture: residual
                // via ModelIn/ModelOut taps, read back to host. No CacheGraphScope.
                hooks.substep_capture_host = [&, cond_in]() {
                    DiffusionParams p = diffusion_params;
                    p.context = &cond_in.c_crossattn;
                    p.y = &cond_in.c_vector;
                    return diffusion_->compute_substep_capture_host(n_threads, p);
                };
                // Substep-path tap-driven host inject.
                hooks.substep_inject_host = [&, cond_in](const sd::Tensor<float>& feat) {
                    DiffusionParams p = diffusion_params;
                    p.context = &cond_in.c_crossattn;
                    p.y = &cond_in.c_vector;
                    return diffusion_->compute_substep_inject_host(n_threads, p, feat, 0, -1);
                };
                if (cache_runtime.granularity() == cache::CacheGranularity::Probe) {
                    // Substep-path tap-driven host probe.
                    hooks.substep_probe_host = [&, cond_in](int depth) {
                        DiffusionParams p = diffusion_params;
                        p.context = &cond_in.c_crossattn;
                        p.y = &cond_in.c_vector;
                        return diffusion_->compute_substep_probe_host(n_threads, p, depth);
                    };
                    // On-GPU DiCache: probe metric + gamma-blend reuse + seed
                    // capture all run on-device — this is the only DiCache path now.
                    // The host DiCache hooks above (probe/capture/inject) remain wired
                    // so the cache engine can fall back if a device hook is ever unset,
                    // and so TaylorSeer/SenCache (Feature methods, no device path)
                    // still reach the host capture/inject path.
                    {
                        const void* branch_key = static_cast<const void*>(&cond_in);
                        const bool delta_minus = cache_runtime.dicache_delta_minus();
                        hooks.substep_probe = [&, cond_in, branch_key, delta_minus](
                                int depth, const cache::CacheOperatorRegistry& operators,
                                const cache::DiCacheSlotBridge& bridge) {
                            DiffusionParams p = diffusion_params;
                            p.context = &cond_in.c_crossattn;
                            p.y = &cond_in.c_vector;
                            return diffusion_->compute_substep_probe_gpu(n_threads, p, depth, branch_key,
                                                                         delta_minus, operators, bridge);
                        };
                        hooks.substep_inject_gpu = [&, cond_in](std::vector<cache::GraphExtension> exts,
                                                                const cache::DiCacheSlotBridge& bridge) {
                            DiffusionParams p = diffusion_params;
                            p.context = &cond_in.c_crossattn;
                            p.y = &cond_in.c_vector;
                            return diffusion_->compute_substep_inject_gpu(n_threads, p, std::move(exts), bridge);
                        };
                        const int probe_depth = cache_runtime.dicache_probe_depth();
                        hooks.substep_capture_probe = [&, cond_in, probe_depth](
                                const cache::DiCacheSlotBridge& bridge) {
                            DiffusionParams p = diffusion_params;
                            p.context = &cond_in.c_crossattn;
                            p.y = &cond_in.c_vector;
                            return diffusion_->compute_substep_capture_probe(n_threads, p, probe_depth, bridge);
                        };
                    }
                }
                // On-GPU MagCache device-slot path (Feature granularity only).
                // Dual-wired alongside the host hooks: the cache engine prefers the
                // device slot when the store is non-null and falls back to host
                // otherwise (and for TaylorSeer/SenCache, which have no device path).
                const bool feature_gpu =
                    cache_runtime.granularity() == cache::CacheGranularity::Feature;
                if (feature_gpu) {
                    hooks.substep_capture = [&, cond_in](std::vector<cache::GraphExtension> exts) {
                        DiffusionParams p = diffusion_params;
                        p.context = &cond_in.c_crossattn;
                        p.y = &cond_in.c_vector;
                        return diffusion_->compute_substep_capture_slot(n_threads, p, std::move(exts));
                    };
                    hooks.substep_inject_slot = [&, cond_in](std::vector<cache::GraphExtension> exts) {
                        DiffusionParams p = diffusion_params;
                        p.context = &cond_in.c_crossattn;
                        p.y = &cond_in.c_vector;
                        return diffusion_->compute_substep_inject_slot(n_threads, p, std::move(exts));
                    };
                }
            }
            return hooks;
        };

        sd::Tensor<float> cond_out;
        const void* cond_key = static_cast<const void*>(&cond);
        if (!use_cfg_parallel) {
            cond_out = cache_enabled
                ? cache_runtime.run_branch(cache::CacheBranch::Cond, cond_key, make_hooks(cond))
                : make_hooks(cond).full();
        }
        if (!use_cfg_parallel && cond_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("SD3 diffusion compute failed at step %d", step + 1);
            }
            diffusion_->free_compute_buffer();
            return false;
        }

        sd::Tensor<float> model_out = cond_out;
        if (!uncond.empty()) {
            sd::Tensor<float> uncond_out;
            const void* uncond_key = static_cast<const void*>(&uncond);
            if (use_cfg_parallel) {
                const bool local_is_uncond = cfg_rank == 0;
                const SDCondition& local_condition = local_is_uncond ? uncond : cond;
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
                        *error = sd_format("SD3 CFG parallel gather failed at step %d", step + 1);
                    }
                    diffusion_->free_compute_buffer();
                    return false;
                }
                uncond_out = std::move(gathered[0]);
                cond_out = std::move(gathered[1]);
            } else {
                uncond_out = cache_enabled
                    ? cache_runtime.run_branch(cache::CacheBranch::Uncond, uncond_key, make_hooks(uncond))
                    : make_hooks(uncond).full();
            }
            if (uncond_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("SD3 unconditional diffusion compute failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
            model_out = uncond_out + cfg_scale * (cond_out - uncond_out);
        }

        // SenCache calibration: finite-diff sensitivities on the combined
        // velocity. Two extra plain forwards/step, calibration only; off under
        // CFG-parallel. SD3 feeds timestep = sigma*1000, so convert here.
        if (cache_enabled && cache_runtime.needs_calibration() && !use_cfg_parallel) {
            auto forward_at = [&](const sd::Tensor<float>& x_raw, float sigma_eval) -> sd::Tensor<float> {
                sd::Tensor<float> ts({1}, std::vector<float>{sigma_eval * 1000.0f});
                DiffusionParams p = diffusion_params;
                p.x = &x_raw;
                p.timesteps = &ts;
                p.context = &cond.c_crossattn;
                p.y = &cond.c_vector;
                sd::Tensor<float> cond_v = diffusion_->compute(n_threads, p);
                if (cond_v.empty() || uncond.empty()) {
                    return cond_v;
                }
                p.context = &uncond.c_crossattn;
                p.y = &uncond.c_vector;
                sd::Tensor<float> uncond_v = diffusion_->compute(n_threads, p);
                if (uncond_v.empty()) {
                    return {};
                }
                return uncond_v + cfg_scale * (cond_v - uncond_v);
            };
            cache_runtime.calibrate(cache::CacheBranch::Cond, cond_key, x, model_out, forward_at);
        }

        x += model_out * (sigma_next - sigma);
        LOG_INFO("sd3 step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
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
    LOG_INFO("sd3 sampling completed, taking %.2fs", (sample_end_ms - sample_start_ms) / 1000.0f);
    diffusion_->free_compute_buffer();

    if (runtime_->parallel_context() != nullptr && !runtime_->parallel_context()->is_root()) {
        return true;
    }

    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    sd::Tensor<float> decoded = vae_->decode(runtime_->n_threads(),
                                             vae_latents,
                                             runtime_->vae_tiling(),
                                             false,
                                             runtime_->circular_x(),
                                             runtime_->circular_y());
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "SD3 VAE decode failed";
        }
        return false;
    }

    const ed_status_t status = tensor_to_image(decoded, image);
    if (status != ED_STATUS_OK) {
        if (error != nullptr) {
            *error = status == ED_STATUS_OUT_OF_MEMORY ? "failed to allocate decoded image" : "decoded SD3 tensor has invalid shape";
        }
        return false;
    }
    return true;
}

}  // namespace edgedit
