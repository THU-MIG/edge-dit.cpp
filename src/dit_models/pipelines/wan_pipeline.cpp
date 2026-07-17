#include "dit_models/pipelines/wan_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <limits>
#include <utility>
#include "parallel/process_group.hpp"
#include "backend/ggml/ggml_extend.hpp"
#include "core/optimization/cache/runtime/cache_engine.hpp"
#ifdef ED_ENABLE_CUDNN_SDPA
#include "backend/cuDNN/ed_cudnn_sdpa.h"
#endif
#ifdef ED_ENABLE_NCCL
#include <cuda_runtime.h>
#endif
#include "dit_models/components/scheduler/denoiser.hpp"
#include "dit_models/components/autoencoders/tae.hpp"
#include "dit_models/components/autoencoders/vae.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "parallel/cfg_parallel.hpp"
#include "utils/rng.hpp"
#include "utils/rng_philox.hpp"
#include "utils/util.h"

// This implementation is intentionally Wan-only.  It keeps the old repository's
// proven inference order but removes the giant EdgeDitGGML coupling:
//   T5 condition -> Flow denoise with WanModel -> WanVAE video decode.
//
// Stage-1 support:
//   - Wan T2V
//   - Flow/Euler sampler
//   - CFG
//   - optional high-noise WanModel split, when high_noise_sample.steps > 0
//   - WanVAE / TinyVideoAutoEncoder decode
//
// Explicitly deferred:
//   - I2V/FLF2V c_concat construction
//   - VACE control-frame context construction
//   - LoRA/cache/preview callbacks
//
// The components are still registered and loadable for I2V/VACE, but generation
// returns ED_STATUS_UNSUPPORTED until the corresponding condition tensors are wired.

namespace edgedit {
namespace {

constexpr int kWanDefaultSteps = 20;
constexpr float kWanDefaultFlowShift = 5.0f;

static const char* safe_cstr(const char* p) {
    return p != nullptr ? p : "";
}

static bool is_blank(const char* p) {
    return p == nullptr || p[0] == '\0';
}

static bool wants_uncond(const ed_video_generation_params_t* params) {
    if (params == nullptr) {
        return false;
    }
    if (!is_blank(params->negative_prompt)) {
        return true;
    }
    return params->sample.cfg_scale != 0.0f && params->sample.cfg_scale != 1.0f;
}

#ifdef ED_ENABLE_NCCL
static bool wan_check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        LOG_DEBUG("wan SP comm prewarm skipped: %s failed: %s", expr, cudaGetErrorString(status));
        return false;
    }
    return true;
}
#endif

static scheduler_t to_internal_scheduler(ed_scheduler_t scheduler) {
    switch (scheduler) {
        case ED_SCHEDULER_KARRAS:
            return KARRAS_SCHEDULER;
        case ED_SCHEDULER_EXPONENTIAL:
            return EXPONENTIAL_SCHEDULER;
        case ED_SCHEDULER_AYS:
            return AYS_SCHEDULER;
        case ED_SCHEDULER_GITS:
            return GITS_SCHEDULER;
        case ED_SCHEDULER_SGM_UNIFORM:
            return SGM_UNIFORM_SCHEDULER;
        case ED_SCHEDULER_SIMPLE:
            return SIMPLE_SCHEDULER;
        case ED_SCHEDULER_SMOOTHSTEP:
            return SMOOTHSTEP_SCHEDULER;
        case ED_SCHEDULER_KL_OPTIMAL:
            return KL_OPTIMAL_SCHEDULER;
        case ED_SCHEDULER_LCM:
            return LCM_SCHEDULER;
        case ED_SCHEDULER_BONG_TANGENT:
            return BONG_TANGENT_SCHEDULER;
        case ED_SCHEDULER_AUTO:
        case ED_SCHEDULER_DISCRETE:
        default:
            return DISCRETE_SCHEDULER;
    }
}

static float resolve_eta(const ed_sample_params_t& params) {
    if (std::isfinite(params.eta)) {
        return params.eta;
    }
    switch (params.sampler) {
        case ED_SAMPLER_EULER_A:
        case ED_SAMPLER_DPM_PLUS_PLUS_2S_A:
        case ED_SAMPLER_ER_SDE:
            return 1.0f;
        default:
            return 0.0f;
    }
}

static int resolve_steps(int steps) {
    return steps > 0 ? steps : kWanDefaultSteps;
}

static float resolve_cfg(float cfg) {
    return cfg == 0.0f ? 1.0f : cfg;
}

static ed_scheduler_t resolve_scheduler(ed_scheduler_t scheduler, ed_sampler_t sampler) {
    if (scheduler != ED_SCHEDULER_AUTO) {
        return scheduler;
    }
    if (sampler == ED_SAMPLER_LCM || sampler == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (sampler == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

static uint8_t clamp_float_to_u8(float value) {
    value = std::max(0.0f, std::min(1.0f, value));
    return static_cast<uint8_t>(value * 255.0f);
}

static ed_image_t tensor_frame_to_image(const sd::Tensor<float>& tensor, int frame_index) {
    ed_image_t image{};
    if (tensor.empty() || frame_index < 0) {
        return image;
    }

    const auto& shape = tensor.shape();
    size_t width = 0;
    size_t height = 0;
    size_t channels = 0;
    size_t frame_count = 0;

    if (shape.size() == 5) {
        width = static_cast<size_t>(shape[0]);
        height = static_cast<size_t>(shape[1]);
        frame_count = static_cast<size_t>(shape[2]);
        channels = static_cast<size_t>(shape[3]);
    } else if (shape.size() == 4) {
        width = static_cast<size_t>(shape[0]);
        height = static_cast<size_t>(shape[1]);
        channels = static_cast<size_t>(shape[2]);
        frame_count = static_cast<size_t>(shape[3]);
    } else {
        return image;
    }

    if (width == 0 || height == 0 || channels == 0 ||
        static_cast<size_t>(frame_index) >= frame_count) {
        return image;
    }

    const size_t pixels = width * height;
    const size_t image_bytes = pixels * channels;
    uint8_t* data = static_cast<uint8_t*>(std::malloc(image_bytes));
    if (data == nullptr) {
        return image;
    }

    const float* src = tensor.data();
    if (shape.size() == 5) {
        const size_t frame_offset = static_cast<size_t>(frame_index) * pixels;
        const size_t channel_stride = pixels * frame_count;
        for (size_t i = 0; i < pixels; ++i) {
            for (size_t c = 0; c < channels; ++c) {
                data[i * channels + c] = clamp_float_to_u8(src[frame_offset + c * channel_stride + i]);
            }
        }
    } else {
        const size_t frame_stride = pixels * channels;
        const size_t frame_offset = static_cast<size_t>(frame_index) * frame_stride;
        for (size_t i = 0; i < pixels; ++i) {
            for (size_t c = 0; c < channels; ++c) {
                data[i * channels + c] = clamp_float_to_u8(src[frame_offset + c * pixels + i]);
            }
        }
    }

    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.channels = static_cast<uint32_t>(channels);
    image.data = data;
    return image;
}

}  // namespace

WanPipeline::WanPipeline(SDVersion version)
    : version_(version) {
}

bool WanPipeline::set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
    return false;
}

int64_t WanPipeline::resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    return static_cast<int64_t>(std::rand());
}

bool WanPipeline::has_prefix(const ModelLoader& loader, const std::string& prefix) {
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (starts_with(item.first, prefix)) {
            return true;
        }
    }
    return false;
}

bool WanPipeline::prepare(const ed_context_params_t&,
                          ModelRuntime& runtime,
                          const ModelLoader& loader,
                          PipelineTensorRegistry& registry,
                          std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    registry.clear();

    if (!ed_version_is_wan(version_)) {
        return set_error(error, "WanPipeline got non-Wan model version");
    }

    if (!build_components(loader, error)) {
        return false;
    }
    if (!init_sampling_runtime(error)) {
        return false;
    }

    configure_runtime_flags();

    if (!register_tensors(registry, error)) {
        return false;
    }
    build_ignore_tensors(registry);

    return true;
}

bool WanPipeline::build_components(const ModelLoader& loader, std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        return set_error(error, "WanPipeline requires initialized ModelRuntime backends");
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool offload = runtime_->offload_params_to_cpu();
    const bool vae_decode_only = true;

    conditioner_ = std::make_shared<T5CLIPEmbedder>(runtime_->clip_backend(),
                                                    offload,
                                                    storage,
                                                    true,
                                                    0,
                                                    true);

    diffusion_ = std::make_shared<WanModel>(runtime_->backend(),
                                            offload,
                                            storage,
                                            "model.diffusion_model",
                                            version_);

    if (has_prefix(loader, "model.high_noise_diffusion_model.")) {
        high_noise_diffusion_ = std::make_shared<WanModel>(runtime_->backend(),
                                                           offload,
                                                           storage,
                                                           "model.high_noise_diffusion_model",
                                                           version_);
    }

    if (diffusion_->get_desc() == "Wan2.1-I2V-14B" ||
        diffusion_->get_desc() == "Wan2.1-FLF2V-14B" ||
        diffusion_->get_desc() == "Wan2.1-I2V-1.3B") {
        clip_vision_ = std::make_shared<FrozenCLIPVisionEmbedder>(runtime_->backend(),
                                                                  offload,
                                                                  storage);
    }

    using_tae_for_main_ = loader.use_tae() && !loader.tae_preview_only();
    if (using_tae_for_main_) {
        vae_ = std::make_shared<TinyVideoAutoEncoder>(runtime_->vae_backend(),
                                                      offload,
                                                      storage,
                                                      "decoder",
                                                      vae_decode_only,
                                                      version_);
    } else {
        vae_ = std::make_shared<WAN::WanVAERunner>(runtime_->vae_backend(),
                                                   offload,
                                                   storage,
                                                   "first_stage_model",
                                                   vae_decode_only,
                                                   version_);
    }

    if (loader.use_tae() && loader.tae_preview_only()) {
        preview_vae_ = std::make_shared<TinyVideoAutoEncoder>(runtime_->vae_backend(),
                                                              offload,
                                                              storage,
                                                              "decoder",
                                                              true,
                                                              version_);
    }

    if (!conditioner_ || !diffusion_ || !vae_) {
        return set_error(error, "WanPipeline failed to construct required components");
    }
    return true;
}

bool WanPipeline::init_sampling_runtime(std::string* error) {
    (void)error;
    denoiser_ = std::make_shared<DiscreteFlowDenoiser>();
    rng_ = std::make_shared<PhiloxRNG>();
    sampler_rng_ = rng_;
    set_flow_shift(std::numeric_limits<float>::infinity());
    return true;
}

void WanPipeline::configure_runtime_flags() {
    const size_t max_graph_vram = runtime_->max_graph_vram_bytes();
    const bool text_flash = runtime_->flash_attention();
    const bool diffusion_flash = runtime_->flash_attention();

    conditioner_->set_max_graph_vram_bytes(max_graph_vram);
    conditioner_->set_flash_attention_enabled(text_flash);

    if (clip_vision_) {
        clip_vision_->set_max_graph_vram_bytes(max_graph_vram);
        clip_vision_->set_flash_attention_enabled(text_flash);
    }

    diffusion_->set_max_graph_vram_bytes(max_graph_vram);
    diffusion_->set_flash_attention_enabled(diffusion_flash);
    diffusion_->set_circular_axes(runtime_->circular_x(), runtime_->circular_y());

    if (high_noise_diffusion_) {
        high_noise_diffusion_->set_max_graph_vram_bytes(max_graph_vram);
        high_noise_diffusion_->set_flash_attention_enabled(diffusion_flash);
        high_noise_diffusion_->set_circular_axes(runtime_->circular_x(), runtime_->circular_y());
    }

    if (runtime_ != nullptr) {
        auto process_group = runtime_->graph_process_group_ref();
        if (process_group != nullptr) {
            diffusion_->set_process_group(process_group);
            LOG_INFO("wan diffusion process group attached: backend=%s rank=%d world_size=%d",
                     edgedit::parallel::backend_name(process_group->backend()),
                     process_group->rank(),
                     process_group->size());

            if (high_noise_diffusion_) {
                high_noise_diffusion_->set_process_group(process_group);
                LOG_INFO("wan high-noise diffusion process group attached: backend=%s rank=%d world_size=%d",
                         edgedit::parallel::backend_name(process_group->backend()),
                         process_group->rank(),
                         process_group->size());
            }
        }
    }

    vae_->set_max_graph_vram_bytes(max_graph_vram);
    vae_->set_flash_attention_enabled(text_flash);

    if (preview_vae_) {
        preview_vae_->set_max_graph_vram_bytes(max_graph_vram);
        preview_vae_->set_flash_attention_enabled(text_flash);
    }
}

bool WanPipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (!conditioner_ || !diffusion_ || !vae_) {
        return set_error(error, "WanPipeline components are not initialized");
    }

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());

    if (clip_vision_) {
        clip_vision_->alloc_params_buffer();
        clip_vision_->get_param_tensors(registry.tensors());
    }

    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors());

    if (high_noise_diffusion_) {
        high_noise_diffusion_->alloc_params_buffer();
        high_noise_diffusion_->get_param_tensors(registry.tensors());
    }

    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), using_tae_for_main_ ? "tae" : "first_stage_model");

    if (preview_vae_) {
        preview_vae_->alloc_params_buffer();
        preview_vae_->get_param_tensors(registry.tensors(), "tae");
    }

    LOG_INFO("wan pipeline registered %zu tensors", registry.tensors().size());
    return true;
}

void WanPipeline::build_ignore_tensors(PipelineTensorRegistry& registry) const {
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");

    if (using_tae_for_main_) {
        registry.ignore_prefix("first_stage_model.");
    } else {
        registry.ignore_prefix("first_stage_model.encoder");
        registry.ignore_prefix("first_stage_model.conv1");
        registry.ignore_prefix("first_stage_model.quant");
        registry.ignore_prefix("tae.encoder");
    }

    // This pipeline is decode-only at stage 1, so do not force-load visual
    // encoder branches unless clip_vision_ is explicitly constructed.
    if (!clip_vision_) {
        registry.ignore_prefix("text_encoders.llm.visual.");
    }
}

void WanPipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr &&
             denoiser_ != nullptr &&
             rng_ != nullptr &&
             sampler_rng_ != nullptr;
}

void WanPipeline::set_flow_shift(float flow_shift) {
    auto flow = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser_);
    if (!flow) {
        return;
    }
    if (!std::isfinite(flow_shift) || flow_shift <= 0.0f) {
        flow_shift = default_flow_shift_;
    }
    flow->set_shift(flow_shift);
}

bool WanPipeline::is_flow_denoiser() const {
    return !!std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser_);
}

int WanPipeline::vae_scale_factor() const {
    return vae_ ? vae_->get_scale_factor() : 8;
}

int WanPipeline::latent_channels() const {
    if (version_ == VERSION_WAN2_2_TI2V) {
        return 48;
    }
    return 16;
}

int WanPipeline::latent_frames(int frames) const {
    if (frames <= 0) {
        return 0;
    }
    return ((frames - 1) / 4) + 1;
}

int WanPipeline::image_seq_len(int width, int height) const {
    const int scale = vae_scale_factor();
    return (height / scale) * (width / scale);
}

void WanPipeline::prewarm_wan_cudnn_sdpa(int width, int height, int frames) const {
#ifdef ED_ENABLE_CUDNN_SDPA
    if (runtime_ == nullptr ||
        !runtime_->flash_attention() ||
        !WAN::wan_env_flag_enabled_or_default("ED_WAN_CUDNN_SDPA_PREWARM", true)) {
        return;
    }

    auto* parallel_context = runtime_->parallel_context();
    const int world_size = parallel_context != nullptr ?
                               std::max(1, parallel_context->sp_parallel_size()) :
                               1;
    if (world_size > 1 && !WAN::wan_env_flag_enabled_or_default("ED_WAN_SP_CUDNN_SDPA_PREWARM", true)) {
        return;
    }

    const auto* wan_model = dynamic_cast<const WanModel*>(diffusion_.get());
    if (wan_model == nullptr) {
        return;
    }

    const std::string desc = diffusion_ != nullptr ? diffusion_->get_desc() : std::string();
    const int64_t num_heads = wan_model->num_heads();
    const int64_t head_dim = wan_model->head_dim();
    if (num_heads <= 0 || head_dim <= 0 || num_heads % world_size != 0) {
        return;
    }

    const int scale = vae_scale_factor();
    const int64_t latent_w = width / scale;
    const int64_t latent_h = height / scale;
    const int64_t latent_t = latent_frames(frames);
    if (latent_w <= 0 || latent_h <= 0 || latent_t <= 0) {
        return;
    }

    constexpr int64_t patch_t = 1;
    constexpr int64_t patch_h = 2;
    constexpr int64_t patch_w = 2;
    const int64_t pad_t = (patch_t - latent_t % patch_t) % patch_t;
    const int64_t pad_h = (patch_h - latent_h % patch_h) % patch_h;
    const int64_t pad_w = (patch_w - latent_w % patch_w) % patch_w;
    const int64_t seq = ((latent_t + pad_t + (patch_t / 2)) / patch_t) *
                        ((latent_h + pad_h + (patch_h / 2)) / patch_h) *
                        ((latent_w + pad_w + (patch_w / 2)) / patch_w);
    if (head_dim != 128) {
        return;
    }

    const int64_t shard_heads = num_heads / world_size;
    int device = 0;
    if (parallel_context != nullptr) {
        device = parallel_context->device();
    }
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const ed_cudnn_sdpa_result_t result = ed_cudnn_sdpa_prewarm_self_attn(device,
                                                                          GGML_TYPE_F32,
                                                                          head_dim,
                                                                          shard_heads,
                                                                          seq,
                                                                          attn_scale,
                                                                          true);
    if (result != ED_CUDNN_SDPA_SUCCESS && result != ED_CUDNN_SDPA_BUILD_PENDING) {
        LOG_DEBUG("wan cuDNN SDPA prewarm skipped: desc=%s device=%d sp=%d h=%" PRId64 " seq=%" PRId64 " result=%s",
                  desc.c_str(),
                  device,
                  world_size,
                  shard_heads,
                  seq,
                  ed_cudnn_sdpa_result_name(result));
    } else {
        LOG_DEBUG("wan cuDNN SDPA prewarm started: desc=%s device=%d sp=%d h=%" PRId64 " seq=%" PRId64 " result=%s",
                  desc.c_str(),
                  device,
                  world_size,
                  shard_heads,
                  seq,
                  ed_cudnn_sdpa_result_name(result));
    }

    const int64_t local_seq = (seq + world_size - 1) / world_size;
    const ed_cudnn_sdpa_result_t cross_result = ed_cudnn_sdpa_prewarm_cross_attn(device,
                                                                                 GGML_TYPE_F32,
                                                                                 head_dim,
                                                                                 num_heads,
                                                                                 local_seq,
                                                                                 512,
                                                                                 attn_scale,
                                                                                 true);
    if (cross_result != ED_CUDNN_SDPA_SUCCESS && cross_result != ED_CUDNN_SDPA_BUILD_PENDING) {
        LOG_DEBUG("wan cuDNN SDPA cross prewarm skipped: desc=%s device=%d sp=%d h=%" PRId64 " sq=%" PRId64 " sk=512 result=%s",
                  desc.c_str(),
                  device,
                  world_size,
                  num_heads,
                  local_seq,
                  ed_cudnn_sdpa_result_name(cross_result));
    } else {
        LOG_DEBUG("wan cuDNN SDPA cross prewarm started: desc=%s device=%d sp=%d h=%" PRId64 " sq=%" PRId64 " sk=512 result=%s",
                  desc.c_str(),
                  device,
                  world_size,
                  num_heads,
                  local_seq,
                  ed_cudnn_sdpa_result_name(cross_result));
    }
#else
    (void)width;
    (void)height;
    (void)frames;
#endif
}

void WanPipeline::prewarm_wan_sp_comm(int width, int height, int frames) const {
    if (runtime_ == nullptr ||
        !WAN::wan_env_flag_enabled_or_default("ED_WAN_SP_COMM_PREWARM", true)) {
        return;
    }

    auto* parallel_context = runtime_->parallel_context();
    if (parallel_context == nullptr ||
        !parallel_context->enabled() ||
        parallel_context->backend() != edgedit::parallel::Backend::kNccl) {
        return;
    }

    const int world_size = std::max(1, parallel_context->sp_parallel_size());
    if (world_size <= 1) {
        return;
    }

    const auto* wan_model = dynamic_cast<const WanModel*>(diffusion_.get());
    if (wan_model == nullptr) {
        return;
    }

    const int64_t num_heads = wan_model->num_heads();
    const int64_t head_dim = wan_model->head_dim();
    if (num_heads <= 0 || head_dim <= 0 || num_heads % world_size != 0) {
        return;
    }

    const int scale = vae_scale_factor();
    const int64_t latent_w = width / scale;
    const int64_t latent_h = height / scale;
    const int64_t latent_t = latent_frames(frames);
    if (latent_w <= 0 || latent_h <= 0 || latent_t <= 0) {
        return;
    }

    constexpr int64_t patch_t = 1;
    constexpr int64_t patch_h = 2;
    constexpr int64_t patch_w = 2;
    const int64_t pad_t = (patch_t - latent_t % patch_t) % patch_t;
    const int64_t pad_h = (patch_h - latent_h % patch_h) % patch_h;
    const int64_t pad_w = (patch_w - latent_w % patch_w) % patch_w;
    const int64_t seq = ((latent_t + pad_t + (patch_t / 2)) / patch_t) *
                        ((latent_h + pad_h + (patch_h / 2)) / patch_h) *
                        ((latent_w + pad_w + (patch_w / 2)) / patch_w);
    if (seq <= 0) {
        return;
    }

    const int64_t padded_seq = ((seq + world_size - 1) / world_size) * world_size;
    const int64_t shard_seq = padded_seq / world_size;
    const int64_t shard_heads = num_heads / world_size;
    const int64_t x_pad = padded_seq - seq;
    const bool roped_half_qkv = x_pad == 0 && runtime_->flash_attention();
    const bool roped_all_half_qkv = roped_half_qkv &&
                                    WAN::wan_sp_f16_q_enabled() &&
                                    head_dim == 128;

    size_t qkv_count_per_peer = 0;
    edgedit::parallel::DataType qkv_dtype = edgedit::parallel::DataType::kFloat32;
    if (roped_all_half_qkv) {
        // Wan's roped-all-half qkv path packs two fp16 lanes into one 32-bit
        // element, so NCCL sees a float32-sized payload.
        qkv_count_per_peer = static_cast<size_t>((head_dim * 3 / 2) * shard_heads * shard_seq);
    } else if (roped_half_qkv) {
        qkv_count_per_peer = static_cast<size_t>(head_dim * 2 * shard_heads * shard_seq);
    } else {
        qkv_count_per_peer = static_cast<size_t>(head_dim * 3 * shard_heads * shard_seq);
    }

    const bool f16_head_to_seq = WAN::wan_sp_f16_head_to_seq_enabled();
    const size_t head_count_per_peer = static_cast<size_t>(head_dim * shard_heads * shard_seq);
    const edgedit::parallel::DataType head_dtype = f16_head_to_seq ?
                                                       edgedit::parallel::DataType::kFloat16 :
                                                       edgedit::parallel::DataType::kFloat32;
    const size_t total_qkv_count = qkv_count_per_peer * static_cast<size_t>(world_size);
    const size_t total_head_count = head_count_per_peer * static_cast<size_t>(world_size);
    if (total_qkv_count == 0 || total_head_count == 0) {
        return;
    }

#ifdef ED_ENABLE_NCCL
    const int device = parallel_context->device();
    void* qkv_in = nullptr;
    void* qkv_out = nullptr;
    void* head_in = nullptr;
    void* head_out = nullptr;

    try {
        if (!wan_check_cuda(cudaSetDevice(device), "cudaSetDevice")) {
            return;
        }

        const size_t qkv_bytes = total_qkv_count * edgedit::parallel::dtype_size(qkv_dtype);
        const size_t head_bytes = total_head_count * edgedit::parallel::dtype_size(head_dtype);
        if (!wan_check_cuda(cudaMalloc(&qkv_in, qkv_bytes), "cudaMalloc qkv_in") ||
            !wan_check_cuda(cudaMalloc(&qkv_out, qkv_bytes), "cudaMalloc qkv_out") ||
            !wan_check_cuda(cudaMalloc(&head_in, head_bytes), "cudaMalloc head_in") ||
            !wan_check_cuda(cudaMalloc(&head_out, head_bytes), "cudaMalloc head_out")) {
            cudaFree(qkv_in);
            cudaFree(qkv_out);
            cudaFree(head_in);
            cudaFree(head_out);
            return;
        }

        if (!wan_check_cuda(cudaMemset(qkv_in, 0, qkv_bytes), "cudaMemset qkv_in") ||
            !wan_check_cuda(cudaMemset(head_in, 0, head_bytes), "cudaMemset head_in")) {
            cudaFree(qkv_in);
            cudaFree(qkv_out);
            cudaFree(head_in);
            cudaFree(head_out);
            return;
        }

        auto& group = parallel_context->world_group();
        auto qkv_work = group.all_to_all_async(edgedit::parallel::Buffer{qkv_in, total_qkv_count, qkv_dtype, device},
                                               edgedit::parallel::Buffer{qkv_out, total_qkv_count, qkv_dtype, device},
                                               qkv_count_per_peer);
        qkv_work->wait();
        auto head_work = group.all_to_all_async(edgedit::parallel::Buffer{head_in, total_head_count, head_dtype, device},
                                                edgedit::parallel::Buffer{head_out, total_head_count, head_dtype, device},
                                                head_count_per_peer);
        head_work->wait();
        group.barrier();

        LOG_DEBUG("wan SP comm prewarm completed: device=%d sp=%d seq=%" PRId64 " shard_seq=%" PRId64 " heads=%" PRId64 " qkv_count=%zu head_count=%zu head_dtype=%s qkv_mode=%s",
                  device,
                  world_size,
                  padded_seq,
                  shard_seq,
                  shard_heads,
                  qkv_count_per_peer,
                  head_count_per_peer,
                  edgedit::parallel::dtype_name(head_dtype),
                  roped_all_half_qkv ? "roped_all_half" : (roped_half_qkv ? "roped_half" : "plain"));
    } catch (const std::exception& e) {
        LOG_DEBUG("wan SP comm prewarm skipped: %s", e.what());
    }

    cudaFree(qkv_in);
    cudaFree(qkv_out);
    cudaFree(head_in);
    cudaFree(head_out);
#else
    (void)qkv_dtype;
    (void)head_dtype;
    (void)total_qkv_count;
    (void)total_head_count;
#endif
}

bool WanPipeline::validate_video_params(const ed_video_generation_params_t* params,
                                        std::string* error) const {
    if (params == nullptr) {
        return set_error(error, "video generation params are null");
    }
    if (params->width <= 0 || params->height <= 0 || params->frames <= 0) {
        return set_error(error, "video width, height, and frames must be positive");
    }
    if (params->width % vae_scale_factor() != 0 || params->height % vae_scale_factor() != 0) {
        return set_error(error, "video width and height must be divisible by VAE scale factor");
    }
    return true;
}

sd::Tensor<float> WanPipeline::generate_init_latent(int width, int height, int frames) const {
    const int scale = vae_scale_factor();
    const int W = width / scale;
    const int H = height / scale;
    const int T = latent_frames(frames);
    const int C = latent_channels();
    return sd::zeros<float>({W, H, T, C, 1});
}

bool WanPipeline::prepare_text_conditions(const ed_video_generation_params_t* params,
                                          const sd::Tensor<float>& concat_latent,
                                          const sd::Tensor<float>& clip_vision_output,
                                          WanVideoConditionPack* conditions,
                                          std::string* error) {
    if (conditions == nullptr) {
        return set_error(error, "internal error: null WanVideoConditionPack");
    }

    ConditionerParams condition_params;
    condition_params.clip_skip = -1;
    condition_params.text = safe_cstr(params->prompt);
    condition_params.zero_out_masked = true;
    condition_params.adm_in_channels = static_cast<int>(diffusion_->get_adm_in_channels());

    const int64_t start_ms = ggml_time_ms();
    conditions->cond = conditioner_->get_learned_condition(runtime_->n_threads(), condition_params);
    conditions->cond.c_concat = concat_latent;
    conditions->cond.c_vector = clip_vision_output;

    if (conditions->cond.c_crossattn.empty()) {
        return set_error(error, "Wan text condition is empty");
    }

    if (wants_uncond(params)) {
        condition_params.text = safe_cstr(params->negative_prompt);
        conditions->uncond = conditioner_->get_learned_condition(runtime_->n_threads(), condition_params);
        conditions->uncond.c_concat = concat_latent;
        conditions->uncond.c_vector = clip_vision_output;
    }

    const int64_t end_ms = ggml_time_ms();
    LOG_INFO("wan text condition completed, taking %.2fs", (end_ms - start_ms) / 1000.0f);
    return true;
}

std::vector<float> WanPipeline::build_sigmas(const ed_sample_params_t& params,
                                             int width,
                                             int height,
                                             int total_steps) const {
    const int steps = resolve_steps(total_steps);
    const ed_scheduler_t scheduler = resolve_scheduler(params.scheduler, params.sampler);
    return denoiser_->get_sigmas(steps,
                                 image_seq_len(width, height),
                                 to_internal_scheduler(scheduler),
                                 version_);
}

std::vector<float> WanPipeline::prepare_sample_timesteps(float sigma,
                                                         int shifted_timestep,
                                                         const sd::Tensor<float>& init_latent,
                                                         const sd::Tensor<float>& denoise_mask) const {
    float t = denoiser_->sigma_to_t(sigma);
    if (shifted_timestep > 0) {
        const float shifted_t_float = t * (static_cast<float>(shifted_timestep) / static_cast<float>(TIMESTEPS));
        int64_t shifted_t = static_cast<int64_t>(std::round(shifted_t_float));
        shifted_t = std::max<int64_t>(0, std::min<int64_t>(TIMESTEPS - 1, shifted_t));
        t = static_cast<float>(shifted_t);
    }

    if (diffusion_ && diffusion_->get_desc() == "Wan2.2-TI2V-5B") {
        std::vector<float> timesteps(static_cast<size_t>(init_latent.shape()[2]), t);
        if (!denoise_mask.empty()) {
            const float value = denoise_mask.dim() == 5 ?
                                denoise_mask.index(0, 0, 0, 0, 0) :
                                denoise_mask.index(0, 0, 0, 0);
            if (value == 0.0f) {
                timesteps[0] = 0.0f;
            }
        }
        return timesteps;
    }

    return {t};
}

void WanPipeline::adjust_sample_step_scalings(int shifted_timestep,
                                              const std::vector<float>& timesteps,
                                              float c_in,
                                              float* c_skip,
                                              float* c_out) const {
    if (shifted_timestep <= 0 || timesteps.empty() || c_skip == nullptr || c_out == nullptr) {
        return;
    }
    const int64_t shifted_t_idx = static_cast<int64_t>(std::round(timesteps[0]));
    const float shifted_sigma = denoiser_->t_to_sigma(static_cast<float>(shifted_t_idx));
    const auto shifted_scaling = denoiser_->get_scalings(shifted_sigma);
    if (shifted_scaling.size() != 3) {
        return;
    }
    const float shifted_c_skip = shifted_scaling[0];
    const float shifted_c_out = shifted_scaling[1];
    const float shifted_c_in = shifted_scaling[2];

    *c_skip = shifted_c_skip * c_in / shifted_c_in;
    *c_out = shifted_c_out;
}

sd::Tensor<float> WanPipeline::run_condition(const std::shared_ptr<DiffusionModel>& model,
                                             DiffusionParams* diffusion_params,
                                             const SDCondition& condition,
                                             const sd::Tensor<float>* c_concat_override,
                                             const std::vector<int>* skip_layers,
                                             std::string* error) {
    if (!model || diffusion_params == nullptr) {
        set_error(error, "internal error: invalid diffusion condition call");
        return {};
    }

    diffusion_params->context = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
    diffusion_params->c_concat = c_concat_override != nullptr ?
                                 c_concat_override :
                                 (condition.c_concat.empty() ? nullptr : &condition.c_concat);
    diffusion_params->y = condition.c_vector.empty() ? nullptr : &condition.c_vector;
    diffusion_params->t5_ids = condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids;
    diffusion_params->t5_weights = condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights;
    diffusion_params->skip_layers = skip_layers;

    sd::Tensor<float> output = model->compute(runtime_->n_threads(), *diffusion_params);
    if (output.empty()) {
        set_error(error, "Wan diffusion model compute failed");
    }
    return output;
}

sd::Tensor<float> WanPipeline::euler_denoise(const std::shared_ptr<DiffusionModel>& model,
                                             const sd::Tensor<float>& x_start,
                                             const std::vector<float>& sigmas,
                                             const WanVideoConditionPack& conditions,
                                             const ed_sample_params_t& sample_params,
                                             const sd::Tensor<float>& init_latent,
                                             const sd::Tensor<float>& denoise_mask,
                                             const sd::Tensor<float>& vace_context,
                                             float vace_strength,
                                             std::string* error) {
    if (!model) {
        set_error(error, "Wan diffusion model is null");
        return {};
    }
    if (sigmas.size() < 2) {
        set_error(error, "Wan sampler requires at least two sigmas");
        return {};
    }

    const float cfg_scale = resolve_cfg(sample_params.cfg_scale);
    const float img_cfg_scale = sample_params.image_cfg_scale == 0.0f ? cfg_scale : sample_params.image_cfg_scale;
    const float eta = resolve_eta(sample_params);
    (void)eta;  // Euler deterministic update does not use eta.
    constexpr int shifted_timestep = 0;

    // One cache controller per stage (per euler_denoise call) so high/low-noise
    // schedules never leak across the stage boundary. Feature/Probe caching is
    // additionally disabled whenever a VACE context is active (the model gates
    // its seam off, so hooks fall back to full compute).
    cache::CacheController cache_runtime;
    const bool cache_vace_ok = vace_context.empty();
    const bool cache_use_cfg_parallel = !conditions.uncond.empty() &&
                                        conditions.img_cond.empty() &&
                                        parallel::cfg_parallel_available(runtime_->parallel_context());
    // Run-level seam availability. Per-step hooks apply the stricter gate
    // (active model + c_concat), so this only needs the loop-invariant part.
    const bool cache_seam_available =
        !cache_use_cfg_parallel && cache_vace_ok && diffusion_->supports_feature_cache();
    // Wire the device store when the on-GPU MagCache path is active (opt-in via
    // ED_WAN_CACHE_GPU, default off — video residuals are large). A null store
    // keeps the lowering on the host path (host hooks stay wired below).
    cache::ICacheDeviceStore* cache_store =
        (cache_seam_available && WAN::WanRunner::feature_gpu_enabled())
            ? model->cache_device_store()
            : nullptr;
    LOG_INFO("wan cache device-slot path: %s", cache_store != nullptr ? "on (GPU)" : "off (host)");
    const bool cache_enabled =
        cache_runtime.init(sample_params, version_, sigmas, cache_seam_available, cache_store,
                           cache_use_cfg_parallel);

    sd::Tensor<float> x = x_start;
    GenerationControl* control = runtime_ != nullptr ? runtime_->generation_control() : nullptr;
    for (size_t i = 0; i + 1 < sigmas.size(); ++i) {
        if (control != nullptr && control->should_cancel()) {
            control->mark_cancelled();
            set_error(error, "generation cancelled");
            model->free_compute_buffer();
            return {};
        }
        const float sigma = sigmas[i];
        const float sigma_next = sigmas[i + 1];
        const int step = static_cast<int>(i + 1);

        std::vector<float> scaling = denoiser_->get_scalings(sigma);
        if (scaling.size() != 3) {
            set_error(error, "Denoiser returned invalid scaling tuple");
            return {};
        }

        float c_skip = scaling[0];
        float c_out = scaling[1];
        float c_in = scaling[2];

        std::vector<float> timesteps_vec = prepare_sample_timesteps(sigma,
                                                                    shifted_timestep,
                                                                    init_latent,
                                                                    denoise_mask);
        adjust_sample_step_scalings(shifted_timestep,
                                    timesteps_vec,
                                    c_in,
                                    &c_skip,
                                    &c_out);

        sd::Tensor<float> timesteps({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
        sd::Tensor<float> guidance({1}, std::vector<float>{sample_params.distilled_guidance});

        sd::Tensor<float> noised_input = x * c_in;
        if (!denoise_mask.empty() && version_ == VERSION_WAN2_2_TI2V) {
            noised_input = noised_input * denoise_mask + init_latent * (1.0f - denoise_mask);
        }

        DiffusionParams diffusion_params;
        diffusion_params.x = &noised_input;
        diffusion_params.timesteps = &timesteps;
        diffusion_params.guidance = &guidance;
        diffusion_params.vace_context = vace_context.empty() ? nullptr : &vace_context;
        diffusion_params.vace_strength = vace_strength;

        const bool use_cfg_parallel = !conditions.uncond.empty() &&
                                      conditions.img_cond.empty() &&
                                      parallel::cfg_parallel_available(runtime_->parallel_context());
        const int cfg_rank = parallel::cfg_parallel_rank(runtime_->parallel_context());

        if (cache_enabled) {
            cache::CacheStepInfo cache_step;
            cache_step.step_index = static_cast<int>(i);
            cache_step.num_steps = static_cast<int>(sigmas.size() - 1);
            cache_step.sigma = sigma;
            cache_step.sigma_next = sigma_next;
            cache_runtime.begin_step(cache_step);
        }

        // Cache hooks for one Wan condition. full() preserves the exact
        // run_condition semantics (c_concat / img-cond overrides); the seam is
        // used only for the plain cond/uncond streams, off under CFG-parallel
        // and VACE.
        auto make_hooks = [&](const SDCondition& cond_in,
                              const sd::Tensor<float>* c_concat_override) {
            cache::CacheRunnerHooks hooks;
            hooks.input = &noised_input;
            hooks.full = [&, cond_in, c_concat_override]() {
                return run_condition(model, &diffusion_params, cond_in, c_concat_override, nullptr, error);
            };
            const bool seam_ok = !use_cfg_parallel && cache_vace_ok &&
                                 c_concat_override == nullptr &&
                                 model->supports_feature_cache() &&
                                 model->feature_cache_available();
            if (seam_ok) {
                auto set_params = [&, cond_in]() {
                    diffusion_params.context = cond_in.c_crossattn.empty() ? nullptr : &cond_in.c_crossattn;
                    diffusion_params.c_concat = cond_in.c_concat.empty() ? nullptr : &cond_in.c_concat;
                    diffusion_params.y = cond_in.c_vector.empty() ? nullptr : &cond_in.c_vector;
                    diffusion_params.t5_ids = cond_in.c_t5_ids.empty() ? nullptr : &cond_in.c_t5_ids;
                    diffusion_params.t5_weights = cond_in.c_t5_weights.empty() ? nullptr : &cond_in.c_t5_weights;
                    diffusion_params.skip_layers = nullptr;
                };
                // Substep-path tap-driven host capture: residual
                // via ModelIn/ModelOut taps, read back to host. No CacheGraphScope.
                hooks.substep_capture_host = [&, set_params]() {
                    set_params();
                    return model->compute_substep_capture_host(runtime_->n_threads(), diffusion_params);
                };
                // Substep-path tap-driven host inject: x_before +
                // feature with the region skipped, no CacheGraphScope.
                hooks.substep_inject_host = [&, set_params](const sd::Tensor<float>& feat) {
                    set_params();
                    return model->compute_substep_inject_host(runtime_->n_threads(), diffusion_params, feat, 0, -1);
                };
                if (cache_runtime.granularity() == cache::CacheGranularity::Probe) {
                    // Substep-path tap-driven host probe: before/probe
                    // via ModelIn/BlockOut[m-1] taps, read back to host.
                    hooks.substep_probe_host = [&, set_params](int depth) {
                        set_params();
                        return model->compute_substep_probe_host(runtime_->n_threads(), diffusion_params, depth);
                    };
                }
                // On-GPU MagCache device-slot path (Feature granularity only,
                // opt-in via ED_WAN_CACHE_GPU). Dual-wired alongside the host
                // hooks: when the device store is null the lowering falls back to
                // host. DiCache (Probe) never reaches here, so it stays host-only.
                const bool feature_gpu =
                    cache_runtime.granularity() == cache::CacheGranularity::Feature &&
                    WAN::WanRunner::feature_gpu_enabled();
                if (feature_gpu) {
                    hooks.substep_capture = [&, set_params](std::vector<cache::GraphExtension> exts) {
                        set_params();
                        return model->compute_substep_capture_slot(runtime_->n_threads(), diffusion_params, std::move(exts));
                    };
                    hooks.substep_inject_slot = [&, set_params](std::vector<cache::GraphExtension> exts) {
                        set_params();
                        return model->compute_substep_inject_slot(runtime_->n_threads(), diffusion_params, std::move(exts));
                    };
                }
            }
            return hooks;
        };

        sd::Tensor<float> cond_out;
        if (!use_cfg_parallel) {
            cond_out = cache_enabled
                ? cache_runtime.run_branch(cache::CacheBranch::Cond,
                                           static_cast<const void*>(&conditions.cond),
                                           make_hooks(conditions.cond, nullptr))
                : run_condition(model, &diffusion_params, conditions.cond, nullptr, nullptr, error);
            if (cond_out.empty()) {
                return {};
            }
        }

        sd::Tensor<float> uncond_out;
        if (!conditions.uncond.empty()) {
            if (use_cfg_parallel) {
                const SDCondition& local_condition = cfg_rank == 0 ? conditions.uncond : conditions.cond;
                sd::Tensor<float> local_out = run_condition(model,
                                                            &diffusion_params,
                                                            local_condition,
                                                            nullptr,
                                                            nullptr,
                                                            error);
                std::vector<sd::Tensor<float>> gathered;
                if (local_out.empty() ||
                    !parallel::cfg_all_gather(*runtime_->parallel_context(), local_out, &gathered, error) ||
                    gathered.size() != 2) {
                    if (error != nullptr && error->empty()) {
                        *error = sd_format("Wan CFG parallel gather failed at step %d", step);
                    }
                    return {};
                }
                uncond_out = std::move(gathered[0]);
                cond_out = std::move(gathered[1]);
            } else {
                uncond_out = cache_enabled
                    ? cache_runtime.run_branch(cache::CacheBranch::Uncond,
                                               static_cast<const void*>(&conditions.uncond),
                                               make_hooks(conditions.uncond, nullptr))
                    : run_condition(model, &diffusion_params, conditions.uncond, nullptr, nullptr, error);
                if (uncond_out.empty()) {
                    return {};
                }
            }
        }

        sd::Tensor<float> img_cond_out;
        if (!conditions.img_cond.empty()) {
            img_cond_out = run_condition(model,
                                         &diffusion_params,
                                         conditions.img_cond,
                                         conditions.cond.c_concat.empty() ? nullptr : &conditions.cond.c_concat,
                                         nullptr,
                                         error);
            if (img_cond_out.empty()) {
                return {};
            }
        }

        sd::Tensor<float> latent_result = cond_out;
        if (!uncond_out.empty()) {
            if (!img_cond_out.empty()) {
                latent_result = uncond_out +
                                img_cfg_scale * (img_cond_out - uncond_out) +
                                cfg_scale * (cond_out - img_cond_out);
            } else {
                latent_result = uncond_out + cfg_scale * (cond_out - uncond_out);
            }
        } else if (!img_cond_out.empty()) {
            latent_result = img_cond_out + cfg_scale * (cond_out - img_cond_out);
        }

        // SenCache calibration: finite-diff sensitivities on the CFG-combined
        // velocity. Two extra plain forwards/step, calibration only; gated to the
        // t2v cond/uncond path (no CFG-parallel, no VACE, no image condition) so
        // the profile matches inference-time seam availability. Wan applies c_in
        // and (TI2V) a mask + timestep re-encoding, so the forward lambda rebuilds
        // the network input from a raw latent via the same helpers as the loop.
        if (cache_enabled && cache_runtime.needs_calibration() &&
            !use_cfg_parallel && cache_vace_ok && conditions.img_cond.empty()) {
            auto forward_at = [&](const sd::Tensor<float>& x_raw, float sigma_eval) -> sd::Tensor<float> {
                std::vector<float> cs = denoiser_->get_scalings(sigma_eval);
                if (cs.size() != 3) {
                    return {};
                }
                std::vector<float> ts_vec = prepare_sample_timesteps(sigma_eval, shifted_timestep,
                                                                     init_latent, denoise_mask);
                sd::Tensor<float> ts_eval({static_cast<int64_t>(ts_vec.size())}, ts_vec);
                sd::Tensor<float> input_eval = x_raw * cs[2];  // c_in
                if (!denoise_mask.empty() && version_ == VERSION_WAN2_2_TI2V) {
                    input_eval = input_eval * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
                std::string calib_err;
                DiffusionParams p = diffusion_params;
                p.x = &input_eval;
                p.timesteps = &ts_eval;
                sd::Tensor<float> cond_v = run_condition(model, &p, conditions.cond, nullptr, nullptr, &calib_err);
                if (cond_v.empty() || conditions.uncond.empty()) {
                    return cond_v;
                }
                sd::Tensor<float> uncond_v = run_condition(model, &p, conditions.uncond, nullptr, nullptr, &calib_err);
                if (uncond_v.empty()) {
                    return {};
                }
                return uncond_v + cfg_scale * (cond_v - uncond_v);
            };
            cache_runtime.calibrate(cache::CacheBranch::Cond,
                                    static_cast<const void*>(&conditions.cond),
                                    x, latent_result, forward_at);
        }

        sd::Tensor<float> denoised = latent_result * c_out + x * c_skip;
        if (!denoise_mask.empty()) {
            denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
        }

        // Deterministic Euler update in sigma space:
        //   d = (x - denoised) / sigma
        //   x_next = x + d * (sigma_next - sigma)
        sd::Tensor<float> d = (x - denoised) / sigma;
        x = x + d * (sigma_next - sigma);

        LOG_DEBUG("wan denoise step %d/%zu sigma %.5f -> %.5f",
                  step,
                  sigmas.size() - 1,
                  sigma,
                  sigma_next);
        if (cache_enabled) {
            cache::CacheStepInfo cache_step;
            cache_step.step_index = static_cast<int>(i);
            cache_step.num_steps = static_cast<int>(sigmas.size() - 1);
            cache_step.sigma = sigma;
            cache_step.sigma_next = sigma_next;
            cache_runtime.end_step(cache_step);
        }
        if (control != nullptr) {
            control->step_done();
        }
    }

    if (cache_enabled) {
        cache_runtime.log_summary(sigmas.size() - 1);
    }
    model->free_compute_buffer();
    return denoiser_->inverse_noise_scaling(sigmas.back(), x);
}

sd::Tensor<float> WanPipeline::sample_video_latent(const ed_video_generation_params_t* params,
                                                   const WanVideoConditionPack& conditions,
                                                   const sd::Tensor<float>& init_latent,
                                                   const sd::Tensor<float>& noise,
                                                   const sd::Tensor<float>& denoise_mask,
                                                   const sd::Tensor<float>& vace_context,
                                                   std::string* error) {
    const int low_steps = resolve_steps(params->sample.steps);
    int high_steps = high_noise_diffusion_ != nullptr ? params->high_noise_sample.steps : 0;
    const int total_steps = low_steps + std::max(0, high_steps);

    std::vector<float> sigmas = build_sigmas(params->sample,
                                             params->width,
                                             params->height,
                                             total_steps);
    if (sigmas.size() < 2) {
        set_error(error, "failed to build Wan sigma schedule");
        return {};
    }

    if (high_steps < 0) {
        high_steps = 0;
        for (size_t i = 0; i < sigmas.size(); ++i) {
            if (sigmas[i] < params->moe_boundary) {
                high_steps = static_cast<int>(i);
                break;
            }
        }
        LOG_DEBUG("wan switching from high noise model at step %d", high_steps);
    }

    sd::Tensor<float> x = denoiser_->noise_scaling(sigmas.front(), noise, init_latent);

    if (high_steps > 0) {
        const size_t split = std::min<size_t>(static_cast<size_t>(high_steps), sigmas.size() - 1);
        std::vector<float> high_sigmas(sigmas.begin(), sigmas.begin() + split + 1);
        x = euler_denoise(high_noise_diffusion_,
                          x,
                          high_sigmas,
                          conditions,
                          params->high_noise_sample,
                          init_latent,
                          denoise_mask,
                          vace_context,
                          params->vace_strength,
                          error);
        if (x.empty()) {
            return {};
        }

        std::vector<float> low_sigmas(sigmas.begin() + split, sigmas.end());
        return euler_denoise(diffusion_,
                             x,
                             low_sigmas,
                             conditions,
                             params->sample,
                             init_latent,
                             denoise_mask,
                             vace_context,
                             params->vace_strength,
                             error);
    }

    return euler_denoise(diffusion_,
                         x,
                         sigmas,
                         conditions,
                         params->sample,
                         init_latent,
                         denoise_mask,
                         vace_context,
                         params->vace_strength,
                         error);
}

ed_status_t WanPipeline::decode_video_latent(const sd::Tensor<float>& latent,
                                             const ed_tiling_params_t& tiling,
                                             ed_video_t* out,
                                             std::string* error) {
    if (latent.empty()) {
        set_error(error, "Wan final latent is empty");
        return ED_STATUS_GENERATION_FAILED;
    }

    const int64_t t0 = ggml_time_ms();
    sd::Tensor<float> vae_latent = vae_->diffusion_to_vae_latents(latent);
    sd::Tensor<float> video = vae_->decode(runtime_->n_threads(),
                                           vae_latent,
                                           tiling,
                                           true,
                                           runtime_->circular_x(),
                                           runtime_->circular_y());
    const int64_t t1 = ggml_time_ms();
    LOG_INFO("wan vae decode completed, taking %.2fs", (t1 - t0) / 1000.0f);

    if (video.empty()) {
        set_error(error, "Wan VAE decode failed");
        return ED_STATUS_GENERATION_FAILED;
    }

    const int frame_count = static_cast<int>(video.shape()[2]);
    ed_image_t* frames = static_cast<ed_image_t*>(std::calloc(static_cast<size_t>(frame_count),
                                                              sizeof(ed_image_t)));
    if (frames == nullptr) {
        set_error(error, "failed to allocate Wan video frames");
        return ED_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < frame_count; ++i) {
        frames[i] = tensor_frame_to_image(video, i);
        if (frames[i].data == nullptr) {
            for (int j = 0; j < i; ++j) {
                std::free(frames[j].data);
            }
            std::free(frames);
            set_error(error, "failed to convert Wan decoded tensor to image");
            return ED_STATUS_OUT_OF_MEMORY;
        }
    }

    out->frames = frames;
    out->frame_count = frame_count;
    return ED_STATUS_OK;
}

ed_status_t WanPipeline::generate_image(const ed_image_generation_params_t*,
                                        ed_image_batch_t* out,
                                        std::string* error) {
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }
    set_error(error, "WanPipeline supports video generation only");
    return ED_STATUS_UNSUPPORTED;
}

ed_status_t WanPipeline::generate_video(const ed_video_generation_params_t* params,
                                        ed_video_t* out,
                                        std::string* error) {
    if (out == nullptr) {
        set_error(error, "video output is null");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    out->frames = nullptr;
    out->frame_count = 0;

    if (!ready_ || runtime_ == nullptr || !conditioner_ || !diffusion_ || !vae_ || !denoiser_) {
        set_error(error, "WanPipeline is not ready");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (!validate_video_params(params, error)) {
        return ED_STATUS_INVALID_ARGUMENT;
    }

    // Stage-1 intentionally supports T2V.  For I2V/VACE the model components are
    // registered, but correct c_concat/vace_context construction still has to be
    // wired before generation is safe.
    if (clip_vision_ != nullptr || params->init_image != nullptr || params->end_image != nullptr) {
        set_error(error, "Wan I2V/FLF2V generation is not wired yet in WanPipeline stage 1");
        return ED_STATUS_UNSUPPORTED;
    }
    if ((params->control_frames != nullptr && params->control_frame_count > 0) ||
        diffusion_->get_desc() == "Wan2.1-VACE-1.3B" ||
        diffusion_->get_desc() == "Wan2.x-VACE-14B") {
        set_error(error, "Wan VACE/control video generation is not wired yet in WanPipeline stage 1");
        return ED_STATUS_UNSUPPORTED;
    }
    if (params->loras != nullptr && params->lora_count > 0) {
        LOG_WARN("WanPipeline stage 1 ignores LoRA list; apply LoRA before prepare() or add adapter wiring");
    }
    if (params->sample.sampler != ED_SAMPLER_AUTO && params->sample.sampler != ED_SAMPLER_EULER) {
        LOG_WARN("WanPipeline stage 1 uses deterministic Euler; requested sampler %d is ignored",
                 static_cast<int>(params->sample.sampler));
    }

    const int64_t seed = resolve_seed(params->seed);
    rng_->manual_seed(seed);
    sampler_rng_->manual_seed(seed);
    set_flow_shift(params->sample.flow_shift);

    const int low_steps = resolve_steps(params->sample.steps);
    const int high_steps = high_noise_diffusion_ != nullptr ? params->high_noise_sample.steps : 0;
    const int total_steps = low_steps + std::max(0, high_steps);
    if (GenerationControl* control = runtime_->generation_control()) {
        control->start(total_steps);
    }

    LOG_INFO("wan generate_video %dx%d frames=%d latent_frames=%d seed=%" PRId64,
             params->width,
             params->height,
             params->frames,
             latent_frames(params->frames),
             seed);
    prewarm_wan_cudnn_sdpa(params->width, params->height, params->frames);
    prewarm_wan_sp_comm(params->width, params->height, params->frames);

    sd::Tensor<float> init_latent = generate_init_latent(params->width,
                                                         params->height,
                                                         params->frames);
    if (init_latent.empty()) {
        set_error(error, "failed to create Wan init latent");
        return ED_STATUS_GENERATION_FAILED;
    }

    sd::Tensor<float> noise = sd::randn_like<float>(init_latent, rng_);
    if (noise.empty()) {
        set_error(error, "failed to create Wan noise latent");
        return ED_STATUS_GENERATION_FAILED;
    }

    WanVideoConditionPack conditions;
    if (!prepare_text_conditions(params,
                                 sd::Tensor<float>(),
                                 sd::Tensor<float>(),
                                 &conditions,
                                 error)) {
        return ED_STATUS_GENERATION_FAILED;
    }

    const int64_t sample_start = ggml_time_ms();
    sd::Tensor<float> final_latent = sample_video_latent(params,
                                                         conditions,
                                                         init_latent,
                                                         noise,
                                                         sd::Tensor<float>(),
                                                         sd::Tensor<float>(),
                                                         error);
    const int64_t sample_end = ggml_time_ms();
    LOG_INFO("wan sampling completed, taking %.2fs", (sample_end - sample_start) / 1000.0f);

    if (final_latent.empty()) {
        if (error != nullptr && error->empty()) {
            *error = "Wan sampling failed";
        }
        return ED_STATUS_GENERATION_FAILED;
    }

    return decode_video_latent(final_latent, runtime_->vae_tiling(), out, error);
}

ed_scheduler_t WanPipeline::default_scheduler(ed_sampler_t method) const {
    if (method == ED_SAMPLER_LCM || method == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (method == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

}  // namespace edgedit
