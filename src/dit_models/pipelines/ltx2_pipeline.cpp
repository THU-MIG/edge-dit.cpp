#include "dit_models/pipelines/ltx2_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <map>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#include "parallel/cfg_parallel.hpp"
#include "utils/rng.hpp"
#include "utils/rng_philox.hpp"
#include "utils/util.h"

namespace edgedit {
namespace {

constexpr int kVideoChannels = 128;
constexpr int kAudioFrequencyBins = 16;
constexpr int kAudioChannels = 8;
constexpr int kDefaultSteps = 20;

sd::Tensor<float> pack_audio_video(const sd::Tensor<float>& video,
                                   const sd::Tensor<float>& audio) {
    if (audio.empty()) {
        return video;
    }
    const int64_t spatial = video.shape()[0] * video.shape()[1] * video.shape()[2];
    const int64_t extra_channels = (audio.numel() + spatial - 1) / spatial;
    std::vector<int64_t> shape = video.shape();
    shape[3] += extra_channels;
    sd::Tensor<float> packed = sd::zeros<float>(shape);
    std::copy_n(video.data(), video.numel(), packed.data());
    std::copy_n(audio.data(), audio.numel(), packed.data() + video.numel());
    return packed;
}

sd::Tensor<float> unpack_audio(const sd::Tensor<float>& packed, int audio_length) {
    if (packed.empty() || audio_length <= 0 || packed.shape()[3] <= kVideoChannels) {
        return {};
    }
    const int64_t required = static_cast<int64_t>(audio_length) * kAudioFrequencyBins * kAudioChannels;
    const int64_t spatial = packed.shape()[0] * packed.shape()[1] * packed.shape()[2];
    const int64_t available = (packed.shape()[3] - kVideoChannels) * spatial;
    if (available < required) {
        return {};
    }
    sd::Tensor<float> audio({kAudioFrequencyBins, audio_length, kAudioChannels, 1});
    std::copy_n(packed.data() + kVideoChannels * spatial, required, audio.data());
    return audio;
}

size_t sum_materialized_params(const std::map<std::string, ggml_tensor*>& tensors) {
    size_t bytes = 0;
    for (const auto& item : tensors) {
        if (item.second != nullptr) {
            bytes += ggml_nbytes(item.second);
        }
    }
    return bytes;
}

bool has_requested_prefix(const std::vector<std::string>& prefixes,
                          const std::string& prefix) {
    return std::find(prefixes.begin(), prefixes.end(), prefix) != prefixes.end();
}

template <typename Runner>
class PhaseParamsGuard {
public:
    explicit PhaseParamsGuard(Runner* runner) : runner_(runner) {}
    ~PhaseParamsGuard() {
        if (active_) {
            runner_->release_params_after_phase();
        }
    }

    bool stage() {
        active_ = runner_ != nullptr && runner_->stage_params_for_phase();
        return active_;
    }

private:
    Runner* runner_ = nullptr;
    bool active_ = false;
};

size_t estimate_ltx_materialized_component_bytes(ggml_backend_t backend,
                                                 const ModelLoader& loader,
                                                 const std::vector<std::string>& prefixes) {
    const auto& storage = loader.get_tensor_storage_map();
    std::map<std::string, ggml_tensor*> tensors;

    if (has_requested_prefix(prefixes, "model.diffusion_model")) {
        LTXAVModel model(backend, false, storage);
        model.get_param_tensors(tensors);
        return sum_materialized_params(tensors);
    }
    if (has_requested_prefix(prefixes, "text_encoders.llm") ||
        has_requested_prefix(prefixes, "text_embedding_projection")) {
        LTXAVEmbedder embedder(backend, false, storage);
        embedder.get_param_tensors(tensors);
        return sum_materialized_params(tensors);
    }
    if (has_requested_prefix(prefixes, "first_stage_model")) {
        const bool has_encoder = std::any_of(storage.begin(), storage.end(), [](const auto& item) {
            return item.first.rfind("first_stage_model.encoder.", 0) == 0;
        });
        LTXVideoVAE vae(backend,
                        false,
                        storage,
                        "first_stage_model",
                        !has_encoder,
                        loader.version());
        vae.get_param_tensors(tensors, "first_stage_model");
        return sum_materialized_params(tensors);
    }
    if (has_requested_prefix(prefixes, "audio_vae") ||
        has_requested_prefix(prefixes, "vocoder")) {
        LTXV::LTXAudioVAERunner audio_vae(backend, false, storage, "");
        audio_vae.get_param_tensors(tensors);
        return sum_materialized_params(tensors);
    }

    // The upscaler is loaded from its own standalone file, whose tensor names have
    // no component prefix. Identify that file by the runner's required root tensor.
    if (prefixes.size() == 1 && prefixes.front().empty() &&
        storage.find("initial_norm.weight") != storage.end()) {
        LTXVUpsampler::LatentUpsamplerRunner upscaler(backend, false, storage);
        upscaler.get_param_tensors(tensors);
        return sum_materialized_params(tensors);
    }
    return 0;
}

sd::Tensor<float> pack_audio_video_mask(const sd::Tensor<float>& video_mask,
                                        const sd::Tensor<float>& video,
                                        const sd::Tensor<float>& audio) {
    if (video_mask.empty() || audio.empty()) {
        return video_mask;
    }
    const int64_t spatial = video.shape()[0] * video.shape()[1] * video.shape()[2];
    const int64_t extra_channels = (audio.numel() + spatial - 1) / spatial;
    sd::Tensor<float> expanded_video_mask = video_mask;
    if (video_mask.shape()[3] == 1 && video.shape()[3] != 1) {
        expanded_video_mask = video_mask * sd::Tensor<float>::ones(video.shape());
    }
    std::vector<int64_t> audio_mask_shape = video.shape();
    audio_mask_shape[3] = extra_channels;
    return sd::ops::concat(expanded_video_mask,
                           sd::Tensor<float>::ones(audio_mask_shape),
                           3);
}

sd::Tensor<float> image_to_tensor(const ed_image_t& image,
                                  int preprocess_width,
                                  int preprocess_height,
                                  int target_width,
                                  int target_height) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 ||
        image.channels == 0 || preprocess_width <= 0 || preprocess_height <= 0 ||
        target_width <= 0 || target_height <= 0) {
        return {};
    }

    const int source_width = static_cast<int>(image.width);
    const int source_height = static_cast<int>(image.height);
    const int source_channels = static_cast<int>(image.channels);
    const float target_aspect = static_cast<float>(preprocess_width) /
                                static_cast<float>(preprocess_height);
    const float source_aspect = static_cast<float>(source_width) / static_cast<float>(source_height);

    int crop_x = 0;
    int crop_y = 0;
    int crop_width = source_width;
    int crop_height = source_height;
    if (source_aspect > target_aspect) {
        crop_width = static_cast<int>(source_height * target_aspect);
        crop_x = (source_width - crop_width) / 2;
    } else if (source_aspect < target_aspect) {
        crop_height = static_cast<int>(source_width / target_aspect);
        crop_y = (source_height - crop_height) / 2;
    }

    std::vector<uint8_t> cropped(static_cast<size_t>(crop_width) * crop_height * 3);
    for (int y = 0; y < crop_height; ++y) {
        for (int x = 0; x < crop_width; ++x) {
            const uint8_t* source = image.data +
                                    (static_cast<size_t>(crop_y + y) * source_width + crop_x + x) *
                                        source_channels;
            uint8_t* destination = cropped.data() +
                                   (static_cast<size_t>(y) * crop_width + x) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                destination[channel] = source[std::min(channel, source_channels - 1)];
            }
        }
    }

    std::vector<uint8_t> resized(static_cast<size_t>(preprocess_width) * preprocess_height * 3);
    if (crop_width == preprocess_width && crop_height == preprocess_height) {
        resized = std::move(cropped);
    } else if (!stbir_resize(cropped.data(),
                             crop_width,
                             crop_height,
                             0,
                             resized.data(),
                             preprocess_width,
                             preprocess_height,
                             0,
                             STBIR_TYPE_UINT8,
                             3,
                             STBIR_ALPHA_CHANNEL_NONE,
                             0,
                             STBIR_EDGE_CLAMP,
                             STBIR_EDGE_CLAMP,
                             STBIR_FILTER_BOX,
                             STBIR_FILTER_BOX,
                             STBIR_COLORSPACE_SRGB,
                             nullptr)) {
        return {};
    }

    sd::Tensor<float> tensor({preprocess_width, preprocess_height, 3, 1});
    for (int y = 0; y < preprocess_height; ++y) {
        for (int x = 0; x < preprocess_width; ++x) {
            const uint8_t* pixel = resized.data() +
                                   (static_cast<size_t>(y) * preprocess_width + x) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                tensor.index(x, y, channel, 0) = static_cast<float>(pixel[channel]) / 255.f;
            }
        }
    }
    if (preprocess_width != target_width || preprocess_height != target_height) {
        tensor = sd::ops::interpolate(tensor, {target_width, target_height, 3, 1});
    }
    return tensor;
}

float latent_corner_to_pixel_frame(int64_t corner_index) {
    constexpr int temporal_scale = 8;
    return std::max(0.f,
                    static_cast<float>(corner_index * temporal_scale + 1 - temporal_scale));
}

void set_video_position(sd::Tensor<float>* positions,
                        int64_t token,
                        float t_start,
                        float t_end,
                        float h_start,
                        float h_end,
                        float w_start,
                        float w_end) {
    positions->index(0, 0, token, 0) = t_start;
    positions->index(1, 0, token, 0) = t_end;
    positions->index(0, 1, token, 0) = h_start;
    positions->index(1, 1, token, 0) = h_end;
    positions->index(0, 2, token, 0) = w_start;
    positions->index(1, 2, token, 0) = w_end;
}

sd::Tensor<float> build_video_positions(int64_t width,
                                        int64_t height,
                                        int64_t target_frames,
                                        int64_t keyframe_frames,
                                        int keyframe_index,
                                        int fps) {
    constexpr int spatial_scale = 32;
    constexpr int temporal_scale = 8;
    sd::Tensor<float> positions({2, 3, width * height * (target_frames + keyframe_frames), 1});
    int64_t token = 0;
    for (int64_t t = 0; t < target_frames; ++t) {
        const float t_start = latent_corner_to_pixel_frame(t) / fps;
        const float t_end = latent_corner_to_pixel_frame(t + 1) / fps;
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_video_position(&positions,
                                   token++,
                                   t_start,
                                   t_end,
                                   static_cast<float>(h * spatial_scale),
                                   static_cast<float>((h + 1) * spatial_scale),
                                   static_cast<float>(w * spatial_scale),
                                   static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    for (int64_t t = 0; t < keyframe_frames; ++t) {
        const float t_start = static_cast<float>(keyframe_index + t * temporal_scale) / fps;
        const float t_end = static_cast<float>(keyframe_index + t * temporal_scale + 1) / fps;
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_video_position(&positions,
                                   token++,
                                   t_start,
                                   t_end,
                                   static_cast<float>(h * spatial_scale),
                                   static_cast<float>((h + 1) * spatial_scale),
                                   static_cast<float>(w * spatial_scale),
                                   static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    return positions;
}

sd::Tensor<float> make_video_timesteps(const sd::Tensor<float>& init_latent,
                                       const sd::Tensor<float>& denoise_mask,
                                       float timestep) {
    const int64_t width = init_latent.shape()[0];
    const int64_t height = init_latent.shape()[1];
    const int64_t frames = init_latent.shape()[2];
    sd::Tensor<float> timesteps({width * height * frames});
    int64_t token = 0;
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                timesteps[token++] = denoise_mask.index(x, y, frame, 0, 0) * timestep;
            }
        }
    }
    return timesteps;
}

uint8_t to_u8(float value) {
    return static_cast<uint8_t>(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
}

}  // namespace

LTX2Pipeline::LTX2Pipeline(SDVersion version) : version_(version) {}

bool LTX2Pipeline::set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
    LOG_ERROR("%s", message.c_str());
    return false;
}

int64_t LTX2Pipeline::resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    return static_cast<int64_t>(std::rand());
}

bool LTX2Pipeline::has_prefix(const ModelLoader& loader, const std::string& prefix) {
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (starts_with(item.first, prefix)) {
            return true;
        }
    }
    return false;
}

bool LTX2Pipeline::prepare_memory_plan(const ed_context_params_t&,
                                       ModelRuntime& runtime,
                                       ModelLoader& loader,
                                       std::string* error) {
    static constexpr int kPlanningContextTokens = 256;
    runtime.set_measured_dit_headroom(0);
    if (!runtime.auto_allocate() || runtime.fit_width() <= 0 ||
        runtime.fit_height() <= 0 || runtime.fit_frames() <= 0) {
        return true;
    }
    if (!ed_version_is_ltxav(loader.version())) {
        return set_error(error, "LTX2Pipeline memory planning got a non-LTX model version");
    }

    runtime.set_component_memory_estimator(
        [&runtime](const ModelLoader& component_loader,
                   const std::vector<std::string>& prefixes) {
            return estimate_ltx_materialized_component_bytes(runtime.backend(),
                                                              component_loader,
                                                              prefixes);
        });

    auto measure_model = std::make_unique<LTXAVModel>(runtime.backend(),
                                                       false,
                                                       loader.get_tensor_storage_map());
    measure_model->set_flash_attention_enabled(runtime.flash_attention());
    const int latent_width = (runtime.fit_width() + 31) / 32;
    const int latent_height = (runtime.fit_height() + 31) / 32;
    // Reserve for the largest LTX input graph: E2V/FLF2V append an encoded
    // keyframe latent, and conditioned denoising uses positions plus per-token
    // timesteps. This is intentionally conservative for plain T2V/I2V.
    const int latent_frames = 2 + (runtime.fit_frames() - 1) / 8;
    const bool measure_audio = has_prefix(loader, "audio_vae.") &&
                               has_prefix(loader, "vocoder.");
    const int audio_length = measure_audio
                                 ? static_cast<int>(std::ceil(
                                       (static_cast<float>(runtime.fit_frames()) /
                                        static_cast<float>(std::max(1, runtime.fit_fps()))) * 25.0f))
                                 : 0;
    // Without an audio VAE, the normal T2V path uses a scalar timestep and has no
    // audio-aware conditioned graph. Probe that graph shape directly; the AV path
    // keeps the conservative conditioned probe for E2V/FLF2V and audio generation.
    const bool probe_conditioned_graph = measure_audio;
    const size_t measured = measure_model->measure_compute_buffer_at(latent_width,
                                                                      latent_height,
                                                                      latent_frames,
                                                                      audio_length,
                                                                      kPlanningContextTokens,
                                                                      probe_conditioned_graph);
    const int vae_latent_frames = 1 + (runtime.fit_frames() - 1) / 8;
    auto measure_video_vae = std::make_unique<LTXVideoVAE>(runtime.backend(),
                                                           false,
                                                           loader.get_tensor_storage_map(),
                                                           "first_stage_model",
                                                           true,
                                                           loader.version());
    const size_t measured_video_vae = measure_video_vae->measure_decode_compute_buffer_at(
        latent_width,
        latent_height,
        vae_latent_frames);
    const size_t budget = runtime.effective_budget_bytes();
    if (!runtime.vae_tiling().enabled && !runtime.vae_tiling().force_disable &&
        budget > 0 && measured_video_vae > budget / 2) {
        runtime.enable_vae_tiling_for_memory();
        LOG_INFO("auto-allocate: enabling LTX VAE tiling (decode compute %.2f GB exceeds half of %.2f GB budget)",
                 measured_video_vae / (1024.0 * 1024.0 * 1024.0),
                 budget / (1024.0 * 1024.0 * 1024.0));
    }
    if (measured > 0) {
        // The shared resident headroom also protects Gemma's encode graph. At the
        // planning context the DiT graph is small, while Gemma's fixed 1024-token
        // minimum can peak above 2 GiB; keep a conservative floor for that phase.
        const size_t ltx_planning_compute_floor =
            static_cast<size_t>(3) * 1024 * 1024 * 1024 / 2;  // 1.5 GiB raw + 1.5 GiB VAE allowance
        runtime.set_measured_dit_headroom(std::max(measured, ltx_planning_compute_floor));
    }
    LOG_INFO("auto-allocate: measured LTX-2.3 compute buffers: DiT %.2f GB, video VAE %.2f GB at latent %dx%dx%d audio=%d fps=%d context=%d conditioned=%s",
             measured / (1024.0 * 1024.0 * 1024.0),
             measured_video_vae / (1024.0 * 1024.0 * 1024.0),
             latent_width,
             latent_height,
             vae_latent_frames,
             audio_length,
             runtime.fit_fps(),
             kPlanningContextTokens,
             probe_conditioned_graph ? "av" : "t2v");
    return true;
}

bool LTX2Pipeline::prepare(const ed_context_params_t& params,
                           ModelRuntime& runtime,
                           const ModelLoader& loader,
                           PipelineTensorRegistry& registry,
                           std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    registry.clear();
    if (!ed_version_is_ltxav(version_)) {
        return set_error(error, "LTX2Pipeline got a non-LTX model version");
    }
    if (runtime.backend() == nullptr || runtime.clip_backend() == nullptr ||
        runtime.vae_backend() == nullptr) {
        return set_error(error, "LTX2Pipeline requires initialized runtime backends");
    }
    if (runtime.parallel_context() != nullptr &&
        runtime.parallel_context()->sp_parallel_size() > 1) {
        return set_error(error, "LTX-2.3 does not support sequence parallelism yet");
    }
    if (runtime.parallel_context() != nullptr &&
        runtime.parallel_context()->tp_parallel_size() > 1) {
        return set_error(error, "LTX-2.3 does not support tensor parallelism yet");
    }

    const auto& storage = loader.get_tensor_storage_map();
    if (!has_prefix(loader, "model.diffusion_model.") ||
        !has_prefix(loader, "text_encoders.llm.") ||
        !has_prefix(loader, "text_embedding_projection.") ||
        !has_prefix(loader, "first_stage_model.")) {
        return set_error(error, "LTX-2.3 requires diffusion, Gemma, embeddings connector, and video VAE weights");
    }
    has_audio_vae_ = has_prefix(loader, "audio_vae.") && has_prefix(loader, "vocoder.");
    has_video_vae_encoder_ = has_prefix(loader, "first_stage_model.encoder.");

    std::unique_ptr<ModelLoader> latent_upscaler_loader;
    if (params.latent_upscaler_path != nullptr && params.latent_upscaler_path[0] != '\0') {
        latent_upscaler_loader = std::make_unique<ModelLoader>();
        if (!latent_upscaler_loader->init_from_file(params.latent_upscaler_path)) {
            return set_error(error,
                             std::string("failed to open LTX latent upscaler: ") +
                                 params.latent_upscaler_path);
        }
    }

    runtime.reset_auto_allocate_state();
    const size_t budget = runtime.effective_budget_bytes();
    size_t remaining = budget;
    const bool diffusion_offload = runtime.dit_offload_params_to_cpu() ||
        runtime.plan_component_offload(loader, "model.diffusion_model", remaining);
    const bool text_offload = runtime.clip_offload_params_to_cpu() ||
        runtime.plan_component_offload(loader,
                                       std::vector<std::string>{"text_encoders.llm",
                                                                "text_embedding_projection"},
                                       remaining);
    const bool video_vae_offload = runtime.vae_offload_params_to_cpu() ||
        runtime.plan_component_offload(loader, "first_stage_model", remaining);
    const bool audio_vae_offload = runtime.vae_offload_params_to_cpu() ||
        (has_audio_vae_ && runtime.plan_component_offload(loader,
                                                          std::vector<std::string>{"audio_vae",
                                                                                   "vocoder"},
                                                          remaining));
    const bool latent_upscaler_offload = runtime.vae_offload_params_to_cpu() ||
        (latent_upscaler_loader &&
         runtime.plan_component_offload(*latent_upscaler_loader, "", remaining));
    diffusion_offload_ = diffusion_offload;
    text_offload_ = text_offload;
    runtime.finalize_auto_segment_budget(budget);

    conditioner_ = std::make_shared<LTXAVEmbedder>(runtime.clip_backend(), text_offload, storage);
    diffusion_ = std::make_shared<LTXAVModel>(runtime.backend(), diffusion_offload, storage);
    video_vae_ = std::make_shared<LTXVideoVAE>(runtime.vae_backend(),
                                                video_vae_offload,
                                                storage,
                                                "first_stage_model",
                                                !has_video_vae_encoder_,
                                                version_);
    if (has_audio_vae_) {
        audio_vae_ = std::make_shared<LTXV::LTXAudioVAERunner>(runtime.vae_backend(),
                                                               audio_vae_offload,
                                                               storage,
                                                               "");
    }
    denoiser_ = std::make_shared<FluxFlowDenoiser>();
    rng_ = std::make_shared<PhiloxRNG>();

    const size_t max_graph_vram = runtime.max_graph_vram_bytes();
    conditioner_->set_flash_attention_enabled(runtime.flash_attention());
    conditioner_->set_max_graph_vram_bytes(max_graph_vram);
    diffusion_->set_flash_attention_enabled(runtime.flash_attention());
    diffusion_->set_max_graph_vram_bytes(max_graph_vram);
    video_vae_->set_flash_attention_enabled(runtime.flash_attention());
    video_vae_->set_max_graph_vram_bytes(max_graph_vram);
    if (audio_vae_) {
        audio_vae_->set_flash_attention_enabled(runtime.flash_attention());
        audio_vae_->set_max_graph_vram_bytes(max_graph_vram);
    }
    if (auto group = runtime.graph_process_group_ref()) {
        diffusion_->set_process_group(group);
    }

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());
    conditioner_->set_max_graph_vram_bytes(
        runtime.text_encoder_segment_budget(conditioner_->get_params_buffer_size(), text_offload));
    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors());
    video_vae_->alloc_params_buffer();
    video_vae_->get_param_tensors(registry.tensors(), "first_stage_model");
    if (audio_vae_) {
        audio_vae_->alloc_params_buffer();
        audio_vae_->get_param_tensors(registry.tensors());
    }
    if (!load_latent_upscaler(params,
                              latent_upscaler_loader.get(),
                              latent_upscaler_offload,
                              error)) {
        return false;
    }

    if (!has_video_vae_encoder_) {
        registry.ignore_prefix("first_stage_model.encoder.");
    }
    registry.ignore_prefix("audio_vae.encoder.");
    registry.ignore_prefix("text_encoders.llm.lm_head.");
    registry.ignore_prefix("model.diffusion_model.__");
    LOG_INFO("LTX-2.3 pipeline registered %zu tensors (audio=%s)",
             registry.tensors().size(), has_audio_vae_ ? "on" : "off");
    return true;
}

bool LTX2Pipeline::load_latent_upscaler(const ed_context_params_t& params,
                                        ModelLoader* loader,
                                        bool offload_params_to_cpu,
                                        std::string* error) {
    latent_upscaler_.reset();
    if (params.latent_upscaler_path == nullptr || params.latent_upscaler_path[0] == '\0') {
        return true;
    }
    if (loader == nullptr) {
        return set_error(error, "LTX latent upscaler metadata was not prepared");
    }
    latent_upscaler_ = std::make_shared<LTXVUpsampler::LatentUpsamplerRunner>(
        runtime_->backend(),
        offload_params_to_cpu,
        loader->get_tensor_storage_map());
    if (!latent_upscaler_->model) {
        latent_upscaler_.reset();
        return set_error(error, "unsupported LTX latent upscaler configuration");
    }
    latent_upscaler_->set_max_graph_vram_bytes(runtime_->max_graph_vram_bytes());
    if (!latent_upscaler_->alloc_params_buffer()) {
        latent_upscaler_.reset();
        return set_error(error, "failed to allocate LTX latent upscaler weights");
    }
    std::map<std::string, ggml_tensor*> tensors;
    latent_upscaler_->get_param_tensors(tensors);
    if (!loader->load_tensors(tensors, {}, runtime_->n_threads(), params.use_mmap)) {
        latent_upscaler_.reset();
        return set_error(error, "failed to load LTX latent upscaler weights");
    }
    LOG_INFO("LTX-2.3 latent upscaler loaded from '%s' (%zu tensors, scale=%.3f)",
             params.latent_upscaler_path,
             tensors.size(),
             latent_upscaler_->config.spatial_scale);
    return true;
}

bool LTX2Pipeline::prepare_latents(const ed_video_generation_params_t& params,
                                   int conditioning_width,
                                   int conditioning_height,
                                   Latents* latents,
                                   std::string* error) {
    if (latents == nullptr) {
        return set_error(error, "LTX-2.3 latent output is null");
    }
    const int latent_width = params.width / 32;
    const int latent_height = params.height / 32;
    const int latent_frames = 1 + (params.frames - 1) / 8;
    latents->video_target_frame_count = latent_frames;
    latents->video_conditioning_frame_count = 0;
    sd::Tensor<float> video = sd::zeros<float>(
        {latent_width, latent_height, latent_frames, kVideoChannels, 1});
    sd::Tensor<float> video_mask;

    const bool has_init = params.init_image != nullptr && params.init_image->data != nullptr;
    const bool has_end = params.end_image != nullptr && params.end_image->data != nullptr;
    if ((has_init || has_end) && !has_video_vae_encoder_) {
        return set_error(error, "LTX-2.3 image conditioning requires video VAE encoder weights");
    }

    const float strength = std::clamp(params.strength, 0.f, 1.f);
    const float conditioned_mask = 1.f - strength;
    if (has_init || has_end) {
        video_mask = sd::Tensor<float>::ones(
            {latent_width, latent_height, latent_frames, 1, 1});
    }

    auto encode_image = [&](const ed_image_t& image, const char* name) {
        sd::Tensor<float> tensor = image_to_tensor(image,
                                                   conditioning_width,
                                                   conditioning_height,
                                                   params.width,
                                                   params.height);
        if (tensor.empty()) {
            set_error(error, std::string("LTX-2.3 invalid ") + name + " image");
            return sd::Tensor<float>();
        }
        tensor.reshape_({params.width, params.height, 1, 3, 1});
        sd::Tensor<float> encoded = video_vae_->encode(runtime_->n_threads(),
                                                        tensor,
                                                        runtime_->vae_tiling(),
                                                        runtime_->circular_x(),
                                                        runtime_->circular_y());
        if (encoded.empty()) {
            set_error(error, std::string("LTX-2.3 failed to encode ") + name + " image");
        }
        return encoded;
    };

    auto assign_condition = [&](const sd::Tensor<float>& condition,
                                int64_t latent_index,
                                const char* name) {
        if (condition.empty() || condition.shape()[0] != video.shape()[0] ||
            condition.shape()[1] != video.shape()[1] ||
            condition.shape()[3] != video.shape()[3] ||
            latent_index < 0 || latent_index + condition.shape()[2] > video.shape()[2]) {
            return set_error(error, std::string("LTX-2.3 invalid ") + name + " latent shape");
        }
        sd::ops::slice_assign(&video,
                              2,
                              latent_index,
                              latent_index + condition.shape()[2],
                              condition);
        sd::ops::fill_slice(&video_mask,
                            2,
                            latent_index,
                            latent_index + condition.shape()[2],
                            conditioned_mask);
        return true;
    };

    if (has_init) {
        sd::Tensor<float> condition = encode_image(*params.init_image, "initial");
        if (!assign_condition(condition, 0, "initial")) {
            return false;
        }
    }
    if (has_end) {
        sd::Tensor<float> condition = encode_image(*params.end_image, "end");
        if (condition.empty()) {
            return false;
        }
        if (params.frames == 1) {
            if (!assign_condition(condition, 0, "end")) {
                return false;
            }
        } else {
            const int64_t target_frames = video.shape()[2];
            if (condition.shape()[0] != video.shape()[0] ||
                condition.shape()[1] != video.shape()[1] ||
                condition.shape()[3] != video.shape()[3]) {
                return set_error(error, "LTX-2.3 invalid end latent shape");
            }
            video = sd::ops::concat(video, condition, 2);
            latents->video_target_frame_count = static_cast<int>(target_frames);
            latents->video_conditioning_frame_count = static_cast<int>(condition.shape()[2]);
            video_mask = sd::ops::concat(
                video_mask,
                sd::full<float>({condition.shape()[0],
                                 condition.shape()[1],
                                 condition.shape()[2],
                                 1,
                                 1},
                                conditioned_mask),
                2);
            const int fps = params.fps > 0 ? params.fps : 24;
            latents->video_positions = build_video_positions(video.shape()[0],
                                                              video.shape()[1],
                                                              target_frames,
                                                              condition.shape()[2],
                                                              params.frames - 1,
                                                              fps);
        }
    }

    const int fps = params.fps > 0 ? params.fps : 24;
    latents->audio_length = has_audio_vae_
                                ? static_cast<int>(std::ceil(
                                      (static_cast<float>(params.frames) / fps) * 25.f))
                                : 0;
    sd::Tensor<float> audio = latents->audio_length > 0
                                  ? sd::zeros<float>({kAudioFrequencyBins,
                                                      latents->audio_length,
                                                      kAudioChannels,
                                                      1})
                                  : sd::Tensor<float>();
    latents->denoise_mask = pack_audio_video_mask(video_mask, video, audio);
    latents->init = pack_audio_video(video, audio);
    return !latents->init.empty();
}

ed_status_t LTX2Pipeline::generate_image(const ed_image_generation_params_t*,
                                         ed_image_batch_t* out,
                                         std::string* error) {
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }
    set_error(error, "LTX-2.3 supports video generation only");
    return ED_STATUS_UNSUPPORTED;
}

bool LTX2Pipeline::prepare_conditions(const ed_video_generation_params_t& params,
                                      Conditions* conditions,
                                      std::string* error) {
    if (conditions == nullptr || conditioner_ == nullptr) {
        return set_error(error, "LTX-2.3 conditioner is not initialized");
    }
    PhaseParamsGuard<LTXAVEmbedder> phase_guard(conditioner_.get());
    if (text_offload_) {
        const size_t params_bytes = conditioner_->get_params_buffer_size();
        const size_t budget = runtime_->phase_staging_budget_bytes();
        const size_t headroom = runtime_->phase_staging_headroom_bytes();
        if (params_bytes <= budget && headroom <= budget - params_bytes) {
            if (phase_guard.stage()) {
                LOG_INFO("LTX-2.3 text phase staged %.2f GB once for positive/negative prompts",
                         params_bytes / (1024.0 * 1024.0 * 1024.0));
            } else {
                LOG_WARN("LTX-2.3 text phase staging failed; using segmented offload");
            }
        } else {
            LOG_INFO("LTX-2.3 text phase staging skipped: weights %.2f GB + headroom %.2f GB exceed %.2f GB budget",
                     params_bytes / (1024.0 * 1024.0 * 1024.0),
                     headroom / (1024.0 * 1024.0 * 1024.0),
                     budget / (1024.0 * 1024.0 * 1024.0));
        }
    }
    ConditionerParams prompt;
    prompt.text = params.prompt != nullptr ? params.prompt : "";
    conditions->cond = conditioner_->get_learned_condition(runtime_->n_threads(), prompt);
    if (conditions->cond.empty()) {
        return set_error(error, "LTX-2.3 positive prompt encoding failed");
    }
    const float cfg = params.sample.cfg_scale == 0.f ? 1.f : params.sample.cfg_scale;
    if (cfg != 1.f) {
        prompt.text = params.negative_prompt != nullptr ? params.negative_prompt : "";
        conditions->uncond = conditioner_->get_learned_condition(runtime_->n_threads(), prompt);
        if (conditions->uncond.empty()) {
            return set_error(error, "LTX-2.3 negative prompt encoding failed");
        }
    }
    return true;
}

sd::Tensor<float> LTX2Pipeline::sample(const ed_video_generation_params_t& params,
                                       const Conditions& conditions,
                                       const sd::Tensor<float>& init_latent,
                                       const sd::Tensor<float>& noise,
                                       const sd::Tensor<float>& denoise_mask,
                                       const sd::Tensor<float>& video_positions,
                                       int audio_length,
                                       const std::vector<float>* sigma_override,
                                       const char* phase,
                                       std::string* error) {
    const int token_count = static_cast<int>(init_latent.shape()[0] * init_latent.shape()[1] * init_latent.shape()[2]);
    std::vector<float> sigmas = sigma_override != nullptr
                                    ? *sigma_override
                                    : denoiser_->get_sigmas(
                                          params.sample.steps > 0 ? params.sample.steps : kDefaultSteps,
                                          token_count,
                                          LTX2_SCHEDULER,
                                          version_);
    const int steps = static_cast<int>(sigmas.size()) - 1;
    if (sigmas.size() < 2) {
        set_error(error, "LTX-2.3 scheduler returned an invalid sigma sequence");
        return {};
    }
    PhaseParamsGuard<LTXAVModel> phase_guard(diffusion_.get());
    if (diffusion_offload_) {
        const auto context_tokens_for = [](const SDCondition& condition) {
            const auto& shape = condition.c_crossattn.shape();
            return shape.size() > 1 && shape[1] > 0 ? static_cast<int>(shape[1]) : 0;
        };
        const int context_tokens = std::max(context_tokens_for(conditions.cond),
                                            context_tokens_for(conditions.uncond));
        const size_t measured_compute = context_tokens > 0
                                            ? diffusion_->measure_compute_buffer_at(
                                                  static_cast<int>(init_latent.shape()[0]),
                                                  static_cast<int>(init_latent.shape()[1]),
                                                  static_cast<int>(init_latent.shape()[2]),
                                                  audio_length,
                                                  context_tokens,
                                                  !denoise_mask.empty())
                                            : 0;
        const size_t params_bytes = diffusion_->get_params_buffer_size();
        const size_t budget = runtime_->phase_staging_budget_bytes();
        const size_t headroom = runtime_->phase_staging_headroom_bytes(measured_compute);
        if (params_bytes <= budget && headroom <= budget - params_bytes) {
            if (phase_guard.stage()) {
                LOG_INFO("LTX-2.3 %s phase staged %.2f GB of diffusion weights once for %d steps "
                         "(compute %.2f GB, reserve %.2f GB)",
                         phase != nullptr ? phase : "base",
                         params_bytes / (1024.0 * 1024.0 * 1024.0),
                         steps,
                         measured_compute / (1024.0 * 1024.0 * 1024.0),
                         headroom / (1024.0 * 1024.0 * 1024.0));
            } else {
                LOG_WARN("LTX-2.3 %s phase staging failed; using segmented offload",
                         phase != nullptr ? phase : "base");
            }
        } else {
            LOG_INFO("LTX-2.3 %s phase staging skipped: weights %.2f GB + reserve %.2f GB exceed %.2f GB budget "
                     "(compute %.2f GB)",
                     phase != nullptr ? phase : "base",
                     params_bytes / (1024.0 * 1024.0 * 1024.0),
                     headroom / (1024.0 * 1024.0 * 1024.0),
                     budget / (1024.0 * 1024.0 * 1024.0),
                     measured_compute / (1024.0 * 1024.0 * 1024.0));
        }
    }
    sd::Tensor<float> x = denoiser_->noise_scaling(sigmas.front(), noise, init_latent);
    const float cfg = params.sample.cfg_scale == 0.f ? 1.f : params.sample.cfg_scale;
    const float frame_rate = params.fps > 0 ? static_cast<float>(params.fps) : 24.f;
    GenerationControl* control = runtime_->generation_control();

    for (int step = 0; step < steps; ++step) {
        if (control != nullptr && control->should_cancel()) {
            control->mark_cancelled();
            set_error(error, "generation cancelled");
            return {};
        }
        const float sigma = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
        const float base_timestep = denoiser_->sigma_to_t(sigma);
        sd::Tensor<float> timestep = denoise_mask.empty()
                                         ? sd::Tensor<float>::from_vector({base_timestep})
                                         : make_video_timesteps(init_latent,
                                                                denoise_mask,
                                                                base_timestep);
        sd::Tensor<float> audio_timestep = denoise_mask.empty()
                                               ? sd::Tensor<float>()
                                               : sd::Tensor<float>::from_vector({base_timestep});
        sd::Tensor<float> model_input = denoise_mask.empty()
                                            ? x
                                            : x * denoise_mask + init_latent * (1.f - denoise_mask);
        DiffusionParams diffusion_params;
        diffusion_params.x = &model_input;
        diffusion_params.timesteps = &timestep;
        diffusion_params.context = &conditions.cond.c_crossattn;
        diffusion_params.audio_timesteps = audio_timestep.empty() ? nullptr : &audio_timestep;
        diffusion_params.audio_length = audio_length;
        diffusion_params.frame_rate = frame_rate;
        diffusion_params.video_positions = video_positions.empty() ? nullptr : &video_positions;
        const bool use_cfg_parallel = !conditions.uncond.empty() &&
                                      parallel::cfg_parallel_available(runtime_->parallel_context());
        sd::Tensor<float> prediction;
        if (use_cfg_parallel) {
            const int cfg_rank = parallel::cfg_parallel_rank(runtime_->parallel_context());
            const bool local_is_uncond = cfg_rank == 0;
            LOG_INFO("LTX-2.3 CFG parallel phase=%s step=%d/%d rank=%d/2 branch=%s",
                     phase != nullptr ? phase : "base",
                     step + 1,
                     steps,
                     cfg_rank,
                     local_is_uncond ? "uncond" : "cond");
            diffusion_params.context = local_is_uncond
                                           ? &conditions.uncond.c_crossattn
                                           : &conditions.cond.c_crossattn;
            sd::Tensor<float> local_prediction = diffusion_->compute(runtime_->n_threads(),
                                                                      diffusion_params);
            std::vector<sd::Tensor<float>> gathered;
            if (local_prediction.empty() ||
                !parallel::cfg_all_gather(*runtime_->parallel_context(),
                                          local_prediction,
                                          &gathered,
                                          error) ||
                gathered.size() != 2) {
                if (error != nullptr && error->empty()) {
                    *error = sd_format("LTX-2.3 CFG parallel gather failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return {};
            }
            prediction = gathered[0] + (gathered[1] - gathered[0]) * cfg;
        } else {
            sd::Tensor<float> cond = diffusion_->compute(runtime_->n_threads(), diffusion_params);
            if (cond.empty()) {
                set_error(error, "LTX-2.3 conditional diffusion compute failed");
                return {};
            }
            prediction = std::move(cond);
            if (!conditions.uncond.empty()) {
                diffusion_params.context = &conditions.uncond.c_crossattn;
                sd::Tensor<float> uncond = diffusion_->compute(runtime_->n_threads(), diffusion_params);
                if (uncond.empty()) {
                    set_error(error, "LTX-2.3 unconditional diffusion compute failed");
                    return {};
                }
                prediction = uncond + (prediction - uncond) * cfg;
            }
        }
        if (denoise_mask.empty()) {
            x += prediction * (sigma_next - sigma);
        } else {
            sd::Tensor<float> denoised = x - prediction * sigma;
            denoised = denoised * denoise_mask + init_latent * (1.f - denoise_mask);
            x += ((x - denoised) / sigma) * (sigma_next - sigma);
        }
        diffusion_->free_compute_buffer();
        if (control != nullptr) {
            control->step_done();
        }
        LOG_INFO("LTX-2.3 %s sampling step %d/%d sigma %.6f -> %.6f",
                 phase != nullptr ? phase : "base",
                 step + 1,
                 steps,
                 sigma,
                 sigma_next);
    }
    return x;
}

std::vector<float> LTX2Pipeline::make_hires_sigmas(
    const ed_video_generation_params_t& params,
    const sd::Tensor<float>& latent,
    std::string* error) {
    if (params.hires_sigmas_count > 0) {
        if (params.hires_sigmas == nullptr || params.hires_sigmas_count < 2) {
            set_error(error, "LTX-2.3 hires sigmas requires at least two values");
            return {};
        }
        std::vector<float> sigmas(params.hires_sigmas,
                                  params.hires_sigmas + params.hires_sigmas_count);
        for (size_t i = 0; i < sigmas.size(); ++i) {
            if (!std::isfinite(sigmas[i]) || sigmas[i] < 0.f ||
                (i > 0 && sigmas[i] > sigmas[i - 1])) {
                set_error(error, "LTX-2.3 hires sigmas must be finite, non-negative, and non-increasing");
                return {};
            }
        }
        return sigmas;
    }

    const float strength = params.hires_denoising_strength;
    if (!std::isfinite(strength) || strength <= 0.f || strength > 1.f) {
        set_error(error, "LTX-2.3 hires denoising strength must be in (0, 1]");
        return {};
    }
    const int effective_steps = params.hires_steps > 0 ? params.hires_steps : 4;
    const int scheduler_steps = std::max(1, static_cast<int>(effective_steps / strength));
    const int token_count = static_cast<int>(latent.shape()[0] * latent.shape()[1] * latent.shape()[2]);
    std::vector<float> sigmas = denoiser_->get_sigmas(scheduler_steps,
                                                       token_count,
                                                       LTX2_SCHEDULER,
                                                       version_);
    size_t encoded_steps = static_cast<size_t>(scheduler_steps * strength);
    if (encoded_steps >= static_cast<size_t>(scheduler_steps)) {
        encoded_steps = static_cast<size_t>(scheduler_steps) - 1;
    }
    const size_t begin = static_cast<size_t>(scheduler_steps) - encoded_steps - 1;
    if (begin >= sigmas.size() - 1) {
        set_error(error, "LTX-2.3 hires scheduler produced an empty refine schedule");
        return {};
    }
    return std::vector<float>(sigmas.begin() + static_cast<std::ptrdiff_t>(begin),
                              sigmas.end());
}

sd::Tensor<float> LTX2Pipeline::upscale_latent(const sd::Tensor<float>& packed_latent,
                                               int audio_length,
                                               std::string* error) {
    if (!latent_upscaler_) {
        set_error(error, "LTX-2.3 hires requested without a loaded latent upscaler");
        return {};
    }
    sd::Tensor<float> video = packed_latent.shape()[3] == kVideoChannels
                                  ? packed_latent
                                  : sd::ops::slice(packed_latent, 3, 0, kVideoChannels);
    sd::Tensor<float> audio = unpack_audio(packed_latent, audio_length);
    sd::Tensor<float> unnormalized = video_vae_->un_normalize_latents(runtime_->n_threads(), video);
    if (unnormalized.empty()) {
        set_error(error, "LTX-2.3 failed to un-normalize video latent before spatial upscale");
        return {};
    }
    sd::Tensor<float> upscaled = latent_upscaler_->compute(runtime_->n_threads(), unnormalized);
    if (upscaled.empty()) {
        set_error(error, "LTX-2.3 spatial latent upscale failed");
        return {};
    }
    upscaled = video_vae_->normalize_latents(runtime_->n_threads(), upscaled);
    if (upscaled.empty()) {
        set_error(error, "LTX-2.3 failed to normalize spatially upscaled latent");
        return {};
    }
    LOG_INFO("LTX-2.3 latent spatial upscale %lldx%lldx%lld -> %lldx%lldx%lld",
             static_cast<long long>(video.shape()[0]),
             static_cast<long long>(video.shape()[1]),
             static_cast<long long>(video.shape()[2]),
             static_cast<long long>(upscaled.shape()[0]),
             static_cast<long long>(upscaled.shape()[1]),
             static_cast<long long>(upscaled.shape()[2]));
    return pack_audio_video(upscaled, audio);
}

ed_status_t LTX2Pipeline::decode(const sd::Tensor<float>& packed_latent,
                                 int audio_length,
                                 int requested_frames,
                                 int target_latent_frames,
                                 ed_video_t* out,
                                 std::string* error) {
    sd::Tensor<float> video_latent = packed_latent.shape()[3] == kVideoChannels
                                         ? packed_latent
                                         : sd::ops::slice(packed_latent, 3, 0, kVideoChannels);
    if (target_latent_frames > 0 && video_latent.shape()[2] > target_latent_frames) {
        video_latent = sd::ops::slice(video_latent, 2, 0, target_latent_frames);
    }
    sd::Tensor<float> video = video_vae_->decode(runtime_->n_threads(),
                                                 video_latent,
                                                 runtime_->vae_tiling(),
                                                 true,
                                                 runtime_->circular_x(),
                                                 runtime_->circular_y());
    if (video.empty() || video.dim() != 5) {
        set_error(error, "LTX-2.3 video VAE decode failed");
        return ED_STATUS_GENERATION_FAILED;
    }
    const int decoded_frames = static_cast<int>(video.shape()[2]);
    const int frame_count = std::min(decoded_frames, requested_frames);
    ed_image_t* frames = static_cast<ed_image_t*>(std::calloc(frame_count, sizeof(ed_image_t)));
    if (frames == nullptr) {
        set_error(error, "failed to allocate LTX-2.3 output frames");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    const int width = static_cast<int>(video.shape()[0]);
    const int height = static_cast<int>(video.shape()[1]);
    const int channels = static_cast<int>(video.shape()[3]);
    const size_t pixels = static_cast<size_t>(width) * height;
    for (int frame = 0; frame < frame_count; ++frame) {
        frames[frame].width = width;
        frames[frame].height = height;
        frames[frame].channels = channels;
        frames[frame].data = static_cast<uint8_t*>(std::malloc(pixels * channels));
        if (frames[frame].data == nullptr) {
            for (int i = 0; i < frame; ++i) std::free(frames[i].data);
            std::free(frames);
            set_error(error, "failed to allocate LTX-2.3 frame pixels");
            return ED_STATUS_OUT_OF_MEMORY;
        }
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            const int x = static_cast<int>(pixel % width);
            const int y = static_cast<int>(pixel / width);
            for (int channel = 0; channel < channels; ++channel) {
                frames[frame].data[pixel * channels + channel] =
                    to_u8(video.index(x, y, frame, channel, 0));
            }
        }
    }
    out->frames = frames;
    out->frame_count = frame_count;

    if (audio_vae_ && audio_length > 0) {
        sd::Tensor<float> audio_latent = unpack_audio(packed_latent, audio_length);
        sd::Tensor<float> waveform = audio_vae_->decode(runtime_->n_threads(), audio_latent);
        if (!waveform.empty()) {
            const int64_t samples = waveform.shape()[0];
            const int channels_out = waveform.dim() > 1 ? static_cast<int>(waveform.shape()[1]) : 1;
            if (samples > 0 && samples <= std::numeric_limits<int>::max() && channels_out > 0) {
                float* audio = static_cast<float*>(std::malloc(
                    static_cast<size_t>(samples) * channels_out * sizeof(float)));
                if (audio != nullptr) {
                    for (int64_t sample_index = 0; sample_index < samples; ++sample_index) {
                        for (int channel = 0; channel < channels_out; ++channel) {
                            audio[sample_index * channels_out + channel] =
                                std::clamp(waveform.index(sample_index, channel, 0, 0), -1.f, 1.f);
                        }
                    }
                    out->audio = audio;
                    out->audio_sample_count = static_cast<int>(samples);
                    out->audio_channels = channels_out;
                    out->audio_sample_rate = audio_vae_->output_sample_rate();
                }
            }
        } else {
            LOG_WARN("LTX-2.3 audio VAE decode failed; returning video without audio");
        }
    }
    return ED_STATUS_OK;
}

ed_status_t LTX2Pipeline::generate_video(const ed_video_generation_params_t* params,
                                         ed_video_t* out,
                                         std::string* error) {
    if (params == nullptr || out == nullptr) {
        set_error(error, "LTX-2.3 video parameters or output are null");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    *out = {};
    if (!ready_ || runtime_ == nullptr || !conditioner_ || !diffusion_ || !video_vae_) {
        set_error(error, "LTX-2.3 pipeline is not ready");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (params->width <= 0 || params->height <= 0 ||
        params->width > std::numeric_limits<int>::max() - 31 ||
        params->height > std::numeric_limits<int>::max() - 31) {
        set_error(error, "LTX-2.3 width and height must be positive");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const int conditioning_width = params->width;
    const int conditioning_height = params->height;
    ed_video_generation_params_t resolved_params = *params;
    resolved_params.width = ((resolved_params.width + 31) / 32) * 32;
    resolved_params.height = ((resolved_params.height + 31) / 32) * 32;
    if (resolved_params.width != params->width || resolved_params.height != params->height) {
        LOG_WARN("align LTX-2.3 generation request up %dx%d to %dx%d (multiple=32)",
                 params->width,
                 params->height,
                 resolved_params.width,
                 resolved_params.height);
    }
    params = &resolved_params;
    if (params->frames <= 0 || (params->frames - 1) % 8 != 0) {
        set_error(error, "LTX-2.3 frame count must satisfy 8k + 1");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (params->control_frame_count > 0 || params->ref_image_count > 0 ||
        params->ref_video_count > 0 || params->ref_audio_count > 0) {
        set_error(error, "LTX-2.3 control and reference inputs are not supported");
        return ED_STATUS_UNSUPPORTED;
    }
    const ed_sampler_t sampler = params->sample.sampler == ED_SAMPLER_AUTO
                                     ? default_sample_method()
                                     : params->sample.sampler;
    const ed_scheduler_t scheduler = params->sample.scheduler == ED_SCHEDULER_AUTO
                                         ? default_scheduler(sampler)
                                         : params->sample.scheduler;
    if (sampler != ED_SAMPLER_EULER || scheduler != ED_SCHEDULER_LTX2) {
        set_error(error, "LTX-2.3 currently requires Euler sampling with the ltx2 scheduler");
        return ED_STATUS_UNSUPPORTED;
    }
    if (params->sample.cache_mode != ED_CACHE_DISABLED) {
        set_error(error, "LTX-2.3 does not support cache acceleration yet");
        return ED_STATUS_UNSUPPORTED;
    }
    if (parallel::cfg_parallel_available(runtime_->parallel_context()) &&
        (params->sample.cfg_scale == 0.f || params->sample.cfg_scale == 1.f)) {
        set_error(error, "LTX-2.3 CFG parallelism requires cfg-scale different from 1");
        return ED_STATUS_INVALID_ARGUMENT;
    }

    const int steps = params->sample.steps > 0 ? params->sample.steps : kDefaultSteps;
    if (GenerationControl* control = runtime_->generation_control()) {
        const int refine_steps = params->hires_enabled
                                     ? (params->hires_sigmas_count > 1
                                            ? params->hires_sigmas_count - 1
                                            : std::max(1, params->hires_steps))
                                     : 0;
        control->start(steps + refine_steps);
    }
    if (params->hires_enabled && !latent_upscaler_) {
        set_error(error, "LTX-2.3 hires requires a latent upscaler in the context");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const int64_t seed = resolve_seed(params->seed);
    rng_->manual_seed(seed);
    Latents latents;
    if (!prepare_latents(*params,
                         conditioning_width,
                         conditioning_height,
                         &latents,
                         error)) {
        return ED_STATUS_GENERATION_FAILED;
    }
    const int fps = params->fps > 0 ? params->fps : 24;
    sd::Tensor<float> noise = sd::randn_like<float>(latents.init, rng_);

    const bool has_init = params->init_image != nullptr && params->init_image->data != nullptr;
    const bool has_end = params->end_image != nullptr && params->end_image->data != nullptr;
    const char* mode = has_init && has_end ? "FLF2V" : has_init ? "I2V" : has_end ? "E2V" : "T2V";
    LOG_INFO("LTX-2.3 %s %dx%d frames=%d fps=%d latent=%dx%dx%d audio=%d seed=%lld",
             mode,
             params->width, params->height, params->frames, fps,
             static_cast<int>(latents.init.shape()[0]),
             static_cast<int>(latents.init.shape()[1]),
             static_cast<int>(latents.init.shape()[2]),
             latents.audio_length,
             static_cast<long long>(seed));
    Conditions conditions;
    if (!prepare_conditions(*params, &conditions, error)) {
        return ED_STATUS_GENERATION_FAILED;
    }
    sd::Tensor<float> final_latent = sample(*params,
                                            conditions,
                                            latents.init,
                                            noise,
                                            latents.denoise_mask,
                                            latents.video_positions,
                                            latents.audio_length,
                                            nullptr,
                                            "base",
                                            error);
    if (final_latent.empty()) {
        return ED_STATUS_GENERATION_FAILED;
    }
    if (params->hires_enabled) {
        final_latent = upscale_latent(final_latent, latents.audio_length, error);
        if (final_latent.empty()) {
            return ED_STATUS_GENERATION_FAILED;
        }

        ed_video_generation_params_t refine_params = *params;
        refine_params.width = static_cast<int>(final_latent.shape()[0]) * 32;
        refine_params.height = static_cast<int>(final_latent.shape()[1]) * 32;
        refine_params.strength = 1.f;
        Latents refine_conditioning;
        if (!prepare_latents(refine_params,
                             conditioning_width,
                             conditioning_height,
                             &refine_conditioning,
                             error)) {
            return ED_STATUS_GENERATION_FAILED;
        }
        if (refine_conditioning.init.shape() != final_latent.shape()) {
            set_error(error, "LTX-2.3 hires conditioning shape does not match upscaled latent");
            return ED_STATUS_GENERATION_FAILED;
        }
        if (!refine_conditioning.denoise_mask.empty()) {
            final_latent = final_latent * refine_conditioning.denoise_mask +
                           refine_conditioning.init * (1.f - refine_conditioning.denoise_mask);
        }
        latents.video_positions = std::move(refine_conditioning.video_positions);
        latents.denoise_mask = std::move(refine_conditioning.denoise_mask);
        latents.audio_length = refine_conditioning.audio_length;
        latents.video_target_frame_count = refine_conditioning.video_target_frame_count;
        latents.video_conditioning_frame_count = refine_conditioning.video_conditioning_frame_count;

        std::vector<float> hires_sigmas = make_hires_sigmas(refine_params, final_latent, error);
        if (hires_sigmas.size() < 2) {
            return ED_STATUS_GENERATION_FAILED;
        }
        LOG_INFO("LTX-2.3 hires refine %dx%d steps=%zu strength=%.3f%s",
                 refine_params.width,
                 refine_params.height,
                 hires_sigmas.size() - 1,
                 params->hires_denoising_strength,
                 params->hires_sigmas_count > 0 ? " custom_sigmas=true" : "");
        sd::Tensor<float> hires_noise = sd::randn_like<float>(final_latent, rng_);
        final_latent = sample(refine_params,
                              conditions,
                              final_latent,
                              hires_noise,
                              latents.denoise_mask,
                              latents.video_positions,
                              latents.audio_length,
                              &hires_sigmas,
                              "hires",
                              error);
        if (final_latent.empty()) {
            return ED_STATUS_GENERATION_FAILED;
        }
    }
    const int target_latent_frames = latents.video_conditioning_frame_count > 0
                                         ? latents.video_target_frame_count
                                         : 0;
    if (runtime_->parallel_context() != nullptr &&
        !runtime_->parallel_context()->is_root()) {
        LOG_INFO("LTX-2.3 rank=%d skipping video/audio VAE decode and output writing",
                 runtime_->parallel_context()->rank());
        return ED_STATUS_OK;
    }
    LOG_INFO("LTX-2.3 rank=%d executing video/audio VAE decode and output writing",
             runtime_->parallel_context() != nullptr ? runtime_->parallel_context()->rank() : 0);
    return decode(final_latent,
                  latents.audio_length,
                  params->frames,
                  target_latent_frames,
                  out,
                  error);
}

}  // namespace edgedit
