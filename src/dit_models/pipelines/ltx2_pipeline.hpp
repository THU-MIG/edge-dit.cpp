#pragma once

#include <memory>
#include <string>

#include "dit_models/pipelines/dit_pipeline.hpp"
#include "dit_models/diffusion_model.hpp"
#include "dit_models/components/autoencoders/ltx_audio_vae.hpp"
#include "dit_models/components/autoencoders/ltx_vae.hpp"
#include "dit_models/components/scheduler/denoiser.hpp"
#include "dit_models/components/super_resolution/ltx_latent_upscaler.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"

struct RNG;

namespace edgedit {

class LTX2Pipeline final : public DiTPipeline {
public:
    explicit LTX2Pipeline(SDVersion version = VERSION_LTXAV);
    ~LTX2Pipeline() override = default;

    const char* name() const override { return "ltx2"; }

    bool prepare(const ed_context_params_t& params,
                 ModelRuntime& runtime,
                 const ModelLoader& loader,
                 PipelineTensorRegistry& registry,
                 std::string* error) override;
    bool prepare_memory_plan(const ed_context_params_t& params,
                             ModelRuntime& runtime,
                             ModelLoader& loader,
                             std::string* error) override;
    void mark_ready() override { ready_ = true; }

    ed_status_t generate_image(const ed_image_generation_params_t* params,
                               ed_image_batch_t* out,
                               std::string* error) override;
    ed_status_t generate_video(const ed_video_generation_params_t* params,
                               ed_video_t* out,
                               std::string* error) override;

    SDVersion version() const override { return version_; }
    bool ready() const override { return ready_; }
    bool supports_image_generation() const override { return false; }
    bool supports_video_generation() const override { return ready_; }
    ed_sampler_t default_sample_method() const override { return ED_SAMPLER_EULER; }
    ed_scheduler_t default_scheduler(ed_sampler_t) const override { return ED_SCHEDULER_LTX2; }

private:
    struct Conditions {
        SDCondition cond;
        SDCondition uncond;
    };

    struct Latents {
        sd::Tensor<float> init;
        sd::Tensor<float> denoise_mask;
        sd::Tensor<float> video_positions;
        int audio_length = 0;
        int video_target_frame_count = 0;
        int video_conditioning_frame_count = 0;
    };

    bool ready_ = false;
    bool has_audio_vae_ = false;
    bool has_video_vae_encoder_ = false;
    bool diffusion_offload_ = false;
    bool text_offload_ = false;
    SDVersion version_ = VERSION_LTXAV;
    ModelRuntime* runtime_ = nullptr;

    std::shared_ptr<LTXAVEmbedder> conditioner_;
    std::shared_ptr<LTXAVModel> diffusion_;
    std::shared_ptr<LTXVideoVAE> video_vae_;
    std::shared_ptr<LTXV::LTXAudioVAERunner> audio_vae_;
    std::shared_ptr<LTXVUpsampler::LatentUpsamplerRunner> latent_upscaler_;
    std::shared_ptr<FluxFlowDenoiser> denoiser_;
    std::shared_ptr<RNG> rng_;

    static bool set_error(std::string* error, const std::string& message);
    static int64_t resolve_seed(int64_t seed);
    static bool has_prefix(const ModelLoader& loader, const std::string& prefix);

    bool load_latent_upscaler(const ed_context_params_t& params,
                              ModelLoader* loader,
                              bool offload_params_to_cpu,
                              std::string* error);
    std::vector<float> make_hires_sigmas(const ed_video_generation_params_t& params,
                                         const sd::Tensor<float>& latent,
                                         std::string* error);
    sd::Tensor<float> upscale_latent(const sd::Tensor<float>& packed_latent,
                                     int audio_length,
                                     std::string* error);

    bool prepare_conditions(const ed_video_generation_params_t& params,
                            Conditions* conditions,
                            std::string* error);
    bool prepare_latents(const ed_video_generation_params_t& params,
                         int conditioning_width,
                         int conditioning_height,
                         Latents* latents,
                         std::string* error);
    sd::Tensor<float> sample(const ed_video_generation_params_t& params,
                             const Conditions& conditions,
                             const sd::Tensor<float>& init_latent,
                             const sd::Tensor<float>& noise,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& video_positions,
                             int audio_length,
                             const std::vector<float>* sigma_override,
                             const char* phase,
                             std::string* error);
    ed_status_t decode(const sd::Tensor<float>& packed_latent,
                       int audio_length,
                       int requested_frames,
                       int target_latent_frames,
                       ed_video_t* out,
                       std::string* error);
};

}  // namespace edgedit
