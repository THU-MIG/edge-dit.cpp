#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/pipelines/dit_pipeline.hpp"
#include "dit_models/components/scheduler/denoiser.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"

struct Conditioner;
struct FrozenCLIPVisionEmbedder;
struct VAE;
struct RNG;

namespace edgedit {

class WanPipeline final : public DiTPipeline {
public:
    explicit WanPipeline(SDVersion version = VERSION_WAN2);
    ~WanPipeline() override = default;

    const char* name() const override { return "wan"; }

    bool prepare(const ed_context_params_t& params,
                 ModelRuntime& runtime,
                 const ModelLoader& loader,
                 PipelineTensorRegistry& registry,
                 std::string* error) override;

    void mark_ready() override;

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
    ed_scheduler_t default_scheduler(ed_sampler_t method) const override;

private:
    struct WanVideoConditionPack {
        SDCondition cond;
        SDCondition uncond;
        SDCondition img_cond;
        SDCondition id_cond;
    };

private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    SDVersion version_ = VERSION_WAN2;

    std::shared_ptr<Conditioner> conditioner_;
    std::shared_ptr<DiffusionModel> diffusion_;
    std::shared_ptr<DiffusionModel> high_noise_diffusion_;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision_;
    std::shared_ptr<VAE> vae_;
    std::shared_ptr<VAE> preview_vae_;

    std::shared_ptr<Denoiser> denoiser_;
    std::shared_ptr<RNG> rng_;
    std::shared_ptr<RNG> sampler_rng_;

    bool using_tae_for_main_ = false;
    float default_flow_shift_ = 5.0f;

    bool build_components(const ModelLoader& loader, std::string* error);
    bool register_tensors(PipelineTensorRegistry& registry, std::string* error);
    void configure_runtime_flags();
    void build_ignore_tensors(PipelineTensorRegistry& registry) const;

    bool init_sampling_runtime(std::string* error);
    void set_flow_shift(float flow_shift);
    bool is_flow_denoiser() const;

    int vae_scale_factor() const;
    int latent_channels() const;
    int latent_frames(int frames) const;
    int image_seq_len(int width, int height) const;
    void prewarm_wan_cudnn_sdpa(int width, int height, int frames) const;
    void prewarm_wan_sp_comm(int width, int height, int frames) const;

    bool validate_video_params(const ed_video_generation_params_t* params,
                               std::string* error) const;

    sd::Tensor<float> generate_init_latent(int width, int height, int frames) const;

    bool prepare_text_conditions(const ed_video_generation_params_t* params,
                                 const sd::Tensor<float>& concat_latent,
                                 const sd::Tensor<float>& clip_vision_output,
                                 WanVideoConditionPack* conditions,
                                 std::string* error);

    std::vector<float> build_sigmas(const ed_sample_params_t& params,
                                    int width,
                                    int height,
                                    int total_steps) const;

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep,
                                                const sd::Tensor<float>& init_latent,
                                                const sd::Tensor<float>& denoise_mask) const;

    void adjust_sample_step_scalings(int shifted_timestep,
                                     const std::vector<float>& timesteps,
                                     float c_in,
                                     float* c_skip,
                                     float* c_out) const;

    sd::Tensor<float> run_condition(const std::shared_ptr<DiffusionModel>& model,
                                    DiffusionParams* diffusion_params,
                                    const SDCondition& condition,
                                    const sd::Tensor<float>* c_concat_override,
                                    const std::vector<int>* skip_layers,
                                    std::string* error);

    sd::Tensor<float> euler_denoise(const std::shared_ptr<DiffusionModel>& model,
                                    const sd::Tensor<float>& x_start,
                                    const std::vector<float>& sigmas,
                                    const WanVideoConditionPack& conditions,
                                    const ed_sample_params_t& sample_params,
                                    const sd::Tensor<float>& init_latent,
                                    const sd::Tensor<float>& denoise_mask,
                                    const sd::Tensor<float>& vace_context,
                                    float vace_strength,
                                    std::string* error);

    sd::Tensor<float> sample_video_latent(const ed_video_generation_params_t* params,
                                          const WanVideoConditionPack& conditions,
                                          const sd::Tensor<float>& init_latent,
                                          const sd::Tensor<float>& noise,
                                          const sd::Tensor<float>& denoise_mask,
                                          const sd::Tensor<float>& vace_context,
                                          std::string* error);

    ed_status_t decode_video_latent(const sd::Tensor<float>& latent,
                                    const ed_tiling_params_t& tiling,
                                    ed_video_t* out,
                                    std::string* error);

    static bool has_prefix(const ModelLoader& loader, const std::string& prefix);
    static bool set_error(std::string* error, const std::string& message);
    static int64_t resolve_seed(int64_t seed);
};

}  // namespace edgedit
