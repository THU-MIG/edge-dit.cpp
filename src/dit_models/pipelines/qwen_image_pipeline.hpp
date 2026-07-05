#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "dit_models/pipelines/dit_pipeline.hpp"

struct Conditioner;
struct VAE;

namespace Qwen {
struct QwenImageRunner;
}

namespace edgedit {

class QwenImagePipeline final : public DiTPipeline {
public:
    explicit QwenImagePipeline(SDVersion version = VERSION_QWEN_IMAGE);
    ~QwenImagePipeline() override;

    const char* name() const override { return "qwen-image"; }

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

    bool supports_image_generation() const override;
    bool supports_video_generation() const override;

    ed_sampler_t default_sample_method() const override;
    ed_scheduler_t default_scheduler(ed_sampler_t method) const override;

private:
    bool ready_ = false;
    bool runtime_weights_loaded_ = false;
    bool diffusion_bf16_ = false;
    ModelRuntime* runtime_ = nullptr;
    SDVersion version_ = VERSION_QWEN_IMAGE;

    std::shared_ptr<Conditioner> conditioner_;
    std::shared_ptr<VAE> vae_;
    std::unique_ptr<Qwen::QwenImageRunner> diffusion_;

    void reset();
    bool has_prefix(const ModelLoader& loader, const std::string& prefix) const;
    bool build_components(const ed_context_params_t& params,
                          const ModelLoader& loader,
                          std::string* error);
    bool register_tensors(PipelineTensorRegistry& registry, std::string* error);
    bool can_generate_image() const;
    bool validate_image_params(const ed_image_generation_params_t* params,
                               std::string* error) const;
    bool generate_one_image(const ed_image_generation_params_t* params,
                            int batch_index,
                            int n_threads,
                            ed_image_t* image,
                            std::string* error);
};

}  // namespace edgedit
