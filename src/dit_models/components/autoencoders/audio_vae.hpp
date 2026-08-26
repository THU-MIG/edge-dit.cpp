#ifndef __SD_MODEL_VAE_AUDIO_VAE_HPP__
#define __SD_MODEL_VAE_AUDIO_VAE_HPP__

#include "backend/ggml/ggml_extend.hpp"

struct AudioVAERunner : public GGMLRunner {
    AudioVAERunner(ggml_backend_t backend,
                   bool offload_params_to_cpu = false)
        : GGMLRunner(backend, offload_params_to_cpu) {}

    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_mem_size()                                         = 0;
    virtual std::string get_desc()                                               = 0;
    virtual sd::Tensor<float> encode(int n_threads,
                                     const sd::Tensor<float>& waveform) {
        ED_UNUSED(n_threads);
        ED_UNUSED(waveform);
        return {};
    }
    virtual sd::Tensor<float> decode(int n_threads,
                                     const sd::Tensor<float>& latent_tensor) = 0;
    virtual int input_sample_rate() const {
        return output_sample_rate();
    }
    virtual int output_sample_rate() const = 0;
};

#endif  // __SD_MODEL_VAE_AUDIO_VAE_HPP__
