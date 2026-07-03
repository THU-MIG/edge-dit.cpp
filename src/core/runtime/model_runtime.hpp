#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "backend/ggml/ggml_extend.hpp"
#include "ggml-backend.h"
#include "edge-dit.h"
#include "parallel/parallel_context.hpp"
#include "parallel/process_group.hpp"
#include "utils/rng.hpp"


namespace edgedit {

using ed_ctx_params_t = ed_context_params_t;
using sample_method_t = ed_sampler_t;
using scheduler_t = ed_scheduler_t;

constexpr sample_method_t EULER_SAMPLE_METHOD = ED_SAMPLER_EULER;
constexpr sample_method_t EULER_A_SAMPLE_METHOD = ED_SAMPLER_EULER_A;
constexpr sample_method_t HEUN_SAMPLE_METHOD = ED_SAMPLER_HEUN;
constexpr sample_method_t DPM2_SAMPLE_METHOD = ED_SAMPLER_DPM2;
constexpr sample_method_t DPMPP2S_A_SAMPLE_METHOD = ED_SAMPLER_DPM_PLUS_PLUS_2S_A;
constexpr sample_method_t DPMPP2M_SAMPLE_METHOD = ED_SAMPLER_DPM_PLUS_PLUS_2M;
constexpr sample_method_t DPMPP2Mv2_SAMPLE_METHOD = ED_SAMPLER_DPM_PLUS_PLUS_2M_V2;
constexpr sample_method_t IPNDM_SAMPLE_METHOD = ED_SAMPLER_IPNDM;
constexpr sample_method_t IPNDM_V_SAMPLE_METHOD = ED_SAMPLER_IPNDM_V;
constexpr sample_method_t LCM_SAMPLE_METHOD = ED_SAMPLER_LCM;
constexpr sample_method_t TCD_SAMPLE_METHOD = ED_SAMPLER_TCD;
constexpr sample_method_t DDIM_TRAILING_SAMPLE_METHOD = ED_SAMPLER_DDIM_TRAILING;
constexpr sample_method_t RES_MULTISTEP_SAMPLE_METHOD = ED_SAMPLER_RES_MULTISTEP;
constexpr sample_method_t RES_2S_SAMPLE_METHOD = ED_SAMPLER_RES_2S;
constexpr sample_method_t ER_SDE_SAMPLE_METHOD = ED_SAMPLER_ER_SDE;

constexpr scheduler_t DISCRETE_SCHEDULER = ED_SCHEDULER_DISCRETE;
constexpr scheduler_t KARRAS_SCHEDULER = ED_SCHEDULER_KARRAS;
constexpr scheduler_t EXPONENTIAL_SCHEDULER = ED_SCHEDULER_EXPONENTIAL;
constexpr scheduler_t AYS_SCHEDULER = ED_SCHEDULER_AYS;
constexpr scheduler_t GITS_SCHEDULER = ED_SCHEDULER_GITS;
constexpr scheduler_t SGM_UNIFORM_SCHEDULER = ED_SCHEDULER_SGM_UNIFORM;
constexpr scheduler_t SIMPLE_SCHEDULER = ED_SCHEDULER_SIMPLE;
constexpr scheduler_t SMOOTHSTEP_SCHEDULER = ED_SCHEDULER_SMOOTHSTEP;
constexpr scheduler_t KL_OPTIMAL_SCHEDULER = ED_SCHEDULER_KL_OPTIMAL;
constexpr scheduler_t LCM_SCHEDULER = ED_SCHEDULER_LCM;
constexpr scheduler_t BONG_TANGENT_SCHEDULER = ED_SCHEDULER_BONG_TANGENT;

struct RuntimeBackends {
    ggml_backend_t backend = nullptr;
    ggml_backend_t clip_backend = nullptr;
    ggml_backend_t vae_backend = nullptr;
    ggml_backend_t control_net_backend = nullptr;

    bool clip_owns_backend = false;
    bool vae_owns_backend = false;
    bool control_net_owns_backend = false;
};

class ModelRuntime final {
public:
    ModelRuntime() = default;
    ~ModelRuntime();

    ModelRuntime(const ModelRuntime&) = delete;
    ModelRuntime& operator=(const ModelRuntime&) = delete;

    bool init(const ed_context_params_t& params, std::string* error);
    bool init(const ed_context_params_t* params, std::string* error);
    void reset();
    void set_parallel_context(parallel::ParallelContext* context) { parallel_context_ = context; }

    bool ready() const { return ready_; }
    bool is_ready() const { return ready_; }

    int n_threads() const { return n_threads_; }
    bool use_mmap() const { return use_mmap_; }
    bool offload_params_to_cpu() const { return offload_params_to_cpu_; }
    bool free_params_immediately() const { return free_params_immediately_; }
    float max_vram() const { return max_vram_; }
    size_t max_graph_vram_bytes() const { return max_graph_vram_bytes_; }
    bool flash_attention() const { return flash_attention_; }
    bool circular_x() const { return circular_x_; }
    bool circular_y() const { return circular_y_; }
    const ed_tiling_params_t& vae_tiling() const { return vae_tiling_; }
    bool parallel_enabled() const { return parallel_context_ != nullptr && parallel_context_->enabled(); }

    ggml_backend_t backend() const { return backends_.backend; }
    ggml_backend_t clip_backend() const { return backends_.clip_backend; }
    ggml_backend_t vae_backend() const { return backends_.vae_backend; }
    ggml_backend_t control_net_backend() const { return backends_.control_net_backend; }

    RNG& rng() { return *rng_; }
    RNG& sampler_rng() { return *sampler_rng_; }
    std::shared_ptr<RNG> rng_ptr() const { return rng_; }
    std::shared_ptr<RNG> sampler_rng_ptr() const { return sampler_rng_; }
    parallel::ParallelContext* parallel_context() const { return parallel_context_; }
    std::shared_ptr<parallel::ProcessGroup> process_group_ref() const {
        if (parallel_context_ == nullptr || !parallel_context_->enabled()) {
            return nullptr;
        }

        return std::shared_ptr<parallel::ProcessGroup>(
            &parallel_context_->world_group(),
            [](parallel::ProcessGroup*) {
            }
        );
    }
private:
    bool ready_ = false;

    int n_threads_ = 0;
    bool use_mmap_ = false;
    bool offload_params_to_cpu_ = false;
    bool free_params_immediately_ = false;

    float max_vram_ = 0.0f;
    size_t max_graph_vram_bytes_ = 0;

    bool flash_attention_ = false;
    bool circular_x_ = false;
    bool circular_y_ = false;
    ed_tiling_params_t vae_tiling_ = {};

    RuntimeBackends backends_;
    parallel::ParallelContext* parallel_context_ = nullptr;

    std::shared_ptr<RNG> rng_;
    std::shared_ptr<RNG> sampler_rng_;

    bool init_threads(const ed_context_params_t& params, std::string* error);
    bool init_flags(const ed_context_params_t& params, std::string* error);
    bool init_backends(const ed_context_params_t& params, std::string* error);
    bool init_rng(const ed_context_params_t& params, std::string* error);

    void release_backends();
    bool fail(std::string* error, const std::string& msg);
};

} // namespace edgedit
