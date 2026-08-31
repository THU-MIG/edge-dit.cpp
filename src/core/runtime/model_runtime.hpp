#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/ggml/ggml_extend.hpp"
#include "ggml-backend.h"
#include "edge-dit.h"
#include "parallel/parallel_context.hpp"
#include "parallel/process_group.hpp"
#include "utils/rng.hpp"


class ModelLoader;

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
constexpr scheduler_t LTX2_SCHEDULER = ED_SCHEDULER_LTX2;

struct RuntimeBackends {
    ggml_backend_t backend = nullptr;
    ggml_backend_t clip_backend = nullptr;
    ggml_backend_t vae_backend = nullptr;
    ggml_backend_t control_net_backend = nullptr;

    bool clip_owns_backend = false;
    bool vae_owns_backend = false;
    bool control_net_owns_backend = false;
};

struct GenerationControl {
    std::atomic<bool> active{false};
    std::atomic<bool> cancel_requested{false};
    std::atomic<bool> cancelled{false};
    std::atomic<int> current_step{0};
    std::atomic<int> total_steps{0};

    void reset_idle() {
        active.store(false, std::memory_order_relaxed);
        cancel_requested.store(false, std::memory_order_relaxed);
        cancelled.store(false, std::memory_order_relaxed);
        current_step.store(0, std::memory_order_relaxed);
        total_steps.store(0, std::memory_order_relaxed);
    }

    void start(int total) {
        current_step.store(0, std::memory_order_relaxed);
        total_steps.store(total > 0 ? total : 0, std::memory_order_relaxed);
        cancel_requested.store(false, std::memory_order_relaxed);
        cancelled.store(false, std::memory_order_relaxed);
        active.store(true, std::memory_order_relaxed);
    }

    void request_cancel() {
        cancel_requested.store(true, std::memory_order_relaxed);
    }

    bool should_cancel() const {
        return active.load(std::memory_order_relaxed) &&
               cancel_requested.load(std::memory_order_relaxed);
    }

    void mark_cancelled() {
        cancelled.store(true, std::memory_order_relaxed);
    }

    bool was_cancelled() const {
        return cancelled.load(std::memory_order_relaxed);
    }

    void step_done() {
        current_step.fetch_add(1, std::memory_order_relaxed);
    }
};

class ModelRuntime final {
public:
    using ComponentMemoryEstimator = std::function<size_t(const ::ModelLoader&,
                                                           const std::vector<std::string>&)>;

    ModelRuntime() = default;
    ~ModelRuntime();

    ModelRuntime(const ModelRuntime&) = delete;
    ModelRuntime& operator=(const ModelRuntime&) = delete;

    bool init(const ed_context_params_t& params, std::string* error);
    bool init(const ed_context_params_t* params, std::string* error);
    void reset();
    void set_parallel_context(parallel::ParallelContext* context) { parallel_context_ = context; }
    void set_generation_control(GenerationControl* control) { generation_control_ = control; }

    bool ready() const { return ready_; }
    bool is_ready() const { return ready_; }

    int n_threads() const { return n_threads_; }
    bool use_mmap() const { return use_mmap_; }
    bool offload_params_to_cpu() const { return offload_params_to_cpu_; }
    // Text encoder should offload (weights on CPU, staged to GPU per encode) when
    // either the global offload flag or the TE-specific flag is set.
    bool clip_offload_params_to_cpu() const { return offload_params_to_cpu_ || text_encoder_offload_; }
    bool minimax_h3_stage_lifecycle() const { return minimax_h3_stage_lifecycle_; }
    // DiT should offload (weights on CPU, staged to GPU per step) when either the
    // global offload flag or the DiT-specific flag is set.
    bool dit_offload_params_to_cpu() const { return offload_params_to_cpu_ || dit_offload_; }
    // VAE should offload (weights on CPU, staged to GPU per decode) when either the
    // global offload flag or the VAE-specific flag is set.
    bool vae_offload_params_to_cpu() const { return offload_params_to_cpu_ || vae_offload_; }
    bool auto_allocate() const { return auto_allocate_; }
    bool auto_fit() const { return auto_fit_; }
    bool free_params_immediately() const { return free_params_immediately_; }
    float max_vram() const { return max_vram_; }
    size_t max_graph_vram_bytes() const { return max_graph_vram_bytes_; }
    void set_max_graph_vram_bytes(size_t bytes) { max_graph_vram_bytes_ = bytes; }
    // Auto-allocate per-component placement (only active under --auto-allocate). Given a
    // component's weight prefix, decide resident-on-GPU vs offload-to-CPU by comparing its
    // quantized weight size against the running budget tally (remaining_free_bytes, seeded
    // with min(--max-vram, live free)). Resident debits the tally and accumulates into an
    // internal resident total. Returns true if the component must offload. Outside
    // auto-allocate mode, returns the legacy global offload flag (no behavior change).
    bool plan_component_offload(const ::ModelLoader& loader,
                                const std::string& weight_prefix,
                                size_t& remaining_free_bytes);
    bool plan_component_offload(const ::ModelLoader& loader,
                                const std::vector<std::string>& weight_prefixes,
                                size_t& remaining_free_bytes);
    // Pipelines with runner-specific type rules can provide the exact materialized
    // parameter size. A zero return value falls back to the storage-map estimate.
    void set_component_memory_estimator(ComponentMemoryEstimator estimator) {
        component_memory_estimator_ = std::move(estimator);
    }
    // Auto-fit scheduler (only active under --auto-fit). Before offload planning, force
    // the DiT ("model.diffusion_model") to the highest quant level in the ladder
    // q8_0 -> q4_k that keeps it resident within the VRAM budget, ignoring the user's
    // --type. On a 4090 a resident lower-quant DiT beats a higher-quant offloaded one
    // (q4 ~= q8 in speed/quality; offload thrashes weights every step). If even q4_k does
    // not fit, leaves the DiT at q8_0 and lets plan_component_offload offload it. No-op
    // outside auto-fit. Takes a NON-const loader.
    void replan_dit_quant_for_budget(::ModelLoader& loader);
    // Call once after all components are decided: sets the graph VRAM budget for offloaded
    // components to (effective_budget - resident_total - compute headroom). No-op outside
    // auto-allocate. Resets the internal resident accumulator for the next model load.
    void finalize_auto_segment_budget(size_t effective_budget_bytes,
                                      size_t additional_slack_bytes = 0);
    void reset_auto_allocate_state() { resident_bytes_total_ = 0; any_component_offloaded_ = false; }
    // Hard-cap budget = min(user --max-vram, live free VRAM). If no --max-vram, = live free.
    // Used to seed the auto-allocate tally and finalize_auto_segment_budget. 0 if CPU backend.
    size_t effective_budget_bytes() const;
    // Bytes available for temporarily staging one offloaded component for an
    // entire execution phase. Unlike effective_budget_bytes(), this subtracts
    // components already committed resident when a user VRAM cap is active.
    size_t phase_staging_budget_bytes() const;
    // Return the reserved bytes for a whole-component execution phase. When the
    // caller has measured the phase's activation buffer for the actual workload,
    // use that measurement instead of the generic resident-component estimate.
    size_t phase_staging_headroom_bytes(size_t phase_compute_bytes = 0) const;
    // Segment-VRAM budget for the text encoder. When the TE is offloaded, returns a budget
    // guaranteed below the TE's own weight size (te_params_bytes) so its graph-cut segments
    // don't collapse into one whole-TE stage (the OOM cause). Resident TE returns the global
    // budget unchanged. Pass conditioner_->get_params_buffer_size() as te_params_bytes.
    size_t text_encoder_segment_budget(size_t te_params_bytes, bool component_offloaded = false) const;
    // Target generation resolution for compute-buffer measurement (0 = unset).
    int fit_width() const { return fit_width_; }
    int fit_height() const { return fit_height_; }
    int fit_frames() const { return fit_frames_; }
    int fit_fps() const { return fit_fps_; }
    // Auto-fit/auto-allocate: cache the measured DiT compute-buffer size (bytes) so the
    // placement judgments use the real activation footprint instead of the fixed 4 GiB
    // constant. Set once in the pipeline's prepare(), before plan_component_offload. 0
    // clears it (fall back to fixed constant). See resident_headroom_bytes().
    void set_measured_dit_headroom(size_t bytes) { measured_dit_headroom_ = bytes; }
    // Headroom reserved for a resident component's own compute buffer + activations. Uses
    // the measured DiT value (× fragmentation margin) when available, else the fixed
    // kResidentComputeHeadroom constant. Defined in the .cpp (needs the constant).
    size_t resident_headroom_bytes() const;
    bool flash_attention() const { return flash_attention_; }
    bool circular_x() const { return circular_x_; }
    bool circular_y() const { return circular_y_; }
    const ed_tiling_params_t& vae_tiling() const { return vae_tiling_; }
    void enable_vae_tiling_for_memory() {
        if (vae_tiling_.force_disable) {
            return;
        }
        vae_tiling_.enabled = true;
        if (vae_tiling_.rel_size_x <= 0.0f) {
            vae_tiling_.rel_size_x = 5.0f;
        }
        if (vae_tiling_.rel_size_y <= 0.0f) {
            vae_tiling_.rel_size_y = 5.0f;
        }
        if (vae_tiling_.target_overlap <= 0.0f) {
            vae_tiling_.target_overlap = 0.25f;
        }
    }
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
    GenerationControl* generation_control() const { return generation_control_; }
    std::shared_ptr<parallel::ProcessGroup> graph_process_group_ref() const {
        if (parallel_context_ == nullptr || !parallel_context_->enabled()) {
            return nullptr;
        }
        const bool graph_parallel = parallel_context_->tp_parallel_size() > 1 ||
                                    parallel_context_->sp_parallel_size() > 1;
        if (!graph_parallel) {
            return nullptr;
        }
        // Pure CFG parallelism communicates at the pipeline level. Graph-level
        // runners should only see the model-parallel group; once CFG+SP is
        // supported this must return the SP/TP subgroup, not the world group.

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
    bool text_encoder_offload_ = false;
    bool minimax_h3_stage_lifecycle_ = false;
    bool dit_offload_ = false;
    bool vae_offload_ = false;
    bool auto_allocate_ = false;
    bool auto_fit_ = false;  // auto-fit: system chooses DiT quant + placement to fit budget
    int fit_width_ = 0;   // target gen resolution for compute-buffer measure (0 = unset)
    int fit_height_ = 0;
    int fit_frames_ = 0;  // target video frame count for compute-buffer measure (0 = image/unset)
    int fit_fps_ = 24;    // target video fps for audio-aware compute-buffer measure
    size_t measured_dit_headroom_ = 0;  // measured DiT compute buffer (bytes); 0 = use fixed constant
    size_t resident_bytes_total_ = 0;  // auto-allocate: bytes decided resident this load
    bool any_component_offloaded_ = false;  // auto-allocate: some component this load was offloaded (needs segment room)
    bool free_params_immediately_ = false;

    float max_vram_ = 0.0f;
    size_t max_graph_vram_bytes_ = 0;

    bool flash_attention_ = false;
    bool circular_x_ = false;
    bool circular_y_ = false;
    ed_tiling_params_t vae_tiling_ = {};

    RuntimeBackends backends_;
    parallel::ParallelContext* parallel_context_ = nullptr;
    GenerationControl* generation_control_ = nullptr;

    std::shared_ptr<RNG> rng_;
    std::shared_ptr<RNG> sampler_rng_;

    bool init_threads(const ed_context_params_t& params, std::string* error);
    bool init_flags(const ed_context_params_t& params, std::string* error);
    bool init_backends(const ed_context_params_t& params, std::string* error);
    void maybe_enable_vae_tiling_for_low_vram();
    bool init_rng(const ed_context_params_t& params, std::string* error);

    void release_backends();
    bool fail(std::string* error, const std::string& msg);
    size_t component_memory_bytes(const ::ModelLoader& loader,
                                  const std::vector<std::string>& weight_prefixes) const;

    ComponentMemoryEstimator component_memory_estimator_;
};

} // namespace edgedit
