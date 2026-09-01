#include "runtime/edge_dit_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "utils/util.h"

namespace edgedit {
namespace {

struct GenerationResetGuard {
    GenerationControl& control;

    ~GenerationResetGuard() {
        control.reset_idle();
    }
};

std::string lower_copy(const char* value) {
    std::string out = value != nullptr ? value : "";
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int inferred_world_size() {
    int world_size = env_int("WORLD_SIZE", 1);
    world_size = env_int("OMPI_COMM_WORLD_SIZE", world_size);
    world_size = env_int("MV2_COMM_WORLD_SIZE", world_size);
    world_size = env_int("SLURM_NTASKS", world_size);
    world_size = env_int("PMI_SIZE", world_size);
    return world_size;
}

int inferred_rank() {
    int rank = env_int("RANK", 0);
    rank = env_int("OMPI_COMM_WORLD_RANK", rank);
    rank = env_int("MV2_COMM_WORLD_RANK", rank);
    rank = env_int("SLURM_PROCID", rank);
    rank = env_int("PMI_RANK", rank);
    return rank;
}

int inferred_local_rank() {
    int local_rank = env_int("LOCAL_RANK", 0);
    local_rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", local_rank);
    local_rank = env_int("MV2_COMM_WORLD_LOCAL_RANK", local_rank);
    local_rank = env_int("SLURM_LOCALID", local_rank);
    local_rank = env_int("PMI_LOCAL_RANK", local_rank);
    return local_rank;
}

int requested_parallel_world_size(const ed_ctx_params_t& params) {
    return std::max({1, params.cfg_parallel_size, params.tp_parallel_size, params.sp_parallel_size});
}

bool runtime_backend_is_cpu() {
    const std::string backend = lower_copy(std::getenv("ED_BACKEND"));
    return backend == "cpu";
}

bool single_visible_device_per_worker() {
    const char* value = std::getenv("ED_CLI_SINGLE_VISIBLE_DEVICE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool has_text(const char* value) {
    return value != nullptr && value[0] != '\0';
}

bool has_ltx_only_context_options(const ed_ctx_params_t& params) {
    return has_text(params.embeddings_connectors_path) ||
           has_text(params.latent_upscaler_path);
}

bool is_ltx_scheduler_incompatible(ed_scheduler_t scheduler, SDVersion version) {
    return scheduler == ED_SCHEDULER_LTX2 && !ed_version_is_ltxav(version);
}

bool is_ltx_hires_incompatible(bool hires_enabled, SDVersion version) {
    return hires_enabled && !ed_version_is_ltxav(version);
}

parallel::ParallelConfig make_parallel_config(const ed_ctx_params_t& params) {
    parallel::ParallelConfig config;
    config.cfg_parallel_size = params.cfg_parallel_size > 0 ? params.cfg_parallel_size : 1;
    config.tp_parallel_size  = params.tp_parallel_size > 0 ? params.tp_parallel_size : 1;
    config.sp_parallel_size  = params.sp_parallel_size > 0 ? params.sp_parallel_size : 1;

    const int requested_world_size = requested_parallel_world_size(params);
    const int launched_world_size  = inferred_world_size();
    const int rank                 = inferred_rank();
    const int local_rank           = inferred_local_rank();

    // These fields must be populated before create_parallel_context().
    // Otherwise the real engine path would still initialize ProcessGroup
    // with the default rank=0/world_size=1 even under mpirun.
    config.world_size = launched_world_size;
    config.rank       = rank;
    config.local_rank = local_rank;
    config.device     = single_visible_device_per_worker() ? 0 : local_rank;

    if (requested_world_size <= 1 && launched_world_size <= 1) {
        config.backend = parallel::Backend::kNone;
        return config;
    }

    if (requested_world_size > 1 && launched_world_size <= 1) {
        throw std::invalid_argument("parallel execution requested but launched world_size is 1");
    }

    if (requested_world_size > 1 && launched_world_size != requested_world_size) {
        throw std::invalid_argument("launched worker count must match requested parallel size");
    }

    if (runtime_backend_is_cpu()) {
        throw std::invalid_argument("distributed CPU runtime backend is not wired into engine execution yet");
    }

    config.backend = parallel::Backend::kNccl;
    return config;
}

} // namespace

bool EdgeDitEngine::init(const ed_ctx_params_t* params) {
    last_error_.clear();

    if (params == nullptr) {
        set_error("EdgeDitEngine::init got null params");
        return false;
    }

    ctx_params_ = *params;

    dit_pipeline_.reset();
    model_loader_.reset();
    runtime_.reset();
    parallel_context_.reset();

    auto cleanup = [&]() {
        dit_pipeline_.reset();
        model_loader_.reset();
        runtime_.reset();
        parallel_context_.reset();
        generation_control_.reset_idle();
    };

    try {
        parallel_context_ = parallel::create_parallel_context(make_parallel_config(ctx_params_));
        runtime_      = std::make_unique<ModelRuntime>();
        model_loader_ = std::make_unique<ModelLoader>();
    } catch (const std::exception& e) {
        set_error(std::string("failed to allocate engine components: ") + e.what());
        cleanup();
        return false;
    }

    runtime_->set_parallel_context(parallel_context_.get());
    runtime_->set_generation_control(&generation_control_);

    if (!runtime_->init(ctx_params_, &last_error_)) {
        set_error(last_error_.empty() ? "ModelRuntime::init failed" : last_error_);
        cleanup();
        return false;
    }

    if (!model_loader_->load_model_files(ctx_params_, &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::load_model_files failed" : last_error_);
        cleanup();
        return false;
    }

    if (!model_loader_->finalize_names_and_version(&last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::finalize_names_and_version failed" : last_error_);
        cleanup();
        return false;
    }

    if (!ed_version_is_ltxav(model_loader_->version()) &&
        has_ltx_only_context_options(ctx_params_)) {
        set_error("embeddings connectors and latent upscaler are supported only by LTX-2.3");
        cleanup();
        return false;
    }

    if (ctx_params_.skip_t5 && !ed_version_is_sd3(model_loader_->version())) {
        set_error("--no-t5 is only supported for SD3 models");
        cleanup();
        return false;
    }

    if (!model_loader_->apply_dtype_policy(ctx_params_, &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::apply_dtype_policy failed" : last_error_);
        cleanup();
        return false;
    }

    dit_pipeline_ = create_dit_pipeline(model_loader_->version(), &last_error_);
    if (dit_pipeline_ == nullptr) {
        set_error(last_error_.empty() ? "failed to create DiT pipeline" : last_error_);
        cleanup();
        return false;
    }

    PipelineTensorRegistry registry;
    if (!dit_pipeline_->prepare_memory_plan(ctx_params_,
                                            *runtime_,
                                            *model_loader_,
                                            &last_error_)) {
        set_error(last_error_.empty() ? "DiT pipeline memory planning failed" : last_error_);
        cleanup();
        return false;
    }
    // Auto-fit: before offload planning, let the runtime choose the DiT quant level
    // (q8_0..q4_k, superseding --type for the DiT) that keeps it resident within the VRAM budget (no-op
    // unless --auto-fit). Must run after apply_dtype_policy (expected_type is set) and
    // before prepare (which reads expected_type for offload decisions) and bind_weights
    // (which materializes the quantized weights). Uses the non-const loader.
    runtime_->replan_dit_quant_for_budget(*model_loader_);

    if (!dit_pipeline_->prepare(ctx_params_,
                                *runtime_,
                                *model_loader_,
                                registry,
                                &last_error_)) {
        set_error(last_error_.empty() ? "DiT pipeline prepare failed" : last_error_);
        cleanup();
        return false;
    }

    if (!model_loader_->bind_weights(registry.tensors(),
                                     registry.ignore_tensors(),
                                     runtime_->n_threads(),
                                     runtime_->use_mmap(),
                                     &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::bind_weights failed" : last_error_);
        cleanup();
        return false;
    }

    model_loader_->log_weight_stats();
    dit_pipeline_->mark_ready();
    if (parallel_context_ != nullptr && parallel_context_->enabled()) {
        parallel_context_->world_group().warmup();
        parallel_context_->world_group().barrier();
    }

    if (parallel_is_root()) {
        LOG_INFO("EdgeDitEngine initialized successfully, version=%s, parallel=%s rank=%d/%d",
                 ed_version_name(dit_pipeline_->version()),
                 parallel_context_ != nullptr ? parallel::backend_name(parallel_context_->backend()) : "none",
                 parallel_rank(),
                 parallel_world_size());
    }
    return true;
}

ed_status_t EdgeDitEngine::generate_image(const ed_image_generation_params_t* params,
                                           ed_image_batch_t* out) {
    last_error_.clear();
    generation_control_.reset_idle();
    GenerationResetGuard guard{generation_control_};
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }

    if (params == nullptr || out == nullptr) {
        set_error("EdgeDitEngine::generate_image got null params or out");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_ == nullptr || !runtime_->ready() || dit_pipeline_ == nullptr || !dit_pipeline_->ready()) {
        set_error("engine is not initialized");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (!supports_image_generation()) {
        set_error("current model/version does not support image generation");
        return ED_STATUS_UNSUPPORTED;
    }
    if (is_ltx_scheduler_incompatible(params->sample.scheduler, dit_pipeline_->version())) {
        set_error("the ltx2 scheduler is supported only by LTX-2.3");
        return ED_STATUS_UNSUPPORTED;
    }

    ed_status_t status = dit_pipeline_->generate_image(params, out, &last_error_);
    if (status != ED_STATUS_OK) {
        if (generation_control_.was_cancelled()) {
            set_error(last_error_.empty() ? "generation cancelled" : last_error_);
            if (out != nullptr) {
                ed_free_image_batch(out);
            }
            return ED_STATUS_CANCELLED;
        }
        if (last_error_.empty()) {
            set_error("image generation failed");
        } else {
            set_error(last_error_);
        }
        if (out != nullptr) {
            ed_free_image_batch(out);
        }
        return status;
    }

    if (!parallel_is_root()) {
        ed_free_image_batch(out);
        return ED_STATUS_OK;
    }

    if (out->images == nullptr || out->count <= 0) {
        ed_free_image_batch(out);
        set_error("DiT pipeline returned empty image batch");
        return ED_STATUS_GENERATION_FAILED;
    }
    return ED_STATUS_OK;
}

ed_status_t EdgeDitEngine::generate_video(const ed_video_generation_params_t* params,
                                           ed_video_t* out) {
    last_error_.clear();
    generation_control_.reset_idle();
    GenerationResetGuard guard{generation_control_};
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }

    if (params == nullptr || out == nullptr) {
        set_error("EdgeDitEngine::generate_video got null params or out");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_ == nullptr || !runtime_->ready() || dit_pipeline_ == nullptr || !dit_pipeline_->ready()) {
        set_error("engine is not initialized");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (is_ltx_scheduler_incompatible(params->sample.scheduler, dit_pipeline_->version())) {
        set_error("the ltx2 scheduler is supported only by LTX-2.3");
        return ED_STATUS_UNSUPPORTED;
    }
    if (is_ltx_hires_incompatible(params->hires_enabled, dit_pipeline_->version())) {
        set_error("hires is supported only by LTX-2.3");
        return ED_STATUS_UNSUPPORTED;
    }
    if (!supports_video_generation()) {
        set_error("current model/version does not support video generation");
        return ED_STATUS_UNSUPPORTED;
    }

    ed_status_t status = dit_pipeline_->generate_video(params, out, &last_error_);
    if (status != ED_STATUS_OK) {
        if (generation_control_.was_cancelled()) {
            set_error(last_error_.empty() ? "generation cancelled" : last_error_);
            if (out != nullptr) {
                ed_free_video(out);
            }
            return ED_STATUS_CANCELLED;
        }
        if (last_error_.empty()) {
            set_error("video generation failed");
        } else {
            set_error(last_error_);
        }
        if (out != nullptr) {
            ed_free_video(out);
        }
        return status;
    }

    if (!parallel_is_root()) {
        ed_free_video(out);
        return ED_STATUS_OK;
    }

    if (out->frames == nullptr || out->frame_count <= 0) {
        ed_free_video(out);
        set_error("DiT pipeline returned empty video");
        return ED_STATUS_GENERATION_FAILED;
    }
    return ED_STATUS_OK;
}

bool EdgeDitEngine::supports_image_generation() const {
    return dit_pipeline_ != nullptr && dit_pipeline_->supports_image_generation();
}

bool EdgeDitEngine::supports_video_generation() const {
    return dit_pipeline_ != nullptr && dit_pipeline_->supports_video_generation();
}

sample_method_t EdgeDitEngine::get_default_sample_method() const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->default_sample_method() : EULER_A_SAMPLE_METHOD;
}

scheduler_t EdgeDitEngine::get_default_scheduler(sample_method_t method) const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->default_scheduler(method) : DISCRETE_SCHEDULER;
}

const char* EdgeDitEngine::pipeline_name() const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->name() : nullptr;
}

const char* EdgeDitEngine::version_name() const {
    return dit_pipeline_ != nullptr ? ed_version_name(dit_pipeline_->version()) : nullptr;
}

bool EdgeDitEngine::parallel_enabled() const {
    return parallel_context_ != nullptr && parallel_context_->enabled();
}

bool EdgeDitEngine::parallel_is_root() const {
    return parallel_context_ == nullptr || parallel_context_->is_root();
}

int EdgeDitEngine::parallel_rank() const {
    return parallel_context_ != nullptr ? parallel_context_->rank() : 0;
}

int EdgeDitEngine::parallel_world_size() const {
    return parallel_context_ != nullptr ? parallel_context_->world_size() : 1;
}

int EdgeDitEngine::parallel_local_rank() const {
    return parallel_context_ != nullptr ? parallel_context_->local_rank() : 0;
}

void EdgeDitEngine::request_cancel() {
    generation_control_.request_cancel();
}

int EdgeDitEngine::progress_current_step() const {
    return generation_control_.current_step.load(std::memory_order_relaxed);
}

int EdgeDitEngine::progress_total_steps() const {
    return generation_control_.total_steps.load(std::memory_order_relaxed);
}

void EdgeDitEngine::set_error(const std::string& msg) {
    last_error_ = msg;
    LOG_ERROR("%s", last_error_.c_str());
}

} // namespace edgedit
