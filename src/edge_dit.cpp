#include "edge-dit.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include "core/runtime/edge_dit_engine.hpp"
#include "core/optimization/cache/cache_types.hpp"
#include "utils/util.h"

struct ed_context {
    ed_context_params_t params = {};
    std::unique_ptr<edgedit::EdgeDitEngine> engine;
    std::string last_error;
    bool initialized = false;
};

static void ed_zero(void * ptr, size_t size) {
    if (ptr != nullptr) {
        std::memset(ptr, 0, size);
    }
}

static void ed_set_error(ed_context_t * ctx, const char * message) {
    if (ctx != nullptr) {
        ctx->last_error = message != nullptr ? message : "";
    }
}

const char * ed_version_string(void) {
    return ED_VERSION_STRING;
}

int ed_version_major(void) {
    return ED_VERSION_MAJOR;
}

int ed_version_minor(void) {
    return ED_VERSION_MINOR;
}

int ed_version_patch(void) {
    return ED_VERSION_PATCH;
}

void ed_context_params_init(ed_context_params_t * params) {
    ed_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->n_threads = 0;
    params->weight_type = ED_DTYPE_AUTO;
    params->use_mmap = true;
    params->auto_allocate = false;
    params->auto_fit = false;
    params->fit_width = 0;
    params->fit_height = 0;
    params->fit_frames = 0;
    params->flash_attention = true;
    params->qwen_image_zero_cond_t = false;
    params->max_vram_gb = 0.0f;
    params->cfg_parallel_size = 1;
    params->tp_parallel_size = 1;
    params->sp_parallel_size = 1;
    params->vae_tiling.enabled = false;
    params->vae_tiling.rel_size_x = 5.0f;
    params->vae_tiling.rel_size_y = 5.0f;
    params->vae_tiling.target_overlap = 0.25f;
}

void ed_sample_params_init(ed_sample_params_t * params) {
    ed_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->sampler = ED_SAMPLER_AUTO;
    params->scheduler = ED_SCHEDULER_AUTO;
    params->steps = 20;
    params->cfg_scale = 1.0f;
    params->image_cfg_scale = 1.0f;
    params->distilled_guidance = 3.5f;
    params->eta = 0.0f;
    params->flow_shift = 0.0f;
    params->cache_mode = ED_CACHE_DISABLED;
    params->cache_reuse_threshold = INFINITY;
    params->cache_start_percent = 0.15f;
    params->cache_end_percent = 0.95f;
    params->cache_error_decay_rate = 1.0f;
    params->cache_use_relative_threshold = true;
    params->cache_reset_error_on_compute = true;
    params->cache_Fn_compute_blocks = 8;
    params->cache_Bn_compute_blocks = 0;
    params->cache_residual_diff_threshold = std::numeric_limits<float>::quiet_NaN();
    params->cache_max_accumulated_residual_diff = -1.0f;
    params->cache_max_warmup_steps = 8;
    params->cache_max_cached_steps = -1;
    params->cache_max_continuous_cached_steps = -1;
    params->cache_taylorseer_n_derivatives = 1;
    params->cache_taylorseer_skip_interval = 1;
    params->cache_scm_policy_dynamic = true;
    params->cache_calibrate_path = nullptr;
    params->cache_profile_path = nullptr;
}

void ed_image_generation_params_init(ed_image_generation_params_t * params) {
    ed_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->width = 1024;
    params->height = 1024;
    params->seed = -1;
    params->batch_count = 1;
    params->strength = 0.75f;
    params->control_strength = 1.0f;
    ed_sample_params_init(&params->sample);
}

void ed_video_generation_params_init(ed_video_generation_params_t * params) {
    ed_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->width = 1024;
    params->height = 1024;
    params->frames = 1;
    params->fps = 24;
    params->seed = -1;
    params->ref_image_size = ED_REF_IMAGE_SIZE_MAX;
    params->strength = 0.75f;
    params->vace_strength = 1.0f;
    params->moe_boundary = 0.5f;
    ed_sample_params_init(&params->sample);
    ed_sample_params_init(&params->high_noise_sample);
    params->high_noise_sample.steps = -1;
    params->hires_enabled = false;
    params->hires_steps = 4;
    params->hires_denoising_strength = 0.7f;
}

ed_context_t* ed_create_context(const ed_context_params_t* params) {
    if (params == nullptr) {
        LOG_ERROR("ed_create_context failed: params is null");
        return nullptr;
    }

    std::unique_ptr<ed_context_t> ctx(new (std::nothrow) ed_context_t());
    if (ctx == nullptr) {
        LOG_ERROR("ed_create_context failed: allocate ed_context failed");
        return nullptr;
    }

    ctx->params = *params;

    try {
        ctx->engine = std::make_unique<edgedit::EdgeDitEngine>();
    } catch (const std::exception& e) {
        ctx->last_error = std::string("failed to allocate EdgeDitEngine: ") + e.what();
        LOG_ERROR("%s", ctx->last_error.c_str());
        return nullptr;
    }

    if (ctx->engine == nullptr) {
        ctx->last_error = "failed to allocate EdgeDitEngine";
        LOG_ERROR("%s", ctx->last_error.c_str());
        return nullptr;
    }

    if (!ctx->engine->init(params)) {
        ctx->last_error = ctx->engine->last_error();
        if (ctx->last_error.empty()) {
            ctx->last_error = "failed to initialize EdgeDitEngine";
        }

        LOG_ERROR("failed to initialize engine: %s", ctx->last_error.c_str());
        return nullptr;
    }

    ctx->initialized = true;

    return ctx.release();
}

void ed_free_context(ed_context_t * ctx) {
    delete ctx;
}

ed_status_t ed_generate_image(
    ed_context_t* ctx,
    const ed_image_generation_params_t* params,
    ed_image_batch_t* out
) {
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }

    if (ctx == nullptr || params == nullptr || out == nullptr) {
        if (ctx != nullptr) {
            ed_set_error(ctx, "invalid argument: ctx, params, or out is null");
        }
        return ED_STATUS_INVALID_ARGUMENT;
    }

    if (!ctx->initialized || ctx->engine == nullptr) {
        ed_set_error(ctx, "engine is not initialized");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }

    ed_image_batch_t tmp = {};
    ed_status_t status = ctx->engine->generate_image(params, &tmp);

    if (status != ED_STATUS_OK) {
        ed_free_image_batch(&tmp);

        std::string err = ctx->engine->last_error();
        if (err.empty()) {
            err = "image generation failed";
        }

        ed_set_error(ctx, err.c_str());
        return status;
    }

    if (!ctx->engine->parallel_is_root()) {
        ed_free_image_batch(&tmp);
        ed_set_error(ctx, "");
        return ED_STATUS_OK;
    }

    if (tmp.images == nullptr || tmp.count <= 0) {
        ed_free_image_batch(&tmp);
        ed_set_error(ctx, "engine returned empty image batch");
        return ED_STATUS_GENERATION_FAILED;
    }
    *out = tmp;
    ed_set_error(ctx, "");
    return ED_STATUS_OK;
}

ed_status_t ed_generate_video(
    ed_context_t* ctx,
    const ed_video_generation_params_t* params,
    ed_video_t* out
) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }

    if (ctx == nullptr || params == nullptr || out == nullptr) {
        if (ctx != nullptr) {
            ed_set_error(ctx, "invalid argument: ctx, params, or out is null");
        }
        return ED_STATUS_INVALID_ARGUMENT;
    }

    if (!ctx->initialized || ctx->engine == nullptr) {
        ed_set_error(ctx, "engine is not initialized");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }

    ed_video_t tmp = {};
    ed_status_t status = ctx->engine->generate_video(params, &tmp);

    if (status != ED_STATUS_OK) {
        ed_free_video(&tmp);

        std::string err = ctx->engine->last_error();
        if (err.empty()) {
            err = "video generation failed";
        }

        ed_set_error(ctx, err.c_str());
        return status;
    }

    if (!ctx->engine->parallel_is_root()) {
        ed_free_video(&tmp);
        ed_set_error(ctx, "");
        return ED_STATUS_OK;
    }

    if (tmp.frames == nullptr || tmp.frame_count <= 0) {
        ed_free_video(&tmp);
        ed_set_error(ctx, "engine returned empty video");
        return ED_STATUS_GENERATION_FAILED;
    }

    *out = tmp;
    ed_set_error(ctx, "");
    return ED_STATUS_OK;
}

void ed_free_image(ed_image_t * image) {
    if (image == nullptr) {
        return;
    }

    std::free(image->data);
    image->data = nullptr;
    image->width = 0;
    image->height = 0;
    image->channels = 0;
}

void ed_free_image_batch(ed_image_batch_t* batch) {
    if (batch == nullptr) {
        return;
    }

    if (batch->images != nullptr && batch->count > 0) {
        for (int i = 0; i < batch->count; ++i) {
            ed_free_image(&batch->images[i]);
        }
    }

    std::free(batch->images);
    batch->images = nullptr;
    batch->count = 0;
}

void ed_free_video(ed_video_t* video) {
    if (video == nullptr) {
        return;
    }

    if (video->frames != nullptr && video->frame_count > 0) {
        for (int i = 0; i < video->frame_count; ++i) {
            ed_free_image(&video->frames[i]);
        }
    }

    std::free(video->frames);
    video->frames = nullptr;
    video->frame_count = 0;
    std::free(video->audio);
    video->audio = nullptr;
    video->audio_sample_count = 0;
    video->audio_channels = 0;
    video->audio_sample_rate = 0;
}

const char * ed_get_last_error(const ed_context_t * ctx) {
    if (ctx == nullptr || ctx->last_error.empty()) {
        return nullptr;
    }

    return ctx->last_error.c_str();
}

const char* ed_context_pipeline_name(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return nullptr;
    }
    return ctx->engine->pipeline_name();
}

const char* ed_context_version_name(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return nullptr;
    }
    return ctx->engine->version_name();
}

bool ed_context_supports_image(const ed_context_t* ctx) {
    return ctx != nullptr && ctx->engine != nullptr && ctx->engine->supports_image_generation();
}

bool ed_context_supports_video(const ed_context_t* ctx) {
    return ctx != nullptr && ctx->engine != nullptr && ctx->engine->supports_video_generation();
}

ed_sampler_t ed_context_default_sampler(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return ED_SAMPLER_AUTO;
    }
    return ctx->engine->get_default_sample_method();
}

ed_scheduler_t ed_context_default_scheduler(const ed_context_t* ctx, ed_sampler_t sampler) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return ED_SCHEDULER_AUTO;
    }
    const ed_sampler_t resolved = sampler == ED_SAMPLER_AUTO
                                      ? ctx->engine->get_default_sample_method()
                                      : sampler;
    return ctx->engine->get_default_scheduler(resolved);
}

void ed_context_request_cancel(ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return;
    }
    ctx->engine->request_cancel();
}

int ed_context_progress_current_step(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return 0;
    }
    return ctx->engine->progress_current_step();
}

int ed_context_progress_total_steps(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return 0;
    }
    return ctx->engine->progress_total_steps();
}

int ed_context_parallel_rank(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return 0;
    }
    return ctx->engine->parallel_rank();
}

int ed_context_parallel_world_size(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return 1;
    }
    return ctx->engine->parallel_world_size();
}

bool ed_context_parallel_is_root(const ed_context_t* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) {
        return true;
    }
    return ctx->engine->parallel_is_root();
}

bool ed_cache_mode_supports_calibration(ed_cache_mode_t mode) {
    return edgedit::cache::cache_mode_supports_calibration(edgedit::cache::cache_mode_from_ld(mode));
}
