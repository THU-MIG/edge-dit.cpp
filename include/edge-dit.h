#ifndef EDGE_DIT_H
#define EDGE_DIT_H

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef ED_BUILD_SHARED_LIB
#    ifdef ED_BUILD_DLL
#      define ED_API __declspec(dllexport)
#    else
#      define ED_API __declspec(dllimport)
#    endif
#  else
#    define ED_API
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define ED_API __attribute__((visibility("default")))
#  else
#    define ED_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef struct ed_context ed_context_t;

ED_API const char * ed_version_string(void);
ED_API int ed_version_major(void);
ED_API int ed_version_minor(void);
ED_API int ed_version_patch(void);

typedef enum ed_status_t {
    ED_STATUS_OK = 0,
    ED_STATUS_ERROR,
    ED_STATUS_INVALID_ARGUMENT,
    ED_STATUS_MODEL_LOAD_FAILED,
    ED_STATUS_GENERATION_FAILED,
    ED_STATUS_OUT_OF_MEMORY,
    ED_STATUS_UNSUPPORTED,
    ED_STATUS_CANCELLED
} ed_status_t;

typedef enum ed_dtype_t {
    ED_DTYPE_AUTO = -1,
    ED_DTYPE_F32  = 0,
    ED_DTYPE_F16  = 1,
    ED_DTYPE_BF16 = 30,
    ED_DTYPE_Q4_0 = 2,
    ED_DTYPE_Q4_1 = 3,
    ED_DTYPE_Q5_0 = 6,
    ED_DTYPE_Q5_1 = 7,
    ED_DTYPE_Q8_0 = 8,
    ED_DTYPE_Q2_K = 10,
    ED_DTYPE_Q3_K = 11,
    ED_DTYPE_Q4_K = 12,
    ED_DTYPE_Q5_K = 13,
    ED_DTYPE_Q6_K = 14
} ed_dtype_t;

typedef enum ed_sampler_t {
    ED_SAMPLER_AUTO = -1,
    ED_SAMPLER_EULER = 0,
    ED_SAMPLER_EULER_A,
    ED_SAMPLER_HEUN,
    ED_SAMPLER_DPM2,
    ED_SAMPLER_DPM_PLUS_PLUS_2S_A,
    ED_SAMPLER_DPM_PLUS_PLUS_2M,
    ED_SAMPLER_DPM_PLUS_PLUS_2M_V2,
    ED_SAMPLER_IPNDM,
    ED_SAMPLER_IPNDM_V,
    ED_SAMPLER_LCM,
    ED_SAMPLER_DDIM_TRAILING,
    ED_SAMPLER_TCD,
    ED_SAMPLER_RES_MULTISTEP,
    ED_SAMPLER_RES_2S,
    ED_SAMPLER_ER_SDE
} ed_sampler_t;

typedef enum ed_scheduler_t {
    ED_SCHEDULER_AUTO = -1,
    ED_SCHEDULER_DISCRETE = 0,
    ED_SCHEDULER_KARRAS,
    ED_SCHEDULER_EXPONENTIAL,
    ED_SCHEDULER_AYS,
    ED_SCHEDULER_GITS,
    ED_SCHEDULER_SGM_UNIFORM,
    ED_SCHEDULER_SIMPLE,
    ED_SCHEDULER_SMOOTHSTEP,
    ED_SCHEDULER_KL_OPTIMAL,
    ED_SCHEDULER_LCM,
    ED_SCHEDULER_BONG_TANGENT,
    ED_SCHEDULER_LTX2
} ed_scheduler_t;

typedef enum ed_ref_image_size_t {
    ED_REF_IMAGE_SIZE_MAX = 0,
    ED_REF_IMAGE_SIZE_MATCH = 1,
} ed_ref_image_size_t;

typedef enum ed_cache_mode_t {
    ED_CACHE_DISABLED = 0,
    ED_CACHE_EASYCACHE,
    ED_CACHE_UCACHE,
    ED_CACHE_DBCACHE,
    ED_CACHE_TAYLORSEER,
    ED_CACHE_CACHE_DIT,
    ED_CACHE_MAGCACHE,
    ED_CACHE_DICACHE,
    ED_CACHE_SENCACHE
} ed_cache_mode_t;

typedef struct ed_image_t {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t * data;
} ed_image_t;

typedef struct ed_audio_t {
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t sample_count;
    const float * data;
} ed_audio_t;

typedef struct ed_ref_video_t {
    const ed_image_t * frames;
    int frame_count;
    int fps;
    ed_audio_t audio;
} ed_ref_video_t;

typedef struct ed_image_batch_t {
    ed_image_t * images;
    int count;
} ed_image_batch_t;

typedef struct ed_video_t {
    ed_image_t * frames;
    int frame_count;
    float * audio;
    int audio_sample_count;
    int audio_channels;
    int audio_sample_rate;
} ed_video_t;

typedef struct ed_lora_t {
    const char * path;
    float scale;
    bool high_noise;
} ed_lora_t;

typedef struct ed_tiling_params_t {
    bool enabled;
    bool force_disable;  // true = user explicitly disabled tiling; suppresses low-VRAM auto-enable
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
} ed_tiling_params_t;

typedef struct ed_context_params_t {
    const char * model_path;

    const char * diffusion_model_path;
    const char * high_noise_diffusion_model_path;
    const char * clip_l_path;
    const char * clip_g_path;
    const char * clip_vision_path;
    const char * t5xxl_path;
    const char * llm_path;
    const char * llm_vision_path;
    const char * vae_path;
    const char * audio_vae_path;
    const char * taesd_path;
    const char * control_net_path;

    int n_threads;
    ed_dtype_t weight_type;
    const char * tensor_type_rules;

    bool use_mmap;
    bool offload_params_to_cpu;
    bool dit_offload;           // DiT weights on CPU, staged to GPU per step (compute on GPU)
    bool text_encoder_offload;  // TE weights on CPU, staged to GPU per encode (compute on GPU)
    bool minimax_h3_stage_lifecycle;  // MiniMax-H3: stage TE/VAE by phase, release them during DiT
    bool auto_allocate;         // budget-capped placement: min(--max-vram, free), per-component resident/offload; external workspaces may raise process peak
    bool auto_fit;              // fully automatic: system picks DiT quantization (q8_0..q4_k) AND placement to fit budget; implies auto_allocate, ignores weight_type
    int fit_width;              // target generation width for auto-allocate/auto-fit compute-buffer measurement (0 = unspecified -> fixed headroom fallback)
    int fit_height;             // target generation height (0 = unspecified)
    int fit_frames;             // target video frame count for compute-buffer measurement (video models only; 0 = image/unspecified)
    bool keep_control_net_on_cpu;
    bool vae_offload;           // VAE weights on CPU, staged to GPU per decode (compute on GPU)
    bool skip_t5;

    bool flash_attention;

    bool qwen_image_zero_cond_t;

    float max_vram_gb;

    ed_tiling_params_t vae_tiling;

    int cfg_parallel_size;
    int tp_parallel_size;
    int sp_parallel_size;

    const char * embeddings_connectors_path;
    const char * latent_upscaler_path;
    int fit_fps;                  // target video fps for audio-aware compute-buffer measurement (0 = default 24)
} ed_context_params_t;

typedef struct ed_sample_params_t {
    ed_sampler_t sampler;
    ed_scheduler_t scheduler;

    int steps;
    float cfg_scale;
    float image_cfg_scale;
    float distilled_guidance;
    float eta;
    float flow_shift;

    ed_cache_mode_t cache_mode;
    float cache_reuse_threshold;
    float cache_start_percent;
    float cache_end_percent;
    float cache_error_decay_rate;
    bool cache_use_relative_threshold;
    bool cache_reset_error_on_compute;
    int cache_Fn_compute_blocks;
    int cache_Bn_compute_blocks;
    float cache_residual_diff_threshold;
    float cache_max_accumulated_residual_diff;
    int cache_max_warmup_steps;
    int cache_max_cached_steps;
    int cache_max_continuous_cached_steps;
    int cache_taylorseer_n_derivatives;
    int cache_taylorseer_skip_interval;
    const char * cache_scm_mask;
    bool cache_scm_policy_dynamic;
    const char * cache_calibrate_path;
    const char * cache_profile_path;
} ed_sample_params_t;

typedef struct ed_image_generation_params_t {
    const char * prompt;
    const char * negative_prompt;

    int width;
    int height;
    int64_t seed;
    int batch_count;

    const ed_image_t * init_image;
    const ed_image_t * mask_image;
    const ed_image_t * control_image;

    const ed_image_t * ref_images;
    int ref_image_count;

    float strength;
    float control_strength;

    ed_sample_params_t sample;

    const ed_lora_t * loras;
    uint32_t lora_count;
} ed_image_generation_params_t;

typedef struct ed_video_generation_params_t {
    const char * prompt;
    const char * negative_prompt;

    int width;
    int height;
    int frames;
    int64_t seed;

    const ed_image_t * init_image;
    const ed_image_t * end_image;

    const ed_image_t * ref_images;
    int ref_image_count;
    ed_ref_image_size_t ref_image_size;
    const ed_ref_video_t * ref_videos;
    int ref_video_count;
    const ed_audio_t * ref_audios;
    int ref_audio_count;

    const ed_image_t * control_frames;
    int control_frame_count;

    float strength;
    float vace_strength;
    float moe_boundary;

    ed_sample_params_t sample;
    ed_sample_params_t high_noise_sample;

    const ed_lora_t * loras;
    uint32_t lora_count;

    int fps;
    bool hires_enabled;
    int hires_steps;
    float hires_denoising_strength;
    const float * hires_sigmas;
    int hires_sigmas_count;
} ed_video_generation_params_t;

ED_API void ed_context_params_init(ed_context_params_t * params);
ED_API void ed_sample_params_init(ed_sample_params_t * params);
ED_API void ed_image_generation_params_init(ed_image_generation_params_t * params);
ED_API void ed_video_generation_params_init(ed_video_generation_params_t * params);

ED_API ed_context_t * ed_create_context(const ed_context_params_t * params);
ED_API void ed_free_context(ed_context_t * ctx);

ED_API ed_status_t ed_generate_image(
    ed_context_t * ctx,
    const ed_image_generation_params_t * params,
    ed_image_batch_t * out
);

ED_API ed_status_t ed_generate_video(
    ed_context_t * ctx,
    const ed_video_generation_params_t * params,
    ed_video_t * out
);

ED_API void ed_free_image(ed_image_t * image);
ED_API void ed_free_image_batch(ed_image_batch_t * batch);
ED_API void ed_free_video(ed_video_t * video);

ED_API const char * ed_get_last_error(const ed_context_t * ctx);
ED_API const char * ed_context_pipeline_name(const ed_context_t * ctx);
ED_API const char * ed_context_version_name(const ed_context_t * ctx);
ED_API bool ed_context_supports_image(const ed_context_t * ctx);
ED_API bool ed_context_supports_video(const ed_context_t * ctx);
ED_API ed_sampler_t ed_context_default_sampler(const ed_context_t * ctx);
ED_API ed_scheduler_t ed_context_default_scheduler(const ed_context_t * ctx, ed_sampler_t sampler);
ED_API void ed_context_request_cancel(ed_context_t * ctx);
ED_API int ed_context_progress_current_step(const ed_context_t * ctx);
ED_API int ed_context_progress_total_steps(const ed_context_t * ctx);
ED_API int ed_context_parallel_rank(const ed_context_t * ctx);
ED_API int ed_context_parallel_world_size(const ed_context_t * ctx);
ED_API bool ed_context_parallel_is_root(const ed_context_t * ctx);

/* True when a cache method consumes a precalibrated data table and can be
 * profiled via cache_calibrate_path. Only such methods accept calibration. */
ED_API bool ed_cache_mode_supports_calibration(ed_cache_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif /* EDGE_DIT_H */
