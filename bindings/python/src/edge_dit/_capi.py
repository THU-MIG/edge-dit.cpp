from __future__ import annotations

import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_float, c_int, c_int64, c_uint32, c_uint8

from ._lib import load_library


class EdContext(ctypes.Structure):
    pass


EdContextHandle = POINTER(EdContext)


class EdImage(ctypes.Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("channels", c_uint32),
        ("data", POINTER(c_uint8)),
    ]


class EdImageBatch(ctypes.Structure):
    _fields_ = [
        ("images", POINTER(EdImage)),
        ("count", c_int),
    ]


class EdVideo(ctypes.Structure):
    _fields_ = [
        ("frames", POINTER(EdImage)),
        ("frame_count", c_int),
    ]


class EdLora(ctypes.Structure):
    _fields_ = [
        ("path", c_char_p),
        ("scale", c_float),
        ("high_noise", c_bool),
    ]


class EdTilingParams(ctypes.Structure):
    _fields_ = [
        ("enabled", c_bool),
        ("tile_size_x", c_int),
        ("tile_size_y", c_int),
        ("target_overlap", c_float),
        ("rel_size_x", c_float),
        ("rel_size_y", c_float),
    ]


class EdContextParams(ctypes.Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("diffusion_model_path", c_char_p),
        ("high_noise_diffusion_model_path", c_char_p),
        ("clip_l_path", c_char_p),
        ("clip_g_path", c_char_p),
        ("clip_vision_path", c_char_p),
        ("t5xxl_path", c_char_p),
        ("llm_path", c_char_p),
        ("llm_vision_path", c_char_p),
        ("vae_path", c_char_p),
        ("taesd_path", c_char_p),
        ("control_net_path", c_char_p),
        ("n_threads", c_int),
        ("weight_type", c_int),
        ("tensor_type_rules", c_char_p),
        ("use_mmap", c_bool),
        ("offload_params_to_cpu", c_bool),
        ("keep_text_encoder_on_cpu", c_bool),
        ("keep_control_net_on_cpu", c_bool),
        ("keep_vae_on_cpu", c_bool),
        ("skip_t5", c_bool),
        ("flash_attention", c_bool),
        ("max_vram_gb", c_float),
        ("vae_tiling", EdTilingParams),
        ("cfg_parallel_size", c_int),
        ("tp_parallel_size", c_int),
        ("sp_parallel_size", c_int),
    ]


class EdSampleParams(ctypes.Structure):
    _fields_ = [
        ("sampler", c_int),
        ("scheduler", c_int),
        ("steps", c_int),
        ("cfg_scale", c_float),
        ("image_cfg_scale", c_float),
        ("distilled_guidance", c_float),
        ("eta", c_float),
        ("flow_shift", c_float),
        ("cache_mode", c_int),
        ("cache_reuse_threshold", c_float),
        ("cache_start_percent", c_float),
        ("cache_end_percent", c_float),
        ("cache_error_decay_rate", c_float),
        ("cache_use_relative_threshold", c_bool),
        ("cache_reset_error_on_compute", c_bool),
        ("cache_Fn_compute_blocks", c_int),
        ("cache_Bn_compute_blocks", c_int),
        ("cache_residual_diff_threshold", c_float),
        ("cache_max_accumulated_residual_diff", c_float),
        ("cache_max_warmup_steps", c_int),
        ("cache_max_cached_steps", c_int),
        ("cache_max_continuous_cached_steps", c_int),
        ("cache_taylorseer_n_derivatives", c_int),
        ("cache_taylorseer_skip_interval", c_int),
        ("cache_scm_mask", c_char_p),
        ("cache_scm_policy_dynamic", c_bool),
    ]


class EdImageGenerationParams(ctypes.Structure):
    _fields_ = [
        ("prompt", c_char_p),
        ("negative_prompt", c_char_p),
        ("width", c_int),
        ("height", c_int),
        ("seed", c_int64),
        ("batch_count", c_int),
        ("init_image", POINTER(EdImage)),
        ("mask_image", POINTER(EdImage)),
        ("control_image", POINTER(EdImage)),
        ("ref_images", POINTER(EdImage)),
        ("ref_image_count", c_int),
        ("strength", c_float),
        ("control_strength", c_float),
        ("sample", EdSampleParams),
        ("loras", POINTER(EdLora)),
        ("lora_count", c_uint32),
    ]


def bind_api(lib: object) -> object:
    lib.ed_context_params_init.argtypes = [POINTER(EdContextParams)]
    lib.ed_context_params_init.restype = None

    lib.ed_sample_params_init.argtypes = [POINTER(EdSampleParams)]
    lib.ed_sample_params_init.restype = None

    lib.ed_image_generation_params_init.argtypes = [POINTER(EdImageGenerationParams)]
    lib.ed_image_generation_params_init.restype = None

    lib.ed_create_context.argtypes = [POINTER(EdContextParams)]
    lib.ed_create_context.restype = EdContextHandle

    lib.ed_free_context.argtypes = [EdContextHandle]
    lib.ed_free_context.restype = None

    lib.ed_generate_image.argtypes = [
        EdContextHandle,
        POINTER(EdImageGenerationParams),
        POINTER(EdImageBatch),
    ]
    lib.ed_generate_image.restype = c_int

    lib.ed_free_image_batch.argtypes = [POINTER(EdImageBatch)]
    lib.ed_free_image_batch.restype = None

    lib.ed_get_last_error.argtypes = [EdContextHandle]
    lib.ed_get_last_error.restype = c_char_p

    lib.ed_context_parallel_is_root.argtypes = [EdContextHandle]
    lib.ed_context_parallel_is_root.restype = c_bool

    return lib


def load_capi(
    *,
    path: str | None = None,
    library: object | None = None,
) -> object:
    return bind_api(load_library(path) if library is None else library)


__all__ = [
    "EdContext",
    "EdContextHandle",
    "EdContextParams",
    "EdImage",
    "EdImageBatch",
    "EdImageGenerationParams",
    "EdLora",
    "EdSampleParams",
    "EdTilingParams",
    "load_capi",
]

