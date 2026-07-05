from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import InvalidArgumentError


def _maybe_fspath(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return os.fspath(value)


@dataclass(slots=True)
class EngineConfig:
    model_path: str | os.PathLike[str] | None = None
    diffusion_model_path: str | os.PathLike[str] | None = None
    high_noise_diffusion_model_path: str | os.PathLike[str] | None = None
    clip_l_path: str | os.PathLike[str] | None = None
    clip_g_path: str | os.PathLike[str] | None = None
    clip_vision_path: str | os.PathLike[str] | None = None
    t5xxl_path: str | os.PathLike[str] | None = None
    llm_path: str | os.PathLike[str] | None = None
    llm_vision_path: str | os.PathLike[str] | None = None
    vae_path: str | os.PathLike[str] | None = None
    taesd_path: str | os.PathLike[str] | None = None
    control_net_path: str | os.PathLike[str] | None = None
    backend: str | None = None
    n_threads: int | None = None
    weight_type: int | str | None = None
    tensor_type_rules: str | None = None
    use_mmap: bool | None = None
    offload_params_to_cpu: bool | None = None
    keep_text_encoder_on_cpu: bool | None = None
    keep_control_net_on_cpu: bool | None = None
    keep_vae_on_cpu: bool | None = None
    skip_t5: bool | None = None
    flash_attention: bool | None = None
    max_vram_gb: float | None = None
    vae_tiling: bool | None = None
    vae_tile_size: float | None = None
    cfg_parallel_size: int | None = None
    tp_parallel_size: int | None = None
    sp_parallel_size: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "model_path",
            "diffusion_model_path",
            "high_noise_diffusion_model_path",
            "clip_l_path",
            "clip_g_path",
            "clip_vision_path",
            "t5xxl_path",
            "llm_path",
            "llm_vision_path",
            "vae_path",
            "taesd_path",
            "control_net_path",
        ):
            setattr(self, field_name, _maybe_fspath(getattr(self, field_name)))
        self.validate()

    def validate(self) -> None:
        has_model = bool(self.model_path)
        has_components = (
            bool(self.diffusion_model_path)
            and bool(self.vae_path)
            and bool(self.clip_l_path)
            and (bool(self.t5xxl_path) or bool(self.skip_t5))
        )
        if not has_model and not has_components:
            raise InvalidArgumentError(
                "provide model_path or the full diffusion_model_path/vae_path/clip_l_path/"
                "(t5xxl_path or skip_t5) set"
            )

        if self.n_threads is not None and self.n_threads < 0:
            raise InvalidArgumentError("n_threads must be >= 0")

        for field_name in ("cfg_parallel_size", "tp_parallel_size", "sp_parallel_size"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise InvalidArgumentError(f"{field_name} must be > 0")

        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            raise InvalidArgumentError("max_vram_gb must be > 0")
        if self.vae_tile_size is not None and self.vae_tile_size <= 0:
            raise InvalidArgumentError("vae_tile_size must be > 0")


@dataclass(slots=True)
class ImageRequest:
    prompt: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    batch_count: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    image_cfg_scale: float | None = None
    guidance: float | None = None
    distilled_guidance: float | None = None
    eta: float | None = None
    flow_shift: float | None = None
    sampler: int | str | None = None
    scheduler: int | str | None = None
    cache_mode: int | str | None = None
    cache_reuse_threshold: float | None = None
    cache_start_percent: float | None = None
    cache_end_percent: float | None = None
    cache_error_decay_rate: float | None = None
    cache_use_relative_threshold: bool | None = None
    cache_reset_error_on_compute: bool | None = None
    cache_Fn_compute_blocks: int | None = None
    cache_Bn_compute_blocks: int | None = None
    cache_residual_diff_threshold: float | None = None
    cache_max_accumulated_residual_diff: float | None = None
    cache_max_warmup_steps: int | None = None
    cache_max_cached_steps: int | None = None
    cache_max_continuous_cached_steps: int | None = None
    cache_taylorseer_n_derivatives: int | None = None
    cache_taylorseer_skip_interval: int | None = None
    cache_scm_mask: str | None = None
    cache_scm_policy_dynamic: bool | None = None
    output_type: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> "ImageRequest":
        if "batch_size" in kwargs:
            if "batch_count" in kwargs:
                raise InvalidArgumentError("use only one of batch_count or batch_size")
            kwargs["batch_count"] = kwargs.pop("batch_size")
        return cls(**kwargs)

    @property
    def effective_guidance(self) -> float | None:
        if self.guidance is not None:
            return self.guidance
        return self.distilled_guidance

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise InvalidArgumentError("prompt is required")

        for field_name in ("width", "height", "steps", "batch_count"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise InvalidArgumentError(f"{field_name} must be > 0")

        if self.guidance is not None and self.distilled_guidance is not None:
            if self.guidance != self.distilled_guidance:
                raise InvalidArgumentError(
                    "guidance and distilled_guidance must match when both are provided"
                )

        for field_name in (
            "cache_reuse_threshold",
            "cache_residual_diff_threshold",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise InvalidArgumentError(f"{field_name} must be >= 0")

        if self.cache_max_accumulated_residual_diff is not None:
            if self.cache_max_accumulated_residual_diff < -1:
                raise InvalidArgumentError("cache_max_accumulated_residual_diff must be >= -1")

        for field_name in (
            "cache_max_warmup_steps",
            "cache_Fn_compute_blocks",
            "cache_Bn_compute_blocks",
            "cache_taylorseer_skip_interval",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise InvalidArgumentError(f"{field_name} must be >= 0")

        if self.cache_taylorseer_n_derivatives is not None and self.cache_taylorseer_n_derivatives < 1:
            raise InvalidArgumentError("cache_taylorseer_n_derivatives must be >= 1")

        if self.cache_error_decay_rate is not None:
            if not 0.0 <= self.cache_error_decay_rate <= 1.0:
                raise InvalidArgumentError("cache_error_decay_rate must be in [0, 1]")

        if self.cache_start_percent is not None or self.cache_end_percent is not None:
            start = self.cache_start_percent if self.cache_start_percent is not None else 0.15
            end = self.cache_end_percent if self.cache_end_percent is not None else 0.95
            if not (0.0 <= start < end <= 1.0):
                raise InvalidArgumentError(
                    "cache window must satisfy 0 <= cache_start_percent < cache_end_percent <= 1"
                )

        if self.output_type is not None and self.output_type not in {"pil", "numpy"}:
            raise InvalidArgumentError("output_type must be one of: pil, numpy")
