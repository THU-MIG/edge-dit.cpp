from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

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
    audio_vae_path: str | os.PathLike[str] | None = None
    taesd_path: str | os.PathLike[str] | None = None
    control_net_path: str | os.PathLike[str] | None = None
    embeddings_connectors_path: str | os.PathLike[str] | None = None
    latent_upscaler_path: str | os.PathLike[str] | None = None
    backend: str | None = None
    n_threads: int | None = None
    weight_type: int | str | None = None
    tensor_type_rules: str | None = None
    use_mmap: bool | None = None
    offload_params_to_cpu: bool | None = None
    dit_offload: bool | None = None
    text_encoder_offload: bool | None = None
    minimax_h3_stage_lifecycle: bool | None = None
    auto_allocate: bool | None = None
    auto_fit: bool | None = None
    fit_width: int | None = None
    fit_height: int | None = None
    fit_frames: int | None = None
    fit_fps: int | None = None
    keep_control_net_on_cpu: bool | None = None
    vae_offload: bool | None = None
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
            "audio_vae_path",
            "taesd_path",
            "control_net_path",
            "embeddings_connectors_path",
            "latent_upscaler_path",
        ):
            setattr(self, field_name, _maybe_fspath(getattr(self, field_name)))
        self.validate()

    def validate(self) -> None:
        has_model = bool(self.model_path)
        has_image_components = (
            bool(self.diffusion_model_path)
            and bool(self.vae_path)
            and bool(self.clip_l_path)
            and (bool(self.t5xxl_path) or bool(self.skip_t5))
        )
        has_minimax_components = (
            bool(self.diffusion_model_path) and bool(self.vae_path) and bool(self.llm_path)
        )
        if not has_model and not has_image_components and not has_minimax_components:
            raise InvalidArgumentError(
                "provide model_path, the image diffusion_model_path/vae_path/clip_l_path/"
                "(t5xxl_path or skip_t5) set, or the MiniMax-H3 "
                "diffusion_model_path/vae_path/llm_path set"
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

        for field_name in ("fit_width", "fit_height", "fit_frames", "fit_fps"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise InvalidArgumentError(f"{field_name} must be >= 0")


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
    init_image: Image.Image | None = None
    mask_image: Image.Image | None = None
    control_image: Image.Image | None = None
    ref_images: list[Image.Image] | tuple[Image.Image, ...] | None = None
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

        for field_name in ("init_image", "mask_image", "control_image"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Image.Image):
                raise InvalidArgumentError(f"{field_name} must be a PIL.Image.Image")

        if self.ref_images is not None:
            if not isinstance(self.ref_images, (list, tuple)):
                raise InvalidArgumentError("ref_images must be a list or tuple of PIL.Image.Image")
            if not self.ref_images:
                raise InvalidArgumentError("ref_images must contain at least one image")
            for image in self.ref_images:
                if not isinstance(image, Image.Image):
                    raise InvalidArgumentError("ref_images must contain only PIL.Image.Image values")

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


@dataclass(slots=True)
class AudioInput:
    samples: Sequence[float] | object
    sample_rate: int
    channels: int = 1

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise InvalidArgumentError("audio sample_rate must be > 0")
        if self.channels <= 0:
            raise InvalidArgumentError("audio channels must be > 0")
        try:
            size = getattr(self.samples, "size", None)
            count = int(size if size is not None else len(self.samples))  # type: ignore[arg-type]
        except TypeError as exc:
            raise InvalidArgumentError("audio samples must be a sized sequence or numpy array") from exc
        if count <= 0 or count % self.channels != 0:
            raise InvalidArgumentError("audio samples must be non-empty interleaved data divisible by channels")

    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class RefVideoInput:
    frames: Sequence[Image.Image]
    fps: int = 24
    audio: AudioInput | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise InvalidArgumentError("reference video fps must be > 0")
        if not self.frames:
            raise InvalidArgumentError("reference video frames must not be empty")
        if any(not isinstance(frame, Image.Image) for frame in self.frames):
            raise InvalidArgumentError("reference video frames must contain only PIL.Image.Image values")


@dataclass(slots=True)
class VideoRequest:
    prompt: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    guidance: float | None = None
    distilled_guidance: float | None = None
    eta: float | None = None
    flow_shift: float | None = None
    sampler: int | str | None = None
    scheduler: int | str | None = None
    init_image: Image.Image | None = None
    end_image: Image.Image | None = None
    ref_images: Sequence[Image.Image] | None = None
    ref_image_size: str | int | None = None
    ref_videos: Sequence[RefVideoInput] | None = None
    ref_audios: Sequence[AudioInput] | None = None
    control_frames: Sequence[Image.Image] | None = None
    strength: float | None = None
    vace_strength: float | None = None
    moe_boundary: float | None = None
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
    hires: bool | None = None
    hires_steps: int | None = None
    hires_denoising_strength: float | None = None
    hires_sigmas: Sequence[float] | None = None
    output_type: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> "VideoRequest":
        return cls(**kwargs)

    @property
    def effective_guidance(self) -> float | None:
        if self.guidance is not None:
            return self.guidance
        return self.distilled_guidance

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise InvalidArgumentError("prompt is required")

        for field_name in ("width", "height", "frames", "fps", "steps", "hires_steps"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise InvalidArgumentError(f"{field_name} must be > 0")

        if self.hires_denoising_strength is not None and not 0.0 < self.hires_denoising_strength <= 1.0:
            raise InvalidArgumentError("hires_denoising_strength must be in (0, 1]")
        if self.hires_sigmas is not None:
            try:
                sigmas = [float(value) for value in self.hires_sigmas]
            except TypeError as exc:
                raise InvalidArgumentError("hires_sigmas must be a numeric sequence") from exc
            except (ValueError, OverflowError) as exc:
                raise InvalidArgumentError("hires_sigmas must be a numeric sequence") from exc
            if len(sigmas) < 2:
                raise InvalidArgumentError("hires_sigmas must contain at least two values")
            if any(
                not math.isfinite(value)
                or value < 0.0
                or (index > 0 and value > sigmas[index - 1])
                for index, value in enumerate(sigmas)
            ):
                raise InvalidArgumentError(
                    "hires_sigmas must be finite, non-negative, and non-increasing"
                )

        for field_name in ("init_image", "end_image"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Image.Image):
                raise InvalidArgumentError(f"{field_name} must be a PIL.Image.Image")
        for field_name in ("ref_images", "control_frames"):
            values = getattr(self, field_name)
            if values is not None:
                if not values or any(not isinstance(value, Image.Image) for value in values):
                    raise InvalidArgumentError(f"{field_name} must contain PIL.Image.Image values")
        if self.ref_videos is not None and any(not isinstance(v, RefVideoInput) for v in self.ref_videos):
            raise InvalidArgumentError("ref_videos must contain RefVideoInput values")
        if self.ref_audios is not None and any(not isinstance(a, AudioInput) for a in self.ref_audios):
            raise InvalidArgumentError("ref_audios must contain AudioInput values")
        if self.ref_audios and not (self.ref_images or self.ref_videos):
            raise InvalidArgumentError("ref_audios require at least one ref_images or ref_videos entry")
        if isinstance(self.ref_image_size, str) and self.ref_image_size not in {"max", "match"}:
            raise InvalidArgumentError("ref_image_size must be 'max', 'match', 0, or 1")
        if isinstance(self.ref_image_size, int) and self.ref_image_size not in {0, 1}:
            raise InvalidArgumentError("ref_image_size must be 'max', 'match', 0, or 1")

        if self.guidance is not None and self.distilled_guidance is not None:
            if self.guidance != self.distilled_guidance:
                raise InvalidArgumentError(
                    "guidance and distilled_guidance must match when both are provided"
                )

        if self.cache_start_percent is not None or self.cache_end_percent is not None:
            start = self.cache_start_percent if self.cache_start_percent is not None else 0.15
            end = self.cache_end_percent if self.cache_end_percent is not None else 0.95
            if not 0.0 <= start < end <= 1.0:
                raise InvalidArgumentError("cache window must satisfy 0 <= start < end <= 1")

        if self.output_type is not None and self.output_type not in {"pil", "numpy"}:
            raise InvalidArgumentError("output_type must be one of: pil, numpy")
