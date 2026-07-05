from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from ._capi import EdContextParams, EdImageBatch, EdImageGenerationParams, load_capi
from ._strings import CStringPool
from .config import EngineConfig, ImageRequest
from .enums import resolve_cache_mode, resolve_dtype, resolve_sampler, resolve_scheduler
from .errors import (
    EdgeDitError,
    EdgeDitClosedError,
    GenerationError,
    InvalidArgumentError,
    ModelLoadError,
    raise_for_status,
)
from .image import batch_to_numpy_images, batch_to_pil_images


@contextmanager
def _temporary_backend(backend: str | None):
    if not backend:
        yield
        return

    had_previous = "ED_BACKEND" in os.environ
    previous = os.environ.get("ED_BACKEND")
    os.environ["ED_BACKEND"] = backend
    try:
        yield
    finally:
        if had_previous:
            assert previous is not None
            os.environ["ED_BACKEND"] = previous
        else:
            os.environ.pop("ED_BACKEND", None)


def _append_context(message: str, lines: list[str]) -> str:
    filtered = [line for line in lines if line]
    if not filtered:
        return message
    return f"{message}\nContext:\n" + "\n".join(f"  - {line}" for line in filtered)


def _describe_path(path: str | os.PathLike[str] | None, label: str) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.exists():
        return f"{label}={resolved}"
    return f"{label}={resolved} (missing)"


def _summarize_request(request: ImageRequest) -> list[str]:
    lines: list[str] = []
    prompt = request.prompt.strip()
    if len(prompt) > 80:
        prompt = prompt[:77] + "..."
    lines.append(f"prompt={prompt!r}")
    if request.width is not None or request.height is not None:
        lines.append(f"size={request.width or '?'}x{request.height or '?'}")
    if request.steps is not None:
        lines.append(f"steps={request.steps}")
    if request.seed is not None:
        lines.append(f"seed={request.seed}")
    if request.output_type is not None:
        lines.append(f"output_type={request.output_type!r}")
    return lines


class Engine:
    def __init__(
        self,
        config: EngineConfig | str | os.PathLike[str] | None = None,
        /,
        *,
        _library: object | None = None,
        _library_path: str | None = None,
        **config_kwargs: object,
    ) -> None:
        self._api = load_capi(path=_library_path, library=_library)
        self._lock = threading.Lock()
        self._closed = False
        self._ctx = None

        if isinstance(config, EngineConfig):
            if config_kwargs:
                raise TypeError("config kwargs are not allowed when config is already provided")
            normalized_config = config
        elif config is None:
            normalized_config = EngineConfig(**config_kwargs)
        else:
            if "model_path" in config_kwargs:
                raise TypeError("model_path was provided twice")
            normalized_config = EngineConfig(model_path=config, **config_kwargs)

        self._config = normalized_config
        self._config_strings = CStringPool()

        ctx_params = EdContextParams()
        self._api.ed_context_params_init(ctypes.byref(ctx_params))
        self._apply_config(ctx_params, self._config, self._config_strings)

        with _temporary_backend(self._config.backend):
            ctx = self._api.ed_create_context(ctypes.byref(ctx_params))

        if not ctx:
            message = _append_context(
                "failed to create edge-dit context; check native logs for details",
                [
                    _describe_path(self._config.model_path, "model_path"),
                    _describe_path(self._config.diffusion_model_path, "diffusion_model_path"),
                    _describe_path(self._config.vae_path, "vae_path"),
                    _describe_path(self._config.clip_l_path, "clip_l_path"),
                    _describe_path(self._config.t5xxl_path, "t5xxl_path"),
                    f"backend={self._config.backend!r}" if self._config.backend else None,
                    (
                        f"max_vram_gb={self._config.max_vram_gb}"
                        if self._config.max_vram_gb is not None
                        else None
                    ),
                    "if you are using Conda, also verify libstdc++ / GLIBCXX compatibility",
                ],
            )
            raise ModelLoadError(message)

        self._ctx = ctx

    @property
    def config(self) -> EngineConfig:
        return self._config

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._ctx:
                self._api.ed_free_context(self._ctx)
                self._ctx = None

    def __enter__(self) -> "Engine":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def generate_image(
        self,
        request: ImageRequest | None = None,
        /,
        **kwargs: object,
    ) -> list[object]:
        self._ensure_open()

        if request is not None and kwargs:
            raise TypeError("pass either an ImageRequest or keyword arguments, not both")

        if request is None:
            request = ImageRequest.from_kwargs(**kwargs)
        elif kwargs:
            raise TypeError("unexpected keyword arguments")

        with self._lock:
            self._ensure_open()
            return self._generate_image_locked(request)

    def _generate_image_locked(self, request: ImageRequest) -> list[object]:
        params = EdImageGenerationParams()
        batch = EdImageBatch()
        strings = CStringPool()

        self._api.ed_image_generation_params_init(ctypes.byref(params))
        self._apply_request(params, request, strings)

        status = self._api.ed_generate_image(self._ctx, ctypes.byref(params), ctypes.byref(batch))

        try:
            try:
                raise_for_status(
                    status,
                    lib=self._api,
                    ctx=self._ctx,
                    default_message="image generation failed",
                )
            except EdgeDitError as exc:
                raise type(exc)(
                    _append_context(
                        str(exc),
                        [
                            _describe_path(self._config.model_path, "model_path"),
                            f"backend={self._config.backend!r}" if self._config.backend else None,
                            *_summarize_request(request),
                        ],
                    )
                ) from exc

            if not self._api.ed_context_parallel_is_root(self._ctx):
                return []

            output_type = request.output_type or "pil"
            if output_type == "numpy":
                images = batch_to_numpy_images(batch)
            else:
                images = batch_to_pil_images(batch)
            if not images:
                raise GenerationError("generation succeeded but output is empty")
            return images
        finally:
            self._api.ed_free_image_batch(ctypes.byref(batch))

    def _ensure_open(self) -> None:
        if self._closed or self._ctx is None:
            raise EdgeDitClosedError("engine is already closed")

    @staticmethod
    def _apply_config(params: EdContextParams, config: EngineConfig, strings: CStringPool) -> None:
        params.model_path = strings.add_optional(config.model_path)
        params.diffusion_model_path = strings.add_optional(config.diffusion_model_path)
        params.high_noise_diffusion_model_path = strings.add_optional(
            config.high_noise_diffusion_model_path
        )
        params.clip_l_path = strings.add_optional(config.clip_l_path)
        params.clip_g_path = strings.add_optional(config.clip_g_path)
        params.clip_vision_path = strings.add_optional(config.clip_vision_path)
        params.t5xxl_path = strings.add_optional(config.t5xxl_path)
        params.llm_path = strings.add_optional(config.llm_path)
        params.llm_vision_path = strings.add_optional(config.llm_vision_path)
        params.vae_path = strings.add_optional(config.vae_path)
        params.taesd_path = strings.add_optional(config.taesd_path)
        params.control_net_path = strings.add_optional(config.control_net_path)
        params.tensor_type_rules = strings.add_optional(config.tensor_type_rules)

        if config.n_threads is not None:
            params.n_threads = config.n_threads
        if config.weight_type is not None:
            params.weight_type = resolve_dtype(config.weight_type)
        if config.use_mmap is not None:
            params.use_mmap = config.use_mmap
        if config.offload_params_to_cpu is not None:
            params.offload_params_to_cpu = config.offload_params_to_cpu
        if config.keep_text_encoder_on_cpu is not None:
            params.keep_text_encoder_on_cpu = config.keep_text_encoder_on_cpu
        if config.keep_control_net_on_cpu is not None:
            params.keep_control_net_on_cpu = config.keep_control_net_on_cpu
        if config.keep_vae_on_cpu is not None:
            params.keep_vae_on_cpu = config.keep_vae_on_cpu
        if config.skip_t5 is not None:
            params.skip_t5 = config.skip_t5
        if config.flash_attention is not None:
            params.flash_attention = config.flash_attention
        if config.max_vram_gb is not None:
            params.max_vram_gb = config.max_vram_gb
        if config.vae_tiling is not None:
            params.vae_tiling.enabled = config.vae_tiling
        if config.vae_tile_size is not None:
            params.vae_tiling.enabled = True
            params.vae_tiling.rel_size_x = config.vae_tile_size
            params.vae_tiling.rel_size_y = config.vae_tile_size
        if config.cfg_parallel_size is not None:
            params.cfg_parallel_size = config.cfg_parallel_size
        if config.tp_parallel_size is not None:
            params.tp_parallel_size = config.tp_parallel_size
        if config.sp_parallel_size is not None:
            params.sp_parallel_size = config.sp_parallel_size

    @staticmethod
    def _apply_request(
        params: EdImageGenerationParams,
        request: ImageRequest,
        strings: CStringPool,
    ) -> None:
        params.prompt = strings.add_optional(request.prompt)
        params.negative_prompt = strings.add_optional(request.negative_prompt or "")

        if request.width is not None:
            params.width = request.width
        if request.height is not None:
            params.height = request.height
        if request.seed is not None:
            params.seed = request.seed
        if request.batch_count is not None:
            params.batch_count = request.batch_count

        if request.steps is not None:
            params.sample.steps = request.steps
        if request.cfg_scale is not None:
            params.sample.cfg_scale = request.cfg_scale
        if request.image_cfg_scale is not None:
            params.sample.image_cfg_scale = request.image_cfg_scale
        if request.effective_guidance is not None:
            params.sample.distilled_guidance = request.effective_guidance
        if request.eta is not None:
            params.sample.eta = request.eta
        if request.flow_shift is not None:
            params.sample.flow_shift = request.flow_shift
        if request.sampler is not None:
            params.sample.sampler = resolve_sampler(request.sampler)
        if request.scheduler is not None:
            params.sample.scheduler = resolve_scheduler(request.scheduler)
        if request.cache_mode is not None:
            params.sample.cache_mode = resolve_cache_mode(request.cache_mode)
        if request.cache_reuse_threshold is not None:
            params.sample.cache_reuse_threshold = request.cache_reuse_threshold
        if request.cache_start_percent is not None:
            params.sample.cache_start_percent = request.cache_start_percent
        if request.cache_end_percent is not None:
            params.sample.cache_end_percent = request.cache_end_percent
        if request.cache_error_decay_rate is not None:
            params.sample.cache_error_decay_rate = request.cache_error_decay_rate
        if request.cache_use_relative_threshold is not None:
            params.sample.cache_use_relative_threshold = request.cache_use_relative_threshold
        if request.cache_reset_error_on_compute is not None:
            params.sample.cache_reset_error_on_compute = request.cache_reset_error_on_compute
        if request.cache_Fn_compute_blocks is not None:
            params.sample.cache_Fn_compute_blocks = request.cache_Fn_compute_blocks
        if request.cache_Bn_compute_blocks is not None:
            params.sample.cache_Bn_compute_blocks = request.cache_Bn_compute_blocks
        if request.cache_residual_diff_threshold is not None:
            params.sample.cache_residual_diff_threshold = request.cache_residual_diff_threshold
        if request.cache_max_accumulated_residual_diff is not None:
            params.sample.cache_max_accumulated_residual_diff = (
                request.cache_max_accumulated_residual_diff
            )
        if request.cache_max_warmup_steps is not None:
            params.sample.cache_max_warmup_steps = request.cache_max_warmup_steps
        if request.cache_max_cached_steps is not None:
            params.sample.cache_max_cached_steps = request.cache_max_cached_steps
        if request.cache_max_continuous_cached_steps is not None:
            params.sample.cache_max_continuous_cached_steps = request.cache_max_continuous_cached_steps
        if request.cache_taylorseer_n_derivatives is not None:
            params.sample.cache_taylorseer_n_derivatives = request.cache_taylorseer_n_derivatives
        if request.cache_taylorseer_skip_interval is not None:
            params.sample.cache_taylorseer_skip_interval = request.cache_taylorseer_skip_interval
        if request.cache_scm_mask is not None:
            params.sample.cache_scm_mask = strings.add_optional(request.cache_scm_mask)
        if request.cache_scm_policy_dynamic is not None:
            params.sample.cache_scm_policy_dynamic = request.cache_scm_policy_dynamic
