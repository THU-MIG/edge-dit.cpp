from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from ._capi import (
    EdAudio,
    EdContextParams,
    EdImage,
    EdImageBatch,
    EdImageGenerationParams,
    EdVideo,
    EdVideoGenerationParams,
    EdRefVideo,
    load_capi,
)
from ._strings import CStringPool
from .config import AudioInput, EngineConfig, ImageRequest, RefVideoInput, VideoRequest
from .enums import resolve_cache_mode, resolve_dtype, resolve_sampler, resolve_scheduler
from .errors import (
    EdgeDitError,
    EdgeDitClosedError,
    GenerationError,
    InvalidArgumentError,
    ModelLoadError,
    raise_for_status,
)
from .image import (
    batch_to_numpy_images,
    batch_to_pil_images,
    video_to_numpy_frames,
    video_to_pil_frames,
)


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
    if request.init_image is not None:
        lines.append(f"init_image={request.init_image.width}x{request.init_image.height}")
    if request.ref_images is not None:
        lines.append(f"ref_images={len(request.ref_images)}")
    if request.output_type is not None:
        lines.append(f"output_type={request.output_type!r}")
    return lines


def _coerce_native_input_image(image: Image.Image, *, field_name: str) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise InvalidArgumentError(f"{field_name} must be a PIL.Image.Image")
    if image.mode in {"L", "RGB", "RGBA"}:
        return image
    return image.convert("RGBA" if "A" in image.getbands() else "RGB")


def _build_native_image(image: Image.Image, *, field_name: str) -> tuple[EdImage, object]:
    normalized = _coerce_native_input_image(image, field_name=field_name)
    raw = normalized.tobytes()
    buffer = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    native = EdImage(
        width=normalized.width,
        height=normalized.height,
        channels=len(normalized.getbands()),
        data=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)),
    )
    return native, buffer


def _build_native_audio(audio: AudioInput, *, field_name: str) -> tuple[EdAudio, object]:
    values = audio.samples
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        values = values.reshape(-1).tolist()  # type: ignore[union-attr]
    try:
        flattened = [float(value) for value in values]  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError(f"{field_name}.samples must contain numeric values") from exc
    if not flattened or len(flattened) % audio.channels:
        raise InvalidArgumentError(f"{field_name}.samples must be divisible by channels")
    buffer = (ctypes.c_float * len(flattened))(*flattened)
    native = EdAudio(
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        sample_count=len(flattened) // audio.channels,
        data=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_float)),
    )
    return native, buffer


class VideoOutput(list[object]):
    """Generated frames plus an optional copied interleaved float32 soundtrack."""

    def __init__(
        self,
        frames: list[object],
        *,
        audio: list[float] | None = None,
        audio_sample_rate: int = 0,
        audio_channels: int = 0,
    ) -> None:
        super().__init__(frames)
        self.audio = audio
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels


def _summarize_video_request(request: VideoRequest) -> list[str]:
    lines: list[str] = []
    prompt = request.prompt.strip()
    if len(prompt) > 80:
        prompt = prompt[:77] + "..."
    lines.append(f"prompt={prompt!r}")
    if request.width is not None or request.height is not None:
        lines.append(f"size={request.width or '?'}x{request.height or '?'}")
    if request.frames is not None:
        lines.append(f"frames={request.frames}")
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

    @property
    def pipeline_name(self) -> str | None:
        self._ensure_open()
        raw = self._api.ed_context_pipeline_name(self._ctx)
        return raw.decode("utf-8", errors="replace") if raw else None

    @property
    def version_name(self) -> str | None:
        self._ensure_open()
        raw = self._api.ed_context_version_name(self._ctx)
        return raw.decode("utf-8", errors="replace") if raw else None

    @property
    def supports_image(self) -> bool:
        self._ensure_open()
        return bool(self._api.ed_context_supports_image(self._ctx))

    @property
    def supports_video(self) -> bool:
        self._ensure_open()
        return bool(self._api.ed_context_supports_video(self._ctx))

    @property
    def default_sampler(self) -> int:
        self._ensure_open()
        return int(self._api.ed_context_default_sampler(self._ctx))

    def default_scheduler(self, sampler: int | str | None = None) -> int:
        self._ensure_open()
        resolved_sampler = -1 if sampler is None else resolve_sampler(sampler)
        return int(self._api.ed_context_default_scheduler(self._ctx, resolved_sampler))

    def request_cancel(self) -> None:
        self._ensure_open()
        self._api.ed_context_request_cancel(self._ctx)

    def progress_steps(self) -> tuple[int, int]:
        self._ensure_open()
        current = int(self._api.ed_context_progress_current_step(self._ctx))
        total = int(self._api.ed_context_progress_total_steps(self._ctx))
        return current, total

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

    def generate_video(
        self,
        request: VideoRequest | None = None,
        /,
        **kwargs: object,
    ) -> list[object]:
        self._ensure_open()

        if request is not None and kwargs:
            raise TypeError("pass either a VideoRequest or keyword arguments, not both")

        if request is None:
            request = VideoRequest.from_kwargs(**kwargs)
        elif kwargs:
            raise TypeError("unexpected keyword arguments")

        with self._lock:
            self._ensure_open()
            return self._generate_video_locked(request)

    def _generate_image_locked(self, request: ImageRequest) -> list[object]:
        params = EdImageGenerationParams()
        batch = EdImageBatch()
        strings = CStringPool()
        keepalive: list[object] = []

        self._api.ed_image_generation_params_init(ctypes.byref(params))
        self._apply_request(params, request, strings, keepalive)

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

    def _generate_video_locked(self, request: VideoRequest) -> VideoOutput:
        params = EdVideoGenerationParams()
        video = EdVideo()
        strings = CStringPool()
        keepalive: list[object] = []

        self._api.ed_video_generation_params_init(ctypes.byref(params))
        self._apply_video_request(params, request, strings, keepalive)

        status = self._api.ed_generate_video(self._ctx, ctypes.byref(params), ctypes.byref(video))

        try:
            try:
                raise_for_status(
                    status,
                    lib=self._api,
                    ctx=self._ctx,
                    default_message="video generation failed",
                )
            except EdgeDitError as exc:
                raise type(exc)(
                    _append_context(
                        str(exc),
                        [
                            _describe_path(self._config.model_path, "model_path"),
                            f"backend={self._config.backend!r}" if self._config.backend else None,
                            *_summarize_video_request(request),
                        ],
                    )
                ) from exc

            if not self._api.ed_context_parallel_is_root(self._ctx):
                return VideoOutput([])

            output_type = request.output_type or "pil"
            if output_type == "numpy":
                frames = video_to_numpy_frames(video)
            else:
                frames = video_to_pil_frames(video)
            if not frames:
                raise GenerationError("generation succeeded but output is empty")
            audio = None
            if video.audio and video.audio_sample_count > 0 and video.audio_channels > 0:
                value_count = video.audio_sample_count * video.audio_channels
                audio = [float(video.audio[index]) for index in range(value_count)]
            return VideoOutput(
                frames,
                audio=audio,
                audio_sample_rate=video.audio_sample_rate,
                audio_channels=video.audio_channels,
            )
        finally:
            self._api.ed_free_video(ctypes.byref(video))

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
        params.audio_vae_path = strings.add_optional(config.audio_vae_path)
        params.taesd_path = strings.add_optional(config.taesd_path)
        params.control_net_path = strings.add_optional(config.control_net_path)
        params.embeddings_connectors_path = strings.add_optional(config.embeddings_connectors_path)
        params.latent_upscaler_path = strings.add_optional(config.latent_upscaler_path)
        params.tensor_type_rules = strings.add_optional(config.tensor_type_rules)

        if config.n_threads is not None:
            params.n_threads = config.n_threads
        if config.weight_type is not None:
            params.weight_type = resolve_dtype(config.weight_type)
        if config.use_mmap is not None:
            params.use_mmap = config.use_mmap
        if config.offload_params_to_cpu is not None:
            params.offload_params_to_cpu = config.offload_params_to_cpu
        if config.dit_offload is not None:
            params.dit_offload = config.dit_offload
        if config.text_encoder_offload is not None:
            params.text_encoder_offload = config.text_encoder_offload
        if config.minimax_h3_stage_lifecycle is not None:
            params.minimax_h3_stage_lifecycle = config.minimax_h3_stage_lifecycle
        if config.auto_allocate is not None:
            params.auto_allocate = config.auto_allocate
        if config.auto_fit is not None:
            params.auto_fit = config.auto_fit
        if config.fit_width is not None:
            params.fit_width = config.fit_width
        if config.fit_height is not None:
            params.fit_height = config.fit_height
        if config.fit_frames is not None:
            params.fit_frames = config.fit_frames
        if config.fit_fps is not None:
            params.fit_fps = config.fit_fps
        if config.keep_control_net_on_cpu is not None:
            params.keep_control_net_on_cpu = config.keep_control_net_on_cpu
        if config.vae_offload is not None:
            params.vae_offload = config.vae_offload
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
        keepalive: list[object],
    ) -> None:
        params.prompt = strings.add_optional(request.prompt)
        params.negative_prompt = strings.add_optional(request.negative_prompt)

        if request.width is not None:
            params.width = request.width
        if request.height is not None:
            params.height = request.height
        if request.seed is not None:
            params.seed = request.seed
        if request.batch_count is not None:
            params.batch_count = request.batch_count
        if request.init_image is not None:
            native_image, raw_buffer = _build_native_image(
                request.init_image,
                field_name="init_image",
            )
            native_image_ptr = ctypes.pointer(native_image)
            params.init_image = native_image_ptr
            keepalive.extend([raw_buffer, native_image, native_image_ptr])
        if request.mask_image is not None:
            native_image, raw_buffer = _build_native_image(
                request.mask_image,
                field_name="mask_image",
            )
            native_image_ptr = ctypes.pointer(native_image)
            params.mask_image = native_image_ptr
            keepalive.extend([raw_buffer, native_image, native_image_ptr])
        if request.control_image is not None:
            native_image, raw_buffer = _build_native_image(
                request.control_image,
                field_name="control_image",
            )
            native_image_ptr = ctypes.pointer(native_image)
            params.control_image = native_image_ptr
            keepalive.extend([raw_buffer, native_image, native_image_ptr])
        if request.ref_images is not None:
            native_ref_images: list[EdImage] = []
            for index, image in enumerate(request.ref_images):
                native_image, raw_buffer = _build_native_image(
                    image,
                    field_name=f"ref_images[{index}]",
                )
                native_ref_images.append(native_image)
                keepalive.extend([raw_buffer, native_image])
            native_array = (EdImage * len(native_ref_images))(*native_ref_images)
            params.ref_images = ctypes.cast(native_array, ctypes.POINTER(EdImage))
            params.ref_image_count = len(native_ref_images)
            keepalive.append(native_array)

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

    @staticmethod
    def _apply_video_request(
        params: EdVideoGenerationParams,
        request: VideoRequest,
        strings: CStringPool,
        keepalive: list[object],
    ) -> None:
        params.prompt = strings.add_optional(request.prompt)
        params.negative_prompt = strings.add_optional(request.negative_prompt or "")

        if request.width is not None:
            params.width = request.width
        if request.height is not None:
            params.height = request.height
        if request.frames is not None:
            params.frames = request.frames
        if request.fps is not None:
            params.fps = request.fps
        if request.seed is not None:
            params.seed = request.seed

        for field_name in ("init_image", "end_image"):
            image = getattr(request, field_name)
            if image is not None:
                native, raw = _build_native_image(image, field_name=field_name)
                pointer = ctypes.pointer(native)
                setattr(params, field_name, pointer)
                keepalive.extend([raw, native, pointer])

        if request.ref_images:
            images: list[EdImage] = []
            for index, image in enumerate(request.ref_images):
                native, raw = _build_native_image(image, field_name=f"ref_images[{index}]")
                images.append(native)
                keepalive.extend([raw, native])
            array = (EdImage * len(images))(*images)
            params.ref_images = ctypes.cast(array, ctypes.POINTER(EdImage))
            params.ref_image_count = len(images)
            keepalive.append(array)
        if request.ref_image_size is not None:
            params.ref_image_size = (
                1 if request.ref_image_size == "match" else 0
                if request.ref_image_size == "max" else int(request.ref_image_size)
            )

        if request.ref_videos:
            videos: list[EdRefVideo] = []
            for video_index, video in enumerate(request.ref_videos):
                native_frames: list[EdImage] = []
                for frame_index, frame in enumerate(video.frames):
                    native, raw = _build_native_image(
                        frame,
                        field_name=f"ref_videos[{video_index}].frames[{frame_index}]",
                    )
                    native_frames.append(native)
                    keepalive.extend([raw, native])
                frame_array = (EdImage * len(native_frames))(*native_frames)
                native_audio = EdAudio()
                if video.audio is not None:
                    native_audio, audio_buffer = _build_native_audio(
                        video.audio, field_name=f"ref_videos[{video_index}].audio"
                    )
                    keepalive.append(audio_buffer)
                videos.append(
                    EdRefVideo(
                        frames=ctypes.cast(frame_array, ctypes.POINTER(EdImage)),
                        frame_count=len(native_frames),
                        fps=video.fps,
                        audio=native_audio,
                    )
                )
                keepalive.append(frame_array)
            video_array = (EdRefVideo * len(videos))(*videos)
            params.ref_videos = ctypes.cast(video_array, ctypes.POINTER(EdRefVideo))
            params.ref_video_count = len(videos)
            keepalive.append(video_array)

        if request.ref_audios:
            audios: list[EdAudio] = []
            for index, audio in enumerate(request.ref_audios):
                native, raw = _build_native_audio(audio, field_name=f"ref_audios[{index}]")
                audios.append(native)
                keepalive.extend([raw, native])
            audio_array = (EdAudio * len(audios))(*audios)
            params.ref_audios = ctypes.cast(audio_array, ctypes.POINTER(EdAudio))
            params.ref_audio_count = len(audios)
            keepalive.append(audio_array)

        if request.control_frames:
            control: list[EdImage] = []
            for index, image in enumerate(request.control_frames):
                native, raw = _build_native_image(image, field_name=f"control_frames[{index}]")
                control.append(native)
                keepalive.extend([raw, native])
            control_array = (EdImage * len(control))(*control)
            params.control_frames = ctypes.cast(control_array, ctypes.POINTER(EdImage))
            params.control_frame_count = len(control)
            keepalive.append(control_array)
        if request.strength is not None:
            params.strength = request.strength
        if request.vace_strength is not None:
            params.vace_strength = request.vace_strength
        if request.moe_boundary is not None:
            params.moe_boundary = request.moe_boundary

        if request.steps is not None:
            params.sample.steps = request.steps
        if request.cfg_scale is not None:
            params.sample.cfg_scale = request.cfg_scale
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
        for field_name in (
            "cache_reuse_threshold", "cache_start_percent", "cache_end_percent",
            "cache_error_decay_rate", "cache_use_relative_threshold",
            "cache_reset_error_on_compute", "cache_Fn_compute_blocks",
            "cache_Bn_compute_blocks", "cache_residual_diff_threshold",
            "cache_max_accumulated_residual_diff", "cache_max_warmup_steps",
            "cache_max_cached_steps", "cache_max_continuous_cached_steps",
            "cache_taylorseer_n_derivatives", "cache_taylorseer_skip_interval",
            "cache_scm_policy_dynamic",
        ):
            value = getattr(request, field_name)
            if value is not None:
                setattr(params.sample, field_name, value)
        if request.cache_mode is not None:
            params.sample.cache_mode = resolve_cache_mode(request.cache_mode)
        if request.cache_scm_mask is not None:
            params.sample.cache_scm_mask = strings.add_optional(request.cache_scm_mask)
        if request.hires is not None:
            params.hires_enabled = request.hires
        if request.hires_steps is not None:
            params.hires_steps = request.hires_steps
        if request.hires_denoising_strength is not None:
            params.hires_denoising_strength = request.hires_denoising_strength
        if request.hires_sigmas is not None:
            sigmas = [float(value) for value in request.hires_sigmas]
            sigma_array = (ctypes.c_float * len(sigmas))(*sigmas)
            params.hires_sigmas = ctypes.cast(sigma_array, ctypes.POINTER(ctypes.c_float))
            params.hires_sigmas_count = len(sigmas)
            keepalive.append(sigma_array)
