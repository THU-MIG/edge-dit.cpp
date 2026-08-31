from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from PIL import Image

from . import __version__
from .config import AudioInput, EngineConfig, ImageRequest, RefVideoInput, VideoRequest
from .engine import Engine
from .errors import EdgeDitError, GenerationCancelledError, InvalidArgumentError, UnsupportedError

_API_PREFIXES = ("/ed/v2", "/edgedit/v2", "/edge-dit/v2")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _display_model_name(config: EngineConfig) -> str:
    if config.model_path:
        return config.model_path
    if config.diffusion_model_path:
        return config.diffusion_model_path
    return "edge-dit-model"


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _base64_png(image: Image.Image) -> str:
    return base64.b64encode(_png_bytes(image)).decode("ascii")


def _find_ffmpeg() -> str | None:
    configured = os.environ.get("EDGE_DIT_FFMPEG")
    if configured and Path(configured).is_file():
        return configured

    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return bundled if bundled and Path(bundled).is_file() else None


def _encode_video_result_mp4(result: dict[str, object], *, fps: int) -> bytes:
    """Encode an in-memory video result to an MP4 download with ffmpeg."""

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to save generated videos as MP4")

    frames = result.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("video result does not contain any frames")

    with tempfile.TemporaryDirectory(prefix="edge-dit-video-") as temp_dir:
        work_dir = Path(temp_dir)
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("b64_png"), str):
                raise ValueError(f"video frame {index} is missing b64_png data")
            try:
                payload = base64.b64decode(frame["b64_png"], validate=True)
            except Exception as exc:
                raise ValueError(f"video frame {index} contains invalid base64 PNG data") from exc
            (work_dir / f"frame-{index:08d}.png").write_bytes(payload)

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(work_dir / "frame-%08d.png"),
        ]

        audio = result.get("audio")
        has_audio = False
        if isinstance(audio, dict) and isinstance(audio.get("b64_f32le"), str):
            sample_rate = int(audio.get("sample_rate") or 0)
            channels = int(audio.get("channels") or 0)
            if sample_rate > 0 and channels > 0:
                try:
                    audio_payload = base64.b64decode(audio["b64_f32le"], validate=True)
                except Exception as exc:
                    raise ValueError("video audio contains invalid base64 float32 data") from exc
                audio_path = work_dir / "audio.f32le"
                audio_path.write_bytes(audio_payload)
                command.extend(
                    [
                        "-f",
                        "f32le",
                        "-ar",
                        str(sample_rate),
                        "-ac",
                        str(channels),
                        "-i",
                        str(audio_path),
                    ]
                )
                has_audio = True

        output_path = work_dir / "output.mp4"
        command.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
        if has_audio:
            command.extend(["-c:a", "aac", "-shortest"])
        command.append(str(output_path))

        completed = subprocess.run(command, capture_output=True, check=False, timeout=300)
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode the generated video: {message or 'unknown error'}")
        return output_path.read_bytes()


def _load_pil_image(raw: bytes, *, field_name: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            return image.copy()
    except Exception as exc:
        raise InvalidArgumentError(f"{field_name} must decode to a valid image: {exc}") from exc


def _decode_image_b64(value: object, *, field_name: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise InvalidArgumentError(f"{field_name} must be a non-empty base64 string")

    payload = value.strip()
    if payload.startswith("data:"):
        header, separator, encoded = payload.partition(",")
        if separator != "," or ";base64" not in header:
            raise InvalidArgumentError(f"{field_name} must use a valid data URL when prefixed with data:")
        payload = encoded

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise InvalidArgumentError(f"{field_name} must be valid base64-encoded image data: {exc}") from exc
    return _load_pil_image(raw, field_name=field_name)


def _image_metadata(image: Image.Image) -> dict[str, object]:
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "channels": len(image.getbands()),
    }


def _normalize_image_payload(body: dict[str, object]) -> ImageRequest:
    payload = dict(body)
    cache = payload.pop("cache", None)
    if cache is not None and not isinstance(cache, dict):
        raise InvalidArgumentError("cache must be a JSON object when provided")

    if isinstance(cache, dict):
        mapping = {
            "mode": "cache_mode",
            "reuse_threshold": "cache_reuse_threshold",
            "start_percent": "cache_start_percent",
            "end_percent": "cache_end_percent",
            "error_decay_rate": "cache_error_decay_rate",
            "use_relative_threshold": "cache_use_relative_threshold",
            "reset_error_on_compute": "cache_reset_error_on_compute",
            "Fn_compute_blocks": "cache_Fn_compute_blocks",
            "Bn_compute_blocks": "cache_Bn_compute_blocks",
            "residual_diff_threshold": "cache_residual_diff_threshold",
            "max_accumulated_residual_diff": "cache_max_accumulated_residual_diff",
            "max_warmup_steps": "cache_max_warmup_steps",
            "max_cached_steps": "cache_max_cached_steps",
            "max_continuous_cached_steps": "cache_max_continuous_cached_steps",
            "taylorseer_n_derivatives": "cache_taylorseer_n_derivatives",
            "taylorseer_skip_interval": "cache_taylorseer_skip_interval",
            "scm_mask": "cache_scm_mask",
            "scm_policy_dynamic": "cache_scm_policy_dynamic",
        }
        for source, target in mapping.items():
            if source in cache and target not in payload:
                payload[target] = cache[source]

    for source, target in (
        ("init_image_b64", "init_image"),
        ("mask_image_b64", "mask_image"),
        ("control_image_b64", "control_image"),
    ):
        if source in payload:
            payload[target] = _decode_image_b64(payload.pop(source), field_name=source)

    if "ref_images_b64" in payload:
        ref_images_b64 = payload.pop("ref_images_b64")
        if not isinstance(ref_images_b64, list):
            raise InvalidArgumentError("ref_images_b64 must be a JSON array of base64 image strings")
        payload["ref_images"] = [
            _decode_image_b64(item, field_name=f"ref_images_b64[{index}]")
            for index, item in enumerate(ref_images_b64)
        ]

    payload["output_type"] = "pil"
    try:
        return ImageRequest.from_kwargs(**payload)
    except TypeError as exc:
        raise InvalidArgumentError(str(exc)) from exc


def _normalize_video_payload(body: dict[str, object]) -> VideoRequest:
    payload = dict(body)
    cache = payload.pop("cache", None)
    if cache is not None and not isinstance(cache, dict):
        raise InvalidArgumentError("cache must be a JSON object when provided")
    if isinstance(cache, dict):
        for source, target in {
            "mode": "cache_mode", "reuse_threshold": "cache_reuse_threshold",
            "start_percent": "cache_start_percent", "end_percent": "cache_end_percent",
            "error_decay_rate": "cache_error_decay_rate",
            "use_relative_threshold": "cache_use_relative_threshold",
            "reset_error_on_compute": "cache_reset_error_on_compute",
            "Fn_compute_blocks": "cache_Fn_compute_blocks",
            "Bn_compute_blocks": "cache_Bn_compute_blocks",
            "residual_diff_threshold": "cache_residual_diff_threshold",
            "max_accumulated_residual_diff": "cache_max_accumulated_residual_diff",
            "max_warmup_steps": "cache_max_warmup_steps",
            "max_cached_steps": "cache_max_cached_steps",
            "max_continuous_cached_steps": "cache_max_continuous_cached_steps",
            "taylorseer_n_derivatives": "cache_taylorseer_n_derivatives",
            "taylorseer_skip_interval": "cache_taylorseer_skip_interval",
            "scm_mask": "cache_scm_mask", "scm_policy_dynamic": "cache_scm_policy_dynamic",
        }.items():
            if source in cache and target not in payload:
                payload[target] = cache[source]
    for source, target in (("init_image_b64", "init_image"), ("end_image_b64", "end_image")):
        if source in payload:
            payload[target] = _decode_image_b64(payload.pop(source), field_name=source)
    for source, target in (("ref_images_b64", "ref_images"), ("control_frames_b64", "control_frames")):
        if source in payload:
            values = payload.pop(source)
            if not isinstance(values, list):
                raise InvalidArgumentError(f"{source} must be a JSON array")
            payload[target] = [
                _decode_image_b64(value, field_name=f"{source}[{index}]")
                for index, value in enumerate(values)
            ]
    if "ref_audios" in payload:
        values = payload["ref_audios"]
        if not isinstance(values, list):
            raise InvalidArgumentError("ref_audios must be a JSON array")
        payload["ref_audios"] = [AudioInput(**value) if isinstance(value, dict) else value for value in values]
    if "ref_videos" in payload:
        values = payload["ref_videos"]
        if not isinstance(values, list):
            raise InvalidArgumentError("ref_videos must be a JSON array")
        converted = []
        for index, value in enumerate(values):
            if not isinstance(value, dict) or not isinstance(value.get("frames_b64"), list):
                raise InvalidArgumentError(f"ref_videos[{index}] needs a frames_b64 array")
            audio = value.get("audio")
            converted.append(RefVideoInput(
                frames=[_decode_image_b64(frame, field_name=f"ref_videos[{index}].frames_b64[{j}]") for j, frame in enumerate(value["frames_b64"])],
                fps=int(value.get("fps", 24)),
                audio=AudioInput(**audio) if isinstance(audio, dict) else None,
            ))
        payload["ref_videos"] = converted
    payload["output_type"] = "pil"
    try:
        return VideoRequest.from_kwargs(**payload)
    except TypeError as exc:
        raise InvalidArgumentError(str(exc)) from exc


def _request_parameters(request: ImageRequest | VideoRequest) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for entry in fields(request):
        if entry.name == "output_type":
            continue
        value = getattr(request, entry.name)
        if value is None:
            continue
        if entry.name in {"init_image", "end_image", "mask_image", "control_image"}:
            parameters[entry.name] = _image_metadata(value)
            continue
        if entry.name == "ref_images":
            parameters[entry.name] = [_image_metadata(image) for image in value]
            continue
        if entry.name == "control_frames":
            parameters[entry.name] = [_image_metadata(image) for image in value]
            continue
        if entry.name == "ref_audios":
            parameters[entry.name] = [{"sample_rate": a.sample_rate, "channels": a.channels, "sample_count": len(a.samples) // a.channels} for a in value]
            continue
        if entry.name == "ref_videos":
            parameters[entry.name] = [{"frame_count": len(v.frames), "fps": v.fps, "has_audio": v.audio is not None} for v in value]
            continue
        parameters[entry.name] = value
    return parameters


def _requested_sampling_steps(request: ImageRequest | VideoRequest) -> int:
    steps = int(request.steps or 0)
    if steps <= 0:
        return 0
    if isinstance(request, ImageRequest):
        batch_count = int(request.batch_count or 1)
        return steps * max(1, batch_count)
    return steps


@dataclass(slots=True)
class GenerationJob:
    job_id: str
    kind: str
    request: ImageRequest | VideoRequest
    created_ms: int = field(default_factory=_now_ms)
    started_ms: int | None = None
    finished_ms: int | None = None
    status: str = "queued"
    cancel_requested: bool = False
    error: str | None = None
    progress_current_step: int = 0
    progress_total_steps: int = 0
    result: dict[str, object] | None = None


class GenerationJobService:
    def __init__(
        self,
        engine: object,
        *,
        model_name: str | None = None,
        job_ttl_seconds: float | None = 3600.0,
    ) -> None:
        self._engine = engine
        self._model_name = model_name or getattr(engine, "pipeline_name", None) or "edge-dit-model"
        self._job_ttl_ms = None if job_ttl_seconds is None else max(0, int(job_ttl_seconds * 1000))
        self._jobs: dict[str, GenerationJob] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._active_job_id: str | None = None
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name="edge-dit-server", daemon=True)
        self._worker.start()

    @property
    def engine(self) -> object:
        return self._engine

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            should_cancel = self._active_job_id is not None
        if should_cancel:
            try:
                self._engine.request_cancel()
            except Exception:
                pass
        self._queue.put(None)
        self._worker.join(timeout=1.0)
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()

    def capabilities(self) -> dict[str, object]:
        default_scheduler = None
        try:
            default_scheduler = self._engine.default_scheduler()
        except TypeError:
            default_scheduler = self._engine.default_scheduler(None)
        except Exception:
            default_scheduler = None

        return {
            "service": "edge-dit-python-server",
            "package_version": __version__,
            "model": self._model_name,
            "pipeline_name": getattr(self._engine, "pipeline_name", None),
            "version_name": getattr(self._engine, "version_name", None),
            "supports": {
                "image": bool(getattr(self._engine, "supports_image", False)),
                "video": bool(getattr(self._engine, "supports_video", False)),
            },
            "defaults": {
                "sampler": getattr(self._engine, "default_sampler", None),
                "scheduler": default_scheduler,
            },
            "endpoints": [
                "/ed/v2/health",
                "/ed/v2/capabilities",
                "/ed/v2/images/generations",
                "/ed/v2/videos/generations",
                "/ed/v2/jobs",
                "/ed/v2/jobs/cleanup",
                "/ed/v2/jobs/{job_id}",
                "/ed/v2/jobs/{job_id}/cancel",
                "/ed/v2/jobs/{job_id}/result",
                "/ed/v2/jobs/{job_id}/video",
            ],
            "aliases": ["/edgedit/v2", "/edge-dit/v2"],
            "semantics": {
                "progress": "sampling_step_only",
                "cancellation": "cooperative_step_boundary",
                "results": "stored_in_memory",
                "job_ttl_ms": self._job_ttl_ms,
            },
        }

    def create_image_job(self, body: dict[str, object]) -> GenerationJob:
        if not bool(getattr(self._engine, "supports_image", False)):
            raise UnsupportedError("the loaded model does not support image generation")

        request = _normalize_image_payload(body)
        return self._create_job("image", request)

    def create_video_job(self, body: dict[str, object]) -> GenerationJob:
        if not bool(getattr(self._engine, "supports_video", False)):
            raise UnsupportedError("the loaded model does not support video generation")

        request = _normalize_video_payload(body)
        return self._create_job("video", request)

    def _create_job(self, kind: str, request: ImageRequest | VideoRequest) -> GenerationJob:
        self.cleanup_expired()
        job = GenerationJob(job_id=str(uuid.uuid4()), kind=kind, request=request)
        with self._lock:
            if self._closed:
                raise RuntimeError("server is shutting down")
            self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        return self._snapshot(job)

    def get_job(self, job_id: str) -> GenerationJob:
        self.cleanup_expired()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            active = self._active_job_id == job_id and job.status in {"running", "cancelling"}

        if active:
            try:
                native_current, native_total = self._engine.progress_steps()
            except Exception:
                native_current, native_total = 0, 0
        else:
            native_current, native_total = 0, 0

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            requested_total = _requested_sampling_steps(job.request)
            still_active = self._active_job_id == job_id and job.status in {"running", "cancelling"}
            if active and still_active:
                job.progress_total_steps = max(
                    job.progress_total_steps,
                    requested_total,
                    int(native_total),
                )
                job.progress_current_step = max(
                    job.progress_current_step,
                    int(native_current),
                )

            total_steps = max(job.progress_total_steps, requested_total)
            current_step = job.progress_current_step
            if job.status == "succeeded":
                current_step = total_steps
                job.progress_current_step = current_step
                job.progress_total_steps = total_steps
            elif job.status in {"failed", "cancelled"}:
                current_step = 0
            if total_steps > 0:
                current_step = min(current_step, total_steps)
            snapshot = self._snapshot(job)

        snapshot.result = {
            "progress": {
                "current_step": int(current_step),
                "total_steps": int(total_steps),
            }
        }
        return snapshot

    def list_jobs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[GenerationJob]:
        self.cleanup_expired()
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_ms, reverse=True)
            if status:
                jobs = [job for job in jobs if job.status == status]
            if kind:
                jobs = [job for job in jobs if job.kind == kind]
            jobs = jobs[: max(0, limit)]
            return [self._snapshot(job) for job in jobs]

    def get_result(self, job_id: str) -> dict[str, object]:
        self.cleanup_expired()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.result is None:
                raise ValueError(job.status)
            return json.loads(json.dumps(job.result))

    def encode_video(self, job_id: str, *, fps: int = 24) -> bytes:
        result = self.get_result(job_id)
        if result.get("object") != "edge_dit.video_generation":
            raise UnsupportedError("the selected job is not a video generation")
        return _encode_video_result_mp4(result, fps=fps)

    def request_cancel(self, job_id: str) -> GenerationJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)

            if job.status in {"succeeded", "failed", "cancelled"}:
                return self._snapshot(job)

            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_ms = _now_ms()
                job.error = "generation cancelled before start"
                return self._snapshot(job)

            if job.status == "running":
                job.status = "cancelling"

        self._engine.request_cancel()
        return self.get_job(job_id)

    def remove_job(self, job_id: str) -> GenerationJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"running", "cancelling"}:
                raise ValueError(job.status)
            snapshot = self._snapshot(job)
            del self._jobs[job_id]
            return snapshot

    def cleanup_expired(self, *, now_ms: int | None = None) -> list[str]:
        if self._job_ttl_ms is None:
            return []
        now = _now_ms() if now_ms is None else now_ms
        removed: list[str] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.finished_ms is None:
                    continue
                if now - job.finished_ms >= self._job_ttl_ms:
                    removed.append(job_id)
                    del self._jobs[job_id]
        return removed

    def job_response(self, job_id: str, *, base_prefix: str) -> dict[str, object]:
        job = self.get_job(job_id)
        progress = job.result["progress"] if job.result is not None else {"current_step": 0, "total_steps": 0}
        expires_ms = None
        if job.finished_ms is not None and self._job_ttl_ms is not None:
            expires_ms = job.finished_ms + self._job_ttl_ms
        return {
            "object": "edge_dit.job",
            "id": job.job_id,
            "kind": job.kind,
            "model": self._model_name,
            "status": job.status,
            "created_ms": job.created_ms,
            "started_ms": job.started_ms,
            "finished_ms": job.finished_ms,
            "expires_ms": expires_ms,
            "cancel_requested": job.cancel_requested,
            "progress": progress,
            "parameters": _request_parameters(job.request),
            "error": job.error,
            "status_url": f"{base_prefix}/jobs/{job.job_id}",
            "cancel_url": f"{base_prefix}/jobs/{job.job_id}/cancel",
            "result_url": f"{base_prefix}/jobs/{job.job_id}/result",
        }

    def job_summary_response(self, job: GenerationJob, *, base_prefix: str) -> dict[str, object]:
        expires_ms = None
        if job.finished_ms is not None and self._job_ttl_ms is not None:
            expires_ms = job.finished_ms + self._job_ttl_ms
        return {
            "object": "edge_dit.job",
            "id": job.job_id,
            "kind": job.kind,
            "model": self._model_name,
            "status": job.status,
            "created_ms": job.created_ms,
            "started_ms": job.started_ms,
            "finished_ms": job.finished_ms,
            "expires_ms": expires_ms,
            "cancel_requested": job.cancel_requested,
            "parameters": _request_parameters(job.request),
            "error": job.error,
            "status_url": f"{base_prefix}/jobs/{job.job_id}",
            "cancel_url": f"{base_prefix}/jobs/{job.job_id}/cancel",
            "result_url": f"{base_prefix}/jobs/{job.job_id}/result",
        }

    def _snapshot(self, job: GenerationJob) -> GenerationJob:
        return GenerationJob(
            job_id=job.job_id,
            kind=job.kind,
            request=job.request,
            created_ms=job.created_ms,
            started_ms=job.started_ms,
            finished_ms=job.finished_ms,
            status=job.status,
            cancel_requested=job.cancel_requested,
            error=job.error,
            progress_current_step=job.progress_current_step,
            progress_total_steps=job.progress_total_steps,
            result=json.loads(json.dumps(job.result)) if job.result is not None else None,
        )

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                return

            try:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None or job.status == "cancelled":
                        continue
                    job.status = "running"
                    job.started_ms = _now_ms()
                    self._active_job_id = job_id

                try:
                    if job.kind == "image":
                        outputs = self._engine.generate_image(job.request)
                    elif job.kind == "video":
                        outputs = self._engine.generate_video(job.request)
                    else:
                        raise RuntimeError(f"unsupported job kind: {job.kind}")
                    result = self._build_result(job, outputs)
                    with self._lock:
                        job.status = "succeeded"
                        job.finished_ms = _now_ms()
                        job.result = result
                        job.error = None
                except GenerationCancelledError as exc:
                    with self._lock:
                        job.status = "cancelled"
                        job.finished_ms = _now_ms()
                        job.error = str(exc)
                except EdgeDitError as exc:
                    with self._lock:
                        job.status = "failed"
                        job.finished_ms = _now_ms()
                        job.error = str(exc)
                except Exception as exc:
                    with self._lock:
                        job.status = "failed"
                        job.finished_ms = _now_ms()
                        job.error = f"unexpected server error: {exc}"
                finally:
                    with self._lock:
                        self._active_job_id = None
            finally:
                self._queue.task_done()

    def _build_result(self, job: GenerationJob, outputs: list[object]) -> dict[str, object]:
        if job.kind == "image":
            return self._build_image_result(job, outputs)
        if job.kind == "video":
            return self._build_video_result(job, outputs)
        raise RuntimeError(f"unsupported job kind: {job.kind}")

    def _build_image_result(self, job: GenerationJob, images: list[object]) -> dict[str, object]:
        encoded = []
        for index, image in enumerate(images):
            if not isinstance(image, Image.Image):
                raise TypeError("server currently expects PIL image outputs")
            encoded.append(
                {
                    "b64_png": _base64_png(image),
                    "metadata": {
                        "index": index,
                        "width": image.width,
                        "height": image.height,
                        "channels": len(image.getbands()),
                        "format": "png",
                    },
                }
            )

        return {
            "object": "edge_dit.image_generation",
            "id": job.job_id,
            "model": self._model_name,
            "created_ms": job.created_ms,
            "completed_ms": _now_ms(),
            "parameters": _request_parameters(job.request),
            "data": encoded,
        }

    def _build_video_result(self, job: GenerationJob, frames: list[object]) -> dict[str, object]:
        encoded = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                raise TypeError("server currently expects PIL frame outputs")
            encoded.append(
                {
                    "b64_png": _base64_png(frame),
                    "metadata": {
                        "index": index,
                        "width": frame.width,
                        "height": frame.height,
                        "channels": len(frame.getbands()),
                        "format": "png",
                    },
                }
            )

        result = {
            "object": "edge_dit.video_generation",
            "id": job.job_id,
            "model": self._model_name,
            "created_ms": job.created_ms,
            "completed_ms": _now_ms(),
            "parameters": _request_parameters(job.request),
            "frame_format": "png",
            "frames": encoded,
        }
        audio = getattr(frames, "audio", None)
        if audio:
            raw = struct.pack(f"<{len(audio)}f", *audio)
            result["audio"] = {
                "b64_f32le": base64.b64encode(raw).decode("ascii"),
                "sample_rate": getattr(frames, "audio_sample_rate", 0),
                "channels": getattr(frames, "audio_channels", 0),
                "sample_count": len(audio) // max(1, getattr(frames, "audio_channels", 1)),
            }
        return result


ImageJobService = GenerationJobService


class EdgeDitServerHandler(BaseHTTPRequestHandler):
    service: ImageJobService
    _request_id: str

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
        self.send_response(int(HTTPStatus.NO_CONTENT))
        self._write_common_headers(0)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _dispatch(self, method: str) -> None:
        self._request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._write_json(
                HTTPStatus.OK,
                {
                    "service": "edge-dit-python-server",
                    "message": "edge-dit server is running",
                    "health": "/ed/v2/health",
                    "capabilities": "/ed/v2/capabilities",
                    "image_generation": "/ed/v2/images/generations",
                    "video_generation": "/ed/v2/videos/generations",
                    "jobs": "/ed/v2/jobs",
                    "aliases": ["/edgedit/v2", "/edge-dit/v2"],
                },
            )
            return

        for prefix in _API_PREFIXES:
            if path == prefix + "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "edge-dit-python-server",
                        "model": self.service.capabilities()["model"],
                    },
                )
                return
            if path == prefix + "/capabilities":
                self._write_json(HTTPStatus.OK, self.service.capabilities())
                return
            if path == prefix + "/images/generations" and method == "POST":
                self._handle_create_job(prefix, "image")
                return
            if path == prefix + "/videos/generations" and method == "POST":
                self._handle_create_job(prefix, "video")
                return
            if path == prefix + "/jobs" and method == "GET":
                self._handle_list_jobs(prefix, parsed.query)
                return
            if path == prefix + "/jobs/cleanup" and method == "POST":
                self._handle_cleanup_jobs()
                return
            if path.startswith(prefix + "/jobs/"):
                self._handle_job_route(method, prefix, path[len(prefix) :], parsed.query)
                return

        self._write_error(HTTPStatus.NOT_FOUND, "unknown endpoint", code="not_found")

    def _handle_create_job(self, prefix: str, kind: str) -> None:
        body = self._read_json_body()
        if body is None:
            return
        try:
            if kind == "image":
                job = self.service.create_image_job(body)
            elif kind == "video":
                job = self.service.create_video_job(body)
            else:
                raise InvalidArgumentError(f"unsupported job kind: {kind}")
            response = self.service.job_response(job.job_id, base_prefix=prefix)
        except InvalidArgumentError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc), code="invalid_request")
            return
        except UnsupportedError as exc:
            self._write_error(HTTPStatus.CONFLICT, str(exc), code="unsupported")
            return
        except EdgeDitError as exc:
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), code="edge_dit_error")
            return

        self._write_json(HTTPStatus.ACCEPTED, response)

    def _handle_list_jobs(self, prefix: str, query: str) -> None:
        params = parse_qs(query, keep_blank_values=False)
        status = params.get("status", [None])[0]
        kind = params.get("kind", [None])[0]
        try:
            limit = int(params.get("limit", ["100"])[0])
        except ValueError:
            self._write_error(HTTPStatus.BAD_REQUEST, "limit must be an integer", code="invalid_request")
            return
        if status and status not in {"queued", "running", "cancelling", "succeeded", "failed", "cancelled"}:
            self._write_error(HTTPStatus.BAD_REQUEST, "unsupported status filter", code="invalid_request")
            return
        if kind and kind not in {"image", "video"}:
            self._write_error(HTTPStatus.BAD_REQUEST, "unsupported kind filter", code="invalid_request")
            return

        jobs = self.service.list_jobs(status=status, kind=kind, limit=limit)
        self._write_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [self.service.job_summary_response(job, base_prefix=prefix) for job in jobs],
                "has_more": False,
            },
        )

    def _handle_cleanup_jobs(self) -> None:
        body = self._read_optional_json_body()
        if body is None:
            return
        now_ms = body.get("now_ms")
        if now_ms is not None and not isinstance(now_ms, int):
            self._write_error(HTTPStatus.BAD_REQUEST, "now_ms must be an integer", code="invalid_request")
            return
        removed = self.service.cleanup_expired(now_ms=now_ms)
        self._write_json(
            HTTPStatus.OK,
            {
                "object": "edge_dit.job_cleanup",
                "removed_count": len(removed),
                "removed_ids": removed,
            },
        )

    def _handle_job_route(self, method: str, prefix: str, suffix: str, query: str = "") -> None:
        parts = [part for part in suffix.split("/") if part]
        if len(parts) < 2 or parts[0] != "jobs":
            self._write_error(HTTPStatus.NOT_FOUND, "unknown endpoint", code="not_found")
            return

        job_id = parts[1]
        action = parts[2] if len(parts) > 2 else None

        try:
            if action is None and method == "GET":
                self._write_json(HTTPStatus.OK, self.service.job_response(job_id, base_prefix=prefix))
                return
            if action == "cancel" and method == "POST":
                job = self.service.request_cancel(job_id)
                status = HTTPStatus.OK if job.status == "cancelled" else HTTPStatus.ACCEPTED
                self._write_json(status, self.service.job_response(job_id, base_prefix=prefix))
                return
            if action == "result" and method == "GET":
                result = self.service.get_result(job_id)
                self._write_json(HTTPStatus.OK, result)
                return
            if action == "video" and method == "GET":
                params = parse_qs(query, keep_blank_values=False)
                try:
                    fps = int(params.get("fps", ["24"])[0])
                except ValueError:
                    self._write_error(HTTPStatus.BAD_REQUEST, "fps must be an integer", code="invalid_request")
                    return
                if fps < 1 or fps > 120:
                    self._write_error(
                        HTTPStatus.BAD_REQUEST,
                        "fps must be between 1 and 120",
                        code="invalid_request",
                    )
                    return
                payload = self.service.encode_video(job_id, fps=fps)
                self._write_bytes(
                    HTTPStatus.OK,
                    payload,
                    content_type="video/mp4",
                    content_disposition=f'attachment; filename="{job_id}.mp4"',
                )
                return
            if action is None and method == "DELETE":
                job = self.service.remove_job(job_id)
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "object": "edge_dit.job_deleted",
                        "id": job.job_id,
                        "kind": job.kind,
                        "status": job.status,
                    },
                )
                return
        except KeyError:
            self._write_error(HTTPStatus.NOT_FOUND, f"unknown job id: {job_id}", code="job_not_found")
            return
        except ValueError as exc:
            if method == "DELETE":
                self._write_error(
                    HTTPStatus.CONFLICT,
                    f"job cannot be deleted while status is {exc}",
                    code="job_active",
                )
            else:
                self._write_error(
                    HTTPStatus.CONFLICT,
                    f"job is not ready; current status is {exc}",
                    code="job_not_ready",
                )
            return
        except UnsupportedError as exc:
            self._write_error(HTTPStatus.CONFLICT, str(exc), code="unsupported")
            return
        except FileNotFoundError as exc:
            self._write_error(HTTPStatus.NOT_IMPLEMENTED, str(exc), code="ffmpeg_unavailable")
            return
        except RuntimeError as exc:
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), code="video_encode_failed")
            return

        self._write_error(HTTPStatus.METHOD_NOT_ALLOWED, "unsupported method for endpoint", code="method_not_allowed")

    def _read_json_body(self) -> dict[str, object] | None:
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError:
            self._write_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length header", code="invalid_request")
            return None

        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            self._write_error(HTTPStatus.BAD_REQUEST, "empty request body", code="invalid_request")
            return None

        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}", code="invalid_json")
            return None

        if not isinstance(body, dict):
            self._write_error(HTTPStatus.BAD_REQUEST, "request body must be a JSON object", code="invalid_request")
            return None
        return body

    def _read_optional_json_body(self) -> dict[str, object] | None:
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError:
            self._write_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length header", code="invalid_request")
            return None

        if length <= 0:
            return {}

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}", code="invalid_json")
            return None

        if not isinstance(body, dict):
            self._write_error(HTTPStatus.BAD_REQUEST, "request body must be a JSON object", code="invalid_request")
            return None
        return body

    def _write_error(
        self,
        status: HTTPStatus,
        message: str,
        *,
        code: str,
        error_type: str = "invalid_request_error",
    ) -> None:
        self._write_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                    "status": int(status),
                    "request_id": self._request_id,
                }
            },
        )

    def _write_json(self, status: HTTPStatus, body: dict[str, object]) -> None:
        if "request_id" not in body:
            body["request_id"] = self._request_id
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self._write_common_headers(len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def _write_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        *,
        content_type: str,
        content_disposition: str | None = None,
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Request-ID", self._request_id)
        self.send_header("Access-Control-Allow-Origin", "*")
        if content_disposition:
            self.send_header("Content-Disposition", content_disposition)
        self.end_headers()
        self.wfile.write(payload)

    def _write_common_headers(self, content_length: int) -> None:
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(content_length))
        self.send_header("X-Request-ID", self._request_id)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")


class EdgeDitHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_http_server(address: tuple[str, int], service: ImageJobService) -> EdgeDitHTTPServer:
    handler = type("BoundEdgeDitServerHandler", (EdgeDitServerHandler,), {"service": service})
    return EdgeDitHTTPServer(address, handler)


def serve(
    engine_config: EngineConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    job_ttl_seconds: float | None = 3600.0,
    engine_factory: Callable[[EngineConfig], object] = Engine,
) -> None:
    engine = engine_factory(engine_config)
    service = ImageJobService(
        engine,
        model_name=_display_model_name(engine_config),
        job_ttl_seconds=job_ttl_seconds,
    )
    server = create_http_server((host, port), service)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the edge-dit Python server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model")
    parser.add_argument("--diffusion-model", dest="diffusion_model_path")
    parser.add_argument("--vae", dest="vae_path")
    parser.add_argument("--audio-vae", dest="audio_vae_path")
    parser.add_argument("--embeddings-connectors", dest="embeddings_connectors_path")
    parser.add_argument("--latent-upscaler", dest="latent_upscaler_path")
    parser.add_argument("--clip_l", dest="clip_l_path")
    parser.add_argument("--clip_g", dest="clip_g_path")
    parser.add_argument("--t5xxl", dest="t5xxl_path")
    parser.add_argument("--llm", dest="llm_path")
    parser.add_argument("--llm-vision", dest="llm_vision_path")
    parser.add_argument("--backend")
    parser.add_argument("--threads", type=int, dest="n_threads")
    parser.add_argument("--type", "--weight-type", dest="weight_type")
    parser.add_argument("--tensor-type-rules")
    parser.add_argument("--max-vram", type=float, dest="max_vram_gb")
    parser.add_argument("--offload-to-cpu", action="store_true", dest="offload_params_to_cpu")
    parser.add_argument("--dit-offload", action="store_true")
    parser.add_argument(
        "--text-encoder-offload",
        "--keep-text-encoder-on-cpu",
        action="store_true",
        dest="text_encoder_offload",
    )
    parser.add_argument(
        "--vae-offload",
        "--keep-vae-on-cpu",
        action="store_true",
        dest="vae_offload",
    )
    parser.add_argument("--auto-allocate", action="store_true", default=True, dest="auto_allocate")
    parser.add_argument("--no-auto-allocate", action="store_false", dest="auto_allocate")
    parser.add_argument("--auto-fit", action="store_true")
    parser.add_argument("--fit-width", type=int)
    parser.add_argument("--fit-height", type=int)
    parser.add_argument("--fit-frames", type=int)
    parser.add_argument("--fit-fps", type=int)
    parser.add_argument("--minimax-h3-stage-lifecycle", action="store_true")
    parser.add_argument("--skip-t5", "--no-t5", action="store_true", dest="skip_t5")
    parser.add_argument("--flash-attention", action="store_true", default=None, dest="flash_attention")
    parser.add_argument("--no-flash-attention", action="store_false", dest="flash_attention")
    parser.add_argument("--vae-tiling", action="store_true", default=None, dest="vae_tiling")
    parser.add_argument("--no-vae-tiling", action="store_false", dest="vae_tiling")
    parser.add_argument(
        "--job-ttl-seconds",
        type=float,
        default=3600.0,
        help="Seconds to retain terminal job metadata/results in memory. Use a negative value to disable TTL cleanup.",
    )
    args = parser.parse_args(argv)

    try:
        config = EngineConfig(
            model_path=args.model,
            diffusion_model_path=args.diffusion_model_path,
            vae_path=args.vae_path,
            audio_vae_path=args.audio_vae_path,
            embeddings_connectors_path=args.embeddings_connectors_path,
            latent_upscaler_path=args.latent_upscaler_path,
            clip_l_path=args.clip_l_path,
            clip_g_path=args.clip_g_path,
            t5xxl_path=args.t5xxl_path,
            llm_path=args.llm_path,
            llm_vision_path=args.llm_vision_path,
            backend=args.backend,
            n_threads=args.n_threads,
            weight_type=args.weight_type,
            tensor_type_rules=args.tensor_type_rules,
            max_vram_gb=args.max_vram_gb,
            offload_params_to_cpu=args.offload_params_to_cpu or None,
            dit_offload=args.dit_offload or None,
            text_encoder_offload=args.text_encoder_offload or None,
            vae_offload=args.vae_offload or None,
            auto_allocate=args.auto_allocate,
            auto_fit=args.auto_fit or None,
            fit_width=args.fit_width,
            fit_height=args.fit_height,
            fit_frames=args.fit_frames,
            fit_fps=args.fit_fps,
            minimax_h3_stage_lifecycle=args.minimax_h3_stage_lifecycle or None,
            skip_t5=args.skip_t5 or None,
            flash_attention=args.flash_attention,
            vae_tiling=args.vae_tiling,
        )
        job_ttl_seconds = None if args.job_ttl_seconds < 0 else args.job_ttl_seconds
        serve(config, host=args.host, port=args.port, job_ttl_seconds=job_ttl_seconds)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
