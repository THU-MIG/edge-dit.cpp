from __future__ import annotations

import base64
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image

from edge_dit import Engine
from edge_dit.config import EngineConfig
from edge_dit.server_v2 import ImageJobService, create_http_server

_REQUEST_ID = "optional-real-server-v2-smoke"
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True, slots=True)
class ScenarioRequest:
    payload: dict[str, object]
    expected_width: int
    expected_height: int
    expected_frames: int | None = None
    input_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServerV2MatrixScenario:
    name: str
    slug: str
    prefix: str
    kind: str
    model_env: str
    model_default: str
    timeout_env: str
    default_timeout_seconds: float
    engine_overrides: dict[str, object]
    request_builder: Callable[[], ScenarioRequest]


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    *,
    request_id: str = _REQUEST_ID,
) -> tuple[int, dict[str, object]]:
    data = None
    headers = {"X-Request-ID": request_id}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    try:
        with _LOCAL_OPENER.open(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _wait_for_terminal_job(base_url: str, status_url: str, timeout_seconds: float) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_body: dict[str, object] | None = None
    while time.time() < deadline:
        status_code, body = _request_json(base_url, "GET", status_url)
        if status_code != 200:
            raise AssertionError(f"unexpected HTTP {status_code} while polling job: {body}")
        last_body = body
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(1.0)
    raise AssertionError(f"timed out waiting for terminal job state; last body was {last_body}")


def _wait_for_job_status(
    base_url: str,
    status_url: str,
    *,
    expected_statuses: set[str],
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_body: dict[str, object] | None = None
    while time.time() < deadline:
        status_code, body = _request_json(base_url, "GET", status_url)
        if status_code != 200:
            raise AssertionError(f"unexpected HTTP {status_code} while polling job: {body}")
        last_body = body
        if body.get("status") in expected_statuses:
            return body
        time.sleep(0.25)
    raise AssertionError(
        f"timed out waiting for job state in {sorted(expected_statuses)}; last body was {last_body}"
    )


def _wait_for_sampling_progress(
    base_url: str,
    status_url: str,
    *,
    expected_total_steps: int,
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_body: dict[str, object] | None = None
    while time.time() < deadline:
        status_code, body = _request_json(base_url, "GET", status_url)
        if status_code != 200:
            raise AssertionError(f"unexpected HTTP {status_code} while polling job: {body}")
        last_body = body
        progress = body.get("progress") if isinstance(body.get("progress"), dict) else {}
        current_step = int(progress.get("current_step", 0) or 0)
        total_steps = int(progress.get("total_steps", 0) or 0)
        if total_steps == expected_total_steps and current_step > 0:
            return body
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.1)
    raise AssertionError(
        "did not observe non-zero sampling progress before the job reached a terminal state; "
        f"expected total_steps={expected_total_steps}, last body was {last_body}"
    )


def _encode_png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_png(payload: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(payload))).copy()


def _build_qwen_edit_input_image(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size))
    for y in range(size):
        for x in range(size):
            image.putpixel(
                (x, y),
                (
                    (x * 255) // max(1, size - 1),
                    (y * 255) // max(1, size - 1),
                    ((x + y) * 255) // max(1, (size * 2) - 2),
                ),
            )
    return image


def _build_flux_kontext_ref_image(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size))
    center = size / 2.0
    radius = size / 3.2
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            distance = (dx * dx + dy * dy) ** 0.5
            ring = max(0.0, 1.0 - abs(distance - radius) / max(1.0, radius * 0.6))
            image.putpixel(
                (x, y),
                (
                    min(255, int(40 + (215 * x) / max(1, size - 1))),
                    min(255, int(30 + (180 * y) / max(1, size - 1))),
                    min(255, int(60 + ring * 180)),
                ),
            )
    return image


def _build_flux_request() -> ScenarioRequest:
    width = int(os.environ.get("EDGE_DIT_FLUX_WIDTH", "256"))
    height = int(os.environ.get("EDGE_DIT_FLUX_HEIGHT", "256"))
    return ScenarioRequest(
        payload={
            "prompt": os.environ.get("EDGE_DIT_FLUX_PROMPT", "server v2 matrix smoke teapot"),
            "width": width,
            "height": height,
            "steps": int(os.environ.get("EDGE_DIT_FLUX_STEPS", "1")),
            "seed": int(os.environ.get("EDGE_DIT_FLUX_SEED", "42")),
        },
        expected_width=width,
        expected_height=height,
    )


def _build_sd3_request() -> ScenarioRequest:
    width = int(os.environ.get("EDGE_DIT_SD3_WIDTH", "256"))
    height = int(os.environ.get("EDGE_DIT_SD3_HEIGHT", "256"))
    return ScenarioRequest(
        payload={
            "prompt": os.environ.get("EDGE_DIT_SD3_PROMPT", "a studio photo of a glass teapot"),
            "width": width,
            "height": height,
            "steps": int(os.environ.get("EDGE_DIT_SD3_STEPS", "1")),
            "seed": int(os.environ.get("EDGE_DIT_SD3_SEED", "42")),
        },
        expected_width=width,
        expected_height=height,
    )


def _build_qwen_image_request() -> ScenarioRequest:
    width = int(os.environ.get("EDGE_DIT_QWEN_IMAGE_WIDTH", "512"))
    height = int(os.environ.get("EDGE_DIT_QWEN_IMAGE_HEIGHT", "512"))
    return ScenarioRequest(
        payload={
            "prompt": os.environ.get("EDGE_DIT_QWEN_IMAGE_PROMPT", "a polished product photo of a glass teapot"),
            "width": width,
            "height": height,
            "steps": int(os.environ.get("EDGE_DIT_QWEN_IMAGE_STEPS", "1")),
            "seed": int(os.environ.get("EDGE_DIT_QWEN_IMAGE_SEED", "42")),
        },
        expected_width=width,
        expected_height=height,
    )


def _build_qwen_image_edit_request() -> ScenarioRequest:
    size = int(os.environ.get("EDGE_DIT_QWEN_IMAGE_EDIT_SIZE", "256"))
    if size % 32 != 0:
        raise AssertionError("EDGE_DIT_QWEN_IMAGE_EDIT_SIZE must be divisible by 32")
    return ScenarioRequest(
        payload={
            "prompt": os.environ.get(
                "EDGE_DIT_QWEN_IMAGE_EDIT_PROMPT",
                "turn this synthetic gradient into a polished product render",
            ),
            "width": size,
            "height": size,
            "steps": int(os.environ.get("EDGE_DIT_QWEN_IMAGE_EDIT_STEPS", "1")),
            "seed": int(os.environ.get("EDGE_DIT_QWEN_IMAGE_EDIT_SEED", "42")),
            "init_image_b64": _encode_png_base64(_build_qwen_edit_input_image(size)),
        },
        expected_width=size,
        expected_height=size,
        input_metadata={"init_image": {"width": size, "height": size}},
    )


def _build_flux_kontext_request() -> ScenarioRequest:
    size = int(os.environ.get("EDGE_DIT_FLUX_KONTEXT_SIZE", "256"))
    return ScenarioRequest(
        payload={
            "prompt": os.environ.get(
                "EDGE_DIT_FLUX_KONTEXT_PROMPT",
                "turn this abstract reference into a glossy studio product poster",
            ),
            "width": size,
            "height": size,
            "steps": int(os.environ.get("EDGE_DIT_FLUX_KONTEXT_STEPS", "1")),
            "seed": int(os.environ.get("EDGE_DIT_FLUX_KONTEXT_SEED", "42")),
            "guidance": float(os.environ.get("EDGE_DIT_FLUX_KONTEXT_GUIDANCE", "3.5")),
            "ref_images_b64": [_encode_png_base64(_build_flux_kontext_ref_image(size))],
        },
        expected_width=size,
        expected_height=size,
        input_metadata={"ref_images": [{"width": size, "height": size}]},
    )


def _build_wan_video_request() -> ScenarioRequest:
    width = int(os.environ.get("EDGE_DIT_VIDEO_WIDTH", "416"))
    height = int(os.environ.get("EDGE_DIT_VIDEO_HEIGHT", "240"))
    frames = int(os.environ.get("EDGE_DIT_VIDEO_FRAMES", "9"))
    return ScenarioRequest(
        payload={
            "prompt": os.environ.get(
                "EDGE_DIT_VIDEO_PROMPT",
                "a small robot walking through a rainy neon street",
            ),
            "width": width,
            "height": height,
            "frames": frames,
            "steps": int(os.environ.get("EDGE_DIT_VIDEO_STEPS", "1")),
            "cfg_scale": float(os.environ.get("EDGE_DIT_VIDEO_CFG_SCALE", "5.0")),
            "flow_shift": float(os.environ.get("EDGE_DIT_VIDEO_FLOW_SHIFT", "5.0")),
            "seed": int(os.environ.get("EDGE_DIT_VIDEO_SEED", "42")),
        },
        expected_width=width,
        expected_height=height,
        expected_frames=frames,
    )


@unittest.skipUnless(
    os.environ.get("EDGE_DIT_RUN_INTEGRATION") == "1",
    "set EDGE_DIT_RUN_INTEGRATION=1 to run real native smoke tests",
)
class OptionalRealServerV2SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library_path = os.environ.get("EDGE_DIT_LIBRARY")
        if not cls.library_path:
            raise unittest.SkipTest("EDGE_DIT_LIBRARY is required for server_v2 integration smoke tests")

    def _require_model_path(self, env_name: str, default: str | None = None) -> str:
        model_path = os.environ.get(env_name, default)
        if not model_path:
            self.skipTest(f"{env_name} is required for this server_v2 integration smoke test")
        if not Path(model_path).exists():
            self.skipTest(f"model path does not exist: {model_path}")
        return model_path

    def _start_server(
        self,
        *,
        model_path: str,
        engine_overrides: dict[str, object] | None = None,
    ) -> None:
        config_kwargs: dict[str, object] = {
            "model_path": model_path,
            "backend": os.environ.get("EDGE_DIT_BACKEND", "cuda"),
            "offload_params_to_cpu": True,
            "keep_text_encoder_on_cpu": True,
            "max_vram_gb": float(os.environ.get("EDGE_DIT_MAX_VRAM_GB", "8.0")),
        }
        for key, value in (engine_overrides or {}).items():
            if value is not None:
                config_kwargs[key] = value

        config = EngineConfig(**config_kwargs)
        engine = Engine(config, _library_path=self.library_path)
        self.service = ImageJobService(engine, model_name=model_path, job_ttl_seconds=3600.0)
        self.httpd = create_http_server(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def _stop_server(self) -> None:
        httpd = getattr(self, "httpd", None)
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
            self.httpd = None
        service = getattr(self, "service", None)
        if service is not None:
            service.close()
            self.service = None
        thread = getattr(self, "thread", None)
        if thread is not None:
            thread.join(timeout=5.0)
            self.thread = None
        self.base_url = None

    def tearDown(self) -> None:
        self._stop_server()

    def _assert_response_request_id(self, body: dict[str, object]) -> None:
        self.assertEqual(body.get("request_id"), _REQUEST_ID)

    def _decode_image_result(self, result: dict[str, object]) -> Image.Image:
        return _decode_png(str(result["data"][0]["b64_png"]))

    def _decode_video_result(self, result: dict[str, object]) -> list[Image.Image]:
        return [_decode_png(str(item["b64_png"])) for item in result["frames"]]

    def _save_result_artifact(
        self,
        scenario: ServerV2MatrixScenario,
        request_spec: ScenarioRequest,
        result: dict[str, object],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if scenario.kind == "image":
            image = self._decode_image_result(result)
            self.assertEqual(image.size, (request_spec.expected_width, request_spec.expected_height))
            image.save(output_path)
            self.assertTrue(output_path.exists())
            return

        self.assertEqual(result["frame_format"], "png")
        frames = self._decode_video_result(result)
        self.assertEqual(len(frames), request_spec.expected_frames)
        self.assertEqual(frames[0].size, (request_spec.expected_width, request_spec.expected_height))
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        self.assertTrue(output_path.exists())

    def _assert_input_metadata(self, parameters: dict[str, object], request_spec: ScenarioRequest) -> None:
        for name, metadata in request_spec.input_metadata.items():
            self.assertIn(name, parameters)
            actual = parameters[name]
            if isinstance(metadata, list):
                self.assertIsInstance(actual, list)
                self.assertEqual(len(actual), len(metadata))
                for expected_item, actual_item in zip(metadata, actual):
                    self.assertEqual(actual_item["width"], expected_item["width"])
                    self.assertEqual(actual_item["height"], expected_item["height"])
            else:
                self.assertEqual(actual["width"], metadata["width"])
                self.assertEqual(actual["height"], metadata["height"])

    def _matrix_output_dir(self) -> Path:
        return Path(os.environ.get("EDGE_DIT_SERVER_V2_MATRIX_OUTPUT_DIR", "/tmp/edge_dit_server_v2_matrix"))

    def _scenario_timeout(self, scenario: ServerV2MatrixScenario) -> float:
        default_text = str(scenario.default_timeout_seconds)
        return float(
            os.environ.get(
                scenario.timeout_env,
                os.environ.get("EDGE_DIT_SERVER_V2_MATRIX_TIMEOUT_SECONDS", default_text),
            )
        )

    def _matrix_scenarios(self) -> list[ServerV2MatrixScenario]:
        scenarios = [
            ServerV2MatrixScenario(
                name="FLUX.1-dev",
                slug="flux-dev",
                prefix="/ed/v2",
                kind="image",
                model_env="EDGE_DIT_FLUX_MODEL_PATH",
                model_default="",
                timeout_env="EDGE_DIT_SERVER_V2_FLUX_TIMEOUT_SECONDS",
                default_timeout_seconds=600.0,
                engine_overrides={},
                request_builder=_build_flux_request,
            ),
            ServerV2MatrixScenario(
                name="stable-diffusion-3-medium-diffusers",
                slug="sd3-medium",
                prefix="/edgedit/v2",
                kind="image",
                model_env="EDGE_DIT_SD3_MODEL_PATH",
                model_default="",
                timeout_env="EDGE_DIT_SERVER_V2_SD3_TIMEOUT_SECONDS",
                default_timeout_seconds=600.0,
                engine_overrides={"skip_t5": True},
                request_builder=_build_sd3_request,
            ),
            ServerV2MatrixScenario(
                name="Qwen-Image",
                slug="qwen-image",
                prefix="/edge-dit/v2",
                kind="image",
                model_env="EDGE_DIT_QWEN_IMAGE_MODEL_PATH",
                model_default="",
                timeout_env="EDGE_DIT_SERVER_V2_QWEN_IMAGE_TIMEOUT_SECONDS",
                default_timeout_seconds=900.0,
                engine_overrides={"weight_type": "q4_k", "keep_vae_on_cpu": True},
                request_builder=_build_qwen_image_request,
            ),
            ServerV2MatrixScenario(
                name="Qwen-Image-Edit",
                slug="qwen-image-edit",
                prefix="/ed/v2",
                kind="image",
                model_env="EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH",
                model_default="",
                timeout_env="EDGE_DIT_SERVER_V2_QWEN_IMAGE_EDIT_TIMEOUT_SECONDS",
                default_timeout_seconds=900.0,
                engine_overrides={"weight_type": "q4_k", "keep_vae_on_cpu": True},
                request_builder=_build_qwen_image_edit_request,
            ),
            ServerV2MatrixScenario(
                name="FLUX.1-Kontext-dev",
                slug="flux-kontext",
                prefix="/edgedit/v2",
                kind="image",
                model_env="EDGE_DIT_FLUX_KONTEXT_MODEL_PATH",
                model_default="",
                timeout_env="EDGE_DIT_SERVER_V2_FLUX_KONTEXT_TIMEOUT_SECONDS",
                default_timeout_seconds=900.0,
                engine_overrides={
                    "keep_vae_on_cpu": True
                    if os.environ.get("EDGE_DIT_FLUX_KONTEXT_KEEP_VAE_ON_CPU") == "1"
                    else None,
                    "weight_type": os.environ.get("EDGE_DIT_FLUX_KONTEXT_WEIGHT_TYPE"),
                },
                request_builder=_build_flux_kontext_request,
            ),
            ServerV2MatrixScenario(
                name="Wan2.1-T2V-1.3B-Diffusers",
                slug="wan-t2v",
                prefix="/edge-dit/v2",
                kind="video",
                model_env="EDGE_DIT_WAN_VIDEO_MODEL_PATH",
                model_default="",
                timeout_env="EDGE_DIT_SERVER_V2_VIDEO_TIMEOUT_SECONDS",
                default_timeout_seconds=900.0,
                engine_overrides={"keep_vae_on_cpu": True},
                request_builder=_build_wan_video_request,
            ),
        ]
        selected = os.environ.get("EDGE_DIT_SERVER_V2_MATRIX_MODELS")
        if not selected:
            return scenarios

        selected_slugs = {item.strip() for item in selected.split(",") if item.strip()}
        filtered = [scenario for scenario in scenarios if scenario.slug in selected_slugs]
        if len(filtered) != len(selected_slugs):
            missing = sorted(selected_slugs - {scenario.slug for scenario in filtered})
            raise AssertionError(f"unknown EDGE_DIT_SERVER_V2_MATRIX_MODELS entries: {', '.join(missing)}")
        return filtered

    def _exercise_matrix_scenario(self, scenario: ServerV2MatrixScenario) -> None:
        model_path = self._require_model_path(scenario.model_env, scenario.model_default)
        request_spec = scenario.request_builder()
        timeout_seconds = self._scenario_timeout(scenario)
        endpoint = (
            f"{scenario.prefix}/images/generations"
            if scenario.kind == "image"
            else f"{scenario.prefix}/videos/generations"
        )
        unsupported_endpoint = (
            f"{scenario.prefix}/videos/generations"
            if scenario.kind == "image"
            else f"{scenario.prefix}/images/generations"
        )
        output_suffix = ".png" if scenario.kind == "image" else ".gif"
        output_path = self._matrix_output_dir() / f"{scenario.slug}{output_suffix}"

        self._start_server(model_path=model_path, engine_overrides=scenario.engine_overrides)
        try:
            status_code, root = _request_json(self.base_url, "GET", "/")
            self.assertEqual(status_code, 200, root)
            self._assert_response_request_id(root)
            self.assertEqual(root["health"], "/ed/v2/health")

            health_status, health = _request_json(self.base_url, "GET", f"{scenario.prefix}/health")
            self.assertEqual(health_status, 200, health)
            self._assert_response_request_id(health)
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["model"], model_path)

            capabilities_status, capabilities = _request_json(
                self.base_url, "GET", f"{scenario.prefix}/capabilities"
            )
            self.assertEqual(capabilities_status, 200, capabilities)
            self._assert_response_request_id(capabilities)
            self.assertEqual(capabilities["model"], model_path)
            self.assertEqual(capabilities["supports"][scenario.kind], True)
            self.assertEqual(capabilities["semantics"]["results"], "stored_in_memory")
            ttl_ms = int(capabilities["semantics"]["job_ttl_ms"])

            unsupported_status, unsupported = _request_json(
                self.base_url,
                "POST",
                unsupported_endpoint,
                {"prompt": "unsupported", "width": 16, "height": 16, "steps": 1},
            )
            if scenario.kind == "image":
                self.assertEqual(capabilities["supports"]["video"], False)
            else:
                self.assertEqual(capabilities["supports"]["image"], False)
            self.assertEqual(unsupported_status, 409, unsupported)
            self.assertEqual(unsupported["error"]["code"], "unsupported")

            create_status, first_job = _request_json(self.base_url, "POST", endpoint, request_spec.payload)
            self.assertEqual(create_status, 202, first_job)
            self.assertEqual(first_job["kind"], scenario.kind)
            self.assertIn(first_job["status"], {"queued", "running", "cancelling"})
            self.assertTrue(str(first_job["status_url"]).startswith(f"{scenario.prefix}/jobs/"))
            self._assert_input_metadata(first_job["parameters"], request_spec)

            second_status, second_job = _request_json(self.base_url, "POST", endpoint, request_spec.payload)
            self.assertEqual(second_status, 202, second_job)
            self.assertEqual(second_job["kind"], scenario.kind)

            list_status, jobs = _request_json(
                self.base_url, "GET", f"{scenario.prefix}/jobs?kind={scenario.kind}&limit=10"
            )
            self.assertEqual(list_status, 200, jobs)
            job_ids = {item["id"] for item in jobs["data"]}
            self.assertIn(first_job["id"], job_ids)
            self.assertIn(second_job["id"], job_ids)

            result_pending_status, result_pending = _request_json(
                self.base_url, "GET", str(first_job["result_url"])
            )
            self.assertEqual(result_pending_status, 409, result_pending)
            self.assertEqual(result_pending["error"]["code"], "job_not_ready")

            delete_active_status, delete_active = _request_json(
                self.base_url, "DELETE", str(first_job["status_url"])
            )
            self.assertEqual(delete_active_status, 409, delete_active)
            self.assertEqual(delete_active["error"]["code"], "job_active")

            cleanup_status, cleanup = _request_json(
                self.base_url, "POST", f"{scenario.prefix}/jobs/cleanup", {}
            )
            self.assertEqual(cleanup_status, 200, cleanup)
            self.assertEqual(cleanup["removed_count"], 0)

            cancel_status, cancel = _request_json(self.base_url, "POST", str(second_job["cancel_url"]))
            self.assertIn(cancel_status, {200, 202}, cancel)
            self.assertIn(cancel["status"], {"queued", "cancelled", "cancelling"})

            cancelled = _wait_for_job_status(
                self.base_url,
                str(second_job["status_url"]),
                expected_statuses={"cancelled"},
                timeout_seconds=60.0,
            )
            self.assertEqual(cancelled["status"], "cancelled", cancelled)

            cancelled_list_status, cancelled_list = _request_json(
                self.base_url, "GET", f"{scenario.prefix}/jobs?status=cancelled&kind={scenario.kind}&limit=10"
            )
            self.assertEqual(cancelled_list_status, 200, cancelled_list)
            self.assertIn(second_job["id"], {item["id"] for item in cancelled_list["data"]})

            terminal = _wait_for_terminal_job(
                self.base_url,
                str(first_job["status_url"]),
                timeout_seconds=timeout_seconds,
            )
            self.assertEqual(terminal["status"], "succeeded", terminal)
            self.assertEqual(terminal["kind"], scenario.kind)

            result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
            self.assertEqual(result_status, 200, result)
            self.assertEqual(
                result["object"],
                "edge_dit.image_generation" if scenario.kind == "image" else "edge_dit.video_generation",
            )
            self._save_result_artifact(scenario, request_spec, result, output_path)

            succeeded_list_status, succeeded_list = _request_json(
                self.base_url, "GET", f"{scenario.prefix}/jobs?status=succeeded&kind={scenario.kind}&limit=10"
            )
            self.assertEqual(succeeded_list_status, 200, succeeded_list)
            self.assertIn(first_job["id"], {item["id"] for item in succeeded_list["data"]})

            future_now_ms = max(int(terminal["finished_ms"]), int(cancelled["finished_ms"])) + ttl_ms + 1
            cleanup_expired_status, cleanup_expired = _request_json(
                self.base_url,
                "POST",
                f"{scenario.prefix}/jobs/cleanup",
                {"now_ms": future_now_ms},
            )
            self.assertEqual(cleanup_expired_status, 200, cleanup_expired)
            self.assertEqual(cleanup_expired["removed_count"], 2, cleanup_expired)
            self.assertIn(first_job["id"], cleanup_expired["removed_ids"])
            self.assertIn(second_job["id"], cleanup_expired["removed_ids"])

            missing_first_status, missing_first = _request_json(
                self.base_url, "GET", str(first_job["status_url"])
            )
            self.assertEqual(missing_first_status, 404, missing_first)
            self.assertEqual(missing_first["error"]["code"], "job_not_found")

            missing_second_status, missing_second = _request_json(
                self.base_url, "GET", str(second_job["status_url"])
            )
            self.assertEqual(missing_second_status, 404, missing_second)
            self.assertEqual(missing_second["error"]["code"], "job_not_found")
        finally:
            self._stop_server()

    def test_generate_image_through_real_server_v2(self) -> None:
        model_path = self._require_model_path("EDGE_DIT_MODEL_PATH")
        self._start_server(model_path=model_path)

        output_path = Path(
            os.environ.get("EDGE_DIT_SERVER_V2_INTEGRATION_OUTPUT", "/tmp/edge_dit_server_v2_integration.png")
        )

        status_code, job = _request_json(
            self.base_url,
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": os.environ.get("EDGE_DIT_PROMPT", "server v2 integration smoke teapot"),
                "width": int(os.environ.get("EDGE_DIT_WIDTH", "256")),
                "height": int(os.environ.get("EDGE_DIT_HEIGHT", "256")),
                "steps": int(os.environ.get("EDGE_DIT_STEPS", "1")),
                "seed": int(os.environ.get("EDGE_DIT_SEED", "42")),
            },
        )
        self.assertEqual(status_code, 202, job)
        self.assertEqual(job["kind"], "image")

        terminal = _wait_for_terminal_job(
            self.base_url,
            str(job["status_url"]),
            timeout_seconds=float(os.environ.get("EDGE_DIT_SERVER_V2_TIMEOUT_SECONDS", "600")),
        )
        self.assertEqual(terminal["status"], "succeeded", terminal)

        result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
        self.assertEqual(result_status, 200, result)
        self.assertEqual(result["object"], "edge_dit.image_generation")

        image = self._decode_image_result(result)
        self.assertEqual(
            image.size,
            (
                int(os.environ.get("EDGE_DIT_WIDTH", "256")),
                int(os.environ.get("EDGE_DIT_HEIGHT", "256")),
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        self.assertTrue(output_path.exists())

    def test_generate_qwen_image_edit_through_real_server_v2_when_enabled(self) -> None:
        if os.environ.get("EDGE_DIT_RUN_SERVER_V2_QWEN_IMAGE_EDIT") != "1":
            self.skipTest(
                "set EDGE_DIT_RUN_SERVER_V2_QWEN_IMAGE_EDIT=1 to run the real server_v2 Qwen image-edit smoke test"
            )

        model_path = self._require_model_path(
            "EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH",
            _DEFAULT_QWEN_IMAGE_EDIT_MODEL_PATH,
        )
        size = int(os.environ.get("EDGE_DIT_QWEN_IMAGE_EDIT_SIZE", "256"))

        self._start_server(
            model_path=model_path,
            engine_overrides={"keep_vae_on_cpu": True, "weight_type": "q4_k"},
        )

        output_path = Path(
            os.environ.get(
                "EDGE_DIT_SERVER_V2_QWEN_IMAGE_EDIT_OUTPUT",
                "/tmp/edge_dit_server_v2_qwen_image_edit_integration.png",
            )
        )
        steps = int(os.environ.get("EDGE_DIT_QWEN_IMAGE_EDIT_STEPS", "2"))

        input_image = _build_qwen_edit_input_image(size)
        init_image_b64 = _encode_png_base64(input_image)

        status_code, job = _request_json(
            self.base_url,
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": os.environ.get(
                    "EDGE_DIT_QWEN_IMAGE_EDIT_PROMPT",
                    "turn this synthetic gradient into a polished product render",
                ),
                "width": size,
                "height": size,
                "steps": steps,
                "seed": int(os.environ.get("EDGE_DIT_QWEN_IMAGE_EDIT_SEED", "42")),
                "init_image_b64": init_image_b64,
            },
        )
        self.assertEqual(status_code, 202, job)
        self.assertEqual(job["kind"], "image")
        self.assertEqual(job["parameters"]["init_image"]["width"], size)
        self.assertEqual(job["parameters"]["init_image"]["height"], size)
        _wait_for_sampling_progress(
            self.base_url,
            str(job["status_url"]),
            expected_total_steps=steps,
            timeout_seconds=float(
                os.environ.get("EDGE_DIT_SERVER_V2_QWEN_IMAGE_EDIT_TIMEOUT_SECONDS", "900")
            ),
        )

        terminal = _wait_for_terminal_job(
            self.base_url,
            str(job["status_url"]),
            timeout_seconds=float(
                os.environ.get("EDGE_DIT_SERVER_V2_QWEN_IMAGE_EDIT_TIMEOUT_SECONDS", "900")
            ),
        )
        self.assertEqual(terminal["status"], "succeeded", terminal)

        result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
        self.assertEqual(result_status, 200, result)
        self.assertEqual(result["object"], "edge_dit.image_generation")

        image = self._decode_image_result(result)
        self.assertEqual(image.size, (size, size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        self.assertTrue(output_path.exists())

    def test_generate_flux_kontext_through_real_server_v2_when_enabled(self) -> None:
        if os.environ.get("EDGE_DIT_RUN_SERVER_V2_FLUX_KONTEXT") != "1":
            self.skipTest(
                "set EDGE_DIT_RUN_SERVER_V2_FLUX_KONTEXT=1 to run the real server_v2 FLUX Kontext smoke test"
            )

        model_path = self._require_model_path(
            "EDGE_DIT_FLUX_KONTEXT_MODEL_PATH",
            _DEFAULT_FLUX_KONTEXT_MODEL_PATH,
        )
        size = int(os.environ.get("EDGE_DIT_FLUX_KONTEXT_SIZE", "256"))

        self._start_server(
            model_path=model_path,
            engine_overrides={
                "keep_vae_on_cpu": True
                if os.environ.get("EDGE_DIT_FLUX_KONTEXT_KEEP_VAE_ON_CPU") == "1"
                else None,
                "weight_type": os.environ.get("EDGE_DIT_FLUX_KONTEXT_WEIGHT_TYPE"),
            },
        )

        output_path = Path(
            os.environ.get(
                "EDGE_DIT_SERVER_V2_FLUX_KONTEXT_OUTPUT",
                "/tmp/edge_dit_server_v2_flux_kontext_integration.png",
            )
        )
        steps = int(os.environ.get("EDGE_DIT_FLUX_KONTEXT_STEPS", "2"))

        ref_image_b64 = _encode_png_base64(_build_flux_kontext_ref_image(size))

        status_code, job = _request_json(
            self.base_url,
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": os.environ.get(
                    "EDGE_DIT_FLUX_KONTEXT_PROMPT",
                    "turn this abstract reference into a glossy studio product poster",
                ),
                "width": size,
                "height": size,
                "steps": steps,
                "seed": int(os.environ.get("EDGE_DIT_FLUX_KONTEXT_SEED", "42")),
                "guidance": float(os.environ.get("EDGE_DIT_FLUX_KONTEXT_GUIDANCE", "3.5")),
                "ref_images_b64": [ref_image_b64],
            },
        )
        self.assertEqual(status_code, 202, job)
        self.assertEqual(job["kind"], "image")
        self.assertEqual(len(job["parameters"]["ref_images"]), 1)
        self.assertEqual(job["parameters"]["ref_images"][0]["width"], size)
        self.assertEqual(job["parameters"]["ref_images"][0]["height"], size)
        _wait_for_sampling_progress(
            self.base_url,
            str(job["status_url"]),
            expected_total_steps=steps,
            timeout_seconds=float(
                os.environ.get("EDGE_DIT_SERVER_V2_FLUX_KONTEXT_TIMEOUT_SECONDS", "900")
            ),
        )

        terminal = _wait_for_terminal_job(
            self.base_url,
            str(job["status_url"]),
            timeout_seconds=float(
                os.environ.get("EDGE_DIT_SERVER_V2_FLUX_KONTEXT_TIMEOUT_SECONDS", "900")
            ),
        )
        self.assertEqual(terminal["status"], "succeeded", terminal)

        result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
        self.assertEqual(result_status, 200, result)
        self.assertEqual(result["object"], "edge_dit.image_generation")

        image = self._decode_image_result(result)
        self.assertEqual(image.size, (size, size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        self.assertTrue(output_path.exists())

    def test_generate_video_through_real_server_v2_when_enabled(self) -> None:
        if os.environ.get("EDGE_DIT_RUN_SERVER_V2_VIDEO") != "1":
            self.skipTest("set EDGE_DIT_RUN_SERVER_V2_VIDEO=1 to run the real server_v2 video smoke test")

        video_model_path = self._require_model_path(
            "EDGE_DIT_VIDEO_MODEL_PATH",
            _DEFAULT_WAN_VIDEO_MODEL_PATH,
        )

        self._start_server(model_path=video_model_path, engine_overrides={"keep_vae_on_cpu": True})

        output_path = Path(
            os.environ.get(
                "EDGE_DIT_SERVER_V2_VIDEO_OUTPUT",
                "/tmp/edge_dit_server_v2_integration.gif",
            )
        )

        status_code, job = _request_json(
            self.base_url,
            "POST",
            "/ed/v2/videos/generations",
            {
                "prompt": os.environ.get(
                    "EDGE_DIT_VIDEO_PROMPT",
                    "a small robot walking through a rainy neon street",
                ),
                "width": int(os.environ.get("EDGE_DIT_VIDEO_WIDTH", "416")),
                "height": int(os.environ.get("EDGE_DIT_VIDEO_HEIGHT", "240")),
                "frames": int(os.environ.get("EDGE_DIT_VIDEO_FRAMES", "9")),
                "steps": int(os.environ.get("EDGE_DIT_VIDEO_STEPS", "1")),
                "cfg_scale": float(os.environ.get("EDGE_DIT_VIDEO_CFG_SCALE", "5.0")),
                "flow_shift": float(os.environ.get("EDGE_DIT_VIDEO_FLOW_SHIFT", "5.0")),
                "seed": int(os.environ.get("EDGE_DIT_VIDEO_SEED", "42")),
            },
        )
        self.assertEqual(status_code, 202, job)
        self.assertEqual(job["kind"], "video")

        terminal = _wait_for_terminal_job(
            self.base_url,
            str(job["status_url"]),
            timeout_seconds=float(os.environ.get("EDGE_DIT_SERVER_V2_VIDEO_TIMEOUT_SECONDS", "900")),
        )
        self.assertEqual(terminal["status"], "succeeded", terminal)

        result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
        self.assertEqual(result_status, 200, result)
        self.assertEqual(result["object"], "edge_dit.video_generation")
        self.assertTrue(result["frames"])

        frames = self._decode_video_result(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        self.assertTrue(output_path.exists())

    def test_run_full_server_v2_model_matrix_when_enabled(self) -> None:
        if os.environ.get("EDGE_DIT_RUN_SERVER_V2_MATRIX") != "1":
            self.skipTest(
                "set EDGE_DIT_RUN_SERVER_V2_MATRIX=1 to run the full sequential server_v2 model matrix"
            )

        for scenario in self._matrix_scenarios():
            with self.subTest(model=scenario.slug):
                self._exercise_matrix_scenario(scenario)


if __name__ == "__main__":
    unittest.main()
