from __future__ import annotations

import base64
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from io import BytesIO

from PIL import Image

from edge_dit.errors import GenerationCancelledError
from edge_dit.server_v2 import ImageJobService, create_http_server

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FakeEngine:
    pipeline_name = "flux"
    version_name = "flux-dev"
    supports_image = True
    supports_video = True
    default_sampler = 1

    def __init__(self, behaviors: list[str] | None = None) -> None:
        self.behaviors = list(behaviors or ["success"])
        self.cancel_requests = 0
        self.requests = []
        self.video_requests = []
        self.closed = False
        self.started = threading.Event()
        self.allow_finish = threading.Event()
        self._cancel_requested = threading.Event()
        self._progress_current = 0
        self._progress_total = 0

    def default_scheduler(self, sampler: int | None = None) -> int:
        return 2

    def progress_steps(self) -> tuple[int, int]:
        return self._progress_current, self._progress_total

    def request_cancel(self) -> None:
        self.cancel_requests += 1
        self._cancel_requested.set()

    def generate_image(self, request) -> list[Image.Image]:
        behavior = self.behaviors.pop(0) if self.behaviors else "success"
        self.requests.append(request)
        self.started.set()
        self._cancel_requested.clear()
        total_steps = request.steps or 4
        self._progress_total = total_steps

        for step in range(1, total_steps + 1):
            self._progress_current = step
            if behavior == "blocking":
                while not self.allow_finish.is_set() and not self._cancel_requested.is_set():
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

            if self._cancel_requested.is_set():
                self._progress_current = 0
                self._progress_total = 0
                raise GenerationCancelledError("generation cancelled")

        self._progress_current = 0
        self._progress_total = 0
        self.allow_finish.clear()
        return [Image.new("RGB", (request.width or 8, request.height or 8), color=(12, 34, 56))]

    def generate_video(self, request) -> list[Image.Image]:
        behavior = self.behaviors.pop(0) if self.behaviors else "success"
        self.video_requests.append(request)
        self.started.set()
        self._cancel_requested.clear()
        total_steps = request.steps or 2
        self._progress_total = total_steps

        for step in range(1, total_steps + 1):
            self._progress_current = step
            if behavior == "blocking":
                while not self.allow_finish.is_set() and not self._cancel_requested.is_set():
                    time.sleep(0.01)
            else:
                time.sleep(0.01)
            if self._cancel_requested.is_set():
                self._progress_current = 0
                self._progress_total = 0
                raise GenerationCancelledError("generation cancelled")

        self._progress_current = 0
        self._progress_total = 0
        self.allow_finish.clear()
        width = request.width or 8
        height = request.height or 8
        frame_count = request.frames or 2
        return [
            Image.new("RGB", (width, height), color=(index, index + 1, index + 2))
            for index in range(frame_count)
        ]

    def close(self) -> None:
        self.closed = True
        self.allow_finish.set()


class ServerV2HTTPTests(unittest.TestCase):
    @staticmethod
    def encode_image(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def start_server(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.service = ImageJobService(engine, model_name="fake-model", job_ttl_seconds=3600)
        self.httpd = create_http_server(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self) -> None:
        httpd = getattr(self, "httpd", None)
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        service = getattr(self, "service", None)
        if service is not None:
            service.close()
        thread = getattr(self, "thread", None)
        if thread is not None:
            thread.join(timeout=1.0)

    def request_json(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with _LOCAL_OPENER.open(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def wait_for_status(self, job_path: str, expected: str, timeout: float = 5.0) -> dict[str, object]:
        deadline = time.time() + timeout
        last_body: dict[str, object] | None = None
        while time.time() < deadline:
            status_code, body = self.request_json("GET", job_path)
            self.assertEqual(status_code, 200)
            last_body = body
            if body["status"] == expected:
                return body
            time.sleep(0.02)
        self.fail(f"timed out waiting for {expected}; last body was {last_body}")

    def test_create_job_and_fetch_result(self) -> None:
        self.start_server(FakeEngine(["success"]))

        status_code, capabilities = self.request_json("GET", "/ed/v2/capabilities")
        self.assertEqual(status_code, 200)
        self.assertTrue(capabilities["supports"]["image"])
        self.assertIn("/ed/v2/jobs/{job_id}", capabilities["endpoints"])

        status_code, body = self.request_json(
            "POST",
            "/edge-dit/v2/images/generations",
            {
                "prompt": "a glass teapot on a table",
                "width": 16,
                "height": 12,
                "steps": 2,
                "cache": {"mode": "disabled"},
            },
        )
        self.assertEqual(status_code, 202)
        self.assertIn("request_id", body)
        self.assertEqual(body["parameters"]["width"], 16)
        self.assertEqual(body["kind"], "image")
        self.assertTrue(body["status_url"].startswith("/edge-dit/v2/jobs/"))

        job = self.wait_for_status(body["status_url"], "succeeded")
        self.assertEqual(job["result_url"], body["result_url"])

        result_status, result = self.request_json("GET", job["result_url"])
        self.assertEqual(result_status, 200)
        self.assertEqual(result["parameters"]["height"], 12)
        self.assertEqual(result["data"][0]["metadata"]["format"], "png")

        decoded = base64.b64decode(result["data"][0]["b64_png"])
        image = Image.open(BytesIO(decoded))
        self.assertEqual(image.size, (16, 12))

    def test_create_image_job_with_init_image(self) -> None:
        self.start_server(FakeEngine(["success"]))

        status_code, body = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": "edit this scene",
                "width": 16,
                "height": 16,
                "steps": 1,
                "init_image_b64": self.encode_image(Image.new("RGB", (4, 5), color=(1, 2, 3))),
            },
        )
        self.assertEqual(status_code, 202)
        self.assertEqual(body["parameters"]["init_image"]["width"], 4)
        self.assertEqual(body["parameters"]["init_image"]["height"], 5)

        self.wait_for_status(body["status_url"], "succeeded")
        self.assertEqual(self.engine.requests[0].init_image.size, (4, 5))

    def test_create_image_job_with_ref_images(self) -> None:
        self.start_server(FakeEngine(["success"]))

        status_code, body = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": "restyle this photo",
                "width": 16,
                "height": 16,
                "steps": 1,
                "ref_images_b64": [
                    self.encode_image(Image.new("RGB", (6, 7), color=(3, 2, 1))),
                    self.encode_image(Image.new("RGB", (8, 9), color=(4, 5, 6))),
                ],
            },
        )
        self.assertEqual(status_code, 202)
        self.assertEqual(len(body["parameters"]["ref_images"]), 2)
        self.assertEqual(body["parameters"]["ref_images"][1]["width"], 8)

        self.wait_for_status(body["status_url"], "succeeded")
        self.assertEqual([image.size for image in self.engine.requests[0].ref_images], [(6, 7), (8, 9)])

    def test_invalid_image_payload_returns_structured_error(self) -> None:
        self.start_server(FakeEngine(["success"]))

        status_code, body = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": "broken",
                "width": 16,
                "height": 16,
                "steps": 1,
                "init_image_b64": "not-base64",
            },
        )
        self.assertEqual(status_code, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")

    def test_create_video_job_and_fetch_frame_result(self) -> None:
        self.start_server(FakeEngine(["success"]))

        status_code, body = self.request_json(
            "POST",
            "/ed/v2/videos/generations",
            {
                "prompt": "a small robot walking through rain",
                "width": 10,
                "height": 6,
                "frames": 3,
                "steps": 2,
            },
        )
        self.assertEqual(status_code, 202)
        self.assertEqual(body["kind"], "video")
        self.assertEqual(body["parameters"]["frames"], 3)

        job = self.wait_for_status(body["status_url"], "succeeded")
        self.assertEqual(job["kind"], "video")

        result_status, result = self.request_json("GET", job["result_url"])
        self.assertEqual(result_status, 200)
        self.assertEqual(result["object"], "edge_dit.video_generation")
        self.assertEqual(result["frame_format"], "png")
        self.assertEqual(len(result["frames"]), 3)

        decoded = base64.b64decode(result["frames"][0]["b64_png"])
        frame = Image.open(BytesIO(decoded))
        self.assertEqual(frame.size, (10, 6))

    def test_cancel_running_job(self) -> None:
        self.start_server(FakeEngine(["blocking"]))

        status_code, body = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "cancel me", "width": 8, "height": 8, "steps": 3},
        )
        self.assertEqual(status_code, 202)
        self.assertTrue(self.engine.started.wait(timeout=2.0))

        status_code, running = self.request_json("GET", body["status_url"])
        self.assertEqual(status_code, 200)
        self.assertIn(running["status"], {"running", "cancelling"})
        self.assertEqual(running["progress"]["total_steps"], 3)

        cancel_status, cancelled = self.request_json("POST", body["cancel_url"])
        self.assertEqual(cancel_status, 202)
        self.assertTrue(cancelled["cancel_requested"])

        terminal = self.wait_for_status(body["status_url"], "cancelled")
        self.assertEqual(terminal["progress"]["current_step"], 0)
        self.assertEqual(self.engine.cancel_requests, 1)

    def test_cancel_queued_job_before_it_starts(self) -> None:
        self.start_server(FakeEngine(["blocking", "success"]))

        first_status, first = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "first", "width": 8, "height": 8, "steps": 2},
        )
        self.assertEqual(first_status, 202)
        self.assertTrue(self.engine.started.wait(timeout=2.0))

        second_status, second = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "second", "width": 8, "height": 8, "steps": 2},
        )
        self.assertEqual(second_status, 202)

        cancel_status, cancel_body = self.request_json("POST", second["cancel_url"])
        self.assertEqual(cancel_status, 200)
        self.assertEqual(cancel_body["status"], "cancelled")

        self.engine.allow_finish.set()
        self.wait_for_status(first["status_url"], "succeeded")
        time.sleep(0.1)

        second_job_status, second_job = self.request_json("GET", second["status_url"])
        self.assertEqual(second_job_status, 200)
        self.assertEqual(second_job["status"], "cancelled")
        self.assertEqual(len(self.engine.requests), 1)

    def test_list_jobs_filters_by_kind_and_status(self) -> None:
        self.start_server(FakeEngine(["success", "success"]))

        image_status, image_job = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "image", "width": 8, "height": 8, "steps": 1},
        )
        self.assertEqual(image_status, 202)
        video_status, video_job = self.request_json(
            "POST",
            "/ed/v2/videos/generations",
            {"prompt": "video", "width": 8, "height": 8, "frames": 2, "steps": 1},
        )
        self.assertEqual(video_status, 202)
        self.wait_for_status(image_job["status_url"], "succeeded")
        self.wait_for_status(video_job["status_url"], "succeeded")

        status_code, body = self.request_json("GET", "/ed/v2/jobs?kind=video&status=succeeded")
        self.assertEqual(status_code, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(body["data"][0]["kind"], "video")
        self.assertEqual(body["data"][0]["status"], "succeeded")

    def test_cleanup_expired_jobs_and_delete_terminal_job(self) -> None:
        self.start_server(FakeEngine(["success", "success"]))

        first_status, first = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "cleanup", "width": 8, "height": 8, "steps": 1},
        )
        self.assertEqual(first_status, 202)
        first_done = self.wait_for_status(first["status_url"], "succeeded")
        self.assertIsInstance(first_done["expires_ms"], int)

        self.service._job_ttl_ms = 1
        time.sleep(0.01)
        cleanup_status, cleanup = self.request_json("POST", "/ed/v2/jobs/cleanup")
        self.assertEqual(cleanup_status, 200)
        self.assertEqual(cleanup["removed_count"], 1)
        self.assertIn(first["id"], cleanup["removed_ids"])

        missing_status, missing = self.request_json("GET", first["status_url"])
        self.assertEqual(missing_status, 404)
        self.assertEqual(missing["error"]["code"], "job_not_found")

        self.service._job_ttl_ms = 3600000
        second_status, second = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "delete", "width": 8, "height": 8, "steps": 1},
        )
        self.assertEqual(second_status, 202)
        self.wait_for_status(second["status_url"], "succeeded")

        delete_status, deleted = self.request_json("DELETE", second["status_url"])
        self.assertEqual(delete_status, 200)
        self.assertEqual(deleted["object"], "edge_dit.job_deleted")
        self.assertEqual(deleted["id"], second["id"])

    def test_not_ready_result_uses_structured_error(self) -> None:
        self.start_server(FakeEngine(["blocking"]))

        status_code, body = self.request_json(
            "POST",
            "/ed/v2/images/generations",
            {"prompt": "not ready", "width": 8, "height": 8, "steps": 2},
        )
        self.assertEqual(status_code, 202)
        self.assertTrue(self.engine.started.wait(timeout=2.0))

        result_status, result = self.request_json("GET", body["result_url"])
        self.assertEqual(result_status, 409)
        self.assertEqual(result["error"]["code"], "job_not_ready")
        self.assertEqual(result["error"]["status"], 409)
        self.assertIn("request_id", result["error"])

        self.engine.allow_finish.set()
        self.wait_for_status(body["status_url"], "succeeded")


if __name__ == "__main__":
    unittest.main()
