from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from edge_dit.config import EngineConfig
from edge_dit.engine import Engine
from edge_dit.server_v2 import ImageJobService, create_http_server

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real smoke test through server_v2")
    parser.add_argument("--config", required=True, help="Path to an image or video smoke JSON config")
    parser.add_argument("--kind", choices=("image", "video"), default="image")
    parser.add_argument("--output", default=None, help="Optional output path override")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--job-ttl-seconds", type=float, default=3600.0)
    return parser


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("top-level config payload must be a JSON object")
    return _expand_env(data)


def _get_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be a JSON object when provided")
    return value


def resolve_output_path(payload: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    output = payload.get("output")
    if isinstance(output, str) and output:
        return output
    return "/tmp/edge_dit_server_v2_smoke.gif" if payload.get("kind") == "video" else "/tmp/edge_dit_server_v2_smoke.png"


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    data = None
    headers = {"X-Request-ID": "server-v2-smoke"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    try:
        with _LOCAL_OPENER.open(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def wait_for_terminal_job(base_url: str, status_url: str, timeout_seconds: float) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status_code, body = request_json(base_url, "GET", status_url)
        if status_code != 200:
            raise RuntimeError(f"failed to poll job: HTTP {status_code}: {body}")
        print(
            "job",
            body.get("id"),
            body.get("status"),
            body.get("progress", {}),
            flush=True,
        )
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(1.0)
    raise TimeoutError(f"job did not finish within {timeout_seconds:.1f}s")


def save_image_result(result: dict[str, object], output: str) -> None:
    data = result.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("image result has no data frames")
    first = data[0]
    if not isinstance(first, dict) or not isinstance(first.get("b64_png"), str):
        raise RuntimeError("image result has invalid b64_png payload")
    image = Image.open(BytesIO(base64.b64decode(first["b64_png"])))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_video_result(result: dict[str, object], output: str) -> None:
    frames_payload = result.get("frames")
    if not isinstance(frames_payload, list) or not frames_payload:
        raise RuntimeError("video result has no frames")

    frames: list[Image.Image] = []
    for item in frames_payload:
        if not isinstance(item, dict) or not isinstance(item.get("b64_png"), str):
            raise RuntimeError("video frame result has invalid b64_png payload")
        frames.append(Image.open(BytesIO(base64.b64decode(item["b64_png"]))).copy())

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def main() -> int:
    args = build_parser().parse_args()
    payload = load_config_file(args.config)
    engine_config = EngineConfig(**_get_mapping(payload, "engine"))
    request_payload = _get_mapping(payload, "request")
    output = resolve_output_path({**payload, "kind": args.kind}, args.output)

    service = ImageJobService(
        Engine(engine_config),
        model_name=engine_config.model_path or "edge-dit-model",
        job_ttl_seconds=args.job_ttl_seconds,
    )
    server = create_http_server(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        endpoint = "/ed/v2/images/generations" if args.kind == "image" else "/ed/v2/videos/generations"
        status_code, job = request_json(base_url, "POST", endpoint, request_payload)
        if status_code != 202:
            raise RuntimeError(f"failed to create job: HTTP {status_code}: {job}")

        terminal = wait_for_terminal_job(base_url, str(job["status_url"]), args.timeout_seconds)
        if terminal.get("status") != "succeeded":
            raise RuntimeError(f"job did not succeed: {terminal}")

        result_status, result = request_json(base_url, "GET", str(terminal["result_url"]))
        if result_status != 200:
            raise RuntimeError(f"failed to fetch result: HTTP {result_status}: {result}")

        if args.kind == "image":
            save_image_result(result, output)
        else:
            save_video_result(result, output)

        print(f"saved server_v2 {args.kind} smoke output to {output}")
        return 0
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
