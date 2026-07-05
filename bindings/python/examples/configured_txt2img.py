from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from edge_dit import Engine, EngineConfig, ImageRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run edge-dit text-to-image from a JSON config file"
    )
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    parser.add_argument("--output", default=None, help="Optional output path override")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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


def build_engine_config(payload: dict[str, Any]) -> EngineConfig:
    return EngineConfig(**_get_mapping(payload, "engine"))


def build_image_request(payload: dict[str, Any]) -> ImageRequest:
    return ImageRequest.from_kwargs(**_get_mapping(payload, "request"))


def resolve_output_path(payload: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    output = payload.get("output", "output.png")
    if not isinstance(output, str) or not output:
        raise ValueError("'output' must be a non-empty string when provided")
    return output


def main() -> int:
    args = parse_args()
    payload = load_config_file(args.config)
    config = build_engine_config(payload)
    request = build_image_request(payload)
    output = resolve_output_path(payload, args.output)

    with Engine(config) as engine:
        images = engine.generate_image(request)
        images[0].save(output)

    print(f"saved image to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
