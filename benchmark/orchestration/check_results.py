#!/usr/bin/env python3
"""Check benchmark result files against the frozen result shape."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any


REQUIRED_RESULT_FILES = [
    "config.resolved.yaml",
    "environment.json",
    "command.txt",
    "stdout.log",
    "stderr.log",
    "status.json",
    "timing.json",
    "gpu_memory.csv",
    "process_memory.csv",
    "result.json",
    "metrics.json",
]

RESULT_STATUS = {"success", "failed", "unsupported", "skipped"}
SYSTEMS = {"edge-dit.cpp", "diffusers", "stable-diffusion.cpp", "xdit"}
TASKS = {"text-to-image", "image-editing", "text-to-video"}

LATENCY_KEYS = [
    "load",
    "first_generation",
    "steady_state_median",
    "steady_state_p90",
    "steady_state_mean",
    "steady_state_std",
    "coefficient_of_variation",
    "text_encoder",
    "dit",
    "vae",
    "per_step_avg",
]
MEMORY_KEYS = ["peak_vram_mib", "peak_host_rss_mib"]
PARALLEL_KEYS = [
    "gpu_count",
    "speedup",
    "scaling_efficiency",
    "communication_ms",
    "all_to_all_ms",
    "packing_ms",
    "receive_preparation_ms",
    "graph_segment_count",
]
QUALITY_KEYS = ["psnr", "ssim", "lpips", "clip", "image_reward"]
CACHE_KEYS = ["mode", "steps_reused", "total_steps", "reuse_ratio"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    missing = [name for name in REQUIRED_RESULT_FILES if not (args.result_dir / name).exists()]
    errors.extend(f"missing {name}" for name in missing)

    result_json = args.result_dir / "result.json"
    if result_json.exists():
        try:
            result = json.loads(result_json.read_text(encoding="utf-8"))
            errors.extend(validate_result(result))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid result.json: {exc}")
    status_json = args.result_dir / "status.json"
    if status_json.exists():
        try:
            status = json.loads(status_json.read_text(encoding="utf-8"))
            if status.get("status") not in RESULT_STATUS:
                errors.append("status.json has invalid status")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid status.json: {exc}")
    environment_json = args.result_dir / "environment.json"
    if environment_json.exists():
        try:
            environment = json.loads(environment_json.read_text(encoding="utf-8"))
            if "hardware" not in environment or "software" not in environment:
                errors.append("environment.json must include hardware and software")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid environment.json: {exc}")

    if errors:
        for error in errors:
            print(error)
        return 1
    print("result directory shape looks complete")
    return 0


def validate_result(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "run_id",
        "status",
        "system",
        "workload",
        "model",
        "task",
        "warmup_runs",
        "measured_runs",
        "latency_ms",
        "memory",
        "parallel",
        "quality",
    ]
    for key in required:
        if key not in result:
            errors.append(f"result.json missing {key}")
    if errors:
        return errors

    if result["status"] not in RESULT_STATUS:
        errors.append("result.json has invalid status")
    if result["system"] not in SYSTEMS:
        errors.append("result.json has invalid system")
    if result["task"] not in TASKS:
        errors.append("result.json has invalid task")
    for key in ["run_id", "workload", "model"]:
        if not isinstance(result[key], str) or not result[key]:
            errors.append(f"result.json {key} must be a non-empty string")
    for key in ["warmup_runs", "measured_runs"]:
        if not isinstance(result[key], int) or result[key] < 0:
            errors.append(f"result.json {key} must be a non-negative integer")

    errors.extend(validate_number_object(result["latency_ms"], LATENCY_KEYS, "latency_ms"))
    errors.extend(validate_number_object(result["memory"], MEMORY_KEYS, "memory"))
    errors.extend(validate_number_object(result["parallel"], PARALLEL_KEYS, "parallel"))
    errors.extend(validate_number_object(result["quality"], QUALITY_KEYS, "quality"))
    if "cache" in result:
        errors.extend(validate_cache_object(result["cache"]))

    parallel = result.get("parallel", {})
    if not isinstance(parallel.get("gpu_count"), int) or parallel.get("gpu_count", 0) < 1:
        errors.append("parallel.gpu_count must be a positive integer")
    graph_segment_count = parallel.get("graph_segment_count")
    if graph_segment_count is not None and (
        not isinstance(graph_segment_count, int) or graph_segment_count < 0
    ):
        errors.append("parallel.graph_segment_count must be null or a non-negative integer")
    return errors


def validate_number_object(
    value: Any,
    keys: list[str],
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{prefix} must be an object"]
    extra = sorted(set(value) - set(keys))
    missing = sorted(set(keys) - set(value))
    for key in missing:
        errors.append(f"{prefix} missing {key}")
    for key in extra:
        errors.append(f"{prefix} has unexpected key {key}")
    for key in keys:
        item = value.get(key)
        if item is None:
            continue
        if not isinstance(item, (int, float)) or item < 0:
            errors.append(f"{prefix}.{key} must be null or a non-negative number")
    return errors


def validate_cache_object(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["cache must be an object"]
    extra = sorted(set(value) - set(CACHE_KEYS))
    missing = sorted(set(CACHE_KEYS) - set(value))
    for key in missing:
        errors.append(f"cache missing {key}")
    for key in extra:
        errors.append(f"cache has unexpected key {key}")
    mode = value.get("mode")
    if mode is not None and (not isinstance(mode, str) or not mode):
        errors.append("cache.mode must be null or a non-empty string")
    for key in ["steps_reused", "total_steps"]:
        item = value.get(key)
        if item is None:
            continue
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            errors.append(f"cache.{key} must be null or a non-negative integer")
    ratio = value.get("reuse_ratio")
    if ratio is not None and (not isinstance(ratio, (int, float)) or not 0.0 <= ratio <= 1.0):
        errors.append("cache.reuse_ratio must be null or a number in [0, 1]")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
