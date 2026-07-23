#!/usr/bin/env python3
"""Aggregate benchmark result.json files into a compact JSON summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help="Result root to scan. May be passed multiple times.",
    )
    parser.add_argument(
        "--include-workload",
        action="append",
        default=[],
        help=(
            "Only include rows for this workload id. May be passed multiple "
            "times. This is used to keep diagnostic workloads out of official "
            "tables."
        ),
    )
    parser.add_argument("--suite-id", default="unknown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    include_workloads = set(args.include_workload)
    result_roots = args.results_dir or [Path("benchmark/results")]
    for result_root, path in iter_result_paths(result_roots):
        item = json.loads(path.read_text(encoding="utf-8"))
        if include_workloads and item.get("workload") not in include_workloads:
            continue
        relative = path.relative_to(result_root)
        suite = relative.parts[0] if relative.parts else args.suite_id
        resolved_config = load_resolved_config(path.parent)
        metrics_path = path.parent / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runner_metrics_path = path.parent / "runner_metrics.json"
        runner_metrics = {}
        if runner_metrics_path.exists():
            runner_metrics = json.loads(runner_metrics_path.read_text(encoding="utf-8"))
            if not isinstance(runner_metrics, dict):
                runner_metrics = {}
        measurement_boundary = metrics.get("measurement_boundary")
        if measurement_boundary is None and item.get("status") == "skipped":
            measurement_boundary = "not executed due to preflight failure"
        latency = item.get("latency_ms", {})
        memory = item.get("memory", {})
        parallel = item.get("parallel", {})
        quality = merged_quality(item, metrics)
        cache = item.get("cache") or runner_metrics.get("cache") or {}
        generation = generation_options(resolved_config)
        results.append(
            {
                "suite": suite,
                "system": item.get("system"),
                "workload": item.get("workload"),
                "model": item.get("model"),
                "task": item.get("task"),
                "scenario": resolved_config.get("scenario_id", "default"),
                "status": item.get("status"),
                "parallel_mode": resolved_config.get("parallel_mode"),
                "gpu_count": parallel.get("gpu_count", 1),
                "width": generation.get("width"),
                "height": generation.get("height"),
                "frames": generation.get("frames"),
                "steps": generation.get("steps"),
                "precision": generation.get("precision"),
                "run_options": resolved_config.get("run_options", {}),
                "load_ms": latency.get("load"),
                "steady_state_median_ms": latency.get("steady_state_median"),
                "steady_state_p90_ms": latency.get("steady_state_p90"),
                "coefficient_of_variation": latency.get("coefficient_of_variation"),
                "text_encoder_ms": latency.get("text_encoder"),
                "dit_ms": latency.get("dit"),
                "vae_ms": latency.get("vae"),
                "per_step_avg_ms": latency.get("per_step_avg"),
                "peak_vram_mib": memory.get("peak_vram_mib"),
                "peak_host_rss_mib": memory.get("peak_host_rss_mib"),
                "speedup": None,
                "scaling_efficiency": None,
                "communication_ms": parallel.get("communication_ms"),
                "all_to_all_ms": parallel.get("all_to_all_ms"),
                "packing_ms": parallel.get("packing_ms"),
                "receive_preparation_ms": parallel.get("receive_preparation_ms"),
                "graph_segment_count": parallel.get("graph_segment_count"),
                "metric_source": metrics.get("metric_source"),
                "measurement_boundary": measurement_boundary,
                "quality_reference": quality_reference(item, resolved_config),
                "psnr": quality.get("psnr"),
                "ssim": quality.get("ssim"),
                "lpips": quality.get("lpips"),
                "clip": quality.get("clip"),
                "image_reward": quality.get("image_reward"),
                "cache_events": runner_metrics.get("cache_events", []),
                "cache_mode": cache.get("mode"),
                "cache_steps_reused": cache.get("steps_reused"),
                "cache_total_steps": cache.get("total_steps"),
                "cache_reuse_ratio": cache.get("reuse_ratio"),
                "sample_output_dir": runner_metrics.get("sample_output_dir"),
            }
        )
    add_parallel_scaling(results)

    summary = {
        "schema_version": 1,
        "suite_id": args.suite_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def iter_result_paths(result_roots: list[Path]) -> list[tuple[Path, Path]]:
    paths = []
    seen = set()
    for root in result_roots:
        resolved_root = root.resolve()
        for path in sorted(root.rglob("result.json")):
            resolved_path = path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            paths.append((resolved_root, resolved_path))
    return sorted(paths, key=lambda item: str(item[1]))


def load_resolved_config(result_dir: Path) -> dict:
    path = result_dir / "config.resolved.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def merged_quality(item: dict, metrics: dict) -> dict[str, Any]:
    result_quality = item.get("quality")
    metric_quality = metrics.get("quality")
    quality: dict[str, Any] = {}
    for source in (metric_quality, result_quality):
        if not isinstance(source, dict):
            continue
        for key in ["psnr", "ssim", "lpips", "clip", "image_reward"]:
            value = source.get(key)
            if quality.get(key) is None:
                quality[key] = value
    return quality


def quality_reference(item: dict, resolved_config: dict) -> str | None:
    value = item.get("quality_reference")
    if isinstance(value, str):
        return value
    workload = resolved_config.get("workload", {})
    if isinstance(workload, dict):
        quality = workload.get("quality", {})
        if isinstance(quality, dict):
            value = quality.get("reference_policy")
            if isinstance(value, str):
                return value
    return None


def generation_options(resolved_config: dict) -> dict[str, Any]:
    workload = resolved_config.get("workload", {})
    if not isinstance(workload, dict):
        return {}
    generation = workload.get("generation", {})
    return generation if isinstance(generation, dict) else {}


def add_parallel_scaling(results: list[dict]) -> None:
    baselines: dict[tuple[object, object, object, object], float] = {}
    for row in results:
        if row.get("status") != "success":
            continue
        if not is_parallel_row(row):
            continue
        if row.get("gpu_count") != 1:
            continue
        median = row.get("steady_state_median_ms")
        if not isinstance(median, (int, float)) or median <= 0:
            continue
        key = (
            row.get("suite"),
            row.get("system"),
            row.get("workload"),
            row.get("parallel_mode"),
        )
        baselines[key] = float(median)

    for row in results:
        if row.get("status") != "success" or not is_parallel_row(row):
            continue
        median = row.get("steady_state_median_ms")
        gpu_count = row.get("gpu_count")
        if not isinstance(median, (int, float)) or median <= 0:
            continue
        if not isinstance(gpu_count, int) or gpu_count <= 0:
            continue
        key = (
            row.get("suite"),
            row.get("system"),
            row.get("workload"),
            row.get("parallel_mode"),
        )
        baseline = baselines.get(key)
        if baseline is None:
            continue
        speedup = baseline / float(median)
        row["speedup"] = speedup
        row["scaling_efficiency"] = speedup / gpu_count


def is_parallel_row(row: dict) -> bool:
    return row.get("parallel_mode") is not None


if __name__ == "__main__":
    raise SystemExit(main())
