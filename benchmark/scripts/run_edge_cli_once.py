#!/usr/bin/env python3
"""Run edge-dit.cpp CLI smoke timing for non-T2I release-gate tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import re


CACHE_SUMMARY_RE = re.compile(
    r"\b(?P<mode>EasyCache|UCache|DBCache|CacheDiT|TaylorSeer|MagCache|DiCache|SenCache)\s+"
    r"(?P<verb>skipped|reused)\s+(?P<count>\d+)/(?P<total>\d+)\s+steps"
    r"(?:\s+\((?P<note>[^)]*)\))?"
)
PHASE_RE = re.compile(
    r"\[\[phase\]\]\s+stage=(?P<stage>encode|denoise|decode)\s+"
    r"event=(?P<event>begin|end)\s+t=(?P<t>[0-9]+(?:\.[0-9]+)?)"
)


def parse_phase_markers(stdout: str) -> tuple[dict[str, "float | None"], dict[str, list[float]]]:
    """Parse `[[phase]] stage=.. event=.. t=..` markers emitted by ed pipelines.

    Takes the last begin/end pair per stage (the final measured pass) and
    returns both per-component ms and [begin, end] epoch windows (used by the
    harness to segment the external gpu_monitor CSV for per-stage peak VRAM).
    """
    latest: dict[str, dict[str, float]] = {}
    for match in PHASE_RE.finditer(stdout):
        latest.setdefault(match.group("stage"), {})[match.group("event")] = float(match.group("t"))
    component: dict[str, "float | None"] = {"text_encoder": None, "dit": None, "vae": None}
    boundaries: dict[str, list[float]] = {}
    for stage, name in (("encode", "text_encoder"), ("denoise", "dit"), ("decode", "vae")):
        events = latest.get(stage, {})
        begin = events.get("begin")
        end = events.get("end")
        if begin is not None and end is not None:
            component[name] = (end - begin) * 1000.0
            boundaries[name] = [begin, end]
    return component, boundaries



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", required=True, choices=["text-to-image", "image-editing", "text-to-video"])
    parser.add_argument("--input-image")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--measured-runs", type=int, required=True)
    parser.add_argument("--devices")
    parser.add_argument("--sp-size", type=int)
    parser.add_argument("--cfg-parallel-size", type=int)
    parser.add_argument("--cache")
    parser.add_argument("--cache-threshold")
    parser.add_argument("--cache-start")
    parser.add_argument("--cache-end")
    parser.add_argument("--cache-error-decay")
    parser.add_argument("--cache-relative-threshold", action="store_true")
    parser.add_argument("--cache-absolute-threshold", action="store_true")
    parser.add_argument("--cache-no-reset-error", action="store_true")
    parser.add_argument("--cache-reset-error", action="store_true")
    parser.add_argument("--cache-fn-blocks")
    parser.add_argument("--cache-bn-blocks")
    parser.add_argument("--cache-residual-threshold")
    parser.add_argument("--cache-max-accumulated-residual-diff")
    parser.add_argument("--cache-warmup-steps")
    parser.add_argument("--cache-max-cached-steps")
    parser.add_argument("--cache-max-continuous-cached-steps")
    parser.add_argument("--cache-taylor-order")
    parser.add_argument("--cache-taylor-skip")
    parser.add_argument("--cache-scm-mask")
    parser.add_argument("--cache-calibrate")
    parser.add_argument("--cache-profile")
    parser.add_argument("--cache-static-scm", action="store_true")
    parser.add_argument("--cache-dynamic-scm", action="store_true")
    parser.add_argument("--qwen-image-zero-cond-t", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-tile-size")
    parser.add_argument("--tensor-type-rules")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--keep-text-encoder-on-cpu", action="store_true")
    parser.add_argument("--keep-vae-on-cpu", action="store_true")
    parser.add_argument("--max-vram")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--profile-graph-cuts", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / "samples" / "edge-cli"
    sample_dir.mkdir(parents=True, exist_ok=True)

    warmup_ms: list[float] = []
    measured_ms: list[float] = []
    runs: list[dict[str, object]] = []
    cache_events: list[dict[str, object]] = []
    last_measured_stdout = ""
    total_runs = args.warmup_runs + args.measured_runs
    for index in range(total_runs):
        phase = "warmup" if index < args.warmup_runs else "measured"
        phase_index = index if phase == "warmup" else index - args.warmup_runs
        output = sample_dir / output_name(args.task, phase, phase_index)
        command = build_command(args, output)
        if index == 0:
            (output_dir / "adapter_command.txt").write_text(
                shlex.join(command) + "\n",
                encoding="utf-8",
            )

        start = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        cache_events.extend(parse_cache_events(completed.stdout + "\n" + completed.stderr))
        runs.append(
            {
                "phase": phase,
                "index": phase_index,
                "returncode": completed.returncode,
                "elapsed_ms": elapsed_ms,
                "output": str(output),
            }
        )
        print(f"[edge-cli-once] {phase} {phase_index} {elapsed_ms / 1000.0:.3f}s", flush=True)
        if completed.returncode != 0:
            write_metadata(output_dir, sample_dir, runs)
            return completed.returncode
        if phase == "warmup":
            warmup_ms.append(elapsed_ms)
        else:
            measured_ms.append(elapsed_ms)
            last_measured_stdout = completed.stdout

    write_metadata(output_dir, sample_dir, runs)
    component_ms, stage_boundaries = parse_phase_markers(last_measured_stdout)
    if component_ms.get("dit") is not None and args.steps > 0:
        component_ms["per_step_avg"] = component_ms["dit"] / args.steps
    else:
        component_ms["per_step_avg"] = None
    metrics = {
        "schema_version": 1,
        "metric_source": "edge_dit_cli_once",
        "measurement_boundary": "cli_single_run_includes_load_and_output_encoding",
        "load_ms": None,
        "warmup_ms": warmup_ms,
        "measured_ms": measured_ms,
        "cache_events": cache_events,
        "component_ms": component_ms,
        "stage_boundaries": stage_boundaries,
        "sample_output_dir": str(sample_dir),
    }
    (output_dir / "runner_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def build_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        args.binary,
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--guidance",
        str(args.guidance),
        "--cfg-scale",
        str(args.cfg_scale),
        "--output",
        str(output),
    ]
    if args.task == "image-editing":
        if not args.input_image:
            raise SystemExit("--input-image is required for image-editing")
        command.extend(["--image", args.input_image])
    if args.task == "text-to-video":
        command.append("--video")
        command.extend(["--frames", str(args.frames)])
        if args.fps is not None:
            command.extend(["--fps", str(args.fps)])
    if args.dtype and args.dtype != "auto":
        command.extend(["--type", args.dtype])
    if args.devices:
        command.extend(["--devices", args.devices])
    if args.sp_size:
        command.extend(["--sp-size", str(args.sp_size)])
    if args.cfg_parallel_size:
        command.extend(["--cfg-parallel-size", str(args.cfg_parallel_size)])
    if args.cache and args.cache != "off":
        command.extend(["--cache", args.cache])
    append_cache_options(command, args)
    if args.qwen_image_zero_cond_t:
        command.append("--qwen-image-zero-cond-t")
    if args.vae_tiling:
        command.append("--vae-tiling")
    if args.vae_tile_size is not None:
        command.extend(["--vae-tile-size", str(args.vae_tile_size)])
    if args.tensor_type_rules:
        command.extend(["--tensor-type-rules", str(args.tensor_type_rules)])
    if args.offload_to_cpu:
        command.append("--offload-to-cpu")
    if args.keep_text_encoder_on_cpu:
        command.append("--keep-text-encoder-on-cpu")
    if args.keep_vae_on_cpu:
        command.append("--keep-vae-on-cpu")
    if args.max_vram:
        command.extend(["--max-vram", args.max_vram])
    if args.no_flash_attention:
        command.append("--no-flash-attention")
    if args.profile_graph_cuts:
        command.append("--profile-graph-cuts")
    return command


def parse_cache_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for match in CACHE_SUMMARY_RE.finditer(text):
        events.append(
            {
                "mode": match.group("mode"),
                "verb": match.group("verb"),
                "count": int(match.group("count")),
                "total": int(match.group("total")),
                "note": match.group("note"),
            }
        )
    return events


def append_cache_options(command: list[str], args: argparse.Namespace) -> None:
    value_flags = [
        ("--cache-threshold", args.cache_threshold),
        ("--cache-start", args.cache_start),
        ("--cache-end", args.cache_end),
        ("--cache-error-decay", args.cache_error_decay),
        ("--cache-fn-blocks", args.cache_fn_blocks),
        ("--cache-bn-blocks", args.cache_bn_blocks),
        ("--cache-residual-threshold", args.cache_residual_threshold),
        ("--cache-max-accumulated-residual-diff", args.cache_max_accumulated_residual_diff),
        ("--cache-warmup-steps", args.cache_warmup_steps),
        ("--cache-max-cached-steps", args.cache_max_cached_steps),
        ("--cache-max-continuous-cached-steps", args.cache_max_continuous_cached_steps),
        ("--cache-taylor-order", args.cache_taylor_order),
        ("--cache-taylor-skip", args.cache_taylor_skip),
        ("--cache-scm-mask", args.cache_scm_mask),
        ("--cache-calibrate", args.cache_calibrate),
        ("--cache-profile", args.cache_profile),
    ]
    for flag, value in value_flags:
        if value is not None:
            command.extend([flag, str(value)])
    bool_flags = [
        ("--cache-relative-threshold", args.cache_relative_threshold),
        ("--cache-absolute-threshold", args.cache_absolute_threshold),
        ("--cache-no-reset-error", args.cache_no_reset_error),
        ("--cache-reset-error", args.cache_reset_error),
        ("--cache-static-scm", args.cache_static_scm),
        ("--cache-dynamic-scm", args.cache_dynamic_scm),
    ]
    for flag, enabled in bool_flags:
        if enabled:
            command.append(flag)


def output_name(task: str, phase: str, index: int) -> str:
    if task == "text-to-video":
        return f"{phase}_{index:03d}.avi"
    return f"{phase}_{index:03d}.png"


def write_metadata(output_dir: Path, sample_dir: Path, runs: list[dict[str, object]]) -> None:
    (sample_dir / "metadata.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
