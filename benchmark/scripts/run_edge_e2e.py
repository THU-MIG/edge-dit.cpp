#!/usr/bin/env python3
"""Run edge-dit.cpp load-once e2e generation timing via ed-sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


PASS_RE = re.compile(
    r"\[ed-sample\]\s+pass\s+\d+/\d+\s+\d+/\d+\s+"
    r"seed=.*?\s+([0-9]+(?:\.[0-9]+)?)s(?P<warmup>\s+\(warmup\))?"
)
PHASE_RE = re.compile(
    r"\[\[phase\]\]\s+stage=(?P<stage>encode|denoise|decode)\s+"
    r"event=(?P<event>begin|end)\s+t=(?P<t>[0-9]+(?:\.[0-9]+)?)"
)
CACHE_SUMMARY_RE = re.compile(
    r"\b(?P<mode>EasyCache|UCache|DBCache|CacheDiT|TaylorSeer|MagCache|DiCache|SenCache)\s+"
    r"(?P<verb>skipped|reused)\s+(?P<count>\d+)/(?P<total>\d+)\s+steps"
    r"(?:\s+\((?P<note>[^)]*)\))?"
)

# --- Actual auto-fit / auto-allocate decisions (engine LOG_INFO, on STDERR) --------- #
# Under --auto-fit / --auto-allocate the engine IGNORES the user's --type and picks the
# real DiT/TE quantization tier and per-component resident/offload placement at load time,
# logging each decision to stderr (src/core/runtime/model_runtime.cpp). When present these
# are ground truth of what actually ran, which can differ from the requested tier in the
# job (e.g. requested q8 but auto-fit downgraded the DiT to q4_K, or offloaded it). We
# parse them so the report can show the real placement instead of the request.
AUTOFIT_TE_QUANT_RE = re.compile(
    r"auto-fit:\s+TE quant set to (?P<q>\S+)"
)
# The three DiT ladder OUTCOMES (exactly one is logged per auto-fit run):
AUTOFIT_DIT_RESIDENT_RE = re.compile(
    r"auto-fit:\s+DiT (?P<q>\S+)\s+[0-9.]+\s+GB fits\s+[0-9.]+\s+GB budget\s*->\s*resident"
)
AUTOFIT_DIT_DOWNGRADE_RE = re.compile(
    r"auto-fit:\s+DiT q8_0\s+[0-9.]+\s+GB\s*->\s*(?P<q>\S+)\s+[0-9.]+\s+GB to fit\s+"
    r"[0-9.]+\s+GB budget\s*\(resident\)"
)
AUTOFIT_DIT_OFFLOAD_RE = re.compile(
    r"auto-fit:\s+DiT does not fit\s+[0-9.]+\s+GB budget even at q4_K[^\n]*->\s*(?P<q>\S+),\s*will offload"
)
# Per-component placement (auto-allocate; also emitted under auto-fit since it implies it).
AUTOALLOC_PLACEMENT_RE = re.compile(
    r"auto-allocate:\s+'(?P<prefix>[^']+)'[^\n]*->\s*(?P<decision>RESIDENT|OFFLOAD)"
)
# Weight-prefix -> friendly component name used in the report.
_AUTO_COMPONENT = {
    "model.diffusion_model": "dit",
    "text_encoders": "te",
    "first_stage_model": "vae",
}

QWEN_FAMILIES = {"Qwen-Image", "Qwen-Image-Edit"}


def infer_model_family_from_index(model_path: str) -> str | None:
    index_path = Path(model_path) / "model_index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    class_name = data.get("_class_name")
    if class_name == "QwenImageEditPipeline":
        return "Qwen-Image-Edit"
    if class_name == "QwenImagePipeline":
        return "Qwen-Image"
    return None


def resolves_to_qwen_family(model_family: str | None, model_path: str) -> bool:
    if model_family in QWEN_FAMILIES:
        return True
    if model_family:
        return False
    return infer_model_family_from_index(model_path) in QWEN_FAMILIES


def edge_negative_prompt_default(
    model_family: str | None,
    model_path: str,
    cfg_scale: float,
    negative_prompt: str | None,
) -> str | None:
    if (
        negative_prompt is None
        and cfg_scale > 1.0
        and resolves_to_qwen_family(model_family, model_path)
    ):
        return " "
    return negative_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diffusion-model", default=None, help="standalone DiT transformer weights (distilled models); --model then points to the base")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="text-to-image", choices=["text-to-image", "image-editing"])
    parser.add_argument("--input-image", default=None)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--model-family", default=None)
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
    parser.add_argument("--qwen-image-zero-cond-t", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vae-tiling", choices=["on", "off", "auto"], default=None)
    parser.add_argument("--vae-tile-size")
    parser.add_argument("--tensor-type-rules")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--text-encoder-offload", action="store_true")
    parser.add_argument("--vae-offload", action="store_true")
    parser.add_argument("--dit-offload", action="store_true")
    parser.add_argument("--max-vram")
    parser.add_argument("--auto-fit", action="store_true",
                        help="engine picks DiT/TE quant + per-component placement (ignores --dtype)")
    parser.add_argument("--auto-allocate", action="store_true",
                        help="engine picks per-component resident/offload placement under --max-vram")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--profile-graph-cuts", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = output_dir / "inputs"
    prompt_dir.mkdir(exist_ok=True)
    prompt_file = prompt_dir / "prompt.txt"
    prompt_file.write_text(args.prompt + "\n", encoding="utf-8")

    sample_dir = output_dir / "samples" / "edge"
    repeat = args.warmup_runs + args.measured_runs
    negative_prompt = edge_negative_prompt_default(
        args.model_family,
        args.model,
        args.cfg_scale,
        args.negative_prompt,
    )
    command = [
        args.binary,
        "--model",
        args.model,
    ]
    if args.diffusion_model:
        command += ["--diffusion-model", args.diffusion_model]
    command += [
        "--prompt_file",
        str(prompt_file),
        "--output_dir",
        str(sample_dir),
        "--backend",
        args.backend,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num_steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--guidance_scale",
        str(args.guidance),
        *(["--flow_shift", str(args.flow_shift)] if args.flow_shift is not None else []),
        "--cfg_scale",
        str(args.cfg_scale),
        "--warmup",
        str(args.warmup_runs),
        "--repeat",
        str(repeat),
        "--start_index",
        "0",
        "--end_index",
        "1",
    ]
    if negative_prompt is not None:
        command.extend(["--negative_prompt", negative_prompt])
    if args.task == "image-editing":
        if not args.input_image:
            raise SystemExit("--input-image is required for image-editing")
        command.extend(["--image", args.input_image])
    if args.dtype and args.dtype != "auto":
        command.extend(["--type", args.dtype])
    if args.devices:
        command.extend(["--devices", args.devices])
    if args.sp_size:
        command.extend(["--sp-size", str(args.sp_size)])
    if args.cfg_parallel_size:
        command.extend(["--cfg-parallel-size", str(args.cfg_parallel_size)])
    if args.cache:
        command.extend(["--cache_method", args.cache])
    append_cache_options(command, args)
    if args.qwen_image_zero_cond_t:
        command.append("--qwen-image-zero-cond-t")
    if args.vae_tiling:
        command.extend(["--vae-tiling", args.vae_tiling])
    if args.vae_tile_size is not None:
        command.extend(["--vae-tile-size", str(args.vae_tile_size)])
    if args.tensor_type_rules:
        command.extend(["--tensor-type-rules", str(args.tensor_type_rules)])
    if args.offload_to_cpu:
        command.append("--offload-to-cpu")
    if args.text_encoder_offload:
        command.append("--text-encoder-offload")
    if args.vae_offload:
        command.append("--vae-offload")
    if args.dit_offload:
        command.append("--dit-offload")
    if args.max_vram:
        command.extend(["--max-vram", args.max_vram])
    if args.auto_fit:
        command.append("--auto-fit")
    if args.auto_allocate:
        command.append("--auto-allocate")
    if args.no_flash_attention:
        command.append("--no-flash-attention")
    if args.profile_graph_cuts:
        command.append("--profile-graph-cuts")

    (output_dir / "adapter_command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode

    warmup_ms, measured_ms = parse_ed_sample_stdout(completed.stdout)
    if len(measured_ms) != args.measured_runs:
        print(
            f"expected {args.measured_runs} measured samples, got {len(measured_ms)}",
            file=sys.stderr,
        )
        return 3

    load_ms = load_ms_from_ed_sample(sample_dir / "timing.json")
    component_ms, stage_boundaries = parse_phase_markers(completed.stdout)
    if component_ms.get("dit") is not None and args.steps > 0:
        component_ms["per_step_avg"] = component_ms["dit"] / args.steps
    else:
        component_ms["per_step_avg"] = (
            (sum(measured_ms) / len(measured_ms) / args.steps) if measured_ms else None
        )
    metrics = {
        "schema_version": 1,
        "metric_source": "edge_dit_ed_sample",
        "measurement_boundary": "load_once_e2e_generation_no_output_encoding",
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "measured_ms": measured_ms,
        "cache_events": parse_cache_events(completed.stdout + "\n" + completed.stderr),
        "component_ms": component_ms,
        "stage_boundaries": stage_boundaries,
        "actual_placement": parse_auto_placement(completed.stderr),
        "sample_output_dir": str(sample_dir),
    }
    (output_dir / "runner_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def parse_phase_markers(stdout: str) -> tuple[dict[str, float | None], dict[str, list[float]]]:
    """Parse `[[phase]] stage=.. event=.. t=..` markers from ed-sample stdout.

    ed-sample runs warmup+repeat passes, so markers repeat. We take the LAST
    complete begin/end pair per stage (the final measured pass) for both the
    component latency and the [begin, end] epoch window used to segment the
    external gpu_monitor CSV.
    """
    latest: dict[str, dict[str, float]] = {}
    for match in PHASE_RE.finditer(stdout):
        stage = match.group("stage")
        event = match.group("event")
        t = float(match.group("t"))
        latest.setdefault(stage, {})[event] = t

    stage_map = (("encode", "text_encoder"), ("denoise", "dit"), ("decode", "vae"))
    component: dict[str, float | None] = {"text_encoder": None, "dit": None, "vae": None}
    boundaries: dict[str, list[float]] = {}
    for stage, name in stage_map:
        events = latest.get(stage, {})
        begin = events.get("begin")
        end = events.get("end")
        if begin is not None and end is not None:
            component[name] = (end - begin) * 1000.0
            boundaries[name] = [begin, end]
    return component, boundaries


def parse_ed_sample_stdout(stdout: str) -> tuple[list[float], list[float]]:
    warmup_ms: list[float] = []
    measured_ms: list[float] = []
    for match in PASS_RE.finditer(stdout):
        value_ms = float(match.group(1)) * 1000.0
        if match.group("warmup"):
            warmup_ms.append(value_ms)
        else:
            measured_ms.append(value_ms)
    return warmup_ms, measured_ms


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


def parse_auto_placement(stderr: str) -> dict[str, object] | None:
    """Parse the engine's ACTUAL auto-fit / auto-allocate decisions from stderr.

    Under --auto-fit / --auto-allocate the engine ignores the requested --type and
    decides the real DiT/TE quantization tier and per-component resident/offload
    placement at load time (logged by src/core/runtime/model_runtime.cpp). We surface
    those so the report can show what actually ran instead of what the job requested.

    Returns None when no auto-* decision was logged (non-auto runs / manual fixed tier),
    which is the normal case and must not be treated as an error. Otherwise returns e.g.
        {"mode": "auto-fit", "dit_quant": "q4_k", "te_quant": "q8_0",
         "dit_placement": "resident", "te_placement": "offload",
         "vae_placement": "resident"}
    Any field the logs didn't reveal is left as None (quant fields are absent under
    plain --auto-allocate, which owns placement only, not quantization).
    """
    def norm_quant(q: str) -> str:
        # engine logs q8_0 / q4_K; jobs & config use lowercase q8_0 / q4_k tokens.
        return q.strip().lower()

    out: dict[str, object] = {
        "mode": None,
        "dit_quant": None,
        "te_quant": None,
        "dit_placement": None,
        "te_placement": None,
        "vae_placement": None,
    }
    saw_autofit = False
    saw_autoalloc = False

    # --- DiT quantization tier (exactly one ladder outcome fires per auto-fit run) --- #
    m = AUTOFIT_TE_QUANT_RE.search(stderr)
    if m:
        out["te_quant"] = norm_quant(m.group("q"))
        saw_autofit = True
    for rx in (AUTOFIT_DIT_RESIDENT_RE, AUTOFIT_DIT_DOWNGRADE_RE, AUTOFIT_DIT_OFFLOAD_RE):
        m = rx.search(stderr)
        if m:
            out["dit_quant"] = norm_quant(m.group("q"))
            saw_autofit = True
            break

    # --- per-component placement (last decision per component wins: a component can be
    # logged twice, e.g. "fits but would squeeze... -> OFFLOAD instead" then the final
    # "-> OFFLOAD+segment"; the last line is the authoritative placement). ------------- #
    for m in AUTOALLOC_PLACEMENT_RE.finditer(stderr):
        saw_autoalloc = True
        comp = _AUTO_COMPONENT.get(m.group("prefix"))
        if comp is not None:
            out[f"{comp}_placement"] = m.group("decision").lower()

    if not (saw_autofit or saw_autoalloc):
        return None
    out["mode"] = "auto-fit" if saw_autofit else "auto-allocate"
    return out


def load_ms_from_ed_sample(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    seconds = data.get("model_load_seconds")
    if isinstance(seconds, (int, float)):
        return float(seconds) * 1000.0
    return None


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


if __name__ == "__main__":
    raise SystemExit(main())
