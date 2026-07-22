#!/usr/bin/env python3
"""Run Diffusers load-once e2e generation timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "f16", "fp32", "f32"], default="bf16")
    parser.add_argument("--task", default="text-to-image")
    parser.add_argument("--model-family", default="FLUX.1")
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--measured-runs", type=int, required=True)
    parser.add_argument(
        "--quant-weights",
        choices=["none", "qint8", "qint4", "qfloat8"],
        default="none",
        help="Optimum-Quanto weight qtype applied to quantized modules.",
    )
    parser.add_argument(
        "--quant-activations",
        choices=["none", "qint8", "qfloat8"],
        default="none",
        help="Optimum-Quanto activation qtype (none = 16-bit float activations).",
    )
    parser.add_argument(
        "--quant-modules",
        default="transformer",
        help="Comma-separated pipeline submodules to quantize (e.g. transformer,text_encoder_2).",
    )
    parser.add_argument(
        "--quant-exclude",
        default="proj_out",
        help="Comma-separated layer-name patterns to keep in full precision (empty to disable).",
    )
    parser.add_argument(
        "--input-image",
        default=None,
        help="Input/reference image path for image-editing tasks.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames to generate for text-to-video tasks.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second used when encoding the output AVI for video tasks.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Use accelerate model CPU offload instead of moving the whole pipeline to CUDA "
        "(needed for large edit/video models on 24G cards).",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling to cut the VAE decode memory peak.",
    )
    parser.add_argument(
        "--no-t5",
        action="store_true",
        help="SD3 only: skip loading the T5-XXL text encoder (text_encoder_3), "
        "keeping only the dual CLIP encoders. Cuts memory, degrades long-prompt adherence.",
    )
    args = parser.parse_args()

    supported_tasks = {"text-to-image", "image-editing", "text-to-video"}
    if args.task not in supported_tasks:
        raise SystemExit(f"Diffusers e2e runner does not support task {args.task!r}")

    import torch
    from diffusers import DiffusionPipeline, FluxPipeline

    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / "samples" / "diffusers"
    sample_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "fp32": torch.float32,
        "f32": torch.float32,
    }[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_cls = resolve_pipeline_cls(args.model_family, args.task)

    load_t0 = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from_pretrained_kwargs = {"torch_dtype": dtype}
    if args.no_t5:
        # SD3's T5-XXL (text_encoder_3) is optional; drop it to save ~9.5GB.
        from_pretrained_kwargs["text_encoder_3"] = None
        from_pretrained_kwargs["tokenizer_3"] = None
    pipe = pipeline_cls.from_pretrained(args.model, **from_pretrained_kwargs)
    quant_summary = apply_quantization(args, pipe)
    if args.offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    synchronize(device)
    load_ms = (time.perf_counter() - load_t0) * 1000.0
    if device == "cuda":
        alloc_bytes = torch.cuda.memory_allocated()
        peak_alloc_bytes = torch.cuda.max_memory_allocated()
        print(
            f"[diffusers-e2e] cuda_alloc_after_load {alloc_bytes / (1024 ** 3):.3f} GiB "
            f"peak {peak_alloc_bytes / (1024 ** 3):.3f} GiB",
            flush=True,
        )
    else:
        alloc_bytes = None
        peak_alloc_bytes = None

    warmup_ms: list[float] = []
    measured_ms: list[float] = []
    metadata: list[dict[str, Any]] = []
    component_runs: list[dict[str, float | None]] = []
    stage_boundaries: dict[str, list[float]] = {}
    total_runs = args.warmup_runs + args.measured_runs
    for index in range(total_runs):
        phase = "warmup" if index < args.warmup_runs else "measured"
        phase_index = index if phase == "warmup" else index - args.warmup_runs
        elapsed_ms, result, markers = run_generation(args, pipe, device)
        if phase == "warmup":
            warmup_ms.append(elapsed_ms)
        else:
            measured_ms.append(elapsed_ms)
            if args.task == "text-to-video":
                save_result_video(result, sample_dir / f"output_{phase_index:03d}.avi", args.fps)
            else:
                save_result_image(result, sample_dir / f"output_{phase_index:03d}.png")
            component, boundaries = components_from_markers(markers)
            component_runs.append(component)
            if boundaries:
                stage_boundaries = boundaries  # last measured run represents steady state
        metadata.append(
            {
                "phase": phase,
                "index": phase_index,
                "elapsed_ms": elapsed_ms,
                "seed": args.seed,
            }
        )
        print(f"[diffusers-e2e] {phase} {phase_index} {elapsed_ms / 1000.0:.3f}s", flush=True)

    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    component_ms = aggregate_components(component_runs)
    component_ms["per_step_avg"] = (
        (component_ms["dit"] / args.steps)
        if component_ms.get("dit") is not None and args.steps > 0
        else ((statistics.mean(measured_ms) / args.steps) if measured_ms else None)
    )
    metrics = {
        "schema_version": 1,
        "metric_source": "diffusers_pipeline",
        "measurement_boundary": "load_once_e2e_generation_no_output_encoding",
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "measured_ms": measured_ms,
        "component_ms": component_ms,
        "stage_boundaries": stage_boundaries,
        "quantization": quant_summary,
        "cuda_alloc_after_load_bytes": alloc_bytes,
        "peak_cuda_alloc_after_load_bytes": peak_alloc_bytes,
        "sample_output_dir": str(sample_dir),
    }
    (output_dir / "runner_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def resolve_pipeline_cls(model_family: str, task: str) -> Any:
    """Map a benchmark model_family/task pair to a concrete Diffusers pipeline class.

    Text-to-image keeps the historical FluxPipeline/DiffusionPipeline behaviour; the
    edit and video families use the dedicated pipelines confirmed available in
    diffusers 0.39 (FluxKontextPipeline / QwenImageEditPipeline / WanPipeline).
    """
    import diffusers

    if task == "image-editing":
        if model_family in ("FLUX.1-Kontext", "FLUX.1"):
            return diffusers.FluxKontextPipeline
        if model_family in ("Qwen-Image-Edit", "Qwen-Image"):
            return diffusers.QwenImageEditPipeline
        return diffusers.AutoPipelineForImage2Image
    if task == "text-to-video":
        if model_family == "Wan":
            return diffusers.WanPipeline
        return diffusers.DiffusionPipeline
    # text-to-image
    if model_family == "FLUX.1":
        return diffusers.FluxPipeline
    return diffusers.DiffusionPipeline


def apply_quantization(args: argparse.Namespace, pipe: Any) -> dict[str, Any]:
    """Quantize selected pipeline submodules in place with Optimum-Quanto.
    """
    if args.quant_weights == "none" and args.quant_activations == "none":
        return {"enabled": False, "weights": None, "activations": None, "modules": []}

    from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize
    from optimum.quanto.nn import QModuleMixin

    qmap = {"none": None, "qint8": qint8, "qint4": qint4, "qfloat8": qfloat8}
    weights = qmap[args.quant_weights]
    activations = qmap[args.quant_activations]
    exclude = [p for p in args.quant_exclude.split(",") if p] or None
    module_names = [m.strip() for m in args.quant_modules.split(",") if m.strip()]

    quantized: list[dict[str, Any]] = []
    for name in module_names:
        module = getattr(pipe, name, None)
        if module is None:
            print(f"[diffusers-e2e] quantize skip: pipeline has no '{name}'", flush=True)
            continue
        # proj_out only exists on the transformer; drop unknown excludes silently.
        quantize(module, weights=weights, activations=activations, exclude=exclude)
        freeze(module)
        # Inspect the resulting QModules to prove weight/activation qtypes stuck.
        n_q = 0
        w_qtypes: set[str] = set()
        a_qtypes: set[str] = set()
        for sub in module.modules():
            if isinstance(sub, QModuleMixin):
                n_q += 1
                if activations is not None:
                    sub.disable_output_quantization()
                if getattr(sub, "weight_qtype", None) is not None:
                    w_qtypes.add(str(sub.weight_qtype))
                a_qtypes.add(str(getattr(sub, "activation_qtype", None)))
        summary = {
            "module": name,
            "quantized_layers": n_q,
            "weight_qtypes": sorted(w_qtypes),
            "activation_qtypes": sorted(a_qtypes),
        }
        quantized.append(summary)
        print(
            f"[diffusers-e2e] quantized {name}: {n_q} layers "
            f"weights={sorted(w_qtypes)} activations={sorted(a_qtypes)}",
            flush=True,
        )

    return {
        "enabled": True,
        "weights": args.quant_weights,
        "activations": args.quant_activations,
        "exclude": exclude,
        "modules": quantized,
    }


def run_generation(args: argparse.Namespace, pipe: Any, device: str) -> tuple[float, Any, dict[str, float]]:
    import torch

    generator = torch.Generator(device=device).manual_seed(args.seed) if device == "cuda" else None

    markers: dict[str, float] = {}

    def emit(stage: str, event: str) -> None:
        synchronize(device)
        now = time.time()
        markers[f"{stage}_{event}"] = now
        print(f"[[phase]] stage={stage} event={event} t={now:.6f}", flush=True)

    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    vae = getattr(pipe, "vae", None)

    state = {"denoise_started": False}

    handles = []
    if transformer is not None:
        def dit_pre_hook(module, inp):
            if not state["denoise_started"]:
                # encode ends when the first denoise step begins
                emit("encode", "end")
                emit("denoise", "begin")
                state["denoise_started"] = True
            return None

        handles.append(transformer.register_forward_pre_hook(dit_pre_hook))

    orig_vae_decode = None
    if vae is not None and hasattr(vae, "decode"):
        orig_vae_decode = vae.decode

        def wrapped_decode(*d_args, **d_kwargs):
            # denoise ends at the first vae.decode entry
            emit("denoise", "end")
            emit("decode", "begin")
            result = orig_vae_decode(*d_args, **d_kwargs)
            emit("decode", "end")
            return result

        vae.decode = wrapped_decode

    synchronize(device)
    start = time.perf_counter()
    emit("encode", "begin")
    call_kwargs = build_call_kwargs(args, generator)
    try:
        result = pipe(**call_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        if orig_vae_decode is not None and vae is not None:
            vae.decode = orig_vae_decode
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result, markers


def build_call_kwargs(args: argparse.Namespace, generator: Any) -> dict[str, Any]:
    """Assemble pipeline __call__ kwargs for the requested task.

    All three task families share prompt/steps/guidance/generator; edit adds the
    reference image and video swaps width/height for num_frames output.
    """
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": generator,
    }
    if args.task == "text-to-video":
        kwargs["width"] = args.width
        kwargs["height"] = args.height
        kwargs["num_frames"] = args.frames
        return kwargs
    if args.task == "image-editing":
        from diffusers.utils import load_image

        if not args.input_image:
            raise SystemExit("image-editing task requires --input-image")
        kwargs["image"] = load_image(args.input_image)
        kwargs["width"] = args.width
        kwargs["height"] = args.height
        return kwargs
    # text-to-image
    kwargs["width"] = args.width
    kwargs["height"] = args.height
    return kwargs


def components_from_markers(markers: dict[str, float]) -> tuple[dict[str, float | None], dict[str, list[float]]]:
    """Turn epoch markers into per-component ms plus [begin, end] boundaries."""

    def span_ms(begin_key: str, end_key: str) -> float | None:
        begin = markers.get(begin_key)
        end = markers.get(end_key)
        if begin is None or end is None:
            return None
        return (end - begin) * 1000.0

    component = {
        "text_encoder": span_ms("encode_begin", "encode_end"),
        "dit": span_ms("denoise_begin", "denoise_end"),
        "vae": span_ms("decode_begin", "decode_end"),
    }
    boundaries: dict[str, list[float]] = {}
    for stage, name in (("encode", "text_encoder"), ("denoise", "dit"), ("decode", "vae")):
        begin = markers.get(f"{stage}_begin")
        end = markers.get(f"{stage}_end")
        if begin is not None and end is not None:
            boundaries[name] = [begin, end]
    return component, boundaries


def aggregate_components(runs: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = ["text_encoder", "dit", "vae"]
    result: dict[str, float | None] = {key: None for key in keys}
    for key in keys:
        values = [run[key] for run in runs if run.get(key) is not None]
        if values:
            result[key] = statistics.mean(values)
    return result



def save_result_image(result: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "images"):
        result.images[0].save(output)
        return
    if hasattr(result, "frames"):
        frames = result.frames[0] if result.frames and isinstance(result.frames[0], list) else result.frames
        frames[0].save(output)
        return
    raise RuntimeError("Diffusers pipeline result has neither images nor frames")


def save_result_video(result: Any, output: Path, fps: int) -> None:
    """Write generated frames to an AVI (MJPG) file via OpenCV.

    Wan/video pipelines return result.frames as a list (per batch) of frames that
    are PIL images or HxWxC arrays; normalize both to uint8 BGR for cv2.
    """
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(result, "frames"):
        raise RuntimeError("Diffusers video pipeline result has no frames")
    raw = result.frames
    # result.frames is typically batched: a numpy array (batch, frames, H, W, C)
    # or a list whose first element is the per-batch frame list. Peel one batch level.
    if isinstance(raw, np.ndarray):
        frames = list(raw[0]) if raw.ndim == 5 else list(raw)
    elif isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple, np.ndarray)):
        first = raw[0]
        frames = list(first) if isinstance(first, (list, tuple)) or getattr(first, "ndim", 0) == 4 else list(raw)
    else:
        frames = list(raw)
    if len(frames) == 0:
        raise RuntimeError("Diffusers video pipeline produced zero frames")

    def to_uint8_rgb(frame: Any) -> "np.ndarray":
        if hasattr(frame, "save"):  # PIL image
            arr = np.asarray(frame)
        else:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    import cv2

    first = to_uint8_rgb(frames[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(max(1, fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open AVI writer for {output}")
    for frame in frames:
        rgb = to_uint8_rgb(frame)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def synchronize(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
