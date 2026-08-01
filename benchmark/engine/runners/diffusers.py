"""Diffusers benchmark runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Any

from .base import BenchmarkRunner, PreflightResult


class DiffusersRunner(BenchmarkRunner):
    def preflight(self) -> PreflightResult:
        required = self.system_config.get("preflight", {}).get("require_python_packages", [])
        python = self.python_executable()
        messages: list[str] = []
        metadata: dict[str, Any] = {
            "python": python,
            "required_packages": required,
        }
        if self.configured_python_path() is not None and not Path(python).exists():
            messages.append(f"missing Diffusers Python: {python}")
            return PreflightResult(
                system_id=self.system_id,
                ok=False,
                messages=messages,
                metadata=metadata,
            )

        missing = [name for name in required if not self.package_importable_with_python(python, name)]
        messages.extend(f"missing Python package: {name}" for name in missing)
        if not missing:
            metadata["package_versions"] = self.package_versions(python, required)
        return PreflightResult(
            system_id=self.system_id,
            ok=not messages,
            messages=messages,
            metadata=metadata,
        )

    def configured_python_path(self) -> Path | None:
        ref = self.system_config.get("python", {}).get("path_ref")
        return self.resolve_path(ref)

    def python_executable(self) -> str:
        configured = self.configured_python_path()
        return str(configured) if configured is not None else "python3"

    def package_importable_with_python(self, python: str, module_name: str) -> bool:
        code = f"import {module_name}"
        return subprocess.run(
            [python, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0

    def package_versions(self, python: str, packages: list[str]) -> dict[str, str | None]:
        code = """
import importlib.metadata as metadata
import json
import sys
packages = sys.argv[1:]
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps(versions, sort_keys=True))
"""
        completed = subprocess.run(
            [python, "-c", code, *packages],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            return {}
        import json

        data = json.loads(completed.stdout)
        return data if isinstance(data, dict) else {}

    def environment_metadata(self) -> dict[str, Any]:
        data = super().environment_metadata()
        python = self.python_executable()
        packages = self.system_config.get("preflight", {}).get("require_python_packages", [])
        data.setdefault("software", {})["diffusers_python"] = {
            "executable": python,
            "package_versions": self.package_versions(python, packages),
        }
        return data

    def extra_env(self, gpu_count: int) -> dict[str, str]:
        existing = os.environ.get("PYTHONPATH", "")
        pythonpath = f"{self.repo_root}:{self.repo_root.parent}"
        if existing:
            pythonpath = f"{pythonpath}:{existing}"
        return {
            "PYTHONPATH": pythonpath,
        }

    def requires_runner_metrics(self) -> bool:
        return True

    def build_execution_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        run_options: dict[str, Any],
        output_dir: Path,
        warmup_runs: int,
        measured_runs: int,
    ) -> list[str]:
        if gpu_count != 1:
            raise NotImplementedError("Diffusers single-GPU benchmark only")
        supported_tasks = {"text-to-image", "image-editing", "text-to-video"}
        if workload["task"] not in supported_tasks:
            raise NotImplementedError(
                f"Diffusers load-once e2e adapter does not support task {workload['task']!r}"
            )
        model_ref = workload["model"]["local_path_ref"]
        model_path = self.resolve_path(model_ref)
        if model_path is None or not model_path.exists():
            raise NotImplementedError(f"missing Diffusers model path for {model_ref}: {model_path}")
        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = self.prompt_text(workload, run_options)
        command = [
            self.python_executable(),
            str(self.repo_root / "benchmark" / "scripts" / "run_diffusers_e2e.py"),
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--output-dir",
            str(output_dir.resolve()),
            "--width",
            str(generation["width"]),
            "--height",
            str(generation["height"]),
            "--steps",
            str(generation["steps"]),
            "--seed",
            str(generation["seed"]),
            "--guidance",
            str(generation["guidance"]),
            "--cfg-scale",
            str(generation.get("cfg_scale", 1.0)),
            "--dtype",
            str(generation["precision"]),
            "--task",
            workload["task"],
            "--model-family",
            workload["model_family"],
            "--warmup-runs",
            str(warmup_runs),
            "--measured-runs",
            str(measured_runs),
        ]
        if workload["task"] == "image-editing":
            input_path = self.resolve_path(workload.get("input_image_ref"))
            if input_path is None or not input_path.exists():
                raise NotImplementedError(
                    f"missing Diffusers edit input image for {workload.get('input_image_ref')!r}: {input_path}"
                )
            command.extend(["--input-image", str(input_path)])
        if workload["task"] == "text-to-video":
            command.extend(["--frames", str(generation.get("frames", 1))])
            if generation.get("fps") is not None:
                command.extend(["--fps", str(generation["fps"])])
        if run_options.get("offload_to_cpu"):
            command.append("--offload")
        if run_options.get("sequential_offload"):
            command.append("--sequential-offload")
        if run_options.get("vae_tiling"):
            command.append("--vae-tiling")
        if run_options.get("no_t5"):
            command.append("--no-t5")
        quant_flags = {
            "quant_weights": "--quant-weights",
            "quant_activations": "--quant-activations",
            "quant_modules": "--quant-modules",
            "quant_exclude": "--quant-exclude",
        }
        for option_key, flag in quant_flags.items():
            if option_key in run_options and run_options[option_key] is not None:
                command.extend([flag, str(run_options[option_key])])
        return command

    def build_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> list[str]:
        if gpu_count != 1:
            raise NotImplementedError("Diffusers single-GPU pilot only")
        model_ref = workload["model"]["local_path_ref"]
        model_path = self.resolve_path(model_ref)
        if model_path is None or not model_path.exists():
            raise NotImplementedError(f"missing Diffusers model path for {model_ref}: {model_path}")
        generation = workload["generation"]
        prompt = self.prompt_text(workload, run_options)
        return [
            self.python_executable(),
            "-m",
            "benchmark.runners.diffusers",
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--width",
            str(generation["width"]),
            "--height",
            str(generation["height"]),
            "--steps",
            str(generation["steps"]),
            "--seed",
            str(generation["seed"]),
            "--guidance",
            str(generation["guidance"]),
            "--dtype",
            str(generation["precision"]),
            "--output",
            "samples/output.avi" if workload["task"] == "text-to-video" else "samples/output.png",
            "--task",
            workload["task"],
            "--model-family",
            workload["model_family"],
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Diffusers FLUX pilot.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "f16", "fp32", "f32"], default="bf16")
    parser.add_argument("--output", default="samples/output.png")
    parser.add_argument("--task", default="text-to-image")
    parser.add_argument("--model-family", default="FLUX.1")
    args = parser.parse_args()

    import torch
    from diffusers import DiffusionPipeline, FluxPipeline

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "fp32": torch.float32,
        "f32": torch.float32,
    }[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_family == "FLUX.1" and args.task == "text-to-image":
        pipeline_cls = FluxPipeline
    else:
        pipeline_cls = DiffusionPipeline
    pipe = pipeline_cls.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "images"):
        result.images[0].save(output)
    elif hasattr(result, "frames"):
        frames = result.frames[0] if result.frames and isinstance(result.frames[0], list) else result.frames
        frames[0].save(output)
    else:
        raise RuntimeError("Diffusers pipeline result has neither images nor frames")


if __name__ == "__main__":
    main()
