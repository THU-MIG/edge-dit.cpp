"""edge-dit.cpp benchmark runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BenchmarkRunner, PreflightResult

QWEN_FAMILIES = {"Qwen-Image", "Qwen-Image-Edit"}


def edge_negative_prompt_default(
    model_family: str | None,
    cfg_scale: float,
    negative_prompt: str | None,
) -> str | None:
    if (
        negative_prompt is None
        and cfg_scale > 1.0
        and model_family in QWEN_FAMILIES
    ):
        return " "
    return negative_prompt


class EdgeDitRunner(BenchmarkRunner):
    def preflight(self) -> PreflightResult:
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        sample_binary = self.edge_sample_binary()
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        messages: list[str] = []
        metadata: dict[str, Any] = {}

        if binary is None or not binary.exists():
            messages.append(f"missing ed-cli binary: {binary}")
        else:
            metadata["binary"] = str(binary)
        if sample_binary is None or not sample_binary.exists():
            messages.append(f"missing ed-sample binary: {sample_binary}")
        else:
            metadata["sample_binary"] = str(sample_binary)

        if repo is None or not repo.exists():
            messages.append(f"missing edge-dit.cpp repository: {repo}")
        else:
            metadata["commit"] = self.git_commit(repo)
            metadata["dirty"] = self.git_dirty(repo)
            ggml = repo / "third_party" / "ggml"
            if ggml.exists():
                metadata["ggml_commit"] = self.git_commit(ggml)

        return PreflightResult(
            system_id=self.system_id,
            ok=not messages,
            messages=messages,
            metadata=metadata,
        )

    def requires_runner_metrics(self) -> bool:
        return True

    def edge_sample_binary(self, run_options: dict[str, Any] | None = None) -> Path | None:
        run_options = run_options or {}
        sample_override = run_options.get("edge_sample_ref") or run_options.get("edge_sample_binary_ref")
        if sample_override:
            return self.resolve_path(str(sample_override))
        sample_ref = self.system_config.get("sample_binary", {}).get("path_ref")
        sample_binary = self.resolve_path(sample_ref)
        if sample_binary is not None:
            return sample_binary
        binary = self.edge_cli_binary(run_options)
        if binary is None:
            return None
        return binary.parent / "ed-sample"

    def edge_cli_binary(self, run_options: dict[str, Any] | None = None) -> Path | None:
        run_options = run_options or {}
        binary_override = run_options.get("edge_cli_ref") or run_options.get("edge_binary_ref")
        if binary_override:
            return self.resolve_path(str(binary_override))
        return self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))

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
        sample_binary = self.edge_sample_binary(run_options)
        # Distilled transformer-only models: --model points to the base (text_encoder/vae/
        # scheduler), --diffusion-model points to the standalone transformer weights.
        base_ref = workload["model"].get("base_model_ref")
        if base_ref:
            model_ref = base_ref
            diffusion_ref = workload["model"]["local_path_ref"]
        else:
            model_ref = workload["model"]["local_path_ref"]
            diffusion_ref = None
        model_path = self.resolve_path(model_ref)
        diffusion_path = self.resolve_path(diffusion_ref) if diffusion_ref else None
        if sample_binary is None:
            raise NotImplementedError("edge-dit path references are not resolved")
        if model_path is None or not model_path.exists():
            raise NotImplementedError(f"missing model path for {model_ref}: {model_path}")

        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = self.prompt_text(workload, run_options)
        negative_prompt = edge_negative_prompt_default(
            workload.get("model_family"),
            float(generation.get("cfg_scale", 1.0)),
            self.negative_prompt_text(workload, run_options),
        )
        command = [
            "python3",
            str(self.repo_root / "benchmark" / "scripts" / "run_edge_e2e.py"),
            "--binary",
            str(sample_binary),
            "--model",
            str(model_path),
            *( ["--diffusion-model", str(diffusion_path)] if diffusion_path else [] ),
            "--prompt",
            prompt,
            "--output-dir",
            str(output_dir.resolve()),
            "--task",
            workload["task"],
            "--model-family",
            workload["model_family"],
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
            str(generation.get("precision", "auto")),
            "--backend",
            "cuda",
            "--warmup-runs",
            str(warmup_runs),
            "--measured-runs",
            str(measured_runs),
        ]
        if negative_prompt is not None:
            command.extend(["--negative-prompt", negative_prompt])
        if workload["task"] == "image-editing":
            input_ref = workload.get("input_image_ref")
            input_path = self.resolve_path(input_ref)
            if input_path is None or not input_path.exists():
                raise NotImplementedError(f"missing image editing input path for {input_ref}: {input_path}")
            command.extend(["--input-image", str(input_path)])
        if workload["task"] == "text-to-video":
            command.extend(["--frames", str(generation.get("frames", 1))])
            if generation.get("fps") is not None:
                command.extend(["--fps", str(generation["fps"])])
        self.apply_edge_wrapper_options(command, workload.get("model_options", {}))
        self.apply_edge_wrapper_options(command, run_options)
        if generation.get("flow_shift") is not None:
            command.extend(["--flow-shift", str(generation["flow_shift"])])
        if gpu_count > 1:
            devices = edge_device_csv(gpu_count)
            command.extend(["--devices", devices])
            if parallel_mode == "sequence":
                command.extend(["--sp-size", str(gpu_count)])
            elif parallel_mode == "cfg":
                command.extend(["--cfg-parallel-size", str(gpu_count)])
        return command

    def apply_edge_wrapper_options(self, command: list[str], options: dict[str, Any]) -> None:
        if options.get("qwen_image_zero_cond_t"):
            command.append("--qwen-image-zero-cond-t")
        vae_tiling = options.get("vae_tiling")
        if vae_tiling is not None:
            # ed-cli/ed-sample --vae-tiling takes a value (on|off|auto), not a bare flag.
            command.extend(["--vae-tiling", "on" if vae_tiling else "off"])
        if options.get("vae_tile_size") is not None:
            command.extend(["--vae-tile-size", str(options["vae_tile_size"])])
        if options.get("tensor_type_rules"):
            command.extend(["--tensor-type-rules", str(options["tensor_type_rules"])])
        if options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if options.get("no_t5"):
            command.append("--no-t5")
        if options.get("text_encoder_offload"):
            command.append("--text-encoder-offload")
        if options.get("vae_offload"):
            command.append("--vae-offload")
        if options.get("dit_offload"):
            command.append("--dit-offload")
        if options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(options["max_vram_gib"])])
        if options.get("auto_fit"):
            command.append("--auto-fit")
        elif options.get("auto_allocate"):
            command.append("--auto-allocate")
        if options.get("flash_attention") is False:
            command.append("--no-flash-attention")
        if options.get("profile_graph_cuts"):
            command.append("--profile-graph-cuts")
        cache = options.get("cache")
        if cache is not None and cache is not False:
            command.extend(["--cache", str(cache)])
        self.apply_cache_tuning_options(command, options)

    def apply_edge_options(self, command: list[str], options: dict[str, Any]) -> None:
        if options.get("qwen_image_zero_cond_t"):
            command.append("--qwen-image-zero-cond-t")
        vae_tiling = options.get("vae_tiling")
        if vae_tiling is not None:
            # ed-cli/ed-sample --vae-tiling takes a value (on|off|auto), not a bare flag.
            command.extend(["--vae-tiling", "on" if vae_tiling else "off"])
        if options.get("vae_tile_size") is not None:
            command.extend(["--vae-tile-size", str(options["vae_tile_size"])])
        if options.get("tensor_type_rules"):
            command.extend(["--tensor-type-rules", str(options["tensor_type_rules"])])
        if options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if options.get("no_t5"):
            command.append("--no-t5")
        if options.get("text_encoder_offload"):
            command.append("--text-encoder-offload")
        if options.get("vae_offload"):
            command.append("--vae-offload")
        if options.get("dit_offload"):
            command.append("--dit-offload")
        if options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(options["max_vram_gib"])])
        if options.get("auto_fit"):
            command.append("--auto-fit")
        elif options.get("auto_allocate"):
            command.append("--auto-allocate")
        if options.get("profile_graph_cuts"):
            command.append("--profile-graph-cuts")
        if options.get("flash_attention") is False:
            command.append("--no-flash-attention")
        cache = options.get("cache")
        if cache is not None and cache is not False:
            command.extend(["--cache", str(cache)])
        self.apply_cache_tuning_options(command, options)
        if options.get("precision") is not None:
            command.extend(["--type", str(options["precision"])])

    def apply_cache_tuning_options(self, command: list[str], options: dict[str, Any]) -> None:
        value_flags = [
            ("cache_threshold", "--cache-threshold"),
            ("cache_reuse_threshold", "--cache-threshold"),
            ("cache_start", "--cache-start"),
            ("cache_end", "--cache-end"),
            ("cache_error_decay", "--cache-error-decay"),
            ("cache_fn_blocks", "--cache-fn-blocks"),
            ("cache_bn_blocks", "--cache-bn-blocks"),
            ("cache_residual_threshold", "--cache-residual-threshold"),
            ("cache_residual_diff_threshold", "--cache-residual-threshold"),
            ("cache_max_accumulated_residual_diff", "--cache-max-accumulated-residual-diff"),
            ("cache_warmup_steps", "--cache-warmup-steps"),
            ("cache_max_warmup_steps", "--cache-warmup-steps"),
            ("cache_max_cached_steps", "--cache-max-cached-steps"),
            ("cache_max_continuous_cached_steps", "--cache-max-continuous-cached-steps"),
            ("cache_taylor_order", "--cache-taylor-order"),
            ("cache_taylorseer_n_derivatives", "--cache-taylor-order"),
            ("cache_taylor_skip", "--cache-taylor-skip"),
            ("cache_taylorseer_skip_interval", "--cache-taylor-skip"),
            ("cache_scm_mask", "--cache-scm-mask"),
            ("cache_calibrate", "--cache-calibrate"),
            ("cache_profile", "--cache-profile"),
        ]
        seen_flags: set[str] = set()
        for key, flag in value_flags:
            value = options.get(key)
            if value is not None and flag not in seen_flags:
                command.extend([flag, str(value)])
                seen_flags.add(flag)

        path_refs = [
            ("cache_calibrate_ref", "--cache-calibrate"),
            ("cache_profile_ref", "--cache-profile"),
        ]
        for key, flag in path_refs:
            ref = options.get(key)
            if ref is None or flag in seen_flags:
                continue
            path = self.resolve_path(str(ref))
            if path is None:
                raise NotImplementedError(f"cache path reference is not resolved: {ref}")
            command.extend([flag, str(path)])
            seen_flags.add(flag)

        bool_flags = [
            ("cache_relative_threshold", "--cache-relative-threshold"),
            ("cache_absolute_threshold", "--cache-absolute-threshold"),
            ("cache_no_reset_error", "--cache-no-reset-error"),
            ("cache_reset_error", "--cache-reset-error"),
            ("cache_static_scm", "--cache-static-scm"),
            ("cache_dynamic_scm", "--cache-dynamic-scm"),
        ]
        for key, flag in bool_flags:
            if options.get(key):
                command.append(flag)
def edge_device_csv(gpu_count: int) -> str:
    """Return physical device ids for edge's MPI launcher."""
    value = os.environ.get("BENCHMARK_CUDA_VISIBLE_DEVICES")
    if value:
        devices = [item.strip() for item in value.split(",") if item.strip()]
        if len(devices) < gpu_count:
            raise ValueError(
                "BENCHMARK_CUDA_VISIBLE_DEVICES must list at least "
                f"{gpu_count} devices for edge parallel execution"
            )
        return ",".join(devices[:gpu_count])
    return ",".join(str(i) for i in range(gpu_count))
