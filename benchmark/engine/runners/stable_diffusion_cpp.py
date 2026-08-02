"""stable-diffusion.cpp benchmark runner."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from .base import BenchmarkRunner, PreflightResult


def _sdcpp_dtype(precision: str) -> str:
    """Normalize a precision label to stable-diffusion.cpp's --type spelling.
    sd.cpp K-quants use an uppercase K (q4_K, q2_K, ...); our method files use
    lowercase (q4_k) which edge-dit accepts but sd.cpp silently ignores (falls
    back to f16), so map the K-quant suffix to uppercase here."""
    import re
    return re.sub(r"^(q[2-8])_k$", r"\1_K", str(precision))


class StableDiffusionCppRunner(BenchmarkRunner):
    def preflight(self) -> PreflightResult:
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        e2e_binary = self.resolve_path(self.system_config.get("e2e_binary", {}).get("path_ref"))
        messages: list[str] = []
        metadata: dict[str, Any] = {
            "force_update_policy": self.system_config.get("repo", {}).get("commit_policy")
        }

        if repo is None or not repo.exists():
            messages.append(f"missing stable-diffusion.cpp repository: {repo}")
        else:
            metadata["commit"] = self.git_commit(repo)
            metadata["dirty"] = self.git_dirty(repo)
            # Warn (do not block) if the checked-out sd.cpp commit differs from the one
            # the benchmark was validated against. offload / --stream-layers / --max-vram
            # and component-loading behavior are version-specific, so a mismatch means the
            # results may not reproduce and the e2e wrapper may need re-validating.
            expected = self.system_config.get("repo", {}).get("expected_commit")
            actual = metadata["commit"]
            if expected and actual and not actual.startswith(str(expected)):
                metadata["commit_mismatch"] = f"expected {expected}, found {actual[:12]}"
                print(
                    f"[preflight] WARNING: stable-diffusion.cpp is at {actual[:12]}, "
                    f"benchmark validated on {expected}; results may not reproduce "
                    f"(offload/stream-layers/component-loading are version-specific).",
                    file=sys.stderr,
                )
            if (
                self.system_config.get("repo", {}).get("commit_policy")
                == "force_latest_origin_main"
                and metadata["dirty"]
            ):
                messages.append(
                    "stable-diffusion.cpp worktree is dirty; refusing destructive force update"
                )

        if binary is None or not binary.exists():
            metadata["binary_missing"] = str(binary)
        else:
            metadata["binary"] = str(binary)

        if e2e_binary is None or not e2e_binary.exists():
            messages.append(f"missing stable-diffusion.cpp e2e wrapper: {e2e_binary}")
        else:
            metadata["e2e_binary"] = str(e2e_binary)

        return PreflightResult(
            system_id=self.system_id,
            ok=not messages,
            messages=messages,
            metadata=metadata,
        )

    def prepare_for_execution(self, force_external_update: bool = False) -> list[str]:
        policy = self.system_config.get("repo", {}).get("commit_policy")
        if policy != "force_latest_origin_main":
            return []
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        if repo is None:
            raise RuntimeError("stable-diffusion.cpp repo path is not resolved")
        if not force_external_update:
            raise RuntimeError(
                "stable-diffusion.cpp requires --force-external-update before execution"
            )
        commands = self.system_config.get("preflight", {}).get("force_update_commands", [])
        executed = []
        for command in commands:
            subprocess.run(command.split(), cwd=repo, check=True)
            executed.append(command)
        return executed

    def requires_runner_metrics(self) -> bool:
        return True

    def build_execution_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        run_options: dict[str, Any],
        output_dir,
        warmup_runs: int,
        measured_runs: int,
    ) -> list[str]:
        if workload["task"] not in ("text-to-image", "image-editing", "text-to-video"):
            raise NotImplementedError(
                f"stable-diffusion.cpp load-once e2e wrapper does not support task {workload['task']!r}"
            )
        if gpu_count != 1:
            raise NotImplementedError("stable-diffusion.cpp is a single-GPU baseline")
        binary = self.resolve_path(self.system_config.get("e2e_binary", {}).get("path_ref"))
        model_ref = workload["model"]["local_path_ref"]
        model_path = self.resolve_path(model_ref)
        if binary is None:
            raise NotImplementedError("stable-diffusion.cpp e2e wrapper path is not resolved")
        if model_path is None or not model_path.exists():
            raise NotImplementedError(
                f"missing stable-diffusion.cpp model path for {model_ref}: {model_path}"
            )

        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = self.prompt_text(workload, run_options)
        cfg_scale, distilled_guidance = sd_cpp_guidance(workload, generation)
        command = [
            str(binary),
            "--output-dir",
            str(output_dir.resolve()),
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
            "--cfg-scale",
            str(cfg_scale),
            "--distilled-guidance",
            str(distilled_guidance),
            "--dtype",
            _sdcpp_dtype(generation.get("precision", "auto")),
            "--backend",
            "cuda",
            "--task",
            workload["task"],
            "--model-family",
            workload["model_family"],
            "--warmup-runs",
            str(warmup_runs),
            "--measured-runs",
            str(measured_runs),
            "--diffusion-fa",
        ]
        if workload["task"] == "text-to-video":
            command.extend(["--video-frames", str(generation.get("frames", 1))])
            if generation.get("fps") is not None:
                command.extend(["--fps", str(generation["fps"])])
        if workload["task"] == "image-editing":
            input_path = self.resolve_path(workload.get("input_image_ref"))
            if input_path is None or not input_path.exists():
                raise NotImplementedError(
                    f"missing stable-diffusion.cpp edit input image for "
                    f"{workload.get('input_image_ref')!r}: {input_path}"
                )
            # Kontext uses reference images (--ref-image); Qwen-Image-Edit uses --init-img.
            if workload["model_family"] in ("Qwen-Image-Edit", "Qwen-Image"):
                command.extend(["--init-img", str(input_path)])
            else:
                command.extend(["--ref-image", str(input_path)])
        command.extend(sd_cpp_component_args(workload, model_path, self.resolve_path))
        if run_options.get("no_t5"):
            # SD3 only: drop the T5-XXL text encoder to save memory (keep dual CLIP).
            if "--t5xxl" in command:
                idx = command.index("--t5xxl")
                del command[idx:idx + 2]
        self.apply_wrapper_options(command, workload.get("model_options", {}))
        self.apply_wrapper_options(command, run_options)
        # VAE tiling defaults ON for stable-diffusion.cpp: it decodes the whole latent
        # in one buffer, so a large VAE (flux/qwen ~6-8 GiB) OOMs at the decode step
        # even after DiT sampling succeeds. Tiling caps that. Inject once here (not in
        # apply_wrapper_options, which is called twice) unless a job forces it off with
        # `vae_tiling: no` or an explicit-yes already added the flag.
        vae_tiling_off = run_options.get("vae_tiling") is False or \
            workload.get("model_options", {}).get("vae_tiling") is False
        if not vae_tiling_off and "--vae-tiling" not in command:
            command.append("--vae-tiling")
        # flow_shift lives in the model yaml's generation block (not run_options), same
        # source the edge runner reads. Forward it so sd.cpp uses the configured shift
        # (e.g. SD3 3.0) instead of its built-in per-model default.
        if generation.get("flow_shift") is not None:
            command.extend(["--flow-shift", str(generation["flow_shift"])])
        return command

    def build_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> list[str]:
        if gpu_count != 1:
            raise NotImplementedError("stable-diffusion.cpp is a single-GPU baseline")
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        model_path = self.resolve_path(workload["model"]["local_path_ref"])
        if binary is None:
            raise NotImplementedError("stable-diffusion.cpp path references are not resolved")
        if model_path is None or not model_path.exists():
            raise NotImplementedError(
                f"missing stable-diffusion.cpp model path: {model_path}"
            )
        run_options = run_options or {}
        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = self.prompt_text(workload, run_options)
        command = [
            str(binary),
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
            "--output",
            "samples/output.avi" if workload["task"] == "text-to-video" else "samples/output.png",
        ]
        if workload["task"] == "text-to-video":
            command.extend(["--mode", "vid_gen", "--video-frames", str(generation.get("frames", 1))])
            if "fps" in generation:
                command.extend(["--fps", str(generation["fps"])])
        if workload["task"] == "image-editing":
            input_path = self.resolve_path(workload.get("input_image_ref"))
            if input_path is None:
                raise NotImplementedError("image editing input path reference is not resolved")
            command.extend(["--init-img", str(input_path)])
        if (
            generation.get("precision")
            and generation["precision"] != "auto"
            and run_options.get("precision") is None
        ):
            command.extend(["--type", _sdcpp_dtype(generation["precision"])])
        if run_options.get("vae_tiling"):
            command.append("--vae-tiling")
        if run_options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if run_options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(run_options["max_vram_gib"])])
        if run_options.get("flash_attention") is True:
            command.append("--fa")
        cache = run_options.get("cache")
        if cache is not None and cache is not False and cache != "off":
            command.extend(["--cache-mode", str(cache)])
        return command

    def apply_wrapper_options(self, command: list[str], options: dict[str, Any]) -> None:
        if options.get("flash_attention") is True:
            command.append("--flash-attn")
        if options.get("diffusion_flash_attention") is True:
            command.append("--diffusion-fa")
        if options.get("vae_tiling"):
            command.append("--vae-tiling")
        if options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(options["max_vram_gib"])])
            # stream-layers makes --max-vram actually cap weight residency (not just the
            # compute graph). Pairs with offload; no effect without --max-vram.
            if options.get("offload_to_cpu"):
                command.append("--stream-layers")
        if options.get("model_args"):
            command.extend(["--model-args", str(options["model_args"])])
        if options.get("sample_method"):
            command.extend(["--sample-method", str(options["sample_method"])])
        if options.get("scheduler"):
            command.extend(["--scheduler", str(options["scheduler"])])
        if options.get("flow_shift") is not None:
            command.extend(["--flow-shift", str(options["flow_shift"])])
        if options.get("qwen_image_layers") is not None:
            command.extend(["--qwen-image-layers", str(options["qwen_image_layers"])])


def sd_cpp_guidance(
    workload: dict[str, Any],
    generation: dict[str, Any],
) -> tuple[float, float]:
    guidance = float(generation.get("guidance", 3.5))
    cfg_scale = float(generation.get("cfg_scale", 1.0))
    # --cfg-scale is sd.cpp's true CFG (txt_cfg); --distilled-guidance is the
    # Flux-style guidance-embed input. cfg_scale always drives true CFG (1.0 for
    # Flux/distilled = single pass, >1 for SD3/Wan = real CFG); guidance always
    # feeds distilled guidance. Same mapping for every family, matching the edge runner.
    return cfg_scale, guidance


def sd_cpp_component_args(workload: dict[str, Any], model_path, resolve_path) -> list[str]:
    family = workload["model_family"]
    if family in ("FLUX.1", "FLUX.1-Kontext"):
        return existing_component_args(
            [
                ("--diffusion-model", first_existing(model_path, [
                    "flux1-kontext-dev.safetensors",
                    "flux1-dev.safetensors",
                    "flux1-schnell.safetensors",
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                ])),
                ("--vae", first_existing(model_path, [
                    "ae.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                ])),
                ("--clip-l", first_existing(model_path, [
                    "text_encoder/model.safetensors",
                ])),
                ("--t5xxl", first_existing(model_path, [
                    "text_encoder_2/model.safetensors.index.json",
                    "text_encoder_2/model.safetensors",
                ])),
            ]
        )
    if family in ("SD3", "SD3.5"):
        # Preferred path: an official all-in-one single-file checkpoint (transformer +
        # dual CLIP + T5 + VAE in one .safetensors, e.g. sd3_medium_incl_clips_t5xxlfp16).
        # sd.cpp reads it natively via -m alone; the diffusers-preconverted transformer
        # path (below) produces a blurry image, so prefer the official file when provided.
        single_file_ref = workload.get("model_options", {}).get(
            "stable_diffusion_cpp_single_file"
        )
        if single_file_ref:
            single_path = resolve_path(str(single_file_ref))
            return existing_component_args([("--model", single_path)])
        converted_ref = workload.get("model_options", {}).get(
            "stable_diffusion_cpp_transformer_ref"
        ) or "stable_diffusion_cpp_sd3_transformer"
        converted_path = resolve_path(str(converted_ref)) if converted_ref else None
        model_flag = "--model" if workload.get("model_options", {}).get(
            "stable_diffusion_cpp_transformer_ref"
        ) else "--diffusion-model"
        return existing_component_args(
            [
                (model_flag, converted_path or first_existing(model_path, [
                    "transformer/diffusion_pytorch_model.safetensors",
                    "transformer/diffusion_pytorch_model.fp16.safetensors",
                ])),
                ("--vae", first_existing(model_path, [
                    "vae/diffusion_pytorch_model.fp16.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                ])),
                ("--clip-l", first_existing(model_path, [
                    "text_encoder/model.fp16.safetensors",
                    "text_encoder/model.safetensors",
                ])),
                ("--clip-g", first_existing(model_path, [
                    "text_encoder_2/model.fp16.safetensors",
                    "text_encoder_2/model.safetensors",
                ])),
                ("--t5xxl", first_existing(model_path, [
                    "text_encoder_3/model.safetensors.index.fp16.json",
                    "text_encoder_3/model.safetensors.index.json",
                ])),
            ]
        )
    if family in ("Qwen-Image", "Qwen-Image-Edit"):
        return existing_component_args(
            [
                ("--diffusion-model", first_existing(model_path, [
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                ])),
                ("--vae", first_existing(model_path, [
                    "vae/diffusion_pytorch_model.safetensors",
                ])),
                ("--llm", first_existing(model_path, [
                    "text_encoder/model.safetensors.index.json",
                ])),
            ]
        )
    if family == "Wan":
        # Preferred: official Comfy-Org repackaged component files (DiT + wan VAE + umt5),
        # which sd.cpp reads natively (Version: Wan 2.x). The diffusers-layout fallback below
        # is NOT recognized by sd.cpp ("get sd version failed"), so Wan only runs on sd.cpp
        # when these official refs are provided.
        mo = workload.get("model_options", {})
        dit_ref = mo.get("stable_diffusion_cpp_wan_dit")
        vae_ref = mo.get("stable_diffusion_cpp_wan_vae")
        t5_ref = mo.get("stable_diffusion_cpp_wan_t5")
        if dit_ref and vae_ref and t5_ref:
            return existing_component_args([
                ("--diffusion-model", resolve_path(str(dit_ref))),
                ("--vae", resolve_path(str(vae_ref))),
                ("--t5xxl", resolve_path(str(t5_ref))),
            ])
        return existing_component_args(
            [
                ("--diffusion-model", first_existing(model_path, [
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                ])),
                ("--vae", first_existing(model_path, [
                    "vae/diffusion_pytorch_model.safetensors",
                ])),
                ("--t5xxl", first_existing(model_path, [
                    "text_encoder/model.safetensors.index.json",
                    "text_encoder/model.safetensors",
                ])),
            ]
        )
    raise NotImplementedError(f"stable-diffusion.cpp has no component mapping for {family}")


def first_existing(base, candidates: list[str]):
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def existing_component_args(items) -> list[str]:
    command: list[str] = []
    missing = []
    for flag, path in items:
        if path is None:
            missing.append(flag)
        else:
            command.extend([flag, str(path)])
    if missing:
        raise NotImplementedError(
            "missing stable-diffusion.cpp component path(s): " + ", ".join(missing)
        )
    return command
