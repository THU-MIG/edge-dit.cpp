from __future__ import annotations

import io
import runpy
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "basic_txt2img.py"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH), run_name="__test__")

build_parser = SCRIPT_GLOBALS["build_parser"]
build_engine_config = SCRIPT_GLOBALS["build_engine_config"]
build_image_request = SCRIPT_GLOBALS["build_image_request"]
main = SCRIPT_GLOBALS["main"]
parse_args = SCRIPT_GLOBALS["parse_args"]

EXPECTED_HELP = textwrap.dedent(
    """\
    usage: basic_txt2img.py [-h] --model MODEL --prompt PROMPT
                            [--negative-prompt NEGATIVE_PROMPT] [--output OUTPUT]
                            [--backend BACKEND] [--width WIDTH] [--height HEIGHT]
                            [--steps STEPS] [--seed SEED] [--guidance GUIDANCE]
                            [--cfg-scale CFG_SCALE]
                            [--image-cfg-scale IMAGE_CFG_SCALE] [--eta ETA]
                            [--flow-shift FLOW_SHIFT] [--sampler SAMPLER]
                            [--scheduler SCHEDULER] [--cache-mode CACHE_MODE]
                            [--cache-reuse-threshold CACHE_REUSE_THRESHOLD]
                            [--cache-start-percent CACHE_START_PERCENT]
                            [--cache-end-percent CACHE_END_PERCENT]
                            [--cache-error-decay-rate CACHE_ERROR_DECAY_RATE]
                            [--cache-use-relative-threshold]
                            [--cache-use-absolute-threshold]
                            [--cache-reset-error-on-compute]
                            [--cache-no-reset-error-on-compute]
                            [--cache-fn-compute-blocks CACHE_FN_COMPUTE_BLOCKS]
                            [--cache-bn-compute-blocks CACHE_BN_COMPUTE_BLOCKS]
                            [--cache-residual-diff-threshold CACHE_RESIDUAL_DIFF_THRESHOLD]
                            [--cache-max-accumulated-residual-diff CACHE_MAX_ACCUMULATED_RESIDUAL_DIFF]
                            [--cache-max-warmup-steps CACHE_MAX_WARMUP_STEPS]
                            [--cache-max-cached-steps CACHE_MAX_CACHED_STEPS]
                            [--cache-max-continuous-cached-steps CACHE_MAX_CONTINUOUS_CACHED_STEPS]
                            [--cache-taylorseer-n-derivatives CACHE_TAYLORSEER_N_DERIVATIVES]
                            [--cache-taylorseer-skip-interval CACHE_TAYLORSEER_SKIP_INTERVAL]
                            [--cache-scm-mask CACHE_SCM_MASK]
                            [--cache-scm-policy-dynamic]
                            [--cache-scm-policy-static] [--threads THREADS]
                            [--weight-type WEIGHT_TYPE]
                            [--tensor-type-rules TENSOR_TYPE_RULES]
                            [--max-vram MAX_VRAM] [--offload-to-cpu]
                            [--keep-text-encoder-on-cpu] [--keep-vae-on-cpu]
                            [--flash-attention] [--no-flash-attention]

    Minimal edge-dit text-to-image example

    options:
      -h, --help            show this help message and exit
      --model MODEL         Path to a model or diffusers directory
      --prompt PROMPT       Prompt text
      --negative-prompt NEGATIVE_PROMPT
                            Negative prompt text
      --output OUTPUT       Output image path
      --backend BACKEND     Backend name, for example cuda or cpu
      --width WIDTH
      --height HEIGHT
      --steps STEPS
      --seed SEED
      --guidance GUIDANCE   Distilled guidance value
      --cfg-scale CFG_SCALE
                            CFG scale
      --image-cfg-scale IMAGE_CFG_SCALE
                            Image CFG scale
      --eta ETA             Sampler eta
      --flow-shift FLOW_SHIFT
                            Flow scheduler shift
      --sampler SAMPLER     Sampler name, for example auto or euler
      --scheduler SCHEDULER
                            Scheduler name, for example auto or karras
      --cache-mode CACHE_MODE
                            Cache mode, for example disabled
      --cache-reuse-threshold CACHE_REUSE_THRESHOLD
      --cache-start-percent CACHE_START_PERCENT
      --cache-end-percent CACHE_END_PERCENT
      --cache-error-decay-rate CACHE_ERROR_DECAY_RATE
      --cache-use-relative-threshold
      --cache-use-absolute-threshold
      --cache-reset-error-on-compute
      --cache-no-reset-error-on-compute
      --cache-fn-compute-blocks CACHE_FN_COMPUTE_BLOCKS
      --cache-bn-compute-blocks CACHE_BN_COMPUTE_BLOCKS
      --cache-residual-diff-threshold CACHE_RESIDUAL_DIFF_THRESHOLD
      --cache-max-accumulated-residual-diff CACHE_MAX_ACCUMULATED_RESIDUAL_DIFF
      --cache-max-warmup-steps CACHE_MAX_WARMUP_STEPS
      --cache-max-cached-steps CACHE_MAX_CACHED_STEPS
      --cache-max-continuous-cached-steps CACHE_MAX_CONTINUOUS_CACHED_STEPS
      --cache-taylorseer-n-derivatives CACHE_TAYLORSEER_N_DERIVATIVES
      --cache-taylorseer-skip-interval CACHE_TAYLORSEER_SKIP_INTERVAL
      --cache-scm-mask CACHE_SCM_MASK
      --cache-scm-policy-dynamic
      --cache-scm-policy-static
      --threads THREADS     CPU thread count
      --weight-type WEIGHT_TYPE
                            Weight type, for example auto or q4_k
      --tensor-type-rules TENSOR_TYPE_RULES
                            Tensor type rules string
      --max-vram MAX_VRAM   Maximum VRAM budget in GiB
      --offload-to-cpu      Keep model weights on CPU and copy them to GPU as
                            needed
      --keep-text-encoder-on-cpu
                            Keep text encoders on CPU
      --keep-vae-on-cpu     Keep VAE on CPU
      --flash-attention     Enable flash attention
      --no-flash-attention  Disable flash attention
    """
)


class BasicTxt2ImgExampleTests(unittest.TestCase):
    def test_help_output_snapshot(self) -> None:
        parser = build_parser()
        parser.prog = "basic_txt2img.py"
        self.assertEqual(parser.format_help(), EXPECTED_HELP)

    def test_parse_args_supports_verified_flux_flags(self) -> None:
        args = parse_args(
            [
                "--model",
                "/models/flux",
                "--prompt",
                "teapot",
                "--backend",
                "cuda",
                "--offload-to-cpu",
                "--keep-text-encoder-on-cpu",
                "--max-vram",
                "8",
                "--steps",
                "1",
                "--guidance",
                "3.5",
            ]
        )
        self.assertTrue(args.offload_to_cpu)
        self.assertTrue(args.keep_text_encoder_on_cpu)
        self.assertEqual(args.max_vram, 8.0)
        self.assertEqual(args.guidance, 3.5)

    def test_build_engine_config_maps_runtime_flags(self) -> None:
        args = parse_args(
            [
                "--model",
                "/models/flux",
                "--prompt",
                "teapot",
                "--backend",
                "cuda",
                "--offload-to-cpu",
                "--keep-text-encoder-on-cpu",
                "--keep-vae-on-cpu",
                "--max-vram",
                "8",
                "--weight-type",
                "auto",
                "--threads",
                "4",
            ]
        )
        config = build_engine_config(args)
        self.assertEqual(config.model_path, "/models/flux")
        self.assertEqual(config.backend, "cuda")
        self.assertTrue(config.offload_params_to_cpu)
        self.assertTrue(config.keep_text_encoder_on_cpu)
        self.assertTrue(config.keep_vae_on_cpu)
        self.assertEqual(config.max_vram_gb, 8.0)
        self.assertEqual(config.weight_type, "auto")
        self.assertEqual(config.n_threads, 4)

    def test_build_image_request_maps_generation_flags(self) -> None:
        args = parse_args(
            [
                "--model",
                "/models/flux",
                "--prompt",
                "teapot",
                "--negative-prompt",
                "blurry",
                "--width",
                "256",
                "--height",
                "320",
                "--steps",
                "2",
                "--seed",
                "42",
                "--guidance",
                "3.5",
                "--cfg-scale",
                "1.0",
                "--image-cfg-scale",
                "1.1",
                "--eta",
                "0.2",
                "--flow-shift",
                "1.15",
                "--sampler",
                "auto",
                "--scheduler",
                "karras",
                "--cache-mode",
                "disabled",
                "--cache-reuse-threshold",
                "0.8",
                "--cache-start-percent",
                "0.2",
                "--cache-end-percent",
                "0.9",
                "--cache-error-decay-rate",
                "0.5",
                "--cache-use-absolute-threshold",
                "--cache-no-reset-error-on-compute",
                "--cache-fn-compute-blocks",
                "12",
                "--cache-bn-compute-blocks",
                "3",
                "--cache-residual-diff-threshold",
                "0.07",
                "--cache-max-accumulated-residual-diff",
                "1.5",
                "--cache-max-warmup-steps",
                "6",
                "--cache-max-cached-steps",
                "20",
                "--cache-max-continuous-cached-steps",
                "4",
                "--cache-taylorseer-n-derivatives",
                "2",
                "--cache-taylorseer-skip-interval",
                "1",
                "--cache-scm-mask",
                "0011",
                "--cache-scm-policy-static",
            ]
        )
        request = build_image_request(args)
        self.assertEqual(request.prompt, "teapot")
        self.assertEqual(request.negative_prompt, "blurry")
        self.assertEqual(request.width, 256)
        self.assertEqual(request.height, 320)
        self.assertEqual(request.steps, 2)
        self.assertEqual(request.seed, 42)
        self.assertEqual(request.guidance, 3.5)
        self.assertEqual(request.cfg_scale, 1.0)
        self.assertEqual(request.image_cfg_scale, 1.1)
        self.assertEqual(request.eta, 0.2)
        self.assertEqual(request.flow_shift, 1.15)
        self.assertEqual(request.sampler, "auto")
        self.assertEqual(request.scheduler, "karras")
        self.assertEqual(request.cache_mode, "disabled")
        self.assertEqual(request.cache_reuse_threshold, 0.8)
        self.assertEqual(request.cache_start_percent, 0.2)
        self.assertEqual(request.cache_end_percent, 0.9)
        self.assertEqual(request.cache_error_decay_rate, 0.5)
        self.assertFalse(request.cache_use_relative_threshold)
        self.assertFalse(request.cache_reset_error_on_compute)
        self.assertEqual(request.cache_Fn_compute_blocks, 12)
        self.assertEqual(request.cache_Bn_compute_blocks, 3)
        self.assertEqual(request.cache_residual_diff_threshold, 0.07)
        self.assertEqual(request.cache_max_accumulated_residual_diff, 1.5)
        self.assertEqual(request.cache_max_warmup_steps, 6)
        self.assertEqual(request.cache_max_cached_steps, 20)
        self.assertEqual(request.cache_max_continuous_cached_steps, 4)
        self.assertEqual(request.cache_taylorseer_n_derivatives, 2)
        self.assertEqual(request.cache_taylorseer_skip_interval, 1)
        self.assertEqual(request.cache_scm_mask, "0011")
        self.assertFalse(request.cache_scm_policy_dynamic)
        self.assertIsNone(request.output_type)

    def test_main_runs_minimal_happy_path(self) -> None:
        args = parse_args(
            [
                "--model",
                "/models/flux",
                "--prompt",
                "teapot",
                "--output",
                "/tmp/test-basic-output.png",
            ]
        )
        captured: dict[str, object] = {}

        class FakeImage:
            def save(self, path: str) -> None:
                captured["saved_path"] = path

        class FakeEngine:
            def __init__(self, config) -> None:
                captured["config"] = config

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def generate_image(self, request):
                captured["request"] = request
                return [FakeImage()]

        stdout = io.StringIO()
        with patch.dict(main.__globals__, {"Engine": FakeEngine, "parse_args": lambda: args}):
            with redirect_stdout(stdout):
                result = main()

        self.assertEqual(result, 0)
        self.assertEqual(captured["saved_path"], "/tmp/test-basic-output.png")
        self.assertEqual(getattr(captured["config"], "model_path"), "/models/flux")
        self.assertEqual(getattr(captured["request"], "prompt"), "teapot")
        self.assertIn("saved image to /tmp/test-basic-output.png", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
