from __future__ import annotations

import argparse

from edge_dit import Engine, EngineConfig, ImageRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal edge-dit text-to-image example")
    parser.add_argument("--model", required=True, help="Path to a model or diffusers directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt text")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--backend", default=None, help="Backend name, for example cuda or cpu")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--guidance", type=float, default=None, help="Distilled guidance value")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG scale")
    parser.add_argument("--image-cfg-scale", type=float, default=None, help="Image CFG scale")
    parser.add_argument("--eta", type=float, default=None, help="Sampler eta")
    parser.add_argument("--flow-shift", type=float, default=None, help="Flow scheduler shift")
    parser.add_argument("--sampler", default=None, help="Sampler name, for example auto or euler")
    parser.add_argument(
        "--scheduler",
        default=None,
        help="Scheduler name, for example auto or karras",
    )
    parser.add_argument("--cache-mode", default=None, help="Cache mode, for example disabled")
    parser.add_argument("--cache-reuse-threshold", type=float, default=None)
    parser.add_argument("--cache-start-percent", type=float, default=None)
    parser.add_argument("--cache-end-percent", type=float, default=None)
    parser.add_argument("--cache-error-decay-rate", type=float, default=None)
    parser.add_argument(
        "--cache-use-relative-threshold",
        dest="cache_use_relative_threshold",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--cache-use-absolute-threshold",
        dest="cache_use_relative_threshold",
        action="store_false",
    )
    parser.add_argument(
        "--cache-reset-error-on-compute",
        dest="cache_reset_error_on_compute",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--cache-no-reset-error-on-compute",
        dest="cache_reset_error_on_compute",
        action="store_false",
    )
    parser.add_argument("--cache-fn-compute-blocks", type=int, default=None)
    parser.add_argument("--cache-bn-compute-blocks", type=int, default=None)
    parser.add_argument("--cache-residual-diff-threshold", type=float, default=None)
    parser.add_argument("--cache-max-accumulated-residual-diff", type=float, default=None)
    parser.add_argument("--cache-max-warmup-steps", type=int, default=None)
    parser.add_argument("--cache-max-cached-steps", type=int, default=None)
    parser.add_argument("--cache-max-continuous-cached-steps", type=int, default=None)
    parser.add_argument("--cache-taylorseer-n-derivatives", type=int, default=None)
    parser.add_argument("--cache-taylorseer-skip-interval", type=int, default=None)
    parser.add_argument("--cache-scm-mask", default=None)
    parser.add_argument(
        "--cache-scm-policy-dynamic",
        dest="cache_scm_policy_dynamic",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--cache-scm-policy-static",
        dest="cache_scm_policy_dynamic",
        action="store_false",
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU thread count")
    parser.add_argument("--weight-type", default=None, help="Weight type, for example auto or q4_k")
    parser.add_argument("--tensor-type-rules", default=None, help="Tensor type rules string")
    parser.add_argument("--max-vram", type=float, default=None, help="Maximum VRAM budget in GiB")
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Keep model weights on CPU and copy them to GPU as needed",
    )
    parser.add_argument(
        "--keep-text-encoder-on-cpu",
        action="store_true",
        help="Keep text encoders on CPU",
    )
    parser.add_argument(
        "--keep-vae-on-cpu",
        action="store_true",
        help="Keep VAE on CPU",
    )
    parser.add_argument(
        "--flash-attention",
        dest="flash_attention",
        action="store_true",
        default=None,
        help="Enable flash attention",
    )
    parser.add_argument(
        "--no-flash-attention",
        dest="flash_attention",
        action="store_false",
        help="Disable flash attention",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_engine_config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model_path=args.model,
        backend=args.backend,
        n_threads=args.threads,
        weight_type=args.weight_type,
        tensor_type_rules=args.tensor_type_rules,
        offload_params_to_cpu=args.offload_to_cpu or None,
        keep_text_encoder_on_cpu=args.keep_text_encoder_on_cpu or None,
        keep_vae_on_cpu=args.keep_vae_on_cpu or None,
        flash_attention=args.flash_attention,
        max_vram_gb=args.max_vram,
    )


def build_image_request(args: argparse.Namespace) -> ImageRequest:
    return ImageRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
        cfg_scale=args.cfg_scale,
        image_cfg_scale=args.image_cfg_scale,
        eta=args.eta,
        flow_shift=args.flow_shift,
        sampler=args.sampler,
        scheduler=args.scheduler,
        cache_mode=args.cache_mode,
        cache_reuse_threshold=args.cache_reuse_threshold,
        cache_start_percent=args.cache_start_percent,
        cache_end_percent=args.cache_end_percent,
        cache_error_decay_rate=args.cache_error_decay_rate,
        cache_use_relative_threshold=args.cache_use_relative_threshold,
        cache_reset_error_on_compute=args.cache_reset_error_on_compute,
        cache_Fn_compute_blocks=args.cache_fn_compute_blocks,
        cache_Bn_compute_blocks=args.cache_bn_compute_blocks,
        cache_residual_diff_threshold=args.cache_residual_diff_threshold,
        cache_max_accumulated_residual_diff=args.cache_max_accumulated_residual_diff,
        cache_max_warmup_steps=args.cache_max_warmup_steps,
        cache_max_cached_steps=args.cache_max_cached_steps,
        cache_max_continuous_cached_steps=args.cache_max_continuous_cached_steps,
        cache_taylorseer_n_derivatives=args.cache_taylorseer_n_derivatives,
        cache_taylorseer_skip_interval=args.cache_taylorseer_skip_interval,
        cache_scm_mask=args.cache_scm_mask,
        cache_scm_policy_dynamic=args.cache_scm_policy_dynamic,
    )


def main() -> int:
    args = parse_args()

    config = build_engine_config(args)
    request = build_image_request(args)

    with Engine(config) as engine:
        images = engine.generate_image(request)
        images[0].save(args.output)

    print(f"saved image to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
