import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import FluxPipeline
from diffusers.models._modeling_parallel import ContextParallelConfig


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--mode", choices=["single", "ulysses"], default="single")
    parser.add_argument("--output", default="/tmp/diffusers_flux.png")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    args = parser.parse_args()

    distributed = args.mode == "ulysses"
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    t0 = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(f"cuda:{local_rank}")
    pipe.set_progress_bar_config(disable=True)
    t1 = time.perf_counter()

    if distributed:
        cp_config = ContextParallelConfig(ring_degree=1, ulysses_degree=world)
        pipe.transformer.enable_parallelism(config=cp_config)

    transformer_times = []
    orig_forward = pipe.transformer.forward

    def timed_forward(*f_args, **f_kwargs):
        synchronize()
        s = time.perf_counter()
        out = orig_forward(*f_args, **f_kwargs)
        synchronize()
        e = time.perf_counter()
        transformer_times.append((e - s) * 1000.0)
        return out

    pipe.transformer.forward = timed_forward

    generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(args.seed)
    synchronize()
    infer_start = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        output_type="pil" if rank == 0 else "latent",
        max_sequence_length=args.max_sequence_length,
    )
    synchronize()
    infer_end = time.perf_counter()

    if rank == 0 and hasattr(result, "images") and result.images is not None and args.output:
        image = result.images[0]
        if hasattr(image, "save"):
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            image.save(args.output)

    payload = {
        "mode": args.mode,
        "rank": rank,
        "world": world,
        "local_rank": local_rank,
        "dtype": args.dtype,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "load_s": t1 - t0,
        "inference_s": infer_end - infer_start,
        "transformer_ms": transformer_times,
        "transformer_sum_ms": sum(transformer_times),
    }
    print("DIFFUSERS_PROFILE " + json.dumps(payload, sort_keys=True), flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
