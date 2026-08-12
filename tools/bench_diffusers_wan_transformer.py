import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.profiler
from diffusers import WanTransformer3DModel
from diffusers.models._modeling_parallel import ContextParallelConfig


def sync():
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["single", "ulysses"], default="single")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--latent-frames", type=int, default=10)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--fuse-qkv", action="store_true")
    parser.add_argument("--text-seq", type=int, default=512)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--profile-row-limit", type=int, default=30)
    parser.add_argument("--profile-sort", default="cuda_time_total")
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
    device = torch.device(f"cuda:{local_rank}")
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    t0 = time.perf_counter()
    model = WanTransformer3DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    if args.attention_backend:
        model.set_attention_backend(args.attention_backend)
    if args.fuse_qkv:
        model.fuse_qkv_projections()
    if distributed:
        model.enable_parallelism(config=ContextParallelConfig(ring_degree=1, ulysses_degree=world))
    t1 = time.perf_counter()

    latent_h = args.height // 8
    latent_w = args.width // 8
    g = torch.Generator(device=device).manual_seed(args.seed)
    hidden_states = torch.randn(
        (1, model.config.in_channels, args.latent_frames, latent_h, latent_w),
        device=device,
        dtype=dtype,
        generator=g,
    )
    prompt_embeds = torch.randn(
        (1, args.text_seq, model.config.text_dim),
        device=device,
        dtype=dtype,
        generator=g,
    )
    negative_prompt_embeds = torch.randn(
        (1, args.text_seq, model.config.text_dim),
        device=device,
        dtype=dtype,
        generator=g,
    )
    timestep = torch.full((1,), 999.0, device=device, dtype=torch.float32)

    def forward_pair():
        with torch.inference_mode():
            cond = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            uncond = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
            out = uncond + 5.0 * (cond - uncond)
        return out

    for _ in range(args.warmup):
        _ = forward_pair()
    sync()

    times = []
    total_start = time.perf_counter()
    for _ in range(args.steps):
        sync()
        start = time.perf_counter()
        out = forward_pair()
        sync()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    total_end = time.perf_counter()

    payload = {
        "width": args.width,
        "height": args.height,
        "latent_frames": args.latent_frames,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "seq": (args.latent_frames // model.config.patch_size[0])
        * (latent_h // model.config.patch_size[1])
        * (latent_w // model.config.patch_size[2]),
        "dtype": args.dtype,
        "param_dtype": str(next(model.parameters()).dtype),
        "attention_backend": args.attention_backend,
        "fuse_qkv": args.fuse_qkv,
        "load_s": t1 - t0,
        "steps": args.steps,
        "warmup": args.warmup,
        "pair_times_ms": times,
        "mean_pair_ms": sum(times) / len(times),
        "total_s": total_end - total_start,
        "checksum": float(out.float().mean().detach().cpu()),
    }
    print("DIFFUSERS_WAN_TRANSFORMER_PROFILE " + json.dumps(payload, sort_keys=True), flush=True)

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(args.profile_steps):
                sync()
                _ = forward_pair()
                sync()
                prof.step()
        events = list(prof.key_averages())

        def event_us(event, *names):
            for name in names:
                value = getattr(event, name, None)
                if value is not None:
                    return float(value)
            return 0.0

        def event_payload(event):
            return {
                "name": event.key,
                "count": int(getattr(event, "count", 0)),
                "cpu_total_ms": event_us(event, "cpu_time_total") / 1000.0,
                "cpu_self_ms": event_us(event, "self_cpu_time_total") / 1000.0,
                "cuda_total_ms": event_us(event, "cuda_time_total", "device_time_total") / 1000.0,
                "cuda_self_ms": event_us(event, "self_cuda_time_total", "self_device_time_total") / 1000.0,
            }

        top_events = sorted(
            events,
            key=lambda event: event_us(event, args.profile_sort, "cuda_time_total", "device_time_total"),
            reverse=True,
        )[: args.profile_row_limit]
        keywords = (
            "nccl",
            "all_to_all",
            "alltoall",
            "all_gather",
            "allgather",
            "send",
            "recv",
            "scaled_dot_product",
            "flash",
            "attention",
        )
        selected_events = [
            event
            for event in events
            if any(keyword in event.key.lower() for keyword in keywords)
        ]
        selected_events = sorted(
            selected_events,
            key=lambda event: event_us(event, "cuda_time_total", "device_time_total", "cpu_time_total"),
            reverse=True,
        )
        profile_payload = {
            "mode": args.mode,
            "rank": rank,
            "world": world,
            "steps": args.profile_steps,
            "sort": args.profile_sort,
            "top": [event_payload(event) for event in top_events],
            "selected": [event_payload(event) for event in selected_events[: args.profile_row_limit]],
        }
        print("DIFFUSERS_WAN_TRANSFORMER_TORCH_PROFILE " + json.dumps(profile_payload, sort_keys=True), flush=True)
        print(
            f"DIFFUSERS_WAN_TRANSFORMER_TORCH_PROFILE_TABLE rank={rank} world={world} sort={args.profile_sort}",
            flush=True,
        )
        print(prof.key_averages().table(sort_by=args.profile_sort, row_limit=args.profile_row_limit), flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
