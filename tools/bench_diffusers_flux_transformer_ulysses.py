import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.profiler
from diffusers.models._modeling_parallel import ContextParallelConfig
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel


def sync():
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def make_flux_ids(img_seq, txt_seq, device):
    side = int(img_seq**0.5)
    if side * side != img_seq:
        raise ValueError(f"img_seq must be square for this benchmark, got {img_seq}")
    img_ids = torch.zeros((img_seq, 3), device=device, dtype=torch.float32)
    rows = torch.arange(side, device=device, dtype=torch.float32).repeat_interleave(side)
    cols = torch.arange(side, device=device, dtype=torch.float32).repeat(side)
    img_ids[:, 1] = rows
    img_ids[:, 2] = cols
    txt_ids = torch.zeros((txt_seq, 3), device=device, dtype=torch.float32)
    return img_ids, txt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["single", "ulysses"], default="single")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--img-seq", type=int, default=4096)
    parser.add_argument("--txt-seq", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=2)
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
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    t0 = time.perf_counter()
    model = FluxTransformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    if args.attention_backend:
        model.set_attention_backend(args.attention_backend)
    if distributed:
        model.enable_parallelism(config=ContextParallelConfig(ring_degree=1, ulysses_degree=world))
    t1 = time.perf_counter()

    g = torch.Generator(device=device).manual_seed(args.seed + rank)
    hidden_states = torch.randn((1, args.img_seq, model.config.in_channels), device=device, dtype=dtype, generator=g)
    encoder_hidden_states = torch.randn(
        (1, args.txt_seq, model.config.joint_attention_dim),
        device=device,
        dtype=dtype,
        generator=g,
    )
    pooled = torch.randn((1, model.config.pooled_projection_dim), device=device, dtype=dtype, generator=g)
    timestep = torch.ones((1,), device=device, dtype=dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype) if model.config.guidance_embeds else None
    img_ids, txt_ids = make_flux_ids(args.img_seq, args.txt_seq, device)

    def run_once():
        with torch.inference_mode():
            out = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]
        return out

    for _ in range(args.warmup):
        _ = run_once()
    sync()

    times = []
    total_start = time.perf_counter()
    for _ in range(args.steps):
        sync()
        start = time.perf_counter()
        out = run_once()
        sync()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    total_end = time.perf_counter()

    checksum = float(out.float().mean().detach().cpu())
    payload = {
        "mode": args.mode,
        "rank": rank,
        "world": world,
        "local_rank": local_rank,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "load_s": t1 - t0,
        "steps": args.steps,
        "warmup": args.warmup,
        "img_seq": args.img_seq,
        "txt_seq": args.txt_seq,
        "times_ms": times,
        "mean_ms": sum(times) / len(times),
        "total_s": total_end - total_start,
        "checksum": checksum,
    }
    print("DIFFUSERS_TRANSFORMER_PROFILE " + json.dumps(payload, sort_keys=True), flush=True)

    if args.profile:
        sync()
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for step in range(args.profile_steps):
                sync()
                with torch.profiler.record_function(f"diffusers_transformer_forward_{step}"):
                    _ = run_once()
                sync()
                prof.step()
        sync()

        events = list(prof.key_averages())

        def event_us(event, *names):
            for name in names:
                value = getattr(event, name, None)
                if value is not None:
                    return float(value)
            return 0.0

        def event_count(event):
            return int(getattr(event, "count", 0))

        def event_payload(event):
            return {
                "name": event.key,
                "count": event_count(event),
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
        print("DIFFUSERS_TRANSFORMER_TORCH_PROFILE " + json.dumps(profile_payload, sort_keys=True), flush=True)
        print(
            f"DIFFUSERS_TRANSFORMER_TORCH_PROFILE_TABLE rank={rank} world={world} sort={args.profile_sort}",
            flush=True,
        )
        print(prof.key_averages().table(sort_by=args.profile_sort, row_limit=args.profile_row_limit), flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
