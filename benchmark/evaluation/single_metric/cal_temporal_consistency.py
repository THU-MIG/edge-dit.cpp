#!/usr/bin/env python3
"""Temporal consistency for generated videos: how smooth / flicker-free the
motion is, measured on ADJACENT frames within each video (no reference video).

For one video with frames f_0..f_{N-1}, we score every adjacent pair (f_t, f_{t+1}):
    lpips_t = LPIPS(f_t, f_{t+1})      # perceptual change between neighbours
    ssim_t  = SSIM(f_t, f_{t+1})       # structural similarity between neighbours

Per video we report:
    lpips_mean / lpips_std  (higher LPIPS = bigger inter-frame change;
                             higher std = uneven change = flicker)
    ssim_mean  / ssim_std   (higher SSIM = neighbours more alike = smoother)

Cross-video aggregation in the summary averages the per-video means, and also
surfaces the mean of per-video stds (flicker indicator). This is the lightweight
adjacent-frame consistency measure (no FVD / optical flow), appropriate for the
small per-config sample sizes in the cross-system matrix.

Reuses the exact frame decode + metric kernels from cal_lpips.py / cal_ssim.py
so scores are directly comparable to the paired-video metrics.

Usage:
    python evaluation/single_metric/cal_temporal_consistency.py \
        --video_dir /path/to/run/samples/<adapter> \
        --image_prefix measured_ \
        --video_exts avi \
        --output_json /path/to/temporal_consistency.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute adjacent-frame temporal consistency (LPIPS/SSIM) per video."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        required=True,
        help="Directory containing generated videos.",
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        default="video_",
        help="Video filename prefix, e.g. video_ / measured_ / output_.",
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        default="avi,mp4,mov,mkv,webm,m4v",
        help="Comma-separated video extensions.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Torch device for LPIPS."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone network.",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=16,
        help="Batch size for LPIPS forward over adjacent-frame pairs.",
    )
    parser.add_argument(
        "--max_items", type=int, default=None, help="Evaluate only first N videos."
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path for per-video stats and summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-video stats.",
    )
    return parser.parse_args()


def collect_files_by_index(
    root_dir: Path, file_prefix: str, exts: Sequence[str]
) -> Dict[int, Path]:
    pattern = re.compile(
        rf"^{re.escape(file_prefix)}(\d+)\.({'|'.join(re.escape(e) for e in exts)})$",
        re.IGNORECASE,
    )
    result: Dict[int, Path] = {}
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in result:
            raise ValueError(
                f"Duplicate video index {idx} under {root_dir}: {result[idx]} and {p}"
            )
        result[idx] = p
    if not result:
        raise FileNotFoundError(
            f"No matching videos under: {root_dir} for prefix '{file_prefix}' exts {list(exts)}"
        )
    return result


def maybe_mkdir_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def decode_frames_bgr(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def arrays_to_lpips_tensor(batch: np.ndarray, device: str) -> torch.Tensor:
    """RGB uint8 [B,H,W,C] -> LPIPS tensor [B,C,H,W] in [-1,1]. (from cal_lpips.py)"""
    if not batch.flags.c_contiguous:
        batch = np.ascontiguousarray(batch)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().float()
    tensor = tensor / 127.5 - 1.0
    return tensor.to(device, non_blocking=True)


@torch.inference_mode()
def lpips_pairs_batched(
    prev_frames_rgb: List[np.ndarray],
    next_frames_rgb: List[np.ndarray],
    loss_fn: torch.nn.Module,
    device: str,
    frame_batch_size: int,
) -> List[float]:
    values: List[float] = []
    for start in range(0, len(prev_frames_rgb), frame_batch_size):
        pb = np.ascontiguousarray(
            np.stack(prev_frames_rgb[start : start + frame_batch_size], axis=0)
        )
        nb = np.ascontiguousarray(
            np.stack(next_frames_rgb[start : start + frame_batch_size], axis=0)
        )
        out = loss_fn(arrays_to_lpips_tensor(pb, device), arrays_to_lpips_tensor(nb, device))
        out = out.reshape(out.shape[0], -1).mean(dim=1)
        values.extend(out.detach().cpu().numpy().astype(np.float64).tolist())
    return values


def compute_ssim_bgr(a: np.ndarray, b: np.ndarray) -> float:
    # Both frames from the same OpenCV decode (BGR); channel order consistent. (from cal_ssim.py)
    value = ssim(a, b, data_range=255, channel_axis=2)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def video_temporal_stats(
    frames_bgr: List[np.ndarray],
    loss_fn: torch.nn.Module,
    device: str,
    frame_batch_size: int,
) -> Dict[str, Any]:
    n = len(frames_bgr)
    if n < 2:
        return {
            "num_frames": n,
            "num_pairs": 0,
            "lpips_mean": None,
            "lpips_std": None,
            "ssim_mean": None,
            "ssim_std": None,
        }

    prev_rgb = [frames_bgr[t][..., ::-1] for t in range(n - 1)]
    next_rgb = [frames_bgr[t + 1][..., ::-1] for t in range(n - 1)]
    lpips_vals = lpips_pairs_batched(prev_rgb, next_rgb, loss_fn, device, frame_batch_size)

    ssim_vals = [compute_ssim_bgr(frames_bgr[t], frames_bgr[t + 1]) for t in range(n - 1)]

    return {
        "num_frames": n,
        "num_pairs": n - 1,
        "lpips_mean": float(statistics.mean(lpips_vals)),
        "lpips_std": float(statistics.pstdev(lpips_vals)) if len(lpips_vals) > 1 else 0.0,
        "ssim_mean": float(statistics.mean(ssim_vals)),
        "ssim_std": float(statistics.pstdev(ssim_vals)) if len(ssim_vals) > 1 else 0.0,
    }


def _agg(values: Sequence[float]) -> Dict[str, float] | None:
    xs = [v for v in values if isinstance(v, (int, float))]
    if not xs:
        return None
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def build_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if r.get("num_pairs", 0) > 0]
    summary: Dict[str, Any] = {
        "count_scored": len(scored),
        "count_total_videos": len(records),
    }
    if not scored:
        return summary
    # Cross-video aggregation of per-video means, plus mean of per-video stds (flicker).
    summary["lpips_adjacent"] = _agg([r["lpips_mean"] for r in scored])
    summary["ssim_adjacent"] = _agg([r["ssim_mean"] for r in scored])
    summary["lpips_flicker_std_mean"] = float(
        statistics.mean([r["lpips_std"] for r in scored])
    )
    summary["ssim_flicker_std_mean"] = float(
        statistics.mean([r["ssim_std"] for r in scored])
    )
    # Flat means for easy table consumption.
    summary["mean"] = summary["lpips_adjacent"]["mean"]  # primary = adjacent LPIPS
    return summary


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "row",
        "index",
        "video",
        "num_frames",
        "num_pairs",
        "lpips_mean",
        "lpips_std",
        "ssim_mean",
        "ssim_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})


def main() -> None:
    args = parse_args()

    args.video_dir = args.video_dir.expanduser().resolve()
    if not args.video_dir.is_dir():
        raise NotADirectoryError(f"video_dir does not exist: {args.video_dir}")

    exts = [
        e.strip().lstrip(".").lower() for e in args.video_exts.split(",") if e.strip()
    ]
    if not exts:
        raise ValueError("--video_exts is empty")

    prefix = args.image_prefix

    print(f"[Info] video_dir: {args.video_dir}")
    print(f"[Info] prefix: {prefix}")
    print(f"[Info] extensions: {exts}")
    print(f"[Info] device: {args.device}")
    print(f"[Info] lpips net: {args.net}")

    video_map = collect_files_by_index(args.video_dir, prefix, exts)
    indices = sorted(video_map)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max_items must be positive.")
        indices = indices[: args.max_items]
    print(f"[Info] videos found: {len(indices)}")

    loss_fn = lpips.LPIPS(net=args.net).to(args.device)
    loss_fn.eval()

    records: List[Dict[str, Any]] = []
    progress = tqdm(enumerate(indices), total=len(indices), desc="Temporal", unit="video")
    for row, idx in progress:
        video_path = video_map[idx]
        frames = decode_frames_bgr(video_path)
        stats = video_temporal_stats(frames, loss_fn, args.device, args.frame_batch_size)
        if stats["num_pairs"] == 0:
            print(f"[Warn] row {row}: video index {idx} has < 2 frames, skipped")
        record = {"row": row, "index": idx, "video": str(video_path), **stats}
        records.append(record)

    summary = build_summary(records)

    print("\n=== Temporal Consistency Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    maybe_mkdir_parent(args.output_json)
    maybe_mkdir_parent(args.output_csv)

    if args.output_csv is not None:
        write_csv(args.output_csv, records)
        print(f"[Info] per-item CSV written to: {args.output_csv}")

    if args.output_json is not None:
        payload = {
            "config": {
                "video_dir": str(args.video_dir),
                "image_prefix": prefix,
                "video_exts": exts,
                "device": args.device,
                "net": args.net,
                "frame_batch_size": args.frame_batch_size,
                "max_items": args.max_items,
            },
            "summary": summary,
            "records": records,
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Info] result JSON written to: {args.output_json}")


if __name__ == "__main__":
    main()
