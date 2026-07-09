#!/usr/bin/env python3
"""Compute LPIPS by recursively searching two directories and pairing images/videos by index.

Optimizations:
    1. Batch LPIPS forward for image mode.
    2. Batch LPIPS forward for video frames.
    3. Avoid one model forward per frame.
    4. Keep a single LPIPS model on one device.

Usage:
    python evaluation/cal_lpips.py \
        --ref_dir /path/to/reference_images \
        --target_dir /path/to/target_images \
        --batch_size 16

    python evaluation/cal_lpips.py \
        --ref_dir /path/to/reference_videos \
        --target_dir /path/to/target_videos \
        --video \
        --frame_batch_size 16
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
from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LPIPS by pairing images/videos from two directories."
    )
    parser.add_argument(
        "--ref_dir",
        type=Path,
        required=True,
        help="Reference image/video root directory.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        required=True,
        help="Target image/video root directory.",
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        default="img_",
        help="Image/video filename prefix, e.g. img_ or video_.",
    )
    parser.add_argument(
        "--image_exts",
        type=str,
        default="jpg,jpeg,png,webp,bmp",
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Compute LPIPS for paired videos instead of paired images.",
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        default="mp4,mov,mkv,avi,webm,m4v",
        help="Comma-separated video extensions used when --video is enabled.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail if any index exists only in one directory.",
    )
    parser.add_argument(
        "--strict_size",
        action="store_true",
        help="Fail if paired images/video frames have different sizes.",
    )
    parser.add_argument(
        "--resize_to_ref",
        action="store_true",
        help="Resize target image/video-frame to reference size when dimensions differ.",
    )
    parser.add_argument(
        "--strict_video_length",
        action="store_true",
        help="Fail if paired videos have different frame counts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone network.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for image-mode LPIPS forward.",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=16,
        help="Batch size for video-frame LPIPS forward.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path for per-item LPIPS and summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-item LPIPS.",
    )
    return parser.parse_args()


def collect_files_by_index(
    root_dir: Path,
    file_prefix: str,
    exts: Sequence[str],
    kind: str,
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
                f"Duplicate {kind} index {idx} found under {root_dir}: "
                f"{result[idx]} and {p}"
            )

        result[idx] = p

    if not result:
        raise FileNotFoundError(f"No matching {kind}s found under: {root_dir}")

    return result


def maybe_mkdir_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    if records:
        fieldnames = list(records[0].keys())
    else:
        fieldnames = ["row", "index", "reference_path", "target_path", "lpips"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})


def build_summary(
    values: Sequence[float],
    total_rows: int,
    skipped_rows: int,
) -> Dict[str, Any]:
    if not values:
        return {
            "count_scored": 0,
            "count_total_rows": total_rows,
            "count_skipped": skipped_rows,
        }

    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    median = (
        sorted_values[mid]
        if len(sorted_values) % 2 == 1
        else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    )

    return {
        "count_scored": len(values),
        "count_total_rows": total_rows,
        "count_skipped": skipped_rows,
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "median": float(median),
        "max": float(max(values)),
    }


def bicubic_resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BICUBIC)
    return int(getattr(Image, "BICUBIC", 3))


def load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def arrays_to_lpips_tensor(batch: np.ndarray, device: str) -> torch.Tensor:
    """Convert RGB uint8 batch [B, H, W, C] to LPIPS tensor [B, C, H, W] in [-1, 1]."""
    if not batch.flags.c_contiguous:
        batch = np.ascontiguousarray(batch)

    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().float()
    tensor = tensor / 127.5 - 1.0
    return tensor.to(device, non_blocking=True)


@torch.inference_mode()
def compute_lpips_batch(
    ref_batch: np.ndarray,
    tgt_batch: np.ndarray,
    loss_fn: torch.nn.Module,
    device: str,
) -> np.ndarray:
    ref_tensor = arrays_to_lpips_tensor(ref_batch, device)
    tgt_tensor = arrays_to_lpips_tensor(tgt_batch, device)

    values = loss_fn(ref_tensor, tgt_tensor)
    values = values.reshape(values.shape[0], -1).mean(dim=1)
    return values.detach().cpu().numpy().astype(np.float64)


def flush_image_batch(
    records: list[Dict[str, Any]],
    values: list[float],
    pending_records: list[Dict[str, Any]],
    pending_ref_imgs: list[np.ndarray],
    pending_tgt_imgs: list[np.ndarray],
    loss_fn: torch.nn.Module,
    device: str,
) -> None:
    if not pending_records:
        return

    ref_batch = np.stack(pending_ref_imgs, axis=0)
    tgt_batch = np.stack(pending_tgt_imgs, axis=0)
    batch_values = compute_lpips_batch(ref_batch, tgt_batch, loss_fn, device)

    for rec, value in zip(pending_records, batch_values):
        rec["lpips"] = float(value)
        records.append(rec)
        values.append(float(value))

    pending_records.clear()
    pending_ref_imgs.clear()
    pending_tgt_imgs.clear()


def compute_image_lpips_records(
    common_indices: Sequence[int],
    ref_map: Dict[int, Path],
    tgt_map: Dict[int, Path],
    loss_fn: torch.nn.Module,
    device: str,
    batch_size: int,
    strict_size: bool,
    resize_to_ref: bool,
) -> tuple[list[Dict[str, Any]], list[float], int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    records: list[Dict[str, Any]] = []
    values: list[float] = []
    skipped_rows = 0

    pending_records: list[Dict[str, Any]] = []
    pending_ref_imgs: list[np.ndarray] = []
    pending_tgt_imgs: list[np.ndarray] = []
    pending_shape: tuple[int, ...] | None = None

    progress = tqdm(
        enumerate(common_indices),
        total=len(common_indices),
        desc="LPIPS",
        unit="image",
    )

    for row, idx in progress:
        ref_path = ref_map[idx]
        tgt_path = tgt_map[idx]

        ref_img = load_rgb_uint8(ref_path)
        tgt_img = load_rgb_uint8(tgt_path)

        if ref_img.shape != tgt_img.shape:
            if strict_size:
                raise ValueError(
                    f"row {row}: size mismatch for index {idx}, "
                    f"ref={ref_img.shape}, tgt={tgt_img.shape}"
                )

            if resize_to_ref:
                with Image.open(tgt_path) as tmp:
                    tmp = tmp.convert("RGB").resize(
                        (ref_img.shape[1], ref_img.shape[0]),
                        bicubic_resample_filter(),
                    )
                    tgt_img = np.asarray(tmp, dtype=np.uint8)
            else:
                print(
                    f"[Warn] row {row}: size mismatch for index {idx}, skipped "
                    f"(ref={ref_img.shape}, tgt={tgt_img.shape})"
                )
                skipped_rows += 1
                continue

        if pending_shape is None:
            pending_shape = ref_img.shape

        if ref_img.shape != pending_shape or len(pending_records) >= batch_size:
            flush_image_batch(
                records=records,
                values=values,
                pending_records=pending_records,
                pending_ref_imgs=pending_ref_imgs,
                pending_tgt_imgs=pending_tgt_imgs,
                loss_fn=loss_fn,
                device=device,
            )
            pending_shape = ref_img.shape

        pending_records.append(
            {
                "row": row,
                "index": idx,
                "reference_path": str(ref_path),
                "target_path": str(tgt_path),
            }
        )
        pending_ref_imgs.append(ref_img)
        pending_tgt_imgs.append(tgt_img)

    flush_image_batch(
        records=records,
        values=values,
        pending_records=pending_records,
        pending_ref_imgs=pending_ref_imgs,
        pending_tgt_imgs=pending_tgt_imgs,
        loss_fn=loss_fn,
        device=device,
    )

    records.sort(key=lambda x: x["row"])
    return records, values, skipped_rows


def make_comparable_video_batch_rgb(
    ref_frames_bgr: list[np.ndarray],
    tgt_frames_bgr: list[np.ndarray],
    strict_size: bool,
    resize_to_ref: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    ref_valid: list[np.ndarray] = []
    tgt_valid: list[np.ndarray] = []
    skipped_frames = 0

    for ref_frame, tgt_frame in zip(ref_frames_bgr, tgt_frames_bgr):
        if ref_frame.shape != tgt_frame.shape:
            if strict_size:
                raise ValueError(
                    f"size mismatch, ref={ref_frame.shape}, tgt={tgt_frame.shape}"
                )

            if resize_to_ref:
                tgt_frame = cv2.resize(
                    tgt_frame,
                    (ref_frame.shape[1], ref_frame.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                skipped_frames += 1
                continue

        # LPIPS expects RGB. Convert BGR to RGB with slicing.
        ref_valid.append(ref_frame[..., ::-1])
        tgt_valid.append(tgt_frame[..., ::-1])

    if not ref_valid:
        return None, None, skipped_frames

    ref_batch = np.ascontiguousarray(np.stack(ref_valid, axis=0))
    tgt_batch = np.ascontiguousarray(np.stack(tgt_valid, axis=0))
    return ref_batch, tgt_batch, skipped_frames


@torch.inference_mode()
def compute_video_lpips_batched(
    ref_video_path: Path,
    tgt_video_path: Path,
    loss_fn: torch.nn.Module,
    device: str,
    strict_size: bool,
    resize_to_ref: bool,
    strict_video_length: bool,
    frame_batch_size: int,
) -> tuple[float, int, int, bool]:
    if frame_batch_size <= 0:
        raise ValueError(f"frame_batch_size must be positive, got {frame_batch_size}")

    ref_cap = cv2.VideoCapture(str(ref_video_path))
    tgt_cap = cv2.VideoCapture(str(tgt_video_path))

    if not ref_cap.isOpened():
        raise RuntimeError(f"Failed to open reference video: {ref_video_path}")
    if not tgt_cap.isOpened():
        raise RuntimeError(f"Failed to open target video: {tgt_video_path}")

    sum_lpips = 0.0
    compared_frames = 0
    skipped_frames = 0
    length_mismatch = False

    try:
        while True:
            ref_frames: list[np.ndarray] = []
            tgt_frames: list[np.ndarray] = []
            reached_end = False

            for _ in range(frame_batch_size):
                ret_ref, ref_frame = ref_cap.read()
                ret_tgt, tgt_frame = tgt_cap.read()

                if not ret_ref or not ret_tgt:
                    if ret_ref != ret_tgt:
                        length_mismatch = True
                    reached_end = True
                    break

                ref_frames.append(ref_frame)
                tgt_frames.append(tgt_frame)

            if not ref_frames:
                break

            ref_batch, tgt_batch, batch_skipped = make_comparable_video_batch_rgb(
                ref_frames_bgr=ref_frames,
                tgt_frames_bgr=tgt_frames,
                strict_size=strict_size,
                resize_to_ref=resize_to_ref,
            )
            skipped_frames += batch_skipped

            if ref_batch is not None and tgt_batch is not None:
                batch_values = compute_lpips_batch(
                    ref_batch=ref_batch,
                    tgt_batch=tgt_batch,
                    loss_fn=loss_fn,
                    device=device,
                )
                sum_lpips += float(batch_values.sum())
                compared_frames += int(batch_values.shape[0])

            if reached_end:
                break

    finally:
        ref_cap.release()
        tgt_cap.release()

    if length_mismatch and strict_video_length:
        raise ValueError(f"Video length mismatch: {ref_video_path} vs {tgt_video_path}")

    if compared_frames == 0:
        raise ValueError(
            f"No comparable frames found for {ref_video_path} and {tgt_video_path}"
        )

    return (
        float(sum_lpips / compared_frames),
        compared_frames,
        skipped_frames,
        length_mismatch,
    )


def compute_video_lpips_records(
    common_indices: Sequence[int],
    ref_map: Dict[int, Path],
    tgt_map: Dict[int, Path],
    loss_fn: torch.nn.Module,
    device: str,
    strict_size: bool,
    resize_to_ref: bool,
    strict_video_length: bool,
    frame_batch_size: int,
) -> tuple[list[Dict[str, Any]], list[float], int]:
    records: list[Dict[str, Any]] = []
    values: list[float] = []
    skipped_rows = 0

    progress = tqdm(
        enumerate(common_indices),
        total=len(common_indices),
        desc="LPIPS",
        unit="video",
    )

    for row, idx in progress:
        ref_path = ref_map[idx]
        tgt_path = tgt_map[idx]

        value, compared_frames, skipped_frames, length_mismatch = (
            compute_video_lpips_batched(
                ref_video_path=ref_path,
                tgt_video_path=tgt_path,
                loss_fn=loss_fn,
                device=device,
                strict_size=strict_size,
                resize_to_ref=resize_to_ref,
                strict_video_length=strict_video_length,
                frame_batch_size=frame_batch_size,
            )
        )

        if length_mismatch:
            print(
                f"[Warn] row {row}: video length mismatch for index {idx}, "
                f"computed on overlapped prefix frames"
            )

        record = {
            "row": row,
            "index": idx,
            "reference_path": str(ref_path),
            "target_path": str(tgt_path),
            "compared_frames": compared_frames,
            "skipped_frames": skipped_frames,
            "lpips": value,
        }

        records.append(record)
        values.append(value)

    return records, values, skipped_rows


def main() -> None:
    args = parse_args()

    args.ref_dir = args.ref_dir.expanduser().resolve()
    if not args.ref_dir.is_dir():
        raise NotADirectoryError(f"ref_dir does not exist: {args.ref_dir}")

    args.target_dir = args.target_dir.expanduser().resolve()
    if not args.target_dir.is_dir():
        raise NotADirectoryError(f"target_dir does not exist: {args.target_dir}")

    image_exts = [
        e.strip().lstrip(".").lower() for e in args.image_exts.split(",") if e.strip()
    ]
    video_exts = [
        e.strip().lstrip(".").lower() for e in args.video_exts.split(",") if e.strip()
    ]

    if args.video:
        exts = video_exts
        if args.image_prefix == "img_":
            args.image_prefix = "video_"
    else:
        exts = image_exts

    if not exts:
        raise ValueError("No valid extensions found for current mode")

    print(f"[Info] reference_dir: {args.ref_dir}")
    print(f"[Info] target_dir: {args.target_dir}")
    print(f"[Info] mode: {'video' if args.video else 'image'}")
    print(f"[Info] prefix: {args.image_prefix}")
    print(f"[Info] extensions: {exts}")
    print(f"[Info] device: {args.device}")
    print(f"[Info] lpips net: {args.net}")
    print(f"[Info] batch_size: {args.batch_size}")
    print(f"[Info] frame_batch_size: {args.frame_batch_size}")

    ref_map = collect_files_by_index(
        args.ref_dir,
        args.image_prefix,
        exts,
        "video" if args.video else "image",
    )
    tgt_map = collect_files_by_index(
        args.target_dir,
        args.image_prefix,
        exts,
        "video" if args.video else "image",
    )

    label = "videos" if args.video else "images"
    print(f"[Info] reference {label} found: {len(ref_map)}")
    print(f"[Info] target {label} found: {len(tgt_map)}")

    ref_indices = set(ref_map)
    tgt_indices = set(tgt_map)
    common_indices = sorted(ref_indices & tgt_indices)
    only_ref = sorted(ref_indices - tgt_indices)
    only_tgt = sorted(tgt_indices - ref_indices)

    if only_ref:
        print(f"[Warn] indices only in reference: {len(only_ref)}")
    if only_tgt:
        print(f"[Warn] indices only in target: {len(only_tgt)}")

    if args.strict_missing and (only_ref or only_tgt):
        raise FileNotFoundError(
            "Some file indices are missing in one of the directories."
        )

    if not common_indices:
        raise ValueError("No overlapping file indices found between two directories.")

    loss_fn = lpips.LPIPS(net=args.net).to(args.device)
    loss_fn.eval()

    if args.video:
        records, values, skipped_rows = compute_video_lpips_records(
            common_indices=common_indices,
            ref_map=ref_map,
            tgt_map=tgt_map,
            loss_fn=loss_fn,
            device=args.device,
            strict_size=args.strict_size,
            resize_to_ref=args.resize_to_ref,
            strict_video_length=args.strict_video_length,
            frame_batch_size=args.frame_batch_size,
        )
    else:
        records, values, skipped_rows = compute_image_lpips_records(
            common_indices=common_indices,
            ref_map=ref_map,
            tgt_map=tgt_map,
            loss_fn=loss_fn,
            device=args.device,
            batch_size=args.batch_size,
            strict_size=args.strict_size,
            resize_to_ref=args.resize_to_ref,
        )

    summary = build_summary(
        values,
        total_rows=len(common_indices),
        skipped_rows=skipped_rows,
    )

    print("\n=== LPIPS Summary ===")
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
                "reference_dir": str(args.ref_dir),
                "target_dir": str(args.target_dir),
                "mode": "video" if args.video else "image",
                "image_prefix": args.image_prefix,
                "extensions": exts,
                "strict_size": args.strict_size,
                "resize_to_ref": args.resize_to_ref,
                "strict_video_length": args.strict_video_length,
                "strict_missing": args.strict_missing,
                "device": args.device,
                "net": args.net,
                "batch_size": args.batch_size,
                "frame_batch_size": args.frame_batch_size,
                "count_only_reference": len(only_ref),
                "count_only_target": len(only_tgt),
            },
            "summary": summary,
            "records": records,
        }

        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[Info] result JSON written to: {args.output_json}")


if __name__ == "__main__":
    main()
