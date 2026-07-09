#!/usr/bin/env python3
"""Compute SSIM by recursively searching two directories and pairing images/videos by index.

Optimizations:
    1. Parallelize image/video pair computation with multiprocessing.
    2. Avoid BGR -> RGB conversion in video mode.
       Both videos are decoded by OpenCV as BGR, so the same channel order is used.
    3. Avoid storing all frame SSIM values; accumulate sum/count instead.

Usage:
    python evaluation/cal_ssim.py \
        --ref_dir /path/to/reference_images \
        --target_dir /path/to/target_images \
        --num_workers 8

    python evaluation/cal_ssim.py \
        --ref_dir /path/to/reference_videos \
        --target_dir /path/to/target_videos \
        --video \
        --num_workers 8 \
        --opencv_threads 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SSIM by pairing images/videos from two directories."
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
        help="Compute SSIM for paired videos instead of paired images.",
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
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 or 1 means serial. -1 means os.cpu_count().",
    )
    parser.add_argument(
        "--opencv_threads",
        type=int,
        default=0,
        help="OpenCV internal threads per worker. Usually 0 or 1 with multiprocessing.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path for per-item SSIM and summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-item SSIM.",
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
        fieldnames = ["row", "index", "reference_path", "target_path", "ssim"]

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


def load_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def bicubic_resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BICUBIC)
    return int(getattr(Image, "BICUBIC", 3))


def compute_ssim(ref_img: np.ndarray, tgt_img: np.ndarray) -> float:
    value = ssim(ref_img, tgt_img, data_range=255, channel_axis=2)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def compute_video_ssim_fast(
    ref_video_path: Path,
    tgt_video_path: Path,
    strict_size: bool,
    resize_to_ref: bool,
    strict_video_length: bool,
) -> tuple[float, int, int, bool]:
    ref_cap = cv2.VideoCapture(str(ref_video_path))
    tgt_cap = cv2.VideoCapture(str(tgt_video_path))

    if not ref_cap.isOpened():
        raise RuntimeError(f"Failed to open reference video: {ref_video_path}")
    if not tgt_cap.isOpened():
        raise RuntimeError(f"Failed to open target video: {tgt_video_path}")

    sum_ssim = 0.0
    compared_frames = 0
    skipped_frames = 0
    length_mismatch = False

    try:
        while True:
            ret_ref, ref_frame = ref_cap.read()
            ret_tgt, tgt_frame = tgt_cap.read()

            if not ret_ref or not ret_tgt:
                if ret_ref != ret_tgt:
                    length_mismatch = True
                break

            # No BGR -> RGB conversion here.
            # Both frames are decoded by OpenCV as BGR, so channel order is consistent.
            ref_img = ref_frame
            tgt_img = tgt_frame

            if ref_img.shape != tgt_img.shape:
                if strict_size:
                    raise ValueError(
                        f"size mismatch, ref={ref_img.shape}, tgt={tgt_img.shape}"
                    )

                if resize_to_ref:
                    tgt_img = cv2.resize(
                        tgt_img,
                        (ref_img.shape[1], ref_img.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    skipped_frames += 1
                    continue

            sum_ssim += compute_ssim(ref_img, tgt_img)
            compared_frames += 1

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
        float(sum_ssim / compared_frames),
        compared_frames,
        skipped_frames,
        length_mismatch,
    )


def process_one_ssim_item(task: tuple[Any, ...]) -> Dict[str, Any]:
    (
        row,
        idx,
        ref_path,
        tgt_path,
        is_video,
        strict_size,
        resize_to_ref,
        strict_video_length,
    ) = task

    if is_video:
        value, compared_frames, skipped_frames, length_mismatch = (
            compute_video_ssim_fast(
                ref_video_path=ref_path,
                tgt_video_path=tgt_path,
                strict_size=strict_size,
                resize_to_ref=resize_to_ref,
                strict_video_length=strict_video_length,
            )
        )

        return {
            "skipped": False,
            "warning": (
                f"[Warn] row {row}: video length mismatch for index {idx}, "
                f"computed on overlapped prefix frames"
                if length_mismatch
                else None
            ),
            "value": value,
            "record": {
                "row": row,
                "index": idx,
                "reference_path": str(ref_path),
                "target_path": str(tgt_path),
                "compared_frames": compared_frames,
                "skipped_frames": skipped_frames,
                "ssim": value,
            },
        }

    ref_img = load_rgb_array(ref_path)
    tgt_img = load_rgb_array(tgt_path)

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
            return {
                "skipped": True,
                "warning": (
                    f"[Warn] row {row}: size mismatch for index {idx}, skipped "
                    f"(ref={ref_img.shape}, tgt={tgt_img.shape})"
                ),
                "value": None,
                "record": None,
            }

    value = compute_ssim(ref_img, tgt_img)

    return {
        "skipped": False,
        "warning": None,
        "value": value,
        "record": {
            "row": row,
            "index": idx,
            "reference_path": str(ref_path),
            "target_path": str(tgt_path),
            "ssim": value,
        },
    }


def init_worker(opencv_threads: int) -> None:
    try:
        cv2.setNumThreads(opencv_threads)
    except Exception:
        pass


def compute_ssim_records(
    common_indices: Sequence[int],
    ref_map: Dict[int, Path],
    tgt_map: Dict[int, Path],
    is_video: bool,
    strict_size: bool,
    resize_to_ref: bool,
    strict_video_length: bool,
    num_workers: int,
    opencv_threads: int,
) -> tuple[list[Dict[str, Any]], list[float], int]:
    tasks = [
        (
            row,
            idx,
            ref_map[idx],
            tgt_map[idx],
            is_video,
            strict_size,
            resize_to_ref,
            strict_video_length,
        )
        for row, idx in enumerate(common_indices)
    ]

    if num_workers == -1:
        actual_workers = os.cpu_count() or 1
    else:
        actual_workers = num_workers

    records: list[Dict[str, Any]] = []
    values: list[float] = []
    skipped_rows = 0

    unit = "video" if is_video else "image"

    if actual_workers <= 1:
        try:
            cv2.setNumThreads(opencv_threads)
        except Exception:
            pass

        iterator = tqdm(tasks, total=len(tasks), desc="SSIM", unit=unit)
        for task in iterator:
            result = process_one_ssim_item(task)

            if result["warning"]:
                print(result["warning"])

            if result["skipped"]:
                skipped_rows += 1
                continue

            record = result["record"]
            value = result["value"]
            assert record is not None
            assert value is not None

            records.append(record)
            values.append(float(value))

    else:
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=init_worker,
            initargs=(opencv_threads,),
        ) as executor:
            futures = [executor.submit(process_one_ssim_item, task) for task in tasks]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="SSIM",
                unit=unit,
            ):
                result = future.result()

                if result["warning"]:
                    print(result["warning"])

                if result["skipped"]:
                    skipped_rows += 1
                    continue

                record = result["record"]
                value = result["value"]
                assert record is not None
                assert value is not None

                records.append(record)
                values.append(float(value))

    records.sort(key=lambda x: x["row"])
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
    print(f"[Info] num_workers: {args.num_workers}")
    print(f"[Info] opencv_threads: {args.opencv_threads}")

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

    records, values, skipped_rows = compute_ssim_records(
        common_indices=common_indices,
        ref_map=ref_map,
        tgt_map=tgt_map,
        is_video=args.video,
        strict_size=args.strict_size,
        resize_to_ref=args.resize_to_ref,
        strict_video_length=args.strict_video_length,
        num_workers=args.num_workers,
        opencv_threads=args.opencv_threads,
    )

    summary = build_summary(
        values,
        total_rows=len(common_indices),
        skipped_rows=skipped_rows,
    )

    print("\n=== SSIM Summary ===")
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
                "num_workers": args.num_workers,
                "opencv_threads": args.opencv_threads,
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
