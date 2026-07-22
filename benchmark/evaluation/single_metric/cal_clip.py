#!/usr/bin/env python3
"""Compute CLIP score (text-image cosine similarity) for one run directory.

Usage:
	python evaluation/cal_clip.py \
		--run_dir /path/to/run_dir \
		--image_dir /path/to/target_images
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise ImportError(
        "transformers is required for cal_clip.py. Install it with: pip install transformers"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP text-image score for one run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing prompts file and generated images.",
    )
    parser.add_argument(
        "--prompts_file",
        type=Path,
        default=None,
        help="Optional prompts file. If not set, auto-search *_prompts.txt under run_dir.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Optional target image directory. If not set, auto-search under run_dir.",
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        default="img_",
        help="Image filename prefix, e.g. img_.",
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
        help="Score paired videos instead of images: decode frames, compute "
        "per-frame CLIP, then average per video.",
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
        help="Fail if prompt-image pairing cannot be found for an index.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name or local path.",
    )
    parser.add_argument(
        "--max_items", type=int, default=None, help="Evaluate only first N prompts."
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path for per-item CLIP score and summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-item CLIP score.",
    )
    return parser.parse_args()


def auto_find_single_file(root: Path, pattern: str, label: str) -> Path:
    matched = sorted(root.rglob(pattern))
    if not matched:
        raise FileNotFoundError(
            f"No {label} found under: {root} with pattern: {pattern}"
        )
    if len(matched) > 1:
        names = "\n".join(str(p) for p in matched[:10])
        extra = "" if len(matched) <= 10 else f"\n... and {len(matched) - 10} more"
        raise ValueError(f"Multiple {label} found under: {root}\n{names}{extra}")
    return matched[0]


def parse_prompt_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.rstrip("\n").strip()
            if text:
                lines.append(text)
    if not lines:
        raise ValueError(f"No valid prompts found in: {path}")
    return lines


def find_best_image_dir(root: Path, image_prefix: str, exts: Sequence[str]) -> Path:
    best_dir: Path | None = None
    best_count = -1

    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        count = 0
        for ext in exts:
            count += len(list(d.glob(f"{image_prefix}*.{ext}")))
        if count > best_count:
            best_count = count
            best_dir = d

    if best_dir is None or best_count <= 0:
        raise FileNotFoundError(
            f"No images found under: {root} for prefix '{image_prefix}' and exts {list(exts)}"
        )

    return best_dir


def collect_images_by_index(
    image_dir: Path, image_prefix: str, exts: Sequence[str]
) -> Dict[int, Path]:
    pattern = re.compile(
        rf"^{re.escape(image_prefix)}(\d+)\.({'|'.join(re.escape(e) for e in exts)})$",
        re.IGNORECASE,
    )
    result: Dict[int, Path] = {}
    for p in image_dir.rglob("*"):
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in result:
            raise ValueError(
                f"Duplicate image index {idx} found under {image_dir}: {result[idx]} and {p}"
            )
        result[idx] = p
    if not result:
        raise FileNotFoundError(f"No matching images found under: {image_dir}")
    return result


def build_pairs_from_prompts(prompts: Sequence[str]) -> List[Tuple[int, str]]:
    return [(idx, prompt) for idx, prompt in enumerate(prompts)]


def maybe_mkdir_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ["row", "index", "prompt", "image", "clip_score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})


def build_summary(
    values: Sequence[float], total_rows: int, skipped_rows: int
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


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def iter_video_frames_rgb(path: Path):
    """Yield PIL RGB frames from a video, decoding with OpenCV (BGR -> RGB)."""
    import cv2  # local import so image-only usage does not require cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb)
    finally:
        cap.release()


def score_image_clip(model, processor, device, image, prompt) -> float:
    inputs = processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    text_features = F.normalize(outputs.text_embeds, dim=-1)
    image_features = F.normalize(outputs.image_embeds, dim=-1)
    return float((text_features * image_features).sum().item())


def score_video_clip(model, processor, device, video_path, prompt) -> tuple[float, int]:
    """Mean per-frame CLIP score for one video against one prompt."""
    total = 0.0
    n = 0
    for frame in iter_video_frames_rgb(video_path):
        total += score_image_clip(model, processor, device, frame, prompt)
        n += 1
    if n == 0:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return total / n, n


def main() -> None:
    args = parse_args()

    args.run_dir = args.run_dir.expanduser().resolve()
    if not args.run_dir.is_dir():
        raise NotADirectoryError(f"run_dir does not exist: {args.run_dir}")

    if args.prompts_file is None:
        args.prompts_file = auto_find_single_file(
            args.run_dir, "*prompts.txt", "prompts file"
        )
    else:
        args.prompts_file = args.prompts_file.expanduser().resolve()

    exts = [
        e.strip().lstrip(".").lower()
        for e in (args.video_exts if args.video else args.image_exts).split(",")
        if e.strip()
    ]
    if not exts:
        raise ValueError("--image_exts/--video_exts is empty")

    if args.image_dir is None:
        args.image_dir = find_best_image_dir(args.run_dir, args.image_prefix, exts)
    else:
        args.image_dir = args.image_dir.expanduser().resolve()
        if not args.image_dir.is_dir():
            raise NotADirectoryError(f"image_dir does not exist: {args.image_dir}")

    prompts = parse_prompt_lines(args.prompts_file)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max_items must be positive.")
        prompts = prompts[: args.max_items]
    pairs = build_pairs_from_prompts(prompts)

    print(f"[Info] run_dir: {args.run_dir}")
    print(f"[Info] prompts_file: {args.prompts_file}")
    print(f"[Info] image_dir: {args.image_dir}")
    print(f"[Info] mode: {'video' if args.video else 'image'}")
    print(f"[Info] device: {args.device}")
    print(f"[Info] model_name: {args.model_name}")
    print(f"[Info] prompts count: {len(prompts)}")

    image_map = collect_images_by_index(args.image_dir, args.image_prefix, exts)
    print(f"[Info] {'videos' if args.video else 'images'} found: {len(image_map)}")

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    records: List[Dict[str, Any]] = []
    values: List[float] = []
    skipped_rows = 0

    progress = tqdm(
        enumerate(pairs),
        total=len(pairs),
        desc="CLIP",
        unit="item",
    )

    with torch.no_grad():
        for row, (idx, prompt) in progress:
            image_path = image_map.get(idx)
            if image_path is None:
                msg = f"row {row}: {'video' if args.video else 'image'} not found for index {idx}"
                if args.strict_missing:
                    raise FileNotFoundError(msg)
                print(f"[Warn] {msg}")
                skipped_rows += 1
                continue

            if args.video:
                value, n_frames = score_video_clip(
                    model, processor, args.device, image_path, prompt
                )
                record = {
                    "row": row,
                    "index": idx,
                    "prompt": prompt,
                    "image": str(image_path),
                    "num_frames": n_frames,
                    "clip_score": value,
                }
            else:
                image = open_rgb_image(image_path)
                value = score_image_clip(
                    model, processor, args.device, image, prompt
                )
                record = {
                    "row": row,
                    "index": idx,
                    "prompt": prompt,
                    "image": str(image_path),
                    "clip_score": value,
                }
            records.append(record)
            values.append(value)

    summary = build_summary(values, total_rows=len(pairs), skipped_rows=skipped_rows)

    print("\n=== CLIP Summary ===")
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
                "run_dir": str(args.run_dir),
                "prompts_file": str(args.prompts_file),
                "image_dir": str(args.image_dir),
                "image_prefix": args.image_prefix,
                "image_exts": exts,
                "video": args.video,
                "strict_missing": args.strict_missing,
                "device": args.device,
                "model_name": args.model_name,
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
