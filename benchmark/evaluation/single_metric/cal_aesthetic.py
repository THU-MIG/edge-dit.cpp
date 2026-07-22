#!/usr/bin/env python3
"""Compute LAION aesthetic score (v2) for one run directory or image dir.

Aesthetic predictor v2 = CLIP ViT-L/14 image embedding (768-d, L2-normalized)
-> small MLP head (christophschuhmann/improved-aesthetic-predictor,
sac+logos+ava1-l14-linearMSE.pth). Output is roughly in [1, 10]; higher is
"more aesthetically pleasing" on average.

The CLIP ViT-L/14 backbone is loaded via HuggingFace transformers
(openai/clip-vit-large-patch14) so it works with the HF mirror and matches the
rest of this evaluation suite; the 768-d projected image embedding is
equivalent to OpenAI CLIP `encode_image` used by the original predictor.

Usage:
    python evaluation/single_metric/cal_aesthetic.py \
        --run_dir /path/to/run_dir \
        --weights /path/to/sac+logos+ava1-l14-linearMSE.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "transformers is required for cal_aesthetic.py. "
        "Install it with: pip install transformers"
    ) from exc


class AestheticMLP(nn.Module):
    """MLP head matching improved-aesthetic-predictor sac+logos+ava1-l14.

    Layer indices (0,2,4,6,7) and dropout placement (1,3,5) reproduce the
    original nn.Sequential so the released state_dict loads directly.
    """

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LAION aesthetic score (v2) for images in a run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Run directory containing generated images (auto-search).",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Image directory. If not set, auto-search under run_dir.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to sac+logos+ava1-l14-linearMSE.pth aesthetic MLP weights.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace CLIP ViT-L/14 model name or local path (768-d embed).",
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
        "per-frame aesthetic score, then average per video.",
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        default="mp4,mov,mkv,avi,webm,m4v",
        help="Comma-separated video extensions used when --video is enabled.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu."
    )
    parser.add_argument(
        "--max_items", type=int, default=None, help="Evaluate only first N images."
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path for per-item score and summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-item score.",
    )
    return parser.parse_args()


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
                f"Duplicate image index {idx} found under {image_dir}: "
                f"{result[idx]} and {p}"
            )
        result[idx] = p
    if not result:
        raise FileNotFoundError(f"No matching images found under: {image_dir}")
    return result


def maybe_mkdir_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ["row", "index", "image", "aesthetic_score"]
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


def score_image_aesthetic(clip_model, mlp, processor, device, image) -> float:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_features = clip_model.get_image_features(**inputs)
    if not isinstance(image_features, torch.Tensor):
        image_features = image_features.pooler_output
    image_features = F.normalize(image_features.float(), dim=-1)
    return float(mlp(image_features).squeeze().item())


def score_video_aesthetic(
    clip_model, mlp, processor, device, video_path
) -> tuple[float, int]:
    """Mean per-frame aesthetic score for one video."""
    total = 0.0
    n = 0
    for frame in iter_video_frames_rgb(video_path):
        total += score_image_aesthetic(clip_model, mlp, processor, device, frame)
        n += 1
    if n == 0:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return total / n, n


def main() -> None:
    args = parse_args()

    exts = [
        e.strip().lstrip(".").lower()
        for e in (args.video_exts if args.video else args.image_exts).split(",")
        if e.strip()
    ]
    if not exts:
        raise ValueError("--image_exts/--video_exts is empty")

    if args.image_dir is not None:
        image_dir = args.image_dir.expanduser().resolve()
        if not image_dir.is_dir():
            raise NotADirectoryError(f"image_dir does not exist: {image_dir}")
    elif args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise NotADirectoryError(f"run_dir does not exist: {run_dir}")
        image_dir = find_best_image_dir(run_dir, args.image_prefix, exts)
    else:
        raise ValueError("Provide either --run_dir or --image_dir.")

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"aesthetic weights not found: {weights_path}")

    print(f"[Info] image_dir: {image_dir}")
    print(f"[Info] mode: {'video' if args.video else 'image'}")
    print(f"[Info] device: {args.device}")
    print(f"[Info] clip_model_name: {args.clip_model_name}")
    print(f"[Info] weights: {weights_path}")

    processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(args.device)
    clip_model.eval()

    embed_dim = int(clip_model.config.projection_dim)
    mlp = AestheticMLP(embed_dim)
    state = torch.load(str(weights_path), map_location="cpu")
    mlp.load_state_dict(state)
    mlp.to(args.device)
    mlp.eval()

    image_map = collect_images_by_index(image_dir, args.image_prefix, exts)
    indices = sorted(image_map)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max_items must be positive.")
        indices = indices[: args.max_items]
    print(f"[Info] {'videos' if args.video else 'images'} found: {len(indices)}")

    records: List[Dict[str, Any]] = []
    values: List[float] = []

    unit = "video" if args.video else "img"
    progress = tqdm(enumerate(indices), total=len(indices), desc="Aesthetic", unit=unit)
    with torch.no_grad():
        for row, idx in progress:
            image_path = image_map[idx]
            if args.video:
                score, n_frames = score_video_aesthetic(
                    clip_model, mlp, processor, args.device, image_path
                )
                record = {
                    "row": row,
                    "index": idx,
                    "image": str(image_path),
                    "num_frames": n_frames,
                    "aesthetic_score": score,
                }
            else:
                image = open_rgb_image(image_path)
                score = score_image_aesthetic(
                    clip_model, mlp, processor, args.device, image
                )
                record = {
                    "row": row,
                    "index": idx,
                    "image": str(image_path),
                    "aesthetic_score": score,
                }
            records.append(record)
            values.append(score)

    summary = build_summary(values, total_rows=len(indices), skipped_rows=0)

    print("\n=== Aesthetic Summary ===")
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
                "image_dir": str(image_dir),
                "image_prefix": args.image_prefix,
                "image_exts": exts,
                "video": args.video,
                "device": args.device,
                "clip_model_name": args.clip_model_name,
                "weights": str(weights_path),
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
