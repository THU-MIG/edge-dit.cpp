#!/usr/bin/env python3
"""Directional CLIP score for image editing: how well the image edit direction
matches the instruction's text direction.

Standard CLIP directional similarity (StyleGAN-NADA / InstructPix2Pix eval):

    img_dir = normalize(emb(output_image)) - normalize(emb(input_image))
    txt_dir = normalize(emb(target_caption)) - normalize(emb(source_caption))
    score   = cosine(img_dir, txt_dir)

A high score means the output moved away from the input in the same semantic
direction the instruction asks for. Unlike plain image-text CLIP, this accounts
for the input image (what the object was) and rewards *relative* change, not just
absolute agreement with the target words.

Caption construction (auto, no per-instruction hard-coding):
    source_caption : neutral anchor for the un-edited image (default "a photo")
    target_caption : the edit instruction text itself (default), describing the
                     desired end-state. Both are overridable via CLI and are
                     recorded in the output JSON so the derived pair is auditable.

All output images share ONE input image (the edit source), paired by index to
per-line instructions.

Usage:
    python evaluation/single_metric/cal_clip_directional.py \
        --input_image /path/to/logo.png \
        --output_dir /path/to/run/samples/<adapter> \
        --instructions_file /path/to/prompts.txt \
        --image_prefix output_ \
        --output_json /path/to/clip_directional.json
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
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "transformers is required for cal_clip_directional.py. "
        "Install it with: pip install transformers"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute directional CLIP score for image-editing runs."
    )
    parser.add_argument(
        "--input_image",
        type=Path,
        required=True,
        help="Single source image shared by all edits (the edit input).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing edited output images.",
    )
    parser.add_argument(
        "--instructions_file",
        type=Path,
        required=True,
        help="Text file, one edit instruction per line, paired to output images by index.",
    )
    parser.add_argument(
        "--source_caption",
        type=str,
        default="a photo",
        help="Neutral anchor caption for the un-edited image.",
    )
    parser.add_argument(
        "--target_template",
        type=str,
        default="{instruction}",
        help="Template for the target caption; {instruction} is substituted per line.",
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        default="output_",
        help="Output image filename prefix, e.g. output_ or measured_.",
    )
    parser.add_argument(
        "--image_exts",
        type=str,
        default="png,jpg,jpeg,webp,bmp,ppm",
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail if an instruction has no matching output image for its index.",
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
        "--max_items", type=int, default=None, help="Evaluate only first N instructions."
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


def parse_prompt_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Instructions file not found: {path}")
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.rstrip("\n").strip()
            if text:
                lines.append(text)
    if not lines:
        raise ValueError(f"No valid instructions found in: {path}")
    return lines


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
        raise FileNotFoundError(
            f"No matching images found under: {image_dir} "
            f"for prefix '{image_prefix}' and exts {list(exts)}"
        )
    return result


def maybe_mkdir_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "row",
        "index",
        "source_caption",
        "target_caption",
        "image",
        "directional_clip",
    ]
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


@torch.no_grad()
def image_embed(model, processor, device, image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    if not isinstance(feats, torch.Tensor):  # transformers>=5 may return an output object
        feats = feats.pooler_output
    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def text_embed(model, processor, device, text) -> torch.Tensor:
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    return F.normalize(feats.float(), dim=-1)


def main() -> None:
    args = parse_args()

    args.input_image = args.input_image.expanduser().resolve()
    if not args.input_image.is_file():
        raise FileNotFoundError(f"input_image not found: {args.input_image}")

    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.output_dir.is_dir():
        raise NotADirectoryError(f"output_dir does not exist: {args.output_dir}")

    exts = [
        e.strip().lstrip(".").lower() for e in args.image_exts.split(",") if e.strip()
    ]
    if not exts:
        raise ValueError("--image_exts is empty")

    instructions = parse_prompt_lines(args.instructions_file)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max_items must be positive.")
        instructions = instructions[: args.max_items]

    if "{instruction}" not in args.target_template:
        raise ValueError("--target_template must contain '{instruction}'")

    print(f"[Info] input_image: {args.input_image}")
    print(f"[Info] output_dir: {args.output_dir}")
    print(f"[Info] instructions: {len(instructions)}")
    print(f"[Info] source_caption: {args.source_caption!r}")
    print(f"[Info] target_template: {args.target_template!r}")
    print(f"[Info] device: {args.device}")
    print(f"[Info] model_name: {args.model_name}")

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    image_map = collect_images_by_index(args.output_dir, args.image_prefix, exts)
    print(f"[Info] output images found: {len(image_map)}")

    input_feat = image_embed(model, processor, args.device, open_rgb_image(args.input_image))
    source_feat = text_embed(model, processor, args.device, args.source_caption)

    records: List[Dict[str, Any]] = []
    values: List[float] = []
    skipped_rows = 0

    progress = tqdm(
        enumerate(instructions), total=len(instructions), desc="dCLIP", unit="item"
    )
    for row, instruction in progress:
        image_path = image_map.get(row)
        if image_path is None:
            msg = f"row {row}: output image not found for index {row}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            print(f"[Warn] {msg}")
            skipped_rows += 1
            continue

        target_caption = args.target_template.format(instruction=instruction)
        out_feat = image_embed(model, processor, args.device, open_rgb_image(image_path))
        target_feat = text_embed(model, processor, args.device, target_caption)

        img_dir = out_feat - input_feat
        txt_dir = target_feat - source_feat
        score = float(F.cosine_similarity(img_dir, txt_dir, dim=-1).item())

        records.append(
            {
                "row": row,
                "index": row,
                "source_caption": args.source_caption,
                "target_caption": target_caption,
                "image": str(image_path),
                "directional_clip": score,
            }
        )
        values.append(score)

    summary = build_summary(values, total_rows=len(instructions), skipped_rows=skipped_rows)

    print("\n=== Directional CLIP Summary ===")
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
                "input_image": str(args.input_image),
                "output_dir": str(args.output_dir),
                "instructions_file": str(args.instructions_file),
                "source_caption": args.source_caption,
                "target_template": args.target_template,
                "image_prefix": args.image_prefix,
                "image_exts": exts,
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
