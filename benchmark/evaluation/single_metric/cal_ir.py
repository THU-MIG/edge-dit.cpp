#!/usr/bin/env python3
"""Compute ImageReward for one run directory by auto-discovering prompts and images.

Expected run layout (example):
	results/flux/original_auto/
		|- *_prompts.txt
		|- *_merged_manifest.json (optional)
		|- <one image folder>/img_0.jpg ...

Usage:
	python evaluation/cal_ir.py \
		--run_dir results/flux/original_auto \
		--model_name /path/to/local/ImageReward-v1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# transformers>=5 compatibility shim for the vendored 2022-era BLIP/BERT inside
# the image-reward package. This shim is SELF-CONTAINED: it patches the
# transformers namespaces BEFORE `import ImageReward`, so a CLEAN
# `pip install image-reward` runs on transformers>=5 with NO edits to the
# installed package source. (Previously these fixes lived as hand edits in
# site-packages/ImageReward/models/BLIP/{med,blip}.py and were not reproducible.)
# ---------------------------------------------------------------------------
from transformers import PreTrainedModel as _PreTrainedModel

# (1) tie_weights on transformers>=5 expects this class attribute to exist.
if not hasattr(_PreTrainedModel, "all_tied_weights_keys"):
    _PreTrainedModel.all_tied_weights_keys = {}

# (2) BLIP/BERT calls self.get_head_mask; removed from ModuleUtilsMixin in v5.
from transformers.modeling_utils import ModuleUtilsMixin as _ModuleUtilsMixin
if not hasattr(_ModuleUtilsMixin, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        # BLIP/BERT scoring path never passes a head_mask; return the no-mask list.
        if head_mask is not None:
            raise NotImplementedError("head_mask pruning not supported by shim")
        return [None] * num_hidden_layers
    _ModuleUtilsMixin.get_head_mask = _get_head_mask

# (3) med.py does `from transformers.pytorch_utils import (apply_chunking_to_forward,
#     prune_linear_layer, find_pruneable_heads_and_indices)`. v5 dropped
#     find_pruneable_heads_and_indices. Inject a vendored copy so the import
#     resolves against a clean install.
import transformers.pytorch_utils as _pytorch_utils
if not hasattr(_pytorch_utils, "find_pruneable_heads_and_indices"):
    import torch as _torch

    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = _torch.arange(len(mask))[mask].long()
        return heads, index

    _pytorch_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

# (4) blip.py init_tokenizer reads tokenizer.additional_special_tokens_ids[0];
#     both `additional_special_tokens` and `additional_special_tokens_ids` were
#     removed from the tokenizer base in v5. Restore them as properties derived
#     from all_special_tokens minus the standard special-token slots, so a clean
#     image-reward install works unmodified. (For BLIP, the only extra token is
#     '[ENC]', which is exactly what init_tokenizer wants at index 0.)
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _TokBase

    def _extra_special_tokens(self):
        standard = set((self.special_tokens_map or {}).values())
        return [t for t in (self.all_special_tokens or []) if t not in standard]

    if not hasattr(_TokBase, "additional_special_tokens"):
        _TokBase.additional_special_tokens = property(_extra_special_tokens)

    if not hasattr(_TokBase, "additional_special_tokens_ids"):
        def _extra_special_token_ids(self):
            return [self.convert_tokens_to_ids(t) for t in _extra_special_tokens(self)]

        _TokBase.additional_special_tokens_ids = property(_extra_special_token_ids)
except Exception:
    pass

import ImageReward
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ImageReward by auto-discovering prompts/images."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory such as results/flux/original_auto.",
    )
    parser.add_argument(
        "--prompts_file",
        type=Path,
        default=None,
        help="Optional prompts txt file. If not set, auto-search *_prompts.txt under run_dir.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Optional image directory. If not set, auto-search a folder that contains img_*.jpg/png.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional merged manifest path. If not set, auto-search *_merged_manifest.json under run_dir.",
    )
    parser.add_argument(
        "--manifest_prompt_field",
        type=str,
        default="prompt",
        help="Prompt field if manifest is used for fallback pairing.",
    )
    parser.add_argument(
        "--manifest_index_field",
        type=str,
        default="index",
        help="Index field if manifest is used for fallback pairing.",
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
        "--pair_mode",
        type=str,
        choices=["prompts", "manifest"],
        default="prompts",
        help="Use prompts txt or manifest to get prompt-index mapping.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ImageReward-v1.0",
        help=(
            "Model identifier for ImageReward.load. Can be 'ImageReward-v1.0' "
            "or a local directory path."
        ),
    )
    parser.add_argument(
        "--download_root",
        type=Path,
        default=None,
        help="Optional local cache root for model weights.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu."
    )
    parser.add_argument(
        "--max_items", type=int, default=None, help="Evaluate only first N items."
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail immediately if any prompt/image is missing.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path for per-item scores and summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-item scores.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item at {i} is not an object/dict.")

    return data


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
        raise FileNotFoundError(
            f"No matching images found under: {image_dir} for prefix '{image_prefix}' and exts {list(exts)}"
        )
    return result


def build_pairs_from_prompts(prompts: Sequence[str]) -> List[Tuple[int, str]]:
    return [(idx, prompt) for idx, prompt in enumerate(prompts)]


def build_pairs_from_manifest(
    items: Sequence[Dict[str, Any]], prompt_field: str, index_field: str
) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    for row, item in enumerate(items):
        if index_field not in item:
            raise ValueError(f"Manifest row {row} missing index field: {index_field}")
        if prompt_field not in item:
            raise ValueError(f"Manifest row {row} missing prompt field: {prompt_field}")

        idx = int(item[index_field])
        prompt = item[prompt_field]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Manifest row {row} has invalid prompt")
        pairs.append((idx, prompt.strip()))
    return pairs


def load_image_reward_model(args: argparse.Namespace):
    model_name_or_path: str = args.model_name
    config = None
    if Path(model_name_or_path).exists():
        model_name_or_path = str(Path(model_name_or_path).resolve() / "ImageReward.pt")
        config = str(Path(args.model_name).resolve() / "med_config.json")

    kwargs: Dict[str, Any] = {"device": args.device}
    if config is not None:
        kwargs["med_config"] = config
    if args.download_root is not None:
        kwargs["download_root"] = str(args.download_root)

    try:
        return ImageReward.load(model_name_or_path, **kwargs)
    except TypeError:
        kwargs.pop("download_root", None)
        return ImageReward.load(model_name_or_path, **kwargs)


def maybe_mkdir_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ["row", "index", "prompt", "image", "score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})


def build_summary(
    scores: Sequence[float], total_rows: int, skipped_rows: int
) -> Dict[str, Any]:
    if not scores:
        return {
            "count_scored": 0,
            "count_total_rows": total_rows,
            "count_skipped": skipped_rows,
        }

    sorted_scores = sorted(scores)
    mid = len(sorted_scores) // 2
    median = (
        sorted_scores[mid]
        if len(sorted_scores) % 2 == 1
        else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    )

    return {
        "count_scored": len(scores),
        "count_total_rows": total_rows,
        "count_skipped": skipped_rows,
        "mean": float(statistics.mean(scores)),
        "std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        "min": float(min(scores)),
        "median": float(median),
        "max": float(max(scores)),
    }


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    if not args.run_dir.is_dir():
        raise NotADirectoryError(f"run_dir does not exist: {args.run_dir}")

    exts = [
        e.strip().lstrip(".").lower() for e in args.image_exts.split(",") if e.strip()
    ]
    if not exts:
        raise ValueError("--image_exts is empty")

    if args.download_root is not None:
        args.download_root = args.download_root.expanduser().resolve()
        os.environ.setdefault("HF_HOME", str(args.download_root))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.download_root))

    if args.prompts_file is None:
        args.prompts_file = auto_find_single_file(
            args.run_dir, "*prompts.txt", "prompts file"
        )
    else:
        args.prompts_file = args.prompts_file.expanduser().resolve()

    if args.manifest is None:
        try:
            args.manifest = auto_find_single_file(
                args.run_dir, "*_merged_manifest.json", "merged manifest"
            )
        except FileNotFoundError:
            args.manifest = None
    else:
        args.manifest = args.manifest.expanduser().resolve()

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

    print(f"[Info] run_dir: {args.run_dir}")
    print(f"[Info] prompts_file: {args.prompts_file}")
    print(f"[Info] image_dir: {args.image_dir}")
    if args.manifest is not None:
        print(f"[Info] manifest: {args.manifest}")
    print(f"[Info] prompts count: {len(prompts)}")
    print(f"[Info] loading ImageReward model: {args.model_name}")
    model = load_image_reward_model(args)

    image_map = collect_images_by_index(args.image_dir, args.image_prefix, exts)
    print(f"[Info] images found: {len(image_map)}")

    if args.pair_mode == "manifest":
        if args.manifest is None:
            raise ValueError("pair_mode=manifest requires a merged manifest")
        manifest_items = load_manifest(args.manifest)
        pairs = build_pairs_from_manifest(
            manifest_items,
            prompt_field=args.manifest_prompt_field,
            index_field=args.manifest_index_field,
        )
        if args.max_items is not None:
            pairs = pairs[: args.max_items]
    else:
        pairs = build_pairs_from_prompts(prompts)

    records: List[Dict[str, Any]] = []
    scores: List[float] = []
    skipped_rows = 0

    progress = tqdm(
        enumerate(pairs),
        total=len(pairs),
        desc="Scoring",
        unit="item",
    )

    for row, (idx, prompt) in progress:
        image_path = image_map.get(idx)
        if image_path is None:
            msg = f"row {row}: image not found for index {idx}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            print(f"[Warn] {msg}")
            skipped_rows += 1
            continue

        score_raw = model.score(prompt, str(image_path))
        score = float(score_raw)

        record = {
            "row": row,
            "index": idx,
            "prompt": prompt,
            "image": str(image_path),
            "score": score,
        }
        records.append(record)
        scores.append(score)

    summary = build_summary(scores, total_rows=len(pairs), skipped_rows=skipped_rows)

    print("\n=== ImageReward Summary ===")
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
                "manifest": str(args.manifest) if args.manifest is not None else None,
                "pair_mode": args.pair_mode,
                "image_dir": str(args.image_dir),
                "image_prefix": args.image_prefix,
                "image_exts": exts,
                "model_name": args.model_name,
                "device": args.device,
            },
            "summary": summary,
            "records": records,
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Info] result JSON written to: {args.output_json}")


if __name__ == "__main__":
    main()
