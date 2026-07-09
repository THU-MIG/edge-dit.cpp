#!/usr/bin/env python3
"""Run unified text-to-image evaluation for PSNR/SSIM/LPIPS/CLIP/ImageReward.

This script orchestrates existing single-metric scripts and aggregates summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified text-to-image evaluation runner."
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory used by ImageReward (auto-discover prompts/images).",
    )
    parser.add_argument(
        "--ref_dir",
        type=Path,
        required=True,
        help="Reference image root directory for PSNR/SSIM/LPIPS/CLIP.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=None,
        help="Target image root directory for PSNR/SSIM/LPIPS/CLIP. Defaults to run_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for metric JSON files. Defaults to run_dir/eval_summary.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="psnr,ssim,lpips,clip,ir",
        help="Comma-separated metrics from: psnr,ssim,lpips,clip,ir",
    )
    parser.add_argument(
        "--skip_if_json_exists",
        action="store_true",
        help="Skip metric computation if its output JSON already exists and reuse that file.",
    )

    parser.add_argument(
        "--image_prefix", type=str, default="img_", help="Image filename prefix."
    )
    parser.add_argument(
        "--image_exts",
        type=str,
        default="jpg,jpeg,png,webp,bmp",
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail if unmatched image indices exist between reference and target.",
    )
    parser.add_argument(
        "--strict_size",
        action="store_true",
        help="Fail if paired images have different sizes.",
    )
    parser.add_argument(
        "--resize_to_ref",
        action="store_true",
        help="Resize target image to reference size when dimensions differ.",
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device used by LPIPS/CLIP/IR."
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="Backbone network for LPIPS.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch32",
        help="Model name/path for CLIP.",
    )
    parser.add_argument(
        "--ir_model_name",
        type=str,
        default="ImageReward-v1.0",
        help="Model name/path for ImageReward.",
    )
    parser.add_argument(
        "--ir_download_root",
        type=Path,
        default=None,
        help="Optional cache root for ImageReward model download.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[Cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    ref_dir = args.ref_dir.expanduser().resolve()
    target_dir = (
        args.target_dir.expanduser().resolve()
        if args.target_dir is not None
        else run_dir
    )

    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir does not exist: {run_dir}")
    if not ref_dir.is_dir():
        raise NotADirectoryError(f"ref_dir does not exist: {ref_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"target_dir does not exist: {target_dir}")

    metric_set = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
    valid_metrics = {"psnr", "ssim", "lpips", "clip", "ir"}
    unknown = sorted(metric_set - valid_metrics)
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    if not metric_set:
        raise ValueError("No metrics selected.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "eval_summary")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parents[1] / "single_metric"
    script_map = {
        "psnr": scripts_dir / "cal_psnr.py",
        "ssim": scripts_dir / "cal_ssim.py",
        "lpips": scripts_dir / "cal_lpips.py",
        "clip": scripts_dir / "cal_clip.py",
        "ir": scripts_dir / "cal_ir.py",
    }
    for name, path in script_map.items():
        if name in metric_set and not path.is_file():
            raise FileNotFoundError(f"Metric script not found for {name}: {path}")

    results: Dict[str, Any] = {}

    if "psnr" in metric_set:
        out = output_dir / "psnr.json"
        if args.skip_if_json_exists and out.is_file():
            print(f"[Info] skip psnr: found existing {out}")
        else:
            cmd = [
                sys.executable,
                str(script_map["psnr"]),
                "--ref_dir",
                str(ref_dir),
                "--target_dir",
                str(target_dir),
                "--image_prefix",
                args.image_prefix,
                "--image_exts",
                args.image_exts,
                "--output_json",
                str(out),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            if args.strict_size:
                cmd.append("--strict_size")
            if args.resize_to_ref:
                cmd.append("--resize_to_ref")
            run_cmd(cmd)
        results["psnr"] = read_json(out).get("summary", {})

    if "ssim" in metric_set:
        out = output_dir / "ssim.json"
        if args.skip_if_json_exists and out.is_file():
            print(f"[Info] skip ssim: found existing {out}")
        else:
            cmd = [
                sys.executable,
                str(script_map["ssim"]),
                "--ref_dir",
                str(ref_dir),
                "--target_dir",
                str(target_dir),
                "--image_prefix",
                args.image_prefix,
                "--image_exts",
                args.image_exts,
                "--output_json",
                str(out),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            if args.strict_size:
                cmd.append("--strict_size")
            if args.resize_to_ref:
                cmd.append("--resize_to_ref")
            run_cmd(cmd)
        results["ssim"] = read_json(out).get("summary", {})

    if "lpips" in metric_set:
        out = output_dir / "lpips.json"
        if args.skip_if_json_exists and out.is_file():
            print(f"[Info] skip lpips: found existing {out}")
        else:
            cmd = [
                sys.executable,
                str(script_map["lpips"]),
                "--ref_dir",
                str(ref_dir),
                "--target_dir",
                str(target_dir),
                "--image_prefix",
                args.image_prefix,
                "--image_exts",
                args.image_exts,
                "--device",
                args.device,
                "--net",
                args.lpips_net,
                "--output_json",
                str(out),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            if args.strict_size:
                cmd.append("--strict_size")
            if args.resize_to_ref:
                cmd.append("--resize_to_ref")
            run_cmd(cmd)
        results["lpips"] = read_json(out).get("summary", {})

    if "clip" in metric_set:
        out = output_dir / "clip.json"
        if args.skip_if_json_exists and out.is_file():
            print(f"[Info] skip clip: found existing {out}")
        else:
            cmd = [
                sys.executable,
                str(script_map["clip"]),
                "--run_dir",
                str(run_dir),
                "--image_dir",
                str(target_dir),
                "--image_prefix",
                args.image_prefix,
                "--image_exts",
                args.image_exts,
                "--device",
                args.device,
                "--model_name",
                args.clip_model_name,
                "--output_json",
                str(out),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            run_cmd(cmd)
        results["clip"] = read_json(out).get("summary", {})

    if "ir" in metric_set:
        out = output_dir / "ir.json"
        if args.skip_if_json_exists and out.is_file():
            print(f"[Info] skip ir: found existing {out}")
        else:
            cmd = [
                sys.executable,
                str(script_map["ir"]),
                "--run_dir",
                str(run_dir),
                "--image_dir",
                str(target_dir),
                "--image_prefix",
                args.image_prefix,
                "--image_exts",
                args.image_exts,
                "--model_name",
                args.ir_model_name,
                "--device",
                args.device,
                "--output_json",
                str(out),
            ]
            if args.ir_download_root is not None:
                cmd.extend(
                    [
                        "--download_root",
                        str(args.ir_download_root.expanduser().resolve()),
                    ]
                )
            if args.strict_missing:
                cmd.append("--strict_missing")
            run_cmd(cmd)
        results["ir"] = read_json(out).get("summary", {})

    final_payload = {
        "config": {
            "run_dir": str(run_dir),
            "ref_dir": str(ref_dir),
            "target_dir": str(target_dir),
            "metrics": sorted(metric_set),
            "image_prefix": args.image_prefix,
            "image_exts": args.image_exts,
            "strict_missing": args.strict_missing,
            "strict_size": args.strict_size,
            "resize_to_ref": args.resize_to_ref,
            "device": args.device,
            "lpips_net": args.lpips_net,
            "clip_model_name": args.clip_model_name,
            "ir_model_name": args.ir_model_name,
            "skip_if_json_exists": args.skip_if_json_exists,
        },
        "summary": results,
    }

    summary_path = output_dir / "summary.json"
    if args.skip_if_json_exists and summary_path.is_file():
        print(f"[Info] skip writing summary: found existing {summary_path}")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print("\n=== Unified Evaluation Summary ===")
    for metric_name in sorted(results):
        print(f"{metric_name}: {results[metric_name]}")
    print(f"[Info] summary written to: {summary_path}")


if __name__ == "__main__":
    main()
