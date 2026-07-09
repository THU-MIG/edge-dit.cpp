#!/usr/bin/env python3
"""Run unified text-to-video evaluation for PSNR/SSIM/LPIPS.

This script orchestrates existing single-metric scripts in video mode and
aggregates summaries.
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
        description="Unified text-to-video evaluation runner."
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory. target_dir defaults to run_dir when omitted.",
    )
    parser.add_argument(
        "--ref_dir",
        type=Path,
        required=True,
        help="Reference video root directory for PSNR/SSIM/LPIPS.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=None,
        help="Target video root directory. Defaults to run_dir.",
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
        default="psnr,ssim,lpips",
        help="Comma-separated metrics from: psnr,ssim,lpips",
    )
    parser.add_argument(
        "--skip_if_json_exists",
        action="store_true",
        help="Skip metric computation if its output JSON already exists and reuse that file.",
    )

    parser.add_argument(
        "--image_prefix", type=str, default="video_", help="Video filename prefix."
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        default="mp4,mov,mkv,avi,webm,m4v",
        help="Comma-separated video extensions.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail if unmatched indices exist between reference and target.",
    )
    parser.add_argument(
        "--strict_size",
        action="store_true",
        help="Fail if paired video frames have different sizes.",
    )
    parser.add_argument(
        "--strict_video_length",
        action="store_true",
        help="Fail if paired videos have different frame counts.",
    )
    parser.add_argument(
        "--resize_to_ref",
        action="store_true",
        help="Resize target video frame to reference size when dimensions differ.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 or 1 means serial. -1 means os.cpu_count().",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=16,
        help="Batch size for video-frame processing.",
    )
    parser.add_argument(
        "--opencv_threads",
        type=int,
        default=0,
        help="OpenCV internal threads per worker. Usually 0 or 1 with multiprocessing.",
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device used by LPIPS."
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="Backbone network for LPIPS.",
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
    valid_metrics = {"psnr", "ssim", "lpips"}
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
                "--video",
                "--image_prefix",
                args.image_prefix,
                "--video_exts",
                args.video_exts,
                "--output_json",
                str(out),
                "--num_workers",
                str(args.num_workers),
                "--frame_batch_size",
                str(args.frame_batch_size),
                "--opencv_threads",
                str(args.opencv_threads),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            if args.strict_size:
                cmd.append("--strict_size")
            if args.strict_video_length:
                cmd.append("--strict_video_length")
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
                "--video",
                "--image_prefix",
                args.image_prefix,
                "--video_exts",
                args.video_exts,
                "--output_json",
                str(out),
                "--num_workers",
                str(args.num_workers),
                "--opencv_threads",
                str(args.opencv_threads),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            if args.strict_size:
                cmd.append("--strict_size")
            if args.strict_video_length:
                cmd.append("--strict_video_length")
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
                "--video",
                "--image_prefix",
                args.image_prefix,
                "--video_exts",
                args.video_exts,
                "--device",
                args.device,
                "--net",
                args.lpips_net,
                "--output_json",
                str(out),
                "--frame_batch_size",
                str(args.frame_batch_size),
            ]
            if args.strict_missing:
                cmd.append("--strict_missing")
            if args.strict_size:
                cmd.append("--strict_size")
            if args.strict_video_length:
                cmd.append("--strict_video_length")
            if args.resize_to_ref:
                cmd.append("--resize_to_ref")
            run_cmd(cmd)
        results["lpips"] = read_json(out).get("summary", {})

    final_payload = {
        "config": {
            "run_dir": str(run_dir),
            "ref_dir": str(ref_dir),
            "target_dir": str(target_dir),
            "metrics": sorted(metric_set),
            "video": True,
            "image_prefix": args.image_prefix,
            "video_exts": args.video_exts,
            "strict_missing": args.strict_missing,
            "strict_size": args.strict_size,
            "strict_video_length": args.strict_video_length,
            "resize_to_ref": args.resize_to_ref,
            "num_workers": args.num_workers,
            "frame_batch_size": args.frame_batch_size,
            "opencv_threads": args.opencv_threads,
            "device": args.device,
            "lpips_net": args.lpips_net,
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
