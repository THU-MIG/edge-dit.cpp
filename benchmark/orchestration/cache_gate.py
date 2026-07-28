#!/usr/bin/env python3
"""Aggregate a completed suite's results and run the cache-regression gate.

This closes the loop between a suite run and the standalone gate: it invokes
``benchmark.analysis.aggregate`` to build a summary JSON from the per-run
``result.json`` files, then ``benchmark.analysis.assert_cache_regression`` to
assert reuse-ratio floors and catch silent full-compute regressions. Both are
run as subprocesses (mirroring benchmark/scripts/run_readme_main_table.sh) so
their argparse/exit-code semantics stay intact and aggregate's --output keeps
run_suite's stdout JSON clean.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

try:
    from .config import load_yaml, repo_root
except ImportError:  # pragma: no cover - direct script execution
    from config import load_yaml, repo_root


# Requested --cache values that mark the no-cache reference run.
_OFF_VALUES = {"off", "none", "disabled", ""}


def run_cache_gate(
    *,
    output_root: Path,
    suite_id: str,
    repo: Path,
    baseline: Path | None = None,
    strict_quality: bool = False,
) -> tuple[int, Path]:
    """Aggregate ``output_root/suite_id`` and run the regression gate over it.

    When ``strict_quality`` is set, first compute PSNR of every cache run against
    its no-cache reference and write it into each ``result.json`` so the gate's
    quality floor is actually enforced (otherwise PSNR is absent and the floor is
    silently downgraded to a warning). The gate is then invoked with
    ``--strict-quality`` so a run that still lacks PSNR fails rather than passes.

    Returns ``(exit_code, summary_path)``:
      0 -- gate passed
      1 -- gate failure (reuse below floor / silent full-compute / no gateable rows)
      2 -- aggregation failed or summary was never written
    """
    suite_dir = output_root / suite_id
    summary = suite_dir / "summary.json"

    if strict_quality:
        populate_psnr_vs_reference(suite_dir)

    aggregate_cmd = [
        sys.executable,
        "-m",
        "benchmark.analysis.aggregate",
        "--results-dir",
        str(suite_dir),
        "--suite-id",
        suite_id,
        "--output",
        str(summary),
    ]
    aggregate_rc = subprocess.run(aggregate_cmd, cwd=str(repo)).returncode
    if aggregate_rc != 0:
        print(
            f"cache gate: aggregate failed (exit {aggregate_rc}) for {suite_dir}",
            file=sys.stderr,
        )
        return 2, summary
    if not summary.exists():
        print(
            f"cache gate: aggregate produced no summary at {summary}",
            file=sys.stderr,
        )
        return 2, summary

    gate_cmd = [
        sys.executable,
        "-m",
        "benchmark.analysis.assert_cache_regression",
        str(summary),
    ]
    if baseline is not None:
        gate_cmd += ["--baseline", str(baseline)]
    if strict_quality:
        gate_cmd.append("--strict-quality")
    # Route the gate's human-readable report to stderr so it stays visible without
    # polluting run_suite's machine-readable stdout JSON.
    gate_rc = subprocess.run(gate_cmd, cwd=str(repo), stdout=sys.stderr).returncode
    return gate_rc, summary


def populate_psnr_vs_reference(suite_dir: Path) -> None:
    """Compute PSNR of each cache run vs the no-cache reference and write it into
    each run's ``result.json`` (``quality.psnr``), so aggregate/the gate can
    enforce the quality floor.

    Groups successful runs under ``suite_dir`` by requested cache mode (the
    ``scenario_id`` in each run's ``config.resolved.yaml``). The reference is the
    mode in ``_OFF_VALUES``. This is intentionally fail-soft: a missing reference
    or unreadable image leaves ``psnr`` absent, which under ``--strict-quality``
    the gate turns into a failure rather than a false pass.
    """
    runs = _scan_runs(suite_dir)
    reference = next((r for r in runs if _is_off(r["cache_mode"])), None)
    if reference is None:
        print(
            f"cache gate: no no-cache reference run under {suite_dir}; "
            "PSNR not computed (quality floor will fail under --strict-quality)",
            file=sys.stderr,
        )
        return
    ref_image = reference["image"]
    if ref_image is None or not ref_image.is_file():
        print(
            f"cache gate: reference image missing ({ref_image}); PSNR not computed",
            file=sys.stderr,
        )
        return

    ref_arr = _load_rgb(ref_image)
    for run in runs:
        if run is reference or _is_off(run["cache_mode"]):
            continue
        image = run["image"]
        if image is None or not image.is_file():
            print(f"cache gate: no image for {run['result_dir'].name}; PSNR skipped",
                  file=sys.stderr)
            continue
        tgt_arr = _load_rgb(image)
        if tgt_arr.shape != ref_arr.shape:
            print(
                f"cache gate: size mismatch for {run['result_dir'].name} "
                f"({tgt_arr.shape} vs ref {ref_arr.shape}); PSNR skipped",
                file=sys.stderr,
            )
            continue
        _write_psnr(run["result_dir"], _psnr(ref_arr, tgt_arr))


def _scan_runs(suite_dir: Path) -> list[dict]:
    """Successful runs under ``suite_dir``: cache mode + reference image path."""
    runs: list[dict] = []
    for result_path in sorted(suite_dir.rglob("result.json")):
        result_dir = result_path.parent
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or result.get("status") != "success":
            continue
        runs.append(
            {
                "result_dir": result_dir,
                "cache_mode": _requested_cache_mode(result_dir),
                "image": _sample_image(result_dir),
            }
        )
    return runs


def _requested_cache_mode(result_dir: Path) -> str | None:
    """Requested cache mode from config.resolved.yaml (run_options.cache, else
    scenario_id). Lower-cased; None if unreadable."""
    config_path = result_dir / "config.resolved.yaml"
    if not config_path.exists():
        return None
    try:
        config = load_yaml(config_path)
    except Exception:  # noqa: BLE001
        return None
    run_options = config.get("run_options")
    if isinstance(run_options, dict) and run_options.get("cache") is not None:
        return str(run_options["cache"]).strip().lower()
    scenario = config.get("scenario_id")
    return str(scenario).strip().lower() if scenario else None


def _sample_image(result_dir: Path) -> Path | None:
    """The measured sample image via runner_metrics.json's sample_output_dir."""
    metrics_path = result_dir / "runner_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sample_dir = metrics.get("sample_output_dir")
    if not isinstance(sample_dir, str):
        return None
    candidate = Path(sample_dir) / "measured_000.png"
    return candidate if candidate.is_file() else None


def _is_off(cache_mode: str | None) -> bool:
    return cache_mode is None or cache_mode in _OFF_VALUES


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def _psnr(ref, tgt) -> float:
    """Same formula as benchmark/evaluation/single_metric/cal_psnr.py."""
    import numpy as np

    mse = float(np.mean((ref - tgt) ** 2))
    if mse < 1e-5:
        return 100.0
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _write_psnr(result_dir: Path, psnr: float) -> None:
    """Write psnr into result.json's quality block (mirrors apply_quality_metrics)."""
    result_path = result_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.setdefault("quality", {})
    result["quality"]["psnr"] = psnr
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_id", help="suite id (subdirectory under --output-root)")
    parser.add_argument("--output-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="baseline spec YAML (defaults to the gate's own default)",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="compute PSNR vs the no-cache reference and enforce the quality floor",
    )
    args = parser.parse_args()
    rc, _ = run_cache_gate(
        output_root=args.output_root,
        suite_id=args.suite_id,
        repo=repo_root(),
        baseline=args.baseline,
        strict_quality=args.strict_quality,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
