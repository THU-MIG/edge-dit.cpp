#!/usr/bin/env python3
"""Assert cache regression floors against an aggregate benchmark summary.

This is the pass/fail gate the benchmark harness lacks: `aggregate.py` produces
numbers, but nothing asserts them against a baseline. This script reads an
aggregate summary (the output of `benchmark/analysis/aggregate.py`) plus a
baseline spec, and fails (exit 1) if any cache run regressed.

It keys every check off the REQUESTED cache mode -- `run_options.cache` in each
summary row -- not the mode the binary logged. That is deliberate: the headline
regression this guards against is "silent full-compute", where a method is
reported as enabled but quietly reuses zero steps. Keying off the logged mode
would make such a row invisible (it logs nothing); keying off the request makes
its collapsed `cache_steps_reused` an assertable failure.

Two floors per method (see benchmark/specs/cache_regression_baseline.yaml):
  - min_reuse_ratio       steps_reused/total_steps must be >= floor.
  - min_psnr_vs_baseline  PSNR vs the cache_off reference must be >= floor.

PSNR is produced by the separate quality-eval pass (evaluate_cache_quality.py),
so a summary may legitimately lack it. Missing PSNR is a WARNING by default and
a FAILURE under --strict-quality (use that in CI once quality eval is wired in).

Usage:
    python3 -m benchmark.analysis.assert_cache_regression <summary.json>
    python3 -m benchmark.analysis.assert_cache_regression <summary.json> \
        --baseline benchmark/specs/cache_regression_baseline.yaml --strict-quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASELINE = Path("benchmark/specs/cache_regression_baseline.yaml")

# Requested --cache value -> substring the C++ policy uses in its summary line.
# Only used for reporting clarity; the gate itself keys off the requested value.
OFF_VALUES = {"off", "none", "disabled", "", None}


class Check:
    """One evaluated row: its verdicts and human-readable detail lines."""

    def __init__(self, mode: str, model: str, workload: str, scenario: str) -> None:
        self.mode = mode
        self.model = model
        self.workload = workload
        self.scenario = scenario
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    @property
    def label(self) -> str:
        return f"{self.mode} / {self.model} [{self.workload}:{self.scenario}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="aggregate summary JSON")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"baseline spec YAML (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="treat a missing PSNR-vs-baseline as a failure instead of a warning",
    )
    parser.add_argument(
        "--include-workload",
        action="append",
        default=[],
        help="only gate rows for this workload id (repeatable)",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(args.baseline.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        print(f"error: baseline {args.baseline} is not a mapping")
        return 2

    rows = summary.get("results", [])
    include = set(args.include_workload)

    checks: list[Check] = []
    for row in rows:
        if include and row.get("workload") not in include:
            continue
        requested = requested_cache_mode(row)
        if requested in OFF_VALUES:
            continue  # the cache_off reference row is not gated
        if row.get("status") != "success":
            # A failed/skipped run is a harness/env problem, not a cache
            # regression per se, but silently ignoring it would let a crash
            # masquerade as "no regression". Surface it as a failure.
            check = Check(requested, model_of(row), row.get("workload", "?"), row.get("scenario", "?"))
            check.failures.append(f"run status is {row.get('status')!r}, expected success")
            checks.append(check)
            continue
        checks.append(evaluate_row(row, requested, baseline, strict_quality=args.strict_quality))

    return report(checks, summary_path=args.summary)


def evaluate_row(
    row: dict[str, Any],
    requested: str,
    baseline: dict[str, Any],
    *,
    strict_quality: bool,
) -> Check:
    model = model_of(row)
    check = Check(requested, model, row.get("workload", "?"), row.get("scenario", "?"))

    reused = row.get("cache_steps_reused")
    total = row.get("cache_total_steps")
    ratio = row.get("cache_reuse_ratio")
    psnr = row.get("psnr")

    if is_expected_disabled(requested, model, baseline):
        # Disabled by design -> expect no reuse, and do NOT apply floors.
        if reused:
            check.failures.append(
                f"expected disabled on {model} but reused {reused}/{total} steps"
            )
        else:
            check.notes.append("disabled by design (expected zero reuse) -- floors skipped")
        return check

    floors = method_floors(requested, baseline)
    min_ratio = floors["min_reuse_ratio"]
    min_psnr = floors["min_psnr_vs_baseline"]

    # ---- reuse floor: the silent-full-compute guard ----
    if reused is None or total is None or ratio is None:
        check.failures.append(
            "no cache summary in row (cache_steps_reused is null) -- "
            "method requested but reported nothing; likely silent full-compute"
        )
    elif total == 0:
        check.failures.append("cache_total_steps is 0 -- cannot evaluate reuse")
    elif ratio < min_ratio:
        check.failures.append(
            f"reuse ratio {ratio:.3f} ({reused}/{total}) below floor {min_ratio:.3f}"
        )
    else:
        check.notes.append(f"reuse {reused}/{total} ({ratio:.3f} >= {min_ratio:.3f})")

    # ---- quality floor: PSNR vs cache_off reference ----
    if psnr is None:
        msg = f"no PSNR-vs-baseline in row (floor {min_psnr:.1f} dB not checked)"
        if strict_quality:
            check.failures.append(msg + " -- --strict-quality is set")
        else:
            check.warnings.append(msg + " -- run evaluate_cache_quality.py to populate it")
    elif psnr < min_psnr:
        check.failures.append(f"PSNR {psnr:.2f} dB below floor {min_psnr:.1f} dB")
    else:
        check.notes.append(f"PSNR {psnr:.2f} dB (>= {min_psnr:.1f})")

    return check


def requested_cache_mode(row: dict[str, Any]) -> Any:
    run_options = row.get("run_options")
    if isinstance(run_options, dict):
        cache = run_options.get("cache")
        if cache is not None:
            return str(cache).strip().lower()
    # Fall back to the logged mode if run_options is unavailable (older rows).
    logged = row.get("cache_mode")
    return str(logged).strip().lower() if logged else None


def model_of(row: dict[str, Any]) -> str:
    return str(row.get("model") or "?")


def method_floors(mode: str, baseline: dict[str, Any]) -> dict[str, float]:
    defaults = baseline.get("defaults", {}) or {}
    methods = baseline.get("methods", {}) or {}
    entry = methods.get(mode, {}) or {}
    return {
        "min_reuse_ratio": float(
            entry.get("min_reuse_ratio", defaults.get("min_reuse_ratio", 0.0))
        ),
        "min_psnr_vs_baseline": float(
            entry.get("min_psnr_vs_baseline", defaults.get("min_psnr_vs_baseline", 0.0))
        ),
    }


def is_expected_disabled(mode: str, model: str, baseline: dict[str, Any]) -> bool:
    disabled = baseline.get("expect_disabled", {}) or {}
    substrings = disabled.get(mode)
    if not isinstance(substrings, list):
        return False
    return any(isinstance(s, str) and s in model for s in substrings)


def report(checks: list[Check], *, summary_path: Path) -> int:
    total_fail = sum(1 for c in checks if c.failures)
    total_warn = sum(1 for c in checks if c.warnings)

    print(f"cache regression gate: {summary_path}")
    print(f"  evaluated {len(checks)} cache run(s)\n")

    for check in checks:
        if check.failures:
            status = "FAIL"
        elif check.warnings:
            status = "WARN"
        else:
            status = "PASS"
        print(f"  [{status}] {check.label}")
        for line in check.failures:
            print(f"      x {line}")
        for line in check.warnings:
            print(f"      ! {line}")
        for line in check.notes:
            print(f"      - {line}")

    print()
    if total_fail:
        print(f"RESULT: FAIL ({total_fail} failing, {total_warn} warning)")
        return 1
    if not checks:
        # Nothing to gate is itself suspicious -- a cache suite that produced no
        # gateable rows almost certainly did not run what was intended.
        print("RESULT: FAIL (no cache runs found to gate)")
        return 1
    print(f"RESULT: PASS ({len(checks)} ok, {total_warn} warning)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
