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

from pathlib import Path
import subprocess
import sys

try:
    from .config import repo_root
except ImportError:  # pragma: no cover - direct script execution
    from config import repo_root


def run_cache_gate(
    *,
    output_root: Path,
    suite_id: str,
    repo: Path,
    baseline: Path | None = None,
) -> tuple[int, Path]:
    """Aggregate ``output_root/suite_id`` and run the regression gate over it.

    Returns ``(exit_code, summary_path)``:
      0 -- gate passed
      1 -- gate failure (reuse below floor / silent full-compute / no gateable rows)
      2 -- aggregation failed or summary was never written
    """
    suite_dir = output_root / suite_id
    summary = suite_dir / "summary.json"

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
    # Route the gate's human-readable report to stderr so it stays visible without
    # polluting run_suite's machine-readable stdout JSON.
    gate_rc = subprocess.run(gate_cmd, cwd=str(repo), stdout=sys.stderr).returncode
    return gate_rc, summary


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
    args = parser.parse_args()
    rc, _ = run_cache_gate(
        output_root=args.output_root,
        suite_id=args.suite_id,
        repo=repo_root(),
        baseline=args.baseline,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
