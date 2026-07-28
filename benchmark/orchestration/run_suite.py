#!/usr/bin/env python3
"""Expand or execute a benchmark suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

try:
    from .config import expand_runs, load_suite_graph, load_yaml, repo_root
except ImportError:  # pragma: no cover - direct script execution
    from config import expand_runs, load_suite_graph, load_yaml, repo_root

sys.path.insert(0, str(repo_root()))
from benchmark.runners import RUNNERS  # noqa: E402
from benchmark.runners.base import print_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Preview commands only")
    parser.add_argument("--execute", action="store_true", help="Execute benchmark commands")
    parser.add_argument("--preflight", action="store_true", help="Run runner preflight checks")
    parser.add_argument("--resume", action="store_true", help="Skip matching successful run directories")
    parser.add_argument("--output-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--systems", nargs="*", help="Optional system_id filter")
    parser.add_argument("--max-runs", type=int, help="Execute or preview at most N runs")
    parser.add_argument("--warmup-runs", type=int, help="Override suite warmup count")
    parser.add_argument("--measured-runs", type=int, help="Override suite measured count")
    parser.add_argument(
        "--force-external-update",
        action="store_true",
        help="Allow configured external baselines to reset to latest origin/main",
    )
    parser.add_argument(
        "--cache-gate",
        action="store_true",
        help="Force the cache-regression gate on after execution",
    )
    parser.add_argument(
        "--no-cache-gate",
        action="store_true",
        help="Force the cache-regression gate off, overriding the suite",
    )
    args = parser.parse_args()

    if args.dry_run == args.execute:
        print("choose exactly one of --dry-run or --execute", file=sys.stderr)
        return 2

    if args.cache_gate and args.no_cache_gate:
        print("choose at most one of --cache-gate or --no-cache-gate", file=sys.stderr)
        return 2

    root = repo_root()
    graph = load_suite_graph(args.suite.resolve(), args.site.resolve())
    runs = filter_runs(expand_runs(graph), args.systems)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    output = {
        "suite_id": graph["suite"]["suite_id"],
        "mode": graph["suite"].get("mode"),
        "runs": [],
        "preflight": [],
    }

    runners = {}
    for system_id, item in graph["systems"].items():
        config = item["config"]
        runner_name = config["runner"]
        runner_cls = RUNNERS[runner_name]
        runner = runner_cls(config, graph["site"], root)
        runners[system_id] = runner
        if args.preflight and system_id in active_system_ids(runs):
            output["preflight"].append(runner.preflight().to_dict())

    if args.execute:
        external_error = external_update_error(graph, runs, args.force_external_update)
        if external_error:
            print(external_error, file=sys.stderr)
            return 2

        failed = False
        warmup_runs, measured_runs = run_counts(graph, args.warmup_runs, args.measured_runs)
        if warmup_runs < 0 or measured_runs < 0:
            print("--warmup-runs and --measured-runs must be non-negative", file=sys.stderr)
            return 2
        for run in runs:
            runner = runners[run["system_id"]]
            workload = graph["workloads"][run["workload_id"]]["config"]
            run_id = make_run_id(run)
            output_dir = (
                args.output_root
                / graph["suite"]["suite_id"]
                / run["system_id"]
                / run["workload_id"]
                / run_id
            )
            if args.resume:
                completed = find_completed_result(
                    args.output_root,
                    graph["suite"]["suite_id"],
                    run,
                )
                if completed is not None:
                    output["runs"].append(
                        {
                            **run,
                            "run_id": completed.name,
                            "output_dir": str(completed),
                            "status": "already_completed",
                        }
                    )
                    continue

            preflight = runner.preflight()
            if not preflight.ok:
                result = runner.write_skipped_result(
                    workload,
                    run["gpu_count"],
                    run["parallel_mode"],
                    output_dir,
                    warmup_runs,
                    measured_runs,
                    run["run_options"],
                    preflight.messages,
                    run["scenario_id"],
                )
                output["runs"].append(
                    {
                        **run,
                        "run_id": run_id,
                        "output_dir": str(output_dir),
                        "status": result["status"],
                        "reason": preflight.messages,
                    }
                )
                failed = True
                continue

            result = runner.execute(
                workload,
                run["gpu_count"],
                run["parallel_mode"],
                output_dir,
                warmup_runs,
                measured_runs,
                run["run_options"],
                args.force_external_update,
                run["scenario_id"],
            )
            output["runs"].append(
                {
                    **run,
                    "run_id": run_id,
                    "output_dir": str(output_dir),
                    "status": result["status"],
                }
            )
            if result["status"] != "success":
                failed = True

        if gate_enabled(graph, args.cache_gate, args.no_cache_gate):
            try:
                from .cache_gate import run_cache_gate
            except ImportError:  # pragma: no cover - direct script execution
                from cache_gate import run_cache_gate

            print("=== cache regression gate ===", file=sys.stderr)
            gate_rc, _ = run_cache_gate(
                output_root=args.output_root,
                suite_id=graph["suite"]["suite_id"],
                repo=root,
                strict_quality=quality_gate_enabled(graph),
            )
            if gate_rc != 0:
                failed = True

        print_json(output)
        return 1 if failed else 0

    for run in runs:
        runner = runners[run["system_id"]]
        workload = graph["workloads"][run["workload_id"]]["config"]
        warmup_runs, measured_runs = run_counts(graph, args.warmup_runs, args.measured_runs)
        output["runs"].append(
            runner.describe_dry_run(
                workload,
                run["gpu_count"],
                run["parallel_mode"],
                run["run_options"],
                warmup_runs,
                measured_runs,
            )
        )

    print_json(output)
    return 0


def active_system_ids(runs: list[dict[str, object]]) -> set[str]:
    return {str(run["system_id"]) for run in runs}


def gate_enabled(
    graph: dict[str, object],
    cli_on: bool,
    cli_off: bool,
) -> bool:
    """Resolve whether the cache-regression gate should run.

    Precedence: an explicit CLI flag wins; otherwise the suite opts in via
    ``reporting.cache_regression_gate``; default off.
    """
    if cli_off:
        return False
    if cli_on:
        return True
    reporting = graph["suite"].get("reporting", {}) or {}
    return bool(reporting.get("cache_regression_gate", False))


def quality_gate_enabled(graph: dict[str, object]) -> bool:
    """Whether the cache gate should also enforce the PSNR quality floor.

    Gated on the suite declaring ``reporting.require_quality_reference``: only a
    suite that runs a no-cache reference can compute PSNR-vs-baseline, so only
    those enable strict quality. Suites without a reference keep the reuse-only
    gate (missing PSNR stays a warning) rather than failing spuriously.
    """
    reporting = graph["suite"].get("reporting", {}) or {}
    return bool(reporting.get("require_quality_reference"))


def filter_runs(
    runs: list[dict[str, object]],
    systems: list[str] | None,
) -> list[dict[str, object]]:
    if not systems:
        return runs
    wanted = set(systems)
    return [run for run in runs if run["system_id"] in wanted]


def run_counts(
    graph: dict[str, object],
    warmup_override: int | None,
    measured_override: int | None,
) -> tuple[int, int]:
    defaults = graph["suite"].get("defaults", {})
    warmup_runs = (
        warmup_override
        if warmup_override is not None
        else int(defaults.get("warmup_runs", 0))
    )
    measured_runs = (
        measured_override
        if measured_override is not None
        else int(defaults.get("measured_runs", 1))
    )
    return warmup_runs, measured_runs


def external_update_error(
    graph: dict[str, object],
    runs: list[dict[str, object]],
    force_external_update: bool,
) -> str | None:
    if force_external_update:
        return None
    systems = graph["systems"]
    for run in runs:
        system = systems[run["system_id"]]["config"]
        if system.get("repo", {}).get("commit_policy") == "force_latest_origin_main":
            return (
                f"{run['system_id']} is configured to force-sync an external baseline; "
                "rerun with --force-external-update or filter it out with --systems"
            )
    return None


def make_run_id(run: dict[str, object]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = run["parallel_mode"] or "single"
    scenario = run.get("scenario_id", "default")
    raw = f"{timestamp}-gpu{run['gpu_count']}-{mode}-{scenario}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def find_completed_result(
    output_root: Path,
    suite_id: str,
    run: dict[str, object],
) -> Path | None:
    base = output_root / suite_id / str(run["system_id"]) / str(run["workload_id"])
    if not base.exists():
        return None
    for result_json in sorted(base.glob("*/result.json")):
        if not result_matches(result_json.parent, run):
            continue
        try:
            result = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if result.get("status") == "success":
            return result_json.parent
    return None


def result_matches(result_dir: Path, run: dict[str, object]) -> bool:
    config_path = result_dir / "config.resolved.yaml"
    if not config_path.exists():
        return False
    try:
        config = load_yaml(config_path)
    except Exception:  # noqa: BLE001
        return False
    system = config.get("system", {})
    workload = config.get("workload", {})
    return (
        system.get("system_id") == run["system_id"]
        and workload.get("workload_id") == run["workload_id"]
        and config.get("gpu_count") == run["gpu_count"]
        and config.get("parallel_mode") == run["parallel_mode"]
        and config.get("scenario_id", run["scenario_id"]) == run["scenario_id"]
        and config.get("run_options", {}) == run.get("run_options", {})
    )


if __name__ == "__main__":
    raise SystemExit(main())
