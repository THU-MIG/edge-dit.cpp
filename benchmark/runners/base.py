"""Shared benchmark runner abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import time
from typing import Any

import yaml

from benchmark.measurement.gpu_monitor import (
    NvidiaSmiMonitor,
    parse_visible_devices,
    segment_peaks,
)
from benchmark.measurement.environment import collect_environment
from benchmark.measurement.process_monitor import process_tree_rss_mib
from benchmark.measurement.timer import summarize_ms


@dataclass
class PreflightResult:
    system_id: str
    ok: bool
    messages: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "ok": self.ok,
            "messages": self.messages,
            "metadata": self.metadata,
        }


class BenchmarkRunner:
    """Base class for benchmark system adapters."""

    def __init__(
        self,
        system_config: dict[str, Any],
        site_config: dict[str, Any],
        repo_root: Path,
    ) -> None:
        self.system_config = system_config
        self.site_config = site_config
        self.repo_root = repo_root
        self.system_id = system_config["system_id"]

    def resolve_path(self, ref: str | None) -> Path | None:
        if not ref:
            return None
        paths = self.site_config.get("paths", {})
        value = paths.get(ref)
        if value is None:
            return None
        return Path(value)

    def prompt_text(
        self,
        workload: dict[str, Any],
        run_options: dict[str, Any] | None = None,
    ) -> str:
        run_options = run_options or {}
        prompt = run_options.get("prompt")
        if isinstance(prompt, str):
            return prompt
        prompt_id = run_options.get("prompt_id")
        if isinstance(prompt_id, str):
            prompt_set = workload.get("resolved_prompt_set", {})
            if isinstance(prompt_set, dict):
                item = prompt_set.get(prompt_id)
                if isinstance(item, dict):
                    return str(item.get("prompt", ""))
            raise NotImplementedError(
                f"prompt_id {prompt_id!r} is not available in workload prompt set"
            )
        return str(workload.get("resolved_prompt", {}).get("prompt", ""))

    def git_commit(self, repo: Path) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def git_dirty(self, repo: Path) -> bool | None:
        try:
            status = subprocess.check_output(
                ["git", "-C", str(repo), "status", "--short"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return bool(status.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def package_importable(self, module_name: str) -> bool:
        code = f"import {module_name}"
        return subprocess.run(
            ["python3", "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0

    def command_available(self, command: str) -> bool:
        return shutil.which(command) is not None

    def preflight(self) -> PreflightResult:
        return PreflightResult(system_id=self.system_id, ok=True)

    def prepare_for_execution(self, force_external_update: bool = False) -> list[str]:
        return []

    def extra_env(self, gpu_count: int) -> dict[str, str]:
        return {}

    def execution_env(self, gpu_count: int) -> dict[str, str]:
        env = os.environ.copy()
        visible_devices = os.environ.get("BENCHMARK_CUDA_VISIBLE_DEVICES")
        if visible_devices:
            devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
            if len(devices) < gpu_count:
                raise ValueError(
                    "BENCHMARK_CUDA_VISIBLE_DEVICES must list at least "
                    f"{gpu_count} devices for this run"
                )
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:gpu_count])
        else:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
        for key, value in self.extra_env(gpu_count).items():
            env[key] = value
        return env

    def build_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def build_execution_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        run_options: dict[str, Any],
        output_dir: Path,
        warmup_runs: int,
        measured_runs: int,
    ) -> list[str]:
        return self.build_command(workload, gpu_count, parallel_mode, run_options)

    def requires_runner_metrics(self) -> bool:
        return False

    def describe_dry_run(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
        warmup_runs: int = 0,
        measured_runs: int = 1,
    ) -> dict[str, Any]:
        run_options = run_options or {}
        try:
            command = self.build_execution_command(
                workload,
                gpu_count,
                parallel_mode,
                run_options,
                Path("."),
                warmup_runs,
                measured_runs,
            )
            unsupported_reason = None
        except NotImplementedError as exc:
            command = []
            unsupported_reason = str(exc) or "command construction is not implemented"
        return {
            "system": self.system_id,
            "workload": workload["workload_id"],
            "gpu_count": gpu_count,
            "parallel_mode": parallel_mode,
            "run_options": run_options,
            "command": command,
            "env": self.extra_env(gpu_count),
            "unsupported_reason": unsupported_reason,
        }

    def execute(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        output_dir: Path,
        warmup_runs: int,
        measured_runs: int,
        run_options: dict[str, Any] | None = None,
        force_external_update: bool = False,
        scenario_id: str = "default",
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        run_options = run_options or {}
        env = self.execution_env(gpu_count)
        setup_messages: list[str] = []
        command: list[str] = []
        measured_latencies: list[float] = []
        process_samples: list[tuple[float, float | None]] = []
        status = "success"
        error_message = None
        runs: list[dict[str, Any]] = []
        runner_metrics: dict[str, Any] | None = None

        try:
            setup_messages = self.prepare_for_execution(force_external_update)
            command = self.build_execution_command(
                workload,
                gpu_count,
                parallel_mode,
                run_options,
                output_dir,
                warmup_runs,
                measured_runs,
            )
        except NotImplementedError as exc:
            status = "unsupported"
            error_message = str(exc) or "command construction is not implemented"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = str(exc)

        self.write_resolved_config(
            output_dir,
            workload,
            gpu_count,
            parallel_mode,
            run_options,
            warmup_runs,
            measured_runs,
            setup_messages,
            scenario_id,
        )
        (output_dir / "command.txt").write_text(
            shlex.join(command) + "\n",
            encoding="utf-8",
        )
        (output_dir / "environment.json").write_text(
            json.dumps(self.environment_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        monitor = NvidiaSmiMonitor(
            output_dir / "gpu_memory.csv",
            visible_devices=parse_visible_devices(env.get("CUDA_VISIBLE_DEVICES")),
        )
        monitor_started = False
        if status == "success" and monitor.available():
            try:
                monitor.start()
                monitor_started = True
            except Exception:
                monitor_started = False

        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        try:
            if status == "success":
                run = self.run_process(
                    command,
                    env,
                    output_dir,
                    stdout_path,
                    stderr_path,
                    "benchmark",
                    0,
                    process_samples,
                )
                runs.append(run)
                if run["returncode"] != 0:
                    status = "failed"
                    error_message = "benchmark process failed"
        except NotImplementedError as exc:
            status = "unsupported"
            error_message = str(exc)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = str(exc)
        finally:
            if monitor_started:
                monitor.stop()
            else:
                self.write_empty_gpu_csv(output_dir / "gpu_memory.csv")
            self.write_process_csv(output_dir / "process_memory.csv", process_samples)

        if status == "success":
            runner_metrics = self.load_runner_metrics(output_dir)
            if runner_metrics is None:
                if self.requires_runner_metrics():
                    status = "failed"
                    error_message = "runner_metrics.json was not produced"
                else:
                    measured_latencies = [runs[0]["elapsed_ms"]] if runs else []
            else:
                measured_latencies = [
                    float(value)
                    for value in runner_metrics.get("measured_ms", [])
                    if isinstance(value, (int, float))
                ]
                if len(measured_latencies) != measured_runs:
                    status = "failed"
                    error_message = (
                        f"runner_metrics.json has {len(measured_latencies)} measured samples, "
                        f"expected {measured_runs}"
                    )

        timing_summary = summarize_ms(measured_latencies)
        peak_host = max(
            (sample for _, sample in process_samples if sample is not None),
            default=None,
        )
        stage_boundaries = (
            runner_metrics.get("stage_boundaries") if runner_metrics else None
        )
        component_vram: dict[str, int | None] = {}
        if stage_boundaries and isinstance(stage_boundaries, dict):
            try:
                component_vram = segment_peaks(
                    output_dir / "gpu_memory.csv",
                    {
                        stage: window
                        for stage, window in stage_boundaries.items()
                        if isinstance(window, (list, tuple)) and len(window) >= 2
                    },
                )
            except Exception:  # noqa: BLE001
                component_vram = {}
        component_vram_mib = {
            "text_encoder": component_vram.get("text_encoder"),
            "dit": component_vram.get("dit"),
            "vae": component_vram.get("vae"),
        }
        component_weight_bytes = (
            runner_metrics.get("component_weight_bytes") if runner_metrics else None
        )
        cache_summary = runner_metrics.get("cache") if runner_metrics else None
        result = self.result_json(
            output_dir.name,
            workload,
            gpu_count,
            status,
            warmup_runs,
            measured_runs,
            timing_summary,
            measured_latencies,
            runner_metrics.get("load_ms") if runner_metrics else None,
            runner_metrics.get("component_ms", {}) if runner_metrics else {},
            monitor.peak_mib() if monitor_started else None,
            peak_host,
            component_vram_mib,
            component_weight_bytes if isinstance(component_weight_bytes, dict) else None,
            cache_summary if isinstance(cache_summary, dict) else None,
        )
        measurement_boundary = (
            runner_metrics.get("measurement_boundary")
            if runner_metrics
            else "process-level command timing"
        )
        metric_source = runner_metrics.get("metric_source") if runner_metrics else "process_wall_time"
        metrics = {
            "schema_version": 1,
            "status": status,
            "metric_source": metric_source,
            "measurement_boundary": measurement_boundary,
            "quality_metrics_available": False,
        }
        status_json = {
            "status": status,
            "error": error_message,
            "setup_messages": setup_messages,
            "measurement_boundary": measurement_boundary,
        }
        timing_json = {
            "runs": runs,
            "runner_metrics": runner_metrics,
            "steady_state_ms": measured_latencies,
            "summary": timing_summary,
        }

        (output_dir / "status.json").write_text(
            json.dumps(status_json, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "timing.json").write_text(
            json.dumps(timing_json, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    def write_skipped_result(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        output_dir: Path,
        warmup_runs: int,
        measured_runs: int,
        run_options: dict[str, Any] | None,
        reason: list[str],
        scenario_id: str = "default",
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "samples").mkdir(exist_ok=True)
        run_options = run_options or {}
        self.write_resolved_config(
            output_dir,
            workload,
            gpu_count,
            parallel_mode,
            run_options,
            warmup_runs,
            measured_runs,
            reason,
            scenario_id,
        )
        for name in ["command.txt", "stdout.log", "stderr.log"]:
            (output_dir / name).write_text("", encoding="utf-8")
        (output_dir / "environment.json").write_text(
            json.dumps(self.environment_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_empty_gpu_csv(output_dir / "gpu_memory.csv")
        self.write_process_csv(output_dir / "process_memory.csv", [])
        timing_summary = summarize_ms([])
        result = self.result_json(
            output_dir.name,
            workload,
            gpu_count,
            "skipped",
            warmup_runs,
            measured_runs,
            timing_summary,
            [],
            None,
            {},
            None,
            None,
        )
        (output_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "error": "; ".join(reason),
                    "setup_messages": reason,
                    "measurement_boundary": "not executed due to preflight failure",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "timing.json").write_text(
            json.dumps(
                {
                    "runs": [],
                    "steady_state_ms": [],
                    "summary": timing_summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "skipped",
                    "metric_source": "not_executed",
                    "measurement_boundary": "not executed due to preflight failure",
                    "quality_metrics_available": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def run_process(
        self,
        command: list[str],
        env: dict[str, str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        phase: str,
        index: int,
        process_samples: list[tuple[float, float | None]],
    ) -> dict[str, Any]:
        start = time.perf_counter()
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a",
            encoding="utf-8",
        ) as stderr:
            stdout.write(f"\n===== {phase} {index} =====\n")
            stderr.write(f"\n===== {phase} {index} =====\n")
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            peak_rss = None
            last_heartbeat = start
            while process.poll() is None:
                sample = process_tree_rss_mib(process.pid)
                process_samples.append((time.time(), sample))
                if sample is not None:
                    peak_rss = sample if peak_rss is None else max(peak_rss, sample)
                now = time.perf_counter()
                if now - last_heartbeat >= 10.0:
                    elapsed_s = now - start
                    print(
                        f"[benchmark-heartbeat] {self.system_id} {phase} "
                        f"elapsed={elapsed_s:.1f}s pid={process.pid}",
                        flush=True,
                    )
                    last_heartbeat = now
                time.sleep(0.2)
            returncode = process.wait()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "phase": phase,
            "index": index,
            "returncode": returncode,
            "elapsed_ms": elapsed_ms,
            "peak_host_rss_mib": peak_rss,
        }

    def write_resolved_config(
        self,
        output_dir: Path,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        run_options: dict[str, Any],
        warmup_runs: int,
        measured_runs: int,
        setup_messages: list[str],
        scenario_id: str,
    ) -> None:
        data = {
            "system": self.system_config,
            "workload": workload,
            "gpu_count": gpu_count,
            "parallel_mode": parallel_mode,
            "scenario_id": scenario_id,
            "run_options": run_options,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "setup_messages": setup_messages,
        }
        with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def result_json(
        self,
        run_id: str,
        workload: dict[str, Any],
        gpu_count: int,
        status: str,
        warmup_runs: int,
        measured_runs: int,
        timing_summary: dict[str, float | None],
        measured_latencies: list[float],
        load_ms: float | None,
        component_ms: dict[str, Any],
        peak_vram_mib: int | None,
        peak_host_rss_mib: float | None,
        component_vram_mib: dict[str, int | None] | None = None,
        component_weight_bytes: dict[str, Any] | None = None,
        cache_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        component_vram_mib = component_vram_mib or {
            "text_encoder": None,
            "dit": None,
            "vae": None,
        }
        return {
            "run_id": run_id,
            "status": status,
            "system": self.system_id,
            "workload": workload["workload_id"],
            "model": workload["model_name"],
            "task": workload["task"],
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "latency_ms": {
                "load": load_ms,
                "first_generation": measured_latencies[0] if measured_latencies else None,
                "steady_state_median": timing_summary["median"],
                "steady_state_p90": timing_summary["p90"],
                "steady_state_mean": timing_summary["mean"],
                "steady_state_std": timing_summary["std"],
                "coefficient_of_variation": timing_summary["coefficient_of_variation"],
                "text_encoder": component_ms.get("text_encoder"),
                "dit": component_ms.get("dit"),
                "vae": component_ms.get("vae"),
                "per_step_avg": component_ms.get("per_step_avg"),
            },
            "memory": {
                "peak_vram_mib": peak_vram_mib,
                "peak_host_rss_mib": peak_host_rss_mib,
                "component_vram_mib": {
                    "text_encoder": component_vram_mib.get("text_encoder"),
                    "dit": component_vram_mib.get("dit"),
                    "vae": component_vram_mib.get("vae"),
                },
                "component_weight_bytes": component_weight_bytes,
            },
            "parallel": {
                "gpu_count": gpu_count,
                "speedup": None,
                "scaling_efficiency": None,
                "communication_ms": None,
                "all_to_all_ms": None,
                "packing_ms": None,
                "receive_preparation_ms": None,
                "graph_segment_count": None,
            },
            "quality": {
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "clip": None,
                "image_reward": None,
            },
            "cache": {
                "mode": cache_summary.get("mode") if cache_summary else None,
                "steps_reused": cache_summary.get("steps_reused") if cache_summary else None,
                "total_steps": cache_summary.get("total_steps") if cache_summary else None,
                "reuse_ratio": cache_summary.get("reuse_ratio") if cache_summary else None,
            },
        }

    def write_empty_gpu_csv(self, path: Path) -> None:
        path.write_text("timestamp_s,gpu_index,memory_used_mib\n", encoding="utf-8")

    def write_process_csv(
        self,
        path: Path,
        samples: list[tuple[float, float | None]],
    ) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("timestamp_s,host_rss_mib\n")
            for timestamp, rss in samples:
                f.write(f"{timestamp},{'' if rss is None else rss}\n")

    def environment_metadata(self) -> dict[str, Any]:
        repos: dict[str, Path] = {}
        edge_repo = self.resolve_path("edge_dit_repo")
        if edge_repo is not None:
            repos["edge-dit.cpp"] = edge_repo
        repo_ref = self.system_config.get("repo", {}).get("path_ref")
        repo = self.resolve_path(repo_ref)
        if repo is not None:
            repos[self.system_id] = repo
        return collect_environment(repos)

    def load_runner_metrics(self, output_dir: Path) -> dict[str, Any] | None:
        path = output_dir / "runner_metrics.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))
