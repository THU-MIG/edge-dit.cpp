"""External GPU memory monitor based on nvidia-smi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import shutil
import subprocess
import threading
import time


@dataclass
class GpuSample:
    timestamp_s: float
    index: int
    memory_used_mib: int


class NvidiaSmiMonitor:
    def __init__(
        self,
        output_csv: Path,
        interval_s: float = 0.2,
        visible_devices: list[int] | None = None,
    ) -> None:
        self.output_csv = output_csv
        self.interval_s = interval_s
        self.visible_devices = visible_devices
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.available():
            raise RuntimeError("nvidia-smi is not available")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.write_csv()

    def peak_mib(self) -> int | None:
        if not self.samples:
            return None
        return max(sample.memory_used_mib for sample in self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._collect_once()
            time.sleep(self.interval_s)

    def _collect_once(self) -> None:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        now = time.time()
        for line in output.splitlines():
            if not line.strip():
                continue
            index_s, used_s = [part.strip() for part in line.split(",", 1)]
            index = int(index_s)
            if self.visible_devices is not None and index not in self.visible_devices:
                continue
            self.samples.append(
                GpuSample(
                    timestamp_s=now,
                    index=index,
                    memory_used_mib=int(used_s),
                )
            )

    def write_csv(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "gpu_index", "memory_used_mib"])
            for sample in self.samples:
                writer.writerow([sample.timestamp_s, sample.index, sample.memory_used_mib])


def parse_visible_devices(value: str | None) -> list[int] | None:
    if not value:
        return None
    devices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            devices.append(int(item))
    return devices or None


def segment_peaks(
    csv_path: Path,
    boundaries: dict[str, list[float]],
    gpu_index: int | None = None,
) -> dict[str, int | None]:
    """Compute the peak memory (MiB) inside each stage window.

    ``boundaries`` maps a stage name to ``[t_begin, t_end]`` epoch seconds, on
    the same clock as ``gpu_memory.csv`` (``time.time()``). For every stage we
    scan the CSV rows whose ``timestamp_s`` falls within the window and take the
    maximum ``memory_used_mib``. When several GPU indices are present the peak is
    taken across all of them unless ``gpu_index`` is supplied.
    """

    result: dict[str, int | None] = {stage: None for stage in boundaries}
    if not boundaries or not Path(csv_path).exists():
        return result

    rows: list[tuple[float, int, int]] = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return result
        for raw in reader:
            if len(raw) < 3:
                continue
            try:
                ts = float(raw[0])
                idx = int(raw[1])
                used = int(raw[2])
            except (ValueError, TypeError):
                continue
            rows.append((ts, idx, used))

    if not rows:
        return result

    for stage, window in boundaries.items():
        if not window or len(window) < 2:
            continue
        t_begin, t_end = float(window[0]), float(window[1])
        if t_end < t_begin:
            t_begin, t_end = t_end, t_begin
        peak: int | None = None
        for ts, idx, used in rows:
            if ts < t_begin or ts > t_end:
                continue
            if gpu_index is not None and idx != gpu_index:
                continue
            peak = used if peak is None else max(peak, used)
        result[stage] = peak
    return result

