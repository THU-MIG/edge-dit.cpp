#!/usr/bin/env python3
"""Readable summary tables from a benchmark results tree.

Complements make_matrix_tables.py (which emits the wide 20-column per-prompt
detail table). This emits narrow, mean-only summary tables that are easy to read:

    summary-speed.md    DiT-pure-sampling ms + end-to-end ms
    summary-memory.md   peak + per-component VRAM
    summary-quality.md  CLIP/aesthetic/IR + PSNR/SSIM/LPIPS-vs-FP16
    summary-all.md       one compact mean table (core columns)

Rows are one mean per (workload, system, precision, budget) group. Reuses
make_matrix_tables' Row + discover_runs for reading, so field semantics match
the detail table exactly.

Usage:
    python benchmark/scripts/summarize.py --results-root benchmark/results/<name> \
                                          --output-dir  benchmark/reports/<name>
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from make_matrix_tables import Row, TASK_COLS  # same-dir reuse of the reader + per-task column specs


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.fmean(vals) if vals else None


def fmt(v, nd=1):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


class Group:
    """Mean over the prompts of one (workload, system, precision, budget, cache)."""
    def __init__(self, rows):
        self.rows = rows
        r0 = rows[0]
        self.workload, self.system, self.precision, self.budget, self.cache = (
            r0.workload, r0.system, r0.precision, r0.budget, r0.cache)
        self.n = len(rows)
        self.ok = sum(1 for r in rows if r.status == "success")
        for attr in ("dit_ms", "e2e_ms", "te_ms", "vae_ms", "peak_vram",
                     "te_vram", "dit_vram", "vae_vram", "clip", "aesthetic", "ir",
                     "psnr", "ssim", "lpips", "dclip", "keep_ssim", "keep_lpips",
                     "tmp_lpips", "tmp_ssim", "tmp_lpips_flk"):
            setattr(self, attr, mean(getattr(r, attr) for r in rows))


def collect(root: Path):
    groups = {}
    for rj in root.rglob("result.json"):
        # skip calibration runs (they live under cache/_calib, not real benchmark runs)
        if "_calib" in rj.parts:
            continue
        r = Row(rj.parent)
        groups.setdefault((r.workload, r.system, r.precision, r.budget, r.cache), []).append(r)
    out = [Group(rs) for rs in groups.values()]
    # stable order: workload, then system, then precision, then budget, then cache
    out.sort(key=lambda g: (g.workload, g.system, g.precision, g.budget, g.cache))
    return out


def _table(groups, headers, cols):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for g in groups:
        lines.append("| " + " | ".join(cols(g)) + " |")
    return "\n".join(lines)


def by_workload(groups):
    wls = []
    for g in groups:
        if g.workload not in wls:
            wls.append(g.workload)
    return wls


def write_speed(groups, out: Path):
    body = ["# Speed summary (mean, unit ms)\n",
            "> For inference speed look at **DiT sampling ms** (reliable). End-to-end excludes model load (load-once boundary), but sd.cpp quantized tiers fold on-the-fly conversion into DiT, so those are inflated and not comparable across systems.\n"]
    for wl in by_workload(groups):
        body.append(f"\n## {wl}\n")
        gs = [g for g in groups if g.workload == wl]
        body.append(_table(gs, ["system", "precision", "budget", "cache", "DiT sampling ms", "end-to-end ms", "TE_ms", "VAE_ms"],
                            lambda g: [g.system, g.precision, g.budget, g.cache,
                                       fmt(g.dit_ms), fmt(g.e2e_ms), fmt(g.te_ms), fmt(g.vae_ms)]))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_memory(groups, out: Path):
    body = ["# VRAM summary (mean, unit MiB)\n",
            "> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.\n"]
    for wl in by_workload(groups):
        body.append(f"\n## {wl}\n")
        gs = [g for g in groups if g.workload == wl]
        body.append(_table(gs, ["system", "precision", "budget", "cache", "peak VRAM", "TE VRAM", "DiT VRAM", "VAE VRAM"],
                            lambda g: [g.system, g.precision, g.budget, g.cache,
                                       fmt(g.peak_vram, 0), fmt(g.te_vram, 0), fmt(g.dit_vram, 0), fmt(g.vae_vram, 0)]))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_quality(groups, out: Path):
    body = ["# Quality summary (mean)\n",
            "> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.\n"]
    for wl in by_workload(groups):
        gs = [g for g in groups if g.workload == wl]
        task = gs[0].rows[0].task if gs and gs[0].rows else "text-to-image"
        cols = TASK_COLS.get(task, TASK_COLS["text-to-image"])
        body.append(f"\n## {wl}  ({task})\n")
        headers = ["system", "precision", "budget", "cache"] + [h for h, _, _ in cols]
        body.append(_table(gs, headers,
                            lambda g, cols=cols: [g.system, g.precision, g.budget, g.cache]
                            + [fmt(getattr(g, attr), prec) for _, attr, prec in cols]))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_all(groups, out: Path):
    body = ["# Summary table (mean, core columns)\n",
            "> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.\n"]
    speed_vram = ["DiTms", "end-to-end ms", "peak VRAM"]
    for wl in by_workload(groups):
        gs = [g for g in groups if g.workload == wl]
        task = gs[0].rows[0].task if gs and gs[0].rows else "text-to-image"
        cols = TASK_COLS.get(task, TASK_COLS["text-to-image"])
        body.append(f"\n## {wl}  ({task})\n")
        headers = ["system", "precision", "budget", "cache"] + speed_vram + [h for h, _, _ in cols]
        body.append(_table(gs, headers,
                            lambda g, cols=cols: [g.system, g.precision, g.budget, g.cache,
                                                  fmt(g.dit_ms), fmt(g.e2e_ms), fmt(g.peak_vram, 0)]
                            + [fmt(getattr(g, attr), prec) for _, attr, prec in cols]))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Readable summary tables from results tree.")
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    groups = collect(args.results_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_speed(groups, args.output_dir / "summary-speed.md")
    write_memory(groups, args.output_dir / "summary-memory.md")
    write_quality(groups, args.output_dir / "summary-quality.md")
    write_all(groups, args.output_dir / "summary-all.md")
    print(f"[summarize] {len(groups)} groups -> 4 summary tables in {args.output_dir}")


if __name__ == "__main__":
    main()
