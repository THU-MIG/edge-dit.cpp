#!/usr/bin/env python3
"""Aggregate the cross-system matrix into ONE markdown report, all metrics, all
tasks. Reads authoritative fields from each run and prints, per (system,
precision, budget), one detail row per prompt plus a mean row.

Sources per run:
  - config.resolved.yaml : system / task / precision / budget / prompt_id (authoritative)
  - result.json          : latency + component timing + component/peak VRAM
  - eval/quality.json    : all quality metrics (written by eval_all.py)
  - metrics.json         : measurement_boundary (latency semantics annotation)

Latency semantics (IMPORTANT, per user guidance): the end-to-end latency column
is contaminated by one-time on-the-fly weight quantization (q4_K/q8 convert can
be 70-170s) and, for sd.cpp, by layer-by-layer load folded into "sampling
completed". So the report SEPARATES:
  - "DiT纯采样ms"  = component denoise time  -> the reliable speed metric
  - "端到端ms(口径)" = full latency + a boundary tag, NOT for cross-system speed claims

Tasks emit different quality columns:
  text-to-image : CLIP / 美学 / IR         + (PSNR/SSIM/LPIPS vs same-system FP16)
  image-editing : dCLIP / keepSSIM / keepLPIPS / 美学 / IR + (量化差异)
  text-to-video : 逐帧CLIP / 逐帧美学 / 时序LPIPS / 时序SSIM / 闪烁std + (量化差异)

Usage:
    python benchmark/scripts/make_matrix_tables.py \
        --results-root benchmark/results/cross-system-matrix \
        --output benchmark/results/cross-system-matrix/tables_matrix.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]


def load_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def budget_of(ro: Dict[str, Any]) -> str:
    if ro.get("max_vram_gib") is not None:
        return f"{int(ro['max_vram_gib'])}g"
    if ro.get("offload") or ro.get("offload_to_cpu"):
        return "offload"
    return "24g"


# diffusers expresses quantization via Quanto qtypes, not a `precision` field.
_QUANTO_PRECISION = {
    ("qfloat8", "qfloat8"): "fp8",
    ("qfloat8", None): "fp8",
    ("qint8", "qint8"): "w8a8",
    ("qint8", None): "w8",
    ("qint4", "qint4"): "w4a4",
    ("qint4", None): "w4",
}


def precision_of(ro: Dict[str, Any]) -> str:
    qw = ro.get("quant_weights")
    if qw:
        qa = ro.get("quant_activations")
        return _QUANTO_PRECISION.get((qw, qa)) or _QUANTO_PRECISION.get((qw, None)) or str(qw)
    return ro.get("precision", "bf16")


def boundary_tag(measurement_boundary: Optional[str]) -> str:
    if not measurement_boundary:
        return "?"
    if "no_output_encoding" in measurement_boundary or "load_once" in measurement_boundary:
        return "净推理"  # load-once, excludes model load + output encoding
    if "includes_load" in measurement_boundary:
        return "含加载+编码"
    return measurement_boundary[:16]


def fmt(v: Any, p: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{p}f}"
    return str(v)


def avg(vals: List[Any]) -> Optional[float]:
    xs = [v for v in vals if isinstance(v, (int, float))]
    return sum(xs) / len(xs) if xs else None


class Row:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        cfg = load_yaml(run_dir / "config.resolved.yaml") or {}
        result = load_json(run_dir / "result.json") or {}
        quality = load_json(run_dir / "eval" / "quality.json") or {}
        metrics_j = load_json(run_dir / "metrics.json") or {}

        wl = cfg.get("workload", {}) or {}
        ro = cfg.get("run_options", {}) or {}
        self.system = (cfg.get("system", {}) or {}).get("system_id") or result.get("system", "")
        self.task = wl.get("task") or result.get("task", "")
        self.workload = wl.get("workload_id") or result.get("workload", "")
        self.precision = precision_of(ro)
        self.budget = budget_of(ro)
        self.prompt_id = ro.get("prompt_id", "")
        self.status = result.get("status", "")

        lat = result.get("latency_ms", {}) or {}
        mem = result.get("memory", {}) or {}
        cvram = mem.get("component_vram_mib", {}) or {}
        self.e2e_ms = lat.get("steady_state_median") or lat.get("first_generation")
        self.dit_ms = lat.get("dit")  # reliable pure-sampling speed metric
        self.te_ms = lat.get("text_encoder")
        self.vae_ms = lat.get("vae")
        self.boundary = boundary_tag(metrics_j.get("measurement_boundary"))
        self.peak_vram = mem.get("peak_vram_mib")
        self.te_vram = cvram.get("text_encoder")
        self.dit_vram = cvram.get("dit")
        self.vae_vram = cvram.get("vae")

        # quality metrics (from eval/quality.json -> metrics{} + quant_vs_fp16{})
        m = quality.get("metrics", {}) or {}
        d = quality.get("quant_vs_fp16") or {}

        def mean(key: str, block: Dict[str, Any] = m) -> Optional[float]:
            s = block.get(key)
            return s.get("mean") if isinstance(s, dict) else None

        self.clip = mean("clip")
        self.aesthetic = mean("aesthetic")
        self.ir = mean("image_reward")
        self.dclip = mean("directional_clip")
        self.keep_ssim = mean("keep_ssim")
        self.keep_lpips = mean("keep_lpips")
        self.tmp_lpips = mean("temporal_lpips")
        self.tmp_ssim = mean("temporal_ssim")
        self.tmp_lpips_flk = mean("temporal_lpips_flicker")
        self.psnr = (d.get("psnr") or {}).get("mean") if d else None
        self.ssim = (d.get("ssim") or {}).get("mean") if d else None
        self.lpips = (d.get("lpips") or {}).get("mean") if d else None


# --------------------------------------------------------------------------- #
# Per-task column specs: (header, attr, precision)
# --------------------------------------------------------------------------- #
COMMON_COLS = [
    ("DiT纯采样ms", "dit_ms", 1),
    ("端到端ms", "e2e_ms", 1),
    ("口径", "boundary", None),
    ("TE_ms", "te_ms", 1),
    ("VAE_ms", "vae_ms", 1),
    ("峰值显存", "peak_vram", 0),
    ("TE显存", "te_vram", 0),
    ("DiT显存", "dit_vram", 0),
    ("VAE显存", "vae_vram", 0),
]

TASK_COLS = {
    "text-to-image": [
        ("CLIP", "clip", 3),
        ("美学", "aesthetic", 2),
        ("IR", "ir", 3),
        ("PSNRvsFP16", "psnr", 2),
        ("SSIMvsFP16", "ssim", 3),
        ("LPIPSvsFP16", "lpips", 3),
    ],
    "image-editing": [
        ("方向CLIP", "dclip", 3),
        ("保持SSIM", "keep_ssim", 3),
        ("保持LPIPS", "keep_lpips", 3),
        ("美学", "aesthetic", 2),
        ("IR", "ir", 3),
        ("PSNRvsFP16", "psnr", 2),
        ("SSIMvsFP16", "ssim", 3),
        ("LPIPSvsFP16", "lpips", 3),
    ],
    "text-to-video": [
        ("逐帧CLIP", "clip", 3),
        ("逐帧美学", "aesthetic", 2),
        ("时序LPIPS", "tmp_lpips", 3),
        ("时序SSIM", "tmp_ssim", 3),
        ("闪烁std", "tmp_lpips_flk", 3),
        ("PSNRvsFP16", "psnr", 2),
        ("SSIMvsFP16", "ssim", 3),
        ("LPIPSvsFP16", "lpips", 3),
    ],
}


def cell(row: Row, attr: str, prec: Optional[int]) -> str:
    v = getattr(row, attr, None)
    if prec is None:
        return str(v) if v else "—"
    return fmt(v, prec)


def mean_cell(rows: List[Row], attr: str, prec: Optional[int]) -> str:
    if prec is None:
        return ""  # boundary/text columns have no mean
    return fmt(avg([getattr(r, attr, None) for r in rows]), prec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = args.results_root.expanduser().resolve()
    out_path = args.output or (root / "tables_matrix.md")

    rows: List[Row] = []
    for rj in root.rglob("result.json"):
        rows.append(Row(rj.parent))

    n_succ = sum(1 for r in rows if r.status == "success")
    out: List[str] = ["# 跨系统对比矩阵结果（全指标一次性汇总）\n"]
    out.append(f"共 {len(rows)} runs | success {n_succ} | failed {len(rows) - n_succ}\n")
    out.append(
        "> **速度口径提醒**：比推理速度请用「DiT纯采样ms」（组件级denoise耗时，可靠）。"
        "「端到端ms」含一次性在线量化转换/模型加载（见「口径」列：净推理=不含加载/编码，含加载+编码=cli单次），"
        "不可用于跨系统速度结论。量化质量损失（PSNR/SSIM/LPIPS vs FP16）仅在同系统内 vs 自己的FP16基准，跨系统不可比。\n"
    )
    out.append(
        "> **sd.cpp 特别注意**：stable-diffusion.cpp 分层边加载边采样，在线量化转换（q4_K/q8，可达数十~上百秒）"
        "会混入 denoise 阶段计时，因此其「DiT纯采样ms」在量化档下同样偏高、不代表纯推理。sd.cpp 的速度需用预量化权重"
        "重测或仅横向参考同档趋势，不可与 edge/diffusers 的 DiT 纯采样直接比。\n"
    )
    out.append(
        "> **主打衡量档为 q8**（画质可用）；q4 仅作极限省显存参考点，画质损失明显，不宜用于速度/质量优势结论。\n"
    )

    # group: (workload, system, precision, budget) -> rows (one per prompt)
    groups: Dict[tuple, List[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.workload, r.system, r.precision, r.budget)].append(r)

    for wl in sorted(set(r.workload for r in rows if r.workload)):
        task = next((r.task for r in rows if r.workload == wl), "text-to-image")
        cols = COMMON_COLS + TASK_COLS.get(task, TASK_COLS["text-to-image"])
        out.append(f"\n## {wl}  ({task})\n")
        header = "| 系统 | 精度 | 预算 | prompt | 状态 | " + " | ".join(h for h, _, _ in cols) + " |"
        sep = "|" + "---|" * (5 + len(cols))
        out.append(header)
        out.append(sep)

        keys = sorted([k for k in groups if k[0] == wl], key=lambda k: (k[1], k[2], k[3]))
        for k in keys:
            grp = sorted(groups[k], key=lambda r: r.prompt_id)
            for r in grp:
                cells = " | ".join(cell(r, a, p) for _, a, p in cols)
                out.append(
                    f"| {r.system} | {r.precision} | {r.budget} | {r.prompt_id} | {r.status} | {cells} |"
                )
            succ = [r for r in grp if r.status == "success"]
            if succ:
                mcells = " | ".join(mean_cell(succ, a, p) for _, a, p in cols)
                out.append(
                    f"| **{k[1]}** | **{k[2]}** | **{k[3]}** | **均值** | **({len(succ)})** | {mcells} |"
                )

    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"WROTE {out_path} ({len(rows)} runs, {len(groups)} config groups)")


if __name__ == "__main__":
    main()
