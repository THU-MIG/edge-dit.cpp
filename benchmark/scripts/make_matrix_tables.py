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
  - "DiT sampling ms"  = component denoise time  -> the reliable speed metric
  - "end-to-end ms(boundary)" = full latency + a boundary tag, NOT for cross-system speed claims

Tasks emit different quality columns:
  text-to-image : CLIP / aesthetic / IR         + (PSNR/SSIM/LPIPS vs same-system FP16)
  image-editing : dCLIP / keepSSIM / keepLPIPS / aesthetic / IR + (quant delta)
  text-to-video : frameCLIP / frameAesth / tempLPIPS / tempSSIM / flicker std + (quant delta)

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
    """Human-readable "which components got offloaded" + max-vram, e.g.
    "te offload + vae offload (max-vram 20g)".

    Names ONLY the components pushed to CPU (resident ones are omitted), so two configs that
    differ only in offload layout never collapse to the same string (the old code returned a
    bare "20g" whenever max_vram was set, silently merging distinct offload / full runs).

    Recognizes BOTH the legacy keys (keep_text_encoder_on_cpu / keep_vae_on_cpu) and the new
    per-component keys (text_encoder_offload / vae_offload / dit_offload) during the migration.
    Whole-model tiers (offload_to_cpu -> "full offload"; sequential_offload ->
    "sequential (full offload)") are exclusive and subsume the per-component names -- note
    that the offload-full method sets offload_to_cpu AND keep_text_encoder_on_cpu together, so
    the whole-model check must win to avoid "full offload + te offload".
    """
    if ro.get("offload_to_cpu"):
        parts = ["full offload"]
    elif ro.get("sequential_offload"):
        parts = ["sequential (full offload)"]
    else:
        parts = []
        if ro.get("keep_text_encoder_on_cpu") or ro.get("text_encoder_offload"):
            parts.append("te offload")
        if ro.get("keep_vae_on_cpu") or ro.get("vae_offload"):
            parts.append("vae offload")
        if ro.get("dit_offload"):
            parts.append("DiT offload")
    label = " + ".join(parts) if parts else "no-offload"
    mv = ro.get("max_vram_gib")
    if mv is not None:
        label += f" (max-vram {int(mv)}g)"
    return label


# diffusers expresses quantization via Quanto qtypes, not a `precision` field.
_QUANTO_PRECISION = {
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


def effective_precision(requested: str, actual: Optional[Dict[str, Any]]) -> str:
    """Precision column that never lies under auto-fit.

    `requested` is what the job asked for (from run_options). When the engine ran in
    --auto-fit it OWNS the DiT quantization and may downgrade it (q8_0 -> q4_k) or ignore
    the request entirely, recorded in runner_metrics.json["actual_placement"]["dit_quant"].
    - non-auto run (actual is None / no dit_quant): return the requested tier unchanged.
    - auto run, request matched:   "q4_k(auto-fit)"        (annotated so it's clear the
                                                             engine chose it, not the user)
    - auto run, request differed:  "q8_0->q4_k(auto-fit)"  (request AND actual, side by side)
    Plain --auto-allocate owns placement only (no dit_quant), so precision stays requested.
    """
    if not isinstance(actual, dict):
        return requested
    dit_q = actual.get("dit_quant")
    if not dit_q:
        return requested
    mode = actual.get("mode") or "auto"
    if str(requested).lower() == str(dit_q).lower():
        return f"{dit_q}({mode})"
    return f"{requested}->{dit_q}({mode})"


# component label used by both manual and auto budget rendering
_COMP_LABEL = {"dit": "DiT offload", "te": "te offload", "vae": "vae offload"}


def effective_budget(ro: Dict[str, Any], actual: Optional[Dict[str, Any]]) -> str:
    """Budget/placement column that reflects the engine's ACTUAL per-component decisions.

    Manual tiers: delegate to budget_of(ro) (which lists the offloaded components + max-vram).
    Auto tiers (--auto-fit / --auto-allocate): the engine placed each component independently
    and recorded it in runner_metrics.json["actual_placement"]. We show the SAME vocabulary as
    the manual path -- only the components it actually offloaded ("DiT offload", "te offload",
    "vae offload"), resident ones omitted -- then the max-vram budget and the auto mode, e.g.
        "te offload + vae offload (max-vram 20g) (auto-fit)"
        "DiT offload (max-vram 20g) (auto-allocate)"
    If an auto run has no recorded placement (missing/failed metrics) we still tag the mode +
    max-vram rather than mislabel it "no-offload".
    """
    if isinstance(actual, dict) and any(actual.get(f"{c}_placement") for c in ("dit", "te", "vae")):
        parts = [
            _COMP_LABEL[c]
            for c in ("dit", "te", "vae")
            if actual.get(f"{c}_placement") == "offload"
        ]
        label = " + ".join(parts) if parts else "no-offload"
        mv = ro.get("max_vram_gib")
        if mv is not None:
            label += f" (max-vram {int(mv)}g)"
        mode = actual.get("mode") or "auto"
        return f"{label} ({mode})"
    # auto requested but placement not recorded -> honest mode+budget, not "no-offload"
    if ro.get("auto_fit") or ro.get("auto_allocate"):
        mode = "auto-fit" if ro.get("auto_fit") else "auto-allocate"
        label = "auto"
        mv = ro.get("max_vram_gib")
        if mv is not None:
            label += f" (max-vram {int(mv)}g)"
        return f"{label} ({mode})"
    return budget_of(ro)


def boundary_tag(measurement_boundary: Optional[str]) -> str:
    if not measurement_boundary:
        return "?"
    if "no_output_encoding" in measurement_boundary or "load_once" in measurement_boundary:
        return "net-inference"  # load-once, excludes model load + output encoding
    if "includes_load" in measurement_boundary:
        return "incl-load+encode"
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
        runner_j = load_json(run_dir / "runner_metrics.json") or {}

        wl = cfg.get("workload", {}) or {}
        ro = cfg.get("run_options", {}) or {}
        self.system = (cfg.get("system", {}) or {}).get("system_id") or result.get("system", "")
        self.task = wl.get("task") or result.get("task", "")
        self.workload = wl.get("workload_id") or result.get("workload", "")
        # Requested tier (what the job asked for). Under --auto-fit / --auto-allocate the
        # engine IGNORES this and picks its own quant + placement; run_edge_e2e.py records
        # the real decisions in runner_metrics.json["actual_placement"]. When that is present
        # we display the ACTUAL tier so the report never claims a request that did not run.
        self.req_precision = precision_of(ro)
        self.req_budget = budget_of(ro)
        self.actual = runner_j.get("actual_placement") if isinstance(runner_j, dict) else None
        self.precision = effective_precision(self.req_precision, self.actual)
        self.budget = effective_budget(ro, self.actual)
        # cache method (its own column, not folded into budget). run.py writes the method name
        # into run_options["cache"] (e.g. taylorseer / dbcache); absent when no cache was used.
        self.cache = ro.get("cache") or "none"
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
SPEED_COLS = [
    ("DiT sampling ms", "dit_ms", 1),
    ("end-to-end ms", "e2e_ms", 1),
    ("boundary", "boundary", None),
    ("TE_ms", "te_ms", 1),
    ("VAE_ms", "vae_ms", 1),
]
VRAM_COLS = [
    ("peak VRAM", "peak_vram", 0),
    ("TE VRAM", "te_vram", 0),
    ("DiT VRAM", "dit_vram", 0),
    ("VAE VRAM", "vae_vram", 0),
]
COMMON_COLS = SPEED_COLS + VRAM_COLS

TASK_COLS = {
    "text-to-image": [
        ("CLIP", "clip", 3),
        ("aesthetic", "aesthetic", 2),
        ("IR", "ir", 3),
        ("PSNRvsFP16", "psnr", 2),
        ("SSIMvsFP16", "ssim", 3),
        ("LPIPSvsFP16", "lpips", 3),
    ],
    "image-editing": [
        ("dir CLIP", "dclip", 3),
        ("keep SSIM", "keep_ssim", 3),
        ("keep LPIPS", "keep_lpips", 3),
        ("aesthetic", "aesthetic", 2),
        ("IR", "ir", 3),
        ("PSNRvsFP16", "psnr", 2),
        ("SSIMvsFP16", "ssim", 3),
        ("LPIPSvsFP16", "lpips", 3),
    ],
    "text-to-video": [
        ("frame CLIP", "clip", 3),
        ("frame aesthetic", "aesthetic", 2),
        ("temporal LPIPS", "tmp_lpips", 3),
        ("temporal SSIM", "tmp_ssim", 3),
        ("flicker std", "tmp_lpips_flk", 3),
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
    ap.add_argument("--no-speed", action="store_true", help="hide speed columns (DiT/e2e/TE/VAE ms)")
    ap.add_argument("--no-vram", action="store_true", help="hide VRAM columns (peak + per-component)")
    ap.add_argument("--no-quality", action="store_true", help="hide quality columns")
    args = ap.parse_args()

    root = args.results_root.expanduser().resolve()
    out_path = args.output or (root / "tables_matrix.md")

    rows: List[Row] = []
    for rj in root.rglob("result.json"):
        rows.append(Row(rj.parent))

    n_succ = sum(1 for r in rows if r.status == "success")
    out: List[str] = ["# Cross-system comparison matrix (all metrics, one-shot aggregate)\n"]
    out.append(f"{len(rows)} runs total | success {n_succ} | failed {len(rows) - n_succ}\n")
    out.append(
        "> **Speed boundary reminder**: to compare inference speed use \"DiT sampling ms\" (component-level denoise time, reliable). "
        "\"end-to-end ms\" includes one-time on-the-fly quantization conversion / model loading (see the \"boundary\" column: "
        "net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. "
        "Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.\n"
    )
    out.append(
        "> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion "
        "(q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its \"DiT sampling ms\" is likewise inflated under "
        "quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as "
        "a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.\n"
    )
    out.append(
        "> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, "
        "and is not suitable for speed/quality advantage claims.\n"
    )

    # group: (workload, system, precision, budget, cache) -> rows (one per prompt).
    # cache is part of the key so two runs that differ only in cache method stay distinct rows.
    groups: Dict[tuple, List[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.workload, r.system, r.precision, r.budget, r.cache)].append(r)

    for wl in sorted(set(r.workload for r in rows if r.workload)):
        task = next((r.task for r in rows if r.workload == wl), "text-to-image")
        cols = []
        if not args.no_speed:
            cols += SPEED_COLS
        if not args.no_vram:
            cols += VRAM_COLS
        if not args.no_quality:
            cols += TASK_COLS.get(task, TASK_COLS["text-to-image"])
        out.append(f"\n## {wl}  ({task})\n")
        header = "| system | precision | budget | cache | prompt | status | " + " | ".join(h for h, _, _ in cols) + " |"
        sep = "|" + "---|" * (6 + len(cols))
        out.append(header)
        out.append(sep)

        keys = sorted([k for k in groups if k[0] == wl], key=lambda k: (k[1], k[2], k[3], k[4]))
        for k in keys:
            grp = sorted(groups[k], key=lambda r: r.prompt_id)
            for r in grp:
                cells = " | ".join(cell(r, a, p) for _, a, p in cols)
                out.append(
                    f"| {r.system} | {r.precision} | {r.budget} | {r.cache} | {r.prompt_id} | {r.status} | {cells} |"
                )
            succ = [r for r in grp if r.status == "success"]
            if succ:
                mcells = " | ".join(mean_cell(succ, a, p) for _, a, p in cols)
                out.append(
                    f"| **{k[1]}** | **{k[2]}** | **{k[3]}** | **{k[4]}** | **mean** | **({len(succ)})** | {mcells} |"
                )

    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"WROTE {out_path} ({len(rows)} runs, {len(groups)} config groups)")


if __name__ == "__main__":
    main()
