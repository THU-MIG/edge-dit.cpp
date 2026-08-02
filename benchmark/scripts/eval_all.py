#!/usr/bin/env python3
"""Unified cross-system evaluation orchestrator (one pass, all metrics, correct
per-task routing).

For every success run under RESULTS_ROOT it:
  1. Reads config.resolved.yaml as the AUTHORITATIVE source of task / system /
     precision / budget / true prompt. The true prompt is looked up via
     run_options.prompt_id -> workload.resolved_prompt_set[prompt_id].prompt,
     NOT the top-level resolved_prompt (which can be a stale default and is the
     root cause of the old backfill scripts scoring against the wrong prompt).
  2. Routes to the correct metrics by task:
       text-to-image : CLIP, aesthetic, ImageReward   (absolute quality)
       image-editing : directional-CLIP, keep-SSIM/LPIPS(vs input), aesthetic, IR
       text-to-video : per-frame CLIP, per-frame aesthetic, temporal-consistency
  3. Computes quantization-vs-FP16 paired PSNR/SSIM/LPIPS for models that fit
     FP16 at 24G (same-system baseline: bf16/f16 at 24g), same-prompt paired.
  4. Loads every model ONCE (CLIP-base, CLIP-large for aesthetic, ImageReward,
     LPIPS) and reuses across all runs -- no per-run subprocess, no stdout scraping.
  5. Writes each run's <run>/eval/quality.json AND back-fills result.json.quality
     so downstream aggregation reads a single source of truth.

Image dir / filename prefix is resolved from ONE authoritative table
(system x task), replacing the three inconsistent copies in the old scripts.

Usage:
    python benchmark/scripts/eval_all.py \
        --results-root benchmark/results/cross-system-matrix \
        --site benchmark/sites/site4090.yaml \
        --aesthetic-weights benchmark/cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]
SINGLE_METRIC = REPO / "benchmark" / "evaluation" / "single_metric"
sys.path.insert(0, str(SINGLE_METRIC))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# --------------------------------------------------------------------------- #
# Authoritative system x task -> (image subdir under samples/, filename prefix)
# --------------------------------------------------------------------------- #
def image_dir_and_prefix(system: str, task: str) -> Tuple[Optional[str], str, List[str]]:
    """Return (subdir_under_samples, filename_prefix, extensions)."""
    is_video = task == "text-to-video"
    is_t2i = task == "text-to-image"
    if system == "edge-dit.cpp":
        if is_t2i:
            return "edge/imgs", "img_", ["png"]
        # edit / video go through ed-cli single-run -> samples/edge-cli, measured_
        return "edge-cli", "measured_", (["avi"] if is_video else ["png"])
    if system == "diffusers":
        return "diffusers", "output_", (["avi"] if is_video else ["png"])
    if system == "stable-diffusion.cpp":
        return "stable-diffusion.cpp", "output_", (["avi"] if is_video else ["png", "ppm"])
    return None, "output_", ["png"]


def budget_of(run_options: Dict[str, Any]) -> str:
    """Derive the memory-budget bucket from the resolved run_options knobs.
    'full' = no VRAM-saving knobs (hardware-independent name for the baseline tier)."""
    if run_options.get("max_vram_gib") is not None:
        return f"{int(run_options['max_vram_gib'])}g"
    if run_options.get("offload") or run_options.get("offload_to_cpu"):
        return "offload"
    return "full"


# diffusers expresses quantization via Optimum-Quanto qtypes in run_options,
# NOT a `precision` field. Map those to the canonical precision label so runs
# are grouped/baselined correctly (a missing map would mislabel every quantized
# diffusers run as bf16 and wrongly treat it as the FP16 baseline).
_QUANTO_PRECISION = {
    ("qfloat8", "qfloat8"): "fp8",
    ("qfloat8", None): "fp8",
    ("qint8", "qint8"): "w8a8",
    ("qint8", None): "w8",
    ("qint4", "qint4"): "w4a4",
    ("qint4", None): "w4",
}


def precision_of(run_options: Dict[str, Any]) -> str:
    """Canonical precision label from run_options (handles edge/sdcpp `precision`
    and diffusers Quanto quant_weights/quant_activations)."""
    qw = run_options.get("quant_weights")
    if qw:
        qa = run_options.get("quant_activations")
        return _QUANTO_PRECISION.get((qw, qa)) or _QUANTO_PRECISION.get((qw, None)) or str(qw)
    return run_options.get("precision", "bf16")


def load_yaml(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def true_prompt(cfg: dict) -> Optional[str]:
    """Authoritative prompt: run_options.prompt_id -> resolved_prompt_set[id].prompt,
    falling back to top-level resolved_prompt only if the set lookup fails."""
    ro = cfg.get("run_options", {}) or {}
    wl = cfg.get("workload", {}) or {}
    pid = ro.get("prompt_id")
    pset = wl.get("resolved_prompt_set", {}) or {}
    if pid and pid in pset and isinstance(pset[pid], dict):
        p = pset[pid].get("prompt")
        if p:
            return p
    rp = wl.get("resolved_prompt", {}) or {}
    return rp.get("prompt")


def resolve_site_path(site: Optional[dict], ref: str) -> Optional[str]:
    if not site:
        return None
    return (site.get("paths", {}) or {}).get(ref)


class RunView:
    """Parsed, authoritative view of one run directory."""

    def __init__(self, run_dir: Path, cfg: dict, result: dict):
        self.dir = run_dir
        self.cfg = cfg
        self.result = result
        wl = cfg.get("workload", {}) or {}
        ro = cfg.get("run_options", {}) or {}
        self.system = cfg.get("system", {}).get("system_id") or result.get("system", "")
        self.task = wl.get("task") or result.get("task", "")
        self.workload = wl.get("workload_id") or result.get("workload", "")
        self.precision = precision_of(ro)
        self.budget = budget_of(ro)
        self.prompt_id = ro.get("prompt_id", "")
        # cache method (if any) on this run. A cached run alters the image, so it can
        # NOT serve as a quantization baseline even at fp16/bf16.
        _cache = ro.get("cache")
        self.cache = str(_cache) if _cache and _cache not in ("none", "off", False) else "none"
        self.prompt = true_prompt(cfg) or ""
        self.input_image_ref = wl.get("input_image_ref")
        subdir, prefix, exts = image_dir_and_prefix(self.system, self.task)
        self.image_subdir = subdir
        self.prefix = prefix
        self.exts = exts

    @property
    def sample_dir(self) -> Optional[Path]:
        if not self.image_subdir:
            return None
        d = self.dir / "samples" / self.image_subdir
        return d if d.is_dir() else None


def discover_runs(root: Path) -> List[RunView]:
    runs: List[RunView] = []
    for rj in root.rglob("result.json"):
        result = load_json(rj)
        if not result or result.get("status") != "success":
            continue
        cfg = load_yaml(rj.parent / "config.resolved.yaml")
        if not cfg:
            continue
        runs.append(RunView(rj.parent, cfg, result))
    return runs


# --------------------------------------------------------------------------- #
# Metric backends (loaded once, reused across all runs)
# --------------------------------------------------------------------------- #
class Metrics:
    def __init__(self, device: str, aesthetic_weights: Path, clip_model: str):
        self.device = device
        self.clip_model_name = clip_model
        self.aesthetic_weights = aesthetic_weights
        self._clip = None  # (model, processor) for CLIP-base scoring
        self._aes = None  # (clip_large, mlp, processor)
        self._ir = None
        self._lpips = None

    # -- lazy loaders ------------------------------------------------------- #
    def clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            proc = CLIPProcessor.from_pretrained(self.clip_model_name)
            model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
            self._clip = (model, proc)
        return self._clip

    def aesthetic(self):
        if self._aes is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from cal_aesthetic import AestheticMLP

            name = "openai/clip-vit-large-patch14"
            proc = CLIPProcessor.from_pretrained(name)
            clip_large = CLIPModel.from_pretrained(name).to(self.device).eval()
            mlp = AestheticMLP(int(clip_large.config.projection_dim))
            state = torch.load(str(self.aesthetic_weights), map_location="cpu")
            mlp.load_state_dict(state)
            mlp.to(self.device).eval()
            self._aes = (clip_large, mlp, proc)
        return self._aes

    def ir(self):
        if self._ir is None:
            import cal_ir  # applies transformers>=5 shim on import
            import ImageReward

            self._ir = ImageReward.load("ImageReward-v1.0", device=self.device)
        return self._ir

    def lpips(self):
        if self._lpips is None:
            import lpips as lpips_lib

            self._lpips = lpips_lib.LPIPS(net="alex").to(self.device).eval()
        return self._lpips

    # -- scorers ------------------------------------------------------------ #
    def score_clip(self, image, prompt) -> float:
        import cal_clip

        model, proc = self.clip()
        return cal_clip.score_image_clip(model, proc, self.device, image, prompt)

    def score_aesthetic(self, image) -> float:
        import cal_aesthetic

        clip_large, mlp, proc = self.aesthetic()
        return cal_aesthetic.score_image_aesthetic(
            clip_large, mlp, proc, self.device, image
        )

    def score_ir(self, prompt, image_path) -> float:
        return float(self.ir().score(prompt, str(image_path)))


def summarize(values: List[float]) -> Dict[str, Any]:
    xs = [v for v in values if isinstance(v, (int, float))]
    if not xs:
        return {"count": 0, "mean": None}
    return {
        "count": len(xs),
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


# --------------------------------------------------------------------------- #
# Per-task scoring
# --------------------------------------------------------------------------- #
def _iter_indexed_images(sample_dir: Path, prefix: str, exts: List[str]):
    """Yield (index, Path) for image files, sorted by index."""
    import cal_clip

    m = cal_clip.collect_images_by_index(sample_dir, prefix, exts)
    for idx in sorted(m):
        yield idx, m[idx]


def _open_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def _frames_rgb(path: Path):
    import cal_clip

    return list(cal_clip.iter_video_frames_rgb(path))


def score_t2i(run: RunView, metrics: Metrics) -> Dict[str, Any]:
    """CLIP + aesthetic + ImageReward on each image, all vs the true prompt."""
    sample_dir = run.sample_dir
    out: Dict[str, Any] = {}
    clip_vals, aes_vals, ir_vals = [], [], []
    for _, path in _iter_indexed_images(sample_dir, run.prefix, run.exts):
        img = _open_rgb(path)
        clip_vals.append(metrics.score_clip(img, run.prompt))
        aes_vals.append(metrics.score_aesthetic(img))
        ir_vals.append(metrics.score_ir(run.prompt, path))
    out["clip"] = summarize(clip_vals)
    out["aesthetic"] = summarize(aes_vals)
    out["image_reward"] = summarize(ir_vals)
    return out


def score_edit(run: RunView, metrics: Metrics, input_image: Optional[Path]) -> Dict[str, Any]:
    """Directional-CLIP + keep-SSIM/LPIPS(vs input) + aesthetic + IR."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    import cal_clip
    from cal_clip_directional import image_embed, text_embed

    sample_dir = run.sample_dir
    out: Dict[str, Any] = {}
    aes_vals, ir_vals, dclip_vals, keep_ssim_vals, keep_lpips_vals = [], [], [], [], []

    model, proc = metrics.clip()
    lpips_fn = metrics.lpips()

    in_img = _open_rgb(input_image) if input_image and input_image.is_file() else None
    in_feat = image_embed(model, proc, metrics.device, in_img) if in_img is not None else None
    src_feat = text_embed(model, proc, metrics.device, "a photo") if in_feat is not None else None
    in_arr = np.asarray(in_img, dtype=np.uint8) if in_img is not None else None

    for idx, path in _iter_indexed_images(sample_dir, run.prefix, run.exts):
        img = _open_rgb(path)
        aes_vals.append(metrics.score_aesthetic(img))
        ir_vals.append(metrics.score_ir(run.prompt, path))
        # directional CLIP: instruction == run.prompt (the edit instruction)
        if in_feat is not None:
            out_feat = image_embed(model, proc, metrics.device, img)
            tgt_feat = text_embed(model, proc, metrics.device, run.prompt)
            score = float(F.cosine_similarity(out_feat - in_feat, tgt_feat - src_feat, dim=-1).item())
            dclip_vals.append(score)
            # keep-* : output vs input, resize output to input size
            out_arr = np.asarray(img.resize((in_arr.shape[1], in_arr.shape[0])), dtype=np.uint8)
            s = ssim(in_arr, out_arr, data_range=255, channel_axis=2)
            keep_ssim_vals.append(float(s[0] if isinstance(s, tuple) else s))
            with torch.inference_mode():
                a = torch.from_numpy(in_arr).permute(2, 0, 1).unsqueeze(0).float().to(metrics.device) / 127.5 - 1
                b = torch.from_numpy(out_arr).permute(2, 0, 1).unsqueeze(0).float().to(metrics.device) / 127.5 - 1
                keep_lpips_vals.append(float(lpips_fn(a, b).item()))

    out["directional_clip"] = summarize(dclip_vals)
    out["keep_ssim"] = summarize(keep_ssim_vals)
    out["keep_lpips"] = summarize(keep_lpips_vals)
    out["aesthetic"] = summarize(aes_vals)
    out["image_reward"] = summarize(ir_vals)
    return out


def score_video(run: RunView, metrics: Metrics) -> Dict[str, Any]:
    """Per-frame CLIP + per-frame aesthetic (mean per video) + temporal consistency."""
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim

    sample_dir = run.sample_dir
    out: Dict[str, Any] = {}
    clip_means, aes_means = [], []
    adj_lpips_means, adj_ssim_means, adj_lpips_stds, adj_ssim_stds = [], [], [], []
    lpips_fn = metrics.lpips()

    for _, path in _iter_indexed_images(sample_dir, run.prefix, run.exts):
        frames = _frames_rgb(path)  # list of PIL RGB
        if not frames:
            continue
        cvals = [metrics.score_clip(f, run.prompt) for f in frames]
        avals = [metrics.score_aesthetic(f) for f in frames]
        clip_means.append(float(statistics.mean(cvals)))
        aes_means.append(float(statistics.mean(avals)))
        # temporal: adjacent-frame LPIPS/SSIM
        arrs = [np.asarray(f, dtype=np.uint8) for f in frames]
        if len(arrs) >= 2:
            lp, ss = [], []
            with torch.inference_mode():
                for t in range(len(arrs) - 1):
                    a = torch.from_numpy(arrs[t]).permute(2, 0, 1).unsqueeze(0).float().to(metrics.device) / 127.5 - 1
                    b = torch.from_numpy(arrs[t + 1]).permute(2, 0, 1).unsqueeze(0).float().to(metrics.device) / 127.5 - 1
                    lp.append(float(lpips_fn(a, b).item()))
                    s = ssim(arrs[t], arrs[t + 1], data_range=255, channel_axis=2)
                    ss.append(float(s[0] if isinstance(s, tuple) else s))
            adj_lpips_means.append(float(statistics.mean(lp)))
            adj_ssim_means.append(float(statistics.mean(ss)))
            adj_lpips_stds.append(float(statistics.pstdev(lp)) if len(lp) > 1 else 0.0)
            adj_ssim_stds.append(float(statistics.pstdev(ss)) if len(ss) > 1 else 0.0)

    out["clip"] = summarize(clip_means)
    out["aesthetic"] = summarize(aes_means)
    out["temporal_lpips"] = summarize(adj_lpips_means)
    out["temporal_ssim"] = summarize(adj_ssim_means)
    out["temporal_lpips_flicker"] = summarize(adj_lpips_stds)
    out["temporal_ssim_flicker"] = summarize(adj_ssim_stds)
    return out


# --------------------------------------------------------------------------- #
# Quantization vs same-system FP16 baseline (paired PSNR/SSIM/LPIPS)
# --------------------------------------------------------------------------- #
def paired_diff(base: RunView, quant: RunView, metrics: Metrics) -> Optional[Dict[str, Any]]:
    """PSNR/SSIM/LPIPS between quant outputs and same-system FP16 outputs,
    paired by image index. Images only (video handled via frame decode)."""
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim

    bdir, qdir = base.sample_dir, quant.sample_dir
    if not bdir or not qdir:
        return None
    is_video = quant.task == "text-to-video"

    def load_units(run: RunView, d: Path):
        import cal_clip

        m = cal_clip.collect_images_by_index(d, run.prefix, run.exts)
        return m

    bmap, qmap = load_units(base, bdir), load_units(quant, qdir)
    common = sorted(set(bmap) & set(qmap))
    if not common:
        return None
    lpips_fn = metrics.lpips()

    def frames_of(path: Path):
        if is_video:
            return [np.asarray(f, dtype=np.uint8) for f in _frames_rgb(path)]
        return [np.asarray(_open_rgb(path), dtype=np.uint8)]

    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for idx in common:
        bf, qf = frames_of(bmap[idx]), frames_of(qmap[idx])
        n = min(len(bf), len(qf))
        for k in range(n):
            b, q = bf[k], qf[k]
            if b.shape != q.shape:
                from PIL import Image

                q = np.asarray(
                    Image.fromarray(q).resize((b.shape[1], b.shape[0])), dtype=np.uint8
                )
            mse = float(np.mean((b.astype(np.float64) - q.astype(np.float64)) ** 2))
            psnr_vals.append(100.0 if mse == 0 else 10.0 * np.log10((255.0**2) / mse))
            s = ssim(b, q, data_range=255, channel_axis=2)
            ssim_vals.append(float(s[0] if isinstance(s, tuple) else s))
            with torch.inference_mode():
                ta = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float().to(metrics.device) / 127.5 - 1
                tb = torch.from_numpy(q).permute(2, 0, 1).unsqueeze(0).float().to(metrics.device) / 127.5 - 1
                lpips_vals.append(float(lpips_fn(ta, tb).item()))

    return {
        "psnr": summarize(psnr_vals),
        "ssim": summarize(ssim_vals),
        "lpips": summarize(lpips_vals),
        "baseline_run": base.dir.name,
    }


# --------------------------------------------------------------------------- #
# Write-back
# --------------------------------------------------------------------------- #
def flat_quality(task: str, scores: Dict[str, Any], diff: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten metric summaries into result.json.quality scalar means."""
    def mean_of(key: str) -> Optional[float]:
        s = scores.get(key)
        return s.get("mean") if isinstance(s, dict) else None

    q: Dict[str, Any] = {
        "clip": mean_of("clip"),
        "aesthetic": mean_of("aesthetic"),
        "image_reward": mean_of("image_reward"),
        "psnr": None,
        "ssim": None,
        "lpips": None,
    }
    if task == "image-editing":
        q["directional_clip"] = mean_of("directional_clip")
        q["keep_ssim"] = mean_of("keep_ssim")
        q["keep_lpips"] = mean_of("keep_lpips")
    if task == "text-to-video":
        q["temporal_lpips"] = mean_of("temporal_lpips")
        q["temporal_ssim"] = mean_of("temporal_ssim")
        q["temporal_lpips_flicker"] = mean_of("temporal_lpips_flicker")
        q["temporal_ssim_flicker"] = mean_of("temporal_ssim_flicker")
    if diff:
        q["psnr"] = diff["psnr"].get("mean")
        q["ssim"] = diff["ssim"].get("mean")
        q["lpips"] = diff["lpips"].get("mean")
    return q


def write_back(run: RunView, scores: Dict[str, Any], diff: Optional[Dict[str, Any]]) -> None:
    eval_dir = run.dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": run.system,
        "task": run.task,
        "workload": run.workload,
        "precision": run.precision,
        "budget": run.budget,
        "prompt_id": run.prompt_id,
        "prompt": run.prompt,
        "metrics": scores,
        "quant_vs_fp16": diff,
    }
    with (eval_dir / "quality.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Back-fill result.json.quality (single source of truth for aggregation).
    rj = run.dir / "result.json"
    result = load_json(rj)
    if result is not None:
        result.setdefault("quality", {})
        result["quality"].update(flat_quality(run.task, scores, diff))
        with rj.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)


# FP16/bf16 baseline tier preference: least VRAM-saving first. offload does not
# change the generated image (only where weights live), so an offloaded FP16 run
# is a valid quant baseline when the un-offloaded one OOMs (large models at 24G).
_BUDGET_PREFERENCE = {"full": 0, "offload": 1}


def _baseline_rank(budget: str) -> int:
    # lower = more preferred; "full" (no offload) wins, then offload, then any Ng bucket.
    return _BUDGET_PREFERENCE.get(budget, 2)


def find_baseline(quant: RunView, runs: List[RunView]) -> Optional[RunView]:
    """Same system + workload + prompt_id FP16/bf16 baseline. Any FP16/bf16 run that
    produced an image qualifies regardless of how it ran (offload/full/auto all
    only move where weights live, not the image), EXCEPT cached runs (cache alters the
    image). Prefers the least VRAM-saving one (un-offloaded first) when several exist."""
    candidates = [
        r for r in runs
        if r.system == quant.system
        and r.workload == quant.workload
        and r.prompt_id == quant.prompt_id
        and r.precision in ("bf16", "f16", "fp16")
        and r.cache == "none"          # cached fp16 is not a valid baseline (image differs)
        and r.sample_dir is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: _baseline_rank(r.budget))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified cross-system evaluation orchestrator.")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--site", type=Path, default=REPO / "benchmark/sites/site4090.yaml")
    p.add_argument(
        "--aesthetic-weights",
        type=Path,
        default=REPO / "benchmark/cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth",
    )
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--only-task", type=str, default=None, choices=["text-to-image", "image-editing", "text-to-video"])
    p.add_argument("--only-system", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only first N runs (smoke test).")
    p.add_argument("--skip-diff", action="store_true", help="Skip quant-vs-FP16 paired diff.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"results-root not found: {root}")

    site = load_yaml(args.site) if args.site and args.site.is_file() else None
    edit_input = resolve_site_path(site, "edit_input_image")
    edit_input_path = Path(edit_input) if edit_input else None

    runs = discover_runs(root)
    if args.only_task:
        runs = [r for r in runs if r.task == args.only_task]
    if args.only_system:
        runs = [r for r in runs if r.system == args.only_system]
    if args.limit:
        runs = runs[: args.limit]
    print(f"[eval_all] {len(runs)} success runs to evaluate under {root}")

    metrics = Metrics(args.device, args.aesthetic_weights, args.clip_model)

    n_ok, n_err = 0, 0
    for i, run in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}] {run.system} {run.workload} {run.precision}/{run.budget} {run.prompt_id}"
        if run.sample_dir is None:
            print(f"{tag} -> SKIP (no sample dir {run.image_subdir})")
            continue
        try:
            if run.task == "text-to-image":
                scores = score_t2i(run, metrics)
            elif run.task == "image-editing":
                scores = score_edit(run, metrics, edit_input_path)
            elif run.task == "text-to-video":
                scores = score_video(run, metrics)
            else:
                print(f"{tag} -> SKIP (unknown task {run.task})")
                continue

            diff = None
            # quant-loss diff only for genuine quantized tiers vs the same-system FP16 baseline.
            # (bf16/f16/fp16 at any budget are NOT quantized -- comparing a non-full bf16 run to the
            #  full bf16 baseline would report offload/tiling diffs as fake "quant loss".)
            if not args.skip_diff and run.precision not in ("bf16", "f16", "fp16"):
                base = find_baseline(run, runs)
                if base is not None and base.dir != run.dir:
                    try:
                        diff = paired_diff(base, run, metrics)
                    except Exception as e:
                        print(f"{tag} -> diff failed: {e}")

            write_back(run, scores, diff)
            n_ok += 1
            print(f"{tag} -> OK")
        except Exception as e:
            n_err += 1
            print(f"{tag} -> ERROR: {e}")

    print(f"[eval_all] done: {n_ok} ok, {n_err} errors, out of {len(runs)}")


if __name__ == "__main__":
    main()
