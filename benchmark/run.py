#!/usr/bin/env python3
"""Benchmark front-end: drive a user job manifest end-to-end (self-orchestrating).

A job (jobs/*.yaml) declares WHAT to test — models x quant (x cache/offload/...) —
in a small, self-explaining manifest. run.py expands it into concrete runs and
drives each one DIRECTLY through the system runners, then evaluates and tabulates:

    job.yaml  ->  expand models x quant x cache x prompt into runs
              ->  runners/<system>.execute(...)     (generate images, metrics, VRAM, result.json, config.resolved.yaml)
              ->  scripts/eval_all.py               (quality metrics, back-filled)
              ->  scripts/make_matrix_tables.py     (wide detail table: tables.md)
              ->  scripts/summarize.py             (readable summary tables: summary-*.md)

This does NOT go through orchestration/run_suite.py or orchestration/config.py
(the suite / scenario_matrix machinery the reconstruction plan identified as the
root cause of coupling). It reuses only the clean, system-agnostic runner.execute()
contract (runners/base.py) plus the standalone eval + table scripts.

Usage:
    python benchmark/run.py --job benchmark/jobs/example-sd3-quant.yaml \
                            --site benchmark/sites/site4090.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parent          # benchmark/
REPO = BENCH.parent

# runners/base.py imports `from benchmark.measurement...`, so REPO must be on
# sys.path and the package imported as `benchmark.runners`.
sys.path.insert(0, str(REPO))

# offload semantic tier -> run_options knobs. edge/sdcpp honor these directly.
OFFLOAD_KNOBS = {
    "none": {},
    # component offload = weights on CPU, staged to GPU per compute (engine renamed the
    # old keep-*-on-cpu flags; these now map to the stage-based --*-offload flags).
    "text-encoder-offload": {"text_encoder_offload": True},     # text encoder weights offloaded (edge-only)
    "vae-offload": {"vae_offload": True},          # VAE weights offloaded (edge-only)
    "dit-offload": {"dit_offload": True},          # DiT weights offloaded (edge-only)
    "full": {"offload_to_cpu": True},              # whole model offloaded
    # diffusers-only: sequential per-submodule offload (more aggressive than full
    # model_cpu_offload, slower). edge/sdcpp runners don't read this key.
    "sequential": {"sequential_offload": True},
    # edge-only: engine-driven automatic placement. auto-allocate decides
    # resident/offload per component under --max-vram; auto-fit is a superset that
    # also picks the DiT/TE quantization (ignoring --type). Pair with max_vram.
    "auto-allocate": {"auto_allocate": True},
    "auto-fit": {"auto_fit": True},
}

# Which systems support each offload tier (normalized system ids). A tier applied
# to a system not listed here is skipped during expansion (cross_system), the same
# way an unsupported quant/cache is — otherwise the knob's key is silently ignored
# by that system's runner and the run misleadingly reports no offload.
OFFLOAD_SYSTEMS = {
    "none": {"edge-dit", "diffusers", "stable-diffusion-cpp"},
    "full": {"edge-dit", "diffusers", "stable-diffusion-cpp"},
    "text-encoder-offload": {"edge-dit"},        # per-component offload, edge-only
    "vae-offload": {"edge-dit"},   # per-component offload, edge-only
    "dit-offload": {"edge-dit"},   # per-component offload, edge-only
    "sequential": {"diffusers"},   # accelerate sequential CPU offload, diffusers-only
    "auto-allocate": {"edge-dit"}, # engine-driven placement, edge-only
    "auto-fit": {"edge-dit"},      # engine-driven placement + quant, edge-only
}

# vae_tiling manifest value -> run_options.vae_tiling. auto = omit (engine decides by VRAM).
# YAML coerces bare yes/no to bool True/False, so accept both those and the string forms.
VAE_TILING = {"auto": None, "yes": True, "no": False, True: True, False: False}

# job `systems` id -> systems/<file>.yaml stem (edge-dit is the friendly alias).
SYSTEM_YAML = {
    "edge-dit": "edge-dit",
    "edge-dit.cpp": "edge-dit",
    "diffusers": "diffusers",
    "sdcpp": "stable-diffusion-cpp",
    "stable-diffusion.cpp": "stable-diffusion-cpp",
}

# job section key -> canonical name used in method files' `cross_system` field.
CROSS_SYSTEM_NAME = {
    "edge-dit": "edge-dit",
    "edge-dit.cpp": "edge-dit",
    "diffusers": "diffusers",
    "sdcpp": "stable-diffusion.cpp",
    "stable-diffusion.cpp": "stable-diffusion.cpp",
}


def method_supports(method: dict, system_key: str) -> bool:
    """Does this method's cross_system list include the given system?
    Missing cross_system = assume supported (don't over-filter)."""
    xs = method.get("cross_system")
    if not xs:
        return True
    return CROSS_SYSTEM_NAME.get(system_key, system_key) in xs

# Framework-generated artifacts under benchmark/cache/ (gitignored). Their paths are
# derivable and should NOT live in the user's site config. run.py fills these into
# site.paths when absent, so the user only declares machine-specific external paths
# (models, binaries, python). A site that DOES set one still wins (explicit override).
CACHE_ARTIFACT_DEFAULTS = {
    "sd3_medium_sdcpp_transformer": "cache/sd_cpp_models/sd3-medium-transformer-sdcpp.fp16.safetensors",
    "stable_diffusion_cpp_e2e": "cache/sd-cpp-e2e/bin/sd_cpp_e2e",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fill_cache_artifact_defaults(site: dict) -> None:
    """Fill framework-generated cache-artifact paths into site.paths when the user
    didn't set them. These live under benchmark/cache/ (gitignored) so their location
    is known to the framework; the user shouldn't have to declare them in the site."""
    paths = site.setdefault("paths", {})
    for ref, rel in CACHE_ARTIFACT_DEFAULTS.items():
        if not paths.get(ref):
            paths[ref] = str((BENCH / rel).resolve())


def die(msg: str) -> None:
    print(f"[run.py] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_job(path: Path) -> dict:
    """Parse a per-system job manifest.

    Top-level shared fields: name, task, prompts, steps, metrics, device.
    Each benchmarked system is its own section keyed by a system alias
    (edge-dit / diffusers / stable-diffusion.cpp / sdcpp); a section carries that
    system's own quant (required) + offload / vae_tiling / cache. A system with no
    section is not tested. This is what makes one job express a full cross-system
    matrix with per-system quant tiers (edge q8_0 vs diffusers w8) in one run.
    """
    job = load_yaml(path)
    for req in ("name", "task"):
        if req not in job:
            die(f"job missing required field: {req}")
    job.setdefault("prompts", 3)
    job.setdefault("steps", "default")
    job.setdefault("metrics", {})
    job.setdefault("device", None)   # top-level optional: which GPU this job runs on (physical GPU index); --device overrides it

    # Collect system sections (keys that are known system aliases).
    segments = {}
    for key, val in job.items():
        if key in SYSTEM_YAML:
            seg = dict(val or {})
            if "quant" not in seg:
                die(f"system section '{key}' missing required field: quant")
            if "model" not in seg:
                die(f"system section '{key}' missing required field: model")
            seg.setdefault("offload", "none")
            seg.setdefault("vae_tiling", "auto")
            seg.setdefault("cache", "none")
            seg.setdefault("max_vram", None)   # GB budget for --max-vram; None = don't pass it
            segments[key] = seg
    if not segments:
        die("job has no system section (add e.g. 'edge-dit:' with a quant list)")
    job["_segments"] = segments
    return job


def as_list(v):
    """Scalar or list -> list (for offload/cache axes that may sweep)."""
    return v if isinstance(v, list) else [v]


def normalize_quant(item) -> dict:
    """A quant list item is either an id string or {type, offload, vae_tiling, convert, cache}.
    `type` is required EXCEPT when offload is auto-fit: auto-fit ignores --type and drives the
    DiT quant itself (q8_0->q4_k ladder), so a type there would be misleading. We fill fp16 as a
    neutral placeholder (the engine discards it); the table shows the engine's actual chosen tier."""
    if isinstance(item, str):
        return {"type": item}
    if isinstance(item, dict):
        if "type" in item:
            return dict(item)
        if item.get("offload") == "auto-fit":
            return {**item, "type": "fp16"}   # auto-fit owns quant; type is a placeholder
    die(f"bad quant item (need id or object with 'type'): {item!r}")


def resolve_method(kind: str, mid: str) -> dict:
    p = BENCH / "methods" / kind / f"{mid}.yaml"
    if not p.is_file():
        die(f"unknown {kind} method '{mid}' (expected {p})")
    return load_yaml(p)


def load_model(mid: str) -> dict:
    p = BENCH / "models" / f"{mid}.yaml"
    if not p.is_file():
        die(f"unknown model '{mid}' (expected {p})")
    return load_yaml(p)


def load_system(system_id: str) -> dict:
    stem = SYSTEM_YAML.get(system_id, system_id)
    p = BENCH / "systems" / f"{stem}.yaml"
    if not p.is_file():
        die(f"unknown system '{system_id}' (expected {p})")
    return load_yaml(p)


def prompt_set_path(model: dict) -> Path:
    """Absolute path to the model's prompt jsonl (model.prompt_set is relative to models/)."""
    return (BENCH / "models" / model.get("prompt_set", "../prompts/text-to-image-v1.jsonl")).resolve()


def load_prompt_set(pset: Path) -> dict:
    """Full {prompt_id: {prompt, ...}} map from a prompt jsonl (== config.py resolved_prompt_set)."""
    out: dict[str, dict] = {}
    with pset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            if pid:
                out[pid] = rec
    if not out:
        die(f"no prompt records in {pset}")
    return out


def first_prompt_ids(pset_map: dict, n: int) -> list[str]:
    return list(pset_map.keys())[:n]


def build_workload(model: dict, task: str, steps, pset_map: dict, pset_file: Path) -> dict:
    """In-memory workload dict for runner.execute(). Mirrors what config.py used to
    load + inject, but built directly (no suite, no yaml round-trip)."""
    gen = dict(model.get("generation", {}))
    # steps: "default" = the model's own generation.steps; a number = explicit override.
    # (per-model / per-run selection happens at expansion time; here steps is a single value.)
    if steps is not None and steps != "default":
        gen["steps"] = int(steps)
    gen.setdefault("frames", 1)
    gen.setdefault("seed", 0)
    gen.setdefault("batch_size", 1)
    gen.setdefault("precision", "bf16")  # per-run quant overrides via run_options
    mid = model["model_id"]
    return {
        "schema_version": 1,
        "workload_id": f"{mid}-{task}",
        "model_family": model.get("model_family", mid),
        "model_name": model.get("model_name", mid),
        "task": task,
        "model": model["model"],
        "input_image_ref": model.get("input_image_ref"),  # edit tasks; None for T2I/T2V
        # model_options carries per-model runner hints (e.g. sd.cpp's converted-transformer ref
        # or official single-file ref), which live under model: in the yaml but runners read
        # from workload["model_options"]. Forward whichever sd.cpp hints the model declares.
        "model_options": {k: model["model"][k] for k in (
            "stable_diffusion_cpp_single_file",
            "stable_diffusion_cpp_transformer_ref",
            "stable_diffusion_cpp_wan_dit",
            "stable_diffusion_cpp_wan_vae",
            "stable_diffusion_cpp_wan_t5",
        ) if model["model"].get(k)},
        "prompt_set": str(pset_file),
        "prompt_id": None,  # set per-run below
        "generation": gen,
        # resolved_prompt_set is what base.prompt_text() + eval_all read as authoritative.
        "resolved_prompt_set": pset_map,
        "resolved_prompt": {},  # set per-run
        "measurement": {
            "output_encoding_in_core_latency": False,
            "require_peak_vram": True,
            "require_peak_host_rss": True,
        },
        "quality": {
            "enabled": True,
            "reference_policy": "same-system-fp16",
            "metrics": ["psnr", "ssim", "lpips", "clip", "image_reward"],
        },
    }


def build_run_options(quant_method: dict, offload: str, vae_tiling, cache_id: str | None, max_vram=None) -> dict:
    """Assemble run_options from a quant method + one offload/vae_tiling/cache combo.
    quant_method options are per-system (edge: {precision: q8_0}; diffusers: {quant_weights: qint8})."""
    opts = dict(quant_method.get("options", {}))
    opts.update(OFFLOAD_KNOBS.get(offload, {}))
    if VAE_TILING.get(vae_tiling) is not None:
        opts["vae_tiling"] = VAE_TILING[vae_tiling]
    if max_vram is not None:
        opts["max_vram_gib"] = float(max_vram)
    if cache_id and cache_id != "none":
        cache_method = resolve_method("cache", cache_id)
        opts.update(cache_method.get("options", {}))
    return opts


def run_id_for(quant_id: str, offload: str, cache_id: str | None, prompt_id: str) -> str:
    parts = [quant_id]
    if offload and offload != "none":
        parts.append(offload)
    if cache_id and cache_id != "none":
        parts.append(cache_id)
    parts.append(prompt_id)
    return "-".join(parts)


def cache_needs_calibration(cache_method: dict, model_family: str) -> bool:
    """Whether this cache method must be calibrated before benchmarking on this model.
    sencache: always. magcache: only SD3 / Wan families. Others: never.
    Driven by the method yaml's `needs_calibration` field, refined by family."""
    if not cache_method.get("needs_calibration"):
        return False
    mid = cache_method.get("method_id", "")
    if mid == "magcache":
        fam = (model_family or "").lower()
        return "sd3" in fam or "wan" in fam
    return True  # sencache (and any other method flagged all-models)


def profile_path_for(model_id: str, task: str, gen: dict, cache_id: str) -> Path:
    """Deterministic calibration-profile path (cache reuse key).
    Granularity: model x task x resolution x steps x cache method."""
    w = int(gen.get("width", 0))
    h = int(gen.get("height", 0))
    steps = int(gen.get("steps", 0))
    name = f"{model_id}-{task}-{w}x{h}-s{steps}-{cache_id}-profile.json"
    return (BENCH / "cache" / name).resolve()


def stage(argv: list[str], label: str, python: str | None = None) -> None:
    interpreter = python or sys.executable
    print(f"[run.py] → {label}: {interpreter} {' '.join(argv)}")
    r = subprocess.run([interpreter, *argv])
    if r.returncode != 0:
        die(f"{label} failed (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark front-end (job manifest -> runs -> tables).")
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None,
                    help="lock this job to a physical GPU (e.g. --device 5); overrides the job device field. Comma-separate for multiple, e.g. 0,1")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the expanded run plan; do not generate.")
    args = ap.parse_args()

    from benchmark.engine.runners import RUNNERS  # imported here so --help works without deps

    job = load_job(args.job)
    site = load_yaml(args.site)
    fill_cache_artifact_defaults(site)
    # device lock: --device wins, else the job top-level device field. Set the env the runner reads,
    # base.py execution_env uses it to set CUDA_VISIBLE_DEVICES so the whole job uses only this card.
    device = args.device if args.device is not None else job.get("device")
    if device is not None:
        os.environ["BENCHMARK_CUDA_VISIBLE_DEVICES"] = str(device)
        print(f"[run.py] locked to GPU(s): {device}")
    task = job["task"]
    out_root = (args.output_root or (BENCH / "results" / job["name"])).resolve()   # per-run artifacts (images/result.json), gitignored
    report_dir = (BENCH / "reports" / job["name"]).resolve()                        # summary + detail tables, tracked in git
    segments = job["_segments"]  # {system_alias: {quant, offload, vae_tiling, cache}}

    plan: list[dict] = []  # each: {system_id, runner_key, workload, run_options, run_dir, run_id}
    seen_calib: set = set()  # (model, quant, cache) -> one calibration run regardless of offload
    skips: list = []  # (system, method) combos skipped because cross_system doesn't support them
    model_cache: dict = {}
    def get_model_bundle(mid, steps):
        # (mid, steps) -> (model, workload, pset_map, pids). model/prompts are steps-independent
        # but workload bakes in steps, so key by both.
        key = (mid, steps)
        if key in model_cache:
            return model_cache[key]
        model = load_model(mid)
        if model.get("task") != task:
            die(f"model {mid} is task={model.get('task')}, job task={task}")
        pset_file = prompt_set_path(model)
        pset_map = load_prompt_set(pset_file)
        pids = first_prompt_ids(pset_map, int(job["prompts"]))
        workload = build_workload(model, task, steps, pset_map, pset_file)
        model_cache[key] = (model, workload, pset_map, pids)
        return model_cache[key]

    for system_id, seg in segments.items():
        sys_cfg = load_system(system_id)
        runner_key = sys_cfg.get("runner")
        if runner_key not in RUNNERS:
            die(f"system '{system_id}' has unknown runner '{runner_key}'")
        for raw_q in seg["quant"]:
            q = normalize_quant(raw_q)  # id string or {type, model, offload, vae_tiling, cache, steps}
            quant_id = q["type"]
            method = resolve_method("quant", quant_id)
            if not method_supports(method, system_id):
                skips.append(f"{system_id}: quant '{quant_id}' not supported (cross_system), skipped")
                continue
            # per-quant object overrides win over the segment default; else sweep the segment list.
            # model / offload / cache / steps are all section dimensions with the same rule.
            models = [q["model"]] if "model" in q else as_list(seg["model"])
            offloads = [q["offload"]] if "offload" in q else as_list(seg["offload"])
            caches = [q["cache"]] if "cache" in q else as_list(seg["cache"])
            vtile = q.get("vae_tiling", seg["vae_tiling"])
            mvram = q.get("max_vram", seg.get("max_vram"))
            steps = q.get("steps", seg.get("steps", job["steps"]))
            for mid in models:
                model, workload, pset_map, pids = get_model_bundle(mid, steps)
                for offload in offloads:
                    if SYSTEM_YAML.get(system_id, system_id) not in OFFLOAD_SYSTEMS.get(offload, set()):
                        skips.append(f"{system_id}: offload '{offload}' not supported (cross_system), skipped")
                        continue
                    for cache_id in caches:
                        if cache_id and cache_id != "none" and not method_supports(
                                resolve_method("cache", cache_id), system_id):
                            skips.append(f"{system_id}: cache '{cache_id}' not supported (cross_system), skipped")
                            continue
                        run_options = build_run_options(method, offload, vtile, cache_id, mvram)
                        # Calibration two-step: sencache/magcache(SD3,Wan) need a profile first.
                        if cache_id and cache_id != "none":
                            cm = resolve_method("cache", cache_id)
                            if cache_needs_calibration(cm, workload.get("model_family", "")):
                                calib_profile = profile_path_for(mid, task, workload["generation"], cache_id)
                                run_options["cache_profile"] = str(calib_profile)
                                ck = (mid, quant_id, cache_id)  # profile is offload-independent
                                if ck not in seen_calib:
                                    seen_calib.add(ck)
                                    calib_ro = build_run_options(method, "none", vtile, cache_id, mvram)
                                    calib_ro["cache_calibrate"] = str(calib_profile)
                                    calib_ro["prompt_id"] = pids[0]
                                    calib_wl = dict(workload)
                                    calib_wl["prompt_id"] = pids[0]
                                    calib_wl["resolved_prompt"] = pset_map[pids[0]]
                                    plan.append({
                                        "system_id": system_id, "runner_key": runner_key,
                                        "sys_cfg": sys_cfg, "workload": calib_wl, "run_options": calib_ro,
                                        "run_dir": (BENCH / "cache" / "_calib" /
                                                    f"{mid}-{quant_id}-{cache_id}").resolve(),
                                        "run_id": f"calib-{mid}-{cache_id}",
                                        "is_calib": True, "profile_path": calib_profile,
                                    })
                        for pid in pids:
                            rid = run_id_for(quant_id, offload, cache_id, pid)
                            run_dir = out_root / system_id / workload["workload_id"] / rid
                            wl = dict(workload)
                            wl["prompt_id"] = pid
                            wl["resolved_prompt"] = pset_map[pid]
                            ro = dict(run_options)
                            ro["prompt_id"] = pid
                            plan.append({
                                "system_id": system_id, "runner_key": runner_key,
                                "sys_cfg": sys_cfg, "workload": wl, "run_options": ro,
                                "run_dir": run_dir, "run_id": rid, "is_calib": False,
                            })

    n_bench = sum(1 for p in plan if not p.get("is_calib"))
    all_models = set()
    for seg in segments.values():
        all_models.update(as_list(seg["model"]))
        for raw_q in seg["quant"]:
            qq = normalize_quant(raw_q)
            if "model" in qq:
                all_models.add(qq["model"])
    print(f"[run.py] expanded {n_bench} runs "
          f"({len(all_models)} models x {len(segments)} systems x "
          f"{int(job['prompts'])} prompts; per-section model/quant/offload/cache/steps)")
    for s in dict.fromkeys(skips):  # de-dup (same skip repeats across models)
        print(f"[run.py] skip → {s}")

    if args.dry_run:
        for p in plan:
            tag = "[CALIB] " if p.get("is_calib") else ""
            print(f"  - {tag}{p['system_id']}/{p['workload']['workload_id']}/{p['run_id']}"
                  f"  run_options={p['run_options']}")
        return

    # Per-system preflight before any real run: surfaces environment problems
    # (e.g. sd.cpp checked out at a commit the benchmark wasn't validated against).
    seen_pf = set()
    for p in plan:
        key = p["runner_key"]
        if key in seen_pf:
            continue
        seen_pf.add(key)
        pf = RUNNERS[key](p["sys_cfg"], site, REPO).preflight()
        for msg in pf.messages:
            print(f"[run.py] preflight {p['system_id']}: {msg}")

    # Drive each run directly through its system runner.
    n_ok = n_skip = n_fail = n_calib = 0
    for i, p in enumerate(plan, 1):
        # Calibration run: skip if the profile already exists (cache reuse).
        if p.get("is_calib"):
            prof = p["profile_path"]
            if prof.is_file():
                print(f"[run.py] ({i}/{len(plan)}) calib {p['run_id']} → profile exists, reuse")
                continue
            print(f"[run.py] ({i}/{len(plan)}) calib {p['run_id']} → --cache-calibrate")
            runner = RUNNERS[p["runner_key"]](p["sys_cfg"], site, REPO)
            result = runner.execute(
                p["workload"], gpu_count=1, parallel_mode=None,
                output_dir=p["run_dir"], warmup_runs=0, measured_runs=1,
                run_options=p["run_options"], scenario_id="calibration",
            )
            st = (result or {}).get("status", "unknown")
            if st == "success" and prof.is_file():
                n_calib += 1
                print(f"[run.py]     calibrated → {prof.name}")
            else:
                n_fail += 1
                print(f"[run.py]     calibration status={st} (profile written: {prof.is_file()})")
            continue
        runner = RUNNERS[p["runner_key"]](p["sys_cfg"], site, REPO)
        print(f"[run.py] ({i}/{len(plan)}) {p['system_id']}/{p['run_id']} → execute")
        result = runner.execute(
            p["workload"], gpu_count=1, parallel_mode=None,
            output_dir=p["run_dir"], warmup_runs=0, measured_runs=1,
            run_options=p["run_options"], scenario_id="default",
        )
        st = (result or {}).get("status", "unknown")
        if st == "success":
            n_ok += 1
        elif st == "unsupported":
            n_skip += 1
        else:
            n_fail += 1
        print(f"[run.py]     status={st}")
    print(f"[run.py] generation done: {n_ok} ok, {n_skip} unsupported, {n_fail} failed"
          + (f", {n_calib} calibrated" if n_calib else ""))

    if n_ok == 0:
        die("no successful runs; skipping eval/tables")

    # metrics toggles (default all on): quality=false skips the whole quality eval
    # (CLIP/aesthetic/IR/quant-loss — the slow part); speed/vram=false just hide those
    # table columns (the data is produced free by execute either way).
    metrics = job["metrics"]
    want_quality = metrics.get("quality", True) is not False
    want_speed = metrics.get("speed", True) is not False
    want_vram = metrics.get("vram", True) is not False

    # Reuse the standalone eval + table scripts (non-suite, discovery by result.json).
    if want_quality:
        eval_argv = [str(BENCH / "scripts/eval_all.py"), "--results-root", str(out_root),
                     "--site", str(args.site)]
        if device is not None:
            # eval inherits CUDA_VISIBLE_DEVICES from this process, so the locked
            # card is remapped to logical cuda:0. Passing the physical index (cuda:2)
            # would be out of range; cuda:0 = the first (or only) locked card.
            eval_argv += ["--device", "cuda:0"]
        # eval_all.py needs torch/transformers (CLIP/aesthetic/LPIPS), which live in the
        # site's diffusers_python env, not necessarily the interpreter running run.py.
        eval_python = site.get("paths", {}).get("diffusers_python")
        stage(eval_argv, "evaluate (eval_all)", python=eval_python)
    else:
        print("[run.py] metrics.quality=false → skipping quality eval")

    report_dir.mkdir(parents=True, exist_ok=True)
    tables = report_dir / "tables.md"
    table_argv = [str(BENCH / "scripts/make_matrix_tables.py"), "--results-root", str(out_root),
                  "--output", str(tables)]
    if not want_speed:
        table_argv.append("--no-speed")
    if not want_vram:
        table_argv.append("--no-vram")
    if not want_quality:
        table_argv.append("--no-quality")
    stage(table_argv, "tables (make_matrix_tables)")
    # Readable summary tables (speed / memory / quality / all) alongside the wide detail table.
    stage([str(BENCH / "scripts/summarize.py"), "--results-root", str(out_root),
           "--output-dir", str(report_dir)], "summary (summarize)")
    print(f"[run.py] done -> reports {report_dir}/ (tables.md + summary-*.md); raw artifacts {out_root}/")


if __name__ == "__main__":
    main()
