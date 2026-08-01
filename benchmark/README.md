# edge-dit.cpp Benchmark

Benchmark framework for edge-dit.cpp (a ggml-based Diffusion Transformer inference engine).

Measures and compares DiT inference: **which models × which quantization / acceleration methods** to test, reporting speed, VRAM, and (optionally) image quality. You can test edge-dit.cpp alone, or compare it side-by-side with **Diffusers** and **stable-diffusion.cpp**.

You only write a single **test manifest** `jobs/*.yaml` (declaring `models × quantization/methods × task`), run one command, and the framework automatically chains **image generation → evaluation → tables**; report tables land in `reports/<job>/` (raw artifacts like images live in `results/<job>/`). You don't need to understand the internal mechanics.

> **Multi-GPU parallelism**: each job can set `device: N` in the manifest (or `--device N` on the command line) to lock one GPU; multiple jobs each take one card and run simultaneously without competing for VRAM.

---

## Install: what you test → what you install

Installation is **layered and on-demand**. Testing edge-dit.cpp alone does **not** require installing Diffusers or stable-diffusion.cpp.

| You want to… | Install what |
|---|---|
| Test **edge-dit.cpp** + compute image quality | `pip install -r requirements/core.txt` + `python scripts/setup_assets.py` (for edge-dit itself see repo-root `scripts/build_cuda.sh`) |
| Also compare **Diffusers** | Add `pip install -r requirements/diffusers.txt` |
| Also compare **stable-diffusion.cpp** | build `sd-cli` + e2e wrapper (repo-root `scripts/build_sd_cpp_e2e.py`); fill in `stable_diffusion_cpp_*` paths in the site |

```bash
# 1. First install the matching torch for your CUDA/driver (core.txt intentionally does not pin the torch version)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # choose per your actual CUDA
pip install -r benchmark/requirements/core.txt

# 2. Download image-quality evaluation weights (LAION aesthetic / ImageReward / CLIP)
python benchmark/scripts/setup_assets.py
```

`core.txt` = everything needed to run edge-dit + compute all image-quality metrics, **excluding** the diffusers stack. `setup_assets.py` is **idempotent** (skips what already exists, `--force` re-downloads) and supports the `HF_ENDPOINT` mirror; weights **are not committed to git** (size/license).

---

## Quick start (3 steps)

> All commands below run from the **repo root** `edge-dit.cpp/` — `run.py` needs to import the runner as the `benchmark` package.

### 1. Pick / write a manifest

The manifest has a **per-system sectioned** structure: shared fields at the top level, and each system to test opens its own section carrying **its own model list + quantization tiers** (`model` and `steps` are per-section dimensions, not shared top-level fields). A minimal manifest = 2 required top-level fields + one system section (which must carry `model` + `quant`):

```yaml
name: my-first-job          # report tables land in reports/my-first-job/, raw artifacts in results/my-first-job/
task: text-to-image         # task type, must match the model

edge-dit:                   # one system section = test that system
  model: [sd3-medium]       # model id, see models/; required inside the section
  quant: [fp16, q8, q4_k]   # quantization tier id, see methods/quant/
# everything else uses defaults: offload=none, vae_tiling=auto, cache=none, prompts=3, all three metrics on
```

**How to write each manifest field, advanced usage (per-quant object override for model/steps/offload/cache, section-level dimension sweeps, metrics three toggles), cross_system filtering, and a batch of ready-to-copy manifests (starting with `example-`) are all explained in [`jobs/README.md`](jobs/README.md).**

### 2. Run

**First time, configure the local site**: `sites/*.yaml` is the only place for machine-specific paths (edge-dit binary, each model directory, python, etc.). Following [`sites/README.md`](sites/README.md), copy `sites/site-example.yaml` into your own `sites/<machine>.yaml` and fill in the actual paths.

```bash
python benchmark/run.py \
  --job  benchmark/jobs/my-first-job.yaml \
  --site benchmark/sites/my-site.yaml
```

Add `--dry-run` to only print the expanded run plan (no image generation), for checking the manifest; add `--output-root <dir>` to change the **raw artifacts** directory (default `results/<name>/`; report tables always go to `reports/<name>/`).

### 3. View results

After running, the **report tables** are in `reports/<name>/` (committed to git), and the **raw artifacts** (images, result.json) are in `results/<name>/` (git-ignored). To view results, look at reports:

```
benchmark/reports/my-first-job/
├── summary-all.md       one at-a-glance overview table (core columns: speed/VRAM/quality, means)  ← look here first
├── summary-speed.md     narrow speed table (DiT sampling / end-to-end / TE / VAE ms)
├── summary-memory.md    narrow VRAM table (peak + per stage)
├── summary-quality.md   narrow quality table (CLIP/aesthetic/IR + quantization loss)
└── tables.md            full per-metric detail (per config, per prompt + mean, 20 columns, for reference)
```

Generated images/videos are under `results/<name>/<system>/.../samples/`.

---

## What run.py does

```
job.yaml
   └─ expand  each system section (model × quant × offload × cache) × prompts  into a set of runs
       └─ directly call engine/runners/<system>.execute() one by one   generate + collect latency/VRAM, write result.json
           └─ scripts/eval_all.py                        backfill image-quality metrics (when quality=true)
               └─ scripts/make_matrix_tables.py          aggregate into a single tables.md
```

Everything runs in-process end-to-end, bypassing the old suite / scenario_matrix mechanism (archived in `archive/`).

---

## "I want to test X, where do I configure it"

| I want to… | Where to configure |
|---|---|
| How to write manifest fields (model/quant/cache/offload/VAE tiling/per-quant object/steps/metrics) | all in [`jobs/README.md`](jobs/README.md) |
| Cross-system comparison | open several system sections, each with its own model list + quantization tiers (see "Three-system integration") |
| Add a model / method / comparison system / switch machine | see "Extending" at the end |
| Re-run evaluation only (no re-generation) | `scripts/eval_all.py` + `make_matrix_tables.py` (see end) |

Usually you only edit two places: **`jobs/`** (what to test) and **`sites/`** (machine paths). `models/` (14) and `methods/` (22) are the libraries you choose from.

---

## Acceleration methods overview

Under `methods/` each method has one yaml, registering its category, `kind` (runtime = just change params and test / build-variant = needs a separately compiled binary), whether it needs calibration, and its cross-system comparable scope. **The authoritative lookup table for "which method is supported by which systems, which can be orchestrated directly by a job, which need calibration" is in [`CAPABILITIES.md`](CAPABILITIES.md).**

In one sentence:

- **Dimensions the current job section can orchestrate directly**: quantization (`quant`), cache (`cache`), offload (`offload`), VAE tiling (`vae_tiling`) — single-card runtime.
- **Attention** (flash on by default; sage/cudnn need build variants) and **parallelism** (cfg/sequence need multi-GPU) already have their capabilities and cross-system comparability registered in `methods/`, but `run.py` currently executes single-card only and has no dedicated job field yet; testing these for now goes through engine-side switches or the corresponding binary.
- **cache baseline**: `none` already includes flash-attention + CUDA fused operators + cuDNN (performance build), on by default and not listed as a method. **Calibration-free and directly sweepable**: `easycache / ucache / dbcache / taylorseer / cache-dit` (shared by edge and sd.cpp), plus edge's own `dicache`. `magcache` is also job-orchestrable on edge: FLUX / Qwen-Image use a built-in table (no calibration), while SD3 / Wan are calibrated automatically — run.py runs a `--cache-calibrate` pass to write the profile, then benchmarks with `--cache-profile` (see [jobs/README.md](jobs/README.md#cache-calibration-note)). `sencache` is not benchmarked on edge yet — **do not put it in the manifest** for now.

---

## Three-system integration

System capabilities are defined in `systems/*.yaml`. After configuring each system's paths in the site, **open a section for that system** in the manifest to use it.

- **edge-dit.cpp** (required, the main subject): first compile `build-cuda/bin/ed-cli` and `ed-sample` (repo-root `scripts/build_cuda.sh`). Quantization tiers `fp16` / `q8` (q8_0) / `q4_k` (weight-only).
- **diffusers** (optional, Python reference): needs a Python with `torch`/`diffusers`/`transformers` installed; the site's `diffusers_python` points to it. Quantization uses **Optimum-Quanto**: `bf16` (baseline) / `fp8` (qfloat8) / `w8a8` (qint8, crashes on SD3); **no q4**; CUDA only.
- **stable-diffusion.cpp** (optional, native baseline): needs `sd-cli` and the e2e wrapper built. dtype `fp16` / `q8` (q8_0) / `q4_k`. Its on-the-fly quantization conversion mixes into the sampling timing, so under quantized tiers "DiT sampling ms" is inflated and speed must be interpreted with care.

**One job runs the entire cross-system matrix**: per-system sectioning lets each system have a section with its own model list + quantization tiers, and one command runs it all — edge/sdcpp accept `fp16/q8/q4_k`, diffusers accepts `bf16/fp8/w8a8`; the three sections expand, generate, evaluate, and aggregate into the same `tables.md`. `jobs/example-cross-system.yaml` is the flagship example. If a section writes a method that system doesn't support, run.py **automatically skips it and prints** `[run.py] skip → ...` during expansion, without running to failure (capability ownership is in `CAPABILITIES.md`).

---

## Reading the tables

`reports/<name>/tables.md` — several prompt detail rows per config + 1 mean row. Columns are system / precision / **budget** / **cache** / component-level times / VRAM / quality. Only rows that are the **same config across different prompts** collapse into one averaged mean row; configs differing in any dimension (precision / budget / cache) stay separate. **Metric conventions (important)**:

- **Compare inference speed with DiT sampling ms** (component-level denoise time), **never with end-to-end ms** — the latter includes one-time on-the-fly quantization conversion and model loading, which contaminates conclusions (especially for sd.cpp quantized tiers).
- **Quantization loss** (PSNR/SSIM/LPIPS) is only meaningful **within the same system** vs its own baseline (fp16 for edge/sdcpp, bf16 for diffusers), and is **not comparable across systems**.
- Cross-system **absolute image quality** (CLIP/aesthetic/ImageReward) can be compared side by side; but when a gap is suspiciously large, first check convention alignment (same model/prompt/seed/resolution/steps/dtype) before drawing conclusions.
- **q8 is the headline usable-quality tier**; q4 is only an extreme VRAM-saving reference point, with obvious quality loss.
- **budget column**: directly lists **which components were offloaded** plus the max-vram budget — e.g. `no-offload`, `te offload`, `full offload`, `full offload (max-vram 20g)`, `sequential (full offload)`, and for auto tiers only the components actually offloaded plus the mode, e.g. `te offload + vae offload (max-vram 20g) (auto-fit)`. (Whole-model tiers `full`/`sequential` subsume the per-component names; auto tiers annotate `(auto-fit)`/`(auto-allocate)`.)
- **cache column**: its own column (separate from budget) showing the cache method used (`none` / `easycache` / `taylorseer` / …); runs differing only in cache method stay distinct rows.
- Task-specific quality columns are routed automatically: t2i = CLIP/aesthetic/IR; editing = directional CLIP + preservation SSIM/LPIPS; video = per-frame + temporal consistency.
- The `metrics` three toggles control table output: `speed:false` hides time columns, `vram:false` hides VRAM columns, `quality:false` hides quality columns and actually skips the quality computation (see `jobs/README.md`).

---

## Directory guide

**Active (you touch these directly)**
- `jobs/` — test manifests, your entry point (`README.md` has the full field docs + example walkthroughs + `example-*` ready-made manifests)
- `models/` — model library, one file per model (14, covering 6 families): SD3 / SD3.5 (incl. distilled turbo) / FLUX.1 (dev/schnell) / FLUX.1-Kontext (incl. distilled lightning) / Qwen-Image (incl. distilled lightning + Edit) / Wan 2.1 (incl. distilled distill)
- `methods/` — method library, categorized as `quant/ cache/ attention/ memory/ parallel/` (22)
- `sites/` — machine config (`site4090.yaml` / `siteh200.yaml` + template), the **only** place allowed to hold machine-specific paths
- `run.py` — front-end entry point

**Framework internals (usually untouched)**
- `systems/` — system capability definitions (runner, tasks, backends, memory modes)
- `engine/runners/` — per-system adapters; `engine/measurement/` — latency/VRAM/environment collection
- `scripts/` — `setup_assets.py` (download quality weights), `eval_all.py` (quality backfill), `make_matrix_tables.py` (tables), `run_*_e2e.py` (per-system execution wrappers), sd.cpp build/convert scripts
- `evaluation/single_metric/` — CLIP / aesthetic / ImageReward / PSNR / SSIM / LPIPS / directional CLIP / temporal consistency metric implementations
- `prompts/` — per-task prompt sets; `requirements/` — `core.txt` / `diffusers.txt`; `results/` / `cache/` — local artifacts (git-ignored)

**Results and reports**
- `results/` — **raw artifacts** auto-produced by run.py (images/result.json/config per run), **git-ignored** (large), regenerable by re-running any time.
- `reports/<job>/` — **report tables** auto-produced by run.py (`tables.md` detail + `summary-*.md` aggregates), **committed to git**, where you look at results (`reports/v0.1.0-alpha/` is a historical release snapshot).

**Legacy system (deprecated, all archived in `archive/`)**: old orchestration / analysis / configs / shell scripts, old specs / schemas contracts; `archive/` is git-ignored and the new flow does not depend on it.

---

## Re-evaluate only (no re-generation)

With existing artifacts and want to swap metrics / regenerate tables, directly call the two standalone scripts:

```bash
python benchmark/scripts/eval_all.py \
  --results-root benchmark/results/<job> \
  --site benchmark/sites/site4090.yaml
python benchmark/scripts/make_matrix_tables.py \
  --results-root benchmark/results/<job> \
  --output benchmark/reports/<job>/tables.md
python benchmark/scripts/summarize.py \
  --results-root benchmark/results/<job> \
  --output-dir benchmark/reports/<job>
```

`eval_all.py` uses each run's `config.resolved.yaml` as the authoritative source (task/system/precision/real prompt), routes metrics by task, loads each metric model only once, and backfills `result.json.quality` and `eval/quality.json`.

---

## Extending (none require changing `engine/`)

- **Add a model**: drop a file into `models/` (copy an existing one, set `model_family` / `task` / default resolution+steps / `prompt_set` / `local_path_ref`), then configure that ref's actual path in the site.
- **Add a quantization/cache/acceleration method**: drop a file into `methods/<category>/` (named `options` + a one-line `description` + tag `kind`/`needs_calibration`/`cross_system`).
- **Add a comparison system**: add `systems/<name>.yaml` + the corresponding adapter in `engine/runners/`.
- **Switch machine**: create `sites/<host>.yaml`, filling only machine-specific paths.
