# Test manifest (job manifest): full field docs + examples

`jobs/*.yaml` is the **single entry point** for writing a benchmark: declare "which models × which quantization/acceleration methods × what task", run one command to get tables. This file is the **authoritative reference + ready-made examples** for the manifest format — reading this one file is enough to write a job.

- To copy a ready-made manifest directly: see [**Examples**](#examples-ready-made-manifests-copy-and-edit-directly) at the end.
- To look up "which systems support a method": see [`../CAPABILITIES.md`](../CAPABILITIES.md).

---

## Structure: per-system sections

The manifest has a **per-system sectioned** structure:

- **Top level** holds fields shared across the three systems (`name` / `task` / `prompts` / `steps` / `metrics` / `device`);
- **Each system to test** opens its own section (section name = system alias), carrying **that system's own model list + quantization/acceleration tiers**;
- **A system without a section = not tested**.

The key idea: `model` and `steps` are **per-section dimensions**, not shared top-level fields. Each system takes what it needs — edge uses q8_0, diffusers uses w8, sdcpp uses q4_k, and each can even test a different set of models — running the entire cross-system matrix in one command.

A minimal manifest = 2 required top-level fields + one system section (which must carry `model` + `quant`):

```yaml
name: my-first-job          # results land in results/my-first-job/
task: text-to-image         # task type, must match the model

edge-dit:                   # one system section = test that system
  model: [sd3-medium]       # model id (see ../models/); required inside the section
  quant: [fp16, q8, q4_k]   # quant is required inside the section
# everything else in the section uses defaults: offload=none, vae_tiling=auto, cache=none
# everything else at top level uses defaults: prompts=3, steps=default, all three metrics on
```

---

## Top-level shared fields

| field | required | default | notes |
|---|---|---|---|
| `name` | ✓ | — | results directory name `results/<name>/` (can be overridden by command-line `--output-root`) |
| `task` | ✓ | — | `text-to-image` / `image-editing` / `text-to-video`, must match each model yaml's `task` |
| `prompts` | | `3` | take the first N prompts from that model's prompt set per config, generate each and average |
| `steps` | | `default` | global default step count; can be overridden per section or per quant tier (see "Advanced 2") |
| `metrics` | | all three on | `quality` / `speed` / `vram` three toggles (see "Advanced 3") |
| `device` | | not locked | lock this job to a specific physical GPU (e.g. `device: 5`); if unset, use the default card. Command-line `--device N` overrides it. When running multiple jobs in parallel on multiple GPUs, assign each job a different card to avoid VRAM contention OOM |

> Note: there is **no** top-level `models` field anymore — `model` lives inside each system section (see below), so different systems can test different models.

## System section fields

The section name can be `edge-dit` (= `edge-dit.cpp`), `diffusers`, or `stable-diffusion.cpp` (= `sdcpp`).

| section field | required | default | notes |
|---|---|---|---|
| `model` | ✓ | — | that system's model id list, taken from `../models/` (14). A **list sweeps this dimension** (each model × each quant); a single id is one model. An `quant` object can also pin `model` to just one tier (see "Advanced 1") |
| `quant` | ✓ | — | that system's quantization tier list; an item can be an **id string** or an **object** (see "Advanced 1"). edge/sdcpp accept `fp16`/`q8`/`q4_k`, diffusers accepts `bf16`/`w8` (see "Quantization tiers per-system" at the end) |
| `offload` | | `none` | which components to offload (**weights on CPU, staged to GPU for compute** — no "compute on CPU" mode). Values: `none` / `full` (whole model, all three systems) · `text-encoder-offload` / `vae-offload` / `dit-offload` (per-component, **edge-only**) · `auto-allocate` / `auto-fit` (engine-driven, **edge-only**) · `sequential` (**diffusers-only**, accelerate per-submodule). A tier not supported by a section's system is skipped during expansion. **scalar single tier, a list sweeps this dimension**. ⚠ `text-encoder-offload` now stages the TE in ~1G segments (TE no longer OOMs), but it leaves the **DiT resident** — on a large model (FLUX ~22.7G, Qwen ~38G, Wan-14B) the run still OOMs on the resident DiT, so use `full` or combine with `dit-offload` for those |
| `max_vram` | | not set | cap compute-graph VRAM in GB (`max_vram: 20` → `--max-vram`), via graph-cut segmentation; **edge and sd.cpp** both honor it. When offloading without it, the edge engine defaults to `0.85 × free VRAM`. Also settable per-quant object (see "Advanced 1") |
| `vae_tiling` | | `auto` | `auto` (engine decides by VRAM) / `yes` / `no` |
| `cache` | | `none` | cache method id; **scalar single tier, a list sweeps this dimension** |
| `steps` | | top-level `steps` | section-level step count override (see "Advanced 2") |

## How the matrix expands

For each system section, expand the Cartesian product `model × quant × offload × cache × prompts`; summing across sections gives the total run count. Adding `--dry-run` prints each expanded run with its `run_options`, and prints the total run count at the start — **after writing the manifest, dry-run to verify before running for real**.

> Example: `edge-dit` section `model(1) × quant(3) × offload(1) × cache(1) × prompts(3) = 9`.

---

## Advanced 1 · per-quant object override (configure one tier individually)

Besides an id string, a `quant` list item can be an **object** `{type: <id>, model:, steps:, offload:, max_vram:, vae_tiling:, cache:}` — the keys inside the object **override the section default**, applying only to that one tier. This is the core of the section-scoped schema: it lets one section mix "most tiers use the section defaults" with "this one tier pins a specific model / step count / switch". (For the edge-only `offload: auto-fit`, `type` is ignored — auto-fit owns the DiT quant and downgrades it under the budget.)

```yaml
edge-dit:
  model: [sd3-medium]                              # section default model
  quant:
    - fp16                                         # bare id: section default model (sd3-medium), no offload
    - q8                                           # bare id: sd3-medium q8
    - {type: q4_k, model: qwen-image, offload: full}   # only qwen-image, q4_k, whole-model offload
    - {model: flux-schnell, type: fp16, steps: 4}      # only flux-schnell, forced 4 steps (distilled)
  offload: none                                    # section default (bare tiers use it; object-overridden tiers do not)
```

Expansion: `sd3-medium` fp16 (none) · `sd3-medium` q8 (none) · `qwen-image` q4_k (offload) · `flux-schnell` fp16 4-step, each × prompts. Keys not written in an object fall back to the section default. Corresponding example `example-per-model-mix.yaml`.

> The same override rule applies to `model` / `offload` / `max_vram` / `cache` / `vae_tiling` / `steps` — all are section dimensions: write a section-level list to sweep, or pin one value inside a `quant` object to fix it for just that tier.

## Advanced 2 · steps (global / per-section / per-tier)

`steps` resolves in three levels, most-specific wins: **quant object `steps` > section `steps` > top-level `steps`**. The top-level default is `default`.

- `default` (default): each model uses `generation.steps` from its own model yaml — **distilled models are naturally few-step** (schnell / turbo / lightning / distill come with ~4–8 steps); leaving the default is recommended;
- **an integer** at top level: force that step count for every run (e.g. `steps: 20`);
- **section-level `steps: N`**: force that step count for the whole section;
- **per-tier `{type: ..., steps: N}`**: give one quant tier its own step count (usually to pin a distilled model's few-step budget, as in "Advanced 1").

```yaml
# top-level default = 20 for everything, but the distilled tier runs 4 steps
steps: 20
edge-dit:
  model: [sd3-medium]
  quant:
    - fp16                                    # 20 steps (top-level)
    - {model: flux-schnell, type: fp16, steps: 4}   # 4 steps (per-tier override)
```

(Note: `steps` lands in each run's workload `generation.steps`, not in `run_options`, so it does not appear in `--dry-run`'s `run_options=`, but the expanded run count and step count are correctly injected.)

## Advanced 3 · metrics three toggles (control evaluation and table output)

`metrics` has three boolean toggles, **all on by default**:

```yaml
metrics:
  quality: true      # quality: CLIP/aesthetic/IR + quantization loss PSNR/SSIM/LPIPS (vs same-system baseline)
  speed:   true      # speed: DiT sampling ms / end-to-end ms / TE_ms / VAE_ms
  vram:    true      # VRAM: peak + TE/DiT/VAE per stage
```

Semantic differences (important):

- `quality: false` → **actually skips quality eval** (does not run `scripts/eval_all.py`, saving a lot of time) + does not show quality columns in the tables;
- `speed: false` / `vram: false` → the data is still produced for free during generation; the toggle **only controls whether these columns are shown** in the tables (passing `--no-speed` / `--no-vram` respectively to the table script).

The specific quality metrics are routed automatically by `task`: t2i = CLIP/aesthetic/IR; image-editing = directional CLIP/preservation SSIM/preservation LPIPS/aesthetic/IR; text-to-video = per-frame CLIP/per-frame aesthetic/temporal consistency (temporal LPIPS/SSIM/flicker std). Quantization loss PSNR/SSIM/LPIPS is computed under all three tasks (vs same-system baseline).

Only care about quality, don't want to see speed/VRAM columns:

```yaml
metrics: {speed: false, vram: false}
```

Corresponding example `example-sweeps.yaml`.

---

## cross_system capability filtering (no worry about misconfiguration)

If a section configures a method that system doesn't support (e.g. `quant:[q8]` in a `diffusers:` section, `quant:[w8]` in an `edge-dit:` section, an `offload` tier the system lacks — `text-encoder-offload`/`vae-offload`/`dit-offload`/`auto-fit`/`auto-allocate` are edge-only, `sequential` is diffusers-only — or a `cache:` item the system lacks), run.py **automatically skips that combination during expansion and prints** a line:

```
[run.py] skip → diffusers: quant 'q8' not supported (cross_system), skipped
[run.py] skip → diffusers: offload 'text-encoder-offload' not supported (cross_system), skipped
```

Neither errors out nor wastes time running to failure. Which method is supported by which systems is in [`../CAPABILITIES.md`](../CAPABILITIES.md).

## Quantization tiers per-system

| system section | accepted quantization tiers | mechanism |
|---|---|---|
| `edge-dit` | `fp16` / `q8` (q8_0) / `q4_k` | weight-only, via `precision` |
| `stable-diffusion.cpp` | `fp16` / `q8` (q8_0) / `q4_k` | weight-only, via `precision` |
| `diffusers` | `bf16` (baseline) / `w8` | Optimum-Quanto, via `quant_weights`. `w8`=qint8 weight-only for all diffusers models (SD3/SD3.5-turbo/flux/qwen), mirrors edge/sd.cpp q8_0. See CAPABILITIES.md |

To compare quantization across systems: write each section's own tiers (and its own `model` list), and one job runs it all in one command (see `example-cross-system` below). Quantization loss is only meaningful within the same system vs its own baseline (fp16 for edge/sdcpp, bf16 for diffusers), not comparable across systems; CLIP/aesthetic/IR absolute quality can be compared side by side.

## cache calibration note

Calibration-free and directly sweepable are `none` / `easycache` / `ucache` / `dbcache` / `taylorseer` / `cache-dit`, plus the edge-only `dicache`.

`magcache` is also job-orchestrable — just list it in a `cache:` sweep like any other method. It consults a per-step magnitude-ratio table: FLUX and Qwen-Image ship a **built-in table** (used directly, no calibration), while SD3 and Wan need a **per-model calibrated profile**. run.py handles the calibration two-step automatically:

1. **Calibrate** — for SD3/Wan it inserts a `--cache-calibrate` run (a full-compute pass that writes the profile) *before* the accelerated runs. The profile path is deterministic (`cache/<model>-<task>-<WxH>-s<steps>-magcache-profile.json`), keyed by model × task × resolution × steps, and is reused across offload tiers and re-runs — an existing profile is not regenerated. The dry-run plan shows the `[CALIB]` line.
2. **Consume** — the accelerated runs get `--cache-profile <that path>` automatically.

So a `cache: [none, magcache]` sweep on an SD3 or Wan section expands to a calibration run plus the benchmarked runs with no extra manifest fields; on FLUX/Qwen it skips calibration and uses the built-in table. Calibration is not available on the GPU path for Flux-Kontext or Qwen-Image-Edit (device-only feature capture) — magcache there falls back to full compute.

`sencache` is not benchmarked on edge yet, so **do not put it in the manifest** for now. See [`../CAPABILITIES.md`](../CAPABILITIES.md) for the full support matrix.

---

**How to run** (from the repo root):

```bash
python benchmark/run.py \
  --job  benchmark/jobs/<manifest>.yaml \
  --site benchmark/sites/site4090.yaml
# add --dry-run to only print the expanded run plan (no generation), for checking the manifest
# add --output-root <dir> to change the results directory (default results/<name>/)
# add --device N to lock this run to physical GPU N (overrides the device field in the job)
```

**Multi-GPU parallelism**: when the machine has multiple cards, assign each job a different `device` (or use `--device`) to run them simultaneously, each on its own card, without competing for VRAM:

```bash
# three jobs each lock one card, running in the background simultaneously (can omit --device when the job top level sets device: 0/1/2)
python benchmark/run.py --job benchmark/jobs/t2i.yaml   --site .../site4090.yaml --device 0 &
python benchmark/run.py --job benchmark/jobs/edit.yaml  --site .../site4090.yaml --device 1 &
python benchmark/run.py --job benchmark/jobs/video.yaml --site .../site4090.yaml --device 2 &
```

> Without locking cards, multiple jobs default to the same card (number 0), competing for VRAM and causing OOM — for multi-GPU parallelism, always use `device`.

---

## Examples (ready-made manifests, copy and edit directly)

Each corresponds to a same-named `example-*.yaml` under `jobs/`, with detailed comments in the file; the table below shows at a glance what capability each demonstrates:

| manifest | demonstrates | run count |
|---|---|:--:|
| `example-cross-system` | true cross-system: three sections, each with its own model list + quantization tiers, one command | 42 |
| `example-per-model-mix` | flagship of the schema: section model list + per-tier `{model, steps}` overrides (different models/steps in one section) | 5 |
| `example-sweeps` | single-system sweeps: section-level lists sweep `offload` / `cache`, plus metrics toggles (quality only) | 6 |
| `example-edit` | `image-editing` task: base + distilled (kontext-lightning) across systems | 18 |
| `example-video` | `text-to-video` task: Wan 1.3b + distilled, video metrics auto-selected by task | 15 |

Three signature patterns (see the corresponding files for the rest):

```yaml
# cross-system in one command (example-cross-system): each of three sections with its
# own model list and its own accepted quantization tiers
task: text-to-image
edge-dit:             { model: [sd3-medium, flux-dev], quant: [fp16, q8, q4_k] }
diffusers:            { model: [sd3-medium, flux-dev], quant: [bf16, w8] }
stable-diffusion.cpp: { model: [sd3-medium, flux-dev], quant: [fp16, q8] }
```
```yaml
# per-model mix (example-per-model-mix): one section, different models/steps per tier
edge-dit:
  model: [sd3-medium]
  quant:
    - fp16                                            # sd3-medium (section model)
    - {type: q4_k, model: qwen-image, offload: full}  # only qwen-image, offloaded
    - {model: flux-schnell, type: fp16, steps: 4}     # only flux-schnell, 4 steps
```
```yaml
# single-system sweep (example-sweeps): section-level lists sweep a dimension
edge-dit:
  model: [sd3-medium]
  quant: [q8]
  offload: [none, full]                 # sweep: each quant runs both
  cache:   [none, easycache, taylorseer] # sweep: calibration-free caches
```

Field semantics are above; method capability attribution is in [`../CAPABILITIES.md`](../CAPABILITIES.md).
