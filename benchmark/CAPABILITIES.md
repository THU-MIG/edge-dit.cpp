# Capability support table: which systems support which methods

Quick lookup for "I want to use a method, which systems can test it". The data source is the `cross_system` field of each `methods/*/*.yaml` and the `capabilities` in `systems/*.yaml`, consistent with `run.py`'s capability-filtering logic.

- ✓ = supported; — = not supported. **If a section configures a method marked —, run.py automatically skips it during expansion and prints `[run.py] skip → ...`** (no error, does not run to failure).
- **System aliases**: `edge-dit` (= edge-dit.cpp), `diffusers`, `stable-diffusion.cpp` (= `sdcpp`, abbreviated sd.cpp in the tables below).
- **kind**: `runtime` = just change parameters and test, same binary; `build-variant` = needs a separately compiled dedicated binary.

---

## Quantization (quant)

Placed in a system section's `quant:` list. edge/sd.cpp are weight-only (via `precision`), diffusers uses Optimum-Quanto (`quant_weights`).

| method id | kind | edge-dit | diffusers | sd.cpp | notes |
|---|---|:--:|:--:|:--:|---|
| `fp16` | runtime | ✓ | ✓ | ✓ | f16 shared by all three systems; the same-system quality baseline for edge/sd.cpp |
| `q8` | runtime | ✓ | — | ✓ | q8_0 weight-only int8. diffusers's weight-only int8 is the separate `w8` method, so q8 is not attributed to diffusers |
| `q4_k` | runtime | ✓ | — | ✓ | 4-bit K-quant, extreme VRAM saving; diffusers has no q4. Quality impact varies by model — ⚠ **SD3 (`sd3-medium`) and SD3.5-turbo (`sd35-medium-turbo`) degrade noticeably under q4_k** (treat their q4_k rows as a VRAM-floor reference, not a usable-quality tier); flux/qwen hold up better |
| `bf16` | runtime | — | ✓ | — | diffusers's unquantized baseline (its same-system quality baseline); edge/sd.cpp use fp16 |
| `w8` | runtime | — | ✓ | — | Optimum-Quanto qint8 weight-only (int8 weights, activations fp16). The single diffusers 8-bit tier for ALL diffusers models (SD3/SD3.5-turbo/flux/qwen) — mirrors edge/sd.cpp q8_0 (also int8 weight-only), tracks the bf16 baseline. Replaces the former qfloat8 `fp8` tier, which was dropped for poor quality (e4m3 3-bit mantissa too coarse: ~5.7% rel-err vs ~0.5% for int8). ⚠ **`w8` is incompatible with `offload: sequential`** — Quanto's `WeightQBytesTensor` cannot be rebuilt during accelerate's per-submodule offload (`WeightQBytesTensor.__new__()` crash) — but **`w8` DOES work with `offload: full`** (whole-model `model_cpu_offload`, which does not rebuild the quantized tensor). So for a large diffusers model that OOMs at w8 resident (e.g. qwen-image), use `offload: full`, not `sequential`. |

**To compare quantization across systems**: write `[fp16, q8, q4_k]` in the edge/sd.cpp sections and `[bf16, w8]` in the diffusers section, and one job runs all three sections together (see `jobs/example-cross-system.yaml`). Quantization loss is only comparable within the same system vs its own baseline, not across systems; CLIP/aesthetic/IR absolute quality can be compared side by side.

---

## Cache (cache)

Placed in a system section's `cache:` list (scalar single tier / list to sweep).

| method id | kind | needs calibration | edge-dit | diffusers | sd.cpp | notes |
|---|---|---|:--:|:--:|:--:|---|
| `easycache` | runtime | no | ✓ | — | ✓ | output change-rate gating |
| `ucache` | runtime | no | ✓ | — | ✓ | EasyCache + adaptive threshold |
| `dbcache` | runtime | no | ✓ | — | ✓ | residual-difference gating (dual-block) |
| `taylorseer` | runtime | no | ✓ | — | ✓ | residual Taylor extrapolation |
| `cache-dit` | runtime | no | ✓ | — | ✓ | DBCache gating + TaylorSeer extrapolation |
| `dicache` | runtime | no | ✓ | — | — | shallow-probe trajectory alignment; **edge-only** |
| `magcache` | runtime | FLUX / Qwen-Image: no (built-in table); SD3 / Wan: yes (run.py calibrates automatically) | ✓ | — | — | magnitude-ratio table step-skipping; job-orchestrable; **edge-only** |
| `sencache` | runtime | **needs calibration, and not yet implemented on-device (disabled in the engine), do not put in the manifest** | ✓ | — | — | sensitivity Jacobian bound; **edge-only** |

**diffusers has no cache methods at all**. Calibration-free and directly sweepable: `none / easycache / ucache / dbcache / taylorseer / cache-dit` (shared by edge and sd.cpp) plus the edge-only `dicache`. `magcache` is also job-orchestrable on edge: FLUX and Qwen-Image use a built-in table, while SD3 and Wan are calibrated automatically by run.py (a `--cache-calibrate` pass writes the profile, then the accelerated run consumes it). `sencache` is not benchmarked on edge yet — leave it out of the manifest for now (see its method yaml).

---

## Attention (attention) / Memory (memory) / Parallelism (parallel)

| category | method id | kind | edge-dit | diffusers | sd.cpp | job-orchestrable | notes |
|---|---|---|:--:|:--:|:--:|:--:|---|
| attention | `flash` | runtime | ✓ | ✓ | ✓ | indirect | on by default (included in baseline); turn it off to compare |
| attention | `cudnn-sdpa` | build-variant | ✓ | ✓ | — | — | auto-triggered at L≥4096, from the performance build |
| attention | `sage` | build-variant | ✓ | — | — | — | SageAttention2; needs `-DED_ENABLE_CUDA_SAGE_ATTN=ON` build + `ED_SAGE_ATTN=1`; **edge-only** |
| memory | `offload: text-encoder-offload` | runtime | ✓ | — | — | ✓ | TE weights offloaded to CPU, **staged to GPU segment-by-segment** (~1G/segment, automatic, no `max_vram` needed), so the TE itself no longer OOMs on staging; **edge-only**. ⚠ this offloads ONLY the TE — the **DiT stays resident**, so on a large model (FLUX ~22.7G, Qwen ~38G, Wan-14B) it still OOMs on the resident DiT + its compute buffer. For large models use `full` (offloads DiT too) or add `dit-offload` |
| memory | `offload: vae-offload` | runtime | ✓ | — | — | ✓ | VAE weights offloaded to CPU, staged to GPU for compute; **edge-only**. Same large-model staging OOM caveat as `text-encoder-offload` |
| memory | `offload: dit-offload` | runtime | ✓ | — | — | ✓ | DiT weights offloaded to CPU, staged to GPU for compute; **edge-only** |
| memory | `offload: full` | runtime | ✓ | ✓ | ✓ | ✓ | whole-model CPU offload (all weights on CPU, staged to GPU per compute); the only offload tier shared by all three systems |
| memory | `offload: sequential` | runtime | — | ✓ | — | ✓ | accelerate per-submodule offload, more aggressive/slower than `full`; **diffusers-only**. ⚠ incompatible with the `w8` quant tier (Quanto quantized tensors crash on per-submodule rebuild) — use it only with `bf16`. For a w8 model that needs offload, use `offload: full` instead (works fine). |
| memory | `offload: auto-allocate` | runtime | ✓ | — | — | ✓ | engine decides resident/offload per component under the `max_vram` budget; **edge-only** |
| memory | `offload: auto-fit` | runtime | ✓ | — | — | ✓ | superset of `auto-allocate` that also auto-picks the DiT quant (q8_0→q4_k, **ignores the quant tier**); **edge-only** |
| memory | `vae-tiling` | runtime | ✓ | — | ✓ | ✓ (`vae_tiling: yes`) | high-resolution VAE tiling; diffusers does not list this memory_mode |
| parallel | `cfg-parallel` | runtime (multi-GPU) | ✓ | ✓ | — | — | CFG parallelism, ~1.77× @ 2 GPUs (H200) |
| parallel | `sequence-parallel` | runtime (multi-GPU) | ✓ | — | — | — | Ulysses sequence parallelism, up to 2.59× measured on FLUX (H200); **edge-only** |

**"job-orchestrable"**: currently `run.py` executes single-card, so the only dimensions a job section can sweep directly are **quant / cache / offload / vae_tiling**. `flash` is the on-by-default baseline (marked "indirect": to compare, turn it off via an engine-side switch); `cudnn-sdpa`/`sage` need build variants; `cfg-parallel`/`sequence-parallel` need a multi-GPU path — these three categories have no dedicated job field yet and must go through engine-side switches or the corresponding binary.

**Offload semantics (unified)**: all `offload` tiers now mean "**weights kept on CPU, staged to the GPU for compute**" — there is no longer any "compute on CPU" mode. `none`/`full` work on all three systems; `text-encoder-offload`/`vae-offload`/`dit-offload`/`auto-allocate`/`auto-fit` are edge-only; `sequential` is diffusers-only. Configuring a tier for a system that doesn't support it makes `run.py` skip that combination during expansion (prints `[run.py] skip → ...`). ⚠ **Per-component offload only moves ONE component**: `text-encoder-offload` now stages the TE in ~1G segments (the TE itself no longer OOMs), but it leaves the **DiT resident** — so on a large model (FLUX DiT ~22.7G, Qwen ~38G, Wan-14B) the run still OOMs on the resident DiT plus its compute buffer, regardless of TE staging. For large models use `full` (offloads all components) or combine `text-encoder-offload` with `dit-offload`; `max_vram` alone does not help because it caps the compute graph, not the resident-DiT weights.

**`max_vram` budget**: a job's system section and each `quant` object accept `max_vram: <GB>`, which caps the compute-graph VRAM via graph-cut segmentation (**edge-dit and sd.cpp** both honor it → `--max-vram`). When offloading without an explicit `max_vram`, the edge engine defaults to `0.85 × free VRAM` to enable segmented compute. sd.cpp additionally gets `--stream-layers` (paired with `--max-vram` + offload) to actually cap weight residency rather than just the compute graph.

---

## System capabilities (systems/*.yaml)

| capability | edge-dit.cpp | diffusers | stable-diffusion.cpp |
|---|---|---|---|
| role | primary (main subject) | python_reference (reference) | native_baseline (native baseline) |
| tasks | t2i / editing / t2v | t2i / editing / t2v | t2i / editing / t2v (**Wan: see note**) |
| backends | cuda / cpu / metal / vulkan | **cuda only** | cuda / cpu / metal / vulkan |
| parallel | cfg / sequence | — | — |
| memory_modes | quantization / cpu_offload (full) / component_placement (te / vae / dit offload, staged) / auto-allocate / auto-fit / vae_tiling / graph_cut (max_vram) | torch_dtype / cpu_offload (full) / sequential_offload / attention_backend | quantization / offload (full) / vae_tiling / graph_cut (max_vram + stream-layers) |

All three systems cover all three tasks (text-to-image / image-editing / text-to-video). edge-dit.cpp is the most capable tier (four backends + multi-GPU parallelism + the most memory modes); diffusers, as the Python reference, is CUDA-only with quantization via torch_dtype/Quanto; sd.cpp is the native baseline, four backends + quantization/offload/VAE tiling, but no multi-GPU parallelism and no exclusive cache. **Note on Wan: stable-diffusion.cpp supports Wan via component-separated loading (`--diffusion-model` + `--vae` + `--t5xxl`). It reads the official Comfy-Org repackaged component files (Wan DiT + `wan_2.1_vae` + `umt5_xxl`) natively — the diffusers `transformer/` directory layout is NOT recognized ("get sd version from file failed"), so the wan model yaml points sd.cpp at the official files via `stable_diffusion_cpp_wan_dit` / `_wan_vae` / `_wan_t5` refs.**

**sd.cpp commit lock**: because sd.cpp's offload / `--stream-layers` / `--max-vram` / component-loading behavior is version-specific, the benchmark records the validated commit in `systems/stable-diffusion-cpp.yaml` (`expected_commit: ea4e566`). Preflight only **warns** (does not block) when the checked-out sd.cpp is at a different commit — results may not reproduce and the e2e wrapper may need re-validating.

---

## Distilled (few-step) model support

Distilled variants come in two packaging formats, and support differs by format:

| format | distilled models | edge-dit | diffusers | sd.cpp |
|---|---|:--:|:--:|:--:|
| full diffusers directory (loads standalone) | `flux-schnell` | ✓ | ✓ | ✓ |
| full diffusers directory, but SD3 on sd.cpp uses an official single-file checkpoint | `sd35-medium-turbo` | ✓ | ✓ | — |
| transformer-only / single-weights-file (needs a `base_model_ref` for TE/VAE/scheduler) | `kontext-lightning`, `qwen-image-lightning`, `wan21-t2v-1.3b-distill` | ✓ | — | — |

**diffusers** currently runs the two full-directory distills (`flux-schnell`, `sd35-medium-turbo`): its runner takes a single `--model` directory and loads it standalone, which those two already are.

**stable-diffusion.cpp runs `flux-schnell` but not `sd35-medium-turbo`.** sd.cpp loads every family by separate components (`--diffusion-model`/`--vae`/`--clip-l`/…), not a single directory. FLUX ships a native fused single file (`flux1-schnell.safetensors`) sd.cpp reads directly, so schnell works. SD3/SD3.5 instead load from an **official all-in-one single-file checkpoint** (transformer + dual CLIP + T5 + VAE in one `.safetensors`, e.g. `sd3_medium_incl_clips_t5xxlfp16.safetensors`), which sd.cpp reads natively via `--model` alone — `sd3-medium` points at it through a `stable_diffusion_cpp_single_file` ref. (An earlier approach pre-converted the diffusers `transformer/` to sd.cpp's layout via `stable_diffusion_cpp_transformer_ref`; it loads but produces a blurry image, so it is kept only as a fallback. Feeding the raw diffusers `transformer/` shards directly fails with `get sd version from file failed`.) `sd35-medium-turbo` has no official single-file checkpoint prepared, so it is sd.cpp-unsupported for now; it is omitted from the sd.cpp section of the t2i job.

The other three distills ship as a transformer only (a `diffusers-transformer-only` subdir or a single `transformer-weights-file`) and declare a `base_model_ref` in their model yaml — the base supplies the text encoder / VAE / scheduler. **Only the edge-dit runner reads `base_model_ref`** (composing base + `--diffusion-model`); the diffusers and sd.cpp runners do not, so those three are edge-only for now. Putting one in a `diffusers:` / `stable-diffusion.cpp:` section fails at load (diffusers raises "neither a valid local path nor a valid repo id" on the transformer index). Adapting them means teaching each runner to load the base and swap in the distilled transformer.

---

> A method's `kind`/`needs calibration`/description is authoritative in `methods/<category>/<id>.yaml`; cross-system attribution is authoritative in its `cross_system` field (this table is aggregated from it). After adding a method/system, please sync this table.
