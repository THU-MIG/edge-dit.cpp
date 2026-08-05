# Computation Reuse

[← Back to README](../../README.md)

## 1. Overview

A diffusion sampling loop runs the same transformer many times, once per
denoising step, over inputs that change only gradually from one step to the
next. Computation reuse exploits that redundancy: instead of recomputing the
full transformer forward at every step, edge-dit.cpp can reuse a cached result
from an earlier step and skip most of the work for the current one.

Reuse is a speed-quality tradeoff. Skipping a forward is faster, but the reused
value is an approximation of what a full compute would have produced, so overly
aggressive reuse degrades output quality. Every cache method is therefore built
around a *decision* — when it is safe to reuse — and a *reconstruction* — what
to substitute for the skipped compute.

These controls are experimental and workload dependent. The useful setting
depends on the model, resolution, step count, and how much quality loss is
acceptable, so validate output for the exact configuration you plan to run.

The precision and memory optimizations that reuse combines with are documented
separately in [Model representation and precision](model-representation-and-precision.md)
and [Memory-efficient execution](memory-efficient-execution.md).

---

## 2. Scope

### Timestep-level reuse

The unit of reuse is a sampling step. On a *full* step the transformer runs
normally and its result (or an intermediate residual) is stored. On a *reuse*
step the stored value drives the output and the local forward is skipped. A
cache method decides, per step, which of the two a step becomes.

Reuse is confined to an active window of the sampling schedule, expressed as a
fraction of the run: the first and last steps — where the latent changes most
and errors are most visible — are computed in full by default, and reuse is
only considered between `--cache-start` (default `0.15`) and `--cache-end`
(default `0.95`). A configurable number of warmup steps at the start of the
window is also always computed in full.

### Reuse granularity

Methods differ in *what* they cache, captured by three granularities:

| Granularity | What is cached and reused | Methods |
|---|---|---|
| **Output** | The whole transformer output (or its step-to-step difference). A reuse step substitutes the cached output directly. | EasyCache, UCache, DBCache, CacheDiT |
| **Feature** | An internal block-stack residual. A reuse step re-runs only the cheap surrounding layers and injects the cached residual in the middle. | TaylorSeer, MagCache, SenCache |
| **Probe** | A small "probe" of the forward is computed to measure change, then the decision selects full compute or a reconstructed reuse. | DiCache |

Feature and Probe reuse require the model to expose a block-stack seam so the
forward can be cut and re-entered. Under sequence parallelism, or on models that
cannot cut their stack, these methods fall back to computing the step in full.

### Decision strategies

Reuse decisions are per method:

- **Threshold on change** — EasyCache and UCache track an error/similarity
  signal across steps and reuse while it stays under a threshold; UCache also
  accumulates the error it skips and decays it (`--cache-error-decay`).
- **Block-count split** — DBCache and CacheDiT always compute a configured
  number of front (`--cache-fn-blocks`) and back (`--cache-bn-blocks`) blocks
  and reuse the middle when the residual difference is small enough
  (`--cache-residual-threshold`).
- **Extrapolation** — TaylorSeer reconstructs a skipped feature from a Taylor
  series over recent steps (`--cache-taylor-order`, `--cache-taylor-skip`).
- **Calibrated reuse** — MagCache and SenCache consult a precalibrated
  per-model table (a magnitude-ratio table / a sensitivity profile) to decide
  reuse. These are the only methods that support calibration; see §4.
- **Static or dynamic schedules** — a fixed steps-computation mask
  (`--cache-scm-mask`) can force which steps compute versus reuse, either as a
  static policy (`--cache-static-scm`) or blended with the dynamic decision.

---

## 3. Public Interfaces

### CLI cache flags

`ed-cli` and `ed-sample` select and tune reuse through `--cache <mode>` plus the
`--cache-*` flags. Supported modes:

```text
off  easycache  ucache  dbcache  taylorseer  cache-dit  magcache  dicache  sencache
```

The full flag list, defaults, and worked examples are in
[Command line usage](../cli.md#performance-flags).

### C API cache configuration

The C API carries the same configuration on `ed_sample_params_t` (see
[`include/edge-dit.h`](../../include/edge-dit.h)). `ed_sample_params_init()`
fills in the defaults; the relevant fields are:

```c
ed_cache_mode_t cache_mode;                     /* ED_CACHE_DISABLED by default */
float           cache_reuse_threshold;          /* EasyCache/UCache */
float           cache_start_percent;            /* 0.15 */
float           cache_end_percent;              /* 0.95 */
float           cache_error_decay_rate;         /* UCache; 1.0 */
bool            cache_use_relative_threshold;
bool            cache_reset_error_on_compute;
int             cache_Fn_compute_blocks;        /* DBCache/CacheDiT; 8 */
int             cache_Bn_compute_blocks;        /* DBCache/CacheDiT; 0 */
float           cache_residual_diff_threshold;  /* unset by default; method-specific */
float           cache_max_accumulated_residual_diff;
int             cache_max_warmup_steps;         /* 8 */
int             cache_max_cached_steps;
int             cache_max_continuous_cached_steps;
int             cache_taylorseer_n_derivatives; /* TaylorSeer; 1 */
int             cache_taylorseer_skip_interval; /* TaylorSeer; 1 */
const char *    cache_scm_mask;                 /* e.g. "1,0,0,1" */
bool            cache_scm_policy_dynamic;
const char *    cache_calibrate_path;           /* MagCache/SenCache */
const char *    cache_profile_path;             /* MagCache/SenCache */
```

`ed_cache_mode_supports_calibration(mode)` reports whether a mode consumes a
calibrated profile (true only for `ED_CACHE_MAGCACHE` and `ED_CACHE_SENCACHE`).

### Python binding configuration

The Python binding exposes the numeric and mask cache fields on its generation
config (`cache_mode`, `cache_reuse_threshold`, `cache_start_percent`,
`cache_end_percent`, `cache_error_decay_rate`, `cache_use_relative_threshold`,
`cache_reset_error_on_compute`, `cache_Fn_compute_blocks`,
`cache_Bn_compute_blocks`, `cache_residual_diff_threshold`,
`cache_max_accumulated_residual_diff`, `cache_max_warmup_steps`,
`cache_max_cached_steps`, `cache_max_continuous_cached_steps`,
`cache_taylorseer_n_derivatives`, `cache_taylorseer_skip_interval`,
`cache_scm_mask`, `cache_scm_policy_dynamic`). `cache_mode` accepts either the
integer enum value or a name string.

Two limitations relative to the C API: the binding does not expose the
`cache_calibrate_path` / `cache_profile_path` fields, so calibration is driven
from the CLI or C API rather than Python; and the name map resolves the first
six modes (`off`, `easycache`, `ucache`, `dbcache`, `taylorseer`, `cache-dit`),
so `magcache`, `dicache`, and `sencache` must be selected by their integer enum
value (6, 7, 8). See [API and bindings](../api.md) for the binding reference.

---

## 4. Calibration

MagCache and SenCache decide reuse from a per-model profile that must exist
before an accelerated run:

- Produce a profile with a calibration run: pass a calibration-capable
  `--cache` mode together with `--cache-calibrate <path>`. The run performs full
  computation and writes the table (MagCache: magnitude-ratio table; SenCache:
  sensitivity profile) to `<path>`.
- Consume it on subsequent runs with `--cache-profile <path>`.

SenCache requires a profile: it must be given `--cache-profile` (or produced in
the same session with `--cache-calibrate`) or it cannot run. A profile is
model- and schedule-specific — recalibrate when the model, resolution, or step
schedule changes.

---

## 5. Validation Plan

Because reuse trades quality for speed, evaluate both together rather than
speed alone.

- **Speed and quality measurement** — measure end-to-end latency (and step
  count skipped) against an uncached baseline for the same seed, and compare
  image quality against that baseline (e.g. PSNR/SSIM/LPIPS, or perceptual
  inspection). The reproducible harness under [`benchmark/`](../../benchmark/README.md)
  runs cached-versus-baseline suites and collects both.
- **Per-model compatibility** — validate each cache method per model family and
  resolution before relying on it. Feature/Probe methods depend on the
  block-stack seam and disable themselves when it is unavailable (e.g. under
  sequence parallelism); confirm reuse actually engages for your configuration.
- **Calibration data requirements** — for MagCache and SenCache, calibrate on a
  representative prompt set at the target resolution and step schedule, and
  recalibrate whenever any of those change.

---

## Related Documentation

- [performance (RTX 4090)](../performance-4090.md)
- [performance (H200): Cache results](../performance-H200.md)
- [Command line usage](../cli.md)
- [Supported models and usage](../models.md)
- [API and bindings](../api.md)
