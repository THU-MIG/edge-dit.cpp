# Performance and Benchmarks

[Back to README](../README.md) · [Current RTX 4090 snapshot](performance-4090.md)

This page is the **NVIDIA H200 benchmark snapshot**. All tables on this
page were measured on H200 and their end-to-end / `Median` / `P90` latencies are
**load-inclusive**. For the current **RTX 4090** snapshot (load-excluded
end-to-end), see [Performance and benchmarks (RTX 4090)](performance-4090.md).
This page answers the feature questions behind the README claims:
single-GPU runtime, parallel execution, computation reuse, Low-VRAM execution,
quantization, VAE tiling, and CUDA operator optimization.

The root README intentionally keeps only one main performance table. This page
keeps the supporting feature results and the reproducibility contract.

## Feature Results Summary

All rows below come from frozen local result roots under `benchmark/results/`.
Latency is reported in seconds. Lower latency is better. For "Relative to
Diffusers", values above 1.00x mean edge-dit.cpp is slower than Diffusers for
that workload; "Speedup" values above 1.00x mean the optimized configuration is
faster than its baseline.

| Capability | Representative workload | Compared configurations | Headline result |
|---|---|---|---|
| Native single-GPU runtime | FLUX.1-dev, SD3 Medium, Qwen-Image; 1024x1024, 50 steps | edge-dit.cpp vs Diffusers vs stable-diffusion.cpp | edge-dit.cpp is within 1.07x-1.19x of Diffusers and 2.68x-5.86x faster than stable-diffusion.cpp. |
| CFG parallelism | SD3 Medium, 1024x1024, 50 steps, CFG 4.5 | 1 GPU vs CFG-2 | CFG-2 reduces latency from 4.387 s to 2.480 s, a 1.77x speedup at 88.4% efficiency. |
| Sequence parallelism | FLUX 1024x1024 and 2048x2048, reported as 50-step latency | SP-1 vs SP-2 vs SP-4 | With the cuDNN-runtime-verified CUDA performance build, SP-4 reaches 2.26x on FLUX 1024 and 2.59x on FLUX 2048. |
| Computation reuse | FLUX.1-dev, 1024x1024, 50 steps, 8 prompts x 3 seeds | Full compute vs EasyCache, CacheDiT, MagCache, DiCache, SenCache | MagCache is the fastest cache point at 2.69x; EasyCache reaches 2.09x; tuned SenCache now reaches 1.92x with 29-30/50 reused steps. |
| Weight quantization | FLUX.1-dev, 1024x1024, 50 steps | BF16 vs Q8_0/Q6_K/Q4_K | Q4_K reduces peak VRAM from 36329 MiB to 15511 MiB, with 1.99x latency slowdown. |
| Low-VRAM execution | FLUX.1-dev, 1024x1024, 50 steps | BF16 performance profile vs Q4_K + CPU placement/offload + graph budget + VAE tiling | The 8 GiB budget profile reaches 8119 MiB peak VRAM on H200, with a 9.52x latency slowdown. |
| VAE tiling | FLUX.1-dev, 2048x2048, 50 steps | Untiled vs 2x2 vs 4x4 tiling | 4x4 VAE tiling reduces peak VRAM by 32.5%, from 56357 MiB to 38043 MiB, with no observed end-to-end latency penalty. |
| CUDA operator optimization | FLUX.1-dev and Qwen-Image, 1024x1024, 50 steps | Generic ggml CUDA path vs optimized non-cuDNN CUDA builds | FLUX latency drops from 22.896 s to 18.835 s, a 1.22x speedup; Qwen gains 1.05x from fused modulation over the CUDA Norm + RoPE build. |

## Overall Performance

Reproduce the README main table as a run.py job — a `text-to-image` manifest
with `edge-dit` / `diffusers` / `stable-diffusion.cpp` sections over the three
workloads below, run against an H200 site with the CUDA `performance` build:

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<readme-main-table>.yaml \
  --site benchmark/sites/<siteh200>.yaml
```

See [the benchmark harness README](../benchmark/README.md) for the manifest
schema and the cross-system example manifest to start from.

The current snapshot evaluates three representative text-to-image workloads.

Contract: local NVIDIA H200 node, CUDA `performance` profile, 1024x1024,
50 denoising steps, batch 1, seed 0, 2 warm-up runs, 10 measured runs,
load-once generation. FLUX.1-dev and Qwen-Image use BF16; Stable Diffusion 3
Medium uses the matched precision available to each runtime. Output encoding is
outside `Median` and `P90`. Note: on this H200 page `Median`/`P90` are
**load-inclusive end-to-end** latency; the RTX 4090 tables in
[performance-4090.md](performance-4090.md) report load-excluded numbers, so the
two are not directly comparable.

| Model | System | Load (s) | Median (s) | P90 (s) | Peak VRAM (MiB) |
|---|---|---:|---:|---:|---:|
| FLUX.1-dev | edge-dit.cpp | 6.645 | 10.784 | 10.861 | 38341 |
| | Diffusers | 14.531 | 10.040 | 10.048 | 37711 |
| | stable-diffusion.cpp | 1.333 | 30.371 | 30.379 | 40331 |
| Stable Diffusion 3 Medium | edge-dit.cpp | 5.840 | 4.003 | 4.049 | 20833 |
| | Diffusers | 11.244 | 3.376 | 3.381 | 20283 |
| | stable-diffusion.cpp | 1.457 | 10.740 | 10.797 | 22997 |
| Qwen-Image | edge-dit.cpp | 11.621 | 10.697 | 10.736 | 59725 |
| | Diffusers | 25.220 | 9.558 | 9.565 | 60935 |
| | stable-diffusion.cpp | 1.782 | 62.671 | 62.728 | 61879 |

Load time follows each runtime's reported initialization boundary and may
reflect different weight materialization or memory-mapping strategies.
Generation latency is the primary cross-runtime performance metric.

This snapshot evaluates Stable Diffusion 3 Medium. Stable Diffusion 3.5 Large
is not included.

## Parallel Execution

Parallel tables report latency and scaling only. Quality metrics are reserved
for the cache speed-quality suite.

These parallel tables are not produced by a run.py job — the harness front end
runs single-card only and has no multi-GPU job field. They are reproduced by
driving `ed-cli` directly with its parallel flags (`--devices`,
`--cfg-parallel-size`, `--sp-size`) on the CUDA `performance` build, with the
NVIDIA CUDA runtime libraries on `LD_LIBRARY_PATH` so cuDNN SDPA executes
instead of falling back to ggml CUDA flash attention. See
[Parallel execution](optimization/parallel-execution.md) and
[Command line usage](cli.md#parallel-execution) for the launch commands. The
older `run_parallel_tables.sh` suite that automated this has been archived under
`benchmark/archive/`.

### CFG Parallelism

Contract: Stable Diffusion 3 Medium, 1024x1024, 50 steps, CFG scale 4.5, BF16,
batch 1, seed 0, load-once generation.

| Mode | GPUs | Median | P90 | Speedup | Efficiency | Max VRAM / GPU |
|---|---:|---:|---:|---:|---:|---:|
| Single GPU | 1 | 4.387 s | 4.399 s | 1.00x | 100.0% | 20831 MiB |
| CFG parallel | 2 | 2.480 s | 2.504 s | 1.77x | 88.4% | 22113 MiB |

### Sequence Parallelism

Contract: local H200 node, CUDA `performance` profile, BF16, batch 1, seed 0,
1 warm-up run, 5 measured runs, load-once generation. FLUX 1024 rows are
measured at 50 steps. FLUX 2048 rows are reported on the same 50-step latency
scale; speedup, efficiency, and peak VRAM come from the measured SP run. This
rerun uses the same CUDA performance build for SP-1, SP-2, and SP-4, with the
Python NVIDIA CUDA runtime libraries on `LD_LIBRARY_PATH` so cuDNN SDPA is
confirmed to execute instead of falling back to ggml CUDA flash attention.

| Workload | GPUs | Median | Speedup | Efficiency | Max VRAM / GPU |
|---|---:|---:|---:|---:|---:|
| flux1-dev-t2i-1024-s50 | 1 | 10.185 s | 1.00x | 100.0% | 38349 MiB |
| flux1-dev-t2i-1024-s50 | 2 | 6.531 s | 1.56x | 78.0% | 39477 MiB |
| flux1-dev-t2i-1024-s50 | 4 | 4.509 s | 2.26x | 56.5% | 40247 MiB |
| flux1-dev-t2i-2048-s50 | 1 | 51.223 s | 1.00x | 100.0% | 56365 MiB |
| flux1-dev-t2i-2048-s50 | 2 | 31.210 s | 1.64x | 82.1% | 57061 MiB |
| flux1-dev-t2i-2048-s50 | 4 | 19.765 s | 2.59x | 64.8% | 57831 MiB |

The runner enables cuDNN SDPA profiling and fails the run if stderr shows a
cuDNN graph-build fallback or missing CUDA runtime library, so fallback timings
are not mixed into this table.

## Computation Reuse

Reproduce this table as a run.py job: a `text-to-image` manifest with an
`edge-dit` FLUX.1-dev section sweeping the `cache` dimension over the methods
below (plus `none` for the full-compute reference), run on the CUDA
`performance` build:

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<cache-reuse>.yaml \
  --site benchmark/sites/<h200>.yaml
```

The manifest covers the full-compute/cache matrix, the MagCache and DiCache
method-default rows, and the tuned SenCache row; SenCache also needs a
calibration profile generated under this workload first (see
[Computation reuse](optimization/computation-reuse.md#4-calibration)). The
PSNR/LPIPS columns are computed by the evaluation stage against matched
full-compute prompt/seed outputs.

Contract: FLUX.1-dev, 1024x1024, 50 steps, BF16, batch 1, 8 prompts x 3 seeds,
1 warm-up run, 5 measured runs, load-once generation with the build used by the
README main table. PSNR and LPIPS compare each method against the matching
`Full compute` prompt/seed output. CLIP is left out of the public table for
this snapshot.

Speedup below is computed against the matched full-compute subset for each row.
The MagCache and DiCache rows use their method-specific default thresholds. The
`0.08` residual threshold remains the DBCache/CacheDiT default and is only
applied to MagCache or DiCache when explicitly passed. The SenCache row uses a
50-step SenCache profile generated under this workload and
`cache_residual_threshold=0.60`.

| Method | Samples | Granularity | Median | Speedup vs Matched Full | Peak VRAM | Saved Steps | PSNR | LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Full compute | 24/24 | full | 10.765 s | 1.00x | 38341 MiB | - | Reference | 0.0000 |
| EasyCache | 24/24 | output | 5.154 s | 2.09x | 38341 MiB | 27/50 | 26.34 | 0.1016 |
| CacheDiT | 24/24 | block/output | 6.406 s | 1.68x | 38341 MiB | 21/50 | 29.03 | 0.0720 |
| MagCache method default | 24/24 | feature | 4.001 s | 2.69x | 38485 MiB | 35/50 | 23.30 | 0.1754 |
| DiCache method default | 24/24 | probe | 6.514 s | 1.65x | 39471 MiB | 30/50 | 26.89 | 0.0995 |
| SenCache tuned t=0.60 | 24/24 | feature | 5.613 s | 1.92x | 40923 MiB | 29/50 | 25.78 | 0.1231 |

## Low-VRAM Execution

### Deployment Profiles

Reproduce this table as a run.py job: a FLUX.1-dev `edge-dit` manifest with one
`quant` object per profile row, combining the precision (`fp16`/`q8`/`q4_k`),
the `offload` tier (e.g. `text-encoder-offload`, or `full` for the parameter-offload rows),
`max_vram`, and `vae_tiling` documented for that profile:

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<low-vram-profiles>.yaml \
  --site benchmark/sites/<h200>.yaml
```

The six rows map to six `quant`-object tiers in one `edge-dit` section (per-tier
overrides of `offload` / `max_vram` / `vae_tiling`). Run on a clean GPU so peak
VRAM is not polluted by unrelated jobs.

Contract: FLUX.1-dev, 1024x1024, 50 steps, BF16/Q8_0/Q4_K depending on profile,
batch 1, seed 0, 1 warm-up run, 5 measured runs, load-once generation.

| Profile | Weight | CPU Placement / Offload | Graph Budget | VAE Tiling | Median | Slowdown | Peak VRAM | Host RAM |
|---|---|---|---:|---|---:|---:|---:|---:|
| Performance | BF16 | none | unlimited | off | 10.960 s | 1.00x | 36553 MiB | 17049 MiB |
| Memory-balanced | Q8_0 | none | unlimited | off | 20.508 s | 1.87x | 21057 MiB | 40782 MiB |
| 24 GiB target | Q4_K | text encoder CPU | unlimited | off | 29.235 s | 2.67x | 12329 MiB | 55819 MiB |
| 16 GiB target | Q4_K | text encoder CPU | unlimited | on | 28.563 s | 2.61x | 9057 MiB | 54431 MiB |
| 8 GiB budget experimental | Q4_K | text encoder CPU + parameter offload | 8 GiB | on | 104.378 s | 9.52x | 8119 MiB | 54381 MiB |
| 12 GiB graph-budget diagnostic | Q4_K | text encoder CPU + parameter offload | 12 GiB | on | 74.033 s | 6.75x | 11041 MiB | 54442 MiB |

The lowest-memory profile in this run reduces peak device memory from
36553 MiB to 8119 MiB on H200 under an 8 GiB budget emulation. The cost is
latency: 104.378 s vs 10.960 s for the BF16 performance profile. The 16 GiB
target is the more practical low-VRAM point in this matrix at 9057 MiB and a
2.61x slowdown. The Q8_0 memory-balanced row uses the matching quantization
rerun because the resource-profile and quantization Q8_0 commands are
identical; the earlier resource-profile Q8_0 run sampled a higher external
peak VRAM.

### Quantization Trade-Off

Reproduce this table as a run.py job: a FLUX.1-dev `edge-dit` manifest whose
`quant` list carries one tier per row (`bf16`, `q8`, and `quant` objects
`{type: q6_k}` / `{type: q4_k}`, plus a `q4_k` tier with
`tensor_type_rules` for the precision-rules row):

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<quantization>.yaml \
  --site benchmark/sites/<h200>.yaml
```

Each tier expands to an independent run process. The quantized rows use the
same FLUX model path with per-tier weight-type selection; no separate
prequantized checkpoint is configured. Run on a clean GPU so peak VRAM is not
polluted by unrelated jobs.

Contract: FLUX.1-dev, 1024x1024, 50 steps, BF16/Q8_0/Q6_K/Q4_K depending on
row, batch 1, seed 0, 1 warm-up run, 5 measured runs, load-once generation.

| Weight Type | Load | Median | Slowdown | Peak VRAM | Host RAM | Policy |
|---|---:|---:|---:|---:|---:|---|
| BF16 | 7.156 s | 11.082 s | 1.00x | 36329 MiB | 16875 MiB | - |
| Q8_0 | 18.429 s | 20.508 s | 1.85x | 21057 MiB | 40782 MiB | - |
| Q6_K | 13.518 s | 26.014 s | 2.35x | 17573 MiB | 54567 MiB | - |
| Q4_K | 41.509 s | 22.056 s | 1.99x | 15511 MiB | 55634 MiB | - |
| Q4_K + precision rules | 41.145 s | 22.007 s | 1.99x | 15511 MiB | 55351 MiB | norm=f16,bias=f32 |

### VAE Tiling

Reproduce this table as a run.py job: a FLUX.1-dev `edge-dit` manifest at
2048x2048 with a `vae_tiling` tier for the untiled (`no`) and tiled (`yes`)
rows:

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<vae-tiling>.yaml \
  --site benchmark/sites/<h200>.yaml
```

The `vae_tiling` field toggles tiling on/off; the tile granularity behind the
2x2 vs 4x4 rows is the relative tile size (see
[Memory-efficient execution](optimization/memory-efficient-execution.md#5-vae-tiling)),
so the finer-grid rows need the tile-size control driven through `ed-cli`
directly. Run on a clean GPU so peak VRAM is not polluted by unrelated jobs.

Contract: FLUX.1-dev, 2048x2048, 50 steps, BF16, batch 1, seed 0,
1 warm-up run, 5 measured runs, load-once generation.

| VAE Mode | Tile Layout | Median | Slowdown | Peak VRAM | VRAM Reduction | Host RAM |
|---|---|---:|---:|---:|---:|---:|
| untiled | full image | 52.641 s | 1.00x | 56357 MiB | 0.0% | 16763 MiB |
| tiled | approximately 2x2 | 51.936 s | 0.99x | 40205 MiB | 28.7% | 17206 MiB |
| tiled | approximately 4x4 | 50.398 s | 0.96x | 38043 MiB | 32.5% | 16451 MiB |

VAE tiling is most visible at high resolution. In this 2048x2048 run, 4x4
tiling saves 18314 MiB of peak device memory. The tiled configurations showed
no end-to-end latency penalty in this run; VAE tiling should be treated as a
device-memory optimization, not a formal inference-speed optimization.

## Recommended Runtime Profiles

These profiles summarize measured deployment trade-offs. Device classes are
recommendations derived from the observed peak VRAM on the H200 benchmark
system; except where explicitly stated, they have not yet been validated on
physical GPUs with the listed memory capacities.

| Profile                     | Suggested deployment target          | Configuration                                                                    | Measured evidence                                                                                                                                                                     |
| --------------------------- | ------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Maximum performance         | 40 GB+ CUDA GPU                      | BF16 + cuDNN SDPA + CUDA Norm, CUDA RoPE, and fused modulation                   | The FLUX.1-dev main-table run completes in 10.784 s at 38341 MiB peak VRAM.                                                                                                           |
| High-throughput approximate | 40 GB+ CUDA GPU                      | BF16 + EasyCache or MagCache                                                     | EasyCache reaches 2.09x speedup; MagCache reaches 2.69x but introduces a larger approximation error.                                                                                  |
| Balanced device memory      | 32 GB class CUDA GPU                 | Q8_0                                                                             | Q8_0 uses 21057 MiB peak VRAM with a 1.85x latency slowdown relative to BF16.                                                                                                         |
| 24 GiB deployment           | 24 GiB dedicated CUDA GPU            | Q4_K                                                                             | Q4_K uses 15511 MiB peak VRAM with a 1.99x latency slowdown.                                                                                                                         |
| Low-VRAM deployment         | 12-16 GiB CUDA GPU                   | Q4_K + text encoder on CPU + VAE tiling                                          | The measured profile uses 9057 MiB peak VRAM with a 2.61x latency slowdown.                                                                                                           |
| Minimum-VRAM experimental   | 8 GiB budget emulation on H200       | Q4_K + text encoder on CPU + parameter offload + 8 GiB graph budget + VAE tiling | The profile reaches 8119 MiB peak VRAM with a 9.52x latency slowdown and 54381 MiB peak host RAM. It has not yet been validated on a physical 8 GiB GPU.                              |

The deployment profiles optimize device-memory usage, not total system-memory
usage. CPU placement, runtime quantization, and parameter offload may require
substantial host RAM. In particular, the minimum-VRAM profile trades both host
memory and latency for lower GPU-memory usage.

Mixed-precision rules have negligible performance overhead in this snapshot,
but they are not quality-characterized and are not listed as a deployment
recommendation.

The measured 12 GiB graph-budget configuration is not recommended as a
deployment profile because it is dominated by the simpler low-VRAM
configuration: it uses more device memory and has substantially higher latency.

## Reproducibility

The benchmark harness drives reproduction through a single front end,
`benchmark/run.py`, which reads a job manifest (`benchmark/jobs/*.yaml`,
declaring models × quantization/acceleration tiers × task) and a machine site
file (`benchmark/sites/*.yaml`, holding the local binary and model paths), then
chains generation → evaluation → table aggregation in one process:

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<manifest>.yaml \
  --site benchmark/sites/<machine>.yaml
```

Add `--dry-run` to print the expanded run plan without generating, and
`--device N` to lock the job to one GPU. Report tables land in
`benchmark/reports/<name>/` (committed) and raw per-run artifacts in
`benchmark/results/<name>/` (git-ignored). See
[the benchmark harness README](../benchmark/README.md) for the manifest schema,
the model/method libraries, and ready-made `example-*` manifests.

The single-card tables on this page — the README main table, quantization,
Low-VRAM profiles, and VAE tiling — reproduce as run.py jobs: write a manifest
that pins the workload, precision, offload, and VAE-tiling tiers documented in
each table's contract, then run the command above. The offline
quantization/tiling/placement flags map to the manifest fields `quant`,
`offload`, `max_vram`, and `vae_tiling`.

The parallel-execution (CFG / sequence-parallel) and CUDA-operator-ablation
tables are **not** currently expressible as a run.py job: run.py executes
single-card runs and has no job field for multi-GPU parallelism or for
build-variant operator ablations yet. Those tables were produced by the
multi-GPU / build-variant tooling that has since been archived under
`benchmark/archive/` (the old `orchestration/`, `configs/suites/`, and
`run_*_table.sh` suite scripts); reproduce them for now with the parallel and
build-variant switches described in
[Parallel execution](optimization/parallel-execution.md) and
[Build and installation](build.md), driving `ed-cli` / `ed-sample` directly.

For any run that relies on cuDNN SDPA, put the NVIDIA CUDA runtime libraries on
`LD_LIBRARY_PATH` and confirm from stderr that cuDNN SDPA executes rather than
falling back to ggml CUDA flash attention; a fallback changes the timings and
must not be mixed into these tables.

### Re-evaluating without re-generating

To recompute metrics or regenerate tables from existing run artifacts without
re-running generation, call the standalone evaluation scripts against a result
root:

```bash
python3 benchmark/scripts/eval_all.py \
  --results-root benchmark/results/<name> \
  --site benchmark/sites/<machine>.yaml
python3 benchmark/scripts/make_matrix_tables.py \
  --results-root benchmark/results/<name> \
  --output benchmark/reports/<name>/tables.md
```

`eval_all.py` treats each run's `config.resolved.yaml` as authoritative
(task / system / precision / prompt), routes quality metrics by task (t2i CLIP /
aesthetic / ImageReward; editing directional CLIP + preservation SSIM / LPIPS;
video per-frame + temporal consistency), and backfills `result.json`. The cache
speed-quality table's PSNR and LPIPS are computed here, against prompt- and
seed-matched full-compute outputs.

## Limitations

* All reported results were measured on NVIDIA H200 GPUs. They validate the
  current CUDA implementation and should not be interpreted as performance
  results on consumer GPUs, Apple Silicon, Jetson, or other edge devices.

* The Low-VRAM and quantized profiles trade device memory for additional
  latency and host-memory usage. The 8 GiB budget profile was evaluated on H200
  and has not yet been validated on a physical 8 GiB GPU.

* The Stable Diffusion benchmark uses Stable Diffusion 3 Medium. Sequence
  parallelism is currently reported for FLUX only.


## Related Documentation

- [Performance and benchmarks (RTX 4090)](performance-4090.md) — the current RTX 4090 snapshot (load-excluded end-to-end)
- [Benchmark harness](../benchmark/README.md)
- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [Command line usage](cli.md)
- [Memory-efficient execution](optimization/memory-efficient-execution.md)
- [Graph and operator optimization](optimization/graph-and-operator-optimization.md)
- [Computation reuse](optimization/computation-reuse.md)
- [Parallel execution](optimization/parallel-execution.md)
