# edge-dit.cpp Benchmark

This directory defines the benchmark contract for edge-dit.cpp v0.1.0-alpha.
It is intentionally split into two layers:

- **Benchmark contract:** specifications, schemas, workload definitions, system
  definitions, hardware metadata, and prompts.
- **Benchmark execution:** runners, measurement utilities, orchestration,
  result checking, aggregation, and report generation.

The existing `benchmark/evaluation/` scripts remain quality-evaluation helpers.
They are reused by the harness for quality evaluation instead of replacing the
inference benchmark runner.

## v1 Goal

The v1 harness freezes and implements:

- what the v0.1.0-alpha benchmark is meant to prove;
- which workloads and systems are in scope;
- how latency, memory, quality, and parallel metrics are represented;
- which local paths are allowed only in local site overrides;
- which result directories must not be committed;
- how each run is expanded, executed, checked, aggregated, and reported.

Official numbers must be generated from result directories by the analysis
scripts. Do not hand-copy benchmark values into reports.

## Layout

```text
benchmark/
├── specs/       Benchmark policy and release-specific contract
├── configs/     Suite, workload, system, hardware, and local site configs
├── prompts/     Prompt sets used by workload configs
├── schemas/     Machine-readable result and environment schemas
├── runners/     System adapters for edge-dit.cpp, Diffusers, stable-diffusion.cpp, and xDiT
├── measurement/ Timing, environment, and resource measurement helpers
├── orchestration/ Suite validation, dry-run expansion, execution, resume, and result checks
├── analysis/    Result aggregation, table generation, and report generation
├── evaluation/  Quality metrics (CLIP, aesthetic, ImageReward, PSNR/SSIM/LPIPS,
│                directional-CLIP for edits, temporal consistency for video)
└── results/     Local benchmark outputs, ignored by Git
```

## Local Outputs

The following directories are local-only and ignored by Git:

```text
benchmark/results/
benchmark/cache/
benchmark/downloads/
benchmark/tmp/
```

Do not commit generated images, videos, raw logs, model weights, downloaded
models, or benchmark result directories.

## Validation

Validate the benchmark contract and configs:

```bash
python3 benchmark/orchestration/validate_config.py
```

Preview the FLUX pilot run matrix without executing models:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --dry-run
```

Build edge-dit.cpp before official runs. The CUDA build script defaults to the
performance profile:

```bash
bash scripts/build_cuda.sh
```

Run the README main-table suite. This is the source for the root README
Performance table and uses FLUX.1-dev, Stable Diffusion 3 Medium, and
Qwen-Image with the same 1024x1024, 50-step, batch-1 setting:

```bash
bash benchmark/scripts/run_readme_main_table.sh
```

The current README snapshot evaluates three representative text-to-image
workloads.

Run the reproducible FLUX.1-dev single-GPU e2e suite:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/nightly-YYYYMMDD
```

The README main-table workloads are the `*-1024-s50` text-to-image configs:
1024x1024, 50 denoising steps, seed 0, batch 1. FLUX.1-dev and Qwen-Image use
BF16; Stable Diffusion 3 Medium uses the matched precision available to each
runtime. Shorter 20-step workloads are smoke or diagnostic only and must not be
aggregated into README or main-table performance numbers. This snapshot
evaluates Stable Diffusion 3 Medium; Stable Diffusion 3.5 Large is not
included.

## Full Performance Page Suites

`docs/performance.md` is backed by a wider feature-results matrix than the root
README. Run these suites into separate frozen result roots, then aggregate those
roots into a `performance-page` summary.

```bash
# Build the five CUDA variants used by cuda-optimization-ablation.
bash benchmark/scripts/build_cuda_ablation_matrix.sh

# Task coverage: T2I, image editing, and video.
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/task-coverage.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/perf-task-coverage-YYYYMMDD

# Parallel scaling: SD3 Medium CFG-2 plus FLUX SP-1/SP-2/SP-4.
bash benchmark/scripts/run_parallel_tables.sh \
  --edge-build-dir build-cuda \
  --output-root benchmark/results/perf-parallel-YYYYMMDD

# Low-VRAM deployment profiles. The runner waits for a clean single GPU by
# default because the public table reports peak VRAM.
bash benchmark/scripts/run_resource_profiles_table.sh \
  --edge-build-dir build-cuda \
  --output-root benchmark/results/perf-resource-profiles-YYYYMMDD

# Quantization latency and memory trade-offs under the same FLUX 50-step
# workload used by the resource profiles table.
bash benchmark/scripts/run_quantization_table.sh \
  --edge-build-dir build-cuda \
  --output-root benchmark/results/perf-quantization-YYYYMMDD

# High-resolution VAE tiling memory trade-offs under a 2048x2048, 50-step FLUX
# workload.
bash benchmark/scripts/run_vae_tiling_table.sh \
  --edge-build-dir build-cuda \
  --output-root benchmark/results/perf-vae-tiling-YYYYMMDD

# Cache speed-quality. This is the only suite that gates public quality metrics.
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cache-quality.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/perf-cache-quality-YYYYMMDD

# CUDA optimization trade-offs.
for suite in cuda-optimization-ablation cuda-optimization-ablation-qwen; do
  python3 benchmark/orchestration/run_suite.py \
    --suite "benchmark/configs/suites/${suite}.yaml" \
    --site benchmark/configs/local/site-h200.yaml \
    --execute \
    --systems edge-dit.cpp \
    --output-root "benchmark/results/perf-${suite}-YYYYMMDD"
done
```

After cache quality evaluation, apply metric summaries back to the matching
target result directories:

```bash
python3 benchmark/analysis/apply_quality_metrics.py \
  --result-dir benchmark/results/perf-cache-quality-YYYYMMDD/.../target-run \
  --eval-summary benchmark/results/perf-cache-quality-YYYYMMDD/.../eval_summary/summary.json
```

Finally aggregate the frozen roots:

```bash
python3 benchmark/analysis/aggregate.py \
  --results-dir benchmark/results/perf-main-table-YYYYMMDD \
  --results-dir benchmark/results/perf-task-coverage-YYYYMMDD \
  --results-dir benchmark/results/perf-parallel-YYYYMMDD \
  --results-dir benchmark/results/perf-cache-quality-YYYYMMDD \
  --results-dir benchmark/results/perf-resource-profiles-YYYYMMDD \
  --results-dir benchmark/results/perf-quantization-YYYYMMDD \
  --results-dir benchmark/results/perf-vae-tiling-YYYYMMDD \
  --results-dir benchmark/results/perf-cuda-optimization-ablation-YYYYMMDD \
  --results-dir benchmark/results/perf-cuda-optimization-ablation-qwen-YYYYMMDD \
  --suite-id performance-page \
  --output benchmark/results/performance-page-YYYYMMDD/summary.json
```

Run the public model smoke suite across all locally configured public-preview
model families:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/model-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/model-smoke-YYYYMMDD
```

Run memory-constrained and optimization probe suites:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/memory.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/memory-YYYYMMDD

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/ablation.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/ablation-YYYYMMDD
```

Generate the local SenCache profile and run the repeated cache-mode matrix:

```bash
mkdir -p benchmark/cache

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cache-calibration-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/cache-calibration-smoke-YYYYMMDD

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cache-matrix.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/cache-matrix-YYYYMMDD
```

Run quick edge-dit.cpp parallel smoke suites:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/parallel-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/parallel-smoke-YYYYMMDD

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cfg-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/cfg-smoke-YYYYMMDD
```

For unattended official FLUX runs, use the release script. It waits for enough
free GPU memory, runs the 50-step suite, and regenerates the benchmark report:

```bash
bash benchmark/scripts/run_flux_s50_release.sh single
```

For 1/2/4 GPU sequence-parallel and xDiT runs:

```bash
bash benchmark/scripts/run_flux_s50_release.sh parallel
```

Run the reproducible FLUX.1-dev parallel e2e suite for edge-dit.cpp:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/nightly-YYYYMMDD
```

Record xDiT baseline preflight or skipped status without mixing it into
edge-dit.cpp numbers:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems xdit \
  --force-external-update \
  --output-root benchmark/results/nightly-YYYYMMDD
```

For a tiny smoke run, override the run counts:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --warmup-runs 0 \
  --measured-runs 1
```

External baselines that are configured with `force_latest_origin_main` require
an explicit opt-in before execution:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems xdit \
  --force-external-update \
  --warmup-runs 0 \
  --measured-runs 1 \
  --max-runs 1
```

The execution harness records process-level wall time, external GPU memory
samples, host RSS samples, environment metadata, command lines, logs, and
machine-readable `result.json` files. Component-level timing fields remain
`null` until a system-specific runner can emit them reliably.

The official latency boundary is **load-once e2e generation**:

- model loading is timed and recorded as `latency_ms.load`;
- warmup generations run in the same process and are excluded from steady-state
  statistics;
- measured samples time one complete generation after the model is already
  loaded;
- output encoding and file writing are outside the core latency where the
  backend makes that split available.

System adapters that participate in official comparisons must write
`runner_metrics.json` with `load_ms`, `warmup_ms`, and `measured_ms`. The
orchestrator refuses official adapters that do not produce this file, so
process startup and model loading cannot be accidentally counted as steady-state
generation latency. edge-dit.cpp, Diffusers, and stable-diffusion.cpp have
load-once wrappers for the README text-to-image main table; xDiT still needs a
matching official wrapper before its numbers can be reported.

The executable adapters live in `benchmark/scripts/`:

```text
benchmark/scripts/run_edge_e2e.py       # wraps build-cuda/bin/ed-sample
benchmark/scripts/run_edge_cli_once.py  # wraps build-cuda/bin/ed-cli for non-T2I smoke
benchmark/scripts/run_diffusers_e2e.py  # runs Diffusers load-once loops
benchmark/scripts/sd_cpp_e2e.cpp        # stable-diffusion.cpp C API load-once wrapper
benchmark/scripts/build_sd_cpp_e2e.py   # builds the stable-diffusion.cpp wrapper
benchmark/scripts/prepare_sdcpp_sd3_transformer.py
```

Inspect completed and pending suite entries:

```bash
python3 benchmark/orchestration/resume_suite.py \
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml
```

Resume a suite while skipping matching successful runs:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --resume
```

Aggregate results:

```bash
python3 benchmark/analysis/aggregate.py \
  --results-dir benchmark/results/nightly-YYYYMMDD \
  --include-workload flux1-dev-t2i-1024-s50 \
  --suite-id nightly-YYYYMMDD \
  --output benchmark/reports/v0.1.0-alpha/summary.json
```

Generate Markdown tables and a report:

```bash
python3 benchmark/analysis/generate_tables.py \
  benchmark/reports/v0.1.0-alpha/summary.json \
  --output benchmark/reports/v0.1.0-alpha/tables.md

python3 benchmark/analysis/generate_report.py \
  benchmark/reports/v0.1.0-alpha/summary.json \
  --tables benchmark/reports/v0.1.0-alpha/tables.md \
  --output benchmark/reports/v0.1.0-alpha/report.md
```

## Cross-System Matrix Evaluation

The cross-system matrix compares edge-dit.cpp against Diffusers and
stable-diffusion.cpp across models x precisions x VRAM budgets, and reports
component-level timing/VRAM plus per-task quality. It generates images/videos,
scores them with the correct metric per task, and aggregates one Markdown table.

### Dependencies

The quality metrics need a Python env with the packages in
`benchmark/requirements.txt` (install a CUDA-matched `torch` first):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r benchmark/requirements.txt
```

ImageReward runs on `transformers>=5` via the self-contained shim at the top of
`benchmark/evaluation/single_metric/cal_ir.py`; a clean `pip install image-reward`
needs no source edits. The LAION aesthetic-v2 head weights
(`sac+logos+ava1-l14-linearMSE.pth`) are not on PyPI — fetch them from the
improved-aesthetic-predictor project and place them under
`benchmark/cache/aesthetic/` (or pass `--aesthetic-weights`).

### One command (generate → evaluate → table)

```bash
nohup bash benchmark/scripts/run_cross_system_matrix.sh \
  > benchmark/results/matrix.log 2>&1 &
```

Stage 1 generates all suites (8 GPUs, one suite per card, `--resume`), stage 2
scores every successful run with `eval_all.py` (sharded by system across 3 GPUs;
quant-vs-FP16 pairing is same-system, so this sharding is safe), stage 3 writes
`tables_matrix.md`.

### Evaluate existing runs only (re-score without regenerating)

```bash
python3 benchmark/scripts/eval_all.py \
  --results-root benchmark/results/cross-system-matrix \
  --site benchmark/configs/local/site-4090.yaml
python3 benchmark/scripts/make_matrix_tables.py \
  --results-root benchmark/results/cross-system-matrix
```

`eval_all.py` reads `config.resolved.yaml` as the authoritative source of
task/system/precision/budget and the true prompt (via
`run_options.prompt_id → resolved_prompt_set`), loads each metric model once,
routes by task, and back-fills both `result.json.quality` and a per-run
`eval/quality.json`. Per-task metrics:

- **text-to-image**: CLIP, aesthetic, ImageReward.
- **image-editing**: directional-CLIP (edit direction vs instruction),
  keep-SSIM/keep-LPIPS (output vs input, edit magnitude), aesthetic, ImageReward.
- **text-to-video**: per-frame CLIP & aesthetic, plus adjacent-frame temporal
  consistency (LPIPS/SSIM mean + flicker std).
- Quantization loss (PSNR/SSIM/LPIPS) is paired against the same-system FP16 run
  only, so it is meaningful only within a system (not across systems).

### Reading the numbers

- Compare inference speed with **DiT pure-sampling ms** (component denoise time),
  not end-to-end latency: end-to-end is contaminated by one-time on-the-fly
  weight quantization (tens to hundreds of seconds) and model load. The table's
  "口径/boundary" column tags each latency as load-once vs. includes-load.
- For stable-diffusion.cpp the convert step folds into the denoise timing too,
  so its DiT sampling ms is also inflated under quantized dtypes.
- Treat **q8** as the primary usable-quality quantization tier; q4 is an
  extreme-VRAM reference point with visible quality loss.
- Cross-system absolute quality scores (CLIP/aesthetic/IR) are comparable; if a
  gap looks implausibly large, check alignment before drawing conclusions.

