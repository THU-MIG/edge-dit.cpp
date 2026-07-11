# Performance and Optimization

[← Back to README](../README.md)

This document covers build profiles, CUDA operators, memory options,
parallelism, cache reuse, profiling, and benchmark expectations.

## Performance Configuration

Official performance work should use:

```bash
CUDA_HOME=/path/to/cuda \
NCCL_ROOT=/path/to/nccl \
CUDNN_ROOT=/path/to/cudnn \
MPI_HOME=/path/to/mpi \
ED_BUILD_PROFILE=performance \
BUILD_DIR=build-cuda-performance \
bash scripts/build_cuda.sh
```

The `performance` profile enables NCCL, MPI, cuDNN SDPA, CUDA Norm, CUDA RoPE,
CUDA Modulation, and parallel runtime support. Missing required dependencies
fail explicitly before CMake configure.

The `minimal` profile is for build and functional validation. It is not the
configuration used for official benchmark results.

Full `performance` profile validation is still pending for v0.1.0-alpha
release sign-off. Do not publish new official benchmark numbers until that gate
is complete.

## Memory Optimization

Common memory controls:

```bash
--vae-tiling
--vae-tile-size <float>
--offload-to-cpu
--keep-text-encoder-on-cpu
--keep-vae-on-cpu
--max-vram <GB>
```

Notes:

- VAE tiling can reduce decode memory at the cost of extra work.
- CPU offload can reduce VRAM pressure but may add host-device transfer
  overhead.
- `--max-vram` limits graph allocation pressure but should be validated against
  the selected model and backend.

## Attention and CUDA Operators

The CUDA path includes optimized helper code for:

- cuDNN SDPA
- Flash Attention selection through the runtime
- CUDA Norm
- CUDA RoPE
- CUDA Modulation
- Flux sequence-parallel helper kernels

CLI controls:

```bash
--flash-attention
--no-flash-attention
```

cuDNN SDPA requires a `performance` build with cuDNN and cudnn-frontend
available. If cuDNN is missing, the performance build fails rather than
silently disabling the path.

## Parallel Inference

The CLI exposes:

```bash
--devices <csv>
--cfg-parallel-size <n>
--cfg-size <n>
--tp-size <n>
--sp-size <n>
```

Current notes:

- CFG parallelism currently supports size 1 or 2.
- Tensor parallel size is reserved in the CLI surface.
- Sequence parallelism is workload dependent.
- Small FLUX workloads can be slower than single-GPU execution because
  communication and graph segmentation overhead can dominate.
- NCCL/MPI multi-worker execution requires a `performance` build and a matching
  launcher world size.

Existing sequence-parallel references:

- [Sequence-parallel benchmark report](sp_benchmark_report.md)
- [FLUX sequence-parallel profiling notes](flux_sp_profile_root_cause.md)

## Cache-Based Reuse

CLI cache modes:

```bash
--cache off|easycache|ucache|dbcache|taylorseer|cache-dit|magcache|dicache|sencache
```

Available tuning knobs include:

```bash
--cache-threshold <float>
--cache-start <float>
--cache-end <float>
--cache-error-decay <float>
--cache-relative-threshold
--cache-absolute-threshold
--cache-no-reset-error
--cache-fn-blocks <int>
--cache-bn-blocks <int>
--cache-residual-threshold <float>
--cache-max-accumulated-residual-diff <float>
--cache-warmup-steps <int>
--cache-max-cached-steps <int>
--cache-max-continuous-cached-steps <int>
--cache-taylor-order <int>
--cache-taylor-skip <int>
--cache-scm-mask <csv>
--cache-static-scm
--cache-calibrate <path>
--cache-profile <path>
```

Cache methods are experimental unless a model-specific validation note says
otherwise. DBCache, TaylorSeer, CacheDiT, EasyCache, UCache, MagCache,
DiCache, and SenCache are exposed through the public C API and CLI, but their
quality and speed tradeoffs are model and prompt dependent.

Calibration support is method-specific. The C API exposes:

```c
bool ed_cache_mode_supports_calibration(ed_cache_mode_t mode);
```

## Profiling

Graph-cut profiling:

```bash
--profile-graph-cuts
--profile-graph-cuts-top <n>
--profile-graph-cuts-all-ranks
```

Runtime environment flags also exist for targeted CUDA, cuDNN, cache, and
sequence-parallel investigation. Treat them as developer diagnostics rather
than stable user-facing API.

## Benchmark Methodology

For each benchmark, record:

- edge-dit.cpp commit
- ggml submodule commit
- `build-config.txt`
- GPU model and count
- driver, CUDA, cuDNN, NCCL, and MPI versions
- backend and build profile
- model family and exact checkpoint
- resolution, frames, steps, seed, sampler, scheduler, guidance, and cache
  settings
- warmup count and measured repetitions
- median latency and peak memory
- enabled optimizations and parallel sizes

Do not mix `minimal` profile results with official performance claims.

## Known Limitations

- v0.1.0-alpha performance sign-off is pending full CUDA performance validation.
- Sequence parallelism is workload dependent.
- Small FLUX runs can regress relative to single GPU because overhead can
  exceed compute savings.
- Cache methods are mostly experimental and should be validated per model,
  prompt class, and quality target.
- Metal and Vulkan are experimental for this project.
- No new unverified performance numbers should be added until the release gate
  is complete.

## Related Documentation

- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)
