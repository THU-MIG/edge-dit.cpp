# Parallel Execution

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

This document covers multi-worker and multi-device execution paths.

## Scope

This optimization point covers:

- CFG parallelism;
- sequence parallelism;
- NCCL/MPI multi-worker execution.

These are documented together because parallel execution depends on a matching
build profile, device layout, launcher world size, and model workload.

## Build Entry Points

```bash
CUDA_HOME=/path/to/cuda \
NCCL_ROOT=/path/to/nccl \
CUDNN_ROOT=/path/to/cudnn \
MPI_HOME=/path/to/mpi \
ED_BUILD_PROFILE=performance \
bash scripts/build_cuda.sh
```

Relevant CMake options include:

```text
ED_ENABLE_PARALLEL
ED_ENABLE_NCCL
ED_ENABLE_MPI
```

## Runtime Entry Points

```bash
--devices <csv>
--cfg-parallel-size <n>
--cfg-size <n>
--sp-size <n>
```

The C API fields include:

```c
ed_context_params_t::cfg_parallel_size
ed_context_params_t::sp_parallel_size
```

## Notes

- CFG parallelism currently supports size 1 or 2 in the public documentation.
- Sequence parallelism is workload dependent.
- Small FLUX workloads can be slower than single-GPU execution because
  communication and graph segmentation overhead can dominate.
- NCCL/MPI multi-worker execution requires a performance build and matching
  launcher configuration.

## Related Documentation

- [Performance and optimization](../performance.md)
- [Build and installation](../build.md#cuda-build-profiles)
- [Sequence-parallel benchmark report](../sp_benchmark_report.md)
- [FLUX sequence-parallel profiling notes](../flux_sp_profile_root_cause.md)
