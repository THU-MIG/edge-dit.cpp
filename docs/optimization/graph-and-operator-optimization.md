# Graph and Operator Optimization

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

This document covers optimized graph lowering and backend operator paths used by
edge-dit.cpp.

## Scope

This optimization point covers:

- cuDNN SDPA;
- DiT-specific CUDA operators;
- tensor-layout optimization around DiT graph regions.

These features are grouped together because operator speedups and graph layout
changes often interact.

## Build Entry Points

Important CUDA build options include:

```text
ED_ENABLE_CUDNN_SDPA
ED_ENABLE_CUDA_NORM
ED_ENABLE_CUDA_ROPE
ED_ENABLE_CUDA_MODULATION
```

The official CUDA performance profile enables the optimized CUDA paths expected
for performance work.

## Runtime Entry Points

```bash
--flash-attention
--no-flash-attention
```

## Current Operator Areas

- cuDNN SDPA
- CUDA Norm
- CUDA RoPE
- CUDA Modulation
- Flux sequence-parallel helper kernels

## Notes

- cuDNN SDPA requires cuDNN and cudnn-frontend in the performance build.
- Fused attention or CUDA helper kernels may change floating-point accumulation
  order, so compare numerical and visual quality rather than assuming identical
  hashes.
- Tensor-layout work focuses on reducing avoidable copies, contiguous
  materialization, reshapes, views, permutes, and concatenations.

## Related Documentation

- [Performance and optimization](../performance.md)
- [Build and installation](../build.md#cuda-build-profiles)
- [Sequence-parallel benchmark report](../sp_benchmark_report.md)
- [FLUX sequence-parallel profiling notes](../flux_sp_profile_root_cause.md)
