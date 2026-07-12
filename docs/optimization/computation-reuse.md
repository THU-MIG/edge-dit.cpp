# Computation Reuse

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

This document covers cache-based reuse for repeated DiT computation across
denoising steps.

## Scope

This optimization point covers:

- timestep-level reuse;
- block-level reuse;
- output, feature, and probe cache policy granularity.

These are documented together because the cache mode, policy granularity, and
quality-speed tradeoff are one runtime surface.

## User Entry Points

```bash
--cache off|easycache|ucache|dbcache|taylorseer|cache-dit|magcache|dicache|sencache
```

Common tuning knobs include:

```bash
--cache-threshold <float>
--cache-start <float>
--cache-end <float>
--cache-fn-blocks <int>
--cache-bn-blocks <int>
--cache-max-cached-steps <int>
--cache-profile <path>
```

## Policy Granularity

- Output policies operate on whole-model input latent / output noise.
- Feature policies operate on block-stack residuals captured through model cache
  seams.
- Probe policies run a shallow prefix of blocks before deciding whether to
  reuse or compute.

## Notes

- Cache methods are experimental unless a model-specific validation note says
  otherwise.
- Cache is a speed-quality tradeoff, not a free optimization.
- Benchmark reports should include latency, skipped work, cache settings, and
  quality metrics together.

## Related Documentation

- [Performance and optimization](../performance.md)
