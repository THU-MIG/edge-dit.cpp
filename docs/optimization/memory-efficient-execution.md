# Memory-Efficient Execution

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

This document covers runtime controls that reduce peak device memory or make a
model fit under a constrained memory budget.

## Scope

This optimization point covers:

- CPU offload;
- graph VRAM control;
- VAE tiling;
- component placement for large text encoders and VAE modules.

These features interact with each other, so they are documented together as one
memory execution strategy.

## User Entry Points

```bash
--offload-to-cpu
--keep-text-encoder-on-cpu
--keep-vae-on-cpu
--vae-tiling
--vae-tile-size <float>
--max-vram <GB>
```

The corresponding C API fields include:

```c
ed_context_params_t::offload_params_to_cpu
ed_context_params_t::keep_text_encoder_on_cpu
ed_context_params_t::keep_vae_on_cpu
ed_context_params_t::vae_tiling
ed_context_params_t::max_vram_gb
```

## Notes

- CPU offload and component placement can lower persistent VRAM usage but may
  add host-device transfer overhead.
- VAE tiling can reduce decode memory at the cost of additional tiled work.
- `--max-vram` limits graph allocation pressure and should be validated for the
  selected model, resolution, frame count, and backend.
- Benchmark reports should record every memory-control flag used.

## Related Documentation

- [Performance and optimization](../performance.md)
- [Supported models and usage](../models.md#quantization-and-memory-options)
