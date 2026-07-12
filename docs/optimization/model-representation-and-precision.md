# Model Representation and Precision

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

This document is the technical landing page for how edge-dit.cpp represents
model weights and controls precision.

## Scope

This optimization point covers:

- quantization;
- mixed-precision loading and execution;
- per-tensor dtype rules for precision-sensitive tensors.

These are tightly coupled in the runtime, so they are documented together
rather than split into separate pages.

## User Entry Points

```bash
--type <dtype>
--tensor-type-rules <rules>
```

The corresponding C API fields are:

```c
ed_context_params_t::weight_type
ed_context_params_t::tensor_type_rules
```

## Notes

- Quantization and mixed precision reduce memory pressure but may change
  numerical results.
- Per-tensor dtype rules are the mechanism for keeping selected tensors, such
  as normalization weights, biases, embeddings, or output layers, in a safer
  precision.
- Dtype choices should be validated per model family, checkpoint, backend, and
  prompt set.
- Benchmark metadata should record the global dtype and any tensor rule string.

## Related Documentation

- [Performance and optimization](../performance.md)
- [Supported models and usage](../models.md#quantization-and-memory-options)
