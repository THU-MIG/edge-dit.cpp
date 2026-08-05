# Model Representation and Precision

[← Back to README](../../README.md)

## 1. Overview

edge-dit.cpp loads Diffusion Transformer weights through ggml tensor storage
and can represent each weight in a range of precisions, from full 32-bit
floating point down to 4-bit block quantization.

Weight representation is the primary lever for the resident memory footprint of
a model. On resource-constrained devices, the transformer, text encoders, and
VAE weights often dominate device memory before any activation or graph
workspace is allocated.

edge-dit.cpp addresses this at load time through:

1. **Global weight-type selection**

   A single requested data type applied to eligible weights while the model is
   loaded.

2. **Per-tensor precision control**

   Regular-expression rules that override the global type for individual
   tensors, so that numerically sensitive layers can stay at higher precision.

Quantization is applied when the model is loaded rather than as an offline
conversion step. Tensors whose representation cannot be changed safely, or that
are already stored at the requested type, are loaded without conversion.

Approximate computation reuse is documented separately in
[Computation reuse](computation-reuse.md), and memory controls that operate on
placement and graph execution rather than weight precision are documented in
[Memory-efficient execution](memory-efficient-execution.md).

---

## 2. Weight Representation and Data Types

Weights are stored as ggml tensors. Each tensor has an individual data type, so
a model does not need to use a single uniform precision.

The public data types are:

| Category | Types | Notes |
|---|---|---|
| Floating point | `f32`, `f16`, `bf16` | Full and half precision |
| Legacy block quantization | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0` | Fixed block layouts |
| K-quant block quantization | `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k` | Higher-quality low-bit formats |

Block-quantized types group consecutive weight values into fixed-size blocks
with shared scaling metadata. A quantized type can therefore only be applied to
a tensor whose leading dimension is a multiple of the block size; other tensors
are kept at their stored type.

The default weight type is `auto`. In this mode edge-dit.cpp does not change the
stored representation: each weight keeps the data type present in the source
checkpoint. Selecting an explicit type opts into conversion during loading.

---

## 3. On-load Quantization

When a global weight type or any per-tensor rule is requested, edge-dit.cpp
resolves a target type for every eligible tensor before reading model data. If
the target type differs from the stored type, tensor data is converted while it
is loaded.

Key properties:

- **Load-time execution by default.** Conversion happens during context
  creation. Running the same configuration again repeats the conversion at load
  time. For large models this can take tens of seconds to minutes (e.g. FLUX.1-dev
  q4_k is ~80s per load), which the offline export below removes.
- **Offline export with `ed-convert`.** `ed-convert` runs the quantization once
  and writes a self-contained, portable GGUF; subsequent runs load the
  pre-quantized weights in seconds and skip on-load conversion. The result is
  bit-identical to the load-time path and works across all model families. For a
  full model (diffusers directory or complete checkpoint) the model version is
  recorded in GGUF metadata so the correct pipeline is selected by any file name;
  a bare transformer-only distill carries no family config, so its GGUF loads as
  a `--diffusion-model` on top of the base directory instead. See
  [CLI usage](../cli.md#pre-quantized-gguf-with-ed-convert).
- **Activation-calibrated quantization (`--imatrix`).** `ed-convert` optionally
  takes an importance vector (per-input-channel `E[x^2]`, produced by
  `tools/imatrix/calibrate.py`) via `--imatrix`. When supplied, the quantizer
  weights each input channel by how much it drives the output instead of the
  uniform default, improving low-bit (e.g. q4_k) quality at no runtime cost.
  See [CLI usage](../cli.md#activation-calibrated-imatrix-quantization).
- **Purpose.** Quantization reduces the resident weight footprint approximately
  in proportion to the bits used per weight. It is primarily a memory-saving
  mechanism; on high-bandwidth GPUs it does not by itself speed up inference,
  and its benefit is largest for models stored at full precision.

The optimization preserves the model architecture. Only the numerical
representation of eligible weights changes. A few tensors are additionally
protected by a precision floor — quantized, but never below a per-tensor
minimum — see [Section 5](#5-precision-preserved-tensors).

---

## 4. Per-tensor Precision Control

A single global type is often too coarse. Aggressively quantizing every tensor
can degrade output quality, while keeping every tensor at high precision wastes
memory on layers that tolerate low-bit representation well.

edge-dit.cpp accepts a set of per-tensor rules that override the global type for
matching tensors.

Each rule has the form:

```text
<name-regex>=<ggml-type-name>
```

Multiple rules are separated by commas:

```text
attn=q4_0,norm=f16,bias=f32
```

Rule evaluation works as follows:

- The left-hand side is a regular expression matched against the **tensor name**.
- The right-hand side is a ggml type name (for example `f16`, `q4_0`, `q8_0`).
- For each tensor, the resolved type starts from the global weight type; rules
  are then evaluated in order and the **first matching rule wins**.
- A resolved per-tensor rule therefore takes precedence over the global type.

Malformed entries are skipped defensively rather than aborting the load:

- An entry without a `=` separator, or with an unknown type name, is ignored
  with a warning.
- An entry whose left-hand side is not a valid regular expression is ignored
  with a warning.

This lets a broad global type coexist with targeted overrides, such as keeping
normalization layers at `f16` while quantizing attention and feed-forward
weights.

---

## 5. Precision-preserved Tensors

Even when a quantized global type is requested, some tensors are deliberately
**kept at their stored precision**, including:

- tensors whose shape is not compatible with the requested block-quantized
  layout;
- small, numerically sensitive parameters such as biases and scales;
- embedding tables and the entry and exit projections of the DiT.

These exclusions apply regardless of the requested global type, and a
per-tensor rule that targets an ineligible tensor still falls back to a
representable type rather than producing an invalid tensor.

The intent is to concentrate low-bit representation on the large, quantization-
tolerant weight matrices while protecting the comparatively small layers where
low precision has a disproportionate effect on output quality.

### Quantization floors

A related mechanism raises the *floor* of a tensor's precision rather than
preserving it outright: when the requested global type is more aggressive than
the floor (compared by bits-per-weight), the tensor is quantized to the floor
type instead. This still quantizes the tensor, just not below the level where
its error becomes visible in the output.

The Qwen-Image modulation projections `img_mod.1` / `txt_mod.1` are floored at
`q8_0`. So under `--type q4_k` these projections load as `q8_0` while the rest
of the DiT is `q4_k` — the on-load weight-type report will show a small `q8_0`
count (120 tensors: 60 blocks × img/txt) alongside the `q4_k` bulk. When the
`zero_cond_t` edit path is active these projections feed a per-token select
`mod_0 + index*(mod_1 - mod_0)`, and the subtraction amplifies k-quant error
enough to break edit instruction-following at `q4_k`; the `q8_0` floor restores
it. The projections are tiny (`dim × 6·dim`), so the extra footprint is
negligible. Floors only ever *raise* precision: `f16`/`bf16`/`q8_0` global runs
and explicit `--tensor-type-rules` targeting these tensors are left untouched.

---

## 6. Public Interfaces

### Command line

```bash
# Global weight type
./build-cuda/bin/ed-cli --backend cuda --model /path/to/model \
  --type q4_k \
  -p "a glass teapot on a wooden table" -W 1024 -H 1024 --steps 20 -o output.png

# Global type with per-tensor overrides
./build-cuda/bin/ed-cli --backend cuda --model /path/to/model \
  --type q4_k --tensor-type-rules "norm=f16,bias=f32" \
  -p "a glass teapot on a wooden table" -W 1024 -H 1024 --steps 20 -o output.png
```

| Flag | Effect |
|---|---|
| `--type <dtype>` (alias `--weight-type`) | Global weight type; accepts `auto`, `f32`, `f16`, `bf16`, and the `q*` types |
| `--tensor-type-rules <csv>` | Comma-separated `<name-regex>=<ggml-type>` overrides |

### C API

The type policy is configured on `ed_context_params_t`:

- `ed_dtype_t weight_type` — global type, defaulting to `ED_DTYPE_AUTO`.
- `const char * tensor_type_rules` — per-tensor override string, or `NULL`.

`ed_dtype_t` enumerates the floating-point and block-quantized types listed in
[Section 2](#2-weight-representation-and-data-types). `ED_DTYPE_AUTO` preserves
the stored representation.

### Python bindings

`EngineConfig` exposes the same policy:

- `weight_type` accepts either the integer enum value or a string alias. String
  aliases are case-insensitive and the separators `-`, `_`, `.`, and space are
  normalized, so `q4-k`, `q4_k`, and `q4.k` resolve to the same type.
- `tensor_type_rules` accepts the same comma-separated rule string as the CLI.

---

## Related Documentation

- [performance (RTX 4090)](../performance-4090.md)
- [performance (H200)](../performance-H200.md)
- [Memory-efficient execution](memory-efficient-execution.md)
- [Graph and operator optimization](graph-and-operator-optimization.md)
- [Command line usage](../cli.md)
- [Supported models and usage](../models.md)
- [API and bindings](../api.md)
