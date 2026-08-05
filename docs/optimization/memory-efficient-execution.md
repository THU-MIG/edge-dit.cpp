# Memory-Efficient Execution

[← Back to performance (RTX 4090)](../performance-4090.md) | [H200 snapshot](../performance-H200.md) | [← Back to README](../../README.md)

## 1. Overview

Running a Diffusion Transformer end to end requires memory for several distinct
things at once: transformer weights, text-encoder and VAE weights, the latent
state, and the temporary workspace of the active compute graph.

On a constrained device, any one of these can be the factor that turns a
runnable configuration into an out-of-memory failure. edge-dit.cpp therefore
separates memory control into mechanisms that act on different parts of this
budget:

1. **Component-selective offload** — offloading an individual component's
   weights (diffusion transformer, text encoders, or VAE) to CPU memory and
   staging that component onto the GPU only while it computes.
2. **Parameter offload** — the shared staging mechanism, plus the full-model
   offload that keeps every weight in CPU memory and moves it onto the GPU only
   while it is needed.
3. **Graph VRAM control** — bounding the peak workspace of the compute graph.
4. **VAE tiling** — decoding large images in overlapping tiles instead of one
   allocation.

These controls are largely orthogonal and can be combined. They trade
additional data movement or scheduling overhead for a lower peak memory
requirement, so the useful combination depends on the specific model,
resolution, and device.

Weight precision, which reduces the resident footprint by changing the numeric
representation of weights, is documented separately in
[Model representation and precision](model-representation-and-precision.md).
The graph-cut machinery that underlies graph VRAM control is described in more
detail in [Graph and operator optimization](graph-and-operator-optimization.md).

---

## 2. Component-Selective Offload

A pipeline is composed of separately loaded components — the diffusion
transformer, one or more text encoders, and the VAE. Their weights do not all
have to stay resident on the GPU at once.

edge-dit.cpp can offload an individual component's weights to CPU memory while
the pipeline still runs on the GPU. Each component has its own flag:

| Control | Component whose weights are offloaded |
|---|---|
| `--dit-offload` | Diffusion transformer |
| `--text-encoder-offload` | CLIP / T5 text encoders |
| `--vae-offload` | VAE encoder / decoder |

For every one of these, the semantics are identical: the component's weights
are kept in CPU memory and staged onto the GPU only for the duration of that
component's compute (a DiT step, a text encode, or a VAE decode), then released
again. **Compute always happens on the GPU** — offload changes only where the
weights rest between uses, not where the math runs. The shared staging
mechanism is described in Section 3.

Each component is handled independently, so a configuration can, for example,
offload the text encoders while leaving the VAE resident on the GPU.
`--offload-to-cpu` is the union of all three: it offloads the diffusion
transformer, text encoders, and VAE together (full-model offload).

Because these controls stage weights between CPU and a GPU backend, they only
have an effect when the runtime backend is a GPU. When inference is already
running on the CPU backend, weights already live where compute happens and the
offload controls are no-ops.

Offloading a component frees the device memory its weights would otherwise
occupy between uses. This is most useful for large text encoders, which can
hold several gigabytes of weights that are only needed during a short
conditioning phase.

### Skipping the T5 text encoder

For SD3-family models specifically, the T5 text encoder is large and optional
relative to the CLIP encoders. edge-dit.cpp can skip loading it entirely,
avoiding both its device memory and its load time.

Skipping the T5 encoder removes its weights from the model rather than moving
them, so the saving is unconditional. It is only valid for SD3-family models;
requesting it for any other model family is rejected with an error rather than
silently ignored. Skipping the encoder reduces prompt adherence, because part
of the text conditioning signal is no longer available.

---

## 3. Parameter Offload

Section 2 selects *which* components offload. This section describes the shared
staging mechanism they all use, and `--offload-to-cpu`, which applies it to the
whole model at once: weights stay resident in CPU memory and are brought onto
the GPU only for the duration of computation.

The lifecycle around a graph execution is:

```text
weights resident in CPU memory
        ↓ before compute
copy required parameters to the GPU
        ↓
execute graph
        ↓ after compute
release GPU parameter memory
```

In the default offload path, the parameters are staged onto the GPU as a group
before the graph runs and released afterwards, so their device residency is
bounded by the execution rather than by the whole session. When combined with
graph VRAM control (Section 4), offload can instead stage only the parameters
required by each graph segment, releasing them again as execution moves to the
next segment.

Offload reduces the steady-state device footprint of the weights at the cost of
repeated CPU-to-GPU transfers. It is therefore a memory-for-bandwidth trade:
most valuable when weights would not otherwise fit, and counterproductive when
device memory is already sufficient.

### Staged text-encoder offload

`--text-encoder-offload` applies this stage-per-use pattern to the text
encoders specifically: their weights are kept in CPU memory and staged onto the
GPU only for the encode pass, then released, while the encode itself still runs
on the GPU. Staging keeps the faster GPU compute while freeing the encoder's
device residency between conditioning phases. `--dit-offload` and
`--vae-offload` are the equivalent single-component controls for the diffusion
transformer and the VAE.

The text encoder is staged **segment by segment**, not as one block: the encode
graph is cut into small pieces (targeting roughly 1 GiB of weights each) and only
one segment's weights occupy the GPU at a time, so the peak device residency is
about one segment rather than the whole encoder. This matters for large text
encoders (FLUX's T5-XXL ~9.4 GiB, Qwen-Image's Qwen2.5-VL ~14 GiB): without
segmenting, staging the entire encoder at once would exceed the free VRAM on
mid/small cards and OOM. Segmentation happens automatically whenever the text
encoder is offloaded — it does **not** require setting `--max-vram`. The
per-segment budget is also capped by the actual free VRAM at encode time (so it
tightens further when the DiT is resident) and by `--max-vram` when set.

Note that `--text-encoder-offload` moves only the text encoder: the diffusion
transformer stays resident on the GPU. On a large model whose DiT alone
approaches the device size (FLUX ~22.7 GiB, Qwen-Image ~38 GiB, Wan-14B), the
run will still run out of memory on the resident DiT and its compute buffer even
though the encoder now stages cleanly. For those, offload the transformer as
well — either `--offload-to-cpu` (full-model offload) or add `--dit-offload`.

### Automatic placement (`--auto-allocate`)

Rather than deciding offload per component by hand, `--auto-allocate` fits
components against a hard VRAM budget of `min(--max-vram, live free VRAM)`.
Components are considered in priority order — diffusion transformer, then text
encoders, then VAE — and each is placed resident on the GPU if its
(post-quantization) weights plus a compute-headroom reserve still fit the
remaining budget; otherwise it is offloaded. After placement, the leftover
budget becomes the graph VRAM budget (Section 4) for whatever was offloaded, so
the largest, most-reused weights get first claim on resident VRAM and the rest
is graph-segmented to stay within the cap.

### Fully automatic budgeting (`--auto-fit`)

`--auto-fit` is a superset of `--auto-allocate`: enabling it implicitly enables
placement, so one flag covers the whole budget. On top of the resident/offload
decision, `--auto-fit` also **chooses the quantization** to make components fit,
ignoring `--type`:

- the text encoders are forced to `q8_0` unconditionally — TE compute time is
  negligible next to the DiT, so `q8_0` is near-lossless yet frees the budget
  that a bf16 TE (FLUX's T5 alone is ~9 GB) would otherwise consume;
- the diffusion transformer walks a `q8_0 → q4_K` ladder, picking the highest
  precision that still fits resident under the budget; if even `q4_K` does not
  fit, it stays at `q8_0` and is offloaded (an offloaded `q8_0` beats a `q4_K`
  that would have to offload anyway).

Use `--auto-fit` when you just want a model to fit a hard VRAM cap without
hand-tuning `--type` and placement; use `--auto-allocate` when you want to keep
control of the quantization (`--type`) and only automate placement.

### Unified memory (Apple Silicon)

On Apple Silicon the CPU and GPU share the same physical memory, so staging
weights between a CPU backend and the Metal backend copies data within one
memory pool — it does not reduce residency, it only adds a second copy and
raises peak memory. edge-dit.cpp detects a unified-memory Metal device at
runtime and, when `--offload-to-cpu` is requested there, ignores it (keeping
weights resident on the runtime backend) and logs a warning. Measured on an
M4 Pro running SD3, this avoids the redundant staging copies and cuts peak
memory by roughly 3–4 GB versus the old behavior, with bit-identical output.
Set `ED_NO_UMA_SHORTCIRCUIT=1` to force the legacy offload path. Discrete GPUs
(CUDA, or Metal on a discrete AMD GPU) are unaffected and offload as before.


---

## 4. Graph VRAM Control

The compute graph needs a temporary workspace in addition to the weights and
latent state. For high resolutions, long video sequences, or large conditioning
inputs, this workspace can dominate peak VRAM.

edge-dit.cpp accepts a graph VRAM budget in gigabytes. The budget drives
graph-cut segmented execution: a large graph is split into segments run
sequentially, so only one segment's workspace is live at a time. A larger
budget merges more adjacent segments (fewer launches, higher peak), while a
smaller budget keeps them split (lower peak, more launches). The resolved plan
is cached across denoising steps.

Two caveats matter in practice:

- **The budget only takes effect together with offload.** Segmentation engages
  when parameters are offloaded and the runtime backend is a GPU; a pure-GPU run
  without offload is not segmented by setting a budget alone.
- **The budget must be large enough for the largest single operation.** It is
  not validated against a lower bound, and the runtime cannot merge below the
  finest segmentation.

A value of zero (the default) disables explicit graph VRAM control. One case
overrides this: when any offload flag is set on a GPU backend but no
`--max-vram` is given, the runtime auto-derives a budget of `0.85 × free VRAM`
so that graph-cut segmentation still engages — without it, offload would copy
every weight back onto the GPU at once and defeat the purpose (a large DiT such
as FLUX would OOM on a 24 GB card). Passing `--max-vram` explicitly overrides
this default. See
[Graph and operator optimization](graph-and-operator-optimization.md) for the
underlying segmentation mechanism and plan caching.

---

## 5. VAE Tiling

VAE decoding of a large image requires a large contiguous workspace, and this
final decode is often the single largest memory spike in an image pipeline.

VAE tiling decodes the latent in overlapping tiles and blends them back
together, so that only one tile's workspace is allocated at a time.

Tiling is controlled by a relative tile size and a target overlap:

- The relative tile size expresses how many tiles span each dimension. The
  default is `5.0` (roughly a 5×5 grid, about a 32×32 latent tile, which
  minimizes the VAE peak empirically); a smaller value such as `2.0` gives a
  coarser 2×2 grid with a higher peak.
- The target overlap is the fractional overlap between neighboring tiles,
  clamped to a sensible range. Overlap exists so that tile boundaries can be
  blended smoothly.

When a relative tile size is provided, it takes precedence over any absolute
pixel tile size. Overlapping regions are combined with a smooth blend weight so
that tile seams do not appear in the output.

```text
latent
   ↓ split into overlapping tiles
per-tile VAE decode (one tile workspace live at a time)
   ↓ blend overlaps
full-resolution image
```

Because each tile runs its own VAE forward pass, tiling lowers the peak decode
memory but increases decode time roughly with the number of tiles. It is
therefore worth enabling for large resolutions on constrained devices and
unnecessary when the untiled decode already fits.

---

## 6. Combining Controls

The controls above target different parts of the memory budget, so they
compose. A typical constrained-device configuration layers several of them:

```bash
# Constrained GPU: quantize weights, drop T5, tile the VAE, offload the text encoders
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 \
  --type q4_k --no-t5 --vae-tiling on --text-encoder-offload \
  -p "a glass teapot on a wooden table" -W 1024 -H 1024 --steps 20 -o output.png

# Very constrained GPU: full offload with a graph VRAM budget
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 \
  --type q4_k --no-t5 --vae-tiling on --offload-to-cpu --max-vram 4 \
  -p "a glass teapot on a wooden table" -W 512 -H 512 --steps 20 -o output.png
```

Each added control removes a different memory bottleneck but adds its own
overhead — transfer time for offload, extra forward passes for tiling, extra
launches for graph segmentation. Validate output quality and latency for the
exact model and resolution being targeted.

---

## 7. Public Interfaces

### Command line

| Flag | Effect |
|---|---|
| `--offload-to-cpu` | Full-model offload: keep all weights (DiT, text encoders, VAE) in CPU memory and stage each component onto the GPU for its compute |
| `--dit-offload` | Keep diffusion-transformer weights in CPU memory and stage them onto the GPU per step (compute runs on the GPU) |
| `--text-encoder-offload` | Keep text-encoder weights in CPU memory and stage them onto the GPU per encode (compute runs on the GPU) |
| `--vae-offload` | Keep VAE weights in CPU memory and stage them onto the GPU per decode (compute runs on the GPU) |
| `--auto-allocate` | Budget-capped automatic placement: fit components (DiT, then text encoders, then VAE) resident on the GPU under a hard `min(--max-vram, free VRAM)` cap, offloading and graph-segmenting whatever does not fit |
| `--auto-fit` | Superset of `--auto-allocate`: also chooses quantization to fit the budget (text encoders forced to `q8_0`, DiT walks a `q8_0 → q4_K` ladder), ignoring `--type` |
| `--max-vram <GB>` | Compute-graph VRAM budget; drives graph-cut segmentation. Used with offload or `--auto-allocate`; when offload is set without it, the runtime defaults to `0.85 × free VRAM` |
| `--vae-tiling <on\|off\|auto>` | VAE tiling: `on` forces it, `off` forces it off (suppressing the low-VRAM auto-enable), `auto` (the default) enables it only on lower-VRAM GPUs (<=25 GiB total) |
| `--vae-tile-size <float>` | Enable VAE tiling and set the relative tile size (larger value = finer grid) |
| `--no-t5` | Skip loading the T5 text encoder (SD3-family only) |

### C API

These map to fields on `ed_context_params_t`:

- `bool offload_params_to_cpu` — full-model parameter offload (DiT, text
  encoders, and VAE together).
- `bool dit_offload`, `bool text_encoder_offload`, `bool vae_offload` —
  offload a single component's weights to CPU and stage it onto the GPU per
  compute. Each is the union with `offload_params_to_cpu` for that component.
- `bool auto_allocate` — budget-capped automatic per-component placement.
- `bool auto_fit` — superset of `auto_allocate` that also picks per-component quantization (TE `q8_0`, DiT `q8_0 → q4_K` ladder) to fit the budget.
- `float max_vram_gb` — graph VRAM budget in gigabytes; `0` disables it (or, with offload on a GPU, defers to the `0.85 × free VRAM` default).
- `bool skip_t5` — skip the T5 text encoder.
- `ed_tiling_params_t vae_tiling` — VAE tiling configuration, with
  `enabled`, `force_disable` (explicit off; suppresses the low-VRAM auto-enable),
  `rel_size_x` / `rel_size_y` (relative tile size, default `5.0`),
  `target_overlap` (default `0.25`), and `tile_size_x` / `tile_size_y`
  (absolute, used only when a relative size is not set).

### Python bindings

`EngineConfig` exposes the offload and memory controls, including
`offload_params_to_cpu`, `max_vram_gb`, `skip_t5`, `vae_tiling`, and
`vae_tile_size`. Setting `vae_tile_size` enables tiling and sets the relative
tile size in both dimensions.

---

## Related Documentation

- [Performance and benchmarks (RTX 4090)](../performance-4090.md)
- [H200 snapshot](../performance-H200.md)
- [Model representation and precision](model-representation-and-precision.md)
- [Graph and operator optimization](graph-and-operator-optimization.md)
- [Command line usage](../cli.md)
- [Supported models and usage](../models.md)
- [API and bindings](../api.md)
