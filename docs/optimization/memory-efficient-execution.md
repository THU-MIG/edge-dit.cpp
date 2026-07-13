# Memory-Efficient Execution

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

## 1. Overview

Running a Diffusion Transformer end to end requires memory for several distinct
things at once: transformer weights, text-encoder and VAE weights, the latent
state, and the temporary workspace of the active compute graph.

On a constrained device, any one of these can be the factor that turns a
runnable configuration into an out-of-memory failure. edge-dit.cpp therefore
separates memory control into mechanisms that act on different parts of this
budget:

1. **Weight placement** — where model weights reside (GPU or CPU), per component.
2. **Parameter offload** — keeping weights in CPU memory and staging them onto
   the GPU only while they are needed.
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

## 2. Component Placement

A pipeline is composed of separately loaded components — the diffusion
transformer, one or more text encoders, and the VAE. These do not all have to
live on the same backend.

edge-dit.cpp can place individual components on the CPU backend while the
diffusion transformer runs on the GPU:

| Control | Component moved to CPU |
|---|---|
| Text-encoder placement | CLIP / T5 text encoders |
| VAE placement | VAE encoder / decoder |

Each component is handled independently, so a configuration can, for example,
keep the text encoders on the CPU while leaving the VAE on the GPU.

Because these controls move a component from the GPU to the CPU, they only have
an effect when the runtime backend is not already the CPU. When inference is
already running on the CPU backend, the placement controls are no-ops.

Moving a component to the CPU frees the device memory its weights would
otherwise occupy. This is most useful for large text encoders, which can hold
several gigabytes of weights that are only needed during a short conditioning
phase.

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

Component placement decides where weights are stored for the whole run.
Parameter offload goes further: it keeps weights resident in CPU memory and
brings them onto the GPU only for the duration of computation.

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

A value of zero (the default) disables graph VRAM control. See
[Graph and operator optimization](graph-and-operator-optimization.md) for the
underlying segmentation mechanism and plan caching.

---

## 5. VAE Tiling

VAE decoding of a large image requires a large contiguous workspace, and this
final decode is often the single largest memory spike in an image pipeline.

VAE tiling decodes the latent in overlapping tiles and blends them back
together, so that only one tile's workspace is allocated at a time.

Tiling is controlled by a relative tile size and a target overlap:

- The relative tile size expresses how many tiles span each dimension. A value
  of `2.0` corresponds to roughly a 2×2 tile grid, while a larger value such as
  `4.0` produces a finer grid of smaller tiles and a correspondingly lower peak.
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
# Constrained GPU: quantize weights, drop T5, tile the VAE, keep encoders on CPU
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 \
  --type q4_k --no-t5 --vae-tiling --keep-text-encoder-on-cpu \
  -p "a glass teapot on a wooden table" -W 1024 -H 1024 --steps 20 -o output.png

# Very constrained GPU: full offload with a graph VRAM budget
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 \
  --type q4_k --no-t5 --vae-tiling --offload-to-cpu --max-vram 4 \
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
| `--offload-to-cpu` | Keep weights in CPU memory and stage them onto the GPU for compute |
| `--keep-text-encoder-on-cpu` | Place the text encoders on the CPU backend |
| `--keep-vae-on-cpu` | Place the VAE on the CPU backend |
| `--max-vram <GB>` | Compute-graph VRAM budget; drives graph-cut segmentation (used with offload) |
| `--vae-tiling` | Enable VAE tiling with the default relative tile size |
| `--vae-tile-size <float>` | Enable VAE tiling and set the relative tile size (larger value = finer grid) |
| `--no-t5` | Skip loading the T5 text encoder (SD3-family only) |

### C API

These map to fields on `ed_context_params_t`:

- `bool offload_params_to_cpu` — parameter offload.
- `bool keep_text_encoder_on_cpu`, `bool keep_vae_on_cpu` — per-component CPU
  placement.
- `float max_vram_gb` — graph VRAM budget in gigabytes; `0` disables it.
- `bool skip_t5` — skip the T5 text encoder.
- `ed_tiling_params_t vae_tiling` — VAE tiling configuration, with
  `enabled`, `rel_size_x` / `rel_size_y` (relative tile size, default `2.0`),
  `target_overlap` (default `0.25`), and `tile_size_x` / `tile_size_y`
  (absolute, used only when a relative size is not set).

### Python bindings

`EngineConfig` exposes the same controls: `offload_params_to_cpu`,
`keep_text_encoder_on_cpu`, `keep_vae_on_cpu`, `max_vram_gb`, `skip_t5`,
`vae_tiling`, and `vae_tile_size`. Setting `vae_tile_size` enables tiling and
sets the relative tile size in both dimensions.

---

## Related Documentation

- [Performance and optimization](../performance.md)
- [Model representation and precision](model-representation-and-precision.md)
- [Graph and operator optimization](graph-and-operator-optimization.md)
- [Command line usage](../cli.md)
- [Supported models and usage](../models.md)
- [API and bindings](../api.md)
