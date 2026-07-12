# Supported Models and Usage

[← Back to README](../README.md)

This document describes the public preview model scope, supported formats, and
model-specific limitations. For runnable commands, see
[Command line usage](cli.md). The source tree contains additional experimental
model scaffolding; only the families below are part of the current public
support commitment.

## Supported Models

| Model family | Task | Common format | Backend coverage | Status |
|---|---|---|---|---|
| SD3 / SD3.5 | Text-to-image | Diffusers directory or component weights | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| FLUX.1 | Text-to-image | Diffusers directory, top-level FLUX safetensors, or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| FLUX.1-Kontext | Image editing / reference-guided generation | Diffusers-style directory or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| Qwen-Image | Text-to-image | Diffusers directory or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| Qwen-Image-Edit | Image editing | Diffusers directory or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| Wan 2.x | Video generation | Diffusers directory or components | CUDA first, CPU functional for validation, Metal/Vulkan experimental | Public preview, still being optimized |

Backend availability means the runtime can be built for that backend. Model
quality, memory use, and speed are workload dependent and should be validated
for the exact model, resolution, and prompt set you plan to use.

## Model Formats

edge-dit.cpp can load:

- Diffusers-style directories.
- Standalone component weights for the diffusion model, VAE, text encoders,
  and model-specific vision/text components.
- `.safetensors` files.
- `.safetensors.index.json` shard indexes.
- GGUF files.

The simplest path is a model directory. Component loading is useful when
weights are stored separately. See [Command line usage](cli.md#model-loading)
for both forms.

## Text-to-Image

### FLUX.1

FLUX.1 text-to-image support uses Diffusers-style model directories,
standalone FLUX safetensors, or compatible component weights.

Command example: [FLUX.1-dev CLI](cli.md#flux1-dev).

### SD3 / SD3.5

SD3-family text-to-image support uses Diffusers-style directories or component
weights.

SD3 supports:

```bash
--no-t5
```

This reduces memory use and prompt adherence. The engine validates that
`--no-t5` is only used with SD3-family models.

Command example: [SD3 / SD3.5 CLI](cli.md#sd3-sd35).

### Qwen-Image

Qwen-Image text-to-image support uses Diffusers-style directories or component
weights.

Some Qwen checkpoints require:

```bash
--qwen-image-zero-cond-t
```

Use it only for checkpoints that need that conditioning behavior.

Command example: [Qwen-Image CLI](cli.md#qwen-image).

## Image Editing

### FLUX.1-Kontext

FLUX.1-Kontext uses an input/reference image via `--image`.

Command example: [FLUX.1-Kontext CLI](cli.md#flux1-kontext).

### Qwen-Image-Edit

Qwen-Image-Edit uses an input/reference image via `--image`. Some checkpoints
also require `--qwen-image-zero-cond-t`.

Command example: [Qwen-Image-Edit CLI](cli.md#qwen-image-edit).

Image editing support depends on the model family and checkpoint format.

## Video Generation

Wan video generation uses `--video`, `--frames`, and `--fps`.

Supported output formats are `auto`, `avi`, `mp4`, `mov`, `mkv`, and `webm`.
The CLI uses `ED_FFMPEG` when set and can also find imageio-ffmpeg binaries in
an active Python environment.

Wan 2.x remains an active optimization target. Validate memory use and output
quality for your exact resolution, frame count, and checkpoint.

Command example: [Wan video CLI](cli.md#video-generation).

## Quantization and Memory Options

The CLI supports on-load weight type selection:

```bash
--type f32|f16|bf16|q4_0|q4_1|q5_0|q5_1|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k
```

Per-tensor overrides are available with:

```bash
--tensor-type-rules "attn=q4_0,norm=f16"
```

Memory-oriented options:

```bash
--vae-tiling
--vae-tile-size <float>
--offload-to-cpu
--keep-text-encoder-on-cpu
--keep-vae-on-cpu
--max-vram <GB>
```

See [Command line usage](cli.md#quantization-and-memory) for runnable examples
and [Performance and optimization](performance.md) for cache, parallelism, and
profiling behavior.

## Model-Specific Limitations

- Public preview support is narrower than the internal enum list in the loader.
- Metal and Vulkan are experimental for DiT workloads and should be validated
  per model.
- Wan video support is available but still being optimized for memory and
  runtime behavior.
- Cache methods and sequence parallelism are workload dependent and may not be
  valid for every model or resolution.
- Some editing models require model-specific conditioning flags such as
  `--qwen-image-zero-cond-t`.
- Component loading requires a complete, compatible set of text encoders, VAE,
  diffusion transformer, and optional vision components for the selected model.

## Related Documentation

- [Build and installation](build.md)
- [Command line usage](cli.md)
- [Performance and optimization](performance.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)

[← Back to README](../README.md)
