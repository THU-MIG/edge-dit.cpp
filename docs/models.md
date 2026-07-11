# Supported Models and Usage

[← Back to README](../README.md)

This document describes the public preview model scope, supported formats, and
common CLI usage. The source tree contains additional experimental model
scaffolding; only the families below are part of the current public support
commitment.

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
- Standalone component weights:
  - `--diffusion-model`
  - `--vae`
  - `--clip_l`
  - `--clip_g`
  - `--t5xxl`
  - `--llm`
  - `--llm-vision`
  - `--clip-vision`
- `.safetensors` files.
- `.safetensors.index.json` shard indexes.
- GGUF files.

The simplest path is a model directory:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/model-dir \
  --prompt "a glass teapot on a wooden table" \
  --output output.png
```

Component loading is useful when weights are stored separately:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --diffusion-model /path/to/transformer.safetensors \
  --vae /path/to/vae.safetensors \
  --clip_l /path/to/clip_l.safetensors \
  --t5xxl /path/to/t5xxl.safetensors \
  --prompt "a glass teapot on a wooden table" \
  --output output.png
```

## Text-to-Image

### FLUX.1

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --guidance 3.5 \
  --seed 0 \
  --output flux.png
```

### SD3 / SD3.5

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/stable-diffusion-3-medium-diffusers \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --cfg-scale 5.0 \
  --flow-shift 3.0 \
  --seed 0 \
  --output sd3.png
```

SD3 supports:

```bash
--no-t5
```

This reduces memory use and prompt adherence. The engine validates that
`--no-t5` is only used with SD3-family models.

### Qwen-Image

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/Qwen-Image \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --seed 0 \
  --output qwen.png
```

Some Qwen checkpoints require:

```bash
--qwen-image-zero-cond-t
```

Use it only for checkpoints that need that conditioning behavior.

## Image Editing

### FLUX.1-Kontext

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-kontext \
  --image /path/to/input.png \
  --prompt "make the object look like brushed metal" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --output edited.png
```

### Qwen-Image-Edit

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/Qwen-Image-Edit \
  --image /path/to/input.png \
  --prompt "change the background to a clean studio" \
  --qwen-image-zero-cond-t \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --output qwen-edit.png
```

Image editing support depends on the model family and checkpoint format.

## Video Generation

Wan video generation uses:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --video \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "a glass teapot rotating on a wooden table" \
  --width 832 \
  --height 480 \
  --frames 40 \
  --fps 16 \
  --steps 20 \
  --cfg-scale 5.0 \
  --flow-shift 5.0 \
  --output wan.avi
```

Supported output formats are `auto`, `avi`, `mp4`, `mov`, `mkv`, and `webm`.
The CLI uses `ED_FFMPEG` when set and can also find imageio-ffmpeg binaries in
an active Python environment.

Wan 2.x remains an active optimization target. Validate memory use and output
quality for your exact resolution, frame count, and checkpoint.

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

See [Performance and optimization](performance.md) for cache, parallelism, and
profiling options.

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
- [Performance and optimization](performance.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)

[← Back to README](../README.md)
