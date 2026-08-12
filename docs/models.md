# Supported Models and Usage

[← Back to README](../README.md)

This document describes the supported model scope, supported formats, and
model-specific limitations. For runnable commands, see
[Command line usage](cli.md). The source tree contains additional experimental
model scaffolding; only the families below are part of the current public
support commitment.

## Supported Models

| Model family | Task | HuggingFace base repo | Common format | Backend coverage | Status |
|---|---|---|---|---|---|
| SD3 / SD3.5 | Text-to-image | [`stabilityai/stable-diffusion-3-medium-diffusers`](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) (and SD3.5 siblings) | Diffusers-style directory or component weights | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| FLUX.1 | Text-to-image | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Diffusers-style directory, top-level FLUX safetensors, or components | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| FLUX.1-Kontext | Image editing / reference-guided generation | [`black-forest-labs/FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | Diffusers-style directory or components | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| FLUX.2 [klein] 4B | Text-to-image / image editing | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Diffusers-style directory or components | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| Qwen-Image | Text-to-image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) | Diffusers-style directory or components | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| Qwen-Image-Edit | Image editing | [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit) | Diffusers-style directory or components | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| Wan 2.1 | Video generation | [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (and Wan2.1-T2V-14B-Diffusers) | Diffusers-style directory or components | CUDA first, CPU functional for validation, Metal/Vulkan experimental | Supported (Vulkan still optimizing) |

Backend availability means the runtime can be built for that backend. Model
quality, memory use, and speed are workload dependent and should be validated
for the exact model, resolution, and prompt set you plan to use. The repo column
lists the base checkpoint each family is validated against; pick the exact
revision/variant (e.g. an SD3.5 sibling) that matches your use case.

## Model Formats

edge-dit.cpp can load:

- Diffusers-style directories.
- Standalone component weights for the diffusion model, VAE, text encoders,
  and model-specific vision/text components.
- `.safetensors` files.
- `.safetensors.index.json` shard indexes.
- GGUF files.

The simplest path is a model directory. Component loading is useful when
weights are stored separately. See [Command line usage](cli.md#basic-invocation)
for both forms.

### Step-distilled variants

Each supported family has step-distilled variants that generate in 4–8 steps
instead of the base model's default. When `--steps` is unset the runtime picks a
default step count per family: 20 for most (FLUX.1, SD3/SD3.5, Qwen-Image,
FLUX.1-Kontext, Wan 2.1), 50 for Qwen-Image-Edit. Distilled checkpoints load
through the same pipeline as the base model; the runtime auto-detects them and
applies a few-step default (schnell 4, the rest 8) instead. Full-weight
distilled checkpoints load directly (`--model` or `--diffusion-model`); LoRA-form
distills must be merged into the base weights offline first.

The distilled variants below use the same columns as the base support matrix
above:

| Distilled variant | Task | HuggingFace repo | Common format | Backend coverage | Status |
|---|---|---|---|---|---|
| FLUX.1-schnell | Text-to-image | [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Full Diffusers directory (`--model`); the repo also has a standalone `transformer/` you can load via `--diffusion-model`  | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| SD3.5-medium-turbo | Text-to-image | [`tensorart/stable-diffusion-3.5-medium-turbo`](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo) | Full Diffusers directory (`--model`) | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| FLUX.1-Kontext Lightning | Image editing | [`camenduru/FLUX.1_Kontext-Lightning`](https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning) | Full Diffusers directory (`--model`); the repo also has a standalone `transformer/` you can load via `--diffusion-model` over a base Kontext | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| Qwen-Image Lightning | Text-to-image | [`lightx2v/Qwen-Image-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Lightning) | **LoRA adapter — merge into base first** | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| Qwen-Image-Edit Lightning | Image editing | [`lightx2v/Qwen-Image-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Lightning) (Edit file) | **LoRA adapter — merge into base first** | CUDA first, CPU/Vulkan functional, Metal experimental | Supported |
| Wan2.1-T2V-1.3B Distill | Video generation | [`lightx2v/Wan2.1-T2V-1.3B-Distill-Models`](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-Distill-Models) | Standalone single full-weight `.safetensors` (load via `--diffusion-model` over a base Wan) | CUDA first, CPU functional for validation, Metal/Vulkan experimental | Supported (Vulkan still optimizing) |

All distilled variants default to a **4–8 step** schedule when `--steps` is unset
(schnell 4, the rest 8). Only the two Qwen-Image variants ship as LoRA adapters
and must be merged into the base weights offline before use with
`scripts/merge_qwen_lora.py` — see [Merging LoRA
weights](optimization/merging-lora-weights.md). The rest are drop-in full
weights. For the exact per-variant run command (step count, `--cfg-scale` /
`--guidance` / `--flow-shift`, standalone-file vs directory form), see
[Few-step distilled models](optimization/few-step-distilled-models.md#4-per-variant-commands).

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

Its few-step **Qwen-Image-Lightning** variant ships as a LoRA adapter, so merge
it into the base weights first (`scripts/merge_qwen_lora.py`); see
[Merging LoRA weights](optimization/merging-lora-weights.md).

Command example: [Qwen-Image CLI](cli.md#qwen-image).

## Image Editing

Image editing support depends on the model family and checkpoint format.

### FLUX.1-Kontext

FLUX.1-Kontext uses an input/reference image via `--image`.

Command example: [FLUX.1-Kontext CLI](cli.md#flux1-kontext).

### Qwen-Image-Edit

Qwen-Image-Edit uses an input/reference image via `--image`.

Its few-step **Qwen-Image-Edit-Lightning** variant ships as a LoRA adapter, so
merge it into the base weights first (`scripts/merge_qwen_lora.py`); see
[Merging LoRA weights](optimization/merging-lora-weights.md).

No extra model-specific CLI switch is required for the documented
Qwen-Image-Edit checkpoints.

Command example: [Qwen-Image-Edit CLI](cli.md#qwen-image-edit).

## Video Generation

Wan video generation uses `--video`, `--frames`, and `--fps`.

MiniMax-H3 video + audio generation supports text, first-frame, first+last-frame,
and Ref2VA conditioning through standalone component loading. See
[MiniMax-H3 usage](minimax-h3.md).

Supported output formats are `auto`, `avi`, `mp4`, `mov`, `mkv`, and `webm`.
The CLI uses `ED_FFMPEG` when set and can also find imageio-ffmpeg binaries in
an active Python environment.

Wan 2.1 remains an active optimization target. Validate memory use and output
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
--vae-tiling on|off|auto
--vae-tile-size <float>
--offload-to-cpu
--dit-offload
--text-encoder-offload
--vae-offload
--auto-allocate
--auto-fit
--max-vram <GB>
```

`--vae-tiling` takes an explicit `on|off|auto` value. Under `--auto-allocate`
with a `--max-vram` budget, the runtime decides per component (diffusion
transformer, text encoder, VAE) what stays resident on the GPU and what streams
from host memory, so large models can run within a fixed VRAM budget.
`--auto-fit` goes one step further — a superset of `--auto-allocate` that also
picks the quantization to fit the budget (text encoders forced to `q8_0`, the
diffusion transformer walking a `q8_0 → q4_K` ladder), ignoring `--type`; use it
to fit a hard VRAM cap without hand-tuning quantization and placement.

See [Command line usage](cli.md#quantization-and-memory) for runnable examples
and [performance (H200)](performance-H200.md) for cache, parallelism, and
profiling behavior.

## Model-Specific Limitations

- Supported model scope is narrower than the internal enum list in the loader.
- Metal and Vulkan are experimental for DiT workloads and should be validated
  per model.
- Wan video support is available but still being optimized for memory and
  runtime behavior.
- Cache methods and sequence parallelism are workload dependent and may not be
  valid for every model or resolution.
- Component loading requires a complete, compatible set of text encoders, VAE,
  diffusion transformer, and optional vision components for the selected model.

## Related Documentation

- [Build and installation](build.md)
- [Command line usage](cli.md)
- [performance (RTX 4090)](performance-4090.md)
- [performance (H200)](performance-H200.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)
