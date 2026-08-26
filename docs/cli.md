# Command Line Usage

[← Back to README](../README.md)

This document is the main command-line reference for edge-dit.cpp. It covers
the `ed-cli` inference binary, common options, model-specific examples, memory
and performance flags, and the `ed-sample` batch/benchmark helper.

Use the binary help as the exhaustive flag reference for the exact build you
are running:

```bash
./build-cuda/bin/ed-cli --help
./build-cuda/bin/ed-sample --help
```

## Binaries

CUDA builds place command-line tools under:

```text
build-cuda/bin/
```

Common entry points:

| Binary | Purpose |
|---|---|
| `ed-cli` | Single image, editing, or video generation run |
| `ed-sample` | Prompt-file based sampling and timing helper |
| `ed-server` | Native HTTP server around the C API |
| `ed-convert` | Offline weight quantization: convert a model to a pre-quantized GGUF |

For build directories used by CPU, Metal, and Vulkan builds, see
[Build and installation](build.md).

## Basic Invocation

The simplest `ed-cli` form loads a Diffusers-style model directory:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/model-dir \
  --prompt "a glass teapot on a wooden table" \
  --output output.png
```

Common generation flags:

```text
--backend auto|cpu|cuda|vulkan|metal|gpu
--prompt <text>
--negative-prompt <text>
--width <int>
--height <int>
--steps <int>
--seed <int64>
--threads <int>
--output <path>
```

`--backend gpu` is an alias for the available GPU backend selected by the
build/runtime environment. `--gpu` is a shorthand for `--backend gpu`.

`--steps` accepts `-1` (or may be omitted) to let the runtime pick the step
count: step-distilled checkpoints (FLUX.1-schnell, Turbo, Lightning, …) default
to a few-step schedule (4 or 8). Base-model benchmark presets use 20 steps for
FLUX.1, FLUX.1-Kontext, SD3, and MiniMax-H3; 30 for Qwen-Image,
Qwen-Image-Edit, and Wan 1.3B; and 50 for Wan 14B. An explicit `--steps N`
always overrides this. See
[Few-step distilled models](optimization/few-step-distilled-models.md).

### Diffusers Directory

Use `--model` for a model directory:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  --prompt "a glass teapot on a wooden table" \
  --output output.png
```

`--model` is the unprefixed, general model input. It is intended for a complete
Diffusers directory or a self-contained single-file/GGUF checkpoint. It may be
combined with component flags, but a bare Diffusers transformer file is not
portable across model families when passed as `--model`: it usually lacks the
text encoders/VAE, and its tensor names may require the diffusion-component
prefix and the adjacent `transformer/config.json` hint.

### Component Weights

Use component paths when weights are stored separately:

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

Available component flags include:

```text
--diffusion-model <path>
--vae <path>
--clip_l <path>
--clip_g <path>
--t5xxl <path>
```

Component loading requires a compatible set of encoders, VAE, diffusion model,
and optional vision components for the selected model family.

`--diffusion-model` specifically identifies a standalone denoising transformer.
The loader adds the canonical diffusion prefix and uses nearby transformer
metadata when available. Therefore use `--diffusion-model`, not `--model`, for
a transformer-only file, together with the family-specific encoder and VAE
flags. The exact required component set differs by family (CLIP/T5, LLM, video
VAE, audio VAE, and so on).

The `--diffusion-model` path can be a **pre-quantized GGUF produced by
`ed-convert`** (see [Pre-quantized GGUF](#pre-quantized-gguf-with-ed-convert)),
so you can quantize just the transformer offline and keep the encoders/VAE in
their original precision — a common way to shrink the largest component while
loading everything else from the standard component files.

## Text-to-Image

<a id="flux1-dev"></a>

### FLUX.1-dev

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --guidance 3.5 \
  --seed 0 \
  --output flux.png
```

`--guidance` controls FLUX distilled guidance.

<a id="flux2"></a>

### FLUX.2 [klein] 4B

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/FLUX.2-klein-4B \
  --prompt "a glass teapot on a wooden table" \
  -W 1024 -H 1024 --steps 4 --guidance 1.0 \
  --output flux2-klein.png
```

For component loading, pass the FLUX.2 transformer through
`--diffusion-model`, its language model through `--llm`, and its VAE through
`--vae`. Components must come from the same FLUX.2 variant.

<a id="sd3-sd35"></a>

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

SD3-family models can skip T5XXL to reduce memory:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/stable-diffusion-3-medium-diffusers \
  --prompt "a glass teapot on a wooden table" \
  --no-t5 \
  --output sd3-no-t5.png
```

`--no-t5` is only valid for SD3-family models and reduces prompt adherence.

### Qwen-Image

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/Qwen-Image \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --cfg-scale 4.0 \
  --seed 0 \
  --output qwen.png
```

## Image Editing

<a id="flux1-kontext"></a>

### FLUX.1-Kontext

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "make the object look like brushed metal" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --guidance 2.5 \
  --output flux-kontext.png
```

`--image` supplies the input/reference image required by FLUX.1-Kontext.

### Qwen-Image-Edit

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/Qwen-Image-Edit \
  --image /path/to/input.png \
  --prompt "change the background to a clean studio" \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --cfg-scale 4.0 \
  --output qwen-edit.png
```

Image editing support depends on the model family and checkpoint format. See
[Supported models and usage](models.md) for the supported-models matrix.

The `--qwen-image-zero-cond-t` flag toggles the `zero_cond_t` modulation path.
Leave it **off** for the plain `Qwen-Image-Edit` checkpoint above. It is only
needed when a checkpoint expects that path: checkpoints that ship
`"zero_cond_t": true` in `transformer/config.json` (e.g. Qwen-Image-Edit-2511)
enable it automatically, and the distilled **Qwen-Image-Edit-Lightning** merge
requires passing the flag explicitly (see
[Few-step distilled models](optimization/few-step-distilled-models.md#qwen-image-edit-lightning-4-steps-image-editing--merge-lora-first)).

## Video Generation

Wan text-to-video uses `--video`:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --video \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "a glass teapot rotating on a wooden table" \
  --width 832 \
  --height 480 \
  --frames 41 \
  --fps 16 \
  --steps 30 \
  --cfg-scale 5.0 \
  --flow-shift 3.0 \
  --output wan.avi
```

The larger **Wan2.1-T2V-14B** can render 720P (`1280x720`). Note the higher
resolution uses `--flow-shift 5.0` (vs `3.0` for 480P):

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --video \
  --model /path/to/Wan2.1-T2V-14B-Diffusers \
  --prompt "a glass teapot rotating on a wooden table" \
  --width 1280 \
  --height 720 \
  --frames 41 \
  --fps 16 \
  --steps 50 \
  --cfg-scale 5.0 \
  --flow-shift 5.0 \
  --output wan-14b-720p.avi
```

Video flags:

```text
--video
--frames <int>
--video-duration <seconds>  # MiniMax-H3 only
--fps <int>
--video-format auto|avi|mp4|mov|mkv|webm
```

The CLI uses `ED_FFMPEG` when set and can also find imageio-ffmpeg binaries in
an active Python environment. Wan 2.1 is available and is
still being optimized for memory use and runtime behavior.

## Few-Step Distilled Models

Step-distilled checkpoints (FLUX.1-schnell, SD3.5-medium-turbo,
Qwen-Image-Lightning, Kontext Lightning, Wan distill, …) generate in 4-8 steps.
Pass `--steps -1` (or omit `--steps`) to let the runtime detect the distilled
model and choose the few-step default; an explicit `--steps N` always overrides.
Most distilled models are guidance-distilled, so leave `--cfg-scale` at its
default `1.0` (single forward) rather than raising it — **except** where the
model card says otherwise (SD3.5-medium-turbo wants `--cfg-scale 1.5`; at `1.0`
its geometry collapses).

Two loading shapes, depending on how the variant is published:

```bash
# Full-weight distilled directory (auto step count)
./build-cuda/bin/ed-cli \
  --backend cuda --type q8_0 \
  --model /path/to/sd3.5-medium-turbo \
  --steps -1 --cfg-scale 1.5 \
  --auto-fit --max-vram 8 --vae-tiling auto \
  --prompt "a glass teapot on a wooden table" \
  --output turbo.png

# Distilled DiT weights + a base model's VAE/text encoders via --diffusion-model
./build-cuda/bin/ed-cli \
  --backend cuda --type q8_0 \
  --model /path/to/qwen-image \
  --diffusion-model /path/to/qwen-image-lightning-merged/transformer/diffusion_pytorch_model.safetensors.index.json \
  --steps -1 --cfg-scale 1.0 \
  --auto-fit --max-vram 8 --vae-tiling auto \
  --prompt "a red apple on a wooden table" \
  --output lightning.png
```

**For a ready-to-run command for every distilled variant** (with the exact
`--cfg-scale` / `--guidance` / `--flow-shift`, standalone-file vs directory
form, and which ones need a LoRA merge first), see
[Few-step distilled models §4](optimization/few-step-distilled-models.md#4-per-variant-commands).
Variants shipped as LoRA adapters (the two Qwen-Image ones) must be merged into
the base weights offline before use — the CLI loads full weights, not LoRA
deltas.

## Quantization and Memory

On-load weight type selection:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --type q4_0 \
  --output flux-q4.png
```

`--type` is a global loading policy: it applies to every eligible tensor in
every loaded component (DiT, text encoder, video VAE, and audio VAE). Biases,
norms, embeddings, protected tensors, and shapes incompatible with a block
quant remain at their stored type.

Supported `--type` values:

```text
preserve f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k
```

`preserve` is the default and leaves every source tensor at its stored type.
The older value `auto` remains accepted as a compatibility alias for
`preserve`; it does not perform automatic quantization.

> **Qwen-Image models and FP16:** the Qwen-Image family (`qwen-image`,
> `qwen-image-edit`, and their distilled/lightning variants) is not supported in
> FP16 — its DiT activations exceed FP16's dynamic range and silently saturate,
> producing a corrupt (all-white) image. If you pass `--type f16` for a Qwen
> model, edge-dit automatically switches it to BF16 and logs a warning. Use
> `bf16` (or a quantized type) explicitly to avoid the warning.

Per-tensor type rules:

```bash
--tensor-type-rules "attn=q4_0,norm=f16"
```

Memory-oriented flags:

```text
--vae-tiling on|off|auto
--vae-tile-size <float>
--offload-to-cpu
--dit-offload
--text-encoder-offload
--vae-offload
--max-vram <GB>
--auto-allocate
--auto-fit
```

`--vae-tiling` is tri-state `on|off|auto` and defaults to `auto`, which enables
tiled VAE decode automatically on low-VRAM GPUs (<=25 GB); pass `off` to force
it off. The offload flags share one semantics — **weights kept on CPU and staged
to the GPU per compute** (compute always runs on the GPU): `--offload-to-cpu`
offloads the whole model, while `--dit-offload` / `--text-encoder-offload` /
`--vae-offload` offload just that one component. `--auto-allocate` places each
component (DiT, text encoder, VAE) against a planning budget of
`min(--max-vram, free)`, keeping a component resident when it fits and
offloading (staging) it otherwise. By itself this remains a placement-planning
budget. On single-device CUDA, `--auto-fit --max-vram <GB>` additionally installs a guarded
allocation ceiling below the requested value, reserving 1 GiB for external
library workspaces. An allocation that cannot fit is rejected before the
process crosses the requested ceiling.

These options are workload dependent. Validate output quality and latency for
the exact model and resolution you plan to run.

### Budget-driven placement (`--auto-allocate`) and full auto (`--auto-fit`)

Two levels of automation size a run to a VRAM planning budget instead of tuning
`--type` and offload flags by hand:

- `--auto-allocate` — you pick the quantization (`--type`); the runtime decides,
  per component (DiT / text encoder / VAE), what stays resident on the GPU and
  what streams from host. `--max-vram` is not a universal process-level peak clamp.
- `--auto-fit` — fully automatic. It **implies `--auto-allocate`** and, in
  addition, chooses the quantization itself: the DiT is driven down the ladder
  `q8_0 → q4_k` to the highest level that stays resident within the budget, the
  text encoder is lowered to at most `q8_0` (an already smaller quant stays unchanged), and
  placement is decided as above. For the text encoder and DiT, this automatic
  decision supersedes the global `--type` without ever increasing an already
  lower-precision source. The VAEs are not replanned: they still follow
  `--type`, where `preserve` means keeping each source tensor's stored type.
  With an explicit `--max-vram` on
  single-device CUDA, it also enables the hard allocation guard; an intrinsically too-large
  graph fails safely instead of exceeding the budget.

`--auto-fit` measures each component's real compute-buffer footprint at the
requested resolution (`-W`/`-H`, and `--frames` for video) to size the resident
headroom, so pass the generation size you intend to use. Without a size it falls
back to a conservative fixed headroom.

MiniMax-H3 applies this policy to four independently placeable components: DiT,
Qwen3-VL, video VAE, and audio VAE. Its measurement graph conservatively includes
mixed image, video, paired-audio, and additional-audio conditioning, so the same
placement remains valid across FL2VA and Ref2VA workflows. The MiniMax-H3 video
VAE also keeps its model-specific fixed `16x16` tiling regardless of the generic
`--vae-tiling`/`--vae-tile-size` values.

For MiniMax-H3, `--video-duration <seconds>` converts a requested duration at
the model's fixed 24 fps to the nearest legal `17k+5` frame count (minimum 22). Keep using
`--video-frames` when an exact legal count is required; the two options are
mutually exclusive.

```bash
# MiniMax-H3 with automatic quantization/placement and a 40 GiB CUDA ceiling.
./build-cuda/bin/ed-cli --video \
  --diffusion-model /path/to/minimax_h3_fl2va-diffusers-Q8_0.gguf \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --llm /path/to/qwen3vl_32b_minimax_h3-Q8_0.gguf \
  --auto-fit --max-vram 40 --vae-tiling auto \
  -W 864 -H 480 --video-duration 5 --steps 20 --cfg-scale 1 \
  --prompt "A cinematic sunset over layered mountain ridges." \
  --video-format mp4 --output minimax-h3-autofit.mp4
```

```bash
# Fully automatic under an 8 GiB budget — system picks DiT quant + placement.
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --auto-fit --max-vram 8 --vae-tiling auto \
  -W 1024 -H 1024 --steps 20 --guidance 3.5 \
  --output flux-autofit.png
```

The largest win is a big DiT under a mid-size budget: for a model like
Qwen-Image (~20 GiB transformer), a mid-size budget can keep the DiT resident at
`q4_k` instead of offloading it, so DiT sampling avoids the per-step host-to-GPU
streaming cost. The benefit is largest exactly when a higher precision would not
fit resident but `q4_k` does.

### Pre-quantized GGUF with `ed-convert`

`--type` quantizes weights on every load. `ed-convert` runs the quantization
once and writes a self-contained GGUF, so later runs load the pre-quantized
weights directly and skip on-load conversion. The GGUF is also a portable
artifact: share it and others can run it on edge-dit.cpp without the original
model or a conversion step.

```bash
# convert once (any quant type; --tensor-type-rules works too)
./build-cuda/bin/ed-convert --model /path/to/FLUX.1-dev --type q4_k --output flux-q4k.gguf

# then load the GGUF like any model
./build-cuda/bin/ed-cli --backend cuda --model flux-q4k.gguf --prompt "..." --output flux.png
```

Notes:

- Accepts the same `--type` values and `--tensor-type-rules` as on-load
  quantization; the quantized weights are bit-identical to the on-load path.
- Works across model families: SD3, FLUX, Qwen-Image, editing (FLUX-Kontext,
  Qwen-Image-Edit), and video (Wan and MiniMax-H3).
- For a full model (a diffusers directory or a complete single-file checkpoint)
  the model family is recorded in the GGUF metadata, so the correct pipeline is
  selected on load regardless of the file name (editing variants included).
- Most useful for large models, where on-load quantization can take tens of
  seconds to minutes while a pre-quantized GGUF loads in seconds.

#### Per-component quantization (quantize the transformer, load by components)

The transformer is by far the largest component, so a common workflow is to
**pre-quantize only the transformer** and load it together with the original
encoders / VAE via component flags — no need to convert the whole model.
`ed-convert` reads the transformer's `config.json` to detect the family and
writes it into the GGUF, so the quantized transformer loads standalone as a
`--diffusion-model`.

```bash
# 1) quantize just the transformer to a portable q8_0 GGUF (q4_k also works)
./build-cuda/bin/ed-convert \
    --diffusion-model /path/to/stable-diffusion-3-medium/transformer/diffusion_pytorch_model.safetensors \
    --type q8_0 --output sd3-dit-q8.gguf

# 2) load it by components: quantized DiT + original VAE + CLIP encoders
./build-cuda/bin/ed-cli --backend cuda \
    --diffusion-model sd3-dit-q8.gguf \
    --vae   /path/to/stable-diffusion-3-medium/vae/diffusion_pytorch_model.safetensors \
    --clip_l /path/to/stable-diffusion-3-medium/text_encoder/model.safetensors \
    --clip_g /path/to/stable-diffusion-3-medium/text_encoder_2/model.safetensors \
    --no-t5 --prompt "a glass teapot on a wooden table" \
    -W 1024 -H 1024 --cfg-scale 5.0 --flow-shift 3.0 --output sd3.png
```

Qwen-Image works the same way — its transformer ships as a shard index, which
`ed-convert` accepts directly. Qwen loads its text encoder through `--llm`:

```bash
# 1) quantize the Qwen-Image transformer (shard index) to q8_0
./build-cuda/bin/ed-convert \
    --diffusion-model /path/to/Qwen-Image/transformer/diffusion_pytorch_model.safetensors.index.json \
    --type q8_0 --output qwen-dit-q8.gguf

# 2) quantized DiT + original VAE + LLM text encoder
./build-cuda/bin/ed-cli --backend cuda \
    --diffusion-model qwen-dit-q8.gguf \
    --vae /path/to/Qwen-Image/vae/diffusion_pytorch_model.safetensors \
    --llm /path/to/Qwen-Image/text_encoder/model.safetensors.index.json \
    --prompt "a glass teapot on a wooden table" \
    -W 1024 -H 1024 --cfg-scale 4.0 --output qwen.png
```

MiniMax-H3 DiT transformers use the same transformer-only conversion path:

```bash
./build-cuda/bin/ed-convert --diffusion-model minimax_h3_fl2va_bf16.safetensors \
  --type q8_0 --output minimax_h3_fl2va-Q8_0.gguf
```

Every component can also be converted independently. The input flag itself
states its semantics; the output GGUF records both a normalized tensor prefix
and `edgedit.component_kind`, so loading does not depend on the file name. Pass
the result to the matching inference component flag:

```bash
# CLIP-L (works with --clip_l; the loader rebases UNet/DiT prefixes)
./build-cuda/bin/ed-convert \
    --clip_l /path/to/text_encoder/model.safetensors \
    --type q8_0 --output clip-l-Q8_0.gguf

# T5-XXL
./build-cuda/bin/ed-convert \
    --t5xxl /path/to/text_encoder_2/model.safetensors.index.json \
    --type q8_0 --output t5xxl-Q8_0.gguf

# MiniMax-H3 Qwen3-VL; tensor signatures select its mapping automatically.
./build-cuda/bin/ed-convert \
    --llm /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
    --type q4_k --output qwen3vl-minimax-Q4_K.gguf
```

Supported component inputs are `--diffusion-model`, `--vae`, `--audio-vae`,
`--clip_l`, `--clip_g`, `--t5xxl`, `--llm`, and `--llm-vision`. Component
conversion requires GGUF output and canonical names (so `--raw-names` is
rejected). A component GGUF loaded through the wrong flag is
rejected from its metadata instead of being silently prefixed incorrectly.
`llm-vision` accepts a vision-only checkpoint; when given a combined Qwen-VL
checkpoint it explicitly selects only the canonical `text_encoders.llm.visual.*`
subtree rather than re-prefixing language tensors as visual tensors.

The converter accepts ordinary F32/F16/BF16/FP8 safetensors and supported GGUF
sources. Packed Comfy U8/NVFP4/AWQ files are not a generic source format:
their packed `.weight`, `comfy_quant`, and scale tensors need the original
quantization kernels and cannot be dequantized by `ed-convert`. Such an input
is rejected (including when the packed tensors are spread across a safetensors
shard index); start from BF16/F16 or from an already supported GGUF instead.
VAE and audio-VAE files use the same standalone rule and can also be included
in a combined GGUF as described below.

Alternatively, `ed-convert` can **merge** the transformer with external
encoders / VAE into one standalone GGUF — pass the components at convert time
and load the single output file with `--model`:

```bash
./build-cuda/bin/ed-convert \
    --diffusion-model /path/to/stable-diffusion-3-medium/transformer/diffusion_pytorch_model.safetensors \
    --vae   /path/to/stable-diffusion-3-medium/vae/diffusion_pytorch_model.safetensors \
    --clip_l /path/to/stable-diffusion-3-medium/text_encoder/model.safetensors \
    --clip_g /path/to/stable-diffusion-3-medium/text_encoder_2/model.safetensors \
    --no-t5 --type q8_0 --output sd3-merged-q8.gguf

./build-cuda/bin/ed-cli --backend cuda --model sd3-merged-q8.gguf --no-t5 \
    --prompt "a glass teapot on a wooden table" \
    -W 1024 -H 1024 --cfg-scale 5.0 --flow-shift 3.0 --output sd3.png
```

MiniMax-H3 can likewise merge its DiT, Qwen3-VL, video VAE, and audio VAE. The
result is loaded as one complete model rather than as a transformer component:

```bash
./build-cuda/bin/ed-convert \
    --diffusion-model /path/to/minimax_h3_fl2va_bf16.safetensors \
    --llm /path/to/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
    --type q4_k --output minimax_h3_fl2va-all-Q4_K.gguf

./build-cuda/bin/ed-cli --backend cuda \
    --model minimax_h3_fl2va-all-Q4_K.gguf --video \
    --prompt "A red panda walking through a bamboo forest" \
    -W 864 -H 480 --frames 22 --steps 20 --cfg-scale 1 \
    --sampler res_multistep --scheduler simple --output minimax-h3.avi
```

Passing one component flag produces a component GGUF; passing two or more
component flags merges them into a complete single-file GGUF. `--model` has a
separate, strict meaning: it must point to a complete model directory and cannot
be combined with component flags. No model-version or component-kind hint is
required.

This also covers **transformer-only distilled checkpoints** (e.g.
Qwen-Image-Lightning, Wan2.1-Distill, Kontext-Lightning), which ship as a bare
DiT `.safetensors` or shard index: `ed-convert` recovers the family from the
transformer's `config.json` and records it in the GGUF, so the quantized DiT
loads as a `--diffusion-model` on top of the base model's encoders / VAE. A full
diffusers-directory checkpoint (base or distilled) can also be converted whole
with `--model <dir>` and then loaded standalone.

### Activation-calibrated imatrix quantization

Low-bit quantization (notably `q4_k`) loses quality because every weight column
is quantized with the same uniform importance. With an *importance matrix*
(imatrix) the quantizer instead weights each input channel by how much it drives
the layer output -- measured offline as the mean squared activation `E[x^2]`
over a few calibration prompts (using activations as the saliency signal is an
idea borrowed from AWQ; this is a per-channel imatrix weighting, not AWQ's
per-channel scaling). `ed-convert` accepts this importance vector via
`--imatrix`:

```bash
# 1) calibrate: run the model over a few prompts to collect per-channel importance
python tools/imatrix/calibrate.py --model /path/to/sd3-medium --outdir imatrix-out \
    --steps 6 --nprompts 16
# -> writes imatrix-out/imatrix.gguf

# 2) convert with the importance vector (works with any low-bit --type)
./build-cuda/bin/ed-convert --model /path/to/sd3-medium --type q4_k \
    --imatrix imatrix-out/imatrix.gguf --output sd3-q4k-imatrix.gguf
```

Notes:

- `--imatrix` only affects the offline conversion; load and inference speed are
  identical to a plain `q4_k` GGUF (the importance vector is not stored).
- Without `--imatrix`, or when a channel's entry is missing/mismatched, the
  quantizer falls back to the uniform (all-ones) weighting, so plain conversion
  is byte-for-byte unchanged.
- The quality gain is real but modest and prompt-dependent (on SD3 `q4_k`,
  roughly sub-dB to ~1 dB PSNR versus plain `q4_k`); it does not raise the
  fundamental `q4_k` quality ceiling. For a larger quality jump, prefer a higher
  bit width (`q6_k`/`q8_0`) or mixed precision via `--tensor-type-rules`.
- `calibrate.py` ships a SD3 calibration pipeline; other model families need the
  calibration pass adapted to their loader.

## Performance Flags

Attention and graph execution:

```text
--flash-attention
--no-flash-attention
--profile-graph-cuts
--profile-graph-cuts-top <n>
--profile-graph-cuts-all-ranks
```

Cache modes:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --cache dbcache \
  --cache-fn-blocks 8 \
  --cache-residual-threshold 0.08 \
  --output flux-cache.png
```

`--cache-residual-threshold 0.08` above is a DBCache/CacheDiT example, not a
global cache default. MagCache, DiCache, and SenCache use their own
method-specific thresholds unless this flag is passed explicitly.

Supported cache mode names:

```text
off easycache ucache dbcache taylorseer cache-dit magcache dicache sencache
```

Common cache flags:

```text
--cache-threshold <float>
--cache-start <float>
--cache-end <float>
--cache-error-decay <float>
--cache-no-reset-error
--cache-relative-threshold
--cache-absolute-threshold
--cache-fn-blocks <int>
--cache-bn-blocks <int>
--cache-residual-threshold <float>
--cache-max-accumulated-residual-diff <float>
--cache-warmup-steps <int>
--cache-max-cached-steps <int>
--cache-max-continuous-cached-steps <int>
--cache-taylor-order <int>
--cache-taylor-skip <int>
--cache-scm-mask <csv>
--cache-calibrate <path>
--cache-profile <path>
--cache-static-scm
```

Cache methods are experimental speed-quality tradeoffs. SenCache requires a
calibrated profile via `--cache-profile` or a calibration run with
`--cache-calibrate`.

## Parallel Execution

CFG parallelism:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --devices 0,1 \
  --cfg-parallel-size 2 \
  --model /path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --output flux-cfg-parallel.png
```

Sequence parallelism:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --devices 0,1,2,3 \
  --sp-size 4 \
  --model /path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --output flux-sp.png
```

Parallel flags:

```text
--devices <csv>
--cfg-parallel-size <n>
--cfg-size <n>
--sp-size <n>
--tp-size <n>
```

`--tp-size` is reserved and currently must remain `1`. Sequence parallelism is
workload dependent; small FLUX runs can be slower than single-GPU execution.
Official performance claims require the `performance` build profile described
in [Build and installation](build.md).

## Batch Sampling and Timing

`ed-sample` reads prompts from a text file and writes results to an output
directory:

```bash
./build-cuda/bin/ed-sample \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --prompt_file prompts.txt \
  --output_dir samples \
  --width 1024 \
  --height 1024 \
  --num_steps 20 \
  --warmup 1 \
  --repeat 3
```

`ed-sample` accepts the same backend, model loading, cache, and basic sampling
options as `ed-cli`, with snake_case aliases for some flags. For LTX-2.3 it
accepts `--end-img` and the latent hires options. For MiniMax-H3 it also accepts
`--end-img`, repeatable `--ref-image`, numbered-frame-directory
`--ref-video`, `--ref-video-audio`, and `--ref-audio`. Generated audio is
written as a WAV sidecar next to each AVI.

## Native Server CLI

Start the native HTTP server:

```bash
./build-cuda/bin/ed-server \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --host 127.0.0.1 \
  --port 8080 \
  --max-vram 20
```

`ed-server` defaults to `--auto-allocate`: it preserves stored tensor types and
automatically places each component within `min(--max-vram, live free VRAM)`.
Omit `--max-vram` to use live free VRAM as the planning budget. Use
`--auto-fit` to let the runtime also choose TE/DiT quantization. For manual
control, pass `--no-auto-allocate`, an optional `--type`/`--tensor-type-rules`,
and any of `--dit-offload`, `--text-encoder-offload`, `--vae-offload`, or
`--offload-to-cpu`.

See [API and bindings](api.md) for HTTP endpoints and curl examples.

The native server accepts MiniMax-H3 and LTX-2.3 component flags
(`--diffusion-model`, `--vae`, `--audio-vae`, `--llm`, plus
`--embeddings-connectors` and `--latent-upscaler` for LTX-2.3) and exposes `POST
/ed/v1/videos/generations`. Video responses contain base64 PNG frames and, when
available, interleaved float32 audio.

## Related Documentation

- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [performance (RTX 4090)](performance-4090.md)
- [performance (H200)](performance-H200.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)
