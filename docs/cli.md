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

## Model Loading

### Diffusers Directory

Use `--model` for a model directory:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  --prompt "a glass teapot on a wooden table" \
  --output output.png
```

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
  --steps 20 \
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
  --steps 20 \
  --output qwen-edit.png
```

Use `--qwen-image-zero-cond-t` for Qwen-Image-Edit-2511. For other
Qwen-Image-Edit models, ignore this option.

Image editing support depends on the model family and checkpoint format. See
[Supported models and usage](models.md) for the public preview support matrix.

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
  --frames 40 \
  --fps 16 \
  --steps 20 \
  --cfg-scale 5.0 \
  --flow-shift 5.0 \
  --output wan.avi
```

Video flags:

```text
--video
--frames <int>
--fps <int>
--video-format auto|avi|mp4|mov|mkv|webm
```

The CLI uses `ED_FFMPEG` when set and can also find imageio-ffmpeg binaries in
an active Python environment. Wan 2.x is available in public preview and is
still being optimized for memory use and runtime behavior.

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

Supported `--type` values:

```text
f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k
```

Per-tensor type rules:

```bash
--tensor-type-rules "attn=q4_0,norm=f16"
```

Memory-oriented flags:

```text
--vae-tiling
--vae-tile-size <float>
--offload-to-cpu
--keep-text-encoder-on-cpu
--keep-vae-on-cpu
--max-vram <GB>
```

These options are workload dependent. Validate output quality and latency for
the exact model and resolution you plan to run.

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
  Qwen-Image-Edit), and video (Wan).
- The model family is recorded in the GGUF metadata, so the correct pipeline is
  selected on load regardless of the file name (editing variants included).
- Most useful for large models, where on-load quantization can take tens of
  seconds to minutes while a pre-quantized GGUF loads in seconds.

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
options as `ed-cli`, with snake_case aliases for some flags.

## Native Server CLI

Start the native HTTP server:

```bash
./build-cuda/bin/ed-server \
  --backend cuda \
  --model /path/to/FLUX.1-dev \
  --host 127.0.0.1 \
  --port 8080
```

See [API and bindings](api.md) for HTTP endpoints and curl examples.

## Related Documentation

- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [Performance and optimization](performance.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)

[← Back to README](../README.md)
