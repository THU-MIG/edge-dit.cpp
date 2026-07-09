# edge-dit.cpp

<div align="center">

**A DiT-first C/C++ inference engine for edge-oriented image, editing, and video generation**

[![Status](https://img.shields.io/badge/status-v0.1--alpha-orange)](#release-status)
[![Backend](https://img.shields.io/badge/backend-CUDA--first-blue)](#backend-support)
[![Models](https://img.shields.io/badge/models-SD3%20%7C%20FLUX%20%7C%20Qwen--Image%20%7C%20Wan-brightgreen)](#supported-models)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

</div>

---

## What is edge-dit.cpp?

**edge-dit.cpp** is a native **C/C++ inference engine** for modern **Diffusion Transformer (DiT)** generation models.

It provides a shared runtime for representative DiT model families, including **SD3/SD3.5**, **FLUX**, **Qwen-Image**, **Qwen-Image-Edit**, **FLUX-Kontext**, and **Wan**, covering:

- **text-to-image** generation;
- **image editing** and reference-guided generation;
- **text-to-video / image-to-video** generation;
- CUDA-first execution with ggml backend integration;
- low-memory deployment through quantization, VAE tiling, and CPU offload;
- DiT-specific optimizations such as CFG parallelism, sequence parallelism, and cache-based timestep reuse;
- embeddable interfaces including **C API**, **CLI**, **HTTP server**, and optional **Python binding**.

The goal of edge-dit.cpp is **not** to be another Python pipeline wrapper. It is designed as a lightweight, embeddable runtime for DiT-based image, editing, and video generation.

---

## Why edge-dit.cpp?

Modern image and video generation models are increasingly moving from UNet-based diffusion models to **DiT / MMDiT / rectified-flow transformer** architectures. However, many existing inference stacks are either:

- Python-heavy and difficult to embed into native applications;
- optimized around a single model family;
- focused on server GPUs rather than local / edge-oriented deployment;
- missing unified support for image generation, image editing, and video generation;
- difficult to benchmark reproducibly across cache, parallelism, memory, and backend configurations.

edge-dit.cpp focuses on a different target:

```text
Native C/C++ runtime
        +
DiT-first model abstraction
        +
CUDA-first inference
        +
low-memory edge deployment
        +
image / editing / video unified pipeline
```

---

## Release Status

edge-dit.cpp is currently in **v0.1-alpha**.

This means:

- CUDA is the **first-class** backend and the main path for official performance evaluation.
- CPU is supported for build, smoke tests, fallback operators, CPU offload, and selected low-speed inference.
- Metal and Vulkan are experimental backends.
- Cache and sequence parallelism are functional but still being tuned.
- Video generation support is available through Wan pipelines, while video-specific memory/runtime optimization is ongoing.

The current release is intended for **research, benchmarking, and early integration**, not yet for production deployment.

---

## Latest News

- **2026-07:** v0.1-alpha scope frozen around SD3/SD3.5, FLUX, Qwen-Image, Qwen-Image-Edit, FLUX-Kontext, and Wan.
- **2026-07:** Added Flux SP profiling and graph-cut instrumentation for diagnosing communication and segmentation overhead.
- **2026-07:** Added cache runtime options including EasyCache, UCache, DBCache, TaylorSeer, CacheDiT, MagCache, and DiCache interfaces.
- **2026-07:** Added Wan text-to-video pipeline support and video output path.

---

## Key Features

### DiT-first Runtime

- Shared runtime context for DiT-based generation models.
- Unified model loading from diffusers-style directories and component paths.
- Common pipeline abstraction for image generation, image editing, and video generation.
- Shared graph execution, memory control, profiling, and backend selection.

### Model Family Coverage

- SD3 / SD3.5 MMDiT compatibility baseline.
- FLUX rectified-flow transformer pipeline.
- Qwen-Image text-aware image generation.
- Qwen-Image-Edit instruction-based and text-aware image editing.
- FLUX-Kontext in-context image generation and editing.
- Wan T2V video DiT pipeline.

### Edge Deployment Stack

- CUDA-first execution.
- CPU build / smoke / offload path.
- On-the-fly weight quantization.
- Mixed tensor dtype rules.
- VAE tiled decoding.
- Text encoder and VAE CPU offload.
- Compute graph VRAM limiting.

### DiT Execution Optimizations

- cuDNN SDPA fast attention path when available.
- Flash attention toggle.
- CFG parallelism.
- Sequence parallelism.
- Graph-cut profiling for SP communication.
- Cache-based timestep / block reuse.
- CUDA profiling hooks for kernel-level analysis.

### Interfaces

- `ed-cli` command-line interface.
- `ed-server` HTTP API.
- Stable C API in `include/edge-dit.h`.
- Optional Python binding under `bindings/python`.

---

## Supported Models

The first-stage support scope is intentionally small and representative. We prioritize full pipeline correctness and reproducible benchmarking over supporting as many models as possible.

| Model family | Task | Status | Notes |
|---|---|---|---|
| **SD3 / SD3.5** | Text-to-image | Full pipeline | MMDiT compatibility baseline |
| **FLUX.1-dev / FLUX.1-schnell** | Text-to-image | Full pipeline | Rectified-flow DiT image generation |
| **FLUX.1-Kontext-dev** | Image editing / reference generation | Full pipeline | In-context image generation and editing |
| **Qwen-Image** | Text-to-image | Full pipeline | Text-aware image generation, especially Chinese text rendering |
| **Qwen-Image-Edit** | Image editing | Full pipeline | Instruction-based and text-aware editing |
| **Wan2.1-T2V-1.3B** | Text-to-video | Experimental full pipeline | Lightweight Wan video pipeline |
| **Wan2.2-T2V-A14B** | Text-to-video | Experimental full pipeline | Larger Wan video pipeline; optimization ongoing |

Models such as **Z-Image**, **Sana**, **LTX-Video**, **HunyuanVideo**, **Wan VACE**, and other next-generation DiT models are tracked as roadmap targets and are **not** part of the v0.1-alpha support commitment.

---

## Backend Support

| Backend | Status | Intended use |
|---|---|---|
| **CUDA** | First-class | Main inference backend for image, editing, and video pipelines |
| **CPU** | Stable / reference | Build, smoke tests, fallback ops, CPU offload, selected low-speed inference |
| **Metal** | Experimental | macOS local demo and edge feasibility validation |
| **Vulkan** | Experimental / roadmap | Cross-vendor GPU exploration; not recommended for production yet |

For v0.1-alpha, **CUDA is the only backend used for official performance numbers**.

---

## Performance Snapshot

Performance numbers in this alpha are benchmark snapshots rather than a final
leaderboard. The current public reports are:

| Report | Scope | Notes |
|---|---|---|
| [SP benchmark report](docs/sp_benchmark_report.md) | SD3, FLUX, Qwen-Image, and Wan sequence parallelism | Historical H200 measurements with reproducible commands |
| [Flux SP profile root cause](docs/flux_sp_profile_root_cause.md) | FLUX sequence-parallel graph-cut profiling | Engineering notes for communication and graph overhead |

> **Note on Sequence Parallelism:** SP is workload-sensitive. It can help long-sequence workloads such as high-resolution image generation and video DiT, but may hurt small-batch, low-step inference due to communication and graph segmentation overhead.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/yiming-l21/edge-dit.cpp
cd edge-dit.cpp
git submodule update --init --recursive
```

### 2. Build

CUDA build:

```bash
bash ./scripts/build_cuda.sh
```

CPU build:

```bash
bash ./scripts/build_cpu.sh
```

Experimental Vulkan build:

```bash
bash ./scripts/build_vulkan.sh
```

Experimental Metal build:

```bash
bash ./scripts/build_metal.sh
```

After building, the CLI binary is expected at:

```bash
./build-cuda/bin/ed-cli
./build-cpu/bin/ed-cli
./build-vulkan/bin/ed-cli
./build-metal/bin/ed-cli
```

---

## Basic Usage

### Text-to-Image with a Diffusers Directory

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  --prompt "a cinematic photo of a cat wearing sunglasses, soft morning light" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --seed 0 \
  --output output.png
```

Short aliases are also supported:

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/flux-dev \
  -p "a cinematic photo of a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 20 -s 0 \
  -o output.png
```

### Component Path Loading

For SD3 and FLUX-style component loading:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --diffusion-model /path/to/transformer.safetensors \
  --clip_l /path/to/clip_l.safetensors \
  --clip_g /path/to/clip_g.safetensors \
  --t5xxl /path/to/t5xxl.safetensors.index.json \
  --vae /path/to/vae.safetensors \
  --prompt "a glass teapot on a wooden table, soft morning light" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --seed 0 \
  --output output.png
```

For SD3 low-memory smoke runs, T5 can be skipped:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3-medium \
  --no-t5 \
  --prompt "a cat wearing sunglasses" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --output output.png
```

### Image Editing

For Qwen-Image-Edit or FLUX-Kontext:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/qwen-image-edit \
  --image input.png \
  --prompt "change the background to a snowy mountain while keeping the subject unchanged" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --seed 0 \
  --output edited.png
```

For Qwen-Image-Edit checkpoints that require `zero_cond_t`:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/qwen-image-edit \
  --image input.png \
  --prompt "replace the text on the sign with 'edge-dit.cpp'" \
  --qwen-image-zero-cond-t \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --output edited.png
```

### Video Generation with Wan

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --video \
  --model /path/to/Wan2.2-T2V-A14B-Diffusers \
  --prompt "a small robot walking through a rainy neon street" \
  --width 832 \
  --height 480 \
  --frames 81 \
  --fps 16 \
  --steps 50 \
  --seed 0 \
  --cfg-scale 5.0 \
  --flow-shift 5.0 \
  --video-format mp4 \
  --output output.mp4
```

If `mp4` is not available in your environment, use `avi`:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --video \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  -p "a dog running on the beach at sunset" \
  -W 832 -H 480 \
  --frames 81 \
  --fps 16 \
  --steps 50 \
  -o output.avi
```

---

## Memory and Quantization

### On-the-Fly Weight Quantization

Use `--type` to quantize weights during loading.

```bash
# Q8 quantization
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --type q8_0 \
  -p "a cat" \
  -W 1024 -H 1024 \
  --steps 20 \
  -o output.png

# Q4_K quantization
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --type q4_k \
  -p "a cat" \
  -W 1024 -H 1024 \
  --steps 20 \
  -o output.png
```

Supported weight types:

```text
f32, f16, bf16,
q4_0, q4_1,
q5_0, q5_1,
q8_0,
q2_k, q3_k, q4_k, q5_k, q6_k
```

### Mixed Tensor Type Rules

Keep sensitive tensors in higher precision while quantizing the rest:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --type q4_k \
  --tensor-type-rules "norm=f16,bias=f32" \
  -p "a cat" \
  -W 1024 -H 1024 \
  --steps 20 \
  -o output.png
```

### VAE Tiling

VAE tiled decoding reduces peak memory for high-resolution image/video decode.

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --vae-tiling \
  -p "a cat in a forest" \
  -W 2048 -H 2048 \
  --steps 20 \
  -o output.png
```

Use a finer tile size:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --vae-tiling \
  --vae-tile-size 4 \
  -p "a cat in a forest" \
  -W 2048 -H 2048 \
  --steps 20 \
  -o output.png
```

### CPU Offload

Keep large components on CPU to reduce GPU memory pressure.

```bash
# Run text encoder on CPU
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --keep-text-encoder-on-cpu \
  --vae-tiling \
  -p "a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 20 \
  -o output.png

# Keep model weights on CPU and limit compute graph VRAM
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --offload-to-cpu \
  --max-vram 8 \
  --vae-tiling \
  -p "a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 20 \
  -o output.png
```

### Low-Memory Example

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/sd3 \
  --type q4_k \
  --no-t5 \
  --vae-tiling \
  --keep-text-encoder-on-cpu \
  -p "a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 20 \
  -o output.png
```

---

## Parallel Inference

### CFG Parallelism

CFG parallelism splits conditional and unconditional branches across multiple GPUs.

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/model \
  -p "a cinematic city skyline at night" \
  -W 1024 -H 1024 \
  --steps 20 \
  --cfg-scale 5.0 \
  --devices 0,1 \
  --cfg-size 2 \
  -o output.png
```

Equivalent option:

```bash
--cfg-parallel-size 2
```

### Sequence Parallelism

Sequence parallelism splits the token sequence across multiple GPUs.

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/model \
  -p "a cinematic city skyline at night" \
  -W 2048 -H 2048 \
  --steps 20 \
  --devices 0,1 \
  --sp-size 2 \
  -o output.png
```

Graph-cut profiling:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  -p "a cinematic photo of a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 6 \
  --devices 0,1 \
  --sp-size 2 \
  --profile-graph-cuts \
  --profile-graph-cuts-top 20 \
  -o output.png
```

> SP is experimental. It is recommended for high-resolution, large-token, or video workloads. For small FLUX workloads such as 1024×1024, batch=1, and low-step inference, SP may be slower than single-GPU execution.

---

## Cache-Based Reuse

edge-dit.cpp exposes a unified cache runtime for experimenting with timestep and block-level reuse.

Supported cache modes:

```text
off, easycache, ucache, dbcache, taylorseer, cache-dit, magcache, dicache
```

### DBCache-lite Example

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  -p "a cinematic photo of a glass teapot on a wooden table" \
  -W 1024 -H 1024 \
  --steps 20 \
  --cache dbcache \
  --cache-fn-blocks 8 \
  --cache-bn-blocks 0 \
  --cache-residual-threshold 0.08 \
  --cache-warmup-steps 4 \
  -o output_cache.png
```

### TaylorSeer Example

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  -p "a cinematic photo of a glass teapot on a wooden table" \
  -W 1024 -H 1024 \
  --steps 20 \
  --cache taylorseer \
  --cache-taylor-order 1 \
  --cache-taylor-skip 1 \
  -o output_taylorseer.png
```

### Static Step Computation Mask

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  -p "a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 20 \
  --cache cache-dit \
  --cache-scm-mask 1,0,0,1,0,0,1 \
  --cache-static-scm \
  -o output_scm.png
```

Cache methods are still experimental. For v0.1-alpha, **DBCache-lite** is the main stable target for speed-quality evaluation.

---

## Optimization Feature Matrix

| Feature | Status | Notes |
|---|---|---|
| CUDA BF16/F16 execution | Stable | Main optimized path |
| cuDNN SDPA fast attention | Stable / model-dependent | Enabled by CUDA build when cuDNN is available |
| Flash attention toggle | Stable | `--flash-attention` / `--no-flash-attention` |
| CFG parallelism | Stable | Useful for CFG-heavy workloads |
| Sequence parallelism | Experimental | Recommended for high-resolution or video workloads |
| Graph-cut profiling | Stable | Used to diagnose SP communication and segmentation |
| DBCache-lite | Experimental | Main v0.1-alpha cache target |
| TaylorSeer / CacheDiT / others | Experimental | Runtime interface available; tuning ongoing |
| Online quantization | Experimental | Q8/Q4 and mixed tensor rules |
| VAE tiling | Stable | Reduces peak memory |
| CPU offload | Experimental | Useful for low-memory configurations |
| Video generation | Experimental full pipeline | Wan T2V supported; optimization ongoing |
| Video VAE chunking | In progress | Targeting Wan video runtime |

---

## HTTP Server

`ed-server` is a small HTTP wrapper around the public edge-dit C API. It keeps one model context loaded in-process and serializes generation calls through that context.

Start server:

```bash
./build-cuda/bin/ed-server \
  --backend cuda \
  --model /path/to/flux-dev \
  --host 127.0.0.1 \
  --port 8080 \
  -W 1024 \
  -H 1024 \
  --steps 20 \
  --guidance 3.5
```

Canonical endpoints:

```text
GET  /ed/v1/health
GET  /ed/v1/models
GET  /ed/v1/capabilities
POST /ed/v1/images/generations
```

Generate an image:

```bash
curl -s http://127.0.0.1:8080/ed/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cinematic photo of a glass teapot on a wooden table, soft morning light",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "seed": 0,
    "distilled_guidance": 3.5,
    "cfg_scale": 1.0
  }' | jq -r '.data[0].b64_png' | base64 -d > output.png
```

The response includes elapsed time, resolved generation parameters, and one base64-encoded PNG per generated image.

---

## C API

The public C API is defined in:

```text
include/edge-dit.h
```

Minimal usage pattern:

```c
#include "edge-dit.h"

ed_context_params_t ctx_params;
ed_context_params_init(&ctx_params);
ctx_params.model_path = "/path/to/flux-dev";
ctx_params.backend = "cuda";

ed_context_t * ctx = ed_create_context(&ctx_params);
if (!ctx) {
    // handle error
}

ed_image_generation_params_t params;
ed_image_generation_params_init(&params);
params.prompt = "a cat wearing sunglasses";
params.width = 1024;
params.height = 1024;
params.sample_params.steps = 20;
params.sample_params.seed = 0;

ed_image_batch_t output;
ed_status_t status = ed_generate_image(ctx, &params, &output);
if (status != ED_STATUS_OK) {
    const char * err = ed_get_last_error(ctx);
    // handle error
}

ed_free_image_batch(&output);
ed_free_context(ctx);
```

Main API surface:

```c
ed_context_t * ed_create_context(const ed_context_params_t * params);
void ed_free_context(ed_context_t * ctx);

ed_status_t ed_generate_image(
    ed_context_t * ctx,
    const ed_image_generation_params_t * params,
    ed_image_batch_t * output
);

ed_status_t ed_generate_video(
    ed_context_t * ctx,
    const ed_video_generation_params_t * params,
    ed_video_t * output
);

const char * ed_get_last_error(const ed_context_t * ctx);
bool ed_context_supports_image(const ed_context_t * ctx);
bool ed_context_supports_video(const ed_context_t * ctx);
```

---

## Python Binding

Python binding lives under:

```text
bindings/python/
```

Example scripts:

```text
bindings/python/examples/basic_txt2img.py
bindings/python/examples/basic_txt2vid.py
bindings/python/examples/configured_txt2img.py
bindings/python/examples/configured_txt2vid.py
bindings/python/examples/server_v2_smoke.py
```

The Python binding is useful for smoke tests, local integration, and quick prototyping. For low-level deployment and performance evaluation, the C API and CLI remain the primary interfaces.

---

## Frontend Console

A development console is available under:

```text
bindings/python/frontend/server_v2-console/
```

It provides local runtime profiles for:

```text
flux-dev
flux-kontext
qwen-image
qwen-image-edit
sd3-medium
wan-t2v
```

This frontend is intended for local development and demonstration. It is not yet part of the stable v0.1-alpha API contract.

---

## Benchmarking

Current benchmark-related scripts:

```text
scripts/benchmark_cache.sh
scripts/benchmark_flux_cache.py
scripts/run_parallel_collective_test.py
scripts/test_parallel.sh
```

### Cache Benchmark

```bash
bash scripts/benchmark_cache.sh
```

or:

```bash
python scripts/benchmark_flux_cache.py \
  --model /path/to/flux-dev \
  --backend cuda \
  --width 1024 \
  --height 1024 \
  --steps 20
```

### Parallel Collective Test

```bash
python scripts/run_parallel_collective_test.py
```

### SP / Communication Tests

```bash
bash scripts/test_parallel.sh
```

### Manual Model Timing

For now, model-level timing can be collected directly from `ed-cli` logs:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  -p "a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 6 \
  --seed 0 \
  -o output.png
```

For SP profiling:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  -p "a cat wearing sunglasses" \
  -W 1024 -H 1024 \
  --steps 6 \
  --devices 0,1 \
  --sp-size 2 \
  --profile-graph-cuts \
  --profile-graph-cuts-top 20 \
  -o output_sp.png
```

Additional model latency, SP, cache, memory, and video benchmark runners are
being consolidated under `scripts/` and `tools/`.

---

## Build Details

### CUDA

CUDA build enables the ggml CUDA backend and edge CUDA helper kernels. cuDNN
SDPA fast attention is enabled when a compatible cuDNN installation is found via
`CUDNN_ROOT` or the current Python / conda environment. Automatic user-level
cuDNN wheel installation is opt-in.

```bash
bash ./scripts/build_cuda.sh
```

Use an existing cuDNN installation:

```bash
CUDNN_ROOT=/path/to/nvidia/cudnn bash ./scripts/build_cuda.sh
```

Allow the build script to install user-level NVIDIA cuDNN CUDA 12 wheels when
cuDNN is not already available:

```bash
ED_INSTALL_CUDNN=ON bash ./scripts/build_cuda.sh
```

Disable cuDNN SDPA:

```bash
ED_ENABLE_CUDNN_SDPA=OFF bash ./scripts/build_cuda.sh
```

### CPU

```bash
bash ./scripts/build_cpu.sh
```

CPU is mainly intended for build validation, smoke tests, fallback operators, CPU offload, and selected low-speed inference.

### Vulkan

```bash
bash ./scripts/build_vulkan.sh
```

If the Vulkan SDK is installed in a non-standard path:

```bash
VULKAN_SDK=/path/to/vulkan-sdk bash ./scripts/build_vulkan.sh
```

Vulkan support is experimental.

### Metal

```bash
bash ./scripts/build_metal.sh
```

Metal build is available on macOS only and remains experimental.

---

## Common CLI Options

<details>
<summary>Model loading options</summary>

| Option | Description |
|---|---|
| `--backend auto\|cpu\|cuda\|vulkan\|metal\|gpu` | Runtime backend |
| `--model <dir>` | Diffusers-style model directory |
| `--diffusion-model <path>` | Standalone DiT transformer weights |
| `--clip_l <path>` | CLIP-L text encoder weights |
| `--clip_g <path>` | CLIP-G text encoder weights |
| `--t5xxl <path>` | T5XXL text encoder weights |
| `--vae <path>` | VAE weights |
| `--no-t5` | Skip T5XXL text encoder for SD3 low-memory runs |

</details>

<details>
<summary>Generation options</summary>

| Option | Description |
|---|---|
| `-p, --prompt <text>` | Prompt text |
| `--negative-prompt <text>` | Negative prompt |
| `-i, --image <path>` | Input / reference image for image-edit models |
| `-W, --width <int>` | Output width |
| `-H, --height <int>` | Output height |
| `--steps <int>` | Sampling steps |
| `-s, --seed <int64>` | Random seed |
| `--guidance <float>` | FLUX distilled guidance |
| `--cfg-scale <float>` | Classifier-free guidance scale |
| `--flow-shift <float>` | Flow scheduler shift |
| `--qwen-image-zero-cond-t` | Enable Qwen-Image zero-cond timestep behavior |
| `-o, --output <path>` | Output image / video path |

</details>

<details>
<summary>Video options</summary>

| Option | Description |
|---|---|
| `--video` | Generate video instead of image |
| `--video-format <fmt>` | `auto`, `avi`, `mp4`, `mov`, `mkv`, `webm` |
| `--frames <int>` | Video frame count |
| `--fps <int>` | Video FPS |

</details>

<details>
<summary>Memory and quantization options</summary>

| Option | Description |
|---|---|
| `--type <dtype>` | Weight type / on-the-fly quantization |
| `--tensor-type-rules <csv>` | Per-tensor dtype override rules |
| `--vae-tiling` | Enable VAE tiled decode |
| `--vae-tile-size <float>` | VAE tile relative size |
| `--offload-to-cpu` | Keep model weights on CPU and copy to GPU per compute |
| `--keep-text-encoder-on-cpu` | Run text encoder on CPU |
| `--keep-vae-on-cpu` | Run VAE on CPU |
| `--max-vram <GB>` | Limit VRAM usage for compute graphs |

</details>

<details>
<summary>Parallelism and profiling options</summary>

| Option | Description |
|---|---|
| `--devices <csv>` | GPU device list, e.g. `0,1` |
| `--cfg-size <n>` | Alias for CFG parallel size |
| `--cfg-parallel-size <n>` | Split CFG cond/uncond branches across GPUs |
| `--sp-size <n>` | Sequence parallel size |
| `--tp-size <n>` | Reserved tensor parallel size; currently use `1` |
| `--profile-graph-cuts` | Print graph-cut compute/communication timing summary |
| `--profile-graph-cuts-top <n>` | Print top graph-cut buckets |
| `--profile-graph-cuts-all-ranks` | Print graph-cut profiling for all ranks |

</details>

<details>
<summary>Cache options</summary>

| Option | Description |
|---|---|
| `--cache <mode>` | Cache mode |
| `--cache-threshold <float>` | EasyCache / UCache reuse threshold |
| `--cache-start <float>` | Cache active window start percentage |
| `--cache-end <float>` | Cache active window end percentage |
| `--cache-fn-blocks <int>` | DBCache / CacheDiT front compute blocks |
| `--cache-bn-blocks <int>` | DBCache / CacheDiT back compute blocks |
| `--cache-residual-threshold <float>` | DBCache residual threshold |
| `--cache-warmup-steps <int>` | Cache warmup steps |
| `--cache-taylor-order <int>` | TaylorSeer derivative order |
| `--cache-taylor-skip <int>` | TaylorSeer skip interval |
| `--cache-scm-mask <csv>` | Static step computation mask |
| `--cache-static-scm` | Use static SCM policy |
| `--cache-dynamic-scm` | Use dynamic SCM policy |

</details>

---

## Repository Layout

```text
edge-dit.cpp/
├── include/
│   └── edge-dit.h                  # public C API
├── examples/
│   ├── cli/                        # ed-cli
│   └── server/                     # ed-server
├── src/
│   ├── core/                       # runtime / backend / graph / memory
│   ├── dit_models/                 # model families and pipelines
│   └── utils/
├── bindings/
│   └── python/                     # optional Python binding and frontend console
├── scripts/
│   ├── build_cuda.sh
│   ├── build_cpu.sh
│   ├── build_vulkan.sh
│   ├── build_metal.sh
│   └── benchmark / parallel scripts
├── tests/
├── tools/
└── third_party/
```

---

## Known Limitations

- v0.1-alpha uses CUDA as the first-class backend. CPU is mainly used for build, smoke tests, fallback, and offload.
- Metal and Vulkan are experimental and are not used for official performance numbers yet.
- Sequence parallelism is not always faster. For small FLUX workloads, single-GPU execution can be faster than 2-GPU SP.
- Cache policies are still experimental. DBCache-lite is the main stable cache target for the first release.
- Wan video support is functional, but video-specific memory/runtime optimization is still ongoing.
- Tensor parallelism is reserved but not implemented yet; use `--tp-size 1`.
- The first-stage model scope is limited to SD3/SD3.5, FLUX, Qwen-Image, Qwen-Image-Edit, FLUX-Kontext, and Wan.
- The public benchmark runner is still being consolidated. Current results should be treated as early internal measurements.

---

## Roadmap

### v0.1-alpha

- CUDA first-class inference.
- SD3 / SD3.5, FLUX, Qwen-Image, Qwen-Image-Edit, FLUX-Kontext, and Wan first-stage support.
- C API, CLI, HTTP server, and Python binding.
- Basic cache runtime.
- Basic CFG parallelism and sequence parallelism.
- Wan T2V pipeline.
- Initial benchmark and profiling scripts.

### v0.2

- Unified benchmark runner for model, backend, SP, cache, memory, and video workloads.
- Improved Flux SP communication planning and graph-cut reduction.
- More stable DBCache-lite speed-quality tuning.
- Stronger Wan video VAE chunking / tiling.
- Q8 / Q4 GGUF export and loading path.
- Python binding stabilization.
- Model zoo metadata.

### Future

- Z-Image, Sana, LTX-Video, HunyuanVideo, Wan VACE, and other next-generation DiT models.
- Metal optimization.
- Vulkan and Android roadmap.
- ComfyUI integration.
- Benchmark leaderboard.
- Technical report / project paper.

---

## Contributing

Contributions are welcome, especially in the following areas:

- model loader and tensor name mapping;
- CUDA kernel profiling and optimization;
- attention backend integration;
- cache policy implementation and benchmarking;
- video-specific memory optimization;
- CPU / Metal / Vulkan backend validation;
- documentation and examples.

Before submitting a pull request, please make sure that:

```bash
git submodule update --init --recursive
bash ./scripts/build_cpu.sh
./build-cpu/bin/ed-cli --help
```

CUDA-related PRs should also pass:

```bash
bash ./scripts/build_cuda.sh
./build-cuda/bin/ed-cli --help
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development and validation
guide.

---

## Acknowledgements

edge-dit.cpp is built on top of and inspired by the broader open-source diffusion and inference ecosystem.

We thank the communities and projects behind:

- ggml / llama.cpp;
- Stable Diffusion 3 / SD3.5;
- FLUX;
- Qwen-Image;
- Wan;
- Diffusers;
- xDiT;
- ParaAttention;
- FlashAttention;
- cuDNN SDPA;
- other open-source DiT, image, editing, and video generation projects.

---

## Citation

If you find edge-dit.cpp useful in your research or applications, please consider citing:

```bibtex
@misc{edgeditcpp2026,
  title        = {edge-dit.cpp: A DiT-first C/C++ Inference Engine for Edge Image, Editing, and Video Generation},
  author       = {edge-dit.cpp Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/yiming-l21/edge-dit.cpp}},
  note         = {GitHub repository}
}
```

This citation points to the repository while a technical report is being prepared.

---

## License

edge-dit.cpp is licensed under the [Apache License 2.0](LICENSE).

Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license files in
`third_party/`.

Model weights are not distributed with this repository. Users are responsible
for obtaining model weights separately and complying with each model provider's
license and usage policy.

---

## Contact

For questions, bug reports, and feature requests, please use GitHub Issues after the repository is public.

---

<div align="center">

**edge-dit.cpp — DiT-first inference for local image, editing, and video generation**

</div>
