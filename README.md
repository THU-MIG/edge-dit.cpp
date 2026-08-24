<p align="center">
  <img src="assets/logo.png" alt="edge-dit.cpp logo" width="100%">
</p>

<h1 align="center">edge-dit.cpp</h1>

<p align="center">
  <strong>A lightweight C/C++ inference engine for efficient Diffusion Transformer (DiT)
  inference on local and resource-constrained devices.</strong>
</p>

[![Status](https://img.shields.io/badge/status-alpha-orange)](#latest-news)
[![Backend](https://img.shields.io/badge/backend-CUDA--first-blue)](#backend-support)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

**edge-dit.cpp** is an open-source, DiT-first C/C++ inference engine for efficient
**Diffusion Transformer (DiT)** inference. Built on **ggml**, it provides a unified
runtime for image generation, image editing, and video generation across local,
edge, and resource-constrained deployment environments.

It supports major DiT model families including **FLUX.1, FLUX.2 [klein] 4B,
Stable Diffusion 3/3.5, Qwen-Image, Wan, and MiniMax-H3**, with explicit control
over model loading, memory usage, graph execution, quantization, device
placement, and backend selection.

## Features

- **Lightweight native DiT runtime**
  - Pure **C/C++** inference built on [ggml](https://github.com/ggml-org/ggml) — **no Python or PyTorch at runtime**
  - Explicit control over tensors, graph execution, memory, and device placement
  - **Multi-backend**: **CUDA** (first-class), **CPU** (portable, optional oneDNN **bf16 AMX** matmul), **Vulkan** (cross-vendor GPU), **Metal** (experimental)
  - Loads **Diffusers** directories, standalone components, **safetensors** (+ shard index), and **GGUF**

- **Unified across tasks and model families**
  - **Text-to-image**, **image editing**, and **video generation** in one runtime
  - SD3/SD3.5, FLUX.1, FLUX.1-Kontext, FLUX.2 [klein] 4B, Qwen-Image,
    Qwen-Image-Edit, Wan 2.1, and MiniMax-H3 video+audio
  - **Few-step distilled models** auto-detected — Turbo / Lightning / schnell default to a **4–8 step** schedule
  - Shared **C API, CLI, HTTP server, and Python** interfaces across every family

- **Fits large models into limited VRAM**
  - **`--auto-fit`** — one flag auto-picks DiT **quantization** (`q8_0`→`q4_K`) and per-component placement; on single-device CUDA, pairing it with `--max-vram` also enforces an allocation ceiling
  - **Layered offload** — streams the diffusion transformer **one block at a time** (async double-buffered on CUDA), so 20 GB+ models run on a 24 GB or smaller card
  - **Per-component offload** (`--dit-offload` / `--text-encoder-offload` / `--vae-offload`) and **VAE tiling**
  - **`ed-convert`** — offline weight quantization to a **portable pre-quantized GGUF** (skips per-load CPU quantization), including independently loadable CLIP/T5/LLM components, per-tensor dtype rules, and activation-calibrated **imatrix**

- **System-level optimization for efficient DiT inference**
  - **[Model representation and precision](docs/optimization/model-representation-and-precision.md)** — quantization, mixed precision, per-tensor dtype, offline GGUF (`ed-convert`)
  - **[Memory-efficient execution](docs/optimization/memory-efficient-execution.md)** — CPU offload, layered offload, graph VRAM budget, VAE tiling, component placement
  - **[Graph and operator optimization](docs/optimization/graph-and-operator-optimization.md)** — cuDNN SDPA, DiT-specific CUDA operators, tensor-layout optimization
  - **[Computation reuse](docs/optimization/computation-reuse.md)** — timestep- and block-level cache reuse
  - **[Few-step distilled models](docs/optimization/few-step-distilled-models.md)** — automatic Turbo/Lightning/schnell scheduling
  - **[Parallel execution](docs/optimization/parallel-execution.md)** — CFG and sequence parallelism, NCCL/MPI multi-worker execution

## Latest News

- **2026-08-14:** 🚀 Added **MiniMax-H3 FL2VA and Ref2VA video+audio generation** with image, video, embedded/paired audio, and mixed references; full or pruned BF16 DiTs, persistent Q8_0 conversion, Q4_K_M weights, and automatic VRAM fitting are supported. MiniMax-H3 output must contain at least 22 frames and satisfy `17k+5` ([usage and H200 results](docs/minimax-h3.md)).
- **2026-08-05:** 🚀 Completed the **RTX 4090 (24 GB) benchmark** — full cross-system speed / VRAM / image-quality across text-to-image, editing, and video ([results](docs/performance-4090.md)).
- **2026-07-30:** 🚀 Added **per-component offload** (`--dit-offload` / `--text-encoder-offload` / `--vae-offload`), unifying all offload paths on one semantics.
- **2026-07-29:** 🚀 Added **`--auto-fit`** — one flag picks DiT quantization and per-component placement; `--auto-fit --max-vram <GB>` enforces the requested single-device CUDA allocation ceiling.
- **2026-07-27:** 🚀 Added **few-step distilled** auto-detection (Turbo/Lightning/schnell → 4–8 steps) and optional **SageAttention** for SD3/Wan.
- **2026-07-23:** 🚀 Added **`ed-convert`** for offline weight quantization to portable pre-quantized GGUF (with activation-calibrated `--imatrix`).
- **2026-07-11:** 🚀 **edge-dit.cpp v0.1.0-alpha** enters **public preview**.
- **2026-07-02:** 🚀 Added **FLUX.1-Kontext** and **Qwen-Image-Edit** image editing.
- **2026-05-26:** 🚀 First pipelines — **FLUX.1-dev**, **SD3**, **Qwen-Image**, **Wan 2.1**, plus **C API / CLI / HTTP server / Python bindings**.

## Supported Models

This release focuses on the model families below. Most ship as a base
checkpoint plus a **few-step distilled variant**; some source files contain
experimental model scaffolding beyond this table, and those are not part of
the current support commitment unless documented in
[Supported Models](docs/models.md).

| Model family | Task | Base checkpoint | Distilled variant (few-step) | Status |
|---|---|---|---|---|
| **SD3 / SD3.5** | Text-to-image | `stabilityai/stable-diffusion-3-medium` | SD3.5-medium-turbo | Supported |
| **FLUX.1** | Text-to-image | `black-forest-labs/FLUX.1-dev` | FLUX.1-schnell | Supported |
| **FLUX.1-Kontext** | Image editing / reference-guided | `black-forest-labs/FLUX.1-Kontext-dev` | Kontext Lightning | Supported |
| **FLUX.2 [klein] 4B** | Text-to-image / image editing | `black-forest-labs/FLUX.2-klein-4B` | Native FLUX.2 [klein] 4B checkpoint | Supported |
| **Qwen-Image** | Text-to-image | `Qwen/Qwen-Image` | Qwen-Image Lightning *(LoRA)* | Supported |
| **Qwen-Image-Edit** | Image editing | `Qwen/Qwen-Image-Edit` | Qwen-Image-Edit Lightning *(LoRA)* | Supported |
| **Wan 2.1** | Video generation | `Wan-AI/Wan2.1-T2V-1.3B` (and 14B) | Wan2.1-T2V-1.3B Distill | Supported (Vulkan still optimizing) |
| **MiniMax-H3** | Video + audio generation | FL2VA / Ref2VA component checkpoints | — | Supported (CUDA validated) |

Distilled checkpoints load through the same pipeline as the base model and are
**auto-detected** (default **4–8 steps** when `--steps` is unset). Most ship as
drop-in full weights; the two Qwen-Image Lightning variants ship as **LoRA
adapters** and must be merged into the base first (`scripts/merge_qwen_lora.py`).
See [Supported Models](docs/models.md) for exact HuggingFace repos, formats,
per-variant run commands, backend coverage, and known limitations.

## Backend Support

| Backend | Status | Notes |
|---|---|---|
| CUDA | First-class | Primary backend for optimized inference |
| CPU | Functional | Portable execution and fallback; optional oneDNN bf16 AMX matmul |
| Metal | Experimental | Early macOS support |
| Vulkan | Functional | Cross-vendor GPU; base model families validated, ~1.3x slower than CUDA |

For dependencies, build profiles, and platform-specific instructions, see
[Build and installation](docs/build.md).

## Performance

The first snapshot below was measured on **RTX 4090 (24 GB)** with the CUDA
`performance` profile. Compare inference speed with **DiT sampling ms**
(cross-system-comparable); **4090 end-to-end excludes model load** (load-once
boundary). Rows compare systems **at matched precision** — 8-bit weight-only
(edge/sd.cpp `q8_0`, Diffusers `w8`); models that don't fit 24 GB resident use an
offload tier (noted in Precision) and are compared within that tier. sd.cpp
quantized tiers fold on-the-fly conversion into the timing and are inflated.
Full 4090 configs, all quant tiers, VRAM and image-quality metrics are in
[Performance and benchmarks (RTX 4090)](docs/performance-4090.md).

| Task | Model | System | Precision / tier | DiT sampling (ms) | E2E excl. load (ms) | Peak VRAM (MiB) |
|---|---|---|---|---:|---:|---:|
| t2i | FLUX.1-dev | edge-dit.cpp | q8_0 | **10569** | **11196** | 19112 |
| | | Diffusers | w8 | 13190 | 14139 | 23866 |
| | | stable-diffusion.cpp | q8_0 | 17797 | 22194 | **18559** |
| t2i | SD3 Medium | edge-dit.cpp | q8_0 | **3434** | 4131 | **9147** |
| | | Diffusers | w8 | **3411** | **3923** | 18172 |
| | | stable-diffusion.cpp | q8_0 | 5087 | 9975 | **9106** |
| t2i | Qwen-Image | edge-dit.cpp | q8_0 (auto-allocate) | 129887 | 131534 | **17019** |
| | | Diffusers | w8 (full-offload) | **54696** | 72270 | 21264 |
| | | stable-diffusion.cpp | q8_0 (full-offload) | 89720 | 102539 | 18799 |
| edit | FLUX.1-Kontext | edge-dit.cpp | q8_0 | **24534** | **25510** | 20111 |
| | | Diffusers | w8 | 27945 | 28704 | 23868 |
| | | stable-diffusion.cpp | q8_0 | 39133 | 44415 | **19418** |
| video | Wan2.1-T2V-1.3B | edge-dit.cpp | q8_0 | **53964** | 59927 | 12176 |
| | | Diffusers | w8 | 56580 | **59598** | 19568 |
| | | stable-diffusion.cpp | q8_0 | 80383 | 111750 | **11308** |

Qwen-Image does not fit 24 GB resident on any runtime, so each row uses that
runtime's working offload tier (edge `q8` auto-allocate, Diffusers `w8`
full-offload, sd.cpp `q8` full-offload) — the budgets differ, so treat these as
per-runtime working points rather than a like-for-like speed ratio.

### H200 snapshot

The table below was measured on 2026-07-13 with the CUDA `performance` profile
on a local NVIDIA H200 node; its `Median`/`P90` are **load-inclusive end-to-end**
latency (a different measurement boundary from the 4090 table above). Full H200
configs and notes are in [Performance and benchmarks](docs/performance-H200.md).

| Model | System | Load (s) | Median (s) | P90 (s) | Peak VRAM (MiB) |
|---|---|---:|---:|---:|---:|
| FLUX.1-dev | edge-dit.cpp | 6.645 | 10.784 | 10.861 | 38341 |
| | Diffusers | 14.531 | 10.040 | 10.048 | 37711 |
| | stable-diffusion.cpp | 1.333 | 30.371 | 30.379 | 40331 |
| Stable Diffusion 3 Medium | edge-dit.cpp | 5.840 | 4.003 | 4.049 | 20833 |
| | Diffusers | 11.244 | 3.376 | 3.381 | 20283 |
| | stable-diffusion.cpp | 1.457 | 10.740 | 10.797 | 22997 |
| Qwen-Image | edge-dit.cpp | 11.621 | 10.697 | 10.736 | 59725 |
| | Diffusers | 25.220 | 9.558 | 9.565 | 60935 |
| | stable-diffusion.cpp | 1.782 | 62.671 | 62.728 | 61879 |

Load time follows each runtime's reported initialization boundary and may
reflect different weight materialization or memory-mapping strategies.
Generation latency is the primary cross-runtime performance metric.

MiniMax-H3 has a separate 124-frame H200 benchmark because it generates video
and audio rather than one image. The table below keeps that result in the same
Performance section, using the same resident-component BF16 setup.

| Workflow | Task | System | Generate | Peak VRAM (MiB) |
|---|---|---|---:|---:|
| FL2VA | Text | edge-dit.cpp | 51.396s | 125,051 |
| | | Diffusers | 53.986s | 128,801 |
| | First frame | edge-dit.cpp | 54.810s | 125,353 |
| | | Diffusers | 57.817s | 129,957 |
| | Last frame | edge-dit.cpp | 55.323s | 125,351 |
| | | Diffusers | 57.793s | 129,957 |
| | First + last frames | edge-dit.cpp | 58.747s | 125,573 |
| | | Diffusers | 61.405s | 130,587 |
| Ref2VA | Image | edge-dit.cpp | 126.915s | 130,445 |
| | | Diffusers | 136.656s | 139,253 |
| | MP4 video / video frames | edge-dit.cpp | 182.649s | 132,661 |
| | | Diffusers | 183.921s | 137,161 |
| | Video frames + paired audio | edge-dit.cpp | 182.667s | 132,663 |
| | | Diffusers | 185.035s | 137,183 |
| | Mixed references | edge-dit.cpp | 301.435s | 138,113 |
| | | Diffusers | 319.435s | 141,915 |

edge-dit.cpp is faster than Diffusers in all four FL2VA and all four Ref2VA
generation paths while using less peak VRAM. FL2VA text-to-video takes
`51.396s` versus `53.986s`; Ref2VA image and mixed-reference generation reach
`1.08x` and `1.06x` speedups. Full and pruned BF16 DiTs are supported directly,
and either can be converted once to persistent Q8_0 GGUF. `--auto-fit` has also
been validated down to a 24 GB VRAM budget. See [MiniMax-H3 usage and
performance](docs/minimax-h3.md) for inputs, weights, memory placement, and the
complete comparison.

## Open-Source Interfaces

edge-dit.cpp exposes the same runtime through several public integration
surfaces:

| Interface | Entry point | Documentation |
|---|---|---|
| CLI | `ed-cli`, `ed-sample` | [Command line usage](docs/cli.md) |
| C API | `include/edge-dit.h` | [API and bindings](docs/api.md#1-c-api) |
| Native HTTP server | `ed-server` | [API and bindings](docs/api.md#2-native-server) |
| Python bindings | `edge_dit` package | [Python setup and image/video tutorial](bindings/python/README.md) |
| Python job server / console | `edge_dit.server`, Python Server Console | [API and bindings](docs/api.md#4-python-server) |

The v0.x API, ABI, CLI flags, and HTTP schemas are public but not yet stable.

## Quick Start

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/THU-MIG/edge-dit.cpp
cd edge-dit.cpp
```

If the repository was cloned without submodules (for example from a source ZIP
archive), fetch them with either:

```bash
git submodule update --init --recursive   # fetch submodules directly
# or, equivalently, run the bootstrap helper (also verifies they populated):
bash scripts/bootstrap.sh
```

Build the default CUDA performance profile:

```bash
bash scripts/build_cuda.sh
```

Verify the installation:

```bash
./build-cuda/bin/ed-cli --help
```

Run FLUX text-to-image inference:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --output output.png
```

The default build uses the official `performance` profile. It enables the
optimized CUDA path and automatically handles user-space dependencies when
possible. For CI or dependency-limited environments, see the optional `minimal`
profile in [Build and installation](docs/build.md).

For a CPU build, run `bash scripts/build_cpu.sh`; it auto-enables oneDNN bf16
AMX matmul acceleration when `third_party/onednn` has been built into
`third_party/onednn/install`, and can be forced off with `ED_ONEDNN=0`.

For full build options (CPU, Metal, Vulkan, and updating an existing checkout)
and command-line usage, see:

- [Build and installation](docs/build.md)
- [Command line usage](docs/cli.md)

## Contributors

Thank you to everyone who has contributed to edge-dit.cpp.

<a href="https://github.com/THU-MIG/edge-dit.cpp/graphs/contributors">
  <img
    src="https://contrib.rocks/image?repo=THU-MIG/edge-dit.cpp&amp;v=2"
    alt="edge-dit.cpp contributors">
</a>

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) and
[Development](docs/development.md).

## Acknowledgements

Model ecosystems and native inference references:

- [Stability-AI/sd3.5](https://github.com/Stability-AI/sd3.5) for SD3/SD3.5
  reference material.
- [black-forest-labs/flux](https://github.com/black-forest-labs/flux) for
  FLUX.1 and FLUX.1-Kontext reference material.
- [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) for Qwen-Image and
  Qwen-Image-Edit reference material.
- [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) for Wan video model
  reference material.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) for
  native diffusion model implementation references.

Runtime, operator, and dependency foundations:

- [ggml](https://github.com/ggml-org/ggml) as the underlying tensor and graph
  runtime.
- [NVIDIA cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend) for
  attention and CNN operator support through cuDNN.
- [NVIDIA NCCL](https://github.com/NVIDIA/nccl) and
  [Open MPI](https://github.com/open-mpi/ompi) for distributed and multi-GPU
  runtime support.
- [nlohmann/json](https://github.com/nlohmann/json),
  [cpp-httplib](https://github.com/yhirose/cpp-httplib), and
  [stb](https://github.com/nothings/stb) for lightweight utility components.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency licenses.

## Citation

Technical report citation coming soon. For now, cite the repository:

```bibtex
@software{edge_dit_cpp,
  title  = {edge-dit.cpp: A Lightweight Native Runtime for Diffusion Transformers on Resource-Constrained Devices},
  author = {edge-dit.cpp contributors},
  url    = {https://github.com/THU-MIG/edge-dit.cpp},
  year   = {2026}
}
```

## License

edge-dit.cpp is released under the [Apache License 2.0](LICENSE).
Third-party components and model weights remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE](NOTICE).
