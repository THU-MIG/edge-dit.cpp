<p align="center">
  <img src="assets/logo.png" alt="edge-dit.cpp logo" width="100%">
</p>

<h1>edge-dit.cpp</h1>

<p align="center">
  <strong>A lightweight native C/C++ runtime for Diffusion Transformer inference
  on resource-constrained devices and local deployment environments.</strong>
</p>

[![Status](https://img.shields.io/badge/status-public_preview-orange)](#development-status)
[![Backend](https://img.shields.io/badge/backend-CUDA--first-blue)](#backend-support)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

edge-dit.cpp is a lightweight, DiT-first C/C++ inference engine designed for local, edge, and resource-constrained deployment. Built on ggml, it provides a unified runtime for image generation, image editing, and video generation, with explicit control over model loading, memory usage, graph execution, and backend selection.

## Latest News

- **2026-07-27:** 🚀 Added optional **SageAttention** (INT8-QK + F16-PV) for SD3 and Wan self-attention (`ED_SAGE_ATTN=1`): a loss-free, opt-in attention speedup (~5–6% on SD3). See [attention optimization](docs/optimization/graph-and-operator-optimization.md#quantized-attention-sageattention-optional).
- **2026-07-24:** 🚀 Added **activation-calibrated imatrix quantization** to `ed-convert` (`--imatrix`): an offline calibration pass weights low-bit (q4_k) quantization toward the most important input channels. See [CLI usage](docs/cli.md#activation-calibrated-imatrix-quantization).
- **2026-07-23:** 🚀 Added **`ed-convert`** for **offline weight quantization** — convert any supported model once into a portable, self-identifying pre-quantized GGUF, then load it directly and skip on-load quantization. Works across image, editing, and video models. See [CLI usage](docs/cli.md#pre-quantized-gguf-with-ed-convert).
- **2026-07-11:** 🚀 **edge-dit.cpp v0.1.0-alpha** enters **public preview**.
- **2026-07-08:** 🚀 Added the **managed development console** for the
  **Python job server**.
- **2026-07-07:** 🚀 Added **FLUX.1-Kontext** image generation and editing
  support.
- **2026-07-05:** 🚀 Added **Python bindings** and examples for native runtime
  integration.
- **2026-07-02:** 🚀 Added **Qwen-Image-Edit** image editing support.
- **2026-05-27:** 🚀 Added **Qwen-Image**, **native C API**, **CLI**, and
  **HTTP server** support.
- **2026-05-26:** 🚀 Added native **SD3** and **Wan 2.x** video pipeline support.
- **2026-05-25:** 🚀 Added the first **FLUX.1-dev** text-to-image backend.

## Features

- **Lightweight native DiT runtime**
  - Pure C/C++ inference built on [ggml-org/ggml](https://github.com/ggml-org/ggml)
  - No Python or PyTorch required at runtime
  - Explicit tensor, graph, memory, and device control
  - Designed for local and resource-constrained deployment

- **DiT engine abstraction and architecture**
  - Loader layer for Diffusers directories, standalone components, safetensors shards, and GGUF
  - Runtime layer for tensor storage, graph execution, memory management, devices, and backend dispatch
  - Model layer for architecture-specific DiT blocks, conditioning, and output adapters
  - Pipeline layer for image generation, image editing, and video generation
  - Shared C API, CLI, HTTP server, and Python interfaces across model families

- **System-level optimization for efficient DiT inference**
  - **[Model representation and precision](docs/optimization/model-representation-and-precision.md)**
    - Quantization, mixed precision, and per-tensor dtype control
    - Offline quantization to portable pre-quantized GGUF (\`ed-convert\`)
  - **[Memory-efficient execution](docs/optimization/memory-efficient-execution.md)**
    - CPU offload, graph VRAM control, VAE tiling, and component placement
  - **[Graph and operator optimization](docs/optimization/graph-and-operator-optimization.md)**
    - cuDNN SDPA, DiT-specific CUDA operators, and tensor-layout optimization
  - **[Computation reuse](docs/optimization/computation-reuse.md)**
    - Timestep- and block-level cache reuse with output, feature, and probe policies
  - **[Parallel execution](docs/optimization/parallel-execution.md)**
    - CFG parallelism, sequence parallelism, and NCCL/MPI multi-worker execution

## Supported Models

The public preview focuses on the model families below. Some source files
contain experimental model scaffolding beyond this table; those are not part of
the current public support commitment unless documented in
[Supported Models](docs/models.md).

| Model family | Task | Status |
|---|---|---|
| SD3 / SD3.5 | Text-to-image | Public preview |
| FLUX.1 | Text-to-image | Public preview |
| FLUX.1-Kontext | Image editing / reference-guided generation | Public preview |
| Qwen-Image | Text-to-image | Public preview |
| Qwen-Image-Edit | Image editing | Public preview |
| Wan 2.x | Video generation | Public preview |

See [Supported Models](docs/models.md) for formats, backend coverage,
model-specific options, examples, and known limitations.

## Backend Support

| Backend | Status | Notes |
|---|---|---|
| CUDA | First-class | Primary backend for optimized inference |
| CPU | Functional | Portable execution, fallback, and offload |
| Metal | Experimental | Early macOS support |
| Vulkan | Experimental | Early cross-vendor GPU support |

For dependencies, build profiles, and platform-specific instructions, see
[Build and installation](docs/build.md).

## Performance

The README main-table snapshot below was measured on 2026-07-13 with the CUDA
`performance` profile on a local NVIDIA H200 node. Full benchmark configs,
commands, environment metadata, raw result roots, and interpretation notes are
available in [Performance and benchmarks](docs/performance.md).

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

## Open-Source Interfaces

edge-dit.cpp exposes the same runtime through several public integration
surfaces:

| Interface | Entry point | Documentation |
|---|---|---|
| CLI | `ed-cli`, `ed-sample` | [Command line usage](docs/cli.md) |
| C API | `include/edge-dit.h` | [API and bindings](docs/api.md#c-api) |
| Native HTTP server | `ed-server` | [API and bindings](docs/api.md#native-http-server) |
| Python bindings | `edge_dit` package | [API and bindings](docs/api.md#python-bindings) |
| Python job server / console | `edge_dit.server_v2`, managed console | [API and bindings](docs/api.md#python-server-v2) |

The v0.x API, ABI, CLI flags, and HTTP schemas are public but not yet stable.

## Quick Start

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/THU-MIG/edge-dit.cpp
cd edge-dit.cpp
```

If the repository was cloned without submodules:

```bash
bash scripts/bootstrap.sh
```

When updating an existing checkout, pull the superproject and submodule state
together:

```bash
git pull --recurse-submodules
git submodule update --init --recursive
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

For full build options and command-line usage, see:

- [Build and installation](docs/build.md)
- [Command line usage](docs/cli.md)

## Contributors

Thank you to everyone who has contributed to edge-dit.cpp.

<a href="https://github.com/THU-MIG/edge-dit.cpp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=THU-MIG/edge-dit.cpp" alt="edge-dit.cpp contributors">
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
- [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) and
  [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) for Wan video model
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
