# edge-dit.cpp

A lightweight, DiT-first C/C++ inference runtime for local and
resource-constrained image, editing, and video generation.

[![Status](https://img.shields.io/badge/status-public_preview-orange)](#development-status)
[![Backend](https://img.shields.io/badge/backend-CUDA--first-blue)](#backend-support)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

edge-dit.cpp is a native runtime for diffusion-transformer inference. It uses
ggml backends, keeps a compact public C API, and exposes CLI, HTTP, and Python
entry points for research, benchmarking, and early integration work.

## Development Status

```text
Repository-ready: yes
Public preview-ready: yes
v0.1.0-alpha release sign-off: pending full CUDA performance validation
```

The project is in the v0.x series. Public API, ABI, CLI flags, model coverage,
and server semantics may change before a stable v1.0 release.

CUDA is the first-class backend and the path intended for official performance
evaluation. Full `performance` profile validation with CUDA, NCCL, MPI, and
cuDNN remains the final v0.1.0-alpha release gate.

## Latest News

- **2026-07:** v0.1.0-alpha release engineering baseline added: explicit CUDA
  build profiles, dependency prechecks, submodule bootstrap, source package
  metadata, version API, and repository hygiene checks.
- **2026-07:** Public preview scope is focused on SD3/SD3.5, FLUX,
  FLUX-Kontext, Qwen-Image, Qwen-Image-Edit, and Wan pipelines.
- **2026-07:** Cache runtime, graph-cut profiling, CFG parallelism, and
  sequence-parallel paths are available for targeted validation.
- **2026-07:** C API, CLI, native HTTP server, Python bindings, and a managed
  development console are available, with v0.x stability caveats.

## Features

- Native C/C++ runtime with ggml backend integration.
- DiT-first model and pipeline abstractions.
- Text-to-image, image editing, and text/video generation paths.
- CUDA, CPU, Metal, and Vulkan backend entry points.
- Diffusers-style model directories and component weight loading.
- Safetensors, safetensors shard indexes, and GGUF loading paths.
- On-load quantization, tensor type rules, VAE tiling, CPU offload, and VRAM
  limiting.
- CUDA fast paths including cuDNN SDPA, CUDA Norm, CUDA RoPE, CUDA Modulation,
  CFG parallelism, sequence parallelism, and cache reuse.
- Public C API, CLI tools, native HTTP server, Python bindings, Python
  `server_v2`, and a local frontend console.

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
| Wan 2.x | Video generation | Public preview, still being optimized |

See [Supported Models](docs/models.md) for formats, backend coverage,
model-specific options, examples, and known limitations.

## Backend Support

| Backend | Status | Notes |
|---|---|---|
| CUDA | First-class | Main target for optimized inference and performance validation |
| CPU | Functional / reference | Build validation, smoke tests, fallback, and CPU offload |
| Metal | Experimental | macOS-only ggml Metal backend path |
| Vulkan | Experimental | Requires Vulkan SDK / shader tooling |

For dependencies, build profiles, and platform-specific instructions, see
[Build and installation](docs/build.md).

## Quick Start

Clone with submodules:

```bash
git clone --recursive https://github.com/yiming-l21/edge-dit.cpp
cd edge-dit.cpp
```

If you already cloned without `--recursive`, initialize dependencies:

```bash
bash scripts/bootstrap.sh
```

Build the minimal CUDA profile for quick validation:

```bash
ED_BUILD_PROFILE=minimal bash scripts/build_cuda.sh
```

For official performance work, use the `performance` profile with CUDA, NCCL,
MPI, and cuDNN installed:

```bash
CUDA_HOME=/path/to/cuda \
NCCL_ROOT=/path/to/nccl \
CUDNN_ROOT=/path/to/cudnn \
MPI_HOME=/path/to/mpi \
ED_BUILD_PROFILE=performance \
bash scripts/build_cuda.sh
```

Run a minimal text-to-image command:

```bash
./build-cuda/bin/ed-cli \
  --backend cuda \
  --model /path/to/flux-dev \
  --prompt "a glass teapot on a wooden table" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --seed 0 \
  --output output.png
```

Use `./build-cuda/bin/ed-cli --help` for the full CLI surface. Detailed model,
memory, cache, parallel, server, and Python usage lives in the documents below.

## Documentation

- [Build and installation](docs/build.md)
- [Supported models and usage](docs/models.md)
- [Performance and optimization](docs/performance.md)
- [C API, server, and Python bindings](docs/api.md)
- [Development and contributing](docs/development.md)

Additional existing notes:

- [Sequence-parallel benchmark report](docs/sp_benchmark_report.md)
- [FLUX sequence-parallel profiling notes](docs/flux_sp_profile_root_cause.md)
- [Python bindings README](bindings/python/README.md)
- [Native HTTP server README](examples/server/README.md)
- [Frontend console runtime guide](bindings/python/frontend/server_v2-console/RUNTIME_CONFIGURATION.md)

## Contributing

Contributions are welcome during the public preview, especially clean build
reports, reproducible model smoke tests, backend fixes, benchmark metadata, and
documentation corrections.

Before opening a pull request, see
[Development and contributing](docs/development.md) and
[CONTRIBUTING.md](CONTRIBUTING.md). The project expects focused validation for
changes that touch runtime behavior, backend selection, cache behavior,
parallelism, or model loading.

## Acknowledgements

edge-dit.cpp builds on ggml and the broader open-source diffusion ecosystem.
The project also depends on platform runtimes such as CUDA, cuDNN, NCCL, MPI,
Metal, Vulkan, and Python packages depending on the selected backend and tools.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party
attribution and dependency notes.

## Citation

There is no formal technical report citation for edge-dit.cpp yet. For now,
cite the repository:

```bibtex
@software{edge_dit_cpp,
  title  = {edge-dit.cpp},
  author = {edge-dit.cpp contributors},
  url    = {https://github.com/yiming-l21/edge-dit.cpp},
  year   = {2026}
}
```

## License

edge-dit.cpp is licensed under the [Apache License 2.0](LICENSE).

Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [NOTICE](NOTICE), and license
files in `third_party/`.

Model weights are not distributed by this repository. Users are responsible for
reviewing and complying with the license and usage policy of each model they
download or run.
