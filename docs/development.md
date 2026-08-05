# Development and Contributing

[← Back to README](../README.md)

This document covers repository layout, validation, contribution expectations,
release packaging, and the current roadmap.

## Repository Layout

```text
include/edge-dit.h                    Public C API
src/edge_dit.cpp                      C API implementation
src/core/runtime/                     Engine, model loading, runtime setup
src/core/backend/                     ggml and CUDA/cuDNN integration
src/core/parallel/                    CFG/SP process-group infrastructure
src/core/optimization/cache/          Cache runtime, policies, and contracts
src/dit_models/                       Model and pipeline implementations
examples/cli/                         ed-cli, ed-sample, and ed-convert
examples/server/                      Native HTTP server
bindings/python/                      Python package, tests, Python Server, console
scripts/                              Build, bootstrap, validation, release tools
docs/                                 User and developer documentation
third_party/                          Submodules and vendored dependencies
```

## Development Setup

Bootstrap submodules:

```bash
bash scripts/bootstrap.sh
```

Build CPU path:

```bash
bash scripts/build_cpu.sh
```

Build minimal CUDA path:

```bash
ED_BUILD_PROFILE=minimal bash scripts/build_cuda.sh
```

For Python tests, use the source tree package:

```bash
PYTHONPATH=bindings/python/src python3 -m pytest bindings/python/tests
```

## Validation

Recommended local validation for documentation and release-engineering changes:

```bash
git diff --check
bash -n scripts/build_cuda.sh scripts/build_cpu.sh scripts/build_metal.sh scripts/build_vulkan.sh scripts/bootstrap.sh scripts/create_source_release.sh
```

CPU configure/build:

```bash
cmake -S . -B build-release-check \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON
cmake --build build-release-check -j 8
```

CUDA performance release gate:

```bash
CUDA_HOME=/path/to/cuda \
NCCL_ROOT=/path/to/nccl \
CUDNN_ROOT=/path/to/cudnn \
MPI_HOME=/path/to/mpi \
ED_BUILD_PROFILE=performance \
BUILD_DIR=build-cuda-performance \
bash scripts/build_cuda.sh
```

Then validate:

```text
complete configure/build
CLI smoke
real model single-GPU CUDA smoke
NCCL/MPI multi-GPU path
cuDNN SDPA path confirmation
fixed benchmark regression
```

## Source Release Packaging

For release source packages that include exact submodule content:

```bash
VERSION=0.1.0-alpha OUT_DIR=dist bash scripts/create_source_release.sh
```

The package records:

- edge-dit.cpp commit
- ggml commit
- submodule manifest data

It excludes `.git`, build outputs, model weights, logs, caches, generated
images, and generated videos.

## Adding a Model

Model additions should include:

- loader detection or explicit component loading rules
- pipeline implementation
- public CLI/API behavior
- memory and backend expectations
- at least one smoke path
- documentation in [Supported models and usage](models.md)

Do not add a model to the public support matrix until a real checkpoint has
been loaded and a minimal generation path has been verified.

## Adding an Operator or Backend

Operator/backend work must document:

- build flags and dependency requirements
- fallback behavior, if any
- supported tensor dtypes and layouts
- backend selection behavior
- tests or reproducible smoke commands
- performance risk and validation results

Changes touching kernel selection, graph construction, tensor layout, precision
policy, cache behavior, NCCL/MPI semantics, sequence parallelism, CFG
parallelism, scheduler math, or model outputs require targeted runtime
validation.

## Pull Request Requirements

Before opening a PR:

- Keep changes scoped.
- Do not commit model weights, generated images/videos, logs, caches, or build
  outputs.
- Run the relevant validation commands.
- Update the appropriate topic document instead of expanding the root README.
- Include benchmark metadata for performance claims.
- State whether full CUDA performance validation was run.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution licensing and
workflow notes.

## Release Process

Current public preview state:

```text
Repository-ready: yes
Public preview-ready: yes
v0.1.0-alpha: released to public preview (2026-07-11)
```

The checklist below is the release gate each subsequent tagged build runs
through:

1. Clean clone with submodules.
2. CPU configure/build.
3. Minimal CUDA configure/build.
4. Full `performance` CUDA configure/build on complete dependency machine.
5. Native CLI smoke.
6. Python tests.
7. Native server and Python server smoke as applicable.
8. Real model single-GPU smoke.
9. NCCL/MPI multi-GPU smoke.
10. cuDNN SDPA path confirmation.
11. Fixed benchmark regression.
12. Source package generation and manifest review.

## Roadmap

Near-term work:

- Keep the full CUDA performance release gate green across tagged builds.
- Stabilize public C API and runtime interfaces.
- Expand verified model/backend matrix.
- Improve benchmark reproducibility and metadata capture.
- Improve model conversion and packaging workflows.
- Continue operator and memory optimization.

This roadmap is intentionally limited to active project directions, not a
promise to support every model scaffold present in the source tree.

## Related Documentation

- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [Command line usage](cli.md)
- [Performance and benchmarks (RTX 4090)](performance-4090.md)
- [H200 snapshot](performance-H200.md)
- [API and bindings](api.md)
