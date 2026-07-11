# Build and Installation

[← Back to README](../README.md)

This document covers clean clone setup, submodules, build profiles, backend
builds, and common build failures.

## Clean Clone

Preferred clone:

```bash
git clone --recursive https://github.com/yiming-l21/edge-dit.cpp
cd edge-dit.cpp
```

If the checkout already exists:

```bash
bash scripts/bootstrap.sh
```

`scripts/bootstrap.sh` verifies that the current directory is a Git checkout,
runs:

```bash
git submodule update --init --recursive
```

and checks that `third_party/ggml/CMakeLists.txt` exists.

GitHub auto-generated Source ZIP archives usually do not include submodule
contents. Use `git clone --recursive`, `scripts/bootstrap.sh`, or a complete
project release source package.

## Prerequisites

Common tools:

- CMake 3.20 or newer.
- C and C++ compilers with C++17 support.
- Git, including submodule support.

Optional backend-specific tools:

- CUDA Toolkit and `nvcc` for CUDA.
- NCCL, MPI, cuDNN, and cudnn-frontend for the official CUDA performance
  profile.
- Apple build tools and frameworks for Metal on macOS.
- Vulkan SDK, `glslc`, Vulkan headers, and SPIRV-Headers for Vulkan.
- Python 3 for tests, hygiene checks, and Python bindings.

The build scripts honor standard overrides:

```text
CMAKE_BIN
CC
CXX
CUDACXX
CUDA_HOME
CUDA_PATH
NCCL_ROOT
CUDNN_ROOT
MPI_HOME
BUILD_DIR
CLEAN
```

## CUDA Build Profiles

CUDA builds use `ED_BUILD_PROFILE`.

### performance

`performance` is the default and official performance configuration.

It enables the CUDA backend and the full performance dependency path:

- NCCL
- MPI
- cuDNN SDPA
- CUDA Norm
- CUDA RoPE
- CUDA Modulation
- parallel runtime support

If an explicitly enabled dependency is missing, the script fails before CMake
configure. It does not silently downgrade to a slower configuration.

```bash
CUDA_HOME=/path/to/cuda \
NCCL_ROOT=/path/to/nccl \
CUDNN_ROOT=/path/to/cudnn \
MPI_HOME=/path/to/mpi \
ED_BUILD_PROFILE=performance \
bash scripts/build_cuda.sh
```

`performance` remains the default:

```bash
bash scripts/build_cuda.sh
```

Full `performance` profile validation is still the final v0.1.0-alpha release
gate. Do not treat `minimal` results as official performance data.

### minimal

`minimal` is a reduced external dependency configuration for build and
functional validation:

```bash
ED_BUILD_PROFILE=minimal bash scripts/build_cuda.sh
```

It disables NCCL, MPI, and cuDNN SDPA by default. It keeps the CUDA backend and
the in-repository CUDA helper paths needed by the current patched ggml CUDA
integration. It is not valid for official benchmark results.

### User overrides

Environment variables set by the user override profile defaults. For example:

```bash
ED_BUILD_PROFILE=performance \
ED_ENABLE_CUDNN_SDPA=OFF \
bash scripts/build_cuda.sh
```

Automatic user-level cuDNN wheel installation is opt-in:

```bash
ED_INSTALL_CUDNN=ON ED_BUILD_PROFILE=performance bash scripts/build_cuda.sh
```

The script does not call package managers unless an explicit install variable
such as `ED_INSTALL_CUDNN=ON` is set.

## CUDA Configuration Summary

Before CMake configure, `scripts/build_cuda.sh` prints a compact summary with:

- profile
- source and build directories
- CMake and compiler paths / versions
- CUDA Toolkit and `nvcc`
- CUDA architectures
- NCCL, MPI, and cuDNN status
- cuDNN SDPA, CUDA Norm, CUDA RoPE, CUDA Modulation
- CFG and sequence parallel status
- build type and library mode

After configure, CMake writes:

```text
<build-dir>/build-config.txt
```

The file records resolved feature switches, dependency versions when available,
CUDA architectures, compiler versions, build type, Git commit, and ggml
submodule commit. It intentionally avoids unnecessary personal paths.

## CPU Build

```bash
bash scripts/build_cpu.sh
```

CPU is mainly for build validation, smoke tests, fallback operators, CPU
offload, and selected low-speed inference.

Output defaults to:

```text
build-cpu/
```

## CUDA Build

Quick validation:

```bash
ED_BUILD_PROFILE=minimal bash scripts/build_cuda.sh
```

Performance validation:

```bash
CUDA_HOME=/path/to/cuda \
NCCL_ROOT=/path/to/nccl \
CUDNN_ROOT=/path/to/cudnn \
MPI_HOME=/path/to/mpi \
ED_BUILD_PROFILE=performance \
BUILD_DIR=build-cuda-performance \
bash scripts/build_cuda.sh
```

Output defaults to:

```text
build-cuda/
```

## Metal Build

Metal is macOS-only and experimental:

```bash
bash scripts/build_metal.sh
```

Output defaults to:

```text
build-metal/
```

## Vulkan Build

Vulkan is experimental. The helper script looks for `glslc`, Vulkan headers,
and SPIRV-Headers:

```bash
bash scripts/build_vulkan.sh
```

Useful overrides:

```bash
VULKAN_SDK=/path/to/vulkan-sdk bash scripts/build_vulkan.sh
VK_EXTRA_HEADERS=/path/to/Vulkan-Headers/include \
SPIRV_HEADERS=/path/to/SPIRV-Headers/include \
bash scripts/build_vulkan.sh
```

Output defaults to:

```text
build-vulkan/
```

## Cleaning

Each build script accepts:

```bash
CLEAN=1 bash scripts/build_cpu.sh
```

`CLEAN=1` only removes the selected build directory.

## Common Build Errors

### Missing ggml submodule

Symptom:

```text
Missing ggml submodule at third_party/ggml
```

Fix:

```bash
bash scripts/bootstrap.sh
```

### Missing NCCL in performance profile

Symptom:

```text
ED_ENABLE_NCCL=ON requires NCCL headers and library
```

Fix:

```bash
NCCL_ROOT=/path/to/nccl ED_BUILD_PROFILE=performance bash scripts/build_cuda.sh
```

### Missing MPI in performance profile

Install an MPI implementation or set:

```bash
MPI_HOME=/path/to/mpi
```

### Missing cuDNN or cudnn-frontend

Install cuDNN and set:

```bash
CUDNN_ROOT=/path/to/cudnn
```

If the vendored cudnn-frontend source is absent, either initialize submodules
or explicitly allow configure-time fetching:

```bash
ED_FETCH_CUDNN_FRONTEND=ON
```

### Python package not found during tests

From the repository root:

```bash
PYTHONPATH=bindings/python/src python3 -m pytest bindings/python/tests
```

## Related Documentation

- [Supported models and usage](models.md)
- [Performance and optimization](performance.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)
