#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CMAKE_BIN=${CMAKE_BIN:-cmake}
BUILD_DIR=${BUILD_DIR:-build-vulkan}
BUILD_TYPE=${BUILD_TYPE:-Release}

# ---------------------------------------------------------------------------
# ggml-vulkan needs three things at build time:
#   1. glslc            (compiles GLSL compute shaders to SPIR-V)
#   2. Vulkan headers   (>= 1.4.x; ggml-vulkan.cpp uses KHR coop-matrix and
#                        VK_EXT_layer_settings symbols absent from older 1.3.x)
#   3. SPIRV-Headers    (spirv/unified1/spirv.hpp -> the spv:: namespace used
#                        by the FP16-RTE SPIR-V patching code)
#
# Preferred: a standard Vulkan SDK providing all three. Point VULKAN_SDK at it.
#
# Fallback (no system SDK, e.g. datacenter boxes): a conda env provides glslc
# and the loader, and the two Khronos header repos are cloned separately:
#     conda install -c conda-forge shaderc vulkan-loader vulkan-tools
#     git clone --depth 1 -b v1.4.313 \
#         https://github.com/KhronosGroup/Vulkan-Headers.git
#     git clone --depth 1 \
#         https://github.com/KhronosGroup/SPIRV-Headers.git
# then export before running this script:
#     VK_EXTRA_HEADERS=/path/Vulkan-Headers/include
#     SPIRV_HEADERS=/path/SPIRV-Headers/include
# The conda env on PATH (CONDA_PREFIX) is auto-detected; no manual VULKAN_SDK
# is required in that case.
# ---------------------------------------------------------------------------

BUILD_PATH=${PATH}
CMAKE_EXTRA_ARGS=()
ISYSTEM_FLAGS=""

# Standard SDK layout (unchanged, back-compatible).
if [[ -n "${VULKAN_SDK:-}" ]]; then
  BUILD_PATH="${VULKAN_SDK}/bin:${BUILD_PATH}"
fi

# Auto-detect an active conda env that ships glslc + loader. Keep its bin on
# PATH (the clean PATH above would otherwise hide it) and hand cmake the loader
# and headers explicitly so it does not fall back to an older system copy.
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/glslc" ]]; then
  BUILD_PATH=${CONDA_PREFIX}/bin:${BUILD_PATH}
  CMAKE_EXTRA_ARGS+=("-DVulkan_GLSLC_EXECUTABLE=${CONDA_PREFIX}/bin/glslc")
  if [[ -f "${CONDA_PREFIX}/lib/libvulkan.so" ]]; then
    CMAKE_EXTRA_ARGS+=("-DVulkan_LIBRARY=${CONDA_PREFIX}/lib/libvulkan.so")
  fi
fi

# Newer Vulkan headers take priority over whatever find_package would pick.
if [[ -n "${VK_EXTRA_HEADERS:-}" ]]; then
  CMAKE_EXTRA_ARGS+=("-DVulkan_INCLUDE_DIR=${VK_EXTRA_HEADERS}")
  ISYSTEM_FLAGS="${ISYSTEM_FLAGS} -isystem ${VK_EXTRA_HEADERS}"
fi

# SPIRV-Headers is a separate repo; ggml-vulkan finds spirv.hpp via
# __has_include, so it only needs to be on the compiler search path.
if [[ -n "${SPIRV_HEADERS:-}" ]]; then
  ISYSTEM_FLAGS="${ISYSTEM_FLAGS} -isystem ${SPIRV_HEADERS}"
fi

if [[ -n "${ISYSTEM_FLAGS}" ]]; then
  CMAKE_EXTRA_ARGS+=("-DCMAKE_CXX_FLAGS=${ISYSTEM_FLAGS# }")
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf -- "${BUILD_DIR}"
fi

if ! env PATH=${BUILD_PATH} command -v glslc >/dev/null 2>&1; then
  echo "warning: glslc not found on PATH; ggml-vulkan shader compilation will fail." >&2
  echo "         Install the Vulkan SDK (shaderc), activate a conda env with" >&2
  echo "         shaderc, or set VULKAN_SDK=/path/to/sdk." >&2
fi

if ! env PATH="${BUILD_PATH}" command -v "${CMAKE_BIN}" >/dev/null 2>&1 && [[ ! -x "${CMAKE_BIN}" ]]; then
  echo "error: CMake was not found. Install CMake or set CMAKE_BIN=/path/to/cmake." >&2
  exit 1
fi

env PATH="${BUILD_PATH}" "${CMAKE_BIN}" -S . -B "${BUILD_DIR}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_VULKAN=ON \
  "${CMAKE_EXTRA_ARGS[@]}"

env PATH="${BUILD_PATH}" "${CMAKE_BIN}" --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/bin/ed-cli" --help
