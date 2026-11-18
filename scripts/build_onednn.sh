#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ONEDNN_DIR="${ROOT_DIR}/third_party/onednn"
BUILD_DIR="${BUILD_DIR:-${ONEDNN_DIR}/build}"
INSTALL_DIR="${ONEDNN_DIR}/install"
JOBS="${JOBS:-4}"

if ! command -v "${CMAKE_BIN}" >/dev/null 2>&1 && [[ ! -x "${CMAKE_BIN}" ]]; then
  echo "error: CMake was not found. Install CMake or set CMAKE_BIN=/path/to/cmake." >&2
  exit 1
fi

# Use the oneDNN revision pinned by this repository, including in a clean clone.
if [[ ! -f "${ONEDNN_DIR}/CMakeLists.txt" ]]; then
  git submodule update --init --recursive -- third_party/onednn
fi

# ggml uses dnnl_threadpool_interop_* with its own thread pool. The default
# OpenMP runtime does not export these symbols and causes linker failures.
# Install locally where scripts/build_cpu.sh automatically finds the package.
"${CMAKE_BIN}" -S "${ONEDNN_DIR}" -B "${BUILD_DIR}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}" \
  -DDNNL_CPU_RUNTIME=THREADPOOL \
  -DDNNL_GPU_RUNTIME=NONE \
  -DDNNL_LIBRARY_TYPE=SHARED \
  -DDNNL_EXPERIMENTAL_UKERNEL=ON \
  -DDNNL_BUILD_TESTS=OFF \
  -DDNNL_BUILD_EXAMPLES=OFF

"${CMAKE_BIN}" --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${JOBS}"
"${CMAKE_BIN}" --install "${BUILD_DIR}" --config "${BUILD_TYPE}"

echo "[build_onednn] installed to ${INSTALL_DIR}"
echo "[build_onednn] next: bash scripts/build_cpu.sh"
echo "[build_onednn] if needed, add ${INSTALL_DIR}/lib64 and ${INSTALL_DIR}/lib to LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)."
