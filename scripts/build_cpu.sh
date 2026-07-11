#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if ! command -v "${CMAKE_BIN}" >/dev/null 2>&1 && [[ ! -x "${CMAKE_BIN}" ]]; then
  echo "error: CMake was not found. Install CMake or set CMAKE_BIN=/path/to/cmake." >&2
  exit 1
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf -- "${BUILD_DIR}"
fi

"${CMAKE_BIN}" -S . -B "${BUILD_DIR}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  -DED_BUILD_EXAMPLES=ON

"${CMAKE_BIN}" --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/bin/ed-cli" --help
