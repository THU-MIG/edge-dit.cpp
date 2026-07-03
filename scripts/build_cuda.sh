#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
CMAKE_BIN=${CMAKE_BIN:-/usr/bin/cmake}
PYTHON_BIN=${PYTHON_BIN:-python3}
BUILD_DIR=${BUILD_DIR:-build-cuda}
ED_ENABLE_NCCL=${ED_ENABLE_NCCL:-ON}
ED_ENABLE_MPI=${ED_ENABLE_MPI:-ON}
ED_ENABLE_CUDNN_SDPA=${ED_ENABLE_CUDNN_SDPA:-ON}
ED_ENABLE_CUDA_NORM=${ED_ENABLE_CUDA_NORM:-ON}
ED_ENABLE_CUDA_ROPE=${ED_ENABLE_CUDA_ROPE:-ON}
ED_INSTALL_CUDNN=${ED_INSTALL_CUDNN:-ON}
NCCL_ROOT=${NCCL_ROOT:-}
CUDNN_ROOT=${CUDNN_ROOT:-}
CLEAN_PATH=${CUDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

truthy() {
  case "${1:-}" in
    1|ON|on|YES|yes|TRUE|true) return 0 ;;
    *) return 1 ;;
  esac
}

python_module_dir() {
  local module_name="$1"
  "${PYTHON_BIN}" - "$module_name" <<'PY' 2>/dev/null || true
import importlib.util
import sys

name = sys.argv[1]
spec = importlib.util.find_spec(name)
if spec is None:
    sys.exit(0)
locations = spec.submodule_search_locations
if locations:
    print(list(locations)[0])
elif spec.origin:
    print(spec.origin)
PY
}

discover_python_nvidia_root() {
  local cudnn_dir
  cudnn_dir="$(python_module_dir nvidia.cudnn)"
  if [[ -n "${cudnn_dir}" && -d "${cudnn_dir}" ]]; then
    dirname "${cudnn_dir}"
  fi
}

install_python_cudnn() {
  echo "Installing user-level NVIDIA cuDNN CUDA 12 wheels via ${PYTHON_BIN} -m pip ..."
  "${PYTHON_BIN}" -m pip install --user --upgrade \
    nvidia-cudnn-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12
}

maybe_add_python_nvidia_runtime_paths() {
  local root="$1"
  if [[ -d "${root}/cuda_nvrtc/lib" ]]; then
    CUDNN_LD_PATHS+=("${root}/cuda_nvrtc/lib")
  fi
  if [[ -d "${root}/cuda_runtime/lib" ]]; then
    CUDNN_LD_PATHS+=("${root}/cuda_runtime/lib")
  fi
  if [[ -d "${root}/cudnn/lib" ]]; then
    CUDNN_LD_PATHS+=("${root}/cudnn/lib")
  fi
}

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
  CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')"
fi

if [[ -z "${CUDA_ARCHITECTURES}" || ! "${CUDA_ARCHITECTURES}" =~ ^[0-9]+$ ]]; then
  # Buildable fallback for common datacenter/workstation NVIDIA GPUs when
  # no driver-visible GPU is available during configure.
  CUDA_ARCHITECTURES="75;80;86;89;90"
fi

echo "CUDA_ROOT=${CUDA_ROOT}"
echo "CMAKE_BIN=${CMAKE_BIN}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
echo "ED_ENABLE_NCCL=${ED_ENABLE_NCCL}"
echo "ED_ENABLE_MPI=${ED_ENABLE_MPI}"
echo "ED_ENABLE_CUDNN_SDPA=${ED_ENABLE_CUDNN_SDPA}"
echo "ED_ENABLE_CUDA_NORM=${ED_ENABLE_CUDA_NORM}"
echo "ED_ENABLE_CUDA_ROPE=${ED_ENABLE_CUDA_ROPE}"
echo "ED_INSTALL_CUDNN=${ED_INSTALL_CUDNN}"
if [[ -n "${NCCL_ROOT}" ]]; then
  echo "NCCL_ROOT=${NCCL_ROOT}"
fi

NCCL_ARGS=()
if [[ -n "${NCCL_ROOT}" ]]; then
  NCCL_ARGS+=("-DNCCL_ROOT=${NCCL_ROOT}")
fi

CUDNN_ARGS=()
CUDNN_LD_PATHS=()
if truthy "${ED_ENABLE_CUDNN_SDPA}"; then
  if [[ -z "${CUDNN_ROOT}" ]]; then
    NVIDIA_PY_ROOT="$(discover_python_nvidia_root)"
    if [[ -z "${NVIDIA_PY_ROOT}" ]]; then
      if truthy "${ED_INSTALL_CUDNN}"; then
        if ! install_python_cudnn; then
          echo "warning: pip installation of cuDNN wheels failed; continuing without cuDNN SDPA." >&2
        fi
        NVIDIA_PY_ROOT="$(discover_python_nvidia_root)"
      fi
    fi
    if [[ -n "${NVIDIA_PY_ROOT}" && -d "${NVIDIA_PY_ROOT}/cudnn" ]]; then
      CUDNN_ROOT="${NVIDIA_PY_ROOT}/cudnn"
      maybe_add_python_nvidia_runtime_paths "${NVIDIA_PY_ROOT}"
    fi
  else
    # If CUDNN_ROOT points into a Python NVIDIA wheel layout, also expose its
    # sibling CUDA runtime/NVRTC libraries for cuDNN's runtime-compiled engines.
    CUDNN_PARENT="$(dirname "${CUDNN_ROOT}")"
    maybe_add_python_nvidia_runtime_paths "${CUDNN_PARENT}"
  fi

  if [[ -n "${CUDNN_ROOT}" && -f "${CUDNN_ROOT}/include/cudnn.h" ]]; then
    echo "CUDNN_ROOT=${CUDNN_ROOT}"
    CUDNN_ARGS+=("-DED_ENABLE_CUDNN_SDPA=ON" "-DCUDNN_ROOT=${CUDNN_ROOT}")
  else
    echo "warning: cuDNN was not found; building CUDA backend without cuDNN SDPA fast attention." >&2
    echo "         Set CUDNN_ROOT=/path/to/cudnn or allow ED_INSTALL_CUDNN=ON to install Python wheels." >&2
    CUDNN_ARGS+=("-DED_ENABLE_CUDNN_SDPA=OFF")
  fi
else
  CUDNN_ARGS+=("-DED_ENABLE_CUDNN_SDPA=OFF")
fi

RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ ${#CUDNN_LD_PATHS[@]} -gt 0 ]]; then
  CUDNN_JOINED_LD_PATH="$(IFS=:; echo "${CUDNN_LD_PATHS[*]}")"
  if [[ -n "${RUNTIME_LD_LIBRARY_PATH}" ]]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_JOINED_LD_PATH}:${RUNTIME_LD_LIBRARY_PATH}"
  else
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_JOINED_LD_PATH}"
  fi
  echo "cuDNN runtime LD_LIBRARY_PATH prefix=${CUDNN_JOINED_LD_PATH}"
fi

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
CUDA_HOME=${CUDA_ROOT} \
CUDA_PATH=${CUDA_ROOT} \
CUDAToolkit_ROOT=${CUDA_ROOT} \
PATH=${CLEAN_PATH} \
${CMAKE_BIN} -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_CUDA=ON \
  -DED_ENABLE_NCCL="${ED_ENABLE_NCCL}" \
  -DED_ENABLE_MPI="${ED_ENABLE_MPI}" \
  -DED_ENABLE_CUDA_NORM="${ED_ENABLE_CUDA_NORM}" \
  -DED_ENABLE_CUDA_ROPE="${ED_ENABLE_CUDA_ROPE}" \
  -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=${CUDA_ROOT} \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT} \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  "${CUDNN_ARGS[@]}" \
  "${NCCL_ARGS[@]}"

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
PATH=${CLEAN_PATH} \
${CMAKE_BIN} --build "${BUILD_DIR}" -j

LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" "./${BUILD_DIR}/bin/ed-cli" --help
