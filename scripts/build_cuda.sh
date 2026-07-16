#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

truthy() {
  case "${1:-}" in
    1|ON|on|YES|yes|TRUE|true) return 0 ;;
    *) return 1 ;;
  esac
}

env_is_set() {
  [[ -n "${!1+x}" ]]
}

apply_default() {
  local name="$1"
  local value="$2"
  if ! env_is_set "${name}"; then
    printf -v "${name}" '%s' "${value}"
  fi
}

fail() {
  echo "error: $*" >&2
  echo "hint: use CMAKE_BIN, CC, CXX, CUDACXX, CUDA_HOME/CUDA_PATH, NCCL_ROOT, CUDNN_ROOT, or MPI_HOME to point at non-standard installs." >&2
  exit 1
}

find_exe() {
  local var_name="$1"
  local fallback="$2"
  local value="${!var_name:-}"
  if [[ -n "${value}" ]]; then
    command -v "${value}" 2>/dev/null || { [[ -x "${value}" ]] && printf '%s\n' "${value}"; }
    return
  fi
  command -v "${fallback}" 2>/dev/null || true
}

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

first_existing_dir_with_file() {
  local required_file="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" && -f "${candidate}/${required_file}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

common_prefix_from_include_lib() {
  local include_dir="$1"
  local _lib_file="$2"
  printf '%s\n' "${include_dir%/include}"
}

python_module_dir() {
  local module_name="$1"
  "${PYTHON_BIN:-python3}" - "${module_name}" <<'PY' 2>/dev/null || true
import importlib.util
import sys

spec = importlib.util.find_spec(sys.argv[1])
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

python_nvidia_cuda_deps_ready() {
  "${PYTHON_BIN:-python3}" - <<'PY' >/dev/null 2>&1
import importlib.util
from pathlib import Path
import sys

required = {
    "nvidia.cudnn": [
        ("include", "cudnn.h"),
        ("include", "cudnn_version.h"),
        ("lib", "libcudnn.so"),
    ],
    "nvidia.cuda_nvrtc": [("lib", "libnvrtc.so")],
    "nvidia.cuda_runtime": [("lib", "libcudart.so")],
}

for module, files in required.items():
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(1)
    root = Path(list(spec.submodule_search_locations)[0])
    for subdir, stem in files:
        if not any((root / subdir).glob(stem + "*")):
            sys.exit(1)
PY
}

install_python_cudnn() {
  if python_nvidia_cuda_deps_ready; then
    echo "Using existing Python NVIDIA cuDNN/CUDA runtime wheels."
    return
  fi
  echo "Installing user-level NVIDIA cuDNN CUDA 12 wheels via ${PYTHON_BIN:-python3} -m pip ..."
  "${PYTHON_BIN:-python3}" -m pip install --user \
    nvidia-cudnn-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12
}

prepare_python_cudnn_root() {
  local nvidia_root="$1"
  local deps_dir
  local out_dir
  local lib
  local name
  local stem

  mkdir -p "${BUILD_DIR}/_deps"
  deps_dir="$(cd "${BUILD_DIR}/_deps" && pwd)"
  out_dir="${deps_dir}/python-cudnn"
  rm -rf -- "${out_dir}"
  mkdir -p "${out_dir}/lib"
  ln -s "${nvidia_root}/cudnn/include" "${out_dir}/include"

  for lib in "${nvidia_root}/cudnn/lib"/libcudnn*.so*; do
    [[ -f "${lib}" ]] || continue
    name="$(basename "${lib}")"
    stem="${name%%.so*}"
    ln -sf "${lib}" "${out_dir}/lib/${name}"
    ln -sf "${lib}" "${out_dir}/lib/${stem}.so"
  done

  # cuDNN 9 splits its backend into engine .so's that dlopen sibling CUDA libs
  # (libcublasLt/libcublas from the cublas wheel, libnvrtc from the nvrtc wheel)
  # lazily at first conv. Those dlopen deps resolve via the *launching binary's*
  # RUNPATH — which includes this out_dir — NOT via the engine lib's own RUNPATH
  # (RUNPATH is not transitive to dlopen'd grandchildren). So the matching-version
  # libs must live here too, or ed-cli aborts mid-run with
  # "Cannot load symbol cublasLtGetVersion". Symlink them alongside cuDNN.
  local sib_dir sib
  for sib_dir in cublas cuda_nvrtc; do
    for sib in "${nvidia_root}/${sib_dir}/lib"/lib*.so*; do
      [[ -f "${sib}" ]] || continue
      name="$(basename "${sib}")"
      stem="${name%%.so*}"
      ln -sf "${sib}" "${out_dir}/lib/${name}"
      ln -sf "${sib}" "${out_dir}/lib/${stem}.so"
    done
  done

  printf '%s\n' "${out_dir}"
}

tool_version() {
  local exe="$1"
  "${exe}" --version 2>/dev/null | head -n 1 || true
}

header_define_version() {
  local header="$1"
  local prefix="$2"
  awk -v p="${prefix}" '
    $1 == "#define" && $2 == p "_MAJOR" { major=$3 }
    $1 == "#define" && $2 == p "_MINOR" { minor=$3 }
    $1 == "#define" && ($2 == p "_PATCH" || $2 == p "_PATCHLEVEL") { patch=$3 }
    END {
      if (major != "") {
        printf "%s.%s.%s", major, minor == "" ? "0" : minor, patch == "" ? "0" : patch
      }
    }
  ' "${header}" 2>/dev/null || true
}

ED_BUILD_PROFILE="${ED_BUILD_PROFILE:-performance}"
case "${ED_BUILD_PROFILE}" in
  performance|minimal) ;;
  *) fail "ED_BUILD_PROFILE must be 'performance' or 'minimal'." ;;
esac

CMAKE_BIN="${CMAKE_BIN:-cmake}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ED_AUTO_INSTALL_DEPS="${ED_AUTO_INSTALL_DEPS:-ON}"

if [[ "${ED_BUILD_PROFILE}" == "performance" ]]; then
  apply_default ED_BUILD_EXAMPLES ON
  apply_default ED_GGML_CUDA ON
  apply_default ED_ENABLE_NCCL ON
  apply_default ED_ENABLE_MPI ON
  apply_default ED_ENABLE_CUDNN_SDPA ON
  apply_default ED_ENABLE_CUDA_NORM ON
  apply_default ED_ENABLE_CUDA_ROPE ON
  apply_default ED_ENABLE_CUDA_MODULATION ON
  apply_default ED_ENABLE_PARALLEL ON
  apply_default ED_INSTALL_CUDNN "${ED_AUTO_INSTALL_DEPS}"
  apply_default ED_INSTALL_CUDNN_FRONTEND "${ED_AUTO_INSTALL_DEPS}"
else
  apply_default ED_BUILD_EXAMPLES ON
  apply_default ED_GGML_CUDA ON
  apply_default ED_ENABLE_NCCL OFF
  apply_default ED_ENABLE_MPI OFF
  apply_default ED_ENABLE_CUDNN_SDPA OFF
  apply_default ED_ENABLE_CUDA_NORM ON
  apply_default ED_ENABLE_CUDA_ROPE ON
  apply_default ED_ENABLE_CUDA_MODULATION ON
  apply_default ED_ENABLE_PARALLEL ON
  apply_default ED_INSTALL_CUDNN OFF
  apply_default ED_INSTALL_CUDNN_FRONTEND OFF
fi

if truthy "${ED_INSTALL_CUDNN_FRONTEND}"; then
  ED_FETCH_CUDNN_FRONTEND="${ED_FETCH_CUDNN_FRONTEND:-ON}"
fi

CMAKE_EXE="$(find_exe CMAKE_BIN cmake)"
[[ -n "${CMAKE_EXE}" ]] || fail "CMake was not found. Install CMake or set CMAKE_BIN=/path/to/cmake."
CC_EXE="$(find_exe CC cc)"
[[ -n "${CC_EXE}" ]] || fail "C compiler was not found. Install a C compiler or set CC=/path/to/cc."
CXX_EXE="$(find_exe CXX c++)"
[[ -n "${CXX_EXE}" ]] || fail "C++ compiler was not found. Install a C++ compiler or set CXX=/path/to/c++."

CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "${CUDA_ROOT}" ]]; then
  if [[ -n "${CUDACXX:-}" ]]; then
    CUDA_ROOT="$(cd "$(dirname "${CUDACXX}")/.." && pwd)"
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  fi
fi
[[ -n "${CUDA_ROOT}" && -d "${CUDA_ROOT}" ]] || fail "CUDA Toolkit was not found. Set CUDA_HOME or CUDA_PATH."

CUDACXX="${CUDACXX:-${CUDA_ROOT}/bin/nvcc}"
NVCC_EXE="$(find_exe CUDACXX nvcc)"
[[ -n "${NVCC_EXE}" ]] || fail "nvcc was not found. Set CUDACXX=/path/to/nvcc or CUDA_HOME=/path/to/cuda."

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf -- "${BUILD_DIR}"
fi

if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
  CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '.' || true)"
fi
if [[ -z "${CUDA_ARCHITECTURES}" || ! "${CUDA_ARCHITECTURES}" =~ ^[0-9]+$ ]]; then
  CUDA_ARCHITECTURES="75;80;86;89;90"
fi

cache_value() {
  local cache="$1"
  local key="$2"
  [[ -f "${cache}" ]] || return 1
  awk -F= -v key="${key}" '
    $1 ~ "^" key ":" { print $2; found=1; exit }
    END { exit found ? 0 : 1 }
  ' "${cache}"
}

maybe_clear_stale_compiler_cache() {
  local cache="${BUILD_DIR}/CMakeCache.txt"
  [[ -f "${cache}" ]] || return 0

  local cached_cc cached_cxx cached_cuda
  cached_cc="$(cache_value "${cache}" CMAKE_C_COMPILER || true)"
  cached_cxx="$(cache_value "${cache}" CMAKE_CXX_COMPILER || true)"
  cached_cuda="$(cache_value "${cache}" CMAKE_CUDA_COMPILER || true)"

  if [[ "${cached_cc}" != "${CC_EXE}" ||
        "${cached_cxx}" != "${CXX_EXE}" ||
        ( -n "${cached_cuda}" && "${cached_cuda}" != "${NVCC_EXE}" ) ]]; then
    echo "CMake compiler cache changed; clearing ${BUILD_DIR}/CMakeCache.txt and ${BUILD_DIR}/CMakeFiles before configure." >&2
    rm -f -- "${cache}"
    rm -rf -- "${BUILD_DIR}/CMakeFiles"
  fi
}

maybe_clear_stale_compiler_cache

CMAKE_ARGS=(
  -S .
  -B "${BUILD_DIR}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DED_BUILD_EXAMPLES=${ED_BUILD_EXAMPLES}"
  "-DED_GGML_CUDA=${ED_GGML_CUDA}"
  "-DED_ENABLE_NCCL=${ED_ENABLE_NCCL}"
  "-DED_ENABLE_MPI=${ED_ENABLE_MPI}"
  "-DED_ENABLE_CUDNN_SDPA=${ED_ENABLE_CUDNN_SDPA}"
  "-DED_ENABLE_CUDA_NORM=${ED_ENABLE_CUDA_NORM}"
  "-DED_ENABLE_CUDA_ROPE=${ED_ENABLE_CUDA_ROPE}"
  "-DED_ENABLE_CUDA_MODULATION=${ED_ENABLE_CUDA_MODULATION}"
  "-DED_ENABLE_PARALLEL=${ED_ENABLE_PARALLEL}"
  -DGGML_CUDA_NCCL=OFF
  "-DCMAKE_C_COMPILER=${CC_EXE}"
  "-DCMAKE_CXX_COMPILER=${CXX_EXE}"
  "-DCMAKE_CUDA_COMPILER=${NVCC_EXE}"
  "-DCMAKE_CUDA_HOST_COMPILER=${CXX_EXE}"
  "-DCUDAToolkit_ROOT=${CUDA_ROOT}"
  "-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
)

if [[ -n "${ED_BUILD_SHARED_LIBS:-}" ]]; then
  CMAKE_ARGS+=("-DED_BUILD_SHARED_LIBS=${ED_BUILD_SHARED_LIBS}")
fi
if [[ -n "${EXTRA_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMAKE_ARGS+=(${EXTRA_CMAKE_ARGS})
fi

NCCL_STATUS="disabled"
NCCL_VERSION="n/a"
NCCL_PATH="n/a"
if truthy "${ED_ENABLE_NCCL}"; then
  NCCL_INCLUDE="$(first_existing_dir_with_file nccl.h \
    "${NCCL_ROOT:-}/include" \
    "${ROOT_DIR}/third_party/nccl/include" \
    "${CUDA_ROOT}/include" \
    /usr/include \
    /usr/local/include || true)"
  NCCL_LIB="$(first_existing_file \
    "${NCCL_ROOT:-}/lib/libnccl.so" \
    "${NCCL_ROOT:-}/lib64/libnccl.so" \
    "${ROOT_DIR}/third_party/nccl/lib/libnccl.so" \
    "${ROOT_DIR}/third_party/nccl/lib64/libnccl.so" \
    "${CUDA_ROOT}/lib64/libnccl.so" \
    /usr/lib64/libnccl.so \
    /usr/lib/libnccl.so \
    /usr/lib/x86_64-linux-gnu/libnccl.so \
    /usr/local/lib/libnccl.so \
    /usr/local/lib64/libnccl.so || true)"
  [[ -n "${NCCL_INCLUDE}" && -f "${NCCL_INCLUDE}/nccl.h" && -n "${NCCL_LIB}" ]] || fail "ED_ENABLE_NCCL=ON requires NCCL headers and library. Install NCCL, set NCCL_ROOT=/path/to/nccl, or provide third_party/nccl."
  NCCL_STATUS="enabled"
  NCCL_PATH="${NCCL_ROOT:-$(common_prefix_from_include_lib "${NCCL_INCLUDE}" "${NCCL_LIB}")}"
  NCCL_VERSION="$(header_define_version "${NCCL_INCLUDE}/nccl.h" NCCL)"
  NCCL_VERSION="${NCCL_VERSION:-unknown}"
  CMAKE_ARGS+=("-DNCCL_ROOT=${NCCL_PATH}")
fi

MPI_STATUS="disabled"
MPI_VERSION="n/a"
MPI_PATH="n/a"
if truthy "${ED_ENABLE_MPI}"; then
  MPI_CXX_EXE=""
  if [[ -n "${MPI_HOME:-}" ]]; then
    MPI_CXX_EXE="$(command -v "${MPI_HOME}/bin/mpicxx" 2>/dev/null || true)"
    CMAKE_ARGS+=("-DMPI_HOME=${MPI_HOME}")
  fi
  MPI_CXX_EXE="${MPI_CXX_EXE:-$(command -v mpicxx 2>/dev/null || command -v mpiCC 2>/dev/null || true)}"
  [[ -n "${MPI_CXX_EXE}" ]] || fail "ED_ENABLE_MPI=ON requires MPI. Install MPI or set MPI_HOME=/path/to/mpi."
  MPI_STATUS="enabled"
  MPI_PATH="${MPI_HOME:-$(cd "$(dirname "${MPI_CXX_EXE}")/.." && pwd)}"
  MPI_VERSION="$("${MPI_CXX_EXE}" --version 2>/dev/null | head -n 1 || true)"
  MPI_VERSION="${MPI_VERSION:-unknown}"
fi

CUDNN_STATUS="disabled"
CUDNN_VERSION="n/a"
CUDNN_PATH_DISPLAY="n/a"
CUDNN_LD_PATHS=()
if truthy "${ED_ENABLE_CUDNN_SDPA}"; then
  [[ -d "${CUDA_ROOT}/lib64" ]] && CUDNN_LD_PATHS+=("${CUDA_ROOT}/lib64")
  [[ -d "${CUDA_ROOT}/lib" ]] && CUDNN_LD_PATHS+=("${CUDA_ROOT}/lib")
  if [[ -z "${CUDNN_ROOT:-}" ]] && truthy "${ED_INSTALL_CUDNN}"; then
    install_python_cudnn
    NVIDIA_PY_ROOT="$(discover_python_nvidia_root)"
    if [[ -n "${NVIDIA_PY_ROOT}" && -d "${NVIDIA_PY_ROOT}/cudnn" ]]; then
      CUDNN_ROOT="$(prepare_python_cudnn_root "${NVIDIA_PY_ROOT}")"
      [[ -d "${CUDNN_ROOT}/lib" ]] && CUDNN_LD_PATHS+=("${CUDNN_ROOT}/lib")
      [[ -d "${NVIDIA_PY_ROOT}/cuda_nvrtc/lib" ]] && CUDNN_LD_PATHS+=("${NVIDIA_PY_ROOT}/cuda_nvrtc/lib")
      [[ -d "${NVIDIA_PY_ROOT}/cuda_runtime/lib" ]] && CUDNN_LD_PATHS+=("${NVIDIA_PY_ROOT}/cuda_runtime/lib")
      [[ -d "${NVIDIA_PY_ROOT}/cudnn/lib" ]] && CUDNN_LD_PATHS+=("${NVIDIA_PY_ROOT}/cudnn/lib")
    fi
  fi
  CUDNN_INCLUDE="$(first_existing_dir \
    "${CUDNN_ROOT:-}/include" \
    "${CUDA_ROOT}/include" || true)"
  CUDNN_LIB="$(first_existing_file \
    "${CUDNN_ROOT:-}/lib/libcudnn.so" \
    "${CUDNN_ROOT:-}/lib/libcudnn.so.9" \
    "${CUDNN_ROOT:-}/lib64/libcudnn.so" \
    "${CUDNN_ROOT:-}/lib64/libcudnn.so.9" \
    "${CUDA_ROOT}/lib64/libcudnn.so" || true)"
  [[ -n "${CUDNN_INCLUDE}" && -f "${CUDNN_INCLUDE}/cudnn.h" && -n "${CUDNN_LIB}" ]] || fail "ED_ENABLE_CUDNN_SDPA=ON requires cuDNN. The build script tried automatic user-level installation when ED_INSTALL_CUDNN=ON. Set CUDNN_ROOT=/path/to/cudnn, or set ED_AUTO_INSTALL_DEPS=OFF to disable automatic dependency installation."
  [[ -f "${ROOT_DIR}/third_party/cudnn-frontend/CMakeLists.txt" || "${ED_FETCH_CUDNN_FRONTEND:-OFF}" == "ON" ]] || fail "cuDNN SDPA requires third_party/cudnn-frontend. The build script enables CMake fetching by default when ED_INSTALL_CUDNN_FRONTEND=ON. Initialize submodules, set ED_FETCH_CUDNN_FRONTEND=ON, or set ED_AUTO_INSTALL_DEPS=OFF to disable automatic dependency fetching."
  CUDNN_STATUS="enabled"
  CUDNN_ROOT="${CUDNN_ROOT:-${CUDNN_INCLUDE%/include}}"
  CUDNN_PATH_DISPLAY="${CUDNN_ROOT}"
  CUDNN_VERSION="$(header_define_version "${CUDNN_INCLUDE}/cudnn_version.h" CUDNN)"
  CUDNN_VERSION="${CUDNN_VERSION:-$(header_define_version "${CUDNN_INCLUDE}/cudnn.h" CUDNN)}"
  CUDNN_VERSION="${CUDNN_VERSION:-unknown}"
  CMAKE_ARGS+=("-DCUDNN_ROOT=${CUDNN_ROOT}")
fi

if [[ -n "${ED_FETCH_CUDNN_FRONTEND:-}" ]]; then
  CMAKE_ARGS+=("-DED_FETCH_CUDNN_FRONTEND=${ED_FETCH_CUDNN_FRONTEND}")
fi

RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ ${#CUDNN_LD_PATHS[@]} -gt 0 ]]; then
  CUDNN_JOINED_LD_PATH="$(IFS=:; echo "${CUDNN_LD_PATHS[*]}")"
  RUNTIME_LD_LIBRARY_PATH="${CUDNN_JOINED_LD_PATH}${RUNTIME_LD_LIBRARY_PATH:+:${RUNTIME_LD_LIBRARY_PATH}}"
  CMAKE_ARGS+=("-DCUDNN_RUNTIME_LIBRARY_DIRS=$(IFS=';'; echo "${CUDNN_LD_PATHS[*]}")")
fi

cat <<EOF
edge-dit CUDA build configuration
  profile: ${ED_BUILD_PROFILE}
  source dir: ${ROOT_DIR}
  build dir: ${BUILD_DIR}
  cmake: ${CMAKE_EXE} ($(tool_version "${CMAKE_EXE}"))
  C compiler: ${CC_EXE} ($(tool_version "${CC_EXE}"))
  C++ compiler: ${CXX_EXE} ($(tool_version "${CXX_EXE}"))
  CUDA Toolkit: ${CUDA_ROOT}
  nvcc: ${NVCC_EXE} ($(tool_version "${NVCC_EXE}"))
  CUDA architectures: ${CUDA_ARCHITECTURES}
  NCCL: ${NCCL_STATUS} version=${NCCL_VERSION} path=${NCCL_PATH}
  MPI: ${MPI_STATUS} version=${MPI_VERSION} path=${MPI_PATH}
  cuDNN: ${CUDNN_STATUS} version=${CUDNN_VERSION} path=${CUDNN_PATH_DISPLAY}
  cuDNN SDPA: ${ED_ENABLE_CUDNN_SDPA}
  CUDA Norm: ${ED_ENABLE_CUDA_NORM}
  CUDA RoPE: ${ED_ENABLE_CUDA_ROPE}
  CUDA Modulation: ${ED_ENABLE_CUDA_MODULATION}
  CFG Parallel: ${ED_ENABLE_PARALLEL}
  Sequence Parallel: ${ED_ENABLE_PARALLEL}
  automatic dependency install: ${ED_AUTO_INSTALL_DEPS}
  install cuDNN Python wheels: ${ED_INSTALL_CUDNN}
  fetch cudnn-frontend: ${ED_FETCH_CUDNN_FRONTEND:-OFF}
  build type: ${BUILD_TYPE}
  library mode: ${ED_BUILD_SHARED_LIBS:-OFF}
  ggml CUDA: ${ED_GGML_CUDA}
  examples: ${ED_BUILD_EXAMPLES}
EOF

"${CMAKE_EXE}" "${CMAKE_ARGS[@]}"
"${CMAKE_EXE}" --build "${BUILD_DIR}" -j

if [[ -x "./${BUILD_DIR}/bin/ed-cli" ]]; then
  LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" "./${BUILD_DIR}/bin/ed-cli" --help
fi
