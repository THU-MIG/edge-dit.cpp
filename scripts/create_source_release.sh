#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VERSION="${VERSION:-0.1.0-alpha}"
OUT_DIR="${OUT_DIR:-dist}"
PKG_DIR="${OUT_DIR}/edge-dit.cpp-${VERSION}"
ARCHIVE="${OUT_DIR}/edge-dit.cpp-${VERSION}.tar.gz"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: release source package must be created from a Git checkout." >&2
  exit 1
fi

if [[ ! -f third_party/ggml/CMakeLists.txt ]]; then
  echo "error: missing third_party/ggml. Run: git submodule update --init --recursive" >&2
  exit 1
fi

EDGE_COMMIT="$(git rev-parse HEAD)"
GGML_COMMIT="$(git -C third_party/ggml rev-parse HEAD)"

rm -rf -- "${PKG_DIR}" "${ARCHIVE}"
mkdir -p -- "${OUT_DIR}"

git archive --format=tar --prefix="edge-dit.cpp-${VERSION}/" HEAD | tar -x -C "${OUT_DIR}"
mkdir -p -- "${PKG_DIR}/third_party"
git -C third_party/ggml archive --format=tar --prefix="edge-dit.cpp-${VERSION}/third_party/ggml/" HEAD | tar -x -C "${OUT_DIR}"

mkdir -p -- "${PKG_DIR}/release"
cat > "${PKG_DIR}/release/dependency-manifest.json" <<EOF
{
  "version": "${VERSION}",
  "edge_dit_commit": "${EDGE_COMMIT}",
  "submodules": {
    "third_party/ggml": "${GGML_COMMIT}"
  }
}
EOF

find "${PKG_DIR}" \
  \( -name .git -o -name build -o -name build-cpu -o -name build-cuda -o -name build-vulkan -o -name build-metal \
     -o -name node_modules -o -name __pycache__ -o -name '*.pyc' -o -name '*.log' -o -name '*.png' -o -name '*.jpg' \
     -o -name '*.jpeg' -o -name '*.mp4' -o -name '*.webm' -o -name '*.safetensors' -o -name '*.gguf' -o -name '*.ckpt' \
     -o -name '*.pt' -o -name '*.pth' \) -prune -exec rm -rf -- {} +

tar -C "${OUT_DIR}" -czf "${ARCHIVE}" "edge-dit.cpp-${VERSION}"
echo "created ${ARCHIVE}"
echo "edge_dit_commit=${EDGE_COMMIT}"
echo "ggml_commit=${GGML_COMMIT}"
