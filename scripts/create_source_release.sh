#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VERSION="${VERSION:-0.1.0}"
OUT_DIR="${OUT_DIR:-dist}"
PKG_DIR="${OUT_DIR}/edge-dit.cpp-${VERSION}"
ARCHIVE="${OUT_DIR}/edge-dit.cpp-${VERSION}.tar.gz"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: release source package must be created from a Git checkout." >&2
  exit 1
fi

# All submodules declared in .gitmodules (ggml, onednn, ...) must be present and
# get archived into the release package + pinned in the manifest. A source ZIP
# missing any submodule fails to build for whoever unpacks it.
SUBMODULE_PATHS=()
while IFS= read -r sub_path; do
  SUBMODULE_PATHS+=("${sub_path}")
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')

for sub_path in "${SUBMODULE_PATHS[@]}"; do
  if [[ -z "$(ls -A "${sub_path}" 2>/dev/null)" ]]; then
    echo "error: missing submodule ${sub_path}. Run: git submodule update --init --recursive" >&2
    exit 1
  fi
done

EDGE_COMMIT="$(git rev-parse HEAD)"

rm -rf -- "${PKG_DIR}" "${ARCHIVE}"
mkdir -p -- "${OUT_DIR}"

git archive --format=tar --prefix="edge-dit.cpp-${VERSION}/" HEAD | tar -x -C "${OUT_DIR}"
mkdir -p -- "${PKG_DIR}/third_party"
for sub_path in "${SUBMODULE_PATHS[@]}"; do
  git -C "${sub_path}" archive --format=tar \
    --prefix="edge-dit.cpp-${VERSION}/${sub_path}/" HEAD | tar -x -C "${OUT_DIR}"
done

# Build the submodules JSON block (one "path": "commit" line per submodule).
sub_json=""
for sub_path in "${SUBMODULE_PATHS[@]}"; do
  sub_commit="$(git -C "${sub_path}" rev-parse HEAD)"
  [[ -n "${sub_json}" ]] && sub_json+=",\n"
  sub_json+="    \"${sub_path}\": \"${sub_commit}\""
done

mkdir -p -- "${PKG_DIR}/release"
cat > "${PKG_DIR}/release/dependency-manifest.json" <<EOF
{
  "version": "${VERSION}",
  "edge_dit_commit": "${EDGE_COMMIT}",
  "submodules": {
$(printf '%b' "${sub_json}")
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
for sub_path in "${SUBMODULE_PATHS[@]}"; do
  echo "${sub_path}=$(git -C "${sub_path}" rev-parse HEAD)"
done
