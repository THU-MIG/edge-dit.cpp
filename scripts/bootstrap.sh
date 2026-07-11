#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: this directory is not a Git checkout." >&2
  echo "hint: GitHub auto-generated Source ZIP archives usually do not include submodule contents." >&2
  echo "recommended:" >&2
  echo "  git clone --recursive <repository-url>" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

git submodule update --init --recursive

if [[ ! -f third_party/ggml/CMakeLists.txt ]]; then
  echo "error: missing third_party/ggml/CMakeLists.txt after submodule update." >&2
  echo "hint: GitHub auto-generated Source ZIP archives usually do not include submodule contents." >&2
  echo "recommended:" >&2
  echo "  git clone --recursive <repository-url>" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

echo "bootstrap complete: third_party/ggml is present."
