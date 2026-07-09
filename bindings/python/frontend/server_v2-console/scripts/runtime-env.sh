#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${APP_ROOT}/../../../.." && pwd)

export EDGE_DIT_REPO_ROOT="${EDGE_DIT_REPO_ROOT:-${REPO_ROOT}}"
export EDGE_DIT_PYTHON_BIN="${EDGE_DIT_PYTHON_BIN:-/usr/bin/python3}"
export EDGE_DIT_LIBRARY="${EDGE_DIT_LIBRARY:-${EDGE_DIT_REPO_ROOT}/build-cuda-shared/bin/libedgedit.so}"
export CUDNN_ROOT="${CUDNN_ROOT:-}"
export EDGE_DIT_DEPENDENCY_DIRS="${EDGE_DIT_DEPENDENCY_DIRS:-${EDGE_DIT_REPO_ROOT}/build-cuda-shared/bin}"
export PYTHONPATH="${EDGE_DIT_REPO_ROOT}/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
