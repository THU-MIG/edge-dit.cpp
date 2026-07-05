#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

: "${EDGE_DIT_LIBRARY:?set EDGE_DIT_LIBRARY to the libedgedit.so path}"
: "${EDGE_DIT_MODEL_PATH:?set EDGE_DIT_MODEL_PATH to the model directory}"

EDGE_DIT_PYTHON=${EDGE_DIT_PYTHON:-/usr/bin/python3}
EDGE_DIT_CONFIG=${EDGE_DIT_CONFIG:-"${REPO_ROOT}/bindings/python/examples/flux_smoke_config.json"}
EDGE_DIT_OUTPUT=${EDGE_DIT_OUTPUT:-/tmp/edge_dit_python_smoke.png}

export PYTHONPATH="${REPO_ROOT}/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
export EDGE_DIT_LIBRARY
export EDGE_DIT_MODEL_PATH

"${EDGE_DIT_PYTHON}" \
  "${REPO_ROOT}/bindings/python/examples/configured_txt2img.py" \
  --config "${EDGE_DIT_CONFIG}" \
  --output "${EDGE_DIT_OUTPUT}"
