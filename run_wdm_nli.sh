#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
        python3 -m venv "${ROOT_DIR}/.venv"
    fi
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/examples/test_wdm_nli.py"
