#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ds9_build_env.sh
source "${ROOT}/scripts/ds9_build_env.sh"

echo "[INFO] DS9 prep root: ${ROOT}"

DS_HOME="$(ds9_require_deepstream_home)"
echo "[OK] DeepStream 9 root: ${DS_HOME}"

CUDA_HOME="$(ds9_require_cuda_home)"
echo "[OK] CUDA root: ${CUDA_HOME}"

ds9_require_python_build_tools
echo "[OK] Python/build tooling available"

echo "[OK] DS9 prerequisites are present"
