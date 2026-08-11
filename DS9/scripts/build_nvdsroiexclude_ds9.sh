#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ds9_build_env.sh
source "${ROOT}/scripts/ds9_build_env.sh"

DS_HOME="$(ds9_require_deepstream_home)"
ds9_require_python_build_tools

BUILD_DIR="${ROOT}/build/nvdsroiexclude"
SRC_DIR="${ROOT}/csrc/nvdsroiexclude"
OUT_DIR="${ROOT}/gst-plugins"

mkdir -p "${BUILD_DIR}" "${OUT_DIR}"
cmake --fresh -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEEPSTREAM_HOME="${DS_HOME}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT_DIR}"
cmake --build "${BUILD_DIR}" --parallel

echo "[OK] Built ${OUT_DIR}/libgstnvdsroiexclude.so"
echo "[INFO] DS9 build complete. Do not install over a non-DS9 runtime tree."
