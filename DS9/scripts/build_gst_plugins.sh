#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DS9_ROOT="${ROOT}/DS9"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-9.0}"
BUILD_DIR="${DS9_ROOT}/build/nvdsroiexclude"
OUT_DIR="${DS9_ROOT}/gst-plugins"

if [[ ! -d "${DS_HOME}" ]]; then
  echo "[FAIL] DeepStream home not found: ${DS_HOME}" >&2
  exit 1
fi
if [[ "${DS_HOME}" == *"deepstream-8.0"* ]]; then
  echo "[FAIL] Refusing to build DS9 plugin against DS8: ${DS_HOME}" >&2
  exit 1
fi
if [[ ! -f "${DS_HOME}/sources/includes/gstnvdsmeta.h" ]]; then
  echo "[FAIL] Missing DeepStream metadata headers under ${DS_HOME}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${OUT_DIR}"
cmake -S "${DS9_ROOT}/csrc/nvdsroiexclude" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT_DIR}"
cmake --build "${BUILD_DIR}" --parallel

echo "[OK] Built DS9 GStreamer plugins in ${OUT_DIR}"
