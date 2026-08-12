#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DS9_ROOT="${ROOT}/DS9"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-9.1}"
BUILD_DIR="${DS9_ROOT}/build/nvdsroiexclude"
OUT_DIR="${DS9_ROOT}/gst-plugins"

if [[ ! -d "${DS_HOME}" ]]; then
  echo "[FAIL] DeepStream home not found: ${DS_HOME}" >&2
  exit 1
fi
if [[ "$(readlink -f -- "${DS_HOME}")" != *"deepstream-9.1" ]]; then
  echo "[FAIL] DS9 plugin must be built against DeepStream 9.1: ${DS_HOME}" >&2
  exit 1
fi
if [[ ! -f "${DS_HOME}/sources/includes/gstnvdsmeta.h" ]]; then
  echo "[FAIL] Missing DeepStream metadata headers under ${DS_HOME}" >&2
  exit 1
fi
if ! grep -Eq 'NVDS_VERSION_MAJOR[[:space:]]+9' "${DS_HOME}/sources/includes/nvds_version.h" \
    || ! grep -Eq 'NVDS_VERSION_MINOR[[:space:]]+1' "${DS_HOME}/sources/includes/nvds_version.h"; then
  echo "[FAIL] DeepStream headers do not identify as version 9.1: ${DS_HOME}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${OUT_DIR}"
cmake --fresh -S "${DS9_ROOT}/csrc/nvdsroiexclude" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEEPSTREAM_HOME="${DS_HOME}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT_DIR}"
cmake --build "${BUILD_DIR}" --parallel

NOESIS_DEEPSTREAM_HOME="${DS_HOME}" \
  "${DS9_ROOT}/gst-plugins/build_noesisforceidr.sh"
NOESIS_DEEPSTREAM_HOME="${DS_HOME}" \
  "${DS9_ROOT}/gst-plugins/build_noesiseos.sh"

echo "[OK] Built DS9 GStreamer plugins in ${OUT_DIR}"
