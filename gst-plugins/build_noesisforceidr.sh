#!/usr/bin/env bash
set -euo pipefail

PLUGIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${PLUGIN_ROOT}/.." && pwd)"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-8.0}"
BUILD_DIR="${NOESIS_FORCE_IDR_BUILD_DIR:-${REPO_ROOT}/build/noesisforceidr-ds8}"
OUT_DIR="${PLUGIN_ROOT}"

if [[ ! -f "${DS_HOME}/sources/includes/gst-nvcustomevent.h" ]]; then
  echo "[FAIL] Missing NVIDIA custom-event header under ${DS_HOME}" >&2
  exit 1
fi
if [[ ! -f "${DS_HOME}/lib/libgstnvcustomhelper.so" ]]; then
  echo "[FAIL] Missing NVIDIA custom-event library under ${DS_HOME}" >&2
  exit 1
fi
if [[ "$(readlink -f -- "${DS_HOME}")" != *"deepstream-8.0" ]]; then
  echo "[FAIL] DS8 plugin must be built against deepstream-8.0: ${DS_HOME}" >&2
  exit 1
fi

cmake -S "${PLUGIN_ROOT}/noesisforceidr" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEEPSTREAM_HOME="${DS_HOME}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT_DIR}"
cmake --build "${BUILD_DIR}" --parallel

echo "[OK] Built DS8 ${OUT_DIR}/libgstnoesisforceidr.so"
