#!/usr/bin/env bash
set -euo pipefail

PLUGIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd -- "${PLUGIN_ROOT}/.." && pwd)"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-9.1}"
BUILD_DIR="${NOESIS_FORCE_IDR_BUILD_DIR:-${DS9_ROOT}/build/noesisforceidr-ds9}"
OUT_DIR="${PLUGIN_ROOT}"

if [[ ! -f "${DS_HOME}/sources/includes/gst-nvcustomevent.h" ]]; then
  echo "[FAIL] Missing NVIDIA custom-event header under ${DS_HOME}" >&2
  exit 1
fi
if [[ ! -f "${DS_HOME}/lib/libgstnvcustomhelper.so" ]]; then
  echo "[FAIL] Missing NVIDIA custom-event library under ${DS_HOME}" >&2
  exit 1
fi
if [[ "$(readlink -f -- "${DS_HOME}")" != *"deepstream-9.1" ]]; then
  echo "[FAIL] DS9 plugin must be built against deepstream-9.1: ${DS_HOME}" >&2
  exit 1
fi

cmake --fresh -S "${PLUGIN_ROOT}/noesisforceidr" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEEPSTREAM_HOME="${DS_HOME}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT_DIR}"
cmake --build "${BUILD_DIR}" --parallel

echo "[OK] Built DS9 ${OUT_DIR}/libgstnoesisforceidr.so"
