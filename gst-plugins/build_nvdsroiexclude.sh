#!/usr/bin/env bash
set -euo pipefail

PLUGIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${PLUGIN_ROOT}/.." && pwd)"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-8.0}"
BUILD_DIR="${NOESIS_ROI_EXCLUDE_BUILD_DIR:-${REPO_ROOT}/build/nvdsroiexclude-ds8}"
OUT_DIR="${PLUGIN_ROOT}"

command -v cmake >/dev/null 2>&1 || {
  echo "[FAIL] cmake is required to build nvdsroiexclude" >&2
  exit 1
}
command -v pkg-config >/dev/null 2>&1 || {
  echo "[FAIL] pkg-config is required to build nvdsroiexclude" >&2
  exit 1
}
pkg-config --exists gstreamer-1.0 gstreamer-base-1.0 || {
  echo "[FAIL] Missing GStreamer development metadata" >&2
  exit 1
}

if [[ ! -d "${DS_HOME}" ]]; then
  echo "[FAIL] DeepStream 8 root not found: ${DS_HOME}" >&2
  exit 1
fi
DS_HOME="$(readlink -f -- "${DS_HOME}")"
VERSION_HEADER="${DS_HOME}/sources/includes/nvds_version.h"
for required in \
  "${VERSION_HEADER}" \
  "${DS_HOME}/sources/includes/nvdsmeta.h" \
  "${DS_HOME}/sources/includes/gstnvdsmeta.h" \
  "${DS_HOME}/lib/libnvds_meta.so" \
  "${DS_HOME}/lib/libnvdsgst_meta.so"; do
  if [[ ! -e "${required}" ]]; then
    echo "[FAIL] Missing required DeepStream 8 build input: ${required}" >&2
    exit 1
  fi
done
if ! grep -Eq '^#[[:space:]]*define[[:space:]]+NVDS_VERSION_MAJOR[[:space:]]+8([[:space:]]|$)' "${VERSION_HEADER}"; then
  echo "[FAIL] DS8 nvdsroiexclude requires DeepStream major 8: ${DS_HOME}" >&2
  exit 1
fi

cmake --fresh -S "${PLUGIN_ROOT}/nvdsroiexclude" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEEPSTREAM_HOME="${DS_HOME}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT_DIR}"
cmake --build "${BUILD_DIR}" --parallel

echo "[OK] Built DS8 ${OUT_DIR}/libgstnvdsroiexclude.so"
