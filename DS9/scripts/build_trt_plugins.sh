#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DS9_ROOT="${ROOT}/DS9"
PLUGIN_SRC="${ROOT}/external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg"
OUT_DIR="${DS9_ROOT}/plugins"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-9.1}"
VENDOR_DS_HOME="/opt/nvidia/deepstream/deepstream"
CUDA_VER="${CUDA_VER:-13.2}"
SM="${SM:-86}"

if [[ ! -d "${PLUGIN_SRC}" ]]; then
  echo "[FAIL] RF-DETR TensorRT plugin source missing: ${PLUGIN_SRC}" >&2
  exit 1
fi
if [[ ! -d "${DS_HOME}" ]] || [[ "$(readlink -f -- "${DS_HOME}")" != *"deepstream-9.1" ]]; then
  echo "[FAIL] DeepStream 9.1 home missing or incompatible: ${DS_HOME}" >&2
  exit 1
fi
if [[ "$(readlink -f -- "${VENDOR_DS_HOME}")" != "$(readlink -f -- "${DS_HOME}")" ]]; then
  echo "[FAIL] TensorRT plugin Makefile SDK link does not resolve to ${DS_HOME}: ${VENDOR_DS_HOME}" >&2
  exit 1
fi
if [[ ! -x "/usr/local/cuda-${CUDA_VER}/bin/nvcc" ]]; then
  echo "[FAIL] nvcc missing for CUDA_VER=${CUDA_VER}" >&2
  exit 1
fi

make -C "${PLUGIN_SRC}" clean CUDA_VER="${CUDA_VER}"
make -C "${PLUGIN_SRC}" CUDA_VER="${CUDA_VER}" SM="${SM}"

mkdir -p "${OUT_DIR}"
cp "${PLUGIN_SRC}/libnvdsinfer_custom_impl_Yolo_seg.so" "${OUT_DIR}/"
echo "[OK] Built DS9 TensorRT plugins in ${OUT_DIR}"
