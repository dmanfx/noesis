#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ds9_build_env.sh
source "${ROOT}/scripts/ds9_build_env.sh"

DS_HOME="$(ds9_require_deepstream_home)"
CUDA_HOME="$(ds9_require_cuda_home)"

make -C "${ROOT}/pipelines/nvdsinfer_yolo26_seg" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_yolo26_seg"
make -C "${ROOT}/pipelines/nvdsinfer_yolo11_seg" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_yolo11_seg"
make -C "${ROOT}/pipelines/nvdsinfer_yolo_detect" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_yolo_detect"
make -C "${ROOT}/pipelines/nvdsinfer_rfdetr" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_rfdetr"
make -C "${ROOT}/pipelines/nvdsinfer_rfdetr_seg" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_rfdetr_seg"
make -C "${ROOT}/pipelines/nvdsinfer_rfdetr_keypoint" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_rfdetr_keypoint"
make -C "${ROOT}/pipelines/nvdsinfer_deimv2_wholebody49" DS_HOME="${DS_HOME}" CUDA_HOME="${CUDA_HOME}" OUT_DIR="${ROOT}/pipelines/nvdsinfer_deimv2_wholebody49"
