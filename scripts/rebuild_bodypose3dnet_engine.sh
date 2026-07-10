#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

ONNX_PATH="${ONNX_PATH:-${ROOT}/models/bodypose3dnet/bodypose3dnet_accuracy.onnx}"
ENGINE_PATH="${ENGINE_PATH:-${ROOT}/models/engines/bodypose3dnet_accuracy_b1_fp16.engine}"
TRTEXEC="${TRTEXEC:-trtexec}"
WORKSPACE_MIB="${WORKSPACE_MIB:-4096}"
SKIP_LOAD_TEST="${SKIP_LOAD_TEST:-0}"

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
  echo "[FAIL] Missing trtexec on PATH. Set TRTEXEC=/path/to/trtexec if needed." >&2
  exit 1
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[FAIL] Missing BodyPose3DNet ONNX: ${ONNX_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${ENGINE_PATH}")"

# Keep this static profile aligned with config/v3dt/* PoseEstimator:
# batchSize=1, networkMode=1 (FP16), inferDims=[3,256,192].
SHAPES="input0:1x3x256x192,k_inv:1x3x3,t_form_inv:1x3x3,scale_normalized_mean_limb_lengths:1x36,mean_limb_lengths:1x36"

BUILD_CMD=(
  "${TRTEXEC}"
  "--onnx=${ONNX_PATH}"
  "--fp16"
  "--saveEngine=${ENGINE_PATH}"
  "--minShapes=${SHAPES}"
  "--optShapes=${SHAPES}"
  "--maxShapes=${SHAPES}"
  "--memPoolSize=workspace:${WORKSPACE_MIB}"
  "--skipInference"
)

printf '[INFO] Building BodyPose3DNet engine:'
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'
"${BUILD_CMD[@]}"

if [[ "${SKIP_LOAD_TEST}" == "1" ]]; then
  echo "[OK] Built ${ENGINE_PATH}"
  exit 0
fi

LOAD_CMD=(
  "${TRTEXEC}"
  "--loadEngine=${ENGINE_PATH}"
  "--duration=1"
  "--warmUp=0"
  "--iterations=1"
  "--noDataTransfers"
)

printf '[INFO] Validating TensorRT load:'
printf ' %q' "${LOAD_CMD[@]}"
printf '\n'
"${LOAD_CMD[@]}"

echo "[OK] Built and validated ${ENGINE_PATH}"
