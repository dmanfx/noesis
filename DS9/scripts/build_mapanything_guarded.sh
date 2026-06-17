#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DS9_ROOT}/.." && pwd)"

IMAGE="${NOESIS_DS9_IMAGE:-nvcr.io/nvidia/deepstream:9.0-triton-multiarch}"
PYTHON_BIN="${NOESIS_MAPANYTHING_PYTHON:-python3}"
GPU_LIMIT_MB="${NOESIS_MAPANYTHING_GPU_GUARD_MB:-9000}"
TIMEOUT_SECONDS="${NOESIS_MAPANYTHING_TRT_TIMEOUT_SECONDS:-1800}"
POLL_SECONDS="${NOESIS_MAPANYTHING_GUARD_POLL_SECONDS:-3}"
CONTAINER_NAME="noesis-mapanything-trt-$$"
ONNX_PATH="${REPO_ROOT}/DS9/models/onnx/mapanything_images_294x518_b3.onnx"
PLAN_PATH="${REPO_ROOT}/DS9/models/engines/mapanything_images_294x518_b3_fp16.plan"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup INT TERM

"${PYTHON_BIN}" - "${ONNX_PATH}" <<'PY'
from pathlib import Path
import sys
import onnx

onnx_path = Path(sys.argv[1])
if not onnx_path.exists() or onnx_path.stat().st_size <= 0:
    raise SystemExit(f"missing ONNX: {onnx_path}")

model = onnx.load(str(onnx_path), load_external_data=False)
missing = []
total = 0
for tensor in model.graph.initializer:
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        total += len(tensor.raw_data)
        continue
    data = {item.key: item.value for item in tensor.external_data}
    location = data.get("location")
    if not location:
        missing.append(f"{tensor.name}: missing external location")
        continue
    sidecar = onnx_path.parent / location
    if not sidecar.exists() or sidecar.stat().st_size <= 0:
        missing.append(str(sidecar))
    else:
        total += sidecar.stat().st_size

if missing:
    raise SystemExit("missing ONNX external tensor data:\n" + "\n".join(missing[:25]))
if total < 1024**3:
    raise SystemExit(f"ONNX tensor data too small for MapAnything: {total} bytes")
print(f"[OK] ONNX external tensor data verified: {total / 1024**3:.2f} GiB")
PY

rm -f "${PLAN_PATH}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --entrypoint /bin/bash \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  -lc 'set -euo pipefail; python3 DS9/scripts/rebuild_engines.py --only mapanything --include-mapanything; test -s DS9/models/engines/mapanything_images_294x518_b3_fp16.plan; ls -lh DS9/models/engines/mapanything_images_294x518_b3_fp16.plan' >/dev/null

start_time="$(date +%s)"
while [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo false)" == "true" ]]; do
  gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 {print int($1)}')"
  now="$(date +%s)"
  if (( gpu_used > GPU_LIMIT_MB )); then
    echo "[GUARD] GPU memory ${gpu_used} MiB exceeded limit ${GPU_LIMIT_MB} MiB; stopping build." >&2
    docker logs --tail 120 "${CONTAINER_NAME}" >&2 || true
    cleanup
    exit 42
  fi
  if (( now - start_time > TIMEOUT_SECONDS )); then
    echo "[GUARD] Build exceeded timeout ${TIMEOUT_SECONDS}s; stopping build." >&2
    docker logs --tail 120 "${CONTAINER_NAME}" >&2 || true
    cleanup
    exit 43
  fi
  sleep "${POLL_SECONDS}"
done

status="$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo 125)"
docker logs "${CONTAINER_NAME}" || true
docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
exit "${status}"
