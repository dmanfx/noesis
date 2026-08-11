#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python3 not found in PATH" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-logs/runtime_smoke}"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_LOG="${LOG_DIR}/run_${RUN_ID}.log"

log() {
  local level="$1"; shift
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf "%s [%s] %s\n" "${ts}" "${level}" "$*" | tee -a "${RUN_LOG}"
}

declare -a CHECK_SUMMARY=()
FAIL_COUNT=0
WARN_COUNT=0

record_check() {
  local status="$1"
  local name="$2"
  local detail="$3"
  CHECK_SUMMARY+=("${status}|${name}|${detail}")
  case "${status}" in
    PASS)
      log "PASS" "${name}: ${detail}"
      ;;
    WARN)
      WARN_COUNT=$((WARN_COUNT + 1))
      log "WARN" "${name}: ${detail}"
      ;;
    FAIL)
      FAIL_COUNT=$((FAIL_COUNT + 1))
      log "FAIL" "${name}: ${detail}"
      ;;
    *)
      log "INFO" "${name}: ${detail}"
      ;;
  esac
}

CONFIG_PATH="config/infer.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  record_check "FAIL" "config/infer.yaml" "Configuration file missing at ${CONFIG_PATH}"
  exit 1
fi

PIPELINE_PATH="noesis/pipelines/ds8_pipeline.py"
if [[ -f "${PIPELINE_PATH}" ]]; then
  record_check "PASS" "Pipeline Scaffold" "Found ${PIPELINE_PATH}"
else
  record_check "FAIL" "Pipeline Scaffold" "Missing ${PIPELINE_PATH}"
fi

mapfile -t ENGINE_PATHS < <("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
with open(cfg_path, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

pgie_engine = cfg.get("models", {}).get("pgie", {}).get("engine", "")
map_engine = cfg.get("models", {}).get("mapanything", {}).get("engine", "")
print(pgie_engine)
print(map_engine)
PY
)

PGIE_ENGINE="${ENGINE_PATHS[0]:-}"
MA_ENGINE="${ENGINE_PATHS[1]:-}"

if [[ -n "${PGIE_ENGINE}" ]] && [[ -f "${PGIE_ENGINE}" ]]; then
  record_check "PASS" "YOLOv11 Engine" "Located ${PGIE_ENGINE}"
else
  record_check "WARN" "YOLOv11 Engine" "Expected engine at ${PGIE_ENGINE:-<unset>} (build may be pending)"
fi

if [[ -n "${MA_ENGINE}" ]] && [[ -f "${MA_ENGINE}" ]]; then
  record_check "PASS" "MapAnything Engine" "Located ${MA_ENGINE}"
else
  record_check "WARN" "MapAnything Engine" "Expected engine at ${MA_ENGINE:-<unset>} (build may be pending)"
fi

if "${PYTHON_BIN}" -c "import uvicorn" >/dev/null 2>&1; then
  record_check "PASS" "uvicorn Import" "uvicorn module available"
else
  record_check "FAIL" "uvicorn Import" "uvicorn not importable by ${PYTHON_BIN}"
fi

if "${PYTHON_BIN}" -c "import fastapi" >/dev/null 2>&1; then
  record_check "PASS" "fastapi Import" "fastapi module available"
else
  record_check "FAIL" "fastapi Import" "fastapi not importable by ${PYTHON_BIN}"
fi

SMOKE_PORT="${SMOKE_PORT:-8900}"
DEPTH_SECONDS="${DEPTH_SECONDS:-20}"
PY_RUNTIME_LOG="${LOG_DIR}/runtime_${RUN_ID}.log"

log "INFO" "Starting embedded runtime smoke for ${DEPTH_SECONDS}s depth burst on port ${SMOKE_PORT}"

set +e
"${PYTHON_BIN}" - "${SMOKE_PORT}" "${DEPTH_SECONDS}" "${ROOT_DIR}" <<'PY' | tee "${PY_RUNTIME_LOG}"
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

port = int(sys.argv[1])
burst_seconds = int(sys.argv[2])
root = Path(sys.argv[3]).resolve()
os.chdir(root)

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

checks_fail = {"value": False}

def emit_check(name: str, status: str, detail: str) -> None:
    if status.upper() == "FAIL":
        checks_fail["value"] = True
    print(f"CHECK|{name}|{status.upper()}|{detail}")

def emit_gpu(sample: dict) -> None:
    label = sample.get("label", "?")
    status = sample.get("status", "SKIP")
    util = sample.get("util", "")
    mem = sample.get("mem", "")
    ts = sample.get("ts", "")
    detail = sample.get("detail") or sample.get("raw", "")
    print(f"GPU|{label}|{status}|{util}|{mem}|{ts}|{detail}")

try:
    from noesis.pipelines import ds8_pipeline
    from noesis.server.depth_api import app
    from noesis.metadata.depth_result import DepthResult
except Exception as exc:
    emit_check("imports", "FAIL", f"Import error: {exc}")
    sys.exit(1)

pipeline_built = False
try:
    pipeline_obj = ds8_pipeline.build_pipeline(root / "config" / "infer.yaml")
    ds8_pipeline.prepare()
    ds8_pipeline.activate()
    pipeline_built = True
    emit_check("pipeline_build", "PASS", "Pipeline graph built and activated")
except Exception as exc:
    emit_check("pipeline_build", "FAIL", f"Failed to build pipeline: {exc}")

if not pipeline_built:
    sys.exit(1)

components = getattr(pipeline_obj, "components", {}) or {}
source_nodes = [name for name in components if name.startswith("source")]
pgie_component = components.get("yolo11_pgie")
map_name = pipeline_obj.config.get("models", {}).get("mapanything", {}).get("name", "mapanything_fullframe")
map_component = components.get(map_name)

pgie_batch = (pgie_component.config if pgie_component else {}).get("batch_size")
ma_batch = (map_component.config if map_component else {}).get("batch_size")

if len(source_nodes) == pgie_batch == ma_batch == 3:
    emit_check("batch_meta_alignment", "PASS", f"{len(source_nodes)} sources aligned with batch size {pgie_batch}")
else:
    emit_check("batch_meta_alignment", "FAIL", f"sources={len(source_nodes)}, pgie_batch={pgie_batch}, map_batch={ma_batch}")

pgie_id = (pgie_component.config if pgie_component else {}).get("gie_id")
ma_id = (map_component.config if map_component else {}).get("gie_id")
if pgie_id is not None and ma_id is not None and pgie_id != ma_id:
    emit_check("meta_id_uniqueness", "PASS", f"pgie={pgie_id}, mapanything={ma_id}")
else:
    emit_check("meta_id_uniqueness", "FAIL", f"gie ids overlap? pgie={pgie_id}, mapanything={ma_id}")

camera_yaml = root / "config" / "cameras.yaml"
if camera_yaml.exists():
    emit_check("camera_intrinsics_config", "PASS", f"Found {camera_yaml}")
else:
    emit_check("camera_intrinsics_config", "WARN", f"Camera intrinsics config missing at {camera_yaml}")

try:
    from noesis.metadata import intrinsics as intrinsics_mod  # type: ignore
    attach_fn = getattr(intrinsics_mod, "attach_intrinsics", None)
    if callable(attach_fn):
        emit_check("intrinsics_helper", "PASS", "attach_intrinsics callable")
    else:
        emit_check("intrinsics_helper", "WARN", "attach_intrinsics helper not implemented")
except Exception as exc:
    emit_check("intrinsics_helper", "WARN", f"Intrinsics helper import failed: {exc}")

try:
    sample_depth = DepthResult(
        source_id=0,
        frame_id=123,
        ts=int(time.time()),
        width=640,
        height=384,
        depth_map_ref="s3://bucket/frame_123.depth",
        minmax=(0.2, 12.5),
    )
    round_trip = DepthResult.from_json(sample_depth.to_json())
    if round_trip == sample_depth:
        emit_check("depth_result_schema", "PASS", "DepthResult round-trip succeeded")
    else:
        emit_check("depth_result_schema", "FAIL", "DepthResult round-trip mismatch")
except Exception as exc:
    emit_check("depth_result_schema", "FAIL", f"DepthResult validation error: {exc}")

import uvicorn

server_config = uvicorn.Config(
    app,
    host="127.0.0.1",
    port=port,
    log_level="warning",
)
server = uvicorn.Server(server_config)
server.install_signal_handlers = lambda: None

server_thread = threading.Thread(target=server.run, name="uvicorn-smoke", daemon=True)
server_thread.start()

startup_deadline = time.time() + 15
while not server.started:
    if not server_thread.is_alive():
        emit_check("depth_api_start", "FAIL", "uvicorn thread exited during startup")
        sys.exit(1)
    if time.time() > startup_deadline:
        emit_check("depth_api_start", "FAIL", "depth API startup timed out")
        server.should_exit = True
        server_thread.join(timeout=2)
        sys.exit(1)
    time.sleep(0.1)

emit_check("depth_api_start", "PASS", f"http://127.0.0.1:{port}")

has_nvidia = shutil.which("nvidia-smi") is not None

def take_gpu_sample(label: str) -> dict:
    if not has_nvidia:
        sample = {"label": label, "status": "SKIP", "detail": "nvidia-smi unavailable"}
        emit_gpu(sample)
        return sample
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        sample = {"label": label, "status": "WARN", "detail": str(exc)}
        emit_gpu(sample)
        return sample
    line = output.strip().splitlines()[0] if output.strip() else ""
    if not line:
        sample = {"label": label, "status": "WARN", "detail": "no GPU data returned"}
        emit_gpu(sample)
        return sample
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        sample = {"label": label, "status": "WARN", "detail": f"unexpected format: {line}"}
        emit_gpu(sample)
        return sample
    try:
        util = float(parts[1])
    except ValueError:
        util = None
    try:
        mem = float(parts[2])
    except ValueError:
        mem = None
    sample = {
        "label": label,
        "status": "OK",
        "ts": parts[0],
        "util": util,
        "mem": mem,
        "raw": line,
    }
    emit_gpu(sample)
    return sample

idle_sample = take_gpu_sample("idle")

depth_url = f"http://127.0.0.1:{port}/api/v1/depth/refresh?seconds={burst_seconds}"
before_enabled = ds8_pipeline.get_pipeline().depth_enabled

try:
    with urllib.request.urlopen(depth_url, timeout=5) as response:
        payload = response.read().decode("utf-8")
        status_code = response.status
except Exception as exc:
    emit_check("depth_refresh_call", "FAIL", f"HTTP error: {exc}")
    server.should_exit = True
    server_thread.join(timeout=2)
    sys.exit(1)

if status_code != 200:
    emit_check("depth_refresh_call", "FAIL", f"HTTP {status_code}")
else:
    emit_check("depth_refresh_call", "PASS", f"HTTP {status_code}")

depth_payload = {}
try:
    depth_payload = json.loads(payload)
    seconds_reported = depth_payload.get("seconds")
    emit_check("depth_refresh_payload", "PASS", f"seconds={seconds_reported}, enabled={depth_payload.get('enabled')}")
except Exception as exc:
    emit_check("depth_refresh_payload", "FAIL", f"JSON decode error: {exc}")

after_enabled = ds8_pipeline.get_pipeline().depth_enabled

mid_wait = max(1.0, min(5.0, burst_seconds / 2.0))
time.sleep(mid_wait)
burst_sample = take_gpu_sample("burst")

remaining_wait = max(1.0, burst_seconds - mid_wait + 1.0)
time.sleep(remaining_wait)
post_sample = take_gpu_sample("post")
final_enabled = ds8_pipeline.get_pipeline().depth_enabled

if not before_enabled and after_enabled and not final_enabled:
    emit_check("depth_toggle_window", "PASS", f"depth_enabled toggled True for ~{burst_seconds}s")
else:
    emit_check(
        "depth_toggle_window",
        "FAIL",
        f"states before={before_enabled}, after={after_enabled}, final={final_enabled}",
    )

gpu_status = "WARN"
gpu_detail = "nvidia-smi unavailable"
if idle_sample.get("status") == burst_sample.get("status") == post_sample.get("status") == "OK":
    util_idle = idle_sample.get("util") or 0.0
    util_burst = burst_sample.get("util") or 0.0
    util_post = post_sample.get("util") or 0.0
    if util_burst > util_idle + 1 or util_burst > util_post + 1:
        gpu_status = "PASS"
        gpu_detail = f"util idle={util_idle}, burst={util_burst}, post={util_post}"
    else:
        gpu_status = "WARN"
        gpu_detail = f"No utilization delta (idle={util_idle}, burst={util_burst}, post={util_post})"
elif not has_nvidia:
    gpu_status = "WARN"
    gpu_detail = "nvidia-smi not available"
else:
    gpu_status = "WARN"
    gpu_detail = "Incomplete GPU samples"

emit_check("gpu_profile", gpu_status, gpu_detail)

server.should_exit = True
server_thread.join(timeout=5)
if server_thread.is_alive():
    server.force_exit = True  # type: ignore[attr-defined]
    server_thread.join(timeout=2)

sys.exit(1 if checks_fail["value"] else 0)
PY
PYTHON_RC=${PIPESTATUS[0]}
set -e

while IFS='|' read -r prefix name status detail; do
  case "${prefix}" in
    CHECK)
      record_check "${status}" "Runtime: ${name}" "${detail}"
      ;;
    GPU)
      log "INFO" "GPU sample (${name}): status=${status}, util=${detail%%|*}"
      ;;
  esac
done < <(grep -E '^(CHECK|GPU)\|' "${PY_RUNTIME_LOG}" || true)

log "INFO" "Runtime smoke python exit code: ${PYTHON_RC}"
if [[ "${PYTHON_RC}" -ne 0 ]]; then
  CHECK_SUMMARY+=("FAIL|Runtime orchestration|Python runtime checks failed")
  FAIL_COUNT=$((FAIL_COUNT + 1))
fi

log "INFO" "----- Smoke Test Summary -----"
for entry in "${CHECK_SUMMARY[@]}"; do
  IFS='|' read -r status name detail <<< "${entry}"
  printf "%-4s %-32s %s\n" "${status}" "${name}" "${detail}" | tee -a "${RUN_LOG}"
done

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  log "INFO" "Smoke test completed with ${FAIL_COUNT} failure(s) and ${WARN_COUNT} warning(s)"
  exit 1
fi

log "INFO" "Smoke test completed successfully with ${WARN_COUNT} warning(s)"
exit 0
