#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$REPO_ROOT/config/infer_v3dt_baseline.yaml}"
CAMERAS_CONFIG="${CAMERAS_CONFIG:-$REPO_ROOT/config/cameras_v3dt_baseline.yaml}"
CALIBRATION_CONFIG="${CALIBRATION_CONFIG:-$REPO_ROOT/config/camera_calibration.json}"
ALIGNMENT_CONFIG="${ALIGNMENT_CONFIG:-$REPO_ROOT/config/ply_alignment.json}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-}"

export REPO_ROOT
export PIPELINE_CONFIG

export NOESIS_V3DT_AUTOGEN_CAMINFO="${NOESIS_V3DT_AUTOGEN_CAMINFO:-0}"
export NOESIS_TRACKING_MODE="${NOESIS_TRACKING_MODE:-v3dt}"
export NOESIS_V3DT_DIAG_LOG=1
export NOESIS_V3DT_DIAG_SESSION="${NOESIS_V3DT_DIAG_SESSION:-v3dt_$(date +%Y%m%d_%H%M%S)}"

DIAG_ROOT="${NOESIS_V3DT_DIAG_DIR:-$REPO_ROOT/diagnostics/trackdump_runs}"
SESSION_DIR="$DIAG_ROOT/$NOESIS_V3DT_DIAG_SESSION"
export NOESIS_V3DT_DIAG_DIR="$SESSION_DIR"
mkdir -p "$SESSION_DIR"
LOG_FILE="$SESSION_DIR/run.log"
LOG_NDJSON="$SESSION_DIR/v3dt_frames_${NOESIS_V3DT_DIAG_SESSION}.ndjson"
touch "$LOG_FILE"
if [[ -s "$LOG_NDJSON" ]]; then
  echo "[capture] WARNING: existing log detected; analysis will include previous sessions unless filtered." | tee -a "$LOG_FILE"
fi

resolve_forensics_paths() {
  python3 - <<'PY'
import os
from pathlib import Path
import yaml

repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
pipeline_path = Path(os.environ.get("PIPELINE_CONFIG", "")).resolve()
tracker_path = ""
caminfo_dir = ""

try:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
except Exception:
    pipeline = {}

tracker_cfg = pipeline.get("tracker") or {}
raw_tracker = tracker_cfg.get("config-file") or tracker_cfg.get("ll-config-file") or ""
if raw_tracker:
    tpath = Path(raw_tracker)
    if not tpath.is_absolute():
        tpath = (repo_root / tpath).resolve()
    tracker_path = str(tpath)

if tracker_path:
    try:
        tracker = yaml.safe_load(Path(tracker_path).read_text(encoding="utf-8")) or {}
        omp = tracker.get("ObjectModelProjection") or {}
        cmf = omp.get("cameraModelFilepath") or []
        if isinstance(cmf, list) and cmf:
            p0 = Path(cmf[0])
            if not p0.is_absolute():
                p0 = (repo_root / p0).resolve()
            caminfo_dir = str(p0.parent)
    except Exception:
        caminfo_dir = ""

print(f"{tracker_path}\t{caminfo_dir}")
PY
}

IFS=$'\t' read -r TRACKER_CONFIG CAMINFO_DIR < <(resolve_forensics_paths)
if [[ -z "${CAMINFO_DIR:-}" ]]; then
  CAMINFO_DIR="$REPO_ROOT/config/v3dt"
fi

cat > "$SESSION_DIR/run_info.txt" <<EOF
timestamp=$(date -Iseconds)
pipeline_config=$PIPELINE_CONFIG
cameras_config=$CAMERAS_CONFIG
calibration_config=$CALIBRATION_CONFIG
alignment_config=$ALIGNMENT_CONFIG
tracker_config=$TRACKER_CONFIG
caminfo_dir=$CAMINFO_DIR
diag_dir=$NOESIS_V3DT_DIAG_DIR
diag_session=$NOESIS_V3DT_DIAG_SESSION
log_ndjson=$LOG_NDJSON
EOF
printenv | rg '^NOESIS_' >> "$SESSION_DIR/run_info.txt" || true

finalize() {
  echo "[capture] copying tracker dumps to $SESSION_DIR" | tee -a "$LOG_FILE"
  shopt -s nullglob
  for dump in /tmp/noesis_track_dump_*.txt; do
    cp -f "$dump" "$SESSION_DIR/"
  done
  echo "[capture] done" | tee -a "$LOG_FILE"
}
trap finalize EXIT INT TERM

echo "[capture] session=$NOESIS_V3DT_DIAG_SESSION" | tee -a "$LOG_FILE"
echo "[capture] pipeline=$PIPELINE_CONFIG" | tee -a "$LOG_FILE"
echo "[capture] cameras=$CAMERAS_CONFIG" | tee -a "$LOG_FILE"
echo "[capture] calibration=$CALIBRATION_CONFIG" | tee -a "$LOG_FILE"
echo "[capture] caminfo_dir=$CAMINFO_DIR" | tee -a "$LOG_FILE"
echo "[capture] log=$LOG_FILE" | tee -a "$LOG_FILE"

set +e
if [[ -n "$CAPTURE_SECONDS" ]]; then
  timeout --signal=INT --kill-after=5 "$CAPTURE_SECONDS" \
    python3 "$REPO_ROOT/noesis/ds8_runtime.py" \
      --pipeline-config "$PIPELINE_CONFIG" \
      --cameras-config "$CAMERAS_CONFIG" 2>&1 | tee -a "$LOG_FILE"
  RUNTIME_STATUS=${PIPESTATUS[0]}
else
  python3 "$REPO_ROOT/noesis/ds8_runtime.py" \
    --pipeline-config "$PIPELINE_CONFIG" \
    --cameras-config "$CAMERAS_CONFIG" 2>&1 | tee -a "$LOG_FILE"
  RUNTIME_STATUS=${PIPESTATUS[0]}
fi
set -e

EFFECTIVE_PIPELINE="$(ls -t "$REPO_ROOT"/build/effective_pipeline_*.yaml 2>/dev/null | head -n 1 || true)"
SNAPSHOT_PIPELINE="${EFFECTIVE_PIPELINE:-$PIPELINE_CONFIG}"

echo "[capture] runtime exited (status=$RUNTIME_STATUS). Running V3DT forensics..." | tee -a "$LOG_FILE"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SNAPSHOT_OUT="$(python3 "$REPO_ROOT/scripts/v3dt_forensics.py" snapshot \
  --pipeline-config "$SNAPSHOT_PIPELINE" \
  --cameras-config "$CAMERAS_CONFIG" \
  --calibration "$CALIBRATION_CONFIG" \
  --alignment "$ALIGNMENT_CONFIG" \
  --caminfo-dir "$CAMINFO_DIR" \
  --tracker-config "$TRACKER_CONFIG" \
  --output-dir "$SESSION_DIR" 2>&1)"
echo "$SNAPSHOT_OUT" | tee -a "$LOG_FILE"
SNAPSHOT_JSON="$(echo "$SNAPSHOT_OUT" | awk '/Wrote snapshot:/{print $3; exit}')"

if [[ -f "$LOG_NDJSON" && -n "$SNAPSHOT_JSON" ]]; then
  REPORT_OUT="$(python3 "$REPO_ROOT/scripts/v3dt_forensics.py" analyze \
    --log "$LOG_NDJSON" \
    --snapshot "$SNAPSHOT_JSON" \
    --scale-sweep auto \
    --output-dir "$SESSION_DIR" 2>&1)"
  echo "$REPORT_OUT" | tee -a "$LOG_FILE"
  REPORT_JSON="$(echo "$REPORT_OUT" | awk '/Wrote report:/{print $3; exit}')"
  if [[ -n "$REPORT_JSON" ]]; then
    PANEL_OUT="$(python3 "$REPO_ROOT/scripts/v3dt_forensics.py" panel \
      --snapshot "$SNAPSHOT_JSON" \
      --report "$REPORT_JSON" \
      --output-dir "$SESSION_DIR" 2>&1)"
    echo "$PANEL_OUT" | tee -a "$LOG_FILE"
  fi
else
  echo "[capture] skipping analysis (missing log or snapshot)." | tee -a "$LOG_FILE"
fi
