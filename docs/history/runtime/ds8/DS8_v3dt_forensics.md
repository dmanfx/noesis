# DS8 V3DT Forensics Toolkit
_Status: current as of 2026-07-10._

This toolkit captures calibration + camInfo state, logs per-frame tracking telemetry, and generates an explicit report/panel so 3D tracking failures can be diagnosed from concrete data instead of guesswork.

All commands share one private artifact directory:

```bash
export NOESIS_V3DT_DIAG_DIR="$HOME/.local/state/noesis/diagnostics"
```

## 1) Capture a Snapshot (Calibration + Projection Math)

```bash
python3 scripts/v3dt_forensics.py snapshot \
  --pipeline-config build/effective_pipeline_yolo11_seg.yaml
```

Outputs:
- `$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.json`
- `$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.md`

The snapshot includes:
- Raw inputs: `config/cameras.yaml`, `config/camera_calibration.json`, `config/ply_alignment.json`, `config/v3dt/camInfo_*.yml`, effective pipeline config, and tracker config.
- Derived fields: scaled intrinsics, K matrices, P matrices, camera pose, axes, yaw/pitch/roll.
- Sanity checks: translation scale ratios, bottom-center ray → floor intersection (with/without Y-flip), and height mismatches.

## 2) Enable Runtime Telemetry Logging (NDJSON)

Set the env vars before running DS8:

```bash
export NOESIS_V3DT_DIAG_LOG=1
# Baseline defaults (tracking works on all cameras)
export NOESIS_V3DT_AUTOGEN_CAMINFO=1
export NOESIS_V3DT_CAMINFO_WORLD_SCALE=1
export NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p
export NOESIS_V3DT_CAMINFO_INVERT_E=1
export NOESIS_V3DT_CAMINFO_Y_FLIP=1
export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy
# Optional: name the session file
export NOESIS_V3DT_DIAG_SESSION=run_kitchen_001
python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium.yaml
```

Outputs:
- `$NOESIS_V3DT_DIAG_DIR/v3dt_frames_<session>.ndjson`

Each line contains a full per-frame payload with track data (including `bbox3d`, `velocity3d`, `visibility`, `image_foot`) and counts.

## 3) Analyze the Log

```bash
python3 scripts/v3dt_forensics.py analyze \
  --log "$NOESIS_V3DT_DIAG_DIR/v3dt_frames_run_kitchen_001.ndjson" \
  --snapshot "$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.json"
```

Outputs:
- `$NOESIS_V3DT_DIAG_DIR/v3dt_report_<timestamp>.json`
- `$NOESIS_V3DT_DIAG_DIR/v3dt_report_<timestamp>.md`

The report flags:
- Missing `bbox3d` coverage.
- Implausible 3D heights (uses `bbox3d.zLen` as height; SV3DT is Z-up).
- Reprojection error between the projected 3D **footpoint** and the 2D bbox bottom center.
- Track length statistics (fragmentation).
- Projection diagnostics: projected 3D box height/width vs the 2D bbox, plus bottom/top/center offsets.
  - Rotations (`xRot/yRot/zRot`) are recorded; the diagnostic projection assumes axis-aligned boxes.
- Depth consistency (Y-forward assumption): compares `bbox3d.yCentre` against the depth implied by
  the 2D bbox height using `fy` and `bbox3d.zLen`. Large ratios indicate depth bias per camera.

### Scale sweep (unit A/B test)

To quickly test whether the 3D outputs are off by a constant scale (cm↔m, etc.), run:

```bash
python3 scripts/v3dt_forensics.py analyze \
  --log "$NOESIS_V3DT_DIAG_DIR/v3dt_frames_run_kitchen_001.ndjson" \
  --snapshot "$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.json" \
  --scale-sweep auto
```

Default sweep scales (auto): `0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0`.

This scales the logged `bbox3d` values in analysis only (no runtime changes) and reports
median reprojection error + median height per scale so you can spot the most plausible unit.

## 4) Render the Panel

```bash
python3 scripts/v3dt_forensics.py panel \
  --snapshot "$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.json" \
  --report "$NOESIS_V3DT_DIAG_DIR/v3dt_report_<timestamp>.json"
```

Outputs:
- `$NOESIS_V3DT_DIAG_DIR/v3dt_panel_<timestamp>.html`

Open the HTML file locally to view the full report. For loopback-only viewing:

```bash
python3 scripts/v3dt_forensics.py serve --port 8777
```

Then open `http://127.0.0.1:8777/v3dt_panel_<timestamp>.html`.
The server refuses non-loopback binds and exposes only validated private panel
HTML files; it does not expose snapshots, reports, raw logs, or directory
listings.

## Artifact privacy and retention

The diagnostics directory must be owned by the service user with mode `0700`.
Every generated artifact is a single-link regular file with mode `0600`; the
tool refuses symlinks, hard links, public directories, and insecure existing
files instead of changing their permissions. Runtime session headers and
snapshots serialize only public pipeline source references and an explicit
non-secret environment allowlist. Materialized RTSP locators and auth values
are never serialized.

Runtime NDJSON stops accepting records at 64 MiB per session and retains at
most eight session logs by default. The bounds may be changed deliberately:

```bash
export NOESIS_V3DT_DIAG_MAX_BYTES=$((128 * 1024 * 1024))
export NOESIS_V3DT_DIAG_MAX_FILES=12
```

The accepted ranges are 1-512 MiB and 1-64 files. Capacity exhaustion drops
later records with a bounded warning; it never rolls into an undeclared file.

## 5) Raw Menon Payload Logging

Every well-formed `set_extrinsics` attempt records the raw payload
(pre-coercion) in the owner-private calibration audit directory:

```
$HOME/.local/state/noesis/calibration/raw/calibration_<timestamp>_<ns>_<camera>.json
```

Set `NOESIS_CALIBRATION_AUDIT_DIR` to another owner-only directory when needed.
Records capture the exact E/Twc/pose payload, unit-coercion notes, and the final
stored `E`; they are mode `0600` single-link files under a mode `0700`
directory. The latest 64 records are retained by default
(`NOESIS_CALIBRATION_AUDIT_MAX_FILES`, range 1-256). If the audit record cannot
be written safely, the calibration mutation fails with
`calibration_audit_failed` instead of proceeding without an audit trail.

## 6) Tilt-only preview from depth (pitch/roll only)

Use this when yaw/translation are trusted but tilt still looks off. The script fits
a ground plane from the **latest MapAnything depth snapshot** and adjusts pitch/roll
while preserving yaw and camera center. It writes a **preview** calibration file so
the main calibration remains untouched.

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera living-room \
  --camera family-room \
  --output config/camera_calibration_preview.json
```

If a camera improves with the **opposite** plane normal, override it explicitly:

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera living-room \
  --camera family-room \
  --normal-sign family-room=-1 \
  --output config/camera_calibration_preview.json
```

If pitch looks correct but roll is near ±180°, flip the image-space Y axis during
plane fitting (depth is image Y-down, calibration is Y-up):

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera living-room \
  --camera family-room \
  --flip-image-y \
  --output config/camera_calibration_preview.json
```

To test with SV3DT, generate camInfo from the preview calibration:

```bash
python3 scripts/generate_v3dt_caminfo.py \
  --pipeline-config config/infer_v3dt_medium.yaml \
  --calibration config/camera_calibration_preview.json \
  --output-dir config/v3dt_preview
```

Then either point the tracker camInfo list at `config/v3dt_preview/` or copy the
preview camInfo files into `config/v3dt/` for a one-off run.

If you are testing a **preview cameras.yaml** (for example, dewarped intrinsics),
pass it through to the tilt preview so the plane fit uses the same intrinsics:

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera family-room \
  --cameras-config config/cameras_preview_charuco_fr_rtsp_dewarp.yaml \
  --output config/camera_calibration_preview_dewarp.json
```

## Axis Remap (SV3DT Z-up)

If your calibration is Y-up but SV3DT expects Z-up, generate camInfo with an axis remap:

```bash
export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy
python3 scripts/generate_v3dt_caminfo.py --pipeline-config config/infer_v3dt_medium.yaml
```

Use the same env when running `snapshot` so the report matches the camInfo math.
