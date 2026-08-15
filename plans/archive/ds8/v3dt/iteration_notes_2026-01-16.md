# V3DT Iteration Notes — 2026-01-16 (Family-room full-FOV dewarp)

**Status (2026-01-22):** Historical log; superseded by the locked baseline in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`. Commands here use
older camInfo polarity and are not the current baseline; revalidate before reuse.

## 1) Full-FOV dewarper K

- `config/dewarper_family_room_charuco_rtsp.txt`: set `dst-focal-length` to `768.715/768.432` and `dst-principal-point` to `640/360` (full FOV, centered).
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`: updated family-room rectified intrinsics to match.

## 2) Pre-tilt capture

- Run:
  - `NOESIS_V3DT_AUTOGEN_CAMINFO=0 NOESIS_V3DT_DIAG_LOG=1 NOESIS_V3DT_DIAG_DIR=diagnostics NOESIS_V3DT_DIAG_SESSION=fr_fullfov_pretilt timeout 90s python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml --cameras-config config/cameras_preview_charuco_fr_rtsp_dewarp.yaml --disable-rest --depth-enable-seconds 25`
- Log: `diagnostics/v3dt_frames_fr_fullfov_pretilt.ndjson`

## 3) Tilt preview from depth

- Run:
  - `python3 scripts/auto_tilt_from_depth.py --camera family-room --cameras-config config/cameras_preview_charuco_fr_rtsp_dewarp.yaml --flip-image-y --input config/camera_calibration.json --output config/camera_calibration_preview_dewarp_fullfov_tilt.json`
- Result: pitch **-25.66°**, roll **0.09°**, `candidate_sign=-1`, tilt angle **20.66°**.

## 4) Regenerate camInfo (preview)

- Env: `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p NOESIS_V3DT_CAMINFO_INVERT_E=1 NOESIS_V3DT_CAMINFO_Y_FLIP=0 NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
- Output: `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_{living-room,kitchen,family-room}.yml`

## 5) Post-tilt capture + forensics

- Log: `diagnostics/v3dt_frames_fr_fullfov_posttilt.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260116_164114.json`
- Report: `diagnostics/v3dt_report_20260116_164123.md`
- Metrics (family-room):
  - `bbox3d%`: 100%
  - reprojection median: **99.2 px**
  - height ratio median: **0.40**
  - width ratio median: **0.58**
  - depth ratio median: **0.44**
  - scale sweep best: **1.0** (no scalar fix)

## 6) Notes

- Living-room/kitchen reported 0 tracks in this run (sources may be idle).
- Full-FOV dewarp fixed center-crop but cuboid size remains small; likely still an extrinsics/pitch or mapping mismatch.

## 7) Charuco principal point + camInfo polarity checks

- Updated Charuco center:
  - `config/dewarper_family_room_charuco_rtsp.txt`: `dst-principal-point=641.3681223147197;356.804752740055`
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`: family-room `cx=641.3681223147197`, `cy=356.804752740055`
- CamInfo generation polarity sweep (offline snapshot):
  - Invert E **required** for tracking; non-inverted camInfo produced zero tracks in live runs.
  - Y-flip **improves reprojection** (see below).

## 8) Canonical full-FOV baseline (best so far)

- Tilt refresh (depth, flip-image-y):
  - `config/camera_calibration_preview_dewarp_fullfov_tilt.json` (pitch **-23.48°**, roll **3.01°**, `candidate_sign=+1`)
- CamInfo generation:
  - `NOESIS_V3DT_CAMINFO_INVERT_E=1`
  - `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
  - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
- Run + report:
  - Log: `diagnostics/v3dt_frames_fr_fullfov_yflip1.ndjson`
  - Snapshot: `diagnostics/v3dt_snapshot_20260116_174216.json`
  - Report: `diagnostics/v3dt_report_20260116_174224.md`
  - Metrics (family-room): reproj **26.7 px**, H ratio **0.58**, W ratio **1.28**, bottom offset **0.6 px**
  - Interpretation: footpoint alignment is good; box height still short (size error varies with vertical position).

## 9) Axis/pitch permutations (ruled out)

- Axis mapping `xyz` (no swap) worsened height ratio:
  - Report: `diagnostics/v3dt_report_20260116_173945.md` (H ratio **0.35**)
- Manual pitch -5° offset (extra downward tilt) did **not** improve height ratio:
  - Calibration: `config/camera_calibration_preview_dewarp_fullfov_tilt_pitchm5.json`
  - Report: `diagnostics/v3dt_report_20260116_174900.md` (H ratio **0.56**, reproj **33.2 px**)

## 10) Tilt preview + MapAnything intrinsics alignment (2026-01-16)

- Updated tilt preview to **prefer `--cameras-config` intrinsics** (Charuco preview now respected instead of `intrinsics.json`).
- Tilt preview output now includes:
  - intrinsics source + scaled K (fx/fy/cx/cy)
  - depth snapshot timestamp + age (quick sanity check for stale depth).
- DS8 calibration bundle now **overrides K using `cameras-config`** intrinsics for MapAnything,
  scaled to streammux size (when base resolution is known or inferred).
  - Ensures MapAnything depth uses the same Charuco K as SV3DT during the dewarp preview workflow.

## 11) Dewarper frame dump disabled (2026-01-16)

- `config/dewarper_family_room_charuco_rtsp.txt`: set `dewarp-dump-frames=0` to stop per-frame RGBA dumps.
- Reason: frame dumps were saturating IO/GPU and collapsing FPS during SV3DT runs.
