# V3DT Iteration Notes — 2026-01-15 (Family-room dewarp tuning)

**Status (2026-01-22):** Historical log; superseded by the locked baseline in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`. Commands here use
older camInfo polarity and are not the current baseline; revalidate before reuse.

## 1) Depth capture + tilt candidates (family-room only)

- Depth capture run:
  - `NOESIS_V3DT_AUTOGEN_CAMINFO=0 NOESIS_V3DT_DIAG_LOG=1 NOESIS_V3DT_DIAG_DIR=diagnostics NOESIS_V3DT_DIAG_SESSION=fr_dewarp_iter2 NOESIS_WS_PORT=6055 timeout 90s python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml --cameras-config config/cameras_preview_charuco_fr_rtsp_dewarp.yaml --disable-rest --depth-enable-seconds 25`
  - Log: `diagnostics/v3dt_frames_fr_dewarp_iter2.ndjson`
- Tilt preview (family-room):
  - `python3 scripts/auto_tilt_from_depth.py --camera family-room --cameras-config config/cameras_preview_charuco_fr_rtsp_dewarp.yaml --flip-image-y --no-write`
  - Candidate sign: **-1** (pitch -23.91°, roll -1.13°, score 1.66)
  - Candidate sign +1 produced pitch +29°, roll -179° (invalid orientation)
- Preview written:
  - `config/camera_calibration_preview_dewarp.json` (from sign -1 result)

## 2) Baseline after tilt (rectified K + dewarper dst K)

- Log: `diagnostics/v3dt_frames_fr_dewarp_iter3.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260115_190957.json`
- Report: `diagnostics/v3dt_report_20260115_191004.md`
- Family-room: H ratio ~0.42, reproj ~36px (still short)

## 3) Intrinsics scaling experiments (family-room)

### 3a) Scale dewarper dst K + camInfo K (2.4x)

- Changes:
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` fx/fy = 1858.85 / 1858.18
  - `config/dewarper_family_room_charuco_rtsp.txt` dst-focal-length = 1858.85 / 1858.18
- Log: `diagnostics/v3dt_frames_fr_dewarp_iter4_fx2p4.ndjson`
- Report: `diagnostics/v3dt_report_20260115_191412.md`
- Family-room: H ratio ~0.30 (worse)

### 3b) Scale camInfo only (2.4x), keep dewarper dst K at rectified

- Changes:
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` fx/fy = 1858.85 / 1858.18
  - `config/dewarper_family_room_charuco_rtsp.txt` dst-focal-length = 774.52 / 774.24 (rectified)
- Log: `diagnostics/v3dt_frames_fr_dewarp_iter5_fx2p4_camonly.ndjson`
- Report: `diagnostics/v3dt_report_20260115_191912.md`
- Family-room: H ratio ~0.89, reproj ~21px (major improvement)

### 3c) Scale camInfo only (2.7x), keep dewarper dst K at rectified

- Changes:
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` fx/fy = 2091.20 / 2090.45
  - `config/dewarper_family_room_charuco_rtsp.txt` dst-focal-length = 774.52 / 774.24 (rectified)
- Log: `diagnostics/v3dt_frames_fr_dewarp_iter6_fx2p7_camonly.ndjson`
- Report: `diagnostics/v3dt_report_20260115_192145.md`
- Family-room: H ratio **1.01**, W ratio **1.03**, bottom offset **3.3px**, top offset **12.9px**

## 4) Current best family-room baseline (preview)

- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` (family-room fx/fy 2.7x)
- `config/dewarper_family_room_charuco_rtsp.txt` (dst-focal-length at rectified K)
- `config/camera_calibration_preview_dewarp.json` (auto-tilt sign -1)
- `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml` regenerated
- Best metrics: `diagnostics/v3dt_report_20260115_192145.md`

## 5) Dewarper zoom-out (black borders) preview

To restore full FoV (with black borders), we reduced the rectified K using
OpenCV `alpha=0.2`:

- Dewarper dst K (alpha=0.2):
  - `config/dewarper_family_room_charuco_rtsp.txt`
  - `dst-focal-length=628.65343644;628.42217409`
- CamInfo K scale kept at 2.7x (relative to new dewarp K):
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  - fx/fy = 1697.36 / 1696.74

Diagnostics:

- Log: `diagnostics/v3dt_frames_fr_dewarp_iter7_alpha0p2.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260115_203926.json`
- Report: `diagnostics/v3dt_report_20260115_203932.md`

Family-room median metrics:

- H ratio ~0.85 (slightly short)
- W ratio ~0.99
- Bottom offset ~0.8px
- Reproj ~18.4px

If we want H ratio ≈ 1.0 with black borders, increase family-room camInfo
fx/fy by ~17–18% (scale factor ~3.18 vs dewarped K).

Note: When zooming out (alpha=0.2), enable `cuda-address-mode=1` in the
dewarper config to avoid edge smearing (shows black borders instead of
stretched pixels).

## 6) Recenter dewarper output (fix asymmetric borders)

We saw asymmetric borders (black on left, right edge cut). Used dewarper dump
to measure the offset and recentre the output.

- Temporarily enabled `dewarp-dump-frames=1` to emit `Dewarper_Output_*_1280x720_0_interleaved.rgba`.
- Measured left margin **138 px** (right margin 0) on the dewarped output.
- Shifted dst principal point left by ~69 px to center the view.

Current dewarper output settings:

- `config/dewarper_family_room_charuco_rtsp.txt`:
  - `dst-principal-point=536.0;359.5`
  - `cuda-address-mode=1`
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`:
  - family-room `cx=536.00` (matches dewarped principal point)

## 7) Target 8px horizontal margins (family-room)

Measured the latest dewarper dump (`Dewarper_Output_0x5c07550_1280x720_0_interleaved.rgba`)
and found **left=10px, right=0px**. To hit symmetric **8px** margins:

- Shift output **left by 5px** (recentre content):
  - `dst-principal-point=514.0;359.5`
  - family-room `cx=514.00`
- Zoom out slightly to increase total margin from 10px → 16px:
  - scale factor = 1264/1270 = 0.9952756
  - `dst-focal-length=634.72218550;634.48869065`
  - family-room `fx/fy=1713.75 / 1713.12` (2.7x relative to dewarper K)
- camInfo regenerated: `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml`

Note: keep `dewarp-dump-frames=1` enabled until margins are verified, then remove.

## 8) Vertical scale drop investigation (2026-01-16)

Observed: family-room cuboids shrink as bbox bottoms move down the frame; tracking drops near the lower third.

Diagnostics:

- `diagnostics/v3dt_frames_fr_line_pre.ndjson`
- `diagnostics/v3dt_frames_fr_line_post.ndjson`

Key finding:

- With `NOESIS_V3DT_CAMINFO_Y_FLIP=1`, bbox3d depth (`yCentre`) increases as bbox bottoms move down the image (inverted depth trend). This yields a depth sign flip and a “line” where tracks collapse.

Fix direction (current best candidate): disable Y flip and keep dewarper + camInfo K consistent.

Applied changes:

- `config/dewarper_family_room_charuco_rtsp.txt`: `dst-focal-length=1447.54380469;1447.01129766` (2.2806×).
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`: family-room fx/fy set to 1447.54/1447.01 (match dewarper).
- camInfo regenerated with `NOESIS_V3DT_CAMINFO_Y_FLIP=0`.
- `config/analytics_exclude_baseline.ini`: stream-2 exclude ROIs disabled during diagnostics.

Pitch sweeps (manual, family-room only):

- pitch -5°: `config/camera_calibration_preview_dewarp_tiltfix_pitch5.json`
  - `diagnostics/v3dt_frames_fr_line_pitch5.ndjson`
- pitch -10°: `config/camera_calibration_preview_dewarp_tiltfix_pitch10.json`
  - `diagnostics/v3dt_frames_fr_line_pitch10.ndjson`
- pitch -15°: `config/camera_calibration_preview_dewarp_tiltfix_pitch15.json`
  - `diagnostics/v3dt_frames_fr_line_pitch15.ndjson`

Chosen preview baseline (pending visual confirmation):

- `config/camera_calibration_preview_dewarp_tiltfix_pitch5.json`
- `NOESIS_V3DT_CAMINFO_Y_FLIP=0`
- dewarper + camInfo K matched at 1447.54/1447.01
- exclude ROIs disabled for family-room via preview config

Next: verify that bbox3d depth stays positive through the lower frame; if not, re-run Charuco with wider coverage or refine extrinsics with a depth-based solve that does not flip roll.
