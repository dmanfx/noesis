# V3DT Iteration Notes — 2026-01-14

**Status (2026-01-22):** Historical log; superseded by the locked baseline in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`. Commands here use
older camInfo polarity and are not the current baseline; revalidate before reuse.

## 1) Depth-tilt preview (flip-image-y) for all cameras

**Goal:** Update pitch/roll from MapAnything depth for living-room, kitchen, family-room (no yaw/translation changes) and retest SV3DT.

**Depth capture run:**

- Command:
  - `NOESIS_V3DT_AUTOGEN_CAMINFO=0 NOESIS_DEPTH_STORE_ENABLED=1 NOESIS_WS_PORT=6034 timeout 45s python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium_preview_flipy_fx129.yaml --depth-enable-seconds 20 --disable-rest`

**Tilt preview output:**

- `config/camera_calibration_preview_flipy_all.json`
- Tilt candidates:
  - living-room: `candidate_sign=-1`, pitch -25.28°, roll +0.39°
  - kitchen: `candidate_sign=+1`, pitch -35.87°, roll +1.70°
  - family-room: `candidate_sign=-1`, pitch -21.94°, roll -2.25°

**camInfo generation:**

- Command:
  - `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1 NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p NOESIS_V3DT_CAMINFO_INVERT_E=1 NOESIS_V3DT_CAMINFO_Y_FLIP=1 NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy python3 scripts/generate_v3dt_caminfo.py --cameras-config config/cameras_preview_flipy_intrinsics.yaml --calibration config/camera_calibration_preview_flipy_all.json --pipeline-config config/infer_v3dt_medium_preview_flipy_fx129.yaml --output-dir config/v3dt_preview_flipy_all_fx129`

**Tracker + pipeline configs:**

- `config/v3dt/nvtracker_sv3dt_preview_flipy_all_fx129.yml`
- `config/infer_v3dt_medium_preview_flipy_all_fx129.yaml`

**Diagnostics run:**

- Command:
  - `NOESIS_V3DT_AUTOGEN_CAMINFO=0 NOESIS_V3DT_DIAG_LOG=1 NOESIS_V3DT_DIAG_DIR=diagnostics NOESIS_V3DT_DIAG_SESSION=tilt_flipy_all_fx129 NOESIS_WS_PORT=6035 timeout 60s python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium_preview_flipy_all_fx129.yaml --disable-rest`
- Log:
  - `diagnostics/v3dt_frames_tilt_flipy_all_fx129.ndjson`
- Snapshot:
  - `diagnostics/v3dt_snapshot_20260114_142240.json`
- Report:
  - `diagnostics/v3dt_report_20260114_142247.md`

**Key metrics (median):**

- living-room: reproj 37.4px, proj height ratio 0.93, depth ratio 1.71
- kitchen: reproj 120.5px, proj height ratio 0.75, depth ratio 1.09
- family-room: reproj 32.0px, proj height ratio 0.37, depth ratio 1.09

**Notes:**

- Living-room height ratio is now close to 1.0 (projection size mostly correct), but depth ratio still high; suggests residual depth bias or metric mismatch in Y-forward assumption.
- Kitchen height ratio improved but still short; reprojection error is higher than expected.
- Family-room remains the biggest mismatch: projected 3D box height is ~1/3 of the 2D bbox height even though footpoints are anchored.

## 2) Family-room sanity spot-check (single-frame)

Using `diagnostics/v3dt_frames_tilt_flipy_all_fx129.ndjson`, a few family-room samples show:

- `bbox3d` values: `zLen ≈ 1.7`, `yCentre ≈ 12–16m`, `xLen ≈ yLen ≈ 0.7`
- 2D bbox height ~250–320px
- Projected 3D height ~85–100px (ratio ~0.27–0.39)

This indicates depth is likely too far (or intrinsics FOV too wide), causing a small projection in image space.

## 3) Next iteration (planned)

### 3a) Family-room intrinsics scale sweep (2.0 / 2.4 / 3.0)

We increased the **family-room** intrinsics scale (relative to base G4) while keeping living-room/kitchen unchanged.

**Configs:**

- 2.0x:
  - Cameras: `config/cameras_preview_flipy_intrinsics_fr200.yaml`
  - CamInfo: `config/v3dt_preview_flipy_all_fr200/`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_flipy_all_fr200.yml`
  - Pipeline: `config/infer_v3dt_medium_preview_flipy_all_fr200.yaml`
  - Log: `diagnostics/v3dt_frames_tilt_flipy_all_fr200.ndjson`
  - Snapshot: `diagnostics/v3dt_snapshot_20260114_143408.json`
  - Report: `diagnostics/v3dt_report_20260114_143427.md`

- 2.4x:
  - Cameras: `config/cameras_preview_flipy_intrinsics_fr240.yaml`
  - CamInfo: `config/v3dt_preview_flipy_all_fr240/`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_flipy_all_fr240.yml`
  - Pipeline: `config/infer_v3dt_medium_preview_flipy_all_fr240.yaml`
  - Log: `diagnostics/v3dt_frames_tilt_flipy_all_fr240.ndjson`
  - Snapshot: `diagnostics/v3dt_snapshot_20260114_143435.json`
  - Report: `diagnostics/v3dt_report_20260114_143443.md`

- 3.0x:
  - Cameras: `config/cameras_preview_flipy_intrinsics_fr300.yaml`
  - CamInfo: `config/v3dt_preview_flipy_all_fr300/`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_flipy_all_fr300.yml`
  - Pipeline: `config/infer_v3dt_medium_preview_flipy_all_fr300.yaml`
  - Log: `diagnostics/v3dt_frames_tilt_flipy_all_fr300.ndjson`
  - Snapshot: `diagnostics/v3dt_snapshot_20260114_143907.json`
  - Report: `diagnostics/v3dt_report_20260114_143913.md`

**Family-room median metrics (proj height ratio / depth ratio / reproj):**

- 1.29x baseline: **0.37 / 1.09 / 32.0px**
- 2.0x: **0.61 / 0.62 / 25.3px**
- 2.4x: **0.74 / 0.50 / 24.9px**
- 3.0x: **0.94 / 0.38 / 26.1px**

**Takeaway:** scaling intrinsics upward fixes the **visual cuboid height** (3.0x is close), but pulls the
depth ratio lower than 1.0, suggesting the depth estimates will be too near unless the camera is truly
zoomed/cropped. This strongly points to a **family-room FOV mismatch** (digital zoom or nonstandard crop).

## 4) ChArUco intrinsics calibration tooling

Added a standalone ChArUco intrinsics calibration tool for creating accurate K values and updating
`intrinsics.json` + `config/cameras.yaml`:

- Script: `scripts/charuco_calibrate_intrinsics.py`
- Guide: `docs/DS8_charuco_intrinsics_calibration.md`

## 5) Family-room ChArUco intrinsics (scaled to 1280x720)

We calibrated the family-room at 2688x1512 (downloaded stream), then scaled K to 1280x720
for DS8 testing. New preview cameras file:

- `config/cameras_preview_charuco_fr.yaml` (family-room model: `unifi_g4_instant_charuco_720`)
- CamInfo: `config/v3dt_preview_charuco_fr/`
- Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr.yml`
- Pipeline: `config/infer_v3dt_medium_preview_charuco_fr.yaml`

Diagnostics:

- Log: `diagnostics/v3dt_frames_charuco_fr_scaled.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260114_204138.json`
- Report: `diagnostics/v3dt_report_20260114_204146.md`

Family-room median metrics (charuco scaled):

- Proj height ratio ~0.29 (too small)
- Depth ratio ~1.86 (too far)
- Reproj ~34.6px

Conclusion: simply scaling the full-res ChArUco intrinsics down to 1280x720
**does not match** the DS8 feed. The substream likely uses a different crop/zoom.

## 6) Family-room ChArUco intrinsics (RTSP 1280x720)

Calibrated directly from the RTSP 1280x720 feed and wired a preview config
without touching the main calibration files:

- Output: `diagnostics/charuco_family_room_rtsp_1280x720.json`
- Preview cameras: `config/cameras_preview_charuco_fr_rtsp.yaml`
- CamInfo: `config/v3dt_preview_charuco_fr_rtsp/`
- Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp.yml`
- Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp.yaml`

Calibration summary (RTSP 1280x720):

- fx 768.72, fy 768.43, cx 641.37, cy 356.80
- Distortion: k1 -0.41490, k2 0.24677, k3 -0.08768
- Reproj ~0.57px (median ~0.43px) using 104/240 frames

Next: run DS8 with the RTSP preview pipeline and capture a new V3DT report to
validate cuboid height and depth ratios under real motion.

## 7) Family-room ChArUco (RTSP 1280x720) SV3DT run

Captured a live RTSP 1280x720 ChArUco calibration and ran SV3DT with the
preview pipeline (family-room RTSP, other cameras unchanged).

- Log: `diagnostics/v3dt_frames_charuco_fr_rtsp.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260114_220731.json`
- Report: `diagnostics/v3dt_report_20260114_220737.md`

Family-room median metrics (RTSP 1280x720):

- Proj height ratio ~0.24 (too small)
- Depth ratio ~1.92 (too far)
- Reproj ~13.7px (reasonable)

Conclusion: direct RTSP intrinsics still produce undersized cuboids. The
depth estimate is ~1.9x too far relative to 2D height, so the next step is
to sweep family-room intrinsics scale upward around ~2.0x and compare.

## 8) Distortion correction via nvdewarper (preview wiring)

SV3DT camInfo ignores lens distortion, so wide-angle feeds require GPU
undistortion before PGIE/tracker. Added a dewarper-based preview setup:

- Dewarper config: `config/dewarper_family_room_charuco_rtsp.txt`
- Rectified intrinsics: `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
- CamInfo: `config/v3dt_preview_charuco_fr_rtsp_dewarp/`
- Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp.yml`
- Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml`

Next: run DS8 with the dewarped preview pipeline, then redo the tilt preview
using the rectified intrinsics (`--cameras-config`).

Note: adjusted DS8 per-source linking to avoid explicit pad hints when linking
`nvurisrcbin` into the dewarper/mux chain. Using pad hints caused linking
failures ("caps not compatible") with dynamic pads.

Follow-up: added a post-dewarper NV12 conversion (RGBA -> NV12) before streammux
to resolve "Input Output Color Format Mismatch" and black frames on the mosaic.

## 9) Dewarper preview runs (RTSP)

Two dewarper logs were captured:

- Pre-tilt: `diagnostics/v3dt_frames_charuco_fr_rtsp_dewarp.ndjson`
- Post-tilt: `diagnostics/v3dt_frames_charuco_fr_rtsp_dewarp_posttilt.ndjson`

Reports:

- Pre-tilt report: `diagnostics/v3dt_report_20260115_144242.md`
- Post-tilt report: `diagnostics/v3dt_report_20260115_144253.md`

Key observations:

- Family-room tracks were sparse (12 pre-tilt, 4 post-tilt).
- Pre-tilt family-room height ratio ~1.20 (close), but depth ratio collapsed to ~0.09.
- Post-tilt family-room height ratio ~0.27, depth ratio ~1.51, reproj ~72.9px.
- Post-tilt living-room/kitchen metrics shifted, implying the tilt preview updated
  all cameras instead of just family-room.

Next: compute a rectified (undistorted) K with an explicit dst focal/principal
point for the dewarper, and re-run tilt for family-room only.

Update: computed rectified K (OpenCV `getOptimalNewCameraMatrix`,
centerPrincipalPoint=True, alpha=0) and wired it into:

- `config/dewarper_family_room_charuco_rtsp.txt` (dst-focal-length / dst-principal-point)
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` (fx/fy/cx/cy updated, distortion zero)

Follow-up: enforce explicit dewarper output caps (width/height) before streammux
to avoid quadrant/cropping artifacts when the dewarper output size is mis-negotiated.
