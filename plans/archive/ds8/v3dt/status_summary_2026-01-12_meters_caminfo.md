# V3DT/SV3DT Status Summary (meters camInfo) — 2026-01-12

**Update (2026-01-22):** superseded by the locked baseline in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`.

## What we were debugging (symptoms)

- With SV3DT enabled (`stateEstimatorType: 3`), tracks/masks/“active tracks” would disappear under motion (especially lower-body occlusions), despite PGIE producing masks.
- In the UI, the blue wireframe 3D cuboid sometimes:
  - collapsed into a single diagonal line across the entire frame, or
  - appeared “stuck” at a mostly fixed location while the person moved, or
  - looked stretched/skewed relative to the person.
- The 2D configuration (`infer_v3dt_2d_test.yaml` / `nvtracker_sv3dt_2d_test.yml`) tracked well across cameras, strongly implicating SV3DT’s 3D projection/model-fit path (camInfo/projection/extrinsics scaling/conventions), not the detector.

## Key findings we used to guide fixes

- `config/camera_calibration.json` stores `E` as **world→camera** (Menon convention), column-major 4×4, with translations in **meters**.
- The critical failure mode for “stretched line” cuboids and sporadic tracking is SV3DT receiving an inconsistent projection/camera model (resolution mismatch, intrinsics scaling mismatch, or unit mismatch), which can make the 3D fit numerically unstable and cause the tracker to terminate tracks rapidly.
- `NOESIS_V3DT_AUTOGEN_CAMINFO=1` runs camInfo generation at DS8 startup (from current `config/cameras.yaml` intrinsics + `config/camera_calibration.json` extrinsics). If this flag is not set, you can be testing with stale camInfo files.

## What we changed (repo changes)

### 1) camInfo generation now defaults to meters

- File: `scripts/generate_v3dt_caminfo.py`
- Change: default `NOESIS_V3DT_CAMINFO_WORLD_SCALE` is now **1** (meters), not 100 (centimeters).
- Result: regenerated camInfo files now contain:
  - `modelInfo.height: 1.7`, `modelInfo.radius: 0.35` (meters)
  - `projectionMatrix_3x4` translation terms (`P[:,3]`) are ~100× smaller than the prior centimeter-scale camInfo.

### 2) SV3DT tracker configs rescaled for meters

If camInfo world-units move from cm→m, the world-space variances/thresholds must be rescaled:
- world-unit variances (units²): multiply by `1e-4` (cm²→m²)
- world-unit distance thresholds: multiply by `0.01` (cm→m)

Updated files:
- `config/v3dt/nvtracker_sv3dt.yml`
- `config/v3dt/nvtracker_sv3dt_lite.yml`
- `config/v3dt/nvtracker_sv3dt_medium.yml`
- `config/v3dt/nvtracker_mv3dt.yml`
- `config/v3dt/nvtracker_sv3dt_sample.yml`
- `config/v3dt/nvtracker_sv3dt_sample_3cam.yml`
- `config/v3dt/nvtracker_sv3dt_sample_single_caminfo.yml`
- `config/v3dt/nvtracker_sv3dt_sample_single_caminfo_inv.yml`

### 3) camInfo files regenerated with the new meters default

Regenerated files:
- `config/v3dt/camInfo_living-room.yml`
- `config/v3dt/camInfo_kitchen.yml`
- `config/v3dt/camInfo_family-room.yml`

## Environment setups we used/observed (and what they did)

### Always-useful for reproducibility

- `NOESIS_V3DT_AUTOGEN_CAMINFO=1` (default)
  - Forces camInfo regeneration at runtime start.
  - Helps ensure tests are not accidentally using stale camInfo.
  - Note: DS8 runtime logs the command using the **effective pipeline** YAML it builds (e.g., `build/effective_pipeline_*.yaml`), which is why logs may show a different `--pipeline-config` path than the one you passed on the CLI.

### Projection/extrinsics toggles we experimented with historically

- `NOESIS_V3DT_CAMINFO_INVERT_E`
  - `0` means “use stored `E` as world→camera”.
  - `1` means “invert stored `E` before projecting”.
  - Menon sends world→camera `E`, so the canonical setting is `0`.
  - We tested inversion while chasing stability, but it was not the root cause.

- `NOESIS_V3DT_CAMINFO_MATRIX_TYPE`
  - `3x4` writes `projectionMatrix_3x4` (DeepStream applies principal-point shift internally).
  - `w2p` writes `projectionMatrix_3x4_w2p` (explicit world→pixel).
  - We tried `w2p` vs `3x4` during troubleshooting; it did not consistently resolve the sporadic tracking for the problematic cameras.

- `NOESIS_V3DT_CAMINFO_Y_FLIP`
  - Tried as an “escape hatch” for image Y-axis convention mismatches.
  - It can fix “head below feet” style projections, but it was not a universal fix for flicker/instability.

- `NOESIS_V3DT_CAMINFO_WORLD_SCALE`
  - Historically set to `100` to generate centimeter-scale camInfo (matching NVIDIA sample conventions).
  - Now defaults to `1` (meters), and tracker configs were rescaled accordingly.

## “What works” right now (as reported after meters camInfo)

- Family-room: still tracks well (as before).
- Kitchen: tracks pretty well, but cuboid is still stretched/diagonal (3D bbox projection still not correct).
- Living-room: tracks show sometimes, but not great; cuboid stretching similar to kitchen.

This suggests the cm↔m mismatch was not the only contributor; remaining issues are likely still in camera-model/projection conventions (per-camera extrinsics correctness, handedness/axis mapping, or per-camera intrinsics correctness), but SV3DT is now operating in a consistent meter-scale configuration.

## Camera-specific “worked best” configurations we’ve seen so far

These are the *notable* configurations observed during debugging (not all were simultaneous across cameras):

- **2D control (all cameras “good”):**
  - Pipeline: `config/infer_v3dt_2d_test.yaml` (2D tracking; SV3DT ObjectModelProjection disabled in `nvtracker_sv3dt_2d_test.yml`)
  - Outcome: best overall tracking stability across all rooms; used as the baseline to prove detector quality was not the main issue.

- **SV3DT cm-scale phase — “family-room works now” baseline (historical):**
  - Command: `NOESIS_V3DT_AUTOGEN_CAMINFO=1 python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium.yaml`
  - camInfo generation settings during this phase:
    - `NOESIS_V3DT_CAMINFO_INVERT_E=0` (use `E` as world→camera; no per-camera override)
    - `NOESIS_V3DT_CAMINFO_Y_FLIP=0`
    - `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` (camInfo in cm; `modelInfo.height: 170`, `radius: 35`)
    - `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=3x4` (writes `projectionMatrix_3x4`)
  - Menon note: family-room yaw was changed from 180→0 and extrinsics resent; family-room still worked.
  - Outcome: family-room tracked well; kitchen/living varied; living-room was often intermittent/flickery.

- **SV3DT “kitchen was best” moment (historical flag-toggling phase):**
  - This was during the earlier “toggle flags until kitchen behaves” troubleshooting window.
  - Key env settings reported as the turning point:
    - `NOESIS_V3DT_CAMINFO_INVERT_E=1` (the big one that brought kitchen back first)
    - With `INVERT_E=1`, switching `NOESIS_V3DT_CAMINFO_Y_FLIP=0` made the cuboid stop “breathing”/resizing wildly and become more anchored.
  - Typical unit and matrix choices at the time:
    - Usually `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` (unless explicitly testing `=1` for meters)
    - camInfo was still being written as `projectionMatrix_3x4_w2p` (before we introduced `projectionMatrix_3x4` as the default)
  - Net: that combo (`INVERT_E=1`, `Y_FLIP=0`, usually `WORLD_SCALE=100`, using `_w2p`) made kitchen look the best it ever had, even though it didn’t fix living/family and later regressed.

- **SV3DT meters-scale phase (current code/config; family-room “good”, kitchen/living “mixed”):**
  - camInfo generated in **meters** (default `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`)
  - SV3DT KF/thresholds rescaled to meters in `config/v3dt/nvtracker_sv3dt*.yml`
  - Outcome (your latest report): family-room still stable; kitchen tracks fairly well but cuboids are still distorted; living-room shows some tracks but not stable.

## Latest baseline (2026-01-13) — all cameras tracking, cuboids still off

**Baseline env (best so far):**

```bash
export NOESIS_V3DT_AUTOGEN_CAMINFO=1
export NOESIS_V3DT_CAMINFO_WORLD_SCALE=1
export NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p
export NOESIS_V3DT_CAMINFO_INVERT_E=1
export NOESIS_V3DT_CAMINFO_Y_FLIP=1
export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy
```

**Observed results:**

- **Kitchen:** 3D tracking is stable; cuboid appears the right overall size but still slightly skewed.
- **Family-room:** 3D tracking is stable; cuboid is anchored but still slightly off in projection.
- **Living-room:** 3D tracking now present; still needs calibration tightening.

**Telling signal:**

- `bbox3d.zLen ≈ 1.7` while `bbox3d.yLen ≈ 0.7` → height is in **Z**, not **Y**.
- DeepStream docs + reference outputs show **foot world position is 2D (X,Y) on the ground plane**, implying **Z‑up**.

**Interpretation / likely root cause:**

- Our calibration + BEV math is **Y‑up**, but SV3DT appears to be **Z‑up**.
- That axis mismatch can keep tracking stable while skewing the 3D cuboid projection (height axis treated as depth/side).

**Next targeted fix (for cuboids):**

- Generate camInfo with a world‑axis remap to Z‑up (swap Y/Z) and retest:
  - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
  - Keep the baseline env above.
- Expectation: cuboids should align (height on vertical axis), and bbox3d height sanity should read from `zLen`.

**Defaults updated (2026-01-13):**

- Code defaults now match the baseline above so new runs track on all cameras without manual exports.

## Scale-sweep A/B (2026-01-13)

To test for cm↔m style unit mismatches, we ran the new forensics scale sweep on the baseline log:

- Command:
  - `PYTHONPATH=. python3 scripts/v3dt_forensics.py analyze --log diagnostics/v3dt_frames_run_20260113_163907.ndjson --snapshot diagnostics/v3dt_snapshot_20260113_163907.json --scale-sweep auto`
- Output:
  - `diagnostics/v3dt_report_20260113_184845.json`
  - `diagnostics/v3dt_report_20260113_184845.md`
  - `diagnostics/v3dt_panel_20260113_184852.html`

**Findings:**

- Best reprojection scale for all cameras is **1.0** (meters). Other scales are dramatically worse.
- Median reprojection error at scale=1.0:
  - living-room: ~13 px
  - kitchen: ~34.7 px
  - family-room: ~16.7 px
- Median bbox3d height remains ~1.7 m at scale=1.0, so the “tiny cuboid” issue is not a global unit mismatch.

## Projection diagnostics (2026-01-13)

We added a projection diagnostic that projects the full 3D cuboid (axis-aligned; rotations ignored) and compares it to the 2D bbox.

- Command:
  - `PYTHONPATH=. python3 scripts/v3dt_forensics.py analyze --log diagnostics/v3dt_frames_run_20260113_163907.ndjson --snapshot diagnostics/v3dt_snapshot_20260113_163907.json --scale-sweep auto`
- Output:
  - `diagnostics/v3dt_report_20260113_193814.json`
  - `diagnostics/v3dt_report_20260113_193814.md`
  - `diagnostics/v3dt_panel_20260113_193821.html`

**Median projected 3D height ratio (proj_h / 2D bbox_h):**

- living-room: **~0.22** (projected 3D box is ~5× shorter than the 2D bbox)
- family-room: **~0.30** (projected 3D box is ~3× shorter)
- kitchen: **~1.07** (projected 3D box height matches 2D bbox height)

**Interpretation:**

- The overlay mismatch is **camera-specific**, not a global unit mismatch.
- Kitchen’s projection math is now consistent; living-room + family-room still project the 3D box too short.
- This points to **per-camera intrinsics or camInfo projection mismatch** (e.g., intrinsics resolution mismatch or wrong camInfo for those cameras), rather than global scale.

## Depth consistency (2026-01-13)

We added a “depth consistency” check that compares bbox3d depth (`yCentre`, assuming Y-forward with Z-up) to the depth implied by the 2D bbox height using `fy` and `bbox3d.zLen`.

- Output:
  - `diagnostics/v3dt_report_20260113_203914.json`
  - `diagnostics/v3dt_report_20260113_203914.md`
  - `diagnostics/v3dt_panel_20260113_203919.html`

**Median depth ratio (bbox3d depth / implied depth):**

- living-room: **~5.55** (depth is ~5.5× too far)
- family-room: **~1.44** (depth ~44% too far)
- kitchen: **~0.96** (aligned)

**Interpretation:**

- Living-room (and mildly family-room) depth estimates are biased too far, which directly explains the “short” cuboids even when footpoints are anchored.
- This is consistent with a per-camera extrinsics or camera-model mismatch (not global scale).

## Post-extrinsics test (living-room + family-room resend, 2026-01-13)

Menon resent extrinsics for living-room + family-room; we regenerated camInfo and ran a short diagnostics capture.

- Snapshot:
  - `diagnostics/v3dt_snapshot_20260113_205606.json`
- Log:
  - `diagnostics/v3dt_frames_post_extrinsics_20260113_205605.ndjson`
- Report:
  - `diagnostics/v3dt_report_20260113_205640.json`
  - `diagnostics/v3dt_report_20260113_205640.md`
  - `diagnostics/v3dt_panel_20260113_205640.html`

**Findings:**

- Living-room improved substantially:
  - Reprojection median dropped to ~3.7 px.
  - Depth ratio reduced to ~2.12 (was ~5.55), but still >1 (depth still too far).
  - Projected height ratio now ~0.56 (still short but closer than ~0.22).
- Kitchen + family-room had no person tracks during this short run, so no metrics yet.

**Interpretation:**

- New extrinsics clearly improved living-room calibration.
- We still need a longer capture with people visible in kitchen + family-room to validate those cameras.

## Post-extrinsics + file sources (2026-01-13)

You swapped all streams to file sources with long people visibility. We reran the diagnostics capture.

- Snapshot:
  - `diagnostics/v3dt_snapshot_20260113_210002.json`
- Log:
  - `diagnostics/v3dt_frames_post_extrinsics_files_20260113_210002.ndjson`
- Report:
  - `diagnostics/v3dt_report_20260113_210030.json`
  - `diagnostics/v3dt_report_20260113_210030.md`
  - `diagnostics/v3dt_panel_20260113_210031.html`

**Findings:**

- Kitchen now has solid metrics (people tracks present):
  - Reprojection ~34.2 px, depth ratio ~0.95, projected height ratio ~1.13 → consistent.
- Living-room regressed with the file source:
  - Reprojection ~11.1 px, depth ratio ~5.15, projected height ratio ~0.23 → still “too far” depth bias.
- Family-room still shows **0** people tracks in this capture, so no metrics yet (likely file has no person or detector missing them).

**Interpretation:**

- Living-room’s depth bias persists in the file run; we should verify the file’s calibration (same camera model but possibly different zoom/crop or lens settings).
- We need a family-room file with visible people (or verify detection) to compute its metrics.

## Commands used most often

- Normal run:
  - `python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium.yaml`
- Force camInfo regeneration:
  - `NOESIS_V3DT_AUTOGEN_CAMINFO=1 python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_medium.yaml`

## Tilt-only preview calibration (2026-01-13)

To isolate pitch/roll (tilt) without touching yaw/translation, we added a depth-plane
tilt preview tool that writes a **separate** calibration file:

```bash
python3 scripts/auto_tilt_from_depth.py \
  --camera living-room \
  --camera family-room \
  --output config/camera_calibration_preview.json
```

Use `scripts/generate_v3dt_caminfo.py --calibration config/camera_calibration_preview.json`
to generate preview camInfo files for comparison without overwriting the main calibration.

## Tilt preview with flip-image-y (all cameras, 2026-01-14)

We reran the depth-plane tilt preview with `--flip-image-y` for **all three cameras** after swapping
all sources to local files with people visible.

### Preview calibration + camInfo

- Preview file: `config/camera_calibration_preview_flipy_all.json`
- CamInfo output: `config/v3dt_preview_flipy_all_fx129/`
- Tracker config: `config/v3dt/nvtracker_sv3dt_preview_flipy_all_fx129.yml`
- Pipeline config: `config/infer_v3dt_medium_preview_flipy_all_fx129.yaml`

### Diagnostics run

- Log: `diagnostics/v3dt_frames_tilt_flipy_all_fx129.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260114_142240.json`
- Report: `diagnostics/v3dt_report_20260114_142247.md`

### Median metrics (from the report)

- **living-room:** reproj ~37.4px, proj height ratio ~0.93, depth ratio ~1.71
- **kitchen:** reproj ~120.5px, proj height ratio ~0.75, depth ratio ~1.09
- **family-room:** reproj ~32.0px, proj height ratio ~0.37, depth ratio ~1.09

### Interpretation

- Living-room height projection is now close to correct, but depth ratio is still high (depth too far).
- Kitchen improved but still projects short; reprojection error remains high.
- Family-room remains the outlier: height projection is still too small (ratio ~0.37) despite good
  foot anchoring and decent reprojection error, pointing to a per-camera intrinsics/FOV mismatch
  rather than a global scale error.

## Family-room intrinsics sweep (2026-01-14)

We scaled **only the family-room** intrinsics upward and reran diagnostics to see if the cuboid height
projection could be fixed without breaking reprojection. Living-room and kitchen configs stayed the same.

| Scale | Report | Proj height ratio | Depth ratio | Reproj (px) |
| --- | --- | --- | --- | --- |
| 1.29x | `diagnostics/v3dt_report_20260114_142247.md` | ~0.37 | ~1.09 | ~32.0 |
| 2.0x | `diagnostics/v3dt_report_20260114_143427.md` | ~0.61 | ~0.62 | ~25.3 |
| 2.4x | `diagnostics/v3dt_report_20260114_143443.md` | ~0.74 | ~0.50 | ~24.9 |
| 3.0x | `diagnostics/v3dt_report_20260114_143913.md` | ~0.94 | ~0.38 | ~26.1 |

**Interpretation:**

- Larger intrinsics scales bring the **cuboid height projection** close to correct.
- Depth ratios drop below 1.0 as scale increases, implying the world depth estimate becomes too near.
- This strongly suggests a **family-room FOV mismatch** (digital zoom/crop or source resolution mismatch).

## Family-room ChArUco (full-res → 1280x720 scaled) test (2026-01-14)

We calibrated ChArUco intrinsics on the downloaded 2688x1512 stream, scaled to 1280x720,
and generated camInfo from the scaled model.

- Cameras config: `config/cameras_preview_charuco_fr.yaml`
- CamInfo: `config/v3dt_preview_charuco_fr/`
- Report: `diagnostics/v3dt_report_20260114_204146.md`

**Result:** family-room projection height ratio ~0.29, depth ratio ~1.86, reproj ~34.6px.

**Interpretation:** scaled full‑res intrinsics do **not** match the DS8 substream.
The substream likely applies a crop/zoom or different ISP path. We need ChArUco
captures on the **actual 1280x720 feed** (or learn its crop/zoom parameters).

## Family-room ChArUco (RTSP 1280x720) calibration (2026-01-14)

We calibrated intrinsics directly on the **RTSP 1280x720** feed and wired a preview
config without touching the main calibration files.

- Output: `diagnostics/charuco_family_room_rtsp_1280x720.json`
- Cameras config: `config/cameras_preview_charuco_fr_rtsp.yaml`
- CamInfo: `config/v3dt_preview_charuco_fr_rtsp/`
- Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp.yaml`

Calibration summary (RTSP 1280x720):

- fx 768.72, fy 768.43, cx 641.37, cy 356.80
- Distortion: k1 -0.41490, k2 0.24677, k3 -0.08768
- Reproj ~0.57px (median ~0.43px)

**Next step:** run DS8 with the RTSP preview pipeline and capture a new V3DT report
to validate cuboid height/depth ratios under real motion.

## Family-room ChArUco RTSP run results (2026-01-14)

Ran DS8 with the RTSP preview pipeline using the 1280x720 ChArUco intrinsics.

- Log: `diagnostics/v3dt_frames_charuco_fr_rtsp.ndjson`
- Snapshot: `diagnostics/v3dt_snapshot_20260114_220731.json`
- Report: `diagnostics/v3dt_report_20260114_220737.md`

Family-room median metrics:

- Proj height ratio ~0.24 (too small)
- Depth ratio ~1.92 (too far)
- Reproj ~13.7px (reasonable)

**Interpretation:** direct RTSP ChArUco intrinsics still yield undersized cuboids.
Depth is ~1.9x too far relative to 2D height, pointing back to a per-camera
intrinsics/FOV scale mismatch. Next step: sweep the family-room intrinsics scale
around ~2.0x and compare.

## Distortion correction preview (nvdewarper) (2026-01-14)

SV3DT camInfo does not consume lens distortion, so wide-angle feeds require
undistortion before tracking. Added a dewarper-based preview setup:

- Dewarper config: `config/dewarper_family_room_charuco_rtsp.txt`
- Rectified intrinsics: `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
- CamInfo: `config/v3dt_preview_charuco_fr_rtsp_dewarp/`
- Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp.yml`
- Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml`

**Next step:** run DS8 with the dewarped preview pipeline and redo the
tilt preview using the rectified intrinsics (`--cameras-config`).

## Dewarper preview results (2026-01-15)

Two dewarper runs were captured:

- Pre-tilt log: `diagnostics/v3dt_frames_charuco_fr_rtsp_dewarp.ndjson`
- Post-tilt log: `diagnostics/v3dt_frames_charuco_fr_rtsp_dewarp_posttilt.ndjson`

Reports:

- Pre-tilt report: `diagnostics/v3dt_report_20260115_144242.md`
- Post-tilt report: `diagnostics/v3dt_report_20260115_144253.md`

Family-room summary:

- Pre-tilt: height ratio ~1.20, depth ratio ~0.09, reproj ~50.9px (sparse tracks).
- Post-tilt: height ratio ~0.27, depth ratio ~1.51, reproj ~72.9px (very sparse tracks).

**Interpretation:** dewarper is active but the rectified K needs to be defined
explicitly (dst focal/principal point). Also, the tilt preview appears to have
updated all cameras; rerun tilt for **family-room only** to avoid regressions.

Update: rectified K computed via OpenCV `getOptimalNewCameraMatrix` and wired
into `config/dewarper_family_room_charuco_rtsp.txt` (dst-focal-length,
dst-principal-point) plus `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
(fx/fy/cx/cy updated, distortion zero).

Follow-up: added explicit dewarper output caps (width/height) before streammux
to prevent quadrant/cropping artifacts from caps negotiation.

## Family-room dewarper + camInfo scale baseline (preview, 2026-01-15)

To dial in the family-room cuboid size with the dewarper enabled:

- **Dewarp config (unchanged rectified K):**
  - `config/dewarper_family_room_charuco_rtsp.txt`
  - `dst-focal-length=774.52;774.24` (rectified K)
- **SV3DT intrinsics (scaled for camInfo only):**
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  - family-room fx/fy scaled **2.7x** (fx 2091.20, fy 2090.45)
- **Tilt preview:**
  - `config/camera_calibration_preview_dewarp.json` (auto-tilt sign -1; pitch -23.9°, roll -1.1°)
- **camInfo:**
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml` regenerated

**Result:**

- Report: `diagnostics/v3dt_report_20260115_192145.md`
- Family-room projection metrics:
  - H ratio **1.01** (proj height matches 2D bbox)
  - W ratio **1.03**
  - Bottom offset **3.3px**, top offset **12.9px**
  - Reproj **23.2px**

This is the current best family-room baseline; cuboid height/width visually align while keeping the dewarper’s rectified output intact.

## Family-room dewarper zoom-out (black borders) preview (2026-01-15)

To recover the clipped FoV, we zoomed out the dewarper by switching to a
rectified K with **alpha=0.2** (black borders expected):

- Dewarper: `config/dewarper_family_room_charuco_rtsp.txt`
  - `dst-focal-length=628.65343644;628.42217409`
- CamInfo K (family-room only): `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  - fx/fy = 1697.36 / 1696.74 (2.7x relative to dewarper K)

Results:

- Report: `diagnostics/v3dt_report_20260115_203932.md`
- Family-room H ratio ~0.85, W ratio ~0.99, reproj ~18.4px
- Slightly short cuboids; increase family-room fx/fy ~17–18% if needed to hit H ratio ≈ 1.0

Note: To avoid edge smearing when zooming out the dewarper (alpha=0.2), set
`cuda-address-mode=1` in `config/dewarper_family_room_charuco_rtsp.txt`. This
forces black borders instead of repeated pixels at the edges.

## Family-room dewarper recenter (2026-01-15)

To fix asymmetric borders (black on left, right edge cropped), we recentered
`dst-principal-point` based on a dewarper dump:

- `config/dewarper_family_room_charuco_rtsp.txt`:
  - `dst-principal-point=570.5;359.5`
  - `cuda-address-mode=1`
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`:
  - family-room `cx=570.50` (match dewarped principal point)

This should produce symmetric black borders and restore the right edge.

Update: dewarper dump showed remaining left margin 69px (right 0), so we shifted
`dst-principal-point` further left:

- `config/dewarper_family_room_charuco_rtsp.txt`: `dst-principal-point=536.0;359.5`
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`: family-room `cx=536.00`

This should remove the remaining right-side crop (centered output).

## Family-room dewarper border fine-tune (2026-01-15)

Measured the latest dewarper dump (`Dewarper_Output_0x5c07550_1280x720_0_interleaved.rgba`)
and found **left=10px, right=0px** margins. To target symmetric 8px margins:

- Dewarper: `config/dewarper_family_room_charuco_rtsp.txt`
  - `dst-principal-point=514.0;359.5`
  - `dst-focal-length=634.72218550;634.48869065`
  - `cuda-address-mode=1`
- CamInfo intrinsics: `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  - family-room `fx/fy=1713.75 / 1713.12` (2.7x)
  - family-room `cx=514.00` (match dewarper principal point)
- camInfo regenerated: `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml`

Next: rerun preview pipeline to confirm margins ~8px and remove `dewarp-dump-frames=1`.

## Family-room vertical scale drop (2026-01-16)

Observed a consistent “line” where cuboids shrink and tracking drops as bbox bottoms move down the frame.
Diagnostics show the bbox3d depth trend is inverted (depth increases as bbox bottoms move down) when
`NOESIS_V3DT_CAMINFO_Y_FLIP=1`.

Mitigation tested:

- Disable Y-flip in camInfo generation (`NOESIS_V3DT_CAMINFO_Y_FLIP=0`).
- Keep dewarper + camInfo K aligned (avoid camInfo-only scaling).
- Scale dewarper K and camInfo K together (2.2806×) to match median height ratio.
- Temporarily disable stream-2 (family-room) exclusion ROIs via `config/analytics_exclude_baseline.ini`.
- Manual pitch sweep (-5°, -10°, -15°) using preview calibration files.

Current preview baseline (pending visual confirmation):

- `config/dewarper_family_room_charuco_rtsp.txt`: `dst-focal-length=1447.54380469;1447.01129766`
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`: family-room fx/fy 1447.54/1447.01, cx 514, cy 359.5
- `config/camera_calibration_preview_dewarp_tiltfix_pitch5.json` (family-room pitch -5°)
- camInfo regenerated with `NOESIS_V3DT_CAMINFO_Y_FLIP=0`
- `config/analytics_exclude_baseline.ini` with stream-2 exclude disabled

Diagnostics captured:

- `diagnostics/v3dt_frames_fr_line_pre.ndjson`
- `diagnostics/v3dt_frames_fr_line_post.ndjson`
- `diagnostics/v3dt_frames_fr_line_pitch5.ndjson`
- `diagnostics/v3dt_frames_fr_line_pitch10.ndjson`
- `diagnostics/v3dt_frames_fr_line_pitch15.ndjson`

Next: visually confirm if depth stays positive across the lower frame; if not, re-run Charuco with wider coverage or
refit extrinsics from depth with roll constraint (avoid 180° flips).

## Family-room full-FOV dewarp (2026-01-16)

To prioritize full FoV tracking, switched the dewarper destination K to match the
source focal length and a centered principal point (no zoom/crop).

- `config/dewarper_family_room_charuco_rtsp.txt`:
  - `dst-focal-length=768.7151688932535;768.4323821247384`
  - `dst-principal-point=640.0;360.0`
- `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`:
  - family-room `fx/fy=768.715/768.432`, `cx=640`, `cy=360`

Pre-tilt capture:

- `diagnostics/v3dt_frames_fr_fullfov_pretilt.ndjson`

Auto-tilt (depth, flip-image-y):

- `config/camera_calibration_preview_dewarp_fullfov_tilt.json` (pitch -25.66°, roll 0.09°, `candidate_sign=-1`)
- camInfo regenerated with `NOESIS_V3DT_CAMINFO_Y_FLIP=0`

Post-tilt capture + forensics:

- `diagnostics/v3dt_frames_fr_fullfov_posttilt.ndjson`
- `diagnostics/v3dt_snapshot_20260116_164114.json`
- `diagnostics/v3dt_report_20260116_164123.md`

Result: family-room bbox3d present in all frames, but boxes remain small (H ratio ~0.40,
W ratio ~0.58), reprojection error ~99px, depth ratio ~0.44. Scale sweep indicates
scale=1.0 is best (not a scalar fix).

## Family-room full-FOV + Y-flip baseline (2026-01-16)

After aligning the Charuco principal point and re-running MapAnything tilt, the
best reprojection results are achieved with **invert E + Y-flip** in camInfo.

Config:

- Dewarper (full FOV, centered): `config/dewarper_family_room_charuco_rtsp.txt`
  - `dst-focal-length=768.7151688932535;768.4323821247384`
  - `dst-principal-point=641.3681223147197;356.804752740055`
- Preview intrinsics: `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  - family-room `fx/fy=768.715/768.432`, `cx=641.368`, `cy=356.805`
- Tilt preview: `config/camera_calibration_preview_dewarp_fullfov_tilt.json`
  - pitch **-23.48°**, roll **3.01°**, `candidate_sign=+1`
- CamInfo generation:
  - `NOESIS_V3DT_CAMINFO_INVERT_E=1`
  - `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
  - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`

Diagnostics:

- Log: `diagnostics/v3dt_frames_fr_fullfov_yflip1.ndjson`
- Report: `diagnostics/v3dt_report_20260116_174224.md`
- Metrics (family-room): reproj **26.7 px**, H ratio **0.58**, W ratio **1.28**,
  bottom offset **0.6 px**.

Interpretation: footpoints are aligned (bottom offset near 0), but 3D height is still
short and varies with vertical position in frame. This suggests a remaining mapping
issue beyond simple scalar scaling (likely pitch/geometry refinement).
