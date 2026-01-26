# V3DT Iteration Notes — 2026-01-17 (Family-room pitch sanity check)

**Status (2026-01-22):** Historical log; superseded by the locked baseline in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`. Commands here use
older camInfo polarity and are not the current baseline; revalidate before reuse.

## 1) Pitch-target experiment (family-room)
- Added `scripts/adjust_pitch_calibration.py` to generate preview calibrations with a pitch delta or target.
- Generated preview calibration targeting pitch -5° from the auto-tilt output:
  - `config/camera_calibration_preview_dewarp_fullfov_tilt_targetm5.json`
- Regenerated camInfo for the dewarp preview pipeline:
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml`
- Ran DS8 preview and captured diagnostics:
  - Log: `diagnostics/v3dt_frames_fr_fullfov_tilt_targetm5.ndjson`
  - Snapshot: `diagnostics/v3dt_snapshot_20260117_002858.json`
  - Report: `diagnostics/v3dt_report_20260117_002906.md`

Result:
- Pitch-target run degraded family-room metrics (H ratio ~0.20, reproj ~39px),
  with stronger size collapse at lower image rows. Restored camInfo based on
  auto-tilt (`config/camera_calibration_preview_dewarp_fullfov_tilt.json`).

## 2) Streammux 720 tests
- Attempted a single-source 1280x720 pipeline:
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_720.yaml`
  - `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_720.yml`
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp_720/`
- Initial run failed because MapAnything engine is fixed batch=3; kept this config
  with `models.mapanything.enable=false` for single-source diagnostics only.

- Ran a multi-source 720 streammux test (keeps batch=3 to match MapAnything engine):
  - Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_mux720.yaml`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_mux720.yml`
  - CamInfo: `config/v3dt_preview_charuco_fr_rtsp_dewarp_mux720/`
  - Log: `diagnostics/v3dt_frames_fr_mux720.ndjson`
  - Snapshot: `diagnostics/v3dt_snapshot_20260117_004242.json`
  - Report: `diagnostics/v3dt_report_20260117_004250.md`

Result:
- Reprojection improved (19.6px) but family-room height ratio stayed ~0.51 and
  depth ratio ~0.31. Streammux 1080 vs 720 is **not** the root cause of the
  size collapse; distortion/geometry mismatch remains.

## 3) Dewarper sanity check (inconclusive)
- Temporarily enabled `dewarp-dump-frames=1` in `config/dewarper_family_room_charuco_rtsp.txt`
  and captured RGBA dumps. Converted the latest dump to PNG and compared against
  an OpenCV-undistorted frame from `/home/mayor/Downloads/familyroomclip.mp4`.
- Basic pixel-diff comparison showed similar deltas to the original frame, but
  without exact frame alignment the result is not conclusive.
- Restored `dewarp-dump-frames=0` to avoid FPS impact.

## 4) Snapshot scaling alignment (2026-01-17)
- `noesis/ds8_runtime.py` now prefers the cameras YAML model resolution for
  snapshot intrinsics scaling (uses `_camera_model_res[camera_id]`), before
  falling back to legacy specs or cx/cy guesses. This keeps snapshot/debug
  intrinsics aligned with `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  (resolution `[1280, 720]`) and matches camInfo scaling behavior.

## 5) Dewarper coefficient order + baseline settings (2026-01-18)
- Corrected `nvdewarper` distortion coefficient order for family-room:
  - `config/dewarper_family_room_charuco_rtsp.txt`
  - Order now matches nvdewarper docs: radial `k1,k2,k3` then tangential `p1,p2`.
- De-hacked family-room intrinsics (restored calibrated fx/fy) and regenerated camInfo:
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_*.yml`
- Tracker tweaks rolled back to baseline after regression test:
  - `TargetManagement.minTrackerConfidence=0.2`
  - `TargetManagement.earlyTerminationAge=10`
- Baseline camInfo generation settings (current best-known for family-room):
  - `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`
  - `NOESIS_V3DT_CAMINFO_INVERT_E=0`
  - `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
  - `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
  - `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`
- Post-dehack run (session `tuned_20260118_144158`, filtered via `run_info.txt`):
  - Kitchen tracking strong (track frames ~90.6%, median length ~50.5 frames)
  - Family-room coverage low (~10.3% frames with tracks), with high det_conf_end
    but low tracker_conf_end (~0.15) and stable world_valid/bbox3d.

## 6) Rectified K recompute + tilt refresh (2026-01-18)
- Recomputed rectified K using OpenCV `getOptimalNewCameraMatrix` with corrected
  distortion order (k1,k2,p1,p2,k3) from `diagnostics/charuco_family_room_rtsp_1280x720.json`:
  - New rectified K: fx=774.5219229701734, fy=774.237000074667, cx=639.5, cy=359.5
- Updated dewarper dst K + rectified intrinsics:
  - `config/dewarper_family_room_charuco_rtsp.txt` (`dst-focal-length`, `dst-principal-point`)
  - `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` (rectified fx/fy/cx/cy)
- Regenerated camInfo in `config/v3dt_preview_charuco_fr_rtsp_dewarp/`.
- Attempted to regenerate tilt extrinsics via `scripts/auto_tilt_from_depth.py`, but
  depth cache was empty (`no_depth`). Next: run DS8 with `--depth-enable-seconds`
  to populate MapAnything depth snapshots, then rerun tilt preview.
