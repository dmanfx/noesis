# V3DT Iteration Notes — 2026-01-22

Goal: refine family-room V3DT tracking via pitch sweep and model height checks.

## Pitch sweep (model height 2.0)

- Calibration: `config/archive/camera_calibration_preview_dewarp_fr_pitch_m11.json`, `config/archive/camera_calibration_preview_dewarp_fr_pitch_m15.json`, `config/archive/camera_calibration_preview_dewarp_fr_pitch_m17.json`
- CamInfo dirs: `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m11_h200/`, `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m15_h200/`, `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m17_h200/`
- Pipeline configs:
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m11_h200.yaml`
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m15_h200.yaml`
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m17_h200.yaml`
- Forensics results (family-room):
  - m11: reproj med 171px, H ratio 1.32, depth ratio 4.36, tracks 7
  - m15: reproj med 48.2px, H ratio 1.14, depth ratio 3.95, tracks 31 (best)
  - m17: 0 tracks (no family-room detections)
- Conclusion: pitch -15 is best so far for family-room.

## Model height check (pitch -15, height 2.6)

- CamInfo dir: `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m15_h260/`
- Pipeline config: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m15_h260.yaml`
- Run `rect774_pitch_m15_h260_20260122_200835` had 0 tracks across cameras (scene likely empty), so height impact is inconclusive.
- Run `rect774_pitch_m15_h260_test_20260122_201427` produced family-room tracks but severe projection errors (reproj med 763.5px, H ratio 4.50, bottom off 986.7px), so height 2.6 appears too tall for the current calibration.

## Prepared height variants (pitch -15)

- CamInfo dirs:
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m15_h220/`
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m15_h240/`
- Tracker configs:
  - `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m15_h220.yml`
  - `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m15_h240.yml`
- Pipeline configs:
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m15_h220.yaml`
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m15_h240.yaml`

## Height sweep results (pitch -15)

- Run `rect774_pitch_m15_h220_test_20260122_202217`:
  - family-room reproj med 52.2px, H ratio 1.21, bottom off 61.5px, tracks 2695 (best so far)
- Run `rect774_pitch_m15_h240_test_20260122_202501`:
  - family-room reproj med 179.4px, H ratio 1.79, bottom off 217.3px, tracks 2158 (worse than h220)
- Conclusion: pitch -15 with model height 2.2 is the current best configuration for family-room.

## Fine pitch sweep (model height 2.2)

- Calibration: `config/archive/camera_calibration_preview_dewarp_fr_pitch_m14.json`, `config/archive/camera_calibration_preview_dewarp_fr_pitch_m16.json`
- CamInfo dirs:
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m14_h220/`
  - `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220/`
- Pipeline configs:
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m14_h220.yaml`
  - `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m16_h220.yaml`
- Run `rect774_pitch_m14_h220_20260122_203937`:
  - family-room reproj med 127.0px, H ratio 1.33, bottom off 184.9px, tracks 21 (worse than m15)
- Run `rect774_pitch_m16_h220_20260122_204148`:
  - family-room reproj med 30.7px, H ratio 1.18, bottom off 57.7px, tracks 194 (best reproj so far)
- Conclusion: pitch -16 with model height 2.2 is the current best for family-room.

## Kitchen + living-room pitch sweep (family-room locked at -16, model height 2.2)

- Current pitch baselines from `camera_calibration_preview_dewarp_fr_pitch_m16.json`:
  - kitchen: -19°, living-room: -9°, family-room: -16°
- Auto-tilt from depth (`scripts/auto_tilt_from_depth.py`) times out due to depth-store scan latency; proceeding with manual pitch sweeps.
- Variant A (kitchen -20, living-room -10):
  - Calibration: `config/archive/camera_calibration_preview_dewarp_fr_pitch_m16_kitchen_m20_living_m10.json`
  - CamInfo dir: `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220_k20_l10/`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220_k20_l10.yml`
  - Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m16_h220_k20_l10.yaml`
- Variant B (kitchen -21, living-room -11):
  - Calibration: `config/archive/camera_calibration_preview_dewarp_fr_pitch_m16_kitchen_m21_living_m11.json`
  - CamInfo dir: `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220_k21_l11/`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220_k21_l11.yml`
  - Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m16_h220_k21_l11.yaml`
- Run `rect774_pitch_m16_h220_k20_l10_20260122_214444`:
  - kitchen reproj med 91.2px, bottom off 95.6px; living-room and family-room had no tracks (likely empty)
- Run `rect774_pitch_m16_h220_k21_l11_20260122_214858`:
  - kitchen reproj med 45.8px, bottom off 46.9px (better than baseline)
  - living-room reproj med 314.2px, bottom off 396.4px (still poor)
  - family-room reproj med 22.3px
- Conclusion: kitchen improves at -21; living-room needs further tilt sweep.

### Next variants (living-room deeper tilt, kitchen locked -21)

- Variant C (kitchen -21, living-room -13):
  - Calibration: `config/archive/camera_calibration_preview_dewarp_fr_pitch_m16_kitchen_m21_living_m13.json`
  - CamInfo dir: `config/v3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220_k21_l13/`
  - Tracker: `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_rect774_pitch_m16_h220_k21_l13.yml`
  - Pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_tuned_rect774_pitch_m16_h220_k21_l13.yaml`
- Variant D (kitchen -21, living-room -15):
  - Calibration: `config/archive/calibration_v3dt_baseline.json`
  - CamInfo dir: `config/v3dt/caminfo_baseline/`
  - Tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
  - Pipeline: `config/infer_v3dt_baseline.yaml`
- Run `rect774_pitch_m16_h220_k21_l13_20260122_215449`:
  - kitchen reproj med 46.0px, bottom off 47.1px (stable)
  - family-room reproj med 18.9px
  - living-room had no tracks (needs presence)
- Run `rect774_pitch_m16_h220_k21_l15_20260122_220402`:
  - kitchen reproj med 46.1px, bottom off 47.2px
  - family-room reproj med 44.8px (worse than l13 run)
  - living-room had no tracks (needs presence)
- Live run `rect774_pitch_m16_h220_k21_l13_live_20260122_224741`:
  - living-room reproj med 79.8px, bottom off 202.5px
  - kitchen reproj med 46.0px, bottom off 47.1px
- Live run `rect774_pitch_m16_h220_k21_l15_live_20260122_225218`:
  - living-room reproj med 67.1px, bottom off 160.5px (better than l13 live)
  - kitchen reproj med 46.0px, bottom off 47.1px
- Conclusion: keep kitchen at -21 and living-room at -15 for now; living-room still shallow but improved.

## BEV alignment scale (meters)

- Updated `config/ply_alignment.json` `units.s_obj_to_m` from 0.010849... to 1.0 to keep world scaling in meters (matches calibration extrinsics).

## Validation

- Captures run via `scripts/run_v3dt_capture.sh` with `CAPTURE_SECONDS=120` and session-specific `NOESIS_V3DT_DIAG_SESSION`.
