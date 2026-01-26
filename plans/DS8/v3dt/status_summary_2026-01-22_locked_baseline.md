# V3DT Status Summary — 2026-01-22 (Locked Baseline)

This document captures the current **SV3DT working baseline**, why it works, the
known shortfalls, and the **confirmed no-go** changes that should not be repeated.

## Current baseline (SV3DT only, DS8 only)

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
- CamInfo dir: `config/v3dt/caminfo_baseline/`
- Cameras config: `config/cameras_v3dt_baseline.yaml`
- Calibration (baseline): `config/archive/calibration_v3dt_baseline.json`
- Dewarper (family-room): `config/dewarper_v3dt_baseline.txt`
- Alignment: `config/ply_alignment.json` with `units.s_obj_to_m=1.0`
- Tracking mode: `v3dt` (`NOESIS_TRACKING_MODE=v3dt` or `--tracking-mode v3dt`)

### Per-camera pitch (preview overrides)

- family-room: **-16 deg** (best reprojection so far)
- kitchen: **-21 deg** (improves reproj + bottom offset)
- living-room: **-15 deg** (improves reproj; still shallow)

### Model dimensions (SV3DT)

- model height: **2.2 m**
- model radius: **0.35 m**

### Required camInfo/env conventions

- `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`
- `NOESIS_V3DT_CAMINFO_INVERT_E=0` (calibration `E` is already world→camera)
- `NOESIS_V3DT_CAMINFO_Y_FLIP=1` (required)
- `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy` (required; SV3DT is Z-up)
- `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1` (meters)
- Use `NOESIS_V3DT_AUTOGEN_CAMINFO=0` with the pre-generated camInfo dir above.

### Why this baseline works (key factors)

- **Pitch sweep per camera** reduced the projected-box bottom offset (footpoint)
  and stabilized the SV3DT fit.
- **Y-flip + world-axis remap (xzy)** aligns SV3DT’s Z-up assumption with the
  Y-up calibration frame.
- **Streammux-only scaling** to 1920×1080 avoids double aspect correction.
- **Meters everywhere** (extrinsics, camInfo, alignment scale) prevents unit drift.

## Recent validation results (live)

From living-room live runs (user in frame):

- `rect774_pitch_m16_h220_k21_l13_live_20260122_224741`  
  living-room reproj med 79.8px, bottom off 202.5px.
- `rect774_pitch_m16_h220_k21_l15_live_20260122_225218`  
  living-room reproj med 67.1px, bottom off 160.5px (better).

From kitchen runs:

- kitchen reproj med ~46px, bottom off ~47px (stable across variants).

Family-room best runs (non-live):

- `rect774_pitch_m16_h220_20260122_204148`  
  family-room reproj med 30.7px, bottom off 57.7px.

## Shortfalls (known gaps)

- **Living-room still shallow:** bottom offset ~160px and reproj ~67px; may need
  additional tilt or a better extrinsics solve.
- **Depth ratio off** for family-room in several runs (depth bias remains).
- **BEV not yet validated** against this baseline; fix after SV3DT stability.
- **MV3DT disabled** pending shared-world calibration.

## Confirmed no-go items (do not repeat)

- **Do not enable PGIE `maintain-aspect-ratio` or `symmetric-padding`.**  
  This double-applies aspect correction and breaks SV3DT projection.
- **Do not change `units.s_obj_to_m` away from 1.0** with meter extrinsics.  
  `0.010849...` shrinks world coords ~100× and corrupts BEV/pixel→world.
- **Do not set `NOESIS_V3DT_CAMINFO_INVERT_E=1`** for the current calibration.  
  It inverts an already world→camera `E` and destabilizes SV3DT.
- **Do not use model height ≥2.4 m** (2.6 was especially bad for family-room).
- **Do not use pitch -17 for family-room** (no tracks observed).
- **Avoid `scripts/auto_tilt_from_depth.py` for fast iteration.**  
  Depth store access is currently too slow (timeouts), so manual pitch sweeps are
  more reliable for now.

## Canonical runtime command

```bash
NOESIS_V3DT_AUTOGEN_CAMINFO=0 \
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer_v3dt_baseline.yaml \
  --cameras-config config/cameras_v3dt_baseline.yaml
```
