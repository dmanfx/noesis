# SV3DT Living-Room Tracking Troubleshooting Log

**Problem**: Living-room camera (source index 0) has intermittent/flickering tracking with SV3DT (stateEstimatorType=3). Kitchen and family-room cameras work well. 2D tracking (stateEstimatorType=1) works fine for all cameras.

**Symptom**: Tracking appears for 1-2 frames then disappears, even when person is standing still. High confidence detections (93%) are shown briefly but don't persist.

---

## Update (2026-01-22)

Living-room SV3DT is now **tracking consistently** with improved projection after
manual pitch sweeps. The locked baseline is documented in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`.

**Current living-room baseline:**
- Pitch: **-15 deg** (preview extrinsics)
- Model height: **2.2 m**
- `NOESIS_V3DT_CAMINFO_Y_FLIP=1`
- `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`
- `NOESIS_V3DT_CAMINFO_INVERT_E=0`
- PGIE aspect ratio **disabled** (`maintain-aspect-ratio=0`, `symmetric-padding=0`)

**Recent live metrics (living-room, user in frame):**
- `rect774_pitch_m16_h220_k21_l15_live_20260122_225218`
  - reproj med **67.1px**
  - bottom offset **160.5px**

**Confirmed no-go for living-room:**
- Pitch **-11**: reproj med **314px** (unstable).
- Pitch **-13**: reproj med **79.8px** (worse than -15).

Remaining shortfall: living-room still appears slightly shallow (bottom offset
~160px). Further tilt refinement may still help, but do **not** regress the
global baseline without a new forensics run.

---

**Note:** The timeline below is historical and includes experiments that are now
superseded (cm-scale camInfo, INVERT_E=1). Use the locked baseline and no-go list
in `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for current guidance.

## Timeline of Attempts

### 1. Initial Diagnosis - Tracker Confidence Thresholds
**What we tried**: 
- Lowered `minDetectorConfidence` from 0.2 → 0.0 (then to 0.1)
- Lowered `ShadowTracker.minDetectorConfidence` from 0.2 → 0.0 (then to 0.1)
- Increased `earlyTerminationAge` from 1 → 10
- Increased `maxShadowTrackingAge` from 150 → 300

**Result**: ❌ No significant improvement. FPS dropped when confidence was too low (0.0).

---

### 2. Tracker Resolution Fix
**What we tried**: Changed `tracker-height` from 1080 → 1056

**Result**: ✅ Fixed pipeline crash error, but tracking still intermittent for living-room.

---

### 3. Unit Scaling Investigation (WORLD_SCALE)
**What we tried**: 
- Discovered SV3DT expects centimeter-scale values
- Changed `NOESIS_V3DT_CAMINFO_WORLD_SCALE` default from 1.0 → 100.0
- Regenerated camInfo files with scaled projection matrices and modelInfo (height=170cm, radius=35cm)

**Result**: ⚠️ Partial improvement. Family-room tracking became stable, but living-room still flickered.

---

### 4. E Matrix Inversion Fix (INVERT_E)
**What we tried**:
- Tracked down a host-RAM leak to the upstream DeepStream `nvtracker` plugin (`mask_params.data` overwrite) and patched it.
- Separately experimented with `NOESIS_V3DT_CAMINFO_INVERT_E` while chasing SV3DT stability.

**Result**: ✅ Memory leak fixed via patched `nvtracker` plugin. `NOESIS_V3DT_CAMINFO_INVERT_E` is not the cause of the leak.

---

### 5. Y-Axis Flip Fix (Y_FLIP)
**What we tried**:
- Discovered "head below feet" in projections for some cameras
- Added `NOESIS_V3DT_CAMINFO_Y_FLIP=1` to flip Y-axis in projection matrix
- Logic: `p[1, :] = -p_orig[1, :] + stream_h * p_orig[2, :]`

**Result**: ✅ All cameras now project "head above feet". Living-room still intermittent.

---

### 6. Calibration Data Consistency Fix
**What we tried**:
- Discovered `camera_calibration.json` had INCONSISTENT units:
  - Kitchen/living-room translations appeared to be in meters
  - Family-room translations appeared to be in centimeters (100x larger)
- Normalized all translations to METERS
- Regenerated camInfo with WORLD_SCALE=100 to convert meters→cm

**Result**: ✅ Improved tracking consistency at the time. This cm-scale camInfo
path is now superseded by the meters baseline; do not reuse without revalidation.

---

### 7. Per-Camera E Matrix Convention Fix
**What we tried**:
- Discovered E matrices had different conventions:
  - Kitchen/family-room: camera→world (needs inversion for P)
  - Living-room: world→camera (should NOT be inverted)
- Added `invert_e` flag per-camera in `camera_calibration.json`
- Modified `generate_v3dt_caminfo.py` and `ds8_runtime.py` to respect per-camera flag

**Result**: ⚠️ Projections now geometrically correct for living-room (verified with test points). Still intermittent.

---

### 8. E Matrix Format Normalization
**What we tried**:
- Transformed living-room's E matrix from world→camera to camera→world
- Set all cameras to `invert_e: true` for consistency
- Verified all E matrices now represent camera→world transform

**Result**: ⚠️ Projections verified correct. Living-room still flickering.

---

### 9. Uncertainty Threshold Relaxation
**What we tried**:
- Increased `locUncertaintyThreshold` from 50.0 → 500.0 (10x)

**Result**: ❌ No noticeable change in behavior.

---

### 10. Track Dump Analysis
**What we found**:
- Track dump files (`/tmp/noesis_track_dump_0.txt`) show many terminated tracks
- 3D position estimates are WILDLY incorrect for living-room tracks:
  - Some positions show values like `-114087 cm` (over 1km away!)
  - This indicates 3D model fitting is numerically unstable

**Insight**: The SV3DT 3D cylinder fitting algorithm is failing intermittently, producing garbage 3D positions.

---

### 11. Projection Matrix Numerical Analysis
**What we found**:
- All cameras have HIGH condition numbers (~2400) - indicates numerical sensitivity
- Living-room: P[2,3] = 166.98 (moderate depth offset)
- Kitchen: P[2,3] = 10.55 (very small - flagged as potential depth instability)
- Family-room: P[2,3] = 1165.45 (large offset)
- Depth sensitivity: ~0.5-3 cm per pixel for all cameras (similar)
- Det(R) = -1.0 for all (improper rotation due to coordinate flip)

**Insight**: Kitchen has the SMALLEST P[2,3] but works fine. Living-room has moderate P[2,3] but fails. So P[2,3] magnitude is NOT the differentiating factor.

---

### 12. Camera Model Verification
**What we found**:
- Living-room/kitchen: `unifi_g3_instant` (native 1920x1080)
- Family-room: `unifi_g4_instant` (native 1280x720)
- Tracker configured for 1920x1056
- Aspect ratio mismatch exists for ALL cameras (scale_x ≠ scale_y)

**Insight**: Aspect ratio mismatch affects all cameras equally, so not the root cause.

---

### 13. Video Analysis (Screen Recording)
**What we observed**:
- Frame 5 (09:48:16): Tracking visible - "person 3 0.93" with pink mask
- Frame 6 (09:48:16): Same second, no tracking visible
- Frames 7-70+: No tracking visible
- Person standing STILL - position not correlated with tracking success
- Track ID = 3 suggests tracks 0, 1, 2 were already created and terminated

**Insight**: Tracks are being rapidly created and terminated. 3D fitting succeeds rarely (~1-2 frames) then fails.

---

## Current State

### What Works:
- Detection (YOLO): ✅ High confidence (93%) detections
- 2D Tracking (stateEstimatorType=1): ✅ Stable for all cameras
- Kitchen SV3DT: ✅ Tracks well
- Family-room SV3DT: ✅ Tracks well

### What Fails:
- Living-room SV3DT (stateEstimatorType=3): ❌ Flickers intermittently
- 3D model fitting produces wildly incorrect positions for living-room

### Current Configuration (updated 2026-01-12):
- camInfo generated in **meters** (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`, now the generator default)
- `NOESIS_V3DT_CAMINFO_INVERT_E=0` (Menon canonical: E is world→camera; set to `1` only if your stored E is camera→world / Twc)
- `NOESIS_V3DT_CAMINFO_Y_FLIP=0` by default (only enable if you have confirmed image-axis convention mismatch)
- SV3DT KF/uncertainty parameters rescaled to **meters** in `config/v3dt/nvtracker_sv3dt.yml` (e.g., `processNoiseVar4Loc` and `locUncertaintyThreshold`)
- `stateEstimatorType=3`
- `config/camera_calibration.json` stores `E` as world→camera (Menon canonical; column-major; translations in meters).

---

## Hypotheses to Test

1. **Living-room calibration data has fundamental errors** - The E matrix values themselves may be incorrect, causing geometry that looks valid but produces unstable 3D fits.

2. **SV3DT has numerical sensitivity to specific projection matrix structures** - Living-room's specific combination of rotation/translation may trigger edge cases.

3. **Camera source index matters** - Something specific about being source 0 may cause issues.

4. **Pose/position-specific fitting failures** - Certain body poses or room positions may cause the 3D cylinder fit to fail more often for living-room's geometry.

5. **Intrinsic/extrinsic interaction** - The combination of living-room's specific intrinsics (G3 instant at 1920x1080) with its extrinsics may create an ill-conditioned system.

---

## Files Modified

- `config/v3dt/nvtracker_sv3dt.yml` - Tracker parameters
- `config/v3dt/camInfo_*.yml` - Generated projection matrices
- `config/camera_calibration.json` - E matrices and invert_e flags
- `scripts/generate_v3dt_caminfo.py` - CamInfo generation script
- `noesis/ds8_runtime.py` - Runtime camInfo auto-generation
- `config/infer_v3dt_medium.yaml` - Pipeline config

---

---

### 14. Deep Numerical Analysis

**Tests performed:**
- Jacobian condition number at typical positions
- Depth sensitivity (dH/dZ) analysis
- X (lateral) position sensitivity
- X-Z coupling analysis
- World coordinate mapping for each camera

**Results:**
| Metric | Living-room | Kitchen | Family-room |
|--------|-------------|---------|-------------|
| Jacobian cond# | 1.06 (best) | 2.43 | 1.63 |
| Depth uncertainty/px | 3.7cm (best) | 8.5cm | 56cm (worst) |
| X uncertainty/px | 1.1cm (best) | 1.6cm | 4.0cm |
| X-Z coupling | 0.23 | 0.10 | 0.00 |

**Conclusion**: Living-room has the BEST numerical properties for 3D tracking!
- Best conditioned Jacobian
- Best depth sensitivity
- Best X sensitivity

**This rules out numerical/geometric instability as the root cause.**

---

## Remaining Hypotheses

Since numerical analysis shows living-room should be the MOST stable:

1. **Calibration data has actual VALUE errors** - The E matrix might not accurately represent living-room's real camera pose, even if format is correct.

2. **Source index 0 bug** - There may be a bug or edge case in nvtracker for the first source.

3. **Detection quality differs** - YOLO might output different bbox quality for living-room due to lighting/scene differences.

4. **Video characteristics** - Living-room video may have different properties affecting tracking.

---

---

### 15. Camera Order Swap Test

**What we tried**:
- Swapped kitchen (was idx 1) to index 0
- Swapped living-room (was idx 0) to index 1
- Updated camInfo order in tracker config to match

**Result**: ❌ Test had side effects (broke kitchen and family-room tracking)

**BUT - Key Finding**:
- **Living-room STILL flickered even at index 1**
- This proves: **Source index 0 is NOT the issue**
- Problem is specific to living-room's calibration or video

**Conclusion**: Ruled out "source index bug" hypothesis.

---

## Remaining Hypotheses (Prioritized)

Since the problem follows living-room specifically:

1. **Living-room calibration has VALUE errors** - The E matrix rotation/translation values may be incorrect even if format is right.

2. **Living-room video characteristics** - Maybe lighting, lens distortion, or video encoding affects YOLO bbox quality.

3. **Living-room intrinsics are wrong** - The camera model might not match actual camera.

---

## Next Steps to Try

1. ~~Swap camera order~~ ✓ Done - ruled out index bug
2. **Use kitchen's E matrix for living-room** - Test if kitchen's calibration produces stable tracking for living-room video (would prove calibration is the issue)
3. **Re-calibrate living-room from scratch** - Get fresh calibration data
4. **Check YOLO bbox quality per camera** - Verify detection consistency

---

### 16. Simplified Calibration Test

**Discovery**: Current living-room calibration has:
- Roll = **12°** (camera tilted sideways?)
- Pitch = **-13°**

User mentioned pitch should be ~11°. The 12° roll is suspicious - if the camera isn't actually tilted sideways, this could cause SV3DT fitting to fail!

**Test**: Replaced living-room E matrix with simplified version:
- 11° pitch only (no roll, no yaw)
- Position: [6.0, 2.6, 0.0] meters
- Backup saved to `config/camera_calibration_backup.json`

**To test**: Run with `NOESIS_V3DT_AUTOGEN_CAMINFO=1`

**Result**: ❌ Still flickers - calibration VALUES are not the issue

---

### 17. Video Source Analysis

**Findings**:
- Living-room: 1920x1080 @ 29.97fps (h264)
- Kitchen: 1920x1080 @ 29.9fps (h264)  
- Family-room: 1280x720 @ 30fps (h264)

All sources look normal. Resolution and frame rate are not the issue.

---

## Summary of Ruled Out Causes

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Source index 0 bug | ❌ Ruled out | Problem follows camera when swapped |
| Numerical instability | ❌ Ruled out | Living-room has BEST condition number |
| E matrix format | ❌ Ruled out | All formats corrected |
| E matrix values (roll/pitch) | ❌ Ruled out | Simplified E still fails |
| P[2,3] depth offset | ❌ Ruled out | Kitchen has smaller P[2,3] and works |
| Pixel height range | ❌ Ruled out | Heights are 154-351px (good range) |
| Video resolution/fps | ❌ Ruled out | Same as kitchen |

---

## Remaining Investigation Needed

1. **Detection bbox quality** - Is YOLO outputting noisy/jittery bboxes for living-room?
2. **Scene content** - Is there something in living-room that confuses detection?
3. **Calibration source** - Where did the original calibration come from? Was it done properly?
4. **Deep SV3DT debugging** - Enable verbose logging in nvtracker to see why fitting fails

---

---

### 18. Persistent Units Drift Fix (cm vs m)

**Discovery**: `config/camera_calibration.json` translations were effectively **centimeter-scale** (e.g., derived camera-center Y ≈ 147–230), but SV3DT camInfo generation assumes calibration is **meters** and applies `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` (meters→cm). This resulted in camInfo projection translations that were **100× too large**, which can make SV3DT’s 3D fitting/association numerically unstable under motion (track termination + missing masks/bboxes).

**Fix (definitive)**:
- Convert stored extrinsics translations in `config/camera_calibration.json` from cm→m.
- Add cm→m translation coercion in `noesis/calibration/manager.py` `CalibrationManager.set_extrinsics()` so future WS `set_extrinsics` writes remain meters (env control: `NOESIS_EXTRINSICS_INPUT_UNITS=auto|m|cm`).
- Update `config/v3dt/camInfo_*.yml` so camInfo↔meters ratio is correct again (expected 100 when camInfo is cm).

**Validation**:
- `pytest tests/test_calibration_manager.py` (PASS; includes an auto cm→m coercion test).
- camInfo ratio-check (`K^-1 @ P` translation vs `inv(E)` translation): ratio_abs(x,y,z) == 100.0 for all cameras (camInfo in cm, calibration in meters).

*Last updated: 2026-01-09*
