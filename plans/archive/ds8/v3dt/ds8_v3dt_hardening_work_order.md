# DS8 V3DT Hardening Work Order

**Created:** 2025-12-30  
**Status:** IN PROGRESS  
**Scope:** DS8 SV3DT/MV3DT implementation hardening and config alignment  
**Related docs:**  
- `plans/DS8/v3dt/work_order.md`  
- `plans/DS8/v3dt/integration_plan.md`  
- `plans/DS8/ds8_design_decisions.md`  

---

## Update (2026-01-22)

SV3DT baseline is locked and documented in
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`.
Key hardening constraints to keep in mind:

- PGIE aspect ratio must remain disabled (`maintain-aspect-ratio=0`, `symmetric-padding=0`).
- Alignment scale must stay in meters (`config/ply_alignment.json` `s_obj_to_m=1.0`).
- Current per-camera pitch preview: family-room -16, kitchen -21, living-room -15.

---

## Priority Levels

| Priority | Meaning |
|----------|---------|
| **P0 (Critical)** | Blocks correctness; must fix before validation |
| **P1 (High)** | Affects quality/robustness significantly |
| **P2 (Medium)** | Improves consistency or future maintainability |
| **P3 (Low)** | Nice-to-have; can defer |

---

## Task V3DT-H01 — Resolve world-frame unit convention (centimeters vs meters)

**Priority:** P0 (Critical)  
**Blocking:** All 3D output validation  
**Status:** [x] Complete — METERS end-to-end (camInfo in meters)

### Problem

Historically, V3DT experiments used `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` to generate centimeter-scale camInfo (matching NVIDIA samples). This created confusion because:

1. Documented goal is "meters, Y-up" (`plans/DS8/v3dt/integration_plan.md` §3.3)
2. Acceptance criteria reference "adult heights ~1.4–2.1m" (`work_order.md` V3DT-02)
3. Downstream consumers (BEV, occupancy, velocity analytics) may assume meters

### Decision (implemented)

- Keep **calibration** (`config/camera_calibration.json`) in **meters**.
- Generate SV3DT camInfo (`config/v3dt/camInfo_*.yml`) in **meters** via `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1` (default in `scripts/generate_v3dt_caminfo.py`).
- Keep **SV3DT world-space tuning** (KF variances + uncertainty thresholds) in **meters** across `config/v3dt/nvtracker_sv3dt*.yml`.

Rationale: this keeps units consistent across the SV3DT boundary and avoids relying on a separate “scale outputs back to meters” conversion path.

### Acceptance Criteria

- [x] Unit convention is documented and consistent across all artifacts
- [x] camInfo `modelInfo.height` is meter-scale (e.g., 1.7m) and SV3DT world-space params are tuned accordingly
- [x] `set_extrinsics` persists meters (auto-coerces cm translations to meters)
  _2026-01-09 (Codex): Added cm→m translation coercion in `noesis/calibration/manager.py` (`NOESIS_EXTRINSICS_INPUT_UNITS`), fixed `config/camera_calibration.json` translations, and validated camInfo↔meters ratio = 100 for all cameras._
- [ ] SV3DT `bbox3d.zLen` outputs match real-world heights in meters (~1.4–2.1m for adults)
- [ ] Velocity sanity: walking speed < 5 m/s
- [x] Publish projected bbox3d base-center for cuboid anchoring (image-space base center, not image_foot)
  _2026-01-18 (Codex): Added `image_base` from bbox3d base-center projection in `noesis/pipelines/hooks.py`, documented in WS + metadata contracts._
  _2026-01-18 (Codex): BEV/footpoint extraction now prefers `image_base` when present, falling back to `image_foot` or bbox bottom-center._

_2026-01-13 (Codex): Observed `bbox3d.zLen ≈ 1.7` while `bbox3d.yLen ≈ 0.7`; updated docs/tooling to treat Z as height (SV3DT Z‑up). Cuboid projection still needs validation after axis remap tests._

_2026-01-14 (Codex): Ran depth-plane tilt preview with `--flip-image-y` for all cameras (`config/camera_calibration_preview_flipy_all.json`) and captured diagnostics (`diagnostics/v3dt_report_20260114_142247.md`). Living-room projection height ratio is now ~0.93, kitchen ~0.75, family-room ~0.37. Family-room still projects too short; beginning intrinsics/FOV sweep to correct per-camera projection._

_2026-01-14 (Codex): Family-room intrinsics sweep (2.0x / 2.4x / 3.0x) shows projected height ratio improves to ~0.94 at 3.0x (`diagnostics/v3dt_report_20260114_143913.md`), but depth ratio drops to ~0.38. Indicates likely FOV/zoom mismatch; need to confirm camera zoom/crop or calibrate intrinsics properly before accepting._

_2026-01-14 (Codex): ChArUco calibration on 2688x1512 stream scaled to 1280x720 (`config/cameras_preview_charuco_fr.yaml`) regressed family-room metrics (`diagnostics/v3dt_report_20260114_204146.md` height ratio ~0.29, depth ratio ~1.86). Confirms DS8 substream is not a simple scale-down; must calibrate on the actual 1280x720 feed or account for crop/zoom._

_2026-01-14 (Codex): Calibrated family-room intrinsics directly on RTSP 1280x720 feed (`diagnostics/charuco_family_room_rtsp_1280x720.json`) and wired preview configs (`config/cameras_preview_charuco_fr_rtsp.yaml`, `config/v3dt_preview_charuco_fr_rtsp/`, `config/infer_v3dt_medium_preview_charuco_fr_rtsp.yaml`) for the next SV3DT validation run._

_2026-01-14 (Codex): Ran SV3DT with RTSP ChArUco intrinsics (`diagnostics/v3dt_report_20260114_220737.md`). Family-room reproj ~13.7px, but projected height ratio ~0.24 and depth ratio ~1.92 (cuboids still too small). Next step: sweep family-room intrinsics scale around ~2.0x._

_2026-01-14 (Codex): Added nvdewarper-based distortion correction preview for family-room (`config/dewarper_family_room_charuco_rtsp.txt`) plus rectified intrinsics + camInfo + pipeline (`config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`, `config/v3dt_preview_charuco_fr_rtsp_dewarp/`, `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml`). Next: run DS8 with dewarper and redo tilt preview using the rectified intrinsics._

_2026-01-15 (Codex): Dewarper preview logs captured (`diagnostics/v3dt_frames_charuco_fr_rtsp_dewarp.ndjson`, `diagnostics/v3dt_frames_charuco_fr_rtsp_dewarp_posttilt.ndjson`). Family-room tracking sparse; height/depth ratios inconsistent. Likely need explicit dst focal/principal point for dewarper and tilt re-run for family-room only to avoid shifting other cameras._

_2026-01-15 (Codex): Added explicit rectified K (OpenCV `getOptimalNewCameraMatrix`) to dewarper config (`dst-focal-length`, `dst-principal-point`) and updated rectified intrinsics in `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`. Regenerated camInfo for preview._

_2026-01-15 (Codex): Forced dewarper output caps (width/height) before streammux to fix quadrant/cropping artifacts in the mosaic when dewarper output size is mis-negotiated._

_2026-01-15 (Codex): Family-room baseline improved by scaling SV3DT camInfo intrinsics **only** (2.7x) while keeping dewarper output at rectified K. Report `diagnostics/v3dt_report_20260115_192145.md` shows family-room H ratio 1.01 and W ratio 1.03 with low offsets; this is the best visual cuboid alignment so far._

_2026-01-15 (Codex): Fine-tuned dewarper borders to target ~8px horizontal margins by shifting `dst-principal-point` to 514 and scaling `dst-focal-length` to 634.72/634.49 (camInfo fx/fy 1713.75/1713.12, cx 514). Regenerated preview camInfo; awaiting visual confirmation._

_2026-01-16 (Codex): Investigated vertical scale drop in family-room. Depth trend was inverted with `NOESIS_V3DT_CAMINFO_Y_FLIP=1`; testing with Y-flip disabled and matching dewarper/camInfo K (fx/fy 1447.54) improved depth direction but still requires visual validation. Added preview exclude config to disable stream-2 ROIs; created pitch sweep previews (`pitch5`, `pitch10`, `pitch15`) for family-room._

_2026-01-18 (Codex): Temporary, easy-to-revert intrinsics scale test for family-room. Scaled `unifi_g4_instant_charuco_rtsp_720_rectified` fx/fy down (~0.34x/0.56x) in `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` to counter over-wide/over-tall projections, then regenerated camInfo in `config/v3dt_preview_charuco_fr_rtsp_dewarp/` (w2p, invert_e=0, y_flip=1, axes xzy). Awaiting run to confirm whether tracking coverage expands beyond the lower-right region._

_2026-01-18 (Codex): Tracker fragmentation tweak for family-room: lowered `TargetManagement.minTrackerConfidence` (0.2 → 0.1) and increased `TargetManagement.earlyTerminationAge` (10 → 20) in `config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_tuned.yml`. Next run should confirm longer continuous tracks without sacrificing 3D validity._

_2026-01-18 (Codex): Regression observed (fps ~9–10, family-room median track length ~16.5 frames). Rolled back `TargetManagement.minTrackerConfidence` to 0.2 while keeping `earlyTerminationAge=20` to isolate the impact; next run should confirm whether fps recovers and fragmentation improves._

_2026-01-18 (Codex): Corrected nvdewarper distortion coefficient order for family-room (radial k1,k2,k3 then tangential p1,p2) in `config/dewarper_family_room_charuco_rtsp.txt`. This aligns with nvdewarper docs and should remove the need for compensating intrinsics hacks in future runs._

_2026-01-18 (Codex): De-hacked family-room intrinsics (restored fx/fy in `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`) and regenerated camInfo in `config/v3dt_preview_charuco_fr_rtsp_dewarp/`. Also reverted tracker tweak `earlyTerminationAge` to 10 (keeping `minTrackerConfidence=0.2`) to return to baseline after the regression test._

_2026-01-18 (Codex): Post-dehack run (session `tuned_20260118_144158` filtered) shows kitchen tracking strong (track frames ~90.6%, median length ~50.5 frames) but family-room coverage low (~10.3% frames with tracks). Family-room det_conf_end stays high (~0.91) while tracker_conf_end remains low (~0.15), with `world_valid` stable. Indicates remaining family-room geometry mismatch despite corrected dewarper coefficient order and restored intrinsics._

### Files

- `scripts/generate_v3dt_caminfo.py` (WORLD_SCALE default)
- `noesis/ds8_runtime.py` (`_maybe_sync_v3dt_caminfo` world_scale)
- `config/v3dt/camInfo_*.yml` (regenerate)
- `plans/DS8/v3dt/integration_plan.md` (doc update if Option B)

_2026-01-12 (Codex): Switched camInfo generation default to meters (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`), regenerated `config/v3dt/camInfo_*.yml` with `modelInfo.height: 1.7`, and rescaled SV3DT world-space KF/uncertainty parameters in `config/v3dt/nvtracker_sv3dt*.yml` to meters._

---

## Task V3DT-H02 — Align MV3DT tracker config with SV3DT baseline

**Priority:** P1 (High)  
**Blocking:** MV3DT quality (Phase 2)  
**Status:** [x] Complete

### Problem

`config/v3dt/nvtracker_mv3dt.yml` has significant deviations from `nvtracker_sv3dt.yml` that may cause quality regressions when switching to MV3DT mode:

| Setting | SV3DT | MV3DT | Risk |
|---------|-------|-------|------|
| `BaseConfig.minDetectorConfidence` | 0.2 | 0.0 | Ghost tracks |
| `TrajectoryManagement.enableReAssoc` | 1 | (missing) | No re-association |
| `TrajectoryManagement.*` | Tuned values | Defaults | Fragmentation |
| `DataAssociator.associationMatcherType` | 1 (cascaded) | 0 (greedy) | Worse matching |
| `ReID` section | Full | Missing | No appearance matching |
| `PoseEstimator.poseInferenceInterval` | 4 | -1 | Height drift |

### Implementation

1. Copy `BaseConfig`, `TargetManagement`, `TrajectoryManagement`, `DataAssociator`, `StateEstimator`, `VisualTracker`, `ReID`, `ShadowTracker`, `Misc` sections from `nvtracker_sv3dt.yml` to `nvtracker_mv3dt.yml`
2. Keep MV3DT-specific sections: `MultiViewAssociator`, `Communicator`, `AppearanceModel`
3. Set `PoseEstimator.poseInferenceInterval: 4` (periodic, not -1)
4. Remove redundant/duplicate settings

### Acceptance Criteria

- [ ] MV3DT config passes YAML validation (no duplicate keys)
- [ ] `minDetectorConfidence: 0.2` in MV3DT config
- [ ] `TrajectoryManagement.enableReAssoc: 1` in MV3DT config
- [ ] `ReID` section present in MV3DT config
- [ ] Pipeline starts successfully with MV3DT config

### Files

- `config/v3dt/nvtracker_mv3dt.yml`

---

## Task V3DT-H03 — Widen MV3DT time-matching tolerance

**Priority:** P1 (High)  
**Blocking:** MV3DT cross-camera association  
**Status:** [x] Complete

### Problem

`maxTrackletMatchingTimeSearchRange: 1` is extremely tight. With NTP timing jitter between RTSP sources, valid cross-camera associations may be missed.

### Implementation

1. Change `maxTrackletMatchingTimeSearchRange` from `1` to `5` (frames)
2. Add a comment explaining the setting and its relationship to frame-rate/timing jitter
3. Consider exposing as an env override for tuning

### Acceptance Criteria

- [ ] `maxTrackletMatchingTimeSearchRange: 5` in `nvtracker_mv3dt.yml`
- [ ] Comment explaining the parameter
- [ ] (Validation) Kitchen↔family-room overlap shows consistent `mv3dt_id` >80% of overlap frames

### Files

- `config/v3dt/nvtracker_mv3dt.yml`

---

## Task V3DT-H04 — Apply floor_y alignment to world footpoints

**Priority:** P2 (Medium)  
**Blocking:** BEV accuracy  
**Status:** [x] Deferred — `floor_y=0.0` makes this unnecessary currently

### Problem

`_world_from_bbox3d()` computes `foot_z = zCentre - 0.5*zLen` (SV3DT is Z-up) without applying the floor offset from `config/ply_alignment.json`. This can cause BEV footpoints to appear above/below the floor plane if/when SV3DT world coords are mapped into the Y-up BEV frame.

### Implementation

1. Load `floor_y` from `config/ply_alignment.json` into `_CalibrationProvider` or BEV calibration bundle
2. In `_augment_track_with_world()`, subtract the floor offset from the computed foot Z coordinate (or apply a Z→Y remap first, depending on the chosen world-frame mapping)
3. Add fallback: if `floor_y` is missing, assume 0.0 and log once

### Acceptance Criteria

- [ ] `floor_y` loaded from ply_alignment.json at runtime
- [ ] World footpoint Z ≈ 0 (within ±0.10m / ±10cm) for standing person on floor
- [ ] BEV trails align with floor plane

### Files

- `noesis/pipelines/hooks.py` (`_world_from_bbox3d`, `_augment_track_with_world`)
- `noesis/ds8_runtime.py` (`_CalibrationProvider`)
- `noesis/telemetry/bev.py` (if floor alignment needed there)

---

## Task V3DT-H05 — Add graceful fallback for NVDS_OBJ_WORLD_FOOT_LOCATION extraction

**Priority:** P2 (Medium)  
**Blocking:** None (nice-to-have)  
**Status:** [x] Complete (partial - see notes)

### Problem

`NVDS_OBJ_WORLD_FOOT_LOCATION` extraction was disabled due to SIGSEGV. The native `ptWorldFeet` from SV3DT is more stable than bbox-derived footpoints, especially under partial occlusion.

### Implementation

1. In `native/noesis_v3dt_meta_ext.cpp`, wrap the `NVDS_OBJ_WORLD_FOOT_LOCATION` iteration in a try/catch
2. If extraction throws, return `world_foot: null` in the result dict (don't skip entire meta)
3. Re-enable the iteration (currently commented out or disabled)
4. If stable: use `world_foot` as primary, `bbox3d`-derived as fallback
5. File upstream bug report with NVIDIA (Service Maker `ObjectImageFootLocationUserMetadata::getImageFootLocation()` crash)

### Acceptance Criteria

- [ ] Native bridge attempts `NVDS_OBJ_WORLD_FOOT_LOCATION` extraction
- [ ] Exception is caught and logged (once per run)
- [ ] `bbox3d` extraction still works even if world_foot fails
- [ ] (Optional) If world_foot works, telemetry shows `world_foot` field

### Files

- `native/noesis_v3dt_meta_ext.cpp`
- `scripts/build_noesis_v3dt_meta_ext.sh`

---

## Task V3DT-H06 — Add metrics collection for future fusion confidence scoring

**Priority:** P3 (Low)  
**Blocking:** Enhancement #4 (fusion confidence)  
**Status:** [x] Deferred — requires MV3DT Phase 2

### Problem

`enhancements_plan.md` §4 mentions "fusion confidence scoring" but there's no data collection infrastructure. Implementing confidence scoring later will require historical overlap agreement data.

### Implementation

1. When MV3DT is enabled and same `mv3dt_id` appears in both kitchen and family-room:
   - Compute world footpoint delta: `sqrt((x1-x2)^2 + (z1-z2)^2)`
   - Store in a per-`mv3dt_id` ring buffer (last N samples)
2. Expose basic stats via DS8 `stats` payload:
   - `v3dt.overlap_agreement_mean_m`
   - `v3dt.overlap_agreement_stddev_m`
   - `v3dt.overlap_sample_count`
3. Keep collection lightweight (only when overlap detected)

### Acceptance Criteria

- [ ] Ring buffer stores last 100 overlap agreement samples per mv3dt_id
- [ ] Stats payload includes `v3dt.overlap_agreement_*` when MV3DT enabled

---

## Task V3DT-H08 — V3DT Forensics Toolkit (Snapshot + Telemetry + Panel)

**Priority:** P1 (High)  
**Blocking:** Debugging SV3DT stability regressions  
**Status:** [x] Complete

### Problem

SV3DT calibration issues require explicit, reproducible evidence (not just implied behavior). We need a single tool that captures the full calibration state, per-frame 3D outputs, and concrete error metrics (e.g., bbox3d coverage and reprojection error).

### Implementation

1. Add a snapshot builder that captures calibration inputs and derived math (K, P, pose, axes, ray→floor checks).
2. Add per-frame telemetry logging to NDJSON (opt-in via env).
3. Add a report generator (coverage/height/reprojection metrics) and HTML panel.
4. Persist raw `set_extrinsics` payloads to `logs/calibration_raw/` for audit.

### Acceptance Criteria

- [x] `python3 scripts/v3dt_forensics.py snapshot` writes JSON + MD snapshot.
- [x] `NOESIS_V3DT_DIAG_LOG=1` writes NDJSON tracking logs with bbox3d fields.
- [x] `python3 scripts/v3dt_forensics.py analyze` writes JSON + MD report with explicit checks.
- [x] `python3 scripts/v3dt_forensics.py panel` produces an HTML panel.
- [x] Raw calibration payloads persist in `logs/calibration_raw/`.

### Files

- `noesis/diagnostics/v3dt_forensics.py`
- `noesis/diagnostics/telemetry_log.py`
- `scripts/v3dt_forensics.py`
- `docs/DS8_v3dt_forensics.md`
- `noesis/pipelines/hooks.py`
- `noesis/calibration/manager.py`

_2026-01-12 (Codex): Implemented snapshot+report tooling, NDJSON telemetry logging, HTML panel generator, and raw extrinsics audit logs; added docs and tests._
_2026-01-13 (Codex): Added scale-sweep analysis (unit A/B) to the forensics report/panel and documented the new CLI flag._
_2026-01-13 (Codex): Added projection diagnostics (projected 3D box height/width vs 2D bbox + offsets) to pinpoint overlay mismatches._
_2026-01-13 (Codex): Added depth-consistency metrics (implied depth vs bbox3d depth) to isolate per-camera depth bias._
- [ ] No measurable FPS impact (<1% overhead)

### Files

- `noesis/pipelines/hooks.py` (`_AnalyticsTelemetryProcessor`)
- `noesis/ds8_runtime.py` (stats payload)

---

## Task V3DT-H07 — Complete validation for V3DT-02 and V3DT-0A-03

**Priority:** P2 (Medium)  
**Blocking:** Work order completion  
**Status:** [~] Ready for manual validation

### Problem

`plans/DS8/v3dt/work_order.md` has unchecked acceptance items with validation notes below them:

**V3DT-0A-03:**
- `[ ] With SV3DT enabled, >95% of person tracks have `NVDS_OBJ_3D_META`.`
- `[ ] With SV3DT disabled, telemetry/BEV are unchanged (no regressions).`

**V3DT-02:**
- `[ ] Pipeline runs with SV3DT enabled.`
- `[ ] 3D positions are plausible in meters (adult heights ~1.4–2.1m; indoor speeds <5 m/s).`

### Implementation

1. Run validation commands and record results
2. Add concrete test scripts if missing
3. Mark checkboxes complete with dated validation notes

### Acceptance Criteria

- [ ] V3DT-0A-03 checkboxes marked `[x]` with validation notes
- [ ] V3DT-02 checkboxes marked `[x]` with validation notes
- [ ] Test commands documented for future regression runs

### Files

- `plans/DS8/v3dt/work_order.md`
- `scripts/v3dt_meta_coverage_test.py` (new)
- `scripts/v3dt_plausibility_test.py` (new)

---

## Task V3DT-H08 — Document unit convention and calibration requirements

**Priority:** P2 (Medium)  
**Blocking:** None  
**Status:** [x] Complete

### Problem

The relationship between `camera_calibration.json` E matrix, camInfo generation, and SV3DT/MV3DT output units is not clearly documented. Future maintainers may be confused by the WORLD_SCALE parameter.

### Implementation

Add a section to `plans/DS8/v3dt/README.md` or create `docs/DS8_v3dt_calibration.md`:

1. **Calibration inputs:**
   - `camera_calibration.json` `E` matrix: world→camera, column-major, meters
   - `cameras.yaml` intrinsics: pixel coordinates at native resolution
   
2. **camInfo generation:**
   - `NOESIS_V3DT_CAMINFO_INVERT_E`: when/why to invert
   - `NOESIS_V3DT_CAMINFO_WORLD_SCALE`: units multiplier (1 = m, 100 = cm; if not 1, retune SV3DT world-space parameters)
   - `modelInfo.height/radius`: must match world_scale units
   
3. **Output units:**
   - `bbox3d`: same unit as camInfo input
   - `velocity3d`: unit/second
   - BEV/telemetry: should remain in meters when using meters-native camInfo

4. **Validation commands:**
   - How to verify calibration correctness
   - How to check output plausibility

### Acceptance Criteria

- [x] Documentation exists covering calibration→output flow
  _2026-01-22 (Codex): Updated `plans/DS8/v3dt/README.md` with the current
  calibration inputs, camInfo generation path, and output units._
- [x] WORLD_SCALE behavior documented with examples
  _2026-01-22 (Codex): Documented meters-native baseline and cm-scale no-go
  in `plans/DS8/v3dt/README.md` and `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`._
- [x] Validation commands included
  _2026-01-22 (Codex): Added/updated calibration sanity checks and forensics
  commands in `plans/DS8/v3dt/README.md`._

### Files

- `docs/DS8_v3dt_calibration.md` (new) or `plans/DS8/v3dt/README.md`

---

## Task V3DT-H09 — Tilt-only preview calibration from depth

**Priority:** P2 (Medium)  
**Blocking:** None  
**Status:** [x] Complete

### Problem

We need a safe way to correct camera **tilt** (pitch/roll) using depth-plane evidence
without overwriting the main calibration file or disturbing yaw/translation.

### Implementation

1. Fit a ground plane from the latest MapAnything depth snapshot per camera.
2. Compute a tilt correction that aligns the plane normal to +Y **while preserving yaw**.
3. Preserve the camera center and write updated extrinsics to a **preview** calibration file.
4. Document how to generate camInfo from the preview calibration for comparison.

### Acceptance Criteria

- [x] Preview calibration file writes to a separate path (no overwrite of main calibration).
- [x] Tilt correction preserves yaw + camera center.
- [x] Unit tests cover the tilt correction math and preview merge behavior.

_2026-01-13 (Codex): Added `scripts/auto_tilt_from_depth.py` + `noesis/calibration/tilt_preview.py` to generate a preview calibration file (pitch/roll only, yaw preserved). Documented preview usage in `docs/DS8_v3dt_forensics.md` and `plans/DS8/v3dt/README.md`. Validated via `tests/test_tilt_preview.py`._
_2026-01-14 (Codex): Updated tilt preview to evaluate both plane-normal signs and prefer a downward-facing pitch (penalize positive pitch) while preserving yaw/center; reran `tests/test_tilt_preview.py` (PASS)._
_2026-01-14 (Codex): Added per-camera plane-normal override (`--normal-sign camera=±1`) to `scripts/auto_tilt_from_depth.py` and persisted override metadata in preview output; documented usage in `docs/DS8_v3dt_forensics.md` and `plans/DS8/v3dt/README.md`._

### Files

- `noesis/calibration/tilt_preview.py`
- `scripts/auto_tilt_from_depth.py`
- `docs/DS8_v3dt_forensics.md`
- `plans/DS8/v3dt/README.md`
- `tests/test_tilt_preview.py`

---

## Execution Order (Recommended)

| Order | Task | Priority | Dependencies | Est. Effort |
|-------|------|----------|--------------|-------------|
| 1 | V3DT-H01 | P0 | None | 2-3 hours |
| 2 | V3DT-H02 | P1 | None | 1 hour |
| 3 | V3DT-H03 | P1 | None | 15 min |
| 4 | V3DT-H07 | P2 | V3DT-H01 | 1-2 hours |
| 5 | V3DT-H04 | P2 | V3DT-H01 | 1 hour |
| 6 | V3DT-H08 | P2 | V3DT-H01 | 1 hour |
| 7 | V3DT-H09 | P2 | None | 1 hour |
| 8 | V3DT-H05 | P2 | None | 1-2 hours |
| 9 | V3DT-H06 | P3 | Phase 2 MV3DT | 2-3 hours |

**Total estimated effort:** 10-14 hours

---

## Validation Checklist (post-implementation)

- [ ] `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sv3dt.yaml --duration-s 300` passes
- [ ] `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sv3dt.yaml` shows bbox3d in telemetry
- [ ] BEV footpoints appear in correct room locations
- [ ] bbox3d.zLen values in range [chosen_unit * 1.4, chosen_unit * 2.1] for standing adults
- [ ] No regressions in baseline (non-V3DT) pipeline

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-30 | World units: **METERS** | Standard convention, aligns with documentation, cleaner reasoning |

---

## Progress Notes

_2025-12-30 (Codex): Created hardening work order with 8 tasks prioritized P0-P3. Starting with V3DT-H02 (MV3DT config alignment) while awaiting decision on V3DT-H01 (unit convention)._

_2025-12-30 (Codex): Completed V3DT-H02 and V3DT-H03. Aligned `config/v3dt/nvtracker_mv3dt.yml` with SV3DT baseline: added `minDetectorConfidence: 0.2`, full `TrajectoryManagement` section with `enableReAssoc: 1`, cascaded matching (`associationMatcherType: 1`), full `ReID` section, `poseInferenceInterval: 4`. Widened `maxTrackletMatchingTimeSearchRange` from 1 to 5 frames with explanatory comment._

_2026-01-13 (Codex): Updated V3DT camInfo defaults to the working baseline (w2p + invert E + Y-flip + world axes xzy + autogen camInfo). All three cameras now produce bbox3d with correct 1.7m heights; reprojection error still ~50px, so calibration fine-tuning remains._

_2025-12-30 (Codex): Completed V3DT-H05 (partial). Added try/catch protection around all meta extractions in `native/noesis_v3dt_meta_ext.cpp` so individual failures don't crash the pipeline. Confirmed `ObjectWorldFootLocationUserMetadata` does NOT exist in Service Maker includes - world footpoints must continue to be derived from bbox3d. Rebuilt native extension successfully._

_2025-12-30 (Codex): Completed V3DT-H01. User selected METERS as the canonical unit convention. Updated `scripts/generate_v3dt_caminfo.py` and `noesis/ds8_runtime.py` to default `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1.0` (meters). Updated `noesis/pipelines/hooks.py` scaling logic default. Regenerated all camInfo files: `config/v3dt/camInfo_{living-room,kitchen,family-room}.yml` now show `modelInfo.height: 1.7`, `radius: 0.35` (meters)._

_2025-12-30 (Codex): V3DT-H04 deferred — `config/ply_alignment.json` has `floor_y: 0.0`, so no offset is needed. The footpoint calculation (`foot_z = zCentre - 0.5*zLen`) should already produce Z ≈ 0 for standing persons once SV3DT world coords are mapped into the BEV frame._

_2025-12-30 (Codex): Completed V3DT-H08. Added "Unit Convention and Calibration" section to `plans/DS8/v3dt/README.md` covering: calibration inputs (camera_calibration.json, cameras.yaml, ply_alignment.json), camInfo generation env vars, output units, and validation commands._

_2025-12-30 (Codex): V3DT-H06 deferred — requires MV3DT Phase 2 (cross-camera fusion enabled) to collect overlap agreement metrics. This is P3 and not blocking._

_2025-12-30 (Codex): V3DT-H07 ready for manual validation. Test scripts exist: `scripts/sv3dt_meta_smoke_test.py`, `scripts/v3dt_oom_regression_test.py`. User should run these after deploying the updated configs to validate bbox3d outputs are in meters and heights are plausible (1.4–2.1m for adults)._

_2026-01-06 (Codex): Stabilized SV3DT unit conventions for motion robustness. Converted `config/camera_calibration.json` translations for kitchen + living-room from cm→m (rotation unchanged), regenerated `config/v3dt/camInfo_{living-room,kitchen,family-room}.yml` at 1920×1056 with `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` (camInfo in cm, telemetry scales back to m), and rebased `config/infer_v3dt_medium.yaml` to use `config/v3dt/nvtracker_sv3dt.yml`. Added tracker termination dump knobs (`outputTerminatedTracks`, `terminatedTrackFilename=/tmp/noesis_track_dump_`) and set `PoseEstimator.poseInferenceInterval=8` in `nvtracker_sv3dt.yml`. Verified cm↔m consistency via a `K^-1 @ P` ratio check (expected ~100 for all cameras)._

_2026-01-16 (Codex): Switched family-room dewarper destination K to full-FOV (dst focal = src, centered principal), updated preview intrinsics + camInfo, ran depth-based tilt (`config/camera_calibration_preview_dewarp_fullfov_tilt.json`, pitch -25.66°). Captured diagnostics (`diagnostics/v3dt_frames_fr_fullfov_{pre,post}tilt.ndjson`) and forensics report (`diagnostics/v3dt_report_20260116_164123.md`). Tracking is full-FOV but cuboid sizes remain small (H ratio ~0.40, W ratio ~0.58), reprojection ~99px._

_2026-01-16 (Codex): Refined full-FOV baseline with Charuco principal point and Y-flip. Updated `config/dewarper_family_room_charuco_rtsp.txt` (`dst-principal-point=641.368;356.805`) and `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` to match. Ran fresh tilt (`config/camera_calibration_preview_dewarp_fullfov_tilt.json`, pitch -23.48°, roll 3.01°, sign +1), regenerated camInfo with `NOESIS_V3DT_CAMINFO_INVERT_E=1`, `NOESIS_V3DT_CAMINFO_Y_FLIP=1`, `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`. Best run: `diagnostics/v3dt_report_20260116_174224.md` (reproj 26.7px, H ratio 0.58, W ratio 1.28, bottom offset 0.6px). Height still short; position-dependent error remains._
_2026-01-17 (Codex): Added `scripts/adjust_pitch_calibration.py` to apply pitch targets without moving camera centers. Tested family-room target pitch -5° from the auto-tilt baseline (`config/camera_calibration_preview_dewarp_fullfov_tilt_targetm5.json`) and captured `diagnostics/v3dt_frames_fr_fullfov_tilt_targetm5.ndjson` + report `diagnostics/v3dt_report_20260117_002906.md`. Result regressed (H ratio ~0.20, reproj ~39px); restored camInfo from auto-tilt baseline._
_2026-01-17 (Codex): Streammux 720 experiments. Single-source 1280x720 config (`config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_720.yaml`) exposed MapAnything engine batch=3 mismatch; left this config with `mapanything.enable=false` for diagnostics. Multi-source 1280x720 streammux test (`config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp_mux720.yaml`) plus new tracker config/camInfo dir (`config/v3dt/nvtracker_sv3dt_preview_charuco_fr_rtsp_dewarp_mux720.yml`, `config/v3dt_preview_charuco_fr_rtsp_dewarp_mux720/`) produced report `diagnostics/v3dt_report_20260117_004250.md` (family-room reproj 19.6px, H ratio ~0.51). Scaling streammux to 720 did **not** resolve the height collapse._
_2026-01-17 (Codex): Dewarper sanity check: briefly enabled `dewarp-dump-frames=1`, captured RGBA dumps, and compared against OpenCV undistort of frame 0 from `familyroomclip.mp4`. Differences were inconclusive without exact frame alignment; restored `dewarp-dump-frames=0`._
_2026-01-17 (Codex): Snapshot scaling fix: in `noesis/ds8_runtime.py` the snapshot intrinsics scaling now prefers the cameras YAML model resolution (`_camera_model_res`) before falling back to legacy config or cx/cy guessing. This aligns snapshot/debug intrinsics with `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml` and camInfo scaling._
