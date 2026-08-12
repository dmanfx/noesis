# DS8 V3DT Work Order (SV3DT + MV3DT)

This is the DS8 implementation checklist for integrating **SV3DT** and **MV3DT** into the canonical Noesis stack (`noesis/`), aligned to `plans/DS8/v3dt/integration_plan.md`.

## Ground truth inputs (already known in this repo)

- Stream/camera order (must match camInfo list order + MV3DT pub/sub config order):
  - source 0 → `living-room` (camera id `0`)
  - source 1 → `kitchen` (camera id `1`)
  - source 2 → `family-room` (camera id `2`)
- Streammux output resolution: 1920×1080 (`config/infer.yaml`)
  - family-room intrinsics are defined at 1280×720 in `config/cameras.yaml` and need scaling to 1920×1080 when building camInfo.
- MV3DT reference config: use NVIDIA's installed DeepStream 3D multi-view
  tracker sample for the selected SDK; do not copy a machine-local scratch path
  into runtime configuration.

## Current SV3DT baseline (2026-01-22)

See `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for the locked
baseline, shortfalls, and confirmed no-go items. Key files:

- Pipeline: `config/infer_v3dt_baseline.yaml`
- Tracker: `config/v3dt/nvtracker_v3dt_baseline.yml`
- CamInfo dir: `config/v3dt/caminfo_baseline/`
- Active calibration: `config/camera_calibration.json`

The dated January status summary is a historical tuning snapshot. Current
shared-world truth and remaining live gates are recorded below.

**2026-08-12 execution decision:** AMC and MV3DT activation are deferred until
the Kitchen geometry is corrected. Living Room and Family Room have no overlap
and therefore no MV3DT edge. Kitchen/Family Room remains the only prospective
edge, but its config must stay disabled until corrected geometry plus
synchronized occupied overlap evidence passes the Phase 2 gates.

## Phase 0a — Pre-calibration staging (do now; MV3DT stays disabled)

### Task V3DT-0A-01 — Create config skeleton (`config/v3dt/`)

**Description**
- Add:
  - `config/v3dt/nvtracker_sv3dt.yml`
  - `config/v3dt/nvtracker_mv3dt.yml` (kept off until Phase 2)
  - `config/v3dt/mqtt_proto_adaptor.txt` (DeepStream MQTT adaptor format)
  - `config/v3dt/pub_sub_info_config_0.yml` (kitchen ↔ family-room only)
  - `config/infer_v3dt.yaml` (SV3DT enabled; patched nvtracker auto-loaded by `noesis/ds8_runtime.py`)
  - `config/infer_v3dt_sv3dt.yaml` (SV3DT enabled; live RTSP; patched nvtracker auto-loaded by `noesis/ds8_runtime.py`)

**Acceptance**
- [x] DS8 pipeline still runs unchanged in baseline mode.
  _2025-12-29 (Codex): Validated `config/infer.yaml` still runs and RSS stays bounded via `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer.yaml --duration-s 20`._
- [x] The new files exist and are referenced by paths that actually exist at runtime.
  _2025-12-27 (Codex): Added `config/v3dt/nvtracker_sv3dt.yml`, `config/v3dt/nvtracker_mv3dt.yml`, `config/v3dt/mqtt_proto_adaptor.txt`, `config/v3dt/pub_sub_info_config_0.yml`, and `config/infer_v3dt.yaml` (pipeline run validation pending)._
  _2025-12-29 (Codex): Split configs: `config/infer_v3dt.yaml` now uses baseline 2D tracker to avoid SV3DT RAM leak; SV3DT moved to `config/infer_v3dt_sv3dt.yaml` (see `plans/DS8/v3dt/oom_killed_infer_v3dt_debug.md`)._
  _2025-12-29 (Codex): Fixed the upstream SV3DT host-RAM leak by patching DeepStream `gst-nvtracker` and re-enabled SV3DT in `config/infer_v3dt.yaml`; validated via `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt.yaml --duration-s 300`._

### Task V3DT-0A-02 — Provision BodyPose3DNet assets/engine

**Description**
- Use NVIDIA’s PoseEstimator keys as in the reference tracker config (`config_tracker.yml`) and point:
  - `PoseEstimator.onnxFile` to a BodyPose3DNet ONNX
  - `PoseEstimator.modelEngineFile` to a built TensorRT engine
- Start with `poseInferenceInterval: -1` (initialize height; reduce steady-state cost) and tune later.

**Acceptance**
- [x] Tracker config references valid files (no missing-path errors).
- [x] Pipeline runs with pose enabled (even if SV3DT/MV3DT are still off).
  _2025-12-27 (Codex): Provisioned BodyPose3DNet ONNX (`models/bodypose3dnet/bodypose3dnet_accuracy.onnx`) + engine (`models/engines/bodypose3dnet_accuracy_b1_fp16.engine`) and validated tracker loads the engine via `python3 scripts/sv3dt_meta_smoke_test.py` (offline `config/infer_v3dt_sample.yaml`)._

### Task V3DT-0A-03 — Hook plumbing: extract `NVDS_OBJ_3D_META`

**Description**
- In `noesis/pipelines/hooks.py`:
  - When SV3DT is enabled, extract `NVDS_OBJ_3D_META` (`NvDsObj3DBbox`) from per-object user meta:
    - Metadata path: `pyds` traversal of `obj_user_meta_list`.
    - DS8: Service Maker Python `ObjectMetadata` does not expose `obj_user_meta_list`, so use the native bridge module `noesis_v3dt_meta_ext` (built via `scripts/build_noesis_v3dt_meta_ext.sh`).
  - Attach additive fields to the per-track payload:
    - `bbox3d` (use `NvDsObj3DBbox` field names)
    - `velocity3d`
    - canonical `world` derived from the V3DT bbox ground endpoint and the
      configured tracker-to-world axis map
- Guardrails:
  - Keep raw `bbox3d`/`velocity3d` in the tracker tuple for diagnostics.
  - Publish `world` only after conversion to Y-up `backend_world_m`; missing or
    malformed axis/bbox data is invalid, not a ray-plane fallback.
  - Do not claim MV3DT dedupe until occupied overlap evidence exists.

**Acceptance**
- [ ] With SV3DT enabled, >95% of person tracks have `NVDS_OBJ_3D_META`.
- [x] With SV3DT disabled, telemetry/BEV are unchanged (no regressions).
  _2026-01-27 (Codex): BEV renderer now stays in calibration world frame (removed camera-local yaw/translation rotation) so legacy tracker coordinates match BEV; validated via synthetic homography check with yaw/pitch calibration (pipeline/UI run pending)._
  _2025-12-27 (Codex): Wired DS8 `NVDS_OBJ_3D_META` extraction via `noesis_v3dt_meta_ext` (Service Maker C++ meta iterators) and plumbed `bbox3d`/`velocity3d`/`world` into DS8 telemetry; validated `bbox3d` appears in tracking WS telemetry via `python3 scripts/sv3dt_meta_smoke_test.py` (offline `config/infer_v3dt_sample.yaml`)._
  _2025-12-28 (Codex): Extended V3DT meta plumbing to also extract `NVDS_OBJ_WORLD_FOOT_LOCATION` (`world_foot`) + `NVDS_OBJ_VISIBILITY` and prefer `world_foot` for the BEV/world footpoint; added DS8 runtime preflight to regenerate `config/v3dt/camInfo_*.yml` from current calibration when a V3DT tracker config is selected (prevents stale camInfo drift after auto-calibrate)._
  _2025-12-29 (Codex): Fixed a reproducible Service Maker SIGSEGV by disabling `NVDS_OBJ_WORLD_FOOT_LOCATION` extraction inside `native/noesis_v3dt_meta_ext.cpp`; `bbox3d` telemetry validated again via `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sample.yaml`._
  _2026-07-11 (Codex): Restored the locked `xzy` tracker tuple to canonical Y-up `backend_world_m` in DS8, the protected reimplementation, and DS9. The V3DT lane now fails closed when axis/bbox metadata is absent or malformed, preserves the native tracker image-foot separately from the opposite cuboid endpoint, and rejects a pre-seeded world point in the wrong frame. Rebuilt/import-smoked both SDK-major native bridges; focused root tests passed (41), focused DS9 V3DT/ownership tests passed (153), and exact V3DT artifact provenance passed. Fresh occupied runtime evidence remains pending._

### Task V3DT-0A-04 — Fix SV3DT host-RAM leak (DeepStream `nvtracker`)

**Description**
- Patch DeepStream’s open-source `gst-nvtracker` plugin to fix an upstream SV3DT host-RAM leak and ensure DS8 runs load the patched plugin automatically for V3DT configs.

**Acceptance**
- [x] SV3DT runs without runaway RSS / OOM-kill for ≥300s in a headless regression run.
  _2025-12-29 (Codex): Added `patches/deepstream/gst-nvtracker_mask_params_leak_fix.patch` + `scripts/build_patched_nvtracker.sh`, and integrated auto-build/auto-load in `noesis/ds8_runtime.py`; validated via `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt.yaml --duration-s 300`._

## Phase 0 — Global calibration (MV3DT blocker)

### Task V3DT-00 — Maintain shared-world extrinsics and prove their live use

**Description**
- Export camera transforms from Menon (preferred) into a **single, shared** world frame:
  - meters, Y-up
  - consistent origin across cameras
- Write to `config/camera_calibration.json` as **world→camera** 4×4 `E`, stored column-major.

**Acceptance**
- [x] Camera centers computed from `E` have meaningful X/Z separation (not all x≈0,z≈0).
- [x] DS8, protected-reimplementation, and DS9 camInfo files match the active
  calibration and locked `w2p` / `xzy` projection contract.
- [ ] A fresh occupied same-session v2 gate covers all three cameras, advances
  per-camera tracker continuity, and keeps native image-foot reprojection at or
  below 40 px p95.
- [ ] MV3DT separately proves synchronized kitchen/family-room overlap, peer
  association, and fused output; shared calibration alone cannot satisfy this.

  _2026-07-11 (Codex): Verified separated camera centers in the active shared
  calibration, byte-identical DS8/DS9 camInfo sets, and exact `xzy` projection
  replay. Protected offline tracks across all three cameras support the 40 px
  native image-foot p95 threshold. The privacy-safe v2 same-session live gate is
  implemented and ownership-enforced, but no fresh live session was launched or
  promoted in this pass._

## Phase 1 — SV3DT bring-up (single-camera 3D tracking)

### Task V3DT-01 — Generate per-camera `camInfo_*.yml`

**Description**
- Build a 3×4 projection matrix for each camera (world→pixel):
  - The locked active profile uses `projectionMatrix_3x4_w2p`, whose matrix
    already includes the pixel principal point. Keep matrix type explicit;
    `projectionMatrix_3x4` has different centering semantics.
  - Use the **streammux output resolution** (e.g., 1920×1080) when scaling intrinsics, not `tracker-width/height`.
    - Rationale: `tracker-width/height` is an internal optimization; using 1920×1056 can introduce non-uniform scaling and distort SV3DT’s model projection (observed as “stretched line” cuboids and sporadic tracks).
  - Scale family-room intrinsics from 1280×720 → 1920×1080 (x1.5 in width, x1.5 in height).
  - Treat `E` as **world→camera** (Menon canonical) and compute `P = K @ E[:3, :]`.
    - Use `NOESIS_V3DT_CAMINFO_INVERT_E=1` only if your stored `E` is camera→world (`Twc`) and must be inverted to obtain world→camera.
  - Locked baseline (2026-01-22) uses:
    - `projectionMatrix_3x4_w2p`, `NOESIS_V3DT_CAMINFO_INVERT_E=0`,
      `NOESIS_V3DT_CAMINFO_Y_FLIP=1`, `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`,
      `NOESIS_V3DT_CAMINFO_WORLD_SCALE=1`.
    - See `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`.
- Write:
  - `config/v3dt/camInfo_living-room.yml`
  - `config/v3dt/camInfo_kitchen.yml`
  - `config/v3dt/camInfo_family-room.yml`

**Acceptance**
- [x] camInfo files parse and load when referenced by `ObjectModelProjection.cameraModelFilepath`.
- [x] SV3DT meta is produced (`NVDS_OBJ_3D_META` present).
  _2025-12-27 (Codex): Generated `config/v3dt/camInfo_{living-room,kitchen,family-room}.yml` using current calibration (camera-local); reload/validation pending once global extrinsics land._
  _2025-12-29 (Codex): Fixed SV3DT runaway RSS by inverting `E` during camInfo generation (`noesis/ds8_runtime.py` + `scripts/generate_v3dt_caminfo.py`); regenerated camInfo and validated via `python3 scripts/v3dt_oom_regression_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt_real_caminfo.yaml --duration-s 120` and `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_sample_3cam_sv3dt_real_caminfo.yaml`._
  _2026-01-06 (Codex): Fixed calibration translation unit mismatch (kitchen + living-room cm→m) and regenerated `config/v3dt/camInfo_{living-room,kitchen,family-room}.yml` for 1920×1056 with `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100` (camInfo in cm, telemetry scales to m)._
  _2026-01-12 (Codex): Updated camInfo generation to follow NVIDIA `deepstream-tracker-3d` sample: generate `projectionMatrix_3x4` at streammux resolution (1920×1080) to avoid non-uniform tracker resize artifacts; autogen now uses streammux dimensions._
  _2026-01-22 (Codex): Locked SV3DT baseline with per-camera pitch preview extrinsics (family-room -16, kitchen -21, living-room -15) and model height 2.2; camInfo generated under `config/v3dt/caminfo_baseline/` and validated via V3DT forensics (see `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`)._
  _2026-05-22 (Codex): Refined the isolated V3DT reimplementation path to publish Menon-facing `track.world` from the V3DT bbox3d-derived floor/contact point (`world_source=v3dt_bbox3d_foot`) without sending public cuboid geometry. Validated 2,925 MP4 samples across living-room, kitchen, and family-room with required payload fields present, no public raw V3DT fields, floor p95 0m, room-hit ratio 1.0, and image-base reprojection p95 below 1e-12px for every camera; exact non-V3DT baseline smoke still advertised `world_source=backend_world_fused` and a sample track used `pose_floor_only`._
  _2026-07-11 (Codex): Corrected the May replay interpretation: the raw tracker tuple is not itself public world. The producer now applies the locked `xzy` map before publishing Y-up `backend_world_m`, while native tracker image-foot and the derived opposite cuboid endpoint remain independently testable. The v2 replay rejects the old raw tuple and any camera-local frame label._

### Task V3DT-02 — Enable SV3DT in `nvtracker_sv3dt.yml`

**Description**
- Base on the existing tracker config, then:
  - `StateEstimator.stateEstimatorType: 3`
  - `ObjectModelProjection.cameraModelFilepath: [...]` (in source order 0,1,2)
  - `ObjectModelProjection.outputFootLocation: 1` (required for `NVDS_OBJ_3D_META`)
  - `ObjectModelProjection.outputVisibility: 1` (optional)
- Keep MV3DT sections off.

**Acceptance**
- [x] Pipeline runs with SV3DT enabled.
- [x] 3D positions are plausible in meters (adult heights ~1.4–2.1m; indoor speeds <5 m/s).
  _2025-12-27 (Codex): Created `config/v3dt/nvtracker_sv3dt.yml` (SV3DT + PoseEstimator) and wired it via `config/infer_v3dt_sv3dt.yaml`; validation pending._
  _2026-01-06 (Codex): Rebased `config/infer_v3dt_medium.yaml` to `config/v3dt/nvtracker_sv3dt.yml` and validated `python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_medium.yaml` passes (bbox3d observed)._
  _2026-05-22 (Codex): Added an isolated protected V3DT reimplementation lane (`noesis/ds8_runtime_v3dt_reimpl.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`, `config/infer_v3dt_reimpl_fast1056_mp4.yaml`) using copied calibration artifacts under `config/v3dt/reimpl/`. Validated `reimpl_fast1056_fullout_yolo26s` with RTSP mosaic enabled: bbox3d smoke passed, 40.4 fps per stream in diagnostics, median image-base error 35.5px living-room / 91.5px kitchen / 68.6px family-room, and bbox3d height median 1.85m._
  _2026-05-24 (Codex): Tuned the isolated V3DT MP4 reimpl lane for the new one-person calibration clips: enabled ReID in `config/infer_v3dt_reimpl_fast1056_mp4.yaml`, added V3DT-only family-room 2688x1512 dewarper calibration, added a V3DT-only analytics exclude file, and added V3DT-only public StableID cap/single-person canonicalization in `hooks_v3dt_reimpl.py`. Validated a 240s WebSocket sample across kitchen, family-room, and living-room with 2,831 tracking messages, `unique_sid_global=[1]`, `duplicate_public_count=0`, `over_cap_count=0`, and no single-track SID switches; final RTSP mosaic confirmed family-room was no longer top-left zoomed._

## Phase 2 — MV3DT bring-up (kitchen ↔ family-room only)

### Task V3DT-03 — Create MV3DT pub/sub graph + MQTT adaptor config

**Description**
- Create `config/v3dt/pub_sub_info_config_0.yml`:
  - kitchen subscribes to family-room
  - family-room subscribes to kitchen
  - living-room subscribes to none
- Create `config/v3dt/mqtt_proto_adaptor.txt` for local Mosquitto.

**Acceptance**
- [ ] With broker down, an explicitly selected MV3DT lane fails loudly rather
  than silently degrading to SV3DT-only.
- [ ] With broker up, tracker connects and publishes/subscribes without repeated reconnect loops.

### Task V3DT-04 — Enable MV3DT in `nvtracker_mv3dt.yml`

**Description**
- Start from the NVIDIA reference sections in the installed DeepStream
  multi-view 3D tracker sample for the selected SDK.
- Enable:
  - `MultiViewAssociator` (including advanced toggles)
  - `Communicator` (`communicatorType: 2`)
- Enable `nvstreammux.sync-inputs=1` when MV3DT is enabled.

**Acceptance**
- [ ] In kitchen↔family overlap, `mv3dt_id` is consistent for the same person (>90% of overlap frames).
- [ ] No MV3DT edge exists for living-room (no false cross-camera merges via MV3DT).

## Phase 3 — Identity fusion (StableID + MV3DT)

### Task V3DT-05 — Use MV3DT IDs as an internal constraint (people-only)

**Description**
- Keep `stable_id` as the only user-visible identity.
- Use MV3DT global IDs (captured as `mv3dt_id`) to make StableID merges deterministic in overlap.

**Acceptance**
- [ ] Stable IDs remain stable in overlap under occlusion where embeddings degrade.
- [ ] No regression when MV3DT is disabled (SV3DT-only and baseline 2D still behave).

## Phase 4 — Adjacent (non-overlap) handoff: living-room ↔ kitchen

### Task V3DT-06 — Add 3D gating for cross-camera StableID matches

**Description**
- Do not add a living-room MV3DT neighbor edge.
- Use a time window + boundary proximity gate (in world coords) to reduce false merges.

**Acceptance**
- [ ] Fewer false merges across living-room/kitchen without harming real handoffs.
