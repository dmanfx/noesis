# DS8 SV3DT + MV3DT Integration Plan (Noesis)

This plan is tailored to the **DS8 canonical stack** (`noesis/`). It assumes we will **not** touch deprecated pre-DS8 runtime code and we will **not** add CPU/appsink branches to DS8.

## Current baseline addendum (2026-01-22)

SV3DT is now working with a locked baseline (per-camera pitch preview + model height 2.2).
See `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md` for the exact
configs, why it works, and confirmed no-go items. This plan still applies for
MV3DT and the shared-world calibration work.

## 0) Executive summary (what we will build)

1. **Global calibration first**: produce a single, shared, metric **world frame** (meters, Y‑up) with correct per-camera extrinsics.
2. **SV3DT everywhere**: enable SV3DT for all cameras for occlusion-robust tracking + 3D state estimation.
3. **MV3DT only where cameras overlap**: start MV3DT only on **kitchen ↔ family-room** (vision neighbors), using local Mosquitto for MQTT.
4. **Stable IDs remain public**: keep `stable_id` as the only user-visible identity (per `plans/DS8/ds8_id_contract_v2.md`). MV3DT’s global tracker ID becomes an internal signal (`mv3dt_id`) used to strengthen StableID assignment and dedupe in overlaps.
5. **3D outputs in telemetry + BEV**: plumb SV3DT/MV3DT 3D bbox + velocity into DS8 tracking telemetry and BEV, while keeping backward compatibility.

Related docs:
- `plans/DS8/v3dt/work_order.md` (task breakdown + acceptance)
- `plans/DS8/v3dt/enhancements_plan.md` (post-bring-up feature ideas)

## 1) Current Noesis DS8 reality (constraints we must respect)

- DS8 pipeline builder: `noesis/pipelines/ds8_pipeline.py`
  - Tracker is `nvtracker` configured from `config/infer.yaml` (`tracker.config-file` → `ll-config-file`).
- Telemetry + BEV:
  - Primary tracking telemetry is built in `noesis/pipelines/hooks.py` (`_AnalyticsTelemetryProcessor`).
  - World coords are currently best-effort via bbox bottom-center ray-plane intersection (no SV3DT meta usage yet).
- Calibration inputs:
  - `config/cameras.yaml` (camera IDs, names, intrinsics models, height_m)
  - `intrinsics.json` (intrinsics models)
  - `config/camera_calibration.json` (extrinsics, column-major 4×4 `E`)
  - `config/ply_alignment.json` (floor_y, s_obj_to_m, align matrix)
- Confirmed stream/camera order (must stay consistent across camInfo + MV3DT pub/sub configs):
  - **Stream 0 / source index 0:** `living-room` (camera id `0`)
    - `config/infer.yaml` source 0: `rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?`
  - **Stream 1 / source index 1:** `kitchen` (camera id `1`)
    - `config/infer.yaml` source 1: `rtsp://192.168.3.214:7447/qt3VqVdZpgG1B4Vk?`
  - **Stream 2 / source index 2:** `family-room` (camera id `2`)
    - `config/infer.yaml` source 2: `rtsp://192.168.3.214:7447/4qWTBhW6b4nLeUFE?`
    - family-room is 1280×720 intrinsics in `config/cameras.yaml` but is scaled to 1920×1080 by streammux (per `config/config_nvdsanalytics_post.ini` comments)
- **Critical blocker:** `config/camera_calibration.json` currently looks **camera-local** (camera centers x≈0,z≈0 for all). MV3DT needs a shared global world frame.

## 2) Inputs we still need from you (blocking for Phase 0/1)

1. **Global camera poses**:
   - If Menon still has camera transforms in your house model: provide/export them (preferred).
   - Otherwise we must do a new calibration procedure (PnP/tag-based/manual correspondences).
2. **MQTT broker details** (for MV3DT MQTT communicator):
   - We can start with the existing local Mosquitto defaults already used in this repo (`config.py` → `IntegrationsSettings`).
   - Confirm whether you want MV3DT to use the same broker/auth as occupancy publishing.

Everything else (SV3DT/MV3DT config shape, meta extraction path, DS8 hook locations) is implementable in-repo.

## 3) Design decisions (site-specific)

### 3.1 Vision-neighbor graph (MV3DT)

- **Kitchen ↔ Family-room**: vision neighbors (true overlap).
- **Living-room**: not an MV3DT neighbor initially (adjacent/non-overlap).

Rationale:
- MV3DT tracklet matching assumes overlapping FoVs with time overlap; connecting non-overlap cameras increases false associations.

### 3.2 Identity strategy (stable_id vs mv3dt_id)

- **Public ID:** `stable_id` only (unchanged UI/WS contracts).
- **Internal IDs:**
  - `tracker_id` (raw DeepStream tracker ID; internal-only per contracts)
  - `mv3dt_id` (MV3DT global tracker ID if MV3DT is enabled and truly global; still internal-only)

How MV3DT helps StableID:
- If MV3DT claims two tracks across overlap belong to the same global ID, StableIDManager can treat that as a strong constraint:
  - fast, deterministic merges within overlap
  - fewer appearance-only mistakes under occlusion

### 3.3 Units / axes

- World units: **meters**
- Axes: **Y-up**
- BEV is metric-accurate (your requirement)

### 3.4 Target classes

- **Phase 1 scope:** people only (class_id==person).
- **Dogs (optional):** defer until Phase 1 is stable; SV3DT/MV3DT are documented around a human cylinder/cuboid model and optional pose anchors, so “dog support” is only safe once we confirm:
  - the tracker supports per-class `modelInfo` (some NVIDIA tooling treats `modelInfo` as a list), and
  - the detector+tracker class mapping is stable in your deployment.

## 4) Deliverables (files and behaviors)

### 4.1 New config artifacts (planned)

- SV3DT camera info files (one per camera):
  - `config/v3dt/camInfo_living-room.yml`
  - `config/v3dt/camInfo_kitchen.yml`
  - `config/v3dt/camInfo_family-room.yml`
- Tracker low-level configs:
  - `config/v3dt/nvtracker_sv3dt.yml` (SV3DT only)
  - `config/v3dt/nvtracker_mv3dt.yml` (SV3DT + MV3DT)
- MV3DT MQTT + neighbor graph:
  - `config/v3dt/mqtt_proto_adaptor.txt` (DeepStream MQTT adaptor config format)
  - `config/v3dt/pub_sub_info_config_0.yml` (vision-neighbor pub/sub config for tracker)
- DS8 pipeline configs (choose one approach):
  - **Option A (recommended):** DS8 pipeline config `config/infer_v3dt_sv3dt.yaml` that points `tracker.config-file` to `config/v3dt/nvtracker_mv3dt.yml`.
  - **Option B:** add a toggle in `config/infer.yaml` and let runtime select tracker config paths.
  - Note: `config/infer_v3dt.yaml` is currently a V3DT-safe config (SV3DT disabled) due to an upstream DeepStream 8.0 nvtracker RAM leak; see `plans/DS8/v3dt/oom_killed_infer_v3dt_debug.md`.

### 4.2 Telemetry changes (planned, additive)

Extend DS8 track payloads (people only) with **optional** fields:
- `bbox3d`: `{xCentre, yCentre, zCentre, xLen, yLen, zLen, xRot, yRot, zRot}`
- `velocity3d`: `[xVel, yVel, zVel]`
- `world`: `[x, y, z]` (or reuse existing `world` field but ensure it is metric and correct)
- `mv3dt_id`: `int` (internal-only unless explicitly requested later)

Update docs accordingly:
- `docs/DS8_api_contracts_ws.md`
- `docs/DS8_metadata_contracts.md`

## 5) Implementation phases (concrete)

### Phase 0a — Pre-calibration staging (do now; MV3DT disabled)

**Goal:** get *everything except shared-world fusion* ready so that when Menon extrinsics land we can flip on MV3DT quickly.

What we can do safely before global extrinsics:

1. **Prepare configs/files** (do not enable MV3DT yet):
   - Create the `config/v3dt/` folder structure and placeholder configs:
     - `config/v3dt/nvtracker_sv3dt.yml`
     - `config/v3dt/nvtracker_mv3dt.yml` (kept off until Phase 0)
     - `config/v3dt/mqtt_proto_adaptor.txt` (use existing Mosquitto settings from current repo defaults)
     - `config/v3dt/pub_sub_info_config_0.yml` (neighbor graph: kitchen ↔ family-room only)
   - Provision BodyPose3DNet assets/engine and wire `PoseEstimator` into the tracker config (pose helps SV3DT even in single-camera mode).
2. **Implement DS8 hook plumbing**:
   - Add code in `noesis/pipelines/hooks.py` to extract `NVDS_OBJ_3D_META` (`NvDsObj3DBbox`) *when it exists* and attach additive fields (`bbox3d`, `velocity3d`, and a best-effort `world`).
     - Metadata path: `pyds` traversal of `obj_user_meta_list`.
     - DS8 path: Service Maker `ObjectMetadata` does not expose `obj_user_meta_list`, so use the native bridge module `noesis_v3dt_meta_ext` (build: `scripts/build_noesis_v3dt_meta_ext.sh`).
   - Keep existing bbox-ray `world` fallback for when SV3DT is off or meta isn’t present.
3. **Add explicit “frame-of-reference” guardrails**:
   - Until Phase 0 is complete, treat any SV3DT-derived “world” coordinates as **camera-local** and do not use them for cross-camera comparisons or dedupe.
   - MV3DT remains disabled until we have a shared global frame.

Acceptance criteria:

- DS8 pipeline still runs in baseline 2D mode unchanged.
- With SV3DT disabled, telemetry/BEV remain unchanged (no regressions).
- With SV3DT enabled but using camera-local calibration, the system:
  - runs without crashing,
  - emits 3D bbox meta for people (if the projection matrices are valid enough),
  - does **not** claim cross-camera consistency.

### Phase 0 — Global calibration (required for MV3DT; strongly recommended for SV3DT→BEV)

**Goal:** produce shared-world extrinsics in `config/camera_calibration.json` such that camera centers are meaningfully separated in X/Z (meters).

Tasks:
1. Define the **canonical world frame**:
   - origin choice (e.g., a corner of kitchen floor)
   - axes (Y-up) and unit scale (meters)
2. Produce `E_world_to_camera` per camera:
   - preferred: export camera transforms from Menon’s house model
   - fallback: manual PnP (2D↔3D correspondences) using your 3D model landmarks
3. Validate in Python:
   - computed camera centers should have distinct X/Z
   - a few known world points project plausibly into each camera frame

Acceptance criteria:
- Camera centers (world) differ across cameras (not all x≈0,z≈0).
- BEV rendering for each camera places footpoints in consistent house locations.

### Phase 1 — SV3DT bring-up (single-camera 3D tracking)

**Goal:** enable SV3DT and verify we can extract 3D bbox meta in DS8 hooks.

Config work:
- Create per-camera `camInfo_*.yml` with:
  - `projectionMatrix_3x4` (preferred; DeepStream shifts by `(w/2,h/2)` internally), or `projectionMatrix_3x4_w2p` (explicit world→pixel)
  - `modelInfo` in meters (height/radius)
- Ensure the projection matrix matches the **actual pixel coordinate system the pipeline sees**:
  - use intrinsics scaled to `nvstreammux.width/height` (your `config/infer.yaml` is 1920×1080)
    - living-room + kitchen intrinsics are already 1920×1080 in `config/cameras.yaml`
    - family-room intrinsics are 1280×720 in `config/cameras.yaml` and must be scaled by 1.5 for 1920×1080 streammux
  - if streammux ever introduces letterbox padding (aspect mismatch), incorporate the pixel offsets into `K` before building `P`
- Create `config/v3dt/nvtracker_sv3dt.yml`:
  - `StateEstimator.stateEstimatorType: 3` (SIMPLE_LOCATION_KF / SIMPLE_LOC)
  - `ObjectModelProjection.cameraModelFilepath: [...]` in **source order**
  - `outputFootLocation: 1` (required for `NVDS_OBJ_3D_META`)
  - `outputVisibility: 1` (optional, still useful even if not readable via pyds)
  - `outputConvexHull: 0` initially (enable later if we decide to patch pyds)

Pose enablement:
- Add `PoseEstimator` section (BodyPose3DNet) to SV3DT tracker config.
- Build / provide TensorRT engine for BodyPose3DNet.

DS8 pipeline work:
- Add a dedicated DS8 pipeline config `config/infer_v3dt_sv3dt.yaml` with:
  - `tracker.config-file: config/v3dt/nvtracker_sv3dt.yml`
  - ensure `nvstreammux.sync-inputs: 1` is available for later MV3DT (can be enabled here too)

Hook work (core):
- In `noesis/pipelines/hooks.py`, when iterating object meta:
  - extract `NVDS_OBJ_3D_META` user meta as `NvDsObj3DBbox` (DS8 uses `noesis_v3dt_meta_ext`)
  - attach `bbox3d/velocity3d/world` to the internal track dict

Acceptance criteria:
- Pipeline runs with SV3DT enabled.
- For person objects, `NVDS_OBJ_3D_META` is present and yields plausible (metric) values.
- BEV uses SV3DT-derived world coords when present (fallback to existing method otherwise).

### Phase 2 — MV3DT bring-up (kitchen ↔ family-room only)

**Goal:** enable MV3DT for overlap pair and verify global ID propagation + fusion.

Config work:
- Create `config/v3dt/pub_sub_info_config_0.yml`:
  - `pubBrokerTopicStr` for each camera stream
  - `subPeerBrokerTopicStrs` such that:
    - kitchen subscribes to family-room
    - family-room subscribes to kitchen
    - living-room subscribes to nobody (initially)
- Create `config/v3dt/mqtt_proto_adaptor.txt` for the tracker MQTT communicator.
- Create `config/v3dt/nvtracker_mv3dt.yml`:
  - includes SV3DT sections (Phase 1)
  - adds `MultiViewAssociator` and `Communicator` sections
  - start by copying the relevant sections (and only then adapting paths/values) from:
    - `/tmp/deepstream_reference_apps/deepstream-tracker-3d-multi-view/config_templates/config_tracker.yml`
  - start with NVIDIA reference values, including the feature toggles that correspond to MV3DT doc sections:
    - `enableLatePeerReAssoc` (missed early adoption mitigation)
    - `enableIDCorrection` (undo incorrect re-associations)
    - `enableSeeThrough` (peer-assisted initiation under occlusion)
    - `enableMsgSync` (time-aligned peer usage; must be validated with your timestamps)
  - tune only after you have “correct global calibration + sync” verified

Runtime/pipeline work:
- Enable `nvstreammux.sync-inputs=1` (you already approved this).
- Confirm timestamps:
  - consider `nvstreammux.attach-sys-ts=1` if RTSP NTP timestamps are unreliable

Hook work:
- Record MV3DT global tracker ID (if MV3DT makes `object_id` global; validate):
  - store it as `mv3dt_id` (internal-only)
  - do **not** expose `track_id` to clients

Acceptance criteria:
- Same person in kitchen and family-room overlap gets consistent `mv3dt_id`.
- Under partial occlusion in one camera, the other camera can maintain ID via MV3DT (quasi-active / see-through as applicable).
- 3D positions from both cameras agree in the shared world frame (within a small tolerance).

### Phase 3 — Identity fusion: StableIDManager + MV3DT

**Goal:** make the system maximally robust and flexible by combining MV3DT geometry with appearance-based Stable IDs.

Implementation:
- Extend StableIDManager to accept an optional `mv3dt_id` hint:
  - maintain mapping `mv3dt_id → stable_id` while that person is active
  - if `mv3dt_id` repeats in another camera, reuse the same `stable_id` immediately
- Add guards:
  - only apply MV3DT→StableID mapping for class `person`
  - require recent observation window to avoid stale mappings

Acceptance criteria:
- Stable IDs remain consistent across kitchen↔family overlap even if embeddings are degraded by occlusion.
- No regression for living-room (SV3DT-only) operation.

### Phase 4 — Adjacent boundary handoff (living-room ↔ kitchen)

**Goal:** improve “non-overlap” transitions without abusing MV3DT.

Approach:
- Keep MV3DT neighbor graph unchanged (no living-room edge).
- Use StableIDManager’s cross-camera matching; optionally add **3D gating**:
  - only accept cross-camera StableID match if last known world position is near the boundary and within a time window

Acceptance criteria:
- Fewer false merges across living-room/kitchen.
- Handoff works when a person exits one FoV and appears in the other shortly after.

## 6) Validation checklist (DS8-focused)

Use `docs/DS8_testing_guide.md` as the baseline. Add these targeted validations.

### 6.1 Recommended scenarios

| Scenario | Description | Key checks |
|---|---|---|
| Single-camera walk | A person walks across **kitchen** only | SV3DT runs, `NVDS_OBJ_3D_META` present, 3D footpoints stay on/near floor |
| Overlap traverse | A person moves through **kitchen ↔ family-room** overlap | MV3DT assigns consistent `mv3dt_id` in both views; StableID remains stable |
| Adjacent handoff | A person exits **kitchen** and appears in **living-room** shortly after (no overlap) | MV3DT is not used; StableID cross-camera match works (with optional 3D gating) |
| Multi-person overlap | 2 people in kitchen↔family-room overlap | Distinct `mv3dt_id` maintained; no false merges |
| MQTT failure | Stop Mosquitto mid-stream | Pipeline continues (SV3DT-only); clear warning logs; reconnection works |

### 6.2 Quantitative targets (initial)

| Metric | Target |
|---|---|
| SV3DT meta presence | `NVDS_OBJ_3D_META` attached to >95% of person tracks after warm-up |
| Floor contact sanity | foot `y` ≈ 0 within ±0.30m (after applying your `floor_y` alignment if any) |
| Indoor speed sanity | horizontal speed < 5 m/s (expected for people indoors) |
| MV3DT overlap ID consistency | >90% of overlap frames share the same `mv3dt_id` for the same person |
| CamInfo reprojection sanity | <10 px error on a few hand-picked 3D→2D correspondences per camera (after streammux scaling) |
| MQTT reconnection | recovers within 30 seconds after broker restart |

### 6.3 Contract checks

- **ID contract:** WS payloads show `stable_id` only; no raw tracker IDs.
- **Frame of reference:** before Phase 0, do not treat SV3DT “world” coords as shared across cameras (must be labeled/guarded as camera-local).

## 7) Risk register (with mitigations)

- **Global calibration missing/incorrect** → MV3DT will fail (IDs won’t propagate correctly; fusion will be wrong).
  - Mitigation: Phase 0 required; validate with reprojection + camera center separation.
- **Family-room camera height** (currently 1.5m) violates SV3DT assumption (“mounted higher than human height”).
  - Mitigation: enable pose estimator; tune `minPeerVisibility4Fusion`; rely on fusion weighting.
- **pyds lacks meta bindings for foot/visibility/convex hull**.
  - Mitigation: rely on `NvDsObj3DBbox` only; patch pyds later only if needed.
- **Performance overhead** (pose + MV3DT) on a single GPU.
  - Mitigation: run SV3DT first; enable pose at `poseInferenceInterval=-1` (initial height only); tune publish rate via MV3DT params.

## 8) Checklist/doc updates to do during implementation

When implementing, update these (per `plans/AGENTS.md`):
- `plans/DS8/ds8_migration_checklist_ds8_pipeline.md` (tracker + streammux config changes)
- `plans/DS8/ds8_migration_checklist_hooks.md` (3D meta extraction, BEV path)
- `plans/DS8/ds8_migration_checklist_telemetry.md` (payload additions)
- `plans/DS8/ds8_migration_checklist_ds8_runtime.md` (calibration bundle / reloads if touched)
- `plans/DS8/ds8_design_decisions.md` (record identity strategy + calibration conventions)
