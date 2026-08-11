# Work Order - Menon World Unification

## Phase 3 - Canonical global world and coherent scene consumption

- [x] Replace production per-camera person fusion with strict `noesis.world.snapshot` v1 state.
  _2026-07-10 (Codex): Added bounded immutable snapshot validation with exact producer/run/sequence/time/frame/identity/lifecycle rules; absent and lost entities clear immediately; validation=`node --test tests/canonical-world-state.test.mjs`; outcome=PASS._
- [x] Apply backend-world to Menon-scene coordinates exactly once at the renderer boundary.
  _2026-07-10 (Codex): Canonical presentation now requires the explicit scene-similarity matrix, emits scene-root entity visuals, rejects double transforms, and leaves per-camera range/clamp/smoothing code diagnostics-only; validation=`node --test tests/world-scene-production-boundary.test.mjs`; outcome=PASS._
- [x] Consume only the promoted coherent scene release in every production scene consumer.
  _2026-07-10 (Codex): Added strict current-payload validation and atomic SHA/size verification for authored OBJ, MTL/textures, and coherent camera artifacts; removed latest/per-revision selection from environment, surface, and reconstruction consumers; validation=`node --test tests/scene-release-service.test.mjs tests/authored-scene-dependencies.test.mjs`; outcome=PASS._
- [x] Validate the final real scene candidate through the exact Menon consumer.
  _2026-07-10 (Codex): `node scripts/validate-scene-release-consumer.mjs --release ../Noesis_Devel/data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json --virtual-twin-root ../Noesis_Devel/data/virtual_twin` verified 3 cameras, 27 camera artifacts, 1 MTL, 4 textures, 34 current-release requests, and zero virtual-twin routes; outcome=PASS._

## Phase 2 - Canonical backend-world refactor

- [x] Make backend world meters the single canonical producer frame and make Menon scene a derived consumer frame.
  _2026-03-31 (Codex): Switched PoseV1/bundle/runtime defaults to `backend_world_m`, moved baseline default calibration to `config/camera_calibration.json`, and updated tracking contract metadata; validation=`pytest -q tests/test_menon_pose_extrinsics.py tests/test_calibration_manager.py tests/test_floorplan_orientation.py`, `python3 scripts/menon_pose_calibration_smoke_test.py --cameras-config config/cameras.yaml --calibration-json config/camera_calibration.json --alignment-json config/ply_alignment.json`; outcome=PASS._
- [x] Remove canonical image-flip inference and yaw-only local BEV conversion from backend geometry paths.
  _2026-03-31 (Codex): Tracking and BEV world/local projection now use the corrected pose basis directly; local BEV uses full world->camera local ground conversion and floorplan payloads are explicitly metric `camera_local_ground_m`; validation=`pytest -q tests/test_floorplan_orientation.py`, `python3 -m compileall -q noesis geometry calibration_bundle.py`; outcome=PASS._
- [x] Make Menon derive camera and track scene-space state from backend world meters instead of treating backend payloads as already scene-native.
  _2026-03-31 (Codex): Added backend-world metadata propagation into Menon calibration payloads, scene derivation in coordinate transforms, backend-world track conversion before path visualization, and calibration-bundle reprojection as the default camera authority; validation=`node --check ../Menon/src/features/calibration/CoordinateTransform.js ../Menon/src/features/calibration/CalibrationManager.js ../Menon/src/features/path-visualization/ReprojectionCameraRegistry.js ../Menon/src/services/WebSocketClient.js`; outcome=PASS._

## Phase 0 - Project scaffold and governance

- [x] Create centralized project documents under `plans/menon_world_unification/`.
  _2026-02-10 (Codex): Created `README.md`, `work_order.md`, `contracts.md`, `decisions.md`, `validation_matrix.md`, `timeline.md`, `session_notes.md`, `handoff.md`; method=file creation; outcome=PASS._
- [x] Add local `AGENTS.md` with read order and evidence rules.
  _2026-02-10 (Codex): Added `plans/menon_world_unification/AGENTS.md`; method=doc review; outcome=PASS._
- [x] Cross-link DS8 master work order to this workspace path.
  _2026-02-10 (Codex): Updated `plans/DS8/ds8_master_work_orders.md` links to `plans/menon_world_unification/...`; method=doc diff; outcome=PASS._
- [x] Run docs consistency checks after doc edits.
  _2026-02-10 (Codex): Updated `scripts/check_agents_docs_consistency.py` to validate active docs only (exclude archive/history hard requirements) and reran `./scripts/check_agents_docs_consistency.py`; outcome=PASS._

## Phase 1 - Baseline integration

- [x] Menon emits pose payload (`position`, `yaw/pitch/roll`, `rotation_order`, `frame`) on camera save.
  _2026-02-10 (Codex): Updated `Menon/src/utils/DeviceManager.js` to send `set_extrinsics` with PoseV1 payload only; method=`node --check src/utils/DeviceManager.js`; outcome=PASS._
- [x] Menon settings export includes `camera_poses` so all camera poses can be shared in one artifact.
  _2026-02-10 (Codex): Updated `Menon/src/ui/components/SettingsPanel.js` and `Menon/server.js` export path to include `camera_poses`; method=code inspection + syntax checks; outcome=PASS._
- [x] Noesis calibration loader accepts pose entries in `config/camera_calibration.json`.
  _2026-02-10 (Codex): Extended `calibration_bundle.py` and `noesis/calibration/manager.py` pose parsing/storage; method=`pytest -q tests/test_calibration_manager.py`; outcome=PASS._
- [x] Noesis converts pose to `E` (world->camera, column-major) with deterministic math.
  _2026-02-10 (Codex): Added/validated `pose_to_E_col_major` path in `calibration_bundle.py`; method=`pytest -q tests/test_menon_pose_extrinsics.py`; outcome=PASS._
- [x] Noesis strict mode can reject startup when pose is missing or invalid for any configured camera.
  _2026-02-10 (Codex): Enabled strict default in `noesis/ds8_runtime.py` and added startup coverage checks; method=`python3 scripts/menon_pose_calibration_smoke_test.py --cameras-config config/cameras.yaml --calibration-json config/camera_calibration_menon_obj.json --alignment-json config/ply_alignment.json`; outcome=PASS._
- [x] Noesis baseline world outputs use unified `menon_scene` world frame.
  _2026-02-10 (Codex): Updated `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `noesis/ds8_runtime.py`, `config/infer.yaml`; method=`python3 -m py_compile ...`; outcome=PASS._

## Phase 1.5 - Consistency and validation

- [x] Align axis conventions between Menon and Noesis, including image-axis flip behavior for ray projection.
  _2026-02-10 (Codex): Preserved Y-up `menon_scene` contract and BEV flip inference while defaulting BEV to world mode; method=code inspection (`noesis/telemetry/bev.py`, `noesis/ds8_runtime.py`); outcome=PASS._
- [x] Validate floor plane orientation and reject invalid camera-floor geometry in strict mode.
  _2026-02-10 (Codex): Added floor-geometry checks in `_CalibrationProvider.validate_pose_coverage()` and smoke validator script; method=`python3 scripts/menon_pose_calibration_smoke_test.py ...`; outcome=PASS._
- [x] Disable meter conversion and unit fallback behavior in strict mode.
  _2026-02-10 (Codex): Removed strict-path meter coercion in `noesis/ds8_runtime.py` and forced scene-unit `unit_scale=1.0` in strict mode; method=`python3 -m py_compile ...`; outcome=PASS._
- [x] Ensure no fallback to legacy `E`/`Twc` paths in strict mode.
  _2026-02-10 (Codex): `set_extrinsics` now requires valid `pose` when strict mode is enabled and rejects legacy-only payloads; method=script contract update `scripts/calibration_rpc_smoke_test.py` + unit compile/tests; outcome=PASS._
- [x] Validate BEV and track outputs against Menon expectations.
  _2026-02-10 (Codex): Ran `python3 scripts/menon_bev_track_parity_smoke_test.py --ws ws://127.0.0.1:6060 --pipeline-config config/infer.yaml --cameras-config config/cameras.yaml --duration 20 --p95-threshold-m 0.50`; outcome=PASS (`p95_err_m=0.013097595290442646`, `comparisons=609`)._

## Phase 1.6 - OBJ-unit baseline lock (no fallback)

- [x] Create a dedicated baseline calibration artifact in Menon OBJ units.
  _2026-02-10 (Codex): Created `config/camera_calibration_menon_obj.json` from pose data and verified with `python3 scripts/menon_pose_calibration_smoke_test.py --cameras-config config/cameras.yaml --calibration-json config/camera_calibration_menon_obj.json --alignment-json config/ply_alignment.json`; outcome=PASS (3/3 cameras valid)._
- [x] Baseline DS8 runtime defaults to the OBJ-unit calibration artifact with no implicit fallback.
  _2026-02-10 (Codex): Updated `_CalibrationProvider` path selection in `noesis/ds8_runtime.py` so non-`v3dt` mode targets `config/camera_calibration_menon_obj.json` unless `NOESIS_CALIBRATION_EXTRINSICS` is set; method=`python3 -m py_compile noesis/ds8_runtime.py`; outcome=PASS._
- [x] Menon tracking path ingestion consumes backend world coordinates without unit rescaling.
  _2026-02-10 (Codex): Removed `objUnitsPerMeter` scaling from `Menon/src/services/WebSocketClient.js`; method=`node --check src/services/WebSocketClient.js`; outcome=PASS._

## Testing and rollout hardening

- [x] Add unit tests for pose parsing and pose->E synthesis.
  _2026-02-10 (Codex): Added `tests/test_menon_pose_extrinsics.py` and parser precedence tests in `tests/test_calibration_manager.py`; method=`pytest -q tests/test_menon_pose_extrinsics.py tests/test_calibration_manager.py`; outcome=PASS._
- [x] Add smoke script for pose calibration validity and floor checks.
  _2026-02-10 (Codex): Added `scripts/menon_pose_calibration_smoke_test.py`; method=`python3 -m py_compile scripts/menon_pose_calibration_smoke_test.py`; outcome=PASS._
- [x] Run focused DS8 runtime smoke checks in baseline mode with strict pose-only enabled.
  _2026-02-10 (Codex): Ran `python3 scripts/calibration_rpc_smoke_test.py --ws ws://127.0.0.1:6058 --pipeline-config config/infer.yaml --cameras-config config/cameras.yaml`; outcome=PASS._
- [x] Record verification evidence in `validation_matrix.md`.
  _2026-02-10 (Codex): Updated `plans/menon_world_unification/validation_matrix.md` with pass/pending status and verification commands/files._

## Acceptance gate checklist

- [x] Gate G0->G1: project scaffold and governance in place.
  _2026-02-10 (Codex): Phase-0 artifacts + local AGENTS/workflow are in place; outcome=PASS._
- [x] Gate G1->G2: pose-based ingestion and baseline world output complete.
  _2026-02-10 (Codex): Pose-authoritative Menon->Noesis path implemented with strict startup gating and world-frame defaults; outcome=PASS (code-level + unit evidence)._
- [x] Gate G2->G3: axis/floor/strict-mode validations complete.
  _2026-02-10 (Codex): Axis/floor checks and strict runtime behavior validated with live parity evidence; outcome=PASS (`p95_err_m=0.013097595290442646`)._
- [x] Gate G3->G4: tests and smoke evidence captured.
  _2026-02-10 (Codex): Captured strict pose smoke, calibration RPC smoke, and BEV/track parity smoke outputs in-session; outcome=PASS._
