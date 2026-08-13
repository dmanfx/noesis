# Reference Documentation

This folder is the live reference for the Noesis video stack. Historical plans
and worklogs live in `docs/history/`.

## Quick Links
- **DeepStream 9.1 direct upgrade:**
  `../DS9/docs/deepstream_9_1_direct_upgrade_plan.md`
- **Start here (DS8):** `DS8_README_FOR_AGENTS.md`
- **Current baselines & defaults:** `DS8_Baselines.md`
- **Pose-assisted StableID:** `DS8_pose_stable_id_integration.md`
- **APIs & contracts:** `DS8_api_contracts_ws.md`, `DS8_api_contracts_rest.md`,
  `DS8_metadata_contracts.md`, `Telemetry_Schema.md`
- **WebSocket overview:** `WebSocket_API.md`
- **Testing:** `DS8_testing_guide.md`
- **YOLO26-seg engine maintenance:** `DS8_yolo26_seg_engine_maintenance.md`
- **Runtime secrets:** `Runtime_Secrets.md`
- **Codebase overview:** `CODEBASE_DESCRIPTION.md`
- **V3DT forensics & calibration:** `DS8_v3dt_forensics.md`
- **Depth (DAv2 + MapAnything):** `depth_metadata.md`,
  `DEPTH_STACK_FLOW_V2.md`, `MapAnything_Depth.md`,
  `MapAnything_Heatmap_Viewer.md`
- **Virtual twin reconstruction:** `Virtual_Twin_Reconstruction.md`
- **Phone-walk fusion reconstruction:** `Phone_Walk_Fusion_Reconstruction.md`
- **Archived references:** see `history/` (archived)

## Navigation by Topic
- **Pipeline & flow:** `flow_diagram_high_level.md`,
  `flow_diagram_low_level.md`, `DS8_MIGRATION_KNOWLEDGE_BASE.md`
- **Config maps:** `Configuration_Map.md`, `CONVENTIONS.md`
- **Telemetry & integrations:** `WebSocket_API.md`, `Telemetry_Schema.md`,
  `Occupancy_Publishing.md`, `Integrations_Playbook.md`
- **ROI & analytics:** `Static_ROI_Exclusion.md`, `DS8_roi_editor.md`
- **Depth & geometry:** `DEPTH_STACK_FLOW_V2.md`, `depth_metadata.md`,
  `MapAnything_Heatmap_Viewer.md`, `Dynamic_Sensors.md`

## History Archive
- Full DS8 migration work orders, checklists, and iteration logs are mirrored
  under `docs/history/ds8/` (copy of `plans/DS8/`). Use those for provenance;
  the docs here are the canonical current guidance.
- Additional historical material also lives under `docs/history/`.

## Maintenance
- Keep live/reference guidance in `docs/`. Move dated logs or experiment notes to
  `docs/history/`.
- Add a “Deprecated / Stale” section to any doc that no longer matches the
  active stack.
- When you edit docs, append a one-line entry under “Doc changes” (date + what changed) for any **substantive** update: new features/processes/workflows, contract or baseline changes, behavior-affecting diagrams. Skip trivial typo/style fixes or purely cosmetic diagram tweaks.
- Status headers (“Status: validated as of …”) should only be updated when the doc has been re-validated against code/configs.

## Doc changes (2026-08-12)
- Made camera-bound PCF Scene Prior data the canonical source for every standard
  Depth drawer reconstruction and diagnostic view, documented automatic
  read-only loading, and retained manual Refresh as a non-displayed static
  comparison capture (`scene_prior_v1.md`, `DS8_api_contracts_ws.md`,
  `../DS9/README.md`).
- Corrected the DeepStream 9.1 completion record after removing per-frame
  identity retention and excess CPU math workers: the accepted live baseline is
  29.8-30 FPS per camera, while the remaining 6-11 ms WebSocket boundary p99 is
  optional optimization work (`../DS9/docs/deepstream_9_1_direct_upgrade_plan.md`).
- Made direct application validation the default for future agents and reserved
  appliance staging, candidate/selector ceremony, promotion, rollback, and
  broad unchanged validation for explicitly requested releases or unavoidable
  external lifecycle transitions (`AGENTS.md`, `DS9/AGENTS.md`,
  `docs/DS8_testing_guide.md`, `DS9/docs/deepstream_9_1_agent_skills.md`,
  `plans/noesis_menon_validation/validation_tiers.md`).
- Defined the exact DeepStream 9.1 direct-upgrade target and parity-first
  execution path, recorded the completed source pins and base-image inspection,
  retained 9.0 results as historical evidence, and explicitly deferred AMC
  plus MV3DT activation pending corrected Kitchen geometry
  (`../DS9/docs/deepstream_9_1_direct_upgrade_plan.md`).

## Doc changes (2026-08-10)
- Documented the validated DA3-conditioned MapAnything phone-walk fusion,
  static-camera authority boundary, retained evidence, and repeatable workflow
  for reconstructing additional rooms (`Phone_Walk_Fusion_Reconstruction.md`).

## Doc changes (2026-08-02)
- Defined Scene Prior standalone previews as calibration-derived,
  reference-camera-facing review artifacts while preserving the canonical
  backend-world grid (`scene_prior_v1.md`).

## Doc changes (2026-07-24)
- Corrected MapAnything ray-to-floorplan alignment to fit only observed
  depth-floor contacts while keeping the authored calibration floor distinct
  from the AGL-corrected depth floor (`MapAnything_Depth.md`).
- Joined canonical entity rooms to the existing per-object overcrowding
  membership labels with exact, single-label, fail-closed resolution shared by
  DS8/V3DT/DS9; camera-derived zones remain non-authoritative diagnostics
  (`Occupancy_Publishing.md`, `DS8_metadata_contracts.md`,
  `DS8_api_contracts_ws.md`).

## Doc changes (2026-07-19)
- Distinguished spatially authoritative nvdsanalytics ROI zones from
  camera-name occupancy fallbacks in tracking observations and canonical world
  source evidence; camera defaults can no longer become `room_id`
  (`DS8_api_contracts_ws.md`, `DS8_testing_guide.md`).

## Doc changes (2026-07-12)
- Versioned the synchronous canonical world journal as storage contract v2:
  exact legacy state is validated before one-time WAL/FULL migration, v2 mode
  drift and retained-history corruption fail closed, and the three-source
  cadence gate now covers 31 cycles with median and mean budgets while live
  soaks own tail latency (`DS8_testing_guide.md`,
  `plans/DS8/ds8_design_decisions.md`).
- Made the canonical DS8/V3DT lifecycle canary safe for immutable-checkpoint
  execution: a mandatory marked external private state cohort owns every
  mutable runtime, calibration, identity, cache, and evidence path; secret
  bytes never cross the launcher boundary; and sequential profiles may reuse
  identity state without overwriting create-once evidence
  (`DS8_testing_guide.md`, `plans/DS8/ds8_design_decisions.md`).
- Defined and adversarially tightened the shared DS8/protected-V3DT/DS9
  publication commit boundary: finite size/count-bounded JSON is encoded once
  and immutably admitted, but one ordered tracking/world/event batch remains
  behind a one-shot release gate until a synchronous exact-count journal
  acknowledgement and private world commit. Commit failure aborts before any
  client delivery and poisons the publisher; mutable fusion is no longer
  public; canonical types never latest-only coalesce; and paired BEV carries a
  typed exact tracking sequence/submission cohort. The queue also owns a 256 MiB
  frozen-byte cap with zero-byte shutdown proof, and authenticated telemetry
  fanout is hard-capped at 16 while health bypasses the set
  (`DS8_api_contracts_ws.md`,
  `DS8_testing_guide.md`, `Telemetry_Schema.md`,
  `DS9/README.md`, `DS9/docs/migration_state.md`,
  `DS9/docs/canonical_world.md`,
  `DS9/docs/validation_runbook.md`,
  `DS9/docs/bev_capture_event_integration.md`,
  `DS9/docs/runtime_ownership.yaml`,
  `plans/DS8/ds8_design_decisions.md`).
- Replaced load-only DS9 MapAnything admission with a correctness-first FP32
  transaction: a pinned batch-three real inference must seal finite/depth/mask/
  distribution/batch evidence before installation, and finalization plus direct
  realization reconciliation independently revalidate the receipt. The invalid
  FP16 plan remains non-qualifying and fresh multi-camera runtime parity is
  still open (`DS9/docs/MapAnything_Engine_Parity_Plan.md`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`, `DS9/docs/known_blockers.md`,
  `plans/DS8/ds8_design_decisions.md`).
- Superseded the unpromoted semantic-v2 draft with replayable semantic v3: an
  exact persisted identity anchor may join independently scheduled pose,
  registered depth, and backend-world frames only across 1.5 seconds of both
  capture and observation time. Its source now seals an ownership-bound
  acquisition interval, a fixed two-second pre-window capture-latency allowance,
  all-frame acquisition and canonical publication-clock bounds,
  window-contiguous but origin-unanchored live publication/lifecycle evidence,
  first-frame-only unanchored tombstones followed by exact in-window
  last-presence provenance, shared strict unique-key JSON, and marker-only
  privacy failures. The neutral gate now serves baseline and V3DT; baseline
  ReID promotion requires both identity and semantic evidence
  (`DS8_api_contracts_ws.md`,
  `DS8_metadata_contracts.md`, `DS8_testing_guide.md`,
  `DS9/docs/runtime_ownership_evidence.md`).
- Corrected the inline floorplan BEV contract: camera-local source/frame/time
  identity is explicit, parity validation uses calibrated transforms instead of
  cross-frame subtraction, and BEV health v2 distinguishes exact-frame
  activity, pre-first-success inactivity, authority state, and actual failures.
  Exact empty frames are active successes. The replayable DS9 floorplan gate is
  v4 while retaining N/N configured-camera floorplan correctness
  (`DS8_api_contracts_ws.md`, `Telemetry_Schema.md`, `DS8_testing_guide.md`,
  `DS9/docs/bev_capture_event_integration.md`).

## Doc changes (2026-07-11)
- Reconciled the shared BEV/floorplan contract with implementation: first
  authority absence is non-publishing `startup_pending`, post-ready loss is
  fatal, exact empty frames are active successes, registered depth is coherent,
  protected V3DT scales to calibration image size, and tracking-first BEV pairs
  use the maximum effective interval with forced lifecycle/count transitions
  (`DS8_api_contracts_ws.md`, `DS9/docs/bev_capture_event_integration.md`,
  `DS9/docs/validation_runbook.md`).
- Declared repository validation helpers as an explicit Python package so an
  unrelated installed `scripts` distribution cannot shadow authenticated DS9
  smoke clients in immutable checkouts (`DS9/README.md`,
  `DS9/docs/validation_runbook.md`).
- Replaced DS9 native mtime freshness guidance with pre-import content
  attestation of all six extension source bundles, ABI filenames, and manifest
  output hashes; documented the controlled rebuild evidence showing NVCC
  binary nondeterminism (`DS9/README.md`, `DS9/docs/validation_runbook.md`,
  `plans/DS8/ds8_design_decisions.md`).
- Documented the shared exact capture-event transaction across DS8, V3DT, and
  DS9: one owned MapAnything valve, cache-only zero-mutation reads, immutable
  snapshot/floorplan identity, depth-only RGB evidence, active-floorplan local
  BEV health, calibration-driven invalidation, and the replayable DS9
  floorplan live-gate v2 (`MapAnything_Depth.md`,
  `DS8_api_contracts_ws.md`, `Telemetry_Schema.md`,
  `DS9/docs/bev_capture_event_integration.md`).
- Defined `height_agl_meta.floor_offset_m` as the exact single bounded AGL
  correction shared by both floorplan generators and added the focused
  dual-path regression to the depth validation matrix (`MapAnything_Depth.md`,
  `DS8_testing_guide.md`).
- Made the DS8/V3DT/DS9 REST boundary metric wire-truthful: FastAPI now renders
  each measured response exactly once, metrics observe the returned bytes, and
  true rolling 10/60-second pooled plus max-path p99 gates prevent sparse slow
  routes from hiding in aggregate percentiles. Hard sample/detail/error caps
  fail closed on live saturation, compact stats retain bounded ranked offenders,
  and DS9 now has exact v1 household resident/identity-health parity with DS8.
  Coverage now includes all 41 framework-rendered product JSON routes; the five verified
  pre-rendered/file responses are named exemptions, and every other unmarked
  success fails closed.
- Restored the authoritative assembled WebSocket boundary budget and shutdown
  ownership contract: producer/executor/send dispatch all remain inside 3 ms,
  intentional coalescing dwell alone is excluded, blocking RPC providers use a
  drained owned executor, and REST/WS/depth/EOS/Map/storage teardown is ordered
  and fail-closed (`DS9/docs/migration_state.md`,
  `MapAnything_Depth.md`, `DS9/docs/MapAnything_Depth.md`,
  `DS9/docs/validation_runbook.md`,
  `plans/zero_copy_gpu/03_boundary_serialization_contract.md`,
  `plans/DS8/ds8_design_decisions.md`).
- Documented the DS9-only MapAnything ownership correction: one exact typed
  native UID/per-frame-layer selector replaces the quarantined generic fallback,
  the metadata-lifetime copy is exactly bounded/timed, owned CPU slices retain
  bounded async postprocessing with explicit runtime drain/join and final-job
  poison detection, and the failed 2026-07-11 depth/floorplan evidence remains
  blocked pending native rebuild and live rerun (`DS9/README.md`,
  `DS9/docs/MapAnything_Depth.md`,
  `DS9/docs/MapAnything_Engine_Parity_Plan.md`,
  `DS9/docs/validation_runbook.md`, `DS9/docs/migration_state.md`).
- Closed occupied identity truth/parity defects: accepted tracker subjects are
  immutable until an evidence-gap expiry without bypassing open-set rejection,
  `embedding_present` and persisted provenance are exact-frame v2 facts, and
  DS9 now enforces the same default household no-auto-merge policy as DS8
  (`DS8_api_contracts_ws.md`, `DS8_metadata_contracts.md`,
  `DS9/docs/migration_state.md`, `plans/household_identity/decisions.md`).
- Closed the DS9 typed ownership selector over its already validated runtime
  image ID, so asset/runtime promotion bindings and terminal CAS retain exact
  image authority rather than dropping it (`DS9/docs/runtime_ownership_evidence.md`,
  `plans/DS8/ds8_design_decisions.md`).
- Replaced the stale V3DT camera-local contract with the locked global-world
  boundary: raw bbox/velocity remain tracker diagnostics, producers apply the
  exact `xzy` map before publishing Y-up `backend_world_m`, native image-foot
  stays independent from image-base replay, and DS9 promotion requires a
  privacy-safe same-session v2 gate while MV3DT overlap/time-sync/fusion remains
  separately unproven (`DS8_Baselines.md`, `DS8_metadata_contracts.md`,
  `DS8_api_contracts_ws.md`, `DS9/README.md`,
  `DS9/docs/validation_runbook.md`,
  `plans/DS8/v3dt/integration_plan.md`).
- Refreshed the shared tracking/depth contract: persisted ReID provenance is an
  all-or-none sequence/model/dimension triad, `depth_present` requires usable
  finite depth, zero-person frames clear presence and continue as bounded
  heartbeats, DAv2 capture/fusion uses a bounded exact-frame rendezvous with
  public counters, native attachment failures are explicit, and occupied
  semantic acceptance remains pending live evidence
  (`DS8_api_contracts_ws.md`, `Telemetry_Schema.md`,
  `DS8_metadata_contracts.md`, `depth_metadata.md`,
  `DEPTH_STACK_FLOW_V2.md`, `DS8_Baselines.md`,
  `DS8_testing_guide.md`).

## Doc changes (2026-07-10)
- Documented the fail-closed analytics boundary: native pre-tracker exclusion
  is the sole DS8/V3DT/DS9 path, ROI mutations require exact hash/sequence
  receipts and fatal ambiguous-commit handling, YAML/INI limits are 4 MiB/1
  MiB, DS9 persists only the analytics pair with per-session evidence, REST
  shutdown retains an analytics quiescence lease, and the authenticated
  occupied-scene restore gate remains explicitly pending
  (`DS8_api_contracts_rest.md`, `DS8_testing_guide.md`,
  `Static_ROI_Exclusion.md`, `DS8_roi_editor.md`,
  `DS8_README_FOR_AGENTS.md`, `CODEBASE_DESCRIPTION.md`,
  `plans/DS8/ds8_design_decisions.md`,
  `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`,
  `plans/DS8/ds8_migration_checklist_hooks.md`).
- Documented the shared DS8/DS9 fail-closed inference boundary: production
  graph configs are atomically derived engine-only artifacts, runtime model and
  native-extension materializers never build, and source-rich configs remain
  explicit offline-maintenance inputs (`DS8_README_FOR_AGENTS.md`,
  `DS8_testing_guide.md`, `DS9/README.md`,
  `plans/DS8/ds8_design_decisions.md`,
  `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`).
- Recorded accepted 30-second baseline and V3DT engine-only lifecycle evidence,
  including deserialization, advancing tracking, source-key absence, orderly
  EOS/wait completion, clean exit, and residual-owner checks
  (`DS8_testing_guide.md`,
  `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`).
- Recorded the shared Swin `fc_pred/256` identity cutover and sequential
  V3DT-to-baseline continuity proof with no gallery/state reset
  (`DS8_testing_guide.md`, `plans/DS8/ds8_design_decisions.md`,
  `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`).
- Documented depth-registration v1's path-based MapAnything identity limitation,
  the narrow semantic compatibility rule for content-addressed runtime configs,
  and the future engine/config content-hash requirement
  (`MapAnything_Depth.md`, `DS8_testing_guide.md`,
  `plans/DS8/ds8_design_decisions.md`).
- Added the explicit guarded DS8 YOLO26-seg `n/s/m` engine maintenance workflow,
  including CPU-only planning, exact build provenance, prior-byte preservation,
  resource bounds, atomic installation, and rejection of false-positive
  `trtexec` load results (`DS8_yolo26_seg_engine_maintenance.md`,
  `DS8_testing_guide.md`).
- Documented the canonical authenticated 30-second DS8 baseline/V3DT lifecycle
  gates and their acknowledged downstream-EOS, fail-closed wait-thread
  quiescence contract (`DS8_testing_guide.md`,
  `plans/DS8/ds8_design_decisions.md`,
  `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`).
- Documented the identity-v2 authority scope boundary: semantic provenance now
  binds the loaded native/Python score transforms, while public promotion
  requires a separately pinned DS8/DS9 cutover artifact with exact coordinator-
  replay and occupied-scene reports (`DS8_api_contracts_rest.md`,
  `DS8_testing_guide.md`, `plans/household_identity/*`).
- Externalized active DS8/V3DT/DS9 camera locators and the MapAnything RPC key
  into strict owner-only appliance state; documented provisioning, rotation,
  public config references, fail-closed validation, and artifact/fingerprint
  privacy (`Runtime_Secrets.md`, `DS8_README_FOR_AGENTS.md`,
  `MapAnything_Depth.md`, `MapAnything_Heatmap_Viewer.md`,
  `DS8_testing_guide.md`).
- Corrected MQTT/Influx documentation to match the dormant runtime call graph,
  removed embedded credential guidance, and documented strict owner-only
  secret-file activation and rotation boundaries
  (`Integrations_Playbook.md`, `Occupancy_Publishing.md`,
  `integrations/occupancy_mqtt_influx.md`, `CODEBASE_DESCRIPTION.md`).
- Documented the complete immutable scene-release bundle, including authored
  OBJ material/texture dependencies, current-release URLs, and the validated
  three-camera candidate (`DS8_api_contracts_rest.md`,
  `Virtual_Twin_Reconstruction.md`).
- Corrected the DS9 MapAnything build documentation to distinguish the now-true
  FP16 builder intent from the still-open provenance and runtime-quality
  artifact gate (`DS9/docs/migration_state.md`,
  `DS9/docs/MapAnything_Engine_Parity_Plan.md`).

## Doc changes (2026-07-03)
- Updated the codebase overview and DS8 decision ledger for the Swin-Tiny ReID
  SGIE baseline, NvDCF tracker re-association profile, and long-term StableID
  gallery memory (`CODEBASE_DESCRIPTION.md`,
  `plans/DS8/ds8_design_decisions.md`).

## Doc changes (2026-06-30)
- Updated the DS8 detector baseline to make YOLO11 detect the default PGIE and
  document YOLO11 / YOLO11-seg `s|m|l` profile assets (`DS8_Baselines.md`).

## Doc changes (2026-06-24)
- Added the DS8 Wholebody49 X pose promoter implementation plan for promoting
  label-only Wholebody keypoint rows into the existing `NOESIS.POSE_FEATURES`
  object metadata contract (`DS8_wholebody49_pose_promoter_plan.md`).

## Doc changes (2026-05-27)
- Added the Noesis/Menon validation toolbox planning workspace and linked it
  from the DS8 testing guide for cross-space calibration, BEV, tracking-world,
  virtual-twin, and Menon-facing validation work (`plans/noesis_menon_validation/`,
  `DS8_testing_guide.md`).
- Added the first reusable validation package, fixture runner, fixture registry,
  minimal fixture, visual artifact index, camera reprojection overlay generation,
  and focused unit tests for common reports, transforms, anchors, camera
  sanity/reprojection, room bounds/wall crossing, tracking motion, BEV
  agreement, and Menon placement agreement (`noesis/validation/`,
  `scripts/noesis_validation_runner.py`,
  `plans/noesis_menon_validation/fixture_registry.json`).
- Added saved/live DS8 WebSocket telemetry validation that feeds tracking and
  BEV messages into the common report schema and writes a per-track audit
  artifact (`scripts/noesis_validation_telemetry_report.py`,
  `plans/noesis_menon_validation/minimal_telemetry.ndjson`).
- Added Menon placement-trace validation for Noesis-world to Menon-scene
  agreement, floor contact, trail agreement, timestamp alignment, and transform
  audit evidence (`scripts/noesis_validation_menon_trace_report.py`,
  `plans/noesis_menon_validation/minimal_menon_trace.json`).
- Added private Menon browser debug capture with fresh same-origin session proof
  and strict canonical-state/presentation/path/cohort admission into the shared
  Menon trace contract for live browser placement validation
  (`scripts/noesis_validation_capture_menon_trace.py`,
  `noesis/validation/menon_browser.py`,
  `plans/noesis_menon_validation/minimal_menon_browser_snapshot.json`).
- Expanded Menon validation with direct BEV-to-Menon trail agreement and avatar
  scale, collision, and movement-orientation checks
  (`noesis/validation/menon.py`,
  `plans/noesis_menon_validation/minimal_menon_trace.json`).
- Added scene coordinate-system and room geometry-constraint validators for
  generated scene fixtures, including units, axes, camera pose convention,
  handedness, origin/scale anchors, doorway plausibility, wall junctions, and
  surface continuity (`noesis/validation/scene.py`,
  `plans/noesis_menon_validation/minimal_fixture.json`).
- Added tracking temporal validators for occlusion bridges and doorway
  transitions, and extended saved telemetry/fixtures with occlusion uncertainty
  evidence (`noesis/validation/tracking.py`,
  `plans/noesis_menon_validation/minimal_telemetry.ndjson`).
- Added first scene-validation checks for plane geometry, room dimensions, mesh
  quality, depth anchors, and semantic object support to the common fixture
  runner (`noesis/validation/scene.py`,
  `plans/noesis_menon_validation/minimal_fixture.json`).
- Added first detection-to-world projection validators for footpoint agreement,
  floor contact, room bounds, person height, ray-floor validity, and projection
  confidence (`noesis/validation/tracking.py`).
- Added a registry-driven regression runner that executes the fixture,
  telemetry, and Menon-trace cases and writes a suite summary
  (`scripts/noesis_validation_regression_runner.py`).
- Expanded visual validation artifacts with camera room-outline/mesh-edge/raw
  footpoint overlays and BEV diagnostic overlays for room polygons, walls,
  doorways, frustums, tracks, and raw footpoints
  (`noesis/validation/visuals.py`).
- Expanded BEV diagnostic overlays with uncertainty ellipses plus StableID,
  projection-confidence, and ReID-confidence annotations
  (`noesis/validation/visuals.py`,
  `plans/noesis_menon_validation/minimal_fixture.json`).
- Added BEV zone-consistency and camera-coverage validators against room
  polygons and camera frustum footprints (`noesis/validation/bev.py`).
- Added golden/artifact regression comparison helpers with artifact size/hash
  checks and optional pixel-level image diffs
  (`noesis/validation/golden.py`,
  `scripts/noesis_validation_regression_runner.py`).
- Added registry-level expected thresholds so the regression runner reports
  expectation misses as regression failures, not just per-case script status
  (`plans/noesis_menon_validation/fixture_registry.json`).
- Expanded telemetry track audit artifacts with projection, temporal, and ReID
  confidence, doorway/occlusion history, impossible-motion events, speed
  summaries, and warning lists (`noesis/validation/telemetry.py`).
- Added validation asset inventory and tier guidance for selecting existing
  smoke tests, pytest coverage, fixture runners, DS8 telemetry checks, and
  Menon trace checks (`plans/noesis_menon_validation/validation_asset_inventory.md`,
  `plans/noesis_menon_validation/validation_tiers.md`).

## Doc changes (2026-02-04)
- Added ReID alias REST contracts, guardrails, and limitations (`DS8_api_contracts_rest.md`).
- Logged the soft-merge alias design decision and updated DS8 hooks/runtime checklists (`plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_hooks.md`, `plans/DS8/ds8_migration_checklist_ds8_runtime.md`).

## Doc changes (2026-03-06)
- Documented the canonical `NOESIS.OBJECT_DEPTH` DS8 object user-meta contract and logged the prototype depth-fusion metadata decision/progress (`DS8_metadata_contracts.md`, `plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_hooks.md`).

## Doc changes (2026-05-03)
- Added and updated the living-room virtual twin reconstruction note covering DS8 MapAnything evidence, ZeroPlane plane fusion, Menon model-surface GLB output, RGB atlas baking, front-most structural visibility assignment, and surface-aware texture-fit work (`Virtual_Twin_Reconstruction.md`, `DS8_api_contracts_rest.md`).

## Doc changes (2026-05-11)
- Updated BEV/floorplan telemetry docs and DS8 progress notes for registered-depth-only BEV display, shared clean floorplan layers across rooms, floorplan-surface snap/drop validity, and height-map inline BEV rendering (`DS8_api_contracts_ws.md`, `Telemetry_Schema.md`, `plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_telemetry.md`).

## Doc changes (2026-07-09)
- Clarified household identity is ON by default (`NOESIS_HOUSEHOLD_IDENTITY` unset/`1`; opt out with `=0`) in REST contracts and household plan docs (`DS8_api_contracts_rest.md`, `plans/household_identity/*`).

## Doc changes (2026-07-10)
- Documented the shared DS8/V3DT/DS9 identity-v2 runtime, default shadow and calibrated-authoritative modes, server-only exact-observation enrollment evidence, whole-frame assignment, proof-only overlap permits, nullable authoritative unknown identity, and explicit ReID engine/layer/dimension provenance (`DS8_api_contracts_rest.md`, `DS8_api_contracts_ws.md`, `DS8_metadata_contracts.md`).
- Hardened identity authority documentation with artifact v2 two-stratum
  benchmark/household evidence, deterministic correlation units,
  truth-person-worst-case benchmark confidence, encounter-worst-case local
  checks, monotonic local tightening, 1% benchmark FAR/misidentification
  ceilings, active semantic/artifact pins, conservative gallery envelopes, and
  owner-only fail-loud evidence
  storage (`DS8_api_contracts_rest.md`, `DS8_testing_guide.md`,
  `plans/household_identity/calibration_and_enrollment.md`).
- Documented the bounded no-follow immutable scene-release boundary, exact
  staged-tree publication, selected-byte serving semantics, shared
  DS8/V3DT/DS9 router parity, and the focused adversarial/real-candidate gate
  (`DS8_api_contracts_rest.md`, `DS8_testing_guide.md`,
  `Virtual_Twin_Reconstruction.md`, `plans/spatial_os/*`).

## Doc changes (2026-07-19)
- Documented strict invalid-world decision provenance, exact observation-to-
  source lineage, and conflict-safe canonical room derivation shared by DS8 and
  DS9 (`DS8_api_contracts_ws.md`, `DS8_metadata_contracts.md`,
  `plans/DS8/ds8_design_decisions.md`,
  `plans/DS8/ds8_migration_checklist_hooks.md`).

## Doc changes (2026-07-29)
- Documented the DS8/DS9 camera-local BEV coverage-envelope union, its
  separation from active MapAnything raster bounds, raw-metric rendering beyond
  the raster, and non-clamping boundary tolerance
  (`DS8_api_contracts_ws.md`, `Telemetry_Schema.md`,
  `plans/DS8/ds8_design_decisions.md`,
  `plans/DS8/ds8_migration_checklist_telemetry.md`).

## Doc changes (2026-07-08)
- Documented producer-owned human pathing realism (`PersonGroundState`): stationary/idle lock, posture-aware floor contact, source hysteresis, human CV filter, trail non-append while locked, and public `motion_mode` / `posture` / `trail_append_allowed` / `idle_jitter_m` fields on tracking + BEV payloads (`DS8_Baselines.md`, `DS8_api_contracts_ws.md`, `DS8_metadata_contracts.md`, `Telemetry_Schema.md`, `DS8_MIGRATION_KNOWLEDGE_BASE.md`, `DS8_README_FOR_AGENTS.md`, `plans/DS8/ds8_design_decisions.md`).

## Doc changes (2026-06-28)
- Documented the DS8 MapAnything-derived top-down floorplan artifact, including its cache location, `get_floorplan` creation path, X/Z raster convention, payload layers, and validation helpers (`MapAnything_Depth.md`, `DEPTH_STACK_FLOW_V2.md`).

## Doc changes (2026-03-07)
- Updated the DS8 seg+depth rebuild notes to document the deterministic linear depth->seg path, canonical frame-space fusion, internal aligned-depth cache, and the no-bbox-fallback object-depth contract (`DS8_metadata_contracts.md`, `plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_hooks.md`, `docs/README.md`).
- Standardized agent-facing no-fallback policy language across repo, DS8, planning, and docs quickstart guidance so agents must surface blockers instead of masking them with alternates (`AGENTS.md`, `noesis/AGENTS.md`, `plans/AGENTS.md`, `plans/menon_world_unification/AGENTS.md`, `plans/DS8/v3dt/AGENTS.md`, `docs/AGENTS.md`, `docs/DS8_README_FOR_AGENTS.md`).

## Doc changes (2026-02-22)
- Updated BEV/floorplan tracking + trails docs to match world-first `menon_scene` (scene units) behavior and frontend-owned smoothing/persistence (`DS8_Baselines.md`, `DS8_README_FOR_AGENTS.md`, `CODEBASE_DESCRIPTION.md`).
- Updated WS contracts for tracking world fields and `floorplan_response` payload shape (`DS8_api_contracts_ws.md`).
- Removed the operator-facing coordinate-toggle plan text to reflect the current single supported mode (`plans/bev_world_space_migration_plan.md`).

## Doc changes (2026-02-07)
- Removed remaining references to the deprecated mosaic JPEG/WebSocket path from active docs.

## Doc changes (2026-02-06)
- Standardized AGENTS policy precedence across repo/docs/plans scopes and clarified archive docs as non-normative (`AGENTS.md`, `docs/AGENTS.md`, `plans/AGENTS.md`, `noesis/AGENTS.md`, `docs/history/ds8/v3dt/AGENTS.md`).
- Reduced V3DT AGENTS duplication by keeping only subtree-specific constraints and locking recovery commands to baseline calibration defaults (`plans/DS8/v3dt/AGENTS.md`, `plans/DS8/v3dt/README.md`).
- Removed active-doc references to deprecated stack naming and deprecated runtime entrypoint terms; active docs are now DS8-only while historical material remains under `docs/history/`.

## Doc changes (2026-02-02)
- Refreshed WebSocket contracts (`DS8_api_contracts_ws.md`) and telemetry summary (`Telemetry_Schema.md`) to match current DS8 runtime output.
- Archived older references for WebSocket/telemetry/depth/occupancy and replaced with DS8 summaries (`WebSocket_API.md`, `depth_metadata.md`, `Occupancy_Publishing.md`, `MapAnything_Depth.md`).
- Updated baselines and overview docs for RTSP→WebRTC mosaic delivery, BEV frame defaults, and ReID/pose/MapAnything env toggles (`DS8_Baselines.md`, `DS8_README_FOR_AGENTS.md`, `CODEBASE_DESCRIPTION.md`).
- Clarified exclusion/ROI handling and MapAnything engine details (`Static_ROI_Exclusion.md`, `DS8_roi_editor.md`, `DS8_MIGRATION_KNOWLEDGE_BASE.md`).
