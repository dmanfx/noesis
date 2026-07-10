# Reference Documentation

This folder is the live reference for the Noesis video stack. Historical plans
and worklogs live in `docs/history/`.

## Quick Links
- **Start here (DS8):** `DS8_README_FOR_AGENTS.md`
- **Current baselines & defaults:** `DS8_Baselines.md`
- **Pose-assisted StableID:** `DS8_pose_stable_id_integration.md`
- **APIs & contracts:** `DS8_api_contracts_ws.md`, `DS8_api_contracts_rest.md`,
  `DS8_metadata_contracts.md`, `Telemetry_Schema.md`
- **WebSocket overview:** `WebSocket_API.md`
- **Testing:** `DS8_testing_guide.md`
- **Codebase overview:** `CODEBASE_DESCRIPTION.md`
- **V3DT forensics & calibration:** `DS8_v3dt_forensics.md`
- **Depth / MapAnything:** `MapAnything_Depth.md`, `MapAnything_Heatmap_Viewer.md`, `DEPTH_STACK_FLOW_V2.md`, `depth_metadata.md`
- **Virtual twin reconstruction:** `Virtual_Twin_Reconstruction.md`
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
- Added Menon browser debug capture and conversion into the shared Menon trace
  contract for live browser placement validation
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
