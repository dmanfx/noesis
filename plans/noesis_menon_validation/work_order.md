# Noesis/Menon Validation Toolbox Work Order

Status: planning scaffold as of 2026-05-27.

This work order sequences the validation toolbox so future agents can implement
it in reviewable chunks. Each phase should produce both automated checks and
human-readable evidence.

## Target implementation shape

Preferred structure:

```text
noesis/validation/
  core/
  geometry/
  camera/
  tracking/
  bev/
  menon/
  reports/
  regression/

tests/test_validation_*.py
scripts/noesis_validation_runner.py
scripts/noesis_menon_reprojection_report.py
scripts/noesis_validation_artifact_index.py
```

Implementation paths may change if the repo already has a better local pattern,
but validators should stay domain-separated and report through one common result
schema.

## Phase 0 - Inventory, fixtures, and acceptance gates

- [x] Inventory existing validation assets and map them into the catalog.
  - Include at least `scripts/menon_bev_track_parity_smoke_test.py`,
    `scripts/validate_virtual_twin_tracking.py`,
    `scripts/validate_virtual_twin_visual_alignment.py`,
    `scripts/menon_pose_calibration_smoke_test.py`,
    `scripts/sanity_check_v3dt_calibration.py`, and existing pytest coverage for
    calibration, BEV, depth registration, virtual twin, and Menon pose.
  - _2026-05-27 (Codex): Added `validation_asset_inventory.md` mapping the existing required scripts and relevant pytest coverage into the tiered toolbox._
- [x] Define a fixture registry for canonical rooms, cameras, clips, anchor
  points, calibration bundles, Menon scene revisions, and expected tolerances.
  - _2026-05-27 (Codex): Added `plans/noesis_menon_validation/fixture_registry.json` and `noesis/validation/fixtures.py`; validated registry resolution through `python3 scripts/noesis_validation_runner.py --fixture-registry plans/noesis_menon_validation/fixture_registry.json --fixture-id minimal_validation_fixture`._
  - _2026-05-27 (Codex): Added `minimal_telemetry` registry entry for saved DS8-style tracking/BEV telemetry validation._
  - _2026-05-27 (Codex): Added `minimal_menon_trace` registry entry for saved Menon placement-trace validation._
  - _2026-05-27 (Codex): Added `minimal_menon_browser_snapshot` registry entry for converting Menon browser debug state into the shared Menon trace contract._
- [x] Create a minimal known-anchor fixture for one room first.
  - Include 5 to 10 anchors such as floor corners, doorway edges, wall
    midpoints, camera mount point, and known static object corners.
  - _2026-05-27 (Codex): Expanded `minimal_fixture.json` to six known anchors: floor corner, wall midpoint, two doorway edges, camera mount, and couch corner; validated through the fixture runner._
- [x] Define artifact storage conventions.
  - Preferred runtime output root: `diagnostics/validation/<run_id>/`.
  - Reports must include enough metadata to reproduce the config, source clips,
    Menon scene revision, and thresholds.
  - _2026-05-27 (Codex): Added `noesis/validation/artifacts.py`; runner writes JSON, Markdown, reprojection PNGs, and `visual/index.json` under `diagnostics/validation/<run_id>/`._
- [x] Define validation tiers.
  - Tier 1: pure unit/schema/math, no GPU.
  - Tier 2: fixture/offline report generation.
  - Tier 3: DS8 runtime validation through `noesis/ds8_runtime.py`.
  - Tier 4: DS8 plus Menon cross-space validation.
  - _2026-05-27 (Codex): Added `validation_tiers.md` with tier definitions, command examples, Menon blocked-evidence handling, and the toolbox adoption gate._

## Phase 1 - Core validation framework

- [x] Add common result objects for pass, warning, fail, blocked, and skipped.
  - _2026-05-27 (Codex): Added `noesis/validation/core.py` with common check/report/status objects and validated via `python3 -m pytest tests/test_validation_toolbox.py -q`._
- [x] Add common confidence categories:
  detection, tracking, ReID, projection, geometry, temporal, semantic, and
  end-to-end.
  - _2026-05-27 (Codex): Added `ConfidenceScores` and report summary serialization; focused tests passed._
- [x] Add failure categories:
  calibration, transform, projection, scene, temporal, semantic, sync, model,
  data quality, regression, and infrastructure.
  - _2026-05-27 (Codex): Added `FailureType` taxonomy matching the artifact contract; focused tests passed._
- [x] Add validators for units, finite values, bounds, transform invertibility,
  matrix convention declarations, handedness, and round-trip conversion.
  - _2026-05-27 (Codex): Added `noesis/validation/transforms.py` with transform matrix, finite-point, frame declaration, and round-trip validators; focused tests passed._
- [x] Add JSON and Markdown report writers.
  - _2026-05-27 (Codex): Added report JSON writer plus Markdown rendering in `noesis/validation/reports.py`; focused tests passed._
- [x] Add unit tests for result serialization, confidence classification,
  failure taxonomy, and transform round trips.
  - _2026-05-27 (Codex): Added `tests/test_validation_toolbox.py`; `python3 -m pytest tests/test_validation_toolbox.py -q` passed._

## Phase 2 - Camera, calibration, and reprojection validators

- [x] Add intrinsics validation.
  - Check resolution match, focal length plausibility, principal point,
    aspect/FOV consistency, distortion model, and raw-vs-dewarped declaration.
  - _2026-05-27 (Codex): Added `validate_intrinsics(...)` in `noesis/validation/camera.py`; focused tests passed._
- [x] Add extrinsics validation.
  - Check camera position, height, yaw/pitch/roll, world-to-camera convention,
    floor-ray intersections, frustum intersection with floor/walls, and known
    world-point projection.
  - _2026-05-27 (Codex): Added `validate_extrinsics(...)` with camera height, forward floor ray, and bottom-center floor ray checks; focused tests passed._
- [x] Add known-anchor validation.
  - Compare expected and reconstructed anchor positions with good, warning, and
    fail tolerances.
  - _2026-05-27 (Codex): Added `validate_known_anchors(...)` in `noesis/validation/geometry.py`; focused tests passed._
- [x] Add camera reprojection overlays.
  - Generate camera image plus projected floor grid, room outline, known
    anchors, detected footpoints, and generated room/mesh edges when available.
  - _2026-05-27 (Codex): Added first overlay renderer for projected floor grid and known-anchor expected-vs-projected marks in `noesis/validation/visuals.py`._
  - _2026-05-27 (Codex): Expanded the camera overlay renderer to include projected room outlines, mesh edges, and detected footpoint markers from fixtures._
  - _2026-05-27 (Codex): Added source-frame detected bbox, projected bbox/3D avatar bbox, and mask-polygon overlay inputs to the fixture camera reprojection renderer; validated through the minimal fixture visual artifact runner._
- [x] Add reprojection metrics.
  - Report mean/max pixel error, anchor alignment, floor-grid alignment, edge
    alignment, footpoint reprojection error, bbox center error, 3D bbox overlap,
    mask overlap, and pose/foot overlap when data exists.
  - _2026-05-27 (Codex): Added known-anchor reprojection metrics in `validate_reprojection_anchors(...)`; bbox aspect/overlap metrics now live in `TRACK.detection_world_projection`, while mask and pose visual metrics remain pending._
- [x] Add tests and fixtures for dewarped-camera declarations.
  - A dewarped stream must explicitly say whether calibration applies to raw
    stream, dewarped stream, and inference resolution.
  - _2026-05-27 (Codex): Minimal fixture declares `stream_kind=dewarped`, `applies_to_raw=false`, `applies_to_dewarped=true`, and `expected_resolution`; focused tests now assert ambiguous raw/dewarped ownership fails._

## Phase 3 - Tracking, temporal, and BEV validators

- [x] Add detection-to-world validators.
  - Check bbox footpoint, mask footpoint, pose ankle/foot agreement,
    depth-assisted footpoint agreement, 3D bbox floor contact, person height,
    bbox aspect ratio, and ray-floor intersection validity.
  - _2026-05-27 (Codex): Added first detection-to-world projection validator for bbox/mask/pose/depth footpoint agreement, floor contact, room bounds, person height, and ray-floor validity._
  - _2026-05-27 (Codex): Extended `TRACK.detection_world_projection` with `image_bbox_xyxy`, `projected_bbox_xyxy`, `projected_bbox_iou`, and `bbox_center_error_px` evidence for bbox aspect and 3D bbox reprojection overlap; validated pass/fail coverage with `python3 -m pytest tests/test_validation_toolbox.py -q` and the minimal fixture runner._
- [x] Add per-frame projection confidence.
  - Combine bbox foot agreement, floor contact, room bounds, reprojection,
    temporal smoothness, and semantic validity into separate category scores,
    not one opaque number.
  - _2026-05-27 (Codex): Added first projection-confidence scorer from footpoint, floor, ray, room-bounds, and height components._
  - _2026-05-27 (Codex): Projection confidence now also incorporates optional source-image bbox aspect, projected 3D bbox overlap, and bbox center-error evidence when present._
  - _2026-05-27 (Codex): Added explicit `reprojection_score`, `temporal_smoothness_score`, and `semantic_validity_score` components plus per-component coverage counts in `TRACK.projection_confidence`; validated with focused pass/weak-confidence tests._
- [x] Add temporal validators.
  - Check position continuity, speed limits, acceleration limits, idle jitter,
    occlusion bridge behavior, room containment, doorway transitions, StableID
    continuity, ReID/geometry contradictions, and split/merge events.
  - _2026-05-27 (Codex): Added first temporal slice for speed, acceleration, and idle jitter in `noesis/validation/tracking.py`._
  - _2026-05-27 (Codex): Added occlusion-bridge validation with plausible bridge speed/duration and explicit uncertainty evidence._
  - _2026-05-27 (Codex): Added identity-continuity and ReID/geometry validators for StableID switches, duplicate simultaneous StableID placement, split/merge evidence, continuous-geometry ReID contradictions, and weak appearance evidence; fixture and telemetry reports now run these checks._
- [x] Add saved/live telemetry ingestion for tracking and BEV report generation.
  - _2026-05-27 (Codex): Added `noesis/validation/telemetry.py` and `scripts/noesis_validation_telemetry_report.py` to parse NDJSON or live WebSocket messages, validate tracking/BEV contracts, run motion and BEV/track agreement checks, and write `tracking/track_audit.json`; validated with `plans/noesis_menon_validation/minimal_telemetry.ndjson`._
  - _2026-05-27 (Codex): Extended telemetry parsing and `tracking/track_audit.json` with optional ReID/appearance identity fields used by the identity continuity checks._
- [x] Add BEV validators.
  - Check track-inside-room, no wall crossing, doorway transitions, speed,
    acceleration, stationary jitter, path smoothness, camera coverage,
    occlusion-aware uncertainty, and zone consistency.
  - _2026-05-27 (Codex): Added reusable BEV/track agreement, room-polygon containment, and wall-crossing checks._
  - _2026-05-27 (Codex): Added BEV zone-consistency and camera-coverage validators using room polygons and camera frustum footprints._
  - _2026-05-27 (Codex): Added doorway-transition validation so room changes must cross known doorway segments._
  - _2026-05-27 (Codex): Added BEV path-smoothness validation for path inflation and abrupt zig-zag heading changes; fixture and telemetry reports now emit `BEV.path_smoothness`._
- [x] Add BEV diagnostic overlays.
  - Room polygons, wall segments, doorways, camera frustums, track trails,
    confidence ellipses, raw footpoints, StableID labels, ReID confidence, and
    projection confidence.
  - _2026-05-27 (Codex): Added first fixture BEV diagnostic overlay renderer for room polygons, walls, doorways, camera frustums, track trails, and raw footpoints._
  - _2026-05-27 (Codex): Expanded the BEV overlay renderer and fixture inputs to draw uncertainty ellipses plus StableID, projection confidence, and ReID confidence annotations._
- [x] Add track audit report output.
  - Include stable ID, current room, world position, projection confidence,
    temporal confidence, ReID confidence, doorway/occlusion history, impossible
    motion events, and warnings.
  - _2026-05-27 (Codex): Expanded `tracking/track_audit.json` to include stable/tracker identity, current room, world position object and vector, projection/temporal/ReID confidence, last doorway transition, last occlusion age, last impossible motion event, speed summary, and warnings._

## Phase 4 - Menon cross-space validators

- [x] Add Noesis world to Menon scene transform audit.
  - Log every transform used to place an entity, including backend world meters,
    alignment matrices, scene units, room fit, and final Menon scene position.
  - _2026-05-27 (Codex): Added first reusable Menon placement and round-trip validators plus fixture schema support._
  - _2026-05-27 (Codex): Added `noesis/validation/menon_trace.py` and `scripts/noesis_validation_menon_trace_report.py` to validate saved Menon traces with explicit `backend_world_m` to `menon_scene` transform audit stages._
  - _2026-05-27 (Codex): Added `noesis/validation/menon_browser.py` and `scripts/noesis_validation_capture_menon_trace.py` to capture Menon browser debug globals and convert declared `backend_world_m` plus Menon scene placement evidence into the same transform-audit trace schema._
- [x] Add world/BEV/Menon agreement checks.
  - Verify world-to-BEV-to-world and world-to-Menon-to-world round trips and path
    shape agreement between BEV trails and Menon trails.
  - _2026-05-27 (Codex): Added world-to-Menon round-trip and placement agreement checks._
  - _2026-05-27 (Codex): Added Menon trail agreement against transformed Noesis world trails._
  - _2026-05-27 (Codex): Added direct BEV-to-Menon trail agreement using BEV X/Z samples transformed through the declared world-to-scene matrix._
  - _2026-05-27 (Codex): Added `BEV.world_round_trip` for explicit `world_to_bev_col_major` plus `world_bev_points` agreement and BEV-to-world round-trip validation in fixture and Menon trace reports; validated with focused pass/drift tests._
- [x] Add Menon avatar/object placement checks.
  - Check scale agreement, avatar floor contact, wall/furniture collision,
    movement orientation, and trail consistency.
  - _2026-05-27 (Codex): Added Menon floor-contact validation for captured placement traces._
  - _2026-05-27 (Codex): Added avatar scale, wall/furniture proxy collision, and movement-orientation validators for Menon trace and fixture evidence._
  - _2026-05-27 (Codex): Added `MENON.object_placement` for static/generated object world-to-scene placement, rendered dimensions, support contact, room compatibility, and wall/furniture collision proxy checks; validated with fixture and Menon trace samples plus focused fail coverage._
- [x] Add timestamp and latency alignment checks.
  - Compare Noesis frame timestamps, telemetry timestamps, Menon display
    timestamps, and render/update lag.
  - _2026-05-27 (Codex): Added timestamp alignment checks for saved Menon trace placements._
  - _2026-05-27 (Codex): Added `MENON.latency_alignment` with explicit Noesis frame, telemetry, Menon update, render, and display timestamps; validated pass/fail coverage in fixture and Menon trace reports._
- [x] Add Menon camera reprojection mode acceptance criteria.
  - Render Menon from a selected camera pose, overlay or compare the source
    frame, and draw detected bbox, projected avatar, room mesh edges, floor grid,
    and anchors.
  - _2026-05-27 (Codex): Added saved-trace acceptance checks for Menon camera reprojection evidence, including required source/render/layer declarations plus anchor/grid/edge pixel errors and bbox/avatar/mask overlap metrics._
  - _2026-05-27 (Codex): Added `menon_camera_reprojection_acceptance.md` with required fixture/live evidence, thresholds, blocked criteria, and acceptance commands._
- [x] Add Menon fixture and live modes.
  - Tools must accept `MENON_ROOT` or `--menon-root`; if unavailable, report
    blocked for Menon-specific checks instead of passing Noesis-only evidence.
  - _2026-05-27 (Codex): Added fixture-mode Menon placement/timestamp checks through the common runner._
  - _2026-05-27 (Codex): Added saved Menon trace mode with `--menon-root`, `MENON_ROOT`, and `--require-menon-root`._
  - _2026-05-27 (Codex): Added Playwright browser snapshot capture with `--validate`; browser snapshots now promote exposed camera-view reprojection debug evidence into `camera_reprojections` so live and fixture modes share `MENON.camera_reprojection`._

## Phase 5 - Scene, mesh, depth, and semantic validators

- [x] Add coordinate-system validation for generated scenes.
  - Check meters, axes, origin stability, camera pose convention, handedness,
    transform round trip, and scale anchors.
  - _2026-05-27 (Codex): Added `SceneCoordinateSystemObservation` and `validate_scene_coordinate_system(...)` to check declared meter units, expected axes, camera pose convention, transform handedness, origin drift, round-trip points, and scale anchors; validated in the minimal fixture and focused tests._
- [x] Add plane and room-geometry validation.
  - Check floor flatness, wall verticality, ceiling/floor parallelism,
    wall/floor orthogonality, room height, wall intersections, door/window
    plausibility, plane normals, and surface continuity.
  - _2026-05-27 (Codex): Added first pure fixture validators for floor/wall/ceiling normals, floor RMS, and room dimension anchors in `noesis/validation/scene.py`._
  - _2026-05-27 (Codex): Added room geometry constraint validation for wall/floor angle observations, wall-junction gaps/overlaps, doorway width/height/floor contact, normal sanity, and surface continuity gaps._
  - _2026-05-27 (Codex): Added explicit window cutout width, height, and sill-height validation under `scene.room_geometry_constraints`; validated focused pass/fail tests and the minimal fixture runner._
- [x] Add mesh-quality validation.
  - Check watertightness where expected, non-manifold topology, inverted normals,
    duplicate vertices/faces, triangle-density bounds, LOD preservation, texture
    atlas alignment, UV overlap, bounding boxes, and collision/BVH parity.
  - _2026-05-27 (Codex): Added first mesh stats validator for bounding boxes, triangle count, watertight expectation, non-manifold edges, inverted faces, duplicate faces, and UV overlap._
  - _2026-05-27 (Codex): Extended `SCENE.mesh_quality` with LOD preservation, texture atlas alignment/stretch, BVH validity, and collision/BVH parity fields; validated pass/fail coverage with `python3 -m pytest tests/test_validation_toolbox.py -q` and the minimal fixture runner._
- [x] Add Menon-specific asset checks.
  - Check room mesh scale/orientation/origin, floor walkability, wall/doorway
    constraints, camera marker position/aim, and debug overlay renderability.
  - _2026-05-27 (Codex): Added `SCENE.menon_assets` fixture validation for room mesh scale/orientation/origin/collision, floor walkability/alignment, doorway blocking, camera marker pose, and required debug overlay layers; validated pass/fail coverage with `python3 -m pytest tests/test_validation_toolbox.py -q` and the minimal fixture runner._
- [x] Add depth and monocular reconstruction validation.
  - Check relative depth ordering, metric scale alignment, temporal stability,
    plane-depth agreement, object-depth agreement, discontinuities, confidence
    weighting, and static-scene stability.
  - _2026-05-27 (Codex): Added first depth-anchor validator for metric error, plane residual, and temporal standard deviation._
  - _2026-05-27 (Codex): Extended depth-anchor validation with relative depth ordering, object-depth agreement, edge/discontinuity alignment, low-confidence fusion-weight checks, and static-scene variance; validated through the minimal fixture and focused tests._
- [x] Add semantic scene validation.
  - Check object support, object-wall intersections, doorway clearance,
    furniture size, object-room compatibility, persistent object consistency,
    known-object anchors, and free-space/walkability.
  - _2026-05-27 (Codex): Added first semantic object validator for support contact, wall intersection, dimension plausibility, and persistent static-object shift._
  - _2026-05-27 (Codex): Extended `SCENE.semantic_objects` with allowed-room, doorway-clearance, free-space, walkable-area, and known-anchor fields; validated pass/fail coverage with `python3 -m pytest tests/test_validation_toolbox.py -q` and the minimal fixture runner._

## Phase 6 - Regression runners and benchmarks

- [x] Define canonical scene-generation regression fixtures.
  - Empty room, known anchors, low light, IR/night, occluded furniture,
    wide-angle/dewarped, and changed furniture.
  - _2026-05-27 (Codex): Added `canonical_regression_fixtures.md` with scene-generation fixture IDs, required evidence, and required checks._
- [x] Define canonical tracking regression fixtures.
  - Straight-line walk, standing still, room crossing, doorway exit, couch
    occlusion, two-person crossing, near-camera person, far-camera person,
    IR/night movement, and camera-specific edge cases.
  - _2026-05-27 (Codex): Added canonical tracking fixture IDs and acceptance evidence for projection, temporal, room, occlusion, identity, and camera-edge cases._
- [x] Define canonical Menon regression fixtures.
  - Static avatar placement, BEV/Menon path agreement, camera-view render
    overlay, room mesh collision, multi-camera frustums, and timestamp sync.
  - _2026-05-27 (Codex): Added canonical Menon fixture IDs and required evidence for static placement, BEV path agreement, camera-view overlays, collision, multi-camera frustums, and timestamp sync._
- [x] Add a benchmark runner that emits versioned JSON summaries and optional
  visual artifacts.
  - _2026-05-27 (Codex): Added `scripts/noesis_validation_regression_runner.py` to execute registered fixture, telemetry, and Menon-trace cases and write `regression_summary.json`._
  - _2026-05-27 (Codex): Extended the regression runner to execute registered Menon browser snapshot cases by converting them into Menon trace reports._
- [x] Add golden-output comparison with numeric thresholds and explicit visual
  artifact diffs for review-only checks.
  - _2026-05-27 (Codex): Added registry-level expected status, minimum check count, and maximum fail/warning/blocked thresholds._
  - _2026-05-27 (Codex): Added `noesis/validation/golden.py` and regression-runner artifact expectations for visual artifact existence/size, sha256 checks, and optional pixel-level golden image diffs with generated diff PNGs._
- [x] Add regression failure categorization so agents know whether to inspect
  calibration, transform, projection, scene, temporal, semantic, sync, model,
  data quality, or recent code changes first.
  - _2026-05-27 (Codex): Regression summaries now aggregate failure categories from non-passing checks and expectation/artifact misses, include a dominant failure category, and provide a suggested diagnostic focus._

## Phase 7 - Documentation, adoption, and agent workflow

- [x] Update `docs/DS8_testing_guide.md` with the validation tier map and runner
  commands once scripts exist.
  - _2026-05-27 (Codex): Added tier definitions and runner commands for fixture, telemetry, Menon trace, and registry regression validation._
- [x] Update relevant WebSocket/metadata/API contracts when report fields or
  telemetry diagnostics become public.
  - _2026-05-27 (Codex): Updated WebSocket tracking telemetry docs, metadata track dictionaries, and REST API status notes for optional validation diagnostics, `backend_world_m` trace naming, and CLI/report-only validation artifacts._
- [x] Add concise checklist entries to DS8 migration or feature work orders when
  a validation layer becomes required for a feature family.
  - _2026-05-27 (Codex): Updated `plans/DS8/ds8_master_work_orders.md` to mark Phase 0 and the first validation slice complete and to require toolbox-tier evidence for new DS8 spatial feature work._
- [x] Add examples of passing, warning, failing, and blocked reports.
  - _2026-05-27 (Codex): Added `report_examples.md` with compact pass/warning/fail/blocked report snippets, agent response patterns, and minimal commands; linked it from `README.md` and the local `AGENTS.md` read order._
- [x] Add a final adoption gate:
  - `python3 -m pytest tests/test_validation_*.py -q`
  - focused DS8 runtime smoke for affected surface
  - Menon parity or blocked evidence when Menon is in scope
  - `./scripts/check_agents_docs_consistency.py`
  - `git diff --check`
  - _2026-05-27 (Codex): Added the adoption gate to `validation_tiers.md` and `AGENTS.md`; validated with pytest, py_compile, fixture report, telemetry report, Menon trace report, registry regression, JSON syntax checks, docs consistency, diff whitespace, and portable-path diff scan. No live DS8 smoke was required for this GPU-free toolbox/documentation slice._

## Implemented commands

```bash
python3 scripts/noesis_validation_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --fixture-id minimal_validation_fixture
python3 scripts/noesis_validation_telemetry_report.py \
  --input plans/noesis_menon_validation/minimal_telemetry.ndjson \
  --run-id minimal_telemetry
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace
python3 scripts/noesis_validation_capture_menon_trace.py \
  --url http://127.0.0.1:5173 \
  --output diagnostics/validation/menon_browser_trace/trace.json \
  --screenshot \
  --validate
python3 scripts/noesis_validation_regression_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --run-id minimal_regression
python3 -m pytest tests/test_validation_toolbox.py -q
```

## First implementation slice

Build these first because they catch the most damaging Noesis/Menon defects:

1. Transform round-trip validator.
2. Validation JSON report schema.
3. Known-anchor validator.
4. Camera reprojection overlay and metrics.
5. Track velocity/teleport detector.
6. Room bounds and wall-crossing validator.
7. BEV/Menon coordinate agreement check.
