# Noesis/Menon Validation Toolbox Plan

Status: implementation scaffold as of 2026-05-27.

This workstream turns Noesis/Menon spatial validation into a layered toolbox. The
goal is not one "looks right" check. The goal is to prove that a result survives
the full chain from pixels, through DS8 world state, into BEV and Menon, and back
to camera-view evidence.

## Doctrine

The primary object of validation is the transform chain:

```text
camera image
-> detection, pose, mask, or depth anchor
-> camera ray
-> backend_world_m
-> BEV frame
-> Menon scene
-> Menon camera-view reprojection
-> original camera image alignment
```

Individual subsystem checks still matter, but a feature is not considered
spatially validated until the relevant links in this chain agree.

## Deliverables

- A shared Python validation package under `noesis/validation/`.
- Focused unit tests under `tests/` for pure math, schemas, reports, and fixture
  validators.
- Runtime/fixture scripts under `scripts/` for capture replay, report generation,
  visual overlay generation, and Menon parity checks.
- A machine-readable validation report format shared by geometry, calibration,
  tracking, BEV, Menon, confidence, and regression validators.
- Visual artifacts that make every spatial pass/fail explainable from camera,
  BEV, and Menon viewpoints.
- Agent-facing instructions that define which validations are required for each
  feature type.

## Workstream files

- `AGENTS.md`: operating rules for future agents.
- `work_order.md`: implementation sequence and acceptance gates.
- `validation_catalog.md`: complete toolbox catalog by domain.
- `artifact_contracts.md`: report schema, confidence levels, failure taxonomy,
  and artifact expectations.
- `validation_asset_inventory.md`: existing scripts/tests mapped into the tiered
  toolbox.
- `validation_tiers.md`: tier definitions and command-selection guidance.
- `report_examples.md`: pass, warning, fail, and blocked report examples plus
  response patterns for future agents.
- `canonical_regression_fixtures.md`: named scene, tracking, and Menon
  regression fixture definitions with required evidence and checks.
- `menon_camera_reprojection_acceptance.md`: fixture and live acceptance rules
  for camera-view Menon/source-frame alignment.

## Related references

- `docs/DS8_testing_guide.md`
- `docs/DS8_api_contracts_ws.md`
- `docs/DS8_api_contracts_rest.md`
- `docs/DS8_metadata_contracts.md`
- `docs/Virtual_Twin_Reconstruction.md`
- `plans/menon_world_unification/validation_matrix.md`
- `plans/DS8/ds8_master_work_orders.md`
- `plans/DS8/ds8_design_decisions.md`

## Implementation posture

Build validators in layers. Start with small deterministic checks that do not
need a GPU, then add DS8 runtime captures, then add Menon cross-space rendering.
Do not mask failures by switching to fallback paths. If the canonical DS8 or
Menon path is unavailable, the validation result should say what is blocked and
which evidence is missing.

## Current first slice

The initial GPU-free toolkit slice now includes:

- shared report/status/confidence/failure-taxonomy objects in `noesis/validation/core.py`;
- JSON and Markdown report writers;
- transform matrix and round-trip validators;
- known-anchor validation;
- camera intrinsics, extrinsics, floor-ray, and known-anchor reprojection checks;
  - dewarped-camera checks require raw/dewarped ownership and active resolution
    declarations;
- track speed, acceleration, idle-jitter, occlusion-bridge, doorway-transition,
  identity-continuity, ReID/geometry-consistency, BEV path-smoothness, and
  BEV/track agreement checks;
- BEV zone-consistency and camera-coverage checks;
- detection-to-world projection checks for footpoint agreement, floor contact,
  room bounds, ray-floor validity, person height, image bbox aspect, projected
  3D bbox overlap, and projection confidence;
- Menon world-to-scene round-trip, placement agreement, and timestamp alignment checks;
- Menon latency alignment checks for telemetry, update, render, and display
  timestamps;
- Menon BEV-to-scene trail agreement plus avatar scale, collision,
  movement-orientation, and static/generated object placement checks;
- Menon camera-view reprojection acceptance checks for source/render layers,
  projected bbox/avatar, room mesh, floor grid, anchors, pixel error, and
  overlap metrics;
- room polygon containment and wall-crossing checks;
- camera reprojection overlay PNG generation with floor grid, known anchors,
  room outline, mesh edges, and detected footpoints;
- BEV diagnostic overlay PNG generation with room polygons, walls, doorways,
  camera frustums, track trails, raw footpoints, confidence ellipses, StableID
  labels, projection confidence, and ReID confidence;
- visual artifact indexing;
- scene coordinate-system, plane geometry, room dimension, room/opening/window
  constraint, mesh quality, Menon asset sanity, depth consistency, and semantic
  object support checks;
  - depth consistency includes metric anchors, relative ordering, plane/object
    agreement, temporal/static stability, edge alignment, and confidence/fusion
    weighting;
  - semantic object checks include support contact, wall intersections,
    doorway clearance, object-room compatibility, persistent-object anchors, and
    free-space/walkability evidence;
- a fixture registry: `plans/noesis_menon_validation/fixture_registry.json`;
  - registry entries can declare expected status, minimum check count, and
    maximum failure/warning/blocked counts for regression gating;
- a fixture runner: `scripts/noesis_validation_runner.py`;
- a saved/live DS8 telemetry runner: `scripts/noesis_validation_telemetry_report.py`;
- a saved Menon placement-trace runner:
  `scripts/noesis_validation_menon_trace_report.py`;
- a registry regression runner:
  `scripts/noesis_validation_regression_runner.py`;
  - regression entries can check expected status/counts, artifact size/hash, and
    optional golden image diffs;
  - regression summaries include failure categories, a dominant category, and a
    suggested diagnostic focus for failed cases;
- a minimal fixture: `plans/noesis_menon_validation/minimal_fixture.json`;
- a minimal DS8-style telemetry capture: `plans/noesis_menon_validation/minimal_telemetry.ndjson`;
- a minimal Menon placement trace:
  `plans/noesis_menon_validation/minimal_menon_trace.json`;
- a Menon browser snapshot adapter and capture CLI:
  `noesis/validation/menon_browser.py` and
  `scripts/noesis_validation_capture_menon_trace.py`;
- a minimal Menon browser debug snapshot:
  `plans/noesis_menon_validation/minimal_menon_browser_snapshot.json`;
- focused unit tests: `tests/test_validation_toolbox.py`.

Quick validation:

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
