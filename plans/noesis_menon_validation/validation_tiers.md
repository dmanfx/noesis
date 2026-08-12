# Validation Tiers

Status: current as of 2026-05-27.

Use the lowest tier that proves the changed behavior, then add higher tiers when
the change crosses runtime, spatial, or Menon boundaries.

## Default execution rule

These tiers describe direct application evidence, not a release ceremony. For
ordinary work, select the narrowest tier covering the changed producer,
contract, consumer, and one practical runtime path, then stop. Do not create
Menon appliance releases, clone state, publish selectors, render deployment
bundles, or run promotion/rollback rehearsals unless the user explicitly asks
for a release or the behavior requires an external service/state transition.
The adoption gate below validates this toolbox itself; it is not a mandatory
precondition for every feature change. Missing private inputs or unavailable
hardware should be reported as blocked evidence rather than replaced by a
wider, unrelated tier.

## Tier 1 - Unit, schema, and math

No GPU or live runtime required.

Use for:

- report schema changes;
- transform, coordinate, geometry, and confidence math;
- fixture parser behavior;
- static contract helpers.

Typical commands:

```bash
python3 -m pytest tests/test_validation_toolbox.py -q
python3 -m py_compile noesis/validation/*.py scripts/noesis_validation_*.py
```

## Tier 2 - Offline fixtures and regression summaries

No GPU or live runtime required. Produces machine-readable reports and visual
artifacts.

Use for:

- transform/camera/anchor/scene/tracking/BEV/Menon fixture changes;
- visual overlay changes;
- regression-threshold updates.

Typical commands:

```bash
python3 scripts/noesis_validation_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --fixture-id minimal_validation_fixture
python3 scripts/noesis_validation_regression_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --run-id minimal_regression
```

## Tier 3 - DS8 runtime and telemetry

Requires DS8 runtime evidence from `noesis/ds8_runtime.py` or a saved DS8
telemetry capture.

Use for:

- tracking-world, BEV, StableID/ReID, depth, ROI, WebSocket, and REST behavior;
- runtime contract changes;
- acceptance evidence where live DS8 behavior matters.

Typical commands:

```bash
python3 scripts/noesis_validation_telemetry_report.py \
  --input plans/noesis_menon_validation/minimal_telemetry.ndjson \
  --run-id minimal_telemetry
python3 scripts/noesis_validation_telemetry_report.py \
  --ws ws://127.0.0.1:6008 \
  --duration 20 \
  --run-id live_tracking_bev_window
```

Run the relevant existing smoke script from `validation_asset_inventory.md` when
the changed surface has a focused runtime gate.

## Tier 4 - DS8 plus Menon cross-space validation

Requires a Menon trace, a Menon checkout, or live Menon browser evidence. If
Menon is required and unavailable, report blocked evidence.

Use for:

- Noesis-world to Menon-scene transform changes;
- virtual-twin assets and room alignment;
- Menon camera-view reprojection;
- BEV/Menon path agreement and latency alignment.

Typical commands:

```bash
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace
umask 077
MENON_VALIDATION_RUN_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/noesis/menon-tier4/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MENON_VALIDATION_RUN_DIR"
python3 scripts/noesis_validation_capture_menon_trace.py \
  --url http://127.0.0.1:5175 \
  --storage-state "$MENON_PLAYWRIGHT_STORAGE_STATE" \
  --output "$MENON_VALIDATION_RUN_DIR/trace.json" \
  --screenshot \
  --validate
MENON_ROOT=../Menon python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --require-menon-root \
  --run-id minimal_menon_trace_with_checkout
```

Tier 4 live evidence is accepted only after a fresh same-origin auth-session
proof and exact coherence across the real canonical world state entities,
presentation/debug snapshot, current render paths, active promoted scene
cohort, and one authored world-to-scene transform. The evidence directory is
`0700` and every capture/report artifact is `0600`.

## Adoption Gate

Before treating this toolbox itself as healthy, run:

```bash
python3 -m pytest tests/test_validation_*.py -q
python3 -m py_compile noesis/validation/*.py scripts/noesis_validation_*.py
python3 scripts/noesis_validation_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --fixture-id minimal_validation_fixture \
  --run-id minimal_validation_fixture
python3 scripts/noesis_validation_telemetry_report.py \
  --input plans/noesis_menon_validation/minimal_telemetry.ndjson \
  --run-id minimal_telemetry
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace
python3 scripts/noesis_validation_regression_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --run-id minimal_regression
./scripts/check_agents_docs_consistency.py
git diff --check
```

For feature work, add the focused DS8 runtime smoke tied to the changed surface.
If Menon is in scope but unavailable, record the blocked Menon evidence instead
of substituting Noesis-only fixture output.
