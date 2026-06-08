# AGENTS.md - plans/noesis_menon_validation

This directory defines the validation toolbox workstream for Noesis, BEV, and
Menon cross-space correctness.

## Policy precedence

- This file extends the repo-root `AGENTS.md` and `plans/AGENTS.md`.
- The root DS8 rules still apply: DS8 is canonical, no hidden fallbacks, no new
  CPU/appsink branches in the DS8 runtime path, and no guessed DeepStream APIs.
- This directory is planning/instructional unless a work item explicitly points
  to implementation files under `noesis/`, `tests/`, `scripts/`, or a Menon
  checkout provided by `MENON_ROOT` or `--menon-root`.

## Required read order

For work in this validation workstream, read these files first:

1. `AGENTS.md`
2. `README.md`
3. `work_order.md`
4. `validation_catalog.md`
5. `artifact_contracts.md`
6. `validation_asset_inventory.md`
7. `validation_tiers.md`
8. `report_examples.md`
9. `canonical_regression_fixtures.md`
10. `menon_camera_reprojection_acceptance.md`
11. `docs/DS8_testing_guide.md`
12. `docs/DS8_api_contracts_ws.md`
13. `docs/DS8_metadata_contracts.md`

If the task changes DS8 runtime behavior, also read
`plans/DS8/ds8_master_work_orders.md` and the relevant DS8 checklist before
editing code.

## When to use this toolbox

Use this workstream whenever a change affects any part of this chain:

```text
camera image -> detection/pose/depth -> backend_world_m -> BEV -> Menon scene -> camera reprojection
```

That includes calibration, depth registration, room geometry, BEV rendering,
tracking world output, StableID/ReID display, virtual-twin assets, Menon
reprojection, and any diagnostic overlay meant to prove spatial correctness.

## Validation depth rules

- For schema-only changes, run schema/unit tests and update the machine-readable
  report contract if fields change.
- For transform or calibration changes, require round-trip, known-anchor,
  floor-ray, and reprojection checks.
- For tracking or BEV changes, require per-track projection confidence, room
  bounds, velocity/jitter, and BEV/track-world agreement checks.
- For Menon-facing changes, require Noesis-to-Menon transform audit and at least
  one cross-space validation against the active Menon scene or a captured Menon
  fixture.
- For visual or geometry deliverables, require visual artifacts that show why the
  numeric result passed or failed.
- For live-runtime acceptance, validate through `noesis/ds8_runtime.py`; do not
  substitute deprecated runtime paths.

## Evidence rules

Every completed work item must record:

- the command, script, fixture, or manual review method used;
- the artifact path or report path produced;
- pass, warning, fail, or blocked status;
- the failure category when a check fails;
- whether Menon was validated live, fixture-only, or not in scope.

Prefer machine-readable JSON reports plus a short human summary. Visual artifacts
are supporting evidence, not replacements for numeric checks.

## Menon checkout handling

- Do not hardcode a machine-local Menon path in code, docs, or scripts.
- Validation tools should accept `--menon-root` or `MENON_ROOT`.
- If Menon is unavailable and the task requires it, mark the check blocked
  rather than silently downgrading to Noesis-only validation.

Use the saved-trace validator when a Menon placement/debug export is available:

```bash
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace
```

Use the browser capture when Menon is running and the acceptance question is
whether the browser is placing live tracks in the same scene frame that Noesis
declares, or when camera-view reprojection evidence is exposed by Menon debug
state:

```bash
python3 scripts/noesis_validation_capture_menon_trace.py \
  --url http://127.0.0.1:5173 \
  --output diagnostics/validation/menon_browser_trace/trace.json \
  --screenshot \
  --validate
```

If Playwright, the page, the browser debug globals, the declared
`backend_world_m` source point, the Menon scene transform, or required
camera-reprojection source/render layers are unavailable, the capture/report
must fail or block rather than substituting fixture evidence.

For Menon-facing acceptance work, pass `--require-menon-root` with either
`--menon-root` or `MENON_ROOT` so missing Menon evidence is reported as blocked.

Use the registry runner when a change should preserve the current fixture,
telemetry, and Menon-trace baselines:

```bash
python3 scripts/noesis_validation_regression_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --run-id minimal_regression
```

Before closing validation-toolbox implementation work, run the adoption gate in
`validation_tiers.md`. For a feature that affects live DS8 runtime behavior, add
the focused `noesis/ds8_runtime.py` smoke named by the relevant DS8 work order.
For a feature that affects Menon but cannot reach a running Menon instance,
record the blocked Menon evidence instead of replacing it with a Noesis-only
pass.

## Progress updates

- Update `work_order.md` checkboxes as items are completed.
- Add dated one-line validation notes under completed items.
- If a non-trivial validation schema, threshold, or pass/fail rule changes, add a
  short decision entry to `plans/DS8/ds8_design_decisions.md`.
- If implementation changes docs or AGENTS files, run
  `./scripts/check_agents_docs_consistency.py` and `git diff --check`.
