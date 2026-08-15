# Noesis/Menon validation toolbox

This directory contains small deterministic fixtures used by
`tests/test_validation_toolbox.py` and the `scripts/noesis_validation_*` tools.
It validates coordinate, telemetry, projection, BEV, and Menon-consumer
contracts without requiring a broad application test.

## Stable fixture paths

- `fixture_registry.json`: fixture inventory and expected report bounds.
- `minimal_fixture.json`: GPU-free geometry/contract characterization.
- `minimal_telemetry.ndjson`: saved tracking/BEV telemetry.
- `minimal_menon_trace.json`: saved world-to-scene consumer trace.
- `minimal_menon_browser_snapshot.json`: saved browser-consumer snapshot.

These paths are code-consumed and must not be moved merely because the original
planning work is complete.

## Use

1. Start with the specific unit/component tests for the code changed.
2. Use `scripts/noesis_validation_regression_runner.py` for the registered
   deterministic fixtures only when their producer/consumer contract changed.
3. Use `scripts/noesis_validation_telemetry_report.py` for saved or short live
   native DS9.1 telemetry.
4. Use `scripts/noesis_validation_menon_trace_report.py` when the Menon
   world-to-scene consumer changed.
5. Use one direct browser/live smoke only when the affected behavior cannot be
   proven from fixtures or saved telemetry.

See `validation_tiers.md` and `validation_catalog.md`. Historical design,
adversarial-matrix, and release-era instructions are retained under
`plans/archive/completed/noesis_menon_validation_ds8_docs/`.
