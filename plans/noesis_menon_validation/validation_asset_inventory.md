# Validation asset inventory

## Code-consumed fixtures

| Asset | Consumer |
| --- | --- |
| `fixture_registry.json` | regression runner and toolbox tests |
| `minimal_fixture.json` | geometry/report fixture tests |
| `minimal_telemetry.ndjson` | telemetry report tests |
| `minimal_menon_trace.json` | Menon trace report tests |
| `minimal_menon_browser_snapshot.json` | browser-to-trace adapter tests |

## Runtime evidence

Live evidence must identify the native DS9.1 run, cameras, coordinate frames,
timestamps, and producer/consumer sequence. Store only the bounded artifact
needed for the result. Do not recursively copy runtime, recording, or staging
trees.
