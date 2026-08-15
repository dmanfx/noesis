# Canonical regression fixtures

`fixture_registry.json` is the machine-readable inventory. The four small
fixtures characterize report and consumer contracts; they are not evidence of
live perception quality.

- `minimal_validation_fixture`: deterministic geometry, transform, room,
  projection, BEV, and overlay checks.
- `minimal_telemetry`: tracking/occlusion/BEV wire and motion checks.
- `minimal_menon_trace`: world-to-scene placement and trail/avatar checks.
- `minimal_menon_browser_snapshot`: browser snapshot adaptation to the shared
  Menon trace contract.

Change a fixture only when the corresponding contract intentionally changes.
Update its registry expectations and run the directly affected toolbox tests.
