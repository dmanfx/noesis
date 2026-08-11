# Integrations Playbook (DS8)
_Status: validated against code on 2026-07-10._

DS8, V3DT, and DS9 do not wire MQTT/Influx publishing. The older integrations
playbook is archived under `docs/history/`; it is not an activation runbook for
the current appliance.

## What Exists
- `noesis.telemetry.publishers.bind_occupancy_publisher(...)` exposes an adapter
  slot, but every active runtime binds it to `None` and no active occupancy
  publisher implementation exists.
- `geometry/depth_publisher.py` is a dormant depth-summary publisher
  (median/p10/p90/conf/valid ratio/sample count). No active code imports or
  constructs it.
- `config.py` contains disabled integration flags and non-secret connection
  metadata. It contains no MQTT password or Influx token.

## How to Wire in DS8
Treat activation as a product change, not a configuration-only operation:

1. Define and test the typed publisher adapter in the shared product boundary.
2. Use `DepthDiagnosticsPublisher.from_settings(...)` so disabled sinks read no
   credentials and enabled sinks fail closed.
3. Bind occupancy through
   `noesis.telemetry.publishers.bind_occupancy_publisher(...)` only after an
   active implementation exists.
4. Apply equivalent product behavior to DS8 and DS9 without adding appsink/CPU
   branches.
5. Validate credentials and client startup before runtime activation; do not
   continue with an enabled sink silently disabled.

The owner-only file contract and rotation procedure are documented in
`docs/integrations/occupancy_mqtt_influx.md`.

## Recommended Contracts
- Occupancy topics/measurements: reuse the schema from `docs/Occupancy_Publishing.md` (room slugs + retained MQTT scalars; nanosecond Influx points on change).
- Depth summaries: use fields (`median`, `p10`, `p90`, `conf_mean`, `valid_ratio`, `sample_count`) keyed by camera/room.

## Notes
- DS8 tracking telemetry already carries occupancy counts (`stats.payload.cameras[*].tracking.occupancy`). Use WebSocket when possible to avoid duplicate plumbing.
- Keep non-secret integration behavior in version-controlled config. Deployment
  tooling may supply only the private credential-file paths through the
  documented `*_FILE` environment variables; raw secret environment variables
  are rejected.
