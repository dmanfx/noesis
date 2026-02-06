# Integrations Playbook (DS8)
_Status: validated against code on 2026-02-02._

DS8 does not ship MQTT/Influx publishing in the default runtime. The legacy DS7 playbook has been archived to `docs/history/ds7/Integrations_Playbook_ds7.md`. This page explains how to hook the existing publishers into DS8 if you need external automations.

## What Exists
- `occupancy_publisher.py`: MQTT + Influx publisher with `publish_state(room_id, occupied, count, ts_ns)`.
- `geometry/depth_publisher.py`: depth summary publisher (median/p10/p90/conf/valid_ratio/sample_count) for MapAnything snapshots.
- Config helpers: `config.py` `IntegrationsSettings` (see DS7 playbook for full key list).

## How to Wire in DS8
1) Instantiate publishers alongside the DS8 runtime bootstrap (e.g., wrap `DepthStorageManager` and `occupancy_publisher` creation before calling `ds8_pipeline.build_pipeline`).
2) Bind to the pipeline:
   - Occupancy: `bind_occupancy_publisher(pipeline, publisher_instance)` (`noesis.telemetry.publishers`).
   - Depth summaries: subscribe to `DepthStorageManager.store(...)` or poll `DepthStorageManager.load_latest_depth(...)` and call `DepthDiagnosticsPublisher` on change.
3) Keep MQTT/Influx failures isolated (publishers already catch exceptions); do **not** introduce appsink/CPU branches in DS8 to feed integrations.

## Recommended Contracts
- Occupancy topics/measurements: reuse the schema from `docs/Occupancy_Publishing.md` (room slugs + retained MQTT scalars; nanosecond Influx points on change).
- Depth summaries: mirror DS7 fields (`median`, `p10`, `p90`, `conf_mean`, `valid_ratio`, `sample_count`) keyed by camera/room.

## Notes
- DS8 tracking telemetry already carries occupancy counts (`stats.payload.cameras[*].tracking.occupancy`). Use WebSocket when possible to avoid duplicate plumbing.
- Keep integration config in version-controlled files (no ad-hoc env overrides) so DS8/DS7 parity tests remain reproducible.

