# Occupancy Publishing (DS8)
_Status: validated against code on 2026-02-02._

DS8 publishes occupancy primarily over WebSocket tracking telemetry. The earlier MQTT/Influx pipeline is archived under `docs/history/`.

## What DS8 Emits by Default
- Source: `_AnalyticsTelemetryProcessor` (`noesis/pipelines/hooks.py`).
- Per-frame occupancy counts are derived from nvdsanalytics ROI status (or a per-camera fallback zone) and published on the **tracking** stream as `tracking.tracks[].zone` plus a per-camera `tracking.occupancy` map in `stats`.
- No MQTT/Influx publishing is wired by default; `bind_occupancy_publisher(pipeline, None)` is called in `noesis/ds8_runtime.py`.

## Extending to MQTT/Influx
- The pipeline exposes an `occupancy_publisher` slot via `noesis.telemetry.publishers.bind_occupancy_publisher(pipeline, publisher_instance)`.
- The publisher instance must implement `publish_state(room_id: str, occupied: bool, count: int, ts_ns: int)`.
- `_AnalyticsTelemetryProcessor._publish_occupancy` will call `publish_state` for each observed zone and will send an explicit vacate (occupied=False, count=0) when a previously seen zone disappears on the current frame.
- There is **no grace window** in DS8; vacates are immediate when zone counts drop to zero.

## Keys and Zones
- Zone IDs come from `analytics.stages[stage].streams[stream_id].roiStatus` in `config/nvdsanalytics.yaml`.
- When analytics does not provide a zone, DS8 falls back to a synthetic zone derived from the camera name.

## Related Telemetry
- WebSocket tracking schema: `docs/DS8_api_contracts_ws.md` (section: Tracking Telemetry)
- Stats payload occupancy snapshot: `docs/Telemetry_Schema.md`
