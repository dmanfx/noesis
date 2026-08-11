# Occupancy Publishing (DS8)
_Status: validated against code and analytics configuration on 2026-07-24._

DS8 publishes occupancy primarily over WebSocket tracking telemetry. The earlier MQTT/Influx pipeline is archived under `docs/history/`.

## What DS8 Emits by Default
- Source: `_AnalyticsTelemetryProcessor` (`noesis/pipelines/hooks.py`).
- The existing post-tracker overcrowding rules emit per-object `ocStatus`
  polygon-membership labels as well as frame-level counts and threshold state.
  Noesis publishes the resolved per-object label on the **tracking** stream as
  `tracking.tracks[].zone`; per-camera counts are also exposed through
  `tracking.occupancy`.
- A camera-name fallback remains available for occupancy, dwell, and operator
  diagnostics when no authoritative analytics membership exists. It is
  explicitly non-authoritative and cannot populate a canonical world entity's
  `room_id`.
- No MQTT/Influx publishing is wired by default; `bind_occupancy_publisher(pipeline, None)` is called in `noesis/ds8_runtime.py`.

## Extending to MQTT/Influx
- The pipeline exposes an `occupancy_publisher` slot via `noesis.telemetry.publishers.bind_occupancy_publisher(pipeline, publisher_instance)`.
- The publisher instance must implement `publish_state(room_id: str, occupied: bool, count: int, ts_ns: int)`.
- `_AnalyticsTelemetryProcessor._publish_occupancy` will call `publish_state` for each observed zone and will send an explicit vacate (occupied=False, count=0) when a previously seen zone disappears on the current frame.
- There is **no grace window** in DS8; vacates are immediate when zone counts drop to zero.
- No active publisher implementation is present. Before adding one, follow the
  owner-only credential and fail-closed activation contract in
  `docs/integrations/occupancy_mqtt_influx.md` and preserve DS8/DS9 parity.

## Keys and Zones
- Authoritative room IDs are declared under
  `analytics.stages.post.streams[stream_id].overcrowding.roi.id` in
  `config/nvdsanalytics.yaml` and rendered as `roi-<room-id>` in the matching
  `[overcrowding-stream-<stream-id>]` section of
  `config/config_nvdsanalytics_post.ini`.
- The active IDs are `LivingRoom`, `Kitchen`, and `FamilyRoom`. These are the
  single room designation; no duplicate post-stage ROI-filtering taxonomy is
  maintained.
- Resolution is exact and fail closed. `ocStatus` is primary. `roiStatus` is
  accepted only as compatibility evidence when no nonempty overcrowding
  membership exists. Exactly one unique, nonempty, unpadded label of at most
  160 characters is required; repeated copies of the same label are harmless,
  while malformed or multiple distinct labels produce no authoritative zone.
  IDs are never trimmed, case-normalized, truncated, or selected first-wins.
- DS8 is canonical; DS9 carries the same post-stage declaration and static INI.
- The analytics ROI editor and hot-reload API own only the pre-tracker
  `exclude` stage. The authoritative post-stage room designation is immutable
  release configuration, not mutable-state data.

## Related Telemetry
- WebSocket tracking schema: `docs/DS8_api_contracts_ws.md` (section: Tracking Telemetry)
- Stats payload occupancy snapshot: `docs/Telemetry_Schema.md`
