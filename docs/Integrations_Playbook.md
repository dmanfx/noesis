# Integrations playbook

Status: native DS9.1 application, 2026-08-15.

No MQTT or Influx publisher is enabled in the canonical runtime. Occupancy and
depth diagnostics are available through the authenticated Noesis contracts and
Menon gateway.

## Existing integration points

- `noesis.telemetry.publishers.bind_occupancy_publisher(...)` defines an
  optional adapter slot; the canonical runtime binds no external publisher.
- Tracking WebSocket payloads carry per-track zone labels and per-camera
  occupancy counts.
- `geometry/depth_publisher.py` is dormant and not constructed by DS9.1.
- Public config contains only non-secret connection metadata. Credentials must
  come from validated owner-only files.

## Adding an external publisher

Treat activation as a product/contract change:

1. Define the typed adapter and failure policy.
2. Keep frame processing GPU-first; publish from metadata, not a new frame
   appsink.
3. Load credentials only when enabled and fail closed if they are invalid.
4. Test the producer, adapter, broker/client contract, and one direct consumer.
5. Measure only the affected serialization/network boundary.

Do not silently continue with a configured sink disabled. Do not add a second
occupancy taxonomy; use canonical room IDs from post-tracker analytics.

The credential contract and proposed MQTT/Influx shapes are in
`integrations/occupancy_mqtt_influx.md`.
