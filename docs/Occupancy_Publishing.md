# Occupancy publication

Status: canonical DS9.1 behavior, 2026-08-15.

Occupancy is published over authenticated tracking telemetry. MQTT/Influx is
not active.

## Source and payload

`DS9/noesis/pipelines/hooks.py` resolves post-tracker `nvdsanalytics`
overcrowding membership. It publishes:

- `tracking.tracks[].zone` for each exact authoritative object membership;
- per-camera `tracking.occupancy` counts;
- transitions and diagnostics in the tracking/stats contracts.

The configured room IDs are `LivingRoom`, `Kitchen`, and `FamilyRoom`. The
mutable ROI editor controls the pre-tracker exclusion stage only; it cannot
rewrite these post-tracker room declarations.

Membership resolution is fail closed. One unique nonempty label is required.
Repeated identical copies are harmless; malformed or multiple distinct labels
produce no authoritative zone. Camera-name fallback may support diagnostics but
cannot populate a canonical world entity's `room_id`.

When an optional occupancy publisher is eventually bound, it must implement:

```text
publish_state(room_id: str, occupied: bool, count: int, ts_ns: int)
```

A zone dropping to zero emits an immediate vacate. See
`api_contracts_ws.md` for the wire contract and
`integrations/occupancy_mqtt_influx.md` for the dormant external integration
design.
