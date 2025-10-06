# Occupancy Publishing (MQTT + InfluxDB + HomeSeer)

This document describes how Noesis publishes room/zone occupancy and MapAnything depth summaries to MQTT and InfluxDB, and how those feeds connect to downstream automations and dashboards.

## Overview

- Publishes retained MQTT topics per room/zone for automation.
- Publishes MapAnything depth summaries for diagnostics dashboards.
- Writes occupancy and depth metrics to InfluxDB for history/analytics.
- Heartbeat periodically refreshes retained topics without writing to Influx.

Dependencies
- Python: `paho-mqtt`, `influxdb-client` (see `requirements.txt`)

## Data Flow

- DeepStream probe computes per-frame occupancy per ROI/zone and detects vacates.
- On each frame, deltas are sent to a shared `OccupancyPublisher`:
  - Initialization and wiring: `main.py` attaches the publisher to the running DeepStream processor.
  - Emission: `deepstream_video_pipeline.py` publishes counts and explicit vacates.

## MQTT Topics

- Occupancy base: `noesis/occupancy/<room>` where `<room>` is a stable slug of the ROI/zone name.
  - Scalars (retained, QoS 1):
    - `noesis/occupancy/<room>/occupied` → "0" or "1"
    - `noesis/occupancy/<room>/count` → "0..N"
  - Envelope (retained, QoS 1): `{ ts: ISO8601, occupied: bool, count: int }`
- Depth summaries: `noesis/geometry/<room>/<camera>/depth_summary` → `{ ts_us, median, p10, p90, conf_mean, valid_ratio, sample_count }`
- Status: `noesis/status` → "online"/"offline" (retained) via MQTT LWT.

Notes
- Retained messages ensure subscribers see the latest state on connect.
- Heartbeat republishes retained MQTT values on an interval; no Influx writes on heartbeat.

## InfluxDB Series

- `occupancy` measurement
  - Tags: `room_id=<room>`
  - Fields: `occupied` (int 0/1), `count` (int)
  - Precision: nanoseconds
- `mde.depth.summary` measurement
  - Tags: `camera`, `room`
  - Fields: `median`, `p10`, `p90`, `conf_mean`, `valid_ratio`, `sample_count`
  - Precision: microseconds
- `mde.scale` measurement
  - Tags: `camera`, `room`
  - Fields: `scale`, `pose_error`
  - Precision: nanoseconds
- Bucket: `integrations.INFLUX_BUCKET_RAW` (default `noesis_raw`)
- Writes occur only on changes to reduce volume.

## Configuration (config.py → AppConfig.integrations)

- Enable: `ENABLE_OCCUPANCY_PUBLISH` (bool)
- Heartbeat: `HEARTBEAT_SEC` (int)
- MQTT: `BASE_TOPIC`, `STATUS_TOPIC`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_QOS`, `MQTT_RETAIN`
- Influx: `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_TOKEN`, `INFLUX_BUCKET_RAW`

## HomeSeer (mcsMQTT) Notes

- Point mcsMQTT at your broker and subscribe `noesis/#`.
- Devices auto-create on first retained publishes:
  - `<room>/occupied` → discrete status (0=Vacant, 1=Occupied)
  - `<room>/count` → numeric
- Exclude these topics from HS→Influx export to avoid duplicate points if HS is also writing to Influx.

## Code References

- Publisher implementation: `occupancy_publisher.py:76`
- Wiring in application: `main.py:418`
- DeepStream emission site (per-frame deltas + vacates): `deepstream_video_pipeline.py:993`

## Verification

- MQTT: `mosquitto_sub -h <host> -t 'noesis/#' -u <user> -P <pass> -v`
- Influx (example):
  - `from(bucket:"noesis_raw") |> range(start:-15m) |> filter(fn:(r)=> r._measurement=="occupancy") |> last()`

## Troubleshooting

- No MQTT publishes: verify broker host/port and credentials in `integrations` config, and that `paho-mqtt` is installed.
- No Influx writes: confirm `INFLUX_URL/ORG/TOKEN/BCKET` and that `influxdb-client` is installed; remember writes only occur on state changes.
- Missing vacate events: ensure ROI names are stable; publisher slugs names to form the `<room>` key.

## See Also

- Setup walkthrough and examples: `docs/integrations/occupancy_mqtt_influx.md`
