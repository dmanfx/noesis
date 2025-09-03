# Integrations Playbook: InfluxDB + HomeSeer via MQTT

This guide documents how the Noesis pipeline integrates with external systems, using our occupancy → MQTT + InfluxDB + HomeSeer setup as a concrete example. It also provides a repeatable pattern to add future integrations and new stats.

## Scope

- What we built: Occupancy state per ROI/zone published to MQTT (for automations) and InfluxDB (for history/analytics), consumed by HomeSeer via mcsMQTT.
- Where it lives: Publisher in Python with config in `config.py`, wired into the DeepStream processor at runtime.
- How to extend: Follow the same publisher pattern to add new topics/measurements or new sinks.

## Architecture

- Data source: DeepStream probe computes per-frame analytics (occupancy counts, transitions, tracks) inside `deepstream_video_pipeline.py`.
- Publisher: `occupancy_publisher.py` handles MQTT retained topics and Influx write-on-change, with background heartbeats for retained MQTT refresh.
- Wiring: `main.py` builds `OccupancyConfig` from `config.integrations`, constructs `OccupancyPublisher`, and attaches it to the processor. The probe calls `publish_state()` when counts change or vacate.
- Isolation: Publishing errors never impact the real-time pipeline; all calls are wrapped in try/except and async where applicable.

## Program Changes (Noesis)

1) Configuration (new section)
- Added `AppConfig.IntegrationsSettings` in `config.py:293` with keys:
  - Enable + cadence: `ENABLE_OCCUPANCY_PUBLISH`, `HEARTBEAT_SEC`
  - MQTT: `BASE_TOPIC`, `STATUS_TOPIC`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_QOS`, `MQTT_RETAIN`
  - Influx: `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_TOKEN`, `INFLUX_BUCKET_RAW`

2) Publisher component
- File: `occupancy_publisher.py:76`
- Behavior:
  - On change → publish retained MQTT scalars + JSON envelope, write an Influx point with nanosecond precision.
  - On heartbeat → republish retained MQTT only (no Influx) to keep state visible to late subscribers.
  - Slugifies ROI names to stable `<room>` ids for topics/tags.
  - Deduplicates by `(occupied,count)` to avoid redundant writes.
  - Gracefully tolerates missing libs (paho-mqtt/influxdb-client) and connection issues.

3) Wiring into runtime
- File: `main.py:418`
- Builds `OccupancyConfig` from `config.integrations` and attaches the created `OccupancyPublisher` to the running DeepStream processor (`self.multi_stream_processor.occupancy_publisher = pub`).

4) Emission site in pipeline
- File: `deepstream_video_pipeline.py:993`
- Each frame:
  - Computes `occupancy = { zone: count }` from object/frame analytics.
  - Calls `publish_state(room_id=zone, occupied=(count>0), count=count, ts_ns=...)` for seen zones.
  - Emits explicit vacates (`occupied=false, count=0`) for zones previously seen but absent this frame.
  - Never raises if publishing fails.

## External Setup

### MQTT Broker (Mosquitto)

- Install: `sudo apt install -y mosquitto mosquitto-clients`
- Configure `/etc/mosquitto/conf.d/noesis.conf`:
  listener 1883
  allow_anonymous false
  password_file /etc/mosquitto/passwd
  persistence true
  persistence_location /var/lib/mosquitto/
  autosave_interval 30
- Create credentials and restart:
  sudo mosquitto_passwd -c /etc/mosquitto/passwd noesis
  sudo systemctl restart mosquitto
- Verify retained messages:
  mosquitto_sub -h <host> -u noesis -P '<pass>' -t 'noesis/occupancy/#' -v

### InfluxDB v2

- Buckets:
  influx bucket create -n noesis_raw -r 90d
  influx bucket create -n noesis_1m -r 730d
- Optional 1-minute rollup task:
  influx task create -n "noesis_occupancy_1m" -d '
  option task = {name: "noesis_occupancy_1m", every: 1m}
  from(bucket: "noesis_raw")
    |> range(start: -task.every)
    |> filter(fn: (r) => r._measurement == "occupancy")
    |> aggregateWindow(every: 1m, fn: last, createEmpty: false)
    |> to(bucket: "noesis_1m")
  '
- Configure token/org/url in `config.integrations`.
- Verify recent point:
  influx query 'from(bucket:"noesis_raw") |> range(start:-15m) |> filter(fn:(r)=> r._measurement=="occupancy") |> last()'

### HomeSeer (mcsMQTT)

- Broker connection: point to Mosquitto with `noesis` user/pass.
- Subscribe: `noesis/#`.
- Device auto-creation: first retained publishes create devices for each room:
  - `<room>/occupied` (discrete 0/1 → Vacant/Occupied)
  - `<room>/count` (numeric)
- Recommended: exclude these devices from HS→Influx export to avoid duplicate points when Noesis writes directly to Influx.
- Automations: trigger on changes to `<room>/occupied` or thresholds on `<room>/count`.

## Operational Checklists

- Service health: Noesis logs indicate MQTT/Influx availability at startup; `noesis/status` retained topic shows online/offline via LWT.
- End-to-end:
  - Move in a monitored ROI → watch MQTT topics update and a new Influx point appear.
  - Clear ROI → observe a vacate event (count=0) and device update in HomeSeer.

## Pattern: Adding a New Integration or Stat

1) Define the schema
- MQTT topics: prefer `base/<entity>/<metric>` for scalars and `base/<entity>` for JSON envelopes.
- Influx: measurement name, tags (identify entity), fields (numeric/boolean), and write precision.
- Retention and rollups: pick buckets and optional tasks.

2) Implement a Publisher (if needed)
- Start with `occupancy_publisher.py` as a template:
  - Connection setup, LWT, retries, dedupe, heartbeat, graceful shutdown.
  - Keep publishing async/non-blocking where possible.

3) Wire it into runtime
- Build a `Config` dataclass under `AppConfig.integrations` with required knobs.
- Initialize publisher in `main.py` and attach to the processor or relevant component.

4) Emit data from the right place
- Frame/analytics-derived stats: `deepstream_video_pipeline.py` (inside the probe that already computes analytics).
- System/process stats: `main.py` (e.g., uptime, app metrics) or a dedicated metrics loop.
- Web telemetry: follow `docs/reference/Telemetry_Schema.md` if you surface data to the front-end.

5) Verify and observe
- CLI tools: `mosquitto_sub`, `influx query`.
- Logs: check for connection errors or missing libraries.

## Pattern: Adding More Occupancy/Analytics Fields

- New MQTT fields: add scalars and include in the JSON envelope. Prefer simple numeric/boolean payloads for device mapping in HomeSeer.
- New Influx fields: add fields to the point with careful consideration of cardinality (keep tags low-cardinality, fields for values).
- Example extension: add `enter_events` and `exit_events` per zone as counters; publish to `noesis/occupancy/<room>/enters` and `.../exits`, and write to Influx under the same measurement.

## Security & Ops Considerations

- Secrets: inject broker passwords and Influx tokens via environment or protected config; avoid committing real tokens.
- QoS/retain: QoS 1 and retained topics are used to guarantee state availability for late subscribers.
- Failure isolation: all publishing is best-effort and never blocks the video pipeline.

## References

- Occupancy publisher: `occupancy_publisher.py:76`
- Wiring: `main.py:418`
- Emission site: `deepstream_video_pipeline.py:993`
- Topic and schema summary: `docs/reference/Occupancy_Publishing.md`
- Setup walkthrough: `docs/integrations/occupancy_mqtt_influx.md`
