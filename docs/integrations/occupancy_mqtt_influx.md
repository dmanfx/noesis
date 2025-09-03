Noesis Occupancy → MQTT + InfluxDB + HomeSeer

Overview
- Publishes retained MQTT topics for room occupancy for HomeSeer (mcsMQTT) automations.
- Writes the same state changes to InfluxDB v2 for history (bucket `noesis_raw`).
- Heartbeat refreshes retained MQTT state; Influx writes only on change.

See Also
- For a step-by-step playbook and a reusable pattern to add future integrations or stats, refer to `docs/reference/Integrations_Playbook.md`.

MQTT Topics
- Base: `noesis/occupancy/<room>` where `<room>` is a slug from ROI/zone name.
- `noesis/occupancy/<room>/occupied`: "0" or "1" (retained, QoS 1)
- `noesis/occupancy/<room>/count`: "0..N" (retained, QoS 1)
- `noesis/occupancy/<room>`: JSON `{ts, occupied, count}` (retained, QoS 1)
- Status: `noesis/status`: "online"/"offline" (retained)

InfluxDB
- Measurement: `occupancy`
- Tags: `room_id=<room>`
- Fields: `occupied` (int), `count` (int)
- Writes to bucket: `noesis_raw`

Config (config.py → AppConfig.integrations)
- `ENABLE_OCCUPANCY_PUBLISH` (bool)
- `HEARTBEAT_SEC` (int)
- MQTT: `BASE_TOPIC`, `STATUS_TOPIC`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_QOS`, `MQTT_RETAIN`
- Influx: `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_TOKEN`, `INFLUX_BUCKET_RAW`

Setup Steps
1) Mosquitto
- `sudo apt install -y mosquitto mosquitto-clients`
- `/etc/mosquitto/conf.d/noesis.conf`:
  listener 1883
  allow_anonymous false
  password_file /etc/mosquitto/passwd
  persistence true
  persistence_location /var/lib/mosquitto/
  autosave_interval 30
- `sudo mosquitto_passwd -c /etc/mosquitto/passwd noesis && sudo systemctl restart mosquitto`

2) InfluxDB v2
- Create buckets and a 1-minute rollup task (optional):
  influx bucket create -n noesis_raw -r 90d || true
  influx bucket create -n noesis_1m -r 730d || true
  influx task create -n "noesis_occupancy_1m" -d '
  option task = {name: "noesis_occupancy_1m", every: 1m}
  from(bucket: "noesis_raw")
    |> range(start: -task.every)
    |> filter(fn: (r) => r._measurement == "occupancy")
    |> aggregateWindow(every: 1m, fn: last, createEmpty: false)
    |> to(bucket: "noesis_1m")
  ' || true

3) HomeSeer (mcsMQTT)
- Broker: Mosquitto with `noesis` user/pass
- Subscribe: `noesis/#`
- Devices auto-create on first retained publish:
  - occupied: status pairs 0=Vacant, 1=Occupied
  - count: numeric
- Exclude these from HS→Influx export to avoid duplicate points.

Verification
- MQTT: `mosquitto_sub -h 127.0.0.1 -t 'noesis/occupancy/#' -u noesis -P 'PASS' -v`
- Influx: `influx query 'from(bucket:"noesis_raw") |> range(start:-15m) |> filter(fn:(r)=> r._measurement=="occupancy") |> last()'`

Notes
- Occupancy is emitted from the DeepStream analytics probe; ROI/zone names are sluggified to form `<room>`.
- When a zone disappears, a "vacant" (count=0) event is published to ensure state transitions are captured.
