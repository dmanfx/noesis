import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

try:
    import paho.mqtt.client as mqtt
except Exception as e:  # pragma: no cover
    mqtt = None  # type: ignore

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
except Exception as e:  # pragma: no cover
    InfluxDBClient = None  # type: ignore
    Point = None  # type: ignore
    WritePrecision = None  # type: ignore
    ASYNCHRONOUS = None  # type: ignore


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_ns(ts_ns: Optional[int]) -> str:
    if not ts_ns:
        return _iso_now()
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()


def _slugify_room(room_id: str) -> str:
    # Minimal, stable slug for MQTT topics
    s = str(room_id).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in [' ', '_', '-', '/']:
            out.append('-')
        else:
            # drop other punctuation
            out.append('-')
    # collapse multiple '-'
    slug = []
    prev_dash = False
    for ch in out:
        if ch == '-' and prev_dash:
            continue
        prev_dash = ch == '-'
        slug.append(ch)
    slug_s = ''.join(slug).strip('-')
    return slug_s or 'unknown'


@dataclass
class OccupancyConfig:
    enabled: bool
    heartbeat_sec: int
    base_topic: str
    status_topic: str
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password: str
    mqtt_qos: int
    mqtt_retain: bool
    influx_url: str
    influx_org: str
    influx_token: str
    influx_bucket_raw: str


class OccupancyPublisher:
    """Publishes occupancy state changes to MQTT and InfluxDB.

    - MQTT publishes retained scalar topics and a JSON envelope per room.
    - InfluxDB writes occur only on state changes (not on heartbeats).
    - A background heartbeat refreshes retained MQTT topics.
    """

    def __init__(self, cfg: OccupancyConfig, logger: Optional[logging.Logger] = None) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger("OccupancyPublisher")
        self._last: Dict[str, Tuple[int, int]] = {}
        self._hb_stop = threading.Event()

        self.client: Optional["mqtt.Client"] = None
        self.ic: Optional["InfluxDBClient"] = None
        self.write_api = None

        # Initialize MQTT
        if mqtt is None:
            self.logger.error("paho-mqtt is not installed; MQTT publishing disabled")
        else:
            try:
                self.client = mqtt.Client(client_id="noesis-occupancy-pub", clean_session=True)
                if self.cfg.mqtt_username:
                    self.client.username_pw_set(self.cfg.mqtt_username, self.cfg.mqtt_password)
                # LWT
                self.client.will_set(
                    self.cfg.status_topic, payload="offline", qos=self.cfg.mqtt_qos, retain=True
                )
                self.client.on_connect = self._on_connect
                self.client.on_disconnect = self._on_disconnect
                self.client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, 60)
                self.client.loop_start()
            except Exception as e:
                self.logger.error(f"Failed to init/connect MQTT client: {e}")
                self.client = None

        # Initialize InfluxDB client (optional)
        if InfluxDBClient is None:
            self.logger.error("influxdb-client is not installed; Influx writes disabled")
        else:
            try:
                self.ic = InfluxDBClient(url=self.cfg.influx_url, token=self.cfg.influx_token, org=self.cfg.influx_org)
                # Use async write to keep the DS thread snappy
                self.write_api = self.ic.write_api(write_options=ASYNCHRONOUS)
            except Exception as e:
                self.logger.error(f"Failed to init InfluxDB client: {e}")
                self.ic = None
                self.write_api = None

        # Start heartbeat thread
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="OccupancyHeartbeat", daemon=True)
        self._hb_thread.start()

        self.logger.info(
            f"OccupancyPublisher ready (MQTT={'on' if self.client else 'off'}, Influx={'on' if self.write_api else 'off'})"
        )

    # MQTT callbacks
    def _on_connect(self, client, userdata, flags, rc):  # pragma: no cover
        try:
            if rc == 0:
                client.publish(self.cfg.status_topic, payload="online", qos=self.cfg.mqtt_qos, retain=True)
                self.logger.info("MQTT connected; status=online published")
            else:
                self.logger.error(f"MQTT connection failed with rc={rc}")
        except Exception:
            pass

    def _on_disconnect(self, client, userdata, rc):  # pragma: no cover
        # loop_start handles reconnects; nothing else needed
        if rc != 0:
            self.logger.warning(f"MQTT disconnected unexpectedly (rc={rc})")

    def close(self):
        try:
            self._hb_stop.set()
            if self._hb_thread.is_alive():
                self._hb_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self.client is not None:
                try:
                    self.client.publish(self.cfg.status_topic, payload="offline", qos=self.cfg.mqtt_qos, retain=True)
                except Exception:
                    pass
                self.client.loop_stop()
                self.client.disconnect()
        except Exception:
            pass
        try:
            if self.ic is not None:
                self.ic.close()
        except Exception:
            pass

    def publish_state(self, room_id: str, occupied: bool, count: int, ts_ns: Optional[int] = None):
        """Publish one room state. Writes to Influx only on change. Always retained MQTT publish on change.

        This is safe to call on each frame; dedupe avoids redundant work.
        """
        try:
            slug = _slugify_room(room_id)
            occ_i = 1 if bool(occupied) else 0
            cnt = int(count)
            last = self._last.get(slug)
            is_change = (last is None) or (last != (occ_i, cnt))
            ts_ns_use = ts_ns or int(time.time_ns())
            ts_iso = _iso_from_ns(ts_ns_use)

            if is_change:
                # MQTT retained scalars + JSON
                base = f"{self.cfg.base_topic}/{slug}"
                self._mqtt_publish(f"{base}/occupied", str(occ_i))
                self._mqtt_publish(f"{base}/count", str(cnt))
                env = {"ts": ts_iso, "occupied": bool(occ_i), "count": cnt}
                self._mqtt_publish(base, json.dumps(env))

                # InfluxDB write on change
                if self.write_api and Point is not None and WritePrecision is not None:
                    try:
                        p = (
                            Point("occupancy")
                            .tag("room_id", slug)
                            .field("occupied", occ_i)
                            .field("count", cnt)
                            .time(ts_ns_use, WritePrecision.NS)
                        )
                        self.write_api.write(bucket=self.cfg.influx_bucket_raw, org=self.cfg.influx_org, record=p)
                    except Exception as e:
                        self.logger.error(f"Influx write error for room {slug}: {e}")

                self._last[slug] = (occ_i, cnt)
        except Exception as e:
            self.logger.error(f"publish_state error for room={room_id}: {e}")

    def _mqtt_publish(self, topic: str, payload: str, tries: int = 3):
        if not self.client:
            return
        delay = 0.5
        for attempt in range(tries):
            try:
                # fire-and-forget; paho handles queueing
                self.client.publish(topic, payload=payload, qos=self.cfg.mqtt_qos, retain=self.cfg.mqtt_retain)
                return
            except Exception as e:
                if attempt == tries - 1:
                    self.logger.error(f"MQTT publish failed for {topic}: {e}")
                time.sleep(delay)
                delay *= 2

    def _heartbeat_loop(self):
        # Periodically refresh retained MQTT state for all known rooms
        interval = max(5, int(self.cfg.heartbeat_sec or 60))
        while not self._hb_stop.is_set():
            try:
                if self.client and self._last:
                    for room, (occ_i, cnt) in list(self._last.items()):
                        base = f"{self.cfg.base_topic}/{room}"
                        env = {"ts": _iso_now(), "occupied": bool(occ_i), "count": int(cnt), "event": "heartbeat"}
                        self._mqtt_publish(f"{base}/occupied", str(occ_i))
                        self._mqtt_publish(f"{base}/count", str(cnt))
                        self._mqtt_publish(base, json.dumps(env))
                # No Influx write on heartbeat
            except Exception:
                pass
            self._hb_stop.wait(interval)

