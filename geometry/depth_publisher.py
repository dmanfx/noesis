"""Publishing utilities for MapAnything depth diagnostics."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - optional dependency
    mqtt = None  # type: ignore

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
except Exception:  # pragma: no cover - optional dependency
    InfluxDBClient = None  # type: ignore
    Point = None  # type: ignore
    WritePrecision = None  # type: ignore
    ASYNCHRONOUS = None  # type: ignore

from utils.rate_limited_logger import RateLimitedLogger
from geometry.depth_source import DepthResult


@dataclass(frozen=True)
class DiagnosticsConfig:
    base_topic: str
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password: str
    mqtt_qos: int
    mqtt_retain: bool
    influx_url: str
    influx_org: str
    influx_token: str
    influx_bucket: str


class DepthDiagnosticsPublisher:
    """Publishes depth summaries to MQTT and InfluxDB."""

    def __init__(self, cfg: DiagnosticsConfig, logger: Optional[logging.Logger] = None) -> None:
        self.cfg = cfg
        self.logger = RateLimitedLogger(logger or logging.getLogger("DepthDiagnosticsPublisher"), rate_limit_seconds=5.0)
        self.client: Optional["mqtt.Client"] = None
        self.influx: Optional["InfluxDBClient"] = None
        self.write_api = None
        self._connect_mqtt()
        self._connect_influx()
        self.logger.info(
            "Depth diagnostics publisher ready (MQTT=%s, Influx=%s)",
            "on" if self.client else "off",
            "on" if self.write_api else "off",
        )

    def _connect_mqtt(self) -> None:
        if mqtt is None:
            self.logger.warning("paho-mqtt not installed; MQTT depth diagnostics disabled")
            return
        try:
            client = mqtt.Client(client_id="noesis-mapanything-pub", clean_session=True)
            if self.cfg.mqtt_username:
                client.username_pw_set(self.cfg.mqtt_username, self.cfg.mqtt_password)
            client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, 60)
            client.loop_start()
            self.client = client
        except Exception as exc:  # pragma: no cover - connection failure
            self.logger.error(f"Failed to initialise MQTT client: {exc}")
            self.client = None

    def _connect_influx(self) -> None:
        if InfluxDBClient is None:
            self.logger.warning("influxdb-client not installed; Influx depth diagnostics disabled")
            return
        try:
            self.influx = InfluxDBClient(url=self.cfg.influx_url, token=self.cfg.influx_token, org=self.cfg.influx_org)
            self.write_api = self.influx.write_api(write_options=ASYNCHRONOUS)
        except Exception as exc:  # pragma: no cover - connection failure
            self.logger.error(f"Failed to initialise InfluxDB client: {exc}")
            self.influx = None
            self.write_api = None

    def publish_depth_summary(self, result: DepthResult, room_id: str) -> None:
        payload = {
            "ts_us": int(result.ts_us),
            "median": result.summary.median,
            "p10": result.summary.p10,
            "p90": result.summary.p90,
            "conf_mean": result.summary.conf_mean,
            "valid_ratio": result.summary.valid_ratio,
            "sample_count": result.summary.sample_count,
        }
        topic = f"{self.cfg.base_topic}/geometry/{room_id}/{result.camera_id}/depth_summary"
        if self.client is not None:
            try:
                self.client.publish(topic, payload=json.dumps(payload), qos=self.cfg.mqtt_qos, retain=self.cfg.mqtt_retain)
            except Exception as exc:  # pragma: no cover
                self.logger.error(f"Failed to publish depth summary to MQTT: {exc}")
        if self.write_api is not None and Point is not None and WritePrecision is not None:
            try:
                point = (
                    Point("mde.depth.summary")
                    .tag("camera", result.camera_id)
                    .tag("room", room_id)
                    .field("median", float(result.summary.median))
                    .field("p10", float(result.summary.p10))
                    .field("p90", float(result.summary.p90))
                    .field("conf_mean", float(result.summary.conf_mean))
                    .field("valid_ratio", float(result.summary.valid_ratio))
                    .field("sample_count", int(result.summary.sample_count))
                    .time(result.ts_us, WritePrecision.US)
                )
                self.write_api.write(bucket=self.cfg.influx_bucket, record=point)
            except Exception as exc:  # pragma: no cover
                self.logger.error(f"Failed to write depth summary to InfluxDB: {exc}")

    def publish_scale_metrics(self, camera_id: str, room_id: str, scale: float, pose_error: Optional[float] = None) -> None:
        if self.write_api is not None and Point is not None and WritePrecision is not None:
            try:
                point = (
                    Point("mde.scale")
                    .tag("camera", camera_id)
                    .tag("room", room_id)
                    .field("scale", float(scale))
                    .time(time.time_ns(), WritePrecision.NS)
                )
                if pose_error is not None:
                    point = point.field("pose_error", float(pose_error))
                self.write_api.write(bucket=self.cfg.influx_bucket, record=point)
            except Exception as exc:  # pragma: no cover
                self.logger.error(f"Failed to write scale metrics to InfluxDB: {exc}")

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        if self.influx is not None:
            try:
                self.influx.close()
            except Exception:
                pass


__all__ = ["DepthDiagnosticsPublisher", "DiagnosticsConfig"]
