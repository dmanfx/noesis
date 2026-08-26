"""Publishing utilities for MapAnything depth diagnostics."""
from __future__ import annotations

import json
import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

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


_MAX_SECRET_BYTES = 16 * 1024


class DiagnosticsConfigurationError(RuntimeError):
    """Raised when an explicitly enabled diagnostics sink is not secure/usable."""


def load_private_secret(
    path: str | Path,
    *,
    purpose: str,
    minimum_length: int = 16,
) -> str:
    """Read one UTF-8 secret from an existing owner-only regular file.

    The function deliberately never creates files or changes permissions.  It uses
    ``O_NOFOLLOW`` plus an inode comparison so a configured symlink or path-swap is
    rejected instead of repaired or followed.
    """

    secret_path = Path(path).expanduser()
    try:
        before = secret_path.lstat()
    except FileNotFoundError as exc:
        raise DiagnosticsConfigurationError(f"{purpose} secret file is missing") from exc
    except OSError as exc:
        raise DiagnosticsConfigurationError(f"{purpose} secret file cannot be inspected") from exc
    if stat.S_ISLNK(before.st_mode):
        raise DiagnosticsConfigurationError(f"{purpose} secret file must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise DiagnosticsConfigurationError(f"{purpose} secret file must be a regular file")
    if before.st_uid != os.geteuid():
        raise DiagnosticsConfigurationError(f"{purpose} secret file must be owned by the service user")
    mode = stat.S_IMODE(before.st_mode)
    if mode not in {0o400, 0o600}:
        raise DiagnosticsConfigurationError(
            f"{purpose} secret file mode must be exactly 0400 or 0600"
        )
    if before.st_nlink != 1:
        raise DiagnosticsConfigurationError(f"{purpose} secret file must have exactly one hard link")
    if before.st_size > _MAX_SECRET_BYTES:
        raise DiagnosticsConfigurationError(f"{purpose} secret file is too large")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(secret_path, flags)
    except OSError as exc:
        raise DiagnosticsConfigurationError(f"{purpose} secret file cannot be opened securely") from exc
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise DiagnosticsConfigurationError(f"{purpose} secret file changed while opening")
        if not stat.S_ISREG(opened.st_mode):
            raise DiagnosticsConfigurationError(f"{purpose} secret file must be a regular file")
        if opened.st_uid != os.geteuid():
            raise DiagnosticsConfigurationError(f"{purpose} secret file must be owned by the service user")
        if stat.S_IMODE(opened.st_mode) not in {0o400, 0o600}:
            raise DiagnosticsConfigurationError(
                f"{purpose} secret file mode must be exactly 0400 or 0600"
            )
        if opened.st_nlink != 1:
            raise DiagnosticsConfigurationError(f"{purpose} secret file must have exactly one hard link")
        with os.fdopen(fd, "rb", closefd=True) as stream:
            fd = -1
            raw = stream.read(_MAX_SECRET_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)

    if len(raw) > _MAX_SECRET_BYTES:
        raise DiagnosticsConfigurationError(f"{purpose} secret file is too large")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiagnosticsConfigurationError(f"{purpose} secret file must contain UTF-8 text") from exc
    if value.endswith("\n"):
        value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    if "\n" in value or "\r" in value or "\x00" in value:
        raise DiagnosticsConfigurationError(f"{purpose} secret file must contain exactly one line")
    if value != value.strip():
        raise DiagnosticsConfigurationError(f"{purpose} secret must not have surrounding whitespace")
    if len(value) < minimum_length:
        raise DiagnosticsConfigurationError(f"{purpose} secret is missing or too short")
    return value


def _setting_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise DiagnosticsConfigurationError(f"{name} must be a boolean")


def _setting_int(value: object, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticsConfigurationError(f"{name} must be an integer") from exc


def _secret_path(value: object) -> Path | None:
    normalized = str(value or "").strip()
    return Path(normalized).expanduser() if normalized else None


@dataclass(frozen=True, slots=True)
class DiagnosticsSettings:
    """Public, disabled-by-default settings for optional diagnostic sinks."""

    mqtt_enabled: bool = False
    influx_enabled: bool = False
    base_topic: str = "noesis/occupancy"
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_username: str = "noesis"
    mqtt_password_file: Path | None = None
    mqtt_qos: int = 1
    mqtt_retain: bool = True
    influx_url: str = "http://127.0.0.1:8086"
    influx_org: str = "Lambda"
    influx_token_file: Path | None = None
    influx_bucket: str = "noesis_raw"


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    mqtt_enabled: bool
    influx_enabled: bool
    base_topic: str
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password_file: Path | None
    mqtt_qos: int
    mqtt_retain: bool
    influx_url: str
    influx_org: str
    influx_token_file: Path | None
    influx_bucket: str

    @property
    def enabled(self) -> bool:
        return self.mqtt_enabled or self.influx_enabled

    @classmethod
    def from_settings(
        cls,
        settings: DiagnosticsSettings | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "DiagnosticsConfig":
        configured = settings or DiagnosticsSettings()
        values = os.environ if environ is None else environ
        if "NOESIS_MQTT_PASSWORD" in values or "NOESIS_INFLUX_TOKEN" in values:
            raise DiagnosticsConfigurationError(
                "plaintext credential environment variables are unsupported; use owner-only *_FILE paths"
            )
        mqtt_password_file = _secret_path(
            values.get("NOESIS_MQTT_PASSWORD_FILE")
            or configured.mqtt_password_file
        )
        influx_token_file = _secret_path(
            values.get("NOESIS_INFLUX_TOKEN_FILE")
            or configured.influx_token_file
        )
        return cls(
            mqtt_enabled=_setting_bool(
                configured.mqtt_enabled,
                name="mqtt_enabled",
            ),
            influx_enabled=_setting_bool(
                configured.influx_enabled,
                name="influx_enabled",
            ),
            base_topic=str(configured.base_topic or "").strip(),
            mqtt_host=str(configured.mqtt_host or "").strip(),
            mqtt_port=_setting_int(
                configured.mqtt_port or 0,
                name="mqtt_port",
            ),
            mqtt_username=str(configured.mqtt_username or "").strip(),
            mqtt_password_file=mqtt_password_file,
            mqtt_qos=_setting_int(
                configured.mqtt_qos,
                name="mqtt_qos",
            ),
            mqtt_retain=_setting_bool(
                configured.mqtt_retain,
                name="mqtt_retain",
            ),
            influx_url=str(configured.influx_url or "").strip(),
            influx_org=str(configured.influx_org or "").strip(),
            influx_token_file=influx_token_file,
            influx_bucket=str(configured.influx_bucket or "").strip(),
        )


class DepthDiagnosticsPublisher:
    """Publishes depth summaries to MQTT and InfluxDB."""

    def __init__(self, cfg: DiagnosticsConfig, logger: Optional[logging.Logger] = None) -> None:
        self.cfg = cfg
        self.logger = RateLimitedLogger(logger or logging.getLogger("DepthDiagnosticsPublisher"), rate_limit_seconds=5.0)
        self.client: Optional["mqtt.Client"] = None
        self.influx: Optional["InfluxDBClient"] = None
        self.write_api = None
        if not cfg.enabled:
            raise DiagnosticsConfigurationError(
                "depth diagnostics publisher requires at least one explicitly enabled sink"
            )
        try:
            if cfg.mqtt_enabled:
                self._connect_mqtt()
            if cfg.influx_enabled:
                self._connect_influx()
        except Exception:
            self.close()
            raise
        self.logger.info(
            "Depth diagnostics publisher ready (MQTT=%s, Influx=%s)",
            "on" if self.client else "off",
            "on" if self.write_api else "off",
        )

    @classmethod
    def from_settings(
        cls,
        settings: DiagnosticsSettings | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> "DepthDiagnosticsPublisher | None":
        """Create the publisher only when a sink is explicitly enabled.

        Secret files are not opened while both sink flags are false.  Once a sink
        is enabled, missing dependencies, credentials, or client setup are fatal.
        """

        cfg = DiagnosticsConfig.from_settings(settings, environ=environ)
        if not cfg.enabled:
            return None
        return cls(cfg, logger=logger)

    def _connect_mqtt(self) -> None:
        if not self.cfg.mqtt_host or not (1 <= self.cfg.mqtt_port <= 65535):
            raise DiagnosticsConfigurationError("MQTT diagnostics host/port is invalid")
        if not self.cfg.base_topic:
            raise DiagnosticsConfigurationError("MQTT diagnostics base topic is required")
        if self.cfg.mqtt_qos not in {0, 1, 2}:
            raise DiagnosticsConfigurationError("MQTT diagnostics QoS must be 0, 1, or 2")
        if not self.cfg.mqtt_username:
            raise DiagnosticsConfigurationError("MQTT diagnostics requires an authenticated username")
        if self.cfg.mqtt_password_file is None:
            raise DiagnosticsConfigurationError("MQTT diagnostics password file is required")
        password = load_private_secret(
            self.cfg.mqtt_password_file,
            purpose="MQTT password",
        )
        if mqtt is None:
            raise DiagnosticsConfigurationError(
                "MQTT diagnostics is enabled but paho-mqtt is not installed"
            )
        client = None
        try:
            client = mqtt.Client(client_id="noesis-mapanything-pub", clean_session=True)
            client.username_pw_set(self.cfg.mqtt_username, password)
            result = client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, 60)
            if result not in (None, 0):
                raise RuntimeError("broker rejected the connection")
            client.loop_start()
            self.client = client
        except Exception:  # pragma: no cover - real client failures are environment-dependent
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass
            raise DiagnosticsConfigurationError(
                "MQTT diagnostics is enabled but client initialization failed"
            ) from None

    def _connect_influx(self) -> None:
        if not self.cfg.influx_url or not self.cfg.influx_org or not self.cfg.influx_bucket:
            raise DiagnosticsConfigurationError("Influx diagnostics URL, org, and bucket are required")
        if self.cfg.influx_token_file is None:
            raise DiagnosticsConfigurationError("Influx diagnostics token file is required")
        token = load_private_secret(
            self.cfg.influx_token_file,
            purpose="Influx token",
        )
        if InfluxDBClient is None:
            raise DiagnosticsConfigurationError(
                "Influx diagnostics is enabled but influxdb-client is not installed"
            )
        influx = None
        try:
            influx = InfluxDBClient(
                url=self.cfg.influx_url,
                token=token,
                org=self.cfg.influx_org,
            )
            write_api = influx.write_api(write_options=ASYNCHRONOUS)
            self.influx = influx
            self.write_api = write_api
        except Exception:  # pragma: no cover - real client failures are environment-dependent
            if influx is not None:
                try:
                    influx.close()
                except Exception:
                    pass
            self.influx = None
            self.write_api = None
            raise DiagnosticsConfigurationError(
                "Influx diagnostics is enabled but client initialization failed"
            ) from None

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
                self.logger.error(
                    "Failed to publish depth summary to MQTT (%s)",
                    type(exc).__name__,
                )
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
                self.logger.error(
                    "Failed to write depth summary to InfluxDB (%s)",
                    type(exc).__name__,
                )

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
                self.logger.error(
                    "Failed to write scale metrics to InfluxDB (%s)",
                    type(exc).__name__,
                )

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


__all__ = [
    "DepthDiagnosticsPublisher",
    "DiagnosticsConfig",
    "DiagnosticsConfigurationError",
    "DiagnosticsSettings",
    "load_private_secret",
]
