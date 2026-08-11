from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.paths import dev_console_root


def _launch_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class LaunchSpec:
    pipeline_config: str = "config/infer.yaml"
    cameras_config: str = "config/cameras.yaml"
    pgie_profile: str = "yolo11_seg"
    size: Optional[str] = "m"
    tracking_mode: str = "baseline"
    runtime_entry: str = "noesis/ds8_runtime.py"
    ws_host: str = "127.0.0.1"
    ws_port: int = 6008
    rest_host: str = "127.0.0.1"
    rest_port: int = 8080
    rtsp_port: int = 8554
    enable_rest: bool = True
    log_level: str = "WARNING"
    depth_enable_seconds: int = 0
    strict_baseline: bool = False
    preset_id: Optional[str] = None
    launch_id: str = field(default_factory=_launch_id)
    env: Dict[str, str] = field(default_factory=dict)
    source_overrides: List[Dict[str, Any]] = field(default_factory=list)
    materialized_pipeline: Optional[str] = None

    @property
    def launch_dir(self) -> Path:
        return (
            dev_console_root(self.env.get("NOESIS_BUILD_DIR")) / self.launch_id
        ).resolve(strict=False)

    @property
    def pipeline_path(self) -> Path:
        return (REPO_ROOT / self.pipeline_config).resolve() if not Path(self.pipeline_config).is_absolute() else Path(self.pipeline_config)

    @property
    def cameras_path(self) -> Path:
        return (REPO_ROOT / self.cameras_config).resolve() if not Path(self.cameras_config).is_absolute() else Path(self.cameras_config)

    def to_argv(self, *, pipeline_config: Optional[str] = None) -> List[str]:
        pipeline = pipeline_config or self.materialized_pipeline or self.pipeline_config
        argv = [
            "python3",
            self.runtime_entry,
            "--pipeline-config",
            str(pipeline),
            "--cameras-config",
            self.cameras_config,
            "--pgie-profile",
            self.pgie_profile,
            "--tracking-mode",
            self.tracking_mode,
            "--ws-host",
            self.ws_host,
            "--ws-port",
            str(int(self.ws_port)),
            "--rest-host",
            self.rest_host,
            "--rest-port",
            str(int(self.rest_port)),
            "--log-level",
            self.log_level,
            "--depth-enable-seconds",
            str(int(self.depth_enable_seconds)),
        ]
        if self.size:
            argv.extend(["--size", str(self.size)])
        argv.append("--enable-rest" if self.enable_rest else "--disable-rest")
        return argv

    def to_env(self, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        merged = dict(base or os.environ)
        merged.update({str(key): str(value) for key, value in self.env.items() if str(key).strip()})
        merged["NOESIS_DEV_CONSOLE_LAUNCH_DIR"] = str(self.launch_dir)
        merged["NOESIS_PGIE_PROFILE"] = str(self.pgie_profile)
        merged["NOESIS_TRACKING_MODE"] = str(self.tracking_mode)
        merged["NOESIS_WS_HOST"] = str(self.ws_host)
        merged["NOESIS_WS_PORT"] = str(int(self.ws_port))
        merged["NOESIS_REST_HOST"] = str(self.rest_host)
        merged["NOESIS_REST_PORT"] = str(int(self.rest_port))
        merged["NOESIS_LOG_LEVEL"] = str(self.log_level)
        return merged

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["launch_dir"] = str(self.launch_dir)
        payload["argv"] = self.to_argv()
        payload["env_overlay"] = {
            "NOESIS_DEV_CONSOLE_LAUNCH_DIR": str(self.launch_dir),
            "NOESIS_PGIE_PROFILE": str(self.pgie_profile),
            "NOESIS_TRACKING_MODE": str(self.tracking_mode),
            **{str(key): str(value) for key, value in self.env.items()},
        }
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "LaunchSpec":
        if not payload:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {key: value for key, value in dict(payload).items() if key in known}
        if values.get("size") == "":
            values["size"] = None
        if "ws_port" in values:
            values["ws_port"] = int(values["ws_port"])
        if "rest_port" in values:
            values["rest_port"] = int(values["rest_port"])
        if "rtsp_port" in values:
            values["rtsp_port"] = int(values["rtsp_port"])
        if "depth_enable_seconds" in values:
            values["depth_enable_seconds"] = int(values["depth_enable_seconds"])
        if "enable_rest" in values:
            values["enable_rest"] = bool(values["enable_rest"])
        if "strict_baseline" in values:
            values["strict_baseline"] = bool(values["strict_baseline"])
        env = values.get("env")
        if env is None:
            values["env"] = {}
        elif isinstance(env, str):
            values["env"] = parse_env_lines(env)
        else:
            values["env"] = {str(key): str(value) for key, value in dict(env).items()}
        raw_sources = values.get("source_overrides")
        if raw_sources is None:
            values["source_overrides"] = []
        elif isinstance(raw_sources, list):
            values["source_overrides"] = [
                dict(item)
                for item in raw_sources
                if isinstance(item, Mapping)
            ]
        else:
            values["source_overrides"] = []
        return cls(**values)


def parse_env_lines(text: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value.strip()
    return env
