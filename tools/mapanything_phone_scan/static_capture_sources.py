"""Source authority and bounded canonical telemetry observation for WO-2E.

The companion Room Walk capture has two independent inputs: an encoded RTSP
recording from one native camera and JSON messages from the canonical DS9
WebSocket.  This module owns the small amount of provenance needed to pair
those inputs.  It deliberately does not touch the DS9 producer or add a
decoded frame branch.

Two boundaries are kept explicit here:

* A source URI is resolved from the active pipeline and the owner-only camera
  registry, then held only on the in-memory authority object.  Public
  snapshots contain the ``uri_secret`` reference and a digest, never the URI.
* WebSocket messages are copied into a bounded, FIFO writer queue.  A slow
  disk, disconnect, queue overflow, or writer error marks the capture partial;
  it never propagates a failure into the canonical producer.

The observer stores raw canonical envelopes.  It does not join messages with
last-seen camera state and does not infer a phone pose, acquisition timestamp,
or camera/body relationship.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

from noesis_core.runtime_secrets import (
    RuntimeSecretError,
    load_camera_uri_registry,
    load_pipeline_config,
    public_pipeline_config,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "DS9" / "config" / "infer.yaml"
DEFAULT_CAMERAS_CONFIG = REPO_ROOT / "config" / "cameras.yaml"

SOURCE_AUTHORITY_SCHEMA = "noesis.phone_capture.static_source_authority.v1"
STATIC_OBSERVER_SCHEMA = "noesis.phone_capture.static_observer.v1"
STATIC_RAW_ENVELOPE_SCHEMA = "noesis.phone_capture.static_raw_envelope.v1"
STATIC_RUNTIME_RECORDER_SCHEMA = "noesis.phone_capture.static_runtime_recorder.v1"

DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = 6008
DEFAULT_REST_HOST = "127.0.0.1"
DEFAULT_REST_PORT = 8080
DEFAULT_WS_PATH = "/"
TRACKING_STALE_TIMEOUT_S = 5.0
NETWORK_STOP_TIMEOUT_S = 6.0
NETWORK_CONNECT_TIMEOUT_S = 5.0
NETWORK_CLOSE_TIMEOUT_S = 2.0
MAX_WS_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_REST_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_DEWARPER_CONFIG_BYTES = 256 * 1024
REST_SNAPSHOT_PATHS: tuple[tuple[str, str], ...] = (
    ("deployment_health", "/api/v1/health/deployment"),
    ("runtime_capabilities", "/api/v1/health/capabilities"),
    ("scene", "/api/v1/scenes/current"),
    ("revisions", "/api/v1/virtual-twin/revisions"),
)

CANONICAL_MESSAGE_TYPES = frozenset(
    {"tracking", "world_snapshot", "world_event", "bev-frame", "bev-status"}
)
INITIAL_EVIDENCE_MESSAGE_TYPES = frozenset(
    {
        "calibration-bundle",
        "stats",
        "deployment-health",
        "runtime-health",
        "run",
        "revisions",
        "health",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(?:pass(?:word)?|secret|token|credential|authorization|api[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
_RTSP_URI_RE = re.compile(r"rtsps?://[^\s'\"<>]+", re.IGNORECASE)
_ISO_UTC_RE = re.compile(r"[.]?\d{0,6}(?:\+00:00|Z)$")


class StaticCaptureSourceError(RuntimeError):
    """Raised when source authority or observer setup cannot be admitted."""


@dataclass(frozen=True)
class RuntimeConfigPaths:
    """Resolved config paths and their authority source.

    ``pipeline_config`` and ``cameras_config`` are intentionally host-local
    fields.  Use :meth:`public_snapshot` before persisting or returning them.
    """

    pipeline_config: Path
    cameras_config: Path
    authority: str
    runtime_pid: int | None = None
    camera_secrets_file: Path | None = field(default=None, repr=False)

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "runtime_pid": self.runtime_pid,
            "pipeline_config_name": self.pipeline_config.name,
            "cameras_config_name": self.cameras_config.name,
            "camera_secrets_file_name": self.camera_secrets_file.name if self.camera_secrets_file else None,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise StaticCaptureSourceError(f"configured file cannot be read: {path.name}") from exc
    return digest.hexdigest()


def _safe_path(raw: str | Path, *, label: str, base_dir: Path | None = None) -> Path:
    value = Path(str(raw).strip()).expanduser()
    if not value.is_absolute():
        value = ((base_dir or REPO_ROOT) / value)
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StaticCaptureSourceError(f"{label} is missing") from exc
    if not resolved.is_file():
        raise StaticCaptureSourceError(f"{label} is not a regular file")
    return resolved


def _proc_text(pid: int, name: str) -> str:
    try:
        return Path(f"/proc/{int(pid)}/{name}").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise StaticCaptureSourceError(f"DS9 runtime process {pid} is not readable") from exc


def _process_overrides(pid: int) -> tuple[dict[str, str], list[str], Path]:
    """Read only the selected DS9 process command line and environment."""

    try:
        env_raw = Path(f"/proc/{int(pid)}/environ").read_bytes()
        cmd_raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        cwd = Path(os.readlink(f"/proc/{int(pid)}/cwd")).resolve()
    except (OSError, ValueError) as exc:
        raise StaticCaptureSourceError(f"DS9 runtime process {pid} is not readable") from exc
    process_env: dict[str, str] = {}
    for item in env_raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        try:
            process_env[key.decode("utf-8")] = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
    argv: list[str] = []
    for item in cmd_raw.split(b"\0"):
        if not item:
            continue
        try:
            argv.append(item.decode("utf-8"))
        except UnicodeDecodeError:
            argv.append("")
    return process_env, argv, cwd


def discover_active_runtime_pid(*, proc_root: str | Path = "/proc") -> int:
    """Find exactly one running canonical DS9 runtime process.

    This small process-table lookup is intentionally strict: zero or multiple
    matching runtimes is an error, so a companion session cannot silently use
    a stale checkout config or the wrong runtime instance.
    """

    root = Path(proc_root)
    matches: list[int] = []
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise StaticCaptureSourceError("the process table is unavailable") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath("cmdline").read_bytes().split(b"\0")
            command = " ".join(item.decode("utf-8", "ignore") for item in argv if item)
        except OSError:
            continue
        if "ds9_runtime.py" not in command or "DS9/noesis" not in command:
            continue
        try:
            matches.append(int(entry.name))
        except ValueError:
            continue
    if len(matches) != 1:
        if not matches:
            raise StaticCaptureSourceError("no active canonical DS9 runtime process was found")
        raise StaticCaptureSourceError("multiple active canonical DS9 runtime processes require an explicit runtime_pid")
    return matches[0]


def _argv_option(argv: Sequence[str], *names: str) -> str | None:
    names_set = set(names)
    for index, value in enumerate(argv):
        if value in names_set and index + 1 < len(argv):
            candidate = str(argv[index + 1]).strip()
            if candidate:
                return candidate
        for name in names_set:
            prefix = f"{name}="
            if value.startswith(prefix):
                candidate = value[len(prefix) :].strip()
                if candidate:
                    return candidate
    return None


def discover_runtime_config_paths(
    *,
    pipeline_config: str | Path | None = None,
    cameras_config: str | Path | None = None,
    runtime_pid: int | None = None,
    camera_secrets_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    require_process: bool = False,
) -> RuntimeConfigPaths:
    """Resolve the active DS9 source config without guessing source URIs.

    Explicit paths win.  Otherwise a selected runtime PID is read from its
    command line/environment, then the current process environment is used.
    The repository defaults are the documented canonical DS9 defaults and are
    used only when no process-specific override exists.  URI values always
    come from :func:`load_camera_uri_registry`; no fallback URI is invented.
    """

    current_env = dict(os.environ if env is None else env)
    selected_pid = runtime_pid
    if selected_pid is None:
        raw_pid = str(current_env.get("NOESIS_DS9_RUNTIME_PID") or "").strip()
        if raw_pid:
            try:
                selected_pid = int(raw_pid)
            except ValueError as exc:
                raise StaticCaptureSourceError("NOESIS_DS9_RUNTIME_PID must be an integer") from exc
    if selected_pid is None and require_process:
        selected_pid = discover_active_runtime_pid()

    process_env: dict[str, str] = {}
    argv: list[str] = []
    process_cwd = REPO_ROOT
    if selected_pid is not None:
        process_env, argv, process_cwd = _process_overrides(int(selected_pid))

    def _choose(explicit: str | Path | None, env_name: str, option_names: tuple[str, ...], default: Path) -> tuple[Path, str]:
        if explicit is not None:
            return _safe_path(explicit, label=env_name, base_dir=REPO_ROOT), "explicit"
        value = _argv_option(argv, *option_names)
        if value:
            return _safe_path(value, label=env_name, base_dir=process_cwd), "runtime_command_line"
        value = str(process_env.get(env_name) or current_env.get(env_name) or "").strip()
        if value:
            return _safe_path(value, label=env_name, base_dir=process_cwd), "runtime_environment"
        if require_process:
            raise StaticCaptureSourceError(f"active DS9 runtime did not declare {env_name}")
        return _safe_path(default, label=env_name, base_dir=REPO_ROOT), "canonical_default"

    pipeline, pipeline_source = _choose(
        pipeline_config,
        "NOESIS_DS9_PIPELINE_CONFIG",
        ("--pipeline-config", "--pipeline_config"),
        DEFAULT_PIPELINE_CONFIG,
    )
    cameras, cameras_source = _choose(
        cameras_config,
        "NOESIS_CAMERAS_CONFIG",
        ("--cameras-config", "--cameras_config"),
        DEFAULT_CAMERAS_CONFIG,
    )
    secret_raw = camera_secrets_file
    if secret_raw is None:
        secret_raw = process_env.get("NOESIS_CAMERA_SECRETS_FILE") or current_env.get("NOESIS_CAMERA_SECRETS_FILE")
    secret_path = (
        _safe_path(secret_raw, label="NOESIS_CAMERA_SECRETS_FILE", base_dir=process_cwd)
        if secret_raw
        else None
    )
    authority = pipeline_source if pipeline_source == cameras_source else f"{pipeline_source}+{cameras_source}"
    return RuntimeConfigPaths(
        pipeline_config=pipeline,
        cameras_config=cameras,
        authority=authority,
        runtime_pid=(int(selected_pid) if selected_pid is not None else None),
        camera_secrets_file=secret_path,
    )


def _resolve_config_file(pipeline_path: Path, raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if text.startswith("DS9/"):
        return (REPO_ROOT / candidate).resolve(strict=False)
    if text.startswith(("config/", "pipelines/", "build/")):
        return (REPO_ROOT / candidate).resolve(strict=False)
    return (pipeline_path.parent / candidate).resolve(strict=False)


def _camera_entries(path: Path) -> dict[int, dict[str, Any]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise StaticCaptureSourceError("camera config cannot be decoded") from exc
    raw_cameras = payload.get("cameras") if isinstance(payload, Mapping) else None
    if not isinstance(raw_cameras, Mapping):
        raise StaticCaptureSourceError("camera config has no cameras mapping")
    result: dict[int, dict[str, Any]] = {}
    for raw_id, raw_entry in raw_cameras.items():
        try:
            source_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise StaticCaptureSourceError("camera source IDs must be integers") from exc
        if source_id < 0 or not isinstance(raw_entry, Mapping):
            raise StaticCaptureSourceError("camera config contains an invalid camera entry")
        camera_id = str(raw_entry.get("name") or "").strip()
        if not camera_id:
            raise StaticCaptureSourceError(f"camera source {source_id} has no camera name")
        result[source_id] = copy.deepcopy(dict(raw_entry))
        result[source_id]["name"] = camera_id
    if not result:
        raise StaticCaptureSourceError("camera config has no usable cameras")
    if len({str(item["name"]) for item in result.values()}) != len(result):
        raise StaticCaptureSourceError("camera names must be unique")
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sanitize_public(value: Any, *, depth: int = 0, max_items: int = 256) -> Any:
    """Produce a bounded JSON-safe payload with secret-bearing fields removed."""

    if depth > 8:
        return "<truncated>"
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= max_items:
                result["<truncated>"] = True
                break
            key = str(raw_key)
            if _SECRET_KEY_RE.search(key):
                continue
            if key.lower() in {"uri", "url", "endpoint"}:
                text_value = str(raw_value or "")
                if urlsplit(text_value).scheme.lower() in {"rtsp", "rtsps"}:
                    continue
            result[key] = _sanitize_public(raw_value, depth=depth + 1, max_items=max_items)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_public(item, depth=depth + 1, max_items=max_items) for item in list(value)[:max_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return _RTSP_URI_RE.sub("rtsp://<redacted>", value)
        return value
    return str(value)


def canonical_timing_provenance() -> dict[str, Any]:
    """Return the current DS9 temporal contract as an explicit evidence row."""

    return {
        "contract": "noesis.ds9.frame_temporal_contract.v1",
        "observed_at_us": {
            "source": "DS9 callback wall clock",
            "implementation": "time.time()",
            "epoch": "unix",
            "status": "estimated",
        },
        "captured_at_us": {
            "source": "DS9 callback wall clock",
            "implementation": "time.time_ns() // 1000",
            "epoch": "unix",
            "status": "estimated",
        },
        "media_pts_ns": {
            "source": "DeepStream frame metadata buf_pts/buffer_pts/pts",
            "domain": "stream_relative_gstreamer_timestamp",
            "camera_epoch_verified": False,
        },
        "independent_rtsp_recording_pts": {
            "domain": "recorder_pipeline_timestamp",
            "shares_canonical_origin": False,
            "camera_epoch_verified": False,
        },
        "http_or_host_receive_clock": {
            "purpose": "bounded correlation evidence only",
            "acquisition_timestamp_claim": False,
        },
    }


def _dewarper_snapshot(pipeline_path: Path, source: Mapping[str, Any], camera: Mapping[str, Any]) -> dict[str, Any]:
    raw_dewarp = source.get("dewarper")
    dewarp = dict(raw_dewarp) if isinstance(raw_dewarp, Mapping) else {}
    enabled = bool(dewarp.get("enable", False))
    config_raw = str(dewarp.get("config-file") or "").strip()
    config_path = _resolve_config_file(pipeline_path, config_raw)
    config_digest = _sha256_file(config_path) if config_path is not None and config_path.is_file() else None
    output_size: list[int] | None = None
    config_text: str | None = None
    config_entries: dict[str, dict[str, Any]] = {}
    if config_path is not None and config_path.is_file():
        try:
            raw_bytes = config_path.read_bytes()
            if len(raw_bytes) > MAX_DEWARPER_CONFIG_BYTES:
                raise StaticCaptureSourceError("dewarper config exceeds bounded provenance size")
            config_text = raw_bytes.decode("utf-8")
            section = "default"
            for line in config_text.splitlines():
                stripped = line.split("#", 1)[0].strip()
                if not stripped:
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    section = stripped[1:-1].strip() or "default"
                    config_entries.setdefault(section, {})
                    continue
                key, separator, raw_value = stripped.partition("=")
                if not separator:
                    continue
                key = key.strip()
                raw_value = raw_value.strip()
                values = [part.strip() for part in raw_value.split(";")]
                parsed_values: list[Any] = []
                for value in values:
                    try:
                        number = float(value)
                    except ValueError:
                        parsed_values.append(value)
                    else:
                        parsed_values.append(int(number) if number.is_integer() else number)
                config_entries.setdefault(section, {})[key] = (
                    parsed_values if len(parsed_values) > 1 else (parsed_values[0] if parsed_values else "")
                )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise StaticCaptureSourceError("dewarper config cannot be preserved") from exc
    values = config_entries.get("property", {})
    raw_width = values.get("output-width")
    raw_height = values.get("output-height")
    try:
        output_width = int(raw_width)
        output_height = int(raw_height)
    except (TypeError, ValueError):
        output_width = output_height = 0
    if output_width > 0 and output_height > 0:
        output_size = [output_width, output_height]
    camera_model = str(camera.get("model") or "").strip() or None
    return {
        "raw_rtsp_pixels": {
            "authority": "selected source encoded RTSP recording",
            "coordinate_space": "camera_raw_rtsp_pixels",
            "acquisition_dimensions": None,
        },
        "canonical_tracker_pixels": {
            "authority": "DS9 post-decode/dewarper streammux frame",
            "coordinate_space": "post_dewarper_streammux_pixels",
            "dewarper_enabled": enabled,
            "dewarper_config_name": config_path.name if config_path is not None else None,
            "dewarper_config_sha256": config_digest,
            "dewarper_config_text": config_text,
            "dewarper_config_entries": config_entries,
            "output_resolution_px": output_size,
            "camera_intrinsics_model": camera_model,
            "camera_intrinsics_role": "rectified_output" if camera_model and "rectified" in camera_model else "configured_camera_model",
        },
        "binding_status": "explicitly_separate_raw_rtsp_from_post_dewarper_tracker",
    }


@dataclass(frozen=True, repr=False)
class StaticSourceAuthority:
    """Private source authority plus a deliberately narrow public view."""

    config_paths: RuntimeConfigPaths
    source_id: int
    camera_id: str
    label: str
    secret_ref: str
    _private_uri: str = field(repr=False, compare=False)
    source_config: Mapping[str, Any] = field(repr=False, compare=False)
    camera_config: Mapping[str, Any] = field(repr=False, compare=False)
    source_config_sha256: str
    pipeline_config_sha256: str
    cameras_config_sha256: str
    dewarper: Mapping[str, Any]

    def __repr__(self) -> str:  # pragma: no cover - defensive secret boundary
        return (
            f"StaticSourceAuthority(source_id={self.source_id!r}, "
            f"camera_id={self.camera_id!r}, secret_ref={self.secret_ref!r})"
        )

    @property
    def private_uri(self) -> str:
        """Return the URI for one in-process recorder only; never serialize it."""

        return self._private_uri

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_AUTHORITY_SCHEMA,
            "source_id": int(self.source_id),
            "camera_id": self.camera_id,
            "label": self.label,
            "source_provenance": f"camera-secret:{self.secret_ref}",
            "secret_ref": self.secret_ref,
            "pipeline_config_sha256": self.pipeline_config_sha256,
            "cameras_config_sha256": self.cameras_config_sha256,
            "source_config_sha256": self.source_config_sha256,
            "config_authority": self.config_paths.public_snapshot(),
            "source_config": _sanitize_public(self.source_config),
            "camera_config": _sanitize_public(self.camera_config),
            "dewarper": _sanitize_public(self.dewarper),
            "timing": canonical_timing_provenance(),
        }


def _resolve_selected_source(selected_camera: str | int, labels: Mapping[int, Mapping[str, Any]]) -> int:
    text = str(selected_camera).strip()
    try:
        numeric = int(text)
    except ValueError:
        numeric = None
    if numeric is not None and numeric in labels:
        return numeric
    matches = [source_id for source_id, camera in labels.items() if str(camera.get("name")) == text]
    if len(matches) == 1:
        return matches[0]
    raise StaticCaptureSourceError("selected camera is not an active configured source")


def resolve_active_source_authority(
    selected_camera: str | int,
    *,
    pipeline_config: str | Path | None = None,
    cameras_config: str | Path | None = None,
    runtime_pid: int | None = None,
    camera_secrets_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    camera_registry: Mapping[str, str] | None = None,
    require_process: bool = False,
) -> StaticSourceAuthority:
    """Resolve one selected camera from active DS9 config and private URI state."""

    # A production call without explicit injection must bind to the one live
    # canonical DS9 process.  Repository defaults are useful for focused tests,
    # but they are not runtime authority for a companion capture.
    effective_env = os.environ if env is None else env
    require_active_process = bool(
        require_process
        or (
            pipeline_config is None
            and cameras_config is None
            and runtime_pid is None
            and not str(effective_env.get("NOESIS_DS9_RUNTIME_PID") or "").strip()
        )
    )

    paths = discover_runtime_config_paths(
        pipeline_config=pipeline_config,
        cameras_config=cameras_config,
        runtime_pid=runtime_pid,
        camera_secrets_file=camera_secrets_file,
        env=env,
        require_process=require_active_process,
    )
    try:
        config = load_pipeline_config(paths.pipeline_config, materialize_secrets=False)
        public_config = public_pipeline_config(config)
    except RuntimeSecretError as exc:
        raise StaticCaptureSourceError("active pipeline config failed secret-contract validation") from exc
    raw_sources = config.get("sources")
    public_sources = public_config.get("sources")
    if not isinstance(raw_sources, list) or not isinstance(public_sources, list) or len(raw_sources) != len(public_sources):
        raise StaticCaptureSourceError("active pipeline config has no valid sources list")
    labels = _camera_entries(paths.cameras_config)
    source_id = _resolve_selected_source(selected_camera, labels)
    if source_id >= len(raw_sources) or not isinstance(raw_sources[source_id], Mapping):
        raise StaticCaptureSourceError("selected camera is absent from the active source list")
    source = dict(raw_sources[source_id])
    public_source = dict(public_sources[source_id])
    secret_ref = str(source.get("uri_secret") or "").strip()
    if not secret_ref:
        raise StaticCaptureSourceError("selected source has no private uri_secret reference")
    try:
        registry = dict(camera_registry) if camera_registry is not None else load_camera_uri_registry(paths.camera_secrets_file)
        private_uri = str(registry[secret_ref])
    except (KeyError, RuntimeSecretError) as exc:
        raise StaticCaptureSourceError("selected source private camera credential is unavailable") from exc
    if urlsplit(private_uri).scheme.lower() not in {"rtsp", "rtsps"}:
        raise StaticCaptureSourceError("selected source private URI is not RTSP")
    camera = labels[source_id]
    camera_id = str(camera["name"])
    source_digest = _sha256_payload(public_source)
    dewarper = _dewarper_snapshot(paths.pipeline_config, public_source, camera)
    return StaticSourceAuthority(
        config_paths=paths,
        source_id=source_id,
        camera_id=camera_id,
        label=camera_id.replace("-", " ").title(),
        secret_ref=secret_ref,
        _private_uri=private_uri,
        source_config=public_source,
        camera_config=camera,
        source_config_sha256=source_digest,
        pipeline_config_sha256=_sha256_file(paths.pipeline_config),
        cameras_config_sha256=_sha256_file(paths.cameras_config),
        dewarper=dewarper,
    )


def list_active_camera_sources(
    *,
    pipeline_config: str | Path | None = None,
    cameras_config: str | Path | None = None,
    runtime_pid: int | None = None,
    camera_secrets_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    camera_registry: Mapping[str, str] | None = None,
    require_process: bool = False,
) -> dict[str, Any]:
    """Return the UI camera inventory without exposing URI credentials."""

    try:
        effective_env = os.environ if env is None else env
        require_active_process = bool(
            require_process
            or (
                pipeline_config is None
                and cameras_config is None
                and runtime_pid is None
                and not str(effective_env.get("NOESIS_DS9_RUNTIME_PID") or "").strip()
            )
        )
        paths = discover_runtime_config_paths(
            pipeline_config=pipeline_config,
            cameras_config=cameras_config,
            runtime_pid=runtime_pid,
            camera_secrets_file=camera_secrets_file,
            env=env,
            require_process=require_active_process,
        )
        config = load_pipeline_config(paths.pipeline_config, materialize_secrets=False)
        public_config = public_pipeline_config(config)
        raw_sources = config.get("sources")
        public_sources = public_config.get("sources")
        cameras = _camera_entries(paths.cameras_config)
        if not isinstance(raw_sources, list) or not isinstance(public_sources, list):
            raise StaticCaptureSourceError("active pipeline config has no valid sources list")
        try:
            registry = dict(camera_registry) if camera_registry is not None else load_camera_uri_registry(paths.camera_secrets_file)
        except RuntimeSecretError:
            registry = {}
        rows: list[dict[str, Any]] = []
        for source_id in sorted(cameras):
            camera = cameras[source_id]
            camera_id = str(camera["name"])
            active = source_id < len(raw_sources) and isinstance(raw_sources[source_id], Mapping)
            source = raw_sources[source_id] if active else {}
            secret_ref = str(source.get("uri_secret") or "").strip() if isinstance(source, Mapping) else ""
            available = bool(active and secret_ref and secret_ref in registry)
            row: dict[str, Any] = {
                "camera_id": camera_id,
                "label": camera_id.replace("-", " ").title(),
                "source_id": int(source_id),
                "available": available,
            }
            if not active:
                row["reason"] = "camera_not_active_in_pipeline"
            elif not secret_ref:
                row["reason"] = "source_has_no_private_uri_reference"
            elif not available:
                row["reason"] = "private_camera_credential_unavailable"
            rows.append(row)
        return {
            "available": any(row["available"] for row in rows),
            "cameras": rows,
            "source_config_sha256": _sha256_payload(public_sources),
            "config_authority": paths.public_snapshot(),
        }
    except (RuntimeSecretError, StaticCaptureSourceError) as exc:
        return {"available": False, "cameras": [], "reason": str(exc)}


@dataclass(frozen=True)
class ObserverLimits:
    """Finite limits for one observer writer."""

    max_queue_items: int = 2048
    max_queue_bytes: int = 32 * 1024 * 1024
    max_records: int = 200_000
    max_file_bytes: int = 512 * 1024 * 1024
    writer_join_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        if self.max_queue_items < 1 or self.max_queue_bytes < 1024 or self.max_records < 1 or self.max_file_bytes < 1024:
            raise ValueError("observer limits must be positive")
        if self.writer_join_timeout_s <= 0.0:
            raise ValueError("writer_join_timeout_s must be positive")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _message_type(message: Mapping[str, Any]) -> str:
    return str(message.get("type") or "").strip()


def _message_source_id(message: Mapping[str, Any]) -> int | None:
    for mapping in (message, message.get("cohort"), message.get("payload")):
        if not isinstance(mapping, Mapping):
            continue
        for key in ("source_id", "sourceId"):
            parsed = _coerce_int(mapping.get(key))
            if parsed is not None and parsed >= 0:
                return parsed
        cohort = mapping.get("cohort")
        if isinstance(cohort, Mapping):
            parsed = _coerce_int(cohort.get("source_id", cohort.get("sourceId")))
            if parsed is not None and parsed >= 0:
                return parsed
    return None


def _is_relevant_message(message: Mapping[str, Any]) -> bool:
    kind = _message_type(message)
    return kind in CANONICAL_MESSAGE_TYPES or kind in INITIAL_EVIDENCE_MESSAGE_TYPES


def _selected_message(message: Mapping[str, Any], selected_source_id: int | None) -> bool:
    kind = _message_type(message)
    if kind in INITIAL_EVIDENCE_MESSAGE_TYPES:
        return True
    source_id = _message_source_id(message)
    return selected_source_id is None or source_id == int(selected_source_id)


@dataclass
class _QueuedEnvelope:
    payload: bytes
    envelope: dict[str, Any]


class StaticCaptureObserver:
    """Bounded FIFO observer for canonical tracking/world/BEV WebSocket JSON."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        selected_source_id: int | None,
        limits: ObserverLimits | None = None,
        initial_provenance: Mapping[str, Any] | None = None,
        writer: Any | None = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.selected_source_id = selected_source_id if selected_source_id is None else int(selected_source_id)
        self.limits = limits or ObserverLimits()
        self._initial_provenance = _sanitize_public(dict(initial_provenance or {}))
        self._external_writer = writer
        self._queue: queue.Queue[_QueuedEnvelope | None] = queue.Queue(maxsize=self.limits.max_queue_items)
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._accepting = False
        self._stop_requested = False
        self._partial = False
        self._partial_reasons: list[str] = []
        self._records = 0
        self._bytes_queued = 0
        self._bytes_written = 0
        self._message_counts: dict[str, int] = {}
        self._next_sequence = 0
        self._started_utc: str | None = None
        self._stopped_utc: str | None = None
        self._last_received_monotonic_ns: int | None = None
        self._last_received_utc: str | None = None
        self._writer_error: str | None = None

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._accepting:
                return self.status()
            if self._thread is not None and self._thread.is_alive():
                raise StaticCaptureSourceError("observer writer is already stopping")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._accepting = True
            self._stop_requested = False
            self._started_utc = _now_utc()
            self._thread = threading.Thread(target=self._writer_loop, name="StaticCaptureObserver", daemon=True)
            self._thread.start()
            return self.status()

    def _mark_partial_locked(self, reason: str) -> None:
        text = str(reason).strip()[:240] or "observer_partial"
        self._partial = True
        if text not in self._partial_reasons and len(self._partial_reasons) < 32:
            self._partial_reasons.append(text)

    def mark_partial(self, reason: str) -> dict[str, Any]:
        with self._lock:
            self._mark_partial_locked(reason)
            return self.status()

    def handle_disconnect(self, reason: str = "websocket_disconnected") -> dict[str, Any]:
        """Record a disconnect and stop accepting messages without raising."""

        with self._lock:
            self._mark_partial_locked(reason)
            self._accepting = False
            self._stop_requested = True
            self._enqueue_stop_locked()
            return self.status()

    def _enqueue_stop_locked(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # The writer will observe stop_requested after draining what it can;
            # the partial marker is the durable loss signal.
            self._mark_partial_locked("observer_stop_queue_full")

    def observe(
        self,
        message: Mapping[str, Any],
        *,
        received_monotonic_ns: int | None = None,
        received_unix_ns: int | None = None,
        received_utc: str | None = None,
    ) -> bool:
        """Enqueue one selected raw message without waiting on disk."""

        if not isinstance(message, Mapping):
            return False
        kind = _message_type(message)
        if not _is_relevant_message(message) or not _selected_message(message, self.selected_source_id):
            return False
        monotonic_ns = int(received_monotonic_ns if received_monotonic_ns is not None else time.monotonic_ns())
        unix_ns = int(received_unix_ns if received_unix_ns is not None else time.time_ns())
        utc = str(received_utc or _now_utc())
        if monotonic_ns < 0 or unix_ns <= 0:
            return False
        with self._lock:
            if not self._accepting or self._stop_requested:
                return False
            if self._records >= self.limits.max_records:
                self._mark_partial_locked("observer_record_limit_exceeded")
                self._accepting = False
                self._stop_requested = True
                self._enqueue_stop_locked()
                return False
            # Serialization is bounded JSON work, not disk I/O. Keeping it
            # inside the same lock as put_nowait preserves FIFO sequence order
            # when a test/client invokes observe concurrently.
            sequence = self._next_sequence
            try:
                raw_message = copy.deepcopy(dict(message))
                envelope = {
                    "schema": STATIC_RAW_ENVELOPE_SCHEMA,
                    "observer_sequence": int(sequence),
                    "message_type": kind,
                    "source_id": _message_source_id(raw_message),
                    "received_monotonic_ns": str(monotonic_ns),
                    "received_unix_ns": str(unix_ns),
                    "received_utc": utc,
                    "message": raw_message,
                }
                payload = _canonical_json(envelope) + b"\n"
            except (TypeError, ValueError, OverflowError) as exc:
                self._mark_partial_locked(f"observer_message_not_json:{type(exc).__name__}")
                return False
            if self._bytes_queued + len(payload) > self.limits.max_queue_bytes:
                self._mark_partial_locked("observer_queue_byte_limit_exceeded")
                self._accepting = False
                self._stop_requested = True
                self._enqueue_stop_locked()
                return False
            try:
                self._queue.put_nowait(_QueuedEnvelope(payload=payload, envelope=envelope))
            except queue.Full:
                self._mark_partial_locked("observer_queue_item_limit_exceeded")
                self._accepting = False
                self._stop_requested = True
                self._enqueue_stop_locked()
                return False
            self._records += 1
            self._next_sequence += 1
            self._bytes_queued += len(payload)
            self._message_counts[kind] = self._message_counts.get(kind, 0) + 1
            self._last_received_monotonic_ns = monotonic_ns
            self._last_received_utc = utc
            return True

    def _writer_loop(self) -> None:
        handle = None

        def _discard_queued() -> None:
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    return
                else:
                    if item is not None:
                        with self._lock:
                            self._bytes_queued = max(0, self._bytes_queued - len(item.payload))
                    self._queue.task_done()
        try:
            if self._external_writer is not None:
                handle = self._external_writer
            else:
                handle = self.output_path.open("ab")
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    with self._lock:
                        should_exit = self._stop_requested and self._queue.empty()
                    if should_exit:
                        break
                    continue
                try:
                    can_write = False
                    with self._lock:
                        if self._bytes_written + len(item.payload) > self.limits.max_file_bytes:
                            self._mark_partial_locked("observer_file_byte_limit_exceeded")
                            self._accepting = False
                            self._stop_requested = True
                            self._bytes_queued = max(0, self._bytes_queued - len(item.payload))
                        else:
                            can_write = True
                    # Disk I/O is intentionally outside the state lock. A
                    # blocked filesystem must not block observe(), status(),
                    # or the manager heartbeat.
                    if can_write:
                        handle.write(item.payload)
                        flush = getattr(handle, "flush", None)
                        if callable(flush):
                            flush()
                        with self._lock:
                            self._bytes_written += len(item.payload)
                            self._bytes_queued = max(0, self._bytes_queued - len(item.payload))
                    else:
                        break
                except Exception as exc:  # noqa: BLE001 - writer must fail closed
                    with self._lock:
                        self._writer_error = type(exc).__name__
                        self._mark_partial_locked("observer_writer_failure")
                        self._accepting = False
                        self._stop_requested = True
                        self._bytes_queued = max(0, self._bytes_queued - len(item.payload))
                    break
                finally:
                    self._queue.task_done()
        except Exception as exc:  # noqa: BLE001 - setup failure is observer-local
            with self._lock:
                self._writer_error = type(exc).__name__
                self._mark_partial_locked("observer_writer_setup_failure")
                self._accepting = False
                self._stop_requested = True
        finally:
            _discard_queued()
            if handle is not None and handle is not self._external_writer:
                try:
                    handle.close()
                except Exception:
                    pass
            with self._lock:
                self._accepting = False

    def stop(self, reason: str = "completed") -> dict[str, Any]:
        with self._lock:
            if reason not in {
                "completed",
                "stopped",
                "user",
                "user_stop",
                "service_shutdown",
                "lease_expired",
                "max_duration_s",
            }:
                self._mark_partial_locked(reason)
            self._accepting = False
            self._stop_requested = True
            self._stopped_utc = _now_utc()
            self._enqueue_stop_locked()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.limits.writer_join_timeout_s)
            if thread.is_alive():
                with self._lock:
                    self._mark_partial_locked("observer_writer_join_timeout")
        with self._lock:
            return self.status()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": STATIC_OBSERVER_SCHEMA,
                "selected_source_id": self.selected_source_id,
                "raw_message_types": sorted(CANONICAL_MESSAGE_TYPES),
                "initial_provenance": copy.deepcopy(self._initial_provenance),
                "timing": canonical_timing_provenance(),
                "output_name": self.output_path.name,
                "limits": {
                    "max_queue_items": self.limits.max_queue_items,
                    "max_queue_bytes": self.limits.max_queue_bytes,
                    "max_records": self.limits.max_records,
                    "max_file_bytes": self.limits.max_file_bytes,
                },
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": STATIC_OBSERVER_SCHEMA,
                "state": "recording" if self._accepting else "stopped" if self._stop_requested else "idle",
                "selected_source_id": self.selected_source_id,
                "partial": self._partial,
                "partial_reasons": list(self._partial_reasons),
                "writer_error": self._writer_error,
                "records": self._records,
                "bytes_written": self._bytes_written,
                "bytes_queued": self._bytes_queued,
                "queue_items": self._queue.qsize(),
                "message_counts": dict(self._message_counts),
                "started_utc": self._started_utc,
                "stopped_utc": self._stopped_utc,
                "last_received_monotonic_ns": str(self._last_received_monotonic_ns) if self._last_received_monotonic_ns is not None else None,
                "last_received_utc": self._last_received_utc,
            }


def build_static_capture_provenance(
    authority: StaticSourceAuthority,
    *,
    calibration_bundle: Mapping[str, Any] | None = None,
    deployment_health: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    revisions: Mapping[str, Any] | None = None,
    coordinate_frame: Mapping[str, Any] | None = None,
    camera_models: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a public initial snapshot for a companion session."""

    return {
        "schema": "noesis.phone_capture.static_capture_provenance.v1",
        "source": authority.public_snapshot(),
        "calibration_bundle": _sanitize_public(dict(calibration_bundle or {})),
        "deployment_health": _sanitize_public(dict(deployment_health or {})),
        "runtime": _sanitize_public(dict(runtime or {})),
        "run": _sanitize_public(dict(run or {})),
        "revisions": _sanitize_public(dict(revisions or {})),
        "coordinate_frame": _sanitize_public(dict(coordinate_frame or {
            "canonical_tracking_world": "backend_world_m",
            "presentation_scene_frame": "menon_scene",
            "authority": "canonical_tracking_world",
        })),
        "camera_models": _sanitize_public(dict(camera_models or {})),
        "timing": canonical_timing_provenance(),
    }


def _runtime_process_settings(authority: StaticSourceAuthority) -> tuple[dict[str, str], list[str]]:
    """Read endpoint/auth settings from the same process that owns authority."""

    if authority.config_paths.runtime_pid is None:
        return dict(os.environ), []
    process_env, argv, _ = _process_overrides(authority.config_paths.runtime_pid)
    return process_env, argv


def _endpoint_value(
    process_env: Mapping[str, str],
    argv: Sequence[str],
    env_name: str,
    option_names: tuple[str, ...],
    default: str,
) -> str:
    value = _argv_option(argv, *option_names)
    if value:
        return value
    value = str(process_env.get(env_name) or "").strip()
    return value or default


def _runtime_endpoints(authority: StaticSourceAuthority) -> tuple[str, str]:
    process_env, argv = _runtime_process_settings(authority)
    ws_override = _endpoint_value(
        process_env,
        argv,
        "NOESIS_WS_URL",
        ("--ws-url",),
        "",
    )
    rest_override = _endpoint_value(
        process_env,
        argv,
        "NOESIS_REST_URL",
        ("--rest-url",),
        "",
    )
    ws_host = _endpoint_value(process_env, argv, "NOESIS_WS_HOST", ("--ws-host",), DEFAULT_WS_HOST)
    ws_port = _endpoint_value(process_env, argv, "NOESIS_WS_PORT", ("--ws-port",), str(DEFAULT_WS_PORT))
    rest_host = _endpoint_value(
        process_env,
        argv,
        "NOESIS_REST_HOST",
        ("--rest-host",),
        DEFAULT_REST_HOST,
    )
    rest_port = _endpoint_value(
        process_env,
        argv,
        "NOESIS_REST_PORT",
        ("--rest-port",),
        str(DEFAULT_REST_PORT),
    )
    ws_path = _endpoint_value(process_env, argv, "NOESIS_WS_PATH", (), DEFAULT_WS_PATH)
    if ws_override:
        parsed_ws = urlsplit(ws_override)
        if parsed_ws.scheme.lower() not in {"ws", "wss"} or not parsed_ws.hostname:
            raise StaticCaptureSourceError("NOESIS_WS_URL must be a ws:// or wss:// URL")
        ws_url = ws_override
    else:
        try:
            ws_port_int = int(ws_port)
        except ValueError as exc:
            raise StaticCaptureSourceError("NOESIS_WS_PORT must be an integer") from exc
        if not 1 <= ws_port_int <= 65535:
            raise StaticCaptureSourceError("NOESIS_WS_PORT is out of range")
        if not ws_path.startswith("/"):
            ws_path = f"/{ws_path}"
        ws_url = f"ws://{ws_host}:{ws_port_int}{ws_path}"
    if rest_override:
        parsed_rest = urlsplit(rest_override)
        if parsed_rest.scheme.lower() not in {"http", "https"} or not parsed_rest.hostname:
            raise StaticCaptureSourceError("NOESIS_REST_URL must be an http:// or https:// URL")
        rest_base = rest_override.rstrip("/")
    else:
        try:
            rest_port_int = int(rest_port)
        except ValueError as exc:
            raise StaticCaptureSourceError("NOESIS_REST_PORT must be an integer") from exc
        if not 1 <= rest_port_int <= 65535:
            raise StaticCaptureSourceError("NOESIS_REST_PORT is out of range")
        rest_scheme = str(process_env.get("NOESIS_REST_SCHEME") or "http").strip().lower()
        if rest_scheme not in {"http", "https"}:
            raise StaticCaptureSourceError("NOESIS_REST_SCHEME must be http or https")
        rest_base = f"{rest_scheme}://{rest_host}:{rest_port_int}"
    return ws_url, rest_base


def _load_runtime_auth(authority: StaticSourceAuthority) -> Any:
    """Load the required bearer helper using the active runtime's token path."""

    from scripts.internal_auth_client import load_required_internal_auth

    process_env, _ = _runtime_process_settings(authority)
    token_file = str(process_env.get("NOESIS_INTERNAL_AUTH_TOKEN_FILE") or "").strip() or None
    return load_required_internal_auth(token_file, env=process_env)


def _connect_authenticated_websocket(uri: str, auth: Any) -> Any:
    """Open the canonical WS through the repository-owned auth helper."""

    from scripts.internal_auth_client import connect_required_websocket

    return connect_required_websocket(
        uri,
        auth,
        compression=None,
        max_size=MAX_WS_MESSAGE_BYTES,
        open_timeout=NETWORK_CONNECT_TIMEOUT_S,
        close_timeout=NETWORK_CLOSE_TIMEOUT_S,
    )


def _fetch_authenticated_rest_json(url: str, auth: Any) -> Mapping[str, Any]:
    """Fetch one bounded authenticated REST JSON response."""

    from scripts.internal_auth_client import build_required_auth_request

    request = build_required_auth_request(url, auth, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_CONNECT_TIMEOUT_S) as response:
            raw = response.read(MAX_REST_RESPONSE_BYTES + 1)
            if len(raw) > MAX_REST_RESPONSE_BYTES:
                raise StaticCaptureSourceError("authenticated REST response exceeds bounded size")
    except urllib.error.HTTPError as exc:
        raise StaticCaptureSourceError(f"authenticated REST returned HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise StaticCaptureSourceError("authenticated REST request failed") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticCaptureSourceError("authenticated REST returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise StaticCaptureSourceError("authenticated REST response must be a JSON object")
    return payload


def _camera_candidates(message: Mapping[str, Any]) -> set[str]:
    candidates: set[str] = set()
    mappings: list[Mapping[str, Any]] = [message]
    for key in ("cohort", "payload", "data"):
        value = message.get(key)
        if isinstance(value, Mapping):
            mappings.append(value)
    for mapping in mappings:
        for key in ("camera", "camera_id", "cameraId", "sensor", "sensor_id", "sensorId"):
            value = mapping.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                text = str(value).strip().lower()
                if text:
                    candidates.add(text)
    return candidates


def _message_matches_authority(message: Mapping[str, Any], authority: StaticSourceAuthority) -> bool:
    """Require source identity and honor any explicit camera identity."""

    source_id = _message_source_id(message)
    if source_id != int(authority.source_id):
        return False
    candidates = _camera_candidates(message)
    if not candidates:
        # The canonical tracking payload binds camera identity through source_id.
        return True
    aliases = {
        str(authority.source_id).lower(),
        str(authority.camera_id).strip().lower(),
        str(authority.label).strip().lower(),
    }
    return bool(candidates & aliases)


def _tracking_publication_sequence(message: Mapping[str, Any]) -> int | None:
    for mapping in (message, message.get("cohort"), message.get("payload")):
        if not isinstance(mapping, Mapping):
            continue
        for key in (
            "tracking_publication_sequence",
            "trackingPublicationSequence",
            "publication_sequence",
            "publicationSequence",
            "seq",
        ):
            value = _coerce_int(mapping.get(key))
            if value is not None and value >= 0:
                return value
    return None


def _tracking_count(message: Mapping[str, Any]) -> int:
    for key in ("tracks", "objects", "observations"):
        rows = message.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            return len(rows)
    count = _coerce_int(message.get("track_count"))
    if count is None:
        count = _coerce_int(message.get("object_count"))
    if count is None:
        count = _coerce_int(message.get("observation_count"))
    return max(0, count or 0)


class CanonicalTrackingRecorder:
    """Authenticated canonical WS observer paired to one static authority.

    The recorder owns only network receipt and bounded JSONL persistence.  It
    never derives a camera pose, changes coordinate frames, or merges the
    independent REST/WS snapshots into a fabricated world state.
    """

    def __init__(self, authority: StaticSourceAuthority, output_dir: Path, max_records: int) -> None:
        if not isinstance(authority, StaticSourceAuthority):
            raise TypeError("CanonicalTrackingRecorder requires StaticSourceAuthority")
        if int(max_records) < 1:
            raise ValueError("max_records must be positive")
        self.authority = authority
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.max_records = int(max_records)
        self.tracking_path = self.output_dir / "tracking.ndjson"
        self.provenance_path = self.output_dir / "provenance.json"
        self.rest_paths = {
            key: self.output_dir / f"rest_{key}.json" for key, _ in REST_SNAPSHOT_PATHS
        }
        self._observer = StaticCaptureObserver(
            self.tracking_path,
            selected_source_id=authority.source_id,
            limits=ObserverLimits(max_records=self.max_records),
        )
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._partial = False
        self._error: str | None = None
        self._started_monotonic_ns: int | None = None
        self._started_unix_ns: int | None = None
        self._last_tracking_monotonic_ns: int | None = None
        self._last_tracking_publication_sequence: int | None = None
        self._tracking_ready = False
        self._tracking_record_count = 0
        self._tracked_count = 0
        self._calibration_seen = False
        self._rest_snapshots: dict[str, dict[str, Any]] = {}
        self._observer_status: dict[str, Any] = {}
        self._stop_reason = "completed"

    def _mark_partial(self, reason: str, *, stop: bool = True) -> None:
        text = str(reason).strip()[:240] or "recorder_partial"
        with self._lock:
            self._partial = True
            if self._error is None:
                self._error = text
            if stop:
                self._status = "failed"
                self._stop_event.set()

    def _sync_observer_status(self) -> dict[str, Any]:
        observed = self._observer.status()
        with self._lock:
            self._observer_status = dict(observed)
            if bool(observed.get("partial")):
                self._partial = True
                if self._error is None:
                    reasons = observed.get("partial_reasons") or ["observer_partial"]
                    self._error = str(reasons[0])
                self._status = "failed"
                self._stop_event.set()
        return observed

    def _write_json_artifact(self, path: Path, payload: Mapping[str, Any]) -> str:
        encoded = _canonical_json(_sanitize_public(dict(payload))) + b"\n"
        if len(encoded) > MAX_REST_RESPONSE_BYTES:
            raise StaticCaptureSourceError(f"artifact {path.name} exceeds bounded size")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
        return hashlib.sha256(encoded).hexdigest()

    def _write_provenance(self) -> None:
        with self._lock:
            payload = build_static_capture_provenance(
                self.authority,
                coordinate_frame={
                    "canonical_tracking_world": "backend_world_m",
                    "presentation_scene_frame": "menon_scene",
                    "authority": "canonical_tracking_world",
                },
            )
            payload.update(
                {
                    "schema": STATIC_RUNTIME_RECORDER_SCHEMA,
                    "started_monotonic_ns": (
                        str(self._started_monotonic_ns)
                        if self._started_monotonic_ns is not None
                        else None
                    ),
                    "started_unix_ns": (
                        str(self._started_unix_ns) if self._started_unix_ns is not None else None
                    ),
                    "rest_snapshots": copy.deepcopy(self._rest_snapshots),
                    "artifacts": {
                        "tracking": self.tracking_path.name,
                        "provenance": self.provenance_path.name,
                        **{f"rest_{key}": path.name for key, path in self.rest_paths.items()},
                    },
                }
            )
        self._write_json_artifact(self.provenance_path, payload)

    def _record_rest_snapshot(
        self,
        key: str,
        path: str,
        payload: Mapping[str, Any] | None,
        error: str | None,
        received_monotonic_ns: int,
        received_unix_ns: int,
    ) -> None:
        record: dict[str, Any] = {
            "schema": STATIC_RUNTIME_RECORDER_SCHEMA,
            "endpoint_path": path,
            "receipt_clock": {
                "received_monotonic_ns": str(received_monotonic_ns),
                "received_unix_ns": str(received_unix_ns),
                "clock_semantics": "host_receipt_only",
            },
            "error": error,
        }
        if payload is not None:
            record["payload"] = _sanitize_public(dict(payload))
        try:
            digest = self._write_json_artifact(self.rest_paths[key], record)
            record["sha256"] = digest
        except Exception as exc:  # noqa: BLE001 - snapshot failure is recorder-local
            self._mark_partial(f"rest_artifact_write_failure:{key}")
            record["error"] = record.get("error") or type(exc).__name__
        with self._lock:
            self._rest_snapshots[key] = record

    async def _snapshot_rest(self, rest_base: str, auth: Any) -> None:
        async def _one(key: str, path: str) -> tuple[str, str, Mapping[str, Any] | None, str | None, int, int]:
            started_mono = time.monotonic_ns()
            started_unix = time.time_ns()
            try:
                payload = await asyncio.to_thread(
                    _fetch_authenticated_rest_json,
                    f"{rest_base}{path}",
                    auth,
                )
                return key, path, payload, None, time.monotonic_ns(), time.time_ns()
            except Exception as exc:  # noqa: BLE001 - endpoint evidence is independent
                return (
                    key,
                    path,
                    None,
                    f"{type(exc).__name__}: {_RTSP_URI_RE.sub('rtsp://<redacted>', str(exc))}",
                    max(started_mono, time.monotonic_ns()),
                    max(started_unix, time.time_ns()),
                )

        results = await asyncio.gather(
            *(_one(key, path) for key, path in REST_SNAPSHOT_PATHS),
            return_exceptions=False,
        )
        for key, path, payload, error, received_mono, received_unix in results:
            self._record_rest_snapshot(key, path, payload, error, received_mono, received_unix)
            if error:
                self._mark_partial(f"rest_snapshot_failure:{key}", stop=False)
        try:
            self._write_provenance()
        except Exception:
            self._mark_partial("provenance_write_failure")

    def _handle_message(self, raw: Any) -> None:
        received_monotonic_ns = time.monotonic_ns()
        received_unix_ns = time.time_ns()
        if isinstance(raw, (bytes, bytearray)):
            if len(raw) > MAX_WS_MESSAGE_BYTES:
                self._mark_partial("websocket_message_too_large")
                return
            try:
                raw = bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                self._mark_partial("websocket_message_not_utf8")
                return
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_WS_MESSAGE_BYTES:
            self._mark_partial("websocket_message_invalid")
            return
        try:
            message = json.loads(raw)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            self._mark_partial("websocket_parse_failure")
            return
        if not isinstance(message, Mapping):
            self._mark_partial("websocket_message_not_object")
            return
        kind = _message_type(message)
        if kind not in CANONICAL_MESSAGE_TYPES and kind not in INITIAL_EVIDENCE_MESSAGE_TYPES:
            return
        if kind in CANONICAL_MESSAGE_TYPES and not _message_matches_authority(message, self.authority):
            return
        if not self._observer.observe(
            message,
            received_monotonic_ns=received_monotonic_ns,
            received_unix_ns=received_unix_ns,
            received_utc=_now_utc(),
        ):
            observed = self._sync_observer_status()
            if observed.get("partial"):
                return
            return
        if kind == "calibration-bundle":
            with self._lock:
                self._calibration_seen = True
        elif kind == "tracking":
            with self._lock:
                self._tracking_ready = True
                self._tracking_record_count += 1
                self._tracked_count += _tracking_count(message)
                self._last_tracking_monotonic_ns = received_monotonic_ns
                self._last_tracking_publication_sequence = _tracking_publication_sequence(message)
        self._sync_observer_status()

    def _check_stale(self) -> None:
        now = time.monotonic_ns()
        with self._lock:
            reference = self._last_tracking_monotonic_ns or self._started_monotonic_ns or now
            ready = self._tracking_ready
        if not ready and (now - reference) / 1e9 > TRACKING_STALE_TIMEOUT_S:
            self._mark_partial("tracking_stale_timeout")
        elif ready and (now - reference) / 1e9 > TRACKING_STALE_TIMEOUT_S:
            self._mark_partial("tracking_stale_timeout")

    async def _network_async(self) -> None:
        process_env, _ = _runtime_process_settings(self.authority)
        del process_env  # endpoint/auth helpers read the same process context
        auth = _load_runtime_auth(self.authority)
        ws_url, rest_base = _runtime_endpoints(self.authority)
        rest_task: asyncio.Task[None] | None = None
        try:
            async with _connect_authenticated_websocket(ws_url, auth) as websocket:
                with self._lock:
                    if self._status != "failed":
                        self._status = "recording"
                rest_task = asyncio.create_task(self._snapshot_rest(rest_base, auth))
                while not self._stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        self._check_stale()
                        continue
                    except Exception as exc:  # noqa: BLE001 - WS disconnect is evidence
                        if not self._stop_event.is_set():
                            self._mark_partial(f"websocket_disconnect:{type(exc).__name__}")
                        break
                    self._handle_message(raw)
                    self._check_stale()
                    if self._stop_event.is_set():
                        break
                if rest_task is not None and not rest_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(rest_task), timeout=3.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        rest_task.cancel()
        finally:
            if rest_task is not None and not rest_task.done():
                rest_task.cancel()
                await asyncio.gather(rest_task, return_exceptions=True)

    def _network_thread(self) -> None:
        try:
            asyncio.run(self._network_async())
        except Exception as exc:  # noqa: BLE001 - network failure is observer-local
            if not self._stop_event.is_set() or self._error is None:
                self._mark_partial(f"network_failure:{type(exc).__name__}")
        finally:
            with self._lock:
                calibration_seen = self._calibration_seen
            if not calibration_seen and self._error is None:
                self._mark_partial("calibration_bundle_missing", stop=False)
            try:
                observed = self._observer.stop(self._stop_reason)
                with self._lock:
                    self._observer_status = dict(observed)
                    if observed.get("partial"):
                        self._partial = True
                        if self._error is None:
                            reasons = observed.get("partial_reasons") or ["observer_partial"]
                            self._error = str(reasons[0])
                    if self._status not in {"failed"}:
                        self._status = "stopped"
            except Exception as exc:  # noqa: BLE001 - finalization is bounded
                self._mark_partial(f"observer_stop_failure:{type(exc).__name__}", stop=False)
            try:
                self._write_provenance()
            except Exception:
                self._mark_partial("provenance_write_failure", stop=False)
            self._done_event.set()

    def start(self) -> None:
        with self._lock:
            if self._status in {"starting", "recording"}:
                return
            if self._thread is not None and self._thread.is_alive():
                raise StaticCaptureSourceError("canonical tracking recorder is already stopping")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._started_monotonic_ns = time.monotonic_ns()
            self._started_unix_ns = time.time_ns()
            self._stop_event.clear()
            self._done_event.clear()
            self._status = "starting"
            self._partial = False
            self._error = None
            self._stop_reason = "completed"
        try:
            self._observer.start()
            self._write_provenance()
        except Exception as exc:  # noqa: BLE001 - setup is recorder-local
            self._mark_partial(f"recorder_setup_failure:{type(exc).__name__}")
            try:
                self._observer.stop("setup_failure")
            except Exception:
                pass
            self._done_event.set()
            return
        self._thread = threading.Thread(target=self._network_thread, name="CanonicalTrackingRecorder", daemon=True)
        self._thread.start()

    def stop(self, reason: str = "user") -> Mapping[str, Any]:
        with self._lock:
            self._stop_reason = str(reason or "user")
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=NETWORK_STOP_TIMEOUT_S)
        if thread is not None and thread.is_alive():
            self._mark_partial("network_stop_timeout", stop=False)
            try:
                observed = self._observer.stop("network_stop_timeout")
                with self._lock:
                    self._observer_status = dict(observed)
            except Exception:
                pass
        elif self._thread is None:
            try:
                observed = self._observer.stop(self._stop_reason)
                with self._lock:
                    self._observer_status = dict(observed)
            except Exception:
                pass
            self._done_event.set()
        return self.status()

    def status(self) -> dict[str, Any]:
        observed = self._sync_observer_status()
        with self._lock:
            result = {
                "schema": STATIC_RUNTIME_RECORDER_SCHEMA,
                "status": self._status,
                "ready": self._tracking_ready,
                "tracking_ready": self._tracking_ready,
                "source_id": int(self.authority.source_id),
                "camera_id": self.authority.camera_id,
                "last_tracking_monotonic_ns": (
                    str(self._last_tracking_monotonic_ns)
                    if self._last_tracking_monotonic_ns is not None
                    else None
                ),
                "last_tracking_publication_sequence": self._last_tracking_publication_sequence,
                "record_count": self._tracking_record_count,
                "tracking_record_count": self._tracking_record_count,
                "tracked_count": self._tracked_count,
                "message_count": int(observed.get("records") or 0),
                "calibration_seen": self._calibration_seen,
                "partial": self._partial,
                "error": self._error,
                "observer": dict(observed),
                "artifacts": {
                    "tracking": self.tracking_path.name,
                    "provenance": self.provenance_path.name,
                    **{f"rest_{key}": path.name for key, path in self.rest_paths.items()},
                },
                "source": self.authority.public_snapshot(),
            }
            return result


__all__ = [
    "CANONICAL_MESSAGE_TYPES",
    "INITIAL_EVIDENCE_MESSAGE_TYPES",
    "CanonicalTrackingRecorder",
    "ObserverLimits",
    "RuntimeConfigPaths",
    "SOURCE_AUTHORITY_SCHEMA",
    "STATIC_OBSERVER_SCHEMA",
    "STATIC_RAW_ENVELOPE_SCHEMA",
    "STATIC_RUNTIME_RECORDER_SCHEMA",
    "StaticCaptureObserver",
    "StaticCaptureSourceError",
    "StaticSourceAuthority",
    "build_static_capture_provenance",
    "canonical_timing_provenance",
    "discover_runtime_config_paths",
    "discover_active_runtime_pid",
    "list_active_camera_sources",
    "resolve_active_source_authority",
]
