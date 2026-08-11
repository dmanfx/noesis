from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml

from .private_paths import (
    PrivatePathError,
    ensure_private_directory,
    read_private_file,
)


CAMERA_SECRETS_ENV = "NOESIS_CAMERA_SECRETS_FILE"
MAPANYTHING_API_KEY_FILE_ENV = "NOESIS_MAPANYTHING_API_KEY_FILE"
DEFAULT_CAMERA_SECRETS_PATH = Path("~/.local/state/noesis/secrets/camera_sources.json")
DEFAULT_MAPANYTHING_API_KEY_PATH = Path("~/.local/state/noesis/secrets/mapanything_rpc.key")
MAX_CAMERA_SECRETS_BYTES = 64 * 1024
MAX_MAPANYTHING_API_KEY_BYTES = 1024

_SECRET_REF_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_AUTHORITY_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{43,512}$")
_RTSP_TEXT_RE = re.compile(r"rtsps?://[^\s'\"<>]+", re.IGNORECASE)
_RTSP_SCHEMES = frozenset({"rtsp", "rtsps"})


class RuntimeSecretError(RuntimeError):
    """Raised when a runtime secret is missing, unsafe, or malformed."""


def redact_runtime_secrets(value: Any) -> str:
    """Remove complete RTSP locators from errors before they enter logs."""

    return _RTSP_TEXT_RE.sub("rtsp://<redacted>", str(value))


def _configured_path(env_name: str, default: Path, explicit: str | Path | None) -> Path:
    raw = explicit if explicit is not None else os.environ.get(env_name, "")
    candidate = Path(str(raw).strip()) if str(raw).strip() else default
    return candidate.expanduser()


def camera_secrets_path(explicit: str | Path | None = None) -> Path:
    return _configured_path(CAMERA_SECRETS_ENV, DEFAULT_CAMERA_SECRETS_PATH, explicit)


def mapanything_api_key_path(explicit: str | Path | None = None) -> Path:
    return _configured_path(
        MAPANYTHING_API_KEY_FILE_ENV,
        DEFAULT_MAPANYTHING_API_KEY_PATH,
        explicit,
    )


def _read_owner_only(path: Path, *, label: str, max_bytes: int) -> bytes:
    try:
        ensure_private_directory(path.parent, label=f"{label} parent")
        return read_private_file(path, label=label, max_bytes=max_bytes)
    except PrivatePathError as exc:
        raise RuntimeSecretError(str(exc)) from exc


def load_mapanything_api_key(path: str | Path | None = None) -> str:
    candidate = mapanything_api_key_path(path)
    payload = _read_owner_only(
        candidate,
        label="MapAnything RPC key",
        max_bytes=MAX_MAPANYTHING_API_KEY_BYTES,
    )
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeSecretError("MapAnything RPC key must be UTF-8") from exc
    if not _AUTHORITY_KEY_RE.fullmatch(value):
        raise RuntimeSecretError(
            "MapAnything RPC key must be at least 43 URL-safe characters"
        )
    return value


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeSecretError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs)
    except UnicodeDecodeError as exc:
        raise RuntimeSecretError(f"{label} must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeSecretError(f"{label} must contain valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeSecretError(f"{label} must contain a JSON object")
    return decoded


def _validate_secret_ref(value: Any, *, context: str) -> str:
    ref = str(value or "").strip()
    if not _SECRET_REF_RE.fullmatch(ref):
        raise RuntimeSecretError(f"{context} must be a lowercase runtime-secret reference")
    return ref


def _validate_camera_uri(value: Any, *, ref: str) -> str:
    uri = str(value or "")
    if not uri or uri != uri.strip() or len(uri.encode("utf-8")) > 4096:
        raise RuntimeSecretError(f"camera secret '{ref}' has an invalid URI")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in uri):
        raise RuntimeSecretError(f"camera secret '{ref}' has an invalid URI")
    parsed = urlsplit(uri)
    if parsed.scheme.lower() not in _RTSP_SCHEMES or not parsed.hostname:
        raise RuntimeSecretError(f"camera secret '{ref}' must be an RTSP or RTSPS URI")
    return uri


def load_camera_uri_registry(path: str | Path | None = None) -> dict[str, str]:
    candidate = camera_secrets_path(path)
    payload = _read_owner_only(
        candidate,
        label="camera source secrets",
        max_bytes=MAX_CAMERA_SECRETS_BYTES,
    )
    decoded = _strict_json_object(payload, label="camera source secrets")
    if set(decoded) != {"version", "sources"} or decoded.get("version") != 1:
        raise RuntimeSecretError(
            "camera source secrets must contain only version=1 and sources"
        )
    raw_sources = decoded.get("sources")
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise RuntimeSecretError("camera source secrets must contain a non-empty sources object")
    if len(raw_sources) > 64:
        raise RuntimeSecretError("camera source secrets exceeds the 64-source limit")
    result: dict[str, str] = {}
    seen_uris: set[str] = set()
    for raw_ref, raw_uri in raw_sources.items():
        ref = _validate_secret_ref(raw_ref, context="camera secret key")
        uri = _validate_camera_uri(raw_uri, ref=ref)
        if uri in seen_uris:
            raise RuntimeSecretError("camera source secrets contains duplicate URI values")
        seen_uris.add(uri)
        result[ref] = uri
    return result


def _is_inline_rtsp_uri(value: Any) -> bool:
    try:
        return urlsplit(str(value or "").strip()).scheme.lower() in _RTSP_SCHEMES
    except ValueError:
        return False


def materialize_pipeline_config(
    config: Mapping[str, Any],
    *,
    camera_registry: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve camera URI references in memory without mutating public config."""

    if not isinstance(config, Mapping):
        raise RuntimeSecretError("pipeline config must be a mapping")
    materialized = copy.deepcopy(dict(config))
    sources = materialized.get("sources")
    if sources is None:
        return materialized
    if not isinstance(sources, list):
        raise RuntimeSecretError("pipeline config sources must be a list")

    needs_registry = any(
        isinstance(source, Mapping) and "uri_secret" in source for source in sources
    )
    registry = dict(camera_registry) if camera_registry is not None else None
    if needs_registry and registry is None:
        registry = load_camera_uri_registry()

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise RuntimeSecretError(f"pipeline source {index} must be a mapping")
        has_ref = "uri_secret" in source
        has_uri = "uri" in source and str(source.get("uri") or "").strip() != ""
        if has_ref:
            ref = _validate_secret_ref(
                source.get("uri_secret"), context=f"pipeline source {index} uri_secret"
            )
            if has_uri:
                raise RuntimeSecretError(
                    f"pipeline source {index} must not combine uri_secret with uri"
                )
            if registry is None or ref not in registry:
                raise RuntimeSecretError(
                    f"pipeline source {index} references missing camera secret '{ref}'"
                )
            source["uri_secret"] = ref
            source["uri"] = _validate_camera_uri(registry[ref], ref=ref)
        elif has_uri and _is_inline_rtsp_uri(source.get("uri")):
            raise RuntimeSecretError(
                f"pipeline source {index} contains an inline RTSP URI; use uri_secret"
            )
    return materialized


def load_pipeline_config(
    path: str | Path,
    *,
    materialize_secrets: bool = True,
    camera_registry: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    candidate = Path(path)
    try:
        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeSecretError(f"pipeline config cannot be read: {candidate}") from exc
    if not isinstance(loaded, Mapping):
        raise RuntimeSecretError("pipeline config must be a mapping")
    if materialize_secrets:
        return materialize_pipeline_config(loaded, camera_registry=camera_registry)
    return copy.deepcopy(dict(loaded))


def public_pipeline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serialization-safe config with all referenced camera URIs removed."""

    public = copy.deepcopy(dict(config))
    sources = public.get("sources")
    if not isinstance(sources, list):
        return public
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        if "uri_secret" in source:
            source["uri_secret"] = _validate_secret_ref(
                source.get("uri_secret"), context=f"pipeline source {index} uri_secret"
            )
            source.pop("uri", None)
        elif _is_inline_rtsp_uri(source.get("uri")):
            raise RuntimeSecretError(
                f"pipeline source {index} contains an inline RTSP URI and cannot be serialized"
            )
    return public


def source_provenance_ref(source: Mapping[str, Any], *, source_id: int) -> str:
    if "uri_secret" in source:
        ref = _validate_secret_ref(
            source.get("uri_secret"), context=f"pipeline source {source_id} uri_secret"
        )
        return f"camera-secret:{ref}"
    uri = str(source.get("uri") or "").strip()
    if _is_inline_rtsp_uri(uri):
        raise RuntimeSecretError(
            f"pipeline source {source_id} contains an inline RTSP URI and cannot enter provenance"
        )
    return uri


__all__ = [
    "CAMERA_SECRETS_ENV",
    "DEFAULT_CAMERA_SECRETS_PATH",
    "DEFAULT_MAPANYTHING_API_KEY_PATH",
    "MAPANYTHING_API_KEY_FILE_ENV",
    "RuntimeSecretError",
    "camera_secrets_path",
    "load_camera_uri_registry",
    "load_mapanything_api_key",
    "load_pipeline_config",
    "mapanything_api_key_path",
    "materialize_pipeline_config",
    "public_pipeline_config",
    "redact_runtime_secrets",
    "source_provenance_ref",
]
