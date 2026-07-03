from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import unquote, urlparse

import yaml

from noesis.ds8_preflight import REPO_ROOT, resolve_config_path
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _display_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "rtsp":
        host = parsed.hostname or "host"
        port = f":{parsed.port}" if parsed.port else ""
        return f"rtsp://{host}{port}/..."
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path or ""))
        return f"file://.../{path.name}" if path.name else "file://..."
    return uri[:120] if len(uri) <= 120 else f"{uri[:120]}..."


def _tcp_probe(host: str, port: int, *, timeout_s: float) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.05, float(timeout_s))):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {"ok": True, "latency_ms": round(elapsed_ms, 1), "error": ""}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {"ok": False, "latency_ms": round(elapsed_ms, 1), "error": str(exc)}


def _file_probe(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "ok": exists,
        "path": str(path),
        "path_rel": _relpath(path),
        "size_bytes": None if stat is None else int(stat.st_size),
        "error": "" if exists else "file missing",
    }


def _camera_by_index(cameras: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    raw = cameras.get(index)
    if raw is None:
        raw = cameras.get(str(index))
    return raw if isinstance(raw, Mapping) else {}


def _intrinsics_ready(model: Mapping[str, Any]) -> bool:
    intrinsics = model.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        return False
    return all(key in intrinsics for key in ("fx", "fy", "cx", "cy"))


def _status(findings: List[Mapping[str, Any]]) -> str:
    if any(item.get("severity") == "block" for item in findings):
        return "blocked"
    if any(item.get("severity") == "warn" for item in findings):
        return "warn"
    return "ready"


def _finding(severity: str, code: str, message: str, detail: str = "") -> Dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, "detail": detail}


def _dewarper_alignment(
    *,
    source: Mapping[str, Any],
    camera_model: Mapping[str, Any],
    pipeline_path: Path,
    cameras_path: Path,
) -> Dict[str, Any]:
    dewarper = source.get("dewarper")
    if not isinstance(dewarper, Mapping) or not bool(dewarper.get("enable", False)):
        return {
            "enabled": False,
            "config": "",
            "config_rel": "",
            "exists": False,
            "expected": "",
            "matches_camera_model": None,
        }
    config_path = resolve_config_path(pipeline_path, dewarper.get("config-file", ""))
    rectification = camera_model.get("rectification") if isinstance(camera_model.get("rectification"), Mapping) else {}
    expected_raw = rectification.get("dewarper_config") if isinstance(rectification, Mapping) else ""
    expected_path = resolve_config_path(cameras_path, expected_raw) if expected_raw else None
    return {
        "enabled": True,
        "config": str(config_path),
        "config_rel": _relpath(config_path),
        "exists": config_path.exists(),
        "expected": "" if expected_path is None else str(expected_path),
        "expected_rel": "" if expected_path is None else _relpath(expected_path),
        "matches_camera_model": None if expected_path is None else config_path == expected_path,
    }


def _source_row(
    *,
    index: int,
    source: Mapping[str, Any],
    camera: Mapping[str, Any],
    intrinsics_models: Mapping[str, Any],
    spec: LaunchSpec,
    pipeline_path: Path,
    cameras_path: Path,
    probe_network: bool,
    timeout_s: float,
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    uri = str(source.get("uri") or "").strip()
    parsed = urlparse(uri)
    camera_name = str(camera.get("name") or f"camera_{index}")
    model_name = str(camera.get("model") or "")
    model = intrinsics_models.get(model_name) if model_name else {}
    model_map = model if isinstance(model, Mapping) else {}

    if not camera:
        findings.append(_finding("block", "camera.missing", f"No camera entry for source {index}"))
    if not model_name:
        findings.append(_finding("block", "camera.model.missing", f"{camera_name} has no intrinsics model"))
    elif not model_map:
        findings.append(_finding("block", "camera.model.unknown", f"Unknown intrinsics model {model_name}"))
    elif not _intrinsics_ready(model_map):
        findings.append(_finding("block", "camera.intrinsics.incomplete", f"{model_name} lacks fx/fy/cx/cy"))

    dewarper = _dewarper_alignment(source=source, camera_model=model_map, pipeline_path=pipeline_path, cameras_path=cameras_path)
    if dewarper["enabled"] and not dewarper["exists"]:
        findings.append(_finding("block", "dewarper.config.missing", "Dewarper config is missing", str(dewarper.get("config_rel") or dewarper.get("config"))))
    if dewarper["matches_camera_model"] is False:
        findings.append(
            _finding(
                "warn",
                "dewarper.camera_model.mismatch",
                "Pipeline dewarper does not match the camera model rectification config",
                f"{dewarper.get('config_rel')} != {dewarper.get('expected_rel')}",
            )
        )

    uri_probe: Dict[str, Any] = {"ok": False, "kind": parsed.scheme or "unknown", "error": ""}
    if not uri:
        findings.append(_finding("block", "source.uri.missing", f"Source {index} has no URI"))
    elif parsed.scheme == "rtsp":
        host = parsed.hostname or ""
        port = int(parsed.port or 554)
        uri_probe.update({"host": host, "port": port})
        if not host:
            findings.append(_finding("block", "source.rtsp.host_missing", "RTSP URI has no host"))
        elif probe_network:
            probe = _tcp_probe(host, port, timeout_s=timeout_s)
            uri_probe.update(probe)
            if not probe["ok"]:
                findings.append(_finding("warn", "source.rtsp.tcp_unreachable", f"RTSP TCP probe failed for {host}:{port}", str(probe.get("error") or "")))
        else:
            uri_probe.update({"ok": None, "error": "network probe skipped"})
    elif parsed.scheme == "file":
        file_path = Path(unquote(parsed.path or ""))
        if not file_path.is_absolute():
            file_path = (REPO_ROOT / file_path).resolve()
        uri_probe.update(_file_probe(file_path))
        if not uri_probe["ok"]:
            findings.append(_finding("block", "source.file.missing", "Source file is missing", str(uri_probe.get("path_rel") or uri_probe.get("path"))))
    else:
        findings.append(_finding("warn", "source.uri.scheme", f"Unsupported or unknown URI scheme: {parsed.scheme or 'none'}"))

    status = _status(findings)
    return {
        "source_id": index,
        "camera_id": camera_name,
        "status": status,
        "element": source.get("element", ""),
        "uri_display": _display_uri(uri),
        "uri_scheme": parsed.scheme or "",
        "uri_probe": uri_probe,
        "latency_ms": source.get("latency"),
        "rtp_protocol": source.get("select-rtp-protocol"),
        "gpu_id": source.get("gpu-id"),
        "camera": {
            "name": camera_name,
            "model": model_name,
            "height_m": camera.get("height_m"),
            "intrinsics_ready": bool(model_map and _intrinsics_ready(model_map)),
            "resolution": model_map.get("resolution"),
        },
        "dewarper": dewarper,
        "findings": findings,
    }


def build_source_readiness(spec: LaunchSpec, *, probe_network: bool = True, timeout_s: float = 0.35) -> Dict[str, Any]:
    pipeline_path = spec.pipeline_path
    cameras_path = spec.cameras_path
    effective = build_effective_config(spec)
    cameras_cfg = _load_mapping(cameras_path)
    cameras = cameras_cfg.get("cameras") if isinstance(cameras_cfg.get("cameras"), Mapping) else {}
    intrinsics_models = cameras_cfg.get("intrinsics_models") if isinstance(cameras_cfg.get("intrinsics_models"), Mapping) else {}
    sources = effective.get("sources") if isinstance(effective.get("sources"), list) else []

    rows = [
        _source_row(
            index=index,
            source=source if isinstance(source, Mapping) else {},
            camera=_camera_by_index(cameras, index),
            intrinsics_models=intrinsics_models,
            spec=spec,
            pipeline_path=pipeline_path,
            cameras_path=cameras_path,
            probe_network=probe_network,
            timeout_s=timeout_s,
        )
        for index, source in enumerate(sources)
    ]

    camera_count = len(cameras)
    source_count = len(sources)
    batch_size = int(effective.get("batch_size", 0) or 0)
    streammux = effective.get("streammux") if isinstance(effective.get("streammux"), Mapping) else {}
    streammux_batch = int(streammux.get("batch-size", 0) or 0)
    alignment: List[Dict[str, Any]] = []
    if source_count != camera_count:
        alignment.append(_finding("block", "sources.camera_count", "Pipeline source count differs from cameras config count", f"{source_count} sources / {camera_count} cameras"))
    if batch_size and source_count and batch_size != source_count:
        alignment.append(_finding("warn", "sources.batch_size", "Pipeline batch_size differs from source count", f"{batch_size} batch / {source_count} sources"))
    if streammux_batch and source_count and streammux_batch != source_count:
        alignment.append(_finding("warn", "sources.streammux_batch", "Streammux batch-size differs from source count", f"{streammux_batch} batch / {source_count} sources"))

    statuses = [row["status"] for row in rows]
    summary = {
        "total": len(rows),
        "ready": statuses.count("ready"),
        "warn": statuses.count("warn"),
        "blocked": statuses.count("blocked"),
        "camera_count": camera_count,
        "source_count": source_count,
        "alignment_status": _status(alignment),
    }
    return {
        "summary": summary,
        "rows": rows,
        "alignment": alignment,
        "pipeline_config": spec.pipeline_config,
        "cameras_config": spec.cameras_config,
        "probe_network": bool(probe_network),
        "timeout_s": float(timeout_s),
    }
