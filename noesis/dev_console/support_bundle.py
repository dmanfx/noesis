from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.diagnostics import diagnostics_snapshot
from noesis.dev_console.gate_catalog import build_gate_catalog
from noesis.dev_console.launch_decision import build_launch_decision
from noesis.dev_console.launch_diff import build_launch_diff
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.launch_plan import build_launch_plan
from noesis.dev_console.live_health import summarize_live_health
from noesis.dev_console.live_probe import probe_live_ws
from noesis.dev_console.log_intelligence import analyze_log_file
from noesis.dev_console.model_matrix import build_model_matrix
from noesis.dev_console.pipeline_flow import describe_flow
from noesis.dev_console.remediation import build_remediation
from noesis.dev_console.source_probe import build_source_readiness
from noesis.dev_console.validator import validate_launch


BUNDLE_ROOT = REPO_ROOT / "diagnostics" / "dev_console" / "bundles"
_SENSITIVE_KEY_TOKENS = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE_KEY")


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in _SENSITIVE_KEY_TOKENS)


def _redact(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(child_key): _redact(child_value, key=str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_redact(item, key=key) for item in value]
    if _is_sensitive_key(key):
        return "<redacted>"
    return value


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _bundle_summary(bundle_dir: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    validation = payload.get("validation") if isinstance(payload.get("validation"), Mapping) else {}
    counts = validation.get("counts") if isinstance(validation.get("counts"), Mapping) else {}
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
    processes = diagnostics.get("processes") if isinstance(diagnostics.get("processes"), list) else []
    artifacts = diagnostics.get("artifacts") if isinstance(diagnostics.get("artifacts"), Mapping) else {}
    live = payload.get("live") if isinstance(payload.get("live"), Mapping) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), Mapping) else {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), Mapping) else {}
    spec = payload.get("spec") if isinstance(payload.get("spec"), Mapping) else {}
    stat = bundle_dir.stat()
    bundle_path = bundle_dir / "bundle.json"
    summary_path = bundle_dir / "summary.md"
    ports = {
        "ws": spec.get("ws_port"),
        "rest": spec.get("rest_port"),
        "rtsp": spec.get("rtsp_port"),
    }
    return {
        "id": bundle_dir.name,
        "created_at": payload.get("created_at") or "",
        "launch_id": payload.get("launch_id") or spec.get("launch_id") or "",
        "bundle_dir": _relpath(bundle_dir),
        "bundle_path": _relpath(bundle_path),
        "summary_path": _relpath(summary_path),
        "bundle_exists": bundle_path.exists(),
        "summary_exists": summary_path.exists(),
        "size_bytes": int(bundle_path.stat().st_size) if bundle_path.exists() else 0,
        "mtime": float(stat.st_mtime),
        "counts": {
            "block": int(counts.get("block", 0) or 0),
            "warn": int(counts.get("warn", 0) or 0),
            "info": int(counts.get("info", 0) or 0),
        },
        "blocking": bool(validation.get("blocking") or counts.get("block")),
        "artifacts": {
            "total": artifacts.get("total", 0),
            "missing": artifacts.get("missing", 0),
            "ready": artifacts.get("ready", False),
        },
        "process_count": len(processes),
        "live_connected": bool(live.get("connected")),
        "runtime": {
            "running": bool(runtime.get("running")),
            "pid": runtime.get("pid"),
            "launch_id": runtime.get("launch_id"),
        },
        "decision": {
            "status": decision.get("status") or "",
            "score": decision.get("score"),
            "summary": decision.get("summary") or "",
        },
        "spec": {
            "pgie_profile": spec.get("pgie_profile") or "",
            "size": spec.get("size"),
            "tracking_mode": spec.get("tracking_mode") or "",
            "ports": ports,
        },
    }


def list_support_bundles(*, limit: int = 30) -> Dict[str, Any]:
    if not BUNDLE_ROOT.exists():
        return {"root": _relpath(BUNDLE_ROOT), "total": 0, "items": []}
    dirs = [
        item
        for item in BUNDLE_ROOT.iterdir()
        if item.is_dir() and (item / "bundle.json").exists()
    ]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    items = [_bundle_summary(bundle_dir, _read_json(bundle_dir / "bundle.json")) for bundle_dir in dirs[: max(1, int(limit))]]
    return {"root": _relpath(BUNDLE_ROOT), "total": len(dirs), "items": items}


def read_support_bundle(bundle_id: str) -> Dict[str, Any]:
    safe_id = Path(str(bundle_id or "")).name
    if not safe_id or safe_id != str(bundle_id):
        raise FileNotFoundError(f"Unknown support bundle: {bundle_id}")
    bundle_dir = BUNDLE_ROOT / safe_id
    bundle_path = bundle_dir / "bundle.json"
    summary_path = bundle_dir / "summary.md"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Unknown support bundle: {bundle_id}")
    payload = _read_json(bundle_path)
    try:
        markdown = summary_path.read_text(encoding="utf-8")
    except Exception:
        markdown = ""
    return {
        "summary": _bundle_summary(bundle_dir, payload),
        "markdown": markdown,
        "bundle": _redact(payload),
    }


def _summary_markdown(bundle: Mapping[str, Any]) -> str:
    validation = bundle.get("validation") if isinstance(bundle.get("validation"), Mapping) else {}
    counts = validation.get("counts") if isinstance(validation.get("counts"), Mapping) else {}
    diagnostics = bundle.get("diagnostics") if isinstance(bundle.get("diagnostics"), Mapping) else {}
    processes = diagnostics.get("processes") if isinstance(diagnostics.get("processes"), list) else []
    artifacts = diagnostics.get("artifacts") if isinstance(diagnostics.get("artifacts"), Mapping) else {}
    live = bundle.get("live") if isinstance(bundle.get("live"), Mapping) else {}
    lines = [
        "# Noesis DS8 Dev Console Support Bundle",
        "",
        f"- Created: {bundle.get('created_at', '')}",
        f"- Launch: {bundle.get('launch_id', '')}",
        f"- Validation: {counts.get('block', 0)} block / {counts.get('warn', 0)} warn / {counts.get('info', 0)} info",
        f"- Artifacts: {artifacts.get('total', 0)} checked / {artifacts.get('missing', 0)} missing",
        f"- Observed processes: {len(processes)}",
        f"- Live WebSocket: {'connected' if live.get('connected') else 'not connected'}",
        "",
        "## Processes",
    ]
    if not processes:
        lines.append("")
        lines.append("No port-owning processes were observed for the selected DS8 ports.")
    for process in processes:
        ports = ", ".join(f"{item.get('label')}:{item.get('port')}" for item in process.get("ports", []))
        state = "console-managed" if process.get("managed_by_console") else "external"
        lines.extend(
            [
                "",
                f"- PID {process.get('pid')} ({state}) {ports}",
                f"  - Command: {process.get('command', '')}",
                f"  - CWD: {process.get('cwd', '')}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_support_bundle(spec: LaunchSpec, runtime_status: Mapping[str, Any]) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    bundle_dir = BUNDLE_ROOT / f"{stamp}-{spec.launch_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
    validation = validate_launch(spec)
    diagnostics = diagnostics_snapshot(spec, managed_pid=managed_pid)
    launch_plan = build_launch_plan(spec)
    launch_diff = build_launch_diff(spec)
    remediation = build_remediation(spec, validation=validation, diagnostics=diagnostics, launch_plan=launch_plan)
    gates = build_gate_catalog(spec)
    log_insights = analyze_log_file(
        str(runtime_status.get("log_path") or "") if runtime_status.get("log_path") else None,
        runtime_status=runtime_status,
    )
    model_matrix = build_model_matrix(spec)
    sources = build_source_readiness(spec, probe_network=True, timeout_s=0.35)
    decision = build_launch_decision(spec, runtime_status=runtime_status, managed_pid=managed_pid, probe_network=True, timeout_s=0.35)
    flow = describe_flow(
        pipeline_config=spec.pipeline_config,
        pgie_profile=spec.pgie_profile,
        size=spec.size,
        tracking_mode=spec.tracking_mode,
        rtsp_port=spec.rtsp_port,
        depth_enable_seconds=spec.depth_enable_seconds,
        env=spec.env,
    )
    try:
        live = probe_live_ws(spec.ws_host, int(spec.ws_port), timeout_s=1.5)
    except Exception as exc:
        live = {"connected": False, "error": str(exc)}
    live_health = summarize_live_health(live)

    bundle: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": now.isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "launch_id": spec.launch_id,
        "spec": _redact(spec.to_dict()),
        "runtime": _redact(dict(runtime_status)),
        "launch_plan": launch_plan,
        "launch_diff": launch_diff,
        "gates": gates,
        "log_insights": log_insights,
        "decision": decision,
        "model_matrix": model_matrix,
        "sources": sources,
        "remediation": remediation,
        "validation": validation,
        "diagnostics": diagnostics,
        "flow": flow,
        "live": live,
        "live_health": live_health,
    }

    bundle_path = bundle_dir / "bundle.json"
    summary_path = bundle_dir / "summary.md"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(_summary_markdown(bundle), encoding="utf-8")

    counts = validation.get("counts", {})
    return {
        "ok": True,
        "bundle_dir": str(bundle_dir),
        "bundle_path": str(bundle_path),
        "summary_path": str(summary_path),
        "created_at": bundle["created_at"],
        "launch_id": spec.launch_id,
        "blocking": bool(validation.get("blocking")),
        "counts": counts,
        "process_count": len(diagnostics.get("processes", [])),
        "live_connected": bool(live.get("connected")),
    }
