from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from noesis.dev_console.diagnostics import diagnostics_snapshot, noesis_port_ownership
from noesis.dev_console.launch_plan import build_launch_plan
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.model_matrix import build_model_matrix
from noesis.dev_console.remediation import build_remediation
from noesis.dev_console.source_probe import build_source_readiness
from noesis.dev_console.validator import validate_launch


_RANK = {"ready": 0, "attention": 1, "blocked": 2, "running": 0}


def _counts(results: Mapping[str, Any]) -> Dict[str, int]:
    raw = results.get("counts") if isinstance(results.get("counts"), Mapping) else {}
    return {
        "block": int(raw.get("block", 0) or 0),
        "warn": int(raw.get("warn", 0) or 0),
        "info": int(raw.get("info", 0) or 0),
    }


def _component(key: str, label: str, status: str, detail: str, *, metric: str = "", action: str = "") -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "detail": detail,
        "metric": metric,
        "action": action,
    }


def _worst_status(components: List[Mapping[str, Any]]) -> str:
    worst = "ready"
    for component in components:
        status = str(component.get("status") or "ready")
        if _RANK.get(status, 0) > _RANK.get(worst, 0):
            worst = status
    return worst


def _busy_non_console_ports(diagnostics: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    ports = diagnostics.get("ports") if isinstance(diagnostics.get("ports"), list) else []
    noesis_ports = set(noesis_port_ownership(diagnostics))
    return [
        item
        for item in ports
        if (
            isinstance(item, Mapping)
            and item.get("busy")
            and item.get("label") != "Console"
            and int(item.get("port") or 0) not in noesis_ports
        )
    ]


def _active_model_row(matrix: Mapping[str, Any], active_id: str) -> Mapping[str, Any]:
    rows = matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
    for row in rows:
        if isinstance(row, Mapping) and row.get("id") == active_id:
            return row
    for row in rows:
        if isinstance(row, Mapping) and row.get("active"):
            return row
    return {}


def _external_ds8_processes(diagnostics: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    processes = diagnostics.get("processes") if isinstance(diagnostics.get("processes"), list) else []
    return [
        item
        for item in processes
        if isinstance(item, Mapping) and item.get("looks_like_ds8") and not item.get("managed_by_console")
    ]


def _score(components: List[Mapping[str, Any]]) -> int:
    score = 100
    for component in components:
        status = str(component.get("status") or "ready")
        if status == "blocked":
            score -= 22
        elif status == "attention":
            score -= 9
    return max(0, min(100, score))


def _primary_action(status: str, remediation: Mapping[str, Any], runtime_status: Mapping[str, Any]) -> Dict[str, Any]:
    if runtime_status.get("running"):
        return {
            "title": "Console-managed runtime is already running",
            "detail": f"Launch {runtime_status.get('launch_id') or 'active'} is supervised by this console.",
            "target": "runtime",
            "severity": "ready",
        }
    actions = remediation.get("actions") if isinstance(remediation.get("actions"), list) else []
    for action in actions:
        if isinstance(action, Mapping) and action.get("severity") == "block":
            return {
                "title": action.get("title") or "Resolve blocking finding",
                "detail": action.get("action") or action.get("reason") or "",
                "target": action.get("target") or "runbook",
                "severity": "blocked",
            }
    for action in actions:
        if isinstance(action, Mapping) and action.get("severity") == "warn":
            return {
                "title": action.get("title") or "Review warning",
                "detail": action.get("action") or action.get("reason") or "",
                "target": action.get("target") or "runbook",
                "severity": "attention",
            }
    if status == "ready":
        return {
            "title": "Ready for a console-owned launch",
            "detail": "Validate is clean, selected artifacts are present, sources are ready, and selected ports are free.",
            "target": "start",
            "severity": "ready",
        }
    return {
        "title": "Review launch evidence",
        "detail": "Refresh the decision board after changing launch inputs.",
        "target": "decision",
        "severity": status,
    }


def build_launch_decision(
    spec: LaunchSpec,
    *,
    runtime_status: Optional[Mapping[str, Any]] = None,
    managed_pid: Optional[int] = None,
    probe_network: bool = True,
    timeout_s: float = 0.35,
) -> Dict[str, Any]:
    runtime = dict(runtime_status or {})
    diagnostics = diagnostics_snapshot(spec, managed_pid=managed_pid)
    validation = validate_launch(spec, managed_pid=managed_pid)
    launch_plan = build_launch_plan(spec)
    matrix = build_model_matrix(spec)
    sources = build_source_readiness(spec, probe_network=probe_network, timeout_s=timeout_s)
    remediation = build_remediation(spec, validation=validation, diagnostics=diagnostics, launch_plan=launch_plan)

    validation_counts = _counts(validation)
    artifacts = launch_plan.get("artifacts") if isinstance(launch_plan.get("artifacts"), Mapping) else {}
    source_summary = sources.get("summary") if isinstance(sources.get("summary"), Mapping) else {}
    matrix_summary = matrix.get("summary") if isinstance(matrix.get("summary"), Mapping) else {}
    active_model = _active_model_row(matrix, str(matrix.get("active_id") or f"{spec.pgie_profile}:{spec.size or ''}"))
    busy_ports = _busy_non_console_ports(diagnostics)
    external_ds8 = _external_ds8_processes(diagnostics)
    gpu = diagnostics.get("gpu") if isinstance(diagnostics.get("gpu"), Mapping) else {}

    components = [
        _component(
            "preflight",
            "Preflight",
            "blocked" if validation.get("blocking") or validation_counts["block"] else ("attention" if validation_counts["warn"] else "ready"),
            f"{validation_counts['block']} block / {validation_counts['warn']} warn / {validation_counts['info']} info",
            metric=str(validation_counts["block"] or validation_counts["warn"] or validation_counts["info"]),
            action="Validate",
        ),
        _component(
            "ports",
            "Ports",
            "blocked" if busy_ports else "ready",
            f"{len(busy_ports)} selected DS8 port conflict(s)",
            metric=str(len(busy_ports)),
            action="Find Free Ports",
        ),
        _component(
            "artifacts",
            "Artifacts",
            "blocked" if int(artifacts.get("missing", 0) or 0) else "ready",
            f"{artifacts.get('total', 0)} checked / {artifacts.get('missing', 0)} missing",
            metric=str(artifacts.get("missing", 0)),
            action="Refresh Plan",
        ),
        _component(
            "model",
            "Selected Model",
            "blocked" if active_model.get("status") == "blocked" else ("attention" if active_model.get("status") == "warn" else "ready"),
            f"{spec.pgie_profile}:{spec.size or 'auto'} {active_model.get('kind') or 'model'}",
            metric=str(active_model.get("status") or "unknown"),
            action="Model Matrix",
        ),
        _component(
            "sources",
            "Sources",
            "blocked" if source_summary.get("blocked") else ("attention" if source_summary.get("warn") else "ready"),
            f"{source_summary.get('ready', 0)} ready / {source_summary.get('warn', 0)} warn / {source_summary.get('blocked', 0)} blocked",
            metric=str(source_summary.get("ready", 0)),
            action="Refresh Sources",
        ),
        _component(
            "external_runtime",
            "External DS8",
            "attention" if external_ds8 else "ready",
            f"{len(external_ds8)} non-console DS8 runtime(s) detected",
            metric=str(len(external_ds8)),
            action="Processes",
        ),
        _component(
            "gpu",
            "GPU",
            "ready" if gpu.get("available") else "attention",
            "visible" if gpu.get("available") else str(gpu.get("error") or "GPU status unavailable"),
            metric="ok" if gpu.get("available") else "check",
            action="Diagnostics",
        ),
    ]
    if runtime.get("running"):
        components.append(
            _component(
                "runtime",
                "Console Runtime",
                "running",
                f"pid {runtime.get('pid')} / launch {runtime.get('launch_id') or '-'}",
                metric="running",
                action="Stop",
            )
        )

    status = "running" if runtime.get("running") else _worst_status(components)
    score = _score(components)
    start_allowed = status == "ready" and not runtime.get("running")
    primary_action = _primary_action(status, remediation, runtime)

    return {
        "status": status,
        "score": score,
        "start_allowed": start_allowed,
        "summary": "Ready for Start" if start_allowed else primary_action["title"],
        "validation": validation,
        "primary_action": primary_action,
        "components": components,
        "top_actions": (remediation.get("actions") if isinstance(remediation.get("actions"), list) else [])[:6],
        "evidence": {
            "validation": {"blocking": bool(validation.get("blocking")), "counts": validation_counts},
            "diagnostics": {
                "busy_ports": [
                    {"label": item.get("label"), "port": item.get("port"), "owner": item.get("owner")}
                    for item in busy_ports
                ],
                "process_count": len(diagnostics.get("processes") if isinstance(diagnostics.get("processes"), list) else []),
                "ds8_runtime_count": len(diagnostics.get("ds8_runtimes") if isinstance(diagnostics.get("ds8_runtimes"), list) else []),
                "external_ds8_runtime_count": len(external_ds8),
            },
            "launch_plan": {
                "change_count": launch_plan.get("change_count", 0),
                "artifacts": {
                    "total": artifacts.get("total", 0),
                    "missing": artifacts.get("missing", 0),
                    "ready": artifacts.get("ready", False),
                },
            },
            "model_matrix": {
                "summary": matrix_summary,
                "active_id": matrix.get("active_id"),
                "active_status": active_model.get("status"),
            },
            "sources": {"summary": source_summary},
        },
    }
