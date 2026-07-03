from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from noesis.dev_console.diagnostics import diagnostics_snapshot
from noesis.dev_console.launch_decision import build_launch_decision
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.model_matrix import build_model_matrix


_STATUS_RANK = {"ready": 0, "running": 1, "attention": 2, "blocked": 3}


def _apply_spec(spec: LaunchSpec) -> Dict[str, Any]:
    return {
        "pipeline_config": spec.pipeline_config,
        "cameras_config": spec.cameras_config,
        "pgie_profile": spec.pgie_profile,
        "size": spec.size,
        "tracking_mode": spec.tracking_mode,
        "ws_port": spec.ws_port,
        "rest_port": spec.rest_port,
        "rtsp_port": spec.rtsp_port,
        "log_level": spec.log_level,
        "depth_enable_seconds": spec.depth_enable_seconds,
        "strict_baseline": spec.strict_baseline,
        "env": dict(spec.env or {}),
    }


def _fingerprint(spec: LaunchSpec) -> Tuple[Any, ...]:
    env_items = tuple(sorted((str(key), str(value)) for key, value in (spec.env or {}).items()))
    return (
        spec.pipeline_config,
        spec.cameras_config,
        spec.pgie_profile,
        spec.size,
        spec.tracking_mode,
        int(spec.ws_port),
        int(spec.rest_port),
        int(spec.rtsp_port),
        spec.log_level,
        int(spec.depth_enable_seconds),
        bool(spec.strict_baseline),
        env_items,
    )


def _changed_fields(base: LaunchSpec, candidate: LaunchSpec) -> List[Dict[str, Any]]:
    labels = {
        "pgie_profile": "PGIE",
        "size": "Size",
        "tracking_mode": "Tracking",
        "ws_port": "WS port",
        "rest_port": "REST port",
        "rtsp_port": "RTSP port",
        "depth_enable_seconds": "Depth gate",
        "strict_baseline": "Strict baseline",
        "log_level": "Log level",
    }
    changes: List[Dict[str, Any]] = []
    for field, label in labels.items():
        before = getattr(base, field)
        after = getattr(candidate, field)
        if before != after:
            changes.append({"field": field, "label": label, "before": before, "after": after})
    if dict(base.env or {}) != dict(candidate.env or {}):
        changes.append(
            {
                "field": "env",
                "label": "Env overlay",
                "before": len(base.env or {}),
                "after": len(candidate.env or {}),
            }
        )
    return changes


def _candidate(
    candidate_id: str,
    title: str,
    intent: str,
    spec: LaunchSpec,
    *,
    tag: str,
    origin: str,
) -> Dict[str, Any]:
    return {
        "id": candidate_id,
        "title": title,
        "intent": intent,
        "tag": tag,
        "origin": origin,
        "spec": spec,
    }


def _first_row(rows: Iterable[Mapping[str, Any]], *, kind: str, statuses: Set[str]) -> Optional[Mapping[str, Any]]:
    preferred_profiles = {
        "masks": ["yolo11_seg", "yolo26_seg", "rfdetr_seg", "wholebody49"],
        "boxes": ["yolo26", "yolo11", "rfdetr"],
    }.get(kind, [])
    preferred_size = {"m": 0, "s": 1, "n": 2, "l": 3, "x": 4}

    candidates = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("kind") == kind
        and str(row.get("status") or "") in statuses
        and row.get("pgie_profile")
    ]
    if not candidates:
        return None

    def sort_key(row: Mapping[str, Any]) -> Tuple[int, int, int, str]:
        profile = str(row.get("pgie_profile") or "")
        size = str(row.get("size") or "")
        status_rank = 0 if row.get("status") == "ready" else 1
        profile_rank = preferred_profiles.index(profile) if profile in preferred_profiles else 99
        return (status_rank, profile_rank, preferred_size.get(size, 9), f"{profile}:{size}")

    return sorted(candidates, key=sort_key)[0]


def _spec_from_row(base: LaunchSpec, row: Mapping[str, Any], *, strict_baseline: Optional[bool] = None) -> LaunchSpec:
    return replace(
        base,
        pgie_profile=str(row.get("pgie_profile") or base.pgie_profile),
        size=str(row.get("size") or base.size or ""),
        strict_baseline=base.strict_baseline if strict_baseline is None else bool(strict_baseline),
        materialized_pipeline=None,
    )


def _summarize_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
    components = decision.get("components") if isinstance(decision.get("components"), list) else []
    blockers = [
        {"label": item.get("label"), "status": item.get("status"), "detail": item.get("detail")}
        for item in components
        if isinstance(item, Mapping) and item.get("status") in {"blocked", "attention"}
    ][:4]
    return {
        "status": decision.get("status") or "unknown",
        "score": int(decision.get("score", 0) or 0),
        "start_allowed": bool(decision.get("start_allowed")),
        "summary": decision.get("summary") or "",
        "primary_action": decision.get("primary_action") or {},
        "blockers": blockers,
    }


def build_launch_candidates(
    spec: LaunchSpec,
    *,
    runtime_status: Optional[Mapping[str, Any]] = None,
    managed_pid: Optional[int] = None,
    probe_network: bool = True,
    timeout_s: float = 0.35,
) -> Dict[str, Any]:
    runtime = dict(runtime_status or {})
    diagnostics = diagnostics_snapshot(spec, managed_pid=managed_pid)
    matrix = build_model_matrix(spec)
    rows = matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
    suggestions = diagnostics.get("suggested_ports") if isinstance(diagnostics.get("suggested_ports"), Mapping) else {}
    busy_ports = [
        item
        for item in diagnostics.get("ports", [])
        if isinstance(item, Mapping) and item.get("busy") and item.get("label") != "Console"
    ]

    raw_candidates: List[Dict[str, Any]] = [
        _candidate(
            "current",
            "Current selection",
            "Evaluate the exact launch form as configured.",
            spec,
            tag="current",
            origin="form",
        )
    ]

    if busy_ports and suggestions:
        raw_candidates.append(
            _candidate(
                "free-ports",
                "Safe free-port clone",
                "Keep the launch recipe intact while moving DS8 network outputs away from observed port owners.",
                replace(
                    spec,
                    ws_port=int(suggestions.get("ws_port") or spec.ws_port),
                    rest_port=int(suggestions.get("rest_port") or spec.rest_port),
                    rtsp_port=int(suggestions.get("rtsp_port") or spec.rtsp_port),
                    materialized_pipeline=None,
                ),
                tag="ports",
                origin="diagnostics",
            )
        )

    mask_row = _first_row(rows, kind="masks", statuses={"ready", "warn"})
    if mask_row:
        raw_candidates.append(
            _candidate(
                "strict-depth-seg",
                "Strict depth segmentation",
                "Use a mask-capable PGIE and strict baseline checks for depth-critical launches.",
                _spec_from_row(spec, mask_row, strict_baseline=True),
                tag="depth",
                origin="model-matrix",
            )
        )

    box_row = _first_row(rows, kind="boxes", statuses={"ready", "warn"})
    if box_row:
        raw_candidates.append(
            _candidate(
                "detector-throughput",
                "Detector throughput",
                "Use a detector PGIE for lighter baseline object tracking when strict depth masks are not required.",
                _spec_from_row(spec, box_row, strict_baseline=False),
                tag="throughput",
                origin="model-matrix",
            )
        )

    if spec.tracking_mode != "v3dt":
        raw_candidates.append(
            _candidate(
                "v3dt-tracking",
                "V3DT tracking pass",
                "Keep the selected model and ports while switching the tracking method to V3DT.",
                replace(spec, tracking_mode="v3dt", materialized_pipeline=None),
                tag="tracking",
                origin="tracking",
            )
        )

    seen: Set[Tuple[Any, ...]] = set()
    candidates: List[Dict[str, Any]] = []
    for raw in raw_candidates:
        candidate_spec = raw["spec"]
        fingerprint = _fingerprint(candidate_spec)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        decision = build_launch_decision(
            candidate_spec,
            runtime_status=runtime,
            managed_pid=managed_pid,
            probe_network=probe_network,
            timeout_s=timeout_s,
        )
        decision_summary = _summarize_decision(decision)
        candidates.append(
            {
                "id": raw["id"],
                "title": raw["title"],
                "intent": raw["intent"],
                "tag": raw["tag"],
                "origin": raw["origin"],
                "status": decision_summary["status"],
                "score": decision_summary["score"],
                "start_allowed": decision_summary["start_allowed"],
                "summary": decision_summary["summary"],
                "primary_action": decision_summary["primary_action"],
                "blockers": decision_summary["blockers"],
                "changes": _changed_fields(spec, candidate_spec),
                "apply_spec": _apply_spec(candidate_spec),
                "model": {
                    "pgie_profile": candidate_spec.pgie_profile,
                    "size": candidate_spec.size,
                    "tracking_mode": candidate_spec.tracking_mode,
                    "strict_baseline": candidate_spec.strict_baseline,
                },
                "ports": {
                    "ws": candidate_spec.ws_port,
                    "rest": candidate_spec.rest_port,
                    "rtsp": candidate_spec.rtsp_port,
                },
            }
        )

    candidates.sort(
        key=lambda item: (
            _STATUS_RANK.get(str(item.get("status")), 9),
            -int(item.get("score", 0) or 0),
            0 if item.get("id") == "current" else 1,
            str(item.get("title") or ""),
        )
    )
    best = candidates[0] if candidates else None
    return {
        "summary": {
            "total": len(candidates),
            "ready": sum(1 for item in candidates if item.get("status") == "ready" and item.get("start_allowed")),
            "attention": sum(1 for item in candidates if item.get("status") == "attention"),
            "blocked": sum(1 for item in candidates if item.get("status") == "blocked"),
            "best_id": best.get("id") if best else "",
            "best_status": best.get("status") if best else "unknown",
        },
        "current_id": "current",
        "items": candidates,
        "diagnostics": {
            "busy_ports": [
                {"label": item.get("label"), "port": item.get("port"), "owner": item.get("owner")}
                for item in busy_ports
            ],
            "suggested_ports": dict(suggestions),
        },
    }
