from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_SEVERITY_RANK = {"info": 0, "warn": 1, "error": 2, "block": 3}
_MAX_EVENTS = 80
_MAX_CONTEXT = 180


_SIGNATURES = [
    {
        "id": "port_conflict",
        "label": "Port Conflict",
        "severity": "block",
        "patterns": (r"address already in use", r"failed to bind", r"bind\(\)", r"port .* busy", r"port .* in use"),
        "target": "ports",
        "action": "Choose free WS/REST/RTSP ports or stop the owning process before launching.",
    },
    {
        "id": "tensorrt_engine",
        "label": "TensorRT Engine",
        "severity": "block",
        "patterns": (r"tensorrt", r"deserialize.*engine", r"engine.*deserialize", r"plan.*version", r"trtexec"),
        "target": "model_artifacts",
        "action": "Rebuild or replace the selected TensorRT engine for the installed CUDA/TensorRT stack.",
    },
    {
        "id": "cuda_gpu",
        "label": "CUDA / GPU",
        "severity": "block",
        "patterns": (r"cuda", r"cudart", r"out of memory", r"\boom\b", r"gpu.*memory", r"no cuda"),
        "target": "gpu",
        "action": "Check GPU visibility and memory pressure, then retry with the intended model/profile.",
    },
    {
        "id": "deepstream_pipeline",
        "label": "DeepStream Pipeline",
        "severity": "error",
        "patterns": (r"pyservicemaker", r"gstreamer", r"\bgst[-_:]", r"nvinfer", r"nvds", r"pipeline.*failed", r"failed to link", r"element .* not found"),
        "target": "pipeline",
        "action": "Inspect the materialized pipeline config and DS8 plugin/model paths.",
    },
    {
        "id": "source_connectivity",
        "label": "Source Connectivity",
        "severity": "warn",
        "patterns": (r"rtsp.*timeout", r"source.*timeout", r"connection refused", r"unauthorized", r"404", r"stream.*not found", r"source.*error"),
        "target": "sources",
        "action": "Refresh Sources and verify camera URI, credentials, and network reachability.",
    },
    {
        "id": "webrtc_rtsp",
        "label": "RTSP / WebRTC",
        "severity": "warn",
        "patterns": (r"webrtc", r"rtsp sink not ready", r"rtsp.*not ready", r"gateway.*failed", r"keyframe"),
        "target": "mosaic",
        "action": "Verify RTSP is enabled, the RTSP port is free, and the WebRTC gateway can consume the mosaic stream.",
    },
    {
        "id": "depth_registration",
        "label": "Depth Registration",
        "severity": "block",
        "patterns": (r"depth[_ -]registration", r"calibration.*fingerprint", r"fingerprint.*mismatch", r"registration.*invalid"),
        "target": "depth_registration",
        "action": "Rebuild the baseline depth-registration artifact for the active camera/model configuration.",
    },
    {
        "id": "python_exception",
        "label": "Python Exception",
        "severity": "error",
        "patterns": (r"traceback \(most recent call last\)", r"exception", r"\berror\b"),
        "target": "runtime",
        "action": "Read the adjacent log context and inspect the module named in the traceback.",
    },
]


def _line_severity(line: str) -> str:
    lower = line.lower()
    if any(token in lower for token in ("critical", "fatal", "traceback", "segmentation fault", "aborting")):
        return "block"
    if re.search(r"(^|[\s\[])(error|exception|failed)(:|\]|\s|$)", lower):
        return "error"
    if re.search(r"(^|[\s\[])(warning|warn)(:|\]|\s|$)", lower):
        return "warn"
    return "info"


def _max_severity(values: Iterable[str]) -> str:
    winner = "info"
    for value in values:
        if _SEVERITY_RANK.get(value, 0) > _SEVERITY_RANK.get(winner, 0):
            winner = value
    return winner


def _trim(line: str) -> str:
    text = re.sub(r"\s+", " ", str(line or "").strip())
    if len(text) <= _MAX_CONTEXT:
        return text
    return text[: _MAX_CONTEXT - 1] + "…"


def _matches(signature: Mapping[str, Any], lower_line: str) -> bool:
    for pattern in signature.get("patterns", ()):
        if re.search(str(pattern), lower_line, flags=re.IGNORECASE):
            return True
    return False


def analyze_runtime_logs(
    lines: Iterable[str],
    *,
    path: Optional[str] = None,
    runtime_status: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    raw_lines = [str(line) for line in lines]
    counts = {"block": 0, "error": 0, "warn": 0, "info": 0}
    signature_hits: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []

    for index, line in enumerate(raw_lines, start=1):
        severity = _line_severity(line)
        lower = line.lower()
        matched = [sig for sig in _SIGNATURES if _matches(sig, lower)]
        if matched:
            severity = _max_severity([severity, *(str(sig["severity"]) for sig in matched)])
        if severity != "info" or matched:
            counts[severity] = counts.get(severity, 0) + 1
            event = {
                "line": index,
                "severity": severity,
                "message": _trim(line),
                "signatures": [str(sig["id"]) for sig in matched],
            }
            events.append(event)
            for signature in matched:
                sig_id = str(signature["id"])
                hit = signature_hits.setdefault(
                    sig_id,
                    {
                        "id": sig_id,
                        "label": signature["label"],
                        "severity": signature["severity"],
                        "target": signature["target"],
                        "action": signature["action"],
                        "count": 0,
                        "last_line": index,
                        "last_message": "",
                    },
                )
                hit["count"] = int(hit["count"]) + 1
                hit["last_line"] = index
                hit["last_message"] = _trim(line)
        else:
            counts["info"] += 1

    events = events[-_MAX_EVENTS:]
    signatures = sorted(
        signature_hits.values(),
        key=lambda item: (-_SEVERITY_RANK.get(str(item.get("severity")), 0), -int(item.get("count", 0)), str(item.get("label", ""))),
    )
    recommendations = [
        {
            "severity": item["severity"],
            "title": item["label"],
            "target": item["target"],
            "action": item["action"],
            "evidence": f"{item['count']} hit(s), latest line {item['last_line']}",
        }
        for item in signatures[:6]
    ]

    status = "clean"
    if counts["block"] or any(item.get("severity") == "block" for item in signatures):
        status = "blocked"
    elif counts["error"] or any(item.get("severity") == "error" for item in signatures):
        status = "error"
    elif counts["warn"] or signatures:
        status = "attention"
    elif not raw_lines:
        status = "empty"

    runtime = dict(runtime_status or {})
    running = bool(runtime.get("running"))
    if status == "clean":
        summary = "No warnings or errors in the sampled console log"
    elif status == "empty":
        summary = "No console-managed runtime log is available yet"
    else:
        summary = f"{counts['block']} block / {counts['error']} error / {counts['warn']} warn in sampled log"

    return {
        "schema_version": 1,
        "path": path,
        "line_count": len(raw_lines),
        "sampled_events": len(events),
        "status": status,
        "summary": summary,
        "running": running,
        "runtime": {
            "running": running,
            "pid": runtime.get("pid"),
            "launch_id": runtime.get("launch_id"),
            "returncode": runtime.get("returncode"),
        },
        "counts": counts,
        "signatures": signatures,
        "recommendations": recommendations,
        "events": events,
    }


def analyze_log_file(path: Optional[str], *, lines: int = 600, runtime_status: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if not path:
        return analyze_runtime_logs([], path=None, runtime_status=runtime_status)
    log_path = Path(path)
    if not log_path.exists():
        return analyze_runtime_logs([], path=str(log_path), runtime_status=runtime_status)
    max_lines = max(1, min(5000, int(lines)))
    text_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return analyze_runtime_logs(text_lines[-max_lines:], path=str(log_path), runtime_status=runtime_status)
