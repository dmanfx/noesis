from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import build_effective_config


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}


def _get(mapping: Mapping[str, Any], path: Iterable[str], default: Any = None) -> Any:
    value: Any = mapping
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _boolish(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value if value is not None else "").strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _bool_text(value: Any, *, default: bool = False) -> str:
    return "1" if _boolish(value, default=default) else "0"


def _number_text(value: Any, default: Any = "0") -> str:
    if value in (None, ""):
        return str(default)
    return str(value)


def _env_value(spec: LaunchSpec, key: str) -> Optional[str]:
    if key not in spec.env:
        return None
    return str(spec.env.get(key, ""))


def _env_bool_control(
    spec: LaunchSpec,
    *,
    control_id: str,
    label: str,
    target: str,
    group: str,
    inherited: Any,
    source_default: str,
    detail: str,
    impact: str,
) -> Dict[str, Any]:
    override = _env_value(spec, target)
    inherited_text = _bool_text(inherited)
    effective_text = _bool_text(override if override not in (None, "") else inherited_text, default=_boolish(inherited))
    explicit = override not in (None, "")
    return {
        "id": control_id,
        "label": label,
        "target": target,
        "target_type": "env",
        "group": group,
        "kind": "boolean",
        "value": override if explicit else "",
        "effective_value": effective_text,
        "inherited_value": inherited_text,
        "effective_display": "on" if effective_text == "1" else "off",
        "source": "env override" if explicit else source_default,
        "explicit": explicit,
        "inheritable": True,
        "active": effective_text == "1",
        "detail": detail,
        "impact": impact,
        "options": [
            {"value": "", "label": "inherit"},
            {"value": "1", "label": "on"},
            {"value": "0", "label": "off"},
        ],
    }


def _env_number_control(
    spec: LaunchSpec,
    *,
    control_id: str,
    label: str,
    target: str,
    group: str,
    inherited: Any,
    source_default: str,
    detail: str,
    impact: str,
    minimum: float,
    maximum: float,
    step: float,
    unit: str,
) -> Dict[str, Any]:
    override = _env_value(spec, target)
    inherited_text = _number_text(inherited)
    effective_text = _number_text(override, inherited_text) if override not in (None, "") else inherited_text
    return {
        "id": control_id,
        "label": label,
        "target": target,
        "target_type": "env",
        "group": group,
        "kind": "number",
        "value": override if override not in (None, "") else "",
        "effective_value": effective_text,
        "inherited_value": inherited_text,
        "effective_display": f"{effective_text} {unit}".strip(),
        "source": "env override" if override not in (None, "") else source_default,
        "explicit": override not in (None, ""),
        "inheritable": True,
        "active": float(effective_text or 0) > 0,
        "detail": detail,
        "impact": impact,
        "min": minimum,
        "max": maximum,
        "step": step,
        "unit": unit,
    }


def _field_control(
    *,
    control_id: str,
    label: str,
    target: str,
    group: str,
    kind: str,
    value: Any,
    detail: str,
    impact: str,
    options: Optional[List[Dict[str, str]]] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None,
    unit: str = "",
) -> Dict[str, Any]:
    if kind == "boolean":
        value_text = _bool_text(value)
        effective_display = "on" if value_text == "1" else "off"
        active = value_text == "1"
    else:
        value_text = _number_text(value) if kind == "number" else str(value or "")
        effective_display = f"{value_text} {unit}".strip()
        active = bool(value_text) and value_text != "0"
    control: Dict[str, Any] = {
        "id": control_id,
        "label": label,
        "target": target,
        "target_type": "field",
        "group": group,
        "kind": kind,
        "value": value_text,
        "effective_value": value_text,
        "inherited_value": "",
        "effective_display": effective_display,
        "source": "launch field",
        "explicit": True,
        "inheritable": False,
        "active": active,
        "detail": detail,
        "impact": impact,
        "unit": unit,
    }
    if options is not None:
        control["options"] = options
    if minimum is not None:
        control["min"] = minimum
    if maximum is not None:
        control["max"] = maximum
    if step is not None:
        control["step"] = step
    return control


def _group(group_id: str, label: str, summary: str, controls: List[Dict[str, Any]]) -> Dict[str, Any]:
    active = sum(1 for item in controls if item.get("active"))
    explicit = sum(1 for item in controls if item.get("explicit") and item.get("target_type") == "env")
    return {
        "id": group_id,
        "label": label,
        "summary": summary,
        "active": active,
        "explicit_overrides": explicit,
        "controls": controls,
    }


def build_gate_catalog(spec: LaunchSpec) -> Dict[str, Any]:
    effective = build_effective_config(spec)
    reid_enabled = _get(effective, ("models", "reid", "enable"), True)
    pose_enabled = _get(effective, ("models", "pose", "enable"), True)
    trails_enabled = _get(effective, ("visualization", "trails", "enabled"), True)
    rtsp_enabled = _get(effective, ("mosaic_output", "rtsp_enabled"), True)
    webrtc_enabled = _get(effective, ("mosaic_output", "mosaic_webrtc_enabled"), True)

    groups = [
        _group(
            "identity",
            "Identity",
            "StableID, ReID, and pose feature gates that affect object continuity.",
            [
                _env_bool_control(
                    spec,
                    control_id="reid",
                    label="ReID",
                    target="NOESIS_REID_ENABLED",
                    group="identity",
                    inherited=reid_enabled,
                    source_default="pipeline",
                    detail="StableID manager integration for tracking telemetry.",
                    impact="Disabling this keeps detector/tracker output but removes appearance-backed stable IDs.",
                ),
                _env_bool_control(
                    spec,
                    control_id="pose_features",
                    label="Pose Features",
                    target="NOESIS_POSE_FEATURES_ENABLED",
                    group="identity",
                    inherited=pose_enabled,
                    source_default="pipeline",
                    detail="Object-level pose feature metadata used by StableID and world anchors.",
                    impact="Disabling this keeps pose SGIE configured but suppresses pose feature attachment.",
                ),
                _env_bool_control(
                    spec,
                    control_id="reid_pose",
                    label="ReID Pose Signal",
                    target="NOESIS_REID_POSE_ENABLED",
                    group="identity",
                    inherited=pose_enabled,
                    source_default="runtime derived",
                    detail="Pose-assisted StableID matching signal.",
                    impact="Use this to force pose-assisted identity matching on or off for a launch.",
                ),
            ],
        ),
        _group(
            "depth",
            "Depth",
            "MapAnything burst gating and baseline depth-tracking mode.",
            [
                _field_control(
                    control_id="tracking_mode",
                    label="Tracking Mode",
                    target="tracking_mode",
                    group="depth",
                    kind="select",
                    value=spec.tracking_mode,
                    detail="Baseline uses the DAv2 depth-tracking lane; v3dt disables it.",
                    impact="Changing this changes the materialized tracking/depth graph.",
                    options=[{"value": "baseline", "label": "baseline"}, {"value": "v3dt", "label": "v3dt"}],
                ),
                _field_control(
                    control_id="depth_window",
                    label="Startup Depth Burst",
                    target="depth_enable_seconds",
                    group="depth",
                    kind="number",
                    value=int(spec.depth_enable_seconds),
                    detail="Open the gated MapAnything branch after launch.",
                    impact="Longer windows spend more GPU on full-frame depth at startup.",
                    minimum=0,
                    maximum=300,
                    step=1,
                    unit="s",
                ),
                _env_number_control(
                    spec,
                    control_id="mapanything_prime",
                    label="MapAnything Prime",
                    target="NOESIS_MAPANYTHING_GATE_PRIME_SECONDS",
                    group="depth",
                    inherited="1.0",
                    source_default="runtime default",
                    detail="Short preroll window before the MapAnything valve closes.",
                    impact="Raise only if the depth branch needs more time to preroll cleanly.",
                    minimum=0,
                    maximum=15,
                    step=0.25,
                    unit="s",
                ),
            ],
        ),
        _group(
            "mosaic",
            "Mosaic",
            "RTSP and WebRTC delivery gates for the mosaic output.",
            [
                _env_bool_control(
                    spec,
                    control_id="rtsp",
                    label="RTSP Output",
                    target="NOESIS_MOSAIC_RTSP_ENABLED",
                    group="mosaic",
                    inherited=rtsp_enabled,
                    source_default="pipeline",
                    detail="Build the DS8 RTSP mosaic branch.",
                    impact="WebRTC needs RTSP available because the gateway consumes the local mosaic stream.",
                ),
                _env_bool_control(
                    spec,
                    control_id="webrtc",
                    label="WebRTC Gateway",
                    target="NOESIS_MOSAIC_WEBRTC_ENABLED",
                    group="mosaic",
                    inherited=webrtc_enabled,
                    source_default="pipeline",
                    detail="Start the RTSP-to-WebRTC mosaic gateway with DS8.",
                    impact="Disable for headless smoke tests or when another gateway owns the port.",
                ),
                _field_control(
                    control_id="rtsp_port",
                    label="RTSP Port",
                    target="rtsp_port",
                    group="mosaic",
                    kind="number",
                    value=int(spec.rtsp_port),
                    detail="Local RTSP server port for the DS8 mosaic.",
                    impact="Choose an unused port before starting alongside another runtime.",
                    minimum=1024,
                    maximum=65535,
                    step=1,
                ),
            ],
        ),
        _group(
            "visuals",
            "Visuals",
            "Overlay gates that affect what operators see in the mosaic and BEV layers.",
            [
                _env_bool_control(
                    spec,
                    control_id="trails",
                    label="Trails",
                    target="NOESIS_TRAILS_RENDER",
                    group="visuals",
                    inherited=trails_enabled,
                    source_default="pipeline",
                    detail="Render motion trails in DS8 overlays.",
                    impact="Disabling trails keeps tracking telemetry but removes trail rendering.",
                ),
                _field_control(
                    control_id="strict_baseline",
                    label="Strict Depth Anchors",
                    target="strict_baseline",
                    group="visuals",
                    kind="boolean",
                    value=bool(spec.strict_baseline),
                    detail="Treat baseline object-depth contract mismatches as launch blockers.",
                    impact="Use this for strict fused-depth baselines; detector-only PGIEs may block.",
                ),
            ],
        ),
    ]

    controls = [control for group in groups for control in group["controls"]]
    explicit_overrides = [control for control in controls if control.get("explicit") and control.get("target_type") == "env"]
    active = [control for control in controls if control.get("active")]
    return {
        "schema_version": 1,
        "spec": {
            "pgie_profile": spec.pgie_profile,
            "size": spec.size,
            "tracking_mode": spec.tracking_mode,
            "depth_enable_seconds": int(spec.depth_enable_seconds),
            "strict_baseline": bool(spec.strict_baseline),
            "rtsp_port": int(spec.rtsp_port),
        },
        "summary": {
            "groups": len(groups),
            "total": len(controls),
            "active": len(active),
            "explicit_overrides": len(explicit_overrides),
            "profile": spec.pgie_profile,
            "size": spec.size,
            "tracking_mode": spec.tracking_mode,
        },
        "groups": groups,
        "controls": controls,
    }
