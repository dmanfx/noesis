from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from noesis.dev_console.launch_spec import LaunchSpec


_SEVERITY_RANK = {"block": 0, "warn": 1, "info": 2}


def _port_owner_summary(diagnostics: Mapping[str, Any], code: str) -> str:
    try:
        port = int(str(code).split(".")[1])
    except Exception:
        return ""
    for item in diagnostics.get("ports", []):
        if not isinstance(item, Mapping) or int(item.get("port") or 0) != port:
            continue
        owner = item.get("owner") if isinstance(item.get("owner"), Mapping) else {}
        users = owner.get("users") or owner.get("raw") or ""
        return f"{item.get('label', 'Port')} {port} owner: {users}" if users else f"{item.get('label', 'Port')} {port} is busy."
    return ""


def _suggested_ports_text(diagnostics: Mapping[str, Any]) -> str:
    suggestions = diagnostics.get("suggested_ports") if isinstance(diagnostics.get("suggested_ports"), Mapping) else {}
    if not suggestions:
        return ""
    return "Suggested free ports: WS {ws_port}, REST {rest_port}, RTSP {rtsp_port}.".format(
        ws_port=suggestions.get("ws_port", "-"),
        rest_port=suggestions.get("rest_port", "-"),
        rtsp_port=suggestions.get("rtsp_port", "-"),
    )


def _action(
    actions: List[Dict[str, Any]],
    *,
    severity: str,
    title: str,
    reason: str,
    action: str,
    target: str = "",
    command: str = "",
    related_codes: Optional[List[str]] = None,
) -> None:
    actions.append(
        {
            "severity": severity,
            "title": title,
            "reason": reason,
            "action": action,
            "target": target,
            "command": command,
            "related_codes": related_codes or [],
        }
    )


def _from_validation(
    spec: LaunchSpec,
    validation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    results = validation.get("results") if isinstance(validation.get("results"), list) else []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        severity = str(item.get("severity") or "info")
        if severity not in {"block", "warn"}:
            continue
        code = str(item.get("code") or "")
        message = str(item.get("message") or "")
        hint = str(item.get("fix_hint") or "")
        if code.startswith("port.") and code.endswith(".busy"):
            _action(
                actions,
                severity="block",
                title="Resolve selected port conflict",
                reason=_port_owner_summary(diagnostics, code) or message,
                action=(
                    "Use Find Free Ports or edit the WS/REST/RTSP fields before Start. "
                    "If the owner is the active DS8 runtime you want to keep, leave it running and launch the console run on alternate ports."
                ),
                target="ports",
                related_codes=[code],
            )
            continue
        if code == "metadata.object_depth.missing_mask":
            _action(
                actions,
                severity=severity,
                title="Choose a segmentation PGIE for strict depth baselines",
                reason=message,
                action=(
                    "Switch PGIE to yolo11_seg, yolo26_seg, or rfdetr_seg when baseline object-depth anchors must be strict. "
                    "For intentional detector-only runs, keep strict baseline disabled and expect missing-mask depth warnings."
                ),
                target="pgie_profile",
                related_codes=[code],
            )
            continue
        if code.startswith("depth_registration."):
            _action(
                actions,
                severity=severity,
                title="Rebuild depth registration",
                reason=message,
                action="Rebuild the baseline DAv2-to-MapAnything registration artifact, then validate again.",
                target="depth_registration",
                command="env CUDA_VISIBLE_DEVICES='' python3 scripts/build_depth_registration.py --output config/depth_registration.json",
                related_codes=[code],
            )
            continue
        if code.endswith(".missing") and code.startswith("model."):
            _action(
                actions,
                severity=severity,
                title="Build or restore selected model artifact",
                reason=message,
                action="Build, download, or point the selected model profile at the required config/engine artifact, then refresh the launch plan.",
                target="artifacts",
                related_codes=[code],
            )
            continue
        if code.startswith("cuda.preflight"):
            _action(
                actions,
                severity=severity,
                title="Use the DS8 runtime environment",
                reason=message,
                action="Run validation from the DeepStream/CUDA environment that exposes the DS8 runtime libraries.",
                target="cuda",
                related_codes=[code],
            )
            continue
        _action(
            actions,
            severity=severity,
            title=code or "Validation finding",
            reason=message,
            action=hint or "Inspect the finding and correct the selected launch inputs before Start.",
            target="validation",
            related_codes=[code] if code else [],
        )
    if spec.strict_baseline and spec.tracking_mode == "baseline" and not any(a["target"] == "pgie_profile" for a in actions):
        _action(
            actions,
            severity="info",
            title="Strict baseline is enabled",
            reason="Strict baseline mode treats depth-fusion contract mismatches as launch blockers.",
            action="Use a segmentation PGIE and current depth registration when strict baseline is enabled.",
            target="strict_baseline",
        )
    return actions


def _from_diagnostics(diagnostics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    artifacts = diagnostics.get("artifacts") if isinstance(diagnostics.get("artifacts"), Mapping) else {}
    missing_items = [item for item in artifacts.get("items", []) if isinstance(item, Mapping) and not item.get("exists")]
    if missing_items:
        labels = ", ".join(str(item.get("label") or item.get("path") or "artifact") for item in missing_items[:5])
        _action(
            actions,
            severity="block",
            title="Restore missing launch artifacts",
            reason=f"{len(missing_items)} referenced artifact(s) are missing: {labels}",
            action="Restore or rebuild the missing files, then refresh the launch plan and rerun validation.",
            target="artifacts",
        )

    processes = diagnostics.get("processes") if isinstance(diagnostics.get("processes"), list) else []
    external_ds8_runtime_seen = False
    for process in processes:
        if not isinstance(process, Mapping):
            continue
        ports = process.get("ports") if isinstance(process.get("ports"), list) else []
        non_console_ports = [p for p in ports if isinstance(p, Mapping) and p.get("label") != "Console"]
        selected_ports = [p for p in non_console_ports if p.get("selected", True)]
        if process.get("looks_like_ds8") and non_console_ports and not process.get("managed_by_console"):
            external_ds8_runtime_seen = True
        if process.get("looks_like_ds8") and selected_ports and not process.get("managed_by_console"):
            port_text = ", ".join(f"{p.get('label')}:{p.get('port')}" for p in selected_ports)
            _action(
                actions,
                severity="warn",
                title="External DS8 runtime owns selected ports",
                reason=f"PID {process.get('pid')} owns {port_text}.",
                action=(
                    "Leave the live runtime alone unless you intentionally want to take it down. "
                    "For a console-owned launch, choose alternate free ports first."
                ),
                target="processes",
            )
            break
    else:
        if external_ds8_runtime_seen:
            runtime = next(
                (
                    process
                    for process in processes
                    if isinstance(process, Mapping) and process.get("looks_like_ds8") and not process.get("managed_by_console")
                ),
                {},
            )
            ports = runtime.get("ports") if isinstance(runtime.get("ports"), list) else []
            port_text = ", ".join(
                f"{p.get('label')}:{p.get('port')}" for p in ports if isinstance(p, Mapping) and p.get("label") != "Console"
            )
            _action(
                actions,
                severity="warn",
                title="External DS8 runtime is already running",
                reason=f"PID {runtime.get('pid')} owns {port_text or 'no mapped listening ports'}.",
                action=(
                    "Review the process inventory before launching from the console. "
                    "If the live runtime should stay up, choose a profile and ports that will not compete with it."
                ),
                target="processes",
            )

    gpu = diagnostics.get("gpu") if isinstance(diagnostics.get("gpu"), Mapping) else {}
    if not gpu.get("available"):
        _action(
            actions,
            severity="warn",
            title="GPU visibility is uncertain",
            reason=str(gpu.get("error") or "nvidia-smi did not report an available GPU."),
            action="Confirm the NVIDIA/DeepStream environment before attempting a GPU-backed DS8 launch.",
            target="gpu",
        )
    return actions


def _dedupe(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for action in actions:
        key = (str(action.get("severity")), str(action.get("title")), str(action.get("target")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return sorted(unique, key=lambda item: (_SEVERITY_RANK.get(str(item.get("severity")), 9), str(item.get("title"))))


def build_remediation(
    spec: LaunchSpec,
    *,
    validation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    launch_plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    actions = _dedupe(_from_validation(spec, validation, diagnostics) + _from_diagnostics(diagnostics))
    block_count = sum(1 for item in actions if item.get("severity") == "block")
    warn_count = sum(1 for item in actions if item.get("severity") == "warn")
    status = "blocked" if block_count else "attention" if warn_count else "ready"
    suggestions = _suggested_ports_text(diagnostics)
    if status == "ready":
        summary = "No blocking remediation is needed for the selected launch."
    elif block_count:
        summary = f"{block_count} blocking action(s) should be resolved before Start."
    else:
        summary = f"{warn_count} warning action(s) should be reviewed before Start."
    if suggestions and any(item.get("target") == "ports" for item in actions):
        summary = f"{summary} {suggestions}"
    plan_summary = ""
    if isinstance(launch_plan, Mapping):
        plan_summary = f"{launch_plan.get('change_count', 0)} tracked config change(s), {launch_plan.get('artifacts', {}).get('missing', 0)} missing artifact(s)."
    return {
        "status": status,
        "summary": summary,
        "block_count": block_count,
        "warn_count": warn_count,
        "actions": actions,
        "plan_summary": plan_summary,
    }
