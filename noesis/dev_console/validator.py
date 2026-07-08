from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from noesis.ds8_preflight import Severity, ValidationResult, has_blocking, run_preflight, validate_ports
from noesis.dev_console.diagnostics import diagnostics_snapshot, noesis_port_ownership
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import materialize_launch_pipeline


def _block_from_exception(code: str, exc: Exception) -> ValidationResult:
    return ValidationResult(
        severity=Severity.BLOCK,
        code=code,
        message=str(exc),
        fix_hint="Correct the launch spec or materialize the required DS8 assets.",
    )


def _port_from_validation_code(code: str) -> Optional[int]:
    match = re.match(r"^port\.(\d+)\.busy$", code)
    if not match:
        return None
    return int(match.group(1))


def normalize_port_validation_results(
    validation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    ownership = noesis_port_ownership(diagnostics)
    if not ownership:
        return dict(validation)

    results = validation.get("results") if isinstance(validation.get("results"), list) else []
    normalized = []
    changed = False
    for item in results:
        if not isinstance(item, Mapping):
            normalized.append(item)
            continue
        next_item = dict(item)
        port = _port_from_validation_code(str(next_item.get("code") or ""))
        owner = ownership.get(port or -1)
        if owner and next_item.get("severity") == "block":
            if owner.get("managed_by_console"):
                next_item.update(
                    {
                        "severity": "info",
                        "code": f"port.{port}.managed",
                        "message": (
                            f"Port {port} is owned by the console-managed DS8 runtime (pid {owner.get('pid')})."
                        ),
                        "fix_hint": "Restart will stop the supervised runtime before relaunching the selected spec.",
                    }
                )
            else:
                next_item.update(
                    {
                        "severity": "info",
                        "code": f"port.{port}.ds8_runtime",
                        "message": (
                            f"Port {port} is in use by active DS8 runtime pid {owner.get('pid')}. "
                            "This is expected while monitoring an external Noesis pipeline."
                        ),
                        "fix_hint": (
                            "No action is required for monitoring. Choose alternate ports only when starting "
                            "a separate console-owned launch."
                        ),
                    }
                )
            changed = True
        normalized.append(next_item)

    if not changed:
        return dict(validation)

    counts = {
        "block": sum(1 for item in normalized if isinstance(item, Mapping) and item.get("severity") == "block"),
        "warn": sum(1 for item in normalized if isinstance(item, Mapping) and item.get("severity") == "warn"),
        "info": sum(1 for item in normalized if isinstance(item, Mapping) and item.get("severity") == "info"),
    }
    payload = dict(validation)
    payload["results"] = normalized
    payload["counts"] = counts
    payload["blocking"] = counts["block"] > 0
    return payload


def validate_launch(
    spec: LaunchSpec,
    *,
    managed_pid: Optional[int] = None,
    normalize_noesis_ports: bool = True,
) -> Dict[str, Any]:
    results = []
    materialized = None
    try:
        materialized = materialize_launch_pipeline(spec, dry_run=True)
        results.extend(
            run_preflight(
                pipeline_path=materialized,
                tracking_mode=spec.tracking_mode,
                strict_baseline=spec.strict_baseline,
            )
        )
        results.extend(validate_ports(spec.ws_host, [int(spec.ws_port)]))
        if spec.enable_rest:
            results.extend(validate_ports(spec.rest_host, [int(spec.rest_port)]))
        results.extend(validate_ports("127.0.0.1", [int(spec.rtsp_port)]))
    except Exception as exc:
        results.append(_block_from_exception("launch.materialize_failed", exc))

    payload = {
        "blocking": has_blocking(results),
        "materialized_pipeline": str(materialized) if materialized else None,
        "results": [result.to_dict() for result in results],
        "counts": {
            "block": sum(1 for item in results if item.severity == Severity.BLOCK),
            "warn": sum(1 for item in results if item.severity == Severity.WARN),
            "info": sum(1 for item in results if item.severity == Severity.INFO),
        },
    }
    if normalize_noesis_ports:
        diagnostics = diagnostics_snapshot(spec, managed_pid=managed_pid)
        payload = normalize_port_validation_results(payload, diagnostics)
    return payload
