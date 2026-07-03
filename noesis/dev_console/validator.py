from __future__ import annotations

from typing import Any, Dict

from noesis.ds8_preflight import Severity, ValidationResult, has_blocking, run_preflight, validate_ports
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import materialize_launch_pipeline


def _block_from_exception(code: str, exc: Exception) -> ValidationResult:
    return ValidationResult(
        severity=Severity.BLOCK,
        code=code,
        message=str(exc),
        fix_hint="Correct the launch spec or materialize the required DS8 assets.",
    )


def validate_launch(spec: LaunchSpec) -> Dict[str, Any]:
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

    return {
        "blocking": has_blocking(results),
        "materialized_pipeline": str(materialized) if materialized else None,
        "results": [result.to_dict() for result in results],
        "counts": {
            "block": sum(1 for item in results if item.severity == Severity.BLOCK),
            "warn": sum(1 for item in results if item.severity == Severity.WARN),
            "info": sum(1 for item in results if item.severity == Severity.INFO),
        },
    }
