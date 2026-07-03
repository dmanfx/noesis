from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noesis.ds8_preflight import (
    REPO_ROOT,
    mask_output_available,
    parse_nvinfer_ini,
    validate_metadata_compatibility,
)
from noesis.dev_console.diagnostics import artifact_audit
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import PROFILE_DEFAULT_SIZE, build_effective_config


PROFILE_SIZE_OPTIONS = {
    "yolo11_seg": ["n", "s", "m", "l", "x"],
    "yolo11": ["n", "s", "m", "l", "x"],
    "yolo26_seg": ["n", "s", "m", "l", "x"],
    "yolo26": ["n", "s", "m", "l", "x"],
    "rfdetr_seg": ["n", "s", "m"],
    "rfdetr": ["n", "s", "m"],
    "wholebody49": ["s", "x"],
}


def _relpath(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    if "://" in text:
        return text
    path = Path(text)
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return text
    return text


def _counts(results: List[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"block": 0, "warn": 0, "info": 0}
    for result in results:
        severity = str(result.get("severity") or "info")
        if severity in counts:
            counts[severity] += 1
    return counts


def _artifact_entry(artifacts: Mapping[str, Any], label: str) -> Dict[str, Any]:
    for item in artifacts.get("items", []):
        if isinstance(item, Mapping) and item.get("label") == label:
            return dict(item)
    return {}


def _model_kind(profile: str, *, mask_available: bool, mode: Optional[str]) -> str:
    if profile == "wholebody49":
        return "boxes" if mode == "boxes" else "masks"
    if profile.endswith("_seg") or mask_available:
        return "masks"
    return "boxes"


def _row_status(*, artifacts_missing: int, counts: Mapping[str, int]) -> str:
    if artifacts_missing or int(counts.get("block", 0) or 0):
        return "blocked"
    if int(counts.get("warn", 0) or 0):
        return "warn"
    return "ready"


def _matrix_row(base_spec: LaunchSpec, *, profile: str, size: str) -> Dict[str, Any]:
    spec = replace(base_spec, pgie_profile=profile, size=size, materialized_pipeline=None)
    try:
        effective = build_effective_config(spec)
        artifacts = artifact_audit(spec)
        validations = [
            item.to_dict()
            for item in validate_metadata_compatibility(
                effective,
                spec.pipeline_path,
                tracking_mode=spec.tracking_mode,
                strict_baseline=spec.strict_baseline,
            )
        ]
    except Exception as exc:
        return {
            "id": f"{profile}:{size}",
            "pgie_profile": profile,
            "size": size,
            "status": "blocked",
            "active": profile == base_spec.pgie_profile and size == (base_spec.size or ""),
            "preferred": size == PROFILE_DEFAULT_SIZE.get(profile),
            "kind": "unknown",
            "error": str(exc),
            "counts": {"block": 1, "warn": 0, "info": 0},
            "artifacts": {"total": 0, "missing": 1, "ready": False, "missing_labels": ["matrix build"]},
            "apply_spec": {"pgie_profile": profile, "size": size, "tracking_mode": base_spec.tracking_mode},
        }

    models = effective.get("models") if isinstance(effective.get("models"), Mapping) else {}
    pgie = models.get("pgie") if isinstance(models.get("pgie"), Mapping) else {}
    pgie_config = str(pgie.get("config-file-path") or "")
    pgie_engine = str(pgie.get("engine") or "")
    pgie_props: Dict[str, str] = {}
    mask_available = False
    pgie_config_path = Path(pgie_config)
    if pgie_config_path.exists():
        pgie_props = parse_nvinfer_ini(pgie_config_path)
        mask_available = mask_output_available(pgie_props)

    counts = _counts(validations)
    missing_items = [
        item
        for item in artifacts.get("items", [])
        if isinstance(item, Mapping) and not item.get("exists")
    ]
    pgie_engine_entry = _artifact_entry(artifacts, "pgie engine")
    pgie_config_entry = _artifact_entry(artifacts, "pgie config")
    status = _row_status(artifacts_missing=int(artifacts.get("missing", 0) or 0), counts=counts)
    return {
        "id": f"{profile}:{size}",
        "pgie_profile": profile,
        "size": size,
        "status": status,
        "active": profile == base_spec.pgie_profile and size == (base_spec.size or ""),
        "preferred": size == PROFILE_DEFAULT_SIZE.get(profile),
        "kind": _model_kind(profile, mask_available=mask_available, mode=pgie.get("wholebody49_mode")),
        "mask_available": mask_available,
        "network_type": pgie_props.get("network-type", ""),
        "config": _relpath(pgie_config),
        "engine": _relpath(pgie_engine),
        "engine_exists": bool(pgie_engine_entry.get("exists")),
        "engine_size_bytes": pgie_engine_entry.get("size_bytes"),
        "config_exists": bool(pgie_config_entry.get("exists")),
        "counts": counts,
        "artifacts": {
            "total": artifacts.get("total", 0),
            "missing": artifacts.get("missing", 0),
            "ready": artifacts.get("ready", False),
            "missing_labels": [str(item.get("label") or item.get("path") or "artifact") for item in missing_items[:8]],
        },
        "top_findings": validations[:4],
        "apply_spec": {"pgie_profile": profile, "size": size, "tracking_mode": base_spec.tracking_mode},
    }


def build_model_matrix(spec: LaunchSpec) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for profile, sizes in PROFILE_SIZE_OPTIONS.items():
        for size in sizes:
            rows.append(_matrix_row(spec, profile=profile, size=size))

    summary = {
        "total": len(rows),
        "ready": sum(1 for row in rows if row.get("status") == "ready"),
        "warn": sum(1 for row in rows if row.get("status") == "warn"),
        "blocked": sum(1 for row in rows if row.get("status") == "blocked"),
    }
    return {
        "summary": summary,
        "rows": rows,
        "active_id": f"{spec.pgie_profile}:{spec.size or ''}",
        "tracking_mode": spec.tracking_mode,
        "pipeline_config": spec.pipeline_config,
        "strict_baseline": spec.strict_baseline,
    }
