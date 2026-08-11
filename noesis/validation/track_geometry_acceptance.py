"""Unified offline admission for canonical tracks against authored and rendered geometry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .authored_scene import (
    GeometryThresholds,
    build_journal_oracle_report,
    load_similarity,
)
from .core import (
    CheckStatus,
    ConfidenceScores,
    FailureType,
    SourceMetadata,
    ValidationCheck,
    ValidationReport,
    worst_status,
)
from .menon_trace import build_menon_trace_report, load_menon_trace


ACCEPTANCE_CONTRACT = "noesis.validation.track_geometry_acceptance"
ACCEPTANCE_CONTRACT_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRODUCTION_RENDERER_CHECK_IDS = (
    "MENON.production_geometry_contract",
    "MENON.production_geometry_transform_authority",
    "MENON.production_renderer_geometry",
    "MENON.production_stage_visibility",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound_room_group_map(
    path: str | Path,
    *,
    authored_scene_path: str | Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Load a reviewed room map only when it is bound to the exact OBJ bytes."""

    source = Path(path).expanduser().resolve()
    scene = Path(authored_scene_path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("room group map must be a JSON object")
    declared = str(payload.get("authored_scene_sha256") or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(declared):
        raise ValueError(
            "room group map must declare a lowercase 64-character authored_scene_sha256"
        )
    actual = sha256_file(scene)
    if declared != actual:
        raise ValueError(
            "room group map authored_scene_sha256 does not match the supplied OBJ "
            f"(declared {declared}, actual {actual})"
        )
    raw_rooms = payload.get("rooms")
    if not isinstance(raw_rooms, Mapping) or not raw_rooms:
        raise ValueError("room group map must contain a non-empty rooms object")
    rooms: dict[str, tuple[str, ...]] = {}
    for raw_label, raw_groups in raw_rooms.items():
        label = str(raw_label).strip()
        if (
            not label
            or isinstance(raw_groups, (str, bytes, bytearray))
            or not isinstance(raw_groups, Sequence)
        ):
            raise ValueError(f"room group map entry {raw_label!r} is invalid")
        groups = tuple(
            dict.fromkeys(str(group).strip() for group in raw_groups if str(group).strip())
        )
        if not groups:
            raise ValueError(f"room group map entry {raw_label!r} has no OBJ groups")
        rooms[label] = groups
    return rooms, {
        "declared_authored_scene_sha256": declared,
        "actual_authored_scene_sha256": actual,
        "room_count": len(rooms),
        "release_id": payload.get("release_id"),
    }


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _mapping_path(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _trace_world_to_scene(payload: Mapping[str, Any]) -> tuple[float, ...] | None:
    for keys in (
        ("world_to_menon_col_major",),
        ("world_to_scene_col_major",),
        ("align", "scene_similarity", "world_to_scene_col_major"),
    ):
        candidate = _mapping_path(payload, keys)
        if len(_sequence(candidate)) != 16:
            continue
        try:
            matrix = tuple(float(item) for item in candidate)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(item) for item in matrix):
            return matrix
    return None


def _check_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(check.get("id")): check
        for check in _sequence(report.get("checks"))
        if isinstance(check, Mapping) and check.get("id")
    }


def _binding_check(
    *,
    check_id: str,
    name: str,
    status: CheckStatus,
    failure_type: FailureType | None,
    metric: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return ValidationCheck(
        id=check_id,
        domain="cohort",
        name=name,
        status=status,
        failure_type=failure_type,
        metric=metric,
        detail=detail,
    ).to_dict()


def _missing_check(check_id: str) -> dict[str, Any]:
    return ValidationCheck(
        id=check_id,
        domain="infrastructure",
        name="required_acceptance_evidence",
        status=CheckStatus.BLOCKED,
        failure_type=FailureType.INFRASTRUCTURE,
        detail=f"Required acceptance evidence {check_id} is absent.",
        suggested_next_diagnostic=(
            "Capture an occupied Menon production-geometry trace and a matching journal window."
        ),
    ).to_dict()


def _gate(
    *,
    name: str,
    checks: Iterable[Mapping[str, Any]],
    required_ids: Iterable[str] = (),
) -> dict[str, Any]:
    by_id = {str(check.get("id")): dict(check) for check in checks if check.get("id")}
    missing = [check_id for check_id in required_ids if check_id not in by_id]
    for check_id in missing:
        by_id[check_id] = _missing_check(check_id)
    ordered = [by_id[key] for key in sorted(by_id)]
    if not ordered:
        ordered = [_missing_check(f"ACCEPTANCE.{name}")]
    statuses = [str(check.get("status") or CheckStatus.BLOCKED.value) for check in ordered]
    status = worst_status(statuses)
    counts = Counter(statuses)
    return {
        "status": status.value,
        "required_check_ids": sorted(set(str(item) for item in required_ids)),
        "missing_check_ids": sorted(missing),
        "check_count": len(ordered),
        "status_counts": {
            value.value: int(counts.get(value.value, 0)) for value in CheckStatus
        },
        "checks": [
            {
                key: check[key]
                for key in (
                    "id",
                    "status",
                    "failure_type",
                    "camera",
                    "metric",
                    "threshold",
                    "detail",
                    "suggested_next_diagnostic",
                )
                if key in check
            }
            for check in ordered
        ],
    }


def _gate_check(
    gate_name: str,
    gate: Mapping[str, Any],
    *,
    failure_type: FailureType,
    detail: str,
) -> ValidationCheck:
    status = CheckStatus(str(gate.get("status") or CheckStatus.BLOCKED.value))
    return ValidationCheck(
        id=f"ACCEPTANCE.{gate_name}",
        domain="acceptance",
        name=gate_name,
        status=status,
        failure_type=None if status == CheckStatus.PASS else failure_type,
        metric={
            "check_count": gate.get("check_count"),
            "status_counts": gate.get("status_counts"),
            "missing_check_ids": gate.get("missing_check_ids"),
        },
        evidence=[
            "authored_scene_report.json",
            "menon_trace_report.json",
            "menon/trace.json",
        ],
        detail=detail,
    )


def build_track_geometry_acceptance(
    *,
    journal_path: str | Path,
    obj_path: str | Path,
    similarity_path: str | Path,
    room_group_map_path: str | Path,
    menon_trace_path: str | Path,
    run_id: str,
    cameras: Iterable[str] = (),
    journal_limit: int = 0,
    max_divergences: int = 100,
    thresholds: GeometryThresholds = GeometryThresholds(),
    horizontal_tolerance_deg: float = 15.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build one gated result plus the two unchanged component reports."""

    selected_cameras = sorted(
        {str(camera).strip() for camera in cameras if str(camera).strip()}
    )
    rooms, room_binding = load_bound_room_group_map(
        room_group_map_path,
        authored_scene_path=obj_path,
    )
    similarity = load_similarity(similarity_path)
    trace = load_menon_trace(menon_trace_path)
    authored_report = build_journal_oracle_report(
        journal_path=journal_path,
        obj_path=obj_path,
        similarity_path=similarity_path,
        run_id=f"{run_id}:authored_scene",
        cameras=selected_cameras,
        journal_limit=max(0, int(journal_limit)),
        max_divergences=max(0, int(max_divergences)),
        thresholds=thresholds,
        horizontal_tolerance_deg=float(horizontal_tolerance_deg),
        room_group_map=rooms,
    )
    menon_report_object = build_menon_trace_report(
        trace,
        run_id=f"{run_id}:menon_trace",
    )
    menon_report = menon_report_object.to_dict()

    authored_checks = _check_map(authored_report)
    menon_checks = _check_map(menon_report)
    observed_cameras = sorted(
        str(item) for item in _sequence(authored_report.get("scope", {}).get("cameras"))
    )
    gate_cameras = selected_cameras or observed_cameras

    room_binding_check = _binding_check(
        check_id="COHORT.room_map_authored_scene",
        name="room_map_authored_scene_binding",
        status=CheckStatus.PASS,
        failure_type=None,
        metric=room_binding,
        detail="The reviewed semantic room map is bound to the exact authored OBJ bytes.",
    )
    trace_matrix = _trace_world_to_scene(trace)
    matrix_error = (
        max(abs(left - right) for left, right in zip(similarity.col_major, trace_matrix))
        if trace_matrix is not None
        else None
    )
    transform_bound = matrix_error is not None and matrix_error <= 1e-9
    transform_binding_check = _binding_check(
        check_id="COHORT.world_to_scene_transform",
        name="world_to_scene_transform_binding",
        status=CheckStatus.PASS if transform_bound else CheckStatus.FAIL,
        failure_type=None if transform_bound else FailureType.TRANSFORM,
        metric={"max_matrix_error": matrix_error},
        detail=(
            "The authored-scene oracle and Menon trace use the same world-to-scene matrix."
            if transform_bound
            else "The Menon trace transform is absent or differs from the oracle similarity."
        ),
    )

    coverage_ids = [
        f"TRACK.world_geometry_coverage.{camera_id}" for camera_id in gate_cameras
    ]
    producer_checks = [
        authored_checks[check_id]
        for check_id in coverage_ids
        if check_id in authored_checks
    ]
    if "TRACK.journal_observations" in authored_checks:
        producer_checks.append(authored_checks["TRACK.journal_observations"])
    producer_gate = _gate(
        name="producer_coverage",
        checks=producer_checks,
        required_ids=coverage_ids or ("TRACK.world_geometry_coverage.<occupied-camera>",),
    )

    physical_required = [
        "SCENE.authored_walkable_inventory",
        "TRANSFORM.world_to_scene_similarity",
        "SCENE.expected_room_authority",
        "MENON.production_physical_placement",
        *[f"MENON.authored_floor_placement.{camera_id}" for camera_id in gate_cameras],
        *[f"SEMANTIC.expected_room_containment.{camera_id}" for camera_id in gate_cameras],
    ]
    physical_checks = [room_binding_check, transform_binding_check]
    for check_id in physical_required:
        source = menon_checks if check_id.startswith("MENON.production_") else authored_checks
        if check_id in source:
            physical_checks.append(source[check_id])
    physical_gate = _gate(
        name="physical_placement",
        checks=physical_checks,
        required_ids=physical_required
        + ["COHORT.room_map_authored_scene", "COHORT.world_to_scene_transform"],
    )

    renderer_checks = [transform_binding_check]
    renderer_checks.extend(
        menon_checks[check_id]
        for check_id in _PRODUCTION_RENDERER_CHECK_IDS
        if check_id in menon_checks
    )
    renderer_gate = _gate(
        name="menon_renderer",
        checks=renderer_checks,
        required_ids=["COHORT.world_to_scene_transform", *_PRODUCTION_RENDERER_CHECK_IDS],
    )
    gates = {
        "producer_coverage": producer_gate,
        "physical_placement": physical_gate,
        "menon_renderer": renderer_gate,
    }

    source = menon_report_object.source
    report = ValidationReport(
        run_id=str(run_id),
        source=SourceMetadata(**source.to_dict()),
        scope={
            "tiers": [
                "offline",
                "persisted_journal",
                "authored_scene",
                "menon_trace",
                "production_geometry",
            ],
            "cameras": gate_cameras,
            "observation_count": authored_report.get("scope", {}).get("observation_count"),
            "world_point_count": authored_report.get("scope", {}).get("world_point_count"),
        },
        confidence=ConfidenceScores(
            projection=authored_report.get("confidence", {}).get("projection"),
            geometry=authored_report.get("confidence", {}).get("geometry"),
            semantic=authored_report.get("confidence", {}).get("semantic"),
        ),
        checks=[
            _gate_check(
                "producer_coverage",
                producer_gate,
                failure_type=FailureType.DATA_QUALITY,
                detail="Canonical producer world-position coverage is independently admitted.",
            ),
            _gate_check(
                "physical_placement",
                physical_gate,
                failure_type=FailureType.CALIBRATION,
                detail="Canonical world points agree with the bound authored scene and room authority.",
            ),
            _gate_check(
                "menon_renderer",
                renderer_gate,
                failure_type=FailureType.PROJECTION,
                detail="Menon's production marker and trail geometry exactly matches canonical placement.",
            ),
        ],
        artifacts={
            "authored_scene_report": "authored_scene_report.json",
            "menon_trace_report": "menon_trace_report.json",
            "menon_trace": "menon/trace.json",
        },
    )
    acceptance = report.to_dict()
    acceptance.update(
        {
            "contract": ACCEPTANCE_CONTRACT,
            "contract_version": ACCEPTANCE_CONTRACT_VERSION,
            "bindings": {
                "journal_sha256": sha256_file(journal_path),
                "authored_scene_sha256": room_binding["actual_authored_scene_sha256"],
                "similarity_sha256": sha256_file(similarity_path),
                "room_group_map_sha256": sha256_file(room_group_map_path),
                "menon_trace_sha256": sha256_file(menon_trace_path),
                "room_map": room_binding,
                "world_to_scene_max_matrix_error": matrix_error,
            },
            "gates": gates,
            "components": {
                "authored_scene": {
                    "artifact": "authored_scene_report.json",
                    "raw_report_summary": authored_report.get("summary"),
                    "known_non_gate_check_ids": ["MENON.rendered_track_placement"],
                },
                "menon_trace": {
                    "artifact": "menon_trace_report.json",
                    "raw_report_summary": menon_report.get("summary"),
                },
            },
        }
    )
    return acceptance, authored_report, menon_report


__all__ = [
    "ACCEPTANCE_CONTRACT",
    "ACCEPTANCE_CONTRACT_VERSION",
    "build_track_geometry_acceptance",
    "load_bound_room_group_map",
    "sha256_file",
]
