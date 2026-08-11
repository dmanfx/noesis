"""Independent checks for Menon's bounded production track-geometry snapshot."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .core import CheckStatus, FailureType, ValidationCheck


PRODUCTION_GEOMETRY_CONTRACT = "menon.production-track-geometry-debug"
PRODUCTION_GEOMETRY_VERSION = 1
_REQUIRED_STRUCTURAL_CHECKS = frozenset(
    {
        "canonical_frame_contract",
        "production_visual_present",
        "world_to_scene_exact",
        "visual_root_identity",
        "rendered_trail_endpoint_exact",
        "rendered_trail_world_endpoint_exact",
        "decorator_input_endpoint_exact",
        "decorator_world_anchor_alignment",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _finite_vector3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"), value.get("z"))
    if len(_sequence(value)) < 3:
        return None
    try:
        result = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _finite_matrix(value: Any) -> tuple[float, ...] | None:
    if len(_sequence(value)) != 16:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _distance3(left: Any, right: Any) -> float | None:
    a = _finite_vector3(left)
    b = _finite_vector3(right)
    if a is None or b is None:
        return None
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _distance_xz(left: Any, right: Any) -> float | None:
    a = _finite_vector3(left)
    b = _finite_vector3(right)
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _recomputed_structural_error(check: Mapping[str, Any]) -> tuple[float | None, float | None]:
    code = str(check.get("code") or "")
    if code == "world_to_scene_exact":
        error = _distance3(check.get("expectedScenePoint"), check.get("actualScenePoint"))
    elif code in {
        "rendered_trail_endpoint_exact",
        "rendered_trail_world_endpoint_exact",
        "decorator_input_endpoint_exact",
    }:
        error = _distance3(check.get("expectedEndpoint"), check.get("actualEndpoint"))
    elif code == "decorator_world_anchor_alignment":
        error = _distance_xz(check.get("trailEndpointWorld"), check.get("decoratorAnchorWorld"))
    elif code == "visual_root_identity":
        try:
            error = float(check.get("maxElementError"))
        except (TypeError, ValueError):
            error = None
    else:
        return None, None
    try:
        tolerance = float(check.get("toleranceScene"))
    except (TypeError, ValueError):
        tolerance = None
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
        tolerance = None
    return error, tolerance


def validate_menon_production_geometry(
    payload: Mapping[str, Any] | None,
    *,
    expected_world_to_scene: Sequence[float] | None = None,
) -> list[ValidationCheck]:
    """Validate exact producer-to-render placement and physical scene admission.

    Menon computes a bounded diagnostic envelope from the same objects used by
    the production renderer.  These checks independently verify its contract,
    recompute the numeric endpoint errors, and keep renderer correctness
    separate from upstream physical placement.
    """

    geometry = _mapping(payload)
    if not geometry:
        return [
            ValidationCheck(
                id="MENON.production_geometry_present",
                domain="menon",
                name="production_geometry_present",
                status=CheckStatus.BLOCKED,
                failure_type=FailureType.INFRASTRUCTURE,
                detail="The browser trace has no bounded production-geometry snapshot.",
                suggested_next_diagnostic=(
                    "Capture getMenonTrackDebugSnapshot().productionGeometry from the deployed Menon build."
                ),
            )
        ]

    checks: list[ValidationCheck] = []
    contract_ok = (
        geometry.get("contract") == PRODUCTION_GEOMETRY_CONTRACT
        and geometry.get("contractVersion") == PRODUCTION_GEOMETRY_VERSION
        and geometry.get("bounded") is True
    )
    checks.append(
        ValidationCheck(
            id="MENON.production_geometry_contract",
            domain="menon",
            name="production_geometry_contract",
            status=CheckStatus.PASS if contract_ok else CheckStatus.FAIL,
            failure_type=None if contract_ok else FailureType.INFRASTRUCTURE,
            metric={
                "contract": geometry.get("contract"),
                "contract_version": geometry.get("contractVersion"),
                "bounded": geometry.get("bounded"),
            },
            detail=(
                "Menon exposed the bounded production geometry contract."
                if contract_ok
                else "Menon's production geometry envelope is missing or has an unsupported contract."
            ),
        )
    )

    actual_matrix = _finite_matrix(geometry.get("worldToSceneMatrix"))
    expected_matrix = _finite_matrix(expected_world_to_scene)
    matrix_error = None
    if actual_matrix is not None and expected_matrix is not None:
        matrix_error = max(abs(a - b) for a, b in zip(actual_matrix, expected_matrix))
    matrix_ok = actual_matrix is not None and (
        expected_matrix is None or (matrix_error is not None and matrix_error <= 1e-9)
    )
    checks.append(
        ValidationCheck(
            id="MENON.production_geometry_transform_authority",
            domain="menon",
            name="production_geometry_transform_authority",
            status=CheckStatus.PASS if matrix_ok else CheckStatus.FAIL,
            failure_type=None if matrix_ok else FailureType.TRANSFORM,
            metric={"max_matrix_error": matrix_error},
            threshold={"max_matrix_error": 1e-9},
            detail=(
                "The production renderer and captured canonical trace use the same scene transform."
                if matrix_ok
                else "The production geometry transform is absent or differs from the canonical trace."
            ),
        )
    )

    validations = [item for item in _sequence(geometry.get("validations")) if isinstance(item, Mapping)]
    truncated = geometry.get("validationTruncated") is True
    structural_failures: list[dict[str, Any]] = []
    physical_statuses: Counter[str] = Counter()
    max_recomputed_error = 0.0
    for validation in validations:
        identity = dict(_mapping(validation.get("identity")))
        by_code = {
            str(item.get("code")): item
            for item in _sequence(validation.get("checks"))
            if isinstance(item, Mapping) and item.get("code")
        }
        missing = sorted(_REQUIRED_STRUCTURAL_CHECKS.difference(by_code))
        failed_codes = sorted(
            code
            for code in _REQUIRED_STRUCTURAL_CHECKS.intersection(by_code)
            if str(by_code[code].get("status")) != "pass"
        )
        numeric_failures: list[str] = []
        for code in _REQUIRED_STRUCTURAL_CHECKS.intersection(by_code):
            error, tolerance = _recomputed_structural_error(by_code[code])
            if error is not None:
                max_recomputed_error = max(max_recomputed_error, float(error))
            if error is not None and tolerance is not None and error > tolerance:
                numeric_failures.append(code)
        if (
            validation.get("contract") != "menon.track.scene.validation"
            or validation.get("contractVersion") != 1
            or validation.get("structuralStatus") != "pass"
            or missing
            or failed_codes
            or numeric_failures
        ):
            structural_failures.append(
                {
                    "identity": identity,
                    "missing_checks": missing,
                    "failed_checks": failed_codes,
                    "numeric_failures": sorted(numeric_failures),
                    "declared_status": validation.get("structuralStatus"),
                }
            )
        physical_statuses[str(validation.get("physicalStatus") or "unavailable")] += 1

    if not validations:
        structural_status = CheckStatus.BLOCKED
        structural_failure_type = FailureType.INFRASTRUCTURE
        structural_detail = "No active canonical tracks were available for production renderer validation."
    elif truncated or structural_failures:
        structural_status = CheckStatus.FAIL
        structural_failure_type = FailureType.PROJECTION
        structural_detail = "At least one canonical track did not survive the exact Menon production render path."
    else:
        structural_status = CheckStatus.PASS
        structural_failure_type = None
        structural_detail = "Every captured canonical track matched its actual Menon marker and trail endpoint."
    checks.append(
        ValidationCheck(
            id="MENON.production_renderer_geometry",
            domain="menon",
            name="production_renderer_geometry",
            status=structural_status,
            failure_type=structural_failure_type,
            metric={
                "validation_count": len(validations),
                "failure_count": len(structural_failures),
                "truncated": truncated,
                "max_recomputed_error_scene": max_recomputed_error,
                "failures": structural_failures[:20],
            },
            detail=structural_detail,
            suggested_next_diagnostic=(
                None
                if structural_status == CheckStatus.PASS
                else "Inspect productionGeometry.validations and stageVisibility for the first rejected entity."
            ),
        )
    )

    if not validations:
        physical_status = CheckStatus.BLOCKED
    elif physical_statuses.get("fail", 0):
        physical_status = CheckStatus.FAIL
    elif physical_statuses.get("warn", 0) or physical_statuses.get("unavailable", 0):
        physical_status = CheckStatus.WARNING
    else:
        physical_status = CheckStatus.PASS
    checks.append(
        ValidationCheck(
            id="MENON.production_physical_placement",
            domain="menon",
            name="production_physical_placement",
            status=physical_status,
            failure_type=(
                FailureType.CALIBRATION
                if physical_status in {CheckStatus.FAIL, CheckStatus.WARNING}
                else FailureType.INFRASTRUCTURE
                if physical_status == CheckStatus.BLOCKED
                else None
            ),
            metric={"validation_count": len(validations), "status_counts": dict(physical_statuses)},
            detail=(
                "All captured track points are physically plausible in the loaded scene."
                if physical_status == CheckStatus.PASS
                else "Renderer geometry is separable from unresolved physical room/floor placement."
            ),
            suggested_next_diagnostic=(
                None
                if physical_status == CheckStatus.PASS
                else "Compare the same backend-world points with the authored-scene OBJ oracle."
            ),
        )
    )

    stage = _mapping(geometry.get("stageVisibility"))
    stage_entries = [item for item in _sequence(stage.get("entries")) if isinstance(item, Mapping)]
    stage_failures: list[dict[str, Any]] = []
    canonical_count = 0
    for entry in stage_entries:
        canonical = _mapping(entry.get("canonicalWorld"))
        if canonical.get("entityPresent") is not True:
            continue
        canonical_count += 1
        presentation = _mapping(entry.get("presentation"))
        renderer = _mapping(entry.get("renderer"))
        trail = _mapping(entry.get("trail"))
        reasons = [
            value
            for value in (
                presentation.get("rejectionReason"),
                renderer.get("rejectionReason"),
                trail.get("rejectionReason"),
            )
            if value not in (None, "trail_has_no_movement_segment_yet")
        ]
        if (
            presentation.get("entityPresent") is not True
            or renderer.get("visualPresent") is not True
            or renderer.get("markerObjectPresent") is not True
            or renderer.get("markerEndpointPresent") is not True
            or renderer.get("markerVisible") is not True
            or trail.get("geometryPresent") is not True
            or trail.get("currentPointPresent") is not True
            or reasons
        ):
            stage_failures.append(
                {
                    "key": entry.get("key"),
                    "rejection_reasons": reasons,
                    "presentation_present": presentation.get("entityPresent"),
                    "visual_present": renderer.get("visualPresent"),
                    "marker_visible": renderer.get("markerVisible"),
                    "trail_present": trail.get("geometryPresent"),
                }
            )
    if canonical_count == 0:
        stage_status = CheckStatus.BLOCKED
    elif stage.get("truncated") is True or stage_failures:
        stage_status = CheckStatus.FAIL
    else:
        stage_status = CheckStatus.PASS
    checks.append(
        ValidationCheck(
            id="MENON.production_stage_visibility",
            domain="menon",
            name="production_stage_visibility",
            status=stage_status,
            failure_type=(
                None
                if stage_status == CheckStatus.PASS
                else FailureType.INFRASTRUCTURE
                if stage_status == CheckStatus.BLOCKED
                else FailureType.PROJECTION
            ),
            metric={
                "canonical_entity_count": canonical_count,
                "failure_count": len(stage_failures),
                "truncated": stage.get("truncated") is True,
                "failures": stage_failures[:20],
            },
            detail=(
                "Every canonical entity is present in the presentation, marker, and trail stages."
                if stage_status == CheckStatus.PASS
                else "At least one production visualization stage is absent or rejected."
            ),
        )
    )
    return checks
