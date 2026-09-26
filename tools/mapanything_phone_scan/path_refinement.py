"""Bounded adapter for review-only camera-path refinement.

The adapter is deliberately small.  It admits only a complete DA3 provider
result whose raw views are retained beside the provider manifest, then calls
the existing :func:`trajectory_refinement.run_trajectory_refinement` engine.
It does not smooth poses, integrate IMU samples, manufacture calibration, or
rewrite the source scan.  A materialized candidate is returned to the caller
only after the engine's visual, pose-deformation, and withheld-range gates
have all accepted it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from .trajectory_motion_review import TrajectoryMotionReviewError
from .trajectory_refinement import (
    TrajectoryRefinementError,
    TrajectoryRefinementSettings,
    run_trajectory_refinement,
)


PATH_REFINEMENT_SCHEMA = "noesis.phone_scan.path_refinement.v1"
MAX_PROVIDER_VIEWS = 256
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_RAW_VIEW_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_RAW_BYTES = 2 * 1024 * 1024 * 1024
MAX_NPZ_MEMBERS_PER_VIEW = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _new_output_dir(output_dir: Path) -> Path:
    output = output_dir.resolve()
    if output.is_symlink():
        raise ValueError("Path-refinement output must not be a symlink")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("Use a new empty path-refinement output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _bound_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise TrajectoryRefinementError(f"{label} must be a relative retained artifact")
    relative = Path(value)
    if ".." in relative.parts:
        raise TrajectoryRefinementError(f"{label} escapes the scan directory")
    declared_path = root / relative
    if declared_path.is_symlink():
        raise TrajectoryRefinementError(f"{label} is symlinked")
    path = declared_path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise TrajectoryRefinementError(f"{label} escapes the scan directory")
    return path


def _provider_manifest_path(scan_dir: Path, source: Mapping[str, Any]) -> Path:
    evidence = source.get("evidence")
    if not isinstance(evidence, Mapping):
        raise TrajectoryRefinementError("_load_provider evidence is missing")
    provider_evidence = evidence.get("provider_manifest")
    if not isinstance(provider_evidence, Mapping):
        raise TrajectoryRefinementError("_load_provider provider-manifest evidence is missing")
    path_value = provider_evidence.get("path")
    if not isinstance(path_value, str):
        raise TrajectoryRefinementError("_load_provider provider-manifest path is missing")
    declared_path = Path(path_value)
    if declared_path.is_symlink():
        raise TrajectoryRefinementError("provider manifest is symlinked")
    path = declared_path.resolve()
    if not path.is_file():
        raise TrajectoryRefinementError("provider manifest is missing or symlinked")
    expected_sha256 = provider_evidence.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise TrajectoryRefinementError("provider-manifest evidence has no SHA-256")
    if _sha256(path) != expected_sha256.lower():
        raise TrajectoryRefinementError(
            "provider manifest changed after _load_provider; exact evidence binding failed"
        )
    return path


def _declared_npz_bytes(path: Path, index: int) -> int:
    """Read only NPZ directory metadata before the engine can allocate arrays."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise TrajectoryRefinementError(
            f"DA3 raw view {index} is not a readable NPZ archive"
        ) from exc
    if not members or len(members) > MAX_NPZ_MEMBERS_PER_VIEW:
        raise TrajectoryRefinementError(
            f"DA3 raw view {index} has too many or no NPZ members"
        )
    total = 0
    for member in members:
        if member.is_dir() or not member.filename.endswith(".npy"):
            raise TrajectoryRefinementError(
                f"DA3 raw view {index} contains an unsupported NPZ member"
            )
        total += int(member.file_size)
        if total > MAX_EXPANDED_RAW_BYTES:
            raise TrajectoryRefinementError(
                f"DA3 raw view {index} exceeds the bounded expanded-array budget"
            )
    return total


def _validate_raw_pose_provenance(
    raw_root: Path, source: Mapping[str, Any], provider_count: int
) -> None:
    """Bind each retained raw view to the already-loaded provider pose order."""
    try:
        source_poses = np.asarray(source["poses"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrajectoryRefinementError("provider poses are unavailable for raw provenance binding") from exc
    if source_poses.shape != (provider_count, 4, 4) or not np.isfinite(source_poses).all():
        raise TrajectoryRefinementError("provider poses are malformed for raw provenance binding")
    for index in range(provider_count):
        path = raw_root / f"view_{index:04d}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                raw_pose = np.asarray(archive["camera_pose"], dtype=np.float64)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise TrajectoryRefinementError(
                f"DA3 raw view {index} has no readable camera_pose provenance"
            ) from exc
        if raw_pose.shape != (4, 4) or not np.isfinite(raw_pose).all():
            raise TrajectoryRefinementError(f"DA3 raw view {index} camera_pose is malformed")
        if not np.allclose(raw_pose, source_poses[index], atol=2e-4, rtol=2e-4):
            raise TrajectoryRefinementError(
                f"DA3 raw view {index} camera_pose disagrees with the loaded provider pose"
            )


def _source_admission(
    scan_dir: Path, source: Mapping[str, Any]
) -> tuple[Path, Path, int, tuple[int, ...] | None, dict[str, Any]]:
    """Return the exact provider manifest/raw root after strict admission."""
    provider = source.get("provider")
    if provider != "da3":
        raise TrajectoryRefinementError(
            "path refinement requires a DA3 provider result; MapAnything is not admitted"
        )
    manifest = source.get("manifest")
    if not isinstance(manifest, Mapping):
        raise TrajectoryRefinementError("_load_provider manifest is missing")
    coordinate_frame = manifest.get("coordinate_frame")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.startswith("da3_metric_world_"):
        raise TrajectoryRefinementError(
            "DA3 provider does not declare a supported unaligned metric coordinate frame"
        )
    prepared = source.get("prepared")
    prepared_frames = prepared.get("frames") if isinstance(prepared, Mapping) else None
    selected = source.get("selected")
    if not isinstance(prepared_frames, list) or not prepared_frames:
        raise TrajectoryRefinementError("_load_provider prepared frame identities are missing")
    if not isinstance(selected, list):
        raise TrajectoryRefinementError("_load_provider selected identities are malformed")
    prepared_count = len(prepared_frames)
    if prepared_count > MAX_PROVIDER_VIEWS:
        raise TrajectoryRefinementError("provider view count exceeds the bounded refinement limit")
    scope = source.get("scope")
    if not isinstance(scope, Mapping):
        raise TrajectoryRefinementError("_load_provider scope is missing")
    is_full = scope.get("status") == "full_prepared_view_set" and selected == list(range(prepared_count))
    is_partial = scope.get("status") == "partial" and selected != list(range(prepared_count))
    if not (is_full or is_partial):
        raise TrajectoryRefinementError(
            "provider scope is inconsistent with its exact prepared identities"
        )
    provider_count = len(selected)
    if provider_count > MAX_PROVIDER_VIEWS:
        raise TrajectoryRefinementError("provider view count exceeds the bounded refinement limit")
    if len(source.get("poses", ())) != provider_count:
        raise TrajectoryRefinementError(
            "provider pose count does not match the selected prepared view set"
        )
    provider_manifest = _provider_manifest_path(scan_dir, source)
    raw_root = provider_manifest.parent / "raw"
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise TrajectoryRefinementError(
            f"DA3 raw output directory is missing beside the provider manifest: {raw_root}"
        )
    expanded_bytes = 0
    for index in range(provider_count):
        raw_path = raw_root / f"view_{index:04d}.npz"
        if raw_path.is_symlink() or not raw_path.is_file():
            raise TrajectoryRefinementError(f"DA3 raw view {index} is missing")
        if raw_path.stat().st_size > MAX_RAW_VIEW_BYTES:
            raise TrajectoryRefinementError(
                f"DA3 raw view {index} exceeds the bounded artifact size limit"
            )
        expanded_bytes += _declared_npz_bytes(raw_path, index)
        if expanded_bytes > MAX_EXPANDED_RAW_BYTES:
            raise TrajectoryRefinementError(
                "DA3 raw views exceed the bounded 2 GiB expanded-array budget"
            )
    prepared_indices = None if is_full else tuple(int(index) for index in selected)
    _validate_raw_pose_provenance(raw_root, source, provider_count)
    return provider_manifest, raw_root, provider_count, prepared_indices, {
        "view_count": provider_count,
        "prepared_view_count": prepared_count,
        "partial_provider": is_partial,
        "prepared_indices": list(prepared_indices) if prepared_indices is not None else None,
        "expanded_npz_bytes": expanded_bytes,
        "max_view_count": MAX_PROVIDER_VIEWS,
        "max_expanded_npz_bytes": MAX_EXPANDED_RAW_BYTES,
    }


def _vio_constraint_candidate(
    scan_dir: Path, state: Mapping[str, Any]
) -> tuple[Path | None, dict[str, Any]]:
    """Find only an already-admitted, retained VIO result.

    The trajectory engine performs the full ``vio_result.v1`` validation and
    derives consecutive camera-relative edges.  This function only decides
    whether a retained result is eligible to be handed to that validator.
    """
    vio = state.get("vio")
    if not isinstance(vio, Mapping):
        return None, {
            "status": "missing_admission",
            "reason": "No retained VIO job result is present; metric VIO constraints are not admitted.",
        }
    if vio.get("status") != "complete":
        return None, {
            "status": "missing_admission",
            "reason": (
                f"Retained VIO status is {vio.get('status')!r}; only a complete, "
                "qualified result can provide path-refinement constraints."
            ),
        }
    results = vio.get("results")
    if not isinstance(results, Mapping):
        return None, {
            "status": "missing_admission",
            "reason": "Completed VIO has no retained results object; no constraints were fabricated.",
        }
    artifact = results.get("artifact")
    try:
        path = _bound_relative(scan_dir, artifact, "retained VIO artifact")
    except TrajectoryRefinementError as exc:
        return None, {"status": "missing_admission", "reason": str(exc)}
    if not path.is_file():
        return None, {
            "status": "missing_admission",
            "reason": f"Completed VIO artifact is missing: {artifact}",
        }
    if path.stat().st_size > MAX_JSON_BYTES:
        return None, {
            "status": "missing_admission",
            "reason": "Completed VIO artifact exceeds the bounded JSON size limit.",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {
            "status": "missing_admission",
            "reason": f"Retained VIO artifact is unreadable: {type(exc).__name__}",
        }
    if not isinstance(payload, Mapping) or payload.get("schema") != "noesis.phone_capture.vio_result.v1":
        return None, {
            "status": "missing_admission",
            "reason": "Retained VIO artifact is not the qualified phone_capture.vio_result.v1 result.",
        }
    if payload.get("accepted_for_metric_vio") is not True:
        return None, {
            "status": "missing_admission",
            "reason": "Retained VIO result is not admitted for metric use; calibration/timing gates remain closed.",
        }
    return path, {
        "status": "eligible",
        "path": str(path),
        "sha256": _sha256(path),
        "reason": "Existing admitted VIO result will be validated and adapted to camera-relative edges.",
    }


def _accepted_poses(
    output_dir: Path, engine_report: Mapping[str, Any], expected_count: int
) -> np.ndarray | None:
    refinement = engine_report.get("refinement")
    if not isinstance(refinement, Mapping) or refinement.get("raw_materialized") is not True:
        return None
    materialized = refinement.get("materialized")
    if not isinstance(materialized, Mapping):
        return None
    value = materialized.get("camera_solution")
    if not isinstance(value, str):
        return None
    candidate = Path(value).resolve()
    try:
        candidate.relative_to(output_dir.resolve())
    except ValueError:
        return None
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        with np.load(candidate, allow_pickle=False) as archive:
            poses = np.asarray(archive["camera_to_world"], dtype=np.float64).copy()
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if poses.shape != (expected_count, 4, 4) or not np.isfinite(poses).all():
        return None
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-3)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3)
    ):
        return None
    return poses


def _adapter_report(
    *,
    scan_dir: Path,
    status: str,
    reason: str | None,
    vio: Mapping[str, Any],
    engine_report: Mapping[str, Any] | None = None,
    report_path: Path | None = None,
    input_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = dict(engine_report or {})
    refinement = report.get("refinement")
    if not isinstance(refinement, Mapping):
        refinement = {"status": "not_run", "raw_materialized": False}
    deformation = refinement.get("pose_deformation")
    accepted_vio = int(refinement.get("accepted_vio_constraint_count") or 0)
    accepted_visual = int(refinement.get("accepted_visual_constraint_count") or 0)
    algorithm = "existing_visual_revisit_pose_graph_only"
    if accepted_vio:
        algorithm += "+qualified_vio_relative_constraints"
    report.update(
        {
            "schema": PATH_REFINEMENT_SCHEMA,
            "status": status,
            "adapter_status": status,
            "review_only": True,
            "scan_id": scan_dir.name,
            "vio_constraint_admission": dict(vio),
            "position_refinement_algorithm": algorithm,
            "constraint_contributions": {
                "visual_revisit_pose_graph": accepted_visual,
                "qualified_vio_relative_constraints": accepted_vio,
                "imu_position_integration": False,
            },
            "imu_position_integration": False,
            "calibration_fabricated": False,
            "accepted_constraint_count": int(refinement.get("accepted_constraint_count") or 0),
            "accepted_visual_constraint_count": accepted_visual,
            "accepted_vio_constraint_count": accepted_vio,
            "accepted_camera_pose_count": int(
                refinement.get("source_raw_view_count") or report.get("source_raw_view_count") or 0
            ) if refinement.get("raw_materialized") is True else 0,
            "accepted_camera_pose_changes": dict(deformation) if isinstance(deformation, Mapping) else None,
            "rejection_reason": reason,
        }
    )
    if input_budget is not None:
        report["input_budget"] = dict(input_budget)
    if report_path is not None:
        report["report_path"] = str(report_path)
    return report


def refine_review_path(
    scan_dir: Path,
    source: Mapping[str, Any],
    state: Mapping[str, Any],
    output_dir: Path,
    *, revalidate_scaled_carrier=None,
) -> dict[str, Any]:
    """Run bounded review-path refinement over an already loaded provider.

    ``source`` is the exact result of ``trajectory_motion_review._load_provider``.
    The return shape is ``{"status", "report", "accepted_camera_poses"}``;
    the last value is a finite ``(N, 4, 4)`` ndarray only when the existing
    refinement engine materialized a candidate after all gates, otherwise
    ``None``.  The original scan and provider raw files are never written.
    """
    scan = Path(scan_dir).resolve()
    if not scan.is_dir():
        raise ValueError("scan_dir must be a retained scan directory")
    if not isinstance(source, Mapping) or not isinstance(state, Mapping):
        raise TypeError("source and state must be mappings")
    output = _new_output_dir(Path(output_dir))

    try:
        provider_manifest, raw_root, frame_count, prepared_indices, input_budget = _source_admission(scan, source)
    except (TrajectoryRefinementError, OSError, ValueError, TypeError) as exc:
        reason = str(exc)[:2000]
        vio_path, vio_info = _vio_constraint_candidate(scan, state)
        del vio_path
        report = _adapter_report(
            scan_dir=scan, status="needs_evidence", reason=reason,
            vio=vio_info, report_path=output / "path_refinement_report.json",
        )
        _write_json(output / "path_refinement_report.json", report)
        return {"status": "needs_evidence", "report": report, "accepted_camera_poses": None}

    vio_path, vio_info = _vio_constraint_candidate(scan, state)
    # The final 30 percent is a deterministic temporal holdout.  It is passed
    # to the existing engine, which excludes it from fitting and evaluates it
    # before allowing materialization.
    holdout_start = max(1, min(frame_count - 1, int(math.floor(frame_count * 0.70))))
    settings = TrajectoryRefinementSettings(
        withheld_ranges=((holdout_start, frame_count),)
    )

    try:
        engine_kwargs: dict[str, Any] = {"vio_constraints": vio_path}
        if revalidate_scaled_carrier is not None:
            engine_kwargs["revalidate_scaled_carrier"] = revalidate_scaled_carrier
        if prepared_indices is not None:
            engine_kwargs["prepared_indices"] = prepared_indices
        engine_report = run_trajectory_refinement(
            scan, raw_root, output, settings, **engine_kwargs
        )
    except (TrajectoryRefinementError, TrajectoryMotionReviewError, OSError, ValueError, TypeError, MemoryError) as exc:
        reason = str(exc)[:2000]
        report = _adapter_report(
            scan_dir=scan, status="needs_evidence", reason=reason,
            vio=vio_info, report_path=output / "path_refinement_report.json",
            input_budget=input_budget,
        )
        _write_json(output / "path_refinement_report.json", report)
        return {"status": "needs_evidence", "report": report, "accepted_camera_poses": None}

    poses = _accepted_poses(output, engine_report, frame_count)
    engine_vio = engine_report.get("vio")
    if vio_path is not None and isinstance(engine_vio, Mapping):
        if engine_vio.get("status") == "accepted":
            vio_info = {**vio_info, "status": "accepted", "accepted_constraint_count": int(engine_report.get("refinement", {}).get("accepted_vio_constraint_count") or 0)}
        else:
            vio_info = {
                **vio_info,
                "status": "rejected",
                "reason": engine_vio.get("rejection_reason") or engine_vio.get("status") or "VIO constraints were rejected by the existing validator",
            }
    status = "accepted" if poses is not None else "rejected"
    refinement = engine_report.get("refinement") or {}
    if poses is not None:
        reason = None
    elif refinement.get("raw_materialized") is True:
        reason = "Existing engine marked a candidate materialized, but its accepted camera_solution is missing or invalid"
    else:
        reason = str(
            refinement.get("rejection_reason")
            or refinement.get("materialization_rejection_reason")
            or refinement.get("status")
            or "Existing refinement gates did not accept a materialized candidate"
        )
    report = _adapter_report(
        scan_dir=scan, status=status, reason=reason, vio=vio_info,
        engine_report=engine_report, report_path=output / "path_refinement_report.json",
        input_budget=input_budget,
    )
    _write_json(output / "path_refinement_report.json", report)
    return {"status": status, "report": report, "accepted_camera_poses": poses}


__all__ = ["PATH_REFINEMENT_SCHEMA", "refine_review_path"]
