"""Output-only static-camera calibration backup and replacement helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256


CALIBRATION_REPLACEMENT_SCHEMA = "noesis.pcf.static_camera_calibration_replacement.v1"


class CalibrationReplacementError(RuntimeError):
    """Raised when a replacement cannot be proven safe to write."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_camera_to_world(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CalibrationReplacementError("camera_to_world must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise CalibrationReplacementError("camera_to_world has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise CalibrationReplacementError("camera_to_world rotation is not orthonormal")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=2e-4):
        raise CalibrationReplacementError("camera_to_world rotation is not proper")
    return matrix


def backup_calibration_source(
    source: Path,
    backup_root: Path,
    *,
    camera_id: str,
) -> dict[str, Any]:
    """Copy exact source bytes and record hashes before a replacement output."""
    source = source.resolve()
    if not source.is_file():
        raise CalibrationReplacementError(f"calibration source is missing: {source}")
    source_sha256 = _sha256(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup_root = backup_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(
        prefix=f"{stamp}_{source.stem}_{camera_id}_{source_sha256[:12]}_",
        dir=backup_root,
    ))
    backup_path = backup_dir / source.name
    shutil.copy2(source, backup_path)
    backup_sha256 = _sha256(backup_path)
    if backup_sha256 != source_sha256:
        raise CalibrationReplacementError("calibration backup hash differs from source")
    manifest = {
        "schema": CALIBRATION_REPLACEMENT_SCHEMA,
        "operation": "exact_backup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_id": str(camera_id),
        "source_file_name": source.name,
        "source_sha256": source_sha256,
        "backup_file_name": backup_path.name,
        "backup_sha256": backup_sha256,
        "exact_bytes": True,
    }
    manifest_path = backup_dir / "backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["backup_manifest"] = manifest_path.name
    return manifest


def _atomic_npz(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one bounded NPZ artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_static_reference_revision(
    template_revision: Path,
    output_revision: Path,
    *,
    camera_id: str,
    replacement_calibration: Path,
    static_depth_world_points: Path,
    frame_identity: Mapping[str, Any],
    frame_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the normal static room revision consumed by PCF.

    ``static_depth_world_points`` is produced by
    :func:`regenerate_static_depth_world_points`.  This function carries those
    points into the existing ``room_points.npz`` and ``room_points_meta.json``
    producer shape, copies one explicitly declared RGB keyframe, and records
    the replacement calibration and frame binding digests.  It never scans or
    modifies the template revision.
    """
    if not isinstance(frame_identity, Mapping) or not all(
        isinstance(frame_identity.get(key), str) and frame_identity[key]
        for key in ("frame_id", "revision", "coordinate_frame")
    ):
        raise CalibrationReplacementError(
            "static reference materialization requires an exact frame identity"
        )
    if frame_identity["coordinate_frame"] != "backend_world_m_stream_points":
        raise CalibrationReplacementError(
            "static reference materialization requires backend_world_m_stream_points"
        )
    template_revision = template_revision.resolve()
    output_revision = output_revision.resolve()
    replacement_calibration = replacement_calibration.resolve()
    static_depth_world_points = static_depth_world_points.resolve()
    if not template_revision.is_dir():
        raise CalibrationReplacementError(f"static revision template is missing: {template_revision}")
    if output_revision.exists():
        raise CalibrationReplacementError(f"static revision output already exists: {output_revision}")
    if not replacement_calibration.is_file():
        raise CalibrationReplacementError(
            f"replacement calibration is missing: {replacement_calibration}"
        )
    if not static_depth_world_points.is_file():
        raise CalibrationReplacementError(
            f"static depth world points are missing: {static_depth_world_points}"
        )
    metadata_path = template_revision / "room_points_meta.json"
    if not metadata_path.is_file():
        raise CalibrationReplacementError(
            f"static revision template metadata is missing: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        calibration = json.loads(replacement_calibration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationReplacementError(
            "static revision template or replacement calibration is unreadable"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("camera") != camera_id:
        raise CalibrationReplacementError(
            "static revision template camera does not match replacement camera"
        )
    calibration_rows = calibration.get("cameras") if isinstance(calibration, dict) else None
    camera = calibration_rows.get(camera_id) if isinstance(calibration_rows, dict) else None
    if not isinstance(camera, dict) or not isinstance(camera.get("E"), list) or len(camera["E"]) != 16:
        raise CalibrationReplacementError(
            "replacement calibration has no usable camera E matrix"
        )
    keyframes = metadata.get("rgb_keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        raise CalibrationReplacementError(
            "static revision template has no declared RGB keyframe"
        )
    key, relative = next(iter(keyframes.items()))
    if not isinstance(key, str) or not isinstance(relative, str):
        raise CalibrationReplacementError("static revision template keyframe reference is malformed")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CalibrationReplacementError("static revision template keyframe escapes its root")
    source_keyframe = (template_revision / relative_path).resolve()
    try:
        source_keyframe.relative_to(template_revision)
    except ValueError as exc:
        raise CalibrationReplacementError("static revision template keyframe escapes its root") from exc
    if not source_keyframe.is_file():
        raise CalibrationReplacementError(f"static revision template keyframe is missing: {source_keyframe}")
    try:
        with np.load(static_depth_world_points, allow_pickle=False) as payload:
            if "world_points" not in payload.files:
                raise CalibrationReplacementError("static depth world points has no world_points array")
            points = np.asarray(payload["world_points"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] <= 0 or not np.isfinite(points).all():
                raise CalibrationReplacementError("static depth world points are malformed")
            colors = np.full((points.shape[0], 3), 205, dtype=np.uint8)
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise CalibrationReplacementError("static depth world points have malformed intrinsics")
    except (OSError, ValueError) as exc:
        raise CalibrationReplacementError("static depth world points are unreadable") from exc
    output_revision.mkdir(parents=True, exist_ok=False)
    try:
        keyframe_target = output_revision / relative_path
        keyframe_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_keyframe, keyframe_target)
        points_path = output_revision / "room_points.npz"
        _atomic_npz(
            points_path,
            {
                "points": points,
                "colors": colors,
                "replacement_source_depth_sha256": np.asarray(_sha256(static_depth_world_points), dtype="U64"),
            },
        )
        output_metadata = {
            **metadata,
            "schema": "noesis.room_reconstruction.stream_points.v4",
            "revision_id": output_revision.name,
            "camera": camera_id,
            "coordinate_frame": "backend_world_m_stream_points",
            "rgb_keyframes": {key: relative_path.as_posix()},
            "intrinsics": intrinsics.tolist(),
            "extrinsics_col_major": [float(value) for value in camera["E"]],
            "source_point_count": int(points.shape[0]),
            "point_count": int(points.shape[0]),
            "source_sample_stride": 1,
            "floor_alignment": {
                "world_correction_col_major": np.eye(4, dtype=np.float64).reshape(-1, order="F").tolist(),
                "source": "replacement_camera_pose_already_in_target_frame",
                "target_floor_y": 0.0,
            },
            "calibrated_floor_y": 0.0,
            "calibration_source": "calibration_replacement_output",
            "calibration_replacement_sha256": _sha256(replacement_calibration),
            "static_depth_world_points_sha256": _sha256(static_depth_world_points),
            "source_frame": dict(frame_identity),
            "frame_binding": dict(frame_binding) if frame_binding is not None else None,
        }
        metadata_output = output_revision / "room_points_meta.json"
        metadata_output.write_text(
            json.dumps(output_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_revision, ignore_errors=True)
        raise
    return {
        "revision": str(output_revision),
        "room_points": {"path": str(points_path), "sha256": _sha256(points_path)},
        "room_points_meta": {"path": str(metadata_output), "sha256": _sha256(metadata_output)},
        "keyframe": {"path": str(keyframe_target), "sha256": _sha256(keyframe_target)},
        "calibration_sha256": _sha256(replacement_calibration),
        "static_depth_world_points_sha256": _sha256(static_depth_world_points),
        "frame_identity": dict(frame_identity),
        "frame_binding": dict(frame_binding) if frame_binding is not None else None,
        "source_revision": str(template_revision),
    }


def regenerate_static_depth_world_points(
    depth_input: Path,
    output: Path,
    *,
    camera_to_world: Any,
    frame_identity: Mapping[str, Any],
    max_points: int = 5_000_000,
) -> dict[str, Any]:
    """Reproject retained camera-space depth through the replacement pose.

    The input must carry ``depth_z``, ``mask``, and calibrated ``intrinsics``.
    No depth or alignment is estimated here; this is the bounded direct
    consumer used by an isolated calibration replacement run.
    """
    if not isinstance(frame_identity, Mapping) or not all(
        isinstance(frame_identity.get(key), str) and frame_identity[key]
        for key in ("frame_id", "revision", "coordinate_frame")
    ):
        raise CalibrationReplacementError(
            "depth regeneration requires an exact target frame identity"
        )
    depth_input = depth_input.resolve()
    output = output.resolve()
    if not depth_input.is_file():
        raise CalibrationReplacementError(f"static depth input is missing: {depth_input}")
    if depth_input == output:
        raise CalibrationReplacementError("depth regeneration output must differ from input")
    if output.exists():
        raise CalibrationReplacementError(f"depth regeneration output already exists: {output}")
    matrix = _validated_camera_to_world(camera_to_world)
    try:
        with np.load(depth_input, allow_pickle=False) as payload:
            if not {"depth_z", "mask", "intrinsics"}.issubset(payload.files):
                raise CalibrationReplacementError(
                    "static depth input needs depth_z, mask, and intrinsics"
                )
            depth = np.asarray(payload["depth_z"], dtype=np.float64)
            mask = np.asarray(payload["mask"], dtype=bool)
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise CalibrationReplacementError("static depth input is unreadable") from exc
    if (
        depth.ndim != 2
        or mask.shape != depth.shape
        or intrinsics.shape != (3, 3)
        or not np.isfinite(depth).all()
        or not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
    ):
        raise CalibrationReplacementError("static depth input has malformed arrays")
    valid = mask & (depth > 0.0)
    point_count = int(np.count_nonzero(valid))
    if point_count <= 0 or point_count > int(max_points):
        raise CalibrationReplacementError(
            f"static depth input has unsupported valid point count: {point_count}"
        )
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (rows.astype(np.float64) - intrinsics[1, 2]) / intrinsics[1, 1] * z
    camera_points = np.column_stack((x, y, z))
    world_points = (matrix[:3, :3] @ camera_points.T).T + matrix[:3, 3]
    if not np.isfinite(world_points).all():
        raise CalibrationReplacementError("regenerated world points are non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary_output = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_output,
            world_points=world_points.astype(np.float32),
            depth_z=depth.astype(np.float32),
            mask=mask,
            intrinsics=intrinsics.astype(np.float64),
            camera_to_world=matrix.astype(np.float64),
            coordinate_frame=np.asarray(str(frame_identity["coordinate_frame"]), dtype="U128"),
            frame_id=np.asarray(str(frame_identity["frame_id"]), dtype="U128"),
            frame_revision=np.asarray(str(frame_identity["revision"]), dtype="U256"),
            source_depth_sha256=np.asarray(_sha256(depth_input), dtype="U64"),
        )
        with temporary_output.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
    source_sha256 = _sha256(depth_input)
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "source_depth": str(depth_input),
        "source_depth_sha256": source_sha256,
        "valid_point_count": point_count,
        "frame_identity": dict(frame_identity),
        "atomic_write": True,
    }


def write_calibration_replacement(
    source: Path,
    output: Path,
    *,
    camera_id: str,
    camera_to_world: Any,
    calibration_from_assembly: Any,
    provenance: Mapping[str, Any],
    backup_root: Path,
    static_depth_input: Path | None = None,
    static_depth_output: Path | None = None,
    static_reference_template: Path | None = None,
    static_reference_output: Path | None = None,
) -> dict[str, Any]:
    """Write a replacement calibration beside an exact backup.

    The source file is never edited.  The report must carry measured canonical
    acceptance and an explicit frame identity; a pose flag alone is rejected.
    """
    if provenance.get("accepted_for_canonical_use") is not True:
        raise CalibrationReplacementError("replacement requires measured canonical acceptance")
    acceptance = provenance.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("status") != "passed":
        raise CalibrationReplacementError("replacement requires a passed measured acceptance report")
    frame_identity = provenance.get("frame_identity")
    if not isinstance(frame_identity, Mapping) or not all(
        isinstance(frame_identity.get(key), str) and frame_identity[key]
        for key in ("frame_id", "revision", "coordinate_frame")
    ):
        raise CalibrationReplacementError("replacement requires an exact frame identity")
    assembly_frame_identity = provenance.get("assembly_frame_identity")
    if not isinstance(assembly_frame_identity, Mapping) or not all(
        isinstance(assembly_frame_identity.get(key), str) and assembly_frame_identity[key]
        for key in ("frame_id", "revision", "coordinate_frame")
    ):
        raise CalibrationReplacementError(
            "replacement requires an exact assembly frame identity"
        )
    matrix = _validated_camera_to_world(camera_to_world)
    assembly_to_calibration = _validated_camera_to_world(calibration_from_assembly)
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise CalibrationReplacementError("replacement output must differ from the source")
    if output.exists():
        raise CalibrationReplacementError(f"replacement output already exists: {output}")
    if output.parent == source.parent and output.name == source.name:
        raise CalibrationReplacementError("replacement output must differ from the source")
    camera_to_calibration = assembly_to_calibration @ matrix
    if (static_depth_input is None) != (static_depth_output is None):
        raise CalibrationReplacementError(
            "static depth regeneration requires both input and output paths"
        )
    if (static_reference_template is None) != (static_reference_output is None):
        raise CalibrationReplacementError(
            "static reference materialization requires both template and output paths"
        )
    if static_reference_template is not None and static_reference_output is not None:
        if static_depth_output is None:
            raise CalibrationReplacementError(
                "static reference materialization requires regenerated static depth"
            )
        if static_reference_template.resolve() == static_reference_output.resolve():
            raise CalibrationReplacementError(
                "static reference output must differ from its template"
            )
        if static_reference_output.exists():
            raise CalibrationReplacementError(
                f"static reference output already exists: {static_reference_output}"
            )
    if static_depth_input is not None and static_depth_output is not None:
        if not static_depth_input.is_file():
            raise CalibrationReplacementError(
                f"static depth input is missing: {static_depth_input}"
            )
        if static_depth_input.resolve() == static_depth_output.resolve():
            raise CalibrationReplacementError(
                "static depth regeneration output must differ from input"
            )
        if static_depth_output.exists():
            raise CalibrationReplacementError(
                f"static depth regeneration output already exists: {static_depth_output}"
            )
    backup = backup_calibration_source(source, backup_root, camera_id=camera_id)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationReplacementError("calibration source is not readable JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cameras"), dict):
        raise CalibrationReplacementError("calibration source has no cameras object")
    camera = payload["cameras"].get(camera_id)
    if not isinstance(camera, dict):
        raise CalibrationReplacementError(f"calibration source is missing {camera_id}")
    camera = dict(camera)
    e_col_major = np.linalg.inv(camera_to_calibration).reshape(-1, order="F").astype(float).tolist()
    camera["E"] = e_col_major
    try:
        from noesis.calibration.pose_v1 import (
            E_col_major_to_pose_v1,
            pose_to_E_col_major,
            POSE_V1_FRAME_BACKEND_WORLD_M,
        )

        pose = E_col_major_to_pose_v1(
            e_col_major,
            source="measured_static_camera_pcf_pnp",
            frame=POSE_V1_FRAME_BACKEND_WORLD_M,
        )
    except Exception as exc:
        raise CalibrationReplacementError("replacement pose conversion failed") from exc
        if pose is None:
            raise CalibrationReplacementError("replacement E did not produce a valid PoseV1")
        # CalibrationManager treats a valid PoseV1 as authoritative and
        # regenerates E from it.  Store that exact round-tripped matrix so
        # the replacement artifact's physical fingerprint remains stable
        # when the manager reloads the pair.
        round_tripped_e = pose_to_E_col_major(pose)
        if not (isinstance(round_tripped_e, list) and len(round_tripped_e) == 16):
            raise CalibrationReplacementError("replacement pose round-trip failed")
        if not np.isfinite(np.asarray(round_tripped_e, dtype=np.float64)).all():
            raise CalibrationReplacementError("replacement pose round-trip is non-finite")
        e_col_major = [float(value) for value in round_tripped_e]
        camera["E"] = e_col_major
        camera["pose"] = pose
    camera["pose_provenance"] = {
        "source": "measured_static_camera_pcf_pnp",
        "frame_identity": dict(frame_identity),
        "acceptance_report_sha256": provenance.get("acceptance_report_sha256"),
        "manual_pose_replaced": True,
    }
    payload["cameras"] = dict(payload["cameras"])
    payload["cameras"][camera_id] = camera
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary_output = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_output, output)
    result = {
        "schema": CALIBRATION_REPLACEMENT_SCHEMA,
        "operation": "replacement_output",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_id": str(camera_id),
        "source_calibration": str(source),
        "source_sha256": _sha256(source),
        "replacement_calibration": str(output),
        "replacement_sha256": _sha256(output),
        "backup": backup,
        "frame_identity": dict(frame_identity),
        "assembly_frame_identity": dict(assembly_frame_identity),
        "assembly_to_calibration_col_major": assembly_to_calibration.reshape(-1, order="F").astype(float).tolist(),
        "direct_dependencies": {},
        "active_source_mutated": False,
    }
    dependency_dir = output.parent / "direct_dependencies"
    dependency_dir.mkdir(parents=True, exist_ok=True)
    camera_solution_path = dependency_dir / "camera_solution.npz"
    _atomic_npz(
        camera_solution_path,
        {
            "camera_to_world": camera_to_calibration.astype(np.float64),
            "world_to_camera": np.asarray(e_col_major, dtype=np.float64).reshape((4, 4), order="F"),
            "coordinate_frame": np.asarray(str(frame_identity["coordinate_frame"]), dtype="U128"),
            "frame_revision": np.asarray(str(frame_identity["revision"]), dtype="U256"),
            "units": np.asarray("m", dtype="U8"),
        },
    )
    regenerated_depth = None
    if static_depth_input is not None and static_depth_output is not None:
        regenerated_depth = regenerate_static_depth_world_points(
            static_depth_input,
            static_depth_output,
            camera_to_world=camera_to_calibration,
            frame_identity=frame_identity,
        )
    frame_binding_path = dependency_dir / "frame_binding_revalidation.json"
    frame_binding_payload = {
        "schema": "noesis.pcf.static_camera_direct_dependencies.v1",
        "source_frame": dict(assembly_frame_identity),
        "target_frame": dict(frame_identity),
        "target_from_source_col_major": assembly_to_calibration.reshape(-1, order="F").astype(float).tolist(),
        "target_from_source_sha256": revisioned_transform_sha256(
            RevisionedFrame(
                str(assembly_frame_identity["frame_id"]),
                str(assembly_frame_identity["revision"]),
            ),
            RevisionedFrame(
                str(frame_identity["frame_id"]),
                str(frame_identity["revision"]),
            ),
            assembly_to_calibration.reshape(-1, order="F").astype(float).tolist(),
        ),
        "camera_solution_sha256": _sha256(camera_solution_path),
        "status": "prepared_frame_binding_revalidation",
    }
    frame_binding_path.write_text(json.dumps(frame_binding_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dependency_manifest = {
        "camera_solution": {"path": str(camera_solution_path), "sha256": _sha256(camera_solution_path)},
        "frame_binding": {"path": str(frame_binding_path), "sha256": _sha256(frame_binding_path)},
        "depth_world_points": (
            regenerated_depth
            if regenerated_depth is not None
            else {"status": "unavailable_no_static_depth_input"}
        ),
    }
    if static_reference_template is not None and static_reference_output is not None:
        dependency_manifest["static_reference_revision"] = materialize_static_reference_revision(
            static_reference_template,
            static_reference_output,
            camera_id=camera_id,
            replacement_calibration=output,
            static_depth_world_points=static_depth_output,
            frame_identity=frame_identity,
            frame_binding=frame_binding_payload,
        )
    result["direct_dependencies"] = dependency_manifest
    manifest_path = output.with_name(output.stem + "_replacement_manifest.json")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["manifest"] = str(manifest_path)
    return result


__all__ = [
    "CALIBRATION_REPLACEMENT_SCHEMA",
    "CalibrationReplacementError",
    "backup_calibration_source",
    "materialize_static_reference_revision",
    "regenerate_static_depth_world_points",
    "write_calibration_replacement",
]
