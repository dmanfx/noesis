#!/usr/bin/env python3
"""Publish one immutable multi-room PCF as the current review assembly.

The publisher binds review-only PCF bytes to the exact promoted scene release,
authored model, calibration bundle, and backend-world-to-Menon transform.  It
does not promote a scene release or mark the reconstruction canonical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from noesis.virtual_twin.store import VirtualTwinStore
from noesis_core.scene_store import SceneReleaseStore


class ReviewAssemblyPublishError(RuntimeError):
    """Raised when an assembly cannot be bound without ambiguity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewAssemblyPublishError(f"{path} is not a JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _world_to_scene_binding(
    world_alignment_path: Path,
) -> tuple[list[float], list[str], str, str]:
    payload = _json(world_alignment_path)
    similarity = payload.get("scene_similarity")
    matrix = (
        similarity.get("world_to_scene_col_major")
        if isinstance(similarity, dict)
        else None
    )
    if (
        not isinstance(similarity, dict)
        or similarity.get("source")
        != "menon_virtual_device_camera_similarity_v1"
        or not isinstance(similarity.get("camera_count"), int)
        or similarity["camera_count"] < 1
        or not isinstance(matrix, list)
        or len(matrix) != 16
        or not all(isinstance(item, (int, float)) for item in matrix)
    ):
        raise ReviewAssemblyPublishError(
            "runtime world alignment has no valid Menon scene similarity"
        )
    reference = [float(item) for item in matrix]
    canonical = [format(value, ".17g") for value in reference]
    return (
        reference,
        canonical,
        _canonical_sha256(canonical),
        _sha256(world_alignment_path),
    )


def _camera_anchor_binding(
    report_path: Path,
    reintegration: dict[str, Any],
) -> dict[str, Any]:
    report = _json(report_path)
    if (
        report.get("schema") != "noesis.pcf.static_camera_anchor.v2"
        or report.get("status") != "passed_review_anchor"
        or report.get("accepted_for_canonical_use") is not False
        or report.get("camera_id") != "family-room"
        or report.get("coordinate_frame") != "family_accepted_backend_world_m"
        or report.get("anchor_mode") != "floor_locked_planar"
        or report.get("pose_convention")
        != "opencv_cam2world_x_right_y_down_z_forward"
    ):
        raise ReviewAssemblyPublishError("static-camera anchor report is not admissible")
    constraints = report.get("constraints")
    if not isinstance(constraints, dict) or any(
        constraints.get(key) is not expected
        for key, expected in {
            "metric_scale_fixed": True,
            "gravity_fixed": True,
            "calibrated_pitch_roll_preserved": True,
            "pnp_translation_used": False,
            "whole_cloud_icp_used": False,
            "bounding_box_anchor_used": False,
            "manual_scene_nudge_used": False,
        }.items()
    ):
        raise ReviewAssemblyPublishError(
            "static-camera anchor weakened its fixed-frame constraints"
        )
    if (
        constraints.get("camera_height_source")
        != "admitted_scene_prior_reference_camera"
        or float(constraints.get("vertical_anchor_translation_m", math.nan)) != 0.0
    ):
        raise ReviewAssemblyPublishError(
            "static-camera anchor does not preserve floor authority"
        )
    evidence = report.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("scene_prior_quality_passed") is not True
        or evidence.get("conditioned_fusion_alignment_passed") is not True
        or evidence.get("fixed_camera_reprojection_admitted") is not True
        or evidence.get("pnp_report_role") != "uncertainty_diagnostic_only"
        or evidence.get("pnp_translation_used") is not False
    ):
        raise ReviewAssemblyPublishError(
            "static-camera anchor lacks admitted Scene Prior support"
        )
    inputs = report.get("inputs")
    surfel_sha256 = (
        reintegration.get("artifacts", {}).get("surfels_npz", {}).get("sha256")
    )
    if (
        not isinstance(inputs, dict)
        or not isinstance(surfel_sha256, str)
        or inputs.get("assembly_npz_sha256") != surfel_sha256
    ):
        raise ReviewAssemblyPublishError(
            "static-camera anchor is not bound to the selected PCF surfels"
        )

    def matrix(name: str) -> list[float]:
        values = report.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 16
            or not all(
                isinstance(item, (int, float)) and math.isfinite(float(item))
                for item in values
            )
        ):
            raise ReviewAssemblyPublishError(f"static-camera {name} is malformed")
        result = [float(item) for item in values]
        if any(abs(result[index]) > 1e-9 for index in (3, 7, 11)) or abs(
            result[15] - 1.0
        ) > 1e-9:
            raise ReviewAssemblyPublishError(
                f"static-camera {name} is not an affine pose"
            )
        return result

    solved = matrix("camera_to_assembly_col_major")
    device_reference = matrix("device_reference_camera_to_assembly_col_major")
    estimate = report.get("estimate")
    if not isinstance(estimate, dict):
        raise ReviewAssemblyPublishError("static-camera estimate is missing")
    return {
        "status": "review_pose_estimate",
        "accepted_for_canonical_use": False,
        "camera_id": "family-room",
        "coordinate_frame": "family_accepted_backend_world_m",
        "pose_convention": report["pose_convention"],
        "anchor_mode": report["anchor_mode"],
        "method": report.get("method"),
        "camera_to_assembly_col_major": solved,
        "device_reference_camera_to_assembly_col_major": device_reference,
        "camera_center_assembly_m": estimate.get("camera_center_assembly_m"),
        "camera_heading_deg": estimate.get("camera_heading_deg"),
        "translation_uncertainty_p80_m": estimate.get(
            "translation_uncertainty_p80_m"
        ),
        "yaw_uncertainty_p80_deg": estimate.get("yaw_uncertainty_p80_deg"),
        "reference_camera_center_displacement_m": estimate.get(
            "device_reference_camera_center_displacement_m"
        ),
        "camera_height_source": constraints.get("camera_height_source"),
        "vertical_anchor_translation_m": constraints.get(
            "vertical_anchor_translation_m"
        ),
        "source_report_sha256": _sha256(report_path),
    }


def _camera_markers_binding(
    report_path: Path,
    *,
    camera_anchor: dict[str, Any],
) -> dict[str, Any]:
    report = _json(report_path)
    markers = report.get("markers")
    constraints = report.get("constraints")
    if (
        report.get("schema") != "noesis.pcf.static_camera_markers.v1"
        or report.get("status") != "review_only"
        or report.get("accepted_for_canonical_use") is not False
        or report.get("coordinate_frame") != "family_accepted_backend_world_m"
        or report.get("color_hex") != "#ffd400"
        or not isinstance(report.get("sphere_radius_m"), (int, float))
        or not 0.05 <= float(report["sphere_radius_m"]) <= 0.5
        or not isinstance(markers, list)
        or len(markers) != 3
        or not isinstance(constraints, dict)
        or constraints.get("same_assembly_transform_as_geometry") is not True
        or constraints.get("bounding_box_anchor_used") is not False
        or constraints.get("manual_scene_nudge_used") is not False
        or constraints.get("reflection_used") is not False
    ):
        raise ReviewAssemblyPublishError("static-camera marker report is not admissible")
    marker_ids: set[str] = set()
    result_markers: list[dict[str, Any]] = []
    for marker in markers:
        position = marker.get("position_assembly_m") if isinstance(marker, dict) else None
        camera_id = marker.get("camera_id") if isinstance(marker, dict) else None
        if (
            camera_id not in {"family-room", "kitchen", "living-room"}
            or camera_id in marker_ids
            or not isinstance(position, list)
            or len(position) != 3
            or not all(
                isinstance(item, (int, float)) and math.isfinite(float(item))
                for item in position
            )
        ):
            raise ReviewAssemblyPublishError("static-camera marker is malformed")
        marker_ids.add(camera_id)
        result_markers.append(
            {
                "camera_id": camera_id,
                "position_assembly_m": [float(item) for item in position],
                "source_role": marker.get("source_role"),
                "room_transform_status": marker.get("room_transform_status"),
            }
        )
    if marker_ids != {"family-room", "kitchen", "living-room"}:
        raise ReviewAssemblyPublishError("static-camera marker set is incomplete")
    family_marker = next(
        marker for marker in result_markers if marker["camera_id"] == "family-room"
    )
    if any(
        abs(float(value) - float(camera_anchor["camera_center_assembly_m"][index]))
        > 1e-8
        for index, value in enumerate(family_marker["position_assembly_m"])
    ):
        raise ReviewAssemblyPublishError(
            "Family Room marker disagrees with the solved camera anchor"
        )
    return {
        "status": "review_camera_positions",
        "accepted_for_canonical_use": False,
        "coordinate_frame": report["coordinate_frame"],
        "method": report.get("method"),
        "sphere_radius_m": float(report["sphere_radius_m"]),
        "color_hex": report["color_hex"],
        "markers": result_markers,
        "source_report_sha256": _sha256(report_path),
    }


def publish(
    *,
    source_glb: Path,
    source_manifest: Path,
    camera_overlay_manifest: Path,
    camera_anchor_report: Path,
    camera_markers_report: Path,
    world_alignment_path: Path,
    scene_store_path: Path,
    virtual_twin_root: Path,
    pcf_storage_root: Path,
    assembly_id: str,
) -> dict[str, Any]:
    for path in (
        source_glb,
        source_manifest,
        camera_overlay_manifest,
        camera_anchor_report,
        camera_markers_report,
        world_alignment_path,
    ):
        if not path.is_file():
            raise ReviewAssemblyPublishError(f"required input is missing: {path}")
    if not assembly_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for char in assembly_id
    ):
        raise ReviewAssemblyPublishError("assembly_id contains unsupported characters")

    twins = VirtualTwinStore(root=virtual_twin_root)
    store = SceneReleaseStore(
        scene_store_path,
        artifact_root=twins.revisions_root,
        bundle_root=twins.root,
    )
    release = store.current()
    if release is None:
        raise ReviewAssemblyPublishError("no scene release is currently promoted")
    store.validate_artifacts(release)
    (
        matrix,
        matrix_canonical,
        matrix_sha256,
        world_alignment_sha256,
    ) = _world_to_scene_binding(world_alignment_path)

    reintegration = _json(source_manifest)
    overlay = _json(camera_overlay_manifest)
    if reintegration.get("accepted_for_canonical_use") is not False:
        raise ReviewAssemblyPublishError(
            "the selected source must explicitly remain non-canonical"
        )
    owner_counts = reintegration.get("npz_contract", {}).get("owner_voxel_counts")
    if not isinstance(owner_counts, dict) or not owner_counts:
        raise ReviewAssemblyPublishError("source manifest has no owner voxel counts")
    point_count = sum(int(value) for value in owner_counts.values())
    uncertainty = overlay.get("camera", {})
    camera_anchor = _camera_anchor_binding(camera_anchor_report, reintegration)
    camera_markers = _camera_markers_binding(
        camera_markers_report,
        camera_anchor=camera_anchor,
    )

    assemblies_root = pcf_storage_root / "review-assemblies"
    destination = assemblies_root / assembly_id
    if destination.exists():
        raise ReviewAssemblyPublishError(f"assembly already exists: {destination}")
    assemblies_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{assembly_id}.", dir=assemblies_root))
    try:
        artifact_name = "multiroom_points.glb"
        copied_glb = temporary / artifact_name
        shutil.copyfile(source_glb, copied_glb)
        artifact_sha256 = _sha256(copied_glb)
        relative_artifact = (
            Path("review-assemblies") / assembly_id / artifact_name
        ).as_posix()
        manifest = {
            "contract": "noesis.scene.review_assembly",
            "contract_version": 3,
            "assembly_id": assembly_id,
            "created_at_us": time.time_ns() // 1_000,
            "status": "review_only",
            "accepted_for_canonical_use": False,
            "coordinate_frame": "backend_world_m",
            "assembly_gauge": "family_accepted_backend_world_m",
            "units": "meters",
            "scene_binding": {
                "release_id": release.release_id,
                "authored_scene_sha256": release.authored_scene.sha256,
                "calibration_bundle_sha256": release.calibration.sha256,
                "config_sha256": release.config.sha256,
                "world_to_scene_source": "noesis_runtime_scene_similarity",
                "world_alignment_sha256": world_alignment_sha256,
                "world_to_scene_sha256": matrix_sha256,
                "world_to_scene_col_major": matrix,
                "world_to_scene_canonical_values": matrix_canonical,
            },
            "artifact": {
                "role": "multiroom_points_glb",
                "relative_path": relative_artifact,
                "sha256": artifact_sha256,
                "size_bytes": copied_glb.stat().st_size,
                "media_type": "model/gltf-binary",
                "point_count": point_count,
                "owner_point_counts": {
                    str(key): int(value) for key, value in owner_counts.items()
                },
            },
            "registration": {
                "status": reintegration.get("registration", {}).get("status"),
                "reason_codes": reintegration.get("registration", {}).get(
                    "reason_codes"
                ),
                "translation_uncertainty_p80_m": uncertainty.get(
                    "review_translation_uncertainty_p80_m"
                ),
                "yaw_uncertainty_bound_deg": uncertainty.get(
                    "review_yaw_uncertainty_bound_deg"
                ),
                "connector_geometry_included": reintegration.get("sources", {})
                .get("connector", {})
                .get("geometry_included"),
            },
            "camera_anchor": camera_anchor,
            "camera_markers": camera_markers,
            "provenance": {
                "source_glb_sha256": _sha256(source_glb),
                "source_reintegration_manifest_sha256": _sha256(source_manifest),
                "source_camera_overlay_manifest_sha256": _sha256(
                    camera_overlay_manifest
                ),
                "source_camera_anchor_report_sha256": _sha256(
                    camera_anchor_report
                ),
                "source_camera_markers_report_sha256": _sha256(
                    camera_markers_report
                ),
                "phone_walk_only": reintegration.get("phone_walk_only") is True,
                "static_camera_points_included": reintegration.get(
                    "static_camera_points_included"
                )
                is True,
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        current_tmp = assemblies_root / ".current.json.tmp"
        current_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(current_tmp, assemblies_root / "current.json")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--camera-overlay-manifest", type=Path, required=True)
    parser.add_argument("--camera-anchor-report", type=Path, required=True)
    parser.add_argument("--camera-markers-report", type=Path, required=True)
    parser.add_argument("--world-alignment", type=Path, required=True)
    parser.add_argument("--scene-store", type=Path, required=True)
    parser.add_argument("--virtual-twin-root", type=Path, required=True)
    parser.add_argument("--pcf-storage-root", type=Path, required=True)
    parser.add_argument("--assembly-id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = publish(
        source_glb=args.source_glb,
        source_manifest=args.source_manifest,
        camera_overlay_manifest=args.camera_overlay_manifest,
        camera_anchor_report=args.camera_anchor_report,
        camera_markers_report=args.camera_markers_report,
        world_alignment_path=args.world_alignment,
        scene_store_path=args.scene_store,
        virtual_twin_root=args.virtual_twin_root,
        pcf_storage_root=args.pcf_storage_root,
        assembly_id=args.assembly_id,
    )
    print(
        json.dumps(
            {
                "assembly_id": result["assembly_id"],
                "release_id": result["scene_binding"]["release_id"],
                "artifact_sha256": result["artifact"]["sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
