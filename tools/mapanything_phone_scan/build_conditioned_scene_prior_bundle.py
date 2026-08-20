#!/usr/bin/env python3
"""Seal one validated conditioned-fusion room walk for Scene Prior v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


SELECTED_CANDIDATE = "prior_conditioned_consensus"
SELECTED_FUSION_DIR = "prior_conditioned_consensus_da3_carrier"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_entry(root: Path, relative: str) -> dict[str, Any]:
    payload = (root / relative).read_bytes()
    return {
        "path": relative,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _write_bytes(root: Path, relative: str, payload: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _copy_file(root: Path, relative: str, source: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source file is missing or unsafe: {source}")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _conditioned_admission_checks(
    candidate: dict[str, Any],
    *,
    metric_scale_preserved: bool,
) -> dict[str, bool]:
    """Gate PCF in the part of a single-view static cloud that is observable.

    A phone walk contains surfaces behind the fixed camera's nearest measured
    surface and outside its field of view. Symmetric full-cloud overlap and a
    raw z-buffer delta therefore penalize valid novel geometry. Use the same
    occlusion-aware static-camera domain as phone-walk alignment, while still
    requiring candidate-specific target support and vertical agreement.
    """

    room = candidate.get("static_cloud_room_bounds_metrics")
    visible = candidate.get("fixed_camera_visible_cloud_metrics")
    vertical = candidate.get("fixed_camera_visible_structure_metrics")
    if not all(isinstance(value, dict) for value in (room, visible, vertical)):
        raise ValueError(
            "selected candidate evaluation lacks visibility-aware static metrics"
        )

    visible_source_count = float(visible["source_point_count"])
    visible_comparable_count = float(visible["comparable_point_count"])
    vertical_source_count = float(vertical["source_point_count"])
    vertical_comparable_count = float(vertical["comparable_point_count"])
    return {
        "validated_da3_backend_alignment_passed": True,
        "conditioned_cloud_targets_backend_world": True,
        "metric_scale_preserved": metric_scale_preserved,
        "fixed_camera_visible_support_admitted": (
            visible_comparable_count >= max(5_000.0, 0.05 * visible_source_count)
        ),
        "fixed_camera_visible_overlap_admitted": (
            float(visible["source_overlap_0_30m"]) >= 0.55
        ),
        "fixed_camera_vertical_support_admitted": (
            vertical_comparable_count >= max(500.0, 0.10 * vertical_source_count)
        ),
        "fixed_camera_vertical_structure_admitted": (
            float(vertical["source_overlap_0_30m"]) >= 0.55
            and float(vertical["plane_residual_median_m"]) <= 0.10
        ),
        "static_target_coverage_admitted": (
            float(room["target_overlap_0_30m"]) >= 0.40
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        help="Passed Noesis alignment directory; defaults to <scan-dir>/alignment.",
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--target-revision", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    scan_dir = args.scan_dir.resolve()
    alignment_dir = (
        args.alignment_dir.resolve()
        if args.alignment_dir is not None
        else scan_dir / "alignment"
    )
    suite_root = args.suite_root.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    target_revision = args.target_revision.resolve()
    calibration = args.calibration.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite source bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    state = _read_json(scan_dir / "scan_state.json")
    prepared = _read_json(scan_dir / "prepared_frames_manifest.json")
    source_alignment_path = alignment_dir / "alignment_report.json"
    source_alignment = _read_json(source_alignment_path)
    if (
        source_alignment.get("status") != "passed"
        or not isinstance(source_alignment.get("quality_gate"), dict)
        or source_alignment["quality_gate"].get("passed") is not True
    ):
        raise ValueError("source Noesis alignment has not passed")
    target = source_alignment.get("target")
    if not isinstance(target, dict) or target.get("camera_id") != args.camera_id:
        raise ValueError("source alignment targets a different camera")
    if target.get("revision_id") != target_revision.name:
        raise ValueError("source alignment targets a different room revision")

    evaluation_path = evaluation_dir / "evaluation_metrics.json"
    evaluation = _read_json(evaluation_path)
    if evaluation.get("coordinate_frame") != "backend_world_m_stream_points":
        raise ValueError("evaluation is not in backend world meters")
    candidates = evaluation.get("candidates")
    candidate = candidates.get(SELECTED_CANDIDATE) if isinstance(candidates, dict) else None
    if not isinstance(candidate, dict):
        raise ValueError("evaluation has no prior-conditioned consensus candidate")
    alignment = candidate.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("selected candidate has no alignment")
    scale = float(alignment.get("scale", math.nan))
    rotation = np.asarray(alignment.get("rotation_row_major"), dtype=np.float64)
    translation = np.asarray(alignment.get("translation"), dtype=np.float64)
    if (
        not math.isfinite(scale)
        or abs(scale - 1.0) > 1e-3
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4
    ):
        raise ValueError("selected candidate alignment is not a rigid metric transform")

    fusion_root = suite_root / SELECTED_FUSION_DIR
    surfel_path = fusion_root / "surfel_points.npz"
    with np.load(surfel_path, allow_pickle=False) as row:
        points = np.asarray(row["points"], dtype=np.float64)
        colors = np.asarray(row["colors"], dtype=np.uint8)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or points.shape[0] < 100
        or not np.isfinite(points).all()
    ):
        raise ValueError("conditioned consensus surfel points are malformed")
    aligned_points = scale * (rotation @ points.T).T + translation
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(aligned_points.astype(np.float32), colors=colors),
        geom_name="prior_conditioned_consensus_backend_world",
    )
    glb_bytes = scene.export(file_type="glb")
    if not isinstance(glb_bytes, bytes):
        raise ValueError("failed to encode aligned conditioned-fusion GLB")

    static_metrics = candidate.get("static_cloud_room_bounds_metrics")
    visible_metrics = candidate.get("fixed_camera_visible_cloud_metrics")
    visible_structure_metrics = candidate.get(
        "fixed_camera_visible_structure_metrics"
    )
    reprojection = candidate.get("fixed_camera_reprojection")
    internal = candidate.get("internal_multiview_reprojection")
    if not all(
        isinstance(value, dict)
        for value in (
            static_metrics,
            visible_metrics,
            visible_structure_metrics,
            reprojection,
            internal,
        )
    ):
        raise ValueError("selected candidate evaluation metrics are incomplete")
    checks = _conditioned_admission_checks(
        candidate,
        metric_scale_preserved=abs(scale - 1.0) <= 1e-3,
    )
    if not all(checks.values()):
        raise ValueError(f"conditioned fusion did not pass bundle gates: {checks}")

    created_at = datetime.now(timezone.utc).isoformat()
    captured_at = str(state.get("created_at") or "")
    if not captured_at:
        raise ValueError("scan state has no capture timestamp")
    scan_id = scan_dir.name
    frame_rows = prepared.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("prepared-frame manifest has no frames")
    first_frame = frame_rows[0]
    if not isinstance(first_frame, dict):
        raise ValueError("prepared-frame manifest has an invalid frame row")
    frame_width = int(first_frame.get("width") or 0)
    frame_height = int(first_frame.get("height") or 0)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("prepared-frame dimensions are invalid")
    orientation = "landscape" if frame_width >= frame_height else "portrait"
    capture_slug = captured_at.replace("-", "").replace(":", "")[:15] + "Z"
    bundle_id = (
        f"{args.camera_id}_{orientation}-conditioned-fusion_"
        f"{capture_slug}_v1"
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        root = Path(temporary)
        aligned_relative = "aligned/conditioned_fusion_static_world.glb"
        _write_bytes(root, aligned_relative, glb_bytes)

        point_manifest = {
            "schema": "noesis.phone_walk.static_world_point_cloud_review.v1",
            "generated_at": created_at,
            "source": str(surfel_path),
            "source_point_count": int(points.shape[0]),
            "source_coordinate_frame": "da3_relative_metric_world",
            "output_coordinate_frame": "backend_world_m_stream_points",
            "alignment": alignment,
            "static_reference_points_included": False,
            "artifacts": {"point_cloud_glb": aligned_relative},
        }
        point_manifest_bytes = _json_bytes(point_manifest)
        point_manifest_relative = "evaluation/static_world_point_cloud_manifest.json"
        _write_bytes(root, point_manifest_relative, point_manifest_bytes)

        conditioned_report = {
            "schema": "noesis.phone_walk.conditioned_fusion.alignment_report.v1",
            "generated_at": created_at,
            "status": "passed",
            "selected_candidate": "prior_conditioned_consensus_da3_carrier",
            "source_capture_id": scan_id,
            "method": (
                "mapanything_da3_prior_conditioned_consensus_then_"
                "all_phone_pose_sim3_to_validated_da3_backend_path"
            ),
            "target": target,
            "quality_gate": {"passed": True, "checks": checks},
            "internal_multiview_reprojection": internal,
            "static_cloud_room_bounds_metrics": static_metrics,
            "fixed_camera_visible_cloud_metrics": visible_metrics,
            "fixed_camera_visible_structure_metrics": visible_structure_metrics,
            "fixed_camera_reprojection": reprojection,
            "source_alignment": {
                "schema": source_alignment.get("schema"),
                "sha256": _sha256(source_alignment_path.read_bytes()),
            },
            "conditioned_cloud_manifest": {
                "schema": point_manifest["schema"],
                "sha256": _sha256(point_manifest_bytes),
            },
        }
        _write_bytes(
            root,
            "alignment/alignment_report.json",
            _json_bytes(conditioned_report),
        )
        identity = {
            "schema": "noesis.phone_walk.conditioned_fusion.world_alignment.v1",
            "source_coordinate_frame": "backend_world_m_stream_points",
            "target_coordinate_frame": "backend_world_m_stream_points",
            "scale": 1.0,
            "world_from_source_row_major": np.eye(4).tolist(),
            "note": (
                "The selected GLB is already materialized in backend world meters; "
                "the upstream transform remains sealed in the cloud manifest and report."
            ),
        }
        _write_bytes(
            root,
            "alignment/identity_backend_world.json",
            _json_bytes(identity),
        )
        _copy_file(root, "calibration/camera_calibration.json", calibration)
        target_relative = f"target/{target_revision.name}/room_points_meta.json"
        _copy_file(root, target_relative, target_revision / "room_points_meta.json")
        _copy_file(root, "evaluation/evaluation_metrics.json", evaluation_path)

        reference = {
            "schema": "noesis.reference.room_scan_bundle.v1",
            "bundle_id": bundle_id,
            "created_at": created_at,
            "capture": {
                "scan_id": scan_id,
                "captured_at": captured_at,
                "source_type": "conditioned_multimodel_room_walk",
                "model": (
                    "facebook/map-anything-apache + depth-anything/DA3Metric-Large "
                    "+ prior_conditioned_consensus_da3_carrier"
                ),
                "orientation": orientation,
                "view_count": len(frame_rows),
            },
            "coordinate_contract": {
                "source_frame": "backend_world_m_stream_points",
                "target_frame": "backend_world_m_stream_points",
                "transform": "alignment/identity_backend_world.json",
            },
            "noesis_reference": {
                "camera_id": args.camera_id,
                "camera_calibration": "calibration/camera_calibration.json",
                "revision": f"target/{target_revision.name}",
            },
            "quality": {
                "passed": True,
                "admission": "validated_prior_conditioned_consensus",
            },
            "review_assets": {
                "aligned_rgb_glb": aligned_relative,
                "alignment_report": "alignment/alignment_report.json",
            },
        }
        _write_bytes(root, "reference.json", _json_bytes(reference))

        inventory_paths = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
        bundle_manifest = {
            "schema": "noesis.reference.room_scan_bundle_manifest.v1",
            "bundle_id": bundle_id,
            "files": [_file_entry(root, relative) for relative in inventory_paths],
        }
        _write_bytes(root, "bundle_manifest.json", _json_bytes(bundle_manifest))
        checksum_paths = sorted([*inventory_paths, "bundle_manifest.json"])
        checksum_lines = [
            f"{_sha256((root / relative).read_bytes())}  {relative}"
            for relative in checksum_paths
        ]
        _write_bytes(root, "SHA256SUMS", ("\n".join(checksum_lines) + "\n").encode())
        os.replace(root, output_dir)

    print(
        json.dumps(
            {
                "bundle_id": bundle_id,
                "output_dir": str(output_dir),
                "source_point_count": int(points.shape[0]),
                "aligned_glb_sha256": _sha256(glb_bytes),
                "quality_gate": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
