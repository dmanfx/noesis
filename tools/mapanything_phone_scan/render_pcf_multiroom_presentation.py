#!/usr/bin/env python3
"""Render a fused PCF home view in one reference camera's ground frame.

This command is deliberately presentation-only.  It reads an existing
multi-room reintegration NPZ, derives the recorded camera-ground basis from a
Scene Prior revision manifest, transforms every point and both retained phone
camera paths in memory, and writes only PNG/GLB review assets plus a provenance
manifest.  The backend-world NPZ is hashed before and after rendering and is
never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.coordinate_frames import (  # noqa: E402
    CAMERA_LOCAL_RASTER_ORIENTATION,
    CameraGroundFrame,
    transform_positions,
)


class PCFPresentationError(RuntimeError):
    """Raised when a presentation frame or source artifact is unsafe."""


@dataclass(frozen=True)
class PresentationFrame:
    reference_camera_id: str
    source_coordinate_frame: str
    target_coordinate_frame: str
    camera_position_world_m: np.ndarray
    camera_right_world_xz: np.ndarray
    camera_forward_world_xz: np.ndarray
    floor_y_m: float
    world_to_presentation: np.ndarray


_REQUIRED_NPZ_FIELDS = {
    "points",
    "colors",
    "owner_room_id",
    "fixed_camera_positions",
    "moving_camera_positions",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _as_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise PCFPresentationError(f"{label} must contain {size} finite values")
    return result


def derive_presentation_frame(scene_prior_manifest: Path) -> PresentationFrame:
    """Derive backend-world to camera-local-ground from recorded metadata."""

    payload = json.loads(scene_prior_manifest.read_text(encoding="utf-8"))
    if payload.get("contract") != "noesis.scene_prior.revision":
        raise PCFPresentationError(
            f"{scene_prior_manifest} is not a Scene Prior revision manifest"
        )
    preview = payload.get("preview")
    derivation = payload.get("derivation")
    grid = payload.get("grid")
    if not isinstance(preview, dict) or not isinstance(derivation, dict):
        raise PCFPresentationError(
            f"{scene_prior_manifest} lacks preview/derivation metadata"
        )
    if not isinstance(grid, dict) or grid.get("coordinate_frame") != "backend_world_m":
        raise PCFPresentationError("Scene Prior grid is not in backend_world_m")
    if preview.get("coordinate_frame") != "camera_local_ground_m":
        raise PCFPresentationError(
            "Scene Prior preview is not in camera_local_ground_m"
        )
    if preview.get("orientation") not in {
        CAMERA_LOCAL_RASTER_ORIENTATION,
        # Read compatibility for already-built immutable Scene Priors.  This
        # legacy label described the pre-PNG numeric grid; it does not change
        # the camera basis derived below.
        "row_increases_camera_forward_column_increases_camera_right",
    }:
        raise PCFPresentationError("Scene Prior preview orientation is unsupported")

    camera = _as_vector(
        preview.get("camera_position_world_m"), 3, "camera_position_world_m"
    )
    right = _as_vector(
        preview.get("camera_right_world_xz"), 2, "camera_right_world_xz"
    )
    forward = _as_vector(
        preview.get("camera_forward_world_xz"), 2, "camera_forward_world_xz"
    )
    floor_y_m = float(derivation.get("floor_y_m", float("nan")))
    if not math.isfinite(floor_y_m):
        raise PCFPresentationError("Scene Prior floor_y_m is not finite")
    if not math.isclose(float(np.linalg.norm(right)), 1.0, abs_tol=2e-6):
        raise PCFPresentationError("camera-right ground vector is not unit length")
    if not math.isclose(float(np.linalg.norm(forward)), 1.0, abs_tol=2e-6):
        raise PCFPresentationError("camera-forward ground vector is not unit length")
    if not math.isclose(float(np.dot(right, forward)), 0.0, abs_tol=2e-6):
        raise PCFPresentationError("camera right/forward ground vectors are not orthogonal")

    transform = CameraGroundFrame(
        camera_world_m=camera,
        camera_right_world=np.asarray([right[0], 0.0, right[1]]),
        camera_forward_world=np.asarray([forward[0], 0.0, forward[1]]),
    ).world_to_camera_local_display_matrix(
        floor_y_m
    )
    determinant = float(np.linalg.det(transform[:3, :3]))
    if not math.isclose(determinant, -1.0, abs_tol=2e-6):
        raise PCFPresentationError(
            "camera presentation must be the display-only OpenCV-down to "
            f"height-up basis (determinant -1), got {determinant:.9f}"
        )
    camera_ground = transform_positions(camera[None], transform)[0]
    if not np.allclose(camera_ground[[0, 2]], 0.0, atol=2e-6):
        raise PCFPresentationError("reference camera does not map to ground origin")
    return PresentationFrame(
        reference_camera_id=str(preview.get("reference_camera_id") or "").strip(),
        source_coordinate_frame="backend_world_m",
        target_coordinate_frame="camera_local_ground_m",
        camera_position_world_m=camera,
        camera_right_world_xz=right,
        camera_forward_world_xz=forward,
        floor_y_m=floor_y_m,
        world_to_presentation=transform,
    )

def _load_source_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "noesis.pcf.multiroom_reintegration.v1":
        raise PCFPresentationError(f"{path} is not a PCF reintegration manifest")
    return payload


def _room_names(source_manifest: Mapping[str, Any]) -> tuple[str, str]:
    contract = source_manifest.get("npz_contract")
    room_ids = contract.get("room_ids") if isinstance(contract, dict) else None
    if isinstance(room_ids, dict):
        return str(room_ids.get("1", "fixed-room")), str(
            room_ids.get("2", "moving-room")
        )
    fixed = source_manifest.get("fixed_room")
    moving = source_manifest.get("moving_room")
    return (
        str(fixed.get("name", "fixed-room")) if isinstance(fixed, dict) else "fixed-room",
        str(moving.get("name", "moving-room"))
        if isinstance(moving, dict)
        else "moving-room",
    )


def _cylinder_between(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: np.ndarray,
) -> Any | None:
    import trimesh

    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-8:
        return None
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    alignment = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector / length)
    if alignment is None:
        alignment = np.eye(4, dtype=np.float64)
    alignment[:3, 3] = (np.asarray(start) + np.asarray(end)) * 0.5
    mesh.apply_transform(alignment)
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def _write_glb(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    owners: np.ndarray,
    fixed_path: np.ndarray,
    moving_path: np.ndarray,
    fixed_name: str,
    moving_name: str,
) -> None:
    import trimesh

    scene = trimesh.Scene()
    for room_id, name in ((1, fixed_name), (2, moving_name)):
        selected = owners == room_id
        if np.any(selected):
            rgba = np.column_stack(
                (
                    colors[selected],
                    np.full(np.count_nonzero(selected), 235, dtype=np.uint8),
                )
            )
            scene.add_geometry(
                trimesh.points.PointCloud(points[selected], colors=rgba),
                node_name=f"{name}_presentation_surfels",
            )
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    radius = max(0.0012 * extent, 0.003)
    for camera_path, name, color in (
        (fixed_path, fixed_name, np.asarray([93, 195, 255, 255], dtype=np.uint8)),
        (moving_path, moving_name, np.asarray([255, 174, 82, 255], dtype=np.uint8)),
    ):
        segments = [
            item
            for item in (
                _cylinder_between(start, end, radius, color)
                for start, end in zip(camera_path[:-1], camera_path[1:])
            )
            if item is not None
        ]
        if segments:
            scene.add_geometry(
                trimesh.util.concatenate(segments),
                node_name=f"{name}_phone_camera_path",
            )
    marker = trimesh.creation.icosphere(subdivisions=2, radius=max(radius * 4.0, 0.04))
    marker.apply_translation([0.0, 0.0, 0.0])
    marker.visual.vertex_colors = np.tile(
        np.asarray([99, 255, 159, 255], dtype=np.uint8),
        (len(marker.vertices), 1),
    )
    scene.add_geometry(marker, node_name="reference_camera_ground_origin")
    scene.export(path)


def _write_png(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    owners: np.ndarray,
    fixed_path: np.ndarray,
    moving_path: np.ndarray,
    fixed_name: str,
    moving_name: str,
    reference_camera_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    figure.patch.set_facecolor("#10151d")
    for axis in axes:
        axis.set_facecolor("#10151d")
        axis.tick_params(colors="#d5dce7")
        axis.xaxis.label.set_color("#d5dce7")
        axis.yaxis.label.set_color("#d5dce7")
        for spine in axis.spines.values():
            spine.set_color("#667386")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("camera right (m)")
        axis.set_ylabel("camera forward (m) · up")
        axis.grid(alpha=0.12)
    axes[0].scatter(
        points[:, 0],
        points[:, 2],
        s=0.25,
        c=colors.astype(np.float32) / 255.0,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Continuous PCF RGB · complete fused evidence", color="white")
    palette = np.asarray(
        [[0, 0, 0], [91, 189, 255], [255, 171, 75]], dtype=np.float32
    ) / 255.0
    axes[1].scatter(
        points[:, 0],
        points[:, 2],
        s=0.25,
        c=palette[owners],
        linewidths=0,
        rasterized=True,
    )
    for axis in axes:
        axis.plot(
            fixed_path[:, 0],
            fixed_path[:, 2],
            color="#5bbdff",
            linewidth=1.7,
            label=f"{fixed_name} phone path",
        )
        axis.plot(
            moving_path[:, 0],
            moving_path[:, 2],
            color="#ffab4b",
            linewidth=1.7,
            label=f"{moving_name} phone path",
        )
        axis.scatter(
            [0.0],
            [0.0],
            marker="^",
            s=95,
            color="#63ff9f",
            edgecolor="white",
            linewidth=0.8,
            zorder=10,
            label=f"{reference_camera_id} camera",
        )
    axes[1].set_title(
        f"Ownership · {fixed_name} fixed · {moving_name} registered",
        color="white",
    )
    axes[1].legend(
        facecolor="#10151d", edgecolor="#8290a4", labelcolor="white", framealpha=0.9
    )
    finite_xz = points[np.isfinite(points).all(axis=1)][:, [0, 2]]
    low = np.min(finite_xz, axis=0) - 0.35
    high = np.max(finite_xz, axis=0) + 0.35
    for axis in axes:
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
    figure.suptitle(
        f"PCF multi-room review in {reference_camera_id} camera-ground presentation",
        color="white",
        fontsize=16,
    )
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_presentation(
    *,
    source_npz: Path,
    scene_prior_manifest: Path,
    output_dir: Path,
    source_reintegration_manifest: Path | None = None,
) -> dict[str, Any]:
    """Write presentation-only assets while proving source geometry immutability."""

    source_npz = source_npz.resolve()
    scene_prior_manifest = scene_prior_manifest.resolve()
    source_reintegration_manifest = (
        source_reintegration_manifest.resolve()
        if source_reintegration_manifest is not None
        else None
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise PCFPresentationError(f"output directory already exists: {output_dir}")
    source_hash_before = _sha256(source_npz)
    source_size_before = source_npz.stat().st_size
    frame = derive_presentation_frame(scene_prior_manifest)
    source_manifest = _load_source_manifest(source_reintegration_manifest)
    fixed_name, moving_name = _room_names(source_manifest)
    with np.load(source_npz, allow_pickle=False) as row:
        missing = sorted(_REQUIRED_NPZ_FIELDS.difference(row.files))
        if missing:
            raise PCFPresentationError(f"{source_npz} is missing {missing}")
        backend_points = np.asarray(row["points"], dtype=np.float32)
        colors = np.asarray(row["colors"], dtype=np.uint8)
        owners = np.asarray(row["owner_room_id"], dtype=np.uint8)
        backend_fixed_path = np.asarray(row["fixed_camera_positions"], dtype=np.float32)
        backend_moving_path = np.asarray(row["moving_camera_positions"], dtype=np.float32)
    if backend_points.ndim != 2 or backend_points.shape[1] != 3 or not len(backend_points):
        raise PCFPresentationError("source points must be one non-empty N x 3 array")
    if colors.shape != backend_points.shape or owners.shape != (len(backend_points),):
        raise PCFPresentationError("source point color/owner arrays are inconsistent")
    if not set(np.unique(owners).tolist()).issubset({1, 2}):
        raise PCFPresentationError("owner_room_id contains unsupported room IDs")
    for name, value in (
        ("fixed camera path", backend_fixed_path),
        ("moving camera path", backend_moving_path),
    ):
        if value.ndim != 2 or value.shape[1] != 3 or not len(value):
            raise PCFPresentationError(f"{name} must be one non-empty N x 3 array")

    transform = frame.world_to_presentation
    presentation_points = transform_positions(backend_points, transform).astype(np.float32)
    presentation_fixed_path = transform_positions(
        backend_fixed_path, transform
    ).astype(np.float32)
    presentation_moving_path = transform_positions(
        backend_moving_path, transform
    ).astype(np.float32)
    camera_in_presentation = transform_positions(
        frame.camera_position_world_m[None], transform
    )[0]
    output_dir.mkdir(parents=True)
    png_path = output_dir / "multiroom_pcf_family_presentation.png"
    glb_path = output_dir / "multiroom_pcf_family_presentation.glb"
    _write_png(
        png_path,
        presentation_points,
        colors,
        owners,
        presentation_fixed_path,
        presentation_moving_path,
        fixed_name,
        moving_name,
        frame.reference_camera_id,
    )
    _write_glb(
        glb_path,
        presentation_points,
        colors,
        owners,
        presentation_fixed_path,
        presentation_moving_path,
        fixed_name,
        moving_name,
    )

    source_hash_after = _sha256(source_npz)
    source_size_after = source_npz.stat().st_size
    backend_unchanged = bool(
        source_hash_before == source_hash_after
        and source_size_before == source_size_after
    )
    if not backend_unchanged:
        raise PCFPresentationError("backend source NPZ changed during rendering")
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    report: dict[str, Any] = {
        "schema": "noesis.pcf.multiroom_presentation.v1",
        "generated_at": _utc_now(),
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "presentation_only": True,
        "backend_geometry": {
            "mutated": False,
            "transformed_npz_written": False,
            "source_coordinate_frame": frame.source_coordinate_frame,
            "source_npz": str(source_npz),
            "source_npz_size_bytes_before": source_size_before,
            "source_npz_size_bytes_after": source_size_after,
            "source_npz_sha256_before": source_hash_before,
            "source_npz_sha256_after": source_hash_after,
            "source_npz_unchanged": backend_unchanged,
            "backend_points_sha256": _array_sha256(backend_points),
            "backend_fixed_camera_path_sha256": _array_sha256(backend_fixed_path),
            "backend_moving_camera_path_sha256": _array_sha256(backend_moving_path),
            "point_count": int(len(backend_points)),
        },
        "source_reintegration_manifest": (
            {
                "path": str(source_reintegration_manifest),
                "sha256": _sha256(source_reintegration_manifest),
                "status": source_manifest.get("status"),
                "registration": source_manifest.get("registration"),
            }
            if source_reintegration_manifest is not None
            else None
        ),
        "scene_prior_manifest": {
            "path": str(scene_prior_manifest),
            "sha256": _sha256(scene_prior_manifest),
            "prior_id": json.loads(
                scene_prior_manifest.read_text(encoding="utf-8")
            ).get("prior_id"),
        },
        "presentation_frame": {
            "reference_camera_id": frame.reference_camera_id,
            "source_coordinate_frame": frame.source_coordinate_frame,
            "target_coordinate_frame": frame.target_coordinate_frame,
            "camera_position_world_m": frame.camera_position_world_m.tolist(),
            "camera_right_world_xz": frame.camera_right_world_xz.tolist(),
            "camera_forward_world_xz": frame.camera_forward_world_xz.tolist(),
            "floor_y_m": frame.floor_y_m,
            "world_to_presentation_row_major": transform.tolist(),
            "linear_determinant": determinant,
            "absolute_determinant": abs(determinant),
            "orthonormal_max_abs_error": float(
                np.max(np.abs(linear.T @ linear - np.eye(3)))
            ),
            "handedness": (
                "camera_display_reflection" if determinant < 0.0 else "proper_rotation"
            ),
            "reference_camera_presentation_xyz_m": camera_in_presentation.tolist(),
            "ground_origin_error_m": float(
                np.linalg.norm(camera_in_presentation[[0, 2]])
            ),
            "screen_contract": {
                "horizontal": "camera_right_positive_x",
                "vertical": "camera_forward_positive_z_up",
                "camera_ground_origin_xz_m": [0.0, 0.0],
            },
        },
        "outputs": {
            "png": {
                "path": png_path.name,
                "sha256": _sha256(png_path),
                "size_bytes": png_path.stat().st_size,
            },
            "glb": {
                "path": glb_path.name,
                "sha256": _sha256(glb_path),
                "size_bytes": glb_path.stat().st_size,
            },
        },
    }
    report_path = output_dir / "presentation_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _write_self_test_manifest(path: Path) -> None:
    payload = {
        "contract": "noesis.scene_prior.revision",
        "contract_version": 1,
        "prior_id": "sceneprior-synthetic",
        "space_id": "synthetic",
        "grid": {"coordinate_frame": "backend_world_m"},
        "derivation": {"floor_y_m": 0.25},
        "preview": {
            "coordinate_frame": "camera_local_ground_m",
            "orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
            "reference_camera_id": "camera-synthetic",
            "camera_position_world_m": [10.0, 2.0, 20.0],
            "camera_right_world_xz": [0.6, 0.8],
            "camera_forward_world_xz": [0.8, -0.6],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="pcf_presentation_smoke_") as value:
        root = Path(value)
        prior_manifest = root / "scene_prior_manifest.json"
        _write_self_test_manifest(prior_manifest)
        frame = derive_presentation_frame(prior_manifest)
        camera = frame.camera_position_world_m
        right_world = np.asarray([0.6, 0.0, 0.8])
        forward_world = np.asarray([0.8, 0.0, -0.6])
        probes = np.stack((camera, camera + right_world, camera + forward_world))
        presented = transform_positions(probes, frame.world_to_presentation)
        np.testing.assert_allclose(presented[0, [0, 2]], [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(presented[1, [0, 2]], [1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(presented[2, [0, 2]], [0.0, 1.0], atol=1e-7)

        points = np.asarray(
            [camera + right_world, camera + forward_world, camera - right_world],
            dtype=np.float32,
        )
        source_npz = root / "source.npz"
        np.savez_compressed(
            source_npz,
            points=points,
            colors=np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            owner_room_id=np.asarray([1, 2, 1], dtype=np.uint8),
            fixed_camera_positions=np.stack((camera, camera + forward_world)).astype(
                np.float32
            ),
            moving_camera_positions=np.stack((camera - right_world, camera)).astype(
                np.float32
            ),
        )
        reintegration_manifest = root / "reintegration.json"
        reintegration_manifest.write_text(
            json.dumps(
                {
                    "schema": "noesis.pcf.multiroom_reintegration.v1",
                    "status": "review_only",
                    "registration": {"accepted_for_canonical_use": False},
                    "npz_contract": {"room_ids": {"1": "family", "2": "kitchen"}},
                }
            ),
            encoding="utf-8",
        )
        source_hash = _sha256(source_npz)
        report = render_presentation(
            source_npz=source_npz,
            scene_prior_manifest=prior_manifest,
            source_reintegration_manifest=reintegration_manifest,
            output_dir=root / "output",
        )
        if _sha256(source_npz) != source_hash:
            raise AssertionError("source NPZ changed during presentation self-test")
        if report["status"] != "review_only" or not report["presentation_only"]:
            raise AssertionError("presentation authority was overstated")
        if report["presentation_frame"]["ground_origin_error_m"] > 1e-7:
            raise AssertionError("reference camera did not map to ground origin")
        if not report["backend_geometry"]["source_npz_unchanged"]:
            raise AssertionError("source immutability proof failed")
        for name in (
            "multiroom_pcf_family_presentation.png",
            "multiroom_pcf_family_presentation.glb",
            "presentation_manifest.json",
        ):
            if not (root / "output" / name).is_file():
                raise AssertionError(f"missing self-test output {name}")
    print("PCF multi-room presentation smoke: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path)
    parser.add_argument("--scene-prior-manifest", type=Path)
    parser.add_argument("--source-reintegration-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.self_test:
        _self_test()
        return 0
    missing = [
        name
        for name in (
            "source_npz",
            "scene_prior_manifest",
            "source_reintegration_manifest",
            "output_dir",
        )
        if getattr(arguments, name) is None
    ]
    if missing:
        raise PCFPresentationError(
            "missing required arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    report = render_presentation(
        source_npz=arguments.source_npz,
        scene_prior_manifest=arguments.scene_prior_manifest,
        source_reintegration_manifest=arguments.source_reintegration_manifest,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
