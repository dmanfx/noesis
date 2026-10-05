"""Measured parallel-floor geometry stays analytic, finite, and PCF-bounded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.build_parallel_floor_surfaces import (
    build,
    clip_convex,
    convex_polygon,
    project_polygon,
)


CALIBRATION_SHA = "c" * 64
TRANSFORM_SHA = "d" * 64
WORLD_REVISION = "test-world-r1"


def _camera_and_binding(normal: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    world_from_camera = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 3.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = {
        "E": np.eye(4).reshape(-1, order="F").tolist(),
        "K": [100.0, 100.0, 320.0, 240.0],
    }
    binding = {
        "target_from_calibration_col_major": world_from_camera.reshape(
            -1, order="F"
        ).tolist(),
        "world_floor_plane": {"normal": normal.tolist(), "offset_m": 0.0},
        "world_frame": {"revision": WORLD_REVISION},
        "camera_calibration_sha256": CALIBRATION_SHA,
        "transform_sha256": TRANSFORM_SHA,
    }
    return camera, binding, world_from_camera


def _build_inputs(tmp_path: Path, rise_m: float) -> Path:
    normal = np.asarray([0.1, np.sqrt(0.99), 0.0])
    camera, binding, _ = _camera_and_binding(normal)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "cameras": {"living-room": camera},
                "frame_binding": binding,
            }
        ),
        encoding="utf-8",
    )

    source_image_path = tmp_path / "source.png"
    Image.new("RGB", (640, 480), color=(0, 0, 0)).save(source_image_path)

    rows = columns = 80
    resolution = 0.05
    min_x = min_z = -2.0
    row_indices, column_indices = np.indices((rows, columns))
    x = min_x + (column_indices + 0.5) * resolution
    z = min_z + (row_indices + 0.5) * resolution
    lower_y = -(normal[0] * x + normal[2] * z) / normal[1]
    residual = np.where((row_indices + column_indices) % 2, 0.01, -0.01)
    grid_path = tmp_path / "pcf-grid.npz"
    np.savez_compressed(
        grid_path,
        observed=np.ones((rows, columns), dtype=np.uint8),
        floor_support_count=np.ones((rows, columns), dtype=np.uint32),
        floor_height_m=(lower_y + residual).astype(np.float32),
    )
    grid_sha = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "pcf-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prior_id": WORLD_REVISION,
                "preview": {
                    "camera_calibration": {"sha256": CALIBRATION_SHA},
                    "reference_camera_id": "living-room",
                },
                "artifacts": [{"role": "grid_npz", "sha256": grid_sha}],
                "grid": {
                    "resolution_m": resolution,
                    "bounds": {
                        "min_x": min_x,
                        "max_x": min_x + columns * resolution,
                        "min_z": min_z,
                        "max_z": min_z + rows * resolution,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    measurement_path = tmp_path / "measurements.json"
    measurement_path.write_text(
        json.dumps(
            {
                "calibration_path": str(calibration_path),
                "source_image_path": str(source_image_path),
                "pcf_manifest_path": str(manifest_path),
                "pcf_grid_path": str(grid_path),
                "camera_id": "living-room",
                "surface_id": "foyer-upper",
                "floor_model": "upper_parallel_to_calibrated_lower",
                "pixel_space": "rectified_source_image",
                "image_size": [640, 480],
                "rise_observations": [
                    {"floor_to_floor_rise_m": rise_m, "endpoint": "left"},
                    {"floor_to_floor_rise_m": rise_m, "endpoint": "right"},
                ],
                "upper_floor_extent_pixels": [
                    [240.0, 210.0],
                    [400.0, 210.0],
                    [400.0, 270.0],
                    [240.0, 270.0],
                ],
                "lower_reference_patch_pixels": [
                    [290.0, 210.0],
                    [350.0, 210.0],
                    [350.0, 270.0],
                    [290.0, 270.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    return measurement_path


def test_project_polygon_matches_analytic_camera_ray_plane_hit() -> None:
    camera, binding, _ = _camera_and_binding(np.asarray([0.0, 1.0, 0.0]))

    points = project_polygon(
        camera,
        binding,
        [[330.0, 250.0], [340.0, 250.0], [330.0, 260.0]],
        offset=-0.25,
    )

    # Pixel ray is (0.1, 0.1, 1); camera origin is at Y=3 and camera +Z
    # points along world -Y, so the Y=.25 plane is reached at range 2.75.
    assert points[0] == pytest.approx([0.275, 0.25, 0.275])


@pytest.mark.parametrize("rise_m", [0.1905, 0.3048, 0.4572])
def test_build_uses_measured_rise_retains_parallel_normal_and_clips_to_pcf(
    tmp_path: Path, rise_m: float
) -> None:
    measurement_path = _build_inputs(tmp_path, rise_m)

    surfaces, review = build(measurement_path)

    surface = surfaces.surfaces[0]
    expected_normal = (0.1, np.sqrt(0.99), 0.0)
    assert review["measured_rise_m"] == pytest.approx(rise_m)
    assert surface.normal_world == pytest.approx(expected_normal)
    assert surface.offset_m == pytest.approx(-expected_normal[1] * rise_m)

    x, z = 0.3, -0.2
    lower_y = -(expected_normal[0] * x + expected_normal[2] * z) / expected_normal[1]
    upper_y = -(expected_normal[0] * x + expected_normal[2] * z + surface.offset_m) / expected_normal[1]
    assert upper_y - lower_y == pytest.approx(rise_m)

    xs = [point[0] for point in surface.extent_xz]
    assert max(xs) == pytest.approx(1.975)
    assert min(xs) >= -1.975 - 1e-8
    assert surface.source_observation_count == review["pcf_covered_cells"]
    assert surface.source_observation_count < 80 * 80
    assert review["lower_local_rms_m"] == pytest.approx(0.01, abs=1e-6)


def test_nonconvex_annotation_is_rejected() -> None:
    concave = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [1.0, 0.5], [2.0, 2.0], [0.0, 2.0]]
    )

    with pytest.raises(ValueError, match="must be convex"):
        convex_polygon(concave)


def test_convex_clipping_returns_intersection_not_unbounded_annotation() -> None:
    subject = np.asarray([[-3.0, -1.0], [3.0, -1.0], [3.0, 1.0], [-3.0, 1.0]])
    bounds = np.asarray([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])

    clipped = clip_convex(subject, bounds)

    assert {tuple(point) for point in np.round(clipped, 8)} == {
        (-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)
    }
