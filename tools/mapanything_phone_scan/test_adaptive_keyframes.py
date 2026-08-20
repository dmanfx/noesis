from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.mapanything_phone_scan.processing import FramePreparationSettings
from tools.mapanything_phone_scan.supplement import _transform_points
from tools.mapanything_phone_scan.windowed_inference import (
    _refine_window_scale_translation,
    _window_ranges,
)


def _write_raw(path: Path, points: np.ndarray) -> None:
    height, width = points.shape[:2]
    np.savez_compressed(
        path,
        world_points=points.astype(np.float32),
        depth_z=np.full((height, width), 2.0, dtype=np.float32),
        confidence=np.full((height, width), 1.0, dtype=np.float32),
        mask=np.ones((height, width), dtype=np.uint8),
        camera_pose=np.eye(4, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        metric_scaling_factor=np.ones(1, dtype=np.float32),
        model_rgb=np.zeros((height, width, 3), dtype=np.float32),
    )


def test_default_selector_has_high_emergency_capacity() -> None:
    settings = FramePreparationSettings()
    assert settings.candidate_fps == 4.0
    assert settings.max_selected_frames == 256


def test_window_ranges_cover_all_views_with_exact_overlap() -> None:
    ranges = _window_ranges(182, 80, 24)
    assert ranges == [(0, 80), (56, 136), (112, 182)]
    assert ranges[0][1] - ranges[1][0] == 24
    assert ranges[1][1] - ranges[2][0] == 24


def test_transform_points_preserves_image_grid_shape() -> None:
    points = np.arange(60, dtype=np.float64).reshape((4, 5, 3))
    transformed = _transform_points(
        points,
        2.0,
        np.eye(3, dtype=np.float64),
        np.asarray([1.0, -2.0, 3.0]),
    )
    assert transformed.shape == points.shape
    np.testing.assert_allclose(transformed, points * 2.0 + [1.0, -2.0, 3.0])


def test_duplicate_surfaces_refine_scale_and_translation(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    true_scale = 1.12
    true_translation = np.asarray([0.35, -0.08, 0.21], dtype=np.float64)
    base_records = []
    append_rows = []
    append_poses = []
    base_poses = []
    for index in range(4):
        source = rng.normal(size=(20, 20, 3)).astype(np.float64)
        source[..., 2] += 4.0
        target = true_scale * source + true_translation
        raw_path = tmp_path / f"base_{index}.npz"
        _write_raw(raw_path, target)
        base_records.append(
            {
                "raw_path": raw_path,
                "scale": 1.0,
                "rotation": np.eye(3),
                "translation": np.zeros(3),
            }
        )
        append_rows.append(
            {
                "world_points": source,
                "depth_z": np.full((20, 20), 2.0),
                "confidence": np.ones((20, 20)),
                "mask": np.ones((20, 20), dtype=np.uint8),
                "camera_pose": np.eye(4),
                "intrinsics": np.eye(3),
                "model_rgb": np.zeros((20, 20, 3)),
            }
        )
        append_pose = np.eye(4, dtype=np.float64)
        append_pose[:3, 3] = [index * 0.5, 0.0, index * 0.1]
        base_pose = append_pose.copy()
        base_pose[:3, 3] = true_scale * append_pose[:3, 3] + true_translation
        append_poses.append(append_pose)
        base_poses.append(base_pose)

    scale, translation, metrics = _refine_window_scale_translation(
        base_records,
        append_rows,
        np.stack(base_poses),
        np.stack(append_poses),
        1.04,
        np.eye(3),
        np.asarray([0.22, -0.03, 0.10]),
    )

    assert abs(scale - true_scale) < 1e-5
    np.testing.assert_allclose(translation, true_translation, atol=1e-5)
    assert all(metrics["checks"].values())
