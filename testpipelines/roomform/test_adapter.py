from __future__ import annotations

from pathlib import Path

import numpy as np

from testpipelines.roomform import main as MODULE


def test_axis_transform_is_right_handed_and_maps_y_to_z() -> None:
    transform = MODULE.AXIS_TRANSFORM
    assert np.isclose(np.linalg.det(transform), 1.0)
    result = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32) @ transform.T
    np.testing.assert_allclose(result, [[1.0, -3.0, 2.0]])


def test_da3_transform_is_right_handed_and_maps_y_down_to_z_up() -> None:
    transform = MODULE.DA3_TO_ROOMFORM_Z_UP
    assert np.isclose(np.linalg.det(transform), 1.0)
    result = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32) @ transform.T
    np.testing.assert_allclose(result, [[1.0, 3.0, -2.0]])


def test_load_point_preserving_fusion(tmp_path: Path) -> None:
    cloud_path = tmp_path / "point_preserving_fusion_2cm.npz"
    points = np.arange(1024 * 3, dtype=np.float32).reshape(1024, 3) / 100.0
    colors = np.full((1024, 3), 127, dtype=np.uint8)
    np.savez_compressed(
        cloud_path,
        points=points,
        colors=colors,
        weights=np.ones(1024, dtype=np.float32),
        sample_counts=np.full(1024, 2, dtype=np.uint32),
        view_counts=np.full(1024, 2, dtype=np.uint16),
    )
    (tmp_path / "point_preserving_fusion_report.json").write_text(
        '{"schema":"noesis.phone_walk.point_preserving_fusion.v1",'
        '"view_count":2,"voxel_m":0.02,"method":"test"}',
        encoding="utf-8",
    )
    camera_solution_path = tmp_path / "camera_solution.npz"
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[1, :3, 3] = [1.0, 2.0, 3.0]
    np.savez_compressed(camera_solution_path, camera_poses=poses)

    cloud = MODULE.load_point_preserving_fusion(cloud_path, camera_solution_path)

    assert cloud.source_kind == "da3_mapanything_point_preserving_fusion"
    assert cloud.points.shape == (1024, 3)
    assert cloud.colors.shape == (1024, 3)
    np.testing.assert_allclose(cloud.stations[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cloud.axis_transform, MODULE.DA3_TO_ROOMFORM_Z_UP)


def test_phone_provider_accepts_legacy_mapanything_and_da3() -> None:
    assert MODULE._phone_provider(
        {"model": {"id": "facebook/map-anything-apache"}}
    ) == ("mapanything", "facebook/map-anything-apache")
    assert MODULE._phone_provider(
        {"provider": "da3", "model_id": "depth-anything/DA3-BASE"}
    ) == ("da3", "depth-anything/DA3-BASE")


def test_family_room_revision_discovery(tmp_path: Path) -> None:
    revision = tmp_path / "vt_family_room_stream_rgbmesh_1"
    revision.mkdir()
    (revision / "manifest.json").write_text(
        '{"camera":"family-room","created_ts_us":3}', encoding="utf-8"
    )
    (revision / "room_points_meta.json").write_text(
        '{"camera":"family-room",'
        '"color_source":"mapanything_persisted_zarr_rgb_texture",'
        '"source":"mapanything_zarr_current_calibration_backprojection_texture_colored"}',
        encoding="utf-8",
    )
    np.savez(
        revision / "room_points.npz",
        points=np.zeros((2, 3)),
        colors=np.zeros((2, 3)),
    )
    assert MODULE.find_cached_room_revision(tmp_path, "family-room") == revision


def test_revision_discovery_is_living_room_rgb_only(tmp_path: Path) -> None:
    wrong = tmp_path / "vt_living_room_stream_rgbmesh_1"
    wrong.mkdir()
    (wrong / "manifest.json").write_text('{"camera":"kitchen","created_ts_us":2}')
    (wrong / "room_points_meta.json").write_text(
        '{"camera":"kitchen","color_source":"mapanything_persisted_zarr_rgb_texture"}'
    )
    np.savez(wrong / "room_points.npz", points=np.zeros((2, 3)), colors=np.zeros((2, 3)))

    good = tmp_path / "vt_living_room_stream_rgbmesh_2"
    good.mkdir()
    (good / "manifest.json").write_text(
        '{"camera":"living-room","created_ts_us":3}'
    )
    (good / "room_points_meta.json").write_text(
        '{"camera":"living-room","color_source":"mapanything_persisted_zarr_rgb_texture",'
        '"source":"mapanything_zarr_current_calibration_backprojection_texture_colored"}'
    )
    np.savez(good / "room_points.npz", points=np.zeros((2, 3)), colors=np.zeros((2, 3)))
    assert MODULE.find_cached_living_room_revision(tmp_path) == good
