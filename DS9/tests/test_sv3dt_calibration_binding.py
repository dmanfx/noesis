"""Startup must reject SDK coordinates that would be labelled in another frame."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "noesis" / "v3dt_assets.py"
SPEC = importlib.util.spec_from_file_location("sv3dt_binding_assets_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets
SPEC.loader.exec_module(assets)


@pytest.fixture
def bound_camera(tmp_path):
    k = np.array([[625.0, 0.0, 914.0], [0.0, 624.0, 561.0], [0.0, 0.0, 1.0]])
    # The target world differs from the raw calibration by a real translation.
    # Supplying raw E below must fail even when its frame label is copied.
    e = np.eye(4)
    e[:3, 3] = [-2.0, 1.7, 4.0]
    snapshot = SimpleNamespace(
        camera_id="test-camera",
        intrinsics=k,
        extrinsics_col_major=tuple(e.flatten(order="F")),
        image_size=(1920, 1080),
        floor_y=0.0,
        floor_plane_normal=(0.0, 1.0, 0.0),
        floor_plane_offset_m=0.0,
        world_frame_id="backend_world_m",
        world_frame_revision="target-revision",
        frame_transform_sha256="a" * 64,
        camera_calibration_sha256="b" * 64,
    )
    axis = np.eye(4)[:, [0, 2, 1, 3]]
    payload = {
        "projectionMatrix_3x4_w2p": (k @ (e @ axis)[:3]).flatten().tolist(),
        "noesis_frame_binding": {
            key: getattr(snapshot, key)
            for key in (
                "world_frame_id", "world_frame_revision", "frame_transform_sha256",
                "camera_calibration_sha256", "floor_y",
            )
        },
    }
    payload["noesis_frame_binding"]["image_size"] = [1920, 1080]
    path = tmp_path / "camInfo.yml"
    provider = SimpleNamespace(world_snapshot=Mock(return_value=snapshot))
    cfg = {
        "v3dt": {"caminfo_world_axes": "xzy", "camera_order": ["test-camera"], "world_frame": "backend_world_m"},
        "streammux": {"width": 1920, "height": 1080},
        "tracker": {"tracker-width": 1920, "tracker-height": 1080},
    }

    def validate():
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        assets.validate_sv3dt_calibration_binding(
            SimpleNamespace(profile="sv3dt", camera_models=(path,)),
            pipeline_config=cfg,
            calibration_provider=provider,
            camera_labels={0: "test-camera"},
        )

    return SimpleNamespace(validate=validate, payload=payload, snapshot=snapshot, provider=provider, cfg=cfg)


def test_accepts_active_target_projection_and_positive_projective_scale(bound_camera):
    bound_camera.payload["projectionMatrix_3x4_w2p"] = [
        2.5 * value for value in bound_camera.payload["projectionMatrix_3x4_w2p"]
    ]
    bound_camera.validate()
    bound_camera.provider.world_snapshot.assert_called_once_with(0, "test-camera")


@pytest.mark.parametrize("fault", [None, "unscaled_projection", "stale_raster", "missing_space"])
def test_explicit_tracker_projection_requires_exact_inverse_and_binding(bound_camera, fault):
    bound_camera.cfg["v3dt"]["caminfo_pixel_space"] = "tracker"
    bound_camera.cfg["tracker"].update({"tracker-width": 960, "tracker-height": 544})
    binding = bound_camera.payload["noesis_frame_binding"]
    binding.update(projection_pixel_space="tracker", tracker_image_size=[960, 544])
    if fault != "unscaled_projection":
        p = np.array(bound_camera.payload["projectionMatrix_3x4_w2p"]).reshape(3, 4)
        bound_camera.payload["projectionMatrix_3x4_w2p"] = (
            np.diag([960 / 1920, 544 / 1080, 1.0]) @ p
        ).flatten().tolist()
    if fault == "stale_raster":
        binding["tracker_image_size"] = [960, 540]
    elif fault == "missing_space":
        binding.pop("projection_pixel_space")
    if fault is None:
        bound_camera.validate()
    else:
        with pytest.raises(assets.V3DTAssetError):
            bound_camera.validate()


@pytest.mark.parametrize("fault", [None, "tracker", "mux", "space"])
def test_asset_preflight_rejects_stale_tracker_raster_binding(tmp_path, fault):
    binding = {
        "projection_pixel_space": "tracker", "tracker_image_size": [960,544],
        "image_size": [1920,1080],
    }
    if fault == "tracker":
        binding["tracker_image_size"] = [1920,1088]
    elif fault == "mux":
        binding["image_size"] = [960,544]
    elif fault == "space":
        binding.pop("projection_pixel_space")
    path = tmp_path / "camInfo.yml"
    path.write_text(yaml.safe_dump({
        "projectionMatrix_3x4_w2p": np.eye(3,4).flatten().tolist(),
        "modelInfo": {"height": 1.7, "radius": 0.35},
        "noesis_frame_binding": binding,
    }))
    errors = []
    assets._validate_caminfo(path, errors, pixel_space="tracker", tracker_raster=(960,544))
    assert bool(errors) == (fault is not None), errors


@pytest.mark.parametrize("fault", ["image_y_flip", "raw_calibration_frame", "negative_scale", "nonfinite"])
def test_rejects_incorrect_sdk_projection(bound_camera, fault):
    p = np.array(bound_camera.payload["projectionMatrix_3x4_w2p"]).reshape(3, 4)
    if fault == "image_y_flip":
        p[1] = 1080 * p[2] - p[1]
    elif fault == "raw_calibration_frame":
        p[:, 3] = 0.0
    elif fault == "negative_scale":
        p *= -1.0
    else:
        p[0, 0] = np.nan
    bound_camera.payload["projectionMatrix_3x4_w2p"] = p.flatten().tolist()
    with pytest.raises(assets.V3DTAssetError, match="projection"):
        bound_camera.validate()


@pytest.mark.parametrize("field", ["world_frame_revision", "frame_transform_sha256", "camera_calibration_sha256"])
def test_rejects_stale_binding_even_with_equal_projection(bound_camera, field):
    bound_camera.payload["noesis_frame_binding"][field] = "stale"
    with pytest.raises(assets.V3DTAssetError, match=field):
        bound_camera.validate()


@pytest.mark.parametrize("fault", ["floor_height", "floor_tilt", "raster", "tracker_raster", "provenance_raster", "axis"])
def test_rejects_floor_raster_and_axis_mismatches(bound_camera, fault):
    if fault == "floor_height":
        bound_camera.snapshot.floor_y = 0.2
    elif fault == "floor_tilt":
        bound_camera.snapshot.floor_plane_normal = (0.0, 0.99, 0.1)
    elif fault == "raster":
        bound_camera.snapshot.image_size = (1280, 720)
    elif fault == "tracker_raster":
        bound_camera.cfg["tracker"]["tracker-width"] = 960
    elif fault == "provenance_raster":
        bound_camera.payload["noesis_frame_binding"]["image_size"] = [1280, 720]
    else:
        bound_camera.cfg["v3dt"]["caminfo_world_axes"] = "xyz"
    with pytest.raises(assets.V3DTAssetError):
        bound_camera.validate()


def test_rejects_missing_active_world_snapshot(bound_camera):
    bound_camera.provider.world_snapshot.return_value = None
    with pytest.raises(assets.V3DTAssetError, match="unavailable"):
        bound_camera.validate()
