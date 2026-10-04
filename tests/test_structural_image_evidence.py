"""Image-bound structural point projection, separate from person localization."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from PIL import Image

from noesis.calibration.structural_image_evidence import (
    project_structural_image_point,
    verify_structural_image_point,
)
from noesis.calibration.structural_registration import StructuralRegistrationError, WallSegment


def _fixture(tmp_path):
    image = tmp_path / "rectified.png"
    Image.new("RGB", (640, 480), (80, 100, 120)).save(image)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    camera_to_world[:3, 3] = [0.0, 2.0, 0.0]
    E = np.linalg.inv(camera_to_world)
    bundle = {
        "cameras": {
            "E": {"room": E.flatten(order="F").tolist()},
            "K": {"room": [500.0, 500.0, 320.0, 240.0]},
            "frame_bindings": {"room": {
                "calibration_frame": {"frame_id": "backend_world_m", "revision": "calibration"},
                "world_frame": {"frame_id": "backend_world_m", "revision": "sceneprior_room"},
                "scene_prior_id": "sceneprior_room",
                "target_from_calibration_col_major": np.eye(4).flatten(order="F").tolist(),
                "world_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
                "camera_calibration_sha256": "a" * 64,
                "transform_sha256": "b" * 64,
            }},
        },
    }
    nz = math.sqrt(0.99)
    wall = WallSegment(
        "pcf_plane_00", (-1.0, 5.0), (1.0, 5.0), 350, 2.0,
        ("pcf:points.glb", "method:vertical_plane_ransac", "voxel_m:0.060",
         "bounds_xz_m:none", f"normal_xyz:0.000000000,0.100000000,{nz:.9f}",
         f"plane_d_m:{-5 * nz:.9f}", f"support_sha256:{'c' * 64}"),
    )
    return image, bundle, wall


def test_opening_hits_wall_then_reduces_along_tangent_to_floor_line(tmp_path) -> None:
    image, bundle, wall = _fixture(tmp_path)
    result = project_structural_image_point(
        bundle=bundle, camera_id="room", image_path=image, image_size=[640, 480],
        pixel=[320.0, 240.0], kind="opening", point_id="jamb", plane_segment=wall,
        surface_y_m=None,
    )
    assert result["point"]["xz"] == pytest.approx([0.0, 5.0], abs=1e-7)
    assert result["image_evidence"]["ray_hit_target_m"][1] == pytest.approx(2.0)
    assert result["image_evidence"]["ray_hit_target_m"][2] < 4.9
    assert result["image_evidence"]["method"] == "ray_measured_pcf_wall_then_floor_line_tangent_v1"
    assert verify_structural_image_point(result, bundle=bundle, camera_id="room", image_path=image,
                                         image_size=[640, 480], pixel=[320.0, 240.0], kind="opening",
                                         point_id="jamb", plane_segment=wall, surface_y_m=None) == result


def test_floor_contact_uses_explicit_support_height_and_not_wall(tmp_path) -> None:
    image, bundle, wall = _fixture(tmp_path)
    result = project_structural_image_point(
        bundle=bundle, camera_id="room", image_path=image, image_size=[640, 480],
        pixel=[320.0, 340.0], kind="corner", point_id="step_lower_corner",
        plane_segment=None, surface_y_m=0.0,
    )
    assert result["point"]["xz"] == pytest.approx([0.0, 10.0], abs=1e-7)
    assert result["image_evidence"]["ray_hit_target_m"][1] == pytest.approx(0.0)
    upper = project_structural_image_point(
        bundle=bundle, camera_id="room", image_path=image, image_size=[640, 480],
        pixel=[320.0, 340.0], kind="corner", point_id="step_upper_corner",
        plane_segment=None, surface_y_m=0.1905,
    )
    assert upper["point"]["xz"] == pytest.approx([0.0, 9.0475], abs=1e-7)
    assert upper["image_evidence"]["ray_hit_target_m"][1] == pytest.approx(0.1905)
    with pytest.raises(StructuralRegistrationError):
        project_structural_image_point(bundle=bundle, camera_id="room", image_path=image,
                                       image_size=[640, 480], pixel=[320.0, 240.0], kind="opening",
                                       point_id="bad", plane_segment=wall, surface_y_m=0.0)


def test_exact_raster_and_bundle_binding_cannot_be_changed(tmp_path) -> None:
    image, bundle, wall = _fixture(tmp_path)
    kwargs = dict(bundle=bundle, camera_id="room", image_path=image, image_size=[640, 480],
                  pixel=[320.0, 240.0], kind="opening", point_id="jamb", plane_segment=wall,
                  surface_y_m=None)
    frozen = project_structural_image_point(**kwargs)
    tampered = copy.deepcopy(frozen)
    tampered["point"]["xz"][1] += 0.5
    with pytest.raises(StructuralRegistrationError):
        verify_structural_image_point(tampered, **kwargs)
    with pytest.raises(StructuralRegistrationError):
        project_structural_image_point(**{**kwargs, "image_size": [800, 480]})
    wrong_bundle = copy.deepcopy(bundle)
    wrong_bundle["cameras"]["K"]["room"][0] = 600.0
    with pytest.raises(StructuralRegistrationError):
        verify_structural_image_point(frozen, **{**kwargs, "bundle": wrong_bundle})
