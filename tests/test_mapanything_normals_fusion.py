from __future__ import annotations

import numpy as np

from noesis.virtual_twin.geometry import (
    PlaneCandidate,
    compute_depth_normals_camera,
    fuse_mapanything_with_planes,
)


def _synthetic_plane() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = 72, 96
    k = np.asarray([[110.0, 0.0, w / 2.0], [0.0, 110.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    mask[14:58, 18:80] = True
    return depth, conf, mask, np.ones((h, w), dtype=bool), k


def test_depth_normals_support_aligned_plane() -> None:
    depth, conf, plane_mask, map_mask, k = _synthetic_plane()
    normals = compute_depth_normals_camera(depth, map_mask, k)

    result = fuse_mapanything_with_planes(
        frame_id="synthetic",
        map_depth=depth,
        map_confidence=conf,
        map_mask=map_mask,
        map_normals_camera=normals,
        intrinsics=k,
        plane_candidates=[
            PlaneCandidate(frame_id="synthetic", plane_id="wall", mask=plane_mask, normal=np.asarray([0, 0, 1]))
        ],
        min_plane_support=64,
        surfel_pixel_step=4,
    )

    assert len(result.planes) == 1
    support = result.planes[0].normal_support
    assert support["status"] == "agree"
    assert support["normal_angular_error_deg_median"] < 1.0
    assert support["mapanything_normal_camera"][2] > 0.9
    assert result.metrics["normal_fusion_status"] == "provided"
    assert result.metrics["normal_supported_plane_count"] == 1
    assert result.metrics["normal_rejected_plane_count"] == 0


def test_contradictory_normals_reject_plane_candidate() -> None:
    depth, conf, plane_mask, map_mask, k = _synthetic_plane()
    normals = np.zeros((*depth.shape, 3), dtype=np.float32)
    normals[plane_mask] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    result = fuse_mapanything_with_planes(
        frame_id="synthetic",
        map_depth=depth,
        map_confidence=conf,
        map_mask=map_mask,
        map_normals_camera=normals,
        map_normals_valid=plane_mask,
        intrinsics=k,
        plane_candidates=[
            PlaneCandidate(frame_id="synthetic", plane_id="bad_wall", mask=plane_mask, normal=np.asarray([0, 0, 1]))
        ],
        min_plane_support=64,
        surfel_pixel_step=4,
    )

    assert result.planes == []
    assert result.metrics["normal_disagreement_plane_count"] == 1
    assert result.metrics["normal_rejected_plane_count"] == 1


def test_missing_normals_preserve_depth_plane_with_explicit_status() -> None:
    depth, conf, plane_mask, map_mask, k = _synthetic_plane()

    result = fuse_mapanything_with_planes(
        frame_id="synthetic",
        map_depth=depth,
        map_confidence=conf,
        map_mask=map_mask,
        intrinsics=k,
        plane_candidates=[
            PlaneCandidate(frame_id="synthetic", plane_id="legacy_wall", mask=plane_mask, normal=np.asarray([0, 0, 1]))
        ],
        min_plane_support=64,
        surfel_pixel_step=4,
    )

    assert len(result.planes) == 1
    assert result.planes[0].normal_support["status"] == "not_provided"
    assert result.metrics["normal_fusion_status"] == "not_provided"
