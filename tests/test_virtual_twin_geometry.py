from __future__ import annotations

import numpy as np

from noesis.virtual_twin.geometry import PlaneCandidate, decode_mask_rle, fuse_mapanything_with_planes


def test_plane_depth_fusion_replaces_noisy_planar_pixels() -> None:
    h, w = 72, 96
    k = np.asarray([[110.0, 0.0, w / 2.0], [0.0, 110.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    mask[14:58, 18:80] = True
    yy, xx = np.indices((h, w))
    depth[mask] += (0.08 * np.sin(xx[mask] * 0.37) + 0.04 * np.cos(yy[mask] * 0.21)).astype(np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    result = fuse_mapanything_with_planes(
        frame_id="synthetic",
        map_depth=depth,
        map_confidence=conf,
        map_mask=np.ones((h, w), dtype=bool),
        intrinsics=k,
        plane_candidates=[PlaneCandidate(frame_id="synthetic", plane_id="wall", mask=mask, normal=np.asarray([0, 0, 1]))],
        min_confidence=0.5,
        min_plane_support=64,
        surfel_pixel_step=4,
    )

    assert len(result.planes) == 1
    assert result.metrics["fused_improved_raw_residual"] is True
    assert result.planes[0].median_residual_m <= result.planes[0].raw_median_residual_m
    assert decode_mask_rle(result.planes[0].mask_rle).shape == (h, w)
    assert result.fused_points_camera.shape[1] == 3
