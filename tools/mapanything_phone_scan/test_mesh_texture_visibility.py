from __future__ import annotations

import json

import numpy as np
import pytest

from tools.mapanything_phone_scan import build_mesh_texture_visibility as visibility
from tools.mapanything_phone_scan import texture_single_room_review_surface as texture
from tools.mapanything_phone_scan.test_texture_single_room_review_surface import _square, _view


def test_explicit_surface_visibility_does_not_claim_raw_depth_agreement():
    vertices, faces = _square()
    view = _view(depth=1.)
    view.mask[:] = False
    view.confidence[:] = 0
    candidates, _ = texture.face_candidates(vertices, faces, [view])
    assert np.all(candidates == -1)
    view.surface_depth = np.full((64, 64), 2., dtype=np.float32)
    candidates, _ = texture.face_candidates(vertices, faces, [view])
    assert np.all(candidates[:, 0] == 0)
    np.testing.assert_array_equal(view.depth, 1.)
    assert not view.mask.any()


def test_surface_visibility_rejects_occlusion_unknown_and_off_photo_rays():
    vertices, _ = _square()
    view = _view()
    view.surface_depth = np.ones((64, 64), dtype=np.float32)
    assert not texture.depth_visible(vertices, view)[0].any()
    view.surface_depth[:] = np.inf
    assert not texture.depth_visible(vertices, view)[0].any()
    view.surface_depth[:] = 2.
    view.model_to_photo[:, 2] = 100.
    assert not texture.depth_visible(vertices, view)[0].any()


def _bound_directory(tmp_path, view):
    view.provenance.update({key: None for key in ("frame_id", "source_frame_index", "consensus_raw",
        "conditioned_provider_raw", "common_intrinsics", "provider_intrinsics", "camera_to_world", "photo_size", "model_to_photo_row_major")})
    path = tmp_path / "depth.npz"
    np.savez_compressed(path, depth=np.full((64, 64), 2., dtype=np.float32))
    row = {"index": 0, "camera_signature": visibility.camera_signature(view), "status": "rendered",
           "height": 64, "width": 64, "depth": {"path": path.name, "sha256": texture._sha(path)}}
    manifest = {"schema": visibility.SCHEMA, "mesh_sha256": "mesh", "status": "complete_review_surface_zbuffers", "views": [row]}
    (tmp_path / "mesh_texture_visibility_manifest.json").write_text(json.dumps(manifest))


def test_visibility_cache_hashes_mesh_camera_mapping_and_actual_depth(tmp_path):
    view = _view()
    _bound_directory(tmp_path, view)
    report = visibility.read_bound_visibility(tmp_path, [view], "mesh")
    assert report["geometry_support_classes_changed"] is False
    assert view.surface_depth.shape == (64, 64)
    with pytest.raises(texture.TextureSurfaceError, match="mesh/view binding"):
        visibility.read_bound_visibility(tmp_path, [view], "wrong-mesh")
    view.provenance["frame_id"] = "changed"
    with pytest.raises(texture.TextureSurfaceError, match="camera/photo mapping"):
        visibility.read_bound_visibility(tmp_path, [view], "mesh")
    view.provenance["frame_id"] = None
    np.savez_compressed(tmp_path / "depth.npz", depth=np.ones((64, 64)))
    with pytest.raises(texture.TextureSurfaceError, match="depth hash"):
        visibility.read_bound_visibility(tmp_path, [view], "mesh")
