"""Focused invariants for texture-only geometry reuse and real-photo sampling."""
import numpy as np
import pytest

from tools.room_mesh_export.photo_texture import bake_photos, bilinear_rgb, witness_candidates
from tools.room_mesh_export.refine_texture import geometry_digest
from tools.room_mesh_export.build_browser_mesh import ray_scene, record
from tools.mapanything_phone_scan.texture_single_room_review_surface import PhotoView
from PIL import Image


def test_geometry_digest_ignores_uv_vertex_duplication_but_preserves_winding():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.uint32)
    original = geometry_digest(vertices, faces)
    assert geometry_digest(vertices[[2, 0, 1]], np.array([[1, 2, 0]])) == original
    assert geometry_digest(np.vstack((vertices, vertices)), np.array([[3, 4, 5]])) == original
    assert geometry_digest(vertices, np.array([[0, 2, 1]])) != original
    vertices[1, 0] += .001
    assert geometry_digest(vertices, faces) != original


def test_witness_votes_merge_same_photo_across_corners():
    ids = np.array([[[4, 1, -1, -1], [4, 2, -1, -1], [4, 3, -1, -1]]], np.int16)
    weights = np.array([[[.5, .8, 0, 0], [.5, .8, 0, 0], [.5, .8, 0, 0]]], np.float32)
    selected, score = witness_candidates(ids, weights, np.full((1, 3), 1/3))
    assert selected[0, 0] == 4
    np.testing.assert_allclose(score[0], [.5, .8/3, .8/3, .8/3], atol=1e-7)
    assert len(set(selected[0])) == 4


def test_witness_no_source_remains_explicitly_missing():
    ids = np.full((2, 3, 4), -1, np.int16)
    selected, score = witness_candidates(ids, np.zeros_like(ids, dtype=np.float32), np.full((2, 3), 1/3))
    assert (selected == -1).all() and (score == 0).all()


def test_photo_bilinear_sampling_is_linear_light_not_gamma_average():
    rgb = np.array([[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [255, 255, 255]]], np.uint8)
    sampled = bilinear_rgb(rgb, np.array([[.5, .5]]))
    np.testing.assert_allclose(sampled, [[.5, .5, .5]], atol=1e-7)


def test_photo_sampling_refuses_outside_or_nonfinite_coordinates():
    rgb = np.zeros((4, 4, 3), np.uint8)
    for uv in (np.array([[-.1, 1]]), np.array([[3.1, 1]]), np.array([[np.nan, 1]])):
        with pytest.raises(ValueError):
            bilinear_rgb(rgb, uv)


@pytest.mark.parametrize("mode", ["photo-best", "photo-blend"])
@pytest.mark.parametrize("visible", [True, False])
def test_photo_atlas_preserves_provider_gate_and_explicit_colour_fallback(tmp_path, monkeypatch, mode, visible):
    vertices = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    photo = tmp_path / "real_photo.png"
    Image.fromarray(np.full((8, 8, 3), [255, 0, 0], np.uint8)).save(photo)
    K = np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]], float)
    view = PhotoView(0, np.ones((8, 8), np.float32), np.full((8, 8), visible),
                     np.ones((8, 8), np.float32), K, np.eye(4), K,
                     np.array([[1, 0, 0], [0, 1, 0]], float), photo, (8, 8), 1., {})
    monkeypatch.setattr("tools.room_mesh_export.photo_texture.verified_views", lambda manifest: ([view], {}, {}))
    witnesses = tmp_path / "witnesses.npz"
    ids = np.full((4, 4), -1, np.int16)
    weights = np.zeros((4, 4), np.float32)
    ids[:, 0], weights[:, 0] = 0, 1
    np.savez(witnesses, source_view_index=ids, source_weight=weights)
    manifest = {"color_bake": {"color_sources": record(witnesses)}, "source": {"photometric_manifest": {}}}
    source_colours = np.tile([0, 0, 1], (4, 1)).astype(np.float32)
    original_vertices, original_faces = vertices.copy(), faces.copy()
    image, report = bake_photos(ray_scene(vertices, faces, 1), faces, source_colours, manifest,
                               vertices, faces, uv, 16, 1, 2, mode=mode)
    np.testing.assert_array_equal(np.asarray(image)[8, 8], [255, 0, 0] if visible else [0, 0, 255])
    assert report["direct_photo_fraction_of_covered"] == int(visible)
    assert report["image_inpainting"] is False
    np.testing.assert_array_equal(vertices, original_vertices)
    np.testing.assert_array_equal(faces, original_faces)
