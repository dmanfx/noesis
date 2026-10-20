"""Ceiling selectivity and frozen-geometry extension regression checks."""
import copy
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import trimesh

from .pcf_ceiling_filter import CeilingPatch, ceiling_triangle_mask, classify_points, fit_plane, discover_ceiling_patches
from .extend_pcf_upper_surfaces import (BoundView, append_mesh, photo_coordinates,
    patch_fingerprint, reviewed_patches, build, source_region_protection,check_view_witnesses)
from .build_high_detail_pcf_hybrid import pack_glb, sha_bytes, unpack_glb
from .build_pcf_review_surface_mesh import SurfaceViewEvidence


def patch():
    x,z = np.meshgrid(np.arange(-1,1.01,.05),np.arange(-1,1.01,.05))
    points = np.column_stack((x.ravel(),np.full(x.size,2.4),z.ravel()))
    return CeilingPatch(np.array([0.,1.,0.]),-2.4,points,{})


def test_ceiling_not_upper_wall_furniture_or_junction():
    points = np.array([[0,2.4,0],[.8,2.4,0],[.82,2.38,0],[0,3.1,0],
                       [0,1.7,0],[1.5,2.4,0]])
    normals = np.array([[0,1,0],[0,1,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0]],float)
    old = points.copy()
    removed,ambiguous = classify_points(points,normals,[patch()])
    assert removed.tolist() == [True,False,False,False,False,False]
    assert ambiguous[1] and ambiguous[2]
    assert np.array_equal(points,old)


def test_no_plane_never_falls_back_to_height_cut():
    points = np.array([[0,8.,0],[0,2.5,0]])
    removed,ambiguous = classify_points(points,np.array([[0,1,0],[1,0,0.]]),[])
    assert not removed.any() and not ambiguous.any()


def test_sloped_ceiling_and_vertical_face_selectivity():
    candidate = patch()
    candidate.seeds[:,1] += .08*candidate.seeds[:,0]
    candidate.normal,candidate.offset = fit_plane(candidate.seeds)
    vertices = np.array([[0,2.4,0],[.05,2.404,0],[0,2.4,.05],
                         [.05,2.2,0],[.05,2.4,0],[.05,2.4,.05],
                         [3,2.64,0],[3.05,2.644,0],[3,2.64,.05]])
    faces = np.arange(9).reshape(-1,3)
    assert ceiling_triangle_mask(vertices,faces,[candidate]).tolist() == [True,False,False]


def test_disconnected_unknown_patch_gap_not_closed():
    candidate = patch()
    candidate.seeds = candidate.seeds[np.abs(candidate.seeds[:,0])>.5]
    vertices = np.array([[-.55,2.4,0],[.55,2.4,0],[.55,2.4,.05]])
    assert not ceiling_triangle_mask(vertices,np.array([[0,1,2]]),[candidate]).any()


def test_ceiling_triangle_at_vertical_junction_abstains():
    vertices = np.array([[0,2.4,0],[.05,2.4,0],[0,2.4,.05]])
    faces = np.array([[0,1,2]])
    assert ceiling_triangle_mask(vertices,faces,[patch()]).all()
    assert not ceiling_triangle_mask(vertices,faces,[patch()],junction_points=np.array([[0,2.38,0]])).any()


def test_append_keeps_original_bin_and_owner_index(tmp_path):
    old = trimesh.Scene()
    mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],
        vertex_colors=[[50,100,150,255]]*3,process=False)
    old.add_geometry(mesh,geom_name="pcf_surface_family",node_name="pcf_surface_family")
    path = tmp_path/"old.glb"; path.write_bytes(old.export(file_type="glb"))
    tree,binary = unpack_glb(path)
    original = copy.deepcopy(tree)
    owner_index = next(i for i,node in enumerate(tree["nodes"]) if node.get("name")=="pcf_surface_family")
    output = bytearray(binary)
    metrics = append_mesh(tree,output,1,mesh,suffix="_upper_rgb")
    assert metrics == {"vertex_count":3,"triangle_count":1}
    assert bytes(output[:len(binary)]) == binary
    assert tree["accessors"][:len(original["accessors"])] == original["accessors"]
    assert tree["bufferViews"][:len(original["bufferViews"])] == original["bufferViews"]
    assert tree["meshes"][:len(original["meshes"])] == original["meshes"]
    assert tree["nodes"][owner_index]["name"] == "pcf_surface_family"
    assert "mesh" not in tree["nodes"][owner_index]
    new = tmp_path/"new.glb"; new.write_bytes(pack_glb(tree,bytes(output)))
    _,readback = unpack_glb(new)
    assert readback[:len(binary)] == binary
    decoded = trimesh.load(new,force="scene",process=False)
    assert sum(len(part.faces) for part in decoded.geometry.values()) == 2


def test_photo_tile_offset_bounds_and_rejected_view():
    evidence = SurfaceViewEvidence(view_index=9,intrinsics=np.eye(3),depth_z=np.ones((4,4)),
        mask=np.ones((4,4),bool),camera_to_world=np.eye(4),quality=np.ones((4,4)))
    view = BoundView(evidence,np.eye(4),np.eye(3),np.array([[1.,0,2],[0,1,2]]),(10,10),{})
    points = np.array([[0,0,1],[20,0,1],[0,0,-1.]])
    uv,inside = photo_coordinates(points,view,10,10)
    assert inside.tolist() == [True,False,False]
    assert uv[0] == pytest.approx([12/80,12/60])  # raw glTF upper-left V, no export flip
    view.affine = None
    assert not photo_coordinates(points,view,10,10)[1].any()


def test_glb_size_bound_is_not_relaxed(monkeypatch):
    from . import build_high_detail_pcf_hybrid as glb
    monkeypatch.setattr(glb,"GLB_MAX_BYTES",32)
    with pytest.raises(ValueError,match="64 MiB"):
        glb.pack_glb({"buffers":[{"byteLength":0}]},b"geometry")


def test_above_camera_furniture_top_needs_supporting_camera_hemisphere():
    x,z = np.meshgrid(np.arange(-1,1.01,.04),np.arange(-1,1.01,.04))
    ceiling = np.column_stack((x.ravel(),np.full(x.size,2.4),z.ravel()))
    furniture = ceiling.copy(); furniture[:,1] = 1.9
    cameras = np.array([[0,1.5,0],[.2,1.5,.2]])
    def votes(points,normals):
        return np.ones(len(points),bool),{
            "support_count":np.full(len(points),2),
            "normal_direction_vote":np.where(points[:,1]>2.1,-1.,1.)}
    _,_,patches,_ = discover_ceiling_patches(np.vstack((ceiling,furniture)),cameras,votes)
    assert patches and all(np.min(p.seeds[:,1])>2.1 for p in patches)


def test_semantic_admission_uses_seed_geometry_not_discovery_order():
    ceiling = patch()
    beam = patch(); beam.seeds = beam.seeds.copy(); beam.seeds[:,1] -= .35
    review = {"accepted_patch_seed_sha256":{"1":[patch_fingerprint(ceiling)]}}
    assert reviewed_patches([ceiling,beam],1,review)[0] is ceiling
    assert reviewed_patches([beam,ceiling],1,review)[0] is ceiling
    changed = patch(); changed.seeds[0,0] += .001
    with pytest.raises(ValueError,match="unknown/duplicate"):
        reviewed_patches([changed,beam],1,review)


@pytest.mark.parametrize("classification,review",[(None,Path("review.json")),
    (Path("classification"),None),(None,None)])
def test_build_requires_both_frozen_classification_and_review_before_io(classification,review):
    # No other arguments or files exist: failure must precede source/output IO.
    with pytest.raises(ValueError,match="frozen classification"):
        build(Namespace(classify_only=False,classification_dir=classification,ceiling_review=review))


def test_source_semantic_protection_requires_matching_measured_surface():
    depth = np.ones((8,8))
    mask = np.ones((8,8),bool); mask[2,3] = False  # includes source uncertainty
    quality = np.ones((8,8)); quality[2,4] = .2
    depth[2,5] = np.nan
    evidence = SurfaceViewEvidence(view_index=0,intrinsics=np.eye(3),depth_z=depth,
        mask=mask,camera_to_world=np.eye(4),quality=quality)
    view = BoundView(evidence,np.eye(4),np.eye(3),None,(0,0),{})
    points = np.array([[2,2,1],[4,4,2],[3,2,1],[4,2,1],[5,2,1],[7,2,1],[0,0,-1]],float)
    original = points.copy()
    regions = [{"view_index":0,"polygon_pixels":[[1,1],[6,1],[6,6],[1,6]],
                "reason":"source-reviewed beam; protect measured depth only"}]
    assert source_region_protection(points,[view],regions).tolist() == [True,False,False,False,False,False,False]
    assert np.array_equal(points,original)


def test_frozen_photo_witness_rejects_conditioned_model_drift():
    from types import SimpleNamespace
    original = {"raw_sha256":"a"*64,"conditioned_model_sha256":"b"*64,
                "camera_to_assembly_row_major":np.eye(4).tolist()}
    views = [SimpleNamespace(witness=copy.deepcopy(original))]
    check_view_witnesses([original],views)
    views[0].witness["conditioned_model_sha256"] = "0"*64
    with pytest.raises(ValueError,match="source-view/model/atlas"):
        check_view_witnesses([original],views)
