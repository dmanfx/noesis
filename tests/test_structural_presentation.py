"""Focused presentation admission, separate from physical registration proof."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from noesis.calibration.structural_presentation import (
    PresentationRegistrationError, _annotation_stability, _matrix_for_floor, _projection_evidence,
    _relative_file, _unchanged_room, _verify_capture, current_structural_presentations,
    _owner_selection, _selection_quality_only,
    _verify_geometry_features,
)
from noesis.calibration.presentation_registration import _canonical_sha256
from noesis.calibration.structural_registration import WallSegment, solve_structural_registration
from noesis_core.contracts.scene_prior import ScenePriorFrameBinding
from noesis.validation.authored_scene import AuthoredSceneGeometry


def _candidate():
    source = [
        WallSegment("west", (0,0),(0,4),300,2),
        WallSegment("east", (6,0),(6,4),300,2),
        WallSegment("back", (0,4),(6,4),300,2),
        WallSegment("front", (0,0),(6,0),300,2),
    ]
    authored = [WallSegment(s.segment_id, tuple(100*v for v in s.start_xz),
                             tuple(100*v for v in s.end_xz), 2, 200) for s in source]
    return solve_structural_registration(source, authored,
        correspondences=[{"source_id":s.segment_id,"authored_id":s.segment_id,
                          "role":"heldout" if s.segment_id == "front" else "fit"} for s in source],
        scene_per_m_hint=90)


def test_measured_floor_translation_preserves_uniform_scale():
    candidate = _candidate()
    assert candidate["solver_status"] == "accepted"
    matrix = _matrix_for_floor(candidate, {"normal":[0,1,0],"offset_m":-1.2}, 30)
    assert (matrix @ [2,1.2,3,1]) == pytest.approx([200,30,300,1])
    assert np.linalg.svd(matrix[:3,:3],compute_uv=False) == pytest.approx([100,100,100])
    assert (matrix @ [2,1.3905,3,1])[1] == pytest.approx(49.05)


def test_explicit_owner_selection_preserves_rejected_quality_and_exact_matrix():
    original = _candidate()
    inputs = deepcopy(original["inputs"])
    inputs["authored_segments"][-1]["start_xz"][1] = -40
    inputs["authored_segments"][-1]["end_xz"][1] = -40
    candidate = solve_structural_registration(**inputs)
    assert candidate["solver_status"] == "rejected"
    assert candidate["reason_codes"] == ["heldout_wall_residual"]
    receipt = _owner_selection({"selection_id":"explicit-intermediate", "authorized_by":"owner",
        "authorization":"Select this known failed-quality intermediate for scene presentation only",
        "reason":"Owner reviewed the visual improvement; physical qualification remains false",
        "structurally_qualified":False, "selected_matrix_sha256s":{
            "room":_canonical_sha256({"matrix":candidate["solution"]["matrix_col_major"]})}}, {"room"})
    with pytest.raises(PresentationRegistrationError, match="geometry gates"):
        _matrix_for_floor(candidate,{"normal":[0,1,0],"offset_m":0},0)
    selected = _matrix_for_floor(candidate,{"normal":[0,1,0],"offset_m":0},0,
                                 owner_selection=receipt,camera="room")
    assert selected.flatten(order="F") == pytest.approx(candidate["solution"]["matrix_col_major"])
    assert candidate["solver_status"] == "rejected"  # selection never rewrites evidence
    changed = deepcopy(receipt)
    changed["selected_matrix_sha256s"]["room"] = "a"*64
    with pytest.raises(PresentationRegistrationError, match="explicitly owner-selected"):
        _matrix_for_floor(candidate,{"normal":[0,1,0],"offset_m":0},0,
                          owner_selection=changed,camera="room")


def test_owner_selection_cannot_claim_pass_or_bypass_validity():
    with pytest.raises(PresentationRegistrationError, match="authorization"):
        _owner_selection(None, {"room"})
    with pytest.raises(PresentationRegistrationError, match="qualification"):
        _owner_selection({"selection_id":"x", "authorized_by":"owner", "authorization":"x",
            "reason":"x", "selected_matrix_sha256s":{"room":"a"*64},
            "structurally_qualified":True}, {"room"})
    with pytest.raises(PresentationRegistrationError, match="validity"):
        _selection_quality_only({"passed":False, "failed_case":{"reason_codes":["structural_geometry_rank_deficient"]}})


def test_retained_room_checks_actual_geometry_not_remote_plane_numbering(tmp_path):
    def geometry(name, *, remote_height=0.0, wall_x=0.0):
        rows = ["g room_local", "v 0 0 0", "v 0 0 4", "v 6 0 4", "v 6 0 0", "f 1 2 3 4",
                "g wall_local", f"v {wall_x} 0 0", f"v {wall_x} 3 0", f"v {wall_x} 3 4",
                f"v {wall_x} 0 4", "f 5 6 7 8", "g room_remote",
                f"v 20 {remote_height} 0", f"v 20 {remote_height} 4",
                f"v 24 {remote_height} 4", f"v 24 {remote_height} 0", "f 9 10 11 12"]
        path = tmp_path / name
        path.write_text("\n".join(rows) + "\n")
        return AuthoredSceneGeometry.from_obj(path)
    original = geometry("original.obj")
    _unchanged_room(original, geometry("remote.obj", remote_height=0.1905), ["room_local"], 1.0)
    with pytest.raises(PresentationRegistrationError, match="retained room geometry changed"):
        _unchanged_room(original, geometry("local.obj", wall_x=0.05), ["room_local"], 1.0)


def test_reviewed_wall_endpoint_is_verified_without_inventing_an_opening(tmp_path, monkeypatch):
    from dataclasses import asdict
    from noesis.calibration.authored_structures import extract_authored_structures
    path = tmp_path / "room.obj"
    path.write_text("g room_floor\nv 0 0 0\nv 0 0 4\nv 6 0 4\nv 6 0 0\nf 1 2 3 4\n"
                    "g wall\nv 0 0 0\nv 0 3 0\nv 0 3 4\nv 0 0 4\nf 5 6 7 8\n")
    geometry = AuthoredSceneGeometry.from_obj(path)
    structures = extract_authored_structures(geometry, ["room_floor"], initial_scene_units_per_meter=1)
    wall = structures.wall_segments[0]
    source = {"point_id":"panel", "kind":"opening", "xz":[0,0], "source_wall_ids":["measured-wall"], "evidence_refs":[]}
    point = {"point_id":"model-panel", "kind":"opening", "xz":list(wall.start_xz)}
    candidate = {"inputs":{"source_segments":[], "authored_segments":[asdict(wall)],
        "scene_per_m_hint":1, "correspondences":[{"source_id":"measured-wall", "authored_id":wall.segment_id}],
        "source_points":[source], "authored_points":[point],
        "point_correspondences":[{"source_id":"panel", "authored_id":"model-panel"}]}}
    selector = {"point_id":"model-panel", "kind":"opening", "selector_type":"wall_segment_endpoint",
        "segment_id":wall.segment_id, "endpoint":"start", "identity_review":"Reviewed exact wall-panel endpoint"}
    monkeypatch.setattr("noesis.calibration.structural_registration.verify_pcf_source_segments", lambda *_:None)
    monkeypatch.setattr("noesis.calibration.structural_presentation._verify_image_points", lambda **_:None)
    kwargs = dict(candidate=candidate, points_path=path, geometry=geometry, groups=["room_floor"],
                  bundle={}, camera="room", image_evidence=[], authored_selectors=[selector], root=tmp_path, support=None)
    _verify_geometry_features(**kwargs)
    point["xz"][1] += .1
    with pytest.raises(PresentationRegistrationError, match="differs from measured OBJ"):
        _verify_geometry_features(**kwargs)
    point["xz"] = list(wall.start_xz)
    source["source_wall_ids"] = ["unmatched-wall"]
    with pytest.raises(PresentationRegistrationError, match="corresponding measured wall"):
        _verify_geometry_features(**kwargs)


def test_joint_raster_bounds_keep_fit_and_holdout_gates(monkeypatch):
    candidate = _candidate()
    point = {"point_id":"contact", "kind":"floor_contact", "xz":[2.0,2.0],
             "source_wall_ids":[], "evidence_refs":["image_sha256:" + "a" * 64, "pixel_uv:50.000000,50.000000"]}
    inputs = {**candidate["inputs"], "source_points":[point],
              "authored_points":[{"point_id":"author", "kind":"floor_contact", "xz":[200,200]}],
              "point_correspondences":[{"source_id":"contact", "authored_id":"author", "role":"heldout"}]}
    candidate = solve_structural_registration(**inputs)
    def projection(**kwargs):
        return {"point":{**point, "xz":[2.0 + 0.1 * (v - 50) for v in kwargs["pixel"]]}}
    monkeypatch.setattr("noesis.calibration.structural_image_evidence.project_structural_image_point", projection)
    evidence = {"point":candidate["inputs"]["source_points"][0], "image_file":{"path":"unused.png"},
        "pixel_bounds":[[49.5,49.5],[50.5,50.5]],
        "image_evidence":{"pixel_uv":[50,50], "image":{"width":100,"height":100},
                          "wall_plane":None,"surface_y_m":0.0}}
    checked = _annotation_stability(candidate=candidate,bundle={},camera="camera",image_evidence=[evidence],root=ROOT)
    assert checked["passed"] and checked["combinations_checked"] == 5
    evidence["pixel_bounds"] = [[47,47],[53,53]]
    rejected = _annotation_stability(candidate=candidate,bundle={},camera="camera",image_evidence=[evidence],root=ROOT)
    assert not rejected["passed"]
    assert "heldout_point_residual" in rejected["failed_case"]["reason_codes"]


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data/mapanything_phone_scans/20260922-120808-2713a4ce/paired_static_reference/paired_static_reference_5474e6ce2a6ae6a7"
BUNDLE = ROOT.parent / "Menon/artifacts/structural-registration-20261003/before/calibration-bundle.json"


@pytest.mark.skipif(not (CAPTURE / "manifest.json").is_file() or not BUNDLE.is_file(),
                    reason="retained calibrated raster unavailable")
def test_real_retained_image_binding_and_current_projection_gates():
    bundle = json.loads(BUNDLE.read_text())
    camera = "living-room"
    evidence = {"image_file":_relative_file(CAPTURE / "keyframe.png",ROOT),
        "capture":{key:_relative_file(CAPTURE / filename,ROOT) for key,filename in
                   (("manifest","manifest.json"),("calibration","calibration.json"),("metadata","room_points_meta.json"))},
        "image_evidence":{"image":{"width":1920,"height":1080}}}
    _verify_capture(evidence,bundle=bundle,camera=camera,root=ROOT)
    changed = deepcopy(bundle)
    changed["cameras"]["K"][camera][0] += 1
    with pytest.raises(PresentationRegistrationError,match="rays no longer match"):
        _verify_capture(evidence,bundle=changed,camera=camera,root=ROOT)

    catalog = json.loads((ROOT / "data/scene_priors/catalog.json").read_text())
    binding = ScenePriorFrameBinding.model_validate(next(row["frame_binding"] for row in catalog["camera_bindings"] if row["camera_id"] == camera))
    contract = binding.frame_transform()
    projection = _projection_evidence(bundle,camera)
    # Contract check only: this fixture is not a structural solve or deployed artifact.
    current = bundle["world_frame_presentations"][contract.target_frame.revision]
    entry = {"camera_id":camera,"source_transform_sha256":contract.transform_sha256,
        "calibration_frame_revision":contract.source_frame.revision,
        "calibration_revision":projection["frame_binding"]["camera_calibration_sha256"],
        "world_floor_plane":projection["frame_binding"]["world_floor_plane"],
        "image_projection":projection, **{key:current[key] for key in
          ("world_to_scene_col_major","world_to_scene_sha256","scene_revision_id")}}
    artifact = {"support_sets":{},"scene_prior_bindings":{camera:binding.model_dump(mode="json")},
        "presentations":{contract.target_frame.revision:{"kind":"structural","entry":entry}},
        "registration_sha256":"a"*64,"authored_scene_sha256":current["authored_scene_sha256"]}
    kwargs = {"frame_contracts":{camera:contract},"support_surfaces":{},
        "calibration_revision":entry["calibration_revision"],
        "scene_prior_bindings":{camera:binding},
        "camera_projection":{camera:{"K":projection["K"],"E":projection["E"]}}}
    output = current_structural_presentations(artifact,**kwargs)
    assert output[contract.target_frame.revision]["world_to_scene_col_major"] == current["world_to_scene_col_major"]
    altered = deepcopy(kwargs)
    altered["camera_projection"][camera]["K"][0] += 1
    with pytest.raises(PresentationRegistrationError,match="camera rays changed"):
        current_structural_presentations(artifact,**altered)
