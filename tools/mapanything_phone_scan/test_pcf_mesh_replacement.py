"""Narrow v4 replacement chain; no temporary switch to an old point selector."""
import copy
import json
from pathlib import Path
import pytest
from .publish_pcf_review_surface_mesh import _sha256, replacement_descriptor


def fixture(tmp_path):
    storage = tmp_path/"storage"; storage.mkdir()
    old = storage/"old.glb"; old.write_bytes(b"original-glb")
    old_manifest = tmp_path/"old.json"
    old_manifest.write_text(json.dumps({"source":{"surfels_npz_sha256":"a"*64}}))
    current = {"contract":"noesis.scene.review_assembly","contract_version":4,"status":"review_only",
        "accepted_for_canonical_use":False,"assembly_id":"old","assembly_gauge":"family_accepted_backend_world_m",
        "artifact":{"role":"multiroom_surface_mesh_glb","relative_path":"old.glb","sha256":_sha256(old),
                    "size_bytes":old.stat().st_size,"owner_triangle_counts":{"1":1},"support":{"old":True}},
        "camera_anchor":{"translation":[0,1.5,0]},
        "provenance":{"source_surface_mesh_manifest_sha256":_sha256(old_manifest),
                      "source_reintegration_manifest_sha256":"b"*64,"source_point_glb_sha256":"c"*64}}
    selector = tmp_path/"current.json"; selector.write_text(json.dumps(current))
    glb = tmp_path/"new.glb"; glb.write_bytes(b"extended-glb")
    new = {"contract":"noesis.pcf.review_surface_mesh","status":"review_only","accepted_for_canonical_use":False,
           "coordinate_frame":current["assembly_gauge"],
           "source":{"surfels_manifest_sha256":"b"*64,"surfels_npz_sha256":"a"*64,
                     "retained_low_glb_sha256":_sha256(old),"retained_low_mesh_manifest_sha256":_sha256(old_manifest),
                     "source_descriptor_sha256":_sha256(selector)},
           "output":{"sha256":_sha256(glb),"size_bytes":glb.stat().st_size,"vertex_count":6,"triangle_count":2},
           "owners":{"1":{"output_triangle_count":2}}}
    manifest = tmp_path/"new.json"; manifest.write_text(json.dumps(new))
    kwargs = dict(current=current,current_path=selector,storage_root=storage,source_mesh_manifest=old_manifest,
                  mesh_glb=glb,mesh_manifest=manifest,assembly_id="new")
    return kwargs,new


def test_replacement_preserves_source_and_mapping_without_activation(tmp_path):
    kwargs,_ = fixture(tmp_path)
    original = copy.deepcopy(kwargs["current"])
    digest = _sha256(kwargs["current_path"])
    result = replacement_descriptor(**kwargs)
    assert result["assembly_id"] == "new" and result["contract_version"] == 4
    assert result["camera_anchor"] == original["camera_anchor"]
    assert result["provenance"]["source_point_glb_sha256"] == "c"*64
    assert result["provenance"]["source_reintegration_manifest_sha256"] == "b"*64
    assert "support" not in result["artifact"]  # no invented legacy ray admission
    assert kwargs["current"] == original and _sha256(kwargs["current_path"]) == digest


@pytest.mark.parametrize("field",["surfels_npz_sha256","surfels_manifest_sha256",
    "retained_low_glb_sha256","retained_low_mesh_manifest_sha256","source_descriptor_sha256"])
def test_replacement_rejects_changed_source_chain(tmp_path,field):
    kwargs,new = fixture(tmp_path)
    new["source"][field] = "0"*64
    kwargs["mesh_manifest"].write_text(json.dumps(new))
    with pytest.raises(ValueError,match="exact current source chain"):
        replacement_descriptor(**kwargs)


def test_replacement_rejects_source_artifact_mutation_and_owner_drift(tmp_path):
    kwargs,new = fixture(tmp_path)
    new["owners"] = {"2":{"output_triangle_count":2}}
    kwargs["mesh_manifest"].write_text(json.dumps(new))
    with pytest.raises(ValueError,match="inventory"):
        replacement_descriptor(**kwargs)
    (kwargs["storage_root"]/"old.glb").write_bytes(b"changed")
    with pytest.raises(ValueError,match="digest mismatch"):
        replacement_descriptor(**kwargs)
