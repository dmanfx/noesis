#!/usr/bin/env python3
"""Read-only direct serialized geometry, source-ray and photo-UV stage checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .build_high_detail_pcf_hybrid import read_json,sha_file,unpack_glb
from .extend_pcf_upper_surfaces import (checked_reference,load_bound_views,
    photo_coordinates,qualify,reference,check_view_witnesses)
from .render_pcf_upper_comparison import accessor_array
from .tracking_review import build_high_detail_tracking_review,build_tracking_review


def primitive(tree,binary,name):
    node = next(node for node in tree["nodes"] if node.get("name")==name)
    rows = tree["meshes"][node["mesh"]]["primitives"]
    if len(rows)!=1:
        raise ValueError("extension primitive is ambiguous")
    row = rows[0]
    return row,accessor_array(tree,binary,row["attributes"]["POSITION"]),accessor_array(tree,binary,row["indices"]).reshape(-1,3)


def verify(stage_dir):
    root = Path(__file__).resolve().parents[2]
    source,stage = read_json(stage_dir/"source_bindings.json"),read_json(stage_dir/"stage_validation.json")
    original_recipe_path = checked_reference(source["binding"],root)
    original_recipe = read_json(original_recipe_path)
    original_descriptor_path = checked_reference(original_recipe["source_descriptor"],root)
    original_low = build_tracking_review(original_recipe_path,
        catalog_path=checked_reference(original_recipe["catalog"],root),
        current_descriptor=read_json(original_descriptor_path),
        current_descriptor_path=original_descriptor_path,repo_root=root)
    multiroom = read_json(checked_reference(original_recipe["reintegration"],root))
    join = read_json(checked_reference(original_recipe["family_kitchen"],root))
    registration = read_json(checked_reference(original_recipe["endpoint_registration"],root))
    photo = read_json(checked_reference(source["photo_source_manifest"],root))
    recipe_path = stage_dir/"binding.validation.json"
    recipe = read_json(recipe_path); descriptor_path = checked_reference(recipe["source_descriptor"],root)
    producer = read_json(checked_reference(recipe["mesh_manifest"],root))
    classification = read_json(checked_reference(producer["ceiling_filter"]["classification"],root))
    low = build_tracking_review(recipe_path,catalog_path=checked_reference(recipe["catalog"],root),
        current_descriptor=read_json(descriptor_path),current_descriptor_path=descriptor_path,repo_root=root)
    high,_ = build_high_detail_tracking_review(stage_dir/"high/tracking_high.validation.json",recipe_path=recipe_path,low_descriptor=low,repo_root=root)
    if low["room_frames"] != original_low["room_frames"] or high["room_frames"] != original_low["room_frames"]:
        raise ValueError("serialized stage changed a retained room frame/source binding")
    trees = {}
    for level in ("low","high"):
        original = checked_reference(source[level+"_glb"],root)
        old_tree,old_bin = unpack_glb(original)
        new_path = stage_dir/level/("multiroom_surface_mesh.glb" if level=="low" else "multiroom_surface_mesh_high_detail.glb")
        if sha_file(new_path)!=stage[level]["sha256"]:
            raise ValueError("stage artifact digest changed")
        tree,binary = unpack_glb(new_path)
        if binary[:len(old_bin)]!=old_bin:
            raise ValueError("retained old binary changed")
        for key in ("accessors","bufferViews","meshes","images","textures","materials"):
            if tree.get(key,[])[:len(old_tree.get(key,[]))] != old_tree.get(key,[]):
                raise ValueError(f"retained {level} {key} changed")
        trees[level] = tree,binary
    reports = {}
    for owner,name in ((1,"family"),(2,"kitchen"),(3,"living")):
        with np.load(stage_dir/f"owner_{owner}_upper_witness.npz",allow_pickle=False) as witness:
            vertices,faces,best = witness["vertices"],witness["faces"],witness["photo_view"]
        _,positions,indices = primitive(*trees["low"],f"pcf_surface_{name}_upper_rgb")
        if not np.array_equal(positions,vertices) or not np.array_equal(indices,faces):
            raise ValueError("serialized upper geometry differs from the checked float32 witness")
        views = load_bound_views(owner,recipe=original_recipe,multiroom=multiroom,prior_join=join,
            registration=registration,photo=photo,root=root)
        check_view_witnesses(classification[str(owner)]["source_view_witnesses"],views)
        corners = positions[indices]
        probes = np.concatenate((corners,corners.mean(1)[:,None],(corners+corners[:,[1,2,0]])/2),axis=1)
        accepted,votes = qualify(probes.reshape(-1,3),views)
        if not accepted.all():
            raise ValueError("serialized triangle sample no longer passes distinct-view source evidence")
        photo_faces = faces[best>=0]
        pair = np.stack((photo_faces.reshape(-1),np.repeat(best[best>=0],3)),axis=1)
        unique,inverse = np.unique(pair,axis=0,return_inverse=True)
        row,hpos,hfaces = primitive(*trees["high"],f"pcf_surface_{name}_upper_photo")
        if not np.array_equal(hpos,vertices[unique[:,0]]) or not np.array_equal(hfaces,inverse.reshape(-1,3)):
            raise ValueError("serialized photo seam geometry changed")
        uv = accessor_array(*trees["high"],row["attributes"]["TEXCOORD_0"])
        expected_uv = np.zeros_like(uv)
        for view in views:
            selected = unique[:,1] == view.evidence.view_index
            if np.any(selected):
                expected_uv[selected],inside = photo_coordinates(vertices[unique[selected,0]],view,
                    photo["parameters"]["atlas_tile_width"],photo["parameters"]["atlas_tile_height"])
                if not inside.all():
                    raise ValueError("serialized photo coordinates escaped the retained image tile")
        if not np.array_equal(uv,expected_uv):
            raise ValueError("serialized raw glTF UVs do not map the recorded source tile")
        reports[str(owner)] = {"triangle_count":len(faces),"all_7_serialized_samples_passed":True,
            "minimum_distinct_supporting_views":int(votes["support_count"].min()),
            "photo_triangle_count":len(photo_faces),"photo_uv_count":len(uv),"photo_uv_exact":True,
            "maximum_y_m":float(positions[:,1].max()),
            "upper_vertical_triangle_count":int((np.abs(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])[:,1]) /
                np.maximum(np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=1),1e-12)<.55).sum())}
    return {"status":"serialized_stage_checks_passed","retained_all_old_bin_accessors_meshes_images_materials_exact":True,
        "strict_low_high_binding_validation_passed":True,"room_frames_identical":low["room_frames"]==high["room_frames"],
        "owners":reports,"active_files_or_services_changed":False}


def archive_baseline(stage_dir):
    """Save two small exact source inputs before the activation owner swaps them.

    Artifact-local writes only; existing copies are never silently overwritten.
    The normal verifier remains read-only unless this option/report is requested.
    """
    root = Path(__file__).resolve().parents[2]
    source = read_json(stage_dir/"source_bindings.json")
    binding = checked_reference(source["binding"],root)
    descriptor = checked_reference(read_json(binding)["source_descriptor"],root)
    saved = {}
    for name,path in (("binding",binding),("descriptor",descriptor)):
        destination = stage_dir/f"retained_source_{name}.json"
        content = path.read_bytes()
        if destination.exists():
            if destination.read_bytes()!=content:
                raise ValueError("retained baseline snapshot already differs")
        else:
            with destination.open("xb") as target:
                target.write(content)
        saved[name] = reference(destination)
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-dir",type=Path,required=True)
    p.add_argument("--report",type=Path)
    p.add_argument("--archive-baseline",action="store_true",
        help="save exact original recipe/selector bytes in the stage before activation")
    args = p.parse_args(); result = verify(args.stage_dir)
    if args.archive_baseline:
        result["retained_source_snapshots"] = archive_baseline(args.stage_dir)
    if args.report:
        args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
