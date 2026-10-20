#!/usr/bin/env python3
"""Stage a ceiling-specific upper-surface extension of frozen low/high PCF meshes.

Original GLB BIN bytes, lower triangles and high-detail atlases are retained.
New geometry must pass same-owner distance and qualified distinct-view RGB-D
checks. Neither this command nor its validation writes an active selector,
binding, catalog, service environment or structural transform.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from .build_high_detail_pcf_hybrid import (
    append_accessor, append_blob, image_index_for_owner, pack_glb, read_json,
    sha_bytes, sha_file, unpack_glb,
)
from .build_high_detail_pcf_mesh import _manifest_source_for_owner, checked_matrix, world_from_local
from .build_pcf_review_surface_mesh import SurfaceViewEvidence, _qualified_surface_depth_votes
from .pcf_ceiling_filter import CeilingPatch, ceiling_triangle_mask, classify_points, discover_ceiling_patches
from .publish_pcf_review_surface_mesh import replacement_descriptor
from .tracking_review import build_high_detail_tracking_review, build_tracking_review


OWNERS = {1:"family",2:"kitchen",3:"living"}
CAMERAS = {1:"fixed_camera_positions",2:"moving_camera_positions",3:"living_camera_positions"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def reference(path: Path) -> dict:
    return {"path":str(path.resolve()),"sha256":sha_file(path),"size_bytes":path.stat().st_size}


def checked_reference(row: dict, root: Path) -> Path:
    path = Path(row["path"])
    path = path if path.is_absolute() else root / path
    if (path.is_symlink() or not path.is_file() or sha_file(path) != row["sha256"]
            or path.stat().st_size != row["size_bytes"]):
        raise ValueError(f"bound source changed: {path}")
    return path.resolve()


@dataclass
class BoundView:
    evidence: SurfaceViewEvidence
    from_assembly: np.ndarray
    model_k: np.ndarray
    affine: np.ndarray | None
    tile_xy: tuple[int,int]
    witness: dict


def load_bound_views(owner, *, recipe, multiroom, prior_join, registration, photo, root):
    raw_root, correction, recorded = _manifest_source_for_owner(
        owner,multiroom,prior_join,registration,recipe,root)
    camera = {1:"family-room",2:"kitchen",3:"living-room"}[owner]
    world = read_json(checked_reference(recipe["rooms"][camera]["source_world_manifest"],root))
    edge = checked_matrix(correction @ world_from_local(world),"source-to-assembly edge")
    rows = photo["owners"][str(owner)]["image_sources"]
    if len(rows) != 48 or [row["view_index"] for row in rows] != list(range(48)):
        raise ValueError("photo source does not retain the 48 exact view identities")
    tile_w, tile_h = photo["parameters"]["atlas_tile_width"],photo["parameters"]["atlas_tile_height"]
    views = []
    for index,row in enumerate(rows):
        raw_path = raw_root / f"view_{index:04d}.npz"
        if sha_file(raw_path) != row["raw_sha256"]:
            raise ValueError(f"owner {owner} raw view {index} changed")
        if recorded is not None:
            original = recorded[index]
            if (original["path"] != str(raw_path) or original["sha256"] != row["raw_sha256"]
                    or not np.allclose(original["effective_world_from_local_row_major"],edge,atol=2e-5)):
                raise ValueError("recorded raw frame edge changed")
        prepared = Path(row["prepared_frame_path"])
        if sha_file(prepared) != row["prepared_frame_sha256"]:
            raise ValueError("original prepared photo changed")
        with np.load(raw_path,allow_pickle=False) as raw:
            pose = edge @ checked_matrix(raw["camera_pose"],"raw camera pose")
            mask = np.asarray(raw["mask"],bool).copy()
            mask &= ~np.asarray(raw["cross_model_uncertain"],bool)
            evidence = SurfaceViewEvidence(view_index=index,
                depth_z=np.asarray(raw["depth_z"],np.float32).copy(),mask=mask,
                intrinsics=np.asarray(raw["intrinsics"],float).copy(),camera_to_world=pose,
                quality=np.asarray(raw["confidence"],np.float32).copy())
        model_path = Path(row["conditioned_model_rgb_path"])
        with np.load(model_path,allow_pickle=False) as model:
            model_k = np.asarray(model["intrinsics"],float).copy()
        affine = np.asarray(row["model_to_atlas_row_major"],float) if row["image_status"] == "accepted" else None
        if affine is not None and (affine.shape != (2,3) or not np.isfinite(affine).all()):
            raise ValueError("accepted source photo map is malformed")
        witness = {**row,"raw_path":str(raw_path),"conditioned_model_sha256":sha_file(model_path),
                   "camera_to_assembly_row_major":pose.tolist(),
                   "effective_world_from_local_row_major":edge.tolist()}
        views.append(BoundView(evidence,np.linalg.inv(pose),model_k,affine,
                              ((index%8)*tile_w,(index//8)*tile_h),witness))
    return views


def qualify(points, views, normals=None):
    return _qualified_surface_depth_votes(points,[view.evidence for view in views],normals=normals)


def check_view_witnesses(recorded,views):
    # The original photo manifest pins prepared/raw images but predates a
    # conditioned-model file hash. Frozen classification records that hash and
    # the actual camera edge: changes must not silently reuse an old atlas map.
    if recorded != [view.witness for view in views]:
        raise ValueError("frozen exact source-view/model/atlas witnesses changed")


def patch_fingerprint(patch):
    return sha_bytes(np.asarray(patch.seeds,dtype="<f8").tobytes(order="C"))


def reviewed_patches(patches,owner,review):
    """Select by frozen seed geometry, NEVER by unstable discovery ordinal."""
    accepted = review["accepted_patch_seed_sha256"][str(owner)]
    known = {patch_fingerprint(patch):patch for patch in patches}
    if len(known) != len(patches) or len(set(accepted)) != len(accepted) or any(key not in known for key in accepted):
        raise ValueError("semantic ceiling review has unknown/duplicate seed geometry")
    return [known[key] for key in accepted]


def points_in_polygon(xy,polygon):
    polygon = np.asarray(polygon,dtype=float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
        raise ValueError("source semantic region must be a finite pixel polygon")
    x,y = np.asarray(xy).T
    inside = np.zeros(len(x),dtype=bool)
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0),strict=True):
        crossing = (a[1]>y)!=(b[1]>y)
        intersection = (b[0]-a[0])*(y-a[1])/(b[1]-a[1] if b[1]!=a[1] else 1)+a[0]
        inside ^= crossing&(x<intersection)
    return inside


def source_region_protection(points,views,regions):
    """Conservative source-reviewed beam/soffit protection, never geometry edits.

    A labelled source pixel protects a matching measured surface only. Source
    mask, uncertainty, confidence and depth still apply; an occluded projection
    cannot protect an unrelated surface behind it.
    """
    protected = np.zeros(len(points),bool)
    for region in regions:
        index = region["view_index"]
        if type(index) is not int or not 0<=index<len(views) or not region.get("reason"):
            raise ValueError("protected semantic region lacks its source view/reason")
        view = views[index]; evidence = view.evidence
        camera = points @ view.from_assembly[:3,:3].T + view.from_assembly[:3,3]
        z = camera[:,2]
        pixels = camera @ evidence.intrinsics.T
        xy = pixels[:,:2]/np.where(z>0,z,1)[:,None]
        uv = np.rint(xy).astype(int)
        h,w = evidence.depth_z.shape
        inside = ((z>=.05)&(z<=12)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
                  &points_in_polygon(xy,region["polygon_pixels"]))
        rows = np.flatnonzero(inside); u,v = uv[rows,0],uv[rows,1]
        depth = evidence.depth_z[v,u]
        matches = (evidence.mask[v,u]&(evidence.quality[v,u]>=.35)&np.isfinite(depth)
                   &(np.abs(z[rows]-depth)<=.05+.03*np.maximum(depth,z[rows])))
        protected[rows[matches]] = True
    return protected


def append_mesh(tree,binary,owner,mesh,*,suffix,material=None,uv=None):
    """Append a child while leaving every retained BIN byte/accessor unchanged."""
    if not len(mesh.faces):
        return {"vertex_count":0,"triangle_count":0}
    name = f"pcf_surface_{OWNERS[owner]}"
    matches = [node for node in tree["nodes"] if node.get("name") == name]
    if len(matches) != 1:
        raise ValueError("bound owner group is absent or ambiguous")
    group = matches[0]
    if "mesh" in group:
        child = len(tree["nodes"])
        tree["nodes"].append({"name":name+"_retained_low","mesh":group.pop("mesh")})
        group.setdefault("children",[]).append(child)
    positions = np.asarray(mesh.vertices,dtype="<f4")
    normals = np.asarray(mesh.vertex_normals,dtype="<f4")
    faces = np.asarray(mesh.faces,dtype="<u4")
    attributes = {}
    for key,values in (("POSITION",positions),("NORMAL",normals)):
        if not np.isfinite(values).all():
            raise ValueError("extension has nonfinite geometry")
        blob = append_blob(tree,binary,values.tobytes(),34962)
        attributes[key] = append_accessor(tree,blob,5126,len(values),"VEC3",
            minimum=values.min(0).tolist() if key == "POSITION" else None,
            maximum=values.max(0).tolist() if key == "POSITION" else None)
    if uv is None:
        colors = np.asarray(mesh.visual.vertex_colors,dtype=np.uint8)
        if colors.shape != (len(positions),4):
            raise ValueError("extension needs explicit measured RGBA")
        blob = append_blob(tree,binary,colors.tobytes(),34962)
        attributes["COLOR_0"] = append_accessor(tree,blob,5121,len(colors),"VEC4",normalized=True)
    else:
        uv = np.asarray(uv,dtype="<f4")
        if uv.shape != (len(positions),2) or not np.isfinite(uv).all() or np.any((uv<0)|(uv>1)):
            raise ValueError("extension UV is invalid")
        blob = append_blob(tree,binary,uv.tobytes(),34962)
        attributes["TEXCOORD_0"] = append_accessor(tree,blob,5126,len(uv),"VEC2")
    blob = append_blob(tree,binary,faces.tobytes(),34963)
    primitive = {"attributes":attributes,"indices":append_accessor(tree,blob,5125,faces.size,"SCALAR"),"mode":4}
    if material is not None:
        primitive["material"] = material
    mesh_index = len(tree["meshes"])
    tree["meshes"].append({"name":name+suffix,"primitives":[primitive]})
    child = len(tree["nodes"])
    tree["nodes"].append({"name":name+suffix,"mesh":mesh_index})
    group.setdefault("children",[]).append(child)
    return {"vertex_count":len(positions),"triangle_count":len(faces)}


def photo_coordinates(points,view,tile_w,tile_h):
    camera = np.asarray(points) @ view.from_assembly[:3,:3].T + view.from_assembly[:3,3]
    z = camera[:,2]
    model = camera @ view.model_k.T
    xy = model[:,:2] / np.where(z>0,z,1)[:,None]
    if view.affine is None:
        return np.zeros((len(points),2)),np.zeros(len(points),bool)
    local = xy @ view.affine[:,:2].T + view.affine[:,2]
    inside = ((z>.05)&(local[:,0]>=1)&(local[:,0]<=tile_w-2)
              &(local[:,1]>=1)&(local[:,1]<=tile_h-2))
    atlas = local + np.asarray(view.tile_xy)
    # These values are written DIRECTLY to glTF, whose V origin is the image's
    # upper edge. Unlike Trimesh's lower-left UV convention, no exporter flip
    # follows this append operation. Existing atlas pixels are never flipped.
    uv = np.column_stack((atlas[:,0]/(tile_w*8),atlas[:,1]/(tile_h*6)))
    return uv,inside


def photo_extension(mesh,views,*,tile_w,tile_h):
    vertices,faces = np.asarray(mesh.vertices),np.asarray(mesh.faces)
    center,normals = vertices[faces].mean(1),np.asarray(mesh.face_normals)
    best,score = np.full(len(faces),-1,dtype=int),np.zeros(len(faces))
    # RGB-D support is already mandatory for every new triangle sample. The
    # selected photo additionally has to cover all corners AND its centroid.
    from .build_high_detail_pcf_mesh import sample_depth, View
    for view in views:
        if view.affine is None:
            continue
        evidence = view.evidence
        adapter = View(evidence.view_index,evidence.depth_z,evidence.mask,evidence.quality,
            evidence.intrinsics,view.from_assembly,view.model_k,view.affine,view.tile_xy,
            Path(view.witness["raw_path"]),view.witness["raw_sha256"],
            Path(view.witness["conditioned_model_rgb_path"]),Path(view.witness["prepared_frame_path"]),
            view.witness["prepared_frame_sha256"],0,0,0.)
        visible,free,quality = sample_depth(vertices,adapter)
        cvisible,cfree,cquality = sample_depth(center,adapter)
        _,inside = photo_coordinates(vertices,view,tile_w,tile_h)
        _,cinside = photo_coordinates(center,view,tile_w,tile_h)
        toward = evidence.camera_to_world[:3,3] - center
        distance = np.linalg.norm(toward,axis=1)
        cosine = np.abs(np.einsum("ij,ij->i",normals,toward))/np.maximum(distance,1e-8)
        possible = (visible[faces].all(1)&cvisible&~free[faces].any(1)&~cfree
                    &inside[faces].all(1)&cinside&(quality[faces].min(1)>=.35)&(cquality>=.35)&(cosine>=.17))
        candidate = np.minimum(cquality,quality[faces].min(1))*np.sqrt(cosine)/np.maximum(distance,.4)
        better = possible&(candidate>score)
        best[better],score[better] = evidence.view_index,candidate[better]
    photo = best >= 0
    selected = faces[photo]
    pair = np.stack((selected.reshape(-1),np.repeat(best[photo],3)),axis=1)
    unique,inverse = np.unique(pair,axis=0,return_inverse=True)
    uv = np.zeros((len(unique),2),np.float32)
    for view in views:
        chosen = unique[:,1] == view.evidence.view_index
        if np.any(chosen):
            uv[chosen],inside = photo_coordinates(vertices[unique[chosen,0]],view,tile_w,tile_h)
            if not inside.all():
                raise ValueError("photo corner unexpectedly escaped its exact tile")
    textured = trimesh.Trimesh(vertices=vertices[unique[:,0]],faces=inverse.reshape(-1,3),process=False)
    fallback = mesh.submesh([~photo],append=True,repair=False) if np.any(~photo) else None
    return textured,uv,fallback,best


def build_extension(points,colors,samples,normals,removed,patches,views,cameras,old_max_y,protected):
    # This lower bound is a finite extension-work window at the old mesh seam,
    # NOT a ceiling or source removal rule. All retained lower bytes survive.
    selected = (~removed)&(points[:,1]>=old_max_y-.25)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[selected].astype(float)))
    cloud.colors = o3d.utility.Vector3dVector(colors[selected].astype(float)/255)
    cloud = cloud.voxel_down_sample(.04)
    values = np.asarray(cloud.points)
    if len(values)<1000:
        raise ValueError("upper extension has insufficient source geometry")
    cloud.normals = o3d.utility.Vector3dVector(normals[cKDTree(samples).query(values,workers=1)[1]])
    _,votes = qualify(values,views,np.asarray(cloud.normals))
    orientation = votes["normal_direction_vote"]
    signed = np.asarray(cloud.normals)
    signed[orientation<0] *= -1
    abstain = orientation == 0
    toward = cameras[cKDTree(cameras).query(values[abstain],workers=1)[1]] - values[abstain]
    signed[abstain] *= np.where(np.einsum("ij,ij->i",signed[abstain],toward)<0,-1,1)[:,None]
    mesh,density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,depth=8,scale=1.05,linear_fit=True,n_threads=1)
    verts = np.asarray(mesh.vertices)
    tree = cKDTree(values)
    distance = tree.query(verts,workers=1)[0]
    bounds = (verts>=values.min(0)-.075).all(1)&(verts<=values.max(0)+.075).all(1)
    keep = bounds&(distance<=.075)&(np.asarray(density)>=np.percentile(density,2))
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
    # Qualify exactly the float32 coordinates that the GLB will serialize.
    verts,faces = np.asarray(mesh.vertices,dtype=np.float32),np.asarray(mesh.triangles)
    corners = verts[faces]
    centroid = corners.mean(1)
    roof = ceiling_triangle_mask(verts,faces,patches,
        junction_points=np.vstack((samples[np.abs(normals[:,1])<.55],points[protected])))
    upper = (corners[:,:,1].max(1)>old_max_y+.002)&(centroid[:,1]>=old_max_y-.02)
    edge = np.linalg.norm(corners-corners[:,[1,2,0]],axis=2).max(1)
    source_distance = tree.query(centroid,workers=1)[0]
    preliminary = upper&~roof&(edge<=.15)&(source_distance<=.065)
    # Seven interior/boundary samples prevent reclosure and unsupported joins.
    probes = np.concatenate((corners[preliminary],centroid[preliminary,None],
                             (corners[preliminary]+corners[preliminary][:,[1,2,0]])/2),axis=1)
    supported,votes = qualify(probes.reshape(-1,3),views)
    accepted = supported.reshape(-1,7).all(1)
    selected_faces = np.flatnonzero(preliminary)[accepted]
    rgb = np.uint8(np.clip(np.rint(np.asarray(mesh.vertex_colors)*255),0,255))
    output = trimesh.Trimesh(vertices=verts,faces=faces[selected_faces],
        vertex_colors=np.column_stack((rgb,np.full(len(rgb),255,np.uint8))),process=False)
    output.remove_unreferenced_vertices()
    return output,{"poisson_triangle_count":len(faces),"ceiling_triangle_rejected_count":int(roof.sum()),
        "upper_distance_edge_candidate_count":int(preliminary.sum()),
        "qualified_ray_rejected_triangle_count":int(np.count_nonzero(~accepted)),
        "new_triangle_count":len(output.faces),"new_vertex_count":len(output.vertices),
        "rgbd_sample_support_count_min":int(votes["support_count"].reshape(-1,7)[accepted].min()) if accepted.any() else None,
        "source_work_window_min_y_m":old_max_y-.25,"maximum_triangle_edge_m":.15,
        "vertex_support_distance_m":.075,"centroid_support_distance_m":.065}


def build(args):
    root = Path(__file__).resolve().parents[2]
    if not args.classify_only and (args.classification_dir is None or args.ceiling_review is None):
        raise ValueError("building needs a frozen classification and source-reviewed seed fingerprints")
    recipe = read_json(args.binding)
    current_path = checked_reference(recipe["source_descriptor"],root)
    current = read_json(current_path)
    catalog = checked_reference(recipe["catalog"],root)
    old_low = build_tracking_review(args.binding,catalog_path=catalog,current_descriptor=current,
        current_descriptor_path=current_path,repo_root=root)
    old_high,high_path = build_high_detail_tracking_review(args.high_manifest,recipe_path=args.binding,
        low_descriptor=old_low,repo_root=root)
    high_compact,high_full = read_json(args.high_manifest),read_json(args.high_manifest.parent/"mesh_manifest.json")
    if sha_file(args.photo_source_manifest) != high_full["source"]["v2_detail_manifest_sha256"]:
        raise ValueError("original photo-view witness differs from the retained high derivative")
    photo = read_json(args.photo_source_manifest)
    old_mesh_manifest = checked_reference(recipe["mesh_manifest"],root)
    reintegration_path = checked_reference(recipe["reintegration"],root)
    multiroom = read_json(reintegration_path)
    join = read_json(checked_reference(recipe["family_kitchen"],root))
    registration = read_json(checked_reference(recipe["endpoint_registration"],root))
    surfels = reintegration_path.parent / multiroom["artifacts"]["surfels_npz"]["path"]
    if sha_file(surfels) != multiroom["artifacts"]["surfels_npz"]["sha256"]:
        raise ValueError("untrimmed source surfels changed")
    low_path = args.pcf_storage_root/current["artifact"]["relative_path"]
    low_tree,low_bin = unpack_glb(low_path); high_tree,high_bin = unpack_glb(high_path)
    low_binary,high_binary = bytearray(low_bin),bytearray(high_bin)
    low_scene = trimesh.load(low_path,force="scene",process=False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/"low").mkdir(); (output/"high").mkdir()
    with np.load(surfels,allow_pickle=False) as data:
        points,colors,owners = data["points"].copy(),data["colors"].copy(),data["owner_room_id"].copy()
        cameras = {owner:data[CAMERAS[owner]].copy() for owner in OWNERS}
    ceiling_mask,ambiguous_mask = np.zeros(len(points),bool),np.zeros(len(points),bool)
    low_metrics,high_metrics,evidence_report = {},copy.deepcopy(high_full["owners"]),{}
    review = read_json(args.ceiling_review) if args.ceiling_review else {}
    if review and review.get("source_surfels_sha256") != sha_file(surfels):
        raise ValueError("ceiling review does not bind the exact source")
    frozen = None
    if args.classification_dir is not None:
        if not review:
            raise ValueError("frozen classification requires an exact semantic review binding")
        selected_classification = checked_reference(review["source_classification"],root)
        if selected_classification != (args.classification_dir/"ceiling_classification.json").resolve():
            raise ValueError("semantic review names a different frozen classification")
        frozen = read_json(selected_classification)
        frozen_bindings = read_json(args.classification_dir/"source_bindings.json")
        for key,current_ref in (("source_surfels",reference(surfels)),("reintegration",reference(reintegration_path)),
                ("binding",reference(args.binding)),("low_glb",reference(low_path)),
                ("high_manifest",reference(args.high_manifest)),("high_glb",reference(high_path)),
                ("photo_source_manifest",reference(args.photo_source_manifest))):
            if frozen_bindings.get(key) != current_ref:
                raise ValueError("frozen ceiling classification belongs to a different exact source chain")
    for owner,name in OWNERS.items():
        print(f"owner {owner}: exact source views and ceiling classification",flush=True)
        views = load_bound_views(owner,recipe=recipe,multiroom=multiroom,prior_join=join,
            registration=registration,photo=photo,root=root)
        rows = np.flatnonzero(owners==owner)
        values = points[rows]
        if frozen is None:
            samples,normals,patches,report = discover_ceiling_patches(values,cameras[owner],lambda p,n:qualify(p,views,n))
            for index,patch in enumerate(patches):
                patch.metrics["patch_id"] = index
                patch.metrics["seed_sha256"] = patch_fingerprint(patch)
            active = patches
        else:
            archive_path = checked_reference(review["candidate_archives"][str(owner)],root)
            report = copy.deepcopy(frozen[str(owner)])
            check_view_witnesses(report["source_view_witnesses"],views)
            with np.load(archive_path,allow_pickle=False) as archive:
                samples,normals = archive["samples"].copy(),archive["normals"].copy()
                patches = [CeilingPatch(np.asarray(row["normal"]),row["offset_m"],
                    archive[f"patch_{row['patch_id']}_seeds"].copy(),copy.deepcopy(row)) for row in report["patches"]]
            active = reviewed_patches(patches,owner,review)
            # Recheck the frozen accepted seed geometry against today's exact
            # source files; require every retained seed's camera hemisphere.
            for patch in active:
                accepted,votes = qualify(patch.seeds,views,np.tile(patch.normal,(len(patch.seeds),1)))
                if not accepted.all() or np.any(votes["normal_direction_vote"]>=0):
                    raise ValueError("frozen ceiling seed RGB-D/hemisphere qualification changed")
        excluded = [row["patch_id"] for row in report["patches"]
                    if row["patch_id"] not in [patch.metrics["patch_id"] for patch in active]]
        nearest = cKDTree(samples).query(values,workers=1)[1]
        removed,ambiguous = classify_points(values,normals[nearest],active)
        protected = source_region_protection(values,views,
            review.get("protected_source_regions",{}).get(str(owner),[]))
        ambiguous |= removed&protected
        removed &= ~protected
        nominated = np.flatnonzero(removed)
        qualified,_ = qualify(values[nominated],views)
        removed[nominated[~qualified]] = False
        ambiguous[nominated[~qualified]] = True
        ceiling_mask[rows],ambiguous_mask[rows] = removed,ambiguous
        report.update(source_point_count=len(rows),qualified_ceiling_removed_count=int(removed.sum()),
            ambiguous_retained_count=int(ambiguous.sum()),retained_patch_ids=excluded,
            source_semantic_protected_point_count=int(protected.sum()),
            protected_source_regions=review.get("protected_source_regions",{}).get(str(owner),[]),
            source_view_witnesses=[view.witness for view in views])
        evidence_report[str(owner)] = report
        np.savez_compressed(output/f"owner_{owner}_ceiling_candidates.npz",samples=samples,normals=normals,
            **{f"patch_{i}_seeds":patch.seeds for i,patch in enumerate(patches)})
        write_json(output/"ceiling_classification.json",evidence_report)
        if args.classify_only:
            continue
        original = low_scene.geometry[f"pcf_surface_{name}"]
        retained_roof = ceiling_triangle_mask(np.asarray(original.vertices),np.asarray(original.faces),active)
        if retained_roof.any():
            raise ValueError("retained base contains qualified ceiling faces; append-only preservation cannot bypass their removal")
        old_max = float(np.asarray(original.vertices)[:,1].max())
        print(f"owner {owner}: upper Poisson, no height cap; {removed.sum()} qualified ceiling points removed",flush=True)
        extension,metrics = build_extension(values,colors[rows],samples,normals,removed,active,views,cameras[owner],old_max,protected)
        if not len(extension.faces):
            raise ValueError(f"owner {owner} has no qualified upper extension")
        extension.export(output/f"owner_{owner}_upper.glb")
        low_counts = append_mesh(low_tree,low_binary,owner,extension,suffix="_upper_rgb")
        photo_mesh,uv,fallback,best = photo_extension(extension,views,
            tile_w=photo["parameters"]["atlas_tile_width"],tile_h=photo["parameters"]["atlas_tile_height"])
        photo_node = next(node for node in high_tree["nodes"] if node.get("name")==f"pcf_surface_{name}_photo")
        material = high_tree["meshes"][photo_node["mesh"]]["primitives"][0]["material"]
        pc = append_mesh(high_tree,high_binary,owner,photo_mesh,suffix="_upper_photo",material=material,uv=uv)
        fc = append_mesh(high_tree,high_binary,owner,fallback,suffix="_upper_rgb") if fallback is not None else {"vertex_count":0,"triangle_count":0}
        low_metrics[str(owner)] = {**metrics,"retained_triangle_count":len(original.faces),
            "retained_base_qualified_ceiling_triangle_count":int(retained_roof.sum()),
            "output_triangle_count":len(original.faces)+low_counts["triangle_count"],
            "output_vertex_count":len(original.vertices)+low_counts["vertex_count"],
            "extension_photo_triangle_count":pc["triangle_count"],"extension_rgb_fallback_triangle_count":fc["triangle_count"]}
        high_metrics[str(owner)]["upper_extension"] = {**metrics,"photo":pc,"rgb_fallback":fc}
        np.savez_compressed(output/f"owner_{owner}_upper_witness.npz",vertices=np.asarray(extension.vertices,dtype=np.float32),
            faces=np.asarray(extension.faces),photo_view=best,source_row_indices=rows,
            classified_ceiling=removed,ambiguous_retained=ambiguous)
    np.savez_compressed(output/"ceiling_source_indices.npz",owner_room_id=owners,
        ceiling_removed=ceiling_mask,ambiguous_retained=ambiguous_mask,
        retained_non_ceiling_or_unclassified=~ceiling_mask&~ambiguous_mask)
    write_json(output/"source_bindings.json",{"source_surfels":reference(surfels),"reintegration":reference(reintegration_path),
        "binding":reference(args.binding),"low_glb":reference(low_path),"high_manifest":reference(args.high_manifest),
        "high_glb":reference(high_path),"photo_source_manifest":reference(args.photo_source_manifest)})
    if args.classify_only:
        return {"status":"classified_not_built","output":str(output),"owners":evidence_report}
    low_glb,high_glb = output/"low/multiroom_surface_mesh.glb",output/"high/multiroom_surface_mesh_high_detail.glb"
    low_glb.write_bytes(pack_glb(low_tree,bytes(low_binary)))
    high_glb.write_bytes(pack_glb(high_tree,bytes(high_binary)))
    for path,original_bin in ((low_glb,low_bin),(high_glb,high_bin)):
        _,new_bin = unpack_glb(path)
        if new_bin[:len(original_bin)] != original_bin:
            raise ValueError("retained GLB binary changed")
    low_output = {"path":low_glb.name,"sha256":sha_file(low_glb),"size_bytes":low_glb.stat().st_size,
        "vertex_count":sum(m["output_vertex_count"] for m in low_metrics.values()),
        "triangle_count":sum(m["output_triangle_count"] for m in low_metrics.values())}
    low_manifest = {"contract":"noesis.pcf.review_surface_mesh","contract_version":1,
        "status":"review_only","accepted_for_canonical_use":False,"coordinate_frame":current["assembly_gauge"],
        "method":"frozen_legacy_lower_mesh_plus_ceiling_specific_rgbd_qualified_upper_extension",
        "source":{"surfels_npz":str(surfels),"surfels_npz_sha256":sha_file(surfels),
            "surfels_manifest":str(reintegration_path),"surfels_manifest_sha256":sha_file(reintegration_path),
            "retained_low_glb_sha256":sha_file(low_path),"retained_low_mesh_manifest_sha256":sha_file(old_mesh_manifest),
            "source_descriptor_sha256":sha_file(current_path)},"output":low_output,"owners":low_metrics,
        "frame_identity":{"coordinate_frame":current["assembly_gauge"],"source_manifest_sha256":sha_file(reintegration_path)},
        "ceiling_filter":{"classification":reference(output/"ceiling_classification.json"),
            "source_indices":reference(output/"ceiling_source_indices.npz"),"review":review},
        "limitations":["unsupported ceiling-like points remain ambiguous","legacy lower mesh is not reclassified or ray-readmitted",
            "new upper geometry is a source-backed presentation derivative, not canonical world admission",
            "registration status and all calibration/catalog/room edges remain unchanged"]}
    low_manifest_path = output/"low/surface_mesh_manifest.json"; write_json(low_manifest_path,low_manifest)
    descriptor = replacement_descriptor(current=current,current_path=current_path,storage_root=args.pcf_storage_root,
        source_mesh_manifest=old_mesh_manifest,mesh_glb=low_glb,mesh_manifest=low_manifest_path,assembly_id=args.assembly_id)
    descriptor_path = output/"low/manifest.json"; write_json(descriptor_path,descriptor)
    production_recipe = copy.deepcopy(recipe)
    production_recipe["source_descriptor"] = {**reference(descriptor_path),"path":str(current_path)}
    production_recipe["mesh_manifest"] = reference(low_manifest_path)
    production_recipe_path = output/"binding.activation.json"; write_json(production_recipe_path,production_recipe)
    validation_recipe = copy.deepcopy(production_recipe)
    validation_recipe["source_descriptor"] = reference(descriptor_path)
    validation_recipe_path = output/"binding.validation.json"; write_json(validation_recipe_path,validation_recipe)
    compact = copy.deepcopy(high_compact)
    compact["assembly_id"] = args.assembly_id
    compact["source"].update(binding_sha256=sha_file(production_recipe_path),low_glb_sha256=low_output["sha256"],
        low_mesh_manifest_sha256=sha_file(low_manifest_path))
    counts = {key:high_compact["output"]["owner_triangle_counts"][key]+m["new_triangle_count"] for key,m in low_metrics.items()}
    high_vertices = high_compact["output"]["vertex_count"] + sum(row["upper_extension"]["photo"]["vertex_count"]
        +row["upper_extension"]["rgb_fallback"]["vertex_count"] for row in high_metrics.values())
    compact["output"].update(sha256=sha_file(high_glb),size_bytes=high_glb.stat().st_size,
        vertex_count=high_vertices,triangle_count=sum(counts.values()),owner_triangle_counts=counts)
    for owner,row in compact["owners"].items():
        delta = high_metrics[owner]["upper_extension"]
        row["direct_photo_triangle_count"] += delta["photo"]["triangle_count"]
        row["fallback_triangle_count"] += delta["rgb_fallback"]["triangle_count"]
        source = args.high_manifest.parent/row["texture_path"]
        destination = output/"high"/row["texture_path"]
        shutil.copyfile(source,destination)
        if sha_file(destination) != row["texture_sha256"]:
            raise ValueError("retained high atlas copy changed")
    full = copy.deepcopy(high_full)
    full.update(assembly_id=args.assembly_id,output=compact["output"],owners=high_metrics,
        method="frozen_high_photo_buffers_plus_ceiling_specific_rgbd_qualified_upper_extension")
    full["source"].update(binding_sha256=sha_file(production_recipe_path),low_glb_sha256=low_output["sha256"],
        low_mesh_manifest_path=str(low_manifest_path),low_mesh_manifest_sha256=sha_file(low_manifest_path),
        retained_high_glb_sha256=sha_file(high_path),retained_high_bin_sha256=sha_bytes(high_bin),
        retained_high_manifest_sha256=sha_file(args.high_manifest.parent/"mesh_manifest.json"))
    full["ceiling_filter"] = low_manifest["ceiling_filter"]
    full_path = output/"high/mesh_manifest.json"; write_json(full_path,full)
    compact["source"]["detail_manifest_sha256"] = sha_file(full_path)
    high_manifest_path = output/"high/tracking_high.json"; write_json(high_manifest_path,compact)
    validation_compact = copy.deepcopy(compact)
    validation_compact["source"]["binding_sha256"] = sha_file(validation_recipe_path)
    high_validation_path = output/"high/tracking_high.validation.json"; write_json(high_validation_path,validation_compact)
    low = build_tracking_review(validation_recipe_path,catalog_path=catalog,current_descriptor=descriptor,
        current_descriptor_path=descriptor_path,repo_root=root)
    high,_ = build_high_detail_tracking_review(high_validation_path,recipe_path=validation_recipe_path,
        low_descriptor=low,repo_root=root)
    if low["room_frames"] != old_low["room_frames"] or high["room_frames"] != old_high["room_frames"]:
        raise ValueError("staged producer changed room transforms or source bindings")
    result = {"status":"staged_not_activated","output":str(output),"low":low_output,"high":compact["output"],
        "retained_low_bin_exact":True,"retained_high_bin_exact":True,"room_frames_exact":True,
        "validation_recipe":reference(validation_recipe_path),"activation_recipe":reference(production_recipe_path),
        "activation_high_manifest":reference(high_manifest_path),"active_selector_unchanged":sha_file(current_path)==recipe["source_descriptor"]["sha256"]}
    write_json(output/"stage_validation.json",result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("binding","high-manifest","photo-source-manifest","pcf-storage-root","output-dir"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--assembly-id",required=True)
    p.add_argument("--classify-only",action="store_true")
    p.add_argument("--ceiling-review",type=Path)
    p.add_argument("--classification-dir",type=Path,
                   help="reuse the exact source-reviewed classification; never rediscover ordinal patch IDs during build")
    return p


if __name__ == "__main__":
    result = build(parser().parse_args())
    print(json.dumps(result if result["status"]!="classified_not_built" else {"status":result["status"],"output":result["output"]},indent=2))
