#!/usr/bin/env python3
"""Matched full-height direct-GLB CPU renders of a staged upper extension.

Uses explicitly supplied retained rasterizer code, exact original source poses,
and serialized old/new low/high meshes. No height clipping, filling or exposure
adjustment. Evidence masks distinguish newly visible geometry from the base.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .build_high_detail_pcf_hybrid import read_json,sha_file,unpack_glb


def accessor_array(tree,binary,index):
    row = tree["accessors"][index]; view = tree["bufferViews"][row["bufferView"]]
    types = {5121:"u1",5123:"<u2",5125:"<u4",5126:"<f4"}
    width = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[row["type"]]
    dtype = np.dtype(types[row["componentType"]])
    if view.get("byteStride",width*dtype.itemsize) != width*dtype.itemsize:
        raise ValueError("diagnostic expects tightly packed frozen mesh attributes")
    offset = view.get("byteOffset",0)+row.get("byteOffset",0)
    return np.frombuffer(binary,dtype=dtype,count=row["count"]*width,offset=offset).reshape(-1,width)


def parts_for(renderer,path):
    tree,binary = unpack_glb(path)
    parts,consumer = renderer.load_parts(path)
    nodes = {node.get("name"):node for node in tree["nodes"]}
    for part in parts:
        if part.has_texture:
            continue
        primitive = tree["meshes"][nodes[part.node_name]["mesh"]]["primitives"]
        if len(primitive)!=1 or "COLOR_0" not in primitive[0]["attributes"]:
            raise ValueError("untextured mesh has no explicit serialized RGB")
        colors = accessor_array(tree,binary,primitive[0]["attributes"]["COLOR_0"])
        if colors.dtype == np.float32:
            colors = np.uint8(np.clip(np.rint(colors*255),0,255))
        if colors.shape[1]==3:
            colors = np.column_stack((colors,np.full(len(colors),255,np.uint8)))
        if colors.shape != (len(part.vertices_world),4):
            raise ValueError("serialized native colour/geometry mismatch")
        part.rgba = np.ascontiguousarray(colors,dtype=np.uint8)
    return parts,consumer


def build(args):
    root = args.stage_dir
    source,classification = read_json(root/"source_bindings.json"),read_json(root/"ceiling_classification.json")
    stage = read_json(root/"stage_validation.json")
    paths = {"old_low":Path(source["low_glb"]["path"]),"new_low":root/"low/multiroom_surface_mesh.glb",
             "old_high":Path(source["high_glb"]["path"]),"new_high":root/"high/multiroom_surface_mesh_high_detail.glb"}
    expected = {"old_low":source["low_glb"]["sha256"],"new_low":stage["low"]["sha256"],
                "old_high":source["high_glb"]["sha256"],"new_high":stage["high"]["sha256"]}
    if any(sha_file(path)!=expected[key] for key,path in paths.items()):
        raise ValueError("matched render source digest changed")
    spec = importlib.util.spec_from_file_location("explicit_retained_pcf_cpu_renderer",args.renderer)
    renderer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = renderer; spec.loader.exec_module(renderer)
    raster = renderer.raster_library(args.rasterizer)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cameras = []
    for owner,index in ((1,29),(1,42),(2,16),(3,3),(3,39)):
        row = classification[str(owner)]["source_view_witnesses"][index]
        if sha_file(Path(row["raw_path"])) != row["raw_sha256"]:
            raise ValueError("matched camera source changed")
        with np.load(row["raw_path"],allow_pickle=False) as data:
            k = np.diag([2.,2.,1.]) @ data["intrinsics"]
        cameras.append((owner,index,k,np.asarray(row["camera_to_assembly_row_major"])))
    consumers,metrics = {},{}
    names = {1:"family",2:"kitchen",3:"living"}
    for variant,path in paths.items():
        parts,consumer = parts_for(renderer,path)
        consumers[variant] = consumer
        for owner,index,k,pose in cameras:
            selected = [part for part in parts if part.node_name.startswith("pcf_surface_"+names[owner])]
            if not selected:
                raise ValueError("render lacks exact room owner")
            slug = f"owner_{owner}_view_{index:02d}"
            print(f"render {slug}/{variant}",flush=True)
            buffers,values = renderer.render(selected,k,pose,width=1008,height=560,raster=raster,flip_texture_v=False)
            for key in ("rgb","flat"):
                Image.fromarray(buffers[key]).save(args.output_dir/f"{slug}_{variant}_{key}.png")
            np.savez_compressed(args.output_dir/f"{slug}_{variant}_framebuffer.npz",
                                depth=buffers["depth"],part=buffers["mesh"],face=buffers["face"])
            metrics.setdefault(slug,{})[variant] = values
        del parts; gc.collect()
    for owner,index,_,_ in cameras:
        slug = f"owner_{owner}_view_{index:02d}"
        for kind in ("rgb","flat"):
            sheet = Image.new("RGB",(2016,1190),(28,28,28)); draw = ImageDraw.Draw(sheet)
            for i,variant in enumerate(("old_low","new_low","old_high","new_high")):
                x,y = (i%2)*1008, (i//2)*595
                draw.text((x+10,y+8),variant.replace("_"," ")+" — original camera, full height",fill="white")
                with Image.open(args.output_dir/f"{slug}_{variant}_{kind}.png") as image:
                    sheet.paste(image,(x,y+30))
            sheet.save(args.output_dir/f"{slug}_{kind}_comparison.jpg",quality=94)
        for level in ("low","high"):
            with np.load(args.output_dir/f"{slug}_old_{level}_framebuffer.npz") as old,np.load(args.output_dir/f"{slug}_new_{level}_framebuffer.npz") as new:
                old_depth,new_depth = old["depth"],new["depth"]
                original,visible = np.isfinite(old_depth),np.isfinite(new_depth)
                changed = original&visible&(new_depth<old_depth-.001)
                added = ~original&visible
                mask = np.zeros((*original.shape,3),np.uint8)
                mask[visible] = (160,160,160); mask[added] = (30,230,255); mask[changed] = (255,160,30)
                Image.fromarray(mask).save(args.output_dir/f"{slug}_{level}_evidence_mask.png")
                metrics[slug][f"{level}_extension_mask"] = {"new_coverage_pixels":int(added.sum()),
                    "new_front_surface_pixels":int(changed.sum()),"retained_original_coverage_pixels":int((original&visible).sum())}
    report = {"status":"matched_direct_glb_render_complete","synthetic_fill":False,"height_cut":False,
        "renderer":{"path":str(args.renderer),"sha256":sha_file(args.renderer)},
        "rasterizer":{"path":str(args.rasterizer),"sha256":sha_file(args.rasterizer)},
        "sources":{key:{"path":str(path),"sha256":expected[key]} for key,path in paths.items()},
        "cameras":metrics,"consumers":consumers,"vertex_RGB_read_directly_from_serialized_accessors":True,
        "evidence_mask_legend":{"grey":"retained visible surface","cyan":"new coverage","orange":"new nearer upper surface"}}
    (args.output_dir/"manifest.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return {"status":report["status"],"output":str(args.output_dir)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage-dir","renderer","rasterizer","output-dir"):
        p.add_argument("--"+flag,type=Path,required=True)
    print(json.dumps(build(p.parse_args()),indent=2))
