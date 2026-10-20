#!/usr/bin/env python3
"""Render original RGB-D frames with depth-qualified ceiling evidence dots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .build_high_detail_pcf_hybrid import sha_file

PALETTE = [(255,50,220),(50,255,255),(255,190,20),(100,255,80),(170,100,255),(255,80,80)]


def visible_pixels(points,raw,pose):
    inverse = np.linalg.inv(pose)
    camera = points @ inverse[:3,:3].T + inverse[:3,3]
    z = camera[:,2]
    projected = camera @ np.asarray(raw["intrinsics"]).T
    u = np.rint(projected[:,0]/np.where(z>0,z,1)).astype(int)
    v = np.rint(projected[:,1]/np.where(z>0,z,1)).astype(int)
    h,w = raw["depth_z"].shape
    inside = (z>.05)&(z<12)&(u>=0)&(u<w)&(v>=0)&(v<h)
    rows = np.flatnonzero(inside)
    depth = raw["depth_z"][v[rows],u[rows]]
    valid = (raw["mask"][v[rows],u[rows]]&~raw["cross_model_uncertain"][v[rows],u[rows]]
             &(raw["confidence"][v[rows],u[rows]]>=.35)
             &(np.abs(depth-z[rows])<=.05+.03*np.maximum(depth,z[rows])))
    rows = rows[valid]
    return np.column_stack((u[rows],v[rows]))


def build(args):
    classification = json.loads((args.classification_dir/"ceiling_classification.json").read_text())
    source = json.loads((args.classification_dir/"source_bindings.json").read_text())["source_surfels"]
    if sha_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("classification source changed")
    with np.load(source["path"],allow_pickle=False) as raw:
        points,owners = raw["points"].copy(),raw["owner_room_id"].copy()
    with np.load(args.classification_dir/"ceiling_source_indices.npz",allow_pickle=False) as masks:
        removed,ambiguous = masks["ceiling_removed"].copy(),masks["ambiguous_retained"].copy()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    records = []
    for owner,index in ((1,29),(1,42),(2,16),(2,27),(3,3),(3,39)):
        report = classification[str(owner)]
        row = report["source_view_witnesses"][index]
        path = Path(row["raw_path"])
        if sha_file(path) != row["raw_sha256"]:
            raise ValueError("source RGB-D changed")
        with np.load(path,allow_pickle=False) as archive:
            raw = {key:archive[key] for key in ("depth_z","intrinsics","mask","confidence","cross_model_uncertain","model_rgb")}
        rgb = np.asarray(raw["model_rgb"])
        if rgb.max()<=1.5:
            rgb = rgb*255
        rgb = np.uint8(np.clip(np.rint(rgb),0,255))
        original = Image.fromarray(rgb).resize((1008,560))
        candidates,final = original.copy(),original.copy()
        draws = ImageDraw.Draw(candidates),ImageDraw.Draw(final)
        with np.load(args.classification_dir/f"owner_{owner}_ceiling_candidates.npz",allow_pickle=False) as archive:
            for patch in report["patches"]:
                patch_id = patch["patch_id"]
                seeds = archive[f"patch_{patch_id}_seeds"]
                pixels = visible_pixels(seeds,raw,np.asarray(row["camera_to_assembly_row_major"]))
                for u,v in pixels:
                    draws[0].ellipse((2*u-2,2*v-2,2*u+2,2*v+2),fill=PALETTE[patch_id%len(PALETTE)])
        for mask,color in ((removed,(255,40,40)),(ambiguous,(255,210,30))):
            pixels = visible_pixels(points[mask&(owners==owner)],raw,np.asarray(row["camera_to_assembly_row_major"]))
            for u,v in np.unique(pixels,axis=0):
                draws[1].point((2*u,2*v),fill=color)
        sheet = Image.new("RGB",(3024,620),(30,30,30))
        for i,image in enumerate((original,candidates,final)):
            sheet.paste(image,(i*1008,30))
        text = ImageDraw.Draw(sheet)
        text.text((10,6),f"Owner {owner}, source view {index}: ORIGINAL",fill="white")
        text.text((1018,6),"QUALIFIED PLANE SEEDS, colors = patch IDs",fill="white")
        text.text((2026,6),"REMOVAL red / AMBIGUOUS RETAINED yellow",fill="white")
        text.text((1018,594),"Patch IDs: "+", ".join(str(p["patch_id"]) for p in report["patches"]),fill="white")
        text.text((10,594),"Original camera/geometry; dots need source depth/mask/confidence. No inferred fill.",fill="white")
        name = f"owner_{owner}_view_{index:02d}.jpg"
        sheet.save(args.output_dir/name,quality=94)
        records.append({"owner":owner,"view":index,"image":name,"raw":str(path),"raw_sha256":row["raw_sha256"]})
    (args.output_dir/"manifest.json").write_text(json.dumps({"classification_sha256":sha_file(args.classification_dir/"ceiling_classification.json"),
        "source_sha256":source["sha256"],"images":records,"synthetic_fill":False},indent=2)+"\n")
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classification-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    print(json.dumps(build(p.parse_args()),indent=2))
