#!/usr/bin/env python3
"""Clean a derived closed presentation shell without moving retained corners."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import trimesh

from tools.mapanything_phone_scan.build_closed_room_review_surface import edge_topology
from tools.mapanything_phone_scan import texture_single_room_review_surface as texture
from tools.mapanything_phone_scan.clean_closed_surface_topology import clean_closed_surface_topology


def build(args):
    if args.output_dir.exists():
        raise ValueError("use a new cleaned completion output directory")
    source_path = args.source_dir / "surface_mesh_manifest.json"
    source = texture._json(source_path)
    glb = args.source_dir / "surface_mesh.glb"
    if (source.get("status") != "review_only" or source.get("accepted_for_canonical_use") is not False
            or not source.get("completion") or source["output"]["sha256"] != texture._sha(glb)):
        raise ValueError("cleanup requires a bound explicitly completed review mesh")
    scene = trimesh.load(glb, force="scene", process=False)
    if len(scene.graph.nodes_geometry) != 1:
        raise ValueError("completed cleanup is bounded to one exact geometry")
    node = scene.graph.nodes_geometry[0]
    transform, name = scene.graph[node]
    if not np.allclose(transform, np.eye(4)):
        raise ValueError("cleanup source has an unbound node transform")
    mesh = scene.geometry[name]
    support = texture.completed_geometry_support(source, str(name), len(mesh.faces))
    cleaned = clean_closed_surface_topology(mesh.vertices, mesh.faces, mesh.visual.vertex_colors, support,
                                            minimum_component_faces=args.minimum_component_faces,
                                            max_faces=getattr(args, "max_faces", 3_000_000),
                                            max_vertices=getattr(args, "max_vertices", 2_000_000))
    result = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, vertex_colors=cleaned.colors,
                             process=False, metadata={"review_only": True, "support_class": "completed_with_face_support_sidecar"})
    output_scene = trimesh.Scene(metadata={"review_only": True, "measured_and_inferred_faces_separately_classified": True})
    output_scene.add_geometry(result, geom_name=name, node_name=name)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_glb = args.output_dir / "surface_mesh.glb"
    output_glb.write_bytes(trimesh.exchange.gltf.export_glb(output_scene, include_normals=True))
    support_path = args.output_dir / "face_geometry_support.npz"
    np.savez_compressed(support_path, face_support_class=cleaned.classification,
        face_local_index=np.arange(len(cleaned.faces), dtype=np.uint32), geometry_name=np.asarray(name),
        class_names=np.asarray(["multi_view_supported", "poisson_inferred", "boundary_cap_inferred"]),
        parent_face_local_index=cleaned.retained_face_indices, vertex_parent_index=cleaned.vertex_source_indices)
    direct = trimesh.load(output_glb, force="mesh", process=False)
    if (not np.array_equal(direct.faces, cleaned.faces)
            or not np.array_equal(direct.vertices, cleaned.vertices.astype(np.float32))):
        raise ValueError("GLB serialization changed cleaned local-fan vertex identity or retained corners")
    topology = edge_topology(direct.vertices, direct.faces)
    if (not topology["closed_two_face_edge_incidence"] or topology["inconsistent_winding_edge_count"]
            or topology["degenerate_triangle_count"]):
        raise ValueError("serialized cleaned completion is not closed and consistently wound")
    report = copy.deepcopy(source)
    report["source"]["parent_completed_surface_manifest"] = texture._record(source_path)
    report["source"]["parent_completed_surface_glb"] = texture._record(glb)
    report["method"] += "_whole_closed_component_cleanup_and_local_vertex_fan_split"
    report["parameters"]["minimum_completed_component_faces"] = args.minimum_component_faces
    report["completion"].update(geometry_support=texture._record(support_path), cleanup=cleaned.report,
        original_exported_consumer_topology=source["completion"]["exported_consumer_topology"],
        exported_consumer_topology=topology, retained_corner_positions_changed=False,
        serialized_local_fan_vertex_identities_preserved=True)
    names = ["multi_view_supported", "poisson_inferred", "boundary_cap_inferred"]
    report["completion"]["class_counts"] = {label: int(np.count_nonzero(cleaned.classification == index)) for index, label in enumerate(names)}
    report["completion"]["class_surface_area_m2"] = {label: float(result.area_faces[cleaned.classification == index].sum()) for index, label in enumerate(names)}
    for owner in report["owners"].values():
        owner.update(output_vertex_count=len(direct.vertices), output_triangle_count=len(direct.faces))
    report["output"] = {"glb": str(output_glb), "sha256": texture._sha(output_glb), "size_bytes": output_glb.stat().st_size,
                         "vertex_count": len(direct.vertices), "triangle_count": len(direct.faces)}
    path = args.output_dir / "surface_mesh_manifest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cleanup": cleaned.report, "output": report["output"], "topology": topology}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-component-faces", type=int, default=64)
    parser.add_argument("--max-faces", type=int, default=3_000_000)
    parser.add_argument("--max-vertices", type=int, default=2_000_000)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
