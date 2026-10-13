#!/usr/bin/env python3
"""Build an isolated September semantic review package from verified experiment evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from noesis_core.contracts.world import WorldSnapshot
from noesis_core.scene_semantics import advisory_context, corners, diagnostics, export_accepted, seal_map


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(path, role, expected=None):
    path = Path(path).resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"source fingerprint mismatch: {path}")
    return {"role": role, "path": str(path), "sha256": sha}


def read_colored_ply(path):
    types = {"float": "<f4", "double": "<f8", "uchar": "u1"}
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
            if len(header) > 64 or not line:
                raise ValueError("invalid bounded PLY header")
        if header[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("expected binary little endian PLY")
        count = int(next(v.split()[-1] for v in header if v.startswith("element vertex ")))
        if not 0 < count <= 2_000_000 or any(v.startswith("element ") and not v.startswith("element vertex ") for v in header):
            raise ValueError("unsupported PLY elements/count")
        properties = [v.split()[1:] for v in header if v.startswith("property ")]
        if [v[1] for v in properties] != ["x", "y", "z", "red", "green", "blue"]:
            raise ValueError("expected XYZ and real RGB only")
        dtype = np.dtype([(name, types[typ]) for typ, name in properties])
        rows = np.fromfile(f, dtype=dtype, count=count)
        if len(rows) != count or f.read(1):
            raise ValueError("PLY byte count mismatch")
    points = np.column_stack([rows[k] for k in ("x", "y", "z")]).astype(np.float64)
    rgb = np.column_stack([rows[k] for k in ("red", "green", "blue")]).astype(np.uint8)
    if not np.isfinite(points).all():
        raise ValueError("non-finite input cloud")
    return points, rgb


def extract_record(path, output, binding, name):
    with path.open("rb") as f:
        record = None
        for line_no in range(1, 7):
            raw = f.readline()
            if not raw:
                break
            row = json.loads(raw)
            if row.get("type") == "world_snapshot":
                record = row
                break
    if record is None:
        raise ValueError("no standalone snapshot in bounded fixture prefix")
    snapshot = WorldSnapshot.model_validate(record["payload"])
    entity = snapshot.entities[0].model_dump(mode="json")
    if (entity["world_frame_revision"] != binding["world_frame_revision"] or
        entity["world_transform_sha256"] != binding["camera_frame_binding"]["transform_sha256"]):
        raise ValueError("historical fixture is incompatible with source alignment")
    (output / f"{name}.source-record.ndjson").write_bytes(raw)
    return {"name": name, "evidence_kind": "real_recorded_canonical_snapshot", "source_path": str(path),
            "source_line": line_no, "source_record_sha256": hashlib.sha256(raw).hexdigest(),
            "snapshot": snapshot.model_dump(mode="json"), "entity": entity,
            "evaluation_time_us": snapshot.published_at_us}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("experiment-root", "output-root", "dimensions", "floor-rise", "window-record", "hearth-record"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    exp, out = args.experiment_root.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError("Refusing to overwrite review package")
    # Existing deliverables inventory verifies audits, generated predictions and photo.
    inventory = {r["relative_path"]: r for r in json.loads((exp / "deliverables_manifest.json").read_text())["files"]}
    def original(relative, role):
        row = inventory.get(relative)
        if row is None:
            raise ValueError(f"missing original deliverables fingerprint: {relative}")
        return fingerprint(exp / relative, role, row["sha256"])
    sources = [original("evidence/input_audit.json", "experiment_input_audit"),
               original("evidence/room_crop_audit.json", "matched_crop_audit"),
               original("outputs/seed_42/baseline.boxes.json", "original_model_boxes"),
               original("outputs/seed_42/baseline.txt", "original_model_layout"),
               original("evidence/withheld_photo.png", "independent_qualitative_photo")]
    audit = json.loads((exp / "evidence/input_audit.json").read_text())
    crop = json.loads((exp / "evidence/room_crop_audit.json").read_text())
    lane = audit["lanes"]["baseline"]
    for key, role in (("source", "original_reconstruction"), ("world_transform", "phone_alignment")):
        row = lane[key]
        sources.append(fingerprint(row["path"], role, row["sha256"]))
    row = crop["lanes"]["baseline"]["cleaned_ply"]
    sources.append(fingerprint(row["path"], "actual_model_input_colored_ply", row["sha256"]))
    if lane["alignment_applied_count"] != 1 or lane["transform_scale"] != 1 or audit["single_view_points_included"]:
        raise ValueError("unexpected preprocessing/alignment identity")
    alignment = json.loads(Path(lane["world_transform"]["path"]).read_text())
    binding = alignment["target_binding"]
    if (alignment["target_coordinate_frame"] != "backend_world_m_stream_points" or
        binding["world_frame"] != "backend_world_m" or binding["world_frame_revision"] != lane["world_frame_revision"]):
        raise ValueError("source-bound frame alias not established")
    S = np.asarray(audit["backend_y_up_to_right_handed_z_up"], dtype=float)
    # Stored audit field has a descriptive object in earlier format.
    if S.shape != (3, 3):
        raise ValueError("unexpected axis-conversion shape")
    R = np.asarray(audit["common_preprocessing"]["yaw_matrix_row_major"])
    M = S.T @ R.T
    if not np.allclose(M.T @ M, np.eye(3), atol=1e-12) or np.linalg.det(M) < 0.999999:
        raise ValueError("invalid inverse preprocessing rotation")
    prepared, rgb = read_colored_ply(Path(row["path"]))
    world = prepared @ M.T
    roundtrip_error = float(np.abs(world @ (R @ S).T - prepared).max())
    raw_boxes = json.loads((exp / "outputs/seed_42/baseline.boxes.json").read_text())
    # Parse only opening-wall associations, never eval model text.
    links = {}
    for line in (exp / "outputs/seed_42/baseline.txt").read_text().splitlines():
        match = re.match(r"(door|window)_(\d+)=(Door|Window)\(wall_(\d+),", line)
        if match:
            links[(match[1], int(match[2]))] = f"sept.wall.{match[4]}"
    elements, corner_errors = [], []
    for box in raw_boxes:
        kind, identifier = box["class"], box["id"]
        index = identifier if kind == "wall" else identifier - {"door": 1000, "window": 2000}.get(kind, 3000)
        group = kind if kind in ("wall", "door", "window") else "object"
        basis = np.asarray(box["rotation"], dtype=float)
        element = {"element_id": f"sept.{group}.{index}", "kind": "wall" if kind == "wall" else "opening" if kind in ("door", "window") else "object",
                   "category": box["label"], "label": box["label"].replace("_", " ") + " " + str(index),
                   "geometry": {"center_m": (M @ box["center"]).tolist(), "size_m": box["scale"], "basis_row_major": (M @ basis).reshape(-1).tolist()},
                   "wall_id": links.get((kind, index)), "review_state": "proposed"}
        elements.append(element)
    proposal_identity = hashlib.sha256((exp / "outputs/seed_42/baseline.boxes.json").read_bytes()).hexdigest()
    sources.extend([fingerprint(args.dimensions, "independent_tape_dimensions"), fingerprint(args.floor_rise, "independent_floor_rise")])
    scene = seal_map({"map_id": "living-room-september-spatiallm-review", "proposal_revision_sha256": proposal_identity,
                      "world_frame_revision": binding["world_frame_revision"],
                      "world_transform_sha256": binding["camera_frame_binding"]["transform_sha256"],
                      "source_artifacts": sources, "prepared_to_backend_rotation": M.reshape(-1).tolist(),
                      "conversion_note": "Undo shared yaw and (x,-z,y) only; phone-to-world alignment already applied exactly once; metres preserved. Model text/boxes already include min-point translation. No second alignment/offset/rescaling. Source-bound backend_world_m_stream_points alias verified against alignment target_binding.",
                      "elements": elements})
    # Check all eight corners, not only symmetric centers.
    signs = np.array(list(__import__("itertools").product((-1,1), repeat=3)))
    for original_box, e in zip(raw_boxes, scene.elements):
        expected = np.asarray(original_box["center"]) + (signs * np.asarray(original_box["scale"])/2) @ np.asarray(original_box["rotation"]).T
        corner_errors.append(float(np.abs(np.asarray(corners(e.geometry)) @ (R @ S).T - expected).max()))
    out.mkdir(parents=True)
    (out / "evidence").mkdir()
    (out / "drafts").mkdir()
    (out / "exports").mkdir()
    write_json(out / "proposals.json", scene.model_dump(mode="json"))
    sample = scene.model_dump(mode="json")
    accepted = {"sept.object.5", "sept.object.6", "sept.object.10", "sept.object.12", "sept.object.15"}
    audit_edits = []
    for e in sample["elements"]:
        before = copy.deepcopy(e)
        if e["kind"] == "wall" or e["element_id"] in accepted:
            e.update(review_state="accepted", review_actor="Codex experiment review", review_reason="Provisional visual proposal accepted for offline display/advisory demonstration; dense-cloud/withheld-photo plausibility only, physical dimensions remain unqualified.")
        if e["element_id"] == "sept.object.11":
            e.update(review_state="rejected", review_actor="Codex experiment review", review_reason="Duplicate coffee-table proposal at the same location; original experiment measured 3D overlap IoU 0.789. Retain object.10 as one provisional table.")
        if e["element_id"] == "sept.door.0":
            e.update(review_state="rejected", review_actor="Codex experiment review", review_reason="Door width extends beyond its associated wall segment; photograph-to-opening correspondence unresolved. Do not clamp or fabricate a new opening.")
        if e != before:
            audit_edits.append({"element_id": e["element_id"], "before": before, "after": e, "actor": e["review_actor"], "reason": e["review_reason"]})
    reviewed = seal_map(sample)
    write_json(out / "reviewed.json", reviewed.model_dump(mode="json"))
    write_json(out / "initial_review_audit.json", {"source_revision": scene.revision_sha256, "result_revision": reviewed.revision_sha256,
               "decision_scope": "Codex provisional experiment review, not Dave approval or physical qualification", "edits": audit_edits})
    write_json(out / "exports/accepted-map.json", export_accepted(reviewed).model_dump(mode="json"))
    # Deterministic display thinning affects only browser rendering, never model/export geometry.
    indices = np.linspace(0, len(world)-1, min(len(world), 140_000), dtype=int)
    world[indices].astype("<f4").tofile(out / "cloud.xyz.f32")
    rgb[indices].tofile(out / "cloud.rgb.u8")
    world.astype("<f4").tofile(out / "cloud-full.xyz.f32")
    rgb.tofile(out / "cloud-full.rgb.u8")
    shutil.copyfile(exp / "evidence/withheld_photo.png", out / "photo.png")
    shutil.copyfile(args.dimensions, out / "dimensions.json")
    shutil.copyfile(args.floor_rise, out / "floor-rise.json")
    fixtures = [extract_record(args.window_record, out / "evidence", binding, "window"),
                extract_record(args.hearth_record, out / "evidence", binding, "hearth")]
    for fixture in fixtures:
        original_entity = copy.deepcopy(fixture["entity"])
        fixture["context"] = advisory_context(reviewed, fixture["entity"], at_us=fixture["evaluation_time_us"])
        assert fixture["entity"] == original_entity
    write_json(out / "recorded-context.json", fixtures)
    report = {"cloud_points": len(world), "browser_points": len(indices), "coordinate_roundtrip_max_error_m": roundtrip_error,
              "all_box_corners_roundtrip_max_error_m": max(corner_errors), "proposed_elements": len(scene.elements),
              "review_states": {state: sum(e.review_state == state for e in reviewed.elements) for state in ("proposed", "accepted", "rejected")},
              "diagnostics": diagnostics(reviewed), "tape_endpoint_binding": "unresolved; no whole-cloud bounds compared as measured spans",
              "numeric_tape_uncertainty": None, "geometry_rescaled_to_tape": False,
              "floor_contact_diagnostic_tolerance_m": .15, "opening_plane_diagnostic_tolerance_m": .10,
              "opening_extent_diagnostic_tolerance_m": .05, "physical_accuracy_qualified": False,
              "seed_variability": "original experiment one seed only; no repeat inference in this CPU review implementation"}
    write_json(out / "validation.json", report)
    write_json(out / "package.json", {"title": "Living Room / September", "private_local_only": True,
               "coordinate_frame": "backend_world_m", "units": "meters", "up_axis": "+Y",
               "world_frame_revision": scene.world_frame_revision, "display_points": len(indices), "full_points": len(world),
               "model": json.loads((exp / "outcome.json").read_text())["model_id"],
               "seed": 42, "snapshot_context": "Historical canonical positions evaluated at recorded publication time; room_id remains null"})
    print(json.dumps({"output_root": str(out), "report": report}, indent=2))


if __name__ == "__main__":
    main()
