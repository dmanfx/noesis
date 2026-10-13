"""Bounded offline semantic editing and read-only canonical-world context."""
from __future__ import annotations

import hashlib
import itertools
import json
import math

from .contracts.scene_semantics import SceneSemanticMap, SemanticBox


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def revision(value):
    payload = value.model_dump(mode="json") if isinstance(value, SceneSemanticMap) else dict(value)
    payload.pop("revision_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def seal_map(payload):
    result = dict(payload)
    result["revision_sha256"] = "0" * 64
    parsed = SceneSemanticMap.model_validate(result)
    return parsed.model_copy(update={"revision_sha256": revision(parsed)})


def validate_map(payload):
    required = {"contract", "contract_version", "coordinate_frame", "units", "world_frame_revision", "world_transform_sha256",
                "source_artifacts", "prepared_to_backend_rotation", "proposal_revision_sha256", "revision_sha256"}
    if not required.issubset(payload):
        raise ValueError("explicit semantic header/frame/source binding required")
    result = SceneSemanticMap.model_validate(payload)
    if result.revision_sha256 != revision(result):
        raise ValueError("semantic revision digest mismatch")
    return result


def corners(box: SemanticBox):
    r, c, s = box.basis_row_major, box.center_m, box.size_m
    return tuple(tuple(c[i] + sum(r[3*i+j]*sign[j]*s[j]/2 for j in range(3)) for i in range(3))
                 for sign in itertools.product((-1, 1), repeat=3))


def local_coordinates(box, position):
    delta = [position[i] - box.center_m[i] for i in range(3)]
    return tuple(sum(box.basis_row_major[3*i+j]*delta[i] for i in range(3)) for j in range(3))


def ground_distance(box, position):
    local = local_coordinates(box, (position[0], box.center_m[1], position[2]))
    return math.hypot(max(abs(local[0])-box.size_m[0]/2, 0), max(abs(local[1])-box.size_m[1]/2, 0))


def diagnostics(scene):
    """Internal consistency only; thresholds are diagnostics, not survey accuracy."""
    rows = []
    by_id = {e.element_id: e for e in scene.elements}
    for e in scene.elements:
        issues = []
        bottom = e.geometry.center_m[1] - e.geometry.size_m[2]/2
        if e.kind == "object" and e.category in ("sofa", "coffee_table", "carpet", "stool", "floor-standing_lamp"):
            if abs(bottom) > .15:
                issues.append({"code": "floor_contact", "offset_m": bottom, "diagnostic_tolerance_m": .15})
        if e.wall_id:
            wall = by_id[e.wall_id]
            local = local_coordinates(wall.geometry, e.geometry.center_m)
            outside = abs(local[0]) + e.geometry.size_m[0]/2 - wall.geometry.size_m[0]/2
            vertical = abs(local[2]) + e.geometry.size_m[2]/2 - wall.geometry.size_m[2]/2
            alignment = abs(sum(e.geometry.basis_row_major[3*i] * wall.geometry.basis_row_major[3*i] for i in range(3)))
            if abs(local[1]) > .10 or outside > .05 or vertical > .05 or alignment < .99:
                issues.append({"code": "opening_wall_mismatch", "plane_offset_m": abs(local[1]),
                               "segment_overrun_m": max(0, outside), "vertical_overrun_m": max(0, vertical),
                               "length_axis_alignment_abs_dot": alignment,
                               "diagnostic_tolerances_m": {"plane": .10, "extent": .05}})
            if e.review_state == "accepted" and wall.review_state != "accepted":
                issues.append({"code": "accepted_opening_without_accepted_wall"})
        rows.append({"element_id": e.element_id, "review_state": e.review_state, "issues": issues})
    return rows


def export_accepted(scene):
    issues = diagnostics(scene)
    invalid = [r["element_id"] for r in issues if r["review_state"] == "accepted" and r["issues"]]
    if invalid:
        raise ValueError("accepted elements have unresolved geometry diagnostics: " + ", ".join(invalid))
    payload = scene.model_dump(mode="json")
    payload["elements"] = [e for e in payload["elements"] if e["review_state"] == "accepted"]
    return seal_map(payload)


def advisory_context(scene, entity, *, at_us, max_distance_m=1.0):
    """Never mutates entity; evaluated at explicit replay/live time, never wall-clock guesses."""
    unknown = lambda reason: {"status": "unknown", "reason": reason, "entity_id": entity.get("entity_id"),
                              "at_us": at_us, "canonical_position_unchanged": True, "room_id": entity.get("room_id")}
    if (entity.get("world_frame") != scene.coordinate_frame
        or entity.get("world_frame_revision") != scene.world_frame_revision
        or entity.get("world_transform_sha256") != scene.world_transform_sha256):
        return unknown("incompatible_frame_binding")
    if entity.get("lifecycle") != "present" or entity.get("position_state") != "measured":
        return unknown("position_is_not_present_measured")
    if entity.get("conflict"):
        return unknown("conflicting_position_evidence")
    observed, expiry = entity.get("observed_at_us"), entity.get("stale_after_us")
    if not isinstance(at_us, int) or not isinstance(observed, int) or not isinstance(expiry, int) or not 0 < observed <= at_us < expiry:
        return unknown("expired_or_future_measurement")
    if entity.get("position_quantity") != "ground_footprint":
        return unknown("position_is_not_ground_footprint")
    sources = entity.get("sources", [])
    if not any(s.get("accepted") is True and s.get("world_frame") == scene.coordinate_frame
               and s.get("position_state") == "measured" and isinstance(s.get("observed_at_us"), int)
               and observed <= s["observed_at_us"] <= at_us and s.get("world_frame_revision") == scene.world_frame_revision
               and s.get("world_transform_sha256") == scene.world_transform_sha256 for s in sources if isinstance(s, dict)):
        return unknown("missing_accepted_source_binding")
    p = entity.get("position")
    if not isinstance(p, dict) or not all(isinstance(p.get(k), (float, int)) and math.isfinite(p[k]) for k in ("x", "y", "z")):
        return unknown("invalid_position")
    position = (p["x"], p["y"], p["z"])
    distances = sorted((ground_distance(e.geometry, position), e.element_id, e.category, e.label)
                       for e in scene.elements if e.kind == "object" and e.review_state == "accepted")
    nearby = [{"element_id": eid, "category": cat, "label": label, "ground_footprint_distance_m": round(d, 4)}
              for d, eid, cat, label in distances if d <= max_distance_m]
    return {"status": "advisory", "entity_id": entity["entity_id"], "at_us": at_us,
            "canonical_position_unchanged": True, "position": p, "room_id": entity.get("room_id"),
            "position_state": entity["position_state"], "lifecycle_generation": entity.get("subject", {}).get("generation"),
            "continuity_segment_id": entity.get("continuity_segment_id"), "semantic_revision": scene.revision_sha256,
            "distance_threshold_m": max_distance_m, "nearby": nearby,
            "qualification": "provisionally_reviewed_geometry_not_physical_accuracy_or_activity_evidence"}
