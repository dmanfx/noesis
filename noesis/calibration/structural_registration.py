"""Floor-locked, structural PCF-to-authored-scene registration.

Only measured PCF wall columns and reviewed authored wall/opening identities
participate.  Camera markers, tracked people, and the inherited presentation
matrix are not observations for this solve.  A scale hint merely expresses
authored coordinates in approximately metric units for numerical optimization.

The serialized result freezes its input features.  ``validate_solve_result``
repeats the solve and all gates from those features; it never trusts a stored
matrix, gate flag, or residual in isolation.  The caller is responsible for
binding those features to the exact admitted PCF and authored-OBJ digests.
"""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


CONTRACT = "noesis.calibration.structural_registration_candidate"
CONTRACT_VERSION = 1
MAX_SEGMENTS = 64
MAX_POINTS = 64
MAX_CORRESPONDENCES = 64


class StructuralRegistrationError(ValueError):
    """Malformed or unrepeatable structural registration evidence."""


@dataclass(frozen=True)
class WallSegment:
    segment_id: str
    start_xz: tuple[float, float]
    end_xz: tuple[float, float]
    support_count: int = 0
    vertical_span: float = 0.0
    source_refs: tuple[str, ...] = ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _point2(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise StructuralRegistrationError(f"{label} requires two coordinates")
    if any(isinstance(v, (bool, str, bytes)) for v in value):
        raise StructuralRegistrationError(f"{label} requires numeric coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) and abs(v) < 1_000_000 for v in point):
        raise StructuralRegistrationError(f"{label} has invalid coordinates")
    return point


def _normal(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise StructuralRegistrationError("degenerate wall segment")
    normal = np.array((-direction[1], direction[0])) / length
    if normal[0] < -1e-9 or (abs(normal[0]) <= 1e-9 and normal[1] < 0):
        normal = -normal
    return normal


def _segments(values: Sequence[Any], *, label: str, minimum_length: float) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_SEGMENTS:
        raise StructuralRegistrationError(f"{label} requires 1..{MAX_SEGMENTS} segments")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        segment_id = _field(value, "segment_id")
        if not isinstance(segment_id, str) or not segment_id or len(segment_id) > 160 or segment_id in seen:
            raise StructuralRegistrationError(f"{label}[{index}] has invalid or duplicate id")
        seen.add(segment_id)
        start = _point2(_field(value, "start_xz"), f"{label}[{index}].start_xz")
        end = _point2(_field(value, "end_xz"), f"{label}[{index}].end_xz")
        start_np, end_np = np.asarray(start), np.asarray(end)
        if float(np.linalg.norm(end_np - start_np)) < minimum_length:
            raise StructuralRegistrationError(f"{label}[{index}] is too short")
        normal = _normal(start_np, end_np)
        declared = _field(value, "normal_xz")
        if declared is not None:
            declared_np = np.asarray(_point2(declared, f"{label}[{index}].normal_xz"))
            if abs(abs(float(declared_np @ normal)) - 1.0) > 0.03:
                raise StructuralRegistrationError(f"{label}[{index}] normal disagrees with endpoints")
        raw_count = _field(value, "support_count", 0)
        raw_span = _field(value, "vertical_span", 0.0)
        if label == "authored_segments" and isinstance(raw_span, (tuple, list)) and len(raw_span) == 2:
            lower, upper = (float(v) for v in raw_span)
            raw_span = upper - lower
        if type(raw_count) is not int or isinstance(raw_span, (bool, str, bytes)):
            raise StructuralRegistrationError(f"{label}[{index}] has invalid support types")
        count = raw_count
        span = float(raw_span)
        refs = _field(value, "source_refs", ())
        max_span = 10_000.0 if label == "authored_segments" else 100.0
        if count < 0 or count > 10_000_000 or not math.isfinite(span) or span < 0 or span > max_span:
            raise StructuralRegistrationError(f"{label}[{index}] has invalid support")
        if not isinstance(refs, (tuple, list)) or len(refs) > 32 or any(not isinstance(r, str) or len(r) > 160 for r in refs):
            raise StructuralRegistrationError(f"{label}[{index}] has invalid source refs")
        result.append({
            "segment_id": segment_id, "start_xz": list(start), "end_xz": list(end),
            "support_count": count, "vertical_span": span, "source_refs": list(refs),
        })
    return result


def _points(values: Sequence[Any] | None, *, label: str) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)) or len(values) > MAX_POINTS:
        raise StructuralRegistrationError(f"{label} exceeds {MAX_POINTS} points")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        point_id = _field(value, "point_id")
        kind = _field(value, "kind")
        if not isinstance(point_id, str) or not point_id or len(point_id) > 160 or point_id in seen:
            raise StructuralRegistrationError(f"{label}[{index}] has invalid or duplicate id")
        if kind not in ("corner", "opening", "floor_contact"):
            raise StructuralRegistrationError(f"{label}[{index}] must be a structural corner, opening, or floor contact")
        seen.add(point_id)
        xz = _point2(_field(value, "xz"), f"{label}[{index}].xz")
        refs = _field(value, "source_wall_ids", ())
        evidence_refs = _field(value, "evidence_refs", ())
        if not isinstance(refs, (tuple, list)) or len(refs) > 8 or any(not isinstance(r, str) or len(r) > 160 for r in refs):
            raise StructuralRegistrationError(f"{label}[{index}] has invalid wall refs")
        if not isinstance(evidence_refs, (tuple, list)) or len(evidence_refs) > 16 or any(not isinstance(r, str) or not r or len(r) > 160 for r in evidence_refs):
            raise StructuralRegistrationError(f"{label}[{index}] has invalid evidence refs")
        result.append({"point_id": point_id, "kind": kind, "xz": list(xz),
                       "source_wall_ids": list(refs), "evidence_refs": list(evidence_refs)})
    return result


def _matches(values: Sequence[Any] | None, *, label: str, src_ids: set[str], dst_ids: set[str]) -> list[dict[str, str]]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)) or len(values) > MAX_CORRESPONDENCES:
        raise StructuralRegistrationError(f"{label} exceeds {MAX_CORRESPONDENCES} matches")
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(values):
        src = _field(value, "source_id")
        dst = _field(value, "authored_id")
        role = _field(value, "role")
        if src not in src_ids or dst not in dst_ids or role not in ("fit", "heldout"):
            raise StructuralRegistrationError(f"{label}[{index}] references unknown feature or role")
        if (src, dst) in seen:
            raise StructuralRegistrationError(f"{label}[{index}] is duplicated")
        seen.add((src, dst))
        result.append({"source_id": src, "authored_id": dst, "role": role})
    if len({m["source_id"] for m in result}) != len(result) or len({m["authored_id"] for m in result}) != len(result):
        raise StructuralRegistrationError(f"{label} must be one-to-one")
    return result


def _image_measurement_key(point: Mapping[str, Any]) -> tuple[str, str] | None:
    refs = point["evidence_refs"]
    images = [ref for ref in refs if ref.startswith("image_sha256:")]
    pixels = [ref for ref in refs if ref.startswith("pixel_uv:")]
    return (images[0], pixels[0]) if len(images) == len(pixels) == 1 else None


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, s), (-s, c)), dtype=np.float64)


def _angle_mod_pi(angle: float) -> float:
    return 0.5 * math.atan2(math.sin(2 * angle), math.cos(2 * angle))


def _line_data(segment: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = np.asarray(segment["start_xz"], dtype=np.float64)
    end = np.asarray(segment["end_xz"], dtype=np.float64)
    normal = _normal(start, end)
    return start, end, normal, float(normal @ ((start + end) * 0.5))


def _coplanar_segments(first: Mapping[str, Any], second: Mapping[str, Any], *, distance_limit: float) -> bool:
    _, _, first_normal, first_offset = _line_data(first)
    _, _, second_normal, second_offset = _line_data(second)
    return (abs(float(first_normal @ second_normal)) >= math.cos(math.radians(5))
            and abs(first_offset - second_offset) <= distance_limit)


def _line_overlap_m(src: Mapping[str, Any], dst: Mapping[str, Any], yaw: float, scale: float, translation: np.ndarray) -> float:
    sa, sb = np.asarray(src["start_xz"]), np.asarray(src["end_xz"])
    ta, tb = np.asarray(dst["start_xz"]), np.asarray(dst["end_xz"])
    direction = (tb - ta) / np.linalg.norm(tb - ta)
    projected = np.sort(np.asarray((direction @ (scale * _rotation(yaw) @ sa + translation), direction @ (scale * _rotation(yaw) @ sb + translation))))
    target = np.sort(np.asarray((direction @ ta, direction @ tb)))
    return max(0.0, float(min(projected[1], target[1]) - max(projected[0], target[0])) / scale)


def _candidate(
    source: dict[str, dict[str, Any]], authored: dict[str, dict[str, Any]],
    source_points: dict[str, dict[str, Any]], authored_points: dict[str, dict[str, Any]],
    walls: list[dict[str, str]], points: list[dict[str, str]],
    hint: float, yaw_seed: float,
) -> dict[str, Any]:
    fit_walls = [m for m in walls if m["role"] == "fit"]
    fit_points = [m for m in points if m["role"] == "fit"]
    first = fit_walls[0]
    _, _, _, source_offset = _line_data(source[first["source_id"]])
    _, _, target_normal, target_offset = _line_data(authored[first["authored_id"]])
    sigma0 = 1.0
    normal0 = target_normal
    translation0 = normal0 * (target_offset / hint - sigma0 * source_offset)
    if fit_points:
        point = fit_points[0]
        translation0 = np.asarray(authored_points[point["authored_id"]]["xz"]) / hint - sigma0 * _rotation(yaw_seed) @ np.asarray(source_points[point["source_id"]]["xz"])

    def residual(parameters: np.ndarray) -> np.ndarray:
        yaw, log_sigma, ux, uz = (float(v) for v in parameters)
        sigma = math.exp(log_sigma)
        rotation = _rotation(yaw)
        translation = np.asarray((ux, uz))
        output: list[float] = []
        for match in fit_walls:
            sa, sb, sn, _ = _line_data(source[match["source_id"]])
            _, _, tn, td = _line_data(authored[match["authored_id"]])
            rotated = rotation @ sn
            angle = math.atan2(float(rotated[0] * tn[1] - rotated[1] * tn[0]), float(rotated @ tn))
            output.append(2.0 * _angle_mod_pi(angle))
            output.append(float(tn @ (sigma * rotation @ ((sa + sb) * 0.5) + translation) - td / hint))
        for match in fit_points:
            sp = np.asarray(source_points[match["source_id"]]["xz"])
            tp = np.asarray(authored_points[match["authored_id"]]["xz"]) / hint
            output.extend((sigma * rotation @ sp + translation - tp).tolist())
        return np.asarray(output, dtype=np.float64)

    initial = np.asarray((yaw_seed, math.log(sigma0), *translation0), dtype=np.float64)
    bounds = (
        np.asarray((yaw_seed - math.pi / 3, math.log(0.1), -1000, -1000)),
        np.asarray((yaw_seed + math.pi / 3, math.log(10.0), 1000, 1000)),
    )
    solved = least_squares(residual, initial, bounds=bounds, loss="soft_l1", f_scale=0.08, max_nfev=300)
    yaw, log_sigma, ux, uz = (float(v) for v in solved.x)
    scale = hint * math.exp(log_sigma)
    translation = hint * np.asarray((ux, uz))
    rotation = _rotation(yaw)
    measurements: list[dict[str, Any]] = []
    for match in walls:
        src, dst = source[match["source_id"]], authored[match["authored_id"]]
        sa, sb, sn, _ = _line_data(src)
        _, _, tn, td = _line_data(dst)
        rotated = rotation @ sn
        angle = math.atan2(float(rotated[0] * tn[1] - rotated[1] * tn[0]), float(rotated @ tn))
        perpendicular_m = abs(float(tn @ (scale * rotation @ ((sa + sb) * 0.5) + translation) - td)) / scale
        measurements.append({**match, "kind": "wall", "perpendicular_m": perpendicular_m,
                             "angle_deg": abs(math.degrees(_angle_mod_pi(angle))),
                             "overlap_m": _line_overlap_m(src, dst, yaw, scale, translation)})
    for match in points:
        sp = np.asarray(source_points[match["source_id"]]["xz"])
        tp = np.asarray(authored_points[match["authored_id"]]["xz"])
        error = float(np.linalg.norm(scale * rotation @ sp + translation - tp) / scale)
        measurements.append({**match, "kind": "point", "error_m": error})
    objective = float(np.linalg.norm(residual(solved.x)))
    validation_terms: list[float] = []
    for measurement in measurements:
        if measurement["role"] != "heldout":
            continue
        if measurement["kind"] == "wall":
            validation_terms.extend((measurement["perpendicular_m"],
                                     2.0 * math.radians(measurement["angle_deg"]),
                                     max(0.0, 0.25 - measurement["overlap_m"])))
        else:
            validation_terms.append(measurement["error_m"])
    validation_m = float(np.linalg.norm(validation_terms))
    # Orientation is selected from fitted evidence only. A heldout observation
    # must never choose between yaw hypotheses before it is used as a gate.
    fit_overlap_deficit = float(np.linalg.norm([
        max(0.0, 0.30 - measurement["overlap_m"])
        for measurement in measurements
        if measurement["role"] == "fit" and measurement["kind"] == "wall"
    ]))
    return {"yaw_rad": yaw, "scene_per_m": scale, "translation_scene_xz": translation.tolist(),
            "measurements": measurements, "objective_m": objective,
            "validation_m": validation_m, "selection_score_m": objective + fit_overlap_deficit,
            "optimizer_success": bool(solved.success)}


def solve_structural_registration(
    source_segments: Sequence[Any], authored_segments: Sequence[Any], *,
    correspondences: Sequence[Any] | None = None,
    source_points: Sequence[Any] | None = None, authored_points: Sequence[Any] | None = None,
    point_correspondences: Sequence[Any] | None = None,
    scene_per_m_hint: float,
) -> dict[str, Any]:
    """Solve a reviewed gravity-locked Sim(2), or return a gated rejection.

    Fit walls and independently measured structural contacts must span scale
    and both translations. At least one separate wall or contact is held out.
    Camera marker and people positions cannot fill missing geometric rank.
    """
    hint = float(scene_per_m_hint)
    if not math.isfinite(hint) or not 1.0 <= hint <= 10_000.0:
        raise StructuralRegistrationError("scene_per_m_hint is invalid")
    src_list = _segments(source_segments, label="source_segments", minimum_length=0.4)
    dst_list = _segments(authored_segments, label="authored_segments", minimum_length=0.4 * hint)
    sp_list = _points(source_points, label="source_points")
    tp_list = _points(authored_points, label="authored_points")
    wall_matches = _matches(correspondences, label="correspondences",
                            src_ids={s["segment_id"] for s in src_list}, dst_ids={s["segment_id"] for s in dst_list})
    point_matches = _matches(point_correspondences, label="point_correspondences",
                             src_ids={p["point_id"] for p in sp_list}, dst_ids={p["point_id"] for p in tp_list})
    inputs = {"source_segments": src_list, "authored_segments": dst_list,
              "correspondences": wall_matches, "source_points": sp_list,
              "authored_points": tp_list, "point_correspondences": point_matches,
              "scene_per_m_hint": hint}
    fit = [m for m in wall_matches if m["role"] == "fit"]
    heldout = [m for m in (*wall_matches, *point_matches) if m["role"] == "heldout"]
    reasons: list[str] = []
    fit_point_matches = [m for m in point_matches if m["role"] == "fit"]
    if len(fit) < 2:
        reasons.append("insufficient_independent_fit_walls")
    if len(fit) + len(fit_point_matches) < 3:
        reasons.append("insufficient_fit_structure")
    if not heldout:
        reasons.append("missing_independent_holdout")
    fit_source_ids = {m["source_id"] for m in fit}
    source_by_id = {s["segment_id"]: s for s in src_list}
    authored_by_id = {s["segment_id"]: s for s in dst_list}
    source_point_by_id = {p["point_id"]: p for p in sp_list}
    authored_point_by_id = {p["point_id"]: p for p in tp_list}
    for index, left in enumerate(point_matches):
        left_source = source_point_by_id[left["source_id"]]
        left_authored = authored_point_by_id[left["authored_id"]]
        for right in point_matches[index + 1:]:
            right_source = source_point_by_id[right["source_id"]]
            right_authored = authored_point_by_id[right["authored_id"]]
            same_source_location_and_proof = (
                set(left_source["evidence_refs"]) == set(right_source["evidence_refs"])
                and np.linalg.norm(np.asarray(left_source["xz"]) - right_source["xz"]) <= 1e-6
            )
            same_image_pixel = (_image_measurement_key(left_source) is not None
                                and _image_measurement_key(left_source) == _image_measurement_key(right_source))
            if same_source_location_and_proof or same_image_pixel:
                reasons.append("duplicate_source_point_measurement")
            if np.linalg.norm(np.asarray(left_authored["xz"]) - right_authored["xz"]) <= 1e-6 * hint:
                reasons.append("duplicate_authored_point_measurement")
    for match in wall_matches:
        source_feature = source_by_id[match["source_id"]]
        if source_feature["support_count"] < 20 or source_feature["vertical_span"] < 1.0:
            reasons.append("weak_source_wall_support")
        if match["role"] == "heldout" and any(
            _coplanar_segments(source_feature, source_by_id[fit_match["source_id"]], distance_limit=0.15)
            or _coplanar_segments(authored_by_id[match["authored_id"]], authored_by_id[fit_match["authored_id"]], distance_limit=0.15 * hint)
            for fit_match in fit
        ):
            reasons.append("heldout_wall_not_independent_plane")
    for match in point_matches:
        source_feature = next(p for p in sp_list if p["point_id"] == match["source_id"])
        if (source_feature["kind"] == "corner" and match["role"] == "heldout"
                and source_feature["source_wall_ids"]
                and not (set(source_feature["source_wall_ids"]) - fit_source_ids)):
            reasons.append("heldout_corner_derived_from_fit_walls")
        if source_feature["kind"] in ("opening", "floor_contact") and not source_feature["evidence_refs"]:
            reasons.append("structural_point_lacks_independent_evidence")
        if source_feature["kind"] == "corner" and not (source_feature["source_wall_ids"] or source_feature["evidence_refs"]):
            reasons.append("structural_point_lacks_independent_evidence")
    if reasons:
        return {"contract": CONTRACT, "contract_version": CONTRACT_VERSION, "inputs": inputs,
                "solver_status": "rejected", "reason_codes": reasons, "solution": None}
    src = {s["segment_id"]: s for s in src_list}
    dst = {s["segment_id"]: s for s in dst_list}
    sp = {p["point_id"]: p for p in sp_list}
    tp = {p["point_id"]: p for p in tp_list}
    # Line offsets plus independently measured structural contacts must
    # determine scale and two translations.  Yaw is constrained by at least
    # two wall normals and then solved jointly with those metric constraints.
    angle_seeds = []
    for match in fit:
        _, _, sn, _ = _line_data(src[match["source_id"]])
        _, _, tn, _ = _line_data(dst[match["authored_id"]])
        angle_seeds.append(-math.atan2(float(sn[0] * tn[1] - sn[1] * tn[0]), float(sn @ tn)))
    seed = 0.5 * math.atan2(sum(math.sin(2 * a) for a in angle_seeds),
                            sum(math.cos(2 * a) for a in angle_seeds))
    rows = []
    rotation_seed = _rotation(seed)
    # Scale and translation are reparameterized about the observed geometry,
    # not the arbitrary PCF world origin.  Otherwise a valid remote room can
    # appear singular merely because its coordinates are large: subtracting
    # the center is a column operation and preserves the actual rank.
    fit_locations = [
        (np.asarray(src[match["source_id"]]["start_xz"])
         + np.asarray(src[match["source_id"]]["end_xz"])) * 0.5
        for match in fit
    ] + [np.asarray(sp[match["source_id"]]["xz"]) for match in fit_point_matches]
    source_center = np.mean(fit_locations, axis=0)
    for match in fit:
        sa, sb, _, _ = _line_data(src[match["source_id"]])
        _, _, tn, _ = _line_data(dst[match["authored_id"]])
        rows.append((float(tn @ (rotation_seed @ ((sa + sb) * 0.5 - source_center))),
                     float(tn[0]), float(tn[1])))
    for match in fit_point_matches:
        rotated = rotation_seed @ (np.asarray(sp[match["source_id"]]["xz"]) - source_center)
        rows.extend(((float(rotated[0]), 1.0, 0.0), (float(rotated[1]), 0.0, 1.0)))
    singular = np.linalg.svd(np.asarray(rows), compute_uv=False)
    rank_ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    if rank_ratio < 0.015:
        reasons.append("structural_geometry_rank_deficient")
    if reasons:
        return {"contract": CONTRACT, "contract_version": CONTRACT_VERSION, "inputs": inputs,
                "solver_status": "rejected", "reason_codes": reasons, "solution": None}
    candidates = [_candidate(src, dst, sp, tp, wall_matches, point_matches, hint, yaw)
                  for yaw in (seed, seed + math.pi)]
    candidates.sort(key=lambda candidate: candidate["selection_score_m"])
    best, alternate = candidates
    fit_measurements = [m for m in best["measurements"] if m["role"] == "fit"]
    hold_measurements = [m for m in best["measurements"] if m["role"] == "heldout"]
    for measurement in fit_measurements:
        if measurement["kind"] == "wall" and (measurement["perpendicular_m"] > 0.15 or measurement["angle_deg"] > 6.0 or measurement["overlap_m"] < 0.30):
            reasons.append("fit_wall_residual")
        if measurement["kind"] == "point" and measurement["error_m"] > 0.15:
            reasons.append("fit_point_residual")
    for measurement in hold_measurements:
        if measurement["kind"] == "wall" and (measurement["perpendicular_m"] > 0.20 or measurement["angle_deg"] > 8.0 or measurement["overlap_m"] < 0.25):
            reasons.append("heldout_wall_residual")
        if measurement["kind"] == "point" and measurement["error_m"] > 0.20:
            reasons.append("heldout_point_residual")
    # The ambiguity decision also uses fit-only evidence. A heldout point or
    # wall may reject the selected orientation, but may not rescue a fit tie.
    if alternate["selection_score_m"] <= best["selection_score_m"] + 0.15:
        reasons.append("orientation_ambiguous")
    if not best["optimizer_success"]:
        reasons.append("optimizer_failed")
    scale, yaw = best["scene_per_m"], best["yaw_rad"]
    tx, tz = best["translation_scene_xz"]
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.asarray(((scale*c, 0.0, scale*s, tx),
                         (0.0, scale, 0.0, 0.0),
                         (-scale*s, 0.0, scale*c, tz),
                         (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)
    solution = {"matrix_col_major": matrix.flatten(order="F").tolist(),
                "scene_per_m": scale, "yaw_rad": yaw,
                "translation_scene_xz": [tx, tz], "wall_rank_ratio": rank_ratio,
                "measurements": best["measurements"],
                "objective_m": best["objective_m"], "validation_m": best["validation_m"],
                "selection_score_m": best["selection_score_m"],
                "alternate_selection_score_m": alternate["selection_score_m"]}
    return {"contract": CONTRACT, "contract_version": CONTRACT_VERSION, "inputs": inputs,
            "solver_status": "rejected" if reasons else "accepted",
            "reason_codes": sorted(set(reasons)), "solution": solution}


def _compare_replay(stored: Any, replayed: Any, *, path: str = "result") -> None:
    if type(stored) is not type(replayed):
        raise StructuralRegistrationError(f"{path} differs from structural replay")
    if isinstance(stored, dict):
        if set(stored) != set(replayed):
            raise StructuralRegistrationError(f"{path} has missing or unknown fields")
        for key in stored:
            _compare_replay(stored[key], replayed[key], path=f"{path}.{key}")
    elif isinstance(stored, list):
        if len(stored) != len(replayed):
            raise StructuralRegistrationError(f"{path} length differs from replay")
        for index, (left, right) in enumerate(zip(stored, replayed)):
            _compare_replay(left, right, path=f"{path}[{index}]")
    elif isinstance(stored, float):
        if not math.isfinite(stored) or not math.isclose(stored, replayed, abs_tol=1e-7, rel_tol=1e-9):
            raise StructuralRegistrationError(f"{path} differs from structural replay")
    elif stored != replayed:
        raise StructuralRegistrationError(f"{path} differs from structural replay")


def validate_solve_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-solve frozen features and reject forged or degraded acceptance."""
    if not isinstance(payload, Mapping) or set(payload) != {"contract", "contract_version", "inputs", "solver_status", "reason_codes", "solution"}:
        raise StructuralRegistrationError("invalid structural result fields")
    if payload["contract"] != CONTRACT or payload["contract_version"] != CONTRACT_VERSION:
        raise StructuralRegistrationError("unsupported structural result contract")
    inputs = payload["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != {"source_segments", "authored_segments", "correspondences", "source_points", "authored_points", "point_correspondences", "scene_per_m_hint"}:
        raise StructuralRegistrationError("invalid frozen structural inputs")
    replayed = solve_structural_registration(**inputs)
    _compare_replay(dict(payload), replayed)
    return replayed


def extract_pcf_wall_segments(points_m: Any, *, grid_m: float = 0.05) -> list[WallSegment]:
    """Extract vertical wall-column line candidates from exact admitted PCF points.

    This is a bounded artifact-build operation, never a live per-frame path.
    The compact output is review evidence, not automatically matched identity.
    """
    import cv2

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) > 2_000_000 or not np.isfinite(points).all():
        raise StructuralRegistrationError("PCF points require a finite bounded Nx3 cloud")
    if not 0.03 <= grid_m <= 0.10:
        raise StructuralRegistrationError("wall grid must be 3–10 cm")
    points = points[(points[:, 1] >= 0.20) & (points[:, 1] <= 2.40)]
    if len(points) < 500:
        return []
    cells = np.floor(points[:, (0, 2)] / grid_m).astype(np.int32)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    ymin = np.full(len(unique), np.inf)
    ymax = np.full(len(unique), -np.inf)
    np.minimum.at(ymin, inverse, points[:, 1])
    np.maximum.at(ymax, inverse, points[:, 1])
    vertical_mask = (counts >= 8) & ((ymax - ymin) >= 1.20)
    occupied = unique[vertical_mask]
    occupied_spans = (ymax - ymin)[vertical_mask]
    if len(occupied) < 25:
        return []
    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0)
    shape = upper - lower + 1
    if int(shape[0] * shape[1]) > 2_000_000:
        raise StructuralRegistrationError("PCF wall raster exceeds bounded area")
    image = np.zeros((int(shape[1]), int(shape[0])), dtype=np.uint8)
    image[occupied[:, 1] - lower[1], occupied[:, 0] - lower[0]] = 255
    hough = cv2.HoughLinesP(image, 1, np.pi / 360, threshold=25,
                            minLineLength=max(12, round(0.8 / grid_m)),
                            maxLineGap=max(3, round(0.3 / grid_m)))
    if hough is None:
        return []
    centers = (occupied.astype(np.float64) + 0.5) * grid_m
    candidates: list[tuple[float, WallSegment]] = []
    for raw in hough[:, 0, :]:
        p0 = (np.asarray((raw[0], raw[1]), dtype=np.float64) + lower + 0.5) * grid_m
        p1 = (np.asarray((raw[2], raw[3]), dtype=np.float64) + lower + 0.5) * grid_m
        vector = p1 - p0
        length = float(np.linalg.norm(vector))
        if length < 0.8:
            continue
        tangent = vector / length
        normal = np.asarray((-tangent[1], tangent[0]))
        relative = centers - p0
        projection = relative @ tangent
        included = (np.abs(relative @ normal) <= 0.11) & (projection >= -0.15) & (projection <= length + 0.15)
        support = centers[included]
        if len(support) < 20:
            continue
        center = np.mean(support, axis=0)
        _, _, basis = np.linalg.svd(support - center, full_matrices=False)
        tangent = basis[0]
        projected = (support - center) @ tangent
        lo, hi = np.percentile(projected, (2, 98))
        if hi - lo < 0.8:
            continue
        start, end = center + lo * tangent, center + hi * tangent
        measured_span = float(np.median(occupied_spans[included]))
        segment = WallSegment(f"pcf_wall_{len(candidates):02d}", tuple(start), tuple(end),
                              len(support), measured_span,
                              ("pcf:points.glb", "method:vertical_columns", f"grid_m:{grid_m:.3f}"))
        candidates.append((float(hi - lo) * math.sqrt(len(support)), segment))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    selected: list[WallSegment] = []
    for _, candidate in candidates:
        ca, cb, cn, cd = _line_data({"start_xz": candidate.start_xz, "end_xz": candidate.end_xz})
        duplicate = False
        for prior in selected:
            pa, pb, pn, pd = _line_data({"start_xz": prior.start_xz, "end_xz": prior.end_xz})
            if abs(float(cn @ pn)) < math.cos(math.radians(5)) or abs(cd - pd) > 0.15:
                continue
            direction = (pb - pa) / np.linalg.norm(pb - pa)
            a = np.sort(np.asarray((direction @ ca, direction @ cb)))
            b = np.sort(np.asarray((direction @ pa, direction @ pb)))
            overlap = max(0.0, float(min(a[1], b[1]) - max(a[0], b[0])))
            if overlap >= 0.5 * min(float(np.linalg.norm(cb - ca)), float(np.linalg.norm(pb - pa))):
                duplicate = True
                break
        if not duplicate:
            selected.append(WallSegment(f"pcf_wall_{len(selected):02d}", candidate.start_xz,
                                        candidate.end_xz, candidate.support_count,
                                        candidate.vertical_span, candidate.source_refs))
        if len(selected) >= MAX_SEGMENTS:
            break
    return selected


def read_pcf_points_glb(path: str | Path) -> np.ndarray:
    """Read a bounded PCF point GLB for offline deterministic extraction.

    The caller must verify the admitted manifest/path digest.  This routine
    does not infer any world-frame authority from the GLB alone.
    """
    import trimesh

    file = Path(path)
    if not file.is_file() or file.stat().st_size > 128 * 1024 * 1024:
        raise StructuralRegistrationError("PCF point GLB is missing or oversized")
    scene = trimesh.load(file, force="scene", process=False)
    clouds: list[np.ndarray] = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        if not np.allclose(transform, np.eye(4), atol=1e-9, rtol=0.0):
            raise StructuralRegistrationError("PCF GLB has unsupported non-identity point transform")
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.points.PointCloud):
            raise StructuralRegistrationError("PCF GLB contains non-point geometry")
        clouds.append(np.asarray(geometry.vertices, dtype=np.float64))
    if not clouds:
        raise StructuralRegistrationError("PCF GLB contains no points")
    points = np.concatenate(clouds)
    if len(points) > 2_000_000 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise StructuralRegistrationError("PCF GLB has invalid point count or coordinates")
    return points


def extract_pcf_wall_planes(
    points_m: Any, *, bounds_xz_m: tuple[float, float, float, float] | None = None,
    voxel_m: float = 0.06,
) -> list[WallSegment]:
    """Extract measured near-vertical PCF planes, including sparse walls.

    The 3-D plane is intersected with the admitted Y=0 floor.  Unlike an XZ
    projection of upper wall points, this retains measured tilt and does not
    silently turn a sloping wall into an upright one.  Horizontal RANSAC
    components are discarded, never treated as structural wall matches.
    """
    import open3d as o3d

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) > 2_000_000 or not np.isfinite(points).all():
        raise StructuralRegistrationError("PCF points require a finite bounded Nx3 cloud")
    if not 0.04 <= voxel_m <= 0.10:
        raise StructuralRegistrationError("plane voxel must be 4–10 cm")
    selected = (points[:, 1] >= 0.25) & (points[:, 1] <= 2.35)
    if bounds_xz_m is not None:
        if not isinstance(bounds_xz_m, (tuple, list)) or len(bounds_xz_m) != 4:
            raise StructuralRegistrationError("bounds_xz_m requires xmin,xmax,zmin,zmax")
        xmin, xmax, zmin, zmax = (float(value) for value in bounds_xz_m)
        if not all(math.isfinite(v) for v in (xmin, xmax, zmin, zmax)) or not (xmin < xmax and zmin < zmax):
            raise StructuralRegistrationError("invalid PCF wall bounds")
        selected &= (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
        selected &= (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
        bounds_ref = "bounds_xz_m:" + ",".join(f"{value:.6f}" for value in (xmin, xmax, zmin, zmax))
    else:
        bounds_ref = "bounds_xz_m:none"
    points = points[selected]
    if len(points) < 500:
        return []
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel_m)
    if len(cloud.points) < 300:
        return []
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel_m, max_nn=30))
    vertical_indices = np.flatnonzero(np.abs(np.asarray(cloud.normals)[:, 1]) < 0.40)
    cloud = cloud.select_by_index(vertical_indices.tolist())
    o3d.utility.random.seed(17)
    candidates: list[WallSegment] = []
    for _ in range(36):
        if len(cloud.points) < 200:
            break
        plane, inliers = cloud.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=350, probability=0.995,
        )
        support = np.asarray(cloud.points)[inliers]
        normal3 = np.asarray(plane[:3], dtype=np.float64)
        normal3 /= np.linalg.norm(normal3)
        cloud = cloud.select_by_index(inliers, invert=True)
        if abs(float(normal3[1])) > 0.30 or len(support) < 200:
            continue
        y_span = float(np.ptp(support[:, 1]))
        horizontal = np.asarray((normal3[0], normal3[2]))
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= 0.95 or y_span < 1.20:
            continue
        horizontal /= horizontal_norm
        offset = -float(plane[3]) / horizontal_norm
        plane_offset = float(plane[3])
        if horizontal[0] < -1e-9 or (abs(horizontal[0]) <= 1e-9 and horizontal[1] < 0):
            horizontal = -horizontal
            offset = -offset
            normal3 = -normal3
            plane_offset = -plane_offset
        tangent = np.asarray((-horizontal[1], horizontal[0]))
        projected = support[:, (0, 2)] @ tangent
        low, high = np.percentile(projected, (2, 98))
        if high - low < 0.80:
            continue
        start = offset * horizontal + low * tangent
        end = offset * horizontal + high * tangent
        sorted_support = np.round(support, 6)
        sorted_support = sorted_support[np.lexsort((sorted_support[:, 2], sorted_support[:, 1], sorted_support[:, 0]))]
        support_sha256 = hashlib.sha256(sorted_support.astype("<f8").tobytes()).hexdigest()
        candidates.append(WallSegment(
            f"pcf_plane_{len(candidates):02d}", tuple(start), tuple(end),
            len(support), y_span,
            ("pcf:points.glb", "method:vertical_plane_ransac", f"voxel_m:{voxel_m:.3f}",
             bounds_ref, "normal_xyz:" + ",".join(f"{value:.9f}" for value in normal3),
             f"plane_d_m:{plane_offset:.9f}", f"support_sha256:{support_sha256}"),
        ))
    candidates.sort(key=lambda item: (-(item.support_count * np.linalg.norm(np.asarray(item.end_xz) - item.start_xz)), item.start_xz))
    return [WallSegment(f"pcf_plane_{index:02d}", item.start_xz, item.end_xz,
                        item.support_count, item.vertical_span, item.source_refs)
            for index, item in enumerate(candidates[:MAX_SEGMENTS])]


def verify_pcf_source_segments(
    points_glb_path: str | Path, frozen_segments: Sequence[Any],
) -> list[dict[str, Any]]:
    """Offline proof that every frozen source wall re-extracts from this GLB.

    The caller independently checks the GLB SHA against its admitted manifest.
    This intentionally reruns bounded Hough/RANSAC only during artifact build,
    never on live presentation loading or a frame callback.
    """
    declared = _segments(frozen_segments, label="source_segments", minimum_length=0.4)
    points = read_pcf_points_glb(points_glb_path)
    extraction_cache: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for segment in declared:
        refs = tuple(segment["source_refs"])
        metadata = dict(ref.split(":", 1) for ref in refs if ":" in ref)
        if metadata.get("pcf") != "points.glb":
            raise StructuralRegistrationError("source wall does not declare PCF points.glb")
        method = metadata.get("method")
        if method == "vertical_columns":
            if set(metadata) != {"pcf", "method", "grid_m"}:
                raise StructuralRegistrationError("invalid vertical-column extraction refs")
            config = (method, metadata["grid_m"])
            if config not in extraction_cache:
                grid = float(metadata["grid_m"])
                extracted = extract_pcf_wall_segments(points, grid_m=grid)
                extraction_cache[config] = {item["segment_id"]: item for item in
                                            _segments(extracted, label="source_segments", minimum_length=0.4)}
        elif method == "vertical_plane_ransac":
            if set(metadata) != {"pcf", "method", "voxel_m", "bounds_xz_m", "normal_xyz", "plane_d_m", "support_sha256"}:
                raise StructuralRegistrationError("invalid vertical-plane extraction refs")
            bounds_text = metadata["bounds_xz_m"]
            bounds = None if bounds_text == "none" else tuple(float(value) for value in bounds_text.split(","))
            config = (method, metadata["voxel_m"], bounds_text)
            if config not in extraction_cache:
                voxel = float(metadata["voxel_m"])
                extracted = extract_pcf_wall_planes(points, bounds_xz_m=bounds, voxel_m=voxel)
                extraction_cache[config] = {item["segment_id"]: item for item in
                                            _segments(extracted, label="source_segments", minimum_length=0.4)}
        else:
            raise StructuralRegistrationError("unsupported source wall extraction method")
        matched = extraction_cache[config].get(segment["segment_id"])
        if matched is None:
            raise StructuralRegistrationError("frozen source wall id does not re-extract")
        _compare_replay(segment, matched, path=f"source wall {segment['segment_id']}")
    return declared


def verify_authored_segments(
    extracted_segments: Sequence[Any], frozen_segments: Sequence[Any], *,
    scene_per_m_hint: float,
) -> list[dict[str, Any]]:
    """Offline proof that selected OBJ walls match a fresh authored extraction.

    The caller binds ``extracted_segments`` to the exact current authored OBJ
    and selected floor groups. This comparison is intentionally independent of
    the optimizer's math replay.
    """
    hint = float(scene_per_m_hint)
    if not math.isfinite(hint) or not 1.0 <= hint <= 10_000.0:
        raise StructuralRegistrationError("scene_per_m_hint is invalid")
    declared = _segments(frozen_segments, label="authored_segments", minimum_length=0.4 * hint)
    selected_ids = {item["segment_id"] for item in declared}
    selected_extracted = [item for item in extracted_segments if _field(item, "segment_id") in selected_ids]
    if len(selected_extracted) != len(selected_ids):
        raise StructuralRegistrationError("selected authored walls are absent or duplicated")
    extracted = {item["segment_id"]: item for item in
                 _segments(selected_extracted, label="authored_segments", minimum_length=0.4 * hint)}
    for segment in declared:
        matched = extracted.get(segment["segment_id"])
        if matched is None:
            raise StructuralRegistrationError("frozen authored wall id does not re-extract")
        _compare_replay(segment, matched, path=f"authored wall {segment['segment_id']}")
    return declared
