"""Focused structural PCF/authored registration contract tests."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from noesis.calibration.structural_registration import (
    StructuralRegistrationError,
    extract_pcf_wall_planes,
    read_pcf_points_glb,
    solve_structural_registration,
    validate_solve_result,
    verify_authored_segments,
    verify_pcf_source_segments,
)


def _fixture(*, opening_error_m: float = 0.0) -> dict:
    scale, yaw = 92.0, math.radians(17)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, s), (-s, c)))
    translation = np.asarray((130.0, -45.0))

    def scene(point):
        return (scale * rotation @ np.asarray(point) + translation).tolist()

    source_geometry = [
        ("west", (0.0, 0.0), (0.0, 3.0)),
        ("east", (4.0, 0.0), (4.0, 3.0)),
        ("front", (0.0, 0.0), (4.0, 0.0)),
        ("rear", (0.0, 3.0), (4.0, 3.0)),
    ]
    source = [{"segment_id": name, "start_xz": list(a), "end_xz": list(b),
               "support_count": 100, "vertical_span": 1.6} for name, a, b in source_geometry]
    authored = [{"segment_id": f"obj_{name}", "start_xz": scene(a), "end_xz": scene(b)}
                for name, a, b in source_geometry]
    walls = [{"source_id": name, "authored_id": f"obj_{name}", "role": "fit"}
             for name in ("west", "east", "front")]
    opening = scene((0.0, 1.4))
    opening[0] += opening_error_m * scale
    return {
        "source_segments": source,
        "authored_segments": authored,
        "correspondences": walls,
        "source_points": [{"point_id": "observed_window_jamb", "kind": "opening", "xz": [0.0, 1.4],
                           "source_wall_ids": ["west"], "evidence_refs": ["independent_color_edge_17"]}],
        "authored_points": [{"point_id": "obj_window_jamb", "kind": "opening", "xz": opening}],
        "point_correspondences": [{"source_id": "observed_window_jamb", "authored_id": "obj_window_jamb",
                                   "role": "heldout"}],
        "scene_per_m_hint": 90.0,
    }


def test_reviewed_walls_and_independent_opening_accept() -> None:
    result = solve_structural_registration(**_fixture())
    assert result["solver_status"] == "accepted", result["reason_codes"]
    assert result["reason_codes"] == []
    assert result["solution"]["scene_per_m"] == pytest.approx(92.0, abs=1e-5)
    assert result["solution"]["yaw_rad"] == pytest.approx(math.radians(17), abs=1e-6)
    assert result["solution"]["measurements"][-1]["error_m"] < 1e-6
    assert validate_solve_result(result) == result


def test_wrong_independent_opening_rejects() -> None:
    result = solve_structural_registration(**_fixture(opening_error_m=0.45))
    assert result["solver_status"] == "rejected"
    assert "heldout_point_residual" in result["reason_codes"]
    assert validate_solve_result(result) == result


def test_heldout_error_cannot_select_or_disambiguate_orientation() -> None:
    matched = solve_structural_registration(**_fixture())
    mismatched = solve_structural_registration(**_fixture(opening_error_m=0.45))
    assert matched["solver_status"] == "accepted"
    assert mismatched["solver_status"] == "rejected"
    for key in ("matrix_col_major", "selection_score_m", "alternate_selection_score_m"):
        assert mismatched["solution"][key] == pytest.approx(matched["solution"][key])
    assert mismatched["solution"]["validation_m"] > matched["solution"]["validation_m"]


def test_source_measurement_alias_cannot_be_fit_and_heldout() -> None:
    inputs = _fixture()
    alias = copy.deepcopy(inputs["source_points"][0])
    alias["point_id"] = "same_pixel_alias"
    inputs["source_points"].append(alias)
    inputs["authored_points"].append({"point_id": "other_obj_feature", "kind": "opening",
                                      "xz": [250.0, 180.0]})
    inputs["point_correspondences"].append({"source_id": "same_pixel_alias",
                                             "authored_id": "other_obj_feature", "role": "fit"})
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "duplicate_source_point_measurement" in result["reason_codes"]
    assert validate_solve_result(result) == result


def test_same_image_pixel_cannot_be_reprojected_as_independent_point() -> None:
    inputs = _fixture()
    inputs["source_points"][0]["evidence_refs"] = ["image_sha256:a", "pixel_uv:15.000000,20.000000"]
    inputs["source_points"].append({"point_id": "same_ray_other_plane", "kind": "opening",
                                    "xz": [1.0, 2.0], "source_wall_ids": ["east"],
                                    "evidence_refs": ["pixel_uv:15.000000,20.000000", "image_sha256:a"]})
    inputs["authored_points"].append({"point_id": "other_obj_feature", "kind": "opening",
                                      "xz": [250.0, 180.0]})
    inputs["point_correspondences"].append({"source_id": "same_ray_other_plane",
                                             "authored_id": "other_obj_feature", "role": "fit"})
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "duplicate_source_point_measurement" in result["reason_codes"]


def test_authored_point_alias_cannot_be_fit_and_heldout() -> None:
    inputs = _fixture()
    inputs["source_points"].append({"point_id": "other_pixel", "kind": "opening",
                                    "xz": [0.0, 2.0], "source_wall_ids": ["west"],
                                    "evidence_refs": ["independent_color_edge_23"]})
    inputs["authored_points"].append({"point_id": "same_obj_vertex_alias", "kind": "opening",
                                      "xz": inputs["authored_points"][0]["xz"][:]})
    inputs["point_correspondences"].append({"source_id": "other_pixel",
                                             "authored_id": "same_obj_vertex_alias", "role": "fit"})
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "duplicate_authored_point_measurement" in result["reason_codes"]


def test_two_distinct_jamb_measurements_remain_independent() -> None:
    inputs = _fixture()
    inputs["source_points"].append({"point_id": "other_jamb", "kind": "opening",
                                    "xz": [0.0, 2.0], "source_wall_ids": ["west"],
                                    "evidence_refs": ["independent_color_edge_23"]})
    scale, yaw = 92.0, math.radians(17)
    rotation = np.asarray(((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))))
    target = (scale * rotation @ np.asarray((0.0, 2.0)) + np.asarray((130.0, -45.0))).tolist()
    inputs["authored_points"].append({"point_id": "obj_other_jamb", "kind": "opening", "xz": target})
    inputs["point_correspondences"].append({"source_id": "other_jamb",
                                             "authored_id": "obj_other_jamb", "role": "heldout"})
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "accepted", result["reason_codes"]


def test_camera_or_person_field_cannot_be_inserted() -> None:
    result = solve_structural_registration(**_fixture())
    forged = copy.deepcopy(result)
    forged["inputs"]["camera_marker_scene_xz"] = [0.0, 0.0]
    with pytest.raises(StructuralRegistrationError):
        validate_solve_result(forged)


def test_saved_matrix_cannot_override_geometric_replay() -> None:
    result = solve_structural_registration(**_fixture())
    forged = copy.deepcopy(result)
    forged["solution"]["matrix_col_major"][12] += 1.0
    with pytest.raises(StructuralRegistrationError):
        validate_solve_result(forged)


def test_corner_intersection_of_fit_walls_is_not_independent() -> None:
    inputs = _fixture()
    inputs["source_points"][0].update(kind="corner", source_wall_ids=["west", "front"], evidence_refs=[])
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "heldout_corner_derived_from_fit_walls" in result["reason_codes"]


def test_duplicate_wall_fragment_cannot_be_heldout() -> None:
    inputs = _fixture()
    inputs["source_points"] = []
    inputs["authored_points"] = []
    inputs["point_correspondences"] = []
    inputs["source_segments"].append({"segment_id": "west_fragment", "start_xz": [0.0, 1.0],
                                      "end_xz": [0.0, 2.0], "support_count": 80, "vertical_span": 1.5})
    authored_west = inputs["authored_segments"][0]
    inputs["authored_segments"].append({**authored_west, "segment_id": "obj_west_fragment"})
    inputs["correspondences"].append({"source_id": "west_fragment", "authored_id": "obj_west_fragment",
                                      "role": "heldout"})
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "heldout_wall_not_independent_plane" in result["reason_codes"]


def test_rank_deficient_three_parallel_walls_reject() -> None:
    inputs = _fixture()
    inputs["source_segments"][2]["start_xz"] = [2.0, 0.0]
    inputs["source_segments"][2]["end_xz"] = [2.0, 3.0]
    inputs["authored_segments"][2]["start_xz"] = inputs["authored_segments"][0]["start_xz"]
    inputs["authored_segments"][2]["end_xz"] = inputs["authored_segments"][0]["end_xz"]
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "structural_geometry_rank_deficient" in result["reason_codes"]


def test_rank_is_invariant_to_world_origin_and_authored_units() -> None:
    original = _fixture()
    baseline = solve_structural_registration(**original)
    assert baseline["solver_status"] == "accepted"
    shifted = copy.deepcopy(original)
    source_offset = np.asarray((15.0, 10.0))
    for wall in shifted["source_segments"]:
        wall["start_xz"] = (np.asarray(wall["start_xz"]) + source_offset).tolist()
        wall["end_xz"] = (np.asarray(wall["end_xz"]) + source_offset).tolist()
    for point in shifted["source_points"]:
        point["xz"] = (np.asarray(point["xz"]) + source_offset).tolist()
    for wall in shifted["authored_segments"]:
        wall["start_xz"] = (np.asarray(wall["start_xz"]) * 2.0).tolist()
        wall["end_xz"] = (np.asarray(wall["end_xz"]) * 2.0).tolist()
    for point in shifted["authored_points"]:
        point["xz"] = (np.asarray(point["xz"]) * 2.0).tolist()
    shifted["scene_per_m_hint"] *= 2.0
    result = solve_structural_registration(**shifted)
    assert result["solver_status"] == "accepted", result["reason_codes"]
    assert result["solution"]["wall_rank_ratio"] == pytest.approx(
        baseline["solution"]["wall_rank_ratio"], abs=1e-10,
    )
    assert result["solution"]["scene_per_m"] == pytest.approx(
        2.0 * baseline["solution"]["scene_per_m"], rel=1e-6,
    )
    assert validate_solve_result(result) == result


def test_two_walls_plus_measured_structural_contact_can_fit() -> None:
    inputs = _fixture()
    inputs["correspondences"] = inputs["correspondences"][:2]
    scale, yaw = 92.0, math.radians(17)
    rotation = np.asarray(((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))))
    fitted_target = (scale * rotation @ np.asarray((0.0, 2.5)) + np.asarray((130.0, -45.0))).tolist()
    inputs["source_points"].append({"point_id": "measured_hearth_contact", "kind": "floor_contact",
                                    "xz": [0.0, 2.5], "source_wall_ids": ["west"],
                                    "evidence_refs": ["rectified_pixel_and_admitted_floor_ray_42"]})
    inputs["authored_points"].append({"point_id": "obj_hearth_contact", "kind": "floor_contact",
                                      "xz": fitted_target})
    inputs["point_correspondences"].append({"source_id": "measured_hearth_contact",
                                             "authored_id": "obj_hearth_contact", "role": "fit"})
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "accepted", result["reason_codes"]
    assert result["solution"]["scene_per_m"] == pytest.approx(92.0, abs=1e-5)


def test_two_walls_without_contact_are_not_scale_identifiable() -> None:
    inputs = _fixture()
    inputs["correspondences"] = inputs["correspondences"][:2]
    result = solve_structural_registration(**inputs)
    assert result["solver_status"] == "rejected"
    assert "insufficient_fit_structure" in result["reason_codes"]


def test_sparse_tilted_wall_plane_intersects_floor_without_horizontal_floor() -> None:
    wall_y, wall_z = np.meshgrid(np.linspace(0.30, 2.30, 35), np.linspace(0.0, 3.0, 60))
    wall = np.column_stack((2.0 + 0.08 * wall_y.ravel(), wall_y.ravel(), wall_z.ravel()))
    floor_x, floor_z = np.meshgrid(np.linspace(0.0, 4.0, 55), np.linspace(0.0, 3.0, 55))
    floor = np.column_stack((floor_x.ravel(), np.full(floor_x.size, 0.5), floor_z.ravel()))
    segments = extract_pcf_wall_planes(np.vstack((wall, floor)))
    assert segments
    best = max(segments, key=lambda segment: segment.support_count)
    assert best.vertical_span >= 1.8
    assert abs(best.start_xz[0] - 2.0) < 0.15
    assert abs(best.end_xz[0] - 2.0) < 0.15


def test_offline_pcf_wall_provenance_reextracts_and_detects_tamper(tmp_path) -> None:
    import trimesh

    wall_y, wall_z = np.meshgrid(np.linspace(0.30, 2.30, 35), np.linspace(0.0, 3.0, 60))
    wall = np.column_stack((2.0 + 0.08 * wall_y.ravel(), wall_y.ravel(), wall_z.ravel()))
    floor_x, floor_z = np.meshgrid(np.linspace(0.0, 4.0, 55), np.linspace(0.0, 3.0, 55))
    floor = np.column_stack((floor_x.ravel(), np.full(floor_x.size, 0.5), floor_z.ravel()))
    path = tmp_path / "points.glb"
    trimesh.Scene(trimesh.points.PointCloud(np.vstack((wall, floor)))).export(path)
    segments = extract_pcf_wall_planes(read_pcf_points_glb(path))
    assert segments
    assert len(verify_pcf_source_segments(path, segments[:1])) == 1
    tampered = replace(segments[0], start_xz=(segments[0].start_xz[0] + 0.1,
                                               segments[0].start_xz[1]))
    with pytest.raises(StructuralRegistrationError):
        verify_pcf_source_segments(path, [tampered])


def test_offline_authored_wall_provenance_checks_selected_geometry() -> None:
    inputs = _fixture()
    authored = inputs["authored_segments"]
    assert len(verify_authored_segments(authored, authored[:2], scene_per_m_hint=90.0)) == 2
    tampered = copy.deepcopy(authored[0])
    tampered["end_xz"][0] += 5.0
    with pytest.raises(StructuralRegistrationError):
        verify_authored_segments(authored, [tampered], scene_per_m_hint=90.0)
