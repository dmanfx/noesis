"""Focused provenance and replay checks for normal-supported PCF source planes."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from noesis.calibration.structural_registration import StructuralRegistrationError, validate_solve_result
from noesis.calibration.structural_roi_evidence import (
    _plane_segment_replay, verify_source_feature_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "data/scene_priors/review/new-model-20261004/kitchen-refit/admission"
PRIOR = ROOT / "data/scene_priors/revisions/sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483"
CALIBRATION = ROOT / "data/virtual_twin/owner-selected-latest-structural-20261004/calibration-bundle.json"


@pytest.fixture(scope="module")
def kitchen_proof():
    files = [ADMISSION / "candidate-source-wall-bound.json",
             ADMISSION / "source-feature-evidence.json", PRIOR / "points.glb", CALIBRATION]
    if any(not path.is_file() for path in files):
        pytest.skip("private retained Kitchen source evidence is not installed")
    candidate = json.loads(files[0].read_text())
    evidence = json.loads(files[1].read_text())
    binding = json.loads(CALIBRATION.read_text())["cameras"]["frame_bindings"]["kitchen"]
    return candidate, evidence, binding


def test_exact_kitchen_source_reextracts_from_admitted_pcf(kitchen_proof):
    candidate, evidence, binding = kitchen_proof
    assert validate_solve_result(candidate) == candidate
    assert candidate["solver_status"] == "rejected"
    assert candidate["reason_codes"] == ["heldout_point_residual"]
    verify_source_feature_evidence(candidate, PRIOR / "points.glb", evidence, ROOT,
                                   offline=True, binding=binding)


def test_startup_replays_frozen_planes_height_and_exact_phone_rays(kitchen_proof):
    candidate, evidence, binding = kitchen_proof
    verify_source_feature_evidence(candidate, PRIOR / "points.glb", evidence, ROOT,
                                   offline=False, binding=binding)
    assert {row["segment_id"] for row in evidence["walls"]} == {
        row["segment_id"] for row in candidate["inputs"]["source_segments"]}
    assert all(point["source_wall_ids"] == ["tv_wall"]
               for point in candidate["inputs"]["source_points"])


@pytest.mark.parametrize("change", ["revision", "wall_tilt", "wall_height", "ray_pixel",
                                     "missing_source_wall", "source_npz", "holdout_role"])
def test_source_proof_fails_closed_on_binding_geometry_and_provenance(kitchen_proof, change):
    candidate, evidence, binding = copy.deepcopy(kitchen_proof)
    if change == "revision":
        evidence["target_world_revision"] = "other-prior"
    elif change == "wall_tilt":
        evidence["walls"][0]["plane"]["normal_xyz"][1] += .10
    elif change == "wall_height":
        evidence["walls"][0]["segment"]["start_xz"][0] += .05
    elif change == "ray_pixel":
        evidence["point_evidence"][0]["pixel_uv"][0] += 1
    elif change == "missing_source_wall":
        candidate["inputs"]["source_points"][0]["source_wall_ids"] = []
    elif change == "source_npz":
        evidence["views"]["0022"]["sha256"] = "0" * 64
    elif change == "holdout_role":
        candidate["inputs"]["point_correspondences"][0]["role"] = "fit"
    with pytest.raises(StructuralRegistrationError):
        verify_source_feature_evidence(candidate, PRIOR / "points.glb", evidence, ROOT,
                                       offline=False, binding=binding)


def test_plane_segment_replay_rejects_flattened_tilt(kitchen_proof):
    _, evidence, _ = kitchen_proof
    wall = copy.deepcopy(evidence["walls"][0])
    review = json.loads((ROOT / evidence["review"]["path"]).read_text())
    reference_y = review["common_observed_reference_y_m"]
    _plane_segment_replay(wall["plane"], wall["segment"], reference_y)
    wall["plane"]["normal_xyz"][1] = 0.0
    with pytest.raises(StructuralRegistrationError):
        _plane_segment_replay(wall["plane"], wall["segment"], reference_y)
