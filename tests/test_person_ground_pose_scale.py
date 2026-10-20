"""Focused scale regression: image size cannot establish seated authority."""
import numpy as np
import pytest

from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    classify_posture,
    legs_are_bent,
    resolve_pose_floor_anchor,
)


def body(*, seated=False, scale=1.0, missing=(), horizontal=False):
    points = np.zeros((17, 3), dtype=np.float32)
    coordinates = {
        5: (100, 100), 6: (140, 100), 11: (105, 180), 12: (135, 180),
        13: (105, 240), 14: (135, 240), 15: (105, 300), 16: (135, 300),
    }
    if seated:
        coordinates.update({5: (100, 120), 6: (140, 120), 11: (110, 190),
                            12: (130, 190), 13: (150, 210), 14: (160, 210),
                            15: (155, 215), 16: (165, 215)})
    for index, xy in coordinates.items():
        points[index] = (*xy, .9)
    if horizontal:
        points[:, :2] = points[:, [1, 0]]
    points[:, :2] = points[:, :2] * scale + [693, 416.5]
    for index in missing:
        points[index, 2] = .26733398
    return points


@pytest.mark.parametrize("scale", (.125, .25, .375, 1., 2.))
@pytest.mark.parametrize("missing", ((), (15,)))
def test_straight_observed_leg_is_not_bent_only_because_small(scale, missing):
    cfg = HumanGroundConfig()
    points = body(scale=scale, missing=missing)
    assert not legs_are_bent(points, config=cfg)
    posture = classify_posture(kpts_abs=points,
                              bbox=[693, 416.5, 60 * scale, 220 * scale],
                              height_ref_scene=None, config=cfg)
    # Small silhouette may remain unknown: this repair never forces standing.
    assert posture in {"standing", "unknown"}
    anchor = resolve_pose_floor_anchor(points, posture=posture, config=cfg)
    assert anchor is not None
    assert anchor.source == ("pose_ankle_floor" if not missing else "pose_single_ankle_floor")


@pytest.mark.parametrize("scale", (.125, .375, 1., 2.))
@pytest.mark.parametrize("missing", ((), (15, 16)))
def test_true_seated_and_occluded_seated_geometry_remains_seated(scale, missing):
    cfg = HumanGroundConfig()
    points = body(seated=True, scale=scale, missing=missing)
    assert legs_are_bent(points, config=cfg)
    posture = classify_posture(kpts_abs=points,
                              bbox=[693, 416.5, 90 * scale, 110 * scale],
                              height_ref_scene=None, config=cfg)
    assert posture == "sitting"
    anchor = resolve_pose_floor_anchor(points, posture=posture, config=cfg)
    if missing:
        assert anchor is None
    else:
        assert anchor.source == "pose_ankle_floor"
        assert not anchor.height_lock_eligible


@pytest.mark.parametrize("scale", (.125, .375, 1., 2.))
def test_observed_lying_body_is_not_relabelled_upright(scale):
    points = body(scale=scale, horizontal=True)
    assert classify_posture(kpts_abs=points, bbox=[693, 416.5, 220 * scale, 60 * scale],
                            height_ref_scene=None, config=HumanGroundConfig()) == "lying"


@pytest.mark.parametrize("scale", (.125, .375, 1., 2.))
def test_unresolved_missing_feet_do_not_gain_extrapolated_floor_authority(scale):
    cfg = HumanGroundConfig()
    points = body(scale=scale, missing=(15, 16))
    assert not legs_are_bent(points, config=cfg)
    assert resolve_pose_floor_anchor(points, posture="unknown", config=cfg) is None
    # Existing explicit-standing inference remains distinct, not observed feet.
    anchor = resolve_pose_floor_anchor(points, posture="standing", config=cfg)
    assert anchor is not None and anchor.source == "pose_leg_floor"


def test_missing_torso_and_feet_are_not_a_pixel_size_seated_or_floor_proof():
    cfg = HumanGroundConfig()
    points = body(scale=.375, missing=(5, 6, 15, 16))
    assert not legs_are_bent(points, config=cfg)
    assert classify_posture(kpts_abs=points, bbox=[693, 416.5, 32.5, 75],
                            height_ref_scene=None, config=cfg) == "unknown"
    assert resolve_pose_floor_anchor(points, posture="unknown", config=cfg) is None
