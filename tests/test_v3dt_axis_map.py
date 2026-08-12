from __future__ import annotations

import math

import pytest

from noesis_core.v3dt_validation import (
    V3DTAxisMap,
    V3DTAxisMapError,
    v3dt_bbox3d_tracker_foot,
    v3dt_bbox3d_world_foot,
)


def test_xzy_map_restores_tracker_z_up_foot_to_canonical_y_up_world() -> None:
    axis_map = V3DTAxisMap.parse("xzy")
    bbox3d = {
        "xCentre": 19.78,
        "yCentre": 21.493,
        "zCentre": 0.925,
        "zLen": 1.85,
    }

    assert v3dt_bbox3d_tracker_foot(bbox3d) == pytest.approx(
        (19.78, 21.493, 0.0)
    )
    assert v3dt_bbox3d_world_foot(bbox3d, axis_map) == pytest.approx(
        (19.78, 0.0, 21.493)
    )


@pytest.mark.parametrize(
    ("spec", "tracker", "world", "canonical_spec"),
    (
        ("xyz", (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), "xyz"),
        ("x,z,y", (1.0, 2.0, 3.0), (1.0, 3.0, 2.0), "xzy"),
        ("x -z y", (1.0, 2.0, 3.0), (1.0, 3.0, -2.0), "x -z y"),
        ("-y x z", (1.0, 2.0, 3.0), (2.0, -1.0, 3.0), "-y x z"),
    ),
)
def test_signed_permutation_is_canonical_and_invertible(
    spec: str,
    tracker: tuple[float, float, float],
    world: tuple[float, float, float],
    canonical_spec: str,
) -> None:
    axis_map = V3DTAxisMap.parse(spec)
    assert axis_map.spec == canonical_spec
    assert axis_map.tracker_to_world(tracker) == pytest.approx(world)
    assert axis_map.world_to_tracker(world) == pytest.approx(tracker)


@pytest.mark.parametrize(
    "spec",
    (
        None,
        "",
        "xy",
        "xyzz",
        "xxz",
        "x y y",
        "x q z",
        "++x y z",
        "x --y z",
        "x + y",
    ),
)
def test_invalid_axis_maps_fail_closed(spec: object) -> None:
    with pytest.raises(V3DTAxisMapError):
        V3DTAxisMap.parse(spec)


@pytest.mark.parametrize(
    "point",
    (
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, True, 3.0),
        (1.0, math.nan, 3.0),
        (1.0, math.inf, 3.0),
        "xyz",
        7,
    ),
)
def test_invalid_points_fail_closed(point: object) -> None:
    with pytest.raises(V3DTAxisMapError):
        V3DTAxisMap.parse("xzy").tracker_to_world(point)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bbox3d",
    (
        {},
        {"xCentre": 1.0, "yCentre": 2.0, "zCentre": 3.0, "zLen": 0.0},
        {"xCentre": 1.0, "yCentre": 2.0, "zCentre": 3.0, "zLen": -1.0},
        {
            "xCentre": 1.0,
            "yCentre": 2.0,
            "zCentre": float("nan"),
            "zLen": 1.0,
        },
    ),
)
def test_invalid_bbox3d_foot_fails_closed(bbox3d: dict[str, object]) -> None:
    with pytest.raises(V3DTAxisMapError):
        v3dt_bbox3d_world_foot(bbox3d, V3DTAxisMap.parse("xzy"))
