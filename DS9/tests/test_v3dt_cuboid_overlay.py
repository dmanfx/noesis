from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from noesis.pipelines import hooks


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("bbox", "anchor", "depth_vector"),
    [
        ((100.0, 80.0, 120.0, 300.0), (160.0, 380.0), (20.0, -30.0)),
        ((0.0, 0.0, 42.0, 88.0), (18.5, 86.0), (-9.0, -6.0)),
        ((1500.0, 400.0, 210.0, 500.0), (1624.0, 905.0), (0.0, -44.0)),
    ],
)
def test_bottom_face_centroid_is_exact_person_base(
    bbox: tuple[float, float, float, float],
    anchor: tuple[float, float],
    depth_vector: tuple[float, float],
) -> None:
    segments = hooks._anchored_cuboid_segments(bbox, anchor, depth_vector)

    front_right, front_left = segments[2]
    back_right, back_left = segments[6]
    centroid = (
        sum(point[0] for point in (front_right, front_left, back_right, back_left))
        / 4.0,
        sum(point[1] for point in (front_right, front_left, back_right, back_left))
        / 4.0,
    )

    assert centroid == pytest.approx(anchor, abs=1e-9)


def test_room_profiles_enable_correction_without_changing_baseline() -> None:
    baseline = yaml.safe_load(
        (REPO_ROOT / "DS9/config/infer.yaml").read_text(encoding="utf-8")
    )

    for profile in (
        "infer_v3dt_living_room_optimized.yaml",
        "infer_v3dt_living_kitchen_tracking_candidate.yaml",
        "infer_v3dt_living_family_phone_optimized.yaml",
    ):
        config = yaml.safe_load(
            (REPO_ROOT / "DS9/config" / profile).read_text(encoding="utf-8")
        )
        assert config["visualization"]["v3dt_cuboid"]["enabled"] is True
    assert "v3dt_cuboid" not in baseline["visualization"]


def test_correction_rejects_non_v3dt_mode() -> None:
    pipeline = SimpleNamespace(
        config={"visualization": {"v3dt_cuboid": {"enabled": True}}}
    )

    with pytest.raises(ValueError, match="only valid for a V3DT tracking mode"):
        hooks.attach_v3dt_cuboid_overlay_hook(
            pipeline,
            tracking_mode="baseline",
        )
