from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from noesis.pipelines import hooks, hooks_v3dt_reimpl
from noesis_core.analytics_zones import resolve_authoritative_analytics_zone
from noesis_core.contracts.base import Matrix3, Vector3
from noesis_core.contracts.identity import IdentityKind, SubjectRef
from noesis_core.contracts.world import (
    EntityLifecycle,
    WorldEntity,
    WorldSourceEvidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("analytics", "expected"),
    [
        ({"ocStatus": ["Kitchen"]}, "Kitchen"),
        ({"roiStatus": ["FamilyRoom"]}, "FamilyRoom"),
        ({"ocStatus": [], "roiStatus": ["LivingRoom"]}, "LivingRoom"),
        (
            {
                "ocStatus": ["Kitchen", "Kitchen"],
                "roiStatus": ["FamilyRoom"],
            },
            "Kitchen",
        ),
        (
            {
                "ocStatus": {"Kitchen": 1, "FamilyRoom": 0},
                "roiStatus": ["FamilyRoom"],
            },
            "Kitchen",
        ),
        (
            {
                "ocStatus": ["Kitchen", "FamilyRoom"],
                "roiStatus": ["Kitchen"],
            },
            None,
        ),
        ({"ocStatus": ["Kitchen", "kitchen"]}, None),
        ({"ocStatus": [""], "roiStatus": ["Kitchen"]}, None),
        ({"ocStatus": [" Kitchen"], "roiStatus": ["Kitchen"]}, None),
        ({"ocStatus": ["Kitchen "], "roiStatus": ["Kitchen"]}, None),
        ({"ocStatus": ["x" * 161], "roiStatus": ["Kitchen"]}, None),
        ({"ocStatus": [7], "roiStatus": ["Kitchen"]}, None),
        ({"ocStatus": {"Kitchen": "unknown"}, "roiStatus": ["Kitchen"]}, None),
        ({"ocStatus": [], "roiStatus": ["Kitchen", "FamilyRoom"]}, None),
        ({"ocStatus": [], "roiStatus": []}, None),
        (None, None),
    ],
)
def test_authoritative_analytics_zone_resolution_is_exact_and_fail_closed(
    analytics: dict[str, Any] | None,
    expected: str | None,
) -> None:
    assert resolve_authoritative_analytics_zone(analytics) == expected


def _processor(module: Any) -> Any:
    processor = object.__new__(module._AnalyticsTelemetryProcessor)
    processor._analytics_obj_meta_type = "NVIDIA.DSANALYTICSOBJ.USER_META"
    processor._tracking_mode_is_v3dt = lambda: False
    return processor


def _rect() -> SimpleNamespace:
    return SimpleNamespace(left=10.0, top=20.0, width=30.0, height=40.0)


@pytest.mark.parametrize("module", [hooks, hooks_v3dt_reimpl])
def test_servicemaker_adapter_uses_oc_membership_before_roi_compatibility(
    module: Any,
) -> None:
    analytics = SimpleNamespace(
        dirStatus="",
        lcStatus=[],
        ocStatus=["Kitchen"],
        roiStatus=["LegacyRoom"],
    )
    obj = SimpleNamespace(
        object_id=7,
        class_id=0,
        confidence=0.9,
        tracker_confidence=0.8,
        rect_params=_rect(),
        nvdsanalytics_obj_items=[analytics],
    )

    track = _processor(module)._build_track_dict_ds8(obj, "kitchen")

    assert track is not None
    assert track["zone"] == "Kitchen"
    assert track["zone_source"] == "nvdsanalytics_roi"
    assert track["zone_authoritative"] is True


@pytest.mark.parametrize("module", [hooks, hooks_v3dt_reimpl])
def test_pyds_adapter_rejects_ambiguous_oc_membership_without_roi_fallback(
    module: Any,
) -> None:
    analytics = SimpleNamespace(
        dirStatus="",
        lcStatus=[],
        ocStatus=["Kitchen", "FamilyRoom"],
        roiStatus=["Kitchen"],
    )
    user_meta = SimpleNamespace(
        base_meta=SimpleNamespace(
            meta_type="NVIDIA.DSANALYTICSOBJ.USER_META",
        ),
        user_meta_data=analytics,
    )
    obj = SimpleNamespace(
        object_id=7,
        class_id=0,
        confidence=0.9,
        tracker_confidence=0.8,
        rect_params=_rect(),
        obj_user_meta_list=[user_meta],
    )

    track = _processor(module)._build_track_dict(obj, "kitchen")

    assert track is not None
    assert "zone" not in track
    assert "zone_source" not in track
    assert "zone_authoritative" not in track


def test_ds9_servicemaker_adapter_executes_the_shared_zone_resolver() -> None:
    script = """
from pathlib import Path
from types import SimpleNamespace
from noesis.pipelines import hooks

expected = Path("DS9/noesis/pipelines/hooks.py").resolve()
assert Path(hooks.__file__).resolve() == expected
processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
processor._analytics_obj_meta_type = "NVIDIA.DSANALYTICSOBJ.USER_META"
processor._tracking_mode_is_v3dt = lambda: False
analytics = SimpleNamespace(
    dirStatus="",
    lcStatus=[],
    ocStatus=["FamilyRoom"],
    roiStatus=["LegacyRoom"],
)
obj = SimpleNamespace(
    object_id=7,
    class_id=0,
    confidence=0.9,
    tracker_confidence=0.8,
    rect_params=SimpleNamespace(left=10.0, top=20.0, width=30.0, height=40.0),
    nvdsanalytics_obj_items=[analytics],
)
track = processor._build_track_dict_ds8(obj, "family-room")
assert track["zone"] == "FamilyRoom"
assert track["zone_source"] == "nvdsanalytics_roi"
assert track["zone_authoritative"] is True
"""
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath = [str(REPO_ROOT / "DS9"), str(REPO_ROOT)]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    completed = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def _subject() -> SubjectRef:
    return SubjectRef(
        subject_id="resident:abc",
        kind=IdentityKind.RESIDENT,
        resident_uuid="abc",
        display_name="Resident",
        stable_id=1,
    )


def _source(
    camera_id: str,
    zone: str,
    *,
    accepted: bool = True,
    zone_source: str = "nvdsanalytics_roi",
    zone_authoritative: bool = True,
) -> WorldSourceEvidence:
    return WorldSourceEvidence(
        observation_id=f"obs:{camera_id}",
        camera_id=camera_id,
        zone=zone,
        zone_source=zone_source,  # type: ignore[arg-type]
        zone_authoritative=zone_authoritative,
        observed_at_us=1_000_000,
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
        ),
        accepted=accepted,
        rejection_reason=None if accepted else "position_conflict",
    )


def _entity(
    sources: tuple[WorldSourceEvidence, ...],
    room_id: str | None,
    *,
    conflict: bool = False,
) -> WorldEntity:
    return WorldEntity(
        entity_id="resident:abc",
        subject=_subject(),
        lifecycle=EntityLifecycle.PRESENT,
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
        ),
        room_id=room_id,
        observed_at_us=1_000_000,
        stale_after_us=2_000_000,
        sources=sources,
        conflict=conflict,
        conflict_reason="room conflict" if conflict else None,
    )


def test_world_entity_requires_exact_same_room_consensus() -> None:
    sources = (
        _source("kitchen-a", "Kitchen"),
        _source("kitchen-b", "Kitchen"),
    )
    assert _entity(sources, "Kitchen").room_id == "Kitchen"

    with pytest.raises(
        ValidationError,
        match="room_id must equal the sole accepted authoritative source zone",
    ):
        _entity(sources, "FamilyRoom")
    with pytest.raises(
        ValidationError,
        match="room_id must equal the sole accepted authoritative source zone",
    ):
        _entity(sources, None)


def test_world_entity_rejects_rooms_without_one_authoritative_vote() -> None:
    diagnostic = _source(
        "kitchen",
        "Kitchen",
        zone_source="camera_default",
        zone_authoritative=False,
    )
    assert _entity((diagnostic,), None).room_id is None
    with pytest.raises(
        ValidationError,
        match="room_id must equal the sole accepted authoritative source zone",
    ):
        _entity((diagnostic,), "Kitchen")

    conflicting = (
        _source("kitchen", "Kitchen"),
        _source("family-room", "FamilyRoom"),
    )
    assert _entity(conflicting, None, conflict=True).room_id is None
    with pytest.raises(
        ValidationError,
        match="conflicting authoritative source zones require conflict=true",
    ):
        _entity(conflicting, None)


def test_world_entity_rejected_authoritative_source_does_not_vote() -> None:
    sources = (
        _source("kitchen", "Kitchen"),
        _source("family-room", "FamilyRoom", accepted=False),
    )
    assert _entity(sources, "Kitchen", conflict=True).room_id == "Kitchen"
