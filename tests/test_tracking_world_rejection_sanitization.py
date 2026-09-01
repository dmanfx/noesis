from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_PATHS = (
    ROOT / "DS9" / "noesis" / "telemetry" / "publishers.py",
    ROOT / "noesis" / "telemetry" / "publishers.py",
)

POSITION_AUTHORITY_FIELDS = (
    "world",
    "world_source",
    "world_filter_prediction",
    "world_prediction_image_foot",
    "world_prediction_provenance",
    "world_inferred_raw_sample",
    "world_inferred_raw_observation",
    "world_inferred_process_observation",
    "world_inferred_process_image_foot",
)


def _load_publishers(path: Path) -> ModuleType:
    module_name = f"_rejection_sanitization_{path.parent.parent.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("publisher_path", PUBLISHER_PATHS, ids=("ds9", "shared"))
def test_rejected_world_row_strips_every_position_authority_field(
    publisher_path: Path,
) -> None:
    publishers = _load_publishers(publisher_path)
    source_track = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "frame_id": 10,
        "world_valid": True,
        "world_quality": "held",
        "world_quality_reason": "producer_candidate",
        "world_measurement_accepted": True,
        "trail_append_allowed": True,
        "trail_break_required": False,
        "world": [1.0, 0.0, 2.0],
        "world_source": "image_motion_prediction",
        "world_filter_prediction": [1.1, 0.0, 2.1],
        "world_prediction_image_foot": [640.0, 700.0],
        "world_prediction_provenance": {"state_integrated": True},
        "world_inferred_raw_sample": [8.0, 0.0, 9.0],
        "world_inferred_raw_observation": [8.1, 0.0, 9.1],
        "world_inferred_process_observation": [1.2, 0.0, 2.2],
        "world_inferred_process_image_foot": [641.0, 701.0],
        "world_rejection_reason": "physical_measurement_rejected",
        "diagnostic_marker": "preserved",
    }

    normalized, admitted_keys = publishers._canonicalize_tracking_world_rows(
        [source_track],
        SimpleNamespace(observations=()),
    )

    assert admitted_keys == ()
    assert len(normalized) == 1
    public_track = normalized[0]
    assert all(field not in public_track for field in POSITION_AUTHORITY_FIELDS)
    assert public_track["world_valid"] is False
    assert public_track["world_quality"] == "invalid"
    assert public_track["world_quality_reason"] == (
        "canonical_world_service_rejected"
    )
    assert public_track["world_measurement_accepted"] is False
    assert public_track["trail_append_allowed"] is False
    assert public_track["trail_break_required"] is True
    assert public_track["world_rejection_reason"] == (
        "physical_measurement_rejected"
    )
    assert public_track["diagnostic_marker"] == "preserved"

    # Sanitization is publication-copy-only; producer state remains available
    # for internal diagnostics and cannot be mutated by the boundary adapter.
    assert source_track["world_valid"] is True
    assert all(field in source_track for field in POSITION_AUTHORITY_FIELDS)


@pytest.mark.parametrize("publisher_path", PUBLISHER_PATHS, ids=("ds9", "shared"))
def test_only_fusion_accepted_observation_binds_tracking_world_row(
    publisher_path: Path,
) -> None:
    publishers = _load_publishers(publisher_path)
    source_track = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "frame_id": 10,
        "world_valid": True,
        "world": [1.0, 0.0, 2.0],
    }
    observation = SimpleNamespace(
        observation_id="accepted-observation",
        payload=SimpleNamespace(
            world=object(),
            tracklet=SimpleNamespace(tracker_id=7, frame_id=10),
        ),
    )
    source = SimpleNamespace(
        observation_id="accepted-observation",
        accepted=True,
    )
    publication = SimpleNamespace(
        observations=(observation,),
        snapshot=SimpleNamespace(
            entities=(SimpleNamespace(sources=(source,)),),
        ),
    )

    normalized, admitted_keys = publishers._canonicalize_tracking_world_rows(
        [source_track],
        publication,
    )

    assert normalized[0]["world_valid"] is True
    assert normalized[0]["world"] == [1.0, 0.0, 2.0]
    assert admitted_keys == ((7, 3, 10),)

    source.accepted = False
    normalized, admitted_keys = publishers._canonicalize_tracking_world_rows(
        [source_track],
        publication,
    )

    assert normalized[0]["world_valid"] is False
    assert "world" not in normalized[0]
    assert admitted_keys == ()
