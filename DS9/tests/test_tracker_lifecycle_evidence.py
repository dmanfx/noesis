from __future__ import annotations

import pytest

from noesis.pipelines.hooks import _tracker_lifecycle_ids_for_frame


def _evidence() -> dict[str, object]:
    return {
        "source_id": 0, "frame_id": 42, "media_pts_ns": 1_400_000_000,
        "shadow_available": True, "terminated_available": True,
        "shadow": [{"tracker_id": 7, "class_id": 0, "tracker_state": 2, "tracker_frame_id": 42}],
        "terminated": [],
    }


def _parse(evidence: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _tracker_lifecycle_ids_for_frame(evidence, source_id=0, frame_id=42, media_pts_ns=1_400_000_000)


def test_only_current_inactive_person_is_positive_native_presence() -> None:
    evidence = _evidence()
    evidence["shadow"] += [
        {"tracker_id": 8, "class_id": 0, "tracker_state": 3, "tracker_frame_id": 42},
        {"tracker_id": 9, "class_id": 0, "tracker_state": 2, "tracker_frame_id": 41},
    ]
    assert _parse(evidence) == ((7,), ())
    evidence["terminated"] = [{"tracker_id": 7, "class_id": 0}]
    assert _parse(evidence) == ((), (7,))


@pytest.mark.parametrize("field,value", [("source_id", 1), ("frame_id", 41), ("media_pts_ns", 1)])
def test_native_presence_requires_exact_batch_frame_identity(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="cohort mismatch"):
        _parse({**_evidence(), field: value})


def test_native_presence_requires_explicit_available_metadata_and_bounded_ids() -> None:
    with pytest.raises(ValueError, match="no metadata receipt"):
        _parse({**_evidence(), "shadow_available": False})
    with pytest.raises(ValueError, match="ID bounds"):
        _parse({**_evidence(), "terminated": [{}] * 257})
    with pytest.raises(ValueError, match="duplicate"):
        _parse({**_evidence(), "shadow": _evidence()["shadow"] * 2})
    assert _parse(None) == ((), ())
