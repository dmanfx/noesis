from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from noesis.identity_v2_osd import IdentityV2PostResolutionOsdProcessor
from noesis.identity_v2_service import IdentityOsdDecision
from noesis.pipelines import hooks


ROOT = Path(__file__).resolve().parents[1]


class _OneShotObjects:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("metadata wrappers were traversed more than once")
        return iter(self.rows)


def _object(
    tracker: int,
    *,
    class_id: int = 0,
    label: str = "#XX",
    confidence: float = 0.91,
):
    return SimpleNamespace(
        object_id=tracker,
        class_id=class_id,
        confidence=confidence,
        obj_label=label,
        text_params=SimpleNamespace(display_text=label),
    )


def _processor(decisions):
    class Service:
        authoritative = True

        @staticmethod
        def lookup_osd_decision(*, camera_id, frame_id, tracker_id):
            return decisions.get((camera_id, frame_id, tracker_id))

    pipeline = SimpleNamespace(identity_v2_service=Service())
    return IdentityV2PostResolutionOsdProcessor(
        pipeline=pipeline,
        camera_labels={0: "camera-a"},
        sensor_id_map={10: 0},
        decimals=2,
    )


def test_post_resolution_osd_uses_exact_decision_and_preserves_depth_fragment() -> None:
    resident = IdentityOsdDecision(
        camera_id="camera-a",
        frame_id=7,
        tracker_id="11",
        identity_state="resident",
        compatibility_sid=1,
        display_name="Alice",
    )
    visitor = IdentityOsdDecision(
        camera_id="camera-a",
        frame_id=7,
        tracker_id="12",
        identity_state="visitor",
        compatibility_sid=1000,
        display_name=None,
    )
    rows = _OneShotObjects(
        [
            _object(11, label="person XX depth=2.37m 0.91"),
            _object(12, label="#XX 0.82", confidence=0.82),
            _object(99, class_id=2, label="dog 0.70", confidence=0.70),
        ]
    )
    frame = SimpleNamespace(
        source_id=10,
        frame_number=7,
        object_items=rows,
    )
    processor = _processor(
        {
            ("camera-a", 7, "11"): resident,
            ("camera-a", 7, "12"): visitor,
        }
    )
    processor.handle_frame_ds8(frame)
    assert rows.iterations == 1
    assert rows.rows[0].text_params.display_text == "#1 Alice depth=2.37m 0.91"
    assert rows.rows[1].text_params.display_text == "#1000 0.82"
    assert rows.rows[2].text_params.display_text == "dog 0.70"
    assert not any("object" in key or "frame" in key for key in processor.__dict__)


@pytest.mark.parametrize(
    "source_id,frame_id,tracker_id",
    [(11, 7, 11), (10, 8, 11), (10, 7, 13)],
)
def test_post_resolution_osd_is_neutral_on_camera_frame_or_tracker_mismatch(
    source_id: int,
    frame_id: int,
    tracker_id: int,
) -> None:
    decision = IdentityOsdDecision(
        camera_id="camera-a",
        frame_id=7,
        tracker_id="11",
        identity_state="resident",
        compatibility_sid=1,
        display_name="Alice",
    )
    obj = _object(tracker_id, label="#999 z=1.20m 0.91")
    processor = _processor({("camera-a", 7, "11"): decision})
    processor.handle_frame_ds8(
        SimpleNamespace(
            source_id=source_id,
            frame_number=frame_id,
            object_items=_OneShotObjects([obj]),
        )
    )
    assert obj.text_params.display_text == "#XX z=1.20m 0.91"
    assert "999" not in obj.text_params.display_text


def test_authoritative_attach_uses_verified_explicit_tiler_sink_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Probe:
        def __init__(self, name, operator):
            self.name = name
            self.operator = operator

    monkeypatch.setattr(hooks, "BatchMetadataOperator", object)
    monkeypatch.setattr(hooks, "Probe", Probe)
    monkeypatch.setattr(
        hooks,
        "_IdentityV2PostResolutionOsdOperator",
        lambda processor: processor,
    )

    class DsPipeline:
        def attach(self, *args, **kwargs):
            calls.append((args, kwargs))

    pipeline = SimpleNamespace(
        identity_v2_service=SimpleNamespace(authoritative=True),
        components={"tiler": SimpleNamespace(name="tiler", config={})},
        osd_label_processor=SimpleNamespace(decimals=2),
        ds_pipeline=DsPipeline(),
    )
    hooks.attach_identity_v2_post_resolution_osd_hook(
        pipeline,
        camera_labels={0: "camera-a"},
        sensor_id_map={0: 0},
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "tiler"
    assert kwargs == {"tips": "sink"}


def test_ds8_v3dt_ds9_post_resolution_osd_parity_is_declared() -> None:
    paths = (
        ROOT / "noesis" / "pipelines" / "hooks.py",
        ROOT / "noesis" / "pipelines" / "hooks_v3dt_reimpl.py",
        ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "attach_identity_v2_post_resolution_osd_hook(" in source
        assert 'tips="sink"' in source
        assert "IdentityV2PostResolutionOsdProcessor" in source
        assert "_IdentityV2PostResolutionOsdOperator" in source
