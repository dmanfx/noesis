from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from noesis import ds8_runtime, ds8_runtime_v3dt_reimpl


class _Trigger:
    def __init__(self, *, reject: bool = False, fail: bool = False) -> None:
        self.reject = reject
        self.fail = fail
        self.accepted = 0
        self.last_ok = False
        self.requests: list[int] = []

    def get(self, name: str):
        if name == "accepted-sequence":
            return self.accepted
        if name == "last-request-ok":
            return self.last_ok
        raise KeyError(name)

    def set(self, values: dict[str, object]) -> None:
        if self.fail:
            raise RuntimeError("event bridge failed")
        requested = int(values["request-sequence"])
        self.requests.append(requested)
        if not self.reject:
            self.accepted = requested
            self.last_ok = True


class _ServiceMakerPipeline:
    def __init__(self, trigger: _Trigger) -> None:
        self.trigger = trigger

    def __getitem__(self, name: str) -> _Trigger:
        assert name == "mosaic_force_idr"
        return self.trigger


def _pipeline(trigger: _Trigger):
    return SimpleNamespace(
        ds_pipeline=_ServiceMakerPipeline(trigger),
        components={
            "mosaic_force_idr": SimpleNamespace(element="noesisforceidr")
        },
    )


@pytest.mark.parametrize(
    "builder",
    [
        ds8_runtime._build_mosaic_keyframe_requester,
        ds8_runtime_v3dt_reimpl._build_mosaic_keyframe_requester,
    ],
)
def test_force_idr_requester_uses_strict_monotonic_native_events(builder) -> None:
    trigger = _Trigger()
    requester = builder(_pipeline(trigger), logging.getLogger("test"))

    assert requester is not None
    requester("webrtc_offer")
    requester("webrtc_connected")

    assert trigger.requests == [1, 2]
    assert trigger.accepted == 2


@pytest.mark.parametrize(
    "builder",
    [
        ds8_runtime._build_mosaic_keyframe_requester,
        ds8_runtime_v3dt_reimpl._build_mosaic_keyframe_requester,
    ],
)
@pytest.mark.parametrize("trigger", [_Trigger(reject=True), _Trigger(fail=True)])
def test_force_idr_requester_reports_native_event_failure(builder, trigger) -> None:
    failures: list[BaseException] = []
    requester = builder(
        _pipeline(trigger),
        logging.getLogger("test"),
        failure_callback=failures.append,
    )

    assert requester is not None
    with pytest.raises(RuntimeError):
        requester("webrtc_offer")
    assert len(failures) == 1


def test_force_idr_requester_rejects_graph_without_native_trigger() -> None:
    pipeline = SimpleNamespace(ds_pipeline=object(), components={})
    assert (
        ds8_runtime._build_mosaic_keyframe_requester(
            pipeline, logging.getLogger("test")
        )
        is None
    )
