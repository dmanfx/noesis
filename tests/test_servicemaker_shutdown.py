from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from noesis_core.servicemaker_shutdown import (
    OrderlyEosError,
    pipeline_expects_finite_source_eos,
    request_orderly_eos,
)


class _EosNode:
    def __init__(self, *, accept: bool = True) -> None:
        self.request = 0
        self.accepted = 0
        self.ok = False
        self.accept = accept

    def get(self, key: str):
        return {
            "request-sequence": self.request,
            "accepted-sequence": self.accepted,
            "last-request-ok": self.ok,
        }[key]

    def set(self, values: dict[str, object]) -> None:
        self.request = int(values["request-sequence"])
        self.ok = self.accept
        if self.accept:
            self.accepted = self.request


class _AsyncEosNode(_EosNode):
    def set(self, values: dict[str, object]) -> None:
        self.request = int(values["request-sequence"])

        def _accept() -> None:
            self.accepted = self.request
            self.ok = True

        timer = threading.Timer(0.02, _accept)
        timer.daemon = True
        timer.start()


def _pipeline(node: _EosNode):
    component_name = "orderly_eos_control"
    return SimpleNamespace(
        shutdown_eos_component_name=component_name,
        components={component_name: SimpleNamespace(element="noesiseos")},
        ds_pipeline={component_name: node},
    )


def test_orderly_eos_requires_exact_monotonic_native_ack() -> None:
    node = _EosNode()
    pipeline = _pipeline(node)

    first = request_orderly_eos(pipeline)
    second = request_orderly_eos(pipeline)

    assert (first.request_sequence, first.accepted_sequence, first.last_request_ok) == (
        1,
        1,
        True,
    )
    assert second.request_sequence == 2
    assert node.accepted == 2


def test_orderly_eos_rejection_fails_closed() -> None:
    with pytest.raises(OrderlyEosError, match="acknowledgement timed out"):
        request_orderly_eos(
            _pipeline(_EosNode(accept=False)), timeout_s=0.02, poll_interval_s=0.002
        )


def test_orderly_eos_waits_for_asynchronous_native_ack() -> None:
    evidence = request_orderly_eos(
        _pipeline(_AsyncEosNode()), timeout_s=0.5, poll_interval_s=0.002
    )
    assert evidence.accepted_sequence == evidence.request_sequence == 1


def test_orderly_eos_requires_repo_owned_bridge() -> None:
    pipeline = SimpleNamespace(components={}, ds_pipeline={})
    with pytest.raises(OrderlyEosError, match="required noesiseos"):
        request_orderly_eos(pipeline)


@pytest.mark.parametrize(
    ("uri", "live_source", "file_loop", "drop_pipeline_eos", "expected"),
    [
        ("file:///tmp/finite.mp4", 0, False, False, True),
        ("file:///tmp/loop.mp4", 0, True, False, False),
        ("file:///tmp/suppressed.mp4", 0, False, True, False),
        ("file:///tmp/live-flag.mp4", 1, False, False, False),
        ("rtsp://camera.invalid/live", 0, False, False, False),
    ],
)
def test_finite_eos_classification_is_explicit(
    uri: str,
    live_source: int,
    file_loop: bool,
    drop_pipeline_eos: bool,
    expected: bool,
) -> None:
    pipeline = SimpleNamespace(
        config={"sources": [{"uri": uri}]},
        components={
            "streammux": SimpleNamespace(
                config={
                    "live-source": live_source,
                    "file-loop": file_loop,
                    "drop-pipeline-eos": drop_pipeline_eos,
                }
            )
        },
    )
    assert pipeline_expects_finite_source_eos(pipeline) is expected
