from __future__ import annotations

from noesis.metadata.depth_result import DepthResult
from noesis.telemetry.publishers import DepthTelemetryPublisher
from noesis_core.runtime_publication import RuntimePublicationGate


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def broadcast_sync(self, message: dict[str, object]) -> None:
        self.messages.append(message)


def test_depth_telemetry_replaces_filesystem_path_with_opaque_reference() -> None:
    recorder = _Recorder()
    publisher = DepthTelemetryPublisher(recorder, RuntimePublicationGate())
    private_path = "/private/noesis/depth/living-room/frame.zarr"
    result = DepthResult(
        source_id=0,
        frame_id=4,
        ts=100,
        width=1920,
        height=1080,
        depth_map_ref=private_path,
        minmax=(0.5, 8.0),
    )

    publisher.publish(result)

    assert result.to_dict()["depth_map_ref"] == private_path
    payload = recorder.messages[0]["payload"]
    assert isinstance(payload, dict)
    public_ref = payload["depth_map_ref"]
    assert isinstance(public_ref, str)
    assert public_ref.startswith("noesis-depth://artifact/")
    assert private_path not in public_ref
    assert "living-room" not in public_ref


def test_public_depth_reference_is_stable_for_the_same_internal_artifact() -> None:
    result = DepthResult(
        source_id=1,
        frame_id=8,
        ts=200,
        width=640,
        height=480,
        depth_map_ref="memory://depth/camera/200",
        minmax=(1.0, 2.0),
    )

    assert result.to_public_dict()["depth_map_ref"] == result.to_public_dict()[
        "depth_map_ref"
    ]
