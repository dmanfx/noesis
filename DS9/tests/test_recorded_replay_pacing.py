from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_pipeline_module():
    module_name = "ds9_recorded_replay_pacing_pipeline"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = ROOT / "DS9" / "noesis" / "pipelines" / "ds8_pipeline.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(root: Path, policy: dict[str, object] | None) -> Path:
    engine = root / "fixture.engine"
    engine.write_bytes(b"engine-fixture")
    infer = root / "fixture.ini"
    infer.write_text(
        "[property]\n"
        "onnx-file=offline-maintenance-source.onnx\n"
        f"model-engine-file={engine}\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "version": 1,
        "batch_size": 2,
        "sources": [
            {"element": "nvurisrcbin", "uri": "file:///tmp/replay.mp4"},
            # The config loader intentionally rejects inline RTSP secrets;
            # this non-file URI is sufficient to prove local-only scoping.
            {"element": "nvurisrcbin", "uri": "udp://camera.invalid/live"},
        ],
        "streammux": {
            "element": "nvstreammux",
            "batch-size": 2,
            "width": 64,
            "height": 64,
        },
        "models": {
            "pgie": {
                "config-file-path": str(infer),
                "engine": str(engine),
            }
        },
        "tracker": {},
        "analytics": {"enable": False},
        "sinks": [{"name": "test_sink", "type": "fakesink", "sync": False}],
    }
    if policy is not None:
        payload["recorded_replay"] = policy
    path = root / "infer.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build(root: Path, policy: dict[str, object] | None):
    pipeline = _load_pipeline_module()
    pipeline._PIPELINE_SINGLETON = None
    with mock.patch.dict(
        os.environ,
        {
            "NOESIS_DS9_STUB_PIPELINE": "1",
            "NOESIS_BUILD_DIR": str(root / "build"),
            "NOESIS_MOSAIC_RTSP_ENABLED": "0",
            "NOESIS_MOSAIC_WEBRTC_ENABLED": "0",
        },
        clear=False,
    ):
        return pipeline.build_pipeline(_write_config(root, policy))


def test_recorded_replay_opt_in_clocks_and_preserves_local_file_only(tmp_path: Path) -> None:
    graph = _build(
        tmp_path,
        {"realtime": True, "preserve_frames": True},
    )

    clock = graph.components["source_replay_clock_0"]
    assert clock.element == "identity"
    assert clock.config == {"sync": True}
    assert graph.components["source_decode_queue_0"].config["leaky"] == 0
    assert graph.components["source_decode_queue_1"].config["leaky"] == 2
    assert "source_replay_clock_1" not in graph.components
    assert ("source_0", "source_replay_clock_0") in graph.ds_pipeline.links
    assert ("source_replay_clock_0", "source_decode_queue_0") in graph.ds_pipeline.links
    assert ("source_1", "source_decode_queue_1") in graph.ds_pipeline.links
    assert graph.components["source_0"].downstream == ["source_replay_clock_0"]
    assert clock.downstream == ["source_decode_queue_0"]


def test_default_and_live_graphs_keep_latest_only_queue_behavior(tmp_path: Path) -> None:
    graph = _build(tmp_path, None)
    assert "source_replay_clock_0" not in graph.components
    assert "source_replay_clock_1" not in graph.components
    for source_id in (0, 1):
        queue = graph.components[f"source_decode_queue_{source_id}"]
        assert queue.config["leaky"] == 2
        assert (f"source_{source_id}", queue.name) in graph.ds_pipeline.links


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        ([], "must be a mapping"),
        ({"realtime": 1}, "realtime must be a boolean"),
        ({"preserve_frames": "yes"}, "preserve_frames must be a boolean"),
        ({"clock": True}, "unsupported keys: clock"),
    ),
)
def test_recorded_replay_policy_rejects_ambiguous_values(
    tmp_path: Path,
    policy: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, policy if isinstance(policy, dict) else policy)  # type: ignore[arg-type]
