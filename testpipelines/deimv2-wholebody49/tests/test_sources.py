from __future__ import annotations

from pathlib import Path

import pytest

import sources


def test_select_sources_requires_three_rtsp():
    payload = {
        "width": 1920,
        "height": 1080,
        "uris": [
            "rtsp://127.0.0.1/a",
            "rtsp://127.0.0.1/b",
            "rtsp://127.0.0.1/c",
        ],
        "sensor_ids": ["0", "1", "2"],
        "sensor_names": ["Living Room Camera", "Kitchen Camera", "Family Room Camera"],
    }
    selected = sources.select_sources(payload)
    assert selected["uris"] == payload["uris"]
    assert selected["sensor_names"] == payload["sensor_names"]


def test_select_sources_rejects_file_sources():
    payload = {
        "uris": ["rtsp://127.0.0.1/a", "file:///tmp/b.mp4", "rtsp://127.0.0.1/c"],
        "sensor_ids": ["0", "1", "2"],
        "sensor_names": ["a", "b", "c"],
    }
    with pytest.raises(RuntimeError, match="requires RTSP"):
        sources.select_sources(payload)


def test_generate_sources_yaml_from_infer_config(tmp_path: Path):
    infer_path = tmp_path / "infer.yaml"
    infer_path.write_text(
        "\n".join(
            [
                "streammux:",
                "  width: 1920",
                "  height: 1080",
                "sources:",
                "  - uri: rtsp://127.0.0.1/living",
                "  - name: Kitchen Camera",
                "    sensor-id: kitchen",
                "    uri: rtsp://127.0.0.1/kitchen",
                "  - uri: rtsp://127.0.0.1/family",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "sources.yaml"
    generated = sources.generate_sources_yaml_from_infer_config(
        output_path=out_path,
        infer_config_path=infer_path,
        force=True,
    )
    assert generated["uris"] == [
        "rtsp://127.0.0.1/living",
        "rtsp://127.0.0.1/kitchen",
        "rtsp://127.0.0.1/family",
    ]
    assert generated["sensor_names"] == [
        "Living Room Camera",
        "Kitchen Camera",
        "Family Room Camera",
    ]
