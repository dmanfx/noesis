from pathlib import Path

from testpipelines.room_layout_seg import sources


def test_generate_sources_yaml_from_infer_config(tmp_path: Path):
    infer_path = tmp_path / "infer.yaml"
    infer_path.write_text(
        "\n".join(
            [
                "sources:",
                "  - name: Family Room Camera",
                "    sensor-id: family",
                "    uri: file:///tmp/family.mp4",
                "  - name: Kitchen Camera",
                "    sensor-id: kitchen",
                "    uri: rtsp://127.0.0.1/stream",
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
    assert generated["uris"] == ["file:///tmp/family.mp4", "rtsp://127.0.0.1/stream"]
    assert generated["sensor_ids"] == ["family", "kitchen"]
    assert generated["sensor_names"] == ["Family Room Camera", "Kitchen Camera"]


def test_select_sources_camera_filter():
    payload = {
        "width": 1920,
        "height": 1080,
        "uris": ["file:///tmp/a.mp4", "file:///tmp/b.mp4"],
        "sensor_ids": ["a", "b"],
        "sensor_names": ["Family Room Camera", "Kitchen Camera"],
    }
    selected = sources.select_sources(payload, camera="Kitchen Camera", max_sources=1)
    assert selected["uris"] == ["file:///tmp/b.mp4"]
    assert selected["sensor_ids"] == ["b"]
    assert selected["sensor_names"] == ["Kitchen Camera"]

