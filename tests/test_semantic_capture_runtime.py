from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_capture_runtime.manager import SemanticCaptureManager


def _fake_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
    output = Path(command[command.index("--output-dir") + 1])
    rows: list[dict[str, object]] = []
    for source_id, (room, stem) in enumerate(
        (("Living Room", "living_room"), ("Kitchen", "kitchen"), ("Family Room", "family_room"))
    ):
        raw = output / f"source_{source_id}_{stem}_raw.jpg"
        masked = output / f"source_{source_id}_{stem}_masked.jpg"
        class_map = output / f"source_{source_id}_{stem}_class_map.png"
        raw.write_bytes(b"jpeg")
        masked.write_bytes(b"jpeg")
        class_map.write_bytes(b"png")
        rows.append(
            {
                "source_id": source_id,
                "room": room,
                "width": 640,
                "height": 640,
                "raw_path": str(raw),
                "masked_path": str(masked),
                "class_map_path": str(class_map),
                "top_classes": [{"id": 0, "name": "wall", "fraction": 1.0}],
            }
        )
    (output / "capture_summary.json").write_text(json.dumps({"captures": rows}), encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="complete", stderr="")


def test_capture_publishes_three_camera_manifest_and_exact_artifacts(tmp_path: Path) -> None:
    manager = SemanticCaptureManager(capture_root=tmp_path, command_runner=_fake_runner)

    manifest = manager.capture("l")

    assert manifest["contract"] == "noesis.semantic_seg.capture"
    assert manifest["model"] == "l"
    assert [row["id"] for row in manifest["cameras"]] == [
        "living-room",
        "kitchen",
        "family-room",
    ]
    assert manager.latest("l")["capture_id"] == manifest["capture_id"]
    assert manager.artifact_path(manifest["capture_id"], "living-room", "raw").read_bytes() == b"jpeg"


def test_capture_rejects_unknown_model_without_running(tmp_path: Path) -> None:
    manager = SemanticCaptureManager(capture_root=tmp_path, command_runner=_fake_runner)
    with pytest.raises(ValueError):
        manager.capture("xl")
