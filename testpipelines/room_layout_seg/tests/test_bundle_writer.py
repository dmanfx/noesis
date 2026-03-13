import json
from pathlib import Path

import numpy as np

from testpipelines.room_layout_seg.probes import write_layout_bundle


def test_write_layout_bundle(tmp_path: Path):
    output_labels = [
        {"output_id": 0, "name": "other", "source_class_ids": [], "palette": [0, 0, 0]},
        {"output_id": 1, "name": "wall", "source_class_ids": [0], "palette": [255, 0, 0]},
        {"output_id": 2, "name": "floor", "source_class_ids": [3], "palette": [0, 255, 0]},
    ]
    avg_probabilities = np.zeros((3, 4, 5), dtype=np.float32)
    avg_probabilities[1, :, :2] = 0.9
    avg_probabilities[2, :, 2:] = 0.8
    avg_probabilities[0] = 1.0 - avg_probabilities[1] - avg_probabilities[2]
    model_info = {
        "model": {"name": "test-model", "base_model_id": "test/model"},
        "input": {"height": 4, "width": 5},
        "output": {"classes": 3, "height": 4, "width": 5},
    }

    manifest = write_layout_bundle(
        output_root=tmp_path,
        sensor_id="cam0",
        sensor_name="Family Room Camera",
        avg_probabilities=avg_probabilities,
        output_labels=output_labels,
        model_info=model_info,
        frames_accumulated=12,
        first_seen_s=1.0,
        last_seen_s=2.0,
    )

    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()
    assert (manifest_path.parent / "layout_class_map.png").exists()
    assert (manifest_path.parent / "layout_preview.png").exists()
    assert (manifest_path.parent / "layout_probabilities_fp16.npz").exists()
    assert (manifest_path.parent / "mask_wall.png").exists()
    assert (manifest_path.parent / "mask_floor.png").exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["frames_accumulated"] == 12
    assert payload["coverage_percent"]["wall"] > 0
    assert payload["coverage_percent"]["floor"] > 0
