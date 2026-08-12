#!/usr/bin/env python3
"""Unit tests for V3DT forensics analysis."""
from __future__ import annotations

import json
from pathlib import Path

from noesis.diagnostics.v3dt_forensics import analyze_tracking_log


def test_analyze_tracking_log(tmp_path: Path) -> None:
    log_path = tmp_path / "tracks.ndjson"
    records = [
        {
            "type": "v3dt_tracking_frame",
            "camera_id": "test-cam",
            "frame_id": 1,
            "tracks": [
                {
                    "track_id": 1,
                    "class_id": 0,
                    "bbox": [10.0, 20.0, 100.0, 200.0],
                    "bbox3d": {
                        "xCentre": 1.0,
                        "yCentre": 1.7,
                        "zCentre": 3.0,
                        "xLen": 0.5,
                        "yLen": 0.5,
                        "zLen": 1.7,
                        "xRot": 0.0,
                        "yRot": 0.0,
                        "zRot": 0.0,
                    },
                }
            ],
        },
        {
            "type": "v3dt_tracking_frame",
            "camera_id": "test-cam",
            "frame_id": 2,
            "tracks": [
                {
                    "track_id": 1,
                    "class_id": 0,
                    "bbox": [12.0, 22.0, 100.0, 200.0],
                    "bbox3d": {
                        "xCentre": 1.1,
                        "yCentre": 1.7,
                        "zCentre": 3.1,
                        "xLen": 0.5,
                        "yLen": 0.5,
                        "zLen": 1.7,
                        "xRot": 0.0,
                        "yRot": 0.0,
                        "zRot": 0.0,
                    },
                }
            ],
        },
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    report = analyze_tracking_log(log_path)
    cam = report["cameras"]["test-cam"]
    assert cam["frames"] == 2
    assert cam["people_tracks"] == 2
    assert cam["bbox3d_tracks"] == 2
    assert abs(float(cam["bbox3d_height"]["median"]) - 1.7) < 1e-6
    assert cam["track_length_frames"]["median"] == 2
