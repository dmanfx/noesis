#!/usr/bin/env python3
"""Unit tests for V3DT diagnostics logger."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from noesis.diagnostics.telemetry_log import (
    TrackingDiagnosticsLogger,
    build_v3dt_session_start_payload,
)
from noesis_core.private_paths import PrivatePathError


def test_diagnostics_logger_writes_ndjson(tmp_path: Path) -> None:
    out_path = tmp_path / "private" / "diag.ndjson"
    logger = TrackingDiagnosticsLogger(output_path=out_path, flush_every=1)
    logger.log_event({"type": "v3dt_session_start", "ts": 1})
    logger.log_frame({"type": "v3dt_tracking_frame", "frame_id": 1, "tracks": []})
    logger.close()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first.get("type") == "v3dt_session_start"
    assert second.get("type") == "v3dt_tracking_frame"
    assert out_path.parent.stat().st_mode & 0o777 == 0o700
    assert out_path.stat().st_mode & 0o777 == 0o600
    assert out_path.stat().st_nlink == 1


def test_session_header_redacts_sources_and_allowlists_environment() -> None:
    payload = build_v3dt_session_start_payload(
        pipeline_config={
            "sources": [
                {
                    "uri_secret": "living-room",
                    "uri": "rtsp://camera-user:camera-password@camera.invalid/live",
                }
            ]
        },
        camera_labels={0: "living-room"},
        sensor_id_map={0: 0},
        environment={
            "NOESIS_TRACKING_MODE": "v3dt",
            "NOESIS_INTERNAL_AUTH_TOKEN": "must-not-serialize",
            "NOESIS_CAMERA_URI": "must-not-serialize",
        },
        observed_at=1.0,
    )

    source = payload["pipeline_config"]["sources"][0]
    assert source == {"uri_secret": "living-room"}
    assert payload["env"] == {"NOESIS_TRACKING_MODE": "v3dt"}
    assert "camera-password" not in json.dumps(payload)


def test_diagnostics_logger_enforces_byte_and_file_retention(tmp_path: Path) -> None:
    private = tmp_path / "bounded"
    first = private / "v3dt_frames_1.ndjson"
    second = private / "v3dt_frames_2.ndjson"
    third = private / "v3dt_frames_3.ndjson"

    logger = TrackingDiagnosticsLogger(
        output_path=first,
        flush_every=1,
        max_bytes=1024,
        max_files=2,
    )
    logger.log_frame({"type": "v3dt_tracking_frame", "payload": "x" * 900})
    logger.log_frame({"type": "v3dt_tracking_frame", "payload": "y" * 900})
    logger.close()
    assert first.stat().st_size <= 1024
    assert logger._dropped >= 1
    os.utime(first, ns=(1, 1))

    logger = TrackingDiagnosticsLogger(output_path=second, max_files=2)
    logger.close()
    os.utime(second, ns=(2, 2))
    logger = TrackingDiagnosticsLogger(output_path=third, max_files=2)
    logger.close()

    assert not first.exists()
    assert second.exists()
    assert third.exists()


def test_diagnostics_logger_rejects_unsafe_parent(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PrivatePathError, match="mode must be 0700"):
        TrackingDiagnosticsLogger(output_path=public / "diag.ndjson")


def test_diagnostics_close_rejects_unresolved_writer_ownership(tmp_path: Path) -> None:
    logger = TrackingDiagnosticsLogger(output_path=tmp_path / "private" / "diag.ndjson")
    logger.close()
    release = threading.Event()
    blocked = threading.Thread(target=release.wait, daemon=True)
    blocked.start()
    logger._thread = blocked
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            logger.close(timeout=0.01)
    finally:
        release.set()
        blocked.join(timeout=1.0)
    logger.close(timeout=0.1)
