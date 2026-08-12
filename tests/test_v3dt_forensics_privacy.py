#!/usr/bin/env python3
"""Privacy contract tests for V3DT forensic artifacts and CLI defaults."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from noesis.diagnostics.v3dt_forensics import build_snapshot
from noesis_core.private_paths import PrivatePathError


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "v3dt_forensics.py"
_SPEC = importlib.util.spec_from_file_location("v3dt_forensics_cli", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI)


def test_snapshot_redacts_materialized_camera_uri_and_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "sources:\n"
        "  - uri_secret: living-room\n"
        "    uri: rtsp://camera-user:camera-password@camera.invalid/live\n",
        encoding="utf-8",
    )
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text("cameras: {}\nintrinsics_models: {}\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"cameras": {}}', encoding="utf-8")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("NOESIS_TRACKING_MODE", "v3dt")
    monkeypatch.setenv("NOESIS_INTERNAL_AUTH_TOKEN", "must-not-serialize")
    monkeypatch.setenv("NOESIS_CAMERA_URI", "must-not-serialize")

    snapshot = build_snapshot(
        pipeline_path=pipeline,
        cameras_path=cameras,
        calibration_path=calibration,
        caminfo_dir=tmp_path / "caminfo",
        alignment_path=alignment,
    )

    source = snapshot["inputs"]["raw"]["pipeline"]["sources"][0]
    assert source == {"uri_secret": "living-room"}
    assert snapshot["inputs"]["env"] == {"NOESIS_TRACKING_MODE": "v3dt"}
    serialized = json.dumps(snapshot)
    assert "camera-password" not in serialized
    assert "must-not-serialize" not in serialized


def test_cli_artifact_writer_creates_private_outputs(tmp_path: Path) -> None:
    output = tmp_path / "private" / "v3dt_report_test.json"

    _CLI._write_output(output, '{"ok":true}')

    assert output.parent.stat().st_mode & 0o777 == 0o700
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.stat().st_nlink == 1
    assert _CLI._read_private_json(output, label="test report") == {"ok": True}


def test_cli_artifact_writer_rejects_public_directory_without_chmod(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)

    with pytest.raises(PrivatePathError, match="mode must be 0700"):
        _CLI._write_output(public / "v3dt_report_test.json", "{}")

    assert public.stat().st_mode & 0o777 == 0o755


def test_cli_panel_server_rejects_non_loopback_bind() -> None:
    args = argparse.Namespace(dir="unused", host="0.0.0.0", port=8777)

    assert _CLI._serve_cmd(args) == 1
