"""Completed-stage checkpoints and retry logs never erase failed evidence."""
import json
from unittest.mock import Mock

import pytest

from tools.mapanything_phone_scan import run_highres_reconstruction_lane as lane


def _child(monkeypatch, code=0):
    child = Mock()
    child.stdout = iter(["bounded stage output\n"])
    child.wait.return_value = code
    child.poll.return_value = code
    monkeypatch.setattr(lane.subprocess, "Popen", Mock(return_value=child))
    return child


def test_retry_keeps_previous_log_and_uses_explicit_checkpoint(tmp_path, monkeypatch):
    previous = tmp_path / "texture.log"
    previous.write_text("previous failed diagnostics\n")
    state = {"stages": [{"name": "texture", "status": "failed"}]}
    checkpoint = tmp_path / "postprocessing_manifest.json"
    _child(monkeypatch)
    lane.run_stage(tmp_path, "texture", ["mock-child"], state, state_path=checkpoint)
    assert previous.read_text() == "previous failed diagnostics\n"
    assert (tmp_path / "texture_attempt_2.log").read_text() == "bounded stage output\n"
    assert json.loads(checkpoint.read_text())["stages"][-1]["status"] == "complete"
    assert not (tmp_path / "lane_manifest.json").exists()


def test_failure_remains_in_checkpoint_and_its_log(tmp_path, monkeypatch):
    state = {"stages": []}
    _child(monkeypatch, code=3)
    with pytest.raises(lane.subprocess.CalledProcessError):
        lane.run_stage(tmp_path, "texture", ["mock-child"], state)
    saved = json.loads((tmp_path / "lane_manifest.json").read_text())
    assert saved["stages"][-1]["status"] == "failed"
    assert (tmp_path / "texture.log").read_text() == "bounded stage output\n"


def test_log_collision_fails_before_external_execution(tmp_path, monkeypatch):
    (tmp_path / "texture.log").write_text("original")
    (tmp_path / "texture_attempt_1.log").write_text("original retry")
    child = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(lane.subprocess, "Popen", child)
    with pytest.raises(ValueError, match="refusing to erase"):
        lane.run_stage(tmp_path, "texture", ["mock-child"], {"stages": []})
    child.assert_not_called()
