from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from . import app as phone_app
from .alignment import NoesisAlignmentError
from .test_phone_scan import _settings


SCAN = "20260906-050646-107f244c"
SESSION = "companion-20260906-050300-a2a5934d"


def _companion() -> dict:
    return {
        "session_id": SESSION,
        "camera_id": "living-room",
        "phone_capture_id": "phone-capture",
        "phone": {"scan_id": SCAN, "archive_sha256": "a" * 64},
        "status": "stopped",
        "error": None,
    }


def _seed(service, companion: dict | None) -> dict:
    service.scan_dir(SCAN).mkdir()
    outputs = {"view_count": 256, "coordinate_frame": "phone_world", "sentinel": "unchanged"}
    state = {
        "id": SCAN,
        "name": "Paired test",
        "status": "complete",
        "provider": "mapanything",
        "outputs": outputs,
    }
    if companion is not None:
        state["companion_capture"] = copy.deepcopy(companion)
    service._write_state_unlocked(SCAN, state)
    return outputs


def _wait(client) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/api/scans/{SCAN}").json()
        if state.get("alignment", {}).get("status") in {"complete", "failed"}:
            return state
        time.sleep(0.01)
    pytest.fail("alignment did not finish")


def test_pair_selects_built_target_and_serves_it_even_when_geometry_rejects(tmp_path, monkeypatch):
    seen = []

    def build(**kwargs):
        assert service._inference_lock.locked()
        assert kwargs["companion"]["session_id"] == SESSION
        assert kwargs["model_settings"].anchor_image is None
        kwargs["progress"](0.5, "Static depth is being fused")
        root = kwargs["scan_dir"] / "paired_static_reference" / "pair-revision"
        root.mkdir(parents=True)
        (root / "calibration.json").write_text("{}")
        (root / "manifest.json").write_text("{}")
        return {
            "status": "complete", "session_id": SESSION,
            "camera_id": "living-room", "revision_id": root.name,
            "target_revision": str(root.relative_to(kwargs["scan_dir"])),
            "calibration_path": str((root / "calibration.json").relative_to(kwargs["scan_dir"])),
            "artifacts": {"manifest": str((root / "manifest.json").relative_to(kwargs["scan_dir"]))},
        }

    def validate(scan_dir, reference):
        assert scan_dir.name == SCAN and reference["revision_id"] == "pair-revision"
        seen.append("validate")

    def align(scan_dir, output_dir, outputs, target, progress):
        assert target.target_revision == scan_dir / "paired_static_reference/pair-revision"
        assert target.calibration_path == target.target_revision / "calibration.json"
        assert outputs["sentinel"] == "unchanged"
        assert not service._inference_lock.locked()
        seen.append("align")
        raise NoesisAlignmentError("vertical_plane_residual")

    monkeypatch.setattr(phone_app, "validate_paired_static_reference", validate)
    app = phone_app.create_app(_settings(tmp_path), paired_static_runner=build, alignment_runner=align)
    service = app.state.phone_scan_service
    monkeypatch.setattr(service.companion_capture, "public_state", lambda _: _companion())
    original = _seed(service, _companion())
    with TestClient(app) as client:
        assert client.get("/api/health").json()["paired_static_alignment_available"] is True
        response = client.post(f"/api/scans/{SCAN}/align-noesis?camera_id=living-room")
        assert response.status_code == 202
        state = _wait(client)
        assert state["outputs"] == original
        aligned = state["alignment"]
        assert aligned["status"] == "failed"
        assert aligned["target_kind"] == "paired_static"
        assert aligned["target_release_id"] is None
        assert aligned["target_revision_id"] == "pair-revision"
        assert aligned["static_reference"]["status"] == "complete"
        assert client.get(aligned["static_reference"]["artifact_urls"]["manifest"]).status_code == 200
        assert not (service.scan_dir(SCAN) / "alignment").exists()
    assert seen == ["validate", "align"]


def test_failed_pair_build_never_uses_saved_target(tmp_path, monkeypatch):
    def build(**kwargs):
        raise RuntimeError("recorded calibration binding is missing")

    def align(*args):
        pytest.fail("failed pair must not fall back to saved geometry")

    app = phone_app.create_app(_settings(tmp_path), paired_static_runner=build, alignment_runner=align)
    service = app.state.phone_scan_service
    monkeypatch.setattr(service.companion_capture, "public_state", lambda _: _companion())
    original = _seed(service, _companion())
    with TestClient(app) as client:
        assert client.post(f"/api/scans/{SCAN}/align-noesis?camera_id=living-room").status_code == 202
        state = _wait(client)
        assert state["outputs"] == original
        assert state["alignment"]["static_reference"]["status"] == "failed"
        assert "binding is missing" in state["alignment"]["error"]


@pytest.mark.parametrize("mutation", ["camera", "archive", "scan", "unfinished", "malformed"])
def test_pair_identity_and_finalization_are_checked_before_queueing(tmp_path, monkeypatch, mutation):
    app = phone_app.create_app(_settings(tmp_path))
    service = app.state.phone_scan_service
    recorded = _companion()
    current = _companion()
    camera = "living-room"
    if mutation == "camera":
        camera = "kitchen"
    elif mutation == "archive":
        current["phone"]["archive_sha256"] = "b" * 64
    elif mutation == "scan":
        current["phone"]["scan_id"] = "different-scan"
    elif mutation == "unfinished":
        current["status"] = "recording"
    else:
        recorded["session_id"] = "../../wrong"
    monkeypatch.setattr(service.companion_capture, "public_state", lambda _: current)
    _seed(service, recorded)
    with TestClient(app) as client:
        response = client.post(f"/api/scans/{SCAN}/align-noesis?camera_id={camera}")
        assert response.status_code in {409, 422}
        assert "alignment" not in service.read_state(SCAN)


def test_completed_paired_target_reaches_pcf_consumer(tmp_path, monkeypatch):
    consumed = []

    def pcf(scan_dir, run_root, state, target, model_settings, progress):
        assert target.target_revision == scan_dir / "paired_static_reference/pair-revision"
        assert target.calibration_path.parent == target.target_revision
        consumed.append(target)
        raise RuntimeError("stop after checking the direct PCF consumer")

    app = phone_app.create_app(_settings(tmp_path), pcf_runner=pcf)
    service = app.state.phone_scan_service
    _seed(service, _companion())
    seen = []
    monkeypatch.setattr(phone_app, "validate_paired_static_reference", lambda *args: seen.append(args))
    reference = {
        "status": "complete", "camera_id": "living-room", "revision_id": "pair-revision",
        "target_revision": "paired_static_reference/pair-revision",
        "calibration_path": "paired_static_reference/pair-revision/calibration.json",
    }
    alignment = {
        "status": "complete",
        "target_kind": "paired_static", "target_camera_id": "living-room",
        "static_reference": reference,
        "results": {
            "target_camera_id": "living-room", "target_revision_id": "pair-revision",
            "quality_gate": {"passed": True},
        },
    }
    service.update_state(SCAN, provider="da3", alignment=alignment)
    with TestClient(app) as client:
        response = client.post(f"/api/scans/{SCAN}/initiate-pcf")
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = service.read_state(SCAN)
            if state["pcf"]["status"] == "failed":
                break
            time.sleep(0.01)
        assert state["pcf"]["status"] == "failed"
        assert "direct PCF consumer" in state["pcf"]["error"]
    assert len(seen) == 2
    assert len(consumed) == 1


def test_alignment_report_paths_survive_atomic_publication(tmp_path):
    building = tmp_path / ".alignment-building"
    final = tmp_path / "alignment"
    building.mkdir()
    report = building / "alignment_report.json"
    transform = building / "phone_ma_to_noesis_world.json"
    report.write_text("{}")
    transform.write_text("{}")
    result = {
        "artifact_paths": {"report": str(report), "transform": str(transform)},
        "files": [{"path": "alignment/alignment_report.json", "size_bytes": 2}],
    }
    phone_app._finalize_alignment_paths(building, final, result)
    building.rename(final)
    saved = json.loads((final / report.name).read_text())
    assert saved["artifact_paths"] == result["artifact_paths"]
    assert all(Path(path).is_file() for path in saved["artifact_paths"].values())
    row = result["files"][0]
    assert row["size_bytes"] == (final / report.name).stat().st_size
    assert row["sha256"] == hashlib.sha256((final / report.name).read_bytes()).hexdigest()
