"""The selected PCF is consumed, never the original scan's failed raw fit."""
from copy import deepcopy
import hashlib
import json
import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from . import app as app_module, path_comparison, path_review
from .capture import validate_capture_manifest
from .test_android_capture import _manifest
from .test_browser_capture import _settings
from .test_path_comparison import paired, write
from .test_trajectory_motion_review import capture
from .test_walk_modes import SOURCE, TARGET, state as save_state
from .walk_intent import validate_walk_intent


@pytest.fixture
def pcf_handoff(paired):
    args, _, _ = paired
    root = args["scan_dir"].parent / "retained_pcf"
    root.mkdir()
    selection = {"kind": "scene_prior_pcf", "prior_id": "prior-room-v1", "camera_id": "living-room",
                 "manifest_sha256": "b" * 64, "frame_binding_sha256": "c" * 64}
    edge = {"contract_version": 1, "source_frame": {"frame_id": "backend_world_m", "revision": "cal1"},
            "target_frame": {"frame_id": "backend_world_m", "revision": "world1"},
            "target_from_source_sha256": "a" * 64, "target_from_source_col_major": np.eye(4).flatten(order="F").tolist(),
            "source_camera_calibration_sha256": "d" * 64, "source_world_alignment_sha256": "e" * 64,
            "target_revision_id": "static-v1", "source_floor_plane": {"normal": [0, 1, 0], "offset_m": 0},
            "target_floor_plane": {"normal": [0, 1, 0], "offset_m": 0}}
    manifest = {"source": {"capture_id": TARGET, "model": "prior_conditioned_consensus_da3_carrier"}}
    write(root / "manifest.json", manifest)
    (root / "points.glb").write_bytes(b"retained PCF fixture")
    assets = {role: {"path": str(root / name), "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                     "size_bytes": (root / name).stat().st_size}
              for role, name in (("manifest", "manifest.json"), ("points", "points.glb"))}
    selection["manifest_sha256"] = assets["manifest"]["sha256"]
    reference = {"selection": selection, "frame_binding": edge, "source_scan_id": TARGET,
                 "label": "Noesis PCF · Living Room", "evidence": deepcopy(assets), "assets": assets, "manifest": manifest}
    alignment = args["scan_dir"] / "alignment"
    report = json.loads((alignment / "alignment_report.json").read_text())
    report["target"]["camera_frame_binding"].update(
        contract="noesis.calibration.frame_binding", contract_version=1,
        calibration_frame=edge["source_frame"], target_from_calibration_col_major=edge["target_from_source_col_major"],
        camera_calibration_sha256=edge["source_camera_calibration_sha256"],
        world_alignment_sha256=edge["source_world_alignment_sha256"], target_revision_id="static-v1",
        scene_prior_id=edge["target_frame"]["revision"], calibration_floor_plane=edge["source_floor_plane"], world_floor_plane=edge["target_floor_plane"])
    write(alignment / "alignment_report.json", report)
    transform = json.loads((alignment / "phone_ma_to_noesis_world.json").read_text())
    transform["target_binding"] = report["target"]
    write(alignment / "phone_ma_to_noesis_world.json", transform)
    args["state"]["walk_intent"] = validate_walk_intent({"mode": "path_refinement", "target_scan_id": TARGET,
                                                        "target_reference": selection})
    args["target_state"] = {"id": TARGET, "status": "complete", "alignment": {"status": "failed"},
                            "outputs": {"artifacts": {"manifest": "missing-original.json"}}}
    args["target_dir"] = args["scan_dir"].parent / "target"
    args["target_dir"].mkdir()
    args["target_reference"] = reference
    args["target_binding"] = {"scan_id": TARGET, "manifest_selection": "scene_prior_pcf", "selection": selection}
    return args, reference


def test_comparison_uses_pcf_with_failed_raw_reference_alignment(pcf_handoff):
    args, reference = pcf_handoff
    result = path_comparison.compare_review_path(**args)
    assert result["status"] == "comparison_ready"
    assert result["tracks"][0]["matched_count"] == 40
    assert result["reference"]["reference_registration"]["selection"] == reference["selection"]
    assert result["reference"]["reference_reconstruction"]["manifest_selection"] == "scene_prior_pcf"
    assert args["target_state"]["alignment"]["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("source_camera_calibration_sha256", "f" * 64),
    ("target_from_source_sha256", "f" * 64),
    ("target_revision_id", "different-static"),
    ("target_frame", {"frame_id": "backend_world_m", "revision": "different-world"}),
])
def test_pcf_cannot_override_captured_camera_binding(pcf_handoff, field, value):
    args, reference = pcf_handoff
    reference["frame_binding"][field] = value
    with pytest.raises(ValueError, match="PCF"):
        path_comparison.compare_review_path(**args)
    assert not (args["output_dir"] / "room_path_reference.json").exists()


def build_args(args):
    return {key: value for key, value in args.items() if key not in {"source", "target_binding"}} | {
        "output_dir": args["output_dir"].parent / "new-review"}


def test_review_retains_exact_pcf_assets_and_does_not_read_raw_target(pcf_handoff):
    args, reference = pcf_handoff
    result = path_review.build_path_review(**build_args(args))
    assert result["reference_reconstruction"]["selection"] == reference["selection"]
    assert result["path_comparison"]["status"] == "comparison_ready"
    assert result["path_comparison"]["matched_count"] == 40
    copied = build_args(args)["output_dir"] / result["artifacts"]["reference_pcf_points"]
    assert copied.read_bytes() == b"retained PCF fixture"
    assert result["accuracy"]["qualified"] is False


def test_recorded_pcf_selection_cannot_fall_back_to_raw(pcf_handoff):
    args, _ = pcf_handoff
    values = build_args(args)
    values.pop("target_reference")
    with pytest.raises(ValueError, match="raw output is not a substitute"):
        path_review.build_path_review(**values)
    assert not values["output_dir"].exists()


def test_changed_pcf_copy_is_rejected(pcf_handoff):
    args, reference = pcf_handoff
    from pathlib import Path
    Path(reference["assets"]["points"]["path"]).write_bytes(b"replaced PCF fixture")
    with pytest.raises(ValueError, match="changed before review"):
        path_review.build_path_review(**build_args(args))


@pytest.mark.parametrize("change", [
    {"prior_id": "../wrong"}, {"camera_id": "wrong/path"}, {"manifest_sha256": "no"},
    {"frame_binding_sha256": "a" * 63}, {"kind": "raw"}, {"untrusted_path": "/tmp/wrong"},
])
def test_intent_reference_is_bounded_and_strict(pcf_handoff, change):
    args, reference = pcf_handoff
    value = deepcopy(args["state"]["walk_intent"])
    value["target_reference"].update(change)
    with pytest.raises(ValueError):
        validate_walk_intent(value)


def test_reconstruction_cannot_claim_pcf_path_selection(pcf_handoff):
    _, reference = pcf_handoff
    with pytest.raises(ValueError, match="only valid for path refinement"):
        validate_walk_intent({"mode": "reconstruction", "target_scan_id": TARGET,
                              "target_reference": reference["selection"]})


def test_android_bundle_normalization_preserves_exact_reference(pcf_handoff):
    args, reference = pcf_handoff
    manifest = _manifest()
    manifest["walk_intent"] = args["state"]["walk_intent"]
    normalized = validate_capture_manifest(manifest)
    assert normalized["walk_intent"]["target_reference"] == reference["selection"]


def test_legacy_path_reference_is_not_silently_upgraded(tmp_path, monkeypatch):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    legacy = {"walk_intent": validate_walk_intent({"mode": "path_refinement", "target_scan_id": TARGET})}
    def reject(*args):
        raise AssertionError("Legacy raw selection must not resolve a newly configured PCF")
    monkeypatch.setattr(service.path_references, "resolve", reject)
    assert service._selected_path_reference(legacy, TARGET) is None


def test_api_advertises_and_rechecks_selection_before_queue(pcf_handoff, monkeypatch, tmp_path):
    args, reference = pcf_handoff
    application = app_module.create_app(_settings(tmp_path / "application"))
    service = application.state.phone_scan_service
    source = save_state(service, SOURCE, purpose=args["state"]["walk_intent"])
    save_state(service, TARGET)
    public = {"status": "available", "label": reference["label"], "selection": reference["selection"]}
    monkeypatch.setattr(service.path_references, "for_scan", lambda scan_id: public if scan_id == TARGET else None)
    def reject(scan_id, selection):
        raise ValueError("selected revision changed")
    monkeypatch.setattr(service.path_references, "resolve", reject)
    with TestClient(application) as client:
        target = client.get(f"/api/scans/{TARGET}").json()
        assert target["path_reference"] == public
        listing = client.get("/api/scans").json()
        assert next(s for s in listing if s["id"] == TARGET)["path_reference"] == public
        result = client.post(f"/api/scans/{SOURCE}/path-review")
        assert result.status_code == 409
        assert "selected revision changed" in result.json()["detail"]
        assert "path_review" not in service.read_state(SOURCE)
    assert service._path_review_slots.acquire(blocking=False)
    service._path_review_slots.release()


def test_api_worker_publishes_selected_pcf_without_raw_reference_fit(pcf_handoff, monkeypatch, tmp_path):
    args, reference = pcf_handoff
    application = app_module.create_app(_settings(tmp_path / "application"))
    service = application.state.phone_scan_service
    shutil.copytree(args["scan_dir"], service.scan_dir(SOURCE))
    service._write_state_unlocked(SOURCE, args["state"])
    target = save_state(service, TARGET)
    target["alignment"] = {"status": "failed", "error": "retained raw fit"}
    service._write_state_unlocked(TARGET, target)
    resolved = []
    def resolve(scan_id, selection):
        assert scan_id == TARGET and selection == reference["selection"]
        resolved.append(scan_id)
        return reference
    monkeypatch.setattr(service.path_references, "resolve", resolve)
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{SOURCE}/path-review")
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = client.get(f"/api/scans/{SOURCE}").json()["path_review"]
            if value["status"] in {"complete", "failed"}:
                break
            time.sleep(.01)
        assert value["status"] == "complete", value
        selected = value["results"]["reference_reconstruction"]
        assert selected["selection"] == reference["selection"]
        assert selected["manifest_selection"] == "scene_prior_pcf"
        assert value["results"]["path_comparison"]["status"] == "reference_ready_comparison_needs_evidence"
        artifact = value["results"]["artifact_urls"]["reference_pcf_points"]
        assert client.get(artifact).content == b"retained PCF fixture"
        assert service.read_state(TARGET) == target
    assert len(resolved) == 2  # Admission and worker revalidation.
