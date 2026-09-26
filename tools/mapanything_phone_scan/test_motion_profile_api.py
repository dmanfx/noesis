"""Focused HTTP/import/VIO integration; no native solver or service is started."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from . import app as app_module
from .app import create_app
from .calibration_jobs import CalibrationJobBusy, CalibrationJobError
from .test_android_capture import FRAME_TIMES, _bundle, _manifest, encoded_video
from .test_browser_capture import _settings, _wait_for_ready
from .vio import VIOSettings


CAMERA_ID = "cal-20260915-120000-111111111111"
IMU_ID = "cal-20260915-120000-222222222222"
NOISE_ID = "cal-20260915-120000-333333333333"
MOTION_ID = "cal-20260915-120000-444444444444"
REFERENCES = {"camera_calibration_id": CAMERA_ID, "imu_calibration_id": IMU_ID, "noise_calibration_id": NOISE_ID}
CAMERA_SELECTION = {"camera_calibration_id": CAMERA_ID, "profile_id": "camera-fixture", "binding_sha256": "binding-fixture"}
SELECTION = {"motion_calibration_id": MOTION_ID, "profile_id": "motion-fixture", "status": "ready_for_short_walks", "maximum_duration_s": 300, "camera_selection": CAMERA_SELECTION}


def backend_settings(tmp_path, monkeypatch):
    # Existence/permission fixture only: never executed by these API tests.
    executable = tmp_path / "openvins-fixture"
    executable.write_text("fixture: do not execute")
    executable.chmod(0o700)
    config = tmp_path / "openvins-fixture.yaml"
    config.write_text("fixture: true\n")
    monkeypatch.setenv("NOESIS_PHONE_SCAN_VIO_EXECUTABLE", str(executable))
    monkeypatch.setenv("NOESIS_PHONE_SCAN_VIO_CONFIG", str(config))
    monkeypatch.setenv("NOESIS_PHONE_SCAN_VIO_ESTIMATOR", "openvins")
    return replace(_settings(tmp_path), vio=VIOSettings(executable=executable, config=config))


class MotionFixture:
    def __init__(self):
        self.selection = deepcopy(SELECTION)
        self.calls = []
        self.reject = None

    def get(self):
        return deepcopy(self.selection)

    def select(self, job_id):
        if job_id is not None and job_id != MOTION_ID:
            raise CalibrationJobError("Finish a passing motion-profile check before selecting it")
        self.selection = deepcopy(SELECTION) if job_id else None
        return self.get()

    def derived_report(self, capture_dir, raw, selection):
        assert raw["metric_vio_allowed"] is False  # never the previous derived cache
        assert "motion_profile" not in raw
        self.calls.append((capture_dir, deepcopy(raw), deepcopy(selection)))
        if self.reject:
            raise ValueError(self.reject)
        derived = deepcopy(raw)
        derived["metric_vio_allowed"] = True
        derived["calibration"]["complete_for_metric_vio"] = True
        derived["short_session"] = {"profile_id": selection["profile_id"], "source_and_pose_times_ns": [[t, t + 1000] for t in FRAME_TIMES]}
        return derived


def client_fixture(tmp_path, monkeypatch, *, frame_processor=None, vio_runner=None):
    options = {}
    if frame_processor is not None:
        options["frame_processor"] = frame_processor
    if vio_runner is not None:
        options["vio_runner"] = vio_runner
    application = create_app(backend_settings(tmp_path, monkeypatch), **options)
    service = application.state.phone_scan_service
    motion = MotionFixture()
    service.motion_selection = motion
    camera_calls = []

    def camera_settings(scan_dir, selection, settings):
        camera_calls.append(deepcopy(selection))
        return settings, {**(selection or {}), "status": "applied" if selection else "not_applied", "reason_codes": []}

    monkeypatch.setattr(service.camera_selection, "frame_settings", camera_settings)
    return TestClient(application), service, motion, camera_calls


def upload(client, video, *, asynchronous=False):
    response = client.post("/api/scans/sensor-bundle", content=_bundle(video), headers={
        "Content-Type": "application/zip", "X-File-Name": "motion-walk.zip",
        **({"Prefer": "respond-async"} if asynchronous else {}),
    })
    assert response.status_code == (202 if asynchronous else 201), response.text
    return response.json()


def wait_vio(client, scan_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/api/scans/{scan_id}").json()
        if state.get("vio", {}).get("status") in {"complete", "failed"}:
            return state
        time.sleep(0.02)
    raise AssertionError("VIO fixture did not finish")


def test_motion_routes_submit_exact_ids_select_clear_and_report_capability(tmp_path, monkeypatch):
    client, service, motion, _ = client_fixture(tmp_path, monkeypatch)
    submitted = []

    def submit(payload):
        submitted.append(payload)
        return {"id": MOTION_ID, "mode": "motion_profile", "status": "queued", "request": payload}

    monkeypatch.setattr(service.calibration_jobs, "submit_motion_profile", submit)
    with client:
        health = client.get("/api/health").json()
        assert health["version"] == client.app.version == "1.13.0"
        assert client.get("/openapi.json").json()["info"]["version"] == "1.13.0"
        capability = health["calibration"]
        assert capability["motion_profile_validation_available"] is True
        assert capability["motion_profile_unavailable_reason"] is None
        response = client.post("/api/calibration/motion-profile-jobs", json=REFERENCES)
        assert response.status_code == 202, response.text
        assert submitted == [REFERENCES]
        assert response.json()["mode"] == "motion_profile"
        assert client.get("/api/calibration/motion-selection").json()["selection"] == SELECTION
        assert client.post("/api/calibration/motion-selection", json={"motion_calibration_id": None}).json() == {"selection": None}
        assert client.post("/api/calibration/motion-selection", json={"motion_calibration_id": MOTION_ID}).json()["selection"] == SELECTION
        assert service.camera_selection.get() is None
        assert client.post("/api/calibration/motion-selection", json={"motion_calibration_id": "unqualified"}).status_code == 422


@pytest.mark.parametrize("route", ["motion-profile-jobs", "motion-selection"])
def test_motion_routes_bound_json_and_reject_malformed_shapes(tmp_path, monkeypatch, route):
    client, _, _, _ = client_fixture(tmp_path, monkeypatch)
    with client:
        url = f"/api/calibration/{route}"
        assert client.post(url, content="{}", headers={"Content-Type": "text/plain"}).status_code == 415
        assert client.post(url, content="x" * 1025, headers={"Content-Type": "application/json"}).status_code == 413
        for payload in (None, [], {}, {"unrelated": "id"}, {"motion_calibration_id": 3}, {**REFERENCES, "metric_vio_allowed": True}):
            assert client.post(url, content=json.dumps(payload), headers={"Content-Type": "application/json"}).status_code == 422
        assert client.post(url, content="{", headers={"Content-Type": "application/json"}).status_code == 422


@pytest.mark.parametrize(("failure", "expected"), [(CalibrationJobBusy("Busy"), 409), (FileNotFoundError("Source missing"), 404), (CalibrationJobError("Source mismatch"), 422)])
def test_motion_job_errors_are_actionable_http_failures(tmp_path, monkeypatch, failure, expected):
    client, service, _, _ = client_fixture(tmp_path, monkeypatch)
    def fail(_):
        raise failure
    monkeypatch.setattr(service.calibration_jobs, "submit_motion_profile", fail)
    with client:
        response = client.post("/api/calibration/motion-profile-jobs", json=REFERENCES)
        assert response.status_code == expected
        assert str(failure) in response.json()["detail"]


@pytest.mark.parametrize("missing", ["executable", "config", "permissions", "environment_drift"])
def test_motion_capabilities_match_configured_consumer_without_running_it(tmp_path, monkeypatch, missing):
    client, service, _, _ = client_fixture(tmp_path, monkeypatch)
    if missing in {"executable", "config"}:
        getattr(service.settings.vio, missing).unlink()
    elif missing == "permissions":
        service.settings.vio.executable.chmod(0o600)
    else:
        monkeypatch.setenv("NOESIS_PHONE_SCAN_VIO_CONFIG", str(tmp_path / "different-config.yaml"))
    with client:
        capability = client.get("/api/health").json()["calibration"]
        assert capability["motion_profile_validation_available"] is False
        assert "OpenVINS" in capability["motion_profile_unavailable_reason"]
        assert client.post("/api/calibration/motion-profile-jobs", json=REFERENCES).status_code == 503


def test_new_import_uses_associated_camera_and_derived_report_without_changing_raw(tmp_path, monkeypatch, encoded_video):
    client, service, motion, camera_calls = client_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(service.camera_selection, "get", lambda: {"camera_calibration_id": "unrelated-global"})
    with client:
        initial = upload(client, encoded_video)
        ready = _wait_for_ready(client, initial["id"])
        assert ready["status"] == "ready", ready
        assert ready["capture"]["motion_calibration_selection"] == SELECTION
        assert ready["capture"]["camera_calibration_selection"] == CAMERA_SELECTION
        assert camera_calls == [CAMERA_SELECTION]
        assert ready["capture"]["metric_vio_allowed"] is True
        assert ready["capture"]["motion_profile"]["status"] == "applied"
        capture_dir = service.scan_dir(initial["id"]) / "capture"
        raw = json.loads((capture_dir / "capture_import.json").read_text())
        assert raw["metric_vio_allowed"] is False and "short_session" not in raw
        derived = json.loads((capture_dir / "motion_profile_import.json").read_text())
        assert derived["metric_vio_allowed"] is True
        assert client.get(ready["capture"]["motion_profile_import_url"]).status_code == 200
        assert (capture_dir / "camera.mp4").read_bytes() == encoded_video.read_bytes()
        assert all(frame["capture_time_ns"] == FRAME_TIMES[frame["source_frame_index"]] for frame in ready["prepared"]["frames"])
        original = deepcopy(ready["capture"])
        motion.select(None)
        assert client.get(f"/api/scans/{initial['id']}").json()["capture"] == original


def test_async_import_snapshots_before_queue_and_retry_never_reselects(tmp_path, monkeypatch, encoded_video):
    client, service, motion, camera_calls = client_fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    importer = app_module.import_capture_bundle
    def delayed_import(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return importer(*args, **kwargs)
    monkeypatch.setattr(app_module, "import_capture_bundle", delayed_import)
    with client:
        try:
            initial = upload(client, encoded_video, asynchronous=True)
            assert entered.wait(2)
            assert initial["capture_profile_selection"]["motion_calibration_selection"] == SELECTION
            motion.select(None)
            release.set()
            ready = _wait_for_ready(client, initial["id"])
            assert ready["status"] == "ready", ready
            assert ready["capture"]["motion_calibration_selection"] == SELECTION
            assert camera_calls == [CAMERA_SELECTION]
            deadline = time.monotonic() + 2.0
            while True:
                retry = client.post("/api/scans/sensor-bundle", content=_bundle(encoded_video), headers={"Content-Type": "application/zip", "X-File-Name": "motion-walk.zip", "Prefer": "respond-async"})
                if retry.status_code != 409:
                    break
                assert retry.json().get("detail") == "Previous capture validation is finishing; retry shortly"
                assert time.monotonic() < deadline, retry.text
                time.sleep(0.01)
            assert retry.status_code == 201
            assert retry.json()["id"] == initial["id"]
            assert retry.json()["capture"]["motion_calibration_selection"] == SELECTION
        finally:
            release.set()


@pytest.mark.parametrize("mismatch", ["profile", "camera", "camera_snapshot", "storage"])
def test_motion_failures_do_not_block_rgb_preparation(tmp_path, monkeypatch, encoded_video, mismatch):
    client, service, motion, _ = client_fixture(tmp_path, monkeypatch)
    if mismatch == "profile":
        motion.reject = "The recording's lens does not match the selected profile"
    elif mismatch == "camera":
        def fail_camera(*_):
            raise ValueError("Selected camera evidence hash changed")
        monkeypatch.setattr(service.camera_selection, "frame_settings", fail_camera)
    elif mismatch == "camera_snapshot":
        motion.selection.pop("camera_selection")
    else:
        def fail_write(*_):
            raise OSError("Disk is read-only")
        monkeypatch.setattr(app_module, "_write_calibration_json", fail_write)
    with client:
        initial = upload(client, encoded_video)
        ready = _wait_for_ready(client, initial["id"])
        assert ready["status"] == "ready", ready
        assert ready["prepared"]["frame_count"] >= 2
        assert ready["capture"]["metric_vio_allowed"] is False
        assert ready["capture"]["motion_profile"]["status"] == "not_applied"
        assert client.post(f"/api/scans/{initial['id']}/initiate-vio").status_code == 409
        assert client.get(f"/api/scans/{initial['id']}").json()["status"] == "ready"
        assert json.loads((service.scan_dir(initial["id"]) / "capture/capture_import.json").read_text())["metric_vio_allowed"] is False


def test_vio_worker_rederives_from_raw_and_keeps_exact_prepared_identity(tmp_path, monkeypatch, encoded_video):
    calls = []
    def runner(capture_dir, output_dir, report, prepared, settings, progress):
        calls.append((deepcopy(report), deepcopy(prepared)))
        assert report["metric_vio_allowed"] is True
        assert report["short_session"]["profile_id"] == SELECTION["profile_id"]
        return {"poses": []}
    client, service, motion, _ = client_fixture(tmp_path, monkeypatch, vio_runner=runner)
    with client:
        initial = upload(client, encoded_video)
        ready = _wait_for_ready(client, initial["id"])
        raw_path = service.scan_dir(initial["id"]) / "capture/capture_import.json"
        raw_bytes = raw_path.read_bytes()
        (raw_path.parent / "motion_profile_import.json").write_text('{"metric_vio_allowed": true, "tampered": true}')
        motion.select(None)  # future-import setting must not change this capture
        assert client.post(f"/api/scans/{initial['id']}/initiate-vio").status_code == 202
        final = wait_vio(client, initial["id"])
        assert final["vio"]["status"] == "complete", final
        assert len(motion.calls) == 3  # preparation, admission, actual worker
        assert len(calls) == 1 and "tampered" not in calls[0][0]
        assert calls[0][1] == ready["prepared"] or calls[0][1]["frames"] == service.read_state(initial["id"])["prepared"]["frames"]
        assert raw_path.read_bytes() == raw_bytes
        assert all(frame["capture_time_ns"] == FRAME_TIMES[frame["source_frame_index"]] for frame in calls[0][1]["frames"])


@pytest.mark.parametrize("stage", ["admission", "worker"])
def test_stale_profile_blocks_both_vio_boundaries_even_if_cached_state_was_ready(tmp_path, monkeypatch, encoded_video, stage):
    runner_calls = []
    client, service, motion, _ = client_fixture(tmp_path, monkeypatch, vio_runner=lambda *args: runner_calls.append(args))
    with client:
        initial = upload(client, encoded_video)
        assert _wait_for_ready(client, initial["id"])["capture"]["metric_vio_allowed"] is True
        queued = []
        monkeypatch.setattr(service._executor, "submit", lambda fn, *args: queued.append((fn, args)))
        if stage == "admission":
            motion.reject = "Selected profile hash changed"
        response = client.post(f"/api/scans/{initial['id']}/initiate-vio")
        if stage == "admission":
            assert response.status_code == 409
            assert not queued
        else:
            assert response.status_code == 202
            motion.reject = "Selected profile hash changed after queueing"
            queued[0][0](*queued[0][1])
            assert service.read_state(initial["id"])["vio"]["status"] == "failed"
        assert runner_calls == []
        final = client.get(f"/api/scans/{initial['id']}").json()
        assert final["capture"]["metric_vio_allowed"] is False
        assert final["status"] == "ready"


def test_profile_snapshot_skips_non_native_and_calibration_and_keeps_no_selection_behavior(tmp_path, monkeypatch, encoded_video):
    client, service, motion, _ = client_fixture(tmp_path, monkeypatch)
    assert service._new_capture_profile_selection({"schema": "browser"}) == {}
    assert service._new_capture_profile_selection({**_manifest(), "calibration_request": {"mode": "camera"}}) == {}
    motion.select(None)
    with client:
        initial = upload(client, encoded_video)
        ready = _wait_for_ready(client, initial["id"])
        assert ready["status"] == "ready"
        assert ready["capture"]["metric_vio_allowed"] is False
        assert ready["capture"]["motion_calibration_selection"] is None
        assert "motion_profile" not in ready["capture"]
        assert not (service.scan_dir(initial["id"]) / "capture/motion_profile_import.json").exists()
        assert motion.calls == []


def test_unselected_ordinary_vio_still_consumes_its_original_report(tmp_path, monkeypatch, encoded_video):
    reports = []
    def runner(capture_dir, output_dir, report, prepared, settings, progress):
        reports.append(deepcopy(report))
        return {"poses": []}
    client, service, motion, _ = client_fixture(tmp_path, monkeypatch, vio_runner=runner)
    motion.select(None)
    with client:
        response = client.post("/api/scans/sensor-bundle", content=_bundle(encoded_video, calibrated=True), headers={"Content-Type": "application/zip", "X-File-Name": "calibrated.zip"})
        assert response.status_code == 201, response.text
        scan_id = response.json()["id"]
        ready = _wait_for_ready(client, scan_id)
        raw_path = service.scan_dir(scan_id) / "capture/capture_import.json"
        original = json.loads(raw_path.read_text())
        assert ready["capture"]["metric_vio_allowed"] is True
        assert client.post(f"/api/scans/{scan_id}/initiate-vio").status_code == 202
        assert wait_vio(client, scan_id)["vio"]["status"] == "complete"
        assert reports == [original]
        assert motion.calls == []
        assert not (raw_path.parent / "motion_profile_import.json").exists()


def test_unreadable_selector_retains_new_rgb_capture_without_global_camera_fallback(tmp_path, monkeypatch, encoded_video):
    client, service, motion, camera_calls = client_fixture(tmp_path, monkeypatch)
    def unreadable():
        raise ValueError("Motion selection JSON is damaged")
    monkeypatch.setattr(motion, "get", unreadable)
    monkeypatch.setattr(service.camera_selection, "get", lambda: pytest.fail("Do not substitute global camera selection"))
    with client:
        assert client.get("/api/calibration/motion-selection").status_code == 422
        initial = upload(client, encoded_video)
        ready = _wait_for_ready(client, initial["id"])
        assert ready["status"] == "ready"
        assert ready["capture"]["metric_vio_allowed"] is False
        assert "damaged" in ready["capture"]["motion_profile"]["message"]
        assert camera_calls == []


@pytest.mark.parametrize("outcome", ["legacy", "accepted", "rejected", "pending"])
def test_default_vio_adapter_preserves_native_quality_ownership_and_never_grants_acceptance(tmp_path, monkeypatch, outcome):
    from .motion_profile import validate_short_walk_runtime
    from .test_capture_vio import _valid_vio_result
    from .vio import POSE_TIME_REFERENCE, SHORT_CONSUMER_VERSION, SHORT_IMAGE_MODEL, VIOError

    short = outcome != "legacy"
    native = _valid_vio_result()
    native["accepted_for_metric_vio"] = not short
    template = native["poses"][0]
    native["poses"] = [
        {**deepcopy(template), "capture_time_ns": (i + 1) * 1_000_000_000,
         "pose_time_ns": (i + 1) * 1_000_000_000 + 7_000_000,
         "source_frame_index": i, "prepared_frame_id": f"capture:source:{i}"}
        for i in range(3)
    ]
    envelope = {"maximum_rotation_during_exposure_rad": 0.01,
                "maximum_rotation_during_readout_rad": 0.02}
    short_metadata = {
        "consumer_version": SHORT_CONSUMER_VERSION, "profile_id": "checked-profile",
        "validation_run": False, "profile_validation_sha256": "a" * 64,
        "motion_envelope": deepcopy(envelope), "validated_motion_envelope": envelope,
    }
    if outcome == "rejected":
        short_metadata["motion_envelope"]["maximum_rotation_during_readout_rad"] = 0.03
    if short:
        # The real native wrapper now supplies this direct-consumer admission
        # independently of the parent motion-envelope check below.
        native["status"] = "completed"
        native["direct_runtime_quality"] = {"accepted": outcome == "accepted"}
        native["frame"].update(pose_time_reference=POSE_TIME_REFERENCE,
                               capture_time_reference="original_camera_sensor_timestamp")
        native["short_session_consumer"] = {
            "consumer_version": SHORT_CONSUMER_VERSION, "profile_id": "checked-profile",
            "validation_run": False, "fixed_camera_imu_calibration": True,
            "native_imu_corrections": "identity", "dense_frame_mapping_verified": True,
            "image_motion_model": SHORT_IMAGE_MODEL, "rolling_shutter_compensated": False,
        }
    source_times = [row["capture_time_ns"] for row in native["poses"]]
    pose_times = [row["pose_time_ns"] for row in native["poses"]]
    prepared = {"frames": [
        {"index": 7 + i, "source_frame_index": i, "capture_time_ns": source_times[i], "sha256": str(i) * 64}
        for i in (0, 2)
    ]}
    report = {"short_session": short_metadata} if short else {}
    generated_config = tmp_path / "generated-config.yaml"
    materialized = tmp_path / "materialized"
    settings = VIOSettings(config=tmp_path / "base-config.yaml")
    # Materialization/native execution have separate contract tests. Exercise the
    # real parent quality callback and final result validation at this adapter.
    monkeypatch.setattr(app_module, "validate_vio_input", lambda *_: None)
    monkeypatch.setattr(app_module, "materialize_openvins_input", lambda *_: (materialized, generated_config))
    native_calls = []
    quality_calls = []

    def check_arguments(root, output, configured, progress):
        assert root == materialized and output == tmp_path / "output"
        assert configured.config == generated_config
        assert settings.config == tmp_path / "base-config.yaml"
        native_calls.append(True)

    def legacy_native(root, output, configured, progress):
        check_arguments(root, output, configured, progress)
        return native  # Deliberately accepts no keyword arguments.

    def short_native(root, output, configured, progress):
        check_arguments(root, output, configured, progress)
        # Native owns the mandatory parent policy. The app must not supply an
        # extra callback that duplicates it or changes the legacy signature.
        if outcome != "pending":
            quality = validate_short_walk_runtime(native, {"short_session": short_metadata, "frame_mapping": native["poses"]})
            quality_calls.append((len(native["poses"]), quality))
            if quality["accepted"] is not True:
                raise VIOError("short-profile runtime quality control rejected this walk")
            native.update(accepted_for_metric_vio=True, runtime_quality_control=quality)
        return native

    monkeypatch.setattr(app_module, "run_openvins", short_native if short else legacy_native)
    arguments = (tmp_path / "capture", tmp_path / "output", report, prepared, settings, lambda *_: None)
    if outcome in {"rejected", "pending"}:
        with pytest.raises(VIOError, match="runtime quality control rejected|not accepted for metric use"):
            app_module._default_vio_runner(*arguments)
        assert native["accepted_for_metric_vio"] is False
    else:
        result = app_module._default_vio_runner(*arguments)
        assert result["accepted_for_metric_vio"] is True
        assert result["quality"]["dense_pose_count"] == 3
        assert result["quality"]["prepared_pose_count"] == 2
        dense = json.loads((tmp_path / "output/dense_camera_trajectory.json").read_text())
        assert [row["capture_time_ns"] for row in dense["poses"]] == source_times
        assert len(dense["poses"]) == 3
        assert [r["capture_time_ns"] for r in result["poses"]] == [source_times[i] for i in (0, 2)]
        assert [r["pose_time_ns"] for r in result["poses"]] == [pose_times[i] for i in (0, 2)]
        assert [r["source_frame_index"] for r in result["poses"]] == [0, 2]
        assert [r["prepared_frame_id"] for r in result["poses"]] == [
            app_module.prepared_frame_identity(row["index"], row["sha256"]) for row in prepared["frames"]]
    assert native_calls == [True]
    assert len(quality_calls) == (1 if outcome in {"accepted", "rejected"} else 0)
    if quality_calls:
        assert quality_calls[0][0] == 3  # Dense quality precedes prepared-view selection.
        assert quality_calls[0][1]["accuracy_validated"] is False
