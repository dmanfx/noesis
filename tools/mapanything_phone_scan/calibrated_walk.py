"""Explicit, per-import camera selection; no live calibration or VIO admission."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import threading
import zipfile

import numpy as np

from .calibration_jobs import CalibrationJobError, _now, _read, _write
from .phone_calibration import import_calibration_bundle, load_phone_calibration
from .roomwalk_calibration import camera_binding_for_capture, camera_profile, _load_camera_reference


def _bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


class CameraSelection:
    def __init__(self, jobs):
        self.jobs = jobs
        self.path = jobs.root / "camera-selection.json"
        self.lock = threading.RLock()

    def get(self):
        with self.lock:
            return _read(self.path, 8192).get("selection") if self.path.is_file() else None

    def select(self, job_id):
        with self.lock:
            if job_id is None:
                _write(self.path, {"selection": None})
                return None
            if not isinstance(job_id, str):
                raise CalibrationJobError("Select a qualified camera job ID")
            job = self.jobs.get(job_id)
            if job["mode"] != "camera" or job["status"] != "completed" or (job.get("result") or {}).get("camera_intrinsics_calibrated") is not True:
                raise CalibrationJobError("Camera job is not qualified for reconstruction intrinsics")
            result = _load_camera_reference(self.jobs.artifact(job_id, "report.json").parent)
            expected = camera_profile(result)
            exported = _read(self.jobs.artifact(job_id, "camera_profile.json"))
            if exported != expected:
                raise CalibrationJobError("Camera profile no longer matches its qualified result")
            profile_path = self._materialize(job_id, result, exported)
            selection = {"camera_calibration_id": job_id, "profile_id": load_phone_calibration(profile_path)["profile_id"],
                         "selected_at": _now(), "binding_sha256": result["binding"]["sha256"]}
            _write(self.path, {"selection": selection})
            return selection

    def _materialize(self, job_id, result, exported):
        # Adapt the qualified result through the existing hash-verified intrinsic
        # bundle importer, preserving the original fit and its validation report.
        camera = {"camera_id": "roomwalk-native", "session": job_id,
                  "model": "opencv_pinhole", "resolution": result["resolution"],
                  "K": result["K"], "D": result["D"],
                  "runtime": {"use_undistorted_frames": False},
                  "status": "qualified", "warnings": ["Camera intrinsics only; metric VIO and live world admission remain separate"],
                  "independent_video_validation": {"mean_error_px": result["quality"]["heldout"]["radial_px"].get("mean"), "pose_group_count": len(result["holdout_frame_indices"])}}
        mode = {"camera_id": camera["camera_id"], "calibration_resolution": camera["resolution"],
                "recorder": "RoomWalk Camera2", "binding": result["binding"]}
        numeric = io.BytesIO()
        np.savez(numeric, camera_matrix=np.asarray(camera["K"], dtype=float), distortion_coefficients=np.asarray(camera["D"], dtype=float), image_size=np.asarray(camera["resolution"]))
        contents = {"camera_noesis.json": _bytes(camera), "capture_mode.json": _bytes(mode),
                    "camera_opencv.npz": numeric.getvalue(), "camera_result.json": _bytes(result),
                    "camera_profile.json": _bytes(exported)}
        manifest = {"camera_id": camera["camera_id"], "session": job_id, "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in contents.items()}}
        # Stable archive bytes make retries compare equal to retained provenance.
        contents["manifest.json"] = _bytes(manifest)
        archive = self.jobs.root / job_id / "camera-handoff.zip"
        if not archive.exists():
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as handle:
                for name, data in contents.items():
                    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                    handle.writestr(info, data)
        profile = import_calibration_bundle(archive, self.jobs.root / "camera-profiles")
        retained = _read(profile.parent / "source/camera_result.json")
        if retained != result:
            raise CalibrationJobError("Retained handoff differs from current camera evidence")
        return profile

    def frame_settings(self, scan_dir: Path, selection, settings):
        """Selection is snapshotted at import; later choices cannot rewrite scans."""
        if not selection:
            return settings, None
        job_id = selection["camera_calibration_id"]
        # Reverify the selected evidence and native optical binding on every use.
        job = self.jobs.get(job_id)
        result = _load_camera_reference(self.jobs.artifact(job_id, "report.json").parent)
        expected = camera_profile(result)
        if job["status"] != "completed" or _read(self.jobs.artifact(job_id, "camera_profile.json")) != expected:
            raise CalibrationJobError("Selected camera evidence is no longer qualified")
        binding = camera_binding_for_capture(scan_dir / "capture")
        summary = {**selection, "accepted_for_metric_vio": False,
                   "status": "not_applied", "reason_codes": list(binding["reason_codes"])}
        if binding["sha256"] != selection["binding_sha256"] or binding["sha256"] != result["binding"]["sha256"]:
            summary["reason_codes"].append("actual_camera_binding_mismatch")
        if summary["reason_codes"]:
            # A mismatched native profile must never fall through to a different
            # globally configured calibration. RGB reconstruction remains usable.
            return replace(settings, phone_camera_calibration=None, phone_camera_capture_mode="unbound"), summary
        profile = self.jobs.root / "camera-profiles" / "roomwalk-native" / job_id / "profile.json"
        loaded = load_phone_calibration(profile)
        if loaded["capture_mode"]["binding"] != binding:
            raise CalibrationJobError("Materialized profile optical binding differs")
        summary.update(status="applied", reason_codes=[])
        return replace(settings, phone_camera_calibration=profile, phone_camera_capture_mode="native_sensor_bundle"), summary
