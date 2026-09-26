"""Bounded, cancellable offline calibration jobs; never activate a live calibration."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import quote


JOB_PATTERN = re.compile(r"^cal-[0-9]{8}-[0-9]{6}-[a-f0-9]{12}$")
ACTIVE = {"queued", "running", "cancelling"}
MAX_REQUEST_BYTES = 64 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024


class CalibrationJobError(ValueError):
    pass


class CalibrationJobBusy(CalibrationJobError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    encoded = json.dumps(payload, allow_nan=False, indent=2).encode()
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _read(path: Path, maximum: int = MAX_REPORT_BYTES) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > maximum:
        raise CalibrationJobError("Calibration metadata exceeds its bound or is not a regular file")
    def reject(value: str) -> None:
        raise ValueError(f"Invalid JSON number: {value}")
    data = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(data, dict):
        raise CalibrationJobError("Calibration metadata must be an object")
    return data


class CalibrationJobs:
    """One CPU child process, two queued requests, and at most 128 retained jobs."""

    def __init__(self, storage_root: Path, *, timeout_s: float = 1800,
                 execute: Callable[[Path, threading.Event], None] | None = None) -> None:
        self.root = storage_root / ".roomwalk-calibration"
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = max(1.0, min(3600.0, float(timeout_s)))
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(3)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RoomWalkCalibration")
        self._cancel: dict[str, threading.Event] = {}
        self._execute = execute or self._run_process
        self._closed = False

    def _directory(self, job_id: str) -> Path:
        if not JOB_PATTERN.fullmatch(job_id):
            raise CalibrationJobError("Invalid calibration job ID")
        directory = self.root / job_id
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError("Calibration job was not found")
        return directory

    def recover(self) -> None:
        with self._lock:
            for job in self.list():
                if job["status"] in ACTIVE:
                    self._update(job["id"], status="failed", message="Calibration was interrupted by a server restart; original evidence is retained")

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for directory in self.root.iterdir():
            if JOB_PATTERN.fullmatch(directory.name) and directory.is_dir() and not directory.is_symlink():
                try:
                    rows.append(self.get(directory.name))
                except (OSError, ValueError):
                    continue
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = _read(self._directory(job_id) / "state.json")
        public = {key: deepcopy(value) for key, value in row.items() if not key.startswith("_")}
        public["artifacts"] = {}
        directory = self._directory(job_id)
        # Only known output files are surfaced; never traverse a model or recording tree.
        for name in ("report.json", "request.json", "source_manifest.json", "camera_result.json", "camera_profile.json", "camera_imu_result.json", "noise_result.json", "noise_allan.png", "motion_profile.json", "motion_validation.json", "profile_vio_result.json", "validation.json", "observations.json", "solver_input.json", "solver_result.json"):
            if (directory / "artifacts" / name).is_file():
                public["artifacts"][Path(name).stem + "_url"] = f"/api/calibration/jobs/{job_id}/files/{quote(name)}"
        if (directory / "worker.log").is_file():
            public["log_url"] = f"/api/calibration/jobs/{job_id}/log"
        progress = directory / "progress.json"
        if row["status"] == "running" and progress.is_file():
            try:
                latest = _read(progress, 8192)
                public["progress"] = max(0.0, min(1.0, float(latest.get("progress", 0))))
                public["message"] = str(latest.get("message", "Processing calibration"))[:1000]
            except (OSError, ValueError):
                pass
        return public

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            path = self._directory(job_id) / "state.json"
            row = _read(path)
            row.update(updates, updated_at=_now())
            _write(path, row)

    def submit(self, scan_id: str, capture_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
        from .roomwalk_calibration import validate_calibration_request
        request = validate_calibration_request(payload)
        if not (capture_dir / "capture_import.json").is_file():
            raise CalibrationJobError("Import a native RoomWalk video and IMU bundle before calibration")
        camera_id = payload.get("camera_calibration_id")
        if request["mode"] == "imu" and not request["allow_provisional_camera"]:
            if not camera_id:
                raise CalibrationJobError("Choose a qualified camera calibration before processing motion; your uploaded recording is retained")
            if not request["board_geometry_confirmed"]:
                raise CalibrationJobError("Confirm the measured printed board dimensions before processing metric motion; your uploaded recording is retained")
        settings: dict[str, Any] = {}
        if camera_id:
            camera = self.get(str(camera_id))
            if camera["mode"] != "camera" or camera["status"] != "completed":
                raise CalibrationJobError("Select a completed camera calibration job")
            if request["mode"] == "imu" and not request["allow_provisional_camera"] and camera.get("result", {}).get("camera_intrinsics_calibrated") is not True:
                raise CalibrationJobError("Select a qualified camera calibration, not merely a completed or rejected camera job")
            settings["camera_calibration_dir"] = str(self._directory(str(camera_id)) / "artifacts")
            request["camera_calibration_id"] = str(camera_id)
        noise_id = payload.get("noise_calibration_id")
        if noise_id:
            noise = self.get(str(noise_id))
            result = noise.get("result") or {}
            if noise["mode"] != "noise" or noise["status"] != "completed" or not (
                result.get("imu_noise_calibrated") is True or result.get("noise_model_usable") is True
            ):
                raise CalibrationJobError("Process a usable one-minute stationary sensor recording first")
            settings["noise_calibration_dir"] = str(self._directory(str(noise_id)) / "artifacts")
            request["noise_calibration_id"] = str(noise_id)
        solver = os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER", "").strip()
        if solver:
            path = Path(solver).resolve(strict=True)
            if not path.is_file() or not os.access(path, os.X_OK):
                raise CalibrationJobError("The configured calibration solver is not executable")
            settings["solver_path"] = str(path)
        return self._submit(scan_id, capture_dir, request, settings)

    def list_imu_recordings(self) -> list[dict[str, Any]]:
        root = self.root.parent / ".imu-calibration"
        if not root.is_dir():
            return []
        rows = []
        for path in root.iterdir():
            if path.is_dir() and not path.is_symlink() and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", path.name):
                try:
                    rows.append(_read(path / "receipt.json", 64 * 1024))
                except (OSError, ValueError):
                    continue
        return sorted(rows, key=lambda row: row["capture_id"], reverse=True)[:32]

    def submit_noise(self, capture_id: str) -> dict[str, Any]:
        if not isinstance(capture_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", capture_id):
            raise CalibrationJobError("Select a retained stationary IMU capture")
        directory = self.root.parent / ".imu-calibration" / capture_id
        if directory.is_symlink() or not (directory / "receipt.json").is_file():
            raise FileNotFoundError("Stationary IMU capture was not found")
        return self._submit(None, directory, {"schema": "roomwalk.noise_request.v1", "mode": "noise", "method": "short_session", "imu_capture_id": capture_id}, {})

    def submit_motion_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Check a fixed short-session profile on withheld board motion with OpenVINS."""
        fields = {"camera_calibration_id": "camera", "imu_calibration_id": "imu", "noise_calibration_id": "noise"}
        if not isinstance(payload, dict) or set(payload) != set(fields):
            raise CalibrationJobError("Choose the camera, camera–IMU, and stationary sensor results")
        request = {"schema": "roomwalk.motion_profile_request.v1", "mode": "motion_profile"}
        settings: dict[str, Any] = {}
        selected = {}
        for field, mode in fields.items():
            value = payload[field]
            if not isinstance(value, str):
                raise CalibrationJobError("Calibration result IDs must be strings")
            job = self.get(value)
            if job["mode"] != mode or job["status"] != "completed":
                raise CalibrationJobError(f"Finish the {mode} calibration step first")
            request[field] = value
            selected[mode] = job
            settings[mode + "_calibration_dir"] = str(self._directory(value) / "artifacts")
        from .motion_profile import assemble_motion_profile
        # Reject mismatched or incomplete choices before occupying a worker.
        assemble_motion_profile(**{key: Path(value) for key, value in settings.items()})
        from .vio import VIOSettings
        vio = VIOSettings.from_env()
        if vio.executable is None or not vio.executable.is_file() or not os.access(vio.executable, os.X_OK):
            raise CalibrationJobError("The RoomWalk server needs its OpenVINS runner configured before checking motion profiles")
        if vio.config is None or not vio.config.is_file():
            raise CalibrationJobError("The RoomWalk server needs its OpenVINS configuration before checking motion profiles")
        settings.update(vio_executable=str(vio.executable.resolve()), vio_config=str(vio.config.resolve()))
        imu_state = _read(self._directory(request["imu_calibration_id"]) / "state.json")
        return self._submit(selected["imu"].get("scan_id"), Path(imu_state["_capture_dir"]), request, settings)

    def _submit(self, scan_id: str | None, capture_dir: Path, request: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        import uuid
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                raise CalibrationJobBusy("Calibration queue is full; wait for a running job or cancel one")
            try:
                if len(self.list()) >= 128:
                    raise CalibrationJobBusy("Calibration history is full; retained jobs need review")
                job_id = datetime.now(timezone.utc).strftime("cal-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:12]
                directory = self.root / job_id
                directory.mkdir(mode=0o700)
                request_bytes = json.dumps(request, sort_keys=True, allow_nan=False).encode()
                state = {"schema": "roomwalk.calibration_job.v1", "id": job_id, "scan_id": scan_id,
                         "mode": request["mode"], "request": request, "status": "queued", "progress": 0.0,
                         "message": "Queued for offline calibration", "created_at": _now(), "updated_at": _now(),
                         "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                         "_capture_dir": str(capture_dir.resolve()), "_settings": settings}
                _write(directory / "state.json", state)
                cancel = threading.Event()
                self._cancel[job_id] = cancel
                self._executor.submit(self._work, job_id, cancel)
            except Exception:
                self._slots.release()
                raise
            return self.get(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            state = self.get(job_id)
            if state["status"] in ACTIVE:
                event = self._cancel.get(job_id)
                if event is None:
                    raise CalibrationJobError("This job is no longer owned by this server process")
                event.set()
                self._update(job_id, status="cancelling", message="Stopping calibration; raw evidence is retained")
            return self.get(job_id)

    def _work(self, job_id: str, cancel: threading.Event) -> None:
        try:
            if cancel.is_set():
                self._update(job_id, status="cancelled", message="Cancelled before processing")
                return
            self._update(job_id, status="running", message="Reading native camera and motion evidence")
            directory = self._directory(job_id)
            self._execute(directory, cancel)
            if cancel.is_set():
                self._update(job_id, status="cancelled", message="Calibration cancelled; source files and partial results are retained")
            else:
                result = _read(directory / "artifacts/report.json")
                outcome = result.get("status", "completed")
                if outcome not in {"completed", "failed", "cancelled"}:
                    raise CalibrationJobError("Calibration processor returned an unknown execution status")
                self._update(job_id, status=outcome, progress=1.0, result=result,
                             message="Processing finished; review the qualification checks before use" if outcome == "completed" else "Calibration did not complete; review retained diagnostics")
        except Exception as exc:
            self._update(job_id, status="cancelled" if cancel.is_set() else "failed", message=str(exc)[:2000], error=type(exc).__name__)
        finally:
            with self._lock:
                self._cancel.pop(job_id, None)
            self._slots.release()

    def _run_process(self, directory: Path, cancel: threading.Event) -> None:
        environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        root = Path(__file__).resolve().parents[2]
        environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
        log = directory / "worker.log"
        with log.open("wb") as output:
            process = subprocess.Popen([sys.executable, "-m", "tools.mapanything_phone_scan.calibration_jobs", "--run-job", str(directory)],
                                       cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + self.timeout_s
            try:
                while process.poll() is None:
                    if cancel.wait(0.2) or time.monotonic() > deadline or log.stat().st_size > 8 * 1024 * 1024:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                        if not cancel.is_set():
                            raise CalibrationJobError("Calibration exceeded its time or log bound; partial results are retained")
                        return
                if process.returncode:
                    raise CalibrationJobError(f"Calibration processor exited with code {process.returncode}; inspect its retained log")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)

    def artifact(self, job_id: str, name: str) -> Path:
        root = self._directory(job_id) / "artifacts"
        target = root / name
        if not name or target.is_symlink() or root.is_symlink() or not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            raise FileNotFoundError("Calibration artifact was not found")
        return target

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for event in self._cancel.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=False)


def _run_job(directory: Path) -> None:
    state = _read(directory / "state.json")
    def progress(fraction: float, message: str) -> None:
        _write(directory / "progress.json", {"progress": float(fraction), "message": str(message)[:1000]})
    settings = dict(state.get("_settings") or {})
    for name in ("camera_calibration_dir", "imu_calibration_dir", "noise_calibration_dir", "solver_path", "vio_executable", "vio_config"):
        if settings.get(name):
            settings[name] = Path(settings[name])
    output = directory / "artifacts"
    if state["mode"] == "noise":
        from .imu_noise_calibration import run_noise_calibration
        result = run_noise_calibration(Path(state["_capture_dir"]), output, progress=progress)
    elif state["mode"] == "motion_profile":
        from .motion_profile import run_motion_profile
        result = run_motion_profile(Path(state["_capture_dir"]), state["request"], output, settings=settings, progress=progress)
    else:
        import cv2
        # This is an isolated offline child, not the long-lived app process.
        # OpenCV's pthread pool ignores OMP_NUM_THREADS and otherwise uses every
        # host CPU. Keep native-resolution detection bounded and deterministic.
        cv2.setNumThreads(2)
        from .roomwalk_calibration import run_calibration
        if state["mode"] == "imu":
            # Dense 90-second 8K target localization needs a larger bounded
            # budget than the sparse camera fit, below the parent's 30 minutes.
            settings.setdefault("timeout_s", 1500)
        result = run_calibration(Path(state["_capture_dir"]), state["request"], output,
                                 settings=settings, progress=progress, cancelled=lambda: False)
    _write(output / "report.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-job", type=Path, required=True)
    _run_job(parser.parse_args().run_job)
