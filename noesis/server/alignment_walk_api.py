from __future__ import annotations

import asyncio
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from noesis.server.internal_auth import InternalAuthConfig
from noesis.validation.alignment_walk import (
    AlignmentWalkCapture,
    append_waypoint_marker,
    create_run_directory,
    default_output_root,
    file_sha256,
    read_ndjson_private,
    validate_scene_binding,
    validate_waypoint_manifest,
    verify_run,
    write_waypoint_calibration_report,
)
from noesis_core.private_paths import read_private_file, validate_private_file


_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_TRACKLET_KEY_RE = re.compile(r"tracklet-[1-9][0-9]{0,9}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVE_TRACK_MAX_AGE_US = 5_000_000
_MAX_JSON_ARTIFACT_BYTES = 32 * 1024 * 1024


class SceneBindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    release_id: str = Field(min_length=1, max_length=200)
    authored_scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_to_scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WaypointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=120)
    camera_id: str = Field(min_length=1, max_length=160)
    label: str | None = Field(default=None, max_length=240)
    rooms: list[str] = Field(default_factory=list, max_length=32)
    expected_scene_xyz: tuple[float, float, float]
    split: Literal["fit", "holdout"]
    pause_s: float | None = Field(default=None, gt=0.0, le=120.0)


class WaypointManifestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: Literal["noesis.alignment.walk_waypoints"]
    contract_version: Literal[1]
    waypoints: list[WaypointRequest] = Field(min_length=1, max_length=128)


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    duration_s: int = Field(default=1800, ge=30, le=3600)
    scene_binding: SceneBindingRequest
    waypoints: WaypointManifestRequest


class MarkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    waypoint_id: str = Field(min_length=1, max_length=120)
    phase: Literal["arrived", "leave"] = "arrived"
    actor: str | None = Field(default=None, max_length=80)
    tracklet_key: str | None = Field(
        default=None,
        pattern=r"^tracklet-[1-9][0-9]{0,9}$",
    )


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    window_before_s: float = Field(default=2.0, gt=0.0, le=30.0)
    window_after_s: float = Field(default=3.0, gt=0.0, le=60.0)
    ambiguity_margin: float = Field(default=0.12, ge=0.0, le=1.0)
    good_error_m: float = Field(default=0.5, gt=0.0, le=10.0)
    fail_error_m: float = Field(default=1.0, gt=0.0, le=20.0)


class AlignmentWalkApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = str(detail)


@dataclass
class _SessionState:
    session_id: str
    run_dir: Path
    capture: AlignmentWalkCapture
    duration_s: int
    started_at_us: int
    deadline_at_us: int
    waypoint_manifest: dict[str, Any]
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    status: str = "recording"
    error: str | None = None
    stop_as_failed: bool = False
    latest_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    websocket_connected: bool = False
    reconnect_count: int = 0
    analysis_summary: dict[str, Any] | None = None
    markers: list[dict[str, Any]] = field(default_factory=list)


def _json_model(value: BaseModel | None) -> dict[str, Any] | None:
    return value.model_dump(mode="json") if value is not None else None


def _waypoint_manifest(request: SessionCreateRequest) -> dict[str, Any]:
    waypoints = []
    for waypoint in request.waypoints.waypoints:
        payload = waypoint.model_dump(mode="json", exclude_none=True)
        expected = payload["expected_scene_xyz"]
        payload["expected_scene_xz"] = [expected[0], expected[2]]
        waypoints.append(payload)
    return validate_waypoint_manifest(
        {
            "contract": request.waypoints.contract,
            "contract_version": request.waypoints.contract_version,
            "waypoints": waypoints,
        }
    )


def _safe_self_ws_uri(host: str, port: int) -> str:
    normalized = str(host or "").strip().strip("[]")
    if normalized in {"", "0.0.0.0", "::", "*"}:
        normalized = "127.0.0.1"
    rendered = f"[{normalized}]" if ":" in normalized else normalized
    return f"ws://{rendered}:{int(port)}"


def _load_private_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(
        read_private_file(
            path,
            label=label,
            max_bytes=_MAX_JSON_ARTIFACT_BYTES,
        ).decode("utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _load_private_json_value(path: Path, *, label: str) -> Any:
    return json.loads(
        read_private_file(
            path,
            label=label,
            max_bytes=_MAX_JSON_ARTIFACT_BYTES,
        ).decode("utf-8")
    )


class AlignmentWalkController:
    """Own one bounded, private alignment-walk capture at a time."""

    def __init__(
        self,
        *,
        ws_host: str,
        ws_port: int,
        auth_config: InternalAuthConfig,
        output_root: str | Path | None = None,
        websocket_connect: Callable[..., Any] | None = None,
    ) -> None:
        self.ws_uri = _safe_self_ws_uri(ws_host, ws_port)
        self.auth_config = auth_config
        self.output_root = Path(output_root or default_output_root())
        self._websocket_connect = websocket_connect
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        self._current: _SessionState | None = None

    def _connector(self) -> Callable[..., Any]:
        if self._websocket_connect is not None:
            return self._websocket_connect
        import websockets

        return websockets.connect

    def _session_id(self, requested: str | None) -> str:
        if requested is not None:
            if _SESSION_ID_RE.fullmatch(requested) is None:
                raise AlignmentWalkApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "session_id contains unsafe characters",
                )
            return requested
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return f"alignment-walk-{timestamp}-{secrets.token_hex(4)}"

    def create(self, request: SessionCreateRequest) -> dict[str, Any]:
        manifest = _waypoint_manifest(request)
        scene_binding = validate_scene_binding(_json_model(request.scene_binding))
        with self._lock:
            if (
                self._current is not None
                and self._current.status in {"recording", "stopping"}
                and self._current.thread is not None
                and self._current.thread.is_alive()
            ):
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT,
                    "an alignment-walk session is already active",
                )
            session_id = self._session_id(request.session_id)
            try:
                run_dir = create_run_directory(self.output_root, session_id)
                capture = AlignmentWalkCapture(
                    run_dir=run_dir,
                    run_id=session_id,
                    ws_uri=self.ws_uri,
                    scene_binding=scene_binding,
                )
                capture.copy_waypoints(manifest)
            except FileExistsError as exc:
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT, str(exc)
                ) from exc
            except (RuntimeError, ValueError) as exc:
                raise AlignmentWalkApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)
                ) from exc
            started_at_us = time.time_ns() // 1_000
            state_value = _SessionState(
                session_id=session_id,
                run_dir=run_dir,
                capture=capture,
                duration_s=int(request.duration_s),
                started_at_us=started_at_us,
                deadline_at_us=(
                    started_at_us + int(request.duration_s) * 1_000_000
                ),
                waypoint_manifest=manifest,
            )
            thread = threading.Thread(
                target=self._run_capture_thread,
                args=(state_value,),
                name=f"alignment-walk-{session_id}",
                daemon=True,
            )
            state_value.thread = thread
            self._current = state_value
            thread.start()
            return self._status_payload(state_value)

    def _authorization_headers(self) -> dict[str, str] | None:
        if self.auth_config.mode != "required":
            return None
        if not self.auth_config.token:
            raise RuntimeError(
                "required internal auth has no bearer token for alignment capture"
            )
        return {"Authorization": f"Bearer {self.auth_config.token}"}

    async def _capture_stream(self, state_value: _SessionState) -> None:
        deadline = time.monotonic() + float(state_value.duration_s)
        consecutive_failures = 0
        while not state_value.stop_event.is_set() and time.monotonic() < deadline:
            try:
                headers = self._authorization_headers()
                connect_args: dict[str, Any] = {
                    "max_size": None,
                    "max_queue": 16,
                    "open_timeout": 5.0,
                    "close_timeout": 2.0,
                    "ping_interval": 20.0,
                    "ping_timeout": 20.0,
                    "proxy": None,
                }
                if headers:
                    connect_args["additional_headers"] = headers
                async with self._connector()(self.ws_uri, **connect_args) as websocket:
                    with self._lock:
                        state_value.websocket_connected = True
                    received_on_connection = False
                    while (
                        not state_value.stop_event.is_set()
                        and time.monotonic() < deadline
                    ):
                        try:
                            raw = await asyncio.wait_for(
                                websocket.recv(), timeout=0.5
                            )
                        except asyncio.TimeoutError:
                            continue
                        if isinstance(raw, (bytes, bytearray)):
                            continue
                        try:
                            message = json.loads(raw)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if not isinstance(message, Mapping):
                            continue
                        received_on_connection = True
                        rows = state_value.capture.process_message(
                            message,
                            received_at_us=time.time_ns() // 1_000,
                            received_monotonic_ns=time.monotonic_ns(),
                        )
                        self._update_latest_tracks(
                            state_value, message=message, rows=rows
                        )
                    if received_on_connection:
                        consecutive_failures = 0
            except Exception:
                if state_value.stop_event.is_set():
                    break
                consecutive_failures += 1
                with self._lock:
                    state_value.websocket_connected = False
                    state_value.reconnect_count += 1
                if consecutive_failures >= 4:
                    raise
                await asyncio.sleep(min(1.0, 0.2 * (2**consecutive_failures)))
            finally:
                with self._lock:
                    state_value.websocket_connected = False
        state_value.stop_event.set()

    def _run_capture_thread(self, state_value: _SessionState) -> None:
        final_status = "failed" if state_value.stop_as_failed else "complete"
        error: str | None = state_value.error
        try:
            asyncio.run(self._capture_stream(state_value))
        except BaseException as exc:
            final_status = "failed"
            error = str(exc)[:500] or exc.__class__.__name__
        if state_value.stop_as_failed:
            final_status = "failed"
            error = state_value.error or error or "alignment capture interrupted"
        try:
            state_value.capture.finish(status=final_status, error=error)
            session = _load_private_json(
                state_value.run_dir / "session.json",
                label="alignment capture session",
            )
            final_status = str(session.get("status") or final_status)
            error = str(session.get("error") or error or "").strip() or None
        except BaseException as exc:
            final_status = "failed"
            error = str(exc)[:500] or exc.__class__.__name__
        with self._lock:
            state_value.status = final_status
            state_value.error = error
            state_value.websocket_connected = False

    def _update_latest_tracks(
        self,
        state_value: _SessionState,
        *,
        message: Mapping[str, Any],
        rows: list[dict[str, Any]],
    ) -> None:
        camera_ids = {
            str(row.get("camera_id"))
            for row in rows
            if str(row.get("camera_id") or "").strip()
        }
        if not camera_ids:
            camera = str(message.get("camera_id") or "").strip()
            if camera:
                camera_ids.add(camera)
        active_keys = {
            str(row.get("tracklet_key"))
            for row in rows
            if str(row.get("tracklet_key") or "").strip()
        }
        with self._lock:
            for key, track in list(state_value.latest_tracks.items()):
                if (
                    str(track.get("camera_id")) in camera_ids
                    and key not in active_keys
                ):
                    state_value.latest_tracks.pop(key, None)
            for row in rows:
                key = str(row.get("tracklet_key") or "")
                if _TRACKLET_KEY_RE.fullmatch(key) is None:
                    continue
                state_value.latest_tracks[key] = {
                    field_name: row.get(field_name)
                    for field_name in (
                        "tracklet_key",
                        "camera_id",
                        "frame_id",
                        "received_at_us",
                        "received_monotonic_ns",
                        "bbox_xywh",
                        "image_foot",
                        "image_size",
                        "confidence",
                        "tracker_confidence",
                    )
                }

    def _active_tracks(self, state_value: _SessionState) -> list[dict[str, Any]]:
        now_us = time.time_ns() // 1_000
        tracks: list[dict[str, Any]] = []
        for key, track in list(state_value.latest_tracks.items()):
            received_at_us = int(track.get("received_at_us") or 0)
            age_us = max(0, now_us - received_at_us)
            if received_at_us <= 0 or age_us > _ACTIVE_TRACK_MAX_AGE_US:
                state_value.latest_tracks.pop(key, None)
                continue
            tracks.append({**track, "age_ms": float(age_us / 1000.0)})
        return sorted(
            tracks,
            key=lambda item: (
                str(item.get("camera_id") or ""),
                str(item.get("tracklet_key") or ""),
            ),
        )

    @staticmethod
    def _marker_view(marker: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: marker.get(key)
            for key in (
                "waypoint_id",
                "phase",
                "tracklet_key",
                "recorded_at_us",
            )
            if marker.get(key) is not None
        }

    def _waypoint_views(
        self,
        manifest: Mapping[str, Any],
        markers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest_by_id = {
            str(marker.get("waypoint_id")): marker
            for marker in markers
            if str(marker.get("waypoint_id") or "").strip()
        }
        result: list[dict[str, Any]] = []
        for waypoint in manifest.get("waypoints", []):
            if not isinstance(waypoint, Mapping):
                continue
            waypoint_id = str(waypoint.get("id") or "")
            marker = latest_by_id.get(waypoint_id)
            result.append(
                {
                    key: value
                    for key, value in {
                        "id": waypoint_id,
                        "camera_id": waypoint.get("camera_id"),
                        "label": waypoint.get("label"),
                        "split": waypoint.get("split"),
                        "state": (
                            str(marker.get("phase"))
                            if marker is not None
                            else "pending"
                        ),
                        "marker": (
                            self._marker_view(marker)
                            if marker is not None
                            else None
                        ),
                    }.items()
                    if value is not None
                }
            )
        return result

    def _status_payload(self, state_value: _SessionState) -> dict[str, Any]:
        with self._lock:
            markers = [
                self._marker_view(marker)
                for marker in state_value.markers[-256:]
            ]
            return {
                "session_id": state_value.session_id,
                "state": state_value.status,
                "status": state_value.status,
                "error": state_value.error,
                "duration_s": state_value.duration_s,
                "started_at_us": state_value.started_at_us,
                "deadline_at_us": state_value.deadline_at_us,
                "websocket_connected": state_value.websocket_connected,
                "reconnect_count": state_value.reconnect_count,
                "counts": dict(sorted(state_value.capture.counts.items())),
                "scene_binding": state_value.capture.scene_binding,
                "scene_binding_verification": (
                    state_value.capture.scene_binding_verification
                ),
                "markers": markers,
                "waypoints": self._waypoint_views(
                    state_value.waypoint_manifest, markers
                ),
                "active_tracks": self._active_tracks(state_value),
                "analysis": state_value.analysis_summary,
            }

    def _disk_session(self, session_id: str) -> tuple[Path, dict[str, Any]]:
        if _SESSION_ID_RE.fullmatch(str(session_id)) is None:
            raise AlignmentWalkApiError(
                status.HTTP_404_NOT_FOUND, "alignment-walk session not found"
            )
        run_dir = self.output_root / str(session_id)
        try:
            payload = _load_private_json(
                run_dir / "session.json", label="alignment capture session"
            )
        except Exception as exc:
            raise AlignmentWalkApiError(
                status.HTTP_404_NOT_FOUND, "alignment-walk session not found"
            ) from exc
        if payload.get("run_id") != session_id:
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment-walk session identity does not match its directory",
            )
        return run_dir, payload

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if self._current is not None and self._current.session_id == session_id:
                return self._status_payload(self._current)
        run_dir, session = self._disk_session(session_id)
        try:
            marker_rows = read_ndjson_private(run_dir / "markers.ndjson")
            markers = [
                self._marker_view(marker)
                for marker in marker_rows[-256:]
            ]
            manifest = _load_private_json(
                run_dir / "waypoints.json",
                label="alignment waypoint manifest",
            )
        except Exception as exc:
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment-walk marker state is invalid",
            ) from exc
        state_name = str(session.get("status") or "unknown")
        return {
            "session_id": session_id,
            "state": state_name,
            "status": state_name,
            "error": session.get("error"),
            "started_at_us": session.get("started_at_us"),
            "counts": session.get("counts", {}),
            "runtime": session.get("runtime", {}),
            "scene_binding": session.get("scene_binding"),
            "scene_binding_verification": session.get(
                "scene_binding_verification"
            ),
            "markers": markers,
            "waypoints": self._waypoint_views(manifest, markers),
            "active_tracks": [],
            "analysis": None,
        }

    def marker(self, session_id: str, request: MarkerRequest) -> dict[str, Any]:
        if request.phase == "arrived" and request.tracklet_key is None:
            raise AlignmentWalkApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "tracklet_key is required for an arrived marker",
            )
        with self._lock:
            state_value = self._current
            if (
                state_value is None
                or state_value.session_id != session_id
                or state_value.status != "recording"
            ):
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT,
                    "alignment-walk session is not actively recording",
                )
        try:
            marker = append_waypoint_marker(
                state_value.run_dir,
                waypoint_id=request.waypoint_id,
                phase=request.phase,
                actor=request.actor,
                tracklet_key=request.tracklet_key,
            )
        except ValueError as exc:
            raise AlignmentWalkApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)
            ) from exc
        except RuntimeError as exc:
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT, str(exc)
            ) from exc
        marker_view = self._marker_view(marker)
        with self._lock:
            state_value.markers.append(marker_view)
        return {
            "session_id": session_id,
            "state": state_value.status,
            "status": state_value.status,
            "marker": marker_view,
        }

    def finish(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            state_value = self._current
            if state_value is None or state_value.session_id != session_id:
                return self.status(session_id)
            if state_value.status not in {"recording", "stopping"}:
                return self._status_payload(state_value)
            state_value.status = "stopping"
            state_value.stop_event.set()
            thread = state_value.thread
        if thread is not None:
            thread.join(timeout=8.0)
            if thread.is_alive():
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT,
                    "alignment-walk capture did not stop within its bound",
                )
        return self._status_payload(state_value)

    def analyze(
        self, session_id: str, request: AnalysisRequest
    ) -> dict[str, Any]:
        run_dir, session = self._disk_session(session_id)
        if session.get("status") != "complete":
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "only a complete alignment-walk session can be analyzed",
            )
        verification = verify_run(run_dir)
        if not verification.get("ok"):
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment-walk source evidence failed verification",
            )
        if request.good_error_m >= request.fail_error_m:
            raise AlignmentWalkApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "good_error_m must be less than fail_error_m",
            )
        if not self._analysis_lock.acquire(blocking=False):
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment-walk analysis is already in progress",
            )
        try:
            try:
                result = write_waypoint_calibration_report(
                    run_dir,
                    window_before_s=request.window_before_s,
                    window_after_s=request.window_after_s,
                    ambiguity_margin=request.ambiguity_margin,
                    good_error_m=request.good_error_m,
                    fail_error_m=request.fail_error_m,
                )
            except (RuntimeError, ValueError) as exc:
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT, str(exc)
                ) from exc
        finally:
            self._analysis_lock.release()
        summary = {
            key: value
            for key, value in result.items()
            if key not in {"report", "markdown"}
        }
        with self._lock:
            if self._current is not None and self._current.session_id == session_id:
                self._current.analysis_summary = summary
        return {"session_id": session_id, **summary}

    def results(self, session_id: str) -> dict[str, Any]:
        with self._analysis_lock:
            return self._results_unlocked(session_id)

    def _results_unlocked(self, session_id: str) -> dict[str, Any]:
        run_dir, session = self._disk_session(session_id)
        if session.get("status") != "complete":
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment-walk results are unavailable before capture completes",
            )
        verification = verify_run(run_dir)
        if not verification.get("ok"):
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment-walk source evidence failed verification",
            )
        index_path = run_dir / "waypoint_calibration_artifact_index.json"
        try:
            index = _load_private_json(
                index_path, label="waypoint calibration artifact index"
            )
        except Exception as exc:
            raise AlignmentWalkApiError(
                status.HTTP_404_NOT_FOUND,
                "alignment-walk analysis results are not available",
            ) from exc
        expected_files = {
            "waypoint_calibration_evidence.json": None,
            "waypoint_calibration_metrics.json": "metrics",
            "waypoint_candidate_similarity.json": "candidate_similarity",
            "waypoint_camera_calibration_candidates.json": "camera_candidates",
            "waypoint_calibration_report.json": "report",
            "waypoint_calibration_report.md": None,
        }
        artifacts = (
            index.get("artifacts")
            if isinstance(index.get("artifacts"), Mapping)
            else {}
        )
        result_artifacts: dict[str, Any] = {}
        for filename, response_key in expected_files.items():
            metadata = (
                artifacts.get(filename)
                if isinstance(artifacts.get(filename), Mapping)
                else {}
            )
            expected_digest = str(metadata.get("sha256") or "").lower()
            path = run_dir / filename
            try:
                validate_private_file(path, label="alignment result artifact")
            except Exception as exc:
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT,
                    f"alignment result artifact is invalid: {filename}",
                ) from exc
            if (
                _SHA256_RE.fullmatch(expected_digest) is None
                or not secrets.compare_digest(file_sha256(path), expected_digest)
            ):
                raise AlignmentWalkApiError(
                    status.HTTP_409_CONFLICT,
                    f"alignment result artifact digest mismatch: {filename}",
                )
            if response_key is not None:
                result_artifacts[response_key] = _load_private_json_value(
                    path, label="alignment result artifact"
                )
        source_index_digest = str(
            index.get("source_capture_artifact_index_sha256") or ""
        ).lower()
        if (
            _SHA256_RE.fullmatch(source_index_digest) is None
            or not secrets.compare_digest(
                source_index_digest, file_sha256(run_dir / "artifact_index.json")
            )
        ):
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment results do not bind to the current capture evidence",
            )
        raw_report = (
            result_artifacts.get("report")
            if isinstance(result_artifacts.get("report"), Mapping)
            else {}
        )
        checks = []
        for check in raw_report.get("checks", []):
            if not isinstance(check, Mapping):
                continue
            checks.append(
                {
                    key: check.get(key)
                    for key in (
                        "id",
                        "domain",
                        "name",
                        "status",
                        "failure_type",
                        "camera",
                        "detail",
                    )
                    if check.get(key) is not None
                }
            )
        raw_candidate = (
            result_artifacts.get("candidate_similarity")
            if isinstance(
                result_artifacts.get("candidate_similarity"), Mapping
            )
            else {}
        )
        candidate_similarity = {
            key: raw_candidate.get(key)
            for key in (
                "status",
                "reason",
                "advisory_only",
                "active_config_modified",
                "fit_waypoint_count",
                "similarity",
                "sha256",
            )
            if raw_candidate.get(key) is not None
        }
        raw_candidate_summary = (
            raw_candidate.get("candidate_summary")
            if isinstance(raw_candidate.get("candidate_summary"), Mapping)
            else {}
        )
        if (
            raw_candidate_summary.get("fit_solver_status")
            not in {"admissible", "blocked"}
            or raw_candidate_summary.get("calibration_candidate_status")
            not in {"admissible", "rejected", "warning", "blocked"}
            or raw_candidate_summary.get("advisory_only") is not True
            or raw_candidate_summary.get("active_config_modified") is not False
        ):
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "alignment result is missing its authoritative candidate summary",
            )
        candidate_summary = {
            key: raw_candidate_summary.get(key)
            for key in (
                "fit_solver_status",
                "calibration_candidate_status",
                "advisory_only",
                "active_config_modified",
                "camera_candidate_status",
                "advisory_similarity_status",
            )
            if raw_candidate_summary.get(key) is not None
        }
        raw_camera_candidates = (
            result_artifacts.get("camera_candidates")
            if isinstance(result_artifacts.get("camera_candidates"), Mapping)
            else {}
        )
        camera_candidates: dict[str, Any] = {}
        raw_cameras = (
            raw_camera_candidates.get("cameras")
            if isinstance(raw_camera_candidates.get("cameras"), Mapping)
            else {}
        )
        for camera_id, camera in raw_cameras.items():
            if not isinstance(camera, Mapping):
                continue
            rotation = (
                camera.get("rotation")
                if isinstance(camera.get("rotation"), Mapping)
                else {}
            )
            depth = (
                camera.get("depth_registration")
                if isinstance(camera.get("depth_registration"), Mapping)
                else {}
            )
            camera_candidates[str(camera_id)] = {
                "rotation": {
                    key: rotation.get(key)
                    for key in (
                        "status",
                        "admission_status",
                        "reason",
                        "admission_rejection_reasons",
                        "fit_waypoint_count",
                        "fit_waypoint_ids",
                        "holdout_used_by_solver",
                        "candidate_E_world_to_camera_col_major",
                        "fixed_camera_center_backend_world_m",
                        "camera_center_preservation_error_m",
                        "fit_metrics",
                        "candidate_sha256",
                        "source_bindings",
                    )
                    if rotation.get(key) is not None
                },
                "depth_registration": {
                    key: depth.get(key)
                    for key in (
                        "status",
                        "admission_status",
                        "reason",
                        "fit_pair_count",
                        "fit_waypoint_ids",
                        "holdout_used_by_solver",
                        "raw_domain_m",
                        "knots_raw_m",
                        "knots_physical_optical_depth_m",
                        "fit_coverage",
                        "fit_error_m",
                        "candidate_sha256",
                        "source_bindings",
                    )
                    if depth.get(key) is not None
                },
                "metrics": camera.get("metrics", {}),
            }
        payload = {
            "session_id": session_id,
            "state": "results_ready",
            "status": "results_ready",
            "scene_binding": session.get("scene_binding"),
            "scene_binding_verification": session.get(
                "scene_binding_verification"
            ),
            "source_verified": True,
            "report": {
                key: raw_report.get(key)
                for key in (
                    "run_id",
                    "status",
                    "level",
                    "scope",
                    "summary",
                )
                if raw_report.get(key) is not None
            },
            "checks": checks,
            "metrics": result_artifacts.get("metrics", {}),
            "candidate_similarity": candidate_similarity,
            "candidate_summary": candidate_summary,
            "camera_candidates": camera_candidates,
        }
        encoded_size = len(
            json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        if encoded_size > 1_800_000:
            raise AlignmentWalkApiError(
                status.HTTP_409_CONFLICT,
                "sanitized alignment result exceeds the gateway response bound",
            )
        return payload

    def close(self) -> None:
        with self._lock:
            state_value = self._current
            if (
                state_value is None
                or state_value.status not in {"recording", "stopping"}
            ):
                return
            state_value.stop_as_failed = True
            state_value.error = "runtime shutdown interrupted alignment capture"
            state_value.status = "stopping"
            state_value.stop_event.set()
            thread = state_value.thread
        if thread is not None:
            thread.join(timeout=8.0)


router = APIRouter(prefix="/api/v1/alignment-walk", tags=["alignment-walk"])


def _controller(request: Request) -> AlignmentWalkController:
    controller = getattr(request.app.state, "alignment_walk_controller", None)
    if not isinstance(controller, AlignmentWalkController):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="alignment-walk controller is unavailable",
        )
    return controller


def _translate(exc: AlignmentWalkApiError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreateRequest, request: Request) -> dict[str, Any]:
    try:
        return _controller(request).create(payload)
    except AlignmentWalkApiError as exc:
        raise _translate(exc) from exc


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request) -> dict[str, Any]:
    try:
        return _controller(request).status(session_id)
    except AlignmentWalkApiError as exc:
        raise _translate(exc) from exc


@router.post("/sessions/{session_id}/markers")
def add_marker(
    session_id: str, payload: MarkerRequest, request: Request
) -> dict[str, Any]:
    try:
        return _controller(request).marker(session_id, payload)
    except AlignmentWalkApiError as exc:
        raise _translate(exc) from exc


@router.post("/sessions/{session_id}/finish")
def finish_session(session_id: str, request: Request) -> dict[str, Any]:
    try:
        return _controller(request).finish(session_id)
    except AlignmentWalkApiError as exc:
        raise _translate(exc) from exc


@router.post("/sessions/{session_id}/analysis")
def analyze_session(
    session_id: str, payload: AnalysisRequest, request: Request
) -> dict[str, Any]:
    try:
        return _controller(request).analyze(session_id, payload)
    except AlignmentWalkApiError as exc:
        raise _translate(exc) from exc


@router.get("/sessions/{session_id}/results")
def get_results(session_id: str, request: Request) -> dict[str, Any]:
    try:
        return _controller(request).results(session_id)
    except AlignmentWalkApiError as exc:
        raise _translate(exc) from exc


def install_alignment_walk_api(
    app: Any,
    *,
    ws_host: str,
    ws_port: int,
    auth_config: InternalAuthConfig,
    output_root: str | Path | None = None,
    websocket_connect: Callable[..., Any] | None = None,
) -> AlignmentWalkController:
    controller = AlignmentWalkController(
        ws_host=ws_host,
        ws_port=ws_port,
        auth_config=auth_config,
        output_root=output_root,
        websocket_connect=websocket_connect,
    )
    app.state.alignment_walk_controller = controller
    app.include_router(router)
    app.add_event_handler("shutdown", controller.close)
    return controller


__all__ = [
    "AlignmentWalkController",
    "AnalysisRequest",
    "MarkerRequest",
    "SceneBindingRequest",
    "SessionCreateRequest",
    "WaypointRequest",
    "install_alignment_walk_api",
    "router",
]
