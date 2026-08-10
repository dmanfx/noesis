from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .alignment import NoesisAlignmentSettings, run_noesis_alignment
from .da3_inference import DA3PhoneScanSettings, run_da3_phone_scan
from .inference import MapAnythingScanSettings, run_mapanything_scan
from .processing import FramePreparationSettings, prepare_video_frames


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
SCAN_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp"}
INFERENCE_PROVIDERS = {"mapanything", "da3"}
RUNNING_STATUSES = {
    "processing_frames",
    "ma_queued",
    "ma_running",
    "da3_queued",
    "da3_running",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _resolve_root(raw: str | None, default: Path) -> Path:
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class PhoneScanSettings:
    storage_root: Path
    static_root: Path
    three_root: Path
    max_upload_bytes: int
    frame: FramePreparationSettings
    mapanything: MapAnythingScanSettings
    da3: DA3PhoneScanSettings
    alignment: NoesisAlignmentSettings

    @classmethod
    def from_env(cls) -> "PhoneScanSettings":
        three_root = _resolve_root(
            os.environ.get("NOESIS_PHONE_SCAN_THREE_ROOT"),
            REPO_ROOT / "oai2-fe" / "node_modules" / "three",
        )
        point_budget = _env_int(
            "NOESIS_PHONE_SCAN_POINT_BUDGET", 600_000, minimum=10_000
        )
        artifact_root_raw = str(os.environ.get("NOESIS_DS9_ARTIFACT_ROOT") or "").strip()
        default_da3_engine = (
            Path(artifact_root_raw)
            / "models"
            / "engines"
            / "da3metric_large_294x518_b3_fp16_trt10.13.engine"
            if artifact_root_raw
            else REPO_ROOT
            / "data"
            / "ds9_artifacts"
            / "models"
            / "engines"
            / "da3metric_large_294x518_b3_fp16_trt10.13.engine"
        )
        alignment_camera_id = os.environ.get(
            "NOESIS_PHONE_SCAN_ALIGNMENT_CAMERA_ID", "living-room"
        ).strip()
        alignment_revision = _resolve_root(
            os.environ.get("NOESIS_PHONE_SCAN_ALIGNMENT_REVISION"),
            REPO_ROOT
            / "data"
            / "virtual_twin"
            / "revisions"
            / "vt_living_room_stream_rgbmesh_20260623T215821_538294633",
        )
        anchor_enabled = os.environ.get(
            "NOESIS_PHONE_SCAN_STATIC_ANCHOR", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        anchor_image = (
            _resolve_root(
                os.environ.get("NOESIS_PHONE_SCAN_STATIC_ANCHOR_IMAGE"),
                alignment_revision
                / "keyframes"
                / f"{alignment_camera_id}_0001.png",
            )
            if anchor_enabled
            else None
        )
        return cls(
            storage_root=_resolve_root(
                os.environ.get("NOESIS_PHONE_SCAN_STORAGE_ROOT"),
                REPO_ROOT / "data" / "mapanything_phone_scans",
            ),
            static_root=APP_ROOT / "static",
            three_root=three_root,
            max_upload_bytes=_env_int(
                "NOESIS_PHONE_SCAN_MAX_UPLOAD_BYTES",
                8 * 1024 * 1024 * 1024,
                minimum=1024 * 1024,
            ),
            frame=FramePreparationSettings(
                target_fps=_env_float("NOESIS_PHONE_SCAN_TARGET_FPS", 2.0, minimum=0.1),
                max_frames=_env_int("NOESIS_PHONE_SCAN_MAX_FRAMES", 48, minimum=2),
                max_edge_px=_env_int("NOESIS_PHONE_SCAN_MAX_EDGE_PX", 1920, minimum=518),
            ),
            mapanything=MapAnythingScanSettings(
                model_id=os.environ.get(
                    "NOESIS_PHONE_SCAN_MODEL_ID", "facebook/map-anything-apache"
                ).strip(),
                device=os.environ.get("NOESIS_PHONE_SCAN_MA_DEVICE", "cuda:0").strip(),
                amp_dtype=os.environ.get("NOESIS_PHONE_SCAN_MA_AMP_DTYPE", "bf16").strip(),
                point_budget=point_budget,
                local_files_only=os.environ.get(
                    "NOESIS_PHONE_SCAN_LOCAL_FILES_ONLY", "1"
                ).strip().lower()
                not in {"0", "false", "no", "off"},
                anchor_image=anchor_image,
            ),
            da3=DA3PhoneScanSettings(
                model_id=os.environ.get(
                    "NOESIS_PHONE_SCAN_DA3_MODEL_ID", "depth-anything/DA3-BASE"
                ).strip(),
                device=os.environ.get("NOESIS_PHONE_SCAN_DA3_DEVICE", "cuda:0").strip(),
                process_res=_env_int(
                    "NOESIS_PHONE_SCAN_DA3_PROCESS_RES", 504, minimum=294
                ),
                ref_view_strategy=os.environ.get(
                    "NOESIS_PHONE_SCAN_DA3_REF_VIEW", "middle"
                ).strip(),
                point_budget=point_budget,
                local_files_only=os.environ.get(
                    "NOESIS_PHONE_SCAN_LOCAL_FILES_ONLY", "1"
                ).strip().lower()
                not in {"0", "false", "no", "off"},
                metric_engine_path=_resolve_root(
                    os.environ.get("NOESIS_PHONE_SCAN_DA3_ENGINE"),
                    default_da3_engine,
                ),
                anchor_image=anchor_image,
            ),
            alignment=NoesisAlignmentSettings(
                camera_id=alignment_camera_id,
                target_revision=alignment_revision,
                calibration_path=_resolve_root(
                    os.environ.get("NOESIS_PHONE_SCAN_ALIGNMENT_CALIBRATION"),
                    REPO_ROOT / "config" / "camera_calibration.json",
                ),
                review_point_budget=point_budget,
            ),
        )


FrameProcessor = Callable[
    [Path, Path, FramePreparationSettings, Callable[[float, str], None]],
    dict[str, Any],
]
InferenceRunner = Callable[
    [Path, Path, dict[str, Any], MapAnythingScanSettings, Callable[[float, str], None]],
    dict[str, Any],
]
DA3InferenceRunner = Callable[
    [Path, Path, dict[str, Any], DA3PhoneScanSettings, Callable[[float, str], None]],
    dict[str, Any],
]
AlignmentRunner = Callable[
    [Path, Path, dict[str, Any], NoesisAlignmentSettings, Callable[[float, str], None]],
    dict[str, Any],
]


class PhoneScanService:
    def __init__(
        self,
        settings: PhoneScanSettings,
        *,
        frame_processor: FrameProcessor = prepare_video_frames,
        inference_runner: InferenceRunner = run_mapanything_scan,
        da3_inference_runner: DA3InferenceRunner = run_da3_phone_scan,
        alignment_runner: AlignmentRunner = run_noesis_alignment,
    ) -> None:
        self.settings = settings
        self.frame_processor = frame_processor
        self.inference_runner = inference_runner
        self.da3_inference_runner = da3_inference_runner
        self.alignment_runner = alignment_runner
        self.settings.storage_root.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._scan_locks: dict[str, threading.RLock] = {}
        self._inference_lock = threading.Lock()
        self._alignment_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PhoneScan")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _lock(self, scan_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._scan_locks.setdefault(scan_id, threading.RLock())

    def scan_dir(self, scan_id: str) -> Path:
        if not SCAN_ID_PATTERN.fullmatch(scan_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found")
        candidate = (self.settings.storage_root / scan_id).resolve()
        try:
            candidate.relative_to(self.settings.storage_root)
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found") from exc
        return candidate

    def _state_path(self, scan_id: str) -> Path:
        return self.scan_dir(scan_id) / "scan_state.json"

    def _read_state_unlocked(self, scan_id: str) -> dict[str, Any]:
        state_path = self._state_path(scan_id)
        if not state_path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Scan state is unreadable") from exc
        if not isinstance(state, dict) or state.get("id") != scan_id:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Scan state is invalid")
        return state

    def _write_state_unlocked(self, scan_id: str, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now()
        path = self._state_path(scan_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def read_state(self, scan_id: str) -> dict[str, Any]:
        with self._lock(scan_id):
            return self._read_state_unlocked(scan_id)

    def update_state(self, scan_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            state.update(changes)
            self._write_state_unlocked(scan_id, state)
            return state

    def recover_interrupted_states(self) -> None:
        for directory in self.settings.storage_root.iterdir():
            if not directory.is_dir() or not SCAN_ID_PATTERN.fullmatch(directory.name):
                continue
            scan_id = directory.name
            try:
                state = self.read_state(scan_id)
            except HTTPException:
                continue
            previous = str(state.get("status") or "")
            if previous == "processing_frames":
                self.update_state(
                    scan_id,
                    status="frame_failed",
                    progress=0.0,
                    message="Frame preparation was interrupted by a tool restart",
                    error="Restart the scan by recording or uploading the video again.",
                )
            elif previous in {"ma_queued", "ma_running", "da3_queued", "da3_running"}:
                provider = "da3" if previous.startswith("da3_") else "mapanything"
                label = "DA3" if provider == "da3" else "MapAnything"
                self.update_state(
                    scan_id,
                    status=f"{'da3' if provider == 'da3' else 'ma'}_failed",
                    progress=0.0,
                    message=f"{label} was interrupted by a tool restart",
                    error="The uploaded video and prepared frames are preserved; retry the selected provider.",
                )
            alignment = state.get("alignment")
            if isinstance(alignment, dict) and alignment.get("status") in {
                "queued",
                "running",
            }:
                alignment.update(
                    {
                        "status": "failed",
                        "progress": 0.0,
                        "message": "Noesis alignment was interrupted by a tool restart",
                        "error": "The original inference outputs are preserved; press Align to Noesis to retry.",
                    }
                )
                self.update_state(scan_id, alignment=alignment)

    def list_states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for directory in self.settings.storage_root.iterdir():
            if not directory.is_dir() or not SCAN_ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                states.append(self.read_state(directory.name))
            except HTTPException:
                continue
        states.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return states

    def submit_frame_preparation(self, scan_id: str) -> None:
        self._executor.submit(self._prepare_worker, scan_id)

    def _progress(self, scan_id: str, fraction: float, message: str) -> None:
        try:
            self.update_state(
                scan_id,
                progress=float(min(1.0, max(0.0, fraction))),
                message=str(message),
            )
        except HTTPException:
            return

    def _prepare_worker(self, scan_id: str) -> None:
        try:
            state = self.read_state(scan_id)
            scan_dir = self.scan_dir(scan_id)
            video_path = scan_dir / str(state["video"]["path"])
            prepared = self.frame_processor(
                video_path,
                scan_dir,
                self.settings.frame,
                lambda fraction, message: self._progress(scan_id, fraction, message),
            )
            self.update_state(
                scan_id,
                status="ready",
                progress=1.0,
                message=f"{prepared['frame_count']} views are ready for MapAnything or DA3",
                error=None,
                prepared=prepared,
            )
        except Exception as exc:
            self.update_state(
                scan_id,
                status="frame_failed",
                progress=0.0,
                message="Frame preparation failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def initiate_mapanything(self, scan_id: str) -> dict[str, Any]:
        return self.initiate_inference(scan_id, "mapanything")

    def initiate_inference(self, scan_id: str, provider: str) -> dict[str, Any]:
        provider = str(provider).strip().lower()
        if provider not in INFERENCE_PROVIDERS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Unsupported provider {provider!r}; choose mapanything or da3",
            )
        prefix = "ma" if provider == "mapanything" else "da3"
        label = "MapAnything" if provider == "mapanything" else "DA3"
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if state.get("status") not in {"ready", f"{prefix}_failed"}:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"Scan cannot initiate {label} while status is {state.get('status')}",
                )
            if not isinstance(state.get("prepared"), dict):
                raise HTTPException(status.HTTP_409_CONFLICT, "Prepared frames are missing")
            state.update(
                {
                    "status": f"{prefix}_queued",
                    "progress": 0.0,
                    "message": f"Waiting for the {label} GPU lane",
                    "error": None,
                    "provider": provider,
                }
            )
            self._write_state_unlocked(scan_id, state)
        self._executor.submit(self._inference_worker, scan_id, provider)
        return state

    def _inference_worker(self, scan_id: str, provider: str) -> None:
        prefix = "ma" if provider == "mapanything" else "da3"
        label = "MapAnything" if provider == "mapanything" else "DA3"
        runner = self.inference_runner if provider == "mapanything" else self.da3_inference_runner
        provider_settings = (
            self.settings.mapanything if provider == "mapanything" else self.settings.da3
        )
        with self._inference_lock:
            scan_dir = self.scan_dir(scan_id)
            build_dir = scan_dir / ".outputs-building"
            final_dir = scan_dir / "outputs"
            try:
                state = self.update_state(
                    scan_id,
                    status=f"{prefix}_running",
                    progress=0.01,
                    message=f"Starting {label}",
                )
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                build_dir.mkdir(parents=True, exist_ok=False)
                result = runner(
                    scan_dir,
                    build_dir,
                    state["prepared"],
                    provider_settings,
                    lambda fraction, message: self._progress(scan_id, fraction, message),
                )
                if final_dir.exists():
                    raise RuntimeError("a completed output directory already exists")
                os.replace(build_dir, final_dir)
                self.update_state(
                    scan_id,
                    status="complete",
                    progress=1.0,
                    message=f"{label} finished; all scan assets are saved",
                    error=None,
                    outputs=result,
                    provider=provider,
                )
            except Exception as exc:
                self.update_state(
                    scan_id,
                    status=f"{prefix}_failed",
                    progress=0.0,
                    message=f"{label} failed; the video and prepared frames are preserved",
                    error=f"{type(exc).__name__}: {exc}",
                )

    def initiate_alignment(self, scan_id: str) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if state.get("status") != "complete" or not isinstance(state.get("outputs"), dict):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Noesis alignment requires completed multi-view outputs",
                )
            alignment = state.get("alignment")
            alignment_status = alignment.get("status") if isinstance(alignment, dict) else None
            if alignment_status not in {None, "failed"}:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"Noesis alignment cannot start while status is {alignment_status}",
                )
            state["alignment"] = {
                "status": "queued",
                "progress": 0.0,
                "message": "Waiting for the Noesis alignment lane",
                "error": None,
                "target_camera_id": self.settings.alignment.camera_id,
                "target_revision_id": self.settings.alignment.target_revision.name,
            }
            self._write_state_unlocked(scan_id, state)
        self._executor.submit(self._alignment_worker, scan_id)
        return state

    def _alignment_progress(self, scan_id: str, fraction: float, message: str) -> None:
        try:
            with self._lock(scan_id):
                state = self._read_state_unlocked(scan_id)
                alignment = state.get("alignment")
                if not isinstance(alignment, dict):
                    return
                alignment.update(
                    {
                        "progress": float(min(1.0, max(0.0, fraction))),
                        "message": str(message),
                    }
                )
                state["alignment"] = alignment
                self._write_state_unlocked(scan_id, state)
        except HTTPException:
            return

    def _alignment_worker(self, scan_id: str) -> None:
        with self._alignment_lock:
            scan_dir = self.scan_dir(scan_id)
            build_dir = scan_dir / ".alignment-building"
            final_dir = scan_dir / "alignment"
            try:
                state = self.read_state(scan_id)
                alignment = dict(state.get("alignment") or {})
                alignment.update(
                    {
                        "status": "running",
                        "progress": 0.01,
                        "message": "Starting gravity-preserving Noesis alignment",
                        "error": None,
                    }
                )
                self.update_state(scan_id, alignment=alignment)
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                build_dir.mkdir(parents=True, exist_ok=False)
                result = self.alignment_runner(
                    scan_dir,
                    build_dir,
                    state["outputs"],
                    self.settings.alignment,
                    lambda fraction, message: self._alignment_progress(
                        scan_id, fraction, message
                    ),
                )
                if final_dir.exists():
                    raise RuntimeError("a completed alignment directory already exists")
                os.replace(build_dir, final_dir)
                alignment.update(
                    {
                        "status": "complete",
                        "progress": 1.0,
                        "message": "Phone reconstruction is aligned to the living-room Noesis world",
                        "error": None,
                        "results": result,
                    }
                )
                self.update_state(scan_id, alignment=alignment)
            except Exception as exc:
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                try:
                    state = self.read_state(scan_id)
                    alignment = dict(state.get("alignment") or {})
                    alignment.update(
                        {
                            "status": "failed",
                            "progress": 0.0,
                            "message": "Noesis alignment failed; original scan outputs are preserved",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    self.update_state(scan_id, alignment=alignment)
                except HTTPException:
                    return

    def delete_scan(self, scan_id: str) -> None:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if state.get("status") in RUNNING_STATUSES:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A running scan cannot be deleted; wait for it to finish or fail",
                )
            alignment = state.get("alignment")
            if isinstance(alignment, dict) and alignment.get("status") in {
                "queued",
                "running",
            }:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A running alignment cannot be deleted; wait for it to finish or fail",
                )
            scan_dir = self.scan_dir(scan_id)
            if scan_dir.parent != self.settings.storage_root:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsafe scan path")
            shutil.rmtree(scan_dir)
        with self._locks_guard:
            self._scan_locks.pop(scan_id, None)


def _asset_url(scan_id: str, relative_path: str) -> str:
    return f"/assets/{scan_id}/{quote(relative_path, safe='/')}"


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(state)
    scan_id = str(public["id"])
    video = public.get("video")
    if isinstance(video, dict) and isinstance(video.get("path"), str):
        video["url"] = _asset_url(scan_id, video["path"])
    prepared = public.get("prepared")
    if isinstance(prepared, dict):
        for key in ("contact_sheet", "manifest"):
            if isinstance(prepared.get(key), str):
                prepared[f"{key}_url"] = _asset_url(scan_id, prepared[key])
        frames = prepared.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                for key in ("frame", "thumbnail"):
                    if isinstance(frame.get(key), str):
                        frame[f"{key}_url"] = _asset_url(scan_id, frame[key])
    outputs = public.get("outputs")
    if isinstance(outputs, dict):
        artifacts = outputs.get("artifacts")
        if isinstance(artifacts, dict):
            outputs["artifact_urls"] = {
                key: _asset_url(scan_id, value)
                for key, value in artifacts.items()
                if isinstance(value, str)
            }
        frames = outputs.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                for key in (
                    "model_rgb",
                    "depth_preview",
                    "confidence_preview",
                    "mask_preview",
                    "raw_npz",
                    "source_frame",
                ):
                    if isinstance(frame.get(key), str):
                        frame[f"{key}_url"] = _asset_url(scan_id, frame[key])
        files = outputs.get("files")
        if isinstance(files, list):
            for item in files:
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    item["url"] = _asset_url(scan_id, item["path"])
    alignment = public.get("alignment")
    if isinstance(alignment, dict):
        results = alignment.get("results")
        if isinstance(results, dict):
            artifacts = results.get("artifacts")
            if isinstance(artifacts, dict):
                results["artifact_urls"] = {
                    key: _asset_url(scan_id, value)
                    for key, value in artifacts.items()
                    if isinstance(value, str)
                }
            files = results.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        item["url"] = _asset_url(scan_id, item["path"])
    return public


def _video_suffix(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_VIDEO_SUFFIXES:
        return suffix
    mime_suffix = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-matroska": ".mkv",
        "video/3gpp": ".3gp",
    }.get(content_type.split(";", 1)[0].strip().lower())
    if mime_suffix:
        return mime_suffix
    raise HTTPException(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "Upload an MP4, MOV, M4V, WebM, MKV, or 3GP video",
    )


def create_app(
    settings: PhoneScanSettings | None = None,
    *,
    frame_processor: FrameProcessor = prepare_video_frames,
    inference_runner: InferenceRunner = run_mapanything_scan,
    da3_inference_runner: DA3InferenceRunner = run_da3_phone_scan,
    alignment_runner: AlignmentRunner = run_noesis_alignment,
) -> FastAPI:
    configured = settings or PhoneScanSettings.from_env()
    if not configured.static_root.is_dir():
        raise RuntimeError(f"phone-scan static assets are missing: {configured.static_root}")
    if not (configured.three_root / "build" / "three.module.js").is_file():
        raise RuntimeError(
            f"Three.js is required for local GLB review: {configured.three_root}"
        )
    service = PhoneScanService(
        configured,
        frame_processor=frame_processor,
        inference_runner=inference_runner,
        da3_inference_runner=da3_inference_runner,
        alignment_runner=alignment_runner,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.recover_interrupted_states()
        yield
        service.shutdown()

    app = FastAPI(
        title="Noesis Multi-View Phone Scan",
        version="1.2.0",
        lifespan=lifespan,
    )
    app.state.phone_scan_service = service

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(configured.static_root / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        usage = shutil.disk_usage(configured.storage_root)
        return {
            "status": "ok",
            "version": app.version,
            "storage_free_bytes": int(usage.free),
            "target_fps": configured.frame.target_fps,
            "max_frames": configured.frame.max_frames,
            "model_id": configured.mapanything.model_id,
            "device": configured.mapanything.device,
            "providers": {
                "mapanything": {
                    "model_id": configured.mapanything.model_id,
                    "available": True,
                },
                "da3": {
                    "model_id": configured.da3.model_id,
                    "metric_engine": str(configured.da3.metric_engine_path),
                    "available": configured.da3.metric_engine_path.is_file(),
                },
            },
            "alignment_camera_id": configured.alignment.camera_id,
            "alignment_revision_id": configured.alignment.target_revision.name,
        }

    @app.get("/api/scans")
    async def list_scans() -> list[dict[str, Any]]:
        return [_public_state(state) for state in service.list_states()]

    @app.get("/api/scans/{scan_id}")
    async def get_scan(scan_id: str) -> dict[str, Any]:
        return _public_state(service.read_state(scan_id))

    @app.post("/api/scans", status_code=status.HTTP_201_CREATED)
    async def create_scan_from_video(
        request: Request,
        name: str = Query(default="Phone room walk", max_length=80),
    ) -> JSONResponse:
        filename = unquote(request.headers.get("x-file-name") or "phone_walk.mp4")
        filename = Path(filename).name[:160] or "phone_walk.mp4"
        content_type = request.headers.get("content-type") or "application/octet-stream"
        suffix = _video_suffix(filename, content_type)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Content-Length") from exc
            if declared > configured.max_upload_bytes:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Video exceeds the upload limit")

        scan_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        scan_dir = service.scan_dir(scan_id)
        scan_dir.mkdir(parents=True, exist_ok=False)
        video_name = f"phone_walk{suffix}"
        video_path = scan_dir / video_name
        temporary = scan_dir / f".{video_name}.uploading"
        state_payload = {
            "schema": "noesis.phone_scan.state.v2",
            "id": scan_id,
            "name": name.strip() or "Phone room walk",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "uploading",
            "progress": 0.0,
            "message": "Receiving phone video",
            "error": None,
            "video": {
                "path": video_name,
                "original_name": filename,
                "content_type": content_type,
            },
        }
        service._write_state_unlocked(scan_id, state_payload)
        received = 0
        try:
            with temporary.open("wb") as handle:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > configured.max_upload_bytes:
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            "Video exceeds the upload limit",
                        )
                    handle.write(chunk)
            if received <= 0:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded video is empty")
            os.replace(temporary, video_path)
            state_payload["video"]["size_bytes"] = received
            state_payload.update(
                {
                    "status": "processing_frames",
                    "progress": 0.0,
                    "message": "Upload saved; preparing multi-view inference frames",
                }
            )
            service._write_state_unlocked(scan_id, state_payload)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if scan_dir.exists():
                shutil.rmtree(scan_dir)
            raise
        service.submit_frame_preparation(scan_id)
        return JSONResponse(_public_state(state_payload), status_code=status.HTTP_201_CREATED)

    @app.post("/api/scans/{scan_id}/initiate-ma", status_code=status.HTTP_202_ACCEPTED)
    async def initiate_mapanything(scan_id: str) -> JSONResponse:
        return JSONResponse(
            _public_state(service.initiate_mapanything(scan_id)),
            status_code=status.HTTP_202_ACCEPTED,
        )

    @app.post("/api/scans/{scan_id}/initiate-inference", status_code=status.HTTP_202_ACCEPTED)
    async def initiate_inference(
        scan_id: str,
        provider: str = Query(default="mapanything"),
    ) -> JSONResponse:
        return JSONResponse(
            _public_state(service.initiate_inference(scan_id, provider)),
            status_code=status.HTTP_202_ACCEPTED,
        )

    @app.post("/api/scans/{scan_id}/align-noesis", status_code=status.HTTP_202_ACCEPTED)
    async def initiate_alignment(scan_id: str) -> JSONResponse:
        return JSONResponse(
            _public_state(service.initiate_alignment(scan_id)),
            status_code=status.HTTP_202_ACCEPTED,
        )

    @app.delete("/api/scans/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_scan(scan_id: str) -> Response:
        service.delete_scan(scan_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    app.mount("/static", StaticFiles(directory=configured.static_root), name="phone-scan-static")
    app.mount(
        "/three/build",
        StaticFiles(directory=configured.three_root / "build"),
        name="phone-scan-three-build",
    )
    app.mount(
        "/three/examples",
        StaticFiles(directory=configured.three_root / "examples" / "jsm"),
        name="phone-scan-three-examples",
    )
    app.mount(
        "/assets",
        StaticFiles(directory=configured.storage_root),
        name="phone-scan-assets",
    )
    return app


app = create_app()


__all__ = ["PhoneScanService", "PhoneScanSettings", "app", "create_app"]
