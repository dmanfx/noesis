from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .alignment import NoesisAlignmentSettings, run_noesis_alignment
from .da3_inference import DA3PhoneScanSettings
from .inference import MapAnythingScanSettings
from .windowed_inference import run_adaptive_mapanything_scan
from .windowed_da3_inference import run_adaptive_da3_phone_scan
from .pcf import run_pcf_review_candidate
from .processing import FramePreparationSettings, prepare_video_frames
from .supplement import (
    SupplementIntegrationSettings,
    materialize_noesis_revision,
    public_active_revision,
    run_supplement_integration,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
SCAN_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
SUPPLEMENT_ID_PATTERN = re.compile(r"^add-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp"}
INFERENCE_PROVIDERS = {"mapanything", "da3"}
MAX_SCAN_NAME_LENGTH = 80
RUNNING_STATUSES = {
    "processing_frames",
    "ma_queued",
    "ma_running",
    "da3_queued",
    "da3_running",
}
SUPPLEMENT_RUNNING_STATUSES = {"uploading", "processing_frames", "queued", "running"}
PCF_RUNNING_STATUSES = {"queued", "running"}
PCF_APPLIANCE_TARGET = "menon-appliance.target"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_display_name(camera_id: str) -> str:
    return " ".join(part.capitalize() for part in camera_id.replace("_", "-").split("-"))


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_root(raw: str | None, default: Path) -> Path:
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _alignment_settings_for_revision(
    *,
    camera_id: str,
    target_revision: Path,
    calibration_path: Path,
    calibration_rows: dict[str, Any],
    review_point_budget: int,
) -> NoesisAlignmentSettings:
    if CAMERA_ID_PATTERN.fullmatch(camera_id) is None:
        raise ValueError(f"alignment release contains invalid camera ID {camera_id!r}")
    calibration = calibration_rows.get(camera_id)
    if not isinstance(calibration, dict) or not isinstance(calibration.get("E"), list):
        raise ValueError(f"alignment camera {camera_id} has no calibrated E matrix")
    metadata = _read_json_object(
        target_revision / "room_points_meta.json",
        label=f"alignment revision metadata for {camera_id}",
    )
    if metadata.get("camera") != camera_id:
        raise ValueError(
            f"alignment revision {target_revision.name} belongs to "
            f"{metadata.get('camera')}, not {camera_id}"
        )
    if metadata.get("coordinate_frame") != "backend_world_m_stream_points":
        raise ValueError(
            f"alignment revision {target_revision.name} is not in backend world coordinates"
        )
    if not (target_revision / "room_points.npz").is_file():
        raise ValueError(f"alignment revision {target_revision.name} has no room_points.npz")
    keyframes = metadata.get("rgb_keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        raise ValueError(f"alignment revision {target_revision.name} has no RGB keyframe")
    for relative in keyframes.values():
        keyframe = (target_revision / str(relative)).resolve()
        try:
            keyframe.relative_to(target_revision)
        except ValueError as exc:
            raise ValueError(
                f"alignment revision {target_revision.name} has an unsafe RGB keyframe path"
            ) from exc
        if not keyframe.is_file():
            raise ValueError(
                f"alignment revision {target_revision.name} is missing {relative}"
            )
    return NoesisAlignmentSettings(
        camera_id=camera_id,
        target_revision=target_revision,
        calibration_path=calibration_path,
        review_point_budget=review_point_budget,
    )


def _alignment_targets_from_release(
    release_path: Path,
    calibration_path: Path,
    *,
    review_point_budget: int,
) -> tuple[tuple[NoesisAlignmentSettings, ...], str]:
    release = _read_json_object(release_path, label="phone-scan alignment release")
    if release.get("contract") != "noesis.scene.release":
        raise ValueError(f"alignment release has an unsupported contract: {release_path}")
    release_id = str(release.get("release_id") or "").strip()
    if not release_id:
        raise ValueError(f"alignment release has no release_id: {release_path}")
    validation_path = release_path.with_name(f"{release_path.stem}.validation.json")
    validation = _read_json_object(
        validation_path,
        label="phone-scan alignment release validation",
    )
    checks = validation.get("checks")
    if (
        validation.get("schema") != "noesis.scene.validation.v1"
        or validation.get("release_id") != release_id
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError(f"alignment release did not pass validation: {release_path}")

    calibration_root = _read_json_object(
        calibration_path,
        label="phone-scan camera calibration",
    )
    raw_calibration_rows = calibration_root.get("cameras", calibration_root)
    if not isinstance(raw_calibration_rows, dict):
        raise ValueError("phone-scan camera calibration has no camera rows")
    calibration_rows = dict(raw_calibration_rows)
    revisions_root = (REPO_ROOT / "data" / "virtual_twin" / "revisions").resolve()
    camera_rows = release.get("cameras")
    if not isinstance(camera_rows, list) or not camera_rows:
        raise ValueError(f"alignment release has no cameras: {release_path}")

    targets: list[NoesisAlignmentSettings] = []
    seen_camera_ids: set[str] = set()
    for row in camera_rows:
        if not isinstance(row, dict):
            raise ValueError(f"alignment release has an invalid camera row: {release_path}")
        camera_id = str(row.get("camera_id") or "").strip()
        revision_id = str(row.get("revision_id") or "").strip()
        artifact_path = str(row.get("artifact_path") or revision_id).strip()
        if camera_id in seen_camera_ids:
            raise ValueError(f"alignment release repeats camera {camera_id}")
        if not revision_id or artifact_path != Path(artifact_path).name:
            raise ValueError(f"alignment release has an unsafe revision for {camera_id}")
        target_revision = (revisions_root / artifact_path).resolve()
        try:
            target_revision.relative_to(revisions_root)
        except ValueError as exc:
            raise ValueError(f"alignment release has an unsafe revision for {camera_id}") from exc
        if target_revision.name != revision_id:
            raise ValueError(
                f"alignment release revision identity mismatch for {camera_id}: "
                f"{revision_id} != {target_revision.name}"
            )
        targets.append(
            _alignment_settings_for_revision(
                camera_id=camera_id,
                target_revision=target_revision,
                calibration_path=calibration_path,
                calibration_rows=calibration_rows,
                review_point_budget=review_point_budget,
            )
        )
        seen_camera_ids.add(camera_id)
    return tuple(targets), release_id


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
    alignment_targets: tuple[NoesisAlignmentSettings, ...] = ()
    alignment_release_id: str | None = None
    pcf_storage_root: Path | None = None
    pcf_pause_appliance: bool = False

    @classmethod
    def from_env(cls) -> "PhoneScanSettings":
        storage_root = _resolve_root(
            os.environ.get("NOESIS_PHONE_SCAN_STORAGE_ROOT"),
            REPO_ROOT / "data" / "mapanything_phone_scans",
        )
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
            / "da3metric_large_294x518_b3_fp16_trt10.16.engine"
            if artifact_root_raw
            else REPO_ROOT
            / "data"
            / "ds9_artifacts"
            / "models"
            / "engines"
            / "da3metric_large_294x518_b3_fp16_trt10.16.engine"
        )
        calibration_path = _resolve_root(
            os.environ.get("NOESIS_PHONE_SCAN_ALIGNMENT_CALIBRATION"),
            REPO_ROOT / "config" / "camera_calibration.json",
        )
        if str(os.environ.get("NOESIS_PHONE_SCAN_ALIGNMENT_REVISION") or "").strip():
            raise ValueError(
                "NOESIS_PHONE_SCAN_ALIGNMENT_REVISION is no longer a safe "
                "multi-camera configuration; set NOESIS_PHONE_SCAN_ALIGNMENT_RELEASE"
            )
        alignment_release_path = _resolve_root(
            os.environ.get("NOESIS_PHONE_SCAN_ALIGNMENT_RELEASE"),
            REPO_ROOT
            / "data"
            / "virtual_twin"
            / "releases"
            / "home_rgbmesh_20260623T2158_v1.json",
        )
        alignment_targets, alignment_release_id = _alignment_targets_from_release(
            alignment_release_path,
            calibration_path,
            review_point_budget=point_budget,
        )
        requested_alignment_camera_id = str(
            os.environ.get("NOESIS_PHONE_SCAN_ALIGNMENT_CAMERA_ID") or ""
        ).strip()
        alignment_by_camera = {
            target.camera_id: target for target in alignment_targets
        }
        if requested_alignment_camera_id:
            alignment = alignment_by_camera.get(requested_alignment_camera_id)
            if alignment is None:
                raise ValueError(
                    f"alignment release has no camera {requested_alignment_camera_id!r}"
                )
        else:
            alignment = alignment_targets[0]
        anchor_enabled = os.environ.get(
            "NOESIS_PHONE_SCAN_STATIC_ANCHOR", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        anchor_image_raw = str(
            os.environ.get("NOESIS_PHONE_SCAN_STATIC_ANCHOR_IMAGE") or ""
        ).strip()
        if anchor_enabled and not anchor_image_raw:
            raise ValueError(
                "NOESIS_PHONE_SCAN_STATIC_ANCHOR_IMAGE is required when the "
                "multi-camera phone-scan static anchor is enabled"
            )
        anchor_image = (
            _resolve_root(anchor_image_raw, REPO_ROOT) if anchor_enabled else None
        )
        return cls(
            storage_root=storage_root,
            static_root=APP_ROOT / "static",
            three_root=three_root,
            max_upload_bytes=_env_int(
                "NOESIS_PHONE_SCAN_MAX_UPLOAD_BYTES",
                8 * 1024 * 1024 * 1024,
                minimum=1024 * 1024,
            ),
            frame=FramePreparationSettings(
                candidate_fps=_env_float(
                    "NOESIS_PHONE_SCAN_CANDIDATE_FPS", 4.0, minimum=0.5
                ),
                max_candidate_frames=_env_int(
                    "NOESIS_PHONE_SCAN_MAX_CANDIDATE_FRAMES", 1200, minimum=8
                ),
                max_selected_frames=_env_int(
                    "NOESIS_PHONE_SCAN_MAX_SELECTED_FRAMES", 256, minimum=8
                ),
                candidate_edge_px=_env_int(
                    "NOESIS_PHONE_SCAN_CANDIDATE_EDGE_PX", 1280, minimum=518
                ),
                feature_edge_px=_env_int(
                    "NOESIS_PHONE_SCAN_FEATURE_EDGE_PX", 640, minimum=320
                ),
                min_keyframe_interval_s=_env_float(
                    "NOESIS_PHONE_SCAN_MIN_KEYFRAME_INTERVAL_S", 0.40, minimum=0.10
                ),
                max_keyframe_interval_s=_env_float(
                    "NOESIS_PHONE_SCAN_MAX_KEYFRAME_INTERVAL_S", 1.25, minimum=0.30
                ),
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
                max_joint_views=_env_int(
                    "NOESIS_PHONE_SCAN_MA_MAX_JOINT_VIEWS", 80, minimum=16
                ),
                window_overlap_views=_env_int(
                    "NOESIS_PHONE_SCAN_MA_WINDOW_OVERLAP_VIEWS", 24, minimum=4
                ),
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
                max_joint_views=_env_int(
                    "NOESIS_PHONE_SCAN_DA3_MAX_JOINT_VIEWS", 48, minimum=16
                ),
                window_overlap_views=_env_int(
                    "NOESIS_PHONE_SCAN_DA3_WINDOW_OVERLAP_VIEWS", 16, minimum=4
                ),
                anchor_image=anchor_image,
            ),
            alignment=alignment,
            alignment_targets=alignment_targets,
            alignment_release_id=alignment_release_id,
            pcf_storage_root=_resolve_root(
                os.environ.get("NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT"),
                storage_root / "pcf",
            ),
            pcf_pause_appliance=_env_bool(
                "NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE", False
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
SupplementRunner = Callable[
    [
        Path,
        Path,
        Path,
        dict[str, Any],
        dict[str, Any],
        str,
        Callable[..., dict[str, Any]],
        Any,
        SupplementIntegrationSettings,
        Callable[[float, str], None],
    ],
    dict[str, Any],
]
PCFRunner = Callable[
    [
        Path,
        Path,
        dict[str, Any],
        NoesisAlignmentSettings,
        MapAnythingScanSettings,
        Callable[[float, str], None],
    ],
    dict[str, Any],
]


def _prepared_paths_in_scan(
    scan_dir: Path,
    prepared_root: Path,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(prepared)

    def rewrite(raw: str) -> str:
        candidate = (prepared_root / raw).resolve()
        try:
            candidate.relative_to(prepared_root.resolve())
            return candidate.relative_to(scan_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"prepared artifact escaped its added-video directory: {raw}") from exc

    for key in ("contact_sheet", "manifest"):
        if isinstance(result.get(key), str):
            result[key] = rewrite(result[key])
    frames = result.get("frames")
    if isinstance(frames, list):
        for row in frames:
            if not isinstance(row, dict):
                continue
            for key in ("frame", "thumbnail"):
                if isinstance(row.get(key), str):
                    row[key] = rewrite(row[key])
    return result


def _pcf_paths_for_state(run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(result)
    prefix = Path("runs") / run_id

    def rewrite(raw: str) -> str:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"PCF artifact path is unsafe: {raw}")
        return (prefix / relative).as_posix()

    artifacts = public.get("artifacts")
    if isinstance(artifacts, dict):
        public["artifacts"] = {
            key: rewrite(value)
            for key, value in artifacts.items()
            if isinstance(value, str)
        }
    files = public.get("files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                item["path"] = rewrite(item["path"])
    public["run_root"] = prefix.as_posix()
    return public


class PhoneScanService:
    def __init__(
        self,
        settings: PhoneScanSettings,
        *,
        frame_processor: FrameProcessor = prepare_video_frames,
        inference_runner: InferenceRunner = run_adaptive_mapanything_scan,
        da3_inference_runner: DA3InferenceRunner = run_adaptive_da3_phone_scan,
        alignment_runner: AlignmentRunner = run_noesis_alignment,
        supplement_runner: SupplementRunner = run_supplement_integration,
        pcf_runner: PCFRunner = run_pcf_review_candidate,
    ) -> None:
        self.settings = settings
        self.frame_processor = frame_processor
        self.inference_runner = inference_runner
        self.da3_inference_runner = da3_inference_runner
        self.alignment_runner = alignment_runner
        self.supplement_runner = supplement_runner
        self.pcf_runner = pcf_runner
        self.supplement_settings = SupplementIntegrationSettings(
            max_total_views=self.settings.mapanything.max_joint_views,
            point_budget=max(
                self.settings.mapanything.point_budget,
                self.settings.da3.point_budget,
            ),
        )
        self.settings.storage_root.mkdir(parents=True, exist_ok=True)
        self.pcf_storage_root = (
            self.settings.pcf_storage_root
            or self.settings.storage_root / "pcf"
        ).resolve()
        if self.pcf_storage_root == self.settings.storage_root:
            raise ValueError("PCF storage root must not equal the phone-scan storage root")
        self.pcf_storage_root.mkdir(parents=True, exist_ok=True)
        configured_alignment_targets = self.settings.alignment_targets or (
            self.settings.alignment,
        )
        self._alignment_targets: dict[str, NoesisAlignmentSettings] = {}
        for target in configured_alignment_targets:
            if target.camera_id in self._alignment_targets:
                raise ValueError(f"duplicate phone-scan alignment camera {target.camera_id}")
            self._alignment_targets[target.camera_id] = target
        if not self._alignment_targets:
            raise ValueError("phone-scan alignment requires at least one camera target")
        self._locks_guard = threading.Lock()
        self._scan_locks: dict[str, threading.RLock] = {}
        self._inference_lock = threading.Lock()
        self._alignment_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PhoneScan")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def public_alignment_targets(self) -> list[dict[str, str]]:
        return [
            {
                "camera_id": target.camera_id,
                "label": _camera_display_name(target.camera_id),
                "revision_id": target.target_revision.name,
            }
            for target in self._alignment_targets.values()
        ]

    @staticmethod
    def _user_unit_active(unit: str) -> bool:
        completed = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return completed.returncode == 0

    @staticmethod
    def _systemctl_user(action: str, unit: str, *, timeout_s: int) -> None:
        completed = subprocess.run(
            ["systemctl", "--user", action, unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown systemd error").strip()
            raise RuntimeError(f"systemctl --user {action} {unit} failed: {detail}")

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

    def supplement_dir(self, scan_id: str, supplement_id: str) -> Path:
        if not SUPPLEMENT_ID_PATTERN.fullmatch(supplement_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Added video not found")
        root = (self.scan_dir(scan_id) / "supplements").resolve()
        candidate = (root / supplement_id).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Added video not found") from exc
        return candidate

    def pcf_scan_dir(self, scan_id: str) -> Path:
        if not SCAN_ID_PATTERN.fullmatch(scan_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found")
        candidate = (self.pcf_storage_root / scan_id).resolve()
        try:
            candidate.relative_to(self.pcf_storage_root)
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

    @staticmethod
    def _supplement_index(state: dict[str, Any], supplement_id: str) -> int:
        supplements = state.get("supplements")
        if not isinstance(supplements, list):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Added video not found")
        for index, supplement in enumerate(supplements):
            if isinstance(supplement, dict) and supplement.get("id") == supplement_id:
                return index
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Added video not found")

    def add_supplement(self, scan_id: str, supplement: dict[str, Any]) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if state.get("status") != "complete" or not isinstance(state.get("outputs"), dict):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Additional video requires a completed reconstruction",
                )
            pcf = state.get("pcf")
            if isinstance(pcf, dict) and pcf.get("status") in PCF_RUNNING_STATUSES:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Wait for the PCF review run to finish before adding another video",
                )
            supplements = list(state.get("supplements") or [])
            if any(
                isinstance(row, dict)
                and row.get("status") in SUPPLEMENT_RUNNING_STATUSES
                for row in supplements
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Another added video is still being prepared or integrated",
                )
            supplements.append(supplement)
            state["supplements"] = supplements
            self._write_state_unlocked(scan_id, state)
            return state

    def update_supplement(
        self,
        scan_id: str,
        supplement_id: str,
        **changes: Any,
    ) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            index = self._supplement_index(state, supplement_id)
            supplement = dict(state["supplements"][index])
            supplement.update(changes)
            supplement["updated_at"] = _utc_now()
            state["supplements"][index] = supplement
            self._write_state_unlocked(scan_id, state)
            return state

    def discard_supplement_state(self, scan_id: str, supplement_id: str) -> None:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            index = self._supplement_index(state, supplement_id)
            del state["supplements"][index]
            self._write_state_unlocked(scan_id, state)

    def rename_scan(self, scan_id: str, name: str) -> dict[str, Any]:
        normalized_name = " ".join(str(name).split())
        if not normalized_name:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Walk name cannot be empty",
            )
        if len(normalized_name) > MAX_SCAN_NAME_LENGTH:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Walk name cannot exceed {MAX_SCAN_NAME_LENGTH} characters",
            )
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            state["name"] = normalized_name
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
            restore_interrupted_appliance = False
            with self._lock(scan_id):
                latest = self._read_state_unlocked(scan_id)
                supplements = latest.get("supplements")
                changed = False
                pcf = latest.get("pcf")
                if isinstance(pcf, dict) and pcf.get("status") in PCF_RUNNING_STATUSES:
                    interrupted_pcf = dict(pcf)
                    runtime_lease = dict(interrupted_pcf.get("runtime_lease") or {})
                    restore_interrupted_appliance = bool(
                        runtime_lease.get("appliance_target") == PCF_APPLIANCE_TARGET
                        and runtime_lease.get("appliance_was_active") is True
                        and runtime_lease.get("pause_requested") is True
                        and runtime_lease.get("restore_passed") is not True
                    )
                    interrupted_pcf.update(
                        {
                            "status": "failed",
                            "progress": 0.0,
                            "message": "PCF was interrupted by a tool restart",
                            "error": (
                                "The DA3 walk, alignment, and any completed PCF stage "
                                "outputs are preserved; retry PCF to create a new run."
                            ),
                            "updated_at": _utc_now(),
                        }
                    )
                    latest["pcf"] = interrupted_pcf
                    changed = True
                if isinstance(supplements, list):
                    for index, supplement in enumerate(supplements):
                        if not isinstance(supplement, dict):
                            continue
                        previous_status = supplement.get("status")
                        updated = dict(supplement)
                        if previous_status in {"uploading", "processing_frames"}:
                            updated.update(
                                {
                                    "status": "frame_failed",
                                    "progress": 0.0,
                                    "message": "Additional-video preparation was interrupted by a tool restart",
                                    "error": "Delete this incomplete addition and upload the video again.",
                                }
                            )
                        elif previous_status in {"queued", "running"}:
                            updated.update(
                                {
                                    "status": "integration_failed",
                                    "progress": 0.0,
                                    "message": "Additional-video integration was interrupted by a tool restart",
                                    "error": "The video and prepared frames are preserved; retry integration.",
                                }
                            )
                        else:
                            continue
                        updated["updated_at"] = _utc_now()
                        supplements[index] = updated
                        changed = True
                if changed:
                    latest["supplements"] = supplements
                    self._write_state_unlocked(scan_id, latest)
            if restore_interrupted_appliance:
                restore_error: Exception | None = None
                try:
                    self._systemctl_user(
                        "start", PCF_APPLIANCE_TARGET, timeout_s=300
                    )
                    if not self._user_unit_active("noesis-appliance.service"):
                        raise RuntimeError(
                            "native Noesis was not active after interrupted PCF recovery"
                        )
                except Exception as exc:
                    restore_error = exc
                try:
                    with self._lock(scan_id):
                        latest = self._read_state_unlocked(scan_id)
                        interrupted_pcf = dict(latest.get("pcf") or {})
                        runtime_lease = dict(
                            interrupted_pcf.get("runtime_lease") or {}
                        )
                        runtime_lease.update(
                            {
                                "restore_passed": restore_error is None,
                                "restore_error": (
                                    None
                                    if restore_error is None
                                    else f"{type(restore_error).__name__}: "
                                    f"{restore_error}"
                                ),
                                "recovered_after_tool_restart": True,
                            }
                        )
                        interrupted_pcf["runtime_lease"] = runtime_lease
                        if restore_error is not None:
                            interrupted_pcf["error"] = (
                                f"{interrupted_pcf.get('error')} Automatic appliance "
                                f"restore also failed: {runtime_lease['restore_error']}"
                            )
                        interrupted_pcf["updated_at"] = _utc_now()
                        latest["pcf"] = interrupted_pcf
                        self._write_state_unlocked(scan_id, latest)
                except HTTPException:
                    continue

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

    def submit_supplement_preparation(self, scan_id: str, supplement_id: str) -> None:
        self._executor.submit(self._prepare_supplement_worker, scan_id, supplement_id)

    def _supplement_progress(
        self,
        scan_id: str,
        supplement_id: str,
        fraction: float,
        message: str,
    ) -> None:
        try:
            self.update_supplement(
                scan_id,
                supplement_id,
                progress=float(min(1.0, max(0.0, fraction))),
                message=str(message),
            )
        except HTTPException:
            return

    def _prepare_supplement_worker(self, scan_id: str, supplement_id: str) -> None:
        try:
            state = self.read_state(scan_id)
            index = self._supplement_index(state, supplement_id)
            supplement = state["supplements"][index]
            scan_dir = self.scan_dir(scan_id)
            supplement_dir = self.supplement_dir(scan_id, supplement_id)
            video_path = scan_dir / str(supplement["video"]["path"])
            frame_settings = replace(
                self.settings.frame,
                max_selected_frames=self.supplement_settings.new_view_limit,
            )
            prepared_local = self.frame_processor(
                video_path,
                supplement_dir,
                frame_settings,
                lambda fraction, message: self._supplement_progress(
                    scan_id, supplement_id, fraction, message
                ),
            )
            prepared = _prepared_paths_in_scan(
                scan_dir, supplement_dir, prepared_local
            )
            provider = str(state.get("provider") or state["outputs"].get("provider") or "mapanything")
            label = "DA3" if provider == "da3" else "MapAnything"
            self.update_supplement(
                scan_id,
                supplement_id,
                status="ready",
                progress=1.0,
                message=(
                    f"{prepared['frame_count']} new views are ready to bridge into the "
                    f"current {label} reconstruction"
                ),
                error=None,
                prepared=prepared,
                provider=provider,
            )
        except Exception as exc:
            try:
                self.update_supplement(
                    scan_id,
                    supplement_id,
                    status="frame_failed",
                    progress=0.0,
                    message="Additional-video frame preparation failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except HTTPException:
                return

    def initiate_supplement_integration(
        self, scan_id: str, supplement_id: str
    ) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            pcf = state.get("pcf")
            if isinstance(pcf, dict) and pcf.get("status") in PCF_RUNNING_STATUSES:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Wait for the PCF review run to finish before integrating another video",
                )
            index = self._supplement_index(state, supplement_id)
            supplement = dict(state["supplements"][index])
            if supplement.get("status") not in {"ready", "integration_failed"}:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "This added video is not ready to integrate",
                )
            if any(
                row_index != index
                and isinstance(row, dict)
                and row.get("status") in SUPPLEMENT_RUNNING_STATUSES
                for row_index, row in enumerate(state["supplements"])
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Another added video is still being prepared or integrated",
                )
            provider = str(
                supplement.get("provider")
                or state.get("provider")
                or state.get("outputs", {}).get("provider")
                or "mapanything"
            ).lower()
            if provider not in INFERENCE_PROVIDERS:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "The base reconstruction provider is unavailable",
                )
            supplement.update(
                {
                    "status": "queued",
                    "progress": 0.0,
                    "message": "Waiting for the reconstruction GPU lane",
                    "error": None,
                    "provider": provider,
                }
            )
            state["supplements"][index] = supplement
            self._write_state_unlocked(scan_id, state)
        self._executor.submit(
            self._supplement_integration_worker, scan_id, supplement_id, provider
        )
        return state

    def _supplement_integration_worker(
        self, scan_id: str, supplement_id: str, provider: str
    ) -> None:
        runner = (
            self.inference_runner if provider == "mapanything" else self.da3_inference_runner
        )
        provider_settings = (
            self.settings.mapanything if provider == "mapanything" else self.settings.da3
        )
        label = "MapAnything" if provider == "mapanything" else "DA3"
        with self._inference_lock:
            scan_dir = self.scan_dir(scan_id)
            supplement_dir = self.supplement_dir(scan_id, supplement_id)
            build_dir = supplement_dir / ".revision-building"
            final_dir = supplement_dir / "revision"
            try:
                state = self.read_state(scan_id)
                supplement_index = self._supplement_index(state, supplement_id)
                supplement = dict(state["supplements"][supplement_index])
                self.update_supplement(
                    scan_id,
                    supplement_id,
                    status="running",
                    progress=0.01,
                    message=f"Starting joint {label} bridge reconstruction",
                    error=None,
                )
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                if final_dir.exists():
                    raise RuntimeError("this added video already has a completed revision")
                result = self.supplement_runner(
                    scan_dir,
                    supplement_dir,
                    build_dir,
                    state,
                    supplement,
                    provider,
                    runner,
                    provider_settings,
                    self.supplement_settings,
                    lambda fraction, message: self._supplement_progress(
                        scan_id, supplement_id, fraction, message
                    ),
                )
                os.replace(build_dir, final_dir)
                with self._lock(scan_id):
                    latest = self._read_state_unlocked(scan_id)
                    index = self._supplement_index(latest, supplement_id)
                    completed = dict(latest["supplements"][index])
                    completed.update(
                        {
                            "status": "complete",
                            "progress": 1.0,
                            "message": "Added video is registered and saved as the current revision",
                            "error": None,
                            "results": result,
                            "updated_at": _utc_now(),
                        }
                    )
                    latest["supplements"][index] = completed
                    latest["active_revision"] = public_active_revision(result)
                    self._write_state_unlocked(scan_id, latest)
            except Exception as exc:
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                try:
                    self.update_supplement(
                        scan_id,
                        supplement_id,
                        status="integration_failed",
                        progress=0.0,
                        message=(
                            "Added video was not merged; the base reconstruction and "
                            "prepared frames are preserved"
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except HTTPException:
                    return

    def initiate_mapanything(self, scan_id: str) -> dict[str, Any]:
        return self.initiate_inference(scan_id, "mapanything")

    def initiate_inference(self, scan_id: str, provider: str) -> dict[str, Any]:
        provider = str(provider).strip().lower()
        if provider not in INFERENCE_PROVIDERS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
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

    def initiate_alignment(self, scan_id: str, camera_id: str) -> dict[str, Any]:
        target = self._alignment_targets.get(str(camera_id).strip())
        if target is None:
            available = ", ".join(self._alignment_targets)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Unknown alignment camera {camera_id!r}; choose one of: {available}",
            )
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
                "message": (
                    f"Waiting to align with the {_camera_display_name(target.camera_id)} "
                    "static camera"
                ),
                "error": None,
                "target_camera_id": target.camera_id,
                "target_revision_id": target.target_revision.name,
                "target_release_id": self.settings.alignment_release_id,
            }
            self._write_state_unlocked(scan_id, state)
        self._executor.submit(self._alignment_worker, scan_id, target.camera_id)
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

    def _alignment_worker(self, scan_id: str, camera_id: str) -> None:
        target = self._alignment_targets[camera_id]
        camera_label = _camera_display_name(camera_id)
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
                        "message": (
                            f"Starting gravity-preserving alignment to the {camera_label} "
                            "static camera"
                        ),
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
                    target,
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
                        "message": (
                            f"Phone reconstruction is aligned to the {camera_label} Noesis world"
                        ),
                        "error": None,
                        "results": result,
                    }
                )
                with self._lock(scan_id):
                    latest = self._read_state_unlocked(scan_id)
                    active = latest.get("active_revision")
                    if isinstance(active, dict) and active.get("supplement_id"):
                        supplement_id = str(active["supplement_id"])
                        supplement_index = self._supplement_index(latest, supplement_id)
                        supplement = dict(latest["supplements"][supplement_index])
                        supplement_result = supplement.get("results")
                        if not isinstance(supplement_result, dict):
                            raise RuntimeError("the active added-video revision is missing")
                        try:
                            supplement_result = materialize_noesis_revision(
                                scan_dir, supplement_result, result
                            )
                            supplement["results"] = supplement_result
                            supplement["updated_at"] = _utc_now()
                            latest["supplements"][supplement_index] = supplement
                            latest["active_revision"] = public_active_revision(
                                supplement_result
                            )
                        except Exception as derivative_exc:
                            alignment["active_revision_derivative_error"] = (
                                f"{type(derivative_exc).__name__}: {derivative_exc}"
                            )
                            active["noesis_derivative_error"] = alignment[
                                "active_revision_derivative_error"
                            ]
                            latest["active_revision"] = active
                    latest["alignment"] = alignment
                    self._write_state_unlocked(scan_id, latest)
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

    def initiate_pcf(self, scan_id: str) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if state.get("status") != "complete" or not isinstance(
                state.get("outputs"), dict
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "PCF requires a completed DA3 reconstruction",
                )
            provider = str(
                state.get("provider") or state.get("outputs", {}).get("provider") or ""
            ).lower()
            if provider != "da3":
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "PCF requires DA3 as the base reconstruction provider",
                )
            alignment = state.get("alignment")
            alignment_results = (
                alignment.get("results") if isinstance(alignment, dict) else None
            )
            if (
                not isinstance(alignment, dict)
                or alignment.get("status") != "complete"
                or not isinstance(alignment_results, dict)
                or not isinstance(alignment_results.get("quality_gate"), dict)
                or alignment_results["quality_gate"].get("passed") is not True
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "PCF requires a completed alignment that passed its quality gate",
                )
            camera_id = str(
                alignment_results.get("target_camera_id")
                or alignment.get("target_camera_id")
                or ""
            )
            target = self._alignment_targets.get(camera_id)
            if target is None:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "The aligned camera is not available in the active scene release",
                )
            if alignment_results.get("target_revision_id") != target.target_revision.name:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "The aligned static revision is not the active camera revision",
                )
            if state.get("active_revision"):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "PCF currently requires the original DA3 walk; an active added-video revision is not silently included",
                )
            if any(
                isinstance(row, dict)
                and row.get("status") in SUPPLEMENT_RUNNING_STATUSES
                for row in state.get("supplements") or []
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Wait for the added-video operation to finish before starting PCF",
                )
            previous = state.get("pcf")
            previous_status = previous.get("status") if isinstance(previous, dict) else None
            if previous_status not in {None, "failed"}:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"PCF cannot start while status is {previous_status}",
                )
            run_id = f"pcf-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            now = _utc_now()
            state["pcf"] = {
                "schema": "noesis.phone_scan.pcf_job.v1",
                "run_id": run_id,
                "created_at": now,
                "updated_at": now,
                "status": "queued",
                "progress": 0.0,
                "message": "Waiting for the PCF reconstruction lane",
                "error": None,
                "target_camera_id": target.camera_id,
                "target_revision_id": target.target_revision.name,
                "source_provider": "da3",
                "source_scope": "original_prepared_walk",
                "review_only": True,
            }
            self._write_state_unlocked(scan_id, state)
        self._executor.submit(self._pcf_worker, scan_id, run_id, target.camera_id)
        return state

    def _pcf_progress(self, scan_id: str, fraction: float, message: str) -> None:
        try:
            with self._lock(scan_id):
                state = self._read_state_unlocked(scan_id)
                pcf = state.get("pcf")
                if not isinstance(pcf, dict) or pcf.get("status") not in PCF_RUNNING_STATUSES:
                    return
                pcf.update(
                    {
                        "progress": float(min(1.0, max(0.0, fraction))),
                        "message": str(message),
                        "updated_at": _utc_now(),
                    }
                )
                state["pcf"] = pcf
                self._write_state_unlocked(scan_id, state)
        except HTTPException:
            return

    def _update_pcf_runtime_lease(
        self, scan_id: str, run_id: str, **changes: Any
    ) -> None:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            pcf = dict(state.get("pcf") or {})
            if pcf.get("run_id") != run_id:
                raise RuntimeError("PCF job identity changed during the runtime lease")
            runtime_lease = dict(pcf.get("runtime_lease") or {})
            runtime_lease.update(changes)
            pcf["runtime_lease"] = runtime_lease
            pcf["updated_at"] = _utc_now()
            state["pcf"] = pcf
            self._write_state_unlocked(scan_id, state)

    def _pcf_worker(self, scan_id: str, run_id: str, camera_id: str) -> None:
        target = self._alignment_targets[camera_id]
        with self._inference_lock:
            pcf_scan_root = self.pcf_scan_dir(scan_id)
            run_root = pcf_scan_root / "runs" / run_id
            appliance_was_active = False
            appliance_paused = False
            runtime_restore_error: Exception | None = None
            runner_error: Exception | None = None
            result: dict[str, Any] | None = None
            try:
                state = self.read_state(scan_id)
                pcf = dict(state.get("pcf") or {})
                if pcf.get("run_id") != run_id:
                    raise RuntimeError("PCF job identity changed before execution")
                if state.get("active_revision"):
                    raise RuntimeError(
                        "an added-video revision became active before PCF started"
                    )
                pcf.update(
                    {
                        "status": "running",
                        "progress": 0.01,
                        "message": "Starting Prior-Conditioned Fusion",
                        "error": None,
                        "runtime_lease": {
                            "configured": bool(self.settings.pcf_pause_appliance),
                            "appliance_target": PCF_APPLIANCE_TARGET,
                            "appliance_was_active": None,
                            "pause_requested": False,
                            "appliance_paused": False,
                            "restore_passed": None,
                        },
                        "updated_at": _utc_now(),
                    }
                )
                self.update_state(scan_id, pcf=pcf)
                run_root.parent.mkdir(parents=True, exist_ok=True)
                run_root.mkdir(exist_ok=False)
                try:
                    if self.settings.pcf_pause_appliance:
                        appliance_was_active = self._user_unit_active(
                            PCF_APPLIANCE_TARGET
                        )
                        self._update_pcf_runtime_lease(
                            scan_id,
                            run_id,
                            appliance_was_active=appliance_was_active,
                        )
                        if appliance_was_active:
                            self._pcf_progress(
                                scan_id,
                                0.02,
                                "Pausing the Noesis/Menon appliance for PCF GPU headroom",
                            )
                            # Record restoration responsibility before issuing the
                            # stop so a phone-tool restart can recover the appliance.
                            self._update_pcf_runtime_lease(
                                scan_id,
                                run_id,
                                pause_requested=True,
                            )
                            self._systemctl_user(
                                "stop", PCF_APPLIANCE_TARGET, timeout_s=150
                            )
                            appliance_paused = True
                            self._update_pcf_runtime_lease(
                                scan_id,
                                run_id,
                                appliance_paused=True,
                            )
                            if self._user_unit_active("noesis-appliance.service"):
                                raise RuntimeError(
                                    "native Noesis remained active after the PCF GPU pause"
                                )
                    result = self.pcf_runner(
                        self.scan_dir(scan_id),
                        run_root,
                        state,
                        target,
                        self.settings.mapanything,
                        lambda fraction, message: self._pcf_progress(
                            scan_id, 0.04 + 0.90 * float(fraction), message
                        ),
                    )
                except Exception as exc:
                    runner_error = exc
                finally:
                    if appliance_was_active:
                        try:
                            self._pcf_progress(
                                scan_id,
                                0.96,
                                "Restoring the native Noesis/Menon appliance",
                            )
                            self._systemctl_user(
                                "start", PCF_APPLIANCE_TARGET, timeout_s=300
                            )
                            if not self._user_unit_active("noesis-appliance.service"):
                                raise RuntimeError(
                                    "native Noesis was not active after appliance restore"
                                )
                        except Exception as exc:
                            runtime_restore_error = exc
                        finally:
                            self._update_pcf_runtime_lease(
                                scan_id,
                                run_id,
                                restore_passed=runtime_restore_error is None,
                                restore_error=(
                                    None
                                    if runtime_restore_error is None
                                    else f"{type(runtime_restore_error).__name__}: "
                                    f"{runtime_restore_error}"
                                ),
                            )
                if runner_error is not None:
                    if runtime_restore_error is not None:
                        raise RuntimeError(
                            f"{runner_error}; appliance restore also failed: "
                            f"{runtime_restore_error}"
                        ) from runner_error
                    raise runner_error
                if result is None:
                    raise RuntimeError("PCF runner returned no result")
                result["runtime_lease"] = {
                    "configured": bool(self.settings.pcf_pause_appliance),
                    "appliance_target": PCF_APPLIANCE_TARGET,
                    "appliance_was_active": appliance_was_active,
                    "appliance_paused": appliance_paused,
                    "restore_passed": runtime_restore_error is None,
                }
                if runtime_restore_error is not None:
                    result["runtime_restore_error"] = (
                        f"{type(runtime_restore_error).__name__}: "
                        f"{runtime_restore_error}"
                    )
                saved_result = _pcf_paths_for_state(run_id, result)
                with self._lock(scan_id):
                    latest = self._read_state_unlocked(scan_id)
                    completed = dict(latest.get("pcf") or {})
                    if completed.get("run_id") != run_id:
                        raise RuntimeError("PCF job identity changed during execution")
                    completed.update(
                        {
                            "status": "complete",
                            "progress": 1.0,
                            "message": (
                                "PCF review candidate and diagnostics are saved"
                                if runtime_restore_error is None
                                else "PCF is saved, but the native appliance needs attention"
                            ),
                            "error": None,
                            "results": saved_result,
                            "updated_at": _utc_now(),
                        }
                    )
                    latest["pcf"] = completed
                    self._write_state_unlocked(scan_id, latest)
            except Exception as exc:
                try:
                    state = self.read_state(scan_id)
                    failed = dict(state.get("pcf") or {})
                    if failed.get("run_id") != run_id:
                        return
                    log_path = run_root / "pcf_run.log"
                    failed.update(
                        {
                            "status": "failed",
                            "progress": 0.0,
                            "message": (
                                "PCF failed; the DA3 walk, alignment, and completed "
                                "stage outputs are preserved"
                            ),
                            "error": f"{type(exc).__name__}: {exc}",
                            "updated_at": _utc_now(),
                        }
                    )
                    if log_path.is_file():
                        failed["log"] = (
                            Path("runs") / run_id / log_path.name
                        ).as_posix()
                    self.update_state(scan_id, pcf=failed)
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
            if any(
                isinstance(row, dict)
                and row.get("status") in SUPPLEMENT_RUNNING_STATUSES
                for row in state.get("supplements") or []
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A running added video cannot be deleted; wait for it to finish or fail",
                )
            pcf = state.get("pcf")
            if isinstance(pcf, dict) and pcf.get("status") in PCF_RUNNING_STATUSES:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A running PCF job cannot be deleted; wait for it to finish or fail",
                )
            scan_dir = self.scan_dir(scan_id)
            if scan_dir.parent != self.settings.storage_root:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsafe scan path")
            shutil.rmtree(scan_dir)
            pcf_scan_dir = self.pcf_scan_dir(scan_id)
            if pcf_scan_dir.exists():
                if pcf_scan_dir.parent != self.pcf_storage_root:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsafe PCF scan path")
                shutil.rmtree(pcf_scan_dir)
        with self._locks_guard:
            self._scan_locks.pop(scan_id, None)

    def delete_supplement(self, scan_id: str, supplement_id: str) -> None:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            index = self._supplement_index(state, supplement_id)
            supplement = state["supplements"][index]
            if supplement.get("status") in SUPPLEMENT_RUNNING_STATUSES:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A running added video cannot be deleted",
                )
            result = supplement.get("results")
            revision_id = result.get("revision_id") if isinstance(result, dict) else None
            if revision_id:
                for other in state["supplements"]:
                    if other is supplement or not isinstance(other, dict):
                        continue
                    other_result = other.get("results")
                    if (
                        isinstance(other_result, dict)
                        and other_result.get("parent_revision_id") == revision_id
                    ):
                        raise HTTPException(
                            status.HTTP_409_CONFLICT,
                            "Delete the newer dependent added video first",
                        )
            supplement_dir = self.supplement_dir(scan_id, supplement_id)
            if supplement_dir.exists():
                shutil.rmtree(supplement_dir)
            del state["supplements"][index]
            active = state.get("active_revision")
            if isinstance(active, dict) and active.get("supplement_id") == supplement_id:
                parent_revision_id = (
                    result.get("parent_revision_id") if isinstance(result, dict) else None
                )
                replacement = None
                for row in state["supplements"]:
                    row_result = row.get("results") if isinstance(row, dict) else None
                    if (
                        isinstance(row_result, dict)
                        and row_result.get("revision_id") == parent_revision_id
                    ):
                        replacement = public_active_revision(row_result)
                        break
                if replacement is None:
                    state.pop("active_revision", None)
                else:
                    state["active_revision"] = replacement
            self._write_state_unlocked(scan_id, state)


def _asset_url(scan_id: str, relative_path: str) -> str:
    return f"/assets/{scan_id}/{quote(relative_path, safe='/')}"


def _pcf_asset_url(scan_id: str, relative_path: str) -> str:
    return f"/pcf-assets/{scan_id}/{quote(relative_path, safe='/')}"


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
    supplements = public.get("supplements")
    if isinstance(supplements, list):
        for supplement in supplements:
            if not isinstance(supplement, dict):
                continue
            supplement_video = supplement.get("video")
            if (
                isinstance(supplement_video, dict)
                and isinstance(supplement_video.get("path"), str)
            ):
                supplement_video["url"] = _asset_url(
                    scan_id, supplement_video["path"]
                )
            supplement_prepared = supplement.get("prepared")
            if isinstance(supplement_prepared, dict):
                for key in ("contact_sheet", "manifest"):
                    if isinstance(supplement_prepared.get(key), str):
                        supplement_prepared[f"{key}_url"] = _asset_url(
                            scan_id, supplement_prepared[key]
                        )
                for frame in supplement_prepared.get("frames") or []:
                    if not isinstance(frame, dict):
                        continue
                    for key in ("frame", "thumbnail"):
                        if isinstance(frame.get(key), str):
                            frame[f"{key}_url"] = _asset_url(scan_id, frame[key])
            results = supplement.get("results")
            if isinstance(results, dict):
                artifacts = results.get("artifacts")
                if isinstance(artifacts, dict):
                    results["artifact_urls"] = {
                        key: _asset_url(scan_id, value)
                        for key, value in artifacts.items()
                        if isinstance(value, str)
                    }
                for item in results.get("files") or []:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        item["url"] = _asset_url(scan_id, item["path"])
                for frame in results.get("capture_views") or []:
                    if not isinstance(frame, dict):
                        continue
                    for key in ("source_frame", "raw_npz"):
                        if isinstance(frame.get(key), str):
                            frame[f"{key}_url"] = _asset_url(scan_id, frame[key])
    active_revision = public.get("active_revision")
    if isinstance(active_revision, dict):
        artifacts = active_revision.get("artifacts")
        if isinstance(artifacts, dict):
            active_revision["artifact_urls"] = {
                key: _asset_url(scan_id, value)
                for key, value in artifacts.items()
                if isinstance(value, str)
            }
    pcf = public.get("pcf")
    if isinstance(pcf, dict):
        if isinstance(pcf.get("log"), str):
            pcf["log_url"] = _pcf_asset_url(scan_id, pcf["log"])
        results = pcf.get("results")
        if isinstance(results, dict):
            artifacts = results.get("artifacts")
            if isinstance(artifacts, dict):
                results["artifact_urls"] = {
                    key: _pcf_asset_url(scan_id, value)
                    for key, value in artifacts.items()
                    if isinstance(value, str)
                }
            files = results.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        item["url"] = _pcf_asset_url(scan_id, item["path"])
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
    inference_runner: InferenceRunner = run_adaptive_mapanything_scan,
    da3_inference_runner: DA3InferenceRunner = run_adaptive_da3_phone_scan,
    alignment_runner: AlignmentRunner = run_noesis_alignment,
    supplement_runner: SupplementRunner = run_supplement_integration,
    pcf_runner: PCFRunner = run_pcf_review_candidate,
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
        supplement_runner=supplement_runner,
        pcf_runner=pcf_runner,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.recover_interrupted_states()
        yield
        service.shutdown()

    app = FastAPI(
        title="Noesis Multi-View Phone Scan",
        version="1.8.1",
        lifespan=lifespan,
    )
    app.state.phone_scan_service = service

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(configured.static_root / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        usage = shutil.disk_usage(configured.storage_root)
        pcf_usage = shutil.disk_usage(service.pcf_storage_root)
        return {
            "status": "ok",
            "version": app.version,
            "storage_free_bytes": int(usage.free),
            "pcf_storage_free_bytes": int(pcf_usage.free),
            "pcf_review_available": True,
            "pcf_pauses_appliance": bool(configured.pcf_pause_appliance),
            "frame_selection": "adaptive_keyframe_selection_v1",
            "candidate_fps": configured.frame.candidate_fps,
            "max_candidate_frames": configured.frame.max_candidate_frames,
            "max_selected_frames": configured.frame.max_selected_frames,
            "mapanything_max_joint_views": configured.mapanything.max_joint_views,
            "mapanything_window_overlap_views": configured.mapanything.window_overlap_views,
            "da3_max_joint_views": configured.da3.max_joint_views,
            "da3_window_overlap_views": configured.da3.window_overlap_views,
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
            "alignment_release_id": configured.alignment_release_id,
            "alignment_targets": service.public_alignment_targets(),
        }

    @app.get("/api/scans")
    async def list_scans() -> list[dict[str, Any]]:
        return [_public_state(state) for state in service.list_states()]

    @app.get("/api/scans/{scan_id}")
    async def get_scan(scan_id: str) -> dict[str, Any]:
        return _public_state(service.read_state(scan_id))

    @app.patch("/api/scans/{scan_id}")
    async def rename_scan(
        scan_id: str,
        name: str = Query(min_length=1, max_length=MAX_SCAN_NAME_LENGTH),
    ) -> dict[str, Any]:
        return _public_state(service.rename_scan(scan_id, name))

    @app.post("/api/scans", status_code=status.HTTP_201_CREATED)
    async def create_scan_from_video(
        request: Request,
        name: str = Query(
            default="Phone room walk",
            max_length=MAX_SCAN_NAME_LENGTH,
        ),
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

    @app.post(
        "/api/scans/{scan_id}/supplements",
        status_code=status.HTTP_201_CREATED,
    )
    async def add_video_to_scan(scan_id: str, request: Request) -> JSONResponse:
        service.read_state(scan_id)
        filename = unquote(request.headers.get("x-file-name") or "additional_walk.mp4")
        filename = Path(filename).name[:160] or "additional_walk.mp4"
        content_type = request.headers.get("content-type") or "application/octet-stream"
        suffix = _video_suffix(filename, content_type)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, "Invalid Content-Length"
                ) from exc
            if declared > configured.max_upload_bytes:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "Video exceeds the upload limit",
                )

        supplement_id = (
            f"add-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        supplement_dir = service.supplement_dir(scan_id, supplement_id)
        supplement_dir.parent.mkdir(parents=True, exist_ok=True)
        supplement_dir.mkdir(exist_ok=False)
        video_name = f"additional_walk{suffix}"
        video_path = supplement_dir / video_name
        relative_video = video_path.relative_to(service.scan_dir(scan_id)).as_posix()
        temporary = supplement_dir / f".{video_name}.uploading"
        now = _utc_now()
        supplement = {
            "schema": "noesis.phone_scan.supplement.state.v1",
            "id": supplement_id,
            "created_at": now,
            "updated_at": now,
            "status": "uploading",
            "progress": 0.0,
            "message": "Receiving additional room video",
            "error": None,
            "video": {
                "path": relative_video,
                "original_name": filename,
                "content_type": content_type,
            },
        }
        state_added = False
        try:
            service.add_supplement(scan_id, supplement)
            state_added = True
            received = 0
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
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, "The uploaded video is empty"
                )
            os.replace(temporary, video_path)
            state = service.update_supplement(
                scan_id,
                supplement_id,
                status="processing_frames",
                progress=0.0,
                message="Additional video saved; preparing new reconstruction views",
                video={
                    **supplement["video"],
                    "size_bytes": received,
                    "sha256": _sha256_file(video_path),
                },
            )
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if supplement_dir.exists():
                shutil.rmtree(supplement_dir)
            if state_added:
                try:
                    service.discard_supplement_state(scan_id, supplement_id)
                except HTTPException:
                    pass
            raise
        service.submit_supplement_preparation(scan_id, supplement_id)
        return JSONResponse(_public_state(state), status_code=status.HTTP_201_CREATED)

    @app.post(
        "/api/scans/{scan_id}/supplements/{supplement_id}/integrate",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def integrate_added_video(
        scan_id: str, supplement_id: str
    ) -> JSONResponse:
        return JSONResponse(
            _public_state(
                service.initiate_supplement_integration(scan_id, supplement_id)
            ),
            status_code=status.HTTP_202_ACCEPTED,
        )

    @app.delete(
        "/api/scans/{scan_id}/supplements/{supplement_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_added_video(scan_id: str, supplement_id: str) -> Response:
        service.delete_supplement(scan_id, supplement_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

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
    async def initiate_alignment(
        scan_id: str,
        camera_id: str = Query(min_length=1, max_length=160),
    ) -> JSONResponse:
        return JSONResponse(
            _public_state(service.initiate_alignment(scan_id, camera_id)),
            status_code=status.HTTP_202_ACCEPTED,
        )

    @app.post("/api/scans/{scan_id}/initiate-pcf", status_code=status.HTTP_202_ACCEPTED)
    async def initiate_pcf(scan_id: str) -> JSONResponse:
        return JSONResponse(
            _public_state(service.initiate_pcf(scan_id)),
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
    app.mount(
        "/pcf-assets",
        StaticFiles(directory=service.pcf_storage_root),
        name="phone-scan-pcf-assets",
    )
    return app


app = create_app()


__all__ = ["PhoneScanService", "PhoneScanSettings", "app", "create_app"]
