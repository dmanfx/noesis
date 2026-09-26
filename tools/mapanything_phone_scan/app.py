from __future__ import annotations

import asyncio
import json
import hashlib
import math
import os
import re
import shutil
import ssl
import stat
import subprocess
import tarfile
import threading
import time
import uuid
import zipfile
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
from .paired_static_reference import (
    prepare_paired_static_reference,
    validate_paired_static_reference,
)
from .windowed_inference import run_adaptive_mapanything_scan
from .windowed_da3_inference import run_adaptive_da3_phone_scan
from .pcf import run_pcf_review_candidate
from .processing import FramePreparationSettings, prepare_video_frames
from .phone_calibration import calibration_summary
from .calibration_jobs import CalibrationJobs, CalibrationJobBusy, CalibrationJobError, MAX_REQUEST_BYTES, _write as _write_calibration_json
from .calibrated_walk import CameraSelection
from .motion_profile import MotionSelection
from .imu_calibration import MAX_BUNDLE_BYTES as MAX_IMU_BUNDLE_BYTES, ImuCalibrationError, retain_imu_bundle
from .prepared_frame_identity import prepared_frame_identity
from .capture import CaptureImportLimits, CaptureImportError, import_capture_bundle, validate_capture_manifest
from .capture_upload import (
    ASYNC_VIDEO_PROBE_TIMEOUT_S,
    UPLOAD_RECEIPT_SCHEMA,
    CaptureUploadBusy,
    CaptureUploadQueue,
)
from .browser_capture import BROWSER_CAPTURE_SCHEMA
from .walk_intent import effective_walk_intent, validate_walk_intent
from .path_review import build_path_review
from .path_reference import PcfPathReferences
from .companion_capture import (
    COMPANION_SESSION_PATTERN,
    CompanionCaptureBusy,
    CompanionCaptureConflict,
    CompanionCaptureError,
    CompanionCaptureLimits,
    CompanionCaptureManager,
)
from .vio import (
    VIOError,
    VIOSettings,
    materialize_openvins_input,
    run_openvins,
    validate_vio_input,
    validate_vio_result,
)
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
    "importing_capture",
    "processing_frames",
    "ma_queued",
    "ma_running",
    "da3_queued",
    "da3_running",
    "vio_queued",
    "vio_running",
}
SUPPLEMENT_RUNNING_STATUSES = {"uploading", "processing_frames", "queued", "running"}
PCF_RUNNING_STATUSES = {"queued", "running"}
PCF_APPLIANCE_TARGET = "menon-appliance.target"
MAX_CA_CERT_BYTES = 256 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _motion_profile_capability(settings: VIOSettings) -> tuple[bool, str | None]:
    """Configuration availability only; never start a native estimator here."""
    if settings.estimator != "openvins":
        return False, "Motion-profile validation requires the configured OpenVINS estimator."
    if settings.executable is None or not settings.executable.is_file() or not os.access(settings.executable, os.X_OK):
        return False, "OpenVINS executable is missing or not executable. Configure NOESIS_PHONE_SCAN_VIO_EXECUTABLE; retained short recordings need not be repeated."
    if settings.config is None or not settings.config.is_file() or not os.access(settings.config, os.R_OK):
        return False, "OpenVINS base configuration is missing or unreadable. Configure NOESIS_PHONE_SCAN_VIO_CONFIG, then retry with the retained jobs."
    # CalibrationJobs/MotionSelection resolve the environment; the normal-walk
    # worker holds startup settings. Those must name the same native consumer.
    try:
        current = VIOSettings.from_env()
        if current.estimator != settings.estimator or current.executable is None or current.config is None or current.executable.resolve() != settings.executable.resolve() or current.config.resolve() != settings.config.resolve():
            return False, "The motion-profile backend and room-walk worker have different OpenVINS settings. Restore one matching executable and base configuration before retrying the retained jobs."
    except (OSError, ValueError) as exc:
        return False, f"OpenVINS configuration could not be verified: {exc}"
    return True, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_display_name(camera_id: str) -> str:
    return " ".join(part.capitalize() for part in camera_id.replace("_", "-").split("-"))


def _finalize_alignment_paths(build_dir: Path, final_dir: Path, result: dict[str, Any]) -> None:
    """Keep reported artifact locations valid after the atomic directory move."""
    paths = result.get("artifact_paths")
    if not isinstance(paths, dict):
        return
    published: dict[str, str] = {}
    for key, value in paths.items():
        source = Path(str(value)).resolve()
        try:
            relative = source.relative_to(build_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("Alignment output artifact is outside its build directory") from exc
        if not source.is_file():
            raise RuntimeError(f"Alignment output artifact is missing: {key}")
        published[key] = str((final_dir / relative).resolve())
    report_path = build_dir / "alignment_report.json"
    report = _read_json_object(report_path, label="alignment report")
    report["output_dir"] = str(final_dir.resolve())
    report["artifact_paths"] = published
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result["artifact_paths"] = published
    for row in result.get("files") or []:
        if isinstance(row, dict) and row.get("path") == "alignment/alignment_report.json":
            row["size_bytes"] = report_path.stat().st_size
            row["sha256"] = _sha256_file(report_path)


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


def _phone_scan_secure_url() -> str:
    hostname = (
        os.environ.get("NOESIS_PHONE_SCAN_TLS_HOSTNAME", "TauntonMainframe.local").strip()
        or "TauntonMainframe.local"
    )
    port = _env_int("NOESIS_PHONE_SCAN_HTTPS_PORT", 8789, minimum=1)
    return f"https://{hostname}:{port}"


def _phone_scan_ca_certificate_path() -> Path:
    # Keep the launcher and API endpoint on one configured public-CA path.
    # Import lazily because the launcher imports this module from main().
    from .__main__ import _ca_certificate_path

    return _ca_certificate_path()


def _validated_ca_certificate(path: Path) -> None:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "The appliance CA certificate is not installed; use the HTTPS listener's configured certificate setup.",
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The configured CA certificate is not a regular file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_CA_CERT_BYTES:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "The configured CA certificate is missing or exceeds the safety limit")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The appliance CA certificate could not be read") from exc
    if b"PRIVATE KEY" in data or b"BEGIN RSA" in data or b"BEGIN EC" in data:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "The configured CA certificate contains private-key material")
    blocks = re.findall(
        rb"-----BEGIN CERTIFICATE-----\s*.*?-----END CERTIFICATE-----",
        data,
        flags=re.DOTALL,
    )
    if (
        not blocks
        or data.count(b"-----BEGIN CERTIFICATE-----")
        != data.count(b"-----END CERTIFICATE-----")
    ):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "The configured CA certificate is not a valid PEM certificate")
    try:
        for block in blocks:
            ssl.PEM_cert_to_DER_cert(block.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "The configured CA certificate is not a valid PEM certificate") from exc


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
    scene_prior_catalog: Path | None = None
    capture_limits: CaptureImportLimits = CaptureImportLimits()
    vio: VIOSettings = VIOSettings()
    companion_limits: CompanionCaptureLimits = CompanionCaptureLimits()

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
        phone_profile_raw = os.environ.get("NOESIS_PHONE_SCAN_CAMERA_CALIBRATION", "").strip()
        phone_profile = _resolve_root(phone_profile_raw, REPO_ROOT) if phone_profile_raw else None
        phone_capture_mode = os.environ.get("NOESIS_PHONE_SCAN_CALIBRATED_CAPTURE_MODE", "unbound").strip()
        if phone_capture_mode not in {"unbound", "uploaded_video", "browser", "native_sensor_bundle"}:
            raise ValueError("NOESIS_PHONE_SCAN_CALIBRATED_CAPTURE_MODE is not a supported capture mode")
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
                phone_camera_calibration=phone_profile,
                phone_camera_capture_mode=phone_capture_mode,
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
            scene_prior_catalog=_resolve_root(
                os.environ.get("NOESIS_PHONE_SCAN_SCENE_PRIOR_CATALOG"),
                REPO_ROOT / "data" / "scene_priors" / "catalog.json",
            ),
            capture_limits=CaptureImportLimits(
                max_archive_bytes=_env_int(
                    "NOESIS_PHONE_SCAN_MAX_CAPTURE_BYTES", 8 * 1024 * 1024 * 1024, minimum=1024 * 1024
                ),
                max_uncompressed_bytes=_env_int(
                    "NOESIS_PHONE_SCAN_MAX_CAPTURE_UNCOMPRESSED_BYTES", 8 * 1024 * 1024 * 1024, minimum=1024 * 1024
                ),
                max_member_bytes=_env_int(
                    "NOESIS_PHONE_SCAN_MAX_CAPTURE_MEMBER_BYTES", 8 * 1024 * 1024 * 1024, minimum=1024 * 1024
                ),
                max_files=_env_int("NOESIS_PHONE_SCAN_MAX_CAPTURE_FILES", 512, minimum=4),
                max_imu_rows=_env_int("NOESIS_PHONE_SCAN_MAX_CAPTURE_IMU_ROWS", 2_000_000, minimum=2),
                max_video_timestamps=_env_int("NOESIS_PHONE_SCAN_MAX_CAPTURE_VIDEO_FRAMES", 1_000_000, minimum=2),
                max_time_gap_s=_env_float("NOESIS_PHONE_SCAN_MAX_CAPTURE_IMU_GAP_S", 0.25, minimum=0.001),
            ),
            vio=VIOSettings.from_env(),
            companion_limits=CompanionCaptureLimits(
                max_duration_s=_env_float(
                    "NOESIS_PHONE_SCAN_COMPANION_MAX_DURATION_S", 15 * 60.0, minimum=0.1
                ),
                lease_s=_env_float(
                    "NOESIS_PHONE_SCAN_COMPANION_LEASE_S", 45.0, minimum=0.1
                ),
                readiness_timeout_s=_env_float(
                    "NOESIS_PHONE_SCAN_COMPANION_READINESS_TIMEOUT_S", 15.0, minimum=0.1
                ),
                max_session_bytes=_env_int(
                    "NOESIS_PHONE_SCAN_COMPANION_MAX_BYTES",
                    8 * 1024 * 1024 * 1024,
                    minimum=1024 * 1024,
                ),
                max_tracking_records=_env_int(
                    "NOESIS_PHONE_SCAN_COMPANION_MAX_TRACKING_RECORDS",
                    1_000_000,
                    minimum=1,
                ),
                max_packet_records=_env_int(
                    "NOESIS_PHONE_SCAN_COMPANION_MAX_PACKET_RECORDS",
                    2_000_000,
                    minimum=1,
                ),
                max_markers=_env_int(
                    "NOESIS_PHONE_SCAN_COMPANION_MAX_MARKERS", 10_000, minimum=1
                ),
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
VioRunner = Callable[
    [Path, Path, dict[str, Any], dict[str, Any], VIOSettings, Callable[[float, str], None]],
    dict[str, Any],
]


def _default_vio_runner(
    capture_dir: Path,
    output_dir: Path,
    capture_report: dict[str, Any],
    prepared: dict[str, Any],
    settings: VIOSettings,
    progress: Callable[[float, str], None],
) -> dict[str, Any]:
    validate_vio_input(capture_report, prepared)
    materialized_dir, generated_config = materialize_openvins_input(
        capture_dir,
        capture_report,
        output_dir,
        settings,
        progress,
    )
    result = run_openvins(
        materialized_dir,
        output_dir,
        replace(settings, config=generated_config),
        progress,
    )
    # Reconstruction consumes selected-view poses, while a reference walk needs
    # the full camera-time trajectory. Retain both without changing either clock
    # or assigning a phone pose to a person's ground point.
    dense = validate_vio_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / "dense_camera_trajectory.json"
    dense_path.write_text(json.dumps(dense, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    dense_pose_count = len(result.get("poses") or [])
    prepared_by_time = {
        int(row["capture_time_ns"]): row
        for row in prepared.get("frames", [])
        if isinstance(row, dict) and isinstance(row.get("capture_time_ns"), int)
    }
    selected_poses: list[dict[str, Any]] = []
    for pose in result.get("poses", []):
        timestamp = int(pose["capture_time_ns"])
        prepared_row = prepared_by_time.get(timestamp)
        if prepared_row is None:
            continue
        pose = dict(pose)
        pose["prepared_frame_id"] = prepared_frame_identity(
            int(prepared_row["index"]), str(prepared_row["sha256"])
        )
        pose["source_frame_index"] = prepared_row.get("source_frame_index")
        selected_poses.append(pose)
    if len(selected_poses) < 2:
        raise VIOError("OpenVINS produced no two poses matching prepared exact frame timestamps")
    result["poses"] = selected_poses
    quality = dict(result.get("quality") or {})
    quality["dense_pose_count"] = dense_pose_count
    quality["prepared_pose_count"] = len(selected_poses)
    result["quality"] = quality
    return validate_vio_result(result)


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
        paired_static_runner: Callable[..., dict[str, Any]] = prepare_paired_static_reference,
        supplement_runner: SupplementRunner = run_supplement_integration,
        pcf_runner: PCFRunner = run_pcf_review_candidate,
        vio_runner: VioRunner = _default_vio_runner,
        companion_manager: CompanionCaptureManager | None = None,
    ) -> None:
        self.settings = settings
        self.path_references = PcfPathReferences(settings.scene_prior_catalog)
        self.frame_processor = frame_processor
        self.inference_runner = inference_runner
        self.da3_inference_runner = da3_inference_runner
        self.alignment_runner = alignment_runner
        self.paired_static_runner = paired_static_runner
        self.supplement_runner = supplement_runner
        self.pcf_runner = pcf_runner
        self.vio_runner = vio_runner
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
        self._path_review_slots = threading.BoundedSemaphore(2)
        self._inference_lock = threading.Lock()
        self._alignment_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PhoneScan")
        self.capture_uploads = CaptureUploadQueue(self.settings.storage_root)
        self.calibration_jobs = CalibrationJobs(self.settings.storage_root)
        self.camera_selection = CameraSelection(self.calibration_jobs)
        self.motion_selection = MotionSelection(self.calibration_jobs, self.camera_selection)
        self.companion_capture = companion_manager or CompanionCaptureManager(
            self.settings.storage_root,
            limits=self.settings.companion_limits,
        )

    def shutdown(self) -> None:
        self.calibration_jobs.close()
        self.capture_uploads.shutdown()
        self.companion_capture.shutdown()
        self._executor.shutdown(wait=False, cancel_futures=False)

    def companion_cameras(self) -> list[dict[str, Any]]:
        return self.companion_capture.list_cameras()

    def start_companion_capture(
        self,
        camera_id: str,
        *,
        client_request_id: str | None = None,
        phone_capture_id: str | None = None,
        clock_probes: Any | None = None,
    ) -> dict[str, Any]:
        state = self.companion_capture.start_session(
            camera_id,
            client_request_id=client_request_id,
            phone_capture_id=phone_capture_id,
            clock_probes=clock_probes,
        )
        return self.companion_capture.public_state(str(state["session_id"]))

    def companion_status(self, session_id: str) -> dict[str, Any]:
        return self.companion_capture.public_state(session_id)

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

    def set_walk_intent(self, scan_id: str, value: Any) -> dict[str, Any]:
        """Attach a review declaration without rewriting a retained recording."""
        try:
            intent = validate_walk_intent(value)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        target_id = intent["target_scan_id"]
        if target_id == scan_id:
            raise HTTPException(status.HTTP_409_CONFLICT, "A walk cannot target itself")
        if target_id is not None:
            target = self.read_state(target_id)
            if target.get("status") != "complete" or not isinstance(target.get("outputs"), dict):
                raise HTTPException(status.HTTP_409_CONFLICT, "The reference reconstruction is not complete")
            if (target.get("walk_intent") or {}).get("mode") == "path_refinement":
                raise HTTPException(status.HTTP_409_CONFLICT, "Choose a reconstruction, not another path walk")
            if intent.get("target_reference") is not None:
                try:
                    self.path_references.resolve(target_id, intent["target_reference"])
                except (ValueError, OSError) as exc:
                    raise HTTPException(status.HTTP_409_CONFLICT, f"PCF reference is unavailable: {exc}") from exc
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            capture = state.get("capture") or {}
            if capture.get("calibration_request") or state.get("status") == "calibration_ready":
                raise HTTPException(status.HTTP_409_CONFLICT, "Calibration recordings remain calibration evidence")
            recorded = capture.get("walk_intent")
            if recorded is not None and recorded != intent:
                raise HTTPException(status.HTTP_409_CONFLICT, "The recorder's retained walk purpose cannot be relabeled")
            if state.get("status") in {"uploading", "importing_capture"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "Wait for the recording import to finish")
            if (state.get("path_review") or {}).get("status") in {"queued", "running"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "Wait for path review to finish")
            state["walk_intent"] = intent
            state["walk_intent_source"] = "capture_manifest" if recorded else "explicit_review_declaration"
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
            if (state.get("walk_intent") or {}).get("mode") == "path_refinement":
                raise HTTPException(status.HTTP_409_CONFLICT, "Add room views to the reference reconstruction, not a path walk")
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

    def add_retained_capture(self, scan_id: str, source_scan_id: str) -> dict[str, Any]:
        """Reuse a saved capture as added views without consuming its raw evidence."""
        if scan_id == source_scan_id:
            raise HTTPException(status.HTTP_409_CONFLICT, "Choose a different capture to add")
        source = self.read_state(source_scan_id)
        intent = effective_walk_intent(source)
        if (source.get("capture") or {}).get("calibration_request") or source.get("status") == "calibration_ready":
            raise HTTPException(status.HTTP_409_CONFLICT, "Calibration takes are not room-coverage captures")
        if intent and intent["mode"] != "reconstruction":
            raise HTTPException(status.HTTP_409_CONFLICT, "Path walks do not automatically change room geometry")
        if intent and intent["target_scan_id"] not in (None, scan_id):
            raise HTTPException(status.HTTP_409_CONFLICT, "The capture names a different reconstruction")
        if source.get("status") not in {"ready", "complete", "ma_failed", "da3_failed"}:
            raise HTTPException(status.HTTP_409_CONFLICT, "Wait for the saved capture's frame preparation")
        with self._lock(scan_id):
            target = self._read_state_unlocked(scan_id)
            for prior in target.get("supplements") or []:
                if (prior.get("source_capture") or {}).get("scan_id") == source_scan_id:
                    return target
            if (target.get("walk_intent") or {}).get("mode") == "path_refinement":
                raise HTTPException(status.HTTP_409_CONFLICT, "Added views require a reconstruction target")
            source_root = self.scan_dir(source_scan_id)
            relative = (source.get("video") or {}).get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise HTTPException(status.HTTP_409_CONFLICT, "The retained capture has no safe video path")
            source_video = source_root / relative
            if source_video.is_symlink() or not source_video.resolve().is_relative_to(source_root) or not source_video.is_file():
                raise HTTPException(status.HTTP_409_CONFLICT, "The retained video is unavailable")
            supplement_id = f"add-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            output = self.supplement_dir(scan_id, supplement_id)
            output.mkdir(parents=True, exist_ok=False)
            added = False
            try:
                # Hard links preserve the raw bytes even if either library entry
                # is later removed, without copying multi-gigabyte recordings.
                linked_video = output / f"additional_walk{source_video.suffix}"
                os.link(source_video, linked_video)
                capture_root = source_root / "capture"
                if capture_root.is_dir() and (capture_root / "capture_import.json").is_file():
                    members = list(capture_root.iterdir())
                    if len(members) > 512 or any(p.is_symlink() or not p.is_file() for p in members):
                        raise HTTPException(status.HTTP_409_CONFLICT, "Retained capture layout cannot be safely reused; its originals are unchanged")
                    retained = output / "capture"
                    retained.mkdir()
                    for member in members:
                        os.link(member, retained / member.name)
                now = _utc_now()
                row = {"schema": "noesis.phone_scan.supplement.state.v1", "id": supplement_id,
                       "created_at": now, "updated_at": now, "status": "processing_frames", "progress": 0.0,
                       "message": "Preparing additional views from the retained capture", "error": None,
                       "source_capture": {"scan_id": source_scan_id, "capture_id": (source.get("capture") or {}).get("capture_id"),
                                          "archive_sha256": (source.get("upload_receipt") or {}).get("sha256"),
                                          "raw_streams_preserved": True},
                       "video": {"path": linked_video.relative_to(self.scan_dir(scan_id)).as_posix(),
                                 "original_name": source_video.name, "content_type": (source.get("video") or {}).get("content_type"),
                                 "size_bytes": source_video.stat().st_size}}
                # Inline admission under this same lock; add_supplement uses an
                # RLock and preserves the existing reconstruction/job guards.
                target = self.add_supplement(scan_id, row)
                added = True
            except Exception:
                if not added:
                    shutil.rmtree(output)
                raise
        self.submit_supplement_preparation(scan_id, supplement_id)
        return target

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
            if previous == "importing_capture" and state.get("upload_receipt"):
                self.update_state(
                    scan_id,
                    status="import_failed",
                    message="Capture validation was interrupted by a tool restart",
                    error="The uploaded archive is preserved; upload the same saved bundle to retry validation.",
                    upload_import={**(state.get("upload_import") or {}), "phase": "interrupted"},
                    upload_receipt={**state["upload_receipt"], "validation_status": "failed"},
                )
                self.capture_uploads.sync_state(directory)
            elif previous == "processing_frames":
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
            vio = state.get("vio")
            if isinstance(vio, dict) and vio.get("status") in {"queued", "running"}:
                self.update_state(
                    scan_id,
                    vio={
                        **vio,
                        "status": "failed",
                        "progress": 0.0,
                        "message": "OpenVINS was interrupted by a tool restart",
                        "error": "The sensor bundle and RGB reconstruction are preserved; retry OpenVINS.",
                    },
                )
            path_review = state.get("path_review")
            if isinstance(path_review, dict) and path_review.get("status") in {"queued", "running"}:
                self.update_state(scan_id, path_review={**path_review, "status": "failed", "progress": 0.0,
                                                       "message": "Path review was interrupted; original evidence is retained",
                                                       "error": "Run Review path again to create a new report"})
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
                reference = alignment.get("static_reference")
                if isinstance(reference, dict) and reference.get("status") in {"queued", "building"}:
                    reference.update(
                        status="failed",
                        message="The paired static build was interrupted; retry alignment",
                        error="The original paired recording is preserved",
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

    def public_state(self, state: dict[str, Any]) -> dict[str, Any]:
        public = _public_state(state)
        if state.get("status") == "complete" and isinstance(state.get("outputs"), dict):
            reference = self.path_references.for_scan(state["id"])
            if reference is not None:
                public["path_reference"] = reference
        return public

    def _selected_path_reference(self, state: dict[str, Any], target_id: str) -> dict[str, Any] | None:
        intent = effective_walk_intent(state) or {}
        selection = intent.get("target_reference")
        if selection is None:
            # Older captures retain their original raw-reference semantics.
            # Only a recorder's explicit PCF selection selects this branch.
            return None
        reference = self.path_references.resolve(target_id, selection)
        camera_id = (state.get("companion_capture") or {}).get("camera_id")
        if camera_id is not None and camera_id != selection["camera_id"]:
            raise ValueError("The paired camera differs from the selected PCF reference camera")
        return reference

    def submit_frame_preparation(self, scan_id: str) -> None:
        self._executor.submit(self._prepare_worker, scan_id)

    def _new_capture_profile_selection(self, manifest: Any) -> dict[str, Any]:
        """Snapshot once at upload admission, before an asynchronous import waits."""
        if not isinstance(manifest, dict) or not isinstance(manifest.get("android_capture"), dict) or manifest.get("calibration_request"):
            return {}
        try:
            motion = deepcopy(self.motion_selection.get())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A broken optional profile must not discard an otherwise valid RGB
            # recording or silently substitute the unrelated global camera fit.
            return {"motion_calibration_selection": None, "camera_calibration_selection": None,
                    "motion_selection_error": f"Motion selection could not be read: {exc}"[:2000]}
        camera = (deepcopy(motion.get("camera_selection")) if isinstance(motion, dict) else None) if motion is not None else deepcopy(self.camera_selection.get())
        return {"motion_calibration_selection": motion, "camera_calibration_selection": camera}

    @staticmethod
    def _has_motion_selection(capture: dict[str, Any]) -> bool:
        return capture.get("motion_calibration_selection") is not None or bool(capture.get("motion_selection_error"))

    def _derive_motion_capture(
        self, scan_id: str, capture: dict[str, Any], camera_preparation: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reverify the import's snapshot from raw evidence; never read our cache.

        Failure blocks only metric VIO. Original capture_import.json, raw sensor
        data and prepared frame identities are never rewritten by this method.
        """
        capture_dir = self.scan_dir(scan_id) / "capture"
        raw_path = capture_dir / "capture_import.json"
        derived_path = capture_dir / "motion_profile_import.json"
        updated = deepcopy(capture)
        selection = capture.get("motion_calibration_selection")
        raw: dict[str, Any] = {}
        source_sha256 = None
        try:
            if raw_path.is_symlink():
                raise ValueError("Raw capture import report cannot be a symbolic link")
            raw = _read_json_object(raw_path, label="Raw capture import report")
            source_sha256 = _sha256_file(raw_path)
            if capture.get("capture_kind") != "android_camera_imu" or capture.get("calibration_request"):
                raise ValueError("A selected motion profile applies only to an ordinary native Android room walk")
            if capture.get("motion_selection_error"):
                raise ValueError(capture["motion_selection_error"])
            if not isinstance(selection, dict) or not isinstance(selection.get("camera_selection"), dict):
                raise ValueError("Selected motion profile has no associated camera-selection snapshot")
            if selection["camera_selection"] != capture.get("camera_calibration_selection"):
                raise ValueError("Motion and frame-preparation camera selections differ; do not reuse these prepared views for VIO")
            available, reason = _motion_profile_capability(self.settings.vio)
            if not available:
                raise ValueError(reason)
            # The owner revalidates job/profile hashes and capture binding using
            # this immutable snapshot, not the current global selector.
            derived = self.motion_selection.derived_report(capture_dir, deepcopy(raw), deepcopy(selection))
            if not isinstance(derived, dict) or not isinstance(derived.get("calibration"), dict):
                raise ValueError("Motion profile returned an invalid derived capture report")
            if not isinstance(camera_preparation, dict) or camera_preparation.get("status") != "applied":
                raise ValueError("The associated camera profile was not applied to frame preparation. Review its binding reasons and record a new matching walk.")
            derived = deepcopy(derived)
            summary = dict(derived.get("motion_profile") or {})
            allowed = derived.get("metric_vio_allowed") is True
            summary.update({"status": "applied" if allowed else "not_applied",
                            "motion_calibration_id": selection.get("motion_calibration_id"),
                            "profile_id": selection.get("profile_id"),
                            "metric_vio_allowed": allowed})
            if not allowed:
                summary.setdefault("reason_codes", ["motion_profile_binding_not_admitted"])
                summary.setdefault("message", "Selected motion profile did not admit this capture. Check the report and phone configuration; RGB reconstruction remains available.")
            else:
                summary["message"] = "Profile reverified and bound to this walk. Run OpenVINS and review the first normal walk; a profile match is not a room-accuracy certificate."
        except (OSError, ValueError, KeyError, TypeError) as exc:
            derived = deepcopy(raw)
            allowed = False
            summary = {"status": "not_applied", "metric_vio_allowed": False,
                       "motion_calibration_id": selection.get("motion_calibration_id") if isinstance(selection, dict) else None,
                       "profile_id": selection.get("profile_id") if isinstance(selection, dict) else None,
                       "reason_codes": ["motion_profile_verification_failed"],
                       "message": f"Motion profile not applied: {exc}"[:2000],
                       "next_action": "Check the selected profile and exact phone/lens/locked-focus configuration. Restore missing profile artifacts and retry OpenVINS to reverify, or capture a new matching walk. RGB reconstruction and the original recording are retained."}
            derived["calibration"] = {**dict(raw.get("calibration") or {}), "complete_for_metric_vio": False}
        derived["metric_vio_allowed"] = allowed
        if not allowed:
            derived["calibration"]["complete_for_metric_vio"] = False
        derived["motion_profile"] = summary
        derived["motion_profile_source"] = {"import_report": "capture/capture_import.json", "sha256": source_sha256}
        # This is a derived cache/report only. Every VIO request and worker run
        # derives it again from the raw report and verified selected artifacts.
        try:
            _write_calibration_json(derived_path, derived)
            updated["motion_profile_import"] = "capture/motion_profile_import.json"
        except (OSError, ValueError, TypeError) as exc:
            allowed = False
            derived["metric_vio_allowed"] = False
            derived["calibration"]["complete_for_metric_vio"] = False
            summary.update(status="not_applied", metric_vio_allowed=False,
                           reason_codes=["motion_profile_report_write_failed"],
                           message=f"Could not retain the derived motion report: {exc}. Restore writable scan storage and retry OpenVINS; RGB preparation remains available."[:2000])
            updated.pop("motion_profile_import", None)
        updated.update(metric_vio_allowed=allowed, calibration=deepcopy(derived["calibration"]),
                       motion_profile=summary)
        return derived, updated

    def initiate_calibration(self, scan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if (state.get("capture") or {}).get("capture_kind") != "android_camera_imu":
                raise CalibrationJobError("Calibration requires the native RoomWalk camera and IMU evidence")
            return self.calibration_jobs.submit(scan_id, self.scan_dir(scan_id) / "capture", payload)

    def _progress(self, scan_id: str, fraction: float, message: str) -> None:
        try:
            self.update_state(
                scan_id,
                progress=float(min(1.0, max(0.0, fraction))),
                message=str(message),
            )
        except HTTPException:
            return

    def initiate_vio(self, scan_id: str) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if (state.get("path_review") or {}).get("status") in {"queued", "running"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "Path refinement already owns this walk's sensor processing")
            capture = state.get("capture")
            if not isinstance(capture, dict):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "OpenVINS requires an imported camera and IMU capture bundle",
                )
            if state.get("status") not in {"ready", "complete"}:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"OpenVINS cannot start while frame preparation status is {state.get('status')}",
                )
            existing = state.get("vio")
            if isinstance(existing, dict) and existing.get("status") in {"queued", "running"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "OpenVINS is already running")
            if capture.get("schema") == BROWSER_CAPTURE_SCHEMA:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "OpenVINS is blocked for browser camera + IMU captures; browser callback timestamps are not native acquisition timestamps",
                )
            if self._has_motion_selection(capture):
                _, capture = self._derive_motion_capture(scan_id, capture, state.get("camera_calibration_selection"))
                state["capture"] = capture
                self._write_state_unlocked(scan_id, state)
            if capture.get("metric_vio_allowed") is not True:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    (capture.get("motion_profile") or {}).get("message") or "Metric OpenVINS is blocked because capture calibration or timing is incomplete",
                )
            vio = {
                "schema": "noesis.phone_capture.vio_job.v1",
                "estimator": "openvins",
                "status": "queued",
                "progress": 0.0,
                "message": "OpenVINS queued",
                "error": None,
            }
            state["vio"] = vio
            self._write_state_unlocked(scan_id, state)
        self._executor.submit(self._vio_worker, scan_id)
        return state

    def initiate_path_review(self, scan_id: str, target_scan_id: str | None = None) -> dict[str, Any]:
        with self._lock(scan_id):
            state = self._read_state_unlocked(scan_id)
            if state.get("status") != "complete" or not isinstance(state.get("outputs"), dict):
                raise HTTPException(status.HTTP_409_CONFLICT, "Run the visual model first so the walk has a camera trajectory")
            if (state.get("capture") or {}).get("calibration_request"):
                raise HTTPException(status.HTTP_409_CONFLICT, "A calibration take is not a path-reference walk")
            if (state.get("path_review") or {}).get("status") in {"queued", "running"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "Path review is already running")
            if any((state.get(key) or {}).get("status") in {"queued", "running"} for key in ("vio", "alignment", "pcf")):
                raise HTTPException(status.HTTP_409_CONFLICT, "Wait for this walk's active processing before refining its path")
            intent = effective_walk_intent(state)
            bound_target = (intent or {}).get("target_scan_id")
            if target_scan_id is not None and bound_target and target_scan_id != bound_target:
                raise HTTPException(status.HTTP_409_CONFLICT, "Review target differs from the recorded room")
            target_id = target_scan_id or bound_target or scan_id
        # Avoid holding locks on two different scans at once.
        target = self.read_state(target_id)
        if target.get("status") != "complete" or not isinstance(target.get("outputs"), dict):
            raise HTTPException(status.HTTP_409_CONFLICT, "The reference reconstruction is unavailable")
        if (target.get("alignment") or {}).get("status") in {"queued", "running"}:
            raise HTTPException(status.HTTP_409_CONFLICT, "Wait for the reference room alignment to finish")
        if target_id != scan_id and (target.get("walk_intent") or {}).get("mode") == "path_refinement":
            raise HTTPException(status.HTTP_409_CONFLICT, "Choose the room reconstruction as the reference")
        try:
            self._selected_path_reference(state, target_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"PCF reference needs attention: {exc}") from exc
        if not self._path_review_slots.acquire(blocking=False):
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Two path reviews are already queued or running; retry when one finishes")
        try:
            with self._lock(scan_id):
                state = self._read_state_unlocked(scan_id)
                if (state.get("path_review") or {}).get("status") in {"queued", "running"}:
                    raise HTTPException(status.HTTP_409_CONFLICT, "Path review is already running")
                if any((state.get(key) or {}).get("status") in {"queued", "running"} for key in ("vio", "alignment", "pcf")):
                    raise HTTPException(status.HTTP_409_CONFLICT, "This walk's processing changed; wait and retry path refinement")
                state["path_review"] = {"status": "queued", "progress": 0.0, "target_scan_id": target_id,
                                        "message": "Queued trajectory refinement and paired Noesis comparison", "error": None}
                self._write_state_unlocked(scan_id, state)
            try:
                self._executor.submit(self._path_review_worker, scan_id, target_id)
            except Exception as exc:
                self.update_state(scan_id, path_review={**state["path_review"], "status": "failed",
                    "message": "Path review could not be queued; retry when the service is available",
                    "error": f"{type(exc).__name__}: {exc}"[:2000]})
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                    "Path review could not be queued; original recordings are unchanged") from exc
        except BaseException:
            self._path_review_slots.release()
            raise
        return state

    def _path_review_worker(self, scan_id: str, target_scan_id: str) -> None:
        relative = f"path_review/run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        try:
            state, target = self.read_state(scan_id), self.read_state(target_scan_id)
            reference = self._selected_path_reference(state, target_scan_id)
            self.update_state(scan_id, path_review={**state["path_review"], "status": "running", "progress": 0.1,
                                                   "message": "Refining visual constraints and registering the paired path"})
            if (state.get("capture") or {}).get("metric_vio_allowed") is True and (state.get("vio") or {}).get("status") != "complete":
                self.update_state(scan_id, vio={"schema": "noesis.phone_capture.vio_job.v1", "estimator": "openvins",
                    "status": "queued", "progress": 0.0, "message": "Path refinement requested the qualified visual-inertial estimator", "error": None})
                self._vio_worker(scan_id)
                state = self.read_state(scan_id)
            session_id = (state.get("companion_capture") or {}).get("session_id")
            companion_dir = self.companion_capture.session_root / session_id if isinstance(session_id, str) and COMPANION_SESSION_PATTERN.fullmatch(session_id) else None

            def revalidate_scaled_carrier(candidate_raw: Path, candidate_manifest: Path,
                                          revalidation_dir: Path, scale_factor: float) -> dict[str, Any]:
                # A qualified VIO scale correction must register its changed
                # geometry independently. It cannot reuse the original transform
                # or fit itself to the Noesis person being evaluated.
                saved_target = self._saved_alignment_target(scan_id, state.get("alignment") or {})
                revalidation_dir.mkdir(parents=True, exist_ok=False)
                candidate = json.loads(candidate_manifest.read_text(encoding="utf-8"))
                with self._alignment_lock:
                    result = run_noesis_alignment(self.scan_dir(scan_id), revalidation_dir, candidate, saved_target,
                        lambda _fraction, _message: None, source_raw_root=candidate_raw,
                        source_output_manifest=candidate_manifest)
                return {**result, "transform_path": str(revalidation_dir / "phone_ma_to_noesis_world.json"),
                        "source_raw": str(candidate_raw), "source_manifest": str(candidate_manifest),
                        "scale_factor": float(scale_factor)}

            report = build_path_review(self.scan_dir(scan_id), state, self.scan_dir(target_scan_id), target,
                                       self.scan_dir(scan_id) / relative, companion_dir=companion_dir,
                                       target_reference=reference,
                                       revalidate_scaled_carrier=revalidate_scaled_carrier)
            results = {key: report[key] for key in ("schema", "status", "review_only", "visual_path", "imu_consistency",
                                                    "path_refinement", "path_comparison", "reference_reconstruction", "sensor_refined_path", "accuracy", "body_ground_reference", "paired_noesis", "limitations")}
            results["artifact"] = f"{relative}/path_review.json"
            results["artifacts"] = {key: f"{relative}/{value}" for key, value in report["artifacts"].items()}
            self.update_state(scan_id, path_review={"status": "complete", "progress": 1.0, "target_scan_id": target_scan_id,
                                                   "message": ("Registered phone/Noesis comparison is ready; separation is not certified accuracy"
                                                               if report["path_comparison"]["status"] == "comparison_ready" else
                                                               "Path processing finished; inspect the reported missing comparison evidence"),
                                                   "error": None, "results": results})
        except Exception as exc:
            self.update_state(scan_id, path_review={"status": "failed", "progress": 0.0, "target_scan_id": target_scan_id,
                                                   "message": "Path review needs attention; original recordings are unchanged",
                                                   "error": f"{type(exc).__name__}: {exc}"[:2000]})
        finally:
            self._path_review_slots.release()

    def _vio_progress(self, scan_id: str, fraction: float, message: str) -> None:
        try:
            with self._lock(scan_id):
                state = self._read_state_unlocked(scan_id)
                vio = dict(state.get("vio") or {})
                if vio.get("status") not in {"queued", "running"}:
                    return
                vio.update({"status": "running", "progress": float(min(1.0, max(0.0, fraction))), "message": str(message)})
                state["vio"] = vio
                self._write_state_unlocked(scan_id, state)
        except HTTPException:
            return

    def _vio_worker(self, scan_id: str) -> None:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        output_dir = self.scan_dir(scan_id) / "vio" / run_id
        try:
            state = self.read_state(scan_id)
            capture = state.get("capture")
            prepared = state.get("prepared")
            if not isinstance(capture, dict) or not isinstance(prepared, dict):
                raise VIOError("timestamped capture or prepared views are missing")
            if self._has_motion_selection(capture):
                capture_report, capture = self._derive_motion_capture(scan_id, capture, state.get("camera_calibration_selection"))
                self.update_state(scan_id, capture=capture)
                if capture_report.get("metric_vio_allowed") is not True:
                    raise VIOError(capture["motion_profile"]["message"])
            else:
                report_path = self.scan_dir(scan_id) / str(capture.get("import_report") or "capture/capture_import.json")
                capture_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.update_state(scan_id, vio={**dict(state.get("vio") or {}), "status": "running", "message": "Starting OpenVINS"})
            result = self.vio_runner(
                self.scan_dir(scan_id) / "capture",
                output_dir,
                capture_report,
                prepared,
                self.settings.vio,
                lambda fraction, message: self._vio_progress(scan_id, fraction, message),
            )
            result["artifact"] = f"vio/{run_id}/vio_result.json"
            if (output_dir / "dense_camera_trajectory.json").is_file():
                result["dense_artifact"] = f"vio/{run_id}/dense_camera_trajectory.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "vio_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            self.update_state(
                scan_id,
                vio={
                    "schema": "noesis.phone_capture.vio_job.v1",
                    "estimator": "openvins",
                    "status": "complete",
                    "progress": 1.0,
                    "message": f"OpenVINS produced {len(result.get('poses') or [])} camera poses",
                    "error": None,
                    "results": result,
                },
            )
        except Exception as exc:
            try:
                state = self.read_state(scan_id)
                vio = dict(state.get("vio") or {})
                vio.update({
                    "status": "failed",
                    "progress": 0.0,
                    "message": "OpenVINS failed; RGB reconstruction remains available",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                self.update_state(scan_id, vio=vio)
            except HTTPException:
                return

    def _prepare_worker(self, scan_id: str) -> None:
        try:
            state = self.read_state(scan_id)
            scan_dir = self.scan_dir(scan_id)
            video_path = scan_dir / str(state["video"]["path"])
            capture = state.get("capture") or {}
            motion_selected = self._has_motion_selection(capture)
            # New native walk modes use their recorded camera configuration.
            # A global uploaded-video calibration must not leak into autofocus
            # captures; a separately selected, verified native profile may apply.
            frame_defaults = self.settings.frame
            if state.get("walk_intent"):
                frame_defaults = replace(frame_defaults, phone_camera_calibration=None, phone_camera_capture_mode="unbound")
            try:
                if motion_selected and not isinstance(capture.get("camera_calibration_selection"), dict):
                    raise ValueError("Motion profile has no verified associated camera selection")
                frame_settings, selected_calibration = self.camera_selection.frame_settings(
                    scan_dir, capture.get("camera_calibration_selection"), frame_defaults
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if not motion_selected and not state.get("walk_intent"):
                    raise
                frame_settings = replace(frame_defaults, phone_camera_calibration=None, phone_camera_capture_mode="unbound")
                selected_calibration = {"status": "not_applied", "reason_codes": ["motion_camera_profile_verification_failed"], "message": str(exc)[:2000], "accepted_for_metric_vio": False}
            if selected_calibration:
                self.update_state(scan_id, camera_calibration_selection=selected_calibration)
            if motion_selected:
                _, capture = self._derive_motion_capture(scan_id, capture, selected_calibration)
                self.update_state(scan_id, capture=capture)
            prepared = self.frame_processor(
                video_path,
                scan_dir,
                frame_settings,
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
            if supplement.get("source_capture"):
                # Do not reinterpret native added views as an uploaded-video
                # calibration mode. Their native timing and IMU files are linked
                # alongside this revision and remain independently readable.
                frame_settings = replace(frame_settings, phone_camera_calibration=None, phone_camera_capture_mode="unbound")
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

    def _paired_alignment_companion(
        self, state: dict[str, Any], camera_id: str
    ) -> dict[str, Any] | None:
        recorded = state.get("companion_capture")
        if recorded is None:
            return None
        if not isinstance(recorded, dict):
            raise HTTPException(409, "The paired static capture reference is malformed")
        session_id = str(recorded.get("session_id") or "")
        if not COMPANION_SESSION_PATTERN.fullmatch(session_id):
            raise HTTPException(409, "The paired static capture has no valid session identity")
        if recorded.get("camera_id") != camera_id:
            raise HTTPException(
                422,
                "This walk must align with its paired static camera: "
                + str(recorded.get("camera_id") or "unknown"),
            )
        try:
            companion = self.companion_capture.public_state(session_id)
        except CompanionCaptureError as exc:
            raise HTTPException(409, f"The paired static capture is unavailable: {exc}") from exc
        phone = companion.get("phone") or {}
        recorded_phone = recorded.get("phone") or {}
        if (
            companion.get("camera_id") != camera_id
            or phone.get("scan_id") != state["id"]
            or not companion.get("phone_capture_id")
            or companion.get("phone_capture_id") != recorded.get("phone_capture_id")
            or not phone.get("archive_sha256")
            or phone.get("archive_sha256") != recorded_phone.get("archive_sha256")
        ):
            raise HTTPException(409, "The paired static capture does not match this phone archive")
        if companion.get("status") != "stopped" or companion.get("error"):
            raise HTTPException(409, "Paired alignment requires a successfully finalized static capture")
        return companion

    def _saved_alignment_target(
        self, scan_id: str, alignment: dict[str, Any]
    ) -> NoesisAlignmentSettings:
        camera_id = str(alignment.get("target_camera_id") or "")
        target = self._alignment_targets.get(camera_id)
        if target is None:
            raise HTTPException(409, "The aligned camera has no configured calibration")
        if alignment.get("target_kind") != "paired_static":
            return target
        reference = alignment.get("static_reference")
        if not isinstance(reference, dict) or reference.get("status") != "complete":
            raise HTTPException(409, "The paired static reconstruction is not complete")
        scan_dir = self.scan_dir(scan_id)
        try:
            validate_paired_static_reference(scan_dir, reference)
        except Exception as exc:
            raise HTTPException(409, f"The paired static reconstruction cannot be verified: {exc}") from exc
        if reference.get("camera_id") != camera_id:
            raise HTTPException(409, "The paired static reconstruction belongs to another camera")
        return replace(
            target,
            target_revision=scan_dir / reference["target_revision"],
            calibration_path=scan_dir / reference["calibration_path"],
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
            if (state.get("path_review") or {}).get("status") in {"queued", "running"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "Wait for path refinement before changing its registration")
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
            companion = self._paired_alignment_companion(state, target.camera_id)
            state["alignment"] = {
                "status": "queued",
                "progress": 0.0,
                "message": (
                    f"Waiting to align with the {_camera_display_name(target.camera_id)} "
                    "static camera"
                ),
                "error": None,
                "target_kind": "paired_static" if companion is not None else "saved_static",
                "target_camera_id": target.camera_id,
                "target_revision_id": None if companion is not None else target.target_revision.name,
                "target_release_id": None if companion is not None else self.settings.alignment_release_id,
            }
            if companion is not None:
                state["alignment"]["static_reference"] = {
                    "status": "queued",
                    "session_id": companion["session_id"],
                    "camera_id": target.camera_id,
                    "message": "Waiting to reconstruct the paired static recording",
                    "error": None,
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
                paired = alignment.get("target_kind") == "paired_static"
                if paired:
                    companion = self._paired_alignment_companion(state, camera_id)
                    if companion is None:
                        raise RuntimeError("The paired static capture disappeared before alignment")
                    reference = dict(alignment.get("static_reference") or {})
                    reference.update(status="building", message="Reconstructing the paired static recording")
                    alignment["static_reference"] = reference
                    self.update_state(scan_id, alignment=alignment)

                    def static_progress(fraction: float, message: str) -> None:
                        reference["message"] = str(message)
                        alignment["progress"] = 0.01 + 0.34 * min(1.0, max(0.0, float(fraction)))
                        alignment["message"] = str(message)
                        self.update_state(scan_id, alignment=alignment)

                    # Static inference shares the phone/PCF GPU lane. It never
                    # enters an always-on DS9 callback or changes its producer.
                    with self._inference_lock:
                        reference = self.paired_static_runner(
                            scan_dir=scan_dir,
                            session_dir=self.companion_capture.session_root / companion["session_id"],
                            companion=companion,
                            model_settings=replace(self.settings.mapanything, anchor_image=None),
                            current_calibration_path=target.calibration_path,
                            progress=static_progress,
                        )
                    alignment["static_reference"] = reference
                    alignment["target_revision_id"] = reference.get("revision_id")
                    target = self._saved_alignment_target(scan_id, alignment)
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
                        scan_id, (0.35 + 0.65 * fraction) if paired else fraction, message
                    ),
                )
                result["target_kind"] = alignment.get("target_kind", "saved_static")
                if paired:
                    result["static_reference"] = deepcopy(alignment["static_reference"])
                if final_dir.exists():
                    raise RuntimeError("a completed alignment directory already exists")
                _finalize_alignment_paths(build_dir, final_dir, result)
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
                    reference = alignment.get("static_reference")
                    if isinstance(reference, dict) and reference.get("status") in {"queued", "building"}:
                        reference.update(
                            status="failed",
                            message="The paired static reconstruction failed; the recordings are preserved",
                            error=f"{type(exc).__name__}: {exc}",
                        )
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
            if (state.get("path_review") or {}).get("status") in {"queued", "running"}:
                raise HTTPException(status.HTTP_409_CONFLICT, "Wait for path refinement before starting PCF")
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
            target = self._saved_alignment_target(scan_id, alignment)
            if target.camera_id != camera_id:
                raise HTTPException(409, "The alignment report names a different static camera")
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
                target = self._saved_alignment_target(scan_id, state.get("alignment") or {})
                if target.camera_id != camera_id:
                    raise RuntimeError("The aligned static camera changed before PCF started")
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
            if any(job["scan_id"] == scan_id and job["status"] in {"queued", "running", "cancelling"}
                   for job in self.calibration_jobs.list()):
                raise HTTPException(409, "Cancel or finish calibration before deleting its source capture")
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
            receipt = state.get("upload_receipt")
            if isinstance(receipt, dict):
                self.capture_uploads.forget(receipt["capture_id"], receipt.get("companion_session_id"), scan_id)
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
    receipt = public.get("upload_receipt")
    if isinstance(receipt, dict):
        public["validation_status"] = receipt.get("validation_status", "pending")
    video = public.get("video")
    if isinstance(video, dict) and isinstance(video.get("path"), str):
        video["url"] = _asset_url(scan_id, video["path"])
    capture = public.get("capture")
    if isinstance(capture, dict):
        for key in (
            "import_report",
            "motion_profile_import",
            "manifest",
            "video_timestamps",
            "imu_normalized",
            "sensor_samples",
        ):
            if isinstance(capture.get(key), str):
                capture[f"{key}_url"] = _asset_url(scan_id, capture[key])
    companion = public.get("companion_capture")
    if isinstance(companion, dict):
        artifacts = companion.get("artifacts")
        if isinstance(artifacts, dict):
            companion["artifact_urls"] = {
                key: f"/companion-assets/{quote(str(companion.get('session_id') or ''), safe='')}/{quote(str(value), safe='/')}"
                for key, value in artifacts.items()
                if isinstance(value, str)
                and Path(value).is_absolute() is False
                and ".." not in Path(value).parts
            }
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
    vio = public.get("vio")
    if isinstance(vio, dict):
        results = vio.get("results")
        if isinstance(results, dict) and isinstance(results.get("artifact"), str):
            results["artifact_url"] = _asset_url(scan_id, results["artifact"])
        if isinstance(results, dict) and isinstance(results.get("dense_artifact"), str):
            results["dense_artifact_url"] = _asset_url(scan_id, results["dense_artifact"])
    path_review = public.get("path_review")
    if isinstance(path_review, dict) and isinstance(path_review.get("results"), dict):
        results = path_review["results"]
        if isinstance(results.get("artifact"), str):
            results["artifact_url"] = _asset_url(scan_id, results["artifact"])
        results["artifact_urls"] = {key: _asset_url(scan_id, value) for key, value in (results.get("artifacts") or {}).items() if isinstance(value, str)}
    alignment = public.get("alignment")
    if isinstance(alignment, dict):
        reference = alignment.get("static_reference")
        if isinstance(reference, dict) and isinstance(reference.get("artifacts"), dict):
            reference["artifact_urls"] = {
                key: _asset_url(scan_id, value)
                for key, value in reference["artifacts"].items()
                if isinstance(value, str)
                and not Path(value).is_absolute()
                and ".." not in Path(value).parts
            }
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


def _read_sensor_bundle_manifest(
    archive_path: Path,
    *,
    limits: CaptureImportLimits,
) -> dict[str, Any] | None:
    """Read only the bounded pairing manifest before allocating a scan.

    ``import_capture_bundle`` remains the authority for full archive
    validation and extraction.  This small preflight is only needed so a
    manually retried TAR can recover its companion identifiers before the
    upload transaction reserves a scan directory.
    """

    manifest_names = {"capture_manifest.json", "manifest.json"}
    maximum = min(32 * 1024 * 1024, int(limits.max_member_bytes))

    def read_bytes(handle: Any) -> bytes:
        data = handle.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("capture manifest exceeds the preflight size limit")
        return data

    def decode(data: bytes) -> dict[str, Any] | None:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    try:
        with zipfile.ZipFile(archive_path) as archive:
            file_count = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                file_count += 1
                if file_count > limits.max_files:
                    return None
                if info.filename not in manifest_names:
                    continue
                if info.file_size > maximum:
                    return None
                with archive.open(info, "r") as handle:
                    return decode(read_bytes(handle))
            return None
    except (OSError, zipfile.BadZipFile, ValueError):
        pass

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            file_count = 0
            while True:
                member = archive.next()
                if member is None:
                    break
                if member.isdir():
                    continue
                file_count += 1
                if file_count > limits.max_files:
                    return None
                if member.name not in manifest_names or not member.isfile():
                    continue
                if member.size > maximum:
                    return None
                handle = archive.extractfile(member)
                if handle is None:
                    return None
                with handle:
                    return decode(read_bytes(handle))
            return None
    except (OSError, tarfile.TarError, ValueError):
        return None


def _sensor_bundle_pairing_reference(
    manifest: dict[str, Any] | None,
) -> dict[str, str | None]:
    """Extract the optional companion reference carried by a capture TAR."""

    if not isinstance(manifest, dict):
        return {
            "session_id": None,
            "camera_id": None,
            "phone_capture_id": None,
            "capture_id": None,
        }
    nested = manifest.get("companion_capture")
    if not isinstance(nested, dict):
        nested = manifest.get("companion")
    nested = nested if isinstance(nested, dict) else {}

    def first_string(*values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    return {
        "session_id": first_string(
            nested.get("session_id"), manifest.get("companion_session_id")
        ),
        "camera_id": first_string(
            nested.get("camera_id"), manifest.get("companion_camera_id")
        ),
        "phone_capture_id": first_string(
            nested.get("phone_capture_id"),
            manifest.get("companion_phone_capture_id"),
            manifest.get("phone_capture_id"),
        ),
        "capture_id": first_string(manifest.get("capture_id")),
    }


def _current_review_assembly(
    pcf_storage_root: Path,
    *,
    alignment_release_id: str | None,
) -> tuple[dict[str, Any], Path]:
    manifest_path = pcf_storage_root / "review-assemblies" / "current.json"
    try:
        if manifest_path.stat().st_size > 256 * 1024:
            raise ValueError("manifest exceeds 256 KiB")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No whole-home PCF review assembly is published",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"The whole-home PCF review manifest is invalid: {exc}",
        ) from exc
    if not isinstance(manifest, dict):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review manifest is not an object",
        )
    if (
        manifest.get("contract") != "noesis.scene.review_assembly"
        or manifest.get("contract_version") not in {3, 4}
        or manifest.get("status") != "review_only"
        or manifest.get("accepted_for_canonical_use") is not False
        or manifest.get("coordinate_frame") != "backend_world_m"
        or manifest.get("units") != "meters"
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review manifest violates its fail-closed contract",
        )
    scene_binding = manifest.get("scene_binding")
    if not isinstance(scene_binding, dict):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review manifest has no scene binding",
        )
    if alignment_release_id and scene_binding.get("release_id") != alignment_release_id:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review is not bound to the active alignment release",
        )
    camera_anchor = manifest.get("camera_anchor")
    provenance = manifest.get("provenance")
    solved_pose = (
        camera_anchor.get("camera_to_assembly_col_major")
        if isinstance(camera_anchor, dict)
        else None
    )
    device_pose = (
        camera_anchor.get("device_reference_camera_to_assembly_col_major")
        if isinstance(camera_anchor, dict)
        else None
    )
    if (
        not isinstance(camera_anchor, dict)
        or camera_anchor.get("status") != "review_pose_estimate"
        or camera_anchor.get("accepted_for_canonical_use") is not False
        or camera_anchor.get("camera_id") != "family-room"
        or camera_anchor.get("coordinate_frame")
        != "family_accepted_backend_world_m"
        or camera_anchor.get("pose_convention")
        != "opencv_cam2world_x_right_y_down_z_forward"
        or camera_anchor.get("anchor_mode") != "floor_locked_planar"
        or camera_anchor.get("camera_height_source")
        != "admitted_scene_prior_reference_camera"
        or camera_anchor.get("vertical_anchor_translation_m") != 0.0
        or not isinstance(solved_pose, list)
        or len(solved_pose) != 16
        or not all(isinstance(value, (int, float)) for value in solved_pose)
        or not isinstance(device_pose, list)
        or len(device_pose) != 16
        or not all(isinstance(value, (int, float)) for value in device_pose)
        or not isinstance(provenance, dict)
        or camera_anchor.get("source_report_sha256")
        != provenance.get("source_camera_anchor_report_sha256")
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review has no valid static-camera anchor",
        )
    camera_markers = manifest.get("camera_markers")
    marker_values = (
        camera_markers.get("markers") if isinstance(camera_markers, dict) else None
    )
    marker_ids = {
        marker.get("camera_id")
        for marker in marker_values or []
        if isinstance(marker, dict)
    }
    family_marker = next(
        (
            marker
            for marker in marker_values or []
            if isinstance(marker, dict) and marker.get("camera_id") == "family-room"
        ),
        None,
    )
    family_position = (
        family_marker.get("position_assembly_m")
        if isinstance(family_marker, dict)
        else None
    )
    anchor_position = camera_anchor.get("camera_center_assembly_m")
    if (
        not isinstance(camera_markers, dict)
        or camera_markers.get("status") != "review_camera_positions"
        or camera_markers.get("accepted_for_canonical_use") is not False
        or camera_markers.get("coordinate_frame")
        != "family_accepted_backend_world_m"
        or camera_markers.get("color_hex") != "#ffd400"
        or not isinstance(camera_markers.get("sphere_radius_m"), (int, float))
        or not 0.05 <= float(camera_markers["sphere_radius_m"]) <= 0.5
        or not isinstance(marker_values, list)
        or len(marker_values) != 3
        or marker_ids != {"family-room", "kitchen", "living-room"}
        or not all(
            isinstance(marker, dict)
            and isinstance(marker.get("position_assembly_m"), list)
            and len(marker["position_assembly_m"]) == 3
            and all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in marker["position_assembly_m"]
            )
            for marker in marker_values
        )
        or not isinstance(anchor_position, list)
        or not isinstance(family_position, list)
        or any(
            abs(float(value) - float(anchor_position[index])) > 1e-8
            for index, value in enumerate(family_position)
        )
        or camera_markers.get("source_report_sha256")
        != provenance.get("source_camera_markers_report_sha256")
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review has no valid static-camera markers",
        )
    artifact = manifest.get("artifact")
    relative = artifact.get("relative_path") if isinstance(artifact, dict) else None
    expected_role = (
        "multiroom_surface_mesh_glb"
        if manifest.get("contract_version") == 4
        else "multiroom_points_glb"
    )
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or artifact.get("role") != expected_role
        or artifact.get("media_type") != "model/gltf-binary"
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review artifact declaration is invalid",
        )
    storage_root = pcf_storage_root.resolve()
    artifact_path = (pcf_storage_root / relative).resolve()
    if storage_root not in artifact_path.parents:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review artifact escapes PCF storage",
        )
    try:
        size = artifact_path.stat().st_size
    except FileNotFoundError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review artifact is missing",
        ) from exc
    if size != artifact.get("size_bytes") or size > 64 * 1024 * 1024:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review artifact size does not match",
        )
    if _sha256_file(artifact_path) != artifact.get("sha256"):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The whole-home PCF review artifact digest does not match",
        )
    return manifest, artifact_path


def create_app(
    settings: PhoneScanSettings | None = None,
    *,
    frame_processor: FrameProcessor = prepare_video_frames,
    inference_runner: InferenceRunner = run_adaptive_mapanything_scan,
    da3_inference_runner: DA3InferenceRunner = run_adaptive_da3_phone_scan,
    alignment_runner: AlignmentRunner = run_noesis_alignment,
    paired_static_runner: Callable[..., dict[str, Any]] = prepare_paired_static_reference,
    supplement_runner: SupplementRunner = run_supplement_integration,
    pcf_runner: PCFRunner = run_pcf_review_candidate,
    vio_runner: VioRunner = _default_vio_runner,
) -> FastAPI:
    configured = settings or PhoneScanSettings.from_env()
    phone_camera_calibration = calibration_summary(configured.frame.phone_camera_calibration)
    if phone_camera_calibration is not None:
        phone_camera_calibration["capture_mode"] = configured.frame.phone_camera_capture_mode
        phone_camera_calibration["status"] = (
            "available_for_matching_capture"
            if configured.frame.phone_camera_capture_mode != "unbound"
            else "registered_capture_mode_unbound"
        )
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
        paired_static_runner=paired_static_runner,
        supplement_runner=supplement_runner,
        pcf_runner=pcf_runner,
        vio_runner=vio_runner,
    )
    companion_upload_locks: dict[str, asyncio.Lock] = {}
    companion_upload_locks_guard = threading.Lock()

    def _companion_upload_lock(session_id: str) -> asyncio.Lock:
        with companion_upload_locks_guard:
            return companion_upload_locks.setdefault(session_id, asyncio.Lock())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.recover_interrupted_states()
        service.calibration_jobs.recover()
        yield
        service.shutdown()

    app = FastAPI(
        title="Noesis Multi-View Phone Scan",
        version="1.13.0",
        lifespan=lifespan,
    )
    app.state.phone_scan_service = service
    phone_diagnostic_lock = threading.Lock()
    imu_upload_lock = asyncio.Lock()
    capture_upload_admission_lock = asyncio.Lock()
    capture_upload_acceptances: set[asyncio.Task[Any]] = set()

    @app.get("/api/calibration/jobs")
    async def list_calibration_jobs() -> dict[str, Any]:
        return {"jobs": await asyncio.to_thread(service.calibration_jobs.list)}

    @app.get("/api/calibration/camera-selection")
    async def get_camera_selection() -> dict[str, Any]:
        return {"selection": await asyncio.to_thread(service.camera_selection.get)}

    async def _motion_json_body(request: Request) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(415, "Expected application/json")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 1024:
                        raise HTTPException(413, "Motion-profile request exceeds 1 KiB")
                    body.extend(chunk)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Expected a motion-profile request object")
            return payload
        except TimeoutError as exc:
            raise HTTPException(408, "Motion-profile request timed out") from exc
        except (ValueError, TypeError, RecursionError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/calibration/motion-profile-jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_motion_profile_job(request: Request) -> dict[str, Any]:
        payload = await _motion_json_body(request)
        fields = {"camera_calibration_id", "imu_calibration_id", "noise_calibration_id"}
        if set(payload) != fields or any(not isinstance(payload[key], str) or not payload[key].strip() or len(payload[key]) > 160 for key in fields):
            raise HTTPException(422, "Choose the three completed camera, camera–IMU and noise job IDs")
        available, reason = _motion_profile_capability(configured.vio)
        if not available:
            raise HTTPException(503, reason)
        try:
            return await asyncio.to_thread(service.calibration_jobs.submit_motion_profile, payload)
        except CalibrationJobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/calibration/motion-selection")
    async def get_motion_selection() -> dict[str, Any]:
        try:
            return {"selection": await asyncio.to_thread(service.motion_selection.get)}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, f"Motion selection could not be verified: {exc}") from exc

    @app.post("/api/calibration/motion-selection")
    async def select_motion_calibration(request: Request) -> dict[str, Any]:
        payload = await _motion_json_body(request)
        if set(payload) != {"motion_calibration_id"} or (payload["motion_calibration_id"] is not None and (not isinstance(payload["motion_calibration_id"], str) or not payload["motion_calibration_id"].strip() or len(payload["motion_calibration_id"]) > 160)):
            raise HTTPException(422, "Expected motion_calibration_id or null to clear")
        try:
            return {"selection": await asyncio.to_thread(service.motion_selection.select, payload["motion_calibration_id"])}
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/calibration/camera-selection")
    async def select_camera_calibration(request: Request) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(415, "Expected application/json")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 1024:
                        raise HTTPException(413, "Camera selection exceeds 1 KiB")
                    body.extend(chunk)
            payload = json.loads(body)
            if not isinstance(payload, dict) or set(payload) != {"camera_calibration_id"}:
                raise ValueError("Expected camera_calibration_id or null to clear")
            selection = await asyncio.to_thread(service.camera_selection.select, payload["camera_calibration_id"])
            return {"selection": selection}
        except TimeoutError as exc:
            raise HTTPException(408, "Camera selection timed out") from exc
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/calibration/imu-recordings")
    async def list_imu_recordings() -> dict[str, Any]:
        return {"recordings": await asyncio.to_thread(service.calibration_jobs.list_imu_recordings)}

    @app.post("/api/calibration/noise-jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_noise_job(request: Request) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(415, "Expected application/json")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 1024:
                        raise HTTPException(413, "Stationary noise request exceeds 1 KiB")
                    body.extend(chunk)
        except TimeoutError as exc:
            raise HTTPException(408, "Stationary noise request timed out") from exc
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Expected a stationary capture ID")
            return await asyncio.to_thread(service.calibration_jobs.submit_noise, payload.get("imu_capture_id"))
        except CalibrationJobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, TypeError, RecursionError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/calibration/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_calibration_job(request: Request) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(415, "Expected application/json")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                        raise HTTPException(413, "Calibration request exceeds 64 KiB")
                    body.extend(chunk)
        except TimeoutError as exc:
            raise HTTPException(408, "Calibration request timed out") from exc
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Expected an object")
            scan_id = payload.get("scan_id")
            if not isinstance(scan_id, str) or not SCAN_ID_PATTERN.fullmatch(scan_id):
                raise ValueError("Select an imported native RoomWalk capture")
            return await asyncio.to_thread(service.initiate_calibration, scan_id, payload)
        except CalibrationJobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "The referenced calibration was not found") from exc
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/calibration/jobs/{job_id}")
    async def get_calibration_job(job_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.calibration_jobs.get, job_id)
        except (CalibrationJobError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/calibration/jobs/{job_id}/cancel")
    async def cancel_calibration_job(job_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.calibration_jobs.cancel, job_id)
        except (CalibrationJobError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/calibration/jobs/{job_id}/files/{name:path}")
    async def calibration_artifact(job_id: str, name: str) -> FileResponse:
        try:
            path = await asyncio.to_thread(service.calibration_jobs.artifact, job_id, name)
            return FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})
        except (CalibrationJobError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/calibration/jobs/{job_id}/log")
    async def calibration_log(job_id: str) -> FileResponse:
        try:
            await asyncio.to_thread(service.calibration_jobs.get, job_id)
            path = service.calibration_jobs.root / job_id / "worker.log"
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError("The job has no processing log yet")
            return FileResponse(path, media_type="text/plain", headers={"X-Content-Type-Options": "nosniff"})
        except (CalibrationJobError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/phone-calibration/imu-bundle", status_code=status.HTTP_201_CREATED)
    async def upload_imu_calibration(request: Request) -> dict[str, Any]:
        """Retain finalized stationary sensor evidence independently of scans."""
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/zip":
            raise HTTPException(415, "Expected application/zip")
        if imu_upload_lock.locked():
            raise HTTPException(409, "An IMU calibration upload is already in progress")
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length") from exc
        if declared < 0 or declared > MAX_IMU_BUNDLE_BYTES:
            raise HTTPException(413, "IMU bundle exceeds 512 MiB")
        async with imu_upload_lock:
            directory = configured.storage_root / ".imu-calibration"
            directory.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(directory).free < 3 * MAX_IMU_BUNDLE_BYTES + 16 * 1024 * 1024:
                raise HTTPException(507, "Not enough space to retain and validate an IMU recording")
            temporary = directory / f".{uuid.uuid4().hex}.uploading"
            try:
                received = 0
                with temporary.open("wb") as handle:
                    async with asyncio.timeout(300):
                        async for chunk in request.stream():
                            received += len(chunk)
                            if received > MAX_IMU_BUNDLE_BYTES:
                                raise HTTPException(413, "IMU bundle exceeds 512 MiB")
                            await asyncio.to_thread(handle.write, chunk)
                if not received:
                    raise HTTPException(400, "The IMU bundle is empty")
                return await asyncio.to_thread(retain_imu_bundle, temporary, directory)
            except TimeoutError as exc:
                raise HTTPException(408, "IMU recording upload timed out") from exc
            except ImuCalibrationError as exc:
                raise HTTPException(422, str(exc)) from exc
            finally:
                temporary.unlink(missing_ok=True)

    @app.post("/api/phone-diagnostics", status_code=status.HTTP_201_CREATED)
    async def upload_phone_diagnostic(request: Request) -> dict[str, Any]:
        """Retain a bounded, explicit capability report; never create a scan."""
        limit = 512 * 1024
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(415, "Expected application/json")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > limit:
                        raise HTTPException(413, "Phone report exceeds 512 KiB")
                    body.extend(chunk)
        except TimeoutError as exc:
            raise HTTPException(408, "Phone report upload timed out") from exc
        try:
            def reject_constant(value: str) -> None:
                raise ValueError(f"Invalid JSON number: {value}")

            report = json.loads(body, parse_constant=reject_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise HTTPException(400, "Invalid phone report JSON") from exc
        if (
            not isinstance(report, dict)
            or report.get("schema") != "noesis.phone_capture.android_capabilities.v1"
            or not isinstance(report.get("device"), dict)
            or not isinstance(report.get("cameras"), list)
            or len(report["cameras"]) > 32
            or any(not isinstance(camera, dict) for camera in report["cameras"])
        ):
            raise HTTPException(422, "Expected an Android camera capability report")
        digest = hashlib.sha256(body).hexdigest()

        def persist() -> None:
            directory = configured.storage_root / ".phone-diagnostics"
            with phone_diagnostic_lock:
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{digest}.json"
                if target.is_file():
                    return
                if sum(1 for _ in directory.glob("*.json")) >= 128:
                    raise HTTPException(507, "Phone report storage is full; retained reports need review")
                if shutil.disk_usage(directory).free < len(body) + 16 * 1024 * 1024:
                    raise HTTPException(507, "Not enough space to retain the phone report")
                temporary = directory / f".{uuid.uuid4().hex}.uploading"
                try:
                    temporary.write_bytes(body)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)

        await asyncio.to_thread(persist)
        return {
            "schema": "noesis.phone_capture.android_diagnostic_receipt.v1",
            "id": digest, "sha256": digest, "size_bytes": len(body),
            "diagnostic_only": True,
        }

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(configured.static_root / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        usage = shutil.disk_usage(configured.storage_root)
        pcf_usage = shutil.disk_usage(service.pcf_storage_root)
        motion_available, motion_unavailable_reason = _motion_profile_capability(configured.vio)
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
            "paired_static_alignment_available": True,
            "phone_camera_calibration": phone_camera_calibration,
            "calibration": {
                "schema": "roomwalk.calibration_capabilities.v1",
                "camera_processing": True,
                "stationary_noise_processing": True,
                "camera_imu_solver_configured": bool(os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER", "").strip()),
                "motion_profile_validation_available": motion_available,
                "motion_profile_unavailable_reason": motion_unavailable_reason,
                "camera_selection_scope": "future_exactly_matching_native_imports",
                "automatic_metric_vio_admission": False,
            },
            "secure_capture_url": _phone_scan_secure_url(),
            "ca_certificate_url": "/api/browser-capture/ca-certificate",
            "sensor_capture": {
                "schema": "noesis.phone_capture.v1",
                "supported_schemas": [
                    "noesis.phone_capture.v1",
                    BROWSER_CAPTURE_SCHEMA,
                ],
                "max_archive_bytes": configured.capture_limits.max_archive_bytes,
                "max_files": configured.capture_limits.max_files,
                "max_imu_rows": configured.capture_limits.max_imu_rows,
                "vio_estimator": configured.vio.estimator,
                "vio_available": bool(
                    configured.vio.executable
                    and configured.vio.executable.is_file()
                    and configured.vio.config
                    and configured.vio.config.is_file()
                ),
            },
        }

    @app.get("/api/browser-capture/ca-certificate")
    async def browser_capture_ca_certificate() -> FileResponse:
        certificate = _phone_scan_ca_certificate_path()
        _validated_ca_certificate(certificate)
        return FileResponse(
            certificate,
            media_type="application/x-x509-ca-cert",
            filename="Noesis-Room-Walk-CA.crt",
            headers={"Cache-Control": "no-store"},
        )

    def _companion_error(exc: CompanionCaptureError) -> HTTPException:
        if isinstance(exc, CompanionCaptureBusy):
            return HTTPException(status.HTTP_409_CONFLICT, str(exc))
        if isinstance(exc, CompanionCaptureConflict):
            return HTTPException(status.HTTP_409_CONFLICT, str(exc))
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))

    async def _companion_body(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Companion request body must be a JSON object")
        return payload

    @app.get("/api/companion-captures/cameras", response_model=None)
    async def companion_capture_cameras() -> dict[str, Any]:
        try:
            rows = await asyncio.to_thread(service.companion_cameras)
            return {"available": bool(rows), "cameras": rows, "reason": None if rows else "no active camera source is available"}
        except CompanionCaptureError as exc:
            # Camera inventory is a capability probe.  Returning a bounded
            # unavailable response lets the browser explain the runtime
            # condition and retry after DS9 is healthy again.
            return {"available": False, "cameras": [], "reason": str(exc)}

    @app.get("/api/companion-captures", response_model=None)
    async def companion_capture_sessions() -> list[dict[str, Any]]:
        return await asyncio.to_thread(service.companion_capture.list_sessions)

    @app.get("/api/companion-captures/clock", response_model=None)
    async def companion_capture_clock() -> dict[str, str]:
        received_mono = time.monotonic_ns()
        received_unix = time.time_ns()
        sent_mono = time.monotonic_ns()
        sent_unix = time.time_ns()
        return {
            "server_received_unix_ns": str(received_unix),
            "server_received_monotonic_ns": str(received_mono),
            "server_sent_unix_ns": str(sent_unix),
            "server_sent_monotonic_ns": str(sent_mono),
            "synchronization_verified": "false",
        }

    @app.post("/api/companion-captures", status_code=status.HTTP_201_CREATED, response_model=None)
    async def start_companion_capture(request: Request) -> JSONResponse:
        payload = await _companion_body(request)
        camera_id = str(payload.get("camera_id") or payload.get("cameraId") or "").strip()
        request_id = request.headers.get("x-client-request-id") or payload.get("client_request_id")
        phone_capture_id = request.headers.get("x-phone-capture-id") or payload.get("phone_capture_id")
        try:
            state = await asyncio.to_thread(
                service.start_companion_capture,
                camera_id,
                client_request_id=str(request_id).strip() if request_id else None,
                phone_capture_id=str(phone_capture_id).strip() if phone_capture_id else None,
                clock_probes=payload.get("clock_probes"),
            )
        except CompanionCaptureError as exc:
            raise _companion_error(exc) from exc
        return JSONResponse(state, status_code=status.HTTP_201_CREATED)

    @app.get("/api/companion-captures/{session_id}", response_model=None)
    async def get_companion_capture(session_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.companion_status, session_id)
        except CompanionCaptureError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    @app.post("/api/companion-captures/{session_id}/heartbeat", response_model=None)
    async def heartbeat_companion_capture(session_id: str, request: Request) -> dict[str, Any]:
        payload = await _companion_body(request)
        try:
            state = await asyncio.to_thread(
                service.companion_capture.heartbeat,
                session_id,
                phone_capture_id=str(payload.get("phone_capture_id") or request.headers.get("x-phone-capture-id") or "") or None,
                client_request_id=str(payload.get("client_request_id") or request.headers.get("x-client-request-id") or "") or None,
                client_time_ms=payload.get("client_time_ms"),
                client_send_time_ms=payload.get("client_send_time_ms"),
                client_receive_time_ms=payload.get("client_receive_time_ms"),
                clock_probes=payload.get("clock_probes"),
            )
            return await asyncio.to_thread(
                service.companion_capture.public_state,
                state["session_id"],
            )
        except CompanionCaptureError as exc:
            raise _companion_error(exc) from exc

    @app.post("/api/companion-captures/{session_id}/markers", response_model=None)
    async def marker_companion_capture(session_id: str, request: Request) -> dict[str, Any]:
        payload = await _companion_body(request)
        try:
            state = await asyncio.to_thread(
                service.companion_capture.add_marker,
                session_id,
                payload,
            )
            response = await asyncio.to_thread(
                service.companion_capture.public_state,
                state["session_id"],
            )
            response["receipt_clock"] = state.get("last_marker_receipt_clock")
            return response
        except CompanionCaptureError as exc:
            raise _companion_error(exc) from exc

    @app.post("/api/companion-captures/{session_id}/stop", response_model=None)
    async def stop_companion_capture(session_id: str, request: Request) -> dict[str, Any]:
        payload = await _companion_body(request)
        try:
            state = await asyncio.to_thread(
                service.companion_capture.stop_session,
                session_id,
                reason=str(payload.get("reason") or "user"),
                phone_capture_id=str(payload.get("phone_capture_id") or request.headers.get("x-phone-capture-id") or "") or None,
                clock_probes=payload.get("clock_probes"),
                markers=payload.get("markers"),
            )
            return await asyncio.to_thread(
                service.companion_capture.public_state,
                state["session_id"],
            )
        except CompanionCaptureError as exc:
            raise _companion_error(exc) from exc

    @app.post("/api/companion-captures/{session_id}/finalize", response_model=None)
    async def finalize_companion_capture(session_id: str, request: Request) -> dict[str, Any]:
        payload = await _companion_body(request)
        try:
            return await asyncio.to_thread(
                service.companion_capture.finalize_session,
                session_id,
                reason=str(payload.get("reason") or "finalize"),
            )
        except CompanionCaptureError as exc:
            raise _companion_error(exc) from exc

    @app.get("/api/v1/scenes/current/review-assemblies/whole-home")
    async def current_whole_home_review_assembly() -> JSONResponse:
        manifest, _ = _current_review_assembly(
            service.pcf_storage_root,
            alignment_release_id=configured.alignment_release_id,
        )
        payload = deepcopy(manifest)
        payload["artifact_url"] = (
            "/api/v1/scenes/current/review-assemblies/whole-home/artifacts/"
            f"{manifest['artifact']['role']}"
        )
        response = JSONResponse(payload)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Noesis-Review-Assembly"] = str(
            manifest.get("assembly_id") or ""
        )
        return response

    @app.get(
        "/api/v1/scenes/current/review-assemblies/whole-home/"
        "artifacts/{artifact_role}"
    )
    async def current_whole_home_review_artifact(artifact_role: str) -> FileResponse:
        manifest, artifact_path = _current_review_assembly(
            service.pcf_storage_root,
            alignment_release_id=configured.alignment_release_id,
        )
        artifact = manifest["artifact"]
        if artifact_role != artifact["role"]:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "The requested whole-home review artifact is not active",
            )
        return FileResponse(
            artifact_path,
            media_type="model/gltf-binary",
            headers={
                "Cache-Control": "no-store",
                "ETag": f'"sha256-{artifact["sha256"]}"',
                "X-Noesis-Artifact-Sha256": artifact["sha256"],
                "X-Noesis-Review-Assembly": str(manifest.get("assembly_id") or ""),
            },
        )

    @app.get("/api/scans")
    async def list_scans() -> list[dict[str, Any]]:
        return [service.public_state(state) for state in service.list_states()]

    @app.get("/api/scans/{scan_id}")
    async def get_scan(scan_id: str) -> dict[str, Any]:
        return service.public_state(service.read_state(scan_id))

    @app.patch("/api/scans/{scan_id}")
    async def rename_scan(
        scan_id: str,
        name: str = Query(min_length=1, max_length=MAX_SCAN_NAME_LENGTH),
    ) -> dict[str, Any]:
        return _public_state(service.rename_scan(scan_id, name))

    @app.post("/api/scans/{scan_id}/walk-intent")
    async def set_scan_walk_intent(scan_id: str, request: Request) -> dict[str, Any]:
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 4096:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Walk intent exceeds its 4 KiB limit")
            raw.extend(chunk)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Walk intent must be JSON") from exc
        return _public_state(service.set_walk_intent(scan_id, value))

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

    @app.post("/api/scans/sensor-bundle", status_code=status.HTTP_201_CREATED)
    async def create_scan_from_sensor_bundle(
        request: Request,
        name: str = Query(default="Phone room walk with sensors", max_length=MAX_SCAN_NAME_LENGTH),
    ) -> JSONResponse:
        filename = unquote(request.headers.get("x-file-name") or "phone_capture.zip")
        filename = Path(filename).name[:160] or "phone_capture.zip"
        content_type = request.headers.get("content-type") or "application/octet-stream"
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Content-Length") from exc
            if declared > configured.capture_limits.max_archive_bytes:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Sensor bundle exceeds the upload limit")

        header_session_id = str(request.headers.get("x-companion-session") or "").strip() or None
        header_camera_id = (
            str(
                request.headers.get("x-companion-camera-id")
                or request.headers.get("x-camera-id")
                or ""
            ).strip()
            or None
        )
        header_capture_id = str(request.headers.get("x-phone-capture-id") or "").strip() or None
        if bool(header_session_id) != bool(header_capture_id):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "X-Companion-Session and X-Phone-Capture-ID must be supplied together",
            )

        if header_camera_id and not (header_session_id and header_capture_id):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "X-Companion-Camera-ID requires X-Companion-Session and X-Phone-Capture-ID",
            )

        # Always receive into a unique storage-root sibling first.  A retry
        # must be hash-checked and paired before it can touch a scan directory.
        temporary = service.settings.storage_root / f".companion-phone-{uuid.uuid4().hex}.uploading"
        received = 0
        archive_digest = hashlib.sha256()
        receive_started = time.monotonic()
        try:
            with temporary.open("wb") as handle:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > configured.capture_limits.max_archive_bytes:
                        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Sensor bundle exceeds the upload limit")
                    handle.write(chunk)
                    archive_digest.update(chunk)
            receive_elapsed_s = max(time.monotonic() - receive_started, 1e-9)
            if received <= 0:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded sensor bundle is empty")
            archive_sha256 = archive_digest.hexdigest()

            manifest = await asyncio.to_thread(
                _read_sensor_bundle_manifest,
                temporary,
                limits=configured.capture_limits,
            )
            manifest_reference = _sensor_bundle_pairing_reference(manifest)
            manifest_session_id = manifest_reference["session_id"]
            manifest_camera_id = manifest_reference["camera_id"]
            manifest_phone_capture_id = manifest_reference["phone_capture_id"]
            manifest_capture_id = manifest_reference["capture_id"]
            manifest_has_pair_fields = any(
                manifest_reference[key]
                for key in ("session_id", "camera_id", "phone_capture_id")
            )
            if manifest_has_pair_fields and not (
                manifest_session_id and (manifest_phone_capture_id or manifest_capture_id)
            ):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "The capture manifest has an incomplete companion reference",
                )

            companion_session_id = header_session_id or manifest_session_id
            companion_capture_id = header_capture_id or manifest_phone_capture_id
            companion_camera_id = header_camera_id or manifest_camera_id
            if companion_session_id and not companion_capture_id:
                companion_capture_id = manifest_capture_id
            if bool(companion_session_id) != bool(companion_capture_id):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "The capture manifest must identify both companion session and phone capture",
                )
            if header_session_id and manifest_session_id and header_session_id != manifest_session_id:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "X-Companion-Session does not match the imported companion manifest",
                )
            if header_camera_id and manifest_camera_id and header_camera_id != manifest_camera_id:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "X-Companion-Camera-ID does not match the imported companion manifest",
                )
            if header_capture_id and manifest_phone_capture_id and header_capture_id != manifest_phone_capture_id:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "X-Phone-Capture-ID does not match the imported companion manifest",
                )
            if companion_capture_id and manifest_capture_id and companion_capture_id != manifest_capture_id:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "X-Phone-Capture-ID does not match the imported capture manifest",
                )

            async def process_upload(
                queued_scan_id: str | None = None, retained_archive: Path | None = None,
            ) -> JSONResponse:
                source_archive = retained_archive or temporary
                scan_id: str | None = None
                scan_dir: Path | None = None
                scan_dir_created = False
                reservation_started = False
                bound_camera_id: str | None = None

                def cleanup_scan_dir() -> None:
                    if queued_scan_id is not None:
                        return
                    if not scan_dir_created or scan_dir is None or not scan_dir.exists():
                        return
                    try:
                        current = service.read_state(scan_id or "")
                    except HTTPException:
                        current = None
                    # Once frame preparation has been published, the scan is
                    # valid evidence even if a later request-side operation
                    # fails.  Never remove that directory from retry cleanup.
                    if isinstance(current, dict) and current.get("status") == "processing_frames":
                        return
                    shutil.rmtree(scan_dir, ignore_errors=True)

                def record_failure(message: str) -> None:
                    if queued_scan_id is not None:
                        failed = service.read_state(queued_scan_id)
                        if (failed.get("upload_import") or {}).get("phase") == "complete":
                            service.update_state(
                                queued_scan_id, status="frame_failed", progress=0.0,
                                message="Capture is validated, but frame preparation could not start",
                                error=message[:1000],
                            )
                            service.capture_uploads.sync_state(service.scan_dir(queued_scan_id))
                            return
                        service.update_state(
                            queued_scan_id, status="import_failed", progress=0.0,
                            message="Capture validation failed; the uploaded archive is preserved",
                            error=message[:1000],
                            upload_import={**(failed.get("upload_import") or {}), "phase": "failed"},
                            upload_receipt={**failed["upload_receipt"], "validation_status": "failed"},
                        )
                        service.capture_uploads.sync_state(service.scan_dir(queued_scan_id))
                        return
                    if not (
                        companion_session_id
                        and companion_capture_id
                        and reservation_started
                        and source_archive.exists()
                    ):
                        source_archive.unlink(missing_ok=True)
                        return
                    try:
                        service.companion_capture.record_upload_failure(
                            companion_session_id,
                            companion_capture_id,
                            source_archive,
                            message,
                        )
                    except CompanionCaptureError:
                        source_archive.unlink(missing_ok=True)

                try:
                    if queued_scan_id is not None:
                        scan_id = queued_scan_id
                        scan_dir = service.scan_dir(scan_id)
                        state_payload = service.read_state(scan_id)
                        bound_camera_id = state_payload["upload_receipt"]["companion_camera_id"]
                        state_payload["upload_import"]["phase"] = "running"
                        state_payload["message"] = "Validating the stored camera and IMU capture"
                        service._write_state_unlocked(scan_id, state_payload)
                    else:
                        if companion_session_id and companion_capture_id:
                            try:
                                companion_public = service.companion_capture.public_state(
                                    companion_session_id
                                )
                                bound_camera_id = str(companion_public.get("camera_id") or "").strip() or None
                                if (
                                    companion_camera_id
                                    and bound_camera_id
                                    and companion_camera_id != bound_camera_id
                                ):
                                    raise CompanionCaptureConflict(
                                        "companion camera ID conflicts with the selected session"
                                    )
                                prior = service.companion_capture.check_phone_archive(
                                    companion_session_id,
                                    companion_capture_id,
                                    archive_sha256,
                                )
                            except CompanionCaptureError as exc:
                                raise _companion_error(exc) from exc
                            if prior is not None:
                                existing_scan_id = str(
                                    (prior.get("phone") or {}).get("scan_id") or ""
                                )
                                if not existing_scan_id:
                                    raise HTTPException(
                                        status.HTTP_409_CONFLICT,
                                        "The companion archive is associated without a scan ID",
                                    )
                                try:
                                    existing = service.read_state(existing_scan_id)
                                except HTTPException as exc:
                                    raise HTTPException(
                                        status.HTTP_409_CONFLICT,
                                        "The companion archive is associated but its scan is unavailable",
                                    ) from exc
                                source_archive.unlink(missing_ok=True)
                                return JSONResponse(
                                    _public_state(existing),
                                    status_code=status.HTTP_200_OK,
                                )
                            try:
                                scan_id = service.companion_capture.reserve_phone_scan_id(
                                    companion_session_id,
                                    companion_capture_id,
                                )
                            except CompanionCaptureError as exc:
                                raise _companion_error(exc) from exc
                            reservation_started = True
                        else:
                            scan_id = (
                                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                                f"{uuid.uuid4().hex[:8]}"
                            )

                        scan_dir = service.scan_dir(scan_id)
                        if scan_dir.exists():
                            # A reserved companion scan must never be replaced by
                            # a retry or by a partially-created directory.
                            raise HTTPException(
                                status.HTTP_409_CONFLICT,
                                "A phone upload for this companion session is already in progress",
                            )
                        scan_dir.mkdir(parents=True, exist_ok=False)
                        scan_dir_created = True
                        state_payload: dict[str, Any] = {
                            "schema": "noesis.phone_scan.state.v3",
                            "id": scan_id,
                            "name": name.strip() or "Phone room walk with sensors",
                            "created_at": _utc_now(),
                            "updated_at": _utc_now(),
                            "status": "uploading",
                            "progress": 0.0,
                            "message": "Receiving timestamped camera and IMU bundle",
                            "error": None,
                            "input_mode": "sensor_bundle",
                            "capture_profile_selection": service._new_capture_profile_selection(manifest),
                        }
                        service._write_state_unlocked(scan_id, state_payload)

                    # Archive extraction and bounded ffprobe/JSON validation
                    # can be materially more expensive than a request callback.
                    report = await asyncio.to_thread(
                        import_capture_bundle,
                        source_archive,
                        scan_dir,
                        limits=(
                            replace(configured.capture_limits, video_probe_timeout_s=ASYNC_VIDEO_PROBE_TIMEOUT_S)
                            if queued_scan_id is not None else configured.capture_limits
                        ),
                    )
                    video_path = str(report["video_path"])
                    report_manifest = report["manifest"]
                    report_schema = str(report_manifest.get("schema") or report.get("schema") or "")
                    if companion_capture_id:
                        report_reference = _sensor_bundle_pairing_reference(report_manifest)
                        report_checks = (
                            ("session_id", companion_session_id, "companion session"),
                            ("camera_id", companion_camera_id or bound_camera_id, "companion camera"),
                            ("phone_capture_id", companion_capture_id, "phone capture"),
                            ("capture_id", companion_capture_id, "capture"),
                        )
                        for field, expected, label in report_checks:
                            actual = report_reference.get(field)
                            if actual and expected and actual != expected:
                                raise HTTPException(
                                    status.HTTP_409_CONFLICT,
                                    f"Imported {label} does not match the paired session",
                                )
                        if not report_reference.get("capture_id"):
                            raise HTTPException(
                                status.HTTP_409_CONFLICT,
                                "Imported capture manifest has no capture_id",
                            )
                    browser_capture = report_schema == BROWSER_CAPTURE_SCHEMA
                    if browser_capture:
                        device = dict(report_manifest.get("device") or {})
                        device_id = str(device.get("id") or "browser-session:unknown")
                        capture_state = {
                            "schema": BROWSER_CAPTURE_SCHEMA,
                            "capture_kind": "browser_camera_imu",
                            "capture_source": "browser camera + IMU",
                            "capture_id": report_manifest["capture_id"],
                            "device": device,
                            "camera_sensor_id": f"{device_id}:camera",
                            "imu_sensor_id": f"{device_id}:imu",
                            "metric_vio_allowed": False,
                            "native_vio_compatible": False,
                            "camera_acquisition_timestamp_verified": False,
                            "calibration": report["calibration"],
                            "timing": report["timing"],
                            "sensors": report["sensors"],
                            "video_frames": report["video"],
                            "interruptions": report["interruptions"],
                            "stop_reason": report["stop_reason"],
                            "coverage": report["coverage"],
                            "manifest": report["manifest_path"],
                            "sensor_samples": report["sensor_samples_path"],
                            "import_report": report["import_report_path"],
                        }
                        state_message = "Browser camera + IMU capture saved; preparing RGB views from encoded video timestamps"
                        video_content_type = str(report_manifest["video"].get("mime_type") or content_type)
                    else:
                        capture_state = {
                            "schema": "noesis.phone_capture.v1",
                            "capture_id": report_manifest["capture_id"],
                            "device": report_manifest["device"],
                            "camera_sensor_id": report_manifest["camera"]["id"],
                            "imu_sensor_id": report_manifest["imu"]["sensor_id"],
                            "metric_vio_allowed": bool(report["metric_vio_allowed"]),
                            "calibration": report["calibration"],
                            "import_report": report["import_report_path"],
                            "video_timestamps": report["video_timestamps_path"],
                            "imu_normalized": report["imu_normalized_path"],
                            "coverage": report["coverage"],
                        }
                        state_message = "Sensor bundle saved; preparing timestamped reconstruction views"
                        if isinstance(report_manifest.get("calibration_request"), dict):
                            capture_state["calibration_request"] = report_manifest["calibration_request"]
                        android_timing = report.get("android_capture")
                        if isinstance(android_timing, dict):
                            capture_state.update(
                                {
                                    "capture_kind": "android_camera_imu",
                                    "capture_source": "Android Camera2 + IMU",
                                    "camera_acquisition_timestamp_verified": android_timing.get("camera_acquisition_timestamp_verified") is True,
                                    "timing": android_timing,
                                    "manifest": report.get("manifest_path"),
                                    "raw_streams_preserved": report.get("raw_streams_preserved") is True,
                                }
                            )
                            if not capture_state["camera_acquisition_timestamp_verified"]:
                                state_message = "Android camera + IMU saved; preparing RGB views from encoded video timestamps"
                        video_content_type = content_type
                    if capture_state.get("capture_kind") == "android_camera_imu" and not capture_state.get("calibration_request"):
                        saved_selection = state_payload.get("capture_profile_selection")
                        if isinstance(saved_selection, dict):
                            capture_state.update(deepcopy(saved_selection))
                        else:
                            # A pre-existing queued import is not a new import:
                            # never attach a newly selected motion profile to it.
                            capture_state["camera_calibration_selection"] = service.camera_selection.get()
                    state_payload.update(
                        {
                            "status": "processing_frames",
                            "progress": 0.0,
                            "message": state_message,
                            "video": {
                                "path": video_path,
                                "original_name": filename,
                                "content_type": video_content_type,
                                "size_bytes": int((scan_dir / video_path).stat().st_size),
                            },
                            "capture": capture_state,
                            "bundle": {"original_name": filename, "size_bytes": received},
                        }
                    )
                    if report_manifest.get("walk_intent") is not None:
                        intent = validate_walk_intent(report_manifest["walk_intent"])
                        capture_state["walk_intent"] = intent
                        state_payload["walk_intent"] = intent
                        state_payload["walk_intent_source"] = "capture_manifest"
                        # Purpose is retained even if its referenced room is no
                        # longer available. A review checks the live binding;
                        # import must not discard an otherwise useful recording.
                        if intent["mode"] == "path_refinement":
                            state_payload["message"] = "Path walk retained; preparing views for trajectory review"
                    if capture_state.get("calibration_request"):
                        state_payload.update(status="calibration_ready", message="Calibration capture retained. Open Calibration to process the board and motion evidence.")
                    if companion_session_id and companion_capture_id:
                        try:
                            service.companion_capture.associate_phone_bundle(
                                companion_session_id,
                                capture_id=companion_capture_id,
                                archive_sha256=archive_sha256,
                                scan_id=scan_id,
                            )
                            state_payload["companion_capture"] = service.companion_capture.public_state(
                                companion_session_id
                            )
                        except CompanionCaptureError as exc:
                            raise _companion_error(exc) from exc
                    if queued_scan_id is not None:
                        state_payload["upload_import"]["phase"] = "complete"
                        state_payload["upload_receipt"]["validation_status"] = "complete"
                    service._write_state_unlocked(scan_id, state_payload)
                    if queued_scan_id is None:
                        source_archive.unlink(missing_ok=True)
                    else:
                        service.capture_uploads.sync_state(scan_dir)
                    if not capture_state.get("calibration_request"):
                        service.submit_frame_preparation(scan_id)
                    return JSONResponse(
                        _public_state(state_payload),
                        status_code=status.HTTP_201_CREATED,
                    )
                except CaptureImportError as exc:
                    record_failure(str(exc))
                    cleanup_scan_dir()
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
                except HTTPException as exc:
                    if queued_scan_id is None and exc.status_code == status.HTTP_409_CONFLICT:
                        source_archive.unlink(missing_ok=True)
                    else:
                        record_failure("HTTP upload/import failure")
                    cleanup_scan_dir()
                    raise
                except Exception:
                    record_failure("unexpected upload/import failure")
                    cleanup_scan_dir()
                    raise

            async def enqueue_native_upload() -> JSONResponse:
                try:
                    normalized_manifest = validate_capture_manifest(manifest or {})
                except CaptureImportError as exc:
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
                capture_id = normalized_manifest["capture_id"]
                async with capture_upload_admission_lock:
                    queue = service.capture_uploads
                    scan_id = queue.lookup(capture_id, companion_session_id)
                    bound_camera_id = None
                    if companion_session_id and companion_capture_id:
                        try:
                            paired = service.companion_capture.public_state(companion_session_id)
                            bound_camera_id = str(paired.get("camera_id") or "").strip() or None
                            if companion_camera_id and bound_camera_id != companion_camera_id:
                                raise CompanionCaptureConflict("companion camera ID conflicts with the selected session")
                            prior = service.companion_capture.check_phone_archive(
                                companion_session_id, companion_capture_id, archive_sha256,
                            )
                            reserved = service.companion_capture.reserve_phone_scan_id(
                                companion_session_id, companion_capture_id,
                            )
                            if scan_id and scan_id != reserved:
                                raise CompanionCaptureConflict("capture receipt conflicts with the reserved companion scan")
                            scan_id = reserved
                            if prior is not None and not (service.read_state(scan_id).get("upload_receipt")):
                                temporary.unlink(missing_ok=True)
                                return JSONResponse(_public_state(service.read_state(scan_id)), status_code=200)
                        except CompanionCaptureError as exc:
                            raise _companion_error(exc) from exc
                    if scan_id is None:
                        scan_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
                    scan_dir = service.scan_dir(scan_id)
                    existing = service.read_state(scan_id) if scan_dir.exists() else None
                    receipt = {
                        "schema": UPLOAD_RECEIPT_SCHEMA, "status": "stored",
                        "size_bytes": received, "sha256": archive_sha256,
                        "capture_id": capture_id,
                        "companion_session_id": companion_session_id,
                        "companion_camera_id": bound_camera_id,
                        "validation_status": "pending",
                        "receive_elapsed_ms": round(receive_elapsed_s * 1000, 3),
                        "receive_mbps": received * 8 / receive_elapsed_s / 1_000_000,
                    }
                    if existing is not None:
                        saved = existing.get("upload_receipt") or {}
                        identity_fields = (
                            "schema", "status", "size_bytes", "sha256", "capture_id",
                            "companion_session_id", "companion_camera_id",
                        )
                        if any(saved.get(key) != receipt[key] for key in identity_fields):
                            raise HTTPException(status.HTTP_409_CONFLICT, "Capture archive conflicts with its stored upload receipt")
                        for metric in ("receive_elapsed_ms", "receive_mbps"):
                            if metric in saved:
                                receipt[metric] = saved[metric]
                        if existing.get("status") != "import_failed":
                            temporary.unlink(missing_ok=True)
                            completed = (existing.get("upload_import") or {}).get("phase") == "complete"
                            code = (200 if companion_session_id else 201) if completed else 202
                            return JSONResponse(_public_state(existing), status_code=code)
                    try:
                        admitted = queue.reserve(scan_id)
                    except CaptureUploadBusy as exc:
                        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc), headers={"Retry-After": "30"}) from exc
                    if not admitted:
                        raise HTTPException(status.HTTP_409_CONFLICT, "Previous capture validation is finishing; retry shortly")
                    state_payload = {
                        **(existing or {}),
                        "schema": "noesis.phone_scan.state.v3", "id": scan_id,
                        "name": name.strip() or "Phone room walk with sensors",
                        "created_at": (existing or {}).get("created_at") or _utc_now(),
                        "status": "importing_capture", "progress": 0.0,
                        "message": "Upload stored; camera and IMU validation is queued",
                        "error": None, "input_mode": "sensor_bundle",
                        "upload_receipt": receipt,
                        "upload_import": {
                            "phase": "queued", "archive_path": "capture_upload.archive",
                            "video_probe_timeout_s": ASYNC_VIDEO_PROBE_TIMEOUT_S,
                        },
                        "bundle": {"original_name": filename, "size_bytes": received},
                    }
                    if existing is None:
                        state_payload["capture_profile_selection"] = service._new_capture_profile_selection(normalized_manifest)
                    if normalized_manifest.get("walk_intent") is not None:
                        state_payload["walk_intent"] = normalized_manifest["walk_intent"]
                        state_payload["walk_intent_source"] = "capture_manifest_pending_validation"
                    try:
                        # Only derived extraction is reset on an explicit exact-
                        # archive retry. The durable original archive is retained.
                        if existing is not None:
                            for derived in (scan_dir / ".capture-importing", scan_dir / "capture"):
                                if derived.exists():
                                    await asyncio.to_thread(shutil.rmtree, derived)
                        retained = await asyncio.to_thread(
                            queue.persist, temporary, scan_dir, state_payload, service._write_state_unlocked,
                        )
                        queue.submit(scan_id, lambda: asyncio.run(process_upload(scan_id, retained)))
                    except BaseException:
                        queue.release(scan_id)
                        if scan_dir.exists() and (scan_dir / "scan_state.json").exists():
                            service.update_state(
                                scan_id, status="import_failed",
                                message="Capture import could not be queued; the stored archive is preserved",
                                error="Upload the same saved bundle to retry validation.",
                            )
                            queue.sync_state(scan_dir)
                        raise
                    return JSONResponse(
                        _public_state(state_payload), status_code=202,
                        headers={"Preference-Applied": "respond-async"},
                    )

            prefer_async = any(
                token.split(";", 1)[0].strip().lower() == "respond-async"
                for token in (request.headers.get("prefer") or "").split(",")
            ) and isinstance(manifest, dict) and manifest.get("schema") == "noesis.phone_capture.v1"
            if prefer_async:
                async def accept_native_upload() -> JSONResponse:
                    try:
                        if companion_session_id and companion_capture_id:
                            async with _companion_upload_lock(companion_session_id):
                                return await enqueue_native_upload()
                        return await enqueue_native_upload()
                    finally:
                        temporary.unlink(missing_ok=True)

                # The completed request body is now owned by this acceptance
                # task. A disconnected client cannot cancel persistence or
                # dequeue an acknowledged import. Keep a strong task reference.
                acceptance = asyncio.create_task(accept_native_upload())
                capture_upload_acceptances.add(acceptance)

                def acceptance_done(task: asyncio.Task[Any]) -> None:
                    capture_upload_acceptances.discard(task)
                    if not task.cancelled():
                        task.exception()

                acceptance.add_done_callback(acceptance_done)
                return await asyncio.shield(acceptance)
            if companion_session_id and companion_capture_id:
                async with _companion_upload_lock(companion_session_id):
                    return await process_upload()
            return await process_upload()
        except HTTPException:
            temporary.unlink(missing_ok=True)
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @app.post("/api/scans/{scan_id}/supplements/from-scan", status_code=status.HTTP_201_CREATED)
    async def add_retained_video_to_scan(scan_id: str, source_scan_id: str = Query()) -> JSONResponse:
        state_payload = await asyncio.to_thread(service.add_retained_capture, scan_id, source_scan_id)
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

    @app.post("/api/scans/{scan_id}/initiate-vio", status_code=status.HTTP_202_ACCEPTED)
    async def initiate_vio(scan_id: str) -> JSONResponse:
        return JSONResponse(
            _public_state(await asyncio.to_thread(service.initiate_vio, scan_id)),
            status_code=status.HTTP_202_ACCEPTED,
        )

    @app.post("/api/scans/{scan_id}/path-review", status_code=status.HTTP_202_ACCEPTED)
    async def initiate_path_review(scan_id: str, target_scan_id: str | None = Query(default=None)) -> JSONResponse:
        return JSONResponse(_public_state(service.initiate_path_review(scan_id, target_scan_id)), status_code=status.HTTP_202_ACCEPTED)

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
    app.mount(
        "/companion-assets",
        StaticFiles(directory=service.companion_capture.session_root),
        name="phone-scan-companion-assets",
    )
    return app


app = create_app()


__all__ = ["PhoneScanService", "PhoneScanSettings", "app", "create_app"]
