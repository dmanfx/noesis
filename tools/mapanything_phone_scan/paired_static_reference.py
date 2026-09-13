"""Build an independent fixed-camera reference from a finalized companion capture.

The phone reconstruction and its poses are deliberately outside this module's
input set.  A companion session supplies a raw RTSP recording, the exact
rectification contract, and one captured calibration bundle.  Static frames are
inferred independently, fused only by per-pixel depth agreement, and mapped to
the captured target world through the explicit calibration frame edge.
"""

from __future__ import annotations

import configparser
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from scripts.build_stream_room_reconstruction import DepthSnapshot, _fuse_depth_snapshots
from scripts.build_stream_room_reconstruction import (
    DewarperFrameTransform,
    _apply_dewarper_transform,
)
from tools.mapanything_phone_scan.inference import MapAnythingScanSettings


ProgressCallback = Callable[[float, str], None]
SCHEMA = "noesis.paired_static_reference.v1"
CALIBRATION_SCHEMA = "noesis.paired_static_reference.calibration.v1"
STATIC_FRAME_COUNT = 6
STATIC_FRAME_FRACTIONS = (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)
DEPTH_AGREEMENT_M = 0.18
FUSION_MIN_OBSERVATIONS = 2
MAX_PACKET_ROWS = 120_000
MAX_JSON_BYTES = 8 * 1024 * 1024


class PairedStaticReferenceError(RuntimeError):
    """Raised when a paired static source cannot be admitted fail-closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise PairedStaticReferenceError(f"{name} contains non-finite values")
    return array


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise PairedStaticReferenceError(f"{name} exceeds its size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedStaticReferenceError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PairedStaticReferenceError(f"{name} must be a JSON object")
    return value


def _rigid(values: Any, *, name: str) -> np.ndarray:
    matrix = _as_float_array(values, name=name)
    if matrix.size != 16:
        raise PairedStaticReferenceError(f"{name} must have sixteen values")
    matrix = matrix.reshape((4, 4), order="F")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or abs(float(np.linalg.det(rotation)) - 1) > 1e-5
    ):
        raise PairedStaticReferenceError(f"{name} is not a proper rigid transform")
    return matrix


def _resolve_child(root: Path, raw: Any, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise PairedStaticReferenceError(f"{name} is missing")
    path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PairedStaticReferenceError(f"{name} escapes its session directory") from exc
    return path


def _companion_value(companion: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = companion
        for part in path.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value is not None:
            return value
    return None


def _session_context(
    scan_dir: Path,
    session_dir: Path,
    companion: Mapping[str, Any],
) -> dict[str, Any]:
    session_dir = session_dir.resolve()
    session = _read_json(session_dir / "session.json", name="companion session")
    if session.get("schema") != "noesis.companion_capture.session.v1":
        raise PairedStaticReferenceError("companion session schema is unsupported")
    if session.get("status") != "complete" or session.get("finalized_at") is None:
        raise PairedStaticReferenceError("companion session is not finalized")
    session_id = str(session.get("session_id") or "").strip()
    camera = session.get("camera") if isinstance(session.get("camera"), Mapping) else {}
    camera_id = str(camera.get("camera_id") or session.get("camera_id") or "").strip()
    if not session_id or not camera_id or session_dir.name != session_id:
        raise PairedStaticReferenceError("companion session lacks session_id or camera_id")
    supplied_session = _companion_value(companion, "session_id", "id")
    if supplied_session is not None and str(supplied_session) != session_id:
        raise PairedStaticReferenceError("companion session ID does not match scan state")
    supplied_camera = _companion_value(companion, "camera_id", "camera.camera_id")
    if supplied_camera is not None and str(supplied_camera) != camera_id:
        raise PairedStaticReferenceError("companion camera ID does not match scan state")

    scan_state = _read_json(scan_dir / "scan_state.json", name="scan state")
    scan_id = str(scan_state.get("id") or scan_dir.name)
    phone_capture_id = str(
        session.get("phone_capture_id")
        or (session.get("phone") or {}).get("capture_id")
        or ""
    ).strip()
    phone = session.get("phone") if isinstance(session.get("phone"), Mapping) else {}
    if not phone_capture_id or str(phone.get("scan_id") or "") != scan_id:
        raise PairedStaticReferenceError("companion session is not associated with this phone scan")
    supplied_phone = _companion_value(companion, "phone_capture_id", "phone.capture_id")
    if supplied_phone is not None and str(supplied_phone) != phone_capture_id:
        raise PairedStaticReferenceError("phone capture ID does not match scan state")
    if not phone.get("archive_sha256") or phone.get("archive_sha256") != _companion_value(companion, "phone.archive_sha256"):
        raise PairedStaticReferenceError("phone archive hash does not match the companion association")

    artifacts = session.get("artifacts") if isinstance(session.get("artifacts"), Mapping) else {}
    raw_video = _resolve_child(session_dir, artifacts.get("static_video"), name="static video")
    packet_path = _resolve_child(session_dir, artifacts.get("packet_timing"), name="packet timing")
    if not raw_video.is_file() or not packet_path.is_file():
        raise PairedStaticReferenceError("finalized companion video or packet timing is missing")
    recorder = session.get("recorder") if isinstance(session.get("recorder"), Mapping) else {}
    if recorder.get("partial") is True or recorder.get("error"):
        raise PairedStaticReferenceError("companion recorder is partial or failed")
    if recorder.get("encoded_ready") is not True or recorder.get("status") not in {"stopped", "complete"}:
        raise PairedStaticReferenceError("companion recorder did not complete cleanly")
    packet_rows: list[dict[str, Any]] = []
    with packet_path.open(encoding="utf-8") as handle:
        for index in range(MAX_PACKET_ROWS + 1):
            line = handle.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                raise PairedStaticReferenceError("packet timing row exceeds its size limit")
            if index >= MAX_PACKET_ROWS:
                raise PairedStaticReferenceError("packet timing exceeds its bounded limit")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PairedStaticReferenceError("packet timing contains invalid JSON") from exc
            if not isinstance(row, dict):
                raise PairedStaticReferenceError("packet timing row is not an object")
            packet_rows.append(row)
    if len(packet_rows) < STATIC_FRAME_COUNT:
        raise PairedStaticReferenceError("companion video has fewer than six packet records")
    pts = []
    for row in packet_rows:
        value = row.get("pts_ns")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PairedStaticReferenceError("packet timing has an invalid PTS")
        pts.append(int(value))
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise PairedStaticReferenceError("packet PTS are not strictly increasing")
    packet_count = recorder.get("packet_count")
    if packet_count is not None and int(packet_count) != len(packet_rows):
        raise PairedStaticReferenceError("recorder packet count does not match packet timing")
    return {
        "session": session,
        "session_id": session_id,
        "camera_id": camera_id,
        "scan_id": scan_id,
        "phone_capture_id": phone_capture_id,
        "video_path": raw_video,
        "packet_path": packet_path,
        "packet_rows": packet_rows,
        "video_sha256": _sha256(raw_video),
        "packet_sha256": _sha256(packet_path),
        "phone_archive_sha256": phone["archive_sha256"],
    }


def _calibration_context(session: Mapping[str, Any], current_path: Path, camera_id: str) -> dict[str, Any]:
    current_path = current_path.resolve()
    current = _read_json(current_path, name="current camera calibration")
    current_rows = current.get("cameras", current)
    if not isinstance(current_rows, Mapping) or not isinstance(current_rows.get(camera_id), Mapping):
        raise PairedStaticReferenceError(f"current calibration has no {camera_id} row")
    current_e = current_rows[camera_id].get("E")

    # The companion observer stores the one calibration bundle as the first
    # tracking envelope.  Do not use any tracking/world observation as geometry.
    tracking_path = _resolve_child(
        Path(str(session.get("_session_dir") or ".")).resolve(),
        (session.get("artifacts") or {}).get("tracking"),
        name="tracking artifact",
    )
    bundle: dict[str, Any] | None = None
    with tracking_path.open(encoding="utf-8") as handle:
        first = handle.readline(MAX_JSON_BYTES + 1)
    if len(first) > MAX_JSON_BYTES:
        raise PairedStaticReferenceError("calibration envelope exceeds its size limit")
    try:
        envelope = json.loads(first)
    except json.JSONDecodeError as exc:
        raise PairedStaticReferenceError("tracking artifact has no calibration bundle") from exc
    if isinstance(envelope, dict) and envelope.get("message_type") == "calibration-bundle":
        bundle = envelope.get("message", {}).get("data")
    if not isinstance(bundle, dict):
        raise PairedStaticReferenceError("tracking artifact does not start with a calibration bundle")
    cameras = bundle.get("cameras") if isinstance(bundle.get("cameras"), Mapping) else {}
    e_table = cameras.get("E") if isinstance(cameras.get("E"), Mapping) else {}
    k_table = cameras.get("K") if isinstance(cameras.get("K"), Mapping) else {}
    binding_table = cameras.get("frame_bindings") if isinstance(cameras.get("frame_bindings"), Mapping) else {}
    e = e_table.get(camera_id)
    k = k_table.get(camera_id)
    binding = binding_table.get(camera_id)
    if not isinstance(e, list) or not isinstance(k, list) or not isinstance(binding, Mapping):
        raise PairedStaticReferenceError("captured calibration bundle lacks E, K, or frame binding")
    if not isinstance(current_e, list) or not np.array_equal(np.asarray(current_e), np.asarray(e)):
        raise PairedStaticReferenceError("captured E does not match current canonical calibration")
    captured_hash = str(binding.get("camera_calibration_sha256") or "")
    current_hash = _sha256(current_path)
    if captured_hash and captured_hash != current_hash:
        raise PairedStaticReferenceError("captured calibration hash does not match current calibration")
    meta = bundle.get("meta") if isinstance(bundle.get("meta"), Mapping) else {}
    if meta.get("cameras_E_semantics") != "camera_from_calibration_frame_raw":
        raise PairedStaticReferenceError("captured E semantics are not raw camera-from-calibration")
    target = binding.get("target_from_calibration_col_major")
    source_floor = binding.get("calibration_floor_plane")
    world_floor = binding.get("world_floor_plane")
    world_frame = binding.get("world_frame")
    if not isinstance(target, list) or len(target) != 16 or not isinstance(source_floor, Mapping) or not isinstance(world_floor, Mapping) or not isinstance(world_frame, Mapping):
        raise PairedStaticReferenceError("captured frame binding is incomplete")
    if world_frame.get("frame_id") != "backend_world_m" or not world_frame.get("revision"):
        raise PairedStaticReferenceError("captured target world frame identity is missing or unsupported")
    target_t = _rigid(target, name="target_from_calibration")
    e_matrix = _rigid(e, name="captured E")
    # Verify the captured edge maps its declared source floor to its declared
    # target floor.  This guards against applying a second guessed leveling.
    source_n = _as_float_array(source_floor.get("normal"), name="source floor normal").reshape(3)
    source_d = float(source_floor.get("offset_m"))
    mapped_n = target_t[:3, :3] @ source_n
    mapped_d = source_d - float(mapped_n @ target_t[:3, 3])
    target_n = _as_float_array(world_floor.get("normal"), name="target floor normal").reshape(3)
    target_d = float(world_floor.get("offset_m"))
    if not np.isfinite([source_d, target_d]).all() or not np.allclose(target_n, [0, 1, 0], atol=1e-6) or abs(target_d) > 1e-6:
        raise PairedStaticReferenceError("captured target floor is not the canonical Y-up zero floor")
    if np.linalg.norm(mapped_n - target_n) > 1e-6 or abs(mapped_d - target_d) > 1e-6:
        raise PairedStaticReferenceError("captured frame edge does not preserve its declared floor binding")
    raw_k = _as_float_array(k, name="captured K")
    if raw_k.shape != (4,) or raw_k[0] <= 0 or raw_k[1] <= 0:
        raise PairedStaticReferenceError("captured K must be [fx, fy, cx, cy]")
    fx, fy, cx, cy = raw_k
    matrix_k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return {
        "bundle": bundle,
        "E": e_matrix,
        "E_col_major": [float(x) for x in e],
        "K": matrix_k,
        "K_flat": [float(x) for x in k],
        "binding": dict(binding),
        "target_from_calibration": target_t,
        "target_from_calibration_col_major": [float(x) for x in target],
        "current_calibration_sha256": current_hash,
        "bundle_sha256": _canonical_sha256(bundle),
        "binding_sha256": _canonical_sha256(binding),
    }


def _dewarper_context(session: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    camera = session.get("camera") if isinstance(session.get("camera"), Mapping) else {}
    authority = camera.get("authority") if isinstance(camera.get("authority"), Mapping) else {}
    canonical = ((authority.get("dewarper") or {}).get("canonical_tracker_pixels") or {})
    config_text = canonical.get("dewarper_config_text")
    config_hash = canonical.get("dewarper_config_sha256")
    if not isinstance(config_text, str) or len(config_text.encode()) > 256 * 1024:
        raise PairedStaticReferenceError("captured dewarper text is missing or too large")
    if hashlib.sha256(config_text.encode()).hexdigest() != config_hash:
        raise PairedStaticReferenceError("captured dewarper text hash does not match")
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    parser.read_string(config_text)
    if "property" not in parser or "surface0" not in parser:
        raise PairedStaticReferenceError("captured dewarper has no single surface")
    props, surface = parser["property"], parser["surface0"]
    if not canonical.get("dewarper_enabled") or surface.get("projection-type") != "4":
        raise PairedStaticReferenceError("captured dewarper is not fisheye-perspective")
    if surface.get("surface-index", "0") != "0" or props.get("num-batch-buffers", "1") != "1":
        raise PairedStaticReferenceError("captured multi-surface dewarping is unsupported")
    def vector(key: str, count: int) -> np.ndarray:
        values = _as_float_array([float(x) for x in surface.get(key, "").split(";") if x.strip()], name=key)
        if values.shape != (count,):
            raise PairedStaticReferenceError(f"captured dewarper {key} has the wrong length")
        return values
    src_fx, src_fy = vector("focal-length", 2)
    dst_fx, dst_fy = vector("dst-focal-length", 2)
    dst_cx, dst_cy = vector("dst-principal-point", 2)
    source_k = np.asarray([[src_fx, 0, float(surface["src-x0"])], [0, src_fy, float(surface["src-y0"])], [0, 0, 1]], dtype=np.float64)
    output_k = np.asarray([[dst_fx, 0, dst_cx], [0, dst_fy, dst_cy], [0, 0, 1]], dtype=np.float64)
    output_size = (int(props["output-width"]), int(props["output-height"]))
    if not np.isfinite(source_k).all() or min(src_fx, src_fy, dst_fx, dst_fy) <= 0 or not all(1 <= n <= 8192 for n in output_size):
        raise PairedStaticReferenceError("captured dewarper calibration is invalid")
    if not np.allclose(output_k, calibration["K"], rtol=0, atol=1e-9):
        raise PairedStaticReferenceError("captured dewarper K does not match calibration bundle K")
    transform = DewarperFrameTransform(
        config_path=Path("captured_dewarper.ini"), source_intrinsics=source_k,
        distortion=vector("distortion", 4).reshape(4, 1),
        rectified_intrinsics=output_k, output_size=output_size,
    )
    return {
        "text": config_text, "sha256": config_hash, "transform": transform,
        "source_intrinsics": source_k.tolist(), "distortion": transform.distortion.reshape(-1).tolist(),
        "rectified_intrinsics": output_k.tolist(), "output_size": list(output_size),
    }


def _probe_video_pts(video_path: Path) -> list[float]:
    """Read bounded decoded-media PTS independently of RTSP packet ordinals."""
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals",
               f"%+#{MAX_PACKET_ROWS + 1}", "-show_entries", "frame=best_effort_timestamp_time",
               "-of", "csv=p=0", str(video_path)]
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        try:
            completed = subprocess.run(command, stdout=output, stderr=error, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PairedStaticReferenceError("static video timestamp probing failed") from exc
        if completed.returncode:
            raise PairedStaticReferenceError("static video timestamps could not be decoded")
        output.seek(0)
        values = []
        for line in output:
            token = line.strip().split(b",", 1)[0]
            if not token:
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise PairedStaticReferenceError("static decoded frame has no valid timestamp") from exc
            if not math.isfinite(value) or (values and value <= values[-1]):
                raise PairedStaticReferenceError("static decoded-media timestamps are not increasing")
            values.append(value)
            if len(values) > MAX_PACKET_ROWS:
                raise PairedStaticReferenceError("static video exceeds the frame limit")
    if len(values) < STATIC_FRAME_COUNT or values[-1] - values[0] > 910:
        raise PairedStaticReferenceError("static video has insufficient frames or exceeds the duration limit")
    return values


def _extract_selected_frames(video_path: Path, indices: list[int], output_dir: Path) -> list[Path]:
    expression = "+".join(f"eq(n,{index})" for index in indices)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-vf",
               "select='" + expression + "'", "-vsync", "0", "-frames:v", str(len(indices)),
               "-start_number", "0", str(output_dir / "source_%04d.png")]
    with tempfile.TemporaryFile() as error:
        try:
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=error, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PairedStaticReferenceError("static frame extraction failed") from exc
        if completed.returncode:
            raise PairedStaticReferenceError("static selected frames could not be decoded")
    paths = [output_dir / f"source_{index:04d}.png" for index in range(len(indices))]
    if not all(path.is_file() for path in paths):
        raise PairedStaticReferenceError("static frame extraction returned an incomplete set")
    return paths


def _explicit_selection(
    path: Path | None, scan: Mapping[str, Any], video_pts: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[list[int], dict[str, Any], list[np.ndarray] | None, int | None]:
    """Resolve recorded frames and hashed rectified exclusions before inference.

    Masks are nonzero only on excluded pixels. They are never inpainted or
    inferred from phone geometry. An absent mask is an explicit clear-frame
    assertion by the selection manifest, not automatic person detection.
    """
    if path is None:
        selected = [int(np.argmin(np.abs(video_pts - (video_pts[0] + f * (video_pts[-1] - video_pts[0]))))) for f in STATIC_FRAME_FRACTIONS]
        return selected, {"frame_count": STATIC_FRAME_COUNT, "fractions": list(STATIC_FRAME_FRACTIONS), "timestamp_domain": "decoded_recording_pts"}, None, None
    path = Path(path).resolve()
    payload = _read_json(path, name="static frame selection")
    if (payload.get("schema") != "noesis.paired_static_reference.selection.v1"
            or payload.get("camera_id") != scan["camera_id"]
            or payload.get("source_video_sha256") != scan["video_sha256"]
            or payload.get("mask_coordinate_space") != "post_dewarper_streammux_pixels"):
        raise PairedStaticReferenceError("static selection video, camera, or mask coordinate binding is invalid")
    rows = payload.get("frames")
    if not isinstance(rows, list) or len(rows) != STATIC_FRAME_COUNT:
        raise PairedStaticReferenceError("explicit static selection requires six observations")
    selected, masks = [], []
    for row in rows:
        index = row.get("source_frame_index") if isinstance(row, Mapping) else None
        if type(index) is not int or not 0 <= index < len(video_pts):
            raise PairedStaticReferenceError("explicit static source frame index is invalid")
        expected_ns = row.get("encoded_pts_ns")
        if type(expected_ns) is not int or abs(expected_ns - round(float(video_pts[index]) * 1e9)) > 2:
            raise PairedStaticReferenceError("explicit static frame PTS does not match decoded recording")
        selected.append(index)
        if row.get("exclusion_mask"):
            mask_path = _resolve_child(path.parent, row["exclusion_mask"], name="static exclusion mask")
            if not mask_path.is_file() or _sha256(mask_path) != row.get("exclusion_mask_sha256"):
                raise PairedStaticReferenceError("static exclusion mask hash mismatch")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (output_size[1], output_size[0]):
                raise PairedStaticReferenceError("static exclusion mask dimensions do not match rectified pixels")
            masks.append(mask > 0)
        elif row.get("visually_reviewed_clear") is True:
            masks.append(np.zeros((output_size[1], output_size[0]), dtype=bool))
        else:
            raise PairedStaticReferenceError("explicit static observation needs an exclusion mask or clear-frame review")
    if selected != sorted(set(selected)):
        raise PairedStaticReferenceError("explicit static frame indices must be distinct and increasing")
    keyframe = payload.get("keyframe_sample_index")
    if type(keyframe) is not int or not 0 <= keyframe < STATIC_FRAME_COUNT:
        raise PairedStaticReferenceError("explicit static keyframe sample index is invalid")
    return selected, {"frame_count": STATIC_FRAME_COUNT, "policy": "explicit_recorded_frames_with_foreground_exclusions",
                      "timestamp_domain": "decoded_recording_pts", "manifest_sha256": _sha256(path),
                      "selection": payload}, masks, keyframe


def _dewarp_valid_mask(raw: np.ndarray, transform: Any) -> tuple[np.ndarray, np.ndarray]:
    image_h, image_w = raw.shape[:2]
    source_k = np.asarray(transform.source_intrinsics, dtype=np.float64).copy()
    authored_w, authored_h = transform.output_size
    if (image_w, image_h) != (authored_w, authored_h):
        source_k[0, 0] *= image_w / authored_w
        source_k[0, 2] *= image_w / authored_w
        source_k[1, 1] *= image_h / authored_h
        source_k[1, 2] *= image_h / authored_h
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        source_k,
        np.asarray(transform.distortion, dtype=np.float64).reshape((-1, 1)),
        np.eye(3, dtype=np.float64),
        np.asarray(transform.rectified_intrinsics, dtype=np.float64),
        tuple(int(v) for v in transform.output_size),
        cv2.CV_32FC1,
    )
    valid = cv2.remap(
        np.ones((image_h, image_w), dtype=np.uint8),
        map1,
        map2,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return np.asarray(_apply_dewarper_transform(raw, transform), dtype=np.uint8), valid


def _array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim >= 1 and value.shape[0] == 1:
        value = value[0]
    return value


def _run_static_inference(model: Any, image_rgb: np.ndarray, intrinsics: np.ndarray, settings: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Infer one frame after preprocessing it with the captured rectified K."""
    import torch
    from mapanything.utils.image import preprocess_inputs

    # MapAnything's known-calibration ray assignment requires float32 inputs.
    # Keep the captured double-precision K in provenance and convert only here.
    processed = preprocess_inputs([
        {"img": image_rgb, "intrinsics": np.asarray(intrinsics, dtype=np.float32)}
    ])[0]
    with torch.inference_mode():
        predictions = model.infer(
            [processed],
            memory_efficient_inference=True,
            use_amp=str(getattr(settings, "device", "cpu")).startswith("cuda"),
            amp_dtype=str(getattr(settings, "amp_dtype", "bf16")),
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
            use_multiview_confidence=False,
        )
    if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], Mapping):
        raise PairedStaticReferenceError("MapAnything returned an invalid static prediction")
    prediction = predictions[0]
    depth = np.squeeze(_array(prediction.get("depth_z"), name="depth_z")).astype(np.float32)
    radial_depth = np.squeeze(_array(prediction.get("depth_along_ray"), name="depth_along_ray")).astype(np.float32)
    confidence = np.squeeze(_array(prediction.get("conf"), name="conf")).astype(np.float32)
    mask = np.squeeze(_array(prediction.get("mask"), name="mask")).astype(bool)
    scaled_k = _array(processed["intrinsics"], name="processed intrinsics").astype(np.float64)
    returned_k = _array(prediction.get("intrinsics"), name="predicted intrinsics").astype(np.float64)
    rgb = _array(prediction.get("img_no_norm"), name="model RGB")
    if depth.ndim != 2 or radial_depth.shape != depth.shape or confidence.shape != depth.shape or mask.shape != depth.shape or scaled_k.shape != (3, 3):
        raise PairedStaticReferenceError("static prediction shapes are inconsistent")
    if returned_k.shape != (3, 3) or not np.isfinite(returned_k).all() or min(returned_k[0, 0], returned_k[1, 1]) <= 0:
        raise PairedStaticReferenceError("static inference returned invalid diagnostic intrinsics")
    if rgb.shape != (*depth.shape, 3) or not np.isfinite(rgb).all():
        raise PairedStaticReferenceError("static inference returned invalid model RGB")
    rgb = np.clip(rgb * 255 if rgb.max() <= 1.5 else rgb, 0, 255).astype(np.uint8)
    # infer() conditions on K but predicts its own rays and recovered K. Its
    # metric depth_along_ray is range, whereas depth_z uses those learned rays.
    # Place that range on the captured calibrated rays and convert to Z for the
    # existing static fusion contract. Learned K remains diagnostic only.
    ys, xs = np.indices(depth.shape, dtype=np.float64)
    ray_norm = np.sqrt(1 + ((xs - scaled_k[0, 2]) / scaled_k[0, 0]) ** 2
                       + ((ys - scaled_k[1, 2]) / scaled_k[1, 1]) ** 2)
    calibrated_depth_z = (radial_depth / ray_norm).astype(np.float32)
    diagnostics = {"model_intrinsics": returned_k, "model_depth_z": depth,
                   "model_depth_along_ray": radial_depth}
    return calibrated_depth_z, confidence, mask, scaled_k, rgb, diagnostics


def _load_model(settings: Any) -> Any:
    try:
        import torch
        from mapanything.models import MapAnything
        if str(getattr(settings, "device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            raise PairedStaticReferenceError("requested static model CUDA device is unavailable")
        model = MapAnything.from_pretrained(
            str(getattr(settings, "model_id")),
            local_files_only=bool(getattr(settings, "local_files_only", True)),
        ).to(torch.device(str(getattr(settings, "device"))))
        model.eval()
        return model
    except PairedStaticReferenceError:
        raise
    except Exception as exc:
        raise PairedStaticReferenceError(f"static MapAnything model could not load: {type(exc).__name__}: {exc}") from exc


def _model_identity(settings: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(settings, "model_id", "")),
        "device": str(getattr(settings, "device", "")),
        "amp_dtype": str(getattr(settings, "amp_dtype", "")),
        "point_budget": int(getattr(settings, "point_budget", 0)),
        "local_files_only": bool(getattr(settings, "local_files_only", True)),
        "inference": "independent_single_frame_calibrated_rays_metric_range_v1",
        "mapanything_version": version("mapanything"),
        "opencv_version": cv2.__version__,
    }


def _depth_preview(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise PairedStaticReferenceError("fused static depth has no valid pixels")
    low, high = np.percentile(depth[valid], [2.0, 98.0])
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    preview = cv2.applyColorMap(np.where(valid, np.round(normalized * 255.0), 0).astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid] = (18, 18, 18)
    return preview


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_rows(root: Path, names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise PairedStaticReferenceError(f"static reference artifact is missing: {name}")
        rows.append({"path": name, "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)})
    return rows


def _validate_reference_files(revision_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise PairedStaticReferenceError("paired static manifest is not complete")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("revision_id") != "paired_static_reference_" + _canonical_sha256(identity)[:16]:
        raise PairedStaticReferenceError("paired static content identity is inconsistent")
    files = manifest.get("files")
    if not isinstance(files, list) or not 8 <= len(files) <= 32:
        raise PairedStaticReferenceError("paired static manifest has an invalid artifact set")
    names = set()
    for row in files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise PairedStaticReferenceError("paired static artifact hash row is malformed")
        if row["path"] in names:
            raise PairedStaticReferenceError("paired static artifact is duplicated")
        names.add(row["path"])
        path = _resolve_child(revision_dir, row["path"], name="paired static artifact")
        if not path.is_file() or path.stat().st_size != row.get("size_bytes") or _sha256(path) != row["sha256"]:
            raise PairedStaticReferenceError(f"paired static artifact hash mismatch: {row['path']}")
    required = {"room_points.npz", "room_points_meta.json", "calibration.json", "fused_depth.npz", "keyframe.png", "depth_preview.png", "captured_dewarper.ini", "calibration_bundle.json"}
    required.update(f"static_{i:04d}.npz" for i in range(STATIC_FRAME_COUNT))
    if not required.issubset(names):
        raise PairedStaticReferenceError("paired static reference is missing a required consumer artifact")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("manifest") != "manifest.json" or any(value != "manifest.json" and value not in names for value in artifacts.values()):
        raise PairedStaticReferenceError("paired static artifact links do not match the hashed files")
    camera_id = manifest.get("camera_id")
    metadata = _read_json(revision_dir / "room_points_meta.json", name="static room metadata")
    calibration = _read_json(revision_dir / "calibration.json", name="static calibration")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise PairedStaticReferenceError("paired static calibration schema is unsupported")
    if metadata.get("camera") != camera_id or metadata.get("revision_id") != manifest.get("revision_id") or metadata.get("world_frame_revision") != manifest.get("world_frame_revision"):
        raise PairedStaticReferenceError("paired static camera or world revision is inconsistent")
    if metadata.get("coordinate_frame") != "backend_world_m_stream_points" or metadata.get("rgb_keyframes") != {camera_id: "keyframe.png"}:
        raise PairedStaticReferenceError("paired static metadata is not compatible with alignment")
    _rigid((calibration.get("cameras") or {}).get(camera_id, {}).get("E"), name="saved static E")
    _rigid((metadata.get("floor_alignment") or {}).get("world_correction_col_major"), name="saved frame binding")
    k = _as_float_array(metadata.get("intrinsics"), name="saved static K")
    if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise PairedStaticReferenceError("saved static K is invalid")
    if metadata.get("camera_frame_binding") != manifest.get("camera_frame_binding") or calibration.get("frame_binding") != manifest.get("camera_frame_binding"):
        raise PairedStaticReferenceError("paired static frame bindings disagree")
    return calibration


def validate_paired_static_reference(scan_dir: Path, reference_dict: Mapping[str, Any]) -> dict[str, Any]:
    scan_dir = Path(scan_dir).resolve()
    revision_dir = _resolve_child(scan_dir, reference_dict.get("target_revision"), name="paired static target revision")
    manifest_path = revision_dir / "manifest.json"
    expected_hash = reference_dict.get("manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or _sha256(manifest_path) != expected_hash:
        raise PairedStaticReferenceError("paired static manifest hash mismatch")
    manifest = _read_json(manifest_path, name="paired static manifest")
    if manifest.get("revision_id") != revision_dir.name:
        raise PairedStaticReferenceError("paired static revision identity is inconsistent")
    for key in ("camera_id", "session_id", "revision_id"):
        if reference_dict.get(key) != manifest.get(key):
            raise PairedStaticReferenceError(f"paired static reference {key} mismatch")
    expected_calibration = (revision_dir / "calibration.json").relative_to(scan_dir).as_posix()
    if reference_dict.get("calibration_path") != expected_calibration:
        raise PairedStaticReferenceError("paired static reference calibration path mismatch")
    calibration = _validate_reference_files(revision_dir, manifest)
    return {"manifest": manifest, "revision_dir": str(revision_dir), "calibration": calibration}


def _cached_result(scan_dir: Path, revision_dir: Path, manifest_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = revision_dir / "manifest.json"
    manifest = _read_json(manifest_path, name="paired static manifest")
    artifacts = dict(manifest.get("artifacts") or {})
    result = {
        "status": "complete",
        "session_id": manifest.get("session_id"),
        "camera_id": manifest.get("camera_id"),
        "revision_id": revision_dir.name,
        "target_revision": str(revision_dir.relative_to(scan_dir)),
        "calibration_path": str((revision_dir / str((manifest.get("calibration") or {}).get("path"))).relative_to(scan_dir)),
        "artifacts": {key: str((revision_dir / str(value)).relative_to(scan_dir)) for key, value in artifacts.items()},
        "manifest_sha256": manifest_sha256 or _sha256(manifest_path),
    }
    return result


def prepare_paired_static_reference(
    scan_dir: Path,
    session_dir: Path,
    companion: Mapping[str, Any],
    model_settings: MapAnythingScanSettings,
    current_calibration_path: Path,
    progress: ProgressCallback,
    *,
    selection_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build a same-camera reference without consulting any phone geometry."""
    scan_dir, session_dir = Path(scan_dir).resolve(), Path(session_dir).resolve()
    progress(0.01, "Verifying the paired recording and captured camera calibration")
    scan = _session_context(scan_dir, session_dir, companion)
    scan["session"]["_session_dir"] = str(session_dir)
    calibration = _calibration_context(scan["session"], Path(current_calibration_path), scan["camera_id"])
    dewarper = _dewarper_context(scan["session"], calibration)
    video_pts = np.asarray(_probe_video_pts(scan["video_path"]), dtype=np.float64)
    selected, sampling, exclusion_masks, keyframe_sample = _explicit_selection(
        selection_manifest, scan, video_pts, tuple(dewarper["output_size"]),
    )
    model_identity = _model_identity(model_settings)
    identity = {
        "schema": SCHEMA, "builder_version": 3,
        "session_id": scan["session_id"], "camera_id": scan["camera_id"],
        "scan_id": scan["scan_id"], "phone_capture_id": scan["phone_capture_id"],
        "phone_archive_sha256": scan["phone_archive_sha256"],
        "source_video_sha256": scan["video_sha256"], "packet_timing_sha256": scan["packet_sha256"],
        "calibration_sha256": calibration["current_calibration_sha256"],
        "calibration_bundle_sha256": calibration["bundle_sha256"],
        "frame_binding_sha256": calibration["binding_sha256"],
        "dewarper_sha256": dewarper["sha256"], "model": model_identity,
        "sampling": sampling,
        "fusion": {"depth_agreement_m": DEPTH_AGREEMENT_M, "min_observations": FUSION_MIN_OBSERVATIONS},
    }
    content_id = "paired_static_reference_" + _canonical_sha256(identity)[:16]
    revision_dir = scan_dir / "paired_static_reference" / content_id
    if revision_dir.exists():
        result = _cached_result(scan_dir, revision_dir)
        checked = validate_paired_static_reference(scan_dir, result)
        if checked["manifest"].get("identity") != identity:
            raise PairedStaticReferenceError("cached paired reference does not match its current inputs")
        progress(1.0, "Verified the saved paired static reference")
        return result
    revision_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = revision_dir.parent / f".{content_id}.building-{uuid.uuid4().hex[:10]}"
    tmp_dir.mkdir()
    model = None
    try:
        progress(0.04, "Selecting static observations across the recorded room walk")
        if len(set(selected)) != STATIC_FRAME_COUNT:
            raise PairedStaticReferenceError("static recording cannot supply six distinct observations")
        source_paths = _extract_selected_frames(scan["video_path"], selected, tmp_dir)
        (tmp_dir / "captured_dewarper.ini").write_text(dewarper["text"], encoding="utf-8")
        _write_json(tmp_dir / "calibration_bundle.json", calibration["bundle"])
        depth_snapshots, frame_rows, rectified_frames = [], [], []
        processed_k = None
        transform = dewarper["transform"]
        progress(0.10, "Loading MapAnything for the paired static recording")
        model = _load_model(model_settings)
        per_frame_files = []
        for sample_index, (frame_index, source_path) in enumerate(zip(selected, source_paths)):
            raw = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if raw is None or (raw.shape[1], raw.shape[0]) != tuple(transform.output_size):
                raise PairedStaticReferenceError("recorded raw dimensions do not match the captured dewarper")
            rectified_bgr, rectified_valid = _dewarp_valid_mask(raw, transform)
            if exclusion_masks is not None:
                rectified_valid &= ~exclusion_masks[sample_index]
                mask_name = f"exclusion_{sample_index:04d}.png"
                if not cv2.imwrite(str(tmp_dir / mask_name), exclusion_masks[sample_index].astype(np.uint8) * 255):
                    raise PairedStaticReferenceError("static exclusion mask could not be saved")
                per_frame_files.append(mask_name)
            depth, confidence, mask, scaled_k, model_rgb, model_diagnostics = _run_static_inference(
                model, cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB), calibration["K"], model_settings,
            )
            if processed_k is not None and (depth.shape != depth_snapshots[0].depth.shape or not np.allclose(scaled_k, processed_k, atol=1e-6, rtol=0)):
                raise PairedStaticReferenceError("static observations do not share one calibrated pixel grid")
            processed_k = scaled_k
            image_transform = scaled_k @ np.linalg.inv(calibration["K"])
            model_valid = cv2.warpPerspective(
                rectified_valid.astype(np.uint8), image_transform, (depth.shape[1], depth.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(bool)
            mask = np.asarray(mask, dtype=bool) & model_valid & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
            if np.count_nonzero(mask) < 5000:
                raise PairedStaticReferenceError(f"static observation {sample_index + 1} has too little valid calibrated depth")
            timestamp_ns = int(round(video_pts[frame_index] * 1e9))
            filename = f"static_{sample_index:04d}.npz"
            np.savez_compressed(
                tmp_dir / filename, depth_z=depth, confidence=confidence, mask=mask.astype(np.uint8),
                intrinsics=scaled_k, model_rgb=model_rgb, source_frame_index=np.asarray([frame_index]),
                encoded_pts_ns=np.asarray([timestamp_ns], dtype=np.int64),
                **model_diagnostics,
            )
            per_frame_files.append(filename)
            depth_snapshots.append(DepthSnapshot(
                path=tmp_dir / filename, timestamp_us=timestamp_ns // 1000,
                depth=depth.astype(np.float32), confidence=confidence.astype(np.float32), mask=mask,
                rgb=cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR),
                source_paths=(scan["video_path"],), source_timestamps_us=(timestamp_ns // 1000,),
            ))
            rectified_frames.append(rectified_bgr)
            frame_rows.append({
                "sample_index": sample_index, "source_frame_index": frame_index,
                "encoded_pts_ns": str(timestamp_ns), "timestamp_domain": "decoded_recording_pts",
                "packet_clock_correspondence_verified": False,
                "raw_frame_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "rectified_frame_sha256": hashlib.sha256(rectified_bgr.tobytes()).hexdigest(),
                "source_image_size": [raw.shape[1], raw.shape[0]],
                "model_image_size": [depth.shape[1], depth.shape[0]],
                "rectified_to_model_pixel_transform": image_transform.tolist(),
                "intrinsics": scaled_k.tolist(), "raw_artifact": filename,
                "model_intrinsics_diagnostic": model_diagnostics["model_intrinsics"].tolist(),
                "depth_semantics": "model_metric_range_on_captured_calibrated_rays_converted_to_z",
                "foreground_excluded_pixels": int(np.count_nonzero(exclusion_masks[sample_index])) if exclusion_masks is not None else 0,
            })
            source_path.unlink()
            progress(0.12 + 0.58 * ((sample_index + 1) / STATIC_FRAME_COUNT), f"Reconstructing static observation {sample_index + 1} of {STATIC_FRAME_COUNT}")
        fused = _fuse_depth_snapshots(
            depth_snapshots, camera_id=scan["camera_id"], min_confidence=0.0,
            min_observations=FUSION_MIN_OBSERVATIONS, depth_agreement_m=DEPTH_AGREEMENT_M,
            fusion_level="paired_static", fusion_mode="independent_static_depth_consensus",
        )
        valid = fused.mask & np.isfinite(fused.depth) & (fused.depth > 0)
        if np.count_nonzero(valid) < 5000 or processed_k is None:
            raise PairedStaticReferenceError("static depth consensus has too little shared surface support")
        ys, xs = np.nonzero(valid)
        z = fused.depth[ys, xs].astype(np.float64)
        camera_points = np.column_stack(((xs - processed_k[0, 2]) * z / processed_k[0, 0], (ys - processed_k[1, 2]) * z / processed_k[1, 1], z))
        camera_to_world = calibration["target_from_calibration"] @ np.linalg.inv(calibration["E"])
        points_world = (camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]).astype(np.float32)
        colors = cv2.cvtColor(fused.rgb, cv2.COLOR_BGR2RGB)[ys, xs]
        budget = int(model_settings.point_budget)
        if len(points_world) > budget:
            keep = np.linspace(0, len(points_world) - 1, budget, dtype=np.int64)
            points_world, colors = points_world[keep], colors[keep]
        np.savez_compressed(tmp_dir / "room_points.npz", points=points_world, colors=colors)
        np.savez_compressed(tmp_dir / "fused_depth.npz", depth_z=fused.depth, confidence=fused.confidence, mask=valid.astype(np.uint8), intrinsics=processed_k, camera_to_world=camera_to_world)
        keyframe = (rectified_frames[keyframe_sample] if keyframe_sample is not None
                    else np.median(np.stack(rectified_frames), axis=0).astype(np.uint8))
        keyframe_masks = {}
        if exclusion_masks is not None:
            keyframe_mask = (~exclusion_masks[keyframe_sample]).astype(np.uint8) * 255
            if not cv2.imwrite(str(tmp_dir / "keyframe_valid_mask.png"), keyframe_mask):
                raise PairedStaticReferenceError("static keyframe mask could not be saved")
            per_frame_files.append("keyframe_valid_mask.png")
            keyframe_masks = {scan["camera_id"]: "keyframe_valid_mask.png"}
        if not cv2.imwrite(str(tmp_dir / "keyframe.png"), keyframe, [cv2.IMWRITE_PNG_COMPRESSION, 4]) or not cv2.imwrite(str(tmp_dir / "depth_preview.png"), _depth_preview(fused.depth, valid)):
            raise PairedStaticReferenceError("static reference image could not be saved")
        world_revision = calibration["binding"]["world_frame"]["revision"]
        calibration_payload = {
            "schema": CALIBRATION_SCHEMA,
            "cameras": {scan["camera_id"]: {"E": calibration["E_col_major"], "K": calibration["K_flat"]}},
            "world_frame": "backend_world_m", "world_frame_revision": world_revision,
            "E_semantics": "camera_from_calibration_frame_raw", "frame_binding": calibration["binding"],
            "source_calibration_sha256": calibration["current_calibration_sha256"],
        }
        _write_json(tmp_dir / "calibration.json", calibration_payload)
        common = {
            "camera": scan["camera_id"], "camera_id": scan["camera_id"], "revision_id": content_id,
            "reference_kind": "paired_static", "companion_session_id": scan["session_id"],
            "coordinate_frame": "backend_world_m_stream_points", "world_frame": "backend_world_m",
            "world_frame_revision": world_revision, "camera_frame_binding": calibration["binding"],
            "floor_alignment": {"status": "captured_calibration_frame_binding", "world_correction_col_major": calibration["target_from_calibration_col_major"]},
            "admission": "independent_alignment_reference_not_promoted_to_live_noesis",
        }
        metadata = {
            **common, "schema": "noesis.room_reconstruction.stream_points.v4",
            "source": "paired_static_recording_independent_mapanything_depth_consensus",
            "point_count": len(points_world), "floor_y": 0.0, "intrinsics": calibration["K"].tolist(),
            "extrinsics_col_major": calibration["E_col_major"],
            "rgb_keyframes": {scan["camera_id"]: "keyframe.png"},
            "keyframe_source": (f"selected_recorded_observation_{keyframe_sample}" if keyframe_sample is not None
                                else "pixelwise_median_of_six_captured_rectified_observations"),
            **({"rgb_keyframe_valid_masks": keyframe_masks} if keyframe_masks else {}),
            "image_size": [keyframe.shape[1], keyframe.shape[0]],
            "bounds": {"min": points_world.min(axis=0).tolist(), "max": points_world.max(axis=0).tolist()},
        }
        _write_json(tmp_dir / "room_points_meta.json", metadata)
        artifacts = {"keyframe": "keyframe.png", "depth_preview": "depth_preview.png", "points": "room_points.npz", "metadata": "room_points_meta.json", "calibration": "calibration.json", "fused_depth": "fused_depth.npz", "manifest": "manifest.json"}
        names = [value for value in artifacts.values() if value != "manifest.json"] + ["captured_dewarper.ini", "calibration_bundle.json"] + per_frame_files
        manifest = {
            **common, "schema": SCHEMA, "status": "complete", "session_id": scan["session_id"],
            "scan_id": scan["scan_id"], "phone_capture_id": scan["phone_capture_id"], "identity": identity,
            "source": {"video_path": str(scan["video_path"]), "video_sha256": scan["video_sha256"],
                       "packet_timing_path": str(scan["packet_path"]), "packet_timing_sha256": scan["packet_sha256"],
                       "packet_count": len(scan["packet_rows"]), "decoded_frame_count": len(video_pts),
                       "encoded_duration_s": float(video_pts[-1] - video_pts[0]), "frames": frame_rows,
                       "acquisition_timing_verified": False, "packet_clock_correspondence_verified": False},
            "calibration": {"path": "calibration.json", "sha256": _sha256(tmp_dir / "calibration.json")},
            "model": model_identity, "fusion": {**(fused.fusion_meta or {}), "source_artifacts": per_frame_files},
            "exclusions": {"phone_geometry_imported": False, "phone_poses_imported": False, "tracking_world_observations_used": False},
            "artifacts": artifacts, "files": _artifact_rows(tmp_dir, names),
        }
        _write_json(tmp_dir / "manifest.json", manifest)
        # Validate while private, before exposing the complete revision.
        temporary_reference = {"target_revision": str(tmp_dir.relative_to(scan_dir)), "manifest_sha256": _sha256(tmp_dir / "manifest.json")}
        _validate_reference_files(tmp_dir, manifest)
        os.replace(tmp_dir, revision_dir)
        result = _cached_result(scan_dir, revision_dir, temporary_reference["manifest_sha256"])
        validate_paired_static_reference(scan_dir, result)
        progress(1.0, "Paired static reconstruction and provenance are saved")
        return result
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        if model is not None:
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


__all__ = [
    "PairedStaticReferenceError",
    "prepare_paired_static_reference",
    "validate_paired_static_reference",
]
