"""Extract sharper texture photos without changing prepared-view identity.

This is deliberately separate from frame preparation.  It decodes one exact
native source frame for every already-selected prepared view, and uses the
prepared JPEGs only to prove the image-to-row mapping.  It does not select new
views, seek by time, modify camera geometry, or promote reconstruction state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity


OUTPUT_SCHEMA = "noesis.mapanything.phone_scan.bound_texture_photos.v1"
OUTPUT_MANIFEST_NAME = "bound_texture_photos_manifest.json"
MAX_LONG_EDGE_PX = 4096
MAX_BOUND_VIEWS = 256
FFMPEG_THREADS = 4
FFMPEG_TIMEOUT_S = 1800
TRUSTED_TIMESTAMP_SOURCE = "android_camera2_sensor_timestamp_exact_encoder_association"
TRUSTED_ASSOCIATION_METHOD = "encoder_pts_us_equals_sensor_timestamp_ns_div_1000"

# The canonical small image is the same gray -> <=640 edge -> 160x90 path used
# by processing.py's repeat guard.  A declared pair must look like the same
# decoded content after JPEG/resampling noise, and no other cohort row may pass.
REPEAT_MEAN_DELTA_MAX = 1.50
REPEAT_P95_DELTA_MAX = 4.0
REPEAT_CHANGED_FRACTION_MAX = 0.02
IMPOSTOR_MEAN_DELTA_GAP_MIN = 4.0
CANONICAL_RMSE_MAX = 4.0
CANONICAL_NCC_MIN = 0.995

# Full-resolution luma and feature checks independently reject a photometrically
# close thumbnail whose spatial mapping is not the same frame geometry.
SSIM_MIN = 0.94
FEATURE_MATCH_MIN = 40
FEATURE_INLIER_MIN = 32
FEATURE_INLIER_RATIO_MIN = 0.75
FEATURE_COVERAGE_MIN = 0.08
FEATURE_MEDIAN_DISPLACEMENT_NORM_MAX = 0.004
FEATURE_P95_DISPLACEMENT_NORM_MAX = 0.012
FEATURE_CORNER_DISPLACEMENT_NORM_MAX = 0.006

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class BoundTexturePhotoError(RuntimeError):
    """Raised when exact source or image-row binding cannot be proven."""


@dataclass(frozen=True)
class PreparedFrame:
    index: int
    frame_id: str
    path: Path
    relative_path: str
    sha256: str
    width: int
    height: int
    source_frame_index: int
    capture_time_ns: int
    timestamp_s: float
    timestamp_source: str


@dataclass(frozen=True)
class ValidatedInputs:
    scan_dir: Path
    source_video: Path
    prepared_source_video: Path
    prepared_manifest_path: Path
    prepared_manifest_sha256: str
    prepared_manifest: dict[str, Any]
    frames: tuple[PreparedFrame, ...]
    source_video_sha256: str
    source_video_size_bytes: int
    source_width: int
    source_height: int
    native_mapping_path: Path
    native_mapping_sha256: str
    capture_import_path: Path
    capture_import_sha256: str
    native_frame_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundTexturePhotoError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise BoundTexturePhotoError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BoundTexturePhotoError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise BoundTexturePhotoError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise BoundTexturePhotoError(f"{label} must be a finite number >= {minimum}")
    return result


def _declared_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BoundTexturePhotoError(f"{label} must be a lowercase SHA-256")
    return value


def _read_image(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise BoundTexturePhotoError(f"{label} is not a readable three-channel image")
    return image


def _lexically_inside(root: Path, relative: Any, label: str) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative.strip():
        raise BoundTexturePhotoError(f"{label} path is missing")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise BoundTexturePhotoError(
            f"{label} path must stay inside the scan directory"
        )
    path = root / relative_path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BoundTexturePhotoError(f"{label} path is unreadable: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise BoundTexturePhotoError(
            f"{label} path resolves outside the scan directory"
        )
    if not resolved.is_file():
        raise BoundTexturePhotoError(f"{label} path is not a file")
    return resolved, relative_path.as_posix()


def _native_timing_evidence(
    scan_dir: Path,
) -> tuple[list[int], Path, str, Path, str, dict[str, Any]]:
    capture = scan_dir / "capture"
    import_path = capture / "capture_import.json"
    report, report_sha = _json(import_path, "native capture import")
    native = report.get("android_capture")
    video = report.get("video")
    if not isinstance(native, dict) or not isinstance(video, dict):
        raise BoundTexturePhotoError(
            "native capture import lacks video association evidence"
        )
    if (
        native.get("camera_acquisition_timestamp_verified") is not True
        or native.get("association_method") != TRUSTED_ASSOCIATION_METHOD
        or video.get("camera_acquisition_timestamp_verified") is not True
        or video.get("timestamp_source") != TRUSTED_TIMESTAMP_SOURCE
    ):
        raise BoundTexturePhotoError(
            "verified native Camera2-to-encoded-frame association is required"
        )

    mapping_path = capture / "video_timestamps_ns.json"
    try:
        mapping_raw = mapping_path.read_bytes()
        raw_timestamps = json.loads(mapping_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundTexturePhotoError(
            f"native frame timestamp mapping is unreadable: {exc}"
        ) from exc
    if not isinstance(raw_timestamps, list) or not raw_timestamps:
        raise BoundTexturePhotoError(
            "native frame timestamp mapping must be a nonempty list"
        )
    if len(raw_timestamps) > 1_000_000:
        raise BoundTexturePhotoError(
            "native frame timestamp mapping exceeds the safety bound"
        )
    timestamps = [
        _exact_int(value, f"native frame timestamp {index}", minimum=1)
        for index, value in enumerate(raw_timestamps)
    ]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise BoundTexturePhotoError(
            "native frame timestamps must be strictly increasing"
        )

    count = len(timestamps)
    for field in (
        "association_row_count",
        "decoded_frame_count",
        "matched_frame_count",
    ):
        if _exact_int(native.get(field), f"native {field}") != count:
            raise BoundTexturePhotoError(
                f"native {field} disagrees with frame timestamps"
            )
    if _exact_int(video.get("frame_count"), "native video frame_count") != count:
        raise BoundTexturePhotoError(
            "native video frame_count disagrees with frame timestamps"
        )
    if (
        _exact_int(video.get("start_time_ns"), "native video start_time_ns", minimum=1)
        != timestamps[0]
        or _exact_int(video.get("end_time_ns"), "native video end_time_ns", minimum=1)
        != timestamps[-1]
    ):
        raise BoundTexturePhotoError(
            "native video endpoints disagree with frame timestamps"
        )
    return (
        timestamps,
        mapping_path.resolve(),
        hashlib.sha256(mapping_raw).hexdigest(),
        import_path.resolve(),
        report_sha,
        report,
    )


def _validate_inputs(
    scan_dir: Path,
    source_video: Path,
    long_edge_px: int,
) -> ValidatedInputs:
    if type(long_edge_px) is not int or not 256 <= long_edge_px <= MAX_LONG_EDGE_PX:
        raise BoundTexturePhotoError(
            f"long-edge-px must be an integer from 256 through {MAX_LONG_EDGE_PX}"
        )
    if long_edge_px % 2:
        raise BoundTexturePhotoError("long-edge-px must be even")
    try:
        scan = scan_dir.resolve(strict=True)
        source = source_video.resolve(strict=True)
    except OSError as exc:
        raise BoundTexturePhotoError(f"input path is unreadable: {exc}") from exc
    if not scan.is_dir() or not source.is_file():
        raise BoundTexturePhotoError(
            "scan-dir must be a directory and source-video a file"
        )

    manifest_path = scan / "prepared_frames_manifest.json"
    prepared, manifest_sha = _json(manifest_path, "prepared frame manifest")
    if prepared.get("schema") != "noesis.mapanything.phone_scan.prepared_frames.v2":
        raise BoundTexturePhotoError("unsupported prepared frame manifest schema")
    rows = prepared.get("frames")
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_BOUND_VIEWS:
        raise BoundTexturePhotoError(
            "prepared frame count must be from 2 through the 256-view preparation bound"
        )
    preparation = prepared.get("preparation")
    if not isinstance(preparation, dict) or _exact_int(
        preparation.get("frame_count"), "prepared frame_count", minimum=1
    ) != len(rows):
        raise BoundTexturePhotoError("prepared frame_count disagrees with frame rows")

    video = prepared.get("video")
    if not isinstance(video, dict):
        raise BoundTexturePhotoError("prepared manifest has no source video identity")
    expected_video_sha = _declared_sha256(video.get("sha256"), "source video sha256")
    expected_video_size = _exact_int(
        video.get("size_bytes"), "source video size_bytes", minimum=1
    )
    source_width = _exact_int(video.get("width"), "source video width", minimum=1)
    source_height = _exact_int(video.get("height"), "source video height", minimum=1)
    reported_frame_count = _exact_int(
        video.get("reported_frame_count"), "source reported_frame_count", minimum=1
    )
    rotation = _finite_float(video.get("rotation_degrees"), "source rotation_degrees")
    if rotation != 0.0:
        raise BoundTexturePhotoError(
            "nonzero source rotation is unsupported because exact noautorotate geometry is required"
        )
    if source.stat().st_size != expected_video_size:
        raise BoundTexturePhotoError(
            "explicit source video size differs from prepared manifest"
        )
    actual_video_sha = _sha256(source)
    if actual_video_sha != expected_video_sha:
        raise BoundTexturePhotoError(
            "explicit source video SHA-256 differs from prepared manifest"
        )

    source_relative = video.get("path")
    if (
        not isinstance(source_relative, str)
        or Path(source_relative).is_absolute()
        or ".." in Path(source_relative).parts
    ):
        raise BoundTexturePhotoError(
            "prepared source video path must be relative and bounded"
        )
    manifest_source_path = scan / source_relative
    try:
        manifest_source_resolved = manifest_source_path.resolve(strict=True)
    except OSError as exc:
        raise BoundTexturePhotoError(
            f"prepared source video path is unreadable: {exc}"
        ) from exc
    if not manifest_source_resolved.is_file():
        raise BoundTexturePhotoError("prepared source video path is not a file")
    if manifest_source_resolved.stat().st_size != expected_video_size:
        raise BoundTexturePhotoError(
            "prepared-manifest source path size no longer matches its identity"
        )
    if (
        not os.path.samefile(manifest_source_resolved, source)
        and _sha256(manifest_source_resolved) != expected_video_sha
    ):
        raise BoundTexturePhotoError(
            "prepared-manifest source path no longer matches its SHA-256 identity"
        )

    (
        native_timestamps,
        native_mapping_path,
        native_mapping_sha,
        capture_import_path,
        capture_import_sha,
        _,
    ) = _native_timing_evidence(scan)
    if reported_frame_count != len(native_timestamps):
        raise BoundTexturePhotoError(
            "prepared reported frame count disagrees with exact native frame mapping"
        )

    calibration = prepared.get("camera_calibration")
    if calibration is not None and (
        not isinstance(calibration, dict) or calibration.get("status") != "not_applied"
    ):
        raise BoundTexturePhotoError(
            "raw native photo extraction requires explicitly not-applied camera calibration"
        )

    frames: list[PreparedFrame] = []
    seen_source_indices: set[int] = set()
    previous_source_index = -1
    previous_capture_time = -1
    previous_timestamp = -math.inf
    prepared_size: tuple[int, int] | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BoundTexturePhotoError(f"prepared frame {index} must be an object")
        if _exact_int(row.get("index"), f"prepared frame {index} index") != index:
            raise BoundTexturePhotoError(f"prepared frame {index} order/index mismatch")
        sha = _declared_sha256(row.get("sha256"), f"prepared frame {index} sha256")
        expected_id = prepared_frame_identity(index, sha)
        if row.get("frame_id") != expected_id:
            raise BoundTexturePhotoError(f"prepared frame {index} frame_id mismatch")
        frame_path, relative_path = _lexically_inside(
            scan, row.get("frame"), f"prepared frame {index}"
        )
        if _sha256(frame_path) != sha:
            raise BoundTexturePhotoError(f"prepared frame {index} file hash mismatch")
        width = _exact_int(row.get("width"), f"prepared frame {index} width", minimum=1)
        height = _exact_int(
            row.get("height"), f"prepared frame {index} height", minimum=1
        )
        image = _read_image(frame_path, f"prepared frame {index}")
        if (image.shape[1], image.shape[0]) != (width, height):
            raise BoundTexturePhotoError(f"prepared frame {index} dimensions mismatch")
        if prepared_size is None:
            prepared_size = (width, height)
        elif prepared_size != (width, height):
            raise BoundTexturePhotoError(
                "prepared frame dimensions are not one fixed cohort"
            )
        if abs(width * source_height - height * source_width) > 2 * max(
            source_width, source_height
        ):
            raise BoundTexturePhotoError(
                f"prepared frame {index} aspect ratio differs from noautorotate source geometry"
            )

        source_index = _exact_int(
            row.get("source_frame_index"), f"prepared frame {index} source_frame_index"
        )
        if source_index in seen_source_indices:
            raise BoundTexturePhotoError(
                "prepared source_frame_index values must be distinct"
            )
        if source_index <= previous_source_index:
            raise BoundTexturePhotoError(
                "prepared source_frame_index values must be strictly increasing"
            )
        if source_index >= len(native_timestamps):
            raise BoundTexturePhotoError(
                f"prepared frame {index} source_frame_index exceeds exact decoded count"
            )
        capture_time = _exact_int(
            row.get("capture_time_ns"),
            f"prepared frame {index} capture_time_ns",
            minimum=1,
        )
        if capture_time != native_timestamps[source_index]:
            raise BoundTexturePhotoError(
                f"prepared frame {index} acquisition time disagrees with native mapping"
            )
        if capture_time <= previous_capture_time:
            raise BoundTexturePhotoError(
                "prepared acquisition times must be strictly increasing"
            )
        timestamp = _finite_float(
            row.get("timestamp_s"), f"prepared frame {index} timestamp_s"
        )
        expected_timestamp = (capture_time - native_timestamps[0]) / 1e9
        if abs(timestamp - expected_timestamp) > 1e-9:
            raise BoundTexturePhotoError(
                f"prepared frame {index} timestamp_s disagrees with native acquisition time"
            )
        if timestamp <= previous_timestamp and index:
            raise BoundTexturePhotoError(
                "prepared timestamps must be strictly increasing"
            )
        timestamp_source = row.get("timestamp_source")
        if timestamp_source != TRUSTED_TIMESTAMP_SOURCE:
            raise BoundTexturePhotoError(
                f"prepared frame {index} lacks trusted exact native timestamp provenance"
            )
        if "calibration_processing" in row:
            raise BoundTexturePhotoError(
                f"prepared frame {index} uses rectified pixels unsupported by raw extraction"
            )
        frames.append(
            PreparedFrame(
                index=index,
                frame_id=expected_id,
                path=frame_path,
                relative_path=relative_path,
                sha256=sha,
                width=width,
                height=height,
                source_frame_index=source_index,
                capture_time_ns=capture_time,
                timestamp_s=timestamp,
                timestamp_source=timestamp_source,
            )
        )
        seen_source_indices.add(source_index)
        previous_source_index = source_index
        previous_capture_time = capture_time
        previous_timestamp = timestamp

    assert prepared_size is not None
    if long_edge_px <= max(prepared_size):
        raise BoundTexturePhotoError(
            "long-edge-px must be larger than the prepared JPEG edge; prepared images are never upsampled"
        )
    if long_edge_px > max(source_width, source_height):
        raise BoundTexturePhotoError(
            "long-edge-px exceeds native source resolution; native video pixels are never upsampled"
        )

    return ValidatedInputs(
        scan_dir=scan,
        source_video=source,
        prepared_source_video=manifest_source_resolved,
        prepared_manifest_path=manifest_path.resolve(),
        prepared_manifest_sha256=manifest_sha,
        prepared_manifest=prepared,
        frames=tuple(frames),
        source_video_sha256=actual_video_sha,
        source_video_size_bytes=expected_video_size,
        source_width=source_width,
        source_height=source_height,
        native_mapping_path=native_mapping_path,
        native_mapping_sha256=native_mapping_sha,
        capture_import_path=capture_import_path,
        capture_import_sha256=capture_import_sha,
        native_frame_count=len(native_timestamps),
    )


def _scale_filter(long_edge_px: int) -> str:
    return (
        f"scale={long_edge_px}:{long_edge_px}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,format=rgb24"
    )


def _ffmpeg_command(
    source_video: Path,
    output_pattern: Path,
    source_indices: Sequence[int],
    long_edge_px: int,
) -> list[str]:
    if not source_indices:
        raise BoundTexturePhotoError("cannot build an empty exact-frame selection")
    selection = "+".join(f"eq(n\\,{index})" for index in source_indices)
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-filter_threads",
        str(FFMPEG_THREADS),
        "-threads",
        str(FFMPEG_THREADS),
        "-hwaccel",
        "none",
        "-noautorotate",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-vf",
        f"select='{selection}',{_scale_filter(long_edge_px)}",
        "-frames:v",
        str(len(source_indices)),
        "-fps_mode",
        "vfr",
        "-threads",
        str(FFMPEG_THREADS),
        "-compression_level",
        "4",
        "-start_number",
        "0",
        str(output_pattern),
    ]


def _run_ffmpeg(command: Sequence[str]) -> None:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise BoundTexturePhotoError(
            f"exact native frame extraction exceeded {FFMPEG_TIMEOUT_S} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-4000:]
        raise BoundTexturePhotoError(
            f"exact native frame extraction failed: {detail or completed.returncode}"
        )


def _feature_gray(gray: np.ndarray) -> np.ndarray:
    scale = min(1.0, 640.0 / float(max(gray.shape)))
    if scale >= 0.999:
        return gray
    return cv2.resize(
        gray,
        (
            max(1, int(round(gray.shape[1] * scale))),
            max(1, int(round(gray.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _canonical_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(_feature_gray(gray), (160, 90), interpolation=cv2.INTER_AREA)


def _ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    mean_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mean_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mean_a * mean_a
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mean_b * mean_b
    covariance = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mean_a * mean_b
    numerator = (2.0 * mean_a * mean_b + 6.5025) * (2.0 * covariance + 58.5225)
    denominator = (mean_a * mean_a + mean_b * mean_b + 6.5025) * (
        var_a + var_b + 58.5225
    )
    value = float(np.mean(numerator / np.maximum(denominator, 1e-12)))
    if not math.isfinite(value):
        raise BoundTexturePhotoError("SSIM calculation produced a non-finite result")
    return value


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    left = a.astype(np.float64).ravel()
    right = b.astype(np.float64).ravel()
    left -= float(np.mean(left))
    right -= float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(left, right) / denominator)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if not len(points):
        return 0.0
    occupied = {
        (
            min(5, max(0, int(point[0] * 6 / width))),
            min(3, max(0, int(point[1] * 4 / height))),
        )
        for point in points
    }
    return len(occupied) / 24.0


def _feature_mapping_proof(
    prepared_gray: np.ndarray,
    extracted_gray: np.ndarray,
    index: int,
) -> dict[str, Any]:
    prepared = _feature_gray(prepared_gray)
    extracted = _feature_gray(extracted_gray)
    if prepared.shape != extracted.shape:
        raise BoundTexturePhotoError(f"view {index} feature-image dimensions differ")
    detector = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        fastThreshold=10,
    )
    prepared_points, prepared_descriptors = detector.detectAndCompute(prepared, None)
    extracted_points, extracted_descriptors = detector.detectAndCompute(extracted, None)
    if prepared_descriptors is None or extracted_descriptors is None:
        raise BoundTexturePhotoError(
            f"view {index} lacks features for spatial identity proof"
        )
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        extracted_descriptors, prepared_descriptors, k=2
    )
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.78 * second.distance
    ]
    if len(good) < FEATURE_MATCH_MIN:
        raise BoundTexturePhotoError(
            f"view {index} has only {len(good)} feature matches; spatial identity is unproven"
        )
    extracted_xy = np.float32([extracted_points[match.queryIdx].pt for match in good])
    prepared_xy = np.float32([prepared_points[match.trainIdx].pt for match in good])
    homography, mask = cv2.findHomography(extracted_xy, prepared_xy, cv2.RANSAC, 2.0)
    if homography is None or mask is None or not np.isfinite(homography).all():
        raise BoundTexturePhotoError(f"view {index} feature mapping is degenerate")
    inliers = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / len(good)
    if inlier_count < FEATURE_INLIER_MIN or inlier_ratio < FEATURE_INLIER_RATIO_MIN:
        raise BoundTexturePhotoError(
            f"view {index} feature mapping lacks same-frame inlier support"
        )
    source_inliers = extracted_xy[inliers]
    target_inliers = prepared_xy[inliers]
    height, width = prepared.shape
    diagonal = math.hypot(width, height)
    displacement = np.linalg.norm(source_inliers - target_inliers, axis=1) / diagonal
    median_displacement = float(np.median(displacement))
    p95_displacement = float(np.percentile(displacement, 95.0))
    source_coverage = _coverage(source_inliers, width, height)
    target_coverage = _coverage(target_inliers, width, height)
    corners = np.float32(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]
    )
    mapped_corners = cv2.perspectiveTransform(corners, homography)
    corner_displacement = float(
        np.max(np.linalg.norm(mapped_corners - corners, axis=2)) / diagonal
    )
    if (
        min(source_coverage, target_coverage) < FEATURE_COVERAGE_MIN
        or median_displacement > FEATURE_MEDIAN_DISPLACEMENT_NORM_MAX
        or p95_displacement > FEATURE_P95_DISPLACEMENT_NORM_MAX
        or corner_displacement > FEATURE_CORNER_DISPLACEMENT_NORM_MAX
    ):
        raise BoundTexturePhotoError(
            f"view {index} feature displacement is inconsistent with same-frame geometry"
        )
    return {
        "feature_edge_px": 640,
        "prepared_feature_count": len(prepared_points),
        "extracted_feature_count": len(extracted_points),
        "ratio_match_count": len(good),
        "homography_inlier_count": inlier_count,
        "homography_inlier_ratio": inlier_ratio,
        "source_grid_coverage": source_coverage,
        "target_grid_coverage": target_coverage,
        "median_raw_match_displacement_norm": median_displacement,
        "p95_raw_match_displacement_norm": p95_displacement,
        "maximum_homography_corner_displacement_norm": corner_displacement,
    }


def _verify_photo_mapping(
    inputs: ValidatedInputs,
    photo_paths: Sequence[Path],
    long_edge_px: int,
) -> list[dict[str, Any]]:
    if len(photo_paths) != len(inputs.frames):
        raise BoundTexturePhotoError(
            "ffmpeg did not produce exactly one photo per prepared frame"
        )

    prepared_canonical: list[np.ndarray] = []
    for frame in inputs.frames:
        if _sha256(frame.path) != frame.sha256:
            raise BoundTexturePhotoError(
                f"prepared frame {frame.index} changed during extraction"
            )
        image = _read_image(frame.path, f"prepared frame {frame.index}")
        if (image.shape[1], image.shape[0]) != (frame.width, frame.height):
            raise BoundTexturePhotoError(
                f"prepared frame {frame.index} dimensions changed during extraction"
            )
        prepared_canonical.append(_canonical_gray(image))

    extracted_canonical: list[np.ndarray] = []
    output_size: tuple[int, int] | None = None
    for index, path in enumerate(photo_paths):
        image = _read_image(path, f"extracted view {index}")
        width, height = image.shape[1], image.shape[0]
        if max(width, height) != long_edge_px or width % 2 or height % 2:
            raise BoundTexturePhotoError(
                f"extracted view {index} does not have the requested even long edge"
            )
        if abs(width * inputs.source_height - height * inputs.source_width) > 2 * max(
            inputs.source_width, inputs.source_height
        ):
            raise BoundTexturePhotoError(
                f"extracted view {index} aspect ratio differs from native source geometry"
            )
        if output_size is None:
            output_size = (width, height)
        elif output_size != (width, height):
            raise BoundTexturePhotoError("extracted photos do not share one fixed size")
        prepared = inputs.frames[index]
        downscaled = cv2.resize(
            image,
            (prepared.width, prepared.height),
            interpolation=cv2.INTER_AREA,
        )
        extracted_canonical.append(_canonical_gray(downscaled))

    mean_matrix = np.empty((len(photo_paths), len(photo_paths)), dtype=np.float32)
    pass_matrix = np.zeros_like(mean_matrix, dtype=bool)
    detailed_stats: dict[tuple[int, int], tuple[float, float, float]] = {}
    for output_index, extracted in enumerate(extracted_canonical):
        for prepared_index, prepared in enumerate(prepared_canonical):
            delta = np.abs(
                extracted.astype(np.int16) - prepared.astype(np.int16)
            ).astype(np.float32)
            mean_delta = float(np.mean(delta))
            mean_matrix[output_index, prepared_index] = mean_delta
            if mean_delta <= REPEAT_MEAN_DELTA_MAX:
                p95_delta = float(np.percentile(delta, 95.0))
                changed_fraction = float(np.mean(delta > 4.0))
                detailed_stats[(output_index, prepared_index)] = (
                    mean_delta,
                    p95_delta,
                    changed_fraction,
                )
                pass_matrix[output_index, prepared_index] = bool(
                    p95_delta <= REPEAT_P95_DELTA_MAX
                    and changed_fraction <= REPEAT_CHANGED_FRACTION_MAX
                )

    for index in range(len(photo_paths)):
        row_matches = np.flatnonzero(pass_matrix[index]).tolist()
        column_matches = np.flatnonzero(pass_matrix[:, index]).tolist()
        if row_matches != [index] or column_matches != [index]:
            raise BoundTexturePhotoError(
                f"view {index} image mapping is mismatched or ambiguous: "
                f"row_matches={row_matches}, column_matches={column_matches}"
            )

    reports: list[dict[str, Any]] = []
    for index, (frame, output_path) in enumerate(
        zip(inputs.frames, photo_paths, strict=True)
    ):
        intended = detailed_stats[(index, index)]
        impostor_values = np.delete(mean_matrix[index], index)
        reverse_impostor_values = np.delete(mean_matrix[:, index], index)
        nearest_impostor = (
            float(np.min(impostor_values)) if impostor_values.size else None
        )
        nearest_reverse_impostor = (
            float(np.min(reverse_impostor_values))
            if reverse_impostor_values.size
            else None
        )
        for value in (nearest_impostor, nearest_reverse_impostor):
            if value is not None and value - intended[0] < IMPOSTOR_MEAN_DELTA_GAP_MIN:
                raise BoundTexturePhotoError(
                    f"view {index} lacks the required unique-image separation margin"
                )

        prepared = _read_image(frame.path, f"prepared frame {index}")
        output_image = _read_image(output_path, f"extracted view {index}")
        extracted = cv2.resize(
            output_image,
            (frame.width, frame.height),
            interpolation=cv2.INTER_AREA,
        )
        prepared_gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
        ssim = _ssim(prepared_gray, extracted_gray)
        ncc = _ncc(prepared_canonical[index], extracted_canonical[index])
        intended_delta = extracted_canonical[index].astype(
            np.float32
        ) - prepared_canonical[index].astype(np.float32)
        rmse = float(np.sqrt(np.mean(intended_delta * intended_delta)))
        if ssim < SSIM_MIN or ncc < CANONICAL_NCC_MIN or rmse > CANONICAL_RMSE_MAX:
            raise BoundTexturePhotoError(
                f"view {index} photometric identity proof failed "
                f"(SSIM={ssim:.6f}, NCC={ncc:.6f}, RMSE={rmse:.3f})"
            )
        feature_report = _feature_mapping_proof(prepared_gray, extracted_gray, index)
        reports.append(
            {
                "index": index,
                "frame_id": frame.frame_id,
                "prepared_photo": {
                    "path": frame.relative_path,
                    "sha256": frame.sha256,
                    "width": frame.width,
                    "height": frame.height,
                },
                "source_frame_index": frame.source_frame_index,
                "capture_time_ns": frame.capture_time_ns,
                "timestamp_s": frame.timestamp_s,
                "timestamp_source": frame.timestamp_source,
                "output_photo": {
                    "path": f"photos/frame_{index:04d}.png",
                    "sha256": _sha256(output_path),
                    "width": int(output_image.shape[1]),
                    "height": int(output_image.shape[0]),
                    "encoding": "png_lossless_rgb24",
                },
                "image_mapping_verification": {
                    "status": "verified_unique_same_decoded_frame_geometry",
                    "canonical_gray_size": [160, 90],
                    "canonical_mean_absolute_delta": intended[0],
                    "canonical_p95_absolute_delta": intended[1],
                    "canonical_changed_fraction_over_4": intended[2],
                    "canonical_rmse": rmse,
                    "canonical_zero_mean_ncc": ncc,
                    "luma_ssim_after_native_downscale": ssim,
                    "nearest_other_prepared_mean_absolute_delta": nearest_impostor,
                    "nearest_other_output_mean_absolute_delta": nearest_reverse_impostor,
                    "minimum_impostor_mean_delta_gap": (
                        min(nearest_impostor, nearest_reverse_impostor) - intended[0]
                        if nearest_impostor is not None
                        and nearest_reverse_impostor is not None
                        else None
                    ),
                    **feature_report,
                },
            }
        )
    return reports


def _verification_policy() -> dict[str, Any]:
    return {
        "prepared_comparison": (
            "native PNG downscaled to the bound prepared dimensions with OpenCV INTER_AREA"
        ),
        "canonicalization": (
            "BGR to gray, INTER_AREA to at most 640-pixel edge, then INTER_AREA to 160x90"
        ),
        "all_pairs_mapping_required": True,
        "declared_pair_must_be_only_repeat_gate_match_in_its_row_and_column": True,
        "canonical_mean_absolute_delta_max": REPEAT_MEAN_DELTA_MAX,
        "canonical_p95_absolute_delta_max": REPEAT_P95_DELTA_MAX,
        "canonical_changed_fraction_over_4_max": REPEAT_CHANGED_FRACTION_MAX,
        "canonical_rmse_max": CANONICAL_RMSE_MAX,
        "canonical_zero_mean_ncc_min": CANONICAL_NCC_MIN,
        "minimum_impostor_mean_delta_gap": IMPOSTOR_MEAN_DELTA_GAP_MIN,
        "luma_ssim_min": SSIM_MIN,
        "feature_match_min": FEATURE_MATCH_MIN,
        "feature_inlier_min": FEATURE_INLIER_MIN,
        "feature_inlier_ratio_min": FEATURE_INLIER_RATIO_MIN,
        "feature_grid_coverage_min": FEATURE_COVERAGE_MIN,
        "feature_median_displacement_norm_max": FEATURE_MEDIAN_DISPLACEMENT_NORM_MAX,
        "feature_p95_displacement_norm_max": FEATURE_P95_DISPLACEMENT_NORM_MAX,
        "feature_corner_displacement_norm_max": FEATURE_CORNER_DISPLACEMENT_NORM_MAX,
    }


def extract_bound_texture_photos(
    scan_dir: Path,
    source_video: Path,
    output_dir: Path,
    *,
    long_edge_px: int = 2560,
) -> Path:
    """Create a new, fail-closed bound-photo directory and return its manifest."""

    requested_output = output_dir.expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise BoundTexturePhotoError(
            "output-dir already exists; existing outputs are never overwritten"
        )
    output = requested_output.resolve()
    if output.exists() or output.is_symlink():
        raise BoundTexturePhotoError(
            "output-dir already exists; existing outputs are never overwritten"
        )
    inputs = _validate_inputs(scan_dir, source_video, long_edge_px)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        photo_dir = stage / "photos"
        photo_dir.mkdir()
        output_pattern = photo_dir / "frame_%04d.png"
        source_indices = [frame.source_frame_index for frame in inputs.frames]
        command = _ffmpeg_command(
            inputs.source_video, output_pattern, source_indices, long_edge_px
        )
        _run_ffmpeg(command)
        photo_paths = sorted(photo_dir.glob("frame_*.png"))
        expected_names = [
            f"frame_{index:04d}.png" for index in range(len(inputs.frames))
        ]
        if [path.name for path in photo_paths] != expected_names:
            raise BoundTexturePhotoError(
                "ffmpeg output names/count do not preserve prepared frame order"
            )
        views = _verify_photo_mapping(inputs, photo_paths, long_edge_px)

        if _sha256(inputs.prepared_manifest_path) != inputs.prepared_manifest_sha256:
            raise BoundTexturePhotoError("prepared manifest changed during extraction")
        if inputs.source_video.stat().st_size != inputs.source_video_size_bytes:
            raise BoundTexturePhotoError("source video size changed during extraction")
        if _sha256(inputs.source_video) != inputs.source_video_sha256:
            raise BoundTexturePhotoError("source video changed during extraction")
        try:
            prepared_source_after = (
                inputs.scan_dir / inputs.prepared_manifest["video"]["path"]
            ).resolve(strict=True)
        except OSError as exc:
            raise BoundTexturePhotoError(
                f"prepared-manifest source path changed during extraction: {exc}"
            ) from exc
        if prepared_source_after != inputs.prepared_source_video:
            raise BoundTexturePhotoError(
                "prepared-manifest source path changed during extraction"
            )
        if prepared_source_after.stat().st_size != inputs.source_video_size_bytes:
            raise BoundTexturePhotoError(
                "prepared-manifest source size changed during extraction"
            )
        if (
            not os.path.samefile(prepared_source_after, inputs.source_video)
            and _sha256(prepared_source_after) != inputs.source_video_sha256
        ):
            raise BoundTexturePhotoError(
                "prepared-manifest source bytes changed during extraction"
            )
        if _sha256(inputs.native_mapping_path) != inputs.native_mapping_sha256:
            raise BoundTexturePhotoError(
                "native timestamp mapping changed during extraction"
            )
        if _sha256(inputs.capture_import_path) != inputs.capture_import_sha256:
            raise BoundTexturePhotoError(
                "native capture import changed during extraction"
            )

        manifest = {
            "schema": OUTPUT_SCHEMA,
            "status": "verified_all_bound_views",
            "prepared_manifest": {
                "path": str(inputs.prepared_manifest_path),
                "sha256": inputs.prepared_manifest_sha256,
                "schema": inputs.prepared_manifest.get("schema"),
                "frame_count": len(inputs.frames),
            },
            "source_video": {
                "path": str(inputs.source_video),
                "prepared_relative_path": inputs.prepared_manifest["video"]["path"],
                "prepared_resolved_path": str(inputs.prepared_source_video),
                "sha256": inputs.source_video_sha256,
                "size_bytes": inputs.source_video_size_bytes,
                "width": inputs.source_width,
                "height": inputs.source_height,
                "rotation_degrees": 0.0,
                "exact_decoded_frame_count": inputs.native_frame_count,
            },
            "native_frame_mapping": {
                "timestamp_source": TRUSTED_TIMESTAMP_SOURCE,
                "association_method": TRUSTED_ASSOCIATION_METHOD,
                "capture_import_path": str(inputs.capture_import_path),
                "capture_import_sha256": inputs.capture_import_sha256,
                "timestamps_path": str(inputs.native_mapping_path),
                "timestamps_sha256": inputs.native_mapping_sha256,
            },
            "extraction": {
                "decoder": "ffmpeg_cpu",
                "hardware_acceleration": "none",
                "noautorotate": True,
                "maximum_threads": FFMPEG_THREADS,
                "selection": "single_decode_pass_exact_decoded_frame_index",
                "source_frame_indices": source_indices,
                "long_edge_px": long_edge_px,
                "scale_filter": _scale_filter(long_edge_px),
                "output_encoding": "png_lossless_rgb24",
                "prepared_pixels_used_as_output": False,
            },
            "geometry": {
                "source_pixel_geometry": "native_noautorotate_uniform_scale_only",
                "intrinsics_modified": False,
                "poses_modified": False,
                "prepared_manifest_modified": False,
            },
            "verification_policy": _verification_policy(),
            "views": views,
        }
        (stage / OUTPUT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise BoundTexturePhotoError(
                "output-dir appeared during extraction; refusing to overwrite it"
            )
        os.rename(stage, output)
        return output / OUTPUT_MANIFEST_NAME
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract lossless high-resolution texture photos for an existing exact-frame-bound "
            "prepared phone cohort."
        )
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-edge-px", type=int, default=2560)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = extract_bound_texture_photos(
            args.scan_dir,
            args.source_video,
            args.output_dir,
            long_edge_px=args.long_edge_px,
        )
    except BoundTexturePhotoError as exc:
        raise SystemExit(f"bound texture photo extraction failed: {exc}") from exc
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundTexturePhotoError",
    "OUTPUT_MANIFEST_NAME",
    "OUTPUT_SCHEMA",
    "extract_bound_texture_photos",
    "main",
]
