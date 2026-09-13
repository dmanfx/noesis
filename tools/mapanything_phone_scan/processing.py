from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .prepared_frame_identity import prepared_frame_identity
from .browser_capture import BROWSER_CAPTURE_SCHEMA
from .phone_calibration import calibration_for_video, rectify_selected_frames
from .imu_motion import load_native_motion


ProgressCallback = Callable[[float, str], None]


class FramePreparationError(RuntimeError):
    """Raised when a phone video cannot produce a trustworthy frame set."""


@dataclass(frozen=True)
class FramePreparationSettings:
    candidate_fps: float = 4.0
    max_candidate_frames: int = 1200
    max_selected_frames: int = 256
    candidate_edge_px: int = 1280
    feature_edge_px: int = 640
    min_keyframe_interval_s: float = 0.40
    max_keyframe_interval_s: float = 1.25
    max_edge_px: int = 1920
    jpeg_quality: int = 94
    phone_camera_calibration: Path | None = None
    phone_camera_capture_mode: str = "unbound"


@dataclass
class _Candidate:
    index: int
    timestamp_s: float
    path: Path
    width: int
    height: int
    luminance_p10: float
    luminance_p50: float
    luminance_p90: float
    blur_score: float
    feature_count: int
    feature_coverage: float
    small_gray: np.ndarray
    keypoints: list[Any]
    descriptors: np.ndarray | None
    content_digest: bytes = b""
    capture_time_ns: int | None = None
    source_frame_index: int | None = None
    timestamp_source: str | None = None
    quality_score: float = 0.0
    imu_motion: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_json(command: list[str], *, description: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise FramePreparationError(f"{description} failed: {detail or completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FramePreparationError(f"{description} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise FramePreparationError(f"{description} returned an invalid payload")
    return payload


def probe_video(video_path: Path) -> dict[str, Any]:
    payload = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation:format=duration,size,format_name",
            "-of",
            "json",
            str(video_path),
        ],
        description="ffprobe",
    )
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise FramePreparationError("the upload contains no readable video stream")
    stream = streams[0]
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    def finite_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0.0 else None

    duration = finite_float(stream.get("duration")) or finite_float(fmt.get("duration"))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise FramePreparationError("the video reports an invalid frame size")
    rate_text = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    try:
        source_fps = float(Fraction(rate_text))
    except (ValueError, ZeroDivisionError):
        source_fps = 0.0
    rotation = float((stream.get("tags") or {}).get("rotate") or 0.0)
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = float(side_data["rotation"])
    if not math.isfinite(rotation):
        raise FramePreparationError("the video reports invalid image rotation")
    return {
        "codec": str(stream.get("codec_name") or "unknown"),
        "width": width,
        "height": height,
        "rotation_degrees": rotation,
        "source_fps": source_fps if math.isfinite(source_fps) else 0.0,
        "duration_s": duration,
        "reported_frame_count": int(stream.get("nb_frames") or 0)
        if str(stream.get("nb_frames") or "").isdigit()
        else None,
        "container": str(fmt.get("format_name") or "unknown"),
        "size_bytes": int(fmt.get("size") or video_path.stat().st_size),
    }


def _contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    samples = frame_paths
    if len(samples) > 24:
        indices = np.linspace(0, len(samples) - 1, 24, dtype=np.int64)
        samples = [samples[int(index)] for index in indices]
    tile_w, tile_h = 320, 180
    cols = 4
    rows = int(math.ceil(len(samples) / cols))
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 18, dtype=np.uint8)
    for index, path in enumerate(samples):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(tile_w / float(image.shape[1]), tile_h / float(image.shape[0]))
        resized = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        y = (index // cols) * tile_h + (tile_h - resized.shape[0]) // 2
        x = (index % cols) * tile_w + (tile_w - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise FramePreparationError("failed to write the prepared-frame contact sheet")


def _run_ffmpeg(command: list[str], *, description: str) -> None:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise FramePreparationError(f"{description} failed: {detail or completed.returncode}")


def _scale_filter(edge_px: int) -> str:
    return (
        f"scale={int(edge_px)}:{int(edge_px)}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


_MAX_ENCODED_PTS_FRAMES = 1_000_000


def _probe_encoded_frame_timestamps(video_path: Path) -> list[float]:
    """Return the encoded video PTS values without manufacturing a cadence.

    The regular browser upload has no trustworthy camera acquisition clock, but
    its encoded PTS still describe which recorded frames exist.  Keeping those
    values lets a long encoder freeze remain a real temporal gap after repeat
    removal.  The bound keeps this preparation-side probe finite and matches the
    native capture import limit.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"%+#{_MAX_ENCODED_PTS_FRAMES + 1}",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    with tempfile.TemporaryFile() as stderr_sink:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_sink,
            text=True,
        )
        timestamps: list[float] = []
        previous = -math.inf
        assert process.stdout is not None

        def abort_probe() -> None:
            if process.poll() is None:
                process.kill()
            process.communicate()

        try:
            for index, line in enumerate(process.stdout):
                if index >= _MAX_ENCODED_PTS_FRAMES:
                    abort_probe()
                    raise FramePreparationError(
                        "the video has too many encoded frame timestamps "
                        f"(>{_MAX_ENCODED_PTS_FRAMES})"
                    )
                value = line.strip().rstrip(",")
                if not value:
                    continue
                try:
                    timestamp = float(value)
                except ValueError as exc:
                    abort_probe()
                    raise FramePreparationError(
                        f"encoded frame timestamp {index} is missing or invalid"
                    ) from exc
                if not math.isfinite(timestamp) or timestamp < previous:
                    abort_probe()
                    raise FramePreparationError("encoded frame timestamps are not monotonic")
                timestamps.append(timestamp)
                previous = timestamp
            return_code = process.wait()
            stderr_sink.flush()
            stderr_sink.seek(0, 2)
            stderr_sink.seek(max(0, stderr_sink.tell() - 2000))
            detail = stderr_sink.read().decode("utf-8", errors="replace").strip()
            if return_code != 0:
                raise FramePreparationError(
                    f"ffprobe encoded frame timestamp extraction failed: "
                    f"{detail or return_code}"
                )
            if not timestamps:
                raise FramePreparationError("the video has no encoded frame timestamps")
            return timestamps
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


def _select_timestamp_indices(
    timestamps_s: list[float], candidate_fps: float, max_frames: int
) -> list[int]:
    """Choose existing encoded frames nearest the requested cadence."""

    if len(timestamps_s) < 2:
        raise FramePreparationError("encoded timestamps produced too few candidate frames")
    selected: list[int] = []
    cursor = 0
    start = timestamps_s[0]
    for output_index in range(int(max_frames)):
        target = start + output_index / max(candidate_fps, 1e-6)
        if target > timestamps_s[-1]:
            break
        while (
            cursor + 1 < len(timestamps_s)
            and abs(timestamps_s[cursor + 1] - target)
            <= abs(timestamps_s[cursor] - target)
        ):
            cursor += 1
        if not selected or selected[-1] != cursor:
            selected.append(cursor)
    if len(selected) < 2:
        raise FramePreparationError("encoded timestamps produced too few candidate frames")
    return selected


def _extract_candidates(
    video_path: Path,
    candidate_dir: Path,
    candidate_fps: float,
    settings: FramePreparationSettings,
    source_capture_timestamps_ns: list[int] | None = None,
) -> list[Path]:
    selected_edge_px = min(settings.candidate_edge_px, settings.max_edge_px)
    selected_indices: list[int] = []
    encoded_timestamps_s: list[float] | None = None
    if source_capture_timestamps_ns:
        if len(source_capture_timestamps_ns) < 2 or any(
            source_capture_timestamps_ns[index] >= source_capture_timestamps_ns[index + 1]
            for index in range(len(source_capture_timestamps_ns) - 1)
        ):
            raise FramePreparationError("source capture timestamps are not strictly increasing")
        cursor = 0
        start_ns = source_capture_timestamps_ns[0]
        for output_index in range(int(settings.max_candidate_frames)):
            target_ns = start_ns + int(round(output_index * 1e9 / max(candidate_fps, 1e-6)))
            if target_ns > source_capture_timestamps_ns[-1]:
                break
            while cursor + 1 < len(source_capture_timestamps_ns) and abs(source_capture_timestamps_ns[cursor + 1] - target_ns) <= abs(source_capture_timestamps_ns[cursor] - target_ns):
                cursor += 1
            if not selected_indices or selected_indices[-1] != cursor:
                selected_indices.append(cursor)
        if len(selected_indices) < 2:
            raise FramePreparationError("source capture timestamps produced too few candidate frames")
        selection = "+".join(f"eq(n\\,{index})" for index in selected_indices)
        video_filter = f"select='{selection}',{_scale_filter(selected_edge_px)},showinfo"
    else:
        encoded_timestamps_s = _probe_encoded_frame_timestamps(video_path)
        selected_indices = _select_timestamp_indices(
            encoded_timestamps_s,
            candidate_fps,
            int(settings.max_candidate_frames),
        )
        selection = "+".join(f"eq(n\\,{index})" for index in selected_indices)
        video_filter = f"select='{selection}',{_scale_filter(selected_edge_px)},showinfo"
    input_flags = ["-noautorotate"] if source_capture_timestamps_ns else []
    command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "info",
            "-nostdin",
            *input_flags,
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            video_filter,
            "-frames:v",
            str(int(settings.max_candidate_frames)),
            "-q:v",
            "3",
            "-fps_mode",
            "vfr",
            "-start_number",
            "0",
            str(candidate_dir / "candidate_%05d.jpg"),
        ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise FramePreparationError(
            f"ffmpeg candidate extraction failed: {detail or completed.returncode}"
        )
    # showinfo is attached to this exact extraction filter chain.  Re-probing
    # the input and guessing which VFR source frame the fps filter selected is
    # not sufficient for acquisition-time provenance.
    matches = re.findall(r"showinfo.*?\bn:\s*(\d+).*?\bpts_time:\s*([-+]?\d+(?:\.\d+)?)", completed.stderr)
    timestamps_s = [float(value) for _, value in sorted(matches, key=lambda row: int(row[0]))]
    paths = sorted(candidate_dir.glob("candidate_*.jpg"))
    if len(timestamps_s) != len(paths):
        raise FramePreparationError(
            "ffmpeg candidate extraction did not return one showinfo timestamp per decoded candidate"
        )
    if encoded_timestamps_s is not None:
        # `select` must preserve the source-frame order and the PTS spacing.
        # Compare intervals so a decoder-wide origin shift does not matter,
        # while still failing closed if the filter produced a different frame
        # sequence than the ffprobe mapping used to build it.
        observed_origin = timestamps_s[0]
        expected_origin = encoded_timestamps_s[selected_indices[0]]
        if any(
            abs(
                (observed - observed_origin)
                - (encoded_timestamps_s[source_index] - expected_origin)
            )
            > 0.025
            for observed, source_index in zip(timestamps_s, selected_indices, strict=True)
        ):
            raise FramePreparationError(
                "ffmpeg selected-frame timestamps disagree with the encoded PTS mapping"
            )
    if source_capture_timestamps_ns:
        selected_timestamps_ns = [source_capture_timestamps_ns[index] for index in selected_indices]
        if len(selected_timestamps_ns) != len(paths):
            raise FramePreparationError("explicit source-frame selection did not preserve frame identity")
        (candidate_dir / "candidate_timestamps_ns.json").write_text(
            json.dumps(selected_timestamps_ns, separators=(",", ":")), encoding="utf-8"
        )
        (candidate_dir / "candidate_source_indices.json").write_text(
            json.dumps(selected_indices, separators=(",", ":")), encoding="utf-8"
        )
    else:
        if encoded_timestamps_s is None:
            raise FramePreparationError("encoded timestamp selection was not initialized")
        selected_timestamps_s = [encoded_timestamps_s[index] for index in selected_indices]
        (candidate_dir / "candidate_timestamps_s.json").write_text(
            json.dumps(selected_timestamps_s, separators=(",", ":")), encoding="utf-8"
        )
    return paths


def _feature_coverage(keypoints: list[Any], width: int, height: int) -> float:
    if not keypoints or width <= 0 or height <= 0:
        return 0.0
    cols, rows = 6, 4
    occupied: set[tuple[int, int]] = set()
    for keypoint in keypoints:
        x, y = keypoint.pt
        occupied.add(
            (
                min(cols - 1, max(0, int(x * cols / width))),
                min(rows - 1, max(0, int(y * rows / height))),
            )
        )
    return float(len(occupied) / (cols * rows))


def _analyze_candidates(
    paths: list[Path],
    candidate_fps: float,
    feature_edge_px: int,
    progress: ProgressCallback,
    capture_timestamps_ns: list[int] | None = None,
    source_frame_indices: list[int] | None = None,
    encoded_timestamps_s: list[float] | None = None,
    timestamp_source: str | None = None,
) -> list[_Candidate]:
    detector = cv2.ORB_create(
        nfeatures=1600,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        fastThreshold=10,
    )
    candidates: list[_Candidate] = []
    capture_time_rows: list[int | None] = None
    if capture_timestamps_ns:
        if len(capture_timestamps_ns) != len(paths):
            raise FramePreparationError(
                "exact candidate timestamp count does not match extracted candidate frames"
            )
        capture_time_rows = [int(value) for value in capture_timestamps_ns]
    if source_frame_indices is not None and len(source_frame_indices) != len(paths):
        raise FramePreparationError(
            "source frame identity count does not match extracted candidate frames"
        )
    if encoded_timestamps_s is not None:
        if len(encoded_timestamps_s) != len(paths):
            raise FramePreparationError(
                "encoded candidate timestamp count does not match extracted candidate frames"
            )
        if any(
            encoded_timestamps_s[index] > encoded_timestamps_s[index + 1]
            for index in range(len(encoded_timestamps_s) - 1)
        ):
            raise FramePreparationError("encoded candidate timestamps are not monotonic")
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FramePreparationError(f"failed to decode candidate frame {path.name}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, feature_edge_px / float(max(gray.shape)))
        feature_gray = (
            gray
            if scale >= 0.999
            else cv2.resize(
                gray,
                (max(1, int(round(gray.shape[1] * scale))), max(1, int(round(gray.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        )
        keypoints, descriptors = detector.detectAndCompute(feature_gray, None)
        keypoints = list(keypoints or [])
        small_gray = cv2.resize(feature_gray, (160, 90), interpolation=cv2.INTER_AREA)
        candidates.append(
            _Candidate(
                index=index,
                timestamp_s=(
                    float((capture_time_rows[index] - capture_time_rows[0]) * 1e-9)
                        if capture_time_rows and capture_time_rows[index] is not None
                    else float(encoded_timestamps_s[index])
                        if encoded_timestamps_s is not None
                    else float(index / candidate_fps)
                ),
                path=path,
                width=int(feature_gray.shape[1]),
                height=int(feature_gray.shape[0]),
                luminance_p10=float(np.percentile(feature_gray, 10.0)),
                luminance_p50=float(np.percentile(feature_gray, 50.0)),
                luminance_p90=float(np.percentile(feature_gray, 90.0)),
                blur_score=float(cv2.Laplacian(feature_gray, cv2.CV_64F).var()),
                feature_count=len(keypoints),
                feature_coverage=_feature_coverage(
                    keypoints, feature_gray.shape[1], feature_gray.shape[0]
                ),
                small_gray=small_gray,
                keypoints=keypoints,
                descriptors=descriptors,
                content_digest=hashlib.blake2s(
                    image.tobytes(),
                    digest_size=16,
                ).digest(),
                capture_time_ns=(capture_time_rows[index] if capture_time_rows else None),
                source_frame_index=(source_frame_indices[index] if source_frame_indices is not None else None),
                timestamp_source=(
                    timestamp_source
                    if capture_time_rows is not None or encoded_timestamps_s is not None
                    else None
                ),
            )
        )
        if index % 16 == 0:
            progress(
                0.20 + 0.30 * ((index + 1) / max(1, len(paths))),
                f"Analyzing candidate view {index + 1} of {len(paths)}",
            )

    blur_values = np.log1p(np.asarray([row.blur_score for row in candidates]))
    blur_low, blur_high = np.percentile(blur_values, (10.0, 90.0))
    blur_span = max(float(blur_high - blur_low), 1e-6)
    for row, log_blur in zip(candidates, blur_values, strict=True):
        sharpness = float(np.clip((log_blur - blur_low) / blur_span, 0.0, 1.0))
        feature_strength = 0.55 * min(1.0, row.feature_count / 700.0) + 0.45 * row.feature_coverage
        midtone = max(0.0, 1.0 - abs(row.luminance_p50 - 112.0) / 112.0)
        dynamic_range = min(1.0, max(0.0, row.luminance_p90 - row.luminance_p10) / 120.0)
        exposure = 0.65 * midtone + 0.35 * dynamic_range
        row.quality_score = float(
            np.clip(0.42 * sharpness + 0.38 * feature_strength + 0.20 * exposure, 0.0, 1.0)
        )
    return candidates


def _point_coverage(points: np.ndarray, width: int, height: int) -> float:
    if points.size == 0 or width <= 0 or height <= 0:
        return 0.0
    cols, rows = 6, 4
    x = np.clip((points[:, 0] * cols / width).astype(np.int32), 0, cols - 1)
    y = np.clip((points[:, 1] * rows / height).astype(np.int32), 0, rows - 1)
    return float(np.unique(y * cols + x).size / (rows * cols))


def _visual_edge(source: _Candidate, target: _Candidate) -> dict[str, Any]:
    appearance_delta = float(np.mean(cv2.absdiff(source.small_gray, target.small_gray)))
    empty = {
        "match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "source_coverage": 0.0,
        "target_coverage": 0.0,
        "median_displacement_norm": 0.0,
        "appearance_delta": appearance_delta,
        "passes_connectivity": False,
    }
    if source.descriptors is None or target.descriptors is None:
        return empty
    if len(source.descriptors) < 2 or len(target.descriptors) < 2:
        return empty
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        source.descriptors, target.descriptors, k=2
    )
    good = [first for first, second in matches if first.distance < 0.76 * second.distance]
    if len(good) < 6:
        return {**empty, "match_count": len(good)}
    source_points = np.asarray(
        [source.keypoints[match.queryIdx].pt for match in good], dtype=np.float32
    )
    target_points = np.asarray(
        [target.keypoints[match.trainIdx].pt for match in good], dtype=np.float32
    )
    _, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        3.0,
        maxIters=1500,
        confidence=0.995,
    )
    inlier_mask = (
        np.zeros((len(good),), dtype=bool)
        if mask is None
        else np.asarray(mask).reshape(-1).astype(bool)
    )
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inlier_count / max(1, len(good)))
    if inlier_count:
        source_inliers = source_points[inlier_mask]
        target_inliers = target_points[inlier_mask]
        diagonal = max(math.hypot(source.width, source.height), 1.0)
        displacement = np.linalg.norm(target_inliers - source_inliers, axis=1)
        median_displacement = float(np.median(displacement) / diagonal)
        source_coverage = _point_coverage(source_inliers, source.width, source.height)
        target_coverage = _point_coverage(target_inliers, target.width, target.height)
    else:
        median_displacement = 0.0
        source_coverage = 0.0
        target_coverage = 0.0
    passes = bool(
        (inlier_count >= 12 and inlier_ratio >= 0.12 and min(source_coverage, target_coverage) >= 0.04)
        or (inlier_count >= 28 and min(source_coverage, target_coverage) >= 0.02)
    )
    return {
        "match_count": len(good),
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "source_coverage": source_coverage,
        "target_coverage": target_coverage,
        "median_displacement_norm": median_displacement,
        "appearance_delta": appearance_delta,
        "passes_connectivity": passes,
    }


def _edge_score(edge: dict[str, Any]) -> float:
    return float(
        0.45 * min(1.0, float(edge["inlier_count"]) / 45.0)
        + 0.25 * min(1.0, float(edge["inlier_ratio"]) / 0.55)
        + 0.30
        * min(
            1.0,
            min(float(edge["source_coverage"]), float(edge["target_coverage"])) / 0.25,
        )
    )


_REPEAT_MEAN_DELTA_MAX = 1.50
_REPEAT_P95_DELTA_MAX = 4.0
_REPEAT_CHANGED_FRACTION_MAX = 0.02
_REPEAT_MIN_MATCHES = 20
_REPEAT_MIN_INLIERS = 16
_REPEAT_MIN_INLIER_RATIO = 0.50
_REPEAT_MIN_COVERAGE = 0.08
_REPEAT_MAX_DISPLACEMENT = 0.004


def _is_frozen_repeat(
    anchor: _Candidate,
    candidate: _Candidate,
    visual_edge: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Classify only strong adjacent repeats as safe to discard.

    The comparison is always against the fixed run anchor.  This keeps a
    sequence of individually small camera motions from chasing the anchor and
    disappearing as one apparent freeze.  Exact decoded content is safe even
    for textureless images; an image with encoding noise must also have strong
    spatial support, because missing features do not prove that nothing moved.
    """

    if (
        anchor.content_digest
        and candidate.content_digest
        and anchor.content_digest == candidate.content_digest
    ):
        return True, "exact_content"
    delta = np.abs(
        anchor.small_gray.astype(np.int16) - candidate.small_gray.astype(np.int16)
    ).astype(np.float32)
    mean_delta = float(np.mean(delta))
    if mean_delta > _REPEAT_MEAN_DELTA_MAX:
        return False, "changed"
    p95_delta = float(np.percentile(delta, 95.0))
    changed_fraction = float(np.mean(delta > 4.0))
    if (
        p95_delta > _REPEAT_P95_DELTA_MAX
        or changed_fraction > _REPEAT_CHANGED_FRACTION_MAX
    ):
        return False, "changed"
    edge = visual_edge if visual_edge is not None else _visual_edge(anchor, candidate)
    if (
        int(edge["match_count"]) >= _REPEAT_MIN_MATCHES
        and int(edge["inlier_count"]) >= _REPEAT_MIN_INLIERS
        and float(edge["inlier_ratio"]) >= _REPEAT_MIN_INLIER_RATIO
        and min(float(edge["source_coverage"]), float(edge["target_coverage"]))
        >= _REPEAT_MIN_COVERAGE
        and float(edge["median_displacement_norm"]) <= _REPEAT_MAX_DISPLACEMENT
    ):
        return True, "spatial_repeat"
    return False, "insufficient_spatial_evidence"


def _deduplicate_candidates(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], dict[str, Any]]:
    """Drop bounded adjacent frozen runs before adaptive selection.

    A run's representative may be the sharpest member, but its anchor remains
    the first retained candidate until a real visual change arrives.  This is
    deliberately a single adjacent pass, so its cost is linear in the bounded
    candidate set and it cannot turn a long frozen recording into fabricated
    cadence frames.
    """

    if not candidates:
        return [], {
            "policy": "adjacent_frozen_run_anchor_v1",
            "candidate_count_before": 0,
            "candidate_count_after": 0,
            "dropped_candidate_count": 0,
            "frozen_run_count": 0,
            "exact_content_drop_count": 0,
            "spatial_repeat_drop_count": 0,
            "dropped_candidate_indices": [],
            "frozen_runs": [],
            "frozen_span_s": _distribution([]),
            "frozen_span_total_s": 0.0,
        }

    retained: list[_Candidate] = []
    dropped_indices: list[int] = []
    frozen_runs: list[dict[str, Any]] = []
    exact_drop_count = 0
    spatial_drop_count = 0
    anchor = candidates[0]
    representative = anchor
    last_in_run = anchor
    run_candidates: list[_Candidate] = [anchor]
    dropped_in_run = 0
    run_drop_reasons: dict[str, int] = {"exact_content": 0, "spatial_repeat": 0}

    def flush_run() -> None:
        nonlocal anchor, representative, last_in_run, run_candidates, dropped_in_run
        if dropped_in_run:
            dropped_indices.extend(
                int(candidate.index)
                for candidate in run_candidates
                if candidate is not representative
            )
            start_s = float(anchor.timestamp_s)
            end_s = float(last_in_run.timestamp_s)
            frozen_runs.append(
                {
                    "anchor_candidate_index": int(anchor.index),
                    "representative_candidate_index": int(representative.index),
                    "start_timestamp_s": start_s,
                    "end_timestamp_s": end_s,
                    "span_s": max(0.0, end_s - start_s),
                    "dropped_count": int(dropped_in_run),
                    "drop_reason_counts": dict(run_drop_reasons),
                }
            )
        retained.append(representative)

    for candidate in candidates[1:]:
        duplicate, reason = _is_frozen_repeat(anchor, candidate)
        if duplicate:
            run_candidates.append(candidate)
            dropped_in_run += 1
            last_in_run = candidate
            run_drop_reasons[reason] = run_drop_reasons.get(reason, 0) + 1
            if reason == "exact_content":
                exact_drop_count += 1
            else:
                spatial_drop_count += 1
            if candidate.quality_score > representative.quality_score:
                representative = candidate
            continue
        flush_run()
        anchor = candidate
        representative = candidate
        last_in_run = candidate
        run_candidates = [candidate]
        dropped_in_run = 0
        run_drop_reasons = {"exact_content": 0, "spatial_repeat": 0}
    flush_run()

    spans = [float(row["span_s"]) for row in frozen_runs]
    summary = {
        "policy": "adjacent_frozen_run_anchor_v1",
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(retained),
        "dropped_candidate_count": len(dropped_indices),
        "frozen_run_count": len(frozen_runs),
        "exact_content_drop_count": exact_drop_count,
        "spatial_repeat_drop_count": spatial_drop_count,
        "dropped_candidate_indices": dropped_indices,
        "frozen_runs": frozen_runs,
        "frozen_span_s": _distribution(spans),
        "frozen_span_total_s": float(sum(spans)),
    }
    return retained, summary


def _select_keyframes(
    candidates: list[_Candidate], settings: FramePreparationSettings
) -> tuple[
    list[int],
    dict[int, str],
    dict[tuple[int, int], dict[str, Any]],
    bool,
    int,
]:
    if len(candidates) < 2:
        raise FramePreparationError("MapAnything needs at least two candidate views")
    edge_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def edge(left: int, right: int) -> dict[str, Any]:
        key = (left, right)
        if key not in edge_cache:
            edge_cache[key] = _visual_edge(candidates[left], candidates[right])
        return edge_cache[key]

    quality_floor = float(np.percentile([row.quality_score for row in candidates], 12.0))
    bootstrap_end = 0
    bootstrap_start_s = float(candidates[0].timestamp_s)
    for candidate_index, candidate in enumerate(candidates):
        if candidate.timestamp_s - bootstrap_start_s > 0.50:
            break
        bootstrap_end = candidate_index
    first = max(range(bootstrap_end + 1), key=lambda index: candidates[index].quality_score)
    selected = [first]
    reasons = {first: "bootstrap_quality"}

    def append_selected(candidate_index: int, reason: str) -> bool:
        """Append only a candidate that is not a frozen repeat of the anchor."""

        if selected:
            previous = selected[-1]
            duplicate, _ = _is_frozen_repeat(
                candidates[previous],
                candidates[candidate_index],
                edge(previous, candidate_index),
            )
            if duplicate:
                return False
        selected.append(candidate_index)
        reasons[candidate_index] = reason
        return True

    index = first + 1
    while index < len(candidates):
        last = selected[-1]
        elapsed = candidates[index].timestamp_s - candidates[last].timestamp_s
        if elapsed < settings.min_keyframe_interval_s:
            index += 1
            continue
        link = edge(last, index)
        novelty = bool(
            float(link["median_displacement_norm"]) >= 0.020
            or float(link["appearance_delta"]) >= 6.0
        )
        quality_ok = candidates[index].quality_score >= quality_floor
        must_cover = elapsed >= settings.max_keyframe_interval_s
        lost_overlap = not bool(link["passes_connectivity"])

        if lost_overlap and index - 1 > last:
            bridge_pool = list(range(max(last + 1, index - 3), index))
            bridge_pool = [
                candidate_index
                for candidate_index in bridge_pool
                if candidates[candidate_index].timestamp_s - candidates[last].timestamp_s
                >= settings.min_keyframe_interval_s * 0.70
            ]
            if bridge_pool:
                bridge = max(
                    bridge_pool,
                    key=lambda candidate_index: (
                        bool(edge(last, candidate_index)["passes_connectivity"]),
                        _edge_score(edge(last, candidate_index)),
                        candidates[candidate_index].quality_score,
                        candidate_index,
                    ),
                )
                if bridge > last:
                    if append_selected(bridge, "connectivity_bridge"):
                        continue

        if (novelty and quality_ok) or must_cover:
            reason = "viewpoint_change" if novelty and quality_ok else "coverage_interval"
            append_selected(index, reason)
        index += 1

    last_candidate = len(candidates) - 1
    if (
        last_candidate > selected[-1]
        and candidates[last_candidate].timestamp_s - candidates[selected[-1]].timestamp_s
        >= settings.min_keyframe_interval_s * 0.70
    ):
        append_selected(last_candidate, "walk_endpoint")

    # Repair weak selected-to-selected links with dense temporal bridge candidates.
    repair_index = 0
    while repair_index < len(selected) - 1:
        left, right = selected[repair_index], selected[repair_index + 1]
        link = edge(left, right)
        if bool(link["passes_connectivity"]) or right - left <= 1:
            repair_index += 1
            continue
        pool = list(range(left + 1, right))
        if not pool:
            repair_index += 1
            continue
        midpoint = (candidates[left].timestamp_s + candidates[right].timestamp_s) * 0.5
        valid_pool: list[int] = []
        for candidate_index in pool:
            left_repeat, _ = _is_frozen_repeat(
                candidates[left], candidates[candidate_index], edge(left, candidate_index)
            )
            right_repeat, _ = _is_frozen_repeat(
                candidates[candidate_index], candidates[right], edge(candidate_index, right)
            )
            if not left_repeat and not right_repeat:
                valid_pool.append(candidate_index)
        if not valid_pool:
            repair_index += 1
            continue
        bridge = max(
            valid_pool,
            key=lambda candidate_index: (
                min(
                    _edge_score(edge(left, candidate_index)),
                    _edge_score(edge(candidate_index, right)),
                ),
                candidates[candidate_index].quality_score,
                -abs(candidates[candidate_index].timestamp_s - midpoint),
            ),
        )
        selected.insert(repair_index + 1, bridge)
        reasons[bridge] = "connectivity_repair"

    pre_limit_selected_count = len(selected)
    selection_limited = False
    while len(selected) > settings.max_selected_frames:
        removable: list[tuple[float, int]] = []
        for position in range(1, len(selected) - 1):
            left, current, right = selected[position - 1 : position + 2]
            elapsed = candidates[right].timestamp_s - candidates[left].timestamp_s
            replacement = edge(left, right)
            if elapsed > settings.max_keyframe_interval_s * 1.35:
                continue
            if not bool(replacement["passes_connectivity"]):
                continue
            if _is_frozen_repeat(
                candidates[left], candidates[right], replacement
            )[0]:
                continue
            current_value = (
                0.50 * candidates[current].quality_score
                + 0.25 * _edge_score(edge(left, current))
                + 0.25 * _edge_score(edge(current, right))
            )
            removable.append((current_value, position))
        if not removable:
            raise FramePreparationError(
                "adaptive selection found more connected keyframes than the emergency "
                f"limit of {settings.max_selected_frames}; raise "
                "NOESIS_PHONE_SCAN_MAX_SELECTED_FRAMES for this unusually long walk"
            )
        _, position = min(removable)
        del selected[position]
        selection_limited = True

    return selected, reasons, edge_cache, selection_limited, pre_limit_selected_count


def _extract_selected_frames(
    frames_dir: Path,
    selected_candidates: list[_Candidate],
) -> list[Path]:
    paths: list[Path] = []
    for index, candidate in enumerate(selected_candidates):
        output_path = frames_dir / f"frame_{index:04d}.jpg"
        shutil.copy2(candidate.path, output_path)
        paths.append(output_path)
    return paths


def _distribution(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"min": None, "p50": None, "p90": None, "max": None}
    return {
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def prepare_video_frames(
    video_path: Path,
    scan_dir: Path,
    settings: FramePreparationSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    progress(0.03, "Inspecting the uploaded video")
    probe = probe_video(video_path)
    duration = probe.get("duration_s")

    capture_timestamps_ns: list[int] | None = None
    capture_report_path = scan_dir / "capture" / "capture_import.json"
    capture_report_schema = ""
    capture_report: dict[str, Any] = {}
    use_native_acquisition_mapping = True
    if capture_report_path.is_file():
        try:
            capture_report = json.loads(capture_report_path.read_text(encoding="utf-8"))
            if isinstance(capture_report, dict):
                capture_report_schema = str(capture_report.get("schema") or "")
                android_timing = capture_report.get("android_capture")
                if isinstance(android_timing, dict):
                    use_native_acquisition_mapping = android_timing.get("camera_acquisition_timestamp_verified") is True
                capture_video = capture_report.get("video")
                if not isinstance(capture_video, dict):
                    capture_video = {}
                probe_timestamp_source = str(capture_video.get("timestamp_source") or "")
                # Browser MediaRecorder WebM can omit container duration. The
                # importer has already validated this encoded duration from
                # ffprobe frame timestamps, so use it for preparation metrics
                # while retaining the source in the prepared manifest.
                if (
                    capture_report_schema == BROWSER_CAPTURE_SCHEMA
                    and (duration is None or duration <= 0.0)
                ):
                    encoded_duration = capture_video.get("encoded_duration_s")
                    try:
                        encoded_duration = float(encoded_duration)
                    except (TypeError, ValueError):
                        encoded_duration = None
                    if (
                        encoded_duration is not None
                        and math.isfinite(encoded_duration)
                        and encoded_duration > 0.0
                    ):
                        duration = encoded_duration
                        probe["duration_s"] = encoded_duration
                        probe["duration_source"] = "capture_import.video.encoded_duration_s"
            else:
                probe_timestamp_source = ""
            # Browser callback/media observations are deliberately not source
            # acquisition timestamps.  Keep normal encoded-PTS extraction for
            # those captures. Android companion bundles use acquisition times
            # only after the importer verifies their original encoded mapping.
            if capture_report_schema != BROWSER_CAPTURE_SCHEMA and use_native_acquisition_mapping:
                timestamp_path = scan_dir / "capture" / "video_timestamps_ns.json"
                raw_timestamps = json.loads(timestamp_path.read_text(encoding="utf-8"))
                if isinstance(raw_timestamps, list) and len(raw_timestamps) >= 2:
                    capture_timestamps_ns = [int(value) for value in raw_timestamps]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise FramePreparationError(
                f"sensor capture timestamp mapping is unreadable: {exc}"
            ) from exc
    else:
        probe_timestamp_source = ""

    capture_mode = (
        "browser" if capture_report_schema == BROWSER_CAPTURE_SCHEMA
        else "native_sensor_bundle" if capture_report_schema
        else "uploaded_video"
    )
    motion, motion_summary = load_native_motion(
        scan_dir / "capture", capture_report if isinstance(capture_report, dict) else {}
    )
    phone_calibration, phone_calibration_summary = calibration_for_video(
        settings.phone_camera_calibration,
        configured_capture_mode=settings.phone_camera_capture_mode,
        capture_mode=capture_mode,
        video=probe,
    )

    candidate_fps = float(settings.candidate_fps)
    if isinstance(duration, (int, float)) and duration > 0.0:
        candidate_fps = min(candidate_fps, settings.max_candidate_frames / float(duration))
    candidate_fps = max(0.5, candidate_fps)

    frames_dir = scan_dir / "frames"
    thumbs_dir = scan_dir / "frame_thumbnails"
    frames_dir.mkdir(parents=True, exist_ok=False)
    thumbs_dir.mkdir(parents=True, exist_ok=False)
    progress(0.08, f"Extracting dense candidates at {candidate_fps:.2f} views/second")
    with tempfile.TemporaryDirectory(prefix=".frame_candidates-", dir=scan_dir) as raw_temp:
        candidate_dir = Path(raw_temp)
        candidate_paths = _extract_candidates(
            video_path,
            candidate_dir,
            candidate_fps,
            settings,
            capture_timestamps_ns,
        )
        exact_candidate_timestamps_ns: list[int] | None = None
        exact_candidate_source_indices: list[int] | None = None
        encoded_candidate_timestamps_s: list[float] | None = None
        if capture_report_path.is_file() and capture_report_schema != BROWSER_CAPTURE_SCHEMA and use_native_acquisition_mapping:
            try:
                if capture_timestamps_ns:
                    exact_candidate_timestamps_ns = [
                        int(value)
                        for value in json.loads(
                            (candidate_dir / "candidate_timestamps_ns.json").read_text(encoding="utf-8")
                        )
                    ]
                    exact_candidate_source_indices = [
                        int(value)
                        for value in json.loads(
                            (candidate_dir / "candidate_source_indices.json").read_text(encoding="utf-8")
                        )
                    ]
                else:
                    exact_timestamps_s = json.loads(
                        (candidate_dir / "candidate_timestamps_s.json").read_text(encoding="utf-8")
                    )
                    capture_epoch_ns = int(
                        (json.loads(capture_report_path.read_text(encoding="utf-8"))
                         .get("manifest", {})
                         .get("clocks", {})
                         .get("camera_start_time_ns") or 0)
                    )
                    exact_candidate_timestamps_ns = [
                        int(round(float(value) * 1e9)) + capture_epoch_ns
                        for value in exact_timestamps_s
                    ]
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise FramePreparationError(
                    f"exact sensor timestamp mapping is unreadable: {exc}"
                ) from exc
        elif not capture_timestamps_ns:
            try:
                encoded_candidate_timestamps_s = [
                    float(value)
                    for value in json.loads(
                        (candidate_dir / "candidate_timestamps_s.json").read_text(
                            encoding="utf-8"
                        )
                    )
                ]
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise FramePreparationError(
                    f"encoded candidate timestamp mapping is unreadable: {exc}"
                ) from exc
        if len(candidate_paths) < 2:
            raise FramePreparationError(
                f"MapAnything needs at least two useful views; only {len(candidate_paths)} candidate was extracted"
            )
        progress(0.20, f"Scoring {len(candidate_paths)} candidate views")
        candidates = _analyze_candidates(
            candidate_paths,
            candidate_fps,
            settings.feature_edge_px,
            progress,
            exact_candidate_timestamps_ns,
            exact_candidate_source_indices,
            encoded_candidate_timestamps_s,
            "encoded_pts" if encoded_candidate_timestamps_s is not None else probe_timestamp_source,
        )
        raw_candidate_count = len(candidates)
        if motion is not None:
            for candidate in candidates:
                candidate.imu_motion = motion.frame(candidate.capture_time_ns)
                if candidate.imu_motion["status"] == "available":
                    candidate.imu_motion["visual_quality_score_before_motion"] = candidate.quality_score
                    candidate.quality_score = max(
                        0.0, candidate.quality_score - candidate.imu_motion["visual_quality_penalty"]
                    )
            motion_summary["candidate_count_with_motion"] = sum(
                row.imu_motion.get("status") == "available" for row in candidates
            )
        candidates, repeat_summary = _deduplicate_candidates(candidates)
        if len(candidates) < 2:
            raise FramePreparationError(
                "MapAnything needs at least two useful views after removing "
                f"{repeat_summary['dropped_candidate_count']} frozen repeat candidates"
            )
        progress(0.52, "Selecting informative views and repairing overlap gaps")
        (
            selected,
            reasons,
            edge_cache,
            selection_limited,
            pre_limit_selected_count,
        ) = _select_keyframes(candidates, settings)
        progress(0.58, f"Retaining {len(selected)} selected reconstruction views")
        frame_paths = _extract_selected_frames(
            frames_dir, [candidates[index] for index in selected]
        )
        calibration_rows = (
            rectify_selected_frames(frame_paths, phone_calibration, phone_calibration_summary)
            if phone_calibration is not None and phone_calibration_summary is not None
            else [{} for _ in frame_paths]
        )
        if phone_calibration is not None:
            # Keep the exact imported profile with this preparation. Its original
            # archive/source fingerprints remain available after service changes.
            (scan_dir / "phone_camera_calibration.json").write_text(
                json.dumps(phone_calibration, indent=2) + "\n", encoding="utf-8"
            )

        progress(0.76, "Building selected-view previews and provenance")
        frame_rows: list[dict[str, Any]] = []
        warning_counts = {"dark": 0, "blurry": 0, "near_duplicate": 0}
        selected_blur = np.asarray([candidates[index].blur_score for index in selected])
        blur_warning_floor = float(np.percentile(selected_blur, 8.0))
        adjacent_edges: list[dict[str, Any]] = []
        for output_index, (candidate_index, frame_path) in enumerate(
            zip(selected, frame_paths, strict=True)
        ):
            candidate = candidates[candidate_index]
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FramePreparationError(f"failed to decode extracted frame {frame_path.name}")
            previous_edge = None
            if output_index > 0:
                key = (selected[output_index - 1], candidate_index)
                if key not in edge_cache:
                    edge_cache[key] = _visual_edge(
                        candidates[selected[output_index - 1]], candidate
                    )
                previous_edge = dict(edge_cache[key])
                adjacent_edges.append(previous_edge)
            warnings: list[str] = []
            if candidate.luminance_p50 < 20.0:
                warnings.append("dark")
            if candidate.blur_score <= blur_warning_floor and candidate.quality_score < 0.35:
                warnings.append("blurry")
            if (
                previous_edge is not None
                and float(previous_edge["appearance_delta"]) < 1.5
                and float(previous_edge["median_displacement_norm"]) < 0.006
            ):
                warnings.append("near_duplicate")
            for warning in warnings:
                warning_counts[warning] += 1

            thumb = cv2.resize(
                image,
                (
                    320,
                    max(1, int(round(image.shape[0] * (320.0 / image.shape[1])))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            thumb_path = thumbs_dir / f"frame_{output_index:04d}.jpg"
            if not cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 84]):
                raise FramePreparationError(f"failed to write thumbnail {thumb_path.name}")
            frame_sha256 = _sha256(frame_path)
            if calibration_rows[output_index]:
                source_hash = calibration_rows[output_index]["calibration_processing"]["source_frame_sha256"]
                calibration_rows[output_index]["calibration_processing"]["source_frame_id"] = prepared_frame_identity(
                    output_index, source_hash
                )
            frame_rows.append(
                {
                    "index": output_index,
                    "frame_id": prepared_frame_identity(output_index, frame_sha256),
                    "candidate_index": int(candidate.index),
                    "timestamp_s": candidate.timestamp_s,
                    "capture_time_ns": candidate.capture_time_ns,
                    "source_frame_index": candidate.source_frame_index,
                    "timestamp_source": (
                        candidate.timestamp_source
                        or ("derived_candidate_period" if candidate.capture_time_ns is None else probe_timestamp_source)
                    ),
                    "selection_reason": reasons.get(candidate_index, "adaptive_selection"),
                    "frame": str(frame_path.relative_to(scan_dir)),
                    "thumbnail": str(thumb_path.relative_to(scan_dir)),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "sha256": frame_sha256,
                    **({"imu_motion": candidate.imu_motion} if candidate.imu_motion else {}),
                    **calibration_rows[output_index],
                    "quality": {
                        "luminance_p10": candidate.luminance_p10,
                        "luminance_p50": candidate.luminance_p50,
                        "luminance_p90": candidate.luminance_p90,
                        "blur_score": candidate.blur_score,
                        "feature_count": candidate.feature_count,
                        "feature_coverage": candidate.feature_coverage,
                        "quality_score": candidate.quality_score,
                        "edge_from_previous": previous_edge,
                        "warnings": warnings,
                    },
                }
            )
            if output_index % 12 == 0:
                progress(
                    0.76 + 0.18 * ((output_index + 1) / len(frame_paths)),
                    f"Saving selected view {output_index + 1} of {len(frame_paths)}",
                )

    contact_sheet = scan_dir / "prepared_frames_contact_sheet.jpg"
    _contact_sheet(frame_paths, contact_sheet)
    intervals = [
        frame_rows[index]["timestamp_s"] - frame_rows[index - 1]["timestamp_s"]
        for index in range(1, len(frame_rows))
    ]
    connectivity_fraction = float(
        np.mean([bool(edge["passes_connectivity"]) for edge in adjacent_edges])
    ) if adjacent_edges else 0.0
    effective_fps = (
        len(frame_rows) / float(duration)
        if isinstance(duration, (int, float)) and duration > 0.0
        else candidate_fps
    )
    selection_summary = {
        "policy": "adaptive_quality_motion_overlap_connectivity_v1",
        "candidate_fps": candidate_fps,
        "candidate_count": raw_candidate_count,
        "candidate_count_after_repeat_dedup": len(candidates),
        "repeat_dedup": repeat_summary,
        "selected_count": len(frame_rows),
        "pre_limit_selected_count": pre_limit_selected_count,
        "rejected_candidate_count": raw_candidate_count - len(frame_rows),
        "emergency_selected_limit": int(settings.max_selected_frames),
        "selection_limited": selection_limited,
        "min_keyframe_interval_s": float(settings.min_keyframe_interval_s),
        "max_keyframe_interval_s": float(settings.max_keyframe_interval_s),
        "selected_interval_s": _distribution(intervals),
        "selected_quality_score": _distribution(
            [float(row["quality"]["quality_score"]) for row in frame_rows]
        ),
        "adjacent_visual_inliers": _distribution(
            [float(edge["inlier_count"]) for edge in adjacent_edges]
        ),
        "adjacent_connectivity_pass_fraction": connectivity_fraction,
        "reason_counts": {
            reason: sum(1 for row in frame_rows if row["selection_reason"] == reason)
            for reason in sorted({str(row["selection_reason"]) for row in frame_rows})
        },
    }
    manifest = {
        "schema": "noesis.mapanything.phone_scan.prepared_frames.v2",
        "video": {
            **probe,
            "path": str(video_path.relative_to(scan_dir)),
            "sha256": _sha256(video_path),
        },
        "preparation": {
            "strategy": "adaptive_keyframe_selection_v1",
            "candidate_fps": candidate_fps,
            "effective_fps": effective_fps,
            "max_candidate_frames": int(settings.max_candidate_frames),
            "max_selected_frames": int(settings.max_selected_frames),
            "candidate_edge_px": int(min(settings.candidate_edge_px, settings.max_edge_px)),
            "feature_edge_px": int(settings.feature_edge_px),
            "max_edge_px": int(settings.max_edge_px),
            "frame_count": len(frame_rows),
            "quality_warning_counts": warning_counts,
            "quality_policy": "adaptive_selection_with_relative_quality_and_connectivity_repair",
            "selection": selection_summary,
        },
        "frames": frame_rows,
        "imu_motion": motion_summary,
        **({"camera_calibration": phone_calibration_summary} if phone_calibration_summary else {}),
    }
    manifest_path = scan_dir / "prepared_frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    progress(1.0, f"Prepared {len(frame_rows)} adaptive reconstruction views")
    return {
        "probe": probe,
        "frame_count": len(frame_rows),
        "candidate_count": raw_candidate_count,
        "candidate_count_after_repeat_dedup": len(candidates),
        "effective_fps": effective_fps,
        "selection": selection_summary,
        "quality_warning_counts": warning_counts,
        "frames": frame_rows,
        "contact_sheet": str(contact_sheet.relative_to(scan_dir)),
        "manifest": str(manifest_path.relative_to(scan_dir)),
        "imu_motion": motion_summary,
        **({"camera_calibration": phone_calibration_summary} if phone_calibration_summary else {}),
    }


__all__ = [
    "FramePreparationError",
    "FramePreparationSettings",
    "prepare_video_frames",
    "probe_video",
]
