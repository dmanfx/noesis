from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


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
    quality_score: float = 0.0


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
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration,size,format_name",
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
    return {
        "codec": str(stream.get("codec_name") or "unknown"),
        "width": width,
        "height": height,
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


def _extract_candidates(
    video_path: Path,
    candidate_dir: Path,
    candidate_fps: float,
    settings: FramePreparationSettings,
) -> list[Path]:
    selected_edge_px = min(settings.candidate_edge_px, settings.max_edge_px)
    video_filter = f"fps={candidate_fps:.8f},{_scale_filter(selected_edge_px)}"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
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
            "-start_number",
            "0",
            str(candidate_dir / "candidate_%05d.jpg"),
        ],
        description="ffmpeg candidate extraction",
    )
    return sorted(candidate_dir.glob("candidate_*.jpg"))


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
) -> list[_Candidate]:
    detector = cv2.ORB_create(
        nfeatures=1600,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        fastThreshold=10,
    )
    candidates: list[_Candidate] = []
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
                timestamp_s=float(index / candidate_fps),
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
    bootstrap_end = min(
        len(candidates) - 1,
        max(0, int(round(0.50 * settings.candidate_fps))),
    )
    first = max(range(bootstrap_end + 1), key=lambda index: candidates[index].quality_score)
    selected = [first]
    reasons = {first: "bootstrap_quality"}

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
                    selected.append(bridge)
                    reasons[bridge] = "connectivity_bridge"
                    continue

        if (novelty and quality_ok) or must_cover:
            reason = "viewpoint_change" if novelty and quality_ok else "coverage_interval"
            selected.append(index)
            reasons[index] = reason
        index += 1

    last_candidate = len(candidates) - 1
    if (
        last_candidate > selected[-1]
        and candidates[last_candidate].timestamp_s - candidates[selected[-1]].timestamp_s
        >= settings.min_keyframe_interval_s * 0.70
    ):
        selected.append(last_candidate)
        reasons[last_candidate] = "walk_endpoint"

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
        bridge = max(
            pool,
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
            video_path, candidate_dir, candidate_fps, settings
        )
        if len(candidate_paths) < 2:
            raise FramePreparationError(
                f"MapAnything needs at least two useful views; only {len(candidate_paths)} candidate was extracted"
            )
        progress(0.20, f"Scoring {len(candidate_paths)} candidate views")
        candidates = _analyze_candidates(
            candidate_paths, candidate_fps, settings.feature_edge_px, progress
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
            frame_rows.append(
                {
                    "index": output_index,
                    "candidate_index": candidate_index,
                    "timestamp_s": candidate.timestamp_s,
                    "selection_reason": reasons.get(candidate_index, "adaptive_selection"),
                    "frame": str(frame_path.relative_to(scan_dir)),
                    "thumbnail": str(thumb_path.relative_to(scan_dir)),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "sha256": _sha256(frame_path),
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
        "candidate_count": len(candidates),
        "selected_count": len(frame_rows),
        "pre_limit_selected_count": pre_limit_selected_count,
        "rejected_candidate_count": len(candidates) - len(frame_rows),
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
    }
    manifest_path = scan_dir / "prepared_frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    progress(1.0, f"Prepared {len(frame_rows)} adaptive reconstruction views")
    return {
        "probe": probe,
        "frame_count": len(frame_rows),
        "candidate_count": len(candidates),
        "effective_fps": effective_fps,
        "selection": selection_summary,
        "quality_warning_counts": warning_counts,
        "frames": frame_rows,
        "contact_sheet": str(contact_sheet.relative_to(scan_dir)),
        "manifest": str(manifest_path.relative_to(scan_dir)),
    }


__all__ = [
    "FramePreparationError",
    "FramePreparationSettings",
    "prepare_video_frames",
    "probe_video",
]
