from __future__ import annotations

import hashlib
import json
import math
import subprocess
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
    target_fps: float = 2.0
    max_frames: int = 48
    max_edge_px: int = 1920
    jpeg_quality: int = 94


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


def prepare_video_frames(
    video_path: Path,
    scan_dir: Path,
    settings: FramePreparationSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    progress(0.03, "Inspecting the uploaded video")
    probe = probe_video(video_path)
    duration = probe.get("duration_s")
    extraction_fps = float(settings.target_fps)
    if isinstance(duration, (int, float)) and duration > 0.0:
        extraction_fps = min(extraction_fps, settings.max_frames / float(duration))
    extraction_fps = max(0.1, extraction_fps)

    frames_dir = scan_dir / "frames"
    thumbs_dir = scan_dir / "frame_thumbnails"
    frames_dir.mkdir(parents=True, exist_ok=False)
    thumbs_dir.mkdir(parents=True, exist_ok=False)
    progress(0.08, f"Extracting up to {settings.max_frames} frames")
    video_filter = (
        f"fps={extraction_fps:.8f},"
        f"scale={int(settings.max_edge_px)}:{int(settings.max_edge_px)}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    completed = subprocess.run(
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
            str(int(settings.max_frames)),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(frames_dir / "frame_%04d.jpg"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise FramePreparationError(f"ffmpeg frame extraction failed: {detail or completed.returncode}")
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 2:
        raise FramePreparationError(
            f"MapAnything needs at least two useful views; only {len(frame_paths)} frame was extracted"
        )

    progress(0.55, "Measuring frame quality and building previews")
    frame_rows: list[dict[str, Any]] = []
    previous_small: np.ndarray | None = None
    warning_counts = {"dark": 0, "blurry": 0, "near_duplicate": 0}
    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FramePreparationError(f"failed to decode extracted frame {frame_path.name}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        luminance_p50 = float(np.percentile(gray, 50.0))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
        motion_score = (
            None
            if previous_small is None
            else float(np.mean(cv2.absdiff(small, previous_small)))
        )
        previous_small = small
        warnings: list[str] = []
        if luminance_p50 < 20.0:
            warnings.append("dark")
        if blur_score < 35.0:
            warnings.append("blurry")
        if motion_score is not None and motion_score < 1.5:
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
        thumb_path = thumbs_dir / f"frame_{index:04d}.jpg"
        if not cv2.imwrite(
            str(thumb_path),
            thumb,
            [cv2.IMWRITE_JPEG_QUALITY, 84],
        ):
            raise FramePreparationError(f"failed to write thumbnail {thumb_path.name}")
        frame_rows.append(
            {
                "index": index,
                "timestamp_s": float(index / extraction_fps),
                "frame": str(frame_path.relative_to(scan_dir)),
                "thumbnail": str(thumb_path.relative_to(scan_dir)),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "sha256": _sha256(frame_path),
                "quality": {
                    "luminance_p50": luminance_p50,
                    "blur_score": blur_score,
                    "motion_score": motion_score,
                    "warnings": warnings,
                },
            }
        )
        if index % 8 == 0:
            progress(0.55 + 0.35 * ((index + 1) / len(frame_paths)), "Measuring prepared frames")

    contact_sheet = scan_dir / "prepared_frames_contact_sheet.jpg"
    _contact_sheet(frame_paths, contact_sheet)
    manifest = {
        "schema": "noesis.mapanything.phone_scan.prepared_frames.v1",
        "video": {
            **probe,
            "path": str(video_path.relative_to(scan_dir)),
            "sha256": _sha256(video_path),
        },
        "preparation": {
            "target_fps": float(settings.target_fps),
            "effective_fps": extraction_fps,
            "max_frames": int(settings.max_frames),
            "max_edge_px": int(settings.max_edge_px),
            "frame_count": len(frame_rows),
            "quality_warning_counts": warning_counts,
            "quality_policy": "informational_only_no_automatic_frame_rejection",
        },
        "frames": frame_rows,
    }
    manifest_path = scan_dir / "prepared_frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    progress(1.0, f"Prepared {len(frame_rows)} MapAnything views")
    return {
        "probe": probe,
        "frame_count": len(frame_rows),
        "effective_fps": extraction_fps,
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
