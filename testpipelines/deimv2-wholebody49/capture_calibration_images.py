"""Capture representative RTSP frames for DEIMv2 Wholebody49 INT8 calibration."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2

try:
    from . import sources
except ImportError:  # pragma: no cover - script execution path
    import sources


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "deimv2_wholebody49" / "int8_calibration"


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "source"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture DEIMv2 Wholebody49 INT8 calibration images")
    parser.add_argument(
        "--sources",
        type=Path,
        default=sources.DEFAULT_SOURCES_PATH,
        help="Path to sources.yaml (generated from config/infer.yaml if missing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for calibration JPGs",
    )
    parser.add_argument(
        "--frames-per-source",
        type=int,
        default=32,
        help="Representative frames to capture from each RTSP source",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.35,
        help="Seconds to wait between saved frames from the same source",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Frames to discard after opening each source",
    )
    parser.add_argument(
        "--source-timeout",
        type=float,
        default=45.0,
        help="Maximum seconds to spend on each source before failing",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPG quality for saved calibration frames",
    )
    return parser.parse_args()


def _open_capture(uri: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open calibration source")
    return capture


def _capture_source(
    *,
    uri: str,
    sensor_name: str,
    sensor_id: str,
    output_dir: Path,
    frames_per_source: int,
    sample_interval: float,
    warmup_frames: int,
    source_timeout: float,
    jpeg_quality: int,
) -> List[Path]:
    capture = _open_capture(uri)
    saved: List[Path] = []
    source_dir = output_dir / f"{sensor_id}_{_safe_name(sensor_name)}"
    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(max(0, int(warmup_frames))):
            capture.read()
        deadline = time.monotonic() + max(1.0, float(source_timeout))
        while len(saved) < frames_per_source and time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            output_path = source_dir / f"calib_{len(saved):04d}.jpg"
            if not cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
                raise RuntimeError(f"Failed to write calibration frame: {output_path}")
            saved.append(output_path)
            time.sleep(max(0.0, float(sample_interval)))
    finally:
        capture.release()
    if len(saved) != frames_per_source:
        raise RuntimeError(
            f"Captured {len(saved)}/{frames_per_source} calibration frames for {sensor_name}; "
            f"increase --source-timeout or check the RTSP source"
        )
    return saved


def main() -> None:
    args = _parse_args()
    if args.frames_per_source <= 0:
        raise SystemExit("--frames-per-source must be > 0")
    if not args.sources.exists():
        sources.generate_sources_yaml_from_infer_config(output_path=args.sources, force=True)
    source_payload = sources.load_sources_yaml(args.sources)
    source_cfg = sources.select_sources(source_payload, required_count=3, require_rtsp=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_saved: List[Path] = []
    for uri, sensor_id, sensor_name in zip(
        source_cfg["uris"],
        source_cfg["sensor_ids"],
        source_cfg["sensor_names"],
    ):
        saved = _capture_source(
            uri=str(uri),
            sensor_name=str(sensor_name),
            sensor_id=str(sensor_id),
            output_dir=output_dir,
            frames_per_source=int(args.frames_per_source),
            sample_interval=float(args.sample_interval),
            warmup_frames=int(args.warmup_frames),
            source_timeout=float(args.source_timeout),
            jpeg_quality=int(args.jpeg_quality),
        )
        all_saved.extend(saved)
        print(f"DEIMV2_INT8_CALIBRATION_CAPTURE source={sensor_name!r} frames={len(saved)}", flush=True)

    print(f"DEIMV2_INT8_CALIBRATION_IMAGES dir={output_dir} frames={len(all_saved)}", flush=True)


if __name__ == "__main__":
    main()
