"""Entry point for the YOLO26 seg + depth DS8 prototype."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model_setup
import pipeline
import prototype_calibration
import sources


def _parse_depth_input_shape(value: str) -> tuple[int, int]:
    text = str(value or "").strip().lower()
    if "x" not in text:
        raise argparse.ArgumentTypeError(f"Depth input shape must be WIDTHxHEIGHT (got: {value})")
    width_text, height_text = text.split("x", 1)
    try:
        width = int(width_text.strip())
        height = int(height_text.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Depth input shape must be WIDTHxHEIGHT (got: {value})") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"Depth input shape must be positive (got: {value})")
    # Depth Anything V2 metric vits uses a DINOv2 patch embed that requires both
    # spatial dimensions to be multiples of 14.
    if (width % 14) != 0 or (height % 14) != 0:
        raise argparse.ArgumentTypeError(
            f"Depth input shape must use dimensions divisible by 14 for DAv2 (got: {value})"
        )
    return width, height


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS8 YOLO26 seg + depth prototype")
    parser.add_argument(
        "--camera",
        type=str,
        default=sources.DEFAULT_CAMERA,
        help="Camera name from sources.yaml. Default: Family Room Camera.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=sources.DEFAULT_SOURCES_PATH,
        help="Path to sources.yaml.",
    )
    parser.add_argument(
        "--infer-config",
        type=Path,
        default=sources.DEFAULT_INFER_CONFIG_PATH,
        help="Main DS8 infer.yaml used to resolve active/file/stream source URIs.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("stream", "file", "active"),
        default="stream",
        help="Choose the RTSP URI, adjacent file URI, or active URI from infer.yaml. Default: stream.",
    )
    parser.add_argument(
        "--depth-preference",
        choices=("auto", "da2", "da3"),
        default="da2",
        help="Depth model preference. Default: da2. Use auto/da3 only when explicitly testing DA3 setup.",
    )
    parser.add_argument("--disable-depth", action="store_true", help="Run the same pipeline without the depth model for A/B benchmarking.")
    parser.add_argument(
        "--depth-every-n-frames",
        type=int,
        default=1,
        help="Run depth inference every N frames and reuse the last aligned depth in between. Default: 1.",
    )
    parser.add_argument(
        "--depth-input-shape",
        type=_parse_depth_input_shape,
        default=model_setup.DEFAULT_DEPTH_INPUT_SIZE,
        help="Depth model input shape as WIDTHxHEIGHT; both dimensions must be divisible by 14. Default: 924x518.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1920,
        help="Canonical DS8 frame width used for display and fusion coordinates. Default: 1920.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=1080,
        help="Canonical DS8 frame height used for display and fusion coordinates. Default: 1080.",
    )
    parser.add_argument("--show-bbox", action="store_true", help="Render detection bboxes in addition to masks/text.")
    parser.add_argument("--headless", action="store_true", help="Use fakesink instead of display.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run for N seconds then stop (0 = run until interrupted).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python log level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_payload = sources.load_sources_yaml(args.sources)
    if args.depth_every_n_frames <= 0:
        raise SystemExit("--depth-every-n-frames must be >= 1")
    source_cfg = sources.select_source(
        source_payload,
        camera=args.camera,
        frame_size=(args.frame_width, args.frame_height),
        infer_config_path=args.infer_config,
        source_mode=args.source_mode,
    )
    calibration_resolver = prototype_calibration.PrototypeCalibrationResolver.from_source_config(
        source_cfg,
        frame_size=(args.frame_width, args.frame_height),
    )
    assets = model_setup.ensure_pipeline_assets(
        args.depth_preference,
        enable_depth=not args.disable_depth,
        depth_input_size=args.depth_input_shape,
        depth_every_n_frames=args.depth_every_n_frames,
    )
    ctx = pipeline.build_pipeline(
        source_cfg,
        assets,
        frame_size=(args.frame_width, args.frame_height),
        headless=args.headless,
        show_bbox=args.show_bbox,
        depth_every_n_frames=args.depth_every_n_frames,
        calibration_resolver=calibration_resolver,
    )
    calibration_snapshot = calibration_resolver.snapshot(calibration_resolver.binding.runtime_source_id)
    if calibration_snapshot is None:
        logging.warning(
            "Prototype calibration unavailable: runtime_source=%s calibration_source=%s camera_id=%s extrinsics=%s",
            calibration_resolver.binding.runtime_source_id,
            calibration_resolver.binding.calibration_source_id,
            calibration_resolver.binding.camera_id,
            calibration_resolver.extrinsics_path,
        )
    else:
        logging.info(
            "Prototype calibration bound runtime_source=%s -> calibration_source=%s camera_id=%s image=%sx%s floor_y=%.3f unit_scale=%.3f extrinsics=%s",
            calibration_resolver.binding.runtime_source_id,
            calibration_resolver.binding.calibration_source_id,
            calibration_resolver.binding.camera_id,
            calibration_snapshot.image_size[0],
            calibration_snapshot.image_size[1],
            calibration_snapshot.floor_y,
            calibration_snapshot.unit_scale,
            calibration_resolver.extrinsics_path,
        )

    logging.info(
        "Launching seg+depth prototype: camera=%s source_mode=%s uri=%s frame=%sx%s depth_enabled=%s depth_every_n_frames=%d depth_input=%sx%s show_bbox=%s seg_engine=%s depth_engine=%s depth_model=%s fallback=%s",
        source_cfg["sensor_name"],
        args.source_mode,
        source_cfg["uri"],
        args.frame_width,
        args.frame_height,
        int(not args.disable_depth),
        args.depth_every_n_frames,
        args.depth_input_shape[0],
        args.depth_input_shape[1],
        int(args.show_bbox),
        assets.seg.engine,
        assets.depth.engine if assets.depth is not None else "disabled",
        assets.depth.model_name if assets.depth is not None else "disabled",
        assets.depth.fallback_reason if assets.depth is not None else None,
    )

    timer = None
    wait_thread = None
    wait_done = threading.Event()

    def _wait_pipeline() -> None:
        try:
            ctx.pipeline.wait()
        finally:
            wait_done.set()

    def _stop_pipeline(reason: str) -> None:
        logging.info("Stopping seg+depth prototype (%s)", reason)
        try:
            ctx.pipeline.stop()
        except Exception:
            logging.exception("Failed to stop seg+depth prototype")

    try:
        ctx.pipeline.start()
        if args.duration and args.duration > 0:
            wait_thread = threading.Thread(target=_wait_pipeline, name="SegDepthPipelineWait", daemon=True)
            wait_thread.start()
            timer = threading.Timer(args.duration, lambda: _stop_pipeline(f"duration={args.duration:.1f}s"))
            timer.daemon = True
            timer.start()
            deadline = time.time() + float(args.duration) + 5.0
            while not wait_done.wait(timeout=0.25):
                if time.time() >= deadline:
                    logging.warning("Timed run stop grace expired; exiting launcher while pipeline wait thread drains")
                    break
        else:
            ctx.pipeline.wait()
    except KeyboardInterrupt:
        _stop_pipeline("keyboard interrupt")
    finally:
        if timer is not None:
            timer.cancel()
        logging.info(ctx.runtime_stats.summary_line())


if __name__ == "__main__":
    main()
