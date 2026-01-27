"""Entry point for the Bidnetpipe segmentation pipeline."""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from . import custom_parser, model_setup, pipeline, sources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS8 BiSeNetV2 segmentation pipeline")
    parser.add_argument("--headless", action="store_true", help="Use fakesink instead of display")
    parser.add_argument(
        "--sources",
        type=Path,
        default=sources.DEFAULT_SOURCES_PATH,
        help="Path to bisenetpipe_sources.yaml",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip model downloads (error if missing)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run for N seconds then stop (0 = run until interrupted)",
    )
    parser.add_argument(
        "--tiler-width",
        type=int,
        default=0,
        help="Tiler output width (0 = auto from model)",
    )
    parser.add_argument(
        "--tiler-height",
        type=int,
        default=0,
        help="Tiler output height (0 = auto from model)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.sources.exists():
        sources.generate_sources_yaml(output_path=args.sources, force=True)

    model_setup.ensure_model_artifacts(allow_download=not args.no_download)
    custom_parser.ensure_custom_parser_lib()
    model_info = model_setup.load_model_info()
    labels = model_setup.load_labels()
    os.environ["BIDNET_FLOOR_CLASS_ID"] = str(labels["floor_class_id"])
    os.environ.setdefault("BIDNET_BG_CLASS_ID", "0")

    heartbeat_event = threading.Event() if args.duration and args.duration > 0 else None
    ctx = pipeline.build_pipeline(
        sources_path=args.sources,
        nvinfer_config=model_setup.NVINFER_CONFIG_PATH,
        model_info=model_info,
        labels=labels,
        headless=args.headless,
        tiler_width=args.tiler_width or None,
        tiler_height=args.tiler_height or None,
        first_heartbeat_event=heartbeat_event,
    )

    timer = None

    try:
        ctx.pipeline.start()
        if args.duration and args.duration > 0:
            if heartbeat_event is not None:
                heartbeat_event.wait(timeout=300.0)
            timer = threading.Timer(args.duration, ctx.pipeline.stop)
            timer.start()
        ctx.pipeline.wait()
    finally:
        if timer is not None:
            timer.cancel()


if __name__ == "__main__":
    main()
