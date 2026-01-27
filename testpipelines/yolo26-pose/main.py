"""Entry point for the YOLO26 pose test pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

import pipeline
import sources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS8 YOLO26 pose test pipeline")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera name from sources.yaml (duplicates to fill batch=3)",
    )
    parser.add_argument(
        "-msize",
        dest="msize",
        choices=("n", "s", "m"),
        default="n",
        help="Model size: n, s, or m",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=sources.DEFAULT_SOURCES_PATH,
        help="Path to sources.yaml",
    )
    parser.add_argument("--headless", action="store_true", help="Use fakesink instead of display")
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
        help="Tiler output width (0 = auto from source config)",
    )
    parser.add_argument(
        "--tiler-height",
        type=int,
        default=0,
        help="Tiler output height (0 = auto from source config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable pose debug logging",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_nvinfer_config(msize: str) -> Path:
    root = _repo_root()
    onnx_path = root / f"models/yolo26{msize}-pose.onnx"
    engine_path = root / f"models/engines/yolo26{msize}-pose_b3_fp16.engine"
    labels_path = Path(__file__).resolve().parent / "labels.txt"
    template_path = Path(__file__).resolve().parent / "config_infer_yolo26_pose.template.ini"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found: {template_path}")

    output_dir = Path(__file__).resolve().parent / "build"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"config_infer_yolo26_pose_{msize}.ini"

    text = template_path.read_text()
    text = text.replace("@ONNX_PATH@", str(onnx_path.resolve()))
    text = text.replace("@ENGINE_PATH@", str(engine_path.resolve()))
    text = text.replace("@LABELS_PATH@", str(labels_path.resolve()))
    output_path.write_text(text)
    return output_path


def main() -> None:
    args = _parse_args()
    if args.debug:
        os.environ["YOLO26_POSE_DEBUG"] = "1"
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            root.addHandler(handler)
        logging.info("YOLO26 pose debug logging enabled")
        if os.environ.get("YOLO26_POSE_PYDS"):
            logging.info("YOLO26 pose pyds fallback enabled")

    source_payload = sources.load_sources_yaml(args.sources)
    source_cfg = sources.select_sources(
        source_payload,
        camera=args.camera,
        required_count=3,
    )

    nvinfer_config = _build_nvinfer_config(args.msize)

    ctx = pipeline.build_pipeline(
        source_cfg,
        nvinfer_config,
        headless=args.headless,
        tiler_width=args.tiler_width or None,
        tiler_height=args.tiler_height or None,
    )

    timer = None
    try:
        ctx.pipeline.start()
        if args.duration and args.duration > 0:
            timer = threading.Timer(args.duration, ctx.pipeline.stop)
            timer.start()
        ctx.pipeline.wait()
    finally:
        if timer is not None:
            timer.cancel()


if __name__ == "__main__":
    main()
