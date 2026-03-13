"""Entry point for the DS8 room-layout segmentation utility pipeline."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

try:
    from . import model_setup, pipeline, sources
except ImportError:  # pragma: no cover - script execution path
    import model_setup
    import pipeline
    import sources


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "room_layout_masks"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS8 room-layout segmentation utility")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Optional camera name from sources.yaml / config/infer.yaml",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=sources.DEFAULT_SOURCES_PATH,
        help="Path to sources.yaml (generated from config/infer.yaml if missing)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=1,
        help="Number of sources to run when --camera is not provided (0 = all, max 3)",
    )
    parser.add_argument("--headless", action="store_true", help="Use fakesink instead of EGL display")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run for N seconds then stop (0 = until interrupted)",
    )
    parser.add_argument(
        "--frames-per-package",
        type=int,
        default=24,
        help="Minimum frames before the layout bundle is first emitted",
    )
    parser.add_argument(
        "--emit-every-frames",
        type=int,
        default=24,
        help="Rewrite the latest bundle every N additional frames",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for emitted layout bundles",
    )
    parser.add_argument(
        "--tiler-width",
        type=int,
        default=0,
        help="Override tiler output width",
    )
    parser.add_argument(
        "--tiler-height",
        type=int,
        default=0,
        help="Override tiler output height",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing model artifacts",
    )
    parser.add_argument(
        "--loop-files",
        action="store_true",
        help="Loop local file URIs instead of exiting at EOF",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.sources.exists():
        sources.generate_sources_yaml_from_infer_config(output_path=args.sources, force=True)
    source_payload = sources.load_sources_yaml(args.sources)
    max_sources = 3 if args.max_sources == 0 else min(3, max(1, int(args.max_sources)))
    source_cfg = sources.select_sources(
        source_payload,
        camera=args.camera,
        max_sources=max_sources,
    )

    artifacts = model_setup.ensure_model_artifacts(allow_download=not args.no_download)
    model_info = model_setup.load_model_info(artifacts.model_info_path)
    output_labels = model_setup.load_output_labels(artifacts.labels_json_path)
    nvinfer_config = model_setup.materialize_nvinfer_config(
        batch_size=len(source_cfg.get("uris", []) or []),
    )

    ctx = pipeline.build_pipeline(
        source_cfg,
        nvinfer_config,
        model_info,
        output_labels,
        output_root=args.output_root,
        frames_per_package=args.frames_per_package,
        emit_every_frames=args.emit_every_frames,
        loop_files=args.loop_files,
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
        manifests = ctx.emitter.finalize()
        for manifest in manifests:
            print(
                "ROOM_LAYOUT_FINAL "
                f"sensor={manifest['sensor_name']} "
                f"manifest={manifest['manifest_path']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
