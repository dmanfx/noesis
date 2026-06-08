"""Entry point for the DEIMv2 Wholebody49 DS8 prototype."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path
from typing import List

try:
    from . import model_setup, parser_setup, pipeline, sources
except ImportError:  # pragma: no cover - script execution path
    import model_setup
    import parser_setup
    import pipeline
    import sources


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "deimv2_wholebody49"
DEFAULT_DURATION_STOP_GRACE_S = 5.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS8 DEIMv2 Wholebody49 prototype")
    parser.add_argument(
        "--sources",
        type=Path,
        default=sources.DEFAULT_SOURCES_PATH,
        help="Path to sources.yaml (generated from config/infer.yaml if missing)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Optional camera name to place first; the prototype still runs all three RTSP streams",
    )
    parser.add_argument(
        "--model-variant",
        default="masks",
        help=f"Model variant or alias. Available: {', '.join(model_setup.MODEL_VARIANT_CHOICES)}",
    )
    parser.add_argument("--headless", action="store_true", help="Use fakesink instead of EGL display")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run for N seconds then stop (0 = run until interrupted)",
    )
    parser.add_argument(
        "--duration-stop-grace",
        type=float,
        default=DEFAULT_DURATION_STOP_GRACE_S,
        help="Seconds to wait for DS8 wait() to return after --duration posts EOS",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for emitted Wholebody49 JSON summaries",
    )
    parser.add_argument(
        "--emit-every-frames",
        type=int,
        default=15,
        help="Write latest JSON summaries every N frames per source",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.50,
        help="Score threshold for object output summaries and body masks",
    )
    parser.add_argument(
        "--attribute-score-threshold",
        type=float,
        default=0.75,
        help="Score threshold for generation, gender, head-pose, and side helper boxes",
    )
    parser.add_argument(
        "--keypoint-score-threshold",
        type=float,
        default=None,
        help="Score threshold for keypoint output summaries (defaults to --score-threshold)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.50,
        help="Mask threshold used by nvdsosd for instance-mask rendering",
    )
    parser.add_argument(
        "--infer-interval",
        type=int,
        default=0,
        help="DeepStream nvinfer interval; 1 infers every other frame and reuses cached overlay on skipped frames",
    )
    parser.add_argument(
        "--disable-cached-overlay",
        action="store_true",
        help="Do not reuse the last detections on skipped tensor frames when --infer-interval is nonzero",
    )
    parser.add_argument(
        "--disable-class-aware-filtering",
        action="store_true",
        help="Disable contextual class-aware suppression after tensor decode",
    )
    parser.add_argument(
        "--disable-smoothing",
        action="store_true",
        help="Disable per-source temporal smoothing for overlay boxes and skeleton points",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.65,
        help="EMA weight for current detections during temporal smoothing",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=300,
        help="Maximum detections to draw per frame; JSON summaries still keep decoded detections",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing model artifacts",
    )
    parser.add_argument(
        "--no-engine-build",
        action="store_true",
        help="Do not build a missing or stale TensorRT engine",
    )
    parser.add_argument(
        "--int8-calibration-cache",
        type=Path,
        default=None,
        help="Existing TensorRT INT8 calibration cache for INT8 model variants",
    )
    parser.add_argument(
        "--int8-calibration-images",
        type=Path,
        default=None,
        help="Representative image directory used to build an INT8 calibration cache and engine",
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _wait_for_pipeline(ctx: pipeline.PipelineContext, *, duration_s: float, stop_grace_s: float) -> bool:
    wait_errors: List[BaseException] = []
    stop_timed_out = False

    def _wait() -> None:
        try:
            ctx.pipeline.wait()
        except BaseException as exc:  # pragma: no cover - exercised by live DS8 runtime
            wait_errors.append(exc)

    wait_thread = threading.Thread(target=_wait, name="deimv2_wholebody49_wait", daemon=True)
    wait_thread.start()
    if duration_s and duration_s > 0:
        wait_thread.join(timeout=float(duration_s))
        if wait_thread.is_alive():
            ctx.pipeline.stop()
            wait_thread.join(timeout=max(0.0, float(stop_grace_s)))
            if wait_thread.is_alive():
                stop_timed_out = True
                print(
                    "DEIMV2_WHOLEBODY49_DURATION_STOP_TIMEOUT "
                    f"duration_s={float(duration_s):.3f} stop_grace_s={float(stop_grace_s):.3f}",
                    flush=True,
                )
    else:
        while wait_thread.is_alive():
            wait_thread.join(timeout=0.5)

    if wait_errors:
        raise wait_errors[0]
    return stop_timed_out


def main() -> None:
    args = _parse_args()
    if int(args.infer_interval) < 0:
        raise SystemExit("--infer-interval must be >= 0")
    try:
        model_variant = model_setup.normalize_variant(args.model_variant)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    has_instance_masks = model_setup.variant_has_instance_masks(model_variant)
    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.sources.exists():
        sources.generate_sources_yaml_from_infer_config(output_path=args.sources, force=True)
    source_payload = sources.load_sources_yaml(args.sources)
    source_cfg = sources.select_sources(source_payload, camera=args.camera, required_count=3, require_rtsp=True)

    model_setup.ensure_model_artifacts(
        allow_download=not args.no_download,
        allow_export=True,
        allow_engine_build=not args.no_engine_build,
        variant=model_variant,
        calibration_cache_path=args.int8_calibration_cache,
        calibration_images_dir=args.int8_calibration_images,
    )
    model_info = model_setup.load_model_info(variant=model_variant)
    parser_lib = parser_setup.ensure_custom_parser_lib() if has_instance_masks else None
    nvinfer_config = model_setup.materialize_nvinfer_config(
        batch_size=len(source_cfg.get("uris", []) or []),
        parser_lib_path=parser_lib,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        infer_interval=args.infer_interval,
        variant=model_variant,
    )

    ctx = pipeline.build_pipeline(
        source_cfg,
        nvinfer_config,
        model_info,
        output_root=args.output_root,
        headless=args.headless,
        tiler_width=args.tiler_width or None,
        tiler_height=args.tiler_height or None,
        emit_every_frames=args.emit_every_frames,
        score_threshold=args.score_threshold,
        attribute_score_threshold=args.attribute_score_threshold,
        keypoint_score_threshold=args.keypoint_score_threshold,
        has_instance_masks=has_instance_masks,
        class_aware_filtering=not args.disable_class_aware_filtering,
        enable_smoothing=not args.disable_smoothing,
        smoothing_alpha=args.smoothing_alpha,
        reuse_last_on_missing_tensor=bool(args.infer_interval > 0 and not args.disable_cached_overlay),
        max_draw=args.max_draw,
    )

    force_exit_after_finalize = False
    try:
        ctx.pipeline.start()
        force_exit_after_finalize = _wait_for_pipeline(
            ctx,
            duration_s=args.duration,
            stop_grace_s=args.duration_stop_grace,
        )
    finally:
        for path in ctx.overlay.finalize():
            print(f"DEIMV2_WHOLEBODY49_FINAL summary={path}", flush=True)
        if force_exit_after_finalize:
            print("DEIMV2_WHOLEBODY49_DURATION_FORCE_EXIT", flush=True)
            os._exit(0)


if __name__ == "__main__":
    main()
