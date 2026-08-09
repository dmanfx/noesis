"""Run YOLO26-sem-ADE20K on the three canonical room streams."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import model_setup
import parser_setup
import pipeline
import sources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS8 YOLO26 ADE20K semantic segmentation")
    parser.add_argument("--size", choices=model_setup.SUPPORTED_SIZES, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=sources.DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--duration", type=float, default=30.0, help="Maximum seconds to await all snapshots")
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 150:
        raise RuntimeError(f"Expected 150 ADE20K labels in {path}, got {len(labels)}")
    return labels


def _bounded_stop(runtime_pipeline: object, *, completed: bool) -> None:
    """Allow graceful Service Maker teardown, then bound a stuck C++ wait."""
    exit_code = 0 if completed else 2
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda _sig, _frame: os._exit(exit_code))
    signal.alarm(5)
    try:
        runtime_pipeline.stop()
        runtime_pipeline.wait()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def main() -> None:
    args = _parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than zero")
    if args.warmup_frames <= 0:
        raise SystemExit("--warmup-frames must be greater than zero")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between zero and one")

    parser_library = parser_setup.ensure_parser()
    nvinfer_config, assets = model_setup.materialize_nvinfer_config(args.size, parser_library)
    source_cfg = sources.load_three_room_sources(args.pipeline_config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "build" / "yolo26_sem_ade20k" / "captures" / run_id / f"yolo26{args.size}"
    )
    labels = _load_labels(assets.labels)
    context = pipeline.build_pipeline(
        source_cfg,
        nvinfer_config,
        output_dir=output_dir,
        model_size=args.size,
        labels=labels,
        warmup_frames=args.warmup_frames,
        alpha=args.alpha,
        headless=args.headless,
    )

    started = False
    stopped = False
    completed = False
    summary: dict[str, object] | None = None
    try:
        context.pipeline.start()
        started = True
        completed = context.snapshot.complete_event.wait(timeout=float(args.duration))
        if completed:
            summary = context.snapshot.finalize()
            print("YOLO26_SEM_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
        _bounded_stop(context.pipeline, completed=completed)
        stopped = True
    finally:
        if started and not stopped:
            try:
                _bounded_stop(context.pipeline, completed=False)
            except Exception:
                pass
    if not completed:
        raise RuntimeError(f"Timed out after {args.duration:.1f}s before all three room snapshots arrived")
    if summary is None:
        raise RuntimeError("Semantic capture completed without a summary")


if __name__ == "__main__":
    main()
