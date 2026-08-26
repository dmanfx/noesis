"""Run one exact batch-3 semantic capture from the canonical room streams."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from . import assets, pipeline, sources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual YOLO26 ADE20K semantic capture")
    parser.add_argument("--size", choices=assets.SUPPORTED_SIZES, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=sources.DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 150:
        raise RuntimeError(f"Expected 150 ADE20K labels in {path}, got {len(labels)}")
    return labels


def _bounded_stop(runtime_pipeline: object, *, completed: bool) -> None:
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
    if args.duration <= 0 or args.warmup_frames <= 0:
        raise SystemExit("duration and warmup frames must be greater than zero")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("alpha must be between zero and one")
    nvinfer_config, resolved = assets.materialize_nvinfer_config(args.size)
    source_cfg = sources.load_three_room_sources(args.pipeline_config)
    context = pipeline.build_pipeline(
        source_cfg,
        nvinfer_config,
        output_dir=args.output_dir,
        model_size=args.size,
        labels=_load_labels(resolved.labels),
        warmup_frames=args.warmup_frames,
        alpha=args.alpha,
        headless=True,
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
