#!/usr/bin/env python3
"""Run the adaptive DA3 carrier over an immutable connector view manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DA3_SOURCE_ROOT = REPO_ROOT / "external" / "Depth-Anything-3" / "src"
if str(DA3_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(DA3_SOURCE_ROOT))
os.environ["PYTHONPATH"] = (
    str(DA3_SOURCE_ROOT)
    + (f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else "")
)

from tools.mapanything_phone_scan.da3_inference import (  # noqa: E402
    DA3PhoneScanSettings,
)
from tools.mapanything_phone_scan.windowed_da3_inference import (  # noqa: E402
    run_windowed_da3_phone_scan,
)


class ConnectorDA3Error(RuntimeError):
    """Raised when the connector view carrier is malformed."""


def _prepared(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in {
        "noesis.pcf.connector.prepared_views.v1",
        "noesis.pcf.connector.prepared_views.v2",
    }:
        raise ConnectorDA3Error(f"{path} is not a connector prepared manifest")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ConnectorDA3Error("connector prepared manifest has too few views")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric-engine",
        type=Path,
        default=Path(
            "data/ds9_artifacts/models/engines/"
            "da3metric_large_294x518_b3_fp16_trt10.16.engine"
        ),
    )
    parser.add_argument("--model-id", default="depth-anything/DA3-BASE")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--point-budget", type=int, default=600_000)
    parser.add_argument("--max-joint-views", type=int, default=48)
    parser.add_argument("--window-overlap-views", type=int, default=16)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    prepared = _prepared(args.prepared_manifest.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ConnectorDA3Error(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    settings = DA3PhoneScanSettings(
        model_id=args.model_id,
        device=args.device,
        process_res=args.process_res,
        point_budget=args.point_budget,
        local_files_only=not args.allow_download,
        metric_engine_path=args.metric_engine.resolve(),
        max_joint_views=args.max_joint_views,
        window_overlap_views=args.window_overlap_views,
        anchor_image=None,
    )

    def progress(fraction: float, message: str) -> None:
        print(f"[{float(fraction):.3f}] {message}", flush=True)

    result = run_windowed_da3_phone_scan(
        args.scan_dir.resolve(), output_dir, prepared, settings, progress
    )
    print(
        json.dumps(
            {
                "view_count": result.get("view_count"),
                "window_count": result.get("window_count"),
                "elapsed_s": result.get("elapsed_s"),
                "coordinate_frame": result.get("coordinate_frame"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
