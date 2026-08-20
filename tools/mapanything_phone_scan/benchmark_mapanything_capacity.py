from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _parse_counts(raw: str) -> list[int]:
    counts = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not counts or counts[0] < 2:
        raise argparse.ArgumentTypeError("counts must contain integers of at least 2")
    return counts


def _spread_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    return sorted(
        {int(value) for value in np.linspace(0, total - 1, count, dtype=np.int64)}
    )


def _tensor_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    valid_fractions: list[float] = []
    confidence_medians: list[float] = []
    for prediction in predictions:
        mask = prediction.get("mask")
        confidence = prediction.get("conf")
        if isinstance(mask, torch.Tensor):
            mask = mask.detach()
            valid_fractions.append(float(mask.float().mean().cpu()))
        if isinstance(confidence, torch.Tensor):
            values = confidence.detach()
            if isinstance(mask, torch.Tensor) and mask.shape == values.shape:
                values = values[mask.bool()]
            values = values[torch.isfinite(values)]
            if values.numel():
                confidence_medians.append(float(torch.median(values).cpu()))
    return {
        "valid_fraction_p50": (
            float(np.median(valid_fractions)) if valid_fractions else None
        ),
        "confidence_p50_of_views": (
            float(np.median(confidence_medians)) if confidence_medians else None
        ),
    }


def benchmark(
    manifest_path: Path,
    output_path: Path,
    counts: list[int],
    *,
    model_id: str,
    device_name: str,
    amp_dtype: str,
    local_files_only: bool,
) -> dict[str, Any]:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_rows = manifest.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) < 2:
        raise ValueError("prepared manifest has fewer than two frames")
    scan_dir = manifest_path.parent
    frame_paths = [scan_dir / str(row["frame"]) for row in frame_rows]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared frames are missing: {missing[:3]}")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_name} is unavailable")
    model = MapAnything.from_pretrained(
        model_id, local_files_only=local_files_only
    ).to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for requested_count in counts:
            indices = _spread_indices(len(frame_paths), requested_count)
            selected_paths = [str(frame_paths[index]) for index in indices]
            views = load_images(selected_paths)
            predictions: Any = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    predictions = model.infer(
                        views,
                        memory_efficient_inference=True,
                        minibatch_size=1,
                        use_amp=True,
                        amp_dtype=amp_dtype,
                        apply_mask=True,
                        mask_edges=True,
                        apply_confidence_mask=False,
                        confidence_percentile=10,
                        use_multiview_confidence=False,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row = {
                    "requested_view_count": requested_count,
                    "view_count": len(indices),
                    "status": "passed",
                    "joint_inference_s": float(time.monotonic() - started),
                    "selected_indices": indices,
                    "selected_timestamp_s": [
                        frame_rows[index].get("timestamp_s") for index in indices
                    ],
                    **_tensor_summary(predictions),
                }
                if device.type == "cuda":
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    row.update(
                        {
                            "cuda_peak_allocated_bytes": int(
                                torch.cuda.max_memory_allocated(device)
                            ),
                            "cuda_peak_reserved_bytes": int(
                                torch.cuda.max_memory_reserved(device)
                            ),
                            "cuda_free_after_inference_bytes": int(free_bytes),
                            "cuda_total_bytes": int(total_bytes),
                        }
                    )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            except torch.OutOfMemoryError as exc:
                row = {
                    "requested_view_count": requested_count,
                    "view_count": len(indices),
                    "status": "cuda_out_of_memory",
                    "joint_inference_s": float(time.monotonic() - started),
                    "error": str(exc),
                }
                if device.type == "cuda":
                    row["cuda_peak_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated(device)
                    )
                    row["cuda_peak_reserved_bytes"] = int(
                        torch.cuda.max_memory_reserved(device)
                    )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                break
            finally:
                predictions = None
                views = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        model.to("cpu")
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema": "noesis.mapanything.phone_scan.capacity_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prepared_manifest": str(manifest_path.resolve()),
        "available_view_count": len(frame_paths),
        "model": {
            "id": model_id,
            "device": device_name,
            "amp_dtype": amp_dtype,
            "memory_efficient_inference": True,
            "minibatch_size": 1,
        },
        "runs": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure safe joint-view capacity for the installed MapAnything model."
    )
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--counts", type=_parse_counts, default=_parse_counts("64,96,128,160"))
    parser.add_argument("--model-id", default="facebook/map-anything-apache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    report = benchmark(
        args.prepared_manifest,
        args.output,
        args.counts,
        model_id=args.model_id,
        device_name=args.device,
        amp_dtype=args.amp_dtype,
        local_files_only=not args.allow_network,
    )
    return 0 if all(row["status"] == "passed" for row in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
