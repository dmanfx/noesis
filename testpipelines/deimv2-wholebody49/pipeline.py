"""DeepStream 8 Service Maker pipeline for DEIMv2 Wholebody49."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pyservicemaker import Pipeline, Probe

try:
    from .wholebody_overlay import Deimv2WholebodyOverlay
except ImportError:  # pragma: no cover - script execution path
    from wholebody_overlay import Deimv2WholebodyOverlay


@dataclass
class PipelineContext:
    pipeline: Pipeline
    config_path: Path
    overlay: Deimv2WholebodyOverlay


def _build_source_properties(source_cfg: Dict[str, object]) -> Dict[str, object]:
    uris = [str(u) for u in (source_cfg.get("uris", []) or [])]
    sensor_ids = [str(s) for s in (source_cfg.get("sensor_ids", []) or [])]
    sensor_names = [str(s) for s in (source_cfg.get("sensor_names", []) or [])]
    if len(uris) != 3:
        raise RuntimeError(f"DEIMv2 prototype expects exactly 3 RTSP URIs, got {len(uris)}")
    non_rtsp = [uri for uri in uris if not uri.lower().startswith("rtsp://")]
    if non_rtsp:
        raise RuntimeError(f"DEIMv2 prototype requires RTSP sources; non-RTSP entries: {non_rtsp}")

    width = int(source_cfg.get("width", 1920) or 1920)
    height = int(source_cfg.get("height", 1080) or 1080)
    return {
        "uri-list": ",".join(uris),
        "sensor-id-list": ",".join(sensor_ids),
        "sensor-name-list": ",".join(sensor_names),
        "port": "0",
        "max-batch-size": len(uris),
        "width": width,
        "height": height,
        "live-source": 1,
        "batched-push-timeout": 40000,
        "enable-padding": 0,
        "nvbuf-memory-type": 0,
        "sync-inputs": 0,
        "drop-pipeline-eos": 1,
        "cache-buffer": 0,
        "sort-batch": 0,
        "align-first-buffer": 0,
        "latency": 100,
        "rtsp-reconnect-interval": 10,
        "init-rtsp-reconnect-interval": 5,
        "rtsp-reconnect-attempts": 4,
        "select-rtp-protocol": 4,
        "cudadec-memtype": 0,
        "sensorID-padID-mapping": 0,
        "file-loop": 0,
    }


def build_pipeline(
    source_cfg: Dict[str, object],
    nvinfer_config: Path,
    model_info: Dict[str, object],
    *,
    output_root: Path,
    headless: bool = False,
    tiler_width: Optional[int] = None,
    tiler_height: Optional[int] = None,
    emit_every_frames: int = 15,
    score_threshold: float = 0.50,
    attribute_score_threshold: float = 0.75,
    keypoint_score_threshold: Optional[float] = None,
    has_instance_masks: bool = True,
    class_aware_filtering: bool = True,
    enable_smoothing: bool = True,
    smoothing_alpha: float = 0.65,
    reuse_last_on_missing_tensor: bool = False,
    max_draw: int = 300,
) -> PipelineContext:
    source_props = _build_source_properties(source_cfg)
    uris = [str(u) for u in (source_cfg.get("uris", []) or [])]
    columns = len(uris)
    rows = int(math.ceil(len(uris) / float(columns)))
    source_width = int(source_cfg.get("width", 1920) or 1920)
    source_height = int(source_cfg.get("height", 1080) or 1080)
    # Keep the visible mosaic screen-sized by default: 3 columns in one 1920x1080
    # output, matching the canonical DS8 mosaic shape.
    tiler_width = int(tiler_width) if tiler_width else source_width
    tiler_height = int(tiler_height) if tiler_height else source_height

    overlay = Deimv2WholebodyOverlay(
        output_root=output_root,
        model_info=model_info,
        sensor_ids=[str(x) for x in (source_cfg.get("sensor_ids", []) or [])],
        sensor_names=[str(x) for x in (source_cfg.get("sensor_names", []) or [])],
        object_score_threshold=score_threshold,
        attribute_score_threshold=attribute_score_threshold,
        keypoint_threshold=score_threshold if keypoint_score_threshold is None else keypoint_score_threshold,
        emit_every_frames=emit_every_frames,
        max_draw=max_draw,
        has_instance_masks=has_instance_masks,
        class_aware_filtering=class_aware_filtering,
        enable_smoothing=enable_smoothing,
        smoothing_alpha=smoothing_alpha,
        reuse_last_on_missing_tensor=reuse_last_on_missing_tensor,
    )

    pipeline = Pipeline("deimv2-wholebody49")
    pipeline.add("nvmultiurisrcbin", "source", source_props)
    pipeline.add(
        "queue",
        "pre_queue",
        {
            "leaky": 2,
            "max-size-buffers": 1,
            "max-size-bytes": 0,
            "max-size-time": 0,
        },
    )
    pipeline.add("nvinfer", "deimv2_infer", {"config-file-path": str(nvinfer_config)})
    pipeline.add(
        "queue",
        "post_queue",
        {
            "leaky": 2,
            "max-size-buffers": 1,
            "max-size-bytes": 0,
            "max-size-time": 0,
        },
    )
    pipeline.add(
        "nvmultistreamtiler",
        "tiler",
        {
            "rows": rows,
            "columns": columns,
            "width": int(tiler_width),
            "height": int(tiler_height),
        },
    )
    pipeline.add(
        "nvdsosd",
        "osd",
        {
            "process-mode": 0,
            "display-mask": 1 if has_instance_masks else 0,
            "display-bbox": 1,
            "display-text": 1,
        },
    )
    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        pipeline.add("nveglglessink", "sink", {"sync": 0, "qos": 0})

    pipeline.link("source", "pre_queue", "deimv2_infer", "post_queue", "tiler", "osd", "sink")
    pipeline.attach("deimv2_infer", Probe("deimv2_wholebody49_overlay", overlay))
    return PipelineContext(pipeline=pipeline, config_path=nvinfer_config, overlay=overlay)
