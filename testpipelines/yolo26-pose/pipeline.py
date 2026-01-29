"""DeepStream 8 Service Maker pipeline for YOLO26 pose."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional

from pyservicemaker import Pipeline, Probe

from debug_probes import FrameCounter
from fps_cap import FpsCap
from pose_overlay import PoseOverlay


@dataclass
class PipelineContext:
    pipeline: Pipeline
    config_path: Path


def _build_source_properties(source_cfg: Dict[str, object]) -> Dict[str, object]:
    uris = source_cfg.get("uris", []) or []
    sensor_ids = source_cfg.get("sensor_ids", []) or []
    sensor_names = source_cfg.get("sensor_names", []) or []
    if len(uris) != 3:
        raise RuntimeError(f"Expected 3 URIs for batch=3, got {len(uris)}")
    width = int(source_cfg.get("width", 1920) or 1920)
    height = int(source_cfg.get("height", 1080) or 1080)
    uri_strings = [str(u) for u in uris]
    file_sources = all(u.lower().startswith("file:") for u in uri_strings)
    return {
        "uri-list": ",".join(str(u) for u in uris),
        "sensor-id-list": ",".join(str(s) for s in sensor_ids),
        "sensor-name-list": ",".join(str(s) for s in sensor_names),
        "port": "0",
        "max-batch-size": len(uris),
        "width": width,
        "height": height,
        "live-source": 0 if file_sources else 1,
        "batched-push-timeout": 40000,
        "enable-padding": 1,
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
        "file-loop": 1 if file_sources else 0,
    }


def build_pipeline(
    source_cfg: Dict[str, object],
    nvinfer_config: Path,
    *,
    headless: bool = False,
    tiler_width: Optional[int] = None,
    tiler_height: Optional[int] = None,
) -> PipelineContext:
    source_props = _build_source_properties(source_cfg)
    width = int(source_cfg.get("width", 1920) or 1920)
    height = int(source_cfg.get("height", 1080) or 1080)
    columns = len(source_cfg.get("uris", []) or [])
    tiler_width = int(tiler_width) if tiler_width else width * max(1, columns)
    tiler_height = int(tiler_height) if tiler_height else height

    pipeline = Pipeline("yolo26-pose")
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
    pipeline.add("nvinfer", "pose_infer", {"config-file-path": str(nvinfer_config)})
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
            "rows": 1,
            "columns": max(1, columns),
            "width": int(tiler_width),
            "height": int(tiler_height),
        },
    )
    pipeline.add(
        "nvdsosd",
        "osd",
        {
            "process-mode": 0,
            "display-mask": 0,
            "display-bbox": 0,
            "display-text": 0,
        },
    )

    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        uri_strings = [str(u) for u in source_cfg.get("uris", []) or []]
        file_sources = all(u.lower().startswith("file:") for u in uri_strings)
        sink_props = {"sync": 0} if file_sources else {}
        pipeline.add("nveglglessink", "sink", sink_props)

    pipeline.link("source", "pre_queue", "pose_infer", "post_queue", "tiler", "osd", "sink")

    debug = os.environ.get("YOLO26_POSE_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if file_sources:
        cap_value = os.environ.get("YOLO26_POSE_FPS_CAP", "30").strip()
        if cap_value:
            cap_fps = float(cap_value)
            if cap_fps > 0:
                pipeline.attach("pose_infer", Probe("fps_cap", FpsCap(cap_fps)))

    if debug:
        pipeline.attach("pose_infer", Probe("pose_frames", FrameCounter("pose_infer")))

    if not os.environ.get("YOLO26_POSE_DISABLE_OVERLAY"):
        overlay = PoseOverlay(gie_id=1, model_size=(640, 640))
        pipeline.attach("pose_infer", Probe("pose_overlay", overlay))
    else:
        pipeline["osd"].set(
            {
                "process-mode": 0,
                "display-mask": 0,
                "display-bbox": 0,
                "display-text": 0,
            }
        )

    return PipelineContext(pipeline=pipeline, config_path=nvinfer_config)
