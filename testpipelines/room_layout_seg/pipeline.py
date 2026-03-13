"""DeepStream 8 Service Maker pipeline for room-layout semantic segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pyservicemaker import Pipeline, Probe

try:
    from .probes import RoomLayoutEmitter
except ImportError:  # pragma: no cover - script execution path
    from probes import RoomLayoutEmitter


@dataclass
class PipelineContext:
    pipeline: Pipeline
    config_path: Path
    emitter: RoomLayoutEmitter


def _is_live_uri(uri: str) -> bool:
    return uri.lower().startswith("rtsp://")


def _build_source_properties(source_cfg: Dict[str, object]) -> Dict[str, object]:
    uris = [str(u) for u in (source_cfg.get("uris", []) or [])]
    sensor_ids = [str(s) for s in (source_cfg.get("sensor_ids", []) or [])]
    sensor_names = [str(s) for s in (source_cfg.get("sensor_names", []) or [])]
    if not uris:
        raise RuntimeError("No URIs provided to room-layout pipeline")
    width = int(source_cfg.get("width", 1920) or 1920)
    height = int(source_cfg.get("height", 1080) or 1080)
    any_live = any(_is_live_uri(uri) for uri in uris)
    any_file = any(uri.lower().startswith("file://") for uri in uris)
    return {
        "uri-list": ",".join(uris),
        "sensor-id-list": ",".join(sensor_ids),
        "sensor-name-list": ",".join(sensor_names),
        "port": "0",
        "max-batch-size": len(uris),
        "width": width,
        "height": height,
        "live-source": 1 if any_live else 0,
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
    }


def build_pipeline(
    source_cfg: Dict[str, object],
    nvinfer_config: Path,
    model_info: Dict[str, object],
    output_labels,
    *,
    output_root: Path,
    frames_per_package: int = 24,
    emit_every_frames: int = 24,
    loop_files: bool = False,
    headless: bool = False,
    tiler_width: Optional[int] = None,
    tiler_height: Optional[int] = None,
) -> PipelineContext:
    source_props = _build_source_properties(source_cfg)
    uris = [str(u) for u in (source_cfg.get("uris", []) or [])]
    if not uris:
        raise RuntimeError("room-layout pipeline requires at least one URI")
    any_live = any(_is_live_uri(uri) for uri in uris)
    any_file = any(uri.lower().startswith("file://") for uri in uris)
    source_props["file-loop"] = 1 if loop_files and any_file and not any_live else 0
    source_props["drop-pipeline-eos"] = 1 if any_live or source_props["file-loop"] else 0

    seg_width = int(model_info["output"]["width"])
    seg_height = int(model_info["output"]["height"])
    columns = min(max(1, len(uris)), 3)
    rows = int(math.ceil(len(uris) / float(columns)))
    tiler_width = int(tiler_width) if tiler_width else seg_width * columns
    tiler_height = int(tiler_height) if tiler_height else seg_height * rows

    emitter = RoomLayoutEmitter(
        output_root=output_root,
        output_labels=output_labels,
        model_info=model_info,
        sensor_ids=[str(x) for x in (source_cfg.get("sensor_ids", []) or [])],
        sensor_names=[str(x) for x in (source_cfg.get("sensor_names", []) or [])],
        frames_per_package=frames_per_package,
        emit_every_frames=emit_every_frames,
    )

    pipeline = Pipeline("room-layout-seg")
    pipeline.add("nvmultiurisrcbin", "source", source_props)
    pipeline.add("nvinfer", "layout_infer", {"config-file-path": str(nvinfer_config)})
    pipeline.add(
        "nvsegvisual",
        "seg_visual",
        {
            "width": seg_width,
            "height": seg_height,
            "gpu-id": 0,
            "batch-size": len(uris),
            "original-background": True,
            "operate-on-seg-meta-id": int(model_info["inference"]["gie_unique_id"]),
            "class-id": 0,
        },
    )
    pipeline.add(
        "nvdsosd",
        "osd",
        {
            "process-mode": 1,
            "display-text": 1,
            "display-bbox": 0,
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
    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        pipeline.add("nveglglessink", "sink", {"sync": 0, "qos": 0})

    pipeline.link("source", "layout_infer", "seg_visual", "osd", "tiler", "sink")
    pipeline.attach("layout_infer", Probe("room_layout_bundle", emitter))
    return PipelineContext(pipeline=pipeline, config_path=nvinfer_config, emitter=emitter)
