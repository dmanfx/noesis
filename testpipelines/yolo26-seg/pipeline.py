"""DeepStream 8 Service Maker pipeline for YOLO26 instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pyservicemaker import Pipeline


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
    return {
        "uri-list": ",".join(str(u) for u in uris),
        "sensor-id-list": ",".join(str(s) for s in sensor_ids),
        "sensor-name-list": ",".join(str(s) for s in sensor_names),
        "port": "0",
        "max-batch-size": len(uris),
        "width": width,
        "height": height,
        "live-source": 1,
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

    pipeline = Pipeline("yolo26-seg")
    pipeline.add("nvmultiurisrcbin", "source", source_props)
    pipeline.add("nvinfer", "seg_infer", {"config-file-path": str(nvinfer_config)})
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
            "process-mode": 1,
            "display-mask": 1,
            "display-bbox": 0,
            "display-text": 1,
        },
    )

    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        pipeline.add("nveglglessink", "sink")

    pipeline.link("source", "seg_infer", "tiler", "osd", "sink")
    return PipelineContext(pipeline=pipeline, config_path=nvinfer_config)
