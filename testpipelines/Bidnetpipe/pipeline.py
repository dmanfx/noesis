"""DeepStream 8 Service Maker pipeline for semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import threading

from pyservicemaker import Pipeline, Probe

from . import model_setup, sources
from .probes import SegmentationHeartbeat


@dataclass
class PipelineContext:
    pipeline: Pipeline
    config_path: Path


def _load_default_stream_size() -> Dict[str, int]:
    width = 1920
    height = 1080
    try:
        import config as app_config

        cfg = getattr(app_config, "config", None)
        if cfg is None:
            cfg = app_config.AppConfig()
        width = int(getattr(cfg.cameras, "CAMERA_WIDTH", width))
        height = int(getattr(cfg.cameras, "CAMERA_HEIGHT", height))
    except Exception:
        pass
    return {"width": width, "height": height}


def _build_source_properties(source_cfg: Dict[str, object]) -> Dict[str, object]:
    uris = source_cfg.get("uris", [])
    sensor_ids = source_cfg.get("sensor_ids", [])
    sensor_names = source_cfg.get("sensor_names", [])
    if len(uris) != 3:
        raise RuntimeError(f"Expected 3 URIs, got {len(uris)}")
    dims = _load_default_stream_size()
    props: Dict[str, object] = {
        "uri-list": ",".join(uris),
        "sensor-id-list": ",".join(sensor_ids),
        "sensor-name-list": ",".join(sensor_names),
        "port": "0",
        "max-batch-size": len(uris),
        "width": dims["width"],
        "height": dims["height"],
        "live-source": 1,
    }
    return props


def build_pipeline(
    sources_path: Path,
    nvinfer_config: Path,
    model_info: Dict[str, object],
    labels: Dict[str, object],
    headless: bool = False,
    tiler_width: Optional[int] = None,
    tiler_height: Optional[int] = None,
    first_heartbeat_event: Optional[threading.Event] = None,
) -> PipelineContext:
    source_cfg = sources.load_sources_yaml(sources_path)
    source_props = _build_source_properties(source_cfg)
    output = model_info["output"]
    seg_width = int(output["width"])
    seg_height = int(output["height"])
    tiler_width = int(tiler_width) if tiler_width else seg_width * 3
    tiler_height = int(tiler_height) if tiler_height else seg_height

    pipeline = Pipeline("bidnetpipe-seg")
    pipeline.add("nvmultiurisrcbin", "source", source_props)
    pipeline.add("nvinfer", "seg_infer", {"config-file-path": str(nvinfer_config)})
    pipeline.add(
        "nvsegvisual",
        "seg_visual",
        {
            # nvsegvisual's width/height control the output frame resolution. If these
            # don't match the segmentation meta resolution, the overlay is applied only
            # to the top-left portion of the frame.
            "width": seg_width,
            "height": seg_height,
            "gpu-id": 0,
            "batch-size": len(source_cfg.get("uris", [])),
            # Use host-memory path so floor-only class_map rewriting in the probe
            # is reflected in the visualization.
            "gpu-on": False,
            "original-background": True,
            # nvinfer gie-unique-id is set to 1 in the generated config.
            "operate-on-seg-meta-id": 1,
            # Background class-id used when original-background=true. For ADE20K
            # models there is no explicit background class; leave default (0) so at
            # least one class can act as background for visualization.
            "class-id": 0,
        },
    )
    pipeline.add("nvdsosd", "osd")
    pipeline.add(
        "nvmultistreamtiler",
        "tiler",
        {
            "rows": 1,
            "columns": 3,
            "width": int(tiler_width),
            "height": int(tiler_height),
        },
    )

    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        pipeline.add("nveglglessink", "sink")

    pipeline.link("source", "seg_infer", "seg_visual", "osd", "tiler", "sink")

    heartbeat = SegmentationHeartbeat(
        labels=labels,
        sensor_ids=[str(s) for s in source_cfg.get("sensor_ids", [])],
        first_heartbeat_event=first_heartbeat_event,
    )
    pipeline.attach("seg_infer", Probe("heartbeat", heartbeat))
    return PipelineContext(pipeline=pipeline, config_path=sources_path)


def build_default_pipeline(
    sources_path: Optional[Path] = None,
    headless: bool = False,
) -> PipelineContext:
    model_setup.ensure_model_artifacts()
    model_info = model_setup.load_model_info()
    labels = model_setup.load_labels()
    sources_path = sources_path or sources.DEFAULT_SOURCES_PATH
    return build_pipeline(
        sources_path=sources_path,
        nvinfer_config=model_setup.NVINFER_CONFIG_PATH,
        model_info=model_info,
        labels=labels,
        headless=headless,
    )
