"""DeepStream 8 Service Maker pipeline for YOLO26 seg + depth fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pyservicemaker import Pipeline, Probe

from debug_probes import (
    AlignedDepthFrameStore,
    DepthFrameProbe,
    DepthOverlayProbe,
    ObjectDepthFusionProbe,
    OverlayStateStore,
    PrototypeRuntimeStats,
)
from model_setup import PipelineAssets


@dataclass
class PipelineContext:
    pipeline: Pipeline
    seg_config_path: Path
    depth_config_path: Path | None
    runtime_stats: PrototypeRuntimeStats


def _build_source_properties(source_cfg: Dict[str, object]) -> Dict[str, object]:
    uri = str(source_cfg["uri"])
    width = int(source_cfg.get("width", 1920) or 1920)
    height = int(source_cfg.get("height", 1080) or 1080)
    return {
        "uri-list": uri,
        "sensor-id-list": str(source_cfg["sensor_id"]),
        "sensor-name-list": str(source_cfg["sensor_name"]),
        "port": "0",
        "max-batch-size": 1,
        "width": width,
        "height": height,
        "live-source": 0 if uri.lower().startswith("file:") else 1,
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
        "file-loop": 1 if uri.lower().startswith("file:") else 0,
    }


def _queue_props() -> Dict[str, int]:
    return {
        "leaky": 2,
        "max-size-buffers": 1,
        "max-size-bytes": 0,
        "max-size-time": 0,
    }


def build_pipeline(
    source_cfg: Dict[str, object],
    assets: PipelineAssets,
    *,
    frame_size: Tuple[int, int],
    headless: bool = False,
    show_bbox: bool = False,
    depth_every_n_frames: int = 1,
    calibration_resolver: Optional[Any] = None,
) -> PipelineContext:
    width = int(frame_size[0] or 1920)
    height = int(frame_size[1] or 1080)
    depth_store = AlignedDepthFrameStore() if assets.depth is not None else None
    overlay_store = OverlayStateStore()
    runtime_stats = PrototypeRuntimeStats(
        depth_enabled=assets.depth is not None,
        depth_every_n_frames=max(1, int(depth_every_n_frames)),
    )

    pipeline = Pipeline("yolo26-seg-depth")
    if calibration_resolver is not None:
        try:
            setattr(pipeline, "camera_labels", calibration_resolver.camera_labels())
            setattr(pipeline, "bev_calibration", calibration_resolver)
        except Exception:
            pass
    source_cfg = dict(source_cfg)
    source_cfg["width"] = width
    source_cfg["height"] = height
    pipeline.add("nvmultiurisrcbin", "source", _build_source_properties(source_cfg))
    pipeline.add("queue", "seg_pre_queue", _queue_props())
    pipeline.add(
        "nvdspreprocess",
        "seg_preproc",
        {"config-file": str(assets.seg.preprocess_config_path)},
    )
    pipeline.add("nvinfer", "seg_infer", {"config-file-path": str(assets.seg.config_path)})
    pipeline.add("queue", "overlay_stage", _queue_props())
    pipeline.add(
        "nvdsosd",
        "osd",
        {
            "process-mode": 1,
            "display-mask": 1,
            "display-bbox": 1 if show_bbox else 0,
            "display-text": 1,
        },
    )

    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        pipeline.add("nveglglessink", "sink", {"sync": 0})

    if assets.depth is not None:
        pipeline.add("queue", "depth_pre_queue", _queue_props())
        pipeline.add("nvinfer", "depth_infer", {"config-file-path": str(assets.depth.config_path)})
        pipeline.link(
            "source",
            "depth_pre_queue",
            "depth_infer",
            "seg_pre_queue",
            "seg_preproc",
            "seg_infer",
            "overlay_stage",
            "osd",
            "sink",
        )
        pipeline.attach(
            "depth_infer",
            Probe(
                "depth_frame_probe",
                DepthFrameProbe(
                    depth_store=depth_store,
                    stats=runtime_stats,
                    depth_gie_id=2,
                    depth_model_name=assets.depth.model_name,
                    depth_unit=assets.depth.unit,
                    depth_is_metric=assets.depth.is_metric,
                    fallback_frame_size=(width, height),
                ),
            ),
        )
    else:
        pipeline.link(
            "source",
            "seg_pre_queue",
            "seg_preproc",
            "seg_infer",
            "overlay_stage",
            "osd",
            "sink",
        )
    pipeline.attach(
        "seg_infer",
        Probe(
            "object_depth_fusion",
            ObjectDepthFusionProbe(
                depth_store=depth_store,
                overlay_store=overlay_store,
                stats=runtime_stats,
                depth_model_name=assets.depth.model_name if assets.depth is not None else "depth-disabled",
                depth_unit=assets.depth.unit if assets.depth is not None else "m",
                depth_is_metric=assets.depth.is_metric if assets.depth is not None else True,
                depth_every_n_frames=max(1, int(depth_every_n_frames)),
                calibration_resolver=calibration_resolver,
            ),
        ),
    )
    if assets.depth is not None:
        pipeline.attach(
            "overlay_stage",
            Probe(
                "depth_overlay",
                DepthOverlayProbe(overlay_store=overlay_store, fallback_frame_size=(width, height)),
            ),
        )
    return PipelineContext(
        pipeline=pipeline,
        seg_config_path=assets.seg.config_path,
        depth_config_path=assets.depth.config_path if assets.depth is not None else None,
        runtime_stats=runtime_stats,
    )
