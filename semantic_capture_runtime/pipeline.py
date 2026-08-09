"""DS8 Service Maker pipeline for YOLO26 ADE20K semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyservicemaker import Pipeline, Probe

from .snapshot import SemanticSnapshotEmitter


@dataclass
class PipelineContext:
    pipeline: Pipeline
    snapshot: SemanticSnapshotEmitter


def _queue_properties() -> dict[str, int]:
    return {
        "leaky": 2,
        "max-size-buffers": 1,
        "max-size-bytes": 0,
        "max-size-time": 0,
    }


def _source_properties(source_cfg: dict[str, object]) -> dict[str, object]:
    uris = [str(value) for value in source_cfg.get("uris", []) or []]
    sensor_ids = [str(value) for value in source_cfg.get("sensor_ids", []) or []]
    sensor_names = [str(value) for value in source_cfg.get("sensor_names", []) or []]
    if len(uris) != 3 or len(sensor_ids) != 3 or len(sensor_names) != 3:
        raise RuntimeError("YOLO26 semantic testpipeline requires exactly three room sources")
    return {
        "uri-list": ",".join(uris),
        "sensor-id-list": ",".join(sensor_ids),
        "sensor-name-list": ",".join(sensor_names),
        "port": "0",
        "max-batch-size": 3,
        "width": int(source_cfg.get("width", 1920) or 1920),
        "height": int(source_cfg.get("height", 1080) or 1080),
        "live-source": 1,
        "batched-push-timeout": 40000,
        "enable-padding": 1,
        "nvbuf-memory-type": 0,
        "sync-inputs": 0,
        "drop-pipeline-eos": 0,
        "cache-buffer": 0,
        "latency": 100,
        "select-rtp-protocol": 4,
        "cudadec-memtype": 0,
        "sensorID-padID-mapping": 0,
    }


def build_pipeline(
    source_cfg: dict[str, object],
    nvinfer_config: Path,
    *,
    output_dir: Path,
    model_size: str,
    labels: list[str],
    warmup_frames: int,
    alpha: float,
    headless: bool,
) -> PipelineContext:
    sensor_names = [str(value) for value in source_cfg.get("sensor_names", []) or []]
    snapshot = SemanticSnapshotEmitter(
        output_dir=output_dir,
        model_size=model_size,
        labels=labels,
        sensor_names=sensor_names,
        warmup_frames=warmup_frames,
        alpha=alpha,
    )
    pipeline = Pipeline(f"yolo26{model_size}-sem-ade20k")
    pipeline.add("nvmultiurisrcbin", "source", _source_properties(source_cfg))
    pipeline.add("queue", "pre_queue", _queue_properties())
    pipeline.add("nvvideoconvert", "infer_convert", {"gpu-id": 0, "nvbuf-memory-type": 0})
    pipeline.add("capsfilter", "infer_caps", {"caps": "video/x-raw(memory:NVMM),format=RGBA"})
    pipeline.add("nvinfer", "semantic_infer", {"config-file-path": str(nvinfer_config)})
    pipeline.add("nvvideoconvert", "snapshot_convert", {"gpu-id": 0, "nvbuf-memory-type": 0})
    pipeline.add("capsfilter", "snapshot_caps", {"caps": "video/x-raw(memory:NVMM),format=RGB"})
    pipeline.add("nvvideoconvert", "display_convert", {"gpu-id": 0, "nvbuf-memory-type": 0})
    pipeline.add("capsfilter", "display_caps", {"caps": "video/x-raw(memory:NVMM),format=RGBA"})
    pipeline.add("queue", "post_queue", _queue_properties())
    pipeline.add(
        "nvmultistreamtiler",
        "tiler",
        {"rows": 1, "columns": 3, "width": 1920, "height": 640, "gpu-id": 0},
    )
    pipeline.add(
        "nvdsosd",
        "osd",
        {"process-mode": 1, "display-mask": 0, "display-bbox": 0, "display-text": 0},
    )
    if headless:
        pipeline.add("fakesink", "sink", {"sync": 0})
    else:
        pipeline.add("nveglglessink", "sink", {"sync": 0, "qos": 0})
    pipeline.link(
        "source",
        "pre_queue",
        "infer_convert",
        "infer_caps",
        "semantic_infer",
        "snapshot_convert",
        "snapshot_caps",
        "display_convert",
        "display_caps",
        "post_queue",
        "tiler",
        "osd",
        "sink",
    )
    pipeline.attach("snapshot_caps", Probe("semantic_snapshots", snapshot))
    return PipelineContext(pipeline=pipeline, snapshot=snapshot)
