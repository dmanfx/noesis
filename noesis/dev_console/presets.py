from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from noesis.ds8_preflight import REPO_ROOT


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    preset_type: str
    runnable: bool
    pipeline_config: str
    cameras_config: str = "config/cameras.yaml"
    pgie_profile: str = "yolo11_seg"
    tracking_mode: str = "baseline"
    source_type: str = "rtsp"
    size: Optional[str] = "m"
    runtime_entry: str = "noesis/ds8_runtime.py"
    enable_rest: bool = True
    required_artifacts: List[str] = field(default_factory=list)
    notes: str = ""
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["exists"] = (REPO_ROOT / self.pipeline_config).exists()
        payload["missing_artifacts"] = missing_artifacts(self)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _existing_configs() -> List[str]:
    return sorted(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "config").glob("infer*.yaml"))


def _base_presets() -> List[Preset]:
    return [
        Preset(
            id="baseline-rtsp-yolo11-seg",
            name="Baseline RTSP + YOLO11 segmentation",
            preset_type="canonical",
            runnable=True,
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo11_seg",
            size="m",
            required_artifacts=[
                "config/depth_registration.json",
                "models/engines/yolo11m-seg_cust.engine",
                "pipelines/nvdsinfer_yolo11_seg/libnvdsinfer_yolo11_seg.so",
            ],
            notes="Default segmentation-first operator profile.",
        ),
        Preset(
            id="baseline-rtsp-yolo11-detect",
            name="Baseline RTSP + YOLO11 detect",
            preset_type="canonical",
            runnable=True,
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo11",
            size="m",
            required_artifacts=["config/depth_registration.json", "models/engines/yolo11m_b3_fp16.engine"],
            notes="Fast detect profile; object-depth mask anchors warn unless strict baseline is disabled.",
        ),
        Preset(
            id="baseline-rtsp-yolo26-seg-s",
            name="Baseline RTSP + YOLO26 segmentation S",
            preset_type="experimental",
            runnable=True,
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo26_seg",
            size="s",
            required_artifacts=[
                "config/depth_registration.json",
                "models/engines/yolo26s-seg_fused_b3_fp16.engine",
                "pipelines/nvdsinfer_yolo26_seg/libnvdsinfer_yolo26_seg.so",
            ],
        ),
        Preset(
            id="baseline-rtsp-yolo26-detect-m",
            name="Baseline RTSP + YOLO26 detect M",
            preset_type="experimental",
            runnable=True,
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo26",
            size="m",
            required_artifacts=["config/depth_registration.json", "models/engines/yolo26m_dynamic_b1-3_fp16.engine"],
        ),
        Preset(
            id="baseline-rtsp-rfdetr-seg-m",
            name="Baseline RTSP + RF-DETR segmentation M",
            preset_type="experimental",
            runnable=True,
            pipeline_config="config/infer.yaml",
            pgie_profile="rfdetr_seg",
            size="m",
            required_artifacts=["config/depth_registration.json", "models/engines/rfdetr_seg_m_432_b3_fp16.engine"],
        ),
        Preset(
            id="baseline-rtsp-wholebody49-s",
            name="Baseline RTSP + Wholebody49 S",
            preset_type="experimental",
            runnable=True,
            pipeline_config="config/infer.yaml",
            pgie_profile="wholebody49",
            size="s",
            required_artifacts=[
                "config/depth_registration.json",
                "models/engines/deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine",
            ],
        ),
        Preset(
            id="v3dt-baseline-yolo11-seg",
            name="V3DT baseline + YOLO11 segmentation",
            preset_type="canonical",
            runnable=True,
            pipeline_config="config/infer_v3dt_baseline.yaml",
            cameras_config="config/cameras_v3dt_baseline.yaml",
            pgie_profile="yolo11_seg",
            tracking_mode="v3dt",
            size="m",
            required_artifacts=["config/v3dt/nvtracker_v3dt_baseline.yml"],
        ),
        Preset(
            id="v3dt-reimpl-fast-mp4",
            name="V3DT reimpl fast MP4",
            preset_type="experimental",
            runnable=True,
            pipeline_config="config/infer_v3dt_reimpl_fast_mp4.yaml",
            cameras_config="config/cameras_v3dt_baseline.yaml",
            pgie_profile="yolo11_seg",
            tracking_mode="v3dt",
            source_type="file",
            size="m",
        ),
        Preset(
            id="v3dt-reimpl-posefast-mp4",
            name="V3DT reimpl posefast MP4",
            preset_type="experimental",
            runnable=True,
            pipeline_config="config/infer_v3dt_reimpl_posefast_mp4.yaml",
            cameras_config="config/cameras_v3dt_baseline.yaml",
            pgie_profile="yolo11_seg",
            tracking_mode="v3dt",
            source_type="file",
            size="m",
        ),
        Preset(
            id="smoke-reid",
            name="ReID smoke",
            preset_type="smoke",
            runnable=True,
            pipeline_config="config/infer_smoke_reid.yaml",
            pgie_profile="yolo11_seg",
            tracking_mode="baseline",
            size="m",
        ),
    ]


def _auto_config_presets() -> List[Preset]:
    known_configs = {preset.pipeline_config for preset in _base_presets()}
    presets: List[Preset] = []
    for config in _existing_configs():
        if config in known_configs:
            continue
        name = Path(config).stem.replace("_", " ").title()
        is_v3dt = "v3dt" in config
        presets.append(
            Preset(
                id=Path(config).stem.replace("_", "-"),
                name=name,
                preset_type="experimental" if is_v3dt else "smoke",
                runnable=True,
                pipeline_config=config,
                cameras_config="config/cameras_v3dt_baseline.yaml" if is_v3dt else "config/cameras.yaml",
                pgie_profile="yolo11_seg",
                tracking_mode="v3dt" if is_v3dt else "baseline",
                source_type="file" if "mp4" in config else "rtsp",
                size="m",
            )
        )
    return presets


def missing_artifacts(preset: Preset) -> List[str]:
    missing: List[str] = []
    for raw in preset.required_artifacts:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            missing.append(raw)
    if not (REPO_ROOT / preset.pipeline_config).exists():
        missing.append(preset.pipeline_config)
    if not (REPO_ROOT / preset.cameras_config).exists():
        missing.append(preset.cameras_config)
    return missing


def list_presets(*, runnable_only: bool = False) -> List[Preset] | List[Dict[str, Any]]:
    presets = _base_presets() + _auto_config_presets()
    by_id: Dict[str, Preset] = {}
    for preset in presets:
        by_id[preset.id] = preset
    ordered = sorted(by_id.values(), key=lambda item: (item.preset_type != "canonical", item.name.lower()))
    if runnable_only:
        return [preset.to_dict() for preset in ordered if preset.runnable]
    return ordered


def get_preset(preset_id: str) -> Optional[Preset]:
    for preset in list_presets():  # type: ignore[assignment]
        if isinstance(preset, Preset) and preset.id == preset_id:
            return preset
    return None
