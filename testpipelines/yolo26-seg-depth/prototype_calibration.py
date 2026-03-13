"""Prototype calibration resolver for the DS8 seg+depth test pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from noesis.calibration.manager import CalibrationManager, CalibrationSnapshot

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERAS_PATH = REPO_ROOT / "config" / "cameras.yaml"
DEFAULT_EXTRINSICS_PATH = REPO_ROOT / "config" / "camera_calibration.json"
DEFAULT_ALIGNMENT_PATH = REPO_ROOT / "config" / "ply_alignment.json"


@dataclass(frozen=True, slots=True)
class PrototypeCalibrationBinding:
    runtime_source_id: int
    calibration_source_id: int
    camera_id: str
    display_name: str


def _camera_entry_for_source_id(cameras_path: Path, source_id: int) -> Optional[Mapping[str, Any]]:
    try:
        payload = yaml.safe_load(cameras_path.read_text(encoding="utf-8")) or {}
    except Exception:
        LOGGER.exception("Failed to read prototype cameras config: %s", cameras_path)
        return None
    cameras = payload.get("cameras") or payload.get("sources") or {}
    if not isinstance(cameras, Mapping):
        return None
    entry = cameras.get(source_id)
    if entry is None:
        entry = cameras.get(str(source_id))
    return entry if isinstance(entry, Mapping) else None


class PrototypeCalibrationResolver:
    """Resolve a single-stream prototype source into canonical DS8 calibration data."""

    def __init__(
        self,
        *,
        binding: PrototypeCalibrationBinding,
        cameras_path: Path = DEFAULT_CAMERAS_PATH,
        extrinsics_path: Path = DEFAULT_EXTRINSICS_PATH,
        alignment_path: Path = DEFAULT_ALIGNMENT_PATH,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self._binding = binding
        self._cameras_path = Path(cameras_path)
        self._extrinsics_path = Path(extrinsics_path)
        self._alignment_path = Path(alignment_path)
        self._manager = CalibrationManager(
            self._cameras_path,
            self._extrinsics_path,
            self._alignment_path,
            streammux_size=(int(frame_size[0] or 0), int(frame_size[1] or 0)),
        )
        # Publish runtime source ids in the same shape that DS8 hooks expect,
        # even though this prototype only runs a single selected camera.
        self._manager.set_camera_labels({int(binding.runtime_source_id): str(binding.camera_id)})

    @classmethod
    def from_source_config(
        cls,
        source_cfg: Mapping[str, object],
        *,
        frame_size: Tuple[int, int],
        cameras_path: Path = DEFAULT_CAMERAS_PATH,
        extrinsics_path: Optional[Path] = None,
        alignment_path: Path = DEFAULT_ALIGNMENT_PATH,
        runtime_source_id: int = 0,
    ) -> "PrototypeCalibrationResolver":
        calibration_source_id = int(source_cfg.get("sensor_id", 0) or 0)
        entry = _camera_entry_for_source_id(Path(cameras_path), calibration_source_id)
        if entry is None:
            raise RuntimeError(
                f"Unable to resolve calibration camera entry for source_id={calibration_source_id} in {cameras_path}"
            )
        camera_id = str(entry.get("name") or "").strip()
        if not camera_id:
            raise RuntimeError(
                f"Calibration camera entry for source_id={calibration_source_id} in {cameras_path} has no name"
            )
        if extrinsics_path is None:
            env_path = str(os.environ.get("NOESIS_CALIBRATION_EXTRINSICS", "") or "").strip()
            extrinsics_path = Path(env_path) if env_path else DEFAULT_EXTRINSICS_PATH
        binding = PrototypeCalibrationBinding(
            runtime_source_id=int(runtime_source_id),
            calibration_source_id=calibration_source_id,
            camera_id=camera_id,
            display_name=str(source_cfg.get("sensor_name") or camera_id),
        )
        return cls(
            binding=binding,
            cameras_path=Path(cameras_path),
            extrinsics_path=Path(extrinsics_path),
            alignment_path=Path(alignment_path),
            frame_size=frame_size,
        )

    def camera_labels(self) -> Dict[int, str]:
        return {int(self._binding.runtime_source_id): str(self._binding.camera_id)}

    def binding_for_runtime_source(self, source_id: int) -> Optional[PrototypeCalibrationBinding]:
        if int(source_id) != int(self._binding.runtime_source_id):
            return None
        return self._binding

    def snapshot(self, source_id: int, camera_id: Optional[str] = None) -> Optional[CalibrationSnapshot]:
        binding = self.binding_for_runtime_source(source_id)
        if binding is None:
            return None
        requested_camera = str(camera_id or binding.camera_id).strip()
        if requested_camera and requested_camera != binding.camera_id:
            LOGGER.debug(
                "Prototype calibration snapshot camera mismatch: runtime_source=%s requested=%s expected=%s",
                source_id,
                requested_camera,
                binding.camera_id,
            )
        snapshot = self._manager.snapshot(binding.calibration_source_id, binding.camera_id)
        if snapshot is None:
            return None
        # The current DS8 runtime baseline treats camera_calibration.json extrinsics
        # as meter-scale already and does not inject align.units.s_obj_to_m into live
        # world projection snapshots. Mirror that behavior here so the prototype's
        # spatial points stay comparable with the main runtime.
        if abs(float(snapshot.unit_scale) - 1.0) > 1e-9:
            LOGGER.debug(
                "Normalizing prototype calibration unit_scale from %s to 1.0 for camera_id=%s",
                snapshot.unit_scale,
                binding.camera_id,
            )
            snapshot = replace(snapshot, unit_scale=1.0)
        return snapshot

    @property
    def binding(self) -> PrototypeCalibrationBinding:
        return self._binding

    @property
    def extrinsics_path(self) -> Path:
        return self._extrinsics_path
