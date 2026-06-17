"""Pose feature metadata payloads for DS8."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass
class PoseFeatureResult:
    source_id: int
    frame_id: int
    object_id: int
    class_id: int
    bbox: Sequence[float]
    score: float
    kpt_mean_conf: float
    kpt_min_conf: float
    kpt_valid_frac: float
    features: Dict[str, float] = field(default_factory=dict)
    keypoints_roi: Optional[Sequence[Sequence[float]]] = None
    keypoints_abs: Optional[Sequence[Sequence[float]]] = None
    stable_id: Optional[int] = None
    model: str = "yolo26-pose"
    version: int = 1
    ts_us: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": "pose_features",
            "version": int(self.version),
            "model": str(self.model),
            "source_id": int(self.source_id),
            "frame_id": int(self.frame_id),
            "object_id": int(self.object_id),
            "class_id": int(self.class_id),
            "bbox": [float(x) for x in self.bbox],
            "score": float(self.score),
            "kpt_mean_conf": float(self.kpt_mean_conf),
            "kpt_min_conf": float(self.kpt_min_conf),
            "kpt_valid_frac": float(self.kpt_valid_frac),
            "features": {str(k): float(v) for k, v in (self.features or {}).items()},
        }
        if self.stable_id is not None:
            payload["stable_id"] = int(self.stable_id)
        if self.keypoints_roi is not None:
            payload["keypoints_roi"] = [
                [float(p[0]), float(p[1]), float(p[2])] for p in self.keypoints_roi
            ]
        if self.keypoints_abs is not None:
            payload["keypoints_abs"] = [
                [float(p[0]), float(p[1]), float(p[2])] for p in self.keypoints_abs
            ]
        if self.ts_us is not None:
            payload["ts_us"] = int(self.ts_us)
        return payload
