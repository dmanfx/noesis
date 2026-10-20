from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]
BBox = Tuple[float, float, float, float]
Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def _coerce_bbox(value: Sequence[Any]) -> BBox:
    items = list(value)
    if len(items) != 4:
        raise ValueError("bbox must contain exactly four values")
    return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _coerce_point2(value: Sequence[Any]) -> Point2:
    items = list(value)
    if len(items) != 2:
        raise ValueError("point2 must contain exactly two values")
    return (float(items[0]), float(items[1]))


def _coerce_point3(value: Sequence[Any]) -> Point3:
    items = list(value)
    if len(items) != 3:
        raise ValueError("point3 must contain exactly three values")
    return (float(items[0]), float(items[1]), float(items[2]))


def _coerce_optional_point2(value: Any) -> Optional[Point2]:
    if value is None:
        return None
    return _coerce_point2(value)


def _coerce_optional_point3(value: Any) -> Optional[Point3]:
    if value is None:
        return None
    return _coerce_point3(value)


@dataclass(frozen=True, slots=True)
class PoseDepthSample:
    """One bounded anatomy patch, including rejected attempts, at its own UV."""

    anchor_source: str
    anchor_uv: Sequence[float]
    sample_count: int
    valid_fraction: float
    mask_area_px: int
    depth_median: Optional[float] = None
    depth_p10: Optional[float] = None
    depth_p90: Optional[float] = None
    evidence_quality: str = "rejected"
    evidence_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.anchor_source not in {"pose_ankle_support", "pose_torso_support", "pose_shoulder_support"}:
            raise ValueError("unsupported pose depth anatomy")
        object.__setattr__(self, "anchor_uv", _coerce_point2(self.anchor_uv))
        if not all(math.isfinite(v) for v in self.anchor_uv):
            raise ValueError("pose depth UV must be finite")
        object.__setattr__(self, "sample_count", max(0, int(self.sample_count)))
        object.__setattr__(self, "mask_area_px", max(0, int(self.mask_area_px)))
        if not math.isfinite(float(self.valid_fraction)) or not 0 <= float(self.valid_fraction) <= 1:
            raise ValueError("invalid pose depth support fraction")
        for key in ("depth_median", "depth_p10", "depth_p90"):
            value = _coerce_optional_float(getattr(self, key))
            if value is not None and not math.isfinite(value):
                raise ValueError("pose depth statistic must be finite")
            object.__setattr__(self, key, value)
        if self.evidence_quality not in {"good", "estimated", "rejected"}:
            raise ValueError("invalid pose depth quality")
        if self.evidence_reason is not None:
            object.__setattr__(self, "evidence_reason", str(self.evidence_reason)[:160])

    def to_dict(self) -> JsonDict:
        return {key: (list(self.anchor_uv) if key == "anchor_uv" else getattr(self, key))
                for key in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class NativePosePatchSupport:
    """Private scalar copy of an actually sampled patch, never a pixel sigma.

    Native metadata transport only. Public object-depth serialization omits it.
    Original measurement/raster fields are deliberately not renewed by caching.
    """

    anchor_source: str
    anchor_uv: Sequence[float]
    capsules: Sequence[Sequence[float]]
    image_size: Sequence[int]
    measurement_frame_id: int
    measurement_ts_us: int

    def __post_init__(self) -> None:
        if self.anchor_source not in {"pose_ankle_support", "pose_torso_support", "pose_shoulder_support"}:
            raise ValueError("body patch anatomy required")
        uv = _coerce_point2(self.anchor_uv)
        size = tuple(self.image_size)
        capsules = tuple(tuple(float(v) for v in c) for c in self.capsules)
        if (len(size) != 2 or any(type(v) is not int or not 8 < v <= 32768 for v in size)
                or not 1 <= len(capsules) <= (2 if self.anchor_source == "pose_ankle_support" else 1)
                or any(len(c) != 6 or not all(math.isfinite(v) for v in c)
                    or min(c[4:]) <= 0 for c in capsules)
                or not all(math.isfinite(v) and 0 <= v <= limit for v, limit in zip(uv, size))
                or type(self.measurement_frame_id) is not int or self.measurement_frame_id < 0
                or type(self.measurement_ts_us) is not int or self.measurement_ts_us < 0):
            raise ValueError("invalid original body patch support")
        object.__setattr__(self, "anchor_uv", uv)
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "capsules", capsules)

    def to_dict(self) -> JsonDict:
        return {"anchor_source": self.anchor_source, "anchor_uv": list(self.anchor_uv),
                "capsules": [list(c) for c in self.capsules], "image_size": list(self.image_size),
                "measurement_frame_id": self.measurement_frame_id,
                "measurement_ts_us": self.measurement_ts_us}

    def enclosing_radius_px(self) -> float:
        """Enclose endpoints AND their sampled radii about the exact anchor."""
        u, v = self.anchor_uv
        return max(math.hypot(x-u, y-v) + r for c in self.capsules
                   for x, y, r in ((c[0], c[1], c[4]), (c[2], c[3], max(c[4:]))))


@dataclass(frozen=True, slots=True)
class ObjectDepthResult:
    source_id: int
    frame_id: int
    object_id: int
    class_id: int
    bbox: Sequence[float]
    score: float
    sampling_mode: str
    status: str
    unit: str
    is_metric: bool
    sample_count: int
    valid_fraction: float
    depth_center: Optional[float] = None
    depth_median: Optional[float] = None
    depth_mean: Optional[float] = None
    depth_p10: Optional[float] = None
    depth_p90: Optional[float] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    mask_area_px: Optional[int] = None
    stable_id: Optional[int] = None
    depth_map_ref: Optional[str] = None
    anchor_uv: Optional[Sequence[float]] = None
    anchor_source: Optional[str] = None
    anchor_depth_m: Optional[float] = None
    anchor_sample_count: Optional[int] = None
    anchor_valid_fraction: Optional[float] = None
    evidence_quality: Optional[str] = None
    evidence_reason: Optional[str] = None
    depth_spread_m: Optional[float] = None
    anchor_depth_spread_m: Optional[float] = None
    pose_depth_samples: Sequence[PoseDepthSample | Mapping[str, Any]] = ()
    pose_joint_confidences: Sequence[float] = ()
    measurement_frame_id: Optional[int] = None
    measurement_ts_us: Optional[int] = None
    measurement_age_us: Optional[int] = None
    measurement_cached: Optional[bool] = None
    depth_tensor_frame_id: Optional[int] = None
    depth_tensor_ts_us: Optional[int] = None
    depth_tensor_age_frames: Optional[int] = None
    depth_tensor_age_us: Optional[int] = None
    world_point: Optional[Sequence[float]] = None
    world_point_depth: Optional[Sequence[float]] = None
    world_point_floor: Optional[Sequence[float]] = None
    projection_method: Optional[str] = None
    spatial_status: Optional[str] = None
    spatial_class: Optional[str] = None
    model: str = "depth-anything"
    version: int = 2
    ts_us: Optional[int] = None
    native_pose_support: Sequence[NativePosePatchSupport | Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        if len(self.native_pose_support) > 2:
            raise ValueError("private body patch budget exceeded")
        object.__setattr__(self, "native_pose_support", tuple(
            p if isinstance(p, NativePosePatchSupport) else NativePosePatchSupport(**p)
            for p in self.native_pose_support))
        if len(self.pose_depth_samples) > 3:
            raise ValueError("pose depth attempts exceed bounded anatomy budget")
        object.__setattr__(self, "pose_depth_samples", tuple(
            sample if isinstance(sample, PoseDepthSample) else PoseDepthSample(**sample)
            for sample in self.pose_depth_samples
        ))
        if len(self.pose_joint_confidences) not in (0, 17):
            raise ValueError("pose confidence diagnostics require 17 joints")
        confidences = tuple(float(value) for value in self.pose_joint_confidences)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidences):
            raise ValueError("invalid pose joint confidence")
        object.__setattr__(self, "pose_joint_confidences", confidences)
        object.__setattr__(self, "source_id", int(self.source_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "bbox", _coerce_bbox(self.bbox))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "sampling_mode", str(self.sampling_mode).strip() or "instance_mask")
        object.__setattr__(self, "status", str(self.status).strip() or "unknown")
        object.__setattr__(self, "unit", str(self.unit).strip() or "relative")
        object.__setattr__(self, "is_metric", bool(self.is_metric))
        object.__setattr__(self, "sample_count", max(0, int(self.sample_count)))
        object.__setattr__(self, "valid_fraction", max(0.0, min(1.0, float(self.valid_fraction))))
        object.__setattr__(self, "depth_center", _coerce_optional_float(self.depth_center))
        object.__setattr__(self, "depth_median", _coerce_optional_float(self.depth_median))
        object.__setattr__(self, "depth_mean", _coerce_optional_float(self.depth_mean))
        object.__setattr__(self, "depth_p10", _coerce_optional_float(self.depth_p10))
        object.__setattr__(self, "depth_p90", _coerce_optional_float(self.depth_p90))
        object.__setattr__(self, "depth_min", _coerce_optional_float(self.depth_min))
        object.__setattr__(self, "depth_max", _coerce_optional_float(self.depth_max))
        if self.mask_area_px is not None:
            object.__setattr__(self, "mask_area_px", max(0, int(self.mask_area_px)))
        if self.stable_id is not None:
            object.__setattr__(self, "stable_id", int(self.stable_id))
        if self.depth_map_ref is not None:
            ref = str(self.depth_map_ref).strip()
            object.__setattr__(self, "depth_map_ref", ref or None)
        object.__setattr__(self, "anchor_uv", _coerce_optional_point2(self.anchor_uv))
        if self.anchor_source is not None:
            object.__setattr__(self, "anchor_source", str(self.anchor_source).strip() or None)
        object.__setattr__(self, "anchor_depth_m", _coerce_optional_float(self.anchor_depth_m))
        if self.anchor_sample_count is not None:
            object.__setattr__(self, "anchor_sample_count", max(0, int(self.anchor_sample_count)))
        if self.anchor_valid_fraction is not None:
            object.__setattr__(self, "anchor_valid_fraction", max(0.0, min(1.0, float(self.anchor_valid_fraction))))
        if self.evidence_quality is not None:
            object.__setattr__(self, "evidence_quality", str(self.evidence_quality).strip() or None)
        if self.evidence_reason is not None:
            object.__setattr__(self, "evidence_reason", str(self.evidence_reason).strip() or None)
        if self.depth_spread_m is not None:
            object.__setattr__(self, "depth_spread_m", max(0.0, float(self.depth_spread_m)))
        if self.anchor_depth_spread_m is not None:
            object.__setattr__(self, "anchor_depth_spread_m", max(0.0, float(self.anchor_depth_spread_m)))
        if self.measurement_frame_id is not None:
            object.__setattr__(self, "measurement_frame_id", int(self.measurement_frame_id))
        if self.measurement_ts_us is not None:
            object.__setattr__(self, "measurement_ts_us", int(self.measurement_ts_us))
        if self.measurement_age_us is not None:
            object.__setattr__(self, "measurement_age_us", max(0, int(self.measurement_age_us)))
        if self.measurement_cached is not None:
            object.__setattr__(self, "measurement_cached", bool(self.measurement_cached))
        if self.depth_tensor_frame_id is not None:
            object.__setattr__(self, "depth_tensor_frame_id", int(self.depth_tensor_frame_id))
        if self.depth_tensor_ts_us is not None:
            object.__setattr__(self, "depth_tensor_ts_us", int(self.depth_tensor_ts_us))
        if self.depth_tensor_age_frames is not None:
            object.__setattr__(
                self,
                "depth_tensor_age_frames",
                max(0, int(self.depth_tensor_age_frames)),
            )
        if self.depth_tensor_age_us is not None:
            object.__setattr__(
                self,
                "depth_tensor_age_us",
                max(0, int(self.depth_tensor_age_us)),
            )
        object.__setattr__(self, "world_point", _coerce_optional_point3(self.world_point))
        object.__setattr__(self, "world_point_depth", _coerce_optional_point3(self.world_point_depth))
        object.__setattr__(self, "world_point_floor", _coerce_optional_point3(self.world_point_floor))
        if self.projection_method is not None:
            object.__setattr__(self, "projection_method", str(self.projection_method).strip() or None)
        if self.spatial_status is not None:
            object.__setattr__(self, "spatial_status", str(self.spatial_status).strip() or None)
        if self.spatial_class is not None:
            object.__setattr__(self, "spatial_class", str(self.spatial_class).strip() or None)
        object.__setattr__(self, "model", str(self.model).strip() or "depth-anything")
        object.__setattr__(self, "version", int(self.version))
        if self.ts_us is not None:
            object.__setattr__(self, "ts_us", int(self.ts_us))

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "type": "object_depth",
            "version": int(self.version),
            "model": str(self.model),
            "source_id": int(self.source_id),
            "frame_id": int(self.frame_id),
            "object_id": int(self.object_id),
            "class_id": int(self.class_id),
            "bbox": [float(x) for x in self.bbox],
            "score": float(self.score),
            "sampling_mode": str(self.sampling_mode),
            "status": str(self.status),
            "unit": str(self.unit),
            "is_metric": bool(self.is_metric),
            "sample_count": int(self.sample_count),
            "valid_fraction": float(self.valid_fraction),
        }
        optional_fields = {
            "depth_center": self.depth_center,
            "depth_median": self.depth_median,
            "depth_mean": self.depth_mean,
            "depth_p10": self.depth_p10,
            "depth_p90": self.depth_p90,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "mask_area_px": self.mask_area_px,
            "stable_id": self.stable_id,
            "depth_map_ref": self.depth_map_ref,
            "anchor_uv": list(self.anchor_uv) if self.anchor_uv is not None else None,
            "anchor_source": self.anchor_source,
            "anchor_depth_m": self.anchor_depth_m,
            "anchor_sample_count": self.anchor_sample_count,
            "anchor_valid_fraction": self.anchor_valid_fraction,
            "evidence_quality": self.evidence_quality,
            "evidence_reason": self.evidence_reason,
            "depth_spread_m": self.depth_spread_m,
            "anchor_depth_spread_m": self.anchor_depth_spread_m,
            "measurement_frame_id": self.measurement_frame_id,
            "measurement_ts_us": self.measurement_ts_us,
            "measurement_age_us": self.measurement_age_us,
            "measurement_cached": self.measurement_cached,
            "depth_tensor_frame_id": self.depth_tensor_frame_id,
            "depth_tensor_ts_us": self.depth_tensor_ts_us,
            "depth_tensor_age_frames": self.depth_tensor_age_frames,
            "depth_tensor_age_us": self.depth_tensor_age_us,
            "world_point": list(self.world_point) if self.world_point is not None else None,
            "world_point_depth": list(self.world_point_depth) if self.world_point_depth is not None else None,
            "world_point_floor": list(self.world_point_floor) if self.world_point_floor is not None else None,
            "projection_method": self.projection_method,
            "spatial_status": self.spatial_status,
            "spatial_class": self.spatial_class,
            "ts_us": self.ts_us,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
        if self.pose_depth_samples:
            payload["pose_depth_samples"] = [sample.to_dict() for sample in self.pose_depth_samples]
        if self.pose_joint_confidences:
            payload["pose_joint_confidences"] = list(self.pose_joint_confidences)
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return json.dumps(self.to_dict(), separators=(",", ":"))
        return json.dumps(self.to_dict(), indent=indent)

    def to_native_dict(self) -> JsonDict:
        """Opaque native attachment/cache only; NOT the public depth contract."""
        payload = self.to_dict()
        if self.native_pose_support:
            payload["native_pose_support"] = [p.to_dict() for p in self.native_pose_support]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectDepthResult":
        data = dict(payload)
        data.pop("type", None)
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "ObjectDepthResult":
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("ObjectDepthResult JSON payload must decode to a dict")
        return cls.from_dict(decoded)
