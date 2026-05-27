from __future__ import annotations

import json
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
    world_point: Optional[Sequence[float]] = None
    world_point_depth: Optional[Sequence[float]] = None
    world_point_floor: Optional[Sequence[float]] = None
    projection_method: Optional[str] = None
    spatial_status: Optional[str] = None
    spatial_class: Optional[str] = None
    model: str = "depth-anything"
    version: int = 2
    ts_us: Optional[int] = None

    def __post_init__(self) -> None:
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
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return json.dumps(self.to_dict(), separators=(",", ":"))
        return json.dumps(self.to_dict(), indent=indent)

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
