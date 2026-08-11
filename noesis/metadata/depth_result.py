from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Tuple

JsonDict = Dict[str, Any]
MinMax = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class DepthResult:
    """Canonical frame-level depth payload shared with downstream consumers."""

    source_id: int
    frame_id: int
    ts: int  # Unix epoch seconds
    width: int
    height: int
    depth_map_ref: str  # URI or key where the depth map is stored
    minmax: MinMax
    unit: str = "m"

    def __post_init__(self) -> None:
        source_id = int(self.source_id)
        frame_id = int(self.frame_id)
        ts = int(self.ts)
        width = int(self.width)
        height = int(self.height)
        if width <= 0 or height <= 0:
            raise ValueError("DepthResult width and height must be positive")
        depth_map_ref = str(self.depth_map_ref).strip()
        if not depth_map_ref:
            raise ValueError("DepthResult depth_map_ref must be a non-empty string")
        minmax = _coerce_minmax(self.minmax)
        unit = str(self.unit or "m")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "ts", ts)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "depth_map_ref", depth_map_ref)
        object.__setattr__(self, "minmax", minmax)
        object.__setattr__(self, "unit", unit)

    def to_dict(self) -> JsonDict:
        """Return a JSON-serialisable payload."""
        return {
            "source_id": self.source_id,
            "frame_id": self.frame_id,
            "ts": self.ts,
            "width": self.width,
            "height": self.height,
            "depth_map_ref": self.depth_map_ref,
            "minmax": [self.minmax[0], self.minmax[1]],
            "unit": self.unit,
        }

    def to_public_dict(self) -> JsonDict:
        """Return telemetry without exposing storage paths or backend URIs."""

        payload = self.to_dict()
        digest = hashlib.sha256(self.depth_map_ref.encode("utf-8")).hexdigest()
        payload["depth_map_ref"] = f"noesis-depth://artifact/{digest}"
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        if indent is None:
            return json.dumps(self.to_dict(), separators=(",", ":"))
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DepthResult":
        """Instantiate from a dictionary payload (e.g. decoded JSON)."""
        data = dict(payload)
        if "ts" not in data:
            raise KeyError("DepthResult payload missing 'ts'")
        ts_value = data["ts"]
        if isinstance(ts_value, datetime):
            ts_value = _datetime_to_epoch_s(ts_value)
        data["ts"] = int(ts_value)
        for field_name in ("source_id", "frame_id", "width", "height"):
            if field_name not in data:
                raise KeyError(f"DepthResult payload missing '{field_name}'")
            data[field_name] = int(data[field_name])

        depth_map_ref = data.get("depth_map_ref")
        if depth_map_ref is None:
            raise KeyError("DepthResult payload missing 'depth_map_ref'")
        data["depth_map_ref"] = str(depth_map_ref)
        data["minmax"] = _coerce_minmax(data.get("minmax", (0.0, 0.0)))
        data["unit"] = str(data.get("unit") or "m")
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "DepthResult":
        """Instantiate from a JSON string."""
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("DepthResult JSON payload must decode to a dict")
        return cls.from_dict(decoded)


def _coerce_minmax(value: Any) -> MinMax:
    if isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise KeyError("minmax mapping must contain 'min' and 'max'")
        return (float(value["min"]), float(value["max"]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ValueError("minmax sequence must contain exactly two values")
        return (float(items[0]), float(items[1]))
    raise TypeError("minmax must be a 2-tuple/list or mapping with 'min'/'max'")


def _datetime_to_epoch_s(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())
