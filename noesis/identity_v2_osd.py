"""Exact post-resolution Identity v2 OSD projection.

The processor receives fresh downstream Service Maker metadata wrappers and
never stores them. Identity is joined only through a bounded exact
camera/frame/tracker decision cache owned by :mod:`noesis.identity_v2_service`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


_DEPTH_FRAGMENT = re.compile(r"\bz=(?:n/a|[-+]?(?:\d+(?:\.\d*)?|\.\d+)m)\b")


def _integer_attr(value: Any, *names: str, default: int = -1) -> int:
    for name in names:
        try:
            raw = getattr(value, name, None)
        except Exception:
            continue
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return int(default)


def _confidence_text(obj_meta: Any, decimals: int) -> str | None:
    try:
        confidence = float(getattr(obj_meta, "confidence", float("nan")))
    except Exception:
        return None
    if not math.isfinite(confidence) or confidence < 0.0:
        return None
    return f"{confidence:.{max(0, int(decimals))}f}"


def _public_identity_label(decision: Any) -> str:
    if decision is None:
        return "#XX"
    state = str(getattr(decision, "identity_state", "") or "")
    try:
        sid = int(getattr(decision, "compatibility_sid", 0) or 0)
    except (TypeError, ValueError):
        sid = 0
    if state not in {"resident", "visitor"} or sid <= 0:
        return "#XX"
    if state == "visitor":
        return f"#{sid}"
    display_name = " ".join(
        str(getattr(decision, "display_name", "") or "").split()
    )
    if not display_name:
        return "#XX"
    return f"#{sid} {display_name[:80]}"


@dataclass
class IdentityV2PostResolutionOsdProcessor:
    pipeline: Any
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int] = field(default_factory=dict)
    decimals: int = 2

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        """Stamp one frame using a single traversal of fresh object wrappers."""

        source_id = _integer_attr(frame_meta, "source_id", "pad_index")
        frame_id = _integer_attr(frame_meta, "frame_number", "frame_num")
        sensor_id = int(self.sensor_id_map.get(source_id, source_id))
        camera_id = str(
            self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
        ).strip()
        service = getattr(self.pipeline, "identity_v2_service", None)
        authoritative = bool(
            service is not None and getattr(service, "authoritative", False)
        )
        lookup = getattr(service, "lookup_osd_decision", None)
        if not authoritative or not callable(lookup):
            raise RuntimeError(
                "post-resolution Identity v2 OSD requires an authoritative service"
            )
        # object_items is deliberately consumed exactly once. No wrapper escapes
        # this loop or survives the callback.
        object_items = getattr(frame_meta, "object_items", None) or ()
        for obj_meta in object_items:
            if _integer_attr(obj_meta, "class_id") != 0:
                continue
            tracker_id = _integer_attr(obj_meta, "object_id")
            if source_id < 0 or frame_id < 0 or tracker_id < 0:
                self._stamp(obj_meta, None)
                continue
            decision = lookup(
                camera_id=camera_id,
                frame_id=frame_id,
                tracker_id=str(tracker_id),
            )
            self._stamp(obj_meta, decision)

    def _stamp(self, obj_meta: Any, decision: Any) -> None:
        text_params = getattr(obj_meta, "text_params", None)
        if text_params is None or not hasattr(text_params, "display_text"):
            return
        current = str(getattr(text_params, "display_text", "") or "")
        parts = [_public_identity_label(decision)]
        depth = _DEPTH_FRAGMENT.search(current)
        if depth is not None:
            parts.append(depth.group(0))
        confidence = _confidence_text(obj_meta, self.decimals)
        if confidence:
            parts.append(confidence)
        label = " ".join(parts)
        text_params.display_text = label
        try:
            setattr(obj_meta, "obj_label", label)
        except Exception:
            pass


class IdentityV2PostResolutionOsdOperator:
    """Runtime-neutral operator body wrapped by adapter-local DS8 classes."""

    def __init__(self, processor: IdentityV2PostResolutionOsdProcessor) -> None:
        self.processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            self.processor.handle_frame_ds8(frame_meta)


__all__ = [
    "IdentityV2PostResolutionOsdOperator",
    "IdentityV2PostResolutionOsdProcessor",
]
