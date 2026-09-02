"""Exact post-resolution Identity v2 OSD projection.

The processor receives fresh downstream Service Maker metadata wrappers and
never stores them. Identity is joined only through a bounded exact
camera/frame/tracker decision cache owned by :mod:`noesis.identity_v2_service`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


_OSD_LABEL_GAP_PX = 8
_OSD_DEFAULT_FONT_SIZE = 14
_CUOSD_PANGO_GLYPH_HEIGHT_EM = 1.85


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


def _osd_font_size(text_params: Any) -> int:
    font_params = getattr(text_params, "font_params", None)
    if font_params is not None:
        for attr in ("font_size", "size"):
            try:
                value = int(getattr(font_params, attr))
            except (AttributeError, TypeError, ValueError):
                continue
            if value > 0:
                return value
    return _OSD_DEFAULT_FONT_SIZE


def _estimate_osd_text_width(text: str, font_size: int) -> int:
    """Estimate the full cuOSD text box width without measuring per frame."""

    # Labels are deliberately limited to digits, a decimal point, and a space.
    # The advances match the installed Pango Serif/Sans glyph metrics. cuOSD
    # also adds a half-font-size margin on both horizontal sides.
    renderer_font_size = max(10, int(font_size))
    advance_em = sum(
        0.72
        if character.isdigit()
        else 0.86
        if character.isalpha()
        else 0.36
        for character in str(text)
    )
    glyph_width = int(math.ceil(advance_em * renderer_font_size))
    x_margin = int(renderer_font_size * 0.5)
    return max(1, glyph_width + (2 * x_margin))


def _estimate_osd_text_box_height(font_size: int) -> int:
    """Estimate the full one-line cuOSD box height for its Pango backend."""

    renderer_font_size = max(10, int(font_size))
    # cuOSD's Pango backend reports a 26 px glyph box for the configured 14 px
    # Serif font. cuOSD then adds int(font_size * 0.25) above and below it.
    glyph_height = int(
        math.ceil(renderer_font_size * _CUOSD_PANGO_GLYPH_HEIGHT_EM)
    )
    y_margin = int(renderer_font_size * 0.25)
    return glyph_height + (2 * y_margin)


def center_osd_text_above_object(obj_meta: Any, text_params: Any) -> None:
    """Place object text centered over the object's bounding box."""

    label = str(getattr(text_params, "display_text", "") or "").strip()
    rect_params = getattr(obj_meta, "rect_params", None)
    if not label or rect_params is None:
        return
    try:
        left = float(getattr(rect_params, "left"))
        top = float(getattr(rect_params, "top"))
        width = float(getattr(rect_params, "width"))
    except (AttributeError, TypeError, ValueError):
        return
    if not all(math.isfinite(value) for value in (left, top, width)) or width <= 0.0:
        return

    font_size = _osd_font_size(text_params)
    text_width = _estimate_osd_text_width(label, font_size)
    text_height = _estimate_osd_text_box_height(font_size)
    x_offset = max(0, int(round(left + (width - text_width) / 2.0)))
    y_offset = max(0, int(round(top - text_height - _OSD_LABEL_GAP_PX)))
    try:
        text_params.x_offset = x_offset
        text_params.y_offset = y_offset
    except (AttributeError, TypeError, ValueError):
        pass


def _public_identity_label(decision: Any) -> str:
    if decision is None:
        return "XX"
    state = str(getattr(decision, "identity_state", "") or "")
    try:
        sid = int(getattr(decision, "compatibility_sid", 0) or 0)
    except (TypeError, ValueError):
        sid = 0
    if state not in {"resident", "visitor"} or sid <= 0:
        return "XX"
    return str(sid)


@dataclass
class IdentityV2PostResolutionOsdProcessor:
    pipeline: Any
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int] = field(default_factory=dict)
    decimals: int = 2

    def handle_servicemaker_frame(self, frame_meta: Any) -> None:
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
        parts = [_public_identity_label(decision)]
        confidence = _confidence_text(obj_meta, self.decimals)
        if confidence:
            parts.append(confidence)
        label = " ".join(parts)
        text_params.display_text = label
        try:
            setattr(obj_meta, "obj_label", label)
        except Exception:
            pass
        center_osd_text_above_object(obj_meta, text_params)


class IdentityV2PostResolutionOsdOperator:
    """Runtime-neutral operator body wrapped by the DS9.1 adapter."""

    def __init__(self, processor: IdentityV2PostResolutionOsdProcessor) -> None:
        self.processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            self.processor.handle_servicemaker_frame(frame_meta)


__all__ = [
    "center_osd_text_above_object",
    "IdentityV2PostResolutionOsdOperator",
    "IdentityV2PostResolutionOsdProcessor",
]
