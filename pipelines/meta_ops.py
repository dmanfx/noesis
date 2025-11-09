"""
Generic DeepStream batch-metadata operator helpers shared across pipelines.

These helpers provide a thin compatibility layer around NvDsBatchMetaOperator,
falling back to legacy pyds traversal when the operator is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

try:  # DeepStream Python bindings
    import pyds  # type: ignore
except Exception:  # pragma: no cover - DS runtime optional at import time
    pyds = None  # type: ignore

try:  # DS8 operator
    from pyds import NvDsBatchMetaOperator  # type: ignore
except Exception:  # pragma: no cover
    NvDsBatchMetaOperator = None  # type: ignore


# ------------------------- Operator lifecycle -------------------------

def create_operator(gst_buffer: Any) -> Optional[Any]:
    if NvDsBatchMetaOperator is None:
        return None
    try:
        return NvDsBatchMetaOperator(gst_buffer)  # type: ignore[call-arg]
    except Exception:
        logger.debug("NvDsBatchMetaOperator construction failed", exc_info=True)
        return None


def _ensure_iterable(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        from collections.abc import Iterable  # local import
        if isinstance(value, Iterable):
            return list(value)
    except Exception:
        pass
    return [value]


# ------------------------- Traversal helpers -------------------------

def iter_frames(operator: Optional[Any], gst_buffer: Any) -> List[Any]:
    # Always try DS8 operator first
    frames: List[Any] = []
    has_operator = operator is not None
    if has_operator:
        getter = getattr(operator, "get_frames", None)
        if callable(getter):
            try:
                frames = _ensure_iterable(getter())
            except Exception:
                logger.debug("operator.get_frames failed", exc_info=True)
        if not frames:  # If get_frames didn't work or returned empty, try iteration
            try:
                frames = list(operator)  # type: ignore
            except Exception:
                pass

    # If operator path yielded results, return them
    if frames:
        return frames

    # Otherwise, fall back to legacy pyds traversal
    logger.debug("iter_frames: DS8 operator returned empty, falling back to legacy traversal (has_operator=%s, buffer_hash=%s)", has_operator, hash(gst_buffer))
    if pyds is None:
        return []
    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    except Exception:
        logger.debug("gst_buffer_get_nvds_batch_meta failed", exc_info=True)
        return []
    node = getattr(batch_meta, "frame_meta_list", None)
    while node:
        try:
            frames.append(pyds.NvDsFrameMeta.cast(node.data))  # type: ignore
        except Exception:
            break
        try:
            node = node.next
        except Exception:
            break
    return frames


def iter_objects(operator: Optional[Any], frame_meta: Any) -> List[Any]:
    # Always try DS8 operator first
    objs: List[Any] = []
    has_operator = operator is not None
    if has_operator:
        getter = getattr(operator, "get_objects", None)
        if callable(getter):
            try:
                objs = _ensure_iterable(getter(frame_meta))
            except Exception:
                logger.debug("operator.get_objects failed", exc_info=True)

    # If operator path yielded objects, return them
    if objs:
        return objs

    # Otherwise, fall back to legacy frame's obj_meta_list traversal
    logger.debug("iter_objects: DS8 operator returned zero objects, falling back to legacy traversal (has_operator=%s)", has_operator)
    if pyds is None:
        return []
    node = getattr(frame_meta, "obj_meta_list", None)
    if node is None:
        logger.debug(
            "iter_objects fallback: obj_meta_list missing/empty for frame_num=%s batch_id=%s",
            getattr(frame_meta, "frame_num", None),
            getattr(frame_meta, "batch_id", None),
        )
        return objs

    while node:
        try:
            objs.append(pyds.NvDsObjectMeta.cast(node.data))  # type: ignore
        except Exception:
            logger.debug(
                "iter_objects fallback: NvDsObjectMeta.cast failed for frame_num=%s",
                getattr(frame_meta, "frame_num", None),
                exc_info=True,
            )
            break
        try:
            node = node.next
        except Exception:
            logger.debug(
                "iter_objects fallback: advancing obj_meta_list failed for frame_num=%s",
                getattr(frame_meta, "frame_num", None),
                exc_info=True,
            )
            break
    return objs


def iter_user_meta(operator: Optional[Any], holder: Any, level: str) -> List[Any]:
    metas: List[Any] = []
    if operator is not None:
        get_list = getattr(operator, "get_user_meta_list", None)
        get_meta = getattr(operator, "get_user_meta", None)
        get_next = getattr(operator, "get_next_user_meta", None)
        if callable(get_list) and callable(get_meta):
            try:
                node = get_list(holder)
            except Exception:
                node = None
            while node:
                try:
                    metas.append(get_meta(node))
                except Exception:
                    break
                if callable(get_next):
                    try:
                        node = get_next(node)
                    except Exception:
                        break
                else:
                    break
            if metas:
                return metas
    if pyds is None:
        return []
    attr = "frame_user_meta_list" if level == "frame" else "obj_user_meta_list"
    node = getattr(holder, attr, None)
    while node:
        try:
            metas.append(pyds.NvDsUserMeta.cast(node.data))  # type: ignore
        except Exception:
            break
        try:
            node = node.next
        except Exception:
            break
    return metas


# ------------------------- Accessors -------------------------

def get_source_id(operator: Optional[Any], frame_meta: Any) -> int:
    if operator is not None:
        getter = getattr(operator, "get_source_id", None)
        if callable(getter):
            try:
                return int(getter(frame_meta))
            except Exception:
                pass
    for name in ("source_id", "pad_index"):
        try:
            return int(getattr(frame_meta, name))
        except Exception:
            continue
    return 0


def get_object_id(operator: Optional[Any], obj_meta: Any) -> int:
    if operator is not None:
        getter = getattr(operator, "get_object_id", None)
        if callable(getter):
            try:
                return int(getter(obj_meta))
            except Exception:
                pass
    try:
        return int(getattr(obj_meta, "object_id"))
    except Exception:
        return 0


def get_class_id(operator: Optional[Any], obj_meta: Any) -> int:
    if operator is not None:
        getter = getattr(operator, "get_class_id", None)
        if callable(getter):
            try:
                return int(getter(obj_meta))
            except Exception:
                pass
    try:
        return int(getattr(obj_meta, "class_id"))
    except Exception:
        return -1


def get_confidence(operator: Optional[Any], obj_meta: Any) -> float:
    if operator is not None:
        getter = getattr(operator, "get_confidence", None)
        if callable(getter):
            try:
                return float(getter(obj_meta))
            except Exception:
                pass
    try:
        return float(getattr(obj_meta, "confidence"))
    except Exception:
        return 0.0


def get_tracker_confidence(operator: Optional[Any], obj_meta: Any) -> Optional[float]:
    if operator is not None:
        getter = getattr(operator, "get_tracker_confidence", None)
        if callable(getter):
            try:
                return float(getter(obj_meta))
            except Exception:
                pass
    try:
        val = getattr(obj_meta, "tracker_confidence")
        return float(val) if val is not None else None
    except Exception:
        return None


def get_rect_params(operator: Optional[Any], obj_meta: Any) -> Optional[Any]:
    if operator is not None:
        getter = getattr(operator, "get_rect_params", None)
        if callable(getter):
            try:
                rect = getter(obj_meta)
                if rect is not None:
                    return rect
            except Exception:
                pass
    return getattr(obj_meta, "rect_params", None)


def rect_to_corners(rect: Any) -> List[tuple[float, float]]:
    try:
        left = float(getattr(rect, "left", 0.0))
        top = float(getattr(rect, "top", 0.0))
        width = float(getattr(rect, "width", 0.0))
        height = float(getattr(rect, "height", 0.0))
    except Exception:
        return []
    x2 = left + width
    y2 = top + height
    return [(left, top), (x2, top), (x2, y2), (left, y2)]


def remove_object(operator: Optional[Any], frame_meta: Any, obj_meta: Any) -> bool:
    if operator is not None:
        remover = getattr(operator, "remove_object", None)
        if callable(remover):
            try:
                remover(frame_meta, obj_meta)
                return True
            except TypeError:
                try:
                    remover(obj_meta)
                    return True
                except Exception:
                    pass
            except Exception:
                logger.debug("operator.remove_object failed", exc_info=True)
    if pyds is None:
        return False
    try:
        pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)  # type: ignore
        return True
    except Exception:
        return False


# ------------------------- Analytics casting -------------------------

def cast_analytics_obj_info(operator: Optional[Any], user_meta: Any) -> Optional[Any]:
    if operator is not None:
        caster = getattr(operator, "cast_to_analytics_obj_info", None)
        if callable(caster):
            try:
                return caster(user_meta)
            except Exception:
                pass
    if pyds is None:
        return None
    try:
        return pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)  # type: ignore
    except Exception:
        return None


def cast_analytics_frame_meta(operator: Optional[Any], user_meta: Any) -> Optional[Any]:
    if operator is not None:
        caster = getattr(operator, "cast_to_analytics_frame_meta", None)
        if callable(caster):
            try:
                return caster(user_meta)
            except Exception:
                pass
    if pyds is None:
        return None
    try:
        return pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)  # type: ignore
    except Exception:
        return None


def get_meta_type(operator: Optional[Any], user_meta: Any) -> Any:
    if operator is not None:
        getter = getattr(operator, "get_meta_type", None)
        if callable(getter):
            try:
                return getter(user_meta)
            except Exception:
                pass
    base = getattr(user_meta, "base_meta", None)
    return getattr(base, "meta_type", None)


def analytics_obj_meta_type() -> Any:
    if pyds is None:
        return "NVIDIA.DSANALYTICSOBJ.USER_META"
    try:
        return pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META")  # type: ignore
    except Exception:
        return None


def analytics_frame_meta_type() -> Any:
    if pyds is None:
        return "NVIDIA.DSANALYTICSFRAME.USER_META"
    try:
        return pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META")  # type: ignore
    except Exception:
        return None


# Optional: re-export DS8 hook operator classes, if available, to centralize API
try:  # pragma: no cover - optional integration
    from noesis.pipelines.hooks import (  # type: ignore
        _AnalyticsTelemetryOperator,
        _ExcludePruneOperator,
        _IntrinsicsOperator,
        _MapAnythingOperator,
    )
    __all__ = [
        "create_operator",
        "iter_frames",
        "iter_objects",
        "iter_user_meta",
        "get_source_id",
        "get_object_id",
        "get_class_id",
        "get_confidence",
        "get_tracker_confidence",
        "get_rect_params",
        "rect_to_corners",
        "remove_object",
        "cast_analytics_obj_info",
        "cast_analytics_frame_meta",
        "get_meta_type",
        "analytics_obj_meta_type",
        "analytics_frame_meta_type",
        "_AnalyticsTelemetryOperator",
        "_ExcludePruneOperator",
        "_IntrinsicsOperator",
        "_MapAnythingOperator",
    ]
except Exception:
    __all__ = [
        "create_operator",
        "iter_frames",
        "iter_objects",
        "iter_user_meta",
        "get_source_id",
        "get_object_id",
        "get_class_id",
        "get_confidence",
        "get_tracker_confidence",
        "get_rect_params",
        "rect_to_corners",
        "remove_object",
        "cast_analytics_obj_info",
        "cast_analytics_frame_meta",
        "get_meta_type",
        "analytics_obj_meta_type",
        "analytics_frame_meta_type",
    ]
