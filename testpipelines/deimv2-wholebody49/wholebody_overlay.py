"""Wholebody49 tensor summary and overlay probe."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack
from pyservicemaker import BatchMetadataOperator, osd

try:
    from .model_setup import GIE_UNIQUE_ID, LABELS_PATH, load_classes
except ImportError:  # pragma: no cover - script execution path
    from model_setup import GIE_UNIQUE_ID, LABELS_PATH, load_classes


LOGGER = logging.getLogger(__name__)

BODY_CLASS_ID = 0
BONE_CLASS_ID = 48
OBJECT_CLASS_IDS = {0, 5, 6, 7, 16, 17, 18, 19, 20, 32, 33, 34, 45, 46, 47, BONE_CLASS_ID}
ATTRIBUTE_CLASS_IDS = {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15}
KEYPOINT_CLASS_IDS = {
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
}
LEFT_SIDE_CLASS_IDS = {23, 27, 30, 33, 37, 40, 43, 46}
RIGHT_SIDE_CLASS_IDS = {24, 28, 31, 34, 38, 41, 44, 47}
SIDE_ATTR_CLASS_IDS = {23, 24, 27, 28, 30, 31, 33, 34, 37, 38, 40, 41, 43, 44, 46, 47}

SKELETON_EDGES = (
    (21, 22),
    (21, 22),
    (21, 25),
    (22, 26),
    (22, 26),
    (26, 29),
    (26, 29),
    (29, 32),
    (29, 32),
    (22, 36),
    (22, 36),
    (25, 35),
    (35, 36),
    (35, 36),
    (36, 39),
    (36, 39),
    (39, 42),
    (39, 42),
    (42, 45),
    (42, 45),
)
BONE_EDGE_PAIRS = (
    (21, 23),
    (21, 24),
    (21, 25),
    (25, 35),
    (23, 27),
    (27, 30),
    (24, 28),
    (28, 31),
    (35, 37),
    (37, 40),
    (40, 43),
    (39, 42),
    (35, 38),
    (38, 41),
    (41, 44),
)
SKELETON_KEYPOINT_IDS = {21, 22, 25, 26, 29, 32, 35, 36, 39, 42, 45}
BONE_RENDER_KEYPOINT_IDS = {class_id for edge in BONE_EDGE_PAIRS for class_id in edge}
SKELETON_ASSIGNMENT_KEYPOINT_IDS = SKELETON_KEYPOINT_IDS | BONE_RENDER_KEYPOINT_IDS
SIDE_PARENT_TO_CHILDREN = {
    22: (23, 24),
    26: (27, 28),
    29: (30, 31),
    32: (33, 34),
    36: (37, 38),
    39: (40, 41),
    42: (43, 44),
    45: (46, 47),
}
SIDE_AWARE_SKELETON_CLASS_IDS = set(SIDE_PARENT_TO_CHILDREN.keys())


def _unique_undirected_edges(edges: Sequence[Tuple[int, int]]) -> set[Tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]}


SKELETON_CONNECTION_DEGREE_LIMITS = Counter(
    class_id
    for edge in _unique_undirected_edges(tuple(SKELETON_EDGES) + tuple(BONE_EDGE_PAIRS))
    for class_id in edge
)
SKELETON_NATURAL_CONNECTION_KEYS = _unique_undirected_edges(tuple(SKELETON_EDGES) + tuple(BONE_EDGE_PAIRS))
SKELETON_NATURAL_NEIGHBOR_CLASS_IDS: Dict[int, set[int]] = {}
for _first_class_id, _second_class_id in SKELETON_NATURAL_CONNECTION_KEYS:
    SKELETON_NATURAL_NEIGHBOR_CLASS_IDS.setdefault(_first_class_id, set()).add(_second_class_id)
    SKELETON_NATURAL_NEIGHBOR_CLASS_IDS.setdefault(_second_class_id, set()).add(_first_class_id)

CLASS_COLORS = {
    "body": (0.0, 0.85, 0.25, 1.0),
    "object": (0.1, 0.75, 1.0, 1.0),
    "keypoint": (1.0, 0.2, 0.15, 1.0),
    "attribute": (1.0, 0.75, 0.1, 1.0),
    "bone": (0.85, 0.25, 1.0, 1.0),
}
ORPHAN_OBJECT_SCORE_THRESHOLD = 0.65
ORPHAN_KEYPOINT_SCORE_THRESHOLD = 0.65
ORPHAN_BONE_SCORE_THRESHOLD = 0.70
BODY_CONTEXT_PADDING_PX = 35.0
KEYPOINT_CONTEXT_PADDING_PX = 6.0


@dataclass
class Detection:
    class_id: int
    class_name: str
    score: float
    bbox: Tuple[float, float, float, float]
    source_idx: int
    group: str
    generation: Optional[str] = None
    gender: Optional[str] = None
    handedness: Optional[str] = None
    head_pose: Optional[str] = None
    related_to: Optional[int] = None
    has_instance_mask: bool = True
    track_id: Optional[int] = None
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_json(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "group": self.group,
            "score": self.score,
            "bbox_xyxy": [float(v) for v in self.bbox],
            "source_idx": self.source_idx,
            "has_instance_mask": self.has_instance_mask,
        }
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        if self.related_to is not None:
            payload["related_to_source_idx"] = self.related_to
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        return payload


class SkeletonLineRegistry:
    def __init__(self) -> None:
        self.line_keys: set[Tuple[int, int]] = set()
        self.endpoint_counts: Counter[int] = Counter()
        self.endpoint_neighbor_slot_counts: Counter[Tuple[int, int, int]] = Counter()

    @staticmethod
    def line_key(first: Detection, second: Detection) -> Tuple[int, int]:
        first_id = id(first)
        second_id = id(second)
        if first_id < second_id:
            return first_id, second_id
        return second_id, first_id

    @staticmethod
    def endpoint_limit(det: Detection) -> int:
        return max(1, int(SKELETON_CONNECTION_DEGREE_LIMITS.get(det.class_id, 1)))

    @staticmethod
    def endpoint_neighbor_slot_key(endpoint: Detection, neighbor: Detection) -> Optional[Tuple[int, int, int]]:
        if neighbor.class_id not in SKELETON_NATURAL_NEIGHBOR_CLASS_IDS.get(endpoint.class_id, ()):
            return None
        neighbor_side_slot = -1
        if endpoint.class_id not in SIDE_AWARE_SKELETON_CLASS_IDS:
            neighbor_side = _detection_side(neighbor)
            if neighbor_side == "left":
                neighbor_side_slot = 0
            elif neighbor_side == "right":
                neighbor_side_slot = 1
        return (id(endpoint), neighbor.class_id, neighbor_side_slot)

    @classmethod
    def endpoint_neighbor_slot_keys(cls, first: Detection, second: Detection) -> Tuple[Tuple[int, int, int], ...]:
        slot_keys = []
        first_slot = cls.endpoint_neighbor_slot_key(first, second)
        if first_slot is not None:
            slot_keys.append(first_slot)
        second_slot = cls.endpoint_neighbor_slot_key(second, first)
        if second_slot is not None:
            slot_keys.append(second_slot)
        return tuple(slot_keys)

    def has_line(self, first: Detection, second: Detection) -> bool:
        return self.line_key(first, second) in self.line_keys

    def has_neighbor_slot_capacity(self, first: Detection, second: Detection) -> bool:
        return all(
            self.endpoint_neighbor_slot_counts[slot_key] < 1
            for slot_key in self.endpoint_neighbor_slot_keys(first, second)
        )

    def connection_slot_priority(self, first: Detection, second: Detection) -> int:
        return 1 if self.has_neighbor_slot_capacity(first, second) else 0

    def can_add(self, first: Detection, second: Detection) -> bool:
        if self.has_line(first, second):
            return False
        return (
            self.endpoint_counts[id(first)] < self.endpoint_limit(first)
            and self.endpoint_counts[id(second)] < self.endpoint_limit(second)
            and self.has_neighbor_slot_capacity(first, second)
        )

    def add(self, first: Detection, second: Detection) -> bool:
        if not self.can_add(first, second):
            return False
        self.line_keys.add(self.line_key(first, second))
        self.endpoint_counts[id(first)] += 1
        self.endpoint_counts[id(second)] += 1
        for slot_key in self.endpoint_neighbor_slot_keys(first, second):
            self.endpoint_neighbor_slot_counts[slot_key] += 1
        return True


def _sanitize_component(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return collapsed.strip("._-") or "source"


def _to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    dlpack_fn = getattr(tensor, "__dlpack__", None)
    if not callable(dlpack_fn):
        return np.asarray(tensor)
    stream = 0
    if torch.cuda.is_available():
        stream = int(torch.cuda.current_stream().cuda_stream)
    capsule = dlpack_fn(stream)
    torch_tensor = torch_dlpack.from_dlpack(capsule)
    return torch_tensor.detach().cpu().numpy()


def _intrinsic_side_for_class(class_id: int) -> Optional[str]:
    if class_id in LEFT_SIDE_CLASS_IDS:
        return "left"
    if class_id in RIGHT_SIDE_CLASS_IDS:
        return "right"
    return None


def _detection_side(det: Detection) -> Optional[str]:
    return det.attributes.get("side") or _intrinsic_side_for_class(det.class_id)


def _apply_intrinsic_sides(detections: Iterable[Detection]) -> None:
    for det in detections:
        side = _intrinsic_side_for_class(det.class_id)
        if side:
            det.attributes.setdefault("side", side)


def class_group(class_id: int) -> str:
    if class_id == BODY_CLASS_ID:
        return "body"
    if class_id == BONE_CLASS_ID:
        return "bone"
    if class_id in KEYPOINT_CLASS_IDS:
        return "keypoint"
    if class_id in ATTRIBUTE_CLASS_IDS:
        return "attribute"
    if class_id in OBJECT_CLASS_IDS:
        return "object"
    return "object"


def _threshold_for_group(group: str, thresholds: Dict[str, float]) -> float:
    if group == "body" or group == "object" or group == "bone":
        return float(thresholds.get("object", 0.35))
    if group == "attribute":
        return float(thresholds.get("attribute", 0.35))
    if group == "keypoint":
        return float(thresholds.get("keypoint", 0.35))
    return float(thresholds.get("object", 0.35))


def _copy_detection(
    det: Detection,
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    score: Optional[float] = None,
    track_id: Optional[int] = None,
) -> Detection:
    return Detection(
        class_id=det.class_id,
        class_name=det.class_name,
        score=det.score if score is None else float(score),
        bbox=det.bbox if bbox is None else bbox,
        source_idx=det.source_idx,
        group=det.group,
        generation=det.generation,
        gender=det.gender,
        handedness=det.handedness,
        head_pose=det.head_pose,
        related_to=det.related_to,
        has_instance_mask=det.has_instance_mask,
        track_id=det.track_id if track_id is None else track_id,
        attributes=dict(det.attributes),
    )


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = _bbox_area(first) + _bbox_area(second) - inter
    return inter / union if union > 0 else 0.0


def _iou(a: Detection, b: Detection) -> float:
    return _bbox_iou(a.bbox, b.bbox)


def _center_distance(a: Detection, b: Detection) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by)


def _find_most_relevant(base_boxes: Iterable[Detection], attr_boxes: Iterable[Detection]) -> None:
    used: set[int] = set()
    for base in base_boxes:
        best: Optional[Tuple[float, float, float, Detection]] = None
        for attr in attr_boxes:
            if attr.source_idx in used:
                continue
            dist = _center_distance(base, attr)
            if dist > 10.0:
                continue
            iou = _iou(base, attr)
            if iou <= 0.0:
                continue
            candidate = (attr.score, iou, -dist, attr)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None:
            continue
        attr = best[3]
        used.add(attr.source_idx)
        attr.related_to = base.source_idx
        if attr.class_id == 1:
            base.attributes["generation"] = "adult"
        elif attr.class_id == 2:
            base.attributes["generation"] = "child"
        elif attr.class_id == 3:
            base.attributes["gender"] = "male"
        elif attr.class_id == 4:
            base.attributes["gender"] = "female"
        elif 8 <= attr.class_id <= 15:
            base.attributes["head_pose"] = attr.class_name
        elif attr.class_id in SIDE_ATTR_CLASS_IDS:
            side = _intrinsic_side_for_class(attr.class_id)
            if side:
                base.attributes["side"] = side


def attach_attributes(detections: List[Detection]) -> None:
    _apply_intrinsic_sides(detections)
    bodies = [det for det in detections if det.class_id == BODY_CLASS_ID]
    heads = [det for det in detections if det.class_id == 7]
    _find_most_relevant(bodies, [det for det in detections if det.class_id in (1, 2)])
    _find_most_relevant(bodies, [det for det in detections if det.class_id in (3, 4)])
    _find_most_relevant(heads, [det for det in detections if 8 <= det.class_id <= 15])
    for parent_id, child_ids in SIDE_PARENT_TO_CHILDREN.items():
        _find_most_relevant(
            [det for det in detections if det.class_id == parent_id],
            [det for det in detections if det.class_id in child_ids],
        )


def nms_keypoints(detections: List[Detection], iou_threshold: float = 0.20) -> List[Detection]:
    kept: List[Detection] = [det for det in detections if det.group != "keypoint"]
    for class_id in sorted({det.class_id for det in detections if det.group == "keypoint"}):
        boxes = sorted((det for det in detections if det.class_id == class_id), key=lambda det: det.score, reverse=True)
        while boxes:
            current = boxes.pop(0)
            kept.append(current)
            boxes = [det for det in boxes if _iou(current, det) < iou_threshold]
    return kept


def _has_body_context(det: Detection, bodies: Sequence[Detection], padding: float) -> bool:
    return any(_keypoint_inside_box(det, body, padding=padding) or _iou(det, body) > 0.005 for body in bodies)


def refine_detections(
    detections: Sequence[Detection],
    *,
    object_score_threshold: float,
    attribute_score_threshold: float,
    keypoint_threshold: float,
    keypoint_candidate_threshold: Optional[float] = None,
) -> List[Detection]:
    keypoint_candidate_threshold = (
        float(keypoint_threshold)
        if keypoint_candidate_threshold is None
        else min(float(keypoint_threshold), float(keypoint_candidate_threshold))
    )
    bodies = [det for det in detections if det.class_id == BODY_CLASS_ID]
    refined: List[Detection] = []
    for det in detections:
        if det.class_id == BODY_CLASS_ID:
            refined.append(det)
            continue
        if det.group == "attribute":
            if det.related_to is not None and det.score >= attribute_score_threshold:
                refined.append(det)
            continue
        if det.group == "keypoint":
            if bodies and _has_body_context(det, bodies, KEYPOINT_CONTEXT_PADDING_PX):
                if det.score >= keypoint_candidate_threshold:
                    refined.append(det)
            elif det.score >= max(float(keypoint_threshold), ORPHAN_KEYPOINT_SCORE_THRESHOLD):
                refined.append(det)
            continue
        if det.group == "bone":
            if bodies and _has_body_context(det, bodies, BODY_CONTEXT_PADDING_PX):
                if det.score >= object_score_threshold:
                    refined.append(det)
            elif det.score >= max(float(object_score_threshold), ORPHAN_BONE_SCORE_THRESHOLD):
                refined.append(det)
            continue
        if bodies and _has_body_context(det, bodies, BODY_CONTEXT_PADDING_PX):
            if det.score >= object_score_threshold:
                refined.append(det)
        elif det.score >= max(float(object_score_threshold), ORPHAN_OBJECT_SCORE_THRESHOLD):
            refined.append(det)
    return refined


@dataclass
class DetectionTrack:
    track_id: int
    class_id: int
    group: str
    bbox: Tuple[float, float, float, float]
    score: float
    hits: int = 1
    missed: int = 0


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_diagonal(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1))


def _ema_bbox(
    previous: Tuple[float, float, float, float],
    current: Tuple[float, float, float, float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    return tuple(
        float((1.0 - alpha) * old_value + alpha * new_value)
        for old_value, new_value in zip(previous, current)
    )  # type: ignore[return-value]


class DetectionSmoother:
    def __init__(
        self,
        *,
        alpha: float = 0.65,
        max_missed: int = 5,
    ) -> None:
        self.alpha = max(0.05, min(1.0, float(alpha)))
        self.max_missed = max(0, int(max_missed))
        self._tracks_by_source: Dict[int, Dict[int, DetectionTrack]] = {}
        self._next_track_id = 1

    @staticmethod
    def _match_score(det: Detection, track: DetectionTrack) -> Optional[float]:
        if det.class_id != track.class_id:
            return None
        iou = _bbox_iou(det.bbox, track.bbox)
        det_cx, det_cy = det.center
        track_cx, track_cy = _bbox_center(track.bbox)
        center_distance = math.hypot(det_cx - track_cx, det_cy - track_cy)
        scale = max(18.0, min(160.0, max(_bbox_diagonal(det.bbox), _bbox_diagonal(track.bbox)) * 0.35))
        if det.group == "body":
            scale = max(scale, min(220.0, _bbox_diagonal(track.bbox) * 0.25))
        elif det.group in {"keypoint", "bone"}:
            scale = max(scale, 32.0)
        if iou < 0.05 and center_distance > scale:
            return None
        proximity = max(0.0, 1.0 - center_distance / max(1.0, scale))
        return iou * 4.0 + proximity + min(0.20, track.hits * 0.02)

    def _new_track(self, det: Detection) -> DetectionTrack:
        track = DetectionTrack(
            track_id=self._next_track_id,
            class_id=det.class_id,
            group=det.group,
            bbox=det.bbox,
            score=det.score,
        )
        self._next_track_id += 1
        return track

    def update(self, source_id: int, detections: Sequence[Detection]) -> List[Detection]:
        tracks = self._tracks_by_source.setdefault(int(source_id), {})
        matched_track_ids: set[int] = set()
        smoothed: List[Detection] = []

        for det in detections:
            best_track: Optional[DetectionTrack] = None
            best_score: Optional[float] = None
            for track in tracks.values():
                if track.track_id in matched_track_ids:
                    continue
                score = self._match_score(det, track)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_track = track

            if best_track is None:
                best_track = self._new_track(det)
                tracks[best_track.track_id] = best_track
                bbox = det.bbox
            else:
                bbox = _ema_bbox(best_track.bbox, det.bbox, self.alpha)
                best_track.bbox = bbox
                best_track.score = det.score
                best_track.hits += 1
                best_track.missed = 0

            matched_track_ids.add(best_track.track_id)
            smoothed.append(_copy_detection(det, bbox=bbox, track_id=best_track.track_id))

        for track_id in list(tracks):
            if track_id in matched_track_ids:
                continue
            tracks[track_id].missed += 1
            if tracks[track_id].missed > self.max_missed:
                del tracks[track_id]
        return smoothed


def decode_detections(
    label_xyxy_score: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    class_names: Sequence[str],
    object_score_threshold: float = 0.35,
    attribute_score_threshold: float = 0.35,
    keypoint_threshold: float = 0.35,
    keypoint_candidate_threshold: Optional[float] = None,
    class_aware_filtering: bool = True,
    max_detections: int = 300,
    has_instance_masks: bool = True,
) -> List[Detection]:
    output = np.asarray(label_xyxy_score, dtype=np.float32)
    if output.ndim != 2 or output.shape[1] < 6:
        raise RuntimeError(f"Expected label_xyxy_score [queries, 6], got {output.shape}")

    coord_max = float(np.nanmax(np.abs(output[:, 1:5]))) if output.size else 0.0
    normalized = coord_max <= 2.0
    thresholds = {
        "object": object_score_threshold,
        "attribute": attribute_score_threshold,
        "keypoint": min(
            float(keypoint_threshold),
            float(keypoint_threshold if keypoint_candidate_threshold is None else keypoint_candidate_threshold),
        ),
    }
    detections: List[Detection] = []
    for source_idx, row in enumerate(output):
        class_id = int(round(float(row[0])))
        if class_id < 0 or class_id >= len(class_names):
            continue
        group = class_group(class_id)
        score = float(row[5])
        if score < _threshold_for_group(group, thresholds):
            continue
        x1, y1, x2, y2 = [float(v) for v in row[1:5]]
        if normalized:
            x1 *= float(frame_width)
            x2 *= float(frame_width)
            y1 *= float(frame_height)
            y2 *= float(frame_height)
        x1 = max(0.0, min(x1, float(frame_width - 1)))
        y1 = max(0.0, min(y1, float(frame_height - 1)))
        x2 = max(0.0, min(x2, float(frame_width - 1)))
        y2 = max(0.0, min(y2, float(frame_height - 1)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(class_names[class_id]),
                score=score,
                bbox=(x1, y1, x2, y2),
                source_idx=source_idx,
                group=group,
                has_instance_mask=bool(has_instance_masks and class_id == BODY_CLASS_ID),
            )
        )

    detections.sort(key=lambda det: det.score, reverse=True)
    if max_detections > 0:
        detections = detections[:max_detections]
    attach_attributes(detections)
    if class_aware_filtering:
        detections = refine_detections(
            detections,
            object_score_threshold=object_score_threshold,
            attribute_score_threshold=attribute_score_threshold,
            keypoint_threshold=keypoint_threshold,
            keypoint_candidate_threshold=keypoint_candidate_threshold,
        )
    return nms_keypoints(detections)


def summarize_detections(detections: Sequence[Detection]) -> Dict[str, object]:
    by_group = Counter(det.group for det in detections)
    by_class = Counter(det.class_name for det in detections)
    return {
        "total": len(detections),
        "by_group": dict(sorted(by_group.items())),
        "by_class": dict(sorted(by_class.items())),
        "bodies": sum(1 for det in detections if det.class_id == BODY_CLASS_ID),
        "mask_capable": sum(1 for det in detections if det.has_instance_mask),
    }


def _keypoint_inside_box(keypoint: Detection, container: Detection, padding: float = 1.0) -> bool:
    x1, y1, x2, y2 = container.bbox
    cx, cy = keypoint.center
    return x1 - padding <= cx <= x2 + padding and y1 - padding <= cy <= y2 + padding


def _body_diagonal(body: Detection) -> float:
    x1, y1, x2, y2 = body.bbox
    return math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1))


def _assign_keypoints_to_bodies(
    detections: Sequence[Detection],
) -> Tuple[Dict[int, int], Dict[int, Detection]]:
    bodies = [det for det in detections if det.class_id == BODY_CLASS_ID]
    body_by_source_idx = {det.source_idx: det for det in bodies}
    assignments: Dict[int, int] = {}
    if not bodies:
        return assignments, body_by_source_idx

    for det in detections:
        if det.class_id not in SKELETON_ASSIGNMENT_KEYPOINT_IDS:
            continue
        containing = [body for body in bodies if _keypoint_inside_box(det, body, padding=2.0)]
        if not containing:
            continue
        body = sorted(containing, key=lambda candidate: (candidate.area, -candidate.score, candidate.source_idx))[0]
        assignments[id(det)] = body.source_idx
    return assignments, body_by_source_idx


def _selected_skeleton_keypoints(
    detections: Sequence[Detection],
    assignments: Dict[int, int],
    body_by_source_idx: Dict[int, Detection],
) -> List[Detection]:
    selected: Dict[Tuple[int, int, str], Tuple[Detection, Tuple[float, float, float]]] = {}
    for det in detections:
        if det.class_id not in SKELETON_ASSIGNMENT_KEYPOINT_IDS:
            continue
        body_source_idx = assignments.get(id(det))
        if body_source_idx is None:
            continue
        body = body_by_source_idx.get(body_source_idx)
        if body is None or body.area <= 0:
            continue
        side = _detection_side(det) or ""
        area_ratio = det.area / max(1.0, body.area)
        body_cx, body_cy = body.center
        det_cx, det_cy = det.center
        normalized_center_distance = math.hypot(det_cx - body_cx, det_cy - body_cy) / max(1.0, _body_diagonal(body))
        key = (body_source_idx, det.class_id, side)
        priority = (area_ratio, det.score, -normalized_center_distance)
        current = selected.get(key)
        if current is None or priority > current[1]:
            selected[key] = (det, priority)
    return [candidate[0] for candidate in selected.values()]


def _same_assigned_body(first: Detection, second: Detection, assignments: Dict[int, int]) -> Optional[int]:
    first_body = assignments.get(id(first))
    second_body = assignments.get(id(second))
    if first_body is None or second_body is None or first_body != second_body:
        return None
    return first_body


def _side_compatible(first: Detection, second: Detection) -> bool:
    first_side_aware = first.class_id in SIDE_AWARE_SKELETON_CLASS_IDS
    second_side_aware = second.class_id in SIDE_AWARE_SKELETON_CLASS_IDS
    first_side = _detection_side(first)
    second_side = _detection_side(second)

    if first_side_aware and first_side is None:
        return False
    if second_side_aware and second_side is None:
        return False
    if first_side_aware and second_side_aware:
        return first_side == second_side
    return True


def _edge_distance_ok(
    first: Detection,
    second: Detection,
    body_source_idx: int,
    body_by_source_idx: Dict[int, Detection],
) -> bool:
    body = body_by_source_idx.get(body_source_idx)
    if body is None:
        max_dist = 500.0
    else:
        max_dist = max(50.0, min(500.0, _body_diagonal(body) * 0.80))
    first_cx, first_cy = first.center
    second_cx, second_cy = second.center
    return (first_cx - second_cx) ** 2 + (first_cy - second_cy) ** 2 <= max_dist * max_dist


def _bone_edge_score(
    bone: Detection,
    first: Detection,
    second: Detection,
    assignments: Dict[int, int],
    registry: SkeletonLineRegistry,
) -> Optional[float]:
    if not registry.can_add(first, second):
        return None
    if _same_assigned_body(first, second, assignments) is None:
        return None
    if not _side_compatible(first, second):
        return None
    if not _keypoint_inside_box(first, bone) or not _keypoint_inside_box(second, bone):
        return None

    x1, y1, x2, y2 = bone.bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    first_cx, first_cy = first.center
    second_cx, second_cy = second.center
    dx = abs(first_cx - second_cx)
    dy = abs(first_cy - second_cy)
    long_axis = max(width, height)
    short_axis = max(1.0, min(width, height))
    long_axis_separation = max(dx, dy) / long_axis
    if long_axis_separation < 0.45:
        return None

    aspect_ratio = long_axis / short_axis
    if aspect_ratio < 1.8 and (dx / width < 0.30 or dy / height < 0.30):
        return None

    midpoint_x = (first_cx + second_cx) * 0.5
    midpoint_y = (first_cy + second_cy) * 0.5
    bone_center_x = (x1 + x2) * 0.5
    bone_center_y = (y1 + y2) * 0.5
    center_offset = math.hypot((midpoint_x - bone_center_x) / width, (midpoint_y - bone_center_y) / height)
    if center_offset > 0.35:
        return None
    return float(bone.score) * 1000.0 + long_axis_separation - center_offset


def _add_bone_supported_lines(
    *,
    lines: List[Tuple[Detection, Detection]],
    registry: SkeletonLineRegistry,
    bones: Sequence[Detection],
    classid_to_keypoints: Dict[int, List[Detection]],
    edge_pairs: Sequence[Tuple[int, int]],
    assignments: Dict[int, int],
) -> None:
    candidate_order = 0
    candidates: List[Tuple[int, float, int, int, Detection, Detection]] = []
    for bone_idx, bone in enumerate(bones):
        for first_id, second_id in edge_pairs:
            for first in classid_to_keypoints.get(first_id, ()):
                if not _keypoint_inside_box(first, bone):
                    continue
                for second in classid_to_keypoints.get(second_id, ()):
                    score = _bone_edge_score(bone, first, second, assignments, registry)
                    if score is None:
                        candidate_order += 1
                        continue
                    slot_priority = registry.connection_slot_priority(first, second)
                    candidates.append((slot_priority, score, candidate_order, bone_idx, first, second))
                    candidate_order += 1

    used_bones: set[int] = set()
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for _, _, _, bone_idx, first, second in candidates:
        if bone_idx in used_bones:
            continue
        if registry.add(first, second):
            used_bones.add(bone_idx)
            lines.append((first, second))


def _add_instance_skeleton_lines(
    *,
    lines: List[Tuple[Detection, Detection]],
    registry: SkeletonLineRegistry,
    classid_to_keypoints: Dict[int, List[Detection]],
    assignments: Dict[int, int],
    body_by_source_idx: Dict[int, Detection],
) -> None:
    for (parent_id, child_id), repeat_count in Counter(SKELETON_EDGES).items():
        parent_list = classid_to_keypoints.get(parent_id, [])
        child_list = classid_to_keypoints.get(child_id, [])
        if not parent_list or not child_list:
            continue

        parent_capacity = {
            id(parent): 1 if _detection_side(parent) else repeat_count
            for parent in parent_list
        }
        child_used: set[int] = set()
        pair_candidates: List[Tuple[float, int, int, Detection, Detection]] = []

        for parent_idx, parent in enumerate(parent_list):
            for child_idx, child in enumerate(child_list):
                body_source_idx = _same_assigned_body(parent, child, assignments)
                if body_source_idx is None:
                    continue
                if not _side_compatible(parent, child):
                    continue
                if not _edge_distance_ok(parent, child, body_source_idx, body_by_source_idx):
                    continue
                if not registry.can_add(parent, child):
                    continue
                parent_cx, parent_cy = parent.center
                child_cx, child_cy = child.center
                dist_sq = (parent_cx - child_cx) ** 2 + (parent_cy - child_cy) ** 2
                pair_candidates.append((dist_sq, parent_idx, child_idx, parent, child))

        pair_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, _, _, parent, child in pair_candidates:
            if parent_capacity.get(id(parent), 0) <= 0 or id(child) in child_used:
                continue
            if registry.add(parent, child):
                parent_capacity[id(parent)] -= 1
                child_used.add(id(child))
                lines.append((parent, child))


def skeleton_lines(detections: Sequence[Detection]) -> List[Tuple[Detection, Detection]]:
    assignments, body_by_source_idx = _assign_keypoints_to_bodies(detections)
    if not assignments:
        return []

    selected_keypoints = _selected_skeleton_keypoints(detections, assignments, body_by_source_idx)
    if not selected_keypoints:
        return []

    classid_to_keypoints: Dict[int, List[Detection]] = {}
    for det in selected_keypoints:
        classid_to_keypoints.setdefault(det.class_id, []).append(det)

    lines: List[Tuple[Detection, Detection]] = []
    registry = SkeletonLineRegistry()
    bones = [det for det in detections if det.class_id == BONE_CLASS_ID]
    if bones:
        _add_bone_supported_lines(
            lines=lines,
            registry=registry,
            bones=bones,
            classid_to_keypoints=classid_to_keypoints,
            edge_pairs=BONE_EDGE_PAIRS,
            assignments=assignments,
        )
        _add_bone_supported_lines(
            lines=lines,
            registry=registry,
            bones=bones,
            classid_to_keypoints=classid_to_keypoints,
            edge_pairs=tuple(dict.fromkeys(SKELETON_EDGES)),
            assignments=assignments,
        )

    _add_instance_skeleton_lines(
        lines=lines,
        registry=registry,
        classid_to_keypoints=classid_to_keypoints,
        assignments=assignments,
        body_by_source_idx=body_by_source_idx,
    )
    return lines


def write_summary_json(
    *,
    output_root: Path,
    sensor_id: str,
    sensor_name: str,
    frame_number: int,
    detections: Sequence[Detection],
    model_info: Dict[str, object],
    reused_tensor_cache: bool = False,
) -> Path:
    sensor_dir = output_root / f"{_sanitize_component(sensor_id)}_{_sanitize_component(sensor_name)}"
    sensor_dir.mkdir(parents=True, exist_ok=True)
    output_path = sensor_dir / "latest_deimv2_wholebody49.json"
    payload = {
        "schema_version": 1,
        "sensor_id": sensor_id,
        "sensor_name": sensor_name,
        "frame_number": int(frame_number),
        "model": model_info.get("model", {}),
        "outputs": model_info.get("outputs", {}),
        "reused_tensor_cache": bool(reused_tensor_cache),
        "summary": summarize_detections(detections),
        "skeleton_line_count": len(skeleton_lines(detections)),
        "detections": [det.to_json() for det in detections],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


class Deimv2WholebodyOverlay(BatchMetadataOperator):
    def __init__(
        self,
        *,
        output_root: Path,
        model_info: Dict[str, object],
        sensor_ids: Sequence[str],
        sensor_names: Sequence[str],
        gie_id: int = GIE_UNIQUE_ID,
        object_score_threshold: float = 0.50,
        attribute_score_threshold: float = 0.75,
        keypoint_threshold: float = 0.50,
        emit_every_frames: int = 15,
        max_detections: int = 300,
        max_draw: int = 300,
        has_instance_masks: bool = True,
        class_aware_filtering: bool = True,
        enable_smoothing: bool = True,
        smoothing_alpha: float = 0.65,
        reuse_last_on_missing_tensor: bool = False,
    ) -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.model_info = model_info
        self.sensor_ids = [str(x) for x in sensor_ids]
        self.sensor_names = [str(x) for x in sensor_names]
        self.gie_id = int(gie_id)
        self.class_names = load_classes(LABELS_PATH)
        self.object_score_threshold = float(object_score_threshold)
        self.attribute_score_threshold = float(attribute_score_threshold)
        self.keypoint_threshold = float(keypoint_threshold)
        self.keypoint_candidate_threshold = min(
            self.keypoint_threshold,
            max(0.35, self.keypoint_threshold - 0.10),
        )
        self.emit_every_frames = max(1, int(emit_every_frames))
        self.max_detections = max(1, int(max_detections))
        self.max_draw = max(1, int(max_draw))
        self.has_instance_masks = bool(has_instance_masks)
        self.class_aware_filtering = bool(class_aware_filtering)
        self.reuse_last_on_missing_tensor = bool(reuse_last_on_missing_tensor)
        self._smoother = DetectionSmoother(alpha=smoothing_alpha) if enable_smoothing else None
        self._logged_layers = False
        self._logged_tensor_cache_reuse = False
        self._last_report_s = 0.0
        self._frame_counts: Counter[int] = Counter()
        self._latest_paths: Dict[int, Path] = {}
        self._last_detections_by_source: Dict[int, List[Detection]] = {}
        self._tensor_cache_reuse_counts: Counter[int] = Counter()

    def _frame_size(self, frame_meta: Any) -> Tuple[int, int]:
        for w_name, h_name in (
            ("pipeline_width", "pipeline_height"),
            ("source_width", "source_height"),
            ("frame_width", "frame_height"),
            ("width", "height"),
        ):
            width = getattr(frame_meta, w_name, None)
            height = getattr(frame_meta, h_name, None)
            if width and height:
                return int(width), int(height)
        return 1920, 1080

    def _source_id(self, frame_meta: Any) -> int:
        for attr in ("source_id", "pad_index", "batch_id"):
            value = getattr(frame_meta, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _sensor(self, source_id: int) -> Tuple[str, str]:
        sensor_id = self.sensor_ids[source_id] if source_id < len(self.sensor_ids) else str(source_id)
        sensor_name = self.sensor_names[source_id] if source_id < len(self.sensor_names) else f"source_{source_id}"
        return sensor_id, sensor_name

    def _select_batch(self, output: np.ndarray, frame_meta: Any) -> np.ndarray:
        if output.ndim != 3:
            return output
        source_id = self._source_id(frame_meta)
        idx = max(0, min(source_id, output.shape[0] - 1))
        return output[idx]

    def _extract_label_output(self, frame_meta: Any) -> np.ndarray:
        label_output, missing_reason = self._find_label_output(frame_meta)
        if label_output is None:
            raise RuntimeError(missing_reason or "Missing label_xyxy_score tensor output")
        return label_output

    def _find_label_output(self, frame_meta: Any) -> Tuple[Optional[np.ndarray], str]:
        tensor_items = getattr(frame_meta, "tensor_items", None)
        if tensor_items is None:
            return None, "Frame metadata missing tensor_items"
        available_ids: List[int] = []
        for item in tensor_items:
            tensor_meta = item.as_tensor_output() if hasattr(item, "as_tensor_output") else item
            unique_id = int(getattr(tensor_meta, "unique_id", -1))
            available_ids.append(unique_id)
            if unique_id != self.gie_id:
                continue
            layer_map = tensor_meta.get_layers()
            if not self._logged_layers:
                self._logged_layers = True
                LOGGER.info("DEIMv2 Wholebody49 tensor layers: %s", list(layer_map.keys()))
            if "label_xyxy_score" not in layer_map:
                return None, f"Missing label_xyxy_score in tensor layers: {list(layer_map.keys())}"
            return _to_numpy(layer_map["label_xyxy_score"]), ""
        return None, f"No tensor output found for gie_id={self.gie_id}; available_ids={available_ids}"

    def _add_text(self, display_meta: Any, text: str, x: int, y: int, *, size: int = 13) -> None:
        label = osd.Text()
        label.x_offset = int(x)
        label.y_offset = int(y)
        label.font.name = osd.FontFamily.Serif
        label.font.size = int(size)
        label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
        label.set_bg_color = 1
        label.bg_color = osd.Color(0.0, 0.0, 0.0, 0.65)
        label.display_text = text
        display_meta.add_text(label)

    def _label_text(self, det: Detection) -> str:
        label = f"{det.class_name} {det.score:.2f}"
        if det.attributes:
            attrs = " ".join(det.attributes.values())
            if attrs:
                label = f"{label} {attrs}"
        return label

    def _append_overlay(self, batch_meta: Any, frame_meta: Any, detections: Sequence[Detection]) -> None:
        display_meta = batch_meta.acquire_display_meta()
        if not display_meta:
            return

        source_id = self._source_id(frame_meta)
        _, sensor_name = self._sensor(source_id)
        summary = summarize_detections(detections)
        group_counts = summary["by_group"]
        lines = skeleton_lines(detections)
        self._add_text(
            display_meta,
            (
                f"{sensor_name} | body {group_counts.get('body', 0)} "
                f"obj {group_counts.get('object', 0)} attr {group_counts.get('attribute', 0)} "
                f"kpt {group_counts.get('keypoint', 0)} bone {group_counts.get('bone', 0)} "
                f"lines {len(lines)}"
            ),
            12,
            12,
            size=14,
        )

        for start, end in lines[:120]:
            line = osd.Line()
            line.x1 = int(start.center[0])
            line.y1 = int(start.center[1])
            line.x2 = int(end.center[0])
            line.y2 = int(end.center[1])
            line.width = 4
            line.color = osd.Color(0.0, 0.85, 1.0, 1.0)
            display_meta.add_line(line)

        drawn = 0
        for det in detections:
            if drawn >= self.max_draw:
                break
            x1, y1, x2, y2 = det.bbox
            color_tuple = CLASS_COLORS.get(det.group, CLASS_COLORS["object"])
            rect = osd.Rect()
            rect.left = int(x1)
            rect.top = int(y1)
            rect.width = int(max(1.0, x2 - x1))
            rect.height = int(max(1.0, y2 - y1))
            if det.class_id == BODY_CLASS_ID:
                rect.border_width = 4
            elif det.group in {"bone", "keypoint"}:
                rect.border_width = 3
            else:
                rect.border_width = 2
            rect.border_color = osd.Color(*color_tuple)
            display_meta.add_rect(rect)
            if det.group == "keypoint":
                circ = osd.Circle()
                circ.xc = int(det.center[0])
                circ.yc = int(det.center[1])
                circ.radius = 4
                circ.width = 2
                circ.color = osd.Color(*color_tuple)
                display_meta.add_circle(circ)
            else:
                if det.class_id in (BODY_CLASS_ID, 7, 16, 32, 33, 34, 45, 46, 47) or det.group in {
                    "attribute",
                    "bone",
                }:
                    self._add_text(
                        display_meta,
                        self._label_text(det),
                        int(x1),
                        int(max(0.0, y1 - 18.0)),
                        size=12 if det.group == "attribute" else 13,
                    )
            drawn += 1

        frame_meta.append(display_meta)

    @staticmethod
    def _is_missing_tensor_error(exc: RuntimeError) -> bool:
        message = str(exc)
        return (
            "Frame metadata missing tensor_items" in message
            or "No tensor output found" in message
            or "Missing label_xyxy_score" in message
        )

    def _cached_detections_for_source(self, source_id: int) -> List[Detection]:
        cached = self._last_detections_by_source.get(source_id, [])
        return list(cached)

    def _decode_frame_detections(
        self,
        frame_meta: Any,
        *,
        frame_w: int,
        frame_h: int,
        source_id: int,
    ) -> Tuple[List[Detection], bool]:
        label_output, missing_reason = self._find_label_output(frame_meta)
        if label_output is None:
            if not self.reuse_last_on_missing_tensor:
                raise RuntimeError(missing_reason)
            if not self._logged_tensor_cache_reuse:
                self._logged_tensor_cache_reuse = True
                LOGGER.info("Reusing cached Wholebody49 detections on skipped tensor frames")
            self._tensor_cache_reuse_counts[source_id] += 1
            return self._cached_detections_for_source(source_id), True
        label_output = self._select_batch(label_output, frame_meta)

        detections = decode_detections(
            label_output,
            frame_width=frame_w,
            frame_height=frame_h,
            class_names=self.class_names,
            object_score_threshold=self.object_score_threshold,
            attribute_score_threshold=self.attribute_score_threshold,
            keypoint_threshold=self.keypoint_threshold,
            keypoint_candidate_threshold=self.keypoint_candidate_threshold,
            class_aware_filtering=self.class_aware_filtering,
            max_detections=self.max_detections,
            has_instance_masks=self.has_instance_masks,
        )
        if self._smoother is not None:
            detections = self._smoother.update(source_id, detections)
        self._last_detections_by_source[source_id] = [_copy_detection(det) for det in detections]
        return detections, False

    def handle_metadata(self, batch_meta: Any) -> None:  # type: ignore[override]
        now_s = time.time()
        for frame_meta in batch_meta.frame_items:
            frame_w, frame_h = self._frame_size(frame_meta)
            source_id = self._source_id(frame_meta)
            sensor_id, sensor_name = self._sensor(source_id)
            detections, reused_tensor_cache = self._decode_frame_detections(
                frame_meta,
                frame_w=frame_w,
                frame_h=frame_h,
                source_id=source_id,
            )
            self._append_overlay(batch_meta, frame_meta, detections)
            self._frame_counts[source_id] += 1
            frame_number = int(getattr(frame_meta, "frame_number", self._frame_counts[source_id]))
            if self._frame_counts[source_id] % self.emit_every_frames == 0:
                path = write_summary_json(
                    output_root=self.output_root,
                    sensor_id=sensor_id,
                    sensor_name=sensor_name,
                    frame_number=frame_number,
                    detections=detections,
                    model_info=self.model_info,
                    reused_tensor_cache=reused_tensor_cache,
                )
                self._latest_paths[source_id] = path
                summary = summarize_detections(detections)
                print(
                    "DEIMV2_WHOLEBODY49_EMIT "
                    f"sensor={sensor_name} frame={frame_number} "
                    f"total={summary['total']} bodies={summary['bodies']} "
                    f"cached={int(reused_tensor_cache)} path={path}",
                    flush=True,
                )

        if now_s - self._last_report_s >= 1.0:
            self._last_report_s = now_s
            if self._frame_counts:
                parts = []
                for source_id, count in sorted(self._frame_counts.items()):
                    _, sensor_name = self._sensor(source_id)
                    cached_count = self._tensor_cache_reuse_counts.get(source_id, 0)
                    cached_suffix = f"/cached{cached_count}" if cached_count else ""
                    parts.append(f"{sensor_name}:{count}{cached_suffix}")
                print("DEIMV2_WHOLEBODY49_HEARTBEAT " + " ".join(parts), flush=True)

    def finalize(self) -> List[Path]:
        return [path for _, path in sorted(self._latest_paths.items())]
