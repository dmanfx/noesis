"""Metadata probes for segmentation heartbeat reporting."""

from __future__ import annotations

import json
import time
import threading
from typing import Dict, List, Optional

import numpy as np
from pyservicemaker import BatchMetadataOperator
from pyservicemaker import osd


class SegmentationHeartbeat(BatchMetadataOperator):
    def __init__(
        self,
        labels: Dict[str, object],
        sensor_ids: List[str],
        report_interval: float = 1.0,
        first_heartbeat_event: Optional[threading.Event] = None,
        report_top_k: int = 5,
        rewrite_for_floor_only_visualization: bool = True,
        visualization_background_class_id: int = 0,
    ) -> None:
        super().__init__()
        self._labels = labels
        self._sensor_ids = sensor_ids
        self._report_interval = float(report_interval)
        self._last_report = time.monotonic()
        self._frame_count = 0
        self._first_heartbeat_event = first_heartbeat_event
        self._heartbeat_started = False
        self._report_top_k = max(0, int(report_top_k))
        self._map_type_logged = False
        self._prob_map_type_logged = False
        self._id_to_name = labels.get("id_to_name", [])
        self._floor_class_id = int(labels["floor_class_id"])
        self._free_space_ids = set(int(x) for x in labels["free_space_ids"])
        self._rewrite_for_floor_only_visualization = bool(rewrite_for_floor_only_visualization)
        self._visualization_background_class_id = int(visualization_background_class_id)

    def _class_name(self, class_id: int) -> str:
        if self._id_to_name and 0 <= class_id < len(self._id_to_name):
            return str(self._id_to_name[class_id])
        return str(class_id)

    def _short_class_name(self, class_id: int) -> str:
        name = self._class_name(class_id)
        primary = name.split(";", 1)[0].strip()
        primary = primary.replace(" ", "_")
        return primary[:14] if primary else str(class_id)

    def _log_meta_types_once(self, seg) -> None:
        class_map = seg.class_map
        if not self._map_type_logged:
            self._map_type_logged = True
            length = None
            shape = None
            dtype = None
            data_ptr: Optional[int] = None
            flags = {}
            if isinstance(class_map, np.ndarray):
                length = class_map.shape[0] if class_map.ndim else 0
                shape = tuple(int(x) for x in class_map.shape)
                dtype = str(class_map.dtype)
                data_ptr = int(class_map.__array_interface__["data"][0])
                flags = {
                    "own_data": bool(class_map.flags["OWNDATA"]),
                    "writeable": bool(class_map.flags["WRITEABLE"]),
                    "c_contig": bool(class_map.flags["C_CONTIGUOUS"]),
                }
            else:
                try:
                    length = len(class_map)
                except Exception:
                    length = None
            print(
                f"BIDNET_SEG_META class_map_type={type(class_map)} len={length} shape={shape} dtype={dtype} data_ptr={hex(data_ptr) if data_ptr is not None else None} flags={flags}",
                flush=True,
            )

        if not self._prob_map_type_logged:
            self._prob_map_type_logged = True
            prob_map = getattr(seg, "class_probabilities_map", None)
            prob_len = None
            prob_shape = None
            prob_dtype = None
            prob_ptr: Optional[int] = None
            prob_flags = {}
            if isinstance(prob_map, np.ndarray):
                prob_len = prob_map.shape[0] if prob_map.ndim else 0
                prob_shape = tuple(int(x) for x in prob_map.shape)
                prob_dtype = str(prob_map.dtype)
                prob_ptr = int(prob_map.__array_interface__["data"][0])
                prob_flags = {
                    "own_data": bool(prob_map.flags["OWNDATA"]),
                    "writeable": bool(prob_map.flags["WRITEABLE"]),
                    "c_contig": bool(prob_map.flags["C_CONTIGUOUS"]),
                }
            else:
                try:
                    prob_len = len(prob_map) if prob_map is not None else None
                except Exception:
                    prob_len = None
            print(
                f"BIDNET_SEG_META prob_map_type={type(prob_map)} len={prob_len} shape={prob_shape} dtype={prob_dtype} data_ptr={hex(prob_ptr) if prob_ptr is not None else None} flags={prob_flags}",
                flush=True,
            )

    def _get_class_map_flat(self, seg) -> np.ndarray:
        self._log_meta_types_once(seg)
        return np.asarray(seg.class_map, dtype=np.int32).reshape(-1)

    def _rewrite_class_map_for_floor_visualization(self, seg, flat_map: np.ndarray) -> bool:
        if not self._rewrite_for_floor_only_visualization:
            return False

        bg_id = self._visualization_background_class_id
        floor_id = self._floor_class_id
        if bg_id == floor_id:
            return False

        class_map_obj = seg.class_map
        if isinstance(class_map_obj, np.ndarray):
            try:
                class_map_obj[...] = np.where(class_map_obj == floor_id, floor_id, bg_id)
            except Exception:
                return False
            # Verify the mutation stuck on the same view.
            try:
                unique_ids = set(int(x) for x in np.unique(class_map_obj))
            except Exception:
                return True
            return unique_ids.issubset({bg_id, floor_id})

        viz = np.where(flat_map == floor_id, floor_id, bg_id).astype(np.int32, copy=False)
        viz_list = viz.tolist()
        try:
            seg.class_map = viz_list
            return True
        except Exception:
            return False

    def handle_metadata(self, batch_meta) -> None:  # type: ignore[override]
        try:
            now = time.monotonic()
            self._frame_count += 1

            stream_stats_map = {}
            mask_drawn_any = False

            for frame_meta in batch_meta.frame_items:
                source_id = int(frame_meta.source_id)
                seg_items = list(frame_meta.segmentation_items)
                seg = None
                for item in seg_items:
                    candidate = item.as_segmentation() if hasattr(item, "as_segmentation") else item
                    if hasattr(candidate, "width"):
                        seg = candidate
                        break
                mask_drawn = seg is not None
                mask_drawn_any = mask_drawn_any or mask_drawn
                floor_count = 0
                free_space_count = 0
                seg_width = 0
                seg_height = 0
                top_classes: List[Dict[str, object]] = []
                viz_top_classes: List[Dict[str, object]] = []
                mask_filtered = False

                if seg is not None:
                    seg_width = int(seg.width)
                    seg_height = int(seg.height)
                    flat_map = self._get_class_map_flat(seg)
                    class_map = flat_map.reshape(seg_height, seg_width)
                    floor_count = int(np.count_nonzero(class_map == self._floor_class_id))
                    if self._free_space_ids == {self._floor_class_id}:
                        free_space_count = floor_count
                    else:
                        free_space_count = int(
                            np.count_nonzero(np.isin(class_map, list(self._free_space_ids)))
                        )

                    if self._report_top_k > 0:
                        try:
                            ids, counts = np.unique(flat_map, return_counts=True)
                            order = np.argsort(-counts)
                            for idx in order[: self._report_top_k]:
                                class_id = int(ids[idx])
                                top_classes.append(
                                    {
                                        "class_id": class_id,
                                        "class_name": self._class_name(class_id),
                                        "count": int(counts[idx]),
                                    }
                                )
                        except Exception:
                            top_classes = []

                    mask_filtered = self._rewrite_class_map_for_floor_visualization(seg, flat_map)
                    if mask_filtered and self._report_top_k > 0:
                        try:
                            viz_flat = np.asarray(seg.class_map, dtype=np.int32).reshape(-1)
                            ids, counts = np.unique(viz_flat, return_counts=True)
                            order = np.argsort(-counts)
                            for idx in order[: self._report_top_k]:
                                class_id = int(ids[idx])
                                viz_top_classes.append(
                                    {
                                        "class_id": class_id,
                                        "class_name": self._class_name(class_id),
                                        "count": int(counts[idx]),
                                    }
                                )
                        except Exception:
                            viz_top_classes = []

                    # Add per-stream OSD label (picked up by nvdsosd downstream).
                    try:
                        display_meta = batch_meta.acquire_display_meta()
                        label = osd.Text()
                        label.x_offset = 10
                        label.y_offset = 10
                        label.font.name = osd.FontFamily.Serif
                        label.font.size = 14
                        label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                        label.set_bg_color = 1
                        label.bg_color = osd.Color(0.0, 0.0, 0.0, 0.6)

                        total_px = max(seg_width * seg_height, 1)
                        floor_pct = 100.0 * float(floor_count) / float(total_px)

                        top_parts = []
                        for entry in (viz_top_classes or top_classes)[:3]:
                            class_id = int(entry["class_id"])
                            pct = 100.0 * float(entry["count"]) / float(total_px)
                            top_parts.append(f"{class_id}:{self._short_class_name(class_id)} {pct:.0f}%")
                        top_str = " | ".join(top_parts) if top_parts else "n/a"
                        label.display_text = f"S{source_id} floor={floor_pct:.0f}%\n{top_str}"
                        display_meta.add_text(label)
                        frame_meta.append(display_meta)
                    except Exception:
                        pass

                sensor_id = (
                    self._sensor_ids[source_id]
                    if source_id < len(self._sensor_ids)
                    else str(source_id)
                )
                floor_name = self._class_name(self._floor_class_id) if self._id_to_name else "floor"
                stream_stats_map[source_id] = (
                    {
                        "source_id": source_id,
                        "sensor_id": sensor_id,
                        "mask_drawn": mask_drawn,
                        "mask_filtered": mask_filtered,
                        "seg_width": seg_width,
                        "seg_height": seg_height,
                        "top_classes": top_classes,
                        "viz_top_classes": viz_top_classes,
                        "floor": {
                            "class_id": self._floor_class_id,
                            "class_name": floor_name,
                            "count": floor_count,
                        },
                        "free_space": {
                            "class_name": "free_space",
                            "count": free_space_count,
                        },
                    }
                )

            for idx, sensor_id in enumerate(self._sensor_ids):
                source_id = idx
                if source_id in stream_stats_map:
                    continue
                stream_stats_map[source_id] = {
                    "source_id": source_id,
                    "sensor_id": sensor_id,
                    "mask_drawn": False,
                    "mask_filtered": False,
                    "seg_width": 0,
                    "seg_height": 0,
                    "top_classes": [],
                    "viz_top_classes": [],
                    "floor": {
                        "class_id": self._floor_class_id,
                        "class_name": self._id_to_name[self._floor_class_id]
                        if self._id_to_name and self._floor_class_id < len(self._id_to_name)
                        else "floor",
                        "count": 0,
                    },
                    "free_space": {
                        "class_name": "free_space",
                        "count": 0,
                    },
                }

            elapsed = now - self._last_report
            if elapsed >= self._report_interval:
                fps = self._frame_count / max(elapsed, 1e-6)
                stream_stats = [stream_stats_map[idx] for idx in sorted(stream_stats_map.keys())]
                payload = {
                    "fps": round(fps, 2),
                    "mask_drawn": mask_drawn_any,
                    "streams": stream_stats,
                }
                print("BIDNET_HEARTBEAT " + json.dumps(payload, sort_keys=True), flush=True)
                if self._first_heartbeat_event and not self._heartbeat_started:
                    self._heartbeat_started = True
                    self._first_heartbeat_event.set()
                self._frame_count = 0
                self._last_report = now
        except Exception as exc:
            # Never let probe exceptions crash the pipeline.
            print(f"BIDNET_PROBE_ERROR {type(exc).__name__}: {exc}", flush=True)
