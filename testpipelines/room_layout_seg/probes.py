"""Metadata probe that emits compact room-layout mask bundles."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from pyservicemaker import BatchMetadataOperator
from pyservicemaker import osd


def _sanitize_component(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return collapsed.strip("._-") or "source"


def _labels_palette(output_labels: List[Dict[str, object]]) -> np.ndarray:
    palette = np.zeros((len(output_labels), 3), dtype=np.uint8)
    for idx, entry in enumerate(output_labels):
        palette[idx] = np.asarray(entry.get("palette", [0, 0, 0]), dtype=np.uint8)
    return palette


def _coverage_percentages(class_map: np.ndarray, output_labels: List[Dict[str, object]]) -> Dict[str, float]:
    total = max(1, int(class_map.size))
    coverage: Dict[str, float] = {}
    for entry in output_labels:
        class_id = int(entry["output_id"])
        coverage[str(entry["name"])] = 100.0 * float(np.count_nonzero(class_map == class_id)) / float(total)
    return coverage


def write_layout_bundle(
    *,
    output_root: Path,
    sensor_id: str,
    sensor_name: str,
    avg_probabilities: np.ndarray,
    output_labels: List[Dict[str, object]],
    model_info: Dict[str, object],
    frames_accumulated: int,
    first_seen_s: float,
    last_seen_s: float,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    sensor_dir = output_root / f"{_sanitize_component(sensor_id)}_{_sanitize_component(sensor_name)}"
    sensor_dir.mkdir(parents=True, exist_ok=True)

    label_map = np.argmax(avg_probabilities, axis=0).astype(np.uint8)
    palette = _labels_palette(output_labels)
    preview_rgb = palette[label_map]

    class_map_path = sensor_dir / "layout_class_map.png"
    preview_path = sensor_dir / "layout_preview.png"
    probabilities_path = sensor_dir / "layout_probabilities_fp16.npz"
    manifest_path = sensor_dir / "layout_manifest.json"

    Image.fromarray(label_map, mode="L").save(class_map_path)
    Image.fromarray(preview_rgb, mode="RGB").save(preview_path)
    np.savez_compressed(probabilities_path, probabilities=avg_probabilities.astype(np.float16))

    mask_files: Dict[str, str] = {}
    for entry in output_labels:
        class_id = int(entry["output_id"])
        class_name = str(entry["name"])
        if class_name == "other":
            continue
        mask_path = sensor_dir / f"mask_{class_name}.png"
        mask = np.where(label_map == class_id, 255, 0).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(mask_path)
        mask_files[class_name] = mask_path.name

    coverage = _coverage_percentages(label_map, output_labels)
    manifest = {
        "schema_version": 1,
        "sensor_id": sensor_id,
        "sensor_name": sensor_name,
        "sensor_dir": str(sensor_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "frames_accumulated": int(frames_accumulated),
        "first_seen_s": float(first_seen_s),
        "last_seen_s": float(last_seen_s),
        "frame_shape": {
            "height": int(avg_probabilities.shape[1]),
            "width": int(avg_probabilities.shape[2]),
        },
        "labels": output_labels,
        "coverage_percent": coverage,
        "files": {
            "class_map_png": class_map_path.name,
            "preview_png": preview_path.name,
            "probabilities_npz": probabilities_path.name,
            "binary_masks": mask_files,
        },
        "model": {
            "name": model_info["model"]["name"],
            "base_model_id": model_info["model"]["base_model_id"],
            "input": model_info["input"],
            "output": model_info["output"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class SourceState:
    sensor_id: str
    sensor_name: str
    prob_sum: Optional[np.ndarray] = None
    frame_count: int = 0
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    last_emitted_frame_count: int = 0


class RoomLayoutEmitter(BatchMetadataOperator):
    def __init__(
        self,
        *,
        output_root: Path,
        output_labels: List[Dict[str, object]],
        model_info: Dict[str, object],
        sensor_ids: List[str],
        sensor_names: List[str],
        frames_per_package: int = 24,
        emit_every_frames: int = 24,
        report_interval_s: float = 1.0,
    ) -> None:
        super().__init__()
        self._output_root = Path(output_root)
        self._output_labels = list(output_labels)
        self._model_info = model_info
        self._sensor_ids = [str(x) for x in sensor_ids]
        self._sensor_names = [str(x) for x in sensor_names]
        self._num_classes = len(self._output_labels)
        self._frames_per_package = max(1, int(frames_per_package))
        self._emit_every_frames = max(1, int(emit_every_frames))
        self._report_interval_s = float(report_interval_s)
        self._last_report_s = 0.0
        self._states: Dict[int, SourceState] = {}
        self._map_type_logged = False
        self._prob_type_logged = False

    def _state_for(self, source_id: int) -> SourceState:
        if source_id not in self._states:
            sensor_id = self._sensor_ids[source_id] if source_id < len(self._sensor_ids) else str(source_id)
            sensor_name = (
                self._sensor_names[source_id]
                if source_id < len(self._sensor_names)
                else f"source_{source_id}"
            )
            self._states[source_id] = SourceState(sensor_id=sensor_id, sensor_name=sensor_name)
        return self._states[source_id]

    def _log_types_once(self, seg) -> None:
        if not self._map_type_logged:
            self._map_type_logged = True
            print(
                "ROOM_LAYOUT_SEG_META class_map_type="
                f"{type(seg.class_map)} prob_type={type(getattr(seg, 'class_probabilities_map', None))}",
                flush=True,
            )

    def _append_osd_label(self, batch_meta, frame_meta, coverage: Dict[str, float]) -> None:
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
            label.display_text = (
                f"floor={coverage.get('floor', 0.0):.0f}% "
                f"wall={coverage.get('wall', 0.0):.0f}% "
                f"ceiling={coverage.get('ceiling', 0.0):.0f}% "
                f"door={coverage.get('door', 0.0):.0f}% "
                f"window={coverage.get('window', 0.0):.0f}%"
            )
            display_meta.add_text(label)
            frame_meta.append(display_meta)
        except Exception:
            pass

    def _emit_state(self, state: SourceState) -> Dict[str, object]:
        if state.prob_sum is None or state.frame_count <= 0:
            raise RuntimeError(f"No probabilities accumulated for {state.sensor_name}")
        avg_probabilities = state.prob_sum / float(state.frame_count)
        manifest = write_layout_bundle(
            output_root=self._output_root,
            sensor_id=state.sensor_id,
            sensor_name=state.sensor_name,
            avg_probabilities=avg_probabilities,
            output_labels=self._output_labels,
            model_info=self._model_info,
            frames_accumulated=state.frame_count,
            first_seen_s=state.first_seen_s,
            last_seen_s=state.last_seen_s,
        )
        state.last_emitted_frame_count = state.frame_count
        return manifest

    def handle_metadata(self, batch_meta) -> None:  # type: ignore[override]
        now_s = time.time()
        for frame_meta in batch_meta.frame_items:
            source_id = int(frame_meta.source_id)
            seg = None
            for item in frame_meta.segmentation_items:
                candidate = item.as_segmentation() if hasattr(item, "as_segmentation") else item
                if hasattr(candidate, "width") and hasattr(candidate, "class_map"):
                    seg = candidate
                    break
            if seg is None:
                continue

            self._log_types_once(seg)
            class_map = np.asarray(seg.class_map, dtype=np.int32).reshape(int(seg.height), int(seg.width))
            prob_map_raw = getattr(seg, "class_probabilities_map", None)
            if prob_map_raw is None:
                raise RuntimeError(
                    "Segmentation probability map missing from nvinfer output; "
                    "room-layout utility requires probability tensors for bundle emission."
                )
            prob_map = np.asarray(prob_map_raw, dtype=np.float32).reshape(
                self._num_classes,
                int(seg.height),
                int(seg.width),
            )

            state = self._state_for(source_id)
            if state.prob_sum is None:
                state.prob_sum = np.zeros_like(prob_map, dtype=np.float32)
                state.first_seen_s = now_s
            state.prob_sum += prob_map
            state.frame_count += 1
            state.last_seen_s = now_s

            coverage = _coverage_percentages(class_map, self._output_labels)
            self._append_osd_label(batch_meta, frame_meta, coverage)

            should_emit = (
                state.frame_count >= self._frames_per_package
                and state.frame_count - state.last_emitted_frame_count >= self._emit_every_frames
            )
            if should_emit:
                manifest = self._emit_state(state)
                print(
                    "ROOM_LAYOUT_EMIT "
                    f"sensor={state.sensor_name} frames={state.frame_count} "
                    f"manifest={manifest['manifest_path']} "
                    f"floor={manifest['coverage_percent'].get('floor', 0.0):.1f} "
                    f"wall={manifest['coverage_percent'].get('wall', 0.0):.1f}",
                    flush=True,
                )

        if now_s - self._last_report_s >= self._report_interval_s:
            self._last_report_s = now_s
            summaries = []
            for state in self._states.values():
                summaries.append(f"{state.sensor_name}:{state.frame_count}")
            if summaries:
                print("ROOM_LAYOUT_HEARTBEAT " + " ".join(summaries), flush=True)

    def finalize(self) -> List[Dict[str, object]]:
        manifests: List[Dict[str, object]] = []
        for state in self._states.values():
            if state.frame_count <= 0:
                continue
            manifests.append(self._emit_state(state))
        return manifests
