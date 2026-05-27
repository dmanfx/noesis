from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .geometry import PlaneCandidate, decode_mask_rle


class ZeroPlaneAdapterError(RuntimeError):
    """Raised when ZeroPlane cannot produce the required planar inference."""


@dataclass(frozen=True)
class ZeroPlaneFrame:
    frame_id: str
    image_size: tuple[int, int]
    planes: list[PlaneCandidate]
    source_path: Path | None = None
    metadata: dict[str, Any] | None = None


def _plane_from_param(param: np.ndarray) -> tuple[np.ndarray, float]:
    coeff = np.asarray(param, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(coeff))
    if norm <= 1e-12:
        raise ValueError("ZeroPlane plane parameter is degenerate")
    normal = coeff / norm
    # ZeroPlane reconstructs depth as 1 / ((normal / offset) dot ray).
    positive_offset = 1.0 / norm
    return normal.astype(np.float32), -float(positive_offset)


def _candidate_from_mapping(frame_id: str, index: int, payload: Mapping[str, Any], base_dir: Path) -> PlaneCandidate:
    mask_raw = payload.get("mask")
    if mask_raw is None and "mask_rle" in payload:
        mask = decode_mask_rle(payload["mask_rle"])
    elif isinstance(mask_raw, str):
        mask_path = (base_dir / mask_raw).resolve()
        mask = np.load(mask_path).astype(bool)
    else:
        mask = np.asarray(mask_raw, dtype=bool)
    if mask.ndim != 2:
        raise ZeroPlaneAdapterError(f"ZeroPlane plane {index} mask must be 2D")

    planar_depth = None
    depth_raw = payload.get("planar_depth")
    if isinstance(depth_raw, str):
        planar_depth = np.load((base_dir / depth_raw).resolve()).astype(np.float32)
    elif depth_raw is not None:
        planar_depth = np.asarray(depth_raw, dtype=np.float32)

    normal = None
    offset = None
    if payload.get("normal") is not None and payload.get("offset") is not None:
        normal = np.asarray(payload["normal"], dtype=np.float32)
        offset = float(payload["offset"])
    elif payload.get("param") is not None:
        normal, offset = _plane_from_param(np.asarray(payload["param"], dtype=np.float32))

    return PlaneCandidate(
        frame_id=str(frame_id),
        plane_id=str(payload.get("plane_id") or f"plane_{index:02d}"),
        mask=mask,
        planar_depth=planar_depth,
        normal=normal,
        offset=offset,
        confidence=float(payload.get("confidence", 1.0) or 1.0),
        semantic_label=str(payload.get("semantic_label") or "").strip() or None,
    )


def load_zeroplane_frame(path: Path, *, frame_id: str | None = None) -> ZeroPlaneFrame:
    source = Path(path)
    if not source.exists():
        raise ZeroPlaneAdapterError(f"ZeroPlane output is missing: {source}")

    planes: list[PlaneCandidate] = []
    metadata: dict[str, Any] = {}
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ZeroPlaneAdapterError(f"ZeroPlane JSON output must be an object: {source}")
        fid = str(frame_id or payload.get("frame_id") or source.stem)
        for index, item in enumerate(payload.get("planes") or []):
            if not isinstance(item, Mapping):
                continue
            planes.append(_candidate_from_mapping(fid, index, item, source.parent))
        size = payload.get("image_size") or payload.get("size")
        if isinstance(size, Sequence) and len(size) >= 2:
            image_size = (int(size[0]), int(size[1]))
        elif planes:
            image_size = (int(planes[0].mask.shape[1]), int(planes[0].mask.shape[0]))
        else:
            image_size = (0, 0)
        metadata = dict(payload.get("metadata") or {})
        return ZeroPlaneFrame(frame_id=fid, image_size=image_size, planes=planes, source_path=source, metadata=metadata)

    if source.suffix.lower() != ".npz":
        raise ZeroPlaneAdapterError(f"unsupported ZeroPlane output format: {source}")

    data = np.load(source, allow_pickle=False)
    if frame_id is not None:
        fid = str(frame_id)
    elif "frame_id" in data:
        fid = str(np.asarray(data["frame_id"]).item())
    else:
        fid = source.stem
    if "masks" in data:
        masks = np.asarray(data["masks"], dtype=bool)
        if masks.ndim == 2:
            masks = masks[None, :, :]
    elif "segmentation" in data:
        segmentation = np.asarray(data["segmentation"], dtype=np.int32)
        labels = [int(v) for v in sorted(np.unique(segmentation)) if int(v) >= 0]
        if labels:
            # The highest ZeroPlane argmax class is typically the non-plane bucket.
            labels = labels[:-1] if len(labels) > 1 else labels
        masks = np.stack([segmentation == label for label in labels], axis=0) if labels else np.zeros((0, *segmentation.shape), dtype=bool)
    else:
        raise ZeroPlaneAdapterError(f"ZeroPlane output lacks masks or segmentation: {source}")

    planar_depth = np.asarray(data["planar_depth"], dtype=np.float32) if "planar_depth" in data else None
    if planar_depth is None and "planes_depth" in data:
        planar_depth = np.asarray(data["planes_depth"], dtype=np.float32)
    normals = np.asarray(data["normals"], dtype=np.float32) if "normals" in data else None
    offsets = np.asarray(data["offsets"], dtype=np.float32) if "offsets" in data else None
    params = np.asarray(data["params"], dtype=np.float32) if "params" in data else None
    confidence = np.asarray(data["confidence"], dtype=np.float32) if "confidence" in data else None
    labels = np.asarray(data["labels"], dtype=object) if "labels" in data else None

    for idx in range(int(masks.shape[0])):
        normal = None
        offset = None
        if normals is not None and offsets is not None and idx < normals.shape[0] and idx < offsets.shape[0]:
            normal = np.asarray(normals[idx], dtype=np.float32)
            offset = float(offsets[idx])
        elif params is not None and idx < params.shape[0]:
            try:
                normal, offset = _plane_from_param(params[idx])
            except ValueError:
                normal = None
                offset = None
        plane_depth = planar_depth if planar_depth is not None and planar_depth.ndim == 2 else None
        if planar_depth is not None and planar_depth.ndim == 3 and idx < planar_depth.shape[0]:
            plane_depth = planar_depth[idx]
        label = None
        if labels is not None and idx < labels.shape[0]:
            label = str(labels[idx])
        conf = float(confidence[idx]) if confidence is not None and idx < confidence.shape[0] else 1.0
        planes.append(
            PlaneCandidate(
                frame_id=fid,
                plane_id=f"plane_{idx:02d}",
                mask=np.asarray(masks[idx], dtype=bool),
                planar_depth=plane_depth,
                normal=normal,
                offset=offset,
                confidence=conf,
                semantic_label=label,
            )
        )
    image_size = (int(masks.shape[2]), int(masks.shape[1])) if masks.ndim == 3 else (0, 0)
    return ZeroPlaneFrame(frame_id=fid, image_size=image_size, planes=planes, source_path=source, metadata=metadata)


@dataclass(frozen=True)
class PrecomputedZeroPlaneAdapter:
    output_dir: Path

    def infer(self, *, frame_id: str, image_path: Path, intrinsics: np.ndarray, image_size: tuple[int, int]) -> ZeroPlaneFrame:
        del image_path, intrinsics, image_size
        base = Path(self.output_dir)
        for suffix in (".npz", ".json"):
            candidate = base / f"{frame_id}{suffix}"
            if candidate.exists():
                return load_zeroplane_frame(candidate, frame_id=frame_id)
        raise ZeroPlaneAdapterError(f"precomputed ZeroPlane output missing for frame {frame_id} in {base}")


@dataclass(frozen=True)
class ZeroPlaneCommandAdapter:
    repo_dir: Path
    checkpoint_path: Path
    config_path: Path
    output_dir: Path
    runner_script: Path
    python: str = sys.executable
    device: str = "cuda"

    def infer(self, *, frame_id: str, image_path: Path, intrinsics: np.ndarray, image_size: tuple[int, int]) -> ZeroPlaneFrame:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_npz = out_dir / f"{frame_id}.npz"
        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        cmd = [
            str(self.python),
            str(self.runner_script),
            "--zeroplane-repo",
            str(self.repo_dir),
            "--config-file",
            str(self.config_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--image",
            str(image_path),
            "--frame-id",
            str(frame_id),
            "--output",
            str(output_npz),
            "--fx",
            str(float(k[0, 0])),
            "--fy",
            str(float(k[1, 1])),
            "--cx",
            str(float(k[0, 2])),
            "--cy",
            str(float(k[1, 2])),
            "--original-w",
            str(int(image_size[0])),
            "--original-h",
            str(int(image_size[1])),
            "--device",
            str(self.device),
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise ZeroPlaneAdapterError(
                "ZeroPlane inference failed "
                f"(exit={proc.returncode})\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
            )
        return load_zeroplane_frame(output_npz, frame_id=frame_id)


__all__ = [
    "PrecomputedZeroPlaneAdapter",
    "ZeroPlaneAdapterError",
    "ZeroPlaneCommandAdapter",
    "ZeroPlaneFrame",
    "load_zeroplane_frame",
]
