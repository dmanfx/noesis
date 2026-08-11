#!/usr/bin/env python3
"""Evaluate MapAnything outputs against independent fixed-corpus annotations.

The evaluator intentionally distinguishes three evidence classes:

* human-authored RGB semantics (floor/obstacle regions and structural edges);
* authored camera calibration and the authored world floor plane;
* model outputs under evaluation.

It does not infer labels from a depth map or floorplan product, and it does not
turn unmeasured object distances into pseudo-ground-truth.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DS9_ROOT = SCRIPT_DIR.parent
REPO_ROOT = DS9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.strict_json import strict_json_loads  # noqa: E402


ANNOTATION_CONTRACT = "noesis.ds9.mapanything_fixed_annotations.v1"
REPORT_CONTRACT = "noesis.ds9.mapanything_fixed_corpus_evaluation.v1"
FIXTURE_CONTRACT = "noesis.ds9.mapanything_raw_fixture.v1"
EXPECTED_OUTPUTS = ("depth", "conf", "mask")


class EvaluationError(RuntimeError):
    """Raised when corpus identity or evaluation input is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return strict_json_loads(
            path.read_text(encoding="utf-8"),
            label=str(path),
        )
    except Exception as exc:
        raise EvaluationError(f"invalid JSON {path}: {exc}") from exc


def _require_sha(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise EvaluationError(f"{label} does not exist: {path}")
    observed = _sha256_file(path)
    if observed != str(expected):
        raise EvaluationError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _parse_dimensions(raw: Any) -> tuple[int, ...]:
    try:
        dims = tuple(int(part) for part in str(raw).lower().split("x"))
    except ValueError as exc:
        raise EvaluationError(f"invalid TensorRT dimensions: {raw!r}") from exc
    if not dims or any(value <= 0 for value in dims):
        raise EvaluationError(f"invalid TensorRT dimensions: {raw!r}")
    return dims


def _parse_output_tensors(payload: Any) -> dict[str, np.ndarray]:
    if not isinstance(payload, list):
        raise EvaluationError("TensorRT output JSON must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            raise EvaluationError("TensorRT output entry must be an object")
        name = str(item.get("name") or "")
        if name in entries:
            raise EvaluationError(f"duplicate TensorRT output: {name}")
        entries[name] = item
    if set(entries) != set(EXPECTED_OUTPUTS):
        raise EvaluationError(
            f"expected outputs {EXPECTED_OUTPUTS}; observed {tuple(sorted(entries))}"
        )

    tensors: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for name in EXPECTED_OUTPUTS:
        entry = entries[name]
        shape = _parse_dimensions(entry.get("dimensions"))
        if len(shape) != 4 or shape[1] != 1:
            raise EvaluationError(f"{name} must have NCHW shape Nx1xHxW; got {shape}")
        values = entry.get("values")
        if not isinstance(values, list):
            raise EvaluationError(f"{name}.values must be a list")
        expected_count = math.prod(shape)
        if len(values) != expected_count:
            raise EvaluationError(
                f"{name} value count {len(values)} does not match {shape}"
            )
        tensor = np.asarray(values, dtype=np.float32).reshape(shape)
        if not np.all(np.isfinite(tensor)):
            raise EvaluationError(f"{name} contains non-finite values")
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise EvaluationError(
                f"output shape mismatch: expected {expected_shape}, got {shape}"
            )
        tensors[name] = tensor[:, 0]
    return tensors


def _resolve_corpus_path(annotation_path: Path, raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    if not path:
        raise EvaluationError("empty corpus path")
    if path.is_absolute():
        return path
    return annotation_path.parent / path


def _dewarper_intrinsics(path: Path) -> tuple[tuple[int, int], np.ndarray]:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    if "property" not in parser or "surface0" not in parser:
        raise EvaluationError(f"invalid dewarper config: {path}")
    props = parser["property"]
    surface = parser["surface0"]
    try:
        width = int(float(props["output-width"]))
        height = int(float(props["output-height"]))
        fx, fy = (float(value) for value in surface["dst-focal-length"].split(";"))
        cx, cy = (
            float(value) for value in surface["dst-principal-point"].split(";")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"invalid dewarper intrinsics: {path}") from exc
    intrinsics = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (width, height), intrinsics


def _validate_calibration(
    annotations: Mapping[str, Any],
    *,
    annotation_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], float, dict[str, np.ndarray]]:
    calibration = annotations.get("calibration")
    if not isinstance(calibration, Mapping):
        raise EvaluationError("annotations.calibration must be an object")

    camera_ref = calibration.get("camera_calibration")
    floor_ref = calibration.get("floor_alignment")
    if not isinstance(camera_ref, Mapping) or not isinstance(floor_ref, Mapping):
        raise EvaluationError("calibration file identities are required")

    def resolve_repo_file(ref: Mapping[str, Any], label: str) -> Path:
        raw = Path(str(ref.get("path") or ""))
        path = raw if raw.is_absolute() else repo_root / raw
        _require_sha(path, str(ref.get("sha256") or ""), label=label)
        return path

    camera_path = resolve_repo_file(camera_ref, "camera calibration")
    floor_path = resolve_repo_file(floor_ref, "floor alignment")
    camera_payload = _load_json(camera_path)
    floor_payload = _load_json(floor_path)
    if not isinstance(camera_payload, Mapping) or not isinstance(
        floor_payload, Mapping
    ):
        raise EvaluationError("calibration files must contain JSON objects")
    cameras = camera_payload.get("cameras")
    if not isinstance(cameras, Mapping):
        raise EvaluationError("camera calibration has no cameras table")
    try:
        floor_y = float(floor_payload["floor_y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("floor alignment has no finite floor_y") from exc
    if not np.isfinite(floor_y):
        raise EvaluationError("floor alignment floor_y is non-finite")

    transforms: dict[str, np.ndarray] = {}
    for camera_id, entry in cameras.items():
        if not isinstance(entry, Mapping):
            continue
        extrinsics = np.asarray(entry.get("E"), dtype=np.float64)
        if extrinsics.size != 16 or not np.all(np.isfinite(extrinsics)):
            continue
        world_to_camera = extrinsics.reshape((4, 4), order="F")
        try:
            transforms[str(camera_id)] = np.linalg.inv(world_to_camera)
        except np.linalg.LinAlgError as exc:
            raise EvaluationError(
                f"singular extrinsics for {camera_id}"
            ) from exc

    dewarper_refs = calibration.get("dewarper_configs")
    if not isinstance(dewarper_refs, Mapping):
        raise EvaluationError("calibration.dewarper_configs must be an object")
    intrinsics_by_camera: dict[str, np.ndarray] = {}
    for camera_id, ref in dewarper_refs.items():
        if not isinstance(ref, Mapping):
            raise EvaluationError(f"invalid dewarper identity for {camera_id}")
        config_path = resolve_repo_file(ref, f"{camera_id} dewarper config")
        output_size, intrinsics = _dewarper_intrinsics(config_path)
        expected_size = tuple(int(value) for value in ref.get("output_size", ()))
        if expected_size != output_size:
            raise EvaluationError(
                f"{camera_id} dewarper size mismatch: "
                f"expected {expected_size}, observed {output_size}"
            )
        intrinsics_by_camera[str(camera_id)] = intrinsics

    return dict(calibration), floor_y, {
        camera_id: np.asarray(transform, dtype=np.float64)
        for camera_id, transform in transforms.items()
    } | {
        f"{camera_id}:K": intrinsics
        for camera_id, intrinsics in intrinsics_by_camera.items()
    }


def _mapped_polygon_mask(
    polygon_px: Sequence[Sequence[float]],
    *,
    source_size: tuple[int, int],
    output_shape: tuple[int, int],
    resized_size: tuple[int, int],
    pad_left: int,
    pad_top: int,
    erode_px_source: float = 0.0,
) -> np.ndarray:
    source_width, source_height = source_size
    output_height, output_width = output_shape
    resized_width, resized_height = resized_size
    if (
        source_width <= 0
        or source_height <= 0
        or resized_width <= 0
        or resized_height <= 0
        or output_width <= 0
        or output_height <= 0
    ):
        raise EvaluationError("invalid source, resized, or output dimensions")
    points = np.asarray(polygon_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise EvaluationError("polygon_px must contain at least three [x,y] points")
    if not np.all(np.isfinite(points)):
        raise EvaluationError("polygon_px contains non-finite coordinates")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= source_width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= source_height)
    ):
        raise EvaluationError("polygon_px lies outside the annotated RGB frame")

    mapped = np.empty_like(points)
    mapped[:, 0] = (
        points[:, 0] * (float(resized_width) / float(source_width))
    ) + float(pad_left)
    mapped[:, 1] = (
        points[:, 1] * (float(resized_height) / float(source_height))
    ) + float(pad_top)
    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(mapped).astype(np.int32)], 1)
    if erode_px_source > 0.0:
        scale = min(
            float(resized_width) / float(source_width),
            float(resized_height) / float(source_height),
        )
        radius = max(1, int(math.ceil(float(erode_px_source) * scale)))
        kernel = np.ones(((radius * 2) + 1, (radius * 2) + 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool)


def _source_pixel_grids(
    *,
    source_size: tuple[int, int],
    output_shape: tuple[int, int],
    resized_size: tuple[int, int],
    pad_left: int,
    pad_top: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_width, source_height = source_size
    output_height, output_width = output_shape
    resized_width, resized_height = resized_size
    u_out, v_out = np.meshgrid(
        np.arange(output_width, dtype=np.float64),
        np.arange(output_height, dtype=np.float64),
        indexing="xy",
    )
    u_source = (
        (u_out - float(pad_left) + 0.5)
        * (float(source_width) / float(resized_width))
    ) - 0.5
    v_source = (
        (v_out - float(pad_top) + 0.5)
        * (float(source_height) / float(resized_height))
    ) - 0.5
    return u_source, v_source


def _world_y_grid(
    depth: np.ndarray,
    *,
    u_source: np.ndarray,
    v_source: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    depth_map = np.asarray(depth, dtype=np.float64)
    if depth_map.shape != u_source.shape or depth_map.shape != v_source.shape:
        raise EvaluationError("depth and source pixel grids must have equal shapes")
    k = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if k.shape != (3, 3) or transform.shape != (4, 4):
        raise EvaluationError("intrinsics and camera_to_world shapes are invalid")
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise EvaluationError("intrinsics focal lengths must be non-zero")
    x_camera = (u_source - cx) * depth_map / fx
    y_camera = (v_source - cy) * depth_map / fy
    row_y = transform[1]
    return (
        (x_camera * row_y[0])
        + (y_camera * row_y[1])
        + (depth_map * row_y[2])
        + row_y[3]
    )


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _region_metrics(
    *,
    region_mask: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    world_y: np.ndarray,
    floor_y: float,
    semantic: str,
) -> dict[str, Any]:
    selected = region_mask & valid
    count = int(np.count_nonzero(selected))
    total = int(np.count_nonzero(region_mask))
    result: dict[str, Any] = {
        "annotated_output_pixels": total,
        "valid_output_pixels": count,
        "valid_fraction": float(count / total) if total else 0.0,
    }
    if count == 0:
        result["status"] = "no_valid_model_output"
        return result
    heights = np.asarray(world_y[selected] - float(floor_y), dtype=np.float64)
    depths = np.asarray(depth[selected], dtype=np.float64)
    result.update(
        {
            "status": "measured",
            "depth_m": {
                "p50": _percentile(depths, 50),
                "p95": _percentile(depths, 95),
            },
            "height_above_authored_floor_m": {
                "signed_p05": _percentile(heights, 5),
                "signed_p50": _percentile(heights, 50),
                "signed_p95": _percentile(heights, 95),
            },
        }
    )
    if semantic == "floor":
        absolute = np.abs(heights)
        result["floor_plane_residual_m"] = {
            "abs_p50": _percentile(absolute, 50),
            "abs_p95": _percentile(absolute, 95),
            "signed_bias": float(np.median(heights)),
            "within_0_08m_fraction": float(np.mean(absolute <= 0.08)),
        }
    elif semantic == "obstacle":
        result["obstacle_clearance"] = {
            "above_0_15m_fraction": float(np.mean(heights > 0.15)),
            "above_0_30m_fraction": float(np.mean(heights > 0.30)),
        }
    return result


def _sample_polyline(points: np.ndarray, *, spacing: float) -> np.ndarray:
    samples: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(distance / max(spacing, 1e-6))) + 1)
        segment = np.linspace(start, end, num=count, endpoint=True)
        if samples:
            segment = segment[1:]
        samples.extend(segment)
    return np.asarray(samples, dtype=np.float64)


def _edge_metrics(
    *,
    edge: Mapping[str, Any],
    depth: np.ndarray,
    valid: np.ndarray,
    source_size: tuple[int, int],
    resized_size: tuple[int, int],
    pad_left: int,
    pad_top: int,
) -> dict[str, Any]:
    points = np.asarray(edge.get("polyline_px"), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise EvaluationError("structural edge polyline_px must have at least 2 points")
    source_width, source_height = source_size
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= source_width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= source_height)
    ):
        raise EvaluationError("structural edge lies outside the annotated RGB frame")

    resized_width, resized_height = resized_size
    scale_x = float(resized_width) / float(source_width)
    scale_y = float(resized_height) / float(source_height)
    mapped = np.empty_like(points)
    mapped[:, 0] = (points[:, 0] * scale_x) + float(pad_left)
    mapped[:, 1] = (points[:, 1] * scale_y) + float(pad_top)
    samples = _sample_polyline(mapped, spacing=1.0)

    safe_depth = np.where(valid & (depth > 0.0), depth, np.nan)
    log_depth = np.log(safe_depth)
    grad_y, grad_x = np.gradient(log_depth)
    gradient = np.hypot(grad_x, grad_y)
    gradient[~np.isfinite(gradient)] = -np.inf

    radius_source = float(edge.get("search_radius_px", 24.0))
    radius = max(1, int(math.ceil(radius_source * max(scale_x, scale_y))))
    output_height, output_width = depth.shape
    distances_source: list[float] = []
    strengths: list[float] = []
    for sample_x, sample_y in samples:
        x0 = max(0, int(math.floor(sample_x)) - radius)
        x1 = min(output_width, int(math.floor(sample_x)) + radius + 1)
        y0 = max(0, int(math.floor(sample_y)) - radius)
        y1 = min(output_height, int(math.floor(sample_y)) + radius + 1)
        patch = gradient[y0:y1, x0:x1]
        if patch.size == 0 or not np.any(np.isfinite(patch)):
            continue
        flat_index = int(np.nanargmax(patch))
        local_y, local_x = np.unravel_index(flat_index, patch.shape)
        best_x = float(x0 + local_x)
        best_y = float(y0 + local_y)
        dx_source = (best_x - sample_x) / scale_x
        dy_source = (best_y - sample_y) / scale_y
        distances_source.append(float(math.hypot(dx_source, dy_source)))
        strengths.append(float(patch[local_y, local_x]))

    distances = np.asarray(distances_source, dtype=np.float64)
    strengths_arr = np.asarray(strengths, dtype=np.float64)
    return {
        "id": str(edge.get("id") or ""),
        "kind": str(edge.get("kind") or "structural_depth_discontinuity"),
        "sample_count": int(distances.size),
        "search_radius_px_source": radius_source,
        "nearest_log_depth_gradient_distance_px_source": {
            "p50": _percentile(distances, 50),
            "p95": _percentile(distances, 95),
        },
        "log_depth_gradient_strength": {
            "p50": _percentile(strengths_arr, 50),
            "p95": _percentile(strengths_arr, 95),
        },
        "status": "measured" if distances.size else "no_valid_model_output",
    }


def evaluate(
    *,
    annotations_path: Path,
    fixture_receipt_path: Path,
    model_output_path: Path,
    model_output_identity_path: Path | None = None,
    candidate_id: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    annotations = _load_json(annotations_path)
    receipt = _load_json(fixture_receipt_path)
    output_payload = _load_json(model_output_path)
    if not isinstance(annotations, Mapping):
        raise EvaluationError("annotation JSON must contain an object")
    if annotations.get("contract") != ANNOTATION_CONTRACT:
        raise EvaluationError(
            f"annotation contract must be {ANNOTATION_CONTRACT}"
        )
    if not isinstance(receipt, Mapping) or receipt.get("contract") != FIXTURE_CONTRACT:
        raise EvaluationError(f"fixture receipt contract must be {FIXTURE_CONTRACT}")
    if bool(receipt.get("identical_batch_members")):
        raise EvaluationError("fixed-corpus evaluation requires distinct scene inputs")

    calibration, floor_y, calibration_arrays = _validate_calibration(
        annotations,
        annotation_path=annotations_path,
        repo_root=repo_root,
    )
    tensors = _parse_output_tensors(output_payload)
    depth_tensor = tensors["depth"]
    conf_tensor = tensors["conf"]
    mask_tensor = tensors["mask"]

    sources = receipt.get("sources")
    frames = annotations.get("frames")
    if not isinstance(sources, list) or not isinstance(frames, list):
        raise EvaluationError("fixture sources and annotation frames must be lists")
    if len(sources) != depth_tensor.shape[0]:
        raise EvaluationError(
            "fixture source count does not match model output batch dimension"
        )
    frames_by_sha: dict[str, Mapping[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise EvaluationError("annotation frame must be an object")
        rgb = frame.get("rgb")
        if not isinstance(rgb, Mapping):
            raise EvaluationError("annotation frame has no RGB identity")
        sha = str(rgb.get("sha256") or "")
        if not sha or sha in frames_by_sha:
            raise EvaluationError("annotation RGB identities must be non-empty and unique")
        rgb_path = _resolve_corpus_path(annotations_path, rgb.get("path"))
        _require_sha(rgb_path, sha, label="annotated RGB")
        frames_by_sha[sha] = frame

    camera_results: dict[str, dict[str, Any]] = {}
    for batch_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise EvaluationError("fixture source must be an object")
        source_sha = str(source.get("sha256") or "")
        frame = frames_by_sha.get(source_sha)
        if frame is None:
            raise EvaluationError(
                f"fixture source {source_sha} has no independent annotation"
            )
        camera_id = str(frame.get("camera_id") or "")
        if not camera_id or camera_id in camera_results:
            raise EvaluationError("camera IDs must be non-empty and unique")
        if camera_id not in calibration_arrays:
            raise EvaluationError(f"no authored extrinsics for {camera_id}")
        intrinsics_key = f"{camera_id}:K"
        if intrinsics_key not in calibration_arrays:
            raise EvaluationError(f"no authored rectified intrinsics for {camera_id}")

        source_width = int(source.get("source_width") or 0)
        source_height = int(source.get("source_height") or 0)
        resized_width = int(source.get("resized_width") or 0)
        resized_height = int(source.get("resized_height") or 0)
        pad_left = int(source.get("pad_left") or 0)
        pad_top = int(source.get("pad_top") or 0)
        rgb = frame["rgb"]
        if [source_width, source_height] != list(rgb.get("size") or []):
            raise EvaluationError(
                f"fixture and annotation RGB dimensions differ for {camera_id}"
            )
        output_shape = tuple(int(value) for value in depth_tensor.shape[-2:])
        if output_shape != tuple(
            int(value) for value in receipt.get("tensor", {}).get("shape", ())[-2:]
        ):
            raise EvaluationError("model output and fixture tensor dimensions differ")

        depth = depth_tensor[batch_index]
        conf = conf_tensor[batch_index]
        mask = mask_tensor[batch_index]
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(conf)
            & np.isfinite(mask)
            & (mask >= 0.5)
        )
        u_source, v_source = _source_pixel_grids(
            source_size=(source_width, source_height),
            output_shape=output_shape,
            resized_size=(resized_width, resized_height),
            pad_left=pad_left,
            pad_top=pad_top,
        )
        world_y = _world_y_grid(
            depth,
            u_source=u_source,
            v_source=v_source,
            intrinsics=calibration_arrays[intrinsics_key],
            camera_to_world=calibration_arrays[camera_id],
        )

        camera_result: dict[str, Any] = {
            "camera_id": camera_id,
            "batch_index": batch_index,
            "rgb_sha256": source_sha,
            "input_mapping": {
                "source_size": [source_width, source_height],
                "resized_size": [resized_width, resized_height],
                "pad_left": pad_left,
                "pad_top": pad_top,
                "output_shape": list(output_shape),
            },
            "model_valid_fraction": float(np.mean(valid)),
        }
        for annotation_key, result_key, semantic in (
            ("floor_regions", "floor_regions", "floor"),
            ("obstacle_regions", "obstacle_regions", "obstacle"),
        ):
            regions = frame.get(annotation_key, [])
            if not isinstance(regions, list):
                raise EvaluationError(f"{camera_id}.{annotation_key} must be a list")
            region_results: list[dict[str, Any]] = []
            aggregate_mask = np.zeros(output_shape, dtype=bool)
            for region in regions:
                if not isinstance(region, Mapping):
                    raise EvaluationError(f"{camera_id} region must be an object")
                region_mask = _mapped_polygon_mask(
                    region.get("polygon_px", []),
                    source_size=(source_width, source_height),
                    output_shape=output_shape,
                    resized_size=(resized_width, resized_height),
                    pad_left=pad_left,
                    pad_top=pad_top,
                    erode_px_source=float(region.get("erode_px", 0.0)),
                )
                aggregate_mask |= region_mask
                metrics = _region_metrics(
                    region_mask=region_mask,
                    valid=valid,
                    depth=depth,
                    world_y=world_y,
                    floor_y=floor_y,
                    semantic=semantic,
                )
                region_results.append(
                    {
                        "id": str(region.get("id") or ""),
                        "annotation_confidence": str(
                            region.get("annotation_confidence") or "unspecified"
                        ),
                        **metrics,
                    }
                )
            aggregate = _region_metrics(
                region_mask=aggregate_mask,
                valid=valid,
                depth=depth,
                world_y=world_y,
                floor_y=floor_y,
                semantic=semantic,
            )
            camera_result[result_key] = region_results
            camera_result[f"{result_key}_aggregate"] = aggregate

        edges = frame.get("structural_edges", [])
        if not isinstance(edges, list):
            raise EvaluationError(f"{camera_id}.structural_edges must be a list")
        camera_result["structural_edges"] = [
            _edge_metrics(
                edge=edge,
                depth=depth,
                valid=valid,
                source_size=(source_width, source_height),
                resized_size=(resized_width, resized_height),
                pad_left=pad_left,
                pad_top=pad_top,
            )
            for edge in edges
            if isinstance(edge, Mapping)
        ]
        room_boundaries = frame.get("room_boundaries", [])
        if not isinstance(room_boundaries, list):
            raise EvaluationError(f"{camera_id}.room_boundaries must be a list")
        boundary_inventory: list[dict[str, Any]] = []
        for boundary in room_boundaries:
            if not isinstance(boundary, Mapping):
                raise EvaluationError(f"{camera_id} room boundary must be an object")
            points = np.asarray(boundary.get("polyline_px"), dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
                raise EvaluationError(
                    f"{camera_id} room boundary must have at least two [x,y] points"
                )
            if (
                not np.all(np.isfinite(points))
                or np.any(points[:, 0] < 0)
                or np.any(points[:, 0] >= source_width)
                or np.any(points[:, 1] < 0)
                or np.any(points[:, 1] >= source_height)
            ):
                raise EvaluationError(
                    f"{camera_id} room boundary lies outside the annotated RGB frame"
                )
            boundary_inventory.append(
                {
                    "id": str(boundary.get("id") or ""),
                    "kind": str(boundary.get("kind") or ""),
                    "scope": str(boundary.get("scope") or ""),
                    "annotation_confidence": str(
                        boundary.get("annotation_confidence") or "unspecified"
                    ),
                    "status": "annotation_only_no_topdown_metric",
                }
            )
        camera_result["room_boundaries"] = boundary_inventory
        camera_results[camera_id] = camera_result

    floor_p50_values: list[float] = []
    floor_p95_values: list[float] = []
    obstacle_values: list[float] = []
    edge_p50_values: list[float] = []
    edge_p95_values: list[float] = []
    floor_annotation_camera_count = 0
    obstacle_annotation_camera_count = 0
    structural_edge_annotation_count = 0
    room_boundary_annotation_count = 0
    for result in camera_results.values():
        if result.get("floor_regions"):
            floor_annotation_camera_count += 1
        if result.get("obstacle_regions"):
            obstacle_annotation_camera_count += 1
        structural_edge_annotation_count += len(result.get("structural_edges", []))
        room_boundary_annotation_count += len(result.get("room_boundaries", []))
        floor_aggregate = result.get("floor_regions_aggregate", {})
        floor_residual = (
            floor_aggregate.get("floor_plane_residual_m", {})
            if isinstance(floor_aggregate, Mapping)
            else {}
        )
        if isinstance(floor_residual, Mapping):
            if isinstance(floor_residual.get("abs_p50"), (int, float)):
                floor_p50_values.append(float(floor_residual["abs_p50"]))
            if isinstance(floor_residual.get("abs_p95"), (int, float)):
                floor_p95_values.append(float(floor_residual["abs_p95"]))
        obstacle_aggregate = result.get("obstacle_regions_aggregate", {})
        obstacle_clearance = (
            obstacle_aggregate.get("obstacle_clearance", {})
            if isinstance(obstacle_aggregate, Mapping)
            else {}
        )
        if isinstance(obstacle_clearance, Mapping) and isinstance(
            obstacle_clearance.get("above_0_15m_fraction"), (int, float)
        ):
            obstacle_values.append(
                float(obstacle_clearance["above_0_15m_fraction"])
            )
        for edge_result in result.get("structural_edges", []):
            distance = edge_result.get(
                "nearest_log_depth_gradient_distance_px_source", {}
            )
            if not isinstance(distance, Mapping):
                continue
            if isinstance(distance.get("p50"), (int, float)):
                edge_p50_values.append(float(distance["p50"]))
            if isinstance(distance.get("p95"), (int, float)):
                edge_p95_values.append(float(distance["p95"]))

    annotation_blockers = annotations.get("blocked_measurements", [])
    return {
        "contract": REPORT_CONTRACT,
        "candidate_id": candidate_id,
        "identity": {
            "annotations": {
                "path": str(annotations_path.resolve()),
                "sha256": _sha256_file(annotations_path),
            },
            "fixture_receipt": {
                "path": str(fixture_receipt_path.resolve()),
                "sha256": _sha256_file(fixture_receipt_path),
            },
            "model_output": {
                "path": str(
                    (
                        model_output_identity_path
                        if model_output_identity_path is not None
                        else model_output_path
                    ).resolve()
                ),
                "sha256": _sha256_file(model_output_path),
            },
            "profile": str(receipt.get("profile") or ""),
        },
        "evidence_classification": {
            "rgb_semantics": "independent_human_annotation",
            "floor_plane": "authored_calibration",
            "depth_confidence_mask": "candidate_under_evaluation",
            "floorplan_or_other_model_output_used_as_truth": False,
        },
        "calibration": {
            "camera_calibration_sha256": calibration["camera_calibration"][
                "sha256"
            ],
            "floor_alignment_sha256": calibration["floor_alignment"]["sha256"],
            "floor_y_m": floor_y,
        },
        "cameras": camera_results,
        "macro": {
            "camera_count": len(camera_results),
            "annotation_coverage": {
                "floor_camera_count": floor_annotation_camera_count,
                "obstacle_camera_count": obstacle_annotation_camera_count,
                "structural_edge_count": structural_edge_annotation_count,
                "room_boundary_snippet_count": room_boundary_annotation_count,
            },
            "floor_plane_abs_residual_m": {
                "camera_median_p50": (
                    float(np.median(floor_p50_values))
                    if floor_p50_values
                    else None
                ),
                "camera_median_p95": (
                    float(np.median(floor_p95_values))
                    if floor_p95_values
                    else None
                ),
            },
            "obstacle_above_floor_0_15m_fraction_camera_median": (
                float(np.median(obstacle_values)) if obstacle_values else None
            ),
            "structural_edge_distance_px_source": {
                "edge_median_p50": (
                    float(np.median(edge_p50_values))
                    if edge_p50_values
                    else None
                ),
                "edge_median_p95": (
                    float(np.median(edge_p95_values))
                    if edge_p95_values
                    else None
                ),
            },
        },
        "blocked_measurements": annotation_blockers,
        "metric_limitations": {
            "floor_plane_residual": (
                "Diagnostic consistency with authored calibration on RGB-labeled "
                "patches; it is not surveyed range ground truth."
            ),
            "obstacle_clearance": (
                "Diagnostic height-above-floor support inside visible obstacle "
                "patches; it is not obstacle boundary precision or recall."
            ),
            "structural_edge_distance": (
                "Nearest maximum log-depth gradient within the search radius may "
                "select an unrelated nearby edge and cannot support promotion alone."
            ),
            "room_boundaries": (
                "Visible RGB snippets are inventoried only; no complete top-down "
                "room-boundary metric is claimed."
            ),
        },
        "claims": {
            "absolute_object_distance_accuracy": False,
            "room_boundary_precision_recall": False,
            "floorplan_obstacle_precision_recall": False,
            "candidate_promotion_supported_by_this_report_alone": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one MapAnything candidate against independent RGB "
            "annotations and authored calibration."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = evaluate(
            annotations_path=args.annotations.resolve(),
            fixture_receipt_path=args.fixture_receipt.resolve(),
            model_output_path=args.model_output.resolve(),
            candidate_id=str(args.candidate_id),
            repo_root=args.repo_root.resolve(),
        )
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256_file(output),
                "candidate_id": args.candidate_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
