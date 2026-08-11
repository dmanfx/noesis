from __future__ import annotations

import configparser
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


logger = logging.getLogger(__name__)

_COMMON_SOURCE_SIZES = (
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)


@dataclass(frozen=True)
class DewarperFovSpec:
    config_path: Path
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    source_k: np.ndarray
    rectified_k: np.ndarray
    distortion: np.ndarray
    source_valid_region: str = "rect"
    source_valid_radius_px: float | None = None
    source_valid_radius_scale: float = 1.0


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _positive_int(value: Any) -> int | None:
    parsed = _positive_float(value)
    return int(parsed) if parsed is not None and int(parsed) > 0 else None


def _parse_float_list(raw: Any, *, count: int, key: str, path: Path) -> list[float]:
    values = [part.strip() for part in str(raw or "").split(";") if part.strip()]
    if len(values) < count:
        raise ValueError(f"{path}: {key} requires {count} values, got {len(values)}")
    parsed = [float(value) for value in values[:count]]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{path}: {key} contains non-finite values")
    return parsed


def _resolve_config_path(raw: str | Path, pipeline_yaml_path: Path | None, repo_root: Path | None) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if pipeline_yaml_path is not None:
        local = (Path(pipeline_yaml_path).resolve().parent / candidate).resolve()
        if local.exists():
            return local
    if repo_root is not None:
        return (Path(repo_root).resolve() / candidate).resolve()
    return candidate.resolve()


def _source_size(
    source_cfg: Mapping[str, Any],
    mask_cfg: Mapping[str, Any],
    *,
    cx: float,
    cy: float,
    output_size: tuple[int, int],
) -> tuple[int, int]:
    dewarper = source_cfg.get("dewarper") if isinstance(source_cfg.get("dewarper"), Mapping) else {}
    for mapping in (mask_cfg, source_cfg, dewarper):
        width = _positive_int(mapping.get("source-width", mapping.get("source_width")))
        height = _positive_int(mapping.get("source-height", mapping.get("source_height")))
        if width is not None and height is not None:
            return width, height

    inferred_w = max(1.0, cx * 2.0)
    inferred_h = max(1.0, cy * 2.0)
    candidates = [
        size
        for size in (*_COMMON_SOURCE_SIZES, output_size)
        if size[0] > cx + 1.0 and size[1] > cy + 1.0
    ]
    if not candidates:
        raise ValueError("source size is not declared and cannot be inferred from the principal point")
    return min(candidates, key=lambda size: abs(size[0] - inferred_w) + abs(size[1] - inferred_h))


def load_dewarper_fov_spec(
    *,
    source_cfg: Mapping[str, Any],
    pipeline_yaml_path: Path | None,
    mask_cfg: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> DewarperFovSpec | None:
    """Load an explicit nvdewarper FoV validity contract.

    Missing validity configuration disables the mask. Invalid configured data
    raises instead of silently pretending the full dewarped rectangle is valid.
    """

    mask_cfg = mask_cfg if isinstance(mask_cfg, Mapping) else {}
    if not bool(mask_cfg.get("enable", True)):
        return None
    dewarper = source_cfg.get("dewarper")
    if not isinstance(dewarper, Mapping) or not bool(dewarper.get("enable", False)):
        return None
    config_raw = str(dewarper.get("config-file") or "").strip()
    if not config_raw:
        raise ValueError("dewarper validity configured for a source without dewarper.config-file")
    config_path = _resolve_config_path(config_raw, pipeline_yaml_path, repo_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"dewarper config not found: {config_path}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), strict=False)
    parser.optionxform = str
    with config_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if "property" not in parser or "surface0" not in parser:
        raise ValueError(f"{config_path}: expected [property] and [surface0]")
    props = parser["property"]
    surface = parser["surface0"]
    projection_type = int(float(surface.get("projection-type", "0")))
    if projection_type != 4:
        raise ValueError(f"{config_path}: only FISH_PERSPECTIVE projection-type=4 is supported")

    output_w = _positive_int(props.get("output-width")) or _positive_int(surface.get("width"))
    output_h = _positive_int(props.get("output-height")) or _positive_int(surface.get("height"))
    if output_w is None or output_h is None:
        raise ValueError(f"{config_path}: invalid dewarper output size")
    source_focal = _parse_float_list(surface.get("focal-length"), count=2, key="focal-length", path=config_path)
    distortion = _parse_float_list(surface.get("distortion"), count=4, key="distortion", path=config_path)
    destination_focal = _parse_float_list(
        surface.get("dst-focal-length", surface.get("focal-length")),
        count=2,
        key="dst-focal-length",
        path=config_path,
    )
    destination_center = _parse_float_list(
        surface.get("dst-principal-point", f"{(output_w - 1) * 0.5};{(output_h - 1) * 0.5}"),
        count=2,
        key="dst-principal-point",
        path=config_path,
    )
    try:
        source_cx = float(surface["src-x0"])
        source_cy = float(surface["src-y0"])
    except Exception as exc:
        raise ValueError(f"{config_path}: numeric src-x0/src-y0 are required") from exc

    source_size = _source_size(
        source_cfg,
        mask_cfg,
        cx=source_cx,
        cy=source_cy,
        output_size=(output_w, output_h),
    )
    region = str(mask_cfg.get("source-valid-region", mask_cfg.get("source_valid_region", "rect"))).strip().lower()
    if region in {"rectangle", "frame"}:
        region = "rect"
    if region not in {"rect", "circle", "ellipse"}:
        raise ValueError(f"unsupported source-valid-region: {region}")
    radius = _positive_float(mask_cfg.get("source-valid-radius-px", mask_cfg.get("source_valid_radius_px")))
    radius_scale = _positive_float(
        mask_cfg.get("source-valid-radius-scale", mask_cfg.get("source_valid_radius_scale", 1.0))
    ) or 1.0

    return DewarperFovSpec(
        config_path=config_path,
        source_size=source_size,
        output_size=(output_w, output_h),
        source_k=np.asarray(
            ((source_focal[0], 0.0, source_cx), (0.0, source_focal[1], source_cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        rectified_k=np.asarray(
            (
                (destination_focal[0], 0.0, destination_center[0]),
                (0.0, destination_focal[1], destination_center[1]),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(4, 1),
        source_valid_region=region,
        source_valid_radius_px=radius,
        source_valid_radius_scale=radius_scale,
    )


def build_dewarper_fov_mask(
    spec: DewarperFovSpec,
    *,
    target_size: tuple[int, int],
    erode_px: int = 1,
) -> np.ndarray:
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_size must be positive")
    output_w, output_h = spec.output_size
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        spec.source_k,
        spec.distortion,
        np.eye(3, dtype=np.float64),
        spec.rectified_k,
        (output_w, output_h),
        cv2.CV_32FC1,
    )
    source_w, source_h = spec.source_size
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x < float(source_w))
        & (map_y < float(source_h))
    )
    if spec.source_valid_region == "circle":
        cx, cy = float(spec.source_k[0, 2]), float(spec.source_k[1, 2])
        radius = spec.source_valid_radius_px
        if radius is None:
            radius = max(cx, cy, float(source_w - 1) - cx, float(source_h - 1) - cy)
        radius = max(1.0, radius * spec.source_valid_radius_scale)
        valid &= ((map_x - cx) ** 2 + (map_y - cy) ** 2) <= radius**2
    elif spec.source_valid_region == "ellipse":
        cx, cy = float(spec.source_k[0, 2]), float(spec.source_k[1, 2])
        scale = spec.source_valid_radius_scale
        radius_x = max(cx, float(source_w - 1) - cx, 1.0) * scale
        radius_y = max(cy, float(source_h - 1) - cy, 1.0) * scale
        valid &= (((map_x - cx) / radius_x) ** 2 + ((map_y - cy) / radius_y) ** 2) <= 1.0
    if erode_px > 0 and np.any(valid):
        size = (int(erode_px) * 2) + 1
        valid = cv2.erode(valid.astype(np.uint8), np.ones((size, size), dtype=np.uint8), iterations=1).astype(bool)
    if (target_w, target_h) != (output_w, output_h):
        valid = cv2.resize(valid.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return np.asarray(valid, dtype=bool)


# Transitional names retained for the focused regression test and older tools.
_DewarperFovSpec = DewarperFovSpec
_load_dewarper_fov_spec = load_dewarper_fov_spec
_build_dewarper_fov_mask = build_dewarper_fov_mask
