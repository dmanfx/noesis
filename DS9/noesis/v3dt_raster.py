"""Validated pixel-raster transforms for V3DT camera projection matrices."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def validate_raster(value: Any, label: str) -> tuple[int, int]:
    """Return a positive ``(width, height)`` pair without coercing bad input.

    Integral scalar types are accepted, but booleans and floating-point values
    are rejected even when they could be converted to integer dimensions.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a two-integer (width, height) raster")
    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{label} must be a two-integer (width, height) raster"
        ) from exc
    if len(dimensions) != 2:
        raise ValueError(f"{label} must contain exactly width and height")

    result: list[int] = []
    for axis, dimension in zip(("width", "height"), dimensions):
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise ValueError(f"{label} {axis} must be an integer")
        normalized = int(dimension)
        if normalized <= 0:
            raise ValueError(f"{label} {axis} must be positive")
        result.append(normalized)
    return result[0], result[1]


def scale_projection_matrix(
    projection: Any,
    from_size: Any,
    to_size: Any,
) -> np.ndarray:
    """Scale a finite 3x4 world-to-pixel projection between raster sizes.

    Raster tuples are ``(width, height)``. The x and y homogeneous projection
    rows are scaled independently; the projective denominator row is unchanged.
    This performs a zero-offset raster scale only and does not alter camera or
    ground geometry.
    """
    source_width, source_height = validate_raster(from_size, "from_size")
    target_width, target_height = validate_raster(to_size, "to_size")

    try:
        matrix = np.asarray(projection, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("projection must be a finite 3x4 matrix or 12 values") from exc
    if matrix.shape == (12,):
        matrix = matrix.reshape(3, 4)
    elif matrix.shape != (3, 4):
        raise ValueError("projection must have shape (3, 4) or contain 12 values")
    if not np.isfinite(matrix).all():
        raise ValueError("projection must contain only finite values")

    row_scales = np.asarray(
        (
            target_width / source_width,
            target_height / source_height,
            1.0,
        ),
        dtype=np.float64,
    )
    return matrix * row_scales[:, np.newaxis]
