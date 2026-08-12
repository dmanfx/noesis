"""Shared fail-closed geometry and timing contracts for V3DT validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


V3DT_BBOX_DEFAULT_ATTEMPTS = 8
V3DT_BBOX_MAX_ATTEMPTS = 32
V3DT_BBOX_RETRY_DELAY_SECONDS = 1.0
V3DT_BBOX_RUNNER_MARGIN_SECONDS = 20.0


class V3DTAxisMapError(ValueError):
    """The configured tracker-to-world axis map is absent or ambiguous."""


def _finite_triplet(value: Sequence[object], *, label: str) -> tuple[float, float, float]:
    try:
        length = len(value)
    except TypeError as exc:
        raise V3DTAxisMapError(
            f"{label} must contain exactly three coordinates"
        ) from exc
    if isinstance(value, (str, bytes, bytearray)) or length != 3:
        raise V3DTAxisMapError(f"{label} must contain exactly three coordinates")
    coordinates: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise V3DTAxisMapError(f"{label} must contain finite numeric coordinates")
        try:
            coordinate = float(item)
        except (OverflowError, TypeError, ValueError) as exc:
            raise V3DTAxisMapError(
                f"{label} must contain finite numeric coordinates"
            ) from exc
        if not math.isfinite(coordinate):
            raise V3DTAxisMapError(f"{label} must contain finite numeric coordinates")
        coordinates.append(coordinate)
    return coordinates[0], coordinates[1], coordinates[2]


@dataclass(frozen=True)
class V3DTAxisMap:
    """One signed-permutation map from tracker coordinates to canonical world.

    ``scripts/generate_v3dt_caminfo.py`` right-multiplies the canonical
    world-to-camera extrinsics by the same matrix. Consequently a point emitted
    by the tracker must be multiplied by this matrix before it is published in
    ``backend_world_m``. Keeping the transform here prevents DS8, DS9, and live
    evidence tooling from independently interpreting a compact axis string.
    """

    spec: str
    # Each tuple is ``(canonical_axis_index, sign)`` for one tracker input axis.
    columns: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @classmethod
    def parse(cls, spec: object) -> "V3DTAxisMap":
        if not isinstance(spec, str):
            raise V3DTAxisMapError("V3DT camInfo world-axis map must be a string")
        normalized = spec.strip().lower()
        if not normalized:
            raise V3DTAxisMapError("V3DT camInfo world-axis map is required")

        tokens = [token for token in normalized.replace(",", " ").split() if token]
        if len(tokens) == 1 and len(tokens[0]) == 3:
            if any(character not in "xyz" for character in tokens[0]):
                raise V3DTAxisMapError("compact V3DT axis maps must be xyz permutations")
            tokens = list(tokens[0])
        if len(tokens) != 3:
            raise V3DTAxisMapError(
                "V3DT camInfo world-axis map must contain exactly three axes"
            )

        axis_indices = {"x": 0, "y": 1, "z": 2}
        used: set[str] = set()
        columns: list[tuple[int, int]] = []
        canonical_tokens: list[str] = []
        for token in tokens:
            sign = -1 if token.startswith("-") else 1
            axis = token.lstrip("+-")
            if axis not in axis_indices or axis in used:
                raise V3DTAxisMapError(
                    "V3DT camInfo world-axis map must be a signed xyz permutation"
                )
            if token.startswith(("+", "-")) and len(token) != 2:
                raise V3DTAxisMapError(
                    "V3DT camInfo world-axis map contains a malformed signed axis"
                )
            used.add(axis)
            columns.append((axis_indices[axis], sign))
            canonical_tokens.append(("-" if sign < 0 else "") + axis)

        canonical_spec = (
            "".join(canonical_tokens)
            if all(len(token) == 1 for token in canonical_tokens)
            else " ".join(canonical_tokens)
        )
        return cls(
            spec=canonical_spec,
            columns=(columns[0], columns[1], columns[2]),
        )

    def tracker_to_world(self, point: Sequence[object]) -> tuple[float, float, float]:
        tracker = _finite_triplet(point, label="V3DT tracker point")
        world = [0.0, 0.0, 0.0]
        for tracker_axis, (world_axis, sign) in enumerate(self.columns):
            world[world_axis] = float(sign) * tracker[tracker_axis]
        return world[0], world[1], world[2]

    def world_to_tracker(self, point: Sequence[object]) -> tuple[float, float, float]:
        world = _finite_triplet(point, label="canonical world point")
        tracker = [0.0, 0.0, 0.0]
        for tracker_axis, (world_axis, sign) in enumerate(self.columns):
            tracker[tracker_axis] = float(sign) * world[world_axis]
        return tracker[0], tracker[1], tracker[2]


def v3dt_bbox3d_tracker_foot(bbox3d: Mapping[str, object]) -> tuple[float, float, float]:
    """Return the tracker-space ground endpoint for an xzy-remapped camInfo.

    The locked tracker profile emits a Z-up cuboid because canonical Y was
    mapped into tracker Z when its camInfo projection was generated. This is an
    explicit property of that camInfo contract, not a claim about the default
    SDK axis convention.
    """

    try:
        center = _finite_triplet(
            (bbox3d["xCentre"], bbox3d["yCentre"], bbox3d["zCentre"]),
            label="V3DT bbox3d center",
        )
        z_len = float(bbox3d["zLen"])
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise V3DTAxisMapError("V3DT bbox3d lacks a finite Z extent") from exc
    if not math.isfinite(z_len) or z_len <= 0.0:
        raise V3DTAxisMapError("V3DT bbox3d Z extent must be finite and positive")
    return center[0], center[1], center[2] - (0.5 * z_len)


def v3dt_bbox3d_world_foot(
    bbox3d: Mapping[str, object],
    axis_map: V3DTAxisMap,
) -> tuple[float, float, float]:
    if not isinstance(axis_map, V3DTAxisMap):
        raise V3DTAxisMapError("V3DT bbox3d conversion requires a parsed axis map")
    return axis_map.tracker_to_world(v3dt_bbox3d_tracker_foot(bbox3d))


@dataclass(frozen=True)
class V3DTBBoxTimeoutContract:
    """One bounded retry schedule shared by the gate and its orchestrator."""

    duration_seconds: float
    attempts: int = V3DT_BBOX_DEFAULT_ATTEMPTS
    retry_delay_seconds: float = V3DT_BBOX_RETRY_DELAY_SECONDS
    runner_margin_seconds: float = V3DT_BBOX_RUNNER_MARGIN_SECONDS

    def __post_init__(self) -> None:
        duration = float(self.duration_seconds)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("V3DT bbox duration must be finite and > 0")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise ValueError("V3DT bbox attempts must be an integer")
        if not 1 <= self.attempts <= V3DT_BBOX_MAX_ATTEMPTS:
            raise ValueError(
                "V3DT bbox attempts must be in "
                f"[1, {V3DT_BBOX_MAX_ATTEMPTS}]"
            )
        retry_delay = float(self.retry_delay_seconds)
        if not math.isfinite(retry_delay) or retry_delay < 0.0:
            raise ValueError("V3DT bbox retry delay must be finite and >= 0")
        margin = float(self.runner_margin_seconds)
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("V3DT bbox runner margin must be finite and > 0")

    @property
    def collection_envelope_seconds(self) -> float:
        """Maximum intentional collection and between-attempt delay budget."""

        return (
            self.attempts * float(self.duration_seconds)
            + (self.attempts - 1) * float(self.retry_delay_seconds)
        )

    @property
    def runner_timeout_seconds(self) -> float:
        """Outer process timeout including startup/teardown scheduling margin."""

        return self.collection_envelope_seconds + float(self.runner_margin_seconds)


__all__ = [
    "V3DTAxisMap",
    "V3DTAxisMapError",
    "V3DTBBoxTimeoutContract",
    "V3DT_BBOX_DEFAULT_ATTEMPTS",
    "V3DT_BBOX_MAX_ATTEMPTS",
    "V3DT_BBOX_RETRY_DELAY_SECONDS",
    "V3DT_BBOX_RUNNER_MARGIN_SECONDS",
    "v3dt_bbox3d_tracker_foot",
    "v3dt_bbox3d_world_foot",
]
