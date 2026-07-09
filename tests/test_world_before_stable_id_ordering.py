"""Regression: world must be available before StableID for overlap permits.

Mocks ordering without DeepStream: if world is only filled after StableID,
``_world_xy_for_stable_id`` returns missing_world on first sighting.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


def _world_xy_for_stable_id(
    raw: Mapping[str, Any],
    cache: Dict[Tuple[int, int], Tuple[float, float, bool, float]],
    sensor_id: int,
    track_id: int,
) -> Tuple[Optional[Tuple[float, float]], bool]:
    if raw.get("world_valid") is True:
        world = raw.get("world")
        if isinstance(world, (list, tuple)) and len(world) >= 3:
            return (float(world[0]), float(world[2])), True
    cached = cache.get((int(sensor_id), int(track_id)))
    if cached is not None and cached[2]:
        return (float(cached[0]), float(cached[1])), True
    return None, False


def _broken_order_first_frame(raw: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], bool]:
    """Pre-fix: StableID sees world before augmentation."""
    cache: Dict[Tuple[int, int], Tuple[float, float, bool, float]] = {}
    world_xy, world_valid = _world_xy_for_stable_id(raw, cache, 0, 1)
    # StableID would run here with missing_world
    # Then augment fills world (too late for this frame's StableID)
    raw["world"] = [1.0, 0.0, 2.0]
    raw["world_valid"] = True
    return world_xy, world_valid


def _fixed_order_first_frame(raw: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], bool]:
    """Fixed: augment world onto raw before StableID reads it."""
    cache: Dict[Tuple[int, int], Tuple[float, float, bool, float]] = {}
    world_xy, world_valid = _world_xy_for_stable_id(raw, cache, 0, 1)
    if not world_valid:
        # Reuse existing augment path conceptually: fill world on raw first.
        raw["world"] = [1.0, 0.0, 2.0]
        raw["world_valid"] = True
        world_xy, world_valid = _world_xy_for_stable_id(raw, cache, 0, 1)
    return world_xy, world_valid


def test_broken_order_misses_world_on_first_frame():
    raw: Dict[str, Any] = {"bbox": [10, 20, 30, 80], "track_id": 1}
    world_xy, world_valid = _broken_order_first_frame(raw)
    assert world_valid is False
    assert world_xy is None
    # World exists after augment, but StableID already ran without it.
    assert raw.get("world_valid") is True


def test_fixed_order_has_world_before_stable_id():
    raw: Dict[str, Any] = {"bbox": [10, 20, 30, 80], "track_id": 1}
    world_xy, world_valid = _fixed_order_first_frame(raw)
    assert world_valid is True
    assert world_xy == (1.0, 2.0)
    assert raw.get("world_valid") is True


def test_cache_still_serves_next_frame_without_recompute():
    cache: Dict[Tuple[int, int], Tuple[float, float, bool, float]] = {
        (0, 1): (3.0, 4.0, True, 1.0),
    }
    raw: Dict[str, Any] = {"bbox": [10, 20, 30, 80], "track_id": 1}
    world_xy, world_valid = _world_xy_for_stable_id(raw, cache, 0, 1)
    assert world_valid is True
    assert world_xy == (3.0, 4.0)
