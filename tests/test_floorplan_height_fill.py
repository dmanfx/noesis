from __future__ import annotations

import numpy as np

from geometry.depth_source import _postprocess_floorplan_height_grid


def test_non_kitchen_uses_legacy_min_fill() -> None:
    height = np.array(
        [
            [2.0, np.nan],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    density = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.7],
        ],
        dtype=np.float32,
    )

    out, meta = _postprocess_floorplan_height_grid(
        "living-room",
        height,
        density,
        min_x=-1.0,
        max_x=1.0,
        min_z=0.0,
        max_z=2.0,
    )

    assert np.isfinite(out).all()
    assert float(out[0, 1]) == np.float32(1.0)
    assert meta["mode"] == "legacy_min_fill"
    assert int(meta["filled_cells"]) == 1


def test_kitchen_uses_legacy_min_fill_too() -> None:
    height = np.array(
        [
            [1.0, np.nan],
            [3.0, np.nan],
        ],
        dtype=np.float32,
    )
    density = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    out, meta = _postprocess_floorplan_height_grid(
        "kitchen",
        height,
        density,
        min_x=-1.0,
        max_x=1.0,
        min_z=0.0,
        max_z=2.0,
    )

    assert np.isfinite(out).all()
    assert float(out[0, 1]) == np.float32(1.0)
    assert float(out[1, 1]) == np.float32(1.0)
    assert meta["mode"] == "legacy_min_fill"
    assert int(meta["filled_cells"]) == 2


def test_empty_floor_fallback() -> None:
    height = np.full((2, 2), np.nan, dtype=np.float32)
    density = np.zeros((2, 2), dtype=np.float32)

    out, meta = _postprocess_floorplan_height_grid(
        "kitchen",
        height,
        density,
        min_x=-1.0,
        max_x=1.0,
        min_z=0.0,
        max_z=2.0,
    )

    assert np.isfinite(out).all()
    assert float(np.max(out)) == 0.0
    assert meta["mode"] == "empty_floor_fallback"
