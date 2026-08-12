from __future__ import annotations

import hashlib
import json
import base64
from pathlib import Path

import numpy as np

from noesis_core.scene_fusion import SceneFusionSet


def _decode_layer(layer: dict[str, object]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(str(layer["grid_b64"])), dtype=np.float32).reshape(
        tuple(layer["grid_shape"])
    )


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def test_scene_fusion_loads_and_composes_canonical_cached_layers(tmp_path: Path) -> None:
    revision = tmp_path / "revisions" / "fusion-a"
    revision.mkdir(parents=True)
    points = {
        "points_world_m": np.asarray([[0.5, 0.5, 1.5]], dtype=np.float32),
        "colors_rgb_u8": np.asarray([[10, 20, 30]], dtype=np.uint8),
        "confidence": np.asarray([0.8], dtype=np.float32),
        "provenance": np.asarray([3], dtype=np.uint8),
        "view_support": np.asarray([2], dtype=np.uint16),
    }
    shape = (2, 2)
    grid = {
        "height_agl_p95_m": np.asarray([[1, 2], [3, 4]], dtype=np.float32),
        "observed": np.ones(shape, dtype=np.uint8),
        "confidence": np.full(shape, 0.8, dtype=np.float32),
        "point_count": np.ones(shape, dtype=np.uint32),
        "obstacle_height_m": np.asarray([[1, np.nan], [2, 3]], dtype=np.float32),
        "obstacle_mask": np.asarray([[1, 0], [1, 1]], dtype=np.uint8),
        "floor_support_count": np.ones(shape, dtype=np.uint32),
        "floor_supported": np.ones(shape, dtype=np.uint8),
        "provenance": np.asarray([[1, 2], [3, 1]], dtype=np.uint8),
    }
    np.savez_compressed(revision / "fused_points.npz", **points)
    np.savez_compressed(revision / "fused_grid.npz", **grid)
    manifest = {
        "contract": "noesis.scene_fusion.diagnostic",
        "contract_version": 1,
        "fusion_id": "fusion-a",
        "status": "passed",
        "diagnostic_only": True,
        "site_id": "site-a",
        "space_id": "room-a",
        "camera_id": "camera-a",
        "floor_y_m": 0.0,
        "reference_camera_to_world_row_major": np.eye(4).reshape(-1).tolist(),
        "method": {"inference": "joint"},
        "quality": {"fixed_to_phone_median_m": 0.1},
        "point_cloud": {
            "point_count": 1,
            "source_counts": {"fixed_only": 0, "phone_only": 0, "fixed_phone_agreement": 1},
        },
        "grid": {
            "bounds": {"min_x": 0, "max_x": 2, "min_z": 0, "max_z": 2},
            "rows": 2,
            "columns": 2,
            "orientation": "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x",
        },
    }
    (revision / "manifest.json").write_text(json.dumps(manifest))
    catalog = {
        "contract": "noesis.scene_fusion.catalog",
        "contract_version": 1,
        "site_id": "site-a",
        "camera_bindings": [
            {
                "camera_id": "camera-a",
                "manifest": _descriptor(revision / "manifest.json", tmp_path),
                "points": _descriptor(revision / "fused_points.npz", tmp_path),
                "grid": _descriptor(revision / "fused_grid.npz", tmp_path),
            }
        ],
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))

    fusions = SceneFusionSet.load(tmp_path / "catalog.json")
    result = fusions.compose_floorplan(
        "camera-a",
        {
            "frame": "camera_local_ground_m",
            "units": "meters",
            "bounds": {"min_x": 0, "max_x": 2, "min_z": 0, "max_z": 2},
            "height_agl": {"grid_shape": [4, 4]},
        },
    )

    assert result["scene_fusion_meta"]["fusion_id"] == "fusion-a"
    assert result["scene_fusion_points"]["point_count"] == 1
    assert result["scene_fusion_height_agl"]["grid_shape"] == [4, 4]
    assert result["scene_fusion_diagnostic_density"]["grid_shape"] == [2, 2]
    assert result["scene_fusion_diagnostic_obstacle_height"]["grid_shape"] == [2, 2]
    assert result["scene_fusion_diagnostic_surface_rgb"]["rgb_shape"] == [2, 2, 3]
    assert np.array_equal(
        _decode_layer(result["scene_fusion_diagnostic_walkable"]),
        np.asarray([[0, 1], [0, 0]], dtype=np.float32),
    )
    diagnostic_meta = result["scene_fusion_meta"]["diagnostic_layers"]
    assert diagnostic_meta["source"] == "admitted_common_frame_fused_points_and_grid"
    assert diagnostic_meta["mapanything_inference_triggered"] is False
    assert diagnostic_meta["registration_triggered"] is False
    assert diagnostic_meta["grid_shape"] == [2, 2]
    assert diagnostic_meta["bounds"] == manifest["grid"]["bounds"]
