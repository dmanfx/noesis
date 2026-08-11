from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "DS9/scripts/evaluate_mapanything_fixed_corpus.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "ds9_mapanything_fixed_corpus_evaluator",
        TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_parse_output_tensors_requires_exact_finite_contract() -> None:
    payload = [
        {
            "name": name,
            "dimensions": "1x1x2x3",
            "values": [float(index + 1) for index in range(6)],
        }
        for name in ("conf", "depth", "mask")
    ]
    parsed = tool._parse_output_tensors(payload)
    assert parsed["depth"].shape == (1, 2, 3)
    assert parsed["depth"][0, 1, 2] == 6.0

    payload[0]["values"][2] = float("nan")
    with pytest.raises(tool.EvaluationError, match="non-finite"):
        tool._parse_output_tensors(payload)


def test_mapped_polygon_mask_honors_resize_padding_and_erosion() -> None:
    mask = tool._mapped_polygon_mask(
        [[2, 2], [7, 2], [7, 7], [2, 7]],
        source_size=(10, 10),
        output_shape=(8, 8),
        resized_size=(6, 6),
        pad_left=1,
        pad_top=1,
        erode_px_source=0,
    )
    assert mask.shape == (8, 8)
    assert mask[2, 2]
    assert mask[5, 5]
    assert not mask[0, 0]

    un_eroded = tool._mapped_polygon_mask(
        [[1, 1], [8, 1], [8, 8], [1, 8]],
        source_size=(10, 10),
        output_shape=(10, 10),
        resized_size=(10, 10),
        pad_left=0,
        pad_top=0,
        erode_px_source=0,
    )
    eroded = tool._mapped_polygon_mask(
        [[1, 1], [8, 1], [8, 8], [1, 8]],
        source_size=(10, 10),
        output_shape=(10, 10),
        resized_size=(10, 10),
        pad_left=0,
        pad_top=0,
        erode_px_source=1,
    )
    assert np.count_nonzero(eroded) < np.count_nonzero(un_eroded)
    assert eroded[4, 4]


def test_world_y_grid_uses_authored_camera_to_world_transform() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    u_source, v_source = np.meshgrid(
        np.arange(2, dtype=np.float64),
        np.arange(2, dtype=np.float64),
        indexing="xy",
    )
    intrinsics = np.eye(3, dtype=np.float64)
    camera_to_world = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_y = tool._world_y_grid(
        depth,
        u_source=u_source,
        v_source=v_source,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )
    np.testing.assert_allclose(world_y, 0.0, atol=1e-12)


def test_evaluate_locks_rgb_calibration_and_candidate_identities(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "camera.png"
    rgb_path.write_bytes(b"fixed-rgb")
    camera_path = tmp_path / "camera_calibration.json"
    floor_path = tmp_path / "floor_alignment.json"
    dewarper_path = tmp_path / "dewarper.txt"

    camera_to_world = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    _write_json(
        camera_path,
        {
            "cameras": {
                "camera": {
                    "E": world_to_camera.reshape(-1, order="F").tolist(),
                }
            }
        },
    )
    _write_json(floor_path, {"floor_y": 0.0})
    dewarper_path.write_text(
        "\n".join(
            (
                "[property]",
                "output-width=4",
                "output-height=4",
                "[surface0]",
                "dst-focal-length=1;1",
                "dst-principal-point=0;0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    annotations_path = tmp_path / "annotations.json"
    annotations = {
        "contract": tool.ANNOTATION_CONTRACT,
        "calibration": {
            "camera_calibration": {
                "path": str(camera_path),
                "sha256": tool._sha256_file(camera_path),
            },
            "floor_alignment": {
                "path": str(floor_path),
                "sha256": tool._sha256_file(floor_path),
            },
            "dewarper_configs": {
                "camera": {
                    "path": str(dewarper_path),
                    "sha256": tool._sha256_file(dewarper_path),
                    "output_size": [4, 4],
                }
            },
        },
        "frames": [
            {
                "camera_id": "camera",
                "rgb": {
                    "path": rgb_path.name,
                    "sha256": tool._sha256_file(rgb_path),
                    "size": [4, 4],
                },
                "floor_regions": [
                    {
                        "id": "floor",
                        "polygon_px": [[0, 0], [3, 0], [3, 3], [0, 3]],
                        "erode_px": 0,
                        "annotation_confidence": "high",
                    }
                ],
                "obstacle_regions": [],
                "structural_edges": [],
            }
        ],
        "blocked_measurements": [
            {
                "measurement": "absolute_object_distance",
                "reason": "no independent physical measurement",
            }
        ],
    }
    _write_json(annotations_path, annotations)

    receipt_path = tmp_path / "fixture.receipt.json"
    receipt = {
        "contract": tool.FIXTURE_CONTRACT,
        "identical_batch_members": False,
        "profile": "synthetic",
        "sources": [
            {
                "sha256": tool._sha256_file(rgb_path),
                "source_width": 4,
                "source_height": 4,
                "resized_width": 4,
                "resized_height": 4,
                "pad_left": 0,
                "pad_top": 0,
            }
        ],
        "tensor": {"shape": [1, 3, 4, 4]},
    }
    _write_json(receipt_path, receipt)

    output_path = tmp_path / "output.json"
    output = [
        {
            "name": name,
            "dimensions": "1x1x4x4",
            "values": [1.0] * 16,
        }
        for name in ("depth", "conf", "mask")
    ]
    _write_json(output_path, output)
    final_output_identity = tmp_path / "committed" / output_path.name

    report = tool.evaluate(
        annotations_path=annotations_path,
        fixture_receipt_path=receipt_path,
        model_output_path=output_path,
        model_output_identity_path=final_output_identity,
        candidate_id="synthetic-candidate",
        repo_root=tmp_path,
    )
    assert report["candidate_id"] == "synthetic-candidate"
    assert report["macro"]["floor_plane_abs_residual_m"][
        "camera_median_p95"
    ] == pytest.approx(0.0)
    assert report["claims"]["absolute_object_distance_accuracy"] is False
    assert report["macro"]["annotation_coverage"] == {
        "floor_camera_count": 1,
        "obstacle_camera_count": 0,
        "structural_edge_count": 0,
        "room_boundary_snippet_count": 0,
    }
    assert "cannot support promotion alone" in report["metric_limitations"][
        "structural_edge_distance"
    ]
    assert (
        report["evidence_classification"][
            "floorplan_or_other_model_output_used_as_truth"
        ]
        is False
    )
    assert report["identity"]["model_output"] == {
        "path": str(final_output_identity.resolve()),
        "sha256": tool._sha256_file(output_path),
    }

    receipt["sources"][0]["sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(tool.EvaluationError, match="no independent annotation"):
        tool.evaluate(
            annotations_path=annotations_path,
            fixture_receipt_path=receipt_path,
            model_output_path=output_path,
            candidate_id="synthetic-candidate",
            repo_root=tmp_path,
        )
