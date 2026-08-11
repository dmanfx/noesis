from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_mapanything_alignment_ab.py"
)
SPEC = importlib.util.spec_from_file_location("mapanything_alignment_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_crop_matches_current_1920x1080_letterbox_inverse() -> None:
    assert MODULE.runtime_crop((294, 518), (1920, 1080)) == (518, 291, 0, 1)
    assert MODULE.runtime_crop((672, 378), (1080, 1920)) == (378, 672, 0, 0)
    with pytest.raises(MODULE.AlignmentEvaluationError, match="positive"):
        MODULE.runtime_crop((294, 518), (0, 1080))


def test_rgb_guided_alignment_preserves_unknown_and_local_metric_range() -> None:
    depth = np.asarray(
        [
            [1.0, 1.0, 4.0, 4.0],
            [1.0, 1.0, 4.0, 4.0],
            [1.0, 1.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    conf = np.ones_like(depth)
    mask = np.ones_like(depth, dtype=bool)
    mask[1, 0] = False
    low_rgb = np.zeros((3, 4, 3), dtype=np.float32)
    low_rgb[:, 2:] = 1.0
    high_rgb = cv2.resize(
        low_rgb,
        (8, 6),
        interpolation=cv2.INTER_NEAREST,
    )

    baseline_depth, _, baseline_valid = MODULE.align_bilinear(
        depth,
        conf,
        mask,
        crop=(4, 3, 0, 0),
        target_size=(8, 6),
    )
    guided_depth, _, guided_valid, diagnostics = MODULE.align_rgb_guided(
        depth,
        conf,
        mask,
        low_rgb,
        high_rgb,
        crop=(4, 3, 0, 0),
        target_size=(8, 6),
    )

    np.testing.assert_array_equal(guided_valid, baseline_valid)
    assert diagnostics["local_convex_hull_violation_count"] == 0
    assert np.nanmin(guided_depth) >= 1.0
    assert np.nanmax(guided_depth) <= 4.0
    assert np.all(np.isnan(guided_depth[~guided_valid]))
    # Immediately left of the RGB boundary, the guide suppresses white-side
    # depth mixing more strongly than plain bilinear interpolation.
    assert guided_depth[2, 3] < baseline_depth[2, 3]


def test_normal_edge_metric_searches_only_along_authored_normal() -> None:
    depth = np.ones((20, 20), dtype=np.float32)
    depth[:, 10:] = 3.0
    valid = np.ones_like(depth, dtype=bool)
    grad_x, grad_y, _, interior = MODULE.depth_gradients(depth, valid)
    report, distances, strengths = MODULE.normal_edge_metrics(
        {
            "id": "vertical-depth-step",
            "kind": "synthetic",
            "polyline_px": [[9.5, 2.0], [9.5, 17.0]],
            "search_radius_px": 4,
        },
        grad_x=grad_x,
        grad_y=grad_y,
        valid=interior,
    )
    assert report["search_axis"] == "authored_polyline_normal_only"
    assert report["sample_count"] >= 4
    assert np.max(distances) <= 1.0
    assert np.min(strengths) > 0.0


def test_identity_and_tensor_contract_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"locked")
    record = {
        "path": str(artifact),
        "sha256": MODULE.sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    assert MODULE.require_identity(record, label="artifact") == artifact
    artifact.write_bytes(b"drifted")
    with pytest.raises(MODULE.AlignmentEvaluationError, match="size mismatch"):
        MODULE.require_identity(record, label="artifact")

    payload = [
        {
            "name": name,
            "dimensions": "1x1x2x2",
            "values": [1.0, 1.0, 1.0, 1.0],
        }
        for name in MODULE.OUTPUT_NAMES
    ]
    parsed = MODULE.parse_output_tensors(payload)
    assert parsed["depth"].shape == (1, 2, 2)
    payload[0]["values"][0] = float("nan")
    with pytest.raises(MODULE.AlignmentEvaluationError, match="non-finite"):
        MODULE.parse_output_tensors(payload)


def test_report_write_is_non_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    MODULE.write_report(output, {"contract": "test", "finite": 1.0})
    assert json.loads(output.read_text(encoding="utf-8"))["contract"] == "test"
    with pytest.raises(MODULE.AlignmentEvaluationError, match="already exists"):
        MODULE.write_report(output, {"contract": "replacement"})
