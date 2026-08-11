from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_rfdetr_1_8_3_pytorch_parity.py"
)
SPEC = importlib.util.spec_from_file_location("rfdetr_pytorch_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


def _matrix() -> dict[str, object]:
    def model(model_id: str, family: str, enabled: bool = True) -> dict[str, object]:
        names = parity.FAMILY_OUTPUTS[family]
        shapes = {
            "dets": [3, 4, 4],
            "labels": [3, 4, 2],
            "masks": [3, 4, 8, 8],
            "keypoints": [3, 4, 34, 8],
        }
        return {
            "id": model_id,
            "family": family,
            "package": "rfdetr",
            "enabled": enabled,
            "outputs": {name: shapes[name] for name in names},
        }

    return {
        "models": [
            model("detect_medium", "detection"),
            model("seg_small", "segmentation"),
            model("seg_medium", "segmentation"),
            model("keypoint_preview", "keypoint"),
            model("disabled", "detection", False),
        ]
    }


def test_default_and_explicit_model_selection_is_ordered_and_fail_closed() -> None:
    matrix = _matrix()
    selected = parity._select_models(matrix, None)
    assert [row["id"] for row in selected] == list(parity.DEFAULT_MODELS)

    selected = parity._select_models(
        matrix, ["keypoint_preview,detect_medium"]
    )
    assert [row["id"] for row in selected] == [
        "keypoint_preview",
        "detect_medium",
    ]

    with pytest.raises(parity.ParityError, match="duplicate"):
        parity._select_models(matrix, ["seg_small", "seg_small"])
    with pytest.raises(parity.ParityError, match="unknown"):
        parity._select_models(matrix, ["not-a-model"])
    with pytest.raises(parity.ParityError, match="disabled"):
        parity._select_models(matrix, ["disabled"])


def test_export_tuple_maps_exact_pytorch_keys_to_onnx_names() -> None:
    boxes = object()
    logits = object()
    masks = object()
    mapped = parity._map_export_outputs(
        (boxes, logits, masks), ("dets", "labels", "masks")
    )
    assert mapped == {
        "dets": ("pred_boxes", boxes),
        "labels": ("pred_logits", logits),
        "masks": ("pred_masks", masks),
    }

    keypoints = object()
    mapped = parity._map_export_outputs(
        (boxes, logits, keypoints), ("dets", "labels", "keypoints")
    )
    assert mapped["keypoints"] == ("pred_keypoints", keypoints)

    with pytest.raises(parity.ParityError, match="exact output tuple"):
        parity._map_export_outputs(
            {"pred_boxes": boxes, "pred_logits": logits},
            ("dets", "labels"),
        )
    with pytest.raises(parity.ParityError, match="output-name contract"):
        parity._map_export_outputs((boxes,), ("unknown",))


def test_tensor_comparison_uses_combined_absolute_relative_gate() -> None:
    reference = np.asarray([0.0, 1.0, 100.0], dtype=np.float32)
    candidate = np.asarray([5.0e-5, 1.0005, 100.05], dtype=np.float32)
    result = parity._compare_tensors(
        reference, candidate, atol=1.0e-4, rtol=1.0e-3
    )
    assert result["passed"] is True
    assert result["violation_count"] == 0
    assert result["element_count"] == 3

    candidate[0] = 2.0e-4
    failed = parity._compare_tensors(
        reference, candidate, atol=1.0e-4, rtol=1.0e-3
    )
    assert failed["passed"] is False
    assert failed["violation_count"] == 1
    assert failed["max_tolerance_excess"] > 0.0

    with pytest.raises(parity.ParityError, match="contract drifted"):
        parity._compare_tensors(
            reference, candidate.reshape(3, 1), atol=1.0e-4, rtol=1.0e-3
        )
    bad = reference.copy()
    bad[0] = np.nan
    with pytest.raises(parity.ParityError, match="contract drifted"):
        parity._compare_tensors(
            bad, candidate, atol=1.0e-4, rtol=1.0e-3
        )


def test_tensor_record_binds_shape_size_hash_and_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tensor_dir = run_dir / "inputs" / "model"
    tensor_dir.mkdir(parents=True)
    tensor = np.arange(12, dtype="<f4").reshape(3, 4)
    path = tensor_dir / "case.bin"
    path.write_bytes(tensor.tobytes())
    record = {
        "path": "inputs/model/case.bin",
        "shape": [3, 4],
        "dtype": "float32",
        "finite": True,
        "size_bytes": 48,
        "sha256": parity._sha256(path),
    }
    assert (
        parity._validate_tensor_record(
            run_dir, record, [3, 4], "fixture input"
        )
        == path
    )

    changed = dict(record, shape=[2, 6])
    with pytest.raises(parity.ParityError, match="record drifted"):
        parity._validate_tensor_record(
            run_dir, changed, [3, 4], "fixture input"
        )
    path.write_bytes(b"\0" * 48)
    with pytest.raises(parity.ParityError, match="bytes changed"):
        parity._validate_tensor_record(
            run_dir, record, [3, 4], "fixture input"
        )


def test_private_report_is_atomic_owner_only_and_never_overwritten(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "private-reports"
    payload = {"schema": "fixture.v1", "passed": True}
    path = parity._publish_private_report(directory, "report-1", payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(directory.glob("*.tmp"))

    with pytest.raises(parity.ParityError, match="already exists"):
        parity._publish_private_report(directory, "report-1", {"changed": True})
    assert json.loads(path.read_text(encoding="utf-8")) == payload
