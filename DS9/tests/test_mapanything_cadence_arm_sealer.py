from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from DS9.scripts import evaluate_mapanything_cadence_ab as cadence
from DS9.scripts import seal_mapanything_cadence_arm as sealer
from DS9.tests.test_mapanything_cadence_ab_analyzer import (
    _capture_store,
    _write_snapshot,
)


def _controls() -> dict[str, str]:
    return {
        key: f"{index + 1:064x}"
        for index, key in enumerate(
            sorted(cadence.REQUIRED_CONTROL_FINGERPRINTS)
        )
    }


def test_control_mapping_requires_exact_five_named_fingerprints() -> None:
    controls = _controls()

    assert sealer._control_mapping(list(controls.items())) == controls

    rows = list(controls.items())[:-1]
    with pytest.raises(ValueError, match="controlled input set mismatch"):
        sealer._control_mapping(rows)
    with pytest.raises(ValueError, match="duplicate controlled input"):
        sealer._control_mapping(
            [*list(controls.items()), next(iter(controls.items()))]
        )


def test_copy_snapshot_copies_only_named_committed_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    snapshot = source / "living-room" / "1000000.zarr"
    other = source / "living-room" / "2000000.zarr"
    for path, timestamp, write_id in (
        (snapshot, 1_000_000, "selected"),
        (other, 2_000_000, "unselected"),
    ):
        _write_snapshot(
            path,
            camera_id="living-room",
            timestamp_us=timestamp,
            sequence=timestamp // 1_000_000,
            write_id=write_id,
            depth=np.ones((2, 3), dtype=np.float32),
            confidence=np.ones((2, 3), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.uint8),
        )

    sealer._copy_snapshot(
        source_store=source,
        destination_store=destination,
        storage_ref="living-room/1000000.zarr",
        camera_id="living-room",
    )

    assert (destination / "living-room/1000000.zarr").is_dir()
    assert not (destination / "living-room/2000000.zarr").exists()


def test_seal_arm_writes_self_hashed_receipt_and_omits_unreferenced_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source-session"
    source_root.mkdir()
    event, result = _capture_store(
        source_root,
        interval_frames=59,
        write_prefix="arm59",
    )
    unreferenced = source_root / "depth/living-room/99999999.zarr"
    _write_snapshot(
        unreferenced,
        camera_id="living-room",
        timestamp_us=99_999_999,
        sequence=99,
        write_id="unreferenced",
        depth=np.ones((2, 3), dtype=np.float32),
        confidence=np.ones((2, 3), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.uint8),
    )
    report = source_root / cadence.live_gate.CANONICAL_REPORT_FILENAME
    source = source_root / cadence.live_gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    report.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    config = source_root / "mapanything.ini"
    config.write_text(
        "[property]\ninterval=59\nmodel-engine-file=same.plan\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sealer,
        "_load_source_session",
        lambda **_kwargs: ({}, [(event, result)]),
    )
    monkeypatch.setattr(
        cadence,
        "load_arm",
        lambda _root, *, expected_interval: object()
        if expected_interval == 59
        else None,
    )
    output = tmp_path / "sealed-arm"

    sealer.seal_arm(
        output_root=output,
        interval_frames=59,
        depth_root=source_root / "depth",
        report_path=report,
        source_path=source,
        infer_config_path=config,
        controlled_input_sha256s=_controls(),
    )

    receipt = json.loads(
        (output / cadence.ARM_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    receipt_digest = receipt.pop("receipt_sha256")
    assert cadence.canonical_json_sha256(receipt) == receipt_digest
    assert receipt["interval_frames"] == 59
    assert receipt["controlled_input_sha256s"] == dict(
        sorted(_controls().items())
    )
    assert not (output / "depth/living-room/99999999.zarr").exists()
    copied = list((output / "depth/living-room").glob("*.zarr"))
    assert len(copied) == 5
