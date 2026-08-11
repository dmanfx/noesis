from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from geometry.depth_source import DepthStorageManager
from scripts.mapanything_cadence_fusion_offline_report import (
    SourceDescriptor,
    _load_descriptor,
    build_offline_report,
    contiguous_windows,
    load_source_report,
    metric_envelope,
    summarize_timestamps,
)


def _descriptor(index: int, *, gap_us: int = 3_000_000) -> SourceDescriptor:
    return SourceDescriptor(
        path=Path(f"/tmp/{index}.zarr"),
        camera_id="living-room",
        timestamp_us=1_000_000 + (index * gap_us),
        sequence=index + 1,
        write_id=f"write-{index}",
        manifest_file_sha256=f"{index:064x}",
        component_sha256s={
            "depth": f"{index + 1:064x}",
            "conf": f"{index + 2:064x}",
            "mask": f"{index + 3:064x}",
        },
    )


def _loaded(values: list[float]):
    depth = [
        np.full((2, 3), value, dtype=np.float32)
        for value in values
    ]
    confidence = [np.ones((2, 3), dtype=np.float32) for _value in values]
    mask = [np.ones((2, 3), dtype=np.uint8) for _value in values]
    return list(zip(depth, confidence, mask))


def _write_committed_snapshot(
    path: Path,
    *,
    corrupt_component_digest: bool = False,
) -> zarr.Group:
    depth = np.ones((2, 3), dtype=np.float32)
    confidence = np.ones((2, 3), dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.uint8)
    group = zarr.open_group(str(path), mode="w")
    group.create_array("depth_z", data=depth)
    group.create_array("conf", data=confidence)
    group.create_array("mask", data=mask)
    group.attrs.update(
        camera_id="living-room",
        timestamp_us=1_000_000,
        sequence=1,
        write_id="write-1",
    )
    components = DepthStorageManager._raw_component_records(
        depth=depth,
        conf=confidence,
        mask=mask,
        rgb=None,
    )
    if corrupt_component_digest:
        components["depth"]["sha256"] = "0" * 64
    manifest = {
        "version": 2,
        "state": "committed",
        "write_id": "write-1",
        "sequence": 1,
        "camera_id": "living-room",
        "timestamp_us": 1_000_000,
        "committed_at_ns": 1,
        "components": components,
        "files": DepthStorageManager._snapshot_file_records(path),
    }
    (path / DepthStorageManager._COMMIT_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return group


def test_interval89_timing_does_not_claim_interval59_period() -> None:
    timestamps = [1_000_000 + (index * 3_000_000) for index in range(6)]

    timing = summarize_timestamps(
        timestamps,
        source_fps=30.0,
        interval_frames=89,
    )

    assert timing["matches_declared_interval"] is True
    assert timing["gap_median_s"] == 3.0
    assert timing["expected_period_s"] == 3.0


def test_contiguous_four_and_five_frame_windows_bound_twelve_second_phase() -> None:
    assert contiguous_windows(6, 4) == [
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
    ]
    assert contiguous_windows(6, 5) == [
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
    ]


def test_metric_envelope_preserves_min_median_and_max() -> None:
    rows = [
        {
            "support": {
                "retained_full_frame_fraction": value,
                "retained_over_eligible_fraction": value,
            },
            "runtime_candidate": {
                "temporal_residual_p50_m": value,
                "temporal_residual_p95_m": value,
                "edge": {"gradient_p95_m": value},
            },
        }
        for value in (0.2, 0.4, 0.9)
    ]

    envelope = metric_envelope(rows)

    assert envelope["sample_count"] == 3
    assert envelope["retained_full_frame_fraction"] == {
        "min": 0.2,
        "median": 0.4,
        "max": 0.9,
    }


def test_offline_report_fails_closed_on_missing_primary_cadence() -> None:
    descriptors = [_descriptor(index) for index in range(6)]
    report = build_offline_report(
        {"living-room": (descriptors, _loaded([2.0] * 6))},
        source_fps=30.0,
        observed_interval_frames=89,
    )

    conclusion = report["cadence_conclusion"]
    assert conclusion["cadence_ab_conclusive"] is False
    assert conclusion["winner"] is None
    assert conclusion["primary_period_present_in_sources"] is False
    assert conclusion["exact_source_frame_evidence_complete"] is False
    assert report["agreement_conclusion"]["winner"] is None
    camera = report["cameras"]["living-room"]
    assert set(camera["interval89_window_sensitivity_min_observations_3"]) == {
        "4",
        "5",
    }
    assert (
        camera["full_cohort_min_observations_2"]["0.18"]["cohort"][
            "required_support"
        ]
        == 4
    )


def test_fixed_report_rejects_mislabeled_interval_or_cohort() -> None:
    descriptors = [_descriptor(index) for index in range(6)]
    loaded = _loaded([2.0] * 6)

    with pytest.raises(ValueError, match="observed_interval_frames=89"):
        build_offline_report(
            {"living-room": (descriptors, loaded)},
            source_fps=30.0,
            observed_interval_frames=59,
        )
    with pytest.raises(ValueError, match="exactly 6 source frames"):
        build_offline_report(
            {"living-room": (descriptors[:5], loaded[:5])},
            source_fps=30.0,
            observed_interval_frames=89,
        )


def test_fixed_report_rejects_timestamps_outside_declared_cadence() -> None:
    descriptors = [
        _descriptor(index, gap_us=2_000_000)
        for index in range(6)
    ]

    with pytest.raises(ValueError, match="does not match declared interval=89"):
        build_offline_report(
            {"living-room": (descriptors, _loaded([2.0] * 6))},
            source_fps=30.0,
            observed_interval_frames=89,
        )


def test_source_descriptor_validates_files_and_component_content(
    tmp_path: Path,
) -> None:
    wrong_digest_path = tmp_path / "wrong-digest" / "1000000.zarr"
    wrong_digest_path.parent.mkdir()
    _write_committed_snapshot(
        wrong_digest_path,
        corrupt_component_digest=True,
    )
    with pytest.raises(ValueError, match="depth content digest mismatch"):
        _load_descriptor(wrong_digest_path, "living-room")

    mutated_path = tmp_path / "mutated" / "1000000.zarr"
    mutated_path.parent.mkdir()
    group = _write_committed_snapshot(mutated_path)
    group["depth_z"][:] = np.full((2, 3), 99.0, dtype=np.float32)
    with pytest.raises(ValueError, match="commit validation failed"):
        _load_descriptor(mutated_path, "living-room")


def test_source_report_binds_the_arrays_loaded_for_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "1000000.zarr"
    _write_committed_snapshot(snapshot_path)
    source_report = tmp_path / "source-report.json"
    source_report.write_text(
        json.dumps(
            {
                "contract": "noesis.mapanything.fusion_quality_report",
                "contract_version": 1,
                "sources": [str(snapshot_path)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.mapanything_cadence_fusion_offline_report.load_depth_snapshot",
        lambda _path: (
            np.full((2, 3), 99.0, dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match="depth changed after validation"):
        load_source_report("living-room", source_report)
