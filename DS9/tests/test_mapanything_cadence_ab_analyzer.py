from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from DS9.scripts import evaluate_mapanything_cadence_ab as cadence
from geometry.depth_source import DepthStorageManager
from noesis_core.capture_event_fusion import canonical_json_sha256
from scripts.mapanything_fusion_quality_report import evaluate_fusion_cohort


def _write_snapshot(
    path: Path,
    *,
    camera_id: str,
    timestamp_us: int,
    sequence: int,
    write_id: str,
    depth: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    attrs: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    group.create_array("depth_z", data=np.asarray(depth, dtype=np.float32))
    group.create_array("conf", data=np.asarray(confidence, dtype=np.float32))
    group.create_array("mask", data=np.asarray(mask, dtype=np.uint8))
    group.attrs.update(
        camera_id=camera_id,
        timestamp_us=timestamp_us,
        sequence=sequence,
        write_id=write_id,
        **(attrs or {}),
    )
    components = DepthStorageManager._raw_component_records(
        depth=np.asarray(depth, dtype=np.float32),
        conf=np.asarray(confidence, dtype=np.float32),
        mask=np.asarray(mask, dtype=np.uint8),
        rgb=None,
    )
    manifest = {
        "version": 2,
        "state": "committed",
        "write_id": write_id,
        "sequence": sequence,
        "camera_id": camera_id,
        "timestamp_us": timestamp_us,
        "committed_at_ns": 1,
        "components": components,
        "files": DepthStorageManager._snapshot_file_records(path),
    }
    (path / DepthStorageManager._COMMIT_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _raw_loaded(
    *,
    write_id: str,
    timestamp_us: int,
    sequence: int,
    frame_number: int,
    pts_ns: int,
    depth_value: float = 2.0,
    timestamp_basis: str = "capture_wall_clock",
    source_id: int | None = 0,
) -> cadence.LoadedSnapshot:
    depth = np.full((3, 4), depth_value, dtype=np.float32)
    confidence = np.ones_like(depth)
    mask = np.ones_like(depth, dtype=np.uint8)
    identity = cadence.SnapshotIdentity(
        path=Path(f"/tmp/{write_id}.zarr"),
        camera_id="living-room",
        timestamp_us=timestamp_us,
        sequence=sequence,
        write_id=write_id,
        manifest_sha256=f"{sequence:064x}",
        content_sha256=f"{sequence + 10:064x}",
        component_sha256s={
            "depth": f"{sequence + 20:064x}",
            "conf": f"{sequence + 30:064x}",
            "mask": f"{sequence + 40:064x}",
        },
        snapshot_role="",
        fusion_level="",
        attrs={
            "source_frame_contract": cadence.SOURCE_FRAME_CONTRACT,
            **({"source_id": source_id} if source_id is not None else {}),
            "source_frame_number": frame_number,
            "source_media_pts_ns": pts_ns,
            "storage_timestamp_basis": timestamp_basis,
        },
    )
    return cadence.LoadedSnapshot(
        identity=identity,
        depth=depth,
        confidence=confidence,
        mask=mask,
    )


def _capture_store(
    root: Path,
    *,
    interval_frames: int,
    write_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    camera_id = "living-room"
    store = root / "depth"
    period = interval_frames + 1
    raw_paths: list[Path] = []
    for index in range(4):
        timestamp_us = 10_000_000 + index * int(period / 30.0 * 1_000_000)
        raw_path = store / camera_id / f"{timestamp_us}.zarr"
        _write_snapshot(
            raw_path,
            camera_id=camera_id,
            timestamp_us=timestamp_us,
            sequence=index + 1,
            write_id=f"{write_prefix}-raw-{index}",
            depth=np.full((5, 6), 2.0 + index * 0.01, dtype=np.float32),
            confidence=np.ones((5, 6), dtype=np.float32),
            mask=np.ones((5, 6), dtype=np.uint8),
            attrs={
                "source_frame_contract": cadence.SOURCE_FRAME_CONTRACT,
                "source_id": 0,
                "source_frame_number": index * period,
                "source_media_pts_ns": index
                * int(period / 30.0 * 1_000_000_000),
                "storage_timestamp_basis": "capture_wall_clock",
            },
        )
        raw_paths.append(raw_path)
    loaded = [
        cadence._load_snapshot(path, expected_camera=camera_id)
        for path in raw_paths
    ]
    fusion = cadence._compact_evaluation(
        evaluate_fusion_cohort(
            np.stack([row.depth for row in loaded]),
            np.stack([row.confidence for row in loaded]),
            np.stack([row.mask for row in loaded]),
            min_observations=3,
            depth_agreement_m=0.18,
            normalize_frame_scale=True,
        )
    )
    source_rows = [
        {
            "camera_id": camera_id,
            "timestamp_us": row.identity.timestamp_us,
            "write_id": row.identity.write_id,
            "sequence": row.identity.sequence,
            "path": str(row.identity.path),
            "storage_ref": row.identity.path.relative_to(store).as_posix(),
            "manifest_sha256": row.identity.manifest_sha256,
            "content_sha256": row.identity.content_sha256,
            "snapshot_role": "",
            "fusion_level": "",
        }
        for row in loaded
    ]
    support = fusion["support"]
    stored_support = {
        "required_observations": fusion["cohort"]["required_support"],
        "strict_majority_observations": fusion["cohort"][
            "strict_majority_support"
        ],
        "eligible_full_frame_fraction": support["eligible_fraction"],
        "consensus_full_frame_fraction": support[
            "retained_full_frame_fraction"
        ],
        "consensus_retained_eligible_fraction": support[
            "retained_over_eligible_fraction"
        ],
        "temporal_absolute_residual_median_m": fusion[
            "temporal_residual_p50_m"
        ],
        "temporal_absolute_residual_p95_m": fusion[
            "temporal_residual_p95_m"
        ],
        "component_evidence": support["component_evidence"],
    }
    event_id = "capture-event-sha256:" + "e" * 64
    fused_timestamp_us = loaded[-1].identity.timestamp_us + 1
    fused_path = store / camera_id / f"{fused_timestamp_us}.zarr"
    _write_snapshot(
        fused_path,
        camera_id=camera_id,
        timestamp_us=fused_timestamp_us,
        sequence=10,
        write_id=f"{write_prefix}-fused",
        depth=np.mean(np.stack([row.depth for row in loaded]), axis=0),
        confidence=np.ones((5, 6), dtype=np.float32),
        mask=np.ones((5, 6), dtype=np.uint8),
        attrs={
            "snapshot_role": "capture_event_fused",
            "fusion_level": "intra_capture",
            "event_id": event_id,
            "source_snapshots": json.dumps(source_rows, separators=(",", ":")),
            "fusion_meta": json.dumps(
                {
                    "source_snapshot_count": len(loaded),
                    "support_evidence": stored_support,
                },
                separators=(",", ":"),
            ),
        },
    )
    fused = cadence._load_snapshot(fused_path, expected_camera=camera_id)
    event = {
        "camera_id": camera_id,
        "baseline_raw_timestamp_us": loaded[0].identity.timestamp_us - 1,
        "parameters": {
            "burst_seconds": 20.0,
            "raw_limit": 24,
            "min_observations": 3,
            "depth_agreement_m": 0.18,
            "max_cohort_span_us": 20_000_000,
        },
        "raw_snapshot_count": len(loaded),
        "rgb": {
            "status": "available",
            "source_id": 0,
            "frame_id": int(loaded[-1].identity.attrs["source_frame_number"]),
            "source_media_pts_ns": int(
                loaded[-1].identity.attrs["source_media_pts_ns"]
            ),
        },
        "fused_snapshot": {
            "camera_id": camera_id,
            "storage_key": camera_id,
            "timestamp_us": fused.identity.timestamp_us,
            "snapshot_id": fused.identity.write_id,
            "artifact_ref": f"depth-zarr:{fused_path.relative_to(store)}",
            "content_sha256": fused.identity.content_sha256,
            "sequence": fused.identity.sequence,
            "manifest_sha256": fused.identity.manifest_sha256,
            "event_id": event_id,
            "source_snapshot_ids": [
                row.identity.write_id for row in loaded
            ],
            "snapshot_role": "capture_event_fused",
            "fusion_level": "intra_capture",
        },
    }
    result = {
        "camera_id": camera_id,
        "calibration_fingerprint": "c" * 64,
    }
    return event, result


def _camera_payload(
    cohort: tuple[cadence.LoadedSnapshot, ...],
    *,
    prefix: str,
    coverage: float,
    residual: float,
) -> dict[str, Any]:
    return {
        "camera_id": "living-room",
        "calibration_fingerprint": "c" * 64,
        "raw_snapshot_identities": [
            {"write_id": f"{prefix}-{index}"}
            for index, _row in enumerate(cohort)
        ],
        "timing": {
            "snapshot_count": len(cohort),
            "source_id": 0,
            "on_declared_interval_lattice": True,
            "media_pts_matches_frame_numbers": True,
        },
        "fusion_at_0.18_m": {
            "cohort": {
                "shape": [3, 4],
                "required_support": 3,
                "strict_majority_support": 3,
            },
            "support": {
                "eligible_fraction": coverage,
                "retained_full_frame_fraction": coverage,
                "retained_over_eligible_fraction": 1.0,
                "retained_p50": 4.0,
                "component_evidence": {
                    "component_count": 1,
                    "hole_count": 0,
                    "hole_pixels": 0,
                    "fragment_fraction": 0.0,
                },
            },
            "temporal_residual_p50_m": residual,
            "temporal_residual_p95_m": residual * 2,
        },
    }


def _arm(
    *,
    interval: int,
    prefix: str,
    coverage: float,
    residual: float,
) -> cadence.ArmAnalysis:
    period = interval + 1
    cohort = tuple(
        _raw_loaded(
            write_id=f"{prefix}-{index}",
            timestamp_us=1_000_000 + index * period * 100_000,
            sequence=index + 1,
            frame_number=index * period,
            pts_ns=index * int(period / 30.0 * 1_000_000_000),
            depth_value=2.0 + index * 0.01,
        )
        for index in range(4)
    )
    return cadence.ArmAnalysis(
        arm_id=f"interval-{interval}",
        root=Path(f"/tmp/{prefix}"),
        interval_frames=interval,
        session_id=f"session-{prefix}",
        runtime_lane="baseline",
        runtime_instance_id="TauntonMainframe",
        runtime_run_id=f"run-{prefix}",
        camera_order=("living-room",),
        controlled_input_sha256s={
            key: f"{index + 1:064x}"
            for index, key in enumerate(
                sorted(cadence.REQUIRED_CONTROL_FINGERPRINTS)
            )
        },
        config_semantics_without_interval={
            "property": {"model-engine-file": "same.plan"}
        },
        cameras={
            "living-room": _camera_payload(
                cohort,
                prefix=prefix,
                coverage=coverage,
                residual=residual,
            )
        },
        cohorts={"living-room": cohort},
    )


def test_fusion_report_includes_runtime_equivalent_topology() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    mask[4, 4] = 0
    mask[0, 7] = 1
    depth = np.broadcast_to(
        np.where(mask, 2.0, np.nan),
        (3, 8, 8),
    ).astype(np.float32)
    confidence = np.broadcast_to(mask, (3, 8, 8)).astype(np.float32)
    masks = np.broadcast_to(mask, (3, 8, 8))

    report = evaluate_fusion_cohort(
        depth,
        confidence,
        masks,
        min_observations=3,
        depth_agreement_m=0.18,
    )

    topology = report["support"]["component_evidence"]
    assert topology["component_count"] == 2
    assert topology["hole_count"] == 1
    assert topology["hole_pixels"] == 1
    assert topology["fragment_pixels"] == 1


def test_source_timing_reports_missed_opportunities_without_faking_frames() -> None:
    cohort = (
        _raw_loaded(
            write_id="a",
            timestamp_us=1_000_000,
            sequence=1,
            frame_number=0,
            pts_ns=0,
        ),
        _raw_loaded(
            write_id="b",
            timestamp_us=4_000_000,
            sequence=2,
            frame_number=90,
            pts_ns=3_000_000_000,
        ),
        _raw_loaded(
            write_id="c",
            timestamp_us=10_000_000,
            sequence=3,
            frame_number=270,
            pts_ns=9_000_000_000,
        ),
    )

    timing = cadence._source_timing(
        cohort,
        interval_frames=89,
        source_fps=30.0,
    )

    assert timing["source_frame_gaps"] == [90, 180]
    assert timing["on_declared_interval_lattice"] is True
    assert timing["every_inference_opportunity_persisted"] is False
    assert timing["missing_opportunity_count"] == 1
    assert timing["media_pts_matches_frame_numbers"] is True


@pytest.mark.parametrize(
    "timestamp_basis",
    ["capture_wall_clock", "source_epoch_pts"],
)
def test_source_timing_accepts_declared_storage_timestamp_bases(
    timestamp_basis: str,
) -> None:
    cohort = tuple(
        _raw_loaded(
            write_id=f"basis-{index}",
            timestamp_us=1_000_000 + index * 3_000_000,
            sequence=index + 1,
            frame_number=index * 90,
            pts_ns=index * 3_000_000_000,
            timestamp_basis=timestamp_basis,
        )
        for index in range(3)
    )

    timing = cadence._source_timing(
        cohort,
        interval_frames=89,
        source_fps=30.0,
    )

    assert timing["storage_timestamp_basis"] == [timestamp_basis] * 3
    assert timing["on_declared_interval_lattice"] is True
    assert timing["media_pts_matches_frame_numbers"] is True


@pytest.mark.parametrize("timestamp_basis", ["wall_clock_fallback", "unknown"])
def test_source_timing_rejects_non_capture_storage_timestamp_basis(
    timestamp_basis: str,
) -> None:
    cohort = tuple(
        _raw_loaded(
            write_id=f"invalid-{index}",
            timestamp_us=1_000_000 + index * 3_000_000,
            sequence=index + 1,
            frame_number=index * 90,
            pts_ns=index * 3_000_000_000,
            timestamp_basis=timestamp_basis,
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="not hardened capture-time evidence"):
        cadence._source_timing(
            cohort,
            interval_frames=89,
            source_fps=30.0,
        )


def test_source_timing_rejects_mixed_capture_timestamp_bases() -> None:
    cohort = tuple(
        _raw_loaded(
            write_id=f"mixed-{index}",
            timestamp_us=1_000_000 + index * 3_000_000,
            sequence=index + 1,
            frame_number=index * 90,
            pts_ns=index * 3_000_000_000,
            timestamp_basis=(
                "capture_wall_clock" if index != 1 else "source_epoch_pts"
            ),
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="mixes storage timestamp bases"):
        cadence._source_timing(
            cohort,
            interval_frames=89,
            source_fps=30.0,
        )


def test_source_timing_reconciles_exact_source_identity() -> None:
    cohort = tuple(
        _raw_loaded(
            write_id=f"source-{index}",
            timestamp_us=1_000_000 + index * 3_000_000,
            sequence=index + 1,
            frame_number=index * 90,
            pts_ns=index * 3_000_000_000,
            timestamp_basis="capture_wall_clock",
            source_id=0,
        )
        for index in range(3)
    )

    timing = cadence._source_timing(
        cohort,
        interval_frames=89,
        source_fps=30.0,
        expected_source_id=0,
    )
    assert timing["source_id"] == 0
    with pytest.raises(ValueError, match="exact RGB camera binding"):
        cadence._source_timing(
            cohort,
            interval_frames=89,
            source_fps=30.0,
            expected_source_id=1,
        )


def test_nvinfer_config_requires_exact_interval_and_normalizes_only_it(
    tmp_path: Path,
) -> None:
    config = tmp_path / "mapanything.ini"
    config.write_text(
        "[property]\ninterval=59\nmodel-engine-file=same.plan\nbatch-size=3\n",
        encoding="utf-8",
    )

    semantics = cadence._load_nvinfer_semantics(
        config,
        expected_interval=59,
    )

    assert "interval" not in semantics["property"]
    assert semantics["property"]["model-engine-file"] == "same.plan"
    with pytest.raises(ValueError, match="interval 59 != 89"):
        cadence._load_nvinfer_semantics(config, expected_interval=89)


def test_snapshot_loader_rejects_committed_array_tamper(tmp_path: Path) -> None:
    snapshot = tmp_path / "depth" / "living-room" / "1000000.zarr"
    _write_snapshot(
        snapshot,
        camera_id="living-room",
        timestamp_us=1_000_000,
        sequence=1,
        write_id="raw-1",
        depth=np.ones((2, 3), dtype=np.float32),
        confidence=np.ones((2, 3), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.uint8),
    )
    cadence._load_snapshot(snapshot, expected_camera="living-room")
    group = zarr.open_group(str(snapshot), mode="r+")
    group["depth_z"][:] = np.full((2, 3), 99.0, dtype=np.float32)

    with pytest.raises(ValueError, match="commit validation failed"):
        cadence._load_snapshot(snapshot, expected_camera="living-room")


def test_capture_analysis_reconciles_exact_raw_and_stored_fusion_evidence(
    tmp_path: Path,
) -> None:
    event, result = _capture_store(
        tmp_path,
        interval_frames=59,
        write_prefix="arm59",
    )

    analysis, cohort = cadence._analyze_capture(
        event=event,
        result=result,
        store_root=tmp_path / "depth",
        interval_frames=59,
        source_fps=30.0,
    )

    assert analysis["timing"]["snapshot_count"] == 4
    assert analysis["timing"]["source_frame_gaps"] == [60, 60, 60]
    assert analysis["timing"]["on_declared_interval_lattice"] is True
    assert analysis["fusion_at_0.18_m"]["cohort"]["required_support"] == 3
    assert len(cohort) == 4


def test_capture_analysis_rejects_source_id_substitution(tmp_path: Path) -> None:
    event, result = _capture_store(
        tmp_path,
        interval_frames=89,
        write_prefix="arm89",
    )
    event["fused_snapshot"]["source_snapshot_ids"][0] = "substituted"

    with pytest.raises(ValueError, match="raw snapshot identity differs"):
        cadence._analyze_capture(
            event=event,
            result=result,
            store_root=tmp_path / "depth",
            interval_frames=89,
            source_fps=30.0,
        )


def test_capture_analysis_rejects_rgb_identity_outside_raw_cohort(
    tmp_path: Path,
) -> None:
    event, result = _capture_store(
        tmp_path,
        interval_frames=89,
        write_prefix="arm89",
    )
    event["rgb"]["frame_id"] = 999_999

    with pytest.raises(ValueError, match="absent from the raw depth cohort"):
        cadence._analyze_capture(
            event=event,
            result=result,
            store_root=tmp_path / "depth",
            interval_frames=89,
            source_fps=30.0,
        )


def test_build_report_is_camera_balanced_and_does_not_invent_winner() -> None:
    interval89 = _arm(
        interval=89,
        prefix="arm89",
        coverage=0.50,
        residual=0.10,
    )
    interval59 = _arm(
        interval=59,
        prefix="arm59",
        coverage=0.70,
        residual=0.08,
    )

    report = cadence.build_report(interval89, interval59)

    assert report["comparison_ready"] is True
    coverage = report["camera_balanced_cadence_comparison"][
        "retained_full_frame_fraction"
    ]["camera_balanced_mean"]
    assert coverage["delta_right_minus_left"] == pytest.approx(0.20)
    assert report["cadence_selection"]["winner"] is None

    interval59.controlled_input_sha256s["engine"] = "f" * 64
    with pytest.raises(ValueError, match="controlled input"):
        cadence.build_report(interval89, interval59)


def test_agreement_replay_uses_identical_selected_cadence_cohort() -> None:
    interval89 = _arm(
        interval=89,
        prefix="arm89",
        coverage=0.50,
        residual=0.10,
    )
    interval59 = _arm(
        interval=59,
        prefix="arm59",
        coverage=0.70,
        residual=0.08,
    )

    report = cadence.build_report(
        interval89,
        interval59,
        agreement_arm="interval-59",
    )

    agreement = report["agreement_ab"]
    camera = agreement["cameras"]["living-room"]
    assert agreement["same_raw_cohort_for_both_arms"] is True
    assert agreement["strict_majority_unchanged"] is True
    assert camera["same_cohort_for_all_agreements"] is True
    assert set(camera["evaluations"]) == {"0.12", "0.18"}
    assert camera["required_support"] == 3
    assert agreement["winner"] is None
    assert agreement["status"] == (
        "exact_explicit_cadence_cohort_replay_with_ready_cadence_comparison"
    )
    assert (
        agreement["selection_source"]
        == "explicit_cli_with_ready_cadence_comparison"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_lane", "candidate", "runtime lane"),
        ("runtime_instance_id", "OtherMainframe", "runtime instance"),
    ],
)
def test_build_report_rejects_runtime_scope_drift(
    field: str,
    value: str,
    message: str,
) -> None:
    interval89 = _arm(
        interval=89,
        prefix="arm89",
        coverage=0.50,
        residual=0.10,
    )
    interval59 = _arm(
        interval=59,
        prefix="arm59",
        coverage=0.70,
        residual=0.08,
    )
    setattr(interval59, field, value)

    with pytest.raises(ValueError, match=message):
        cadence.build_report(interval89, interval59)


def test_build_report_rejects_camera_source_binding_drift() -> None:
    interval89 = _arm(
        interval=89,
        prefix="arm89",
        coverage=0.50,
        residual=0.10,
    )
    interval59 = _arm(
        interval=59,
        prefix="arm59",
        coverage=0.70,
        residual=0.08,
    )
    interval59.cameras["living-room"]["timing"]["source_id"] = 1

    with pytest.raises(ValueError, match="source identity changed"):
        cadence.build_report(interval89, interval59)


def test_agreement_replay_labels_blocked_cadence_selection_honestly() -> None:
    interval89 = _arm(
        interval=89,
        prefix="arm89",
        coverage=0.50,
        residual=0.10,
    )
    interval59 = _arm(
        interval=59,
        prefix="arm59",
        coverage=0.70,
        residual=0.08,
    )
    interval89.cameras["living-room"]["timing"][
        "on_declared_interval_lattice"
    ] = False

    report = cadence.build_report(
        interval89,
        interval59,
        agreement_arm="interval-89",
    )

    assert report["comparison_ready"] is False
    assert report["cadence_selection"]["status"] == "blocked_by_timing_evidence"
    agreement = report["agreement_ab"]
    assert agreement["status"] == (
        "exact_explicit_cadence_cohort_replay_with_blocked_cadence_comparison"
    )
    assert (
        agreement["selection_source"]
        == "explicit_cli_with_blocked_cadence_comparison"
    )
    assert agreement["selected_cadence_arm"] == "interval-89"
    assert agreement["same_raw_cohort_for_both_arms"] is True
    assert agreement["winner"] is None


def test_load_arm_validates_self_hashed_receipt_before_live_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "arm"
    root.mkdir()
    config = root / "mapanything.ini"
    config.write_text(
        "[property]\ninterval=59\nmodel-engine-file=same.plan\n",
        encoding="utf-8",
    )
    (root / "depth").mkdir()
    receipt: dict[str, Any] = {
        "contract": cadence.ARM_RECEIPT_CONTRACT,
        "contract_version": cadence.ARM_RECEIPT_VERSION,
        "arm_id": "interval-59",
        "interval_frames": 59,
        "source_fps": 30.0,
        "burst_seconds": 20.0,
        "min_observations": 3,
        "depth_agreement_m": 0.18,
        "infer_config": {
            "path": config.name,
            "sha256": cadence._sha256_file(config),
        },
        "live_gate": {
            "report_path": live_name
            if (live_name := "mapanything-depth-quality.json")
            else "",
            "report_sha256": "a" * 64,
            "source_path": "mapanything-depth-quality-source.json",
            "source_sha256": "b" * 64,
        },
        "snapshot_store": "depth",
        "controlled_input_sha256s": {
            key: f"{index + 1:064x}"
            for index, key in enumerate(
                sorted(cadence.REQUIRED_CONTROL_FINGERPRINTS)
            )
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    (root / cadence.ARM_RECEIPT_FILENAME).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        cadence.load_arm(root, expected_interval=59)

    receipt["arm_id"] = "interval-89"
    (root / cadence.ARM_RECEIPT_FILENAME).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="self-digest"):
        cadence.load_arm(root, expected_interval=59)
