from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


identity_gate = _load_script(
    "ds9_identity_shadow_live_gate_test",
    "DS9/scripts/ds9_identity_shadow_live_gate.py",
)
floorplan_gate = _load_script(
    "ds9_floorplan_live_gate_test",
    "DS9/scripts/ds9_floorplan_live_gate.py",
)


def _health(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "runtime_mode": "shadow",
        "public_authority_cutover_status": "blocked",
        "authority_cutover_artifact_id": None,
        "runtime_model_fingerprint": "a" * 64,
        "runtime_model_layer": "fc_pred",
        "runtime_embedding_dim": 256,
        "scoring_calibration_status": "uncalibrated_default",
        "scoring_authority_scope": None,
        "resident_count": 0,
        "observation_cache_entries": 2,
        "observation_cache_max_entries": 512,
        "observation_cache_ttl_s": 10.0,
    }
    payload.update(overrides)
    return payload


def _identity_track(
    *,
    camera: str,
    tracker: int,
    frame: int,
    observed_at_us: int,
    state: str,
    subject: str | None,
    sid: int | None,
    reason: str,
    resident_uuid: str | None = None,
    display_name: str | None = None,
    visitor_generation: int | None = None,
    overlap_permit: bool = False,
) -> dict[str, object]:
    return {
        "type": "unused",
        "class_id": 0,
        "camera_id": camera,
        "tracker_id": tracker,
        "frame_id": frame,
        "observed_at_us": observed_at_us,
        "embedding_present": True,
        "identity_observation_key": {
            "run_id": "runtime-run-test",
            "camera_id": camera,
            "tracker_id": str(tracker),
            "frame_id": frame,
            "observation_id": "obs1:" + f"{frame:064x}"[-64:],
        },
        "identity_v2": {
            "mode": "shadow",
            "state": state,
            "reason": reason,
            "subject_id": subject,
            "compatibility_sid": sid,
            "display_name": display_name,
            "resident_uuid": resident_uuid,
            "visitor_generation": visitor_generation,
            "calibrated_confidence": 0.8,
            "provisional_evidence_count": 0,
            "overlap_permit": overlap_permit,
            "fresh_embedding": True,
        },
    }


def _identity_source_evidence(
    collector: object,
) -> dict[str, object]:
    return {
        "filename": "identity-open-set-occupied-source.json",
        "sha256": "d" * 64,
        "message_count": collector.tracking_messages,
        "row_count": collector.source_row_count,
        "first_observed_at_us": 1_000_000,
        "last_observed_at_us": 2_000_000,
        "observed_run_id": collector.observed_run_id,
    }


def _sealed_identity_bundle(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.chmod(0o700)
    collector = identity_gate.IdentityEvidenceCollector()
    collector.observe_payload(
        {
            "type": "tracking",
            "tracks": [
                _identity_track(
                    camera="kitchen",
                    tracker=1,
                    frame=1,
                    observed_at_us=1_000_000,
                    state="unknown",
                    subject=None,
                    sid=None,
                    reason="provisional_evidence_pending",
                ),
                _identity_track(
                    camera="kitchen",
                    tracker=2,
                    frame=10,
                    observed_at_us=1_100_000,
                    state="visitor",
                    subject="visitor:runtime:generation:0",
                    sid=1000,
                    reason="matched",
                    visitor_generation=0,
                ),
                _identity_track(
                    camera="kitchen",
                    tracker=2,
                    frame=11,
                    observed_at_us=1_200_000,
                    state="visitor",
                    subject="visitor:runtime:generation:0",
                    sid=1000,
                    reason="matched",
                    visitor_generation=0,
                ),
            ],
        }
    )
    raw_health = _health()
    health = identity_gate._validate_health_snapshot(
        raw_health,
        expected_layer="fc_pred",
        expected_dimension=256,
    )
    source = identity_gate._source_transcript_document(
        session_id="identity-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before=raw_health,
        health_after=raw_health,
        require_cross_camera=False,
        require_open_set=True,
        min_fresh_embeddings=2,
    )
    source_raw = identity_gate._encoded_private_json(source)
    report = identity_gate._build_report(
        session_id="identity-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before=health,
        health_after=health,
        require_cross_camera=False,
        require_open_set=True,
        min_fresh_embeddings=2,
        source_evidence=identity_gate._source_evidence_metadata(
            filename=identity_gate.BASELINE_SOURCE_TRANSCRIPT_FILENAME,
            encoded=source_raw,
            document=source,
            collector=collector,
        ),
        errors=[],
    )
    report_path = tmp_path / identity_gate.BASELINE_CANONICAL_REPORT_FILENAME
    source_path = tmp_path / identity_gate.BASELINE_SOURCE_TRANSCRIPT_FILENAME
    report_path.write_bytes(identity_gate._encoded_private_json(report))
    source_path.write_bytes(source_raw)
    report_path.chmod(0o600)
    source_path.chmod(0o600)
    return report_path, source_path


def test_identity_health_requires_shadow_blocked_authority_and_exact_model() -> None:
    before, after = identity_gate._validate_health_pair(
        _health(),
        _health(observation_cache_entries=4),
        expected_layer="fc_pred",
        expected_dimension=256,
    )
    assert before["runtime_mode"] == "shadow"
    assert after["observation_cache_entries"] == 4

    with pytest.raises(ValueError, match="remain shadow"):
        identity_gate._validate_health_snapshot(
            _health(runtime_mode="authoritative"),
            expected_layer="fc_pred",
            expected_dimension=256,
        )
    with pytest.raises(ValueError, match="remain blocked"):
        identity_gate._validate_health_snapshot(
            _health(public_authority_cutover_status="verified"),
            expected_layer="fc_pred",
            expected_dimension=256,
        )


def test_identity_sealed_report_requires_v2_health_backed_exact_replay(
    tmp_path: Path,
) -> None:
    report_path, source_path = _sealed_identity_bundle(tmp_path)

    validated = identity_gate.validate_sealed_identity_shadow_report(
        report_path,
        source_path,
        pipeline_config=REPO_ROOT / "DS9/config/infer.yaml",
        session_id="identity-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        require_cross_camera=False,
        require_open_set=True,
    )
    assert validated["ok"] is True

    tampered = json.loads(report_path.read_bytes())
    tampered["counts"]["fresh_embedding_rows"] += 1
    report_path.write_bytes(identity_gate._encoded_private_json(tampered))
    report_path.chmod(0o600)
    with pytest.raises(ValueError, match="exactly replay"):
        identity_gate.validate_sealed_identity_shadow_report(
            report_path,
            source_path,
            pipeline_config=REPO_ROOT / "DS9/config/infer.yaml",
            session_id="identity-gate-test",
            runtime_lane="baseline",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            require_cross_camera=False,
            require_open_set=True,
        )


def test_identity_sealed_report_rejects_duplicate_source_key(tmp_path: Path) -> None:
    report_path, source_path = _sealed_identity_bundle(tmp_path)
    source_path.write_bytes(
        source_path.read_bytes().replace(
            b'"schema_version": 2,',
            b'"schema_version": 2,\n  "schema_version": 2,',
            1,
        )
    )
    source_path.chmod(0o600)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        identity_gate.validate_sealed_identity_shadow_report(
            report_path,
            source_path,
            pipeline_config=REPO_ROOT / "DS9/config/infer.yaml",
            session_id="identity-gate-test",
            runtime_lane="baseline",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            require_cross_camera=False,
            require_open_set=True,
        )


def test_identity_sealed_report_rejects_nonfinite_source_number(tmp_path: Path) -> None:
    report_path, source_path = _sealed_identity_bundle(tmp_path)
    source_path.write_bytes(
        source_path.read_bytes().replace(
            b'"continuity_gap_s": 2.0',
            b'"continuity_gap_s": NaN',
            1,
        )
    )
    source_path.chmod(0o600)

    with pytest.raises(ValueError, match="non-finite JSON number"):
        identity_gate.validate_sealed_identity_shadow_report(
            report_path,
            source_path,
            pipeline_config=REPO_ROOT / "DS9/config/infer.yaml",
            session_id="identity-gate-test",
            runtime_lane="baseline",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            require_cross_camera=False,
            require_open_set=True,
        )


def test_identity_gate_separates_continuity_cross_camera_and_open_set_claims() -> None:
    collector = identity_gate.IdentityEvidenceCollector()
    unknown = _identity_track(
        camera="kitchen",
        tracker=1,
        frame=1,
        observed_at_us=1_000_000,
        state="unknown",
        subject=None,
        sid=None,
        reason="provisional_evidence_pending",
    )
    visitor_a1 = _identity_track(
        camera="kitchen",
        tracker=2,
        frame=10,
        observed_at_us=1_100_000,
        state="visitor",
        subject="visitor:session-a:generation:0",
        sid=1000,
        reason="visitor_provisional_confirmed",
        visitor_generation=0,
    )
    visitor_a2 = _identity_track(
        camera="kitchen",
        tracker=2,
        frame=11,
        observed_at_us=1_200_000,
        state="visitor",
        subject="visitor:session-a:generation:0",
        sid=1000,
        reason="matched",
        visitor_generation=0,
    )
    visitor_b = _identity_track(
        camera="family-room",
        tracker=9,
        frame=20,
        observed_at_us=1_250_000,
        state="visitor",
        subject="visitor:session-a:generation:0",
        sid=1000,
        reason="matched",
        visitor_generation=0,
        overlap_permit=True,
    )
    collector.observe_payload(
        {
            "type": "tracking",
            "tracks": [unknown, visitor_a1, visitor_a2, visitor_b],
        }
    )
    assert collector.errors == []
    assert collector.continuity_subjects == 1
    assert collector.cross_camera_subject_count == 1
    assert collector.open_set_non_force_rows == 1

    report = identity_gate._build_report(
        session_id="identity-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before={"runtime_mode": "shadow"},
        health_after={"runtime_mode": "shadow"},
        require_cross_camera=True,
        require_open_set=True,
        min_fresh_embeddings=2,
        source_evidence=_identity_source_evidence(collector),
        errors=[],
    )
    assert report["ok"] is True
    claims = report["claims"]
    assert claims["cross_camera_assignment_continuity"]["status"] == "observed"
    assert claims["open_set_non_force"]["status"] == "observed"
    assert claims["semantic_accuracy"]["status"] == "not_evaluated"
    assert claims["public_authority"]["status"] == "blocked"
    assert "session-a" not in json.dumps(report)


def test_identity_gate_fails_subject_flip_and_opt_in_missing_evidence() -> None:
    collector = identity_gate.IdentityEvidenceCollector()
    first = _identity_track(
        camera="kitchen",
        tracker=2,
        frame=10,
        observed_at_us=1_100_000,
        state="visitor",
        subject="visitor:first:generation:0",
        sid=1000,
        reason="matched",
        visitor_generation=0,
    )
    flipped = _identity_track(
        camera="kitchen",
        tracker=2,
        frame=11,
        observed_at_us=1_200_000,
        state="visitor",
        subject="visitor:second:generation:0",
        sid=1001,
        reason="matched",
        visitor_generation=0,
    )
    collector.observe_payload({"type": "tracking", "tracks": [first, flipped]})
    assert any("subject flipped" in error for error in collector.errors)

    report = identity_gate._build_report(
        session_id="identity-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before={"runtime_mode": "shadow"},
        health_after={"runtime_mode": "shadow"},
        require_cross_camera=True,
        require_open_set=True,
        min_fresh_embeddings=2,
        source_evidence=_identity_source_evidence(collector),
        errors=[],
    )
    assert report["ok"] is False
    assert any("cross-camera" in error for error in report["errors"])
    assert any("open-set" in error for error in report["errors"])


def test_identity_default_health_run_records_unexercised_semantics_without_passing_them() -> None:
    collector = identity_gate.IdentityEvidenceCollector()
    first = _identity_track(
        camera="kitchen",
        tracker=2,
        frame=10,
        observed_at_us=1_100_000,
        state="visitor",
        subject="visitor:session-a:generation:0",
        sid=1000,
        reason="matched",
        visitor_generation=0,
    )
    second = _identity_track(
        camera="kitchen",
        tracker=2,
        frame=11,
        observed_at_us=1_200_000,
        state="visitor",
        subject="visitor:session-a:generation:0",
        sid=1000,
        reason="matched",
        visitor_generation=0,
    )
    collector.observe_payload({"type": "tracking", "tracks": [first, second]})

    report = identity_gate._build_report(
        session_id="identity-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before={"runtime_mode": "shadow"},
        health_after={"runtime_mode": "shadow"},
        require_cross_camera=False,
        require_open_set=False,
        min_fresh_embeddings=2,
        source_evidence=_identity_source_evidence(collector),
        errors=[],
    )

    assert report["ok"] is True
    claims = report["claims"]
    assert claims["cross_camera_assignment_continuity"]["status"] == "not_observed"
    assert claims["cross_camera_assignment_continuity"]["required"] is False
    assert claims["open_set_non_force"]["status"] == "not_observed"
    assert claims["open_set_non_force"]["required"] is False
    assert claims["semantic_accuracy"]["status"] == "not_evaluated"


def _grid(values: list[float]) -> dict[str, object]:
    return {
        "grid_b64": __import__("base64").b64encode(
            struct.pack("<" + "f" * len(values), *values)
        ).decode("ascii"),
        "grid_shape": [2, 2],
        "value_min": min(values),
        "value_max": max(values),
    }


def _height_agl_grid_with_word(index: int, word: int) -> dict[str, object]:
    raw = bytearray(struct.pack("<ffff", 0.0, 0.1, 0.2, 0.3))
    raw[index * 4 : (index + 1) * 4] = struct.pack("<I", word)
    return {
        "grid_b64": __import__("base64").b64encode(bytes(raw)).decode("ascii"),
        "grid_shape": [2, 2],
        "value_min": 0.0,
        "value_max": 0.3,
    }


def _fusion_quality() -> dict[str, object]:
    return {
        "contract": "noesis.capture_event.depth_quality.v1",
        "support_valid_fraction": 1.0,
        "median_support": 2.0,
        "support_evidence": {
            "contract": "noesis.depth.fusion.support.v1",
            "cohort_size": 2,
            "full_frame_pixels": 4,
            "required_observations": 2,
            "fixed_min_observations": 2,
            "cohort_floor_observations": 2,
            "strict_majority_observations": 2,
            "tie_policy": "reject_exact_half_support",
            "min_observation_ratio": 0.6,
            "ratio_required_observations": 2,
            "quarantined_frame_count": 0,
            "quarantined_frame_indices": [],
            "eligible_pixels": 4,
            "eligible_full_frame_fraction": 1.0,
            "consensus_pixels": 4,
            "consensus_full_frame_fraction": 1.0,
            "consensus_retained_eligible_fraction": 1.0,
            "support_count_histogram": {"0": 0, "1": 0, "2": 4},
            "component_evidence": {
                "connectivity": 8,
                "component_count": 1,
                "largest_component_pixels": 4,
                "largest_component_fraction": 1.0,
                "fragment_pixels": 0,
                "fragment_fraction": 0.0,
                "hole_count": 0,
                "hole_pixels": 0,
            },
            "temporal_absolute_residual_median_m": 0.01,
            "temporal_absolute_residual_p95_m": 0.02,
        },
        "support_quality_gate": {
            "contract": "noesis.depth.fusion.quality_gate.v1",
            "metric": "consensus_full_frame_fraction",
            "observed": 1.0,
            "required": 0.4,
            "passed": True,
        },
        "frame_scale_normalization": {
            "contract": "noesis.depth.fusion.frame_scale_normalization.v1",
            "enabled": True,
            "algorithm": "depth_times_cohort_median_over_frame_median",
            "statistic": "median_valid_depth",
            "baseline": 2.0,
            "minimum_relative_change": 0.08,
            "accepted_factor_bounds": [0.5, 2.0],
            "frame_medians": [2.0, 2.0],
            "proposed_factors": [1.0, 1.0],
            "factors": [1.0, 1.0],
            "applied": [False, False],
            "rejected": [False, False],
            "rejection_policy": "quarantine_entire_frame",
        },
    }


def _floorplan_payload(
    *,
    camera: str = "kitchen",
    request_id: str = "request-1",
    snapshot_ts: int = 9_500_000,
) -> dict[str, object]:
    snapshot_ref = f"{camera}/capture-fused.zarr"
    snapshot_id = f"capture-fused-{camera}"
    snapshot_content_sha256 = "c" * 64
    event_id = "capture-event-sha256:" + "d" * 64
    capture_event: dict[str, object] = {
        "contract": "noesis.capture_event_controller",
        "contract_version": 1,
        "camera_id": camera,
        "request_kind": "floorplan",
        "capture_mode": "depth_rgb_exact",
        "baseline_raw_timestamp_us": snapshot_ts - 500_000,
        "baseline_mapanything_idle": {
            "active_captures": 0,
            "unfinished_tasks": 0,
            "worker_started": True,
            "worker_alive": True,
            "accepting": True,
        },
        "baseline_storage_flush": {
            "frontier_sequence": 20,
            "completed": True,
            "timed_out": False,
            "pending_sequences": [],
            "failed_sequences": [],
            "poisoned": False,
        },
        "postburst_mapanything_idle": {
            "active_captures": 0,
            "unfinished_tasks": 0,
            "worker_started": True,
            "worker_alive": True,
            "accepting": True,
        },
        "postburst_storage_flush": {
            "frontier_sequence": 24,
            "completed": True,
            "timed_out": False,
            "pending_sequences": [],
            "failed_sequences": [],
            "poisoned": False,
        },
        "parameters": {
            "burst_seconds": 4.0,
            "raw_limit": 24,
            "min_observations": 2,
            "depth_agreement_m": 0.18,
            "max_cohort_span_us": 20_000_000,
        },
        "fusion_evidence_sha256": "e" * 64,
        "raw_snapshot_count": 2,
        "fusion_quality": _fusion_quality(),
        "rgb": {
            "status": "available",
            "provider_configured": True,
            "source_id": 0,
            "batch_id": 0,
            "captured_at_us": snapshot_ts - 100_000,
            "frame_id": 42,
            "source_media_pts_ns": 12_000_000,
            "width": 1920,
            "height": 1080,
            "color_space": "rgb8",
            "content_sha256": "9" * 64,
        },
        "fused_snapshot": {
            "camera_id": camera,
            "storage_key": camera,
            "timestamp_us": snapshot_ts,
            "snapshot_id": snapshot_id,
            "artifact_ref": f"depth-zarr:{snapshot_ref}",
            "content_sha256": snapshot_content_sha256,
            "sequence": 24,
            "manifest_sha256": "f" * 64,
            "event_id": event_id,
            "source_snapshot_ids": ["raw-1", "raw-2"],
            "snapshot_role": "capture_event_fused",
            "fusion_level": "intra_capture",
        },
    }
    payload: dict[str, object] = {
        "type": "floorplan_response",
        "request_id": request_id,
        "camera_id": camera,
        "cache_only": False,
        "ts": snapshot_ts + 500_000,
        "snapshot_ts": snapshot_ts,
        "snapshot_ref": snapshot_ref,
        "snapshot_id": snapshot_id,
        "snapshot_content_sha256": snapshot_content_sha256,
        "frame": "camera_local_ground_m",
        "orientation": "camera_ground_right_forward",
        "floorplan_contract_version": 10,
        "units": "meters",
        "s_obj_to_m": 1.0,
        "bounds": {"min_x": -1.0, "max_x": 1.0, "min_z": 0.0, "max_z": 2.0},
        "scale_m_per_px": 0.5,
        "scale_scene_per_px": 0.5,
        "grid_res_m": 0.5,
        "grid_res_scene": 0.5,
        "max_extent_m": 20.0,
        "point_count": 20,
        "density": _grid([0.0, 0.2, 0.5, 1.0]),
        "observed": _grid([0.0, 1.0, 1.0, 1.0]),
        "unknown": _grid([1.0, 0.0, 0.0, 0.0]),
        "height": _grid([0.1, 0.2, 0.3, 0.4]),
        "height_agl": _grid([0.0, 0.1, 0.2, 0.3]),
        "distance": _grid([1.0, 2.0, 3.0, 4.0]),
        "walkable": _grid([1.0, 1.0, 0.0, 1.0]),
        "inferred_walkable": _grid([1.0, 0.0, 0.0, 0.0]),
        "observation_meta": {
            "contract": "noesis.floorplan.observation.v1",
            "observed_definition": "one_or_more_valid_projected_depth_points",
            "unknown_definition": (
                "zero_valid_projected_depth_points_within_grid_bounds"
            ),
            "observed_cells": 3,
            "unknown_cells": 1,
            "total_cells": 4,
        },
        "served_from_cache": False,
        "image_flip": {"u": False, "v": False},
        "calibration_fingerprint": "b" * 64,
        "depth_burst_triggered": True,
        "depth_burst_fresh": True,
        "capture_event": capture_event,
    }
    payload["capture_event_evidence_sha256"] = (
        floorplan_gate._canonical_json_sha256(capture_event)
    )
    return payload


def _cache_payload(fresh: dict[str, object], *, request_id: str) -> dict[str, object]:
    payload = copy.deepcopy(fresh)
    payload["request_id"] = request_id
    payload["cache_only"] = True
    payload["served_from_cache"] = True
    for key in (
        "depth_burst_triggered",
        "depth_burst_fresh",
        "capture_event",
        "capture_event_evidence_sha256",
    ):
        payload.pop(key)
    return payload


def _health_result(
    results: dict[str, dict[str, object]],
    *,
    inactive_cameras: tuple[str, ...] = (),
) -> dict[str, object]:
    camera_ids = tuple(results)
    inactive = set(inactive_cameras)
    active = set(camera_ids).difference(inactive)
    return {
        "bev_frame": "camera_local_ground_m",
        "bev_health": {
            "contract": "noesis.bev.health",
            "contract_version": 2,
            "healthy": True,
            "config_ready": True,
            "renderer_ready": True,
            "rendering_active": bool(active),
            "configured_camera_count": len(camera_ids),
            "active_camera_count": len(active),
            "inactive_camera_count": len(inactive),
            "failed_camera_count": 0,
            "configured_cameras": sorted(camera_ids),
            "active_cameras": sorted(active),
            "inactive_cameras": sorted(inactive),
            "failed_cameras": [],
            "unexpected_cameras": [],
            "expected_camera_count": len(camera_ids),
            "observed_camera_count": len(active),
            "expected_cameras": sorted(camera_ids),
            "missing_cameras": [],
            "unhealthy_cameras": [],
            "cameras": {
                camera_id: {
                    "state": (
                        "inactive_ready"
                        if camera_id in inactive
                        else "active_ready"
                    ),
                    "configured": True,
                    "active": camera_id in active,
                    "healthy": True,
                    "success_count": 0 if camera_id in inactive else 3,
                    "failure_count": 0,
                    "last_success_ts_us": (
                        None if camera_id in inactive else 11_000_000
                    ),
                    "last_failure_ts_us": None,
                    "last_failure_stage": None,
                    "last_failure_type": None,
                }
                for camera_id in camera_ids
            },
        },
        "active_floorplan_health": {
            "contract": "noesis.active_floorplan.health",
            "contract_version": 1,
            "healthy": True,
            "configured_camera_count": len(camera_ids),
            "active_camera_count": len(camera_ids),
            "missing_cameras": [],
            "rejection_count": 0,
            "stale_count": 0,
            "conflict_count": 0,
            "last_error": None,
            "reset_count": 0,
            "last_reset": None,
            "cameras": {
                camera_id: {
                    "snapshot_ts_us": result["snapshot_ts_us"],
                    "floorplan_ts_us": result["floorplan_ts_us"],
                    "snapshot_ref": result["snapshot_ref"],
                    "snapshot_id": result["snapshot_id"],
                    "snapshot_content_sha256": result[
                        "snapshot_content_sha256"
                    ],
                    "calibration_fingerprint": result[
                        "calibration_fingerprint"
                    ],
                    "frame": "camera_local_ground_m",
                    "units": "meters",
                }
                for camera_id, result in results.items()
            },
        },
        "capture_event_controller_health": {
            "contract": "noesis.capture_event_controller_health",
            "contract_version": 1,
            "canonical_camera_count": len(camera_ids),
            "alias_count": len(camera_ids) * 2,
            "request_kinds": ["depth", "floorplan"],
            "shared_gate_scope": "process",
            "shared_gate_owned": False,
            "healthy": True,
            "last_fatal_error_code": None,
            "counters": {
                "requests_total": len(camera_ids),
                "admitted_total": len(camera_ids),
                "completed_total": len(camera_ids),
                "failed_total": 0,
                "busy_total": 0,
                "cache_only_rejected_total": 0,
                "fusion_total": len(camera_ids),
                "fatal_barrier_failures_total": 0,
            },
            "cameras": {
                camera_id: {
                    "active": False,
                    "active_request_kind": None,
                    "requests_total": 1,
                    "completed_total": 1,
                    "failed_total": 0,
                    "busy_total": 0,
                    "last_error_code": None,
                    "last_fused_timestamp_us": result["snapshot_ts_us"],
                    "last_fused_snapshot_id": result["snapshot_id"],
                }
                for camera_id, result in results.items()
            },
        },
    }


def test_floorplan_gate_requires_fresh_nonempty_contract_grid() -> None:
    evidence = floorplan_gate._validate_floorplan_payload(
        _floorplan_payload(),
        camera_id="kitchen",
        request_id="request-1",
        max_age_sec=2.0,
        now_us=10_000_000,
    )
    assert evidence["point_count"] == 20
    assert evidence["grid_shape"] == [2, 2]
    assert evidence["served_from_cache"] is False
    assert evidence["capture_event_rgb_status"] == "available"
    assert evidence["capture_event_rgb_evidence"]["source_id"] == 0
    assert evidence["capture_event_rgb_evidence"]["frame_id"] == 42
    assert evidence["capture_event_rgb_evidence"]["source_media_pts_ns"] == 12_000_000
    assert evidence["snapshot_ref"] == "kitchen/capture-fused.zarr"
    assert evidence["floorplan_contract_version"] == 10
    assert evidence["observation_meta"]["observed_cells"] == 3
    assert evidence["observation_meta"]["unknown_cells"] == 1
    assert evidence["inferred_walkable_present"] is True
    assert {
        "observed",
        "unknown",
        "height",
        "height_agl",
        "walkable",
        "inferred_walkable",
    } <= set(evidence["layer_sha256s"])

    zero = _floorplan_payload()
    zero["point_count"] = 0
    with pytest.raises(ValueError, match="point_count"):
        floorplan_gate._validate_floorplan_payload(
            zero,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


def test_floorplan_gate_preserves_legacy_depth_only_rgb_evidence() -> None:
    payload = _floorplan_payload()
    capture = payload["capture_event"]
    assert isinstance(capture, dict)
    capture["capture_mode"] = "depth_only"
    capture["rgb"] = {
        "status": "not_requested",
        "provider_configured": False,
    }
    payload["capture_event_evidence_sha256"] = (
        floorplan_gate._canonical_json_sha256(capture)
    )
    evidence = floorplan_gate._validate_floorplan_payload(
        payload,
        camera_id="kitchen",
        request_id="request-1",
        max_age_sec=2.0,
        now_us=10_000_000,
    )
    assert evidence["capture_event_rgb_status"] == "not_requested"
    assert evidence["capture_event_rgb_evidence"] == {
        "status": "not_requested",
        "provider_configured": False,
    }
    with pytest.raises(ValueError, match="age bound"):
        floorplan_gate._validate_floorplan_payload(
            _floorplan_payload(snapshot_ts=1_000_000),
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("source_media_pts_ns", (1 << 64) - 1, "CLOCK_TIME_NONE"),
        ("width", 16_385, "provider bounds"),
        ("height", 16_385, "provider bounds"),
        ("width", 16_384, "provider bounds"),
    ),
)
def test_floorplan_gate_rejects_unbounded_exact_rgb_evidence(
    field: str,
    value: int,
    match: str,
) -> None:
    payload = _floorplan_payload()
    capture = payload["capture_event"]
    assert isinstance(capture, dict)
    rgb = capture["rgb"]
    assert isinstance(rgb, dict)
    rgb[field] = value
    if field == "width" and value == 16_384:
        rgb["height"] = 1_366
    payload["capture_event_evidence_sha256"] = (
        floorplan_gate._canonical_json_sha256(capture)
    )

    with pytest.raises(ValueError, match=match):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


def test_floorplan_gate_rejects_missing_or_inconsistent_fusion_quality() -> None:
    payload = _floorplan_payload()
    capture = payload["capture_event"]
    assert isinstance(capture, dict)
    capture.pop("fusion_quality")
    payload["capture_event_evidence_sha256"] = (
        floorplan_gate._canonical_json_sha256(capture)
    )
    with pytest.raises(ValueError, match="schema drifted"):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )

    payload = _floorplan_payload()
    capture = payload["capture_event"]
    assert isinstance(capture, dict)
    quality = capture["fusion_quality"]
    assert isinstance(quality, dict)
    quality["support_valid_fraction"] = 0.5
    payload["capture_event_evidence_sha256"] = (
        floorplan_gate._canonical_json_sha256(capture)
    )
    with pytest.raises(ValueError, match="fractions do not reconcile"):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("complement", "exact complements"),
        ("density", "projected-point density"),
        ("metadata", "observation metadata"),
        ("missing_inferred", "present exactly when walkable"),
        ("wrong_inferred", "walkable intersect unknown"),
    ),
)
def test_floorplan_gate_rejects_v8_observation_contract_drift(
    mutation: str,
    match: str,
) -> None:
    payload = _floorplan_payload()
    if mutation == "complement":
        payload["unknown"] = _grid([1.0, 0.0, 0.0, 1.0])
    elif mutation == "density":
        payload["observed"] = _grid([1.0, 0.0, 1.0, 1.0])
        payload["unknown"] = _grid([0.0, 1.0, 0.0, 0.0])
    elif mutation == "metadata":
        payload["observation_meta"]["observed_cells"] = 2
    elif mutation == "missing_inferred":
        payload.pop("inferred_walkable")
    else:
        payload["inferred_walkable"] = _grid([1.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match=match):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


def test_floorplan_gate_accepts_canonical_height_agl_nan_in_unknown_cell() -> None:
    payload = _floorplan_payload()
    payload["height_agl"] = _height_agl_grid_with_word(0, 0x7FC00000)

    evidence = floorplan_gate._validate_floorplan_payload(
        payload,
        camera_id="kitchen",
        request_id="request-1",
        max_age_sec=2.0,
        now_us=10_000_000,
    )

    assert evidence["grid_shape"] == [2, 2]


def test_floorplan_gate_rejects_canonical_height_agl_nan_in_observed_cell() -> None:
    payload = _floorplan_payload()
    payload["height_agl"] = _height_agl_grid_with_word(1, 0x7FC00000)

    with pytest.raises(ValueError, match="only in unknown cells"):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


def test_floorplan_gate_rejects_noncanonical_height_agl_nan() -> None:
    payload = _floorplan_payload()
    payload["height_agl"] = _height_agl_grid_with_word(0, 0x7FC00001)

    with pytest.raises(ValueError, match="invalid float32"):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


def test_floorplan_gate_derives_every_configured_camera_from_reviewed_configs(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "infer.yaml"
    cameras = tmp_path / "cameras.yaml"
    pipeline.write_text(
        "sources:\n  - {element: nvurisrcbin}\n  - {element: nvurisrcbin}\n  - {element: nvurisrcbin, enable: false}\n",
        encoding="utf-8",
    )
    cameras.write_text(
        "cameras:\n  0: {name: living-room}\n  1: {name: kitchen}\n  2: {name: disabled}\n",
        encoding="utf-8",
    )
    assert floorplan_gate._active_camera_ids(pipeline, cameras) == (
        "living-room",
        "kitchen",
    )

    report = floorplan_gate._build_report(
        session_id="floorplan-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        camera_ids=("living-room", "kitchen"),
        results={"living-room": {"camera_id": "living-room"}},
        cache_results={},
        after_fresh_health=None,
        after_cache_only_health=None,
        max_snapshot_age_s=2.0,
        errors=[],
    )
    assert report["ok"] is False
    assert report["all_configured_camera_floorplans_validated"] is False


def test_floorplan_health_accepts_empty_camera_as_inactive_ready() -> None:
    results = {
        camera: floorplan_gate._validate_floorplan_payload(
            _floorplan_payload(
                camera=camera,
                snapshot_ts=10_000_000 + index,
                request_id=f"request-{index}",
            ),
            camera_id=camera,
            request_id=f"request-{index}",
            max_age_sec=2.0,
            now_us=10_100_000,
        )
        for index, camera in enumerate(("living-room", "kitchen"))
    }
    health = _health_result(results, inactive_cameras=("kitchen",))

    validated = floorplan_gate._validate_health_result(
        health,
        camera_ids=("living-room", "kitchen"),
        fresh_results=results,
    )

    bev = validated["bev_health"]
    assert bev["healthy"] is True
    assert bev["active_cameras"] == ["living-room"]
    assert bev["inactive_cameras"] == ["kitchen"]
    assert bev["failed_camera_count"] == 0

    failed = copy.deepcopy(health)
    failed_bev = failed["bev_health"]
    failed_bev["healthy"] = False
    failed_bev["renderer_ready"] = False
    failed_bev["failed_camera_count"] = 1
    failed_bev["failed_cameras"] = ["kitchen"]
    failed_bev["unhealthy_cameras"] = ["kitchen"]
    failed_bev["cameras"]["kitchen"]["healthy"] = False
    failed_bev["cameras"]["kitchen"]["state"] = "failed"
    failed_bev["cameras"]["kitchen"]["failure_count"] = 1
    with pytest.raises(ValueError, match="readiness/activity"):
        floorplan_gate._validate_health_result(
            failed,
            camera_ids=("living-room", "kitchen"),
            fresh_results=results,
        )


def test_floorplan_source_transcript_is_minimal_and_replayable() -> None:
    fresh_payload = _floorplan_payload()
    result = floorplan_gate._validate_floorplan_payload(
        fresh_payload,
        camera_id="kitchen",
        request_id="request-1",
        max_age_sec=2.0,
        now_us=10_000_000,
    )
    cache_payload = _cache_payload(fresh_payload, request_id="request-cache-1")
    cache_result = floorplan_gate._validate_floorplan_payload(
        cache_payload,
        camera_id="kitchen",
        request_id="request-cache-1",
        max_age_sec=2.0,
        now_us=10_000_000,
        mode="cache_only",
        expected_fresh=result,
    )
    health = _health_result({"kitchen": result})
    events = [{
        "type": "validated_exact_floorplan_capture",
        "observed_at_us": 2_000_001,
        "request_id": "request-1",
        "attempt": 1,
        "camera_id": "kitchen",
        "result": result,
        "capture_event": fresh_payload["capture_event"],
    }, {
        "type": "validated_floorplan_runtime_health",
        "observed_at_us": 2_000_002,
        "phase": "after_fresh",
        "result": health,
    }, {
        "type": "validated_cache_only_floorplan",
        "observed_at_us": 2_000_003,
        "request_id": "request-cache-1",
        "attempt": 1,
        "camera_id": "kitchen",
        "result": cache_result,
    }, {
        "type": "validated_floorplan_runtime_health",
        "observed_at_us": 2_000_004,
        "phase": "after_cache_only",
        "result": copy.deepcopy(health),
    }]
    source = floorplan_gate._source_transcript_document(
        session_id="floorplan-gate-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        camera_ids=("kitchen",),
        max_snapshot_age_s=2.0,
        events=events,
    )
    replay = floorplan_gate._replay_source_events(
        source["messages"], camera_ids=("kitchen",)
    )
    assert replay["fresh_results"] == {"kitchen": result}
    assert replay["cache_results"] == {"kitchen": cache_result}
    encoded = json.dumps(source["messages"], sort_keys=True)
    for forbidden in ("grid_b64", "depth_b64", "embedding", "token", "secret"):
        assert forbidden not in encoded

    poisoned = copy.deepcopy(events)
    poisoned[2]["result"]["snapshot_id"] = "different"
    with pytest.raises(ValueError, match="mutated immutable evidence"):
        floorplan_gate._replay_source_events(
            poisoned, camera_ids=("kitchen",)
        )

    mutated_health = copy.deepcopy(events)
    mutated_health[3]["result"]["capture_event_controller_health"]["counters"][
        "requests_total"
    ] += 1
    with pytest.raises(ValueError, match="health mutation"):
        floorplan_gate._replay_source_events(
            mutated_health, camera_ids=("kitchen",)
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("rgb", "content_sha256"),
        ("snapshot", "fused identity"),
        ("digest", "evidence digest"),
        ("cache", "non-cache capture"),
        ("frontier", "clean storage frontier"),
    ),
)
def test_floorplan_gate_rejects_adversarial_capture_evidence(
    mutation: str,
    match: str,
) -> None:
    payload = _floorplan_payload()
    capture = payload["capture_event"]
    assert isinstance(capture, dict)
    if mutation == "rgb":
        capture["rgb"]["content_sha256"] = "not-a-digest"
    elif mutation == "snapshot":
        capture["fused_snapshot"]["snapshot_id"] = "substituted"
    elif mutation == "digest":
        payload["capture_event_evidence_sha256"] = "0" * 64
    elif mutation == "cache":
        payload["served_from_cache"] = True
    else:
        capture["postburst_storage_flush"]["pending_sequences"] = [24]
    if mutation not in {"digest", "cache"}:
        payload["capture_event_evidence_sha256"] = (
            floorplan_gate._canonical_json_sha256(capture)
        )
    with pytest.raises(ValueError, match=match):
        floorplan_gate._validate_floorplan_payload(
            payload,
            camera_id="kitchen",
            request_id="request-1",
            max_age_sec=2.0,
            now_us=10_000_000,
        )


def test_floorplan_live_collection_serializes_fresh_then_cache_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_ids = ("living-room", "kitchen")

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []
            self.responses: asyncio.Queue[str] = asyncio.Queue()
            self.fresh_payloads: dict[str, dict[str, object]] = {}

        async def send(self, raw: str) -> None:
            request = json.loads(raw)
            self.sent.append(request)
            camera = str(request["camera"])
            request_id = str(request["request_id"])
            if request["cache_only"] is False:
                payload = _floorplan_payload(
                    camera=camera,
                    request_id=request_id,
                    snapshot_ts=time.time_ns() // 1_000 - 10_000,
                )
                self.fresh_payloads[camera] = payload
                await self.responses.put(json.dumps(payload))
                if len(self.fresh_payloads) == len(camera_ids):
                    results = {
                        key: floorplan_gate._validate_floorplan_payload(
                            value,
                            camera_id=key,
                            request_id=str(value["request_id"]),
                            max_age_sec=120.0,
                        )
                        for key, value in self.fresh_payloads.items()
                    }
                    health = _health_result(results)
                    await self.responses.put(
                        json.dumps(
                            {
                                "type": "stats",
                                "payload": {
                                    "pipeline": {
                                        "bev": {
                                            "frame": health["bev_frame"],
                                            "health": health["bev_health"],
                                        },
                                        "active_floorplan": health[
                                            "active_floorplan_health"
                                        ],
                                        "capture_event_fusion": health[
                                            "capture_event_controller_health"
                                        ],
                                    }
                                },
                            }
                        )
                    )
            else:
                await self.responses.put(
                    json.dumps(
                        _cache_payload(
                            self.fresh_payloads[camera], request_id=request_id
                        )
                    )
                )
                cache_count = sum(
                    request["cache_only"] is True for request in self.sent
                )
                if cache_count == len(camera_ids):
                    results = {
                        key: floorplan_gate._validate_floorplan_payload(
                            value,
                            camera_id=key,
                            request_id=str(value["request_id"]),
                            max_age_sec=120.0,
                        )
                        for key, value in self.fresh_payloads.items()
                    }
                    health = _health_result(results)
                    await self.responses.put(
                        json.dumps(
                            {
                                "type": "stats",
                                "payload": {
                                    "pipeline": {
                                        "bev": {
                                            "frame": health["bev_frame"],
                                            "health": health["bev_health"],
                                        },
                                        "active_floorplan": health[
                                            "active_floorplan_health"
                                        ],
                                        "capture_event_fusion": health[
                                            "capture_event_controller_health"
                                        ],
                                    }
                                },
                            }
                        )
                    )

        async def recv(self) -> str:
            return await self.responses.get()

    websocket = FakeWebSocket()

    class Connection:
        async def __aenter__(self) -> FakeWebSocket:
            return websocket

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        floorplan_gate,
        "connect_required_websocket",
        lambda *_args, **_kwargs: Connection(),
    )
    fresh, cache, before, after, errors, events = asyncio.run(
        floorplan_gate._collect_floorplans(
            "ws://127.0.0.1:6008",
            object(),
            camera_ids=camera_ids,
            max_age_sec=120.0,
            timeout_s=5.0,
            max_attempts=1,
        )
    )
    assert errors == []
    assert tuple(fresh) == camera_ids
    assert tuple(cache) == camera_ids
    assert before is not None and after is not None
    assert [(row["camera"], row["cache_only"]) for row in websocket.sent] == [
        ("living-room", False),
        ("kitchen", False),
        ("living-room", True),
        ("kitchen", True),
    ]
    floorplan_gate._replay_source_events(events, camera_ids=camera_ids)
