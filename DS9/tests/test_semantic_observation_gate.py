from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from noesis_core.runtime_world import create_runtime_world_service

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "ds9_semantic_observation_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("ds9_semantic_observation_gate_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

MODEL_SHA = "a" * 64
SEMANTIC_SHA = "b" * 64
CALIBRATION_SHA = "1" * 64
WORLD_MODEL_SHA = "2" * 64
CONFIG_SHA = "3" * 64
RUN_ID = "runtime-run-test"
INSTANCE_ID = "runtime-instance-test"
SESSION_ID = "session-a"
CAMERA_ID = "kitchen"
TRACKER_ID = 7
RESIDENT_UUID = "11111111-1111-4111-8111-111111111111"
RESIDENT_SUBJECT = f"resident:{RESIDENT_UUID}"
VISITOR_SESSION = "22222222-2222-4222-8222-222222222222"
VISITOR_SUBJECT = f"visitor:{VISITOR_SESSION}:generation:3"
ACQUISITION_STARTED_AT_US = 1
ACQUISITION_FINISHED_AT_US = 5_000_000


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(
    *,
    state: str = "resident",
    subject_id: str | None = RESIDENT_SUBJECT,
    compatibility_sid: int | None = 1,
    resident_uuid: str | None = RESIDENT_UUID,
    visitor_generation: int | None = None,
    fresh_embedding: bool = False,
) -> dict[str, object]:
    return {
        "mode": "shadow",
        "state": state,
        "reason": "accepted",
        "subject_id": subject_id,
        "compatibility_sid": compatibility_sid,
        "display_name": "Resident" if state == "resident" else None,
        "resident_uuid": resident_uuid,
        "visitor_generation": visitor_generation,
        "fresh_embedding": fresh_embedding,
        "evidence_persistence": None,
    }


def _evidence_row(**overrides: object):
    candidate = {
        "subject_id": RESIDENT_SUBJECT,
        "identity_kind": "resident",
        "raw_similarity": 0.9,
        "hard_allowed": True,
        "hard_constraint_reason": None,
        "gallery_exemplar_count": 2,
    }
    body: dict[str, object] = {
        "contract": "noesis.identity.shadow_score_evidence",
        "contract_version": 2,
        "sequence": 7,
        "previous_event_id": "d" * 64,
        "observed_at_us": 1_000_000,
        "source": "shadow",
        "session_id": SESSION_ID,
        "runtime": "ds9",
        "runtime_mode": "shadow",
        "run_id": RUN_ID,
        "camera_id": CAMERA_ID,
        "tracker_id": str(TRACKER_ID),
        "frame_id": 11,
        "observation_id": "identity-observation-anchor",
        "model_sha256": MODEL_SHA,
        "model_semantic_profile_sha256": SEMANTIC_SHA,
        "model_layer": "fc_pred",
        "embedding_dim": 256,
        "quality": 0.9,
        "candidates": [candidate],
        "pre_prior_winner_id": RESIDENT_SUBJECT,
        "final_winner_id": RESIDENT_SUBJECT,
        "final_outcome": "resident",
        "calibrated_confidence": 0.9,
        "reject_reason": None,
        "prior_changed_winner": False,
    }
    body.update(overrides)
    event_id = hashlib.sha256(
        b"noesis-identity-shadow-evidence-v2\0" + _canonical_bytes(body)
    ).hexdigest()
    return gate.ShadowIdentityEvidenceRecord.model_validate(
        {"event_id": event_id, **body}
    )


def _snapshot(**row_overrides: object):
    row = _evidence_row(**row_overrides)
    payload = _canonical_bytes(row.model_dump(mode="json")) + b"\n"
    return gate.EvidenceSnapshot(
        rows_by_sequence={row.sequence: row},
        byte_count=len(payload),
        row_count=1,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_path="/private/session/runtime/identity_v2.jsonl",
        payload=payload,
    )


def _tracking_payload(
    *,
    frame_id: int = 11,
    observed_at_us: int = 1_000_000,
    source_id: int = 0,
    camera_id: str = CAMERA_ID,
    tracker_id: int = TRACKER_ID,
    run_id: str = RUN_ID,
    sequence: int = 1,
    provenance: bool = True,
    identity: dict[str, object] | None = None,
    pose: bool = True,
    depth: bool = True,
    world: bool = True,
    calibration_sha256: str = CALIBRATION_SHA,
    world_model_sha256: str = WORLD_MODEL_SHA,
    config_sha256: str = CONFIG_SHA,
    identity_observation_id: str = "identity-observation-anchor",
    lifecycle_generation: int = 1,
    publication_sequence: int | None = None,
    captured_at_us: int | None = None,
    media_pts_ns: int | None = None,
) -> dict[str, object]:
    publication_sequence = (
        int(sequence) - 1
        if publication_sequence is None
        else int(publication_sequence)
    )
    captured_at_us = (
        int(observed_at_us) if captured_at_us is None else int(captured_at_us)
    )
    media_pts_ns = (
        int(observed_at_us) * 1000 if media_pts_ns is None else int(media_pts_ns)
    )
    identity_payload = copy.deepcopy(
        identity
        if identity is not None
        else _identity(fresh_embedding=provenance)
    )
    identity_payload["fresh_embedding"] = provenance
    identity_payload["evidence_persistence"] = "durable" if provenance else None
    observation_payload: dict[str, object] = {
        "tracklet": {
            "run_id": run_id,
            "camera_id": camera_id,
            "source_id": source_id,
            "tracker_id": tracker_id,
            "frame_id": frame_id,
            "observed_at_us": observed_at_us,
        },
        "class_id": 0,
        "detection_confidence": 0.9,
        "tracker_confidence": 0.8,
        "bbox_xywh": [10.0, 20.0, 100.0, 200.0],
        "image_size": [1920, 1080],
        "zone": camera_id,
        "world": (
            {
                "position": {"x": 1.0, "y": 0.0, "z": 2.0},
                "covariance": {
                    "values": [
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                    ]
                },
                "frame": "backend_world_m",
                "units": "meters",
                "source": "pose_floor_only",
                "quality": "good",
                "confidence": 0.9,
                "reason": None,
            }
            if world
            else None
        ),
        "pose_present": pose,
        "depth_present": depth,
        "occluded": False,
    }
    track: dict[str, object] = {
        "class_id": 0,
        "camera_id": camera_id,
        "tracker_id": tracker_id,
        "tracker_lifecycle_generation": lifecycle_generation,
        "frame_id": frame_id,
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "image_size": [1920, 1080],
        "observed_at_us": observed_at_us,
        "embedding_present": provenance,
        "identity_v2": identity_payload,
        "pose_present": pose,
        "depth_status": "ok" if depth else "missing",
        "world_valid": world,
        "world_frame": "backend_world_m" if world else "image_px",
        "world": [1.0, 0.0, 2.0] if world else None,
    }
    if depth:
        track["depth_registration_status"] = "ok"
        track["depth_registered_m"] = 2.5
        track["depth_used_m"] = 2.5
    if provenance:
        provenance_fields = {
            "embedding_sequence": 7,
            "embedding_model_sha256": MODEL_SHA,
            "embedding_dimension": 256,
        }
        track.update(provenance_fields)
        observation_payload.update(provenance_fields)
        track["identity_observation_key"] = {
            "run_id": run_id,
            "camera_id": camera_id,
            "tracker_id": str(tracker_id),
            "frame_id": frame_id,
            "observation_id": identity_observation_id,
        }
    observation = {
        "contract": "noesis.observation.person",
        "contract_version": 1,
        "observation_id": f"{run_id}:{source_id}:{sequence}:{tracker_id}",
        "producer": {
            "runtime": "ds9",
            "instance_id": INSTANCE_ID,
            "run_id": run_id,
            "software_revision": "test",
        },
        "sequence": sequence,
        "captured_at_us": captured_at_us,
        "observed_at_us": observed_at_us,
        "published_at_us": observed_at_us + 1,
        "capture_time_status": "synced",
        "media_pts_ns": media_pts_ns,
        "coordinate_frame": "backend_world_m" if world else "image_px",
        "units": "meters" if world else "pixels",
        "calibration": {"role": "camera_calibration", "sha256": calibration_sha256},
        "model": {"role": "tracking_model_manifest", "sha256": world_model_sha256},
        "config": {"role": "pipeline_config", "sha256": config_sha256},
        "payload": observation_payload,
    }
    return {
        "type": "tracking",
        "source_id": source_id,
        "frame_id": frame_id,
        "captured_at_us": captured_at_us,
        "observed_at_us": observed_at_us,
        "capture_time_status": "synced",
        "media_pts_ns": media_pts_ns,
        "tracking_continuity_contract": gate.TRACKING_CONTINUITY_CONTRACT,
        "tracking_continuity_contract_version": (
            gate.TRACKING_CONTINUITY_CONTRACT_VERSION
        ),
        "tracking_publication_sequence": publication_sequence,
        "tracker_lifecycle_tombstones": [],
        "track_count": 1,
        "tracks": [track],
        "observation_contract": "noesis.observation.person",
        "observation_contract_version": 1,
        "observations": [observation],
    }


def _empty_tracking_payload(
    *,
    frame_id: int,
    observed_at_us: int,
    publication_sequence: int | None = None,
    tombstones: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    publication_sequence = (
        int(frame_id) - 11
        if publication_sequence is None
        else int(publication_sequence)
    )
    return {
        "type": "tracking",
        "source_id": 0,
        "frame_id": frame_id,
        "captured_at_us": observed_at_us,
        "observed_at_us": observed_at_us,
        "capture_time_status": "synced",
        "media_pts_ns": observed_at_us * 1000,
        "tracking_continuity_contract": gate.TRACKING_CONTINUITY_CONTRACT,
        "tracking_continuity_contract_version": (
            gate.TRACKING_CONTINUITY_CONTRACT_VERSION
        ),
        "tracking_publication_sequence": publication_sequence,
        "tracker_lifecycle_tombstones": list(tombstones or []),
        "track_count": 0,
        "tracks": [],
        "observation_contract": "noesis.observation.person",
        "observation_contract_version": 1,
        "observations": [],
    }


def _stats(errors: list[str] | None = None) -> dict[str, object]:
    return {
        "type": "stats",
        "payload": {"pipeline": {"errors": list(errors or [])}},
    }


def _evaluate(
    tracking: dict[str, object] | list[dict[str, object]],
    *,
    snapshot=None,
    pipeline_errors: list[str] | None = None,
    acquisition_started_at_us: int = ACQUISITION_STARTED_AT_US,
    acquisition_finished_at_us: int = ACQUISITION_FINISHED_AT_US,
):
    collector = gate.SemanticObservationCollector()
    collector.observe_payload(_stats(pipeline_errors))
    rows = tracking if isinstance(tracking, list) else [tracking]
    for payload in rows:
        collector.observe_payload(payload)
    source_document = gate._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
        collector=collector,
    )
    source_encoded = gate._encoded_source_transcript(source_document)
    report = gate._evaluate(
        collector,
        _snapshot() if snapshot is None else snapshot,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
        sealed_snapshot_filename="semantic-identity-evidence.jsonl",
        source_evidence=gate._source_evidence_metadata(
            filename="semantic-observation-source.json",
            encoded=source_encoded,
            document=source_document,
        ),
    )
    return collector, report


def test_semantic_gate_accepts_exact_frame_and_private_anchor() -> None:
    collector, report = _evaluate(_tracking_payload())

    assert collector.errors == []
    assert report["schema_version"] == 3
    assert report["contract_version"] == 3
    assert report["ok"] is True
    assert report["status"] == "pass"
    assert report["counts"]["accepted_semantic_cohorts"] == 1
    assert report["sample_cohort"]["span_us"] == 0
    assert gate._raw_vector_paths(report) == []


@pytest.mark.parametrize("anchor_at_end", (False, True))
def test_semantic_gate_accepts_both_exact_temporal_endpoints(
    anchor_at_end: bool,
) -> None:
    start = 1_000_000
    end = start + gate.SEMANTIC_COHORT_MAX_SPAN_US
    if anchor_at_end:
        rows = [
            _tracking_payload(
                frame_id=10,
                observed_at_us=start,
                sequence=1,
                provenance=False,
                pose=True,
                depth=False,
                world=False,
            ),
            _tracking_payload(
                frame_id=11,
                observed_at_us=start + 500_000,
                sequence=2,
                provenance=False,
                pose=False,
                depth=True,
                world=False,
            ),
            _tracking_payload(
                frame_id=12,
                observed_at_us=start + 1_000_000,
                sequence=3,
                provenance=False,
                pose=False,
                depth=False,
                world=True,
            ),
            _tracking_payload(
                frame_id=13,
                observed_at_us=end,
                sequence=4,
                pose=False,
                depth=False,
                world=False,
            ),
        ]
        snapshot = _snapshot(frame_id=13, observed_at_us=end)
    else:
        rows = [
            _tracking_payload(pose=False, depth=False, world=False),
            _tracking_payload(
                frame_id=12,
                observed_at_us=start + 500_000,
                sequence=2,
                provenance=False,
                pose=True,
                depth=False,
                world=False,
            ),
            _tracking_payload(
                frame_id=13,
                observed_at_us=start + 1_000_000,
                sequence=3,
                provenance=False,
                pose=False,
                depth=True,
                world=False,
            ),
            _tracking_payload(
                frame_id=14,
                observed_at_us=end,
                sequence=4,
                provenance=False,
                pose=False,
                depth=False,
                world=True,
            ),
        ]
        snapshot = _snapshot()

    _collector, report = _evaluate(rows, snapshot=snapshot)

    assert report["ok"] is True
    assert report["sample_cohort"]["span_us"] == gate.SEMANTIC_COHORT_MAX_SPAN_US


@pytest.mark.parametrize("anchor_at_end", (False, True))
def test_semantic_gate_rejects_one_microsecond_beyond_either_endpoint(
    anchor_at_end: bool,
) -> None:
    start = 1_000_000
    delta = gate.SEMANTIC_COHORT_MAX_SPAN_US + 1
    if anchor_at_end:
        rows = [
            _tracking_payload(
                frame_id=10,
                observed_at_us=start,
                sequence=1,
                provenance=False,
            ),
            _tracking_payload(
                frame_id=11,
                observed_at_us=start + delta,
                sequence=2,
                pose=False,
                depth=False,
                world=False,
            ),
        ]
        snapshot = _snapshot(frame_id=11, observed_at_us=start + delta)
    else:
        rows = [
            _tracking_payload(pose=False, depth=False, world=False),
            _tracking_payload(
                frame_id=12,
                observed_at_us=start + delta,
                sequence=2,
                provenance=False,
            ),
        ]
        snapshot = _snapshot()

    _collector, report = _evaluate(rows, snapshot=snapshot)

    assert report["ok"] is False
    assert report["component_availability"] == {
        "pose": True,
        "usable_depth": True,
        "backend_world_m": True,
    }
    assert report["counts"]["accepted_semantic_cohorts"] == 0


def test_semantic_gate_reports_components_independently_of_cross_camera_rejection() -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _tracking_payload(
            frame_id=12,
            observed_at_us=1_500_000,
            source_id=1,
            camera_id="family-room",
            tracker_id=9,
            sequence=1,
            provenance=False,
        ),
    ]

    _collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert all(report["component_availability"].values())
    assert report["counts"]["accepted_semantic_cohorts"] == 0


def test_semantic_gate_rejects_cross_run_component() -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _tracking_payload(
            frame_id=12,
            observed_at_us=1_500_000,
            sequence=2,
            run_id="other-runtime-run",
            provenance=False,
        ),
    ]

    _collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert any("runtime identity drifted" in error for error in report["errors"])


def test_semantic_gate_rejects_artifact_fingerprint_splice() -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _tracking_payload(
            frame_id=12,
            observed_at_us=1_500_000,
            sequence=2,
            provenance=False,
            calibration_sha256="9" * 64,
        ),
    ]

    _collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert report["counts"]["fingerprint_discontinuities"] >= 1
    assert report["counts"]["accepted_semantic_cohorts"] == 0


def test_semantic_gate_rejects_tracker_id_reuse_after_observed_absence() -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _empty_tracking_payload(
            frame_id=12,
            observed_at_us=1_250_000,
            tombstones=[
                {
                    "camera_id": CAMERA_ID,
                    "tracker_id": TRACKER_ID,
                    "tracker_lifecycle_generation": 1,
                    "last_seen_frame_id": 11,
                    "last_seen_observed_at_us": 1_000_000,
                    "disappeared_at_frame_id": 12,
                    "disappeared_at_observed_at_us": 1_250_000,
                }
            ],
        ),
        _tracking_payload(
            frame_id=13,
            observed_at_us=1_500_000,
            sequence=2,
            provenance=False,
            lifecycle_generation=2,
            publication_sequence=2,
        ),
    ]

    _collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert report["counts"]["tracker_reuse_discontinuities"] >= 1


@pytest.mark.parametrize(
    "middle_identity",
    (
        _identity(
            state="unknown",
            subject_id=None,
            compatibility_sid=None,
            resident_uuid=None,
        ),
        _identity(
            subject_id="resident:33333333-3333-4333-8333-333333333333",
            compatibility_sid=2,
            resident_uuid="33333333-3333-4333-8333-333333333333",
        ),
        _identity(compatibility_sid=2),
    ),
)
def test_semantic_gate_rejects_null_conflicting_or_sid_discontinuity(
    middle_identity: dict[str, object],
) -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _tracking_payload(
            frame_id=12,
            observed_at_us=1_250_000,
            sequence=2,
            provenance=False,
            identity=middle_identity,
            pose=False,
            depth=False,
            world=False,
        ),
        _tracking_payload(
            frame_id=13,
            observed_at_us=1_500_000,
            sequence=3,
            provenance=False,
        ),
    ]

    _collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert report["counts"]["identity_discontinuities"] >= 1


def test_semantic_gate_accepts_visitor_generation_continuity() -> None:
    visitor = _identity(
        state="visitor",
        subject_id=VISITOR_SUBJECT,
        compatibility_sid=1000,
        resident_uuid=None,
        visitor_generation=3,
        fresh_embedding=True,
    )
    candidate = {
        "subject_id": VISITOR_SUBJECT,
        "identity_kind": "visitor",
        "raw_similarity": 0.9,
        "hard_allowed": True,
        "hard_constraint_reason": None,
        "gallery_exemplar_count": 2,
    }
    snapshot = _snapshot(
        candidates=[candidate],
        pre_prior_winner_id=VISITOR_SUBJECT,
        final_winner_id=VISITOR_SUBJECT,
        final_outcome="visitor",
    )

    _collector, report = _evaluate(
        _tracking_payload(identity=visitor),
        snapshot=snapshot,
    )

    assert report["ok"] is True
    assert report["sample_cohort"]["identity"]["visitor_generation"] == 3


def test_semantic_gate_rejects_identity_evidence_from_another_session() -> None:
    _collector, report = _evaluate(
        _tracking_payload(),
        snapshot=_snapshot(session_id="other-session"),
    )

    assert report["ok"] is False
    assert any("does not match" in error for error in report["errors"])


def test_semantic_gate_blocks_when_occupied_scene_is_not_observed() -> None:
    _collector, report = _evaluate(
        _empty_tracking_payload(frame_id=11, observed_at_us=1_000_000)
    )

    assert report["ok"] is False
    assert report["status"] == "blocked"
    assert any("occupied scene" in error for error in report["errors"])


def test_semantic_gate_rejects_serializer_only_false_pass() -> None:
    payload = _tracking_payload()
    payload["observations"] = []

    collector, report = _evaluate(payload)

    assert any("matching canonical observation" in error for error in collector.errors)
    assert report["ok"] is False


def test_semantic_gate_rejects_partial_embedding_provenance_triad() -> None:
    payload = _tracking_payload()
    track = payload["tracks"][0]
    assert isinstance(track, dict)
    track.pop("embedding_dimension")

    collector, report = _evaluate(payload)

    assert any("partial embedding provenance" in error for error in collector.errors)
    assert report["ok"] is False


@pytest.mark.parametrize("persistence", ("queued", "dropped"))
def test_semantic_gate_accepts_live_embedding_with_explicit_pending_persistence(
    persistence: str,
) -> None:
    payload = _tracking_payload(provenance=False)
    track = payload["tracks"][0]
    assert isinstance(track, dict)
    track["embedding_present"] = True
    track["identity_observation_key"] = {
        "run_id": RUN_ID,
        "camera_id": CAMERA_ID,
        "tracker_id": str(TRACKER_ID),
        "frame_id": 11,
        "observation_id": "identity-observation-anchor",
    }
    identity = track["identity_v2"]
    assert isinstance(identity, dict)
    identity["fresh_embedding"] = True
    identity["evidence_persistence"] = persistence

    collector, _report = _evaluate(payload)

    assert not any(
        "requires persisted provenance" in error
        or "lacks persisted provenance" in error
        for error in collector.errors
    )


@pytest.mark.parametrize("surface", ("depth", "world"))
def test_semantic_gate_rejects_nonfinite_depth_or_world(surface: str) -> None:
    payload = _tracking_payload()
    track = payload["tracks"][0]
    assert isinstance(track, dict)
    if surface == "depth":
        track["depth_used_m"] = float("nan")
    else:
        track["world"] = [1.0, 0.0, float("inf")]

    _collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert report["counts"]["accepted_semantic_cohorts"] == 0


def test_semantic_gate_rejects_pipeline_errors() -> None:
    _collector, report = _evaluate(
        _tracking_payload(),
        pipeline_errors=["native bridge failed"],
    )

    assert report["ok"] is False
    assert report["checks"]["pipeline_errors_absent"] is False


def test_semantic_gate_rejects_raw_embedding_vector_fields_on_both_surfaces() -> None:
    payload = copy.deepcopy(_tracking_payload())
    track = payload["tracks"][0]
    assert isinstance(track, dict)
    track["embedding"] = [0.1, 0.2]

    collector, report = _evaluate(payload)

    assert collector.raw_vector_fields == 1
    assert report["ok"] is False
    assert report["checks"]["raw_embedding_vectors_absent"] is False
    assert gate._raw_vector_paths(report) == []
    assert gate._raw_vector_paths(collector.source_messages) == []
    assert collector.source_messages[-1][gate.SOURCE_REDACTION_MARKER] == {
        "raw_vector_field_count": 1
    }

    private_row = _evidence_row().model_dump(mode="json")
    private_row["embedding_vector"] = [0.1]
    private_payload = (json.dumps(private_row) + "\n").encode()
    with pytest.raises(ValueError, match="raw embedding vector"):
        gate._parse_evidence_snapshot_bytes(
            private_payload,
            source_path="/private/evidence.jsonl",
        )


def test_semantic_gate_rejects_tracking_frame_replay() -> None:
    payload = _tracking_payload()

    collector, report = _evaluate([payload, copy.deepcopy(payload)])

    assert report["ok"] is False
    assert any("replayed" in error for error in collector.errors)


def test_semantic_gate_rejects_source_message_bound_overflow() -> None:
    collector = gate.SemanticObservationCollector()
    collector.source_messages = [{} for _ in range(gate.MAX_SOURCE_MESSAGES)]

    collector.observe_payload(_stats())

    assert "semantic source transcript message bound exceeded" in collector.errors


def test_semantic_gate_rejects_evidence_sequence_replay() -> None:
    row = _evidence_row().model_dump(mode="json")
    payload = _canonical_bytes(row) + b"\n" + _canonical_bytes(row) + b"\n"

    with pytest.raises(ValueError, match="sequence is not contiguous"):
        gate._parse_evidence_snapshot_bytes(
            payload,
            source_path="/private/evidence.jsonl",
        )


def test_semantic_source_contract_v3_locks_policy_and_privacy() -> None:
    collector, _report = _evaluate(_tracking_payload())
    source = gate._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=ACQUISITION_FINISHED_AT_US,
        collector=collector,
    )

    assert source["schema_version"] == 3
    assert source["contract_version"] == 3
    assert source["policy"]["maximum_cohort_span_us"] == 1_500_000
    assert source["policy"]["tracker_continuity"] == (
        "contiguous_within_acquisition_window_unanchored_origin_v1"
    )
    assert source["policy"]["publication_sequence_origin"] == (
        "externally_unanchored_live_partial_capture_v1"
    )
    assert source["policy"]["lifecycle_generation_origin"] == (
        "externally_unanchored_live_partial_capture_v1"
    )
    assert source["policy"]["tombstone_last_presence"] == (
        "exact_within_window_after_unanchored_origin_v1"
    )
    assert source["acquisition_window"] == gate._acquisition_window_document(
        started_at_us=ACQUISITION_STARTED_AT_US,
        finished_at_us=ACQUISITION_FINISHED_AT_US,
    )
    assert source["privacy"] == gate.SOURCE_PRIVACY_POLICY
    assert gate._raw_vector_paths(source) == []


def test_semantic_gate_uses_actual_runtime_world_artifact_roles(
    tmp_path: Path,
) -> None:
    class CalibrationProvider:
        @staticmethod
        def snapshot(source_id: int, camera_id: str):
            return {"source_id": source_id, "camera_id": camera_id, "fx": 500.0}

    service = create_runtime_world_service(
        runtime="ds9",
        pipeline_config={"models": {}, "tracker": {}},
        camera_labels={0: CAMERA_ID},
        calibration_provider=CalibrationProvider(),
        repo_root=tmp_path,
        run_id=RUN_ID,
        instance_id=INSTANCE_ID,
        software_revision="test",
        journal_path=tmp_path / "world.sqlite3",
    )
    try:
        publication = service.publish(
            0,
            [
                {
                    "camera_id": CAMERA_ID,
                    "tracker_id": TRACKER_ID,
                    "frame_id": 11,
                    "observed_at_us": 1_000_000,
                    "captured_at_us": 1_000_000,
                    "capture_time_status": "synced",
                    "media_pts_ns": 1_000_000_000,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "image_size": [100, 100],
                }
            ],
            metadata={"camera_id": CAMERA_ID},
        )
        observation = publication.observations[0]
        assert observation.calibration.role == "camera_calibration"
        assert observation.model.role == "tracking_model_manifest"
        assert observation.config.role == "pipeline_config"
        assert gate._semantic_fingerprints(observation).calibration_sha256
    finally:
        service.close()


def test_semantic_gate_rejects_unavailable_runtime_calibration(tmp_path: Path) -> None:
    class MissingCalibrationProvider:
        @staticmethod
        def snapshot(_source_id: int, _camera_id: str):
            return None

    service = create_runtime_world_service(
        runtime="ds9",
        pipeline_config={"models": {}, "tracker": {}},
        camera_labels={0: CAMERA_ID},
        calibration_provider=MissingCalibrationProvider(),
        repo_root=tmp_path,
        run_id=RUN_ID,
        instance_id=INSTANCE_ID,
        software_revision="test",
        journal_path=tmp_path / "world-missing.sqlite3",
    )
    try:
        observation = service.publish(
            0,
            [
                {
                    "camera_id": CAMERA_ID,
                    "tracker_id": TRACKER_ID,
                    "frame_id": 11,
                    "observed_at_us": 1_000_000,
                    "captured_at_us": 1_000_000,
                    "capture_time_status": "synced",
                    "media_pts_ns": 1_000_000_000,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "image_size": [100, 100],
                }
            ],
            metadata={"camera_id": CAMERA_ID},
        ).observations[0]
        assert observation.calibration.role == "camera_calibration_unavailable"
        with pytest.raises(ValueError, match="fingerprint roles drifted"):
            gate._semantic_fingerprints(observation)
    finally:
        service.close()


@pytest.mark.parametrize(
    ("field_name", "values"),
    (
        ("embeddings", [0.1] * 256),
        ("reidFeatures", [0.1, 0.2]),
        ("appearance_descriptor", [0.1, 0.2]),
        ("opaque_payload", [float("nan")] + [0.1] * 255),
    ),
)
def test_semantic_projection_rejects_and_never_seals_embedding_vectors(
    field_name: str,
    values: list[float],
) -> None:
    payload = _tracking_payload()
    payload["tracks"][0][field_name] = values

    collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert collector.raw_vector_fields >= 1
    assert collector.source_messages[-1] == {
        "type": "tracking",
        gate.SOURCE_REDACTION_MARKER: {"raw_vector_field_count": 1},
    }
    assert gate._raw_vector_paths(collector.source_messages) == []


def test_semantic_projection_drops_arbitrary_nonsemantic_fields() -> None:
    payload = _tracking_payload()
    payload["debug_note"] = "must-not-be-sealed"
    payload["tracks"][0]["debug_scalar"] = 42

    collector, report = _evaluate(payload)

    assert report["ok"] is True
    assert "debug_note" not in collector.source_messages[-1]
    assert "debug_scalar" not in collector.source_messages[-1]["tracks"][0]


@pytest.mark.parametrize(
    ("container", "field_name", "value"),
    (
        (
            "track",
            "embedding",
            base64.b64encode(b"semantic-private-embedding" * 32).decode(),
        ),
        (
            "track",
            "embedding",
            {
                "format": "f32",
                "shape": [16, 16],
                "rows": [[0.125] * 16 for _ in range(16)],
            },
        ),
        ("track", "appearance_latent", [[0.125] * 16 for _ in range(16)]),
        ("track", "opaque_payload", {str(index): 0.125 for index in range(256)}),
        ("track", "opaque_payload", ["0.125"] * 256),
        (
            "identity_v2",
            "embedding_blob",
            base64.b64encode(b"semantic-private-embedding" * 32).decode(),
        ),
        ("identity_v2", "appearance_latent", [[0.125] * 16 for _ in range(16)]),
        (
            "identity_observation_key",
            "authorization",
            "Bearer semantic-private-token",
        ),
        ("track", "session_cookie", "semantic-private-cookie"),
        ("track", "debug_note", "Bearer semantic-private-token"),
    ),
)
def test_semantic_privacy_violations_collapse_to_marker_only(
    container: str,
    field_name: str,
    value: object,
) -> None:
    payload = copy.deepcopy(_tracking_payload())
    track = payload["tracks"][0]
    target = track if container == "track" else track[container]
    target[field_name] = value

    collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert collector.raw_vector_fields >= 1
    source_message = collector.source_messages[-1]
    assert set(source_message) == {"type", gate.SOURCE_REDACTION_MARKER}
    assert source_message["type"] == "tracking"
    marker = source_message[gate.SOURCE_REDACTION_MARKER]
    assert set(marker) == {"raw_vector_field_count"}
    assert 1 <= marker["raw_vector_field_count"] <= 8
    encoded = json.dumps(source_message, sort_keys=True)
    assert field_name not in encoded
    assert "semantic-private" not in encoded
    assert gate._raw_vector_paths(source_message) == []


def test_semantic_stats_privacy_failure_never_seals_error_material() -> None:
    collector, report = _evaluate(
        _tracking_payload(),
        pipeline_errors=["Authorization: Bearer semantic-private-token"],
    )

    assert report["ok"] is False
    assert collector.source_messages[0] == {
        "type": "stats",
        gate.SOURCE_REDACTION_MARKER: {"raw_vector_field_count": 1},
    }
    assert "semantic-private" not in json.dumps(collector.source_messages)


def test_semantic_projection_exactly_bounds_nested_identity_mappings() -> None:
    payload = copy.deepcopy(_tracking_payload())
    track = payload["tracks"][0]
    track["identity_v2"]["debug_note"] = "not-semantic-evidence"
    track["identity_observation_key"]["debug_note"] = "not-semantic-evidence"

    collector, report = _evaluate(payload)

    assert report["ok"] is True
    projected = collector.source_messages[-1]["tracks"][0]
    assert set(projected["identity_v2"]) == set(gate.IDENTITY_V2_PROJECTION_FIELDS)
    assert set(projected["identity_observation_key"]) == set(
        gate.IDENTITY_OBSERVATION_KEY_PROJECTION_FIELDS
    )
    assert "not-semantic-evidence" not in json.dumps(projected)
    assert gate._raw_vector_paths(payload) == []


def _sealed_semantic_bundle(
    directory: Path,
) -> tuple[Path, Path, Path, Path]:
    collector = gate.SemanticObservationCollector()
    collector.observe_payload(_stats())
    collector.observe_payload(_tracking_payload())
    snapshot = _snapshot()
    source = gate._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=ACQUISITION_FINISHED_AT_US,
        collector=collector,
    )
    source_raw = gate._encoded_source_transcript(source)
    source_metadata = gate._source_evidence_metadata(
        filename=gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        encoded=source_raw,
        document=source,
    )
    report = gate._evaluate(
        collector,
        snapshot,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=ACQUISITION_FINISHED_AT_US,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
        sealed_snapshot_filename=gate.CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
        source_evidence=source_metadata,
    )
    report_path = directory / gate.CANONICAL_REPORT_FILENAME
    snapshot_path = directory / gate.CANONICAL_IDENTITY_SNAPSHOT_FILENAME
    source_path = directory / gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    for path, payload in (
        (report_path, gate._encoded_report(report)),
        (snapshot_path, snapshot.payload),
        (source_path, source_raw),
    ):
        path.write_bytes(payload)
        path.chmod(0o600)
    return report_path, snapshot_path, source_path, Path(snapshot.source_path)


def _validate_semantic_bundle(
    report_path: Path,
    snapshot_path: Path,
    source_path: Path,
    expected_identity_path: Path,
) -> None:
    gate.validate_sealed_semantic_report(
        report_path,
        snapshot_path,
        source_path,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        expected_identity_evidence_path=expected_identity_path,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
    )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("report_key_order", "report is not canonical JSON"),
        ("source_whitespace", "source transcript is not canonical JSON"),
        ("snapshot_key_order", "row is not canonical JSON"),
        ("snapshot_number_spelling", "row is not canonical JSON"),
    ),
)
def test_semantic_sealed_replay_rejects_noncanonical_raw_encodings(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    report_path, snapshot_path, source_path, expected_identity_path = (
        _sealed_semantic_bundle(tmp_path)
    )
    _validate_semantic_bundle(
        report_path,
        snapshot_path,
        source_path,
        expected_identity_path,
    )

    if mutation == "report_key_order":
        report = json.loads(report_path.read_bytes())
        reordered = dict(reversed(tuple(report.items())))
        report_path.write_bytes(
            (json.dumps(reordered, indent=2, sort_keys=False) + "\n").encode()
        )
    elif mutation == "source_whitespace":
        source = json.loads(source_path.read_bytes())
        source_path.write_bytes(
            (json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    elif mutation == "snapshot_key_order":
        row = json.loads(snapshot_path.read_bytes())
        reordered = dict(reversed(tuple(row.items())))
        snapshot_path.write_bytes(
            (json.dumps(reordered, sort_keys=False, separators=(",", ":")) + "\n").encode()
        )
    else:
        raw = snapshot_path.read_bytes()
        needle = b'"calibrated_confidence":0.9'
        assert needle in raw
        snapshot_path.write_bytes(raw.replace(needle, b'"calibrated_confidence":0.900'))

    with pytest.raises(ValueError, match=expected_error):
        _validate_semantic_bundle(
            report_path,
            snapshot_path,
            source_path,
            expected_identity_path,
        )


def test_semantic_sealed_replay_rejects_nested_projection_tamper(
    tmp_path: Path,
) -> None:
    collector = gate.SemanticObservationCollector()
    collector.observe_payload(_stats())
    collector.observe_payload(_tracking_payload())
    snapshot = _snapshot()
    source = gate._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=ACQUISITION_FINISHED_AT_US,
        collector=collector,
    )
    source_raw = gate._encoded_source_transcript(source)
    source_metadata = gate._source_evidence_metadata(
        filename=gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        encoded=source_raw,
        document=source,
    )
    report = gate._evaluate(
        collector,
        snapshot,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=ACQUISITION_FINISHED_AT_US,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
        sealed_snapshot_filename=gate.CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
        source_evidence=source_metadata,
    )
    report_path = tmp_path / gate.CANONICAL_REPORT_FILENAME
    snapshot_path = tmp_path / gate.CANONICAL_IDENTITY_SNAPSHOT_FILENAME
    source_path = tmp_path / gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME

    def write_private(path: Path, payload: bytes) -> None:
        path.write_bytes(payload)
        path.chmod(0o600)

    write_private(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    write_private(snapshot_path, snapshot.payload)
    write_private(source_path, source_raw)
    expected_identity_path = Path(snapshot.source_path)
    gate.validate_sealed_semantic_report(
        report_path,
        snapshot_path,
        source_path,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        expected_identity_evidence_path=expected_identity_path,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
    )

    tampered_source = copy.deepcopy(source)
    tracking = tampered_source["messages"][-1]
    tracking["tracks"][0]["identity_v2"]["debug_note"] = "projection-tamper"
    tampered_raw = gate._encoded_source_transcript(tampered_source)
    tampered_report = copy.deepcopy(report)
    tampered_report["source_evidence"] = gate._source_evidence_metadata(
        filename=gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        encoded=tampered_raw,
        document=tampered_source,
    )
    write_private(source_path, tampered_raw)
    write_private(
        report_path,
        (json.dumps(tampered_report, indent=2, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(ValueError, match="source transcript is not canonical"):
        gate.validate_sealed_semantic_report(
            report_path,
            snapshot_path,
            source_path,
            session_id=SESSION_ID,
            runtime_lane="v3dt",
            runtime_instance_id=INSTANCE_ID,
            runtime_run_id=RUN_ID,
            expected_identity_evidence_path=expected_identity_path,
            expected_model_layer="fc_pred",
            expected_embedding_dimension=256,
        )

    acquisition_tamper = copy.deepcopy(source)
    acquisition_tamper["acquisition_window"][
        "capture_pre_window_leeway_us"
    ] += 1
    write_private(
        source_path,
        gate._encoded_source_transcript(acquisition_tamper),
    )
    write_private(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    with pytest.raises(
        ValueError,
        match="acquisition window is not canonical",
    ):
        gate.validate_sealed_semantic_report(
            report_path,
            snapshot_path,
            source_path,
            session_id=SESSION_ID,
            runtime_lane="v3dt",
            runtime_instance_id=INSTANCE_ID,
            runtime_run_id=RUN_ID,
            expected_identity_evidence_path=expected_identity_path,
            expected_model_layer="fc_pred",
            expected_embedding_dimension=256,
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing_status", "missing_registered", "raw_passthrough", "used_mismatch"),
)
def test_semantic_gate_requires_coherent_registered_depth(mutation: str) -> None:
    payload = _tracking_payload()
    track = payload["tracks"][0]
    if mutation == "missing_status":
        track.pop("depth_registration_status")
    elif mutation == "missing_registered":
        track.pop("depth_registered_m")
    elif mutation == "raw_passthrough":
        track["depth_registration_status"] = "raw_passthrough"
    else:
        track["depth_used_m"] = 3.5

    _collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert report["checks"]["usable_depth_component_observed"] is False


def test_semantic_gate_rejects_duplicate_public_track_association() -> None:
    payload = _tracking_payload()
    payload["tracks"].append(copy.deepcopy(payload["tracks"][0]))
    payload["track_count"] = 2

    collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert any("duplicate public track association" in item for item in collector.errors)
    assert report["checks"]["one_to_one_track_observation_association"] is False


def test_semantic_gate_rejects_capture_span_beyond_observed_span() -> None:
    anchor = _tracking_payload(
        frame_id=11,
        observed_at_us=3_000_000,
        captured_at_us=3_000_000,
        sequence=1,
        pose=False,
        depth=False,
        world=False,
    )
    component = _tracking_payload(
        frame_id=12,
        observed_at_us=4_000_000,
        captured_at_us=1,
        media_pts_ns=1_000,
        sequence=2,
        provenance=False,
    )

    _collector, report = _evaluate(
        [anchor, component],
        snapshot=_snapshot(observed_at_us=3_000_000),
    )

    assert report["ok"] is False
    assert report["counts"]["accepted_semantic_cohorts"] == 0


def test_semantic_gate_rejects_missing_publication_sequence() -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _tracking_payload(
            frame_id=13,
            observed_at_us=1_500_000,
            sequence=2,
            publication_sequence=2,
            provenance=False,
        ),
    ]

    collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert any("source sequence is discontinuous" in item for item in collector.errors)


def test_semantic_gate_rejects_lifecycle_change_without_tombstone() -> None:
    rows = [
        _tracking_payload(pose=False, depth=False, world=False),
        _tracking_payload(
            frame_id=12,
            observed_at_us=1_500_000,
            sequence=2,
            lifecycle_generation=2,
            provenance=False,
        ),
    ]

    collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert any("tombstones do not exactly cover" in item for item in collector.errors)


def test_semantic_gate_accepts_exact_capture_pre_window_leeway_endpoint() -> None:
    acquisition_started = 10_000_000
    observed = acquisition_started
    captured = acquisition_started - gate.SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US

    _collector, report = _evaluate(
        _tracking_payload(
            observed_at_us=observed,
            captured_at_us=captured,
        ),
        snapshot=_snapshot(observed_at_us=observed),
        acquisition_started_at_us=acquisition_started,
        acquisition_finished_at_us=acquisition_started + 1_000_000,
    )

    assert report["ok"] is True
    assert report["checks"]["acquisition_window_enforced"] is True
    assert report["counts"]["acquisition_window_violations"] == 0


@pytest.mark.parametrize(
    ("observed_at_us", "captured_at_us"),
    (
        (10_000_000, 7_999_999),
        (9_999_999, 9_999_999),
        (11_000_001, 11_000_001),
    ),
)
def test_semantic_gate_rejects_samples_outside_exact_acquisition_window(
    observed_at_us: int,
    captured_at_us: int,
) -> None:
    _collector, report = _evaluate(
        _tracking_payload(
            observed_at_us=observed_at_us,
            captured_at_us=captured_at_us,
        ),
        snapshot=_snapshot(observed_at_us=observed_at_us),
        acquisition_started_at_us=10_000_000,
        acquisition_finished_at_us=11_000_000,
    )

    assert report["ok"] is False
    assert report["counts"]["acquisition_window_violations"] == 1
    assert report["counts"]["accepted_semantic_cohorts"] == 0
    assert any("escapes acquisition window" in item for item in report["errors"])


def test_semantic_gate_rejects_empty_tracking_frame_outside_acquisition() -> None:
    anchor = _tracking_payload(
        observed_at_us=10_500_000,
        captured_at_us=10_500_000,
    )
    empty = _empty_tracking_payload(
        frame_id=1,
        observed_at_us=11_000_001,
        publication_sequence=0,
    )
    empty["source_id"] = 1

    collector, report = _evaluate(
        [anchor, empty],
        snapshot=_snapshot(observed_at_us=10_500_000),
        acquisition_started_at_us=10_000_000,
        acquisition_finished_at_us=11_000_000,
    )

    assert collector.errors == []
    assert report["ok"] is False
    assert report["counts"]["acquisition_window_violations"] == 1
    assert report["source_evidence"]["first_observed_at_us"] == 10_500_000
    assert report["source_evidence"]["last_observed_at_us"] == 11_000_001
    assert report["source_evidence"]["last_captured_at_us"] == 11_000_001


def test_semantic_gate_rejects_future_canonical_publication_clock() -> None:
    payload = _tracking_payload()
    payload["observations"][0]["published_at_us"] = 10**18

    _collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert report["counts"]["publication_clock_violations"] == 1
    assert report["checks"][
        "publication_clock_within_acquisition_window"
    ] is False
    assert any("publication clock escapes" in item for item in report["errors"])


def test_semantic_gate_counts_unmatched_canonical_publication_clock() -> None:
    payload = _tracking_payload()
    unmatched = copy.deepcopy(payload["observations"][0])
    unmatched["observation_id"] = f"{RUN_ID}:0:99:999"
    unmatched["sequence"] = 99
    unmatched["published_at_us"] = 10**18
    unmatched["payload"]["tracklet"]["tracker_id"] = 999
    payload["observations"].append(unmatched)

    collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert collector.canonical_observations == 2
    assert report["counts"]["publication_clock_violations"] == 1
    assert report["checks"][
        "publication_clock_within_acquisition_window"
    ] is False


def test_semantic_json_inputs_reject_duplicate_object_keys_before_projection() -> None:
    duplicate_embedding = (
        '{"type":"tracking","embedding":[0.1],"embedding":null}'
    )
    with pytest.raises(gate.StrictJSONError) as duplicate_public:
        gate.strict_json_loads(
            duplicate_embedding,
            label="duplicate public telemetry",
        )
    assert duplicate_public.value.reason == "duplicate_key"
    with pytest.raises(gate.StrictJSONError) as duplicate_report:
        gate._strict_json_object(
            b'{"ok":true,"ok":false}',
            label="duplicate semantic report",
        )
    assert duplicate_report.value.reason == "duplicate_key"

    collector = gate.SemanticObservationCollector()
    collector.observe_source_rejection("duplicate_json_key")
    source = gate._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="baseline",
        runtime_instance_id=INSTANCE_ID,
        runtime_run_id=RUN_ID,
        acquisition_started_at_us=ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=ACQUISITION_FINISHED_AT_US,
        collector=collector,
    )
    assert source["messages"] == [
        {
            "type": "semantic_source_rejection",
            "reason": "duplicate_json_key",
        }
    ]
    assert "duplicate_json_key" in gate._encoded_source_transcript(source).decode()


def test_identity_snapshot_rejects_duplicate_keys_before_model_validation() -> None:
    raw = json.dumps(_evidence_row().model_dump(mode="json"), sort_keys=True)
    duplicate = raw.replace(
        '"sequence": 7',
        '"sequence": 7, "sequence": 7',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        gate._parse_evidence_snapshot_bytes(
            (duplicate + "\n").encode(),
            source_path="/private/evidence.jsonl",
        )


def test_semantic_gate_qualifies_unanchored_partial_capture_origins() -> None:
    collector, report = _evaluate(
        _tracking_payload(
            publication_sequence=999,
            lifecycle_generation=99,
        )
    )

    assert report["ok"] is True
    assert collector.errors == []
    assert report["checks"][
        "tracking_publication_contiguous_within_window_unanchored_origin"
    ] is True
    assert report["checks"][
        "tracker_lifecycle_contiguous_within_window_unanchored_origin"
    ] is True


def test_semantic_gate_accepts_first_frame_unanchored_origin_tombstone() -> None:
    payload = _tracking_payload()
    payload["tracker_lifecycle_tombstones"] = [
        {
            "camera_id": CAMERA_ID,
            "tracker_id": TRACKER_ID + 100,
            "tracker_lifecycle_generation": 9,
            "last_seen_frame_id": 10,
            "last_seen_observed_at_us": 900_000,
            "disappeared_at_frame_id": 11,
            "disappeared_at_observed_at_us": 1_000_000,
        }
    ]

    collector, report = _evaluate(payload)

    assert collector.errors == []
    assert report["ok"] is True
    assert report["counts"]["unanchored_origin_tombstones"] == 1
    assert report["checks"][
        "tombstone_last_published_presence_exact_after_unanchored_origin"
    ] is True


def test_semantic_gate_does_not_count_simultaneously_present_origin_tombstone() -> None:
    payload = _tracking_payload()
    payload["tracker_lifecycle_tombstones"] = [
        {
            "camera_id": CAMERA_ID,
            "tracker_id": TRACKER_ID,
            "tracker_lifecycle_generation": 1,
            "last_seen_frame_id": 10,
            "last_seen_observed_at_us": 900_000,
            "disappeared_at_frame_id": 11,
            "disappeared_at_observed_at_us": 1_000_000,
        }
    ]

    collector, report = _evaluate(payload)

    assert report["ok"] is False
    assert report["counts"]["unanchored_origin_tombstones"] == 0
    assert any("simultaneously present" in item for item in collector.errors)


def test_semantic_gate_rejects_later_unanchored_tombstone() -> None:
    unanchored_tombstone = {
        "camera_id": CAMERA_ID,
        "tracker_id": TRACKER_ID + 100,
        "tracker_lifecycle_generation": 9,
        "last_seen_frame_id": 11,
        "last_seen_observed_at_us": 1_000_000,
        "disappeared_at_frame_id": 12,
        "disappeared_at_observed_at_us": 1_250_000,
    }

    collector, report = _evaluate(
        [
            _tracking_payload(),
            _empty_tracking_payload(
                frame_id=12,
                observed_at_us=1_250_000,
                publication_sequence=1,
                tombstones=[unanchored_tombstone],
            ),
        ]
    )

    assert report["ok"] is False
    assert report["counts"]["unanchored_origin_tombstones"] == 0
    assert any(
        "not previously published in-window" in item
        for item in collector.errors
    )


def test_semantic_gate_requires_exact_tombstone_last_published_presence() -> None:
    forged_tombstone = {
        "camera_id": CAMERA_ID,
        "tracker_id": TRACKER_ID,
        "tracker_lifecycle_generation": 1,
        "last_seen_frame_id": 0,
        "last_seen_observed_at_us": 1,
        "disappeared_at_frame_id": 12,
        "disappeared_at_observed_at_us": 1_250_000,
    }
    rows = [
        _tracking_payload(),
        _empty_tracking_payload(
            frame_id=12,
            observed_at_us=1_250_000,
            publication_sequence=1,
            tombstones=[forged_tombstone],
        ),
    ]

    collector, report = _evaluate(rows)

    assert report["ok"] is False
    assert any(
        "does not match actual published presence" in item
        for item in collector.errors
    )
    assert report["checks"][
        "tombstone_last_published_presence_exact_after_unanchored_origin"
    ] is False


def test_semantic_gate_accepts_exact_tombstone_last_published_presence() -> None:
    exact_tombstone = {
        "camera_id": CAMERA_ID,
        "tracker_id": TRACKER_ID,
        "tracker_lifecycle_generation": 1,
        "last_seen_frame_id": 11,
        "last_seen_observed_at_us": 1_000_000,
        "disappeared_at_frame_id": 12,
        "disappeared_at_observed_at_us": 1_250_000,
    }

    collector, report = _evaluate(
        [
            _tracking_payload(),
            _empty_tracking_payload(
                frame_id=12,
                observed_at_us=1_250_000,
                publication_sequence=1,
                tombstones=[exact_tombstone],
            ),
        ]
    )

    assert collector.errors == []
    assert report["ok"] is True
    assert report["checks"][
        "tombstone_last_published_presence_exact_after_unanchored_origin"
    ] is True
