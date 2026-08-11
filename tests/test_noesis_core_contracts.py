from __future__ import annotations

from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from noesis_core.contracts.actions import (
    ActionApproval,
    ActionStatus,
    ActorRef,
    ActorRole,
    ExecutionReceipt,
)
from noesis_core.contracts.base import ArtifactFingerprint, Matrix3, ProducerRef, Vector3
from noesis_core.contracts.compat import ContractCompatibilityError, require_supported_contract
from noesis_core.contracts.identity import (
    IdentityDecision,
    IdentityEvidenceSummary,
    IdentityKind,
    IdentityOutcome,
    SubjectRef,
    TrackletRef,
)
from noesis_core.contracts.observation import (
    ObservationEnvelope,
    PersonObservation,
    WorldObservationDiagnostics,
    WorldPositionObservation,
)
from noesis_core.contracts.scene import SceneArtifact, SceneCameraRevision, SceneRelease
from noesis_core.scene_files import MAX_SCENE_ARTIFACT_BYTES
from scripts.export_noesis_core_schemas import export_schemas, export_typescript


SHA_A = "a" * 64
SHA_B = "b" * 64


def _producer() -> ProducerRef:
    return ProducerRef(runtime="ds8", instance_id="appliance", run_id="run-1", software_revision="abc123")


def _tracklet() -> TrackletRef:
    return TrackletRef(
        run_id="run-1",
        camera_id="kitchen",
        source_id=0,
        tracker_id=7,
        frame_id=11,
        observed_at_us=200,
    )


def _fingerprint(role: str, sha: str = SHA_A) -> ArtifactFingerprint:
    return ArtifactFingerprint(role=role, sha256=sha, version="v1")


def test_observation_envelope_rejects_time_travel() -> None:
    payload = PersonObservation(
        tracklet=_tracklet(),
        bbox_xywh=(10.0, 20.0, 30.0, 40.0),
        image_size=(1920, 1080),
        world=WorldPositionObservation(
            position=Vector3(x=1.0, y=0.0, z=2.0),
            covariance=Matrix3(values=(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2)),
            frame="backend_world_m",
            units="meters",
            source="pose_depth_fused",
            quality="good",
            confidence=0.9,
        ),
    )
    with pytest.raises(ValidationError, match="captured_at_us"):
        ObservationEnvelope(
            contract="noesis.observation.person",
            contract_version=1,
            observation_id="obs-1",
            producer=_producer(),
            sequence=1,
            captured_at_us=300,
            observed_at_us=200,
            published_at_us=400,
            capture_time_status="synced",
            coordinate_frame="backend_world_m",
            units="meters",
            calibration=_fingerprint("calibration"),
            model=_fingerprint("detector"),
            config=_fingerprint("config"),
            payload=payload,
        )


def test_observation_can_declare_capture_time_unavailable_without_guessing() -> None:
    payload = PersonObservation(
        tracklet=_tracklet(),
        bbox_xywh=(10.0, 20.0, 30.0, 40.0),
        image_size=(1920, 1080),
    )
    envelope = ObservationEnvelope(
        contract="noesis.observation.person",
        contract_version=1,
        observation_id="obs-2",
        producer=_producer(),
        sequence=2,
        captured_at_us=None,
        observed_at_us=200,
        published_at_us=201,
        capture_time_status="unavailable",
        media_pts_ns=1234,
        coordinate_frame="image_px",
        units="pixels",
        calibration=_fingerprint("calibration"),
        model=_fingerprint("detector"),
        config=_fingerprint("config"),
        payload=payload,
    )
    assert envelope.captured_at_us is None
    assert envelope.media_pts_ns == 1234


@pytest.mark.parametrize(
    "partial",
    [
        {"embedding_sequence": 0},
        {"embedding_model_sha256": SHA_A},
        {"embedding_dimension": 256},
        {"embedding_sequence": 0, "embedding_model_sha256": SHA_A},
    ],
)
def test_person_observation_rejects_partial_embedding_provenance(
    partial: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="embedding provenance requires"):
        PersonObservation(
            tracklet=_tracklet(),
            bbox_xywh=(10.0, 20.0, 30.0, 40.0),
            image_size=(1920, 1080),
            **partial,
        )


def test_person_observation_accepts_complete_embedding_provenance() -> None:
    observation = PersonObservation(
        tracklet=_tracklet(),
        bbox_xywh=(10.0, 20.0, 30.0, 40.0),
        image_size=(1920, 1080),
        embedding_sequence=0,
        embedding_model_sha256=SHA_A,
        embedding_dimension=256,
    )

    assert observation.embedding_sequence == 0
    assert observation.embedding_model_sha256 == SHA_A
    assert observation.embedding_dimension == 256


def test_person_observation_requires_explicit_spatial_zone_authority() -> None:
    authoritative = PersonObservation(
        tracklet=_tracklet(),
        bbox_xywh=(10.0, 20.0, 30.0, 40.0),
        image_size=(1920, 1080),
        zone="Kitchen",
        zone_source="nvdsanalytics_roi",
        zone_authoritative=True,
    )
    camera_default = PersonObservation(
        tracklet=_tracklet(),
        bbox_xywh=(10.0, 20.0, 30.0, 40.0),
        image_size=(1920, 1080),
        zone="Kitchen",
        zone_source="camera_default",
        zone_authoritative=False,
    )

    assert authoritative.zone_authoritative is True
    assert camera_default.zone_authoritative is False

    with pytest.raises(
        ValidationError,
        match="camera-default zones cannot be spatially authoritative",
    ):
        PersonObservation(
            tracklet=_tracklet(),
            bbox_xywh=(10.0, 20.0, 30.0, 40.0),
            image_size=(1920, 1080),
            zone="Kitchen",
            zone_source="camera_default",
            zone_authoritative=True,
        )


@pytest.mark.parametrize("zone", ["", " Kitchen", "Kitchen ", "x" * 161])
def test_person_observation_rejects_noncanonical_zone_labels(zone: str) -> None:
    with pytest.raises(ValidationError):
        PersonObservation(
            tracklet=_tracklet(),
            bbox_xywh=(10.0, 20.0, 30.0, 40.0),
            image_size=(1920, 1080),
            zone=zone,
            zone_source="nvdsanalytics_roi",
            zone_authoritative=True,
        )


def test_invalid_world_diagnostics_require_explicit_first_divergence() -> None:
    with pytest.raises(
        ValidationError,
        match="invalid world diagnostics require first_divergence_reason",
    ):
        PersonObservation(
            tracklet=_tracklet(),
            bbox_xywh=(10.0, 20.0, 30.0, 40.0),
            image_size=(1920, 1080),
            world_diagnostics=WorldObservationDiagnostics(
                floor_candidate_m=Vector3(x=1.0, y=0.0, z=2.0),
            ),
        )


def test_world_diagnostics_reject_nonfinite_raw_candidates() -> None:
    with pytest.raises(
        ValidationError,
        match="world diagnostic candidate points must be finite",
    ):
        WorldObservationDiagnostics(
            first_divergence_reason="invalid_candidate",
            floor_candidate_m=Vector3(x=float("nan"), y=0.0, z=2.0),
        )


def test_unknown_identity_never_exposes_subject_or_prior_acceptance() -> None:
    evidence = IdentityEvidenceSummary(
        embedding_present=True,
        appearance_similarity=0.61,
        ambiguity_margin=0.01,
        prior_contribution=0.02,
        independent_observation_count=2,
        quality_accepted=True,
    )
    decision = IdentityDecision(
        contract="noesis.identity.decision",
        contract_version=1,
        decision_id="decision-1",
        tracklets=(_tracklet(),),
        outcome=IdentityOutcome.UNKNOWN,
        subject=None,
        decided_at_us=250,
        evidence=evidence,
        absolute_open_set_floor=0.72,
        reject_reason="below_open_set_floor",
        prior_changed_winner=False,
    )
    assert decision.subject is None
    with pytest.raises(ValidationError, match="cannot turn rejection"):
        decision.model_copy(update={"prior_changed_winner": True}, deep=True).__class__.model_validate(
            {**decision.model_dump(), "prior_changed_winner": True}
        )


def test_resident_acceptance_cannot_fall_below_absolute_floor() -> None:
    subject = SubjectRef(
        subject_id="resident:abc",
        kind=IdentityKind.RESIDENT,
        resident_uuid="abc",
        display_name="Resident",
        stable_id=1,
    )
    evidence = IdentityEvidenceSummary(
        embedding_present=True,
        appearance_similarity=0.70,
        ambiguity_margin=0.08,
        prior_contribution=0.02,
        independent_observation_count=3,
        quality_accepted=True,
    )
    with pytest.raises(ValidationError, match="below the absolute"):
        IdentityDecision(
            contract="noesis.identity.decision",
            contract_version=1,
            decision_id="decision-2",
            tracklets=(_tracklet(),),
            outcome=IdentityOutcome.RESIDENT,
            subject=subject,
            decided_at_us=250,
            evidence=evidence,
            absolute_open_set_floor=0.72,
            required_similarity=0.72,
            prior_changed_winner=True,
        )


def test_success_receipt_requires_provider_readback() -> None:
    with pytest.raises(ValidationError, match="provider readback"):
        ExecutionReceipt(
            contract="noesis.action.receipt",
            contract_version=1,
            receipt_id="receipt-1",
            action_id="action-1",
            status=ActionStatus.SUCCEEDED,
            started_at_us=100,
            completed_at_us=200,
            provider="homeseer",
        )


def test_agent_cannot_approve_its_own_action() -> None:
    with pytest.raises(ValidationError, match="may not approve"):
        ActionApproval(
            contract="noesis.action.approval",
            contract_version=1,
            approval_id="approval-1",
            action_id="action-1",
            approver=ActorRef(actor_id="agent", role=ActorRole.AGENT),
            issued_at_us=100,
            expires_at_us=200,
            decision="approve",
        )


def test_scene_release_rejects_mixed_calibration_cohort() -> None:
    camera = SceneCameraRevision(
        camera_id="kitchen",
        revision_id="rev-1",
        captured_at_us=100,
        calibration_bundle_sha256=SHA_B,
        calibration_sha256=SHA_B,
        model_bundle_sha256=SHA_A,
        model_sha256=SHA_A,
        manifest_sha256=SHA_A,
        artifact_path="kitchen/rev-1",
        artifacts=(
            SceneArtifact(
                role="surface",
                relative_path="surface.glb",
                sha256=SHA_A,
                size_bytes=10,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="calibration bundle"):
        SceneRelease(
            contract="noesis.scene.release",
            contract_version=1,
            release_id="release-1",
            release_version=1,
            created_at_us=200,
            created_by="owner",
            calibration=_fingerprint("calibration", SHA_A),
            model=_fingerprint("model", SHA_A),
            config=_fingerprint("config", SHA_A),
            authored_scene=_fingerprint("authored_scene", SHA_A),
            authored_scene_path="releases/release-1/home.obj",
            authored_scene_size_bytes=10,
            authored_scene_dependencies=(),
            cohort_max_delta_us=100,
            cameras=(camera,),
            validation_report_sha256=SHA_A,
            validation_report_path="releases/release-1/validation.json",
            validation_report_size_bytes=10,
        )


def test_scene_release_rejects_cross_release_authored_dependencies() -> None:
    camera = SceneCameraRevision(
        camera_id="kitchen",
        revision_id="rev-1",
        captured_at_us=100,
        calibration_bundle_sha256=SHA_A,
        calibration_sha256=SHA_A,
        model_bundle_sha256=SHA_A,
        model_sha256=SHA_A,
        manifest_sha256=SHA_A,
        artifact_path="kitchen/rev-1",
        artifacts=(
            SceneArtifact(
                role="surface",
                relative_path="surface.glb",
                sha256=SHA_A,
                size_bytes=10,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="owned by the release directory"):
        SceneRelease(
            contract="noesis.scene.release",
            contract_version=1,
            release_id="release-1",
            release_version=1,
            created_at_us=200,
            created_by="owner",
            calibration=_fingerprint("calibration", SHA_A),
            model=_fingerprint("model", SHA_A),
            config=_fingerprint("config", SHA_A),
            authored_scene=_fingerprint("authored_scene", SHA_A),
            authored_scene_path="releases/release-1/home.obj",
            authored_scene_size_bytes=10,
            authored_scene_dependencies=(
                SceneArtifact(
                    role="material_library_000",
                    relative_path="releases/release-2/home.mtl",
                    sha256=SHA_A,
                    size_bytes=10,
                ),
            ),
            cohort_max_delta_us=100,
            cameras=(camera,),
            validation_report_sha256=SHA_A,
            validation_report_path="releases/release-1/validation.json",
            validation_report_size_bytes=10,
        )


def test_scene_release_rejects_empty_immutable_files() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SceneArtifact(
            role="surface",
            relative_path="surface.glb",
            sha256=SHA_A,
            size_bytes=0,
        )

    camera = SceneCameraRevision(
        camera_id="kitchen",
        revision_id="rev-1",
        captured_at_us=100,
        calibration_bundle_sha256=SHA_A,
        calibration_sha256=SHA_A,
        model_bundle_sha256=SHA_A,
        model_sha256=SHA_A,
        manifest_sha256=SHA_A,
        artifact_path="kitchen/rev-1",
        artifacts=(
            SceneArtifact(
                role="surface",
                relative_path="surface.glb",
                sha256=SHA_A,
                size_bytes=10,
            ),
        ),
    )
    valid = SceneRelease(
        contract="noesis.scene.release",
        contract_version=1,
        release_id="release-1",
        release_version=1,
        created_at_us=200,
        created_by="owner",
        calibration=_fingerprint("calibration", SHA_A),
        model=_fingerprint("model", SHA_A),
        config=_fingerprint("config", SHA_A),
        authored_scene=_fingerprint("authored_scene", SHA_A),
        authored_scene_path="releases/release-1/home.obj",
        authored_scene_size_bytes=10,
        authored_scene_dependencies=(),
        cohort_max_delta_us=100,
        cameras=(camera,),
        validation_report_sha256=SHA_A,
        validation_report_path="releases/release-1/validation.json",
        validation_report_size_bytes=10,
    )
    for field in ("authored_scene_size_bytes", "validation_report_size_bytes"):
        payload = valid.model_dump(mode="json")
        payload[field] = 0
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SceneRelease.model_validate(payload)


def test_scene_release_rejects_total_file_and_byte_bombs() -> None:
    def camera(camera_index: int, artifact_count: int, size_bytes: int = 1):
        return SceneCameraRevision(
            camera_id=f"camera-{camera_index}",
            revision_id=f"revision-{camera_index}",
            captured_at_us=100,
            calibration_bundle_sha256=SHA_A,
            calibration_sha256=SHA_A,
            model_bundle_sha256=SHA_A,
            model_sha256=SHA_A,
            manifest_sha256=SHA_A,
            artifact_path=f"revision-{camera_index}",
            artifacts=tuple(
                SceneArtifact(
                    role=f"artifact_{artifact_index:03d}",
                    relative_path=f"artifact-{artifact_index:03d}.bin",
                    sha256=SHA_A,
                    size_bytes=size_bytes,
                )
                for artifact_index in range(artifact_count)
            ),
        )

    common = {
        "contract": "noesis.scene.release",
        "contract_version": 1,
        "release_id": "release-1",
        "release_version": 1,
        "created_at_us": 200,
        "created_by": "owner",
        "calibration": _fingerprint("calibration", SHA_A),
        "model": _fingerprint("model", SHA_A),
        "config": _fingerprint("config", SHA_A),
        "authored_scene": _fingerprint("authored_scene", SHA_A),
        "authored_scene_path": "releases/release-1/home.obj",
        "authored_scene_size_bytes": 1,
        "authored_scene_dependencies": (),
        "cohort_max_delta_us": 100,
        "validation_report_sha256": SHA_A,
        "validation_report_path": "releases/release-1/validation.json",
        "validation_report_size_bytes": 1,
    }
    with pytest.raises(ValidationError, match="512-file limit"):
        SceneRelease(
            **common,
            cameras=tuple(camera(index, 127) for index in range(4)),
        )
    with pytest.raises(ValidationError, match="1073741824-byte limit"):
        SceneRelease(
            **common,
            cameras=(camera(0, 4, MAX_SCENE_ARTIFACT_BYTES),),
        )


def test_contract_compatibility_rejects_unknown_version() -> None:
    with pytest.raises(ContractCompatibilityError):
        require_supported_contract(
            {"contract": "noesis.world.snapshot", "contract_version": 2},
            contract="noesis.world.snapshot",
            supported_versions=(1,),
        )


def test_schema_export_is_deterministic(tmp_path: Path) -> None:
    assert export_schemas(tmp_path, check=False) == []
    assert export_schemas(tmp_path, check=True) == []
    assert (tmp_path / "observation_envelope.schema.json").exists()

    typescript_path = tmp_path / "noesis-contracts.ts"
    assert export_typescript(typescript_path, check=False) == []
    assert export_typescript(typescript_path, check=True) == []
    typescript = typescript_path.read_text(encoding="utf-8")
    assert "export type ObservationEnvelope" in typescript
    assert "export type IdentityDecision" in typescript
    assert "readonly contract_version: 1" in typescript


def test_versioned_fixture_validates_against_runtime_model() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "contracts" / "fixtures" / "v1" / "capability_health.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    from noesis_core.contracts.health import CapabilityHealth

    validated = CapabilityHealth.model_validate(payload)
    assert validated.contract_version == 1
    assert validated.capabilities[0].capability == "tracking"
