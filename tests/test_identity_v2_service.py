from __future__ import annotations

import hashlib
import json
import os

import pytest
import yaml

import noesis.identity_v2_service as identity_service_module
from noesis.identity_v2_service import (
    IdentityFramePrimitive,
    IdentityV2ConfigurationError,
    create_identity_v2_service,
    identity_authority_runtime_profile_sha256,
    identity_model_semantic_profile_sha256,
)
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.contracts.identity_calibration import (
    one_sided_binomial_upper_confidence_bound,
)
from noesis_core.world import GlobalWorldFusion
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts


@pytest.fixture(autouse=True)
def _gpu_free_loaded_reid_transform(monkeypatch):
    """Model an exact loaded bridge without requiring DeepStream in unit CI."""

    monkeypatch.setattr(
        identity_service_module,
        "_loaded_reid_transform_component",
        lambda: {
            "module": "noesis_reid_meta_ext",
            "status": "loaded",
            "basename": "noesis_reid_meta_ext.test.so",
            "sha256": "f" * 64,
        },
    )


def _assets(tmp_path, *, cameras=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = tmp_path / "reid.engine"
    engine.write_bytes(b"actual-tensorrt-engine-bytes")
    topology = tmp_path / "camera_topology.yaml"
    camera_rows = cameras or {"camera-a": 0, "camera-b": 1}
    topology.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "cameras": {
                    name: {"source_id": source} for name, source in camera_rows.items()
                },
                "overlaps": (
                    [
                        {
                            "cameras": ["camera-a", "camera-b"],
                            "max_world_dist_m": 1.0,
                            "max_time_delta_s": 0.5,
                            "require_appearance_sim": 0.8,
                            "enabled": True,
                        }
                    ]
                    if set(camera_rows) == {"camera-a", "camera-b"}
                    else []
                ),
                "overlap_allow_appearance_only": False,
            }
        ),
        encoding="utf-8",
    )
    nvinfer = tmp_path / "reid.ini"
    nvinfer.write_text(
        "\n".join(
            (
                "[property]",
                f"model-engine-file={engine.name}",
                "process-mode=2",
                "network-type=1",
                "model-color-format=0",
                "net-scale-factor=0.01735207",
                "offsets=123.675;116.28;103.53",
                "infer-dims=3;256;128",
                "maintain-aspect-ratio=0",
                "operate-on-gie-id=1",
                "operate-on-class-ids=0",
                "output-tensor-meta=1",
                "classifier-async-mode=0",
                "",
            )
        ),
        encoding="utf-8",
    )
    config = {
        "models": {
            "reid": {
                "enable": True,
                "engine": str(engine),
                "config-file-path": str(nvinfer),
                "layer": "features",
                "embedding_dim": 4,
            }
        }
    }
    pipeline_yaml = tmp_path / "infer.yaml"
    pipeline_yaml.write_text("models: {}\n", encoding="utf-8")
    return engine, topology, config, pipeline_yaml


def _calibration_metrics(known, unknown):
    return {
        "confidence_level": 0.95,
        "confidence_method": "clopper_pearson_one_sided",
        "confidence_unit": "truth_person_worst_case",
        "observation_count": known + unknown,
        "known_observation_count": known,
        "unknown_observation_count": unknown,
        "challenge_eligible_observation_count": known + unknown,
        "evidence_unit_count": known + unknown,
        "known_evidence_unit_count": known,
        "unknown_evidence_unit_count": unknown,
        "encounter_count": known + unknown,
        "known_encounter_count": known,
        "unknown_encounter_count": unknown,
        "challenge_eligible_encounter_count": known + unknown,
        "correct_accept_encounter_count": known,
        "false_reject_encounter_count": 0,
        "misidentification_encounter_count": 0,
        "false_accept_encounter_count": 0,
        "person_count": known + unknown,
        "known_person_count": known,
        "unknown_person_count": unknown,
        "known_challenge_person_count": known,
        "unknown_challenge_person_count": unknown,
        "correct_accept_person_count": known,
        "false_reject_person_count": 0,
        "misidentification_person_count": 0,
        "false_accept_person_count": 0,
        "misidentification_challenge_person_count": 0,
        "false_accept_challenge_person_count": 0,
        "person_far": 0.0,
        "person_frr": 0.0,
        "person_misidentification_rate": 0.0,
        "person_identification_accuracy": 1.0,
        "far": 0.0,
        "far_upper_confidence_bound": one_sided_binomial_upper_confidence_bound(
            0, unknown
        ),
        "frr": 0.0,
        "frr_upper_confidence_bound": one_sided_binomial_upper_confidence_bound(
            0, known
        ),
        "misidentification_rate": 0.0,
        "misidentification_upper_confidence_bound": (
            one_sided_binomial_upper_confidence_bound(0, known)
        ),
        "unknown_rejection_rate": 1.0,
        "identification_accuracy": 1.0,
        "resident_prior_changed_encounter_count": 0,
        "resident_prior_beneficial_encounter_count": 0,
        "resident_prior_harmful_encounter_count": 0,
        "resident_prior_rejection_rescue_encounter_count": 0,
    }


def _calibration_artifact(engine, *, layer, dimension, semantic_profile_sha256):
    unit_policy = {
        "algorithm": "session_tracklet_time_bucket_v1",
        "bucket_duration_us": 5_000_000,
        "maximum_observations_per_unit": 300,
        "fit_thinning": "temporal_cover_center_representative_v1",
        "maximum_fit_units_per_encounter": 8,
        "fit_representatives_per_unit": 1,
        "authority_aggregation": "encounter_worst_case_v1",
    }
    provenance = {
        "source_name": "service-test",
        "source_revision": "v2",
        "source_manifest_sha256": "4" * 64,
        "model_semantic_profile_sha256": semantic_profile_sha256,
        "labeling_protocol_revision": "v2",
        "population_basis": "licensed_real_people",
    }
    policy = {
        "appearance_floor": 0.70,
        "quality_floor": 0.50,
        "calibrated_confidence_floor": 0.55,
        "ambiguity_margin_floor": 0.025,
        "calibration_slope": 12.0,
        "calibration_midpoint": 0.70,
        "resident_prior_bonus": 0.025,
        "resident_prior_cap": 0.05,
        "unknown_utility": 0.0,
    }
    return {
        "contract": "noesis.identity.open_set_calibration",
        "contract_version": 2,
        "generated_at_us": 1,
        "generator": "scripts/identity_v2_calibrate.py",
        "generator_revision": "test-revision",
        "model_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "model_layer": layer,
        "embedding_dim": dimension,
        "model_semantic_profile_sha256": semantic_profile_sha256,
        "authority_scope": "open_set_scorer_policy_only",
        "gallery_limits": {
            "maximum_resident_candidate_count": 3,
            "maximum_visitor_candidate_count": 1,
            "maximum_total_candidate_count": 4,
            "maximum_exemplars_per_candidate": 4,
        },
        "evidence_unit_policy": unit_policy,
        "benchmark_dataset": {
            "evidence_stratum": "benchmark",
            "dataset_sha256": "1" * 64,
            "evidence_sha256": "2" * 64,
            "labels_sha256": "3" * 64,
            "split_strategy": "subject_disjoint",
            "evidence_sources": ["replay"],
            "runtimes": ["replay"],
            "runtime_modes": ["shadow"],
            "truth_kinds": ["resident", "unknown"],
            "provenance": provenance,
            "observation_count": 700,
            "evidence_unit_count": 700,
            "encounter_count": 700,
        },
        "household_dataset": {
            "evidence_stratum": "household",
            "dataset_sha256": "5" * 64,
            "evidence_sha256": "6" * 64,
            "labels_sha256": "7" * 64,
            "split_strategy": "session_disjoint",
            "evidence_sources": ["shadow"],
            "runtimes": ["test"],
            "runtime_modes": ["shadow"],
            "truth_kinds": ["resident", "unknown"],
            "provenance": {
                **provenance,
                "source_manifest_sha256": "8" * 64,
                "population_basis": "owner_consented_household_people",
            },
            "observation_count": 60,
            "evidence_unit_count": 60,
            "encounter_count": 60,
        },
        "benchmark_policy": policy,
        "policy": policy,
        "fit": {
            "score_semantics": "balanced_monotonic_match_score_not_posterior",
            "selected_evidence_unit_count": 100,
            "selected_encounter_count": 100,
            "selected_person_count": 100,
            "positive_pair_count": 50,
            "negative_pair_count": 250,
            "balanced_log_loss": 0.1,
            "balanced_brier_score": 0.01,
            "iterations": 1,
        },
        "benchmark_training_metrics": _calibration_metrics(50, 50),
        "benchmark_holdout_metrics": _calibration_metrics(300, 300),
        "household_training_metrics": _calibration_metrics(10, 10),
        "household_holdout_metrics": _calibration_metrics(20, 20),
        "acceptance": {
            "passed": True,
            "confidence_level": 0.95,
            "confidence_method": "clopper_pearson_one_sided",
            "max_benchmark_holdout_far_upper_confidence_bound": 0.01,
            "max_benchmark_holdout_frr": 0.35,
            "max_benchmark_holdout_misidentification_upper_confidence_bound": 0.01,
            "max_household_false_accept_encounters": 0,
            "max_household_misidentification_encounters": 0,
            "max_household_holdout_frr": 0.35,
            "max_harmful_prior_encounters": 0,
            "max_prior_rejection_rescue_encounters": 0,
            "minimum_benchmark_train_known_persons": 50,
            "minimum_benchmark_train_unknown_persons": 50,
            "minimum_benchmark_holdout_known_persons": 300,
            "minimum_benchmark_holdout_unknown_persons": 300,
            "minimum_household_train_known_encounters": 10,
            "minimum_household_train_unknown_encounters": 10,
            "minimum_household_holdout_known_encounters": 20,
            "minimum_household_holdout_unknown_encounters": 20,
        },
    }


def _authority_cutover_env(
    tmp_path,
    *,
    engine,
    topology,
    config,
    semantic_profile_sha256,
    scoring_artifact_sha256,
):
    coordinator = tmp_path / "coordinator-replay-report.json"
    coordinator.write_bytes(b'{"passed":true,"gate":"coordinator-replay"}\n')
    coordinator.chmod(0o600)
    occupied = tmp_path / "occupied-scene-report.json"
    occupied.write_bytes(b'{"passed":true,"gate":"occupied-scene"}\n')
    occupied.chmod(0o600)
    payload = {
        "contract": "noesis.identity.authority_cutover",
        "contract_version": 1,
        "approved_at_us": 3,
        "approved_by": "test-owner",
        "runtime": "ds8",
        "model_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "model_layer": config["models"]["reid"]["layer"],
        "embedding_dim": config["models"]["reid"]["embedding_dim"],
        "model_semantic_profile_sha256": semantic_profile_sha256,
        "scoring_artifact_sha256": scoring_artifact_sha256,
        "authority_runtime_profile_sha256": (
            identity_authority_runtime_profile_sha256(runtime="ds8")
        ),
        "camera_topology_sha256": hashlib.sha256(topology.read_bytes()).hexdigest(),
        "camera_ids": ["camera-a", "camera-b"],
        "authority_scope": "full_identity_runtime_public_cutover",
        "coordinator_replay": {
            "evidence_kind": "whole_frame_coordinator_replay",
            "evidence_path": str(coordinator),
            "evidence_sha256": hashlib.sha256(coordinator.read_bytes()).hexdigest(),
            "evidence_size_bytes": coordinator.stat().st_size,
            "evidence_revision": "test-v1",
            "completed_at_us": 1,
            "passed": True,
        },
        "occupied_scene": {
            "evidence_kind": "occupied_scene_runtime",
            "evidence_path": str(occupied),
            "evidence_sha256": hashlib.sha256(occupied.read_bytes()).hexdigest(),
            "evidence_size_bytes": occupied.stat().st_size,
            "evidence_revision": "test-v1",
            "completed_at_us": 2,
            "passed": True,
        },
    }
    gate = tmp_path / "authority-cutover.json"
    gate.write_text(json.dumps(payload), encoding="utf-8")
    gate.chmod(0o600)
    return {
        "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT": str(gate),
        "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256": hashlib.sha256(
            gate.read_bytes()
        ).hexdigest(),
    }


def test_authority_cutover_rejects_duplicate_selector_keys(tmp_path) -> None:
    artifact = tmp_path / "authority-cutover.json"
    artifact.write_text(
        '{"runtime":"ds9","runtime":"ds8"}',
        encoding="utf-8",
    )
    artifact.chmod(0o600)
    topology = tmp_path / "topology.yaml"
    topology.write_text("version: 1\n", encoding="utf-8")

    with pytest.raises(
        IdentityV2ConfigurationError,
        match="duplicate JSON object key",
    ):
        identity_service_module._load_authority_cutover_artifact(
            environ={
                "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT": str(artifact),
                "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256": (
                    hashlib.sha256(artifact.read_bytes()).hexdigest()
                ),
            },
            repo_root=tmp_path,
            runtime="ds8",
            model_fingerprint="a" * 64,
            model_layer="features",
            embedding_dim=4,
            model_semantic_profile_sha256="b" * 64,
            scoring_artifact_id="sha256:" + "c" * 64,
            authority_runtime_profile_sha256="d" * 64,
            topology_path=topology,
            camera_ids=("camera-a",),
        )


def _service(tmp_path, *, mode="shadow", calibration=False, extra_env=None):
    engine, topology, config, pipeline_yaml = _assets(tmp_path)
    env = {
        "NOESIS_IDENTITY_V2_MODE": mode,
        "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
        "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "identity.sqlite3"),
    }
    if calibration:
        semantic_profile_sha256 = identity_model_semantic_profile_sha256(
            reid_config=config["models"]["reid"],
            model_fingerprint=hashlib.sha256(engine.read_bytes()).hexdigest(),
            model_layer=config["models"]["reid"]["layer"],
            embedding_dim=config["models"]["reid"]["embedding_dim"],
            repo_root=tmp_path.resolve(),
            pipeline_yaml_path=pipeline_yaml.resolve(),
        )
        artifact = tmp_path / "scoring.json"
        artifact.write_text(
            json.dumps(
                _calibration_artifact(
                    engine,
                    layer=config["models"]["reid"]["layer"],
                    dimension=config["models"]["reid"]["embedding_dim"],
                    semantic_profile_sha256=semantic_profile_sha256,
                )
            ),
            encoding="utf-8",
        )
        artifact.chmod(0o600)
        env["NOESIS_IDENTITY_V2_SCORING_ARTIFACT"] = str(artifact)
        env["NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256"] = hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
        env["NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256"] = (
            semantic_profile_sha256
        )
        if mode == "authoritative":
            env.update(
                _authority_cutover_env(
                    tmp_path,
                    engine=engine,
                    topology=topology,
                    config=config,
                    semantic_profile_sha256=semantic_profile_sha256,
                    scoring_artifact_sha256=env[
                        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256"
                    ],
                )
            )
    if extra_env:
        env.update(extra_env)
    service = create_identity_v2_service(
        pipeline_config=config,
        pipeline_yaml_path=pipeline_yaml,
        camera_labels={0: "camera-a", 1: "camera-b"},
        repo_root=tmp_path,
        run_id="world-run-1",
        environ=env,
    )
    assert service is not None
    return service, engine


def _primitive(
    *,
    camera="camera-a",
    tracker="7",
    frame=1,
    embedding=(1.0, 0.0, 0.0, 0.0),
    world=(1.0, 0.0, 2.0),
):
    public = {
        "stable_id": 9001,
        "identity_state": "visitor",
        "identity_kind": "visitor",
        "resident_uuid": "legacy-resident",
        "display_name": "Legacy",
        "visitor_generation": 99,
    }
    return IdentityFramePrimitive(
        camera_id=camera,
        tracker_id=tracker,
        frame_id=frame,
        public_track=public,
        embedding=embedding,
        bbox=(10.0, 10.0, 80.0, 300.0),
        frame_size=(1920, 1080),
        detection_confidence=0.95,
        tracker_confidence=0.9,
        world_xyz=world,
        world_valid=True,
    )


def _enroll(service, *, name="Alice", sid=1):
    resident = service.store.create_resident(
        display_name=name,
        compatibility_sid=sid,
        model_fingerprint=service.model_fingerprint,
        embedding_dim=service.embedding_dim,
    )
    service.store.add_enrollment_anchor(
        resident_uuid=resident.resident_uuid,
        vector=(1.0, 0.0, 0.0, 0.0),
        model_fingerprint=service.model_fingerprint,
        embedding_dim=service.embedding_dim,
        observation_id="enrollment",
        independence_key="enrollment",
    )
    service.runtime.refresh()
    return resident


def test_factory_fingerprints_actual_engine_and_requires_explicit_profile(tmp_path):
    service, engine = _service(tmp_path)
    try:
        assert (
            service.model_fingerprint == hashlib.sha256(engine.read_bytes()).hexdigest()
        )
        assert service.model_layer == "features"
        assert service.embedding_dim == 4
        assert service.run_id == "world-run-1"
        assert service.mode.value == "shadow"
    finally:
        service.close()

    _, topology, config, pipeline_yaml = _assets(tmp_path / "missing-profile")
    del config["models"]["reid"]["layer"]
    with pytest.raises(IdentityV2ConfigurationError, match="models.reid.layer"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run",
            environ={
                "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
                "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "other.sqlite3"),
            },
        )


def test_authoritative_requires_artifact_backed_scoring(tmp_path):
    engine, topology, config, pipeline_yaml = _assets(tmp_path)
    with pytest.raises(IdentityV2ConfigurationError, match="artifact-backed"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run",
            environ={
                "NOESIS_IDENTITY_V2_MODE": "authoritative",
                "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
                "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "identity.sqlite3"),
            },
        )
    assert engine.is_file()


def test_scoring_artifact_is_bound_to_actual_model_profile(tmp_path):
    engine, topology, config, pipeline_yaml = _assets(tmp_path)
    artifact = tmp_path / "scoring.json"
    artifact.write_text(
        json.dumps(
            {
                "contract": "noesis.identity.open_set_calibration",
                "contract_version": 1,
                "model_sha256": "0" * 64,
                "model_layer": "features",
                "embedding_dim": 4,
                "policy": {},
            }
        ),
        encoding="utf-8",
    )
    artifact.chmod(0o600)
    with pytest.raises(IdentityV2ConfigurationError, match="contract is invalid"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run",
            environ={
                "NOESIS_IDENTITY_V2_MODE": "authoritative",
                "NOESIS_IDENTITY_V2_SCORING_ARTIFACT": str(artifact),
                "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
                "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
                "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "identity.sqlite3"),
            },
        )
    assert engine.is_file()


def test_authoritative_artifact_requires_independent_bytes_and_active_semantic_pins(
    tmp_path,
):
    engine, topology, config, pipeline_yaml = _assets(tmp_path)
    reid = config["models"]["reid"]
    model_sha256 = hashlib.sha256(engine.read_bytes()).hexdigest()
    semantic_sha256 = identity_model_semantic_profile_sha256(
        reid_config=reid,
        model_fingerprint=model_sha256,
        model_layer=reid["layer"],
        embedding_dim=reid["embedding_dim"],
        repo_root=tmp_path.resolve(),
        pipeline_yaml_path=pipeline_yaml.resolve(),
    )
    artifact = tmp_path / "scoring.json"
    artifact.write_text(
        json.dumps(
            _calibration_artifact(
                engine,
                layer=reid["layer"],
                dimension=reid["embedding_dim"],
                semantic_profile_sha256=semantic_sha256,
            )
        ),
        encoding="utf-8",
    )
    artifact.chmod(0o600)
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    base = {
        "NOESIS_IDENTITY_V2_MODE": "authoritative",
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT": str(artifact),
        "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
        "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "identity.sqlite3"),
    }

    def create(env):
        return create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run",
            environ=env,
        )

    with pytest.raises(IdentityV2ConfigurationError, match="independently approved"):
        create(base)
    with pytest.raises(IdentityV2ConfigurationError, match="SHA-256.*approved pin"):
        create(
            {
                **base,
                "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256": "0" * 64,
            }
        )
    pinned = {
        **base,
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256": artifact_sha256,
    }
    with pytest.raises(IdentityV2ConfigurationError, match="MODEL_SEMANTIC_PROFILE"):
        create(pinned)
    with pytest.raises(IdentityV2ConfigurationError, match="derived from the active"):
        create(
            {
                **pinned,
                "NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256": "0" * 64,
            }
        )

    nvinfer = tmp_path / "reid.ini"
    nvinfer.write_text(
        nvinfer.read_text(encoding="utf-8").replace(
            "offsets=123.675;116.28;103.53",
            "offsets=0;0;0",
        ),
        encoding="utf-8",
    )
    changed_semantic_sha256 = identity_model_semantic_profile_sha256(
        reid_config=reid,
        model_fingerprint=model_sha256,
        model_layer=reid["layer"],
        embedding_dim=reid["embedding_dim"],
        repo_root=tmp_path.resolve(),
        pipeline_yaml_path=pipeline_yaml.resolve(),
    )
    assert changed_semantic_sha256 != semantic_sha256
    with pytest.raises(IdentityV2ConfigurationError, match="derived from the active"):
        create(
            {
                **pinned,
                "NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256": semantic_sha256,
            }
        )


def test_semantic_profile_binds_loaded_reid_transform_binary(tmp_path, monkeypatch):
    engine, _topology, config, pipeline_yaml = _assets(tmp_path)
    reid = config["models"]["reid"]
    model_sha256 = hashlib.sha256(engine.read_bytes()).hexdigest()
    original = identity_model_semantic_profile_sha256(
        reid_config=reid,
        model_fingerprint=model_sha256,
        model_layer=reid["layer"],
        embedding_dim=reid["embedding_dim"],
        repo_root=tmp_path.resolve(),
        pipeline_yaml_path=pipeline_yaml.resolve(),
    )
    monkeypatch.setattr(
        identity_service_module,
        "_loaded_reid_transform_component",
        lambda: {
            "module": "noesis_reid_meta_ext",
            "status": "loaded",
            "basename": "noesis_reid_meta_ext.test.so",
            "sha256": "0" * 64,
        },
    )
    changed = identity_model_semantic_profile_sha256(
        reid_config=reid,
        model_fingerprint=model_sha256,
        model_layer=reid["layer"],
        embedding_dim=reid["embedding_dim"],
        repo_root=tmp_path.resolve(),
        pipeline_yaml_path=pipeline_yaml.resolve(),
    )
    assert changed != original


def test_authoritative_requires_loaded_transform_and_separate_cutover_evidence(
    tmp_path, monkeypatch
):
    engine, topology, config, pipeline_yaml = _assets(tmp_path)
    base = {
        "NOESIS_IDENTITY_V2_MODE": "authoritative",
        "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
        "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "identity.sqlite3"),
    }
    with monkeypatch.context() as missing_transform:
        missing_transform.setattr(
            identity_service_module,
            "_loaded_reid_transform_component",
            lambda: {
                "module": "noesis_reid_meta_ext",
                "status": "unavailable",
                "reason": "test",
            },
        )
        with pytest.raises(IdentityV2ConfigurationError, match="no substitute"):
            create_identity_v2_service(
                pipeline_config=config,
                pipeline_yaml_path=pipeline_yaml,
                camera_labels={0: "camera-a", 1: "camera-b"},
                repo_root=tmp_path,
                run_id="run",
                environ=base,
            )
    reid = config["models"]["reid"]
    semantic_sha256 = identity_model_semantic_profile_sha256(
        reid_config=reid,
        model_fingerprint=hashlib.sha256(engine.read_bytes()).hexdigest(),
        model_layer=reid["layer"],
        embedding_dim=reid["embedding_dim"],
        repo_root=tmp_path.resolve(),
        pipeline_yaml_path=pipeline_yaml.resolve(),
    )
    scoring = tmp_path / "scoring.json"
    scoring.write_text(
        json.dumps(
            _calibration_artifact(
                engine,
                layer=reid["layer"],
                dimension=reid["embedding_dim"],
                semantic_profile_sha256=semantic_sha256,
            )
        ),
        encoding="utf-8",
    )
    scoring.chmod(0o600)
    scoring_sha256 = hashlib.sha256(scoring.read_bytes()).hexdigest()
    calibrated = {
        **base,
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT": str(scoring),
        "NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256": scoring_sha256,
        "NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256": semantic_sha256,
    }
    with pytest.raises(
        IdentityV2ConfigurationError, match="separate scorer-independent"
    ):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run",
            environ=calibrated,
        )

    gate_env = _authority_cutover_env(
        tmp_path,
        engine=engine,
        topology=topology,
        config=config,
        semantic_profile_sha256=semantic_sha256,
        scoring_artifact_sha256=scoring_sha256,
    )
    service = create_identity_v2_service(
        pipeline_config=config,
        pipeline_yaml_path=pipeline_yaml,
        camera_labels={0: "camera-a", 1: "camera-b"},
        repo_root=tmp_path,
        run_id="run",
        environ={**calibrated, **gate_env},
    )
    assert service is not None
    assert service.authoritative
    assert service.authority_cutover_artifact_id
    service.close()

    occupied = tmp_path / "occupied-scene-report.json"
    original_occupied = occupied.read_bytes()
    occupied.write_bytes(original_occupied + b"tampered\n")
    with pytest.raises(IdentityV2ConfigurationError, match="size changed"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run-2",
            environ={**calibrated, **gate_env},
        )

    occupied.write_bytes(original_occupied.replace(b"occupied", b"0ccupied", 1))
    assert occupied.stat().st_size == len(original_occupied)
    with pytest.raises(IdentityV2ConfigurationError, match="SHA-256 changed"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run-3",
            environ={**calibrated, **gate_env},
        )

    occupied.write_bytes(original_occupied)
    linked = tmp_path / "occupied-scene-report-hardlink.json"
    os.link(occupied, linked)
    try:
        with pytest.raises(IdentityV2ConfigurationError, match="hard link"):
            create_identity_v2_service(
                pipeline_config=config,
                pipeline_yaml_path=pipeline_yaml,
                camera_labels={0: "camera-a", 1: "camera-b"},
                repo_root=tmp_path,
                run_id="run-4",
                environ={**calibrated, **gate_env},
            )
    finally:
        linked.unlink()

    gate_path = tmp_path / "authority-cutover.json"
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_payload["runtime"] = "ds9"
    gate_path.write_text(json.dumps(gate_payload), encoding="utf-8")
    runtime_mismatch_env = {
        **calibrated,
        "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT": str(gate_path),
        "NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256": hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest(),
    }
    with pytest.raises(IdentityV2ConfigurationError, match="runtime does not match"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 1: "camera-b"},
            repo_root=tmp_path,
            run_id="run-5",
            environ=runtime_mismatch_env,
        )


def test_complete_source_frame_calls_coordinator_once_and_never_exposes_embedding(
    tmp_path, monkeypatch
):
    service, _ = _service(tmp_path)
    try:
        calls = []
        original = service.coordinator.process_frame

        def wrapped(rows, **kwargs):
            calls.append(tuple(rows))
            return original(rows, **kwargs)

        monkeypatch.setattr(service.coordinator, "process_frame", wrapped)
        first = _primitive(tracker="1")
        second = _primitive(tracker="2", embedding=(0.0, 1.0, 0.0, 0.0))
        first.diagnostic_track = {"stable_id": 9001}
        second.diagnostic_track = {"stable_id": 9002}
        result = service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(second, first),
            observed_at=10.0,
        )
        assert len(calls) == 1
        assert len(calls[0]) == 2
        assert result.evidence_count == 2
        assert "embedding" not in first.public_track
        key = first.public_track["identity_observation_key"]
        assert key["run_id"] == "world-run-1"
        assert key["camera_id"] == "camera-a"
        assert key["tracker_id"] == "1"
        assert key["observation_id"].startswith("obs1:")
        assert first.public_track["identity_v2"]["mode"] == "shadow"
        assert first.public_track["stable_id"] == 9001
        assert first.diagnostic_track["identity_v2"]["mode"] == "shadow"
        assert first.diagnostic_track["stable_id"] == 9001
    finally:
        service.close()


def test_shadow_service_exports_score_only_evidence_with_explicit_session(tmp_path):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, _ = _service(
        tmp_path,
        extra_env={
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "test",
        },
    )
    try:
        primitive = _primitive(frame=1)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(primitive,),
            observed_at=10.0,
        )
        rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["session_id"] == "household-session-a"
        assert rows[0]["source"] == "shadow"
        assert "embedding" not in rows[0]
        assert "vector" not in rows[0]
        assert rows[0]["embedding_dim"] == 4
        assert primitive.public_track["identity_v2"]["evidence_persistence"] == (
            "durable"
        )
        assert primitive.public_track["embedding_sequence"] == rows[0]["sequence"]
        assert (
            primitive.public_track["embedding_model_sha256"] == rows[0]["model_sha256"]
        )
        assert primitive.public_track["embedding_dimension"] == rows[0]["embedding_dim"]
        assert service.evidence_recorder.health().recorded_event_count == 1
    finally:
        service.close()


def test_ds9_default_evidence_runtime_uses_async_writer_without_override(tmp_path):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, _ = _service(
        tmp_path,
        extra_env={
            "NOESIS_DEEPSTREAM_MAJOR": "9",
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
        },
    )
    try:
        assert service.evidence_recorder.runtime == "ds9"
        assert service.evidence_recorder.async_mode is True
    finally:
        service.close()


def test_async_ds9_service_does_not_publish_queued_evidence_as_durable(
    tmp_path,
):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, _ = _service(
        tmp_path,
        extra_env={
            "NOESIS_DEEPSTREAM_MAJOR": "9",
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
        },
    )
    try:
        primitive = _primitive(frame=1)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(primitive,),
            observed_at=10.0,
        )
        public = primitive.public_track
        assert public["embedding_present"] is True
        assert public["identity_observation_key"]["observation_id"].startswith("obs1:")
        assert "embedding_sequence" not in public
        assert public["identity_v2"]["fresh_embedding"] is True
        assert public["identity_v2"]["evidence_persistence"] == "queued"
        assert primitive.diagnostic_track is None
    finally:
        service.close()


def test_persisted_evidence_sequences_link_to_exact_public_observations(tmp_path):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, engine = _service(
        tmp_path,
        extra_env={
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "test",
        },
    )
    try:
        first = _primitive(tracker="9", embedding=(1.0, 0.0, 0.0, 0.0))
        second = _primitive(tracker="2", embedding=(0.0, 1.0, 0.0, 0.0))
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(first, second),
            observed_at=10.0,
        )

        persisted = {
            row["observation_id"]: row
            for row in (
                json.loads(line) for line in evidence_path.read_text().splitlines()
            )
        }
        assert len(persisted) == 2
        for primitive in (first, second):
            public = primitive.public_track
            observation_id = public["identity_observation_key"]["observation_id"]
            row = persisted[observation_id]
            assert public["embedding_sequence"] == row["sequence"]
            assert (
                public["embedding_model_sha256"]
                == hashlib.sha256(engine.read_bytes()).hexdigest()
            )
            assert public["embedding_model_sha256"] == row["model_sha256"]
            assert public["embedding_dimension"] == row["embedding_dim"] == 4
            assert "embedding" not in public
            assert "vector" not in public
    finally:
        service.close()


def test_embedding_provenance_is_omitted_without_persisted_evidence(tmp_path):
    service, _ = _service(tmp_path)
    try:
        primitive = _primitive()
        primitive.public_track.update(
            {
                "embedding_sequence": 999,
                "embedding_model_sha256": "f" * 64,
                "embedding_dimension": 999,
            }
        )
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(primitive,),
            observed_at=10.0,
        )
        assert (
            not {
                "embedding_sequence",
                "embedding_model_sha256",
                "embedding_dimension",
            }
            & primitive.public_track.keys()
        )
    finally:
        service.close()


def test_exact_frame_embedding_truth_replaces_stale_legacy_diagnostics(tmp_path):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, _ = _service(
        tmp_path,
        extra_env={
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "test",
        },
    )
    stale_fields = {
        "embedding_present": True,
        "embedding_sequence": 999,
        "embedding_model_sha256": "f" * 64,
        "embedding_dimension": 999,
    }
    try:
        missing = _primitive(frame=1, embedding=None)
        missing.public_track.update(stale_fields)
        missing.diagnostic_track = dict(stale_fields)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(missing,),
            observed_at=10.0,
        )
        assert missing.public_track["embedding_present"] is False
        assert missing.diagnostic_track["embedding_present"] is False
        assert missing.public_track["identity_v2"]["fresh_embedding"] is False
        assert "identity_observation_key" not in missing.public_track
        for field in (
            "embedding_sequence",
            "embedding_model_sha256",
            "embedding_dimension",
        ):
            assert field not in missing.public_track
            assert field not in missing.diagnostic_track

        fresh = _primitive(frame=2)
        fresh.public_track.update(stale_fields)
        fresh.diagnostic_track = dict(stale_fields)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=2,
            primitives=(fresh,),
            observed_at=10.1,
        )
        assert fresh.public_track["embedding_present"] is True
        assert fresh.diagnostic_track["embedding_present"] is True
        assert fresh.public_track["identity_v2"]["fresh_embedding"] is True
        assert fresh.public_track["identity_observation_key"]["frame_id"] == 2
        for field in (
            "embedding_sequence",
            "embedding_model_sha256",
            "embedding_dimension",
        ):
            assert fresh.public_track[field] == fresh.diagnostic_track[field]
        rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
        assert len(rows) == 1
        assert fresh.public_track["embedding_sequence"] == rows[0]["sequence"]
    finally:
        service.close()


def test_evidence_append_failure_leaves_no_embedding_provenance(tmp_path, monkeypatch):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, _ = _service(
        tmp_path,
        extra_env={
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "test",
        },
    )
    try:
        primitive = _primitive()
        primitive.public_track.update(
            {
                "embedding_sequence": 999,
                "embedding_model_sha256": "f" * 64,
                "embedding_dimension": 999,
            }
        )

        def fail_append(**_kwargs):
            raise RuntimeError("durable evidence append failed")

        monkeypatch.setattr(service.evidence_recorder, "append_frame", fail_append)
        with pytest.raises(RuntimeError, match="durable evidence append failed"):
            service.process_source_frame(
                camera_id="camera-a",
                frame_id=1,
                primitives=(primitive,),
                observed_at=10.0,
            )
        assert (
            not {
                "embedding_sequence",
                "embedding_model_sha256",
                "embedding_dimension",
            }
            & primitive.public_track.keys()
        )
    finally:
        service.close()


def test_continuity_hold_removes_fresh_embedding_provenance(tmp_path):
    evidence_path = tmp_path / "evidence" / "shadow.jsonl"
    service, _ = _service(
        tmp_path,
        extra_env={
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(evidence_path),
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "household-session-a",
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME": "test",
        },
    )
    try:
        _enroll(service)
        for frame in (1, 2):
            service.process_source_frame(
                camera_id="camera-a",
                frame_id=frame,
                primitives=(_primitive(frame=frame),),
                observed_at=10.0 + (frame / 10.0),
            )
        held = _primitive(frame=3, embedding=None)
        held.public_track.update(
            {
                "embedding_sequence": 999,
                "embedding_model_sha256": "f" * 64,
                "embedding_dimension": 999,
            }
        )
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=3,
            primitives=(held,),
            observed_at=10.3,
        )

        assert held.public_track["identity_v2"]["fresh_embedding"] is False
        assert (
            not {
                "embedding_sequence",
                "embedding_model_sha256",
                "embedding_dimension",
            }
            & held.public_track.keys()
        )
    finally:
        service.close()


def test_evidence_capture_refuses_authoritative_mode(tmp_path):
    with pytest.raises(IdentityV2ConfigurationError, match="only in shadow"):
        _service(
            tmp_path,
            mode="authoritative",
            calibration=True,
            extra_env={
                "NOESIS_IDENTITY_V2_EVIDENCE_PATH": str(tmp_path / "evidence.jsonl"),
                "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID": "session-a",
            },
        )


def test_authoritative_evidence_rejection_remains_unknown_through_world_adapter(
    tmp_path,
):
    service, _ = _service(tmp_path, mode="authoritative", calibration=True)
    try:
        _enroll(service)
        primitive = _primitive(embedding=(0.0, 1.0, 0.0, 0.0))
        primitive.diagnostic_track = {
            "stable_id": 9001,
            "identity_state": "visitor",
            "identity_kind": "visitor",
            "display_name": "Legacy",
        }
        primitive.public_track.update(
            {
                "camera_id": "camera-a",
                "tracker_id": 7,
                "frame_id": 1,
                "observed_at_us": 10_000_000,
                "capture_time_status": "estimated",
                "media_pts_ns": 1_000,
                "bbox": [10.0, 10.0, 80.0, 300.0],
                "image_size": [1920, 1080],
                "confidence": 0.95,
                "tracker_confidence": 0.9,
                "world": [1.0, 0.0, 2.0],
                "world_valid": True,
                "world_frame": "backend_world_m",
                "world_quality": "good",
                "world_source": "identity-v2-test",
            }
        )
        result = service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(primitive,),
            observed_at=10.0,
        )
        assert result.evidence_count == 1
        public = primitive.public_track
        assert public["stable_id"] is None
        assert public["display_name"] is None
        assert public["resident_uuid"] is None
        assert public["visitor_generation"] is None
        assert public["identity_state"] == "unknown"
        assert public["identity_kind"] == "unknown"
        assert primitive.diagnostic_track["stable_id"] is None
        assert primitive.diagnostic_track["identity_state"] == "unknown"
        assert primitive.diagnostic_track["identity_kind"] == "unknown"
        assert primitive.diagnostic_track["display_name"] is None
        assert primitive.diagnostic_track["identity_v2"]["state"] == "unknown"
        producer = ProducerRef(
            runtime="ds8",
            instance_id="identity-v2-test",
            run_id=service.run_id,
            software_revision="test",
        )
        artifacts = WorldArtifacts(
            calibration=ArtifactFingerprint(role="camera_calibration", sha256="a" * 64),
            model=ArtifactFingerprint(role="tracking_models", sha256="b" * 64),
            config=ArtifactFingerprint(role="runtime_config", sha256="c" * 64),
        )
        world = CanonicalWorldService(
            producer=producer,
            artifacts=artifacts,
            fusion=GlobalWorldFusion(producer),
            clock_us=lambda: 10_000_001,
        )
        publication = world.publish(0, [public], metadata={})
        assert publication.snapshot.entities[0].subject.kind.value == "unknown"
        assert (
            publication.snapshot.entities[0].entity_id
            == "unknown:world-run-1:camera-a:7"
        )
    finally:
        service.close()


def test_authoritative_osd_cache_is_exact_bounded_and_never_guesses(tmp_path):
    service, _ = _service(
        tmp_path,
        mode="authoritative",
        calibration=True,
        extra_env={"NOESIS_IDENTITY_V2_OSD_CACHE_FRAMES": "2"},
    )
    try:
        _enroll(service)
        for frame in (1, 2, 3):
            primitive = _primitive(frame=frame)
            service.process_source_frame(
                camera_id="camera-a",
                frame_id=frame,
                primitives=(primitive,),
                observed_at=10.0 + frame,
            )
        exact = service.lookup_osd_decision(
            camera_id="camera-a", frame_id=3, tracker_id="7"
        )
        assert exact is not None
        assert exact.identity_state == "resident"
        assert exact.compatibility_sid == 1
        assert exact.display_name == "Alice"
        assert (
            service.lookup_osd_decision(
                camera_id="camera-b", frame_id=3, tracker_id="7"
            )
            is None
        )
        assert (
            service.lookup_osd_decision(
                camera_id="camera-a", frame_id=2, tracker_id="8"
            )
            is None
        )
        assert (
            service.lookup_osd_decision(
                camera_id="camera-a", frame_id=1, tracker_id="7"
            )
            is None
        )
    finally:
        service.close()


def test_authoritative_osd_cache_uses_raw_public_tracker_after_epoch_scope(tmp_path):
    service, _ = _service(
        tmp_path,
        mode="authoritative",
        calibration=True,
    )
    try:
        _enroll(service)
        primitive = _primitive(tracker="7@source_epoch:1", frame=1)
        primitive.public_track["tracker_id"] = 7
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(primitive,),
            observed_at=11.0,
        )

        exact = service.lookup_osd_decision(
            camera_id="camera-a",
            frame_id=1,
            tracker_id="7",
        )
        assert exact is not None
        assert exact.tracker_id == "7"
        assert exact.identity_state == primitive.public_track["identity_v2"]["state"]
        assert (
            service.lookup_osd_decision(
                camera_id="camera-a",
                frame_id=1,
                tracker_id="7@source_epoch:1",
            )
            is None
        )
    finally:
        service.close()


def test_authoritative_missing_embedding_is_provisional_not_open_set_unknown(tmp_path):
    service, _ = _service(tmp_path, mode="authoritative", calibration=True)
    try:
        primitive = _primitive(embedding=None)
        result = service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(primitive,),
            observed_at=10.0,
        )
        assert result.evidence_count == 0
        public = primitive.public_track
        assert public["stable_id"] is None
        assert public["identity_v2"]["state"] == "provisional"
        assert public["identity_v2"]["reason"] == "server_embedding_unavailable"
        assert public["identity_state"] == "provisional"
        assert public["identity_kind"] == "provisional"
    finally:
        service.close()


def test_authoritative_resident_and_visitor_publish_only_compatibility_sid(tmp_path):
    service, _ = _service(tmp_path, mode="authoritative", calibration=True)
    try:
        resident = _enroll(service, name="Alice", sid=1)
        first = _primitive(tracker="resident", frame=1)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(first,),
            observed_at=10.0,
        )
        second = _primitive(tracker="resident", frame=2)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=2,
            primitives=(second,),
            observed_at=10.1,
        )
        assert second.public_track["stable_id"] == 1
        assert second.public_track["identity_state"] == "resident"
        assert second.public_track["identity_kind"] == "resident"
        assert second.public_track["resident_uuid"] == resident.resident_uuid
        assert second.public_track["display_name"] == "Alice"
        assert second.public_track["visitor_generation"] is None
        assert second.public_track["reid_identity"] is None
        assert second.public_track["appearance_id"] is None
        assert second.public_track["identity_v2"]["subject_id"] is None

        for frame in (1, 2):
            visitor = _primitive(
                tracker="visitor",
                frame=frame,
                embedding=(0.0, 1.0, 0.0, 0.0),
            )
            service.process_source_frame(
                camera_id="camera-a",
                frame_id=frame,
                primitives=(visitor,),
                observed_at=11.0 + frame * 0.1,
            )
            assert visitor.public_track["stable_id"] is None
        visitor = _primitive(
            tracker="visitor",
            frame=3,
            embedding=(0.0, 1.0, 0.0, 0.0),
        )
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=3,
            primitives=(visitor,),
            observed_at=11.3,
        )
        assert visitor.public_track["identity_state"] == "visitor"
        assert visitor.public_track["identity_kind"] == "visitor"
        assert visitor.public_track["stable_id"] == 1000
        assert visitor.public_track["visitor_generation"] == 1
        assert visitor.public_track["display_name"] is None
        assert visitor.public_track["resident_uuid"] is None
    finally:
        service.close()


def test_overlap_permit_requires_topology_world_and_appearance_proof(tmp_path):
    service, _ = _service(tmp_path)
    try:
        _enroll(service)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(_primitive(camera="camera-a", tracker="a", frame=1),),
            observed_at=10.0,
        )
        known_a = _primitive(camera="camera-a", tracker="a", frame=2)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=2,
            primitives=(known_a,),
            observed_at=10.1,
        )
        assert known_a.public_track["identity_v2"]["state"] == "resident"

        bad_world = _primitive(
            camera="camera-b", tracker="b", frame=1, world=(20.0, 0.0, 20.0)
        )
        denied = service.process_source_frame(
            camera_id="camera-b",
            frame_id=1,
            primitives=(bad_world,),
            observed_at=10.2,
        )
        assert denied.overlap_permit_count == 0

        proof = _primitive(camera="camera-b", tracker="b", frame=2)
        permitted = service.process_source_frame(
            camera_id="camera-b",
            frame_id=2,
            primitives=(proof,),
            observed_at=10.3,
        )
        assert permitted.overlap_permit_count == 1

        wrong_appearance = _primitive(
            camera="camera-b",
            tracker="c",
            frame=1,
            embedding=(0.0, 1.0, 0.0, 0.0),
        )
        denied_appearance = service.process_source_frame(
            camera_id="camera-b",
            frame_id=1,
            primitives=(wrong_appearance,),
            observed_at=10.4,
        )
        assert denied_appearance.overlap_permit_count == 0
    finally:
        service.close()


def test_empty_source_frame_clears_camera_local_identity_proof_state(tmp_path):
    service, _ = _service(tmp_path)
    try:
        _enroll(service)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=1,
            primitives=(_primitive(camera="camera-a", tracker="a", frame=1),),
            observed_at=10.0,
        )
        known = _primitive(camera="camera-a", tracker="a", frame=2)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=2,
            primitives=(known,),
            observed_at=10.1,
        )
        assert known.public_track["identity_v2"]["state"] == "resident"
        held = _primitive(camera="camera-a", tracker="a", frame=3, embedding=None)
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=3,
            primitives=(held,),
            observed_at=10.15,
        )
        assert held.public_track["identity_v2"]["state"] == "resident"
        assert held.public_track["identity_v2"]["fresh_embedding"] is False
        assert not service._recent_proofs  # type: ignore[attr-defined]
        assert service._held_overlays  # type: ignore[attr-defined]
        result = service.process_source_frame(
            camera_id="camera-a",
            frame_id=4,
            primitives=(),
            observed_at=10.2,
        )
        assert result.primitive_count == 0
        assert not service._recent_proofs  # type: ignore[attr-defined]
        assert not service._held_overlays  # type: ignore[attr-defined]
        after_disappearance = _primitive(
            camera="camera-a", tracker="a", frame=5, embedding=None
        )
        service.process_source_frame(
            camera_id="camera-a",
            frame_id=5,
            primitives=(after_disappearance,),
            observed_at=10.3,
        )
        assert after_disappearance.public_track["identity_v2"]["state"] == "provisional"
    finally:
        service.close()


def test_topology_must_exactly_match_runtime_camera_map(tmp_path):
    _, topology, config, pipeline_yaml = _assets(tmp_path)
    with pytest.raises(IdentityV2ConfigurationError, match="exactly match"):
        create_identity_v2_service(
            pipeline_config=config,
            pipeline_yaml_path=pipeline_yaml,
            camera_labels={0: "camera-a", 2: "camera-b"},
            repo_root=tmp_path,
            run_id="run",
            environ={
                "NOESIS_IDENTITY_V2_CAMERA_TOPOLOGY": str(topology),
                "NOESIS_IDENTITY_V2_STORE": str(tmp_path / "identity.sqlite3"),
            },
        )
