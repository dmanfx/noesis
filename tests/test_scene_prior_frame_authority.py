from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from noesis.scene_prior_builder import (
    ScenePriorBuildError,
    ScenePriorBuildConfig,
    _camera_preview_frame,
    _bundle_inputs,
    _load_points_glb,
    _source_physical_frame,
    build_accepted_room_to_home_binding,
    build_scene_prior,
    import_accepted_room_to_home_binding,
)
from noesis.calibration.manager import CalibrationManager, CalibrationValidationError
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene_prior import (
    ScenePriorFrameBinding,
    ScenePriorFrameRef,
    ScenePriorMetricFrame,
    ScenePriorWorldToScenePresentation,
    metric_frame_revision_sha256,
    scene_revision_sha256,
    world_to_scene_presentation_sha256,
)
from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256
from noesis.virtual_twin.artifacts import write_points_glb


def _artifact(role: str, letter: str) -> ArtifactFingerprint:
    return ArtifactFingerprint(role=role, sha256=letter * 64)


def _accepted_inputs() -> tuple[dict, dict]:
    provenance = (
        _artifact("camera_calibration_physical", "a"),
        _artifact("world_alignment_physical", "b"),
    )
    target_revision = metric_frame_revision_sha256(
        "backend_world_m",
        (0.0, 1.0, 0.0),
        -1.0,
        tuple(item.sha256 for item in provenance),
    )
    matrix = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    kwargs = {
        "artifact_revision_id": "room-artifact-v2",
        "source_frame_revision": "room-source-v2",
        "target_coordinate_revision": target_revision,
        "target_from_source_col_major": matrix,
        "source_floor_normal": (0.0, 1.0, 0.0),
        "source_floor_offset_m": -1.0,
        "target_floor_normal": (0.0, 1.0, 0.0),
        "target_floor_offset_m": -1.0,
        "source_camera_calibration_sha256": "d" * 64,
        "source_world_alignment_sha256": "e" * 64,
        "source_camera_calibration_physical_sha256": "a" * 64,
        "source_world_alignment_physical_sha256": "b" * 64,
        "target_revision_id": "home-revision",
        "target_revision_metadata_sha256": "f" * 64,
        "metric_frame_provenance": provenance,
        "world_to_scene_col_major": matrix,
        "world_to_scene_provenance": _artifact("world_to_scene_presentation", "c"),
    }
    # The report's edge digest is filled after the builder calculates it.
    return kwargs, {}


def test_metric_frame_revision_binds_floor_plane_and_presentation_is_similarity() -> None:
    provenance = (
        _artifact("camera_calibration_physical", "a"),
        _artifact("world_alignment_physical", "b"),
    )
    revision = metric_frame_revision_sha256(
        "backend_world_m", (0.0, 1.0, 0.0), -1.0,
        tuple(item.sha256 for item in provenance),
    )
    frame = {
        "frame_id": "backend_world_m",
        "revision": revision,
    }
    metric = ScenePriorMetricFrame(
        contract="noesis.scene_prior.metric_frame",
        contract_version=1,
        frame=frame,
        units="meters",
        floor_plane={"frame": frame, "normal": (0.0, 1.0, 0.0), "offset_m": -1.0},
        physical_provenance=provenance,
        accepted=True,
    )
    assert metric.frame.revision == revision
    with pytest.raises(ValueError, match="metric frame revision"):
        ScenePriorMetricFrame(
            contract="noesis.scene_prior.metric_frame",
            contract_version=1,
            frame=frame,
            units="meters",
            floor_plane={"frame": frame, "normal": (0.0, 1.0, 0.0), "offset_m": -0.9},
            physical_provenance=provenance,
            accepted=True,
        )

    with pytest.raises(ValueError, match="uniform similarity"):
        ScenePriorWorldToScenePresentation(
            contract="noesis.scene_prior.world_to_scene_presentation",
            contract_version=1,
            world_frame=frame,
            render_frame="menon_scene",
            world_to_scene_col_major=(
                1.0, 0.0, 0.0, 0.0,
                0.2, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ),
            world_to_scene_sha256="0" * 64,
            scene_revision_id="1" * 64,
            source_transform_sha256s=("1" * 64,),
            provenance=_artifact("presentation", "c"),
            accepted=True,
        )


def test_presentation_digest_uses_cross_language_ieee754_bytes() -> None:
    frame = ScenePriorFrameRef(frame_id="backend_world_m", revision="a" * 64)
    matrix = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        2**-18, -0.0, 1e-300, 1.0,
    )
    assert world_to_scene_presentation_sha256(
        frame, "menon_scene", matrix
    ) == "1f45dc9be2c24f331d62e594eae543bc3da91a82e4464430f124b8b561641c6b"
    assert scene_revision_sha256(
        frame, "menon_scene", matrix
    ) == "a69d3362a73984eee3b19e90b99c72453152e5a6f506b9be71624ee9b40f6045"

def test_room_to_home_builder_requires_exact_passed_holdout_report() -> None:
    kwargs, _ = _accepted_inputs()
    # Obtain the exact edge identity with a report assembled from the same
    # immutable source/target and matrix values.
    target_revision = kwargs["target_coordinate_revision"]
    from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256

    edge_digest = revisioned_transform_sha256(
        RevisionedFrame("backend_world_m", kwargs["source_frame_revision"]),
        RevisionedFrame("backend_world_m", target_revision),
        kwargs["target_from_source_col_major"],
    )
    report = {
        "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
        "status": "passed",
        "accepted_for_canonical_use": True,
        "disposition": "validated_cross_session_registration",
        "holdout": {"source": "pose_graph", "status": "passed"},
        "pose_graph": {"accepted": True, "reason_codes": []},
        "bound_transform": {
            "source_frame": {"frame_id": "backend_world_m", "revision": kwargs["source_frame_revision"]},
            "target_frame": {"frame_id": "backend_world_m", "revision": target_revision},
            "target_from_source_col_major": list(kwargs["target_from_source_col_major"]),
            "target_from_source_sha256": edge_digest,
            "source_floor_plane": {
                "normal": list(kwargs["source_floor_normal"]),
                "offset_m": kwargs["source_floor_offset_m"],
            },
            "target_floor_plane": {
                "normal": list(kwargs["target_floor_normal"]),
                "offset_m": kwargs["target_floor_offset_m"],
            },
        },
    }
    binding = build_accepted_room_to_home_binding(**kwargs, acceptance_report=report)
    assert binding.contract_version == 2
    assert binding.metric_frame is not None
    assert binding.artifact_revision_id == "room-artifact-v2"

    review_only = {**report, "accepted_for_canonical_use": False, "disposition": "review_only"}
    with pytest.raises(ScenePriorBuildError, match="not accepted"):
        build_accepted_room_to_home_binding(**kwargs, acceptance_report=review_only)

    serialized = {**binding.model_dump(mode="json"), "acceptance_report": report}
    restored = import_accepted_room_to_home_binding(serialized)
    assert restored == binding
    with pytest.raises(ScenePriorBuildError, match="acceptance report"):
        import_accepted_room_to_home_binding(binding.model_dump(mode="json"))


def test_normal_v2_builder_catalog_manager_roundtrip_applies_nonidentity_edge(
    tmp_path: Path,
) -> None:
    """Exercise the ordinary builder path with an exact, synthetic accepted edge."""

    pytest.importorskip("trimesh")

    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    # The aligned GLB is already materialized by this prior's historical
    # floor-leveling correction.  The accepted edge below targets a new
    # metric revision, so the normal builder must apply only new @ inv(old).
    old_angle = np.deg2rad(9.0)
    old_correction_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.12],
            [0.0, np.cos(old_angle), -np.sin(old_angle), -0.20],
            [0.0, np.sin(old_angle), np.cos(old_angle), 0.08],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    old_correction = [
        float(value) for value in old_correction_matrix.flatten(order="F")
    ]
    bundle = tmp_path / "bundle"
    for relative in (
        "aligned",
        "alignment",
        "calibration",
        "target/revision-a",
    ):
        (bundle / relative).mkdir(parents=True)
    points_path = tmp_path / "points.glb"
    points = np.asarray(
        [
            (0.25 * column, height, 0.25 * row)
            for row in range(21)
            for column in range(21)
            for height in (0.0, 0.04, 0.22, 0.48)
        ],
        dtype=np.float32,
    )
    homogeneous_points = np.concatenate(
        (points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)),
        axis=1,
    )
    materialized_points = (
        old_correction_matrix @ homogeneous_points.T
    ).T[:, :3].astype(np.float32)
    write_points_glb(
        points_path,
        materialized_points,
        np.full((len(points), 3), 180, dtype=np.uint8),
    )
    (bundle / "aligned/points.glb").write_bytes(points_path.read_bytes())
    alignment_report = {
        "schema": "noesis.phone_walk.conditioned_fusion.alignment_report.v1",
        "status": "passed",
        "quality_gate": {"passed": True},
    }
    transform = {
        "schema": "noesis.phone_walk.conditioned_fusion.world_alignment.v1",
        "source_coordinate_frame": "backend_world_m_stream_points",
        "target_coordinate_frame": "backend_world_m_stream_points",
        "scale": 1.0,
        "world_from_source_row_major": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    calibration = {
        "cameras": {
            "camera-a": {"E": identity},
            "camera-b": {"E": identity},
        }
    }
    target_metadata = {
        "schema": "noesis.room_reconstruction.stream_points.v4",
        "revision_id": "revision-a",
        "camera": "camera-a",
        "coordinate_frame": "backend_world_m_stream_points",
        "calibrated_floor_y": 0.0,
        "floor_alignment": {
            "world_correction_col_major": old_correction,
            "target_floor_y": 0.0,
        },
    }
    world_to_scene = {"world_to_scene_col_major": identity, "floor_y": 0.0}
    payloads = {
        "aligned/points.glb": (bundle / "aligned/points.glb").read_bytes(),
        "alignment/alignment_report.json": (json.dumps(alignment_report) + "\n").encode(),
        "alignment/identity_backend_world.json": (json.dumps(transform) + "\n").encode(),
        "calibration/camera_calibration.json": (json.dumps(calibration) + "\n").encode(),
        "target/revision-a/room_points_meta.json": (json.dumps(target_metadata) + "\n").encode(),
    }
    for relative, payload in payloads.items():
        (bundle / relative).write_bytes(payload)
    reference = {
        "schema": "noesis.reference.room_scan_bundle.v1",
        "bundle_id": "bundle-v2-test",
        "created_at": "2026-08-02T00:00:00Z",
        "quality": {"passed": True},
        "coordinate_contract": {
            "target_frame": "backend_world_m_stream_points",
            "transform": "alignment/identity_backend_world.json",
        },
        "review_assets": {
            "aligned_rgb_glb": "aligned/points.glb",
            "alignment_report": "alignment/alignment_report.json",
        },
        "noesis_reference": {
            "camera_id": "camera-a",
            "camera_calibration": "calibration/camera_calibration.json",
            "revision": "target/revision-a",
        },
        "capture": {
            "scan_id": "capture-v2-test",
            "captured_at": "2026-08-01T00:00:00Z",
            "source_type": "mapanything_multiview_room_walk",
            "model": "test",
        },
    }
    payloads["reference.json"] = (json.dumps(reference) + "\n").encode()
    manifest = {
        "schema": "noesis.reference.room_scan_bundle_manifest.v1",
        "bundle_id": "bundle-v2-test",
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for relative, payload in sorted(payloads.items())
        ],
    }
    payloads["bundle_manifest.json"] = (json.dumps(manifest) + "\n").encode()
    for relative, payload in payloads.items():
        (bundle / relative).write_bytes(payload)
    sums = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {relative}\n"
        for relative, payload in sorted(payloads.items())
    )
    (bundle / "SHA256SUMS").write_text(sums, encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    authored_scene = repo_root / "tests/fixtures/authored_scene_oracle/simple_home.obj"
    authored_map = {
        "contract": "noesis.authored_scene.room_group_map",
        "contract_version": 1,
        "authored_scene_sha256": hashlib.sha256(authored_scene.read_bytes()).hexdigest(),
        "rooms": {"Living Room": ["room_living"]},
    }
    room_map_path = tmp_path / "room_group_map.json"
    room_map_path.write_text(json.dumps(authored_map) + "\n", encoding="utf-8")
    world_path = tmp_path / "world_to_scene.json"
    world_path.write_text(json.dumps(world_to_scene) + "\n", encoding="utf-8")

    inputs = _bundle_inputs(bundle)
    source, camera_physical, alignment_physical = _source_physical_frame(inputs)
    preview = _camera_preview_frame(inputs)
    yaw = np.deg2rad(12.0)
    yaw_matrix = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    edge_matrix = yaw_matrix @ old_correction_matrix
    edge_matrix[:3, 3] += np.asarray([0.25, 0.0, 0.0])
    edge_values = tuple(float(value) for value in edge_matrix.flatten(order="F"))
    provenance = (
        ArtifactFingerprint(role="camera_calibration_physical", sha256=camera_physical),
        ArtifactFingerprint(role="world_alignment_physical", sha256=alignment_physical),
    )
    target_revision = metric_frame_revision_sha256(
        "backend_world_m", (0.0, 1.0, 0.0), 0.0,
        tuple(item.sha256 for item in provenance),
    )
    target = RevisionedFrame("backend_world_m", target_revision)
    edge_digest = revisioned_transform_sha256(source, target, edge_values)
    report = {
        "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
        "status": "passed",
        "accepted_for_canonical_use": True,
        "disposition": "validated_cross_session_registration",
        "holdout": {"source": "pose_graph", "status": "passed"},
        "pose_graph": {"accepted": True, "reason_codes": []},
        "bound_transform": {
            "source_frame": {"frame_id": "backend_world_m", "revision": source.revision},
            "target_frame": {"frame_id": "backend_world_m", "revision": target_revision},
            "target_from_source_col_major": list(edge_values),
            "target_from_source_sha256": edge_digest,
            "source_floor_plane": {
                "normal": list(preview.source_floor_normal),
                "offset_m": preview.source_floor_offset_m,
            },
            "target_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
        },
    }
    binding = build_accepted_room_to_home_binding(
        artifact_revision_id="producer-artifact",
        source_frame_revision=source.revision,
        target_coordinate_revision=target_revision,
        target_from_source_col_major=edge_values,
        source_floor_normal=preview.source_floor_normal,
        source_floor_offset_m=preview.source_floor_offset_m,
        target_floor_normal=(0.0, 1.0, 0.0),
        target_floor_offset_m=0.0,
        source_camera_calibration_sha256=hashlib.sha256(payloads["calibration/camera_calibration.json"]).hexdigest(),
        source_world_alignment_sha256=hashlib.sha256(payloads["alignment/identity_backend_world.json"]).hexdigest(),
        source_camera_calibration_physical_sha256=camera_physical,
        source_world_alignment_physical_sha256=alignment_physical,
        target_revision_id="revision-a",
        target_revision_metadata_sha256=hashlib.sha256(payloads["target/revision-a/room_points_meta.json"]).hexdigest(),
        metric_frame_provenance=provenance,
        acceptance_report=report,
        world_to_scene_col_major=identity,
        world_to_scene_provenance=ArtifactFingerprint(
            role="world_to_scene", sha256=hashlib.sha256(world_path.read_bytes()).hexdigest()
        ),
    )
    output_root = tmp_path / "scene_priors"
    result = build_scene_prior(
        ScenePriorBuildConfig(
            source_bundle=bundle,
            site_id="site-v2-test",
            space_id="living-room-v2-test",
            semantic_rooms=("Living Room",),
            authored_scene=authored_scene,
            room_group_map=room_map_path,
            world_to_scene=world_path,
            output_root=output_root,
            camera_ids=("camera-a",),
            grid_resolution_m=0.25,
            accepted_frame_binding=binding,
            registration_acceptance_report=report,
        )
    )
    manifest = json.loads(result.manifest_path.read_text())
    expected_camera = edge_matrix[:3, 3]
    assert manifest["preview"]["camera_position_world_m"] == pytest.approx(
        expected_camera.tolist()
    )
    selected_points, _ = _load_points_glb((result.revision_dir / "points.glb").read_bytes())
    expected_known_point = (
        edge_matrix
        @ np.asarray([0.0, 0.48, 0.0, 1.0], dtype=np.float64)
    )[:3]
    nearest_known = np.min(
        np.linalg.norm(selected_points - expected_known_point[None, :], axis=1)
    )
    assert nearest_known < 1e-5
    catalog_binding = json.loads(result.catalog_path.read_text())["camera_bindings"][0]["frame_binding"]

    edge_b_matrix = edge_matrix.copy()
    edge_b_matrix[:3, 3] += np.asarray([-0.75, 0.0, 0.0])
    edge_b_values = tuple(float(value) for value in edge_b_matrix.flatten(order="F"))
    edge_b_digest = revisioned_transform_sha256(source, target, edge_b_values)
    binding_b_payload = json.loads(json.dumps(catalog_binding))
    binding_b_payload["target_from_source_col_major"] = list(edge_b_values)
    binding_b_payload["target_from_source_sha256"] = edge_b_digest
    binding_b_payload["presentation"]["source_transform_sha256s"] = [edge_b_digest]
    binding_b = ScenePriorFrameBinding.model_validate(binding_b_payload)
    binding_a = ScenePriorFrameBinding.model_validate(catalog_binding)

    cameras_yaml = tmp_path / "cameras.yaml"
    cameras_yaml.write_text(
        """intrinsics_models:\n  test_model:\n    intrinsics:\n      fx: 800.0\n      fy: 800.0\n      cx: 640.0\n      cy: 360.0\ncameras:\n  0:\n    name: camera-a\n    model: test_model\n  1:\n    name: camera-b\n    model: test_model\n""",
        encoding="utf-8",
    )
    camera_path = tmp_path / "camera_calibration.json"
    camera_path.write_bytes(payloads["calibration/camera_calibration.json"])
    alignment_path = tmp_path / "ply_alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "matrix": identity,
                "floor_y": 0.0,
                "units": {"s_obj_to_m": 1.0},
                "scene_similarity": {
                    "world_to_scene_col_major": identity,
                    "target_world_to_scene_col_major": identity,
                    "target_registration_sha256": hashlib.sha256(world_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    manager = CalibrationManager(
        cameras_yaml,
        camera_path,
        alignment_path,
        raw_audit_dir=tmp_path / "audit",
        scene_prior_frame_bindings={
            "camera-a": binding_a,
            "camera-b": binding_b,
        },
    )
    manager.set_camera_labels({0: "camera-a", 1: "camera-b"})
    contract = manager.frame_contract("camera-a")
    assert contract.target_frame.revision == target_revision
    assert contract.target_from_source_col_major[12] == pytest.approx(edge_values[12])
    assert manager.frame_contract("camera-b").transform_sha256 == edge_b_digest
    presentations = manager.world_frame_presentations()
    assert presentations[target_revision]["world_to_scene_col_major"] == identity
    assert presentations[target_revision]["source_transform_sha256s"] == sorted(
        [edge_digest, edge_b_digest]
    )
    manager._align["scene_similarity"]["target_world_to_scene_col_major"][12] = 0.5
    with pytest.raises(CalibrationValidationError, match="target authored registration"):
        manager.world_frame_presentations()
