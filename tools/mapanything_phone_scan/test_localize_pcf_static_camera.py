from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from noesis.calibration.manager import CalibrationManager
from noesis.scene_prior_builder import (
    ScenePriorBuildConfig,
    _bundle_inputs,
    _camera_preview_frame,
    _source_physical_frame,
    build_accepted_room_to_home_binding,
    build_scene_prior,
)
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene_prior import (
    ScenePriorFrameBinding,
    metric_frame_revision_sha256,
)
from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256
from noesis.virtual_twin.artifacts import write_points_glb

from tools.mapanything_phone_scan.calibration_replacement import (
    CalibrationReplacementError,
    regenerate_static_depth_world_points,
    write_calibration_replacement,
)
from tools.mapanything_phone_scan.localize_pcf_static_camera import (
    StaticCameraLocalizationError,
    _adapt_static_image_metadata,
    _load_independent_scale_evidence,
    _load_model_rgb,
    _nearest_pcf_point,
    _validate_excluded_ranges,
    _validate_static_image_contract,
    _write_localization_failure_report,
)


def _static_fixture(tmp_path: Path) -> tuple[Path, dict]:
    image_path = tmp_path / "static.jpg"
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    cv2.rectangle(image, (3, 3), (28, 20), (40, 180, 90), 2)
    assert cv2.imwrite(str(image_path), image)
    metadata = {
        "schema": "noesis.pcf.static_camera_image.v1",
        "camera_id": "living-room",
        "image_size": [32, 24],
        "intrinsics": [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]],
        "distortion_model": "none",
        "distortion": [],
        "rectification": {"status": "rectified", "config_revision": "fixture-r1"},
        "source_frame": {
            "frame_id": "living-room-rgb-0001",
            "revision": "camera-revision-r1",
            "coordinate_frame": "backend_world_m",
            "units": "m",
            "image_sha256": "",
        },
    }
    from tools.mapanything_phone_scan.localize_pcf_static_camera import _sha256

    metadata["source_frame"]["image_sha256"] = _sha256(image_path)
    return image_path, metadata


def test_static_image_contract_requires_exact_calibration_provenance(tmp_path: Path) -> None:
    image_path, metadata = _static_fixture(tmp_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    intrinsics, distortion, contract = _validate_static_image_contract(
        metadata, image, image_path=image_path, camera_id="living-room"
    )
    assert intrinsics.shape == (3, 3)
    assert distortion is None
    assert contract["source_frame"]["revision"] == "camera-revision-r1"

    wrong = dict(metadata)
    wrong["source_frame"] = dict(metadata["source_frame"])
    wrong["source_frame"]["revision"] = "wrong-revision"
    with pytest.raises(StaticCameraLocalizationError, match="does not match the keyframe"):
        # The digest check remains independent of revision and catches a
        # changed image before any geometric estimate is attempted.
        wrong["source_frame"]["image_sha256"] = "0" * 64
        _validate_static_image_contract(wrong, image, image_path=image_path, camera_id="living-room")


def test_localizer_uses_serialized_model_rgb_grid_for_depth_projection() -> None:
    model_rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    model_rgb[1, 3] = [7, 11, 13]
    mask = np.ones((2, 4), dtype=bool)
    world_points = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    confidence = np.ones((2, 4), dtype=np.float64)
    agreement = np.ones((2, 4), dtype=bool)

    loaded = _load_model_rgb({"model_rgb": model_rgb}, mask)
    assert loaded.shape == (2, 4, 3)
    np.testing.assert_array_equal(loaded, model_rgb)
    point = _nearest_pcf_point(
        phone_pixel=(3.0, 1.0),
        phone_size=(loaded.shape[1], loaded.shape[0]),
        world_points=world_points,
        mask=mask,
        confidence=confidence,
        agreement=agreement,
    )
    np.testing.assert_array_equal(point, world_points[1, 3])


def test_localization_failure_report_preserves_bounded_rejection_evidence(
    tmp_path: Path,
) -> None:
    row = {
        "view_index": 12,
        "fit_excluded": True,
        "accepted_for_fit": False,
        "rejection_reasons": ["excluded_from_fit", "spatial_support_gate_failed"],
        "_objects": np.zeros((1, 3)),
    }
    path = _write_localization_failure_report(
        tmp_path / "failure",
        camera_id="living-room",
        full_pcf_pose=True,
        excluded_ranges=((12, 13),),
        frame_count=20,
        evaluated_views=[row],
        candidates=[],
        inputs={"source_world_manifest_sha256": "manifest-sha"},
        failure_reason="only 0 independent phone views localized the static camera",
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["configured_minimum_independent_views"] == 4
    assert report["evidence"]["fit_eligible_view_count"] == 0
    assert report["evidence"]["heldout_evaluated_view_count"] == 1
    assert report["evidence"]["rejection_reason_counts"] == {
        "excluded_from_fit": 1,
        "spatial_support_gate_failed": 1,
    }
    assert report["evidence"]["per_view"][0]["view_index"] == 12


def test_legacy_stream_points_keyframe_is_adapted_as_review_only(tmp_path: Path) -> None:
    image_path, metadata = _static_fixture(tmp_path)
    legacy_keyframe = tmp_path / "keyframes" / "living-room_0001.jpg"
    legacy_keyframe.parent.mkdir()
    legacy_keyframe.write_bytes(image_path.read_bytes())
    legacy = {
        "schema": "noesis.room_reconstruction.stream_points.v4",
        "camera": "living-room",
        "revision_id": "stream-r1",
        "coordinate_frame": "backend_world_m_stream_points",
        "intrinsics": metadata["intrinsics"],
        "rgb_keyframes": {"living-room_0001": "keyframes/living-room_0001.jpg"},
    }
    metadata_path = tmp_path / "room_points_meta.json"
    metadata_path.write_text(json.dumps(legacy), encoding="utf-8")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    adapted = _adapt_static_image_metadata(
        legacy,
        image_path=image_path,
        image=image,
        camera_id="living-room",
        metadata_path=metadata_path,
    )
    assert adapted["schema"] == "noesis.pcf.static_camera_image.v1"
    assert adapted["input_schema"] == "noesis.room_reconstruction.stream_points.v4"
    assert adapted["distortion_provenance"] == "missing_legacy_stream_points_metadata"
    assert adapted["rectification"]["status"] == "unknown"


def test_excluded_ranges_are_bounded_half_open_intervals() -> None:
    assert _validate_excluded_ranges(((2, 5), (8, 10)), 12) == ((2, 5), (8, 10))
    with pytest.raises(StaticCameraLocalizationError):
        _validate_excluded_ranges(((2, 6), (5, 8)), 12)
    with pytest.raises(StaticCameraLocalizationError):
        _validate_excluded_ranges(((0, 13),), 12)


def test_independent_scale_evidence_rejects_wrong_frame_without_alignment_fit(tmp_path: Path) -> None:
    image_path, metadata = _static_fixture(tmp_path)
    evidence_path = tmp_path / "scale.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "noesis.pcf.static_camera.scale_evidence.v1",
                "units": "m",
                "alignment_fit_used": False,
                "source_frame": {
                    "frame_id": "living-room-rgb-0001",
                    "revision": "wrong-revision",
                    "coordinate_frame": "backend_world_m",
                },
                "measurements": [
                    {"independent": True, "provenance": "tape-a", "measured_distance_m": 1.0, "candidate_distance_m": 1.0},
                    {"independent": True, "provenance": "tape-b", "measured_distance_m": 2.0, "candidate_distance_m": 2.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StaticCameraLocalizationError, match="source frame"):
        _load_independent_scale_evidence(evidence_path, source_frame=metadata["source_frame"])


def test_replacement_writes_exact_backup_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "camera_calibration.json"
    original = {
        "cameras": {
            "living-room": {
                "E": np.eye(4).reshape(-1, order="F").tolist(),
                "pose": {"source": "manual-approximate"},
            },
            "family-room": {"E": np.eye(4).reshape(-1, order="F").tolist()},
        }
    }
    source.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
    original_bytes = source.read_bytes()
    output = tmp_path / "replacement" / "camera_calibration.json"
    provenance = {
        "accepted_for_canonical_use": True,
        "acceptance": {"status": "passed"},
        "frame_identity": {
            "frame_id": "living-room-rgb-0001",
            "revision": "camera-revision-r1",
            "coordinate_frame": "backend_world_m",
            "registration_fingerprint": "fp-r1",
        },
        "assembly_frame_identity": {
            "frame_id": "pcf-assembly-living-r1",
            "revision": "assembly-revision-r1",
            "coordinate_frame": "pcf_assembly_metric_world_m",
        },
    }
    moved = np.eye(4)
    moved[:3, 3] = [1.0, 2.0, 3.0]
    result = write_calibration_replacement(
        source,
        output,
        camera_id="living-room",
        camera_to_world=moved,
        calibration_from_assembly=np.eye(4),
        provenance=provenance,
        backup_root=tmp_path / "backups",
    )
    assert source.read_bytes() == original_bytes
    assert result["active_source_mutated"] is False
    backup_path = next((tmp_path / "backups").glob("*/camera_calibration.json"))
    assert backup_path.read_bytes() == original_bytes
    replacement = json.loads(output.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        np.asarray(replacement["cameras"]["living-room"]["E"]).reshape(4, 4, order="F"),
        np.linalg.inv(moved),
    )


def test_replacement_rejects_pose_flag_without_measured_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "camera_calibration.json"
    source.write_text(json.dumps({"cameras": {"living-room": {"E": np.eye(4).reshape(-1, order="F").tolist()}}}), encoding="utf-8")
    with pytest.raises(CalibrationReplacementError, match="measured acceptance"):
        write_calibration_replacement(
            source,
            tmp_path / "out.json",
            camera_id="living-room",
            camera_to_world=np.eye(4),
            calibration_from_assembly=np.eye(4),
            provenance={"accepted_for_canonical_use": True},
            backup_root=tmp_path / "backups",
        )


def test_replacement_refuses_same_or_preexisting_output_before_backup(tmp_path: Path) -> None:
    source = tmp_path / "camera_calibration.json"
    source.write_text(
        json.dumps({"cameras": {"living-room": {"E": np.eye(4).reshape(-1, order="F").tolist()}}}),
        encoding="utf-8",
    )
    provenance = {
        "accepted_for_canonical_use": True,
        "acceptance": {"status": "passed"},
        "frame_identity": {
            "frame_id": "backend_world_m",
            "revision": "backend-r1",
            "coordinate_frame": "backend_world_m",
        },
        "assembly_frame_identity": {
            "frame_id": "assembly_world",
            "revision": "assembly-r1",
            "coordinate_frame": "pcf_assembly_metric_world_m",
        },
    }
    with pytest.raises(CalibrationReplacementError, match="differ"):
        write_calibration_replacement(
            source,
            source,
            camera_id="living-room",
            camera_to_world=np.eye(4),
            calibration_from_assembly=np.eye(4),
            provenance=provenance,
            backup_root=tmp_path / "backups-same",
        )
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(CalibrationReplacementError, match="already exists"):
        write_calibration_replacement(
            source,
            existing,
            camera_id="living-room",
            camera_to_world=np.eye(4),
            calibration_from_assembly=np.eye(4),
            provenance=provenance,
            backup_root=tmp_path / "backups-existing",
        )
    assert not (tmp_path / "backups-same").exists()
    assert not (tmp_path / "backups-existing").exists()


def test_static_depth_direct_consumer_reprojects_with_replacement_pose(tmp_path: Path) -> None:
    depth_input = tmp_path / "static_depth.npz"
    depth = np.asarray([[2.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    mask = depth > 0.0
    intrinsics = np.asarray([[10.0, 0.0, 0.5], [0.0, 10.0, 0.5], [0.0, 0.0, 1.0]])
    np.savez_compressed(depth_input, depth_z=depth, mask=mask, intrinsics=intrinsics)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    output = tmp_path / "direct" / "static_world_points.npz"
    result = regenerate_static_depth_world_points(
        depth_input,
        output,
        camera_to_world=pose,
        frame_identity={
            "frame_id": "backend_world_m",
            "revision": "backend-r1",
            "coordinate_frame": "backend_world_m",
        },
    )
    assert result["valid_point_count"] == 3
    with np.load(output) as payload:
        assert payload["world_points"].shape == (3, 3)
        np.testing.assert_allclose(payload["camera_to_world"], pose)
        np.testing.assert_allclose(payload["world_points"][0], [0.9, 1.9, 5.0], atol=1e-6)


def test_replacement_updates_pose_and_e_for_calibration_manager_reload(tmp_path: Path) -> None:
    source = tmp_path / "camera_calibration.json"
    source.write_text(
        json.dumps(
            {
                "cameras": {
                    "living-room": {
                        "E": np.eye(4).reshape(-1, order="F").tolist(),
                        "pose": {"source": "manual-approximate"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "replacement" / "camera_calibration.json"
    provenance = {
        "accepted_for_canonical_use": True,
        "acceptance": {"status": "passed"},
        "frame_identity": {
            "frame_id": "backend_world_m",
            "revision": "backend-r1",
            "coordinate_frame": "backend_world_m",
        },
        "assembly_frame_identity": {
            "frame_id": "assembly_world",
            "revision": "assembly-r1",
            "coordinate_frame": "pcf_assembly_metric_world_m",
        },
    }
    moved = np.eye(4, dtype=np.float64)
    moved[:3, 3] = [1.0, 2.0, 3.0]
    depth_input = tmp_path / "static_depth.npz"
    np.savez_compressed(
        depth_input,
        depth_z=np.asarray([[2.0]], dtype=np.float32),
        mask=np.asarray([[True]]),
        intrinsics=np.asarray(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    depth_output = tmp_path / "replacement" / "direct_dependencies" / "static_world_points.npz"
    replacement_result = write_calibration_replacement(
        source,
        output,
        camera_id="living-room",
        camera_to_world=moved,
        calibration_from_assembly=np.eye(4),
        provenance=provenance,
        backup_root=tmp_path / "backups",
        static_depth_input=depth_input,
        static_depth_output=depth_output,
    )
    assert replacement_result["direct_dependencies"]["depth_world_points"]["valid_point_count"] == 1
    cameras_yaml = tmp_path / "cameras.yaml"
    cameras_yaml.write_text(
        """
intrinsics_models:
  test_model:
    intrinsics:
      fx: 800.0
      fy: 800.0
      cx: 640.0
      cy: 360.0
cameras:
  0:
    name: living-room
    model: test_model
    height_m: 2.5
""",
        encoding="utf-8",
    )
    alignment = tmp_path / "ply_alignment.json"
    alignment.write_text(
        json.dumps({"matrix": np.eye(4).reshape(-1).tolist(), "floor_y": 0.0, "units": {"s_obj_to_m": 1.0}}),
        encoding="utf-8",
    )
    manager = CalibrationManager(
        cameras_yaml_path=cameras_yaml,
        camera_calibration_json_path=output,
        ply_alignment_json_path=alignment,
        streammux_size=(1280, 720),
        raw_audit_dir=tmp_path / "audit",
    )
    manager.set_camera_labels({0: "living-room"})
    snapshot = manager.snapshot(0, "living-room")
    assert snapshot is not None
    np.testing.assert_allclose(
        np.asarray(snapshot.extrinsics_col_major).reshape((4, 4), order="F"),
        np.linalg.inv(moved),
        atol=1e-6,
    )
    binding = json.loads(
        (output.parent / "direct_dependencies" / "frame_binding_revalidation.json").read_text(
            encoding="utf-8"
        )
    )
    assert binding["status"] == "prepared_frame_binding_revalidation"
    assert binding["source_frame"]["coordinate_frame"] == "pcf_assembly_metric_world_m"
    assert binding["target_frame"]["coordinate_frame"] == "backend_world_m"
    with np.load(depth_output) as direct:
        np.testing.assert_allclose(direct["camera_to_world"], moved)


def test_replacement_materializes_normal_static_revision_for_pcf_loader(tmp_path: Path) -> None:
    """The replacement output must reach the existing static-reference consumer."""
    source = tmp_path / "camera_calibration.json"
    source.write_text(
        json.dumps(
            {"cameras": {"living-room": {"E": np.eye(4).reshape(-1, order="F").tolist()}}}
        ),
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement" / "camera_calibration.json"
    depth_input = tmp_path / "static_depth.npz"
    height, width = 80, 80
    intrinsics = np.asarray(
        [[40.0, 0.0, 39.5], [0.0, 40.0, 39.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth = np.full((height, width), 2.0, dtype=np.float32)
    np.savez_compressed(depth_input, depth_z=depth, mask=np.ones_like(depth, dtype=bool), intrinsics=intrinsics)
    depth_output = tmp_path / "replacement" / "direct_dependencies" / "static_world_points.npz"
    template = tmp_path / "template_revision"
    keyframe = template / "keyframes" / "living-room_0001.png"
    keyframe.parent.mkdir(parents=True)
    assert cv2.imwrite(str(keyframe), np.full((height, width, 3), 180, dtype=np.uint8))
    identity_col_major = np.eye(4).reshape(-1, order="F").tolist()
    (template / "room_points_meta.json").write_text(
        json.dumps(
            {
                "schema": "noesis.room_reconstruction.stream_points.v4",
                "revision_id": "template-r1",
                "camera": "living-room",
                "coordinate_frame": "backend_world_m_stream_points",
                "intrinsics": intrinsics.tolist(),
                "rgb_keyframes": {"living-room_0001": "keyframes/living-room_0001.png"},
                "floor_alignment": {"world_correction_col_major": identity_col_major},
            }
        ),
        encoding="utf-8",
    )
    reference_output = tmp_path / "replacement" / "static_revision"
    frame_identity = {
        "frame_id": "backend_world_m",
        "revision": "backend-r1",
        "coordinate_frame": "backend_world_m_stream_points",
    }
    assembly_identity = {
        "frame_id": "assembly-r1",
        "revision": "assembly-revision-r1",
        "coordinate_frame": "pcf_assembly_metric_world_m",
    }
    moved = np.eye(4, dtype=np.float64)
    moved[:3, 3] = [1.0, 2.0, 3.0]
    result = write_calibration_replacement(
        source,
        replacement,
        camera_id="living-room",
        camera_to_world=moved,
        calibration_from_assembly=np.eye(4),
        provenance={
            "accepted_for_canonical_use": True,
            "acceptance": {"status": "passed"},
            "frame_identity": frame_identity,
            "assembly_frame_identity": assembly_identity,
        },
        backup_root=tmp_path / "backups",
        static_depth_input=depth_input,
        static_depth_output=depth_output,
        static_reference_template=template,
        static_reference_output=reference_output,
    )
    direct = result["direct_dependencies"]["static_reference_revision"]
    assert Path(direct["room_points"]["path"]).is_file()
    output_metadata = json.loads(Path(direct["room_points_meta"]["path"]).read_text(encoding="utf-8"))
    assert output_metadata["calibration_replacement_sha256"] == result["replacement_sha256"]
    assert output_metadata["frame_binding"]["target_frame"] == frame_identity

    from tools.mapanything_phone_scan.run_mapanything_prior_variants import _load_static_reference

    static = _load_static_reference(reference_output, replacement, "living-room")
    assert static.projected_point_count >= 5_000
    np.testing.assert_allclose(static.camera_to_world, moved, atol=1e-6)


def test_replacement_builds_isolated_v2_bundle_and_manager_binding(tmp_path: Path) -> None:
    """The replacement calibration must survive the normal WO-1 builder path."""
    pytest.importorskip("trimesh")
    identity_col_major = np.eye(4, dtype=np.float64).reshape(-1, order="F").tolist()
    source = tmp_path / "camera_calibration.json"
    source.write_text(
        json.dumps(
            {"cameras": {"living-room": {"E": identity_col_major}}}
        ),
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement" / "camera_calibration.json"
    depth_input = tmp_path / "static_depth.npz"
    height, width = 80, 80
    intrinsics = np.asarray(
        [[40.0, 0.0, 39.5], [0.0, 40.0, 39.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth = np.full((height, width), 2.0, dtype=np.float32)
    np.savez_compressed(
        depth_input,
        depth_z=depth,
        mask=np.ones_like(depth, dtype=bool),
        intrinsics=intrinsics,
    )
    depth_output = tmp_path / "replacement" / "direct_dependencies" / "static_world_points.npz"
    template = tmp_path / "template_revision"
    keyframe = template / "keyframes" / "living-room_0001.png"
    keyframe.parent.mkdir(parents=True)
    assert cv2.imwrite(str(keyframe), np.full((height, width, 3), 180, dtype=np.uint8))
    (template / "room_points_meta.json").write_text(
        json.dumps(
            {
                "schema": "noesis.room_reconstruction.stream_points.v4",
                "revision_id": "template-r1",
                "camera": "living-room",
                "coordinate_frame": "backend_world_m_stream_points",
                "intrinsics": intrinsics.tolist(),
                "rgb_keyframes": {"living-room_0001": "keyframes/living-room_0001.png"},
                "floor_alignment": {"world_correction_col_major": identity_col_major},
            }
        ),
        encoding="utf-8",
    )
    static_revision = tmp_path / "replacement" / "static_revision"
    frame_identity = {
        "frame_id": "backend_world_m",
        "revision": "static-r1",
        "coordinate_frame": "backend_world_m_stream_points",
    }
    assembly_identity = {
        "frame_id": "assembly-r1",
        "revision": "assembly-r1",
        "coordinate_frame": "pcf_assembly_metric_world_m",
    }
    moved = np.eye(4, dtype=np.float64)
    moved[:3, 3] = [1.0, 2.0, 3.0]
    replacement_result = write_calibration_replacement(
        source,
        replacement,
        camera_id="living-room",
        camera_to_world=moved,
        calibration_from_assembly=np.eye(4),
        provenance={
            "accepted_for_canonical_use": True,
            "acceptance": {"status": "passed"},
            "frame_identity": frame_identity,
            "assembly_frame_identity": assembly_identity,
        },
        backup_root=tmp_path / "backups",
        static_depth_input=depth_input,
        static_depth_output=depth_output,
        static_reference_template=template,
        static_reference_output=static_revision,
    )

    bundle = tmp_path / "bundle"
    for relative in ("aligned", "alignment", "calibration", "target/revision-a"):
        (bundle / relative).mkdir(parents=True)
    with np.load(static_revision / "room_points.npz") as points_payload:
        points = np.asarray(points_payload["points"], dtype=np.float32)
    points_glb = tmp_path / "points.glb"
    write_points_glb(points_glb, points, np.full((len(points), 3), 180, dtype=np.uint8))
    (bundle / "aligned/points.glb").write_bytes(points_glb.read_bytes())
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
        "world_from_source_row_major": np.eye(4, dtype=np.float64).tolist(),
    }
    target_metadata = json.loads(
        (static_revision / "room_points_meta.json").read_text(encoding="utf-8")
    )
    target_metadata["revision_id"] = "revision-a"
    file_payloads = {
        "aligned/points.glb": (bundle / "aligned/points.glb").read_bytes(),
        "alignment/alignment_report.json": (json.dumps(alignment_report) + "\n").encode(),
        "alignment/identity_backend_world.json": (json.dumps(transform) + "\n").encode(),
        "calibration/camera_calibration.json": replacement.read_bytes(),
        "target/revision-a/room_points_meta.json": (json.dumps(target_metadata) + "\n").encode(),
    }
    for relative, payload in file_payloads.items():
        (bundle / relative).write_bytes(payload)
    replacement_calibration_sha = hashlib.sha256(replacement.read_bytes()).hexdigest()
    source_alignment_sha = hashlib.sha256(file_payloads["alignment/identity_backend_world.json"]).hexdigest()
    file_payloads["reference.json"] = (
        json.dumps(
            {
                "schema": "noesis.reference.room_scan_bundle.v1",
                "bundle_id": "replacement-bundle-v1",
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
                    "camera_id": "living-room",
                    "camera_calibration": "calibration/camera_calibration.json",
                    "revision": "target/revision-a",
                },
                    "capture": {
                        "scan_id": "replacement-capture",
                        "captured_at": "2026-09-04T00:00:00Z",
                            "source_type": "conditioned_multimodel_room_walk",
                        "model": "fixture",
                    },
                    "created_at": "2026-09-04T00:00:01Z",
                }
        )
        + "\n"
    ).encode()
    file_payloads["bundle_manifest.json"] = (
        json.dumps(
            {
                "schema": "noesis.reference.room_scan_bundle_manifest.v1",
                "bundle_id": "replacement-bundle-v1",
                "files": [
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                    for relative, payload in sorted(file_payloads.items())
                ],
            }
        )
        + "\n"
    ).encode()
    for relative, payload in file_payloads.items():
        (bundle / relative).write_bytes(payload)
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {relative}\n"
            for relative, payload in sorted(file_payloads.items())
        ),
        encoding="utf-8",
    )

    inputs = _bundle_inputs(bundle)
    source_frame, camera_physical, alignment_physical = _source_physical_frame(inputs)
    preview = _camera_preview_frame(inputs)
    target_revision = metric_frame_revision_sha256(
        "backend_world_m",
        (0.0, 1.0, 0.0),
        0.0,
        tuple((
            ArtifactFingerprint(role="camera_calibration_physical", sha256=camera_physical),
            ArtifactFingerprint(role="world_alignment_physical", sha256=alignment_physical),
        )[index].sha256 for index in range(2)),
    )
    edge_values = tuple(identity_col_major)
    edge_digest = revisioned_transform_sha256(source_frame, RevisionedFrame("backend_world_m", target_revision), edge_values)
    acceptance_report = {
        "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
        "status": "passed",
        "accepted_for_canonical_use": True,
        "disposition": "validated_cross_session_registration",
        "holdout": {"source": "pose_graph", "status": "passed"},
        "pose_graph": {"accepted": True, "reason_codes": []},
        "bound_transform": {
            "source_frame": {"frame_id": source_frame.frame_id, "revision": source_frame.revision},
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
    provenance = (
        ArtifactFingerprint(role="camera_calibration_physical", sha256=camera_physical),
        ArtifactFingerprint(role="world_alignment_physical", sha256=alignment_physical),
    )
    binding = build_accepted_room_to_home_binding(
        artifact_revision_id="replacement-prior",
        source_frame_revision=source_frame.revision,
        target_coordinate_revision=target_revision,
        target_from_source_col_major=edge_values,
        source_floor_normal=preview.source_floor_normal,
        source_floor_offset_m=preview.source_floor_offset_m,
        target_floor_normal=(0.0, 1.0, 0.0),
        target_floor_offset_m=0.0,
        source_camera_calibration_sha256=replacement_calibration_sha,
        source_world_alignment_sha256=source_alignment_sha,
        source_camera_calibration_physical_sha256=camera_physical,
        source_world_alignment_physical_sha256=alignment_physical,
        target_revision_id="revision-a",
        target_revision_metadata_sha256=hashlib.sha256(file_payloads["target/revision-a/room_points_meta.json"]).hexdigest(),
        metric_frame_provenance=provenance,
        acceptance_report=acceptance_report,
    )
    authored_scene = Path(__file__).resolve().parents[2] / "tests/fixtures/authored_scene_oracle/simple_home.obj"
    (tmp_path / "room_group_map.json").write_text(
        json.dumps(
            {
                "contract": "noesis.authored_scene.room_group_map",
                "contract_version": 1,
                "authored_scene_sha256": hashlib.sha256(authored_scene.read_bytes()).hexdigest(),
                "rooms": {"Living Room": ["room_living"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "world_to_scene.json").write_text(
        json.dumps({"world_to_scene_col_major": identity_col_major, "floor_y": 0.0}),
        encoding="utf-8",
    )
    build_result = build_scene_prior(
        ScenePriorBuildConfig(
            source_bundle=bundle,
            site_id="replacement-site",
            space_id="living-room-replacement",
            semantic_rooms=("Living Room",),
            authored_scene=authored_scene,
            room_group_map=tmp_path / "room_group_map.json",
            world_to_scene=tmp_path / "world_to_scene.json",
            output_root=tmp_path / "scene_priors",
            camera_ids=("living-room",),
            accepted_frame_binding=binding,
            registration_acceptance_report=acceptance_report,
        )
    )
    catalog_binding = json.loads(build_result.catalog_path.read_text(encoding="utf-8"))["camera_bindings"][0]["frame_binding"]
    assert catalog_binding["contract_version"] == 2
    manager_cameras = tmp_path / "cameras.yaml"
    manager_cameras.write_text(
        """
intrinsics_models:
  test_model:
    intrinsics:
      fx: 800.0
      fy: 800.0
      cx: 640.0
      cy: 360.0
cameras:
  0:
    name: living-room
    model: test_model
""",
        encoding="utf-8",
    )
    manager_alignment = tmp_path / "ply_alignment.json"
    manager_alignment.write_text(
        json.dumps({"matrix": identity_col_major, "floor_y": 0.0, "units": {"s_obj_to_m": 1.0}}),
        encoding="utf-8",
    )
    manager = CalibrationManager(
        manager_cameras,
        replacement,
        manager_alignment,
        streammux_size=(1280, 720),
        raw_audit_dir=tmp_path / "audit",
        scene_prior_frame_bindings={
            "living-room": ScenePriorFrameBinding.model_validate(catalog_binding)
        },
    )
    manager.set_camera_labels({0: "living-room"})
    assert manager.frame_contract("living-room").target_frame.revision == target_revision
    snapshot = manager.snapshot(0, "living-room")
    assert snapshot is not None
    np.testing.assert_allclose(
        np.asarray(snapshot.extrinsics_col_major).reshape((4, 4), order="F"),
        np.linalg.inv(moved),
        atol=1e-6,
    )
    assert replacement_result["active_source_mutated"] is False
