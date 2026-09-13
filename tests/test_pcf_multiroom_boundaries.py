from __future__ import annotations

import numpy as np
import json
from argparse import Namespace
from pathlib import Path
from hashlib import sha256

from tools.mapanything_phone_scan.build_pcf_review_surface_mesh import (
    build,
    _triangle_support_mask,
)
from tools.mapanything_phone_scan.extend_pcf_multiroom_with_connector import (
    _append_source,
)
from tools.mapanything_phone_scan.publish_pcf_review_surface_mesh import publish
from tools.mapanything_phone_scan.reintegrate_pcf_rooms import (
    ReintegrationSettings,
    RoomFusion,
    RoomSource,
    ViewEvidence,
    _aggregate_view,
    _compose_output,
    _depth_visibility_evidence,
    _record_dtype,
    _three_dimensional_compatibility,
)


def _record(points: list[tuple[float, float, float]], *, views: int = 3) -> np.ndarray:
    result = np.zeros(len(points), dtype=_record_dtype(1))
    values = np.asarray(points, dtype=np.float64)
    result["key"] = np.floor(values / 0.025).astype(np.int32)
    result["point_weighted_sum"] = values
    result["color_weighted_sum"] = 128.0
    result["weight_sum"] = 1.0
    result["confidence_sum"] = 1.0
    result["observation_count"] = views
    result["agreement_observation_count"] = views
    result["agreement_weight_sum"] = 1.0
    result["source_selection_counts"][:, 1] = views
    result["view_mask_words"] = np.uint64((1 << views) - 1)
    return result


def _view_evidence() -> list[ViewEvidence]:
    return [
        ViewEvidence(
            view_index=0,
            intrinsics=np.asarray(
                [[10.0, 0.0, 32.0], [0.0, 10.0, 32.0], [0.0, 0.0, 1.0]]
            ),
            depth_z=np.full((64, 64), 2.0, dtype=np.float32),
            mask=np.ones((64, 64), dtype=bool),
            camera_to_world=np.eye(4, dtype=np.float64),
        )
    ]


def _fusion(name: str, records: np.ndarray) -> RoomFusion:
    return RoomFusion(
        source=RoomSource(name=name, raw_root=None, world_manifest=None),  # type: ignore[arg-type]
        records=records,
        camera_positions=np.asarray([[-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]),
        view_files=[],
        counters={},
        view_count=2,
        view_mask_words=1,
        view_evidence=_view_evidence(),
    )


def test_same_xz_different_height_is_retained_but_3d_overlap_is_suppressed() -> None:
    fixed = _fusion("family", _record([(0.0, 0.0, 2.0)]))
    moving = _fusion(
        "kitchen",
        _record([(0.0, 0.0, 2.0), (0.0, 0.03, 2.0), (0.0, 0.50, 2.0)]),
    )
    output, suppressed, ownership = _compose_output(
        fixed,
        moving,
        ReintegrationSettings(ownership_dilation_m=0.0),
        None,
    )

    owners = output["owner_room_id"]
    output_points = output["points"]
    assert np.count_nonzero(owners == 1) == 1
    assert np.count_nonzero(owners == 2) == 1
    assert np.min(np.linalg.norm(output_points - [0.0, 0.50, 2.0], axis=1)) < 1e-6
    assert ownership["moving_xz_authority_candidate_count"] == 3
    assert ownership["moving_3d_compatible_voxel_count"] == 2
    assert ownership["moving_complementary_height_retained_count"] == 1
    assert ownership["moving_suppressed_by_3d_compatibility_count"] == 1
    assert len(suppressed["points"]) == 2
    assert np.all(suppressed["suppression_reason_mask"] & 4)


def test_extension_uses_3d_compatibility_inside_buffered_columns() -> None:
    existing_records = _record([(0.0, 0.0, 2.0)])
    existing = _fusion("existing", existing_records)
    from tools.mapanything_phone_scan.extend_pcf_multiroom_with_connector import (
        _record_payload,
    )

    output = _record_payload(existing_records)
    output.update(
        {
            "owner_room_id": np.ones(1, dtype=np.uint8),
            "room_contribution_mask": np.ones(1, dtype=np.uint8),
            "room_presence_mask": np.ones(1, dtype=np.uint8),
        }
    )
    source_records = _record([(0.0, 0.03, 2.0), (0.0, 0.50, 2.0)])
    source = _record_payload(source_records)
    report = _append_source(
        output,
        source,
        owner_id=3,
        room_bit=4,
        ownership_cell_m=0.10,
        authority_buffer_m=0.75,
        existing_view_evidence=existing.view_evidence,
        source_view_evidence=existing.view_evidence,
    )
    assert report["suppressed_by_3d_compatibility_count"] == 1
    assert report["complementary_height_retained_count"] == 1
    assert np.count_nonzero(output["owner_room_id"] == 3) == 1
    assert np.min(
        np.linalg.norm(output["points"] - [0.0, 0.50, 2.0], axis=1)
    ) < 1e-6


def test_agreement_is_reported_without_a_second_confidence_bonus() -> None:
    common = dict(
        points=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        colors=np.full((2, 3), 128, dtype=np.uint8),
        confidence=np.asarray([0.5, 0.5]),
        disagreement=np.full(2, np.nan),
        map_reliability=np.full(2, np.nan),
        da3_reliability=np.full(2, np.nan),
        source_selection=np.ones(2, dtype=np.uint8),
        view_index=0,
        view_mask_words=1,
        settings=ReintegrationSettings(),
    )
    result = _aggregate_view(
        agreement=np.asarray([False, True]),
        **common,
    )
    assert np.allclose(result["weight_sum"], [0.5, 0.5])


def test_rgbd_projection_distinguishes_surface_free_space_and_unknown() -> None:
    supported, unknown, contradiction = _depth_visibility_evidence(
        np.asarray(
            [
                [0.0, 0.0, 2.0],  # measured surface
                [0.0, 0.0, 1.0],  # in front of measured depth: free space
                [0.0, 0.0, 3.0],  # behind measured depth: occluded
                [0.0, 0.0, -1.0],  # behind camera: unknown
            ],
            dtype=np.float64,
        ),
        _view_evidence(),
        minimum_depth_m=0.05,
        maximum_depth_m=12.0,
        absolute_tolerance_m=0.01,
        relative_tolerance=0.0,
    )
    assert supported.tolist() == [True, False, False, False]
    assert contradiction.tolist() == [False, True, False, False]
    assert unknown.tolist() == [False, False, True, True]


def test_cross_room_ray_marks_free_space_contradiction_but_occlusion_unknown() -> None:
    fixed_view = _view_evidence()[0]
    moving_view = ViewEvidence(
        view_index=0,
        intrinsics=fixed_view.intrinsics,
        depth_z=np.full((64, 64), 1.8, dtype=np.float32),
        mask=fixed_view.mask,
        camera_to_world=fixed_view.camera_to_world,
    )
    (
        compatible,
        source_valid,
        fixed_valid,
        nearest,
        cross_free,
        cross_unknown,
        source_own_contradiction,
        fixed_own_contradiction,
        source_own_unknown,
        fixed_own_unknown,
    ) = (
        _three_dimensional_compatibility(
            np.asarray([[0.0, 0.0, 2.0]]),
            np.asarray([[0.0, 0.0, 1.8], [0.0, 0.0, 3.0]]),
            [fixed_view],
            [moving_view],
            distance_m=0.06,
            minimum_depth_m=0.05,
            maximum_depth_m=12.0,
            depth_tolerance_m=0.01,
            depth_tolerance_fraction=0.0,
        )
    )
    assert source_valid.tolist() == [True, False]
    assert fixed_valid.tolist() == [True]
    assert np.allclose(nearest, [0.2, 1.0])
    assert compatible.tolist() == [False, False]
    assert cross_free.tolist() == [True, False]
    assert cross_unknown.tolist() == [False, True]
    assert source_own_contradiction.tolist() == [False, False]
    assert fixed_own_contradiction.tolist() == [False]
    assert source_own_unknown.tolist() == [False, True]
    assert fixed_own_unknown.tolist() == [False]


def test_near_cross_ray_contradiction_is_retained_as_uncertain() -> None:
    intrinsics = np.asarray(
        [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]
    )
    fixed_depth = np.full((64, 64), 2.0, dtype=np.float32)
    # A nearby source point projects to a different retained pixel whose
    # measured surface is 0.2 m farther away, exceeding the RGB-D tolerance.
    fixed_depth[32, 34] = 2.2
    fixed = _fusion("family", _record([(0.0, 0.0, 2.0)]))
    fixed.view_evidence = [
        ViewEvidence(
            view_index=0,
            intrinsics=intrinsics,
            depth_z=fixed_depth,
            mask=np.ones((64, 64), dtype=bool),
            camera_to_world=np.eye(4),
        )
    ]
    moving = _fusion("kitchen", _record([(0.04, 0.0, 2.0)]))
    moving.view_evidence = [
        ViewEvidence(
            view_index=0,
            intrinsics=intrinsics,
            depth_z=np.full((64, 64), 2.0, dtype=np.float32),
            mask=np.ones((64, 64), dtype=bool),
            camera_to_world=np.eye(4),
        )
    ]
    output, _, ownership = _compose_output(
        fixed,
        moving,
        ReintegrationSettings(ownership_dilation_m=0.0),
        None,
    )
    moving_rows = output["owner_room_id"] == 2
    assert np.count_nonzero(moving_rows) == 1
    assert output["geometry_status"][moving_rows].tolist() == [1]
    assert ownership["moving_3d_compatible_voxel_count"] == 0


def test_triangle_centroid_support_rejects_an_opening_bridge() -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        triangles=o3d.utility.Vector3iVector([[0, 1, 2]]),
    )
    keep, unsupported, ray_rejected = _triangle_support_mask(
        mesh,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        _view_evidence(),
        0.1,
    )
    assert not bool(keep[0])
    assert unsupported == 1
    assert ray_rejected == 0


def test_mesh_builder_carries_support_classes_and_frame_identity(tmp_path: Path) -> None:
    x, z = np.meshgrid(np.linspace(-1.0, 1.0, 45), np.linspace(-1.0, 1.0, 45))
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel() + 2.0)).astype(
        np.float32
    )
    npz_path = tmp_path / "surfels.npz"
    np.savez_compressed(
        npz_path,
        points=points,
        colors=np.full((len(points), 3), 180, dtype=np.uint8),
        owner_room_id=np.ones(len(points), dtype=np.uint8),
        view_count=np.where(np.arange(len(points)) == 0, 1, 2).astype(np.uint16),
        fixed_camera_positions=np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32),
    )
    raw_path = tmp_path / "view_0000.npz"
    np.savez_compressed(
        raw_path,
        depth_z=np.full((64, 64), 2.0, dtype=np.float32),
        mask=np.ones((64, 64), dtype=bool),
        intrinsics=np.asarray(
            [[20.0, 0.0, 32.0], [0.0, 20.0, 32.0], [0.0, 0.0, 1.0]]
        ),
        camera_pose=np.eye(4, dtype=np.float32),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "accepted_for_canonical_use": False,
                "output_coordinate_frame": "family_accepted_backend_world_m",
                "registration": {"status": "rejected"},
                "fixed_room": {
                    "raw_views": [
                        {
                            "view_index": 0,
                            "path": str(raw_path),
                            "effective_world_from_local_row_major": np.eye(4).tolist(),
                        }
                    ]
                },
                "artifacts": {
                    "surfels_npz": {"sha256": sha256(npz_path.read_bytes()).hexdigest()}
                },
            }
        ),
        encoding="utf-8",
    )
    report = build(
        Namespace(
            surfels_npz=npz_path,
            surfels_manifest=manifest_path,
            output_glb=tmp_path / "mesh.glb",
            output_manifest=tmp_path / "mesh_manifest.json",
            voxel_size_m=0.04,
            normal_radius_m=0.14,
            support_distance_m=0.10,
            ceiling_cutaway_m=1.85,
            poisson_depth=5,
            density_percentile=2.0,
            minimum_component_triangles=2,
            poisson_threads=1,
        )
    )
    assert report["frame_identity"]["coordinate_frame"] == (
        "family_accepted_backend_world_m"
    )
    assert report["parameters"]["triangle_support_and_camera_ray_checks"] is True
    assert report["owners"]["1"]["support_class_counts"]["observed"] == len(points) - 1
    assert report["owners"]["1"]["support_class_counts"]["uncertain"] == 1
    assert report["owners"]["1"]["classes"]["uncertain"]["geometry_mode"] == (
        "review_points"
    )


def test_surface_publisher_carries_support_contract(tmp_path: Path) -> None:
    storage = tmp_path / "pcf"
    storage.mkdir()
    (storage / "review-assemblies").mkdir()
    source_bytes = b"source-points"
    source_path = storage / "source.glb"
    source_path.write_bytes(source_bytes)
    source_hash = sha256(source_bytes).hexdigest()
    current_path = tmp_path / "current.json"
    current_path.write_text(
        json.dumps(
            {
                "contract": "noesis.scene.review_assembly",
                "contract_version": 3,
                "assembly_id": "whole-home",
                "status": "review_only",
                "accepted_for_canonical_use": False,
                "artifact": {
                    "role": "multiroom_points_glb",
                    "relative_path": "source.glb",
                    "sha256": source_hash,
                },
                "provenance": {
                    "source_reintegration_manifest_sha256": "m" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    mesh_bytes = b"surface-mesh"
    mesh_path = tmp_path / "mesh.glb"
    mesh_path.write_bytes(mesh_bytes)
    mesh_hash = sha256(mesh_bytes).hexdigest()
    mesh_manifest = tmp_path / "mesh.json"
    mesh_manifest.write_text(
        json.dumps(
            {
                "contract": "noesis.pcf.review_surface_mesh",
                "status": "review_only",
                "accepted_for_canonical_use": False,
                "frame_identity": {"coordinate_frame": "family_backend_world_m"},
                "source": {
                    "surfels_manifest_sha256": "m" * 64,
                },
                "owners": {
                    "1": {
                        "output_triangle_count": 4,
                        "support_class_counts": {
                            "observed": 7,
                            "uncertain": 2,
                            "unknown": 1,
                        },
                    }
                },
                "output": {
                    "sha256": mesh_hash,
                    "size_bytes": len(mesh_bytes),
                    "vertex_count": 8,
                    "triangle_count": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    structural = tmp_path / "structural.json"
    structural.write_text(
        json.dumps(
            {
                "contract": "noesis.pcf.menon_structural_alignment",
                "assembly_id": "whole-home",
                "source": {"review_artifact_sha256": source_hash},
            }
        ),
        encoding="utf-8",
    )
    output_structural = tmp_path / "structural-out.json"
    descriptor = publish(
        Namespace(
            pcf_storage_root=storage,
            current_descriptor=current_path,
            mesh_glb=mesh_path,
            mesh_manifest=mesh_manifest,
            source_structural_alignment=structural,
            structural_output=[output_structural],
            assembly_id="surface-review",
            activate=False,
        )
    )
    published = json.loads(Path(descriptor["descriptor"]).read_text(encoding="utf-8"))
    support = published["artifact"]["support"]
    assert support["coordinate_frame"] == "family_backend_world_m"
    assert support["class_counts"] == {"observed": 7, "uncertain": 2, "unknown": 1}
    assert support["triangle_support_checks"] is True
    assert support["ray_visibility_checks"] is True
