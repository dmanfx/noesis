from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tools.mapanything_phone_scan.alignment import (
    NoesisAlignmentError,
    NoesisAlignmentSettings,
    _fixed_camera_comparable_mask,
    _fixed_camera_visible_cloud_metrics,
    _resolve_target_camera_orientation,
    _resolve_target_cloud_for_calibrated_camera,
)
from tools.mapanything_phone_scan.app import APP_ROOT, REPO_ROOT, PhoneScanSettings, create_app
from tools.mapanything_phone_scan.da3_inference import DA3PhoneScanSettings
from tools.mapanything_phone_scan.inference import MapAnythingScanSettings
from tools.mapanything_phone_scan.processing import (
    FramePreparationSettings,
    prepare_video_frames,
)


def _settings(tmp_path: Path) -> PhoneScanSettings:
    living_room_alignment = NoesisAlignmentSettings(
        camera_id="living-room",
        target_revision=tmp_path / "living-room-revision",
        calibration_path=tmp_path / "camera_calibration.json",
        review_point_budget=10_000,
    )
    kitchen_alignment = NoesisAlignmentSettings(
        camera_id="kitchen",
        target_revision=tmp_path / "kitchen-revision",
        calibration_path=tmp_path / "camera_calibration.json",
        review_point_budget=10_000,
    )
    family_room_alignment = NoesisAlignmentSettings(
        camera_id="family-room",
        target_revision=tmp_path / "family-room-revision",
        calibration_path=tmp_path / "camera_calibration.json",
        review_point_budget=10_000,
    )
    return PhoneScanSettings(
        storage_root=tmp_path / "scans",
        static_root=APP_ROOT / "static",
        three_root=REPO_ROOT / "oai2-fe" / "node_modules" / "three",
        max_upload_bytes=10 * 1024 * 1024,
        frame=FramePreparationSettings(
            candidate_fps=4.0,
            max_candidate_frames=48,
            max_selected_frames=24,
            candidate_edge_px=640,
            feature_edge_px=640,
            max_edge_px=640,
        ),
        mapanything=MapAnythingScanSettings(point_budget=10_000),
        da3=DA3PhoneScanSettings(
            point_budget=10_000,
            metric_engine_path=tmp_path / "da3metric.engine",
        ),
        alignment=living_room_alignment,
        alignment_targets=(
            family_room_alignment,
            kitchen_alignment,
            living_room_alignment,
        ),
        alignment_release_id="test-home-release",
        pcf_storage_root=tmp_path / "pcf",
    )


def _wait_for_status(client: TestClient, scan_id: str, expected: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/scans/{scan_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"scan {scan_id} did not reach {expected}")


def _wait_for_alignment(client: TestClient, scan_id: str, expected: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/scans/{scan_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload.get("alignment", {}).get("status") == expected:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"scan {scan_id} alignment did not reach {expected}")


def _wait_for_pcf(client: TestClient, scan_id: str, expected: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/scans/{scan_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload.get("pcf", {}).get("status") == expected:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"scan {scan_id} PCF did not reach {expected}")


def test_whole_home_review_assembly_is_release_bound_and_digest_verified(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assembly_id = "pcf-home-review"
    assembly_dir = settings.pcf_storage_root / "review-assemblies" / assembly_id
    assembly_dir.mkdir(parents=True)
    artifact = assembly_dir / "multiroom_points.glb"
    artifact.write_bytes(b"review-glb-bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "contract": "noesis.scene.review_assembly",
        "contract_version": 3,
        "assembly_id": assembly_id,
        "created_at_us": 1,
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "coordinate_frame": "backend_world_m",
        "assembly_gauge": "family_accepted_backend_world_m",
        "units": "meters",
        "scene_binding": {"release_id": "test-home-release"},
        "artifact": {
            "role": "multiroom_points_glb",
            "relative_path": (
                f"review-assemblies/{assembly_id}/multiroom_points.glb"
            ),
            "sha256": digest,
            "size_bytes": artifact.stat().st_size,
            "media_type": "model/gltf-binary",
            "point_count": 3,
            "owner_point_counts": {"1": 1, "2": 1, "3": 1},
        },
        "camera_anchor": {
            "status": "review_pose_estimate",
            "accepted_for_canonical_use": False,
            "camera_id": "family-room",
            "coordinate_frame": "family_accepted_backend_world_m",
            "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
            "anchor_mode": "floor_locked_planar",
            "camera_height_source": "admitted_scene_prior_reference_camera",
            "vertical_anchor_translation_m": 0.0,
            "camera_to_assembly_col_major": [
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                14,
                2,
                11,
                1,
            ],
            "device_reference_camera_to_assembly_col_major": [
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                15,
                2,
                13,
                1,
            ],
            "camera_center_assembly_m": [14, 2, 11],
            "source_report_sha256": "a" * 64,
        },
        "camera_markers": {
            "status": "review_camera_positions",
            "accepted_for_canonical_use": False,
            "coordinate_frame": "family_accepted_backend_world_m",
            "method": "static_camera_centers_composed_through_recorded_room_transforms",
            "sphere_radius_m": 0.18,
            "color_hex": "#ffd400",
            "markers": [
                {
                    "camera_id": "family-room",
                    "position_assembly_m": [14, 2, 11],
                },
                {
                    "camera_id": "kitchen",
                    "position_assembly_m": [9, 2.2, 5],
                },
                {
                    "camera_id": "living-room",
                    "position_assembly_m": [3, 1.9, 8],
                },
            ],
            "source_report_sha256": "b" * 64,
        },
        "provenance": {
            "source_camera_anchor_report_sha256": "a" * 64,
            "source_camera_markers_report_sha256": "b" * 64,
        },
    }
    current = settings.pcf_storage_root / "review-assemblies" / "current.json"
    current.write_text(json.dumps(manifest), encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        descriptor = client.get(
            "/api/v1/scenes/current/review-assemblies/whole-home"
        )
        assert descriptor.status_code == 200
        assert descriptor.json()["assembly_id"] == assembly_id
        assert descriptor.json()["artifact_url"].endswith(
            "/artifacts/multiroom_points_glb"
        )
        response = client.get(descriptor.json()["artifact_url"])
        assert response.status_code == 200
        assert response.content == artifact.read_bytes()
        assert response.headers["x-noesis-artifact-sha256"] == digest

        missing_role = client.get(
            "/api/v1/scenes/current/review-assemblies/whole-home/"
            "artifacts/multiroom_surface_mesh_glb"
        )
        assert missing_role.status_code == 404

        mesh_artifact = assembly_dir / "multiroom_surface_mesh.glb"
        mesh_artifact.write_bytes(b"review-mesh-glb-bytes")
        mesh_digest = hashlib.sha256(mesh_artifact.read_bytes()).hexdigest()
        point_artifact = manifest["artifact"]
        manifest["contract_version"] = 4
        manifest["artifact"] = {
            "role": "multiroom_surface_mesh_glb",
            "relative_path": (
                f"review-assemblies/{assembly_id}/multiroom_surface_mesh.glb"
            ),
            "sha256": mesh_digest,
            "size_bytes": mesh_artifact.stat().st_size,
            "media_type": "model/gltf-binary",
            "vertex_count": 12,
            "triangle_count": 18,
            "owner_triangle_counts": {"1": 6, "2": 6, "3": 6},
        }
        current.write_text(json.dumps(manifest), encoding="utf-8")
        mesh_descriptor = client.get(
            "/api/v1/scenes/current/review-assemblies/whole-home"
        )
        assert mesh_descriptor.status_code == 200
        assert mesh_descriptor.json()["artifact_url"].endswith(
            "/artifacts/multiroom_surface_mesh_glb"
        )
        mesh_response = client.get(mesh_descriptor.json()["artifact_url"])
        assert mesh_response.status_code == 200
        assert mesh_response.content == mesh_artifact.read_bytes()
        manifest["contract_version"] = 3
        manifest["artifact"] = point_artifact

        manifest["camera_markers"]["markers"][2]["camera_id"] = "kitchen"
        current.write_text(json.dumps(manifest), encoding="utf-8")
        bad_markers = client.get(
            "/api/v1/scenes/current/review-assemblies/whole-home"
        )
        assert bad_markers.status_code == 409
        manifest["camera_markers"]["markers"][2]["camera_id"] = "living-room"

        manifest["scene_binding"]["release_id"] = "wrong-release"
        current.write_text(json.dumps(manifest), encoding="utf-8")
        rejected = client.get(
            "/api/v1/scenes/current/review-assemblies/whole-home"
        )
        assert rejected.status_code == 409


def test_target_camera_orientation_preserves_forward_facing_pose() -> None:
    target_points = np.asarray(
        [[-1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [1.0, 0.5, 4.0]],
        dtype=np.float64,
    )

    camera_to_world, metrics = _resolve_target_camera_orientation(
        target_points,
        np.eye(4, dtype=np.float64),
    )

    np.testing.assert_allclose(camera_to_world, np.eye(4))
    assert metrics["local_yaw_correction_deg"] == 0.0
    assert metrics["selected_forward_fraction"] == 1.0


def test_target_camera_orientation_rejects_decisive_half_turn() -> None:
    target_points = np.asarray(
        [[-1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [1.0, 0.5, 4.0]],
        dtype=np.float64,
    )
    reversed_camera_to_world = np.eye(4, dtype=np.float64)
    reversed_camera_to_world[:3, :3] = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )

    with pytest.raises(NoesisAlignmentError, match="correct the camera calibration"):
        _resolve_target_camera_orientation(
            target_points,
            reversed_camera_to_world,
        )


def test_target_cloud_rejects_half_turn_instead_of_rotating_geometry() -> None:
    target_points = np.asarray(
        [[-1.0, 0.2, 2.0], [0.0, 1.0, 3.0], [1.0, 0.5, 4.0]],
        dtype=np.float64,
    )
    reversed_camera_to_world = np.eye(4, dtype=np.float64)
    reversed_camera_to_world[:3, :3] = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )

    with pytest.raises(NoesisAlignmentError, match="correct the camera calibration"):
        _resolve_target_cloud_for_calibrated_camera(
            target_points,
            reversed_camera_to_world,
        )


def test_fixed_camera_visibility_excludes_only_occluded_source_points() -> None:
    intrinsics = np.asarray(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target_depth = np.full((10, 10), np.inf, dtype=np.float64)
    target_depth[5, 5] = 2.0
    source = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.2], [0.0, 0.0, 2.5]],
        dtype=np.float64,
    )

    comparable, metrics = _fixed_camera_comparable_mask(
        source,
        target_depth,
        np.eye(4, dtype=np.float64),
        intrinsics,
        cell_px=1,
        occlusion_tolerance_m=0.30,
    )

    assert comparable.tolist() == [True, True, False]
    assert metrics["target_supported_point_count"] == 3.0
    assert metrics["comparable_point_count"] == 2.0


def test_fixed_camera_visibility_keeps_bad_foreground_in_score() -> None:
    intrinsics = np.asarray(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
    target_depth = np.full((10, 10), np.inf, dtype=np.float64)
    target_depth[5, 5] = 2.0
    source = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.1], [0.0, 0.0, 2.5]],
        dtype=np.float64,
    )

    metrics = _fixed_camera_visible_cloud_metrics(
        source,
        target,
        target_depth,
        np.eye(4, dtype=np.float64),
        intrinsics,
        cell_px=1,
        occlusion_tolerance_m=0.30,
    )

    assert metrics["comparable_point_count"] == 2.0
    assert metrics["source_overlap_0_30m"] == 0.5


def test_prepare_video_frames_preserves_full_walk_coverage(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    video = scan_dir / "phone_walk.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (640, 360),
    )
    assert writer.isOpened()
    for index in range(30):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = 40 + index * 3
        cv2.rectangle(frame, (20 + index * 7, 80), (180 + index * 7, 280), (220, 180, 80), -1)
        cv2.putText(frame, str(index), (260, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()

    progress_rows: list[tuple[float, str]] = []
    result = prepare_video_frames(
        video,
        scan_dir,
        FramePreparationSettings(
            candidate_fps=4.0,
            max_candidate_frames=48,
            max_selected_frames=24,
            candidate_edge_px=640,
            feature_edge_px=640,
            max_edge_px=640,
        ),
        lambda fraction, message: progress_rows.append((fraction, message)),
    )

    assert 3 <= result["frame_count"] <= 10
    assert result["candidate_count"] >= result["frame_count"]
    assert result["selection"]["policy"] == "adaptive_quality_motion_overlap_connectivity_v1"
    assert result["selection"]["selection_limited"] is False
    assert result["frames"][0]["timestamp_s"] <= 0.5
    assert result["frames"][-1]["timestamp_s"] >= 2.0
    assert (scan_dir / result["manifest"]).is_file()
    assert (scan_dir / result["contact_sheet"]).is_file()
    assert progress_rows[-1][0] == 1.0


def test_phone_scan_api_upload_initiate_review_and_delete(tmp_path: Path) -> None:
    def fake_prepare(
        video_path: Path,
        scan_dir: Path,
        _: FramePreparationSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        progress(0.5, "Preparing test views")
        frames_dir = scan_dir / "frames"
        thumbs_dir = scan_dir / "frame_thumbnails"
        frames_dir.mkdir()
        thumbs_dir.mkdir()
        rows = []
        for index in range(2):
            frame = frames_dir / f"frame_{index:04d}.jpg"
            thumb = thumbs_dir / f"frame_{index:04d}.jpg"
            frame.write_bytes(b"frame")
            thumb.write_bytes(b"thumb")
            rows.append(
                {
                    "index": index,
                    "timestamp_s": float(index),
                    "frame": str(frame.relative_to(scan_dir)),
                    "thumbnail": str(thumb.relative_to(scan_dir)),
                    "quality": {"warnings": []},
                }
            )
        contact = scan_dir / "prepared_frames_contact_sheet.jpg"
        manifest = scan_dir / "prepared_frames_manifest.json"
        contact.write_bytes(b"contact")
        manifest.write_text("{}", encoding="utf-8")
        assert video_path.is_file()
        progress(1.0, "Prepared")
        return {
            "probe": {"duration_s": 2.0, "codec": "test"},
            "frame_count": 2,
            "effective_fps": 1.0,
            "quality_warning_counts": {},
            "frames": rows,
            "contact_sheet": contact.name,
            "manifest": manifest.name,
        }

    def fake_inference(
        _: Path,
        output_dir: Path,
        prepared: dict[str, Any],
        __: MapAnythingScanSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        assert prepared["frame_count"] == 2
        progress(0.5, "Saving test reconstruction")
        views = output_dir / "views"
        raw = output_dir / "raw"
        views.mkdir()
        raw.mkdir()
        artifacts = {
            "reconstruction_glb": "outputs/reconstruction_points.glb",
            "trajectory_preview": "outputs/camera_trajectory_topdown.png",
            "trajectory_json": "outputs/camera_trajectory.json",
            "camera_solution_npz": "outputs/camera_solution.npz",
            "manifest": "outputs/scan_outputs_manifest.json",
        }
        for relative in (
            "reconstruction_points.glb",
            "camera_trajectory_topdown.png",
            "camera_trajectory.json",
            "camera_solution.npz",
            "scan_outputs_manifest.json",
        ):
            (output_dir / relative).write_bytes(relative.encode())
        frame_rows = []
        for index in range(2):
            paths = {}
            for kind in ("rgb", "depth", "confidence", "mask"):
                path = views / f"view_{index:04d}_{kind}.png"
                path.write_bytes(kind.encode())
                paths[kind] = f"outputs/views/{path.name}"
            raw_path = raw / f"view_{index:04d}.npz"
            raw_path.write_bytes(b"raw")
            frame_rows.append(
                {
                    "index": index,
                    "timestamp_s": float(index),
                    "source_frame": f"frames/frame_{index:04d}.jpg",
                    "model_rgb": paths["rgb"],
                    "depth_preview": paths["depth"],
                    "confidence_preview": paths["confidence"],
                    "mask_preview": paths["mask"],
                    "raw_npz": f"outputs/raw/{raw_path.name}",
                    "depth": {"valid_fraction": 1.0, "p50_m": 2.0},
                }
            )
        progress(1.0, "Complete")
        return {
            "view_count": 2,
            "review_point_count": 100,
            "scale": {"median": 1.0},
            "artifacts": artifacts,
            "frames": frame_rows,
            "files": [
                {"path": "outputs/reconstruction_points.glb", "size_bytes": 5}
            ],
        }

    def fake_alignment(
        _: Path,
        output_dir: Path,
        outputs: dict[str, Any],
        settings: NoesisAlignmentSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        assert outputs["view_count"] == 2
        assert settings.camera_id == "family-room"
        progress(0.5, "Aligning test room")
        comparison = output_dir / "noesis_phone_comparison.glb"
        report = output_dir / "alignment_report.json"
        comparison.write_bytes(b"comparison")
        report.write_text("{}", encoding="utf-8")
        progress(1.0, "Aligned")
        return {
            "schema": "noesis.mapanything.phone_scan.alignment_outputs.v1",
            "coordinate_frame": "backend_world_m_stream_points",
            "target_camera_id": settings.camera_id,
            "target_revision_id": settings.target_revision.name,
            "quality_gate": {"passed": True, "candidate_objective_margin": 0.2},
            "vertical_structure": {
                "source_overlap_0_30m": 0.8,
                "target_overlap_0_30m": 0.7,
                "plane_residual_median_m": 0.04,
            },
            "full_cloud": {
                "source_overlap_0_30m": 0.75,
                "target_overlap_0_30m": 0.6,
            },
            "artifacts": {
                "comparison_glb": "alignment/noesis_phone_comparison.glb",
                "report": "alignment/alignment_report.json",
            },
            "files": [
                {
                    "path": "alignment/noesis_phone_comparison.glb",
                    "size_bytes": comparison.stat().st_size,
                },
                {
                    "path": "alignment/alignment_report.json",
                    "size_bytes": report.stat().st_size,
                },
            ],
        }

    app = create_app(
        _settings(tmp_path),
        frame_processor=fake_prepare,
        inference_runner=fake_inference,
        alignment_runner=fake_alignment,
    )
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["alignment_release_id"] == "test-home-release"
        assert [
            target["camera_id"] for target in health.json()["alignment_targets"]
        ] == ["family-room", "kitchen", "living-room"]
        created = client.post(
            "/api/scans?name=Living%20room%20walk",
            content=b"phone video bytes",
            headers={"Content-Type": "video/mp4", "X-File-Name": "walk.mp4"},
        )
        assert created.status_code == 201
        scan_id = created.json()["id"]
        ready = _wait_for_status(client, scan_id, "ready")
        assert ready["prepared"]["frame_count"] == 2
        assert ready["prepared"]["frames"][0]["thumbnail_url"].startswith("/assets/")

        blank_name = client.patch(f"/api/scans/{scan_id}?name=%20%20%20")
        assert blank_name.status_code == 400
        renamed = client.patch(
            f"/api/scans/{scan_id}?name=Living%20Room%20Daylight"
        )
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "Living Room Daylight"
        assert client.get(f"/api/scans/{scan_id}").json()["name"] == "Living Room Daylight"
        assert client.get("/api/scans").json()[0]["name"] == "Living Room Daylight"

        initiated = client.post(f"/api/scans/{scan_id}/initiate-ma")
        assert initiated.status_code == 202
        complete = _wait_for_status(client, scan_id, "complete")
        assert complete["name"] == "Living Room Daylight"
        glb_url = complete["outputs"]["artifact_urls"]["reconstruction_glb"]
        assert client.get(glb_url).status_code == 200
        assert complete["outputs"]["frames"][0]["raw_npz_url"].startswith("/assets/")

        assert client.post(f"/api/scans/{scan_id}/align-noesis").status_code == 422
        assert client.post(
            f"/api/scans/{scan_id}/align-noesis?camera_id=garage"
        ).status_code == 422
        alignment_started = client.post(
            f"/api/scans/{scan_id}/align-noesis?camera_id=family-room"
        )
        assert alignment_started.status_code == 202
        assert alignment_started.json()["alignment"]["target_camera_id"] == "family-room"
        assert alignment_started.json()["alignment"]["target_revision_id"] == "family-room-revision"
        assert alignment_started.json()["alignment"]["target_release_id"] == "test-home-release"
        aligned = _wait_for_alignment(client, scan_id, "complete")
        assert aligned["alignment"]["results"]["quality_gate"]["passed"] is True
        assert aligned["alignment"]["results"]["target_camera_id"] == "family-room"
        comparison_url = aligned["alignment"]["results"]["artifact_urls"]["comparison_glb"]
        assert client.get(comparison_url).status_code == 200

        deleted = client.delete(f"/api/scans/{scan_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/scans/{scan_id}").status_code == 404


def test_scan_id_path_traversal_is_rejected(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/scans/not-a-scan").status_code == 404
        assert client.patch("/api/scans/not-a-scan?name=Kitchen").status_code == 404
        assert client.delete("/api/scans/not-a-scan").status_code == 404


def test_phone_scan_da3_provider_selector_uses_da3_runner(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_da3(
        _: Path,
        output_dir: Path,
        prepared: dict[str, Any],
        settings: DA3PhoneScanSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        calls.append(settings.model_id)
        assert prepared["frame_count"] == 2
        progress(0.5, "DA3 test inference")
        artifact = output_dir / "reconstruction_points.glb"
        artifact.write_bytes(b"da3")
        progress(1.0, "DA3 complete")
        return {
            "provider": "da3",
            "view_count": 2,
            "review_point_count": 1,
            "scale": {"median": 1.0},
            "artifacts": {"reconstruction_glb": "outputs/reconstruction_points.glb"},
            "frames": [],
            "files": [
                {
                    "path": "outputs/reconstruction_points.glb",
                    "size_bytes": artifact.stat().st_size,
                }
            ],
        }

    app = create_app(_settings(tmp_path), da3_inference_runner=fake_da3)
    service = app.state.phone_scan_service
    scan_id = "20260808-120000-1234abcd"
    scan_dir = service.scan_dir(scan_id)
    scan_dir.mkdir(parents=True)
    service._write_state_unlocked(
        scan_id,
        {
            "schema": "noesis.phone_scan.state.v2",
            "id": scan_id,
            "name": "DA3 selector test",
            "created_at": "2026-08-08T12:00:00+00:00",
            "updated_at": "2026-08-08T12:00:00+00:00",
            "status": "ready",
            "progress": 1.0,
            "message": "ready",
            "error": None,
            "video": {"path": "phone_walk.mp4", "size_bytes": 1},
            "prepared": {"frame_count": 2, "frames": [{}, {}]},
        },
    )
    with TestClient(app) as client:
        response = client.post(
            f"/api/scans/{scan_id}/initiate-inference?provider=da3"
        )
        assert response.status_code == 202
        assert response.json()["provider"] == "da3"
        complete = _wait_for_status(client, scan_id, "complete")
        assert complete["outputs"]["provider"] == "da3"
        assert client.get(
            complete["outputs"]["artifact_urls"]["reconstruction_glb"]
        ).status_code == 200
    assert calls == ["depth-anything/DA3-BASE"]


def test_aligned_da3_walk_can_build_saved_pcf_review(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_pcf(
        scan_dir: Path,
        run_root: Path,
        state: dict[str, Any],
        target: NoesisAlignmentSettings,
        _: MapAnythingScanSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        calls.append((str(state["id"]), target.camera_id))
        assert scan_dir.name == state["id"]
        progress(0.5, "Building fake PCF")
        consensus = run_root / "prior_conditioned_consensus_da3_carrier"
        evaluation = run_root / "evaluation_static_world" / "prior_conditioned_consensus"
        consensus.mkdir(parents=True)
        evaluation.mkdir(parents=True)
        glb = consensus / "consensus_surfel_reconstruction.glb"
        diagnostics = evaluation / "static_world_heatmap_diagnostics.png"
        log = run_root / "pcf_run.log"
        glb.write_bytes(b"pcf")
        diagnostics.write_bytes(b"png")
        log.write_text("complete\n", encoding="utf-8")
        progress(1.0, "Fake PCF complete")
        return {
            "schema": "noesis.phone_scan.pcf_review.v1",
            "method": "prior_conditioned_consensus_da3_carrier",
            "review_only": True,
            "published_to_scene_prior": False,
            "coordinate_frame": "backend_world_m_stream_points",
            "target_camera_id": target.camera_id,
            "target_revision_id": target.target_revision.name,
            "view_count": 12,
            "fusion": {"agreement_fraction_of_both": 0.9},
            "surfel_fusion": {"surfel_count": 1234},
            "multiview_consistency": {"consensus": {"p80_error_m": 0.05}},
            "heldout_even_to_odd_reprojection": {
                "consensus": {
                    "even_frame_map_to_odd_frame_depth_median_m": 0.04,
                    "odd_frame_valid_pixel_coverage_fraction": 0.8,
                }
            },
            "evaluation": {
                "fixed_camera_visible_cloud_metrics": {
                    "source_overlap_0_30m": 0.75
                }
            },
            "artifacts": {
                "pcf_glb": glb.relative_to(run_root).as_posix(),
                "diagnostic_layers": diagnostics.relative_to(run_root).as_posix(),
                "run_log": log.relative_to(run_root).as_posix(),
            },
            "files": [
                {
                    "path": glb.relative_to(run_root).as_posix(),
                    "size_bytes": glb.stat().st_size,
                }
            ],
        }

    app = create_app(
        replace(_settings(tmp_path), pcf_pause_appliance=True),
        pcf_runner=fake_pcf,
    )
    service = app.state.phone_scan_service
    appliance_active = True
    runtime_calls: list[tuple[str, str]] = []

    def fake_unit_active(unit: str) -> bool:
        return appliance_active if unit in {
            "menon-appliance.target",
            "noesis-appliance.service",
        } else False

    def fake_systemctl(action: str, unit: str, *, timeout_s: int) -> None:
        nonlocal appliance_active
        assert timeout_s > 0
        runtime_calls.append((action, unit))
        appliance_active = action == "start"

    service._user_unit_active = fake_unit_active
    service._systemctl_user = fake_systemctl
    scan_id = "20260816-010000-1234abcd"
    scan_dir = service.scan_dir(scan_id)
    scan_dir.mkdir(parents=True)
    service._write_state_unlocked(
        scan_id,
        {
            "schema": "noesis.phone_scan.state.v2",
            "id": scan_id,
            "name": "Aligned DA3 PCF test",
            "created_at": "2026-08-16T01:00:00+00:00",
            "updated_at": "2026-08-16T01:00:00+00:00",
            "status": "complete",
            "progress": 1.0,
            "message": "complete",
            "error": None,
            "provider": "da3",
            "video": {"path": "phone_walk.mp4", "size_bytes": 1},
            "prepared": {"frame_count": 12, "frames": []},
            "outputs": {"provider": "da3", "view_count": 12},
            "alignment": {
                "status": "complete",
                "target_camera_id": "family-room",
                "target_revision_id": "family-room-revision",
                "results": {
                    "target_camera_id": "family-room",
                    "target_revision_id": "family-room-revision",
                    "quality_gate": {"passed": True},
                },
            },
        },
    )

    with TestClient(app) as client:
        started = client.post(f"/api/scans/{scan_id}/initiate-pcf")
        assert started.status_code == 202
        assert started.json()["pcf"]["source_scope"] == "original_prepared_walk"
        complete = _wait_for_pcf(client, scan_id, "complete")
        assert complete["pcf"]["results"]["review_only"] is True
        runtime_lease = complete["pcf"]["results"]["runtime_lease"]
        assert runtime_lease["appliance_paused"] is True
        assert runtime_lease["restore_passed"] is True
        glb_url = complete["pcf"]["results"]["artifact_urls"]["pcf_glb"]
        assert glb_url.startswith(f"/pcf-assets/{scan_id}/runs/pcf-")
        assert client.get(glb_url).content == b"pcf"
        assert client.post(f"/api/scans/{scan_id}/initiate-pcf").status_code == 409
        pcf_root = service.pcf_scan_dir(scan_id)
        assert pcf_root.is_dir()
        assert client.delete(f"/api/scans/{scan_id}").status_code == 204
        assert not pcf_root.exists()

    assert calls == [(scan_id, "family-room")]
    assert runtime_calls == [
        ("stop", "menon-appliance.target"),
        ("start", "menon-appliance.target"),
    ]


def test_interrupted_pcf_restores_recorded_appliance_lease(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    service = app.state.phone_scan_service
    scan_id = "20260816-020000-1234abcd"
    scan_dir = service.scan_dir(scan_id)
    scan_dir.mkdir(parents=True)
    service._write_state_unlocked(
        scan_id,
        {
            "schema": "noesis.phone_scan.state.v2",
            "id": scan_id,
            "name": "Interrupted PCF test",
            "created_at": "2026-08-16T02:00:00+00:00",
            "updated_at": "2026-08-16T02:00:00+00:00",
            "status": "complete",
            "progress": 1.0,
            "message": "complete",
            "error": None,
            "provider": "da3",
            "pcf": {
                "run_id": "pcf-20260816-020000-1234abcd",
                "status": "running",
                "runtime_lease": {
                    "appliance_target": "menon-appliance.target",
                    "appliance_was_active": True,
                    "pause_requested": True,
                    "appliance_paused": True,
                    "restore_passed": None,
                },
            },
        },
    )
    runtime_calls: list[tuple[str, str]] = []
    service._systemctl_user = lambda action, unit, *, timeout_s: runtime_calls.append(
        (action, unit)
    )
    service._user_unit_active = lambda unit: unit == "noesis-appliance.service"

    service.recover_interrupted_states()

    recovered = service.read_state(scan_id)["pcf"]
    assert recovered["status"] == "failed"
    assert recovered["runtime_lease"]["restore_passed"] is True
    assert recovered["runtime_lease"]["recovered_after_tool_restart"] is True
    assert runtime_calls == [("start", "menon-appliance.target")]
