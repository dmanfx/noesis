from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from fastapi.testclient import TestClient

from tools.mapanything_phone_scan.alignment import NoesisAlignmentSettings
from tools.mapanything_phone_scan.app import APP_ROOT, REPO_ROOT, PhoneScanSettings, create_app
from tools.mapanything_phone_scan.da3_inference import DA3PhoneScanSettings
from tools.mapanything_phone_scan.inference import MapAnythingScanSettings
from tools.mapanything_phone_scan.processing import (
    FramePreparationSettings,
    prepare_video_frames,
)


def _settings(tmp_path: Path) -> PhoneScanSettings:
    return PhoneScanSettings(
        storage_root=tmp_path / "scans",
        static_root=APP_ROOT / "static",
        three_root=REPO_ROOT / "oai2-fe" / "node_modules" / "three",
        max_upload_bytes=10 * 1024 * 1024,
        frame=FramePreparationSettings(target_fps=2.0, max_frames=12, max_edge_px=640),
        mapanything=MapAnythingScanSettings(point_budget=10_000),
        da3=DA3PhoneScanSettings(
            point_budget=10_000,
            metric_engine_path=tmp_path / "da3metric.engine",
        ),
        alignment=NoesisAlignmentSettings(
            camera_id="living-room",
            target_revision=tmp_path / "target-revision",
            calibration_path=tmp_path / "camera_calibration.json",
            review_point_budget=10_000,
        ),
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
        FramePreparationSettings(target_fps=2.0, max_frames=12, max_edge_px=640),
        lambda fraction, message: progress_rows.append((fraction, message)),
    )

    assert 5 <= result["frame_count"] <= 7
    assert result["frames"][0]["timestamp_s"] == 0.0
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
        assert settings.camera_id == "living-room"
        progress(0.5, "Aligning test room")
        comparison = output_dir / "noesis_phone_comparison.glb"
        report = output_dir / "alignment_report.json"
        comparison.write_bytes(b"comparison")
        report.write_text("{}", encoding="utf-8")
        progress(1.0, "Aligned")
        return {
            "schema": "noesis.mapanything.phone_scan.alignment_outputs.v1",
            "coordinate_frame": "backend_world_m_stream_points",
            "target_camera_id": "living-room",
            "target_revision_id": "test-revision",
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

        initiated = client.post(f"/api/scans/{scan_id}/initiate-ma")
        assert initiated.status_code == 202
        complete = _wait_for_status(client, scan_id, "complete")
        glb_url = complete["outputs"]["artifact_urls"]["reconstruction_glb"]
        assert client.get(glb_url).status_code == 200
        assert complete["outputs"]["frames"][0]["raw_npz_url"].startswith("/assets/")

        alignment_started = client.post(f"/api/scans/{scan_id}/align-noesis")
        assert alignment_started.status_code == 202
        aligned = _wait_for_alignment(client, scan_id, "complete")
        assert aligned["alignment"]["results"]["quality_gate"]["passed"] is True
        comparison_url = aligned["alignment"]["results"]["artifact_urls"]["comparison_glb"]
        assert client.get(comparison_url).status_code == 200

        deleted = client.delete(f"/api/scans/{scan_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/scans/{scan_id}").status_code == 404


def test_scan_id_path_traversal_is_rejected(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/scans/not-a-scan").status_code == 404
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
