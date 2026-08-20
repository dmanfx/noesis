from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi.testclient import TestClient

from tools.mapanything_phone_scan.app import create_app
from tools.mapanything_phone_scan.inference import MapAnythingScanSettings
from tools.mapanything_phone_scan.processing import FramePreparationSettings
from tools.mapanything_phone_scan.supplement import (
    SupplementIntegrationSettings,
    _estimate_bridge_similarity,
)
from tools.mapanything_phone_scan.test_phone_scan import _settings


def _yaw(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )


def test_duplicate_bridge_poses_recover_append_to_base_similarity() -> None:
    append = []
    for index, center in enumerate(
        ([0.0, 0.0, 0.0], [0.8, 0.1, 0.2], [1.4, -0.1, 0.9], [0.5, 0.05, 1.5])
    ):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _yaw(index * 7.0)
        pose[:3, 3] = center
        append.append(pose)
    append_poses = np.stack(append)
    expected_scale = 1.08
    expected_rotation = _yaw(23.0)
    expected_translation = np.asarray([2.4, -0.3, 1.1], dtype=np.float64)
    base_poses = append_poses.copy()
    base_poses[:, :3, :3] = expected_rotation[None, :, :] @ append_poses[:, :3, :3]
    base_poses[:, :3, 3] = (
        expected_scale * (expected_rotation @ append_poses[:, :3, 3].T).T
        + expected_translation
    )

    scale, rotation, translation, metrics = _estimate_bridge_similarity(
        base_poses,
        append_poses,
        SupplementIntegrationSettings(),
    )

    assert abs(scale - expected_scale) < 1e-8
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-8)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-8)
    assert all(metrics["checks"].values())


def _wait_for_scan(client: TestClient, scan_id: str, status: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        payload = client.get(f"/api/scans/{scan_id}").json()
        if payload.get("status") == status:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"scan did not reach {status}")


def _wait_for_supplement(
    client: TestClient, scan_id: str, supplement_id: str, status: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        payload = client.get(f"/api/scans/{scan_id}").json()
        rows = {row["id"]: row for row in payload.get("supplements") or []}
        if rows.get(supplement_id, {}).get("status") == status:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"added video did not reach {status}")


def test_additional_video_api_prepares_integrates_and_deletes_revision(
    tmp_path: Path,
) -> None:
    def fake_prepare(
        video_path: Path,
        work_dir: Path,
        settings: FramePreparationSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        frames = work_dir / "frames"
        thumbnails = work_dir / "frame_thumbnails"
        frames.mkdir()
        thumbnails.mkdir()
        rows = []
        for index in range(2):
            frame = frames / f"frame_{index:04d}.jpg"
            thumbnail = thumbnails / f"frame_{index:04d}.jpg"
            frame.write_bytes(f"frame-{index}".encode())
            thumbnail.write_bytes(f"thumb-{index}".encode())
            rows.append(
                {
                    "index": index,
                    "timestamp_s": float(index),
                    "frame": frame.relative_to(work_dir).as_posix(),
                    "thumbnail": thumbnail.relative_to(work_dir).as_posix(),
                    "quality": {"warnings": []},
                }
            )
        contact = work_dir / "prepared_frames_contact_sheet.jpg"
        manifest = work_dir / "prepared_frames_manifest.json"
        contact.write_bytes(b"contact")
        manifest.write_text("{}", encoding="utf-8")
        progress(1.0, "Prepared")
        assert video_path.is_file()
        return {
            "probe": {"duration_s": 2.0, "codec": "test"},
            "frame_count": 2,
            "effective_fps": 1.0,
            "quality_warning_counts": {},
            "frames": rows,
            "contact_sheet": contact.name,
            "manifest": manifest.name,
            "preparation_limit": settings.max_selected_frames,
        }

    def fake_inference(
        _: Path,
        output_dir: Path,
        prepared: dict[str, Any],
        __: MapAnythingScanSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        raw = output_dir / "raw"
        raw.mkdir()
        artifact = output_dir / "reconstruction_points.glb"
        manifest = output_dir / "scan_outputs_manifest.json"
        artifact.write_bytes(b"base-glb")
        manifest.write_text("{}", encoding="utf-8")
        frames = []
        for index, row in enumerate(prepared["frames"]):
            raw_path = raw / f"view_{index:04d}.npz"
            raw_path.write_bytes(b"raw")
            frames.append(
                {
                    "index": index,
                    "source_frame": row["frame"],
                    "raw_npz": f"outputs/raw/{raw_path.name}",
                }
            )
        progress(1.0, "Complete")
        return {
            "provider": "mapanything",
            "view_count": len(frames),
            "review_point_count": 2,
            "scale": {"median": 1.0},
            "artifacts": {
                "reconstruction_glb": "outputs/reconstruction_points.glb",
                "manifest": "outputs/scan_outputs_manifest.json",
            },
            "frames": frames,
            "files": [],
        }

    def fake_supplement(
        _: Path,
        __: Path,
        output_dir: Path,
        state: dict[str, Any],
        supplement: dict[str, Any],
        provider: str,
        ___: Callable[..., dict[str, Any]],
        ____: Any,
        settings: SupplementIntegrationSettings,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        assert provider == "mapanything"
        assert supplement["prepared"]["preparation_limit"] == settings.new_view_limit
        output_dir.mkdir()
        merged = output_dir / "merged_reconstruction.glb"
        report = output_dir / "integration_report.json"
        merged.write_bytes(b"merged-glb")
        report.write_text("{}", encoding="utf-8")
        prefix = f"supplements/{supplement['id']}/revision"
        progress(1.0, "Merged")
        return {
            "schema": "noesis.phone_scan.supplement.revision.v1",
            "revision_id": f"{state['id']}:add:{supplement['id']}",
            "parent_revision_id": f"{state['id']}:base",
            "supplement_id": supplement["id"],
            "generated_at": "2026-08-15T12:00:00+00:00",
            "provider": provider,
            "coordinate_frame": "original_phone_reconstruction_base",
            "fusion": {"point_count": 123, "new_only_voxel_count": 9},
            "artifacts": {
                "merged_reconstruction_glb": f"{prefix}/{merged.name}",
                "report": f"{prefix}/{report.name}",
            },
            "files": [],
        }

    app = create_app(
        _settings(tmp_path),
        frame_processor=fake_prepare,
        inference_runner=fake_inference,
        supplement_runner=fake_supplement,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/scans?name=Kitchen",
            content=b"base-video",
            headers={"Content-Type": "video/mp4", "X-File-Name": "base.mp4"},
        )
        scan_id = created.json()["id"]
        _wait_for_scan(client, scan_id, "ready")
        assert client.post(f"/api/scans/{scan_id}/initiate-ma").status_code == 202
        _wait_for_scan(client, scan_id, "complete")

        uploaded = client.post(
            f"/api/scans/{scan_id}/supplements",
            content=b"additional-video",
            headers={"Content-Type": "video/mp4", "X-File-Name": "second.mp4"},
        )
        assert uploaded.status_code == 201
        supplement_id = uploaded.json()["supplements"][0]["id"]
        ready = _wait_for_supplement(client, scan_id, supplement_id, "ready")
        addition = ready["supplements"][0]
        assert addition["video"]["url"].startswith("/assets/")
        assert addition["prepared"]["contact_sheet_url"].startswith("/assets/")

        started = client.post(
            f"/api/scans/{scan_id}/supplements/{supplement_id}/integrate"
        )
        assert started.status_code == 202
        complete = _wait_for_supplement(client, scan_id, supplement_id, "complete")
        active = complete["active_revision"]
        assert active["supplement_id"] == supplement_id
        merged_url = active["artifact_urls"]["merged_reconstruction_glb"]
        assert client.get(merged_url).content == b"merged-glb"

        deleted = client.delete(
            f"/api/scans/{scan_id}/supplements/{supplement_id}"
        )
        assert deleted.status_code == 204
        after = client.get(f"/api/scans/{scan_id}").json()
        assert after.get("supplements") == []
        assert "active_revision" not in after
