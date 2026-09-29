from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from noesis.diagnostics.v3dt_forensics import analyze_tracking_log, build_snapshot
from noesis.v3dt_raster import scale_projection_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
SANITY_SCRIPT = REPO_ROOT / "scripts" / "sanity_check_v3dt_calibration.py"
SANITY_SPEC = importlib.util.spec_from_file_location(
    "v3dt_sanity_raster_diagnostics_test", SANITY_SCRIPT
)
assert SANITY_SPEC is not None and SANITY_SPEC.loader is not None
sanity = importlib.util.module_from_spec(SANITY_SPEC)
sys.modules[SANITY_SPEC.name] = sanity
SANITY_SPEC.loader.exec_module(sanity)


def test_sanity_checker_normalizes_only_explicit_tracker_projection() -> None:
    mux_size = (1920, 1080)
    tracker_size = (1920, 1088)
    projection_mux = np.array(
        [[900.0, 2.0, 960.0, 10.0], [0.0, 850.0, 540.0, 20.0], [0.0, 0.0, 1.0, 0.0]]
    )
    projection_tracker = scale_projection_matrix(
        projection_mux, mux_size, tracker_size
    )
    tracker_caminfo = {
        "noesis_frame_binding": {
            "projection_pixel_space": "tracker",
            "image_size": list(mux_size),
            "tracker_image_size": list(tracker_size),
        }
    }
    tracker_pipeline = {
        "v3dt": {"profile": "sv3dt", "caminfo_pixel_space": "tracker"},
        "streammux": {"width": mux_size[0], "height": mux_size[1]},
        "tracker": {"tracker-width": tracker_size[0], "tracker-height": tracker_size[1]},
    }

    normalized, space, declared_tracker_size = sanity.projection_in_mux_pixels(
        projection_tracker,
        "projectionMatrix_3x4_w2p",
        tracker_caminfo,
        tracker_pipeline,
    )
    assert space == "tracker"
    assert declared_tracker_size == tracker_size
    np.testing.assert_allclose(normalized, projection_mux)
    with pytest.raises(ValueError, match="requires noesis_frame_binding"):
        sanity.projection_in_mux_pixels(
            projection_tracker,
            "projectionMatrix_3x4_w2p",
            {},
            tracker_pipeline,
        )

    unchanged, space, no_tracker_size = sanity.projection_in_mux_pixels(
        projection_mux, "projectionMatrix_3x4_w2p", {}, {}
    )
    assert space == "mux"
    assert no_tracker_size is None
    np.testing.assert_array_equal(unchanged, projection_mux)


def _write_tracker_snapshot_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path, np.ndarray]:
    mux_size = (1280, 720)
    tracker_size = (640, 360)
    cameras_path = root / "cameras.yaml"
    cameras_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "intrinsics_models": {
                    "test_model": {
                        "resolution": list(mux_size),
                        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
                    }
                },
                "cameras": {0: {"name": "test-cam", "model": "test_model", "height_m": 2.0}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    camera_center = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, 3] = -camera_center
    calibration_path = root / "camera_calibration.json"
    calibration_path.write_text(
        json.dumps({"cameras": {"test-cam": {"E": extrinsics.flatten(order="F").tolist()}}}),
        encoding="utf-8",
    )

    alignment_path = root / "ply_alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                "floor_y": 0.0,
                "units": {"s_obj_to_m": 1.0},
            }
        ),
        encoding="utf-8",
    )

    pipeline_path = root / "pipeline.yaml"
    pipeline_path.write_text(
        yaml.safe_dump(
            {
                "streammux": {"width": mux_size[0], "height": mux_size[1]},
                "tracker": {"tracker-width": tracker_size[0], "tracker-height": tracker_size[1]},
                "v3dt": {"profile": "sv3dt", "caminfo_pixel_space": "tracker"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # The forensics tool's default axis binding is xzy.
    axis_map = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    axis4 = np.eye(4, dtype=np.float64)
    axis4[:3, :3] = axis_map
    intrinsics = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    projection_mux = intrinsics @ (extrinsics @ axis4)[:3, :]
    projection_tracker = scale_projection_matrix(projection_mux, mux_size, tracker_size)
    caminfo_dir = root / "caminfo"
    caminfo_dir.mkdir()
    (caminfo_dir / "camInfo_test-cam.yml").write_text(
        yaml.safe_dump(
            {
                "projectionMatrix_3x4_w2p": projection_tracker.flatten().tolist(),
                "modelInfo": {"height": 1.7, "radius": 0.35},
                "noesis_frame_binding": {
                    "projection_pixel_space": "tracker",
                    "image_size": list(mux_size),
                    "tracker_image_size": list(tracker_size),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return (
        pipeline_path,
        cameras_path,
        calibration_path,
        caminfo_dir,
        alignment_path,
        projection_mux,
    )


def test_forensics_snapshot_preserves_tracker_projection_and_exposes_mux_projection(
    tmp_path: Path,
) -> None:
    pipeline, cameras, calibration, caminfo, alignment, projection_mux = (
        _write_tracker_snapshot_inputs(tmp_path)
    )
    snapshot = build_snapshot(
        pipeline_path=pipeline,
        cameras_path=cameras,
        calibration_path=calibration,
        caminfo_dir=caminfo,
        alignment_path=alignment,
    )

    camera = snapshot["cameras"]["test-cam"]["caminfo"]
    assert snapshot["pipeline"]["caminfo_pixel_space"] == "tracker"
    assert camera["projection_pixel_space"] == "tracker"
    assert camera["image_size"] == [1280, 720]
    assert camera["tracker_image_size"] == [640, 360]
    np.testing.assert_allclose(
        np.asarray(camera["projectionMatrix_mux"]).reshape(3, 4), projection_mux
    )
    assert camera["diff_to_expected"] < 1e-8


def test_forensics_snapshot_reports_malformed_tracker_binding_without_comparing_p(
    tmp_path: Path,
) -> None:
    pipeline, cameras, calibration, caminfo_dir, alignment, _ = (
        _write_tracker_snapshot_inputs(tmp_path)
    )
    caminfo_path = caminfo_dir / "camInfo_test-cam.yml"
    caminfo = yaml.safe_load(caminfo_path.read_text(encoding="utf-8"))
    caminfo["noesis_frame_binding"] = "malformed-binding"
    caminfo_path.write_text(yaml.safe_dump(caminfo, sort_keys=False), encoding="utf-8")

    snapshot = build_snapshot(
        pipeline_path=pipeline,
        cameras_path=cameras,
        calibration_path=calibration,
        caminfo_dir=caminfo_dir,
        alignment_path=alignment,
    )

    camera = snapshot["cameras"]["test-cam"]["caminfo"]
    assert camera["image_size"] is None
    assert camera["tracker_image_size"] is None
    assert camera["projectionMatrix_mux"] is None
    assert camera["diff_to_expected"] is None
    assert any(
        issue["code"] == "caminfo_pixel_space_binding"
        for issue in snapshot["issues"]
    )


@pytest.mark.parametrize("include_mux_projection", [False, True])
def test_forensics_analysis_projects_public_boxes_with_mux_pixels(
    tmp_path: Path, include_mux_projection: bool
) -> None:
    mux_size = (1920, 1080)
    tracker_size = (1920, 1088)
    projection_mux = np.array(
        [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    projection_tracker = scale_projection_matrix(
        projection_mux, mux_size, tracker_size
    )
    caminfo = {
        "type": "projectionMatrix_3x4_w2p",
        "projectionMatrix": projection_tracker.flatten().tolist(),
        "projection_pixel_space": "tracker",
        "configured_pixel_space": "tracker",
        "image_size": list(mux_size),
        "tracker_image_size": list(tracker_size),
    }
    if include_mux_projection:
        caminfo["projectionMatrix_mux"] = projection_mux.flatten().tolist()
    snapshot = {
        "pipeline": {
            "streammux": {"width": mux_size[0], "height": mux_size[1]},
            "tracker": {"width": tracker_size[0], "height": tracker_size[1]},
            "caminfo_pixel_space": "tracker",
            "v3dt_profile": "sv3dt",
        },
        "cameras": {
            "test-cam": {
                "caminfo": caminfo,
                "intrinsics_scaled": {"fy": 100.0},
            }
        },
    }

    x, y, z, z_len = 1.0, 2.0, 10.0, 2.0
    u = (100.0 * x + 50.0 * 9.0) / 9.0
    v = (100.0 * y + 40.0 * 9.0) / 9.0
    width = height = 20.0
    log_path = tmp_path / "tracks.ndjson"
    log_path.write_text(
        json.dumps(
            {
                "type": "v3dt_tracking_frame",
                "camera_id": "test-cam",
                "frame_id": 1,
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 0,
                        "bbox": [u - width / 2.0, v - height, width, height],
                        "bbox3d": {
                            "xCentre": x,
                            "yCentre": y,
                            "zCentre": z,
                            "xLen": 0.5,
                            "yLen": 0.5,
                            "zLen": z_len,
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze_tracking_log(log_path, snapshot=snapshot)
    cam = report["cameras"]["test-cam"]
    assert cam["reprojection_error_px"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert not any(item["code"] == "caminfo_pixel_space_binding" for item in report["warnings"])
