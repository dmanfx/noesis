from __future__ import annotations

import textwrap
import time
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import noesis.metadata.intrinsics as intrinsics
from noesis.metadata.intrinsics import CameraConfigLoader, attach_intrinsics, load_cameras_yaml


@pytest.fixture(autouse=True)
def _reset_intrinsics_globals():
    intrinsics._LOGGED_SOURCES.clear()
    intrinsics._MISSING_SOURCES.clear()
    intrinsics._META_TYPE_ID = None
    if intrinsics._GLOBAL_LOADER is not None:
        intrinsics._GLOBAL_LOADER.invalidate()
    yield


def _write_config(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_load_cameras_yaml_handles_models(tmp_path):
    config_path = tmp_path / "cameras.yaml"
    _write_config(
        config_path,
        """
        cameras:
          0:
            intrinsics:
              fx: 640
              fy: 645
              cx: 320
              cy: 240
              distortion_coeffs: [0.01, -0.02, 0.003]
              height_m: 2.5
          1:
            model: g4
            height_m: 3.0
        intrinsics_models:
          g4:
            resolution: [640, 480]
            intrinsics:
              K_matrix:
                - [500, 0, 320]
                - [0, 505, 240]
                - [0, 0, 1]
              distortion:
                k1: 0.001
                k2: -0.002
        """,
    )

    cameras = load_cameras_yaml(config_path)

    assert cameras[0].fx == pytest.approx(640.0)
    assert cameras[0].k2 == pytest.approx(-0.02)
    assert cameras[0].height_m == pytest.approx(2.5)

    assert cameras[1].fx == pytest.approx(500.0)
    assert cameras[1].fy == pytest.approx(505.0)
    assert cameras[1].cx == pytest.approx(320.0)
    assert cameras[1].cy == pytest.approx(240.0)
    assert cameras[1].k1 == pytest.approx(0.001)
    assert cameras[1].k2 == pytest.approx(-0.002)
    assert cameras[1].height_m == pytest.approx(3.0)
    assert cameras[1].width == 640
    assert cameras[1].height == 480


def test_camera_config_loader_refreshes_on_change(tmp_path):
    config_path = tmp_path / "cameras.yaml"
    _write_config(
        config_path,
        """
        cameras:
          0:
            intrinsics:
              fx: 100
              fy: 110
              cx: 50
              cy: 55
        """,
    )
    loader = CameraConfigLoader(config_path)

    assert loader.get(0).fx == pytest.approx(100.0)

    time.sleep(0.01)
    _write_config(
        config_path,
        """
        cameras:
          0:
            intrinsics:
              fx: 200
              fy: 210
              cx: 90
              cy: 95
        """,
    )

    assert loader.get(0).fx == pytest.approx(200.0)
    assert loader.get(0).cx == pytest.approx(90.0)


def test_attach_intrinsics_to_mapping(tmp_path):
    config_path = tmp_path / "cameras.yaml"
    _write_config(
        config_path,
        """
        cameras:
          0:
            intrinsics:
              fx: 300
              fy: 301
              cx: 150
              cy: 151
              k1: 0.01
              k2: -0.02
              k3: 0.03
              height_m: 2.75
        """,
    )
    loader = CameraConfigLoader(config_path)
    frame_meta: dict[str, object] = {}

    attach_intrinsics(frame_meta, 0, loader=loader)
    first_payload = frame_meta["user_meta"]["intrinsics"]  # type: ignore[index]
    assert first_payload["fx"] == pytest.approx(300.0)  # type: ignore[index]
    assert first_payload["height_m"] == pytest.approx(2.75)  # type: ignore[index]

    first_payload["fx"] = 999  # type: ignore[index]
    attach_intrinsics(frame_meta, 0, loader=loader)
    second_payload = frame_meta["user_meta"]["intrinsics"]  # type: ignore[index]

    assert first_payload is not second_payload
    assert second_payload["fx"] == pytest.approx(300.0)  # type: ignore[index]
    assert second_payload["k2"] == pytest.approx(-0.02)  # type: ignore[index]

    missing_frame: dict[str, object] = {}
    attach_intrinsics(missing_frame, 9, loader=loader)
    assert "user_meta" not in missing_frame
