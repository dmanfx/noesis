"""Pixel-ground-truth tests independent of any phone camera fit or holdout."""
import cv2
import shutil
import numpy as np
import pytest
from scipy.special import erf

from .corner_refinement import POLICY, _edge_intersection, refine_native_corners
from .roomwalk_calibration import DEFAULT_BOARD, _board_object


@pytest.mark.parametrize("angle", [0, 0.4, 1.0])
@pytest.mark.parametrize("shear", [-0.2, 0, 0.2])
@pytest.mark.parametrize("gap", [0, 6, 12])
@pytest.mark.parametrize("sigma", [1, 3, 6])
def test_opposing_edges_known_geometry_under_blur_rotation_and_noise(angle, shear, gap, sigma):
    rng = np.random.default_rng(43)
    axes = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    axes = axes @ np.array([[1, shear], [0, 1.]])
    axes /= np.linalg.norm(axes, axis=0)
    y, x = np.mgrid[:193, :193]
    uv = np.stack((x - 96.25, y - 96.7), -1) @ np.linalg.inv(axes).T
    u, v = uv[:, :, 0], uv[:, :, 1]
    cdf = lambda z: (1 + erf(z / (sigma * 2**0.5))) / 2
    dark = cdf(u-gap/2)*cdf(v-gap/2) + cdf(-u-gap/2)*cdf(-v-gap/2)
    image = np.clip(220 - 180*dark + 0.03*x + rng.normal(0, 0.6, x.shape), 0, 255).astype(np.uint8)
    seed = np.array([92., 94.])
    approximate_axes = axes @ np.array([[np.cos(0.03), -np.sin(0.03)], [np.sin(0.03), np.cos(0.03)]])
    for _ in range(3):
        seed, evidence = _edge_intersection(image, seed, approximate_axes, 40)
        assert seed is not None, evidence
    # +0.5 is the ChArUco coordinate convention, not a fitted camera offset.
    assert np.linalg.norm(seed - [96.75, 97.2]) < 0.15
    assert not evidence["reason_codes"]


def board_image():
    board = _board_object(DEFAULT_BOARD)
    image = board.generateImage((1200, 1680))
    # Shrink black quadrants symmetrically, then soften edges. Marker IDs and
    # ideal checker intersection coordinates remain known independently.
    image = cv2.GaussianBlur(cv2.dilate(image, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
    return board, cv2.copyMakeBorder(image, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=255)


def test_complete_detector_and_refiner_keep_original_ids_and_known_grid():
    board, image = board_image()
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, markers, _ = cv2.aruco.CharucoDetector(board, detectorParams=params).detectBoard(image)
    initial = corners.copy()
    points, evidence, failures = refine_native_corners(image, corners, ids, markers, 10)
    assert len(points) == len(ids) >= 100
    assert not failures
    truth = board.getChessboardCorners()[ids.ravel(), :2] * (120 / 0.018) + 64
    assert np.max(np.linalg.norm(points - truth, axis=1)) < 0.2
    assert evidence["policy"] == POLICY
    np.testing.assert_array_equal(corners, initial)
    np.testing.assert_array_equal(evidence["initial_points"], initial.reshape(-1, 2))
    assert all(12 <= c["support_radius_px"] <= 64 for c in evidence["corners"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires native ffmpeg")
def test_decoder_refinement_hull_consumer_integration(tmp_path):
    from .roomwalk_calibration import _detect_capture, _Run, CalibrationSettings
    _, image = board_image()
    source = tmp_path / "known-board.png"
    assert cv2.imwrite(str(source), image)
    run = _Run(CalibrationSettings(timeout_s=30), None, None)
    observations = _detect_capture(source, [image.shape[1], image.shape[0]], [0], [0],
                                   DEFAULT_BOARD, run, tmp_path)
    row = observations["frames"][0]
    assert observations["corner_refinement_policy"] == POLICY
    assert not row["rejection_reasons"]
    assert len(row["ids"]) >= 100
    assert row["hull_fraction"] > 0.5
    assert row["corner_refinement"]["policy"] == POLICY
    assert observations["decode_command"][observations["decode_command"].index("-frames:v")+1] == "1"


def test_blank_or_fully_masked_patch_cannot_supply_a_corner():
    seed = np.array([96., 96.])
    point, evidence = _edge_intersection(np.zeros((193, 193), np.uint8), seed, np.eye(2), 40)
    assert point is None
    assert evidence["reason_codes"] == ["weak_or_masked_checker_edge"]
    image = np.zeros((193, 193), np.uint8)
    image[:96, :96] = image[96:, 96:] = 255
    point, evidence = _edge_intersection(image, seed, np.eye(2), 40, np.zeros_like(image))
    assert point is None
    assert evidence["reason_codes"] == ["weak_or_masked_checker_edge"]


def test_failed_corner_rejects_view_and_retains_all_evidence(monkeypatch):
    from . import corner_refinement as module
    board, image = board_image()
    corners, ids, markers, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)

    def unsafe_point(_gray, seed, *_args):
        return seed + 100, {"reason_codes": []}

    monkeypatch.setattr(module, "_edge_intersection", unsafe_point)
    points, evidence, failures = refine_native_corners(image, corners, ids, markers, 10)
    assert failures == ["native_corner_refinement_failed"]
    assert len(points) == len(ids) == len(evidence["corners"])
    assert all(e["reason_codes"] == ["refinement_displacement_exceeded"] for e in evidence["corners"])
    np.testing.assert_array_equal(points, corners.reshape(-1, 2))


def test_missing_or_degenerate_geometry_is_explicit():
    image = np.zeros((300, 300), np.uint8)
    corners = np.array([[20+i*10, 20] for i in range(6)], np.float32)
    marker = np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]], np.float32)
    _, evidence, failures = refine_native_corners(image, corners, np.arange(6), [marker], 10)
    assert failures
    assert all(e["reason_codes"] == ["degenerate_local_board_geometry"] for e in evidence["corners"])
    _, evidence, failures = refine_native_corners(image, corners, np.arange(6), [], 10)
    assert failures
    assert evidence["reason_codes"] == ["missing_refinement_geometry"]
