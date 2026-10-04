"""Focused tests for offline exact-PCF region extraction, not admission."""
import numpy as np
import pytest

from noesis.calibration.structural_registration import StructuralRegistrationError
from noesis.calibration.structural_roi_planes import (
    fit_supported_region_plane, grow_connected_plane_support, line_at_observed_height,
    select_admitted_region_support,
)


def wall(*, height=1.6, tilt=.1):
    y, z = np.meshgrid(np.linspace(.3, .3+height, 36), np.linspace(1, 3, 36))
    return np.column_stack((2-tilt*y.ravel(), y.ravel(), z.ravel()))


def test_region_selects_only_existing_admitted_points():
    points = wall()
    cloud = np.concatenate((points, points+np.array([3, 0, 0])))
    indices = select_admitted_region_support(cloud, points, maximum_distance_m=.08)
    np.testing.assert_array_equal(indices, np.arange(len(points)))
    np.testing.assert_array_equal(cloud[indices], points)


def test_fit_retains_observed_tilt_and_shared_height_slice():
    measured = wall()
    fit = fit_supported_region_plane(measured)
    assert fit["extraction_support_passed"]
    np.testing.assert_allclose(fit["normal_xyz"], np.array([1, .1, 0])/np.sqrt(1.01), atol=1e-7)
    assert fit["p90_abs_distance_m"] < 1e-7
    segment = line_at_observed_height(fit, segment_id="observed", reference_y_m=1.3)
    for key in ("start_xz", "end_xz"):
        xyz = [segment[key][0], 1.3, segment[key][1]]
        assert abs(np.dot(xyz, fit["normal_xyz"])+fit["offset_m"]) < 1e-7
    assert abs(segment["start_xz"][0]-1.87) < 1e-7


def test_thin_patch_does_not_relax_height_or_tilt_gates():
    fit = fit_supported_region_plane(wall(height=.4, tilt=.5))
    assert not fit["extraction_support_passed"]
    assert {"height_span", "plane_tilt"}.issubset(fit["reason_codes"])


@pytest.mark.parametrize("distance", [0, .181, float("nan"), True])
def test_association_tolerance_remains_bounded(distance):
    with pytest.raises(StructuralRegistrationError):
        select_admitted_region_support(wall(), wall(), maximum_distance_m=distance)


def test_degenerate_or_nonfinite_support_is_rejected():
    with pytest.raises(StructuralRegistrationError):
        fit_supported_region_plane(np.column_stack((np.arange(10), np.zeros((10, 2)))))
    cloud = wall(); cloud[0, 0] = float("nan")
    with pytest.raises(StructuralRegistrationError):
        fit_supported_region_plane(cloud)


def test_connected_support_recovers_wall_but_not_disconnected_coplanar_surface():
    points = wall(tilt=0)
    cloud = np.concatenate((points, points+np.array([0, 0, 8])))
    seeds = np.arange(400, 700)
    fit = fit_supported_region_plane(points[seeds])
    selected = grow_connected_plane_support(cloud, seeds, fit)
    np.testing.assert_array_equal(selected, np.arange(len(points)))
    assert fit_supported_region_plane(cloud[selected])["extraction_support_passed"]
