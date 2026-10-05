"""Direct checks of the metric mapping fit, without runtime/GPU fixtures."""
import numpy as np
import pytest

from noesis.calibration.depth_registration_builder import (
    DepthRegistrationBuildError, _enforce_monotonic, fit_monotonic_piecewise_mapping,
)


def test_noisy_high_bin_does_not_raise_all_farther_depths():
    observed = np.array([2., 5., 3., 4., 6.])
    fitted = _enforce_monotonic(observed)
    assert fitted == pytest.approx([2., 4., 4., 4., 6.])
    old = np.maximum.accumulate(observed)
    assert np.sum((fitted - observed) ** 2) < np.sum((old - observed) ** 2)


def test_pooling_uses_measured_bin_support_and_preserves_true_plateau():
    assert _enforce_monotonic(np.array([5., 3., 6.]), np.array([1., 3., 1.])) == pytest.approx([3.5, 3.5, 6.])
    assert _enforce_monotonic(np.array([3., 3., 3.])) == pytest.approx([3., 3., 3.])
    with pytest.raises(DepthRegistrationBuildError):
        _enforce_monotonic(np.array([2., 3.]), np.array([1., 0.]))


def test_fitted_camera_z_mapping_keeps_domain_and_signal():
    raw = np.linspace(1., 9., 1024)
    target = .7 * raw + .2
    fit = fit_monotonic_piecewise_mapping(raw, target)
    assert np.all(np.diff(fit.knots_raw_m) > 0)
    assert np.all(np.diff(fit.knots_registered_m) > 0)
    assert raw.min() < fit.knots_raw_m[0] < fit.knots_raw_m[-1] < raw.max()
    assert fit.fit_metrics["monotonic_fit_method"] == "weighted_pool_adjacent_violators"
    assert fit.fit_metrics["median_abs_error_m"] < .05
