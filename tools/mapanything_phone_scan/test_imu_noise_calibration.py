from __future__ import annotations

import hashlib
import json
import math
import zipfile

import numpy as np
import pytest

from .imu_calibration import HEADER, SCHEMA
from .imu_noise_calibration import (
    _region, analyze_noise_stream, overlapping_allan, run_noise_calibration,
    short_noise_model_has_evidence,
)


def test_overlapping_estimator_matches_direct_adjacent_clusters():
    values = np.random.default_rng(7).normal(size=128)
    factors = np.array([1, 2, 5, 13])
    taus, actual = overlapping_allan(values, 200, factors)
    expected = []
    for m in factors:
        differences = [np.mean(values[i+m:i+2*m]) - np.mean(values[i:i+m]) for i in range(len(values)-2*m+1)]
        expected.append(np.sqrt(np.mean(np.square(differences))/2))
    np.testing.assert_allclose(taus, factors / 200)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_openvins_noise_units_and_sqrt_three_convention():
    taus = np.geomspace(0.05, 300, 60)
    white, walk = 0.003, 0.00017
    fit = _region(taus, white / np.sqrt(taus), -0.5, np.ones_like(taus, dtype=bool))
    assert fit["coefficient"] == pytest.approx(white)
    fit = _region(taus, walk * np.sqrt(taus / 3), 0.5, np.ones_like(taus, dtype=bool))
    assert fit["coefficient"] == pytest.approx(walk)


def test_default_short_capture_measures_white_and_labels_drift_prior():
    rate = 200
    count = rate * 60
    sigma = 0.001
    values = np.random.default_rng(37).normal(0, sigma * math.sqrt(rate), (count, 3))
    times = 18_900_000_000_000_789 + np.arange(count, dtype=np.int64) * 5_000_000
    result = analyze_noise_stream(times, values, "gyroscope")
    assert result["qualified"] is True, result["reason_codes"]
    assert result["method"] == "short_session"
    assert result["minimum_duration_s"] == 58
    assert result["random_walk_measured"] is False
    for axis in result["axes"]:
        assert axis["fits"]["white"]["coefficient"] == pytest.approx(sigma, rel=0.1)
        assert axis["fits"]["white"]["heldout_passed"]
        assert axis["fits"]["random_walk"] is None
    prior = result["drift_prior"]
    assert prior["kind"] == "model_prior" and prior["measured"] is False
    assert prior["long_term_random_walk_measured"] is False
    assert prior["coefficient"] > 0 and math.isfinite(prior["coefficient"])
    expected = 3 * np.maximum(prior["block_variation_xyz"], prior["white_floor_xyz"]) * np.sqrt(3 / prior["block_duration_s"])
    np.testing.assert_allclose(prior["coefficient_xyz"], expected)
    assert prior["coefficient"] == max(expected)
    assert result["resampling_applied"] is False
    assert result["train_sample_range"][1] == result["heldout_sample_range"][0]


@pytest.mark.parametrize("span,accepted", [(57.995, False), (58.0, True), (59.995, True), (299.995, True)])
def test_short_duration_uses_observed_timestamp_span_not_requested_record_length(span, accepted):
    count = round(span * 200) + 1
    values = np.random.default_rng(37).normal(0, 0.001 * np.sqrt(200), (count, 3))
    times = 18_900_000_000_000_789 + np.arange(count, dtype=np.int64) * 5_000_000
    result = analyze_noise_stream(times, values, "gyroscope")
    assert result["qualified"] is accepted, result["reason_codes"]
    assert result["duration_s"] == pytest.approx(span)
    assert ("stationary_recording_shorter_than_58_seconds" in result["reason_codes"]) is not accepted


def test_full_allan_is_explicit_and_does_not_invent_unobserved_walk():
    count = 12_000
    values = np.random.default_rng(37).normal(0, 0.001 * np.sqrt(200), (count, 3))
    result = analyze_noise_stream(np.arange(count, dtype=np.int64) * 5_000_000, values,
                                  "gyroscope", method="full_allan")
    assert not result["qualified"]
    assert "stationary_recording_shorter_than_three_hours" in result["reason_codes"]
    assert any("random_walk_not_observable" in reason for reason in result["reason_codes"])
    assert "drift_prior" not in result


def test_heldout_motion_and_cadence_cannot_be_hidden_by_training_fit():
    rng = np.random.default_rng(101)
    count = 20_000
    values = rng.normal(0, 0.001 * np.sqrt(200), (count, 3))
    values[count//2:] *= 5
    values[-2000:] += 0.1
    times = np.arange(count, dtype=np.int64) * 5_000_000
    times[-5000:] += 10_000_000
    report = analyze_noise_stream(times, values, "gyroscope", minimum_duration_s=0)
    assert not report["qualified"]
    assert "sample_cadence_not_uniform_enough_for_unresampled_allan" in report["reason_codes"]
    assert "gyroscope_white" not in report
    assert any("heldout_unstable" in reason for reason in report["reason_codes"])


def test_three_hour_synthetic_white_plus_bias_walk_recovers_both_terms():
    rate, count = 50, 10800 * 50
    white, walk = 0.0001, 0.00005
    rng = np.random.default_rng(721)
    xyz = rng.normal(0, white * np.sqrt(rate), (count, 3))
    xyz += np.cumsum(rng.normal(0, walk / np.sqrt(rate), (count, 3)), axis=0)
    timestamps = 1_000_000_000 + np.arange(count, dtype=np.int64) * (1_000_000_000 // rate)
    result = analyze_noise_stream(timestamps, xyz, "gyroscope", method="full_allan")
    assert result["qualified"] is True, result["reason_codes"]
    for axis in result["axes"]:
        assert axis["fits"]["white"]["coefficient"] == pytest.approx(white, rel=0.05)
        assert axis["fits"]["random_walk"]["coefficient"] == pytest.approx(walk, rel=0.2)
        assert axis["fits"]["white"]["heldout_passed"]
        assert axis["fits"]["random_walk"]["heldout_passed"]


@pytest.mark.parametrize("bad", ["nan", "nonmonotonic"])
def test_invalid_streams_are_rejected(bad):
    times = np.arange(100, dtype=np.int64) * 5_000_000
    values = np.ones((100, 3))
    if bad == "nan":
        values[25, 2] = np.nan
    else:
        times[30] = times[29]
    with pytest.raises(ValueError):
        analyze_noise_stream(times, values, "accelerometer")


def noise_fixture(root, *, duration_s=None, acquisition_issues=None):
    root.mkdir()
    streams, manifest_streams = {}, {}
    rng = np.random.default_rng(37)
    count = 64 if duration_s is None else round(duration_s * 200)
    for kind, filename, z in (("accelerometer", "accel.csv", 9.8), ("gyroscope", "gyro.csv", 0)):
        values = np.zeros((count, 3)) if duration_s is None else rng.normal(
            0, (0.003 if kind == "accelerometer" else 0.001) * np.sqrt(200), (count, 3))
        values[:, 2] += z
        raw = (",".join(HEADER) + "\n" + "".join(
            f"{1_000_000_000+i*5_000_000},{x},{y},{v},0,0,0,3,{1_000_000_000+i*5_000_000}\n"
            for i, (x, y, v) in enumerate(values))).encode()
        (root / filename).write_bytes(raw)
        streams[kind] = {"sample_count": count, "sha256": hashlib.sha256(raw).hexdigest()}
        manifest_streams[kind] = {"sensor_id": "test:" + kind, "file": filename}
    manifest = {"schema": SCHEMA, "capture_id": root.name, "device": {"id": "test"}, "streams": manifest_streams}
    (root / "imu_capture_manifest.json").write_text(json.dumps(manifest))
    with zipfile.ZipFile(root / "capture_bundle.zip", "w") as archive:
        for name in ("imu_capture_manifest.json", "accel.csv", "gyro.csv"):
            archive.write(root / name, name)
    receipt = {"capture_id": root.name, "bundle_sha256": hashlib.sha256((root / "capture_bundle.zip").read_bytes()).hexdigest(),
               "streams": streams, "acquisition_issues": ["less_than_three_hours"] if acquisition_issues is None else acquisition_issues}
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def test_noise_processing_retains_curve_and_rejects_short_evidence(tmp_path):
    source = noise_fixture(tmp_path / "source")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    result = run_noise_calibration(source, tmp_path / "output")
    assert result["status"] == "completed"
    assert result["imu_noise_calibrated"] is False
    assert result["noise_model_usable"] is False
    assert result["accepted_for_metric_vio"] is False
    assert result["noise"] == {}
    assert (tmp_path / "output/noise_allan.png").read_bytes().startswith(b"\x89PNG")
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_noise_calibration(source, tmp_path / "output")


def test_real_short_job_retains_hash_bound_four_term_candidate_and_honest_provenance(tmp_path):
    source = noise_fixture(tmp_path / "source", duration_s=60)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    result = run_noise_calibration(source, tmp_path / "output")
    assert result["noise_model_usable"] is True, result["reason_codes"]
    assert result["noise_model_status"] == "short_session_candidate"
    assert result["imu_noise_calibrated"] is False
    assert result["accepted_for_metric_vio"] is False
    assert result["quality"]["status"] == "qualified"
    assert result["acquisition_issues"] == ["less_than_three_hours"]
    assert "less_than_three_hours" not in result["reason_codes"]
    assert len(result["noise"]) == 4
    for key, value in result["noise"].items():
        assert math.isfinite(value) and value > 0
        provenance = result["noise_provenance"][key]
        measured = key.endswith("noise_density")
        assert provenance["measured"] is measured
        assert provenance["kind"] == ("measured" if measured else "model_prior")
        assert provenance["coefficient"] == value
        assert provenance["source"] and provenance["derivation"]
    assert short_noise_model_has_evidence(result)
    for name, info in result["artifacts"].items():
        path = tmp_path / "output" / name
        assert path.stat().st_size == info["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}


@pytest.mark.parametrize("issue", ["dropped_sensor_records", "gyroscope_timestamp_failure", "recording_interrupted"])
def test_short_job_never_ignores_acquisition_failures(tmp_path, issue):
    source = noise_fixture(tmp_path / "source", duration_s=60, acquisition_issues=["less_than_three_hours", issue])
    result = run_noise_calibration(source, tmp_path / "output")
    assert result["noise_model_usable"] is False
    assert result["noise_model_status"] == "insufficient_evidence"
    assert issue in result["reason_codes"]
    assert result["noise"] == {}


@pytest.mark.parametrize("bad", ["constant", "motion", "jitter", "unsupported_rate"])
def test_short_checks_reject_unmeasurable_or_nonstationary_data(bad):
    count = 12_000
    times = np.arange(count, dtype=np.int64) * 5_000_000
    values = np.random.default_rng(37).normal(0, 0.001 * np.sqrt(200), (count, 3))
    if bad == "constant":
        values[:] = 0
    elif bad == "motion":
        values[4000:4200] += 0.05  # One second hidden by ten-second means.
    elif bad == "jitter":
        times[1::2] += 100_000
    else:
        times *= 20
    result = analyze_noise_stream(times, values, "gyroscope")
    assert not result["qualified"]
    assert "drift_prior" not in result


@pytest.mark.parametrize("bad", ["float_timestamps", "negative_timestamps", "huge_values", "unknown_method"])
def test_malformed_input_is_not_a_noise_fit(bad):
    times = np.arange(64, dtype=np.int64) * 5_000_000
    values = np.zeros((64, 3))
    method = "short_session"
    if bad == "float_timestamps":
        times = times.astype(float)
    elif bad == "negative_timestamps":
        times -= 500_000_000
    elif bad == "huge_values":
        values[:] = 1e200
    else:
        method = "something_else"
    with pytest.raises(ValueError):
        analyze_noise_stream(times, values, "gyroscope", method=method)


def test_tampered_stream_does_not_produce_noise_calibration(tmp_path):
    source = noise_fixture(tmp_path / "source")
    with (source / "accel.csv").open("a") as handle:
        handle.write("unexpected data\n")
    with pytest.raises(ValueError, match="checksum or count"):
        run_noise_calibration(source, tmp_path / "output")


@pytest.mark.parametrize("target", ["same", "nested"])
def test_noise_output_cannot_mutate_retained_capture(tmp_path, target):
    source = noise_fixture(tmp_path / "source")
    output = source if target == "same" else source / "new-output"
    with pytest.raises(ValueError, match="separate"):
        run_noise_calibration(source, output)
    assert not (source / "new-output").exists()
