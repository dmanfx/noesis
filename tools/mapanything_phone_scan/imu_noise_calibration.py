"""Short stationary noise candidates; full Allan characterization is opt-in.

The OpenVINS continuous-time convention is sigma_white/sqrt(tau) and
sigma_random_walk*sqrt(tau/3). This is a noise fit, not a bias, scale, camera
extrinsic, timing, or live-runtime calibration. Raw sensor bias estimates are
not subtracted, matching the RoomWalk camera recorder's sensor consumer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .imu_calibration import HEADER, MAX_STREAM_ROWS, SCHEMA

POLICY = "roomwalk.stationary_noise_validation.v1"
SHORT_POLICY = "roomwalk.short_stationary_noise_validation.v1"
SHORT_MINIMUM_DURATION_S = 58.0
DRIFT_PRIOR_POLICY = "roomwalk.block_variation_drift_prior.v1"
UNITS = {"gyroscope_noise_density": "rad/s/sqrt(Hz)",
         "gyroscope_random_walk": "rad/s^2/sqrt(Hz)",
         "accelerometer_noise_density": "m/s^2/sqrt(Hz)",
         "accelerometer_random_walk": "m/s^3/sqrt(Hz)"}
SOURCES = ["https://docs.openvins.com/gs-calibration.html",
           "https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model",
           "https://github.com/ori-drs/allan_variance_ros/blob/master/scripts/analysis.py",
           "https://allantools.readthedocs.io/en/stable/readme_copy.html"]


def overlapping_allan(values: np.ndarray, rate_hz: float, factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact overlapping adjacent-cluster estimator, without resampling/filtering."""
    if values.ndim != 1 or not np.isfinite(values).all() or not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("Allan deviation needs a finite scalar stream and positive sample rate")
    integral = np.concatenate(([0.0], np.cumsum(values - np.mean(values), dtype=np.float64)))
    taus, deviations = [], []
    for factor in factors:
        m = int(factor)
        if m < 1 or 2 * m >= len(values):
            continue
        difference = (integral[2*m:] - 2 * integral[m:-m] + integral[:-2*m]) / m
        taus.append(m / rate_hz)
        deviations.append(float(np.sqrt(np.mean(difference * difference) / 2)))
    return np.asarray(taus), np.asarray(deviations)


def _region(taus: np.ndarray, deviation: np.ndarray, slope: float, allowed: np.ndarray) -> dict[str, Any] | None:
    # Select a contiguous training-only region, then freeze it for held-out data.
    indices = np.flatnonzero(allowed & (deviation > 0))
    best: tuple[float, dict[str, Any]] | None = None
    for start in range(max(0, len(indices) - 3)):
        stop = start + 4
        while stop < len(indices) and taus[indices[stop-1]] / taus[indices[start]] < 2:
            stop += 1
        chosen = indices[start:stop]
        if len(chosen) < 4 or taus[chosen[-1]] / taus[chosen[0]] < 2:
            continue
        x, y = np.log(taus[chosen]), np.log(deviation[chosen])
        measured, _ = np.polyfit(x, y, 1)
        intercept = float(np.mean(y - slope * x))
        error = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
        if abs(measured - slope) > 0.15 or error > 0.15:
            continue
        factor = math.exp(intercept) * (math.sqrt(3) if slope > 0 else 1)
        candidate = {"coefficient": factor, "slope": float(measured), "expected_slope": slope,
                     "tau_range_s": [float(taus[chosen[0]]), float(taus[chosen[-1]])],
                     "indices": chosen.tolist(), "training_log_rmse": error}
        score = abs(measured - slope) + error
        if best is None or score < best[0]:
            best = score, candidate
    return None if best is None else best[1]


def analyze_noise_stream(timestamps_ns: np.ndarray, xyz: np.ndarray, kind: str,
                         *, method: str = "short_session",
                         minimum_duration_s: float | None = None) -> dict[str, Any]:
    if method not in {"short_session", "full_allan"}:
        raise ValueError("Noise method must be short_session or full_allan")
    if minimum_duration_s is None:
        minimum_duration_s = SHORT_MINIMUM_DURATION_S if method == "short_session" else 10799
    if not math.isfinite(minimum_duration_s) or minimum_duration_s < 0:
        raise ValueError("Noise minimum duration must be finite and nonnegative")
    if kind not in {"accelerometer", "gyroscope"} or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(timestamps_ns) or len(xyz) < 32:
        raise ValueError("Noise processing needs at least 32 timestamped three-axis samples")
    if len(xyz) > MAX_STREAM_ROWS or not np.isfinite(xyz).all() or np.any(np.abs(xyz) > 1e6):
        raise ValueError("Noise stream is excessive or nonfinite")
    if timestamps_ns.ndim != 1 or timestamps_ns.dtype.kind not in "iu" or np.any(timestamps_ns < 0) or np.any(timestamps_ns > np.iinfo(np.int64).max):
        raise ValueError("Noise timestamps must be nonnegative integer nanoseconds")
    timestamps_ns = timestamps_ns.astype(np.int64, copy=False)
    intervals = np.diff(timestamps_ns)
    if np.any(intervals <= 0):
        raise ValueError("Noise timestamps must be strictly increasing")
    duration = float((int(timestamps_ns[-1]) - int(timestamps_ns[0])) * 1e-9)
    dt = float(np.median(intervals)) * 1e-9
    rate = 1 / dt
    reasons = []
    if duration < minimum_duration_s:
        reasons.append("stationary_recording_shorter_than_58_seconds" if method == "short_session"
                       else "stationary_recording_shorter_than_three_hours")
    jitter = float(np.percentile(np.abs(intervals * 1e-9 - dt), 95) / dt)
    maximum_gap = float(np.max(intervals)) * 1e-9
    if jitter > 0.01 or maximum_gap > 1.5 * dt:
        reasons.append("sample_cadence_not_uniform_enough_for_unresampled_allan")
    if not 20 <= rate <= 1000:
        reasons.append("unsupported_sensor_rate")
    block = max(1, round(rate * 10))
    full_blocks = len(xyz) // block
    centers = xyz[:full_blocks*block].reshape(full_blocks, block, 3).mean(axis=1) if full_blocks else xyz.mean(axis=0, keepdims=True)
    center = np.median(centers, axis=0)
    drift = float(np.max(np.linalg.norm(centers - center, axis=1)))
    rms = float(np.sqrt(np.mean(np.sum((xyz - center) ** 2, axis=1))))
    if kind == "accelerometer":
        if drift > 0.15 or rms > 0.5 or not 8.5 < float(np.linalg.norm(center)) < 11.2:
            reasons.append("stationarity_not_supported")
    elif drift > 0.02 or rms > 0.05 or float(np.linalg.norm(center)) > 0.1:
        reasons.append("stationarity_not_supported")
    # A brief move can average away inside ten-second blocks. Check short
    # blocks too, including the tail; this does not certify physical stillness.
    if method == "short_session":
        short_block = max(1, round(rate))
        short_centers = np.asarray([xyz[i:i+short_block].mean(axis=0)
                                    for i in range(0, len(xyz)-short_block+1, short_block)]
                                   + [xyz[-short_block:].mean(axis=0)])
        short_drift = float(np.max(np.linalg.norm(short_centers - center, axis=1)))
        if short_drift > (0.15 if kind == "accelerometer" else 0.02):
            reasons.append("stationarity_not_supported")
    split = len(xyz) // 2
    # Each largest cluster has at least 20 non-overlapping pairs per half.
    maximum_factor = max(1, min(split, len(xyz)-split) // 40)
    factors = np.unique(np.maximum(1, np.round(np.geomspace(1, maximum_factor, 60)).astype(int)))
    axes = []
    for axis in range(3):
        taus, train = overlapping_allan(xyz[:split, axis], rate, factors)
        _, test = overlapping_allan(xyz[split:, axis], rate, factors)
        fits = {}
        for name, slope, allowed in (("white", -0.5, (taus >= max(4*dt, 0.04)) & (taus <= 0.5)),
                                     ("random_walk", 0.5, taus >= 10)):
            if name == "random_walk" and method == "short_session":
                fits[name] = None
                continue
            fit = _region(taus, train, slope, allowed)
            if fit is None:
                reasons.append(f"{kind}_{'xyz'[axis]}_{name}_not_observable")
                fits[name] = None
                continue
            selected = np.array(fit.pop("indices"), dtype=int)
            factor = np.sqrt(3 / taus[selected]) if slope > 0 else np.sqrt(taus[selected])
            candidate = float(np.exp(np.mean(np.log(np.maximum(test[selected], np.finfo(float).tiny) * factor))))
            ratio = candidate / fit["coefficient"]
            heldout_slope = float(np.polyfit(np.log(taus[selected]), np.log(np.maximum(test[selected], np.finfo(float).tiny)), 1)[0])
            prediction = fit["coefficient"] * (np.sqrt(taus[selected] / 3) if slope > 0 else 1 / np.sqrt(taus[selected]))
            log_error = float(np.sqrt(np.mean(np.log(np.maximum(test[selected], np.finfo(float).tiny) / prediction) ** 2)))
            stable = 0.5 <= ratio <= 2 and abs(heldout_slope - slope) <= 0.2 and log_error <= math.log(2)
            fit.update(heldout_coefficient=candidate, heldout_ratio=ratio, heldout_slope=heldout_slope,
                       heldout_log_rmse=log_error, heldout_passed=bool(stable))
            if not stable:
                reasons.append(f"{kind}_{'xyz'[axis]}_{name}_heldout_unstable")
            fits[name] = fit
        axes.append({"axis": "xyz"[axis], "tau_s": taus.tolist(), "training_allan": train.tolist(),
                     "heldout_allan": test.tolist(), "fits": fits})
    result = {"sample_count": len(xyz), "duration_s": duration, "rate_hz": rate,
            "method": method, "minimum_duration_s": minimum_duration_s,
            "timestamp_jitter_p95_fraction": jitter, "maximum_gap_s": maximum_gap,
            "block_mean_max_deviation": drift, "centered_rms": rms, "axes": axes,
            "reason_codes": sorted(set(reasons)), "qualified": not reasons,
            "resampling_applied": False, "sensor_bias_correction_applied": False,
            "train_sample_range": [0, split], "heldout_sample_range": [split, len(xyz)]}
    if method == "short_session":
        result["one_second_block_mean_max_deviation"] = short_drift
        result["random_walk_measured"] = False
        if not reasons:
            result["drift_prior"] = _drift_prior(xyz, rate, axes)
    return result


def _drift_prior(xyz, rate, axes):
    """Engineering envelope, NOT an observed +1/2 Allan slope.

    For T-second means, white ADEV is N/sqrt(T). A Brownian-bias model
    predicts ADEV q*sqrt(T/3). Assign q = 3*max(A_block, N/sqrt(T))*sqrt(3/T),
    where A_block is the largest adjacent mean difference / sqrt(2).
    The factor three is a local conservative policy, not a confidence bound
    or a manufacturer specification. Both halves set this prior envelope;
    only the white fit has a frozen training region and independent holdout.
    Five-second evidence cannot establish thermal or multi-minute drift.
    """
    block = max(1, round(rate * 5))
    means = xyz[:len(xyz)//block*block].reshape(-1, block, 3).mean(axis=1)
    tau = block / rate
    variations = np.max(np.abs(np.diff(means, axis=0)), axis=0) / math.sqrt(2)
    densities = np.asarray([max(row["fits"]["white"]["coefficient"],
                                row["fits"]["white"]["heldout_coefficient"]) for row in axes])
    floors = densities / math.sqrt(tau)
    candidates = 3 * np.maximum(variations, floors) * math.sqrt(3 / tau)
    return {"policy": DRIFT_PRIOR_POLICY, "kind": "model_prior", "measured": False,
            "source": DRIFT_PRIOR_POLICY,
            "derivation": "3 * max(max_abs_adjacent_block_mean_delta / sqrt(2), white_density / sqrt(T)) * sqrt(3 / T)",
            "block_duration_s": tau, "block_count": len(means), "inflation_factor": 3.0,
            "block_variation_xyz": variations.tolist(), "white_floor_xyz": floors.tolist(),
            "coefficient_xyz": candidates.tolist(), "coefficient": float(np.max(candidates)),
            "uses_training_and_heldout_blocks": True, "long_term_random_walk_measured": False}


def _noise_terms(results, method):
    noise, provenance = {}, {}
    for kind, result in results.items():
        for term, fit_name in (("noise_density", "white"), ("random_walk", "random_walk")):
            key = kind + "_" + term
            if term == "random_walk" and method == "short_session":
                item = dict(result["drift_prior"])
            else:
                coefficient = max(max(row["fits"][fit_name]["coefficient"],
                                      row["fits"][fit_name]["heldout_coefficient"]) for row in result["axes"])
                item = {"kind": "measured", "measured": True, "coefficient": coefficient,
                        "source": "overlapping_allan_train_holdout",
                        "derivation": "maximum of all axis training and heldout coefficients in the frozen training-selected slope region",
                        "heldout_passed": True}
            noise[key] = item["coefficient"]
            provenance[key] = {**item, "units": UNITS[key]}
    return noise, provenance


def short_noise_model_has_evidence(reference):
    """Validate the short-result contract, not source integrity or VIO admission.

    Consumers must ALSO check artifact hashes and exact device/sensor binding.
    Four positive numbers plus a usable flag are not sufficient evidence.
    """
    try:
        if (reference["schema"] != "roomwalk.imu_noise_calibration.v1"
                or reference["status"] != "completed"
                or reference["method"] != "short_session"
                or reference["noise_model_usable"] is not True
                or reference["noise_model_status"] != "short_session_candidate"
                or reference["imu_noise_calibrated"] is not False
                or reference["quality"]["status"] != "qualified"
                or reference["quality"]["policy"] != SHORT_POLICY
                or reference["reason_codes"]):
            return False
        for kind in ("accelerometer", "gyroscope"):
            stream = reference["streams"][kind]
            if (stream["qualified"] is not True or stream["reason_codes"]
                    or stream["method"] != "short_session"
                    or not SHORT_MINIMUM_DURATION_S <= stream["duration_s"]
                    or stream["random_walk_measured"] is not False
                    or [row["axis"] for row in stream["axes"]] != list("xyz")):
                return False
            for row in stream["axes"]:
                fit = row["fits"]["white"]
                if (fit["heldout_passed"] is not True or row["fits"]["random_walk"] is not None
                        or any(type(fit[k]) not in (int, float) or not math.isfinite(fit[k]) or fit[k] <= 0
                               for k in ("coefficient", "heldout_coefficient"))):
                    return False
            prior = stream["drift_prior"]
            tau = prior["block_duration_s"]
            if (prior["policy"] != DRIFT_PRIOR_POLICY or prior["source"] != DRIFT_PRIOR_POLICY
                    or prior["kind"] != "model_prior" or prior["measured"] is not False
                    or prior["long_term_random_walk_measured"] is not False
                    or prior["uses_training_and_heldout_blocks"] is not True
                    or prior["inflation_factor"] != 3 or not 4.9 <= tau <= 5.1
                    or prior["block_count"] < 11):
                return False
            for axis, row in enumerate(stream["axes"]):
                fit = row["fits"]["white"]
                density = max(fit["coefficient"], fit["heldout_coefficient"])
                floor = density / math.sqrt(tau)
                variation = prior["block_variation_xyz"][axis]
                if not math.isfinite(variation) or variation < 0:
                    return False
                expected = 3 * max(variation, floor) * math.sqrt(3 / tau)
                if (not math.isclose(prior["white_floor_xyz"][axis], floor, rel_tol=1e-12)
                        or not math.isclose(prior["coefficient_xyz"][axis], expected, rel_tol=1e-12)):
                    return False
            if prior["coefficient"] != max(prior["coefficient_xyz"]):
                return False
        noise, provenance = _noise_terms(reference["streams"], "short_session")
        return (all(type(noise[k]) in (int, float) and math.isfinite(noise[k]) and noise[k] > 0 for k in UNITS)
                and reference["noise"] == noise and reference["noise_provenance"] == provenance)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError):
        return False


def _stream(path: Path, expected: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    count = int(expected["sample_count"])
    if not 32 <= count <= MAX_STREAM_ROWS or path.is_symlink():
        raise ValueError("Stationary stream is too short, excessive, or not a regular file")
    timestamps, xyz = np.empty(count, dtype=np.int64), np.empty((count, 3), dtype=np.float64)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header = handle.readline(2049)
        digest.update(header)
        if next(csv.reader([header.decode()])) != HEADER:
            raise ValueError("Stationary sensor header changed after import")
        for index in range(count):
            line = handle.readline(2049)
            if not line or len(line) > 2048:
                raise ValueError("Stationary sensor stream changed after import")
            digest.update(line)
            row = next(csv.reader([line.decode()]))
            if len(row) != len(HEADER):
                raise ValueError("Stationary sensor row changed after import")
            timestamps[index] = int(row[0])
            xyz[index] = [float(value) for value in row[1:4]]
        if handle.read(1) or digest.hexdigest() != expected["sha256"]:
            raise ValueError("Stationary sensor checksum or count changed after import")
    return timestamps, xyz


def run_noise_calibration(capture_dir: Path, output_dir: Path, *,
                          method: str = "short_session",
                          progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    if method not in {"short_session", "full_allan"}:
        raise ValueError("Noise method must be short_session or full_allan")
    capture_dir, output_dir = Path(capture_dir).resolve(), Path(output_dir).resolve()
    if output_dir == capture_dir or output_dir.is_relative_to(capture_dir) or capture_dir.is_relative_to(output_dir):
        raise ValueError("Noise output must be separate from its retained capture")
    progress = progress or (lambda *_: None)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("receipt.json", "imu_capture_manifest.json"):
        path = capture_dir / name
        if path.is_symlink() or path.stat().st_size > 64 * 1024:
            raise ValueError("Stationary calibration metadata exceeds its bound")
    manifest = json.loads((capture_dir / "imu_capture_manifest.json").read_text())
    receipt = json.loads((capture_dir / "receipt.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("capture_id") != receipt.get("capture_id"):
        raise ValueError("Stationary recording identity is inconsistent")
    archive = capture_dir / "capture_bundle.zip"
    if archive.is_symlink() or not 0 < archive.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError("Stationary source archive is missing or excessive")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != receipt["bundle_sha256"]:
        raise ValueError("Stationary source archive changed after import")
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open("imu_capture_manifest.json") as member:
            if member.read(65537) != (capture_dir / "imu_capture_manifest.json").read_bytes():
                raise ValueError("Stationary device metadata changed after import")
    acquisition_issues = list(receipt.get("acquisition_issues") or [])
    # This legacy import diagnostic is irrelevant to short characterization;
    # preserve it as acquisition evidence, never discard timing/drop issues.
    results, reasons = {}, [issue for issue in acquisition_issues
                           if not (method == "short_session" and issue == "less_than_three_hours")]
    for index, kind in enumerate(("accelerometer", "gyroscope")):
        progress(index * 0.45, f"Checking {kind} cadence, stationarity and held-out noise")
        filename = "accel.csv" if kind == "accelerometer" else "gyro.csv"
        timestamps, values = _stream(capture_dir / filename, receipt["streams"][kind])
        result = analyze_noise_stream(timestamps, values, kind, method=method)
        results[kind] = result
        reasons.extend(result["reason_codes"])
        del timestamps, values
    qualified = not reasons
    noise, provenance = _noise_terms(results, method) if qualified else ({}, {})
    report = {"schema": "roomwalk.imu_noise_calibration.v1", "status": "completed", "mode": "noise",
              "method": method,
              "quality": {"status": "qualified" if qualified else "insufficient_evidence",
                          "scope": "noise_model", "policy": SHORT_POLICY if method == "short_session" else POLICY},
              "noise_model_usable": qualified,
              "noise_model_status": ("short_session_candidate" if method == "short_session" else "full_allan_measured") if qualified else "insufficient_evidence",
              "noise_provenance": provenance,
              "imu_noise_calibrated": qualified and method == "full_allan", "accepted_for_metric_vio": False,
              "camera_intrinsics_calibrated": False, "camera_imu_extrinsics_calibrated": False, "time_offset_calibrated": False,
              "reason_codes": sorted(set(reasons)), "acquisition_issues": acquisition_issues,
              "noise": noise, "streams": results,
              "capture_id": manifest["capture_id"], "device": manifest["device"], "sensors": manifest["streams"],
              "units": UNITS,
              "source_bundle_sha256": receipt["bundle_sha256"], "sources": SOURCES,
              "limitations": (["Short-session random-walk coefficients are engineering model priors, NOT measured long-term Allan terms.",
                               "The five-second envelope uses both halves; only white density has independent train/holdout validation.",
                               "No thermal repeatability, 1-5 minute trajectory accuracy, or metric-VIO admission is established."]
                              if method == "short_session" else
                              ["Full Allan characterization is optional; three hours does not guarantee observable bias random walk."]) +
                             ["Online estimator noise inflation is not a measured sensor coefficient.",
                              "Stationarity checks do not independently certify a fixed physical pose or thermal repeatability."]}
    (output_dir / "noise_result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output_dir / "source_manifest.json").write_text(json.dumps({"capture_id": manifest["capture_id"], "bundle_sha256": receipt["bundle_sha256"], "streams": receipt["streams"]}, indent=2))
    progress(0.95, "Writing stationary noise report")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for plot, (kind, result) in zip(axes, results.items()):
        for axis, color in zip(result["axes"], ("#19786d", "#b86d28", "#4568ae")):
            for key, style, label in (("training_allan", "-", " train"), ("heldout_allan", "--", " held out")):
                positive = np.asarray(axis[key]) > 0
                if np.any(positive):
                    plot.loglog(np.asarray(axis["tau_s"])[positive], np.asarray(axis[key])[positive], style, color=color, label=axis["axis"] + label)
        plot.set(title=kind.capitalize(), xlabel="Averaging time (s)", ylabel="Allan deviation (m/s²)" if kind == "accelerometer" else "Allan deviation (rad/s)")
        if plot.get_legend_handles_labels()[0]:
            plot.legend(fontsize=7)
        else:
            plot.text(0.5, 0.5, "No measurable variation\nNoise fit unavailable", ha="center", transform=plot.transAxes)
        plot.grid(True, alpha=0.2)
    figure.savefig(output_dir / "noise_allan.png", dpi=160)
    plt.close(figure)
    # The enclosing report binds the separate, non-self-referential result used
    # by the camera–IMU solver, plus its raw-data manifest and plot.
    report["artifacts"] = {}
    for name in ("noise_result.json", "source_manifest.json", "noise_allan.png"):
        path = output_dir / name
        report["artifacts"][name] = {"path": name, "bytes": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return report
