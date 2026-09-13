"""Evaluate a camera-origin OpenVINS result against EuRoC ground truth.

Ground truth is loaded only by this evaluator.  The estimator receives the
camera/IMU streams and shipped calibration, never the ground-truth trajectory.
ATE uses a best-fit SE(3) alignment; the best-fit Sim(3) scale is reported as a
diagnostic and is never used to turn a metric result into a passing result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from bisect import bisect_left
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _gt_rows(path: Path) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    timestamps: list[int] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith("#"):
                continue
            # EuRoC timestamps are integer nanoseconds.  Keep them as text
            # until conversion so a float round trip cannot alter matching.
            timestamps.append(int(row[0].strip()))
            values = [float(value) for value in row[1:]]
            positions.append(np.asarray(values[0:3], dtype=np.float64))
            q_wxyz = values[3:7]
            rotations.append(
                Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            )
    if len(timestamps) < 2:
        raise ValueError("ground-truth trajectory has fewer than two rows")
    return timestamps, positions, rotations


def _camera_times(path: Path, limit: int) -> list[int]:
    timestamps: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and not row[0].lstrip().startswith("#"):
                timestamps.append(int(row[0]))
    timestamps = timestamps[:limit]
    if len(timestamps) < 2:
        raise ValueError("camera trajectory has fewer than two rows")
    return timestamps


def _camera_translation(sensor_yaml: Path) -> np.ndarray:
    text = sensor_yaml.read_text(encoding="utf-8")
    match = re.search(r"T_BS\s*:.*?data\s*:\s*\[([^]]+)\]", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("camera sensor YAML has no T_BS data")
    values = [float(value) for value in re.split(r"[\s,]+", match.group(1).strip()) if value]
    if len(values) != 16:
        raise ValueError("camera T_BS is not 4x4")
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("camera T_BS is not homogeneous")
    return matrix[:3, 3]


def _nearest_index(timestamps: list[int], value: int) -> int:
    index = bisect_left(timestamps, value)
    if index == 0:
        return 0
    if index == len(timestamps):
        return index - 1
    before, after = timestamps[index - 1], timestamps[index]
    return index if after - value < value - before else index - 1


def evaluate(
    result_path: Path,
    ground_truth_path: Path,
    sensor_yaml: Path,
    subset_frames: int,
    camera_data_path: Path | None = None,
) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    poses = result.get("poses")
    if not isinstance(poses, list) or len(poses) < 2:
        raise ValueError("result contains fewer than two poses")
    gt_times, gt_positions, gt_rotations = _gt_rows(ground_truth_path)
    camera_translation = _camera_translation(sensor_yaml)
    estimated: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    timestamp_errors: list[int] = []
    for row in poses:
        timestamp = int(row["capture_time_ns"])
        index = _nearest_index(gt_times, timestamp)
        estimated.append(np.asarray(row["T_vio_world_camera"], dtype=np.float64)[:3, 3])
        expected.append(gt_positions[index] + gt_rotations[index] @ camera_translation)
        timestamp_errors.append(abs(gt_times[index] - timestamp))
    estimated_array = np.asarray(estimated)
    expected_array = np.asarray(expected)
    estimated_center = estimated_array.mean(axis=0)
    expected_center = expected_array.mean(axis=0)
    cross_covariance = ((estimated_array - estimated_center).T @ (expected_array - expected_center)) / len(poses)
    left, _, right = np.linalg.svd(cross_covariance)
    alignment_rotation = right.T @ left.T
    if np.linalg.det(alignment_rotation) < 0.0:
        right[-1] *= -1.0
        alignment_rotation = right.T @ left.T
    alignment_translation = expected_center - alignment_rotation @ estimated_center
    aligned = (alignment_rotation @ estimated_array.T).T + alignment_translation
    se3_error = np.linalg.norm(aligned - expected_array, axis=1)
    singular_values = np.linalg.svd(cross_covariance, compute_uv=False)
    estimated_variance = np.mean(np.sum((estimated_array - estimated_center) ** 2, axis=1))
    sim3_scale = float(singular_values.sum() / estimated_variance) if estimated_variance > 0.0 else float("nan")
    sim3_aligned = (sim3_scale * alignment_rotation @ estimated_array.T).T + (
        expected_center - sim3_scale * alignment_rotation @ estimated_center
    )
    sim3_error = np.linalg.norm(sim3_aligned - expected_array, axis=1)
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    report: dict[str, object] = {
        "schema": "noesis.phone_capture.vio_benchmark.v1",
        "dataset": "EuRoC MH_01_easy",
        "estimator": result.get("estimator"),
        "subset_frames": int(subset_frames),
        "num_cameras": 1,
        "use_stereo": False,
        "estimator_initialized_without_gt": True,
        "gt_used_for_evaluation_only": True,
        "gt_camera_origin": "state_groundtruth_estimate0 body origin plus calibrated cam0 p_CinI",
        "matched_poses": len(poses),
        "tracking_ratio": quality.get("tracking_ratio"),
        "initialization_delay_s": quality.get("initialization_delay_s"),
        "retained_interval_s": quality.get("retained_interval_s"),
        "estimated_time_start_ns": int(poses[0]["capture_time_ns"]),
        "estimated_time_end_ns": int(poses[-1]["capture_time_ns"]),
        "camera_frame_coverage_fraction": len(poses) / int(subset_frames),
        "timestamp_match_max_abs_ns": max(timestamp_errors),
        "ate_se3_rmse_m": float(np.sqrt(np.mean(se3_error**2))),
        "ate_se3_median_m": float(np.median(se3_error)),
        "ate_se3_max_m": float(np.max(se3_error)),
        "sim3_scale_diagnostic": sim3_scale,
        "ate_sim3_rmse_m": float(np.sqrt(np.mean(sim3_error**2))),
        "ate_sim3_median_m": float(np.median(sim3_error)),
        "align_method": "best_fit_SE3_for_ATE; best_fit_Sim3_scale_reported_diagnostic_only",
        "camera_origin_evaluation": True,
    }
    segments = result.get("segments")
    if isinstance(segments, list):
        report["reset_segments"] = sum(
            1 for segment in segments if isinstance(segment, dict) and segment.get("reset") is True
        )
    source_indices = [row.get("source_frame_index") for row in poses if isinstance(row, dict)]
    if source_indices and all(isinstance(index, int) for index in source_indices):
        indices = [int(index) for index in source_indices]
        report["first_retained_source_frame_index"] = min(indices)
        report["last_retained_source_frame_index"] = max(indices)
        report["uninitialized_or_unemitted_frames"] = max(0, int(subset_frames) - len(set(indices)))
        report["retained_source_frame_gap_count"] = sum(
            1 for before, after in zip(sorted(set(indices)), sorted(set(indices))[1:]) if after != before + 1
        )
    if camera_data_path is not None:
        camera_times = _camera_times(camera_data_path, int(subset_frames))
        report["source_camera_start_ns"] = camera_times[0]
        report["source_camera_end_ns"] = camera_times[-1]
        report["source_camera_span_s"] = (camera_times[-1] - camera_times[0]) * 1e-9
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--sensor-yaml", type=Path, required=True)
    parser.add_argument("--camera-data", type=Path)
    parser.add_argument("--subset-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.result, args.ground_truth, args.sensor_yaml, args.subset_frames, args.camera_data)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
