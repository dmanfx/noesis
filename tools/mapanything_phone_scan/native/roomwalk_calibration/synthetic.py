"""Deterministic CPU fixture; ground truth never enters solver inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def fixture(
    offset_ns: int = 17_000_000,
    duration: float = 10.0,
    skew_ns: int = 0,
    exposure_ns: int = 0,
):
    epoch = 2_714_000_000_000_789
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([0.12, -0.08, 0.06]).as_matrix()
    transform[:3, 3] = [0.035, -0.018, 0.025]
    gravity = np.array([0.0, 9.81, 0.0])
    K = np.array([[1100.0, 0.0, 640.0], [0.0, 1095.0, 480.0], [0.0, 0.0, 1.0]])
    objects = np.array(
        [[x * 0.018, y * 0.018, 0.0] for y in range(1, 14) for x in range(1, 10)]
    )

    def state(t):
        a = np.array(
            [
                0.26 * np.sin(1.3 * t),
                0.22 * np.sin(1.7 * t + 0.4),
                0.19 * np.sin(2.1 * t + 0.1),
            ]
        )
        R = Rotation.from_euler("xyz", a).as_matrix()
        p = np.array(
            [
                0.09 + 0.065 * np.sin(1.6 * t),
                0.12 + 0.055 * np.cos(1.9 * t),
                -0.7 + 0.06 * np.sin(1.4 * t),
            ]
        )
        acc = np.array(
            [
                -0.065 * 1.6**2 * np.sin(1.6 * t),
                -0.055 * 1.9**2 * np.cos(1.9 * t),
                -0.06 * 1.4**2 * np.sin(1.4 * t),
            ]
        )
        return R, p, acc

    def timestamp(t):
        return epoch + int(round(t * 1e9))

    accel, gyro = [], []
    for t in np.arange(0.0, duration, 0.005):
        R, p, a = state(t)
        accel.append(
            {"timestamp_ns": timestamp(t), "xyz": (R.T @ (a + gravity)).tolist()}
        )
    for t in np.arange(0.0013, duration, 0.0053):
        h = 1e-5
        w = Rotation.from_matrix(state(t - h)[0].T @ state(t + h)[0]).as_rotvec() / (
            2 * h
        )
        gyro.append({"timestamp_ns": timestamp(t), "xyz": w.tolist()})
    frames = []
    for i, t in enumerate(np.arange(0.3, duration - 0.3, 0.1)):
        center = t + (exposure_ns + skew_ns) * 0.5e-9
        R, p, _ = state(center + offset_ns * 1e-9)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = p
        camera = pose @ transform
        inv = np.linalg.inv(camera)
        pixels = cv2.projectPoints(
            objects, cv2.Rodrigues(inv[:3, :3])[0], inv[:3, 3], K, None
        )[0].reshape(-1, 2)
        if skew_ns:
            # Solve each point's implicit rolling row exposure, rather than
            # applying a 2D warp to a global-shutter fixture.
            for _ in range(5):
                for j in range(len(objects)):
                    sample = (
                        t + exposure_ns * 0.5e-9 + pixels[j, 1] / 960 * skew_ns * 1e-9
                    )
                    rr, pp, _ = state(sample + offset_ns * 1e-9)
                    ci = np.eye(4)
                    ci[:3, :3], ci[:3, 3] = rr, pp
                    invrow = np.linalg.inv(ci @ transform)
                    pixels[j] = cv2.projectPoints(
                        objects[j : j + 1],
                        cv2.Rodrigues(invrow[:3, :3])[0],
                        invrow[:3, 3],
                        K,
                        None,
                    )[0].reshape(2)
        visible = (
            (pixels[:, 0] > 0)
            & (pixels[:, 0] < 1280)
            & (pixels[:, 1] > 0)
            & (pixels[:, 1] < 960)
        )
        ids = np.flatnonzero(visible)
        if len(ids) >= 20:
            frames.append(
                {
                    "timestamp_ns": timestamp(center),
                    "sensor_timestamp_ns": timestamp(t),
                    "corner_timestamps_ns": [
                        timestamp(
                            t
                            + exposure_ns * 0.5e-9
                            + pixels[j, 1] / 960 * skew_ns * 1e-9
                        )
                        for j in ids
                    ],
                    "frame_index": i,
                    "ids": ids.tolist(),
                    "pixels": pixels[visible].tolist(),
                    "T_target_camera": camera.tolist(),
                }
            )
    seed = transform.copy()
    seed[:3, :3] = seed[:3, :3] @ Rotation.from_rotvec([0.015, -0.02, 0.01]).as_matrix()
    seed[:3, 3] = 0
    payload = {
        "schema": "roomwalk.basalt_input.v1",
        "target_type": "charuco",
        "point_timing_model": "synthetic_row_exposure_midpoint",
        "object_points": objects.tolist(),
        "K": K.tolist(),
        "resolution": [1280, 960],
        "initial_T_imu_camera": seed.tolist(),
        "initial_cam_time_offset_ns": 0,
        "initial_gravity_target": gravity.tolist(),
        "accel": accel,
        "gyro": gyro,
        "frames": frames,
        "weights": {
            "gyro_sample_sigma_rad_s": 0.015,
            "accel_sample_sigma_m_s2": 0.15,
            "provenance": "synthetic test residual normalization; not phone noise",
        },
        "refine_imu_scale": True,
        "knot_spacing_ns": 60_000_000,
        "max_iterations": 60,
        "timeout_s": 240,
    }
    return payload, {
        "T_imu_camera": transform.tolist(),
        "cam_time_offset_ns": offset_ns,
        "rolling_shutter_skew_ns": skew_ns,
        "exposure_time_ns": exposure_ns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-ns", type=int, default=17_000_000)
    parser.add_argument("--skew-ns", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data, truth = fixture(
        args.offset_ns,
        skew_ns=args.skew_ns,
        exposure_ns=2_000_000 if args.skew_ns else 0,
    )
    source = args.output / "input.json"
    source.write_text(json.dumps(data, allow_nan=False))
    (args.output / "truth.json").write_text(json.dumps(truth))
    with (args.output / "solver.log").open("w") as log:
        subprocess.run(
            [str(args.executable), str(source), str(args.output / "result.json")],
            check=True,
            stdout=log,
            stderr=log,
            timeout=260,
        )
    result = json.loads((args.output / "result.json").read_text())
    got, wanted = np.array(result["T_imu_camera"]), np.array(truth["T_imu_camera"])
    errors = {
        "rotation_error_rad": float(
            Rotation.from_matrix(got[:3, :3].T @ wanted[:3, :3]).magnitude()
        ),
        "translation_error_m": float(np.linalg.norm(got[:3, 3] - wanted[:3, 3])),
        "offset_error_ns": abs(result["cam_time_offset_ns"] - args.offset_ns),
        "status": result["status"],
        "accepted_for_metric_vio": result["accepted_for_metric_vio"],
    }
    errors["passed"] = (
        errors["rotation_error_rad"] < 0.01
        and errors["translation_error_m"] < 0.01
        and errors["offset_error_ns"] < 1_000_000
        and errors["status"] == "converged"
    )
    (args.output / "validation.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(json.dumps(errors, indent=2))
    if not errors["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
