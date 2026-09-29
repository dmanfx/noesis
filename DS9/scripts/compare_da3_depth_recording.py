#!/usr/bin/env python3
"""Small offline depth comparison on retained, calibrated static-camera frames.

This does not change the runtime. PCF agreement is a geometric consistency
check, not independent ground-truth depth accuracy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh


def trt_outputs(engine: Path, name: str, images: np.ndarray, out: Path, tag: str):
    input_file = out / f"{tag}-{name}-input.raw"
    output_file = out / f"{tag}-{name}-output.json"
    times_file = out / f"{tag}-{name}-times.json"
    input_file.write_bytes(np.ascontiguousarray(images, dtype=np.float32).tobytes())
    command = [
        "trtexec", f"--loadEngine={engine}",
        f"--loadInputs={name}:{input_file}",
        f"--exportOutput={output_file}", f"--exportTimes={times_file}",
        "--warmUp=100", "--duration=1", "--iterations=1",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{engine.name}: {result.stderr[-3000:]} {result.stdout[-3000:]}")
    outputs = {}
    for row in json.loads(output_file.read_text()):
        shape = tuple(int(value) for value in row["dimensions"].split("x"))
        outputs[row["name"]] = np.asarray(row["values"], np.float32).reshape(shape)
    times = json.loads(times_file.read_text())
    return outputs, float(np.median([float(row["computeMs"]) for row in times]))


def pcf_zbuffer(pcf: Path, calibration: Path, manifest: dict) -> np.ndarray:
    scene = trimesh.load(pcf)
    xyz = np.asarray(next(iter(scene.geometry.values())).vertices, dtype=np.float64)
    camera = json.loads(calibration.read_text())
    E = np.asarray(camera["cameras"]["living-room"]["E"]).reshape(4, 4, order="F")
    T = np.asarray(camera["frame_binding"]["target_from_calibration_col_major"]).reshape(4, 4, order="F")
    E = E @ np.linalg.inv(T)
    xyz_cam = xyz @ E[:3, :3].T + E[:3, 3]
    z = xyz_cam[:, 2]
    fx, fy, cx, cy = camera["cameras"]["living-room"]["K"]
    u_rect = fx * xyz_cam[:, 0] / z + cx
    v_rect = fy * xyz_cam[:, 1] / z + cy
    A = np.asarray(manifest["source"]["frames"][0]["rectified_to_model_pixel_transform"])
    u = np.rint(A[0, 0] * u_rect + A[0, 2]).astype(np.int64)
    v = np.rint(A[1, 1] * v_rect + A[1, 2]).astype(np.int64)
    valid = np.isfinite(z) & (z > 0.4) & (z < 12) & (u >= 0) & (u < 518) & (v >= 0) & (v < 294)
    flat = np.full(294 * 518, np.inf, dtype=np.float32)
    np.minimum.at(flat, v[valid] * 518 + u[valid], z[valid].astype(np.float32))
    return flat.reshape(294, 518)


def robust_affine(x: np.ndarray, y: np.ndarray):
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    for _ in range(4):
        scale, offset = np.polyfit(x, y, 1)
        residual = y - (scale * x + offset)
        center = np.median(residual)
        mad = np.median(np.abs(residual - center)) + 1e-6
        select = np.abs(residual - center) < 3 * 1.4826 * mad
        x, y = x[select], y[select]
    return float(scale), float(offset), int(len(x))


def error_summary(pred: np.ndarray, reference: np.ndarray, mask: np.ndarray):
    valid = mask & np.isfinite(pred) & (pred > 0)
    e = np.abs(pred[valid] - np.broadcast_to(reference, pred.shape)[valid])
    return {"pixels": int(len(e)), "median_abs_m": float(np.median(e)),
            "p90_abs_m": float(np.quantile(e, .9)),
            "within_0_5m": float(np.mean(e < .5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--pcf", type=Path, required=True)
    parser.add_argument("--metric-engine", type=Path, required=True)
    parser.add_argument("--small-engine", type=Path, required=True)
    parser.add_argument("--dav2-engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.static_dir / "manifest.json").read_text())
    files = [args.static_dir / f"static_{i:04d}.npz" for i in range(6)]
    rgb = [np.load(path, allow_pickle=False)["model_rgb"].copy() for path in files]
    ref = pcf_zbuffer(args.pcf, args.static_dir / "calibration.json", manifest)
    mask = np.isfinite(ref)
    # Exclude image borders, where both dewarping and sparse PCF support are poor.
    mask[:8] = False; mask[-8:] = False; mask[:, :8] = False; mask[:, -8:] = False
    # Pixel-wise median around each sparse point provides modest tolerance to
    # point rasterization without manufacturing unsupported PCF geometry.
    model_predictions = {}
    timings = {}
    for model, engine, input_name in (
        ("da3_small", args.small_engine, "images"),
        ("da3metric_large", args.metric_engine, "images"),
        ("dav2", args.dav2_engine, "input"),
    ):
        depths, compute = [], []
        for batch in range(2):
            tensor = np.stack([image.transpose(2, 0, 1) for image in rgb[batch * 3:(batch + 1) * 3]])
            outputs, timing = trt_outputs(engine, input_name, tensor, args.out, f"{model}-{batch}")
            value = outputs["depth"]
            depths.extend(np.asarray(value, np.float32).reshape(3, 294, 518))
            compute.append(timing)
        model_predictions[model] = np.asarray(depths)
        timings[model] = float(np.median(compute))

    camera = json.loads((args.static_dir / "calibration.json").read_text())
    fx = camera["cameras"]["living-room"]["K"][0]
    model_fx = fx * manifest["source"]["frames"][0]["rectified_to_model_pixel_transform"][0][0]
    model_predictions["da3metric_large"] *= model_fx / 300.0

    # Calibrate relative and existing metric heads from the earlier PCF using
    # frames 0..2; report only frames 3..5 as comparison observations.
    mappings = {}
    for name in ("da3_small", "dav2"):
        train_x = model_predictions[name][:3, mask].ravel()
        train_y = np.broadcast_to(ref[mask], (3, int(mask.sum()))).ravel()
        mappings[name] = robust_affine(train_x, train_y)

    heldout = {}
    for name, depths in model_predictions.items():
        native = depths[3:]
        summary = {"native": error_summary(native, ref, np.broadcast_to(mask, native.shape))}
        if name in mappings:
            scale, offset, _ = mappings[name]
            compared = scale * native + offset
            summary["pcf_mapped"] = error_summary(compared, ref,
                                                   np.broadcast_to(mask, native.shape))
        else:
            compared = native
        summary["pcf_depth_bins"] = {}
        for lo, hi in ((1, 2), (2, 4), (4, 8)):
            subset = mask & (ref >= lo) & (ref < hi)
            if subset.any():
                summary["pcf_depth_bins"][f"{lo}-{hi}m"] = error_summary(
                    compared, ref, np.broadcast_to(subset, native.shape))
        heldout[name] = summary

    # Raw depth range separation and stationary-scene temporal consistency.
    variability = {}
    for name, depths in model_predictions.items():
        median = np.median(depths, axis=0)
        deviation = np.median(np.abs(depths - median[None]), axis=0)
        variability[name] = {"median_frame_deviation_raw": float(np.median(deviation[mask])),
                             "p90_frame_deviation_raw": float(np.quantile(deviation[mask], .9)),
                             "p10_raw": float(np.quantile(depths[:, mask], .1)),
                             "p90_raw": float(np.quantile(depths[:, mask], .9))}
    report = {"frames_pts_s": [float(np.load(path)["encoded_pts_ns"][0]) / 1e9 for path in files],
              "pcf_pixels": int(mask.sum()), "pcf_point_cloud": str(args.pcf),
              "comparison_note": "PCF is an earlier multiview phone reconstruction that included DA3Metric-Large; agreement is not independent accuracy",
              "input_note": "same retained 294x518 rectified MapAnything RGB tensors for all models; native DAv2 live stretch differs slightly",
              "metric_rule": f"DA3Metric raw depth times calibrated model fx / 300 = {model_fx / 300:.6f}",
              "affine_mapping": {k: {"scale": v[0], "offset_m": v[1], "fit_pixels": v[2]}
                                 for k, v in mappings.items()},
              "heldout_frames_3_5": heldout, "variability": variability,
              "median_batch3_compute_ms": timings}
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
