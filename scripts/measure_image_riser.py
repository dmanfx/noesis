#!/usr/bin/env python3
"""Review-only riser metrology from a calibrated image and visible edge lines.

The lower edge is intersected with the bound calibrated floor. The upper edge
is intersected with gravity lines through those lower-edge points. This measures
a vertical front face, not a complete upper support plane or its footprint.
No person track, phone optical center, or assumed step height enters the solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def measure(calibration: dict, edges: dict) -> dict:
    camera = calibration["cameras"][edges["camera_id"]]
    binding = calibration["frame_binding"]
    if binding["world_frame"]["frame_id"] != "backend_world_m":
        raise ValueError("gravity direction requires the canonical Y-up world frame")
    world_from_camera = (
        np.asarray(binding["target_from_calibration_col_major"]).reshape(4, 4, order="F")
        @ np.linalg.inv(np.asarray(camera["E"]).reshape(4, 4, order="F"))
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    fx, fy, cx, cy = camera["K"]
    intrinsics = np.asarray([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    plane = binding["world_floor_plane"]
    normal = np.asarray(plane["normal"])
    origin = world_from_camera[:3, 3]
    gravity_projection = intrinsics @ camera_from_world[:3, 1]
    lower = np.asarray(edges["lower_edge_pixels"], dtype=float)
    upper = np.asarray(edges["upper_edge_pixels"], dtype=float)
    if lower.shape != (2, 2) or upper.shape != (2, 2):
        raise ValueError("each edge needs two rectified source-image endpoints")
    if not np.isfinite(np.r_[lower.ravel(), upper.ravel()]).all():
        raise ValueError("edge pixels must be finite")

    def solve(bottom, top):
        line = np.cross(np.r_[top[0], 1.], np.r_[top[1], 1.])
        denominator = float(line @ gravity_projection)
        if abs(denominator) < 1e-8:
            raise ValueError("top edge does not constrain vertical height")
        samples = []
        for fraction in (.15, .35, .55, .75, .9):
            pixel = (1. - fraction) * bottom[0] + fraction * bottom[1]
            ray = world_from_camera[:3, :3] @ np.linalg.solve(intrinsics, np.r_[pixel, 1.])
            incidence = float(normal @ ray)
            if abs(incidence) < 1e-8:
                raise ValueError("lower edge is at the calibrated floor horizon")
            distance = -(float(normal @ origin) + plane["offset_m"]) / incidence
            if distance <= 0:
                raise ValueError("lower edge floor intersection is behind camera")
            base = origin + distance * ray
            projected_base = intrinsics @ (camera_from_world[:3, :3] @ base + camera_from_world[:3, 3])
            height = -float(line @ projected_base) / denominator
            raised = projected_base + height * gravity_projection
            top_pixel = raised[:2] / raised[2]
            top_ray = world_from_camera[:3, :3] @ np.linalg.solve(intrinsics, np.r_[top_pixel, 1.])
            flat_distance = -(float(normal @ origin) + plane["offset_m"]) / float(normal @ top_ray)
            flat_hit = origin + flat_distance * top_ray
            samples.append({"bottom_pixel": pixel.tolist(), "top_pixel": (raised[:2] / raised[2]).tolist(),
                            "base_world_m": base.tolist(), "vertical_height_m": height,
                            "top_pixel_on_lower_floor_world_m": flat_hit.tolist(),
                            "single_floor_horizontal_bias_at_riser_top_m": float(np.linalg.norm((flat_hit - base)[[0, 2]]))})
        return samples

    samples = solve(lower, upper)
    # Endpoint sensitivity only: these are not independent surveys, confidence
    # intervals, nor a substitute for calibration / lens / nosing uncertainty.
    delta = float(edges["endpoint_sensitivity_px"])
    if not 0 < delta <= 10:
        raise ValueError("endpoint sensitivity must be in (0, 10] pixels")
    medians = []
    for edge_index in range(2):
        for endpoint in range(2):
            for axis in range(2):
                for sign in (-1., 1.):
                    trial = [lower.copy(), upper.copy()]
                    trial[edge_index][endpoint, axis] += sign * delta
                    medians.append(float(np.median([s["vertical_height_m"] for s in solve(*trial)])))
    return {"authority": "review_only", "method": "calibrated_floor_vertical_riser_edge_intersection",
            "camera_id": edges["camera_id"], "world_binding": binding,
            "annotations": edges, "samples": samples,
            "median_vertical_height_m": float(np.median([s["vertical_height_m"] for s in samples])),
            "endpoint_sensitivity_median_range_m": [min(medians), max(medians)],
            "limitations": ["Requires correctly identified top and bottom of a vertical front face.",
                            "Nosing overhang, occlusion, lens residual, floor and camera calibration errors are not bounded here.",
                            "Samples share one image and calibration; their spread is not independent measurement uncertainty.",
                            "This measures a riser only, not the upper floor normal, finite extent, or a live-admissible surface."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(json.loads(args.calibration.read_text()), json.loads(args.edges.read_text()))
    result["input_sha256"] = {"calibration": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                              "edges": hashlib.sha256(args.edges.read_bytes()).hexdigest()}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("median_vertical_height_m", "endpoint_sensitivity_median_range_m", "authority")}))


if __name__ == "__main__":
    main()
