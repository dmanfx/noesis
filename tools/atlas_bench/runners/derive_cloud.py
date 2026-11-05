"""Bounded point-cloud derivation from downloaded Atlas bundles.

The output stays in the Atlas-estimated gauge. This does not register it or
promote it into PCF, calibration, Scene Prior or tracking authority.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from pathlib import Path

from runners.worldlabs import BenchError, private_json


def derive(run: Path, destination: Path, stride: int = 8, max_points: int = 1_000_000) -> dict:
    import cv2
    import numpy as np
    from PIL import Image
    import json
    read = lambda p: json.loads(p.read_text())
    state = read(run / "run.json")
    if state["state"] != "completed":
        raise BenchError("Derivation requires a completed parsed RGBD run")
    if not 1 <= stride <= 64 or not 1 <= max_points <= 2_000_000:
        raise BenchError("Derivation must use stride 1-64 and at most two million points")
    response = read(run / "response.json")
    artifacts = read(run / "artifacts.json")
    lookup = {row["field"]: run / row["path"] for row in artifacts}
    per_frame_limit = max_points // max(1, len(response["frames"]))
    chunks = []
    for index, frame in enumerate(response["frames"]):
        depth_path = lookup.get(f"frames.{index}.depth.depthAsset")
        if depth_path is None:
            raise BenchError("Point derivation needs linear-depth EXRs for every frame")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        camera = frame["camera"]
        intrinsics, extrinsics = camera["intrinsics"], camera["extrinsics"]
        yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[::stride, ::stride]
        valid = np.isfinite(z) & (z > 0)
        points = np.column_stack(((xx[valid] - intrinsics["cx"]) / intrinsics["fx"] * z[valid],
                                  (yy[valid] - intrinsics["cy"]) / intrinsics["fy"] * z[valid], z[valid]))
        if extrinsics["coordinateSystem"] == "rub":
            points *= [1, -1, -1]
        x, y, zq, w = extrinsics["quaternion"]
        rotation = np.array([[1-2*(y*y+zq*zq), 2*(x*y-zq*w), 2*(x*zq+y*w)],
                             [2*(x*y+zq*w), 1-2*(x*x+zq*zq), 2*(y*zq-x*w)],
                             [2*(x*zq-y*w), 2*(y*zq+x*w), 1-2*(x*x+y*y)]])
        points = points @ rotation.T + extrinsics["position"]
        with Image.open(lookup[f"frames.{index}.imageAsset"]) as image:
            colors = np.asarray(image.convert("RGB"))[::stride, ::stride][valid]
        # A deterministic bound without frame-count-dependent memory growth.
        sample = np.linspace(0, len(points)-1, min(len(points), per_frame_limit), dtype=int)
        chunks.append((points[sample], colors[sample]))
    count = sum(len(points) for points, _ in chunks)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise BenchError("Derived output exists; select a new --derived-id") from exc
    path = destination / "atlas_points.ply"
    with path.open("wb") as handle:
        handle.write((f"ply\nformat binary_little_endian 1.0\ncomment Atlas estimated gauge; metric scale unverified\n"
                      f"element vertex {count}\nproperty float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        for points, colors in chunks:
            records = np.empty(len(points), dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
            records["xyz"], records["rgb"] = points, colors
            handle.write(records.tobytes())
    path.chmod(0o600)
    provenance = {"source_run": str(run), "baseline_sha256": state["baseline_sha256"],
                  "operation_id": state["operation_id"], "source_artifacts": artifacts,
                  "point_count": count, "stride": stride, "max_points": max_points,
                  "coordinate_frame": "Atlas estimated world; unregistered",
                  "distance_unit": "estimated scene units; metric accuracy unverified",
                  "depth_rows": "OpenCV top-left; no additional vertical flip",
                  "authority": "derived offline review only"}
    private_json(destination / "derivation.json", provenance)
    return {"cloud": str(path), "point_count": count, "provenance": str(destination / "derivation.json")}
