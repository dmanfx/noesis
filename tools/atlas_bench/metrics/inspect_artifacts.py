"""Parse Atlas bundles without importing or changing any PCF implementation."""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path

from runners.worldlabs import BenchError


def dependencies() -> dict:
    import cv2
    import numpy as np
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory(prefix="atlas-exr-check-") as folder:
        path = str(Path(folder) / "parser.exr")
        sample = np.array([[1., 2.], [np.nan, 4.]], dtype=np.float32)
        if not cv2.imwrite(path, sample):
            raise BenchError("OpenCV cannot encode an EXR parser fixture")
        decoded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.shape != (2, 2) or not np.isnan(decoded[1, 0]):
            raise BenchError("OpenCV EXR parser failed its orientation/NaN check")
    return {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__,
            "exr": "float32; top-left rows; NaN preserved"}


def validate_camera(camera: dict, resolution: list[int] | None = None) -> None:
    import numpy as np
    try:
        intrinsics, extrinsics = camera["intrinsics"], camera["extrinsics"]
        w, h = intrinsics["width"], intrinsics["height"]
        values = [intrinsics[k] for k in ("fx", "fy", "cx", "cy")]
        position, quaternion = extrinsics["position"], extrinsics["quaternion"]
        if (w <= 0 or h <= 0 or not np.isfinite(values).all()
                or min(values[:2]) <= 0 or len(position) != 3 or len(quaternion) != 4
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or abs(float(np.linalg.norm(quaternion)) - 1.) > 0.002
                or extrinsics["coordinateSystem"] not in ("rub", "rdf")):
            raise ValueError("invalid camera fields")
        if resolution and [w, h] != resolution:
            raise ValueError("camera output grid differs from requested resolution")
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchError(f"Unparseable Atlas camera: {exc}") from exc


def inspect_frames(run: Path, response: dict, artifacts: list[dict],
                   expected_count: int, require_depth: bool = True) -> dict:
    import cv2
    import numpy as np
    from PIL import Image
    frames = response.get("frames", [])
    if len(frames) != expected_count:
        raise BenchError(f"Expected {expected_count} returned frames, got {len(frames)}")
    lookup = {row["field"]: run / row["path"] for row in artifacts}
    inspected = []
    for index, frame in enumerate(frames):
        validate_camera(frame["camera"], [1280, 720])
        intrinsics = frame["camera"]["intrinsics"]
        expected_shape = (intrinsics["height"], intrinsics["width"])
        prefix = f"frames.{index}"
        rgb = lookup.get(f"{prefix}.imageAsset")
        if rgb is None:
            raise BenchError("Returned frame is missing RGB")
        with Image.open(rgb) as image:
            image.load()
            if image.size != expected_shape[::-1]:
                raise BenchError("Returned RGB does not match its camera grid")
        row = {"index": index, "rgb_grid": list(expected_shape[::-1]),
               "coordinate_system": frame["camera"]["extrinsics"]["coordinateSystem"]}
        depth_path = lookup.get(f"{prefix}.depth.depthAsset")
        if require_depth and depth_path is None:
            raise BenchError("Returned frame is missing linear depth EXR")
        if depth_path:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.ndim != 2 or depth.shape != expected_shape or depth.dtype.kind != "f":
                raise BenchError("Depth EXR is not a single-channel float image on the camera grid")
            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                raise BenchError("Depth EXR contains no finite positive geometry")
            row["depth"] = {"dtype": str(depth.dtype), "grid": list(depth.shape[::-1]),
                            "finite_positive_fraction": float(valid.mean()),
                            "nonfinite_fraction": float((~np.isfinite(depth)).mean()),
                            "percentiles_world_units": np.percentile(depth[valid], [5, 50, 95]).tolist(),
                            "unit_status": "Atlas estimated scene units; metric scale unverified"}
        confidence_path = lookup.get(f"{prefix}.depth.confidenceAsset")
        if confidence_path:
            with Image.open(confidence_path) as confidence:
                confidence.load()
                if confidence.size != expected_shape[::-1]:
                    raise BenchError("Confidence grid differs from camera/depth")
                row["confidence"] = {"mode": confidence.mode, "grid": list(confidence.size),
                                     "interpretation": "provider score, not calibrated probability"}
        inspected.append(row)
    return {"status": "parsed", "frames": inspected,
            "authority": "offline review only", "pcf_modified": False,
            "accuracy_acceptance": "not measured by smoke"}


def metric_hooks() -> dict:
    requirements = {
        "plane_wall_floor_residuals": "Atlas-to-PCF registration and independent labeled planes",
        "cross_view_consistency": "joint posed views and occlusion masks",
        "metric_scale_error": "independent known metric geometry; scale both depth and cameras",
        "coverage_completeness": "matched crop and visibility; distinguish missing from generated",
        "floaters_outliers": "registered cloud with finite-depth and independent visibility masks",
        "heldout_rgb_agreement": "untouched real holdouts and generated views at matched cameras",
        "heldout_depth_agreement": "heldout PCF/DA3 comparison, marked correlated; static separately",
        "confidence_vs_error": "matched independent residuals; confidence is not a probability",
        "chisel_geometry_preservation": "input rendered planar-z and returned depth in same metric frame",
    }
    return {name: {"state": "not_measured", "requires": requirement}
            for name, requirement in requirements.items()}
