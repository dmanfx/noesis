"""Offline, image-region-supported planes from exact admitted PCF points.

Regions identify physical surfaces; they do not prescribe normal, location,
floor height, authored correspondence or a presentation transform. This module
does not participate in runtime calibration or relax structural admission.
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from noesis.calibration.structural_registration import StructuralRegistrationError

MAX_POINTS = 2_000_000
MAX_REGION_POINTS = 100_000


def _points(values: Any, *, maximum: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 3 or not 3 <= len(array) <= maximum
            or not np.isfinite(array).all()):
        raise StructuralRegistrationError(f"{label}: expected bounded finite Nx3 points")
    return array


def select_admitted_region_support(
    admitted_points: Any, region_points: Any, *, maximum_distance_m: float,
) -> np.ndarray:
    """Select existing GLB indices near quality-admitted observed ROI geometry.

    No generated/nearest substituted points are returned. The distance is an
    explicit evidence-selection tolerance, not an accuracy or validity claim.
    It may not exceed the existing 18 cm carrier/PCF association slab.
    """
    cloud = _points(admitted_points, maximum=MAX_POINTS, label="admitted PCF")
    region = _points(region_points, maximum=MAX_REGION_POINTS, label="source region")
    if (isinstance(maximum_distance_m, bool) or not math.isfinite(maximum_distance_m)
            or not 0 < maximum_distance_m <= 0.18):
        raise StructuralRegistrationError("region association distance must be explicit and at most 18 cm")
    distance, _ = cKDTree(region).query(cloud, distance_upper_bound=maximum_distance_m, workers=1)
    return np.flatnonzero(np.isfinite(distance))


def estimate_admitted_normals(admitted_points: Any) -> np.ndarray:
    """Offline local surface normals at the existing extractor's neighborhood."""
    import open3d as o3d

    points = _points(admitted_points, maximum=MAX_POINTS, label="admitted PCF")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.18, max_nn=30))
    return np.asarray(cloud.normals).copy()


def grow_connected_plane_support(admitted_points: Any, seed_indices: Any,
                                 seed_plane: dict[str, Any], *, local_normals: Any = None) -> np.ndarray:
    """Recover connected PCF support around an image-measured plane seed.

    Keep the original observed plane fixed for selection: no iterative plane
    chasing, authored normal or target offset. The 5 cm slab and 6 cm voxel
    match the existing global extractor. Connectivity tolerates at most one
    empty voxel; it cannot substitute geometry absent from the admitted PCF.
    Local normals must be vertical and agree with the measured seed within
    30 degrees. This excludes a horizontal counter/floor intersection strip;
    it does not prescribe the wall's direction. The returned support still
    needs an unconstrained fit and unchanged support gates.
    """
    cloud = _points(admitted_points, maximum=MAX_POINTS, label="admitted PCF")
    seeds = np.asarray(seed_indices)
    if (seeds.ndim != 1 or seeds.dtype.kind not in "iu" or len(seeds) < 3
            or seeds.min() < 0 or seeds.max() >= len(cloud)):
        raise StructuralRegistrationError("invalid bounded plane seed indices")
    n = np.asarray(seed_plane["normal_xyz"], dtype=float)
    d = float(seed_plane["offset_m"])
    if (n.shape != (3,) or not np.isfinite(n).all() or not math.isfinite(d)
            or not math.isclose(float(np.linalg.norm(n)), 1, abs_tol=1e-6)):
        raise StructuralRegistrationError("invalid observed plane seed")
    normals = estimate_admitted_normals(cloud) if local_normals is None else np.asarray(local_normals)
    if normals.shape != cloud.shape or not np.isfinite(normals).all():
        raise StructuralRegistrationError("invalid admitted local normals")
    candidates = np.flatnonzero((np.abs(cloud@n+d) <= .05) & (np.abs(normals[:, 1]) < .40)
                               & (np.abs(normals@n) >= math.cos(math.radians(30))))
    voxels = np.floor(cloud[candidates]/.06).astype(np.int64)
    unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    if len(unique) > 200_000:
        raise StructuralRegistrationError("connected plane support exceeds voxel work bound")
    lookup = {tuple(v): i for i, v in enumerate(unique)}
    allowed_seeds = np.intersect1d(candidates, seeds)
    seed_cells = {lookup[tuple(v)] for v in np.floor(cloud[allowed_seeds]/.06).astype(np.int64)}
    if not seed_cells:
        return np.empty(0, dtype=int)
    # Axis-neighbor jumps are bounded to two voxels. No diagonal jump across
    # an unrelated surface; ordinary face/edge/corner neighbors are retained.
    neighbors = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                 for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
    neighbors += [(d, 0, 0) for d in (-2, 2)] + [(0, d, 0) for d in (-2, 2)] + [(0, 0, d) for d in (-2, 2)]
    # Prefer the component with most image-observed seed cells, not the largest
    # arbitrary coplanar room surface. Other seed islands remain diagnostics.
    remaining = set(seed_cells)
    best = set()
    best_seed_count = 0
    while remaining:
        first = min(remaining)
        component = {first}
        pending = deque([first])
        while pending:
            cell = unique[pending.popleft()]
            for delta in neighbors:
                neighbor = lookup.get(tuple(cell+delta))
                if neighbor is not None and neighbor not in component:
                    component.add(neighbor); pending.append(neighbor)
        seed_count = len(component & seed_cells)
        remaining.difference_update(component)
        if seed_count > best_seed_count:
            best, best_seed_count = component, seed_count
    return candidates[np.isin(inverse, list(best))]


def fit_supported_region_plane(points: Any) -> dict[str, Any]:
    """Measure an unconstrained 3D plane without forcing a wall upright.

    The normal is estimated from observed points, including measured tilt.
    Existing global-extraction support/height/length/tilt bounds are retained;
    independent holdouts and the structural solver remain separate gates.
    """
    cloud = _points(points, maximum=MAX_POINTS, label="plane support")
    center = np.median(cloud, axis=0)
    sample = cloud[np.linspace(0, len(cloud)-1, min(len(cloud), 24_000), dtype=int)]
    _, singular_values, basis = np.linalg.svd(sample-center, full_matrices=False)
    if singular_values[1] <= 1e-9:
        raise StructuralRegistrationError("region support is a line, not an observable plane")
    seed = basis[-1]
    angles = [math.atan2(seed[2], seed[0]), math.asin(float(np.clip(seed[1], -1, 1))), 0.]

    def normal(parameters):
        yaw, elevation = parameters[:2]
        return np.asarray([math.cos(elevation)*math.cos(yaw), math.sin(elevation),
                           math.cos(elevation)*math.sin(yaw)])

    solved = least_squares(lambda p: (sample-center)@normal(p)+p[2], angles,
                           loss="soft_l1", f_scale=0.03, max_nfev=200)
    n = normal(solved.x)
    d = float(solved.x[2] - n@center)
    if n[np.flatnonzero(np.abs(n)>1e-9)[0]] < 0:
        n, d = -n, -d
    residual = cloud@n+d
    p90_distance = float(np.quantile(np.abs(residual), .90))
    horizontal = n[[0,2]]
    horizontal_norm = float(np.linalg.norm(horizontal))
    tangent = np.asarray([-horizontal[1],horizontal[0]])/max(horizontal_norm,1e-12)
    interval = np.percentile(cloud[:,[0,2]]@tangent, [2,98])
    height_span = float(np.ptp(cloud[:,1]))
    voxel_support_count = len(np.unique(np.floor(cloud / 0.06).astype(np.int64), axis=0))
    reasons = []
    if not solved.success: reasons.append("plane_optimizer_failed")
    if voxel_support_count<200: reasons.append("support_count")
    if abs(float(n[1]))>.30 or horizontal_norm<=.95: reasons.append("plane_tilt")
    if height_span<1.20: reasons.append("height_span")
    if interval[1]-interval[0]<.80: reasons.append("tangent_span")
    # This reports fit residuals rather than trimming inconvenient points away.
    sorted_points = np.round(cloud,6)
    sorted_points = sorted_points[np.lexsort((sorted_points[:,2], sorted_points[:,1], sorted_points[:,0]))]
    return {"status":"review_only", "normal_xyz":n.tolist(), "offset_m":d,
            "point_count":len(cloud), "vertical_span_m":height_span,
            "voxel_support_count":voxel_support_count,
            "tangent_interval_m":interval.tolist(), "reference_y_m":float(np.median(cloud[:,1])),
            "median_abs_distance_m":float(np.median(np.abs(residual))),
            "p90_abs_distance_m":p90_distance,
            "singular_values":singular_values.tolist(),
            "support_xyz_sha256":hashlib.sha256(sorted_points.astype('<f8').tobytes()).hexdigest(),
            "extraction_support_passed":not reasons, "reason_codes":reasons,
            "coherent_plane_support_passed":p90_distance <= .05,
            "review_candidate_passed":not reasons and p90_distance <= .05,
            "limitation":"Support extraction alone does not admit a structural mapping or prove physical accuracy."}


def line_at_observed_height(plane: dict[str, Any], *, segment_id: str,
                            reference_y_m: float | None = None) -> dict[str, Any]:
    """Slice at a measured support height, optionally shared across surfaces.

    Callers supplying a common height must derive it from the observed support;
    this is not a floor-height or upright-wall assumption.
    """
    n = np.asarray(plane['normal_xyz'], dtype=float)
    hnorm = float(np.linalg.norm(n[[0,2]]))
    if hnorm<=.95:
        raise StructuralRegistrationError("plane is not near vertical")
    horizontal = n[[0,2]]/hnorm
    tangent = np.asarray([-horizontal[1],horizontal[0]])
    reference_y = float(plane['reference_y_m'] if reference_y_m is None else reference_y_m)
    if not math.isfinite(reference_y):
        raise StructuralRegistrationError("plane slice height must be finite")
    offset = -(float(plane['offset_m'])+n[1]*reference_y)/hnorm
    lo,hi = plane['tangent_interval_m']
    return {'segment_id':segment_id,'start_xz':(offset*horizontal+lo*tangent).tolist(),
            'end_xz':(offset*horizontal+hi*tangent).tolist(),
            'support_count':plane['voxel_support_count'],'vertical_span':plane['vertical_span_m'],
            'source_refs':['method:admitted_pcf_image_roi_plane',
                           'extent:observed_support_not_architectural_endpoints',
                           'support_sha256:'+plane['support_xyz_sha256'],
                           'normal_xyz:'+','.join(format(v,'.17g') for v in n),
                           'plane_d_m:'+format(plane['offset_m'],'.17g'),
                           'reference_y_m:'+format(reference_y,'.17g')]}
