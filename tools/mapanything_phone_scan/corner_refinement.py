"""Bounded, image-only ChArUco localization from opposing checker edges.

Keep the encoded image and board geometry unchanged. White gutters/defocus can
bias a single gradient-intersection estimate toward one black quadrant. Fit all
four visible half-edges and intersect their opposing midlines instead. No camera
parameters, fitted residuals or heldout membership enter this measurement.
"""

from __future__ import annotations

import cv2
import numpy as np

POLICY = "native_opposing_checker_edges_v1"
PATCH_RADIUS = 96


def _local_geometry(ids, points, squares_x):
    grid = np.column_stack((ids % (squares_x - 1), ids // (squares_x - 1)))
    for i in range(len(ids)):
        neighbors = np.argsort(np.linalg.norm(grid - grid[i], axis=1))[:12]
        design = np.column_stack((grid[neighbors] - grid[i], np.ones(len(neighbors))))
        affine, _, rank, _ = np.linalg.lstsq(design, points[neighbors], rcond=None)
        axes = affine[:2].T
        lengths = np.linalg.norm(axes, axis=0)
        if rank < 3 or not np.isfinite(axes).all() or min(lengths) <= 0:
            yield None
            continue
        axes = axes / lengths
        if abs(np.linalg.det(axes)) < 0.25:
            yield None
            continue
        yield axes, min(64, max(12, int(0.2 * min(lengths))))


def _edge_intersection(gray, seed, axes, radius, valid=None):
    """Measure one corner in a native-pixel patch; return reason on failure."""
    v, u = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    # OpenCV ChArUco coordinates have +0.5 relative to integer pixel centres.
    sample = np.stack((u, v), axis=-1) @ axes.T + seed - 0.5
    mx, my = sample[:, :, 0].astype(np.float32), sample[:, :, 1].astype(np.float32)
    patch = cv2.remap(gray, mx, my, cv2.INTER_LINEAR)
    support = (mx >= 3) & (mx < gray.shape[1] - 4) & (my >= 3) & (my < gray.shape[0] - 4)
    if valid is not None:
        support &= cv2.remap(valid, mx, my, cv2.INTER_NEAREST) > 0
    smooth = cv2.GaussianBlur(patch.astype(float), (0, 0), 1)
    gu = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3) / 8
    gv = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3) / 8
    lines, widths, polarities = [], [], []
    for axis in range(2):
        x, y = (u, v) if axis == 0 else (v, u)
        gradient = gu if axis == 0 else gv
        for sign in (-1, 1):
            mask = ((abs(x) <= radius * 0.65) & (sign * y >= radius * 0.45)
                    & (sign * y <= radius * 0.85) & support)
            polarity = float(np.sign(np.sum(gradient[mask])))
            polarities.append(polarity)
            # Half an 8-bit intensity step suppresses flat-region quantization.
            weights = np.maximum(0, gradient * polarity - 0.5)
            rows = []
            for yy in np.unique(y[mask]):
                selected = mask & (y == yy)
                ww, xx = weights[selected], x[selected]
                total = ww.sum()
                if len(ww) < radius or total < 15:
                    continue
                edge = float(np.sum(ww * xx) / total)
                widths.append(float(np.sqrt(np.sum(ww * (xx - edge) ** 2) / total)))
                rows.append((float(yy), edge))
            if len(rows) < 5:
                return None, {"reason_codes": ["weak_or_masked_checker_edge"]}
            rows = np.asarray(rows)
            design = np.column_stack((rows[:, 0], np.ones(len(rows))))
            coefficients = np.linalg.lstsq(design, rows[:, 1], rcond=None)[0]
            rms = float(np.sqrt(np.mean((rows[:, 1] - design @ coefficients) ** 2)))
            lines.append((coefficients, rms))
    if polarities[0] * polarities[1] != -1 or polarities[2] * polarities[3] != -1:
        return None, {"reason_codes": ["inconsistent_checker_edge_polarity"]}
    # Opposing lines u=a*v+b and v=c*u+d define the two gutter midlines.
    a, b = (lines[0][0] + lines[1][0]) / 2
    c, d = (lines[2][0] + lines[3][0]) / 2
    system = np.array([[1, -a], [-c, 1]])
    if abs(np.linalg.det(system)) < 0.25:
        return None, {"reason_codes": ["degenerate_checker_edges"]}
    local = np.linalg.solve(system, [b, d])
    point = seed + axes @ local
    if not np.isfinite(point).all():
        return None, {"reason_codes": ["nonfinite_checker_intersection"]}
    return point, {
        "reason_codes": [],
        "edge_width_px": float(np.median(widths)),
        "edge_line_rms_px": max(line[1] for line in lines),
        "opposing_gap_px": [abs(lines[0][0][1] - lines[1][0][1]),
                            abs(lines[2][0][1] - lines[3][0][1])],
    }


def refine_native_corners(gray, corners, ids, marker_corners, squares_x):
    """Retain every input; any failed corner rejects its view, not just a point."""
    initial = np.asarray(corners, np.float32).reshape(-1, 2).copy()
    ids = np.asarray(ids).reshape(-1)
    points = initial.astype(float)
    evidence = {"policy": POLICY, "initial_points": initial.astype(float).tolist(), "corners": []}
    failed = False
    if (len(ids) != len(initial) or len(ids) < 6 or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(initial).all() or len(marker_corners) == 0):
        evidence["reason_codes"] = ["missing_refinement_geometry"]
        return points, evidence, ["native_corner_refinement_failed"]
    h, w = gray.shape
    for i, (original, geometry) in enumerate(zip(initial, _local_geometry(ids, initial, squares_x))):
        item = {"reason_codes": []}
        evidence["corners"].append(item)
        if geometry is None:
            item["reason_codes"].append("degenerate_local_board_geometry")
            failed = True
            continue
        axes, radius = geometry
        item["support_radius_px"] = radius
        origin = np.floor(original).astype(int) - PATCH_RADIUS
        x, y = origin
        patch = np.zeros((2 * PATCH_RADIUS + 1, 2 * PATCH_RADIUS + 1), np.uint8)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + len(patch)), min(h, y + len(patch))
        if x1 <= x0 or y1 <= y0:
            item["reason_codes"].append("corner_outside_image")
            failed = True
            continue
        patch[y0-y:y1-y, x0-x:x1-x] = gray[y0:y1, x0:x1]
        valid = np.ones(patch.shape, np.uint8)
        for marker in marker_corners:
            cv2.fillConvexPoly(valid, np.round(np.asarray(marker).reshape(-1, 2) - origin).astype(np.int32), 0)
        # Suppress marker gradients, including Gaussian/Sobel support around them.
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        valid[(xx+x < 4) | (xx+x >= w-4) | (yy+y < 4) | (yy+y >= h-4)] = 0
        local = original - origin
        current = local.copy()
        converged = False
        for iteration in range(10):
            target, measurement = _edge_intersection(patch, current, axes, radius, valid)
            item.update(measurement)
            if target is None:
                break
            if np.max(np.abs(target - local)) > radius * 0.5:
                item["reason_codes"].append("refinement_displacement_exceeded")
                break
            delta = np.max(np.abs(target - current))
            current = target
            # remap has 1/32-pixel interpolation-coordinate quantization. This
            # stopping tolerance is not the independent calibration error limit.
            if delta < 0.05:
                converged = True
                break
        item["iterations"] = iteration + 1
        points[i] = current + origin
        item["shift_px"] = float(np.linalg.norm(points[i] - original))
        if not converged and not item["reason_codes"]:
            item["reason_codes"].append("refinement_not_converged")
        failed |= bool(item["reason_codes"])
    return points, evidence, (["native_corner_refinement_failed"] if failed else [])
