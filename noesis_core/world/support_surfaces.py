"""Small, SDK-independent surface queries for calibrated world rays."""

from __future__ import annotations

import math
from dataclasses import dataclass

from noesis_core.contracts.base import Matrix3
from noesis_core.contracts.support_surface import (
    SupportSurface,
    SupportSurfaceReference,
    SupportSurfaceSet,
)


Vector = tuple[float, float, float]
RayJacobian = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class SurfaceIntersection:
    surface: SupportSurfaceReference
    point_world_m: Vector
    range_m: float
    incidence_sin: float


def contains_xz(surface: SupportSurface | SupportSurfaceReference, x: float, z: float) -> bool:
    """Include the boundary of a convex observed extent, not its bounding box."""

    if not math.isfinite(x) or not math.isfinite(z):
        return False
    vertices = surface.extent_xz
    sign = 1.0 if _signed_area_twice(vertices) > 0 else -1.0
    for i, (ax, az) in enumerate(vertices):
        bx, bz = vertices[(i + 1) % len(vertices)]
        cross = (bx - ax) * (z - az) - (bz - az) * (x - ax)
        if sign * cross < -1e-8:
            return False
    return True


def constrained_plane_covariance(
    surface: SupportSurfaceReference,
    covariance: Matrix3,
    *,
    vertical_displacement_m: float = 0.0,
) -> Matrix3:
    """Conservatively propagate a 3-D covariance through ``Y=plane(X,Z)``.

    The horizontal point remains the CI estimate. ``J C Jᵀ`` propagates its
    uncertainty through the actual plane slope. The original vertical plane
    residual ``r=Y-plane(X,Z)`` retains its variance rather than vanishing.
    Tangent/residual correlations can have either sign, so the smallest scalar
    diagonal inflation that dominates their symmetric cross term is added.
    Thus the result PSD-dominates the input C, while an already plane-aligned
    covariance is unchanged. The squared mean correction accounts for CI
    displacement, not unmeasured calibration or plane-normal error.
    """

    if not math.isfinite(vertical_displacement_m):
        raise ValueError("plane correction displacement must be finite")
    nx, ny, nz = surface.normal_world
    slope_x, slope_z = -nx / ny, -nz / ny
    v = covariance.values
    cxx, cxz, czz = float(v[0]), 0.5 * (float(v[2]) + float(v[6])), float(v[8])
    cxy, cyy, cyz = 0.5 * (float(v[1]) + float(v[3])), float(v[4]), 0.5 * (float(v[5]) + float(v[7]))

    tangent_xy = slope_x * cxx + slope_z * cxz
    tangent_yy = slope_x * tangent_xy + slope_z * (slope_x * cxz + slope_z * czz)
    tangent_yz = slope_x * cxz + slope_z * czz
    # h=(-slope_x, 1, -slope_z), r=h.p. c=J C h is its
    # tangent/residual covariance vector; C = J C Jᵀ + Var(r) eeᵀ
    # + c eᵀ + e cᵀ for e=(0,1,0).
    ch_x = cxy - slope_x * cxx - slope_z * cxz
    ch_y = cyy - slope_x * cxy - slope_z * cyz
    ch_z = cyz - slope_x * cxz - slope_z * czz
    residual_variance = max(0.0, ch_y - slope_x * ch_x - slope_z * ch_z)
    tangent_residual_y = slope_x * ch_x + slope_z * ch_z
    cross_radius = math.sqrt(ch_x * ch_x + tangent_residual_y * tangent_residual_y + ch_z * ch_z)
    # Largest eigenvalue of c eᵀ + e cᵀ is c_y + ||c||. Therefore
    # alpha*I - cross is PSD, preserving a covariance bound after projection.
    alpha = max(0.0, tangent_residual_y + cross_radius)
    out = (
        cxx + alpha, tangent_xy, cxz,
        tangent_xy, tangent_yy + residual_variance + alpha + vertical_displacement_m ** 2, tangent_yz,
        cxz, tangent_yz, czz + alpha,
    )
    if not all(math.isfinite(value) for value in out):
        raise ValueError("plane-constrained covariance must be finite")
    return Matrix3(values=out)


def elevation_at(surface: SupportSurface, x: float, z: float) -> float | None:
    if not contains_xz(surface, x, z):
        return None
    nx, ny, nz = surface.normal_world
    return -(nx * x + nz * z + surface.offset_m) / ny


def calibrated_ray_intersections(
    surfaces: SupportSurfaceSet,
    *,
    camera_id: str,
    world_frame_revision: str,
    calibration_revision: str,
    world_transform_sha256: str,
    origin_world_m: Vector,
    direction_world: Vector,
    max_range_m: float,
    min_incidence_sin: float = 0.0,
    require_authority: bool = True,
) -> tuple[SurfaceIntersection, ...]:
    """Return distinct positive ray hits; the caller must resolve ambiguity.

    The ray's world transform is owned by the caller. Exact supplied identity
    must match this set. A review-only set yields no live candidates by default.
    No nearest-plane or cross-level averaging is performed.
    """

    if (
        camera_id != surfaces.camera_id
        or world_frame_revision != surfaces.world_frame_revision
        or calibration_revision != surfaces.calibration_revision
        or world_transform_sha256 != surfaces.world_transform_sha256
        or (require_authority and surfaces.authority != "measured_calibration")
    ):
        return ()
    if not all(math.isfinite(v) for v in (*origin_world_m, *direction_world, max_range_m,
                                          min_incidence_sin)):
        return ()
    ray_norm = math.sqrt(sum(v * v for v in direction_world))
    if ray_norm <= 0 or max_range_m <= 0 or not 0 <= min_incidence_sin <= 1:
        return ()

    hits: list[SurfaceIntersection] = []
    for surface in surfaces.surfaces:
        normal = surface.normal_world
        incidence = sum(n * r for n, r in zip(normal, direction_world))
        incidence_sin = abs(incidence) / ray_norm
        if incidence_sin <= 1e-6 or incidence_sin < min_incidence_sin:
            continue
        signed_origin = sum(n * o for n, o in zip(normal, origin_world_m)) + surface.offset_m
        ray_t = -signed_origin / incidence
        distance = ray_t * ray_norm
        if ray_t <= 0 or distance > max_range_m:
            continue
        point = tuple(o + ray_t * r for o, r in zip(origin_world_m, direction_world))
        if not contains_xz(surface, point[0], point[2]):
            continue
        hits.append(SurfaceIntersection(
            surface=SupportSurfaceReference.from_surface(surface),
            point_world_m=point,  # type: ignore[arg-type]
            range_m=distance,
            incidence_sin=incidence_sin,
        ))
    return tuple(sorted(hits, key=lambda hit: hit.range_m))


def select_unique_intersection(hits: tuple[SurfaceIntersection, ...]) -> SurfaceIntersection | None:
    """Do not silently choose a level when a ray has multiple support hits."""

    return hits[0] if len(hits) == 1 else None


def intersection_covariance(
    surface: SupportSurface,
    *,
    origin_world_m: Vector,
    direction_world: Vector,
    ray_direction_jacobian: RayJacobian,
    pixel_sigma_px: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Propagate pixel-ray and measured local plane-residual uncertainty.

    ``ray_direction_jacobian`` is d(world ray direction)/d(image u,v) from the
    exact calibrated/dewarped camera. This covariance excludes calibration and
    plane-normal uncertainty; callers must add those measured terms before
    claiming an absolute position bound.
    """

    scalars = (*origin_world_m, *direction_world, pixel_sigma_px,
               *(v for row in ray_direction_jacobian for v in row))
    if not all(math.isfinite(v) for v in scalars) or pixel_sigma_px < 0:
        raise ValueError("ray and pixel uncertainty must be finite")
    n = surface.normal_world
    q = sum(a * b for a, b in zip(n, direction_world))
    if abs(q) <= 1e-6:
        raise ValueError("ray is nearly parallel to the support plane")
    signed_origin = sum(a * b for a, b in zip(n, origin_world_m)) + surface.offset_m
    t = -signed_origin / q
    if t <= 0:
        raise ValueError("ray-plane intersection must be positive")
    # dp/dr = t*I - (t/q)*r*n^T. Then chain through d(ray)/d(pixel).
    j = []
    for axis in range(3):
        row = []
        for pixel_axis in range(2):
            d_ray = tuple(ray_direction_jacobian[k][pixel_axis] for k in range(3))
            row.append(t * (d_ray[axis] - direction_world[axis]
                            * sum(n[k] * d_ray[k] for k in range(3)) / q))
        j.append(row)
    plane_j = tuple(-r / q for r in direction_world)
    pixel_variance = pixel_sigma_px * pixel_sigma_px
    plane_variance = surface.residual_sigma_m * surface.residual_sigma_m
    covariance = tuple(
        pixel_variance * sum(j[row][k] * j[col][k] for k in range(2))
        + plane_variance * plane_j[row] * plane_j[col]
        for row in range(3) for col in range(3)
    )
    return covariance  # type: ignore[return-value]


def _signed_area_twice(vertices: tuple[tuple[float, float], ...]) -> float:
    return sum(
        vertices[i][0] * vertices[(i + 1) % len(vertices)][1]
        - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
        for i in range(len(vertices))
    )
