"""Offline, source-aware completion of static-camera floorplan shadows.

The functions in this module never mutate the raw static or phone inputs.  They
derive a visibility mask from the static camera, prepare a phone surface from
multi-view voxels, and admit phone authority only where both conditions hold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ShadowFillSettings:
    """Metric thresholds for the offline shadow-completion pass."""

    occluder_min_height_m: float = 0.25
    occluder_max_height_m: float = 1.80
    occluder_clearance_m: float = 0.05
    occluder_min_component_cells: int = 4
    occluder_dilation_cells: int = 1
    wall_support_min: float = 0.10
    angular_step_deg: float = 0.30
    minimum_camera_range_m: float = 0.20
    phone_surface_radius_m: float = 0.04
    phone_height_bin_m: float = 0.08
    # Every eligible scene-fusion voxel already contains at least two phone
    # views.  One voxel is therefore sufficient evidence for its own 5 cm
    # footprint; requiring neighbouring voxels would add no new view evidence.
    phone_min_cluster_points: int = 1
    phone_min_voxel_confidence: float = 0.05
    phone_max_height_m: float = 1.80
    phone_max_surface_spread_m: float = 0.12
    phone_upper_mode_min_weight_ratio: float = 0.45
    phone_min_quality: float = 0.35
    static_surface_agreement_m: float = 0.22
    phone_higher_surface_margin_m: float = 0.12
    artifact_smoothing_sigma_m: float = 0.03
    artifact_window_m: float = 0.28
    artifact_energy_percentile: float = 70.0
    artifact_min_radial_alignment: float = 0.45
    artifact_min_oscillation_fraction: float = 0.40
    artifact_min_height_residual_m: float = 0.35
    artifact_region_min_height_residual_m: float = 0.20
    artifact_max_density: float = 0.005
    artifact_min_component_cells: int = 4
    artifact_dilation_m: float = 0.08
    region_phone_min_views: int = 3
    region_min_phone_cells: int = 3
    region_min_phone_coverage: float = 0.20
    region_floor_max_phone_height_m: float = 0.35
    region_floor_min_fraction: float = 0.70
    region_floor_max_spread_m: float = 0.20
    region_flat_min_phone_coverage: float = 0.35
    region_flat_max_spread_m: float = 0.25
    phone_floor_region_max_gap_m: float = 0.12
    phone_floor_region_raised_margin_m: float = 0.02
    phone_floor_region_min_evidence_cells: int = 3
    phone_floor_visibility_max_height_m: float = 0.12
    phone_floor_visibility_min_views: int = 3
    floor_artifact_min_phone_cells: int = 12
    floor_artifact_min_phone_fraction: float = 0.70
    floor_artifact_min_foreground_cells: int = 4
    floor_artifact_expansion_m: float = 0.12
    floor_artifact_max_density: float = 0.0075
    floor_artifact_max_island_cells: int = 3


@dataclass(frozen=True)
class OcclusionShadow:
    shadow: np.ndarray
    occluder: np.ndarray
    occluder_height_m: np.ndarray


@dataclass(frozen=True)
class PhoneSurface:
    height_m: np.ndarray
    quality: np.ndarray
    support: np.ndarray
    spread_m: np.ndarray
    good: np.ndarray
    eligible_voxel_count: int


@dataclass(frozen=True)
class RadialOcclusionArtifact:
    seed: np.ndarray
    region: np.ndarray
    radial_curvature_energy: np.ndarray
    radial_alignment: np.ndarray
    oscillation_fraction: np.ndarray
    energy_threshold: float


@dataclass(frozen=True)
class RegionSurfaceCompletion:
    admitted: np.ndarray
    floor: np.ndarray
    flat: np.ndarray
    height_m: np.ndarray
    component_id: np.ndarray
    component_support: np.ndarray
    component_coverage: np.ndarray
    component_spread_m: np.ndarray
    floor_component_count: int
    flat_component_count: int


@dataclass(frozen=True)
class PhoneFloorRegionCompletion:
    admitted: np.ndarray
    direct_evidence: np.ndarray
    interpolated: np.ndarray
    nearest_floor_distance_m: np.ndarray
    quality: np.ndarray
    support: np.ndarray
    spread_m: np.ndarray
    component_id: np.ndarray
    component_count: int


@dataclass(frozen=True)
class PhoneFloorVisibility:
    visible: np.ndarray
    view_support: np.ndarray
    quality: np.ndarray
    floor_point_count: int
    unique_view_cell_count: int


@dataclass(frozen=True)
class FloorUnderlayMasks:
    """Disjoint floor and trusted-foreground layers on the static BEV grid."""

    floor: np.ndarray
    foreground: np.ndarray
    removed_artifact: np.ndarray
    floor_confirmed_artifact: np.ndarray
    expanded_artifact: np.ndarray
    artifact_component_id: np.ndarray
    artifact_phone_support: np.ndarray
    artifact_phone_fraction: np.ndarray
    accepted_component_count: int


@dataclass(frozen=True)
class FloorUnderlayResult:
    height: np.ndarray
    height_agl: np.ndarray
    density: np.ndarray
    observed: np.ndarray
    unknown: np.ndarray
    changed: np.ndarray
    authority: np.ndarray


@dataclass(frozen=True)
class ShadowFillResult:
    height: np.ndarray
    height_agl: np.ndarray
    density: np.ndarray
    observed: np.ndarray
    unknown: np.ndarray
    admitted: np.ndarray
    changed: np.ndarray
    unresolved: np.ndarray
    replacement_candidate: np.ndarray
    preserved_static_surface: np.ndarray
    authority: np.ndarray


def floorplan_grid_centers(
    bounds: Mapping[str, Any],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X/Z cell-center grids in canonical row-zero=max-Z order."""

    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("floorplan shape must be positive")
    try:
        min_x = float(bounds["min_x"])
        max_x = float(bounds["max_x"])
        min_z = float(bounds["min_z"])
        max_z = float(bounds["max_z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("floorplan bounds are incomplete") from exc
    if not all(np.isfinite([min_x, max_x, min_z, max_z])):
        raise ValueError("floorplan bounds must be finite")
    if max_x <= min_x or max_z <= min_z:
        raise ValueError("floorplan bounds must have positive area")

    dx = (max_x - min_x) / cols
    dz = (max_z - min_z) / rows
    x = min_x + (np.arange(cols, dtype=np.float64) + 0.5) * dx
    z = max_z - (np.arange(rows, dtype=np.float64) + 0.5) * dz
    return np.meshgrid(x, z)


def _require_same_shape(reference: np.ndarray, **arrays: np.ndarray) -> None:
    shape = np.asarray(reference).shape
    if len(shape) != 2:
        raise ValueError("floorplan arrays must be two-dimensional")
    for name, value in arrays.items():
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} shape does not match the floorplan")


def _remove_small_components(mask: np.ndarray, minimum_cells: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if minimum_cells <= 1 or not np.any(binary):
        return binary.copy()
    labels, count = ndi.label(binary)
    if count <= 0:
        return np.zeros_like(binary)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum_cells)
    if keep.size:
        keep[0] = False
    return keep[labels]


def compute_static_floor_occlusion_shadow(
    *,
    structural_height_m: np.ndarray,
    surface_observed: np.ndarray,
    raw_height_agl_m: np.ndarray,
    wall_support: np.ndarray,
    room_footprint: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    camera_height_m: float,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> OcclusionShadow:
    """Cast finite floor shadows from coherent static-camera occluders.

    Each angular ray is scanned away from the fixed camera at X=Z=0.  An
    occluder blocks floor cells only until a ray from the camera can clear its
    top, so low furniture creates a finite shadow rather than hiding the entire
    remainder of the room.
    """

    structural = np.asarray(structural_height_m, dtype=np.float32)
    surface = np.asarray(surface_observed, dtype=bool)
    raw_agl = np.asarray(raw_height_agl_m, dtype=np.float32)
    wall = np.asarray(wall_support, dtype=np.float32)
    footprint = np.asarray(room_footprint, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(
        structural,
        surface_observed=surface,
        raw_height_agl_m=raw_agl,
        wall_support=wall,
        room_footprint=footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    camera_height = float(camera_height_m)
    if not np.isfinite(camera_height) or not 0.5 <= camera_height <= 4.0:
        raise ValueError("camera height must be finite and within [0.5, 4.0] m")

    min_height = float(settings.occluder_min_height_m)
    max_height = float(settings.occluder_max_height_m)
    coherent_horizontal = (
        footprint
        & surface
        & np.isfinite(structural)
        & (structural >= min_height)
        & (structural <= max_height)
    )
    coherent_vertical = (
        footprint
        & np.isfinite(raw_agl)
        & np.isfinite(wall)
        & (wall >= float(settings.wall_support_min))
        & (raw_agl >= min_height)
        & (raw_agl <= max_height)
    )
    occluder_height = np.zeros_like(structural, dtype=np.float32)
    occluder_height[coherent_horizontal] = structural[coherent_horizontal]
    occluder_height[coherent_vertical] = np.maximum(
        occluder_height[coherent_vertical],
        raw_agl[coherent_vertical],
    )
    occluder = _remove_small_components(
        occluder_height >= min_height,
        int(settings.occluder_min_component_cells),
    )
    if np.any(occluder):
        occluder = ndi.binary_closing(
            occluder,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        ) & footprint
        dilation = max(0, int(settings.occluder_dilation_cells))
        if dilation:
            size = (2 * dilation) + 1
            spread_height = ndi.maximum_filter(
                np.where(occluder_height > 0.0, occluder_height, 0.0),
                size=size,
                mode="constant",
                cval=0.0,
            )
            occluder = (spread_height >= min_height) & footprint
            occluder_height = np.where(
                occluder,
                spread_height,
                0.0,
            ).astype(np.float32, copy=False)
        else:
            occluder_height = np.where(
                occluder,
                occluder_height,
                0.0,
            ).astype(np.float32, copy=False)

    shadow = compute_height_occlusion_shadow(
        occluder_height_m=occluder_height,
        target_height_m=np.zeros_like(occluder_height, dtype=np.float32),
        room_footprint=footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        camera_height_m=camera_height,
        settings=settings,
    )
    return OcclusionShadow(
        shadow=shadow,
        occluder=occluder,
        occluder_height_m=occluder_height,
    )


def compute_height_occlusion_shadow(
    *,
    occluder_height_m: np.ndarray,
    target_height_m: np.ndarray,
    room_footprint: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    camera_height_m: float,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> np.ndarray:
    """Return cells whose requested height lies below a nearer ray horizon."""

    occluder_height = np.asarray(occluder_height_m, dtype=np.float32)
    target_height = np.asarray(target_height_m, dtype=np.float32)
    footprint = np.asarray(room_footprint, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(
        occluder_height,
        target_height_m=target_height,
        room_footprint=footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    camera_height = float(camera_height_m)
    if not np.isfinite(camera_height) or not 0.5 <= camera_height <= 4.0:
        raise ValueError("camera height must be finite and within [0.5, 4.0] m")

    radius = np.hypot(x_grid, z_grid)
    angle = np.arctan2(x_grid, z_grid)
    angular_step = math.radians(float(settings.angular_step_deg))
    if not np.isfinite(angular_step) or angular_step <= 0.0:
        raise ValueError("angular step must be positive")
    angle_bin = np.floor((angle + math.pi) / angular_step).astype(np.int64)
    flat_footprint = footprint.ravel()
    flat_radius = radius.ravel()
    flat_bins = angle_bin.ravel()
    flat_occluder = occluder_height.ravel()
    flat_target = target_height.ravel()
    flat_shadow = np.zeros(flat_footprint.shape, dtype=bool)
    clearance = max(0.0, float(settings.occluder_clearance_m))
    minimum_range = max(0.0, float(settings.minimum_camera_range_m))
    min_height = float(settings.occluder_min_height_m)

    for bin_id in np.unique(flat_bins[flat_footprint]):
        indices = np.flatnonzero(flat_footprint & (flat_bins == bin_id))
        if indices.size <= 0:
            continue
        ordered = indices[np.argsort(flat_radius[indices], kind="stable")]
        horizon_slope = -math.inf
        for index in ordered:
            cell_range = float(flat_radius[index])
            requested_height = float(flat_target[index])
            if not np.isfinite(requested_height):
                requested_height = 0.0
            if cell_range >= minimum_range:
                target_slope = (requested_height - camera_height) / max(
                    cell_range,
                    1e-6,
                )
                if horizon_slope > target_slope:
                    flat_shadow[index] = True

            obstacle_height = float(flat_occluder[index])
            if obstacle_height < min_height or cell_range < minimum_range:
                continue
            effective_height = max(0.0, obstacle_height - clearance)
            obstacle_slope = (effective_height - camera_height) / max(
                cell_range,
                1e-6,
            )
            horizon_slope = max(horizon_slope, obstacle_slope)

    return flat_shadow.reshape(occluder_height.shape) & footprint


def _regular_grid_resolution_m(
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
) -> tuple[float, float]:
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    if x_grid.shape != z_grid.shape or x_grid.ndim != 2:
        raise ValueError("floorplan coordinate grids must be matching 2D arrays")
    if x_grid.shape[1] < 2 or z_grid.shape[0] < 2:
        raise ValueError("floorplan coordinate grids are too small")
    dx_values = np.diff(x_grid, axis=1)
    dz_values = np.diff(z_grid, axis=0)
    dx = float(np.median(dx_values))
    dz = float(np.median(dz_values))
    if not np.isfinite(dx) or not np.isfinite(dz) or dx <= 0.0 or dz >= 0.0:
        raise ValueError(
            "floorplan grid must use increasing X columns and decreasing Z rows"
        )
    if not (
        np.allclose(dx_values, dx, rtol=1e-4, atol=1e-8)
        and np.allclose(dz_values, dz, rtol=1e-4, atol=1e-8)
    ):
        raise ValueError("floorplan coordinate grids must be regular")
    return dx, abs(dz)


def compute_radial_occlusion_artifact(
    *,
    raw_height_agl_m: np.ndarray,
    structural_height_m: np.ndarray,
    density: np.ndarray,
    observed: np.ndarray,
    wall_support: np.ndarray,
    room_footprint: np.ndarray,
    floor_shadow: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> RadialOcclusionArtifact:
    """Find sparse comb artifacts that run along fixed-camera sight lines.

    A floor shadow is managed separately from the top surface currently stored
    in that X/Z column.  This detector marks a stored top surface replaceable
    only when it also has the characteristic alternating, camera-radial shape
    of an occlusion edge.  Ordinary furniture inside a floor shadow therefore
    remains static-camera authority.
    """

    height = np.asarray(raw_height_agl_m, dtype=np.float32)
    structural = np.asarray(structural_height_m, dtype=np.float32)
    density_values = np.asarray(density, dtype=np.float32)
    observed_mask = np.asarray(observed, dtype=bool)
    wall = np.asarray(wall_support, dtype=np.float32)
    footprint = np.asarray(room_footprint, dtype=bool)
    shadow = np.asarray(floor_shadow, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(
        height,
        structural_height_m=structural,
        density=density_values,
        observed=observed_mask,
        wall_support=wall,
        room_footprint=footprint,
        floor_shadow=shadow,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    dx, dz = _regular_grid_resolution_m(x_grid, z_grid)
    cell_size = math.sqrt(dx * dz)

    valid = footprint & observed_mask & np.isfinite(height)
    sigma_cells = max(
        0.25,
        float(settings.artifact_smoothing_sigma_m) / cell_size,
    )
    weighted_height = ndi.gaussian_filter(
        np.where(valid, height, 0.0),
        sigma=sigma_cells,
        mode="nearest",
    )
    smoothed_weight = ndi.gaussian_filter(
        valid.astype(np.float32),
        sigma=sigma_cells,
        mode="nearest",
    )
    smoothed_height = np.divide(
        weighted_height,
        smoothed_weight,
        out=np.zeros_like(weighted_height, dtype=np.float32),
        where=smoothed_weight > 0.05,
    )

    x_axis = x_grid[0]
    z_axis = z_grid[:, 0]
    gradient_z, gradient_x = np.gradient(
        smoothed_height,
        z_axis,
        x_axis,
        edge_order=1,
    )
    gradient_zz, gradient_zx = np.gradient(
        gradient_z,
        z_axis,
        x_axis,
        edge_order=1,
    )
    gradient_xz, gradient_xx = np.gradient(
        gradient_x,
        z_axis,
        x_axis,
        edge_order=1,
    )

    camera_range = np.hypot(x_grid, z_grid)
    radial_x = np.divide(
        x_grid,
        camera_range,
        out=np.zeros_like(x_grid),
        where=camera_range > 1e-6,
    )
    radial_z = np.divide(
        z_grid,
        camera_range,
        out=np.zeros_like(z_grid),
        where=camera_range > 1e-6,
    )
    radial_derivative = (radial_x * gradient_x) + (radial_z * gradient_z)
    radial_second_derivative = (
        (radial_x * radial_x * gradient_xx)
        + (radial_x * radial_z * (gradient_xz + gradient_zx))
        + (radial_z * radial_z * gradient_zz)
    )
    window_cells = max(
        3,
        int(round(float(settings.artifact_window_m) / cell_size)),
    )
    if window_cells % 2 == 0:
        window_cells += 1
    radial_energy = ndi.uniform_filter(
        np.abs(radial_second_derivative),
        size=window_cells,
        mode="nearest",
    )
    gradient_magnitude = np.hypot(gradient_x, gradient_z)
    radial_alignment = np.divide(
        np.abs(radial_derivative),
        gradient_magnitude,
        out=np.zeros_like(gradient_magnitude),
        where=gradient_magnitude > 1e-6,
    )

    row_coordinates, column_coordinates = np.indices(height.shape, dtype=np.float64)
    sample_step_m = min(dx, dz)
    outward_rows = row_coordinates + (radial_z * sample_step_m / (z_axis[1] - z_axis[0]))
    outward_columns = column_coordinates + (radial_x * sample_step_m / dx)
    inward_rows = row_coordinates - (radial_z * sample_step_m / (z_axis[1] - z_axis[0]))
    inward_columns = column_coordinates - (radial_x * sample_step_m / dx)
    outward_derivative = ndi.map_coordinates(
        radial_derivative,
        [outward_rows, outward_columns],
        order=1,
        mode="nearest",
    )
    inward_derivative = ndi.map_coordinates(
        radial_derivative,
        [inward_rows, inward_columns],
        order=1,
        mode="nearest",
    )
    oscillation_fraction = ndi.uniform_filter(
        ((outward_derivative * inward_derivative) < 0.0).astype(np.float32),
        size=window_cells,
        mode="nearest",
    )

    energy_population = radial_energy[valid & np.isfinite(radial_energy)]
    percentile = float(settings.artifact_energy_percentile)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("artifact energy percentile must be within [0, 100]")
    energy_threshold = (
        float(np.percentile(energy_population, percentile))
        if energy_population.size
        else math.inf
    )
    height_residual = height - structural
    seed = (
        valid
        & shadow
        & np.isfinite(structural)
        & np.isfinite(density_values)
        & np.isfinite(wall)
        & (wall < float(settings.wall_support_min))
        & (height_residual >= float(settings.artifact_min_height_residual_m))
        & (density_values <= float(settings.artifact_max_density))
        & (radial_energy >= energy_threshold)
        & (radial_alignment >= float(settings.artifact_min_radial_alignment))
        & (
            oscillation_fraction
            >= float(settings.artifact_min_oscillation_fraction)
        )
    )
    seed = _remove_small_components(
        seed,
        int(settings.artifact_min_component_cells),
    )
    dilation_cells = max(
        0,
        int(round(float(settings.artifact_dilation_m) / cell_size)),
    )
    region = (
        ndi.binary_dilation(seed, iterations=dilation_cells)
        if dilation_cells > 0
        else seed.copy()
    )
    region &= (
        valid
        & shadow
        & np.isfinite(structural)
        & np.isfinite(density_values)
        & np.isfinite(wall)
        & (wall < float(settings.wall_support_min))
        & (
            height_residual
            >= float(settings.artifact_region_min_height_residual_m)
        )
        & (density_values <= (float(settings.artifact_max_density) * 1.5))
    )

    return RadialOcclusionArtifact(
        seed=seed,
        region=region,
        radial_curvature_energy=radial_energy.astype(np.float32, copy=False),
        radial_alignment=radial_alignment.astype(np.float32, copy=False),
        oscillation_fraction=oscillation_fraction.astype(np.float32, copy=False),
        energy_threshold=energy_threshold,
    )


def build_floor_underlay_masks(
    *,
    room_footprint: np.ndarray,
    foreground_candidate: np.ndarray,
    floor_shadow: np.ndarray,
    radial_artifact_region: np.ndarray,
    phone_floor_visibility: np.ndarray,
    density: np.ndarray,
    wall_support: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> FloorUnderlayMasks:
    """Build a complete floor underlay without painting through furniture.

    Phone observations classify radial artifact components; they are not used
    as a sparse paint layer.  A component is removable only when enough of its
    cells have direct multi-view floor visibility.  Cleanup then expands a
    bounded metric distance through sparse, non-wall shadow cells.  Everything
    left in ``foreground_candidate`` remains exact static-camera authority.
    """

    footprint = np.asarray(room_footprint, dtype=bool)
    candidate = np.asarray(foreground_candidate, dtype=bool)
    shadow = np.asarray(floor_shadow, dtype=bool)
    artifact = np.asarray(radial_artifact_region, dtype=bool)
    phone_floor = np.asarray(phone_floor_visibility, dtype=bool)
    density_values = np.asarray(density, dtype=np.float32)
    wall = np.asarray(wall_support, dtype=np.float32)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(
        footprint,
        foreground_candidate=candidate,
        floor_shadow=shadow,
        radial_artifact_region=artifact,
        phone_floor_visibility=phone_floor,
        density=density_values,
        wall_support=wall,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    if np.any(candidate & ~footprint):
        raise ValueError("foreground candidate must be inside the room footprint")
    dx, dz = _regular_grid_resolution_m(x_grid, z_grid)

    minimum_phone_cells = int(settings.floor_artifact_min_phone_cells)
    minimum_foreground_cells = int(settings.floor_artifact_min_foreground_cells)
    minimum_phone_fraction = float(settings.floor_artifact_min_phone_fraction)
    expansion_m = float(settings.floor_artifact_expansion_m)
    maximum_density = float(settings.floor_artifact_max_density)
    maximum_island_cells = int(settings.floor_artifact_max_island_cells)
    if minimum_phone_cells < 1 or minimum_foreground_cells < 1:
        raise ValueError("floor-artifact support thresholds must be positive")
    if not 0.0 <= minimum_phone_fraction <= 1.0:
        raise ValueError("floor-artifact phone fraction must be within [0, 1]")
    if not np.isfinite(expansion_m) or expansion_m < 0.0:
        raise ValueError("floor-artifact expansion must be finite and non-negative")
    if not np.isfinite(maximum_density) or maximum_density < 0.0:
        raise ValueError("floor-artifact density gate must be finite and non-negative")
    if maximum_island_cells < 0:
        raise ValueError("floor-artifact island limit must be non-negative")

    labels, component_count = ndi.label(
        artifact & footprint,
        structure=np.ones((3, 3), dtype=bool),
    )
    component_support = np.zeros(footprint.shape, dtype=np.uint32)
    component_fraction = np.zeros(footprint.shape, dtype=np.float32)
    floor_confirmed = np.zeros(footprint.shape, dtype=bool)
    accepted_component_count = 0
    for component in range(1, component_count + 1):
        component_mask = labels == component
        component_cells = int(np.count_nonzero(component_mask))
        phone_cells = int(np.count_nonzero(component_mask & phone_floor))
        foreground_cells = int(np.count_nonzero(component_mask & candidate))
        phone_fraction = phone_cells / max(1, component_cells)
        component_support[component_mask] = phone_cells
        component_fraction[component_mask] = phone_fraction
        if (
            phone_cells >= minimum_phone_cells
            and foreground_cells >= minimum_foreground_cells
            and phone_fraction >= minimum_phone_fraction
        ):
            floor_confirmed |= component_mask
            accepted_component_count += 1

    if np.any(floor_confirmed):
        artifact_distance_m = ndi.distance_transform_edt(
            ~floor_confirmed,
            sampling=(dz, dx),
        )
        expanded_artifact = artifact_distance_m <= (expansion_m + 1e-9)
    else:
        expanded_artifact = np.zeros(footprint.shape, dtype=bool)
    expanded_artifact &= footprint

    sparse_non_wall = (
        np.isfinite(density_values)
        & np.isfinite(wall)
        & (density_values <= maximum_density)
        & (wall < float(settings.wall_support_min))
    )
    removed = candidate & shadow & expanded_artifact & sparse_non_wall

    # A bounded expansion can sever one- or two-cell artifact remnants from the
    # foreground.  Remove only tiny, floor-corroborated islands inside the same
    # accepted envelope; never erode a connected couch or wall boundary.
    if maximum_island_cells > 0:
        provisional_foreground = candidate & ~removed
        island_labels, island_count = ndi.label(
            provisional_foreground,
            structure=np.ones((3, 3), dtype=bool),
        )
        for component in range(1, island_count + 1):
            island = island_labels == component
            island_cells = int(np.count_nonzero(island))
            if island_cells > maximum_island_cells:
                continue
            phone_fraction = float(np.count_nonzero(island & phone_floor)) / max(
                1,
                island_cells,
            )
            if (
                np.all(~island | expanded_artifact)
                and np.all(~island | sparse_non_wall)
                and phone_fraction >= minimum_phone_fraction
            ):
                removed |= island

    foreground = candidate & ~removed
    floor = footprint & ~foreground
    if np.any(floor & foreground):
        raise AssertionError("floor underlay overlaps trusted foreground")
    if not np.array_equal(floor | foreground, footprint):
        raise AssertionError("floor and foreground do not cover the room footprint")

    return FloorUnderlayMasks(
        floor=floor,
        foreground=foreground,
        removed_artifact=removed,
        floor_confirmed_artifact=floor_confirmed,
        expanded_artifact=expanded_artifact,
        artifact_component_id=labels.astype(np.int32, copy=False),
        artifact_phone_support=component_support,
        artifact_phone_fraction=component_fraction,
        accepted_component_count=accepted_component_count,
    )


def apply_floor_underlay(
    *,
    static_height: np.ndarray,
    static_height_agl: np.ndarray,
    static_density: np.ndarray,
    static_observed: np.ndarray,
    room_footprint: np.ndarray,
    masks: FloorUnderlayMasks,
    floor_height: float = 0.0,
    floor_height_agl: float = 0.0,
    floor_density: float = 0.10,
) -> FloorUnderlayResult:
    """Composite one complete floor plane beneath exact static foreground."""

    height = np.asarray(static_height, dtype=np.float32)
    height_agl = np.asarray(static_height_agl, dtype=np.float32)
    density = np.asarray(static_density, dtype=np.float32)
    observed = np.asarray(static_observed, dtype=bool)
    footprint = np.asarray(room_footprint, dtype=bool)
    floor = np.asarray(masks.floor, dtype=bool)
    foreground = np.asarray(masks.foreground, dtype=bool)
    _require_same_shape(
        height,
        static_height_agl=height_agl,
        static_density=density,
        static_observed=observed,
        room_footprint=footprint,
        floor=floor,
        foreground=foreground,
    )
    floor_height_value = float(floor_height)
    floor_agl_value = float(floor_height_agl)
    floor_density_value = float(floor_density)
    if not all(
        np.isfinite([floor_height_value, floor_agl_value, floor_density_value])
    ):
        raise ValueError("floor values must be finite")
    if not 0.0 < floor_density_value <= 1.0:
        raise ValueError("floor density must be within (0, 1]")
    if np.any(floor & foreground) or not np.array_equal(
        floor | foreground,
        footprint,
    ):
        raise ValueError("floor and foreground must be disjoint and cover the footprint")

    completed_height = height.copy()
    completed_agl = height_agl.copy()
    completed_density = density.copy()
    completed_observed = observed.copy()
    completed_height[floor] = floor_height_value
    completed_agl[floor] = floor_agl_value
    completed_density[floor] = floor_density_value
    completed_observed[floor] = True
    changed = floor & (
        ~np.isfinite(height)
        | ~np.isfinite(height_agl)
        | ~np.isfinite(density)
        | (height != floor_height_value)
        | (height_agl != floor_agl_value)
        | (density != floor_density_value)
        | ~observed
    )
    authority = np.zeros(height.shape, dtype=np.uint8)
    authority[foreground] = 1
    authority[floor] = 3

    outside_floor = ~floor
    if not (
        np.array_equal(
            completed_height[outside_floor],
            height[outside_floor],
            equal_nan=True,
        )
        and np.array_equal(
            completed_agl[outside_floor],
            height_agl[outside_floor],
            equal_nan=True,
        )
        and np.array_equal(
            completed_density[outside_floor],
            density[outside_floor],
            equal_nan=True,
        )
        and np.array_equal(completed_observed[outside_floor], observed[outside_floor])
    ):
        raise AssertionError("floor compositing changed trusted foreground or outside cells")

    return FloorUnderlayResult(
        height=completed_height,
        height_agl=completed_agl,
        density=completed_density,
        observed=completed_observed,
        unknown=~completed_observed,
        changed=changed,
        authority=authority,
    )


def scene_fusion_points_to_static_floorplan(
    *,
    points_world_m: np.ndarray,
    reference_camera_to_world: np.ndarray,
    floor_y_m: float,
    scene_to_static_transform: np.ndarray,
) -> np.ndarray:
    """Convert backend-world fusion points to the stored static floorplan frame."""

    points = np.asarray(points_world_m, dtype=np.float64)
    pose = np.asarray(reference_camera_to_world, dtype=np.float64)
    transform = np.asarray(scene_to_static_transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world_m must be an Nx3 array")
    if pose.shape != (4, 4) or transform.shape != (4, 4):
        raise ValueError("camera pose and scene transform must be 4x4")
    if not (
        np.isfinite(points).all()
        and np.isfinite(pose).all()
        and np.isfinite(transform).all()
        and np.isfinite(float(floor_y_m))
    ):
        raise ValueError("scene-fusion transforms and points must be finite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("scene-to-static transform has an invalid homogeneous row")

    camera_world = pose[:3, 3]
    forward = pose[:3, 2].copy()
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("reference camera has no ground-plane forward direction")
    forward /= forward_norm
    right = np.cross(np.asarray([0.0, 1.0, 0.0]), forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise ValueError("reference camera has no ground-plane right direction")
    right /= right_norm
    if float(np.dot(right, pose[:3, 0])) < 0.0:
        right *= -1.0

    delta = points - camera_world
    scene_local = np.column_stack(
        (
            delta @ right,
            points[:, 1] - float(floor_y_m),
            delta @ forward,
            np.ones(points.shape[0], dtype=np.float64),
        )
    )
    return (scene_local @ transform.T)[:, :3]


def phone_view_points_to_static_floorplan(
    *,
    points_phone_world_m: np.ndarray,
    source_camera_to_world: np.ndarray,
    admitted_camera_to_backend_world: np.ndarray,
    phone_to_fixed_refinement: np.ndarray,
    reference_camera_to_world: np.ndarray,
    floor_y_m: float,
    scene_to_static_transform: np.ndarray,
) -> np.ndarray:
    """Project original phone-only MA points through the admitted pose bridge."""

    points = np.asarray(points_phone_world_m, dtype=np.float64)
    source_pose = np.asarray(source_camera_to_world, dtype=np.float64)
    admitted_pose = np.asarray(admitted_camera_to_backend_world, dtype=np.float64)
    refinement = np.asarray(phone_to_fixed_refinement, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_phone_world_m must be an Nx3 array")
    for name, transform in (
        ("source camera pose", source_pose),
        ("admitted camera pose", admitted_pose),
        ("phone refinement", refinement),
    ):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{name} must be a finite 4x4 transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"{name} has an invalid homogeneous row")
    if not np.isfinite(points).all():
        raise ValueError("phone view points must be finite")

    # Original phone outputs store points in the phone-only MA world frame.
    # Recover camera-local geometry, then use the quality-gated pose retained
    # for that exact video frame by the joint inference.
    camera_points = (points - source_pose[:3, 3]) @ source_pose[:3, :3]
    backend_points = (
        camera_points @ admitted_pose[:3, :3].T
    ) + admitted_pose[:3, 3]
    refined_points = (
        backend_points @ refinement[:3, :3].T
    ) + refinement[:3, 3]
    return scene_fusion_points_to_static_floorplan(
        points_world_m=refined_points,
        reference_camera_to_world=reference_camera_to_world,
        floor_y_m=floor_y_m,
        scene_to_static_transform=scene_to_static_transform,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative.size <= 0 or cumulative[-1] <= 0.0:
        return float(np.median(values))
    index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def build_phone_surface(
    *,
    points_static_m: np.ndarray,
    confidence: np.ndarray,
    provenance: np.ndarray,
    view_support: np.ndarray,
    room_footprint: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> PhoneSurface:
    """Prepare a locally coherent phone surface from preserved fusion voxels.

    Phone-only voxels need two phone views.  Fixed/phone agreement voxels need
    three total views, which guarantees at least two phone views because the
    fixed anchor contributes at most one distinct view id.
    """

    points = np.asarray(points_static_m, dtype=np.float64)
    confidence_values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    provenance_values = np.asarray(provenance, dtype=np.uint8).reshape(-1)
    view_values = np.asarray(view_support, dtype=np.int64).reshape(-1)
    footprint = np.asarray(room_footprint, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(
        footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_static_m must be an Nx3 array")
    count = points.shape[0]
    if not (
        confidence_values.size == count
        and provenance_values.size == count
        and view_values.size == count
    ):
        raise ValueError("phone voxel arrays have incompatible lengths")

    has_phone = (provenance_values & np.uint8(2)) != 0
    has_fixed = (provenance_values & np.uint8(1)) != 0
    enough_views = np.where(has_fixed, view_values >= 3, view_values >= 2)
    eligible = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence_values)
        & has_phone
        & enough_views
        & (confidence_values >= float(settings.phone_min_voxel_confidence))
        & (points[:, 1] >= -0.08)
        & (points[:, 1] <= float(settings.phone_max_height_m))
    )
    selected_points = points[eligible]
    selected_confidence = confidence_values[eligible]
    selected_views = view_values[eligible]
    shape = footprint.shape
    height = np.full(shape, np.nan, dtype=np.float32)
    quality = np.zeros(shape, dtype=np.float32)
    support = np.zeros(shape, dtype=np.uint16)
    spread = np.full(shape, np.nan, dtype=np.float32)
    good = np.zeros(shape, dtype=bool)
    if selected_points.shape[0] <= 0:
        return PhoneSurface(
            height_m=height,
            quality=quality,
            support=support,
            spread_m=spread,
            good=good,
            eligible_voxel_count=0,
        )

    radius = float(settings.phone_surface_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("phone surface radius must be positive")
    height_bin = float(settings.phone_height_bin_m)
    if not np.isfinite(height_bin) or height_bin <= 0.0:
        raise ValueError("phone height bin must be positive")
    tree = cKDTree(selected_points[:, [0, 2]])
    spatial_sigma = max(radius * (2.0 / 3.0), 1e-6)

    for row, col in zip(*np.where(footprint)):
        center = np.asarray([x_grid[row, col], z_grid[row, col]])
        neighbours = tree.query_ball_point(center, r=radius)
        if len(neighbours) < int(settings.phone_min_cluster_points):
            continue
        indices = np.asarray(neighbours, dtype=np.int64)
        candidate_points = selected_points[indices]
        candidate_height = candidate_points[:, 1]
        distance_sq = np.sum(
            (candidate_points[:, [0, 2]] - center[None, :]) ** 2,
            axis=1,
        )
        spatial_weight = np.exp(-0.5 * distance_sq / (spatial_sigma ** 2))
        weights = (
            selected_confidence[indices]
            * np.clip(selected_views[indices], 1, 4)
            * spatial_weight
        )
        height_bins = np.floor(
            (candidate_height + (height_bin * 0.5)) / height_bin
        ).astype(np.int64)
        unique_bins = np.unique(height_bins)
        bin_weight = np.asarray(
            [float(np.sum(weights[height_bins == item])) for item in unique_bins]
        )
        strongest_weight = float(np.max(bin_weight))
        credible_bins: list[int] = []
        for item, item_weight in zip(unique_bins, bin_weight):
            item_height = candidate_height[height_bins == item]
            if item_height.size < int(settings.phone_min_cluster_points):
                continue
            if item_height.size > 2:
                item_spread = float(
                    np.percentile(item_height, 90.0)
                    - np.percentile(item_height, 10.0)
                )
            else:
                item_spread = float(np.ptp(item_height))
            if item_spread > float(settings.phone_max_surface_spread_m):
                continue
            if item_weight < (
                strongest_weight
                * float(settings.phone_upper_mode_min_weight_ratio)
            ):
                continue
            credible_bins.append(int(item))
        if not credible_bins:
            continue
        # A top-down elevation map represents the highest coherent surface in
        # an X/Z column.  This prevents a well-sampled floor mode from erasing
        # a couch/table top that the phone also observed reliably.
        selected_bin = max(credible_bins)
        in_cluster = height_bins == selected_bin
        cluster_count = int(np.count_nonzero(in_cluster))
        if cluster_count < int(settings.phone_min_cluster_points):
            continue

        cluster_height = candidate_height[in_cluster]
        cluster_weight = weights[in_cluster]
        cluster_confidence = selected_confidence[indices][in_cluster]
        cluster_distance = np.sqrt(distance_sq[in_cluster])
        surface_height = _weighted_median(cluster_height, cluster_weight)
        if cluster_height.size > 2:
            surface_spread = float(
                np.percentile(cluster_height, 90.0)
                - np.percentile(cluster_height, 10.0)
            )
        else:
            surface_spread = float(np.ptp(cluster_height))
        if not np.isfinite(surface_spread):
            continue

        support_quality = min(1.0, cluster_count / 4.0)
        confidence_quality = float(
            np.clip(np.median(cluster_confidence) / 0.25, 0.0, 1.0)
        )
        distance_quality = float(
            math.exp(-((float(np.median(cluster_distance)) / radius) ** 2))
        )
        dispersion_quality = float(
            np.clip(
                1.0
                - (
                    surface_spread
                    / max(float(settings.phone_max_surface_spread_m), 1e-6)
                ),
                0.0,
                1.0,
            )
        )
        surface_quality = (
            (0.35 * support_quality)
            + (0.35 * confidence_quality)
            + (0.30 * distance_quality)
        ) * (0.5 + (0.5 * dispersion_quality))

        height[row, col] = max(0.0, float(surface_height))
        quality[row, col] = float(np.clip(surface_quality, 0.0, 1.0))
        support[row, col] = np.uint16(min(cluster_count, 65535))
        spread[row, col] = surface_spread
        good[row, col] = (
            surface_spread <= float(settings.phone_max_surface_spread_m)
            and surface_quality >= float(settings.phone_min_quality)
        )

    return PhoneSurface(
        height_m=height,
        quality=quality,
        support=support,
        spread_m=spread,
        good=good,
        eligible_voxel_count=int(np.count_nonzero(eligible)),
    )


def build_phone_surface_from_views(
    *,
    points_static_m: np.ndarray,
    view_indices: np.ndarray,
    confidence: np.ndarray,
    room_footprint: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> PhoneSurface:
    """Build a 4 cm top surface from original phone-view observations.

    Dense pixels from one image cannot manufacture support: observations are
    first collapsed by view, floorplan cell, and height mode.  A surface mode
    is admitted only when at least two distinct phone views observed it.
    """

    points = np.asarray(points_static_m, dtype=np.float64)
    views = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    confidence_values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    footprint = np.asarray(room_footprint, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(footprint, x_grid_m=x_grid, z_grid_m=z_grid)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_static_m must be an Nx3 array")
    if views.size != points.shape[0] or confidence_values.size != points.shape[0]:
        raise ValueError("phone view arrays have incompatible lengths")
    if np.any(views < 0):
        raise ValueError("phone view indices must be non-negative")

    dx, dz = _regular_grid_resolution_m(x_grid, z_grid)
    rows, columns = footprint.shape
    min_x = float(x_grid[0, 0] - (dx * 0.5))
    max_x = float(x_grid[0, -1] + (dx * 0.5))
    max_z = float(z_grid[0, 0] + (dz * 0.5))
    min_z = float(z_grid[-1, 0] - (dz * 0.5))
    height_bin = float(settings.phone_height_bin_m)
    if not np.isfinite(height_bin) or height_bin <= 0.0:
        raise ValueError("phone height bin must be positive")
    height_bin_count = max(
        2,
        int(math.ceil((float(settings.phone_max_height_m) + 0.12) / height_bin)),
    )

    eligible = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence_values)
        & (confidence_values >= float(settings.phone_min_voxel_confidence))
        & (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 2] >= min_z)
        & (points[:, 2] < max_z)
        & (points[:, 1] >= -0.08)
        & (points[:, 1] <= float(settings.phone_max_height_m))
    )
    selected = points[eligible]
    selected_views = views[eligible]
    selected_confidence = confidence_values[eligible]
    shape = footprint.shape
    height = np.full(shape, np.nan, dtype=np.float32)
    quality = np.zeros(shape, dtype=np.float32)
    support = np.zeros(shape, dtype=np.uint16)
    spread = np.full(shape, np.nan, dtype=np.float32)
    good = np.zeros(shape, dtype=bool)
    if selected.shape[0] <= 0:
        return PhoneSurface(
            height_m=height,
            quality=quality,
            support=support,
            spread_m=spread,
            good=good,
            eligible_voxel_count=0,
        )

    columns_index = np.floor((selected[:, 0] - min_x) / dx).astype(np.int64)
    rows_index = np.floor((max_z - selected[:, 2]) / dz).astype(np.int64)
    inside = (
        (rows_index >= 0)
        & (rows_index < rows)
        & (columns_index >= 0)
        & (columns_index < columns)
    )
    selected = selected[inside]
    selected_views = selected_views[inside]
    selected_confidence = selected_confidence[inside]
    rows_index = rows_index[inside]
    columns_index = columns_index[inside]
    cell_index = (rows_index * columns) + columns_index
    in_footprint = footprint.ravel()[cell_index]
    selected = selected[in_footprint]
    selected_views = selected_views[in_footprint]
    selected_confidence = selected_confidence[in_footprint]
    cell_index = cell_index[in_footprint]
    height_index = np.clip(
        np.floor((selected[:, 1] + (height_bin * 0.5)) / height_bin).astype(
            np.int64
        ),
        0,
        height_bin_count - 1,
    )

    # Collapse all pixels from a single image before counting view support.
    view_cell_height_key = (
        ((selected_views * (rows * columns)) + cell_index) * height_bin_count
    ) + height_index
    unique_view_key, inverse = np.unique(
        view_cell_height_key,
        return_inverse=True,
    )
    weights = np.maximum(selected_confidence, 1e-6)
    weight_sum = np.bincount(inverse, weights=weights)
    weighted_height_sum = np.bincount(
        inverse,
        weights=selected[:, 1] * weights,
    )
    per_view_height = np.divide(
        weighted_height_sum,
        weight_sum,
        out=np.zeros_like(weighted_height_sum),
        where=weight_sum > 0.0,
    )
    cell_height_key = unique_view_key % ((rows * columns) * height_bin_count)
    order = np.argsort(cell_height_key, kind="stable")
    cell_height_key = cell_height_key[order]
    per_view_height = per_view_height[order]
    mode_key, mode_start, mode_support = np.unique(
        cell_height_key,
        return_index=True,
        return_counts=True,
    )
    mode_height = np.add.reduceat(per_view_height, mode_start) / mode_support
    mode_min = np.minimum.reduceat(per_view_height, mode_start)
    mode_max = np.maximum.reduceat(per_view_height, mode_start)
    mode_spread = mode_max - mode_min
    mode_cell = mode_key // height_bin_count
    mode_height_index = mode_key % height_bin_count

    unique_cells, cell_start, cell_mode_count = np.unique(
        mode_cell,
        return_index=True,
        return_counts=True,
    )
    minimum_views = 2
    for cell, start, count in zip(unique_cells, cell_start, cell_mode_count):
        mode_slice = slice(int(start), int(start + count))
        mode_views = mode_support[mode_slice]
        mode_dispersion = mode_spread[mode_slice]
        eligible_mode = (
            (mode_views >= minimum_views)
            & (
                mode_dispersion
                <= float(settings.phone_max_surface_spread_m)
            )
        )
        if not np.any(eligible_mode):
            continue
        strongest_support = int(np.max(mode_views[eligible_mode]))
        credible = eligible_mode & (
            mode_views
            >= (
                strongest_support
                * float(settings.phone_upper_mode_min_weight_ratio)
            )
        )
        credible_indices = np.flatnonzero(credible)
        selected_mode = int(
            credible_indices[
                np.argmax(mode_height_index[mode_slice][credible_indices])
            ]
        )
        flat_cell = int(cell)
        row, column = divmod(flat_cell, columns)
        selected_support = int(mode_views[selected_mode])
        selected_spread = float(mode_dispersion[selected_mode])
        support_quality = min(1.0, selected_support / 4.0)
        dispersion_quality = float(
            np.clip(
                1.0
                - (
                    selected_spread
                    / max(float(settings.phone_max_surface_spread_m), 1e-6)
                ),
                0.0,
                1.0,
            )
        )
        surface_quality = (0.70 * support_quality) + (0.30 * dispersion_quality)
        height[row, column] = max(
            0.0,
            float(mode_height[mode_slice][selected_mode]),
        )
        quality[row, column] = float(np.clip(surface_quality, 0.0, 1.0))
        support[row, column] = np.uint16(min(selected_support, 65535))
        spread[row, column] = selected_spread
        good[row, column] = surface_quality >= float(settings.phone_min_quality)

    return PhoneSurface(
        height_m=height,
        quality=quality,
        support=support,
        spread_m=spread,
        good=good,
        eligible_voxel_count=int(unique_view_key.size),
    )


def build_phone_floor_visibility_from_views(
    *,
    points_static_m: np.ndarray,
    view_indices: np.ndarray,
    confidence: np.ndarray,
    room_footprint: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> PhoneFloorVisibility:
    """Rasterize the aligned floor seen directly by distinct phone views.

    This intentionally does not choose an upper height mode and does not
    dilate, interpolate, or grow the result.  Every admitted cell contains
    floor-height observations from the configured number of distinct views.
    """

    points = np.asarray(points_static_m, dtype=np.float64)
    views = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    confidence_values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    footprint = np.asarray(room_footprint, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(footprint, x_grid_m=x_grid, z_grid_m=z_grid)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_static_m must be an Nx3 array")
    if views.size != points.shape[0] or confidence_values.size != points.shape[0]:
        raise ValueError("phone view arrays have incompatible lengths")
    if np.any(views < 0):
        raise ValueError("phone view indices must be non-negative")

    dx, dz = _regular_grid_resolution_m(x_grid, z_grid)
    rows, columns = footprint.shape
    min_x = float(x_grid[0, 0] - (dx * 0.5))
    max_x = float(x_grid[0, -1] + (dx * 0.5))
    max_z = float(z_grid[0, 0] + (dz * 0.5))
    min_z = float(z_grid[-1, 0] - (dz * 0.5))
    floor_max = float(settings.phone_floor_visibility_max_height_m)
    minimum_views = int(settings.phone_floor_visibility_min_views)
    if floor_max < 0.0:
        raise ValueError("phone floor-visibility height must be non-negative")
    if minimum_views <= 0:
        raise ValueError("phone floor visibility requires at least one view")

    eligible = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence_values)
        & (confidence_values >= float(settings.phone_min_voxel_confidence))
        & (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 2] >= min_z)
        & (points[:, 2] < max_z)
        & (points[:, 1] >= -0.08)
        & (points[:, 1] <= floor_max)
    )
    selected = points[eligible]
    selected_views = views[eligible]
    support = np.zeros(footprint.shape, dtype=np.uint16)
    quality = np.zeros(footprint.shape, dtype=np.float32)
    if selected.shape[0] <= 0:
        return PhoneFloorVisibility(
            visible=np.zeros_like(footprint),
            view_support=support,
            quality=quality,
            floor_point_count=0,
            unique_view_cell_count=0,
        )

    column_index = np.floor((selected[:, 0] - min_x) / dx).astype(np.int64)
    row_index = np.floor((max_z - selected[:, 2]) / dz).astype(np.int64)
    inside = (
        (row_index >= 0)
        & (row_index < rows)
        & (column_index >= 0)
        & (column_index < columns)
    )
    row_index = row_index[inside]
    column_index = column_index[inside]
    selected_views = selected_views[inside]
    cell_index = (row_index * columns) + column_index
    in_footprint = footprint.ravel()[cell_index]
    cell_index = cell_index[in_footprint]
    selected_views = selected_views[in_footprint]
    if cell_index.size <= 0:
        return PhoneFloorVisibility(
            visible=np.zeros_like(footprint),
            view_support=support,
            quality=quality,
            floor_point_count=0,
            unique_view_cell_count=0,
        )

    # Each view contributes at most one vote to a cell regardless of how many
    # dense depth pixels landed there.
    view_cell_key = (selected_views * (rows * columns)) + cell_index
    unique_view_cell = np.unique(view_cell_key)
    unique_cell = unique_view_cell % (rows * columns)
    support_flat = np.bincount(
        unique_cell,
        minlength=rows * columns,
    )
    support = np.minimum(support_flat, np.iinfo(np.uint16).max).astype(
        np.uint16
    ).reshape(rows, columns)
    visible = footprint & (support >= minimum_views)
    quality[visible] = np.clip(
        support[visible].astype(np.float32) / max(4.0, float(minimum_views)),
        0.0,
        1.0,
    )
    return PhoneFloorVisibility(
        visible=visible,
        view_support=support,
        quality=quality,
        floor_point_count=int(cell_index.size),
        unique_view_cell_count=int(unique_view_cell.size),
    )


def build_region_surface_completion(
    *,
    artifact_region: np.ndarray,
    phone_surface: PhoneSurface,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> RegionSurfaceCompletion:
    """Classify and flatten whole phone-confirmed occlusion-artifact regions.

    Phone observations classify a connected region; they are not used as a
    sparse paint brush.  A floor component becomes exactly 0 m AGL.  A raised
    component is flattened only when its phone heights form one tight mode.
    """

    artifact = np.asarray(artifact_region, dtype=bool)
    phone_height = np.asarray(phone_surface.height_m, dtype=np.float32)
    phone_good = np.asarray(phone_surface.good, dtype=bool)
    phone_support = np.asarray(phone_surface.support, dtype=np.uint16)
    _require_same_shape(
        artifact,
        phone_height=phone_height,
        phone_good=phone_good,
        phone_support=phone_support,
    )
    labels, component_count = ndi.label(artifact)
    admitted = np.zeros_like(artifact)
    floor = np.zeros_like(artifact)
    flat = np.zeros_like(artifact)
    completion_height = np.full(artifact.shape, np.nan, dtype=np.float32)
    component_id = np.zeros(artifact.shape, dtype=np.int32)
    component_support = np.zeros(artifact.shape, dtype=np.uint16)
    component_coverage = np.zeros(artifact.shape, dtype=np.float32)
    component_spread = np.full(artifact.shape, np.nan, dtype=np.float32)
    floor_count = 0
    flat_count = 0

    strong_phone = (
        phone_good
        & np.isfinite(phone_height)
        & (phone_support >= int(settings.region_phone_min_views))
    )
    minimum_coverage = float(settings.region_min_phone_coverage)
    flat_minimum_coverage = float(settings.region_flat_min_phone_coverage)
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("region phone coverage must be within [0, 1]")
    if not 0.0 <= flat_minimum_coverage <= 1.0:
        raise ValueError("flat-region phone coverage must be within [0, 1]")

    for label in range(1, component_count + 1):
        component = labels == label
        cell_count = int(np.count_nonzero(component))
        if cell_count <= 0:
            continue
        supported = component & strong_phone
        support_count = int(np.count_nonzero(supported))
        coverage = support_count / cell_count
        required_support = max(
            int(settings.region_min_phone_cells),
            int(math.ceil(cell_count * minimum_coverage)),
        )
        if support_count < required_support:
            continue
        values = phone_height[supported].astype(np.float64, copy=False)
        p10, median, p90 = np.percentile(values, [10.0, 50.0, 90.0])
        spread = float(p90 - p10)
        floor_fraction = float(
            np.mean(values <= float(settings.region_floor_max_phone_height_m))
        )
        is_floor = (
            floor_fraction >= float(settings.region_floor_min_fraction)
            and spread <= float(settings.region_floor_max_spread_m)
        )
        is_flat = (
            not is_floor
            and coverage >= flat_minimum_coverage
            and spread <= float(settings.region_flat_max_spread_m)
        )
        if not is_floor and not is_flat:
            continue

        surface_height = 0.0 if is_floor else max(0.0, float(median))
        admitted[component] = True
        floor[component] = is_floor
        flat[component] = is_flat
        completion_height[component] = surface_height
        component_id[component] = label
        component_support[component] = np.uint16(min(support_count, 65535))
        component_coverage[component] = coverage
        component_spread[component] = spread
        if is_floor:
            floor_count += 1
        else:
            flat_count += 1

    return RegionSurfaceCompletion(
        admitted=admitted,
        floor=floor,
        flat=flat,
        height_m=completion_height,
        component_id=component_id,
        component_support=component_support,
        component_coverage=component_coverage,
        component_spread_m=component_spread,
        floor_component_count=floor_count,
        flat_component_count=flat_count,
    )


def build_phone_floor_region_completion(
    *,
    shadow: np.ndarray,
    phone_surface: PhoneSurface,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    settings: ShadowFillSettings = ShadowFillSettings(),
) -> PhoneFloorRegionCompletion:
    """Turn dense multi-view floor evidence into bounded floor regions.

    Direct phone samples are semantic evidence, not isolated paint cells.  A
    shadow cell belongs to the same floor region when it is within the metric
    gap limit and floor evidence is closer than any raised phone surface.  The
    raised evidence therefore supplies the furniture boundary while short
    sampling gaps inside the visible floor become one continuous region.
    """

    shadow_mask = np.asarray(shadow, dtype=bool)
    phone_height = np.asarray(phone_surface.height_m, dtype=np.float32)
    phone_quality = np.asarray(phone_surface.quality, dtype=np.float32)
    phone_support = np.asarray(phone_surface.support, dtype=np.uint16)
    phone_spread = np.asarray(phone_surface.spread_m, dtype=np.float32)
    phone_good = np.asarray(phone_surface.good, dtype=bool)
    x_grid = np.asarray(x_grid_m, dtype=np.float64)
    z_grid = np.asarray(z_grid_m, dtype=np.float64)
    _require_same_shape(
        shadow_mask,
        phone_height=phone_height,
        phone_quality=phone_quality,
        phone_support=phone_support,
        phone_spread=phone_spread,
        phone_good=phone_good,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    dx, dz = _regular_grid_resolution_m(x_grid, z_grid)
    strong_phone = (
        phone_good
        & np.isfinite(phone_height)
        & (phone_support >= int(settings.region_phone_min_views))
    )
    direct_floor = (
        shadow_mask
        & strong_phone
        & (phone_height <= float(settings.region_floor_max_phone_height_m))
    )
    raised_surface = shadow_mask & strong_phone & ~direct_floor

    empty_distance = np.full(shadow_mask.shape, np.inf, dtype=np.float32)
    empty_quality = np.zeros(shadow_mask.shape, dtype=np.float32)
    empty_support = np.zeros(shadow_mask.shape, dtype=np.uint16)
    empty_spread = np.full(shadow_mask.shape, np.nan, dtype=np.float32)
    empty_component = np.zeros(shadow_mask.shape, dtype=np.int32)
    if not np.any(direct_floor):
        return PhoneFloorRegionCompletion(
            admitted=np.zeros_like(shadow_mask),
            direct_evidence=direct_floor,
            interpolated=np.zeros_like(shadow_mask),
            nearest_floor_distance_m=empty_distance,
            quality=empty_quality,
            support=empty_support,
            spread_m=empty_spread,
            component_id=empty_component,
            component_count=0,
        )

    floor_distance, nearest = ndi.distance_transform_edt(
        ~direct_floor,
        sampling=(dz, dx),
        return_indices=True,
    )
    if np.any(raised_surface):
        raised_distance = ndi.distance_transform_edt(
            ~raised_surface,
            sampling=(dz, dx),
        )
    else:
        raised_distance = np.full(shadow_mask.shape, np.inf, dtype=np.float64)

    max_gap = float(settings.phone_floor_region_max_gap_m)
    raised_margin = float(settings.phone_floor_region_raised_margin_m)
    if max_gap < 0.0 or raised_margin < 0.0:
        raise ValueError("phone floor-region distances must be non-negative")
    region = (
        shadow_mask
        & (floor_distance <= max_gap)
        & ((floor_distance + raised_margin) < raised_distance)
    )
    region |= direct_floor

    labels, label_count = ndi.label(region)
    direct_counts = np.bincount(
        labels[direct_floor],
        minlength=label_count + 1,
    )
    keep_label = direct_counts >= int(
        settings.phone_floor_region_min_evidence_cells
    )
    keep_label[0] = False
    admitted = keep_label[labels]
    kept_labels = np.flatnonzero(keep_label)
    remap = np.zeros(label_count + 1, dtype=np.int32)
    remap[kept_labels] = np.arange(1, kept_labels.size + 1, dtype=np.int32)
    component_id = remap[labels]

    nearest_rows = nearest[0]
    nearest_columns = nearest[1]
    nearest_quality = phone_quality[nearest_rows, nearest_columns]
    nearest_support = phone_support[nearest_rows, nearest_columns]
    nearest_spread = phone_spread[nearest_rows, nearest_columns]
    distance_factor = np.clip(
        1.0 - (0.50 * floor_distance / max(max_gap, 1e-6)),
        0.50,
        1.0,
    )
    quality = np.zeros(shadow_mask.shape, dtype=np.float32)
    quality[admitted] = np.clip(
        nearest_quality[admitted] * distance_factor[admitted],
        float(settings.phone_min_quality),
        1.0,
    )
    support = np.zeros(shadow_mask.shape, dtype=np.uint16)
    support[admitted] = nearest_support[admitted]
    spread = np.full(shadow_mask.shape, np.nan, dtype=np.float32)
    spread[admitted] = nearest_spread[admitted]
    distance = np.asarray(floor_distance, dtype=np.float32)
    distance[~admitted] = np.inf

    return PhoneFloorRegionCompletion(
        admitted=admitted,
        direct_evidence=direct_floor,
        interpolated=admitted & ~direct_floor,
        nearest_floor_distance_m=distance,
        quality=quality,
        support=support,
        spread_m=spread,
        component_id=component_id,
        component_count=int(kept_labels.size),
    )


def apply_shadow_phone_authority(
    *,
    static_height: np.ndarray,
    static_height_agl: np.ndarray,
    static_density: np.ndarray,
    static_observed: np.ndarray,
    room_footprint: np.ndarray,
    shadow: np.ndarray,
    phone_surface: PhoneSurface,
    replacement_candidate: np.ndarray | None = None,
    admission_mask: np.ndarray | None = None,
) -> ShadowFillResult:
    """Use phone authority only in static-camera shadows with good coverage."""

    height = np.asarray(static_height, dtype=np.float32)
    height_agl = np.asarray(static_height_agl, dtype=np.float32)
    density = np.asarray(static_density, dtype=np.float32)
    observed = np.asarray(static_observed, dtype=bool)
    footprint = np.asarray(room_footprint, dtype=bool)
    shadow_mask = np.asarray(shadow, dtype=bool)
    candidate = (
        shadow_mask.copy()
        if replacement_candidate is None
        else np.asarray(replacement_candidate, dtype=bool)
    )
    admission = (
        np.ones_like(shadow_mask, dtype=bool)
        if admission_mask is None
        else np.asarray(admission_mask, dtype=bool)
    )
    _require_same_shape(
        height,
        static_height_agl=height_agl,
        static_density=density,
        static_observed=observed,
        room_footprint=footprint,
        shadow=shadow_mask,
        replacement_candidate=candidate,
        admission_mask=admission,
        phone_height=phone_surface.height_m,
        phone_quality=phone_surface.quality,
        phone_good=phone_surface.good,
    )

    if np.any(candidate & ~shadow_mask):
        raise ValueError("replacement candidate must be a subset of static shadow")
    admitted = (
        footprint
        & candidate
        & admission
        & np.asarray(phone_surface.good, dtype=bool)
        & np.isfinite(phone_surface.height_m)
    )
    completed_height = height.copy()
    completed_agl = height_agl.copy()
    completed_density = density.copy()
    completed_observed = observed.copy()
    completed_height[admitted] = phone_surface.height_m[admitted]
    completed_agl[admitted] = phone_surface.height_m[admitted]
    phone_density = np.clip(phone_surface.quality * 0.10, 1e-6, 1.0)
    completed_density[admitted] = np.maximum(
        completed_density[admitted],
        phone_density[admitted],
    )
    completed_observed[admitted] = True
    completed_unknown = ~completed_observed
    changed = admitted & (
        ~np.isfinite(height)
        | (np.abs(height - phone_surface.height_m) > 1e-6)
        | ~observed
    )
    unresolved = footprint & candidate & ~admitted
    preserved_static_surface = footprint & shadow_mask & ~candidate

    # 0 outside/unobserved, 1 static-visible authority, 2 unresolved candidate,
    # 3 phone authority, 4 floor-shadow column with a preserved static surface.
    authority = np.zeros(height.shape, dtype=np.uint8)
    authority[footprint & ~shadow_mask & observed] = 1
    authority[unresolved] = 2
    authority[admitted] = 3
    authority[preserved_static_surface] = 4

    outside = ~admitted
    if not (
        np.array_equal(completed_height[outside], height[outside], equal_nan=True)
        and np.array_equal(completed_agl[outside], height_agl[outside], equal_nan=True)
        and np.array_equal(completed_density[outside], density[outside], equal_nan=True)
        and np.array_equal(completed_observed[outside], observed[outside])
    ):
        raise AssertionError("shadow completion changed a cell outside phone authority")

    return ShadowFillResult(
        height=completed_height,
        height_agl=completed_agl,
        density=completed_density,
        observed=completed_observed,
        unknown=completed_unknown,
        admitted=admitted,
        changed=changed,
        unresolved=unresolved,
        replacement_candidate=candidate,
        preserved_static_surface=preserved_static_surface,
        authority=authority,
    )
