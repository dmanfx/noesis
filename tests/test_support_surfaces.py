"""Measured-plane geometry must stay bound, finite, and unambiguous."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from noesis_core.contracts.base import Matrix3
from noesis_core.contracts.support_surface import SupportSurface, SupportSurfaceReference, SupportSurfaceSet
from noesis_core.world.support_surfaces import (
    calibrated_ray_intersections,
    contains_xz,
    constrained_plane_covariance,
    elevation_at,
    intersection_covariance,
    select_unique_intersection,
)


HASH = "a" * 64
TRANSFORM = "b" * 64


def surface(*, surface_id="lower", elevation=0.2, extent=None):
    return SupportSurface(
        surface_id=surface_id,
        revision="survey-1",
        normal_world=(0.0, 1.0, 0.0),
        offset_m=-elevation,
        extent_xz=extent or ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        residual_sigma_m=0.02,
        source_observation_count=8,
        source_kind="independent_static_metric_depth",
        source_artifact_sha256=HASH,
    )


def surface_set(*, authority="measured_calibration", surfaces=None):
    return SupportSurfaceSet(
        camera_id="living-room",
        world_frame_revision="world-1",
        calibration_revision="cal-1",
        world_transform_sha256=TRANSFORM,
        evidence_sha256=HASH,
        authority=authority,
        surfaces=surfaces if surfaces is not None else (surface(),),
    )


def hits(collection, **overrides):
    args = dict(camera_id="living-room", world_frame_revision="world-1",
                calibration_revision="cal-1", world_transform_sha256=TRANSFORM,
                origin_world_m=(1.0, 2.0, 1.0), direction_world=(0.0, -1.0, 0.0),
                max_range_m=10.0)
    args.update(overrides)
    return calibrated_ray_intersections(collection, **args)


def test_sloped_elevation_and_convex_extent():
    n = (0.0, math.sqrt(0.99), -0.1)
    plane = surface().model_copy(update={"normal_world": n, "offset_m": -0.2})
    assert contains_xz(plane, 0.0, 1.0)
    assert elevation_at(plane, 1.0, 1.0) == pytest.approx(0.3 / n[1])
    assert elevation_at(plane, 3.0, 1.0) is None
    with pytest.raises(ValidationError, match="convex"):
        surface(extent=((0, 0), (2, 0), (0.5, 0.5), (2, 2), (0, 2)))
    pentagon = tuple((1.0 + math.cos(2 * math.pi * i / 5),
                      1.0 + math.sin(2 * math.pi * i / 5)) for i in range(5))
    star = tuple(pentagon[i] for i in (0, 2, 4, 1, 3))
    with pytest.raises(ValidationError, match="convex"):
        surface(extent=star)
    with pytest.raises(ValidationError, match="unit length"):
        surface().model_copy(update={"normal_world": (0, 2, 0)}).model_validate(
            surface().model_copy(update={"normal_world": (0, 2, 0)}).model_dump()
        )


def test_exact_binding_authority_and_positive_bounded_ray():
    collection = surface_set()
    found = hits(collection)
    assert len(found) == 1
    assert found[0].point_world_m == pytest.approx((1.0, 0.2, 1.0))
    assert found[0].range_m == pytest.approx(1.8)
    assert found[0].surface.surface_id == "lower"
    assert hits(collection, calibration_revision="other") == ()
    assert hits(collection, world_transform_sha256="c" * 64) == ()
    assert hits(collection, max_range_m=1.0) == ()
    assert hits(collection, direction_world=(0.0, 1.0, 0.0)) == ()
    assert hits(surface_set(authority="review_only")) == ()


def test_overlapping_levels_remain_distinct_and_ambiguous():
    collection = surface_set(surfaces=(surface(), surface(surface_id="upper", elevation=0.5)))
    found = hits(collection)
    assert [row.surface.surface_id for row in found] == ["upper", "lower"]
    assert select_unique_intersection(found) is None
    assert select_unique_intersection(hits(surface_set())) is not None
    with pytest.raises(ValidationError, match="unique"):
        surface_set(surfaces=(surface(), surface()))


def test_pixel_and_plane_covariance_propagation():
    # Vertical ray from (1, 2, 1), floor y=.2, local camera ray slopes
    # u->+x and v->+z at 0.001 per pixel. Analytic diagonal is observable.
    cov = intersection_covariance(
        surface(), origin_world_m=(1.0, 2.0, 1.0),
        direction_world=(0.0, -1.0, 0.0),
        ray_direction_jacobian=((0.001, 0.0), (0.0, 0.0), (0.0, 0.001)),
        pixel_sigma_px=2.0,
    )
    assert cov[0] == pytest.approx((1.8 * 0.001 * 2.0) ** 2)
    assert cov[4] == pytest.approx(0.02 ** 2)
    assert cov[8] == pytest.approx(cov[0])
    assert cov[1] == cov[3] == cov[5] == cov[7] == 0.0
    with pytest.raises(ValueError, match="parallel"):
        intersection_covariance(
            surface(), origin_world_m=(1, 2, 1), direction_world=(1, 0, 0),
            ray_direction_jacobian=((0, 0), (0, 0), (0, 0)), pixel_sigma_px=2,
        )


def test_reference_copies_and_validates_observed_polygon():
    reference = SupportSurfaceReference.from_surface(surface())
    assert reference.extent_xz == surface().extent_xz
    assert contains_xz(reference, 1.0, 1.0)
    assert not contains_xz(reference, 3.0, 1.0)
    with pytest.raises(ValidationError, match="extent_xz"):
        SupportSurfaceReference(
            surface_id="missing-extent", revision="survey-1",
            normal_world=(0.0, 1.0, 0.0), offset_m=0.0,
        )
    with pytest.raises(ValidationError, match="convex"):
        SupportSurfaceReference(
            surface_id="concave", revision="survey-1",
            normal_world=(0.0, 1.0, 0.0), offset_m=0.0,
            extent_xz=((0., 0.), (2., 0.), (.5, .5), (2., 2.), (0., 2.)),
        )


def test_slope_covariance_retains_residual_and_dominates_original():
    slope = 0.1
    norm = math.sqrt(1.0 + slope * slope)
    reference = SupportSurfaceReference(
        surface_id="slope", revision="survey-1",
        normal_world=(-slope / norm, 1.0 / norm, 0.0), offset_m=0.0,
        extent_xz=((0., 0.), (2., 0.), (2., 2.), (0., 2.)),
    )
    inconsistent = Matrix3(values=(1., 0., 0., 0., .01, 0., 0., 0., .2))
    corrected = constrained_plane_covariance(reference, inconsistent)
    assert corrected.values[4] >= slope * slope * inconsistent.values[0] + inconsistent.values[4]
    assert corrected.values[1] == pytest.approx(slope * inconsistent.values[0])
    difference = np.asarray(corrected.values).reshape(3, 3) - np.asarray(inconsistent.values).reshape(3, 3)
    assert np.min(np.linalg.eigvalsh(difference)) >= -1e-10

    aligned = Matrix3(values=(.2, .02, 0., .02, .012, 0., 0., 0., .3))
    assert constrained_plane_covariance(reference, aligned).values == pytest.approx(aligned.values)
