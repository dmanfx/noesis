"""Exact legacy contact pixels, with temporary work bounded to their support."""
import numpy as np
import pytest

from noesis.pipelines import hooks


def _full_grid_oracle(crop, contacts):
    x0, y0, x1, y1 = crop
    gy, gx = np.ogrid[y0:y1, x0:x1]
    px = np.asarray(gx, dtype=np.float32) + 0.5
    py = np.asarray(gy, dtype=np.float32) + 0.5
    result = np.zeros((y1-y0, x1-x0), dtype=bool)
    for ax, ay, bx, by, radius, ankle_radius in contacts:
        dx, dy = float(bx)-float(ax), float(by)-float(ay)
        length_sq = dx*dx + dy*dy
        if length_sq > 1.e-6:
            t = np.clip(((px-float(ax))*dx + (py-float(ay))*dy)/length_sq, 0., 1.)
        else:
            t = np.zeros(result.shape, dtype=np.float32)
        nx, ny = float(ax)+t*dx, float(ay)+t*dy
        result |= ((np.square(px-nx)+np.square(py-ny) <= radius*radius)
                   | (np.square(px-float(bx))+np.square(py-float(by)) <= ankle_radius*ankle_radius))
    return result


def _pose(crop, variant):
    x0, y0, x1, y1 = crop
    rng = np.random.default_rng(1000+variant)
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[:, :2] = rng.uniform((-30., -30.), (x1-x0+30., y1-y0+30.), (17, 2)) + (x0, y0)
    pose[:, 2] = .9
    if variant % 4 == 0:  # A zero-length segment and overlapping ankle disks.
        pose[13:17, :2] = (x0+(x1-x0)*.5, y0+(y1-y0)*.5)
    if variant % 4 == 1:
        pose[13:15, 2] = .1  # Ankle-only disk.
    if variant % 4 == 2:
        pose[15:17, 2] = .1  # No ankle support.
    return pose


@pytest.mark.parametrize('origin', [(0, 0), (100, 100), (-130, -210), (1910, 1070),
                                   (2**28-20, 2**28-20)])
def test_contact_patch_is_bitwise_legacy_full_grid_for_clips_edges_and_large_origins(origin):
    crop = (*origin, origin[0]+80, origin[1]+140)
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    for variant in range(32):
        body, contact, uv, visible, capsules = processor._pose_capsule_masks(
            _pose(crop, variant), crop_rect=crop, bbox=(*origin, 80., 140.))
        assert body.shape == contact.shape == (140, 80)
        assert body.dtype == contact.dtype == np.bool_
        assert np.array_equal(contact, _full_grid_oracle(crop, capsules)), (origin, variant)
        assert visible == len(capsules)
        assert (uv is None) == (not capsules)


def test_native_contact_temporary_grids_do_not_scale_with_whole_person_roi(monkeypatch):
    real_grid = np.ogrid
    areas = []

    class ObservedGrid:
        def __getitem__(self, slices):
            areas.append((slices[0].stop-slices[0].start)*(slices[1].stop-slices[1].start))
            return real_grid[slices]

    monkeypatch.setattr(hooks.np, 'ogrid', ObservedGrid())
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    crop = (100, 100, 600, 1000)
    pose = np.zeros((17, 3), dtype=np.float64)
    for index, point in {5:(220,220), 6:(400,220), 11:(240,580), 12:(380,580),
                         13:(250,800), 14:(370,800), 15:(250,960), 16:(370,960)}.items():
        pose[index] = (*point, .9)
    _, contact, _, _, capsules = processor._pose_capsule_masks(
        pose, crop_rect=crop, bbox=(100., 100., 500., 900.))
    assert np.count_nonzero(contact) > 0 and len(capsules) == 2
    assert areas and max(areas) < 450000//20


def test_large_origin_roundoff_keeps_legacy_pixels_outside_real_valued_disk():
    x0 = 2**28-20
    crop = (x0, x0, x0+40, x0+40)
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[13:17] = (2**28, 2**28, .9)
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    _, mask, _, _, capsules = processor._pose_capsule_masks(
        pose, crop_rect=crop, bbox=(float(x0), float(x0), 40., 40.))
    assert np.array_equal(mask, _full_grid_oracle(crop, capsules))
    assert mask[20, 13]  # Float32 rounding: outside the exact disk plus one pixel.
