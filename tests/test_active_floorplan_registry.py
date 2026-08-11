from __future__ import annotations

import pytest

from noesis_core.active_floorplan import ActiveFloorplanError, ActiveFloorplanRegistry

_FINGERPRINT = "a" * 64
_SNAPSHOT_DIGEST = "b" * 64
_SNAPSHOT_ID = "snapshot-write-id"


def _alignment() -> dict[str, object]:
    return {
        "version": 1,
        "quality": "ok",
        "reason": "fit_ok",
        "source": "floorplan_depth_snapshot",
        "from": "calibrated_floor_contact_ray_camera_local_xz",
        "to": "floorplan_depth_camera_heading_ground",
        "matrix_2x3": [[1.0, 0.0, 0.2], [0.0, 1.0, -1.0]],
        "sample_count": 128,
        "inlier_count": 120,
        "sample_mode": "walkable_valid_mask_conf",
        "residual_m": {"p50": 0.01, "p90": 0.015, "p95": 0.02, "max": 0.03},
        "determinant": 1.0,
    }


def _payload(*, timestamp_us: int = 2_000_000) -> dict[str, object]:
    return {
        "camera_id": "living-room",
        "ts": timestamp_us + 10,
        "snapshot_ts": timestamp_us,
        "frame": "camera_local_ground_m",
        "orientation": "camera_ground_right_forward",
        "floorplan_contract_version": 10,
        "units": "meters",
        "served_from_cache": False,
        "grid_res_m": 0.15,
        "max_extent_m": 20.0,
        "bounds": {"min_x": -2.0, "max_x": 2.0, "min_z": 0.0, "max_z": 5.0},
        "density": {"grid_shape": [32, 48]},
        "calibration_fingerprint": _FINGERPRINT,
        "snapshot_id": _SNAPSHOT_ID,
        "snapshot_content_sha256": _SNAPSHOT_DIGEST,
        "ray_to_floorplan_alignment": _alignment(),
    }


def _snapshot_ref(timestamp_us: int) -> str:
    return f"snapshots/20260711/12/{timestamp_us}.zarr"


def _record(
    registry: ActiveFloorplanRegistry,
    camera_id: str,
    payload: dict[str, object],
    *,
    snapshot_ref: str | None = None,
) -> bool:
    timestamp_us = int(payload.get("snapshot_ts", payload.get("snapshot_ts_us", 0)))
    return registry.record(
        camera_id,
        payload,
        snapshot_ref=snapshot_ref or _snapshot_ref(timestamp_us),
    )


def test_registry_is_alias_bounded_and_preserves_exact_floorplan_space() -> None:
    registry = ActiveFloorplanRegistry({"0": "living-room", "living-room": "living-room"})

    assert registry.record(
        "0",
        _payload(),
        snapshot_ref="living-room/20260711/12/2000000.zarr",
    )
    assert registry.bounds_for("living-room") == {
        "camera_id": "living-room",
        "snapshot_ts_us": 2_000_000,
        "floorplan_ts_us": 2_000_010,
        "served_from_cache": False,
        "grid_res_m": 0.15,
        "max_extent_m": 20.0,
        "grid_shape": [32, 48],
        "grid_shape_source": "density",
        "bounds": {"min_x": -2.0, "max_x": 2.0, "min_z": 0.0, "max_z": 5.0},
        "frame": "camera_local_ground_m",
        "orientation": "camera_ground_right_forward",
        "floorplan_contract_version": 10,
        "units": "meters",
        "calibration_fingerprint": _FINGERPRINT,
        "snapshot_ref": "living-room/20260711/12/2000000.zarr",
        "snapshot_id": _SNAPSHOT_ID,
        "snapshot_content_sha256": _SNAPSHOT_DIGEST,
        "snapshot_identity": {
            "camera_id": "living-room",
            "snapshot_ts_us": 2_000_000,
            "snapshot_ref": "living-room/20260711/12/2000000.zarr",
            "snapshot_id": _SNAPSHOT_ID,
            "snapshot_content_sha256": _SNAPSHOT_DIGEST,
            "calibration_fingerprint": _FINGERPRINT,
        },
        "source": "active_floorplan",
        "ray_to_floorplan_alignment": _alignment(),
    }
    assert registry.health_snapshot()["healthy"] is True
    assert registry.health_snapshot()["missing_cameras"] == []


def test_registry_rejects_unknown_camera_cross_camera_and_bad_space() -> None:
    registry = ActiveFloorplanRegistry({"0": "living-room"})

    with pytest.raises(ActiveFloorplanError, match="not configured"):
        _record(registry, "attic", _payload())
    bad_camera = _payload()
    bad_camera["camera_id"] = "attic"
    with pytest.raises(ActiveFloorplanError, match="not configured"):
        _record(registry, "0", bad_camera)
    bad_frame = _payload()
    bad_frame["frame"] = "backend_world_m"
    with pytest.raises(ActiveFloorplanError, match="camera_local_ground_m"):
        _record(registry, "0", bad_frame)


def test_registry_ignores_error_and_stale_success_without_erasing_current() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    assert _record(registry, "living-room", _payload(timestamp_us=2_000_000))
    assert registry.record("living-room", {"error": "no_depth"}) is False
    assert (
        _record(registry, "living-room", _payload(timestamp_us=1_000_000)) is False
    )
    assert registry.bounds_for("living-room")["snapshot_ts_us"] == 2_000_000  # type: ignore[index]


def test_registry_rejects_missing_grid_nonfinite_bounds_and_unbounded_alignment() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    missing_grid = _payload()
    missing_grid.pop("density")
    with pytest.raises(ActiveFloorplanError, match="grid shape"):
        _record(registry, "living-room", missing_grid)

    nonfinite = _payload()
    nonfinite["bounds"] = {
        "min_x": float("nan"),
        "max_x": 2.0,
        "min_z": 0.0,
        "max_z": 5.0,
    }
    with pytest.raises(ActiveFloorplanError, match="finite"):
        _record(registry, "living-room", nonfinite)

    huge = _payload()
    huge_alignment = _alignment()
    huge_alignment["reason"] = "x" * 513
    huge["ray_to_floorplan_alignment"] = huge_alignment
    with pytest.raises(ActiveFloorplanError, match="bounded"):
        _record(registry, "living-room", huge)


def test_registry_returns_deeply_isolated_records() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    assert _record(registry, "living-room", payload)

    first = registry.bounds_for("living-room")
    assert first is not None
    first["bounds"]["min_x"] = -999.0
    first["grid_shape"][0] = 999
    first["ray_to_floorplan_alignment"]["matrix_2x3"][0][0] = 999
    first["snapshot_identity"]["snapshot_ref"] = "mutated"

    # Mutating either the caller-owned payload or a returned record cannot
    # mutate the registry's canonical copy.
    payload["bounds"]["max_x"] = 999.0  # type: ignore[index]
    payload["density"]["grid_shape"][1] = 999  # type: ignore[index]
    payload["ray_to_floorplan_alignment"]["matrix_2x3"][0][1] = 999  # type: ignore[index]

    second = registry.bounds_for("living-room")
    assert second is not None
    assert second["bounds"] == {
        "min_x": -2.0,
        "max_x": 2.0,
        "min_z": 0.0,
        "max_z": 5.0,
    }
    assert second["grid_shape"] == [32, 48]
    assert second["ray_to_floorplan_alignment"]["matrix_2x3"] == [
        [1.0, 0.0, 0.2],
        [0.0, 1.0, -1.0],
    ]
    assert second["snapshot_identity"]["snapshot_ref"] == _snapshot_ref(2_000_000)


def test_registry_orders_by_exact_snapshot_before_generation_timestamp() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    current = _payload(timestamp_us=2_000_000)
    current["ts"] = 9_000_000
    assert _record(registry, "living-room", current)

    # A newer generation timestamp cannot make geometry from an older raw
    # snapshot current.
    older_snapshot = _payload(timestamp_us=1_000_000)
    older_snapshot["ts"] = 10_000_000
    assert _record(registry, "living-room", older_snapshot) is False

    # Conversely, the exact newer snapshot wins even when its generation clock
    # is numerically lower than the previous response's clock.
    newer_snapshot = _payload(timestamp_us=3_000_000)
    newer_snapshot["ts"] = 8_000_000
    assert _record(registry, "living-room", newer_snapshot)
    assert registry.bounds_for("living-room")["snapshot_ts_us"] == 3_000_000  # type: ignore[index]


def test_registry_rejects_conflicts_for_the_same_floorplan_version() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    assert registry.record(
        "living-room", payload, snapshot_ref="exact/2000000.zarr"
    )

    replay = _payload()
    replay["served_from_cache"] = True
    assert registry.record(
        "living-room", replay, snapshot_ref="exact/2000000.zarr"
    )

    conflicting_bounds = _payload()
    conflicting_bounds["bounds"] = {
        "min_x": -3.0,
        "max_x": 2.0,
        "min_z": 0.0,
        "max_z": 5.0,
    }
    with pytest.raises(ActiveFloorplanError, match="conflicting payload"):
        registry.record(
            "living-room", conflicting_bounds, snapshot_ref="exact/2000000.zarr"
        )

    with pytest.raises(ActiveFloorplanError, match="conflicting payload"):
        registry.record(
            "living-room", _payload(), snapshot_ref="other/2000000.zarr"
        )

    assert registry.bounds_for("living-room")["bounds"]["min_x"] == -2.0  # type: ignore[index]


@pytest.mark.parametrize(
    "aliases",
    [
        {"0": "living-room", "living-room": "kitchen"},
        {"living-room": "kitchen", "0": "living-room"},
        {"living-room": "kitchen", "0": " living-room "},
    ],
)
def test_registry_rejects_alias_canonical_collisions(
    aliases: dict[str, str],
) -> None:
    with pytest.raises(ActiveFloorplanError, match="collides with canonical"):
        ActiveFloorplanRegistry(aliases)


def test_registry_health_requires_every_configured_camera() -> None:
    registry = ActiveFloorplanRegistry(
        {"0": "living-room", "1": "kitchen", "2": "garage"}
    )
    assert registry.health_snapshot() == {
        "contract": "noesis.active_floorplan.health",
        "contract_version": 1,
        "healthy": False,
        "configured_camera_count": 3,
        "active_camera_count": 0,
        "missing_cameras": ["garage", "kitchen", "living-room"],
        "rejection_count": 0,
        "stale_count": 0,
        "conflict_count": 0,
        "last_error": None,
        "reset_count": 0,
        "last_reset": None,
        "cameras": {},
    }

    assert _record(registry, "0", _payload())
    partial = registry.health_snapshot()
    assert partial["healthy"] is False
    assert partial["active_camera_count"] == 1
    assert partial["missing_cameras"] == ["garage", "kitchen"]

    kitchen = _payload(timestamp_us=3_000_000)
    kitchen["camera_id"] = "kitchen"
    assert _record(registry, "1", kitchen)
    assert registry.health_snapshot()["healthy"] is False

    garage = _payload(timestamp_us=4_000_000)
    garage["camera_id"] = "garage"
    assert _record(registry, "2", garage)
    complete = registry.health_snapshot()
    assert complete["healthy"] is True
    assert complete["active_camera_count"] == 3
    assert complete["missing_cameras"] == []


def test_registry_bounds_alias_cardinality_including_canonical_names() -> None:
    valid = {f"alias-{index}": "living-room" for index in range(255)}
    valid["living-room"] = "living-room"
    ActiveFloorplanRegistry(valid)

    too_many_explicit = {
        f"alias-{index}": "living-room" for index in range(257)
    }
    with pytest.raises(ActiveFloorplanError, match="must not exceed 256"):
        ActiveFloorplanRegistry(too_many_explicit)

    # The automatically registered canonical name is part of the same bound.
    missing_canonical = {
        f"alias-{index}": "living-room" for index in range(256)
    }
    with pytest.raises(ActiveFloorplanError, match="must not exceed 256"):
        ActiveFloorplanRegistry(missing_canonical)


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_registry_requires_strict_boolean_cache_identity(value: object) -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    if value is None:
        payload.pop("served_from_cache")
    else:
        payload["served_from_cache"] = value
    with pytest.raises(ActiveFloorplanError, match="must be boolean"):
        _record(registry, "living-room", payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"bounds": {"min_x": -10_001.0, "max_x": 2.0, "min_z": 0.0, "max_z": 5.0}}, "bounded"),
        ({"grid_res_m": float("inf")}, "finite"),
        ({"grid_res_m": 10_001.0}, "bounded"),
        ({"max_extent_m": 10_001.0}, "bounded"),
        ({"density": {"grid_shape": [4097, 4097]}}, "grid shape"),
        ({"density": {"grid_shape": [32, 48], "value_max": float("inf")}}, "finite"),
        ({"density": {"grid_shape": [32, 48], "value_max": 1_000_001.0}}, "bounded"),
        (
            {
                "density": {
                    "grid_shape": [32, 48],
                    "value_min": 2.0,
                    "value_max": 1.0,
                }
            },
            "greater than or equal",
        ),
    ],
)
def test_registry_rejects_unbounded_floorplan_geometry(
    mutation: dict[str, object],
    message: str,
) -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    payload.update(mutation)
    with pytest.raises(ActiveFloorplanError, match=message):
        _record(registry, "living-room", payload)


def test_registry_requires_every_present_layer_to_use_the_same_shape() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    mismatched = _payload()
    mismatched["height"] = {"grid_shape": [32, 47]}
    with pytest.raises(ActiveFloorplanError, match="one grid shape"):
        _record(registry, "living-room", mismatched)

    malformed = _payload()
    malformed["height"] = {"grid_shape": [32, 48, 1]}
    with pytest.raises(ActiveFloorplanError, match="exactly two"):
        _record(registry, "living-room", malformed)

    matching = _payload()
    matching["height"] = {
        "grid_shape": [32, 48],
        "value_min": 0.0,
        "value_max": 2.0,
    }
    matching["distance"] = {"shape": (32, 48)}
    assert _record(registry, "living-room", matching)


@pytest.mark.parametrize(
    ("snapshot_ref", "message"),
    [
        (None, "portable relative path"),
        ("/snapshots/2000000.zarr", "portable relative path"),
        ("snapshots/../2000000.zarr", "portable relative path"),
        ("snapshots\\2000000.zarr", "portable relative path"),
        ("snapshots/2000001.zarr", "exact snapshot timestamp"),
        (f"{'x' * 1025}/2000000.zarr", "bounded"),
    ],
)
def test_registry_requires_bounded_portable_exact_snapshot_ref(
    snapshot_ref: str | None,
    message: str,
) -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    with pytest.raises(ActiveFloorplanError, match=message):
        registry.record("living-room", _payload(), snapshot_ref=snapshot_ref)


@pytest.mark.parametrize("fingerprint", [None, "A" * 64, "a" * 63, "z" * 64])
def test_registry_binds_a_canonical_calibration_fingerprint(
    fingerprint: object,
) -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    if fingerprint is None:
        payload.pop("calibration_fingerprint")
    else:
        payload["calibration_fingerprint"] = fingerprint
    with pytest.raises(ActiveFloorplanError, match="lowercase SHA-256"):
        _record(registry, "living-room", payload)


def test_registry_rejects_conflicting_snapshot_timestamp_aliases() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    payload["snapshot_ts_us"] = 2_000_001
    with pytest.raises(ActiveFloorplanError, match="timestamp fields conflict"):
        _record(registry, "living-room", payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"extra": "arbitrary"}, "unsupported fields"),
        ({"matrix_2x3": [[1.0, 0.0], [0.0, 1.0]]}, "shape 2x3"),
        (
            {"matrix_2x3": [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0]]},
            "finite",
        ),
        ({"inlier_count": 129}, "counts are inconsistent"),
        (
            {"residual_m": {"p50": 0.2, "p90": 0.1, "p95": 0.3, "max": 0.4}},
            "residuals are inconsistent",
        ),
        ({"determinant": 2.0}, "determinant does not match"),
    ],
)
def test_registry_validates_alignment_schema_and_shape(
    mutation: dict[str, object],
    message: str,
) -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    alignment = _alignment()
    alignment.update(mutation)
    payload["ray_to_floorplan_alignment"] = alignment
    with pytest.raises(ActiveFloorplanError, match=message):
        _record(registry, "living-room", payload)


def test_registry_accepts_bounded_unavailable_alignment_without_fit_values() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    payload = _payload()
    payload["ray_to_floorplan_alignment"] = {
        "version": 1,
        "quality": "unavailable",
        "reason": "insufficient_surface_samples",
        "source": "floorplan_depth_snapshot",
        "from": "calibrated_floor_contact_ray_camera_local_xz",
        "to": "floorplan_depth_camera_heading_ground",
        "sample_count": 12,
        "sample_mode": "valid_mask_conf",
    }
    assert _record(registry, "living-room", payload)

    payload_with_null_fit = _payload(timestamp_us=3_000_000)
    unavailable = dict(payload["ray_to_floorplan_alignment"])  # type: ignore[arg-type]
    unavailable["matrix_2x3"] = None
    payload_with_null_fit["ray_to_floorplan_alignment"] = unavailable
    with pytest.raises(ActiveFloorplanError, match="must not contain fit values"):
        _record(registry, "living-room", payload_with_null_fit)


def test_registry_reports_rejection_stale_conflict_and_last_error_truth() -> None:
    registry = ActiveFloorplanRegistry({"living-room": "living-room"})
    assert _record(registry, "living-room", _payload())

    assert registry.record("living-room", {"error": "no_depth"}) is False
    assert _record(registry, "living-room", _payload(timestamp_us=1_000_000)) is False
    conflicting = _payload()
    conflicting["bounds"] = {
        "min_x": -3.0,
        "max_x": 2.0,
        "min_z": 0.0,
        "max_z": 5.0,
    }
    with pytest.raises(ActiveFloorplanError, match="conflicting payload"):
        _record(registry, "living-room", conflicting)

    invalid = _payload(timestamp_us=3_000_000)
    invalid["served_from_cache"] = "false"
    with pytest.raises(ActiveFloorplanError, match="must be boolean"):
        _record(registry, "living-room", invalid)

    health = registry.health_snapshot()
    assert health["rejection_count"] == 4
    assert health["stale_count"] == 1
    assert health["conflict_count"] == 1
    assert health["last_error"] == {
        "kind": "invalid",
        "camera_id": "living-room",
        "message": "served_from_cache must be boolean",
        "rejection_sequence": 4,
    }


def test_registry_clear_one_and_all_exposes_reset_truth() -> None:
    registry = ActiveFloorplanRegistry({"0": "living-room", "1": "kitchen"})
    assert _record(registry, "0", _payload())
    kitchen = _payload(timestamp_us=3_000_000)
    kitchen["camera_id"] = "kitchen"
    assert _record(registry, "1", kitchen)
    assert registry.health_snapshot()["healthy"] is True

    assert registry.clear("0") == 1
    partial = registry.health_snapshot()
    assert partial["healthy"] is False
    assert partial["active_camera_count"] == 1
    assert partial["missing_cameras"] == ["living-room"]
    assert partial["reset_count"] == 1
    assert partial["last_reset"] == {
        "scope": "camera",
        "camera_id": "living-room",
        "removed_count": 1,
        "sequence": 1,
    }
    assert registry.bounds_for("living-room") is None

    assert registry.clear() == 1
    cleared = registry.health_snapshot()
    assert cleared["healthy"] is False
    assert cleared["active_camera_count"] == 0
    assert cleared["missing_cameras"] == ["kitchen", "living-room"]
    assert cleared["reset_count"] == 2
    assert cleared["last_reset"] == {
        "scope": "all",
        "removed_count": 1,
        "sequence": 2,
    }
    assert registry.clear() == 0
    assert registry.health_snapshot()["last_reset"]["removed_count"] == 0
