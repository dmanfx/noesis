from __future__ import annotations

from noesis_core.tracking_continuity import (
    TrackingLifecycleRegistry,
    pair_safe_publication_interval_s,
)


def _track(
    tracker_id: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "camera_id": "kitchen",
        "tracker_id": tracker_id,
    }
    if bbox is not None:
        row["bbox"] = list(bbox)
    return row


def test_tombstone_binds_last_published_not_last_processed_presence() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=11,
        observed_at_us=1_000_000,
        tracks=[first],
    )
    assert update.tracker_keys_changed is True
    assert update.tombstones == ()
    assert first["tracker_lifecycle_generation"] == 1
    registry.mark_published(update)

    carried = _track(7)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=12,
        observed_at_us=1_100_000,
        tracks=[carried],
    )
    assert update.tracker_keys_changed is False
    assert carried["tracker_lifecycle_generation"] == 1

    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=13,
        observed_at_us=1_200_000,
        tracks=[],
    )
    assert update.tracker_keys_changed is True
    assert update.tombstones == (
        {
            "camera_id": "kitchen",
            "tracker_id": 7,
            "tracker_lifecycle_generation": 1,
            "last_seen_frame_id": 11,
            "last_seen_observed_at_us": 1_000_000,
            "disappeared_at_frame_id": 13,
            "disappeared_at_observed_at_us": 1_200_000,
        },
    )
    registry.mark_published(update)

    reappeared = _track(7)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=14,
        observed_at_us=1_300_000,
        tracks=[reappeared],
    )
    assert update.tracker_keys_changed is True
    assert reappeared["tracker_lifecycle_generation"] == 2


def test_same_count_tracker_replacement_is_a_forced_transition() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7)
    first_update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=1,
        observed_at_us=1,
        tracks=[first],
    )
    registry.mark_published(first_update)
    replacement = _track(8)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=2,
        observed_at_us=2,
        tracks=[replacement],
    )
    assert update.tracker_keys_changed is True
    assert replacement["tracker_lifecycle_generation"] == 2
    assert update.tombstones[0]["tracker_id"] == 7


def test_failed_disappearance_publish_retries_with_current_frame_receipt() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7)
    published = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=11,
        observed_at_us=1_000_000,
        tracks=[first],
    )
    registry.mark_published(published)
    skipped = _track(7)
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=12,
        observed_at_us=1_100_000,
        tracks=[skipped],
    )

    failed_attempt = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=13,
        observed_at_us=1_200_000,
        tracks=[],
    )
    assert failed_attempt.tracker_keys_changed is True
    assert failed_attempt.tombstones[0]["last_seen_frame_id"] == 11
    assert failed_attempt.tombstones[0]["disappeared_at_frame_id"] == 13

    retry = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=14,
        observed_at_us=1_300_000,
        tracks=[],
    )
    # The first absence already forced the queued disappearance. Repeated
    # empty frames must not remain forced while that receipt drains.
    assert retry.tracker_keys_changed is False
    assert retry.tombstones[0]["last_seen_frame_id"] == 11
    assert retry.tombstones[0]["last_seen_observed_at_us"] == 1_000_000
    assert retry.tombstones[0]["disappeared_at_frame_id"] == 14
    assert retry.tombstones[0]["disappeared_at_observed_at_us"] == 1_300_000
    registry.mark_published(retry)

    settled = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=15,
        observed_at_us=1_400_000,
        tracks=[],
    )
    assert settled.tracker_keys_changed is False
    assert settled.tombstones == ()


def test_duplicate_tracker_is_rejected_before_publication() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7)
    duplicate = _track(7)
    try:
        registry.update_frame(
            source_id=0,
            camera_id="kitchen",
            frame_id=1,
            observed_at_us=1,
            tracks=[first, duplicate],
        )
    except ValueError as exc:
        assert "duplicate tracker_id" in str(exc)
    else:  # pragma: no cover - assertion helper without pytest dependency
        raise AssertionError("duplicate tracker_id was accepted")


def test_pair_safe_interval_uses_slower_tracking_or_bev_ceiling() -> None:
    assert pair_safe_publication_interval_s(
        track_count=1,
        tracking_interval_s=1.0 / 15.0,
        empty_tracking_interval_s=0.5,
        bev_interval_s=1.0 / 12.0,
        bev_active=True,
    ) == 1.0 / 12.0
    assert pair_safe_publication_interval_s(
        track_count=0,
        tracking_interval_s=1.0 / 15.0,
        empty_tracking_interval_s=0.5,
        bev_interval_s=1.0 / 12.0,
        bev_active=True,
    ) == 0.5


def test_tracking_cadence_is_unchanged_without_bev() -> None:
    assert pair_safe_publication_interval_s(
        track_count=1,
        tracking_interval_s=1.0 / 15.0,
        empty_tracking_interval_s=0.5,
        bev_interval_s=1.0 / 12.0,
        bev_active=False,
    ) == 1.0 / 15.0


def test_source_reset_starts_new_epoch_without_reusing_tracker_generation() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=42,
        observed_at_us=4_200_000,
        tracks=[first],
    )
    registry.mark_published(update)

    assert registry.reset_source(0) == 1
    replayed = _track(7)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=0,
        observed_at_us=4_300_000,
        tracks=[replayed],
    )

    assert update.source_epoch == 1
    assert replayed["tracker_lifecycle_generation"] == 2


def test_frame_generation_reservations_are_unique_and_match_commit() -> None:
    registry = TrackingLifecycleRegistry()
    first_generation = registry.peek_generation(
        0,
        7,
        frame_id=3,
        observed_at_us=3_000_000,
    )
    second_generation = registry.peek_generation(
        0,
        8,
        frame_id=3,
        observed_at_us=3_000_000,
    )
    first = _track(7)
    second = _track(8)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=3,
        observed_at_us=3_000_000,
        tracks=[first, second],
    )

    assert (first_generation, second_generation) == (1, 2)
    assert first["tracker_lifecycle_generation"] == first_generation
    assert second["tracker_lifecycle_generation"] == second_generation
    assert [item.generation for item in update.active_lifecycles] == [1, 2]


def test_compatible_one_frame_reappearance_reuses_generation_after_tombstone() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7, (100.0, 80.0, 60.0, 120.0))
    present = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=10,
        observed_at_us=1_000_000,
        tracks=[first],
    )
    registry.mark_published(registry.prepare_publication(present))

    absent = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=11,
        observed_at_us=1_033_333,
        tracks=[],
    )
    absent_for_publication = registry.prepare_publication(absent)
    assert absent_for_publication.tombstones[0]["tracker_id"] == 7
    assert absent_for_publication.tombstones[0][
        "tracker_lifecycle_generation"
    ] == 1
    registry.mark_published(absent_for_publication)

    reappeared_bbox = (104.0, 82.0, 61.0, 119.0)
    reserved = registry.peek_generation(
        0,
        7,
        frame_id=12,
        observed_at_us=1_066_666,
        bbox=reappeared_bbox,
    )
    reappeared = _track(7, reappeared_bbox)
    update = registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=12,
        observed_at_us=1_066_666,
        tracks=[reappeared],
    )

    assert reserved == 1
    assert reappeared["tracker_lifecycle_generation"] == 1
    assert update.active_lifecycles[0].generation == 1


def test_nearby_but_size_incompatible_reappearance_gets_new_generation() -> None:
    registry = TrackingLifecycleRegistry()
    first = _track(7, (100.0, 80.0, 60.0, 120.0))
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=1,
        observed_at_us=1_000_000,
        tracks=[first],
    )
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=2,
        observed_at_us=1_033_333,
        tracks=[],
    )

    # Center remains close, but the 3x scale change is not a plausible
    # one-frame continuation of the retired box.
    incompatible_bbox = (40.0, -40.0, 180.0, 360.0)
    reserved = registry.peek_generation(
        0,
        7,
        frame_id=3,
        observed_at_us=1_066_666,
        bbox=incompatible_bbox,
    )
    reappeared = _track(7, incompatible_bbox)
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=3,
        observed_at_us=1_066_666,
        tracks=[reappeared],
    )

    assert reserved == 2
    assert reappeared["tracker_lifecycle_generation"] == 2


def test_expired_reappearance_gap_gets_new_generation() -> None:
    registry = TrackingLifecycleRegistry()
    bbox = (100.0, 80.0, 60.0, 120.0)
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=1,
        observed_at_us=1_000_000,
        tracks=[_track(7, bbox)],
    )
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=2,
        observed_at_us=1_033_333,
        tracks=[],
    )
    reappeared = _track(7, bbox)
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=20,
        observed_at_us=1_400_001,
        tracks=[reappeared],
    )

    assert reappeared["tracker_lifecycle_generation"] == 2


def test_retired_generation_grace_state_is_bounded() -> None:
    registry = TrackingLifecycleRegistry(max_retired_per_source=2)
    tracks = [
        _track(1, (10.0, 10.0, 40.0, 80.0)),
        _track(2, (100.0, 10.0, 40.0, 80.0)),
        _track(3, (200.0, 10.0, 40.0, 80.0)),
    ]
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=1,
        observed_at_us=1_000_000,
        tracks=tracks,
    )
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=2,
        observed_at_us=1_033_333,
        tracks=[],
    )

    assert registry.retired_lifecycle_count(0) == 2
    evicted = _track(1, (11.0, 11.0, 40.0, 80.0))
    registry.update_frame(
        source_id=0,
        camera_id="kitchen",
        frame_id=3,
        observed_at_us=1_066_666,
        tracks=[evicted],
    )
    assert evicted["tracker_lifecycle_generation"] == 4
    assert registry.retired_lifecycle_count(0) <= 2
