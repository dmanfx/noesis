from __future__ import annotations

from noesis_core.tracking_continuity import (
    TrackingLifecycleRegistry,
    pair_safe_publication_interval_s,
)


def _track(tracker_id: int) -> dict[str, object]:
    return {"camera_id": "kitchen", "tracker_id": tracker_id}


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
    assert retry.tracker_keys_changed is True
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
