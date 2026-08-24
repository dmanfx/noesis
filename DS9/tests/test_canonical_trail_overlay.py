from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import numpy as np

# DS9's conftest puts the adapter ahead of the repository root.  Add the root
# explicitly for shared geometry modules while retaining the canonical package
# precedence for ``noesis``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geometry.homography import project_world_to_image
from noesis.pipelines import hooks as canonical_hooks
from noesis.telemetry.bev import CalibrationSnapshot


CANONICAL_HOOKS = (
    Path(__file__).resolve().parents[1] / "noesis" / "pipelines" / "hooks.py"
).resolve()


@dataclass
class _Rect:
    left: float
    top: float
    width: float
    height: float


@dataclass
class _Object:
    object_id: int
    rect_params: _Rect
    class_id: int = 0


@dataclass
class _Frame:
    object_items: list[Any]
    source_id: int = 0
    frame_num: int = 0
    source_width: int = 100
    source_height: int = 50
    compositor_rect: Any = None
    appended: list[Any] = field(default_factory=list)

    def append(self, item: Any) -> None:
        self.appended.append(item)


class _Analytics:
    def __init__(self) -> None:
        self.tracks: dict[int, dict[int, dict[str, Any]]] = {}
        self.calls = 0

    def get_active_track_map(self, sensor_id: int) -> dict[int, dict[str, Any]]:
        self.calls += 1
        return {
            int(track_id): dict(track)
            for track_id, track in self.tracks.get(int(sensor_id), {}).items()
        }


def _calibration(
    *,
    camera_id: str = "cam0",
    world_frame_revision: str = "rev-a",
) -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id=camera_id,
        intrinsics=np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        # Camera at (0, 2, -5), looking down the +Z axis.  This is enough for
        # a deterministic world-to-image reprojection test.
        extrinsics_col_major=list(np.eye(4, dtype=np.float64).flatten(order="F")),
        floor_y=0.0,
        image_size=(100, 50),
        unit_scale=1.0,
        world_frame_id="backend_world_m",
        world_frame_revision=world_frame_revision,
    )


class _CalibrationProvider:
    def __init__(self, active: CalibrationSnapshot, raw: CalibrationSnapshot | None = None) -> None:
        self.active = active
        self.raw = raw or active
        self.world_calls = 0
        self.raw_calls = 0

    def world_snapshot(self, source_id: int, camera_id: str) -> CalibrationSnapshot:
        self.world_calls += 1
        return self.active

    def snapshot(self, source_id: int, camera_id: str) -> CalibrationSnapshot:
        self.raw_calls += 1
        return self.raw


class _Batch:
    def acquire_display_meta(self) -> Any:
        return SimpleNamespace(n_lines=0, n_labels=0, add_line=lambda _line: None, add_text=lambda _text: None)


def _processor(
    *,
    analytics: _Analytics | None = None,
    provider: _CalibrationProvider | None = None,
) -> tuple[canonical_hooks.TrailOverlayProcessor, _Analytics, _CalibrationProvider]:
    analytics = analytics or _Analytics()
    provider = provider or _CalibrationProvider(_calibration())
    pipeline = SimpleNamespace(
        components={
            "tiler": SimpleNamespace(
                config={"width": 300, "height": 50, "columns": 3, "rows": 1}
            )
        },
        frame_size=(100, 50),
        batch_size=3,
        camera_labels={0: "cam0", 1: "cam1", 2: "cam2"},
        bev_calibration=provider,
        analytics_telemetry_processor=analytics,
        stable_id_mgr=None,
    )
    config = canonical_hooks.TrailOverlayConfig(
        enabled=True,
        class_ids=frozenset({0}),
        anchor_mode="floor_plane_gravity_drop",
        draw_stride=1,
        min_step_px=0.0,
        min_dt_s=0.0,
        smooth_tau_s=10.0,
        max_speed_px_per_s=1.0,
        gap_predict_ttl_s=0.0,
        show_labels=False,
    )
    return canonical_hooks.TrailOverlayProcessor(pipeline, config), analytics, provider


def _track(
    *,
    frame_id: int,
    image_base: tuple[float, float] | None = (50.0, 25.0),
    image_size: tuple[int, int] = (100, 50),
    generation: int = 1,
    world: tuple[float, float, float] = (0.0, 0.0, 5.0),
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tracker_id": 7,
        "frame_id": int(frame_id),
        "tracker_lifecycle_generation": int(generation),
        "world": list(world),
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-a",
        "image_size": list(image_size),
        "trail_append_allowed": True,
    }
    if image_base is not None:
        row["image_base"] = list(image_base)
    return row


def _step(
    processor: canonical_hooks.TrailOverlayProcessor,
    analytics: _Analytics,
    *,
    source_id: int,
    frame_id: int,
    track: dict[str, Any] | None,
    bbox: _Rect = _Rect(10.0, 10.0, 20.0, 20.0),
    now: float = 1.0,
    object_present: bool = True,
) -> canonical_hooks._TrailTrackState:
    analytics.tracks[source_id] = {7: track} if track is not None else {}
    frame = _Frame(
        object_items=[_Object(7, bbox)] if object_present else [],
        source_id=source_id,
        frame_num=frame_id,
    )
    processor._handle_frame(frame, _Batch(), now=now)  # type: ignore[attr-defined]
    return processor._tracks[source_id].get(7)  # type: ignore[attr-defined,return-value]


def test_import_is_the_canonical_ds9_hooks_module() -> None:
    assert Path(canonical_hooks.__file__).resolve() == CANONICAL_HOOKS


def test_missing_compositor_rect_maps_source_0_1_2_to_configured_3x1_tiles() -> None:
    processor, _analytics, _provider = _processor()
    for source_id, expected_x in ((0, 50.0), (1, 150.0), (2, 250.0)):
        mapped = processor._source_to_mosaic(  # type: ignore[attr-defined]
            _Frame([], source_id=source_id), 50.0, 25.0, (100, 50)
        )
        assert mapped == (expected_x, 25.0)


def test_out_of_source_v_is_rejected_instead_of_clamped_to_tile_bottom() -> None:
    processor, _analytics, _provider = _processor()
    mapped = processor._source_to_mosaic(_Frame([], source_id=1), 50.0, 1000.0, (100, 50))  # type: ignore[attr-defined]
    assert mapped is None


def test_track_declared_image_size_controls_anchor_scaling_and_map_is_fetched_once() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=1,
        frame_id=10,
        track=_track(frame_id=10, image_base=(50.0, 25.0), image_size=(100, 50)),
    )
    assert analytics.calls == 1
    # Source 1 starts at x=100; the declared 100x50 anchor is its center.
    assert state.points[-1].x == 150.0
    assert state.points[-1].y == 25.0


def test_frame_n_plus_one_row_is_not_joined_to_frame_n_and_never_uses_bbox() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=10,
        track=_track(frame_id=11, image_base=(90.0, 45.0)),
        bbox=_Rect(0.0, 0.0, 20.0, 20.0),
    )
    assert state is None


def test_missing_exact_canonical_row_clears_existing_trail_without_refreshing_it() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=10,
        track=_track(frame_id=10, image_base=(20.0, 20.0)),
        now=1.0,
    )
    assert state is not None
    assert len(state.points) == 1
    prior_last_seen = state.last_seen_ts

    missing_state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=11,
        track=None,
        bbox=_Rect(60.0, 20.0, 20.0, 20.0),
        now=2.0,
    )

    assert missing_state is state
    assert list(state.points) == []
    assert state.last_seen_ts == prior_last_seen


def test_world_revision_mismatch_clears_existing_trail_and_emits_no_anchor() -> None:
    active_a = _calibration()
    active_b = _calibration(world_frame_revision="rev-b")
    provider = _CalibrationProvider(active=active_a)
    processor, analytics, _provider = _processor(provider=provider)
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=1,
        track=_track(frame_id=1),
        now=1.0,
    )
    assert state is not None
    assert len(state.points) == 1

    provider.active = active_b
    mismatched = _track(frame_id=2)
    result = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=2,
        track=mismatched,
        now=2.0,
    )

    assert result is state
    assert list(state.points) == []
    assert state.last_measure_world_x is None
    assert state.last_measure_world_z is None


def test_invalid_exact_canonical_row_clears_existing_trail_without_refreshing_it() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=20,
        track=_track(frame_id=20, image_base=(20.0, 20.0)),
        now=1.0,
    )
    prior_last_seen = state.last_seen_ts
    invalid = _track(frame_id=21, image_base=(20.0, 20.0))
    invalid["world_valid"] = False
    invalid.pop("world", None)
    invalid_state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=21,
        track=invalid,
        now=2.0,
    )

    assert invalid_state is state
    assert list(state.points) == []
    assert state.last_seen_ts == prior_last_seen


def test_exact_absence_without_sdk_object_breaks_same_generation_return() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=30,
        track=_track(frame_id=30, generation=4, world=(0.0, 0.0, 5.0)),
        now=1.0,
    )
    _step(
        processor,
        analytics,
        source_id=0,
        frame_id=31,
        track=_track(frame_id=31, generation=4, world=(0.5, 0.0, 5.0)),
        now=2.0,
    )
    assert len(state.points) == 2

    absent_state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=32,
        track=None,
        object_present=False,
        now=3.0,
    )
    assert absent_state is state
    assert list(state.points) == []

    returned_state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=33,
        track=_track(frame_id=33, generation=4, world=(1.0, 0.0, 5.0)),
        now=4.0,
    )
    assert returned_state is state
    assert state.tracker_lifecycle_generation == 4
    assert len(state.points) == 1


def test_lifecycle_generation_change_resets_reused_tracker_id_before_new_point() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=1,
        track=_track(frame_id=1, image_base=(20.0, 20.0), generation=1),
        now=1.0,
    )
    _step(
        processor,
        analytics,
        source_id=0,
        frame_id=2,
        track=_track(frame_id=2, image_base=(30.0, 20.0), generation=1),
        now=2.0,
    )
    assert len(state.points) == 2
    _step(
        processor,
        analytics,
        source_id=0,
        frame_id=3,
        track=_track(frame_id=3, image_base=(80.0, 20.0), generation=2),
        now=3.0,
    )
    assert state.tracker_lifecycle_generation == 2
    assert len(state.points) == 1
    assert state.points[-1].x == 50.0
    assert state.points[-1].y == 25.0


def test_anchor_basis_change_resets_history_and_floor_anchor_bypasses_osd_estimator() -> None:
    processor, analytics, _provider = _processor()
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=1,
        track=_track(frame_id=1, image_base=(10.0, 10.0), image_size=(100, 50)),
        now=1.0,
    )
    _step(
        processor,
        analytics,
        source_id=0,
        frame_id=2,
        track=_track(frame_id=2, image_base=(90.0, 40.0), image_size=(200, 100)),
        now=2.0,
    )
    assert len(state.points) == 1
    # The 1 px/s clamp and 10 s EMA are intentionally bypassed for canonical
    # floor anchors; the point is the exact transformed source anchor.
    # Both rows are rendered from the active world snapshot.  The second row
    # declares a different image size, so the active-calibration center (50,25)
    # scales to (100,50) in the 200x100 source and maps back to (50,25) in the
    # 100x50 source-0 tile.  Its image_base=(90,40) must not be render authority.
    assert state.points[-1].x == 50.0
    assert state.points[-1].y == 25.0


def test_floor_trail_uses_active_world_projection_and_ignores_image_anchor() -> None:
    active = _calibration()
    raw = _calibration()
    raw.intrinsics[0, 2] = 0.0
    provider = _CalibrationProvider(active=active, raw=raw)
    processor, analytics, provider = _processor(provider=provider)

    # Deliberately put image_base far from the active world projection.  A
    # floor trail must land at the same source pixel that BEV receives from
    # the canonical world row, not at this detector-provided image anchor.
    expected = project_world_to_image(
        [0.0, 0.0, 5.0],
        active.intrinsics,
        active.extrinsics_col_major,
        active.image_size,
        unit_scale=1.0,
    )
    assert expected is not None
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=1,
        track=_track(frame_id=1, image_base=(3.0, 47.0)),
    )

    assert provider.world_calls == 1
    assert provider.raw_calls == 0
    assert state.points[-1].x == expected[0]
    assert state.points[-1].y == expected[1]


def test_reprojection_uses_active_world_snapshot_not_raw_snapshot() -> None:
    active = _calibration()
    raw = _calibration()
    raw.intrinsics[0, 2] = 0.0
    provider = _CalibrationProvider(active=active, raw=raw)
    processor, analytics, provider = _processor(provider=provider)
    # Active identity calibration projects this point to the image center.
    expected = project_world_to_image(
        [0.0, 0.0, 5.0],
        active.intrinsics,
        active.extrinsics_col_major,
        active.image_size,
        unit_scale=1.0,
    )
    assert expected is not None
    state = _step(
        processor,
        analytics,
        source_id=0,
        frame_id=1,
        track=_track(frame_id=1, image_base=None),
    )
    assert provider.world_calls == 1
    assert provider.raw_calls == 0
    assert state.points[-1].x == expected[0]
    assert state.points[-1].y == expected[1]
