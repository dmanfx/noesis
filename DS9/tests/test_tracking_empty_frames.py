from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def test_ds9_processor_publishes_rate_limited_advancing_empty_frames() -> None:
    script = textwrap.dedent(
        """
        from pathlib import Path
        from types import SimpleNamespace

        from noesis.pipelines import hooks
        from noesis.telemetry.publishers import TrackingPublicationReceipt
        from noesis_core.runtime_publication import RuntimePublicationGate

        assert Path(hooks.__file__).resolve().is_relative_to(
            (Path.cwd() / "DS9").resolve()
        )

        published = []

        class Publisher:
            def __init__(self):
                self.sequence = 0

            def publish(self, source_id, tracks, *, frame_metadata=None):
                published.append(
                    (int(source_id), list(tracks), dict(frame_metadata or {}))
                )
                receipt = TrackingPublicationReceipt(
                    source_id=int(source_id),
                    frame_id=int(frame_metadata["frame_id"]),
                    observed_at_us=int(frame_metadata["observed_at_us"]),
                    tracking_publication_sequence=self.sequence,
                    outbound_submission_id=self.sequence * 2 + 1,
                    outbound_message_count=1,
                )
                self.sequence += 1
                return receipt

        pipeline = SimpleNamespace(config={}, stable_id_mgr=None)
        processor = hooks._AnalyticsTelemetryProcessor(
            pipeline=pipeline,
            tracking_pub=Publisher(),
            camera_labels={0: "living-room"},
            sensor_id_map={},
            publication_gate=RuntimePublicationGate(),
        )
        processor._tracking_empty_publish_interval_s = 0.5
        now = [100.0]
        hooks.time.time = lambda: now[0]

        def empty_frame(frame_id):
            return SimpleNamespace(
                source_id=0,
                frame_number=frame_id,
                frame_width=1280,
                frame_height=720,
                buf_pts=frame_id * 1000,
                object_items=[],
            )

        processor.handle_servicemaker_frame(empty_frame(20))
        now[0] = 100.1
        processor.handle_servicemaker_frame(empty_frame(21))
        now[0] = 100.6
        processor.handle_servicemaker_frame(empty_frame(22))

        assert [(source, tracks) for source, tracks, _meta in published] == [
            (0, []),
            (0, []),
        ]
        assert [meta["frame_id"] for _source, _tracks, meta in published] == [
            20,
            22,
        ]
        assert processor.get_active_track_map(0) == {}
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(DS9_ROOT), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_ds9_processor_forces_same_count_tracker_replacement_publication() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace
        from noesis.pipelines import hooks
        from noesis.telemetry.bev import BevPublicationReceipt
        from noesis.telemetry.publishers import TrackingPublicationReceipt
        from noesis_core.runtime_publication import RuntimePublicationGate

        bev_frames = []

        class Renderer:
            def render_and_publish(self, **kwargs):
                bev_frames.append(int(kwargs["frame_id"]))
                return BevPublicationReceipt(
                    status="admitted",
                    camera_id=str(kwargs["camera_id"]),
                    source_id=int(kwargs["source_id"]),
                    frame_id=int(kwargs["frame_id"]),
                    observed_at_us=int(kwargs["observed_at_us"]),
                    tracking_publication_sequence=int(
                        kwargs["tracking_publication_sequence"]
                    ),
                    tracking_outbound_submission_id=int(
                        kwargs["tracking_outbound_submission_id"]
                    ),
                    outbound_submission_id=int(
                        kwargs["tracking_outbound_submission_id"]
                    ) + 1,
                )

            def record_input_failure(self, *_args, **_kwargs):
                raise AssertionError("unexpected BEV failure")

        class Calibration:
            def snapshot(self, _source_id, _camera_id):
                return SimpleNamespace(image_size=(1280, 720))

        processor = hooks._AnalyticsTelemetryProcessor(
            pipeline=SimpleNamespace(config={}, stable_id_mgr=None),
            tracking_pub=SimpleNamespace(),
            camera_labels={0: "kitchen"},
            sensor_id_map={},
            bev_renderer=Renderer(),
            bev_calibration=Calibration(),
            publication_gate=RuntimePublicationGate(),
        )
        processor._tracking_publish_interval_s = 10.0
        processor._bev_publish_interval_s = 10.0
        first = {"camera_id": "kitchen", "tracker_id": 7}
        initial = processor._tracking_lifecycle.update_frame(
            source_id=0,
            camera_id="kitchen",
            frame_id=1,
            observed_at_us=1_000_000,
            tracks=[first],
        )
        assert processor._publish_gate_due(
            processor._last_tracking_publish_ts_by_sensor,
            processor._last_tracking_count_by_sensor,
            sensor_id=0,
            now_ts=1.0,
            count=1,
            interval_s=10.0,
            counter_prefix="tracking",
            update=False,
            force=initial.tracker_keys_changed,
        )
        processor._tracking_lifecycle.mark_published(initial)
        processor._last_tracking_publish_ts_by_sensor[0] = 1.0
        processor._last_tracking_count_by_sensor[0] = 1
        processor._publish_bev(
            0,
            "kitchen",
            SimpleNamespace(frame_number=1, buf_pts=1_000),
            [],
            now_ts=1.0,
            track_count=1,
            paired_with_tracking=True,
            tracking_receipt=TrackingPublicationReceipt(
                source_id=0,
                frame_id=1,
                observed_at_us=1_000_000,
                tracking_publication_sequence=0,
                outbound_submission_id=1,
                outbound_message_count=1,
            ),
        )
        replacement = {"camera_id": "kitchen", "tracker_id": 8}
        changed = processor._tracking_lifecycle.update_frame(
            source_id=0,
            camera_id="kitchen",
            frame_id=2,
            observed_at_us=1_100_000,
            tracks=[replacement],
        )
        assert changed.tracker_keys_changed is True
        assert changed.tombstones[0]["tracker_id"] == 7
        assert processor._publish_gate_due(
            processor._last_tracking_publish_ts_by_sensor,
            processor._last_tracking_count_by_sensor,
            sensor_id=0,
            now_ts=1.1,
            count=1,
            interval_s=10.0,
            counter_prefix="tracking",
            update=False,
            force=changed.tracker_keys_changed,
        )
        processor._publish_bev(
            0,
            "kitchen",
            SimpleNamespace(frame_number=2, buf_pts=2_000),
            [],
            now_ts=1.1,
            track_count=1,
            paired_with_tracking=True,
            tracking_receipt=TrackingPublicationReceipt(
                source_id=0,
                frame_id=2,
                observed_at_us=1_100_000,
                tracking_publication_sequence=1,
                outbound_submission_id=3,
                outbound_message_count=1,
            ),
        )
        assert bev_frames == [1, 2]
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(DS9_ROOT), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_ds9_processor_uses_pair_safe_tracking_bev_cadence_above_15fps() -> None:
    script = textwrap.dedent(
        """
        import os
        from types import SimpleNamespace

        from noesis.pipelines import hooks
        from noesis.telemetry.bev import BevPublicationReceipt
        from noesis.telemetry.publishers import TrackingPublicationReceipt
        from noesis_core.runtime_publication import RuntimePublicationGate

        os.environ["NOESIS_REID_TEST_MODE"] = "1"
        os.environ["NOESIS_TRACKING_PUBLISH_MAX_HZ"] = "15"
        os.environ["NOESIS_BEV_PUBLISH_MAX_HZ"] = "12"
        events = []

        class Publisher:
            def __init__(self):
                self.sequence = 0

            def publish(self, _source_id, _tracks, *, frame_metadata=None):
                events.append(("tracking", int(frame_metadata["frame_id"])))
                receipt = TrackingPublicationReceipt(
                    source_id=int(_source_id),
                    frame_id=int(frame_metadata["frame_id"]),
                    observed_at_us=int(frame_metadata["observed_at_us"]),
                    tracking_publication_sequence=self.sequence,
                    outbound_submission_id=self.sequence * 2 + 1,
                    outbound_message_count=1,
                )
                self.sequence += 1
                return receipt

        class Renderer:
            def render_and_publish(self, **kwargs):
                events.append(("bev", int(kwargs["frame_id"])))
                return BevPublicationReceipt(
                    status="admitted",
                    camera_id=str(kwargs["camera_id"]),
                    source_id=int(kwargs["source_id"]),
                    frame_id=int(kwargs["frame_id"]),
                    observed_at_us=int(kwargs["observed_at_us"]),
                    tracking_publication_sequence=int(
                        kwargs["tracking_publication_sequence"]
                    ),
                    tracking_outbound_submission_id=int(
                        kwargs["tracking_outbound_submission_id"]
                    ),
                    outbound_submission_id=int(
                        kwargs["tracking_outbound_submission_id"]
                    ) + 1,
                )

            def record_input_failure(self, *_args, **_kwargs):
                raise AssertionError("paired BEV unexpectedly failed")

        class Calibration:
            def snapshot(self, _source_id, _camera_id):
                return SimpleNamespace(image_size=(1280, 720))

        processor = hooks._AnalyticsTelemetryProcessor(
            pipeline=SimpleNamespace(config={}, stable_id_mgr=None),
            tracking_pub=Publisher(),
            camera_labels={0: "living-room"},
            sensor_id_map={},
            bev_renderer=Renderer(),
            bev_calibration=Calibration(),
            publication_gate=RuntimePublicationGate(),
        )
        now = [100.0]
        hooks.time.time = lambda: now[0]

        def empty_frame(frame_id):
            return SimpleNamespace(
                source_id=0,
                frame_number=frame_id,
                frame_width=1280,
                frame_height=720,
                buf_pts=frame_id * 1000,
                object_items=[],
            )

        frame_times = {}
        for frame_id in range(13):
            now[0] = 100.0 + frame_id / 30.0
            frame_times[frame_id] = now[0]
            processor.handle_servicemaker_frame(empty_frame(frame_id))

        assert len(events) >= 8 and len(events) % 2 == 0
        assert all(
            events[index][0] == "tracking"
            and events[index + 1] == ("bev", events[index][1])
            for index in range(0, len(events), 2)
        )
        paired = [events[index][1] for index in range(0, len(events), 2)]
        assert all(
            frame_times[right] - frame_times[left] >= 1.0 / 12.0 - 1e-9
            for left, right in zip(paired, paired[1:])
        )

        os.environ.pop("NOESIS_REID_TEST_MODE")
        now[0] += 0.01
        processor.handle_servicemaker_frame(empty_frame(100))
        os.environ["NOESIS_REID_TEST_MODE"] = "1"
        now[0] += 0.01
        processor.handle_servicemaker_frame(empty_frame(101))
        assert events[-4:] == [
            ("tracking", 100),
            ("bev", 100),
            ("tracking", 101),
            ("bev", 101),
        ]
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(DS9_ROOT), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
