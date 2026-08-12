from __future__ import annotations

import importlib.util
import ast
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from noesis_core.runtime_world import create_runtime_capability_monitor, create_runtime_world_service


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_publishers():
    path = REPO_ROOT / "DS9" / "noesis" / "telemetry" / "publishers.py"
    spec = importlib.util.spec_from_file_location("ds9_tracking_publishers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publishers = _load_publishers()


@dataclass(frozen=True)
class _Snapshot:
    camera_id: str
    intrinsics: tuple[tuple[float, float, float], ...]
    image_size: tuple[int, int]


class _CalibrationProvider:
    def snapshot(self, _source_id: int, camera_id: str) -> _Snapshot:
        return _Snapshot(
            camera_id=camera_id,
            intrinsics=((1.0, 0.0, 2.0), (0.0, 1.0, 3.0), (0.0, 0.0, 1.0)),
            image_size=(100, 50),
        )


class _WebSocketRecorder:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.next_submission_id = 1

    def admit_broadcast_batch_sync(
        self,
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> Any:
        submission_id = self.next_submission_id
        self.next_submission_id += 1
        receipt = SimpleNamespace(
            submission_id=submission_id,
            message_count=len(messages),
        )

        class _Admission:
            def __init__(self, owner: _WebSocketRecorder) -> None:
                self.receipt = receipt
                self._owner = owner

            def commit_then_release(self, commit: Any) -> Any:
                result = commit()
                self._owner.messages.extend(messages)
                return result

        return _Admission(self)


class WorldSnapshotRuntimeTests(unittest.TestCase):
    def test_ds9_publisher_emits_canonical_snapshot_and_health_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "detector.engine"
            model_path.write_bytes(b"ds9-model-evidence")
            service = create_runtime_world_service(
                runtime="ds9",
                pipeline_config={"models": {"pgie": {"engine": str(model_path)}}},
                camera_labels={0: "kitchen"},
                calibration_provider=_CalibrationProvider(),
                repo_root=root,
                run_id="ds9-test-run",
                instance_id="ds9-test",
                software_revision="test-revision",
                journal_path=root / "world.sqlite3",
            )
            monitor = create_runtime_capability_monitor(service)
            websocket = _WebSocketRecorder()
            publisher = publishers.TrackingTelemetryPublisher(
                websocket,
                metadata_getter=lambda _source_id, _tracks: {
                    "camera_id": "kitchen",
                    "image_size": [100, 50],
                    "calibration_version": "test-calibration",
                },
                world_service=service,
                health_monitor=monitor,
            )
            observed_at_us = max(1, time.time_ns() // 1_000 - 10_000)
            publisher.publish(
                0,
                [
                    {
                        "camera_id": "kitchen",
                        "tracker_id": 7,
                        "frame_id": 2,
                        "observed_at_us": observed_at_us,
                        "capture_time_status": "estimated",
                        "media_pts_ns": 1234,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "image_size": [100, 50],
                        "stable_id": 1001,
                        "identity_kind": "visitor",
                        "visitor_generation": 3,
                        "world": [1.0, 0.0, 2.0],
                        "world_valid": True,
                        "world_frame": "backend_world_m",
                    }
                ],
            )
            service.close()

            self.assertEqual(
                [message["type"] for message in websocket.messages],
                ["tracking", "world_snapshot", "world_event"],
            )
            tracking = websocket.messages[0]
            self.assertEqual(tracking["observation_contract"], "noesis.observation.person")
            self.assertEqual(tracking["observations"][0]["media_pts_ns"], 1234)
            self.assertEqual(tracking["observations"][0]["model"]["role"], "tracking_model_manifest")
            entity_id = tracking["world_snapshot"]["entities"][0]["entity_id"]
            self.assertEqual(entity_id, "visitor:ds9-test-run:1001:g3")
            self.assertEqual(websocket.messages[2]["payload"]["event_type"], "appeared")

            health = monitor.snapshot(generated_at_us=max(1, time.time_ns() // 1_000))
            self.assertEqual({row.status.value for row in health.capabilities}, {"healthy"})

    def test_ds9_runtime_wires_shared_world_health_factory_and_router(self) -> None:
        runtime = (REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py").read_text(encoding="utf-8")
        self.assertIn("create_runtime_world_service(", runtime)
        self.assertIn('runtime="ds9"', runtime)
        self.assertIn("create_runtime_capability_monitor(world_service)", runtime)
        self.assertIn("health_api.register_capability_monitor_getter", runtime)
        self.assertIn("app.include_router(health_api.router)", runtime)
        self.assertIn("app.include_router(scene_api.router)", runtime)
        self.assertIn("world_service=world_service", runtime)
        self.assertIn("health_monitor=capability_monitor", runtime)

    def test_ds9_tracking_publisher_has_no_ds8_contract_drift(self) -> None:
        def tracking_class(path: Path) -> str:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            node = next(
                item
                for item in tree.body
                if isinstance(item, ast.ClassDef)
                and item.name == "TrackingTelemetryPublisher"
            )
            return ast.get_source_segment(source, node) or ""

        ds8 = tracking_class(REPO_ROOT / "noesis" / "telemetry" / "publishers.py")
        ds9 = tracking_class(
            REPO_ROOT / "DS9" / "noesis" / "telemetry" / "publishers.py"
        )
        self.assertEqual(ds9, ds8)

    def test_both_ds9_public_track_paths_apply_time_and_identity_contracts(self) -> None:
        hooks = (REPO_ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py").read_text(encoding="utf-8")
        self.assertEqual(hooks.count("temporal_contract = _frame_temporal_contract"), 2)
        self.assertEqual(hooks.count("identity_contract = _stable_identity_contract"), 2)
        self.assertGreaterEqual(hooks.count("**temporal_contract"), 2)
        self.assertGreaterEqual(hooks.count("**identity_contract"), 2)

    def test_ds8_v3dt_and_ds9_emit_the_same_zone_authority_contract(self) -> None:
        paths = (
            REPO_ROOT / "noesis" / "pipelines" / "hooks.py",
            REPO_ROOT / "noesis" / "pipelines" / "hooks_v3dt_reimpl.py",
            REPO_ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py",
        )
        for path in paths:
            hooks = path.read_text(encoding="utf-8")
            self.assertIn(
                "from noesis_core.analytics_zones import "
                "resolve_authoritative_analytics_zone",
                hooks,
            )
            self.assertEqual(
                hooks.count(
                    "return resolve_authoritative_analytics_zone(analytics_meta)"
                ),
                1,
                path,
            )
            self.assertNotIn("def _iter_roi_labels(", hooks)
            self.assertEqual(hooks.count('"zone_source": zone_source'), 4, path)
            self.assertEqual(
                hooks.count('"zone_authoritative": zone_authoritative'),
                4,
                path,
            )
            self.assertEqual(
                hooks.count('track["zone_source"] = "nvdsanalytics_roi"'),
                2,
                path,
            )
            self.assertEqual(
                hooks.count('track["zone_authoritative"] = True'),
                2,
                path,
            )
            self.assertEqual(
                hooks.count('zone_source = "camera_default" if zone else None'),
                2,
                path,
            )


if __name__ == "__main__":
    unittest.main()
