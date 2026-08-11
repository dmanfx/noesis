from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_adapter():
    path = REPO_ROOT / "DS9" / "noesis" / "telemetry" / "world_contract_adapter.py"
    spec = importlib.util.spec_from_file_location("ds9_world_contract_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


class WorldContractAdapterTests(unittest.TestCase):
    def test_frame_timing_keeps_media_pts_relative_and_nonnegative(self) -> None:
        fields = adapter.frame_temporal_contract(
            SimpleNamespace(buf_pts=-50),
            observed_at_s=1_700_000_000.25,
        )
        self.assertEqual(fields["observed_at_us"], 1_700_000_000_250_000)
        self.assertEqual(fields["capture_time_status"], "estimated")
        self.assertEqual(fields["media_pts_ns"], 0)

    def test_visitor_generation_comes_from_shared_diagnostics(self) -> None:
        class Manager:
            def get_track_diagnostics(self, sensor_id: int, tracker_id: int):
                self.request = (sensor_id, tracker_id)
                return {
                    "identity_kind": "visitor",
                    "identity_state": "recognized_visitor",
                    "visitor_generation": 4,
                }

        manager = Manager()
        fields = adapter.stable_identity_contract(manager, sensor_id=2, tracker_id=17)
        self.assertEqual(manager.request, (2, 17))
        self.assertEqual(fields["identity_kind"], "visitor")
        self.assertEqual(fields["visitor_generation"], 4)

    def test_negative_visitor_generation_is_not_published(self) -> None:
        fields = adapter.stable_identity_contract(
            None,
            sensor_id=0,
            tracker_id=1,
            diagnostics={"identity_kind": "visitor", "visitor_generation": -1},
        )
        self.assertNotIn("visitor_generation", fields)


if __name__ == "__main__":
    unittest.main()
