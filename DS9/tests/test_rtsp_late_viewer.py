from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MosaicLateViewerTests(unittest.TestCase):
    def test_ds9_uses_shm_au_transport_and_allocates_webrtc_slots_on_demand(self) -> None:
        pipeline_source = (
            ROOT / "DS9" / "noesis" / "pipelines" / "deepstream_pipeline.py"
        ).read_text(encoding="utf-8")
        runtime_source = (
            ROOT / "DS9" / "noesis" / "ds9_runtime_core.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sink_tee.downstream.append(mosaic_queue.name)", pipeline_source)
        self.assertNotIn("NOESIS_MOSAIC_RTSP_DEMAND_GATED", pipeline_source)
        self.assertIn("MosaicH264ShmFeeder", runtime_source)
        self.assertIn("h264_feeder=mosaic_h264_feeder", runtime_source)
        self.assertIn("register_webrtc_gateway_factory", runtime_source)
        self.assertIn("NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS", runtime_source)


if __name__ == "__main__":
    unittest.main()
