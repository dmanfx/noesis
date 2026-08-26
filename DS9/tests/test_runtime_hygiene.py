from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeHygieneTests(unittest.TestCase):
    def test_all_active_runtimes_have_no_agent_debug_sink_or_machine_path(self) -> None:
        retired_paths = (
            REPO_ROOT / "noesis" / "ds8_runtime.py",
            REPO_ROOT / "noesis" / "ds8_runtime_v3dt_reimpl.py",
        )
        for path in retired_paths:
            self.assertFalse(path.exists(), f"retired runtime was restored: {path}")

        paths = (
            REPO_ROOT / "noesis" / "mosaic_glib_context.py",
            REPO_ROOT / "noesis" / "mosaic_h264_bridge.py",
            REPO_ROOT / "noesis" / "mosaic_webrtc_gateway.py",
            REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",
        )
        for forbidden in (
            "_agent_debug_log_path",
            "NOESIS_AGENT_DEBUG_LOG",
            "debug-session",
            "#region agent log",
            ".cursor/debug.log",
            "/home/mayor",
        ):
            for path in paths:
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(forbidden, source, f"{forbidden!r} found in {path}")


if __name__ == "__main__":
    unittest.main()
