#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis import ds8_runtime
from noesis.pipelines import hooks
from websocket_server import WebSocketServer


class _DummyPipeline:
    def __init__(self) -> None:
        self.prepared = True
        self.activated = True
        self.depth_enabled = False
        self.analytics_reload_count = 0
        self.errors = []
        self.frame_size = (1920, 1080)
        self.latency_collector = None
        self.components = {
            "tiler": type("Tiler", (), {"config": {"width": 1920, "height": 1080, "columns": 1, "rows": 1}})()
        }

    def depth_fps(self) -> float:
        return 0.0


def main() -> int:
    hooks.reset_core_path_instrumentation()
    ws = WebSocketServer(stats_callback=None)

    # Simulate one boundary JSON serialization sample.
    ws._record_boundary_serialization(1.2, 128)  # type: ignore[attr-defined]
    ws_metrics = ws.get_boundary_serialization_metrics()
    if ws_metrics.get("count", 0) < 1:
        print("[FAIL] boundary serialization metrics did not record")
        return 1

    # Simulate one core-path violation counter sample.
    hooks._CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(  # type: ignore[attr-defined]
        location="smoke.test",
        reason="synthetic",
    )
    callback = ds8_runtime._build_stats_callback(  # type: ignore[attr-defined]
        _DummyPipeline(),
        {0: "cam0"},
        ws_metrics_getter=ws.get_boundary_serialization_metrics,
        ws_metrics_resetter=ws.reset_boundary_serialization_metrics,
    )
    payload = callback()
    pipeline_payload = payload.get("pipeline", {})
    required = (
        "zero_copy_core_enabled",
        "zero_copy_violations",
        "boundary_cpu_serialization_p99_ms",
        "zero_copy_core",
    )
    missing = [k for k in required if k not in pipeline_payload]
    if missing:
        print(f"[FAIL] missing zero-copy stats keys: {missing}")
        return 1

    if not pipeline_payload.get("zero_copy_core_enabled", False):
        print("[FAIL] zero_copy_core_enabled is false")
        return 1

    print(
        "[PASS] zero-copy stats present",
        f"violations={pipeline_payload.get('zero_copy_violations')}",
        f"p99_ms={pipeline_payload.get('boundary_cpu_serialization_p99_ms')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
