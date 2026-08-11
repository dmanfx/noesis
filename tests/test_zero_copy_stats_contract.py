from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from noesis import ds8_runtime
from noesis.pipelines import hooks
from noesis.server import boundary_metrics
from websocket_server import WebSocketServer


ROOT = Path(__file__).resolve().parents[1]


class _DummyStableIDManager:
    def get_sid_metrics(self) -> Dict[str, Any]:
        return {
            "stableid_backend_mode": "gpu",
            "stableid_gpu_match_p50_ms": 0.4,
            "stableid_gpu_match_p95_ms": 0.9,
            "stableid_gallery_size": 12,
        }


class _DummyPipeline:
    def __init__(self) -> None:
        self.prepared = True
        self.activated = True
        self.depth_enabled = True
        self.analytics_reload_count = 1
        self.errors: list[str] = []
        self.frame_size = (1920, 1080)
        self.latency_collector = None
        self.stable_id_mgr = _DummyStableIDManager()
        self.components: Dict[str, Any] = {
            "tiler": type("Tiler", (), {"config": {"width": 1920, "height": 1080, "columns": 1, "rows": 1}})()
        }

    def depth_fps(self) -> float:
        return 8.0


def test_ws_stats_contract_includes_zero_copy_fields() -> None:
    hooks.reset_core_path_instrumentation()
    boundary_metrics.reset_boundary_serialization_metrics()

    hooks._increment_core_counter("tensor_host_copies_total.mapanything", 3)  # type: ignore[attr-defined]
    hooks._increment_core_counter("tensor_host_copies_total.reid", 2)  # type: ignore[attr-defined]
    hooks._increment_core_counter("tensor_host_copies_total.pose", 4)  # type: ignore[attr-defined]
    hooks._increment_core_counter("tensor_boundary_copy_bytes_total.pose_meta", 256)  # type: ignore[attr-defined]
    hooks._increment_core_counter("tensor_boundary_copy_bytes_total.depth_store", 512)  # type: ignore[attr-defined]
    hooks._CORE_PATH_INSTRUMENTATION.record_stage_timing(  # type: ignore[attr-defined]
        metric="analytics.handle_frame_ds8",
        duration_ns=1234,
        item_count=2,
    )

    ws = WebSocketServer(stats_callback=None)
    ws._record_boundary_serialization_stage(0.6, 80, channel="ws", route="broadcast", message_type="stats", stage="total")
    ws._record_boundary_serialization_stage(1.2, 120, channel="ws", route="get_ma_depth", message_type="ma_depth_response", stage="total")

    boundary_metrics.record_boundary_stage(
        0.7,
        96,
        channel="rest",
        route="/api/v1/depth/refresh",
        message_type="DepthRefreshResponse",
        stage="total",
        include_budget=True,
    )

    callback = ds8_runtime._build_stats_callback(  # type: ignore[attr-defined]
        _DummyPipeline(),
        {0: "cam0"},
        ws_metrics_getter=ws.get_boundary_serialization_metrics,
        ws_metrics_resetter=ws.reset_boundary_serialization_metrics,
        runtime_state={"analytics_state_poisoned": "unit-test divergence"},
    )
    payload = callback()

    assert payload["stack"] == "ds8"
    assert isinstance(payload.get("timestamp"), (float, int))

    pipeline_payload = payload["pipeline"]
    assert pipeline_payload["zero_copy_core_enabled"] is True
    assert isinstance(int(pipeline_payload["zero_copy_violations"]), int)
    assert isinstance(pipeline_payload["stableid_backend_mode"], str)
    assert isinstance(pipeline_payload["stableid_gpu_match_p95_ms"], (float, int))
    assert isinstance(pipeline_payload["boundary_cpu_serialization_p99_ms"], (float, int))
    assert isinstance(pipeline_payload["boundary_cpu_serialization_ws_p99_ms"], (float, int))
    assert isinstance(pipeline_payload["boundary_cpu_serialization_rest_p99_ms"], (float, int))
    assert pipeline_payload["analytics_state_poisoned"] == "unit-test divergence"

    zero_copy_core = pipeline_payload["zero_copy_core"]
    assert isinstance(zero_copy_core, dict)
    counters = zero_copy_core["counters"]
    assert int(counters.get("tensor_host_copies_total.mapanything", 0)) == 3
    assert int(counters.get("tensor_host_copies_total.reid", 0)) == 2
    assert int(counters.get("tensor_host_copies_total.pose", 0)) == 4
    assert int(counters.get("tensor_boundary_copy_bytes_total.pose_meta", 0)) == 256
    assert int(counters.get("tensor_boundary_copy_bytes_total.depth_store", 0)) == 512
    stage_timings = zero_copy_core["stage_timings"]
    assert int(stage_timings["analytics.handle_frame_ds8"]["count"]) == 1
    assert int(stage_timings["analytics.handle_frame_ds8"]["last_items"]) == 2

    boundary_block = zero_copy_core["boundary_serialization_metrics"]
    assert isinstance(boundary_block.get("ws"), dict)
    assert isinstance(boundary_block.get("rest"), dict)

    callback.clear_stats()  # type: ignore[attr-defined]
    assert ws.get_boundary_serialization_metrics()["count"] == 0
    assert boundary_metrics.get_boundary_serialization_metrics()["count"] == 0


def test_all_zero_copy_live_gates_use_fail_closed_boundary_tracker() -> None:
    gate_paths = (
        ROOT / "scripts" / "zero_copy_smoke_test.py",
        ROOT / "scripts" / "zero_copy_stats_smoke_test.py",
        ROOT / "DS9" / "scripts" / "zero_copy_smoke_test.py",
    )
    for path in gate_paths:
        source = path.read_text(encoding="utf-8")
        assert "BoundaryGateTracker" in source
        assert "boundary_gate.observe(pipe)" in source
        assert "boundary_gate.failure(" in source

    ds9_smoke = gate_paths[-1].read_text(encoding="utf-8")
    assert "add_auth_token_file_argument(parser)" in ds9_smoke
    assert "connect_required_websocket" in ds9_smoke
    assert "build_required_auth_request" in ds9_smoke
    assert "configure_required_auth_environment" in ds9_smoke
