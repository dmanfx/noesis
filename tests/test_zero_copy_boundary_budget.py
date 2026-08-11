from __future__ import annotations

import pytest

from noesis.server import boundary_metrics
from scripts.zero_copy_boundary_diagnostics import (
    BoundaryGateTracker,
    extract_boundary_diagnostics,
)
from websocket_server import WebSocketServer


def _expected_p99(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    idx = int(round(0.99 * (len(ordered) - 1)))
    return float(ordered[idx])


def test_ws_boundary_percentiles_are_deterministic_and_ms_units() -> None:
    ws = WebSocketServer(stats_callback=None)
    samples_ms = [0.25, 0.75, 1.5, 2.0, 3.25]

    for val in samples_ms:
        ws._record_boundary_serialization_stage(
            val,
            64,
            channel="ws",
            route="broadcast",
            message_type="stats",
            stage="total",
            include_budget=True,
        )

    # provider_wait is tracked for observability but excluded from budget totals.
    ws._record_boundary_serialization_stage(
        9.5,
        0,
        channel="ws",
        route="get_ma_depth",
        message_type="ma_depth_response",
        stage="provider_wait",
        include_budget=False,
    )

    metrics = ws.get_boundary_serialization_metrics()
    assert int(metrics.get("count", 0)) == len(samples_ms)
    assert float(metrics["p99_ms"]) == _expected_p99(samples_ms)
    assert float(metrics["p50_ms"]) == 1.5
    assert int(metrics.get("violations", 0)) == 1  # only 3.25ms exceeds 3.0ms budget

    stage_key = "ws|get_ma_depth|ma_depth_response|provider_wait|ok"
    stages = metrics.get("stages", {})
    assert stage_key in stages
    assert int(stages[stage_key]["count"]) == 1


def test_rest_boundary_percentiles_are_deterministic_and_ms_units() -> None:
    boundary_metrics.reset_boundary_serialization_metrics()
    samples_ms = [0.4, 0.8, 1.2, 2.4, 2.8]
    for val in samples_ms:
        boundary_metrics.record_boundary_stage(
            val,
            96,
            channel="rest",
            route="/api/v1/depth/refresh",
            message_type="DepthRefreshResponse",
            stage="total",
            include_budget=True,
        )

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert int(metrics.get("count", 0)) == len(samples_ms)
    assert float(metrics["p99_ms"]) == _expected_p99(samples_ms)
    assert float(metrics["p95_ms"]) == _expected_p99(samples_ms)
    assert int(metrics.get("violations", 0)) == 0

    boundary_metrics.reset_boundary_serialization_metrics()


def test_rest_boundary_uses_true_rolling_windows_and_max_path_p99(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_s = [100.0]
    monkeypatch.setattr(boundary_metrics, "_CLOCK", lambda: now_s[0])
    boundary_metrics.reset_boundary_serialization_metrics()

    # A single sparse slow path is deliberately diluted below aggregate p99.
    for _ in range(200):
        boundary_metrics.record_boundary_stage(
            0.2,
            32,
            channel="rest",
            route="/fast",
            message_type="FastResponse",
            stage="total",
            include_budget=True,
        )
    boundary_metrics.record_boundary_stage(
        10.0,
        64,
        channel="rest",
        route="/sparse-slow",
        message_type="SlowResponse",
        stage="total",
        include_budget=True,
    )
    boundary_metrics.record_boundary_stage(
        99.0,
        0,
        channel="rest",
        route="/provider",
        message_type="ProviderWait",
        stage="provider_wait",
        include_budget=False,
    )
    # Even the same total-stage key is ignored when that observation is
    # explicitly outside the boundary budget.
    boundary_metrics.record_boundary_stage(
        99.0,
        0,
        channel="rest",
        route="/sparse-slow",
        message_type="SlowResponse",
        stage="total",
        include_budget=False,
    )

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["p99_10s_ms"] == 0.2
    assert metrics["p99_60s_ms"] == 0.2
    assert metrics["max_path_p99_10s_ms"] == 10.0
    assert metrics["max_path_p99_60s_ms"] == 10.0
    assert metrics["max_path_p99_ms"] == 10.0
    assert metrics["windows"]["10s"]["count"] == 201
    assert metrics["windows"]["60s"]["count"] == 201

    # After eleven seconds the old burst is absent from 10s but remains in 60s.
    now_s[0] = 111.0
    boundary_metrics.record_boundary_stage(
        0.3,
        48,
        channel="rest",
        route="/fast",
        message_type="FastResponse",
        stage="total",
        include_budget=True,
    )
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["p99_10s_ms"] == 0.3
    assert metrics["max_path_p99_10s_ms"] == 0.3
    assert metrics["max_path_p99_60s_ms"] == 10.0
    assert metrics["windows"]["10s"]["count"] == 1
    assert metrics["windows"]["60s"]["count"] == 202

    # Once the 60s horizon passes, only the newer sample survives.
    now_s[0] = 161.0
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["windows"]["60s"]["count"] == 1
    assert metrics["max_path_p99_60s_ms"] == 0.3
    assert metrics["lifetime_count"] == 202

    now_s[0] = 172.0
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["windows"]["10s"]["count"] == 0
    assert metrics["windows"]["60s"]["count"] == 0
    assert metrics["max_path_p99_ms"] is None
    assert metrics["lifetime_count"] == 202
    boundary_metrics.reset_boundary_serialization_metrics()


def test_rest_boundary_overflow_fails_safe_on_sparse_slow_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boundary_metrics, "_CLOCK", lambda: 500.0)
    boundary_metrics.reset_boundary_serialization_metrics()

    # Fill every named slot, then pool enough fast overflow paths to hide one
    # slow path from overflow p99. Overflow max must remain admission authority.
    for index in range(511):
        boundary_metrics.record_boundary_stage(
            0.2,
            16,
            route=f"/named-{index}",
            message_type="Response",
            stage="total",
            include_budget=True,
        )
    for index in range(200):
        boundary_metrics.record_boundary_stage(
            0.2,
            16,
            route=f"/overflow-fast-{index}",
            message_type="Response",
            stage="total",
            include_budget=True,
        )
    boundary_metrics.record_boundary_stage(
        10.0,
        16,
        route="/overflow-slow",
        message_type="Response",
        stage="total",
        include_budget=True,
    )

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    overflow_key = "rest|__overflow__|__overflow__|__overflow__|__overflow__"
    assert metrics["p99_ms"] == 0.2
    assert metrics["stages"][overflow_key]["p99_ms"] == 0.2
    assert metrics["stages"][overflow_key]["max_ms"] == 10.0
    assert metrics["max_path_p99_10s_ms"] == 10.0
    assert metrics["max_path_p99_60s_ms"] == 10.0
    assert metrics["detail_limit"] == 512
    assert metrics["detail_truncated"] is True
    assert len(metrics["routes"]) == 512
    assert len(metrics["stages"]) == 512
    compact = boundary_metrics.get_boundary_serialization_metrics_compact()
    assert compact["budget_path_count"] == 512
    assert compact["top_budget_paths_limit"] == 8
    assert compact["top_budget_paths_truncated"] is True
    assert len(compact["top_budget_paths"]) == 8
    assert compact["top_budget_paths"][0]["key"] == overflow_key
    assert compact["top_budget_paths"][0]["authority_60s_ms"] == 10.0
    assert "routes" not in compact
    assert "stages" not in compact
    boundary_metrics.reset_boundary_serialization_metrics()


def test_rest_boundary_sample_saturation_is_bounded_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_s = [100.0]
    monkeypatch.setattr(boundary_metrics, "_CLOCK", lambda: now_s[0])
    monkeypatch.setattr(boundary_metrics, "_MAX_WINDOW_SAMPLES", 4)
    boundary_metrics.reset_boundary_serialization_metrics()

    for _ in range(4):
        boundary_metrics.record_boundary_stage(
            0.2,
            8,
            route="/burst",
            message_type="Response",
            stage="total",
            include_budget=True,
        )
    boundary_metrics.record_boundary_stage(
        10.0,
        8,
        route="/burst",
        message_type="Response",
        stage="total",
        include_budget=True,
    )
    boundary_metrics.record_boundary_stage(
        0.2,
        8,
        route="/burst",
        message_type="Response",
        stage="total",
        include_budget=True,
    )

    metrics = boundary_metrics.get_boundary_serialization_metrics()
    stage_key = "rest|/burst|Response|total|ok"
    assert metrics["sample_limit"] == 4
    assert metrics["count"] == 4
    assert metrics["lifetime_count"] == 6
    assert metrics["sample_dropped_total"] == 2
    assert metrics["sample_saturated_10s"] is True
    assert metrics["sample_saturated_60s"] is True
    assert metrics["path_sample_saturated_10s"] is True
    assert metrics["path_sample_saturated_60s"] is True
    assert metrics["max_path_p99_10s_ms"] == 10.0
    assert metrics["max_path_p99_60s_ms"] == 10.0
    assert metrics["stages"][stage_key]["count"] == 4
    assert metrics["stages"][stage_key]["lifetime_count"] == 6
    assert metrics["stages"][stage_key]["sample_dropped_total"] == 2

    now_s[0] = 111.0
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["sample_saturated_10s"] is False
    assert metrics["sample_saturated_60s"] is True
    assert metrics["max_path_p99_10s_ms"] is None
    assert metrics["max_path_p99_60s_ms"] == 10.0

    now_s[0] = 161.0
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert metrics["sample_saturated_60s"] is False
    assert metrics["path_sample_saturated_60s"] is False
    assert metrics["max_path_p99_ms"] is None
    assert metrics["sample_dropped_total"] == 2
    boundary_metrics.reset_boundary_serialization_metrics()


def test_rest_compact_getter_skips_route_and_nonbudget_stage_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boundary_metrics, "_CLOCK", lambda: 700.0)
    boundary_metrics.reset_boundary_serialization_metrics()
    boundary_metrics.record_boundary_stage(
        0.5,
        32,
        route="/budgeted",
        message_type="Response",
        stage="total",
        include_budget=True,
    )
    for index in range(40):
        boundary_metrics.record_boundary_stage(
            0.1,
            0,
            route=f"/diagnostic-{index}",
            message_type="Response",
            stage="response_model",
            include_budget=False,
        )

    summary_calls = 0
    original = boundary_metrics._window_summaries

    def counted(*args, **kwargs):
        nonlocal summary_calls
        summary_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary_metrics, "_window_summaries", counted)
    compact = boundary_metrics.get_boundary_serialization_metrics_compact()

    assert summary_calls == 2  # aggregate plus the one budgeted path
    assert "routes" not in compact
    assert "stages" not in compact
    assert compact["max_path_p99_ms"] == 0.5
    assert compact["budget_path_count"] == 1
    assert compact["top_budget_paths"] == [
        {
            "key": "rest|/budgeted|Response|total|ok",
            "p99_10s_ms": 0.5,
            "p99_60s_ms": 0.5,
            "max_10s_ms": 0.5,
            "max_60s_ms": 0.5,
            "authority_10s_ms": 0.5,
            "authority_60s_ms": 0.5,
            "sample_saturated_10s": False,
            "sample_saturated_60s": False,
        }
    ]
    boundary_metrics.reset_boundary_serialization_metrics()


def test_boundary_failure_diagnostics_name_only_budgeted_total_stage() -> None:
    ws = WebSocketServer(stats_callback=None)
    ws._record_boundary_serialization_stage(
        12.0,
        0,
        channel="ws",
        route="get_floorplan",
        message_type="floorplan_response",
        stage="worker_dispatch_wait",
        include_budget=False,
    )
    ws._record_boundary_serialization_stage(
        4.25,
        128,
        channel="ws",
        route="broadcast",
        message_type="world_snapshot",
        stage="total",
        include_budget=True,
    )
    ws_metrics = ws.get_boundary_serialization_metrics()
    diagnostics = extract_boundary_diagnostics(
        {
            "boundary_cpu_serialization_p99_ms": 4.25,
            "zero_copy_core": {
                "boundary_serialization_metrics": {
                    "ws": ws_metrics,
                    "rest": {},
                }
            },
        },
        allowed_p99_ms=3.0,
    )

    assert diagnostics["combined_p99_ms"] == 4.25
    assert diagnostics["privacy"]["payload_content"] == "absent"
    assert diagnostics["offending_budget_total_stages"] == [
        {
            "channel": "ws",
            "key": "ws|broadcast|world_snapshot|total|ok",
            "count": 1,
            "avg_ms": 4.25,
            "p50_ms": 4.25,
            "p95_ms": 4.25,
            "p99_ms": 4.25,
            "max_ms": 4.25,
            "last_ms": 4.25,
            "total_bytes": 128,
            "last_payload_bytes": 128,
            "budget_ms": None,
            "violations": 1,
        }
    ]
    stage_keys = {
        item["key"] for item in diagnostics["channels"]["ws"]["budget_total_stages"]
    }
    assert (
        "ws|get_floorplan|floorplan_response|worker_dispatch_wait|ok" not in stage_keys
    )


def test_boundary_gate_tracker_fails_closed_on_missing_or_invalid_truth() -> None:
    missing_p99 = BoundaryGateTracker()
    missing_p99.observe({"boundary_serialization_errors_total": 0})
    assert missing_p99.failure(allowed_p99_ms=3.0) == "boundary_p99_missing"
    assert missing_p99.evidence()["boundary_p99_missing_samples"] == 1

    missing_errors = BoundaryGateTracker()
    missing_errors.observe({"boundary_cpu_serialization_p99_ms": 0.5})
    assert (
        missing_errors.failure(allowed_p99_ms=3.0)
        == "boundary_error_counter_missing"
    )
    assert missing_errors.evidence()["boundary_error_counter_missing_samples"] == 1

    invalid = BoundaryGateTracker()
    invalid.observe(
        {
            "boundary_cpu_serialization_p99_ms": float("nan"),
            "boundary_serialization_errors_total": -1,
        }
    )
    assert invalid.failure(allowed_p99_ms=3.0) == "boundary_p99_missing"


def test_boundary_gate_tracker_rejects_error_growth_and_nonzero_baseline() -> None:
    growth = BoundaryGateTracker()
    growth.observe(
        {
            "boundary_cpu_serialization_p99_ms": 0.5,
            "boundary_serialization_errors_total": 0,
        }
    )
    growth.observe(
        {
            "boundary_cpu_serialization_p99_ms": 0.7,
            "boundary_serialization_errors_total": 2,
        }
    )
    assert (
        growth.failure(allowed_p99_ms=3.0)
        == "boundary_serialization_errors_grew"
    )
    assert growth.evidence()["first_boundary_errors_total"] == 0
    assert growth.evidence()["final_boundary_errors_total"] == 2
    assert growth.evidence()["max_boundary_errors_total"] == 2

    nonzero = BoundaryGateTracker()
    for _ in range(2):
        nonzero.observe(
            {
                "boundary_cpu_serialization_p99_ms": 0.5,
                "boundary_serialization_errors_total": 1,
            }
        )
    assert (
        nonzero.failure(allowed_p99_ms=3.0)
        == "boundary_serialization_errors_present"
    )


def test_boundary_gate_tracker_enforces_p99_only_after_complete_clean_evidence() -> None:
    clean = BoundaryGateTracker()
    changed = clean.observe(
        {
            "boundary_cpu_serialization_p99_ms": 2.5,
            "boundary_serialization_errors_total": 0,
        }
    )
    assert changed is True
    assert clean.failure(allowed_p99_ms=3.0) is None

    changed = clean.observe(
        {
            "boundary_cpu_serialization_p99_ms": 3.5,
            "boundary_serialization_errors_total": 0,
        }
    )
    assert changed is True
    assert clean.failure(allowed_p99_ms=3.0) == "boundary_p99_exceeded"
    assert clean.evidence()["max_boundary_p99_ms"] == 3.5
