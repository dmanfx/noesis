from __future__ import annotations

from noesis.telemetry.latency_metrics import LatencyCollector, LatencyWindow


def test_latency_window_prunes_old_samples() -> None:
    w = LatencyWindow(window_sec=10.0)
    w.add(100.0, 10.0)
    w.add(105.0, 20.0)

    snap = w.snapshot(111.0)  # cutoff=101; first sample (t=100) pruned
    assert snap.count == 1
    assert snap.max == 20.0


def test_latency_window_percentiles_linear_interpolation_even_n() -> None:
    w = LatencyWindow(window_sec=100.0)
    w.add(0.0, 0.0)
    w.add(0.0, 10.0)

    snap = w.snapshot(0.0)
    assert snap.count == 2
    assert snap.p50 == 5.0
    assert snap.p95 == 9.5
    assert snap.max == 10.0


def test_latency_window_percentiles_linear_interpolation_odd_n() -> None:
    w = LatencyWindow(window_sec=100.0)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        w.add(0.0, v)

    snap = w.snapshot(0.0)
    assert snap.count == 5
    assert snap.p50 == 3.0
    assert snap.p95 == 4.8
    assert snap.max == 5.0


def test_latency_collector_disabled_when_env_var_unset(monkeypatch) -> None:
    monkeypatch.delenv("NVDS_ENABLE_LATENCY_MEASUREMENT", raising=False)
    c = LatencyCollector(window_sec=10.0)

    agg = c.snapshot_aggregate(now_sec=0.0)
    assert agg["enabled"] is False
    assert agg.get("reason") == "env_disabled"

    by_source = c.snapshot_by_source(now_sec=0.0)
    assert by_source == {}


def test_latency_collector_clear_is_safe(monkeypatch) -> None:
    monkeypatch.delenv("NVDS_ENABLE_LATENCY_MEASUREMENT", raising=False)
    # Exercise clear() even when disabled; should be a no-op and not throw.
    c = LatencyCollector(window_sec=10.0)
    c.clear()
