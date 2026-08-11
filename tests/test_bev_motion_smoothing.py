from __future__ import annotations

import math

from noesis.telemetry.motion_smoothing import MotionGatedAlphaBetaSmoother, MotionSmoothingConfig


def test_smoother_clamps_large_jumps() -> None:
    cfg = MotionSmoothingConfig(
        enabled=True,
        max_speed_mps=2.0,
        max_jump_m=0.5,
        alpha=1.0,  # simplify: output equals clamped measurement
        beta=0.0,
        ttl_s=10.0,
        reset_after_s=10.0,
    )
    smoother = MotionGatedAlphaBetaSmoother(cfg)

    key = ("cam0", 123)
    x0, z0 = smoother.update(key, ts_s=0.0, meas_x=0.0, meas_z=0.0)
    assert (x0, z0) == (0.0, 0.0)

    # dt=0.1s => allowed move = 2.0*0.1 + 0.5 = 0.7m
    x1, z1 = smoother.update(key, ts_s=0.1, meas_x=10.0, meas_z=0.0)
    assert math.isclose(z1, 0.0, abs_tol=1e-6)
    assert 0.0 <= x1 <= 0.7 + 1e-6
    assert math.isclose(x1, 0.7, abs_tol=1e-6)


def test_smoother_resets_after_long_gaps() -> None:
    cfg = MotionSmoothingConfig(
        enabled=True,
        max_speed_mps=2.0,
        max_jump_m=0.5,
        alpha=0.5,
        beta=0.1,
        ttl_s=10.0,
        reset_after_s=0.5,
    )
    smoother = MotionGatedAlphaBetaSmoother(cfg)
    key = ("cam0", 1)

    smoother.update(key, ts_s=0.0, meas_x=0.0, meas_z=0.0)
    x1, z1 = smoother.update(key, ts_s=1.0, meas_x=5.0, meas_z=6.0)  # dt > reset_after_s
    assert (x1, z1) == (5.0, 6.0)


def test_smoother_prunes_state_by_ttl() -> None:
    cfg = MotionSmoothingConfig(enabled=True, ttl_s=0.25, reset_after_s=10.0)
    smoother = MotionGatedAlphaBetaSmoother(cfg)
    key = ("cam0", 42)

    smoother.update(key, ts_s=0.0, meas_x=1.0, meas_z=2.0)
    smoother.prune(now_s=0.3)

    # State was pruned, so this behaves like a cold start.
    x, z = smoother.update(key, ts_s=0.31, meas_x=9.0, meas_z=8.0)
    assert (x, z) == (9.0, 8.0)
