from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Tuple


@dataclass(frozen=True)
class MotionSmoothingConfig:
    enabled: bool = True
    max_speed_mps: float = 4.0
    max_jump_m: float = 0.75
    alpha: float = 0.45
    beta: float = 0.10
    ttl_s: float = 2.5
    reset_after_s: float = 1.25
    min_dt_s: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "max_speed_mps", max(0.0, float(self.max_speed_mps)))
        object.__setattr__(self, "max_jump_m", max(0.0, float(self.max_jump_m)))
        object.__setattr__(self, "alpha", float(min(1.0, max(0.0, float(self.alpha)))))
        object.__setattr__(self, "beta", float(min(1.0, max(0.0, float(self.beta)))))
        object.__setattr__(self, "ttl_s", max(0.0, float(self.ttl_s)))
        object.__setattr__(self, "reset_after_s", max(0.0, float(self.reset_after_s)))
        object.__setattr__(self, "min_dt_s", max(0.0, float(self.min_dt_s)))

    @classmethod
    def from_mapping(cls, cfg: Any) -> "MotionSmoothingConfig":
        if not isinstance(cfg, dict):
            return cls()

        def _bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off"):
                return False
            return default

        def _float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        return cls(
            enabled=_bool(cfg.get("enabled"), True),
            max_speed_mps=_float(cfg.get("max_speed_mps"), 4.0),
            max_jump_m=_float(cfg.get("max_jump_m"), 0.75),
            alpha=_float(cfg.get("alpha"), 0.45),
            beta=_float(cfg.get("beta"), 0.10),
            ttl_s=_float(cfg.get("ttl_s"), 2.5),
            reset_after_s=_float(cfg.get("reset_after_s"), 1.25),
            min_dt_s=_float(cfg.get("min_dt_s"), 1e-3),
        )


@dataclass
class _TrackState:
    x: float
    z: float
    vx: float
    vz: float
    ts: float
    last_seen_ts: float


class MotionGatedAlphaBetaSmoother:
    """Constant-velocity alpha-beta tracker with a motion gate that clamps outliers.

    Intended for physical tracks (meters) where large instantaneous jumps are implausible.
    """

    def __init__(self, config: MotionSmoothingConfig) -> None:
        self.config = config
        self._states: Dict[Hashable, _TrackState] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def prune(self, now_s: float) -> None:
        if not self.config.ttl_s:
            return
        ttl = float(self.config.ttl_s)
        expired = [k for k, st in self._states.items() if (float(now_s) - float(st.last_seen_ts)) > ttl]
        for k in expired:
            self._states.pop(k, None)

    def reset(self, key: Hashable) -> None:
        self._states.pop(key, None)

    def update(self, key: Hashable, ts_s: float, meas_x: float, meas_z: float) -> Tuple[float, float]:
        if not self.enabled:
            return float(meas_x), float(meas_z)

        ts_s = float(ts_s)
        meas_x = float(meas_x)
        meas_z = float(meas_z)

        state = self._states.get(key)
        if state is None:
            self._states[key] = _TrackState(
                x=meas_x,
                z=meas_z,
                vx=0.0,
                vz=0.0,
                ts=ts_s,
                last_seen_ts=ts_s,
            )
            return meas_x, meas_z

        dt = ts_s - float(state.ts)
        if dt <= 0.0:
            state.last_seen_ts = ts_s
            return float(state.x), float(state.z)

        reset_after = float(self.config.reset_after_s)
        if reset_after and dt > reset_after:
            self._states[key] = _TrackState(
                x=meas_x,
                z=meas_z,
                vx=0.0,
                vz=0.0,
                ts=ts_s,
                last_seen_ts=ts_s,
            )
            return meas_x, meas_z

        pred_x = float(state.x) + float(state.vx) * dt
        pred_z = float(state.z) + float(state.vz) * dt

        err_x = meas_x - pred_x
        err_z = meas_z - pred_z
        dist = (err_x * err_x + err_z * err_z) ** 0.5

        allowed = float(self.config.max_jump_m) + float(self.config.max_speed_mps) * dt
        if allowed > 0.0 and dist > allowed:
            scale = allowed / dist
            meas_x = pred_x + err_x * scale
            meas_z = pred_z + err_z * scale
            err_x = meas_x - pred_x
            err_z = meas_z - pred_z

        alpha = float(self.config.alpha)
        beta = float(self.config.beta)

        x = pred_x + alpha * err_x
        z = pred_z + alpha * err_z

        vx = float(state.vx)
        vz = float(state.vz)
        if dt >= float(self.config.min_dt_s) and dt > 0.0:
            vx = vx + (beta / dt) * err_x
            vz = vz + (beta / dt) * err_z

        self._states[key] = _TrackState(x=x, z=z, vx=vx, vz=vz, ts=ts_s, last_seen_ts=ts_s)
        return float(x), float(z)


__all__ = [
    "MotionSmoothingConfig",
    "MotionGatedAlphaBetaSmoother",
]
