from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


_SUMMARY_FIELDS = (
    "count",
    "avg_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "last_ms",
    "total_bytes",
    "last_payload_bytes",
    "budget_ms",
    "violations",
)
_MAX_COMPONENTS = 16


@dataclass
class BoundaryGateTracker:
    """Accumulate fail-closed boundary evidence across periodic stats samples."""

    samples: int = 0
    p99_samples: int = 0
    error_counter_samples: int = 0
    max_boundary_p99_ms: float | None = None
    first_boundary_errors_total: int | None = None
    final_boundary_errors_total: int | None = None
    max_boundary_errors_total: int | None = None

    def observe(self, pipeline_payload: Mapping[str, Any]) -> bool:
        """Observe one stats payload and report whether the maximum p99 changed."""

        self.samples += 1
        previous_max = self.max_boundary_p99_ms
        p99 = _finite_number(
            pipeline_payload.get("boundary_cpu_serialization_p99_ms")
        )
        if p99 is not None and float(p99) >= 0.0:
            numeric_p99 = float(p99)
            self.p99_samples += 1
            if (
                self.max_boundary_p99_ms is None
                or numeric_p99 > self.max_boundary_p99_ms
            ):
                self.max_boundary_p99_ms = numeric_p99

        raw_errors = _finite_number(
            pipeline_payload.get("boundary_serialization_errors_total")
        )
        if raw_errors is not None:
            numeric_errors = float(raw_errors)
            if numeric_errors >= 0.0 and numeric_errors.is_integer():
                error_total = int(numeric_errors)
                self.error_counter_samples += 1
                if self.first_boundary_errors_total is None:
                    self.first_boundary_errors_total = error_total
                self.final_boundary_errors_total = error_total
                if (
                    self.max_boundary_errors_total is None
                    or error_total > self.max_boundary_errors_total
                ):
                    self.max_boundary_errors_total = error_total

        return self.max_boundary_p99_ms != previous_max

    def failure(self, *, allowed_p99_ms: float) -> str | None:
        """Return the first authoritative boundary-gate failure, if any."""

        if self.p99_samples != self.samples:
            return "boundary_p99_missing"
        if self.error_counter_samples != self.samples:
            return "boundary_error_counter_missing"
        first_errors = int(self.first_boundary_errors_total or 0)
        final_errors = int(self.final_boundary_errors_total or 0)
        max_errors = int(self.max_boundary_errors_total or 0)
        if final_errors > first_errors:
            return "boundary_serialization_errors_grew"
        if max_errors > 0:
            return "boundary_serialization_errors_present"
        if (
            self.max_boundary_p99_ms is not None
            and self.max_boundary_p99_ms > float(allowed_p99_ms)
        ):
            return "boundary_p99_exceeded"
        return None

    def evidence(self) -> dict[str, int | float | None]:
        return {
            "max_boundary_p99_ms": self.max_boundary_p99_ms,
            "boundary_p99_samples": int(self.p99_samples),
            "boundary_p99_missing_samples": int(self.samples - self.p99_samples),
            "boundary_error_counter_samples": int(self.error_counter_samples),
            "boundary_error_counter_missing_samples": int(
                self.samples - self.error_counter_samples
            ),
            "first_boundary_errors_total": self.first_boundary_errors_total,
            "final_boundary_errors_total": self.final_boundary_errors_total,
            "max_boundary_errors_total": self.max_boundary_errors_total,
        }


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return int(value) if isinstance(value, int) else numeric


def _summary(value: Any) -> dict[str, int | float | None]:
    source = value if isinstance(value, Mapping) else {}
    return {field: _finite_number(source.get(field)) for field in _SUMMARY_FIELDS}


def _safe_component_key(value: Any) -> str:
    return " ".join(str(value).split())[:256]


def _rank_total_stages(
    value: Any,
    *,
    allowed_p99_ms: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stages = value if isinstance(value, Mapping) else {}
    ranked: list[dict[str, Any]] = []
    for raw_key, raw_summary in stages.items():
        key = _safe_component_key(raw_key)
        parts = key.split("|")
        if len(parts) < 5 or parts[-2] != "total":
            continue
        item = {"key": key, **_summary(raw_summary)}
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            float(item.get("p99_ms") or -1.0),
            float(item.get("max_ms") or -1.0),
            str(item.get("key") or ""),
        ),
        reverse=True,
    )
    bounded = ranked[:_MAX_COMPONENTS]
    offenders = [
        item
        for item in bounded
        if item.get("p99_ms") is not None
        and float(item["p99_ms"]) > float(allowed_p99_ms)
    ]
    return bounded, offenders


def extract_boundary_diagnostics(
    pipeline_payload: Mapping[str, Any],
    *,
    allowed_p99_ms: float,
) -> dict[str, Any]:
    """Return a bounded, privacy-safe attribution of the zero-copy budget."""

    zero_copy = pipeline_payload.get("zero_copy_core")
    boundary = (
        zero_copy.get("boundary_serialization_metrics")
        if isinstance(zero_copy, Mapping)
        else None
    )
    boundary = boundary if isinstance(boundary, Mapping) else {}
    channels: dict[str, Any] = {}
    all_offenders: list[dict[str, Any]] = []
    for channel in ("ws", "rest"):
        raw = boundary.get(channel)
        raw = raw if isinstance(raw, Mapping) else {}
        total_stages, offenders = _rank_total_stages(
            raw.get("stages"),
            allowed_p99_ms=allowed_p99_ms,
        )
        channels[channel] = {
            **_summary(raw),
            "budget_total_stages": total_stages,
        }
        all_offenders.extend(
            {"channel": channel, **item} for item in offenders
        )
    all_offenders.sort(
        key=lambda item: float(item.get("p99_ms") or -1.0), reverse=True
    )
    return {
        "contract": "noesis.zero_copy.boundary_diagnostics",
        "contract_version": 1,
        "allowed_p99_ms": float(allowed_p99_ms),
        "combined_p99_ms": _finite_number(
            pipeline_payload.get("boundary_cpu_serialization_p99_ms")
        ),
        "channels": channels,
        "offending_budget_total_stages": all_offenders[:_MAX_COMPONENTS],
        "privacy": {
            "payload_content": "absent",
            "component_keys": "bounded_metric_labels_only",
        },
    }
