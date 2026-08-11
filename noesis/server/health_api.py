from __future__ import annotations

import time
from typing import Callable

from fastapi import APIRouter, HTTPException, Request

from noesis_core.contracts.health import CapabilityHealth
from noesis_core.contracts.appliance import DeploymentHealth
from noesis_core.appliance import (
    DeploymentHealthBinding,
    ApplianceNotReadyError,
)
from noesis_core.health import CapabilityMonitor
from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    measure_rest_response_model,
)


router = APIRouter(route_class=BoundaryMetricsRoute)
_MONITOR_GETTER: Callable[[], CapabilityMonitor | None] | None = None
_DEPLOYMENT_BINDING_GETTER: Callable[[], DeploymentHealthBinding | None] | None = None


def register_capability_monitor_getter(
    getter: Callable[[], CapabilityMonitor | None],
) -> None:
    global _MONITOR_GETTER
    _MONITOR_GETTER = getter


def register_deployment_binding_getter(
    getter: Callable[[], DeploymentHealthBinding | None],
) -> None:
    global _DEPLOYMENT_BINDING_GETTER
    _DEPLOYMENT_BINDING_GETTER = getter


def clear_runtime_bindings() -> None:
    """Release process-local runtime owners after shutdown or startup abort."""
    global _MONITOR_GETTER, _DEPLOYMENT_BINDING_GETTER
    _MONITOR_GETTER = None
    _DEPLOYMENT_BINDING_GETTER = None


def _monitor() -> CapabilityMonitor:
    if _MONITOR_GETTER is None:
        raise HTTPException(status_code=503, detail="capability monitor not registered")
    monitor = _MONITOR_GETTER()
    if monitor is None:
        raise HTTPException(status_code=503, detail="capability monitor unavailable")
    return monitor


@router.get(
    "/api/v1/health/capabilities",
    response_model=CapabilityHealth,
    response_model_exclude_none=True,
)
def capability_health(request: Request) -> CapabilityHealth:
    snapshot = _monitor().snapshot(generated_at_us=max(1, time.time_ns() // 1_000))
    with measure_rest_response_model(
        "/api/v1/health/capabilities:get", "CapabilityHealth"
    ) as model_measurement:
        response = snapshot
    mark_rest_response(
        request,
        "/api/v1/health/capabilities:get",
        "CapabilityHealth",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get(
    "/api/v1/health/deployment",
    response_model=DeploymentHealth,
)
def deployment_health(request: Request) -> DeploymentHealth:
    if _DEPLOYMENT_BINDING_GETTER is None:
        raise HTTPException(status_code=503, detail="appliance deployment binding not registered")
    binding = _DEPLOYMENT_BINDING_GETTER()
    if binding is None:
        raise HTTPException(status_code=503, detail="appliance deployment binding unavailable")
    try:
        snapshot = binding.deployment_health(_monitor())
    except ApplianceNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with measure_rest_response_model(
        "/api/v1/health/deployment:get", "DeploymentHealth"
    ) as model_measurement:
        response = snapshot
    mark_rest_response(
        request,
        "/api/v1/health/deployment:get",
        "DeploymentHealth",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


__all__ = [
    "register_capability_monitor_getter",
    "register_deployment_binding_getter",
    "clear_runtime_bindings",
    "router",
]
