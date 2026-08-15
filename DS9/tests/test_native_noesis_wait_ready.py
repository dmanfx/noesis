from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "native_noesis_wait_ready.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "native_noesis_wait_ready_test_module", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ready = _load_module()


def test_validate_capabilities_requires_healthy_tracking_and_world() -> None:
    payload = {
        "contract": "noesis.capability.health",
        "instance_id": "inst-1",
        "run_id": "run-1",
        "capabilities": [
            {"capability": "tracking_observations", "status": "healthy"},
            {"capability": "global_world", "status": "healthy"},
        ],
    }
    assert ready.validate_capabilities(payload) == {
        "instance_id": "inst-1",
        "run_id": "run-1",
    }


def test_validate_capabilities_rejects_unhealthy_world() -> None:
    payload = {
        "contract": "noesis.capability.health",
        "instance_id": "inst-1",
        "run_id": "run-1",
        "capabilities": [
            {"capability": "tracking_observations", "status": "healthy"},
            {"capability": "global_world", "status": "stale"},
        ],
    }
    with pytest.raises(ready.NativeReadinessError, match="global_world"):
        ready.validate_capabilities(payload)


def test_validate_deployment_matches_env_identity_not_selector_file() -> None:
    identity = {
        "NOESIS_DEPLOYMENT_ID": "deploy-test-native-host",
        "NOESIS_HEALTH_SELECTOR_SHA256": "a" * 64,
        "NOESIS_STATE_RELEASE_ID": "state-test-native-host",
        "NOESIS_SOFTWARE_REVISION": "b" * 40,
    }
    payload = {
        "contract": "noesis.appliance.deployment_health",
        "contract_version": 1,
        "ready": True,
        "runtime_family": "ds9",
        "runtime_variant": "ds9:baseline",
        "deployment_id": identity["NOESIS_DEPLOYMENT_ID"],
        "selector_sha256": identity["NOESIS_HEALTH_SELECTOR_SHA256"],
        "state_release_id": identity["NOESIS_STATE_RELEASE_ID"],
        "software_revision": identity["NOESIS_SOFTWARE_REVISION"],
        "instance_id": "inst-1",
        "run_id": "run-1",
        "generated_at_us": 123,
    }
    assert (
        ready.validate_deployment(
            payload,
            identity=identity,
            producer={"instance_id": "inst-1", "run_id": "run-1"},
        )["ready"]
        is True
    )


def test_validate_deployment_rejects_wrong_family() -> None:
    identity = {
        "NOESIS_DEPLOYMENT_ID": "deploy-test-native-host",
        "NOESIS_HEALTH_SELECTOR_SHA256": "a" * 64,
        "NOESIS_STATE_RELEASE_ID": "state-test-native-host",
        "NOESIS_SOFTWARE_REVISION": "b" * 40,
    }
    payload = {
        "contract": "noesis.appliance.deployment_health",
        "contract_version": 1,
        "ready": True,
        "runtime_family": "ds8",
        "runtime_variant": "ds8:baseline",
        "deployment_id": identity["NOESIS_DEPLOYMENT_ID"],
        "selector_sha256": identity["NOESIS_HEALTH_SELECTOR_SHA256"],
        "state_release_id": identity["NOESIS_STATE_RELEASE_ID"],
        "software_revision": identity["NOESIS_SOFTWARE_REVISION"],
        "instance_id": "inst-1",
        "run_id": "run-1",
        "generated_at_us": 123,
    }
    with pytest.raises(ready.NativeReadinessError, match="backend identity"):
        ready.validate_deployment(
            payload,
            identity=identity,
            producer={"instance_id": "inst-1", "run_id": "run-1"},
        )
