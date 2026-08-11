from __future__ import annotations

import json
from pathlib import Path

import pytest

from noesis.calibration.depth_registration import DepthRegistrationManager
from noesis.calibration.world_fusion_policy import (
    CONTRACT_VERSION,
    WorldFusionPolicyError,
    load_world_fusion_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "config" / "world_measurement_fusion_policy.json"
CAMERAS = ("living-room", "kitchen", "family-room")


@pytest.mark.parametrize(
    ("lane", "registration_path"),
    (
        ("ds8", REPO_ROOT / "config" / "depth_registration.json"),
        ("ds9", REPO_ROOT / "DS9" / "config" / "depth_registration.json"),
    ),
)
def test_canonical_world_fusion_policy_binds_each_runtime_lane(
    lane: str,
    registration_path: Path,
) -> None:
    policy = load_world_fusion_policy(
        POLICY_PATH,
        runtime_lane=lane,
        active_camera_ids=CAMERAS,
        depth_registration=DepthRegistrationManager.load(registration_path),
    )
    assert len(policy.policy_id) == 64
    assert CONTRACT_VERSION == 2
    assert policy.evidence["range_capture_id"] == "alignment-walk-20260719T1733Z-recovered"
    assert {profile.floor_ray_max_range_m for profile in policy.cameras.values()} == {22.0}
    assert policy.profile("family-room").floor_weight_scale == 1.0
    assert policy.profile("family-room").depth_weight_scale == 0.0
    assert policy.profile("kitchen").floor_weight_scale == 0.0
    assert policy.profile("kitchen").depth_weight_scale == 1.0
    assert policy.profile("kitchen").floor_only_allowed is False
    assert policy.profile("living-room").floor_weight_scale == 1.0
    assert policy.profile("living-room").depth_weight_scale == 1.0
    assert policy.profile("living-room").floor_only_allowed is False


def _write_policy(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_world_fusion_policy_rejects_camera_set_drift(tmp_path: Path) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["cameras"].pop("kitchen")
    with pytest.raises(WorldFusionPolicyError, match="camera set"):
        load_world_fusion_policy(
            _write_policy(tmp_path, payload),
            runtime_lane="ds8",
            active_camera_ids=CAMERAS,
            depth_registration=DepthRegistrationManager.load(
                REPO_ROOT / "config" / "depth_registration.json"
            ),
        )


def test_world_fusion_policy_rejects_registration_binding_drift(tmp_path: Path) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["cameras"]["family-room"]["registration_ids"]["ds8"] = "family-room:wrong"
    with pytest.raises(WorldFusionPolicyError, match="registration binding mismatch"):
        load_world_fusion_policy(
            _write_policy(tmp_path, payload),
            runtime_lane="ds8",
            active_camera_ids=CAMERAS,
            depth_registration=DepthRegistrationManager.load(
                REPO_ROOT / "config" / "depth_registration.json"
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("floor_weight_scale", float("nan"), "non-finite JSON constant"),
        ("depth_weight_scale", 1.1, "within"),
    ),
)
def test_world_fusion_policy_rejects_invalid_scales(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["cameras"]["living-room"][field] = value
    with pytest.raises(WorldFusionPolicyError, match=message):
        load_world_fusion_policy(
            _write_policy(tmp_path, payload),
            runtime_lane="ds8",
            active_camera_ids=CAMERAS,
            depth_registration=DepthRegistrationManager.load(
                REPO_ROOT / "config" / "depth_registration.json"
            ),
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (True, "must be numeric"),
        (float("nan"), "non-finite JSON constant"),
        (0.0, "within"),
        (-1.0, "within"),
        (100.1, "within"),
    ),
)
def test_world_fusion_policy_rejects_invalid_floor_ray_range(
    tmp_path: Path,
    value: object,
    message: str,
) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["cameras"]["living-room"]["floor_ray_max_range_m"] = value
    with pytest.raises(WorldFusionPolicyError, match=message):
        load_world_fusion_policy(
            _write_policy(tmp_path, payload),
            runtime_lane="ds8",
            active_camera_ids=CAMERAS,
            depth_registration=DepthRegistrationManager.load(
                REPO_ROOT / "config" / "depth_registration.json"
            ),
        )


def test_ds8_ds9_policy_parsers_remain_byte_identical() -> None:
    assert (
        REPO_ROOT / "noesis" / "calibration" / "world_fusion_policy.py"
    ).read_bytes() == (
        REPO_ROOT / "DS9" / "noesis" / "calibration" / "world_fusion_policy.py"
    ).read_bytes()
