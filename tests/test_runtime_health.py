from __future__ import annotations

from noesis_core.runtime_health import require_complete_camera_health


def test_empty_configured_camera_is_inactive_ready() -> None:
    health = require_complete_camera_health(
        {
            "healthy": True,
            "renderer_ready": True,
            "cameras": {
                "living-room": {
                    "healthy": True,
                    "success_count": 4,
                    "failure_count": 0,
                }
            },
        },
        ("living-room", "kitchen"),
    )
    assert health["healthy"] is True
    assert health["renderer_ready"] is True
    assert health["config_ready"] is True
    assert health["active_cameras"] == ["living-room"]
    assert health["inactive_cameras"] == ["kitchen"]
    assert health["cameras"]["kitchen"]["state"] == "inactive_ready"
    assert health["cameras"]["kitchen"]["healthy"] is True
    assert health["configured_camera_count"] == 2
    assert health["observed_camera_count"] == 1


def test_actual_renderer_failure_is_unhealthy() -> None:
    health = require_complete_camera_health(
        {
            "healthy": True,
            "cameras": {
                "living-room": {
                    "healthy": True,
                    "success_count": 1,
                    "failure_count": 0,
                },
                "kitchen": {
                    "healthy": False,
                    "success_count": 0,
                    "failure_count": 1,
                },
            },
        },
        ("living-room", "kitchen"),
    )
    assert health["healthy"] is False
    assert health["missing_cameras"] == []
    assert health["failed_cameras"] == ["kitchen"]
    assert health["active_cameras"] == ["living-room"]
    assert health["inactive_cameras"] == ["kitchen"]
    assert health["cameras"]["kitchen"]["state"] == "failed"


def test_existing_healthy_zero_success_row_is_inactive_ready() -> None:
    health = require_complete_camera_health(
        {
            "healthy": True,
            "renderer_ready": True,
            "cameras": {
                "living-room": {
                    "healthy": True,
                    "success_count": 0,
                    "failure_count": 0,
                }
            },
        },
        ("living-room",),
    )

    assert health["healthy"] is True
    assert health["active_camera_count"] == 0
    assert health["inactive_cameras"] == ["living-room"]
    assert health["cameras"]["living-room"]["state"] == "inactive_ready"
    assert health["cameras"]["living-room"]["active"] is False


def test_health_rejects_unexpected_renderer_camera() -> None:
    health = require_complete_camera_health(
        {
            "healthy": True,
            "renderer_ready": True,
            "cameras": {
                "living-room": {
                    "healthy": True,
                    "success_count": 1,
                    "failure_count": 0,
                },
                "unexpected": {
                    "healthy": True,
                    "success_count": 1,
                    "failure_count": 0,
                },
            },
        },
        ("living-room", "kitchen"),
    )
    assert health["healthy"] is False
    assert health["unexpected_cameras"] == ["unexpected"]


def test_invalid_or_negative_activity_counts_fail_the_camera_contract() -> None:
    for invalid_count in (-1, True, 1.5, "1", None):
        health = require_complete_camera_health(
            {
                "healthy": True,
                "renderer_ready": True,
                "cameras": {
                    "living-room": {
                        "healthy": True,
                        "success_count": invalid_count,
                        "failure_count": 0,
                    }
                },
            },
            ("living-room",),
        )

        assert health["healthy"] is False
        assert health["failed_cameras"] == ["living-room"]
        assert health["cameras"]["living-room"]["state"] == "failed"
        assert health["cameras"]["living-room"]["last_failure_type"] == (
            "InvalidActivityCount"
        )


def test_failure_before_first_success_cannot_be_inactive_ready() -> None:
    health = require_complete_camera_health(
        {
            "healthy": True,
            "renderer_ready": True,
            "cameras": {
                "living-room": {
                    "healthy": True,
                    "success_count": 0,
                    "failure_count": 1,
                }
            },
        },
        ("living-room",),
    )

    assert health["healthy"] is False
    assert health["failed_cameras"] == ["living-room"]
    assert health["cameras"]["living-room"]["state"] == "failed"
