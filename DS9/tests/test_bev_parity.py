from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from noesis.telemetry.bev import (
    BevRenderFailure,
    BevRenderer,
    CalibrationSnapshot,
    Footpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"
CANONICAL_BEV = REPO_ROOT / "noesis" / "telemetry" / "bev.py"


def _run_forced_ds9_characterization() -> dict[str, object]:
    script = r"""
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

repo = Path(sys.argv[1]).resolve()
adapter = Path(sys.argv[2]).resolve()
sys.path = [str(adapter), str(repo)] + [
    value
    for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]
for name in tuple(sys.modules):
    if name == "noesis" or name.startswith("noesis."):
        sys.modules.pop(name, None)

bev = importlib.import_module("noesis.telemetry.bev")
spec = importlib.util.spec_from_file_location(
    "forced_ds9_bev_characterization",
    repo / "tests" / "test_bev_renderer_world_smoothing.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load canonical BEV characterization")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rows = []
for name in sorted(value for value in vars(module) if value.startswith("test_")):
    function = getattr(module, name)
    monkeypatch = pytest.MonkeyPatch()
    try:
        parameters = tuple(inspect.signature(function).parameters)
        if parameters == ("monkeypatch",):
            function(monkeypatch)
        elif parameters:
            raise RuntimeError(f"unsupported characterization fixture: {name}{parameters}")
        else:
            function()
        rows.append({"name": name, "status": "pass"})
    except Exception as exc:
        rows.append(
            {
                "name": name,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        monkeypatch.undo()

print(
    json.dumps(
        {
            "module_origin": str(Path(bev.__file__).resolve()),
            "total": len(rows),
            "passed": sum(row["status"] == "pass" for row in rows),
            "rows": rows,
        },
        sort_keys=True,
    )
)
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", script, str(REPO_ROOT), str(DS9_ROOT)],
        cwd="/",
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def _run_forced_ds9_footpoint_probe() -> dict[str, object]:
    script = r"""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

repo = Path(sys.argv[1]).resolve()
adapter = Path(sys.argv[2]).resolve()
sys.path = [str(adapter), str(repo)] + [
    value for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]
for name in tuple(sys.modules):
    if name == "noesis" or name.startswith("noesis."):
        sys.modules.pop(name, None)

hooks = importlib.import_module("noesis.pipelines.hooks")
processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
processor.pipeline = SimpleNamespace(config={"visualization": {"trails": {"class_ids": [0]}}})
processor._bev_class_ids_ready = False
processor._bev_class_ids = frozenset({0})
footpoint = processor._footpoint_from_track(
    {
        "class_id": 0,
        "tracker_id": 44,
        "stable_id": 5,
        "frame_id": 77,
        "image_size": [100, 50],
        "image_foot": [60.0, 45.0],
        "bbox": [50.0, 5.0, 20.0, 40.0],
        "depth_status": "ok",
        "depth_registration_status": "ok",
        "depth_registered_m": 3.5,
        "depth_used_m": 3.5,
        "world_valid": True,
        "world": [1.0, 0.0, 2.0],
    },
    (100, 50),
    target_image_size=(200, 100),
)
invalid_status = processor._footpoint_from_track(
    {
        "class_id": 0,
        "tracker_id": 45,
        "bbox": [50.0, 5.0, 20.0, 40.0],
        "depth_status": "no_depth",
        "depth_registration_status": "ok",
        "depth_registered_m": 3.5,
        "depth_used_m": 3.5,
    },
    (100, 50),
)
mismatched_used = processor._footpoint_from_track(
    {
        "class_id": 0,
        "tracker_id": 46,
        "bbox": [50.0, 5.0, 20.0, 40.0],
        "depth_status": "ok",
        "depth_registration_status": "ok",
        "depth_registered_m": 3.5,
        "depth_used_m": 7.0,
    },
    (100, 50),
)
world_only = processor._footpoint_from_track(
    {
        "class_id": 0,
        "tracker_id": 47,
        "tracker_lifecycle_generation": 3,
        "stable_id": 6,
        "frame_id": 78,
        "world_valid": True,
        "world": [1.25, 0.0, 2.5],
        "world_frame": "backend_world_m",
        "world_frame_revision": "revision-1",
    },
    (100, 50),
)
unbound_world_only = processor._footpoint_from_track(
    {
        "class_id": 0,
        "tracker_id": 48,
        "tracker_lifecycle_generation": 3,
        "world_valid": True,
        "world": [1.25, 0.0, 2.5],
        "world_frame": "backend_world_m",
    },
    (100, 50),
)
print(json.dumps({
    "hooks_origin": str(Path(hooks.__file__).resolve()),
    "frame_id": footpoint.frame_id,
    "depth_m": footpoint.depth_m,
    "depth_source": footpoint.depth_source,
    "bbox": list(footpoint.bbox),
    "image_size": list(footpoint.image_size),
    "u": footpoint.u,
    "v": footpoint.v,
    "invalid_status_depth_m": invalid_status.depth_m,
    "mismatched_used_depth_m": mismatched_used.depth_m,
    "world_only": {
        "present": world_only is not None,
        "u": world_only.u if world_only is not None else None,
        "v": world_only.v if world_only is not None else None,
        "method": world_only.method if world_only is not None else None,
        "tracker_id": world_only.tracker_id if world_only is not None else None,
        "generation": (
            world_only.tracker_lifecycle_generation
            if world_only is not None
            else None
        ),
        "world": (
            [world_only.world_x, world_only.world_z]
            if world_only is not None
            else None
        ),
        "world_frame": world_only.world_frame if world_only is not None else None,
        "world_frame_revision": (
            world_only.world_frame_revision if world_only is not None else None
        ),
        "canonical_world_required": (
            world_only.canonical_world_required if world_only is not None else None
        ),
    },
    "unbound_world_only_present": unbound_world_only is not None,
    "debug": footpoint.debug,
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", script, str(REPO_ROOT), str(DS9_ROOT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def _calibration() -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id="cam0",
        intrinsics=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=[
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        floor_y=0.0,
        image_size=(1280, 720),
    )


def _active_floorplan() -> dict[str, object]:
    return {
        "camera_id": "cam0",
        "bounds": {"min_x": -4.0, "max_x": 4.0, "min_z": 0.0, "max_z": 8.0},
        "grid_shape": [80, 80],
        "grid_res_m": 0.1,
        "frame": "camera_local_ground_m",
        "units": "meters",
        "snapshot_ts_us": 100,
        "floorplan_ts_us": 200,
        "snapshot_id": "snapshot-100",
        "snapshot_content_sha256": "a" * 64,
        "calibration_fingerprint": "b" * 64,
        "source": "active_floorplan",
    }


class _Ws:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.fail = False
        self.next_submission_id = 1

    @staticmethod
    def response_model_timing_since(_started_ns):
        return {"duration_ms": 0.0}

    def broadcast_bev_sync(self, payload, **_kwargs):  # type: ignore[no-untyped-def]
        if self.fail:
            raise RuntimeError("boundary_closed")
        self.messages.append(payload)
        receipt = SimpleNamespace(
            submission_id=self.next_submission_id,
            message_count=1,
        )
        self.next_submission_id += 1
        return receipt


def test_forced_ds9_import_runs_all_canonical_bev_characterizations() -> None:
    result = _run_forced_ds9_characterization()
    assert result["module_origin"] == str(CANONICAL_BEV.resolve())
    assert result["total"] >= 25
    assert result["passed"] == result["total"], [
        row for row in result["rows"] if row["status"] != "pass"
    ]


def test_ds9_has_no_bev_shadow_and_resolves_the_canonical_owner() -> None:
    import noesis.telemetry.bev as bev

    assert not (DS9_ROOT / "noesis" / "telemetry" / "bev.py").exists()
    assert Path(bev.__file__).resolve() == CANONICAL_BEV.resolve()


def test_ds91_uses_the_approved_bev_coverage_envelope_contract() -> None:
    ds9_config = yaml.safe_load(
        (DS9_ROOT / "config" / "infer.yaml").read_text(encoding="utf-8")
    )

    coverage = ds9_config["bev"]["coverage_envelopes"]
    assert set(coverage["cameras"]) == {"living-room"}
    assert [
        region["id"]
        for region in coverage["cameras"]["living-room"]["regions"]
    ] == ["living-room", "foyer"]


def test_forced_ds9_hook_preserves_complete_footpoint_contract() -> None:
    payload = _run_forced_ds9_footpoint_probe()

    assert payload["hooks_origin"] == str(
        (DS9_ROOT / "noesis" / "pipelines" / "hooks.py").resolve()
    )
    assert payload["frame_id"] == 77
    assert payload["depth_m"] == 3.5
    assert payload["depth_source"] == "depth_registered_m"
    assert payload["bbox"] == [100.0, 10.0, 40.0, 80.0]
    assert payload["image_size"] == [200, 100]
    assert payload["u"] == 120.0
    assert payload["v"] == 90.0
    assert payload["invalid_status_depth_m"] is None
    assert payload["mismatched_used_depth_m"] is None
    assert payload["world_only"] == {
        "present": True,
        "u": None,
        "v": None,
        "method": "world",
        "tracker_id": 47,
        "generation": 3,
        "world": [1.25, 2.5],
        "world_frame": "backend_world_m",
        "world_frame_revision": "revision-1",
        "canonical_world_required": True,
    }
    assert payload["unbound_world_only_present"] is False
    assert payload["debug"]["track_frame_id"] == 77
    assert payload["debug"]["image_scale"] == [2.0, 2.0]


def test_root_first_telemetry_package_still_resolves_ds9_adapter() -> None:
    script = r"""
import importlib
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
adapter = Path(sys.argv[2]).resolve()
sys.path = [str(repo), str(adapter)] + [
    value
    for value in sys.path
    if value and Path(value).resolve() not in {repo, adapter}
]
for name in tuple(sys.modules):
    if name == "noesis" or name.startswith("noesis."):
        sys.modules.pop(name, None)

telemetry = importlib.import_module("noesis.telemetry")
world = importlib.import_module("noesis.telemetry.world_contract_adapter")
print(
    json.dumps(
        {
            "telemetry_origin": str(Path(telemetry.__file__).resolve()),
            "world_origin": str(Path(world.__file__).resolve()),
        },
        sort_keys=True,
    )
)
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", script, str(REPO_ROOT), str(DS9_ROOT)],
        cwd="/",
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "telemetry_origin": str(
            (REPO_ROOT / "noesis" / "telemetry" / "__init__.py").resolve()
        ),
        "world_origin": str(
            (
                DS9_ROOT
                / "noesis"
                / "telemetry"
                / "world_contract_adapter.py"
            ).resolve()
        ),
    }


def test_bev_failure_callback_and_health_reject_stale_homography_reuse() -> None:
    ws = _Ws()
    failures: list[BaseException] = []
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="world",
        failure_callback=failures.append,
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    initial_receipt = renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[
            Footpoint(
                u=640.0,
                v=360.0,
                stable_id=7,
                tracker_id=101,
                world_x=1.0,
                world_z=2.0,
            )
        ],
        timestamp_us=1_000_000,
    )
    assert ws.messages[-1]["type"] == "bev-frame"
    assert initial_receipt.status == "admitted"
    assert renderer.health_snapshot()["healthy"] is True

    def fail_homography(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("calibration_drift")

    renderer.h_cache.get = fail_homography  # type: ignore[method-assign]
    failed_homography_receipt = renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[
            Footpoint(
                u=640.0,
                v=360.0,
                stable_id=7,
                tracker_id=101,
                world_x=2.0,
                world_z=3.0,
            )
        ],
        timestamp_us=2_000_000,
    )

    assert ws.messages[-1]["type"] == "bev-status"
    assert failed_homography_receipt.status == "failed"
    assert ws.messages[-1]["error"] == "homography_failed"
    assert len([row for row in ws.messages if row["type"] == "bev-frame"]) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], BevRenderFailure)
    assert failures[0].stage == "homography"  # type: ignore[attr-defined]
    health = renderer.health_snapshot("cam0")
    assert health["healthy"] is False
    assert health["cameras"]["cam0"]["failure_count"] == 1
    assert health["cameras"]["cam0"]["last_failure_stage"] == "homography"


def test_bev_renderer_without_occupied_activity_is_ready_not_active() -> None:
    renderer = BevRenderer(
        _Ws(),
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )

    health = renderer.health_snapshot()

    assert health["contract_version"] == 2
    assert health["healthy"] is True
    assert health["renderer_ready"] is True
    assert health["rendering_active"] is False
    assert health["active_camera_count"] == 0
    assert health["failed_camera_count"] == 0


def test_bev_frame_carries_exact_source_frame_and_observation_identity() -> None:
    ws = _Ws()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="world",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False

    renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[
            Footpoint(
                u=640.0,
                v=700.0,
                stable_id=7,
                tracker_id=101,
                world_x=1.0,
                world_z=2.0,
                frame_id=44,
            )
        ],
        timestamp_us=8_000_000,
        source_id=2,
        frame_id=44,
        observed_at_us=9_000_000,
    )

    frame = ws.messages[-1]
    assert frame["type"] == "bev-frame"
    assert frame["sourceId"] == 2
    assert frame["frameId"] == 44
    assert frame["observedAtUs"] == 9_000_000
    assert frame["footpoints"][0]["frameId"] == 44


def test_bev_publish_boundary_failure_is_visible_to_health() -> None:
    ws = _Ws()
    ws.fail = True
    failures: list[BaseException] = []
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="world",
        failure_callback=failures.append,
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[
            Footpoint(
                u=640.0,
                v=360.0,
                stable_id=7,
                tracker_id=101,
                world_x=1.0,
                world_z=2.0,
            )
        ],
        timestamp_us=1_000_000,
    )
    assert len(failures) == 1
    assert isinstance(failures[0], BevRenderFailure)
    assert failures[0].stage == "publish"  # type: ignore[attr-defined]
    assert renderer.health_snapshot()["healthy"] is False


@pytest.mark.parametrize("failure_mode", ("invalid", "error"))
def test_configured_floorplan_provider_fails_closed_and_records_health(
    failure_mode: str,
) -> None:
    def provider(_camera_id: str):
        if failure_mode == "missing":
            return None
        if failure_mode == "invalid":
            return {"bounds": {"min_x": -1.0, "max_x": 1.0}}
        raise RuntimeError("registry_unavailable")

    ws = _Ws()
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=provider,
    )
    renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[],
        timestamp_us=1_000_000,
        source_id=0,
        frame_id=1,
        observed_at_us=1_000_000,
    )

    assert [message["type"] for message in ws.messages] == ["bev-status"]
    assert ws.messages[0]["error"] == "active_floorplan_failed"
    health = renderer.health_snapshot("cam0")
    assert health["healthy"] is False
    assert health["cameras"]["cam0"]["success_count"] == 0
    assert health["cameras"]["cam0"]["failure_count"] == 1
    assert health["cameras"]["cam0"]["last_failure_stage"] == "active_floorplan"


def test_floorplan_startup_pending_is_silent_but_post_ready_loss_is_fatal() -> None:
    ws = _Ws()
    provider_state: dict[str, object | None] = {"value": None}
    shutdown_requested = False
    failures: list[BaseException] = []

    def fail_runtime(error: BaseException) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        failures.append(error)

    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera_id: provider_state["value"],
        failure_callback=fail_runtime,
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]

    pending_receipt = renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[],
        timestamp_us=1_000_000,
        source_id=0,
        frame_id=1,
        observed_at_us=1_000_000,
    )

    assert ws.messages == []
    assert pending_receipt.status == "startup_pending"
    assert pending_receipt.frame_id == 1
    assert pending_receipt.outbound_submission_id is None
    assert failures == []
    assert shutdown_requested is False
    pending = renderer.health_snapshot("cam0")
    assert pending["healthy"] is True
    assert pending["rendering_active"] is False
    assert pending["floorplan_authority_pending_cameras"] == ["cam0"]
    assert pending["cameras"]["cam0"]["floorplan_authority_state"] == (
        "startup_pending"
    )

    provider_state["value"] = _active_floorplan()
    admitted_receipt = renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[],
        timestamp_us=2_000_000,
        source_id=0,
        frame_id=2,
        observed_at_us=2_000_000,
    )

    assert [message["type"] for message in ws.messages] == ["bev-frame"]
    assert admitted_receipt.status == "admitted"
    assert admitted_receipt.frame_id == 2
    assert admitted_receipt.outbound_submission_id == 1
    assert ws.messages[0]["footpoints"] == []
    assert failures == []
    ready = renderer.health_snapshot("cam0")
    assert ready["floorplan_authority_ready_cameras"] == ["cam0"]
    assert ready["cameras"]["cam0"]["floorplan_authority_state"] == "ready"

    provider_state["value"] = None
    failed_receipt = renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[],
        timestamp_us=3_000_000,
        source_id=0,
        frame_id=3,
        observed_at_us=3_000_000,
    )

    assert [message["type"] for message in ws.messages] == [
        "bev-frame",
        "bev-status",
    ]
    assert ws.messages[-1]["error"] == "active_floorplan_failed"
    assert failed_receipt.status == "failed"
    assert isinstance(failed_receipt.failure, BevRenderFailure)
    assert len(failures) == 1
    assert shutdown_requested is True
    failed = renderer.health_snapshot("cam0")
    assert failed["healthy"] is False
    assert failed["cameras"]["cam0"]["floorplan_authority_state"] == "lost"


def test_non_authority_failure_preserves_accepted_floorplan_state() -> None:
    failures: list[BaseException] = []
    renderer = BevRenderer(
        _Ws(),
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
        floorplan_bounds_provider=lambda _camera_id: _active_floorplan(),
        failure_callback=failures.append,
    )

    def fail_homography(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("calibration_drift")

    renderer.h_cache.get = fail_homography  # type: ignore[method-assign]
    renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[],
        timestamp_us=1_000_000,
        source_id=0,
        frame_id=1,
        observed_at_us=1_000_000,
    )

    assert len(failures) == 1
    health = renderer.health_snapshot("cam0")
    assert health["healthy"] is False
    assert health["cameras"]["cam0"]["last_failure_stage"] == "homography"
    assert health["cameras"]["cam0"]["floorplan_authority_state"] == "ready"


def test_provider_free_mode_publishes_exact_paired_empty_frame() -> None:
    ws = _Ws()
    ws.next_submission_id = 8
    renderer = BevRenderer(
        ws,
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="world",
    )
    renderer.h_cache.get = lambda *args, **kwargs: np.eye(3, dtype=np.float64)  # type: ignore[method-assign]
    renderer.config_per_cam["cam0"] = renderer.set_overlay("cam0", True)
    renderer.config_per_cam["cam0"].auto_fit_extents = False
    receipt = renderer.render_and_publish(
        "cam0",
        _calibration(),
        footpoints=[],
        timestamp_us=1_000_000,
        source_id=0,
        frame_id=1,
        observed_at_us=1_000_000,
        tracking_publication_sequence=3,
        tracking_outbound_submission_id=7,
    )

    assert ws.messages[-1]["type"] == "bev-frame"
    assert ws.messages[-1]["footpoints"] == []
    assert ws.messages[-1]["frameId"] == 1
    assert ws.messages[-1]["cohort"] == {
        "source_id": 0,
        "frame_id": 1,
        "observed_at_us": 1_000_000,
        "tracking_publication_sequence": 3,
        "tracking_outbound_submission_id": 7,
    }
    assert receipt.status == "admitted"
    assert receipt.tracking_outbound_submission_id == 7
    assert receipt.outbound_submission_id == 8
    assert renderer.health_snapshot("cam0")["healthy"] is True


def test_public_input_failure_boundary_is_visible_before_rendering() -> None:
    renderer = BevRenderer(
        _Ws(),
        trails_cfg={"enabled": False},
        smoothing_cfg={"enabled": False},
        frame="camera_local_ground_m",
    )

    failure = renderer.record_input_failure(
        "cam0",
        stage="calibration_snapshot",
        timestamp_us=1_000_000,
        cause=LookupError("missing calibration"),
    )

    assert failure.stage == "calibration_snapshot"
    health = renderer.health_snapshot("cam0")
    assert health["healthy"] is False
    assert health["cameras"]["cam0"]["success_count"] == 0
    assert health["cameras"]["cam0"]["failure_count"] == 1
