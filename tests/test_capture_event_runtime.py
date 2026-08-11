from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from noesis.capture_event_controller import (
    CanonicalCameraAliases,
    CaptureEventController,
)
from noesis.capture_event_runtime import (
    CaptureEventRuntimeError,
    CaptureEventRuntimeProviders,
    _capture_event_evidence,
    build_capture_event_request,
)
from noesis.depth_capture_event import DepthStorageFlushEvidence
from noesis_core.capture_event_fusion import FusedDepthSnapshot, RawDepthSnapshot
from noesis_core.mapanything_lifecycle import MapAnythingIdleReceipt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_capture_event_evidence_digest_mismatch_fails_closed() -> None:
    class _Outcome:
        compact_evidence_sha256 = "0" * 64

        @staticmethod
        def compact_evidence_payload() -> dict[str, object]:
            return {"contract": "tampered"}

    try:
        _capture_event_evidence(_Outcome())  # type: ignore[arg-type]
    except CaptureEventRuntimeError as exc:
        assert exc.code == "exact_depth_invalid"
    else:
        raise AssertionError("capture-event evidence digest mismatch was accepted")


def _depth_payload(
    timestamp_us: int,
    *,
    snapshot_id: str = "cached-1",
    snapshot_ref: str = "living-room/cached-1.zarr",
    content_sha256: str | None = None,
    role: str = "capture_event_fused",
    fusion_level: str = "intra_capture",
) -> dict[str, Any]:
    digest = content_sha256 or _digest("cached-content")
    component_url = (
        f"/api/v1/depth/snapshots/living-room/{snapshot_id}/components"
    )
    return {
        "contract": "noesis.depth.bulk_snapshot",
        "contract_version": 1,
        "ts": timestamp_us,
        "shape": [2, 3],
        "snapshot_id": snapshot_id,
        "snapshot_ref": snapshot_ref,
        "content_sha256": digest,
        "role": role,
        "fusion_level": fusion_level,
        "components": {
            "depth": {
                "component": "depth",
                "dtype": "<f4",
                "shape": [2, 3],
                "byte_count": 24,
                "sha256": _digest("depth"),
                "url": (
                    f"{component_url}/depth?snapshot_ref={snapshot_ref}"
                    f"&content_sha256={digest}"
                ),
            },
            "conf": {
                "component": "conf",
                "dtype": "<f4",
                "shape": [2, 3],
                "byte_count": 24,
                "sha256": _digest("conf"),
                "url": (
                    f"{component_url}/conf?snapshot_ref={snapshot_ref}"
                    f"&content_sha256={digest}"
                ),
            },
            "mask": {
                "component": "mask",
                "dtype": "|u1",
                "shape": [2, 3],
                "byte_count": 6,
                "sha256": _digest("mask"),
                "url": (
                    f"{component_url}/mask?snapshot_ref={snapshot_ref}"
                    f"&content_sha256={digest}"
                ),
            },
        },
        "normals": {
            "mode": "client_derived_depth_gradient_v1",
            "space": "camera",
            "dtype": "float32",
        },
    }


def _raw(index: int) -> RawDepthSnapshot:
    return RawDepthSnapshot(
        camera_id="living-room",
        storage_key="living-room",
        timestamp_us=1_000_000 + index * 100_000,
        snapshot_id=f"raw-{index}",
        artifact_ref=f"depth-zarr:living-room/{index}.zarr",
        content_sha256=_digest(f"raw-content-{index}"),
        sequence=index,
        manifest_sha256=_digest(f"raw-manifest-{index}"),
    )


class _MapAnything:
    fail = False

    def wait_idle(self, *, timeout_s: float = 5.0) -> MapAnythingIdleReceipt:
        if self.fail:
            raise TimeoutError("fixture idle failure")
        return MapAnythingIdleReceipt(
            active_captures=0,
            unfinished_tasks=0,
            worker_started=True,
            worker_alive=True,
            accepting=True,
        )


class _Storage:
    def __init__(self) -> None:
        self.raw: tuple[RawDepthSnapshot, ...] = ()
        self.cached: dict[str, Any] | None = None
        self.latest_calls = 0
        self.exact_load_calls = 0
        self.fuse_calls = 0
        self.floorplan_calls: list[dict[str, Any]] = []
        self.last_fused: FusedDepthSnapshot | None = None

    def flush_capture_frontier(
        self,
        *,
        timeout_s: float,
    ) -> DepthStorageFlushEvidence:
        return DepthStorageFlushEvidence(
            frontier_sequence=10,
            completed=True,
            timed_out=False,
            pending_sequences=(),
            failed_sequences=(),
            poisoned=False,
        )

    def max_raw_timestamp(self, camera_id: str) -> int:
        assert camera_id == "living-room"
        return self.raw[-1].timestamp_us if self.raw else 0

    def list_raw_snapshots(
        self,
        storage_key: str,
        *,
        camera_id: str,
        after_timestamp_us: int,
        limit: int,
    ) -> Sequence[RawDepthSnapshot]:
        assert storage_key == camera_id == "living-room"
        return tuple(row for row in self.raw if row.timestamp_us > after_timestamp_us)[
            -limit:
        ]

    def fuse_raw_snapshots(
        self,
        storage_key: str,
        snapshots: Sequence[RawDepthSnapshot],
        *,
        rgb_frame: object | None,
        event_id: str,
        min_observations: int,
        depth_agreement_m: float,
    ) -> FusedDepthSnapshot:
        assert rgb_frame is None
        self.fuse_calls += 1
        self.last_fused = FusedDepthSnapshot(
            camera_id=storage_key,
            storage_key=storage_key,
            timestamp_us=snapshots[-1].timestamp_us + 1,
            snapshot_id="fused-1",
            artifact_ref="depth-zarr:living-room/fused-1.zarr",
            content_sha256=_digest("fused-content"),
            sequence=max(row.sequence for row in snapshots) + 1,
            manifest_sha256=_digest("fused-manifest"),
            event_id=event_id,
            source_snapshot_ids=tuple(row.snapshot_id for row in snapshots),
        )
        return self.last_fused

    @staticmethod
    def validate_fused_snapshot(
        snapshot: FusedDepthSnapshot,
    ) -> FusedDepthSnapshot:
        return snapshot

    def describe_latest_depth_bulk(
        self,
        camera_id: str,
        ts_max_us: int | None = None,
    ) -> dict[str, Any] | None:
        assert camera_id == "living-room"
        self.latest_calls += 1
        return dict(self.cached) if self.cached is not None else None

    def describe_depth_snapshot_bulk_exact(
        self,
        *,
        camera_id: str,
        storage_ref: str,
        snapshot_id: str,
        content_sha256: str,
    ) -> dict[str, Any]:
        self.exact_load_calls += 1
        snapshot = self.last_fused
        assert snapshot is not None
        return _depth_payload(
            snapshot.timestamp_us,
            snapshot_id=snapshot_id,
            snapshot_ref=storage_ref,
            content_sha256=content_sha256,
            role=snapshot.snapshot_role,
            fusion_level=snapshot.fusion_level,
        )

    def generate_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.floorplan_calls.append(dict(kwargs))
        if not kwargs:
            return {
                "error": "no_cached_floorplan" if cache_only else "no_depth",
                "camera_id": camera_id,
            }
        snapshot = self.last_fused
        assert snapshot is not None
        return {
            "camera_id": camera_id,
            "snapshot_ref": kwargs["snapshot_ref"],
            "snapshot_id": kwargs["snapshot_id"],
            "snapshot_content_sha256": kwargs["snapshot_content_sha256"],
            "snapshot_ts": snapshot.timestamp_us,
            "frame": "camera_local_ground_m",
            "units": "meters",
            "bounds": {"min_x": 0.0, "max_x": 1.0, "min_z": 0.0, "max_z": 1.0},
            "scale_m_per_px": 0.5,
            "density": {"grid_shape": [2, 2], "value_min": 0.0, "value_max": 1.0},
        }


class _Harness:
    def __init__(self) -> None:
        self.storage = _Storage()
        self.mapanything = _MapAnything()
        self.gate_open = False
        self.failures: list[tuple[str, BaseException]] = []
        self.recorded: list[tuple[str, Mapping[str, Any]]] = []
        aliases = CanonicalCameraAliases({0: "living-room"})

        def burst_waiter(_seconds: float) -> bool:
            self.storage.raw = (_raw(1), _raw(2), _raw(3))
            return False

        self.controller = CaptureEventController(
            aliases=aliases,
            storage=self.storage,
            mapanything=self.mapanything,
            set_depth_gate=self._set_gate,
            depth_gate_is_open=lambda: self.gate_open,
            burst_waiter=burst_waiter,
        )
        self.providers = CaptureEventRuntimeProviders(
            aliases=aliases,
            storage=self.storage,
            controller_getter=lambda: self.controller,
            depth_branch_available=lambda: True,
            shutdown_requested=lambda: False,
            runtime_failure=lambda component, error: self.failures.append(
                (component, error)
            ),
            record_floorplan=self._record_floorplan,
        )

    def _set_gate(self, opened: bool) -> None:
        self.gate_open = opened

    def _record_floorplan(self, camera: str, payload: Mapping[str, Any]) -> bool:
        self.recorded.append((camera, payload))
        return True


def test_capture_defaults_preserve_long_quality_cohort_and_three_frame_support(
    monkeypatch,
) -> None:
    for name in (
        "NOESIS_DEPTH_RPC_ENABLE_SECONDS",
        "NOESIS_FLOORPLAN_DEPTH_ENABLE_SECONDS",
        "NOESIS_CAPTURE_EVENT_MIN_OBSERVATIONS",
        "NOESIS_CAPTURE_EVENT_DRAIN_TIMEOUT_SECONDS",
        "NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)

    request = build_capture_event_request(
        "living-room",
        request_kind="depth",
    )

    assert request.burst_seconds == 20.0
    assert request.min_observations == 3
    assert request.drain_timeout_s >= 35.0


def test_capture_drain_timeout_cannot_undercut_storage_commit(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S", "40")
    monkeypatch.setenv("NOESIS_CAPTURE_EVENT_DRAIN_TIMEOUT_SECONDS", "44")

    try:
        build_capture_event_request("living-room", request_kind="depth")
    except CaptureEventRuntimeError as exc:
        assert exc.code == "capture_event_configuration_invalid"
    else:
        raise AssertionError("unsafe capture drain timeout was accepted")


def test_cache_only_returns_before_controller_and_never_captures() -> None:
    harness = _Harness()
    harness.storage.cached = _depth_payload(123)
    harness.providers.controller_getter = lambda: (_ for _ in ()).throw(
        AssertionError("cache-only touched controller")
    )

    response = harness.providers.depth_provider("0", cache_only=True)

    assert response["ok"] is True
    assert response["served_from_cache"] is True
    assert harness.storage.fuse_calls == 0
    assert harness.storage.exact_load_calls == 0


def test_fresh_depth_loads_only_the_exact_fused_descriptor() -> None:
    harness = _Harness()

    response = harness.providers.depth_provider("0")

    assert response["ok"] is True
    assert response["served_from_cache"] is False
    assert harness.storage.latest_calls == 0
    assert harness.storage.exact_load_calls == 1
    assert harness.storage.fuse_calls == 1
    payload = response["payload"]
    assert payload["snapshot_id"] == "fused-1"
    assert payload["capture_event"]["rgb"] == {
        "status": "not_requested",
        "provider_configured": False,
    }


def test_fatal_barrier_error_has_no_stale_payload_and_escalates() -> None:
    harness = _Harness()
    harness.storage.cached = _depth_payload(999)
    harness.mapanything.fail = True

    response = harness.providers.depth_provider("living-room")

    assert response["ok"] is False
    assert response["error"] == "mapanything_idle_failed"
    assert "payload" not in response
    assert [component for component, _error in harness.failures] == ["capture_event"]


def test_floorplan_targets_exact_fused_snapshot_and_records_compact_evidence() -> None:
    harness = _Harness()

    response = harness.providers.floorplan_provider("0")

    assert response.get("error") is None
    assert harness.storage.floorplan_calls == [{
        "snapshot_ref": "living-room/fused-1.zarr",
        "snapshot_id": "fused-1",
        "snapshot_content_sha256": _digest("fused-content"),
    }]
    assert response["snapshot_ts"] == harness.storage.last_fused.timestamp_us
    assert response["capture_event"]["rgb"]["status"] == "not_requested"
    assert harness.recorded[-1][0] == "living-room"


def test_floorplan_reuses_manual_depth_snapshot_without_second_capture() -> None:
    harness = _Harness()
    depth_response = harness.providers.depth_provider("living-room")
    depth_payload = depth_response["payload"]
    assert harness.storage.fuse_calls == 1

    harness.providers.controller_getter = lambda: (_ for _ in ()).throw(
        AssertionError("exact-snapshot floorplan reopened capture controller")
    )
    response = harness.providers.floorplan_provider(
        "living-room",
        max_age_sec=0,
        grid_res_m=0.15,
        snapshot_ref=depth_payload["snapshot_ref"],
        snapshot_id=depth_payload["snapshot_id"],
        snapshot_content_sha256=depth_payload["content_sha256"],
    )

    assert response.get("error") is None
    assert response["snapshot_id"] == depth_payload["snapshot_id"]
    assert response["exact_snapshot_reused"] is True
    assert response["depth_burst_triggered"] is False
    assert response["depth_burst_fresh"] is False
    assert "capture_event" not in response
    assert harness.storage.fuse_calls == 1
    assert harness.storage.floorplan_calls[-1] == {
        "snapshot_ref": "living-room/fused-1.zarr",
        "snapshot_id": "fused-1",
        "snapshot_content_sha256": _digest("fused-content"),
    }


def test_floorplan_rejects_partial_or_cache_mixed_snapshot_identity() -> None:
    harness = _Harness()

    partial = harness.providers.floorplan_provider(
        "living-room",
        snapshot_ref="living-room/fused-1.zarr",
    )
    mixed = harness.providers.floorplan_provider(
        "living-room",
        cache_only=True,
        snapshot_ref="living-room/fused-1.zarr",
        snapshot_id="fused-1",
        snapshot_content_sha256=_digest("fused-content"),
    )

    assert partial == {
        "error": "invalid_snapshot_identity",
        "camera_id": "living-room",
    }
    assert mixed == {
        "error": "invalid_floorplan_request",
        "camera_id": "living-room",
    }
    assert harness.storage.floorplan_calls == []
    assert harness.storage.fuse_calls == 0


def test_floorplan_request_validation_precedes_storage() -> None:
    harness = _Harness()
    response = harness.providers.floorplan_provider(
        "living-room",
        grid_res_m=float("nan"),
    )
    assert response == {
        "error": "invalid_floorplan_request",
        "camera_id": "living-room",
    }
    assert harness.storage.floorplan_calls == []
    assert harness.storage.fuse_calls == 0


def test_depth_timestamp_contract_rejects_boolean_float_and_fractional_text() -> None:
    harness = _Harness()

    for invalid in (True, 1.25, "1.25", "1e6", "-1"):
        response = harness.providers.depth_provider(
            "living-room",
            ts_max_us=invalid,
        )
        assert response["error"] == "invalid_ts_max_us"

    assert harness.storage.latest_calls == 0
    assert harness.storage.fuse_calls == 0


def test_fresh_floorplan_never_uses_generic_latest_snapshot_path() -> None:
    harness = _Harness()

    def _reject_generic(*_args: object, **kwargs: object) -> dict[str, Any]:
        if not kwargs.get("snapshot_ref"):
            raise AssertionError("fresh floorplan used generic latest snapshot")
        return _Storage.generate_topdown_floorplan(
            harness.storage,
            "living-room",
            **kwargs,
        )

    harness.storage.generate_topdown_floorplan = _reject_generic

    response = harness.providers.floorplan_provider("living-room")

    assert response.get("error") is None
    assert harness.storage.fuse_calls == 1
    assert response["snapshot_id"] == "fused-1"


def test_floorplan_cache_only_miss_never_touches_controller_or_registry() -> None:
    harness = _Harness()
    harness.providers.controller_getter = lambda: (_ for _ in ()).throw(
        AssertionError("cache-only floorplan touched controller")
    )

    response = harness.providers.floorplan_provider(
        "living-room",
        cache_only=True,
    )

    assert response["error"] == "no_cached_floorplan"
    assert harness.storage.floorplan_calls == [{}]
    assert harness.storage.fuse_calls == 0
    assert harness.recorded == []


def test_stale_registry_rejection_is_stable_and_nonfatal() -> None:
    harness = _Harness()
    harness.providers.record_floorplan = lambda _camera, _payload: False

    response = harness.providers.floorplan_provider("living-room")

    assert response == {
        "error": "stale_floorplan_version",
        "camera_id": "living-room",
    }
    assert harness.failures == []


def test_runtime_mains_have_one_shared_controller_path_and_no_second_reader() -> None:
    for relative in (
        "noesis/ds8_runtime.py",
        "noesis/ds8_runtime_v3dt_reimpl.py",
        "DS9/noesis/ds9_runtime_core.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "CaptureEventController(" in source
        assert "capture_event_runtime_providers" in source
        assert "_fuse_capture_event_snapshots" not in source
        assert "candidate_keys" not in source
    root_source = (REPO_ROOT / "noesis/ds8_runtime.py").read_text(encoding="utf-8")
    assert "VideoCapture" not in root_source
    assert "_capture_camera_rgb_for_depth_snapshot" not in root_source
    ds9_source = (REPO_ROOT / "DS9/noesis/ds9_runtime_core.py").read_text(
        encoding="utf-8"
    )
    assert "NOESIS_DS8_STUB_PIPELINE" not in ds9_source
