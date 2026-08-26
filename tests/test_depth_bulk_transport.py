from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from starlette.requests import ClientDisconnect

from geometry import depth_source
from geometry.depth_source import (
    DepthBulkTransportError,
    DepthStorageManager,
    DepthStoragePoisonedError,
    SnapshotComponentDescriptor,
    SnapshotComponentStream,
    StorageLifecycle,
)
from noesis.capture_event_runtime import (
    validate_depth_bulk_component_bytes,
    validate_depth_bulk_snapshot_descriptor,
)
from noesis.server import depth_api
from noesis.server.internal_auth import configure_internal_rest_app
from noesis_core.capture_event_fusion import canonical_json_sha256


def _manager(tmp_path: Path) -> DepthStorageManager:
    return DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        worker_count=1,
        max_worker_count=1,
        zarr_clevel=0,
        zarr_chunk_px=2,
    )


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = (np.arange(24, dtype=np.float32).reshape(4, 6) + 0.25).astype(
        np.float32
    )
    conf = np.linspace(0.1, 0.9, 24, dtype=np.float32).reshape(4, 6)
    mask = (np.arange(24, dtype=np.uint8).reshape(4, 6) % 2).astype(np.uint8)
    rgb = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    return depth, conf, mask, rgb


def _commit(manager: DepthStorageManager, timestamp_us: int = 1_700_000_000_000_001):
    depth, conf, mask, rgb = _arrays()
    receipt = manager.store(
        "living-room",
        timestamp_us,
        depth,
        conf,
        mask,
        rgb=rgb,
        attrs={
            "snapshot_role": "capture_event_fused",
            "fusion_level": "intra_capture",
        },
    ).wait(timeout=5.0)
    return receipt, {"depth": depth, "conf": conf, "mask": mask, "rgb": rgb}


def _close(manager: DepthStorageManager) -> None:
    receipt = manager.shutdown(timeout=5.0)
    assert receipt.completed
    assert receipt.state is StorageLifecycle.CLOSED


def test_v2_commit_manifest_and_compact_descriptor_bind_raw_components(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    receipt, arrays = _commit(manager)
    manifest = manager._read_and_validate_commit_manifest(receipt.path)  # type: ignore[attr-defined]

    assert manifest["version"] == 2
    assert set(manifest["components"]) == {"depth", "conf", "mask", "rgb"}
    for component, array in arrays.items():
        canonical = np.ascontiguousarray(
            array,
            dtype=np.dtype("<f4" if component in {"depth", "conf"} else "|u1"),
        )
        row = manifest["components"][component]
        assert row == {
            "component": component,
            "dtype": "<f4" if component in {"depth", "conf"} else "|u1",
            "shape": list(canonical.shape),
            "byte_count": canonical.nbytes,
            "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        }

    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    assert payload["contract"] == "noesis.depth.bulk_snapshot"
    assert payload["contract_version"] == 1
    assert payload["shape"] == [4, 6]
    assert payload["snapshot_id"] == receipt.write_id
    assert payload["role"] == "capture_event_fused"
    assert payload["fusion_level"] == "intra_capture"
    assert payload["normals"] == {
        "mode": "client_derived_depth_gradient_v1",
        "space": "camera",
        "dtype": "float32",
    }
    assert "_b64" not in json.dumps(payload)
    assert all(
        row["url"].startswith("/api/v1/depth/snapshots/living-room/")
        for row in payload["components"].values()
    )
    assert validate_depth_bulk_snapshot_descriptor(
        payload,
        camera_id="living-room",
    )
    assert not validate_depth_bulk_snapshot_descriptor(
        payload,
        camera_id="kitchen",
    )
    _close(manager)


def test_public_cache_never_publishes_raw_or_lets_newer_raw_displace_fused(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    depth, conf, mask, rgb = _arrays()
    raw_only = manager.store(
        "living-room",
        1_700_000_000_000_001,
        depth,
        conf,
        mask,
        rgb=rgb,
    ).wait(timeout=5.0)
    assert manager.describe_latest_depth_bulk("living-room") is None

    fused, _arrays_by_name = _commit(manager, 1_700_000_000_000_002)
    newer_raw = manager.store(
        "living-room",
        1_700_000_000_000_003,
        depth,
        conf,
        mask,
        rgb=rgb,
    ).wait(timeout=5.0)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    assert payload["snapshot_id"] == fused.write_id
    assert payload["role"] == "capture_event_fused"
    assert payload["fusion_level"] == "intra_capture"

    raw_descriptor = manager.describe_snapshot(newer_raw.path)
    with pytest.raises(DepthBulkTransportError) as exact_error:
        manager.describe_depth_snapshot_bulk_exact(
            camera_id=raw_descriptor.camera_id,
            storage_ref=raw_descriptor.storage_ref,
            snapshot_id=raw_descriptor.write_id,
            content_sha256=raw_descriptor.content_sha256,
        )
    assert exact_error.value.code == "bulk_snapshot_role_invalid"
    with pytest.raises(DepthBulkTransportError) as component_error:
        manager.open_depth_snapshot_component(
            camera_id=raw_descriptor.camera_id,
            storage_ref=raw_descriptor.storage_ref,
            snapshot_id=raw_descriptor.write_id,
            content_sha256=raw_descriptor.content_sha256,
            component="depth",
        )
    assert component_error.value.code == "bulk_snapshot_role_invalid"
    assert manager.read_pin_count(raw_only.path) == 0
    assert manager.read_pin_count(newer_raw.path) == 0
    _close(manager)


def test_component_streams_are_chunked_byte_exact_and_release_their_lease(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    receipt, arrays = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    assert manager.read_pin_count(receipt.path) == 1

    component_rows = tuple(arrays.items())
    for component, source in component_rows:
        stream = manager.open_depth_snapshot_component(
            camera_id="living-room",
            storage_ref=payload["snapshot_ref"],
            snapshot_id=payload["snapshot_id"],
            content_sha256=payload["content_sha256"],
            component=component,
            max_chunk_bytes=64 * 1024,
        )
        assert manager.read_pin_count(receipt.path) == 2
        encoded = b"".join(stream.iter_bytes())
        dtype = "<f4" if component in {"depth", "conf"} else "|u1"
        expected = np.ascontiguousarray(source, dtype=np.dtype(dtype)).tobytes(
            order="C"
        )
        assert encoded == expected
        assert len(encoded) == payload["components"][component]["byte_count"]
        assert hashlib.sha256(encoded).hexdigest() == payload["components"][component][
            "sha256"
        ]
        assert validate_depth_bulk_component_bytes(
            payload["components"][component],
            encoded,
        )
        assert not validate_depth_bulk_component_bytes(
            payload["components"][component],
            encoded + b"tamper",
        )
        assert manager.read_pin_count(receipt.path) == 1
    assert manager.bulk_transfer_session_count == 1
    _close(manager)


def test_component_stream_lease_and_latest_public_snapshot_survive_retention(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    manager._retention_us = 1  # type: ignore[attr-defined]
    manager.prune("living-room")
    assert receipt.path.exists()
    assert manager.read_pin_count(receipt.path) == 1

    stream = manager.open_depth_snapshot_component(
        camera_id="living-room",
        storage_ref=payload["snapshot_ref"],
        snapshot_id=payload["snapshot_id"],
        content_sha256=payload["content_sha256"],
        component="depth",
    )
    newer, _newer_arrays = _commit(manager, receipt.ts_us + 1)
    manager.prune("living-room")
    assert receipt.path.exists()
    assert newer.path.exists()
    assert manager.read_pin_count(receipt.path) == 2

    stream.close()
    assert manager.read_pin_count(receipt.path) == 1
    manager._expire_bulk_transfer_sessions(  # type: ignore[attr-defined]
        now=time.monotonic() + 301.0
    )
    manager.prune("living-room")
    assert not receipt.path.exists()
    assert newer.path.exists()
    latest = manager.describe_latest_depth_bulk("living-room")
    assert latest is not None
    assert latest["snapshot_id"] == newer.write_id
    _close(manager)


def test_newer_public_snapshot_supersedes_older_retention_protection(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    older, _older_arrays = _commit(manager, 1_700_000_000_000_001)
    newer, _newer_arrays = _commit(manager, 1_700_000_000_000_002)

    manager._retention_us = 1  # type: ignore[attr-defined]
    manager.prune("living-room")

    assert not older.path.exists()
    assert newer.path.exists()
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    assert payload["snapshot_id"] == newer.write_id
    _close(manager)


def test_descriptor_grace_expires_without_retention_or_followup_calls(
    tmp_path: Path,
) -> None:
    manager = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        worker_count=1,
        max_worker_count=1,
        zarr_clevel=0,
        zarr_chunk_px=2,
        bulk_transfer_idle_grace_s=1.0,
        bulk_transfer_max_lifetime_s=2.0,
    )
    receipt, _arrays_by_name = _commit(manager)
    assert manager.describe_latest_depth_bulk("living-room") is not None
    assert manager.read_pin_count(receipt.path) == 1

    deadline = time.monotonic() + 3.0
    while manager.read_pin_count(receipt.path) and time.monotonic() < deadline:
        time.sleep(0.02)

    # Do not call bulk_transfer_session_count or the private expiry helper
    # before this assertion: the manager-owned reaper must do the release.
    assert manager.read_pin_count(receipt.path) == 0
    assert manager.bulk_transfer_session_count == 0
    _close(manager)


@pytest.mark.parametrize("poison_source", ("system", "writer"))
def test_storage_poison_rejects_new_descriptor_admission_but_reaps_existing_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison_source: str,
) -> None:
    manager = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        worker_count=1,
        max_worker_count=1,
        zarr_clevel=0,
        zarr_chunk_px=2,
        bulk_transfer_idle_grace_s=1.0,
        bulk_transfer_max_lifetime_s=2.0,
    )
    receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    assert manager.read_pin_count(receipt.path) == 1

    if poison_source == "system":
        manager._record_system_failure(  # type: ignore[attr-defined]
            "unit_test",
            OSError("injected storage poison"),
        )
    else:
        def fail_write(_job: object) -> None:
            raise OSError("injected storage poison")

        monkeypatch.setattr(manager, "_write_snapshot", fail_write)
        depth, conf, mask, rgb = _arrays()
        with pytest.raises(DepthStoragePoisonedError):
            manager.store(
                "living-room",
                1_700_000_000_000_002,
                depth,
                conf,
                mask,
                rgb=rgb,
            )
    assert manager.lifecycle_state is StorageLifecycle.OPEN
    assert manager.poison is not None

    # The exact URL published before poison remains usable during its original
    # bounded grace. Its component lease is independent of the descriptor pin.
    stream = manager.open_depth_snapshot_component(
        camera_id="living-room",
        storage_ref=payload["snapshot_ref"],
        snapshot_id=payload["snapshot_id"],
        content_sha256=payload["content_sha256"],
        component="depth",
    )
    assert manager.read_pin_count(receipt.path) == 2

    # Poison must not publish or renew any descriptor grace, even for the same
    # immutable identity that already owns a bounded transfer session.
    with pytest.raises(DepthBulkTransportError) as descriptor_error:
        manager.describe_depth_snapshot_bulk_exact(
            camera_id="living-room",
            storage_ref=payload["snapshot_ref"],
            snapshot_id=payload["snapshot_id"],
            content_sha256=payload["content_sha256"],
        )
    assert descriptor_error.value.code == "bulk_snapshot_unavailable"

    deadline = time.monotonic() + 3.0
    while manager.read_pin_count(receipt.path) > 1 and time.monotonic() < deadline:
        time.sleep(0.02)

    # Reading the pin count does not run expiry: this proves the lifetime
    # reaper survived poison. The already-open stream still owns its one pin.
    assert manager.read_pin_count(receipt.path) == 1
    assert manager.bulk_transfer_session_count == 0
    with pytest.raises(DepthBulkTransportError) as new_descriptor_error:
        manager.describe_latest_depth_bulk("living-room")
    assert new_descriptor_error.value.code == "bulk_snapshot_unavailable"
    assert manager.read_pin_count(receipt.path) == 1

    stream.close()
    assert manager.read_pin_count(receipt.path) == 0
    _close(manager)


def test_v1_snapshots_remain_readable_but_are_never_bulk_served(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    receipt, arrays = _commit(manager)
    manifest_path = receipt.path / manager._COMMIT_MANIFEST_NAME  # type: ignore[attr-defined]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 1
    manifest.pop("components")
    manager._write_manifest(receipt.path, manifest)  # type: ignore[attr-defined]
    _close(manager)

    manager = _manager(tmp_path)
    descriptor = manager.describe_snapshot(receipt.path)
    legacy_payload = manager.load_depth_snapshot_exact(
        camera_id=descriptor.camera_id,
        storage_ref=descriptor.storage_ref,
        snapshot_id=descriptor.write_id,
        content_sha256=descriptor.content_sha256,
    )
    assert base64.b64decode(legacy_payload["depth_b64"]) == arrays[
        "depth"
    ].tobytes(order="C")

    with pytest.raises(DepthBulkTransportError) as latest_error:
        manager.describe_latest_depth_bulk("living-room")
    assert latest_error.value.code == "bulk_component_manifest_missing"
    with pytest.raises(DepthBulkTransportError) as exact_error:
        manager.describe_depth_snapshot_bulk_exact(
            camera_id=descriptor.camera_id,
            storage_ref=descriptor.storage_ref,
            snapshot_id=descriptor.write_id,
            content_sha256=descriptor.content_sha256,
        )
    assert exact_error.value.code == "bulk_component_manifest_missing"
    _close(manager)


def test_exact_identity_and_allowlist_fail_closed_without_path_fallback(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    _receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None

    with pytest.raises(DepthBulkTransportError) as mismatch:
        manager.open_depth_snapshot_component(
            camera_id="living-room",
            storage_ref=payload["snapshot_ref"],
            snapshot_id=payload["snapshot_id"],
            content_sha256="0" * 64,
            component="depth",
        )
    assert mismatch.value.code == "bulk_snapshot_identity_mismatch"

    with pytest.raises(DepthBulkTransportError) as traversal:
        manager.open_depth_snapshot_component(
            camera_id="living-room",
            storage_ref="../outside.zarr",
            snapshot_id=payload["snapshot_id"],
            content_sha256=payload["content_sha256"],
            component="depth",
        )
    assert traversal.value.code == "bulk_snapshot_identity_mismatch"

    with pytest.raises(DepthBulkTransportError) as component_error:
        manager.open_depth_snapshot_component(
            camera_id="living-room",
            storage_ref=payload["snapshot_ref"],
            snapshot_id=payload["snapshot_id"],
            content_sha256=payload["content_sha256"],
            component="../../attrs",
        )
    assert component_error.value.code == "bulk_component_not_found"
    _close(manager)


def test_bulk_descriptor_and_component_stream_enforce_shared_resource_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    receipt, _arrays_by_name = _commit(manager)
    monkeypatch.setattr(depth_source, "DEPTH_BULK_MAX_COMPONENT_BYTES", 8)

    with pytest.raises(DepthBulkTransportError) as descriptor_error:
        manager.describe_latest_depth_bulk("living-room")
    assert descriptor_error.value.code == "bulk_snapshot_resource_limit_exceeded"

    descriptor = manager.describe_snapshot(receipt.path)
    with pytest.raises(DepthBulkTransportError) as stream_error:
        manager.open_depth_snapshot_component(
            camera_id=descriptor.camera_id,
            storage_ref=descriptor.storage_ref,
            snapshot_id=descriptor.write_id,
            content_sha256=descriptor.content_sha256,
            component="depth",
        )
    assert stream_error.value.code == "bulk_snapshot_unavailable"
    assert manager.read_pin_count(receipt.path) == 0
    _close(manager)


def test_bulk_api_streams_exact_headers_and_bytes(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    receipt, arrays = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    depth_api.app.state.depth_storage = manager
    try:
        with TestClient(depth_api.app) as client:
            response = client.get(payload["components"]["depth"]["url"])
        expected = arrays["depth"].astype("<f4", copy=False).tobytes(order="C")
        assert response.status_code == 200
        assert response.content == expected
        assert response.headers["content-length"] == str(len(expected))
        assert response.headers["x-noesis-component-sha256"] == hashlib.sha256(
            expected
        ).hexdigest()
        assert response.headers["x-noesis-snapshot-id"] == receipt.write_id
        assert response.headers["content-type"].startswith("application/octet-stream")
        assert "no-store" in response.headers["cache-control"]
        assert manager.read_pin_count(receipt.path) == 1
    finally:
        depth_api.app.state.depth_storage = None
        _close(manager)


@pytest.mark.parametrize(
    ("asgi_spec_version", "disconnect_mode"),
    (("2.3", "receive"), ("2.4", "send")),
)
def test_bulk_api_disconnect_closes_component_pin_then_grace_expires(
    tmp_path: Path,
    asgi_spec_version: str,
    disconnect_mode: str,
) -> None:
    manager = _manager(tmp_path)
    receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    split = urlsplit(payload["components"]["depth"]["url"])
    snapshot = manager.describe_snapshot(receipt.path)

    # Keep storage setup small, but make the injected real component stream
    # yield three chunks so the receive-side disconnect is deterministic.
    streamed_depth = np.arange(3 * 16_384, dtype="<f4").reshape(3, 16_384)
    streamed_bytes = streamed_depth.tobytes(order="C")
    completed_states: list[bool] = []

    def finish_component(component: str, completed: bool) -> None:
        completed_states.append(completed)
        manager._finish_bulk_transfer_component(  # type: ignore[attr-defined]
            snapshot,
            component,
            completed,
        )

    stream = SnapshotComponentStream(
        descriptor=SnapshotComponentDescriptor(
            component="depth",
            dtype="<f4",
            shape=streamed_depth.shape,
            byte_count=len(streamed_bytes),
            sha256=hashlib.sha256(streamed_bytes).hexdigest(),
        ),
        snapshot=snapshot,
        array=streamed_depth,
        lease=manager.acquire_read_lease(receipt.path),
        max_chunk_bytes=64 * 1024,
        on_close=finish_component,
    )
    assert manager.read_pin_count(receipt.path) == 2

    first_body_sent = asyncio.Event()
    received_body_bytes = 0

    async def exercise_disconnect() -> None:
        request_delivered = False

        async def receive() -> dict[str, object]:
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {
                    "type": "http.request",
                    "body": b"",
                    "more_body": False,
                }
            if disconnect_mode == "receive":
                await first_body_sent.wait()
                return {"type": "http.disconnect"}
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def send(message: dict[str, object]) -> None:
            nonlocal received_body_bytes
            if message.get("type") != "http.response.body":
                return
            body = bytes(message.get("body", b""))
            if body:
                received_body_bytes += len(body)
                if disconnect_mode == "send":
                    raise OSError("simulated disconnected client")
                first_body_sent.set()
                # Let StreamingResponse's pre-ASGI-2.4 receive task observe the
                # disconnect before it can request the next synchronous chunk.
                await asyncio.sleep(0)

        scope = {
            "type": "http",
            # ASGI <2.4 exercises StreamingResponse's explicit disconnect
            # listener. ASGI 2.4+ reports the same condition as an OSError
            # from ``send`` and is covered separately below.
            "asgi": {"version": "3.0", "spec_version": asgi_spec_version},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": split.path,
            "raw_path": split.path.encode("ascii"),
            "query_string": split.query.encode("ascii"),
            "headers": [],
            "client": ("testclient", 0),
            "server": ("testserver", 80),
            "root_path": "",
        }
        if disconnect_mode == "send":
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(depth_api.app(scope, receive, send), timeout=2.0)
        else:
            await asyncio.wait_for(depth_api.app(scope, receive, send), timeout=2.0)

    depth_api.app.state.depth_storage = SimpleNamespace(
        open_depth_snapshot_component=lambda **_kwargs: stream
    )
    try:
        asyncio.run(exercise_disconnect())
        assert 0 < received_body_bytes < len(streamed_bytes)
        assert completed_states == [False]
        assert manager.read_pin_count(receipt.path) == 1
        sessions = tuple(manager._bulk_transfer_sessions.values())  # type: ignore[attr-defined]
        assert len(sessions) == 1
        assert manager._expire_bulk_transfer_sessions(  # type: ignore[attr-defined]
            now=max(sessions[0].idle_deadline, sessions[0].absolute_deadline)
        ) == 1
        assert manager.read_pin_count(receipt.path) == 0
    finally:
        depth_api.app.state.depth_storage = None
        _close(manager)


@pytest.mark.parametrize(
    "query_suffix",
    (
        "&snapshot_ref=living-room/substituted.zarr",
        "&content_sha256=" + ("0" * 64),
        "&unexpected=value",
    ),
)
def test_bulk_api_rejects_duplicate_or_unknown_query_fields(
    tmp_path: Path,
    query_suffix: str,
) -> None:
    manager = _manager(tmp_path)
    _receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    depth_api.app.state.depth_storage = manager
    try:
        with TestClient(depth_api.app) as client:
            response = client.get(
                payload["components"]["depth"]["url"] + query_suffix
            )
        assert response.status_code == 400
        assert response.json()["detail"] == "bulk_snapshot_identity_invalid"
    finally:
        depth_api.app.state.depth_storage = None
        _close(manager)


def test_runtime_auth_middleware_protects_bulk_component_stream(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "store")
    _receipt, arrays = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    app = FastAPI()
    app.include_router(depth_api.app.router)
    app.state.depth_storage = manager
    auth = configure_internal_rest_app(
        app,
        env={
            "NOESIS_INTERNAL_AUTH_MODE": "required",
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE": str(tmp_path / "gateway-token"),
        },
    )
    assert auth.token is not None
    url = payload["components"]["mask"]["url"]
    try:
        with TestClient(app) as client:
            unauthorized = client.get(url)
            authorized = client.get(
                url,
                headers={"Authorization": f"Bearer {auth.token}"},
            )
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"] == "internal_auth_required"
        assert authorized.status_code == 200
        assert authorized.content == arrays["mask"].tobytes(order="C")
    finally:
        _close(manager)


def test_shared_depth_api_and_ds9_runtime_storage_wiring_remain_connected() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "noesis/server/depth_api.py").is_file()
    for relative in ("DS9/noesis/ds9_runtime_core.py",):
        source = (root / relative).read_text(encoding="utf-8")
        assert "rest_app.state.depth_storage = storage_manager" in source


def test_descriptor_grace_is_bounded_and_shutdown_releases_it(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    assert manager.bulk_transfer_session_count == 1
    assert manager.read_pin_count(receipt.path) == 1

    shutdown = manager.shutdown(timeout=5.0)
    assert shutdown.completed
    assert manager.bulk_transfer_session_count == 0
    assert manager.read_pin_count(receipt.path) == 0


def test_repeated_descriptor_read_never_renews_absolute_transfer_lifetime(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    key = next(iter(manager._bulk_transfer_sessions))  # type: ignore[attr-defined]
    absolute_deadline = manager._bulk_transfer_sessions[  # type: ignore[attr-defined]
        key
    ].absolute_deadline

    repeated = manager.describe_depth_snapshot_bulk_exact(
        camera_id="living-room",
        storage_ref=payload["snapshot_ref"],
        snapshot_id=payload["snapshot_id"],
        content_sha256=payload["content_sha256"],
    )

    assert repeated["snapshot_id"] == payload["snapshot_id"]
    session = manager._bulk_transfer_sessions[key]  # type: ignore[attr-defined]
    assert session.absolute_deadline == absolute_deadline
    assert session.idle_deadline <= absolute_deadline
    assert manager._expire_bulk_transfer_sessions(  # type: ignore[attr-defined]
        now=absolute_deadline
    ) == 1
    assert manager.read_pin_count(receipt.path) == 0
    _close(manager)


def test_descriptor_integer_shape_and_capture_evidence_are_browser_exact(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    _receipt, _arrays_by_name = _commit(manager)
    payload = manager.describe_latest_depth_bulk("living-room")
    assert payload is not None
    evidence = {
        "contract": "noesis.capture_event_controller",
        "parameters": {"burst_seconds": 4.0, "depth_agreement_m": 0.18},
    }
    payload["capture_event"] = evidence
    payload["capture_event_evidence_sha256"] = canonical_json_sha256(evidence)
    assert validate_depth_bulk_snapshot_descriptor(
        payload,
        camera_id="living-room",
    )

    tampered = json.loads(json.dumps(payload))
    tampered["capture_event"]["parameters"]["burst_seconds"] = 5.0
    assert not validate_depth_bulk_snapshot_descriptor(
        tampered,
        camera_id="living-room",
    )
    for invalid_shape in ([True, 6], ["4", 6], [4.0, 6]):
        invalid = dict(payload)
        invalid["shape"] = invalid_shape
        assert not validate_depth_bulk_snapshot_descriptor(
            invalid,
            camera_id="living-room",
        )
    invalid_ts = dict(payload)
    invalid_ts["ts"] = 1 << 53
    assert not validate_depth_bulk_snapshot_descriptor(
        invalid_ts,
        camera_id="living-room",
    )
    _close(manager)


def test_immutable_manifest_validation_cache_reuses_and_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    receipt, _arrays_by_name = _commit(manager)
    manager._invalidate_manifest_validation_cache()  # type: ignore[attr-defined]
    original = DepthStorageManager._read_and_validate_commit_manifest.__func__
    calls = 0

    def counted(cls: type[DepthStorageManager], path: Path, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(cls, path, **kwargs)

    monkeypatch.setattr(
        DepthStorageManager,
        "_read_and_validate_commit_manifest",
        classmethod(counted),
    )
    first = manager.describe_latest_depth_bulk("living-room")
    assert first is not None
    first_scan_count = calls
    assert first_scan_count == 1
    for _ in range(3):
        assert manager.describe_depth_snapshot_bulk_exact(
            camera_id="living-room",
            storage_ref=first["snapshot_ref"],
            snapshot_id=first["snapshot_id"],
            content_sha256=first["content_sha256"],
        )["snapshot_id"] == receipt.write_id
    assert calls == first_scan_count

    manifest_path = receipt.path / manager._COMMIT_MANIFEST_NAME  # type: ignore[attr-defined]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manager._write_manifest(receipt.path, manifest)  # type: ignore[attr-defined]
    assert manager.describe_snapshot(receipt.path).write_id == receipt.write_id
    assert calls == first_scan_count + 1
    _close(manager)
