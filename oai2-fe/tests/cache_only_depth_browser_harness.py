#!/usr/bin/env python3
"""Serve exact persisted depth/floorplan artifacts to the dashboard, cache-only.

This is a browser-acceptance fixture, not an inference substitute. It exposes
only committed fused snapshots and precomputed floorplan responses, records
every client request, and treats any fresh-depth or non-cache floorplan request
as an acceptance failure.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import numpy as np
import uvicorn


CAMERAS = ("living-room", "kitchen", "family-room")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


class Fixture:
    def __init__(
        self,
        *,
        repo_root: Path,
        render_root: Path,
        evidence_dir: Path,
    ) -> None:
        repo_root = repo_root.resolve()
        render_root = render_root.resolve()
        evidence_dir = evidence_dir.resolve()
        if not (repo_root / "DS9" / "AGENTS.md").is_file():
            raise RuntimeError("repo root does not contain DS9/AGENTS.md")
        if not render_root.is_dir():
            raise RuntimeError(f"render root not found: {render_root}")
        evidence_dir.mkdir(parents=True, exist_ok=False)

        # Select the DS9 overlay implementation without importing a DS8
        # runtime or starting any pipeline.
        sys.path.insert(0, str(repo_root / "DS9"))
        sys.path.insert(1, str(repo_root))
        from geometry.depth_source import (  # pylint: disable=import-outside-toplevel
            DepthBulkTransportError,
            DepthStorageManager,
        )
        from noesis.calibration.manager import (  # pylint: disable=import-outside-toplevel
            CalibrationManager,
        )

        self._bulk_error_type = DepthBulkTransportError
        self.evidence_dir = evidence_dir
        self.request_log_path = evidence_dir / "harness-requests.json"
        self._lock = threading.Lock()
        self._started_at = time.time()
        self.floorplans: dict[str, dict[str, Any]] = {}
        self.snapshot_identities: dict[str, dict[str, str]] = {}
        self.source_artifacts: dict[str, dict[str, Any]] = {}

        depth_bases: set[Path] = set()
        for camera in CAMERAS:
            matches = sorted(render_root.glob(f"{camera}_*"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one render directory for {camera}, found {len(matches)}"
                )
            camera_dir = matches[0]
            floorplan_path = camera_dir / "ws_floorplan_response__grid0.1.json"
            disk_snapshot_path = camera_dir / "disk_depth_snapshot.json"
            grids_path = camera_dir / "grids__grid0.1.npz"
            floorplan = _json_file(floorplan_path)
            disk_snapshot = _json_file(disk_snapshot_path)
            if not grids_path.is_file():
                raise RuntimeError(f"floorplan grids not found: {grids_path}")
            hydrated_layers: dict[str, dict[str, Any]] = {}
            with np.load(grids_path, allow_pickle=False) as grids:
                for layer_name in (
                    "density",
                    "height",
                    "height_agl",
                    "distance",
                    "gradient",
                    "obstacle_height",
                    "walkable",
                ):
                    layer = floorplan.get(layer_name)
                    if not isinstance(layer, dict):
                        raise RuntimeError(
                            f"floorplan layer {layer_name} missing for {camera}"
                        )
                    if layer.get("grid_b64") != "<grid_b64 omitted>":
                        continue
                    if layer_name not in grids:
                        raise RuntimeError(
                            f"NPZ layer {layer_name} missing for {camera}"
                        )
                    values = np.ascontiguousarray(
                        grids[layer_name],
                        dtype=np.dtype("<f4"),
                    )
                    if list(values.shape) != list(layer.get("grid_shape") or []):
                        raise RuntimeError(
                            f"NPZ shape mismatch for {camera}/{layer_name}"
                        )
                    raw = values.tobytes(order="C")
                    layer["grid_b64"] = base64.b64encode(raw).decode("ascii")
                    hydrated_layers[layer_name] = {
                        "shape": list(values.shape),
                        "byte_count": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
            depth_base = Path(str(disk_snapshot["depth_base"])).resolve()
            snapshot_path = Path(str(disk_snapshot["snapshot_path"])).resolve()
            if not snapshot_path.is_dir():
                raise RuntimeError(f"snapshot path not found: {snapshot_path}")
            if snapshot_path.parent == snapshot_path:
                raise RuntimeError("invalid snapshot path")
            try:
                storage_ref = snapshot_path.relative_to(depth_base).as_posix()
            except ValueError as exc:
                raise RuntimeError("snapshot path escapes declared depth base") from exc
            if str(floorplan.get("camera_id")) != camera:
                raise RuntimeError(f"floorplan camera mismatch for {camera}")
            self.floorplans[camera] = floorplan
            self.snapshot_identities[camera] = {
                "storage_ref": storage_ref,
                "snapshot_id": str(floorplan["snapshot_id"]),
                "content_sha256": str(floorplan["snapshot_content_sha256"]),
            }
            self.source_artifacts[camera] = {
                "floorplan_path": str(floorplan_path),
                "floorplan_sha256": _sha256_file(floorplan_path),
                "grids_path": str(grids_path),
                "grids_sha256": _sha256_file(grids_path),
                "hydrated_layers": hydrated_layers,
                "snapshot_path": str(snapshot_path),
                "snapshot_timestamp_us": int(disk_snapshot["snapshot_ts_us"]),
            }
            depth_bases.add(depth_base)

        if len(depth_bases) != 1:
            raise RuntimeError("fixture cameras do not share one depth store")
        depth_base = next(iter(depth_bases))
        self.storage = DepthStorageManager(
            base_path=depth_base,
            max_snapshots_per_camera=0,
            retention_minutes=0.0,
            enable_async=False,
            enforce_async=False,
            worker_count=1,
            max_worker_count=1,
        )

        for camera, identity in self.snapshot_identities.items():
            descriptor = self.storage.describe_snapshot(
                depth_base / identity["storage_ref"]
            )
            if (
                descriptor.camera_id != camera
                or descriptor.write_id != identity["snapshot_id"]
                or descriptor.content_sha256 != identity["content_sha256"]
            ):
                raise RuntimeError(f"snapshot identity mismatch for {camera}")
            self.source_artifacts[camera].update(
                {
                    "snapshot_ref": descriptor.storage_ref,
                    "snapshot_id": descriptor.write_id,
                    "snapshot_content_sha256": descriptor.content_sha256,
                    "snapshot_manifest_sha256": descriptor.manifest_sha256,
                }
            )

        calibration = CalibrationManager(
            cameras_yaml_path=repo_root / "config" / "cameras.yaml",
            camera_calibration_json_path=repo_root
            / "config"
            / "camera_calibration.json",
            ply_alignment_json_path=repo_root / "config" / "ply_alignment.json",
            streammux_size=(1920, 1080),
            raw_audit_dir=evidence_dir / "calibration-audit-unused",
        )
        calibration.set_camera_labels(dict(enumerate(CAMERAS)))
        self.calibration_bundle = calibration.calibration_bundle()
        if sorted(self.calibration_bundle.get("cameras", {}).get("K", {})) != sorted(
            CAMERAS
        ):
            raise RuntimeError("calibration bundle does not cover all fixture cameras")

        self.log: dict[str, Any] = {
            "contract": "noesis.depth-panel.cache-only-browser-harness.v1",
            "started_at_unix_s": self._started_at,
            "source_artifacts": self.source_artifacts,
            "ws_connections": 0,
            "ws_messages": [],
            "cache_only_depth_requests": [],
            "cache_only_floorplan_requests": [],
            "fresh_depth_requests": [],
            "non_cache_floorplan_requests": [],
            "component_requests": [],
        }
        self._persist()

    def _persist(self) -> None:
        with self._lock:
            payload = copy.deepcopy(self.log)
            payload["updated_at_unix_s"] = time.time()
            payload["fresh_request_count"] = len(payload["fresh_depth_requests"])
            payload["non_cache_floorplan_request_count"] = len(
                payload["non_cache_floorplan_requests"]
            )
            payload["cache_only_request_count"] = len(
                payload["cache_only_depth_requests"]
            ) + len(payload["cache_only_floorplan_requests"])
            payload["component_request_count"] = len(payload["component_requests"])
            encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            pending = self.request_log_path.with_suffix(".json.pending")
            pending.write_text(encoded, encoding="utf-8")
            pending.replace(self.request_log_path)

    def record_ws(self, value: dict[str, Any]) -> None:
        row = {
            "received_at_unix_s": time.time(),
            "type": str(value.get("type") or ""),
            "camera": str(value.get("camera") or ""),
            "request_id": str(value.get("request_id") or ""),
            "cache_only": value.get("cache_only"),
        }
        with self._lock:
            self.log["ws_messages"].append(row)
            if row["type"] == "get_ma_depth_cache" and row["cache_only"] is True:
                self.log["cache_only_depth_requests"].append(row)
            elif row["type"] == "get_ma_depth":
                self.log["fresh_depth_requests"].append(row)
            elif row["type"] == "get_floorplan":
                if row["cache_only"] is True:
                    self.log["cache_only_floorplan_requests"].append(row)
                else:
                    self.log["non_cache_floorplan_requests"].append(row)
        self._persist()

    def record_component(
        self,
        *,
        camera: str,
        snapshot_id: str,
        component: str,
        byte_count: int,
        sha256: str,
    ) -> None:
        with self._lock:
            self.log["component_requests"].append(
                {
                    "served_at_unix_s": time.time(),
                    "camera": camera,
                    "snapshot_id": snapshot_id,
                    "component": component,
                    "byte_count": byte_count,
                    "sha256": sha256,
                }
            )
        self._persist()

    def status(self) -> dict[str, Any]:
        self._persist()
        return _json_file(self.request_log_path)

    def close(self) -> None:
        self._persist()
        self.storage.shutdown(timeout=10.0)


def _build_app(fixture: Fixture) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            fixture.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/__harness/status")
    async def harness_status() -> dict[str, Any]:
        return fixture.status()

    @app.get(
        "/api/v1/depth/snapshots/{camera}/{snapshot_id}/components/{component}"
    )
    async def depth_component(
        camera: str,
        snapshot_id: str,
        component: str,
        snapshot_ref: str = Query(...),
        content_sha256: str = Query(...),
    ) -> Response:
        try:
            stream = fixture.storage.open_depth_snapshot_component(
                camera_id=camera,
                storage_ref=snapshot_ref,
                snapshot_id=snapshot_id,
                content_sha256=content_sha256,
                component=component,
            )
            try:
                body = b"".join(stream.iter_bytes())
            finally:
                stream.close()
        except fixture._bulk_error_type as exc:
            raise HTTPException(status_code=404, detail=exc.code) from exc
        descriptor = stream.descriptor
        fixture.record_component(
            camera=camera,
            snapshot_id=snapshot_id,
            component=component,
            byte_count=len(body),
            sha256=descriptor.sha256,
        )
        return Response(
            content=body,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "Content-Length": str(len(body)),
                "X-Noesis-Component-Sha256": descriptor.sha256,
                "X-Noesis-Snapshot-Id": snapshot_id,
            },
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        with fixture._lock:
            fixture.log["ws_connections"] += 1
        fixture._persist()
        await websocket.send_json(
            {
                "type": "calibration-bundle",
                "data": fixture.calibration_bundle,
            }
        )
        await websocket.send_json(
            {
                "type": "stats",
                "payload": {
                    "uptime": 0,
                    "cameras": {
                        camera: {
                            "status": "cached-fixture",
                            "frame_count": 0,
                            "tracking": {"occupancy": {}, "active_tracks": []},
                        }
                        for camera in CAMERAS
                    },
                    "pipeline": {
                        "mosaic_layout": {
                            "mosaic_w": 1920,
                            "mosaic_h": 1080,
                            "rows": 1,
                            "cols": 3,
                            "source_count": 3,
                            "sources": [
                                {"source_id": index, "camera_id": camera}
                                for index, camera in enumerate(CAMERAS)
                            ],
                            "frame_w": 1920,
                            "frame_h": 1080,
                        }
                    },
                },
            }
        )
        try:
            while True:
                value = await websocket.receive_json()
                if not isinstance(value, dict):
                    continue
                fixture.record_ws(value)
                message_type = str(value.get("type") or "")
                camera = str(value.get("camera") or "")
                request_id = str(value.get("request_id") or "")
                if message_type == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": value.get("timestamp")}
                    )
                    continue
                if message_type == "get_floorplan":
                    if value.get("cache_only") is not True or camera not in CAMERAS:
                        await websocket.send_json(
                            {
                                "type": "floorplan_response",
                                "request_id": request_id,
                                "camera_id": camera,
                                "error": "cache_only_fixture_refused_request",
                            }
                        )
                        continue
                    payload = copy.deepcopy(fixture.floorplans[camera])
                    payload.update(
                        {
                            "type": "floorplan_response",
                            "request_id": request_id,
                            "camera_id": camera,
                            "cache_only": True,
                            "served_from_cache": True,
                        }
                    )
                    await websocket.send_json(payload)
                    continue
                if message_type == "get_ma_depth_cache":
                    if value.get("cache_only") is not True or camera not in CAMERAS:
                        continue
                    identity = fixture.snapshot_identities[camera]
                    descriptor = fixture.storage.describe_depth_snapshot_bulk_exact(
                        camera_id=camera,
                        storage_ref=identity["storage_ref"],
                        snapshot_id=identity["snapshot_id"],
                        content_sha256=identity["content_sha256"],
                    )
                    await websocket.send_json(
                        {
                            "type": "ma_depth_response",
                            "ok": True,
                            "camera": camera,
                            "request_id": request_id,
                            "cache_only": True,
                            "served_from_cache": True,
                            "payload": descriptor,
                        }
                    )
                    continue
                if message_type == "get_ma_depth":
                    await websocket.send_json(
                        {
                            "type": "ma_depth_response",
                            "ok": False,
                            "camera": camera,
                            "request_id": request_id,
                            "cache_only": False,
                            "error": "fresh_inference_forbidden_in_fixture",
                        }
                    )
        except WebSocketDisconnect:
            return

    return app


def main() -> int:
    args = _parse_args()
    fixture = Fixture(
        repo_root=args.repo_root,
        render_root=args.render_root,
        evidence_dir=args.evidence_dir,
    )
    uvicorn.run(
        _build_app(fixture),
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
