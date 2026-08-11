from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from noesis.server.virtual_twin_api import app


def _write_revision(root: Path, revision_id: str) -> None:
    rev = root / "revisions" / revision_id
    rev.mkdir(parents=True)
    manifest = {
        "revision_id": revision_id,
        "camera": "living-room",
        "created_ts_us": 123,
        "artifacts": {
            "manifest": "manifest.json",
            "metrics": "metrics.json",
            "tracking_alignment": "tracking_alignment.json",
            "surfaces_glb": "surfaces.glb",
        },
    }
    (rev / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (rev / "metrics.json").write_text(json.dumps({"accepted_plane_count": 3}), encoding="utf-8")
    (rev / "tracking_alignment.json").write_text(json.dumps({"pose_correction": {"status": "ok"}}), encoding="utf-8")
    (rev / "surfaces.glb").write_bytes(b"glTF")
    (root / "latest").write_text(f"{revision_id}\n", encoding="utf-8")


def test_virtual_twin_latest_and_artifact_routes(tmp_path: Path, monkeypatch) -> None:
    _write_revision(tmp_path, "rev_a")
    monkeypatch.setenv("NOESIS_VIRTUAL_TWIN_ROOT", str(tmp_path))
    client = TestClient(app)

    latest = client.get("/api/v1/virtual-twin/latest")
    assert latest.status_code == 200
    payload = latest.json()
    assert payload["revision_id"] == "rev_a"
    assert payload["artifact_urls"]["surfaces_glb"].endswith("/surfaces.glb")

    manifest = client.get("/api/v1/virtual-twin/revisions/rev_a/manifest")
    assert manifest.status_code == 200
    assert manifest.json()["manifest"]["camera"] == "living-room"

    artifact = client.get("/api/v1/virtual-twin/revisions/rev_a/artifacts/surfaces.glb")
    assert artifact.status_code == 200
    assert artifact.content == b"glTF"
