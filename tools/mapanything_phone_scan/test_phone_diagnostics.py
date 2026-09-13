from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from .app import create_app
from .test_browser_capture import _settings


def _report():
    return {
        "schema": "noesis.phone_capture.android_capabilities.v1",
        "device": {"model": "diagnostic test"},
        "cameras": [{"id": "0", "supported8k": False, "reason": "Missing standard 8K mode"}],
    }


def test_phone_report_retains_exact_bytes_deduplicates_and_never_creates_scan(tmp_path):
    settings = _settings(tmp_path)
    raw = json.dumps(_report(), indent=2).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post("/api/phone-diagnostics", content=raw, headers={"Content-Type": "application/json"})
            assert response.status_code == 201
            assert response.json()["id"] == digest
            assert response.json()["diagnostic_only"] is True
        assert client.get("/api/scans").json() == []
    saved = list((settings.storage_root / ".phone-diagnostics").iterdir())
    assert len(saved) == 1
    assert saved[0].read_bytes() == raw


@pytest.mark.parametrize("raw,content_type,status", [
    (b"{}", "text/plain", 415),
    (b"not json", "application/json", 400),
    (b"[1]", "application/json", 422),
    (b'{"schema":"noesis.phone_capture.v1"}', "application/json", 422),
    (b'{"value":NaN}', "application/json", 400),
    (b"x" * (512 * 1024 + 1), "application/json", 413),
])
def test_invalid_report_is_not_persisted(tmp_path, raw, content_type, status):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/phone-diagnostics", content=raw, headers={"Content-Type": content_type})
        assert response.status_code == status
    assert not (settings.storage_root / ".phone-diagnostics").exists()


def test_report_storage_cap_preserves_existing_reports(tmp_path):
    settings = _settings(tmp_path)
    directory = settings.storage_root / ".phone-diagnostics"
    directory.mkdir(parents=True)
    for index in range(128):
        (directory / f"{index:064x}.json").write_text("existing report")
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/phone-diagnostics", json=_report())
        assert response.status_code == 507
    assert len(list(directory.iterdir())) == 128
    assert all(path.read_text() == "existing report" for path in directory.iterdir())
