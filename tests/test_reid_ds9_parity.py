from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from DS9.noesis.server import reid_api as ds9_reid_api
from noesis.server import reid_api as ds8_reid_api


class _HouseholdManager:
    household_mode = True

    def __init__(self) -> None:
        self._residents: dict[str, dict[str, Any]] = {}

    def list_residents(self) -> list[dict[str, Any]]:
        return list(self._residents.values())

    def enroll_resident(self, **kwargs: Any) -> dict[str, Any]:
        stable_id = int(kwargs.get("stable_id") or kwargs.get("visitor_id") or 1)
        row = {
            "uuid": "resident-1",
            "stable_id": stable_id,
            "display_name": str(kwargs["display_name"]),
            "created_ts": 100.0,
            "embedding_count": 2,
            "gallery_embeddings": 2,
        }
        self._residents[row["uuid"]] = row
        return dict(row)

    def patch_resident(self, resident_uuid: str, **kwargs: Any) -> dict[str, Any]:
        row = self._residents[resident_uuid]
        if kwargs.get("display_name") is not None:
            row["display_name"] = str(kwargs["display_name"])
        return dict(row)

    def delete_resident(self, resident_uuid: str) -> dict[str, Any]:
        return dict(self._residents.pop(resident_uuid))

    def get_identity_health(self) -> dict[str, Any]:
        residents = self.list_residents()
        return {
            "household_mode": True,
            "resident_count": len(residents),
            "visitor_count": 0,
            "active_unique": len(residents),
            "residents": residents,
            "metrics": {"parity": True},
        }


def test_ds8_ds9_reid_source_and_openapi_are_exact_parity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "noesis/server/reid_api.py").read_bytes() == (
        repo_root / "DS9/noesis/server/reid_api.py"
    ).read_bytes()
    ds8_schema = ds8_reid_api.app.openapi()
    ds9_schema = ds9_reid_api.app.openapi()
    assert ds8_schema["paths"] == ds9_schema["paths"]
    assert ds8_schema["components"]["schemas"] == ds9_schema["components"]["schemas"]


def test_ds8_ds9_household_resident_behavior_is_wire_identical() -> None:
    ds8_manager = _HouseholdManager()
    ds9_manager = _HouseholdManager()
    ds8_reid_api.register_reid_manager_getter(lambda: ds8_manager)
    ds9_reid_api.register_reid_manager_getter(lambda: ds9_manager)
    ds8_client = TestClient(ds8_reid_api.app)
    ds9_client = TestClient(ds9_reid_api.app)

    operations = [
        ("GET", "/api/v1/reid/residents", None),
        ("POST", "/api/v1/reid/residents/enroll", {"display_name": "Alex"}),
        (
            "PATCH",
            "/api/v1/reid/residents/resident-1",
            {"display_name": "Alex Resident"},
        ),
        ("GET", "/api/v1/reid/identity_health", None),
        ("DELETE", "/api/v1/reid/residents/resident-1", None),
    ]
    for method, path, body in operations:
        ds8_response = ds8_client.request(method, path, json=body)
        ds9_response = ds9_client.request(method, path, json=body)
        assert ds8_response.status_code == ds9_response.status_code
        assert ds8_response.content == ds9_response.content
        assert (
            ds8_response.headers["content-type"] == ds9_response.headers["content-type"]
        )
