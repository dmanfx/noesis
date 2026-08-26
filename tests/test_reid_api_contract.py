from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from noesis.server import reid_api


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


def test_reid_openapi_exposes_the_shared_resident_contract() -> None:
    schema = reid_api.app.openapi()
    assert "/api/v1/reid/residents" in schema["paths"]
    assert "/api/v1/reid/residents/enroll" in schema["paths"]
    assert "/api/v1/reid/residents/{resident_uuid}" in schema["paths"]
    assert "/api/v1/reid/identity_health" in schema["paths"]


def test_reid_household_resident_endpoints_keep_the_wire_contract() -> None:
    manager = _HouseholdManager()
    reid_api.register_reid_manager_getter(lambda: manager)
    client = TestClient(reid_api.app)

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
        response = client.request(method, path, json=body)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
