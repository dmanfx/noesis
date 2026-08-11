from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import reid_v2_api
from reid.identity_v2 import (
    EnrollmentObservationKey,
    IdentityStore,
    IdentityV2Runtime,
    RuntimeObservation,
)

FINGERPRINT = "api-v2-test-profile"
DIMENSION = 4

app = FastAPI()
app.include_router(reid_v2_api.router)


def _key(observation: str, *, tracker: str = "tracker-1", frame: int = 10) -> dict:
    return {
        "run_id": "run-a",
        "camera_id": "camera-a",
        "tracker_id": tracker,
        "frame_id": frame,
        "observation_id": observation,
    }


def _proposal_payload(
    observation: str,
    *,
    name: str = "Alice",
    sid: int | None = 1,
    tracker: str = "tracker-1",
    frame: int = 10,
    update_resident_uuid: str | None = None,
    ttl_s: float = 300.0,
) -> dict:
    payload = {
        "key": _key(observation, tracker=tracker, frame=frame),
        "display_name": name,
        "ttl_s": ttl_s,
    }
    if sid is not None:
        payload["compatibility_sid"] = sid
    if update_resident_uuid is not None:
        payload["update_resident_uuid"] = update_resident_uuid
    return payload


def _cache(
    runtime: IdentityV2Runtime,
    payload: dict,
    *,
    vector=(1.0, 0.0, 0.0, 0.0),
    quality: float = 0.95,
) -> None:
    runtime.ingest_observations(
        (
            RuntimeObservation(
                key=EnrollmentObservationKey(**payload["key"]),
                quality=quality,
                embedding=tuple(vector),
            ),
        )
    )


def _runtime(tmp_path, now):
    store = IdentityStore(tmp_path / "identity.sqlite3")
    runtime = IdentityV2Runtime(
        store,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        clock=lambda: now[0],
    )
    reid_v2_api.register_identity_v2_runtime_getter(lambda: runtime)
    return store, runtime, TestClient(app)


def _confirm(client: TestClient, proposal: dict, *, key: dict | None = None):
    return client.post(
        f"/api/v2/reid/enrollment/proposals/{proposal['proposal_uuid']}/confirm",
        json={
            "key": key or proposal["key"],
            "evidence_digest": proposal["evidence_digest"],
        },
    )


def test_api_is_explicitly_unavailable_until_v2_runtime_is_registered() -> None:
    reid_v2_api.register_identity_v2_runtime_getter(lambda: None)
    response = TestClient(app).get("/api/v2/reid/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "identity-v2 runtime unavailable"


def test_evidence_and_migration_review_are_read_only_biometric_free_statuses(
    tmp_path,
) -> None:
    now = [100.0]
    store, runtime, client = _runtime(tmp_path, now)
    review = {
        "contract": "noesis.identity.legacy_migration_review",
        "contract_version": 1,
        "source_digest": "a" * 64,
        "migration_key": "identity-v2:test",
        "apply_disposition": "partial",
        "full_migration_safe": False,
        "requires_accept_partial": True,
        "blocked_resident_count": 2,
        "migratable_resident_count": 0,
        "residents_without_importable_anchors": ["resident-a"],
        "duplicate_normalized_name_groups": [],
        "residents": [],
        "issues": [],
        "operator_action": "Resolve duplicates explicitly.",
        "apply_available_from_api": False,
    }
    recorder = SimpleNamespace(
        health=lambda: SimpleNamespace(
            contract="noesis.identity.shadow_score_evidence",
            contract_version=1,
            session_id="session-a",
            source="shadow",
            runtime="test",
            recorded_event_count=12,
            last_observed_at_us=123,
        )
    )
    service = SimpleNamespace(
        mode=SimpleNamespace(value="shadow"),
        model_layer="features",
        evidence_recorder=recorder,
        migration_review=review,
    )
    reid_v2_api.register_identity_v2_service_getter(lambda: service)
    try:
        evidence = client.get("/api/v2/reid/evidence/status")
        assert evidence.status_code == 200
        assert evidence.json()["contains_embeddings"] is False
        assert evidence.json()["recorded_event_count"] == 12
        assert "path" not in evidence.json()

        migration = client.get("/api/v2/reid/migration/review")
        assert migration.status_code == 200
        assert migration.json()["available"] is True
        assert migration.json()["apply_available_from_api"] is False
        assert "embedding" not in migration.text
        assert all(route.path != "/api/v2/reid/migration/apply" for route in app.routes)
    finally:
        reid_v2_api.register_identity_v2_service_getter(lambda: None)
        store.close()


def test_closed_identity_service_returns_503_instead_of_using_closed_store(
    tmp_path,
) -> None:
    now = [100.0]
    store, _runtime_value, client = _runtime(tmp_path, now)
    reid_v2_api.register_identity_v2_service_getter(
        lambda: SimpleNamespace(closed=True)
    )
    try:
        response = client.get("/api/v2/reid/health")
        assert response.status_code == 503
        assert response.json()["detail"] == "identity-v2 runtime is closed"
    finally:
        reid_v2_api.register_identity_v2_service_getter(lambda: None)
        store.close()


def test_typed_enrollment_health_resident_and_replay_lifecycle(tmp_path) -> None:
    now = [100.0]
    store, runtime, client = _runtime(tmp_path, now)
    try:
        health = client.get("/api/v2/reid/health")
        assert health.status_code == 200
        assert health.json()["schema_version"] == 2
        assert health.json()["resident_count"] == 0
        assert health.json()["scoring_calibration_status"] == "uncalibrated_default"
        assert health.json()["scoring_calibration_artifact_id"] is None
        assert health.json()["scoring_authority_scope"] is None
        assert health.json()["runtime_model_fingerprint"] == FINGERPRINT
        assert health.json()["runtime_model_semantic_profile_sha256"] is None
        assert health.json()["runtime_embedding_dim"] == DIMENSION
        assert health.json()["calibrated_maximum_total_candidates"] is None
        assert health.json()["observation_cache_entries"] == 0
        assert health.json()["observation_cache_max_entries"] >= 1

        payload = _proposal_payload("observation-1")
        missing = client.post("/api/v2/reid/enrollment/proposals", json=payload)
        assert missing.status_code == 409
        assert "no server-produced" in missing.json()["detail"]
        _cache(runtime, payload)
        forged = dict(payload)
        forged["embedding"] = [0.0, 1.0, 0.0, 0.0]
        rejected_forgery = client.post("/api/v2/reid/enrollment/proposals", json=forged)
        assert rejected_forgery.status_code == 422
        proposed = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=payload,
        )
        assert proposed.status_code == 201, proposed.text
        proposal = proposed.json()
        assert proposal["state"] == "pending"
        assert proposal["action"] == "create_resident"
        assert "embedding" not in proposal
        consumed = client.post("/api/v2/reid/enrollment/proposals", json=payload)
        assert consumed.status_code == 409
        assert "already consumed" in consumed.json()["detail"]

        listed = client.get("/api/v2/reid/enrollment/proposals")
        assert listed.status_code == 200
        assert listed.json()["count"] == 1
        assert (
            listed.json()["proposals"][0]["proposal_uuid"] == proposal["proposal_uuid"]
        )

        now[0] = 101.0
        confirmed = _confirm(client, proposal)
        assert confirmed.status_code == 200, confirmed.text
        confirmation = confirmed.json()
        assert confirmation["idempotent"] is False
        resident_uuid = confirmation["resident"]["resident_uuid"]
        assert confirmation["proposal"]["state"] == "confirmed"

        replay = _confirm(client, proposal)
        assert replay.status_code == 200
        assert replay.json()["idempotent"] is True
        assert (
            replay.json()["anchor_exemplar_uuid"]
            == confirmation["anchor_exemplar_uuid"]
        )

        residents = client.get("/api/v2/reid/residents")
        assert residents.status_code == 200
        assert residents.json()["count"] == 1
        assert residents.json()["residents"][0]["resident_uuid"] == resident_uuid
        health = client.get("/api/v2/reid/health").json()
        assert health["resident_count"] == 1
        assert health["enrollment_anchor_count"] == 1
        assert health["confirmed_enrollment_proposals"] == 1
    finally:
        store.close()


def test_api_stale_confirmation_and_expiry_are_conflicts_not_mutations(
    tmp_path,
) -> None:
    now = [100.0]
    store, runtime, client = _runtime(tmp_path, now)
    try:
        weak_payload = _proposal_payload("weak")
        _cache(runtime, weak_payload, quality=0.1)
        weak = client.post("/api/v2/reid/enrollment/proposals", json=weak_payload)
        assert weak.status_code == 422
        assert "quality floor" in weak.json()["detail"]
        stale_payload = _proposal_payload("stale", ttl_s=2.0)
        _cache(runtime, stale_payload)
        proposed = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=stale_payload,
        ).json()
        wrong_key = dict(proposed["key"])
        wrong_key["frame_id"] += 1
        stale = _confirm(client, proposed, key=wrong_key)
        assert stale.status_code == 409
        assert "exact observation evidence" in stale.json()["detail"]

        wrong_digest = client.post(
            f"/api/v2/reid/enrollment/proposals/{proposed['proposal_uuid']}/confirm",
            json={"key": proposed["key"], "evidence_digest": "0" * 64},
        )
        assert wrong_digest.status_code == 409

        now[0] = 102.0
        expired = _confirm(client, proposed)
        assert expired.status_code == 410
        fetched = client.get(
            f"/api/v2/reid/enrollment/proposals/{proposed['proposal_uuid']}"
        )
        assert fetched.status_code == 200
        assert fetched.json()["state"] == "expired"
        assert client.get("/api/v2/reid/health").json()["resident_count"] == 0
    finally:
        store.close()


def test_api_duplicate_name_requires_explicit_update_and_patch_rejects_collision(
    tmp_path,
) -> None:
    now = [100.0]
    store, runtime, client = _runtime(tmp_path, now)
    try:
        alice_payload = _proposal_payload("alice")
        _cache(runtime, alice_payload)
        alice_proposal = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=alice_payload,
        ).json()
        alice = _confirm(client, alice_proposal).json()["resident"]

        second_payload = _proposal_payload(
            "alice-second",
            name=" ALICE ",
            sid=2,
            tracker="tracker-2",
            frame=20,
        )
        _cache(runtime, second_payload)
        implicit_duplicate = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=second_payload,
        )
        assert implicit_duplicate.status_code == 409
        assert "update_resident_uuid" in implicit_duplicate.json()["detail"]

        explicit_payload = _proposal_payload(
            "alice-second",
            name=" ALICE ",
            sid=None,
            tracker="tracker-2",
            frame=20,
            update_resident_uuid=alice["resident_uuid"],
        )
        explicit = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=explicit_payload,
        )
        assert explicit.status_code == 201, explicit.text
        assert explicit.json()["action"] == "add_anchor"
        assert _confirm(client, explicit.json()).status_code == 200

        bob_payload = _proposal_payload(
            "bob",
            name="Bob",
            sid=2,
            tracker="tracker-3",
            frame=30,
        )
        _cache(runtime, bob_payload, vector=(0.0, 1.0, 0.0, 0.0))
        bob_proposal = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=bob_payload,
        ).json()
        bob = _confirm(client, bob_proposal).json()["resident"]
        collision = client.patch(
            f"/api/v2/reid/residents/{bob['resident_uuid']}",
            json={"display_name": "alice"},
        )
        assert collision.status_code == 409
        renamed = client.patch(
            f"/api/v2/reid/residents/{bob['resident_uuid']}",
            json={"display_name": "Robert"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["resident"]["normalized_name"] == "robert"
    finally:
        store.close()


def test_api_delete_cascades_enrollment_evidence_and_uses_delete_method(
    tmp_path,
) -> None:
    now = [100.0]
    store, runtime, client = _runtime(tmp_path, now)
    try:
        payload = _proposal_payload("delete")
        _cache(runtime, payload)
        proposal = client.post(
            "/api/v2/reid/enrollment/proposals",
            json=payload,
        ).json()
        resident = _confirm(client, proposal).json()["resident"]
        deleted = client.delete(f"/api/v2/reid/residents/{resident['resident_uuid']}")
        assert deleted.status_code == 200
        assert deleted.json()["resident_deleted"] is True
        assert deleted.json()["exemplars_deleted"] == 1
        assert client.get("/api/v2/reid/residents").json()["count"] == 0
        assert (
            client.get(
                f"/api/v2/reid/enrollment/proposals/{proposal['proposal_uuid']}"
            ).status_code
            == 404
        )
        assert (
            client.delete(
                f"/api/v2/reid/residents/{resident['resident_uuid']}"
            ).status_code
            == 404
        )
    finally:
        store.close()


def test_api_route_mutations_use_only_post_patch_and_delete() -> None:
    methods_by_path = {}
    for route in reid_v2_api.router.routes:
        if route.path.startswith("/api/v2/reid"):
            methods_by_path.setdefault(route.path, set()).update(route.methods or ())
    assert methods_by_path["/api/v2/reid/enrollment/proposals"] == {"GET", "POST"}
    assert methods_by_path[
        "/api/v2/reid/enrollment/proposals/{proposal_uuid}/confirm"
    ] == {"POST"}
    assert methods_by_path["/api/v2/reid/residents/{resident_uuid}"] == {
        "PATCH",
        "DELETE",
    }
    assert all("PUT" not in methods for methods in methods_by_path.values())
