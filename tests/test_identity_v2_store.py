from __future__ import annotations

import sqlite3
import stat

import pytest

from reid.identity_v2 import (
    DuplicateResidentError,
    EnrollmentObservationKey,
    IdentityStore,
    IdentityStoreError,
    ModelProfileMismatch,
    VisitorSlotInUse,
)

FINGERPRINT = "swin-tiny:test-profile:v1"
DIMENSION = 4


def _resident(store: IdentityStore, *, name: str = "Alice", sid: int = 1):
    return store.create_resident(
        display_name=name,
        compatibility_sid=sid,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        now=10.0,
    )


def _anchor(
    store: IdentityStore, resident_uuid: str, *, suffix: str = "one", now: float = 11.0
):
    return store.add_enrollment_anchor(
        resident_uuid=resident_uuid,
        vector=(1.0, 0.25, -0.5, 0.0),
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        observation_id=f"anchor-observation-{suffix}",
        independence_key=f"anchor-independent-{suffix}",
        now=now,
    )


def test_schema_permissions_and_resident_survive_restart(tmp_path) -> None:
    state_dir = tmp_path / "private" / "identity"
    db_path = state_dir / "identity.sqlite3"
    store = IdentityStore(db_path)
    resident = _resident(store, name="Alice Example")
    assert store.schema_version == 2
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    store.close()

    with IdentityStore(db_path) as reopened:
        restored = reopened.get_resident(resident.resident_uuid)
        assert restored.display_name == "Alice Example"
        assert restored.compatibility_sid == 1
        assert reopened.schema_version == 2
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


def test_existing_schema_v1_database_migrates_explicitly_to_v2(tmp_path) -> None:
    db_path = tmp_path / "identity.sqlite3"
    connection = sqlite3.connect(db_path)
    IdentityStore._migrate_0_to_1(connection)
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (1, 1.0)"
    )
    connection.commit()
    connection.close()
    db_path.chmod(0o600)

    with IdentityStore(db_path) as store:
        assert store.schema_version == 2
        versions = store._conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 2]
        tables = {
            str(row[0])
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"visitor_exemplars", "enrollment_proposals"} <= tables


def test_resident_names_and_compatibility_sids_are_unique(tmp_path) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        first = _resident(store, name="  Alice   Smith  ", sid=1)
        assert first.display_name == "Alice Smith"
        assert first.normalized_name == "alice smith"

        with pytest.raises(DuplicateResidentError):
            _resident(store, name="ALICE smith", sid=2)
        with pytest.raises(DuplicateResidentError):
            _resident(store, name="Bob", sid=1)
        assert len(store.list_residents()) == 1


def test_visitor_slot_reuse_mints_noncolliding_generations_across_restart(
    tmp_path,
) -> None:
    db_path = tmp_path / "identity.sqlite3"
    store = IdentityStore(db_path)
    first = store.open_visitor_session(
        slot=1000,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        ttl_s=30.0,
        now=100.0,
    )
    assert first.generation == 1
    with pytest.raises(VisitorSlotInUse):
        store.open_visitor_session(
            slot=1000,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=30.0,
            now=101.0,
        )
    store.release_visitor_session(first.session_uuid, now=102.0)
    second = store.open_visitor_session(
        slot=1000,
        model_fingerprint=FINGERPRINT,
        embedding_dim=DIMENSION,
        ttl_s=30.0,
        now=103.0,
    )
    assert second.generation == 2
    assert second.session_uuid != first.session_uuid
    store.close()

    with IdentityStore(db_path) as reopened:
        reopened.release_visitor_session(second.session_uuid, now=104.0)
        third = reopened.open_visitor_session(
            slot=1000,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=30.0,
            now=105.0,
        )
        assert third.generation == 3
        assert len({first.session_uuid, second.session_uuid, third.session_uuid}) == 3


def test_session_touches_extend_ttl_but_cannot_resurrect_expired_incarnations(
    tmp_path,
) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        visitor = store.open_visitor_session(
            slot=1000,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=2.0,
            now=100.0,
        )
        provisional = store.create_provisional_session(
            tracklet_key="camera-a:session",
            ttl_s=2.0,
            now=100.0,
        )
        renewed = store.touch_visitor_session(visitor.session_uuid, now=101.0)
        store.touch_provisional_session(provisional, now=101.0)
        assert renewed.last_seen_at == 101.0
        assert renewed.expires_at == 103.0
        assert store.health(now=102.5).active_visitor_sessions == 1
        assert store.health(now=102.5).active_provisional_sessions == 1

        with pytest.raises(IdentityStoreError, match="expired"):
            store.touch_visitor_session(visitor.session_uuid, now=103.0)
        with pytest.raises(IdentityStoreError, match="expired"):
            store.touch_provisional_session(provisional, now=103.0)
        purged = store.purge_expired(now=103.0)
        assert purged.visitor_sessions == 1
        assert purged.provisional_sessions == 1

        replacement = store.open_visitor_session(
            slot=1000,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=2.0,
            now=104.0,
        )
        assert replacement.generation == 2
        with pytest.raises(KeyError):
            store.touch_visitor_session(visitor.session_uuid, now=104.0)


def test_anchors_are_float32_profile_checked_immutable_and_deterministic(
    tmp_path,
) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        resident = _resident(store)
        later = _anchor(store, resident.resident_uuid, suffix="later", now=20.0)
        earlier = _anchor(store, resident.resident_uuid, suffix="earlier", now=15.0)

        storage = store._conn.execute(
            "SELECT typeof(embedding), length(embedding) FROM resident_exemplars WHERE exemplar_uuid = ?",
            (later.exemplar_uuid,),
        ).fetchone()
        assert tuple(storage) == ("blob", DIMENSION * 4)
        with pytest.raises(ModelProfileMismatch):
            store.add_enrollment_anchor(
                resident_uuid=resident.resident_uuid,
                vector=(1.0, 0.0, 0.0, 0.0),
                model_fingerprint="different-model",
                embedding_dim=DIMENSION,
                observation_id="wrong-model",
                independence_key="wrong-model",
            )
        with pytest.raises(ModelProfileMismatch):
            store.load_resident_gallery(
                resident.resident_uuid,
                model_fingerprint=FINGERPRINT,
                embedding_dim=8,
            )

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE resident_exemplars SET embedding = ? WHERE exemplar_uuid = ?",
                    (b"unsafe", earlier.exemplar_uuid),
                )
        with pytest.raises(sqlite3.IntegrityError, match="only be deleted"):
            with store.transaction() as conn:
                conn.execute(
                    "DELETE FROM resident_exemplars WHERE exemplar_uuid = ?",
                    (earlier.exemplar_uuid,),
                )

        gallery = store.load_resident_gallery(
            resident.resident_uuid,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
        )
        assert [row.exemplar_uuid for row in gallery] == [
            earlier.exemplar_uuid,
            later.exemplar_uuid,
        ]
        assert gallery[0].vector == pytest.approx((1.0, 0.25, -0.5, 0.0))


def test_quarantine_requires_independent_corroboration_before_promotion(
    tmp_path,
) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        resident = _resident(store)
        exemplar = store.add_quarantine_exemplar(
            resident_uuid=resident.resident_uuid,
            vector=(0.0, 1.0, 0.0, 0.0),
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            observation_id="camera-a:track-1:frame-10",
            independence_key="camera-a:tracklet-1",
            ttl_s=300.0,
            now=100.0,
        )
        assert exemplar.role == "quarantine"
        assert exemplar.corroboration_count == 1

        duplicate_observation = store.corroborate_quarantine(
            exemplar.exemplar_uuid,
            observation_id="camera-a:track-1:frame-10",
            independence_key="camera-b:tracklet-9",
            now=101.0,
        )
        assert duplicate_observation.duplicate_observation
        assert duplicate_observation.corroboration_count == 1
        duplicate_tracklet = store.corroborate_quarantine(
            exemplar.exemplar_uuid,
            observation_id="camera-a:track-1:frame-11",
            independence_key="camera-a:tracklet-1",
            now=102.0,
        )
        assert duplicate_tracklet.duplicate_observation
        assert duplicate_tracklet.corroboration_count == 1

        second = store.corroborate_quarantine(
            exemplar.exemplar_uuid,
            observation_id="camera-b:track-9:frame-20",
            independence_key="camera-b:tracklet-9",
            required_observations=3,
            now=103.0,
        )
        assert second.corroboration_count == 2
        assert not second.promoted
        third = store.corroborate_quarantine(
            exemplar.exemplar_uuid,
            observation_id="camera-a:track-2:frame-30",
            independence_key="camera-a:tracklet-2",
            required_observations=3,
            now=104.0,
        )
        assert third.corroboration_count == 3
        assert third.promoted
        promoted = store.get_exemplar(exemplar.exemplar_uuid)
        assert promoted.role == "adaptive"
        assert promoted.expires_at is None
        assert promoted.promoted_at == 104.0


def test_ttl_purge_removes_only_ephemeral_and_quarantine_rows(tmp_path) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        resident = _resident(store)
        anchor = _anchor(store, resident.resident_uuid)
        quarantine = store.add_quarantine_exemplar(
            resident_uuid=resident.resident_uuid,
            vector=(0.0, 1.0, 0.0, 0.0),
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            observation_id="quarantine",
            independence_key="tracklet-quarantine",
            ttl_s=2.0,
            now=100.0,
        )
        visitor = store.open_visitor_session(
            slot=1000,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=2.0,
            now=100.0,
        )
        provisional = store.create_provisional_session(
            tracklet_key="camera-a:42",
            ttl_s=2.0,
            now=100.0,
        )

        purged = store.purge_expired(now=103.0)
        assert purged.visitor_sessions == 1
        assert purged.provisional_sessions == 1
        assert purged.quarantine_exemplars == 1
        assert (
            store.get_resident(resident.resident_uuid).resident_uuid
            == resident.resident_uuid
        )
        assert store.get_exemplar(anchor.exemplar_uuid).role == "enrollment_anchor"
        with pytest.raises(KeyError):
            store.get_exemplar(quarantine.exemplar_uuid)
        with pytest.raises(KeyError):
            store.get_visitor_session(visitor.session_uuid)
        assert provisional

        health = store.health(now=103.0)
        assert health.resident_count == 1
        assert health.residents_with_anchors == 1
        assert health.enrollment_anchor_count == 1
        assert health.quarantine_exemplar_count == 0
        assert health.active_visitor_sessions == 0
        assert health.active_provisional_sessions == 0


def test_resident_deletion_cascades_all_biometrics(tmp_path) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        resident = _resident(store)
        anchor = _anchor(store, resident.resident_uuid)
        quarantine = store.add_quarantine_exemplar(
            resident_uuid=resident.resident_uuid,
            vector=(0.0, 1.0, 0.0, 0.0),
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            observation_id="quarantine",
            independence_key="tracklet-1",
            ttl_s=300.0,
        )
        store.corroborate_quarantine(
            quarantine.exemplar_uuid,
            observation_id="other-observation",
            independence_key="tracklet-2",
        )

        result = store.delete_resident(resident.resident_uuid)
        assert result.resident_deleted
        assert result.exemplars_deleted == 2
        with pytest.raises(KeyError):
            store.get_resident(resident.resident_uuid)
        with pytest.raises(KeyError):
            store.get_exemplar(anchor.exemplar_uuid)
        with pytest.raises(KeyError):
            store.get_exemplar(quarantine.exemplar_uuid)
        health = store.health()
        assert health.resident_count == 0
        assert health.enrollment_anchor_count == 0
        assert health.quarantine_exemplar_count == 0


def test_health_is_derived_from_rows_and_transactions_rollback_atomically(
    tmp_path,
) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        alice = _resident(store, name="Alice", sid=1)
        _anchor(store, alice.resident_uuid)
        _resident(store, name="Bob", sid=2)
        store.open_visitor_session(
            slot=1000,
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=100.0,
            now=10.0,
        )
        store.create_provisional_session(
            tracklet_key="camera-a:1", ttl_s=100.0, now=10.0
        )

        health = store.health(now=20.0)
        assert health.resident_count == 2
        assert health.residents_with_anchors == 1
        assert health.residents_without_anchors == 1
        assert health.active_visitor_sessions == 1
        assert health.visitor_slot_count == 1
        assert health.active_provisional_sessions == 1
        assert health.model_profiles == ((FINGERPRINT, DIMENSION, 2),)

        with pytest.raises(sqlite3.IntegrityError):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO provisional_sessions VALUES (?, ?, ?, ?, ?, ?)",
                    ("rollback-one", "rollback-track", 20.0, 20.0, 30.0, 10.0),
                )
                conn.execute(
                    "INSERT INTO provisional_sessions VALUES (?, ?, ?, ?, ?, ?)",
                    ("rollback-two", "rollback-track", 20.0, 20.0, 30.0, 10.0),
                )
        count = store._conn.execute(
            "SELECT COUNT(*) FROM provisional_sessions WHERE tracklet_key = 'rollback-track'"
        ).fetchone()[0]
        assert count == 0


def test_expired_quarantine_cannot_be_promoted(tmp_path) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        resident = _resident(store)
        exemplar = store.add_quarantine_exemplar(
            resident_uuid=resident.resident_uuid,
            vector=(1.0, 0.0, 0.0, 0.0),
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            observation_id="first",
            independence_key="tracklet-first",
            ttl_s=1.0,
            now=10.0,
        )
        with pytest.raises(IdentityStoreError, match="expired"):
            store.corroborate_quarantine(
                exemplar.exemplar_uuid,
                observation_id="second",
                independence_key="tracklet-second",
                now=11.0,
            )


def test_enrollment_proposal_evidence_is_immutable_and_confirmation_is_atomic(
    tmp_path,
) -> None:
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        key = EnrollmentObservationKey(
            run_id="run",
            camera_id="camera",
            tracker_id="tracker",
            frame_id=10,
            observation_id="observation",
        )
        proposed = store.create_enrollment_proposal(
            key=key,
            observation_quality=0.95,
            observation_evidence=("sharp_crop", "frontal_pose"),
            display_name="Alice",
            vector=(1.0, 0.0, 0.0, 0.0),
            model_fingerprint=FINGERPRINT,
            embedding_dim=DIMENSION,
            ttl_s=30.0,
            compatibility_sid=1,
            now=10.0,
        )
        with pytest.raises(sqlite3.IntegrityError, match="evidence is immutable"):
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE enrollment_proposals SET embedding = ? WHERE proposal_uuid = ?",
                    (b"mutated", proposed.proposal.proposal_uuid),
                )

        confirmed = store.confirm_enrollment_proposal(
            proposed.proposal.proposal_uuid,
            expected_key=key,
            evidence_digest=proposed.proposal.evidence_digest,
            now=11.0,
        )
        assert confirmed.proposal.observation_quality == 0.95
        assert confirmed.proposal.observation_evidence == ("sharp_crop", "frontal_pose")
        assert store.health(now=11.0).resident_count == 1
        assert store.health(now=11.0).enrollment_anchor_count == 1
        with pytest.raises(sqlite3.IntegrityError, match="proposal is immutable"):
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE enrollment_proposals SET state = 'cancelled' WHERE proposal_uuid = ?",
                    (proposed.proposal.proposal_uuid,),
                )
        with pytest.raises(sqlite3.IntegrityError, match="only be deleted"):
            with store.transaction() as conn:
                conn.execute(
                    "DELETE FROM enrollment_proposals WHERE proposal_uuid = ?",
                    (proposed.proposal.proposal_uuid,),
                )
        assert store.delete_resident(confirmed.resident.resident_uuid).resident_deleted
        with pytest.raises(KeyError):
            store.get_enrollment_proposal(proposed.proposal.proposal_uuid)
