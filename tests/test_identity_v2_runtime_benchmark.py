from __future__ import annotations

import time

from reid.identity_v2 import (
    EnrollmentObservationKey,
    IdentityStore,
    IdentityV2Runtime,
    RuntimeObservation,
)


def test_household_hot_path_benchmark(tmp_path) -> None:
    """Guard the intended household scale without requiring a benchmark plugin."""

    dimension = 256
    fingerprint = "identity-v2-benchmark"
    with IdentityStore(tmp_path / "identity.sqlite3") as store:
        for sid in range(1, 5):
            resident = store.create_resident(
                display_name=f"Resident {sid}",
                compatibility_sid=sid,
                model_fingerprint=fingerprint,
                embedding_dim=dimension,
            )
            for exemplar_index in range(16):
                vector = [0.0] * dimension
                vector[sid - 1] = 1.0
                vector[4 + exemplar_index] = exemplar_index * 0.0001
                store.add_enrollment_anchor(
                    resident_uuid=resident.resident_uuid,
                    vector=vector,
                    model_fingerprint=fingerprint,
                    embedding_dim=dimension,
                    observation_id=f"resident-{sid}-anchor-{exemplar_index}",
                    independence_key=f"resident-{sid}-anchor-{exemplar_index}",
                )
        runtime = IdentityV2Runtime(
            store,
            model_fingerprint=fingerprint,
            embedding_dim=dimension,
        )
        observations = []
        for index in range(8):
            vector = [0.0] * dimension
            vector[index % 4] = 1.0
            observations.append(
                RuntimeObservation(
                    key=EnrollmentObservationKey(
                        run_id="benchmark-run",
                        camera_id=f"camera-{index % 4}",
                        tracker_id=f"tracker-{index}",
                        frame_id=100,
                        observation_id=f"observation-{index}",
                    ),
                    quality=0.95,
                    embedding=tuple(vector),
                )
            )

        runtime.resolve_batch(observations)
        iterations = 100
        started = time.perf_counter()
        for _ in range(iterations):
            decisions = runtime.resolve_batch(observations)
            assert len(decisions) == len(observations)
        elapsed = time.perf_counter() - started
        average_ms = elapsed / iterations * 1000.0
        # This generous ceiling catches accidental DB/I/O work or combinatorial
        # regressions while tolerating loaded CI hosts.
        assert average_ms < 100.0, f"identity-v2 hot batch averaged {average_ms:.2f} ms"
