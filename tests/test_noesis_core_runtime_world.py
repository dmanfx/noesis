from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from noesis_core.runtime_world import create_runtime_world_service
from noesis_core.journal import AsyncContractJournal, ContractJournal


@dataclass(frozen=True)
class _Snapshot:
    camera_id: str
    intrinsics: list[list[float]]
    image_size: tuple[int, int]


class _CalibrationProvider:
    def snapshot(self, source_id: int, camera_id: str) -> _Snapshot | None:
        if camera_id == "missing":
            return None
        return _Snapshot(
            camera_id=camera_id,
            intrinsics=[[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
            image_size=(100, 50),
        )


def _track(camera_id: str) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "tracker_id": 4,
        "frame_id": 9,
        "observed_at_us": 100,
        "capture_time_status": "estimated",
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "image_size": [100, 50],
    }


def test_runtime_world_service_fingerprints_real_file_content(tmp_path: Path) -> None:
    model = tmp_path / "model.engine"
    model.write_bytes(b"model-a")
    service_a = create_runtime_world_service(
        runtime="ds8",
        pipeline_config={"models": {"detector": {"engine": str(model)}}},
        camera_labels={0: "kitchen"},
        calibration_provider=_CalibrationProvider(),
        repo_root=tmp_path,
        run_id="run-a",
        instance_id="test",
        software_revision="rev-a",
        journal_path=tmp_path / "journal-a.sqlite3",
    )
    first = service_a.publish(0, [_track("kitchen")], metadata={"camera_id": "kitchen"})
    assert isinstance(service_a._journal, AsyncContractJournal)
    service_a._journal.flush()
    assert isinstance(service_a._journal.journal, ContractJournal)
    assert len(service_a._journal.journal.records()) == 2

    model.write_bytes(b"model-b")
    service_b = create_runtime_world_service(
        runtime="ds8",
        pipeline_config={"models": {"detector": {"engine": str(model)}}},
        camera_labels={0: "kitchen"},
        calibration_provider=_CalibrationProvider(),
        repo_root=tmp_path,
        run_id="run-b",
        instance_id="test",
        software_revision="rev-b",
        journal_path=tmp_path / "journal-b.sqlite3",
    )
    second = service_b.publish(0, [_track("kitchen")], metadata={"camera_id": "kitchen"})
    service_a.close()
    service_b.close()

    assert first.observations[0].model.sha256 != second.observations[0].model.sha256
    assert first.observations[0].calibration.role == "camera_calibration"
    reopened = ContractJournal(tmp_path / "journal-a.sqlite3")
    try:
        persisted = reopened.records()
        assert [record.payload["contract"] for record in persisted] == [
            "noesis.observation.person",
            "noesis.world.snapshot",
        ]
    finally:
        reopened.close()


def test_missing_calibration_is_explicit_in_provenance(tmp_path: Path) -> None:
    service = create_runtime_world_service(
        runtime="ds9",
        pipeline_config={},
        camera_labels={0: "missing"},
        calibration_provider=_CalibrationProvider(),
        repo_root=tmp_path,
        run_id="run-a",
        instance_id="test",
        software_revision="rev-a",
        journal_path=tmp_path / "journal.sqlite3",
    )
    publication = service.publish(0, [_track("missing")], metadata={"camera_id": "missing"})
    service.close()
    assert publication.observations[0].calibration.role == "camera_calibration_unavailable"


def test_runtime_world_config_fingerprint_uses_public_camera_reference(tmp_path: Path) -> None:
    base = {
        "version": 1,
        "sources": [{"uri_secret": "living-room"}],
        "models": {"detector": {"name": "test"}},
    }
    configs = [
        {**base, "sources": [{"uri_secret": "living-room", "uri": f"rtsp://camera.invalid/private-{suffix}"}]}
        for suffix in ("one", "two")
    ]
    publications = []
    for index, config in enumerate(configs):
        service = create_runtime_world_service(
            runtime="ds8",
            pipeline_config=config,
            camera_labels={0: "kitchen"},
            calibration_provider=_CalibrationProvider(),
            repo_root=tmp_path,
            run_id=f"run-{index}",
            instance_id="test",
            software_revision="rev",
            journal_path=tmp_path / f"journal-{index}.sqlite3",
        )
        publications.append(
            service.publish(0, [_track("kitchen")], metadata={"camera_id": "kitchen"})
        )
        service.close()

    assert publications[0].observations[0].config.sha256 == publications[1].observations[0].config.sha256
