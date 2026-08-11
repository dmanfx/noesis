from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "wholebody49_occupied_scene_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("wholebody49_occupied_gate_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _events(
    *,
    mode: str,
    occupied: bool = True,
    source_ids: tuple[int, ...] = (0, 1, 2),
    frame_step: int = 6,
    cpu_violations: int = 0,
    pipeline_errors: list[str] | None = None,
) -> list[dict[str, object]]:
    base = 1_000_000_000
    events: list[dict[str, object]] = []
    frame_ids = {source_id: 0 for source_id in source_ids}
    for seconds in range(0, 33, 2):
        for offset, source_id in enumerate(source_ids):
            events.append(
                {
                    "type": "tracking",
                    "observed_at_us": base + seconds * 1_000_000 + offset + 1,
                    "source_id": source_id,
                    "frame_id": frame_ids[source_id],
                    "person_track_count": (
                        1 if occupied and source_id == source_ids[0] else 0
                    ),
                }
            )
            frame_ids[source_id] += frame_step
        events.append(
            {
                "type": "stats",
                "observed_at_us": base + seconds * 1_000_000 + len(source_ids) + 1,
                "counters": {
                    gate.MASK_COUNTER: 1 if mode == "masks" and occupied else 0,
                    gate.BBOX_COUNTER: 1 if mode == "boxes" and occupied else 0,
                    gate.CPU_VIOLATION_COUNTER: cpu_violations,
                },
                "pipeline_errors": list(pipeline_errors or []),
                "pipeline_prepared": True,
                "pipeline_activated": True,
                "application_running": True,
            }
        )
    return events


def _analyze(
    *,
    lane: str = "wholebody49-s",
    mode: str = "masks",
    events: list[dict[str, object]] | None = None,
    expected_source_ids: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    return gate.analyze_evidence(
        session_id="wholebody-quality-test",
        runtime_lane=lane,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        mode=mode,
        expected_source_ids=expected_source_ids,
        events=events if events is not None else _events(mode=mode),
    )


def test_mask_and_box_lanes_require_exact_parser_semantics() -> None:
    masks = _analyze()
    assert masks["ok"] is True
    assert masks["checks"]["mask_path_active"] is True

    boxes = _analyze(
        lane="wholebody49-x",
        mode="boxes",
        events=_events(mode="boxes"),
    )
    assert boxes["ok"] is True
    assert boxes["checks"]["bbox_path_active"] is True

    drifted_events = _events(mode="boxes")
    for event in drifted_events:
        if event["type"] == "stats":
            event["counters"][gate.MASK_COUNTER] = 1
    drifted = _analyze(
        lane="wholebody49-x",
        mode="boxes",
        events=drifted_events,
    )
    assert drifted["ok"] is False
    assert drifted["checks"]["mask_path_active"] is False


def test_empty_house_is_reported_honestly_and_never_promoted() -> None:
    result = _analyze(events=_events(mode="masks", occupied=False))
    assert result["ok"] is False
    assert result["status"] == "fail"
    assert result["scene_status"] == "empty"
    assert result["checks"]["occupied_person_tracks"] is False
    assert result["checks"]["mask_path_active"] is False


def test_source_complete_frame_rate_and_cadence_are_fail_closed() -> None:
    missing_source = _analyze(
        events=_events(mode="masks", source_ids=(0, 1)),
    )
    assert missing_source["checks"]["source_inventory_exact"] is False

    slow = _analyze(events=_events(mode="masks", frame_step=2))
    assert slow["checks"]["source_frame_rate_sufficient"] is False

    stalled_events = _events(mode="masks")
    stalled_events[:] = [
        event
        for event in stalled_events
        if not (
            event["type"] == "tracking"
            and event["source_id"] == 2
            and 4_000_000
            < event["observed_at_us"] - 1_000_000_000
            < 12_000_000
        )
    ]
    stalled = _analyze(events=stalled_events)
    assert stalled["checks"]["tracking_cadence_within_limit"] is False


def test_cpu_copy_pipeline_error_and_frame_regression_fail() -> None:
    violated = _analyze(
        events=_events(
            mode="masks",
            cpu_violations=1,
            pipeline_errors=["redacted_pipeline_error"],
        )
    )
    assert violated["checks"]["core_cpu_copy_violations_absent"] is False
    assert violated["checks"]["pipeline_errors_absent"] is False

    regressed_events = _events(mode="masks")
    source_zero = [
        event
        for event in regressed_events
        if event["type"] == "tracking" and event["source_id"] == 0
    ]
    source_zero[-1]["frame_id"] = source_zero[-2]["frame_id"]
    regressed = _analyze(events=regressed_events)
    assert regressed["checks"]["source_frames_advancing"] is False


def test_source_transcript_is_minimal_timestamped_and_replayable() -> None:
    events = _events(mode="masks")
    source = gate._source_transcript_document(
        session_id="wholebody-quality-test",
        runtime_lane="wholebody49-s",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        mode="masks",
        expected_source_ids=(0, 1, 2),
        events=events,
    )
    replay = gate._replay_source_events(
        source["messages"], expected_source_ids=source["expected_source_ids"]
    )
    assert replay["tracking_messages"] > 0
    assert replay["tracks_seen"] > 0
    encoded = json.dumps(source["messages"], sort_keys=True)
    for forbidden in ("embedding", "image_b64", "token", "secret"):
        assert forbidden not in encoded

    poisoned = [dict(events[0], embedding=[0.1, 0.2])]
    try:
        gate._replay_source_events(poisoned, expected_source_ids=(0, 1, 2))
    except ValueError as exc:
        assert "schema drifted" in str(exc)
    else:
        raise AssertionError("raw embedding field was accepted in Wholebody source")


def _sealed_bundle(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.chmod(0o700)
    events = _events(mode="masks")
    source = gate._source_transcript_document(
        session_id="wholebody-quality-test",
        runtime_lane="wholebody49-s",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        mode="masks",
        expected_source_ids=(0, 1, 2),
        events=events,
    )
    source_raw = gate._encoded_private_json(source)
    report = gate.analyze_evidence(
        session_id="wholebody-quality-test",
        runtime_lane="wholebody49-s",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        mode="masks",
        expected_source_ids=(0, 1, 2),
        events=events,
        source_evidence=gate._source_evidence_metadata(
            encoded=source_raw,
            document=source,
        ),
    )
    report_path = tmp_path / gate.CANONICAL_REPORT_FILENAME
    source_path = tmp_path / gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    report_path.write_bytes(gate._encoded_private_json(report))
    source_path.write_bytes(source_raw)
    report_path.chmod(0o600)
    source_path.chmod(0o600)
    return report_path, source_path


def test_sealed_wholebody_report_requires_exact_strict_replay(tmp_path: Path) -> None:
    report_path, source_path = _sealed_bundle(tmp_path)

    validated = gate.validate_sealed_wholebody49_report(
        report_path,
        source_path,
        session_id="wholebody-quality-test",
        runtime_lane="wholebody49-s",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        mode="masks",
        expected_source_ids=(0, 1, 2),
    )
    assert validated["ok"] is True

    tampered = json.loads(report_path.read_bytes())
    tampered["tracks_seen"] += 1
    report_path.write_bytes(gate._encoded_private_json(tampered))
    report_path.chmod(0o600)
    with pytest.raises(ValueError, match="exactly replay"):
        gate.validate_sealed_wholebody49_report(
            report_path,
            source_path,
            session_id="wholebody-quality-test",
            runtime_lane="wholebody49-s",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            mode="masks",
            expected_source_ids=(0, 1, 2),
        )


def test_sealed_wholebody_report_rejects_duplicate_source_key(tmp_path: Path) -> None:
    report_path, source_path = _sealed_bundle(tmp_path)
    source_raw = source_path.read_bytes().replace(
        b'"schema_version": 1,',
        b'"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    source_path.write_bytes(source_raw)
    source_path.chmod(0o600)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        gate.validate_sealed_wholebody49_report(
            report_path,
            source_path,
            session_id="wholebody-quality-test",
            runtime_lane="wholebody49-s",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            mode="masks",
            expected_source_ids=(0, 1, 2),
        )
