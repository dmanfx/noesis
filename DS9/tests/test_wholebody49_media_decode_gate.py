from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "wholebody49_media_decode_gate.py"
SPEC = importlib.util.spec_from_file_location("wholebody49_media_gate_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _probe_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "ok": True,
        "answer_video_direction": "sendonly",
        "ice_state": "connected",
        "peer_state": "connected",
        "rtp_packets": 20,
        "decoded_frames": 3,
        "saw_src_pad": True,
        "src_caps": "video/x-raw, format=I420; application/x-rtp, encoding-name=H264",
        "min_rtp": 10,
        "min_decoded": 1,
        "rtsp_decoded_frames": 4,
        "rtsp_min_decoded": 1,
    }
    result.update(overrides)
    return result


def _source(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": "wholebody-media-test",
        "runtime_lane": "wholebody49-s",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "started_at_us": 1_000_000,
        "finished_at_us": 7_000_000,
        "probe_result": _probe_result(),
    }
    values.update(overrides)
    return gate._source_document(**values)


def test_media_gate_requires_direct_rtsp_and_webrtc_decoded_frames() -> None:
    source = _source()
    raw = gate._encoded(source)
    report = gate.analyze_source(
        source,
        source_evidence=gate._source_evidence(raw, source),
    )
    assert report["ok"] is True
    assert report["checks"]["direct_rtsp_decode_threshold_met"] is True
    assert report["checks"]["decoded_frame_threshold_met"] is True

    for mutation in (
        {"decoded_frames": 0},
        {"rtsp_decoded_frames": 0},
        {"rtp_packets": 9},
        {"src_caps": "video/x-raw"},
        {"ice_state": "attacker-controlled-state"},
    ):
        failed_source = _source(probe_result=_probe_result(**mutation))
        failed = gate.analyze_source(
            failed_source,
            source_evidence=gate._source_evidence(
                gate._encoded(failed_source), failed_source
            ),
        )
        assert failed["ok"] is False


def test_media_source_is_session_bound_and_privacy_closed() -> None:
    source = _source()
    encoded = json.dumps(source, sort_keys=True)
    for forbidden in ("candidate:", "a=ice", "secret-value", "image_b64"):
        assert forbidden not in encoded
    unexpected = _source(
        probe_result=_probe_result(ice_state="attacker-controlled-state")
    )
    assert unexpected["sample"]["ice_state"] == "unexpected"
    for field, value, label in (
        ("ice_state", "attacker-controlled-state", "ICE state"),
        ("peer_state", "attacker-controlled-state", "peer state"),
        ("answer_video_direction", "attacker-controlled-state", "answer direction"),
    ):
        injected = _source()
        injected["sample"][field] = value
        with pytest.raises(ValueError, match=label):
            gate.analyze_source(
                injected,
                source_evidence=gate._source_evidence(
                    gate._encoded(injected), injected
                ),
            )
    relabeled = dict(source, session_id="other-session")
    report = gate.analyze_source(
        relabeled,
        source_evidence=gate._source_evidence(gate._encoded(relabeled), relabeled),
    )
    assert report["session_id"] == "other-session"
    with pytest.raises(ValueError, match="Wholebody49 lane"):
        gate._source_document(
            session_id="wholebody-media-test",
            runtime_lane="baseline",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            started_at_us=1_000_000,
            finished_at_us=7_000_000,
            probe_result=_probe_result(),
        )


def test_media_evidence_is_immutable_owner_private(tmp_path: Path) -> None:
    parent = tmp_path / "launcher"
    parent.mkdir(mode=0o700)
    source = _source()
    path = parent / gate.CANONICAL_SOURCE_FILENAME
    gate._write_private_json(path, source)
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="replace immutable"):
        gate._write_private_json(path, source)


def _sealed_bundle(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.chmod(0o700)
    source = _source()
    source_raw = gate._encoded(source)
    report = gate.analyze_source(
        source,
        source_evidence=gate._source_evidence(source_raw, source),
    )
    report_path = tmp_path / gate.CANONICAL_REPORT_FILENAME
    source_path = tmp_path / gate.CANONICAL_SOURCE_FILENAME
    report_path.write_bytes(gate._encoded(report))
    source_path.write_bytes(source_raw)
    report_path.chmod(0o600)
    source_path.chmod(0o600)
    return report_path, source_path


def test_sealed_media_report_requires_exact_strict_replay(tmp_path: Path) -> None:
    report_path, source_path = _sealed_bundle(tmp_path)

    validated = gate.validate_sealed_wholebody49_media_report(
        report_path,
        source_path,
        session_id="wholebody-media-test",
        runtime_lane="wholebody49-s",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
    )
    assert validated["ok"] is True

    tampered = json.loads(report_path.read_bytes())
    tampered["metrics"]["decoded_frames"] += 1
    report_path.write_bytes(gate._encoded(tampered))
    report_path.chmod(0o600)
    with pytest.raises(ValueError, match="exactly replay"):
        gate.validate_sealed_wholebody49_media_report(
            report_path,
            source_path,
            session_id="wholebody-media-test",
            runtime_lane="wholebody49-s",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
        )


def test_sealed_media_report_rejects_duplicate_source_key(tmp_path: Path) -> None:
    report_path, source_path = _sealed_bundle(tmp_path)
    source_path.write_bytes(
        source_path.read_bytes().replace(
            b'"schema_version": 1,',
            b'"schema_version": 1,\n  "schema_version": 1,',
            1,
        )
    )
    source_path.chmod(0o600)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        gate.validate_sealed_wholebody49_media_report(
            report_path,
            source_path,
            session_id="wholebody-media-test",
            runtime_lane="wholebody49-s",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
        )
