#!/usr/bin/env python3
"""Seal a privacy-safe decoded-media gate for a Wholebody49 runtime session."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_create_private_file,
    read_private_file,
    require_fresh_private_file_bundle,
)
from noesis_core.strict_json import strict_json_loads  # noqa: E402

PROBE_SCRIPT = REPO_ROOT / "scripts" / "webrtc_gateway_smoke_test.py"

CONTRACT = "noesis.ds9.wholebody49_media_decode_gate"
CONTRACT_VERSION = 1
SOURCE_CONTRACT = "noesis.ds9.wholebody49-media-decode-source"
CANONICAL_REPORT_FILENAME = "wholebody49-media-decode.json"
CANONICAL_SOURCE_FILENAME = "wholebody49-media-decode-source.json"
SOURCE_PRIVACY_POLICY = {
    "payload_policy": "aggregate_transport_and_decode_counters_only",
    "image_frames": "absent",
    "sdp_and_ice": "absent",
    "network_addresses": "absent",
    "secrets": "absent",
}
MAX_JSON_BYTES = 256 * 1024
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
WHOLEBODY_LANES = frozenset({"wholebody49-s", "wholebody49-x"})
ICE_STATES = frozenset(
    {"new", "checking", "connected", "completed", "failed", "disconnected", "closed"}
)
PEER_STATES = frozenset(
    {"new", "connecting", "connected", "disconnected", "failed", "closed"}
)


def _closed_state(value: object, allowed: frozenset[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "unexpected"


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:  # noqa: E721
        raise ValueError(f"{label} must be an exact non-negative integer")
    return value


def _validate_identity(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
) -> tuple[str, str, str, str]:
    normalized_session = str(session_id).strip().lower()
    normalized_lane = str(runtime_lane).strip().lower()
    instance_id = str(runtime_instance_id)
    run_id = str(runtime_run_id)
    if SESSION_RE.fullmatch(normalized_session) is None:
        raise ValueError("session_id is invalid")
    if normalized_lane not in WHOLEBODY_LANES:
        raise ValueError("decoded-media evidence requires a reviewed Wholebody49 lane")
    if (
        RUNTIME_ID_RE.fullmatch(instance_id) is None
        or RUNTIME_ID_RE.fullmatch(run_id) is None
    ):
        raise ValueError("runtime instance/run identity is invalid")
    return normalized_session, normalized_lane, instance_id, run_id


def _source_document(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    started_at_us: int,
    finished_at_us: int,
    probe_result: Mapping[str, object],
) -> dict[str, object]:
    session_id, runtime_lane, runtime_instance_id, runtime_run_id = (
        _validate_identity(
            session_id=session_id,
            runtime_lane=runtime_lane,
            runtime_instance_id=runtime_instance_id,
            runtime_run_id=runtime_run_id,
        )
    )
    started = _exact_nonnegative_int(started_at_us, "started_at_us")
    finished = _exact_nonnegative_int(finished_at_us, "finished_at_us")
    if started <= 0 or finished < started:
        raise ValueError("decoded-media observation window is invalid")
    expected_keys = {
        "ok",
        "answer_video_direction",
        "ice_state",
        "peer_state",
        "rtp_packets",
        "decoded_frames",
        "saw_src_pad",
        "src_caps",
        "min_rtp",
        "min_decoded",
        "rtsp_decoded_frames",
        "rtsp_min_decoded",
    }
    if set(probe_result) != expected_keys:
        raise ValueError("WebRTC probe result schema drifted")
    answer_direction = str(probe_result.get("answer_video_direction") or "")
    ice_state = _closed_state(probe_result.get("ice_state"), ICE_STATES)
    peer_state = _closed_state(probe_result.get("peer_state"), PEER_STATES)
    src_caps = str(probe_result.get("src_caps") or "")
    if answer_direction not in {"sendonly", "sendrecv", "recvonly", "inactive", "unknown"}:
        raise ValueError("WebRTC answer direction is invalid")
    rtp_packets = _exact_nonnegative_int(probe_result.get("rtp_packets"), "rtp_packets")
    decoded_frames = _exact_nonnegative_int(
        probe_result.get("decoded_frames"), "decoded_frames"
    )
    min_rtp = _exact_nonnegative_int(probe_result.get("min_rtp"), "min_rtp")
    min_decoded = _exact_nonnegative_int(
        probe_result.get("min_decoded"), "min_decoded"
    )
    rtsp_decoded_frames = _exact_nonnegative_int(
        probe_result.get("rtsp_decoded_frames"), "rtsp_decoded_frames"
    )
    rtsp_min_decoded = _exact_nonnegative_int(
        probe_result.get("rtsp_min_decoded"), "rtsp_min_decoded"
    )
    if type(probe_result.get("ok")) is not bool or type(  # noqa: E721
        probe_result.get("saw_src_pad")
    ) is not bool:
        raise ValueError("WebRTC boolean evidence is invalid")
    return {
        "schema_version": 1,
        "contract": SOURCE_CONTRACT,
        "contract_version": 1,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": runtime_instance_id,
        "runtime_run_id": runtime_run_id,
        "privacy": dict(SOURCE_PRIVACY_POLICY),
        "started_at_us": started,
        "finished_at_us": finished,
        "sample": {
            "probe_ok": probe_result["ok"],
            "answer_video_direction": answer_direction,
            "ice_state": ice_state,
            "peer_state": peer_state,
            "rtp_packets": rtp_packets,
            "decoded_frames": decoded_frames,
            "saw_src_pad": probe_result["saw_src_pad"],
            "h264_caps_observed": "H264" in src_caps.upper(),
            "min_rtp": min_rtp,
            "min_decoded": min_decoded,
            "rtsp_decoded_frames": rtsp_decoded_frames,
            "rtsp_min_decoded": rtsp_min_decoded,
        },
    }


def _encoded(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _source_evidence(
    encoded: bytes, document: Mapping[str, object]
) -> dict[str, object]:
    return {
        "filename": CANONICAL_SOURCE_FILENAME,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "started_at_us": document.get("started_at_us"),
        "finished_at_us": document.get("finished_at_us"),
    }


def analyze_source(
    document: Mapping[str, object],
    *,
    source_evidence: Mapping[str, object],
) -> dict[str, object]:
    expected_root = {
        "schema_version",
        "contract",
        "contract_version",
        "session_id",
        "runtime_lane",
        "runtime_instance_id",
        "runtime_run_id",
        "privacy",
        "started_at_us",
        "finished_at_us",
        "sample",
    }
    if set(document) != expected_root:
        raise ValueError("decoded-media source root schema drifted")
    if (
        document.get("schema_version") != 1
        or type(document.get("schema_version")) is not int  # noqa: E721
        or document.get("contract") != SOURCE_CONTRACT
        or document.get("contract_version") != 1
        or type(document.get("contract_version")) is not int  # noqa: E721
        or document.get("privacy") != SOURCE_PRIVACY_POLICY
    ):
        raise ValueError("decoded-media source envelope drifted")
    session_id, runtime_lane, instance_id, run_id = _validate_identity(
        session_id=str(document.get("session_id") or ""),
        runtime_lane=str(document.get("runtime_lane") or ""),
        runtime_instance_id=str(document.get("runtime_instance_id") or ""),
        runtime_run_id=str(document.get("runtime_run_id") or ""),
    )
    started = _exact_nonnegative_int(document.get("started_at_us"), "started_at_us")
    finished = _exact_nonnegative_int(document.get("finished_at_us"), "finished_at_us")
    if started <= 0 or finished < started:
        raise ValueError("decoded-media source window is invalid")
    sample = document.get("sample")
    expected_sample = {
        "probe_ok",
        "answer_video_direction",
        "ice_state",
        "peer_state",
        "rtp_packets",
        "decoded_frames",
        "saw_src_pad",
        "h264_caps_observed",
        "min_rtp",
        "min_decoded",
        "rtsp_decoded_frames",
        "rtsp_min_decoded",
    }
    if not isinstance(sample, Mapping) or set(sample) != expected_sample:
        raise ValueError("decoded-media sample schema drifted")
    if sample.get("answer_video_direction") not in {
        "sendonly",
        "sendrecv",
        "recvonly",
        "inactive",
        "unknown",
    }:
        raise ValueError("decoded-media answer direction is outside its closed enum")
    if sample.get("ice_state") not in ICE_STATES | {"unexpected"}:
        raise ValueError("decoded-media ICE state is outside its closed enum")
    if sample.get("peer_state") not in PEER_STATES | {"unexpected"}:
        raise ValueError("decoded-media peer state is outside its closed enum")
    for key in ("probe_ok", "saw_src_pad", "h264_caps_observed"):
        if type(sample.get(key)) is not bool:  # noqa: E721
            raise ValueError(f"decoded-media sample {key} must be boolean")
    rtp_packets = _exact_nonnegative_int(sample.get("rtp_packets"), "rtp_packets")
    decoded_frames = _exact_nonnegative_int(
        sample.get("decoded_frames"), "decoded_frames"
    )
    min_rtp = _exact_nonnegative_int(sample.get("min_rtp"), "min_rtp")
    min_decoded = _exact_nonnegative_int(sample.get("min_decoded"), "min_decoded")
    rtsp_decoded_frames = _exact_nonnegative_int(
        sample.get("rtsp_decoded_frames"), "rtsp_decoded_frames"
    )
    rtsp_min_decoded = _exact_nonnegative_int(
        sample.get("rtsp_min_decoded"), "rtsp_min_decoded"
    )
    checks = {
        "probe_completed": sample.get("probe_ok") is True,
        "connection_states_accepted": sample.get("ice_state")
        in {"connected", "completed"}
        and sample.get("peer_state") == "connected",
        "answer_sends_video": sample.get("answer_video_direction")
        in {"sendonly", "sendrecv"},
        "rtp_threshold_met": rtp_packets >= min_rtp,
        "decoded_frame_threshold_met": decoded_frames >= min_decoded
        and decoded_frames > 0,
        "decoded_h264_pad_observed": sample.get("saw_src_pad") is True
        and sample.get("h264_caps_observed") is True,
        "direct_rtsp_decode_threshold_met": rtsp_decoded_frames
        >= rtsp_min_decoded
        and rtsp_decoded_frames > 0,
    }
    ok = all(checks.values())
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": instance_id,
        "runtime_run_id": run_id,
        "ok": ok,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "metrics": {
            "rtp_packets": rtp_packets,
            "decoded_frames": decoded_frames,
            "rtsp_decoded_frames": rtsp_decoded_frames,
            "observation_duration_seconds": (finished - started) / 1_000_000.0,
        },
        "thresholds": {
            "minimum_rtp_packets": min_rtp,
            "minimum_decoded_frames": min_decoded,
            "minimum_rtsp_decoded_frames": rtsp_min_decoded,
        },
        "source_evidence": dict(source_evidence),
        "errors": [],
    }


def _read_private_json_object(
    path: Path,
    *,
    expected_filename: str,
    label: str,
) -> tuple[bytes, Mapping[str, object]]:
    candidate = Path(path).expanduser().absolute()
    if candidate.name != expected_filename:
        raise ValueError(f"{label} filename is not canonical")
    raw = read_private_file(candidate, label=label, max_bytes=MAX_JSON_BYTES)
    if not raw:
        raise ValueError(f"{label} is empty")
    document = strict_json_loads(raw, label=label)
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw, document


def validate_sealed_wholebody49_media_report(
    report_path: Path,
    source_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
) -> Mapping[str, object]:
    """Strictly replay decoded-media evidence from its owner-private source."""

    if Path(report_path).expanduser().absolute().parent != Path(
        source_path
    ).expanduser().absolute().parent:
        raise ValueError(
            "Wholebody49 media report and source must share one evidence directory"
        )
    report_raw, report = _read_private_json_object(
        report_path,
        expected_filename=CANONICAL_REPORT_FILENAME,
        label="Wholebody49 media report",
    )
    source_raw, source = _read_private_json_object(
        source_path,
        expected_filename=CANONICAL_SOURCE_FILENAME,
        label="Wholebody49 media source",
    )
    if (
        source.get("session_id") != str(session_id).strip().lower()
        or source.get("runtime_lane") != str(runtime_lane).strip().lower()
        or source.get("runtime_instance_id") != str(runtime_instance_id)
        or source.get("runtime_run_id") != str(runtime_run_id)
    ):
        raise ValueError("Wholebody49 media source identity binding drifted")
    if source_raw != _encoded(source):
        raise ValueError("Wholebody49 media source is not canonical")
    source_evidence = _source_evidence(source_raw, source)
    recomputed = analyze_source(source, source_evidence=source_evidence)
    if report != recomputed or report_raw != _encoded(recomputed):
        raise ValueError(
            "Wholebody49 media report does not exactly replay from sealed source"
        )
    if report.get("ok") is not True or report.get("status") != "pass":
        raise ValueError("Wholebody49 media report is not a passing exact replay")
    return report


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    destination = path.expanduser().absolute()
    if destination.name not in {
        CANONICAL_REPORT_FILENAME,
        CANONICAL_SOURCE_FILENAME,
    }:
        raise ValueError("decoded-media output filename is not canonical")
    raw = _encoded(payload)
    try:
        atomic_create_private_file(
            destination,
            raw,
            label=f"immutable decoded-media evidence {destination.name}",
            max_bytes=MAX_JSON_BYTES,
        )
    except PrivatePathError as exc:
        raise ValueError(str(exc)) from exc


def _probe_rtsp_decode(
    uri: str, *, duration_s: float, minimum_decoded_frames: int
) -> int:
    parsed = urlparse(str(uri))
    if (
        parsed.scheme != "rtsp"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or (parsed.port or 554) != 8554
        or parsed.path != "/mosaic"
    ):
        raise ValueError("direct RTSP decode requires the canonical loopback mosaic")
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    escaped = str(uri).replace("\\", "\\\\").replace('"', '\\"')
    pipeline = Gst.parse_launch(
        f'rtspsrc location="{escaped}" latency=100 ! rtph264depay ! h264parse '
        "! avdec_h264 ! fakesink name=decode_sink sync=false signal-handoffs=true"
    )
    sink = pipeline.get_by_name("decode_sink")
    if sink is None:
        raise RuntimeError("direct RTSP decode sink was not created")
    decoded_frames = 0

    def on_handoff(_sink: object, _buffer: object, _pad: object) -> None:
        nonlocal decoded_frames
        decoded_frames += 1

    sink.connect("handoff", on_handoff)
    state = pipeline.set_state(Gst.State.PLAYING)
    if state == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("direct RTSP decode pipeline did not enter PLAYING")
    bus = pipeline.get_bus()
    deadline = time.monotonic() + float(duration_s)
    try:
        while time.monotonic() < deadline:
            message = bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if message is None:
                continue
            if message.type == Gst.MessageType.ERROR:
                error, _debug = message.parse_error()
                raise RuntimeError(f"direct RTSP decode failed: {error}")
            if message.type == Gst.MessageType.EOS:
                break
    finally:
        pipeline.set_state(Gst.State.NULL)
    if decoded_frames < int(minimum_decoded_frames):
        raise RuntimeError(
            "direct RTSP decode did not reach its decoded-frame threshold"
        )
    return decoded_frames


def _probe(args: argparse.Namespace) -> tuple[int, int, Mapping[str, object]]:
    started_at_us = time.time_ns() // 1_000
    rtsp_decoded_frames = _probe_rtsp_decode(
        str(args.rtsp_url),
        duration_s=float(args.rtsp_duration),
        minimum_decoded_frames=int(args.rtsp_min_decoded),
    )
    command = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--ws",
        str(args.ws),
        "--duration",
        str(args.duration),
        "--pt",
        str(args.pt),
        "--min-rtp",
        str(args.min_rtp),
        "--min-decoded",
        str(args.min_decoded),
        "--auth-token-file",
        str(args.auth_token_file),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(args.duration) + 30.0,
    )
    finished_at_us = time.time_ns() // 1_000
    if completed.returncode != 0:
        raise RuntimeError(f"decoded-media probe exited {completed.returncode}")
    try:
        payload = strict_json_loads(
            completed.stdout,
            label="Wholebody49 WebRTC probe output",
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("decoded-media probe did not return one JSON document") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("decoded-media probe result root is not an object")
    combined = dict(payload)
    combined["rtsp_decoded_frames"] = rtsp_decoded_frames
    combined["rtsp_min_decoded"] = int(args.rtsp_min_decoded)
    return started_at_us, finished_at_us, combined


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-lane", choices=sorted(WHOLEBODY_LANES), required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--rtsp-url", default="rtsp://127.0.0.1:8554/mosaic")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--rtsp-duration", type=float, default=6.0)
    parser.add_argument("--pt", type=int, default=103)
    parser.add_argument("--min-rtp", type=int, default=10)
    parser.add_argument("--min-decoded", type=int, default=1)
    parser.add_argument("--rtsp-min-decoded", type=int, default=1)
    parser.add_argument("--auth-token-file", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        require_fresh_private_file_bundle(
            (args.source_out, args.out),
            label="Wholebody49 decoded-media evidence bundle",
        )
        if not float(args.duration) > 0 or not float(args.rtsp_duration) > 0:
            raise ValueError("media probe durations must be > 0")
        if (
            int(args.min_rtp) <= 0
            or int(args.min_decoded) <= 0
            or int(args.rtsp_min_decoded) <= 0
        ):
            raise ValueError("decoded-media thresholds must be positive")
        started_at_us, finished_at_us, probe_result = _probe(args)
        source = _source_document(
            session_id=args.session_id,
            runtime_lane=args.runtime_lane,
            runtime_instance_id=args.runtime_instance_id,
            runtime_run_id=args.runtime_run_id,
            started_at_us=started_at_us,
            finished_at_us=finished_at_us,
            probe_result=probe_result,
        )
        source_raw = _encoded(source)
        source_metadata = _source_evidence(source_raw, source)
        report = analyze_source(source, source_evidence=source_metadata)
        _write_private_json(args.source_out, source)
        _write_private_json(args.out, report)
    except Exception as exc:
        print(f"[FAIL] decoded-media evidence unavailable: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
