import csv
import hashlib
import json
import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from . import path_comparison as comparison
from .test_trajectory_motion_review import capture
from .test_path_review import retained
from .trajectory_motion_review import _load_provider
from . import app as app_module
from .test_browser_capture import _settings


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def paired(capture):
    state = retained(capture)
    scan = capture["scan"]
    source = _load_provider(scan, scan / "outputs/scan_outputs_manifest.json", True)
    binding = {"manifest_sha256": source["evidence"]["provider_manifest"]["sha256"],
               "coordinate_frame": source["manifest"]["coordinate_frame"]}
    alignment = scan / "alignment"
    alignment.mkdir()
    frame_binding = {"world_frame": {"frame_id": "backend_world_m", "revision": "world1"}, "transform_sha256": "a" * 64}
    target = {"camera_id": "living-room", "world_frame": "backend_world_m", "world_frame_revision": "world1",
              "camera_frame_binding": frame_binding, "coordinate_frame": "backend_world_m_stream_points"}
    write(alignment / "alignment_report.json", {"status": "passed", "target": target,
          "inputs": {"phone_source": {"output_manifest": {"sha256": binding["manifest_sha256"]}}}})
    write(alignment / "phone_ma_to_noesis_world.json", {"scale": 1., "world_from_mapanything_row_major": np.eye(4).tolist(),
          "source_output_manifest": {"sha256": binding["manifest_sha256"]}, "target_binding": target,
          "source_coordinate_frame": binding["coordinate_frame"], "target_coordinate_frame": target["coordinate_frame"]})
    state["alignment"] = {"status": "complete", "results": {"artifacts": {"report": "alignment/alignment_report.json"}}}
    companion = capture["tmp"] / "session"
    companion.mkdir()
    origin = capture["origin"]
    host_offset = 1_780_000_000_000_000_000
    mono_offset = 9_000_000_000
    probes = []
    for delta in (-1_000_000_000, 11_000_000_000):
        start = origin + delta
        probes.append({"client_probe": {"client_clock": "android.elapsedRealtimeNanos",
            "client_send_elapsed_realtime_ns": str(start), "client_receive_elapsed_realtime_ns": str(start + 10_000_000),
            "server_received_unix_ns": str(start + host_offset + 5_000_000), "server_sent_unix_ns": str(start + host_offset + 5_000_000),
            "server_received_monotonic_ns": str(start + mono_offset + 5_000_000), "server_sent_monotonic_ns": str(start + mono_offset + 5_000_000)},
            # Deliberately unrelated later acknowledgement; not a clock probe.
            "server_received_unix_ns": str(start + host_offset + 900_000_000)})
    state["companion_capture"]["phone"] = {"archive_sha256": "archive"}
    session = {"session_id": "retained-session", "camera": {"camera_id": "living-room"}, "status": "complete",
               "phone": {"scan_id": state["id"], "archive_sha256": "archive"}, "clock_exchanges": probes,
               "artifacts": {"tracking": "tracking.ndjson"}}
    write(companion / "session.json", session)
    tracking = []
    for i, pose in enumerate(source["poses"]):
        observed = (origin + round(i * .25 * 1e9) + host_offset) // 1000
        track = {"camera_id": "living-room", "source_epoch": 0, "tracker_id": 7, "tracker_lifecycle_generation": 1,
                 "stable_id": 8, "world": (pose[:3, 3] + [.1, -1.5, 0.]).tolist(), "world_valid": True,
                 "world_frame": "backend_world_m", "world_frame_revision": "world1", "world_transform_sha256": "a" * 64,
                 "world_quantity": "ground_footprint", "world_measurement_accepted": True, "world_quality": "good",
                 "world_source": "pose_depth_fused", "observed_at_us": observed, "frame_id": i, "trail_segment_id": 0}
        tracking.append({"observer_sequence": i, "message": {"type": "tracking", "camera_id": "living-room", "frame_id": i,
                                    "source_id": 0, "tracking_publication_sequence": i, "media_pts_ns": i * 250_000_000,
                                    "cohort": {"observed_at_us": observed}, "tracks": [track]}})
    (companion / "tracking.ndjson").write_text("".join(json.dumps(row) + "\n" for row in tracking))
    output = capture["tmp"] / "comparison"
    output.mkdir()
    return dict(scan_dir=scan, source=source, state=state, target_dir=scan, target_state=state,
                target_binding=binding, output_dir=output, companion_dir=companion), tracking, session


def test_registered_paired_path_is_actually_joined_and_compared(paired):
    args, _, _ = paired
    raw = args["companion_dir"].joinpath("tracking.ndjson").read_bytes()
    result = comparison.compare_review_path(**args)
    assert result["status"] == "comparison_ready"
    assert result["timing"]["status"] == "callback_clock_review"
    assert result["timing"]["synchronization_verified"] is False
    track = result["tracks"][0]
    assert track["matched_count"] == 40
    assert track["separation_m"]["median"] == pytest.approx(.1)
    assert result["person_identity_verified"] is False
    assert result["accuracy_qualified"] is False
    assert result["registration"]["fitted_to_noesis_tracks"] is False
    assert args["output_dir"].joinpath("path_comparison.csv").is_file()
    assert args["companion_dir"].joinpath("tracking.ndjson").read_bytes() == raw


def test_refined_path_preserves_original_and_uses_same_independent_transform(paired):
    args, _, _ = paired
    corrected = args["source"]["poses"].copy()
    corrected[:, 0, 3] += .04
    result = comparison.compare_review_path(**args, refined_poses=corrected)
    assert result["reference"]["position_refined"] is True
    pair = result["tracks"][0]["pairs"][0]
    assert pair["horizontal_separation_m"] == pytest.approx(.06)
    assert pair["original_horizontal_separation_m"] == pytest.approx(.1)


@pytest.mark.parametrize("wrong_world", [False, True])
def test_scaled_refinement_uses_its_own_hash_bound_registration(paired, wrong_world):
    args, _, _ = paired
    directory = args["output_dir"] / "candidate_alignment"
    directory.mkdir()
    manifest = directory / "candidate_manifest.json"
    write(manifest, {"coordinate_frame": args["source"]["manifest"]["coordinate_frame"]})
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = json.loads((args["scan_dir"] / "alignment/alignment_report.json").read_text())
    transform = json.loads((args["scan_dir"] / "alignment/phone_ma_to_noesis_world.json").read_text())
    report["inputs"]["phone_source"]["output_manifest"]["sha256"] = digest
    transform["source_output_manifest"]["sha256"] = digest
    matrix = np.eye(4); matrix[0, 3] = .04
    transform["world_from_mapanything_row_major"] = matrix.tolist()
    if wrong_world:
        for target in (report["target"], transform["target_binding"]):
            target["world_frame_revision"] = "world2"
            target["camera_frame_binding"]["world_frame"]["revision"] = "world2"
    write(directory / "alignment_report.json", report)
    write(directory / "phone_ma_to_noesis_world.json", transform)
    if wrong_world:
        with pytest.raises(ValueError, match="changed the target world"):
            comparison.compare_review_path(**args, refined_poses=args["source"]["poses"],
                refined_alignment_dir=directory, refined_output_manifest=manifest)
    else:
        result = comparison.compare_review_path(**args, refined_poses=args["source"]["poses"],
            refined_alignment_dir=directory, refined_output_manifest=manifest)
        assert result["tracks"][0]["separation_m"]["median"] == pytest.approx(.06)
        assert result["reference"]["refined_manifest"]["sha256"] == digest


@pytest.mark.parametrize("key,value,reason", [("world_frame_revision", "wrong", "world_binding_mismatch"),
    ("world_transform_sha256", "wrong", "world_binding_mismatch"), ("world_valid", False, "world_unavailable"),
    ("world_measurement_accepted", False, "not_current_ground_measurement"), ("world_source", "anchor_hold", "not_current_ground_measurement"),
    ("observed_at_us", 1, "cohort_mismatch")])
def test_incompatible_or_predicted_rows_are_not_reference_evidence(paired, key, value, reason):
    args, rows, _ = paired
    for row in rows:
        row["message"]["tracks"][0][key] = value
    args["companion_dir"].joinpath("tracking.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = comparison.compare_review_path(**args)
    assert result["status"] == "reference_ready_no_matching_world"
    assert result["excluded_observations"][reason] == 40


def test_missing_pair_keeps_registered_reference(paired):
    args, _, _ = paired
    args["companion_dir"] = None
    result = comparison.compare_review_path(**args)
    assert result["status"] == "reference_ready_comparison_needs_evidence"
    assert len(result["reference"]["poses"]) == 40


def test_alignment_hash_and_reference_revision_do_not_silently_mix(paired):
    args, _, _ = paired
    path = args["scan_dir"] / "alignment/phone_ma_to_noesis_world.json"
    value = json.loads(path.read_text())
    value["source_output_manifest"]["sha256"] = "changed"
    write(path, value)
    with pytest.raises(ValueError, match="exact selected provider"):
        comparison.compare_review_path(**args)
    assert not args["output_dir"].joinpath("room_path_reference.json").exists()


def test_native_clock_never_extrapolates_or_uses_outer_acknowledgement(paired):
    args, _, session = paired
    native = comparison._load_native(args["scan_dir"], args["source"])
    times, result = comparison._native_clock(session, native)
    assert times[0] == int(native["times_ns"][0]) + 1_780_000_000_000_000_000
    session["clock_exchanges"].pop()
    with pytest.raises(ValueError, match="two ordered"):
        comparison._native_clock(session, native)


def test_gaps_identity_changes_and_lifecycles_break_the_drawn_path(paired):
    args, rows, _ = paired
    rows[10]["message"]["tracks"][0]["world_valid"] = False
    rows[20]["message"]["tracks"][0]["stable_id"] = 9
    for row in rows[30:]:
        row["message"]["tracks"][0]["tracker_lifecycle_generation"] = 2
    args["companion_dir"].joinpath("tracking.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = comparison.compare_review_path(**args)
    assert len(result["tracks"]) == 2
    first = {p["provider_index"]: p for p in result["tracks"][0]["pairs"]}
    assert 10 not in first
    assert first[11]["break_before"] is True
    assert first[20]["break_before"] is True
    assert result["tracks"][1]["pairs"][0]["break_before"] is True


def light_fixture(args, rows, session):
    path = args["companion_dir"] / "temporal.json"
    native = comparison._load_native(args["scan_dir"], args["source"])
    temporal = {"schema": "noesis.offline.light_cue_temporal_alignment.v1", "scan_id": args["state"]["id"],
                "camera_id": "living-room", "companion_session_id": "retained-session",
                "input_identity": {"archive_sha256": "archive"},
                "light_alignment": {"equation": "static_mkv_pts_s = phone_mp4_pts_s + offset_s", "clock_rate": 1.0,
                                    "phone_native_origin_ns": native["origin_ns"], "offset_s": 1., "local_review_allowance_s": .1},
                "static_ds9_mapping": {"static_decoded_frame_index_equals_ds9_frame_id_minus": 0},
                "outputs": {"canonical_channels": "light.csv"}}
    write(path, temporal)
    mapped = []
    for i, row in enumerate(rows):
        msg = row["message"]
        mapped.append({"observer_sequence": i, "message_type": "tracking", "source_id": 0, "source_epoch": "",
                       "ds9_frame_id": i, "tracking_publication_sequence": i, "complete_saved_cohort": "True",
                       "recorded_static_frame_exists": "True", "mapping_status": "exact_cohort_to_unique_differential_pts_pair",
                       "observed_at_us": msg["cohort"]["observed_at_us"], "exact_ds9_tracking_media_pts_ns": msg["media_pts_ns"],
                       "derived_static_frame_index": i, "static_decoded_pts_s": i * .25 + 1, "estimated_phone_pts_s": i * .25})
    with path.with_name("light.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapped[0])); writer.writeheader(); writer.writerows(mapped)
    return path, temporal


def test_explicit_light_evidence_joins_exact_cohort_and_does_not_need_http_probes(paired):
    args, rows, session = paired
    path, _ = light_fixture(args, rows, session)
    session["clock_exchanges"] = []
    write(args["companion_dir"] / "session.json", session)
    result = comparison.compare_review_path(**args, temporal_alignment_path=path)
    assert result["status"] == "comparison_ready"
    assert result["timing"]["status"] == "light_cue_review"
    assert result["timing"]["clock_drift_verified"] is False
    assert result["tracks"][0]["matched_count"] == 40


def test_wrong_light_archive_fails_without_silently_using_callback_timing(paired):
    args, rows, session = paired
    path, temporal = light_fixture(args, rows, session)
    temporal["input_identity"]["archive_sha256"] = "different"
    write(path, temporal)
    result = comparison.compare_review_path(**args, temporal_alignment_path=path)
    assert result["status"] == "reference_ready_comparison_needs_evidence"
    assert "exact paired phone archive" in result["reason"]
    assert result["timing"]["status"] == "not_available"


def test_light_frame_map_cannot_join_a_different_tracking_publication(paired):
    args, rows, session = paired
    path, _ = light_fixture(args, rows, session)
    rows[0]["message"]["tracking_publication_sequence"] = 999
    args["companion_dir"].joinpath("tracking.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = comparison.compare_review_path(**args, temporal_alignment_path=path)
    assert result["status"] == "reference_ready_comparison_needs_evidence"
    assert "exact retained tracking cohort" in result["reason"]


@pytest.mark.parametrize("qualified", [False, True])
def test_real_api_worker_serves_the_registered_comparison_artifact(paired, monkeypatch, qualified):
    args, _, session = paired
    application = app_module.create_app(_settings(args["output_dir"] / "service"))
    service = application.state.phone_scan_service
    vio_calls = []
    def vio_worker(scan_id):
        vio_calls.append(scan_id)
        service.update_state(scan_id, vio={"status": "complete", "results": {}})
    monkeypatch.setattr(service, "_vio_worker", vio_worker)
    scan_id = args["state"]["id"]
    source_root = service.scan_dir(scan_id)
    shutil.copytree(args["scan_dir"], source_root)
    session_id = "companion-20260921-120000-abcdef12"
    args["state"]["companion_capture"]["session_id"] = session["session_id"] = session_id
    args["state"]["capture"]["metric_vio_allowed"] = qualified
    session_root = service.companion_capture.session_root / session_id
    shutil.copytree(args["companion_dir"], session_root)
    write(session_root / "session.json", session)
    service._write_state_unlocked(scan_id, args["state"])
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{scan_id}/path-review")
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = client.get(f"/api/scans/{scan_id}").json()
            if state["path_review"]["status"] in {"complete", "failed"}:
                break
            time.sleep(.01)
        assert state["path_review"]["status"] == "complete", state["path_review"]
        result = state["path_review"]["results"]
        assert result["path_comparison"]["matched_count"] == 40
        assert result["paired_noesis"]["clocks_joined"] is True
        artifact = client.get(result["artifact_urls"]["comparison"])
        assert artifact.status_code == 200
        assert artifact.json()["tracks"][0]["matched_count"] == 40
        assert client.get(result["artifact_urls"]["comparison_csv"]).status_code == 200
        assert vio_calls == ([scan_id] if qualified else [])
