"""Compare an independently registered phone path with retained Noesis records.

This is an offline diagnostic, not a second world producer. Registration comes
only from the existing geometry alignment, never from the person being tested.
The native HTTP clock bridge is explicitly approximate: DS9 callback latency
and the phone-to-body offset are not made to disappear by fitting the paths.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory_motion_review import _integer, _json, _load_native, _read, _validate_poses

SCHEMA = "noesis.phone_scan.path_comparison.v1"
MAX_TRACKING_BYTES = 256 * 1024 * 1024
MAX_TRACKING_ROWS = 200_000
MAX_TRACKS = 32
MAX_MATCH_GAP_S = 0.15


def _relative_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a retained relative artifact path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Artifact is missing or outside its retained root")
    return path


def _evidence(path: Path, raw: bytes) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _alignment(root: Path, state: dict, manifest_hash: str, coordinate_frame: str,
               directory: Path | None = None) -> dict:
    if directory is None:
        job = state.get("alignment") or {}
        if job.get("status") != "complete":
            raise ValueError("Align this reconstruction to its static camera before comparing paths")
        report_path = _relative_file(root, ((job.get("results") or {}).get("artifacts") or {}).get("report"))
        directory = report_path.parent
    report_path = directory / "alignment_report.json"
    report, raw = _json(report_path)
    transform_path = directory / "phone_ma_to_noesis_world.json"
    transform, transform_raw = _json(transform_path)
    target = report.get("target") or {}
    binding = transform.get("target_binding") or {}
    if (report.get("status") != "passed"
            or report.get("inputs", {}).get("phone_source", {}).get("output_manifest", {}).get("sha256") != manifest_hash
            or transform.get("source_output_manifest", {}).get("sha256") != manifest_hash
            or transform.get("source_coordinate_frame") != coordinate_frame
            or transform.get("scale") != 1.0
            or target.get("coordinate_frame") != "backend_world_m_stream_points"
            or transform.get("target_coordinate_frame") != target.get("coordinate_frame")):
        raise ValueError("Passed metric geometry alignment must bind the exact selected provider")
    for key in ("world_frame", "world_frame_revision", "camera_frame_binding"):
        if not target.get(key) or binding.get(key) != target[key]:
            raise ValueError(f"Alignment {key} binding is absent or inconsistent")
    frame = target["camera_frame_binding"]
    if (target["world_frame"] != "backend_world_m" or not frame.get("transform_sha256")
            or frame.get("world_frame") != {"frame_id": target["world_frame"], "revision": target["world_frame_revision"]}):
        raise ValueError("Path comparison needs an explicit revision-bound backend Y-up world")
    matrix = _validate_poses([transform.get("world_from_mapanything_row_major")], 1)[0]
    return {"matrix": matrix, "target": target,
            "evidence": {"report": _evidence(report_path, raw), "transform": _evidence(transform_path, transform_raw)}}


def _pcf_registration(reference: dict, registered_target: dict) -> dict:
    """Check the already-admitted PCF edge against independent walk alignment.

    A Scene Prior is already a registered room artifact, not another phone
    provider with fabricated poses. Its catalog edge must be the same edge the
    paired camera actually published. No point/track fitting happens here.
    """
    selection = reference["selection"]
    edge = reference["frame_binding"]
    captured = registered_target.get("camera_frame_binding") or {}
    if (registered_target.get("camera_id") != selection["camera_id"]
            or registered_target.get("world_frame") != edge["target_frame"]["frame_id"]
            or registered_target.get("world_frame_revision") != edge["target_frame"]["revision"]):
        raise ValueError("Selected PCF and walk registration have different camera/world bindings")
    expected = {
        "contract": "noesis.calibration.frame_binding",
        "contract_version": edge["contract_version"],
        "calibration_frame": edge["source_frame"],
        "world_frame": edge["target_frame"],
        "transform_sha256": edge["target_from_source_sha256"],
        "target_from_calibration_col_major": edge["target_from_source_col_major"],
        "camera_calibration_sha256": edge["source_camera_calibration_sha256"],
        "world_alignment_sha256": edge["source_world_alignment_sha256"],
        "target_revision_id": edge["target_revision_id"],
        "scene_prior_id": edge.get("artifact_revision_id") or edge["target_frame"]["revision"],
        "calibration_floor_plane": {k: edge["source_floor_plane"][k] for k in ("normal", "offset_m")},
        "world_floor_plane": {k: edge["target_floor_plane"][k] for k in ("normal", "offset_m")},
    }
    if edge.get("contract_version") == 2:
        from noesis_core.contracts.scene_prior import ScenePriorFrameBinding
        normalized_edge = ScenePriorFrameBinding.model_validate(edge).model_dump(mode="json")
        expected.update(artifact_revision_id=edge["artifact_revision_id"],
                        camera_calibration_physical_sha256=edge["source_camera_calibration_physical_sha256"],
                        world_alignment_physical_sha256=edge["source_world_alignment_physical_sha256"],
                        metric_frame=normalized_edge["metric_frame"],
                        presentation=normalized_edge["presentation"])
    for key, value in expected.items():
        if captured.get(key) != value:
            raise ValueError(f"Selected PCF {key} differs from the captured camera binding")
    return {"target": registered_target,
            "evidence": {"kind": "scene_prior_pcf", "selection": selection,
                         "source": reference["evidence"], "frame_binding": edge}}


def _native_clock(session: dict, native: dict) -> tuple[list[int | None], dict]:
    """Bridge native monotonic camera times to host wall time; do not fit motion."""
    probes = []
    for exchange in session.get("clock_exchanges") or []:
        p = exchange.get("client_probe") or {}
        if p.get("client_clock") != "android.elapsedRealtimeNanos":
            continue
        try:
            a = _integer(p.get("client_send_elapsed_realtime_ns"), "probe send")
            b = _integer(p.get("client_receive_elapsed_realtime_ns"), "probe receive")
            c = _integer(p.get("server_received_unix_ns"), "probe host receive")
            d = _integer(p.get("server_sent_unix_ns"), "probe host send")
            m = _integer(p.get("server_received_monotonic_ns"), "probe host monotonic")
            n = _integer(p.get("server_sent_monotonic_ns"), "probe host monotonic")
        except ValueError:
            continue
        # The embedded response belongs to this round trip. The outer session
        # receipt occurs later and MUST NOT be substituted for that response.
        if not 0 <= d - c <= b - a <= 500_000_000 or n < m:
            continue
        probes.append((a, b, n - b, m - a, c - m, (b - a - (d - c)) / 2e9))
    probes.sort()
    if len(probes) < 2 or any(b[0] <= a[0] for a, b in zip(probes, probes[1:])):
        raise ValueError("At least two ordered native clock exchanges are required for the review join")
    if max(p[4] for p in probes) - min(p[4] for p in probes) > 50_000_000:
        raise ValueError("Host wall clock changed during the paired recording")
    lower, upper = max(p[2] for p in probes), min(p[3] for p in probes)
    if lower > upper:
        raise ValueError("Native clock probes do not support a common unit-rate offset; measured timing is needed")
    theta = (lower + upper) // 2
    wall_offsets = sorted(p[4] for p in probes)
    wall = wall_offsets[len(wall_offsets) // 2]
    joined = []
    for raw_stamp in native["times_ns"]:
        stamp = int(raw_stamp)
        if not probes[0][0] <= stamp <= probes[-1][1]:
            joined.append(None)
        else:
            joined.append(stamp + theta + wall)
    return joined, {"status": "callback_clock_review", "method": "native_http_unit_rate_offset_interval",
                    "probe_count": len(probes), "network_half_roundtrip_max_s": max(p[5] for p in probes),
                    "conditional_offset_half_width_s": (upper - lower) / 2e9,
                    "phone_to_host_monotonic_offset_ns": str(theta), "host_unix_minus_monotonic_ns": str(wall),
                    "clock_rate_assumed": 1.0, "clock_drift_verified": False,
                    "outside_probe_interval_count": sum(t is None for t in joined),
                    "extrapolated": False, "synchronization_verified": False,
                    "static_acquisition_latency_known": False,
                    "limitation": "HTTP uncertainty is not an end-to-end acquisition error bound; DS9 timestamps are callback observations."}


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (float, int) for v in value):
        return None
    try:
        point = [float(v) for v in value]
    except (ValueError, OverflowError):
        return None
    return point if all(math.isfinite(v) for v in point) else None


def _tracks(path: Path, camera_id: str, target: dict) -> tuple[dict, dict, dict]:
    if path.stat().st_size > MAX_TRACKING_BYTES:
        raise ValueError("Paired tracking exceeds the 256 MiB review budget")
    groups, rejected, digest = {}, Counter(), hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            if index >= MAX_TRACKING_ROWS or len(line) > 4 * 1024 * 1024:
                raise ValueError("Paired tracking exceeds the bounded record budget")
            digest.update(line)
            envelope = json.loads(line)
            msg = envelope.get("message") or {}
            if msg.get("type") != "tracking" or msg.get("camera_id") != camera_id:
                continue
            for track in msg.get("tracks") or []:
                consumed += 1
                if consumed > MAX_TRACKING_ROWS:
                    raise ValueError("Paired tracking exceeds the bounded observation budget")
                if track.get("camera_id", camera_id) != camera_id:
                    rejected["camera_mismatch"] += 1
                    continue
                generation = track.get("tracker_lifecycle_generation")
                tracker_id = track.get("tracker_id")
                if type(generation) is not int or type(tracker_id) is not int:
                    rejected["missing_lifecycle_identity"] += 1
                    continue
                # A StableID can span tracker lifecycles; never connect those
                # segments or choose a person by closeness to the phone path.
                epoch = track.get("source_epoch")
                if type(epoch) is not int:
                    rejected["missing_source_epoch"] += 1
                    continue
                key = f"{camera_id}:{epoch}:{tracker_id}:{generation}"
                if key not in groups:
                    if len(groups) >= MAX_TRACKS:
                        rejected["track_budget"] += 1
                        continue
                    groups[key] = {"track_key": key, "tracker_id": tracker_id, "lifecycle_generation": generation, "source_epoch": epoch,
                                   "stable_ids": set(), "rows": [], "invalid_world_count": 0}
                group = groups[key]
                stable = track.get("stable_id")
                if type(stable) is int and stable > 0:
                    group["stable_ids"].add(stable)
                point = _point(track.get("world"))
                if not track.get("world_valid") or point is None:
                    rejected["world_unavailable"] += 1
                    group["invalid_world_count"] += 1
                    continue
                if (track.get("world_frame") != target["world_frame"]
                        or track.get("world_frame_revision") != target["world_frame_revision"]
                        or track.get("world_transform_sha256") != target["camera_frame_binding"]["transform_sha256"]):
                    rejected["world_binding_mismatch"] += 1
                    continue
                if (track.get("world_quantity") != "ground_footprint"
                        or track.get("world_measurement_accepted") is not True
                        or track.get("world_quality") in {"held", "invalid"}
                        or track.get("world_source") in {"cv_prediction", "image_motion_prediction", "anchor_hold"}):
                    rejected["not_current_ground_measurement"] += 1
                    continue
                try:
                    observed = _integer(track.get("observed_at_us", msg.get("observed_at_us")), "Noesis observation time")
                    cohort_time = _integer(msg.get("cohort", {}).get("observed_at_us"), "cohort time")
                    frame_id = _integer(track.get("frame_id"), "Noesis frame identity")
                except ValueError:
                    rejected["missing_cohort_time"] += 1
                    continue
                if observed != cohort_time or frame_id != msg.get("frame_id"):
                    rejected["cohort_mismatch"] += 1
                    continue
                group["rows"].append({"time_ns": observed * 1000, "world": point, "frame_id": frame_id,
                                      "stable_id": stable, "world_source": track.get("world_source"),
                                      "observer_sequence": envelope.get("observer_sequence"),
                                      "source_id": msg.get("source_id"), "source_epoch": epoch,
                                      "publication_sequence": msg.get("tracking_publication_sequence"),
                                      "media_pts_ns": msg.get("media_pts_ns"),
                                      "trail_break_required": bool(track.get("trail_break_required")),
                                      "trail_segment_id": track.get("trail_segment_id")})
    for group in groups.values():
        group["stable_ids"] = sorted(group["stable_ids"])
        group["rows"].sort(key=lambda r: r["time_ns"])
        if any(b["time_ns"] <= a["time_ns"] for a, b in zip(group["rows"], group["rows"][1:])):
            raise ValueError("Repeated/reversed Noesis observation time within a track lifecycle")
    return groups, dict(rejected), {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _light_clock(path: Path, session: dict, state: dict, native: dict, groups: dict) -> tuple[list[int], dict]:
    """Consume explicitly supplied light evidence, with exact retained cohort joins."""
    temporal, raw = _json(path)
    if (temporal.get("schema") != "noesis.offline.light_cue_temporal_alignment.v1"
            or temporal.get("scan_id") != state["id"] or temporal.get("camera_id") != session["camera_id"]
            or temporal.get("companion_session_id") != session["session_id"]
            or temporal.get("input_identity", {}).get("archive_sha256") != session["phone"]["archive_sha256"]):
        raise ValueError("Light timing must bind the exact paired phone archive, scan and camera")
    light, mapping = temporal.get("light_alignment") or {}, temporal.get("static_ds9_mapping") or {}
    if (light.get("equation") != "static_mkv_pts_s = phone_mp4_pts_s + offset_s" or light.get("clock_rate") != 1.0
            or light.get("phone_native_origin_ns") != native["origin_ns"]):
        raise ValueError("Unsupported light timing equation or native origin")
    offset, allowance = light.get("offset_s"), light.get("local_review_allowance_s")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (offset, allowance)) or not 0 <= allowance <= 1:
        raise ValueError("Light timing needs a finite offset and explicit review allowance")
    timeline_path = _relative_file(path.parent, temporal.get("outputs", {}).get("canonical_channels"))
    timeline_raw = _read(timeline_path)
    entries = {}
    for index, row in enumerate(csv.DictReader(io.StringIO(timeline_raw.decode("utf-8")))):
        if index >= MAX_TRACKING_ROWS:
            raise ValueError("Light timeline exceeds the review row limit")
        if row.get("message_type") != "tracking":
            continue
        if row.get("recorded_static_frame_exists") != "True" or row.get("complete_saved_cohort") != "True":
            continue
        if row.get("mapping_status") != "exact_cohort_to_unique_differential_pts_pair":
            continue
        sequence = _integer(row.get("observer_sequence"), "light observer sequence")
        if sequence in entries:
            raise ValueError("Repeated tracking identity in light timing")
        pts, phone_pts = float(row["static_decoded_pts_s"]), float(row["estimated_phone_pts_s"])
        if not math.isfinite(pts) or not math.isfinite(phone_pts) or abs(pts - offset - phone_pts) > 1e-8:
            raise ValueError("Light timeline disagrees with its recorded timing equation")
        frame = _integer(row["ds9_frame_id"], "light frame")
        if int(row["derived_static_frame_index"]) != frame - mapping.get("static_decoded_frame_index_equals_ds9_frame_id_minus", -1):
            raise ValueError("Light timeline disagrees with its static frame mapping")
        entries[sequence] = row
    removed = 0
    for group in groups.values():
        supported = []
        for track in group["rows"]:
            row = entries.get(track["observer_sequence"])
            if row is None:
                removed += 1
                continue
            expected = {"source_id": track["source_id"],
                        "ds9_frame_id": track["frame_id"], "tracking_publication_sequence": track["publication_sequence"],
                        "exact_ds9_tracking_media_pts_ns": track["media_pts_ns"], "observed_at_us": track["time_ns"] // 1000}
            if any(_integer(row.get(k), f"light {k}") != v for k, v in expected.items()):
                raise ValueError("Light timing does not match the exact retained tracking cohort")
            if row.get("source_epoch") not in (None, "") and _integer(row["source_epoch"], "light source epoch") != track["source_epoch"]:
                raise ValueError("Light timing source epoch differs from its retained track")
            phone_pts = float(row["estimated_phone_pts_s"])
            if not 0 <= phone_pts <= native["video_end_s"]:
                removed += 1
                continue
            track["time_ns"] = native["origin_ns"] + round(phone_pts * 1e9)
            supported.append(track)
        group["rows"] = supported
    return [int(t) for t in native["times_ns"]], {
        "status": "light_cue_review", "method": "bound_light_events_and_exact_recorded_tracking_cohort",
        "offset_s": offset, "local_review_allowance_s": allowance, "unsupported_track_observations": removed,
        "synchronization_verified": False, "clock_rate_assumed": 1.0, "clock_drift_verified": False,
        "limitation": light.get("full_walk_application") or "Light-cue timing is a review estimate, not verified whole-walk synchronization.",
        "evidence": {"temporal_alignment": _evidence(path, raw), "timeline": _evidence(timeline_path, timeline_raw)}}


def compare_review_path(scan_dir: Path, source: dict, state: dict, target_dir: Path,
                        target_state: dict, target_binding: dict, output_dir: Path,
                        companion_dir: Path | None, *, refined_poses: np.ndarray | None = None,
                        source_alignment_dir: Path | None = None,
                        target_alignment_dir: Path | None = None,
                        temporal_alignment_path: Path | None = None,
                        refined_alignment_dir: Path | None = None,
                        refined_output_manifest: Path | None = None,
                        target_reference: dict | None = None) -> dict:
    """Write room-frame paths and per-lifecycle same-time separation diagnostics."""
    original_hash = source["evidence"]["provider_manifest"]["sha256"]
    registered = _alignment(scan_dir, state, original_hash, source["manifest"]["coordinate_frame"], source_alignment_dir)
    if target_reference is not None:
        if target_reference["source_scan_id"] != target_state["id"]:
            raise ValueError("The selected PCF belongs to a different reference scan")
        reference = _pcf_registration(target_reference, registered["target"])
    elif target_state["id"] == state["id"]:
        reference = registered
    else:
        reference = _alignment(target_dir, target_state, target_binding["manifest_sha256"],
                               target_binding["coordinate_frame"], target_alignment_dir)
    source_target = registered["target"]
    for key in ("world_frame", "world_frame_revision", "camera_frame_binding"):
        if reference["target"].get(key) != source_target.get(key):
            raise ValueError("Walk and reference reconstruction have different world bindings")
    original = registered["matrix"] @ source["poses"]
    refined_registration = registered
    refined_binding = None
    if refined_alignment_dir is not None:
        if refined_output_manifest is None or refined_poses is None:
            raise ValueError("Scaled refinement needs its own manifest, poses and passed static registration")
        candidate_manifest, candidate_raw = _json(refined_output_manifest)
        refined_binding = _evidence(refined_output_manifest, candidate_raw)
        refined_registration = _alignment(scan_dir, state, refined_binding["sha256"],
                                           candidate_manifest["coordinate_frame"], refined_alignment_dir)
        for key in ("world_frame", "world_frame_revision", "camera_frame_binding"):
            if refined_registration["target"].get(key) != source_target.get(key):
                raise ValueError("Scaled refinement alignment changed the target world binding")
    selected = original if refined_poses is None else refined_registration["matrix"] @ _validate_poses(refined_poses, len(original))
    path_rows = []
    previous_index, previous_time = None, None
    for i, prepared_index in enumerate(source["selected"]):
        frame = source["prepared"]["frames"][prepared_index]
        stamp = float(frame["timestamp_s"])
        path_rows.append({"provider_index": i, "prepared_frame_id": frame["frame_id"], "phone_time_s": stamp,
                          "capture_time_ns": str(frame["capture_time_ns"]) if type(frame.get("capture_time_ns")) is int else None,
                          "original_world_m": original[i, :3, 3].tolist(), "reference_world_m": selected[i, :3, 3].tolist(),
                          "camera_to_world": selected[i].tolist(),
                          "break_before": bool(previous_index is None or prepared_index != previous_index + 1 or stamp - previous_time > 2.0
                          or (i > 0 and source["windows"][i] != source["windows"][i - 1]))})
        previous_index, previous_time = prepared_index, stamp
    reference_path = {"schema": "noesis.phone_scan.room_path_reference.v1", "review_only": True,
                      "origin": "camera_optical_center", "units": "meters", "axes": "x_y_up_z",
                      "world_frame": source_target["world_frame"], "world_frame_revision": source_target["world_frame_revision"],
                      "world_transform_sha256": source_target["camera_frame_binding"]["transform_sha256"],
                      "reference_scan_id": target_state["id"], "source_manifest_sha256": original_hash,
                      "position_refined": refined_poses is not None, "body_ground_reference": False,
                      "poses": path_rows, "registration": refined_registration["evidence"],
                      "original_registration": registered["evidence"], "refined_manifest": refined_binding,
                      "reference_registration": reference["evidence"]}
    if target_reference is not None:
        reference_path["reference_reconstruction"] = target_binding
    (output_dir / "room_path_reference.json").write_text(json.dumps(reference_path, indent=2, allow_nan=False) + "\n")
    report = {"schema": SCHEMA, "status": "reference_ready", "review_only": True,
              "registration": {"status": "registered", "world_frame": source_target["world_frame"],
                               "world_frame_revision": source_target["world_frame_revision"],
                               "fitted_to_noesis_tracks": False, "reference_scan_id": target_state["id"]},
              "reference": reference_path, "timing": {"status": "not_available"}, "tracks": [],
              "person_identity_verified": False, "accuracy_qualified": False,
              "metric_semantics": "horizontal_camera_center_to_person_ground_separation_not_position_error",
              "limitations": ["The camera center is only a close-body positional proxy; its anatomical offset has not been measured.",
                              "No track is automatically declared to be the phone carrier.",
                              "Static-camera callback latency and alignment uncertainty affect separation; no 10 cm claim is made."],
              "artifacts": {"room_reference": "room_path_reference.json", "comparison": "path_comparison.json"}}
    try:
        if companion_dir is None:
            raise ValueError("No retained paired Noesis session is available")
        session_path = companion_dir / "session.json"
        session, session_raw = _json(session_path)
        # Retained session.json is the internal lifecycle record; the public
        # adapter calls historical "complete" records "stopped" and flattens
        # camera_id. Normalize those documented fields, not arbitrary aliases.
        session["camera_id"] = (session.get("camera") or {}).get("camera_id")
        recorded = state.get("companion_capture") or {}
        if (session.get("session_id") != recorded.get("session_id")
                or session.get("camera_id") != recorded.get("camera_id")
                or session.get("camera_id") != source_target.get("camera_id")
                or session.get("status") not in {"stopped", "complete"} or session.get("error")
                or any((session.get(k) or {}).get("partial") or (session.get(k) or {}).get("error") for k in ("recorder", "tracking"))
                or session.get("phone", {}).get("scan_id") != state["id"]
                or not session.get("phone", {}).get("archive_sha256")
                or session["phone"]["archive_sha256"] != recorded.get("phone", {}).get("archive_sha256")):
            raise ValueError("Paired session must bind the exact phone archive, scan and static camera")
        if source_target.get("reference_kind") == "paired_static" and source_target.get("companion_session_id") != session["session_id"]:
            raise ValueError("Path alignment belongs to a different paired session")
        native = _load_native(scan_dir, source)
        tracking_path = _relative_file(companion_dir, session.get("artifacts", {}).get("tracking"))
        groups, rejected, track_evidence = _tracks(tracking_path, session["camera_id"], source_target)
        times, timing = (_light_clock(temporal_alignment_path, session, state, native, groups) if temporal_alignment_path is not None
                         else _native_clock(session, native))
        report.update(timing=timing, excluded_observations=rejected,
                      evidence={"session": _evidence(session_path, session_raw), "tracking": track_evidence,
                                "native": native["evidence"]})
        csv_rows = []
        for key, group in groups.items():
            rows = group.pop("rows")
            track_times = np.asarray([r["time_ns"] for r in rows], dtype=np.int64)
            pairs = []
            for i, stamp in enumerate(times):
                if stamp is None or not rows:
                    continue
                insertion = int(np.searchsorted(track_times, stamp))
                candidates = [k for k in (insertion - 1, insertion) if 0 <= k < len(rows)]
                nearest = min(candidates, key=lambda k: abs(int(track_times[k]) - stamp))
                dt = (int(track_times[nearest]) - stamp) / 1e9
                if abs(dt) > MAX_MATCH_GAP_S:
                    continue
                track = rows[nearest]
                delta = selected[i, [0, 2], 3] - np.asarray(track["world"])[[0, 2]]
                baseline = original[i, [0, 2], 3] - np.asarray(track["world"])[[0, 2]]
                pair = {"provider_index": i, "phone_time_s": path_rows[i]["phone_time_s"],
                        "prepared_frame_id": path_rows[i]["prepared_frame_id"], "frame_id": track["frame_id"],
                        "time_delta_s": dt, "reference_world_m": selected[i, :3, 3].tolist(),
                        "noesis_world_m": track["world"], "horizontal_separation_m": float(np.linalg.norm(delta)),
                        "original_horizontal_separation_m": float(np.linalg.norm(baseline)),
                        "stable_id": track["stable_id"], "world_source": track["world_source"],
                        "trail_segment_id": track["trail_segment_id"],
                        "break_before": path_rows[i]["break_before"] or track["trail_break_required"]}
                if not pairs or pairs[-1]["provider_index"] != i - 1 or any(pairs[-1][k] != pair[k] for k in ("stable_id", "trail_segment_id")):
                    pair["break_before"] = True
                pairs.append(pair)
                csv_rows.append({"track_key": key, **{k: pair[k] for k in ("phone_time_s", "prepared_frame_id", "frame_id", "time_delta_s", "stable_id", "horizontal_separation_m", "original_horizontal_separation_m")}})
            separation = np.asarray([p["horizontal_separation_m"] for p in pairs])
            report["tracks"].append({**group, "current_world_observation_count": len(rows), "matched_count": len(pairs),
                                     "match_coverage": len(pairs) / len(path_rows), "pairs": pairs,
                                     "separation_m": {"median": float(np.median(separation)), "p95": float(np.percentile(separation, 95)), "max": float(np.max(separation))} if len(pairs) else None})
        report["status"] = "comparison_ready" if csv_rows else "reference_ready_no_matching_world"
        report["match_policy"] = {"method": "nearest_current_observation_per_phone_pose", "max_gap_s": MAX_MATCH_GAP_S,
                                  "interpolated": False, "motion_fitted_clock": False, "cross_lifecycle_join": False}
        if csv_rows:
            with (output_dir / "path_comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
            report["artifacts"]["comparison_csv"] = "path_comparison.csv"
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report["status"] = "reference_ready_comparison_needs_evidence"
        report["reason"] = str(exc)[:2000]
    (output_dir / "path_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
