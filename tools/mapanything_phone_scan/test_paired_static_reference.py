from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import paired_static_reference as paired
from tools.mapanything_phone_scan.inference import MapAnythingScanSettings


CAMERA_ID = "living-room"
SCAN_ID = "scan-fixture"
SESSION_ID = "companion-fixture"
PHONE_CAPTURE_ID = "phone-fixture"
PHONE_ARCHIVE_SHA256 = "a" * 64
DEWARP_NAME = "fixture_dewarper.txt"
VIDEO_FRAME_COUNT = 12
RAW_WIDTH = 128
RAW_HEIGHT = 96
MODEL_WIDTH = 256
MODEL_HEIGHT = 192


def test_installed_model_preprocessing_receives_float32_calibration():
    pytest.importorskip("mapanything.utils.image")
    import torch

    class CalibrationConsumer:
        def infer(self, views, **kwargs):
            view = views[0]
            assert view["intrinsics"].dtype == torch.float32
            height, width = view["img"].shape[-2:]
            depth = torch.ones((1, height, width, 1), dtype=torch.float32)
            return [{
                "depth_z": depth, "depth_along_ray": depth * 2,
                "conf": depth, "mask": depth.bool(),
                "intrinsics": view["intrinsics"] * torch.tensor([[[0.9, 1, 1], [1, 0.9, 1], [1, 1, 1]]]),
                "img_no_norm": torch.zeros((1, height, width, 3)),
            }]

    source_k = np.asarray([[625, 0, 914], [0, 624, 561], [0, 0, 1]], dtype=np.float64)
    source_k_before = source_k.copy()
    depth, _, mask, processed_k, rgb, diagnostics = paired._run_static_inference(
        CalibrationConsumer(), np.zeros((1080, 1920, 3), dtype=np.uint8),
        source_k, MapAnythingScanSettings(device="cpu"),
    )
    assert depth.shape == mask.shape == rgb.shape[:2]
    assert processed_k.shape == (3, 3)
    ys, xs = np.indices(depth.shape)
    camera_points = np.stack(((xs - processed_k[0, 2]) * depth / processed_k[0, 0],
                              (ys - processed_k[1, 2]) * depth / processed_k[1, 1], depth), axis=-1)
    np.testing.assert_allclose(np.linalg.norm(camera_points, axis=-1), 2.0, atol=1e-6)
    assert not np.allclose(diagnostics["model_intrinsics"], processed_k)
    np.testing.assert_array_equal(source_k, source_k_before)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _dewarper_text() -> str:
    # The fixture uses equal source and rectified intrinsics so the dewarper
    # remains a real, deterministic OpenCV transform while keeping the test
    # independent of a repository camera config.
    return """[property]
output-width=128
output-height=96
num-batch-buffers=1
dewarp-dump-frames=0

[surface0]
projection-type=4
surface-index=0
width=128
height=96
focal-length=4;4
distortion=0;0;0;0
src-x0=64
src-y0=48
dst-focal-length=4;4
dst-principal-point=64;48
"""


def _make_video(root: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the encoded static fixture")
    frames = root / "video_frames"
    frames.mkdir()
    for index in range(VIDEO_FRAME_COUNT):
        image = np.zeros((RAW_HEIGHT, RAW_WIDTH, 3), dtype=np.uint8)
        image[:, :, 0] = 17 + index
        image[:, :, 1] = 39
        image[:, :, 2] = 91
        cv2.putText(
            image,
            str(index),
            (3, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        assert cv2.imwrite(str(frames / f"frame-{index:02d}.png"), image)
    output = root / "static_camera.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "5",
            "-i",
            str(frames / "frame-%02d.png"),
            "-c:v",
            "libx264",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            str(output),
        ],
        check=True,
    )
    return output


def _fixture(tmp_path: Path) -> dict[str, Any]:
    scan_dir = tmp_path / "scan"
    session_dir = tmp_path / SESSION_ID
    scan_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    _write_json(scan_dir / "scan_state.json", {"id": SCAN_ID})
    video = _make_video(session_dir)

    dewarper_text = _dewarper_text()
    dewarper_hash = hashlib.sha256(dewarper_text.encode("utf-8")).hexdigest()
    e = np.eye(4, dtype=np.float64)
    e_col_major = e.reshape(-1, order="F").tolist()
    target = np.eye(4, dtype=np.float64)
    target[0, 3] = 0.5
    target_col_major = target.reshape(-1, order="F").tolist()
    k = [4.0, 4.0, 64.0, 48.0]

    calibration_path = tmp_path / "camera_calibration.json"
    calibration = {
        "cameras": {
            CAMERA_ID: {
                "E": e_col_major,
                "K": k,
            }
        }
    }
    _write_json(calibration_path, calibration)
    calibration_hash = _sha256(calibration_path)

    binding = {
        "contract": "noesis.camera.frame_binding.v1",
        "contract_version": 1,
        "camera_calibration_sha256": calibration_hash,
        "target_from_calibration_col_major": target_col_major,
        "calibration_frame": "living_room_calibration",
        "calibration_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
        "world_floor_plane": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
        "world_frame": {"frame_id": "backend_world_m", "revision": "world-r1"},
    }
    bundle = {
        "schema": "noesis.calibration.bundle.v1",
        "meta": {"cameras_E_semantics": "camera_from_calibration_frame_raw"},
        "cameras": {
            "E": {CAMERA_ID: e_col_major},
            "K": {CAMERA_ID: k},
            "frame_bindings": {CAMERA_ID: binding},
        },
    }
    tracking_path = session_dir / "tracking.ndjson"
    tracking_path.write_text(
        json.dumps(
            {
                "message_type": "calibration-bundle",
                "message": {"data": bundle},
            }
        )
        + "\n"
        + json.dumps({"message_type": "tracking", "message": {"data": {}}})
        + "\n",
        encoding="utf-8",
    )

    # Deliberately use nonuniform recorder PTS and unrelated claimed frame
    # numbers.  The builder must sample the decoded media PTS/frame sequence,
    # while retaining these packet rows as timing provenance only.
    packet_path = session_dir / "packet_timing.jsonl"
    packet_rows = []
    pts = 1_000_000_000
    packet_row_count = 8
    for index in range(packet_row_count):
        pts += 9_000_000 + index * 137_000
        packet_rows.append(
            {
                "pts_ns": pts,
                "duration_ns": 200_000_000,
                "keyframe": True,
                "frame_index": 1000 + index * 17,
            }
        )
    packet_path.write_text(
        "".join(json.dumps(row) + "\n" for row in packet_rows),
        encoding="utf-8",
    )

    camera_authority = {
        "camera_id": CAMERA_ID,
        "source_id": 0,
        "dewarper": {
            "canonical_tracker_pixels": {
                "authority": "captured fixture dewarper",
                "camera_intrinsics_role": "rectified_output",
                "coordinate_space": "post_dewarper_streammux_pixels",
                "dewarper_enabled": True,
                "dewarper_config_name": DEWARP_NAME,
                "dewarper_config_sha256": dewarper_hash,
                "dewarper_config_text": dewarper_text,
                "output_resolution_px": [RAW_WIDTH, RAW_HEIGHT],
            }
        },
    }
    session = {
        "schema": "noesis.companion_capture.session.v1",
        "status": "complete",
        "finalized_at": "2026-09-06T00:00:00+00:00",
        "session_id": SESSION_ID,
        "camera_id": CAMERA_ID,
        "camera": {"camera_id": CAMERA_ID, "authority": camera_authority},
        "phone_capture_id": PHONE_CAPTURE_ID,
        "phone": {
            "capture_id": PHONE_CAPTURE_ID,
            "scan_id": SCAN_ID,
            "archive_sha256": PHONE_ARCHIVE_SHA256,
        },
        "artifacts": {
            "static_video": video.name,
            "packet_timing": packet_path.name,
            "tracking": tracking_path.name,
        },
        "recorder": {
            "status": "stopped",
            "encoded_ready": True,
            "partial": False,
            "packet_count": packet_row_count,
        },
    }
    _write_json(session_dir / "session.json", session)

    settings = MapAnythingScanSettings(
        model_id="fixture/model",
        device="cpu",
        amp_dtype="fp32",
        point_budget=20_000,
        local_files_only=True,
    )
    return {
        "scan_dir": scan_dir,
        "session_dir": session_dir,
        "session": session,
        "video": video,
        "packet_rows": packet_rows,
        "calibration_path": calibration_path,
        "calibration": calibration,
        "binding": binding,
        "dewarper_text": dewarper_text,
        "dewarper_hash": dewarper_hash,
        "settings": settings,
        "companion": {
            "session_id": SESSION_ID,
            "camera_id": CAMERA_ID,
            "phone_capture_id": PHONE_CAPTURE_ID,
            "phone": {"archive_sha256": PHONE_ARCHIVE_SHA256},
        },
    }


class _FixtureModel:
    def to(self, *_args: Any, **_kwargs: Any) -> "_FixtureModel":
        return self


def _patch_static_model(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def load_model(_settings: Any) -> _FixtureModel:
        calls.append({"kind": "load"})
        return _FixtureModel()

    def run_inference(
        _model: Any,
        image_rgb: np.ndarray,
        intrinsics: np.ndarray,
        _settings: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        calls.append(
            {
                "kind": "infer",
                "image_shape": tuple(image_rgb.shape),
                "intrinsics": np.asarray(intrinsics, dtype=np.float64).copy(),
            }
        )
        height, width = MODEL_HEIGHT, MODEL_WIDTH
        depth = np.full((height, width), 2.0, dtype=np.float32)
        confidence = np.ones((height, width), dtype=np.float32)
        mask = np.ones((height, width), dtype=bool)
        processed_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        processed_rgb[:, :, :] = [17, 39, 91]
        processed_k = np.asarray(
            [[8.0, 0.0, 128.0], [0.0, 8.0, 96.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return depth, confidence, mask, processed_k, processed_rgb, {"model_intrinsics": processed_k}

    monkeypatch.setattr(paired, "_load_model", load_model)
    monkeypatch.setattr(paired, "_run_static_inference", run_inference)
    return calls


def _build(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calls = _patch_static_model(monkeypatch)
    result = paired.prepare_paired_static_reference(
        fixture["scan_dir"],
        fixture["session_dir"],
        fixture["companion"],
        fixture["settings"],
        fixture["calibration_path"],
        lambda *_args: None,
    )
    return result, calls


def test_static_reference_uses_decoded_pts_known_k_and_captured_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    decoded_pts = paired._probe_video_pts(fixture["video"])
    assert len(decoded_pts) == VIDEO_FRAME_COUNT
    assert all(right > left for left, right in zip(decoded_pts, decoded_pts[1:]))

    result, calls = _build(fixture, monkeypatch)
    revision = fixture["scan_dir"] / result["target_revision"]
    assert result["status"] == "complete"
    assert len([row for row in calls if row["kind"] == "infer"]) == 6
    for call in calls:
        if call["kind"] == "infer":
            np.testing.assert_allclose(
                call["intrinsics"],
                [[4.0, 0.0, 64.0], [0.0, 4.0, 48.0], [0.0, 0.0, 1.0]],
            )

    manifest = json.loads((revision / "manifest.json").read_text(encoding="utf-8"))
    source_rows = manifest["source"]["frames"]
    assert [row["source_frame_index"] for row in source_rows] == [1, 3, 5, 6, 8, 10]
    assert all(row["source_frame_index"] != fixture["packet_rows"][i]["frame_index"] for i, row in enumerate(source_rows))
    assert all(
        int(row["encoded_pts_ns"]) == pytest.approx(
            decoded_pts[row["source_frame_index"]] * 1e9, abs=2
        )
        for row in source_rows
    )

    with np.load(revision / "room_points.npz") as payload:
        points = np.asarray(payload["points"], dtype=np.float64)
        colors = np.asarray(payload["colors"], dtype=np.uint8)
    assert points.ndim == 2 and points.shape[1] == 3
    expected_center = np.asarray([0.5, 0.0, 2.0])
    assert np.min(np.linalg.norm(points - expected_center, axis=1)) < 1e-5
    center = int(np.argmin(np.linalg.norm(points - expected_center, axis=1)))
    np.testing.assert_array_equal(colors[center], [17, 39, 91])

    meta = json.loads((revision / "room_points_meta.json").read_text(encoding="utf-8"))
    assert meta["camera"] == CAMERA_ID
    assert meta["coordinate_frame"] == "backend_world_m_stream_points"
    np.testing.assert_allclose(meta["intrinsics"], [[4.0, 0.0, 64.0], [0.0, 4.0, 48.0], [0.0, 0.0, 1.0]])
    assert meta["floor_alignment"]["world_correction_col_major"] == fixture["binding"]["target_from_calibration_col_major"]
    keyframes = meta["rgb_keyframes"]
    assert keyframes and all((revision / path).is_file() for path in keyframes.values())

    calibration_payload = json.loads((revision / "calibration.json").read_text(encoding="utf-8"))
    assert calibration_payload["cameras"][CAMERA_ID]["E"] == fixture["calibration"]["cameras"][CAMERA_ID]["E"]


def _selected_fixture(fixture: dict[str, Any]) -> Path:
    root = fixture["scan_dir"]
    mask_path = root / "person_exclusion.png"
    mask = np.zeros((RAW_HEIGHT, RAW_WIDTH), dtype=np.uint8)
    mask[38:58, 54:74] = 255
    assert cv2.imwrite(str(mask_path), mask)
    pts = paired._probe_video_pts(fixture["video"])
    path = root / "static_selection.json"
    _write_json(path, {
        "schema": "noesis.paired_static_reference.selection.v1",
        "camera_id": CAMERA_ID, "source_video_sha256": _sha256(fixture["video"]),
        "mask_coordinate_space": "post_dewarper_streammux_pixels", "keyframe_sample_index": 5,
        "frames": [{"source_frame_index": i, "encoded_pts_ns": round(pts[i] * 1e9),
                    "exclusion_mask": mask_path.name, "exclusion_mask_sha256": _sha256(mask_path)}
                   for i in [0, 2, 4, 7, 9, 11]],
    })
    return path


def test_explicit_recorded_selection_excludes_foreground_before_depth_fusion(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    path = _selected_fixture(fixture)
    _patch_static_model(monkeypatch)
    result = paired.prepare_paired_static_reference(
        fixture["scan_dir"], fixture["session_dir"], fixture["companion"],
        fixture["settings"], fixture["calibration_path"], lambda *_: None,
        selection_manifest=path,
    )
    revision = fixture["scan_dir"] / result["target_revision"]
    manifest = json.loads((revision / "manifest.json").read_text())
    assert [r["source_frame_index"] for r in manifest["source"]["frames"]] == [0, 2, 4, 7, 9, 11]
    assert manifest["identity"]["sampling"]["manifest_sha256"] == _sha256(path)
    with np.load(revision / "fused_depth.npz") as raw:
        assert not raw["mask"][96, 128]
        assert np.count_nonzero(raw["mask"]) > 5000
    valid = cv2.imread(str(revision / "keyframe_valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert valid[48, 64] == 0
    meta = json.loads((revision / "room_points_meta.json").read_text())
    assert meta["keyframe_source"] == "selected_recorded_observation_5"
    assert meta["rgb_keyframe_valid_masks"] == {CAMERA_ID: "keyframe_valid_mask.png"}
    paired.validate_paired_static_reference(fixture["scan_dir"], result)


@pytest.mark.parametrize("change", ["video", "pts", "mask_hash", "order", "unreviewed"])
def test_explicit_selection_rejects_unbound_or_inconsistent_evidence(tmp_path, change):
    fixture = _fixture(tmp_path)
    path = _selected_fixture(fixture)
    payload = json.loads(path.read_text())
    if change == "video":
        payload["source_video_sha256"] = "0" * 64
    elif change == "pts":
        payload["frames"][0]["encoded_pts_ns"] += 10_000_000
    elif change == "mask_hash":
        payload["frames"][0]["exclusion_mask_sha256"] = "0" * 64
    elif change == "order":
        payload["frames"].reverse()
    else:
        payload["frames"][0].pop("exclusion_mask")
    _write_json(path, payload)
    with pytest.raises(paired.PairedStaticReferenceError):
        paired._explicit_selection(path, {"camera_id": CAMERA_ID, "video_sha256": _sha256(fixture["video"])},
                                   np.asarray(paired._probe_video_pts(fixture["video"])), (RAW_WIDTH, RAW_HEIGHT))


def test_static_reference_rejects_pairing_and_captured_input_hash_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    bad_companion = dict(fixture["companion"])
    bad_companion["session_id"] = "other-session"
    with pytest.raises(paired.PairedStaticReferenceError, match="session ID"):
        paired.prepare_paired_static_reference(
            fixture["scan_dir"], fixture["session_dir"], bad_companion,
            fixture["settings"], fixture["calibration_path"], lambda *_args: None,
        )

    session_path = fixture["session_dir"] / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    pixels = session["camera"]["authority"]["dewarper"]["canonical_tracker_pixels"]
    pixels["dewarper_config_text"] += "\n# tampered\n"
    _write_json(session_path, session)
    with pytest.raises(paired.PairedStaticReferenceError, match="dewarper.*hash|hash.*dewarper"):
        paired.prepare_paired_static_reference(
            fixture["scan_dir"], fixture["session_dir"], fixture["companion"],
            fixture["settings"], fixture["calibration_path"], lambda *_args: None,
        )

    # Recreate the clean session, then change the active E while retaining the
    # captured E/binding.  The static reference must fail before model load.
    fixture = _fixture(tmp_path / "calibration-mismatch")
    calibration = fixture["calibration"]
    calibration["cameras"][CAMERA_ID]["E"][12] = 0.25
    _write_json(fixture["calibration_path"], calibration)
    with pytest.raises(paired.PairedStaticReferenceError, match="E|calibration"):
        paired.prepare_paired_static_reference(
            fixture["scan_dir"], fixture["session_dir"], fixture["companion"],
            fixture["settings"], fixture["calibration_path"], lambda *_args: None,
        )


def test_static_reference_validation_requires_alignment_consumer_artifacts_and_cameras_e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    result, _calls = _build(fixture, monkeypatch)
    revision = fixture["scan_dir"] / result["target_revision"]
    assert (revision / "room_points.npz").is_file()
    assert (revision / "room_points_meta.json").is_file()

    # Exercise the existing static-reference consumer against the exact file
    # names and calibration shape used by alignment/MapAnything variants.
    from tools.mapanything_phone_scan.run_mapanything_prior_variants import _load_static_reference

    loaded = _load_static_reference(revision, fixture["calibration_path"], CAMERA_ID)
    assert loaded.point_count >= 5_000
    assert loaded.projected_point_count >= 5_000

    broken_calibration = tmp_path / "broken_calibration.json"
    _write_json(broken_calibration, {"cameras": {CAMERA_ID: {}}})
    with pytest.raises(Exception, match="calibrated E matrix"):
        _load_static_reference(revision, broken_calibration, CAMERA_ID)

    points_path = revision / "room_points.npz"
    points_path.write_bytes(points_path.read_bytes() + b"tampered")
    with pytest.raises(paired.PairedStaticReferenceError, match="hash mismatch"):
        paired.validate_paired_static_reference(fixture["scan_dir"], result)


def test_static_reference_cache_reuse_skips_model_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    first, calls = _build(fixture, monkeypatch)
    assert len([row for row in calls if row["kind"] == "load"]) == 1

    def fail_load(_settings: Any) -> Any:
        raise AssertionError("cache reuse initialized the static model")

    monkeypatch.setattr(paired, "_load_model", fail_load)
    second = paired.prepare_paired_static_reference(
        fixture["scan_dir"], fixture["session_dir"], fixture["companion"],
        fixture["settings"], fixture["calibration_path"], lambda *_args: None,
    )
    assert second["revision_id"] == first["revision_id"]
    assert second["manifest_sha256"] == first["manifest_sha256"]
