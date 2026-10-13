from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import extract_bound_texture_photos as extractor
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _scene(seed: int, *, width: int = 640, height: int = 360) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = rng.integers(0, 256, (45, 80, 3), dtype=np.uint8)
    image = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    for index in range(36):
        color = tuple(int(value) for value in rng.integers(15, 241, 3))
        center = tuple(int(value) for value in rng.integers([0, 0], [width, height]))
        radius = int(rng.integers(3, 22))
        cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        start = tuple(int(value) for value in rng.integers([0, 0], [width, height]))
        end = tuple(int(value) for value in rng.integers([0, 0], [width, height]))
        cv2.line(image, start, end, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(
        image,
        f"BOUND-{seed}",
        (25, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return image


def _fixture(tmp_path: Path, *, duplicate_views: bool = False) -> dict[str, object]:
    scan = tmp_path / "scan"
    frames_dir = scan / "frames"
    capture = scan / "capture"
    frames_dir.mkdir(parents=True)
    capture.mkdir()
    source = capture / "camera.mp4"
    source.write_bytes(b"synthetic exact video identity\x00" * 31)

    native_images = [_scene(11), _scene(22), _scene(33)]
    if duplicate_views:
        native_images[1] = native_images[0].copy()
    source_indices = [0, 2, 4]
    timestamps = [1_000_000_000 + index * 33_333_333 for index in range(5)]
    rows = []
    for index, (source_index, native) in enumerate(
        zip(source_indices, native_images, strict=True)
    ):
        prepared = cv2.resize(native, (320, 180), interpolation=cv2.INTER_AREA)
        path = frames_dir / f"frame_{index:04d}.jpg"
        assert cv2.imwrite(str(path), prepared, [cv2.IMWRITE_JPEG_QUALITY, 94])
        sha = _sha(path)
        rows.append(
            {
                "index": index,
                "frame_id": prepared_frame_identity(index, sha),
                "frame": f"frames/{path.name}",
                "sha256": sha,
                "width": 320,
                "height": 180,
                "source_frame_index": source_index,
                "capture_time_ns": timestamps[source_index],
                "timestamp_s": (timestamps[source_index] - timestamps[0]) / 1e9,
                "timestamp_source": extractor.TRUSTED_TIMESTAMP_SOURCE,
            }
        )
    prepared_manifest = {
        "schema": "noesis.mapanything.phone_scan.prepared_frames.v2",
        "video": {
            "codec": "fixture",
            "container": "fixture",
            "duration_s": 0.2,
            "height": 360,
            "path": "capture/camera.mp4",
            "reported_frame_count": len(timestamps),
            "rotation_degrees": 0.0,
            "sha256": _sha(source),
            "size_bytes": source.stat().st_size,
            "source_fps": 30.0,
            "width": 640,
        },
        "preparation": {"frame_count": len(rows)},
        "frames": rows,
        "camera_calibration": {"status": "not_applied"},
    }
    manifest_path = scan / "prepared_frames_manifest.json"
    _write_json(manifest_path, prepared_manifest)
    _write_json(capture / "video_timestamps_ns.json", timestamps)
    _write_json(
        capture / "capture_import.json",
        {
            "schema": "noesis.phone_capture.v1",
            "android_capture": {
                "camera_acquisition_timestamp_verified": True,
                "association_method": extractor.TRUSTED_ASSOCIATION_METHOD,
                "association_row_count": len(timestamps),
                "decoded_frame_count": len(timestamps),
                "matched_frame_count": len(timestamps),
            },
            "video": {
                "camera_acquisition_timestamp_verified": True,
                "timestamp_source": extractor.TRUSTED_TIMESTAMP_SOURCE,
                "frame_count": len(timestamps),
                "start_time_ns": timestamps[0],
                "end_time_ns": timestamps[-1],
            },
        },
    )
    return {
        "scan": scan,
        "source": source,
        "manifest_path": manifest_path,
        "native_images": native_images,
        "output": tmp_path / "output",
    }


def _fake_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    images: list[np.ndarray],
    *,
    order: list[int] | None = None,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str]) -> None:
        calls.append(list(command))
        pattern = Path(command[-1])
        selected = order if order is not None else list(range(len(images)))
        for output_index, image_index in enumerate(selected):
            output = pattern.parent / f"frame_{output_index:04d}.png"
            assert cv2.imwrite(str(output), images[image_index])

    monkeypatch.setattr(extractor, "_run_ffmpeg", run)
    return calls


def _edit_manifest(fixture: dict[str, object], edit) -> None:
    path = fixture["manifest_path"]
    assert isinstance(path, Path)
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    _write_json(path, value)


def test_extracts_one_lossless_native_photo_per_strict_bound_row(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    calls = _fake_ffmpeg(monkeypatch, fixture["native_images"])
    manifest_path = extractor.extract_bound_texture_photos(
        fixture["scan"], fixture["source"], fixture["output"], long_edge_px=640
    )

    assert manifest_path == fixture["output"] / extractor.OUTPUT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == extractor.OUTPUT_SCHEMA
    assert manifest["status"] == "verified_all_bound_views"
    assert manifest["prepared_manifest"]["sha256"] == _sha(fixture["manifest_path"])
    assert manifest["source_video"]["sha256"] == _sha(fixture["source"])
    assert manifest["extraction"]["prepared_pixels_used_as_output"] is False
    assert manifest["geometry"] == {
        "source_pixel_geometry": "native_noautorotate_uniform_scale_only",
        "intrinsics_modified": False,
        "poses_modified": False,
        "prepared_manifest_modified": False,
    }
    assert [row["source_frame_index"] for row in manifest["views"]] == [0, 2, 4]
    assert all(
        row["image_mapping_verification"]["status"]
        == "verified_unique_same_decoded_frame_geometry"
        for row in manifest["views"]
    )
    assert all(row["output_photo"]["width"] == 640 for row in manifest["views"])
    assert all(row["output_photo"]["height"] == 360 for row in manifest["views"])
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("-hwaccel") + 1] == "none"
    assert "-noautorotate" in command
    assert "-n" in command
    assert command.count("-threads") == 2
    assert all(
        command[index + 1] == "4"
        for index, item in enumerate(command)
        if item == "-threads"
    )
    video_filter = command[command.index("-vf") + 1]
    assert "eq(n\\,0)+eq(n\\,2)+eq(n\\,4)" in video_filter
    assert "fps=" not in video_filter
    assert "select='" in video_filter


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_real_ffmpeg_command_selects_exact_ordinals_in_one_decode_pass(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    originals = []
    for index in range(5):
        image = np.full(
            (180, 320, 3), (20 + 30 * index, 90, 210 - 20 * index), np.uint8
        )
        cv2.putText(
            image,
            f"ordinal-{index}",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        originals.append(image)
        assert cv2.imwrite(str(inputs / f"input_{index:04d}.png"), image)
    source = tmp_path / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-framerate",
            "5",
            "-start_number",
            "0",
            "-i",
            str(inputs / "input_%04d.png"),
            "-frames:v",
            "5",
            "-c:v",
            "png",
            "-pix_fmt",
            "rgb24",
            str(source),
        ],
        check=True,
        timeout=30,
    )
    command = extractor._ffmpeg_command(source, outputs / "frame_%04d.png", [1, 3], 320)
    assert "-ss" not in command and "-sseof" not in command
    extractor._run_ffmpeg(command)
    paths = sorted(outputs.glob("frame_*.png"))
    assert [path.name for path in paths] == ["frame_0000.png", "frame_0001.png"]
    np.testing.assert_array_equal(cv2.imread(str(paths[0])), originals[1])
    np.testing.assert_array_equal(cv2.imread(str(paths[1])), originals[3])


def test_byte_identical_explicit_video_copy_is_accepted_by_identity(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    source_copy = tmp_path / "explicit-camera-copy.mp4"
    source_copy.write_bytes(fixture["source"].read_bytes())
    _fake_ffmpeg(monkeypatch, fixture["native_images"])
    manifest_path = extractor.extract_bound_texture_photos(
        fixture["scan"], source_copy, fixture["output"], long_edge_px=640
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_video"]["path"] == str(source_copy.resolve())
    assert manifest["source_video"]["prepared_resolved_path"] == str(
        fixture["source"].resolve()
    )


@pytest.mark.parametrize("case", ["swap", "unrelated", "ambiguous"])
def test_visual_mapping_fails_closed_for_wrong_or_ambiguous_rows(
    tmp_path, monkeypatch, case
):
    fixture = _fixture(tmp_path, duplicate_views=case == "ambiguous")
    images = fixture["native_images"]
    if case == "swap":
        order = [1, 0, 2]
    elif case == "unrelated":
        images = list(images)
        images[1] = _scene(999)
        order = None
    else:
        order = None
    _fake_ffmpeg(monkeypatch, images, order=order)
    with pytest.raises(
        extractor.BoundTexturePhotoError, match="mismatched or ambiguous"
    ):
        extractor.extract_bound_texture_photos(
            fixture["scan"], fixture["source"], fixture["output"], long_edge_px=640
        )
    assert not fixture["output"].exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("bool_source_index", "must be an integer"),
        ("duplicate_source_index", "must be distinct"),
        ("nonmonotonic_source_index", "strictly increasing"),
        ("out_of_range_source_index", "exceeds exact decoded count"),
        ("timestamp_source", "trusted exact native"),
        ("timestamp_value", "disagrees with native acquisition time"),
        ("timestamp_string", "finite number"),
        ("calibration_status", "explicitly not-applied"),
        ("calibration_processing", "rectified pixels"),
        ("frame_dimensions", "dimensions mismatch"),
        ("frame_hash", "file hash mismatch"),
        ("source_sha", "SHA-256 differs"),
    ],
)
def test_preflight_rejects_broken_source_or_prepared_identity(
    tmp_path, monkeypatch, case, message
):
    fixture = _fixture(tmp_path)
    if case == "bool_source_index":
        _edit_manifest(
            fixture,
            lambda value: value["frames"][1].__setitem__("source_frame_index", True),
        )
    elif case == "duplicate_source_index":
        _edit_manifest(
            fixture,
            lambda value: value["frames"][1].__setitem__("source_frame_index", 0),
        )
    elif case == "nonmonotonic_source_index":

        def edit(value):
            value["frames"][1]["source_frame_index"] = 4
            value["frames"][1]["capture_time_ns"] = 1_000_000_000 + 4 * 33_333_333
            value["frames"][1]["timestamp_s"] = 4 * 33_333_333 / 1e9
            value["frames"][2]["source_frame_index"] = 2
            value["frames"][2]["capture_time_ns"] = 1_000_000_000 + 2 * 33_333_333
            value["frames"][2]["timestamp_s"] = 2 * 33_333_333 / 1e9

        _edit_manifest(fixture, edit)
    elif case == "out_of_range_source_index":
        _edit_manifest(
            fixture,
            lambda value: value["frames"][2].__setitem__("source_frame_index", 5),
        )
    elif case == "timestamp_source":
        _edit_manifest(
            fixture,
            lambda value: value["frames"][1].__setitem__(
                "timestamp_source", "encoded_pts"
            ),
        )
    elif case == "timestamp_value":
        _edit_manifest(
            fixture, lambda value: value["frames"][1].__setitem__("timestamp_s", 0.5)
        )
    elif case == "timestamp_string":
        _edit_manifest(
            fixture,
            lambda value: value["frames"][1].__setitem__(
                "timestamp_s", str(value["frames"][1]["timestamp_s"])
            ),
        )
    elif case == "calibration_status":
        _edit_manifest(
            fixture,
            lambda value: value["camera_calibration"].__setitem__("status", "unknown"),
        )
    elif case == "calibration_processing":
        _edit_manifest(
            fixture,
            lambda value: value["frames"][1].__setitem__("calibration_processing", {}),
        )
    elif case == "frame_dimensions":
        _edit_manifest(
            fixture, lambda value: value["frames"][1].__setitem__("width", 318)
        )
    elif case == "frame_hash":
        path = fixture["scan"] / "frames/frame_0001.jpg"
        path.write_bytes(path.read_bytes() + b"changed")
    elif case == "source_sha":
        _edit_manifest(
            fixture, lambda value: value["video"].__setitem__("sha256", "0" * 64)
        )
    monkeypatch.setattr(
        extractor,
        "_run_ffmpeg",
        lambda _command: pytest.fail("ffmpeg must not run after a preflight failure"),
    )
    with pytest.raises(extractor.BoundTexturePhotoError, match=message):
        extractor.extract_bound_texture_photos(
            fixture["scan"], fixture["source"], fixture["output"], long_edge_px=640
        )
    assert not fixture["output"].exists()


@pytest.mark.parametrize(
    ("long_edge", "message"),
    [
        (255, "from 256"),
        (641, "must be even"),
        (4098, "from 256"),
        (320, "larger than the prepared JPEG edge"),
        (642, "exceeds native source resolution"),
    ],
)
def test_long_edge_is_bounded_and_never_upsamples_prepared_or_native(
    tmp_path, monkeypatch, long_edge, message
):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        extractor,
        "_run_ffmpeg",
        lambda _command: pytest.fail("ffmpeg must not run after a preflight failure"),
    )
    with pytest.raises(extractor.BoundTexturePhotoError, match=message):
        extractor.extract_bound_texture_photos(
            fixture["scan"],
            fixture["source"],
            fixture["output"],
            long_edge_px=long_edge,
        )


def test_existing_output_is_never_touched(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    output = fixture["output"]
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(
        extractor,
        "_run_ffmpeg",
        lambda _command: pytest.fail("ffmpeg must not run for an existing output"),
    )
    with pytest.raises(extractor.BoundTexturePhotoError, match="already exists"):
        extractor.extract_bound_texture_photos(
            fixture["scan"], fixture["source"], output, long_edge_px=640
        )
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_failed_ffmpeg_leaves_no_partial_output(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)

    def fail(_command):
        raise extractor.BoundTexturePhotoError("fixture decoder failure")

    monkeypatch.setattr(extractor, "_run_ffmpeg", fail)
    with pytest.raises(
        extractor.BoundTexturePhotoError, match="fixture decoder failure"
    ):
        extractor.extract_bound_texture_photos(
            fixture["scan"], fixture["source"], fixture["output"], long_edge_px=640
        )
    assert not fixture["output"].exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_prepared_manifest_source_is_rechecked_after_decode(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    source_copy = tmp_path / "explicit-camera-copy.mp4"
    source_copy.write_bytes(fixture["source"].read_bytes())

    def run(command):
        pattern = Path(command[-1])
        for index, image in enumerate(fixture["native_images"]):
            assert cv2.imwrite(str(pattern.parent / f"frame_{index:04d}.png"), image)
        original = fixture["source"].read_bytes()
        fixture["source"].write_bytes(bytes([original[0] ^ 1]) + original[1:])

    monkeypatch.setattr(extractor, "_run_ffmpeg", run)
    with pytest.raises(
        extractor.BoundTexturePhotoError, match="prepared-manifest source bytes changed"
    ):
        extractor.extract_bound_texture_photos(
            fixture["scan"], source_copy, fixture["output"], long_edge_px=640
        )
    assert not fixture["output"].exists()


def test_one_view_cannot_claim_all_pairs_ambiguity_proof(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)

    def edit(value):
        value["frames"] = value["frames"][:1]
        value["preparation"]["frame_count"] = 1

    _edit_manifest(fixture, edit)
    monkeypatch.setattr(
        extractor,
        "_run_ffmpeg",
        lambda _command: pytest.fail("ffmpeg must not run after a preflight failure"),
    )
    with pytest.raises(extractor.BoundTexturePhotoError, match="from 2 through"):
        extractor.extract_bound_texture_photos(
            fixture["scan"], fixture["source"], fixture["output"], long_edge_px=640
        )


def test_ffmpeg_timeout_is_finite_and_reported(monkeypatch):
    def expire(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", extractor.FFMPEG_TIMEOUT_S)

    monkeypatch.setattr(extractor.subprocess, "run", expire)
    with pytest.raises(extractor.BoundTexturePhotoError, match="exceeded 1800 seconds"):
        extractor._run_ffmpeg(["ffmpeg"])
