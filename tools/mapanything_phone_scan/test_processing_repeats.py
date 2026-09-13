from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import processing
from tools.mapanything_phone_scan.processing import (
    FramePreparationSettings,
    _Candidate,
    _analyze_candidates,
    _deduplicate_candidates,
    _extract_candidates,
    _select_keyframes,
)


def _candidate(
    index: int,
    timestamp_s: float,
    value: int,
    *,
    digest: bytes | None = None,
    quality_score: float = 0.5,
    capture_time_ns: int | None = None,
    source_frame_index: int | None = None,
) -> _Candidate:
    gray = np.full((90, 160), value, dtype=np.uint8)
    return _Candidate(
        index=index,
        timestamp_s=timestamp_s,
        path=Path(f"candidate_{index:05d}.jpg"),
        width=160,
        height=90,
        luminance_p10=float(value),
        luminance_p50=float(value),
        luminance_p90=float(value),
        blur_score=1.0,
        feature_count=0,
        feature_coverage=0.0,
        small_gray=gray,
        keypoints=[],
        descriptors=None,
        content_digest=(digest if digest is not None else bytes([index + 1])),
        capture_time_ns=capture_time_ns,
        source_frame_index=source_frame_index,
        quality_score=quality_score,
    )


def _strong_repeat_edge(*, appearance_delta: float = 0.5) -> dict[str, object]:
    return {
        "match_count": 100,
        "inlier_count": 90,
        "inlier_ratio": 0.9,
        "source_coverage": 0.75,
        "target_coverage": 0.75,
        "median_displacement_norm": 0.001,
        "appearance_delta": appearance_delta,
        "passes_connectivity": True,
    }


def test_multisecond_exact_freeze_keeps_best_representative_and_reports_span() -> None:
    candidates = [
        _candidate(0, 0.0, 80, digest=b"freeze", quality_score=0.20, capture_time_ns=1000, source_frame_index=4),
        _candidate(1, 1.0, 80, digest=b"freeze", quality_score=0.90, capture_time_ns=2000, source_frame_index=8),
        _candidate(2, 2.0, 80, digest=b"freeze", quality_score=0.30, capture_time_ns=3000, source_frame_index=12),
        _candidate(3, 5.0, 120, digest=b"change", quality_score=0.60, capture_time_ns=6000, source_frame_index=20),
    ]

    retained, summary = _deduplicate_candidates(candidates)

    assert [row.index for row in retained] == [1, 3]
    assert retained[0].capture_time_ns == 2000
    assert retained[0].source_frame_index == 8
    assert summary["dropped_candidate_count"] == 2
    assert summary["dropped_candidate_indices"] == [0, 2]
    assert summary["frozen_run_count"] == 1
    assert summary["frozen_runs"][0]["span_s"] == 2.0
    assert summary["frozen_span_s"]["max"] == 2.0


def test_slight_encoding_noise_drops_only_with_strong_spatial_evidence(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    base = rng.integers(20, 220, (90, 160), dtype=np.uint8)
    noisy = np.clip(base.astype(np.int16) + rng.integers(-1, 2, base.shape), 0, 255).astype(np.uint8)
    first = _candidate(0, 0.0, 0, digest=b"base")
    second = _candidate(1, 0.25, 0, digest=b"noise")
    first.small_gray = base
    second.small_gray = noisy
    monkeypatch.setattr(processing, "_visual_edge", lambda *_args: _strong_repeat_edge())

    retained, summary = _deduplicate_candidates([first, second])

    assert [row.index for row in retained] == [0]
    assert summary["spatial_repeat_drop_count"] == 1
    assert summary["exact_content_drop_count"] == 0


def test_anchor_comparison_preserves_cumulative_slow_motion(monkeypatch) -> None:
    monkeypatch.setattr(processing, "_visual_edge", lambda *_args: _strong_repeat_edge())
    candidates = [_candidate(index, index * 0.25, index) for index in range(5)]

    retained, summary = _deduplicate_candidates(candidates)

    assert [row.index for row in retained] == [0, 2, 4]
    assert summary["dropped_candidate_indices"] == [1, 3]


def test_scene_jump_with_weak_connectivity_is_retained() -> None:
    first = _candidate(0, 0.0, 20, digest=b"first")
    jump = _candidate(1, 0.5, 220, digest=b"jump")

    retained, summary = _deduplicate_candidates([first, jump])

    assert [row.index for row in retained] == [0, 1]
    assert summary["dropped_candidate_count"] == 0


def test_repair_and_endpoint_do_not_reintroduce_frozen_candidates(monkeypatch) -> None:
    candidates = [
        _candidate(0, 0.0, 20, digest=b"a", quality_score=0.9),
        _candidate(1, 0.4, 20, digest=b"a", quality_score=0.1),
        _candidate(2, 0.8, 80, digest=b"c", quality_score=0.4),
        _candidate(3, 1.2, 180, digest=b"d", quality_score=0.8),
        _candidate(4, 1.6, 180, digest=b"d", quality_score=0.2),
        _candidate(5, 2.0, 180, digest=b"d", quality_score=0.1),
    ]
    weak_pairs = {(0, 2), (0, 3), (0, 4)}

    def edge(left: _Candidate, right: _Candidate) -> dict[str, object]:
        if (left.index, right.index) in weak_pairs:
            return {
                **_strong_repeat_edge(appearance_delta=4.0),
                "passes_connectivity": False,
                "match_count": 2,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "source_coverage": 0.0,
                "target_coverage": 0.0,
            }
        return _strong_repeat_edge(appearance_delta=0.0 if left.index == right.index else 8.0)

    monkeypatch.setattr(processing, "_visual_edge", edge)
    settings = FramePreparationSettings(
        candidate_fps=4.0,
        min_keyframe_interval_s=0.3,
        max_keyframe_interval_s=1.25,
        max_selected_frames=8,
    )

    selected, reasons, _, _, _ = _select_keyframes(candidates, settings)

    assert selected == [0, 2, 4]
    assert reasons[2] == "connectivity_repair"
    assert 5 not in selected
    assert all(
        not processing._is_frozen_repeat(candidates[left], candidates[right], edge(candidates[left], candidates[right]))[0]
        for left, right in zip(selected, selected[1:])
    )


def test_frozen_gap_exceeds_selection_interval_without_fabrication(monkeypatch) -> None:
    monkeypatch.setattr(processing, "_visual_edge", lambda *_args: _strong_repeat_edge(appearance_delta=8.0))
    candidates = [
        _candidate(0, 0.0, 40, digest=b"frozen"),
        _candidate(1, 1.0, 40, digest=b"frozen"),
        _candidate(2, 2.0, 40, digest=b"frozen"),
        _candidate(3, 5.0, 100, digest=b"changed"),
    ]
    deduplicated, summary = _deduplicate_candidates(candidates)
    selected, _, _, _, _ = _select_keyframes(
        deduplicated,
        FramePreparationSettings(candidate_fps=4.0, max_keyframe_interval_s=1.25),
    )

    assert summary["frozen_runs"][0]["span_s"] == 2.0
    assert [deduplicated[index].index for index in selected] == [0, 3]
    assert deduplicated[selected[1]].timestamp_s - deduplicated[selected[0]].timestamp_s == 5.0


def test_encoded_vfr_selection_preserves_pts_and_does_not_fill_gap(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    colors = ((20, 40, 200), (20, 40, 200), (40, 180, 30), (40, 180, 30))
    for index, color in enumerate(colors):
        image = np.full((64, 96, 3), color, dtype=np.uint8)
        cv2.putText(image, str(index), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        assert cv2.imwrite(str(source_dir / f"frame-{index}.png"), image)
    concat = tmp_path / "frames.txt"
    concat.write_text(
        "\n".join(
            [*(f"file '{source_dir / f'frame-{index}.png'}'\nduration {duration}" for index, duration in enumerate((0.10, 0.20, 1.10, 0.20))),
             f"file '{source_dir / 'frame-3.png'}'"]
        ),
        encoding="utf-8",
    )
    video = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "1000", str(video),
        ],
        check=True,
    )
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    paths = _extract_candidates(
        video,
        candidate_dir,
        4.0,
        FramePreparationSettings(candidate_fps=4.0, max_candidate_frames=16, candidate_edge_px=96, max_edge_px=96),
    )
    timestamps = json.loads((candidate_dir / "candidate_timestamps_s.json").read_text())

    assert len(paths) >= 2
    assert len(paths) == len(timestamps)
    assert all(timestamps[index] <= timestamps[index + 1] for index in range(len(timestamps) - 1))
    assert max(
        timestamps[index + 1] - timestamps[index] for index in range(len(timestamps) - 1)
    ) > 0.5


def test_encoded_pts_origin_is_retained_in_candidate_timestamp(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(2):
        image = np.zeros((90, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (20 + index * 4, 20), (120 + index * 4, 70), (220, 120, 40), -1)
        path = tmp_path / f"candidate-{index}.jpg"
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    candidates = _analyze_candidates(
        paths,
        4.0,
        160,
        lambda _fraction, _message: None,
        encoded_timestamps_s=[12.0, 12.25],
        timestamp_source="encoded_pts",
    )

    assert [candidate.timestamp_s for candidate in candidates] == [12.0, 12.25]
    assert [candidate.timestamp_source for candidate in candidates] == [
        "encoded_pts",
        "encoded_pts",
    ]


def test_encoded_pts_probe_fails_at_finite_read_cap(monkeypatch, tmp_path: Path) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = iter(("0.0\n", "0.1\n", "0.2\n", "0.3\n"))
            self.stderr = iter(())
            self.killed = False

        def poll(self) -> None:
            return None if not self.killed else -9

        def kill(self) -> None:
            self.killed = True

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        def wait(self) -> int:
            return 0

    process = FakeProcess()
    command_rows: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        command_rows.append(command)
        return process

    monkeypatch.setattr(processing.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(processing, "_MAX_ENCODED_PTS_FRAMES", 3)

    with pytest.raises(processing.FramePreparationError):
        processing._probe_encoded_frame_timestamps(tmp_path / "video.webm")

    assert process.killed is True
    assert "%+#4" in command_rows[0]
