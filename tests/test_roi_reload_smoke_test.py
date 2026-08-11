from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import roi_reload_smoke_test as gate


def _stage() -> dict[str, object]:
    return {
        "stage": "exclude",
        "config_width": 1920,
        "config_height": 1080,
        "defaults": {"class_ids": [-1], "inverse_roi": False},
        "streams": [
            {
                "stream_id": "0",
                "label": "living_room",
                "enable": False,
                "rois": [
                    {
                        "id": "MIRROR",
                        "description": "original",
                        "points_px": [
                            [10.0, 10.0],
                            [20.0, 10.0],
                            [20.0, 20.0],
                        ],
                    }
                ],
            },
            {
                "stream_id": "1",
                "label": "kitchen",
                "enable": True,
                "rois": [
                    {
                        "id": "ART",
                        "description": None,
                        "points_px": [
                            [100.0, 100.0],
                            [200.0, 100.0],
                            [200.0, 200.0],
                        ],
                    }
                ],
            },
        ],
    }


def _receipt(
    sequence: int,
    digest: str,
    *,
    objects_removed: int,
) -> dict[str, object]:
    return {
        "request_sequence": sequence,
        "accepted_sequence": sequence,
        "failed_sequence": 0,
        "active_config_sha256": digest,
        "reload_error_count": 0,
        "objects_removed_count": objects_removed,
    }


def _phase(
    name: str,
    receipt: dict[str, object],
    *,
    first: int,
    last: int,
    real_person_frames: int = 0,
    real_person_advances: int = 0,
    empty_frames: int = 0,
) -> gate.PhaseEvidence:
    return gate.PhaseEvidence(
        name=name,
        target_messages=2 if last > first else 1,
        first_sequence=first,
        last_sequence=last,
        sequence_advances=1 if last > first else 0,
        first_frame_id=first,
        last_frame_id=last,
        frame_advances=1 if last > first else 0,
        real_person_frames=real_person_frames,
        real_person_frame_advances=real_person_advances,
        max_consecutive_empty_frames=empty_frames,
        native_receipt=dict(receipt),
    )


def _args(tmp_path: Path) -> SimpleNamespace:
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text(
        "cameras:\n  0:\n    name: living-room\n  1:\n    name: kitchen\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        ws="ws://127.0.0.1:6008",
        rest="http://127.0.0.1:8080",
        camera="kitchen",
        cameras_config=cameras,
        pipeline_config=Path("config/infer.yaml"),
        baseline_timeout=10.0,
        excluded_timeout=10.0,
        restore_timeout=10.0,
        evidence=tmp_path / "evidence.json",
        no_spawn=True,
        hot_restore=True,
        auth_token_file=None,
    )


def _response(
    stage: dict[str, object], receipt: dict[str, object]
) -> dict[str, object]:
    return {
        **stage,
        "config_path": "/private/runtime/state/nvdsanalytics.yaml",
        "reloaded": True,
        "reload_receipt": receipt,
    }


def test_full_frame_stage_changes_only_selected_stream() -> None:
    original = _stage()
    original_hash = gate._json_sha256(original)

    temporary = gate._temporary_full_frame_stage(original, "1")

    assert gate._json_sha256(original) == original_hash
    assert temporary["streams"][0] == original["streams"][0]
    target = temporary["streams"][1]
    assert target["enable"] is True
    assert target["rois"] == [
        {
            "id": gate.FULL_FRAME_ROI_ID,
            "description": "Occupied-scene hot-restore gate",
            "points_px": [
                [0.0, 0.0],
                [1920.0, 0.0],
                [1920.0, 1080.0],
                [0.0, 1080.0],
            ],
        }
    ]


def test_real_person_predicate_rejects_synthetic_or_malformed_tracks() -> None:
    synthetic = {
        "camera_id": "kitchen",
        "stable_id": 1,
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "frame_id": 4,
        "class_id": 0,
    }
    real = {
        **synthetic,
        "tracker_id": 81,
        "confidence": 0.84,
    }

    assert gate._real_person_frame_id(synthetic) is None
    assert gate._real_person_frame_id(real) == 4
    assert gate._real_person_frame_id({**real, "bbox": [0, 0, 0, 10]}) is None
    assert gate._real_person_frame_id({**real, "confidence": float("nan")}) is None


def test_tracking_requires_monotonic_world_and_real_frame_sequences() -> None:
    phase = gate.PhaseEvidence(name="baseline")
    base_track = {
        "class_id": 0,
        "tracker_id": 9,
        "bbox": [1.0, 2.0, 30.0, 60.0],
        "confidence": 0.75,
    }
    gate._record_tracking(
        phase,
        {
            "camera_id": "kitchen",
            "frame_id": 100,
            "world_snapshot": {"sequence": 11},
            "tracks": [{**base_track, "frame_id": 100}],
        },
        camera_id="kitchen",
    )
    gate._record_tracking(
        phase,
        {
            "camera_id": "kitchen",
            "frame_id": 101,
            "world_snapshot": {"sequence": 14},
            "tracks": [{**base_track, "frame_id": 101}],
        },
        camera_id="kitchen",
    )

    assert phase.target_messages == 2
    assert phase.sequence_advances == 1
    assert phase.real_person_frames == 2
    assert phase.real_person_frame_advances == 1

    with pytest.raises(gate.GateError, match="target_tracking_sequence_not_advancing"):
        gate._record_tracking(
            phase,
            {
                "camera_id": "kitchen",
                "frame_id": 102,
                "world_snapshot": {"sequence": 14},
                "tracks": [],
            },
            camera_id="kitchen",
        )


def test_tracking_accepts_advancing_canonical_empty_frames() -> None:
    phase = gate.PhaseEvidence(name="excluded")
    for frame_id, sequence in ((200, 20), (201, 21), (202, 22)):
        gate._record_tracking(
            phase,
            {
                "camera_id": "kitchen",
                "frame_id": frame_id,
                "world_snapshot": {"sequence": sequence},
                "tracks": [],
            },
            camera_id="kitchen",
        )

    assert phase.target_messages == 3
    assert phase.frame_advances == 2
    assert phase.sequence_advances == 2
    assert phase.max_consecutive_empty_frames == 3


def test_reload_receipt_requires_exact_acceptance_hash_and_zero_errors() -> None:
    original = _receipt(0, "a" * 64, objects_removed=2)
    accepted = _receipt(1, "b" * 64, objects_removed=2)

    gate._validate_initial_receipt(original)
    gate._validate_reload_receipt(accepted, previous=original)

    with pytest.raises(gate.GateError, match="native_reload_not_accepted"):
        gate._validate_reload_receipt(
            {**accepted, "accepted_sequence": 0}, previous=original
        )
    with pytest.raises(gate.GateError, match="native_reload_error_count_nonzero"):
        gate._validate_reload_receipt(
            {**accepted, "reload_error_count": 1}, previous=original
        )
    with pytest.raises(gate.GateError, match="native_active_hash_mismatch"):
        gate._validate_reload_receipt(
            accepted, previous=original, expected_hash="c" * 64
        )


def test_rest_client_uses_internal_bearer_without_query_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "roi-gate-internal-secret-" + ("x" * 48)
    token_path = tmp_path / "gateway-token"
    token_path.write_text(secret + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    auth = gate.load_required_internal_auth(token_path)
    observed: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def read(self, _limit: int) -> bytes:
            return b'{"stage":"exclude"}'

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        observed["request"] = request
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)

    result = gate._request_json(
        "GET", "http://127.0.0.1:8080/api/v1/analytics/rois", auth
    )

    request = observed["request"]
    assert result == {"stage": "exclude"}
    assert request.get_header("Authorization") == f"Bearer {secret}"
    assert secret not in request.full_url


def test_unoccupied_baseline_is_blocked_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    original = _stage()
    original_receipt = _receipt(0, "a" * 64, objects_removed=4)
    posted: list[dict[str, object]] = []

    monkeypatch.setattr(gate, "_get_original_stage", lambda *_a, **_k: original)

    async def fake_collect(*_args: object, **kwargs: object) -> gate.PhaseEvidence:
        assert kwargs["name"] == "baseline"
        return _phase("baseline", original_receipt, first=10, last=11)

    monkeypatch.setattr(gate, "_collect_phase", fake_collect)
    monkeypatch.setattr(
        gate,
        "_post_stage",
        lambda _rest, _auth, stage: posted.append(dict(stage)) or {},
    )

    report = gate._run_gate(args, object())

    assert report["ok"] is False
    assert report["status"] == "blocked"
    assert report["errors"] == ["baseline_real_person_absent"]
    assert posted == []


def test_failure_after_mutation_still_restores_and_proves_person_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    original = _stage()
    original_receipt = _receipt(0, "a" * 64, objects_removed=4)
    temporary_receipt = _receipt(1, "b" * 64, objects_removed=4)
    restore_receipt = _receipt(2, "a" * 64, objects_removed=7)
    posts: list[str] = []
    get_calls = 0

    def fake_get(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal get_calls
        get_calls += 1
        return original

    def fake_post(
        _rest: str, _auth: object, stage: dict[str, object]
    ) -> dict[str, object]:
        is_restore = gate._json_sha256(stage) == gate._json_sha256(original)
        posts.append("restore" if is_restore else "temporary")
        receipt = restore_receipt if is_restore else temporary_receipt
        return _response(stage, receipt)

    async def fake_collect(*_args: object, **kwargs: object) -> gate.PhaseEvidence:
        name = str(kwargs["name"])
        if name == "baseline":
            return _phase(
                name,
                original_receipt,
                first=10,
                last=11,
                real_person_frames=2,
                real_person_advances=1,
            )
        if name == "excluded":
            # Continuing frames are present, but the person never disappears.
            return _phase(
                name, temporary_receipt, first=12, last=13, empty_frames=0
            )
        return _phase(
            name,
            restore_receipt,
            first=14,
            last=15,
            real_person_frames=1,
        )

    monkeypatch.setattr(gate, "_get_original_stage", fake_get)
    monkeypatch.setattr(gate, "_post_stage", fake_post)
    monkeypatch.setattr(gate, "_collect_phase", fake_collect)

    report = gate._run_gate(args, object())

    assert posts == ["temporary", "restore"]
    assert get_calls == 3
    assert report["ok"] is False
    assert report["status"] == "fail"
    assert report["errors"] == ["excluded_person_tracks_remain"]
    assert report["checks"]["restore_native_receipt_exact"] is True
    assert report["checks"]["objects_removed_count_rose"] is True
    assert report["checks"]["restored_get_semantic_equality"] is True
    assert report["checks"]["restored_person_returned"] is True


def test_complete_occupied_hot_restore_transaction_passes_all_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    original = _stage()
    original_receipt = _receipt(0, "a" * 64, objects_removed=5)
    temporary_receipt = _receipt(1, "b" * 64, objects_removed=5)
    restore_receipt = _receipt(2, "a" * 64, objects_removed=11)

    monkeypatch.setattr(gate, "_get_original_stage", lambda *_a, **_k: original)

    def fake_post(
        _rest: str, _auth: object, stage: dict[str, object]
    ) -> dict[str, object]:
        is_restore = gate._json_sha256(stage) == gate._json_sha256(original)
        return _response(
            stage, restore_receipt if is_restore else temporary_receipt
        )

    async def fake_collect(*_args: object, **kwargs: object) -> gate.PhaseEvidence:
        name = str(kwargs["name"])
        if name == "baseline":
            return _phase(
                name,
                original_receipt,
                first=30,
                last=31,
                real_person_frames=2,
                real_person_advances=1,
            )
        if name == "excluded":
            return _phase(
                name, temporary_receipt, first=32, last=33, empty_frames=3
            )
        return _phase(
            name,
            restore_receipt,
            first=34,
            last=35,
            real_person_frames=1,
        )

    monkeypatch.setattr(gate, "_post_stage", fake_post)
    monkeypatch.setattr(gate, "_collect_phase", fake_collect)

    report = gate._run_gate(args, object())

    assert report["ok"] is True
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["errors"] == []
    assert report["stage_hashes"]["original"] == report["stage_hashes"]["restored"]
    assert report["receipts"]["temporary"] == temporary_receipt
    assert report["receipts"]["restored"] == restore_receipt
    serialized = json.dumps(report, sort_keys=True)
    assert "://" not in serialized
    assert "Bearer" not in serialized
    assert "points_px" not in serialized


def test_restore_failure_is_terminal_even_after_successful_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    original = _stage()
    original_receipt = _receipt(0, "a" * 64, objects_removed=1)
    temporary_receipt = _receipt(1, "b" * 64, objects_removed=1)
    post_count = 0

    monkeypatch.setattr(gate, "_get_original_stage", lambda *_a, **_k: original)

    def fake_post(
        _rest: str, _auth: object, stage: dict[str, object]
    ) -> dict[str, object]:
        nonlocal post_count
        post_count += 1
        if post_count == 2:
            raise gate.GateError("restore_transport_failed")
        return _response(stage, temporary_receipt)

    async def fake_collect(*_args: object, **kwargs: object) -> gate.PhaseEvidence:
        name = str(kwargs["name"])
        if name == "baseline":
            return _phase(
                name,
                original_receipt,
                first=20,
                last=21,
                real_person_frames=2,
                real_person_advances=1,
            )
        return _phase(
            name, temporary_receipt, first=22, last=23, empty_frames=3
        )

    monkeypatch.setattr(gate, "_post_stage", fake_post)
    monkeypatch.setattr(gate, "_collect_phase", fake_collect)

    report = gate._run_gate(args, object())

    assert post_count == 2
    assert report["ok"] is False
    assert report["status"] == "restore_failed"
    assert report["errors"] == ["restore_transport_failed"]


def test_private_evidence_is_bounded_owner_only_and_contains_no_endpoints(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "private" / "roi.json"
    payload = {
        "contract": gate.CONTRACT,
        "ok": True,
        "camera_id": "kitchen",
        "errors": [],
    }

    gate._write_private_json(report_path, payload)

    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    saved = report_path.read_text(encoding="utf-8")
    assert json.loads(saved) == payload
    assert "://" not in saved
    assert "Bearer" not in saved


def test_ds9_wrapper_execs_canonical_gate_with_argv_and_exit_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_path = (
        Path(__file__).resolve().parents[1]
        / "DS9"
        / "scripts"
        / "roi_reload_smoke_test.py"
    )
    spec = importlib.util.spec_from_file_location("ds9_roi_wrapper_test", wrapper_path)
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    observed: list[str] = []

    def fake_execv(executable: str, argv: list[str]) -> None:
        observed.extend([executable, *argv])

    monkeypatch.setattr(wrapper.os, "execv", fake_execv)
    with pytest.raises(AssertionError, match="returned unexpectedly"):
        wrapper.main(["--hot-restore", "--camera", "kitchen"])

    assert observed == [
        sys.executable,
        sys.executable,
        str(wrapper.CANONICAL_GATE),
        "--hot-restore",
        "--camera",
        "kitchen",
    ]
    assert wrapper.CANONICAL_GATE == (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "roi_reload_smoke_test.py"
    )
