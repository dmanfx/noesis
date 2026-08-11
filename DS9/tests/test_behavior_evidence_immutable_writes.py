from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str) -> ModuleType:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


identity = _load(
    "immutable_evidence_identity_gate",
    "DS9/scripts/ds9_identity_shadow_live_gate.py",
)
floorplan = _load(
    "immutable_evidence_floorplan_gate",
    "DS9/scripts/ds9_floorplan_live_gate.py",
)
semantic = _load(
    "immutable_evidence_semantic_gate",
    "DS9/scripts/ds9_semantic_observation_smoke_test.py",
)
v3dt = _load(
    "immutable_evidence_v3dt_gate",
    "DS9/scripts/v3dt_world_contract_smoke_test.py",
)
wholebody = _load(
    "immutable_evidence_wholebody_gate",
    "DS9/scripts/wholebody49_occupied_scene_smoke_test.py",
)
media = _load(
    "immutable_evidence_media_gate",
    "DS9/scripts/wholebody49_media_decode_gate.py",
)


Writer = Callable[[Path, Mapping[str, object]], object]


@pytest.mark.parametrize(
    ("name", "filename", "writer"),
    (
        (
            "identity",
            identity.BASELINE_CANONICAL_REPORT_FILENAME,
            identity._write_private_json,
        ),
        (
            "floorplan",
            floorplan.CANONICAL_REPORT_FILENAME,
            floorplan._write_private_json,
        ),
        (
            "semantic",
            semantic.CANONICAL_REPORT_FILENAME,
            semantic._write_report,
        ),
        ("v3dt", v3dt.CANONICAL_REPORT_FILENAME, v3dt._write_private_json),
        (
            "wholebody",
            wholebody.CANONICAL_REPORT_FILENAME,
            wholebody._write_private_json,
        ),
        ("media", media.CANONICAL_REPORT_FILENAME, media._write_private_json),
    ),
)
def test_behavior_evidence_writers_are_owner_private_and_no_replace(
    tmp_path: Path,
    name: str,
    filename: str,
    writer: Writer,
) -> None:
    parent = tmp_path / name
    parent.mkdir(mode=0o700)
    destination = parent / filename

    writer(destination, {"generation": 1})
    original = destination.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert destination.stat().st_nlink == 1

    with pytest.raises((ValueError, RuntimeError), match="already exists"):
        writer(destination, {"generation": 2})

    assert destination.read_bytes() == original
    assert destination.stat().st_nlink == 1
    assert not list(parent.glob(f".{filename}.tmp-*"))


@pytest.mark.parametrize(
    "writer",
    (
        identity._write_private_json,
        floorplan._write_private_json,
        semantic._write_report,
        v3dt._write_private_json,
        wholebody._write_private_json,
        media._write_private_json,
    ),
)
def test_behavior_evidence_writers_reject_noncanonical_filenames(
    tmp_path: Path,
    writer: Writer,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="filename is not canonical"):
        writer(parent / "substituted.json", {"generation": 1})
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize(
    ("filename", "writer"),
    (
        (
            identity.BASELINE_CANONICAL_REPORT_FILENAME,
            identity._write_private_json,
        ),
        (v3dt.CANONICAL_REPORT_FILENAME, v3dt._write_private_json),
        (
            wholebody.CANONICAL_REPORT_FILENAME,
            wholebody._write_private_json,
        ),
        (media.CANONICAL_REPORT_FILENAME, media._write_private_json),
    ),
)
@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), float("-inf")))
def test_create_once_writers_reject_nonfinite_before_publication(
    tmp_path: Path,
    filename: str,
    writer: Writer,
    nonfinite: float,
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / filename

    with pytest.raises(ValueError):
        writer(destination, {"metric": nonfinite})

    assert not destination.exists()
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize(
    ("module", "parser_name", "source_filename", "report_filename"),
    (
        (
            identity,
            "parse_args",
            identity.BASELINE_SOURCE_TRANSCRIPT_FILENAME,
            identity.BASELINE_CANONICAL_REPORT_FILENAME,
        ),
        (
            floorplan,
            "parse_args",
            floorplan.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            floorplan.CANONICAL_REPORT_FILENAME,
        ),
        (
            semantic,
            "parse_args",
            semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            semantic.CANONICAL_REPORT_FILENAME,
        ),
        (
            v3dt,
            "_parse_args",
            v3dt.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            v3dt.CANONICAL_REPORT_FILENAME,
        ),
        (
            wholebody,
            "_parse_args",
            wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            wholebody.CANONICAL_REPORT_FILENAME,
        ),
        (
            media,
            "_parse_args",
            media.CANONICAL_SOURCE_FILENAME,
            media.CANONICAL_REPORT_FILENAME,
        ),
    ),
)
def test_behavior_gate_main_requires_fresh_bundle_before_filling_partial_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    parser_name: str,
    source_filename: str,
    report_filename: str,
) -> None:
    parent = tmp_path / module.__name__
    parent.mkdir(mode=0o700)
    source = parent / source_filename
    report = parent / report_filename
    source.write_bytes(b"existing immutable source\n")
    source.chmod(0o600)
    arguments = SimpleNamespace(
        source_out=source,
        out=report,
        snapshot_out=parent / semantic.CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
        min_bbox3d_coverage=0.95,
        pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
        calibration_config=REPO_ROOT / "config/camera_calibration.json",
        alignment_config=REPO_ROOT / "config/ply_alignment.json",
        launcher_evidence_dir=parent,
        session_id="fresh-bundle-test",
        runtime_lane="v3dt",
    )
    monkeypatch.setattr(module, parser_name, lambda *_args, **_kwargs: arguments)
    if module is v3dt:
        monkeypatch.setattr(module, "build_config_binding", lambda **_kwargs: {})

    result = module.main([]) if parser_name == "_parse_args" else module.main()

    assert result == 1
    assert source.read_bytes() == b"existing immutable source\n"
    assert not report.exists()
