import json
from pathlib import Path

import pytest

from tools.mapanything_phone_scan import diagnose_alignment as diagnostic


def _arguments(tmp_path: Path):
    scan = tmp_path / "scan"
    scan.mkdir()
    state = scan / "scan_state.json"
    state.write_text(json.dumps({"id": "fixture", "outputs": {"view_count": 2}}))
    args = ["--scan-dir", str(scan), "--camera-id", "fixture-camera",
            "--target-revision", str(tmp_path / "target"),
            "--calibration-path", str(tmp_path / "calibration.json"),
            "--output-dir", str(tmp_path / "diagnostic")]
    return args, state


def test_early_rejection_is_saved_without_changing_scan(tmp_path, monkeypatch):
    args, state = _arguments(tmp_path)
    before = state.read_bytes()

    def reject(*_args, **_kwargs):
        raise diagnostic.NoesisAlignmentError("camera reference faces away")

    monkeypatch.setattr(diagnostic, "run_noesis_alignment", reject)
    assert diagnostic.main(args) == 2
    summary = json.loads((tmp_path / "diagnostic/run_summary.json").read_text())
    assert summary["status"] == "rejected"
    assert summary["error"] == "camera reference faces away"
    assert state.read_bytes() == before
    assert summary["command"][2] == "tools.mapanything_phone_scan.diagnose_alignment"


def test_diagnostic_output_cannot_overwrite_scan_or_existing_experiment(tmp_path):
    args, state = _arguments(tmp_path)
    with pytest.raises(SystemExit):
        diagnostic.main([*args[:-1], str(state.parent / "alignment")])
    existing = tmp_path / "diagnostic"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("previous experiment")
    with pytest.raises(SystemExit):
        diagnostic.main(args)
    assert sentinel.read_text() == "previous experiment"


def test_explicit_relative_source_paths_bind_the_comparison(tmp_path, monkeypatch):
    args, _state = _arguments(tmp_path)
    alternate = tmp_path / "comparison"
    alternate.mkdir()
    raw = alternate / "raw"
    raw.mkdir()
    manifest = alternate / "outputs.json"
    manifest.write_text(json.dumps({"view_count": 3, "provider": "comparison"}))
    monkeypatch.chdir(tmp_path)

    def run(_scan, _out, outputs, _settings, _progress, **kwargs):
        assert outputs["provider"] == "comparison"
        assert kwargs["source_raw_root"] == raw.resolve()
        assert kwargs["source_output_manifest"] == manifest.resolve()
        return {"quality_gate": {"passed": True}}

    monkeypatch.setattr(diagnostic, "run_noesis_alignment", run)
    assert diagnostic.main([*args, "--source-raw-root", "comparison/raw",
                            "--source-output-manifest", "comparison/outputs.json"]) == 0
