from __future__ import annotations

from pathlib import Path

from noesis.pipelines import deepstream_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_pipeline_module_resolves_inside_ds9() -> None:
    assert Path(deepstream_pipeline.__file__).resolve() == (
        REPO_ROOT / "DS9" / "noesis" / "pipelines" / "deepstream_pipeline.py"
    ).resolve()
    assert deepstream_pipeline.DeepStreamPipeline.__name__ == "DeepStreamPipeline"


def test_active_ds9_code_imports_only_the_canonical_pipeline_module() -> None:
    active_sources = (
        REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",
        REPO_ROOT / "noesis" / "server" / "analytics_api.py",
        REPO_ROOT / "noesis" / "server" / "depth_api.py",
        REPO_ROOT / "noesis" / "server" / "scene_prior_api.py",
    )
    for path in active_sources:
        source = path.read_text(encoding="utf-8")
        assert "deepstream_pipeline" in source
        assert "ds8_pipeline" not in source


def test_runtime_and_pipeline_emit_ds9_identity() -> None:
    runtime_source = (REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py").read_text(
        encoding="utf-8"
    )
    pipeline_source = Path(deepstream_pipeline.__file__).read_text(encoding="utf-8")

    assert runtime_source.count('"stack": "ds9"') == 2
    assert '"stack": "ds8"' not in runtime_source
    assert 'DSPipeline("noesis-ds9")' in pipeline_source
    assert 'DSPipeline("noesis-ds8")' not in pipeline_source
    assert "NOESIS_DS9_FPS_PROBE" in pipeline_source
    assert "NOESIS_DS8_" not in pipeline_source


def test_servicemaker_hooks_use_neutral_identifiers() -> None:
    hooks_source = (REPO_ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py").read_text(
        encoding="utf-8"
    )

    assert "handle_servicemaker_frame" in hooks_source
    assert "handle_servicemaker_batch" in hooks_source
    assert "handle_frame_ds8" not in hooks_source
    assert "handle_batch_ds8" not in hooks_source
