from __future__ import annotations

import importlib.util
import json
import stat
import sys
import configparser
import fcntl
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"
SCRIPTS_ROOT = DS9_ROOT / "scripts"
for import_root in (DS9_ROOT, SCRIPTS_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from noesis.mapanything_profiles import (  # noqa: E402
    MapAnythingProfileError,
    get_mapanything_profile,
    resolve_runtime_mapanything_profile,
)


def _load_tool():
    path = SCRIPTS_ROOT / "mapanything_profile_tool.py"
    spec = importlib.util.spec_from_file_location("ds9_mapanything_profile_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def test_canonical_and_hr0_profiles_are_exact_and_isolated() -> None:
    canonical = get_mapanything_profile("canonical_294x518_b3_fp32")
    hr0 = get_mapanything_profile("hr0_378x672_b3_fp32")

    assert canonical.status == "canonical"
    assert canonical.fixed_shape == "3x3x294x518"
    assert canonical.output_bytes_per_frame == 1_827_504
    assert canonical.engine.endswith("mapanything_images_294x518_b3_fp32.plan")

    assert hr0.status == "candidate"
    assert hr0.fixed_shape == "3x3x378x672"
    assert hr0.pixels_per_frame == 254_016
    assert hr0.input_batch_bytes == 9_144_576
    assert hr0.output_bytes_per_frame == 3_048_192
    assert hr0.output_batch_bytes == 9_144_576
    assert "/candidates/" in hr0.engine
    assert "/onnx/candidates/hr0_378x672_b3_fp32/" in hr0.onnx
    assert hr0.engine != canonical.engine
    assert hr0.onnx != canonical.onnx
    assert hr0.infer_config != canonical.infer_config
    assert hr0.trtexec_build_args[:3] == (
        "--minShapes=images:3x3x378x672",
        "--optShapes=images:3x3x378x672",
        "--maxShapes=images:3x3x378x672",
    )
    artifact_root = REPO_ROOT / "artifact-root-fixture"
    assert hr0.workspace_path("engine", artifact_root=artifact_root) == (
        artifact_root
        / "models/engines/candidates/mapanything_images_378x672_b3_fp32.plan"
    )

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(REPO_ROOT / hr0.infer_config)
    properties = parser["property"]
    assert properties["infer-dims"] == "3;378;672"
    assert int(properties["batch-size"]) == hr0.batch_size
    assert int(properties["interval"]) == hr0.interval
    assert properties["model-engine-file"] == hr0.engine
    active_yaml = (DS9_ROOT / "config/infer.yaml").read_text(encoding="utf-8")
    assert hr0.name not in active_yaml
    assert Path(hr0.engine).name not in active_yaml


def test_v11_control_matches_canonical_resolution_but_is_fully_isolated() -> None:
    canonical = get_mapanything_profile("canonical_294x518_b3_fp32")
    control = get_mapanything_profile("v11_control_294x518_b3_fp32")
    hr0 = get_mapanything_profile("hr0_378x672_b3_fp32")

    assert get_mapanything_profile().name == canonical.name
    assert control.status == "candidate"
    assert control.fixed_shape == canonical.fixed_shape == "3x3x294x518"
    assert control.precision == canonical.precision == hr0.precision == "fp32"
    assert control.patch_size == canonical.patch_size == hr0.patch_size == 14
    assert control.output_layers == canonical.output_layers == hr0.output_layers
    assert control.trtexec_build_args[:3] == (
        "--minShapes=images:3x3x294x518",
        "--optShapes=images:3x3x294x518",
        "--maxShapes=images:3x3x294x518",
    )

    for field in ("onnx", "engine", "infer_config", "functional_fixture"):
        assert len(
            {
                str(getattr(canonical, field)),
                str(getattr(control, field)),
                str(getattr(hr0, field)),
            }
        ) == 3
    assert "/onnx/candidates/v11_control_294x518_b3_fp32/" in control.onnx
    assert "/engines/candidates/" in control.engine
    assert "/engine_validation/mapanything/v11_control_294x518/" in (
        control.functional_fixture
    )

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(REPO_ROOT / control.infer_config)
    properties = parser["property"]
    assert properties["infer-dims"] == "3;294;518"
    assert int(properties["batch-size"]) == control.batch_size
    assert int(properties["interval"]) == control.interval
    assert properties["model-engine-file"] == control.engine

    plan = tool._command_plan(
        control,
        mapanything_repo=REPO_ROOT / "src/mapanything",
        trtexec="/usr/bin/trtexec",
    )
    export = plan["commands"]["export_onnx"]
    build = plan["commands"]["build_candidate"]
    assert export[export.index("--h") + 1] == "294"
    assert export[export.index("--w") + 1] == "518"
    assert export[export.index("--hf-revision") + 1] == (
        tool.MAPANYTHING_HF_REVISION
    )
    assert plan["model_source"]["mapanything_version"] == "1.1.3"
    assert plan["model_source"]["mapanything_commit"] == (
        tool.MAPANYTHING_SOURCE_COMMIT
    )
    assert any(control.engine in item for item in build)
    assert not any(canonical.engine in item or hr0.engine in item for item in build)
    assert plan["canonical_runtime_modified"] is False
    assert plan["promotion_state"] == "isolated_candidate_only"

    active_yaml = (DS9_ROOT / "config/infer.yaml").read_text(encoding="utf-8")
    assert control.name not in active_yaml
    assert Path(control.engine).name not in active_yaml
    assert Path(control.infer_config).name not in active_yaml


def test_runtime_profile_defaults_canonical_and_rejects_crossed_engine() -> None:
    canonical = resolve_runtime_mapanything_profile(
        {
            "batch_size": 3,
            "engine": (
                "DS9/models/engines/"
                "mapanything_images_294x518_b3_fp32.plan"
            ),
        }
    )
    assert canonical.name == "canonical_294x518_b3_fp32"

    hr0 = resolve_runtime_mapanything_profile(
        {
            "profile": "hr0_378x672_b3_fp32",
            "batch_size": 3,
            "config-file-path": (
                "DS9/pipelines/config_infer_secondary_mapanything_hr0.ini"
            ),
            "engine": (
                "DS9/models/engines/candidates/"
                "mapanything_images_378x672_b3_fp32.plan"
            ),
        }
    )
    assert (hr0.input_height, hr0.input_width) == (378, 672)

    with pytest.raises(MapAnythingProfileError, match="requires engine"):
        resolve_runtime_mapanything_profile(
            {
                "profile": "hr0_378x672_b3_fp32",
                "batch_size": 3,
                "config-file-path": (
                    "DS9/pipelines/config_infer_secondary_mapanything_hr0.ini"
                ),
                "engine": canonical.engine,
            }
        )
    with pytest.raises(MapAnythingProfileError, match="requires inference config"):
        resolve_runtime_mapanything_profile(
            {
                "profile": "hr0_378x672_b3_fp32",
                "batch_size": 3,
                "engine": hr0.engine,
            }
        )
    with pytest.raises(MapAnythingProfileError, match="requires inference config"):
        resolve_runtime_mapanything_profile(
            {
                "profile": "hr0_378x672_b3_fp32",
                "batch_size": 3,
                "engine": hr0.engine,
                "config-file-path": canonical.infer_config,
            }
        )


def test_hr0_plan_has_exact_candidate_commands_and_no_canonical_write() -> None:
    hr0 = get_mapanything_profile("hr0_378x672_b3_fp32")
    plan = tool._command_plan(
        hr0,
        mapanything_repo=REPO_ROOT / "src/mapanything",
        trtexec="/usr/bin/trtexec",
    )
    commands = plan["commands"]

    export = commands["export_onnx"]
    assert export[export.index("--output-name") + 1] == (
        "mapanything_images_378x672_b3.onnx"
    )
    assert export[export.index("--report-name") + 1] == (
        "mapanything_images_378x672_b3.export_report.txt"
    )
    assert "--fail-if-output-exists" in export
    assert export[export.index("--h") + 1] == "378"
    assert export[export.index("--w") + 1] == "672"
    assert export[export.index("--hf-model-id") + 1] == (
        tool.MAPANYTHING_HF_MODEL_ID
    )
    assert export[export.index("--hf-revision") + 1] == (
        tool.MAPANYTHING_HF_REVISION
    )
    assert export[export.index("--device") + 1] == "cuda"
    assert "--skip-eager-smoke" in export
    assert plan["model_source"] == {
        "mapanything_version": "1.1.3",
        "mapanything_commit": tool.MAPANYTHING_SOURCE_COMMIT,
        "uniception_min_version": "0.1.7",
        "huggingface_model_id": tool.MAPANYTHING_HF_MODEL_ID,
        "huggingface_revision": tool.MAPANYTHING_HF_REVISION,
        "strict_checkpoint_load": True,
        "export_device": "cuda",
    }

    build = commands["build_candidate"]
    assert "--minShapes=images:3x3x378x672" in build
    assert "--optShapes=images:3x3x378x672" in build
    assert "--maxShapes=images:3x3x378x672" in build
    assert "--fp16" not in build
    assert "--bf16" not in build
    assert any("/engines/candidates/" in item for item in build)
    assert not any(
        item.endswith("mapanything_images_294x518_b3_fp32.plan")
        for item in build
    )
    assert plan["canonical_runtime_modified"] is False
    assert plan["promotion_state"] == "isolated_candidate_only"


def test_hr0_source_inspection_pins_bundle_and_rejects_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = get_mapanything_profile("hr0_378x672_b3_fp32")
    onnx_path = tmp_path / "candidate.onnx"
    bundle = {
        "path": str(onnx_path),
        "sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "files": [{"label": "main", "path": str(onnx_path)}],
    }
    contract = {
        "opsets": [{"domain": "ai.onnx", "version": 17}],
        "inputs": [
            {
                "name": "images",
                "dtype": "FLOAT",
                "shape": ["batch", 3, 378, 672],
            }
        ],
        "outputs": [
            {
                "name": name,
                "dtype": "FLOAT",
                "shape": ["batch", 1, 378, 672],
            }
            for name in ("depth", "conf", "mask")
        ],
        "external_initializer_count": 632,
    }
    monkeypatch.setattr(tool, "input_bundle_record", lambda _path: bundle)
    monkeypatch.setattr(tool, "onnx_contract", lambda _path: contract)

    receipt = tool.inspect_source(profile, onnx_path)
    assert receipt["source"]["bundle_sha256"] == "2" * 64
    assert receipt["expected_runtime"]["output_bytes_per_frame"] == 3_048_192

    contract["inputs"][0]["shape"][-1] = 518
    with pytest.raises(MapAnythingProfileError, match="input differs"):
        tool.inspect_source(profile, onnx_path)
    contract["inputs"][0]["shape"][-1] = 672
    contract["outputs"][0]["shape"][-1] = "width"
    with pytest.raises(MapAnythingProfileError, match="output tensor contract"):
        tool.inspect_source(profile, onnx_path)


def test_prepare_hr0_fixture_is_full_resolution_derived_and_exact(tmp_path: Path) -> None:
    profile = get_mapanything_profile("hr0_378x672_b3_fp32")
    source = tmp_path / "full-res.png"
    # BGR input with unequal channels makes the RGB conversion observable.
    image = np.zeros((756, 1344, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    assert cv2.imwrite(str(source), image)

    output = tmp_path / "images-b3.raw"
    receipt_path = tmp_path / "images-b3.raw.receipt.json"
    receipt = tool.prepare_fixture(profile, [source], output, receipt_path)

    assert output.stat().st_size == profile.input_batch_bytes
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    batch = np.fromfile(output, dtype="<f4").reshape(
        profile.batch_size,
        3,
        profile.input_height,
        profile.input_width,
    )
    assert np.array_equal(batch[0], batch[1])
    assert np.array_equal(batch[1], batch[2])
    assert np.isclose(batch[0, 0, 0, 0], 30.0 / 255.0)
    assert np.isclose(batch[0, 1, 0, 0], 20.0 / 255.0)
    assert np.isclose(batch[0, 2, 0, 0], 10.0 / 255.0)
    assert receipt["identical_batch_members"] is True
    assert receipt["tensor"]["shape"] == [3, 3, 378, 672]
    assert json.loads(receipt_path.read_text()) == receipt

    with pytest.raises(FileExistsError):
        tool.prepare_fixture(profile, [source], output, receipt_path)


def test_prepare_hr0_fixture_rejects_upscaled_source(tmp_path: Path) -> None:
    profile = get_mapanything_profile("hr0_378x672_b3_fp32")
    source = tmp_path / "small.png"
    assert cv2.imwrite(str(source), np.zeros((189, 336, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="non-upscaled source"):
        tool.prepare_fixture(
            profile,
            [source],
            tmp_path / "small.raw",
            tmp_path / "small.receipt.json",
        )


def test_candidate_output_summary_is_strict_and_reports_batch_deltas(
    tmp_path: Path,
) -> None:
    profile = replace(
        get_mapanything_profile("hr0_378x672_b3_fp32"),
        input_height=2,
        input_width=2,
        patch_size=2,
    )
    output = tmp_path / "functional-output.json"
    payload = [
        {
            "name": name,
            "dimensions": "3x1x2x2",
            "values": values,
        }
        for name, values in (
            ("depth", [1.0, 2.0, 3.0, 4.0] * 3),
            ("conf", [1.0] * 12),
            ("mask", [1.0, 1.0, 1.0, 0.0] * 3),
        )
    ]
    output.write_text(json.dumps(payload), encoding="utf-8")

    summary = tool.summarize_output(profile, output)
    assert summary["outputs"]["depth"]["finite_fraction"] == 1.0
    assert summary["outputs"]["depth"]["batch_max_abs_delta"] == 0.0
    assert summary["outputs"]["mask"]["mask_coverage_at_0_5"] == 0.75
    assert summary["promotion_state"] == "measurement_only"

    output.write_text(output.read_text().replace("1.0", "NaN", 1))
    with pytest.raises(ValueError, match="strict JSON"):
        tool.summarize_output(profile, output)

    payload[0]["values"][0] = "1.0"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON numbers"):
        tool.summarize_output(profile, output)


def test_guarded_run_rejects_repo_local_artifact_root() -> None:
    with pytest.raises(tool.GuardedRunError, match="repo-local"):
        tool._require_external_artifact_root(DS9_ROOT)


def test_guarded_run_aggregates_capacity_by_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    free_bytes = 100 * tool.GIB
    monkeypatch.setattr(
        tool.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=free_bytes),
    )
    evidence = tool._capacity_evidence(
        artifact_root=tmp_path,
        cache_root=tmp_path / "cache",
        temp_root=tmp_path / "tmp",
    )
    assert len(evidence) == 1
    assert evidence[0]["growth_bytes"] == (
        tool.ARTIFACT_GROWTH_BYTES
        + tool.CACHE_GROWTH_BYTES
        + tool.TEMP_GROWTH_BYTES
    )
    assert evidence[0]["required_free_bytes"] == 56 * tool.GIB

    monkeypatch.setattr(
        tool.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=55 * tool.GIB),
    )
    with pytest.raises(tool.GuardedRunError, match="insufficient free space"):
        tool._capacity_evidence(
            artifact_root=tmp_path,
            cache_root=tmp_path / "cache",
            temp_root=tmp_path / "tmp",
        )


def test_guarded_run_rejects_existing_outputs_and_held_lock(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate.plan"
    output.write_bytes(b"occupied")
    with pytest.raises(tool.GuardedRunError, match="existing HR-0 output"):
        tool._require_outputs_absent([output])

    lock = tmp_path / ".other-engine-build.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    descriptor = lock.open("r+b")
    try:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert lock in tool._coordination_lock_paths(tmp_path)
        with pytest.raises(tool.GuardedRunError, match="owns"):
            tool._probe_coordination_locks(tmp_path)
    finally:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
        descriptor.close()


def test_guarded_run_materializes_only_staged_candidate_outputs(
    tmp_path: Path,
) -> None:
    profile = get_mapanything_profile("hr0_378x672_b3_fp32")
    artifact_root = tmp_path / "artifacts"
    evidence_root = (
        artifact_root
        / "models/engine_maintenance"
        / f"mapanything_{profile.name}_evidence"
    )
    plan = tool._command_plan(
        profile,
        mapanything_repo=REPO_ROOT / "src/mapanything",
        trtexec="/usr/local/bin/trtexec",
        artifact_root=artifact_root,
    )
    images = []
    for index in range(4):
        image = tmp_path / f"source-{index}.png"
        assert cv2.imwrite(
            str(image),
            np.zeros((378, 672, 3), dtype=np.uint8),
        )
        images.append(image)
    preflight = {
        "profile": profile.as_dict(),
        "artifact_root": str(artifact_root),
        "evidence_root": str(evidence_root),
        "executables": {
            "python": "/opt/mapanything-export/bin/python",
            "trtexec": "/usr/local/bin/trtexec",
            "polygraphy": "/usr/bin/polygraphy",
        },
        "plan": plan,
    }
    commands, paths = tool._materialize_guarded_commands(
        preflight,
        functional_image=images[0],
        scene_images=images[1:],
        run_token="test-run",
    )
    assert ".partial-test-run" in str(paths["staging_onnx_dir"])
    assert ".partial-test-run" in str(paths["staging_engine"])
    assert ".partial-test-run" in str(paths["staging_fixture"])
    assert not any(
        str(paths["final_engine"]) in item
        for item in commands["build_candidate"]
    )
    assert any(
        str(paths["staging_engine"]) in item
        for item in commands["build_candidate"]
    )
    assert commands["prepare_scene_fixture"].count("--image") == 3
    assert all(
        str(path.resolve()) in commands["prepare_scene_fixture"]
        for path in images[1:]
    )
    assert commands["export_onnx"][0] == "/opt/mapanything-export/bin/python"


def test_source_probe_records_pinned_isolated_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "mapanything"
    repo.mkdir()
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")

    def run(command, **_kwargs):
        if command[0] == "git" and "rev-parse" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=tool.MAPANYTHING_SOURCE_COMMIT + "\n",
            )
        if command[0] == "git" and "status" in command:
            return SimpleNamespace(returncode=0, stdout="")
        assert command[0] == str(python)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "mapanything_version": "1.1.3",
                    "mapanything_module": str(
                        repo / "mapanything/models/mapanything/model.py"
                    ),
                    "uniception_version": "0.1.7",
                    "python": str(python),
                    "python_prefix": str(python.parent.parent),
                }
            ),
        )

    monkeypatch.setattr(tool.subprocess, "run", run)
    result = tool._probe_mapanything_source_environment(
        mapanything_repo=repo,
        python_executable=python,
    )

    assert result["mapanything_commit"] == tool.MAPANYTHING_SOURCE_COMMIT
    assert result["uniception_version"] == "0.1.7"
    assert result["python"] == str(python)
    assert result["python_prefix"] == str(python.parent.parent)
    assert result["huggingface_revision"] == tool.MAPANYTHING_HF_REVISION
    assert result["checkpoint_load"] == "strict"


def test_source_probe_rejects_old_mapanything_before_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "mapanything"
    repo.mkdir()

    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="fde8425513178bb4f89fba9828193e6be3ece248\n",
        ),
    )
    with pytest.raises(tool.GuardedRunError, match="official source commit"):
        tool._probe_mapanything_source_environment(
            mapanything_repo=repo,
            python_executable=Path(sys.executable),
        )


def test_python_resolution_preserves_virtualenv_entrypoint(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "venv/bin/python"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(Path(sys.executable).resolve())

    assert tool._resolve_executable(
        str(entrypoint),
        "test Python",
        preserve_invocation_path=True,
    ) == entrypoint.absolute()
    assert tool._resolve_executable(str(entrypoint), "test Python") == (
        Path(sys.executable).resolve()
    )


def test_guarded_run_requires_exact_ds9_trtexec_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "&&&& RUNNING TensorRT.trtexec "
                f"[{tool.DS9_TRTEXEC_BANNER}] [b9]\n"
            ),
        ),
    )
    result = tool._probe_trtexec_toolchain(Path("/opt/ds9/trtexec"))
    assert result["expected_banner"] == tool.DS9_TRTEXEC_BANNER
    assert tool.DS9_TRTEXEC_BANNER in result["observed_banner_line"]
    assert len(result["probe_sha256"]) == 64

    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="TensorRT v101303\n",
        ),
    )
    with pytest.raises(tool.GuardedRunError, match="does not match"):
        tool._probe_trtexec_toolchain(Path("/opt/host/trtexec"))


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["/usr/bin/python3", "/repo/export_to_onnx.py", "--h", "378"],
            "mapanything_export",
        ),
        (
            [
                "/venv/bin/python",
                "-u",
                "/repo/mapanything_profile_tool.py",
                "guarded-run",
            ],
            "mapanything_guarded_run",
        ),
        (["/usr/src/tensorrt/bin/trtexec", "--loadEngine=x.plan"], "trtexec"),
        (
            [
                "/usr/bin/git",
                "diff",
                "--",
                "utils/onnx2trt/export_ma_onnx/export_to_onnx.py",
            ],
            "",
        ),
        (
            [
                "/usr/bin/python3",
                "-c",
                "print('export_to_onnx.py')",
            ],
            "",
        ),
    ],
)
def test_conflicting_process_reason_uses_invocation_identity(
    args: list[str],
    expected: str,
) -> None:
    assert tool._conflicting_process_reason(args) == expected


def test_guarded_run_cli_defaults_to_read_only() -> None:
    args = tool._parser().parse_args(
        [
            "guarded-run",
            "--artifact-root",
            "/external/artifacts",
            "--functional-image",
            "functional.png",
            "--scene-image",
            "one.png",
            "--scene-image",
            "two.png",
            "--scene-image",
            "three.png",
        ]
    )
    assert args.execute is False
    assert args.dry_run is False
    assert args.confirm_exclusive_gpu_window is False
    assert args.python == sys.executable
    assert args.export_device == "cuda"


def test_guarded_run_execute_requires_second_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_mapanything_profile("hr0_378x672_b3_fp32")
    monkeypatch.setattr(
        tool,
        "_guarded_preflight",
        lambda *_args, **_kwargs: {"profile": profile.as_dict()},
    )
    monkeypatch.setattr(
        tool,
        "_execute_guarded_run",
        lambda *_args, **_kwargs: pytest.fail(
            "execution must not begin without the confirmation flag"
        ),
    )
    with pytest.raises(
        tool.GuardedRunError,
        match="confirm-exclusive-gpu-window",
    ):
        tool.main(
            [
                "guarded-run",
                "--artifact-root",
                "/external/artifacts",
                "--functional-image",
                "functional.png",
                "--scene-image",
                "one.png",
                "--scene-image",
                "two.png",
                "--scene-image",
                "three.png",
                "--execute",
            ]
        )
