from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ds8_yolo26_seg_engine_maintenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ds8_yolo26_seg_engine_maintenance_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_onnx(path: Path) -> None:
    graph = helper.make_graph(
        [],
        "fixed_yolo26_seg_contract",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [3, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [3, 30, 4102])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    parser_dir = repo / "pipelines" / "nvdsinfer_yolo26_seg"
    parser_dir.mkdir(parents=True)
    (parser_dir / "libnvdsinfer_yolo26_seg.so").write_bytes(b"parser-binary")
    (parser_dir / "nvdsinfer_yolo26_seg.cpp").write_text("// parser source\n", encoding="utf-8")
    (repo / "pipelines" / "config_preproc.ini").write_text("[property]\ntensor-name=images\n", encoding="utf-8")
    (repo / "pipelines" / "config_infer_primary_yolo26_seg.template.ini").write_text(
        """[property]
onnx-file=@ONNX_PATH@
model-engine-file=@ENGINE_PATH@
labelfile-path=@LABELS_PATH@
custom-lib-path=@CUSTOM_LIB@
batch-size=3
network-type=3
network-mode=2
disable-output-host-copy=1
output-blob-names=output0
parse-bbox-instance-mask-func-name=NvDsInferParseYolo26Seg
topk=30
""",
        encoding="utf-8",
    )
    models = repo / "models"
    engines = models / "engines"
    engines.mkdir(parents=True)
    (models / "coco_labels.txt").write_text("person\n", encoding="utf-8")
    for size in "nsm":
        _write_onnx(models / f"yolo26{size}-seg_fused.onnx")
        (engines / f"yolo26{size}-seg_fused_b3_fp16.engine").write_bytes(
            f"prior-{size}".encode("ascii")
        )

    cuda_home = tmp_path / "cuda-13.0"
    cuda_home.mkdir()
    (cuda_home / "version.json").write_text(
        json.dumps(
            {
                "cuda": {"version": "13.0.2"},
                "cuda_cudart": {"version": "13.0.96"},
            }
        ),
        encoding="utf-8",
    )
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    trtexec.chmod(0o755)
    return repo, cuda_home, trtexec


def _plan(module, tmp_path: Path, *, sizes: tuple[str, ...] = ("s",)):
    repo, cuda_home, trtexec = _fixture_repo(tmp_path)
    evidence = tmp_path / "evidence"
    plan = module._collect_plan(
        repo_root=repo,
        sizes=sizes,
        evidence_root=evidence,
        run_id="test-run",
        trtexec=trtexec,
        cuda_home=cuda_home,
    )
    return plan, repo, cuda_home, trtexec, evidence


def test_plan_is_read_only_and_never_invokes_gpu_or_trtexec(tmp_path, monkeypatch, capsys) -> None:
    module = _load_module()
    repo, cuda_home, trtexec = _fixture_repo(tmp_path)
    evidence = tmp_path / "evidence"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan mode invoked a subprocess")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    result = module.main(
        [
            "--plan",
            "--sizes",
            "n,s,m",
            "--repo-root",
            str(repo),
            "--evidence-root",
            str(evidence),
            "--trtexec",
            str(trtexec),
            "--cuda-home",
            str(cuda_home),
        ]
    )

    assert result == 0
    assert not evidence.exists()
    assert not (
        repo / "models" / "engines" / ".noesis-yolo26-seg-engine-maintenance.lock"
    ).exists()
    output = capsys.readouterr().out
    assert "CPU/read-only; no subprocesses run" in output
    assert output.count("source_sha256=") == 3
    assert "--saveEngine=" in output
    assert "--loadEngine=" in output
    assert output.count("validate-installed:") == 3


def test_false_pass_trtexec_log_is_rejected_even_with_exit_zero() -> None:
    module = _load_module()
    false_pass = """
[I] [TRT] Loaded engine size: 23 MiB
[E] Error[6]: IRuntime::deserializeCudaEngine: incompatible plan
[E] Engine deserialization failed
[E] Error[4]: Failed to read header from the stream
[I] Skipped inference phase since --skipInference is added.
&&&& PASSED TensorRT.trtexec [TensorRT v101303] [b9]
"""
    result = module.CommandResult(0, false_pass, 0.1, 100)

    with pytest.raises(module.MaintenanceError, match="load validation failed"):
        module._validate_load_result(result)


def test_positive_deserialization_markers_are_required() -> None:
    module = _load_module()
    valid = """
[I] [TRT] Loaded engine size: 24 MiB
[I] Engine deserialized in 0.05 sec.
[I] Skipped inference phase since --skipInference is added.
&&&& PASSED TensorRT.trtexec [TensorRT v101303] [b9]
"""
    module._validate_load_result(module.CommandResult(0, valid, 0.1, 100))


def test_trtexec_version_banner_does_not_override_failed_version_probe() -> None:
    module = _load_module()
    deceptive = module.subprocess.CompletedProcess(
        ["trtexec", "--version"],
        1,
        "&&&& RUNNING TensorRT.trtexec [TensorRT v101303] [b9]\n"
        "[E] Model missing or format not recognized\n"
        "&&&& FAILED TensorRT.trtexec [TensorRT v101303] [b9]\n",
        "",
    )
    with pytest.raises(module.MaintenanceError, match="returncode=1"):
        module._validate_trtexec_profile_probe(deceptive)


def test_trtexec_help_banner_is_accepted_as_exact_profile_evidence() -> None:
    module = _load_module()
    help_probe = module.subprocess.CompletedProcess(
        ["trtexec", "--help"],
        0,
        "&&&& RUNNING TensorRT.trtexec [TensorRT v101303] [b9] # trtexec --help\n"
        "=== Model Options ===\n",
        "",
    )
    assert module._validate_trtexec_profile_probe(help_probe) == "TensorRT v101303 [b9]"


def test_failed_load_preserves_prior_and_never_installs_candidate(tmp_path, monkeypatch) -> None:
    module = _load_module()
    plan, repo, cuda_home, trtexec, evidence = _plan(module, tmp_path)
    target = repo / "models" / "engines" / "yolo26s-seg_fused_b3_fp16.engine"
    prior = target.read_bytes()

    monkeypatch.setattr(module, "MIN_ENGINE_BYTES", 1)
    monkeypatch.setattr(module, "_validate_real_platform", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(module, "_require_no_compute_owners", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_require_capacity", lambda _plan: None)
    def fake_run(command, **_kwargs):
        log_path = Path(_kwargs["log_path"])
        log_path.write_text("simulated\n", encoding="utf-8")
        save = next((part for part in command if str(part).startswith("--saveEngine=")), None)
        if save is not None:
            Path(str(save).split("=", 1)[1]).write_bytes(b"candidate")
            return module.CommandResult(0, "&&&& PASSED TensorRT.trtexec\n", 0.1, 100)
        return module.CommandResult(
            0,
            "[E] Error[6]: incompatible\n&&&& PASSED TensorRT.trtexec\n",
            0.1,
            100,
        )

    monkeypatch.setattr(module, "_run_bounded", fake_run)
    with pytest.raises(module.MaintenanceError, match="load validation failed"):
        module._execute_plan(plan, trtexec=trtexec, cuda_home=cuda_home)

    assert target.read_bytes() == prior
    assert (evidence / "test-run" / "prior" / target.name).read_bytes() == prior
    manifest = json.loads((evidence / "test-run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not Path(plan["engines"][0]["temporary"]).exists()


def test_validated_candidate_installs_atomically_after_prior_copy(tmp_path, monkeypatch) -> None:
    module = _load_module()
    plan, repo, cuda_home, trtexec, evidence = _plan(module, tmp_path)
    target = repo / "models" / "engines" / "yolo26s-seg_fused_b3_fp16.engine"
    prior = target.read_bytes()

    monkeypatch.setattr(module, "MIN_ENGINE_BYTES", 1)
    monkeypatch.setattr(module, "_validate_real_platform", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(module, "_require_no_compute_owners", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_require_capacity", lambda _plan: None)
    calls: list[Path] = []

    def fake_run(command, **_kwargs):
        log_path = Path(_kwargs["log_path"])
        log_path.write_text("simulated\n", encoding="utf-8")
        calls.append(log_path)
        save = next((part for part in command if str(part).startswith("--saveEngine=")), None)
        if save is not None:
            Path(str(save).split("=", 1)[1]).write_bytes(b"validated-candidate")
            return module.CommandResult(0, "&&&& PASSED TensorRT.trtexec\n", 0.1, 123)
        return module.CommandResult(
            0,
            "[I] [TRT] Loaded engine size: 1 MiB\n"
            "[I] Engine deserialized in 0.01 sec.\n"
            "[I] Skipped inference phase since --skipInference is added.\n"
            "&&&& PASSED TensorRT.trtexec\n",
            0.1,
            124,
        )

    monkeypatch.setattr(module, "_run_bounded", fake_run)
    manifest_path = module._execute_plan(plan, trtexec=trtexec, cuda_home=cuda_home)

    assert target.read_bytes() == b"validated-candidate"
    assert (evidence / "test-run" / "prior" / target.name).read_bytes() == prior
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["engines"][0]["result"]["max_gpu_memory_mib"] == 124
    assert (evidence / "test-run" / "logs" / "load-installed-s.log").is_file()
    assert "installed_load_duration_seconds" in manifest["engines"][0]["result"]
    assert [path.name for path in calls] == ["build-s.log", "load-s.log", "load-installed-s.log"]


def test_final_installed_path_must_also_deserialize(tmp_path, monkeypatch) -> None:
    module = _load_module()
    plan, repo, cuda_home, trtexec, evidence = _plan(module, tmp_path)
    target = repo / "models" / "engines" / "yolo26s-seg_fused_b3_fp16.engine"

    monkeypatch.setattr(module, "MIN_ENGINE_BYTES", 1)
    monkeypatch.setattr(module, "_validate_real_platform", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(module, "_require_no_compute_owners", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_require_capacity", lambda _plan: None)
    load_count = 0

    def fake_run(command, **_kwargs):
        nonlocal load_count
        Path(_kwargs["log_path"]).write_text("simulated\n", encoding="utf-8")
        save = next((part for part in command if str(part).startswith("--saveEngine=")), None)
        if save is not None:
            Path(str(save).split("=", 1)[1]).write_bytes(b"candidate")
            return module.CommandResult(0, "&&&& PASSED TensorRT.trtexec\n", 0.1, 100)
        load_count += 1
        if load_count == 1:
            return module.CommandResult(
                0,
                "[I] [TRT] Loaded engine size: 1 MiB\n"
                "[I] Engine deserialized in 0.01 sec.\n"
                "[I] Skipped inference phase since --skipInference is added.\n",
                0.1,
                100,
            )
        return module.CommandResult(
            0,
            "[E] Error[6]: installed-path failure\n&&&& PASSED TensorRT.trtexec\n",
            0.1,
            100,
        )

    monkeypatch.setattr(module, "_run_bounded", fake_run)
    with pytest.raises(module.MaintenanceError, match="load validation failed"):
        module._execute_plan(plan, trtexec=trtexec, cuda_home=cuda_home)

    manifest = json.loads((evidence / "test-run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert load_count == 2
    assert target.read_bytes() == b"candidate"


def test_real_build_refuses_any_preexisting_compute_owner(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_query_compute_owners",
        lambda: [{"pid": 42, "process_name": "other-runtime", "used_memory_mib": 1000}],
    )
    with pytest.raises(module.MaintenanceError, match="compute owner"):
        module._require_no_compute_owners()


def test_real_maintenance_lock_is_nonblocking_under_contention(tmp_path) -> None:
    module = _load_module()
    lock_path = tmp_path / "engines" / ".maintenance.lock"
    with module._maintenance_lock(lock_path):
        with pytest.raises(module.MaintenanceError, match="another YOLO26-seg"):
            with module._maintenance_lock(lock_path):
                raise AssertionError("contended lock was acquired")
