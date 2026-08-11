from __future__ import annotations

import configparser
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from DS9.noesis import depth_tracking_materialization as ds9_depth
from DS9.noesis import v3dt_assets as ds9_v3dt
from DS9.noesis.pipelines import ds8_pipeline as ds9_pipeline
from noesis import depth_tracking_materialization as ds8_depth
from noesis import deimv2_wholebody49_assets as wholebody49_assets
from noesis import ds8_runtime, ds8_runtime_v3dt_reimpl
from noesis import reid_swin_profile
from noesis import yolo26_seg_materialization as ds8_yolo26_materialization
from noesis.pipelines import ds8_pipeline as ds8_pipeline
from noesis_core import inference_runtime_contract as contract


ROOT = Path(__file__).resolve().parents[1]
DS9_RUNTIME_CORE = ROOT / "DS9" / "noesis" / "ds9_runtime_core.py"


NVINFER_BUILD_KEYS = {
    "custom-network-config",
    "engine-create-func-name",
    "int8-calib-file",
    "model-file",
    "onnx-file",
    "proto-file",
    "tlt-encoded-model",
    "tlt-model-key",
    "uff-file",
    "uff-input-blob-name",
    "uff-input-dims",
    "uff-input-order",
}
NVTRACKER_BUILD_KEYS = {
    "calibrationTableFile",
    "onnxFile",
    "tltEncodedModel",
    "tltModelKey",
    "uffFile",
}


def test_ds9_native_materializer_defaults_to_owned_extension_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOESIS_NATIVE_EXT_DIR", raising=False)
    assert ds9_depth._native_ext_dir() == (
        ds9_depth.REPO_ROOT / "native_extensions"
    ).resolve()


def _read_properties(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(path, encoding="utf-8")
    return parser["property"]


def test_nvinfer_runtime_config_is_derived_engine_only_and_source_is_immutable(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "selected.engine"
    engine.write_bytes(b"engine")
    labels = tmp_path / "labels.txt"
    labels.write_text("person\n", encoding="utf-8")
    source = tmp_path / "source.ini"
    original = (
        "[property]\n"
        "onnx-file=maintenance.onnx\n"
        "engine-create-func-name=OfflineBuilder\n"
        "model-engine-file=stale.engine\n"
        "labelfile-path=labels.txt\n"
        "batch-size=3\n"
    )
    source.write_text(original, encoding="utf-8")

    output = contract.materialize_nvinfer_engine_only_config(
        source_config=source,
        engine_path=engine,
        output_root=tmp_path / "build",
        component_name="primary detector",
        repo_root=tmp_path,
    )

    assert source.read_text(encoding="utf-8") == original
    assert output != source
    assert output.is_relative_to(tmp_path / "build" / "runtime_inference" / "nvinfer")
    properties = _read_properties(output)
    assert not (set(properties) & NVINFER_BUILD_KEYS)
    assert Path(properties["model-engine-file"]) == engine.resolve()
    assert Path(properties["labelfile-path"]) == labels.resolve()


@pytest.mark.parametrize("empty", [False, True])
def test_nvinfer_runtime_config_rejects_missing_or_empty_engine(
    tmp_path: Path,
    empty: bool,
) -> None:
    source = tmp_path / "source.ini"
    source.write_text("[property]\nonnx-file=model.onnx\n", encoding="utf-8")
    engine = tmp_path / "missing.engine"
    if empty:
        engine.touch()

    with pytest.raises(contract.EngineOnlyRuntimeError, match="missing or empty"):
        contract.materialize_nvinfer_engine_only_config(
            source_config=source,
            engine_path=engine,
            output_root=tmp_path / "build",
            component_name="pgie",
            repo_root=tmp_path,
        )
    assert not (tmp_path / "build").exists()


def test_nvinfer_runtime_config_rejects_custom_engine_builder_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "selected.engine"
    engine.write_bytes(b"engine")
    library = tmp_path / "parser.so"
    library.write_bytes(b"elf")
    source = tmp_path / "source.ini"
    source.write_text(
        "[property]\ncustom-lib-path=parser.so\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(contract.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        contract.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="00000000 T NvDsInferCudaEngineGet\n",
            stderr="",
        ),
    )

    with pytest.raises(contract.EngineOnlyRuntimeError, match="engine-building symbol"):
        contract.materialize_nvinfer_engine_only_config(
            source_config=source,
            engine_path=engine,
            output_root=tmp_path / "build",
            component_name="pgie",
            repo_root=tmp_path,
        )


def test_nvtracker_runtime_config_strips_model_sources_and_resolves_assets(
    tmp_path: Path,
) -> None:
    reid_engine = tmp_path / "reid.engine"
    pose_engine = tmp_path / "pose.engine"
    camera = tmp_path / "camera.yml"
    for path in (reid_engine, pose_engine, camera):
        path.write_bytes(b"asset")
    source = tmp_path / "tracker.yml"
    source_payload = {
        "ReID": {
            "tltEncodedModel": "reid.etlt",
            "tltModelKey": "secret",
            "modelEngineFile": str(reid_engine),
        },
        "PoseEstimator": {
            "onnxFile": "pose.onnx",
            "modelEngineFile": str(pose_engine),
        },
        "ObjectModelProjection": {"cameraModelFilepath": [str(camera)]},
    }
    source.write_text(yaml.safe_dump(source_payload), encoding="utf-8")

    output = contract.materialize_nvtracker_engine_only_config(
        source_config=source,
        output_root=tmp_path / "build",
        component_name="tracker",
        repo_root=tmp_path,
    )

    assert yaml.safe_load(source.read_text(encoding="utf-8")) == source_payload
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    rendered = output.read_text(encoding="utf-8")
    assert not any(f"{key}:" in rendered for key in NVTRACKER_BUILD_KEYS)
    assert Path(payload["ReID"]["modelEngineFile"]) == reid_engine.resolve()
    assert Path(payload["PoseEstimator"]["modelEngineFile"]) == pose_engine.resolve()
    assert Path(payload["ObjectModelProjection"]["cameraModelFilepath"][0]) == camera.resolve()


def test_nvtracker_runtime_config_rejects_source_without_engine(tmp_path: Path) -> None:
    source = tmp_path / "tracker.yml"
    source.write_text("ReID:\n  tltEncodedModel: model.etlt\n", encoding="utf-8")

    with pytest.raises(contract.EngineOnlyRuntimeError, match="without modelEngineFile"):
        contract.materialize_nvtracker_engine_only_config(
            source_config=source,
            output_root=tmp_path / "build",
            component_name="tracker",
            repo_root=tmp_path,
        )


def _minimal_pipeline_config(
    tmp_path: Path,
    *,
    force_engine_rebuild: bool | None = None,
) -> Path:
    engine = tmp_path / "pgie.engine"
    engine.write_bytes(b"engine")
    source = tmp_path / "pgie.ini"
    source.write_text(
        "[property]\n"
        "onnx-file=offline.onnx\n"
        f"model-engine-file={engine}\n",
        encoding="utf-8",
    )
    tracker_engine = tmp_path / "tracker.engine"
    tracker_engine.write_bytes(b"tracker")
    tracker_source = tmp_path / "tracker.yml"
    tracker_source.write_text(
        yaml.safe_dump(
            {
                "ReID": {
                    "tltEncodedModel": "offline.etlt",
                    "modelEngineFile": str(tracker_engine),
                }
            }
        ),
        encoding="utf-8",
    )
    pgie = {
        "config-file-path": str(source),
        "engine": str(engine),
    }
    if force_engine_rebuild is not None:
        pgie["force_engine_rebuild"] = force_engine_rebuild
    payload = {
        "version": 1,
        "batch_size": 1,
        "sources": [{"element": "nvurisrcbin", "uri": "file:///tmp/input.mp4"}],
        "streammux": {"element": "nvstreammux", "batch-size": 1},
        "models": {"pgie": pgie},
        "tracker": {"config-file": str(tracker_source)},
        "analytics": {"enable": False},
        "sinks": [{"name": "sink", "type": "fakesink", "sync": False}],
    }
    config = tmp_path / "infer.yaml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config


@pytest.mark.parametrize("module", [ds8_pipeline, ds9_pipeline])
def test_ds8_and_ds9_graphs_receive_only_derived_engine_configs(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_DS8_STUB_PIPELINE", "1")
    monkeypatch.setenv("NOESIS_BUILD_DIR", str(tmp_path / "build"))
    module._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]

    graph = module.build_pipeline(_minimal_pipeline_config(tmp_path))  # type: ignore[attr-defined]

    pgie = graph.components["yolo11_pgie"]
    properties = _read_properties(Path(pgie.config["config-file-path"]))
    assert not (set(properties) & NVINFER_BUILD_KEYS)
    assert Path(properties["model-engine-file"]) == Path(
        pgie.config["model-engine-file"]
    )
    tracker_path = Path(graph.components["tracker"].config["ll-config-file"])
    tracker_text = tracker_path.read_text(encoding="utf-8")
    assert not any(f"{key}:" in tracker_text for key in NVTRACKER_BUILD_KEYS)
    module._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]


@pytest.mark.parametrize("module", [ds8_pipeline, ds9_pipeline])
def test_pipeline_sanitizes_direct_ll_tracker_config(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_DS8_STUB_PIPELINE", "1")
    monkeypatch.setenv("NOESIS_BUILD_DIR", str(tmp_path / "build"))
    config_path = _minimal_pipeline_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["tracker"]["ll-config-file"] = payload["tracker"].pop("config-file")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    module._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]

    graph = module.build_pipeline(config_path)  # type: ignore[attr-defined]

    tracker_path = Path(graph.components["tracker"].config["ll-config-file"])
    assert tracker_path.is_relative_to(tmp_path / "build" / "runtime_inference")
    assert "tltEncodedModel:" not in tracker_path.read_text(encoding="utf-8")
    module._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]


@pytest.mark.parametrize("module", [ds8_pipeline, ds9_pipeline])
@pytest.mark.parametrize("value", [False, True])
def test_pipeline_rejects_force_engine_rebuild_even_when_false(
    module: object,
    value: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_DS8_STUB_PIPELINE", "1")
    module._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]

    with pytest.raises(contract.EngineOnlyRuntimeError, match="force_engine_rebuild"):
        module.build_pipeline(  # type: ignore[attr-defined]
            _minimal_pipeline_config(tmp_path, force_engine_rebuild=value)
        )


@pytest.mark.parametrize("module", [ds8_pipeline, ds9_pipeline])
@pytest.mark.parametrize("pgie", [None, {}, {"config-file-path": "missing.ini"}])
def test_pipeline_rejects_missing_or_incomplete_active_pgie(
    module: object,
    pgie: dict[str, str] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_DS8_STUB_PIPELINE", "1")
    payload = {
        "version": 1,
        "models": {} if pgie is None else {"pgie": pgie},
        "tracker": {},
    }
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    module._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]

    with pytest.raises(contract.EngineOnlyRuntimeError, match="pgie|both"):
        module.build_pipeline(config)  # type: ignore[attr-defined]


def _configure_depth_module(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_engine: bool,
) -> tuple[Path, Path]:
    template = tmp_path / "depth.template.ini"
    template.write_text(
        "[property]\n"
        "onnx-file=@ONNX_PATH@\n"
        "model-engine-file=@ENGINE_PATH@\n"
        "batch-size=@BATCH_SIZE@\n"
        "interval=@INTERVAL@\n"
        "gie-unique-id=@GIE_ID@\n"
        "infer-dims=3;@HEIGHT@;@WIDTH@\n",
        encoding="utf-8",
    )
    engine = tmp_path / "depth.engine"
    if create_engine:
        engine.write_bytes(b"engine")
    monkeypatch.setattr(module, "_CONFIG_TEMPLATE", template)
    monkeypatch.setattr(module, "_B3_ENGINE_TEMPLATE", engine)
    monkeypatch.setattr(module, "_B3_ONNX_TEMPLATE", tmp_path / "offline.onnx")
    monkeypatch.setattr(module, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(
        module,
        "_export_batch_onnx",
        lambda **_kwargs: pytest.fail("runtime attempted ONNX export"),
    )
    monkeypatch.setattr(
        module,
        "_build_engine",
        lambda **_kwargs: pytest.fail("runtime attempted TensorRT build"),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("runtime attempted subprocess"),
    )
    return template, engine


@pytest.mark.parametrize("module", [ds8_depth, ds9_depth])
def test_depth_runtime_materialization_is_engine_only_and_never_builds(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template, engine = _configure_depth_module(
        module, tmp_path, monkeypatch, create_engine=True
    )
    original = template.read_text(encoding="utf-8")

    assets = module.materialize_depth_tracking_assets()  # type: ignore[attr-defined]

    assert template.read_text(encoding="utf-8") == original
    assert assets.engine_path == engine
    properties = _read_properties(assets.config_path)
    assert "onnx-file" not in properties
    assert Path(properties["model-engine-file"]) == engine.resolve()


@pytest.mark.parametrize("module", [ds8_depth, ds9_depth])
def test_depth_runtime_materialization_fails_when_engine_is_missing_without_building(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_depth_module(module, tmp_path, monkeypatch, create_engine=False)

    with pytest.raises(FileNotFoundError, match="TensorRT engine"):
        module.materialize_depth_tracking_assets()  # type: ignore[attr-defined]
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    ("ensure_name", "source_name", "extension_name"),
    [
        (
            "ensure_native_object_depth_extension",
            "noesis_depth_meta_ext.cpp",
            "noesis_depth_meta_ext",
        ),
        (
            "ensure_native_depth_tracking_tensor_extension",
            "noesis_depth_tracking_tensor_ext.cpp",
            "noesis_depth_tracking_tensor_ext",
        ),
    ],
)
@pytest.mark.parametrize(
    ("module", "state"),
    [
        (ds8_depth, "missing"),
        (ds8_depth, "stale"),
        (ds9_depth, "missing"),
    ],
)
def test_native_runtime_extension_checks_never_compile(
    module: object,
    ensure_name: str,
    source_name: str,
    extension_name: str,
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("NOESIS_NATIVE_EXT_DIR", raising=False)
    monkeypatch.delenv("NOESIS_NATIVE_BUILD_SCRIPT_DIR", raising=False)
    source = tmp_path / "native" / source_name
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    extension = tmp_path / f"{extension_name}{suffix}"
    if state == "stale":
        extension.write_bytes(b"extension")
        os.utime(extension, (1, 1))
        os.utime(source, (2, 2))
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("runtime attempted native compilation"),
    )

    with pytest.raises(RuntimeError, match=f"{state}|missing or empty") as error:
        getattr(module, ensure_name)()
    assert "Run " in str(error.value)


def test_ds9_v3dt_runtime_materializer_emits_engine_only_tracker_yaml(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "build"
    output_root.mkdir()
    reid_engine = output_root / "reid.engine"
    pose_engine = output_root / "pose.engine"
    reid_engine.write_bytes(b"reid")
    pose_engine.write_bytes(b"pose")
    tracker_source = tmp_path / "tracker.yml"
    tracker_source.write_text(
        yaml.safe_dump(
            {
                "ObjectModelProjection": {"cameraModelFilepath": []},
                "ReID": {
                    "tltEncodedModel": "source.etlt",
                    "tltModelKey": "nvidia_tao",
                    "modelEngineFile": "old.engine",
                },
                "PoseEstimator": {
                    "onnxFile": "source.onnx",
                    "modelEngineFile": "old.engine",
                },
            }
        ),
        encoding="utf-8",
    )
    camera = tmp_path / "camera.yml"
    camera.write_bytes(b"camera")
    bundle = SimpleNamespace(
        tracker_config=tracker_source,
        camera_models=(camera,),
        tracker_reid_source=tmp_path / "source.etlt",
        tracker_reid_engine=reid_engine,
        bodypose_source=tmp_path / "source.onnx",
        bodypose_engine=pose_engine,
    )

    output = ds9_v3dt.materialize_v3dt_tracker_config(
        bundle,
        output_root / "tracker.yml",
        output_root=output_root,
    )

    rendered = output.read_text(encoding="utf-8")
    payload = yaml.safe_load(rendered)
    assert not any(f"{key}:" in rendered for key in NVTRACKER_BUILD_KEYS)
    assert Path(payload["ReID"]["modelEngineFile"]) == reid_engine.resolve()
    assert Path(payload["PoseEstimator"]["modelEngineFile"]) == pose_engine.resolve()


def test_ds9_v3dt_offline_materializer_retains_only_tracker_build_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "build"
    output_root.mkdir()
    reid_source = output_root / "source.etlt"
    pose_engine = output_root / "pose.engine"
    reid_source.write_bytes(b"locked-etlt")
    monkeypatch.setattr(
        ds9_v3dt,
        "TRACKER_REID_SOURCE_SHA256",
        hashlib.sha256(reid_source.read_bytes()).hexdigest(),
    )
    pose_engine.write_bytes(b"pose")
    tracker_source = tmp_path / "tracker.yml"
    tracker_source.write_text(
        yaml.safe_dump(
            {
                "ObjectModelProjection": {"cameraModelFilepath": []},
                "ReID": {
                    "batchSize": 32,
                    "networkMode": 1,
                    "onnxFile": "forbidden.onnx",
                    "tltEncodedModel": "source.etlt",
                    "tltModelKey": "nvidia_tao",
                    "modelEngineFile": "old.engine",
                },
                "PoseEstimator": {
                    "onnxFile": "source.onnx",
                    "modelEngineFile": "old.engine",
                },
            }
        ),
        encoding="utf-8",
    )
    camera = tmp_path / "camera.yml"
    camera.write_bytes(b"camera")
    installed = output_root / "installed.engine"
    generated = ds9_v3dt.derive_nvmot_tracker_engine_path(
        reid_source, batch_size=32, gpu_id=0, network_mode=1
    )
    bundle = SimpleNamespace(
        tracker_config=tracker_source,
        camera_models=(camera,),
        tracker_reid_source=reid_source,
        tracker_reid_engine=installed,
        bodypose_source=tmp_path / "source.onnx",
        bodypose_engine=pose_engine,
    )

    output = ds9_v3dt.materialize_v3dt_tracker_build_config(
        bundle,
        output_root / "tracker-build.yml",
        output_root=output_root,
        tracker_reid_source=reid_source,
        tracker_reid_generated_engine=generated,
        gpu_id=0,
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert not generated.exists()
    assert Path(payload["ReID"]["modelEngineFile"]) == generated.resolve()
    assert Path(payload["ReID"]["tltEncodedModel"]) == reid_source.resolve()
    assert payload["ReID"]["tltModelKey"] == "nvidia_tao"
    assert "onnxFile" not in payload["ReID"]
    assert Path(payload["PoseEstimator"]["modelEngineFile"]) == pose_engine.resolve()
    assert not (set(payload["PoseEstimator"]) & NVTRACKER_BUILD_KEYS)


def test_ds9_v3dt_offline_materializer_rejects_unsafe_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "build"
    output_root.mkdir()
    reid_source = output_root / "source.etlt"
    pose_engine = output_root / "pose.engine"
    reid_source.write_bytes(b"locked-etlt")
    monkeypatch.setattr(
        ds9_v3dt,
        "TRACKER_REID_SOURCE_SHA256",
        hashlib.sha256(reid_source.read_bytes()).hexdigest(),
    )
    pose_engine.write_bytes(b"pose")
    tracker_source = tmp_path / "tracker.yml"
    tracker_source.write_text(
        yaml.safe_dump(
            {
                "ObjectModelProjection": {"cameraModelFilepath": []},
                "ReID": {"batchSize": 32, "networkMode": 1},
                "PoseEstimator": {},
            }
        ),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        tracker_config=tracker_source,
        camera_models=(),
        tracker_reid_source=reid_source,
        tracker_reid_engine=output_root / "installed.engine",
        bodypose_engine=pose_engine,
    )

    with pytest.raises(ds9_v3dt.V3DTAssetError, match="exact derived path"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "wrong-derived.yml",
            output_root=output_root,
            tracker_reid_source=reid_source,
            tracker_reid_generated_engine=output_root / "wrong.engine",
            gpu_id=0,
        )

    generated = ds9_v3dt.derive_nvmot_tracker_engine_path(
        reid_source, batch_size=32, gpu_id=0, network_mode=1
    )
    existing = generated
    existing.write_bytes(b"do-not-replace")
    with pytest.raises(ds9_v3dt.V3DTAssetError, match="must be absent"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "existing.yml",
            output_root=output_root,
            tracker_reid_source=reid_source,
            tracker_reid_generated_engine=existing,
            gpu_id=0,
        )

    existing.unlink()
    linked_target = output_root / "linked-target.engine"
    linked_target.write_bytes(b"do-not-follow")
    generated.symlink_to(linked_target)
    with pytest.raises(ds9_v3dt.V3DTAssetError, match="must not be a symlink"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "linked.yml",
            output_root=output_root,
            tracker_reid_source=reid_source,
            tracker_reid_generated_engine=generated,
            gpu_id=0,
        )
    generated.unlink()

    outside_source = tmp_path / "outside" / reid_source.name
    outside_source.parent.mkdir()
    outside_source.write_bytes(reid_source.read_bytes())
    outside_generated = ds9_v3dt.derive_nvmot_tracker_engine_path(
        outside_source, batch_size=32, gpu_id=0, network_mode=1
    )
    with pytest.raises(ds9_v3dt.V3DTAssetError, match="transaction output root"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "outside.yml",
            output_root=output_root,
            tracker_reid_source=outside_source,
            tracker_reid_generated_engine=outside_generated,
            gpu_id=0,
        )

    renamed_source = output_root / "renamed.etlt"
    renamed_source.write_bytes(reid_source.read_bytes())
    with pytest.raises(ds9_v3dt.V3DTAssetError, match="locked filename"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "renamed-source.yml",
            output_root=output_root,
            tracker_reid_source=renamed_source,
            tracker_reid_generated_engine=ds9_v3dt.derive_nvmot_tracker_engine_path(
                renamed_source, batch_size=32, gpu_id=0, network_mode=1
            ),
            gpu_id=0,
        )

    hardlink_parent = output_root / "hardlink-source"
    hardlink_parent.mkdir()
    hardlinked_source = hardlink_parent / reid_source.name
    os.link(reid_source, hardlinked_source)
    with pytest.raises(ds9_v3dt.V3DTAssetError, match="single-link"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "hardlinked-source.yml",
            output_root=output_root,
            tracker_reid_source=hardlinked_source,
            tracker_reid_generated_engine=ds9_v3dt.derive_nvmot_tracker_engine_path(
                hardlinked_source, batch_size=32, gpu_id=0, network_mode=1
            ),
            gpu_id=0,
        )
    hardlinked_source.unlink()

    linked_source = output_root / "linked-source.etlt"
    linked_source.symlink_to(reid_source)
    linked_source_bundle = SimpleNamespace(
        **{**vars(bundle), "tracker_reid_source": linked_source}
    )
    with pytest.raises(ds9_v3dt.V3DTAssetError, match="must not be a symlink"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            linked_source_bundle,
            output_root / "linked-source.yml",
            output_root=output_root,
            tracker_reid_source=linked_source,
            tracker_reid_generated_engine=ds9_v3dt.derive_nvmot_tracker_engine_path(
                linked_source, batch_size=32, gpu_id=0, network_mode=1
            ),
            gpu_id=0,
        )

    real_input_parent = output_root / "real-input-parent"
    real_input_parent.mkdir()
    ancestor_source = real_input_parent / "source.etlt"
    ancestor_source.write_bytes(reid_source.read_bytes())
    ancestor_pose = real_input_parent / "pose.engine"
    ancestor_pose.write_bytes(b"pose")
    linked_input_parent = output_root / "linked-input-parent"
    linked_input_parent.symlink_to(real_input_parent, target_is_directory=True)
    linked_source_parent_bundle = SimpleNamespace(
        **{
            **vars(bundle),
            "tracker_reid_source": linked_input_parent / "source.etlt",
        }
    )
    with pytest.raises(
        ds9_v3dt.V3DTAssetError, match="tracker ReID source parent contains a symlink"
    ):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            linked_source_parent_bundle,
            output_root / "linked-source-parent.yml",
            output_root=output_root,
            tracker_reid_source=linked_input_parent / "source.etlt",
            tracker_reid_generated_engine=ds9_v3dt.derive_nvmot_tracker_engine_path(
                linked_input_parent / "source.etlt",
                batch_size=32,
                gpu_id=0,
                network_mode=1,
            ),
            gpu_id=0,
        )

    linked_pose_parent_bundle = SimpleNamespace(
        **{
            **vars(bundle),
            "bodypose_engine": linked_input_parent / "pose.engine",
        }
    )
    with pytest.raises(
        ds9_v3dt.V3DTAssetError,
        match="generated BodyPose engine parent contains a symlink",
    ):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            linked_pose_parent_bundle,
            output_root / "linked-pose-parent.yml",
            output_root=output_root,
            tracker_reid_source=reid_source,
            tracker_reid_generated_engine=generated,
            gpu_id=0,
        )


def test_ds9_v3dt_offline_materializer_rejects_tampered_tracker_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "build"
    output_root.mkdir()
    reid_source = output_root / "source.etlt"
    reid_source.write_bytes(b"locked-etlt")
    monkeypatch.setattr(
        ds9_v3dt,
        "TRACKER_REID_SOURCE_SHA256",
        hashlib.sha256(reid_source.read_bytes()).hexdigest(),
    )
    pose_engine = output_root / "pose.engine"
    pose_engine.write_bytes(b"pose")
    tracker_source = tmp_path / "tracker.yml"
    tracker_source.write_text(
        yaml.safe_dump(
            {
                "ObjectModelProjection": {"cameraModelFilepath": []},
                "ReID": {"batchSize": 32, "networkMode": 1},
                "PoseEstimator": {},
            }
        ),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        tracker_config=tracker_source,
        camera_models=(),
        tracker_reid_source=reid_source,
        tracker_reid_engine=output_root / "installed.engine",
        bodypose_engine=pose_engine,
    )
    reid_source.write_bytes(b"tampered-etlt")

    with pytest.raises(ds9_v3dt.V3DTAssetError, match="provenance lock"):
        ds9_v3dt.materialize_v3dt_tracker_build_config(
            bundle,
            output_root / "tampered-source.yml",
            output_root=output_root,
            tracker_reid_source=reid_source,
            tracker_reid_generated_engine=ds9_v3dt.derive_nvmot_tracker_engine_path(
                reid_source, batch_size=32, gpu_id=0, network_mode=1
            ),
            gpu_id=0,
        )


def test_ds9_v3dt_tracker_builder_uses_only_offline_materializer() -> None:
    source = (ROOT / "DS9/scripts/build_v3dt_tracker_engine.py").read_text(
        encoding="utf-8"
    )
    assert "materialize_v3dt_tracker_build_config(" in source
    assert "materialize_v3dt_tracker_config(" not in source


@pytest.mark.parametrize(
    ("module", "profile"),
    [
        (ds8_runtime, "yolo11"),
        (ds8_runtime_v3dt_reimpl, "rfdetr_seg"),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_runtime_profile_preflight_rejects_missing_or_empty_engine_before_sources(
    module: object,
    profile: str,
    empty: bool,
    tmp_path: Path,
) -> None:
    engine = tmp_path / "missing.engine"
    if empty:
        engine.touch()
    config = {"models": {"pgie": {"engine": str(engine)}}}
    yaml_path = tmp_path / "infer.yaml"
    yaml_path.write_text("version: 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="engine missing or empty"):
        module._preflight_pgie_profile(  # type: ignore[attr-defined]
            profile,
            config,
            yaml_path,
            SimpleNamespace(info=lambda *_args, **_kwargs: None),
        )


def test_runtime_cores_contain_no_nvinfer_source_rebuild_allowance() -> None:
    for path in (
        Path(ds8_runtime.__file__),
        Path(ds8_runtime_v3dt_reimpl.__file__),
        DS9_RUNTIME_CORE,
    ):
        source = path.read_text(encoding="utf-8").lower()
        assert "nvinfer will attempt to build" not in source
        assert "available to rebuild it" not in source
        assert "validate_reid_swin_onnx_source" not in source
        assert "onnx_path.read_bytes" not in source


def test_ds8_yolo11_seg_preflight_uses_engine_contract_without_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "yolo11s-seg_cust_fused.engine"
    engine.write_bytes(b"engine")
    parser_library = tmp_path / "parser.so"
    parser_library.write_bytes(b"parser")
    labels = tmp_path / "labels.txt"
    labels.write_text("person\n", encoding="utf-8")
    plugin_library = tmp_path / "plugins.so"
    plugin_library.write_bytes(b"plugins")
    preprocess = tmp_path / "preprocess.ini"
    preprocess.write_text("[property]\ntensor-name=images\n", encoding="utf-8")
    pgie = tmp_path / "pgie.ini"
    pgie.write_text(
        "[property]\n"
        f"custom-lib-path={parser_library}\n"
        f"labelfile-path={labels}\n"
        "gie-unique-id=1\n"
        "batch-size=3\n"
        "network-type=3\n"
        "parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg\n"
        "output-instance-mask=1\n",
        encoding="utf-8",
    )
    pipeline_config = {
        "preprocess": {"config-file": str(preprocess)},
        "models": {
            "pgie": {
                "config-file-path": str(pgie),
                "engine": str(engine),
            }
        },
    }
    yaml_path = tmp_path / "infer.yaml"
    yaml_path.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setenv("NOESIS_YOLO11_SEG_TRT_PLUGIN_LIB", str(plugin_library))
    monkeypatch.setattr(ds8_runtime, "_YOLO11_SEG_TRT_PLUGIN_LOADED", False)
    loaded: list[Path] = []
    monkeypatch.setattr(
        ds8_runtime.ctypes,
        "CDLL",
        lambda path, **_kwargs: loaded.append(Path(path)) or object(),
    )
    logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        debug=lambda *_args, **_kwargs: None,
    )

    ds8_runtime._preflight_pgie_profile(
        "yolo11_seg",
        pipeline_config,
        yaml_path,
        logger,
    )

    assert loaded == [plugin_library.resolve()]
    assert not (tmp_path / "model.onnx").exists()


def test_ds9_reid_preflight_accepts_valid_engine_when_onnx_source_is_absent(
    tmp_path: Path,
) -> None:
    engine = tmp_path / reid_swin_profile.REID_SWIN_ENGINE_NAME
    engine.write_bytes(b"engine")
    config_path = tmp_path / reid_swin_profile.REID_SWIN_CONFIG_NAME
    source = tmp_path / "absent" / reid_swin_profile.REID_SWIN_ONNX_NAME
    config_path.write_text(
        "[property]\n"
        f"onnx-file={source}\n"
        f"model-engine-file={engine}\n"
        "batch-size=16\n"
        "network-mode=2\n"
        "gie-unique-id=3\n"
        "process-mode=2\n"
        "network-type=1\n"
        "model-color-format=0\n"
        "net-scale-factor=0.01735207\n"
        "offsets=123.675;116.28;103.53\n"
        "infer-dims=3;256;128\n"
        "maintain-aspect-ratio=0\n"
        "operate-on-gie-id=1\n"
        "operate-on-class-ids=0\n"
        "input-object-min-width=24\n"
        "input-object-min-height=48\n"
        "secondary-reinfer-interval=12\n"
        "output-tensor-meta=1\n"
        "classifier-async-mode=0\n",
        encoding="utf-8",
    )
    pipeline_config = {
        "models": {
            "reid": {
                "enable": True,
                "name": "reid_sgie",
                "config-file-path": str(config_path),
                "engine": str(engine),
                "batch_size": 16,
                "gie_id": 3,
                "layer": "fc_pred",
                "embedding_dim": 256,
                "attach_tensor_meta": True,
            }
        }
    }
    yaml_path = tmp_path / "infer.yaml"
    yaml_path.write_text("version: 1\n", encoding="utf-8")

    code = """
import json
import logging
import sys
from pathlib import Path
from noesis import ds9_runtime_core as runtime
runtime._preflight_reid_profile(
    json.loads(sys.argv[1]),
    Path(sys.argv[2]),
    logging.getLogger('ds9-reid-engine-only-test'),
)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "DS9"), str(ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(pipeline_config), str(yaml_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not source.exists()


@pytest.mark.parametrize(("profile", "size"), [("yolo11", None), ("yolo26", "m")])
def test_ds9_detection_materializer_absolutizes_labels_before_relocation(
    tmp_path: Path,
    profile: str,
    size: str | None,
) -> None:
    checkout = tmp_path / "checkout"
    pipeline_dir = checkout / "DS9" / "pipelines"
    parser = pipeline_dir / "nvdsinfer_yolo_detect" / "libnvdsparsebbox_yolo.so"
    parser.parent.mkdir(parents=True)
    parser.write_bytes(b"parser")
    (pipeline_dir / "config_infer_primary_yolo11.ini").write_text(
        "[property]\n"
        "onnx-file=missing.onnx\n"
        "custom-lib-path=../stale-parser.so\n"
        "model-engine-file=stale.engine\n"
        "labelfile-path=../models/coco_labels.txt\n"
        "batch-size=3\n",
        encoding="utf-8",
    )
    (pipeline_dir / "config_preproc.ini").write_text(
        "[property]\n"
        "network-input-shape=3;3;640;640\n"
        "processing-width=640\n"
        "processing-height=640\n"
        "tensor-name=input\n",
        encoding="utf-8",
    )

    model_dir = tmp_path / "artifacts" / "models"
    engine_dir = model_dir / "engines"
    engine_dir.mkdir(parents=True)
    labels = model_dir / "coco_labels.txt"
    labels.write_text("person\n", encoding="utf-8")
    engine_name = (
        "yolo11m_b3_fp16.engine"
        if profile == "yolo11"
        else "yolo26m_b3_fp16.engine"
    )
    engine = engine_dir / engine_name
    engine.write_bytes(b"engine")
    build_dir = tmp_path / "private-runtime" / "build"

    code = """
import configparser
import json
import logging
import sys
from pathlib import Path

from noesis import ds9_runtime_core as runtime
from noesis_core import inference_runtime_contract as contract

profile = sys.argv[1]
size = None if sys.argv[2] == '-' else sys.argv[2]
assets = runtime._materialize_yolo_detect_pgie_ini(
    profile,
    size,
    logging.getLogger('ds9-detect-relocation-test'),
)
parser = configparser.ConfigParser(interpolation=None, strict=False)
parser.read(assets['pgie_config'], encoding='utf-8')
first_stage_labels = parser['property']['labelfile-path']
contract._assert_parser_only_library = lambda _path: None
derived = contract.materialize_nvinfer_engine_only_config(
    source_config=Path(assets['pgie_config']),
    engine_path=Path(assets['engine']),
    output_root=Path(sys.argv[3]),
    component_name='pgie',
    repo_root=Path(sys.argv[4]),
)
parser = configparser.ConfigParser(interpolation=None, strict=False)
parser.read(derived, encoding='utf-8')
print(json.dumps({
    'first_stage_labels': first_stage_labels,
    'derived_labels': parser['property']['labelfile-path'],
    'source_config': str(assets['pgie_config']),
    'derived_config': str(derived),
}))
"""
    env = dict(os.environ)
    env.update(
        {
            "NOESIS_MODEL_DIR": str(model_dir),
            "NOESIS_ONNX_DIR": str(model_dir / "onnx"),
            "NOESIS_ENGINE_DIR": str(engine_dir),
            "NOESIS_PIPELINE_DIR": str(pipeline_dir),
            "NOESIS_BUILD_DIR": str(build_dir),
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT / "DS9"), str(ROOT), env.get("PYTHONPATH", ""))
            ),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            profile,
            size or "-",
            str(tmp_path / "derived"),
            str(checkout),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["source_config"]).is_relative_to(build_dir)
    assert Path(payload["derived_config"]).is_relative_to(tmp_path / "derived")
    assert Path(payload["first_stage_labels"]) == labels.resolve()
    assert Path(payload["derived_labels"]) == labels.resolve()


@pytest.mark.parametrize("empty", [False, True])
def test_ds9_detection_materializer_rejects_missing_or_empty_labels(
    tmp_path: Path,
    empty: bool,
) -> None:
    model_dir = tmp_path / "external-artifacts" / "models"
    model_dir.mkdir(parents=True)
    labels = model_dir / "coco_labels.txt"
    if empty:
        labels.touch()
    build_dir = tmp_path / "private-runtime" / "build"
    code = """
import logging
from noesis import ds9_runtime_core as runtime
runtime._materialize_yolo_detect_pgie_ini(
    'yolo26',
    'm',
    logging.getLogger('ds9-detect-label-failure-test'),
)
"""
    env = dict(os.environ)
    env.update(
        {
            "NOESIS_MODEL_DIR": str(model_dir),
            "NOESIS_ONNX_DIR": str(model_dir / "onnx"),
            "NOESIS_ENGINE_DIR": str(model_dir / "engines"),
            "NOESIS_PIPELINE_DIR": str(ROOT / "DS9" / "pipelines"),
            "NOESIS_BUILD_DIR": str(build_dir),
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT / "DS9"), str(ROOT), env.get("PYTHONPATH", ""))
            ),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "YOLO detection labels file missing or empty" in (
        result.stdout + result.stderr
    )
    assert not (build_dir / "config_infer_primary_yolo26_m.ini").exists()


def test_ds9_detection_materializer_requires_label_field_before_writing(
    tmp_path: Path,
) -> None:
    pipeline_dir = tmp_path / "checkout" / "DS9" / "pipelines"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "config_infer_primary_yolo11.ini").write_text(
        "[property]\n"
        "custom-lib-path=stale-parser.so\n"
        "model-engine-file=stale.engine\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "external-artifacts" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "coco_labels.txt").write_text("person\n", encoding="utf-8")
    build_dir = tmp_path / "private-runtime" / "build"
    code = """
import logging
from noesis import ds9_runtime_core as runtime
runtime._materialize_yolo_detect_pgie_ini(
    'yolo26',
    'm',
    logging.getLogger('ds9-detect-label-field-test'),
)
"""
    env = dict(os.environ)
    env.update(
        {
            "NOESIS_MODEL_DIR": str(model_dir),
            "NOESIS_ONNX_DIR": str(model_dir / "onnx"),
            "NOESIS_ENGINE_DIR": str(model_dir / "engines"),
            "NOESIS_PIPELINE_DIR": str(pipeline_dir),
            "NOESIS_BUILD_DIR": str(build_dir),
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT / "DS9"), str(ROOT), env.get("PYTHONPATH", ""))
            ),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "model-engine-file/labelfile-path" in (result.stdout + result.stderr)
    assert not (build_dir / "config_infer_primary_yolo26_m.ini").exists()


def test_ds9_canonical_profile_selection_is_mode_specific_and_overridable() -> None:
    code = """
import json
from argparse import Namespace
from noesis import ds9_runtime_core as runtime

def select(mode, profile, explicit, size=None):
    return runtime._resolve_pgie_selection(
        Namespace(
            pgie_profile=profile,
            _pgie_profile_explicit=explicit,
            size=size,
        ),
        mode,
    )

print(json.dumps({
    'baseline': select('baseline', 'yolo26', False),
    'v3dt': select('v3dt', 'yolo26', False),
    'explicit': select('v3dt', 'yolo11_seg', True),
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "DS9"), str(ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    selections = json.loads(result.stdout)
    assert selections["baseline"] == ["yolo26", "m"]
    assert selections["v3dt"] == ["yolo26_seg", "s"]
    assert selections["explicit"] == ["yolo11_seg", None]


def test_yolo26_runtime_materializer_omits_absent_onnx_source(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "yolo26s.engine"
    engine.write_bytes(b"engine")
    source = tmp_path / "missing.onnx"

    assets = ds8_yolo26_materialization.materialize_yolo26_seg_configs(
        size="s",
        batch_size=3,
        src_ids=(0, 1, 2),
        onnx_path=source,
        engine_path=engine,
        pgie_output_path=tmp_path / "pgie.ini",
        preprocess_output_path=tmp_path / "preprocess.ini",
        include_model_source=False,
    )

    assert not source.exists()
    properties = _read_properties(assets["pgie_config"])
    assert "onnx-file" not in properties
    assert Path(properties["model-engine-file"]) == engine.resolve()
    preprocess = _read_properties(assets["preprocess_config"])
    assert preprocess["tensor-name"] == "images"


def test_wholebody49_runtime_materializer_omits_absent_onnx_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models"
    engine_dir = model_dir / "engines"
    engine_dir.mkdir(parents=True)
    engine = engine_dir / "deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine"
    engine.write_bytes(b"engine")
    source_dir = model_dir / "onnx"
    monkeypatch.setenv("NOESIS_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("NOESIS_ONNX_DIR", str(source_dir))
    monkeypatch.setenv("NOESIS_ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("NOESIS_PIPELINE_DIR", str(Path("pipelines").resolve()))
    monkeypatch.setenv("NOESIS_BUILD_DIR", str(tmp_path / "build"))

    assets = wholebody49_assets.materialize_wholebody49_configs(
        size="s",
        batch_size=3,
        src_ids=(0, 1, 2),
        include_model_source=False,
    )

    assert not source_dir.exists()
    properties = _read_properties(Path(assets["pgie_config"]))
    assert "onnx-file" not in properties
    assert Path(properties["model-engine-file"]) == engine.resolve()


def test_ds8_default_seg_materializer_omits_absent_onnx_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / "pgie.template.ini"
    template.write_text(
        "[property]\n"
        "onnx-file=missing.onnx\n"
        "model-engine-file=old.engine\n"
        "labelfile-path=labels.txt\n"
        "custom-lib-path=parser.so\n",
        encoding="utf-8",
    )
    preprocess_template = tmp_path / "preprocess.template.ini"
    preprocess_template.write_text(
        "[property]\n"
        "network-input-shape=3;3;640;640\n"
        "processing-width=640\n"
        "processing-height=640\n"
        "tensor-name=images\n"
        "src-ids=0;1;2\n",
        encoding="utf-8",
    )
    engine = tmp_path / "model.engine"
    labels = tmp_path / "labels.txt"
    parser = tmp_path / "parser.so"
    for path in (engine, labels, parser):
        path.write_bytes(b"asset")
    source = tmp_path / "missing.onnx"
    monkeypatch.setattr(
        ds8_runtime,
        "_resolve_yolo11_seg_assets",
        lambda _size: {
            "label": "fixture",
            "size": "s",
            "template": template,
            "preprocess_template": preprocess_template,
            "preprocess_output": tmp_path / "preprocess.ini",
            "tensor_name": "images",
            "onnx": source,
            "engine": engine,
            "labels": labels,
            "parser_lib": parser,
            "output": tmp_path / "pgie.ini",
        },
    )

    assets = ds8_runtime._materialize_yolo11_seg_pgie_ini(
        "s",
        SimpleNamespace(info=lambda *_args, **_kwargs: None),
        src_ids=(0, 1, 2),
    )

    assert not source.exists()
    properties = _read_properties(assets["pgie_config"])
    assert "onnx-file" not in properties
    assert Path(properties["model-engine-file"]) == engine.resolve()
