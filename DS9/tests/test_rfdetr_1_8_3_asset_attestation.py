from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from noesis import rfdetr_1_8_3_assets as assets


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_source_sha(label: str, path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def runtime_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    ds9_root = repo_root / "DS9"
    artifact_root = tmp_path / "artifacts"
    model_root = artifact_root / "models"
    onnx_root = model_root / "onnx"
    engine_root = model_root / "engines"
    pipeline_root = ds9_root / "pipelines"
    build_root = ds9_root / "build"
    matrix_path = ds9_root / "config" / "rfdetr_1_8_3_models.json"

    monkeypatch.setattr(assets, "DS9_ROOT", ds9_root)
    monkeypatch.setattr(assets, "MATRIX_PATH", matrix_path)
    monkeypatch.setenv("NOESIS_MODEL_DIR", str(model_root))
    monkeypatch.setenv("NOESIS_ONNX_DIR", str(onnx_root))
    monkeypatch.setenv("NOESIS_ENGINE_DIR", str(engine_root))
    monkeypatch.setenv("NOESIS_PIPELINE_DIR", str(pipeline_root))
    monkeypatch.setenv("NOESIS_BUILD_DIR", str(build_root))

    release = {
        "package": "rfdetr",
        "version": "1.8.3",
        "git_tag": "1.8.3",
        "git_commit": "3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86",
        "source": "https://github.com/roboflow/rf-detr",
        "onnx_opset": 17,
        "batch_size": 3,
        "dynamic_batch": False,
    }
    outputs = {"dets": [3, 100, 4], "labels": [3, 100, 91]}
    onnx_name = "rfdetr-nano_b3_rgb01_imagenet_divstd.onnx"
    engine_name = (
        "rfdetr-nano_b3_rgb01_imagenet_divstd_fp32_no_tf32.engine"
    )
    onnx = (
        onnx_root / "rfdetr" / "1.8.3" / "runtime" / onnx_name
    )
    engine = (
        engine_root
        / "rfdetr"
        / "1.8.3"
        / "runtime"
        / "fp32_no_tf32"
        / engine_name
    )
    onnx.parent.mkdir(parents=True)
    engine.parent.mkdir(parents=True)
    onnx.write_bytes(b"reviewed runtime onnx")
    engine.write_bytes(b"reviewed fp32 no-tf32 engine")

    provenance_root = model_root / "provenance" / "rfdetr" / "1.8.3"
    onnx_receipt = (
        provenance_root
        / "detect_nano.runtime_onnx.sub_div_float32_v1.json"
    )
    tensor_contract = {
        "input": {"input": [3, 3, 384, 384]},
        "outputs": outputs,
        "opset_imports": {"ai.onnx": 17},
        "runtime_adapter": assets.RUNTIME_ADAPTER_SPEC,
        "normalized_input_consumer_count": 1,
    }
    onnx_receipt_payload = {
        "schema": assets.SOURCE_PROVENANCE_SCHEMA,
        "artifact_kind": "runtime_onnx",
        "model_id": "detect_nano",
        "family": "detection",
        "variant": "nano",
        "license": "Apache-2.0",
        "release": release,
        "filename": onnx.name,
        "size_bytes": onnx.stat().st_size,
        "sha256": _sha256(onnx),
        "adapter": assets.RUNTIME_ADAPTER_SPEC,
        "source": {
            "filename": "rfdetr-nano_b3.onnx",
            "size_bytes": 123,
            "sha256": "a" * 64,
            "receipt": "detect_nano.onnx.json",
            "receipt_sha256": "b" * 64,
        },
        "tensor_contract": tensor_contract,
    }
    _write_json(onnx_receipt, onnx_receipt_payload)

    engine_receipt = (
        provenance_root / "runtime" / "fp32_no_tf32" / "detect_nano.engine.json"
    )
    engine_receipt_payload = {
        "schema": assets.RUNTIME_ENGINE_RECEIPT_SCHEMA,
        "promotion_status": "unpromoted",
        "runtime_selected": False,
        "artifact_role": "runtime_input_engine",
        "model_id": "detect_nano",
        "family": "detection",
        "variant": "nano",
        "release": release,
        "source": {
            "runtime_onnx": (
                "models/onnx/rfdetr/1.8.3/runtime/" + onnx.name
            ),
            "runtime_onnx_size_bytes": onnx.stat().st_size,
            "runtime_onnx_sha256": _sha256(onnx),
            "runtime_onnx_receipt": (
                "models/provenance/rfdetr/1.8.3/" + onnx_receipt.name
            ),
            "runtime_onnx_receipt_sha256": _sha256(onnx_receipt),
            "tensor_contract": tensor_contract,
        },
        "build_contract": {
            "profile": "fp32_no_tf32",
            "precision": "fp32",
            "fp16_enabled": False,
            "tf32_enabled": False,
            "batch": {"mode": "static", "size": 3},
            "trtexec_precision_args": ["--noTF32"],
        },
        "engine": {
            "path": (
                "models/engines/rfdetr/1.8.3/runtime/fp32_no_tf32/"
                + engine.name
            ),
            "size_bytes": engine.stat().st_size,
            "sha256": _sha256(engine),
        },
        "runtime_input_contract": {
            "input_contract": assets.RUNTIME_INPUT_CONTRACT,
            "adapter_revision": assets.RUNTIME_ADAPTER_REVISION,
            "adapter": assets.RUNTIME_ADAPTER_SPEC,
        },
    }
    engine_receipt_payload["receipt_sha256"] = assets._json_digest(
        engine_receipt_payload
    )
    _write_json(engine_receipt, engine_receipt_payload)

    runtime = {
        "input_contract": assets.RUNTIME_INPUT_CONTRACT,
        "adapter_revision": assets.RUNTIME_ADAPTER_REVISION,
        "onnx_filename": onnx.name,
        "engine_profile": "fp32_no_tf32",
        "engine_filename": engine.name,
        "onnx_sha256": _sha256(onnx),
        "onnx_receipt_sha256": _sha256(onnx_receipt),
        "engine_sha256": _sha256(engine),
        "engine_receipt_sha256": _sha256(engine_receipt),
    }
    matrix = {
        "schema": assets.MATRIX_SCHEMA,
        "release": release,
        "models": [
            {
                "id": "detect_nano",
                "family": "detection",
                "variant": "nano",
                "enabled": True,
                "license": "Apache-2.0",
                "resolution": 384,
                "onnx_filename": "rfdetr-nano_b3.onnx",
                "runtime": runtime,
                "outputs": outputs,
            }
        ],
    }
    _write_json(matrix_path, matrix)

    parser_source = (
        pipeline_root / "nvdsinfer_rfdetr" / "nvdsinfer_rfdetr.cpp"
    )
    parser_binary = (
        pipeline_root / "nvdsinfer_rfdetr" / "libnvdsinfer_rfdetr.so"
    )
    parser_source.parent.mkdir(parents=True)
    parser_source.write_bytes(b"reviewed parser source")
    parser_binary.write_bytes(b"reviewed parser binary")
    parser_source_label = (
        "DS9/pipelines/nvdsinfer_rfdetr/nvdsinfer_rfdetr.cpp"
    )
    manifest = {
        "schema_version": 2,
        "manifest_id": "noesis-ds9-artifacts",
        "schema": "DS9/docs/asset_manifest.schema.json",
        "target": {
            "cuda": "13.1",
            "tensorrt": "10.14.1.48",
            "deepstream": {"major": 9, "version": "9.0"},
        },
        "artifacts": [
            {
                "id": "parser.rfdetr_detect",
                "kind": "nvinfer_parser",
                "role": "RF-DETR detect-only output parser",
                "output": (
                    "DS9/pipelines/nvdsinfer_rfdetr/"
                    "libnvdsinfer_rfdetr.so"
                ),
                "sources": [parser_source_label],
                "builder": "DS9/scripts/build_custom_parsers.sh",
                "required_profiles": ["rfdetr", "full"],
                "state": "staged_unverified",
                "compatibility": {
                    "deepstream_major": 9,
                    "cuda": "13.1",
                    "tensorrt": "10.14.1.48",
                },
                "provenance": {
                    "source_sha256": _manifest_source_sha(
                        parser_source_label, parser_source
                    ),
                    "output_sha256": _sha256(parser_binary),
                    "built_at_utc": "2026-07-25T01:43:54Z",
                    "build_host": "test-host",
                    "command": "test build",
                },
            }
        ],
    }
    manifest_path = ds9_root / "asset_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    pgie_template = pipeline_root / "config_infer_primary_rfdetr.template.ini"
    pgie_template.write_text(
        """[property]
onnx-file=@ONNX_PATH@
model-engine-file=@ENGINE_PATH@
labelfile-path=@LABELS_PATH@
custom-lib-path=@CUSTOM_LIB@
batch-size=@BATCH_SIZE@
interval=1
gie-unique-id=1
network-mode=@NETWORK_MODE@
input-tensor-from-meta=1
maintain-aspect-ratio=0
symmetric-padding=0
cluster-mode=4
num-detected-classes=1
operate-on-class-ids=0
output-tensor-meta=0
network-type=0
parse-bbox-func-name=NvDsInferParseRFDETR
output-blob-names=dets;labels

[class-attrs-all]
pre-cluster-threshold=0.5
topk=@TOPK@
""",
        encoding="utf-8",
    )
    preproc_template = (
        pipeline_root / "config_preproc_rfdetr_1_8_3.template.ini"
    )
    preproc_template.write_text(
        """[property]
network-input-shape=@BATCH_SIZE@;3;@RESOLUTION@;@RESOLUTION@
processing-width=@RESOLUTION@
processing-height=@RESOLUTION@
tensor-name=input
network-color-format=0
tensor-data-type=0
maintain-aspect-ratio=0
symmetric-padding=0
custom-lib-path=@PREPROCESS_LIB@
src-ids=@SRC_IDS@
""",
        encoding="utf-8",
    )
    (model_root / "coco_labels.txt").write_text("person\n", encoding="utf-8")

    return {
        "matrix": matrix_path,
        "onnx": onnx,
        "onnx_receipt": onnx_receipt,
        "engine": engine,
        "engine_receipt": engine_receipt,
        "engine_receipt_payload": engine_receipt_payload,
        "manifest": manifest_path,
        "parser_source": parser_source,
        "parser": parser_binary,
        "pgie_output": (
            build_root / "config_infer_primary_rfdetr_1_8_3_detection_n.ini"
        ),
        "preproc_output": (
            build_root
            / "config_preproc_rfdetr_1_8_3_detection_n_b3.ini"
        ),
    }


def _materialize() -> dict[str, Path | str | int | bool]:
    return assets.materialize_rfdetr_1_8_3_configs(
        family="detection",
        size="n",
        batch_size=3,
        src_ids=(0, 1, 2),
    )


def _update_matrix_anchor(
    tree: dict[str, Any], key: str, digest: str
) -> None:
    payload = json.loads(tree["matrix"].read_text(encoding="utf-8"))
    payload["models"][0]["runtime"][key] = digest
    _write_json(tree["matrix"], payload)


def _select_fp16_profile(tree: dict[str, Any]) -> tuple[Path, Path]:
    fp16_engine_name = (
        "rfdetr-nano_b3_rgb01_imagenet_divstd_fp16_tf32.engine"
    )
    fp16_engine = (
        tree["engine"].parent.parent / "fp16_tf32" / fp16_engine_name
    )
    fp16_engine.parent.mkdir(parents=True)
    fp16_engine.write_bytes(tree["engine"].read_bytes())

    receipt_payload = json.loads(
        tree["engine_receipt"].read_text(encoding="utf-8")
    )
    receipt_payload["build_contract"] = {
        "profile": "fp16_tf32",
        "precision": "fp16",
        "fp16_enabled": True,
        "tf32_enabled": True,
        "batch": {"mode": "static", "size": 3},
        "trtexec_precision_args": ["--fp16"],
    }
    receipt_payload["engine"] = {
        "path": (
            "models/engines/rfdetr/1.8.3/runtime/fp16_tf32/"
            + fp16_engine.name
        ),
        "size_bytes": fp16_engine.stat().st_size,
        "sha256": _sha256(fp16_engine),
    }
    receipt_payload.pop("receipt_sha256")
    receipt_payload["receipt_sha256"] = assets._json_digest(
        receipt_payload
    )
    fp16_receipt = (
        tree["engine_receipt"].parent.parent
        / "fp16_tf32"
        / tree["engine_receipt"].name
    )
    _write_json(fp16_receipt, receipt_payload)

    matrix = json.loads(tree["matrix"].read_text(encoding="utf-8"))
    runtime = matrix["models"][0]["runtime"]
    runtime.update(
        {
            "engine_profile": "fp16_tf32",
            "engine_filename": fp16_engine.name,
            "engine_sha256": _sha256(fp16_engine),
            "engine_receipt_sha256": _sha256(fp16_receipt),
        }
    )
    _write_json(tree["matrix"], matrix)
    return fp16_engine, fp16_receipt


def test_materialization_attests_exact_runtime_chain(
    runtime_tree: dict[str, Any],
) -> None:
    result = _materialize()
    assert Path(result["pgie_config"]) == runtime_tree["pgie_output"]
    assert Path(result["preprocess_config"]) == runtime_tree["preproc_output"]
    assert "onnx-file" not in runtime_tree["pgie_output"].read_text(
        encoding="utf-8"
    )
    assert "network-mode=0" in runtime_tree["pgie_output"].read_text(
        encoding="utf-8"
    )
    assert "interval=1" in runtime_tree["pgie_output"].read_text(
        encoding="utf-8"
    )


def test_materialization_accepts_reviewed_fp16_tf32_profile(
    runtime_tree: dict[str, Any],
) -> None:
    fp16_engine, fp16_receipt = _select_fp16_profile(runtime_tree)

    result = _materialize()

    assert Path(result["engine"]) == fp16_engine
    assert fp16_receipt.is_file()
    assert "network-mode=2" in runtime_tree["pgie_output"].read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("batch_size", (1, 2, 4))
def test_materialization_rejects_nonrelease_batch_size(
    runtime_tree: dict[str, Any], batch_size: int
) -> None:
    with pytest.raises(
        ValueError, match="requires static batch_size=3"
    ):
        assets.materialize_rfdetr_1_8_3_configs(
            family="detection",
            size="n",
            batch_size=batch_size,
            src_ids=(0, 1, 2),
        )
    assert not runtime_tree["pgie_output"].exists()


@pytest.mark.parametrize(
    "name",
    ("onnx", "engine", "parser_source", "parser"),
)
def test_materialization_rejects_changed_runtime_bytes(
    runtime_tree: dict[str, Any], name: str
) -> None:
    runtime_tree[name].write_bytes(b"changed after review")
    with pytest.raises(assets.RFDETRAssetAttestationError):
        _materialize()
    assert not runtime_tree["pgie_output"].exists()


def test_materialization_rejects_missing_revision_bound_receipt(
    runtime_tree: dict[str, Any],
) -> None:
    runtime_tree["onnx_receipt"].unlink()
    with pytest.raises(
        assets.RFDETRAssetAttestationError,
        match="runtime ONNX receipt is missing",
    ):
        _materialize()


def test_materialization_rejects_symlinked_engine(
    runtime_tree: dict[str, Any],
) -> None:
    engine = runtime_tree["engine"]
    target = engine.with_name("target.engine")
    target.write_bytes(engine.read_bytes())
    engine.unlink()
    engine.symlink_to(target.name)
    with pytest.raises(
        assets.RFDETRAssetAttestationError, match="contains a symlink"
    ):
        _materialize()


def test_materialization_rejects_duplicate_receipt_metadata(
    runtime_tree: dict[str, Any],
) -> None:
    receipt = runtime_tree["onnx_receipt"]
    receipt.write_text(
        '{"schema":"first","schema":"second"}\n',
        encoding="utf-8",
    )
    _update_matrix_anchor(
        runtime_tree, "onnx_receipt_sha256", _sha256(receipt)
    )
    with pytest.raises(
        assets.RFDETRAssetAttestationError, match="not strict JSON"
    ):
        _materialize()


def test_materialization_rejects_precision_receipt_drift(
    runtime_tree: dict[str, Any],
) -> None:
    payload = runtime_tree["engine_receipt_payload"]
    payload["build_contract"]["tf32_enabled"] = True
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = assets._json_digest(payload)
    _write_json(runtime_tree["engine_receipt"], payload)
    _update_matrix_anchor(
        runtime_tree,
        "engine_receipt_sha256",
        _sha256(runtime_tree["engine_receipt"]),
    )
    with pytest.raises(
        assets.RFDETRAssetAttestationError,
        match="engine receipt contract drifted",
    ):
        _materialize()


def test_materialization_rejects_malformed_parser_provenance(
    runtime_tree: dict[str, Any],
) -> None:
    manifest = yaml.safe_load(
        runtime_tree["manifest"].read_text(encoding="utf-8")
    )
    manifest["artifacts"][0]["provenance"]["built_at_utc"] = "not-a-time"
    runtime_tree["manifest"].write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(
        assets.RFDETRAssetAttestationError,
        match="provenance metadata is malformed",
    ):
        _materialize()


def test_parser_provenance_rejects_raw_source_sha(
    runtime_tree: dict[str, Any],
) -> None:
    manifest = yaml.safe_load(
        runtime_tree["manifest"].read_text(encoding="utf-8")
    )
    raw_sha = _sha256(runtime_tree["parser_source"])
    assert (
        raw_sha
        != manifest["artifacts"][0]["provenance"]["source_sha256"]
    )
    manifest["artifacts"][0]["provenance"]["source_sha256"] = raw_sha
    runtime_tree["manifest"].write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(
        assets.RFDETRAssetAttestationError,
        match="bytes do not match manifest provenance",
    ):
        _materialize()


def test_runtime_contract_requires_all_matrix_digest_anchors(
    runtime_tree: dict[str, Any],
) -> None:
    matrix = json.loads(runtime_tree["matrix"].read_text(encoding="utf-8"))
    matrix["models"][0]["runtime"].pop("engine_receipt_sha256")
    _write_json(runtime_tree["matrix"], matrix)
    with pytest.raises(ValueError, match="runtime engine receipt sha256"):
        assets.resolve_rfdetr_1_8_3_assets("detection", "n")


def test_runtime_contract_rejects_unknown_engine_profile(
    runtime_tree: dict[str, Any],
) -> None:
    matrix = json.loads(runtime_tree["matrix"].read_text(encoding="utf-8"))
    matrix["models"][0]["runtime"]["engine_profile"] = "fp8_unknown"
    _write_json(runtime_tree["matrix"], matrix)
    with pytest.raises(
        ValueError, match="runtime engine profile must be one of"
    ):
        assets.resolve_rfdetr_1_8_3_assets("detection", "n")


def test_disabled_row_is_rejected_before_runtime_contract_parsing(
    runtime_tree: dict[str, Any],
) -> None:
    matrix = json.loads(runtime_tree["matrix"].read_text(encoding="utf-8"))
    matrix["models"][0]["enabled"] = False
    matrix["models"][0].pop("runtime")
    _write_json(runtime_tree["matrix"], matrix)
    with pytest.raises(ValueError, match="runtime model is disabled"):
        assets.resolve_rfdetr_1_8_3_assets("detection", "n")
