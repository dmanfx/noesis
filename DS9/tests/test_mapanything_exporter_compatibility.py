from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTER_PATH = (
    REPO_ROOT / "utils/onnx2trt/export_ma_onnx/export_to_onnx.py"
)


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "mapanything_exporter_compatibility_test",
        EXPORTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_exporter()


class CompatibleMapAnything:
    def __init__(
        self,
        name,
        encoder_config,
        use_register_tokens_from_encoder,
        info_sharing_mlp_layer_str,
    ):
        pass


class CompatibleDINOv2:
    def __init__(
        self,
        name,
        norm_returned_features,
        torch_hub_pretrained,
    ):
        pass


class LegacyMapAnything:
    def __init__(self, name, encoder_config):
        pass


class LegacyDINOv2:
    def __init__(self, name, *args, **kwargs):
        pass


def _v11_config() -> dict:
    return {
        "name": "mapanything",
        "encoder_config": {
            "encoder_str": "dinov2",
            "uses_torch_hub": True,
            "name": "dinov2_giant_24_layers",
            "norm_returned_features": False,
            "torch_hub_pretrained": False,
        },
        "use_register_tokens_from_encoder": True,
        "info_sharing_mlp_layer_str": "swiglufused",
    }


def test_v11_checkpoint_contract_accepts_matching_source_apis() -> None:
    exporter._validate_hf_config_compatibility(
        model_class=CompatibleMapAnything,
        encoder_class=CompatibleDINOv2,
        model_config=_v11_config(),
        uniception_version="0.1.7",
    )


def test_v11_checkpoint_contract_rejects_old_uniception_without_shim() -> None:
    with pytest.raises(
        exporter.ModelCompatibilityError,
        match=r"UniCeption >=0\.1\.7",
    ):
        exporter._validate_hf_config_compatibility(
            model_class=CompatibleMapAnything,
            encoder_class=CompatibleDINOv2,
            model_config=_v11_config(),
            uniception_version="0.1.4",
        )


def test_v11_checkpoint_contract_rejects_silently_ignored_model_options() -> None:
    with pytest.raises(
        exporter.ModelCompatibilityError,
        match="info_sharing_mlp_layer_str",
    ):
        exporter._validate_hf_config_compatibility(
            model_class=LegacyMapAnything,
            encoder_class=CompatibleDINOv2,
            model_config=_v11_config(),
            uniception_version="0.1.7",
        )


def test_v11_checkpoint_contract_rejects_silently_forwarded_encoder_option() -> None:
    with pytest.raises(
        exporter.ModelCompatibilityError,
        match="norm_returned_features",
    ):
        exporter._validate_hf_config_compatibility(
            model_class=CompatibleMapAnything,
            encoder_class=LegacyDINOv2,
            model_config=_v11_config(),
            uniception_version="0.1.7",
        )


def test_huggingface_load_is_revision_pinned_and_strict() -> None:
    observed = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            observed["model_id"] = model_id
            observed.update(kwargs)
            return object()

    cfg = SimpleNamespace(
        hf_model_id="facebook/map-anything-apache",
        hf_revision="4cf3561e403dcec91b41629f0ce7793e3f04d15c",
    )
    exporter._load_hf_pretrained_model(cfg, FakeModel)

    assert observed == {
        "model_id": "facebook/map-anything-apache",
        "revision": "4cf3561e403dcec91b41629f0ce7793e3f04d15c",
        "strict": True,
    }


def test_export_device_is_explicit_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert exporter._resolve_export_device("cpu").type == "cpu"
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA export was requested"):
        exporter._resolve_export_device("cuda")


def test_wrapper_onnx_contract_keeps_only_batch_dynamic(tmp_path: Path) -> None:
    class FakeMapAnything(exporter.nn.Module):
        def forward(self, views, **_kwargs):
            image = views[0]["img"]
            value = image[:, 0]
            points = exporter.torch.stack((value, value, value), dim=-1)
            return [
                {
                    "pts3d_cam": points,
                    "conf": value,
                    "non_ambiguous_mask": value > 0,
                }
            ]

    wrapper = exporter.MapAnythingDepthWrapper(
        FakeMapAnything(),
        "test",
        exporter.torch.zeros(1, 3, 1, 1),
        exporter.torch.ones(1, 3, 1, 1),
        use_fused_input=False,
        output_height=2,
        output_width=4,
        return_conf_mask=True,
    )
    target = tmp_path / "wrapper.onnx"
    exporter.torch.onnx.export(
        wrapper,
        (exporter.torch.rand(1, 3, 2, 4),),
        str(target),
        dynamo=False,
        opset_version=17,
        input_names=["images"],
        output_names=["depth", "conf", "mask"],
        dynamic_axes={
            "images": {0: "batch"},
            "depth": {0: "batch"},
            "conf": {0: "batch"},
            "mask": {0: "batch"},
        },
    )
    model = onnx.load(str(target), load_external_data=False)

    def dimensions(value_info):
        return [
            dimension.dim_param or dimension.dim_value
            for dimension in value_info.type.tensor_type.shape.dim
        ]

    assert [dimensions(value) for value in model.graph.output] == [
        ["batch", 1, 2, 4],
        ["batch", 1, 2, 4],
        ["batch", 1, 2, 4],
    ]
