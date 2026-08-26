from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from noesis import rfdetr_1_8_3_assets as assets
from noesis.pipelines import hooks


DS9_ROOT = Path(__file__).resolve().parents[1]


def _keypoint_pgie_properties() -> dict[str, str]:
    return {
        "batch-size": "3",
        "gie-unique-id": "1",
        "input-tensor-from-meta": "0",
        "maintain-aspect-ratio": "0",
        "symmetric-padding": "0",
        "cluster-mode": "4",
        "num-detected-classes": "1",
        "operate-on-class-ids": "0",
        "output-tensor-meta": "1",
        "onnx-file": "",
        "model-engine-file": "/models/keypoint.engine",
        "network-type": "0",
        "network-mode": "2",
        "parse-bbox-func-name": "NvDsInferParseRFDETRKeypoint",
        "output-blob-names": "dets;labels;keypoints",
        "disable-output-host-copy": "0",
        "net-scale-factor": "0.00392156862745098",
        "infer-dims": "3;576;576",
        "model-color-format": "0",
        "scaling-filter": "1",
    }


def _keypoint_class_attrs() -> dict[str, str]:
    return {
        "pre-cluster-threshold": "0.4",
        "topk": "100",
    }


def _rfdetr_bridge_config() -> dict[str, object]:
    return {
        "enable": True,
        "tensor_source": "rfdetr_pgie_frame",
        "gie_id": 1,
        "attach_component": "world_observation_stage",
        "model_size": [576, 576],
        "score_threshold": 0.4,
        "kpt_threshold": 0.35,
        "letterbox": False,
        "match_min_iou": 0.7,
        "match_ambiguity_margin": 0.05,
        "pose_cache_max_age_frames": 0,
    }


def test_keypoint_preview_is_the_only_reviewed_runtime_variant() -> None:
    assert assets.RFDETR_KEYPOINT_VARIANTS == ("preview",)
    row = assets._model_row("keypoint", "preview")
    assert row["id"] == "keypoint_preview"
    assert row["outputs"] == {
        "dets": [3, 100, 4],
        "labels": [3, 100, 2],
        "keypoints": [3, 100, 34, 8],
    }


def test_keypoint_pgie_requires_frame_owned_host_tensor_meta() -> None:
    props = _keypoint_pgie_properties()
    assets.validate_rfdetr_pgie_properties(
        props,
        family="keypoint",
        engine_profile="fp16_tf32",
        query_count=100,
        batch_size=3,
    )

    props["input-tensor-from-meta"] = "1"
    with pytest.raises(ValueError, match="input-tensor-from-meta=0"):
        assets.validate_rfdetr_pgie_properties(
            props,
            family="keypoint",
            engine_profile="fp16_tf32",
            query_count=100,
            batch_size=3,
        )


def test_keypoint_pgie_locks_precision_profile_to_network_mode() -> None:
    props = _keypoint_pgie_properties()
    for profile, network_mode in (
        ("fp32_no_tf32", "0"),
        ("fp16_tf32", "2"),
    ):
        props["network-mode"] = network_mode
        assets.validate_rfdetr_pgie_properties(
            props,
            family="keypoint",
            engine_profile=profile,
            query_count=100,
            batch_size=3,
        )

    props["network-mode"] = "0"
    with pytest.raises(ValueError, match="requires network-mode=2"):
        assets.validate_rfdetr_pgie_properties(
            props,
            family="keypoint",
            engine_profile="fp16_tf32",
            query_count=100,
            batch_size=3,
        )
    with pytest.raises(ValueError, match="engine profile must be one of"):
        assets.validate_rfdetr_pgie_properties(
            props,
            family="keypoint",
            engine_profile="fp8_unknown",
            query_count=100,
            batch_size=3,
        )


def test_keypoint_class_attrs_lock_parser_and_bridge_thresholds() -> None:
    assert assets.validate_rfdetr_class_attrs(
        _keypoint_class_attrs(),
        family="keypoint",
        query_count=100,
        expected_threshold=0.4,
    ) == pytest.approx(0.4)

    for key, value, expected in (
        ("topk", "", "requires topk=100"),
        ("pre-cluster-threshold", "", "requires a numeric"),
        ("pre-cluster-threshold", "0.39", "requires pre-cluster-threshold=0.4"),
    ):
        attrs = _keypoint_class_attrs()
        attrs[key] = value
        with pytest.raises(ValueError, match=expected):
            assets.validate_rfdetr_class_attrs(
                attrs,
                family="keypoint",
                query_count=100,
                expected_threshold=0.4,
            )


def test_keypoint_template_uses_exact_native_preprocessing_contract() -> None:
    text = (
        DS9_ROOT
        / "pipelines"
        / "config_infer_primary_rfdetr_keypoint.template.ini"
    ).read_text(encoding="utf-8")
    for expected in (
        "net-scale-factor=0.00392156862745098",
        "infer-dims=3;576;576",
        "input-tensor-from-meta=0",
        "network-mode=@NETWORK_MODE@",
        "maintain-aspect-ratio=0",
        "symmetric-padding=0",
        "scaling-filter=1",
        "disable-output-host-copy=0",
        "output-blob-names=dets;labels;keypoints",
        "output-tensor-meta=1",
        "parse-bbox-func-name=NvDsInferParseRFDETRKeypoint",
    ):
        assert expected in text


def test_keypoint_parser_locks_raw_output_and_box_contracts() -> None:
    source = (
        DS9_ROOT
        / "pipelines"
        / "nvdsinfer_rfdetr_keypoint"
        / "nvdsinfer_rfdetr_keypoint.cpp"
    ).read_text(encoding="utf-8")
    for expected in (
        "constexpr std::size_t kQueryCount = 100",
        "constexpr std::size_t kClassCount = 2",
        "constexpr std::size_t kKeypointSlots = 34",
        "constexpr std::size_t kPersonClassIndex = 1",
        "constexpr std::size_t kPersonKeypointOffset = 17",
        "RF-DETR 1.8.3 exports raw pred_boxes as normalized cx,cy,w,h",
        "logits_buffer + kPersonClassIndex",
        "official_fused_person_score",
        "kKeypointTraceAlpha = 0.2",
        "missing class-0 pre-cluster threshold",
        "object.classId = 0",
        "query-to-object association bridge",
    ):
        assert expected in source
    assert "NOESIS_RFDETR_PERSON_CLASS_IDX" not in source


def test_detection_and_segmentation_parsers_lock_coco_person_class() -> None:
    for relative in (
        "nvdsinfer_rfdetr/nvdsinfer_rfdetr.cpp",
        "nvdsinfer_rfdetr_seg/nvdsinfer_rfdetr_seg.cpp",
    ):
        source = (DS9_ROOT / "pipelines" / relative).read_text(
            encoding="utf-8"
        )
        assert "constexpr std::size_t kPersonClassIndex = 1" in source
        assert "NOESIS_RFDETR_PERSON_CLASS_IDX" not in source


def test_keypoint_parser_is_in_ds9_build_and_artifact_inventory() -> None:
    build_script = (
        DS9_ROOT / "scripts" / "build_all_parsers_ds9.sh"
    ).read_text(encoding="utf-8")
    manifest = (DS9_ROOT / "asset_manifest.yaml").read_text(encoding="utf-8")
    assert 'pipelines/nvdsinfer_rfdetr_keypoint"' in build_script
    assert "id: parser.rfdetr_keypoint" in manifest
    assert (
        "output: "
        "DS9/pipelines/nvdsinfer_rfdetr_keypoint/"
        "libnvdsinfer_rfdetr_keypoint.so"
    ) in manifest


def test_pose_hook_selects_rfdetr_bridge_without_yolo_sgie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_POSE_FEATURES_ENABLED", "1")
    component = SimpleNamespace(config={}, name="world_observation_stage")
    pipeline = SimpleNamespace(
        config={
            "models": {
                "pose": {"enable": False},
                "rfdetr_keypoint": _rfdetr_bridge_config(),
            }
        },
        components={"world_observation_stage": component},
        ds_pipeline=None,
    )

    hooks.attach_pose_feature_hook(pipeline)

    processor = component.config["_pose_feature_processor"]
    assert processor.tensor_source == "rfdetr_pgie_frame"
    assert processor.model_label == "rfdetr-keypoint-preview-1.8.3"
    assert processor.gie_id == 1
    assert processor.letterbox is False
    assert processor.cache_max_age_frames == 0


def test_rfdetr_pose_probe_attach_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_POSE_FEATURES_ENABLED", "1")
    monkeypatch.setattr(hooks, "BatchMetadataOperator", object())
    monkeypatch.setattr(
        hooks,
        "Probe",
        lambda name, operator: SimpleNamespace(name=name, operator=operator),
    )

    class Pipeline:
        @staticmethod
        def attach(component_name: str, probe: object) -> None:
            raise RuntimeError(f"attach failed for {component_name}")

    component = SimpleNamespace(config={}, name="world_observation_stage")
    pipeline = SimpleNamespace(
        config={
            "models": {
                "pose": {"enable": False},
                "rfdetr_keypoint": _rfdetr_bridge_config(),
            }
        },
        components={"world_observation_stage": component},
        ds_pipeline=Pipeline(),
    )
    with pytest.raises(
        RuntimeError,
        match="RF-DETR keypoint pose feature probe attachment failed",
    ):
        hooks.attach_pose_feature_hook(pipeline)


def test_pose_hook_rejects_hidden_yolo_pose_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_POSE_FEATURES_ENABLED", "1")
    pipeline = SimpleNamespace(
        config={
            "models": {
                "pose": {
                    "enable": True,
                    "name": "yolo26_pose",
                },
                "rfdetr_keypoint": _rfdetr_bridge_config(),
            }
        },
        components={},
        ds_pipeline=None,
    )
    with pytest.raises(ValueError, match="cannot run with the YOLO pose SGIE"):
        hooks.attach_pose_feature_hook(pipeline)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("score_threshold", 0.0, "score_threshold must match"),
        ("kpt_threshold", 0.0, "kpt_threshold must be 0.35"),
        ("match_min_iou", 0.0, "match_min_iou must be in"),
        (
            "pose_cache_max_age_frames",
            1,
            "pose_cache_max_age_frames must be 0",
        ),
    ),
)
def test_pose_hook_rejects_rfdetr_contract_substitution(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setenv("NOESIS_POSE_FEATURES_ENABLED", "1")
    bridge = _rfdetr_bridge_config()
    bridge[key] = value
    pipeline = SimpleNamespace(
        config={
            "models": {
                "pose": {"enable": False},
                "rfdetr_keypoint": bridge,
            }
        },
        components={
            "world_observation_stage": SimpleNamespace(
                config={},
                name="world_observation_stage",
            )
        },
        ds_pipeline=None,
    )
    with pytest.raises(ValueError, match=message):
        hooks.attach_pose_feature_hook(pipeline)


def test_rfdetr_native_match_rows_are_bound_to_exact_python_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        @staticmethod
        def extract_rfdetr_keypoint_matches(
            frame_meta: object,
            gie_id: int,
            score_threshold: float,
            min_iou: float,
            ambiguity_margin: float,
        ) -> dict[str, object]:
            assert gie_id == 1
            assert score_threshold == pytest.approx(0.4)
            assert min_iou == pytest.approx(0.7)
            assert ambiguity_margin == pytest.approx(0.05)
            keypoints = [[10.0 + index, 20.0 + index, 0.9] for index in range(17)]
            return {
                "matches": [
                    {
                        "object_index": 0,
                        "object_id": 42,
                        "query_index": 7,
                        "base_score": 0.85,
                        "match_iou": 0.9,
                        "bbox": [5.0, 6.0, 100.0, 200.0],
                        "score": 0.8,
                        "keypoints_roi": keypoints,
                        "keypoints_abs": keypoints,
                    }
                ],
                "diagnostics": {
                    "matched_objects": 1,
                    "ambiguous_objects": 0,
                    "unmatched_objects": 0,
                    "person_objects": 1,
                    "person_queries": 1,
                },
            }

    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", Native())
    processor = hooks.PoseFeatureProcessor(
        pipeline=SimpleNamespace(),
        gie_id=1,
        tensor_source="rfdetr_pgie_frame",
        score_threshold=0.4,
        match_min_iou=0.7,
        match_ambiguity_margin=0.05,
    )
    obj = SimpleNamespace(
        object_id=42,
        rect_params=SimpleNamespace(
            left=5.0,
            top=6.0,
            width=100.0,
            height=200.0,
        ),
    )
    result = processor._extract_rfdetr_frame_native(
        SimpleNamespace(),
        [obj],
    )

    score, roi, absolute = result[0]
    assert score == pytest.approx(0.8)
    assert roi.shape == (17, 3)
    assert absolute.shape == (17, 3)
    assert np.isfinite(roi).all()


def test_rfdetr_matched_rows_bypass_yolo_pose_budget_and_attach_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attached: list[tuple[object, object, str, bool]] = []

    class Native:
        @staticmethod
        def attach_pose_features(
            batch_meta: object,
            obj_meta: object,
            payload_json: str,
            replace_existing: bool,
        ) -> bool:
            attached.append(
                (batch_meta, obj_meta, payload_json, replace_existing)
            )
            return True

    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", Native())
    monkeypatch.setenv("NOESIS_POSE_FEATURES_PER_FRAME_MAX", "2")
    processor = hooks.PoseFeatureProcessor(
        pipeline=SimpleNamespace(),
        gie_id=1,
        tensor_source="rfdetr_pgie_frame",
        score_threshold=0.4,
        match_min_iou=0.7,
        match_ambiguity_margin=0.05,
        cache_max_age_frames=0,
    )
    keypoints = np.asarray(
        [[10.0 + index, 20.0 + index, 0.9] for index in range(17)],
        dtype=np.float32,
    )
    processor._extract_rfdetr_frame_native = lambda frame, objects: {
        1: (0.8, keypoints, keypoints),
        2: (0.75, keypoints, keypoints),
    }
    processor._compute_features = lambda keypoints, roi_w, roi_h: (
        {"height_proxy": 1.0},
        (0.9, 0.9, 1.0),
    )

    objects = [
        SimpleNamespace(
            class_id=0,
            object_id=index + 40,
            rect_params=SimpleNamespace(
                left=5.0,
                top=6.0,
                width=100.0,
                height=200.0,
            ),
        )
        for index in range(3)
    ]
    frame = SimpleNamespace(
        object_items=objects,
        source_id=0,
        frame_number=1,
        buf_pts=1_000_000,
    )
    processor.handle_servicemaker_frame(object(), frame)
    assert [row[1].object_id for row in attached] == [41, 42]

    monkeypatch.setattr(
        hooks,
        "noesis_pose_meta_ext",
        SimpleNamespace(attach_pose_features=lambda *args: False),
    )
    with pytest.raises(RuntimeError, match="native metadata attach returned false"):
        processor.handle_servicemaker_frame(object(), frame)


def test_native_source_uses_frame_tensor_meta_and_strict_mutual_match() -> None:
    source = (
        DS9_ROOT / "native" / "noesis_pose_meta_ext.cpp"
    ).read_text(encoding="utf-8")
    for expected in (
        "read_rfdetr_keypoint_tensors",
        "frame_meta.iterate(",
        "NVDSINFER_TENSOR_OUTPUT_META",
        "matching_meta_count != 1U",
        "best_object_for_query != object_index",
        "query_margin < ambiguity_margin",
        "object_margin < ambiguity_margin",
        "kRFDETRPersonKeypointOffset = 17U",
        "width * kRFDETRNetworkWidth < 1.0f",
        "rfdetr_official_fused_person_score",
        "kRFDETRKeypointTraceAlpha = 0.2",
        'payload["base_score"] = query.base_score',
        '"extract_rfdetr_keypoint_matches"',
    ):
        assert expected in source
