from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import model_setup
from wholebody_overlay import (
    Detection,
    DetectionSmoother,
    Deimv2WholebodyOverlay,
    class_group,
    decode_detections,
    refine_detections,
    skeleton_lines,
    summarize_detections,
    write_summary_json,
)


def test_classes_and_model_contract():
    classes = model_setup.load_classes()
    assert len(classes) == 49
    assert classes[0] == "body"
    assert classes[48] == "bone"
    info = model_setup._build_model_info()
    assert set(info["outputs"]) == {"label_xyxy_score", "masks"}
    assert info["model"]["variant"] == "masks"
    assert info["model"]["variant_key"] == "dinov3_s_masks"
    assert info["input"]["onnx_preprocess"] == "ImageNet mean/std normalization baked into graph"
    boxes_info = model_setup._build_model_info("boxes")
    assert set(boxes_info["outputs"]) == {"label_xyxy_score"}
    assert boxes_info["model"]["variant"] == "boxes"
    assert boxes_info["model"]["variant_key"] == "dinov3_s_boxes"
    assert boxes_info["inference"]["engine"].endswith("_boxes_n_batch_ds8norm_b3_fp16.engine")
    x_boxes_info = model_setup._build_model_info("x_boxes")
    assert set(x_boxes_info["outputs"]) == {"label_xyxy_score"}
    assert x_boxes_info["model"]["variant_key"] == "dinov3_x_boxes"
    assert x_boxes_info["inference"]["precision"] == "fp16"
    assert "dinov3_x" in x_boxes_info["inference"]["engine"]
    x_int8_info = model_setup._build_model_info("x_boxes_int8")
    assert set(x_int8_info["outputs"]) == {"label_xyxy_score"}
    assert x_int8_info["model"]["variant_key"] == "dinov3_x_boxes_int8"
    assert x_int8_info["inference"]["precision"] == "int8"
    assert x_int8_info["inference"]["network_mode"] == 1
    assert x_int8_info["inference"]["engine"].endswith("_b3_int8.engine")
    assert x_int8_info["inference"]["int8_requires_representative_calibration"] is True


def test_model_variant_aliases_and_unavailable_l_are_explicit():
    assert model_setup.normalize_variant("boxes") == "dinov3_s_boxes"
    assert model_setup.normalize_variant("x-boxes") == "dinov3_x_boxes"
    assert model_setup.normalize_variant("x-int8") == "dinov3_x_boxes_int8"
    assert model_setup.variant_has_instance_masks("x_masks") is True
    assert model_setup.variant_has_instance_masks("x_boxes") is False
    assert model_setup.variant_has_instance_masks("x_boxes_int8") is False
    with pytest.raises(ValueError, match="DINOv3-L is not present"):
        model_setup.normalize_variant("dinov3_l_boxes")
    with pytest.raises(ValueError, match="DINOv3-L is not present"):
        model_setup.normalize_variant("dinov3_l_boxes_int8")


def test_materialized_nvinfer_config_exposes_all_outputs(tmp_path: Path):
    parser_lib = tmp_path / "libdeimv2_wholebody49_parser.so"
    parser_lib.write_bytes(b"fake")
    config_path = model_setup.materialize_nvinfer_config(
        batch_size=3,
        parser_lib_path=parser_lib,
        score_threshold=0.42,
        mask_threshold=0.55,
        infer_interval=2,
    )
    text = config_path.read_text(encoding="utf-8")
    assert "network-type=3" in text
    assert "output-instance-mask=1" in text
    assert "output-tensor-meta=1" in text
    assert "output-blob-names=label_xyxy_score;masks" in text
    assert "maintain-aspect-ratio=0" in text
    assert "interval=2" in text
    assert "pre-cluster-threshold=0.420000" in text
    assert "segmentation-threshold=0.550000" in text


def test_materialized_boxes_nvinfer_config_disables_masks():
    config_path = model_setup.materialize_nvinfer_config(
        batch_size=3,
        score_threshold=0.42,
        infer_interval=1,
        variant="boxes",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "network-type=100" in text
    assert "output-tensor-meta=1" in text
    assert "output-blob-names=label_xyxy_score" in text
    assert "interval=1" in text
    assert "output-instance-mask" not in text
    assert "parse-bbox-instance-mask-func-name" not in text
    assert "custom-lib-path" not in text
    assert "segmentation-threshold" not in text
    assert "_boxes_n_batch_ds8norm_b3_fp16.engine" in text


def test_materialized_x_boxes_nvinfer_config_uses_x_engine():
    config_path = model_setup.materialize_nvinfer_config(
        batch_size=3,
        score_threshold=0.42,
        variant="x_boxes",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "network-type=100" in text
    assert "output-blob-names=label_xyxy_score" in text
    assert "deimv2_dinov3_x_wholebody49_boxes_n_batch_ds8norm_b3_fp16.engine" in text
    assert "output-instance-mask" not in text


def test_materialized_x_boxes_int8_nvinfer_config_uses_int8_engine():
    config_path = model_setup.materialize_nvinfer_config(
        batch_size=3,
        score_threshold=0.42,
        variant="x_boxes_int8",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "network-type=100" in text
    assert "network-mode=1" in text
    assert "output-blob-names=label_xyxy_score" in text
    assert "deimv2_dinov3_x_wholebody49_boxes_n_batch_ds8norm_b3_int8.engine" in text
    assert "output-instance-mask" not in text


def test_x_boxes_int8_engine_build_requires_calibration_data(monkeypatch, tmp_path: Path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"placeholder")
    monkeypatch.setattr(model_setup, "ensure_variant_onnx", lambda *args, **kwargs: onnx_path)
    monkeypatch.setattr(model_setup, "_engine_is_current", lambda *args, **kwargs: False)

    with pytest.raises(FileNotFoundError, match="no calibration data"):
        model_setup.build_engine(variant="x_boxes_int8", allow_build=True)


def test_wholebody_class_groups_match_upstream_roles():
    assert class_group(23) == "keypoint"
    assert class_group(33) == "object"
    assert class_group(35) == "keypoint"
    assert class_group(46) == "object"
    assert class_group(48) == "bone"


def test_decode_detections_preserves_wholebody_outputs(tmp_path: Path):
    classes = model_setup.load_classes()
    output = np.zeros((11, 6), dtype=np.float32)
    output[:] = [48, 0.0, 0.0, 0.01, 0.01, 0.0]
    output[0] = [0, 0.10, 0.20, 0.50, 0.90, 0.95]
    output[1] = [1, 0.10, 0.20, 0.50, 0.90, 0.80]
    output[2] = [3, 0.10, 0.20, 0.50, 0.90, 0.82]
    output[3] = [7, 0.20, 0.18, 0.34, 0.35, 0.88]
    output[4] = [8, 0.20, 0.18, 0.34, 0.35, 0.81]
    output[5] = [22, 0.22, 0.42, 0.24, 0.44, 0.72]
    output[6] = [23, 0.221, 0.421, 0.241, 0.441, 0.76]
    output[7] = [26, 0.24, 0.55, 0.26, 0.57, 0.70]
    output[8] = [27, 0.241, 0.551, 0.261, 0.571, 0.74]
    output[9] = [29, 0.25, 0.68, 0.27, 0.70, 0.69]
    output[10] = [30, 0.251, 0.681, 0.271, 0.701, 0.73]

    detections = decode_detections(
        output,
        frame_width=1920,
        frame_height=1080,
        class_names=classes,
        object_score_threshold=0.35,
        attribute_score_threshold=0.35,
        keypoint_threshold=0.35,
    )
    body = next(det for det in detections if det.class_id == 0)
    assert body.attributes["generation"] == "adult"
    assert body.attributes["gender"] == "male"
    assert summarize_detections(detections)["mask_capable"] == 1
    assert len(skeleton_lines(detections)) >= 2

    manifest = write_summary_json(
        output_root=tmp_path,
        sensor_id="0",
        sensor_name="Living Room Camera",
        frame_number=12,
        detections=detections,
        model_info=model_setup._build_model_info(),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["summary"]["bodies"] == 1
    assert payload["outputs"]["masks"]["shape"] == ["N", 1240, 80, 80]


def test_decode_detections_can_mark_label_only_variant():
    classes = model_setup.load_classes()
    output = np.asarray([[0, 0.10, 0.20, 0.50, 0.90, 0.95]], dtype=np.float32)
    detections = decode_detections(
        output,
        frame_width=1920,
        frame_height=1080,
        class_names=classes,
        object_score_threshold=0.35,
        attribute_score_threshold=0.35,
        keypoint_threshold=0.35,
        has_instance_masks=False,
    )
    assert summarize_detections(detections)["bodies"] == 1
    assert summarize_detections(detections)["mask_capable"] == 0


def _manual_detection(
    class_id: int,
    bbox: tuple[float, float, float, float],
    source_idx: int,
    score: float = 0.90,
) -> Detection:
    classes = model_setup.load_classes()
    det = Detection(
        class_id=class_id,
        class_name=classes[class_id],
        score=score,
        bbox=bbox,
        source_idx=source_idx,
        group=class_group(class_id),
        has_instance_mask=class_id == 0,
    )
    if "left" in det.class_name:
        det.attributes["side"] = "left"
    if "right" in det.class_name:
        det.attributes["side"] = "right"
    return det


def test_class_aware_filtering_keeps_contextual_keypoints_and_drops_orphans():
    detections = [
        _manual_detection(0, (100, 100, 500, 900), 0, score=0.92),
        _manual_detection(22, (180, 240, 205, 265), 1, score=0.42),
        _manual_detection(23, (900, 240, 925, 265), 2, score=0.42),
        _manual_detection(1, (700, 700, 900, 900), 3, score=0.90),
        _manual_detection(7, (120, 105, 220, 220), 4, score=0.58),
    ]

    refined = refine_detections(
        detections,
        object_score_threshold=0.50,
        attribute_score_threshold=0.75,
        keypoint_threshold=0.50,
        keypoint_candidate_threshold=0.40,
    )

    source_ids = {det.source_idx for det in refined}
    assert 0 in source_ids
    assert 1 in source_ids
    assert 4 in source_ids
    assert 2 not in source_ids
    assert 3 not in source_ids


def test_detection_smoother_reuses_same_class_tracks_without_cross_class_matching():
    smoother = DetectionSmoother(alpha=0.50)
    first = smoother.update(0, [_manual_detection(0, (10, 20, 110, 220), 0)])
    second = smoother.update(0, [_manual_detection(0, (20, 20, 120, 220), 1)])
    third = smoother.update(0, [_manual_detection(7, (20, 20, 120, 220), 2)])

    assert first[0].track_id == second[0].track_id
    assert second[0].bbox == (15.0, 20.0, 115.0, 220.0)
    assert third[0].track_id != second[0].track_id


def test_overlay_reuses_cached_detections_when_tensor_is_missing(tmp_path: Path):
    class FrameMeta:
        source_id = 0
        tensor_items: list[object] = []

    cached = _manual_detection(0, (10, 20, 110, 220), 0)
    cached.track_id = 7
    overlay = Deimv2WholebodyOverlay(
        output_root=tmp_path,
        model_info=model_setup._build_model_info("x_boxes"),
        sensor_ids=["0"],
        sensor_names=["source"],
        has_instance_masks=False,
        reuse_last_on_missing_tensor=True,
    )
    overlay._last_detections_by_source[0] = [cached]

    detections, reused = overlay._decode_frame_detections(
        FrameMeta(),
        frame_w=1920,
        frame_h=1080,
        source_id=0,
    )

    assert reused is True
    assert len(detections) == 1
    assert detections[0].track_id == 7
    assert detections[0].bbox == cached.bbox
    assert overlay._tensor_cache_reuse_counts[0] == 1


def test_overlay_missing_tensor_still_errors_without_cache(tmp_path: Path):
    class FrameMeta:
        source_id = 0
        tensor_items: list[object] = []

    overlay = Deimv2WholebodyOverlay(
        output_root=tmp_path,
        model_info=model_setup._build_model_info("x_boxes"),
        sensor_ids=["0"],
        sensor_names=["source"],
        has_instance_masks=False,
        reuse_last_on_missing_tensor=False,
    )

    with pytest.raises(RuntimeError, match="No tensor output found"):
        overlay._decode_frame_detections(
            FrameMeta(),
            frame_w=1920,
            frame_h=1080,
            source_id=0,
        )


def test_skeleton_lines_keep_body_instances_separate():
    detections = [
        _manual_detection(0, (0, 0, 110, 210), 0),
        _manual_detection(0, (200, 0, 310, 210), 1),
        _manual_detection(22, (30, 35, 45, 50), 2),
        _manual_detection(26, (35, 80, 50, 95), 3),
        _manual_detection(29, (40, 125, 55, 140), 4),
        _manual_detection(22, (235, 35, 250, 50), 5),
        _manual_detection(26, (240, 80, 255, 95), 6),
        _manual_detection(29, (245, 125, 260, 140), 7),
    ]
    for det in detections:
        if det.class_id in (22, 26, 29):
            det.attributes["side"] = "left"

    lines = skeleton_lines(detections)
    assert lines
    for start, end in lines:
        assert (start.center[0] < 150 and end.center[0] < 150) or (
            start.center[0] > 150 and end.center[0] > 150
        )


def test_bone_boxes_support_side_specific_edges():
    detections = [
        _manual_detection(0, (0, 0, 140, 220), 0),
        _manual_detection(48, (18, 28, 58, 105), 1),
        _manual_detection(23, (24, 36, 38, 50), 2),
        _manual_detection(27, (36, 82, 50, 96), 3),
    ]

    class_pairs = {tuple(sorted((start.class_id, end.class_id))) for start, end in skeleton_lines(detections)}
    assert (23, 27) in class_pairs


def test_tensorrt_engine_contract_if_present():
    if not model_setup.ENGINE_PATH.exists():
        pytest.skip(f"TensorRT engine not present: {model_setup.ENGINE_PATH}")
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(model_setup.ENGINE_PATH.read_bytes())
    assert engine is not None
    shapes = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shapes[name] = tuple(engine.get_tensor_shape(name))
    assert shapes["images"] == (-1, 3, 640, 640)
    assert shapes["label_xyxy_score"] == (-1, 1240, 6)
    assert shapes["masks"] == (-1, 1240, 80, 80)


def test_boxes_tensorrt_engine_contract_if_present():
    if not model_setup.BOXES_ENGINE_PATH.exists():
        pytest.skip(f"TensorRT boxes engine not present: {model_setup.BOXES_ENGINE_PATH}")
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(model_setup.BOXES_ENGINE_PATH.read_bytes())
    assert engine is not None
    shapes = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shapes[name] = tuple(engine.get_tensor_shape(name))
    assert shapes["images"] == (-1, 3, 640, 640)
    assert shapes["label_xyxy_score"] == (-1, 1240, 6)
    assert "masks" not in shapes


def test_x_boxes_tensorrt_engine_contract_if_present():
    if not model_setup.X_BOXES_ENGINE_PATH.exists():
        pytest.skip(f"TensorRT X boxes engine not present: {model_setup.X_BOXES_ENGINE_PATH}")
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(model_setup.X_BOXES_ENGINE_PATH.read_bytes())
    assert engine is not None
    shapes = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shapes[name] = tuple(engine.get_tensor_shape(name))
    assert shapes["images"] == (-1, 3, 640, 640)
    assert shapes["label_xyxy_score"] == (-1, 1240, 6)
    assert "masks" not in shapes


def test_x_boxes_int8_tensorrt_engine_contract_if_present():
    if not model_setup.X_BOXES_INT8_ENGINE_PATH.exists():
        pytest.skip(f"TensorRT X boxes INT8 engine not present: {model_setup.X_BOXES_INT8_ENGINE_PATH}")
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(model_setup.X_BOXES_INT8_ENGINE_PATH.read_bytes())
    assert engine is not None
    shapes = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shapes[name] = tuple(engine.get_tensor_shape(name))
    assert shapes["images"] == (-1, 3, 640, 640)
    assert shapes["label_xyxy_score"] == (-1, 1240, 6)
    assert "masks" not in shapes
