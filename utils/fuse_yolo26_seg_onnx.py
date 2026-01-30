#!/usr/bin/env python3
"""
Fuse YOLO26-seg ONNX (output0 + proto output1) into a single-output ONNX for DeepStream.

Input model expectations:
  - input:  images [3, 3, 640, 640] (static b3)
  - output0 [3, 300, 38] = [x1,y1,x2,y2,score,class, 32 mask coeffs]
  - output1 [3, 32, 160, 160] = proto features (net/4)

Output model:
  - output0 [3, K, 6 + (mask_side*mask_side)] where K = --max-det
      = [x1,y1,x2,y2,score,class, mask_flat...]
  - output1 removed from graph outputs (still used internally).

Mask compose strategy (GPU-friendly, small output):
  ROIAlign(proto, boxes, batch_indices) -> [B*K, 32, S, S]
  MatMul(coeffs, roi_feat_flat) -> [B*K, S*S]
  Sigmoid -> mask probabilities
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _find_value_producer(model: onnx.ModelProto, value_name: str) -> onnx.NodeProto | None:
    for node in model.graph.node:
        if value_name in node.output:
            return node
    return None


def _rename_node_output(model: onnx.ModelProto, old: str, new: str) -> None:
    node = _find_value_producer(model, old)
    if node is None:
        raise SystemExit(f"[FATAL] Unable to find producer of tensor: {old}")
    node.output[:] = [new if o == old else o for o in node.output]
    # Also update any consumer inputs (paranoia: should be none for output0 in the source model).
    for n in model.graph.node:
        n.input[:] = [new if i == old else i for i in n.input]


def _make_i64_init(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _make_i64_scalar(name: str, value: int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def _shape_from_value_info(vi: onnx.ValueInfoProto) -> list[int | str]:
    tt = vi.type.tensor_type
    out: list[int | str] = []
    for d in tt.shape.dim:
        if d.dim_value:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def fuse_yolo26_seg(in_path: Path, out_path: Path, *, mask_side: int, max_det: int) -> None:
    if mask_side <= 0:
        raise SystemExit(f"[FATAL] --mask-side must be > 0 (got {mask_side})")
    if max_det <= 0:
        raise SystemExit(f"[FATAL] --max-det must be > 0 (got {max_det})")

    model = onnx.load(str(in_path))

    # Validate expected outputs.
    outputs = {o.name: o for o in model.graph.output}
    if "output0" not in outputs or "output1" not in outputs:
        raise SystemExit(
            f"[FATAL] Expected graph outputs 'output0' and 'output1'. Got: {list(outputs.keys())}"
        )

    out0_shape = _shape_from_value_info(outputs["output0"])
    out1_shape = _shape_from_value_info(outputs["output1"])
    if out0_shape != [3, 300, 38]:
        raise SystemExit(f"[FATAL] Unexpected output0 shape: {out0_shape} (expected [3,300,38])")
    if out1_shape != [3, 32, 160, 160]:
        raise SystemExit(f"[FATAL] Unexpected output1 shape: {out1_shape} (expected [3,32,160,160])")

    b = 3
    k_in = 300
    if max_det > k_in:
        raise SystemExit(f"[FATAL] --max-det must be <= {k_in} (got {max_det})")
    k = int(max_det)
    coeff_dim = 32
    mask_len = mask_side * mask_side

    # Rename original output0 so we can emit a new tensor named output0.
    _rename_node_output(model, "output0", "output0_raw")

    # Slice detections dimension to max_det: output0_raw[:, 0:max_det, :].
    # This reduces the ROIAlign workload and output tensor size.

    # Build batch indices [0..B-1] repeated K times.
    batch_indices = []
    for bi in range(b):
        batch_indices.extend([bi] * k)

    # Initializers used by slicing/reshaping.
    inits: list[onnx.TensorProto] = [
        _make_i64_init("starts_det_0", [0]),
        _make_i64_init("ends_det_k", [k]),
        _make_i64_init("axes_det", [1]),
        _make_i64_init("starts_0_6", [0]),
        _make_i64_init("ends_0_6", [6]),
        _make_i64_init("starts_0_4", [0]),
        _make_i64_init("ends_0_4", [4]),
        _make_i64_init("starts_6_38", [6]),
        _make_i64_init("ends_6_38", [38]),
        _make_i64_init("axes_last", [2]),
        _make_i64_init("steps_1", [1]),
        _make_i64_init("shape_boxes_rois", [-1, 4]),
        _make_i64_init("shape_coeffs", [-1, coeff_dim]),
        _make_i64_init("shape_coeffs_matmul", [-1, 1, coeff_dim]),
        _make_i64_init("shape_roi_feat_flat", [-1, coeff_dim, mask_len]),
        _make_i64_init("shape_mask_flat", [-1, mask_len]),
        _make_i64_init("shape_mask_bk", [b, k, mask_len]),
        numpy_helper.from_array(np.asarray(batch_indices, dtype=np.int64), name="batch_indices"),
    ]

    # Nodes.
    nodes: list[onnx.NodeProto] = []

    # out0_topk = output0_raw[:, 0:max_det, :]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["output0_raw", "starts_det_0", "ends_det_k", "axes_det", "steps_1"],
            outputs=["output0_topk"],
            name="slice_det_topk",
        )
    )

    # fields6 = output0_topk[:, :, 0:6]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["output0_topk", "starts_0_6", "ends_0_6", "axes_last", "steps_1"],
            outputs=["fields6"],
            name="slice_fields6",
        )
    )

    # boxes = output0_topk[:, :, 0:4]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["output0_topk", "starts_0_4", "ends_0_4", "axes_last", "steps_1"],
            outputs=["boxes_bk4"],
            name="slice_boxes",
        )
    )

    # coeffs = output0_topk[:, :, 6:38]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["output0_topk", "starts_6_38", "ends_6_38", "axes_last", "steps_1"],
            outputs=["coeffs_bk32"],
            name="slice_coeffs",
        )
    )

    # rois = reshape(boxes) -> [B*K,4]
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["boxes_bk4", "shape_boxes_rois"],
            outputs=["rois"],
            name="reshape_rois",
        )
    )

    # coeffs_flat = reshape(coeffs) -> [B*K,32]
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["coeffs_bk32", "shape_coeffs"],
            outputs=["coeffs_flat"],
            name="reshape_coeffs_flat",
        )
    )

    # coeffs_mat = reshape(coeffs_flat) -> [B*K,1,32]
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["coeffs_flat", "shape_coeffs_matmul"],
            outputs=["coeffs_mat"],
            name="reshape_coeffs_mat",
        )
    )

    # roi_feat = ROIAlign(proto, rois, batch_indices) -> [B*K,32,S,S]
    nodes.append(
        helper.make_node(
            "RoiAlign",
            inputs=["output1", "rois", "batch_indices"],
            outputs=["roi_feat"],
            coordinate_transformation_mode="half_pixel",
            mode="avg",
            output_height=mask_side,
            output_width=mask_side,
            sampling_ratio=0,
            spatial_scale=0.25,
            name="roi_align_proto",
        )
    )

    # roi_feat_flat = reshape -> [B*K,32,S*S]
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["roi_feat", "shape_roi_feat_flat"],
            outputs=["roi_feat_flat"],
            name="reshape_roi_feat_flat",
        )
    )

    # mask_logits = MatMul([B*K,1,32], [B*K,32,S*S]) -> [B*K,1,S*S]
    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["coeffs_mat", "roi_feat_flat"],
            outputs=["mask_logits_1"],
            name="matmul_coeffs_roi",
        )
    )

    # mask_logits = reshape -> [B*K,S*S]
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["mask_logits_1", "shape_mask_flat"],
            outputs=["mask_logits"],
            name="reshape_mask_logits",
        )
    )

    # mask = sigmoid(mask_logits)
    nodes.append(
        helper.make_node(
            "Sigmoid",
            inputs=["mask_logits"],
            outputs=["mask_probs"],
            name="sigmoid_mask",
        )
    )

    # mask_bk = reshape -> [B,K,S*S]
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=["mask_probs", "shape_mask_bk"],
            outputs=["mask_bk"],
            name="reshape_mask_bk",
        )
    )

    # output0 = concat(fields6, mask_bk) axis=2
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=["fields6", "mask_bk"],
            outputs=["output0"],
            axis=2,
            name="concat_output0_fused",
        )
    )

    # Append fusion nodes + initializers.
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(inits)

    # Replace graph outputs: output0 only.
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                "output0",
                TensorProto.FLOAT,
                [b, k, 6 + mask_len],
            )
        ]
    )

    # Remove stale value_info entries if they exist (best-effort; not required for validity).
    # Ensure the renamed raw output isn't declared as a graph output anymore.
    for vi in list(model.graph.value_info):
        if vi.name == "output0":
            model.graph.value_info.remove(vi)

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse YOLO26-seg ONNX into output0-only ONNX")
    p.add_argument("--in", dest="in_path", required=True, help="Input ONNX (yolo26*-seg.onnx)")
    p.add_argument("--out", dest="out_path", required=True, help="Output ONNX (fused)")
    p.add_argument(
        "--mask-side",
        type=int,
        default=64,
        help="Mask side length S for per-box masks (SxS). Default: 64",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=30,
        help="Max detections K to keep from the model output (first K rows). Default: 30",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fuse_yolo26_seg(
        Path(args.in_path),
        Path(args.out_path),
        mask_side=int(args.mask_side),
        max_det=int(args.max_det),
    )
    print(f"[OK] wrote: {args.out_path}")


if __name__ == "__main__":
    main()
