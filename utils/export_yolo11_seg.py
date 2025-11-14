# export_yolo11_seg.py — coeffs+proto exporter for DeepStream (no in-graph mask compose)

import os, sys, onnx, torch, torch.nn as nn

from copy import deepcopy

from ultralytics import YOLO

from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder, Segment

import ultralytics.utils, ultralytics.models.yolo as yolo_mod

sys.modules["ultralytics.yolo"] = yolo_mod

sys.modules["ultralytics.yolo.utils"] = ultralytics.utils



def force_onnx_export_mode(model):

    model.export = True; model.eval(); model.training = False

    for _, m in model.named_modules():

        setattr(m, "export", True); setattr(m, "format", "onnx")

        for attr in ("forward_export","_forward_onnx","forward_onnx","export_forward"):

            if hasattr(m, attr): m.forward = getattr(m, attr); break

        if isinstance(m, (Detect, Segment)):

            try: m.export = True

            except: pass



class NMS(torch.autograd.Function):

    @staticmethod

    def forward(self, boxes, scores, score_thr, iou_thr, max_out):

        B, N, C = scores.shape

        K = max_out

        num = torch.full((B,1), K, dtype=torch.int32)

        db = torch.randn(B, K, 4, device=boxes.device, dtype=boxes.dtype)

        ds = torch.rand(B, K, device=boxes.device, dtype=boxes.dtype)

        dc = torch.randint(0, C, (B, K), device=boxes.device, dtype=torch.int32)

        return num, db, ds, dc

    @staticmethod

    def symbolic(g, boxes, scores, score_thr, iou_thr, max_out):

        return g.op("TRT::EfficientNMS_TRT", boxes, scores,
                    score_threshold_f=score_thr, iou_threshold_f=iou_thr,
                    max_output_boxes_i=max_out, background_class_i=-1,
                    score_activation_i=0, class_agnostic_i=0,
                    box_coding_i=0, outputs=4)



class DeepStreamOutputs(nn.Module):

    """Keep NMS boxes/scores/classes and output raw coeffs + proto"""

    def __init__(self, nc, conf_thr, iou_thr, max_det):

        super().__init__()

        self.nc, self.conf, self.iou, self.max_det = nc, conf_thr, iou_thr, max_det

    def forward(self, x):

        # x[0]: [B, M, N] where N = 4 + C + P ; x[1]: proto [B, P, H, W]

        preds = x[0].transpose(1, 2)

        boxes  = preds[:, :, :4]

        scores = preds[:, :, 4:self.nc+4]

        coeffs = preds[:, :, self.nc+4:]   # P coeffs

        protos = x[1]                      # [B, P, H, W]

        num, det_boxes, det_scores, det_classes = NMS.apply(

            boxes, scores, self.conf, self.iou, self.max_det

        )

        # No in-graph compose; return tensors needed for GPU-side compose

        return det_boxes, det_scores, det_classes, coeffs, protos, boxes



def yolo11_seg_export(weights, device, fuse=True):

    model = YOLO(weights); model = deepcopy(model.model).to(device)

    for p in model.parameters(): p.requires_grad = False

    model.eval().float()

    if fuse: model = model.fuse()

    for _, m in model.named_modules():

        setattr(m, "export", True); setattr(m, "format", "onnx")

        if isinstance(m, (Detect, RTDETRDecoder, Segment)):

            m.dynamic = False; m.export = True; m.format = "onnx"

        elif isinstance(m, C2f):

            m.forward = m.forward_split

    return model



def suppress_warnings():

    import warnings

    for w in (torch.jit.TracerWarning, UserWarning, DeprecationWarning, FutureWarning, ResourceWarning):

        warnings.filterwarnings("ignore", category=w)



def main(args):

    suppress_warnings()

    print(f"\nStarting: {args.weights}\nOpening YOLO11-Seg model")

    device = torch.device("cpu")

    model = yolo11_seg_export(args.weights, device)

    force_onnx_export_mode(model)

    if hasattr(model, "names") and len(getattr(model, "names", {})) > 0:

        print("Creating labels.txt file")

        with open("labels.txt","w",encoding="utf-8") as f:

            for name in model.names.values(): f.write(f"{name}\n")

    model = nn.Sequential(model, DeepStreamOutputs(len(model.names), args.conf_threshold, args.iou_threshold, args.max_detections))

    img_size = args.size if len(args.size)==2 else [args.size[0], args.size[0]]

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)

    out = args.out or "models/yolo11m-seg_cust.onnx"

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print("Exporting the model to ONNX")

    torch.onnx.export(

        model, onnx_input_im, out, verbose=False, opset_version=args.opset, do_constant_folding=True,

        input_names=["images"], output_names=["boxes","scores","classes","coeffs","proto","pre_boxes"],

        dynamic_axes={"images":{0:"batch"}, "boxes":{0:"batch"},

                      "scores":{0:"batch"}, "classes":{0:"batch"},

                      "coeffs":{0:"batch"}, "proto":{0:"batch"},

                      "pre_boxes":{0:"batch"}} if args.dynamic else None,

        training=torch.onnx.TrainingMode.EVAL, keep_initializers_as_inputs=False

    )

    if args.simplify:

        print("Simplifying ONNX")

        import onnxslim

        m = onnx.load(out); m = onnxslim.slim(m); onnx.save(m, out)

    print(f"Done: {out}\n")



def parse_args():

    import argparse; p = argparse.ArgumentParser("DeepStream YOLO11-Seg export (coeffs+proto)")

    p.add_argument("-w","--weights", required=True, type=str)

    p.add_argument("-s","--size", nargs="+", type=int, default=[640])

    p.add_argument("--img", nargs="+", type=int)

    p.add_argument("--opset", type=int, default=18)

    p.add_argument("--simplify", action="store_true")

    p.add_argument("--dynamic", action="store_true")

    p.add_argument("--batch", type=int, default=1)

    p.add_argument("--conf-threshold", type=float, default=0.25)

    p.add_argument("--iou-threshold",  type=float, default=0.45)

    p.add_argument("--max-detections", type=int,   default=30)

    p.add_argument("--out", type=str, default=None)

    a = p.parse_args()

    if a.img: a.size = a.img

    if not os.path.isfile(a.weights): raise SystemExit("Invalid weights file")

    if a.dynamic and a.batch>1: raise SystemExit("dynamic batch conflicts with static --batch")

    return a



if __name__ == "__main__":

    main(parse_args())