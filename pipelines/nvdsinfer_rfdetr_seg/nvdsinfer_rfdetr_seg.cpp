#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

template <typename T>
inline T clamp(T v, T lo, T hi) {
  return std::min(hi, std::max(lo, v));
}

inline float sigmoid(float x) {
  // Numerically stable sigmoid.
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

int getenv_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  try {
    return std::stoi(std::string(raw));
  } catch (...) {
    return fallback;
  }
}

bool validate_float_layer(const NvDsInferLayerInfo& layer, const char* label) {
  if (!layer.buffer) {
    std::cerr << "[rfdetr-seg] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[rfdetr-seg] " << label << " must be FLOAT" << std::endl;
    return false;
  }
  return true;
}

struct ParsedLayerDims {
  std::size_t q = 0;
  std::size_t c = 0;
  std::size_t h = 0;
  std::size_t w = 0;
};

bool parse_boxes_dims(const NvDsInferLayerInfo& boxes, ParsedLayerDims& out) {
  const int nd = boxes.inferDims.numDims;
  if (nd == 2) {
    // [Q, 4]
    if (boxes.inferDims.d[1] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[0]);
    return true;
  }
  if (nd == 3) {
    // [B, Q, 4] (B should be 1 for per-batch parsing)
    if (boxes.inferDims.d[2] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[1]);
    return true;
  }
  return false;
}

bool parse_logits_dims(const NvDsInferLayerInfo& logits, ParsedLayerDims& out) {
  const int nd = logits.inferDims.numDims;
  if (nd == 2) {
    // [Q, C]
    out.q = static_cast<std::size_t>(logits.inferDims.d[0]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[1]);
    return out.c > 0;
  }
  if (nd == 3) {
    // [B, Q, C] (B should be 1)
    out.q = static_cast<std::size_t>(logits.inferDims.d[1]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[2]);
    return out.c > 0;
  }
  return false;
}

bool parse_masks_dims(const NvDsInferLayerInfo& masks, ParsedLayerDims& out) {
  const int nd = masks.inferDims.numDims;
  if (nd == 3) {
    // [Q, Hm, Wm]
    out.q = static_cast<std::size_t>(masks.inferDims.d[0]);
    out.h = static_cast<std::size_t>(masks.inferDims.d[1]);
    out.w = static_cast<std::size_t>(masks.inferDims.d[2]);
    return out.h > 0 && out.w > 0;
  }
  if (nd == 4) {
    // [B, Q, Hm, Wm] (B should be 1)
    out.q = static_cast<std::size_t>(masks.inferDims.d[1]);
    out.h = static_cast<std::size_t>(masks.inferDims.d[2]);
    out.w = static_cast<std::size_t>(masks.inferDims.d[3]);
    return out.h > 0 && out.w > 0;
  }
  return false;
}

const NvDsInferLayerInfo* find_layer_by_name(const std::vector<NvDsInferLayerInfo>& layers,
                                             const char* name) {
  if (!name || !*name) return nullptr;
  for (const auto& layer : layers) {
    if (!layer.layerName) continue;
    if (std::strcmp(layer.layerName, name) == 0) return &layer;
  }
  return nullptr;
}

void set_bbox(float x1, float y1, float x2, float y2, unsigned net_w, unsigned net_h,
              NvDsInferInstanceMaskInfo& obj) {
  x1 = clamp(x1, 0.0f, static_cast<float>(net_w));
  y1 = clamp(y1, 0.0f, static_cast<float>(net_h));
  x2 = clamp(x2, 0.0f, static_cast<float>(net_w));
  y2 = clamp(y2, 0.0f, static_cast<float>(net_h));

  obj.left = x1;
  obj.top = y1;
  obj.width = clamp(x2 - x1, 0.0f, static_cast<float>(net_w));
  obj.height = clamp(y2 - y1, 0.0f, static_cast<float>(net_h));
}

bool copy_mask_roi(const float* mask_src, std::size_t mask_h, std::size_t mask_w,
                   float x1, float y1, float x2, float y2, unsigned net_w, unsigned net_h,
                   NvDsInferInstanceMaskInfo& obj) {
  if (!mask_src || mask_h == 0 || mask_w == 0 || net_w == 0 || net_h == 0) return false;

  const float fx1 = (x1 / static_cast<float>(net_w)) * static_cast<float>(mask_w);
  const float fy1 = (y1 / static_cast<float>(net_h)) * static_cast<float>(mask_h);
  const float fx2 = (x2 / static_cast<float>(net_w)) * static_cast<float>(mask_w);
  const float fy2 = (y2 / static_cast<float>(net_h)) * static_cast<float>(mask_h);

  const int ix1 = clamp(static_cast<int>(std::floor(fx1)), 0, static_cast<int>(mask_w) - 1);
  const int iy1 = clamp(static_cast<int>(std::floor(fy1)), 0, static_cast<int>(mask_h) - 1);
  const int ix2 = clamp(static_cast<int>(std::ceil(fx2)), ix1 + 1, static_cast<int>(mask_w));
  const int iy2 = clamp(static_cast<int>(std::ceil(fy2)), iy1 + 1, static_cast<int>(mask_h));

  const std::size_t roi_w = static_cast<std::size_t>(ix2 - ix1);
  const std::size_t roi_h = static_cast<std::size_t>(iy2 - iy1);
  if (roi_w == 0 || roi_h == 0) return false;

  obj.mask_width = static_cast<unsigned>(roi_w);
  obj.mask_height = static_cast<unsigned>(roi_h);
  obj.mask_size = static_cast<unsigned>(roi_w * roi_h * sizeof(float));
  obj.mask = new float[roi_w * roi_h];

  for (std::size_t y = 0; y < roi_h; ++y) {
    const float* src_row =
        mask_src + (static_cast<std::size_t>(iy1) + y) * mask_w + static_cast<std::size_t>(ix1);
    float* dst_row = obj.mask + y * roi_w;
    for (std::size_t x = 0; x < roi_w; ++x) {
      dst_row[x] = sigmoid(src_row[x]);
    }
  }

  return true;
}

}  // namespace

extern "C" bool NvDsInferParseRFDETRSeg(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  if (outputLayersInfo.size() < 3) {
    std::cerr << "[rfdetr-seg] expected >=3 output layers (boxes, logits, masks), got "
              << outputLayersInfo.size() << std::endl;
    return false;
  }

  // Read outputs by name. Require output-blob-names in the PGIE INI to include:
  //   dets   -> boxes [Q,4] or [B,Q,4] (normalized cxcywh)
  //   labels -> logits [Q,C] or [B,Q,C] (RF-DETR uses sigmoid scores; COCO ids are 1..90)
  //   masks  -> mask logits [Q,Hm,Wm] or [B,Q,Hm,Wm]
  const NvDsInferLayerInfo* boxes_layer = find_layer_by_name(outputLayersInfo, "dets");
  const NvDsInferLayerInfo* logits_layer = find_layer_by_name(outputLayersInfo, "labels");
  const NvDsInferLayerInfo* masks_layer = find_layer_by_name(outputLayersInfo, "masks");
  if (!boxes_layer || !logits_layer || !masks_layer) {
    std::cerr << "[rfdetr-seg] missing required output layers. Found:";
    for (const auto& layer : outputLayersInfo) {
      std::cerr << " " << (layer.layerName ? layer.layerName : "<null>");
    }
    std::cerr << std::endl;
    return false;
  }

  if (!validate_float_layer(*boxes_layer, "dets")) return false;
  if (!validate_float_layer(*logits_layer, "labels")) return false;
  if (!validate_float_layer(*masks_layer, "masks")) return false;

  ParsedLayerDims boxes_dims{};
  ParsedLayerDims logits_dims{};
  ParsedLayerDims masks_dims{};
  if (!parse_boxes_dims(*boxes_layer, boxes_dims) || boxes_dims.q == 0) {
    std::cerr << "[rfdetr-seg] unexpected boxes dims (expected [Q,4] or [1,Q,4])" << std::endl;
    return false;
  }
  if (!parse_logits_dims(*logits_layer, logits_dims) || logits_dims.q == 0 || logits_dims.c == 0) {
    std::cerr << "[rfdetr-seg] unexpected logits dims (expected [Q,C] or [1,Q,C])" << std::endl;
    return false;
  }
  if (!parse_masks_dims(*masks_layer, masks_dims) || masks_dims.q == 0 || masks_dims.h == 0 ||
      masks_dims.w == 0) {
    std::cerr << "[rfdetr-seg] unexpected masks dims (expected [Q,H,W] or [1,Q,H,W])" << std::endl;
    return false;
  }

  const std::size_t q = logits_dims.q;
  const std::size_t c = logits_dims.c;
  if (boxes_dims.q != q || masks_dims.q != q) {
    std::cerr << "[rfdetr-seg] Q mismatch: boxes.q=" << boxes_dims.q << " logits.q=" << q
              << " masks.q=" << masks_dims.q << std::endl;
    return false;
  }

  // RF-DETR (coco) uses COCO category IDs as class indices (max_obj_id=90),
  // so "person" is typically class index 1 (COCO category id 1).
  const int person_class_idx = getenv_int("NOESIS_RFDETR_PERSON_CLASS_IDX", 1);
  if (person_class_idx < 0 || static_cast<std::size_t>(person_class_idx) >= c) {
    std::cerr << "[rfdetr-seg] invalid NOESIS_RFDETR_PERSON_CLASS_IDX=" << person_class_idx
              << " (C=" << c << ")" << std::endl;
    return false;
  }

  const float threshold =
      !detectionParams.perClassPreclusterThreshold.empty() ? detectionParams.perClassPreclusterThreshold[0] : 0.0f;

  const auto* boxes = static_cast<const float*>(boxes_layer->buffer);
  const auto* logits = static_cast<const float*>(logits_layer->buffer);
  const auto* masks = static_cast<const float*>(masks_layer->buffer);

  const std::size_t box_stride = 4;
  const std::size_t logit_stride = c;
  const std::size_t mask_stride = masks_dims.h * masks_dims.w;

  objectList.clear();
  objectList.reserve(q);

  for (std::size_t i = 0; i < q; ++i) {
    // RF-DETR uses sigmoid probabilities (per-class, not softmax). For DS8 parity
    // we only emit "person" detections as DS classId 0.
    const float score = sigmoid(logits[i * logit_stride + static_cast<std::size_t>(person_class_idx)]);
    if (score < threshold) continue;

    const float cx = boxes[i * box_stride + 0];
    const float cy = boxes[i * box_stride + 1];
    const float bw = boxes[i * box_stride + 2];
    const float bh = boxes[i * box_stride + 3];

    const float x1 = (cx - bw * 0.5f) * static_cast<float>(networkInfo.width);
    const float y1 = (cy - bh * 0.5f) * static_cast<float>(networkInfo.height);
    const float x2 = (cx + bw * 0.5f) * static_cast<float>(networkInfo.width);
    const float y2 = (cy + bh * 0.5f) * static_cast<float>(networkInfo.height);

    NvDsInferInstanceMaskInfo obj{};
    set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj);
    if (obj.width < 1.0f || obj.height < 1.0f) continue;

    obj.classId = 0;  // DS "person"
    obj.detectionConfidence = score;

    const float* mask_src = masks + i * mask_stride;
    if (!copy_mask_roi(mask_src, masks_dims.h, masks_dims.w, obj.left, obj.top,
                       obj.left + obj.width, obj.top + obj.height, networkInfo.width,
                       networkInfo.height, obj)) {
      continue;
    }

    objectList.emplace_back(obj);
  }

  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseRFDETRSeg);
