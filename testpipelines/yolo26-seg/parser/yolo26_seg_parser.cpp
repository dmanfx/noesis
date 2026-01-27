// SPDX-FileCopyrightText: 2026 Noesis
// SPDX-License-Identifier: MIT
//
// DeepStream 8 custom parser for YOLO26-Seg (end-to-end output0 + prototype output1)
//
// output0 layout per detection row:
//   [x1, y1, x2, y2, score, class_id, mask_coeffs...]
// output1 layout:
//   [mask_dim, mask_h, mask_w] (or [B, mask_dim, mask_h, mask_w])

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

template <typename T>
inline T clamp(T v, T lo, T hi) {
  return std::min(hi, std::max(lo, v));
}

inline float sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
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

bool validate_float_layer(const NvDsInferLayerInfo& layer, const char* label) {
  if (!layer.buffer) {
    std::cerr << "[yolo26-seg] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[yolo26-seg] " << label << " must be FLOAT" << std::endl;
    return false;
  }
  return true;
}

struct DetDims {
  std::size_t num = 0;
  std::size_t channels = 0;
};

struct ProtoDims {
  std::size_t c = 0;
  std::size_t h = 0;
  std::size_t w = 0;
};

bool parse_det_dims(const NvDsInferLayerInfo& dets, DetDims& out) {
  const int nd = dets.inferDims.numDims;
  if (nd == 2) {
    out.num = static_cast<std::size_t>(dets.inferDims.d[0]);
    out.channels = static_cast<std::size_t>(dets.inferDims.d[1]);
    return out.num > 0 && out.channels > 0;
  }
  if (nd == 3) {
    out.num = static_cast<std::size_t>(dets.inferDims.d[1]);
    out.channels = static_cast<std::size_t>(dets.inferDims.d[2]);
    return out.num > 0 && out.channels > 0;
  }
  return false;
}

bool parse_proto_dims(const NvDsInferLayerInfo& proto, ProtoDims& out) {
  const int nd = proto.inferDims.numDims;
  if (nd == 3) {
    out.c = static_cast<std::size_t>(proto.inferDims.d[0]);
    out.h = static_cast<std::size_t>(proto.inferDims.d[1]);
    out.w = static_cast<std::size_t>(proto.inferDims.d[2]);
    return out.c > 0 && out.h > 0 && out.w > 0;
  }
  if (nd == 4) {
    out.c = static_cast<std::size_t>(proto.inferDims.d[1]);
    out.h = static_cast<std::size_t>(proto.inferDims.d[2]);
    out.w = static_cast<std::size_t>(proto.inferDims.d[3]);
    return out.c > 0 && out.h > 0 && out.w > 0;
  }
  return false;
}

bool set_bbox(float x1, float y1, float x2, float y2, unsigned net_w, unsigned net_h,
              NvDsInferInstanceMaskInfo& obj) {
  x1 = clamp(x1, 0.0f, static_cast<float>(net_w));
  y1 = clamp(y1, 0.0f, static_cast<float>(net_h));
  x2 = clamp(x2, 0.0f, static_cast<float>(net_w));
  y2 = clamp(y2, 0.0f, static_cast<float>(net_h));

  obj.left = x1;
  obj.top = y1;
  obj.width = clamp(x2 - x1, 0.0f, static_cast<float>(net_w));
  obj.height = clamp(y2 - y1, 0.0f, static_cast<float>(net_h));
  return obj.width >= 1.0f && obj.height >= 1.0f;
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

extern "C" bool NvDsInferParseYolo26Seg(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  const NvDsInferLayerInfo* det_layer = find_layer_by_name(outputLayersInfo, "output0");
  const NvDsInferLayerInfo* proto_layer = find_layer_by_name(outputLayersInfo, "output1");
  if (!det_layer || !proto_layer) {
    if (outputLayersInfo.size() >= 2) {
      det_layer = &outputLayersInfo[0];
      proto_layer = &outputLayersInfo[1];
    } else {
      std::cerr << "[yolo26-seg] expected 2 output layers, got " << outputLayersInfo.size() << std::endl;
      return false;
    }
  }

  if (!validate_float_layer(*det_layer, "output0")) return false;
  if (!validate_float_layer(*proto_layer, "output1")) return false;

  DetDims det_dims{};
  ProtoDims proto_dims{};
  if (!parse_det_dims(*det_layer, det_dims)) {
    std::cerr << "[yolo26-seg] unexpected output0 dims" << std::endl;
    return false;
  }
  if (!parse_proto_dims(*proto_layer, proto_dims)) {
    std::cerr << "[yolo26-seg] unexpected output1 dims" << std::endl;
    return false;
  }

  const std::size_t coeff_offset = 6;
  if (det_dims.channels < coeff_offset + proto_dims.c) {
    std::cerr << "[yolo26-seg] output0 channels too small: " << det_dims.channels
              << " (need >= " << (coeff_offset + proto_dims.c) << ")" << std::endl;
    return false;
  }

  const float* det_buf = static_cast<const float*>(det_layer->buffer);
  const float* proto_buf = static_cast<const float*>(proto_layer->buffer);

  const std::size_t proto_size = proto_dims.h * proto_dims.w;
  objectList.clear();
  objectList.reserve(det_dims.num);

  for (std::size_t i = 0; i < det_dims.num; ++i) {
    const std::size_t base = i * det_dims.channels;
    const float score = det_buf[base + 4];
    const int cls = static_cast<int>(det_buf[base + 5]);

    if (cls < 0) continue;

    float threshold = 0.0f;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
      const std::size_t cls_idx = static_cast<std::size_t>(cls);
      if (cls_idx < detectionParams.perClassPreclusterThreshold.size()) {
        threshold = detectionParams.perClassPreclusterThreshold[cls_idx];
      } else {
        threshold = detectionParams.perClassPreclusterThreshold[0];
      }
    }
    if (score < threshold) continue;

    const float x1 = det_buf[base + 0];
    const float y1 = det_buf[base + 1];
    const float x2 = det_buf[base + 2];
    const float y2 = det_buf[base + 3];

    NvDsInferInstanceMaskInfo obj{};
    if (!set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj)) continue;

    const float* coeffs = det_buf + base + coeff_offset;
    std::vector<float> mask_logits(proto_size, 0.0f);

    for (std::size_t c = 0; c < proto_dims.c; ++c) {
      const float coeff = coeffs[c];
      if (coeff == 0.0f) continue;
      const float* proto = proto_buf + c * proto_size;
      for (std::size_t idx = 0; idx < proto_size; ++idx) {
        mask_logits[idx] += coeff * proto[idx];
      }
    }

    if (!copy_mask_roi(mask_logits.data(), proto_dims.h, proto_dims.w,
                       x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj)) {
      continue;
    }

    obj.classId = cls;
    obj.detectionConfidence = score;
    objectList.emplace_back(obj);
  }

  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Seg);
