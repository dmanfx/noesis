// SPDX-FileCopyrightText: 2026 Noesis
// SPDX-License-Identifier: MIT
//
// DeepStream 8 custom parser for DEIMv2 Wholebody49 outputs.
//
// label_xyxy_score layout per query:
//   [class_id, x1, y1, x2, y2, score]
// masks layout:
//   [query, 80, 80] probability masks, or [B, query, 80, 80] with the
//   DeepStream parser receiving the current batch pointer.
//
// The production DS8 profile supports two promoted prototype variants:
//   - DINOv3-S masks: label_xyxy_score + masks -> instance mask metadata.
//   - DINOv3-X boxes: label_xyxy_score only -> bbox metadata.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

template <typename T>
inline T clamp(T value, T low, T high) {
  return std::min(high, std::max(low, value));
}

const NvDsInferLayerInfo* find_layer_by_name(const std::vector<NvDsInferLayerInfo>& layers,
                                             const char* name) {
  for (const auto& layer : layers) {
    if (layer.layerName && std::strcmp(layer.layerName, name) == 0) {
      return &layer;
    }
  }
  return nullptr;
}

bool validate_float_layer(const NvDsInferLayerInfo& layer, const char* label) {
  if (!layer.buffer) {
    std::cerr << "[deimv2-wholebody49] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[deimv2-wholebody49] " << label << " must be FLOAT" << std::endl;
    return false;
  }
  return true;
}

struct LabelDims {
  std::size_t queries = 0;
  std::size_t channels = 0;
};

struct MaskDims {
  std::size_t queries = 0;
  std::size_t height = 0;
  std::size_t width = 0;
};

bool parse_label_dims(const NvDsInferLayerInfo& layer, LabelDims& out) {
  const int nd = layer.inferDims.numDims;
  if (nd == 2) {
    out.queries = static_cast<std::size_t>(layer.inferDims.d[0]);
    out.channels = static_cast<std::size_t>(layer.inferDims.d[1]);
    return out.queries > 0 && out.channels >= 6;
  }
  if (nd == 3) {
    if (layer.inferDims.d[2] >= 6) {
      out.queries = static_cast<std::size_t>(layer.inferDims.d[1]);
      out.channels = static_cast<std::size_t>(layer.inferDims.d[2]);
      return out.queries > 0;
    }
    if (layer.inferDims.d[1] >= 6) {
      out.queries = static_cast<std::size_t>(layer.inferDims.d[0]);
      out.channels = static_cast<std::size_t>(layer.inferDims.d[1]);
      return out.queries > 0;
    }
  }
  return false;
}

bool parse_mask_dims(const NvDsInferLayerInfo& layer, MaskDims& out) {
  const int nd = layer.inferDims.numDims;
  if (nd == 3) {
    out.queries = static_cast<std::size_t>(layer.inferDims.d[0]);
    out.height = static_cast<std::size_t>(layer.inferDims.d[1]);
    out.width = static_cast<std::size_t>(layer.inferDims.d[2]);
    return out.queries > 0 && out.height > 0 && out.width > 0;
  }
  if (nd == 4) {
    out.queries = static_cast<std::size_t>(layer.inferDims.d[1]);
    out.height = static_cast<std::size_t>(layer.inferDims.d[2]);
    out.width = static_cast<std::size_t>(layer.inferDims.d[3]);
    return out.queries > 0 && out.height > 0 && out.width > 0;
  }
  return false;
}

float threshold_for_class(const NvDsInferParseDetectionParams& params, int class_id) {
  if (params.perClassPreclusterThreshold.empty()) {
    return 0.0f;
  }
  const std::size_t idx = static_cast<std::size_t>(std::max(0, class_id));
  if (idx < params.perClassPreclusterThreshold.size()) {
    return params.perClassPreclusterThreshold[idx];
  }
  return params.perClassPreclusterThreshold[0];
}

bool set_bbox(float x1, float y1, float x2, float y2, bool normalized,
              const NvDsInferNetworkInfo& network_info,
              NvDsInferInstanceMaskInfo& obj) {
  const float net_w = static_cast<float>(network_info.width);
  const float net_h = static_cast<float>(network_info.height);
  if (normalized) {
    x1 *= net_w;
    x2 *= net_w;
    y1 *= net_h;
    y2 *= net_h;
  }

  x1 = clamp(x1, 0.0f, net_w);
  y1 = clamp(y1, 0.0f, net_h);
  x2 = clamp(x2, 0.0f, net_w);
  y2 = clamp(y2, 0.0f, net_h);
  obj.left = x1;
  obj.top = y1;
  obj.width = clamp(x2 - x1, 0.0f, net_w);
  obj.height = clamp(y2 - y1, 0.0f, net_h);
  return obj.width >= 1.0f && obj.height >= 1.0f;
}

bool set_detection_bbox(float x1, float y1, float x2, float y2, bool normalized,
                        const NvDsInferNetworkInfo& network_info,
                        NvDsInferObjectDetectionInfo& obj) {
  const float net_w = static_cast<float>(network_info.width);
  const float net_h = static_cast<float>(network_info.height);
  if (normalized) {
    x1 *= net_w;
    x2 *= net_w;
    y1 *= net_h;
    y2 *= net_h;
  }

  x1 = clamp(x1, 0.0f, net_w);
  y1 = clamp(y1, 0.0f, net_h);
  x2 = clamp(x2, 0.0f, net_w);
  y2 = clamp(y2, 0.0f, net_h);
  obj.left = x1;
  obj.top = y1;
  obj.width = clamp(x2 - x1, 0.0f, net_w);
  obj.height = clamp(y2 - y1, 0.0f, net_h);
  return obj.width >= 1.0f && obj.height >= 1.0f;
}

bool copy_mask_roi(const float* mask_src, const MaskDims& mask_dims,
                   float x1, float y1, float x2, float y2, bool normalized,
                   const NvDsInferNetworkInfo& network_info,
                   NvDsInferInstanceMaskInfo& obj) {
  if (!mask_src || mask_dims.height == 0 || mask_dims.width == 0) {
    return false;
  }
  const float net_w = static_cast<float>(network_info.width);
  const float net_h = static_cast<float>(network_info.height);
  if (!normalized) {
    x1 /= net_w;
    x2 /= net_w;
    y1 /= net_h;
    y2 /= net_h;
  }

  const float fx1 = clamp(x1, 0.0f, 1.0f) * static_cast<float>(mask_dims.width);
  const float fy1 = clamp(y1, 0.0f, 1.0f) * static_cast<float>(mask_dims.height);
  const float fx2 = clamp(x2, 0.0f, 1.0f) * static_cast<float>(mask_dims.width);
  const float fy2 = clamp(y2, 0.0f, 1.0f) * static_cast<float>(mask_dims.height);

  const int ix1 = clamp(static_cast<int>(std::floor(fx1)), 0, static_cast<int>(mask_dims.width) - 1);
  const int iy1 = clamp(static_cast<int>(std::floor(fy1)), 0, static_cast<int>(mask_dims.height) - 1);
  const int ix2 = clamp(static_cast<int>(std::ceil(fx2)), ix1 + 1, static_cast<int>(mask_dims.width));
  const int iy2 = clamp(static_cast<int>(std::ceil(fy2)), iy1 + 1, static_cast<int>(mask_dims.height));
  const std::size_t roi_w = static_cast<std::size_t>(ix2 - ix1);
  const std::size_t roi_h = static_cast<std::size_t>(iy2 - iy1);
  if (roi_w == 0 || roi_h == 0) {
    return false;
  }

  obj.mask_width = static_cast<unsigned int>(roi_w);
  obj.mask_height = static_cast<unsigned int>(roi_h);
  obj.mask_size = static_cast<unsigned int>(roi_w * roi_h * sizeof(float));
  obj.mask = new float[roi_w * roi_h];

  for (std::size_t y = 0; y < roi_h; ++y) {
    const float* src_row =
        mask_src + (static_cast<std::size_t>(iy1) + y) * mask_dims.width + static_cast<std::size_t>(ix1);
    float* dst_row = obj.mask + y * roi_w;
    for (std::size_t x = 0; x < roi_w; ++x) {
      dst_row[x] = clamp(src_row[x], 0.0f, 1.0f);
    }
  }
  return true;
}

struct Candidate {
  std::size_t query = 0;
  int class_id = -1;
  float score = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

constexpr int kBodyClassId = 0;

}  // namespace

extern "C" bool NvDsInferParseDeimv2Wholebody49(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  const NvDsInferLayerInfo* label_layer = find_layer_by_name(outputLayersInfo, "label_xyxy_score");
  const NvDsInferLayerInfo* mask_layer = find_layer_by_name(outputLayersInfo, "masks");
  if (!label_layer || !mask_layer) {
    if (outputLayersInfo.size() >= 2) {
      label_layer = &outputLayersInfo[0];
      mask_layer = &outputLayersInfo[1];
    } else {
      std::cerr << "[deimv2-wholebody49] expected label and mask outputs, got "
                << outputLayersInfo.size() << std::endl;
      return false;
    }
  }
  if (!validate_float_layer(*label_layer, "label_xyxy_score")) return false;
  if (!validate_float_layer(*mask_layer, "masks")) return false;

  LabelDims label_dims{};
  MaskDims mask_dims{};
  if (!parse_label_dims(*label_layer, label_dims)) {
    std::cerr << "[deimv2-wholebody49] unexpected label_xyxy_score dims" << std::endl;
    return false;
  }
  if (!parse_mask_dims(*mask_layer, mask_dims)) {
    std::cerr << "[deimv2-wholebody49] unexpected masks dims" << std::endl;
    return false;
  }

  const float* labels = static_cast<const float*>(label_layer->buffer);
  const float* masks = static_cast<const float*>(mask_layer->buffer);
  const bool normalized_boxes = [&]() {
    float max_coord = 0.0f;
    const std::size_t sample_count = std::min<std::size_t>(label_dims.queries, 32);
    for (std::size_t i = 0; i < sample_count; ++i) {
      const std::size_t base = i * label_dims.channels;
      max_coord = std::max(max_coord, std::fabs(labels[base + 1]));
      max_coord = std::max(max_coord, std::fabs(labels[base + 2]));
      max_coord = std::max(max_coord, std::fabs(labels[base + 3]));
      max_coord = std::max(max_coord, std::fabs(labels[base + 4]));
    }
    return max_coord <= 2.0f;
  }();

  std::vector<Candidate> candidates;
  candidates.reserve(label_dims.queries);
  for (std::size_t i = 0; i < label_dims.queries; ++i) {
    const std::size_t base = i * label_dims.channels;
    const int class_id = static_cast<int>(std::round(labels[base + 0]));
    const float score = labels[base + 5];
    if (class_id < 0) continue;
    if (detectionParams.numClassesConfigured > 0 &&
        static_cast<unsigned int>(class_id) >= detectionParams.numClassesConfigured) {
      continue;
    }
    if (score < threshold_for_class(detectionParams, class_id)) continue;
    candidates.push_back({i, class_id, score, labels[base + 1], labels[base + 2], labels[base + 3], labels[base + 4]});
  }

  std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.score > b.score;
  });

  objectList.clear();
  objectList.reserve(candidates.size());
  const std::size_t mask_plane_size = mask_dims.height * mask_dims.width;
  for (const Candidate& candidate : candidates) {
    if (candidate.class_id != kBodyClassId) {
      continue;
    }
    if (candidate.query >= mask_dims.queries) {
      continue;
    }
    NvDsInferInstanceMaskInfo obj{};
    if (!set_bbox(candidate.x1, candidate.y1, candidate.x2, candidate.y2,
                  normalized_boxes, networkInfo, obj)) {
      continue;
    }
    const float* mask_src = masks + candidate.query * mask_plane_size;
    if (!copy_mask_roi(mask_src, mask_dims, candidate.x1, candidate.y1, candidate.x2, candidate.y2,
                       normalized_boxes, networkInfo, obj)) {
      continue;
    }
    obj.classId = static_cast<unsigned int>(candidate.class_id);
    obj.detectionConfidence = clamp(candidate.score, 0.0f, 1.0f);
    objectList.emplace_back(obj);
  }
  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseDeimv2Wholebody49);

extern "C" bool NvDsInferParseDeimv2Wholebody49Boxes(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  const NvDsInferLayerInfo* label_layer = find_layer_by_name(outputLayersInfo, "label_xyxy_score");
  if (!label_layer) {
    if (!outputLayersInfo.empty()) {
      label_layer = &outputLayersInfo[0];
    } else {
      std::cerr << "[deimv2-wholebody49] expected label output, got 0 layers" << std::endl;
      return false;
    }
  }
  if (!validate_float_layer(*label_layer, "label_xyxy_score")) return false;

  LabelDims label_dims{};
  if (!parse_label_dims(*label_layer, label_dims)) {
    std::cerr << "[deimv2-wholebody49] unexpected label_xyxy_score dims" << std::endl;
    return false;
  }

  const float* labels = static_cast<const float*>(label_layer->buffer);
  const bool normalized_boxes = [&]() {
    float max_coord = 0.0f;
    const std::size_t sample_count = std::min<std::size_t>(label_dims.queries, 32);
    for (std::size_t i = 0; i < sample_count; ++i) {
      const std::size_t base = i * label_dims.channels;
      max_coord = std::max(max_coord, std::fabs(labels[base + 1]));
      max_coord = std::max(max_coord, std::fabs(labels[base + 2]));
      max_coord = std::max(max_coord, std::fabs(labels[base + 3]));
      max_coord = std::max(max_coord, std::fabs(labels[base + 4]));
    }
    return max_coord <= 2.0f;
  }();

  std::vector<Candidate> candidates;
  candidates.reserve(label_dims.queries);
  for (std::size_t i = 0; i < label_dims.queries; ++i) {
    const std::size_t base = i * label_dims.channels;
    const int class_id = static_cast<int>(std::round(labels[base + 0]));
    const float score = labels[base + 5];
    if (class_id != kBodyClassId) continue;
    if (detectionParams.numClassesConfigured > 0 &&
        static_cast<unsigned int>(class_id) >= detectionParams.numClassesConfigured) {
      continue;
    }
    if (score < threshold_for_class(detectionParams, class_id)) continue;
    candidates.push_back({i, class_id, score, labels[base + 1], labels[base + 2], labels[base + 3], labels[base + 4]});
  }

  std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.score > b.score;
  });

  objectList.clear();
  objectList.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    NvDsInferObjectDetectionInfo obj{};
    if (!set_detection_bbox(candidate.x1, candidate.y1, candidate.x2, candidate.y2,
                            normalized_boxes, networkInfo, obj)) {
      continue;
    }
    obj.classId = static_cast<unsigned int>(candidate.class_id);
    obj.detectionConfidence = clamp(candidate.score, 0.0f, 1.0f);
    objectList.emplace_back(obj);
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseDeimv2Wholebody49Boxes);
