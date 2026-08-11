// SPDX-FileCopyrightText: 2026 Noesis
// SPDX-License-Identifier: MIT
//
// Strict DeepStream parser for the promoted DEIMv2 Wholebody49 outputs.
//
// label_xyxy_score is exactly [1240, 6] per callback:
//   [class_id, normalized_x1, normalized_y1, normalized_x2, normalized_y2, score]
// masks is exactly [1240, 80, 80] per callback. DeepStream strips the fixed
// batch dimension before invoking a custom parser, so batch-bearing output
// dimensions are deliberately rejected here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

constexpr char kLabelLayerName[] = "label_xyxy_score";
constexpr char kMaskLayerName[] = "masks";
constexpr std::size_t kQueryCount = 1240;
constexpr std::size_t kLabelChannels = 6;
constexpr std::size_t kMaskHeight = 80;
constexpr std::size_t kMaskWidth = 80;
constexpr int kBodyClassId = 0;

template <typename T>
inline T clamp(T value, T low, T high) {
  return std::min(high, std::max(low, value));
}

bool reject(const char* message) {
  std::cerr << "[deimv2-wholebody49] " << message << std::endl;
  return false;
}

bool resolve_exact_layers(const std::vector<NvDsInferLayerInfo>& layers,
                          bool require_masks,
                          const NvDsInferLayerInfo*& label_layer,
                          const NvDsInferLayerInfo*& mask_layer) {
  const std::size_t expected_count = require_masks ? 2U : 1U;
  if (layers.size() != expected_count) {
    return reject("unexpected output-layer count");
  }

  label_layer = nullptr;
  mask_layer = nullptr;
  for (const auto& layer : layers) {
    if (layer.layerName == nullptr) {
      return reject("output layer has no name");
    }
    if (std::strcmp(layer.layerName, kLabelLayerName) == 0) {
      if (label_layer != nullptr) {
        return reject("duplicate label_xyxy_score layer");
      }
      label_layer = &layer;
      continue;
    }
    if (require_masks && std::strcmp(layer.layerName, kMaskLayerName) == 0) {
      if (mask_layer != nullptr) {
        return reject("duplicate masks layer");
      }
      mask_layer = &layer;
      continue;
    }
    return reject("unexpected output layer name");
  }

  if (label_layer == nullptr || (require_masks && mask_layer == nullptr)) {
    return reject("required named output layer is missing");
  }
  return true;
}

bool validate_float_layer(const NvDsInferLayerInfo& layer, const char* label) {
  if (layer.buffer == nullptr) {
    std::cerr << "[deimv2-wholebody49] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[deimv2-wholebody49] " << label << " must be FLOAT" << std::endl;
    return false;
  }
  return true;
}

bool validate_label_dims(const NvDsInferLayerInfo& layer) {
  constexpr std::size_t kElements = kQueryCount * kLabelChannels;
  return layer.inferDims.numDims == 2U &&
         layer.inferDims.d[0] == kQueryCount &&
         layer.inferDims.d[1] == kLabelChannels &&
         layer.inferDims.numElements == kElements;
}

bool validate_mask_dims(const NvDsInferLayerInfo& layer) {
  constexpr std::size_t kElements = kQueryCount * kMaskHeight * kMaskWidth;
  return layer.inferDims.numDims == 3U &&
         layer.inferDims.d[0] == kQueryCount &&
         layer.inferDims.d[1] == kMaskHeight &&
         layer.inferDims.d[2] == kMaskWidth &&
         layer.inferDims.numElements == kElements;
}

bool validate_network_info(const NvDsInferNetworkInfo& network_info) {
  return network_info.width > 0U && network_info.height > 0U;
}

float threshold_for_class(const NvDsInferParseDetectionParams& params,
                          int class_id) {
  if (params.perClassPreclusterThreshold.empty()) {
    return 0.0F;
  }
  const std::size_t index = static_cast<std::size_t>(std::max(0, class_id));
  if (index < params.perClassPreclusterThreshold.size()) {
    return params.perClassPreclusterThreshold[index];
  }
  return params.perClassPreclusterThreshold[0];
}

struct Candidate {
  std::size_t query = 0;
  int class_id = -1;
  float score = 0.0F;
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
};

bool exact_class_id(float raw, int& class_id) {
  if (!std::isfinite(raw)) {
    return false;
  }
  const float integral = std::round(raw);
  const double exact_integral = static_cast<double>(integral);
  if (raw != integral ||
      exact_integral < static_cast<double>(std::numeric_limits<int>::min()) ||
      exact_integral > static_cast<double>(std::numeric_limits<int>::max())) {
    return false;
  }
  class_id = static_cast<int>(integral);
  return true;
}

bool collect_candidates(const float* labels,
                        const NvDsInferParseDetectionParams& detection_params,
                        std::array<Candidate, kQueryCount>& candidates,
                        std::size_t& candidate_count) {
  candidate_count = 0;
  for (std::size_t query = 0; query < kQueryCount; ++query) {
    const std::size_t base = query * kLabelChannels;
    for (std::size_t channel = 0; channel < kLabelChannels; ++channel) {
      if (!std::isfinite(labels[base + channel])) {
        return reject("label_xyxy_score contains a non-finite value");
      }
    }

    int class_id = -1;
    if (!exact_class_id(labels[base], class_id)) {
      return reject("label_xyxy_score contains a non-integral class ID");
    }
    const float score = labels[base + 5];
    if (class_id < 0) {
      continue;
    }
    if (detection_params.numClassesConfigured > 0 &&
        static_cast<unsigned int>(class_id) >=
            detection_params.numClassesConfigured) {
      continue;
    }
    if (score < threshold_for_class(detection_params, class_id)) {
      continue;
    }
    candidates[candidate_count++] = {
        query,       class_id,          score,          labels[base + 1],
        labels[base + 2], labels[base + 3], labels[base + 4]};
  }

  std::sort(candidates.begin(), candidates.begin() + candidate_count,
            [](const Candidate& left, const Candidate& right) {
              return left.score > right.score;
            });
  return true;
}

template <typename Object>
bool set_normalized_bbox(const Candidate& candidate,
                         const NvDsInferNetworkInfo& network_info,
                         Object& object) {
  if (network_info.width == 0U || network_info.height == 0U) {
    return false;
  }
  const float network_width = static_cast<float>(network_info.width);
  const float network_height = static_cast<float>(network_info.height);
  const float x1 = clamp(candidate.x1, 0.0F, 1.0F) * network_width;
  const float y1 = clamp(candidate.y1, 0.0F, 1.0F) * network_height;
  const float x2 = clamp(candidate.x2, 0.0F, 1.0F) * network_width;
  const float y2 = clamp(candidate.y2, 0.0F, 1.0F) * network_height;

  object.left = x1;
  object.top = y1;
  object.width = clamp(x2 - x1, 0.0F, network_width);
  object.height = clamp(y2 - y1, 0.0F, network_height);
  return object.width >= 1.0F && object.height >= 1.0F;
}

struct MaskRoi {
  std::unique_ptr<float[]> values;
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int size_bytes = 0;
};

bool build_mask_roi(const float* mask_source,
                    const Candidate& candidate,
                    MaskRoi& result) {
  const float x1 = clamp(candidate.x1, 0.0F, 1.0F) *
                   static_cast<float>(kMaskWidth);
  const float y1 = clamp(candidate.y1, 0.0F, 1.0F) *
                   static_cast<float>(kMaskHeight);
  const float x2 = clamp(candidate.x2, 0.0F, 1.0F) *
                   static_cast<float>(kMaskWidth);
  const float y2 = clamp(candidate.y2, 0.0F, 1.0F) *
                   static_cast<float>(kMaskHeight);

  const int ix1 = clamp(static_cast<int>(std::floor(x1)), 0,
                        static_cast<int>(kMaskWidth) - 1);
  const int iy1 = clamp(static_cast<int>(std::floor(y1)), 0,
                        static_cast<int>(kMaskHeight) - 1);
  const int ix2 = clamp(static_cast<int>(std::ceil(x2)), ix1 + 1,
                        static_cast<int>(kMaskWidth));
  const int iy2 = clamp(static_cast<int>(std::ceil(y2)), iy1 + 1,
                        static_cast<int>(kMaskHeight));
  const std::size_t roi_width = static_cast<std::size_t>(ix2 - ix1);
  const std::size_t roi_height = static_cast<std::size_t>(iy2 - iy1);
  const std::size_t mask_plane_size = kMaskHeight * kMaskWidth;
  const float* query_mask = mask_source + candidate.query * mask_plane_size;

  for (std::size_t row = 0; row < roi_height; ++row) {
    const float* source_row =
        query_mask + (static_cast<std::size_t>(iy1) + row) * kMaskWidth +
        static_cast<std::size_t>(ix1);
    for (std::size_t column = 0; column < roi_width; ++column) {
      if (!std::isfinite(source_row[column])) {
        return reject("masks contains a non-finite ROI value");
      }
    }
  }

  try {
    result.values = std::make_unique<float[]>(roi_width * roi_height);
  } catch (...) {
    return reject("unable to allocate instance-mask ROI");
  }
  for (std::size_t row = 0; row < roi_height; ++row) {
    const float* source_row =
        query_mask + (static_cast<std::size_t>(iy1) + row) * kMaskWidth +
        static_cast<std::size_t>(ix1);
    float* destination_row = result.values.get() + row * roi_width;
    for (std::size_t column = 0; column < roi_width; ++column) {
      destination_row[column] = clamp(source_row[column], 0.0F, 1.0F);
    }
  }
  result.width = static_cast<unsigned int>(roi_width);
  result.height = static_cast<unsigned int>(roi_height);
  result.size_bytes =
      static_cast<unsigned int>(roi_width * roi_height * sizeof(float));
  return true;
}

void release_mask_objects(std::vector<NvDsInferInstanceMaskInfo>& objects) {
  for (auto& object : objects) {
    delete[] object.mask;
    object.mask = nullptr;
  }
  objects.clear();
}

}  // namespace

extern "C" bool NvDsInferParseDeimv2Wholebody49(
    std::vector<NvDsInferLayerInfo> const& output_layers,
    NvDsInferNetworkInfo const& network_info,
    NvDsInferParseDetectionParams const& detection_params,
    std::vector<NvDsInferInstanceMaskInfo>& object_list) {
  const NvDsInferLayerInfo* label_layer = nullptr;
  const NvDsInferLayerInfo* mask_layer = nullptr;
  if (!resolve_exact_layers(output_layers, true, label_layer, mask_layer) ||
      !validate_float_layer(*label_layer, kLabelLayerName) ||
      !validate_float_layer(*mask_layer, kMaskLayerName)) {
    return false;
  }
  if (!validate_label_dims(*label_layer)) {
    return reject("label_xyxy_score must have exact callback shape [1240,6]");
  }
  if (!validate_mask_dims(*mask_layer)) {
    return reject("masks must have exact callback shape [1240,80,80]");
  }
  if (!validate_network_info(network_info)) {
    return reject("network dimensions must be positive");
  }

  const float* labels = static_cast<const float*>(label_layer->buffer);
  const float* masks = static_cast<const float*>(mask_layer->buffer);
  std::array<Candidate, kQueryCount> candidates{};
  std::size_t candidate_count = 0;
  if (!collect_candidates(labels, detection_params, candidates,
                          candidate_count)) {
    return false;
  }

  std::vector<NvDsInferInstanceMaskInfo> parsed;
  try {
    parsed.reserve(candidate_count);
  } catch (...) {
    return reject("unable to reserve instance-mask results");
  }
  for (std::size_t index = 0; index < candidate_count; ++index) {
    const Candidate& candidate = candidates[index];
    if (candidate.class_id != kBodyClassId) {
      continue;
    }
    NvDsInferInstanceMaskInfo object{};
    if (!set_normalized_bbox(candidate, network_info, object)) {
      continue;
    }
    MaskRoi mask;
    if (!build_mask_roi(masks, candidate, mask)) {
      release_mask_objects(parsed);
      return false;
    }
    object.classId = static_cast<unsigned int>(kBodyClassId);
    object.detectionConfidence = clamp(candidate.score, 0.0F, 1.0F);
    object.mask = mask.values.get();
    object.mask_width = mask.width;
    object.mask_height = mask.height;
    object.mask_size = mask.size_bytes;
    try {
      parsed.emplace_back(object);
    } catch (...) {
      release_mask_objects(parsed);
      return reject("unable to append instance-mask result");
    }
    mask.values.release();
  }
  object_list = std::move(parsed);
  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(
    NvDsInferParseDeimv2Wholebody49);

extern "C" bool NvDsInferParseDeimv2Wholebody49Boxes(
    std::vector<NvDsInferLayerInfo> const& output_layers,
    NvDsInferNetworkInfo const& network_info,
    NvDsInferParseDetectionParams const& detection_params,
    std::vector<NvDsInferObjectDetectionInfo>& object_list) {
  const NvDsInferLayerInfo* label_layer = nullptr;
  const NvDsInferLayerInfo* unused_mask_layer = nullptr;
  if (!resolve_exact_layers(output_layers, false, label_layer,
                            unused_mask_layer) ||
      !validate_float_layer(*label_layer, kLabelLayerName)) {
    return false;
  }
  if (!validate_label_dims(*label_layer)) {
    return reject("label_xyxy_score must have exact callback shape [1240,6]");
  }
  if (!validate_network_info(network_info)) {
    return reject("network dimensions must be positive");
  }

  const float* labels = static_cast<const float*>(label_layer->buffer);
  std::array<Candidate, kQueryCount> candidates{};
  std::size_t candidate_count = 0;
  if (!collect_candidates(labels, detection_params, candidates,
                          candidate_count)) {
    return false;
  }

  std::vector<NvDsInferObjectDetectionInfo> parsed;
  try {
    parsed.reserve(candidate_count);
  } catch (...) {
    return reject("unable to reserve bbox results");
  }
  for (std::size_t index = 0; index < candidate_count; ++index) {
    const Candidate& candidate = candidates[index];
    if (candidate.class_id != kBodyClassId) {
      continue;
    }
    NvDsInferObjectDetectionInfo object{};
    if (!set_normalized_bbox(candidate, network_info, object)) {
      continue;
    }
    object.classId = static_cast<unsigned int>(kBodyClassId);
    object.detectionConfidence = clamp(candidate.score, 0.0F, 1.0F);
    try {
      parsed.emplace_back(object);
    } catch (...) {
      return reject("unable to append bbox result");
    }
  }
  object_list = std::move(parsed);
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseDeimv2Wholebody49Boxes);
