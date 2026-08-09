/* DeepStream 8 semantic parser for Ultralytics YOLO26-sem exports.
 *
 * Current Ultralytics semantic ONNX exports finish with ArgMax + Cast and
 * expose one UINT8 class-id map named output0 for each image. This parser
 * copies that exact map into NvDsInferSegmentationMeta without inventing
 * probabilities that the exported graph does not provide.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

const NvDsInferLayerInfo* find_output(
    const std::vector<NvDsInferLayerInfo>& output_layers) {
  for (const auto& layer : output_layers) {
    if (layer.layerName != nullptr && std::strcmp(layer.layerName, "output0") == 0) {
      return &layer;
    }
  }
  return nullptr;
}

bool resolve_spatial_dims(const NvDsInferDims& dims, int& height, int& width) {
  if (dims.numDims == 2U) {
    height = dims.d[0];
    width = dims.d[1];
    return height > 0 && width > 0;
  }
  if (dims.numDims == 3U && dims.d[0] == 1) {
    height = dims.d[1];
    width = dims.d[2];
    return height > 0 && width > 0;
  }
  return false;
}

}  // namespace

extern "C" bool NvDsInferParseYolo26SemanticADE20K(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float /* segmentationThreshold */,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap) {
  if (classificationMap == nullptr) {
    std::cerr << "[yolo26-sem-ade20k] classificationMap is null" << std::endl;
    return false;
  }
  if (numClasses != 150U) {
    std::cerr << "[yolo26-sem-ade20k] expected 150 classes, got " << numClasses
              << std::endl;
    return false;
  }

  const NvDsInferLayerInfo* layer = find_output(outputLayersInfo);
  if (layer == nullptr || layer->buffer == nullptr) {
    std::cerr << "[yolo26-sem-ade20k] output0 is missing or empty" << std::endl;
    return false;
  }
  if (layer->dataType != UINT8) {
    std::cerr << "[yolo26-sem-ade20k] output0 must be UINT8, got dataType="
              << static_cast<int>(layer->dataType) << std::endl;
    return false;
  }

  int height = 0;
  int width = 0;
  if (!resolve_spatial_dims(layer->inferDims, height, width)) {
    std::cerr << "[yolo26-sem-ade20k] output0 must be [H,W] or [1,H,W]" << std::endl;
    return false;
  }
  if (height != static_cast<int>(networkInfo.height) ||
      width != static_cast<int>(networkInfo.width)) {
    std::cerr << "[yolo26-sem-ade20k] output0 dimensions " << width << "x" << height
              << " do not match network " << networkInfo.width << "x"
              << networkInfo.height << std::endl;
    return false;
  }

  const auto* source = static_cast<const std::uint8_t*>(layer->buffer);
  const std::size_t pixels = static_cast<std::size_t>(height) *
                             static_cast<std::size_t>(width);
  for (std::size_t index = 0; index < pixels; ++index) {
    const unsigned int class_id = static_cast<unsigned int>(source[index]);
    if (class_id >= numClasses) {
      std::cerr << "[yolo26-sem-ade20k] invalid class id " << class_id
                << " at pixel " << index << std::endl;
      return false;
    }
    classificationMap[index] = static_cast<int>(class_id);
  }
  classProbabilityMap = nullptr;
  return true;
}

CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(
    NvDsInferParseYolo26SemanticADE20K);
