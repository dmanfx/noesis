#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "nvdsinfer_custom_impl.h"

namespace {

constexpr std::size_t kQueryCount = 100;
constexpr std::size_t kClassCount = 2;
constexpr std::size_t kKeypointSlots = 34;
constexpr std::size_t kKeypointValues = 8;
constexpr std::size_t kPersonClassIndex = 1;
constexpr std::size_t kPersonKeypointOffset = 17;
constexpr std::size_t kPersonKeypointCount = 17;
constexpr double kKeypointTraceAlpha = 0.2;
constexpr double kMinimumLogWeightArgument = 1.0e-12;

std::atomic<unsigned long long> g_frame_counter{0};

template <typename T>
inline T clamp(T value, T lower, T upper) {
  return std::min(upper, std::max(lower, value));
}

inline float sigmoid(float value) {
  if (value >= 0.0f) {
    const float z = std::exp(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(value);
  return z / (1.0f + z);
}

double log_add_exp(double left, double right) {
  if (left == -std::numeric_limits<double>::infinity()) return right;
  if (right == -std::numeric_limits<double>::infinity()) return left;
  const double maximum = std::max(left, right);
  return maximum + std::log1p(std::exp(-std::abs(left - right)));
}

bool official_fused_person_score(
    const float* query_keypoints,
    float base_score,
    float& fused_score) {
  if (!query_keypoints || !std::isfinite(base_score) || base_score < 0.0f) {
    return false;
  }
  double weighted_log_trace_sum =
      -std::numeric_limits<double>::infinity();
  double log_weight_sum = -std::numeric_limits<double>::infinity();
  for (std::size_t keypoint = 0;
       keypoint < kPersonKeypointCount;
       ++keypoint) {
    const float* row =
        query_keypoints +
        (kPersonKeypointOffset + keypoint) * kKeypointValues;
    const double findable_logit = static_cast<double>(row[2]);
    const double log_l11 = static_cast<double>(row[4]);
    const double l21 = static_cast<double>(row[5]);
    const double log_l22 = static_cast<double>(row[6]);
    if (!std::isfinite(findable_logit) || !std::isfinite(log_l11) ||
        !std::isfinite(l21) || !std::isfinite(log_l22)) {
      return false;
    }

    const double findable = static_cast<double>(
        sigmoid(static_cast<float>(findable_logit)));
    const double log_weight =
        std::log(std::max(findable, kMinimumLogWeightArgument));
    const double log_t1 = -2.0 * log_l11;
    const double log_t2 = -2.0 * log_l22;
    const double log_t3 =
        2.0 *
            std::log(std::max(std::abs(l21), kMinimumLogWeightArgument)) +
        log_t1 + log_t2;
    const double log_trace =
        log_add_exp(log_add_exp(log_t1, log_t2), log_t3);
    weighted_log_trace_sum = log_add_exp(
        weighted_log_trace_sum, log_trace + log_weight);
    log_weight_sum = log_add_exp(log_weight_sum, log_weight);
  }

  const double log_mean_trace =
      weighted_log_trace_sum - log_weight_sum;
  const double fused =
      static_cast<double>(base_score) *
      std::exp(-kKeypointTraceAlpha * log_mean_trace);
  fused_score = static_cast<float>(fused);
  return std::isfinite(log_mean_trace) && std::isfinite(fused) &&
         std::isfinite(fused_score) && fused_score >= 0.0f;
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
    std::cerr << "[rfdetr-keypoint] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[rfdetr-keypoint] " << label << " must be FLOAT" << std::endl;
    return false;
  }
  return true;
}

bool is_device_ptr(const void* pointer) {
  if (!pointer) return false;
  cudaPointerAttributes attributes{};
  const cudaError_t error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeDevice ||
         attributes.type == cudaMemoryTypeManaged;
#else
  return attributes.memoryType == cudaMemoryTypeDevice;
#endif
}

const NvDsInferLayerInfo* find_layer_by_name(
    const std::vector<NvDsInferLayerInfo>& layers,
    const char* name) {
  if (!name || !*name) return nullptr;
  for (const auto& layer : layers) {
    if (layer.layerName && std::strcmp(layer.layerName, name) == 0) {
      return &layer;
    }
  }
  return nullptr;
}

bool parse_boxes_dims(const NvDsInferLayerInfo& layer) {
  const int nd = layer.inferDims.numDims;
  if (nd == 2) {
    return layer.inferDims.d[0] == static_cast<int>(kQueryCount) &&
           layer.inferDims.d[1] == 4;
  }
  return nd == 3 && layer.inferDims.d[0] == 1 &&
         layer.inferDims.d[1] == static_cast<int>(kQueryCount) &&
         layer.inferDims.d[2] == 4;
}

bool parse_logits_dims(const NvDsInferLayerInfo& layer) {
  const int nd = layer.inferDims.numDims;
  if (nd == 2) {
    return layer.inferDims.d[0] == static_cast<int>(kQueryCount) &&
           layer.inferDims.d[1] == static_cast<int>(kClassCount);
  }
  return nd == 3 && layer.inferDims.d[0] == 1 &&
         layer.inferDims.d[1] == static_cast<int>(kQueryCount) &&
         layer.inferDims.d[2] == static_cast<int>(kClassCount);
}

bool parse_keypoint_dims(const NvDsInferLayerInfo& layer) {
  const int nd = layer.inferDims.numDims;
  if (nd == 3) {
    return layer.inferDims.d[0] == static_cast<int>(kQueryCount) &&
           layer.inferDims.d[1] == static_cast<int>(kKeypointSlots) &&
           layer.inferDims.d[2] == static_cast<int>(kKeypointValues);
  }
  return nd == 4 && layer.inferDims.d[0] == 1 &&
         layer.inferDims.d[1] == static_cast<int>(kQueryCount) &&
         layer.inferDims.d[2] == static_cast<int>(kKeypointSlots) &&
         layer.inferDims.d[3] == static_cast<int>(kKeypointValues);
}

bool decode_box_to_xyxy(
    const float* box,
    unsigned int network_width,
    unsigned int network_height,
    float& x1,
    float& y1,
    float& x2,
    float& y2) {
  if (!box || network_width == 0 || network_height == 0) return false;
  const float cx = box[0];
  const float cy = box[1];
  const float width = box[2];
  const float height = box[3];
  if (!std::isfinite(cx) || !std::isfinite(cy) ||
      !std::isfinite(width) || !std::isfinite(height) ||
      width <= 0.0f || height <= 0.0f) {
    return false;
  }

  // RF-DETR 1.8.3 exports raw pred_boxes as normalized cx,cy,w,h.
  x1 = (cx - width * 0.5f) * static_cast<float>(network_width);
  y1 = (cy - height * 0.5f) * static_cast<float>(network_height);
  x2 = (cx + width * 0.5f) * static_cast<float>(network_width);
  y2 = (cy + height * 0.5f) * static_cast<float>(network_height);
  return std::isfinite(x1) && std::isfinite(y1) &&
         std::isfinite(x2) && std::isfinite(y2);
}

void set_bbox(
    float x1,
    float y1,
    float x2,
    float y2,
    unsigned int network_width,
    unsigned int network_height,
    NvDsInferObjectDetectionInfo& object) {
  x1 = clamp(x1, 0.0f, static_cast<float>(network_width));
  y1 = clamp(y1, 0.0f, static_cast<float>(network_height));
  x2 = clamp(x2, 0.0f, static_cast<float>(network_width));
  y2 = clamp(y2, 0.0f, static_cast<float>(network_height));
  object.left = x1;
  object.top = y1;
  object.width = clamp(
      x2 - x1, 0.0f, static_cast<float>(network_width));
  object.height = clamp(
      y2 - y1, 0.0f, static_cast<float>(network_height));
}

}  // namespace

extern "C" bool NvDsInferParseRFDETRKeypoint(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  if (outputLayersInfo.size() != 3) {
    std::cerr << "[rfdetr-keypoint] expected exactly 3 output layers "
                 "(dets, labels, keypoints), got "
              << outputLayersInfo.size() << std::endl;
    return false;
  }
  if (networkInfo.width == 0 || networkInfo.height == 0) {
    std::cerr << "[rfdetr-keypoint] network dimensions must be positive"
              << std::endl;
    return false;
  }

  const NvDsInferLayerInfo* boxes =
      find_layer_by_name(outputLayersInfo, "dets");
  const NvDsInferLayerInfo* logits =
      find_layer_by_name(outputLayersInfo, "labels");
  const NvDsInferLayerInfo* keypoints =
      find_layer_by_name(outputLayersInfo, "keypoints");
  if (!boxes || !logits || !keypoints) {
    std::cerr << "[rfdetr-keypoint] missing required output layers. Found:";
    for (const auto& layer : outputLayersInfo) {
      std::cerr << " "
                << (layer.layerName ? layer.layerName : "<null>");
    }
    std::cerr << std::endl;
    return false;
  }
  if (!validate_float_layer(*boxes, "dets") ||
      !validate_float_layer(*logits, "labels") ||
      !validate_float_layer(*keypoints, "keypoints")) {
    return false;
  }
  if (!parse_boxes_dims(*boxes)) {
    std::cerr << "[rfdetr-keypoint] dets must be [100,4] or [1,100,4]"
              << std::endl;
    return false;
  }
  if (!parse_logits_dims(*logits)) {
    std::cerr << "[rfdetr-keypoint] labels must be [100,2] or [1,100,2]"
              << std::endl;
    return false;
  }
  if (!parse_keypoint_dims(*keypoints)) {
    std::cerr << "[rfdetr-keypoint] keypoints must be [100,34,8] "
                 "or [1,100,34,8]"
              << std::endl;
    return false;
  }

  const int debug_every =
      getenv_int("NOESIS_RFDETR_KEYPOINT_DEBUG_EVERY", 0);
  if (detectionParams.perClassPreclusterThreshold.empty()) {
    std::cerr << "[rfdetr-keypoint] missing class-0 pre-cluster threshold"
              << std::endl;
    return false;
  }
  const float threshold =
      detectionParams.perClassPreclusterThreshold[0];
  if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
    std::cerr << "[rfdetr-keypoint] class-0 pre-cluster threshold must be "
                 "finite and in [0,1]"
              << std::endl;
    return false;
  }

  const auto* boxes_buffer = static_cast<const float*>(boxes->buffer);
  const auto* logits_buffer = static_cast<const float*>(logits->buffer);
  const auto* keypoints_buffer =
      static_cast<const float*>(keypoints->buffer);
  const bool boxes_on_device = is_device_ptr(boxes->buffer);
  const bool logits_on_device = is_device_ptr(logits->buffer);
  const bool keypoints_on_device = is_device_ptr(keypoints->buffer);
  if (boxes_on_device != logits_on_device ||
      boxes_on_device != keypoints_on_device) {
    std::cerr << "[rfdetr-keypoint] inconsistent output buffer locations"
              << std::endl;
    return false;
  }

  std::vector<float> boxes_host;
  std::vector<float> person_logits_host;
  std::vector<float> keypoints_host;
  if (boxes_on_device) {
    boxes_host.resize(kQueryCount * 4);
    const cudaError_t boxes_error = cudaMemcpy(
        boxes_host.data(),
        boxes_buffer,
        boxes_host.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (boxes_error != cudaSuccess) {
      std::cerr << "[rfdetr-keypoint] cudaMemcpy dets failed: "
                << cudaGetErrorString(boxes_error) << std::endl;
      return false;
    }

    person_logits_host.resize(kQueryCount);
    const cudaError_t logits_error = cudaMemcpy2D(
        person_logits_host.data(),
        sizeof(float),
        logits_buffer + kPersonClassIndex,
        kClassCount * sizeof(float),
        sizeof(float),
        kQueryCount,
        cudaMemcpyDeviceToHost);
    if (logits_error != cudaSuccess) {
      std::cerr << "[rfdetr-keypoint] cudaMemcpy person logits failed: "
                << cudaGetErrorString(logits_error) << std::endl;
      return false;
    }

    keypoints_host.resize(
        kQueryCount * kKeypointSlots * kKeypointValues);
    const cudaError_t keypoints_error = cudaMemcpy(
        keypoints_host.data(),
        keypoints_buffer,
        keypoints_host.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (keypoints_error != cudaSuccess) {
      std::cerr << "[rfdetr-keypoint] cudaMemcpy keypoints failed: "
                << cudaGetErrorString(keypoints_error) << std::endl;
      return false;
    }
  }

  objectList.clear();
  objectList.reserve(kQueryCount);
  float max_base_person_score = 0.0f;
  float max_fused_person_score = 0.0f;
  for (std::size_t query = 0; query < kQueryCount; ++query) {
    const float person_logit = boxes_on_device
        ? person_logits_host[query]
        : logits_buffer[
              query * kClassCount +
              kPersonClassIndex];
    if (!std::isfinite(person_logit)) {
      std::cerr << "[rfdetr-keypoint] non-finite person logit at query "
                << query << std::endl;
      return false;
    }
    const float base_score = sigmoid(person_logit);
    max_base_person_score = std::max(max_base_person_score, base_score);
    const float* query_keypoints =
        (boxes_on_device ? keypoints_host.data() : keypoints_buffer) +
        query * kKeypointSlots * kKeypointValues;
    float score = 0.0f;
    if (!official_fused_person_score(
            query_keypoints, base_score, score)) {
      std::cerr << "[rfdetr-keypoint] invalid uncertainty-fused score at query "
                << query << std::endl;
      return false;
    }
    max_fused_person_score = std::max(max_fused_person_score, score);
    if (score < threshold) continue;

    const float* box = boxes_on_device
        ? boxes_host.data() + query * 4
        : boxes_buffer + query * 4;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    if (!decode_box_to_xyxy(
            box,
            networkInfo.width,
            networkInfo.height,
            x1,
            y1,
            x2,
            y2)) {
      std::cerr << "[rfdetr-keypoint] invalid normalized cxcywh box at query "
                << query << std::endl;
      return false;
    }

    NvDsInferObjectDetectionInfo object{};
    set_bbox(
        x1,
        y1,
        x2,
        y2,
        networkInfo.width,
        networkInfo.height,
        object);
    if (object.width < 1.0f || object.height < 1.0f) continue;
    object.classId = 0;
    object.detectionConfidence = score;
    object.rotation_angle = 0.0f;
    objectList.emplace_back(object);
  }

  if (debug_every > 0) {
    const unsigned long long frame = g_frame_counter.fetch_add(1) + 1;
    const auto divisor = static_cast<unsigned long long>(debug_every);
    if (divisor > 0 && frame % divisor == 0) {
      std::cerr << "[rfdetr-keypoint] frame=" << frame
                << " q=" << kQueryCount
                << " kept=" << objectList.size()
                << " person_idx=" << kPersonClassIndex
                << " threshold=" << threshold
                << " device_buf=" << (boxes_on_device ? 1 : 0)
                << " max_base_person_score=" << max_base_person_score
                << " max_fused_person_score=" << max_fused_person_score
                << std::endl;
    }
  }

  // The NvDsInferParseDetectionParams API has no field for preserving the
  // originating query index on NvDsInferObjectDetectionInfo. Raw keypoints are
  // therefore exposed as PGIE frame tensor meta and require a separate,
  // reviewed query-to-object association bridge before pose metadata can be
  // attached to individual objects.
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRFDETRKeypoint);
