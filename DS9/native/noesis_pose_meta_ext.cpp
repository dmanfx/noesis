#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime_api.h>

#include "metadata.hpp"
#include "nvdsmeta.h"
#include "tensor.hpp"

namespace py = pybind11;

namespace {

struct JsonPayload {
  std::string json;
};

bool pose_meta_ext_debug_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* raw = std::getenv("NOESIS_POSE_META_EXT_DEBUG");
    std::string s = raw ? std::string(raw) : std::string();
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    enabled = (!s.empty() && s != "0" && s != "false" && s != "no" && s != "off") ? 1 : 0;
  }
  return enabled == 1;
}

size_t pose_meta_max_payload_bytes() {
  static bool initialized = false;
  static size_t max_bytes = 65536U;
  if (!initialized) {
    initialized = true;
    const char* raw = std::getenv("NOESIS_POSE_META_MAX_JSON_BYTES");
    if (raw && *raw) {
      try {
        const unsigned long parsed = std::stoul(std::string(raw));
        if (parsed >= 1024UL) {
          max_bytes = static_cast<size_t>(parsed);
        }
      } catch (...) {
      }
    }
  }
  return max_bytes;
}

bool pose_payload_within_limit(const std::string& payload_json) {
  const size_t max_bytes = pose_meta_max_payload_bytes();
  if (payload_json.size() <= max_bytes) {
    return true;
  }
  if (pose_meta_ext_debug_enabled()) {
    std::fprintf(
        stderr,
        "[noesis_pose_meta_ext] payload too large: size=%zu max=%zu\n",
        payload_json.size(),
        max_bytes);
  }
  return false;
}

int pose_meta_type() {
  static const int type = static_cast<int>(
      nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.POSE_FEATURES")));
  return type;
}

void* pose_meta_copy(void* opaque_meta, void* /*user_data*/) {
  if (opaque_meta == nullptr) return nullptr;
  // Wrap the callback's opaque metadata through the public DS9 API instead of
  // depending on the underlying metadata structure layout.
  deepstream::UserMetadata user_meta(opaque_meta);
  auto* source = static_cast<JsonPayload*>(user_meta.userData());
  return source != nullptr ? static_cast<void*>(new JsonPayload(*source)) : nullptr;
}

void pose_meta_release(void* opaque_meta, void* /*user_data*/) {
  if (opaque_meta == nullptr) return;
  deepstream::UserMetadata user_meta(opaque_meta);
  delete static_cast<JsonPayload*>(user_meta.userData());
}

template <typename OwnerMetadata>
bool attach_pose_json(
    deepstream::BatchMetadata& batch_meta,
    OwnerMetadata& owner_meta,
    const std::string& payload_json,
    bool replace_existing) {
  if (!pose_payload_within_limit(payload_json)) {
    return false;
  }

  const int meta_type = pose_meta_type();
  if (replace_existing) {
    // DS9 exposes public append but not removal. Updating our owned payload in
    // place preserves one logical record without reaching into SDK internals.
    bool found = false;
    bool updated = false;
    owner_meta.iterate(
        [&](const deepstream::UserMetadata& user_meta) {
          if (found) return;
          found = true;
          auto* payload = static_cast<JsonPayload*>(user_meta.userData());
          if (payload != nullptr) {
            payload->json = payload_json;
            updated = true;
          }
        },
        meta_type);
    if (found) {
      return updated;
    }
  }

  deepstream::UserMetadata user_meta(nullptr);
  if (!batch_meta.acquire(user_meta)) {
    return false;
  }
  auto payload = std::make_unique<JsonPayload>(JsonPayload{payload_json});
  user_meta.setMetaType(meta_type);
  user_meta.setUserData(payload.get(), pose_meta_copy, pose_meta_release);
  payload.release();
  owner_meta.append(user_meta);
  return true;
}

void throw_on_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(unsigned int target_device) {
    throw_on_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice failed");
    if (previous_device_ != static_cast<int>(target_device)) {
      throw_on_cuda(cudaSetDevice(static_cast<int>(target_device)), "cudaSetDevice failed");
      restore_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_ = false;
};

class TensorLayerMap {
 public:
  // DS9 getLayers() allocates observer Tensor wrappers. Own the wrappers for
  // this synchronous probe call; the inference buffers themselves remain SDK-owned.
  explicit TensorLayerMap(deepstream::TensorOutputUserMetadata& tensor_meta)
      : layers_(tensor_meta.getLayers()) {}

  ~TensorLayerMap() {
    std::unordered_set<deepstream::Tensor*> released;
    for (const auto& entry : layers_) {
      if (entry.second != nullptr && released.insert(entry.second).second) {
        delete entry.second;
      }
    }
  }

  TensorLayerMap(const TensorLayerMap&) = delete;
  TensorLayerMap& operator=(const TensorLayerMap&) = delete;

  const std::unordered_map<std::string, deepstream::Tensor*>& get() const { return layers_; }

 private:
  std::unordered_map<std::string, deepstream::Tensor*> layers_;
};

std::optional<uint64_t> tensor_num_elements(const deepstream::Tensor& tensor) {
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty() || shape.size() != static_cast<size_t>(tensor.rank())) {
    return std::nullopt;
  }
  uint64_t count = 1ULL;
  for (uint64_t dim : shape) {
    if (dim == 0ULL || count > std::numeric_limits<uint64_t>::max() / dim) {
      return std::nullopt;
    }
    count *= dim;
  }
  return count;
}

bool tensor_is_contiguous(const deepstream::Tensor& tensor) {
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty() || shape.size() != static_cast<size_t>(tensor.rank())) {
    return false;
  }
  uint64_t expected_stride = 1ULL;
  for (size_t offset = 0; offset < shape.size(); ++offset) {
    const size_t axis = shape.size() - 1U - offset;
    if (tensor.stride(static_cast<unsigned int>(axis)) != expected_stride) {
      return false;
    }
    if (shape[axis] == 0ULL ||
        expected_stride > std::numeric_limits<uint64_t>::max() / shape[axis]) {
      return false;
    }
    expected_stride *= shape[axis];
  }
  return true;
}

std::string tensor_shape_string(const deepstream::Tensor& tensor) {
  const deepstream::TensorShape shape = tensor.shape();
  std::string rendered = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    rendered += std::to_string(shape[i]);
    if (i + 1U < shape.size()) rendered += ",";
  }
  rendered += "]";
  return rendered;
}

std::vector<float> copy_float32_tensor_to_host(const deepstream::Tensor& tensor) {
  if (tensor.dtype() != deepstream::Tensor::FLOAT || tensor.bits() != 32U) {
    throw std::runtime_error("Pose tensor must be FLOAT32");
  }
  if (!tensor_is_contiguous(tensor)) {
    throw std::runtime_error("Pose tensor must be contiguous");
  }
  const auto count_opt = tensor_num_elements(tensor);
  if (!count_opt.has_value() || *count_opt == 0ULL ||
      *count_opt > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
    throw std::runtime_error("Pose tensor shape is invalid");
  }
  const size_t count = static_cast<size_t>(*count_opt);
  const uint64_t expected_bytes = static_cast<uint64_t>(count) * sizeof(float);
  if (tensor.size() != expected_bytes || tensor.data() == nullptr) {
    throw std::runtime_error("Pose tensor storage does not match its public shape");
  }

  std::vector<float> values(count);
  switch (tensor.deviceType()) {
    case deepstream::Tensor::CPU:
      std::memcpy(values.data(), tensor.data(), static_cast<size_t>(expected_bytes));
      break;
    case deepstream::Tensor::GPU: {
      CudaDeviceGuard guard(tensor.deviceId());
      throw_on_cuda(
          cudaMemcpy(
              values.data(),
              tensor.data(),
              static_cast<size_t>(expected_bytes),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy pose tensor failed");
      break;
    }
    default:
      throw std::runtime_error("Pose tensor has no supported device");
  }
  return values;
}

struct PoseMatrixAccessor {
  int rows = 0;
  int cols = 0;
  std::function<float(int, int)> at;
};

constexpr size_t kRFDETRQueryCount = 100U;
constexpr size_t kRFDETRClassCount = 2U;
constexpr size_t kRFDETRKeypointSlots = 34U;
constexpr size_t kRFDETRPersonKeypointOffset = 17U;
constexpr size_t kRFDETRPersonKeypointCount = 17U;
constexpr size_t kRFDETRKeypointValues = 8U;
constexpr float kRFDETRNetworkWidth = 576.0f;
constexpr float kRFDETRNetworkHeight = 576.0f;
constexpr double kRFDETRKeypointTraceAlpha = 0.2;
constexpr double kRFDETRMinimumLogWeightArgument = 1.0e-12;

struct RFDETRKeypointTensors {
  std::vector<float> boxes;
  std::vector<float> logits;
  std::vector<float> keypoints;
};

struct RFDETRObjectCandidate {
  size_t frame_object_index = 0U;
  unsigned long object_id = 0UL;
  NvOSD_RectParams rect{};
};

struct RFDETRQueryCandidate {
  size_t query_index = 0U;
  float base_score = 0.0f;
  float score = 0.0f;
  float left = 0.0f;
  float top = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
};

float clipf(float value, float lower, float upper);

bool tensor_has_shape(
    const deepstream::Tensor& tensor,
    const std::vector<uint64_t>& expected) {
  deepstream::TensorShape shape = tensor.shape();
  if (shape.size() == expected.size() + 1U && shape.front() == 1U) {
    shape.erase(shape.begin());
  }
  return shape == expected;
}

deepstream::Tensor* require_named_tensor(
    const std::unordered_map<std::string, deepstream::Tensor*>& layers,
    const char* name,
    const std::vector<uint64_t>& expected_shape) {
  const auto found = layers.find(name);
  if (found == layers.end() || found->second == nullptr) {
    throw std::runtime_error(
        std::string("RF-DETR keypoint tensor meta is missing layer ") + name);
  }
  if (!tensor_has_shape(*found->second, expected_shape)) {
    throw std::runtime_error(
        std::string("RF-DETR keypoint tensor ") + name +
        " has unexpected shape " + tensor_shape_string(*found->second));
  }
  return found->second;
}

RFDETRKeypointTensors read_rfdetr_keypoint_tensors(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id) {
  std::optional<RFDETRKeypointTensors> decoded;
  unsigned int matching_meta_count = 0U;
  std::optional<std::string> decode_error;
  frame_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta ||
            tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }
        ++matching_meta_count;
        if (decoded.has_value() || decode_error.has_value()) return;
        try {
          TensorLayerMap owned_layers(tensor_meta);
          const auto& layers = owned_layers.get();
          deepstream::Tensor* boxes = require_named_tensor(
              layers, "dets", {kRFDETRQueryCount, 4U});
          deepstream::Tensor* logits = require_named_tensor(
              layers, "labels", {kRFDETRQueryCount, kRFDETRClassCount});
          deepstream::Tensor* keypoints = require_named_tensor(
              layers,
              "keypoints",
              {
                  kRFDETRQueryCount,
                  kRFDETRKeypointSlots,
                  kRFDETRKeypointValues,
              });
          decoded = RFDETRKeypointTensors{
              copy_float32_tensor_to_host(*boxes),
              copy_float32_tensor_to_host(*logits),
              copy_float32_tensor_to_host(*keypoints),
          };
        } catch (const std::exception& error) {
          decode_error = error.what();
        }
      },
      NVDSINFER_TENSOR_OUTPUT_META);

  if (matching_meta_count == 0U) {
    throw std::runtime_error(
        "RF-DETR keypoint PGIE frame tensor meta is missing");
  }
  if (matching_meta_count != 1U) {
    throw std::runtime_error(
        "RF-DETR keypoint PGIE requires exactly one frame tensor meta record");
  }
  if (decode_error.has_value()) {
    throw std::runtime_error(*decode_error);
  }
  if (!decoded.has_value() ||
      decoded->boxes.size() != kRFDETRQueryCount * 4U ||
      decoded->logits.size() != kRFDETRQueryCount * kRFDETRClassCount ||
      decoded->keypoints.size() !=
          kRFDETRQueryCount * kRFDETRKeypointSlots *
              kRFDETRKeypointValues) {
    throw std::runtime_error(
        "RF-DETR keypoint PGIE tensor storage contract is invalid");
  }
  return std::move(*decoded);
}

float sigmoid_stable(float value) {
  if (value >= 0.0f) {
    const float z = std::exp(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(value);
  return z / (1.0f + z);
}

double log_add_exp_stable(double left, double right) {
  if (left == -std::numeric_limits<double>::infinity()) return right;
  if (right == -std::numeric_limits<double>::infinity()) return left;
  const double maximum = std::max(left, right);
  return maximum + std::log1p(std::exp(-std::abs(left - right)));
}

float rfdetr_official_fused_person_score(
    const RFDETRKeypointTensors& tensors,
    size_t query_index,
    float base_score) {
  const size_t query_base =
      query_index * kRFDETRKeypointSlots * kRFDETRKeypointValues;
  double weighted_log_trace_sum =
      -std::numeric_limits<double>::infinity();
  double log_weight_sum = -std::numeric_limits<double>::infinity();
  for (size_t keypoint = 0U;
       keypoint < kRFDETRPersonKeypointCount;
       ++keypoint) {
    const size_t offset =
        query_base +
        (kRFDETRPersonKeypointOffset + keypoint) *
            kRFDETRKeypointValues;
    const double findable_logit =
        static_cast<double>(tensors.keypoints[offset + 2U]);
    const double log_l11 =
        static_cast<double>(tensors.keypoints[offset + 4U]);
    const double l21 =
        static_cast<double>(tensors.keypoints[offset + 5U]);
    const double log_l22 =
        static_cast<double>(tensors.keypoints[offset + 6U]);
    if (!std::isfinite(findable_logit) || !std::isfinite(log_l11) ||
        !std::isfinite(l21) || !std::isfinite(log_l22)) {
      throw std::runtime_error(
          "RF-DETR keypoint output contains non-finite score components");
    }

    const double findable = static_cast<double>(
        sigmoid_stable(static_cast<float>(findable_logit)));
    const double log_weight = std::log(
        std::max(findable, kRFDETRMinimumLogWeightArgument));
    const double log_t1 = -2.0 * log_l11;
    const double log_t2 = -2.0 * log_l22;
    const double log_t3 =
        2.0 * std::log(std::max(
                  std::abs(l21), kRFDETRMinimumLogWeightArgument)) +
        log_t1 + log_t2;
    const double log_trace = log_add_exp_stable(
        log_add_exp_stable(log_t1, log_t2), log_t3);
    weighted_log_trace_sum = log_add_exp_stable(
        weighted_log_trace_sum, log_trace + log_weight);
    log_weight_sum =
        log_add_exp_stable(log_weight_sum, log_weight);
  }

  const double log_mean_trace =
      weighted_log_trace_sum - log_weight_sum;
  const double fused =
      static_cast<double>(base_score) *
      std::exp(-kRFDETRKeypointTraceAlpha * log_mean_trace);
  const float fused_score = static_cast<float>(fused);
  if (!std::isfinite(log_mean_trace) || !std::isfinite(fused) ||
      !std::isfinite(fused_score) || fused_score < 0.0f) {
    throw std::runtime_error(
        "RF-DETR keypoint uncertainty-fused person score is invalid");
  }
  return fused_score;
}

float rect_iou(
    float left_a,
    float top_a,
    float width_a,
    float height_a,
    float left_b,
    float top_b,
    float width_b,
    float height_b) {
  if (!(width_a > 0.0f && height_a > 0.0f &&
        width_b > 0.0f && height_b > 0.0f)) {
    return 0.0f;
  }
  const float right_a = left_a + width_a;
  const float bottom_a = top_a + height_a;
  const float right_b = left_b + width_b;
  const float bottom_b = top_b + height_b;
  const float intersection_width =
      std::max(0.0f, std::min(right_a, right_b) - std::max(left_a, left_b));
  const float intersection_height =
      std::max(0.0f, std::min(bottom_a, bottom_b) - std::max(top_a, top_b));
  const float intersection = intersection_width * intersection_height;
  const float union_area =
      width_a * height_a + width_b * height_b - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

std::vector<RFDETRObjectCandidate> collect_rfdetr_person_objects(
    const deepstream::FrameMetadata& frame_meta) {
  std::vector<RFDETRObjectCandidate> objects;
  size_t object_index = 0U;
  frame_meta.iterate(
      [&](const deepstream::ObjectMetadata& object_meta) {
        if (object_meta.classId() == 0U) {
          const NvOSD_RectParams& rect = object_meta.rectParams();
          if (std::isfinite(rect.left) && std::isfinite(rect.top) &&
              std::isfinite(rect.width) && std::isfinite(rect.height) &&
              rect.width > 0.0f && rect.height > 0.0f) {
            objects.push_back(RFDETRObjectCandidate{
                object_index,
                object_meta.objectId(),
                rect,
            });
          }
        }
        ++object_index;
      });
  return objects;
}

std::vector<RFDETRQueryCandidate> collect_rfdetr_person_queries(
    const RFDETRKeypointTensors& tensors,
    float frame_width,
    float frame_height,
    float score_threshold) {
  std::vector<RFDETRQueryCandidate> queries;
  queries.reserve(kRFDETRQueryCount);
  for (size_t query = 0U; query < kRFDETRQueryCount; ++query) {
    const float person_logit =
        tensors.logits[query * kRFDETRClassCount + 1U];
    if (!std::isfinite(person_logit)) {
      throw std::runtime_error(
          "RF-DETR keypoint labels contain a non-finite person logit");
    }
    const float base_score = sigmoid_stable(person_logit);
    const float score = rfdetr_official_fused_person_score(
        tensors, query, base_score);
    if (score < score_threshold) continue;
    const float* box = tensors.boxes.data() + query * 4U;
    const float cx = box[0];
    const float cy = box[1];
    const float width = box[2];
    const float height = box[3];
    if (!std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0f || height <= 0.0f) {
      throw std::runtime_error(
          "RF-DETR keypoint dets contain an invalid normalized cxcywh row");
    }
    if (width * kRFDETRNetworkWidth < 1.0f ||
        height * kRFDETRNetworkHeight < 1.0f) {
      continue;
    }
    const float x1 = clipf((cx - width * 0.5f) * frame_width, 0.0f, frame_width);
    const float y1 = clipf((cy - height * 0.5f) * frame_height, 0.0f, frame_height);
    const float x2 = clipf((cx + width * 0.5f) * frame_width, 0.0f, frame_width);
    const float y2 = clipf((cy + height * 0.5f) * frame_height, 0.0f, frame_height);
    if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;
    queries.push_back(RFDETRQueryCandidate{
        query,
        base_score,
        score,
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    });
  }
  return queries;
}

py::dict decode_rfdetr_keypoint_query(
    const RFDETRKeypointTensors& tensors,
    const RFDETRQueryCandidate& query,
    const RFDETRObjectCandidate& object,
    float frame_width,
    float frame_height,
    float iou) {
  py::list keypoints_roi;
  py::list keypoints_abs;
  const size_t query_base =
      query.query_index * kRFDETRKeypointSlots * kRFDETRKeypointValues;
  for (size_t keypoint = 0U;
       keypoint < kRFDETRPersonKeypointCount;
       ++keypoint) {
    const size_t offset =
        query_base +
        (kRFDETRPersonKeypointOffset + keypoint) *
            kRFDETRKeypointValues;
    const float raw_x = tensors.keypoints[offset];
    const float raw_y = tensors.keypoints[offset + 1U];
    const float raw_confidence = tensors.keypoints[offset + 2U];
    if (!std::isfinite(raw_x) || !std::isfinite(raw_y) ||
        !std::isfinite(raw_confidence)) {
      throw std::runtime_error(
          "RF-DETR keypoint output contains non-finite active values");
    }
    const float absolute_x = clipf(raw_x * frame_width, 0.0f, frame_width);
    const float absolute_y = clipf(raw_y * frame_height, 0.0f, frame_height);
    const float roi_x = clipf(
        absolute_x - object.rect.left, 0.0f, object.rect.width);
    const float roi_y = clipf(
        absolute_y - object.rect.top, 0.0f, object.rect.height);
    const float confidence = sigmoid_stable(raw_confidence);
    keypoints_roi.append(py::make_tuple(roi_x, roi_y, confidence));
    keypoints_abs.append(
        py::make_tuple(absolute_x, absolute_y, confidence));
  }

  py::dict payload;
  payload["object_index"] = object.frame_object_index;
  payload["object_id"] = object.object_id;
  payload["query_index"] = query.query_index;
  payload["match_iou"] = iou;
  payload["base_score"] = query.base_score;
  payload["score"] = query.score;
  payload["bbox"] = py::make_tuple(
      object.rect.left,
      object.rect.top,
      object.rect.width,
      object.rect.height);
  payload["keypoints_roi"] = std::move(keypoints_roi);
  payload["keypoints_abs"] = std::move(keypoints_abs);
  return payload;
}

py::dict extract_rfdetr_keypoint_matches_impl(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id,
    float score_threshold,
    float min_iou,
    float ambiguity_margin) {
  if (gie_id <= 0) {
    throw std::invalid_argument("RF-DETR keypoint gie_id must be positive");
  }
  if (!std::isfinite(score_threshold) ||
      score_threshold < 0.0f || score_threshold > 1.0f) {
    throw std::invalid_argument(
        "RF-DETR keypoint score threshold must be in [0,1]");
  }
  if (!std::isfinite(min_iou) || min_iou <= 0.0f || min_iou > 1.0f) {
    throw std::invalid_argument(
        "RF-DETR keypoint match IoU must be in (0,1]");
  }
  if (!std::isfinite(ambiguity_margin) ||
      ambiguity_margin < 0.0f || ambiguity_margin >= 1.0f) {
    throw std::invalid_argument(
        "RF-DETR keypoint ambiguity margin must be in [0,1)");
  }

  const unsigned int pipeline_width = frame_meta.pipelineWidth();
  const unsigned int pipeline_height = frame_meta.pipelineHeight();
  if (pipeline_width == 0U || pipeline_height == 0U) {
    throw std::runtime_error(
        "RF-DETR keypoint frame has no pipeline coordinate dimensions");
  }
  const float frame_width = static_cast<float>(pipeline_width);
  const float frame_height = static_cast<float>(pipeline_height);
  const RFDETRKeypointTensors tensors =
      read_rfdetr_keypoint_tensors(frame_meta, gie_id);
  const std::vector<RFDETRObjectCandidate> objects =
      collect_rfdetr_person_objects(frame_meta);
  const std::vector<RFDETRQueryCandidate> queries =
      collect_rfdetr_person_queries(
          tensors, frame_width, frame_height, score_threshold);

  std::vector<std::vector<float>> ious(
      objects.size(), std::vector<float>(queries.size(), 0.0f));
  for (size_t object_index = 0U;
       object_index < objects.size();
       ++object_index) {
    const auto& object = objects[object_index];
    for (size_t query_index = 0U;
         query_index < queries.size();
         ++query_index) {
      const auto& query = queries[query_index];
      ious[object_index][query_index] = rect_iou(
          object.rect.left,
          object.rect.top,
          object.rect.width,
          object.rect.height,
          query.left,
          query.top,
          query.width,
          query.height);
    }
  }

  py::list matches;
  size_t unmatched_objects = 0U;
  size_t ambiguous_objects = 0U;
  for (size_t object_index = 0U;
       object_index < objects.size();
       ++object_index) {
    if (queries.empty()) {
      ++unmatched_objects;
      continue;
    }
    size_t best_query = 0U;
    float best_iou = -1.0f;
    float second_query_iou = -1.0f;
    for (size_t query_index = 0U;
         query_index < queries.size();
         ++query_index) {
      const float iou = ious[object_index][query_index];
      if (iou > best_iou) {
        second_query_iou = best_iou;
        best_iou = iou;
        best_query = query_index;
      } else if (iou > second_query_iou) {
        second_query_iou = iou;
      }
    }
    if (best_iou < min_iou) {
      ++unmatched_objects;
      continue;
    }

    size_t best_object_for_query = 0U;
    float best_object_iou = -1.0f;
    float second_object_iou = -1.0f;
    for (size_t other_object = 0U;
         other_object < objects.size();
         ++other_object) {
      const float iou = ious[other_object][best_query];
      if (iou > best_object_iou) {
        second_object_iou = best_object_iou;
        best_object_iou = iou;
        best_object_for_query = other_object;
      } else if (iou > second_object_iou) {
        second_object_iou = iou;
      }
    }
    const float query_margin =
        best_iou - std::max(0.0f, second_query_iou);
    const float object_margin =
        best_object_iou - std::max(0.0f, second_object_iou);
    if (best_object_for_query != object_index ||
        query_margin < ambiguity_margin ||
        object_margin < ambiguity_margin) {
      ++ambiguous_objects;
      continue;
    }
    matches.append(decode_rfdetr_keypoint_query(
        tensors,
        queries[best_query],
        objects[object_index],
        frame_width,
        frame_height,
        best_iou));
  }

  py::dict diagnostics;
  diagnostics["person_objects"] = objects.size();
  diagnostics["person_queries"] = queries.size();
  diagnostics["matched_objects"] = py::len(matches);
  diagnostics["unmatched_objects"] = unmatched_objects;
  diagnostics["ambiguous_objects"] = ambiguous_objects;
  diagnostics["score_threshold"] = score_threshold;
  diagnostics["min_iou"] = min_iou;
  diagnostics["ambiguity_margin"] = ambiguity_margin;
  diagnostics["pipeline_width"] = pipeline_width;
  diagnostics["pipeline_height"] = pipeline_height;

  py::dict result;
  result["matches"] = std::move(matches);
  result["diagnostics"] = std::move(diagnostics);
  return result;
}

std::optional<PoseMatrixAccessor> build_pose_accessor(
    const deepstream::Tensor& tensor,
    const std::vector<float>& values) {
  const int min_cols = 56;
  const int max_cols = 128;
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty()) return std::nullopt;

  int axis = -1;
  int best_distance = std::numeric_limits<int>::max();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > static_cast<uint64_t>(std::numeric_limits<int>::max())) continue;
    const int dim = static_cast<int>(shape[i]);
    if (dim < min_cols || dim > max_cols) continue;
    const int distance = std::abs(dim - 57);
    if (distance < best_distance) {
      best_distance = distance;
      axis = static_cast<int>(i);
    }
  }
  if (axis < 0) return std::nullopt;

  uint64_t rows_u64 = 1ULL;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) == axis) continue;
    if (shape[i] == 0ULL || rows_u64 > std::numeric_limits<uint64_t>::max() / shape[i]) {
      return std::nullopt;
    }
    rows_u64 *= shape[i];
  }
  if (rows_u64 == 0ULL || rows_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const int rows = static_cast<int>(rows_u64);
  const int cols = static_cast<int>(shape[static_cast<size_t>(axis)]);

  std::vector<uint64_t> strides(shape.size(), 1ULL);
  for (size_t offset = 1; offset < shape.size(); ++offset) {
    const size_t i = shape.size() - 1U - offset;
    strides[i] = strides[i + 1U] * shape[i + 1U];
  }
  std::vector<int> row_axes;
  row_axes.reserve(shape.size() - 1U);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) != axis) row_axes.push_back(static_cast<int>(i));
  }

  const float* base = values.data();
  return PoseMatrixAccessor{
      rows,
      cols,
      [base, shape, strides, row_axes, axis](int row, int col) -> float {
        uint64_t index = static_cast<uint64_t>(col) * strides[static_cast<size_t>(axis)];
        int remaining = row;
        for (int j = static_cast<int>(row_axes.size()) - 1; j >= 0; --j) {
          const int current_axis = row_axes[static_cast<size_t>(j)];
          const int dim = static_cast<int>(shape[static_cast<size_t>(current_axis)]);
          const int coordinate = dim > 0 ? remaining % dim : 0;
          remaining = dim > 0 ? remaining / dim : remaining;
          index += static_cast<uint64_t>(coordinate) *
                   strides[static_cast<size_t>(current_axis)];
        }
        return base[index];
      }};
}

std::pair<std::string, deepstream::Tensor*> select_pose_layer(
    const std::unordered_map<std::string, deepstream::Tensor*>& layers) {
  const auto named = layers.find("output0");
  if (named == layers.end()) {
    return {"", nullptr};
  }
  return {named->first, named->second};
}

bool extract_best_pose_row(
    deepstream::TensorOutputUserMetadata& tensor_meta,
    float score_threshold,
    std::vector<float>& row_out) {
  TensorLayerMap owned_layers(tensor_meta);
  const auto selected = select_pose_layer(owned_layers.get());
  if (selected.second == nullptr) return false;

  std::vector<float> values = copy_float32_tensor_to_host(*selected.second);
  if (pose_meta_ext_debug_enabled()) {
    std::fprintf(
        stderr,
        "[noesis_pose_meta_ext] uid=%u layer=%s dtype=%d bits=%u device=%d shape=%s score_th=%.3f\n",
        tensor_meta.uniqueId(),
        selected.first.c_str(),
        static_cast<int>(selected.second->dtype()),
        selected.second->bits(),
        static_cast<int>(selected.second->deviceType()),
        tensor_shape_string(*selected.second).c_str(),
        score_threshold);
  }

  auto accessor_opt = build_pose_accessor(*selected.second, values);
  if (!accessor_opt.has_value()) return false;
  const PoseMatrixAccessor accessor = *accessor_opt;
  if (accessor.rows <= 0 || accessor.cols < 56) return false;

  float best_score = -std::numeric_limits<float>::infinity();
  int best_row = -1;
  for (int row = 0; row < accessor.rows; ++row) {
    const float score = accessor.at(row, 4);
    if (!std::isfinite(score)) continue;
    if (score > best_score) {
      best_score = score;
      best_row = row;
    }
  }
  if (best_row < 0 || best_score < score_threshold) {
    if (pose_meta_ext_debug_enabled()) {
      std::fprintf(
          stderr,
          "[noesis_pose_meta_ext] no row over threshold: best=%.4f threshold=%.4f rows=%d cols=%d\n",
          best_score,
          score_threshold,
          accessor.rows,
          accessor.cols);
    }
    return false;
  }

  row_out.resize(static_cast<size_t>(accessor.cols));
  for (int col = 0; col < accessor.cols; ++col) {
    row_out[static_cast<size_t>(col)] = accessor.at(best_row, col);
  }
  return true;
}

float clipf(float value, float lower, float upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

std::optional<py::dict> decode_pose_keypoints_payload(
    const deepstream::ObjectMetadata& obj_meta,
    int gie_id,
    int model_w,
    int model_h,
    float score_threshold,
    bool letterbox) {
  const NvOSD_RectParams& rect = obj_meta.rectParams();
  if (rect.width <= 0.0f || rect.height <= 0.0f) return std::nullopt;

  std::vector<float> best_row;
  std::optional<std::string> matched_error;
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (!best_row.empty()) return;
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta || tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }
        try {
          (void)extract_best_pose_row(tensor_meta, score_threshold, best_row);
        } catch (const std::exception& exc) {
          matched_error = exc.what();
        }
      },
      NVDSINFER_TENSOR_OUTPUT_META);
  if (best_row.empty()) {
    if (matched_error.has_value() && pose_meta_ext_debug_enabled()) {
      std::fprintf(stderr, "[noesis_pose_meta_ext] %s\n", matched_error->c_str());
    }
    return std::nullopt;
  }

  constexpr size_t pose_values = 17U * 3U;
  if (best_row.size() < pose_values + 5U) return std::nullopt;

  const float left = rect.left;
  const float top = rect.top;
  const float roi_w = rect.width;
  const float roi_h = rect.height;
  const float score = best_row[4];
  const size_t key_start = best_row.size() - pose_values;

  float max_xywh = 0.0f;
  for (size_t i = 0; i < 4U && i < best_row.size(); ++i) {
    max_xywh = std::max(max_xywh, std::fabs(best_row[i]));
  }
  const bool normalized = max_xywh <= 2.0f;

  float gain = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
  if (letterbox && roi_w > 0.0f && roi_h > 0.0f && model_w > 0 && model_h > 0) {
    const float model_w_f = static_cast<float>(model_w);
    const float model_h_f = static_cast<float>(model_h);
    gain = std::min(model_w_f / roi_w, model_h_f / roi_h);
    pad_x = (model_w_f - roi_w * gain) * 0.5f;
    pad_y = (model_h_f - roi_h * gain) * 0.5f;
  }

  py::list keypoints_roi;
  py::list keypoints_abs;
  for (size_t keypoint = 0; keypoint < 17U; ++keypoint) {
    float x = best_row[key_start + keypoint * 3U];
    float y = best_row[key_start + keypoint * 3U + 1U];
    const float confidence = best_row[key_start + keypoint * 3U + 2U];

    if (normalized) {
      x *= static_cast<float>(model_w);
      y *= static_cast<float>(model_h);
    }
    if (letterbox && gain > 0.0f) {
      x = (x - pad_x) / gain;
      y = (y - pad_y) / gain;
    } else if (!letterbox && model_w > 0 && model_h > 0) {
      x *= roi_w / static_cast<float>(model_w);
      y *= roi_h / static_cast<float>(model_h);
    }

    x = clipf(x, 0.0f, roi_w);
    y = clipf(y, 0.0f, roi_h);
    keypoints_roi.append(py::make_tuple(x, y, confidence));
    keypoints_abs.append(py::make_tuple(x + left, y + top, confidence));
  }

  py::dict payload;
  payload["score"] = score;
  payload["keypoints_roi"] = std::move(keypoints_roi);
  payload["keypoints_abs"] = std::move(keypoints_abs);
  return payload;
}

}  // namespace

bool attach_pose_features(
    deepstream::BatchMetadata& batch_meta,
    deepstream::ObjectMetadata& obj_meta,
    const std::string& payload_json,
    bool replace_existing) {
  return attach_pose_json(batch_meta, obj_meta, payload_json, replace_existing);
}

bool attach_pose_features_frame(
    deepstream::BatchMetadata& batch_meta,
    deepstream::FrameMetadata& frame_meta,
    const std::string& payload_json,
    bool replace_existing) {
  return attach_pose_json(batch_meta, frame_meta, payload_json, replace_existing);
}

py::object extract_pose_features(const deepstream::ObjectMetadata& obj_meta) {
  std::optional<std::string> payload;
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (payload.has_value()) return;
        auto* stored = static_cast<JsonPayload*>(user_meta.userData());
        if (stored != nullptr) payload = stored->json;
      },
      pose_meta_type());
  return payload.has_value() ? py::cast(*payload) : py::none();
}

py::object extract_pose_keypoints(
    const deepstream::ObjectMetadata& obj_meta,
    int gie_id,
    int model_w,
    int model_h,
    float score_threshold,
    bool letterbox) {
  auto decoded = decode_pose_keypoints_payload(
      obj_meta, gie_id, model_w, model_h, score_threshold, letterbox);
  if (!decoded.has_value()) {
    return py::none();
  }
  return std::move(*decoded);
}

PYBIND11_MODULE(noesis_pose_meta_ext, m) {
  m.doc() =
      "Noesis DS9 public-Service-Maker helper for pose tensors and pose-feature user metadata.";
  m.def(
      "attach_pose_features",
      &attach_pose_features,
      py::arg("batch_meta"),
      py::arg("obj_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES through BatchMetadata/UserMetadata to an object.");
  m.def(
      "attach_pose_features_frame",
      &attach_pose_features_frame,
      py::arg("batch_meta"),
      py::arg("frame_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES through BatchMetadata/UserMetadata to a frame.");
  m.def(
      "extract_pose_features",
      &extract_pose_features,
      py::arg("obj_meta"),
      "Extract NOESIS.POSE_FEATURES through public ObjectMetadata iteration.");
  m.def(
      "extract_pose_keypoints",
      &extract_pose_keypoints,
      py::arg("obj_meta"),
      py::arg("gie_id"),
      py::arg("model_w"),
      py::arg("model_h"),
      py::arg("score_threshold") = 0.25f,
      py::arg("letterbox") = true,
      "Extract the best pose row through TensorOutputUserMetadata/Tensor.");
  m.def(
      "extract_rfdetr_keypoint_matches",
      &extract_rfdetr_keypoint_matches_impl,
      py::arg("frame_meta"),
      py::arg("gie_id"),
      py::arg("score_threshold") = 0.4f,
      py::arg("min_iou") = 0.7f,
      py::arg("ambiguity_margin") = 0.05f,
      "Decode frame-owned RF-DETR PGIE tensors and strictly associate "
      "person queries with object metadata.");
  m.def("pose_meta_type", []() { return pose_meta_type(); });
}
