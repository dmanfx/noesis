#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsinfer.h"
#include "metadata.hpp"
#include "nvdsmeta.h"

namespace py = pybind11;

namespace {
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

struct ObjMetaAccessor : public deepstream::ObjectMetadata {
  using deepstream::Metadata::data_;
};

struct FrameMetaAccessor : public deepstream::FrameMetadata {
  using deepstream::Metadata::data_;
};

static bool s_pose_meta_inited = false;
static NvDsMetaType s_pose_meta_type = NVDS_USER_META;
NvDsMetaType pose_meta_type() {
  if (!s_pose_meta_inited) {
    s_pose_meta_type = nvds_get_user_meta_type((gchar*)"NOESIS.POSE_FEATURES");
    s_pose_meta_inited = true;
  }
  return s_pose_meta_type;
}

gpointer pose_meta_copy(gpointer data, gpointer /*user_data*/) {
  // DeepStream calls user-meta copy/release with NvDsUserMeta* (not user_meta_data).
  // Return a deep copy of user_meta_data; DeepStream assigns it to the copied meta.
  if (!data) return nullptr;
  auto* user_meta = static_cast<NvDsUserMeta*>(data);
  if (!user_meta->user_meta_data) return nullptr;
  return g_strdup(static_cast<const gchar*>(user_meta->user_meta_data));
}

void pose_meta_release(gpointer data, gpointer /*user_data*/) {
  // DeepStream calls user-meta copy/release with NvDsUserMeta* (not user_meta_data).
  // Free only the user_meta_data we allocated in attach_pose_features*.
  if (!data) return;
  auto* user_meta = static_cast<NvDsUserMeta*>(data);
  if (user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = nullptr;
  }
}

NvDsObjectMeta* unwrap_object_meta(const deepstream::ObjectMetadata& obj_meta) {
  auto* accessor = reinterpret_cast<const ObjMetaAccessor*>(&obj_meta);
  return reinterpret_cast<NvDsObjectMeta*>(accessor->data_);
}

NvDsFrameMeta* unwrap_frame_meta(const deepstream::FrameMetadata& frame_meta) {
  auto* accessor = reinterpret_cast<const FrameMetaAccessor*>(&frame_meta);
  return reinterpret_cast<NvDsFrameMeta*>(accessor->data_);
}

struct PoseMatrixAccessor {
  int rows = 0;
  int cols = 0;
  std::function<float(int, int)> at;
};

std::optional<PoseMatrixAccessor> build_pose_accessor(
    const NvDsInferTensorMeta* tensor_meta,
    const NvDsInferLayerInfo& layer) {
  if (!tensor_meta) return std::nullopt;
  if (layer.dataType != FLOAT) return std::nullopt;
  if (!tensor_meta->out_buf_ptrs_host) return std::nullopt;

  const int min_cols = 56;
  const int max_cols = 128;
  const NvDsInferDims& dims = layer.inferDims;
  if (dims.numDims <= 0) return std::nullopt;

  uint64_t n_elems = 1;
  std::vector<int> dvals;
  dvals.reserve(static_cast<size_t>(dims.numDims));
  for (int i = 0; i < dims.numDims; ++i) {
    int d = static_cast<int>(dims.d[i]);
    if (d <= 0) return std::nullopt;
    dvals.push_back(d);
    n_elems *= static_cast<uint64_t>(d);
  }

  const auto* base = static_cast<const float*>(layer.buffer);
  if (tensor_meta->out_buf_ptrs_host) {
    const size_t layer_idx = static_cast<size_t>(&layer - tensor_meta->output_layers_info);
    if (layer_idx < static_cast<size_t>(tensor_meta->num_output_layers) &&
        tensor_meta->out_buf_ptrs_host[layer_idx]) {
      base = static_cast<const float*>(tensor_meta->out_buf_ptrs_host[layer_idx]);
    }
  }
  if (!base) return std::nullopt;

  int axis = -1;
  int best_dist = std::numeric_limits<int>::max();
  for (int i = 0; i < dims.numDims; ++i) {
    const int dim = dvals[static_cast<size_t>(i)];
    if (dim < min_cols || dim > max_cols) continue;
    const int dist = std::abs(dim - 57);
    if (dist < best_dist) {
      best_dist = dist;
      axis = i;
    }
  }
  if (axis < 0) return std::nullopt;

  uint64_t rows_u64 = 1ULL;
  for (int i = 0; i < dims.numDims; ++i) {
    if (i == axis) continue;
    rows_u64 *= static_cast<uint64_t>(dvals[static_cast<size_t>(i)]);
  }
  if (rows_u64 == 0ULL || rows_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const int rows = static_cast<int>(rows_u64);
  const int cols = dvals[static_cast<size_t>(axis)];

  std::vector<uint64_t> strides(static_cast<size_t>(dims.numDims), 1ULL);
  for (int i = dims.numDims - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * static_cast<uint64_t>(dvals[static_cast<size_t>(i + 1)]);
  }
  std::vector<int> row_axes;
  row_axes.reserve(static_cast<size_t>(dims.numDims - 1));
  for (int i = 0; i < dims.numDims; ++i) {
    if (i != axis) row_axes.push_back(i);
  }

  return PoseMatrixAccessor{
      rows,
      cols,
      [base, dvals, strides, row_axes, axis](int r, int c) -> float {
        uint64_t idx = static_cast<uint64_t>(c) * strides[static_cast<size_t>(axis)];
        int tmp = r;
        for (int j = static_cast<int>(row_axes.size()) - 1; j >= 0; --j) {
          const int ax = row_axes[static_cast<size_t>(j)];
          const int dim = dvals[static_cast<size_t>(ax)];
          const int coord = (dim > 0) ? (tmp % dim) : 0;
          tmp = (dim > 0) ? (tmp / dim) : tmp;
          idx += static_cast<uint64_t>(coord) * strides[static_cast<size_t>(ax)];
        }
        return base[idx];
      }};
}

bool extract_best_pose_row(const NvDsInferTensorMeta* tensor_meta,
                           int gie_id,
                           float score_threshold,
                           std::vector<float>& row_out) {
  if (!tensor_meta) return false;
  if (tensor_meta->unique_id != static_cast<guint>(gie_id)) return false;
  if (tensor_meta->num_output_layers == 0U || !tensor_meta->output_layers_info) return false;

  int selected_layer = -1;
  for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
    const auto& layer = tensor_meta->output_layers_info[i];
    const char* name = layer.layerName;
    if (name && std::string(name) == "output0") {
      selected_layer = static_cast<int>(i);
      break;
    }
  }
  if (selected_layer < 0 && tensor_meta->num_output_layers == 1U) {
    selected_layer = 0;
  }
  if (selected_layer < 0) {
    return false;
  }

  const auto& layer = tensor_meta->output_layers_info[static_cast<size_t>(selected_layer)];
  if (pose_meta_ext_debug_enabled()) {
    const auto& dims = layer.inferDims;
    std::string shape = "[";
    for (int i = 0; i < dims.numDims; ++i) {
      shape += std::to_string(static_cast<int>(dims.d[i]));
      if (i + 1 < dims.numDims) shape += ",";
    }
    shape += "]";
    g_print(
        "[noesis_pose_meta_ext] uid=%u layer=%s dtype=%d shape=%s score_th=%.3f\n",
        tensor_meta->unique_id,
        layer.layerName ? layer.layerName : "(null)",
        static_cast<int>(layer.dataType),
        shape.c_str(),
        score_threshold);
  }
  auto accessor_opt = build_pose_accessor(tensor_meta, layer);
  if (!accessor_opt.has_value()) return false;
  const PoseMatrixAccessor accessor = *accessor_opt;
  if (accessor.rows <= 0 || accessor.cols <= 0) return false;
  if (accessor.cols < 56) return false;

  float best_score = -std::numeric_limits<float>::infinity();
  int best_row = -1;
  for (int r = 0; r < accessor.rows; ++r) {
    float score = accessor.at(r, 4);
    if (!std::isfinite(score)) continue;
    if (score > best_score) {
      best_score = score;
      best_row = r;
    }
  }
  if (best_row < 0 || best_score < score_threshold) {
    if (pose_meta_ext_debug_enabled()) {
      g_print(
          "[noesis_pose_meta_ext] no row over threshold: best=%.4f threshold=%.4f rows=%d cols=%d\n",
          best_score,
          score_threshold,
          accessor.rows,
          accessor.cols);
    }
    return false;
  }

  row_out.resize(static_cast<size_t>(accessor.cols));
  for (int c = 0; c < accessor.cols; ++c) {
    row_out[static_cast<size_t>(c)] = accessor.at(best_row, c);
  }
  return true;
}

float clipf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

std::optional<py::dict> decode_pose_keypoints_payload(const deepstream::ObjectMetadata& obj_meta,
                                                       int gie_id,
                                                       int model_w,
                                                       int model_h,
                                                       float score_threshold,
                                                       bool letterbox) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) return std::nullopt;
  if (obj->rect_params.width <= 0.0f || obj->rect_params.height <= 0.0f) return std::nullopt;

  std::vector<float> best_row;
  bool found = false;
  for (GList* node = obj->obj_user_meta_list; node != nullptr; node = node->next) {
    auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
    if (!user_meta || user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
      continue;
    }
    auto* tensor_meta = static_cast<NvDsInferTensorMeta*>(user_meta->user_meta_data);
    if (!tensor_meta) continue;
    if (extract_best_pose_row(tensor_meta, gie_id, score_threshold, best_row)) {
      found = true;
      break;
    }
  }
  if (!found || best_row.empty()) return std::nullopt;

  const size_t pose_values = static_cast<size_t>(17 * 3);
  if (best_row.size() < pose_values + 5U) return std::nullopt;

  const float left = obj->rect_params.left;
  const float top = obj->rect_params.top;
  const float roi_w = obj->rect_params.width;
  const float roi_h = obj->rect_params.height;
  const float score = best_row[4];
  const size_t key_start = best_row.size() - pose_values;

  float max_xywh = 0.0f;
  for (size_t i = 0; i < 4 && i < best_row.size(); ++i) {
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
    const float new_w = roi_w * gain;
    const float new_h = roi_h * gain;
    pad_x = (model_w_f - new_w) * 0.5f;
    pad_y = (model_h_f - new_h) * 0.5f;
  }

  py::list keypoints_roi;
  py::list keypoints_abs;
  for (size_t k = 0; k < 17; ++k) {
    float x = best_row[key_start + k * 3];
    float y = best_row[key_start + k * 3 + 1];
    const float c = best_row[key_start + k * 3 + 2];

    if (normalized) {
      x *= static_cast<float>(model_w);
      y *= static_cast<float>(model_h);
    }
    if (letterbox && gain > 0.0f) {
      x = (x - pad_x) / gain;
      y = (y - pad_y) / gain;
    } else if (!letterbox && model_w > 0 && model_h > 0) {
      x = x * (roi_w / static_cast<float>(model_w));
      y = y * (roi_h / static_cast<float>(model_h));
    }

    x = clipf(x, 0.0f, roi_w);
    y = clipf(y, 0.0f, roi_h);
    keypoints_roi.append(py::make_tuple(x, y, c));
    keypoints_abs.append(py::make_tuple(static_cast<float>(x + left), static_cast<float>(y + top), c));
  }

  py::dict payload;
  payload["score"] = score;
  payload["keypoints_roi"] = keypoints_roi;
  payload["keypoints_abs"] = keypoints_abs;
  return payload;
}

}  // namespace

bool attach_pose_features(const deepstream::ObjectMetadata& obj_meta,
                          const std::string& payload_json,
                          bool replace_existing) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) {
    return false;
  }
  NvDsBatchMeta* batch_meta = obj->base_meta.batch_meta;
  if (!batch_meta) {
    return false;
  }

  NvDsMetaType meta_type = pose_meta_type();
  if (replace_existing) {
    std::vector<NvDsUserMeta*> to_remove;
    for (GList* node = obj->obj_user_meta_list; node != nullptr; node = node->next) {
      auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
      if (user_meta && user_meta->base_meta.meta_type == meta_type) {
        to_remove.push_back(user_meta);
      }
    }
    for (auto* user_meta : to_remove) {
      nvds_remove_user_meta_from_object(obj, user_meta);
    }
  }

  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    return false;
  }

  user_meta->base_meta.meta_type = meta_type;
  user_meta->user_meta_data = g_strdup(payload_json.c_str());
  user_meta->base_meta.copy_func = pose_meta_copy;
  user_meta->base_meta.release_func = pose_meta_release;
  user_meta->base_meta.batch_meta = batch_meta;
  nvds_add_user_meta_to_obj(obj, user_meta);
  return true;
}

bool attach_pose_features_frame(const deepstream::FrameMetadata& frame_meta,
                                const std::string& payload_json,
                                bool replace_existing) {
  NvDsFrameMeta* frame = unwrap_frame_meta(frame_meta);
  if (!frame) {
    return false;
  }
  NvDsBatchMeta* batch_meta = frame->base_meta.batch_meta;
  if (!batch_meta) {
    return false;
  }
  NvDsMetaType meta_type = pose_meta_type();
  if (replace_existing) {
    std::vector<NvDsUserMeta*> to_remove;
    for (GList* node = frame->frame_user_meta_list; node != nullptr; node = node->next) {
      auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
      if (user_meta && user_meta->base_meta.meta_type == meta_type) {
        to_remove.push_back(user_meta);
      }
    }
    for (auto* user_meta : to_remove) {
      nvds_remove_user_meta_from_frame(frame, user_meta);
    }
  }

  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    return false;
  }
  user_meta->base_meta.meta_type = meta_type;
  user_meta->user_meta_data = g_strdup(payload_json.c_str());
  user_meta->base_meta.copy_func = pose_meta_copy;
  user_meta->base_meta.release_func = pose_meta_release;
  user_meta->base_meta.batch_meta = batch_meta;
  nvds_add_user_meta_to_frame(frame, user_meta);
  return true;
}

py::object extract_pose_features(const deepstream::ObjectMetadata& obj_meta) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) {
    return py::none();
  }
  NvDsMetaType meta_type = pose_meta_type();
  for (GList* node = obj->obj_user_meta_list; node != nullptr; node = node->next) {
    auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
    if (!user_meta || user_meta->base_meta.meta_type != meta_type) {
      continue;
    }
    if (!user_meta->user_meta_data) {
      return py::none();
    }
    const char* payload = static_cast<const char*>(user_meta->user_meta_data);
    if (!payload) {
      return py::none();
    }
    return py::str(payload);
  }
  return py::none();
}

py::object extract_pose_keypoints(const deepstream::ObjectMetadata& obj_meta,
                                  int gie_id,
                                  int model_w,
                                  int model_h,
                                  float score_threshold,
                                  bool letterbox) {
  auto decoded = decode_pose_keypoints_payload(
      obj_meta,
      gie_id,
      model_w,
      model_h,
      score_threshold,
      letterbox);
  if (!decoded.has_value()) {
    return py::none();
  }
  return *decoded;
}

PYBIND11_MODULE(noesis_pose_meta_ext, m) {
  m.doc() = "Noesis DS8 helper bindings for attaching pose feature user meta.";
  m.def(
      "attach_pose_features",
      &attach_pose_features,
      py::arg("obj_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES user meta (JSON string) to an object.");
  m.def(
      "attach_pose_features_frame",
      &attach_pose_features_frame,
      py::arg("frame_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES user meta (JSON string) to a frame.");
  m.def(
      "extract_pose_features",
      &extract_pose_features,
      py::arg("obj_meta"),
      "Extract NOESIS.POSE_FEATURES user meta JSON payload from an object.");
  m.def(
      "extract_pose_keypoints",
      &extract_pose_keypoints,
      py::arg("obj_meta"),
      py::arg("gie_id"),
      py::arg("model_w"),
      py::arg("model_h"),
      py::arg("score_threshold") = 0.25f,
      py::arg("letterbox") = true,
      "Extract best pose row from NVDSINFER_TENSOR_OUTPUT_META and return keypoints_abs + score.");
  m.def("pose_meta_type", []() { return static_cast<int>(pose_meta_type()); });
}
