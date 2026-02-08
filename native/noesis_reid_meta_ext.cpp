#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsinfer.h"
#include "metadata.hpp"
#include "nvdsmeta.h"

namespace py = pybind11;

namespace {

bool reid_meta_ext_debug_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* raw = std::getenv("NOESIS_REID_META_EXT_DEBUG");
    std::string s = raw ? std::string(raw) : std::string();
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    enabled = (!s.empty() && s != "0" && s != "false" && s != "no" && s != "off") ? 1 : 0;
  }
  return enabled == 1;
}

struct ObjMetaAccessor : public deepstream::ObjectMetadata {
  using deepstream::Metadata::data_;
};

NvDsObjectMeta* unwrap_object_meta(const deepstream::ObjectMetadata& obj_meta) {
  auto* accessor = reinterpret_cast<const ObjMetaAccessor*>(&obj_meta);
  return reinterpret_cast<NvDsObjectMeta*>(accessor->data_);
}

uint64_t dims_num_elements(const NvDsInferDims& dims) {
  if (dims.numDims <= 0) {
    return 0ULL;
  }
  uint64_t total = 1ULL;
  for (int i = 0; i < dims.numDims; ++i) {
    const int d = static_cast<int>(dims.d[i]);
    if (d <= 0) {
      return 0ULL;
    }
    total *= static_cast<uint64_t>(d);
  }
  return total;
}

std::string dims_to_string(const NvDsInferDims& dims) {
  std::string shape = "[";
  for (int i = 0; i < dims.numDims; ++i) {
    shape += std::to_string(static_cast<int>(dims.d[i]));
    if (i + 1 < dims.numDims) shape += ",";
  }
  shape += "]";
  return shape;
}

const NvDsInferLayerInfo* select_reid_layer(const NvDsInferTensorMeta* tensor_meta,
                                            const std::string& layer_name,
                                            int expected_dim) {
  if (!tensor_meta || tensor_meta->num_output_layers == 0U || !tensor_meta->output_layers_info) {
    return nullptr;
  }

  const NvDsInferLayerInfo* selected = nullptr;

  if (!layer_name.empty()) {
    for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
      const auto& layer = tensor_meta->output_layers_info[i];
      if (layer.layerName && layer_name == std::string(layer.layerName)) {
        selected = &layer;
        break;
      }
    }
  }
  if (selected) {
    return selected;
  }

  if (tensor_meta->num_output_layers == 1U) {
    return &tensor_meta->output_layers_info[0];
  }

  const NvDsInferLayerInfo* best = nullptr;
  int best_score = std::numeric_limits<int>::max();
  for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
    const auto& layer = tensor_meta->output_layers_info[i];
    const uint64_t elems = dims_num_elements(layer.inferDims);
    if (elems == 0ULL) continue;
    if (expected_dim > 0 && static_cast<int>(elems) % expected_dim == 0) {
      const int score = std::abs(static_cast<int>(elems) - expected_dim);
      if (score < best_score) {
        best_score = score;
        best = &layer;
      }
    }
  }
  if (best) {
    return best;
  }
  return &tensor_meta->output_layers_info[0];
}

const float* layer_host_ptr(const NvDsInferTensorMeta* tensor_meta, const NvDsInferLayerInfo& layer) {
  if (!tensor_meta) return nullptr;
  if (layer.dataType != FLOAT) return nullptr;

  const float* base = static_cast<const float*>(layer.buffer);
  if (tensor_meta->out_buf_ptrs_host && tensor_meta->output_layers_info) {
    const size_t idx = static_cast<size_t>(&layer - tensor_meta->output_layers_info);
    if (idx < static_cast<size_t>(tensor_meta->num_output_layers) && tensor_meta->out_buf_ptrs_host[idx]) {
      base = static_cast<const float*>(tensor_meta->out_buf_ptrs_host[idx]);
    }
  }
  return base;
}

std::optional<py::array_t<float>> extract_reid_embedding_impl(const deepstream::ObjectMetadata& obj_meta,
                                                              int gie_id,
                                                              const std::string& layer_name,
                                                              int expected_dim,
                                                              bool normalize) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) return std::nullopt;

  for (GList* node = obj->obj_user_meta_list; node != nullptr; node = node->next) {
    auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
    if (!user_meta || user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
      continue;
    }
    auto* tensor_meta = static_cast<NvDsInferTensorMeta*>(user_meta->user_meta_data);
    if (!tensor_meta) continue;
    if (tensor_meta->unique_id != static_cast<guint>(gie_id)) continue;

    const NvDsInferLayerInfo* layer = select_reid_layer(tensor_meta, layer_name, expected_dim);
    if (!layer) continue;
    const float* src = layer_host_ptr(tensor_meta, *layer);
    if (!src) continue;
    const uint64_t total_u64 = dims_num_elements(layer->inferDims);
    if (total_u64 == 0ULL || total_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) continue;
    const int total = static_cast<int>(total_u64);

    int emb_dim = total;
    if (expected_dim > 0) {
      if (total == expected_dim) {
        emb_dim = expected_dim;
      } else if (total > expected_dim && (total % expected_dim) == 0) {
        emb_dim = expected_dim;
      }
    }
    if (emb_dim <= 0) continue;

    std::vector<float> out(static_cast<size_t>(emb_dim));
    std::copy(src, src + emb_dim, out.begin());

    if (normalize) {
      double sum_sq = 0.0;
      for (float v : out) {
        if (!std::isfinite(v)) {
          sum_sq = 0.0;
          break;
        }
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
      }
      if (!(sum_sq > 0.0) || !std::isfinite(sum_sq)) {
        continue;
      }
      const float inv_norm = static_cast<float>(1.0 / std::sqrt(sum_sq));
      for (float& v : out) {
        v *= inv_norm;
      }
    }

    if (reid_meta_ext_debug_enabled()) {
      const auto shape = dims_to_string(layer->inferDims);
      g_print(
          "[noesis_reid_meta_ext] uid=%u layer=%s shape=%s emb_dim=%d normalize=%d\n",
          tensor_meta->unique_id,
          layer->layerName ? layer->layerName : "(null)",
          shape.c_str(),
          emb_dim,
          normalize ? 1 : 0);
    }
    py::array_t<float> arr(out.size(), out.data());
    return arr;
  }
  return std::nullopt;
}

}  // namespace

py::object extract_reid_embedding(const deepstream::ObjectMetadata& obj_meta,
                                  int gie_id,
                                  const std::string& layer_name,
                                  int expected_dim,
                                  bool normalize) {
  auto emb = extract_reid_embedding_impl(obj_meta, gie_id, layer_name, expected_dim, normalize);
  if (!emb.has_value()) {
    return py::none();
  }
  return emb.value();
}

PYBIND11_MODULE(noesis_reid_meta_ext, m) {
  m.doc() = "Noesis DS8 helper bindings for extracting ReID embeddings from object tensor metadata.";
  m.def(
      "extract_reid_embedding",
      &extract_reid_embedding,
      py::arg("obj_meta"),
      py::arg("gie_id") = 3,
      py::arg("layer_name") = "features",
      py::arg("expected_dim") = 512,
      py::arg("normalize") = true,
      "Extract ReID embedding from NVDSINFER_TENSOR_OUTPUT_META for one object. "
      "Returns a float32 numpy array or None.");
}
