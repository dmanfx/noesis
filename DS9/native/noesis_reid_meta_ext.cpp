#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    throw std::runtime_error("ReID tensor must be FLOAT32");
  }
  if (!tensor_is_contiguous(tensor)) {
    throw std::runtime_error("ReID tensor must be contiguous");
  }
  const auto count_opt = tensor_num_elements(tensor);
  if (!count_opt.has_value() || *count_opt == 0ULL ||
      *count_opt > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
    throw std::runtime_error("ReID tensor shape is invalid");
  }
  const size_t count = static_cast<size_t>(*count_opt);
  const uint64_t expected_bytes = static_cast<uint64_t>(count) * sizeof(float);
  if (tensor.size() != expected_bytes || tensor.data() == nullptr) {
    throw std::runtime_error("ReID tensor storage does not match its public shape");
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
          "cudaMemcpy ReID tensor failed");
      break;
    }
    default:
      throw std::runtime_error("ReID tensor has no supported device");
  }
  return values;
}

deepstream::Tensor* select_reid_layer(
    const std::unordered_map<std::string, deepstream::Tensor*>& layers,
    const std::string& layer_name) {
  if (layer_name.empty()) return nullptr;
  const auto named = layers.find(layer_name);
  return named != layers.end() ? named->second : nullptr;
}

std::optional<std::vector<float>> extract_reid_embedding_impl(
    const deepstream::ObjectMetadata& obj_meta,
    int gie_id,
    const std::string& layer_name,
    int expected_dim,
    bool normalize) {
  if (gie_id < 0 || layer_name.empty() || expected_dim <= 0) {
    return std::nullopt;
  }

  std::optional<std::vector<float>> result;
  std::optional<std::string> matched_error;

  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (result.has_value()) return;
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta || tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }

        try {
          TensorLayerMap owned_layers(tensor_meta);
          deepstream::Tensor* selected =
              select_reid_layer(owned_layers.get(), layer_name);
          if (selected == nullptr) {
            matched_error = "matching ReID tensor metadata is missing the requested output layer";
            return;
          }
          std::vector<float> values = copy_float32_tensor_to_host(*selected);
          if (values.size() != static_cast<size_t>(expected_dim)) {
            matched_error = "matching ReID tensor does not have the required embedding dimension";
            return;
          }

          double sum_sq = 0.0;
          for (float value : values) {
            if (!std::isfinite(value)) {
              matched_error = "matching ReID tensor contains a non-finite embedding";
              return;
            }
            if (normalize) {
              sum_sq += static_cast<double>(value) * static_cast<double>(value);
            }
          }
          if (normalize) {
            if (!(sum_sq > 0.0) || !std::isfinite(sum_sq)) {
              matched_error = "matching ReID tensor has no finite nonzero embedding";
              return;
            }
            const float inv_norm = static_cast<float>(1.0 / std::sqrt(sum_sq));
            for (float& value : values) value *= inv_norm;
          }

          if (reid_meta_ext_debug_enabled()) {
            std::fprintf(
                stderr,
                "[noesis_reid_meta_ext] uid=%u shape=%s dtype=%d bits=%u device=%d emb_dim=%d normalize=%d\n",
                tensor_meta.uniqueId(),
                tensor_shape_string(*selected).c_str(),
                static_cast<int>(selected->dtype()),
                selected->bits(),
                static_cast<int>(selected->deviceType()),
                expected_dim,
                normalize ? 1 : 0);
          }
          result = std::move(values);
        } catch (const std::exception& exc) {
          matched_error = exc.what();
        }
      },
      NVDSINFER_TENSOR_OUTPUT_META);

  if (!result.has_value() && matched_error.has_value() && reid_meta_ext_debug_enabled()) {
    std::fprintf(stderr, "[noesis_reid_meta_ext] %s\n", matched_error->c_str());
  }
  return result;
}

}  // namespace

py::object extract_reid_embedding(
    const deepstream::ObjectMetadata& obj_meta,
    int gie_id,
    const std::string& layer_name,
    int expected_dim,
    bool normalize) {
  auto embedding = extract_reid_embedding_impl(
      obj_meta, gie_id, layer_name, expected_dim, normalize);
  if (!embedding.has_value()) {
    return py::none();
  }
  py::array_t<float> array(embedding->size());
  std::memcpy(
      array.mutable_data(), embedding->data(), embedding->size() * sizeof(float));
  return std::move(array);
}

PYBIND11_MODULE(noesis_reid_meta_ext, m) {
  m.doc() =
      "Noesis DS9 public-Service-Maker helper for extracting ReID embeddings "
      "from object tensor metadata.";
  m.def(
      "extract_reid_embedding",
      &extract_reid_embedding,
      py::arg("obj_meta"),
      py::arg("gie_id") = 3,
      py::arg("layer_name") = "fc_pred",
      py::arg("expected_dim") = 256,
      py::arg("normalize") = true,
      "Extract a ReID embedding through TensorOutputUserMetadata/Tensor. "
      "Returns a copied float32 numpy array or None.");
}
