#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <glib.h>
#include <gst/gst.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>

#include "gstnvdsinfer.h"
#include "metadata.hpp"
#include "nvdsmeta.h"

namespace py = pybind11;

extern "C" cudaError_t noesis_sample_roi_values_cuda(
    const float* depth,
    int frame_w,
    int frame_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float* out_values,
    float* out_center,
    cudaStream_t stream);

extern "C" cudaError_t noesis_sample_masked_roi_values_cuda(
    const float* depth,
    const uint8_t* mask,
    int frame_w,
    int frame_h,
    int mask_w,
    int mask_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float* out_values,
    float* out_center,
    cudaStream_t stream);

extern "C" cudaError_t noesis_sample_masked_person_roi_values_cuda(
    const float* depth,
    const uint8_t* mask,
    int frame_w,
    int frame_h,
    int mask_w,
    int mask_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float* out_all,
    float* out_lower,
    float* out_torso,
    float* out_center,
    cudaStream_t stream);

namespace {

struct FrameMetaAccessor : public deepstream::FrameMetadata {
  using deepstream::Metadata::data_;
};

NvDsFrameMeta* unwrap_frame_meta(const deepstream::FrameMetadata& frame_meta) {
  auto* accessor = reinterpret_cast<const FrameMetaAccessor*>(&frame_meta);
  return reinterpret_cast<NvDsFrameMeta*>(accessor->data_);
}

std::vector<int> layer_shape(const NvDsInferLayerInfo& layer) {
  std::vector<int> dims;
  const NvDsInferDims& infer_dims = layer.inferDims;
  for (int i = 0; i < infer_dims.numDims; ++i) {
    const int value = static_cast<int>(infer_dims.d[i]);
    if (value <= 0) continue;
    dims.push_back(value);
  }
  return dims;
}

std::optional<std::pair<int, int>> depth_hw(const NvDsInferLayerInfo& layer) {
  const std::vector<int> dims = layer_shape(layer);
  if (dims.size() == 2U) {
    return std::make_pair(dims[0], dims[1]);
  }
  if (dims.size() == 3U) {
    if (dims[0] == 1) {
      return std::make_pair(dims[1], dims[2]);
    }
    if (dims[2] == 1) {
      return std::make_pair(dims[0], dims[1]);
    }
  }
  if (dims.size() == 4U && dims[0] == 1 && dims[1] == 1) {
    return std::make_pair(dims[2], dims[3]);
  }
  return std::nullopt;
}

std::optional<size_t> depth_layer_index(const NvDsInferTensorMeta* tensor_meta) {
  if (!tensor_meta || tensor_meta->num_output_layers == 0U || !tensor_meta->output_layers_info) {
    return std::nullopt;
  }

  const std::vector<std::string> preferred_names = {"depth", "pred", "output"};
  for (const std::string& target : preferred_names) {
    for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
      const NvDsInferLayerInfo& layer = tensor_meta->output_layers_info[i];
      const char* raw_name = layer.layerName;
      if (raw_name && target == raw_name) {
        return static_cast<size_t>(i);
      }
    }
  }

  if (tensor_meta->num_output_layers == 1U) {
    return 0U;
  }

  for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
    const NvDsInferLayerInfo& layer = tensor_meta->output_layers_info[i];
    if (depth_hw(layer).has_value()) {
      return static_cast<size_t>(i);
    }
  }
  return std::nullopt;
}

void throw_on_cuda(cudaError_t status, const char* what);

size_t layer_element_count(const NvDsInferLayerInfo& layer) {
  size_t count = 1U;
  const NvDsInferDims& infer_dims = layer.inferDims;
  for (int i = 0; i < infer_dims.numDims; ++i) {
    const int value = static_cast<int>(infer_dims.d[i]);
    if (value <= 0) continue;
    count *= static_cast<size_t>(value);
  }
  return count;
}

py::array_t<float> copy_tensor_layer_to_numpy(const NvDsInferTensorMeta* tensor_meta, size_t layer_idx) {
  if (!tensor_meta || layer_idx >= tensor_meta->num_output_layers || !tensor_meta->output_layers_info) {
    throw std::runtime_error("Invalid tensor layer index");
  }
  const NvDsInferLayerInfo& layer = tensor_meta->output_layers_info[layer_idx];
  std::vector<int> dims = layer_shape(layer);
  const size_t count = layer_element_count(layer);
  if (count == 0U) {
    throw std::runtime_error("Tensor layer has zero elements");
  }
  if (dims.empty()) {
    dims.push_back(static_cast<int>(count));
  }

  std::vector<py::ssize_t> shape;
  shape.reserve(dims.size());
  for (const int dim : dims) {
    shape.push_back(static_cast<py::ssize_t>(std::max(1, dim)));
  }

  py::array_t<float> out(shape);
  float* dst = static_cast<float*>(out.mutable_data());
  const void* src_host = tensor_meta->out_buf_ptrs_host ? tensor_meta->out_buf_ptrs_host[layer_idx] : nullptr;
  const void* src_dev = tensor_meta->out_buf_ptrs_dev ? tensor_meta->out_buf_ptrs_dev[layer_idx] : nullptr;

  switch (layer.dataType) {
    case FLOAT:
      if (src_host) {
        std::memcpy(dst, src_host, count * sizeof(float));
      } else if (src_dev) {
        throw_on_cuda(
            cudaMemcpy(dst, src_dev, count * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy tensor layer FLOAT failed");
      } else {
        throw std::runtime_error("Tensor layer FLOAT buffer is unavailable");
      }
      break;
    case HALF: {
      std::vector<uint16_t> half_values(count);
      if (src_host) {
        std::memcpy(half_values.data(), src_host, count * sizeof(uint16_t));
      } else if (src_dev) {
        throw_on_cuda(
            cudaMemcpy(half_values.data(), src_dev, count * sizeof(uint16_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy tensor layer HALF failed");
      } else {
        throw std::runtime_error("Tensor layer HALF buffer is unavailable");
      }
      for (size_t i = 0; i < count; ++i) {
        __half h;
        std::memcpy(&h, &half_values[i], sizeof(uint16_t));
        dst[i] = __half2float(h);
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported tensor layer dtype");
  }
  return out;
}

void throw_on_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void throw_on_npp(NppStatus status, const char* what) {
  if (status == NPP_SUCCESS) return;
  throw std::runtime_error(std::string(what) + ": NPP status " + std::to_string(static_cast<int>(status)));
}

NppStreamContext default_npp_stream_context() {
  NppStreamContext ctx{};
  int device_id = 0;
  cudaDeviceProp props{};
  unsigned int stream_flags = 0U;
  throw_on_cuda(cudaGetDevice(&device_id), "cudaGetDevice failed");
  throw_on_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties failed");
  throw_on_cuda(cudaStreamGetFlags(nullptr, &stream_flags), "cudaStreamGetFlags failed");
  ctx.hStream = nullptr;
  ctx.nCudaDeviceId = device_id;
  ctx.nMultiProcessorCount = props.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor = props.major;
  ctx.nCudaDevAttrComputeCapabilityMinor = props.minor;
  ctx.nStreamFlags = stream_flags;
  ctx.nReserved0 = 0;
  return ctx;
}

const uint8_t* binary_mask_data(
    const py::array& mask_array,
    int expected_h,
    int expected_w,
    float threshold,
    std::vector<uint8_t>& storage,
    const char* label) {
  py::buffer_info mask_info = mask_array.request();
  if (mask_info.ndim != 2) {
    throw std::runtime_error(std::string(label) + " mask must be 2D");
  }
  const int mask_h = static_cast<int>(mask_info.shape[0]);
  const int mask_w = static_cast<int>(mask_info.shape[1]);
  if (mask_w != expected_w || mask_h != expected_h) {
    throw std::runtime_error(std::string(label) + " mask shape must match ROI size");
  }
  if (mask_info.ptr == nullptr) {
    throw std::runtime_error(std::string(label) + " mask buffer is unavailable");
  }

  const size_t item_size = static_cast<size_t>(std::max<py::ssize_t>(1, mask_info.itemsize));
  const bool c_contiguous =
      mask_info.strides.size() >= 2U &&
      static_cast<size_t>(mask_info.strides[1]) == item_size &&
      static_cast<size_t>(mask_info.strides[0]) == item_size * static_cast<size_t>(mask_w);
  const std::string format = mask_info.format;
  const bool byte_mask =
      item_size == 1U &&
      (format == py::format_descriptor<uint8_t>::format() || format == py::format_descriptor<bool>::format());
  if (c_contiguous && byte_mask) {
    return static_cast<const uint8_t*>(mask_info.ptr);
  }

  py::array_t<float, py::array::c_style | py::array::forcecast> mask_float =
      py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(mask_array);
  if (!mask_float) {
    throw std::runtime_error(std::string(label) + " mask could not be converted to float for thresholding");
  }
  py::buffer_info float_info = mask_float.request();
  if (float_info.ndim != 2 ||
      static_cast<int>(float_info.shape[0]) != expected_h ||
      static_cast<int>(float_info.shape[1]) != expected_w ||
      float_info.ptr == nullptr) {
    throw std::runtime_error(std::string(label) + " mask could not be converted to a contiguous 2D buffer");
  }
  const float* src = static_cast<const float*>(float_info.ptr);
  const size_t count = static_cast<size_t>(expected_w) * static_cast<size_t>(expected_h);
  storage.resize(count);
  for (size_t idx = 0; idx < count; ++idx) {
    storage[idx] = src[idx] > threshold ? static_cast<uint8_t>(1U) : static_cast<uint8_t>(0U);
  }
  return storage.data();
}

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  explicit CudaBuffer(size_t count) { allocate(count); }
  ~CudaBuffer() { reset(); }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0U;
  }

  CudaBuffer& operator=(CudaBuffer&& other) noexcept {
    if (this == &other) return *this;
    reset();
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0U;
    return *this;
  }

  void allocate(size_t count) {
    reset();
    if (count == 0U) return;
    throw_on_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "cudaMalloc failed");
    count_ = count;
  }

  void ensure_capacity(size_t count) {
    if (count_ >= count) return;
    allocate(count);
  }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0U;
    }
  }

  T* get() const { return ptr_; }
  size_t count() const { return count_; }

  T* release() noexcept {
    T* raw = ptr_;
    ptr_ = nullptr;
    count_ = 0U;
    return raw;
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0U;
};

class AlignedDepthFrameDevice {
 public:
  AlignedDepthFrameDevice(float* device_ptr, int frame_w, int frame_h, int depth_w, int depth_h)
      : device_ptr_(device_ptr), frame_w_(frame_w), frame_h_(frame_h), depth_w_(depth_w), depth_h_(depth_h) {}

  ~AlignedDepthFrameDevice() {
    if (device_ptr_ != nullptr) {
      cudaFree(device_ptr_);
      device_ptr_ = nullptr;
    }
  }

  AlignedDepthFrameDevice(const AlignedDepthFrameDevice&) = delete;
  AlignedDepthFrameDevice& operator=(const AlignedDepthFrameDevice&) = delete;

  py::array_t<float> copy_roi_to_numpy(int left, int top, int width, int height) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("copy_roi_to_numpy requires a positive ROI size");
    }

    const int x0 = std::max(0, std::min(left, frame_w_));
    const int y0 = std::max(0, std::min(top, frame_h_));
    const int x1 = std::max(x0, std::min(left + width, frame_w_));
    const int y1 = std::max(y0, std::min(top + height, frame_h_));
    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;
    if (roi_w <= 0 || roi_h <= 0) {
      throw std::runtime_error("copy_roi_to_numpy resolved an empty ROI");
    }

    py::array_t<float> out({roi_h, roi_w});
    float* dst = static_cast<float*>(out.mutable_data());
    const float* src = device_ptr_ + (static_cast<size_t>(y0) * static_cast<size_t>(frame_w_)) + static_cast<size_t>(x0);
    throw_on_cuda(
        cudaMemcpy2D(
            dst,
            static_cast<size_t>(roi_w) * sizeof(float),
            src,
            static_cast<size_t>(frame_w_) * sizeof(float),
            static_cast<size_t>(roi_w) * sizeof(float),
            static_cast<size_t>(roi_h),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy2D depth ROI copy failed");
    return out;
  }

  py::dict sample_roi_stats(int left, int top, int width, int height, int max_samples) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("sample_roi_stats requires a positive ROI size");
    }

    const int x0 = std::max(0, std::min(left, frame_w_));
    const int y0 = std::max(0, std::min(top, frame_h_));
    const int x1 = std::max(x0, std::min(left + width, frame_w_));
    const int y1 = std::max(y0, std::min(top + height, frame_h_));
    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;
    if (roi_w <= 0 || roi_h <= 0) {
      throw std::runtime_error("sample_roi_stats resolved an empty ROI");
    }

    const int sample_cap = std::max(128, max_samples);
    const double total_px = static_cast<double>(roi_w) * static_cast<double>(roi_h);
    const int stride = std::max(1, static_cast<int>(std::ceil(std::sqrt(total_px / static_cast<double>(sample_cap)))));
    const int sampled_rows = static_cast<int>((roi_h + stride - 1) / stride);
    const int sampled_cols = static_cast<int>((roi_w + stride - 1) / stride);
    const int sampled_area = std::max(1, sampled_rows * sampled_cols);

    const size_t sampled_count = static_cast<size_t>(sampled_area) + 1U;
    thread_local CudaBuffer<float> sampled_device;
    sampled_device.ensure_capacity(sampled_count);
    throw_on_cuda(
        noesis_sample_roi_values_cuda(
            device_ptr_,
            frame_w_,
            frame_h_,
            x0,
            y0,
            roi_w,
            roi_h,
            stride,
            sampled_cols,
            sampled_area,
            sampled_device.get(),
            sampled_device.get() + static_cast<size_t>(sampled_area),
            nullptr),
        "CUDA depth ROI compact sampler launch failed");

    thread_local std::vector<float> sampled_host;
    sampled_host.resize(sampled_count);
    throw_on_cuda(
        cudaMemcpy(
            sampled_host.data(),
            sampled_device.get(),
            sampled_count * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy depth ROI compact samples failed");

    thread_local std::vector<float> values;
    values.clear();
    values.reserve(static_cast<size_t>(sampled_area));
    for (int idx = 0; idx < sampled_area; ++idx) {
      const float value = sampled_host[static_cast<size_t>(idx)];
      if (std::isfinite(value)) {
        values.push_back(value);
      }
    }

    const float center_value = sampled_host[static_cast<size_t>(sampled_area)];

    py::dict out;
    out["roi_area_px"] = roi_w * roi_h;
    out["sampled_area_px"] = sampled_area;
    out["sample_count"] = static_cast<int>(values.size());
    out["valid_fraction"] = static_cast<double>(values.size()) / static_cast<double>(sampled_area);
    if (std::isfinite(center_value)) {
      out["depth_center"] = static_cast<double>(center_value);
    } else {
      out["depth_center"] = py::none();
    }
    if (values.empty()) {
      out["depth_median"] = py::none();
      out["depth_mean"] = py::none();
      out["depth_p10"] = py::none();
      out["depth_p90"] = py::none();
      out["depth_min"] = py::none();
      out["depth_max"] = py::none();
      return out;
    }

    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values) {
      sum += static_cast<double>(value);
    }
    auto percentile = [](const std::vector<float>& sorted_values, double p) -> double {
      if (sorted_values.empty()) return std::numeric_limits<double>::quiet_NaN();
      const double clamped = std::max(0.0, std::min(100.0, p));
      const double pos = (clamped / 100.0) * static_cast<double>(sorted_values.size() - 1U);
      const size_t lo = static_cast<size_t>(std::floor(pos));
      const size_t hi = static_cast<size_t>(std::ceil(pos));
      if (lo == hi) return static_cast<double>(sorted_values[lo]);
      const double frac = pos - static_cast<double>(lo);
      return (static_cast<double>(sorted_values[lo]) * (1.0 - frac)) + (static_cast<double>(sorted_values[hi]) * frac);
    };

    out["depth_median"] = percentile(values, 50.0);
    out["depth_mean"] = sum / static_cast<double>(values.size());
    out["depth_p10"] = percentile(values, 10.0);
    out["depth_p90"] = percentile(values, 90.0);
    out["depth_min"] = static_cast<double>(values.front());
    out["depth_max"] = static_cast<double>(values.back());
    return out;
  }

  py::dict sample_masked_roi_stats(
      int left,
      int top,
      int width,
      int height,
      py::array mask_array,
      float threshold,
      int max_samples) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("sample_masked_roi_stats requires a positive ROI size");
    }

    const int x0 = std::max(0, std::min(left, frame_w_));
    const int y0 = std::max(0, std::min(top, frame_h_));
    const int x1 = std::max(x0, std::min(left + width, frame_w_));
    const int y1 = std::max(y0, std::min(top + height, frame_h_));
    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;
    if (roi_w <= 0 || roi_h <= 0) {
      throw std::runtime_error("sample_masked_roi_stats resolved an empty ROI");
    }

    const int mask_h = roi_h;
    const int mask_w = roi_w;
    thread_local std::vector<uint8_t> mask_binary;
    const uint8_t* mask_host = binary_mask_data(
        mask_array,
        mask_h,
        mask_w,
        threshold,
        mask_binary,
        "sample_masked_roi_stats");

    const int sample_cap = std::max(128, max_samples);
    int mask_area = 0;
    for (int idx = 0; idx < mask_w * mask_h; ++idx) {
      if (mask_host[idx] != 0U) {
        ++mask_area;
      }
    }
    const double sampling_area = static_cast<double>(std::max(1, mask_area));
    const int stride = std::max(1, static_cast<int>(std::ceil(std::sqrt(sampling_area / static_cast<double>(sample_cap)))));
    const int sampled_rows = static_cast<int>((roi_h + stride - 1) / stride);
    const int sampled_cols = static_cast<int>((roi_w + stride - 1) / stride);
    const int sampled_area = std::max(1, sampled_rows * sampled_cols);

    int sampled_mask_area = 0;
    for (int yy = 0; yy < roi_h; yy += stride) {
      for (int xx = 0; xx < roi_w; xx += stride) {
        if (mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] != 0U) {
          ++sampled_mask_area;
        }
      }
    }

    const size_t mask_count = static_cast<size_t>(mask_w) * static_cast<size_t>(mask_h);
    thread_local CudaBuffer<uint8_t> mask_device;
    mask_device.ensure_capacity(mask_count);
    throw_on_cuda(
        cudaMemcpy(
            mask_device.get(),
            mask_host,
            mask_count * sizeof(uint8_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy object-depth mask to device failed");

    const size_t sampled_count = static_cast<size_t>(sampled_area) + 1U;
    thread_local CudaBuffer<float> sampled_device;
    sampled_device.ensure_capacity(sampled_count);
    throw_on_cuda(
        noesis_sample_masked_roi_values_cuda(
            device_ptr_,
            mask_device.get(),
            frame_w_,
            frame_h_,
            mask_w,
            mask_h,
            x0,
            y0,
            roi_w,
            roi_h,
            stride,
            sampled_cols,
            sampled_area,
            sampled_device.get(),
            sampled_device.get() + static_cast<size_t>(sampled_area),
            nullptr),
        "CUDA masked depth ROI compact sampler launch failed");

    thread_local std::vector<float> sampled_host;
    sampled_host.resize(sampled_count);
    throw_on_cuda(
        cudaMemcpy(
            sampled_host.data(),
            sampled_device.get(),
            sampled_count * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy masked depth ROI compact samples failed");

    thread_local std::vector<float> values;
    values.clear();
    values.reserve(static_cast<size_t>(std::max(0, sampled_mask_area)));
    for (int idx = 0; idx < sampled_area; ++idx) {
      const float value = sampled_host[static_cast<size_t>(idx)];
      if (std::isfinite(value)) {
        values.push_back(value);
      }
    }

    const float center_value = sampled_host[static_cast<size_t>(sampled_area)];

    py::dict out;
    out["roi_area_px"] = roi_w * roi_h;
    out["mask_area_px"] = mask_area;
    out["sampled_area_px"] = sampled_area;
    out["sampled_mask_area_px"] = sampled_mask_area;
    out["sample_count"] = static_cast<int>(values.size());
    out["valid_fraction"] = sampled_mask_area > 0
        ? static_cast<double>(values.size()) / static_cast<double>(sampled_mask_area)
        : 0.0;
    if (std::isfinite(center_value)) {
      out["depth_center"] = static_cast<double>(center_value);
    } else {
      out["depth_center"] = py::none();
    }
    if (values.empty()) {
      out["depth_median"] = py::none();
      out["depth_mean"] = py::none();
      out["depth_p10"] = py::none();
      out["depth_p90"] = py::none();
      out["depth_min"] = py::none();
      out["depth_max"] = py::none();
      return out;
    }

    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values) {
      sum += static_cast<double>(value);
    }
    auto percentile = [](const std::vector<float>& sorted_values, double p) -> double {
      if (sorted_values.empty()) return std::numeric_limits<double>::quiet_NaN();
      const double clamped = std::max(0.0, std::min(100.0, p));
      const double pos = (clamped / 100.0) * static_cast<double>(sorted_values.size() - 1U);
      const size_t lo = static_cast<size_t>(std::floor(pos));
      const size_t hi = static_cast<size_t>(std::ceil(pos));
      if (lo == hi) return static_cast<double>(sorted_values[lo]);
      const double frac = pos - static_cast<double>(lo);
      return (static_cast<double>(sorted_values[lo]) * (1.0 - frac)) + (static_cast<double>(sorted_values[hi]) * frac);
    };

    out["depth_median"] = percentile(values, 50.0);
    out["depth_mean"] = sum / static_cast<double>(values.size());
    out["depth_p10"] = percentile(values, 10.0);
    out["depth_p90"] = percentile(values, 90.0);
    out["depth_min"] = static_cast<double>(values.front());
    out["depth_max"] = static_cast<double>(values.back());
    return out;
  }

  py::dict sample_masked_person_roi_stats(
      int left,
      int top,
      int width,
      int height,
      py::array mask_array,
      float threshold,
      int max_samples) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("sample_masked_person_roi_stats requires a positive ROI size");
    }

    const int x0 = std::max(0, std::min(left, frame_w_));
    const int y0 = std::max(0, std::min(top, frame_h_));
    const int x1 = std::max(x0, std::min(left + width, frame_w_));
    const int y1 = std::max(y0, std::min(top + height, frame_h_));
    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;
    if (roi_w <= 0 || roi_h <= 0) {
      throw std::runtime_error("sample_masked_person_roi_stats resolved an empty ROI");
    }

    const int mask_h = roi_h;
    const int mask_w = roi_w;
    thread_local std::vector<uint8_t> mask_binary;
    const uint8_t* mask_host = binary_mask_data(
        mask_array,
        mask_h,
        mask_w,
        threshold,
        mask_binary,
        "sample_masked_person_roi_stats");

    auto in_center_band_host = [](int x, int y, int roi_w_value, int roi_h_value, double y0_ratio, double y1_ratio, double center_width_ratio) -> bool {
      int band_y0 = static_cast<int>(std::floor(static_cast<double>(roi_h_value) * y0_ratio));
      int band_y1 = static_cast<int>(std::ceil(static_cast<double>(roi_h_value) * y1_ratio));
      band_y0 = std::max(0, std::min(roi_h_value, band_y0));
      band_y1 = std::max(band_y0 + 1, std::min(roi_h_value, band_y1));
      int band_w = static_cast<int>(std::round(static_cast<double>(roi_w_value) * center_width_ratio));
      band_w = std::max(1, std::min(roi_w_value, band_w));
      const double center_x = static_cast<double>(roi_w_value) * 0.5;
      int band_x0 = static_cast<int>(std::round(center_x - (static_cast<double>(band_w) * 0.5)));
      int band_x1 = static_cast<int>(std::round(center_x + (static_cast<double>(band_w) * 0.5)));
      band_x0 = std::max(0, std::min(roi_w_value, band_x0));
      band_x1 = std::max(band_x0 + 1, std::min(roi_w_value, band_x1));
      return y >= band_y0 && y < band_y1 && x >= band_x0 && x < band_x1;
    };
    auto eroded_host = [mask_host, mask_w, mask_h](int x, int y) -> bool {
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= mask_h) return false;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= mask_w) return false;
          if (mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] == 0U) {
            return false;
          }
        }
      }
      return true;
    };

    int mask_area = 0;
    int lower_mask_area = 0;
    int torso_mask_area = 0;
    int max_mask_y = -1;
    for (int yy = 0; yy < roi_h; ++yy) {
      for (int xx = 0; xx < roi_w; ++xx) {
        const bool active = mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] != 0U;
        if (!active) continue;
        ++mask_area;
        max_mask_y = std::max(max_mask_y, yy);
        const bool eroded = eroded_host(xx, yy);
        if (eroded && in_center_band_host(xx, yy, roi_w, roi_h, 0.88, 1.0, 0.35)) {
          ++lower_mask_area;
        }
        if (eroded && in_center_band_host(xx, yy, roi_w, roi_h, 0.35, 0.70, 0.50)) {
          ++torso_mask_area;
        }
      }
    }

    int foot_x = -1;
    int foot_y = -1;
    if (max_mask_y >= 0) {
      const int foot_band_top = std::max(0, max_mask_y - std::max(1, static_cast<int>(std::round(static_cast<double>(roi_h) * 0.12))));
      for (int yy = max_mask_y; yy >= foot_band_top && foot_y < 0; --yy) {
        int active_center_count = 0;
        for (int xx = 0; xx < roi_w; ++xx) {
          const bool active = mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] != 0U;
          if (active && in_center_band_host(xx, yy, roi_w, roi_h, 0.0, 1.0, 0.35)) {
            ++active_center_count;
          }
        }
        if (active_center_count <= 0) {
          continue;
        }
        const int target_lo = (active_center_count - 1) / 2;
        const int target_hi = active_center_count / 2;
        int seen = 0;
        int median_lo = -1;
        int median_hi = -1;
        for (int xx = 0; xx < roi_w; ++xx) {
          const bool active = mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] != 0U;
          if (!active || !in_center_band_host(xx, yy, roi_w, roi_h, 0.0, 1.0, 0.35)) {
            continue;
          }
          if (seen == target_lo) {
            median_lo = xx;
          }
          if (seen == target_hi) {
            median_hi = xx;
            break;
          }
          ++seen;
        }
        if (median_lo >= 0 && median_hi >= 0) {
          foot_x = (median_lo + median_hi) / 2;
          foot_y = yy;
        }
      }
    }

    const int sample_cap = std::max(128, max_samples);
    const double sampling_area = static_cast<double>(std::max(1, mask_area));
    const int stride = std::max(1, static_cast<int>(std::ceil(std::sqrt(sampling_area / static_cast<double>(sample_cap)))));
    const int sampled_rows = static_cast<int>((roi_h + stride - 1) / stride);
    const int sampled_cols = static_cast<int>((roi_w + stride - 1) / stride);
    const int sampled_area = std::max(1, sampled_rows * sampled_cols);

    int sampled_mask_area = 0;
    int sampled_lower_area = 0;
    int sampled_torso_area = 0;
    for (int yy = 0; yy < roi_h; yy += stride) {
      for (int xx = 0; xx < roi_w; xx += stride) {
        const bool active = mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] != 0U;
        if (!active) continue;
        ++sampled_mask_area;
        const bool eroded = eroded_host(xx, yy);
        if (eroded && in_center_band_host(xx, yy, roi_w, roi_h, 0.88, 1.0, 0.35)) {
          ++sampled_lower_area;
        }
        if (eroded && in_center_band_host(xx, yy, roi_w, roi_h, 0.35, 0.70, 0.50)) {
          ++sampled_torso_area;
        }
      }
    }

    const size_t mask_count = static_cast<size_t>(mask_w) * static_cast<size_t>(mask_h);
    thread_local CudaBuffer<uint8_t> mask_device;
    mask_device.ensure_capacity(mask_count);
    throw_on_cuda(
        cudaMemcpy(
            mask_device.get(),
            mask_host,
            mask_count * sizeof(uint8_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy person object-depth mask to device failed");

    const size_t sampled_area_size = static_cast<size_t>(sampled_area);
    const size_t sampled_count = (sampled_area_size * 3U) + 1U;
    thread_local CudaBuffer<float> sampled_device;
    sampled_device.ensure_capacity(sampled_count);
    float* all_device = sampled_device.get();
    float* lower_device = all_device + sampled_area_size;
    float* torso_device = lower_device + sampled_area_size;
    float* center_device = torso_device + sampled_area_size;
    throw_on_cuda(
        noesis_sample_masked_person_roi_values_cuda(
            device_ptr_,
            mask_device.get(),
            frame_w_,
            frame_h_,
            mask_w,
            mask_h,
            x0,
            y0,
            roi_w,
            roi_h,
            stride,
            sampled_cols,
            sampled_area,
            all_device,
            lower_device,
            torso_device,
            center_device,
            nullptr),
        "CUDA person masked depth ROI sampler launch failed");

    thread_local std::vector<float> sampled_host;
    sampled_host.resize(sampled_count);
    throw_on_cuda(
        cudaMemcpy(
            sampled_host.data(),
            sampled_device.get(),
            sampled_count * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy person masked depth ROI compact samples failed");

    auto collect_values = [sampled_area_size](const std::vector<float>& samples, size_t offset, int reserve_count) -> std::vector<float> {
      std::vector<float> values;
      values.reserve(static_cast<size_t>(std::max(0, reserve_count)));
      for (size_t idx = 0; idx < sampled_area_size; ++idx) {
        const float value = samples[offset + idx];
        if (std::isfinite(value)) {
          values.push_back(value);
        }
      }
      return values;
    };
    auto add_stats = [](py::dict& out, const char* prefix, std::vector<float>& values, int sampled_denominator) {
      const std::string key_prefix(prefix ? prefix : "");
      out[py::str(key_prefix + "sample_count")] = static_cast<int>(values.size());
      out[py::str(key_prefix + "valid_fraction")] = sampled_denominator > 0
          ? static_cast<double>(values.size()) / static_cast<double>(sampled_denominator)
          : 0.0;
      if (values.empty()) {
        out[py::str(key_prefix + "depth_median")] = py::none();
        out[py::str(key_prefix + "depth_mean")] = py::none();
        out[py::str(key_prefix + "depth_p10")] = py::none();
        out[py::str(key_prefix + "depth_p90")] = py::none();
        out[py::str(key_prefix + "depth_min")] = py::none();
        out[py::str(key_prefix + "depth_max")] = py::none();
        return;
      }
      std::sort(values.begin(), values.end());
      double sum = 0.0;
      for (float value : values) {
        sum += static_cast<double>(value);
      }
      auto percentile = [&values](double p) -> double {
        const double clamped = std::max(0.0, std::min(100.0, p));
        const double pos = (clamped / 100.0) * static_cast<double>(values.size() - 1U);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        if (lo == hi) return static_cast<double>(values[lo]);
        const double frac = pos - static_cast<double>(lo);
        return (static_cast<double>(values[lo]) * (1.0 - frac)) + (static_cast<double>(values[hi]) * frac);
      };
      out[py::str(key_prefix + "depth_median")] = percentile(50.0);
      out[py::str(key_prefix + "depth_mean")] = sum / static_cast<double>(values.size());
      out[py::str(key_prefix + "depth_p10")] = percentile(10.0);
      out[py::str(key_prefix + "depth_p90")] = percentile(90.0);
      out[py::str(key_prefix + "depth_min")] = static_cast<double>(values.front());
      out[py::str(key_prefix + "depth_max")] = static_cast<double>(values.back());
    };

    std::vector<float> all_values = collect_values(sampled_host, 0U, sampled_mask_area);
    std::vector<float> lower_values = collect_values(sampled_host, sampled_area_size, sampled_lower_area);
    std::vector<float> torso_values = collect_values(sampled_host, sampled_area_size * 2U, sampled_torso_area);
    const float center_value = sampled_host[sampled_area_size * 3U];

    py::dict out;
    out["roi_area_px"] = roi_w * roi_h;
    out["mask_area_px"] = mask_area;
    out["lower_mask_area_px"] = lower_mask_area;
    out["torso_mask_area_px"] = torso_mask_area;
    out["sampled_area_px"] = sampled_area;
    out["sampled_mask_area_px"] = sampled_mask_area;
    out["sampled_lower_mask_area_px"] = sampled_lower_area;
    out["sampled_torso_mask_area_px"] = sampled_torso_area;
    if (foot_x >= 0 && foot_y >= 0) {
      out["foot_u"] = static_cast<double>(x0 + foot_x);
      out["foot_v"] = static_cast<double>(y0 + foot_y);
    } else {
      out["foot_u"] = py::none();
      out["foot_v"] = py::none();
    }
    if (std::isfinite(center_value)) {
      out["depth_center"] = static_cast<double>(center_value);
    } else {
      out["depth_center"] = py::none();
    }
    add_stats(out, "", all_values, sampled_mask_area);
    add_stats(out, "lower_", lower_values, sampled_lower_area);
    add_stats(out, "torso_", torso_values, sampled_torso_area);
    return out;
  }

  int frame_width() const { return frame_w_; }
  int frame_height() const { return frame_h_; }
  int depth_width() const { return depth_w_; }
  int depth_height() const { return depth_h_; }

 private:
  float* device_ptr_ = nullptr;
  int frame_w_ = 0;
  int frame_h_ = 0;
  int depth_w_ = 0;
  int depth_h_ = 0;
};

std::shared_ptr<AlignedDepthFrameDevice> align_depth_layer_to_frame(
    const NvDsInferTensorMeta* tensor_meta,
    const NvDsInferLayerInfo& layer,
    size_t layer_idx,
    int frame_w,
    int frame_h) {
  if (frame_w <= 0 || frame_h <= 0) {
    throw std::runtime_error("Target frame size must be positive");
  }
  if (!tensor_meta->out_buf_ptrs_dev || !tensor_meta->out_buf_ptrs_dev[layer_idx]) {
    throw std::runtime_error("Depth tracking tensor device buffer is unavailable");
  }

  const auto hw = depth_hw(layer);
  if (!hw.has_value()) {
    throw std::runtime_error("Depth tracking tensor layer shape is unsupported");
  }
  const int depth_h = hw->first;
  const int depth_w = hw->second;
  if (depth_w <= 0 || depth_h <= 0) {
    throw std::runtime_error("Depth tracking tensor shape is invalid");
  }

  const void* src_device = tensor_meta->out_buf_ptrs_dev[layer_idx];
  CudaBuffer<float> src_float;
  const NppStreamContext npp_ctx = default_npp_stream_context();

  switch (layer.dataType) {
    case FLOAT:
      break;
    case HALF: {
      src_float.allocate(static_cast<size_t>(depth_w) * static_cast<size_t>(depth_h));
      throw_on_npp(
          nppiConvert_16f32f_C1R_Ctx(
              static_cast<const Npp16f*>(src_device),
              depth_w * static_cast<int>(sizeof(uint16_t)),
              static_cast<Npp32f*>(src_float.get()),
              depth_w * static_cast<int>(sizeof(float)),
              NppiSize{depth_w, depth_h},
              npp_ctx),
          "nppiConvert_16f32f_C1R failed");
      src_device = static_cast<const void*>(src_float.get());
      break;
    }
    default:
      throw std::runtime_error("Unsupported depth tracking tensor dtype");
  }

  CudaBuffer<float> aligned(static_cast<size_t>(frame_w) * static_cast<size_t>(frame_h));
  if (depth_w == frame_w && depth_h == frame_h) {
    throw_on_cuda(
        cudaMemcpy(
            aligned.get(),
            src_device,
            static_cast<size_t>(frame_w) * static_cast<size_t>(frame_h) * sizeof(float),
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy device depth frame copy failed");
  } else {
    throw_on_npp(
        nppiResize_32f_C1R_Ctx(
            static_cast<const Npp32f*>(src_device),
            depth_w * static_cast<int>(sizeof(float)),
            NppiSize{depth_w, depth_h},
            NppiRect{0, 0, depth_w, depth_h},
            static_cast<Npp32f*>(aligned.get()),
            frame_w * static_cast<int>(sizeof(float)),
            NppiSize{frame_w, frame_h},
            NppiRect{0, 0, frame_w, frame_h},
            NPPI_INTER_LINEAR,
            npp_ctx),
        "nppiResize_32f_C1R failed");
  }

  throw_on_cuda(cudaDeviceSynchronize(), "Depth frame GPU alignment synchronize failed");
  return std::make_shared<AlignedDepthFrameDevice>(aligned.release(), frame_w, frame_h, depth_w, depth_h);
}

}  // namespace

py::object capture_aligned_depth_frame(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id,
    int frame_w,
    int frame_h) {
  NvDsFrameMeta* frame = unwrap_frame_meta(frame_meta);
  if (!frame) return py::none();

  for (GList* node = frame->frame_user_meta_list; node != nullptr; node = node->next) {
    auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
    if (!user_meta || user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
      continue;
    }
    auto* tensor_meta = static_cast<NvDsInferTensorMeta*>(user_meta->user_meta_data);
    if (!tensor_meta || tensor_meta->unique_id != static_cast<guint>(gie_id)) {
      continue;
    }
    const auto layer_idx = depth_layer_index(tensor_meta);
    if (!layer_idx.has_value()) {
      continue;
    }
    const NvDsInferLayerInfo& layer = tensor_meta->output_layers_info[*layer_idx];
    return py::cast(align_depth_layer_to_frame(tensor_meta, layer, *layer_idx, frame_w, frame_h));
  }
  return py::none();
}

py::object capture_tensor_layers(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id) {
  NvDsFrameMeta* frame = unwrap_frame_meta(frame_meta);
  if (!frame) return py::none();

  for (GList* node = frame->frame_user_meta_list; node != nullptr; node = node->next) {
    auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
    if (!user_meta || user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
      continue;
    }
    auto* tensor_meta = static_cast<NvDsInferTensorMeta*>(user_meta->user_meta_data);
    if (!tensor_meta || tensor_meta->unique_id != static_cast<guint>(gie_id)) {
      continue;
    }

    py::dict layers;
    for (guint i = 0; i < tensor_meta->num_output_layers; ++i) {
      const NvDsInferLayerInfo& layer = tensor_meta->output_layers_info[i];
      const char* raw_name = layer.layerName;
      std::string name = raw_name && raw_name[0] ? raw_name : ("layer_" + std::to_string(i));
      layers[py::str(name)] = copy_tensor_layer_to_numpy(tensor_meta, static_cast<size_t>(i));
    }
    return std::move(layers);
  }
  return py::none();
}

PYBIND11_MODULE(noesis_depth_tracking_tensor_ext, m) {
  m.doc() = "Noesis DS8 helper bindings for extracting and aligning baseline DAv2 tensors on-device.";
  py::class_<AlignedDepthFrameDevice, std::shared_ptr<AlignedDepthFrameDevice>>(m, "AlignedDepthFrameDevice")
      .def("copy_roi_to_numpy", &AlignedDepthFrameDevice::copy_roi_to_numpy, py::arg("left"), py::arg("top"), py::arg("width"), py::arg("height"))
      .def(
          "sample_roi_stats",
          &AlignedDepthFrameDevice::sample_roi_stats,
          py::arg("left"),
          py::arg("top"),
          py::arg("width"),
          py::arg("height"),
          py::arg("max_samples") = 4096)
      .def(
          "sample_masked_roi_stats",
          &AlignedDepthFrameDevice::sample_masked_roi_stats,
          py::arg("left"),
          py::arg("top"),
          py::arg("width"),
          py::arg("height"),
          py::arg("mask"),
          py::arg("threshold") = 0.5F,
          py::arg("max_samples") = 4096)
      .def(
          "sample_masked_person_roi_stats",
          &AlignedDepthFrameDevice::sample_masked_person_roi_stats,
          py::arg("left"),
          py::arg("top"),
          py::arg("width"),
          py::arg("height"),
          py::arg("mask"),
          py::arg("threshold") = 0.5F,
          py::arg("max_samples") = 4096)
      .def_property_readonly("frame_width", &AlignedDepthFrameDevice::frame_width)
      .def_property_readonly("frame_height", &AlignedDepthFrameDevice::frame_height)
      .def_property_readonly("depth_width", &AlignedDepthFrameDevice::depth_width)
      .def_property_readonly("depth_height", &AlignedDepthFrameDevice::depth_height);
  m.def(
      "capture_aligned_depth_frame",
      &capture_aligned_depth_frame,
      py::arg("frame_meta"),
      py::arg("gie_id"),
      py::arg("frame_w"),
      py::arg("frame_h"),
      "Capture a frame-level DAv2 tensor from device memory and align it to canonical frame size on the GPU.");
  m.def(
      "capture_tensor_layers",
      &capture_tensor_layers,
      py::arg("frame_meta"),
      py::arg("gie_id"),
      "Capture all frame-level tensor layers for a nvinfer unique-id as CPU float arrays.");
}
