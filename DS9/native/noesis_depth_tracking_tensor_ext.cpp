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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>

#include "metadata.hpp"
#include "nvdsmeta.h"
#include "tensor.hpp"

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
    const float* mask,
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
    float threshold,
    float* out_values,
    float* out_center,
    cudaStream_t stream);

extern "C" cudaError_t noesis_sample_masked_person_roi_values_cuda(
    const float* depth,
    const float* mask,
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
    float threshold,
    float* out_all,
    float* out_lower,
    float* out_torso,
    float* out_center,
    cudaStream_t stream);

namespace {

void throw_on_cuda(cudaError_t status, const char* what);

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(unsigned int target_device) {
    throw_on_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice failed");
    if (previous_device_ != static_cast<int>(target_device)) {
      throw_on_cuda(
          cudaSetDevice(static_cast<int>(target_device)), "cudaSetDevice failed");
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
  // DS9 getLayers() allocates observer Tensor wrappers. Own those wrappers for
  // this synchronous call; inference storage remains owned by the SDK.
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

  const std::unordered_map<std::string, deepstream::Tensor*>& get() const {
    return layers_;
  }

 private:
  std::unordered_map<std::string, deepstream::Tensor*> layers_;
};

std::vector<int> layer_shape(const deepstream::Tensor& tensor) {
  std::vector<int> dims;
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty() || shape.size() != static_cast<size_t>(tensor.rank())) {
    return dims;
  }
  dims.reserve(shape.size());
  for (uint64_t value : shape) {
    if (value == 0ULL ||
        value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return {};
    }
    dims.push_back(static_cast<int>(value));
  }
  return dims;
}

std::optional<std::pair<int, int>> depth_hw(const deepstream::Tensor& tensor) {
  const std::vector<int> dims = layer_shape(tensor);
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

deepstream::Tensor* select_depth_layer(
    const std::unordered_map<std::string, deepstream::Tensor*>& layers) {
  const auto selected = layers.find("depth");
  return selected != layers.end() ? selected->second : nullptr;
}

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

void copy_tensor_storage_to_host(
    const deepstream::Tensor& tensor,
    void* destination,
    size_t byte_count,
    const char* cuda_error) {
  if (tensor.data() == nullptr || tensor.size() != byte_count) {
    throw std::runtime_error("Tensor storage does not match its public shape");
  }
  switch (tensor.deviceType()) {
    case deepstream::Tensor::CPU:
      std::memcpy(destination, tensor.data(), byte_count);
      return;
    case deepstream::Tensor::GPU: {
      CudaDeviceGuard guard(tensor.deviceId());
      throw_on_cuda(
          cudaMemcpy(
              destination, tensor.data(), byte_count, cudaMemcpyDeviceToHost),
          cuda_error);
      return;
    }
    default:
      throw std::runtime_error("Tensor has no supported storage device");
  }
}

py::array_t<float> copy_mapanything_frame_layer_to_numpy(
    const deepstream::Tensor& tensor,
    int expected_height,
    int expected_width) {
  // gst-nvinfer attach_tensor_output_meta already advances each frame's tensor
  // pointer by its batch position while retaining per-frame inferDims.  This
  // reader must copy the public tensor as-is and must never apply batchId again.
  if (expected_height <= 0 || expected_width <= 0) {
    throw std::runtime_error("MapAnything output dimensions are invalid");
  }
  if (!tensor_is_contiguous(tensor)) {
    throw std::runtime_error("MapAnything tensor layer must be contiguous");
  }
  const std::vector<int> dims = layer_shape(tensor);
  if (dims.size() != 3U || dims[0] != 1 ||
      dims[1] != expected_height || dims[2] != expected_width) {
    throw std::runtime_error(
        "MapAnything tensor does not carry the exact per-frame 1xHxW shape");
  }
  const auto total_count_opt = tensor_num_elements(tensor);
  const uint64_t expected_count =
      static_cast<uint64_t>(expected_height) *
      static_cast<uint64_t>(expected_width);
  if (!total_count_opt.has_value() || *total_count_opt != expected_count) {
    throw std::runtime_error("MapAnything per-frame tensor shape is invalid");
  }
  if (expected_count > static_cast<uint64_t>(
                           std::numeric_limits<size_t>::max() /
                           sizeof(float))) {
    throw std::runtime_error("MapAnything per-frame tensor is too large");
  }
  const size_t element_count = static_cast<size_t>(expected_count);

  const std::vector<py::ssize_t> shape{
      static_cast<py::ssize_t>(expected_height),
      static_cast<py::ssize_t>(expected_width)};
  py::array_t<float> out(shape);
  float* destination = static_cast<float*>(out.mutable_data());
  if (tensor.dtype() != deepstream::Tensor::FLOAT) {
    throw std::runtime_error("Unsupported MapAnything tensor layer dtype");
  }

  switch (tensor.bits()) {
    case 32U: {
      const size_t byte_count = element_count * sizeof(float);
      {
        // The nvinfer-owned pointer is dereferenced synchronously while frame
        // metadata is alive, but the Python GIL need not be held during D2H.
        py::gil_scoped_release release;
        copy_tensor_storage_to_host(
            tensor,
            destination,
            byte_count,
            "cudaMemcpy MapAnything FLOAT32 frame tensor failed");
      }
      break;
    }
    case 16U: {
      const size_t byte_count = element_count * sizeof(uint16_t);
      std::vector<uint16_t> half_values(element_count);
      {
        py::gil_scoped_release release;
        copy_tensor_storage_to_host(
            tensor,
            half_values.data(),
            byte_count,
            "cudaMemcpy MapAnything FLOAT16 frame tensor failed");
        for (size_t index = 0U; index < element_count; ++index) {
          __half value;
          std::memcpy(&value, &half_values[index], sizeof(uint16_t));
          destination[index] = __half2float(value);
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported MapAnything tensor layer bit width");
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

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  CudaBuffer(size_t count, unsigned int device_id) {
    allocate(count, device_id);
  }
  ~CudaBuffer() { reset(); }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  CudaBuffer(CudaBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_), device_id_(other.device_id_) {
    other.ptr_ = nullptr;
    other.count_ = 0U;
    other.device_id_ = -1;
  }

  CudaBuffer& operator=(CudaBuffer&& other) noexcept {
    if (this == &other) return *this;
    reset();
    ptr_ = other.ptr_;
    count_ = other.count_;
    device_id_ = other.device_id_;
    other.ptr_ = nullptr;
    other.count_ = 0U;
    other.device_id_ = -1;
    return *this;
  }

  void allocate(size_t count, unsigned int device_id) {
    reset();
    if (count == 0U) return;
    CudaDeviceGuard device_guard(device_id);
    T* allocated = nullptr;
    throw_on_cuda(
        cudaMalloc(reinterpret_cast<void**>(&allocated), count * sizeof(T)),
        "cudaMalloc failed");
    ptr_ = allocated;
    count_ = count;
    device_id_ = static_cast<int>(device_id);
  }

  void ensure_capacity(size_t count, unsigned int device_id) {
    if (ptr_ != nullptr && count_ >= count &&
        device_id_ == static_cast<int>(device_id)) {
      return;
    }
    allocate(count, device_id);
  }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      try {
        CudaDeviceGuard device_guard(static_cast<unsigned int>(device_id_));
        (void)cudaFree(ptr_);
      } catch (...) {
      }
      ptr_ = nullptr;
      count_ = 0U;
      device_id_ = -1;
    }
  }

  T* get() const { return ptr_; }
  size_t count() const { return count_; }

  T* release() noexcept {
    T* raw = ptr_;
    ptr_ = nullptr;
    count_ = 0U;
    device_id_ = -1;
    return raw;
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0U;
  int device_id_ = -1;
};

class AlignedDepthFrameDevice {
 public:
  AlignedDepthFrameDevice(
      float* device_ptr,
      unsigned int device_id,
      int frame_w,
      int frame_h,
      int depth_w,
      int depth_h)
      : device_ptr_(device_ptr),
        device_id_(device_id),
        frame_w_(frame_w),
        frame_h_(frame_h),
        depth_w_(depth_w),
        depth_h_(depth_h) {}

  ~AlignedDepthFrameDevice() {
    if (device_ptr_ != nullptr) {
      try {
        CudaDeviceGuard device_guard(device_id_);
        (void)cudaFree(device_ptr_);
      } catch (...) {
      }
      device_ptr_ = nullptr;
    }
  }

  AlignedDepthFrameDevice(const AlignedDepthFrameDevice&) = delete;
  AlignedDepthFrameDevice& operator=(const AlignedDepthFrameDevice&) = delete;

  py::array_t<float> copy_roi_to_numpy(int left, int top, int width, int height) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    CudaDeviceGuard device_guard(device_id_);
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
    CudaDeviceGuard device_guard(device_id_);
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
    sampled_device.ensure_capacity(sampled_count, device_id_);
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
      py::array_t<float, py::array::c_style | py::array::forcecast> mask_array,
      float threshold,
      int max_samples) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    CudaDeviceGuard device_guard(device_id_);
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

    py::buffer_info mask_info = mask_array.request();
    if (mask_info.ndim != 2) {
      throw std::runtime_error("sample_masked_roi_stats mask must be 2D");
    }
    const int mask_h = static_cast<int>(mask_info.shape[0]);
    const int mask_w = static_cast<int>(mask_info.shape[1]);
    if (mask_w != roi_w || mask_h != roi_h) {
      throw std::runtime_error("sample_masked_roi_stats mask shape must match ROI size");
    }
    const float* mask_host = static_cast<const float*>(mask_info.ptr);
    if (mask_host == nullptr) {
      throw std::runtime_error("sample_masked_roi_stats mask buffer is unavailable");
    }

    const int sample_cap = std::max(128, max_samples);
    int mask_area = 0;
    for (int idx = 0; idx < mask_w * mask_h; ++idx) {
      if (mask_host[idx] > threshold) {
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
        if (mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] > threshold) {
          ++sampled_mask_area;
        }
      }
    }

    const size_t mask_count = static_cast<size_t>(mask_w) * static_cast<size_t>(mask_h);
    thread_local CudaBuffer<float> mask_device;
    mask_device.ensure_capacity(mask_count, device_id_);
    throw_on_cuda(
        cudaMemcpy(
            mask_device.get(),
            mask_host,
            mask_count * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpy object-depth mask to device failed");

    const size_t sampled_count = static_cast<size_t>(sampled_area) + 1U;
    thread_local CudaBuffer<float> sampled_device;
    sampled_device.ensure_capacity(sampled_count, device_id_);
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
            threshold,
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
      py::array_t<float, py::array::c_style | py::array::forcecast> mask_array,
      float threshold,
      int max_samples) const {
    if (device_ptr_ == nullptr) {
      throw std::runtime_error("Aligned depth frame device buffer is unavailable");
    }
    CudaDeviceGuard device_guard(device_id_);
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

    py::buffer_info mask_info = mask_array.request();
    if (mask_info.ndim != 2) {
      throw std::runtime_error("sample_masked_person_roi_stats mask must be 2D");
    }
    const int mask_h = static_cast<int>(mask_info.shape[0]);
    const int mask_w = static_cast<int>(mask_info.shape[1]);
    if (mask_w != roi_w || mask_h != roi_h) {
      throw std::runtime_error("sample_masked_person_roi_stats mask shape must match ROI size");
    }
    const float* mask_host = static_cast<const float*>(mask_info.ptr);
    if (mask_host == nullptr) {
      throw std::runtime_error("sample_masked_person_roi_stats mask buffer is unavailable");
    }

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
    auto eroded_host = [mask_host, mask_w, mask_h, threshold](int x, int y) -> bool {
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= mask_h) return false;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= mask_w) return false;
          if (mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] <= threshold) {
            return false;
          }
        }
      }
      return true;
    };

    int mask_area = 0;
    int lower_mask_area = 0;
    int torso_mask_area = 0;
    for (int yy = 0; yy < roi_h; ++yy) {
      for (int xx = 0; xx < roi_w; ++xx) {
        const bool active = mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] > threshold;
        if (!active) continue;
        ++mask_area;
        const bool eroded = eroded_host(xx, yy);
        if (eroded && in_center_band_host(xx, yy, roi_w, roi_h, 0.88, 1.0, 0.35)) {
          ++lower_mask_area;
        }
        if (eroded && in_center_band_host(xx, yy, roi_w, roi_h, 0.35, 0.70, 0.50)) {
          ++torso_mask_area;
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
        const bool active = mask_host[(static_cast<size_t>(yy) * static_cast<size_t>(mask_w)) + static_cast<size_t>(xx)] > threshold;
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
    thread_local CudaBuffer<float> mask_device;
    mask_device.ensure_capacity(mask_count, device_id_);
    throw_on_cuda(
        cudaMemcpy(
            mask_device.get(),
            mask_host,
            mask_count * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpy person object-depth mask to device failed");

    const size_t sampled_area_size = static_cast<size_t>(sampled_area);
    const size_t sampled_count = (sampled_area_size * 3U) + 1U;
    thread_local CudaBuffer<float> sampled_device;
    sampled_device.ensure_capacity(sampled_count, device_id_);
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
            threshold,
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
  unsigned int device_id() const { return device_id_; }

 private:
  float* device_ptr_ = nullptr;
  unsigned int device_id_ = 0U;
  int frame_w_ = 0;
  int frame_h_ = 0;
  int depth_w_ = 0;
  int depth_h_ = 0;
};

std::shared_ptr<AlignedDepthFrameDevice> align_depth_layer_to_frame(
    const deepstream::Tensor& tensor,
    int frame_w,
    int frame_h) {
  if (frame_w <= 0 || frame_h <= 0) {
    throw std::runtime_error("Target frame size must be positive");
  }
  if (tensor.deviceType() != deepstream::Tensor::GPU || tensor.data() == nullptr) {
    throw std::runtime_error("Depth tracking tensor device buffer is unavailable");
  }
  if (!tensor_is_contiguous(tensor)) {
    throw std::runtime_error("Depth tracking tensor must be contiguous");
  }

  const auto hw = depth_hw(tensor);
  if (!hw.has_value()) {
    throw std::runtime_error("Depth tracking tensor layer shape is unsupported");
  }
  const int depth_h = hw->first;
  const int depth_w = hw->second;
  if (depth_w <= 0 || depth_h <= 0) {
    throw std::runtime_error("Depth tracking tensor shape is invalid");
  }
  const auto elements = tensor_num_elements(tensor);
  const uint64_t expected_elements =
      static_cast<uint64_t>(depth_w) * static_cast<uint64_t>(depth_h);
  if (!elements.has_value() || *elements != expected_elements) {
    throw std::runtime_error("Depth tracking tensor shape is not a single depth map");
  }

  const unsigned int device_id = tensor.deviceId();
  CudaDeviceGuard device_guard(device_id);
  const void* src_device = tensor.data();
  CudaBuffer<float> src_float;
  const NppStreamContext npp_ctx = default_npp_stream_context();

  if (tensor.dtype() != deepstream::Tensor::FLOAT) {
    throw std::runtime_error("Unsupported depth tracking tensor dtype");
  }
  switch (tensor.bits()) {
    case 32U:
      if (tensor.size() != expected_elements * sizeof(float)) {
        throw std::runtime_error("Depth tracking FLOAT32 tensor storage size is invalid");
      }
      break;
    case 16U: {
      if (tensor.size() != expected_elements * sizeof(uint16_t)) {
        throw std::runtime_error("Depth tracking FLOAT16 tensor storage size is invalid");
      }
      src_float.allocate(
          static_cast<size_t>(depth_w) * static_cast<size_t>(depth_h),
          device_id);
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
      throw std::runtime_error("Unsupported depth tracking tensor bit width");
  }

  CudaBuffer<float> aligned(
      static_cast<size_t>(frame_w) * static_cast<size_t>(frame_h), device_id);
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
  return std::make_shared<AlignedDepthFrameDevice>(
      aligned.release(), device_id, frame_w, frame_h, depth_w, depth_h);
}

}  // namespace

py::object capture_aligned_depth_frame(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id,
    int frame_w,
    int frame_h) {
  std::shared_ptr<AlignedDepthFrameDevice> result;
  frame_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (result != nullptr) return;
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta ||
            tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }
        TensorLayerMap owned_layers(tensor_meta);
        deepstream::Tensor* selected = select_depth_layer(owned_layers.get());
        if (selected == nullptr) return;
        result = align_depth_layer_to_frame(
            *selected, frame_w, frame_h);
      },
      NVDSINFER_TENSOR_OUTPUT_META);
  return result != nullptr ? py::cast(result) : py::none();
}

py::object capture_mapanything_tensor_layers_exact(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id,
    int expected_height,
    int expected_width) {
  if (gie_id <= 0) {
    throw std::runtime_error("MapAnything gie_id must be positive");
  }
  if (expected_height <= 0 || expected_width <= 0) {
    throw std::runtime_error("MapAnything output dimensions are invalid");
  }

  size_t matching_meta_count = 0U;
  frame_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (tensor_meta &&
            tensor_meta.uniqueId() == static_cast<unsigned int>(gie_id)) {
          ++matching_meta_count;
        }
      },
      NVDSINFER_TENSOR_OUTPUT_META);
  if (matching_meta_count == 0U) {
    return py::none();
  }
  if (matching_meta_count != 1U) {
    throw std::runtime_error(
        "MapAnything frame contains ambiguous duplicate tensor metadata for the configured gie_id");
  }

  py::dict layers;
  bool captured = false;
  frame_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (captured) return;
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta ||
            tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }
        TensorLayerMap owned_layers(tensor_meta);
        const auto& owned = owned_layers.get();
        static const std::vector<std::string> required_layers{
            "depth", "conf", "mask"};
        if (owned.size() != required_layers.size()) {
          throw std::runtime_error(
              "MapAnything tensor metadata does not expose exactly depth/conf/mask");
        }
        for (const std::string& name : required_layers) {
          const auto selected = owned.find(name);
          if (selected == owned.end() || selected->second == nullptr) {
            throw std::runtime_error(
                "MapAnything tensor metadata is missing required layer " + name);
          }
          layers[py::str(name)] = copy_mapanything_frame_layer_to_numpy(
              *selected->second, expected_height, expected_width);
        }
        captured = true;
      },
      NVDSINFER_TENSOR_OUTPUT_META);
  if (!captured) {
    throw std::runtime_error(
        "MapAnything tensor metadata disappeared during exact capture");
  }
  return std::move(layers);
}

PYBIND11_MODULE(noesis_depth_tracking_tensor_ext, m) {
  m.doc() = "Noesis DS9 helper bindings for DAv2 device frames and exact MapAnything tensor capture.";
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
      .def_property_readonly("depth_height", &AlignedDepthFrameDevice::depth_height)
      .def_property_readonly("device_id", &AlignedDepthFrameDevice::device_id);
  m.def(
      "capture_aligned_depth_frame",
      &capture_aligned_depth_frame,
      py::arg("frame_meta"),
      py::arg("gie_id"),
      py::arg("frame_w"),
      py::arg("frame_h"),
      "Capture a frame-level DAv2 tensor from device memory and align it to canonical frame size on the GPU.");
  m.def(
      "capture_mapanything_tensor_layers_exact",
      &capture_mapanything_tensor_layers_exact,
      py::arg("frame_meta"),
      py::arg("gie_id"),
      py::arg("expected_height"),
      py::arg("expected_width"),
      "Capture the one exact MapAnything UID and per-frame depth/conf/mask tensors as CPU float arrays.");
}
