#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
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
