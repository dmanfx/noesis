// SPDX-FileCopyrightText: 2026 Noesis
// SPDX-License-Identifier: MIT
//
// DeepStream 9 custom bbox parser for DS9-owned YOLO detector profiles.
//
// Supported detector outputs:
//   - YOLO11 detector: [N, 6] rows, observed from ONNX as [B, 8400, 6]
//   - YOLO26 detector: [N, 6] rows, observed from ONNX as [B, 300, 6]
//
// Each row is expected to be:
//   [x1, y1, x2, y2, confidence, class_id]

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "nvdsinfer_custom_impl.h"

namespace {

template <typename T>
inline T clamp(T v, T lo, T hi) {
  return std::min(hi, std::max(lo, v));
}

enum class LayoutKind {
  kRowsLast,
  kRowsTransposed2D,
  kRowsLast3D,
  kRowsMiddle3D,
};

struct RowLayout {
  LayoutKind kind = LayoutKind::kRowsLast;
  std::size_t rows = 0;
  std::size_t rows_per_group = 0;
  std::size_t groups = 1;
  std::size_t columns = 6;
  std::size_t total_elements = 0;
};

const NvDsInferLayerInfo* first_output_layer(const std::vector<NvDsInferLayerInfo>& layers) {
  for (const auto& layer : layers) {
    if (!layer.isInput) return &layer;
  }
  return layers.empty() ? nullptr : &layers[0];
}

bool parse_layout(const NvDsInferLayerInfo& layer, RowLayout& out) {
  const auto& dims = layer.inferDims;
  out.total_elements = dims.numElements;

  if (dims.numDims == 2) {
    const std::size_t d0 = dims.d[0];
    const std::size_t d1 = dims.d[1];
    if (d1 == 6) {
      out.kind = LayoutKind::kRowsLast;
      out.rows = d0;
      out.rows_per_group = d0;
      out.columns = 6;
      return out.rows > 0;
    }
    if (d0 == 6) {
      out.kind = LayoutKind::kRowsTransposed2D;
      out.rows = d1;
      out.rows_per_group = d1;
      out.columns = 6;
      return out.rows > 0;
    }
    return false;
  }

  if (dims.numDims == 3) {
    const std::size_t d0 = dims.d[0];
    const std::size_t d1 = dims.d[1];
    const std::size_t d2 = dims.d[2];
    if (d2 == 6) {
      out.kind = LayoutKind::kRowsLast3D;
      out.groups = d0;
      out.rows_per_group = d1;
      out.rows = d0 * d1;
      out.columns = 6;
      return out.rows > 0;
    }
    if (d1 == 6) {
      out.kind = LayoutKind::kRowsMiddle3D;
      out.groups = d0;
      out.rows_per_group = d2;
      out.rows = d0 * d2;
      out.columns = 6;
      return out.rows > 0;
    }
  }

  return false;
}

std::size_t layout_index(const RowLayout& layout, std::size_t row, std::size_t col) {
  switch (layout.kind) {
    case LayoutKind::kRowsLast:
      return row * layout.columns + col;
    case LayoutKind::kRowsTransposed2D:
      return col * layout.rows + row;
    case LayoutKind::kRowsLast3D: {
      const std::size_t group = row / layout.rows_per_group;
      const std::size_t local = row % layout.rows_per_group;
      return group * layout.rows_per_group * layout.columns + local * layout.columns + col;
    }
    case LayoutKind::kRowsMiddle3D: {
      const std::size_t group = row / layout.rows_per_group;
      const std::size_t local = row % layout.rows_per_group;
      return group * layout.columns * layout.rows_per_group + col * layout.rows_per_group + local;
    }
  }
  return row * layout.columns + col;
}

bool is_device_ptr(const void* ptr) {
  if (!ptr) return false;
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
  return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

float class_threshold(int class_id, const NvDsInferParseDetectionParams& params) {
  if (class_id < 0) return 1e9f;
  if (params.numClassesConfigured > 0 &&
      static_cast<unsigned>(class_id) >= params.numClassesConfigured) {
    return 1e9f;
  }
  if (params.perClassPreclusterThreshold.empty()) return 0.0f;
  const std::size_t idx = static_cast<std::size_t>(class_id);
  if (idx < params.perClassPreclusterThreshold.size()) {
    return params.perClassPreclusterThreshold[idx];
  }
  return params.perClassPreclusterThreshold[0];
}

bool set_bbox(float x1, float y1, float x2, float y2,
              const NvDsInferNetworkInfo& network,
              NvDsInferObjectDetectionInfo& object) {
  x1 = clamp(x1, 0.0f, static_cast<float>(network.width));
  y1 = clamp(y1, 0.0f, static_cast<float>(network.height));
  x2 = clamp(x2, 0.0f, static_cast<float>(network.width));
  y2 = clamp(y2, 0.0f, static_cast<float>(network.height));

  object.left = x1;
  object.top = y1;
  object.width = clamp(x2 - x1, 0.0f, static_cast<float>(network.width));
  object.height = clamp(y2 - y1, 0.0f, static_cast<float>(network.height));
  return object.width >= 1.0f && object.height >= 1.0f;
}

template <typename ReadValue>
bool parse_rows(const RowLayout& layout,
                ReadValue read_value,
                const NvDsInferNetworkInfo& network,
                const NvDsInferParseDetectionParams& params,
                std::vector<NvDsInferObjectDetectionInfo>& objects) {
  objects.clear();
  objects.reserve(std::min<std::size_t>(layout.rows, 300));

  for (std::size_t row = 0; row < layout.rows; ++row) {
    const float confidence = read_value(row, 4);
    const int class_id = static_cast<int>(std::lround(read_value(row, 5)));
    if (confidence < class_threshold(class_id, params)) continue;

    NvDsInferObjectDetectionInfo object{};
    if (!set_bbox(
            read_value(row, 0),
            read_value(row, 1),
            read_value(row, 2),
            read_value(row, 3),
            network,
            object)) {
      continue;
    }

    object.classId = static_cast<unsigned>(class_id);
    object.detectionConfidence = confidence;
    objects.emplace_back(object);
  }

  return true;
}

bool parse_yolo_rows(std::vector<NvDsInferLayerInfo> const& output_layers,
                     NvDsInferNetworkInfo const& network,
                     NvDsInferParseDetectionParams const& params,
                     std::vector<NvDsInferObjectDetectionInfo>& objects) {
  const NvDsInferLayerInfo* layer = first_output_layer(output_layers);
  if (!layer || !layer->buffer) {
    std::cerr << "[yolo-detect] missing output layer/buffer" << std::endl;
    return false;
  }

  RowLayout layout{};
  if (!parse_layout(*layer, layout)) {
    std::cerr << "[yolo-detect] expected output rows with six columns, got dims=[";
    for (unsigned i = 0; i < layer->inferDims.numDims; ++i) {
      if (i) std::cerr << ",";
      std::cerr << layer->inferDims.d[i];
    }
    std::cerr << "]" << std::endl;
    return false;
  }

  const std::size_t total = layout.total_elements ? layout.total_elements : layout.rows * layout.columns;
  const bool device = is_device_ptr(layer->buffer);

  if (layer->dataType == FLOAT) {
    std::vector<float> host;
    const float* data = static_cast<const float*>(layer->buffer);
    if (device) {
      host.resize(total);
      const cudaError_t err = cudaMemcpy(
          host.data(), layer->buffer, total * sizeof(float), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        std::cerr << "[yolo-detect] cudaMemcpy FLOAT output failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
      }
      data = host.data();
    }
    return parse_rows(
        layout,
        [&](std::size_t row, std::size_t col) -> float {
          return data[layout_index(layout, row, col)];
        },
        network,
        params,
        objects);
  }

  if (layer->dataType == HALF) {
    std::vector<__half> host;
    const __half* data = static_cast<const __half*>(layer->buffer);
    if (device) {
      host.resize(total);
      const cudaError_t err = cudaMemcpy(
          host.data(), layer->buffer, total * sizeof(__half), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        std::cerr << "[yolo-detect] cudaMemcpy HALF output failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
      }
      data = host.data();
    }
    return parse_rows(
        layout,
        [&](std::size_t row, std::size_t col) -> float {
          return __half2float(data[layout_index(layout, row, col)]);
        },
        network,
        params,
        objects);
  }

  std::cerr << "[yolo-detect] unsupported output data type: " << layer->dataType << std::endl;
  return false;
}

}  // namespace

extern "C" bool NvDsInferParseYoloDetect(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  return parse_yolo_rows(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  return NvDsInferParseYoloDetect(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloDetect);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
