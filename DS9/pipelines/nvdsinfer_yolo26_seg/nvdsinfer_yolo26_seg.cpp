// SPDX-FileCopyrightText: 2026 Noesis
// SPDX-License-Identifier: MIT
//
// DeepStream 9.1 custom parser for YOLO26-Seg (fused output0 only).
//
// Expected output tensor layout per detection row:
//   [x1, y1, x2, y2, score, class_id, mask_flattened...]
//
// Notes:
// - The ONNX/engine must be fused to remove the proto tensor output to avoid
//   large device->host copies and CPU-side mask composition.
// - When `disable-output-host-copy=1` is enabled in the nvinfer config, DeepStream
//   will pass device pointers into this parser. In that case we do minimal D2H
//   copies: one cudaMemcpy2D for the first 6 values of each row, and per-object
//   copies for mask data for rows that pass thresholds.

#include <algorithm>
#include <cmath>
#include <cstddef>
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

struct DetDims {
  std::size_t num = 0;
  std::size_t channels = 0;
};

const NvDsInferLayerInfo* find_layer_by_name(const std::vector<NvDsInferLayerInfo>& layers,
                                             const char* name) {
  if (!name || !*name) return nullptr;
  for (const auto& layer : layers) {
    if (!layer.layerName) continue;
    if (std::strcmp(layer.layerName, name) == 0) return &layer;
  }
  return nullptr;
}

bool validate_layer_dtype(const NvDsInferLayerInfo& layer, const char* label) {
  if (!layer.buffer) {
    std::cerr << "[yolo26-seg] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT && layer.dataType != HALF) {
    std::cerr << "[yolo26-seg] " << label << " must be FLOAT or HALF" << std::endl;
    return false;
  }
  return true;
}

bool parse_det_dims(const NvDsInferLayerInfo& dets, DetDims& out) {
  const int nd = dets.inferDims.numDims;
  if (nd == 2) {
    out.num = static_cast<std::size_t>(dets.inferDims.d[0]);
    out.channels = static_cast<std::size_t>(dets.inferDims.d[1]);
    return out.num > 0 && out.channels > 0;
  }
  if (nd == 3) {
    // Support [1, num, channels] (DeepStream may preserve a unit batch dim).
    out.num = static_cast<std::size_t>(dets.inferDims.d[1]);
    out.channels = static_cast<std::size_t>(dets.inferDims.d[2]);
    return out.num > 0 && out.channels > 0;
  }
  return false;
}

float class_threshold(int cls, const NvDsInferParseDetectionParams& det) {
  if (cls < 0) return 1e9f;
  if (det.perClassPreclusterThreshold.empty()) return 0.0f;
  const std::size_t idx = static_cast<std::size_t>(cls);
  if (idx < det.perClassPreclusterThreshold.size()) return det.perClassPreclusterThreshold[idx];
  return det.perClassPreclusterThreshold[0];
}

unsigned infer_mask_side(std::size_t mask_len, unsigned net_w) {
  if (mask_len == 0) return 0;
  unsigned side = static_cast<unsigned>(std::lround(std::sqrt(static_cast<double>(mask_len))));
  if (static_cast<std::size_t>(side) * static_cast<std::size_t>(side) == mask_len) return side;
  // Fallback: net/4 is the common proto resolution, but this is only used on mismatch.
  side = std::max(1u, net_w / 4);
  return side;
}

bool set_bbox(float x1, float y1, float x2, float y2, unsigned net_w, unsigned net_h,
              NvDsInferInstanceMaskInfo& obj) {
  x1 = clamp(x1, 0.0f, static_cast<float>(net_w));
  y1 = clamp(y1, 0.0f, static_cast<float>(net_h));
  x2 = clamp(x2, 0.0f, static_cast<float>(net_w));
  y2 = clamp(y2, 0.0f, static_cast<float>(net_h));

  obj.left = x1;
  obj.top = y1;
  obj.width = clamp(x2 - x1, 0.0f, static_cast<float>(net_w));
  obj.height = clamp(y2 - y1, 0.0f, static_cast<float>(net_h));
  return obj.width >= 1.0f && obj.height >= 1.0f;
}

bool is_device_ptr(const void* p) {
  if (!p) return false;
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, p);
  if (err != cudaSuccess) {
    // Clear sticky error in case caller checks later.
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return (attr.type == cudaMemoryTypeDevice) || (attr.type == cudaMemoryTypeManaged);
#else
  return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

}  // namespace

extern "C" bool NvDsInferParseYolo26Seg(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  const NvDsInferLayerInfo* out0 = find_layer_by_name(outputLayersInfo, "output0");
  if (!out0) {
    if (outputLayersInfo.empty()) {
      std::cerr << "[yolo26-seg] expected output layers, got 0" << std::endl;
      return false;
    }
    out0 = &outputLayersInfo[0];
  }

  if (!validate_layer_dtype(*out0, "output0")) return false;

  DetDims dims{};
  if (!parse_det_dims(*out0, dims)) {
    std::cerr << "[yolo26-seg] unexpected output0 dims" << std::endl;
    return false;
  }
  if (dims.channels < 7) {
    std::cerr << "[yolo26-seg] output0 channels too small: " << dims.channels << std::endl;
    return false;
  }

  const std::size_t mask_len = dims.channels - 6;
  const unsigned mask_side = infer_mask_side(mask_len, networkInfo.width);
  if (mask_side == 0) {
    std::cerr << "[yolo26-seg] invalid mask length: " << mask_len << std::endl;
    return false;
  }

  const bool device_buf = is_device_ptr(out0->buffer);

  objectList.clear();
  objectList.reserve(dims.num);

  if (!device_buf) {
    // Fast-path: DeepStream already copied outputs to host.
    if (out0->dataType == FLOAT) {
      const float* buf = static_cast<const float*>(out0->buffer);
      for (std::size_t i = 0; i < dims.num; ++i) {
        const std::size_t base = i * dims.channels;
        const float score = buf[base + 4];
        const int cls = static_cast<int>(buf[base + 5]);
        if (cls < 0) continue;
        if (score < class_threshold(cls, detectionParams)) continue;

        const float x1 = buf[base + 0];
        const float y1 = buf[base + 1];
        const float x2 = buf[base + 2];
        const float y2 = buf[base + 3];

        NvDsInferInstanceMaskInfo obj{};
        if (!set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj)) continue;

        obj.classId = cls;
        obj.detectionConfidence = score;

        obj.mask_width = mask_side;
        obj.mask_height = mask_side;
        obj.mask_size = static_cast<unsigned>(mask_len * sizeof(float));
        obj.mask = new float[mask_len];
        std::memcpy(obj.mask, buf + base + 6, mask_len * sizeof(float));

        objectList.emplace_back(obj);
      }
      return true;
    }

    // Host pointer but FP16 output: convert to float.
    const __half* buf = static_cast<const __half*>(out0->buffer);
    for (std::size_t i = 0; i < dims.num; ++i) {
      const std::size_t base = i * dims.channels;
      const float score = __half2float(buf[base + 4]);
      const int cls = static_cast<int>(__half2float(buf[base + 5]));
      if (cls < 0) continue;
      if (score < class_threshold(cls, detectionParams)) continue;

      const float x1 = __half2float(buf[base + 0]);
      const float y1 = __half2float(buf[base + 1]);
      const float x2 = __half2float(buf[base + 2]);
      const float y2 = __half2float(buf[base + 3]);

      NvDsInferInstanceMaskInfo obj{};
      if (!set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj)) continue;

      obj.classId = cls;
      obj.detectionConfidence = score;

      obj.mask_width = mask_side;
      obj.mask_height = mask_side;
      obj.mask_size = static_cast<unsigned>(mask_len * sizeof(float));
      obj.mask = new float[mask_len];
      for (std::size_t j = 0; j < mask_len; ++j) {
        obj.mask[j] = __half2float(buf[base + 6 + j]);
      }

      objectList.emplace_back(obj);
    }
    return true;
  }

  // Device-pointer path: DeepStream did not copy outputs to host. We pull only what we need.
  if (out0->dataType == FLOAT) {
    const float* dev = static_cast<const float*>(out0->buffer);

    // Copy [num,6] header region in a single call.
    std::vector<float> header(dims.num * 6);
    const cudaError_t hdr_err = cudaMemcpy2D(
        header.data(),
        6 * sizeof(float),
        dev,
        dims.channels * sizeof(float),
        6 * sizeof(float),
        dims.num,
        cudaMemcpyDeviceToHost);
    if (hdr_err != cudaSuccess) {
      std::cerr << "[yolo26-seg] cudaMemcpy2D header failed: " << cudaGetErrorString(hdr_err)
                << std::endl;
      return false;
    }

    std::vector<float> mask_tmp(mask_len);

    for (std::size_t i = 0; i < dims.num; ++i) {
      const std::size_t hb = i * 6;
      const float score = header[hb + 4];
      const int cls = static_cast<int>(header[hb + 5]);
      if (cls < 0) continue;
      if (score < class_threshold(cls, detectionParams)) continue;

      const float x1 = header[hb + 0];
      const float y1 = header[hb + 1];
      const float x2 = header[hb + 2];
      const float y2 = header[hb + 3];

      NvDsInferInstanceMaskInfo obj{};
      if (!set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj)) continue;

      const cudaError_t m_err = cudaMemcpy(
          mask_tmp.data(),
          dev + (i * dims.channels + 6),
          mask_len * sizeof(float),
          cudaMemcpyDeviceToHost);
      if (m_err != cudaSuccess) {
        std::cerr << "[yolo26-seg] cudaMemcpy mask failed: " << cudaGetErrorString(m_err)
                  << std::endl;
        continue;
      }

      obj.classId = cls;
      obj.detectionConfidence = score;

      obj.mask_width = mask_side;
      obj.mask_height = mask_side;
      obj.mask_size = static_cast<unsigned>(mask_len * sizeof(float));
      obj.mask = new float[mask_len];
      std::memcpy(obj.mask, mask_tmp.data(), mask_len * sizeof(float));

      objectList.emplace_back(obj);
    }
    return true;
  }

  // Device pointer with FP16 output: copy as half, convert to float on CPU.
  const __half* dev = static_cast<const __half*>(out0->buffer);
  std::vector<__half> header_h(dims.num * 6);
  const cudaError_t hdr_err = cudaMemcpy2D(
      header_h.data(),
      6 * sizeof(__half),
      dev,
      dims.channels * sizeof(__half),
      6 * sizeof(__half),
      dims.num,
      cudaMemcpyDeviceToHost);
  if (hdr_err != cudaSuccess) {
    std::cerr << "[yolo26-seg] cudaMemcpy2D header failed: " << cudaGetErrorString(hdr_err)
              << std::endl;
    return false;
  }

  std::vector<__half> mask_tmp_h(mask_len);

  for (std::size_t i = 0; i < dims.num; ++i) {
    const std::size_t hb = i * 6;
    const float score = __half2float(header_h[hb + 4]);
    const int cls = static_cast<int>(__half2float(header_h[hb + 5]));
    if (cls < 0) continue;
    if (score < class_threshold(cls, detectionParams)) continue;

    const float x1 = __half2float(header_h[hb + 0]);
    const float y1 = __half2float(header_h[hb + 1]);
    const float x2 = __half2float(header_h[hb + 2]);
    const float y2 = __half2float(header_h[hb + 3]);

    NvDsInferInstanceMaskInfo obj{};
    if (!set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj)) continue;

    const cudaError_t m_err = cudaMemcpy(
        mask_tmp_h.data(),
        dev + (i * dims.channels + 6),
        mask_len * sizeof(__half),
        cudaMemcpyDeviceToHost);
    if (m_err != cudaSuccess) {
      std::cerr << "[yolo26-seg] cudaMemcpy mask failed: " << cudaGetErrorString(m_err)
                << std::endl;
      continue;
    }

    obj.classId = cls;
    obj.detectionConfidence = score;

    obj.mask_width = mask_side;
    obj.mask_height = mask_side;
    obj.mask_size = static_cast<unsigned>(mask_len * sizeof(float));
    obj.mask = new float[mask_len];
    for (std::size_t j = 0; j < mask_len; ++j) {
      obj.mask[j] = __half2float(mask_tmp_h[j]);
    }

    objectList.emplace_back(obj);
  }

  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Seg);
