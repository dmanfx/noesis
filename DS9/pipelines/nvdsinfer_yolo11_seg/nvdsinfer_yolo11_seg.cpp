// SPDX-FileCopyrightText: 2025 Noesis
// SPDX-License-Identifier: MIT
//
// DeepStream 9.1 custom parser for YOLO11-Seg (fused output)
//
// Assumes the ONNX was exported with TensorRT plugins:
//  - TRT::EfficientNMSX_TRT returns [x1,y1,x2,y2,score,class]
//  - TRT::ROIAlignX_TRT + matmul compose per-detection masks at net/4
// The output tensor layout per detection row is:
//  [x1, y1, x2, y2, score, class_id, mask_flattened...]
//
// Build example:
//  g++ -std=c++17 -shared -fPIC -O2 \
//    -I/opt/nvidia/deepstream/deepstream/sources/includes \
//    -o libnvdsinfer_yolo11_seg.so nvdsinfer_yolo11_seg.cpp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

template <typename T>
inline T clamp(T v, T lo, T hi) { return std::min(hi, std::max(lo, v)); }

static void addBBoxProposal(float x1, float y1, float x2, float y2,
                            unsigned netW, unsigned netH,
                            int classId, float score,
                            NvDsInferInstanceMaskInfo& b) {
  x1 = clamp(x1, 0.f, (float)netW);
  y1 = clamp(y1, 0.f, (float)netH);
  x2 = clamp(x2, 0.f, (float)netW);
  y2 = clamp(y2, 0.f, (float)netH);

  b.left   = x1;
  b.top    = y1;
  b.width  = clamp(x2 - x1, 0.f, (float)netW);
  b.height = clamp(y2 - y1, 0.f, (float)netH);

  if (b.width < 1.f || b.height < 1.f) return;

  b.classId = classId;
  b.detectionConfidence = score;
}

static void addSegProposal(const float* maskData, std::size_t maskLen,
                           unsigned netW, unsigned netH,
                           NvDsInferInstanceMaskInfo& b) {
  if (!maskData || maskLen == 0) return;
  // Infer mask grid from vector length (supports 128x128, 160x160, etc.)
  unsigned side = (unsigned)std::lround(std::sqrt((double)maskLen));
  if ((std::size_t)side * (std::size_t)side != maskLen) {
    // Fallback to net/4 if length is unexpected
    side = std::max(1u, (unsigned)(netW / 4));
    // Optionally: log a warning here
  }
  b.mask_width  = side;
  b.mask_height = side;
  const std::size_t expected = (std::size_t)side * (std::size_t)side;
  b.mask_size = (unsigned)(expected * sizeof(float));
  b.mask = new float[expected];
  // Copy exactly expected elements; if exporter emitted less/more, clamp
  const std::size_t copyLen = std::min(maskLen, expected);
  std::memcpy(b.mask, maskData, copyLen * sizeof(float));
  if (copyLen < expected) {
    std::memset(b.mask + copyLen, 0, (expected - copyLen) * sizeof(float));
  }
}

static bool parseFused(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& net,
    const NvDsInferParseDetectionParams& det,
    std::vector<NvDsInferInstanceMaskInfo>& out) {
  if (layers.empty()) {
    std::cerr << "[yolo11-seg] no output layers" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& out0 = layers[0];
  if (!out0.buffer || out0.dataType != FLOAT) {
    std::cerr << "[yolo11-seg] invalid output0 buffer/datatype" << std::endl;
    return false;
  }

  const int nd = out0.inferDims.numDims;
  if (nd < 2) {
    std::cerr << "[yolo11-seg] unexpected output dims" << std::endl;
    return false;
  }

  // Support [num, channels] or [B, num, channels]
  std::size_t num = 0, channels = 0;
  if (nd == 2) {
    num = (std::size_t)out0.inferDims.d[0];
    channels = (std::size_t)out0.inferDims.d[1];
  } else {
    num = (std::size_t)out0.inferDims.d[1];
    channels = (std::size_t)out0.inferDims.d[2];
  }
  if (channels < 7) {
    std::cerr << "[yolo11-seg] channels too small: " << channels << std::endl;
    return false;
  }

  const std::size_t maskLen = channels - 6; // [x1,y1,x2,y2,score,class] + mask
  const float* buf = static_cast<const float*>(out0.buffer);

  out.clear();
  out.reserve(num);

  for (std::size_t i = 0; i < num; ++i) {
    const std::size_t base = i * channels;
    const float score = buf[base + 4];
    const int cls     = static_cast<int>(buf[base + 5]);

    // Apply per-class precluster threshold (parser-side minimal filter)
    if (cls < 0 || (std::size_t)cls >= det.perClassPreclusterThreshold.size()) continue;
    if (score < det.perClassPreclusterThreshold[cls]) continue;

    NvDsInferInstanceMaskInfo b{};
    addBBoxProposal(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3],
                    net.width, net.height, cls, score, b);
    if (b.width < 1.f || b.height < 1.f) continue;

    addSegProposal(buf + base + 6, maskLen, net.width, net.height, b);
    out.emplace_back(b);
  }

  return true;
}

} // namespace

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                      NvDsInferNetworkInfo const& networkInfo,
                      NvDsInferParseDetectionParams const& detectionParams,
                      std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  return parseFused(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloSeg);
