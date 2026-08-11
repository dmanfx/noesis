#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "nvdsinfer_custom_impl.h"

namespace {

std::atomic<unsigned long long> g_frame_counter{0};

template <typename T>
inline T clamp(T v, T lo, T hi) {
  return std::min(hi, std::max(lo, v));
}

inline float sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

int getenv_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  try {
    return std::stoi(std::string(raw));
  } catch (...) {
    return fallback;
  }
}

bool validate_float_layer(const NvDsInferLayerInfo& layer, const char* label) {
  if (!layer.buffer) {
    std::cerr << "[rfdetr] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[rfdetr] " << label << " must be FLOAT" << std::endl;
    return false;
  }
  return true;
}

bool is_device_ptr(const void* p) {
  if (!p) return false;
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, p);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return (attr.type == cudaMemoryTypeDevice) || (attr.type == cudaMemoryTypeManaged);
#else
  return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

struct ParsedLayerDims {
  std::size_t q = 0;
  std::size_t c = 0;
};

bool parse_boxes_dims(const NvDsInferLayerInfo& boxes, ParsedLayerDims& out) {
  const int nd = boxes.inferDims.numDims;
  if (nd == 2) {
    if (boxes.inferDims.d[1] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[0]);
    return out.q == 300;
  }
  if (nd == 3) {
    if (boxes.inferDims.d[0] != 1 || boxes.inferDims.d[2] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[1]);
    return out.q == 300;
  }
  return false;
}

bool parse_logits_dims(const NvDsInferLayerInfo& logits, ParsedLayerDims& out) {
  const int nd = logits.inferDims.numDims;
  if (nd == 2) {
    out.q = static_cast<std::size_t>(logits.inferDims.d[0]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[1]);
    return out.c == 91;
  }
  if (nd == 3) {
    if (logits.inferDims.d[0] != 1) return false;
    out.q = static_cast<std::size_t>(logits.inferDims.d[1]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[2]);
    return out.c == 91;
  }
  return false;
}

const NvDsInferLayerInfo* find_layer_by_name(const std::vector<NvDsInferLayerInfo>& layers,
                                             const char* name) {
  if (!name || !*name) return nullptr;
  for (const auto& layer : layers) {
    if (!layer.layerName) continue;
    if (std::strcmp(layer.layerName, name) == 0) return &layer;
  }
  return nullptr;
}

void set_bbox(float x1, float y1, float x2, float y2, unsigned net_w, unsigned net_h,
              NvDsInferObjectDetectionInfo& obj) {
  x1 = clamp(x1, 0.0f, static_cast<float>(net_w));
  y1 = clamp(y1, 0.0f, static_cast<float>(net_h));
  x2 = clamp(x2, 0.0f, static_cast<float>(net_w));
  y2 = clamp(y2, 0.0f, static_cast<float>(net_h));

  obj.left = x1;
  obj.top = y1;
  obj.width = clamp(x2 - x1, 0.0f, static_cast<float>(net_w));
  obj.height = clamp(y2 - y1, 0.0f, static_cast<float>(net_h));
}

bool decode_box_to_xyxy(const float* box4, unsigned net_w, unsigned net_h,
                        float& x1, float& y1, float& x2, float& y2) {
  if (!box4 || net_w == 0 || net_h == 0) return false;
  const float cx = box4[0];
  const float cy = box4[1];
  const float width = box4[2];
  const float height = box4[3];
  if (!std::isfinite(cx) || !std::isfinite(cy) ||
      !std::isfinite(width) || !std::isfinite(height) ||
      width <= 0.0f || height <= 0.0f) {
    return false;
  }

  // RF-DETR 1.8.3 exports raw pred_boxes in normalized cx,cy,w,h format.
  // Do not guess xyxy from coordinate ordering: valid cxcywh rows frequently
  // satisfy width > cx and height > cy.
  x1 = (cx - width * 0.5f) * static_cast<float>(net_w);
  y1 = (cy - height * 0.5f) * static_cast<float>(net_h);
  x2 = (cx + width * 0.5f) * static_cast<float>(net_w);
  y2 = (cy + height * 0.5f) * static_cast<float>(net_h);
  return std::isfinite(x1) && std::isfinite(y1) &&
         std::isfinite(x2) && std::isfinite(y2);
}

}  // namespace

extern "C" bool NvDsInferParseRFDETR(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  if (outputLayersInfo.size() != 2) {
    std::cerr << "[rfdetr] expected exactly 2 output layers (dets, labels), got "
              << outputLayersInfo.size() << std::endl;
    return false;
  }
  if (networkInfo.width == 0 || networkInfo.height == 0) {
    std::cerr << "[rfdetr] network dimensions must be positive" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo* boxes_layer = find_layer_by_name(outputLayersInfo, "dets");
  const NvDsInferLayerInfo* logits_layer = find_layer_by_name(outputLayersInfo, "labels");
  if (!boxes_layer || !logits_layer) {
    std::cerr << "[rfdetr] missing required output layers. Found:";
    for (const auto& layer : outputLayersInfo) {
      std::cerr << " " << (layer.layerName ? layer.layerName : "<null>");
    }
    std::cerr << std::endl;
    return false;
  }

  if (!validate_float_layer(*boxes_layer, "dets")) return false;
  if (!validate_float_layer(*logits_layer, "labels")) return false;

  ParsedLayerDims boxes_dims{};
  ParsedLayerDims logits_dims{};
  if (!parse_boxes_dims(*boxes_layer, boxes_dims) || boxes_dims.q == 0) {
    std::cerr << "[rfdetr] unexpected boxes dims (expected [Q,4] or [B,Q,4])" << std::endl;
    return false;
  }
  if (!parse_logits_dims(*logits_layer, logits_dims) || logits_dims.q == 0 || logits_dims.c == 0) {
    std::cerr << "[rfdetr] unexpected logits dims (expected [300,91] or [1,300,91])" << std::endl;
    return false;
  }
  if (boxes_dims.q != logits_dims.q) {
    std::cerr << "[rfdetr] Q mismatch: boxes.q=" << boxes_dims.q
              << " logits.q=" << logits_dims.q << std::endl;
    return false;
  }

  const std::size_t q = logits_dims.q;
  const std::size_t c = logits_dims.c;
  constexpr std::size_t kPersonClassIndex = 1;
  const int debug_every = getenv_int("NOESIS_RFDETR_DEBUG_EVERY", 0);
  if (kPersonClassIndex >= c) {
    std::cerr << "[rfdetr] COCO person class index " << kPersonClassIndex
              << " is unavailable (C=" << c << ")" << std::endl;
    return false;
  }

  const float threshold =
      !detectionParams.perClassPreclusterThreshold.empty() ? detectionParams.perClassPreclusterThreshold[0] : 0.0f;

  const auto* boxes_dev = static_cast<const float*>(boxes_layer->buffer);
  const auto* logits_dev = static_cast<const float*>(logits_layer->buffer);
  const std::size_t box_stride = 4;
  const std::size_t logit_stride = c;
  const bool boxes_on_device = is_device_ptr(boxes_layer->buffer);
  const bool logits_on_device = is_device_ptr(logits_layer->buffer);
  const bool device_buf = boxes_on_device || logits_on_device;

  if (device_buf && !(boxes_on_device && logits_on_device)) {
    std::cerr << "[rfdetr] inconsistent tensor buffer locations: boxes_device="
              << boxes_on_device << " logits_device=" << logits_on_device << std::endl;
    return false;
  }

  std::vector<float> boxes_host;
  std::vector<float> person_logits_host;
  std::vector<float> logits_debug_host;
  if (device_buf) {
    boxes_host.resize(q * box_stride);
    const cudaError_t box_err =
        cudaMemcpy(boxes_host.data(), boxes_dev, boxes_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (box_err != cudaSuccess) {
      std::cerr << "[rfdetr] cudaMemcpy boxes failed: " << cudaGetErrorString(box_err) << std::endl;
      return false;
    }

    person_logits_host.resize(q);
    const cudaError_t person_err = cudaMemcpy2D(
        person_logits_host.data(),
        sizeof(float),
        logits_dev + kPersonClassIndex,
        logit_stride * sizeof(float),
        sizeof(float),
        q,
        cudaMemcpyDeviceToHost);
    if (person_err != cudaSuccess) {
      std::cerr << "[rfdetr] cudaMemcpy2D person logits failed: " << cudaGetErrorString(person_err)
                << std::endl;
      return false;
    }

    if (debug_every > 0) {
      logits_debug_host.resize(q * c);
      const cudaError_t logits_err =
          cudaMemcpy(logits_debug_host.data(), logits_dev, logits_debug_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
      if (logits_err != cudaSuccess) {
        std::cerr << "[rfdetr] cudaMemcpy logits debug failed: " << cudaGetErrorString(logits_err)
                  << std::endl;
        return false;
      }
    }
  }

  objectList.clear();
  objectList.reserve(q);
  std::size_t kept = 0;
  std::size_t score_pass = 0;
  std::size_t bbox_pass = 0;
  float max_person_score = 0.0f;
  float max_nonbg_score = 0.0f;
  int max_nonbg_cls = -1;

  for (std::size_t i = 0; i < q; ++i) {
    if (debug_every > 0) {
      const float* logits_row = nullptr;
      if (device_buf && !logits_debug_host.empty()) {
        logits_row = logits_debug_host.data() + i * logit_stride;
      } else if (!device_buf) {
        logits_row = logits_dev + i * logit_stride;
      }
      if (logits_row) {
        for (std::size_t cls = 1; cls < c; ++cls) {
          const float s = sigmoid(logits_row[cls]);
          if (s > max_nonbg_score) {
            max_nonbg_score = s;
            max_nonbg_cls = static_cast<int>(cls);
          }
        }
      }
    }

    const float person_logit = device_buf
        ? person_logits_host[i]
        : logits_dev[i * logit_stride + kPersonClassIndex];
    if (!std::isfinite(person_logit)) {
      std::cerr << "[rfdetr] non-finite person logit at query " << i << std::endl;
      return false;
    }
    const float score = sigmoid(person_logit);
    if (score > max_person_score) max_person_score = score;
    if (score < threshold) continue;
    ++score_pass;

    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    const float* box_ptr = device_buf ? (boxes_host.data() + i * box_stride) : (boxes_dev + i * box_stride);
    if (!decode_box_to_xyxy(
            box_ptr, networkInfo.width, networkInfo.height, x1, y1, x2, y2)) {
      std::cerr << "[rfdetr] invalid normalized cxcywh box at query " << i << std::endl;
      return false;
    }

    NvDsInferObjectDetectionInfo obj{};
    set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj);
    if (obj.width < 1.0f || obj.height < 1.0f) continue;
    ++bbox_pass;

    obj.classId = 0;
    obj.detectionConfidence = score;
    obj.rotation_angle = 0.0f;
    objectList.emplace_back(obj);
    ++kept;
  }

  if (debug_every > 0) {
    const auto n = g_frame_counter.fetch_add(1) + 1;
    const unsigned long long den = static_cast<unsigned long long>(debug_every);
    if (den > 0 && (n % den) == 0ULL) {
      std::cerr << "[rfdetr] frame=" << n << " q=" << q << " kept=" << kept
                << " person_idx=" << kPersonClassIndex << " threshold=" << threshold
                << " device_buf=" << (device_buf ? 1 : 0)
                << " score_pass=" << score_pass << " bbox_pass=" << bbox_pass
                << " max_person_score=" << max_person_score
                << " max_nonbg_score=" << max_nonbg_score
                << " max_nonbg_cls=" << max_nonbg_cls << std::endl;
    }
  }

  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRFDETR);
