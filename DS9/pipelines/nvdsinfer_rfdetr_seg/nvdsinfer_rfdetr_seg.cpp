#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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
  // Numerically stable sigmoid.
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
    std::cerr << "[rfdetr-seg] missing " << label << " buffer" << std::endl;
    return false;
  }
  if (layer.dataType != FLOAT) {
    std::cerr << "[rfdetr-seg] " << label << " must be FLOAT" << std::endl;
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
  std::size_t h = 0;
  std::size_t w = 0;
};

bool parse_boxes_dims(const NvDsInferLayerInfo& boxes, ParsedLayerDims& out) {
  const int nd = boxes.inferDims.numDims;
  if (nd == 2) {
    // [Q, 4]
    if (boxes.inferDims.d[1] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[0]);
    return out.q == 100 || out.q == 200 || out.q == 300;
  }
  if (nd == 3) {
    // [1, Q, 4] for a per-frame parser invocation.
    if (boxes.inferDims.d[0] != 1 || boxes.inferDims.d[2] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[1]);
    return out.q == 100 || out.q == 200 || out.q == 300;
  }
  return false;
}

bool parse_logits_dims(const NvDsInferLayerInfo& logits, ParsedLayerDims& out) {
  const int nd = logits.inferDims.numDims;
  if (nd == 2) {
    // [Q, C]
    out.q = static_cast<std::size_t>(logits.inferDims.d[0]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[1]);
    return out.c == 91;
  }
  if (nd == 3) {
    // [1, Q, C] for a per-frame parser invocation.
    if (logits.inferDims.d[0] != 1) return false;
    out.q = static_cast<std::size_t>(logits.inferDims.d[1]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[2]);
    return out.c == 91;
  }
  return false;
}

bool parse_masks_dims(const NvDsInferLayerInfo& masks, ParsedLayerDims& out) {
  const int nd = masks.inferDims.numDims;
  if (nd == 3) {
    // [Q, Hm, Wm]
    out.q = static_cast<std::size_t>(masks.inferDims.d[0]);
    out.h = static_cast<std::size_t>(masks.inferDims.d[1]);
    out.w = static_cast<std::size_t>(masks.inferDims.d[2]);
    return out.h > 0 && out.w > 0;
  }
  if (nd == 4) {
    // [1, Q, Hm, Wm] for a per-frame parser invocation.
    if (masks.inferDims.d[0] != 1) return false;
    out.q = static_cast<std::size_t>(masks.inferDims.d[1]);
    out.h = static_cast<std::size_t>(masks.inferDims.d[2]);
    out.w = static_cast<std::size_t>(masks.inferDims.d[3]);
    return out.h > 0 && out.w > 0;
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
              NvDsInferInstanceMaskInfo& obj) {
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
  x1 = (cx - width * 0.5f) * static_cast<float>(net_w);
  y1 = (cy - height * 0.5f) * static_cast<float>(net_h);
  x2 = (cx + width * 0.5f) * static_cast<float>(net_w);
  y2 = (cy + height * 0.5f) * static_cast<float>(net_h);
  return std::isfinite(x1) && std::isfinite(y1) &&
         std::isfinite(x2) && std::isfinite(y2);
}

float bilinear_mask_value(const float* mask_src, std::size_t mask_h,
                          std::size_t mask_w, float source_x, float source_y) {
  const float clamped_x = clamp(
      source_x, 0.0f, static_cast<float>(mask_w - 1));
  const float clamped_y = clamp(
      source_y, 0.0f, static_cast<float>(mask_h - 1));
  const std::size_t x0 = static_cast<std::size_t>(std::floor(clamped_x));
  const std::size_t y0 = static_cast<std::size_t>(std::floor(clamped_y));
  const std::size_t x1 = std::min(x0 + 1, mask_w - 1);
  const std::size_t y1 = std::min(y0 + 1, mask_h - 1);
  const float wx = clamped_x - static_cast<float>(x0);
  const float wy = clamped_y - static_cast<float>(y0);
  const float top =
      mask_src[y0 * mask_w + x0] * (1.0f - wx) +
      mask_src[y0 * mask_w + x1] * wx;
  const float bottom =
      mask_src[y1 * mask_w + x0] * (1.0f - wx) +
      mask_src[y1 * mask_w + x1] * wx;
  return top * (1.0f - wy) + bottom * wy;
}

bool copy_mask_roi_bilinear(const float* mask_src, std::size_t mask_h,
                            std::size_t mask_w, unsigned net_w, unsigned net_h,
                            NvDsInferInstanceMaskInfo& obj) {
  if (!mask_src || mask_h == 0 || mask_w == 0 || net_w == 0 || net_h == 0) return false;
  if (!std::isfinite(obj.left) || !std::isfinite(obj.top) ||
      !std::isfinite(obj.width) || !std::isfinite(obj.height) ||
      obj.width <= 0.0f || obj.height <= 0.0f) {
    return false;
  }

  // NvDsInferInstanceMaskInfo stores a bbox-relative mask. Sample that ROI
  // from the full RF-DETR mask-logit plane with align_corners=False geometry,
  // rather than copying an integer low-resolution crop and changing its scale.
  const std::size_t roi_w = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(
             obj.width * static_cast<float>(mask_w) /
             static_cast<float>(net_w))));
  const std::size_t roi_h = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(
             obj.height * static_cast<float>(mask_h) /
             static_cast<float>(net_h))));
  if (roi_w == 0 || roi_h == 0) return false;
  if (roi_h > std::numeric_limits<std::size_t>::max() / roi_w ||
      roi_w * roi_h >
          static_cast<std::size_t>(std::numeric_limits<unsigned>::max()) /
              sizeof(float)) {
    return false;
  }

  obj.mask_width = static_cast<unsigned>(roi_w);
  obj.mask_height = static_cast<unsigned>(roi_h);
  obj.mask_size = static_cast<unsigned>(roi_w * roi_h * sizeof(float));
  obj.mask = new float[roi_w * roi_h];

  for (std::size_t y = 0; y < roi_h; ++y) {
    const float network_y =
        obj.top + (static_cast<float>(y) + 0.5f) * obj.height /
                      static_cast<float>(roi_h);
    const float source_y =
        network_y * static_cast<float>(mask_h) / static_cast<float>(net_h) -
        0.5f;
    for (std::size_t x = 0; x < roi_w; ++x) {
      const float network_x =
          obj.left + (static_cast<float>(x) + 0.5f) * obj.width /
                         static_cast<float>(roi_w);
      const float source_x =
          network_x * static_cast<float>(mask_w) /
              static_cast<float>(net_w) -
          0.5f;
      const float logit =
          bilinear_mask_value(mask_src, mask_h, mask_w, source_x, source_y);
      if (!std::isfinite(logit)) {
        delete[] obj.mask;
        obj.mask = nullptr;
        obj.mask_width = 0;
        obj.mask_height = 0;
        obj.mask_size = 0;
        return false;
      }
      obj.mask[y * roi_w + x] = sigmoid(logit);
    }
  }

  return true;
}

}  // namespace

extern "C" bool NvDsInferParseRFDETRSeg(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  if (outputLayersInfo.size() != 3) {
    std::cerr << "[rfdetr-seg] expected exactly 3 output layers (dets, labels, masks), got "
              << outputLayersInfo.size() << std::endl;
    return false;
  }
  if (networkInfo.width == 0 || networkInfo.height == 0) {
    std::cerr << "[rfdetr-seg] network dimensions must be positive" << std::endl;
    return false;
  }

  // Read outputs by name. Require output-blob-names in the PGIE INI to include:
  //   dets   -> boxes [Q,4] or [B,Q,4] (normalized cxcywh)
  //   labels -> logits [Q,C] or [B,Q,C] (RF-DETR uses sigmoid scores; COCO ids are 1..90)
  //   masks  -> mask logits [Q,Hm,Wm] or [B,Q,Hm,Wm]
  const NvDsInferLayerInfo* boxes_layer = find_layer_by_name(outputLayersInfo, "dets");
  const NvDsInferLayerInfo* logits_layer = find_layer_by_name(outputLayersInfo, "labels");
  const NvDsInferLayerInfo* masks_layer = find_layer_by_name(outputLayersInfo, "masks");
  if (!boxes_layer || !logits_layer || !masks_layer) {
    std::cerr << "[rfdetr-seg] missing required output layers. Found:";
    for (const auto& layer : outputLayersInfo) {
      std::cerr << " " << (layer.layerName ? layer.layerName : "<null>");
    }
    std::cerr << std::endl;
    return false;
  }

  if (!validate_float_layer(*boxes_layer, "dets")) return false;
  if (!validate_float_layer(*logits_layer, "labels")) return false;
  if (!validate_float_layer(*masks_layer, "masks")) return false;

  ParsedLayerDims boxes_dims{};
  ParsedLayerDims logits_dims{};
  ParsedLayerDims masks_dims{};
  if (!parse_boxes_dims(*boxes_layer, boxes_dims) || boxes_dims.q == 0) {
    std::cerr << "[rfdetr-seg] unexpected boxes dims (expected [Q,4] or [1,Q,4])" << std::endl;
    return false;
  }
  if (!parse_logits_dims(*logits_layer, logits_dims) || logits_dims.q == 0 || logits_dims.c == 0) {
    std::cerr << "[rfdetr-seg] unexpected logits dims (expected [Q,C] or [1,Q,C])" << std::endl;
    return false;
  }
  if (!parse_masks_dims(*masks_layer, masks_dims) || masks_dims.q == 0 || masks_dims.h == 0 ||
      masks_dims.w == 0) {
    std::cerr << "[rfdetr-seg] unexpected masks dims (expected [Q,H,W] or [1,Q,H,W])" << std::endl;
    return false;
  }

  const std::size_t q = logits_dims.q;
  const std::size_t c = logits_dims.c;
  if (boxes_dims.q != q || masks_dims.q != q) {
    std::cerr << "[rfdetr-seg] Q mismatch: boxes.q=" << boxes_dims.q << " logits.q=" << q
              << " masks.q=" << masks_dims.q << std::endl;
    return false;
  }
  if (masks_dims.h * 4 != networkInfo.height ||
      masks_dims.w * 4 != networkInfo.width) {
    std::cerr << "[rfdetr-seg] mask plane must be exactly network resolution / 4"
              << std::endl;
    return false;
  }

  // RF-DETR 1.8.3 COCO uses category IDs as class indices (max_obj_id=90);
  // person is fixed at COCO category/index 1.
  constexpr std::size_t kPersonClassIndex = 1;
  const int debug_every = getenv_int("NOESIS_RFDETR_DEBUG_EVERY", 0);
  if (kPersonClassIndex >= c) {
    std::cerr << "[rfdetr-seg] COCO person class index " << kPersonClassIndex
              << " is unavailable (C=" << c << ")" << std::endl;
    return false;
  }

  const float threshold =
      !detectionParams.perClassPreclusterThreshold.empty() ? detectionParams.perClassPreclusterThreshold[0] : 0.0f;

  const auto* boxes_dev = static_cast<const float*>(boxes_layer->buffer);
  const auto* logits_dev = static_cast<const float*>(logits_layer->buffer);
  const auto* masks_dev = static_cast<const float*>(masks_layer->buffer);

  const std::size_t box_stride = 4;
  const std::size_t logit_stride = c;
  const std::size_t mask_stride = masks_dims.h * masks_dims.w;
  const bool boxes_on_device = is_device_ptr(boxes_layer->buffer);
  const bool logits_on_device = is_device_ptr(logits_layer->buffer);
  const bool masks_on_device = is_device_ptr(masks_layer->buffer);
  const bool device_buf = boxes_on_device || logits_on_device || masks_on_device;

  if (device_buf && !(boxes_on_device && logits_on_device && masks_on_device)) {
    std::cerr << "[rfdetr-seg] inconsistent tensor buffer locations: boxes_device=" << boxes_on_device
              << " logits_device=" << logits_on_device << " masks_device=" << masks_on_device << std::endl;
    return false;
  }

  std::vector<float> boxes_host;
  std::vector<float> person_logits_host;
  std::vector<float> logits_debug_host;
  std::vector<float> mask_scratch;
  if (device_buf) {
    boxes_host.resize(q * box_stride);
    const cudaError_t box_err =
        cudaMemcpy(boxes_host.data(), boxes_dev, boxes_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (box_err != cudaSuccess) {
      std::cerr << "[rfdetr-seg] cudaMemcpy boxes failed: " << cudaGetErrorString(box_err) << std::endl;
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
      std::cerr << "[rfdetr-seg] cudaMemcpy2D person logits failed: " << cudaGetErrorString(person_err)
                << std::endl;
      return false;
    }

    if (debug_every > 0) {
      logits_debug_host.resize(q * c);
      const cudaError_t logits_err =
          cudaMemcpy(logits_debug_host.data(), logits_dev, logits_debug_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
      if (logits_err != cudaSuccess) {
        std::cerr << "[rfdetr-seg] cudaMemcpy logits debug failed: " << cudaGetErrorString(logits_err)
                  << std::endl;
        return false;
      }
    }

    mask_scratch.resize(mask_stride);
  }

  objectList.clear();
  objectList.reserve(q);
  std::size_t kept = 0;
  std::size_t score_pass = 0;
  std::size_t bbox_pass = 0;
  std::size_t mask_pass = 0;
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
    // RF-DETR uses sigmoid probabilities (per-class, not softmax). For Noesis
    // parity we only emit "person" detections as DeepStream classId 0.
    const float person_logit = device_buf
        ? person_logits_host[i]
        : logits_dev[i * logit_stride + kPersonClassIndex];
    if (!std::isfinite(person_logit)) {
      std::cerr << "[rfdetr-seg] non-finite person logit at query " << i << std::endl;
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
      std::cerr << "[rfdetr-seg] invalid normalized cxcywh box at query " << i
                << std::endl;
      return false;
    }

    NvDsInferInstanceMaskInfo obj{};
    set_bbox(x1, y1, x2, y2, networkInfo.width, networkInfo.height, obj);
    if (obj.width < 1.0f || obj.height < 1.0f) continue;
    ++bbox_pass;

    obj.classId = 0;  // DS "person"
    obj.detectionConfidence = score;

    const float* mask_src = nullptr;
    if (device_buf) {
      const cudaError_t mask_err =
          cudaMemcpy(mask_scratch.data(), masks_dev + i * mask_stride, mask_stride * sizeof(float), cudaMemcpyDeviceToHost);
      if (mask_err != cudaSuccess) {
        std::cerr << "[rfdetr-seg] cudaMemcpy mask failed: " << cudaGetErrorString(mask_err) << std::endl;
        continue;
      }
      mask_src = mask_scratch.data();
    } else {
      mask_src = masks_dev + i * mask_stride;
    }
    if (!copy_mask_roi_bilinear(
            mask_src, masks_dims.h, masks_dims.w, networkInfo.width,
            networkInfo.height, obj)) {
      continue;
    }
    ++mask_pass;

    objectList.emplace_back(obj);
    ++kept;
  }

  if (debug_every > 0) {
    const auto n = g_frame_counter.fetch_add(1) + 1;
    const unsigned long long den = static_cast<unsigned long long>(debug_every);
    if (den > 0 && (n % den) == 0ULL) {
      std::cerr << "[rfdetr-seg] frame=" << n << " q=" << q << " kept=" << kept
                << " person_idx=" << kPersonClassIndex << " threshold=" << threshold
                << " device_buf=" << (device_buf ? 1 : 0)
                << " score_pass=" << score_pass << " bbox_pass=" << bbox_pass
                << " mask_pass=" << mask_pass
                << " max_person_score=" << max_person_score
                << " max_nonbg_score=" << max_nonbg_score
                << " max_nonbg_cls=" << max_nonbg_cls << std::endl;
    }
  }

  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseRFDETRSeg);
