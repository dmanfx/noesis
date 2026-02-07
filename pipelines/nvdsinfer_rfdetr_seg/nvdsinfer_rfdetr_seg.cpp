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
    return true;
  }
  if (nd == 3) {
    // [B, Q, 4] (B should be 1 for per-batch parsing)
    if (boxes.inferDims.d[2] != 4) return false;
    out.q = static_cast<std::size_t>(boxes.inferDims.d[1]);
    return true;
  }
  return false;
}

bool parse_logits_dims(const NvDsInferLayerInfo& logits, ParsedLayerDims& out) {
  const int nd = logits.inferDims.numDims;
  if (nd == 2) {
    // [Q, C]
    out.q = static_cast<std::size_t>(logits.inferDims.d[0]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[1]);
    return out.c > 0;
  }
  if (nd == 3) {
    // [B, Q, C] (B should be 1)
    out.q = static_cast<std::size_t>(logits.inferDims.d[1]);
    out.c = static_cast<std::size_t>(logits.inferDims.d[2]);
    return out.c > 0;
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
    // [B, Q, Hm, Wm] (B should be 1)
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

void decode_box_to_xyxy(const float* box4, unsigned net_w, unsigned net_h,
                        float& x1, float& y1, float& x2, float& y2) {
  const float a = box4[0];
  const float b = box4[1];
  const float c = box4[2];
  const float d = box4[3];

  // Preferred path for current export graph:
  // dets are x1,y1,x2,y2 either normalized [0..1] or absolute pixels.
  if (c > a && d > b) {
    const bool looks_normalized =
        (a >= -0.5f && b >= -0.5f && c <= 1.5f && d <= 1.5f);
    if (looks_normalized) {
      x1 = a * static_cast<float>(net_w);
      y1 = b * static_cast<float>(net_h);
      x2 = c * static_cast<float>(net_w);
      y2 = d * static_cast<float>(net_h);
    } else {
      x1 = a;
      y1 = b;
      x2 = c;
      y2 = d;
    }
    return;
  }

  // Backward-compatible fallback: cx,cy,w,h normalized.
  const float cx = a;
  const float cy = b;
  const float bw = c;
  const float bh = d;
  x1 = (cx - bw * 0.5f) * static_cast<float>(net_w);
  y1 = (cy - bh * 0.5f) * static_cast<float>(net_h);
  x2 = (cx + bw * 0.5f) * static_cast<float>(net_w);
  y2 = (cy + bh * 0.5f) * static_cast<float>(net_h);
}

bool copy_mask_roi(const float* mask_src, std::size_t mask_h, std::size_t mask_w,
                   float x1, float y1, float x2, float y2, unsigned net_w, unsigned net_h,
                   NvDsInferInstanceMaskInfo& obj) {
  if (!mask_src || mask_h == 0 || mask_w == 0 || net_w == 0 || net_h == 0) return false;

  const float fx1 = (x1 / static_cast<float>(net_w)) * static_cast<float>(mask_w);
  const float fy1 = (y1 / static_cast<float>(net_h)) * static_cast<float>(mask_h);
  const float fx2 = (x2 / static_cast<float>(net_w)) * static_cast<float>(mask_w);
  const float fy2 = (y2 / static_cast<float>(net_h)) * static_cast<float>(mask_h);

  const int ix1 = clamp(static_cast<int>(std::floor(fx1)), 0, static_cast<int>(mask_w) - 1);
  const int iy1 = clamp(static_cast<int>(std::floor(fy1)), 0, static_cast<int>(mask_h) - 1);
  const int ix2 = clamp(static_cast<int>(std::ceil(fx2)), ix1 + 1, static_cast<int>(mask_w));
  const int iy2 = clamp(static_cast<int>(std::ceil(fy2)), iy1 + 1, static_cast<int>(mask_h));

  const std::size_t roi_w = static_cast<std::size_t>(ix2 - ix1);
  const std::size_t roi_h = static_cast<std::size_t>(iy2 - iy1);
  if (roi_w == 0 || roi_h == 0) return false;

  obj.mask_width = static_cast<unsigned>(roi_w);
  obj.mask_height = static_cast<unsigned>(roi_h);
  obj.mask_size = static_cast<unsigned>(roi_w * roi_h * sizeof(float));
  obj.mask = new float[roi_w * roi_h];

  for (std::size_t y = 0; y < roi_h; ++y) {
    const float* src_row =
        mask_src + (static_cast<std::size_t>(iy1) + y) * mask_w + static_cast<std::size_t>(ix1);
    float* dst_row = obj.mask + y * roi_w;
    for (std::size_t x = 0; x < roi_w; ++x) {
      dst_row[x] = sigmoid(src_row[x]);
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
  if (outputLayersInfo.size() < 3) {
    std::cerr << "[rfdetr-seg] expected >=3 output layers (boxes, logits, masks), got "
              << outputLayersInfo.size() << std::endl;
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

  // RF-DETR (coco) uses COCO category IDs as class indices (max_obj_id=90),
  // so "person" is typically class index 1 (COCO category id 1).
  const int person_class_idx = getenv_int("NOESIS_RFDETR_PERSON_CLASS_IDX", 1);
  const int debug_every = getenv_int("NOESIS_RFDETR_DEBUG_EVERY", 0);
  if (person_class_idx < 0 || static_cast<std::size_t>(person_class_idx) >= c) {
    std::cerr << "[rfdetr-seg] invalid NOESIS_RFDETR_PERSON_CLASS_IDX=" << person_class_idx
              << " (C=" << c << ")" << std::endl;
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
        logits_dev + static_cast<std::size_t>(person_class_idx),
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
    // RF-DETR uses sigmoid probabilities (per-class, not softmax). For DS8 parity
    // we only emit "person" detections as DS classId 0.
    const float person_logit = device_buf
        ? person_logits_host[i]
        : logits_dev[i * logit_stride + static_cast<std::size_t>(person_class_idx)];
    const float score = sigmoid(person_logit);
    if (score > max_person_score) max_person_score = score;
    if (score < threshold) continue;
    ++score_pass;

    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    const float* box_ptr = device_buf ? (boxes_host.data() + i * box_stride) : (boxes_dev + i * box_stride);
    decode_box_to_xyxy(box_ptr, networkInfo.width, networkInfo.height, x1, y1, x2, y2);

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
    if (!copy_mask_roi(mask_src, masks_dims.h, masks_dims.w, obj.left, obj.top,
                       obj.left + obj.width, obj.top + obj.height, networkInfo.width,
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
                << " person_idx=" << person_class_idx << " threshold=" << threshold
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
