#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

#include <gst/gst.h>
#include <glib.h>
#include <cmath>

#include "nvdsmeta.h"
#include "nvds_roi_meta.h"
#include "nvdspreprocess_meta.h"
#include "gstnvdsinfer.h"


#pragma pack(push, 1)
typedef struct {
  float depth_m;
  float conf;
  int32_t sid;
  int32_t flags;
} MaDepthBinV1;
#pragma pack(pop)

namespace {
using std::isfinite;

struct LayerView {
  const float* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 1;
  int batch_stride = 0;

  bool valid() const noexcept {
    return data != nullptr && width > 0 && height > 0 && channels > 0;
  }

  int channel_stride() const noexcept { return width * height; }

  float value_at(int y, int x) const {
    if (!valid()) return 0.0f;
    int spatial_index = y * width + x;
    if (spatial_index < 0 || spatial_index >= width * height) return 0.0f;
    float accum = 0.0f;
    int stride = channel_stride();
    const float* base = data;
    for (int c = 0; c < channels; ++c) {
      accum += base[c * stride + spatial_index];
    }
    if (channels > 1) {
      accum /= static_cast<float>(channels);
    }
    return accum;
  }
};

static std::atomic<int> g_last_roi_count{0};
static std::atomic<int> g_last_attached_count{0};

gpointer depth_bin_copy(gpointer data, gpointer /*user_data*/) {
  if (!data) return nullptr;
  const MaDepthBinV1* src = reinterpret_cast<const MaDepthBinV1*>(data);
  MaDepthBinV1* dst = static_cast<MaDepthBinV1*>(g_malloc(sizeof(MaDepthBinV1)));
  if (!dst) return nullptr;
  *dst = *src;
  g_print("MA depth copy: src=%p dst=%p\n", (const void*)src, (void*)dst);
  return dst;
}

void depth_bin_release(gpointer data, gpointer /*user_data*/) {
  if (!data) return;
  g_print("MA depth release: data=%p\n", data);
  g_free(data);
}

std::string to_lower(const gchar* name) {
  std::string lower;
  if (!name) return lower;
  while (*name) {
    unsigned char ch = static_cast<unsigned char>(*name);
    lower.push_back(static_cast<char>(std::tolower(ch)));
    ++name;
  }
  return lower;
}

bool contains_ci(const gchar* name, const char* token) {
  if (!name || !token) return false;
  std::string lower_name = to_lower(name);
  std::string lower_token = to_lower(token);
  return lower_name.find(lower_token) != std::string::npos;
}

LayerView make_layer_view(const NvDsInferLayerInfo* layer) {
  LayerView view;
  if (!layer || !layer->buffer) {
    return view;
  }
  if (layer->dataType != FLOAT) {
    return view;
  }
  int num_dims = static_cast<int>(layer->inferDims.numDims);
  if (num_dims <= 0) {
    return view;
  }
  int width = 1;
  int height = 1;
  int channels = 1;
  int batch = 1;
  if (num_dims >= 1) {
    width = static_cast<int>(layer->inferDims.d[num_dims - 1]);
  }
  if (num_dims >= 2) {
    height = static_cast<int>(layer->inferDims.d[num_dims - 2]);
  }
  if (num_dims >= 3) {
    channels = static_cast<int>(layer->inferDims.d[num_dims - 3]);
  }
  if (num_dims >= 4) {
    batch = static_cast<int>(layer->inferDims.d[num_dims - 4]);
  }
  if (width <= 0 || height <= 0) {
    return view;
  }
  if (channels <= 0) {
    channels = 1;
  }
  if (batch <= 0) {
    batch = 1;
  }
  view.data = static_cast<const float*>(layer->buffer);
  view.width = width;
  view.height = height;
  view.channels = channels;
  view.batch_stride = channels * width * height;
  return view;
}

struct DepthComputationResult {
  NvDsObjectMeta* obj = nullptr;
  float depth_m = 0.0f;
  float conf = 0.0f;
  int flags = 0;
};

bool has_depth_meta(const NvDsObjectMeta* obj, NvDsMetaType target_meta) {
  if (!obj) return false;
  for (NvDsMetaList* l = obj->obj_user_meta_list; l != nullptr; l = l->next) {
    NvDsUserMeta* um = reinterpret_cast<NvDsUserMeta*>(l->data);
    if (!um) continue;
    if (um->base_meta.meta_type == target_meta) {
      return true;
    }
  }
  return false;
}

NvDsInferTensorMeta* find_tensor_meta(NvDsObjectMeta* obj,
                                      unsigned int sgie_uid) {
  if (!obj) return nullptr;
  for (NvDsMetaList* l = obj->obj_user_meta_list; l != nullptr; l = l->next) {
    NvDsUserMeta* um = reinterpret_cast<NvDsUserMeta*>(l->data);
    if (!um) continue;
    if (um->base_meta.meta_type !=
        static_cast<NvDsMetaType>(NVDSINFER_TENSOR_OUTPUT_META)) {
      continue;
    }
    NvDsInferTensorMeta* tensor =
        reinterpret_cast<NvDsInferTensorMeta*>(um->user_meta_data);
    if (!tensor) continue;
    if (static_cast<unsigned int>(tensor->unique_id) == sgie_uid) {
      return tensor;
    }
  }
  return nullptr;
}

bool compute_depth_from_tensor(const NvDsInferTensorMeta* tensor_meta,
                               const NvDsObjectMeta* obj_meta,
                               float min_conf_threshold,
                               DepthComputationResult* out) {
  if (!tensor_meta || !obj_meta || !out) return false;
  const NvOSD_RectParams& rect = obj_meta->rect_params;
  float width = rect.width;
  float height = rect.height;
  if (!(width > 1e-3f) || !(height > 1e-3f)) {
    return false;
  }

  const NvDsInferLayerInfo* depth_layer = nullptr;
  const NvDsInferLayerInfo* conf_layer = nullptr;
  const NvDsInferLayerInfo* mask_layer = nullptr;
  const NvDsInferLayerInfo* scale_layer = nullptr;
  const NvDsInferLayerInfo* pose_layer = nullptr;

  for (unsigned int i = 0; i < tensor_meta->num_output_layers; ++i) {
    NvDsInferLayerInfo* info = &tensor_meta->output_layers_info[i];
    const gchar* name = info->layerName;
    if (!name) continue;
    if (!depth_layer && contains_ci(name, "depth")) {
      depth_layer = info;
      continue;
    }
    if (!conf_layer && contains_ci(name, "conf")) {
      conf_layer = info;
      continue;
    }
    if (!mask_layer && contains_ci(name, "mask")) {
      mask_layer = info;
      continue;
    }
    if (!scale_layer && contains_ci(name, "scale")) {
      scale_layer = info;
      continue;
    }
    if (!pose_layer && contains_ci(name, "pose")) {
      pose_layer = info;
      continue;
    }
  }

  (void)scale_layer;
  (void)pose_layer;

  if (!depth_layer) {
    return false;
  }

  LayerView depth_view = make_layer_view(depth_layer);
  if (!depth_view.valid()) {
    return false;
  }

  LayerView conf_view = make_layer_view(conf_layer);
  LayerView mask_view = make_layer_view(mask_layer);

  float left = rect.left;
  float top = rect.top;
  float anchor_x = left + (width * 0.5f);
  float anchor_y = top + height;
  float ux = (anchor_x - left) / width;
  float uy = (anchor_y - top) / height;
  if (!isfinite(ux) || !isfinite(uy)) {
    return false;
  }

  if (ux < 0.0f) ux = 0.0f;
  if (ux > 1.0f) ux = 1.0f;
  if (uy < 0.0f) uy = 0.0f;
  if (uy > 1.0f) uy = 1.0f;

  float fx = ux * static_cast<float>(depth_view.width - 1);
  float fy = uy * static_cast<float>(depth_view.height - 1);
  int cx = static_cast<int>(std::lround(fx));
  int cy = static_cast<int>(std::lround(fy));

  std::vector<float> samples;
  samples.reserve(25);
  float conf_accum = 0.0f;
  int conf_samples = 0;

  for (int dy = -2; dy <= 2; ++dy) {
    int y = cy + dy;
    if (y < 0 || y >= depth_view.height) continue;
    for (int dx = -2; dx <= 2; ++dx) {
      int x = cx + dx;
      if (x < 0 || x >= depth_view.width) continue;

      if (mask_view.valid()) {
        float mask_val = mask_view.value_at(y, x);
        if (!isfinite(mask_val) || mask_val < 0.5f) {
          continue;
        }
      }

      if (conf_view.valid()) {
        float conf_val = conf_view.value_at(y, x);
        if (!isfinite(conf_val) || conf_val < min_conf_threshold) {
          continue;
        }
        conf_accum += conf_val;
        ++conf_samples;
      }

      float depth_val = depth_view.value_at(y, x);
      if (!isfinite(depth_val) || depth_val <= 0.0f) {
        continue;
      }
      samples.push_back(depth_val);
    }
  }

  if (samples.empty()) {
    return false;
  }

  std::sort(samples.begin(), samples.end());
  float depth_m = samples[samples.size() / 2];
  if (samples.size() % 2 == 0 && samples.size() >= 2) {
    float other = samples[(samples.size() / 2) - 1];
    depth_m = (depth_m + other) * 0.5f;
  }

  float conf_mean = 1.0f;
  if (conf_view.valid() && conf_samples > 0) {
    conf_mean = conf_accum / static_cast<float>(conf_samples);
  }

  int flags = 0;
  if (depth_m > 0.0f && !samples.empty()) {
    flags |= 0x1;
  }

  out->depth_m = depth_m;
  out->conf = conf_mean;
  out->flags = flags;
  return true;
}

}  // namespace

extern "C" int ds_preprocess_shim_version() {
  return 0x20251015;
}

extern "C" int ds_attach_depth_bin(NvDsBatchMeta* batch_meta,
                                   NvDsObjectMeta* obj_meta,
                                   guint meta_type,
                                   float depth_m,
                                   float conf,
                                   int32_t sid,
                                   int32_t flags) {
  if (!batch_meta || !obj_meta) {
    return 0;
  }

  MaDepthBinV1* payload =
      static_cast<MaDepthBinV1*>(g_malloc(sizeof(MaDepthBinV1)));
  if (!payload) {
    return 0;
  }

  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    g_free(payload);
    return 0;
  }

  payload->depth_m = depth_m;
  payload->conf = conf;
  payload->sid = sid;
  payload->flags = flags;

  user_meta->user_meta_data = payload;
  user_meta->base_meta.meta_type = static_cast<NvDsMetaType>(meta_type);
  user_meta->base_meta.copy_func = depth_bin_copy;
  user_meta->base_meta.release_func = depth_bin_release;
  user_meta->base_meta.batch_meta = batch_meta;

  g_print("MA depth attach: obj=%p um=%p payload=%p depth=%.3f conf=%.3f flags=%d type=%u\n",
          (void*)obj_meta, (void*)user_meta, (void*)payload, depth_m, conf, flags,
          (unsigned)meta_type);
  nvds_add_user_meta_to_obj(obj_meta, user_meta);
  g_last_attached_count.fetch_add(1, std::memory_order_relaxed);
  return 1;
}

extern "C" int ds_sgie_compute_and_attach(NvDsBatchMeta* batch_meta,
                                          unsigned int sgie_uid,
                                          unsigned int depth_meta_type,
                                          float min_conf) {
  if (!batch_meta) {
    return 0;
  }
  g_print("ds_sgie_compute_and_attach called\n");

  int examined_objects = 0;
  int attached_objects = 0;
  std::vector<DepthComputationResult> pending;

  for (NvDsMetaList* frame_list = batch_meta->frame_meta_list; frame_list != nullptr;
       frame_list = frame_list->next) {
    NvDsFrameMeta* frame_meta =
        reinterpret_cast<NvDsFrameMeta*>(frame_list->data);
    if (!frame_meta) continue;

    for (NvDsMetaList* obj_list = frame_meta->obj_meta_list; obj_list != nullptr;
         obj_list = obj_list->next) {
      NvDsObjectMeta* obj_meta =
          reinterpret_cast<NvDsObjectMeta*>(obj_list->data);
      if (!obj_meta) continue;
      ++examined_objects;

      if (has_depth_meta(obj_meta, static_cast<NvDsMetaType>(depth_meta_type))) {
        continue;
      }

      NvDsInferTensorMeta* tensor_meta =
          find_tensor_meta(obj_meta, sgie_uid);
      if (!tensor_meta) {
        continue;
      }

      DepthComputationResult result;
      result.obj = obj_meta;
      if (compute_depth_from_tensor(tensor_meta, obj_meta, min_conf, &result)) {
        if (result.flags & 0x1) {
          pending.push_back(result);
        }
      }
    }
  }

  for (const DepthComputationResult& result : pending) {
    if (!result.obj) continue;
    int attached =
        ds_attach_depth_bin(batch_meta,
                            result.obj,
                            static_cast<guint>(depth_meta_type),
                            result.depth_m,
                            result.conf,
                            -1,
                            result.flags);
    if (attached > 0) {
      ++attached_objects;
    }
  }

  g_last_roi_count.store(examined_objects, std::memory_order_relaxed);
  g_last_attached_count.store(attached_objects, std::memory_order_relaxed);
  return attached_objects;
}
