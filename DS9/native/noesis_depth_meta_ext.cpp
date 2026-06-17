#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <string>
#include <vector>

#include <glib.h>
#include <gst/gst.h>

#include "metadata.hpp"
#include "nvdsmeta.h"

namespace py = pybind11;

namespace {

struct ObjMetaAccessor : public deepstream::ObjectMetadata {
  using deepstream::Metadata::data_;
};

static bool s_depth_meta_inited = false;
static NvDsMetaType s_depth_meta_type = NVDS_USER_META;

NvDsMetaType object_depth_meta_type() {
  if (!s_depth_meta_inited) {
    s_depth_meta_type = nvds_get_user_meta_type((gchar*)"NOESIS.OBJECT_DEPTH");
    s_depth_meta_inited = true;
  }
  return s_depth_meta_type;
}

gpointer object_depth_meta_copy(gpointer data, gpointer /*user_data*/) {
  if (!data) return nullptr;
  auto* user_meta = static_cast<NvDsUserMeta*>(data);
  if (!user_meta->user_meta_data) return nullptr;
  return g_strdup(static_cast<const gchar*>(user_meta->user_meta_data));
}

void object_depth_meta_release(gpointer data, gpointer /*user_data*/) {
  if (!data) return;
  auto* user_meta = static_cast<NvDsUserMeta*>(data);
  if (user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = nullptr;
  }
}

NvDsObjectMeta* unwrap_object_meta(const deepstream::ObjectMetadata& obj_meta) {
  auto* accessor = reinterpret_cast<const ObjMetaAccessor*>(&obj_meta);
  return reinterpret_cast<NvDsObjectMeta*>(accessor->data_);
}

}  // namespace

bool attach_object_depth(const deepstream::ObjectMetadata& obj_meta,
                         const std::string& payload_json,
                         bool replace_existing) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) {
    return false;
  }
  NvDsBatchMeta* batch_meta = obj->base_meta.batch_meta;
  if (!batch_meta) {
    return false;
  }
  NvDsMetaType meta_type = object_depth_meta_type();
  if (replace_existing) {
    std::vector<NvDsUserMeta*> to_remove;
    for (GList* node = obj->obj_user_meta_list; node != nullptr; node = node->next) {
      auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
      if (user_meta && user_meta->base_meta.meta_type == meta_type) {
        to_remove.push_back(user_meta);
      }
    }
    for (auto* user_meta : to_remove) {
      nvds_remove_user_meta_from_object(obj, user_meta);
    }
  }

  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    return false;
  }
  user_meta->base_meta.meta_type = meta_type;
  user_meta->user_meta_data = g_strdup(payload_json.c_str());
  user_meta->base_meta.copy_func = object_depth_meta_copy;
  user_meta->base_meta.release_func = object_depth_meta_release;
  user_meta->base_meta.batch_meta = batch_meta;
  nvds_add_user_meta_to_obj(obj, user_meta);
  return true;
}

py::object extract_object_depth(const deepstream::ObjectMetadata& obj_meta) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) {
    return py::none();
  }
  NvDsMetaType meta_type = object_depth_meta_type();
  for (GList* node = obj->obj_user_meta_list; node != nullptr; node = node->next) {
    auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
    if (!user_meta || user_meta->base_meta.meta_type != meta_type) {
      continue;
    }
    if (!user_meta->user_meta_data) {
      return py::none();
    }
    const char* payload = static_cast<const char*>(user_meta->user_meta_data);
    if (!payload) {
      return py::none();
    }
    return py::str(payload);
  }
  return py::none();
}

py::object extract_object_mask(const deepstream::ObjectMetadata& obj_meta) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) {
    return py::none();
  }

  const NvOSD_MaskParams& mask = obj->mask_params;
  const int width = static_cast<int>(mask.width);
  const int height = static_cast<int>(mask.height);
  const int size = static_cast<int>(mask.size);
  if (width <= 0 || height <= 0 || size <= 0 || !mask.data) {
    return py::none();
  }

  const std::size_t expected_elems = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t expected_bytes = expected_elems * sizeof(float);
  const std::size_t raw_size = static_cast<std::size_t>(size);
  if (raw_size < expected_elems && raw_size < expected_bytes) {
    return py::none();
  }

  py::array_t<float> array({height, width});
  std::memcpy(array.mutable_data(), mask.data, expected_bytes);

  py::dict payload;
  payload["width"] = width;
  payload["height"] = height;
  payload["threshold"] = static_cast<float>(mask.threshold);
  payload["data"] = std::move(array);
  return payload;
}

PYBIND11_MODULE(noesis_depth_meta_ext, m) {
  m.doc() = "Noesis DS8 helper bindings for attaching object depth user meta.";
  m.def(
      "attach_object_depth",
      &attach_object_depth,
      py::arg("obj_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.OBJECT_DEPTH user meta (JSON string) to an object.");
  m.def(
      "extract_object_depth",
      &extract_object_depth,
      py::arg("obj_meta"),
      "Extract NOESIS.OBJECT_DEPTH user meta JSON payload from an object.");
  m.def(
      "extract_object_mask",
      &extract_object_mask,
      py::arg("obj_meta"),
      "Extract a copied instance-mask payload from an object if available.");
  m.def("object_depth_meta_type", []() { return static_cast<int>(object_depth_meta_type()); });
}
