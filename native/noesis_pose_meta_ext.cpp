#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include <glib.h>

#include "metadata.hpp"
#include "nvdsmeta.h"

namespace py = pybind11;

namespace {

struct ObjMetaAccessor : public deepstream::ObjectMetadata {
  using deepstream::Metadata::data_;
};

struct FrameMetaAccessor : public deepstream::FrameMetadata {
  using deepstream::Metadata::data_;
};

static bool s_pose_meta_inited = false;
static NvDsMetaType s_pose_meta_type = NVDS_USER_META;
NvDsMetaType pose_meta_type() {
  if (!s_pose_meta_inited) {
    s_pose_meta_type = nvds_get_user_meta_type((gchar*)"NOESIS.POSE_FEATURES");
    s_pose_meta_inited = true;
  }
  return s_pose_meta_type;
}

gpointer pose_meta_copy(gpointer data, gpointer /*user_data*/) {
  // DeepStream calls user-meta copy/release with NvDsUserMeta* (not user_meta_data).
  // Return a deep copy of user_meta_data; DeepStream assigns it to the copied meta.
  if (!data) return nullptr;
  auto* user_meta = static_cast<NvDsUserMeta*>(data);
  if (!user_meta->user_meta_data) return nullptr;
  return g_strdup(static_cast<const gchar*>(user_meta->user_meta_data));
}

void pose_meta_release(gpointer data, gpointer /*user_data*/) {
  // DeepStream calls user-meta copy/release with NvDsUserMeta* (not user_meta_data).
  // Free only the user_meta_data we allocated in attach_pose_features*.
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

NvDsFrameMeta* unwrap_frame_meta(const deepstream::FrameMetadata& frame_meta) {
  auto* accessor = reinterpret_cast<const FrameMetaAccessor*>(&frame_meta);
  return reinterpret_cast<NvDsFrameMeta*>(accessor->data_);
}

}  // namespace

bool attach_pose_features(const deepstream::ObjectMetadata& obj_meta,
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

  NvDsMetaType meta_type = pose_meta_type();
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
  user_meta->base_meta.copy_func = pose_meta_copy;
  user_meta->base_meta.release_func = pose_meta_release;
  user_meta->base_meta.batch_meta = batch_meta;
  nvds_add_user_meta_to_obj(obj, user_meta);
  return true;
}

bool attach_pose_features_frame(const deepstream::FrameMetadata& frame_meta,
                                const std::string& payload_json,
                                bool replace_existing) {
  NvDsFrameMeta* frame = unwrap_frame_meta(frame_meta);
  if (!frame) {
    return false;
  }
  NvDsBatchMeta* batch_meta = frame->base_meta.batch_meta;
  if (!batch_meta) {
    return false;
  }
  NvDsMetaType meta_type = pose_meta_type();
  if (replace_existing) {
    std::vector<NvDsUserMeta*> to_remove;
    for (GList* node = frame->frame_user_meta_list; node != nullptr; node = node->next) {
      auto* user_meta = static_cast<NvDsUserMeta*>(node->data);
      if (user_meta && user_meta->base_meta.meta_type == meta_type) {
        to_remove.push_back(user_meta);
      }
    }
    for (auto* user_meta : to_remove) {
      nvds_remove_user_meta_from_frame(frame, user_meta);
    }
  }

  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    return false;
  }
  user_meta->base_meta.meta_type = meta_type;
  user_meta->user_meta_data = g_strdup(payload_json.c_str());
  user_meta->base_meta.copy_func = pose_meta_copy;
  user_meta->base_meta.release_func = pose_meta_release;
  user_meta->base_meta.batch_meta = batch_meta;
  nvds_add_user_meta_to_frame(frame, user_meta);
  return true;
}

py::object extract_pose_features(const deepstream::ObjectMetadata& obj_meta) {
  NvDsObjectMeta* obj = unwrap_object_meta(obj_meta);
  if (!obj) {
    return py::none();
  }
  NvDsMetaType meta_type = pose_meta_type();
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

PYBIND11_MODULE(noesis_pose_meta_ext, m) {
  m.doc() = "Noesis DS8 helper bindings for attaching pose feature user meta.";
  m.def(
      "attach_pose_features",
      &attach_pose_features,
      py::arg("obj_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES user meta (JSON string) to an object.");
  m.def(
      "attach_pose_features_frame",
      &attach_pose_features_frame,
      py::arg("frame_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES user meta (JSON string) to a frame.");
  m.def(
      "extract_pose_features",
      &extract_pose_features,
      py::arg("obj_meta"),
      "Extract NOESIS.POSE_FEATURES user meta JSON payload from an object.");
  m.def("pose_meta_type", []() { return static_cast<int>(pose_meta_type()); });
}
