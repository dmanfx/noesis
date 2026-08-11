#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "metadata.hpp"
#include "nvdsmeta.h"

namespace py = pybind11;

namespace {

struct JsonPayload {
  std::string json;
};

int object_depth_meta_type() {
  static const int type = static_cast<int>(
      nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.OBJECT_DEPTH")));
  return type;
}

void* object_depth_meta_copy(void* opaque_meta, void* /*user_data*/) {
  if (opaque_meta == nullptr) return nullptr;
  deepstream::UserMetadata user_meta(opaque_meta);
  auto* source = static_cast<JsonPayload*>(user_meta.userData());
  return source != nullptr ? static_cast<void*>(new JsonPayload(*source)) : nullptr;
}

void object_depth_meta_release(void* opaque_meta, void* /*user_data*/) {
  if (opaque_meta == nullptr) return;
  deepstream::UserMetadata user_meta(opaque_meta);
  delete static_cast<JsonPayload*>(user_meta.userData());
}

}  // namespace

bool attach_object_depth(
    deepstream::BatchMetadata& batch_meta,
    deepstream::ObjectMetadata& obj_meta,
    const std::string& payload_json,
    bool replace_existing) {
  const int meta_type = object_depth_meta_type();
  if (replace_existing) {
    // DS9 exposes public append but not removal. Update this extension's owned
    // payload in place so replacement preserves one logical metadata record.
    bool found = false;
    bool updated = false;
    obj_meta.iterate(
        [&](const deepstream::UserMetadata& user_meta) {
          if (found) return;
          found = true;
          auto* payload = static_cast<JsonPayload*>(user_meta.userData());
          if (payload != nullptr) {
            payload->json = payload_json;
            updated = true;
          }
        },
        meta_type);
    if (found) {
      return updated;
    }
  }

  deepstream::UserMetadata user_meta(nullptr);
  if (!batch_meta.acquire(user_meta)) {
    return false;
  }
  auto payload = std::make_unique<JsonPayload>(JsonPayload{payload_json});
  user_meta.setMetaType(meta_type);
  user_meta.setUserData(
      payload.get(), object_depth_meta_copy, object_depth_meta_release);
  payload.release();
  obj_meta.append(user_meta);
  return true;
}

py::object extract_object_depth(const deepstream::ObjectMetadata& obj_meta) {
  std::optional<std::string> payload;
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (payload.has_value()) return;
        auto* stored = static_cast<JsonPayload*>(user_meta.userData());
        if (stored != nullptr) payload = stored->json;
      },
      object_depth_meta_type());
  return payload.has_value() ? py::cast(*payload) : py::none();
}

py::object extract_object_mask(const deepstream::ObjectMetadata& obj_meta) {
  const NvOSD_MaskParams& mask = obj_meta.maskParams();
  const int width = static_cast<int>(mask.width);
  const int height = static_cast<int>(mask.height);
  const int size = static_cast<int>(mask.size);
  if (width <= 0 || height <= 0 || size <= 0 || mask.data == nullptr) {
    return py::none();
  }

  const std::size_t expected_elems =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t expected_bytes = expected_elems * sizeof(float);
  const std::size_t raw_size = static_cast<std::size_t>(size);
  if (raw_size < expected_bytes) {
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
  m.doc() =
      "Noesis DS9 public-Service-Maker helper for object depth user metadata.";
  m.def(
      "attach_object_depth",
      &attach_object_depth,
      py::arg("batch_meta"),
      py::arg("obj_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.OBJECT_DEPTH through BatchMetadata/UserMetadata.");
  m.def(
      "extract_object_depth",
      &extract_object_depth,
      py::arg("obj_meta"),
      "Extract NOESIS.OBJECT_DEPTH through public ObjectMetadata iteration.");
  m.def(
      "extract_object_mask",
      &extract_object_mask,
      py::arg("obj_meta"),
      "Copy an object instance mask through ObjectMetadata.maskParams().");
  m.def("object_depth_meta_type", []() { return object_depth_meta_type(); });
}
