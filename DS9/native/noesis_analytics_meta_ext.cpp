#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "metadata.hpp"
#include "nvds_analytics_meta.h"

namespace py = pybind11;

namespace {

int analytics_object_meta_type() {
  return static_cast<int>(NVDS_USER_OBJ_META_NVDSANALYTICS);
}

}  // namespace

py::list extract_analytics(const deepstream::ObjectMetadata& obj_meta) {
  py::list records;
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::AnalyticsObjInfo analytics(user_meta);
        if (!analytics) return;

        py::dict record;
        record["unique_id"] = analytics.getUniqueId();
        record["roiStatus"] = analytics.getRoiStatus();
        record["ocStatus"] = analytics.getOcStatus();
        record["lcStatus"] = analytics.getLcStatus();
        record["dirStatus"] = analytics.getDirStatus();
        record["objStatus"] = analytics.getObjStatus();
        records.append(std::move(record));
      },
      analytics_object_meta_type());
  return records;
}

PYBIND11_MODULE(noesis_analytics_meta_ext, m) {
  m.doc() =
      "Noesis DS9 public-Service-Maker helper for object analytics metadata.";
  m.def(
      "extract_analytics",
      &extract_analytics,
      py::arg("obj_meta"),
      "Extract all NvDsAnalyticsObjInfo records through ObjectMetadata iteration.");
  m.def(
      "analytics_object_meta_type",
      []() { return analytics_object_meta_type(); });
}
