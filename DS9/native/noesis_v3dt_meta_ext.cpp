#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "metadata.hpp"
#include "nvdsmeta.h"
#include "nvds_tracker_meta.h"

#include <cstdio>
#include <exception>

namespace py = pybind11;

// Static flags to limit logging noise (log once per process)
static bool s_logged_visibility_error = false;
static bool s_logged_image_foot_error = false;
static bool s_logged_bbox3d_error = false;

py::object extract_obj_3d_meta(const deepstream::ObjectMetadata& obj_meta) {
  py::dict result;
  bool found_bbox3d = false;

  // --- Visibility extraction (with try/catch for robustness) ---
  // V3DT-H05: Wrap in try/catch so extraction failures don't crash the pipeline.
  try {
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::ObjectVisibilityUserMetadata visibility_meta(user_meta);
        float visibility = visibility_meta.getVisibility();
        result["visibility"] = visibility;
      },
      NVDS_OBJ_VISIBILITY);
  } catch (const std::exception& e) {
    if (!s_logged_visibility_error) {
      fprintf(stderr,
              "[noesis_v3dt_meta_ext] NVDS_OBJ_VISIBILITY extraction failed "
              "(logged once): %s\n",
              e.what());
      s_logged_visibility_error = true;
    }
  } catch (...) {
    if (!s_logged_visibility_error) {
      fprintf(stderr,
              "[noesis_v3dt_meta_ext] NVDS_OBJ_VISIBILITY extraction failed "
              "(logged once): unknown error\n");
      s_logged_visibility_error = true;
    }
  }

  // --- Image foot location extraction (with try/catch for robustness) ---
  // This is the foot location in image/pixel coordinates.
  // Note: NVDS_OBJ_WORLD_FOOT_LOCATION (world coordinates) is NOT exposed by
  // Service Maker - there is no ObjectWorldFootLocationUserMetadata class.
  // BEV/telemetry must derive world footpoints from bbox3d instead.
  try {
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::ObjectImageFootLocationUserMetadata foot_meta(user_meta);
        auto foot = foot_meta.getImageFootLocation();
        py::list pt;
        pt.append(foot.first);
        pt.append(foot.second);
        result["image_foot"] = std::move(pt);
      },
      NVDS_OBJ_IMAGE_FOOT_LOCATION);
  } catch (const std::exception& e) {
    if (!s_logged_image_foot_error) {
      fprintf(stderr,
              "[noesis_v3dt_meta_ext] NVDS_OBJ_IMAGE_FOOT_LOCATION extraction "
              "failed (logged once): %s\n",
              e.what());
      s_logged_image_foot_error = true;
    }
  } catch (...) {
    if (!s_logged_image_foot_error) {
      fprintf(stderr,
              "[noesis_v3dt_meta_ext] NVDS_OBJ_IMAGE_FOOT_LOCATION extraction "
              "failed (logged once): unknown error\n");
      s_logged_image_foot_error = true;
    }
  }

  // --- 3D bbox extraction (with try/catch for robustness) ---
  // This is the core SV3DT/MV3DT output. Preserve the tracker-profile tuple;
  // the Python producer owns the explicit axis/extent conversion to canonical
  // backend_world_m.
  try {
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::Object3DBBoxUserMetadata bbox_meta(user_meta);
        NvDsObj3DBbox* bbox = bbox_meta.get3DBbox();
        if (bbox == nullptr) {
          return;
        }

        py::dict bbox_dict;
        bbox_dict["xCentre"] = bbox->xCentre;
        bbox_dict["yCentre"] = bbox->yCentre;
        bbox_dict["zCentre"] = bbox->zCentre;
        bbox_dict["xLen"] = bbox->xLen;
        bbox_dict["yLen"] = bbox->yLen;
        bbox_dict["zLen"] = bbox->zLen;
        bbox_dict["xRot"] = bbox->xRot;
        bbox_dict["yRot"] = bbox->yRot;
        bbox_dict["zRot"] = bbox->zRot;

        py::list velocity;
        velocity.append(bbox->xVel);
        velocity.append(bbox->yVel);
        velocity.append(bbox->zVel);

        result["bbox3d"] = std::move(bbox_dict);
        result["velocity3d"] = std::move(velocity);
        found_bbox3d = true;
      },
      NVDS_OBJ_3D_META);
  } catch (const std::exception& e) {
    if (!s_logged_bbox3d_error) {
      fprintf(stderr,
              "[noesis_v3dt_meta_ext] NVDS_OBJ_3D_META extraction failed "
              "(logged once): %s\n",
              e.what());
      s_logged_bbox3d_error = true;
    }
  } catch (...) {
    if (!s_logged_bbox3d_error) {
      fprintf(stderr,
              "[noesis_v3dt_meta_ext] NVDS_OBJ_3D_META extraction failed "
              "(logged once): unknown error\n");
      s_logged_bbox3d_error = true;
    }
  }

  if (!found_bbox3d) {
    return py::none();
  }
  return std::move(result);
}

PYBIND11_MODULE(noesis_v3dt_meta_ext, m) {
  m.doc() =
      "Noesis helper bindings for accessing nvtracker SV3DT/MV3DT user meta "
      "(NVDS_OBJ_3D_META, NVDS_OBJ_VISIBILITY, NVDS_OBJ_IMAGE_FOOT_LOCATION) "
      "from Service Maker Python ObjectMetadata.\n\n"
      "Note: NVDS_OBJ_WORLD_FOOT_LOCATION is NOT exposed by Service Maker; "
      "bbox3d and velocity values remain in the selected tracker profile's "
      "tuple. The producer must validate that profile's extent/axis contract "
      "before publishing canonical world coordinates.";
  m.def("extract_obj_3d_meta", &extract_obj_3d_meta,
        "Extract NVDS_OBJ_3D_META (NvDsObj3DBbox), NVDS_OBJ_VISIBILITY, and "
        "NVDS_OBJ_IMAGE_FOOT_LOCATION from a Service Maker ObjectMetadata. "
        "Extraction failures for individual meta types are logged once and "
        "do not prevent extraction of other meta types. Returns None if "
        "bbox3d is not found.");
}
