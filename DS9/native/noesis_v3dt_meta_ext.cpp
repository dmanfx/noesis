#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "buffer.hpp"
#include "gstnvdsmeta.h"
#include "metadata.hpp"
#include "nvdsmeta.h"
#include "nvds_tracker_meta.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace py = pybind11;

// Static flags to limit logging noise (log once per process)
static bool s_logged_visibility_error = false;
static bool s_logged_image_foot_error = false;
static bool s_logged_bbox3d_error = false;

namespace {

bool nearly_equal(float lhs, float rhs, float epsilon = 1e-5F) {
  return std::fabs(lhs - rhs) <= epsilon;
}

bool only_lines(const NvDsDisplayMeta& meta) {
  return meta.num_rects == 0 && meta.num_labels == 0 &&
         meta.num_arrows == 0 && meta.num_circles == 0;
}

bool only_circles(const NvDsDisplayMeta& meta) {
  return meta.num_rects == 0 && meta.num_labels == 0 &&
         meta.num_lines == 0 && meta.num_arrows == 0;
}

bool is_tracker_bbox3d_debug_meta(const NvDsDisplayMeta& meta) {
  if (!only_lines(meta) || meta.num_lines == 0 || meta.num_lines > 12) {
    return false;
  }
  for (guint index = 0; index < meta.num_lines; ++index) {
    const NvOSD_LineParams& line = meta.line_params[index];
    if (line.line_width != 1 || !nearly_equal(line.line_color.red, 0.0F) ||
        !nearly_equal(line.line_color.green, 0.0F) ||
        !nearly_equal(line.line_color.blue, 1.0F) ||
        !nearly_equal(line.line_color.alpha, 1.0F)) {
      return false;
    }
  }
  return true;
}

bool is_tracker_foot_debug_meta(const NvDsDisplayMeta& meta) {
  if (!only_circles(meta) || meta.num_circles != 1) {
    return false;
  }
  const NvOSD_CircleParams& circle = meta.circle_params[0];
  return circle.radius == 5 && circle.has_bg_color == 1 &&
         nearly_equal(circle.circle_color.red, 1.0F) &&
         nearly_equal(circle.circle_color.green, 0.0F) &&
         nearly_equal(circle.circle_color.blue, 0.0F) &&
         nearly_equal(circle.circle_color.alpha, 1.0F) &&
         nearly_equal(circle.bg_color.red, 1.0F) &&
         nearly_equal(circle.bg_color.green, 0.0F) &&
         nearly_equal(circle.bg_color.blue, 0.0F) &&
         nearly_equal(circle.bg_color.alpha, 1.0F);
}

}  // namespace

py::dict scrub_tracker_projection_display_meta(deepstream::Buffer& buffer) {
  py::dict result;
  unsigned int frames_seen = 0;
  unsigned int bbox3d_removed = 0;
  unsigned int foot_removed = 0;

  // Retain an independent GstBuffer reference so `give()` never transfers the
  // reference owned by Service Maker's live probe object.
  deepstream::Buffer retained(buffer);
  OpaqueBuffer* opaque = retained.give();
  auto* gst_buffer = reinterpret_cast<GstBuffer*>(opaque);
  if (gst_buffer == nullptr) {
    result["frames_seen"] = frames_seen;
    result["bbox3d_removed"] = bbox3d_removed;
    result["foot_removed"] = foot_removed;
    return result;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
  if (batch_meta != nullptr) {
    for (NvDsMetaList* frame_node = batch_meta->frame_meta_list;
         frame_node != nullptr; frame_node = frame_node->next) {
      auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_node->data);
      if (frame_meta == nullptr) continue;
      ++frames_seen;
      NvDsMetaList* display_node = frame_meta->display_meta_list;
      while (display_node != nullptr) {
        NvDsMetaList* next = display_node->next;
        auto* display_meta = static_cast<NvDsDisplayMeta*>(display_node->data);
        if (display_meta != nullptr &&
            is_tracker_bbox3d_debug_meta(*display_meta)) {
          nvds_remove_display_meta_from_frame(frame_meta, display_meta);
          ++bbox3d_removed;
        } else if (display_meta != nullptr &&
                   is_tracker_foot_debug_meta(*display_meta)) {
          nvds_remove_display_meta_from_frame(frame_meta, display_meta);
          ++foot_removed;
        }
        display_node = next;
      }
    }
  }
  gst_buffer_unref(gst_buffer);

  result["frames_seen"] = frames_seen;
  result["bbox3d_removed"] = bbox3d_removed;
  result["foot_removed"] = foot_removed;
  return result;
}

py::dict extract_person_base(const deepstream::ObjectMetadata& obj_meta) {
  const NvOSD_RectParams& rect = obj_meta.rectParams();
  const NvOSD_MaskParams& mask = obj_meta.maskParams();
  float anchor_x = rect.left + (0.5F * rect.width);
  float anchor_y = rect.top + rect.height;
  const char* source = "bbox_bottom";

  const std::size_t expected =
      static_cast<std::size_t>(mask.width) *
      static_cast<std::size_t>(mask.height) * sizeof(float);
  if (mask.data != nullptr && mask.width > 0 && mask.height > 0 &&
      static_cast<std::size_t>(mask.size) >= expected && rect.width > 0.0F &&
      rect.height > 0.0F) {
    const float threshold = std::isfinite(mask.threshold) ? mask.threshold : 0.5F;
    int max_row = -1;
    for (unsigned int row = 0; row < mask.height; ++row) {
      const float* values = mask.data + (static_cast<std::size_t>(row) * mask.width);
      for (unsigned int column = 0; column < mask.width; ++column) {
        if (std::isfinite(values[column]) && values[column] > threshold) {
          max_row = static_cast<int>(row);
          break;
        }
      }
    }
    if (max_row >= 0) {
      const int band_height = std::max(
          1, static_cast<int>(std::lround(static_cast<double>(mask.height) * 0.12)));
      const int first_row = std::max(0, max_row - band_height + 1);
      std::vector<unsigned int> support_columns;
      for (int row = first_row; row <= max_row; ++row) {
        const float* values =
            mask.data + (static_cast<std::size_t>(row) * mask.width);
        for (unsigned int column = 0; column < mask.width; ++column) {
          if (std::isfinite(values[column]) && values[column] > threshold) {
            support_columns.push_back(column);
          }
        }
      }
      if (!support_columns.empty()) {
        const std::size_t middle = support_columns.size() / 2;
        std::nth_element(
            support_columns.begin(), support_columns.begin() + middle,
            support_columns.end());
        const float median_column =
            static_cast<float>(support_columns[middle]) + 0.5F;
        anchor_x = rect.left +
                   (median_column / static_cast<float>(mask.width)) * rect.width;
        anchor_y = rect.top +
                   (static_cast<float>(max_row + 1) /
                    static_cast<float>(mask.height)) *
                       rect.height;
        source = "instance_mask_base";
      }
    }
  }

  py::dict result;
  result["x"] = anchor_x;
  result["y"] = anchor_y;
  result["source"] = source;
  return result;
}

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
  m.def(
      "scrub_tracker_projection_display_meta",
      &scrub_tracker_projection_display_meta,
      py::arg("buffer"),
      "Remove only nvtracker's built-in blue projected-cuboid and red "
      "image-foot debug DisplayMeta while preserving V3DT user metadata.");
  m.def(
      "extract_person_base",
      &extract_person_base,
      py::arg("obj_meta"),
      "Return the instance-mask base gravity point in source-frame pixels, "
      "falling back to the tracked bbox bottom-center.");
}
