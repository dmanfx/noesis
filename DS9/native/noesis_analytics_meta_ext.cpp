#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "metadata.hpp"
#include "nvds_analytics_meta.h"
#include "nvds_tracker_meta.h"

namespace py = pybind11;

namespace {

int analytics_object_meta_type() {
  return static_cast<int>(NVDS_USER_OBJ_META_NVDSANALYTICS);
}

constexpr unsigned int kMaxLifecycleStreams = 64;
constexpr unsigned int kMaxLifecycleTargetsPerStream = 256;

struct LifecycleFrame {
  unsigned int pad_index = 0;
  int frame_id = 0;
  uint64_t media_pts_ns = 0;
  bool shadow_available = false;
  bool terminated_available = false;
  struct ShadowTarget {
    uint64_t tracker_id;
    uint32_t frame_id;
    uint32_t age;
    int state;
  };
  std::vector<ShadowTarget> shadow;
  std::vector<uint64_t> terminated;
};

// NvDCF owns these buffers. Copy only bounded scalar lifecycle evidence while
// the batch is alive; never retain SDK pointers, trajectories or world points.
void copy_tracker_misc(
    const deepstream::BatchMetadata& batch_meta, int meta_type,
    std::array<LifecycleFrame, kMaxLifecycleStreams>& frames,
    unsigned int frame_count, bool shadow) {
  batch_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        const auto* batch = static_cast<const NvDsTargetMiscDataBatch*>(
            user_meta.userData());
        if (!batch || batch->numFilled > batch->numAllocated ||
            batch->numFilled > kMaxLifecycleStreams ||
            (batch->numFilled && !batch->list)) {
          throw std::runtime_error("invalid tracker lifecycle batch bounds");
        }
        for (unsigned int stream_index = 0; stream_index < batch->numFilled;
             ++stream_index) {
          const auto& stream = batch->list[stream_index];
          if (stream.numFilled > stream.numAllocated ||
              stream.numFilled > kMaxLifecycleTargetsPerStream ||
              (stream.numFilled && !stream.list)) {
            throw std::runtime_error("invalid tracker lifecycle stream bounds");
          }
          LifecycleFrame* current = nullptr;
          for (unsigned int index = 0; index < frame_count; ++index) {
            if (frames[index].pad_index == stream.streamID) {
              current = &frames[index];
              break;
            }
          }
          // Misc metadata may contain another source's archived history. It
          // supplies no current-cohort presence authority for an absent frame.
          if (!current) continue;
          bool& available = shadow ? current->shadow_available
                                   : current->terminated_available;
          if (available) {
            throw std::runtime_error("duplicate tracker lifecycle stream");
          }
          available = true;
          for (unsigned int target_index = 0; target_index < stream.numFilled;
               ++target_index) {
            const auto& target = stream.list[target_index];
            if (target.classId != 0) continue;  // Canonical people only.
            if (shadow) {
              // DS9.1 allocates exactly one shadow frame and does not
              // initialize target.numAllocated. Do not read that field here.
              if (target.numObj != 1 || !target.list) {
                throw std::runtime_error("invalid tracker shadow frame bounds");
              }
              const auto& sample = target.list[0];
              current->shadow.push_back({target.uniqueId, sample.frameNum,
                  sample.age, static_cast<int>(sample.trackerState)});
            } else {
              // Termination belongs to this publication batch, even when the
              // final stored trajectory frame predates the publication frame.
              current->terminated.push_back(target.uniqueId);
            }
          }
        }
      },
      meta_type);
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

py::dict extract_tracker_lifecycle(const deepstream::BatchMetadata& batch_meta) {
  std::array<LifecycleFrame, kMaxLifecycleStreams> frames;
  unsigned int count = 0;
  batch_meta.iterate([&](const deepstream::FrameMetadata& frame) {
    if (count >= kMaxLifecycleStreams || frame.frameNum() < 0) {
      throw std::runtime_error("invalid tracker lifecycle current frame bounds");
    }
    for (unsigned int index = 0; index < count; ++index) {
      if (frames[index].pad_index == frame.padIndex()) {
        throw std::runtime_error("duplicate tracker lifecycle current source");
      }
    }
    auto& current = frames[count++];
    current.pad_index = frame.padIndex();
    current.frame_id = frame.frameNum();
    current.media_pts_ns = frame.bufferPTS();
    current.shadow.reserve(32);
    current.terminated.reserve(32);
  });
  copy_tracker_misc(batch_meta, NVDS_TRACKER_SHADOW_LIST_META, frames, count, true);
  copy_tracker_misc(batch_meta, NVDS_TRACKER_TERMINATED_LIST_META, frames, count, false);
  py::dict records;
  for (unsigned int index = 0; index < count; ++index) {
    const auto& current = frames[index];
    py::dict record;
    record["source_id"] = current.pad_index;
    record["frame_id"] = current.frame_id;
    record["media_pts_ns"] = current.media_pts_ns;
    record["shadow_available"] = current.shadow_available;
    record["terminated_available"] = current.terminated_available;
    py::list shadow_rows;
    for (const auto& target : current.shadow) {
      py::dict row;
      row["tracker_id"] = target.tracker_id;
      row["class_id"] = 0;
      row["tracker_frame_id"] = target.frame_id;
      row["tracker_state"] = target.state;
      row["age"] = target.age;
      shadow_rows.append(std::move(row));
    }
    py::list terminated_rows;
    for (uint64_t tracker_id : current.terminated) {
      py::dict row;
      row["tracker_id"] = tracker_id;
      row["class_id"] = 0;
      terminated_rows.append(std::move(row));
    }
    record["shadow"] = std::move(shadow_rows);
    record["terminated"] = std::move(terminated_rows);
    records[py::int_(current.pad_index)] = std::move(record);
  }
  return records;
}

PYBIND11_MODULE(noesis_analytics_meta_ext, m) {
  m.doc() =
      "Noesis DS9 public-Service-Maker scalar analytics and tracker metadata.";
  m.def(
      "extract_analytics",
      &extract_analytics,
      py::arg("obj_meta"),
      "Extract all NvDsAnalyticsObjInfo records through ObjectMetadata iteration.");
  m.def(
      "analytics_object_meta_type",
      []() { return analytics_object_meta_type(); });
  m.def(
      "extract_tracker_lifecycle", &extract_tracker_lifecycle,
      py::arg("batch_meta"),
      "Copy exact-batch bounded person shadow/termination evidence, without positions.");
}
