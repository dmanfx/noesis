#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gst/gst.h>
#include "gst-nvevent.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <deque>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include "element.hpp"
#include "buffer_probe.hpp"
#include "common_factory.hpp"

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

struct OriginalSourceClock {
  unsigned int source_id, pad_index, batch_id;
  int original_frame;
  uint64_t pts_ns, ntp_ns;
  uint64_t buffer_scope;
};
std::atomic<uint64_t> source_buffer_scope{0};
std::atomic<uint64_t> extraction_scope{0};

int source_clock_meta_type() {
  static int type = static_cast<int>(nvds_get_user_meta_type(
      const_cast<gchar*>("NOESIS.XC3D.ORIGINAL_SOURCE_CLOCK.v1")));
  return type;
}
int source_buffer_meta_type() {
  static int type = static_cast<int>(nvds_get_user_meta_type(
      const_cast<gchar*>("NOESIS.XC3D.ORIGINAL_BUFFER_SCOPE.v1")));
  return type;
}
void* source_clock_copy(void* opaque, void*) {
  deepstream::UserMetadata meta(opaque);
  auto* value = static_cast<OriginalSourceClock*>(meta.userData());
  return value ? new OriginalSourceClock(*value) : nullptr;
}
void source_clock_release(void* opaque, void*) {
  deepstream::UserMetadata meta(opaque);
  delete static_cast<OriginalSourceClock*>(meta.userData());
}

void stamp_source_acquisition(deepstream::BatchMetadata& batch) {
  unsigned int count = 0;
  std::array<bool, 3> slots{};
  const uint64_t scope = ++source_buffer_scope;
  batch.iterate([&](const deepstream::UserMetadata&) {
    throw std::runtime_error("duplicate original buffer scope");
  }, source_buffer_meta_type());
  deepstream::UserMetadata buffer_meta(nullptr);
  if (!batch.acquire(buffer_meta)) throw std::runtime_error("source buffer SDK metadata pool exhausted");
  auto buffer_value = std::make_unique<OriginalSourceClock>(OriginalSourceClock{0, 0, 0, 0, 0, 0, scope});
  buffer_meta.setMetaType(source_buffer_meta_type());
  buffer_meta.setUserData(buffer_value.get(), source_clock_copy, source_clock_release);
  buffer_value.release();
  batch.append(buffer_meta);
  batch.iterate([&](const deepstream::FrameMetadata& frame) {
    if (++count > 3 || frame.padIndex() >= 3 || frame.batchId() >= 3 || frame.frameNum() < 0
        || slots[frame.batchId()])
      throw std::runtime_error("XC3D source frame bounds");
    slots[frame.batchId()] = true;
    frame.iterate([&](const deepstream::UserMetadata&) {
      throw std::runtime_error("duplicate original source clock metadata");
    }, source_clock_meta_type());
    deepstream::UserMetadata meta(nullptr);
    if (!batch.acquire(meta)) throw std::runtime_error("source clock SDK metadata pool exhausted");
    auto value = std::make_unique<OriginalSourceClock>(OriginalSourceClock{
        frame.sourceId(), frame.padIndex(), frame.batchId(), frame.frameNum(),
        frame.bufferPTS(), frame.ntpTimestamp(), scope});
    meta.setMetaType(source_clock_meta_type());
    meta.setUserData(value.get(), source_clock_copy, source_clock_release);
    value.release();
    auto owner = frame;
    owner.append(meta);
  });
}

py::dict extract_source_acquisition(const deepstream::BatchMetadata& batch) {
  py::dict result;
  py::list rows;
  const uint64_t scope = ++extraction_scope;
  unsigned int frames = 0;
  std::array<bool, 3> slots{};
  uint64_t original_scope = 0;
  unsigned int buffer_receipts = 0;
  batch.iterate([&](const deepstream::UserMetadata& meta) {
    if (++buffer_receipts > 1) throw std::runtime_error("duplicate original current buffer scope");
    auto* stored = static_cast<const OriginalSourceClock*>(meta.userData());
    if (!stored || !stored->buffer_scope) throw std::runtime_error("null original current buffer scope");
    original_scope = stored->buffer_scope;
  }, source_buffer_meta_type());
  batch.iterate([&](const deepstream::FrameMetadata& frame) {
    if (++frames > 3 || frame.padIndex() >= 3 || frame.batchId() >= 3 || slots[frame.batchId()])
      throw std::runtime_error("XC3D current frame bounds/duplicate slot");
    slots[frame.batchId()] = true;
    py::dict row;
    row["scope"] = scope;
    row["slot"] = frame.batchId();
    row["source_id"] = frame.padIndex();
    row["sdk_source_id"] = frame.sourceId();
    row["post_tracker_frame"] = frame.frameNum();
    row["post_tracker_pts_ns"] = frame.bufferPTS();
    row["post_tracker_ntp_ns"] = frame.ntpTimestamp();
    row["receipt_valid"] = false;
    unsigned int matches = 0;
    try { frame.iterate([&](const deepstream::UserMetadata& meta) {
      if (++matches > 1) throw std::runtime_error("duplicate original source clock receipt");
      auto* stored = static_cast<const OriginalSourceClock*>(meta.userData());
      if (!stored) throw std::runtime_error("empty original source clock receipt");
      row["original_source_id"] = stored->pad_index;
      row["original_sdk_source_id"] = stored->source_id;
      row["original_batch_id"] = stored->batch_id;
      row["original_buffer_scope"] = stored->buffer_scope;
      row["original_frame"] = stored->original_frame;
      row["original_pts_ns"] = stored->pts_ns;
      row["original_ntp_ns"] = stored->ntp_ns;
      if (!original_scope || original_scope != stored->buffer_scope)
        throw std::runtime_error("original current-buffer scope mismatch");
      row["receipt_valid"] = stored->buffer_scope != 0 && stored->pad_index == frame.padIndex()
          && stored->source_id == frame.sourceId() && stored->batch_id == frame.batchId()
          && stored->pts_ns == frame.bufferPTS() && stored->ntp_ns == frame.ntpTimestamp();
    }, source_clock_meta_type()); }
    catch (const std::exception& error) {
      row["receipt_valid"] = false;
      row["private_error"] = error.what();
    }
    if (!matches) row["private_error"] = "original source receipt absent";
    rows.append(std::move(row));
  });
  result["scope"] = scope;
  result["original_buffer_scope"] = original_scope;
  result["rows"] = std::move(rows);
  return result;
}

py::tuple extract_frame_identity(const deepstream::FrameMetadata& frame) {
  return py::make_tuple(frame.batchId(), frame.padIndex(), frame.sourceId(),
                       frame.frameNum(), frame.bufferPTS(), frame.ntpTimestamp());
}

constexpr char kSenderReportAnchorFactory[] = "noesis_xc3d_sr_anchor_v1";

// Plain native SDK product: no Python callback and no buffer/metadata access.
class SenderReportAnchorHandler final : public deepstream::BufferProbe::IBufferObserver {
 public:
  deepstream::probeReturn handleBuffer(deepstream::BufferProbe&, const deepstream::Buffer&) override {
    return deepstream::probeReturn::Probe_Ok;
  }
};

class SenderReportAnchorFactory final : public deepstream::CustomFactory {
 public:
  SenderReportAnchorFactory()
      : deepstream::CustomFactory(kSenderReportAnchorFactory, deepstream::BufferProbe::type()) {}
  deepstream::CustomObject* createObject(const std::string& name) override {
    return new deepstream::BufferProbe(name, kSenderReportAnchorFactory, new SenderReportAnchorHandler());
  }
};

std::mutex sender_report_factory_mutex;
// SDK registration is process-lived. Never free/unregister this pointer, even
// on a failed registration attempt; retries cannot allocate another factory.
SenderReportAnchorFactory* sender_report_factory = nullptr;
bool sender_report_factory_registered = false;

bool register_sender_report_anchor_factory() {
  std::lock_guard<std::mutex> lock(sender_report_factory_mutex);
  auto& common = deepstream::CommonFactory::getInstance();
  auto* existing = common.getCustomFactory(kSenderReportAnchorFactory);
  if (sender_report_factory) {
    if (!sender_report_factory_registered || existing != sender_report_factory)
      throw std::runtime_error("SR anchor factory registration identity changed or failed");
    return true;
  }
  if (existing) throw std::runtime_error("SR anchor factory key belongs to a foreign factory");
  sender_report_factory = new SenderReportAnchorFactory();
  if (!common.addCustomFactory(sender_report_factory, kSenderReportAnchorFactory))
    throw std::runtime_error("SR anchor factory registration failed");
  sender_report_factory_registered = true;
  if (common.getCustomFactory(kSenderReportAnchorFactory) != sender_report_factory)
    throw std::runtime_error("SR anchor factory registration identity mismatch");
  return true;
}

// Root-only source diagnostic. Node.find exposes this plain native product as
// the SDK's registered CustomObject, never a cast/unwrap of Node or Python Probe.
// The STATIC mux.src anchor identifies its exact GstPipeline; only that owned
// pipeline's direct nvurisrcbin child may supply the SR observation boundary.
// Only fixed SR scalar fields survive the callback. No packet/frame is retained.
struct GstObjectUnref {
  void operator()(GstObject* object) const noexcept { gst_object_unref(object); }
};
using OwnedGstObject = std::unique_ptr<GstObject, GstObjectUnref>;

class SenderReportState : public std::enable_shared_from_this<SenderReportState> {
  struct Connection { GstObject* object; gulong id; bool pad; };
  struct SR { uint32_t ssrc, rtp; uint64_t ntp, arrival, session; unsigned int source; };
  std::vector<Connection> connections_;
  std::array<std::deque<SR>, 3> rows_;
  std::mutex mutex_;
  std::mutex connections_mutex_;
  std::atomic<bool> closing_{false};
  std::atomic<uint64_t> drops_{0};
  std::unordered_map<GstPad*, uint64_t> sessions_;
  unsigned int source_;
  using Context = std::shared_ptr<SenderReportState>;
  static void release_signal(gpointer opaque, GClosure*) { delete static_cast<Context*>(opaque); }
  static void release_probe(gpointer opaque) { delete static_cast<Context*>(opaque); }
  static uint32_t be32(const guint8* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
  }
  static GstPadProbeReturn probe(GstPad* pad, GstPadProbeInfo* info, gpointer opaque) {
    auto self = *static_cast<Context*>(opaque);
    if (self->closing_) return GST_PAD_PROBE_OK;
    auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    GstMapInfo map = GST_MAP_INFO_INIT;
    if (!buffer || !gst_buffer_map(buffer, &map, GST_MAP_READ)) return GST_PAD_PROBE_OK;
    // RTCP compound packet header: version 2, type 200 SR, length in 32-bit words.
    // This bound parser corresponds to installed gstrtcpbuffer.h sender fields.
    try { if (map.size <= 65536) for (size_t offset = 0; offset + 4 <= map.size;) {
      const guint8* p = map.data + offset;
      size_t length = (size_t(p[2]) * 256 + p[3] + 1) * 4;
      if ((p[0] >> 6) != 2 || length < 4 || length > map.size - offset) break;
      if (p[1] == 200 && length >= 28 + size_t(p[0] & 31) * 24) {
        uint64_t ntp = (uint64_t(be32(p + 8)) << 32) | be32(p + 12);
        uint64_t arrival = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::unique_lock<std::mutex> lock(self->mutex_, std::try_to_lock);
        if (!lock.owns_lock()) ++self->drops_;
        else {
          const auto session = self->sessions_.find(pad);
          if (session == self->sessions_.end()) ++self->drops_;
          else {
            auto& rows = self->rows_[self->source_];
            if (rows.size() == 32) { rows.pop_front(); ++self->drops_; }
            rows.push_back(SR{be32(p + 4), be32(p + 16), ntp, arrival, session->second, self->source_});
          }
        }
      }
      offset += length;
    } } catch (...) { ++self->drops_; }  // Never unwind through a C callback.
    gst_buffer_unmap(buffer, &map);
    return GST_PAD_PROBE_OK;
  }
  void attach_pad(GstElement* element, GstPad* pad) {
    if (!pad || g_strcmp0(GST_PAD_NAME(pad), "sync_src") != 0) return;
    OwnedGstObject parent(GST_OBJECT(gst_pad_get_parent_element(pad)));
    if (parent.get() != GST_OBJECT(element)) return;
    std::lock_guard<std::mutex> connections_lock(connections_mutex_);
    if (closing_) return;
    for (const auto& connection : connections_)
      if (connection.pad && connection.object == GST_OBJECT(pad)) return;
    if (connections_.size() >= 32) { ++drops_; return; }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sessions_[pad] = sessions_.size() + 1;
    }
    auto* context = new Context(shared_from_this());
    auto id = gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, probe, context, release_probe);
    connections_.push_back({GST_OBJECT(gst_object_ref(pad)), id, true});
  }
  static void pad_added(GstElement* element, GstPad* pad, gpointer opaque) {
    auto self = *static_cast<Context*>(opaque);
    try { self->attach_pad(element, pad); } catch (...) { ++self->drops_; }
  }
  static void child_added(GstBin*, GstBin*, GstElement* child, gpointer opaque) {
    auto self = *static_cast<Context*>(opaque);
    try { self->attach_session(child); } catch (...) { ++self->drops_; }
  }
 public:
  explicit SenderReportState(unsigned int source) : source_(source) {
    if (source >= 3) throw std::runtime_error("SR observer source id must be in 0..2");
    connections_.reserve(32);
  }
  void attach_session(GstElement* element) {
    auto* factory = gst_element_get_factory(element);
    if (!factory || std::string(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory))) != "rtpsession") return;
    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      if (closing_) return;
      bool watched = false;
      for (const auto& connection : connections_)
        watched |= !connection.pad && connection.object == GST_OBJECT(element);
      if (!watched) {
        if (connections_.size() >= 32) { ++drops_; return; }
        auto* context = new Context(shared_from_this());
        auto id = g_signal_connect_data(element, "pad-added", G_CALLBACK(pad_added), context,
                                        release_signal, GConnectFlags(0));
        connections_.push_back({GST_OBJECT(gst_object_ref(element)), id, false});
      }
    }
    // Watch BEFORE querying: a SOMETIMES pad can appear on either side of this
    // lookup. Both paths converge on the same duplicate-safe bounded attachment.
    OwnedGstObject pad(GST_OBJECT(gst_element_get_static_pad(element, "sync_src")));
    if (pad) attach_pad(element, GST_PAD(pad.get()));
  }
  void watch_source(GstElement* source) {
    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      if (closing_) return;
      auto* context = new Context(shared_from_this());
      auto id = g_signal_connect_data(source, "deep-element-added", G_CALLBACK(child_added), context,
                                      release_signal, GConnectFlags(0));
      connections_.push_back({GST_OBJECT(gst_object_ref(source)), id, false});
    }
    auto* iterator = gst_bin_iterate_recurse(GST_BIN(source));
    GValue value = G_VALUE_INIT;
    unsigned int inspected = 0;
    while (inspected++ < 128) {
      const auto status = gst_iterator_next(iterator, &value);
      if (status == GST_ITERATOR_RESYNC) { gst_iterator_resync(iterator); continue; }
      if (status != GST_ITERATOR_OK) break;
      attach_session(GST_ELEMENT(g_value_get_object(&value)));
      g_value_reset(&value);
    }
    if (G_VALUE_TYPE(&value)) g_value_unset(&value);
    gst_iterator_free(iterator);
  }
  void close() {
    if (closing_.exchange(true)) return;
    std::vector<Connection> connections;
    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      connections.swap(connections_);
    }
    // Callbacks own shared state through their GLib closures/probe data. A
    // concurrently executing callback cannot dereference a destroyed observer.
    for (auto it = connections.rbegin(); it != connections.rend(); ++it) {
      if (it->pad) gst_pad_remove_probe(GST_PAD(it->object), it->id);
      else g_signal_handler_disconnect(it->object, it->id);
      gst_object_unref(it->object);
    }
  }
  py::dict drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    py::list reports;
    for (auto& rows : rows_) while (!rows.empty()) {
      auto row = rows.front(); rows.pop_front();
      py::dict result;
      result["source_id"] = row.source; result["ssrc"] = row.ssrc;
      result["rtp"] = row.rtp; result["ntp_fixed64"] = row.ntp;
      result["host_arrival_ns"] = row.arrival;
      result["session_generation"] = row.session;
      reports.append(result);
    }
    py::dict result; result["reports"] = reports; result["drops"] = drops_.exchange(0);
    return result;
  }
};

class SenderReportObserver {
  std::shared_ptr<SenderReportState> state_;
 public:
  SenderReportObserver(deepstream::CustomObject& anchor, unsigned int source_id) {
    if (source_id >= 3) throw std::runtime_error("SR observer source id must be in 0..2");
    auto* probe = dynamic_cast<deepstream::BufferProbe*>(&anchor);
    if (!probe) throw std::runtime_error("SR observer anchor must be a native BufferProbe");
    auto* mux = probe->getTarget();
    if (!mux || anchor.getName() != "xc3d_sr_pipeline_anchor" || mux->getName() != "mux" ||
        mux->getProbe(anchor.getName()) != probe)
      throw std::runtime_error("SR observer pipeline anchor/mux name or attached ownership mismatch");
    auto* mux_object = mux->getGObject();
    if (!GST_IS_ELEMENT(mux_object)) throw std::runtime_error("SR observer mux target is not an element");
    auto* mux_factory = gst_element_get_factory(GST_ELEMENT(mux_object));
    if (!mux_factory || std::string(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(mux_factory))) != "nvstreammux")
      throw std::runtime_error("SR observer pipeline anchor must bind the configured nvstreammux");
    OwnedGstObject pipeline(gst_object_get_parent(mux_object));
    if (!pipeline || !GST_IS_PIPELINE(pipeline.get()))
      throw std::runtime_error("SR observer mux immediate parent must be its GstPipeline");
    const std::string source_name = "source" + std::to_string(source_id);
    // Public scoped lookup recurses; the immediate-parent check rejects any
    // nested same-name child rather than mistaking it for the direct source.
    auto* source_element = gst_bin_get_by_name(GST_BIN(pipeline.get()), source_name.c_str());
    OwnedGstObject object(source_element ? GST_OBJECT(source_element) : nullptr);
    if (!object || !GST_IS_BIN(object.get()))
      throw std::runtime_error("SR observer exact pipeline source is absent or not a bin");
    OwnedGstObject source_parent(gst_object_get_parent(object.get()));
    std::unique_ptr<gchar, decltype(&g_free)> name(gst_object_get_name(object.get()), &g_free);
    if (source_parent.get() != pipeline.get() || !name || source_name != name.get())
      throw std::runtime_error("SR observer source name or direct pipeline ownership mismatch");
    auto* factory = gst_element_get_factory(GST_ELEMENT(object.get()));
    if (!factory || std::string(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory))) != "nvurisrcbin")
      throw std::runtime_error("SR observer must bind the configured native nvurisrcbin");
    state_ = std::make_shared<SenderReportState>(source_id);
    try { state_->watch_source(GST_ELEMENT(object.get())); }
    catch (...) { state_->close(); throw; }
  }
  ~SenderReportObserver() { state_->close(); }
  py::dict drain() { return state_->drain(); }
};

// Serialized native source termination, not inferred from silence or Python
// metadata. Three fixed bits survive; no frames, SDK metadata, callback into
// Python, or growing event queue. A stop is terminal for this static graph/run.
class SourceStopState : public std::enable_shared_from_this<SourceStopState> {
  using Context = std::shared_ptr<SourceStopState>;
  std::atomic<unsigned int> mask_{0};
  std::atomic<bool> closed_{false};
  std::mutex connection_mutex_;
  GstPad* pad_ = nullptr;
  gulong probe_id_ = 0;
  static void release(gpointer opaque) { delete static_cast<Context*>(opaque); }
  static GstPadProbeReturn event(GstPad*, GstPadProbeInfo* info, gpointer opaque) {
    auto self = *static_cast<Context*>(opaque);
    if (self->closed_) return GST_PAD_PROBE_OK;
    auto* value = GST_PAD_PROBE_INFO_EVENT(info);
    if (!value) return GST_PAD_PROBE_OK;
    guint source = G_MAXUINT;
    if (GST_EVENT_TYPE(value) == static_cast<GstEventType>(GST_NVEVENT_STREAM_EOS))
      gst_nvevent_parse_stream_eos(value, &source);
    else if (GST_EVENT_TYPE(value) == static_cast<GstEventType>(GST_NVEVENT_PAD_DELETED))
      gst_nvevent_parse_pad_deleted(value, &source);
    else return GST_PAD_PROBE_OK;
    // Bit 3 is an explicit fail-closed invalid source diagnostic.
    self->mask_.fetch_or(source < 3 ? 1u << source : 8u);
    return GST_PAD_PROBE_OK;
  }
 public:
  void attach(GstPad* pad) {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    if (!pad || pad_ || closed_) throw std::runtime_error("source stop observer attachment invalid");
    auto* context = new Context(shared_from_this());
    probe_id_ = gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, event, context, release);
    if (!probe_id_) { delete context; throw std::runtime_error("source stop event probe unavailable"); }
    pad_ = GST_PAD(gst_object_ref(pad));
  }
  unsigned int snapshot() const {
    const auto mask = mask_.load();
    if (mask & 8u) throw std::runtime_error("source stop event outside configured three-source graph");
    return mask;
  }
  void close() {
    GstPad* pad;
    gulong id;
    {
      std::lock_guard<std::mutex> lock(connection_mutex_);
      if (closed_.exchange(true)) return;
      pad = pad_; id = probe_id_; pad_ = nullptr; probe_id_ = 0;
    }
    // Never hold the connection lock across native callback disconnection.
    if (pad) { gst_pad_remove_probe(pad, id); gst_object_unref(pad); }
  }
};

class SourceStopObserver {
  std::shared_ptr<SourceStopState> state_;
 public:
  SourceStopObserver(deepstream::CustomObject& anchor, const std::string& target_name) {
    auto* probe = dynamic_cast<deepstream::BufferProbe*>(&anchor);
    auto* target = probe ? probe->getTarget() : nullptr;
    if (!target || anchor.getName() != "xc3d_source_stop_anchor" ||
        target_name.empty() || target->getName() != target_name ||
        target->getProbe(anchor.getName()) != probe)
      throw std::runtime_error("source stop anchor target/attached ownership mismatch");
    auto* object = target->getGObject();
    if (!GST_IS_ELEMENT(object)) throw std::runtime_error("source stop target must be an element");
    OwnedGstObject parent(gst_object_get_parent(object));
    if (!parent || !GST_IS_PIPELINE(parent.get()))
      throw std::runtime_error("source stop target must belong directly to its GstPipeline");
    OwnedGstObject pad(GST_OBJECT(gst_element_get_static_pad(GST_ELEMENT(object), "src")));
    if (!pad) throw std::runtime_error("source stop target requires its actual src pad");
    state_ = std::make_shared<SourceStopState>();
    try { state_->attach(GST_PAD(pad.get())); }
    catch (...) { state_->close(); throw; }
  }
  ~SourceStopObserver() { state_->close(); }
  unsigned int snapshot() const { return state_->snapshot(); }
  void close() { state_->close(); }
};

struct LifecycleFrame {
  unsigned int pad_index = 0;
  unsigned int slot = 0, sdk_source_id = 0;
  int frame_id = 0;
  uint64_t media_pts_ns = 0, ntp_ns = 0;
  bool shadow_available = false;
  bool terminated_available = false;
  bool shadow_stream_seen = false, terminated_stream_seen = false;
  bool termination_barrier = false;
  unsigned int source_frame_count = 0;
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
  unsigned int metadata_count = 0;
  batch_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (++metadata_count > 1) throw std::runtime_error("duplicate tracker lifecycle batch metadata");
        const auto* batch = static_cast<const NvDsTargetMiscDataBatch*>(
            user_meta.userData());
        if (!batch || batch->numFilled > batch->numAllocated ||
            batch->numFilled > kMaxLifecycleStreams ||
            (batch->numFilled && !batch->list)) {
          throw std::runtime_error("invalid tracker lifecycle batch bounds");
        }
        // nvtracker queueFrames() invokes RetrieveMiscData once per ordered
        // same-source frame group, accumulating into this SAME misc batch.
        // streamID is a source, not a unique current-frame identity.
        std::unordered_map<unsigned int, unsigned int> source_groups;
        std::unordered_map<unsigned int, uint64_t> source_surfaces;
        std::unordered_map<unsigned int, bool> source_shadow_order;
        for (unsigned int stream_index = 0; stream_index < batch->numFilled;
             ++stream_index) {
          const auto& stream = batch->list[stream_index];
          if (stream.numFilled > stream.numAllocated ||
              stream.numFilled > kMaxLifecycleTargetsPerStream ||
              (stream.numFilled && !stream.list)) {
            throw std::runtime_error("invalid tracker lifecycle stream bounds");
          }
          LifecycleFrame* barrier = nullptr;
          unsigned int matches = 0;
          for (unsigned int index = 0; index < frame_count; ++index) {
            if (frames[index].pad_index == stream.streamID) {
              ++matches;
              auto& frame = frames[index];
              if (!barrier || frame.frame_id > barrier->frame_id) barrier = &frame;
            }
          }
          // Misc metadata may contain another source's archived history. It
          // supplies no current-cohort presence authority for an absent frame.
          if (!barrier) {
            if (!shadow && stream.numFilled) throw std::runtime_error("termination has no current-buffer source barrier");
            continue;
          }
          if (++source_groups[stream.streamID] > matches)
            throw std::runtime_error("tracker lifecycle group count exceeds current source frames");
          source_shadow_order.emplace(stream.streamID, true);
          const auto surface = source_surfaces.emplace(stream.streamID, stream.surfaceStreamID);
          if (!surface.second && surface.first->second != stream.surfaceStreamID)
            throw std::runtime_error("ambiguous tracker lifecycle surface stream");
          std::array<bool, kMaxLifecycleStreams> group_frames{};
          LifecycleFrame* shadow_group_frame = nullptr;
          // A source misc snapshot is not an empty receipt for every repeated
          // frame. A one-frame source is unambiguous; repeated shadow rows must
          // match their current SDK counter exactly and uniquely.
          if (shadow && matches == 1) barrier->shadow_available = true;
          if (!shadow) {
            barrier->terminated_available = true;
            barrier->termination_barrier = true;
          }
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
              LifecycleFrame* current = nullptr;
              for (unsigned int index = 0; index < frame_count; ++index)
                if (frames[index].pad_index == stream.streamID && frames[index].frame_id == int(sample.frameNum)) {
                  if (current) throw std::runtime_error("ambiguous tracker shadow frame counter");
                  current = &frames[index];
                }
              if (!current) throw std::runtime_error("tracker shadow sample has no exact current frame");
              // One shadow stream block is one processed frame, never a
              // multi-frame history. This is also required before using the
              // complete group's cardinality to prove unstamped empties.
              if (shadow_group_frame && shadow_group_frame != current)
                throw std::runtime_error("tracker shadow group spans multiple current frames");
              if (!shadow_group_frame) {
                unsigned int ordinal = 1;
                for (unsigned int index = 0; index < frame_count; ++index)
                  ordinal += frames[index].pad_index == stream.streamID &&
                      frames[index].frame_id < current->frame_id;
                if (ordinal != source_groups[stream.streamID])
                  source_shadow_order[stream.streamID] = false;
                shadow_group_frame = current;
              }
              const auto current_index = static_cast<unsigned int>(current - frames.data());
              if (!group_frames[current_index]) {
                if (current->shadow_stream_seen)
                  throw std::runtime_error("duplicate tracker shadow exact-frame group");
                current->shadow_stream_seen = true;
                group_frames[current_index] = true;
              }
              for (const auto& prior : current->shadow)
                if (prior.tracker_id == target.uniqueId)
                  throw std::runtime_error("duplicate tracker shadow current target");
              current->shadow_available = true;
              if (current->shadow.size() >= kMaxLifecycleTargetsPerStream)
                throw std::runtime_error("tracker shadow per-frame target capacity");
              current->shadow.push_back({target.uniqueId, sample.frameNum,
                  sample.age, static_cast<int>(sample.trackerState)});
            } else {
              // Termination belongs to this publication batch, even when the
              // final stored trajectory frame predates the publication frame.
              if (barrier->terminated.size() >= kMaxLifecycleTargetsPerStream)
                throw std::runtime_error("tracker termination barrier target capacity");
              for (auto prior : barrier->terminated)
                if (prior == target.uniqueId)
                  throw std::runtime_error("duplicate tracker termination target");
              barrier->terminated.push_back(target.uniqueId);
            }
          }
        }
        if (shadow) {
          // DS9.1 resets shadow numFilled before this misc buffer is reused;
          // queueFrames/RetrieveMiscData accumulate its per-frame blocks and
          // updateUserMeta attaches them only after ALL original-buffer groups
          // succeed. With one surface, complete source cardinality, ascending
          // source-group anchors and unique exact stamped groups above, every
          // remaining frame has an empty person snapshot. No counter/time/ID
          // is synthesized for that block.
          // Missing groups remain unavailable, not an inferred empty receipt.
          for (unsigned int index = 0; index < frame_count; ++index) {
            auto& frame = frames[index];
            const auto groups = source_groups.find(frame.pad_index);
            unsigned int matches = 0;
            for (unsigned int other = 0; other < frame_count; ++other)
              matches += frames[other].pad_index == frame.pad_index;
            if (groups != source_groups.end() && groups->second == matches &&
                source_shadow_order.at(frame.pad_index))
              frame.shadow_available = true;
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
    if (count >= kMaxLifecycleStreams || frame.frameNum() < 0 || frame.batchId() >= kMaxLifecycleStreams) {
      throw std::runtime_error("invalid tracker lifecycle current frame bounds");
    }
    for (unsigned int index = 0; index < count; ++index) {
      if (frames[index].slot == frame.batchId()) {
        throw std::runtime_error("duplicate tracker lifecycle current slot");
      }
      if (frames[index].pad_index == frame.padIndex() &&
          ((frames[index].frame_id == frame.frameNum()) ||
           (frames[index].media_pts_ns == frame.bufferPTS()) ||
           ((frames[index].frame_id < frame.frameNum()) != (frames[index].media_pts_ns < frame.bufferPTS())))) {
        throw std::runtime_error("ambiguous tracker lifecycle intra-buffer epoch/order");
      }
    }
    auto& current = frames[count++];
    current.pad_index = frame.padIndex();
    current.slot = frame.batchId();
    current.sdk_source_id = frame.sourceId();
    current.frame_id = frame.frameNum();
    current.media_pts_ns = frame.bufferPTS();
    current.ntp_ns = frame.ntpTimestamp();
    current.shadow.reserve(32);
    current.terminated.reserve(32);
  });
  for (unsigned int index = 0; index < count; ++index)
    for (unsigned int other = 0; other < count; ++other)
      frames[index].source_frame_count += frames[index].pad_index == frames[other].pad_index;
  copy_tracker_misc(batch_meta, NVDS_TRACKER_SHADOW_LIST_META, frames, count, true);
  copy_tracker_misc(batch_meta, NVDS_TRACKER_TERMINATED_LIST_META, frames, count, false);
  py::dict records;
  py::list rows;
  const uint64_t scope = ++extraction_scope;
  for (unsigned int index = 0; index < count; ++index) {
    const auto& current = frames[index];
    py::dict record;
    record["scope"] = scope;
    record["slot"] = current.slot;
    record["sdk_source_id"] = current.sdk_source_id;
    record["post_tracker_frame"] = current.frame_id;
    record["post_tracker_pts_ns"] = current.media_pts_ns;
    record["post_tracker_ntp_ns"] = current.ntp_ns;
    record["source_id"] = current.pad_index;
    record["frame_id"] = current.frame_id;
    record["media_pts_ns"] = current.media_pts_ns;
    record["shadow_available"] = current.shadow_available;
    record["terminated_available"] = current.terminated_available;
    record["termination_boundary"] = current.termination_barrier ? "complete_buffer_source_barrier" : "not_at_buffer_barrier";
    if (current.source_frame_count > 1 && !current.shadow_available)
      record["lifecycle_unavailable"] = "repeated_source_shadow_snapshot_not_exact";
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
    rows.append(std::move(record));
  }
  records["scope"] = scope;
  records["rows"] = std::move(rows);
  return records;
}

PYBIND11_MODULE(noesis_analytics_meta_ext, m) {
  m.doc() =
      "Noesis DS9 public-Service-Maker scalar analytics and tracker metadata.";
  m.def("stamp_source_acquisition", &stamp_source_acquisition, py::arg("batch_meta"));
  m.def("extract_source_acquisition", &extract_source_acquisition, py::arg("batch_meta"));
  m.def("extract_frame_identity", &extract_frame_identity, py::arg("frame_meta"));
  m.def("register_sender_report_anchor_factory", &register_sender_report_anchor_factory);
  py::class_<SenderReportObserver>(m, "SenderReportObserver")
      .def(py::init<deepstream::CustomObject&, unsigned int>(), py::arg("anchor"), py::arg("source_id"))
      .def("drain", &SenderReportObserver::drain);
  py::class_<SourceStopObserver>(m, "SourceStopObserver")
      .def(py::init<deepstream::CustomObject&, const std::string&>(), py::arg("anchor"), py::arg("target_name"))
      .def("snapshot", &SourceStopObserver::snapshot)
      .def("close", &SourceStopObserver::close);
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
