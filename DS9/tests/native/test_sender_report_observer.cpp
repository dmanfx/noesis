// Real GStreamer scalar-only harness; no camera, network, SDK graph or GPU.
#include <pybind11/embed.h>
#include <iostream>
#include "../../native/noesis_analytics_meta_ext.cpp"

static void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}

static GstPad* add_sync_pad(GstElement* session) {
  auto* pad = gst_pad_new("sync_src", GST_PAD_SRC);
  require(pad && gst_element_add_pad(session, pad), "add synthetic SOMETIMES pad");
  require(gst_pad_set_active(pad, TRUE), "activate scalar-only source pad");
  gst_pad_push_event(pad, gst_event_new_stream_start("synthetic-no-network"));
  GstSegment segment;
  gst_segment_init(&segment, GST_FORMAT_TIME);
  gst_pad_push_event(pad, gst_event_new_segment(&segment));
  return pad;  // Borrowed from the owned session.
}

static void push_sr(GstPad* pad) {
  const guint8 packet[28] = {0x80,200,0,6, 0,0,0,77, 0x83,0xaa,0x7e,0x80,
                            0,0,0,0, 0,0,1,44, 0,0,0,0, 0,0,0,0};
  auto* buffer = gst_buffer_new_allocate(nullptr, sizeof(packet), nullptr);
  require(buffer != nullptr, "allocate bounded synthetic RTCP SR");
  require(gst_buffer_fill(buffer, 0, packet, sizeof(packet)) == sizeof(packet), "fill RTCP SR");
  gst_pad_push(pad, buffer);  // No peer is needed to exercise the actual native probe.
}

int main() {
  py::scoped_interpreter interpreter{};
  gst_init(nullptr, nullptr);
  try {
    for (bool late : {false, true}) {
      OwnedGstObject session(GST_OBJECT(gst_element_factory_make("rtpsession", nullptr)));
      require(bool(session), "installed rtpsession unavailable");
      auto state = std::make_shared<SenderReportState>(1);
      GstPad* pad = nullptr;
      if (!late) pad = add_sync_pad(GST_ELEMENT(session.get()));
      state->attach_session(GST_ELEMENT(session.get()));
      if (late) pad = add_sync_pad(GST_ELEMENT(session.get()));
      state->attach_session(GST_ELEMENT(session.get()));  // Both paths must be idempotent.
      push_sr(pad);
      auto first = state->drain();
      auto reports = first["reports"].cast<py::list>();
      require(py::len(reports) == 1, "existing/late pad must produce exactly one SR");
      auto row = reports[0].cast<py::dict>();
      require(row["source_id"].cast<int>() == 1 && row["ssrc"].cast<int>() == 77
              && row["rtp"].cast<int>() == 300 && row["session_generation"].cast<int>() == 1,
              "original scalar fields or session binding changed");
      require(first["drops"].cast<int>() == 0, "unexpected scalar observer drop");
      state->close();
      state->close();
      push_sr(pad);
      require(py::len(state->drain()["reports"].cast<py::list>()) == 0,
              "disconnected observer received data");
      std::weak_ptr<SenderReportState> weak = state;
      state.reset();
      require(weak.expired(), "callback state cycle survived close");
    }
    std::cout << "{\"existing_pad\":true,\"late_pad\":true,\"duplicate_safe\":true,"
                 "\"original_scalars\":true,\"disconnect\":true,\"state_released\":true,"
                 "\"network_sources\":0,\"gpu_inference\":false}" << std::endl;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
