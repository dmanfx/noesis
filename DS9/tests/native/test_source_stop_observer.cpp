// Actual installed serialized events, no cameras/network/media/GPU.
#include <pybind11/embed.h>
#include <iostream>
#include "../../native/noesis_analytics_meta_ext.cpp"

static void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}
int main() {
  py::scoped_interpreter interpreter{};
  gst_init(nullptr, nullptr);
  try {
    OwnedGstObject element(GST_OBJECT(gst_element_factory_make("identity", nullptr)));
    require(bool(element), "installed identity element unavailable");
    OwnedGstObject pad(GST_OBJECT(gst_element_get_static_pad(GST_ELEMENT(element.get()), "src")));
    require(bool(pad) && gst_pad_set_active(GST_PAD(pad.get()), TRUE), "activate actual src pad");
    auto state = std::make_shared<SourceStopState>();
    state->attach(GST_PAD(pad.get()));
    require(state->snapshot() == 0, "no invented stop before event");
    gst_pad_push_event(GST_PAD(pad.get()), gst_event_new_stream_start("no-network-native-stop"));
    require(state->snapshot() == 0, "stream start cannot stop a source");
    gst_pad_push_event(GST_PAD(pad.get()), gst_nvevent_new_stream_eos(1));
    require(state->snapshot() == 2, "actual source1 EOS not latched");
    gst_pad_push_event(GST_PAD(pad.get()), gst_nvevent_new_stream_eos(1));
    require(state->snapshot() == 2, "duplicate native EOS not idempotent");
    gst_pad_push_event(GST_PAD(pad.get()), gst_nvevent_new_pad_deleted(2));
    require(state->snapshot() == 6, "actual source2 deletion not latched");
    state->close(); state->close();
    gst_pad_push_event(GST_PAD(pad.get()), gst_nvevent_new_stream_eos(0));
    require(state->snapshot() == 6, "closed observer still receives events");
    std::weak_ptr<SourceStopState> weak = state;
    state.reset(); require(weak.expired(), "event context retained after close");
    auto bad = std::make_shared<SourceStopState>(); bad->attach(GST_PAD(pad.get()));
    gst_pad_push_event(GST_PAD(pad.get()), gst_nvevent_new_stream_eos(3));
    bool refused = false;
    try { bad->snapshot(); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "out-of-graph source accepted"); bad->close();
    std::cout << "{\"native_stream_eos\":true,\"native_pad_deleted\":true,\"idempotent\":true,"
                 "\"closed\":true,\"state_released\":true,\"invalid_source_refused\":true,"
                 "\"network_sources\":0,\"gpu_inference\":false}" << std::endl;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
