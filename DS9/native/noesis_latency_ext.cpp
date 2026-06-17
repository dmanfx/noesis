#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <exception>

#include "buffer.hpp"

namespace py = pybind11;

namespace {

// Log only once per process to avoid spam if the API is missing or throws.
static bool s_logged_error = false;

py::list measure_buffer_latency(const deepstream::Buffer& buffer) {
  py::list out;
  try {
    auto samples = buffer.measureLatency();
    for (const auto& sample : samples) {
      py::dict d;
      d["source_id"] = sample.source_id;
      d["frame_num"] = sample.frame_num;
      d["latency_ms"] = sample.latency;  // DeepStream reports milliseconds
      out.append(std::move(d));
    }
  } catch (const std::exception& e) {
    if (!s_logged_error) {
      std::fprintf(stderr,
                   "[noesis_latency_ext] measureLatency() failed (logged once): %s\n",
                   e.what());
      s_logged_error = true;
    }
  } catch (...) {
    if (!s_logged_error) {
      std::fprintf(stderr,
                   "[noesis_latency_ext] measureLatency() failed (logged once): unknown error\n");
      s_logged_error = true;
    }
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(noesis_latency_ext, m) {
  m.doc() = "Noesis DS8 helper bindings for DeepStream Service Maker latency sampling";
  m.def("measure_buffer_latency", &measure_buffer_latency,
        py::arg("buffer"),
        "Return a list of {source_id, frame_num, latency_ms} for the given Service Maker Buffer.");
}
