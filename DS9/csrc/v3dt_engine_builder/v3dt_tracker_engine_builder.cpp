#include <dlfcn.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nvdstracker.h"

namespace {

constexpr unsigned int kDefaultTrackerConversionPoolSize = 4U;

struct Options {
  std::string tracker_config;
  std::string tracker_library;
  unsigned int streams = 3;
  unsigned int width = 1920;
  unsigned int height = 1080;
  unsigned int gpu_id = 0;
};

unsigned int parse_uint(const char* raw, const char* label) {
  char* end = nullptr;
  const unsigned long value = std::strtoul(raw, &end, 10);
  if (raw == end || end == nullptr || *end != '\0' || value > 0xffffffffUL) {
    throw std::runtime_error(std::string("invalid ") + label + ": " + raw);
  }
  return static_cast<unsigned int>(value);
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value after " + arg);
    }
    const char* value = argv[++index];
    if (arg == "--tracker-config") {
      options.tracker_config = value;
    } else if (arg == "--tracker-lib") {
      options.tracker_library = value;
    } else if (arg == "--streams") {
      options.streams = parse_uint(value, "streams");
    } else if (arg == "--width") {
      options.width = parse_uint(value, "width");
    } else if (arg == "--height") {
      options.height = parse_uint(value, "height");
    } else if (arg == "--gpu-id") {
      options.gpu_id = parse_uint(value, "gpu-id");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.tracker_config.empty() || options.tracker_library.empty()) {
    throw std::runtime_error("--tracker-config and --tracker-lib are required");
  }
  if (options.streams == 0 || options.width == 0 || options.height == 0) {
    throw std::runtime_error("streams, width, and height must be positive");
  }
  return options;
}

template <typename Function>
Function symbol(void* handle, const char* name) {
  dlerror();
  void* raw = dlsym(handle, name);
  if (const char* error = dlerror()) {
    throw std::runtime_error(std::string("unable to load ") + name + ": " + error);
  }
  return reinterpret_cast<Function>(raw);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    void* library = dlopen(options.tracker_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      throw std::runtime_error(std::string("unable to load tracker library: ") + dlerror());
    }
    try {
      using Query = decltype(&NvMOT_Query);
      using Init = decltype(&NvMOT_Init);
      using Deinit = decltype(&NvMOT_DeInit);
      const Query query_fn = symbol<Query>(library, "NvMOT_Query");
      const Init init_fn = symbol<Init>(library, "NvMOT_Init");
      const Deinit deinit_fn = symbol<Deinit>(library, "NvMOT_DeInit");

      std::string config_path = options.tracker_config;
      if (config_path.size() > UINT16_MAX) {
        throw std::runtime_error("tracker config path exceeds the NvMOT uint16 length contract");
      }
      NvMOTPerTransformBatchConfig transform{};
      // NvMOT_Query is context-bound in DS9, so bootstrap NvMOT_Init with the
      // explicit dGPU requirements of this V3DT graph and verify them below.
      transform.bufferType = NVBUF_MEM_CUDA_DEVICE;
      transform.maxWidth = options.width;
      transform.maxHeight = options.height;
      transform.maxPitch = options.width * 2U * 4U;
      transform.maxSize = transform.maxPitch * options.height;

      NvMOTConfig config{};
      config.computeConfig = NVMOTCOMP_GPU;
      config.maxStreams = options.streams;
      // Match gst-nvtracker's default four converted surfaces per stream. The
      // helper initializes the exact NvMOT graph but never submits a frame.
      config.maxBufSurfAddrSize =
          options.streams * kDefaultTrackerConversionPoolSize;
      config.numTransforms = 1U;
      config.perTransformBatchConfig = &transform;
      config.miscConfig.gpuId = options.gpu_id;
      config.customConfigFilePathSize = static_cast<uint16_t>(config_path.size());
      config.customConfigFilePath = config_path.data();

      NvMOTContextHandle context = nullptr;
      NvMOTConfigResponse response{};
      const auto deinitialize = [&]() {
        if (context != nullptr) {
          deinit_fn(context);
          context = nullptr;
        }
      };
      try {
        const NvMOTStatus init_status = init_fn(&config, &context, &response);
        if (init_status != NvMOTStatus_OK || context == nullptr ||
            response.summaryStatus != NvMOTConfigStatus_OK) {
          throw std::runtime_error(
              "NvMOT_Init failed to initialize the DS9 V3DT model graph");
        }

        NvMOTQuery query{};
        query.contextHandle = context;
        const NvMOTStatus query_status = query_fn(
            static_cast<uint16_t>(config_path.size()), config_path.data(), &query);
        if (query_status != NvMOTStatus_OK) {
          throw std::runtime_error("NvMOT_Query rejected the V3DT tracker context");
        }
        if ((query.computeConfig & NVMOTCOMP_GPU) == 0U) {
          throw std::runtime_error("NvMOT_Query did not advertise GPU compute");
        }
        if (query.numTransforms != 1U) {
          throw std::runtime_error(
              "NvMOT_Query did not advertise the required input transform");
        }
        if (query.memType != NVBUF_MEM_CUDA_DEVICE) {
          throw std::runtime_error(
              "NvMOT_Query did not advertise CUDA device memory");
        }
        if (query.batchMode != NvMOTBatchMode_Batch) {
          throw std::runtime_error("NvMOT_Query did not advertise batch processing");
        }
      } catch (...) {
        deinitialize();
        throw;
      }
      deinitialize();
      std::cout << "[OK] DS9 NvMOT V3DT model graph initialized" << std::endl;
    } catch (...) {
      dlclose(library);
      throw;
    }
    dlclose(library);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << std::endl;
    return 1;
  }
}
