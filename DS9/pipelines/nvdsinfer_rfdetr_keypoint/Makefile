DS_HOME ?= /opt/nvidia/deepstream/deepstream-9.0
CUDA_HOME ?= /usr/local/cuda-13.1
OUT_DIR ?= .

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC \
            -I$(DS_HOME)/sources/includes \
            -I$(CUDA_HOME)/include
LDFLAGS := -shared -L$(CUDA_HOME)/lib64 -lcudart
TARGET := $(OUT_DIR)/libnvdsinfer_rfdetr_keypoint.so
SOURCES := nvdsinfer_rfdetr_keypoint.cpp
NVDS_VERSION_H := $(DS_HOME)/sources/includes/nvds_version.h

all: $(TARGET)

check-ds9:
	@test -f "$(NVDS_VERSION_H)" || (echo "[FAIL] Missing DS9 header: $(NVDS_VERSION_H)" >&2; exit 1)
	@grep -Eq 'NVDS_VERSION_MAJOR[[:space:]]+9|NVDS_VERSION[[:space:]].*9' "$(NVDS_VERSION_H)" || (echo "[FAIL] DS_HOME does not identify as DeepStream 9: $(DS_HOME)" >&2; exit 1)

$(TARGET): check-ds9 $(SOURCES)
	@mkdir -p "$(OUT_DIR)"
	$(CXX) $(CXXFLAGS) -shared $(SOURCES) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean check-ds9
