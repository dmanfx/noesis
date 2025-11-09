DS_HOME ?= /opt/nvidia/deepstream/deepstream
CUDA_HOME ?= /usr/local/cuda

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC \
            -I$(DS_HOME)/sources/includes \
            -I$(CUDA_HOME)/include
LDFLAGS := -shared -L$(CUDA_HOME)/lib64 -lcudart
TARGET := libnvdsinfer_yolo11_seg.so
SOURCES := nvdsinfer_yolo11_seg.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
