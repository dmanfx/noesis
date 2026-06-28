DS_HOME ?= /opt/nvidia/deepstream/deepstream
CUDA_HOME ?= /usr/local/cuda

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC \
            -I$(DS_HOME)/sources/includes \
            -I$(CUDA_HOME)/include
LDFLAGS := -shared -L$(CUDA_HOME)/lib64 -lcudart
TARGET := libnvdsinfer_deimv2_wholebody49.so
SOURCES := nvdsinfer_deimv2_wholebody49.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
