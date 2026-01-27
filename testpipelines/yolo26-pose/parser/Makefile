DS_HOME ?= /opt/nvidia/deepstream/deepstream-8.0
CUDA_HOME ?= /usr/local/cuda

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC \
            -I$(DS_HOME)/sources/includes \
            -I$(CUDA_HOME)/include
LDFLAGS := -shared -L$(CUDA_HOME)/lib64 -lcudart
TARGET := libyolo26_pose_parser.so
SOURCES := yolo26_pose_parser.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
