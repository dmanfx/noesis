CUDA_VER?=12.6
DS?=/opt/nvidia/deepstream/deepstream-7.1

CXX?=g++
CXXFLAGS+=-fPIC -O2 -std=c++14 \
  -I$(DS)/sources/includes \
  -I$(DS)/sources/gst-plugins/gst-nvdspreprocess \
  -I$(DS)/sources/gst-plugins/gst-nvinfer \
  -I/usr/include/gstreamer-1.0 \
  -I/usr/include/glib-2.0 -I/usr/lib/$(shell uname -m)-linux-gnu/glib-2.0/include

LDFLAGS+=-shared -L$(DS)/lib -Wl,-rpath,$(DS)/lib
LIBS+=-lnvds_meta -lgstreamer-1.0 -lglib-2.0 -lgobject-2.0

TARGET=libds_preprocess_shim.so
SRCS=ds_preprocess_shim.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)
