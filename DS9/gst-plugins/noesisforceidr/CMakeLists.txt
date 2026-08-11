cmake_minimum_required(VERSION 3.16)
project(noesis_force_idr_ds9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0)

set(DEEPSTREAM_HOME "$ENV{NOESIS_DEEPSTREAM_HOME}" CACHE PATH "DeepStream SDK root")
if(NOT DEEPSTREAM_HOME)
  set(DEEPSTREAM_HOME "/opt/nvidia/deepstream/deepstream-9.0")
endif()

set(NV_EVENT_HEADER "${DEEPSTREAM_HOME}/sources/includes/gst-nvcustomevent.h")
set(NV_EVENT_LIBRARY "${DEEPSTREAM_HOME}/lib/libgstnvcustomhelper.so")
if(NOT EXISTS "${NV_EVENT_HEADER}")
  message(FATAL_ERROR "Missing NVIDIA custom-event header: ${NV_EVENT_HEADER}")
endif()
if(NOT EXISTS "${NV_EVENT_LIBRARY}")
  message(FATAL_ERROR "Missing NVIDIA custom-event library: ${NV_EVENT_LIBRARY}")
endif()

add_library(gstnoesisforceidr SHARED gstnoesisforceidr.cpp)
target_include_directories(gstnoesisforceidr PRIVATE
  ${GST_INCLUDE_DIRS}
  "${DEEPSTREAM_HOME}/sources/includes"
)
target_compile_options(gstnoesisforceidr PRIVATE
  ${GST_CFLAGS_OTHER}
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)
target_compile_definitions(gstnoesisforceidr PRIVATE
  PACKAGE="noesis"
  PACKAGE_VERSION="1.0.0"
)
target_link_directories(gstnoesisforceidr PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(gstnoesisforceidr PRIVATE
  ${GST_LIBRARIES}
  "${NV_EVENT_LIBRARY}"
)
set_target_properties(gstnoesisforceidr PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "gstnoesisforceidr"
  BUILD_RPATH "${DEEPSTREAM_HOME}/lib"
  INSTALL_RPATH "${DEEPSTREAM_HOME}/lib"
)
