FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch@sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="Noesis DS9 development/runtime"
LABEL org.opencontainers.image.base.digest="sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a"
LABEL org.opencontainers.image.source="Noesis DS9/docker/Dockerfile"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# The NVIDIA image records several codec packages in dpkg while omitting their
# shared objects from the final filesystem. Explicit version pins plus
# --reinstall make the resulting GStreamer surface deterministic.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      python3-dev=3.12.3-0ubuntu2.1 \
      ripgrep=14.1.0-1 \
      gstreamer1.0-nice=0.1.21-2build3 \
      gstreamer1.0-libav=1.24.1-1build1 \
      gstreamer1.0-plugins-good=1.24.2-1ubuntu1.4 \
      gstreamer1.0-plugins-bad=1.24.2-1ubuntu4 \
      gstreamer1.0-plugins-ugly=1.24.1-1build1 \
    && apt-get install -y --reinstall --no-install-recommends \
      libflac12t64=1.4.3+ds-2.1ubuntu2 \
      libdca0=0.0.7-2build1 \
      libdvdnav4=6.1.1-3build1 \
      libdvdread8t64=6.1.3-1.1build1 \
      libjbig0=2.1-6.1ubuntu2 \
      libmjpegutils-2.1-0t64=1:2.1.0+debian-8.1build1 \
      libmplex2-2.1-0t64=1:2.1.0+debian-8.1build1 \
      libmp3lame0=3.100-6build1 \
      libmpg123-0t64=1.32.5-1ubuntu1.1 \
      libvpx9=1.14.0-1ubuntu2.3 \
      libx264-164=2:0.164.3108+git31e19f9-1 \
      libx265-199=3.5-2build1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt /opt/noesis/requirements.lock.txt

RUN python3 -m pip install --no-cache-dir -r /opt/noesis/requirements.lock.txt \
    && python3 -m pip check \
    && test "$(python3 -c 'import pybind11; print(pybind11.__version__)')" = "2.12.0"

RUN GST_REGISTRY=/tmp/noesis-ds9-gst-registry.bin \
      gst-inspect-1.0 nice >/dev/null \
    && GST_REGISTRY=/tmp/noesis-ds9-gst-registry.bin \
      gst-inspect-1.0 avdec_h264 >/dev/null \
    && rm -f /tmp/noesis-ds9-gst-registry.bin
