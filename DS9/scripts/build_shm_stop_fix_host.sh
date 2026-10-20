#!/usr/bin/env bash
# Build ONLY the pinned same-factory SHM plugin; never install/select it.
set -euo pipefail
if [[ $# != 1 || ! -d "$1" || "$1" != /* ]]; then
  printf 'usage: bash build_shm_stop_fix_host.sh EXISTING_ABSOLUTE_OUTPUT_DIR\n' >&2
  exit 2
fi
output_dir="$(realpath -- "$1")"
case "$output_dir" in
  /|/usr|/usr/*|/opt|/opt/*|/lib|/lib/*|/etc|/etc/*)
    printf 'refusing system output directory\n' >&2; exit 2 ;;
esac
for dependency in gstreamer-1.0 gstreamer-base-1.0; do
  [[ "$(pkg-config --modversion "$dependency")" == 1.24.2 ]] || {
    printf 'requires installed %s 1.24.2 headers/libraries\n' "$dependency" >&2; exit 2;
  }
done
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../native/gst_shm_stop_fix" && pwd)"
read -r -a flags <<< "$(pkg-config --cflags --libs gstreamer-base-1.0)"
cc -std=c11 -D_GNU_SOURCE -DSHM_PIPE_USE_GLIB -O2 -fPIC -shared -Wl,-z,defs \
  '-DVERSION="1.24.2"' '-DPACKAGE="gst-plugins-bad"' '-DGST_LICENSE="LGPL"' \
  '-DGST_PACKAGE_NAME="GStreamer Bad Plugins (Noesis stop fix)"' \
  '-DGST_PACKAGE_ORIGIN="https://gstreamer.freedesktop.org/"' \
  "$source_dir/gstshm.c" "$source_dir/gstshmsink.c" "$source_dir/gstshmsrc.c" \
  "$source_dir/shmpipe.c" "$source_dir/shmalloc.c" \
  "${flags[@]}" -lrt -o "$output_dir/libgstshm.so"
printf '%s\n' "$output_dir/libgstshm.so"
