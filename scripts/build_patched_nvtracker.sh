#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_DIR_DEFAULT="/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvtracker"
SRC_DIR="${1:-$SRC_DIR_DEFAULT}"

PATCH_FILE="$REPO_ROOT/patches/deepstream/gst-nvtracker_mask_params_leak_fix.patch"

OUT_DIR="$REPO_ROOT/build/gst-plugins"
BUILD_ROOT="$REPO_ROOT/build/nvtracker_patched_src"
DEEPSTREAM_SYS_DIR="/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream"
DEEPSTREAM_OVERRIDE_DIR="$REPO_ROOT/build/gst-plugins-deepstream"
GST_SYS_DIR="/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
GST_SYS_OVERRIDE_DIR="$REPO_ROOT/build/gst-plugins-system"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[FAIL] DeepStream nvtracker source dir not found: $SRC_DIR" >&2
  exit 2
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "[FAIL] Patch file not found: $PATCH_FILE" >&2
  exit 2
fi

CUDA_VER=""
if [[ -L /usr/local/cuda ]]; then
  cuda_target="$(readlink -f /usr/local/cuda || true)"
  if [[ "$cuda_target" =~ cuda-([0-9]+)\.([0-9]+)$ ]]; then
    CUDA_VER="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  fi
fi
if [[ -z "$CUDA_VER" ]]; then
  # Fallback: pick the highest /usr/local/cuda-* entry.
  cuda_dir="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ "$cuda_dir" =~ cuda-([0-9]+)\.([0-9]+)$ ]]; then
    CUDA_VER="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  fi
fi
if [[ -z "$CUDA_VER" ]]; then
  echo "[FAIL] Unable to determine CUDA_VER (expected /usr/local/cuda or /usr/local/cuda-<major>.<minor>)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

SRC_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
if [[ ! -d "$SRC_ROOT/includes" ]]; then
  echo "[FAIL] DeepStream sources root missing includes/: $SRC_ROOT" >&2
  exit 2
fi
if [[ ! -d "$SRC_ROOT/gst-plugins/gst-nvdspreprocess" ]]; then
  echo "[FAIL] DeepStream sources root missing gst-nvdspreprocess/: $SRC_ROOT" >&2
  exit 2
fi

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT/sources/gst-plugins"

cp -av "$SRC_ROOT/includes" "$BUILD_ROOT/sources/" >/dev/null
cp -av "$SRC_ROOT/gst-plugins/gst-nvdspreprocess" "$BUILD_ROOT/sources/gst-plugins/" >/dev/null
cp -av "$SRC_ROOT/gst-plugins/gst-nvtracker" "$BUILD_ROOT/sources/gst-plugins/" >/dev/null

PATCH_DIR="$BUILD_ROOT/sources/gst-plugins/gst-nvtracker"
patch -p0 -d "$PATCH_DIR" <"$PATCH_FILE" >/dev/null

echo "[INFO] Building patched nvtracker plugin (CUDA_VER=$CUDA_VER) ..."
CUDA_VER="$CUDA_VER" make -C "$PATCH_DIR" clean >/dev/null || true
CUDA_VER="$CUDA_VER" make -C "$PATCH_DIR" -j"$(nproc)" >/dev/null

cp -av "$PATCH_DIR/libnvdsgst_tracker.so" "$OUT_DIR/libnvdsgst_tracker.so" >/dev/null

if [[ -d "$DEEPSTREAM_SYS_DIR" ]]; then
  rm -rf "$DEEPSTREAM_OVERRIDE_DIR"
  mkdir -p "$DEEPSTREAM_OVERRIDE_DIR"
  for so in "$DEEPSTREAM_SYS_DIR"/*.so; do
    base="$(basename "$so")"
    if [[ "$base" == "libnvdsgst_tracker.so" ]]; then
      ln -sf "$OUT_DIR/libnvdsgst_tracker.so" "$DEEPSTREAM_OVERRIDE_DIR/$base"
    else
      ln -sf "$so" "$DEEPSTREAM_OVERRIDE_DIR/$base"
    fi
  done
fi

if [[ -d "$GST_SYS_DIR" ]]; then
  rm -rf "$GST_SYS_OVERRIDE_DIR"
  mkdir -p "$GST_SYS_OVERRIDE_DIR"
  for so in "$GST_SYS_DIR"/*.so; do
    base="$(basename "$so")"
    ln -sf "$so" "$GST_SYS_OVERRIDE_DIR/$base"
  done
fi

echo "[PASS] Built patched plugin: $OUT_DIR/libnvdsgst_tracker.so"
echo "To use it for DS8 runs:"
echo "  export NOESIS_USE_PATCHED_NVTRACKER=1"
