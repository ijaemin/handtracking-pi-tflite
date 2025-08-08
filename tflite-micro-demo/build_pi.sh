#!/usr/bin/env bash
#
# build_pi.sh: Build and run the TFLite Micro demo on Raspberry Pi
# Prerequisite: Clone tflite-micro into the same parent directory, or set TFLM_DIR to its path.
set -e


# Determine repository root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths (can override TFLM_DIR to point elsewhere)
MICRO_DIR="${TFLM_DIR:-$HOME/tflite-micro}"
SRC_DIR="$REPO_ROOT/tflite-micro-demo/src"
BUILD_DIR="$MICRO_DIR/user_build"

# 1) Build TFLite Micro library if needed
cd "$MICRO_DIR"
make -f tensorflow/lite/micro/tools/make/Makefile microlite

# 2) Prepare build dir
mkdir -p "$BUILD_DIR"

# 3) Compile sources
g++ -std=c++17 -O2 -DTF_LITE_STATIC_MEMORY \
  -I"$MICRO_DIR" \
  -I"$MICRO_DIR"/tensorflow/lite/micro/tools/make/downloads \
  -I"$MICRO_DIR"/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -I"$MICRO_DIR"/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I"$MICRO_DIR"/gen/linux_aarch64_default_gcc/genfiles \
  $(pkg-config --cflags opencv4) \
  -c "$SRC_DIR/main.cc" -o "$BUILD_DIR/main.o"

g++ -std=c++17 -O2 -c "$SRC_DIR/hand_model_data.cc" -o "$BUILD_DIR/hand_model_data.o"
g++ -std=c++17 -O2 -c "$SRC_DIR/anchors_model_data.cc" -o "$BUILD_DIR/anchors_model_data.o"

# 4) Link executable
g++ -std=c++17 -O2 \
  -o "$BUILD_DIR/hand_demo" \
  "$BUILD_DIR/main.o" "$BUILD_DIR/hand_model_data.o" "$BUILD_DIR/anchors_model_data.o" \
  "$MICRO_DIR"/gen/linux_aarch64_default_gcc/lib/libtensorflow-microlite.a -lm \
  $(pkg-config --libs opencv4)

echo "Build complete. Run with:"
echo "  $BUILD_DIR/hand_demo --cam"