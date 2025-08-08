# Makefile for TFLite Micro demo on Raspberry Pi

# Paths
TFLM_DIR ?= $(HOME)/tflite-micro
SRC_DIR  := src
BUILD_DIR:= $(TFLM_DIR)/user_build
LIB      := $(TFLM_DIR)/gen/linux_aarch64_default_gcc/lib/libtensorflow-microlite.a
CXXFLAGS := -std=c++17 -O2 -DTF_LITE_STATIC_MEMORY \
             -I$(TFLM_DIR) \
             -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads \
             -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
             -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
             -I$(TFLM_DIR)/gen/linux_aarch64_default_gcc/genfiles \
             $(shell pkg-config --cflags opencv4)
LDFLAGS  := -lm $(LIB) $(shell pkg-config --libs opencv4)

# Default target: build executable
all: $(BUILD_DIR)/hand_demo

# Build demo executable
$(BUILD_DIR)/hand_demo: $(BUILD_DIR)/main.o \
                       $(BUILD_DIR)/hand_model_data.o \
                       $(BUILD_DIR)/anchors_model_data.o
	$(CXX) $^ -o $@ $(LDFLAGS)

# Compile main.cc
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cc
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile model data
$(BUILD_DIR)/hand_model_data.o: $(SRC_DIR)/hand_model_data.cc
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/anchors_model_data.o: $(SRC_DIR)/anchors_model_data.cc
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)