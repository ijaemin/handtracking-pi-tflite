# Hand Tracking with TFLite & TFLite Micro on Raspberry Pi

Quantized SSD-Lite MobileNetV2 hand-tracking demos running on Raspberry Pi OS.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Directory Structure](#directory-structure)  
3. [Python Demo](#python-demo)  
4. [TFLite Micro Demo](#tflite-micro-demo)  
5. [Utilities](#utilities)  
6. [License](#license)  

---

## Project Overview

이 프로젝트는 Raspberry Pi에서 양자화된 SSD-Lite MobileNetV2 기반 핸드트래킹 예제입니다.

- **Python Demo**: TFLite 모델(Python)
- **TFLite Micro Demo**: TFLite-micro 모델(C++ POSIX)

---

## Directory Structure

```
handtracking-pi-tflite/
├── python-demo/           # TFLite Python scripts & models
├── sample_images/         # Test images (JPEG, PPM)
├── scripts/               # Utility scripts (e.g., make_ppm.sh)
├── tflite-micro-demo/     # C++ demo for TFLite Micro
│   ├── Makefile           # Incremental build
│   ├── build_pi.sh        # Full build script
│   └── src/               # Source & model data
└── .gitignore
```

---

## Python Demo

사전 준비 및 실행 방법:

```
cd python-demo
pip3 install -r requirements.txt

# Single-image inference
python3 infer_image_int8.py \
  --model hand_int8_io_noanchors.tflite \
  --anchors anchors_ssd300.npy \
  --image ../sample_images/sample.jpg

# Inference + NMS
python3 infer_image_int8_nms.py \
  --model hand_int8_io_noanchors.tflite \
  --anchors anchors_ssd300.npy \
  --image ../sample_images/sample.jpg

# Live camera
python3 live_cam_int8.py
```

---

## TFLite Micro Demo

Raspberry Pi에서 빌드 및 실행:

```
# 1) Install OS deps
sudo apt update
sudo apt install -y build-essential pkg-config libopencv-dev python3-pip

# 2) Clone & build TFLite Micro library (once)
git clone https://github.com/tensorflow/tflite-micro.git
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile microlite

# 3) Build demo
cd ../handtracking-pi-tflite/tflite-micro-demo
make            # or: ./build_pi.sh

# 4) Run
./user_build/hand_demo --cam
# or with a PPM image:
./user_build/hand_demo ../sample_images/sample.ppm
```

---

## Utilities

- **PPM 변환**:
  ```bash
  ./scripts/make_ppm.sh sample_images/sample.jpg sample_images/sample.ppm
  ```
