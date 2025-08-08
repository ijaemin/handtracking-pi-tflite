# Hand Tracking with TFLite & TFLite Micro on Raspberry Pi

라즈베리 파이용 SSD-Lite MobileNetV2 기반 Hand Detection/Tracking 모델 데모 (TFLite/TFLite Micro)

## Getting Started

- [Python Quickstart](python-demo/)
- [TFLite Micro Quickstart](tflite-micro-demo/)

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Overview](#project-overview)
3. [Environment & Requirements](#environment--requirements)
4. [Directory Structure](#directory-structure)
5. [Python Demo](#python-demo)
6. [TFLite Micro Demo](#tflite-micro-demo)
7. [Utilities](#utilities)
8. [Known Issues](#known-issues)
9. [License & Credits](#license--credits)

## Project Overview

이 프로젝트는 라즈베리 파이용 **Hand Detection/Tracking 모델 데모**입니다. SSD-Lite MobileNetV2 기반으로, 이미지/카메라 입력을 모두 지원하며 **빠른 재현 및 성능 측정**을 제공합니다. **INT8 양자화**를 적용해 메모리 사용량과 지연을 줄였으며, 추가 가속기 없이 구동하기 쉽습니다.

- **Python Demo**: TFLite 모델(Python) → [바로가기](python-demo/)
- **TFLite Micro Demo**: TFLite-micro 모델(C++ POSIX) → [바로가기](tflite-micro-demo/)

## Environment & Requirements

> 아래 항목은 실제 테스트 환경을 반영합니다.

**Hardware**

- Device: Raspberry Pi 5 Model B
- Camera: Raspberry Pi Camera Module 3 (IMX708)

**OS / Toolchain**

- OS: Debian GNU/Linux 12 (bookworm)
- Arch/Kernel: aarch64, 6.12.34+rpt-rpi-2712 (2025-06-26)
- Python: 3.11.2 (venv)

**Python dependencies**

- `requirements.txt` 참고 (예: numpy, opencv-python, tflite-runtime 등)

**빠른 설치 예시**

```bash
cd python-demo
pip3 install -r requirements.txt
```

**검증된 환경**
| Device | OS (build date) | Python | Notes |
|---|---|---|---|
| Raspberry Pi 5 Model B | Debian 12 (bookworm), kernel 6.12.34+rpt-rpi-2712 (2025-06-26), aarch64 | 3.11.2 (venv) | Camera: Raspberry Pi Camera Module 3 (IMX708) |

## Directory Structure

```
handtracking-pi-tflite/
├── python-demo/
│   ├── models/  # model assets
│   └── …        # scripts
├── tflite-micro-demo/
│   ├── Makefile
│   ├── build_pi.sh
│   └── src/     # sources & model data
├── sample_images/
├── scripts/
│   └── make_ppm.sh
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── README.md
└── .gitignore
```

## Python Demo

TFLite(Python) 기반 데모입니다. 이미지/카메라 입력을 지원하며 빠른 재현에 적합합니다.  
자세한 실행 방법과 옵션은 [python-demo/](python-demo/)를 참고하세요.

## TFLite Micro Demo

TFLite Micro(C++/POSIX) 기반 데모입니다. 카메라 입력과 PPM 파일 추론을 지원합니다.  
자세한 실행 방법과 옵션은 [tflite-micro-demo/](tflite-micro-demo/)를 참고하세요.

## Utilities

- **PPM 변환**:
  ```bash
  ./scripts/make_ppm.sh sample_images/sample.jpg sample_images/sample.ppm
  ```

## Known Issues

## License & Credits

### Our Code

- Unless otherwise noted, the code in this repository is released under the **MIT License**.  
  Copyright (c) 2025 jaemin lee

### Third-Party

- **Hand model & checkpoints**: Derived from **victordibia/handtracking** (MIT License).  
  © 2020 Victor Dibia. See the original repository and include the MIT license notice.
- **Upstream training dataset**: **EgoHands** (Bambach et al., ICCV 2015).
- **Runtimes/Libraries**: TensorFlow Lite / TensorFlow Lite Micro — **Apache-2.0**.

### Notices

- This project redistributes or references model weights converted from the victordibia/handtracking checkpoints.  
  We include the **original MIT license text** for those assets and acknowledge the **EgoHands** dataset.
