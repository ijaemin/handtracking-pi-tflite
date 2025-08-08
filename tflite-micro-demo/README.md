# TFLite Micro Demo

라즈베리 파이에서 **TensorFlow Lite Micro(C++/POSIX)** 로 손 검출/추론을 실행하는 데모입니다. CPU 전용으로 동작하며, 모델/앵커 등 자산은 `src/`에 포함되어 있습니다.

## Requirements

- GNU Make, g++ (표준 빌드 도구)
- 인터넷 연결(최초 한 번 TFLite Micro 소스 가져오기)

## Quickstart

```bash
# 리포 루트에서 시작
cd ~/handtracking-pi-tflite

# 1) TFLite Micro 가져오기 + microlite 빌드 (최초 1회)
git clone https://github.com/tensorflow/tflite-micro.git
make -C tflite-micro -f tensorflow/lite/micro/tools/make/Makefile microlite

# 2) 데모 빌드
cd tflite-micro-demo
make                    # 또는 ./build_pi.sh 가 있다면 그 스크립트를 사용
```

## Usage

```bash
# 카메라 입력
./hand_demo --cam

# 단일 이미지 입력 (PPM)
cd ..
./scripts/make_ppm.sh sample_images/sample.jpg sample_images/sample.ppm
cd tflite-micro-demo
./hand_demo ../sample_images/sample.ppm
```

> 실행 파일 이름/위치는 `make` 설정에 따라 달라질 수 있습니다. 빌드 후 `find . -maxdepth 2 -type f -executable -name "*hand*"` 로 확인하세요.

## Files

- `Makefile` : 증분 빌드용
- `build_pi.sh` : 풀 빌드 스크립트(있는 경우)
- `src/` : 소스 코드 및 모델 데이터
- `hand_demo` : 빌드 산출 실행 파일(이름은 환경에 따라 다를 수 있음)

## CLI Options

| Option       | Description          | Example                                   |
| ------------ | -------------------- | ----------------------------------------- |
| `--cam`      | 카메라 입력으로 실행 | `./hand_demo --cam`                       |
| `<ppm_path>` | PPM 이미지 파일 추론 | `./hand_demo ../sample_images/sample.ppm` |

> 추가 옵션은 `./hand_demo -h` 로 확인하세요.

## Performance

> 10–15초 실행 후 콘솔/창에 표시되는 평균 FPS를 기록하세요. 동일 조건(해상도/조명)에서 2–3회 측정 권장.

| Device                 | Input            | Resolution | FPS (avg)            | Notes      |
| ---------------------- | ---------------- | ---------- | -------------------- | ---------- |
| Raspberry Pi 5 Model B | Camera (`--cam`) | 300×300    | ~0.16 FPS (≈6392 ms) |            |
| Raspberry Pi 5 Model B | PPM file         | 300×300    | ~0.16 FPS (≈6392 ms) | 10-run avg |
