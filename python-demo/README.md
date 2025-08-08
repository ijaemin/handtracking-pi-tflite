# Python Demo

라즈베리 파이에서 **TensorFlow Lite(Python)** 로 손 검출/추론을 실행하는 데모입니다. 모델/앵커는 `python-demo/models/` 경로를 기본값으로 사용합니다.

## Quickstart

```bash
# 1) 가상환경 (권장)
python -m venv .venv && source .venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 실행
python live_cam_int8.py
# 또는 단일 이미지 추론
python infer_image_int8.py --image ../sample_images/sample.jpg
```

## Files

- `live_cam_int8.py` : 카메라 입력 실시간 추론
- `infer_image_int8.py` : 단일 이미지 추론
- `infer_image_int8_nms.py` : 단일 이미지 + NMS 적용
- `models/` : 모델 자산( `hand_int8_io_noanchors.tflite`, `anchors_ssd300.npy` )

## CLI Options

### infer_image_int8.py

| Option             | Description                      | Default                                | Example                                        |
| ------------------ | -------------------------------- | -------------------------------------- | ---------------------------------------------- |
| `--image <path>`   | Single image to run inference on | –                                      | `--image ../sample_images/sample.jpg`          |
| `--model <path>`   | TFLite model path                | `models/hand_int8_io_noanchors.tflite` | `--model models/hand_int8_io_noanchors.tflite` |
| `--anchors <path>` | SSD anchors (.npy)               | `models/anchors_ssd300.npy`            | `--anchors models/anchors_ssd300.npy`          |

### infer_image_int8_nms.py

| Arg/Option         | Description                    | Default                                | Example                                        |
| ------------------ | ------------------------------ | -------------------------------------- | ---------------------------------------------- |
| `<image_path>`     | Single image path (positional) | –                                      | `../sample_images/sample.jpg`                  |
| `[score]`          | (optional) Score threshold     | see `-h`                               | `0.35`                                         |
| `[nms_iou]`        | (optional) NMS IoU threshold   | see `-h`                               | `0.5`                                          |
| `--model <path>`   | Override model path            | `models/hand_int8_io_noanchors.tflite` | `--model models/hand_int8_io_noanchors.tflite` |
| `--anchors <path>` | Override anchors path          | `models/anchors_ssd300.npy`            | `--anchors models/anchors_ssd300.npy`          |

### live_cam_int8.py

| Option             | Description                          | Default                                | Example                                        |
| ------------------ | ------------------------------------ | -------------------------------------- | ---------------------------------------------- |
| _(none required)_  | Run with default camera              | –                                      | `python live_cam_int8.py`                      |
| `--model <path>`   | Override model path (if supported)   | `models/hand_int8_io_noanchors.tflite` | `--model models/hand_int8_io_noanchors.tflite` |
| `--anchors <path>` | Override anchors path (if supported) | `models/anchors_ssd300.npy`            | `--anchors models/anchors_ssd300.npy`          |
| `--width <int>`    | Capture width (if supported)         | see `-h`                               | `--width 640`                                  |
| `--height <int>`   | Capture height (if supported)        | see `-h`                               | `--height 480`                                 |

> 각 스크립트의 실제 인자/기본값은 버전에 따라 다를 수 있습니다. 최신 목록은 `-h/--help`로 확인하세요.

## Performance

| Device                 | Script                  | Resolution | NMS | FPS (avg)             | Notes      |
| ---------------------- | ----------------------- | ---------- | --- | --------------------- | ---------- |
| Raspberry Pi 5 Model B | live_cam_int8.py        | 300×300    | on  | 20.34 FPS             | 15s bench  |
| Raspberry Pi 5 Model B | infer_image_int8.py     | 300×300    | off | ~5.9 img/s (≈168 ms)  | 50-run avg |
| Raspberry Pi 5 Model B | infer_image_int8_nms.py | 300×300    | on  | ~3.45 img/s (≈290 ms) | 50-run avg |
