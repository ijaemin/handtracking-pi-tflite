import sys, math
import numpy as np
from PIL import Image
from pathlib import Path

# Prefer tflite_runtime interpreter
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_int8_io_noanchors.tflite"
ANCH_PATH  = SCRIPT_DIR / "anchors_ssd300.npy"
# Default to sample_images/sample.jpg in the project root
DEFAULT_IMG = SCRIPT_DIR.parent / "sample_images" / "sample.jpg"
IMG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG

IN_W = IN_H = 300  # Model input size
# SSD-Lite scale factors (adjust if config differs)
Y_SCALE, X_SCALE, H_SCALE, W_SCALE = 10.0, 10.0, 5.0, 5.0

def load_img_int8(path):
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    img_r = img.resize((IN_W, IN_H), Image.BILINEAR)
    u8 = np.asarray(img_r, dtype=np.uint8)
    q = (u8.astype(np.int16) - 128).astype(np.int8)
    return q[None, ...], (w0, h0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def decode_boxes(raw_boxes, anchors):
    # raw_boxes: [1917,4] (ty, tx, th, tw) — anchors: [1917,4] (y_c, x_c, h, w), all normalized
    ty, tx, th, tw = np.split(raw_boxes, 4, axis=-1)
    ay, ax, ah, aw = np.split(anchors, 4, axis=-1)

    ycenter = ty / Y_SCALE * ah + ay
    xcenter = tx / X_SCALE * aw + ax
    h = np.exp(th / H_SCALE) * ah
    w = np.exp(tw / W_SCALE) * aw

    y1 = ycenter - 0.5 * h
    x1 = xcenter - 0.5 * w
    y2 = ycenter + 0.5 * h
    x2 = xcenter + 0.5 * w

    # clip to [0,1]
    y1 = np.clip(y1, 0.0, 1.0); x1 = np.clip(x1, 0.0, 1.0)
    y2 = np.clip(y2, 0.0, 1.0); x2 = np.clip(x2, 0.0, 1.0)
    return np.concatenate([x1, y1, x2, y2], axis=-1)  # [N,4], xyxy normalized

def main():
    # Load model and anchors
    interp = Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    outs = interp.get_output_details()
    b_scale, b_zp = outs[0]["quantization"]
    c_scale, c_zp = outs[1]["quantization"]

    anchors = np.load(str(ANCH_PATH))  # (1917,4) float32

    # Prepare input
    x, (w0, h0) = load_img_int8(IMG_PATH)
    assert inp["dtype"] == np.int8
    interp.set_tensor(inp["index"], x)
    interp.invoke()

    boxes_q = interp.get_tensor(outs[0]["index"])[0]  # [1917,4] int8
    class_q = interp.get_tensor(outs[1]["index"])[0]  # [1917,2] int8

    # Dequantize outputs
    boxes = b_scale * (boxes_q.astype(np.int32) - b_zp)     # float
    logits = c_scale * (class_q.astype(np.int32) - c_zp)    # float
    probs = softmax(logits, axis=-1)                        # [N,2]
    hand_scores = probs[:, 1]                               # class 1 = hand assumed

    # Simple top-K selection (NMS to follow)
    K = 5
    idx = np.argsort(hand_scores)[-K:][::-1]
    boxes_dec = decode_boxes(boxes[idx], anchors[idx])      # [K,4] normalized xyxy
    # Scale boxes to original resolution
    boxes_px = boxes_dec.copy()
    boxes_px[:, [0,2]] *= w0
    boxes_px[:, [1,3]] *= h0

    print("Top-%d detections (x1,y1,x2,y2,score):" % K)
    for i in range(K):
        x1,y1,x2,y2 = boxes_px[i]
        s = float(hand_scores[idx[i]])
        print(f"{i+1}: {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}  score={s:.3f}")

    print(f"\nInference OK on: {IMG_PATH}")

if __name__ == "__main__":
    main()