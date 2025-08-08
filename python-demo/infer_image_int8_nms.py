import sys, math
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Prefer tflite_runtime interpreter
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback

# Define paths relative to this script
SCRIPT_DIR  = Path(__file__).resolve().parent
MODEL_PATH  = SCRIPT_DIR / "models" / "hand_int8_io_noanchors.tflite"
ANCH_PATH   = SCRIPT_DIR / "models" / "anchors_ssd300.npy"
DEFAULT_IMG = SCRIPT_DIR.parent / "sample_images" / "sample.jpg"
IMG_PATH    = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG

IN_W = IN_H = 300
# SSD-Lite scale factors (assume default pipeline config)
Y_SCALE, X_SCALE, H_SCALE, W_SCALE = 10.0, 10.0, 5.0, 5.0

SCORE_THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
IOU_THR   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
MAX_DETS  = 20

def load_img_int8(path):
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    img_r = img.resize((IN_W, IN_H), Image.BILINEAR)
    u8  = np.asarray(img_r, dtype=np.uint8)
    q   = (u8.astype(np.int16) - 128).astype(np.int8)
    return img, q[None, ...], (w0, h0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def decode_boxes(raw_boxes, anchors):
    ty, tx, th, tw = np.split(raw_boxes, 4, axis=-1)
    ay, ax, ah, aw = np.split(anchors,   4, axis=-1)
    ycenter = ty / Y_SCALE * ah + ay
    xcenter = tx / X_SCALE * aw + ax
    h = np.exp(th / H_SCALE) * ah
    w = np.exp(tw / W_SCALE) * aw
    y1 = ycenter - 0.5 * h
    x1 = xcenter - 0.5 * w
    y2 = ycenter + 0.5 * h
    x2 = xcenter + 0.5 * w
    y1 = np.clip(y1, 0.0, 1.0); x1 = np.clip(x1, 0.0, 1.0)
    y2 = np.clip(y2, 0.0, 1.0); x2 = np.clip(x2, 0.0, 1.0)
    return np.concatenate([x1, y1, x2, y2], axis=-1)

def iou_xyxy(a, b):
    # a: [N,4], b: [M,4]
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    return inter / (area_a[:,None] + area_b[None,:] - inter + 1e-6)

def nms(boxes, scores, iou_thr=0.5, max_dets=20):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i:i+1], boxes[order[1:]])[0]
        remain = np.where(ious <= iou_thr)[0] + 1
        order = order[remain]
    return np.array(keep, dtype=np.int32)

def main():
    # Load model and anchors
    interp = Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    outs = interp.get_output_details()
    b_scale, b_zp = outs[0]["quantization"]
    c_scale, c_zp = outs[1]["quantization"]
    anchors = np.load(str(ANCH_PATH))

    # Prepare input and run inference
    img0, x, (w0, h0) = load_img_int8(IMG_PATH)
    assert inp["dtype"] == np.int8
    interp.set_tensor(inp["index"], x); interp.invoke()

    boxes_q = interp.get_tensor(outs[0]["index"])[0]
    class_q = interp.get_tensor(outs[1]["index"])[0]

    boxes  = b_scale * (boxes_q.astype(np.int32) - b_zp)
    logits = c_scale * (class_q.astype(np.int32) - c_zp)
    probs  = softmax(logits, axis=-1)
    hand_scores = probs[:, 1]  # class 1 = hand

    # Score filtering
    sel = np.where(hand_scores >= SCORE_THR)[0]
    if sel.size == 0:
        print(f"No detections over threshold {SCORE_THR}. Try lowering it (e.g., 0.25).")
        return
    boxes_dec = decode_boxes(boxes[sel], anchors[sel])  # [N,4] normalized
    boxes_px = boxes_dec.copy()
    boxes_px[:, [0,2]] *= w0
    boxes_px[:, [1,3]] *= h0

    # NMS
    keep = nms(boxes_px, hand_scores[sel], iou_thr=IOU_THR, max_dets=MAX_DETS)
    boxes_px = boxes_px[keep]
    scores_k = hand_scores[sel][keep]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(img0)
    for (x1,y1,x2,y2), s in zip(boxes_px, scores_k):
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
        draw.text((x1, max(0,y1-12)), f"{s:.3f}", fill=(255,0,0))

    out_path = SCRIPT_DIR / "out.jpg"
    img0.save(out_path, quality=92)

    print(f"Detections: {len(scores_k)}  (thr={SCORE_THR}, iou={IOU_THR})")
    for i,(bb, s) in enumerate(zip(boxes_px, scores_k), 1):
        x1,y1,x2,y2 = bb
        print(f"{i}: {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f} score={s:.3f}")
    print(f"Saved: {out_path}")
    print("Done.")
if __name__ == "__main__":
    main()
