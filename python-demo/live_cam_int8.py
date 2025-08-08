import os, sys, time, numpy as np
from pathlib import Path
from PIL import Image
from picamera2 import Picamera2

# Prefer tflite_runtime interpreter
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback

# OpenCV is optional
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "hand_int8_io_noanchors.tflite"
ANCH_PATH  = SCRIPT_DIR / "models" / "anchors_ssd300.npy"

IN_W = IN_H = 300
Y_SCALE, X_SCALE, H_SCALE, W_SCALE = 10.0, 10.0, 5.0, 5.0

SCORE_THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
IOU_THR   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
MAX_DETS  = int(sys.argv[3]) if len(sys.argv) > 3 else 10
TOPK_PRE_NMS = 200  # Use top K before NMS
MIN_BOX_AREA = 900  # pixel^2 (e.g., 30x30)

HAND_CLASS_INDEX = int(os.getenv("HAND_CLS", "1"))  # default 1; try HAND_CLS=0 if needed

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
    y1 = np.clip(ycenter - 0.5 * h, 0.0, 1.0)
    x1 = np.clip(xcenter - 0.5 * w, 0.0, 1.0)
    y2 = np.clip(ycenter + 0.5 * h, 0.0, 1.0)
    x2 = np.clip(xcenter + 0.5 * w, 0.0, 1.0)
    return np.concatenate([x1, y1, x2, y2], axis=-1)

def iou_xyxy(a, b):
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
    interp = Interpreter(model_path=str(MODEL_PATH), num_threads=2)
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    outs = interp.get_output_details()
    b_scale, b_zp = outs[0]["quantization"]
    c_scale, c_zp = outs[1]["quantization"]
    anchors = np.load(str(ANCH_PATH))

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()
    time.sleep(0.2)

    if HAS_CV2:
        cv2.namedWindow("hand-int8", cv2.WINDOW_NORMAL)

    t0 = time.time(); fcnt = 0
    try:
        while True:
            frame = picam2.capture_array()
            h0, w0, _ = frame.shape

            img_r = Image.fromarray(frame).convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
            u8 = np.asarray(img_r, dtype=np.uint8)
            x  = (u8.astype(np.int16) - 128).astype(np.int8)[None, ...]
            interp.set_tensor(inp["index"], x)
            interp.invoke()

            boxes_q = interp.get_tensor(outs[0]["index"])[0]
            class_q = interp.get_tensor(outs[1]["index"])[0]
            boxes  = b_scale * (boxes_q.astype(np.int32) - b_zp)
            logits = c_scale * (class_q.astype(np.int32) - c_zp)
            hand_logits = logits[:, HAND_CLASS_INDEX]
            hand_scores = 1.0 / (1.0 + np.exp(-hand_logits))  # sigmoid for binary SSD

            sel0 = np.where(hand_scores >= SCORE_THR)[0]
            if sel0.size > 0:
                # Use top K before NMS
                order = np.argsort(hand_scores[sel0])[::-1]
                sel = sel0[order[:TOPK_PRE_NMS]]

                boxes_dec = decode_boxes(boxes[sel], anchors[sel])
                boxes_px = boxes_dec.copy()
                boxes_px[:, [0,2]] *= w0
                boxes_px[:, [1,3]] *= h0

                # Remove small boxes (noise suppression)
                areas = (boxes_px[:,2]-boxes_px[:,0]) * (boxes_px[:,3]-boxes_px[:,1])
                keep_sz = np.where(areas >= MIN_BOX_AREA)[0]
                boxes_px = boxes_px[keep_sz]
                scores_sel = hand_scores[sel][keep_sz]

                if boxes_px.shape[0] > 0:
                    keep = nms(boxes_px, scores_sel, iou_thr=IOU_THR, max_dets=MAX_DETS)
                    boxes_px = boxes_px[keep]
                    scores_k = scores_sel[keep]
                else:
                    boxes_px = np.empty((0,4), dtype=np.float32)
                    scores_k = np.empty((0,), dtype=np.float32)
            else:
                boxes_px = np.empty((0,4), dtype=np.float32)
                scores_k = np.empty((0,), dtype=np.float32)

            if HAS_CV2:
                vis = frame.copy()
                for (x1,y1,x2,y2), s in zip(boxes_px, scores_k):
                    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                    cv2.putText(vis, f"{s:.2f}", (int(x1), max(0,int(y1)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                fcnt += 1
                if fcnt % 10 == 0:
                    dt = time.time() - t0
                    fps = fcnt / dt if dt > 0 else 0.0
                    cv2.putText(vis, f"FPS:{fps:.1f} thr:{SCORE_THR} iou:{IOU_THR}",
                                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("hand-int8", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            else:
                pass
    finally:
        picam2.stop()
        if HAS_CV2:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
