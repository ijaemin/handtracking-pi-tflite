#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "hand_model_data.h"
#include "anchors_model_data.h"

// Input tensor dimensions
constexpr int IMG_W = 300;
constexpr int IMG_H = 300;

// ---- Minimal PPM loader (P6, 300x300 RGB, maxval=255) → int8(u8-128)
static bool LoadPPM300RGB(const char* path, int8_t* dst, size_t dst_bytes, int zp /*=-128*/) {
  FILE* f = std::fopen(path, "rb"); if (!f) return false;
  char magic[3] = {0};
  if (std::fscanf(f, "%2s", magic) != 1 || std::strcmp(magic, "P6")) { std::fclose(f); return false; }
  int w=0,h=0,maxv=0;
  if (std::fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) { std::fclose(f); return false; }
  if (w != 300 || h != 300 || maxv != 255) { std::fclose(f); return false; }
  std::fgetc(f); // consume one whitespace after header
  const size_t need = (size_t)w * h * 3;
  if (dst_bytes < need) { std::fclose(f); return false; }
  for (size_t i=0; i<need; ++i) {
    int c = std::fgetc(f); if (c == EOF) { std::fclose(f); return false; }
    dst[i] = (int8_t)((unsigned char)c + zp); // zp=-128 → int8 = u8 - 128
  }
  std::fclose(f); return true;
}

// --- NMS helpers (no STL, no dynamic alloc) ---
static inline float IoU_xyxy(const float a[4], const float b[4]) {
  float ax1=a[0], ay1=a[1], ax2=a[2], ay2=a[3];
  float bx1=b[0], by1=b[1], bx2=b[2], by2=b[3];
  float ix1 = ax1 > bx1 ? ax1 : bx1;
  float iy1 = ay1 > by1 ? ay1 : by1;
  float ix2 = ax2 < bx2 ? ax2 : bx2;
  float iy2 = ay2 < by2 ? ay2 : by2;
  float iw = ix2 - ix1; if (iw < 0) iw = 0;
  float ih = iy2 - iy1; if (ih < 0) ih = 0;
  float inter = iw * ih;
  float area_a = (ax2-ax1) * (ay2-ay1);
  float area_b = (bx2-bx1) * (by2-by1);
  float denom = area_a + area_b - inter;
  return denom > 0 ? (inter / (denom + 1e-6f)) : 0.0f;
}

// boxes: {x1,y1,x2,y2} * count (flattened), scores: length=count
// keep: 출력 인덱스 기록, 리턴값 = 보존 개수
static int NmsHard(const float* boxes, const float* scores, int count,
                   float iou_thr, int* keep, int max_keep) {
  // Simple score‑descending sort (insertion sort); count is usually ≤200
  int order[512];  // TOPK_PRE_NMS 최대값에 맞춰 충분히 크게
  if (count > (int)(sizeof(order)/sizeof(order[0])))
    count = (int)(sizeof(order)/sizeof(order[0]));
  for (int i=0;i<count;++i) order[i] = i;
  for (int i=1;i<count;++i) {
    int j=i; int oi=order[i];
    while (j>0 && scores[oi] > scores[order[j-1]]) {
      order[j] = order[j-1]; --j;
    }
    order[j] = oi;
  }

  int kept = 0;
  for (int idx=0; idx<count && kept < max_keep; ++idx) {
    int i = order[idx];
    const float* bi = &boxes[4*i];
    bool suppress = false;
    for (int k=0; k<kept; ++k) {
      int j = keep[k];
      const float* bj = &boxes[4*j];
      if (IoU_xyxy(bi, bj) > iou_thr) { suppress = true; break; }
    }
    if (!suppress) keep[kept++] = i;
  }
  return kept;
}

// --- Save PPM with drawn boxes (expects w=h=300, int8 input with zp=-128) ---
static inline void SetPx(uint8_t* img, int w, int h, int x, int y,
                         uint8_t R, uint8_t G, uint8_t B) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  size_t idx = (size_t)(y * w + x) * 3;
  img[idx+0] = R; img[idx+1] = G; img[idx+2] = B;
}
static void DrawRect(uint8_t* img, int w, int h, int x1, int y1, int x2, int y2,
                     int thick, uint8_t R, uint8_t G, uint8_t B) {
  if (x1 > x2) { int t=x1; x1=x2; x2=t; }
  if (y1 > y2) { int t=y1; y1=y2; y2=t; }
  for (int t=0; t<thick; ++t) {
    for (int x=x1; x<=x2; ++x) { SetPx(img,w,h,x,y1+t,R,G,B); SetPx(img,w,h,x,y2-t,R,G,B); }
    for (int y=y1; y<=y2; ++y) { SetPx(img,w,h,x1+t,y,R,G,B); SetPx(img,w,h,x2-t,y,R,G,B); }
  }
}
static void SavePPMWithBoxes(const char* out_path,
                             const int8_t* src_int8, int w, int h,
                             const float* boxes_norm, const int* keep, int kept) {
  // 1) int8( -128..127 ) → u8( 0..255 )
  static uint8_t img[IMG_W * IMG_H * 3];
  const int N = w*h*3;
  for (int i=0; i<N; ++i) img[i] = (uint8_t)((int)src_int8[i] + 128);

  // 2) Draw boxes (red, 2 px thick)
  for (int k=0; k<kept; ++k) {
    int r = keep[k];
    int x1 = (int)std::lround(boxes_norm[4*r+0] * (w-1));
    int y1 = (int)std::lround(boxes_norm[4*r+1] * (h-1));
    int x2 = (int)std::lround(boxes_norm[4*r+2] * (w-1));
    int y2 = (int)std::lround(boxes_norm[4*r+3] * (h-1));
    if (x1<0) x1=0; if (y1<0) y1=0; if (x2>w-1) x2=w-1; if (y2>h-1) y2=h-1;
    DrawRect(img, w, h, x1, y1, x2, y2, /*thick=*/2, /*R,G,B=*/255, 0, 0);
  }

  // 3) Save PPM (P6)
  FILE* f = std::fopen(out_path, "wb");
  if (!f) { std::perror("fopen"); return; }
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  std::fwrite(img, 1, (size_t)w*h*3, f);
  std::fclose(f);
}

// Save raw int8 input (zp=-128) to PPM without boxes
static void SavePPMRaw(const char* out_path, const int8_t* src_int8, int w, int h) {
  FILE* f = std::fopen(out_path, "wb");
  if (!f) { std::perror("fopen"); return; }
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  const size_t N = (size_t)w*h*3;
  for (size_t i=0; i<N; ++i) {
    uint8_t u = (uint8_t)((int)src_int8[i] + 128); // int8 → u8
    std::fputc(u, f);
  }
  std::fclose(f);
}

// --- Camera helpers ---
// BGR (OpenCV) → RGB (model input) 300×300, uint8→int8 (u8‑128)
static void FrameToInt8RGB300(const cv::Mat& bgr, int8_t* dst) {
  cv::Mat rgb, small;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, small, cv::Size(300, 300), 0, 0, cv::INTER_LINEAR);
  const uint8_t* p = small.ptr<uint8_t>(0);
  const size_t N = 300u * 300u * 3u;
  for (size_t i=0; i<N; ++i) dst[i] = (int8_t)((int)p[i] - 128);
}
static void RunCameraLoop(tflite::MicroInterpreter& interpreter,
                          TfLiteTensor* input,
                          float score_thr,
                          int pre_topk,
                          float nms_iou,
                          int post_topk) {
  cv::VideoCapture cap(
    "libcamerasrc ! video/x-raw,width=640,height=480,format=RGB ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink", 
    cv::CAP_GSTREAMER
  );
  if (!cap.isOpened()) { std::fprintf(stderr, "Camera open failed\n"); return; }
  cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  cv::Mat frame, vis;
  std::vector<int> keep_buf(512); // PRE_TOPK buffer size (sufficiently large)

  const TfLiteTensor* box_t = interpreter.output(0); // [1,1917,4] int8
  const TfLiteTensor* cls_t = interpreter.output(1); // [1,1917,2] int8
  const int N = box_t->dims->data[1];
  const int num_anchors = static_cast<int>(anchors_ssd300_len / 4);
  if (N != num_anchors) std::printf("Anchor count mismatch: %d vs %d\n", N, num_anchors);

  const float b_scale = box_t->params.scale; const int b_zp = box_t->params.zero_point;
  const float c_scale = cls_t->params.scale; const int c_zp = cls_t->params.zero_point;
  const int8_t* bq = box_t->data.int8;
  const int8_t* cq = cls_t->data.int8;

  // SSD-Lite scaling
  const float YS = 10.f, XS = 10.f, HS = 5.f, WS = 5.f;

  // Working buffers
  std::vector<float> boxes_buf; boxes_buf.resize(pre_topk * 4);
  std::vector<float> scores_buf; scores_buf.resize(pre_topk);
  std::vector<int>   cand_idx(pre_topk, -1);

  std::puts("Camera loop: ESC to exit");
  static int frame_count = 0;
  static auto last_log = std::chrono::high_resolution_clock::now();
  static int skip_counter = 0;   // frame skip counter
  while (true) {
    if (!cap.read(frame) || frame.empty()) continue;
    if (++skip_counter % 10 != 0) continue;
    vis = frame.clone();

  // 1) Preprocess frame → fill input buffer
  FrameToInt8RGB300(frame, input->data.int8);

  // 2) Run inference
  if (interpreter.Invoke() != kTfLiteOk) { std::fprintf(stderr, "Invoke failed\n"); continue; }

  // 3) Threshold-based Top-K selection (insertion sort)
    int cand_count = 0;
    for (int k=0;k<pre_topk;++k){ scores_buf[k] = -1e9f; cand_idx[k] = -1; }
    for (int i=0;i<N;++i) {
      const float hand_logit = c_scale * (static_cast<int>(cq[2*i+1]) - c_zp);
      const float score = 1.f / (1.f + std::exp(-hand_logit));
      if (score < score_thr) continue;
      if (cand_count < pre_topk || score > scores_buf[pre_topk-1]) {
        int limit = cand_count < pre_topk ? (cand_count+1) : pre_topk;
        int pos = limit-1; if (limit > pre_topk) limit = pre_topk;
        while (pos>0 && score > scores_buf[pos-1]) {
          if (pos < pre_topk) { scores_buf[pos] = scores_buf[pos-1]; cand_idx[pos] = cand_idx[pos-1]; }
          --pos;
        }
        scores_buf[pos] = score; cand_idx[pos] = i;
        if (cand_count < pre_topk) ++cand_count;
      }
    }

    int kept = 0;
    if (cand_count > 0) {
      // 4) Decode candidates
      for (int r=0; r<cand_count; ++r) {
        const int i = cand_idx[r];
        const float tx = b_scale * (static_cast<int>(bq[4*i+0]) - b_zp);
        const float ty = b_scale * (static_cast<int>(bq[4*i+1]) - b_zp);
        const float tw = b_scale * (static_cast<int>(bq[4*i+2]) - b_zp);
        const float th = b_scale * (static_cast<int>(bq[4*i+3]) - b_zp);
        const float ay = anchors_ssd300[4*i+0];
        const float ax = anchors_ssd300[4*i+1];
        const float ah = anchors_ssd300[4*i+2];
        const float aw = anchors_ssd300[4*i+3];
        const float ycenter = ty / YS * ah + ay;
        const float xcenter = tx / XS * aw + ax;
        const float h = std::exp(th / HS) * ah;
        const float w = std::exp(tw / WS) * aw;
        float x1 = xcenter - 0.5f * w;
        float y1 = ycenter - 0.5f * h;
        float x2 = xcenter + 0.5f * w;
        float y2 = ycenter + 0.5f * h;
        if (x1<0) x1=0; if (y1<0) y1=0; if (x2>1) x2=1; if (y2>1) y2=1;
        boxes_buf[4*r+0] = x1; boxes_buf[4*r+1] = y1; boxes_buf[4*r+2] = x2; boxes_buf[4*r+3] = y2;
      }
      // 5) Non-Max Suppression
      kept = NmsHard(boxes_buf.data(), scores_buf.data(), cand_count, nms_iou,
                     keep_buf.data(), post_topk);
    }

    // 6) Visualization (preserve original frame aspect ratio)
    const int W = frame.cols, H = frame.rows;
    for (int k=0; k<kept; ++k) {
      int r = keep_buf[k];
      int x1 = (int)std::lround(boxes_buf[4*r+0] * (W-1));
      int y1 = (int)std::lround(boxes_buf[4*r+1] * (H-1));
      int x2 = (int)std::lround(boxes_buf[4*r+2] * (W-1));
      int y2 = (int)std::lround(boxes_buf[4*r+3] * (H-1));
      cv::rectangle(vis, cv::Rect(cv::Point(x1,y1), cv::Point(x2,y2)), cv::Scalar(0,0,255), 2);
      char txt[64]; std::snprintf(txt, sizeof(txt), "S=%.2f", scores_buf[r]);
      cv::putText(vis, txt, cv::Point(x1, std::max(0,y1-6)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
    }

    cv::imshow("hand-cam", vis);
    int key = cv::waitKey(1);
    if (key == 27) break; // ESC
    using dsec = std::chrono::duration<double>;
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = dsec(now - last_log).count();
    if (elapsed >= 1.0) {
      std::printf("FPS: %.2f, Objects: %d\n", frame_count / elapsed, kept);
      frame_count = 0;
      last_log = now;
    }
    frame_count++;
  }
}

int main(int argc, char** argv) {
  tflite::InitializeTarget();

  // Load model from embedded byte array
  const tflite::Model* model = tflite::GetModel(hand_int8_io_noanchors_calib_tflite);
  std::printf("Model schema version: %d\n", model->version());

  // Register only the kernels the model needs
  tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddLogistic();

  // Static tensor arena (POSIX/macOS demo: roomy)
  constexpr size_t kArenaSize = 2 * 1024 * 1024 + 832 * 1024;
  static uint8_t tensor_arena[kArenaSize];

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::fprintf(stderr, "AllocateTensors failed (arena too small?)\n");
    return 2;
  }

  // Print IO shapes
  TfLiteTensor* input = interpreter.input(0);
  std::printf("Input: type=%d shape=[", input->type);
  for (int i = 0; i < input->dims->size; ++i) {
    std::printf("%d%s", input->dims->data[i], (i + 1 < input->dims->size) ? "," : "");
  }
  std::printf("]\n");
  for (int i = 0; i < interpreter.outputs_size(); ++i) {
    TfLiteTensor* out = interpreter.output(i);
    std::printf("Output%d: type=%d shape=[", i, out->type);
    for (int j = 0; j < out->dims->size; ++j) {
      std::printf("%d%s", out->dims->data[j], (j + 1 < out->dims->size) ? "," : "");
    }
    std::printf("]\n");
  }

// Enter camera loop in --cam mode and exit afterwards
if (argc > 1 && std::strcmp(argv[1], "--cam") == 0) {
  RunCameraLoop(interpreter, input, /*score_thr=*/0.62f, /*pre_topk=*/100, /*nms_iou=*/0.15f, /*post_topk=*/5);
  return 0;
}

// Load image if given: ./hand_demo image.ppm  (else zero-fill)
const char* img_path = (argc > 1) ? argv[1] : nullptr;
bool loaded = false;
if (input->type == kTfLiteInt8) {
  if (img_path) {
    loaded = LoadPPM300RGB(img_path, input->data.int8, input->bytes, -128);
    std::puts(loaded ? "Loaded PPM image" : "Failed to load PPM; using zeros");
  }
  if (!loaded) std::memset(input->data.int8, 0, input->bytes);
}
SavePPMRaw("./in.ppm", input->data.int8, 300, 300);
static int8_t input_copy[300*300*3];
std::memcpy(input_copy, input->data.int8, 300*300*3);
  if (interpreter.Invoke() != kTfLiteOk) {
    std::fprintf(stderr, "Invoke failed\n");
    return 3;
  }
  for (int i = 0; i < interpreter.outputs_size(); ++i) {
    TfLiteTensor* out = interpreter.output(i);
    std::printf("Out%d quant: scale=%g zp=%d bytes=%zu\n", i,
                out->params.scale, out->params.zero_point, out->bytes);
    int show = out->bytes < 8 ? out->bytes : 8;
    std::printf("Out%d first %d vals (int8):", i, show);
    for (int j = 0; j < show; ++j) std::printf(" %d", out->data.int8[j]);
    std::printf("\n");
  }

  // ---- Decode + NMS pipeline ----
  const TfLiteTensor* box_t = interpreter.output(0); // [1,1917,4] int8
  const TfLiteTensor* cls_t = interpreter.output(1); // [1,1917,2] int8
  const int N = box_t->dims->data[1];
  const int num_anchors = static_cast<int>(anchors_ssd300_len / 4);
  if (N != num_anchors) {
    std::printf("Anchor count mismatch: %d vs %d\n", N, num_anchors);
  }

  const float b_scale = box_t->params.scale;
  const int   b_zp    = box_t->params.zero_point;
  const float c_scale = cls_t->params.scale;
  const int   c_zp    = cls_t->params.zero_point;
  const int8_t* bq = box_t->data.int8;
  const int8_t* cq = cls_t->data.int8;

  // SSD‑Lite scaling (matching training settings)
  const float YS = 10.f, XS = 10.f, HS = 5.f, WS = 5.f;

  // ---- Tuning parameters ----
  const float SCORE_THR = 0.65f;   // Stable threshold observed on Raspberry Pi
  const int   PRE_TOPK  = 100;     // NMS candidate limit
  const float NMS_IOU   = 0.15f;   // IoU threshold
  const int   POST_TOPK = 5;      // Max outputs
  const int IMG_W = 300, IMG_H = 300;  // Input size

  // 0) Threshold‑based Top‑K selection (score descending, insertion sort)
  float cand_scores[PRE_TOPK]; int cand_idx[PRE_TOPK];
  for (int k=0;k<PRE_TOPK;++k){ cand_scores[k] = -1e9f; cand_idx[k] = -1; }

  int cand_count = 0;
  for (int i=0;i<N;++i) {
    // binary: class[1] = hand logit
    const float hand_logit = c_scale * (static_cast<int>(cq[2*i+1]) - c_zp);
    const float score = 1.f / (1.f + std::exp(-hand_logit));
    if (score < SCORE_THR) continue;

    if (cand_count < PRE_TOPK || score > cand_scores[PRE_TOPK-1]) {
      int limit = cand_count < PRE_TOPK ? (cand_count+1) : PRE_TOPK;
      int pos = limit-1;
      if (limit > PRE_TOPK) limit = PRE_TOPK;
      while (pos>0 && score > cand_scores[pos-1]) {
        if (pos < PRE_TOPK) { cand_scores[pos] = cand_scores[pos-1]; cand_idx[pos] = cand_idx[pos-1]; }
        --pos;
      }
      cand_scores[pos] = score; cand_idx[pos] = i;
      if (cand_count < PRE_TOPK) ++cand_count;
    }
  }

  if (cand_count == 0) {
    std::puts("NMS: no candidates above threshold.");
  } else {
    // 1) Decode selected candidates (normalized xyxy)
    float boxes[PRE_TOPK*4];
    float scores[PRE_TOPK];
    for (int r=0; r<cand_count; ++r) {
      const int i = cand_idx[r];
      scores[r] = cand_scores[r];

      // raw box: (tx, ty, tw, th)
      const float tx = b_scale * (static_cast<int>(bq[4*i+0]) - b_zp);
      const float ty = b_scale * (static_cast<int>(bq[4*i+1]) - b_zp);
      const float tw = b_scale * (static_cast<int>(bq[4*i+2]) - b_zp);
      const float th = b_scale * (static_cast<int>(bq[4*i+3]) - b_zp);

      const float ay = anchors_ssd300[4*i+0];
      const float ax = anchors_ssd300[4*i+1];
      const float ah = anchors_ssd300[4*i+2];
      const float aw = anchors_ssd300[4*i+3];

      const float ycenter = ty / YS * ah + ay;
      const float xcenter = tx / XS * aw + ax;
      const float h = std::exp(th / HS) * ah;
      const float w = std::exp(tw / WS) * aw;

      float x1 = xcenter - 0.5f * w;
      float y1 = ycenter - 0.5f * h;
      float x2 = xcenter + 0.5f * w;
      float y2 = ycenter + 0.5f * h;
      if (x1<0) x1=0; if (y1<0) y1=0; if (x2>1) x2=1; if (y2>1) y2=1;

      boxes[4*r+0] = x1; boxes[4*r+1] = y1; boxes[4*r+2] = x2; boxes[4*r+3] = y2;
    }

    // 2) Non‑Max Suppression
    int keep[PRE_TOPK];
    int kept = NmsHard(boxes, scores, cand_count, NMS_IOU, keep, POST_TOPK);

    // 3) Print results
    std::puts("NMS results (x1,y1,x2,y2, score) and [px]:");
    for (int k=0; k<kept; ++k) {
      int r = keep[k];
      // Convert to pixel coordinates (clamped to [0,W‑1]/[0,H‑1])
      int x1p = (int)std::lround(boxes[4*r+0] * (IMG_W - 1)); if (x1p < 0) x1p = 0; if (x1p > IMG_W-1) x1p = IMG_W-1;
      int y1p = (int)std::lround(boxes[4*r+1] * (IMG_H - 1)); if (y1p < 0) y1p = 0; if (y1p > IMG_H-1) y1p = IMG_H-1;
      int x2p = (int)std::lround(boxes[4*r+2] * (IMG_W - 1)); if (x2p < 0) x2p = 0; if (x2p > IMG_W-1) x2p = IMG_W-1;
      int y2p = (int)std::lround(boxes[4*r+3] * (IMG_H - 1)); if (y2p < 0) y2p = 0; if (y2p > IMG_H-1) y2p = IMG_H-1;

      std::printf("%d) %.3f, %.3f, %.3f, %.3f  score=%.3f  [px:%d,%d,%d,%d]\n",
                  k+1,
                  boxes[4*r+0], boxes[4*r+1], boxes[4*r+2], boxes[4*r+3],
                  scores[r],
                  x1p, y1p, x2p, y2p);
    }
    if (kept == 0) std::puts("(none)");
    SavePPMWithBoxes("./out.ppm", input_copy, IMG_W, IMG_H, boxes, keep, kept);
    std::puts("Saved: ./out.ppm");
  }

  std::puts("TFLM invoke OK");
  return 0;
}