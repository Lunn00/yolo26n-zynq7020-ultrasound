#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include "xil_types.h"

#define YOLO_INPUT_W 224
#define YOLO_INPUT_H 224
#define YOLO_ANCHORS 1029
/* YOLO26n has reg_max = 1: the head emits four DIRECT distances per anchor
   instead of yolov8n's 4 x 16 DFL bins, so there is no softmax expectation to
   compute -- just dist2bbox, exactly like the DFL branch but without the
   expectation.  This is the single biggest post-processing saving of the model
   change (4 distances per anchor instead of 64). */
#define YOLO_BOX_CHANNELS 4
#define YOLO_MAX_BOXES 128

typedef struct {
    float x1, y1, x2, y2;
    float score;
} YoloBox;

void yolo_preprocess_bgr(const u8 *bgr, int src_w, int src_h,
                         float *dst, int dst_w, int dst_h);

void yolo_preprocess_i8_bgr(const u8 *bgr, int src_w, int src_h,
                            u8 *dst, int dst_w, int dst_h);

void yolo_preprocess_i8_bgr_row(const u8 *bgr, int src_w, int src_h,
                                int src_row_bytes, u8 *dst,
                                int dst_w, int dst_h);

int yolo_decode_nms(const s16 *pred, int img_w, int img_h,
                    float conf_th, float iou_th,
                    YoloBox *boxes, int max_boxes);

float yolo_max_conf(const s16 *pred);

void yolo_map_boxes_to_display(YoloBox *boxes, int num_boxes,
                               int img_w, int img_h,
                               int fb_w, int fb_h);

/* Same letterbox mapping, but into an arbitrary sub-rectangle of the frame
 * buffer.  Must be called with the SAME scale/offset the image was blitted
 * with, otherwise the boxes drift relative to the picture. */
void yolo_map_boxes_to_rect(YoloBox *boxes, int num_boxes,
                            float scale, int pad_x, int pad_y);

void yolo_draw_boxes(u8 *fb, int fb_w, int fb_h,
                     const YoloBox *boxes, int num_boxes);

void yolo_draw_status_text(u8 *fb, int fb_w, int fb_h, int x, int y,
                           const char *text, u8 r, u8 g, u8 b, int scale);

/* Raw drawing helpers for the touch button bar (no background plate). */
void yolo_fill_rect(u8 *fb, int fb_w, int fb_h,
                    int x0, int y0, int w, int h, u8 r, u8 g, u8 b);

void yolo_draw_rect(u8 *fb, int fb_w, int fb_h,
                    int x0, int y0, int w, int h, int t,
                    u8 r, u8 g, u8 b);

void yolo_draw_text(u8 *fb, int fb_w, int fb_h, int x, int y,
                    const char *text, u8 r, u8 g, u8 b, int scale);

int  yolo_text_width(const char *text, int scale);

#endif
