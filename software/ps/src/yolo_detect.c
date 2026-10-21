#include "yolo_detect.h"

#include "xil_printf.h"

#include <stdint.h>
#include <string.h>

static float yolo_expf(float x)
{
    const float LOG2E = 1.4426950408889634f;
    const float LN2 = 0.6931471805599453f;
    float a = x * LOG2E;
    int n = (int)a;
    if (a < (float)n)
        n--;
    if (n < -126)
        n = -126;
    if (n > 127)
        n = 127;
    float r = x - (float)n * LN2;
    float p =
        1.0f + r * (1.0f + r * (0.5f + r *
        (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f)))));
    union {
        float f;
        uint32_t u;
    } e;
    e.u = ((uint32_t)(n + 127)) << 23;
    return e.f * p;
}

static float yolo_sigmoid(float x)
{
    return 1.0f / (1.0f + yolo_expf(-x));
}

void yolo_preprocess_bgr(const u8 *bgr, int src_w, int src_h,
                         float *dst, int dst_w, int dst_h)
{
    const float scale_x = (float)dst_w / (float)src_w;
    const float scale_y = (float)dst_h / (float)src_h;
    const float scale = scale_x < scale_y ? scale_x : scale_y;
    const int new_w = (int)(src_w * scale + 0.5f);
    const int new_h = (int)(src_h * scale + 0.5f);
    const int pad_x = (dst_w - new_w) / 2;
    const int pad_y = (dst_h - new_h) / 2;
    const float pad_val = 114.0f / 255.0f;

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < dst_h; y++) {
            for (int x = 0; x < dst_w; x++) {
                float v = pad_val;
                const int ix = x - pad_x;
                const int iy = y - pad_y;
                if (ix >= 0 && iy >= 0 && ix < new_w && iy < new_h) {
                    const int sx = (int)(ix / scale);
                    const int sy = (int)(iy / scale);
                    if (sx < src_w && sy < src_h) {
                        const u8 *p = bgr + ((sy * src_w + sx) * 3);
                        v = p[2 - c] / 255.0f;
                    }
                }
                dst[(c * dst_h + y) * dst_w + x] = v;
            }
        }
    }
}

static void preprocess_i8_bgr_row(const u8 *bgr, int src_w, int src_h,
                                  int src_row_bytes, int pixel_bytes, u8 *dst,
                                  int dst_w, int dst_h)
{
    const float scale_x = (float)dst_w / (float)src_w;
    const float scale_y = (float)dst_h / (float)src_h;
    const float scale = scale_x < scale_y ? scale_x : scale_y;
    const int new_w = (int)(src_w * scale + 0.5f);
    const int new_h = (int)(src_h * scale + 0.5f);
    const int pad_x = (dst_w - new_w) / 2;
    const int pad_y = (dst_h - new_h) / 2;

    /* ----------------------------------------------------------------------
     * Print the geometry this call actually computes.
     *
     * The tensor on the board has mean 21.95 with min/max 0/238 and three
     * identical channels.  Rebuilding that tensor from the SD card's 1.BMP
     * (709x749, luma mean 22.50) reproduces those numbers ONLY with pad_x =
     * pad_y = 0:
     *
     *     M3X3 + letterbox 114  -> mean 27.24-27.34, max 229-235
     *     M3X3 + full stretch   -> mean 22.33-22.43, max 229-235
     *     board                 -> mean 21.95,     max 238
     *
     * A 114-padded tensor cannot average 21.95 unless its unpadded content
     * were 16.74, i.e. 25% darker than any reconstruction can produce.  So
     * either this function is not the one running, or it is not computing the
     * letterbox its source appears to describe.  These numbers distinguish
     * those two cases in a single line, and they are the first thing to check
     * if the tensor ever disagrees with the host again.
     * -------------------------------------------------------------------- */
    xil_printf("[LBX] src=%dx%d dst=%dx%d scale=%d/1000 nw=%d nh=%d pad=%d,%d str=%d\r\n",
               src_w, src_h, dst_w, dst_h, (int)(scale * 1000.0f + 0.5f),
               new_w, new_h, pad_x, pad_y, src_row_bytes);

    /* Matches Python qact(x, 1/255, 0): UINT8 input with zero point 0. */
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < dst_h; y++) {
            for (int x = 0; x < dst_w; x++) {
                int qv = 114; /* 114/255 letterbox padding */
                const int ix = x - pad_x;
                const int iy = y - pad_y;
                if (ix >= 0 && iy >= 0 && ix < new_w && iy < new_h) {
                    const int sx = (int)(ix / scale);
                    const int sy = (int)(iy / scale);
                    if (sx < src_w && sy < src_h) {
                        /* 源缓冲是 BGRX（4 字节/像素，见 main.c 的 IMG_PIXEL_BYTES），
                           前三字节才是 BGR，所以步长是 4 而不是 3。 */
                        const u8 *p = bgr + sy * src_row_bytes + sx * pixel_bytes;
                        qv = (int)(p[2 - c] + 0.5f);
                    }
                }
                if (qv < 0)
                    qv = 0;
                if (qv > 255)
                    qv = 255;
                dst[(c * dst_h + y) * dst_w + x] = (u8)qv;
            }
        }
    }
}

void yolo_preprocess_i8_bgr(const u8 *bgr, int src_w, int src_h,
                            u8 *dst, int dst_w, int dst_h)
{
    preprocess_i8_bgr_row(bgr, src_w, src_h, src_w * 3, 3, dst, dst_w, dst_h);
}

void yolo_preprocess_i8_bgr_row(const u8 *bgr, int src_w, int src_h,
                                int src_row_bytes, u8 *dst,
                                int dst_w, int dst_h)
{
    preprocess_i8_bgr_row(bgr, src_w, src_h, src_row_bytes, 4, dst, dst_w, dst_h);
}

static void anchor_info(int idx, int *stride, float *cx, float *cy)
{
    int start;
    if (idx < 784) {
        *stride = 8;
        start = 0;
    } else if (idx < 980) {
        *stride = 16;
        start = 784;
    } else {
        *stride = 32;
        start = 980;
    }
    const int grid = YOLO_INPUT_W / *stride;
    const int rem = idx - start;
    *cx = (float)(rem % grid) + 0.5f;
    *cy = (float)(rem / grid) + 0.5f;
}

/* YOLO26: no DFL.
 *
 * The engine writes four raw distances (l, t, r, b) in grid units, one per box
 * channel, so "decoding" a side is just a scale change from the hardware's
 * signed Q10.  This replaces yolov8n's dfl_dist(), which ran a 16-bin softmax
 * expectation per side -- 4 x 16 exponentials plus a max scan per candidate
 * anchor.  With reg_max = 1 the head emits the distances directly. */
static float raw_dist(const s16 *box, int side, int anchor)
{
    return (float)box[side * YOLO_ANCHORS + anchor] / 1024.0f;
}

static float iou(const YoloBox *a, const YoloBox *b)
{
    const float ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
    const float iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    const float ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
    const float iy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f)
        return 0.0f;
    const float inter = iw * ih;
    const float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    const float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    return inter / (area_a + area_b - inter);
}

int yolo_decode_nms(const s16 *pred, int img_w, int img_h,
                    float conf_th, float iou_th,
                    YoloBox *boxes, int max_boxes)
{
    const s16 *box = pred;
    const s16 *conf = pred + YOLO_BOX_CHANNELS * YOLO_ANCHORS;
    const float scale_x = (float)YOLO_INPUT_W / (float)img_w;
    const float scale_y = (float)YOLO_INPUT_H / (float)img_h;
    const float scale = scale_x < scale_y ? scale_x : scale_y;
    const int new_w = (int)(img_w * scale + 0.5f);
    const int new_h = (int)(img_h * scale + 0.5f);
    const float pad_x = ((float)YOLO_INPUT_W - (float)new_w) * 0.5f;
    const float pad_y = ((float)YOLO_INPUT_H - (float)new_h) * 0.5f;

    YoloBox cand[YOLO_MAX_BOXES];
    int n = 0;

    for (int a = 0; a < YOLO_ANCHORS; a++) {
        /* The verified HLS output contains no exact-zero confidence logits.
           Reject zeros explicitly: sigmoid(0)=0.5 would otherwise turn an
           invalid/unwritten entry into a high-priority false candidate. */
        if (conf[a] == 0)
            continue;
        const float score = yolo_sigmoid((float)conf[a] / 1024.0f);
        if (score < conf_th)
            continue;

        int stride;
        float cx, cy;
        anchor_info(a, &stride, &cx, &cy);

        const float dl = raw_dist(box, 0, a);
        const float dt = raw_dist(box, 1, a);
        const float dr = raw_dist(box, 2, a);
        const float db = raw_dist(box, 3, a);

        float x1 = (cx - dl) * (float)stride;
        float y1 = (cy - dt) * (float)stride;
        float x2 = (cx + dr) * (float)stride;
        float y2 = (cy + db) * (float)stride;

        x1 = (x1 - pad_x) / scale;
        y1 = (y1 - pad_y) / scale;
        x2 = (x2 - pad_x) / scale;
        y2 = (y2 - pad_y) / scale;
        if (x1 < 0.0f) x1 = 0.0f;
        if (y1 < 0.0f) y1 = 0.0f;
        if (x2 > (float)img_w) x2 = (float)img_w;
        if (y2 > (float)img_h) y2 = (float)img_h;
        if (x2 <= x1 || y2 <= y1)
            continue;

        YoloBox next;
        next.x1 = x1;
        next.y1 = y1;
        next.x2 = x2;
        next.y2 = y2;
        next.score = score;
        if (n < YOLO_MAX_BOXES) {
            cand[n++] = next;
        } else {
            int worst = 0;
            for (int i = 1; i < n; i++) {
                if (cand[i].score < cand[worst].score)
                    worst = i;
            }
            if (next.score > cand[worst].score)
                cand[worst] = next;
        }
    }

    /* Sort by score descending. */
    for (int i = 0; i < n - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (cand[j].score > cand[best].score)
                best = j;
        }
        if (best != i) {
            YoloBox t = cand[i];
            cand[i] = cand[best];
            cand[best] = t;
        }
    }

    int out_n = 0;
    for (int i = 0; i < n && out_n < max_boxes; i++) {
        int keep = 1;
        for (int j = 0; j < out_n; j++) {
            if (iou(&cand[i], &boxes[j]) > iou_th) {
                keep = 0;
                break;
            }
        }
        if (keep)
            boxes[out_n++] = cand[i];
    }
    return out_n;
}

float yolo_max_conf(const s16 *pred)
{
    const s16 *conf = pred + YOLO_BOX_CHANNELS * YOLO_ANCHORS;
    s16 best = conf[0];
    for (int a = 1; a < YOLO_ANCHORS; a++) {
        if (conf[a] > best)
            best = conf[a];
    }
    return yolo_sigmoid((float)best / 1024.0f);
}

void yolo_map_boxes_to_display(YoloBox *boxes, int num_boxes,
                               int img_w, int img_h,
                               int fb_w, int fb_h)
{
    if (img_w <= 0 || img_h <= 0)
        return;
    const float scale =
        (float)fb_w / img_w < (float)fb_h / img_h
            ? (float)fb_w / img_w : (float)fb_h / img_h;
    const int nw = (int)(img_w * scale + 0.5f);
    const int nh = (int)(img_h * scale + 0.5f);
    const float pad_x = (float)((fb_w - nw) / 2);
    const float pad_y = (float)((fb_h - nh) / 2);
    for (int i = 0; i < num_boxes; i++) {
        boxes[i].x1 = boxes[i].x1 * scale + pad_x;
        boxes[i].y1 = boxes[i].y1 * scale + pad_y;
        boxes[i].x2 = boxes[i].x2 * scale + pad_x;
        boxes[i].y2 = boxes[i].y2 * scale + pad_y;
    }
}

/* Explicit scale/offset form: the caller letterboxed the image into a panel
 * (not the whole frame), so it passes back exactly the transform it used. */
void yolo_map_boxes_to_rect(YoloBox *boxes, int num_boxes,
                            float scale, int pad_x, int pad_y)
{
    for (int i = 0; i < num_boxes; i++) {
        boxes[i].x1 = boxes[i].x1 * scale + (float)pad_x;
        boxes[i].y1 = boxes[i].y1 * scale + (float)pad_y;
        boxes[i].x2 = boxes[i].x2 * scale + (float)pad_x;
        boxes[i].y2 = boxes[i].y2 * scale + (float)pad_y;
    }
}

static const char font_chars[] = " -." "0123456789" ":="
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ%";

/* 5x7 bitmap font, one byte per row, bit 4 = leftmost pixel. */
static const u8 font5x7[][7] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, /* space */
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, /* - */
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, /* . */
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, /* 0 */
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, /* 1 */
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, /* 2 */
    {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}, /* 3 */
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, /* 4 */
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, /* 5 */
    {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E}, /* 6 */
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, /* 7 */
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, /* 8 */
    {0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E}, /* 9 */
    {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}, /* : */
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, /* = */
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, /* A */
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, /* B */
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, /* C */
    {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}, /* D */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, /* E */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, /* F */
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, /* G */
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, /* H */
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, /* I */
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, /* J */
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, /* K */
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, /* L */
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, /* M */
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, /* N */
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, /* O */
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, /* P */
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, /* Q */
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, /* R */
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, /* S */
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, /* T */
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, /* U */
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, /* V */
    {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}, /* W */
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, /* X */
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, /* Y */
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, /* Z */
    {0x19,0x19,0x02,0x04,0x08,0x13,0x13}, /* % */
};

static void fill_rect(u8 *fb, int fb_w, int fb_h,
                      int x0, int y0, int w, int h,
                      u8 r, u8 g, u8 b)
{
    for (int y = y0; y < y0 + h && y < fb_h; y++) {
        if (y < 0)
            continue;
        for (int x = x0; x < x0 + w && x < fb_w; x++) {
            if (x < 0)
                continue;
            u8 *p = fb + ((u32)y * fb_w + x) * 3;
            p[0] = b;
            p[1] = g;
            p[2] = r;
        }
    }
}

static void draw_char(u8 *fb, int fb_w, int fb_h, int x, int y,
                      char c, u8 r, u8 g, u8 b, int scale)
{
    const char *idx = strchr(font_chars, c);
    if (!idx)
        return;
    const u8 *glyph = font5x7[idx - font_chars];
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (!(glyph[row] & (1 << (4 - col))))
                continue;
            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {
                    const int px = x + col * scale + dx;
                    const int py = y + row * scale + dy;
                    if (px < 0 || py < 0 || px >= fb_w || py >= fb_h)
                        continue;
                    u8 *p = fb + ((u32)py * fb_w + px) * 3;
                    p[0] = b;
                    p[1] = g;
                    p[2] = r;
                }
            }
        }
    }
}

/* ---- exported drawing helpers used by the on-screen button bar ---------- */

void yolo_fill_rect(u8 *fb, int fb_w, int fb_h,
                    int x0, int y0, int w, int h, u8 r, u8 g, u8 b)
{
    fill_rect(fb, fb_w, fb_h, x0, y0, w, h, r, g, b);
}

void yolo_draw_rect(u8 *fb, int fb_w, int fb_h,
                    int x0, int y0, int w, int h, int t,
                    u8 r, u8 g, u8 b)
{
    fill_rect(fb, fb_w, fb_h, x0, y0, w, t, r, g, b);
    fill_rect(fb, fb_w, fb_h, x0, y0 + h - t, w, t, r, g, b);
    fill_rect(fb, fb_w, fb_h, x0, y0, t, h, r, g, b);
    fill_rect(fb, fb_w, fb_h, x0 + w - t, y0, t, h, r, g, b);
}

/* Same 5x7 font as yolo_draw_status_text but with no background plate, so it
 * can be stamped on top of a coloured button. */
void yolo_draw_text(u8 *fb, int fb_w, int fb_h, int x, int y,
                    const char *text, u8 r, u8 g, u8 b, int scale)
{
    const int len = (int)strlen(text);
    for (int i = 0; i < len; i++)
        draw_char(fb, fb_w, fb_h, x + i * 6 * scale, y, text[i], r, g, b, scale);
}

/* Pixel width / height of a string drawn by yolo_draw_text. */
int yolo_text_width(const char *text, int scale)
{
    return (int)strlen(text) * 6 * scale - scale;
}

void yolo_draw_status_text(u8 *fb, int fb_w, int fb_h, int x, int y,
                           const char *text, u8 r, u8 g, u8 b, int scale)
{
    const int len = (int)strlen(text);
    const int tw = len * 6 * scale;
    const int th = 7 * scale;
    fill_rect(fb, fb_w, fb_h, x - 4, y - 4, tw + 8, th + 8, 24, 24, 24);
    for (int i = 0; i < len; i++)
        draw_char(fb, fb_w, fb_h, x + i * 6 * scale, y,
                  text[i], r, g, b, scale);
}

void yolo_draw_boxes(u8 *fb, int fb_w, int fb_h,
                     const YoloBox *boxes, int num_boxes)
{
    for (int b = 0; b < num_boxes; b++) {
        const int x1 = (int)(boxes[b].x1 + 0.5f);
        const int y1 = (int)(boxes[b].y1 + 0.5f);
        const int x2 = (int)(boxes[b].x2 + 0.5f);
        const int y2 = (int)(boxes[b].y2 + 0.5f);
        for (int t = 0; t < 3; t++) {
            for (int x = x1 - t; x <= x2 + t; x++) {
                if (x < 0 || x >= fb_w)
                    continue;
                if (y1 - t >= 0 && y1 - t < fb_h) {
                    u8 *p = fb + ((y1 - t) * fb_w + x) * 3;
                    p[0] = 0;
                    p[1] = 255;
                    p[2] = 0;
                }
                if (y2 + t >= 0 && y2 + t < fb_h) {
                    u8 *p = fb + ((y2 + t) * fb_w + x) * 3;
                    p[0] = 0;
                    p[1] = 255;
                    p[2] = 0;
                }
            }
            for (int y = y1 - t; y <= y2 + t; y++) {
                if (y < 0 || y >= fb_h)
                    continue;
                if (x1 - t >= 0 && x1 - t < fb_w) {
                    u8 *p = fb + ((y * fb_w) + x1 - t) * 3;
                    p[0] = 0;
                    p[1] = 255;
                    p[2] = 0;
                }
                if (x2 + t >= 0 && x2 + t < fb_w) {
                    u8 *p = fb + ((y * fb_w) + x2 + t) * 3;
                    p[0] = 0;
                    p[1] = 255;
                    p[2] = 0;
                }
            }
        }
    }
}
