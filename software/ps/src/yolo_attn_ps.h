/* YOLO26 C2PSA attention, executed on the PS.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS ON THE CPU
 * ---------------------------------------------------------------------------
 * `/attn/` needs: two tensor views (q and k are slices of one qkv conv output),
 * a 49x49 batched matmul, a softmax, a second matmul, a DEPTHWISE 3x3 `pe`
 * convolution and a residual add.  Two of those (the views and the depthwise
 * conv) are outside the PL engine's layer model, and the whole block is only
 * 921,984 MACs -- 0.036% of the network.  Adding them to the PL would have
 * meant re-cutting the dense kernel's inner loop, whose structure is what keeps
 * WNS positive (+0.077 ns on the deployed yolov8n build).  So the PL runs the
 * layer RANGES around the attention and the PS fills in the two slots.
 *
 * ---------------------------------------------------------------------------
 * THE ALGORITHM (ultralytics Attention.forward, verified against ONNX to 2e-7
 * relative error by scripts/verify_attention_unit.py)
 * ---------------------------------------------------------------------------
 *     qkv  = qkv_conv(x)                       [C=256, H*W=49]
 *     view -> [num_heads=2, 128, 49]
 *     split(128) -> q [32], k [32], v [64]   (per head)
 *     attn = softmax(q^T k / sqrt(32))         [49 x 49]
 *     y    = (v @ attn^T)                      [64, 49]
 *     out  = y + pe(v)                         <- pe takes V, not y!
 *     then the proj conv (PL) consumes `out`.
 *
 * The `pe` input is the single easiest thing to get wrong here: feeding the
 * attention RESULT to pe instead of v produces shapes that all fit, plausible
 * numbers, and a max error of ~88% of the signal.  It cost a full debug cycle.
 *
 * ---------------------------------------------------------------------------
 * DATA SOURCES (all offsets come straight out of the generated layer row)
 * ---------------------------------------------------------------------------
 *   work + L.in_off[0]      int8 qkv activations (stored as uint8 codes)
 *   act_scales + in_ascale_off[0]   float scale[ic], int16 zero[ic]
 *   act_scales + ascale_off         float scale[oc], int16 zero[oc]  (output)
 *   weights + 8 + wq_off            FP32 pe weight (oc,1,3,3)
 *   weights + 8 + bias_off          FP32 pe bias   (oc,)
 *   L.c0 = num_heads, L.c1 = key_dim, L.oc = num_heads*head_dim
 */

#ifndef YOLO_ATTN_PS_H
#define YOLO_ATTN_PS_H

#include <stdint.h>

#include "yolo_layers_i8.h"

/* ---------------------------------------------------------------------------
 * NO libm.  The standalone BSP this application links against does not pull in
 * libm, so `expf`/`sqrtf` fail at link time (undefined reference).  The yolov8n
 * application already hand-rolls `yolo_expf` for exactly this reason; these are
 * the same idea, kept local to this header so it stays self-contained.
 *
 * Accuracy: the exponential is a range-reduced degree-5 Taylor series with an
 * exact power-of-two scale, good to ~1e-7 relative in float -- far below the
 * int8 quantisation step that follows, and it matched the host reference in the
 * HLS testbench (which runs the same code).
 * ------------------------------------------------------------------------- */
static float yolo_attn_expf(float x)
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
    const float r = x - (float)n * LN2;
    const float p =
        1.0f + r * (1.0f + r * (0.5f + r *
        (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f)))));
    union {
        float f;
        uint32_t u;
    } e;
    e.u = ((uint32_t)(n + 127)) << 23;
    return e.f * p;
}

static float yolo_attn_sqrtf(float x)
{
    if (x <= 0.0f)
        return 0.0f;
    /* Newton iterations from a bit-trick seed; 4 rounds is ~1e-7 in float. */
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    v.u = (v.u >> 1) + 0x1fbd1df5u;
    float y = v.f;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}

#define YOLO_ATTN_MAX_IC 256
#define YOLO_ATTN_MAX_OC 256
#define YOLO_ATTN_MAX_N 64
#define YOLO_ATTN_MAX_K 9
#define YOLO_ATTN_WEIGHT_BASE 8

typedef struct {
    int ok;
    int num_heads;
    int key_dim;
    int head_dim;
    int ic;
    int oc;
    int n;
    int tokens_max;   /* max |out| over the layer */
    int clipped;      /* saturations at 0/255 */
} YoloAttnStat;

/* Scratch: ~85 KB of BSS, i.e. DDR on the PS.  Kept out of the stack. */
static float yolo_attn_x[YOLO_ATTN_MAX_IC * YOLO_ATTN_MAX_N];
static float yolo_attn_y[YOLO_ATTN_MAX_OC * YOLO_ATTN_MAX_N];
static float yolo_attn_a[YOLO_ATTN_MAX_N * YOLO_ATTN_MAX_N];
static float yolo_attn_v[YOLO_ATTN_MAX_OC * YOLO_ATTN_MAX_N];

static int yolo_attn_ps(const int8_t *weights, const uint8_t *act_scales,
                        uint8_t *work, const YoloLayerI8 *L, YoloAttnStat *st)
{
    const int nh = (int)L->c0;
    const int kd = (int)L->c1;
    const int ic = (int)L->ic;
    const int oc = (int)L->oc;
    const int oh = (int)L->oh;
    const int ow = (int)L->ow;
    const int n = oh * ow;
    const int hd = (nh > 0) ? oc / nh : 0;
    const int per_head = 2 * kd + hd;
    const int k = (int)L->k;

    if (st) {
        st->ok = 0;
        st->num_heads = nh;
        st->key_dim = kd;
        st->head_dim = hd;
        st->ic = ic;
        st->oc = oc;
        st->n = n;
        st->tokens_max = 0;
        st->clipped = 0;
    }
    if (L->op != OP_ATTENTION)
        return -1;
    if (nh <= 0 || kd <= 0 || hd <= 0 || nh * per_head != ic || nh * hd != oc)
        return -1;
    if (n > YOLO_ATTN_MAX_N || ic > YOLO_ATTN_MAX_IC || oc > YOLO_ATTN_MAX_OC)
        return -1;
    if (k != 3 || k * k > YOLO_ATTN_MAX_K)
        return -1;

    const float *in_s = (const float *)(act_scales + L->in_ascale_off[0]);
    const int16_t *in_z =
        (const int16_t *)(act_scales + L->in_ascale_off[0] + 4 * ic);
    const float *out_s = (const float *)(act_scales + L->ascale_off);
    const int16_t *out_z =
        (const int16_t *)(act_scales + L->ascale_off + 4 * oc);
    const float *pe_w =
        (const float *)(weights + YOLO_ATTN_WEIGHT_BASE + (int)L->wq_off);
    const float *pe_b =
        (const float *)(weights + YOLO_ATTN_WEIGHT_BASE + (int)L->bias_off);
    const uint8_t *src = work + (int)L->in_off[0];
    uint8_t *dst = work + (int)L->out_off;

    /* ---- dequantize qkv: x[c][i] = (code - zero) * scale ------------------ */
    for (int c = 0; c < ic; c++) {
        const float s = in_s[c];
        const float z = (float)in_z[c];
        const uint8_t *p = src + c * n;
        float *d = yolo_attn_x + c * n;
        for (int i = 0; i < n; i++)
            d[i] = ((float)p[i] - z) * s;
    }

    /* ---- per head: attn = softmax(q^T k * 1/sqrt(kd)); y = v @ attn^T ----- */
    const float softmax_scale = 1.0f / yolo_attn_sqrtf((float)kd);
    for (int h = 0; h < nh; h++) {
        const float *q = yolo_attn_x + (h * per_head) * n;
        const float *kk = yolo_attn_x + (h * per_head + kd) * n;
        const float *vv = yolo_attn_x + (h * per_head + 2 * kd) * n;
        float *vy = yolo_attn_y + h * hd * n;

        for (int i = 0; i < n; i++) {
            float *row = yolo_attn_a + i * n;
            for (int j = 0; j < n; j++) {
                float acc = 0.0f;
                for (int c = 0; c < kd; c++)
                    acc += q[c * n + i] * kk[c * n + j];
                row[j] = acc * softmax_scale;
            }
            /* softmax over j, max-subtracted for stability */
            float m = row[0];
            for (int j = 1; j < n; j++)
                if (row[j] > m)
                    m = row[j];
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                row[j] = yolo_attn_expf(row[j] - m);
                sum += row[j];
            }
            const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (int j = 0; j < n; j++)
                row[j] *= inv;
        }

        for (int c = 0; c < hd; c++) {
            const float *vc = vv + c * n;
            float *yc = vy + c * n;
            for (int i = 0; i < n; i++) {
                const float *row = yolo_attn_a + i * n;
                float acc = 0.0f;
                for (int j = 0; j < n; j++)
                    acc += vc[j] * row[j];
                yc[i] = acc;
            }
        }
    }

    /* ---- v reshaped to [oc][oh][ow], needed for pe ------------------------ */
    for (int h = 0; h < nh; h++) {
        const float *vv = yolo_attn_x + (h * per_head + 2 * kd) * n;
        for (int c = 0; c < hd; c++) {
            const float *s = vv + c * n;
            float *d = yolo_attn_v + (h * hd + c) * n;
            for (int i = 0; i < n; i++)
                d[i] = s[i];
        }
    }

    /* ---- out = y + pe(v), pe = depthwise 3x3 pad 1 over v ----------------- */
    const int pad = 1;
    int clipped = 0;
    float tokens_max = 0.0f;
    for (int c = 0; c < oc; c++) {
        const float *vc = yolo_attn_v + c * n;
        const float *yc = yolo_attn_y + c * n;
        const float *wc = pe_w + c * k * k;
        const float bc = pe_b[c];
        const float s = out_s[c];
        const float z = (float)out_z[c];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float acc = 0.0f;
                for (int ky = 0; ky < k; ky++) {
                    const int iy = oy - pad + ky;
                    if (iy < 0 || iy >= oh)
                        continue;
                    for (int kx = 0; kx < k; kx++) {
                        const int ix = ox - pad + kx;
                        if (ix < 0 || ix >= ow)
                            continue;
                        acc += wc[ky * k + kx] * vc[iy * ow + ix];
                    }
                }
                const float real = yc[oy * ow + ox] + acc + bc;
                float q = real / s + z;
                q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
                int qi = (int)q;
                if (qi < 0) {
                    qi = 0;
                    clipped++;
                }
                if (qi > 255) {
                    qi = 255;
                    clipped++;
                }
                /* The output channel index is NOT optional.
                 *
                 * This line used to read `dst[oy * ow + ox]`, so all 128 output
                 * channels wrote the same 49-byte slot: only 49 of the layer's
                 * 6272 output bytes were ever produced and the rest kept whatever
                 * the work buffer happened to hold (zeros after the pre-run
                 * memset).  csim still passed -- its acceptance bar only looked at
                 * ~10 active anchors -- and so did the board's self-test, which is
                 * exactly how a live bug survives a green test suite.  Measured:
                 * 49/6272 bytes written before, 6235/6272 after; the whole-tensor
                 * error drops from 7.61 to 1.81 (Q10). */
                dst[c * n + oy * ow + ox] = (uint8_t)qi;
                if (real > tokens_max)
                    tokens_max = real;
                else if (-real > tokens_max)
                    tokens_max = -real;
            }
        }
    }

    if (st) {
        st->ok = 1;
        st->clipped = clipped;
        st->tokens_max = (int)tokens_max;
    }
    return 0;
}

#endif /* YOLO_ATTN_PS_H */
