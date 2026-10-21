/* YOLO26 depthwise convolution, executed on the PS.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS ON THE CPU AND NOT IN THE FABRIC
 * ---------------------------------------------------------------------------
 * The peak systolic engine's layer model is "every output channel consumes all
 * `ic` input channels".  A depthwise conv (ic=1, groups=oc) violates it, and the
 * dense inner loop cannot be adapted: it hoists one input sample per cycle and
 * shares it across 32 output channels, which is impossible when the input
 * channel depends on the output channel.
 *
 * A dedicated PL kernel WAS written and verified (see the git-history-style notes
 * in yolo_hls_i8_26.cpp).  It worked numerically but cost ~10,000 LUT, and the
 * Zynq-7020 has 53,200: synthesis came out at 108% with the placer reporting
 * WNS -0.473 ns, i.e. it does not fit and does not close.  The deployed yolov8n
 * build sits at 36,974 LUT / WNS +0.077 ns, so moving these six layers to the CPU
 * puts the fabric back on the profile that is already known to sign off.
 *
 * The price is small: 1,382,976 MACs of the network's 316 million (0.44%), and
 * the PL is invoked over layer RANGES anyway (it has to be, for the C2PSA
 * attention), so each depthwise layer just becomes one more range boundary.
 *
 * ---------------------------------------------------------------------------
 * EXACTNESS
 * ---------------------------------------------------------------------------
 * This is NOT an approximation of the hardware path, it IS the hardware path:
 * the same int8 weights, the same per-channel zero point, the same Q31
 * multiplier/shift requantisation and the same integer SiLU.  `requant_q31` and
 * `silu_q10` below are byte-for-byte ports of the HLS kernel's, so csim comparing
 * "PL + PS ops" against the integer reference still means what it says.
 */

#ifndef YOLO_DW_PS_H
#define YOLO_DW_PS_H

#include <stdint.h>

#include "yolo_layers_i8.h"
#include "yolo_fixed_params_i8.h"

#define YOLO_DW_WEIGHT_BASE 8

/* Ports of the HLS kernel's fixed-point helpers.  Keep in sync with
   yolo_hls_i8_26.cpp -- if these drift, the PS and PL stop agreeing. */
static int32_t yolo_ps_requant_q31(int32_t x, int32_t multiplier, int32_t rshift)
{
    const int64_t prod = (int64_t)x * (int64_t)multiplier;
    if (rshift <= 0) {
        const int lshift = -rshift;
        if (lshift >= 31)
            return x < 0 ? (int32_t)0x80000000u : 0x7fffffff;
        const int64_t v = prod << lshift;
        if (v > 0x7fffffffLL)
            return 0x7fffffff;
        if (v < -2147483648LL)
            return (int32_t)0x80000000u;
        return (int32_t)v;
    }
    if (rshift >= 63)
        return 0;
    const int64_t round = (int64_t)1 << (rshift - 1);
    if (prod >= 0)
        return (int32_t)((prod + round) >> rshift);
    return (int32_t)(-(((-prod) + round) >> rshift));
}

static int32_t yolo_ps_silu_q10(int32_t x)
{
    static const uint16_t sig_q15[17] = {
        11, 30, 81, 221, 589, 1554, 3906, 8813, 16384,
        23955, 28862, 31214, 32179, 32547, 32687, 32738, 32757
    };
    if (x <= -(8 << YOLO_I8_HEAD_FRAC_BITS))
        return (int32_t)(((int64_t)x * sig_q15[0]) >> 15);
    if (x >= (8 << YOLO_I8_HEAD_FRAC_BITS))
        return (int32_t)(((int64_t)x * sig_q15[16]) >> 15);
    const int integer = x >> YOLO_I8_HEAD_FRAC_BITS;
    const int frac = x - (integer << YOLO_I8_HEAD_FRAC_BITS);
    const int idx = integer + 8;
    const int32_t y0 = sig_q15[idx];
    const int32_t dy = (int32_t)sig_q15[idx + 1] - y0;
    const int32_t sigmoid = y0 +
        ((dy * frac + (1 << (YOLO_I8_HEAD_FRAC_BITS - 1))) >> YOLO_I8_HEAD_FRAC_BITS);
    const int64_t prod = (int64_t)x * sigmoid;
    return prod >= 0 ? (int32_t)((prod + (1 << 14)) >> 15)
                     : (int32_t)(-(((-prod) + (1 << 14)) >> 15));
}

/* True when the PL engine cannot represent this conv (grouped/depthwise). */
static int yolo_layer_is_depthwise(const YoloLayerI8 *L)
{
    return L->op == OP_CONV && (int)L->ic == 1 && (int)L->oc > 1 &&
           (int)L->nin == 1;
}

/* Run one depthwise conv entirely on the CPU, in place in `work`.
 * Returns 0 on success, -1 if the layer is not a supported depthwise conv. */
static int yolo_conv_dw_ps(const int8_t *weights, const uint8_t *fixed_params,
                           uint8_t *work, int idx)
{
    const YoloLayerI8 *L = &yolo_layers_i8[idx];
    const YoloFixedParamDesc *Q = &yolo_fixed_desc[idx];

    if (!yolo_layer_is_depthwise(L))
        return -1;
    if ((int)L->k != 3 || (int)L->stride != 1 || (int)L->pad != 1)
        return -1; /* every depthwise conv in YOLO26n-224 is 3x3 s1 p1 */

    const int oc = (int)L->oc;
    const int ih = (int)L->ih, iw = (int)L->iw;
    const int oh = (int)L->oh, ow = (int)L->ow;
    const int8_t *wbase = weights + YOLO_DW_WEIGHT_BASE + (int)L->wq_off;
    const int16_t *in_zp = (const int16_t *)(fixed_params + Q->in0_zp_off);
    const int16_t *out_zp = (const int16_t *)(fixed_params + Q->out_zp_off);
    const int32_t *bias = (const int32_t *)(fixed_params + Q->bias_off);
    const int32_t *mul0 = (const int32_t *)(fixed_params + Q->mul0_off);
    const int32_t *shift0 = (const int32_t *)(fixed_params + Q->shift0_off);
    const int32_t *mul1 = (const int32_t *)(fixed_params + Q->mul1_off);
    const int32_t *shift1 = (const int32_t *)(fixed_params + Q->shift1_off);

    for (int o = 0; o < oc; o++) {
        const uint8_t *src = work + (int)L->in_off[0] + o * ih * iw;
        uint8_t *dst = work + (int)L->out_off + o * oh * ow;
        const int8_t *w = wbase + o * 9;
        const int z = (int)in_zp[o];
        const int32_t bo = bias[o];
        const int32_t m0 = mul0[o], s0 = shift0[o];
        const int silu_on = (L->silu && !L->out_target) ? 1 : 0;
        const int32_t m1 = silu_on ? mul1[o] : 0;
        const int32_t s1 = silu_on ? shift1[o] : 0;
        const int32_t zo = L->out_target ? 0 : (int32_t)out_zp[o];

        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    const int iy = oy - 1 + ky;
                    if (iy < 0 || iy >= ih)
                        continue;
                    const uint8_t *row = src + iy * iw;
                    for (int kx = 0; kx < 3; kx++) {
                        const int ix = ox - 1 + kx;
                        if (ix < 0 || ix >= iw)
                            continue;
                        acc += (int32_t)w[ky * 3 + kx] * ((int32_t)row[ix] - z);
                    }
                }
                int32_t q = yolo_ps_requant_q31(acc + bo, m0, s0);
                if (L->out_target) {
                    if (q < -32768) q = -32768;
                    if (q > 32767) q = 32767;
                    /* int16 pred lives outside `work`; not used by any depthwise
                       layer in this network, so refuse rather than guess. */
                    return -2;
                }
                if (silu_on)
                    q = yolo_ps_requant_q31(yolo_ps_silu_q10(q), m1, s1);
                q += zo;
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                dst[oy * ow + ox] = (uint8_t)q;
            }
        }
    }
    return 0;
}

#endif /* YOLO_DW_PS_H */
