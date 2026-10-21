/* ===========================================================================
 *  filter_sw.c —— 斑点抑制 / 增强的 PS 软件实现（多档）
 *  各档语义见 filter_sw.h。
 *
 *  实现方式：两块灰度平面 A/B 交替做中间结果，最后展开回 BGRX。
 *      MED3    : lum -> A; med3 A->B;                      expand B
 *      MED3x2  : lum -> A; med3 A->B; med3 B->A;           expand A
 *      MED3x3  : 再多跑一遍
 *      GAUSS5  : lum -> A; 水平 A->B; 垂直 B->A;            expand A
 *  每趟只读写一块平面，边界都是**边缘复制 clamp**（与 PL 版 IP 一致）。
 * ===========================================================================
 */

#include <stddef.h>
#include "filter_sw.h"

/* 两块灰度中间平面，各 4 MB，放 BSS（板端在 DDR 里）。
 * 用两块而不是一块，是为了让"读一块、写另一块"显然安全，不必去推就地语义。
 * crt0 启动时清零这 8 MB，约十几毫秒，可忽略。 */
static uint8_t s_planeA[FILT_SW_MAX_W * FILT_SW_MAX_H];
static uint8_t s_planeB[FILT_SW_MAX_W * FILT_SW_MAX_H];

uint8_t *filter_sw_plane_a(void) { return s_planeA; }
uint8_t *filter_sw_plane_b(void) { return s_planeB; }

/* ---------------------------------------------------------------- 模式名字 */

const char *filter_sw_mode_name(int mode)
{
    switch (mode) {
    case FILT_MODE_OFF:    return "OFF";
    case FILT_MODE_MED3:   return "MED3";
    case FILT_MODE_MED3X2: return "M3X2";
    case FILT_MODE_MED3X3: return "M3X3";
    case FILT_MODE_GAUSS5: return "GAUSS";
    case FILT_MODE_GRAY: return "GRAY";
    case FILT_MODE_RGBMED: return "RGBM3";
    case FILT_MODE_RGBGAUSS: return "RGBG5";
    default:               return "?";
    }
}

const char *filter_sw_mode_desc(int mode)
{
    switch (mode) {
    case FILT_MODE_OFF:    return "raw BGR, no filter (= \"original\")";
    case FILT_MODE_MED3:   return "luma + 3x3 median (= \"med3\")";
    case FILT_MODE_MED3X2: return "luma + 3x3 median x2";
    case FILT_MODE_MED3X3: return "luma + 3x3 median x3";
    case FILT_MODE_GAUSS5: return "luma + 5x5 gaussian sigma=1.2";
    case FILT_MODE_GRAY: return "luma only, no smoothing";
    case FILT_MODE_RGBMED: return "BGR channelwise 3x3 median";
    case FILT_MODE_RGBGAUSS: return "BGR channelwise 5x5 gaussian";
    default:               return "unknown";
    }
}

int filter_sw_mode_next(int mode)
{
    if (mode < 0 || mode >= FILT_MODE_COUNT - 1)
        return 0;
    return mode + 1;
}

/* ---------------------------------------------------------------- 第一步：亮度 */

#define FSW_YMUL   15
#define FSW_RCOEF  9798u
#define FSW_GCOEF  19235u
#define FSW_BCOEF  3735u

/* BGRX -> 亮度（与 PL 版 filt_lum 完全一致） */
static inline uint8_t fsw_lum(uint32_t px)
{
    const uint32_t b =  px        & 0xFFu;
    const uint32_t g = (px >>  8) & 0xFFu;
    const uint32_t r = (px >> 16) & 0xFFu;
    const uint32_t y = r * FSW_RCOEF + g * FSW_GCOEF + b * FSW_BCOEF
                       + (1u << (FSW_YMUL - 1));
    return (uint8_t)(y >> FSW_YMUL);
}

static void fsw_lum_pass(const uint8_t *src, uint8_t *plane,
                         int w, int h, int stride)
{
    for (int y = 0; y < h; y++) {
        const uint8_t *s = src + (size_t)y * stride;
        uint8_t *g = plane + (size_t)y * w;
        for (int x = 0; x < w; x++, s += 4)
            g[x] = fsw_lum((uint32_t)s[0] | ((uint32_t)s[1] << 8) |
                           ((uint32_t)s[2] << 16));
    }
}

/* ---------------------------------------------------------------- 3x3 中值 */

#define FSW_CSWAP(a, b) do { uint8_t t_ = (a); if (t_ > (b)) { (a) = (b); (b) = t_; } } while (0)

/* 三个辅助函数：与 HLS 原文的 filt_min3 / filt_max3 / filt_med3 一一对应。
   刻意**不合并、不"优化"**——移植这类比较器网络时手写得更紧凑非常容易搞错方向
   （本文件第一版就把 max(a0,a1,a2) 写成了 max(min(a0,a1),a2)，靠主机逐位比对
   才抓出来），保持与原文同形是最稳的。 */
static inline uint8_t fsw_min3(uint8_t a, uint8_t b, uint8_t c)
{
    const uint8_t ab = (a < b) ? a : b;
    return (ab < c) ? ab : c;
}

static inline uint8_t fsw_max3(uint8_t a, uint8_t b, uint8_t c)
{
    const uint8_t ab = (a > b) ? a : b;
    return (ab > c) ? ab : c;
}

static inline uint8_t fsw_med3(uint8_t a, uint8_t b, uint8_t c)
{
    const uint8_t lo = (a < b) ? a : b;
    const uint8_t hi = (a < b) ? b : a;
    const uint8_t t  = (hi < c) ? hi : c;
    return (lo > t) ? lo : t;
}

/* 19 比较器 3x3 中值；入参顺序无关。与 PL 版 filt_med9 逐行相同。 */
static inline uint8_t fsw_med9(uint8_t v0, uint8_t v1, uint8_t v2,
                               uint8_t v3, uint8_t v4, uint8_t v5,
                               uint8_t v6, uint8_t v7, uint8_t v8)
{
    uint8_t a0 = v0, b0 = v1, c0 = v2;
    uint8_t a1 = v3, b1 = v4, c1 = v5;
    uint8_t a2 = v6, b2 = v7, c2 = v8;
    FSW_CSWAP(a0, b0); FSW_CSWAP(b0, c0); FSW_CSWAP(a0, b0);
    FSW_CSWAP(a1, b1); FSW_CSWAP(b1, c1); FSW_CSWAP(a1, b1);
    FSW_CSWAP(a2, b2); FSW_CSWAP(b2, c2); FSW_CSWAP(a2, b2);
    /* 每行已升序：a=min, b=median, c=max */
    const uint8_t hi_of_min = fsw_max3(a0, a1, a2);
    const uint8_t med_of_med = fsw_med3(b0, b1, b2);
    const uint8_t lo_of_max = fsw_min3(c0, c1, c2);
    return fsw_med3(hi_of_min, med_of_med, lo_of_max);
}

/* in -> out，3x3 中值，行列都 clamp。要求 in != out。 */
static void fsw_med3_pass(const uint8_t *in, uint8_t *out, int w, int h)
{
    for (int oy = 0; oy < h; oy++) {
        const uint8_t *r0 = in + (size_t)((oy > 0) ? (oy - 1) : 0) * w;
        const uint8_t *r1 = in + (size_t)oy * w;
        const uint8_t *r2 = in + (size_t)((oy + 1 < h) ? (oy + 1) : (h - 1)) * w;
        uint8_t *d = out + (size_t)oy * w;
        for (int x = 0; x < w; x++) {
            const int x0 = (x > 0) ? (x - 1) : 0;
            const int x2 = (x + 1 < w) ? (x + 1) : (w - 1);
            d[x] = fsw_med9(r0[x0], r0[x], r0[x2],
                            r1[x0], r1[x], r1[x2],
                            r2[x0], r2[x], r2[x2]);
        }
    }
}

/* ---------------------------------------------------------------- 5x5 高斯 */

/* σ=1.2 的 5 抽头定点核（Q8，和 = 256），与 cv2.GaussianBlur((5,5),0) 同量级。
   22/62/88/62/22 由 exp(-d^2/(2*1.2^2)) 归一化取整而来。 */
#define FSW_G0 22
#define FSW_G1 62
#define FSW_G2 88

/* 水平方向 in -> out（行内 5 抽头，左右 clamp） */
static void fsw_gauss_h(const uint8_t *in, uint8_t *out, int w, int h)
{
    for (int y = 0; y < h; y++) {
        const uint8_t *s = in + (size_t)y * w;
        uint8_t *d = out + (size_t)y * w;
        for (int x = 0; x < w; x++) {
            const int x0 = (x > 1) ? (x - 2) : 0;
            const int x1 = (x > 0) ? (x - 1) : 0;
            const int x2 = x;
            const int x3 = (x + 1 < w) ? (x + 1) : (w - 1);
            const int x4 = (x + 2 < w) ? (x + 2) : (w - 1);
            const unsigned acc = (unsigned)FSW_G0 * s[x0] + (unsigned)FSW_G1 * s[x1]
                               + (unsigned)FSW_G2 * s[x2] + (unsigned)FSW_G1 * s[x3]
                               + (unsigned)FSW_G0 * s[x4] + 128u;
            d[x] = (uint8_t)(acc >> 8);
        }
    }
}

/* 垂直方向 in -> out（列方向 5 抽头，上下 clamp） */
static void fsw_gauss_v(const uint8_t *in, uint8_t *out, int w, int h)
{
    for (int y = 0; y < h; y++) {
        const int y0 = (y > 1) ? (y - 2) : 0;
        const int y1 = (y > 0) ? (y - 1) : 0;
        const int y2 = y;
        const int y3 = (y + 1 < h) ? (y + 1) : (h - 1);
        const int y4 = (y + 2 < h) ? (y + 2) : (h - 1);
        const uint8_t *r0 = in + (size_t)y0 * w;
        const uint8_t *r1 = in + (size_t)y1 * w;
        const uint8_t *r2 = in + (size_t)y2 * w;
        const uint8_t *r3 = in + (size_t)y3 * w;
        const uint8_t *r4 = in + (size_t)y4 * w;
        uint8_t *d = out + (size_t)y * w;
        for (int x = 0; x < w; x++) {
            const unsigned acc = (unsigned)FSW_G0 * r0[x] + (unsigned)FSW_G1 * r1[x]
                               + (unsigned)FSW_G2 * r2[x] + (unsigned)FSW_G1 * r3[x]
                               + (unsigned)FSW_G0 * r4[x] + 128u;
            d[x] = (uint8_t)(acc >> 8);
        }
    }
}

/* ---------------------------------------------------------------- 最后一步 */

/* 灰度平面 -> BGRX（B=G=R=Y，第 4 字节 0） */
static void fsw_expand(const uint8_t *plane, uint8_t *dst, int w, int h, int stride)
{
    for (int y = 0; y < h; y++) {
        const uint8_t *s = plane + (size_t)y * w;
        uint8_t *d = dst + (size_t)y * stride;
        for (int x = 0; x < w; x++) {
            const uint8_t v = s[x];
            d[0] = v; d[1] = v; d[2] = v; d[3] = 0;
            d += 4;
        }
    }
}

/* ------------------------------------------------------------------ 主函数 */

int filter_sw_apply(int mode, const uint8_t *src, uint8_t *dst,
                    int w, int h, int stride)
{
    if (src == 0 || dst == 0)
        return -1;
    if (mode <= FILT_MODE_OFF || mode >= FILT_MODE_COUNT)
        return -3;
    if (w < 3 || h < 3 || w > FILT_SW_MAX_W || h > FILT_SW_MAX_H)
        return -2;

    if (stride < w * 4)
        return -1;

    uint8_t *A = s_planeA;
    uint8_t *B = s_planeB;

    /* Read each complete source channel before writing that channel to dst, preserving the
       API's in-place guarantee. A/B are reused for one channel at a time;
       caller src==dst is safe because only that same channel is overwritten. */
    if (mode == FILT_MODE_RGBMED || mode == FILT_MODE_RGBGAUSS) {
        for (int ch = 0; ch < 3; ++ch) {
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    A[(size_t)y*w+x] = src[(size_t)y*stride+x*4+ch];
            const uint8_t *result;
            if (mode == FILT_MODE_RGBMED) {
                fsw_med3_pass(A, B, w, h);
                result = B;
            } else {
                fsw_gauss_h(A, B, w, h);
                fsw_gauss_v(B, A, w, h);
                result = A;
            }
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    dst[(size_t)y*stride+x*4+ch] = result[(size_t)y*w+x];
        }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                dst[(size_t)y*stride+x*4+3] = 0;
        return 0;
    }

    /* 第一步永远一样：BGRX -> 亮度平面 */
    fsw_lum_pass(src, A, w, h, stride);

    switch (mode) {
    case FILT_MODE_GRAY:
        fsw_expand(A, dst, w, h, stride);
        break;
    case FILT_MODE_MED3:
        fsw_med3_pass(A, B, w, h);
        fsw_expand(B, dst, w, h, stride);
        break;
    case FILT_MODE_MED3X2:
        fsw_med3_pass(A, B, w, h);
        fsw_med3_pass(B, A, w, h);
        fsw_expand(A, dst, w, h, stride);
        break;
    case FILT_MODE_MED3X3:
        fsw_med3_pass(A, B, w, h);
        fsw_med3_pass(B, A, w, h);
        fsw_med3_pass(A, B, w, h);
        fsw_expand(B, dst, w, h, stride);
        break;
    case FILT_MODE_GAUSS5:
        fsw_gauss_h(A, B, w, h);
        fsw_gauss_v(B, A, w, h);
        fsw_expand(A, dst, w, h, stride);
        break;
    default:
        return -3;
    }
    return 0;
}
