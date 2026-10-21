/* ===========================================================================
 *  filter_sw.h —— 斑点抑制 / 增强的 **PS 软件实现**（多档可选）
 *
 *  为什么在 PS：PL 侧那个滤波 IP 占 LUT≈3.5K / DSP 14，把只有 +0.145 ns 余量的
 *  YOLO IP 推成 −0.230 ns / 222 个失败端点。搬到 PS 之后时序完全收敛
 *  （WNS +0.009 ns / 0 个失败端点），而且**再也不用为算法复杂度付资源代价** ——
 *  所以这里可以放比单个 3x3 中值更"狠"的算子。
 *
 *  各档语义（都用 BGRX 输入输出，内部先做亮度合成）：
 *
 *    FILT_MODE_OFF     不滤波（原图直通，对应消融实验的 "original"）
 *    FILT_MODE_MED3    亮度 + **3x3 中值**。与 PL 版 IP **逐位一致**，
 *                      对应消融实验的 "med3"。人眼几乎看不出（只扰动约 3% 像素）
 *    FILT_MODE_MED3X2  3x3 中值跑**两遍**。视觉上开始明显
 *    FILT_MODE_MED3X3  跑**三遍**。肉眼一眼能分辨（迭代中值是很经典的强平滑）
 *    FILT_MODE_GAUSS5  亮度 + **5x5 高斯**（σ=1.2，分离式两遍）。
 *                      非保边，边缘变糊，视觉上和迭代中值是不同的"味道"
 *
 *  ⚠️ 精度口径（详见 filter_study/消融实验结果.md）：
 *    · 现行 3x3 中值相对 `gray` 对照的**精度独立贡献 ≈ 0**，它的定位是
 *      **视觉增强 / 斑点抑制**，不是精度指标；
 *    · 越强的平滑越可能**损害**检测（med5 在 conf=0.15 下就比 med3 多 1 个漏检）。
 *      所以"更胆大"是**观感/演示**上的选择，要讲精度必须拿数据说话 ——
 *      `run_int8_forward_sim.py --variants ... med3x2 med3x3 gauss5` 可以离线先量。
 *
 *  本文件只用 <stdint.h>/<stddef.h>，**不含任何 Xilinx 依赖**，因此可以直接用主机
 *  gcc 编译，与 Python 参考实现做逐位比对（filter_study/verify_filter_sw.py）。
 * ===========================================================================
 */
#ifndef FILTER_SW_H
#define FILTER_SW_H

#include <stdint.h>

/* 与 yolo_filter3x3.cpp 的 FILT_MAX_W / FILT_MAX_H 保持一致 */
#define FILT_SW_MAX_W 1024
#define FILT_SW_MAX_H 4096

typedef enum {
    FILT_MODE_OFF    = 0,   /* 原图直通 */
    FILT_MODE_MED3   = 1,   /* 亮度 + 3x3 中值（与 PL IP 逐位一致） */
    FILT_MODE_MED3X2 = 2,   /* 3x3 中值 x2 */
    FILT_MODE_MED3X3 = 3,   /* 3x3 中值 x3 */
    FILT_MODE_GAUSS5 = 4,   /* 亮度 + 5x5 高斯 σ=1.2 */
    FILT_MODE_GRAY   = 5,   /* Luma only: isolate color removal */
    FILT_MODE_RGBMED = 6,   /* Independent B/G/R 3x3 median */
    FILT_MODE_RGBGAUSS = 7, /* Independent B/G/R fixed-point Gaussian */
    FILT_MODE_COUNT  = 8
} filter_sw_mode;

/* 短名（≤5 字符，给屏上按钮用）/ 一句话描述（给模式行与 UART 用） */
const char *filter_sw_mode_name(int mode);
const char *filter_sw_mode_desc(int mode);
/* 模式编号自增循环（FILT_MODE_OFF -> MED3 -> ... -> OFF） */
int filter_sw_mode_next(int mode);

/* 就近滤波：读 src、写 dst。两个缓冲都是 BGRX，每行 stride 字节。
 * src 与 dst 允许相同（就地，此时原图会被覆盖）。
 *
 * 返回  0  成功
 *      -2  尺寸超出支持范围（w<3 或 h<3 或 w>MAX_W 或 h>MAX_H），未做任何事
 *      -1  参数非法（空指针 / stride < w*4）
 *      -3  mode 非法或 == FILT_MODE_OFF（OFF 由调用方直接旁路，不必调进来）
 */
int filter_sw_apply(int mode, const uint8_t *src, uint8_t *dst,
                    int w, int h, int stride);

/* 供主机测试用：两块灰度中间平面的地址（板端就是这两块 4MB 的 BSS） */
uint8_t *filter_sw_plane_a(void);
uint8_t *filter_sw_plane_b(void);

#endif /* FILTER_SW_H */
