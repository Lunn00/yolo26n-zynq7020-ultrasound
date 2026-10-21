#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ui_widgets.h"

#include "xaxivdma.h"
#include "xaxivdma_i.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xtime_l.h"
#include "ff.h"

#include "yolo_detect.h"
#include "touch_hw.h"
#include "filter_sw.h"

/* Production profile: diagnostic runs and SD dumps are opt-in. */
#ifndef YOLO_BOOT_SELFTEST
#define YOLO_BOOT_SELFTEST 0
#endif
#ifndef YOLO_RUN_DIAGNOSTICS
#define YOLO_RUN_DIAGNOSTICS 0
#endif
#ifndef YOLO_REQUIRE_SCALES
#define YOLO_REQUIRE_SCALES 0
#endif
/* YOLO26 Conv1x1 fit: generated 111-layer table and parameters.  The HLS
   kernel includes the same role header; every scheduled layer runs in PL.
   The original attention blocks are fitted residual 1x1 convolutions and
   the grouped head convolutions are standard dense convolutions. */
#include "yolo_layers_i8.h"
#include "yolo_fixed_params_i8.h"
#include "yolo_layer_roles.h"
#if YOLO_BOOT_SELFTEST
#include "yolo_selftest_vectors.h"
#endif

#define LCD_WIDTH       800
#define LCD_HEIGHT      480
#define PIXEL_BYTES     3
#define LINE_BYTES      (LCD_WIDTH * PIXEL_BYTES)
#define FRAME_SIZE      (LINE_BYTES * LCD_HEIGHT)

/* 图像缓冲区改成 4 字节/像素（BGRX），这样滤波 IP 可以是干净的 1 像素/周期
   32bit 读写，不必在硬件里做 3 字节流的解包（那种写法最容易出只在板上才暴露的错）。
   byte0=B, byte1=G, byte2=R, byte3=0。 */
#define IMG_PIXEL_BYTES 4

#define FB_ADDR         0x10000000
#define VDMA_DEV_ID     XPAR_AXI_VDMA_0_DEVICE_ID

#define INPUT_BUF       0x20000000
#define WORK_BUF        0x21000000
#define PRED_BUF        0x22000000
#define WEIGHTS_BUF     0x23000000
#define FIXED_PARAMS_BUF 0x23800000
/* YOLO26: the activation scales the PS operators read.  Slotted after the
   fixed-param blob (which grew from 137,300 to 216,356 bytes) and before
   IMG_BUF, so nothing in the v8 layout moves. */
#define ACT_SCALES_BUF  0x24000000
#define IMG_BUF         0x25000000
#define MAX_IMG_W       3840
#define MAX_IMG_H       2160
#define MAX_IMG_BYTES   (MAX_IMG_W * MAX_IMG_H * IMG_PIXEL_BYTES)

/* ---------------------------------------------------------------------------
 * FLT_BUF：滤波结果缓冲（**非破坏性**，原图一直留在 IMG_BUF）
 *
 * 滤波已经搬到 PS（filter_sw.c）。它不再就地改写，而是 IMG_BUF -> FLT_BUF，
 * 所以：
 *      FILTER ON  -> 跑一次软件滤波到 FLT_BUF，g_img_src = FLT_BUF
 *      FILTER OFF -> g_img_src = IMG_BUF      —— **零成本**，连一次 memcpy 都不用
 * 比 PL 版（就地 + 每次载入 memcpy 一份原图）还省。
 *
 * 顺带的好处：IMG_BUF 不再被任何外部 master 碰，所以那对
 * flush-before / invalidate-after 也整段消失了（PL 版走 S_AXI_HP1 绕 D-cache，
 * 必须做；软件版完全不需要）。
 *
 * 地址放在 IMG_BUF 的**最大可能范围之后**，避免大图时两个缓冲重叠。
 * DDR 是 1GB（XPAR_PS7_DDR_0_S_AXI_HIGHADDR = 0x3FFFFFFF），这段是空闲的。
 * --------------------------------------------------------------------------- */
#define FLT_BUF         0x27000000u
#if (FLT_BUF < (IMG_BUF + MAX_IMG_BYTES))
#error "FLT_BUF 与 IMG_BUF 的最大范围重叠：大图时会踩内存，必须把 FLT_BUF 往后放。"
#endif
#if ((FLT_BUF + MAX_IMG_BYTES) > 0x3F000000u)
#error "FLT_BUF 超出 DDR 范围。"
#endif

/* 当前生效的图像源：指向 IMG_BUF（未滤波）或 FLT_BUF（已滤波）。
   letterbox 显示与送进 YOLO 的预处理都读它，保证屏上和网络看到的是同一张图。 */
static const u8 *g_img_src = (const u8 *)IMG_BUF;


/* Blob sizes come from the generated YOLO26 headers.  Literal byte counts
   would risk loading an incompatible model from the SD card. */
#define WEIGHTS_BYTES     YOLO_I8_WEIGHT_BYTES
#define FIXED_PARAM_BYTES YOLO_I8_FIXED_PARAM_BYTES
/* The scale blob is retained as an SD card integrity check.  Inference uses
   the fixed-param blob for PL requantization. */
#define ACT_SCALE_BYTES   YOLO_I8_ACT_SCALE_BYTES
/* FNV-1a32 of the three files this build expects on the SD card.  Recompute with
   scripts/print_blob_sums.py in yolo26_hls/scripts whenever the exporter writes
   new blobs; a mismatch disables inference on purpose. */
#define WEIGHTS_SUM_EXPECT 0x80AAAC75u
#define FIXED_SUM_EXPECT   0x0D138542u
#define ACT_SCALES_SUM_EXPECT 0x1B181395u
#define YOLO_TIMEOUT_MS 600000u
#define YOLO_CONF_TH    0.15f
#define YOLO_IOU_TH     0.45f
#define APP_BUILD_TAG   "YOLO26N224_LCD_UI_V2_0928"
#define INPUT_BYTES     (YOLO_INPUT_W * YOLO_INPUT_H * 3)
#define PRED_ELEMS      ((YOLO_BOX_CHANNELS + 1) * YOLO_ANCHORS)
#define PRED_BYTES      (PRED_ELEMS * 2)

/* ---------------------------------------------------------------------------
 * PRED_BUF 的 cache 维护长度必须是整行
 *
 * PRED_BYTES = 65*1029*2 = 133,770，**不是 32 的整数倍**（尾 10 字节）。
 * BSP 的 Xil_DCacheInvalidateRange() 对"不足一行"的尾巴不是纯 invalidate，而是
 * 对那一行调用 Xil_L1DCacheFlushLine()，也就是 clean & invalidate
 * （见 standalone_v7_6/src/xil_cache.c）。clean 会把 CPU 侧的旧内容写回 DDR，
 * 语义上不干净 —— 万一那一行当时是脏的，就会把 IP 刚 DMA 进来的结果盖掉，
 * 表现为"同一张图连按 RUN，max_conf 在 14/15 之间跳"。
 *
 * 把长度补到 32 的整数倍，尾巴那行就走和其余行完全一样的纯 invalidate 路径。
 * 多出来的 22 字节落在 PRED_BUF(0x22000000) 之后、WEIGHTS_BUF(0x23000000) 之前
 * 的空闲 DDR 里，invalidate 它没有任何副作用（下面的 #if 保证这一点）。
 * --------------------------------------------------------------------------- */
#define PRED_BYTES_ALIGNED (((PRED_BYTES) + 31u) & ~31u)
#if ((PRED_BUF) + (PRED_BYTES_ALIGNED) > (WEIGHTS_BUF))
#error "PRED_BUF 补齐到整行后会压到 WEIGHTS_BUF，必须给中间留出 padding。"
#endif

/* ---------------------------------------------------------------------------
 * Blob placement.
 *
 * The generated blob sizes must fit their fixed DDR slots.  An overlap would
 * silently corrupt activations and detections, so these checks fail the build.
 * --------------------------------------------------------------------------- */
#if ((WEIGHTS_BUF) + (WEIGHTS_BYTES) > (FIXED_PARAMS_BUF))
#error "WEIGHTS_BUF + WEIGHTS_BYTES 压到 FIXED_PARAMS_BUF，调整内存布局。"
#endif
#if ((FIXED_PARAMS_BUF) + (FIXED_PARAM_BYTES) > (ACT_SCALES_BUF))
#error "FIXED_PARAMS_BUF + FIXED_PARAM_BYTES 压到 ACT_SCALES_BUF，调整内存布局。"
#endif
#if ((ACT_SCALES_BUF) + (ACT_SCALE_BYTES) > (IMG_BUF))
#error "ACT_SCALES_BUF + ACT_SCALE_BYTES 压到 IMG_BUF，调整内存布局。"
#endif

/* ---------------------------------------------------------------------------
 * 滤波 IP（yolo_filter3x3）的 s_axilite 基地址
 *
 * 刻意**硬编码**而不是用 XPAR_* 宏：该地址是在 BD 里显式指定的 0x41000000
 * （见 scripts\vivado_impl_touch_p64_150.tcl 的 assign_bd_address）。
 * 这样就不必重新生成 XSA / 更新 BSP，现有平台与工程保持有效。
 * 若改了 BD 里的地址，这里必须同步修改。
 * --------------------------------------------------------------------------- */
#define FILTER_BASEADDR 0x41000000u
#define FILTER_REG_CTRL      0x00
#define FILTER_REG_IN_IMG    0x10
#define FILTER_REG_OUT_IMG   0x1C
#define FILTER_REG_WIDTH     0x28
#define FILTER_REG_HEIGHT    0x30
#define FILTER_MAX_W         1024   /* 必须与 yolo_filter3x3.cpp 的 FILT_MAX_W 一致 */
#define FILTER_MAX_H         4096   /* 必须与 yolo_filter3x3.cpp 的 FILT_MAX_H 一致 */
#define FILTER_TIMEOUT_MS    30000u

#ifdef XPAR_YOLO8N_HLS_I8_0_S_AXI_CONTROL_BASEADDR
#define YOLO_BASEADDR XPAR_YOLO8N_HLS_I8_0_S_AXI_CONTROL_BASEADDR
#else
#define YOLO_BASEADDR 0x40000000
#endif

#define YOLO_REG_CTRL      0x00
#define YOLO_REG_INPUT     0x10
#define YOLO_REG_WEIGHTS   0x1C
#define YOLO_REG_ACT_SCALES 0x28
#define YOLO_REG_WORK      0x34
#define YOLO_REG_PRED      0x40
/* The 111-layer Conv1x1 IP uses the 921 five-pointer AXI-Lite map.  It has no
   li0/li1 range registers: every start runs the complete network. */
#if YOLO_I8_NUM_LAYERS != 111u
#error "This application requires the 111-layer Conv1x1 YOLO26 table"
#endif

/* 800x480: full-height image at left, status and 2-column controls at right. */
#define MAX_IMG_IDX 9999
#define BTN_COUNT UI_BUTTONS
#define IMG_AREA_W 470
#define IMG_AREA_H LCD_HEIGHT
#define IMG_PAD_BOTTOM 4
#define CTRL_X 470
#define CTRL_W 330
#define HUD_X CTRL_X
#define HUD_W CTRL_W
#define HUD_AREA_H 256
#define HUD_TEXT_DX 10
#define CTRL_BTN_H 56
#define CTRL_BTN_Y0 256
#define BTN_X0 CTRL_X
#define BTN_W CTRL_W
#define FILT_SWITCH_DELAY_MS 60u
#define HUD_BG_R 18
#define HUD_BG_G 25
#define HUD_BG_B 34
enum { BTN_PREV=UI_PREV, BTN_NEXT=UI_NEXT, BTN_RUN=UI_RUN, BTN_AUTO=UI_AUTO,
       BTN_SELECT=UI_SELECT, BTN_FILTER=UI_FILTER, BTN_COMPARE=UI_COMPARE };

static XAxiVdma vdma;
static FATFS fatfs;
static FIL fil;
static int img_w;
static int img_h;
static int img_row;
static int g_img_idx = 1;
static int g_bmp_ok = 0;
static int g_auto_run = 0;

/* Three modes selected from the complete dataset experiment. RAW is default. */
static const int FILT_UI_CYCLE[] = {
    FILT_MODE_OFF, FILT_MODE_GAUSS5, FILT_MODE_RGBGAUSS
};
#define FILT_UI_N ((int)(sizeof(FILT_UI_CYCLE) / sizeof(FILT_UI_CYCLE[0])))


static int g_filter_mode = FILT_MODE_OFF;

typedef struct {
    int valid,conf,count;
    YoloBox boxes[YOLO_MAX_BOXES];
    u32 filter_us,pre_us,post_us,pl_us,call_us,wall_us;
} UiResult;
static UiResult g_results[3];
static YoloBox g_source_boxes[YOLO_MAX_BOXES];
static int g_source_box_n;
static UiRoi g_roi;
static int g_roi_valid,g_roi_manual;
static int g_keypad,g_compare,g_compare_mode=FILT_MODE_GAUSS5;
static int g_compare_filter_ok,g_compare_overwrote_filter;
static UiKeyboard g_keyboard;
static u32 g_filter_us,g_pre_us,g_post_us,g_pl_us,g_call_us,g_wall_us,g_compare_filter_us;
static int g_flt_mode=-1;
static u32 g_flt_us;
static int g_catalog[MAX_IMG_IDX],g_catalog_n,g_catalog_ready;
static u8 g_image_background[IMG_AREA_W*IMG_AREA_H*PIXEL_BYTES];
static int g_background_valid;
static int ui_cache_index(int mode){return mode==FILT_MODE_GAUSS5?1:mode==FILT_MODE_RGBGAUSS?2:0;}
static void ui_cache_background(u8 *fb)
{
    for(int y=0;y<IMG_AREA_H;++y)
        memcpy(g_image_background+y*IMG_AREA_W*3,fb+y*LINE_BYTES,IMG_AREA_W*3);
    g_background_valid=1;
}
static void ui_restore_background(u8 *fb)
{
    if(!g_background_valid)return;
    for(int y=0;y<IMG_AREA_H;++y)
        memcpy(fb+y*LINE_BYTES,g_image_background+y*IMG_AREA_W*3,IMG_AREA_W*3);
}

static int filter_ui_next(int mode)
{
    for (int i = 0; i < FILT_UI_N; i++)
        if (FILT_UI_CYCLE[i] == mode)
            return FILT_UI_CYCLE[(i + 1) % FILT_UI_N];
    return FILT_UI_CYCLE[0];
}

static u32 fnv1a32(const u8 *p, u32 n)
{
    u32 h = 2166136261u;
    while (n--) {
        h ^= *p++;
        h *= 16777619u;
    }
    return h;
}

static u32 elapsed_ms(XTime t0, XTime t1)
{
    /* The ARM global timer used by XTime ticks at CPU_CLK / 2. */
    return (u32)((t1 - t0) * 1000 /
                 (XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ / 2));
}

/* 微秒版。注意先转 u64 再乘：50 us 就是 1.6 万多个 tick，
   直接 (ticks * 1000000) 会溢出 u32。 */
static u32 elapsed_us(XTime t0, XTime t1)
{
    return (u32)(((u64)(t1 - t0) * 1000000ULL) /
                 (u64)(XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ / 2));
}

/* ===========================================================================
 *  性能剖析（诊断用）：定位 AUTO 巡览里每张图的时间去向
 *
 *  实测 30 张图 36 s（1.2 s/张），而推理 + 滤波只有 ~250 ms，
 *  差的 ~950 ms 必须靠数据定位，不能靠猜。每次 AUTO 循环打印一行：
 *    [PERF] auto img=7 wall=1204 ms sum=1099 ms
 *    [PERF]  sd=512 flushimg=28 filter=26 lbx=263 pre=48 infer=196
 *            dec=9 flushfb=14 touch=3 (ms)
 *
 *  sd          load_bmp_from_sd()（含 SD 读 + BGRX 解包）
 *  flushimg    IMG_BUF 的 cache flush(2.1MB) + 滤波后的 invalidate
 *              —— **搬 PS 之后这里应该是 0**：软件滤波走 CPU，D-cache 天生一致，
 *              不需要任何 cache 维护。这一列归零就是搬迁省下的开销。
 *  filter      滤波 IP（只在 CPU 侧等待的时间）
 *  lbx         letterbox_to_fb()（软件双线性，38.4 万像素）
 *  pre         预处理 + INPUT flush + pred 准备
 *  infer       start_yolo_ip()（阻塞等 IP 完成）
 *  dec         NMS 解码 + 坐标映射 + 画框
 *  flushfb     状态行/按钮重绘 + 整帧 flush(1.15MB)
 *  touch       本循环里 3 次触摸轮询合计
 *
 *  APP_PERF=0 可整体关掉（关掉后 pf_* 变成空操作，不影响行为）。
 * =========================================================================== */
#ifndef APP_PERF
#define APP_PERF 0
#endif

/* Ticks are CPU_CLK/2; used by the AUTO-loop wall-clock timer as well, so it is
 * defined even when APP_PERF is 0. */
static u32 us_between(XTime t0, XTime t1)
{
    return (u32)((t1 - t0) * 1000000ULL /
                 (XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ / 2));
}

/* Slot ids live outside the #if so the pf_add() calls below still compile when
 * APP_PERF is 0 (macro arguments are parsed even if the macro discards them). */
#define PF_N 9
enum { PF_SD = 0, PF_FLUSH_IMG, PF_FILTER, PF_LBX, PF_PRE, PF_INFER,
       PF_POST, PF_FLUSH_FB, PF_TOUCH };

#if APP_PERF
static const char *const pf_name[PF_N] __attribute__((unused)) = {
    "sd", "flushimg", "filter", "lbx", "pre",
    "infer", "dec", "flushfb", "touch"
};
static u32 pf_us[PF_N];
static int pf_img_idx;

#define pf_set_img(i) (pf_img_idx = (i))

static void pf_add(int slot, XTime t0, XTime t1)
{
    if (slot >= 0 && slot < PF_N)
        pf_us[slot] += us_between(t0, t1);
}

static void pf_reset(void)
{
    for (int i = 0; i < PF_N; i++)
        pf_us[i] = 0;
}

static void pf_report(const char *tag, u32 wall_ms)
{
    u32 sum = 0;
    for (int i = 0; i < PF_N; i++)
        sum += pf_us[i];
    xil_printf("[PERF] %s img=%d wall=%u ms sum=%u ms\r\n",
               tag, pf_img_idx, (unsigned)wall_ms, (unsigned)(sum / 1000u));
    xil_printf("[PERF]  sd=%u flushimg=%u filter=%u lbx=%u pre=%u "
               "infer=%u dec=%u flushfb=%u touch=%u (ms)\r\n",
               (unsigned)(pf_us[PF_SD] / 1000u),
               (unsigned)(pf_us[PF_FLUSH_IMG] / 1000u),
               (unsigned)(pf_us[PF_FILTER] / 1000u),
               (unsigned)(pf_us[PF_LBX] / 1000u),
               (unsigned)(pf_us[PF_PRE] / 1000u),
               (unsigned)(pf_us[PF_INFER] / 1000u),
               (unsigned)(pf_us[PF_POST] / 1000u),
               (unsigned)(pf_us[PF_FLUSH_FB] / 1000u),
               (unsigned)(pf_us[PF_TOUCH] / 1000u));
    pf_reset();
}
#else
#define pf_add(slot, t0, t1) do { (void)(slot); } while (0)
#define pf_reset()           do { } while (0)
#define pf_report(tag, ms)   do { (void)(tag); (void)(ms); } while (0)
#define pf_set_img(i)        do { (void)(i); } while (0)
#endif

/* 只把改动过的横条写回 DDR，避免每次重绘按钮都刷整帧（1.15 MB）。 */
static void flush_band(int y0, int h)
{
    if (y0 < 0)
        y0 = 0;
    if (y0 + h > LCD_HEIGHT)
        h = LCD_HEIGHT - y0;
    if (h <= 0)
        return;
    Xil_DCacheFlushRange((INTPTR)(FB_ADDR + (u32)y0 * LINE_BYTES),
                         (u32)h * LINE_BYTES);
}

static void draw_buttons(u8 *fb){ui_main_buttons(fb,g_filter_mode,g_auto_run);}

/* 竖排命中判定：控制列内的某一行 */
static int btn_hit(int x,int y){return ui_main_hit(x,y);}

static void prepare_pred_buffer(s16 *pred)
{
    memset(pred, 0, PRED_BYTES);
    for (int a = 0; a < YOLO_ANCHORS; a++)
        pred[YOLO_BOX_CHANNELS * YOLO_ANCHORS + a] = (s16)-32768;
    Xil_DCacheFlushRange((INTPTR)PRED_BUF, PRED_BYTES);
}

#if YOLO_RUN_DIAGNOSTICS
static void log_pred_stats(const s16 *pred)
{
    const s16 *conf = pred + YOLO_BOX_CHANNELS * YOLO_ANCHORS;
    s16 min_v = conf[0];
    s16 max_v = conf[0];
    int zeros = 0;
    int positive = 0;
    int untouched = 0;
    for (int a = 0; a < YOLO_ANCHORS; a++) {
        if (conf[a] < min_v) min_v = conf[a];
        if (conf[a] > max_v) max_v = conf[a];
        if (conf[a] == 0) zeros++;
        if (conf[a] > 0) positive++;
        if (conf[a] == (s16)-32768) untouched++;
    }
    xil_printf("[YOLO] conf_q10 min=%d max=%d zero=%d positive=%d untouched=%d\r\n",
               min_v, max_v, zeros, positive, untouched);
}
#endif

static void delay_ms(u32 ms)
{
    XTime t0, now;
    XTime_GetTime(&t0);
    do {
        XTime_GetTime(&now);
    } while (elapsed_ms(t0, now) < ms);
}

static int image_file_exists(int idx)
{
    char path[16];FILINFO info;
    if(idx<1||idx>MAX_IMG_IDX)return 0;
    snprintf(path,sizeof(path),"%d.bmp",idx);
    return f_stat(path,&info)==FR_OK && !(info.fattrib&AM_DIR);
}
static void ui_catalog(void)
{
    if(g_catalog_ready)return;
    DIR dir;FILINFO info;g_catalog_n=0;
    if(f_opendir(&dir,"/")!=FR_OK)return;
    while(f_readdir(&dir,&info)==FR_OK && info.fname[0]) {
        if(info.fattrib&AM_DIR)continue;
        char *end;long id=strtol(info.fname,&end,10);
        if(id<1||id>MAX_IMG_IDX||end==info.fname||strlen(end)!=4||end[0]!='.')continue;
        if((end[1]!='b'&&end[1]!='B')||(end[2]!='m'&&end[2]!='M')||(end[3]!='p'&&end[3]!='P'))continue;
        /* Match canonical numeric names; do not index 001.bmp as 1.bmp. */
        char expected[16];snprintf(expected,sizeof(expected),"%ld",id);
        if((size_t)(end-info.fname)!=strlen(expected)||strncmp(info.fname,expected,strlen(expected)))continue;
        if(g_catalog_n<MAX_IMG_IDX)g_catalog[g_catalog_n++]=(int)id;
    }
    f_closedir(&dir);
    for(int i=1;i<g_catalog_n;++i){int value=g_catalog[i],j=i;while(j>0&&g_catalog[j-1]>value){g_catalog[j]=g_catalog[j-1];--j;}g_catalog[j]=value;}
    g_catalog_ready=1;
    xil_printf("[UI] indexed %d BMP files\r\n",g_catalog_n);
}

static int find_first_image(void){ui_catalog();return g_catalog_n?g_catalog[0]:0;}

static int find_next_image(int idx){ui_catalog();for(int i=0;i<g_catalog_n;++i)if(g_catalog[i]>idx)return g_catalog[i];return g_catalog_n?g_catalog[0]:0;}

static int find_prev_image(int idx){ui_catalog();for(int i=g_catalog_n-1;i>=0;--i)if(g_catalog[i]<idx)return g_catalog[i];return g_catalog_n?g_catalog[g_catalog_n-1]:0;}

/* ===========================================================================
 * 右侧 HUD 文字面板
 *
 * 所有字幕都画在 x>=HUD_X 的面板里，结构上不可能压到左边的图片。
 * 面板内容全部来自这几个全局量，任何一处改完状态调 hud_redraw() 即可，
 * 不需要（也不允许）再往图上写字。
 *
 * 字体是 5x7 点阵且**只有大写字母 + 数字 + " -.:="**：小写字母和 '%' 会被
 * draw_char 静默丢弃（画成空白）。所以这里的文案必须全大写、不带百分号。
 * 顺带修掉了上一版的显示 bug —— 它直接用 filter_sw_mode_desc() 的小写描述，
 * 屏上实际只显示出零星几个字符。
 * =========================================================================== */
static char        g_status_txt[64] = "BOOTING";   /* 要装得下调用方 char[48] 的串 */
static u8          g_status_r = 255, g_status_g = 255, g_status_b = 0;
static int         g_have_infer = 0;    /* 本次载入后跑过推理没有 */
static int         g_box_n = 0;
static int         g_max_conf_pct = 0;
/* 载入的图超过了软件滤波的尺寸上限（FILT_SW_MAX_W/H），滤波被**静默跳过**。
   这个必须上屏：否则在屏上看起来就是"切了滤波档但画面没变"，
   会被当成滤波无效而白白排查半天。 */
static int         g_filter_skipped = 0;

/* 滤波档位的人话说明（全大写，适配 5x7 字体）。
   ⚠️ 字体只有 " -.0123456789:=A-Z"，**没有 '+'**！写成 "LUMA + 3X3" 的话
   那个加号会被 draw_char 静默丢掉，屏上变成 "LUMA  3X3"。filter_study\
   render_lcd_preview.py 会逐像素检查这一点，verify_lcd_layout.py 也查字体表。 */


/* 状态行单独重画（只清这一条 + 只刷这 24 行）。
   推理进行中的进度提示（"YOLO RUN 120 MS"）用它 —— 每 100 ms 刷一次，
   不能走整块面板重画的路径，否则轮询等待里几十 KB 的 DDR 流量会拖慢推理本身。 */
#define HUD_STATUS_H 24

static void hud_draw_status_line(u8 *fb)
{
    yolo_fill_rect(fb,800,480,470,38,330,24,18,25,34);
    yolo_draw_text(fb,800,480,482,42,g_status_txt,g_status_r,g_status_g,g_status_b,2);
}

static void hud_status_only(u8 *fb)
{
    hud_draw_status_line(fb);
    flush_band(38,24);
}

static void draw_hud(u8 *fb)
{
    UiPanel p={0};p.status=g_status_txt;p.image=g_img_idx;p.mode=g_filter_mode;
    p.have_image=g_bmp_ok;p.have_result=g_have_infer;p.confidence=g_max_conf_pct;p.boxes=g_box_n;
    p.skipped=g_filter_skipped;p.roi_ready=g_roi_valid;p.r=g_status_r;p.g=g_status_g;p.b=g_status_b;
    p.filter_us=g_filter_us;p.pre_us=g_pre_us;p.post_us=g_post_us;p.pl_us=g_pl_us;p.call_us=g_call_us;p.wall_us=g_wall_us;
    ui_main_panel(fb,&p);
}
static void ui_reset_image(void)
{
    memset(g_results,0,sizeof(g_results));g_roi_valid=0;g_roi_manual=0;g_source_box_n=0;
    g_have_infer=0;g_box_n=0;g_max_conf_pct=0;g_flt_mode=-1;g_background_valid=0;
    g_pre_us=g_post_us=g_pl_us=g_call_us=g_wall_us=0;
}
static void ui_save_result(void)
{
    UiResult *r=&g_results[ui_cache_index(g_filter_mode)];
    r->valid=g_have_infer;r->conf=g_max_conf_pct;r->count=g_source_box_n;
    memcpy(r->boxes,g_source_boxes,r->count*sizeof(r->boxes[0]));
    r->filter_us=g_filter_us;r->pre_us=g_pre_us;r->post_us=g_post_us;
    r->pl_us=g_pl_us;r->call_us=g_call_us;r->wall_us=g_wall_us;
}
static void ui_restore_result(int mode)
{
    UiResult *r=&g_results[ui_cache_index(mode)];
    g_have_infer=r->valid;g_box_n=r->valid?r->count:0;g_max_conf_pct=r->valid?r->conf:0;
    g_source_box_n=r->valid?r->count:0;
    memcpy(g_source_boxes,r->boxes,g_source_box_n*sizeof(r->boxes[0]));
    g_pre_us=r->pre_us;g_post_us=r->post_us;g_pl_us=r->pl_us;g_call_us=r->call_us;g_wall_us=r->wall_us;
}

/* HUD 只占右边一列，但帧缓冲按行线性排布，逐行精确刷 cache 反而更慢；
   统一刷上半屏（0 .. CTRL_BTN_Y0）即可，图区顺带刷一遍是幂等的。 */
static void hud_redraw(u8 *fb)
{
    draw_hud(fb);
    flush_band(0, HUD_AREA_H);
}

/* 状态行的唯一入口：记下文字/颜色，然后重画整块 HUD 面板。
   调用点不用关心版式，也不允许直接往图上写字。 */
static void show_status(u8 *fb, const char *s, u8 r, u8 g, u8 b)
{
    snprintf(g_status_txt, sizeof(g_status_txt), "%s", s);
    g_status_r = r;
    g_status_g = g;
    g_status_b = b;
    hud_redraw(fb);
}

static const u8 color_table[8][3] = {
    {255,255,255}, {255,255,0}, {0,255,255}, {0,255,0},
    {255,0,255},   {255,0,0},   {0,0,255},   {0,0,0},
};

static void fill_color_bars(u8 *fb)
{
    const int bar_w = LCD_WIDTH / 8;
    for (int y = 0; y < LCD_HEIGHT; y++) {
        for (int x = 0; x < LCD_WIDTH; x++) {
            int idx = x / bar_w;
            if (idx > 7)
                idx = 7;
            u8 *p = fb + (y * LCD_WIDTH + x) * 3;
            p[0] = color_table[idx][0];
            p[1] = color_table[idx][1];
            p[2] = color_table[idx][2];
        }
    }
}

static int vdma_init(XAxiVdma *inst, u16 dev_id)
{
    XAxiVdma_Config *cfg = XAxiVdma_LookupConfig(dev_id);
    int st;
    if (!cfg)
        return XST_FAILURE;
    st = XAxiVdma_CfgInitialize(inst, cfg, cfg->BaseAddress);
    if (st != XST_SUCCESS)
        return st;
    XAxiVdma_Reset(inst, XAXIVDMA_READ);
    while (XAxiVdma_ResetNotDone(inst, XAXIVDMA_READ));
    XAxiVdma_IntrDisable(inst, XAXIVDMA_IXR_ALL_MASK, XAXIVDMA_READ);
    return XST_SUCCESS;
}

static int vdma_start(XAxiVdma *inst, u32 fb_addr)
{
    XAxiVdma_DmaSetup cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.VertSizeInput      = LCD_HEIGHT;
    cfg.HoriSizeInput      = LINE_BYTES;
    cfg.Stride             = LINE_BYTES;
    cfg.FrameDelay         = 0;
    cfg.EnableCircularBuf  = 1;
    cfg.EnableSync         = 1;
    cfg.PointNum           = 0;
    cfg.EnableFrameCounter = 0;
    cfg.FixedFrameStoreAddr = 0;
    cfg.GenLockRepeat      = 1;
    for (int i = 0; i < inst->MaxNumFrames; i++)
        cfg.FrameStoreStartAddr[i] = fb_addr + i * FRAME_SIZE;
    int st = XAxiVdma_DmaConfig(inst, XAXIVDMA_READ, &cfg);
    if (st)
        return st;
    st = XAxiVdma_DmaSetBufferAddr(inst, XAXIVDMA_READ, cfg.FrameStoreStartAddr);
    if (st)
        return st;
    return XAxiVdma_DmaStart(inst, XAXIVDMA_READ);
}

static int load_bmp_from_sd(const char *path, u8 *fb)
{
    (void)fb;
    u8 hdr[54];
    UINT br;
    /* BMP 每行的字节数按 4 对齐（源是 3 字节/像素） */
    static u8 rowtmp[MAX_IMG_W * 3 + 4];
    int st = f_open(&fil, path, FA_READ);
    if (st) {
        xil_printf("BMP open err=%d\r\n", st);
        return -1;
    }
    if (f_read(&fil, hdr, 54, &br) != FR_OK || br < 54) {
        xil_printf("BMP header read fail\r\n");
        f_close(&fil);
        return -1;
    }
    int h_raw = hdr[0x16] | (hdr[0x17] << 8) | (hdr[0x18] << 16) | (hdr[0x19] << 24);
    int w = hdr[0x12] | (hdr[0x13] << 8) | (hdr[0x14] << 16) | (hdr[0x15] << 24);
    int h = h_raw < 0 ? -h_raw : h_raw;
    int bpp = hdr[0x1C] | (hdr[0x1D] << 8);
    int comp = hdr[0x1E] | (hdr[0x1F] << 8) | (hdr[0x20] << 16) | (hdr[0x21] << 24);
    int off = hdr[0x0A] | (hdr[0x0B] << 8) | (hdr[0x0C] << 16) | (hdr[0x0D] << 24);
    const int srow = (3 * w + 3) & ~3;      /* 文件里每行的步长（3 字节/像素，4 对齐） */
    const int drow = w * IMG_PIXEL_BYTES;   /* IMG_BUF 里每行的步长（4 字节/像素） */
    if (w <= 0 || h <= 0 || w > MAX_IMG_W || h > MAX_IMG_H ||
        (u64)drow * h > MAX_IMG_BYTES) {
        xil_printf("BMP too big: %dx%d\r\n", w, h);
        f_close(&fil);
        return -3;
    }
    if (bpp != 24 || comp != 0) {
        xil_printf("BMP unsupported: %d-bit comp=%d\r\n", bpp, comp);
        f_close(&fil);
        return -2;
    }
    if (off > 0)
        f_lseek(&fil, off);
    u8 *buf = (u8 *)IMG_BUF;
    const int top_down = h_raw < 0;
    for (int i = 0; i < h; i++) {
        const int dst_row = top_down ? i : (h - 1 - i);
        u8 *dst = buf + (u32)dst_row * drow;
        if (f_read(&fil, rowtmp, srow, &br) != FR_OK || br != (UINT)srow) {
            xil_printf("BMP row %d short read %u/%d\r\n", i, br, srow);
            f_close(&fil);
            return -1;
        }
        /* 3 字节/像素（BGR）-> 4 字节/像素（BGRX）：多一个 0 填充字节。
           滤波 IP 按 32bit 字处理，这样每像素一次对齐的 32bit 读写。 */
        for (int x = 0; x < w; x++) {
            const u8 *s = rowtmp + x * 3;
            u8 *d = dst + x * IMG_PIXEL_BYTES;
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = 0;
        }
    }
    f_close(&fil);
    img_w = w;
    img_h = h;
    img_row = drow;
    xil_printf("BMP: %dx%d %d-bit srow=%d -> BGRX row=%d\r\n", w, h, bpp, srow, drow);
    return 0;
}

/* ---------------------------------------------------------------------------
 * 图片区 -> 帧缓冲。只画左边 IMG_AREA_W x IMG_AREA_H 这一块，
 * 右边 HUD_X..LCD_WIDTH 直接填面板底色（文字由 draw_hud 再叠上去）。
 * 缩放/偏移记在 g_lbx_* 里，**框映射必须用同一组值**，否则框会偏。
 * --------------------------------------------------------------------------- */
static float g_lbx_scale = 1.0f;
static int   g_lbx_pad_x = 0;
static int   g_lbx_pad_y = 0;

static void letterbox_to_fb(u8 *fb)
{
    const u8 *src = g_img_src;
    const int box_w = IMG_AREA_W;
    const int box_h = IMG_AREA_H - IMG_PAD_BOTTOM;
    const float scale =
        (float)box_w / img_w < (float)box_h / img_h
            ? (float)box_w / img_w : (float)box_h / img_h;
    const int nw = (int)(img_w * scale + 0.5f);
    const int nh = (int)(img_h * scale + 0.5f);
    const int pad_x = (box_w - nw) / 2;
    const int pad_y = (box_h - nh) / 2;

    g_lbx_scale = scale;
    g_lbx_pad_x = pad_x;
    g_lbx_pad_y = pad_y;

    for (int y = 0; y < IMG_AREA_H; y++) {
        for (int x = 0; x < LCD_WIDTH; x++) {
            u8 *p = fb + ((u32)y * LCD_WIDTH + x) * 3;

            /* 右侧控制列：填底色即可，按钮由 draw_buttons 再画，字幕永远画这里 */
            if (x >= CTRL_X) {
                p[0] = HUD_BG_B;
                p[1] = HUD_BG_G;
                p[2] = HUD_BG_R;
                continue;
            }

            const int sx = x - pad_x;
            const int sy = y - pad_y;
            if (sx < 0 || sy < 0 || sx >= nw || sy >= nh) {
                p[0] = 114;
                p[1] = 114;
                p[2] = 114;
                continue;
            }
            const float fx = (float)sx / scale;
            const float fy = (float)sy / scale;
            int x0 = (int)fx;
            int y0 = (int)fy;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            if (x1 >= img_w)
                x1 = img_w - 1;
            if (y1 >= img_h)
                y1 = img_h - 1;
            const float dx = fx - x0;
            const float dy = fy - y0;
            /* IMG_BUF 是 4 字节/像素（BGRX），前三字节才是 BGR */
            const u8 *p00 = src + (u32)y0 * img_row + (u32)x0 * IMG_PIXEL_BYTES;
            const u8 *p10 = src + (u32)y0 * img_row + (u32)x1 * IMG_PIXEL_BYTES;
            const u8 *p01 = src + (u32)y1 * img_row + (u32)x0 * IMG_PIXEL_BYTES;
            const u8 *p11 = src + (u32)y1 * img_row + (u32)x1 * IMG_PIXEL_BYTES;
            const float w00 = (1.0f - dx) * (1.0f - dy);
            const float w10 = dx * (1.0f - dy);
            const float w01 = (1.0f - dx) * dy;
            const float w11 = dx * dy;
            for (int c = 0; c < 3; c++) {
                const float v = p00[c] * w00 + p10[c] * w10 +
                                p01[c] * w01 + p11[c] * w11;
                p[c] = (u8)(v + 0.5f);
            }
        }
    }
}

static void list_sd_root(void)
{
    DIR dir;
    FILINFO fno;
    FRESULT res = f_opendir(&dir, "/");
    if (res != FR_OK) {
        xil_printf("[SD] opendir err=%d\r\n", res);
        return;
    }
    xil_printf("[SD] root files:\r\n");
    int n = 0;
    for (;;) {
        res = f_readdir(&dir, &fno);
        if (res != FR_OK || fno.fname[0] == 0)
            break;
        xil_printf("  %s %u\r\n", fno.fname, (u32)fno.fsize);
        if (++n >= 40)
            break;
    }
    f_closedir(&dir);
}

static int load_sd_file(const char *path, u8 *dst, u32 want, const char *tag)
{
    int st = f_open(&fil, path, FA_READ);
    if (st != FR_OK) {
        xil_printf("[%s] open %s err=%d\r\n", tag, path, st);
        return st;
    }
    UINT br = 0;
    FRESULT read_st = f_read(&fil, dst, want, &br);
    u32 sz = (u32)f_size(&fil);
    f_close(&fil);
    if (read_st != FR_OK || br != want || sz != want) {
        xil_printf("[%s] %s err=%d size=%u read=%u want=%u\r\n",
                   tag, path, read_st, sz, br, want);
        return -1;
    }
    xil_printf("[%s] %s size=%u OK\r\n", tag, path, sz);
    return 0;
}

static int load_weights_pair(u8 *weights, u8 *params, int *w_ok, int *p_ok)
{
    const char *w_names[] = {"yolo_i8.bin", "yolo_w.bin", "weights.bin"};
    /* The xilffs BSP has LFN disabled, so use 8.3 short names first. */
    const char *p_names[] = {"yolo_q.bin", "params.bin",
                             "yolo_fixed_params_i8.bin"};
    *w_ok = 0;
    *p_ok = 0;
    for (int i = 0;
         i < (int)(sizeof(w_names) / sizeof(w_names[0])) && !*w_ok; i++)
        *w_ok = (load_sd_file(w_names[i], weights, WEIGHTS_BYTES,
                              "WEIGHTS") == 0);
    for (int i = 0;
         i < (int)(sizeof(p_names) / sizeof(p_names[0])) && !*p_ok; i++)
        *p_ok = (load_sd_file(p_names[i], params, FIXED_PARAM_BYTES,
                              "PARAMS") == 0);
    return (*w_ok && *p_ok) ? 0 : -1;
}

/* This scale blob is not passed to the accelerator.  Load and checksum it to
 * ensure all three SD card model files belong to this exported candidate. */
#if YOLO_REQUIRE_SCALES
static int load_act_scales(u8 *dst, int *ok)
{
    const char *names[] = {"yolo_s.bin", "scales.bin",
                           "yolo_act_scales_i8.bin"};
    *ok = 0;
    for (int i = 0; i < (int)(sizeof(names) / sizeof(names[0])) && !*ok; i++)
        *ok = (load_sd_file(names[i], dst, ACT_SCALE_BYTES, "SCALES") == 0);
    return *ok ? 0 : -1;
}
#endif

/* Board A/B switch.  0 = do not repaint the status line while the accelerator
   runs.  The old 100 ms repaint forced a D-cache flush of the framebuffer and
   kept the display VDMA reading DDR for the whole 305 ms inference, competing
   with the HLS feature reads.  Build with -DYOLO_LCD_PROGRESS=1 to restore the
   old behaviour for comparison. */
#ifndef YOLO_LCD_PROGRESS
#define YOLO_LCD_PROGRESS 0
#endif

/* 等待加速器完成时，读 PL 状态寄存器的最小间隔（微秒）。
   见 start_yolo_ip() 里"轮询节制"那段：死循环无脑读会打出几十万次总线访问，
   和 HLS 的 DDR 读抢互联，给"同一张图每次结果略不同"贡献随机性。 */
#define YOLO_POLL_US 50u

/* ---------------------------------------------------------------------------
 * 每次推理前把加速器的 DDR 暂存区清零
 *
 * WORK_BUF 是 HLS 设计的中转 feature map（YOLO_I8_WORK_BYTES = 1122688 字节，
 * 直接从生成的 yolo_layers_i8.h 取，见 WORK_BYTES 的定义），**CPU 从来不初始化它**
 * —— 只把地址写进寄存器。
 * 也就是说它装的是"上一轮跑剩的内容"，而不是零。
 *
 * 正常数据流是"先写后读"，暂存区初值无关紧要；但只要设计里有任何一条路径
 * **先读后写**（或者复位后残留一个没被覆盖的角落），结果就会依赖上一轮的残留 ——
 * 那就成了跨轮次的隐式状态，正是"同一张图每次结果略不同"的经典成因之一。
 *
 * 实测线索支持这个怀疑：连按 6 次 RUN，**第 1 次明显离群**
 * （max_conf 23%，之后 5 次都落在 11.5%~12.2%）。第 1 次看到的正是上电后的残留，
 * 后面几次看到的是"上一轮自己"，所以互相接近。
 *
 * 代价：903 KB 的 memset + flush，约 1~3 ms，占 ~1.2 s/张 的千分之几。
 * 换来"第 1 次和第 N 次完全同条件"，值得。
 *
 * ★ 这**只能**解释"第 1 次离群"，解释不了"第 2~6 次之间仍在飘"：
 *   那次实测里输入校验和 [IN] fnv 六次完全相同、而 [PRED] head 六次全不同，
 *   后者指向 PL 侧时序余量。见 01_说明书\运行结果不确定性_排查记录.md。
 * --------------------------------------------------------------------------- */
#ifndef YOLO_CLEAR_WORK
#define YOLO_CLEAR_WORK 1
#endif
/* MUST equal YOLO_I8_WORK_BYTES in the layer table.  Taking it from the generated
   header instead of typing the number is deliberate: this constant is used for
   the per-inference memset and the D-cache flush, so a stale value silently
   leaves a tail of the scratch buffer un-cleared and un-flushed.  On YOLO26 that
   tail is 219,520 bytes (1,122,688 vs yolov8n's 903,168) and it would show up on
   the board as boxes that disagree with csim. */
#define WORK_BYTES YOLO_I8_WORK_BYTES
#if ((WORK_BUF) + (WORK_BYTES) > (PRED_BUF))
#error "WORK_BUF + WORK_BYTES 会压到 PRED_BUF，必须先调整内存布局。"
#endif

/* Profiling build switch.  1 = add the li0/li1 registers (profiling IP only) and
   run the network as YOLO_PROFILE_NGRP slices, printing one UART line per slice.
   The delivered application keeps this at 0. */
#ifndef YOLO_PROFILE_RANGES
#define YOLO_PROFILE_RANGES 0
#endif
#define YOLO_PROFILE_NGRP   15

/* ---------------------------------------------------------------------------
 * 滤波实现选择：
 *   USE_PL_FILTER = 1 -> 用 PL 里的 yolo_filter3x3 IP（**只有 919 的比特流**才有）
 *   USE_PL_FILTER = 0 -> 用 PS 软件滤波 filter_sw.c（921 的比特流，默认）
 *
 * 两者**逐位一致**（filter_sw.c 已用主机逐字节比对 40/40 张确认），
 * 但**必须与比特流匹配**：拿 921 的比特流去调 0x41000000 会读到默认从设备。
 * 921 之所以把滤波搬走：那个 IP 的 2.7K LUT / 15 DSP 把本来只有 +0.145 ns
 * 余量的 YOLO IP 推成了 −0.230 ns / 222 个失败端点。搬走之后时序能回到闭合。
 * --------------------------------------------------------------------------- */
#ifndef USE_PL_FILTER
#define USE_PL_FILTER 0
#endif

#if USE_PL_FILTER
/* ---------------------------------------------------------------------------
 * 斑点抑制预处理：亮度合成 + 3x3 中值（IP: yolo_filter3x3）
 *
 * 就地滤波：in/out 都指向 IMG_BUF。IP 内部的处理顺序保证不会回读已覆盖的数据
 * （处理输出行 oy 时读输入行 oy+1，且行内写比读滞后 1 个像素）。
 * 前置条件是缓冲已按 BGRX（4 字节/像素）排布，见 load_bmp_from_sd。
 *
 * 返回 0 成功；-1 超时；-2 图像尺寸超出 IP 支持范围，由调用方跳过。
 *
 * 注意：本函数必须定义在 `#if YOLO_PROFILE_RANGES` 之外 ——
 *       该宏默认为 0，放进去会被条件编译掉，导致链接期 undefined reference。
 * --------------------------------------------------------------------------- */
static int filter_apply(void)
{
    /* IP 内部把 width/height 钳位到 FILT_MAX_W/FILT_MAX_H，超限的图只会被处理一部分，
       结果不一致。所以超限时直接跳过滤波，由调用方决定怎么提示。 */
    if (img_w < 3 || img_h < 3 || img_w > FILTER_MAX_W || img_h > FILTER_MAX_H)
        return -2;
    const u32 base = FILTER_BASEADDR;
    XTime t0, now;
    Xil_Out32(base + FILTER_REG_CTRL, 0x00);
    Xil_Out32(base + FILTER_REG_IN_IMG, (u32)IMG_BUF);
    Xil_Out32(base + FILTER_REG_IN_IMG + 4, 0);
    Xil_Out32(base + FILTER_REG_OUT_IMG, (u32)IMG_BUF);   /* 就地 */
    Xil_Out32(base + FILTER_REG_OUT_IMG + 4, 0);
    Xil_Out32(base + FILTER_REG_WIDTH, (u32)img_w);
    Xil_Out32(base + FILTER_REG_HEIGHT, (u32)img_h);
    Xil_Out32(base + FILTER_REG_CTRL, 0x01);
    XTime_GetTime(&t0);
    while ((Xil_In32(base + FILTER_REG_CTRL) & 0x02) == 0) {
        XTime_GetTime(&now);
        if (elapsed_ms(t0, now) >= FILTER_TIMEOUT_MS) {
            xil_printf("[FILTER] TIMEOUT ctrl=0x%08X\r\n",
                       (unsigned)Xil_In32(base + FILTER_REG_CTRL));
            return -1;
        }
    }
    return 0;
}
#endif /* USE_PL_FILTER */

/* Run the accelerator over the layer slice [li0, li1).
 *
 * This used to be compiled only under YOLO_PROFILE_RANGES (the profiling IP
 * build), but for YOLO26 the range argument is ALWAYS present in the hardware and
 * is the mechanism by which the PS interleaves its own operators, so it is now
 * unconditional.  fb is accepted for call-site symmetry and otherwise unused:
 * the display is not repainted between slices. */
#if YOLO_PROFILE_RANGES
#error "YOLO_PROFILE_RANGES requires a different IP with li0/li1 registers"
#endif

#if 0
static int start_yolo_ip_range(u8 *fb, int li0, int li1)
{
    const u32 base = YOLO_BASEADDR;
    XTime t0, now;
    Xil_Out32(base + YOLO_REG_CTRL, 0x00);
    Xil_Out32(base + YOLO_REG_INPUT, (u32)INPUT_BUF);
    Xil_Out32(base + YOLO_REG_INPUT + 4, 0);
    Xil_Out32(base + YOLO_REG_WEIGHTS, (u32)WEIGHTS_BUF);
    Xil_Out32(base + YOLO_REG_WEIGHTS + 4, 0);
    Xil_Out32(base + YOLO_REG_ACT_SCALES, (u32)FIXED_PARAMS_BUF);
    Xil_Out32(base + YOLO_REG_ACT_SCALES + 4, 0);
    Xil_Out32(base + YOLO_REG_WORK, (u32)WORK_BUF);
    Xil_Out32(base + YOLO_REG_WORK + 4, 0);
    Xil_Out32(base + YOLO_REG_PRED, (u32)PRED_BUF);
    Xil_Out32(base + YOLO_REG_PRED + 4, 0);
    Xil_Out32(base + YOLO_REG_LI0, (u32)li0);
    Xil_Out32(base + YOLO_REG_LI1, (u32)li1);
    (void)fb;
    Xil_Out32(base + YOLO_REG_CTRL, 0x01);
    XTime_GetTime(&t0);
    while ((Xil_In32(base + YOLO_REG_CTRL) & 0x02) == 0) {
        XTime_GetTime(&now);
        if (elapsed_ms(t0, now) >= 30000u) {
            xil_printf("[PROF] slice %d-%d TIMEOUT ctrl=0x%08X\r\n",
                       li0, li1, (unsigned)Xil_In32(base + YOLO_REG_CTRL));
            return -1;
        }
    }
    return 0;
}
#endif

#if 0
static int profile_run_all(u8 *fb)
{
    /* YOLOv8n-only profiling helper: it assumes a ~80-layer table split into
       5-layer slices, which is wrong for YOLO26's 115 layers.  Keep it behind the
       flag (default off) so nobody mistakes its numbers for a real profile. */
    static const int bounds[YOLO_PROFILE_NGRP + 1] =
        {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75};
    u32 tot = 0;
    int rc = 0;
    xil_printf("[PROF] %d slices of 5 layers\r\n", YOLO_PROFILE_NGRP);
    for (int g = 0; g < YOLO_PROFILE_NGRP; g++) {
        XTime a, b;
        XTime_GetTime(&a);
        int r = start_yolo_ip_range(fb, bounds[g], bounds[g + 1]);
        XTime_GetTime(&b);
        u32 ms = elapsed_ms(a, b);
        tot += ms;
        if (r != 0)
            rc = -1;
        xil_printf("[PROF] li %2d-%2d : %5u ms  rc=%d\r\n",
                   bounds[g], bounds[g + 1], (unsigned)ms, r);
    }
    xil_printf("[PROF] TOTAL %u ms\r\n", (unsigned)tot);
    return rc;
}
#endif

static int start_yolo_ip(u8 *fb)
{
    const u32 base = YOLO_BASEADDR;
    XTime t0, now, t_poll;
#if YOLO_LCD_PROGRESS
    u32 last_ms = 0;
#endif
    Xil_Out32(base + YOLO_REG_CTRL, 0x00);
    Xil_Out32(base + YOLO_REG_INPUT, (u32)INPUT_BUF);
    Xil_Out32(base + YOLO_REG_INPUT + 4, 0);
    Xil_Out32(base + YOLO_REG_WEIGHTS, (u32)WEIGHTS_BUF);
    Xil_Out32(base + YOLO_REG_WEIGHTS + 4, 0);
    Xil_Out32(base + YOLO_REG_ACT_SCALES, (u32)FIXED_PARAMS_BUF);
    Xil_Out32(base + YOLO_REG_ACT_SCALES + 4, 0);
    Xil_Out32(base + YOLO_REG_WORK, (u32)WORK_BUF);
    Xil_Out32(base + YOLO_REG_WORK + 4, 0);
    Xil_Out32(base + YOLO_REG_PRED, (u32)PRED_BUF);
    Xil_Out32(base + YOLO_REG_PRED + 4, 0);
    if (fb) {
        snprintf(g_status_txt, sizeof(g_status_txt), "YOLO RUN 0 MS");
        g_status_r = 255; g_status_g = 255; g_status_b = 0;
        hud_status_only(fb);
    }
    Xil_Out32(base + YOLO_REG_CTRL, 0x01);
    XTime_GetTime(&t0);
    XTime_GetTime(&t_poll);
    for (;;) {
        XTime_GetTime(&now);
        const u32 ms = elapsed_ms(t0, now);
#if YOLO_LCD_PROGRESS
        if (fb && ms - last_ms >= 100) {
            snprintf(g_status_txt, sizeof(g_status_txt), "YOLO RUN %u MS",
                     (unsigned int)ms);
            hud_status_only(fb);
            last_ms = ms;
        }
#endif
        if (ms >= YOLO_TIMEOUT_MS) {
            xil_printf("[YOLO] ap_done timeout ctrl=0x%08X\r\n",
                       Xil_In32(base + YOLO_REG_CTRL));
            return -1;
        }
        /* -------------------------------------------------------------------
         * 轮询节制：**不要**死循环无脑读 AXI-Lite 控制寄存器。
         *
         * 原来那一版在这 194 ms 里会打出几十万次 PL 总线访问；这些访问要穿过
         * ps7_0_axi_periph/中央互联，而 HLS 加速器正在用 HP 口猛读 DDR。
         * 两边抢互联和 DDR 控制器，会让"加速器跑的过程"本身带有随机成分 ——
         * 这正是"同一张图连按 RUN，结果差一点点"的候选来源之一。
         *
         * 改成每 YOLO_POLL_US 才读一次：完成判定最多晚这么多（相对 194 ms 可忽略），
         * 但总线压力降两个数量级，而且**每次运行的访问模式变得高度一致**。
         * 诊断"结果不确定"之前，先把这种随机的环境干扰拿掉。
         * ----------------------------------------------------------------- */
        if (elapsed_us(t_poll, now) >= YOLO_POLL_US) {
            XTime_GetTime(&t_poll);
            if (Xil_In32(base + YOLO_REG_CTRL) & 0x02)
                break;
        }
    }
    XTime_GetTime(&now);
    g_pl_us=elapsed_us(t0,now);
    return 0;
}

/* YOLO26 Conv1x1 fit: 111 PL layers, one accelerator call.  The role header
 * is shared with HLS; reject any unexpected PS-side operation. */
static int count_ps_layers(void)
{
    int n = 0;
    for (int i = 0; i < (int)YOLO_I8_NUM_LAYERS; i++) {
        const int op = yolo_layers_i8[i].op;
        if (op != OP_COPY && op != OP_CONV && op != OP_SPLIT &&
            op != OP_MAXPOOL) {
            xil_printf("[YOLO26] unsupported layer %d op=%d\r\n", i, op);
            n++;
            continue;
        }
        if (yolo_layer_index_is_ps(i)) {
            if (n < 8)
                xil_printf("[YOLO26] layer %d (op=%u) is declared PS-side\r\n",
                           i, (unsigned)yolo_layers_i8[i].op);
            n++;
        }
    }
    return n;
}

static int start_yolo_ip(u8 *fb);

static int run_yolo26_inference(u8 *fb)
{
    XTime t0, t1;

    /* 表里有 PS 层，而这个 build 已经没有链接任何 PS 算子了（yolo_attn_ps.h /
       yolo_dw_ps.h 都不再包含）。这时唯一安全的做法是拒绝运行：内核会跳过它不认识
       的层，pred 会"看起来像那么回事"但是错的，那正是这个项目反复踩的坑。 */
    if (count_ps_layers() != 0) {
        xil_printf("[YOLO26] table has PS-side layers but this build links no PS "
                   "operators; refusing to run\r\n");
        return -1;
    }

    /* 921-style single-call IP: five pointer registers, no layer range. */
    XTime_GetTime(&t0);
    const int rc = start_yolo_ip(fb);
    XTime_GetTime(&t1);
    const u32 ms = elapsed_ms(t0, t1);
    if (rc != 0) {
        xil_printf("[YOLO26] PL range [0,%u) failed\r\n",
                   (unsigned)YOLO_I8_NUM_LAYERS);
        return -1;
    }
    xil_printf("[YOLO26] slice 0: PL[0,%u)=%u ms  PS=0 ms\r\n",
               (unsigned)YOLO_I8_NUM_LAYERS, (unsigned)ms);
    xil_printf("[YOLO26] total PL=%u ms  PS=0 ms  (sum %u ms)\r\n",
               (unsigned)ms, (unsigned)ms);
    return 0;
}

/* Optional board self-test compares a preprocessed INT8 input with the
 * accelerator's INT16 Q10 prediction. It bypasses BMP reading and filtering
 * to isolate the PS/PL boundary. The public package omits the image-derived
 * reference vectors; provide an authorized pair before enabling this mode.
 */
#if YOLO_BOOT_SELFTEST
static void yolo26_selftest(u8 *fb)
{
    s16 *pred = (s16 *)PRED_BUF;
    XTime t0, t1;

    xil_printf("[SELFTEST] embedded vectors: %u input bytes, %u ref values\r\n",
               (unsigned)YOLO_VERIFY_INPUT_BYTES,
               (unsigned)YOLO_VERIFY_REF_ELEMS);
    memcpy((void *)INPUT_BUF, yolo_verify_input, YOLO_VERIFY_INPUT_BYTES);
    Xil_DCacheFlushRange((INTPTR)INPUT_BUF, YOLO_VERIFY_INPUT_BYTES);
    xil_printf("[SELFTEST] input fnv=0x%08X\r\n",
               (unsigned)fnv1a32((const u8 *)INPUT_BUF, YOLO_VERIFY_INPUT_BYTES));

    prepare_pred_buffer(pred);
    /* Known state, timed OUTSIDE the inference so the ms is comparable to a RUN. */
    memset((void *)WORK_BUF, 0, WORK_BYTES);
    Xil_DCacheFlushRange((INTPTR)WORK_BUF, WORK_BYTES);

    XTime_GetTime(&t0);
    const int rc = run_yolo26_inference(fb);
    XTime_GetTime(&t1);
    const u32 ms = elapsed_ms(t0, t1);
    if (rc != 0) {
        xil_printf("[SELFTEST] FAIL: inference returned %d after %u ms\r\n", rc,
                   (unsigned)ms);
        show_status(fb, "SELFTEST FAIL", 255, 0, 0);
        return;
    }
    Xil_DCacheInvalidateRange((INTPTR)PRED_BUF, PRED_BYTES_ALIGNED);

    /* ----------------------------------------------------------------------
     * Dump the self-test head output as well.
     *
     * This is the control for the real-image dump: the same binary, the same
     * weights and the same layer schedule, differing only in the input tensor.
     * The input here is an embedded vector with a host-known checksum, so
     * `diff_board_dump.py` can check this pair first -- if the self-test pair
     * matches and the BMP pair does not, the accelerator is exonerated by
     * construction and the input tensor is the whole story; if the self-test
     * pair does not match either, the harness itself is suspect and the BMP
     * comparison must not be read as evidence.
     * -------------------------------------------------------------------- */
    {
        FIL sf;
        if (f_open(&sf, "st_pred.bin", FA_WRITE | FA_CREATE_ALWAYS) == FR_OK) {
            UINT bw = 0;
            const FRESULT wr = f_write(&sf, (const void *)PRED_BUF, PRED_BYTES, &bw);
            f_close(&sf);
            xil_printf("[SELFTEST] dumped %u bytes (st=%d)\r\n", (unsigned)bw, (int)wr);
        }
    }

    const float conf_logit_025 = -1.0986123f;
    const int anchors = YOLO_ANCHORS;
    const int bch = YOLO_BOX_CHANNELS;
    int active = 0, hw_argmax = 0, ref_argmax = 0;
    long active_max = 0, all_max = 0;
    int hw_top = -32768, ref_top = -32768;

    for (int a = 0; a < anchors; a++) {
        const int ci = bch * anchors + a;
        const int hw_c = (int)pred[ci];
        const int rf_c = (int)yolo_verify_ref_q10[ci];
        if (hw_c > hw_top) { hw_top = hw_c; hw_argmax = a; }
        if (rf_c > ref_top) { ref_top = rf_c; ref_argmax = a; }
        const int is_active = ((float)rf_c / 1024.0f) > conf_logit_025;
        if (is_active)
            active++;
        for (int c = 0; c < bch + 1; c++) {
            const int i = c * anchors + a;
            long d = (long)pred[i] - (long)yolo_verify_ref_q10[i];
            if (d < 0) d = -d;
            if (d > all_max) all_max = d;
            if (is_active && d > active_max)
                active_max = d;
        }
    }

    xil_printf("[SELFTEST] %u ms, active anchors=%d (csim 10), "
               "active max |d|=%ld LSB (%.3f Q10, csim 0.227)\r\n",
               (unsigned)ms, active, active_max, (double)active_max / 1024.0);
    xil_printf("[SELFTEST] all max |d|=%ld LSB; hw argmax=%d q10=%d; "
               "ref argmax=%d q10=%d\r\n",
               all_max, hw_argmax, hw_top, ref_argmax, ref_top);
    if (active > 0 && active_max < 512) {
        xil_printf("[SELFTEST] PASS (bar: active max |d| < 512 LSB)\r\n");
        show_status(fb, "SELFTEST PASS", 0, 255, 0);
    } else {
        xil_printf("[SELFTEST] FAIL (bar: active max |d| < 512 LSB)\r\n");
        show_status(fb, "SELFTEST FAIL", 255, 0, 0);
    }
}
#endif

/* ---------------------------------------------------------------------------
 * 以前的"滤波模式行"已经被 HUD 面板取代（见 draw_hud）。
 * 这里只留一个空壳给旧的调用点，避免遗漏，实际不做任何事。
 * ------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------
 * 把 IMG_BUF 的内容渲染到帧缓冲：
 *   letterbox(图片区) -> HUD 文字面板 -> 按钮
 * 换图和 FILTER 切换都走这里，保证两条路径画出来的东西完全一样。
 * HUD 一定画在 letterbox 之后（letterbox 会把面板底色也铺一遍）。
 * --------------------------------------------------------------------------- */
static void render_image(u8 *fb, const char *status)
{
    XTime w0, w1;
    XTime_GetTime(&w0);
    letterbox_to_fb(fb);
    ui_cache_background(fb);
    XTime_GetTime(&w1);
    pf_add(PF_LBX, w0, w1);

    XTime_GetTime(&w0);
    show_status(fb, status, 255, 255, 0);
    draw_buttons(fb);
    Xil_DCacheFlushRange((INTPTR)FB_ADDR, FRAME_SIZE);
    XTime_GetTime(&w1);
    pf_add(PF_FLUSH_FB, w0, w1);
}

/* ---------------------------------------------------------------------------
 * 按 g_filter_mode 决定 g_img_src 指向谁：
 *   OFF        -> g_img_src = IMG_BUF，**什么都不用做**（原图一直好好在那儿）
 *   其它档位    -> 用 PS 软件滤波把 IMG_BUF 处理到 FLT_BUF，g_img_src = FLT_BUF
 *
 * 与 PL 版的两个区别（都是搬 PS 之后白赚的）：
 *   ① 不再需要任何 cache 维护 —— 滤波走 CPU，D-cache 天生一致；
 *      PL 版走 S_AXI_HP1 绕开 D-cache，每次都要 flush 2.12MB + invalidate 2.12MB。
 *   ② 切 OFF 是零成本（只是改个指针），PL 版还得 memcpy 回原图。
 *
 * 返回 0 成功；-2 表示尺寸超出软件版支持范围，此时退回原图（g_img_src = IMG_BUF）。
 * --------------------------------------------------------------------------- */
static int apply_filter_mode(void)
{
    XTime ft0, ft1;
    int frc;

    if (g_filter_mode == FILT_MODE_OFF) {
        g_filter_us=0;
        g_img_src = (const u8 *)IMG_BUF;
        g_filter_skipped = 0;
        xil_printf("[FILTER] BYPASSED (raw BGR = ablation variant \"original\")\r\n");
        return 0;
    }

    XTime_GetTime(&ft0);
    frc = filter_sw_apply(g_filter_mode, (const u8 *)IMG_BUF, (u8 *)FLT_BUF,
                          img_w, img_h, img_row);
    XTime_GetTime(&ft1);
    pf_add(PF_FILTER, ft0, ft1);
    g_filter_us=elapsed_us(ft0,ft1);
    g_flt_mode=frc==0?g_filter_mode:-1;g_flt_us=g_filter_us;

    if (frc == 0) {
        g_img_src = (const u8 *)FLT_BUF;
        g_filter_skipped = 0;
        xil_printf("[FILTER] sw %s %dx%d in %u ms (%s)\r\n",
                   filter_sw_mode_name(g_filter_mode), img_w, img_h,
                   (unsigned)elapsed_ms(ft0, ft1),
                   filter_sw_mode_desc(g_filter_mode));
    } else {
        g_img_src = (const u8 *)IMG_BUF;   /* 退回原图，不装作滤过了 */
        g_filter_skipped = 1;
        xil_printf("[FILTER] sw skipped rc=%d (size %dx%d, max %dx%d) "
                   "-- raw image is being used, NOT filtered\r\n",
                   frc, img_w, img_h, FILT_SW_MAX_W, FILT_SW_MAX_H);
    }
    return frc;
}

/* ---------------------------------------------------------------------------
 * 换图：读 BMP -> 存原图副本 -> 按当前模式处理 -> letterbox -> 画界面
 *
 * 滤波现在是 PS 软件、且**非破坏性**：原图一直留在 IMG_BUF，
 * 滤波结果写 FLT_BUF。所以这里既不需要存副本，也不需要 cache 维护。
 * 返回 0 表示新图已显示。
 * --------------------------------------------------------------------------- */
static int load_and_show(u8 *fb, int idx)
{
    char path[16];
    int bmp_st;
    XTime w0, w1;

    ui_reset_image();
    pf_set_img(idx);
    sprintf(path, "%d.bmp", idx);
    XTime_GetTime(&w0);
    bmp_st = load_bmp_from_sd(path, fb);
    if (bmp_st != 0) {
        g_bmp_ok=0;
        XTime_GetTime(&w1);
        pf_add(PF_SD, w0, w1);
        xil_printf("BMP load err=%d -> color bars\r\n", bmp_st);
        fill_color_bars(fb);
        if (bmp_st == -2)
            show_status(fb, "BMP FORMAT", 255, 0, 0);
        else if (bmp_st == -3)
            show_status(fb, "IMG TOO BIG", 255, 0, 0);
        else
            show_status(fb, "BMP FAIL", 255, 0, 0);
        draw_buttons(fb);
        flush_band(CTRL_BTN_Y0, LCD_HEIGHT - CTRL_BTN_Y0);
        return -1;
    }

    /* 软件滤波是非破坏性的：原图一直留在 IMG_BUF，不需要再存副本 */
    XTime_GetTime(&w1);
    pf_add(PF_SD, w0, w1);

    /* -----------------------------------------------------------------------
     * 载入后的原图校验和（诊断用，只打印不做事）
     *
     * 现场排"某几张图 max_conf=0"这类问题时，这一个数就能把链路上游一段切开：
     *   ① 和 PC 侧算出来的期望值**一致**
     *        -> SD 读 + BGRX 解包这一段是好的，问题在下游（预处理或 YOLO IP）
     *   ② **不一致**
     *        -> 卡上文件坏了 / 拷贝不完整 / 板子读卡出错，下面的推理自然全错
     * 期望值由 filter_study\bmp_bgrx_fnv.py 用**逐字节复刻 load_bmp_from_sd**
     * 的方式算出（输出 BGRX 就是每像素 B,G,R,0，行序自上而下）。
     * 放在滤波之前算，所以这里哈希的是**未滤波的原图**。
     * -------------------------------------------------------------------- */
#if YOLO_RUN_DIAGNOSTICS
    xil_printf("[BMP] %s %dx%d row=%d fnv=0x%08X\r\n",
               path, img_w, img_h, img_row,
               (unsigned)fnv1a32((const u8 *)IMG_BUF,
                                 (u32)img_row * (u32)img_h));
#endif

    g_bmp_ok=1;
    apply_filter_mode();

    /* 新图 = 上一次的推理结果作废，HUD 上的 CONF 必须清零重来 */
    g_have_infer = 0;
    g_box_n = 0;
    g_max_conf_pct = 0;

    render_image(fb, "YOLO READY");

    xil_printf("Letterbox -> FB 800x480\r\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 单次推理：预处理 -> 启动 IP -> 取结果 -> NMS -> 画框
 * 结束后重画按钮条（框可能盖到按钮上），并只把改动过的区域刷回 DDR。
 * --------------------------------------------------------------------------- */
static void run_inference_once(u8 *fb, int weights_ok)
{
    s16 *pred = (s16 *)PRED_BUF;
    u8 *input = (u8 *)INPUT_BUF;
    YoloBox boxes[YOLO_MAX_BOXES];
    int box_n = 0;
    u32 infer_ms = 0;
    int infer_ok = 0;
    XTime t0, t1, p0, p1, run0,run1;
    if(!g_bmp_ok){show_status(fb,"LOAD IMAGE FIRST",255,120,60);return;}
    XTime_GetTime(&run0);
    ui_restore_background(fb);
    g_have_infer=0;g_source_box_n=0;
    g_pre_us=g_post_us=g_pl_us=g_call_us=g_wall_us=0;

    xil_printf("[STUDY] image=%d mode=%s\r\n", g_img_idx, filter_sw_mode_name(g_filter_mode));

    if (!weights_ok) {
        show_status(fb, "MODEL BAD", 255, 0, 0);
        return;
    }

    XTime_GetTime(&p0);
    if (g_bmp_ok)
        yolo_preprocess_i8_bgr_row(g_img_src, img_w, img_h,
                                   img_row, input,
                                   YOLO_INPUT_W, YOLO_INPUT_H);
    else
        yolo_preprocess_i8_bgr(fb, LCD_WIDTH, LCD_HEIGHT, input,
                               YOLO_INPUT_W, YOLO_INPUT_H);
    Xil_DCacheFlushRange((INTPTR)INPUT_BUF, INPUT_BYTES);

    /* -----------------------------------------------------------------------
     * 输入张量的校验和：把"输入不一样"和"加速器算得不一样"分开
     *
     * 排"同一张图连按 RUN 结果差一点点"时，链条上有两个可疑环节：
     *   ① 喂给加速器的张量本身就不一样（SD 读、滤波、预处理、cache）
     *   ② 张量一样、但 PL 算出来的不一样（时序 / 竞态）
     * 只印一个 [PRED] 校验和是分不开这两者的 —— 必须两边都印。
     *
     * 先 invalidate 再算，读到的才是 **DDR 里真实躺着的东西**，
     * 也就是 PL 通过 HP 口真正会读到的东西（而不是 CPU cache 里的副本）。
     * INPUT_BYTES = 224*224*3 = 150,528 是 32 的整数倍，invalidate 长度天然对齐。
     *
     * 判据（配合 [PRED] head）：
     *   [IN] fnv 变   -> 问题在 PS 侧，PL 无辜，去查上游
     *   [IN] fnv 不变、[PRED] head 变 -> **PL 真的不确定**（时序没真正收敛）
     *   两个都不变    -> 硬件是确定的
     * -------------------------------------------------------------------- */
#if YOLO_RUN_DIAGNOSTICS
    Xil_DCacheInvalidateRange((INTPTR)INPUT_BUF, INPUT_BYTES);
    xil_printf("[IN]  fnv=0x%08X bytes=%u\r\n",
               (unsigned)fnv1a32((const u8 *)INPUT_BUF, INPUT_BYTES),
               (unsigned)INPUT_BYTES);
    /* ------------------------------------------------------------------------
     * Per-channel statistics of the tensor that is about to be fed to the PL.
     *
     * The hardware self-test PASSES (261 LSB vs csim's 239) on a tensor produced
     * by the host pipeline, yet this path yields all-negative head logits
     * (positive=0, max=-7096) on the same weights and the same layer table.  The
     * accelerator is therefore fine and the difference must be IN THIS TENSOR.
     * These three lines make that difference visible: a channel-order swap shows
     * up as ch0 > ch2 where the reference has ch0 < ch2 (mean 33.0/31.8/30.9 for
     * the self-test vector), and a scaling error shows up as a mean far from ~32.
     * ------------------------------------------------------------------------ */
    {
        const u8 *p = (const u8 *)INPUT_BUF;
        const unsigned plane = (unsigned)YOLO_INPUT_W * (unsigned)YOLO_INPUT_H;
        for (int c = 0; c < 3; c++) {
            const u8 *q = p + (unsigned)c * plane;
            unsigned lo = 255, hi = 0;
            unsigned long sum = 0;
            for (unsigned i = 0; i < plane; i++) {
                const unsigned v = q[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
                sum += v;
            }
            xil_printf("[IN]  ch%d min=%3u max=%3u mean=%6u.%02u\r\n", c, lo, hi,
                       (unsigned)(sum / plane), (unsigned)((sum % plane) * 100 / plane));
        }
        xil_printf("[IN]  reference vector: ch0 mean 32.99, ch1 31.76, ch2 30.92\r\n");

        /* ------------------------------------------------------------------
         * Dump the tensor to the SD card.
         *
         * The measurements no longer fit the code: preprocess_i8_bgr_row pads
         * with 114 over 6 columns each side for a 709x749 source, which puts the
         * mean at ~27.3, yet the board reports 21.95 -- i.e. as if there were
         * almost no padding at all.  Rather than keep inferring what the
         * pipeline does, write the bytes out and diff them against the host
         * reconstruction.  One file, no ambiguity.
         * ---------------------------------------------------------------- */
        FIL df;
        char dump_name[24];
        snprintf(dump_name, sizeof(dump_name), "i%dm%d.bin", g_img_idx, g_filter_mode);
        if (f_open(&df, dump_name, FA_WRITE | FA_CREATE_ALWAYS) == FR_OK) {
            UINT bw = 0;
            const FRESULT wr = f_write(&df, (const void *)INPUT_BUF, INPUT_BYTES, &bw);
            f_close(&df);
            xil_printf("[IN]  dumped %u bytes to %s (st=%d)\r\n",
                       (unsigned)bw, dump_name, (int)wr);
        } else {
            xil_printf("[IN]  could not open %s\r\n", dump_name);
        }
    }

    #endif
    prepare_pred_buffer(pred);

#if YOLO_CLEAR_WORK
    /* 把加速器暂存区恢复成"已知初值"，消掉跨轮次的隐式状态（见 YOLO_CLEAR_WORK） */
    memset((void *)WORK_BUF, 0, WORK_BYTES);
    Xil_DCacheFlushRange((INTPTR)WORK_BUF, WORK_BYTES);
#endif
    XTime_GetTime(&p1);
    pf_add(PF_PRE, p0, p1);
    g_pre_us=elapsed_us(p0,p1);

    XTime_GetTime(&t0);
    const int ip_rc = run_yolo26_inference(fb);
    if (ip_rc == 0) {
        XTime_GetTime(&t1);
        infer_ms = elapsed_ms(t0, t1);
        g_call_us=elapsed_us(t0,t1);
        pf_add(PF_INFER, t0, t1);
        infer_ok = 1;
        xil_printf("[YOLO] inference %u ms\r\n", (unsigned int)infer_ms);

        XTime_GetTime(&p0);
        /* 用**补齐到整行**的长度 invalidate：PRED_BYTES 本身不是 32 的整数倍，
           用原长度会让尾部那行走 clean&invalidate 而不是纯 invalidate。见
           文件上方 PRED_BYTES_ALIGNED 的说明。 */
        Xil_DCacheInvalidateRange((INTPTR)PRED_BUF, PRED_BYTES_ALIGNED);
#if YOLO_RUN_DIAGNOSTICS
        log_pred_stats(pred);

        /* -------------------------------------------------------------------
         * 预测缓冲的校验和（诊断用，只打印不做事）
         *
         * 排"同一张图连按 RUN 结果不一样"这类问题时，这个数是最直接的判据：
         *
         *   fnv   = 整个 PRED_BUF
         *   head  = **cache 行对齐的前缀**（133,760 字节，IP 真正算出来的部分）
         *   tail  = 最后不足一行的那 10 字节
         *
         *   连按几次 RUN（**中途千万别换滤波档**，换档会合法地改变输入张量）：
         *     head 变            -> PL 侧确实不确定：时序没真正收敛，或者握手竞态
         *     head 不变 / tail 变 -> 数据本身是确定的，问题在 cache 尾行没读干净
         *     两个都不变          -> 硬件是确定的，读数差异来自别处（逻辑/显示）
         *
         * 配合 [BMP] fnv 一起看：BMP 稳定而 PRED 不稳，就锁在 PL 里了。
         * ----------------------------------------------------------------- */
        {
            const u32 alen = (u32)PRED_BYTES & ~31u;
            const u32 tlen = (u32)PRED_BYTES - alen;
            xil_printf("[PRED] fnv=0x%08X head=0x%08X tail=0x%08X (alen=%u tlen=%u)\r\n",
                       (unsigned)fnv1a32((const u8 *)PRED_BUF, PRED_BYTES),
                       (unsigned)fnv1a32((const u8 *)PRED_BUF, alen),
                       (unsigned)fnv1a32((const u8 *)PRED_BUF + alen, tlen),
                       (unsigned)alen, (unsigned)tlen);
        }

        /* 最强锚点的**原始 q10 值**和它在哪一格。
           只印百分比会把 0.1449 / 0.1451 这种边界差异藏起来 —— 而
           "CONF 在 14 和 15 之间跳" 恰恰就是这么来的。印出 q10 才能看出
           到底差了多少、以及最强锚点有没有换位置。 */
        {
            const s16 *conf = pred + YOLO_BOX_CHANNELS * YOLO_ANCHORS;
            int am = 0;
            for (int a = 1; a < YOLO_ANCHORS; a++)
                if (conf[a] > conf[am])
                    am = a;
            xil_printf("[PRED] argmax=%d q10=%d untouch_tail=%d\r\n",
                       am, (int)conf[am], (int)conf[YOLO_ANCHORS - 1]);
        }

        /* -------------------------------------------------------------------
         * Dump the head output to the SD card.
         *
         * With in_dump.bin this closes the loop completely: the host can run
         * the same INT8 schedule on the very bytes the board fed in, and
         * compare the resulting pred buffer against the board's.  That
         * separates the two remaining possibilities outright --
         *   pred differs from the host  -> the PL/tiling is at fault
         *   pred matches the host       -> the accelerator is right and the
         *                                  defect is in decode/NMS/threshold
         * ----------------------------------------------------------------- */
        {
            FIL pf;
            char dump_name[24];
            snprintf(dump_name, sizeof(dump_name), "p%dm%d.bin", g_img_idx, g_filter_mode);
            if (f_open(&pf, dump_name, FA_WRITE | FA_CREATE_ALWAYS) == FR_OK) {
                UINT bw = 0;
                const FRESULT wr = f_write(&pf, (const void *)PRED_BUF, PRED_BYTES, &bw);
                f_close(&pf);
                xil_printf("[PRED] dumped %u bytes to %s (st=%d)\r\n",
                           (unsigned)bw, dump_name, (int)wr);
            }
        }

#endif
        box_n = yolo_decode_nms(pred,
                                g_bmp_ok ? img_w : LCD_WIDTH,
                                g_bmp_ok ? img_h : LCD_HEIGHT,
                                YOLO_CONF_TH, YOLO_IOU_TH,
                                boxes, YOLO_MAX_BOXES);
        g_source_box_n=box_n;
        memcpy(g_source_boxes,boxes,box_n*sizeof(boxes[0]));
        if(!g_roi_valid && box_n>0){
            int best=0;for(int i=1;i<box_n;++i)if(boxes[i].score>boxes[best].score)best=i;
            g_roi_valid=ui_roi_from_box(&g_roi,&boxes[best],img_w,img_h);g_roi_manual=0;
        }
        /* 用 letterbox 时**同一组** scale/pad_x/pad_y 映射，框才落在图片区里 */
        if (g_bmp_ok)
            yolo_map_boxes_to_rect(boxes, box_n,
                                   g_lbx_scale, g_lbx_pad_x, g_lbx_pad_y);
        {
            u32 mc = (u32)(yolo_max_conf(pred) * 100.0f + 0.5f);
            xil_printf("[YOLO] boxes=%d max_conf=%d%%\r\n", box_n, mc);
            /* 上屏用：这就是 HUD 里 "CONF nn" 那个读数，和日志同源 */
            g_max_conf_pct = (int)mc;
            g_box_n = box_n;
            g_have_infer = 1;
        }
        for (int i = 0; i < box_n; i++) {
            u32 score = (u32)(boxes[i].score * 100.0f + 0.5f);
            xil_printf("  %d: (%d,%d)-(%d,%d) %d%%\r\n",
                       i, (int)boxes[i].x1, (int)boxes[i].y1,
                       (int)boxes[i].x2, (int)boxes[i].y2, score);
        }
        yolo_draw_boxes(fb, LCD_WIDTH, LCD_HEIGHT, boxes, box_n);
        XTime_GetTime(&p1);
        pf_add(PF_POST, p0, p1);
        g_post_us=elapsed_us(p0,p1);
    } else {
        XTime_GetTime(&t1);
        infer_ms = elapsed_ms(t0, t1);
        g_call_us=elapsed_us(t0,t1);
        pf_add(PF_INFER, t0, t1);
        xil_printf("[YOLO] inference FAILED after %u ms\r\n",
                   (unsigned int)infer_ms);
    }

    XTime_GetTime(&p0);
    if (infer_ok) {
        char s[48];
        snprintf(s, sizeof(s), box_n>0?"TARGET DETECTED":"NO TARGET");
        if (box_n > 0)
            show_status(fb, s, 0, 255, 0);
        else
            show_status(fb, s, 255, 255, 0);
    } else {
        g_have_infer = 0;
        show_status(fb, "YOLO TIMEOUT", 255, 0, 0);
    }

    /* 画框/文字可能压到底部按钮条，重画一遍保证按钮始终可见可点。 */
    draw_buttons(fb);
    Xil_DCacheFlushRange((INTPTR)FB_ADDR, FRAME_SIZE);
    XTime_GetTime(&p1);
    pf_add(PF_FLUSH_FB, p0, p1);
    XTime_GetTime(&run1);g_wall_us=elapsed_us(run0,run1);
    if(infer_ok&&!g_filter_skipped)ui_save_result();else g_results[ui_cache_index(g_filter_mode)].valid=0;
    draw_hud(fb);flush_band(0,HUD_AREA_H);
    xil_printf("[TIMING] img=%d mode=%s filter=%u pre=%u post=%u pl_wait=%u call=%u run_wall=%u us\r\n",
       g_img_idx,ui_mode_name(g_filter_mode),(unsigned)g_filter_us,(unsigned)g_pre_us,(unsigned)g_post_us,
       (unsigned)g_pl_us,(unsigned)g_call_us,(unsigned)g_wall_us);

}

/* ---------------------------------------------------------------------------
 * 按钮动作。返回 1 表示这张图的显示内容变了（需要重画按钮条）。
 * --------------------------------------------------------------------------- */
#include "ui_main_extra.inc"

static void handle_button(int btn, u8 *fb, int weights_ok)
{
    if(btn!=BTN_AUTO)g_auto_run=0;
    switch (btn) {
    case BTN_SELECT:
        g_keypad=1;ui_keyboard_reset(&g_keyboard);
        ui_keyboard_draw(fb,&g_keyboard);
        Xil_DCacheFlushRange((INTPTR)FB_ADDR,FRAME_SIZE);break;
    case BTN_COMPARE:
        ui_open_compare(fb,weights_ok);break;
    case BTN_PREV:
    case BTN_NEXT: {
        const int next = (btn == BTN_NEXT) ? find_next_image(g_img_idx)
                                           : find_prev_image(g_img_idx);
        if (next <= 0) {
            xil_printf("[UI] no other image on the SD card\r\n");
            show_status(fb, "NO OTHER IMG", 255, 0, 0);
            break;
        }
        xil_printf("[UI] %s -> %d.bmp\r\n",
                   btn == BTN_NEXT ? "NEXT" : "PREV", next);
        g_img_idx = next;
        g_bmp_ok = (load_and_show(fb, g_img_idx) == 0);
        break;
    }
    case BTN_RUN:
        xil_printf("[UI] RUN\r\n");
        run_inference_once(fb, weights_ok);
        break;
    case BTN_AUTO:
        if(!g_bmp_ok){show_status(fb,"LOAD IMAGE FIRST",255,120,60);break;}
        g_auto_run = !g_auto_run;
        xil_printf("[UI] AUTO %s (%s)\r\n", g_auto_run ? "ON" : "OFF",
                   g_auto_run ? "infer then auto-advance" : "stopped");
        if (g_auto_run)
            show_status(fb, "AUTO RUNNING", 0, 255, 0);
        draw_buttons(fb);
        flush_band(CTRL_BTN_Y0, LCD_HEIGHT - CTRL_BTN_Y0);
        break;
    case BTN_FILTER:
        /* -------------------------------------------------------------------
         * 先等一小会儿再真正切档。
         *
         * 为什么需要：这一下按完之后是"跑一次软件滤波（几十 ms）+ 整屏重绘"，
         * 手快的话第二下会落在第一下还没做完的时候 —— 屏幕还在刷、模式已经在
         * 变，看起来就是"按一下跳了两档 / 跳档乱套"。
         * 这个延时让手指离开、让电容屏的回弹抖动弹完，再开始干活，交互就干净了。
         * 延时很短（见 FILT_SWITCH_DELAY_MS），感觉不出来，但足以吃掉抖动。
         * ----------------------------------------------------------------- */
        delay_ms(FILT_SWITCH_DELAY_MS);

        /* 只在"上屏的两档" + OFF 之间循环（实测效果前二） */
        g_filter_mode = filter_ui_next(g_filter_mode);
        xil_printf("[UI] FILTER -> %s (%s) [settle %u ms]\r\n",
                   filter_sw_mode_name(g_filter_mode),
                   filter_sw_mode_desc(g_filter_mode),
                   (unsigned)FILT_SWITCH_DELAY_MS);
        if (g_bmp_ok) {
            /* 立刻对**当前这张图**重新处理并重绘 —— 这才叫即时 A/B 对比：
               选到滤波档 -> 跑一次 PS 软件滤波到 FLT_BUF（几十 ms）
               选到 OFF   -> 只把 g_img_src 指回 IMG_BUF（**零成本**，原图一直在）
               都不需要重新读 SD 卡，所以可以来回快切比较。
               换档后上一次的 max_conf 已经不对应当前输入，必须清掉，
               否则 HUD 上会留着一个属于别的滤波档的读数 —— 那是最容易骗到人的 bug。 */
            g_have_infer = 0;
            g_box_n = 0;
            g_max_conf_pct = 0;
            apply_filter_mode();
            {
                char s[48];
                if (g_filter_mode == FILT_MODE_OFF)
                    snprintf(s, sizeof(s), "FILTER OFF");
                else
                    snprintf(s, sizeof(s), "%s READY",
                             filter_sw_mode_name(g_filter_mode));
                render_image(fb, s);
            }
        } else {
            hud_redraw(fb);
            draw_buttons(fb);
            flush_band(CTRL_BTN_Y0, LCD_HEIGHT - CTRL_BTN_Y0);
        }
        break;
    default:
        break;
    }
}

/* ---------------------------------------------------------------------------
 * AUTO 用：推理完一张后立刻换到下一张。
 * 复用 NEXT 的规则（find_next_image），所以走到最后一张后会回到第一张继续，
 * 自动巡览不会自己停下 -- 要停就按 AUTO。
 * --------------------------------------------------------------------------- */
static void auto_advance(u8 *fb)
{
    const int next = find_next_image(g_img_idx);
    if (next <= 0) {
        g_auto_run=0;
        xil_printf("[UI] AUTO: no image to advance to\r\n");
        return;
    }
    xil_printf("[UI] AUTO -> %d.bmp\r\n", next);
    g_img_idx = next;
    g_bmp_ok = (load_and_show(fb, g_img_idx) == 0);
}

/* ---------------------------------------------------------------------------
 * 轮询一次触摸：按下沿 -> 命中判定 -> 动作。
 * 只在"按下沿"触发，必须松手后才允许下一次（防抖），所以按住不放不会连发。
 * 返回本次触发的按钮编号（0..BTN_COUNT-1），没触发返回 -1。
 *
 * 抽成函数是为了在 AUTO 一轮里的多个时刻都能调用（推理前 / 推理后 / 换图后），
 * 否则一轮要跑好几百毫秒，AUTO 会按不停。
 * --------------------------------------------------------------------------- */
static UiTouchGate g_touch_gate;

static int ui_poll_touch(u8 *fb, int weights_ok)
{
    int tx = 0, ty = 0;
    int fired = -1;
    XTime k0, k1;
    int down;

    XTime_GetTime(&k0);
    down = touch_hw_scan(&tx, &ty);
    XTime_GetTime(&k1);
    pf_add(PF_TOUCH, k0, k1);

    if (ui_touch_gate_accept(&g_touch_gate,down,(u32)(k1 / (XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ / 2000u)))) {
        int raw_x = 0, raw_y = 0;
        touch_hw_last_raw(&raw_x,&raw_y);
        if(g_keypad){
            int key=ui_keyboard_hit(tx,ty);if(key>=0)ui_key_action(key,fb);
        }else if(g_compare){
            int action=ui_compare_hit(tx,ty);if(action>=0)ui_compare_action(action,fb,weights_ok);
        }else{
            int btn=btn_hit(tx,ty);
            xil_printf("[TOUCH] raw=(%d,%d) -> (%d,%d) btn=%d\r\n",raw_x,raw_y,tx,ty,btn);
            if(btn>=0){handle_button(btn,fb,weights_ok);fired=btn;}
            else if(tx<IMG_AREA_W)ui_image_touch(tx,ty,fb);
        }
    }

    return fired;
}

int main(void)
{
    u8 *fb = (u8 *)FB_ADDR;
    int weights_ok = 0;
    int sd_ok = 1;
    int have_w = 0;
    int have_p = 0;
    int touch_ok = 0;
    int st;

    xil_printf("\r\n=== SD BMP -> YOLO26 HLS -> LCD 800x480 (touch UI) ===\r\n");
    xil_printf("[APP] build=%s\r\n", APP_BUILD_TAG);
    xil_printf("[YOLO] control base=0x%08X\r\n", YOLO_BASEADDR);
    xil_printf("[YOLO] ctrl=0x%08X\r\n",
               Xil_In32(YOLO_BASEADDR + YOLO_REG_CTRL));

    /* Who executes what, decided by the SAME header the kernel includes.
       Printed at boot because "the app still splits the run at the attentions"
       is invisible otherwise: it looks like a working inference, just slower.
       0 PS layers is required for this Conv1x1 schedule -- the whole network is one
       IP call. */
    xil_printf("[YOLO] %u layers, %d PS-side (0 -> one PL call)\r\n",
               (unsigned)YOLO_I8_NUM_LAYERS, count_ps_layers());
    xil_printf("[YOLO] blobs: weights=%u params=%u work=%u bytes\r\n",
               (unsigned)WEIGHTS_BYTES, (unsigned)FIXED_PARAM_BYTES,
               (unsigned)WORK_BYTES);

    /* 触摸控制器（GT9xx / FT5206，软件模拟 I2C 走 PS EMIO）。
       放在 SD 挂载之前：即使没有卡，屏幕上也要能提示触摸是否就绪。 */
    touch_ok = (touch_hw_init() == 0);
    if (!touch_ok)
        xil_printf("[TOUCH] init FAILED - buttons will not respond\r\n");

    st = f_mount(&fatfs, "", 1);
    if (st != FR_OK) {
        xil_printf("Mount err=%d -> color bars\r\n", st);
        fill_color_bars(fb);
        show_status(fb, "SD FAIL", 255, 0, 0);
        sd_ok = 0;
    } else {
        list_sd_root();
        g_img_idx = find_first_image();
        if (g_img_idx > 0) {
            g_bmp_ok = (load_and_show(fb, g_img_idx) == 0);
        } else {
            fill_color_bars(fb);
            show_status(fb, "NO IMG", 255, 0, 0);
            draw_buttons(fb);
        }
    }

    if (!touch_ok) {
        show_status(fb, "NO TOUCH", 255, 0, 0);
    } else {
        /* HUD 由 render_image() 画过一遍；这里只在没有图的情况下补一次，
           保证即使 SD 卡读不出图也能看到 FILTER 的当前状态。 */
        if (!g_bmp_ok)
            hud_redraw(fb);
    }

    Xil_DCacheFlushRange((INTPTR)FB_ADDR, FRAME_SIZE);
    st = vdma_init(&vdma, VDMA_DEV_ID);
    if (st) {
        xil_printf("VDMA init FAIL\r\n");
        return -1;
    }
    st = vdma_start(&vdma, FB_ADDR);
    if (st) {
        xil_printf("VDMA start FAIL\r\n");
        return -1;
    }
    xil_printf("Display running, FB=0x%08X\r\n", FB_ADDR);

    if (sd_ok &&
        load_weights_pair((u8 *)WEIGHTS_BUF, (u8 *)FIXED_PARAMS_BUF,
                          &have_w, &have_p) == 0) {
        u32 wsum = fnv1a32((const u8 *)WEIGHTS_BUF, WEIGHTS_BYTES);
        u32 psum = fnv1a32((const u8 *)FIXED_PARAMS_BUF, FIXED_PARAM_BYTES);
        xil_printf("[YOLO] weights loaded sum=0x%08X %s (expect 0x%08X)\r\n",
                   wsum, wsum == WEIGHTS_SUM_EXPECT ? "OK" : "BAD",
                   WEIGHTS_SUM_EXPECT);
        xil_printf("[YOLO] params  loaded sum=0x%08X %s (expect 0x%08X)\r\n",
                   psum, psum == FIXED_SUM_EXPECT ? "OK" : "BAD",
                   FIXED_SUM_EXPECT);
        weights_ok = (wsum == WEIGHTS_SUM_EXPECT &&
                      psum == FIXED_SUM_EXPECT);
        xil_printf("[STUDY] image=%d mode=%s\r\n", g_img_idx, filter_sw_mode_name(g_filter_mode));

    if (!weights_ok) {
            xil_printf("[YOLO] model checksum mismatch; inference disabled\r\n");
            show_status(fb, "MODEL BAD", 255, 0, 0);
        } else {
#if YOLO_REQUIRE_SCALES
            /* The scale blob is packaged and checked as part of the model set. */
            int have_s = 0;
            if (load_act_scales((u8 *)ACT_SCALES_BUF, &have_s) == 0) {
                xil_printf("[YOLO] scales  loaded sum=0x%08X %s (expect 0x%08X, "
                           "%u bytes)\r\n",
                           (unsigned)fnv1a32((const u8 *)ACT_SCALES_BUF,
                                             ACT_SCALE_BYTES),
                           fnv1a32((const u8 *)ACT_SCALES_BUF, ACT_SCALE_BYTES) ==
                                   ACT_SCALES_SUM_EXPECT ? "OK" : "BAD",
                           ACT_SCALES_SUM_EXPECT, (unsigned)ACT_SCALE_BYTES);
            } else {
                xil_printf("[YOLO] activation scales MISSING -> bad card; "
                           "inference disabled\r\n");
                show_status(fb, "MISS SCALES", 255, 0, 0);
                weights_ok = 0;
            }
#endif
        }
        if (weights_ok) {
            /* Weights and quantisation parameters are read-only from here on, so
               push them out of D-cache once.  The old code flushed all 3.2 MB
               before every single inference even though nothing had changed. */
            Xil_DCacheFlushRange((INTPTR)WEIGHTS_BUF, WEIGHTS_BYTES);
            Xil_DCacheFlushRange((INTPTR)FIXED_PARAMS_BUF, FIXED_PARAM_BYTES);
#if YOLO_REQUIRE_SCALES
            Xil_DCacheFlushRange((INTPTR)ACT_SCALES_BUF, ACT_SCALE_BYTES);
#endif
        }
    } else {
        xil_printf("[YOLO] weights/params missing -> display without boxes\r\n");
        if (have_w)
            show_status(fb, "MISS PARAMS", 255, 0, 0);
        else if (have_p)
            show_status(fb, "MISS WEIGHTS", 255, 0, 0);
        else
            show_status(fb, "NO WEIGHTS", 255, 0, 0);
        weights_ok = 0;
    }

#if YOLO_BOOT_SELFTEST
    if (weights_ok)
        yolo26_selftest(fb);
#endif

    xil_printf("[UI] PREV/NEXT=image RUN=infer AUTO=sweep FILTER=%s\r\n",
               filter_sw_mode_name(g_filter_mode));
    xil_printf("[UI] FILTER cycles OFF/GAUSS/RGBG5; "
               "press RUN to refresh boxes\r\n");
    delay_ms(200);

    /* -----------------------------------------------------------------------
     * 主循环：轮询触摸（不再有阻塞式按键等待）。
     *   按下沿 -> 命中判定 -> 动作；松手后才允许下一次触发（防抖）。
     *
     * AUTO 模式：推理完一张 -> 立刻换到下一张（重新滤波 + letterbox）-> 再推理，
     * 依次往后，到最后一张后回到第一张继续。每个耗时步骤前后都回来轮询触摸，
     * 所以 AUTO 随时按得停。
     * --------------------------------------------------------------------- */
    for (;;) {
        ui_poll_touch(fb, weights_ok);

        if (!(g_auto_run && weights_ok) || g_keypad || g_compare)
            continue;

        {
            XTime c0, c1;
            XTime_GetTime(&c0);

            run_inference_once(fb, weights_ok);

            /* 推理是阻塞的（几百 ms），结束后先收一次触摸：可能是要按停
               AUTO，也可能是要改 PREV/NEXT/RUN。 */
            ui_poll_touch(fb, weights_ok);
            if (g_auto_run) {
                auto_advance(fb);

                /* 换图 + 滤波 + letterbox 同样要几百 ms，再收一次触摸。 */
                ui_poll_touch(fb, weights_ok);
            }

            XTime_GetTime(&c1);
            if (g_auto_run) {
                pf_report("auto", us_between(c0, c1) / 1000u);
            } else {
                pf_reset();       /* 刚被按停，这一轮的统计作废 */
            }
        }
    }

    return 0;
}
