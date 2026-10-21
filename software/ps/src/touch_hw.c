/* ===========================================================================
 *  touch_hw.c - bit-banged I2C + GT9xx / FT5206 capacitive touch driver
 *
 *  Everything here is a port of the official ALIENTEK example
 *  34_touch_draw_lcd (TOUCH\myiic.c, TOUCH\gt9147.c, TOUCH\ft5206.c,
 *  emio_iic_cfg\emio_iic_cfg.c), with three deliberate changes:
 *
 *   1) the EMIO <-> XGpioPs glue lives here instead of in a separate
 *      emio_iic_cfg module, so the whole touch stack is self contained;
 *   2) the panel's native coordinate range is read back from the GT9xx
 *      configuration registers (0x8048..0x804B) instead of being hard-coded
 *      from the 4.3" panel.  The 4.3" module reports 480x800 (portrait) while
 *      this ATK-MD0700R 7" module reports 800x480; scaling from the self
 *      reported range makes both work with no rebuild;
 *   3) the 184-byte configuration table from the example is deliberately NOT
 *      downloaded.  ALIENTEK only sends it for the 4.3" MCU panel (lcd_id
 *      0x5510) because that table encodes 480x800 and would mis-configure a
 *      7" panel.  The table is kept below behind TOUCH_GT9XX_SEND_CFG=0 so it
 *      can be re-enabled on the bench if a panel ever needs it.
 * ===========================================================================
 */

#include <stdio.h>
#include <string.h>

#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xgpiops.h"
#include "xtime_l.h"

#include "touch_hw.h"

/* ------------------------------------------------------------------ pins --
 * XGpioPs numbers MIO 0..53 first and the EMIO GPIO bank starts at 54, so
 * EMIO[0..3] are pins 54..57.  Matches emio_iic_cfg.h of the reference. */
#define EMIO_SCL_NUM     54   /* R19 */
#define EMIO_SDA_NUM     55   /* P20, external PULLUP */
#define EMIO_CT_RST_NUM  56   /* M19 */
#define EMIO_CT_INT_NUM  57   /* U19 */

/* ---------------------------------------------------------------- I2C IDs --
 * GT9xx (GT911 / GT9147 / GT9271) uses a 16-bit register address and the
 * 7-bit slave address 0x14 (write byte 0x28) when CT_INT is held high across
 * the reset, or 0x5D (write byte 0xBA) when it is held low.
 * FT5x06 / FT5206 uses an 8-bit register address and 7-bit address 0x38. */
#define GT_WR_28         0x28u
#define GT_RD_29         0x29u
#define GT_WR_BA         0xBAu
#define GT_RD_BB         0xBBu
#define FT_WR_70         0x70u
#define FT_RD_71         0x71u

/* GT9xx registers (GT911/GT9147 share the map). */
#define GT_CTRL_REG      0x8040u   /* soft reset */
#define GT_CFGS_REG      0x8047u   /* config table start (= version byte) */
#define GT_XMAX_REG      0x8048u   /* 2 bytes, little endian */
#define GT_YMAX_REG      0x804Au   /* 2 bytes, little endian */
#define GT_CHECK_REG     0x80FFu
#define GT_PID_REG       0x8140u   /* 4 bytes, ASCII, e.g. "911" */
#define GT_GSTID_REG     0x814Eu   /* b7 = buffer ready, b3:0 = touch count */
#define GT_TP1_REG       0x8150u   /* 8 bytes per point */

/* FT5206 registers. */
#define FT_DEVIDE_MODE   0x00u
#define FT_REG_NUM_FINGER 0x02u
#define FT_TP1_REG       0x03u
#define FT_ID_G_LIB_VERSION 0xA1u
#define FT_ID_G_MODE     0xA4u
#define FT_ID_G_THGROUP  0x80u
#define FT_ID_G_PERIODACTIVE 0x88u

/* Chip ids reported by touch_hw_init(). */
#define CHIP_NONE   0
#define CHIP_GT9XX  1
#define CHIP_FT5206 2

/* Display geometry this driver maps onto (kept local so the file has no
 * dependency on main.c). */
#define TOUCH_FB_W  800
#define TOUCH_FB_H  480

/* --------------------------------------------------------------------------
 *  Orientation / calibration knobs.
 *
 *  The mapping is derived automatically from the controller's own reported
 *  range: a "wide" range (x > y) is already landscape and is used as is, a
 *  "tall" range (y > x) is treated as portrait and its axes are swapped.
 *  TOUCH_FORCE_SWAP overrides that when the auto rule guesses wrong.
 *  Every touch-down is printed on the UART as
 *      [TOUCH] raw=(x,y) -> (X,Y)
 *  so the flips can be fixed on the bench with these three macros alone -
 *  a rebuild of the PS application only, no bitstream change.
 * ------------------------------------------------------------------------ */
#define TOUCH_FORCE_SWAP (-1)   /* -1 = auto, 0 = never, 1 = always */
#define TOUCH_FLIP_X     0      /* 1 = mirror the 800-pixel axis */
#define TOUCH_FLIP_Y     0      /* 1 = mirror the 480-pixel axis */
#define TOUCH_GT9XX_SEND_CFG 0  /* see the file header - leave at 0 */

/* How long a touch point stays "down" after the last fresh frame.  Must be
 * comfortably longer than the controller's report period (~10 ms) so that
 * polling faster than the controller does not look like a release. */
#define TOUCH_HOLD_MS    60u

/* The 184-byte table is only meaningful for the 4.3" panel; see header. */
#if TOUCH_GT9XX_SEND_CFG
static const u8 gt9xx_cfg_tbl[] = {
    0x60,0xE0,0x01,0x20,0x03,0x05,0x35,0x00,0x02,0x08,
    0x1E,0x08,0x50,0x3C,0x0F,0x05,0x00,0x00,0xFF,0x67,
    0x50,0x00,0x00,0x18,0x1A,0x1E,0x14,0x89,0x28,0x0A,
    0x30,0x2E,0xBB,0x0A,0x03,0x00,0x00,0x02,0x33,0x1D,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x32,0x00,0x00,
    0x2A,0x1C,0x5A,0x94,0xC5,0x02,0x07,0x00,0x00,0x00,
    0xB5,0x1F,0x00,0x90,0x28,0x00,0x77,0x32,0x00,0x62,
    0x3F,0x00,0x52,0x50,0x00,0x52,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x0F,
    0x0F,0x03,0x06,0x10,0x42,0xF8,0x0F,0x14,0x00,0x00,
    0x00,0x00,0x1A,0x18,0x16,0x14,0x12,0x10,0x0E,0x0C,
    0x0A,0x08,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x29,0x28,0x24,0x22,0x20,0x1F,0x1E,0x1D,
    0x0E,0x0C,0x0A,0x08,0x06,0x05,0x04,0x02,0x00,0xFF,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,
};
#endif

static XGpioPs   touch_gpio;
static int       touch_gpio_ok = 0;
static int       s_chip        = CHIP_NONE;
static int       s_gt_wr       = GT_WR_28;
static int       s_gt_rd       = GT_RD_29;
static int       s_raw_xmax    = 0;     /* controller's own X output max */
static int       s_raw_ymax    = 0;     /* controller's own Y output max */
static int       s_swap        = 0;     /* raw axes swapped for the display */
static int       s_last_raw_x  = 0;
static int       s_last_raw_y  = 0;
static int       s_have_point  = 0;
static XTime     s_last_point  = 0;

/* ------------------------------------------------------------ timing ----- */
static u32 s_ticks_per_us = 0;

static void th_delay_us(u32 us)
{
    XTime now, target;
    if (s_ticks_per_us == 0)
        return;
    XTime_GetTime(&now);
    target = now + (XTime)us * (XTime)s_ticks_per_us;
    do {
        XTime_GetTime(&now);
    } while (now < target);
}

static void th_delay_ms(u32 ms)
{
    while (ms--)
        th_delay_us(1000u);
}

/* --------------------------------------------------------- I2C bit bang -- */
static void sda_out(void)
{
    XGpioPs_SetDirectionPin(&touch_gpio, EMIO_SDA_NUM, 1);
    XGpioPs_SetOutputEnablePin(&touch_gpio, EMIO_SDA_NUM, 1);
}

static void sda_in(void)
{
    XGpioPs_SetOutputEnablePin(&touch_gpio, EMIO_SDA_NUM, 0);
    XGpioPs_SetDirectionPin(&touch_gpio, EMIO_SDA_NUM, 0);
}

static void iic_scl(int v)
{
    XGpioPs_WritePin(&touch_gpio, EMIO_SCL_NUM, v ? 1u : 0u);
}

static void iic_sda(int v)
{
    XGpioPs_WritePin(&touch_gpio, EMIO_SDA_NUM, v ? 1u : 0u);
}

static int read_sda(void)
{
    return XGpioPs_ReadPin(&touch_gpio, EMIO_SDA_NUM) ? 1 : 0;
}

static void ct_rst(int v)
{
    XGpioPs_WritePin(&touch_gpio, EMIO_CT_RST_NUM, v ? 1u : 0u);
}

static void ct_int_dir(int out)
{
    XGpioPs_SetDirectionPin(&touch_gpio, EMIO_CT_INT_NUM, out ? 1u : 0u);
    XGpioPs_SetOutputEnablePin(&touch_gpio, EMIO_CT_INT_NUM, out ? 1u : 0u);
}

static void ct_int(int v)
{
    XGpioPs_WritePin(&touch_gpio, EMIO_CT_INT_NUM, v ? 1u : 0u);
}

/* Bus clock period; 2 us per quarter phase -> ~125 kHz, same as the example. */
static void ct_delay(void)
{
    th_delay_us(2);
}

static void iic_init(void)
{
    sda_out();
    iic_scl(1);
    iic_sda(1);
}

static void iic_start(void)
{
    sda_out();
    iic_sda(1);
    iic_scl(1);
    ct_delay();
    iic_sda(0);
    ct_delay();
    iic_scl(0);
    ct_delay();
}

static void iic_stop(void)
{
    sda_out();
    iic_scl(0);
    iic_sda(0);
    ct_delay();
    iic_scl(1);
    ct_delay();
    iic_sda(1);
    ct_delay();
}

/* 0 = ACK received, 1 = no ACK (bus error / no device) */
static int iic_wait_ack(void)
{
    u32 err = 0;
    sda_in();
    iic_sda(1);
    ct_delay();
    iic_scl(1);
    ct_delay();
    while (read_sda()) {
        if (++err > 250u) {
            iic_stop();
            return 1;
        }
    }
    iic_scl(0);
    ct_delay();
    return 0;
}

static void iic_ack(void)
{
    iic_scl(0);
    sda_out();
    iic_sda(0);
    ct_delay();
    iic_scl(1);
    ct_delay();
    iic_scl(0);
    ct_delay();
}

static void iic_nack(void)
{
    iic_scl(0);
    sda_out();
    iic_sda(1);
    ct_delay();
    iic_scl(1);
    ct_delay();
    iic_scl(0);
    ct_delay();
}

static void iic_send_byte(u8 txd)
{
    sda_out();
    iic_scl(0);
    for (int t = 0; t < 8; t++) {
        iic_sda((txd & 0x80u) ? 1 : 0);
        txd = (u8)(txd << 1);
        ct_delay();
        iic_scl(1);
        ct_delay();
        iic_scl(0);
        ct_delay();
    }
}

static u8 iic_read_byte(int ack)
{
    u8 receive = 0;
    sda_in();
    for (int i = 0; i < 8; i++) {
        iic_scl(0);
        ct_delay();
        iic_scl(1);
        receive = (u8)(receive << 1);
        if (read_sda())
            receive |= 1u;
        ct_delay();
    }
    if (ack)
        iic_ack();
    else
        iic_nack();
    return receive;
}

/* ------------------------------------------------- GT9xx register access -- */
/* 16-bit register address, 7-bit slave address in s_gt_wr / s_gt_rd. */
static int gt_wr_reg(u16 reg, const u8 *buf, u8 len)
{
    int ret = 0;
    iic_start();
    iic_send_byte((u8)s_gt_wr);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_send_byte((u8)(reg >> 8));
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_send_byte((u8)(reg & 0xFFu));
    if (iic_wait_ack()) { iic_stop(); return 1; }
    for (u8 i = 0; i < len; i++) {
        iic_send_byte(buf[i]);
        ret = iic_wait_ack();
        if (ret)
            break;
    }
    iic_stop();
    return ret;
}

/* 0 = success (every ACK seen and byte clocked), 1 = bus error */
static int gt_rd_reg(u16 reg, u8 *buf, u8 len)
{
    iic_start();
    iic_send_byte((u8)s_gt_wr);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_send_byte((u8)(reg >> 8));
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_send_byte((u8)(reg & 0xFFu));
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_start();
    iic_send_byte((u8)s_gt_rd);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    for (u8 i = 0; i < len; i++)
        buf[i] = iic_read_byte(i == (u8)(len - 1) ? 0 : 1);
    iic_stop();
    return 0;
}

static u16 gt_rd_u16(u16 reg)
{
    u8 b[2] = {0, 0};
    if (gt_rd_reg(reg, b, 2) != 0)
        return 0;
    return (u16)(((u16)b[1] << 8) | b[0]);
}

/* ------------------------------------------------ FT5206 register access -- */
/* 8-bit register address, 7-bit slave address 0x38. */
static int ft_wr_reg(u8 reg, const u8 *buf, u8 len)
{
    int ret = 0;
    iic_start();
    iic_send_byte((u8)FT_WR_70);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_send_byte(reg);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    for (u8 i = 0; i < len; i++) {
        iic_send_byte(buf[i]);
        ret = iic_wait_ack();
        if (ret)
            break;
    }
    iic_stop();
    return ret;
}

static int ft_rd_reg(u8 reg, u8 *buf, u8 len)
{
    iic_start();
    iic_send_byte((u8)FT_WR_70);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_send_byte(reg);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    iic_start();
    iic_send_byte((u8)FT_RD_71);
    if (iic_wait_ack()) { iic_stop(); return 1; }
    for (u8 i = 0; i < len; i++)
        buf[i] = iic_read_byte(i == (u8)(len - 1) ? 0 : 1);
    iic_stop();
    return 0;
}

/* --------------------------------------------------------- reset / probe -- */
/* CT_INT level across the reset selects the GT911 slave address; it is then
 * released to input so the controller can drive the interrupt line. */
static void ct_reset(int int_high)
{
    ct_int_dir(1);
    ct_int(int_high);
    th_delay_ms(2);
    ct_rst(0);
    th_delay_ms(10);
    ct_rst(1);
    th_delay_ms(10);
    ct_int_dir(0);
    th_delay_ms(120);
}

static int pid_looks_like_gt9xx(const u8 *pid)
{
    /* "911", "9147", "9271", "928" ... accept four decimal digits or the
     * three-digit 911 form; reject an all-0x00 / all-0xFF bus garbage read. */
    for (int i = 0; i < 3; i++) {
        if (pid[i] < '0' || pid[i] > '9')
            return 0;
    }
    return 1;
}

static int probe_gt9xx(void)
{
    static const struct { int wr, rd; int int_high; } cand[2] = {
        { GT_WR_28, GT_RD_29, 1 },   /* 7-bit 0x14 - what the example uses */
        { GT_WR_BA, GT_RD_BB, 0 },   /* 7-bit 0x5D - GT911 alternate */
    };
    for (int c = 0; c < 2; c++) {
        u8 pid[5];
        u8 tmp[2];
        memset(pid, 0, sizeof(pid));
        ct_reset(cand[c].int_high);
        iic_init();
        s_gt_wr = cand[c].wr;
        s_gt_rd = cand[c].rd;
        if (gt_rd_reg(GT_PID_REG, pid, 4) != 0)
            continue;
        if (!pid_looks_like_gt9xx(pid))
            continue;

        xil_printf("[TOUCH] GT9xx PID=%c%c%c%c at i2c wr=0x%02X\r\n",
                   pid[0], pid[1], pid[2], pid[3], (unsigned)s_gt_wr);

        /* Soft reset, then release - same sequence as the example. */
        tmp[0] = 0x02;
        gt_wr_reg(GT_CTRL_REG, tmp, 1);
        th_delay_ms(10);
        tmp[0] = 0x00;
        gt_wr_reg(GT_CTRL_REG, tmp, 1);
        th_delay_ms(10);

        /* The controller tells us its own output range - use it instead of
         * assuming the 4.3" panel's 480x800. */
        s_raw_xmax = gt_rd_u16(GT_XMAX_REG);
        s_raw_ymax = gt_rd_u16(GT_YMAX_REG);
        if (s_raw_xmax <= 0 || s_raw_xmax > 4096) s_raw_xmax = TOUCH_FB_W;
        if (s_raw_ymax <= 0 || s_raw_ymax > 4096) s_raw_ymax = TOUCH_FB_H;

#if TOUCH_GT9XX_SEND_CFG
        {
            u8 cfgver = 0;
            if (gt_rd_reg(GT_CFGS_REG, &cfgver, 1) == 0 && cfgver < 0x60) {
                u8 sum = 0;
                u8 chk[2];
                for (u32 i = 0; i < sizeof(gt9xx_cfg_tbl); i++)
                    sum = (u8)(sum + gt9xx_cfg_tbl[i]);
                chk[0] = (u8)((~sum) + 1);
                chk[1] = 1;   /* 1 = also persist to the controller flash */
                gt_wr_reg(GT_CFGS_REG, gt9xx_cfg_tbl,
                          (u8)sizeof(gt9xx_cfg_tbl));
                gt_wr_reg(GT_CHECK_REG, chk, 2);
                xil_printf("[TOUCH] config table v%02X downloaded\r\n",
                           (unsigned)cfgver);
            }
        }
#endif
        s_chip = CHIP_GT9XX;
        return 0;
    }
    return -1;
}

static int probe_ft5206(void)
{
    u8 ver[2] = {0, 0};
    u8 tmp;
    ct_reset(1);
    iic_init();
    if (ft_rd_reg((u8)FT_ID_G_LIB_VERSION, ver, 2) != 0)
        return -1;
    /* Reference accepts 0x3003 plus the 0x0001 / 0x0002 revisions. */
    if (!((ver[0] == 0x30 && ver[1] == 0x03) || ver[1] == 0x01 ||
          ver[1] == 0x02))
        return -1;

    xil_printf("[TOUCH] FT5206 lib ver=0x%02X%02X\r\n",
               (unsigned)ver[0], (unsigned)ver[1]);
    tmp = 0;
    ft_wr_reg((u8)FT_DEVIDE_MODE, &tmp, 1);      /* normal operation */
    ft_wr_reg((u8)FT_ID_G_MODE, &tmp, 1);        /* polling mode */
    tmp = 22;                                    /* touch threshold */
    ft_wr_reg((u8)FT_ID_G_THGROUP, &tmp, 1);
    tmp = 12;                                    /* active period */
    ft_wr_reg((u8)FT_ID_G_PERIODACTIVE, &tmp, 1);

    /* FT5x06 reports in panel pixels; the 7" module is natively 800x480. */
    s_raw_xmax = TOUCH_FB_W;
    s_raw_ymax = TOUCH_FB_H;
    s_chip = CHIP_FT5206;
    return 0;
}

/* ------------------------------------------------------------- mapping --- */
static int scale_axis(int raw, int raw_max, int disp_max)
{
    int v;
    if (raw_max <= 0)
        raw_max = disp_max;
    v = (int)(((long)raw * disp_max) / raw_max);
    if (v < 0)
        v = 0;
    if (v > disp_max - 1)
        v = disp_max - 1;
    return v;
}

static void map_point(int raw_x, int raw_y, int *out_x, int *out_y)
{
    int x, y;
    if (s_swap) {
        x = scale_axis(raw_y, s_raw_ymax, TOUCH_FB_W);
        y = scale_axis(raw_x, s_raw_xmax, TOUCH_FB_H);
    } else {
        x = scale_axis(raw_x, s_raw_xmax, TOUCH_FB_W);
        y = scale_axis(raw_y, s_raw_ymax, TOUCH_FB_H);
    }
#if TOUCH_FLIP_X
    x = TOUCH_FB_W - 1 - x;
#endif
#if TOUCH_FLIP_Y
    y = TOUCH_FB_H - 1 - y;
#endif
    *out_x = x;
    *out_y = y;
}

/* --------------------------------------------------------------- public -- */
int touch_hw_init(void)
{
    XGpioPs_Config *cfg;

    s_ticks_per_us = XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ / 2000000u;
    if (s_ticks_per_us == 0)
        s_ticks_per_us = 1;

    cfg = XGpioPs_LookupConfig(XPAR_XGPIOPS_0_DEVICE_ID);
    if (cfg == NULL) {
        xil_printf("[TOUCH] XGpioPs lookup failed\r\n");
        return -1;
    }
    if (XGpioPs_CfgInitialize(&touch_gpio, cfg, cfg->BaseAddr) != XST_SUCCESS) {
        xil_printf("[TOUCH] XGpioPs init failed\r\n");
        return -1;
    }
    touch_gpio_ok = 1;

    /* SCL / SDA / CT_RST are outputs, CT_INT starts as an output too so the
     * reset sequence can drive the GT911 address-select level. */
    XGpioPs_SetDirectionPin(&touch_gpio, EMIO_SCL_NUM, 1);
    XGpioPs_SetOutputEnablePin(&touch_gpio, EMIO_SCL_NUM, 1);
    XGpioPs_SetDirectionPin(&touch_gpio, EMIO_SDA_NUM, 1);
    XGpioPs_SetOutputEnablePin(&touch_gpio, EMIO_SDA_NUM, 1);
    XGpioPs_SetDirectionPin(&touch_gpio, EMIO_CT_RST_NUM, 1);
    XGpioPs_SetOutputEnablePin(&touch_gpio, EMIO_CT_RST_NUM, 1);
    iic_init();

    if (probe_gt9xx() != 0 && probe_ft5206() != 0) {
        xil_printf("[TOUCH] no controller found (tried GT9xx, FT5206)\r\n");
        s_chip = CHIP_NONE;
        return -1;
    }

    if (TOUCH_FORCE_SWAP >= 0)
        s_swap = TOUCH_FORCE_SWAP;
    else
        s_swap = (s_raw_xmax > s_raw_ymax) ? 0 : 1;

    touch_hw_info();
    return 0;
}

void touch_hw_info(void)
{
    xil_printf("[TOUCH] chip=%s raw=%dx%d display=%dx%d swap=%d\r\n",
               s_chip == CHIP_GT9XX ? "GT9xx" :
               (s_chip == CHIP_FT5206 ? "FT5206" : "none"),
               s_raw_xmax, s_raw_ymax, TOUCH_FB_W, TOUCH_FB_H, s_swap);
}

static int gt9xx_scan(int *raw_x, int *raw_y)
{
    u8 st = 0;
    u8 zero = 0;
    if (gt_rd_reg(GT_GSTID_REG, &st, 1) != 0)
        return 0;
    if ((st & 0x80u) == 0)
        return 0;              /* no NEW frame from the controller this poll */
    {
        const int n = st & 0x0Fu;
        int got = 0;
        if (n >= 1 && n <= 5) {
            u8 b[4];
            if (gt_rd_reg(GT_TP1_REG, b, 4) == 0) {
                *raw_x = ((int)b[1] << 8) | b[0];
                *raw_y = ((int)b[3] << 8) | b[2];
                got = 1;
            }
        }
        /* The buffer-ready flag must be cleared for every frame, including the
         * n==0 frame the controller sends when the finger lifts. */
        gt_wr_reg(GT_GSTID_REG, &zero, 1);
        return got;
    }
}

static int ft5206_scan(int *raw_x, int *raw_y)
{
    u8 st = 0;
    u8 b[4];
    if (ft_rd_reg((u8)FT_REG_NUM_FINGER, &st, 1) != 0)
        return 0;
    if ((st & 0x0Fu) < 1 || (st & 0x0Fu) > 5)
        return 0;
    if (ft_rd_reg((u8)FT_TP1_REG, b, 4) != 0)
        return 0;
    *raw_x = (((int)b[0] & 0x0F) << 8) | b[1];
    *raw_y = (((int)b[2] & 0x0F) << 8) | b[3];
    return 1;
}

/* Time since the last fresh touch point, in milliseconds. */
static u32 ms_since(XTime t0)
{
    XTime now;
    if (s_ticks_per_us == 0)
        return 0xFFFFFFFFu;
    XTime_GetTime(&now);
    return (u32)((now - t0) * 1000u / ((XTime)s_ticks_per_us * 1000u));
}

int touch_hw_scan(int *x, int *y)
{
    int rx = 0, ry = 0, fresh;

    if (s_chip == CHIP_NONE || !touch_gpio_ok)
        return 0;

    if (s_chip == CHIP_GT9XX)
        fresh = gt9xx_scan(&rx, &ry);
    else
        fresh = ft5206_scan(&rx, &ry);

    if (fresh) {
        s_last_raw_x = rx;
        s_last_raw_y = ry;
        XTime_GetTime(&s_last_point);
        s_have_point = 1;
    } else if (s_have_point) {
        /* The GT9xx only publishes a new frame every ~10 ms while a finger is
         * held.  This loop polls far faster than that, so a bare "no new frame
         * this time" must NOT be reported as a release - it would flap the
         * down edge and fire the same button over and over.  Hold the last
         * point for TOUCH_HOLD_MS instead; a real lift is either an explicit
         * n==0 frame or the hold window expiring. */
        if (ms_since(s_last_point) >= TOUCH_HOLD_MS)
            s_have_point = 0;
    }

    if (!s_have_point)
        return 0;

    map_point(s_last_raw_x, s_last_raw_y, x, y);
    return 1;
}

void touch_hw_last_raw(int *raw_x, int *raw_y)
{
    if (raw_x)
        *raw_x = s_last_raw_x;
    if (raw_y)
        *raw_y = s_last_raw_y;
}
