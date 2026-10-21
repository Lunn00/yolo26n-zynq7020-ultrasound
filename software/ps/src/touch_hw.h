/* ===========================================================================
 *  touch_hw.h - ATK-MD0700R (7" 800x480 RGB) capacitive touch on ZYNQ PS EMIO
 *
 *  The panel's controller is a Goodix GT9xx part (GT911 on the MD0700R V1.4,
 *  see the ALIENTEK driver write-up) reached over a software (bit-banged) I2C
 *  bus built from four PS EMIO GPIO pins - the panel does NOT use a PS I2C
 *  controller:
 *
 *      EMIO[0] = SCL     R19
 *      EMIO[1] = SDA     P20  (PULLUP)
 *      EMIO[2] = CT_RST  M19
 *      EMIO[3] = CT_INT  U19
 *
 *  A small FT5206 / FT5x06 fallback is kept because ALIENTEK shipped the same
 *  7" 800x480 module with both Goodix and FocalTech controllers over the years;
 *  touch_hw_init() probes for both and reports what it found on the UART.
 *
 *  Coordinates handed back by touch_hw_scan() are already LCD pixels for the
 *  800x480 framebuffer, so callers can hit-test directly.
 * ===========================================================================
 */
#ifndef TOUCH_HW_H
#define TOUCH_HW_H

#include "xil_types.h"

/* Returns 0 on success (a controller answered), negative on failure. */
int touch_hw_init(void);

/* Re-print the detected controller / resolution summary on the UART. */
void touch_hw_info(void);

/* 1 = finger down (and *x,*y are valid LCD pixels), 0 = no touch. */
int touch_hw_scan(int *x, int *y);

/* Raw controller coordinates behind the most recent successful scan; for the
 * bring-up log line that lets the orientation macros be corrected on the bench. */
void touch_hw_last_raw(int *raw_x, int *raw_y);

#endif /* TOUCH_HW_H */
