#ifndef UI_WIDGETS_H
#define UI_WIDGETS_H
#include "yolo_detect.h"
#include "filter_sw.h"

enum { UI_PREV, UI_NEXT, UI_RUN, UI_AUTO, UI_SELECT, UI_FILTER, UI_COMPARE, UI_BUTTONS };
enum { UI_KEY_DEL=10, UI_KEY_OK, UI_KEY_CANCEL };
typedef struct { char digits[5]; int length; char message[40]; } UiKeyboard;
typedef struct { int x, y, w, h; } UiRoi;
typedef struct {
    const char *status;
    int image, mode, have_image, have_result, confidence, boxes;
    int skipped, roi_ready;
    u8 r,g,b;
    u32 filter_us, pre_us, post_us, pl_us, call_us, wall_us;
} UiPanel;

/* One press per stable release; all modal transitions share the same gate. */
typedef struct { int latched,up_tracking,have_press; u32 up_since,last_press; } UiTouchGate;
int ui_touch_gate_accept(UiTouchGate *gate,int down,u32 now_ms);
void ui_crop_boxes(u8 *fb,const UiRoi *roi,int x,int y,int w,int h,const YoloBox *boxes,int count);
const char *ui_mode_name(int mode);
int ui_main_hit(int x,int y);
void ui_main_buttons(u8 *fb,int mode,int automatic);
void ui_main_panel(u8 *fb,const UiPanel *p);
int ui_keyboard_hit(int x,int y);
void ui_keyboard_reset(UiKeyboard *k);
void ui_keyboard_edit(UiKeyboard *k,int key);
int ui_keyboard_value(const UiKeyboard *k);
void ui_keyboard_draw(u8 *fb,const UiKeyboard *k);
int ui_roi_from_box(UiRoi *roi,const YoloBox *box,int w,int h);
int ui_roi_from_point(UiRoi *roi,int x,int y,int w,int h);
void ui_crop_draw(u8 *fb,const u8 *bgrx,int iw,int ih,int stride,
                  const UiRoi *roi,int x,int y,int w,int h);
int ui_compare_hit(int x,int y);
void ui_compare_buttons(u8 *fb,int mode);
#endif
