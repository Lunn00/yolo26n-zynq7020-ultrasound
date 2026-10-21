/* Included after run_inference_once: application actions share its validated path. */
static void ui_repaint(u8 *fb)
{
    ui_restore_background(fb);
    if(g_have_infer && g_source_box_n) {
        YoloBox b[YOLO_MAX_BOXES];
        memcpy(b,g_source_boxes,g_source_box_n*sizeof(b[0]));
        yolo_map_boxes_to_rect(b,g_source_box_n,g_lbx_scale,g_lbx_pad_x,g_lbx_pad_y);
        yolo_draw_boxes(fb,LCD_WIDTH,LCD_HEIGHT,b,g_source_box_n);
    }
    if(g_roi_valid && g_roi_manual) {
        int x=(int)(g_roi.x*g_lbx_scale+g_lbx_pad_x);
        int y=(int)(g_roi.y*g_lbx_scale+g_lbx_pad_y);
        int w=(int)(g_roi.w*g_lbx_scale),h=(int)(g_roi.h*g_lbx_scale);
        yolo_draw_rect(fb,800,480,x,y,w,h,2,255,204,60);
    }
    draw_hud(fb); draw_buttons(fb);
    Xil_DCacheFlushRange((INTPTR)FB_ADDR,FRAME_SIZE);
}
static int ui_filter_for_compare(void)
{
    if(g_flt_mode==g_compare_mode) {
        g_compare_filter_us=g_flt_us; return 0;
    }
    XTime a,b;XTime_GetTime(&a);
    int rc=filter_sw_apply(g_compare_mode,(const u8 *)IMG_BUF,(u8 *)FLT_BUF,img_w,img_h,img_row);
    XTime_GetTime(&b);g_compare_filter_us=elapsed_us(a,b);
    g_flt_mode=rc==0?g_compare_mode:-1;g_flt_us=g_compare_filter_us;
    /* FLT_BUF now contains the comparison mode, not necessarily the main mode. */
    g_compare_overwrote_filter=1;
    return rc;
}
static void ui_compare_draw(u8 *fb)
{
    char s[80];
    yolo_fill_rect(fb,800,480,0,0,800,480,18,25,34);
    snprintf(s,sizeof(s),"COMPARE %d.BMP",g_img_idx);
    yolo_draw_text(fb,800,480,12,8,s,223,238,247,3);
    yolo_draw_text(fb,800,480,12,38,"SAME TARGET - IMAGE MAX CONF",255,207,100,2);
    UiResult *raw=&g_results[0],*enh=&g_results[ui_cache_index(g_compare_mode)];
    if(raw->valid)snprintf(s,sizeof(s),"RAW %d%%",raw->conf);
    else snprintf(s,sizeof(s),"RAW - RUN BOTH");
    yolo_draw_text(fb,800,480,12,62,s,231,238,242,3);
    if(enh->valid)snprintf(s,sizeof(s),"%s %d%%",ui_mode_name(g_compare_mode),enh->conf);
    else snprintf(s,sizeof(s),"%s - RUN BOTH",ui_mode_name(g_compare_mode));
    yolo_draw_text(fb,800,480,412,62,s,112,231,206,3);
    ui_crop_draw(fb,(const u8 *)IMG_BUF,img_w,img_h,img_row,&g_roi,8,92,384,282);
    if(raw->valid)ui_crop_boxes(fb,&g_roi,8,92,384,282,raw->boxes,raw->count);
    if(g_compare_filter_ok){
        ui_crop_draw(fb,(const u8 *)FLT_BUF,img_w,img_h,img_row,&g_roi,408,92,384,282);
        if(enh->valid)ui_crop_boxes(fb,&g_roi,408,92,384,282,enh->boxes,enh->count);
    }else yolo_draw_text(fb,800,480,420,170,"FILTER UNAVAILABLE",255,100,70,2);
    yolo_fill_rect(fb,800,480,8,382,384,34,30,48,65);
    yolo_fill_rect(fb,800,480,408,382,384,34,30,48,65);
    if(raw->valid)snprintf(s,sizeof(s),"PS 0 MS PL %u MS",(unsigned)(raw->pl_us/1000));
    else snprintf(s,sizeof(s),"PRESS RUN BOTH");
    yolo_draw_text(fb,800,480,18,392,s,185,211,230,2);
    if(enh->valid && g_compare_filter_ok)snprintf(s,sizeof(s),"PS %u MS PL %u MS",(unsigned)(g_compare_filter_us/1000),(unsigned)(enh->pl_us/1000));
    else snprintf(s,sizeof(s),"PRESS RUN BOTH");
    yolo_draw_text(fb,800,480,418,392,s,185,211,230,2);
    ui_compare_buttons(fb,g_compare_mode);
    Xil_DCacheFlushRange((INTPTR)FB_ADDR,FRAME_SIZE);
}
static void ui_compare_close(u8 *fb)
{
    g_compare=0;
    apply_filter_mode();
    letterbox_to_fb(fb);ui_cache_background(fb);
    g_compare_overwrote_filter=0;
    ui_restore_result(g_filter_mode);
    snprintf(g_status_txt,sizeof(g_status_txt),g_have_infer?(g_box_n>0?"TARGET DETECTED":"NO TARGET"):"YOLO READY");
    ui_repaint(fb);
}
static void ui_open_compare(u8 *fb,int weights_ok)
{
    g_auto_run=0;
    if(!g_bmp_ok){show_status(fb,"LOAD IMAGE FIRST",255,120,60);return;}
    if(!g_roi_valid && !g_have_infer) run_inference_once(fb,weights_ok);
    if(!g_roi_valid){show_status(fb,"TAP IMAGE FOR ROI",255,190,80);return;}
    g_compare=1;
    g_compare_mode=g_filter_mode==FILT_MODE_OFF?FILT_MODE_GAUSS5:g_filter_mode;
    g_compare_overwrote_filter=0;
    g_compare_filter_ok=ui_filter_for_compare()==0;
    ui_compare_draw(fb);
}
static void ui_compare_action(int action,u8 *fb,int weights_ok)
{
    if(action==2){ui_compare_close(fb);return;}
    if(action==0){
        g_compare_mode=g_compare_mode==FILT_MODE_GAUSS5?FILT_MODE_RGBGAUSS:FILT_MODE_GAUSS5;
        /* A previous compare operation may already have overwritten FLT_BUF. */
        g_compare_filter_ok=ui_filter_for_compare()==0;
        ui_compare_draw(fb);return;
    }
    if(action==1){
        int original_mode=g_filter_mode;
        g_compare=0;
        for(int i=0;i<2;++i){
            g_filter_mode=i==0?FILT_MODE_OFF:g_compare_mode;
            g_have_infer=0;g_box_n=0;g_max_conf_pct=0;
            apply_filter_mode();render_image(fb,"COMPARE RUNNING");
            run_inference_once(fb,weights_ok);
        }
        g_compare_filter_us=g_filter_us;
        g_compare_filter_ok=!g_filter_skipped;
        g_filter_mode=original_mode;
        g_compare_overwrote_filter=1;
        /* Do not let the pointer claim that the buffer still has original_mode. */
        g_img_src=(const u8 *)IMG_BUF;
        g_compare=1;ui_compare_draw(fb);
    }
}
static void ui_key_action(int key,u8 *fb)
{
    if(key==UI_KEY_CANCEL){g_keypad=0;ui_repaint(fb);return;}
    if(key==UI_KEY_OK){
        int idx=ui_keyboard_value(&g_keyboard);
        if(idx<1||idx>MAX_IMG_IDX)strcpy(g_keyboard.message,"USE 1 TO 9999");
        else if(!image_file_exists(idx))strcpy(g_keyboard.message,"BMP NOT FOUND");
        else {
            g_keypad=0;g_img_idx=idx;g_catalog_ready=0;
            g_bmp_ok=load_and_show(fb,idx)==0;
            xil_printf("[UI] SELECT -> %d.bmp\r\n",idx);return;
        }
    }else ui_keyboard_edit(&g_keyboard,key);
    ui_keyboard_draw(fb,&g_keyboard);
    Xil_DCacheFlushRange((INTPTR)FB_ADDR,FRAME_SIZE);
}
static void ui_image_touch(int x,int y,u8 *fb)
{
    if(!g_bmp_ok||x<g_lbx_pad_x||y<g_lbx_pad_y)return;
    int sx=(int)((x-g_lbx_pad_x)/g_lbx_scale),sy=(int)((y-g_lbx_pad_y)/g_lbx_scale);
    if(sx<0||sx>=img_w||sy<0||sy>=img_h)return;
    g_auto_run=0;
    g_roi_valid=ui_roi_from_point(&g_roi,sx,sy,img_w,img_h);g_roi_manual=1;
    xil_printf("[UI] ROI manual x=%d y=%d w=%d h=%d\r\n",g_roi.x,g_roi.y,g_roi.w,g_roi.h);
    show_status(fb,"ROI SELECTED",110,225,210);ui_repaint(fb);
}
