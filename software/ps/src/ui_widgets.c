#include "ui_widgets.h"
#include <stdio.h>
#include <string.h>

static void text(u8 *fb,int x,int y,const char *s,int scale,u8 r,u8 g,u8 b)
{ yolo_draw_text(fb,800,480,x,y,s,r,g,b,scale); }
static void rect(u8 *fb,int x,int y,int w,int h,u8 r,u8 g,u8 b)
{ yolo_fill_rect(fb,800,480,x,y,w,h,r,g,b); }
static void button(u8 *fb,int x,int y,int w,int h,const char *s,int scale,u8 r,u8 g,u8 b)
{
    rect(fb,x,y,w,h,r,g,b);
    yolo_draw_rect(fb,800,480,x,y,w,h,2,110,132,152);
    text(fb,x+(w-yolo_text_width(s,scale))/2,y+(h-7*scale)/2,s,scale,240,246,250);
}
const char *ui_mode_name(int mode)
{ return mode==FILT_MODE_OFF?"RAW":filter_sw_mode_name(mode); }
int ui_touch_gate_accept(UiTouchGate *g,int down,u32 now)
{
    if(!down) {
        if(!g->up_tracking){g->up_tracking=1;g->up_since=now;}
        if(g->latched && (u32)(now-g->up_since)>=80u)g->latched=0;
        return 0;
    }
    g->up_tracking=0;
    if(g->latched)return 0;
    g->latched=1;
    if(g->have_press && (u32)(now-g->last_press)<300u)return 0;
    g->have_press=1;g->last_press=now;return 1;
}
static const int main_order[7]={UI_PREV,UI_NEXT,UI_SELECT,UI_FILTER,UI_RUN,UI_AUTO,UI_COMPARE};
int ui_main_hit(int x,int y)
{
    if(x<474||x>=796||y<260||y>=476)return -1;
    int row=(y-260)/56;
    if((y-260)%56>=48)return -1;
    if(row==3)return UI_COMPARE;
    int col=x>=639;
    if((col==0&&x>=631)||(col==1&&x<639))return -1;
    return main_order[row*2+col];
}
void ui_main_buttons(u8 *fb,int mode,int automatic)
{
    const char *labels[7]={"PREV","NEXT","SELECT",ui_mode_name(mode),"RUN",automatic?"STOP":"AUTO","COMPARE"};
    rect(fb,470,256,330,224,18,25,34);
    for(int i=0;i<7;++i) {
        int id=main_order[i],x=i==6?474:474+(i%2)*165,y=260+(i/2)*56;
        u8 r=32,g=43,b=58;
        if(id==UI_RUN){r=24;g=95;b=63;}
        if(id==UI_SELECT){r=34;g=65;b=95;}
        if(id==UI_AUTO && automatic){r=145;g=45;b=38;}
        if(id==UI_FILTER){r=25;g=mode==FILT_MODE_OFF?65:80;b=120;}
        button(fb,x,y,i==6?322:157,48,labels[i],3,r,g,b);
    }
}
static void timing(u8 *fb,int y,const char *name,u32 us,int valid,int scale,u8 r,u8 g,u8 b)
{
    char s[40];
    if(valid) snprintf(s,sizeof(s),"%s %u MS",name,(unsigned)(us/1000));
    else snprintf(s,sizeof(s),"%s -- MS",name);
    text(fb,490,y,s,scale,r,g,b);
}
void ui_main_panel(u8 *fb,const UiPanel *p)
{
    char s[48];
    rect(fb,470,0,330,256,18,25,34);
    if(p->have_image) snprintf(s,sizeof(s),"IMG %d.BMP",p->image);
    else snprintf(s,sizeof(s),"NO IMAGE");
    text(fb,482,10,s,3,231,240,247);
    text(fb,482,42,p->status,2,p->r,p->g,p->b);
    snprintf(s,sizeof(s),"MODE %s",ui_mode_name(p->mode));
    text(fb,482,68,s,3,105,226,211);
    if(p->skipped) snprintf(s,sizeof(s),"FILTER SKIPPED - RAW");
    else if(p->have_result) snprintf(s,sizeof(s),"CONF %d%% BOX %d",p->confidence,p->boxes);
    else snprintf(s,sizeof(s),"PRESS RUN");
    text(fb,482,100,s,p->skipped?2:3,p->skipped?255:(p->have_result?0:238),p->skipped?120:(p->have_result?255:225),p->have_result&&!p->skipped?0:135);
    rect(fb,480,132,310,88,30,48,65);
    timing(fb,140,"PS FILTER",p->filter_us,p->have_image,2,185,211,230);
    timing(fb,165,"PL",p->pl_us,p->have_result,3,120,219,249);
    timing(fb,197,"INFER",p->call_us,p->have_result,2,233,241,247);
    text(fb,482,234,p->roi_ready?"ROI READY - COMPARE":"TAP TARGET TO ZOOM",2,255,207,100);
}
void ui_keyboard_reset(UiKeyboard *k) { memset(k,0,sizeof(*k)); }
void ui_keyboard_edit(UiKeyboard *k,int key)
{
    k->message[0]=0;
    if(key==UI_KEY_DEL) { if(k->length) k->digits[--k->length]=0; return; }
    if(key<0||key>9) return;
    if(k->length==4) { strcpy(k->message,"MAX 4 DIGITS"); return; }
    k->digits[k->length++]=(char)('0'+key); k->digits[k->length]=0;
}
int ui_keyboard_value(const UiKeyboard *k)
{
    if(!k->length) return -1;
    int n=0;
    for(int i=0;i<k->length;++i)n=n*10+k->digits[i]-'0';
    return n;
}
int ui_keyboard_hit(int x,int y)
{
    if(x>=208&&x<592&&y>=410&&y<452)return UI_KEY_CANCEL;
    if(x<208||x>=592||y<140||y>=400)return -1;
    int col=(x-208)/132,row=(y-140)/68;
    if(col>2||row>3||(x-208)%132>=120||(y-140)%68>=56)return -1;
    if(row<3)return row*3+col+1;
    return col==0?UI_KEY_DEL:col==1?0:UI_KEY_OK;
}
void ui_keyboard_draw(u8 *fb,const UiKeyboard *k)
{
    char s[40];
    rect(fb,170,20,460,440,22,33,48);
    yolo_draw_rect(fb,800,480,170,20,460,440,3,82,176,206);
    text(fb,196,36,"SELECT IMAGE",3,226,243,250);
    snprintf(s,sizeof(s),"ID: %s",k->length?k->digits:"----");
    text(fb,208,76,s,3,113,235,211);
    text(fb,208,114,k->message[0]?k->message:"ENTER 1 TO 9999",2,k->message[0]?255:165,175,150);
    for(int row=0;row<4;++row)for(int col=0;col<3;++col){
        char label[8];
        if(row<3)snprintf(label,sizeof(label),"%d",row*3+col+1);
        else strcpy(label,col==0?"DEL":col==1?"0":"OK");
        button(fb,208+col*132,140+row*68,120,56,label,3,32,(row==3&&col==2)?100:53,73);
    }
    button(fb,208,410,384,42,"CANCEL",2,73,46,50);
}
static int clamp(int v,int lo,int hi){return v<lo?lo:v>hi?hi:v;}
int ui_roi_from_point(UiRoi *r,int x,int y,int w,int h)
{
    if(w<=0||h<=0)return 0;
    r->w=w<160?w:160;r->h=h<160?h:160;
    r->x=clamp(x-r->w/2,0,w-r->w);r->y=clamp(y-r->h/2,0,h-r->h);return 1;
}
int ui_roi_from_box(UiRoi *r,const YoloBox *b,int w,int h)
{
    if(w<=0||h<=0||b->x2<=b->x1||b->y2<=b->y1)return 0;
    int bw=(int)(b->x2-b->x1+1),bh=(int)(b->y2-b->y1+1);
    r->w=clamp(bw+(bw+4)/5,16,w);r->h=clamp(bh+(bh+4)/5,16,h);
    if(r->w>w)r->w=w;
    if(r->h>h)r->h=h;
    r->x=clamp((int)((b->x1+b->x2)*0.5f)-r->w/2,0,w-r->w);
    r->y=clamp((int)((b->y1+b->y2)*0.5f)-r->h/2,0,h-r->h);return 1;
}
void ui_crop_draw(u8 *fb,const u8 *src,int iw,int ih,int stride,const UiRoi *r,int x,int y,int w,int h)
{
    rect(fb,x,y,w,h,8,13,20);
    if(!src||iw<=0||ih<=0||stride<iw*4||r->w<=0||r->h<=0||r->x<0||r->y<0||r->x+r->w>iw||r->y+r->h>ih)return;
    float scale=(float)w/r->w<(float)h/r->h?(float)w/r->w:(float)h/r->h;
    int nw=(int)(r->w*scale+0.5f),nh=(int)(r->h*scale+0.5f);
    int px=x+(w-nw)/2,py=y+(h-nh)/2;
    for(int dy=0;dy<nh;++dy)for(int dx=0;dx<nw;++dx){
        float fx=(dx+0.5f)/scale-0.5f,fy=(dy+0.5f)/scale-0.5f;
        if(fx<0)fx=0;
        if(fy<0)fy=0;
        int a=(int)fx,b=(int)fy;
        if(a>=r->w)a=r->w-1;
        if(b>=r->h)b=r->h-1;
        int aa=a+1<r->w?a+1:a,bb=b+1<r->h?b+1:b;
        float u=fx-a,v=fy-b;
        if(u>1)u=1;
        if(v>1)v=1;
        int ox=px+dx,oy=py+dy;
        if(ox<0||ox>=800||oy<0||oy>=480)continue;
        u8 *d=fb+(oy*800+ox)*3;
        const u8 *p=src+(r->y+b)*stride+(r->x+a)*4,*q=src+(r->y+b)*stride+(r->x+aa)*4;
        const u8 *s=src+(r->y+bb)*stride+(r->x+a)*4,*t=src+(r->y+bb)*stride+(r->x+aa)*4;
        for(int c=0;c<3;++c)d[c]=(u8)(p[c]*(1-u)*(1-v)+q[c]*u*(1-v)+s[c]*(1-u)*v+t[c]*u*v+0.5f);
    }
    yolo_draw_rect(fb,800,480,x,y,w,h,2,78,106,126);
}
void ui_crop_boxes(u8 *fb,const UiRoi *r,int x,int y,int w,int h,const YoloBox *boxes,int count)
{
    if(r->w<=0||r->h<=0)return;
    float scale=(float)w/r->w<(float)h/r->h?(float)w/r->w:(float)h/r->h;
    int nw=(int)(r->w*scale+0.5f),nh=(int)(r->h*scale+0.5f);
    int px=x+(w-nw)/2,py=y+(h-nh)/2;
    for(int i=0;i<count;++i){
        const YoloBox *b=&boxes[i];
        if(b->x2<r->x||b->x1>=r->x+r->w||b->y2<r->y||b->y1>=r->y+r->h)continue;
        int x1=(int)(px+(b->x1-r->x+0.5f)*scale-0.5f+0.5f);
        int x2=(int)(px+(b->x2-r->x+0.5f)*scale-0.5f+0.5f);
        int y1=(int)(py+(b->y1-r->y+0.5f)*scale-0.5f+0.5f);
        int y2=(int)(py+(b->y2-r->y+0.5f)*scale-0.5f+0.5f);
        /* Clip each original stroke: do not invent a frame at the crop edge. */
        for(int t=0;t<3;++t){
            int left=x1-t,right=x2+t,top=y1-t,bottom=y2+t;
            if(top>=py&&top<py+nh)rect(fb,clamp(left,px,px+nw),top,clamp(right+1,px,px+nw)-clamp(left,px,px+nw),1,0,255,0);
            if(bottom>=py&&bottom<py+nh)rect(fb,clamp(left,px,px+nw),bottom,clamp(right+1,px,px+nw)-clamp(left,px,px+nw),1,0,255,0);
            if(left>=px&&left<px+nw)rect(fb,left,clamp(top,py,py+nh),1,clamp(bottom+1,py,py+nh)-clamp(top,py,py+nh),0,255,0);
            if(right>=px&&right<px+nw)rect(fb,right,clamp(top,py,py+nh),1,clamp(bottom+1,py,py+nh)-clamp(top,py,py+nh),0,255,0);
        }
    }
}
int ui_compare_hit(int x,int y)
{
    if(y<424||y>=480)return -1;
    if(x>=8&&x<256)return 0;
    if(x>=276&&x<524)return 1;
    if(x>=544&&x<792)return 2;
    return -1;
}
void ui_compare_buttons(u8 *fb,int mode)
{
    char s[24];snprintf(s,sizeof(s),"MODE %s",ui_mode_name(mode));
    button(fb,8,424,248,56,s,3,25,74,105);
    button(fb,276,424,248,56,"RUN BOTH",3,24,95,63);
    button(fb,544,424,248,56,"BACK",3,42,54,65);
}
