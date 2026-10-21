# Simulate the delivered 224-input YOLO26n INT8 kernel against an authorized
# input/reference pair. Match the synthesis settings, then stop before csynth.
set script_dir [file dirname [file normalize [info script]]]
set hls_root   [file normalize [file join $script_dir ..]]
set src_dir    [file join $hls_root src]
set build_dir  [file join $hls_root build_csim_p64_150]
file mkdir $build_dir
cd $build_dir

set tbargs [file normalize [file join $hls_root data]]/

set period 6.666
open_project -reset yolo26n_i8_224_p64_150_csim
set_top yolo8n_hls_i8
set cflags {-DYOLO_I8_CONV_TILE=64 -DYOLO_I8_FUSE_SPPF_ONCHIP=0}
add_files [file join $src_dir yolo_hls_i8_26.cpp] -cflags $cflags
add_files -tb [file join $src_dir tb_conv1x1.cpp] -cflags $cflags
open_solution solution1 -flow_target vivado
set_part {xc7z020clg400-2}
create_clock -period $period -name default
csim_design -argv $tbargs
exit
