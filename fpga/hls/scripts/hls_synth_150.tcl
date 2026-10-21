# Synthesize the delivered YOLO26n INT8 kernel for xc7z020 at 150 MHz.
set script_dir [file dirname [file normalize [info script]]]
set hls_root [file normalize [file join $script_dir ..]]
set src_dir [file join $hls_root src]
set build_dir [file join $hls_root build_synth_p64_150]
file mkdir $build_dir
cd $build_dir

open_project -reset yolo26_conv1x1_i8_224_p64_150
set_top yolo8n_hls_i8
set cflags {-DYOLO_I8_CONV_TILE=64 -DYOLO_I8_FUSE_SPPF_ONCHIP=0}
add_files [file join $src_dir yolo_hls_i8_26.cpp] -cflags $cflags
open_solution solution1 -flow_target vivado
set_part {xc7z020clg400-2}
create_clock -period 6.666 -name default
csynth_design
exit
