# Export the synthesized YOLO26n kernel as a Vivado IP package.
set script_dir [file dirname [file normalize [info script]]]
set hls_root [file normalize [file join $script_dir ..]]
set build_dir [file join $hls_root build_synth_p64_150]
cd $build_dir
open_project yolo26_conv1x1_i8_224_p64_150
set_top yolo8n_hls_i8
open_solution solution1 -flow_target vivado
config_export -format ip_catalog -rtl verilog -vendor xilinx.com -library hls -version 1.0
export_design -rtl verilog -format ip_catalog
exit
