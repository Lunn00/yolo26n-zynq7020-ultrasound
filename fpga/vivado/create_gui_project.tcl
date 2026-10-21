# Create an editable Vivado 2021.2 GUI project from the shipped block design.
# This script never synthesizes or changes the delivered BOOT.BIN / bitstream.
set root [file dirname [file normalize [info script]]]
set project_dir [file join $root gui_project_alias]
set project_name yolo26n_gui_alias
set xpr [file join $project_dir $project_name.xpr]
if {[file exists $xpr]} {
    error "GUI project already exists: $xpr"
}
create_project $project_name $project_dir -part xc7z020clg400-2
set_property ip_repo_paths [list [file join $root ip_repository_gui_alias]] [current_project]
update_ip_catalog -rebuild

set bd_dir [file join $project_dir ${project_name}.srcs sources_1 bd system]
file mkdir [file dirname $bd_dir]
file copy [file join $root project_sources gui_alias_bd] $bd_dir
set bd [file join $bd_dir system.bd]
add_files -norecurse $bd
add_files -norecurse [file join $root project_sources system_wrapper.v]
add_files -fileset constrs_1 -norecurse [file join $root project_sources pin.xdc]
set_property top system_wrapper [current_fileset]
open_bd_design [get_files system.bd]
set accelerator [get_bd_cells -quiet yolo8n_hls_i8_0]
if {[llength $accelerator] != 1} {error "Expected one packaged accelerator instance"}
set_property NAME yolo26n_hls_i8_0 $accelerator
set actual_vlnv [get_property VLNV [get_bd_cells yolo26n_hls_i8_0]]
if {$actual_vlnv ne "xilinx.com:hls:yolo26n_hls_i8:1.0"} {error "Unexpected accelerator IP: $actual_vlnv"}
validate_bd_design
save_bd_design
update_compile_order -fileset sources_1
puts "GUI_PROJECT_READY: $xpr"
close_project
exit
