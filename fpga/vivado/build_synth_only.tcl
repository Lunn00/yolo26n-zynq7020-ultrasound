# Recreate the delivered Block Design with the selected HLS IP at 150 MHz.
set root [file dirname [file normalize [info script]]]
set out [file join $root output]
file mkdir $out
create_project yolo26_speed [file join $root project] -part xc7z020clg400-2 -force
set_property ip_repo_paths [list [file join $root ip_repository]] [current_project]
update_ip_catalog -rebuild
set bd_dir [file join $root project yolo26_speed.srcs sources_1 bd system]
file mkdir $bd_dir
set bd [file join $bd_dir system.bd]
file copy -force [file join $root project_sources system.bd] $bd
add_files -norecurse $bd
add_files -norecurse [file join $root project_sources system_wrapper.v]
add_files -fileset constrs_1 -norecurse [file join $root project_sources pin.xdc]
set_property top system_wrapper [current_fileset]
open_bd_design [get_files system.bd]
set ip [get_ips -quiet *yolo8n_hls_i8*]
if {[llength $ip] == 0} {error "YOLO HLS IP missing from block design"}
upgrade_ip $ip -log [file join $out ip_upgrade.log]
validate_bd_design
save_bd_design
generate_target all [get_files system.bd]
update_compile_order -fileset sources_1

# A same-VLNV HLS IP can otherwise leave a stale cached module in the BD.
set fh [open [file join $root ip_repository hdl verilog yolo8n_hls_i8.v] r]
set expected [read $fh]
close $fh
set found 0
foreach f [concat \
    [glob -nocomplain [file join $root project *.gen sources_1 bd system ipshared * hdl verilog yolo8n_hls_i8.v]] \
    [glob -nocomplain [file join $root project yolo26_speed.gen sources_1 bd system ipshared * hdl verilog yolo8n_hls_i8.v]] \
    [glob -nocomplain [file join $root project_sources ipshared * hdl verilog yolo8n_hls_i8.v]]] {
    set fh [open $f r]
    set got [read $fh]
    close $fh
    if {[string equal $expected $got]} {set found 1; puts "CANDIDATE_IP_MATCH: $f"}
}
if {!$found} {error "Generated BD IP does not match candidate export"}

synth_design -top system_wrapper -part xc7z020clg400-2 -flatten_hierarchy rebuilt -directive PerformanceOptimized -retiming
report_utilization -file [file join $out utilization_synth.rpt]
write_checkpoint -force [file join $out post_synth.dcp]
puts SYNTH_ONLY_DONE
close_project
exit

