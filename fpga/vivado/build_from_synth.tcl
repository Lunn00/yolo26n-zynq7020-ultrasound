set root [file dirname [file normalize [info script]]]
set out [file join $root output]
open_checkpoint [file join $out post_synth.dcp]
opt_design -directive Explore
place_design -directive ExtraTimingOpt
phys_opt_design -directive AggressiveExplore
report_utilization -file [file join $out utilization_placed.rpt]
report_timing_summary -file [file join $out timing_placed.rpt] -report_unconstrained
write_checkpoint -force [file join $out placed.dcp]
route_design -directive Explore
report_timing_summary -file [file join $out timing_summary.rpt] -report_unconstrained
report_utilization -file [file join $out utilization.rpt]
write_checkpoint -force [file join $out routed.dcp]
write_bitstream -force [file join $out system_wrapper.bit]
puts "CANDIDATE_BIT: [file join $out system_wrapper.bit]"
exit
