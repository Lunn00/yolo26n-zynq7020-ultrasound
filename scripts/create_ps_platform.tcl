# Run with XSCT 2021.2. All inputs and outputs are relative to this repository.
set repo [file normalize [file join [file dirname [info script]] ..]]
set xsa [file join $repo prebuilt ps_platform_0928.xsa]
set out [file join $repo build ps_platform]
set name yolo26n_platform

if {![file exists $xsa]} {
    puts stderr "Missing hardware platform: $xsa"
    exit 1
}

if {[catch {
    platform create -name $name -hw $xsa -out $out
    platform write
    domain create -name standalone_ps7_cortexa9_0 -os standalone -proc ps7_cortexa9_0 -runtime cpp -arch 32-bit
    domain active standalone_ps7_cortexa9_0
    bsp setlib -name xilffs -ver 4.6
    bsp write
    platform generate
} problem]} {
    puts stderr "PS platform generation failed: $problem"
    exit 1
}
puts "Generated PS platform under $out"
exit 0
