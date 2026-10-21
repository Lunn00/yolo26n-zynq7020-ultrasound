param(
    [string]$VitisHlsBat = $env:VITIS_HLS_BAT,
    [switch]$SkipCsim
)
$ErrorActionPreference = 'Stop'
if (-not $VitisHlsBat) {
    $tool = Get-Command vitis_hls.bat -ErrorAction SilentlyContinue
    if ($tool) { $VitisHlsBat = $tool.Source }
}
if (-not $VitisHlsBat -or -not (Test-Path -LiteralPath $VitisHlsBat)) {
    throw 'Set VITIS_HLS_BAT to the Vitis HLS 2021.2 vitis_hls.bat path.'
}
$hls = Join-Path (Split-Path -Parent $PSScriptRoot) 'fpga\hls'
$scripts = @('hls_synth_150.tcl', 'hls_export_150.tcl')
if (-not $SkipCsim) {
    foreach ($name in @('verify_input_i8.bin', 'verify_pred_i8.bin')) {
        if (-not (Test-Path -LiteralPath (Join-Path $hls "data\$name"))) {
            throw "C simulation needs fpga/hls/data/$name (review_before_publish/test_vectors)."
        }
    }
    $scripts = @('csim_yolo26_p64_150.tcl') + $scripts
}
foreach ($name in $scripts) {
    & $VitisHlsBat -f (Join-Path $hls "scripts\$name")
    if ($LASTEXITCODE -ne 0) { throw "Vitis HLS failed: $name" }
}
$ipDir = Join-Path $hls 'build_synth_p64_150\yolo26_conv1x1_i8_224_p64_150\solution1\impl\ip'
$ipZip = Join-Path $ipDir 'xilinx_com_hls_yolo8n_hls_i8_1_0.zip'
if (-not (Test-Path -LiteralPath $ipZip)) { throw "HLS IP export not found: $ipZip" }
Write-Output "HLS IP archive: $ipZip"
