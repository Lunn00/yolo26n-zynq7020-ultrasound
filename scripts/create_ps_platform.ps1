param([string]$XsctBat = $env:XSCT_BAT)
$ErrorActionPreference = 'Stop'
if (-not $XsctBat) {
    $tool = Get-Command xsct.bat -ErrorAction SilentlyContinue
    if ($tool) { $XsctBat = $tool.Source }
}
if (-not $XsctBat -or -not (Test-Path -LiteralPath $XsctBat)) {
    throw 'Set XSCT_BAT to the Vitis 2021.2 xsct.bat path.'
}
$script = Join-Path $PSScriptRoot 'create_ps_platform.tcl'
& $XsctBat $script
if ($LASTEXITCODE -ne 0) { throw 'XSCT platform generation failed.' }
$root = Split-Path -Parent $PSScriptRoot
$bspDir = Join-Path $root 'build\ps_platform\yolo26n_platform\export\yolo26n_platform\sw\yolo26n_platform\standalone_ps7_cortexa9_0'
if (-not (Test-Path -LiteralPath (Join-Path $bspDir 'bspinclude\include\xparameters.h'))) {
    throw "Generated BSP not found at expected location: $bspDir"
}
Write-Output "BSP directory: $bspDir"
