param(
    [Parameter(Mandatory = $true)][string]$BspDir,
    [string]$Gcc = $env:ARM_GCC
)
$ErrorActionPreference = 'Stop'
if (-not $Gcc) {
    $tool = Get-Command arm-none-eabi-gcc.exe -ErrorAction SilentlyContinue
    if ($tool) { $Gcc = $tool.Source }
}
if (-not $Gcc -or -not (Test-Path -LiteralPath $Gcc)) {
    throw 'Set ARM_GCC to the Vitis 2021.2 arm-none-eabi-gcc.exe path.'
}
$BspDir = (Resolve-Path -LiteralPath $BspDir -ErrorAction Stop).Path
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root 'software\ps\src'
$bspInclude = Join-Path $BspDir 'bspinclude\include'
$bspLib = Join-Path $BspDir 'bsplib\lib'
foreach ($path in @($bspInclude, $bspLib)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing BSP directory: $path" }
}
$stage = Join-Path $root 'build\ps'
$objectsDir = Join-Path $stage 'objects'
New-Item -ItemType Directory -Path $objectsDir -Force | Out-Null
$objects = @()
foreach ($name in @('main','filter_sw','touch_hw','yolo_detect','yolo_selftest_vectors','ui_widgets')) {
    $object = Join-Path $objectsDir "$name.o"
    & $Gcc -Wall -O2 -g3 -c -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard "-I$bspInclude" (Join-Path $src "$name.c") -o $object
    if ($LASTEXITCODE -ne 0) { throw "ARM compilation failed: $name" }
    $objects += $object
}
$elf = Join-Path $stage 'yolo26_lcd_ui.elf'
Push-Location $src
try {
    & $Gcc -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard '-Wl,-build-id=none' '-specs=xilinx.spec' '-Wl,-T' '-Wl,lscript.ld' "-L$bspLib" -o $elf @objects '-Wl,--start-group,-lxil,-lgcc,-lc,--end-group' '-Wl,--start-group,-lxilffs,-lxil,-lgcc,-lc,--end-group'
    if ($LASTEXITCODE -ne 0) { throw 'ARM linking failed.' }
} finally {
    Pop-Location
}
Write-Output "Built $elf"
