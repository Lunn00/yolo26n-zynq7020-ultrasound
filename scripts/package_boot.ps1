param(
    [string]$BootgenBat = $env:BOOTGEN_BAT,
    [string]$Bitstream,
    [string]$FsblElf,
    [string]$ApplicationElf
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BootgenBat) {
    $tool = Get-Command bootgen.bat -ErrorAction SilentlyContinue
    if ($tool) { $BootgenBat = $tool.Source }
}
if (-not $BootgenBat -or -not (Test-Path -LiteralPath $BootgenBat)) {
    throw 'Set BOOTGEN_BAT to the Vitis 2021.2 bootgen.bat path.'
}
if (-not $Bitstream) { $Bitstream = Join-Path $root 'prebuilt\system_wrapper.bit' }
if (-not $FsblElf) { $FsblElf = Join-Path $root 'prebuilt\fsbl.elf' }
if (-not $ApplicationElf) { $ApplicationElf = Join-Path $root 'prebuilt\yolo26_lcd_ui.elf' }
foreach ($inputFile in @($Bitstream, $FsblElf, $ApplicationElf)) {
    if (-not (Test-Path -LiteralPath $inputFile)) { throw "Missing input: $inputFile" }
}
$stage = Join-Path $root 'build\bootgen'
New-Item -ItemType Directory -Path $stage -Force | Out-Null
Copy-Item -LiteralPath $FsblElf -Destination (Join-Path $stage 'fsbl.elf') -Force
Copy-Item -LiteralPath $Bitstream -Destination (Join-Path $stage 'system_wrapper.bit') -Force
Copy-Item -LiteralPath $ApplicationElf -Destination (Join-Path $stage 'yolo26_lcd_ui.elf') -Force
$bif = Join-Path $stage 'boot.bif'
$content = "the_ROM_image:`n{`n  [bootloader] $stage\fsbl.elf`n  $stage\system_wrapper.bit`n  $stage\yolo26_lcd_ui.elf`n}`n"
[IO.File]::WriteAllText($bif, $content, [Text.Encoding]::ASCII)
$boot = Join-Path $stage 'BOOT.BIN'
& $BootgenBat -image $bif -arch zynq -o $boot -w on
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $boot)) { throw 'Bootgen failed.' }
Get-FileHash -Algorithm SHA256 -LiteralPath $boot
