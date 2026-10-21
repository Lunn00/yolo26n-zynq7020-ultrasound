param([Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Destination -PathType Container)) { throw "Destination directory not found: $Destination" }
$files = @{
    'boot.bin' = 'BOOT.BIN'
    'yolo_i8.bin' = 'YOLO_I8.BIN'
    'yolo_q.bin' = 'YOLO_Q.BIN'
    'yolo_s.bin' = 'YOLO_S.BIN'
}
foreach ($sourceName in $files.Keys) {
    $source = Join-Path $root "sd_card/$sourceName"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $files[$sourceName]) -Force
}
Write-Host 'Copied runtime filenames to SD destination. Add an authorized numbered BMP before boot.'
