param(
    [string]$VivadoBat = $env:VIVADO_BAT,
    [string]$IpArchive
)
$ErrorActionPreference = 'Stop'
if (-not $VivadoBat) {
    $tool = Get-Command vivado.bat -ErrorAction SilentlyContinue
    if ($tool) { $VivadoBat = $tool.Source }
}
if (-not $VivadoBat -or -not (Test-Path -LiteralPath $VivadoBat)) {
    throw 'Set VIVADO_BAT to the Vivado 2021.2 vivado.bat path.'
}
$vivado = Join-Path (Split-Path -Parent $PSScriptRoot) 'fpga\vivado'
$root = Split-Path -Parent $PSScriptRoot
if (-not $IpArchive) { $IpArchive = Join-Path $root 'prebuilt\yolo26n_vivado_ip_repository.zip' }
$IpArchive = (Resolve-Path -LiteralPath $IpArchive -ErrorAction Stop).Path
$ipRepository = Join-Path $vivado 'ip_repository'
$resolvedVivado = (Resolve-Path -LiteralPath $vivado).Path
$resolvedIpRepository = [IO.Path]::GetFullPath($ipRepository)
if (-not $resolvedIpRepository.StartsWith($resolvedVivado + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unexpected IP repository path: $resolvedIpRepository"
}
if (Test-Path -LiteralPath $resolvedIpRepository) {
    Remove-Item -LiteralPath $resolvedIpRepository -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedIpRepository -Force | Out-Null
Expand-Archive -LiteralPath $IpArchive -DestinationPath $resolvedIpRepository -Force
if (-not (Test-Path -LiteralPath (Join-Path $resolvedIpRepository 'component.xml'))) {
    throw "HLS IP archive has no component.xml at its root: $IpArchive"
}
foreach ($name in @('build_synth_only.tcl', 'build_from_synth.tcl')) {
    & $VivadoBat -mode batch -source (Join-Path $vivado $name) -nojournal -nolog
    if ($LASTEXITCODE -ne 0) { throw "Vivado failed: $name" }
}
