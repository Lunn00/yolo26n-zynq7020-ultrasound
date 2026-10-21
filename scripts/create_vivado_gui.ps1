param(
    [string]$VivadoBat = $env:VIVADO_BAT,
    [switch]$NoOpen
)
$ErrorActionPreference = 'Stop'
if (-not $VivadoBat) {
    $tool = Get-Command vivado.bat -ErrorAction SilentlyContinue
    if ($tool) { $VivadoBat = $tool.Source }
}
if (-not $VivadoBat -or -not (Test-Path -LiteralPath $VivadoBat)) {
    throw 'Set VIVADO_BAT to the Vivado 2021.2 vivado.bat path.'
}
$root = Split-Path -Parent $PSScriptRoot
$vivado = Join-Path $root 'fpga\vivado'
$ipRepository = Join-Path $vivado 'ip_repository_gui_alias'
$ipArchive = Join-Path $root 'prebuilt\yolo26n_vivado_ip_repository.zip'
if (-not (Test-Path -LiteralPath (Join-Path $ipRepository 'component.xml'))) {
    New-Item -ItemType Directory -Path $ipRepository -Force | Out-Null
    Expand-Archive -LiteralPath $ipArchive -DestinationPath $ipRepository -Force
}
$component = Join-Path $ipRepository 'component.xml'
$oldName = '<spirit:name>yolo8n_hls_i8</spirit:name>'
$newName = '<spirit:name>yolo26n_hls_i8</spirit:name>'
$xml = [System.IO.File]::ReadAllText($component)
if ($xml.Contains($oldName)) {
    [System.IO.File]::WriteAllText($component, $xml.Replace($oldName, $newName))
} elseif (-not $xml.Contains($newName)) {
    throw 'Unexpected HLS IP component name in archive.'
}
$xpr = Join-Path $vivado 'gui_project_alias\yolo26n_gui_alias.xpr'
if (-not (Test-Path -LiteralPath $xpr)) {
    $batchArgs = @('-mode','batch','-source',(Join-Path $vivado 'create_gui_project.tcl'),'-nojournal','-nolog')
    $process = Start-Process -FilePath $VivadoBat -ArgumentList $batchArgs -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $xpr)) {
        throw 'Vivado GUI project creation failed.'
    }
}
Write-Output "GUI project: $xpr"
if (-not $NoOpen) {
    Start-Process -FilePath $VivadoBat -ArgumentList @('-mode','gui','-project',$xpr,'-source',(Join-Path $vivado 'open_gui_bd.tcl')) -WindowStyle Normal
}
