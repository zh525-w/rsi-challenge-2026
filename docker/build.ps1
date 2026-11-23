# 构建并提交镜像（Windows PowerShell）
# 用法：powershell -File docker\build.ps1 [-Tag rsi-cpu-abc:2.0] [-ExportTo ..\rsi-cpu-abc-2.0.tar]
param(
    [string]$Tag = "rsi-cpu-abc:2.0",
    [string]$ExportTo = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
docker build -t $Tag -f "$root\docker\Dockerfile" $root
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
if ($ExportTo) {
    docker save -o $ExportTo $Tag
    if ($LASTEXITCODE -ne 0) { throw "docker save failed" }
    Get-Item $ExportTo | Select-Object Name, Length
}
Write-Output "built $Tag"
