# Build and verify the A-track image on Windows (requires Docker Desktop).
# Usage:  powershell -ExecutionPolicy Bypass -File docker/build.ps1
#         powershell -ExecutionPolicy Bypass -File docker/build.ps1 -Save
#
# The submitted archive must be an UNCOMPRESSED `docker save` tar:
#   docker save -o rsi-a-cpu-1.0.tar rsi-a-cpu:1.0
# Do not submit a ZIP, a source tar, or a `docker export` archive.

param([switch]$Save)

$ErrorActionPreference = "Stop"
$Name = "rsi-a-cpu:1.0"

Write-Host "[1/4] build $Name"
docker build -t $Name -f docker/Dockerfile .

Write-Host "[2/4] selfcheck inside the image (version, checkpoint, official tests)"
docker run --rm --entrypoint python $Name scripts/selfcheck.py

Write-Host "[3/4] describe mode (read-only capability manifest)"
docker run --rm --entrypoint python $Name scripts/run_harness.py describe

if ($Save) {
    Write-Host "[4/4] export image tar"
    docker save -o rsi-a-cpu-1.0.tar $Name
    Write-Host "exported rsi-a-cpu-1.0.tar"
} else {
    Write-Host "[4/4] skip export (add -Save to export)"
}
