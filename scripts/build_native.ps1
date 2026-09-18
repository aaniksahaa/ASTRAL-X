param(
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "native"),
    [string]$CudaArch = "all-major"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# This script lives in scripts/; the repository root is one level up.
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc was not found. Install a CUDA toolkit or build a CPU-only artifact."
}

if ($env:JAVA_HOME) {
    $Jdk = $env:JAVA_HOME
} else {
    $Javac = (Get-Command javac -ErrorAction Stop).Source
    $Jdk = Split-Path (Split-Path $Javac -Parent) -Parent
}

$JniInclude = Join-Path $Jdk "include"
$JniPlatformInclude = Join-Path $JniInclude "win32"
if (-not (Test-Path (Join-Path $JniPlatformInclude "jni_md.h"))) {
    throw "Could not find Windows JNI headers under $JniPlatformInclude"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$MinCudaCc = 0
if ($CudaArch -eq "all-major") {
    $Supported = & nvcc --list-gpu-arch |
        ForEach-Object { if ($_ -match '^compute_(\d+)$') { [int]$Matches[1] } } |
        Sort-Object
    if ($Supported) { $MinCudaCc = $Supported[0] }
} elseif ($CudaArch -match '^(?:sm|compute)_(\d+)$') {
    $MinCudaCc = [int]$Matches[1]
}

$Common = @(
    "-arch=$CudaArch",
    "-O3",
    "--shared",
    "-Xcompiler=/MD",
    "-I$JniInclude",
    "-I$JniPlatformInclude",
    "-DASTRALX_MIN_CUDA_CC=$MinCudaCc"
)

$Libraries = @(
    @{ Source = "astralx_weight.cu";     Output = "astralx_weight.dll" },
    @{ Source = "astralx_dp.cu";         Output = "astralx_dp.dll" },
    @{ Source = "astralx_dist.cu";       Output = "astralx_dist.dll" },
    @{ Source = "astralx_similarity.cu"; Output = "astralx_sim.dll" }
)

Write-Host "=== Building ASTRAL-X native GPU libraries ==="
Write-Host "  JDK         : $Jdk"
Write-Host "  CUDA arch   : $CudaArch"
Write-Host "  Minimum CC  : $MinCudaCc"
Write-Host "  Output      : $OutputDir"

foreach ($Library in $Libraries) {
    $Source = Join-Path $RepoRoot (Join-Path "src/native" $Library.Source)
    $Output = Join-Path $OutputDir $Library.Output
    Write-Host "  Building    : $Source -> $Output"
    & nvcc @Common "-o" $Output $Source
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed while building $($Library.Source) (exit $LASTEXITCODE)"
    }
    Write-Host "  OK"
}

Write-Host "=== Native build complete ==="
