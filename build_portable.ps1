param(
    [switch]$CpuOnly,
    [switch]$WithCuda,
    [string]$CudaArch = "all-major",
    [string]$OutputDir = (Join-Path $PSScriptRoot "dist"),
    [switch]$NoArchive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($CpuOnly -and $WithCuda) {
    throw "Use either -CpuOnly or -WithCuda, not both."
}

foreach ($Tool in @("javac", "jar", "jlink", "jpackage")) {
    if (-not (Get-Command $Tool -ErrorAction SilentlyContinue)) {
        throw "'$Tool' is required to build the artifact (JDK 21 or newer expected)."
    }
}

$JavacVersion = (& javac -version 2>&1).ToString()
if ($JavacVersion -notmatch '(\d+)') {
    throw "Could not determine javac version from '$JavacVersion'."
}
if ([int]$Matches[1] -lt 21) {
    throw "JDK 21 or newer is required; found $JavacVersion."
}

$MainSource = Get-Content (Join-Path $PSScriptRoot "src/astralx/Main.java") -Raw
if ($MainSource -notmatch 'VERSION\s*=\s*"([^"]+)"') {
    throw "Could not determine ASTRAL-X version from Main.java."
}
$Version = $Matches[1]
$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
$PlatformArch = switch ($Arch) {
    "x64"   { "x86_64" }
    "arm64" { "arm64" }
    default { $Arch }
}

$NvccAvailable = [bool](Get-Command nvcc -ErrorAction SilentlyContinue)
$IncludeCuda = -not $CpuOnly -and $NvccAvailable
if ($WithCuda -and -not $NvccAvailable) {
    throw "-WithCuda was requested, but nvcc was not found."
}
if ($IncludeCuda -and $PlatformArch -ne "x86_64") {
    if ($WithCuda) {
        throw "The Windows CUDA artifact is currently supported only on x86-64."
    }
    $IncludeCuda = $false
}

$Capability = if ($IncludeCuda) { "cuda-with-cpu-fallback" } else { "cpu" }
# One public artifact per OS/CPU family. The CUDA-enabled image already carries
# the complete CPU implementation and falls back automatically.
$Artifact = "astralx-$Version-windows-$PlatformArch"
$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("astralx-portable-" + [guid]::NewGuid())

try {
    New-Item -ItemType Directory -Force -Path $Work, $OutputDir | Out-Null
    $Build = Join-Path $Work "classes"
    $InputDir = Join-Path $Work "input"
    $Runtime = Join-Path $Work "runtime"
    $JpackageOut = Join-Path $Work "jpackage"
    New-Item -ItemType Directory -Force -Path $Build, $InputDir, $JpackageOut | Out-Null

    Write-Host "=== ASTRAL-X portable build ==="
    Write-Host "  Version      : $Version"
    Write-Host "  Platform     : windows-$PlatformArch"
    Write-Host "  CUDA bundle  : $IncludeCuda"
    Write-Host "  Artifact     : $Artifact"

    $Sources = Get-ChildItem (Join-Path $PSScriptRoot "src") -Recurse -Filter *.java |
        ForEach-Object { $_.FullName }
    & javac -d $Build -sourcepath (Join-Path $PSScriptRoot "src") @Sources
    if ($LASTEXITCODE -ne 0) { throw "javac failed (exit $LASTEXITCODE)" }

    $JarPath = Join-Path $InputDir "astralx.jar"
    & jar --create --file $JarPath --main-class astralx.Main -C $Build .
    if ($LASTEXITCODE -ne 0) { throw "jar failed (exit $LASTEXITCODE)" }

    if ($IncludeCuda) {
        & (Join-Path $PSScriptRoot "build_native.ps1") -OutputDir $InputDir -CudaArch $CudaArch
    }

    & jlink --add-modules java.base --strip-debug --no-header-files --no-man-pages `
        --compress=zip-6 --output $Runtime
    if ($LASTEXITCODE -ne 0) { throw "jlink failed (exit $LASTEXITCODE)" }

    & jpackage --type app-image --name astralx --app-version $Version `
        --input $InputDir --main-jar astralx.jar --main-class astralx.Main `
        --runtime-image $Runtime --dest $JpackageOut `
        --java-options '-Djava.library.path=$APPDIR' `
        --java-options '-Dfile.encoding=UTF-8' `
        --java-options '-XX:InitialRAMPercentage=2.0' `
        --java-options '-XX:MaxRAMPercentage=85.0' `
        --java-options '-XX:ErrorFile=astralx-hotspot-crash-%p.log'
    if ($LASTEXITCODE -ne 0) { throw "jpackage failed (exit $LASTEXITCODE)" }

    $Image = Join-Path $JpackageOut "astralx"
    $ExampleDir = Join-Path $Image "example"
    New-Item -ItemType Directory -Force -Path $ExampleDir | Out-Null
    Copy-Item (Join-Path $PSScriptRoot "all_gt_bs_rooted_37.tre") `
        (Join-Path $ExampleDir "all_gt_37.tre")
    Copy-Item (Join-Path $PSScriptRoot "true_37.tre") `
        (Join-Path $ExampleDir "true_37.tre")

    $Readme = @"
ASTRAL-X $Version - self-contained windows-$PlatformArch build

Run in PowerShell or Command Prompt:
  .\astralx.exe --help
  .\astralx.exe --diagnose
  .\astralx.exe -i gene_trees.tre -o species_tree.tre

Ready-made 37-taxon example (run from this directory):
  .\astralx.exe -i example\all_gt_37.tre -o example\predicted_st_37.tre --search-mode local -vv

The example directory initially contains:
  all_gt_37.tre   input gene trees
  true_37.tre     reference/true species tree

After the command finishes, example\predicted_st_37.tre contains the inferred
species tree. The reference tree is provided for comparison and is not used by
ASTRAL-X during inference.

No Java installation is required. This artifact includes CUDA acceleration: $IncludeCuda.
CUDA still requires a compatible NVIDIA GPU and installed NVIDIA driver. If CUDA cannot
be used, ASTRAL-X automatically explains why and falls back to CPU. Use --gpu-strict only
when falling back would be undesirable.
"@
    Set-Content -Path (Join-Path $Image "README.txt") -Value $Readme -Encoding UTF8

    $BuildInfo = @(
        "version=$Version",
        "platform=windows-$PlatformArch",
        "capability=$Capability",
        "built_utc=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))",
        "java=$((& java -version 2>&1 | Select-Object -First 1).ToString())"
    )
    if ($IncludeCuda) {
        $BuildInfo += "nvcc=$((& nvcc --version | Select-Object -Last 1).ToString())"
        $BuildInfo += "cuda_arch=$CudaArch"
    }
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $Commit = (& git -C $PSScriptRoot rev-parse HEAD 2>$null)
        if (-not $Commit) { $Commit = "unknown" }
        $BuildInfo += "git_commit=$Commit"
    }
    Set-Content -Path (Join-Path $Image "BUILD-INFO.txt") -Value $BuildInfo -Encoding UTF8

    $Launcher = Join-Path $Image "astralx.exe"
    & $Launcher --version
    if ($LASTEXITCODE -ne 0) { throw "Packaged --version smoke test failed." }
    & $Launcher --cpu --diagnose | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Packaged --diagnose smoke test failed." }
    $SmokeTree = Join-Path $Work "smoke-species-tree.tre"
    & $Launcher --cpu --search-mode full -q `
        -i (Join-Path $ExampleDir "all_gt_37.tre") -o $SmokeTree
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $SmokeTree) -or
            (Get-Item $SmokeTree).Length -eq 0) {
        throw "Packaged end-to-end inference smoke test failed."
    }

    $FinalImage = Join-Path $OutputDir $Artifact
    if (Test-Path $FinalImage) { Remove-Item -Recurse -Force $FinalImage }
    Move-Item $Image $FinalImage

    if (-not $NoArchive) {
        $Archive = Join-Path $OutputDir "$Artifact.zip"
        if (Test-Path $Archive) { Remove-Item -Force $Archive }
        Compress-Archive -Path $FinalImage -DestinationPath $Archive -CompressionLevel Optimal
        $Hash = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant()
        Set-Content -Path "$Archive.sha256" -Value "$Hash  $([IO.Path]::GetFileName($Archive))" -Encoding ASCII
        Write-Host "  Archive      : $Archive"
        Write-Host "  SHA-256      : $Archive.sha256"
    }

    Write-Host "Portable application ready: $FinalImage\astralx.exe"
}
finally {
    if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
}
