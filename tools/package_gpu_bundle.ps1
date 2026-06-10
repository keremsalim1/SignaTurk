param(
    [string]$OutputDir = "dist",
    [string]$PackageName = "tsl-nexus-gpu-bundle.zip"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outDirPath = Join-Path $root $OutputDir
$stage = Join-Path $outDirPath "_tsl_nexus_gpu_bundle"
$zipPath = Join-Path $outDirPath $PackageName

New-Item -ItemType Directory -Force -Path $outDirPath | Out-Null
Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$files = @(
    "backend.py",
    "database.py",
    "models.py",
    "landmark_smoother.py",
    "requirements.txt",
    "requirements-gpu.txt",
    "README.md",
    "PROJECT_STATUS.md"
)

$dirs = @(
    "frontend",
    "signaturk_runtime",
    "tools",
    "model\signaturk"
)

foreach ($file in $files) {
    $source = Join-Path $root $file
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $stage $file) -Force
    }
}

foreach ($dir in $dirs) {
    $source = Join-Path $root $dir
    if (Test-Path $source) {
        $target = Join-Path $stage $dir
        New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Recurse -Force
    }
}

$landmarks = Join-Path $root "model\landmarks.json"
if (Test-Path $landmarks) {
    $targetModelDir = Join-Path $stage "model"
    New-Item -ItemType Directory -Force -Path $targetModelDir | Out-Null
    Copy-Item -LiteralPath $landmarks -Destination (Join-Path $targetModelDir "landmarks.json") -Force
}

$cleanupPatterns = @(
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".cursor",
    ".vscode",
    "dist",
    "run_logs",
    "local_dev.db",
    "*.pyc"
)

foreach ($pattern in $cleanupPatterns) {
    Get-ChildItem -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like $pattern } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
}

$stageItems = Get-ChildItem -LiteralPath $stage -Force
Compress-Archive -Path $stageItems.FullName -DestinationPath $zipPath -Force
Remove-Item -LiteralPath $stage -Recurse -Force

$zip = Get-Item $zipPath
Write-Host "Created: $zipPath"
Write-Host "Size: $([math]::Round($zip.Length / 1MB, 1)) MB"
