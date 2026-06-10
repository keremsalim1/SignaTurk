param(
    [string]$TargetDir = "model\signaturk\models\hf_vitpose"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$files = @(
    @{
        Url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
        Out = Join-Path $TargetDir "yolox_s.onnx"
    },
    @{
        Url = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-s-wholebody.onnx"
        Out = Join-Path $TargetDir "vitpose-s-wholebody.onnx"
    }
)

foreach ($file in $files) {
    if ((Test-Path $file.Out) -and ((Get-Item $file.Out).Length -gt 1MB)) {
        Write-Host "Already present: $($file.Out)"
        continue
    }

    Write-Host "Downloading $($file.Url)"
    curl.exe -L --retry 5 --retry-delay 5 --connect-timeout 30 `
        -o $file.Out `
        $file.Url

    if (-not (Test-Path $file.Out) -or ((Get-Item $file.Out).Length -lt 1MB)) {
        throw "Download failed or file is too small: $($file.Out)"
    }
}

Write-Host ""
Write-Host "Done. Start backend with:"
Write-Host '$env:SIGNATURK_EXTRACTOR="hf_vitpose"'
Write-Host '.\.venv\Scripts\python.exe -m uvicorn backend:app --host 127.0.0.1 --port 8000'
