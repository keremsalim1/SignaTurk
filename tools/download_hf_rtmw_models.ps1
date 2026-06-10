param(
    [string]$OutDir = "model\signaturk\models\rtmw_hf"
)

$ErrorActionPreference = "Stop"

$files = @(
    @{
        Name = "yolox_m.onnx"
        Url = "https://huggingface.co/memescreamer/yolox-onnx/resolve/main/yolox_m.onnx"
        MinBytes = 90000000
    },
    @{
        Name = "rtmw-dw-x-l_simcc-cocktail14.onnx"
        Url = "https://huggingface.co/Izymka/rtmw-dw-x-l_simcc-cocktail14/resolve/main/rtmw-dw-x-l_simcc-cocktail14.onnx"
        MinBytes = 200000000
    }
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($file in $files) {
    $target = Join-Path $OutDir $file.Name
    Write-Host "Downloading $($file.Name) ..."
    curl.exe -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 -o $target $file.Url

    $item = Get-Item $target
    if ($item.Length -lt $file.MinBytes) {
        throw "Downloaded file is too small: $target ($($item.Length) bytes). It may be an HTML error page."
    }

    Write-Host "OK $($file.Name): $([math]::Round($item.Length / 1MB, 1)) MB"
}

Write-Host ""
Write-Host "Done. Start backend with:"
Write-Host '$env:SIGNATURK_EXTRACTOR="hf_rtmw"'
Write-Host '.\.venv\Scripts\python.exe -m uvicorn backend:app --host 127.0.0.1 --port 8000'
