param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python runtime not found: $Python. Create the venv and install requirements-gpu.txt first."
}

$env:SIGNATURK_EXTRACTOR = "hf_rtmw"
$env:SIGNATURK_REALTIME_STREAMS = "skeleton_v2,skeleton_v4_seed,skeleton_v6_seed,hand_stream"
$env:SIGNATURK_LIVE_MIN_FRAMES = "5"
$env:SIGNATURK_LIVE_WINDOW_S = "2.2"
$env:SIGNATURK_CAPTURE_INTERVAL_MS = "62"
$env:SIGNATURK_WS_DRAIN_LIMIT = "128"
$env:SIGNATURK_PREDICTION_COOLDOWN_S = "0.8"
if (-not $env:SIGNATURK_KPT_THR) { $env:SIGNATURK_KPT_THR = "0.05" }
if (-not $env:SIGNATURK_FILTER_MODEL_HANDS) { $env:SIGNATURK_FILTER_MODEL_HANDS = "false" }
if (-not $env:SIGNATURK_OVERLAY_KPT_THR) { $env:SIGNATURK_OVERLAY_KPT_THR = "0.20" }
if (-not $env:SIGNATURK_MIN_HAND_POINTS) { $env:SIGNATURK_MIN_HAND_POINTS = "6" }
if (-not $env:SIGNATURK_MIN_HAND_MEAN_CONF) { $env:SIGNATURK_MIN_HAND_MEAN_CONF = "0.20" }
if (-not $env:SIGNATURK_MAX_HAND_BBOX_SPAN) { $env:SIGNATURK_MAX_HAND_BBOX_SPAN = "0.34" }
if (-not $env:SIGNATURK_MAX_HAND_BBOX_AREA) { $env:SIGNATURK_MAX_HAND_BBOX_AREA = "0.06" }

Write-Host "Starting TSL Nexus GPU profile..."
Write-Host "URL: http://$HostAddress`:$Port/"
& $Python -m uvicorn backend:app --host $HostAddress --port $Port
